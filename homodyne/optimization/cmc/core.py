"""CMC core module - main entry point.

This module provides the fit_mcmc_jax() function that serves as the
main entry point for CMC analysis, matching the CLI signature.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from homodyne.optimization.cmc.backends import select_backend
from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.data_prep import (
    prepare_mcmc_data,
    shard_data_random,
    shard_data_stratified,
)
from homodyne.optimization.cmc.diagnostics import (
    summarize_diagnostics,
)
from homodyne.optimization.cmc.model import xpcs_model_scaled
from homodyne.optimization.cmc.priors import get_param_names_in_order
from homodyne.optimization.cmc.results import CMCResult
from homodyne.optimization.cmc.sampler import run_nuts_sampling
from homodyne.utils.logging import get_logger, with_context

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


def _resolve_max_points_per_shard(
    analysis_mode: str,
    n_total: int,
    max_points_per_shard: int | str | None,
    max_shards: int = 2000,
) -> int:
    """Determine optimal shard size based on mode and data volume.

    NUTS MCMC is O(n) per iteration - evaluates ALL points in a shard.
    Laminar flow (7 params) needs ~10x smaller shards than static (3 params)
    due to complex gradient computation (trigonometric functions, cumulative integrals).

    Scaling guidelines (laminar_flow mode with 2 chains, 1000 iterations):
    - 5K points → ~2-3 min/shard
    - 10K points → ~5-8 min/shard (sweet spot)
    - 20K points → ~15-25 min/shard
    - 50K points → ~45-75 min/shard
    - 100K points → ~2+ hours/shard (too slow)

    Memory scalability for shard combination:
    - Each shard result: ~100KB (13 params × 2 chains × 1500 samples × 8 bytes)
    - Peak memory: ~6 × K MB where K = number of shards
    - Safe limits: K=1000 → ~6GB, K=2000 → ~12GB, K=5000 → ~30GB

    For production HPC environments:
    - Bebop (36 cores, 128GB): max ~2500 shards → 10K points for <25M datasets
    - Improv (128 cores, 256GB): max ~5000 shards → 8K points for <40M datasets
    - Personal (8-36 cores, 32GB): max ~500 shards → ~5M dataset limit

    Parameters
    ----------
    analysis_mode : str
        Analysis mode: "static" or "laminar_flow".
    n_total : int
        Total number of data points.
    max_points_per_shard : int | str | None
        User-specified shard size or "auto".
    max_shards : int
        Maximum number of shards to create (caps memory usage).
        Default 2000 requires ~12GB for combination phase.
    """
    if max_points_per_shard is not None and max_points_per_shard != "auto":
        return int(max_points_per_shard)

    # Auto-detection based on analysis mode and dataset size
    if analysis_mode == "laminar_flow":
        # Laminar flow needs smaller shards due to complex gradients (7+ params)
        if n_total >= 100_000_000:      # 100M+ points
            base = 5_000                 # ~20K shards, ~3 min each
        elif n_total >= 50_000_000:     # 50M+ points
            base = 6_000                 # ~8K shards, ~4 min each
        elif n_total >= 20_000_000:     # 20M+ points
            base = 8_000                 # ~2.5K shards, ~5 min each
        elif n_total >= 2_000_000:      # 2M+ points
            base = 10_000                # ~200-300 shards, ~5-8 min each
        else:
            base = 20_000                # Small datasets can use larger shards
    else:
        # Static mode (3 params) - simpler gradients, can handle larger shards
        if n_total >= 100_000_000:      # 100M+ points
            base = 50_000                # ~2K shards
        elif n_total >= 50_000_000:     # 50M+ points
            base = 80_000                # ~625 shards
        else:
            base = 100_000               # Default for static mode

    # Cap shard count to prevent memory exhaustion during combination
    # Each shard result ~100KB → max_shards=2000 needs ~12GB peak memory
    estimated_shards = n_total // base
    if estimated_shards > max_shards:
        # Increase shard size to respect max_shards limit
        adjusted = (n_total + max_shards - 1) // max_shards
        # Don't go too large for laminar_flow - cap at 50K for runtime
        if analysis_mode == "laminar_flow":
            if adjusted > 50_000:
                final_shards = n_total // 50_000
                # Warn user: need more memory for very large laminar_flow datasets
                logger.warning(
                    f"Dataset ({n_total:,} points) exceeds recommended limits for laminar_flow. "
                    f"Will create {final_shards:,} shards (>max_shards={max_shards}). "
                    f"Ensure sufficient memory (~{final_shards * 6 // 1000}GB) or reduce dataset size."
                )
            adjusted = min(adjusted, 50_000)
        return adjusted

    return base


def _cap_laminar_max_points(max_points_per_shard: int, logger) -> int:
    """Guard rails for laminar_flow to keep shards within reasonable runtime.

    Caps overly large user values that would routinely exceed per-shard timeouts.
    """

    # Allow larger shards for smoke runs; still guard against runaway values.
    cap = 3_000_000
    if max_points_per_shard > cap:
        logger.warning(
            f"max_points_per_shard={max_points_per_shard:,} is high for laminar_flow; "
            f"capping to {cap:,} to keep per-shard runtime tractable"
        )
        return cap
    return max_points_per_shard


def _compute_suggested_timeout(
    *,
    cost_per_shard: int,
    max_timeout: int,
    secs_per_unit: float = 2.0e-5,
    safety_factor: float = 5.0,
    min_timeout: int = 600,
) -> int:
    """Derive a timeout (seconds) from shard cost with clamping.

    cost_per_shard = num_chains * (num_warmup + num_samples) * max_points_per_shard
    """

    raw = safety_factor * secs_per_unit * cost_per_shard
    clamped = min(max_timeout, max(min_timeout, raw))
    return int(clamped)


def _fmt_time(secs: float) -> str:
    """Format time nicely for display."""
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs / 60:.1f}min"
    else:
        return f"{secs / 3600:.1f}h"


def _estimate_n_workers() -> int:
    """Estimate the number of workers that will be used by the multiprocessing backend.

    This mirrors the logic in MultiprocessingBackend.__init__ to provide
    accurate runtime estimates before the backend is instantiated.

    Returns
    -------
    int
        Estimated number of worker processes.
    """
    import os
    import multiprocessing as mp

    # Try to get physical core count (same logic as multiprocessing backend)
    try:
        logical_cores = mp.cpu_count()
    except NotImplementedError:
        logical_cores = 4  # Conservative default

    # Estimate physical cores (assume 2 threads per core for HT)
    physical_cores_estimate = max(1, logical_cores // 2)

    # Reserve 1 core for main process (same as backend)
    n_workers = max(1, physical_cores_estimate - 1)

    return n_workers


def _log_runtime_estimate(
    logger,
    n_shards: int,
    n_chains: int,
    n_warmup: int,
    n_samples: int,
    avg_points_per_shard: int,
    n_workers: int | None = None,
    analysis_mode: str = "static",
) -> float:
    """Log estimated CMC runtime for user awareness.

    Provides rough estimates based on empirical observations:
    - JIT compilation: ~30-60s per worker process
    - MCMC step: ~0.1-0.5s per iteration (varies with point count)

    Returns
    -------
    float
        Estimated total runtime in seconds.
    """
    # Estimate worker count if not provided
    if n_workers is None:
        n_workers = _estimate_n_workers()

    # Estimate per-shard time
    jit_overhead_per_shard = 45  # seconds, average JIT compilation
    iterations_per_shard = n_chains * (n_warmup + n_samples)

    # MCMC step time scales roughly with point count
    # Empirical: ~0.0001s per point per iteration for moderate complexity
    base_secs_per_iteration = 0.2 + (avg_points_per_shard / 100_000) * 0.3

    # Analysis mode factor - laminar_flow has more parameters and complexity
    mode_factor = 1.5 if analysis_mode == "laminar_flow" else 1.0
    secs_per_iteration = base_secs_per_iteration * mode_factor

    sampling_time_per_shard = iterations_per_shard * secs_per_iteration

    total_per_shard = jit_overhead_per_shard + sampling_time_per_shard

    # Parallel execution estimate
    batches = (n_shards + n_workers - 1) // n_workers
    total_parallel = batches * total_per_shard

    logger.info(
        f"Runtime estimate: {_fmt_time(total_parallel)} total "
        f"({n_shards} shards / {n_workers} workers, "
        f"~{_fmt_time(total_per_shard)}/shard with {iterations_per_shard:,} iterations)"
    )

    return total_parallel


def _log_runtime_comparison(
    logger,
    estimated_time: float,
    actual_time: float,
) -> None:
    """Log comparison of estimated vs actual runtime.

    Parameters
    ----------
    logger
        Logger instance.
    estimated_time : float
        Estimated runtime in seconds.
    actual_time : float
        Actual runtime in seconds.
    """
    if estimated_time <= 0:
        return

    accuracy = actual_time / estimated_time * 100
    diff = actual_time - estimated_time

    if accuracy < 50:
        status = "much faster than estimated"
    elif accuracy < 90:
        status = "faster than estimated"
    elif accuracy <= 110:
        status = "close to estimate"
    elif accuracy <= 150:
        status = "slower than estimated"
    else:
        status = "much slower than estimated"

    logger.info(
        f"Runtime: {_fmt_time(actual_time)} actual vs {_fmt_time(estimated_time)} estimated "
        f"({accuracy:.0f}% - {status})"
    )

    # Provide suggestions if significantly off
    if accuracy > 200:
        logger.info(
            "  → Consider reducing num_samples or num_chains for faster runs"
        )
    elif accuracy < 30:
        logger.info(
            "  → Actual runtime much faster than expected - estimate may be conservative"
        )


def _infer_time_step(t1: np.ndarray, t2: np.ndarray) -> float:
    """Infer time step from pooled t1/t2 arrays (seconds).

    Uses the median positive difference across all unique time points to avoid
    being skewed by repeated values from meshgrid flattening.
    """
    time_values = np.unique(np.concatenate([np.asarray(t1), np.asarray(t2)]))
    if time_values.size < 2:
        return 1.0

    diffs = np.diff(time_values)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return 1.0

    return float(np.median(positive_diffs))


def fit_mcmc_jax(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    analysis_mode: str,
    method: str = "mcmc",
    cmc_config: dict[str, Any] | None = None,
    initial_values: dict[str, float] | None = None,
    parameter_space: ParameterSpace = None,
    dt: float | None = None,
    output_dir: Path | str | None = None,
    progress_bar: bool = True,
    run_id: str | None = None,
    **kwargs,
) -> CMCResult:
    """Run CMC (Consensus Monte Carlo) analysis on XPCS data.

    This function signature matches the CLI call in cli/commands.py:1201.

    Parameters
    ----------
    data : np.ndarray
        Pooled C2 correlation data, shape (n_total,).
    t1 : np.ndarray
        Pooled time coordinates t1, shape (n_total,).
    t2 : np.ndarray
        Pooled time coordinates t2, shape (n_total,).
    phi : np.ndarray
        Pooled phi angles, shape (n_total,).
    q : float
        Wavevector magnitude.
    L : float
        Stator-rotor gap length (nm).
    analysis_mode : str
        Analysis mode: "static" or "laminar_flow".
    method : str
        Method identifier (always "mcmc" for CMC).
    cmc_config : dict[str, Any] | None
        CMC configuration from ConfigManager.get_cmc_config().
    initial_values : dict[str, float] | None
        Initial parameter values from ConfigManager.get_initial_parameters().
    parameter_space : ParameterSpace
        Parameter space with bounds and priors from ParameterSpace.from_config().
    dt : float | None
        Time step for physics model. If None, inferred from pooled time arrays.
    output_dir : Path | str | None
        Output directory for saving results.
    progress_bar : bool
        Whether to show progress bar during sampling.
    run_id : str | None
        Optional identifier used to correlate logs across shards/backends.
    **kwargs
        Additional keyword arguments (for compatibility).

    Returns
    -------
    CMCResult
        Complete result with posterior samples and diagnostics.

    Raises
    ------
    ValueError
        If data validation fails.
    RuntimeError
        If MCMC sampling fails.

    Examples
    --------
    >>> from homodyne.optimization.cmc import fit_mcmc_jax
    >>> result = fit_mcmc_jax(
    ...     data=c2_pooled,
    ...     t1=t1_pooled,
    ...     t2=t2_pooled,
    ...     phi=phi_pooled,
    ...     q=0.01,
    ...     L=2000000.0,
    ...     analysis_mode="laminar_flow",
    ...     method="mcmc",
    ...     cmc_config=config.get_cmc_config(),
    ...     initial_values=config.get_initial_parameters(),
    ...     parameter_space=parameter_space,
    ... )
    >>> print(result.convergence_status)
    converged
    """
    start_time = time.perf_counter()
    run_identifier = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_logger = with_context(logger, run=run_identifier, analysis="cmc")

    # Normalize analysis mode
    if "static" in analysis_mode.lower():
        analysis_mode = "static"

    run_logger.info(
        f"Starting CMC analysis: {len(data):,} points, mode={analysis_mode}, q={q:.4f}"
    )

    # =========================================================================
    # 1. Parse configuration
    # =========================================================================
    if cmc_config is None:
        cmc_config = {}
    config = CMCConfig.from_dict(cmc_config)
    config.run_id = getattr(config, "run_id", None) or run_identifier

    # Log configuration
    run_logger.info(
        f"CMC config: {config.num_chains} chains, "
        f"{config.num_warmup} warmup, {config.num_samples} samples"
    )

    # =========================================================================
    # 2. Prepare and validate data
    # =========================================================================
    prepared = prepare_mcmc_data(data, t1, t2, phi)

    # Log initial values if provided
    if initial_values:
        run_logger.info(
            f"Initial values: {', '.join(f'{k}={v:.4g}' for k, v in list(initial_values.items())[:5])}..."
        )
    else:
        run_logger.info("No initial values provided, using midpoint defaults")

    # =========================================================================
    # 3. Determine if CMC sharding is needed
    # =========================================================================
    def _int_like(val) -> bool:
        return isinstance(val, int) or (isinstance(val, str) and val.isdigit())

    forced_shards = _int_like(config.num_shards) or _int_like(
        config.max_points_per_shard
    )
    use_cmc = config.should_enable_cmc(prepared.n_total) or forced_shards

    # Resolve max_points_per_shard - critical for NUTS tractability
    # Scale inversely with parameter count: more params = fewer points per shard
    max_points_setting = config.max_points_per_shard
    max_per_shard = _resolve_max_points_per_shard(
        analysis_mode, prepared.n_total, max_points_setting
    )
    if analysis_mode == "laminar_flow":
        max_per_shard = _cap_laminar_max_points(max_per_shard, run_logger)
    if max_points_setting is None or max_points_setting == "auto":
        run_logger.info(
            f"Auto-selected max_points_per_shard={max_per_shard} for {analysis_mode} mode "
            f"(n_total={prepared.n_total:,})"
        )

    # Derive a suggested per-shard timeout from cost
    cost_per_shard = (
        config.num_chains * (config.num_warmup + config.num_samples) * max_per_shard
    )
    suggested_timeout = _compute_suggested_timeout(
        cost_per_shard=cost_per_shard,
        max_timeout=config.per_shard_timeout,
    )
    run_logger.info(
        f"Suggested per-shard timeout: {suggested_timeout}s (cost={cost_per_shard:,}, "
        f"chains={config.num_chains}, warmup+samples={config.num_warmup + config.num_samples}, "
        f"max_points_per_shard={max_per_shard:,}, clamp=[600,{config.per_shard_timeout}])"
    )

    requested_shards = int(config.num_shards) if _int_like(config.num_shards) else None

    if use_cmc and prepared.n_phi > 1:
        # Shard by phi angle (stratified)
        num_shards = (
            requested_shards
            if requested_shards is not None
            else config.get_num_shards(prepared.n_total, prepared.n_phi)
        )

        shards = shard_data_stratified(
            prepared, num_shards, max_points_per_shard=max_per_shard
        )
        total_shard_points = sum(s.n_total for s in shards)
        run_logger.info(
            f"Using CMC with {len(shards)} shards (stratified by phi), "
            f"{total_shard_points:,} total points"
        )
        estimated_runtime = _log_runtime_estimate(
            run_logger,
            n_shards=len(shards),
            n_chains=config.num_chains,
            n_warmup=config.num_warmup,
            n_samples=config.num_samples,
            avg_points_per_shard=total_shard_points // len(shards),
            analysis_mode=analysis_mode,
        )
    elif use_cmc and prepared.n_phi == 1:
        # Single phi angle but large dataset - use random sharding
        # shard_data_random handles num_shards calculation and capping internally
        shards = shard_data_random(
            prepared,
            num_shards=requested_shards,  # Honor explicit shards when provided
            max_points_per_shard=max_per_shard,
            max_shards=100,  # Same cap as stratified
        )
        total_shard_points = sum(s.n_total for s in shards)
        run_logger.info(
            f"Using CMC with {len(shards)} shards (random split, single phi), "
            f"{total_shard_points:,} total points"
        )
        estimated_runtime = _log_runtime_estimate(
            run_logger,
            n_shards=len(shards),
            n_chains=config.num_chains,
            n_warmup=config.num_warmup,
            n_samples=config.num_samples,
            avg_points_per_shard=total_shard_points // len(shards),
            analysis_mode=analysis_mode,
        )
    else:
        shards = None
        estimated_runtime = 0.0  # No estimate for single-shard
        run_logger.info("Using single-shard MCMC (no CMC sharding)")

    # =========================================================================
    # 4. Build model function
    # =========================================================================
    # CRITICAL FIX (Dec 2025): Construct time_grid with PROPER dt spacing
    # Previously used np.unique(t1, t2) which gave incorrect grid density
    # when data is subsampled or pooled from shards with different time points.
    #
    # The physics integration (trapezoidal cumsum) depends critically on grid density:
    # - With dt=0.1s and t_max=100s, need 1001 points for correct physics
    # - Using np.unique gave variable n_points (e.g., 201 with subsampled data)
    # - This caused up to 26% error in C2 values vs NLSQ (see scripts/compare_nlsq_cmc_c2.py)
    #
    # Fix: Construct time_grid from config dt, NOT from data unique values

    # First determine dt to use (config dt takes precedence)
    inferred_dt = _infer_time_step(prepared.t1, prepared.t2)
    dt_used = dt if dt is not None else inferred_dt

    if not np.isfinite(dt_used) or dt_used <= 0:
        dt_used = inferred_dt if np.isfinite(inferred_dt) and inferred_dt > 0 else 0.1
        run_logger.warning(
            f"Invalid dt provided; using inferred fallback dt={dt_used:.6g} seconds"
        )
    else:
        rel_diff = (
            abs(dt_used - inferred_dt) / max(inferred_dt, 1e-12)
            if np.isfinite(inferred_dt) and inferred_dt > 0
            else 0.0
        )
        if dt is None:
            run_logger.info(f"Inferred dt from pooled times: dt={dt_used:.6g} seconds")
        elif rel_diff > 1e-3:
            run_logger.warning(
                f"dt mismatch: provided dt={dt_used:.6g}s vs inferred {inferred_dt:.6g}s; "
                "using provided dt for physics; results may not match NLSQ"
            )

    # CRITICAL: Construct time_grid with CORRECT dt spacing to match NLSQ physics
    # The grid must have the same density as NLSQ (e.g., dt=0.1s gives 1001 points for [0, 100])
    t1_np = np.asarray(prepared.t1)
    t2_np = np.asarray(prepared.t2)
    t_min = 0.0  # Always start from t=0 for consistent integration
    t_max = float(max(t1_np.max(), t2_np.max()))
    n_time_points = int(round(t_max / dt_used)) + 1
    time_grid_np = np.linspace(t_min, t_max, n_time_points)
    time_grid = jnp.array(time_grid_np)

    # Log time_grid construction details
    run_logger.info(
        f"[CMC] time_grid constructed with dt={dt_used:.6g}s: "
        f"n_points={n_time_points}, range=[{t_min:.6g}, {t_max:.6g}]"
    )
    run_logger.info(
        f"[CMC] Data time ranges: t1=[{t1_np.min():.6g}, {t1_np.max():.6g}], "
        f"t2=[{t2_np.min():.6g}, {t2_np.max():.6g}]"
    )

    # Verify grid spacing matches config dt
    actual_grid_dt = (time_grid_np[1] - time_grid_np[0]) if len(time_grid_np) > 1 else dt_used
    if abs(actual_grid_dt - dt_used) > 1e-6:
        run_logger.warning(
            f"[CMC] Grid spacing {actual_grid_dt:.6g}s differs from config dt={dt_used:.6g}s"
        )

    model_kwargs = {
        "data": jnp.array(prepared.data),
        "t1": jnp.array(prepared.t1),
        "t2": jnp.array(prepared.t2),
        "phi_unique": jnp.array(prepared.phi_unique),
        "phi_indices": jnp.array(prepared.phi_indices),
        "q": q,
        "L": L,
        "dt": dt_used,
        "time_grid": time_grid,
        "analysis_mode": analysis_mode,
        "parameter_space": parameter_space,
        "n_phi": prepared.n_phi,
        "noise_scale": prepared.noise_scale,
    }

    # DEBUG: Log model_kwargs for diagnosis
    run_logger.info(
        f"[CMC DEBUG] model_kwargs: q={q:.6g}, L={L:.6g}, dt={dt_used:.6g}, "
        f"n_phi={prepared.n_phi}, noise_scale={prepared.noise_scale:.6g}"
    )
    run_logger.info(f"[CMC DEBUG] phi_unique: {prepared.phi_unique}")

    # DEBUG: Compute and log D values at sample times to verify physics
    if initial_values:
        D0_init = initial_values.get("D0", 1e10)
        alpha_init = initial_values.get("alpha", -0.5)
        D_offset_init = initial_values.get("D_offset", 1e9)
        # Sample D at a few time points
        t_samples = np.array([0.0, 1.0, 10.0, 50.0])
        t_safe = t_samples + 1e-10
        D_samples = D0_init * (t_safe**alpha_init) + D_offset_init
        run_logger.info(
            f"[CMC DEBUG] D(t) at sample times with initial params:\n"
            f"  D0={D0_init:.4g}, alpha={alpha_init:.4g}, D_offset={D_offset_init:.4g}\n"
            f"  t=[0, 1, 10, 50] → D={D_samples}"
        )
        # Compute expected prefactor
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_used
        run_logger.info(
            f"[CMC DEBUG] Physics prefactor: 0.5*q²*dt = 0.5*{q}²*{dt_used} = {wavevector_q_squared_half_dt:.6g}"
        )

    # =========================================================================
    # 5. Select backend and run sampling
    # =========================================================================
    if shards is not None and len(shards) > 1:
        # Use parallel backend for CMC
        backend = select_backend(config)
        run_logger.info(f"Using backend: {backend.get_name()}")

        # Enforce timeout only where supported (multiprocessing). Others log advisory.
        if backend.get_name().startswith("multiprocessing"):
            effective_timeout = min(config.per_shard_timeout, suggested_timeout)
            if effective_timeout != config.per_shard_timeout:
                run_logger.info(
                    f"Applying tighter per_shard_timeout={effective_timeout}s based on shard cost"
                )
            config.per_shard_timeout = effective_timeout
        else:
            run_logger.warning(
                f"Backend '{backend.get_name()}' does not enforce per_shard_timeout; "
                f"suggested={suggested_timeout}s (cap={config.per_shard_timeout}s)"
            )

        mcmc_samples = backend.run(
            model=xpcs_model_scaled,
            model_kwargs=model_kwargs,
            config=config,
            shards=shards,
            initial_values=initial_values,
            parameter_space=parameter_space,
            analysis_mode=analysis_mode,
            progress_bar=progress_bar,
        )
        stats_warmup = 0.0  # Not tracked for parallel
        stats_total = time.perf_counter() - start_time
    else:
        # Single-shard: run directly
        mcmc_samples, stats = run_nuts_sampling(
            model=xpcs_model_scaled,
            model_kwargs=model_kwargs,
            config=config,
            initial_values=initial_values,
            parameter_space=parameter_space,
            n_phi=prepared.n_phi,
            analysis_mode=analysis_mode,
            progress_bar=progress_bar,
        )
        stats_warmup = stats.warmup_time
        stats_total = stats.total_time

    # =========================================================================
    # 6. Create result
    # =========================================================================
    from homodyne.optimization.cmc.sampler import SamplingStats

    final_stats = SamplingStats(
        warmup_time=stats_warmup,
        sampling_time=stats_total - stats_warmup,
        total_time=stats_total,
        num_divergent=mcmc_samples.extra_fields.get("diverging", np.array([0])).sum(),
    )

    result = CMCResult.from_mcmc_samples(
        mcmc_samples=mcmc_samples,
        stats=final_stats,
        analysis_mode=analysis_mode,
        n_warmup=config.num_warmup,
    )

    # =========================================================================
    # 7. Log summary
    # =========================================================================
    summary = summarize_diagnostics(
        r_hat=result.r_hat,
        ess_bulk=result.ess_bulk,
        divergences=result.divergences,
        n_samples=result.n_samples,
        n_chains=result.n_chains,
        num_shards=result.num_shards,
    )
    run_logger.info(f"CMC complete: {result.convergence_status}")
    run_logger.info(summary)

    # Log parameter estimates
    stats_dict = result.get_posterior_stats()
    for name in get_param_names_in_order(prepared.n_phi, analysis_mode)[:5]:
        if name in stats_dict:
            s = stats_dict[name]
            run_logger.info(
                f"  {name}: {s['mean']:.4g} +/- {s['std']:.4g} "
                f"(R-hat={s['r_hat']:.3f}, ESS={s['ess_bulk']:.0f})"
            )

    total_time = time.perf_counter() - start_time
    run_logger.info(f"Total execution time: {total_time:.1f}s")

    # Log runtime comparison if we had an estimate
    if estimated_runtime > 0:
        _log_runtime_comparison(run_logger, estimated_runtime, total_time)

    return result


def run_cmc_analysis(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    analysis_mode: str,
    config: CMCConfig,
    parameter_space: ParameterSpace,
    initial_values: dict[str, float] | None = None,
    dt: float | None = None,
) -> CMCResult:
    """Simplified interface for CMC analysis.

    This is a convenience wrapper around fit_mcmc_jax() that takes
    a CMCConfig object directly instead of a dict.

    Parameters
    ----------
    data, t1, t2, phi : np.ndarray
        Data arrays.
    q, L : float
        Physics parameters.
    analysis_mode : str
        Analysis mode.
    config : CMCConfig
        CMC configuration object.
    parameter_space : ParameterSpace
        Parameter space.
    initial_values : dict[str, float] | None
        Initial values.
    dt : float | None
        Time step (None infers from pooled time arrays).

    Returns
    -------
    CMCResult
        Analysis result.
    """
    return fit_mcmc_jax(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=q,
        L=L,
        analysis_mode=analysis_mode,
        method="mcmc",
        cmc_config=config.to_dict(),
        initial_values=initial_values,
        parameter_space=parameter_space,
        dt=dt,
    )
