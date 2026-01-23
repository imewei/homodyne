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

from homodyne.core.scaling_utils import estimate_per_angle_scaling
from homodyne.optimization.cmc.backends import select_backend
from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.data_prep import (
    prepare_mcmc_data,
    shard_data_angle_balanced,
    shard_data_random,
    shard_data_stratified,
)
from homodyne.optimization.cmc.diagnostics import (
    compute_precision_analysis,
    log_precision_analysis,
    summarize_diagnostics,
)
from homodyne.optimization.cmc.model import get_xpcs_model
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
    n_phi: int = 1,
) -> int:
    """Determine optimal shard size based on mode, data volume, and angle count.

    NUTS MCMC is O(n) per iteration - evaluates ALL points in a shard.
    Laminar flow (7 params) needs ~10x smaller shards than static (3 params)
    due to complex gradient computation (trigonometric functions, cumulative integrals).

    CRITICAL (Jan 2026): Angle-aware scaling for multi-angle datasets.
    When n_phi is small (e.g., 3 angles), each shard contains points from all angles,
    making gradient computation ~n_phi times more expensive than single-angle shards.
    We scale the base shard size inversely with angle count to compensate.

    Scaling guidelines (laminar_flow mode with 2 chains, 1000 iterations):
    - 5K points → ~2-3 min/shard (single angle)
    - 10K points → ~5-8 min/shard (single angle)
    - 10K points with 3 angles → ~15-25 min/shard (angle scaling effect)
    - 3K points with 3 angles → ~5-8 min/shard (after angle-aware fix)

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
    n_phi : int
        Number of phi angles in the dataset. Used for angle-aware scaling.
        Default 1 (single angle - no scaling applied).
    """
    if max_points_per_shard is not None and max_points_per_shard != "auto":
        return int(max_points_per_shard)

    # =========================================================================
    # Angle-aware scaling factor (Jan 2026 fix for few-angle timeout issue)
    # =========================================================================
    # Multi-angle datasets with random sharding have ALL angles in each shard.
    # This makes gradient computation ~n_phi times more expensive.
    # Scale shard size inversely to compensate.
    if n_phi <= 3:
        angle_factor = 0.3  # 30% of base size for 1-3 angles
    elif n_phi <= 5:
        angle_factor = 0.5  # 50% for 4-5 angles
    elif n_phi <= 10:
        angle_factor = 0.7  # 70% for 6-10 angles
    else:
        angle_factor = 1.0  # Full size for many angles (stratified sharding preferred)

    # Auto-detection based on analysis mode and dataset size
    if analysis_mode == "laminar_flow":
        # Laminar flow needs smaller shards due to complex gradients (7+ params)
        if n_total >= 100_000_000:  # 100M+ points
            base = 5_000  # ~20K shards, ~3 min each
        elif n_total >= 50_000_000:  # 50M+ points
            base = 6_000  # ~8K shards, ~4 min each
        elif n_total >= 20_000_000:  # 20M+ points
            base = 8_000  # ~2.5K shards, ~5 min each
        elif n_total >= 2_000_000:  # 2M+ points
            base = 10_000  # ~200-300 shards, ~5-8 min each
        else:
            base = 20_000  # Small datasets can use larger shards
    else:
        # Static mode (3 params) - simpler gradients, can handle larger shards
        if n_total >= 100_000_000:  # 100M+ points
            base = 50_000  # ~2K shards
        elif n_total >= 50_000_000:  # 50M+ points
            base = 80_000  # ~625 shards
        else:
            base = 100_000  # Default for static mode

    # Apply angle-aware scaling
    scaled_base = int(base * angle_factor)
    # Enforce minimum shard size to avoid excessive sharding
    scaled_base = max(scaled_base, 1000)

    # Log angle-aware scaling if factor < 1.0
    if angle_factor < 1.0:
        logger.info(
            f"Angle-aware shard sizing: n_phi={n_phi} → factor={angle_factor:.1f}, "
            f"base={base:,} → scaled={scaled_base:,}"
        )

    # Cap shard count to prevent memory exhaustion during combination
    # Each shard result ~100KB → max_shards=2000 needs ~12GB peak memory
    estimated_shards = n_total // scaled_base
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

    return scaled_base


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
    secs_per_unit: float = 5.0e-5,
    safety_factor: float = 5.0,
    min_timeout: int = 600,
) -> int:
    """Derive a timeout (seconds) from shard cost with clamping.

    cost_per_shard = num_chains * (num_warmup + num_samples) * max_points_per_shard

    Note: secs_per_unit=5.0e-5 with safety_factor=5.0 provides ~2.5x headroom
    above observed real-world runtimes to handle variance across different machines.
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
        logger.info("  → Consider reducing num_samples or num_chains for faster runs")
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
    nlsq_result: dict | None = None,
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
    nlsq_result : dict | None
        Optional NLSQ result dictionary for warm-start priors. When provided,
        builds informative priors centered on NLSQ estimates, improving
        convergence speed and reducing divergences. Should contain parameter
        values and optionally uncertainties (see extract_nlsq_values_for_cmc).
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

    # =========================================================================
    # 2b. Determine per-angle mode and select appropriate model (v2.18.0+)
    # =========================================================================
    effective_per_angle_mode = config.get_effective_per_angle_mode(prepared.n_phi)
    xpcs_model = get_xpcs_model(effective_per_angle_mode)
    run_logger.info(
        f"CMC per-angle mode: {config.per_angle_mode} → {effective_per_angle_mode} "
        f"(n_phi={prepared.n_phi}, threshold={config.constant_scaling_threshold})"
    )

    # =========================================================================
    # 2c. Estimate fixed per-angle scaling for constant mode (v2.18.0+)
    # =========================================================================
    # Mode semantics:
    # - "auto": xpcs_model_averaged SAMPLES single averaged contrast/offset (10 params)
    #           No fixed arrays needed - the model samples them
    # - "constant": xpcs_model_constant uses FIXED per-angle arrays (8 params)
    #           Requires fixed_contrast/fixed_offset arrays from quantile estimation
    # - "individual": xpcs_model_scaled SAMPLES per-angle contrast/offset
    #           No fixed arrays needed - the model samples them
    fixed_contrast: jnp.ndarray | None = None
    fixed_offset: jnp.ndarray | None = None

    if effective_per_angle_mode in ("constant", "constant_averaged"):
        # CONSTANT/CONSTANT_AVERAGED mode: Use FIXED values from quantile estimation
        # Get contrast/offset bounds from parameter_space
        contrast_bounds = parameter_space.get_bounds("contrast")
        offset_bounds = parameter_space.get_bounds("offset")

        # Estimate per-angle contrast/offset from quantile analysis
        run_logger.info(
            f"{effective_per_angle_mode.upper()} mode: Estimating FIXED scaling from data quantiles..."
        )
        scaling_estimates = estimate_per_angle_scaling(
            c2_data=prepared.data,
            t1=prepared.t1,
            t2=prepared.t2,
            phi_indices=prepared.phi_indices,
            n_phi=prepared.n_phi,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
            log=run_logger,
        )

        # Build per-angle arrays from estimates
        contrast_per_angle = np.array(
            [scaling_estimates[f"contrast_{i}"] for i in range(prepared.n_phi)]
        )
        offset_per_angle = np.array(
            [scaling_estimates[f"offset_{i}"] for i in range(prepared.n_phi)]
        )

        fixed_contrast = jnp.array(contrast_per_angle)
        fixed_offset = jnp.array(offset_per_angle)

        if effective_per_angle_mode == "constant_averaged":
            # CONSTANT_AVERAGED mode: Model will internally average these
            run_logger.info(
                f"CONSTANT_AVERAGED mode: Using FIXED AVERAGED scaling (NLSQ parity):\n"
                f"  contrast: per-angle range=[{contrast_per_angle.min():.4f}, {contrast_per_angle.max():.4f}], "
                f"avg={contrast_per_angle.mean():.4f} (will be used)\n"
                f"  offset: per-angle range=[{offset_per_angle.min():.4f}, {offset_per_angle.max():.4f}], "
                f"avg={offset_per_angle.mean():.4f} (will be used)\n"
                f"  Parameters: 7 physical + 1 sigma = 8 total (scaling fixed, averaged)"
            )
        else:
            # CONSTANT mode: Different value per angle
            run_logger.info(
                f"CONSTANT mode: Using FIXED per-angle scaling (NOT sampled):\n"
                f"  contrast: mean={contrast_per_angle.mean():.4f}, "
                f"range=[{contrast_per_angle.min():.4f}, {contrast_per_angle.max():.4f}]\n"
                f"  offset: mean={offset_per_angle.mean():.4f}, "
                f"range=[{offset_per_angle.min():.4f}, {offset_per_angle.max():.4f}]\n"
                f"  Parameters: 7 physical + 1 sigma = 8 total (scaling fixed)"
            )
    elif effective_per_angle_mode == "auto":
        # AUTO mode: xpcs_model_averaged will SAMPLE single averaged contrast/offset
        # No fixed arrays needed - log the expected behavior
        run_logger.info(
            "AUTO mode: Will SAMPLE averaged contrast/offset (NLSQ parity):\n"
            "  Parameters: 2 averaged scaling + 7 physical + 1 sigma = 10 total"
        )
    else:
        # INDIVIDUAL mode: xpcs_model_scaled will SAMPLE per-angle contrast/offset
        run_logger.info(
            f"INDIVIDUAL mode: Will SAMPLE per-angle contrast/offset:\n"
            f"  Parameters: {prepared.n_phi * 2} per-angle + 7 physical + 1 sigma = "
            f"{prepared.n_phi * 2 + 8} total"
        )

    # Log initial values if provided
    if initial_values:
        run_logger.info(
            f"Initial values: {', '.join(f'{k}={v:.4g}' for k, v in list(initial_values.items())[:5])}..."
        )
    else:
        run_logger.info("No initial values provided, using midpoint defaults")

    # =========================================================================
    # 2d. NLSQ Warm-Start (Jan 2026): Use NLSQ results for better init values
    # =========================================================================
    if nlsq_result is not None:
        from homodyne.optimization.cmc.priors import extract_nlsq_values_for_cmc

        nlsq_values, nlsq_uncertainties = extract_nlsq_values_for_cmc(nlsq_result)

        # Override initial_values with NLSQ estimates (if not already provided)
        if initial_values is None:
            initial_values = {}

        # Merge NLSQ values into initial_values (NLSQ takes precedence for physical params)
        physical_params = ["D0", "alpha", "D_offset"]
        if analysis_mode == "laminar_flow":
            physical_params.extend(
                ["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
            )

        nlsq_used = []
        for param in physical_params:
            if param in nlsq_values:
                initial_values[param] = nlsq_values[param]
                nlsq_used.append(param)

        run_logger.info(
            f"NLSQ warm-start: Using NLSQ estimates for {len(nlsq_used)} params: "
            f"{', '.join(f'{p}={nlsq_values[p]:.4g}' for p in nlsq_used[:5])}"
            + ("..." if len(nlsq_used) > 5 else "")
        )

        # Log NLSQ uncertainties if available (useful for posterior comparison)
        if nlsq_uncertainties:
            unc_str = ", ".join(
                f"{p}±{nlsq_uncertainties[p]:.4g}"
                for p in nlsq_used[:5]
                if p in nlsq_uncertainties
            )
            if unc_str:
                run_logger.info(f"NLSQ uncertainties: {unc_str}")

    # =========================================================================
    # 3. Determine if CMC sharding is needed
    # =========================================================================
    def _int_like(val) -> bool:
        return isinstance(val, int) or (isinstance(val, str) and val.isdigit())

    forced_shards = _int_like(config.num_shards) or _int_like(
        config.max_points_per_shard
    )
    use_cmc = config.should_enable_cmc(prepared.n_total, analysis_mode) or forced_shards

    # Resolve max_points_per_shard - critical for NUTS tractability
    # Scale inversely with parameter count: more params = fewer points per shard
    max_points_setting = config.max_points_per_shard
    max_per_shard = _resolve_max_points_per_shard(
        analysis_mode, prepared.n_total, max_points_setting, n_phi=prepared.n_phi
    )
    if analysis_mode == "laminar_flow":
        max_per_shard = _cap_laminar_max_points(max_per_shard, run_logger)
    if max_points_setting is None or max_points_setting == "auto":
        run_logger.info(
            f"Auto-selected max_points_per_shard={max_per_shard} for {analysis_mode} mode "
            f"(n_total={prepared.n_total:,}, n_phi={prepared.n_phi})"
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

    sharding_mode = config.sharding_strategy

    # CRITICAL FIX (Jan 2026): Force random sharding for multi-angle datasets with global parameters.
    # Stratified sharding (by angle) creates disjoint posteriors that cannot be combined
    # by Consensus MC for global parameters (like D0, alpha, phi0).
    if use_cmc and prepared.n_phi > 1 and sharding_mode == "stratified":
        run_logger.warning(
            "Overriding sharding_strategy='stratified' -> 'random' for multi-angle data. "
            "Stratified sharding violates Consensus MC assumptions for global parameters."
        )
        sharding_mode = "random"

    if sharding_mode == "stratified":
        # Shard by phi angle (stratified) - Only valid for disjoint models (no global params)
        # or single-angle data (where n_phi=1, handled above)
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
    elif use_cmc:
        # Sharding strategy selection (Jan 2026 enhancement):
        # - Multi-angle (n_phi > 1): Use angle-balanced sharding for consistent posteriors
        # - Single-angle: Use random sharding (i.i.d. statistically correct for Consensus MC)
        if prepared.n_phi > 1:
            # Angle-balanced sharding ensures each shard has proportional angle coverage
            # This prevents heterogeneous posteriors (e.g., D_offset CV=1.58)
            shards = shard_data_angle_balanced(
                prepared,
                num_shards=requested_shards,  # Honor explicit shards when provided
                max_points_per_shard=max_per_shard,
                max_shards=500,  # Higher cap for multi-angle (smaller shards)
                min_angle_coverage=0.8,  # Require 80% angle coverage per shard
            )
            sharding_desc = "angle-balanced"
        else:
            # Single-angle: random sharding is fine
            shards = shard_data_random(
                prepared,
                num_shards=requested_shards,  # Honor explicit shards when provided
                max_points_per_shard=max_per_shard,
                max_shards=100,  # Same cap as stratified
            )
            sharding_desc = "random"
        total_shard_points = sum(s.n_total for s in shards)
        run_logger.info(
            f"Using CMC with {len(shards)} shards ({sharding_desc}), "
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
        # Check for mismatch between config dt and data dt
        rel_diff = (
            abs(dt_used - inferred_dt) / max(inferred_dt, 1e-12)
            if np.isfinite(inferred_dt) and inferred_dt > 0
            else 0.0
        )
        if dt is None:
            run_logger.info(f"Inferred dt from pooled times: dt={dt_used:.6g} seconds")
        elif rel_diff > 1e-2:  # >1% mismatch is significant
            # CRITICAL FIX (Jan 2026): Prioritize DATA TRUTH over config for dt
            # If config says dt=0.1s but data says dt=1e-5s, using dt=0.1s constructs
            # a coarse grid that collapses all data to index 0 (g1=1.0, no decay).
            # We MUST use the inferred_dt to match the actual data timestamps.
            run_logger.warning(
                f"[CMC] CRITICAL dt mismatch detected!\n"
                f"  Config dt:   {dt_used:.6g}s\n"
                f"  Inferred dt: {inferred_dt:.6g}s\n"
                f"  Mismatch:    {rel_diff:.1%} (>1%)\n"
                f"Action: OVERRIDING config dt with Inferred dt to prevent physics collapse.\n"
                f"Please check your configuration or data timestamps."
            )
            dt_used = inferred_dt
        elif rel_diff > 1e-4:
            run_logger.info(
                f"[CMC] Minor dt mismatch ({rel_diff:.2%}): {dt_used:.6g}s vs {inferred_dt:.6g}s. Using config dt."
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
    actual_grid_dt = (
        (time_grid_np[1] - time_grid_np[0]) if len(time_grid_np) > 1 else dt_used
    )
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

    # Add fixed scaling arrays for constant mode (v2.18.0+)
    if effective_per_angle_mode == "constant" and fixed_contrast is not None:
        model_kwargs["fixed_contrast"] = fixed_contrast
        model_kwargs["fixed_offset"] = fixed_offset

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

        # T047: Log shard progress start
        run_logger.info(
            f"Starting CMC sampling: {len(shards)} shards, "
            f"{config.num_chains} chains, {config.num_warmup}+{config.num_samples} samples"
        )
        mcmc_samples = backend.run(
            model=xpcs_model,
            model_kwargs=model_kwargs,
            config=config,
            shards=shards,
            initial_values=initial_values,
            parameter_space=parameter_space,
            analysis_mode=analysis_mode,
            progress_bar=progress_bar,
        )
        run_logger.info(f"CMC sampling completed: all {len(shards)} shards finished")
        stats_warmup = 0.0  # Not tracked for parallel
        stats_total = time.perf_counter() - start_time
    else:
        # Single-shard: run directly
        mcmc_samples, stats = run_nuts_sampling(
            model=xpcs_model,
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

    # =========================================================================
    # 8. Precision analysis (Jan 2026): Compare CMC posteriors to NLSQ
    # =========================================================================
    if nlsq_result is not None:
        from homodyne.optimization.cmc.priors import extract_nlsq_values_for_cmc

        nlsq_values, nlsq_uncertainties = extract_nlsq_values_for_cmc(nlsq_result)

        precision_analysis = compute_precision_analysis(
            cmc_result=stats_dict,
            nlsq_result=nlsq_values,
            nlsq_uncertainties=nlsq_uncertainties,
        )

        # Log comprehensive precision report
        log_precision_analysis(precision_analysis, log_fn=run_logger.info)

        # Warn if CMC significantly disagrees with NLSQ
        high_z_params = [
            (p, m.get("z_score", 0))
            for p, m in precision_analysis.items()
            if m.get("z_score", 0) > 3
        ]
        if high_z_params:
            run_logger.warning(
                "CMC-NLSQ disagreement (z > 3): "
                + ", ".join(f"{p} (z={z:.1f})" for p, z in high_z_params)
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
