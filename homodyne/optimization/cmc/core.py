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
    shard_data_stratified,
)
from homodyne.optimization.cmc.diagnostics import (
    summarize_diagnostics,
)
from homodyne.optimization.cmc.model import xpcs_model
from homodyne.optimization.cmc.priors import get_param_names_in_order
from homodyne.optimization.cmc.results import CMCResult
from homodyne.optimization.cmc.sampler import run_nuts_sampling
from homodyne.utils.logging import get_logger, with_context

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


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
    dt: float = 0.1,
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
    dt : float
        Time step for physics model.
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
    use_cmc = config.should_enable_cmc(prepared.n_total)

    if use_cmc and prepared.n_phi > 1:
        # Shard by phi angle
        num_shards = config.get_num_shards(prepared.n_total, prepared.n_phi)
        shards = shard_data_stratified(prepared, num_shards)
        run_logger.info(f"Using CMC with {len(shards)} shards (stratified by phi)")
    else:
        shards = None
        run_logger.info("Using single-shard MCMC (no CMC sharding)")

    # =========================================================================
    # 4. Build model function
    # =========================================================================
    # Build the 1D time grid once to mirror NLSQ trapezoidal integration
    time_grid = jnp.array(np.unique(np.asarray(prepared.t1)))

    model_kwargs = {
        "data": jnp.array(prepared.data),
        "t1": jnp.array(prepared.t1),
        "t2": jnp.array(prepared.t2),
        "phi": jnp.array(prepared.phi),
        "phi_indices": jnp.array(prepared.phi_indices),
        "q": q,
        "L": L,
        "dt": dt,
        "time_grid": time_grid,
        "analysis_mode": analysis_mode,
        "parameter_space": parameter_space,
        "n_phi": prepared.n_phi,
        "noise_scale": prepared.noise_scale,
    }

    # =========================================================================
    # 5. Select backend and run sampling
    # =========================================================================
    if shards is not None and len(shards) > 1:
        # Use parallel backend for CMC
        backend = select_backend(config)
        run_logger.info(f"Using backend: {backend.get_name()}")

        mcmc_samples = backend.run(
            model=xpcs_model,
            model_kwargs=model_kwargs,
            config=config,
            shards=shards,
            initial_values=initial_values,
            parameter_space=parameter_space,
            analysis_mode=analysis_mode,
        )
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
    dt: float = 0.1,
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
    dt : float
        Time step.

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
