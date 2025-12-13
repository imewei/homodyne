"""NUTS sampler wrapper for CMC analysis.

This module provides utilities for running NumPyro NUTS sampling
with proper initialization and progress tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.priors import (
    build_init_values_dict,
    get_param_names_in_order,
)
from homodyne.optimization.cmc.scaling import (
    compute_scaling_factors,
    transform_initial_values_to_z,
)
from homodyne.utils.logging import get_logger, with_context


def _compute_mcmc_safe_d0(
    initial_values: dict[str, float] | None,
    q: float,
    dt: float,
    time_grid: np.ndarray | None,
    logger_inst,
    *,
    target_g1: float = 0.5,
    g1_threshold: float = 0.1,
) -> dict[str, float] | None:
    """Check if initial D0 causes g1→0 and compute MCMC-safe adjustment.

    When D0 is very large (or alpha very negative), the diffusion integral
    can become enormous, causing g1 = exp(-integral) → 0. This creates
    a flat likelihood surface with vanishing gradients, causing NUTS to
    reject all proposals (0% acceptance rate).

    This function detects this condition and computes a scaled D0 that
    produces g1 ≈ target_g1 at a typical time lag, ensuring gradients
    are alive for MCMC exploration.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial parameter values containing D0, alpha, D_offset.
    q : float
        Wavevector magnitude.
    dt : float
        Time step.
    time_grid : np.ndarray | None
        Time grid for integration.
    logger_inst
        Logger instance for warnings.
    target_g1 : float
        Target g1 value for adjusted D0 (default 0.5).
    g1_threshold : float
        Threshold below which D0 is adjusted (default 0.1).

    Returns
    -------
    dict[str, float] | None
        Adjusted initial values if D0 was scaled, None otherwise.
    """
    if initial_values is None:
        return None

    # Get diffusion parameters
    d0 = initial_values.get("D0")
    alpha = initial_values.get("alpha")
    d_offset = initial_values.get("D_offset")

    if d0 is None or alpha is None or d_offset is None:
        return None

    if time_grid is None or len(time_grid) < 2:
        return None

    # Safety checks
    if not np.isfinite(d0) or not np.isfinite(alpha) or not np.isfinite(d_offset):
        return None

    try:
        # Compute D(t) on the time grid
        # D(t) = D0 * t^alpha + D_offset
        epsilon = 1e-10
        time_safe = np.asarray(time_grid) + epsilon
        D_grid = d0 * (time_safe**alpha) + d_offset
        D_grid = np.maximum(D_grid, 1e-10)

        # Compute trapezoidal cumsum (without dt scaling)
        if len(D_grid) > 1:
            trap_avg = 0.5 * (D_grid[:-1] + D_grid[1:])
            cumsum = np.concatenate([[0.0], np.cumsum(trap_avg)])
        else:
            cumsum = np.cumsum(D_grid)

        # Estimate integral at typical time lag (1/4 to 3/4 of range)
        n = len(cumsum)
        idx_low = n // 4
        idx_high = 3 * n // 4
        integral_estimate = abs(cumsum[idx_high] - cumsum[idx_low])

        # Compute g1 = exp(-0.5 * q^2 * dt * integral)
        prefactor = 0.5 * q**2 * dt
        log_g1 = -prefactor * integral_estimate
        log_g1_clipped = max(log_g1, -700.0)  # Prevent underflow
        g1_estimate = np.exp(log_g1_clipped)

        logger_inst.debug(
            f"[MCMC-SAFE] g1 estimate: D0={d0:.4g}, alpha={alpha:.4g}, "
            f"integral={integral_estimate:.4g}, log_g1={log_g1:.4g}, g1={g1_estimate:.4g}"
        )

        # If g1 is too small, compute scaled D0
        if g1_estimate < g1_threshold:
            # For g1 = target_g1:
            # log(target_g1) = -prefactor * target_integral
            # target_integral = -log(target_g1) / prefactor
            target_log_g1 = np.log(target_g1)
            target_integral = -target_log_g1 / prefactor

            # Scale factor: how much smaller should the integral be?
            if integral_estimate > 0:
                scale_factor = target_integral / integral_estimate
            else:
                scale_factor = 0.01  # Fallback

            # Apply scaling to D0 (approximately linear for moderate adjustments)
            # Also adjust D_offset proportionally for consistency
            new_d0 = d0 * scale_factor
            new_d_offset = d_offset * scale_factor

            # Ensure new values are within reasonable range
            new_d0 = max(new_d0, 1.0)  # Minimum D0
            new_d_offset = max(new_d_offset, -1e6)  # Allow negative but bound

            logger_inst.warning(
                f"⚠️ MCMC-SAFE ADJUSTMENT: Initial D0={d0:.4g} causes g1≈{g1_estimate:.2e} (vanishing gradients). "
                f"Scaling D0 to {new_d0:.4g} (×{scale_factor:.4f}) for MCMC exploration stability. "
                f"The sampler can still converge to optimal values if supported by likelihood."
            )

            # Return adjusted values
            adjusted = dict(initial_values)
            adjusted["D0"] = new_d0
            adjusted["D_offset"] = new_d_offset
            return adjusted

        return None  # No adjustment needed

    except Exception as e:
        logger_inst.debug(f"[MCMC-SAFE] Check failed: {e}")
        return None


if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


@dataclass
class SamplingStats:
    """Statistics from MCMC sampling.

    Attributes
    ----------
    warmup_time : float
        Time spent in warmup phase (seconds).
    sampling_time : float
        Time spent in sampling phase (seconds).
    total_time : float
        Total sampling time (seconds).
    num_divergent : int
        Number of divergent transitions.
    accept_prob : float
        Mean acceptance probability.
    step_size : float
        Final step size.
    tree_depth : float
        Mean tree depth.
    """

    warmup_time: float = 0.0
    sampling_time: float = 0.0
    total_time: float = 0.0
    num_divergent: int = 0
    accept_prob: float = 0.0
    step_size: float = 0.0
    tree_depth: float = 0.0


@dataclass
class MCMCSamples:
    """Container for MCMC samples.

    Attributes
    ----------
    samples : dict[str, np.ndarray]
        Parameter samples, shape (n_chains, n_samples) per parameter.
    param_names : list[str]
        Parameter names in sampling order.
    n_chains : int
        Number of chains.
    n_samples : int
        Number of samples per chain.
    extra_fields : dict[str, Any]
        Additional MCMC info (divergences, energy, etc.).
    """

    samples: dict[str, np.ndarray]
    param_names: list[str]
    n_chains: int
    n_samples: int
    extra_fields: dict[str, Any] = field(default_factory=dict)


def create_init_strategy(
    initial_values: dict[str, float] | None,
    param_names: list[str],
    use_init_to_value: bool = True,
    z_space_values: dict[str, float] | None = None,
) -> Callable:
    """Create initialization strategy for NUTS.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial values from config (original space).
    param_names : list[str]
        Expected parameter names in order.
    use_init_to_value : bool
        If True, use init_to_value when values provided.
    z_space_values : dict[str, float] | None
        Initial values in z-space (for scaled model). If provided,
        these are used directly as {name}_z values.

    Returns
    -------
    Callable
        NumPyro initialization function.
    """
    # For scaled model, use z-space values
    if z_space_values is not None and use_init_to_value:
        if z_space_values:
            logger.debug(
                f"Using init_to_value (z-space) for {len(z_space_values)} params: "
                f"{list(z_space_values.keys())[:5]}..."
            )
            return init_to_value(values=z_space_values)

    # For unscaled model, use original values
    if initial_values is not None and use_init_to_value:
        # Filter to only parameters we're sampling (exclude deterministics)
        init_dict = {}
        for name in param_names:
            if name in initial_values:
                init_dict[name] = initial_values[name]

        if init_dict:
            logger.debug(
                f"Using init_to_value for {len(init_dict)} params: {list(init_dict.keys())}"
            )
            return init_to_value(values=init_dict)

    # Fallback to median initialization
    logger.debug("Using init_to_median (no initial values)")
    return init_to_median()


def run_nuts_sampling(
    model: Callable,
    model_kwargs: dict[str, Any],
    config: CMCConfig,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
    n_phi: int,
    analysis_mode: str,
    rng_key: jax.random.PRNGKey | None = None,
    progress_bar: bool = True,
) -> tuple[MCMCSamples, SamplingStats]:
    """Run NUTS sampling with configuration.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    model_kwargs : dict[str, Any]
        Keyword arguments to pass to model.
    config : CMCConfig
        CMC configuration.
    initial_values : dict[str, float] | None
        Initial parameter values from config.
    parameter_space : ParameterSpace
        Parameter space for building init values.
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    rng_key : jax.random.PRNGKey | None
        Random key. If None, creates from seed.
    progress_bar : bool
        Whether to show progress bar.

    Returns
    -------
    tuple[MCMCSamples, SamplingStats]
        Samples and timing statistics.
    """
    run_logger = with_context(logger, run=getattr(config, "run_id", None))

    # Get parameter names in correct order
    param_names = get_param_names_in_order(n_phi, analysis_mode)
    # Add sigma (noise parameter)
    param_names_with_sigma = param_names + ["sigma"]

    # Build full init values dict if needed
    # Extract data arrays from model_kwargs for data-driven estimation
    c2_data = model_kwargs.get("data")
    t1_data = model_kwargs.get("t1")
    t2_data = model_kwargs.get("t2")
    phi_indices = model_kwargs.get("phi_indices")

    # Convert JAX arrays to numpy for estimation (if needed)
    if c2_data is not None and hasattr(c2_data, "__array__"):
        c2_data = np.asarray(c2_data)
    if t1_data is not None and hasattr(t1_data, "__array__"):
        t1_data = np.asarray(t1_data)
    if t2_data is not None and hasattr(t2_data, "__array__"):
        t2_data = np.asarray(t2_data)
    if phi_indices is not None and hasattr(phi_indices, "__array__"):
        phi_indices = np.asarray(phi_indices)

    # =========================================================================
    # MCMC-SAFE D0 CHECK: Detect and fix vanishing gradient regime
    # =========================================================================
    # When D0 is very large (or alpha very negative), g1 → 0 everywhere,
    # causing vanishing gradients and 0% NUTS acceptance rate.
    # This check detects that condition and scales D0 to ensure gradients are alive.
    q = model_kwargs.get("q", 0.01)
    dt = model_kwargs.get("dt", 0.1)
    time_grid = model_kwargs.get("time_grid")
    if time_grid is not None and hasattr(time_grid, "__array__"):
        time_grid_np = np.asarray(time_grid)
    else:
        time_grid_np = time_grid

    adjusted_init = _compute_mcmc_safe_d0(
        initial_values=initial_values,
        q=q,
        dt=dt,
        time_grid=time_grid_np,
        logger_inst=run_logger,
    )

    # Use adjusted values if D0 was scaled
    effective_init_values = (
        adjusted_init if adjusted_init is not None else initial_values
    )

    full_init = build_init_values_dict(
        n_phi=n_phi,
        analysis_mode=analysis_mode,
        initial_values=effective_init_values,
        parameter_space=parameter_space,
        c2_data=c2_data,
        t1=t1_data,
        t2=t2_data,
        phi_indices=phi_indices,
    )

    # =========================================================================
    # GRADIENT BALANCING: Transform initial values to z-space for scaled model
    # =========================================================================
    # The scaled model (xpcs_model_scaled) samples in normalized z-space where
    # z ~ Normal(0, 1). We need to transform our initial values to this space
    # for proper initialization with init_to_value.
    scalings = compute_scaling_factors(parameter_space, n_phi, analysis_mode)
    z_space_init = transform_initial_values_to_z(full_init, scalings)

    # Log scaling transformation info
    run_logger.info(
        f"Gradient balancing: {len(scalings)} params transformed to unit scale. "
        f"Sample scale range: {min(s.scale for s in scalings.values()):.2e} to "
        f"{max(s.scale for s in scalings.values()):.2e}"
    )

    # Create init strategy with z-space values for scaled model
    init_strategy = create_init_strategy(
        full_init, param_names_with_sigma, z_space_values=z_space_init
    )

    # Create NUTS kernel
    kernel = NUTS(
        model,
        init_strategy=init_strategy,
        target_accept_prob=config.target_accept_prob,
    )

    # Create MCMC runner
    mcmc = MCMC(
        kernel,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        progress_bar=progress_bar,
    )

    # Create RNG key
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    # Run sampling with timing
    run_logger.info(
        f"Starting NUTS sampling: {config.num_chains} chains, "
        f"{config.num_warmup} warmup, {config.num_samples} samples"
    )

    start_time = time.perf_counter()
    run_logger.info("NUTS phase: JIT compile + sampling started (may take minutes)...")

    try:
        mcmc.run(rng_key, **model_kwargs)
    except Exception as e:
        run_logger.error(f"MCMC sampling failed: {e}")
        raise RuntimeError(f"MCMC sampling failed: {e}") from e

    total_time = time.perf_counter() - start_time
    run_logger.info(f"NUTS finished in {total_time:.1f}s")

    # Extract samples
    samples = mcmc.get_samples(group_by_chain=True)

    # Convert to numpy and proper format
    samples_np: dict[str, np.ndarray] = {}
    for name, arr in samples.items():
        samples_np[name] = np.array(arr)

    # Get extra fields (divergences, etc.)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    extra_fields = {k: np.array(v) for k, v in extra.items()}

    # Compute statistics
    num_divergent = 0
    if "diverging" in extra_fields:
        num_divergent = int(np.sum(extra_fields["diverging"]))

    accept_prob = 0.0
    if "accept_prob" in extra_fields:
        accept_prob = float(np.mean(extra_fields["accept_prob"]))

    # Critical warning for zero acceptance rate
    if accept_prob < 0.001:
        run_logger.warning(
            "⚠️ CRITICAL: Acceptance rate is essentially 0% - all proposals rejected! "
            "This indicates severe sampling problems. Possible causes:\n"
            "  1. Initial values are outside prior support or at boundaries\n"
            "  2. Likelihood returns -inf due to numerical issues (NaN/overflow)\n"
            "  3. Prior is too narrow for the data\n"
            "  4. Step size adaptation failed during warmup\n"
            "Consider: checking initial values, widening priors, or running NLSQ first."
        )

    # Check for numerical issues exposed by the model
    if "n_numerical_issues" in extra_fields:
        n_issues = int(np.sum(extra_fields["n_numerical_issues"]))
        if n_issues > 0:
            total_evals = config.num_samples * config.num_chains
            issue_rate = n_issues / max(total_evals, 1)
            run_logger.warning(
                f"⚠️ Numerical issues detected: {n_issues:,} NaN/Inf occurrences "
                f"({issue_rate:.1%} of evaluations). "
                "This may indicate parameter combinations causing overflow in physics model."
            )

    # Estimate warmup vs sampling time (rough estimate)
    warmup_ratio = config.num_warmup / (config.num_warmup + config.num_samples)
    warmup_time = total_time * warmup_ratio
    sampling_time = total_time * (1 - warmup_ratio)

    stats = SamplingStats(
        warmup_time=warmup_time,
        sampling_time=sampling_time,
        total_time=total_time,
        num_divergent=num_divergent,
        accept_prob=accept_prob,
    )

    run_logger.info(
        f"Sampling complete in {total_time:.1f}s, "
        f"{num_divergent} divergences, "
        f"accept_prob={accept_prob:.3f}"
    )

    # Create MCMCSamples object
    mcmc_samples = MCMCSamples(
        samples=samples_np,
        param_names=[k for k in samples_np.keys() if k != "obs"],
        n_chains=config.num_chains,
        n_samples=config.num_samples,
        extra_fields=extra_fields,
    )

    return mcmc_samples, stats


def run_nuts_with_retry(
    model: Callable,
    model_kwargs: dict[str, Any],
    config: CMCConfig,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
    n_phi: int,
    analysis_mode: str,
    max_retries: int = 3,
    rng_key: jax.random.PRNGKey | None = None,
) -> tuple[MCMCSamples, SamplingStats]:
    """Run NUTS sampling with automatic retry on failure.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    model_kwargs : dict[str, Any]
        Model arguments.
    config : CMCConfig
        Configuration.
    initial_values : dict[str, float] | None
        Initial values.
    parameter_space : ParameterSpace
        Parameter space.
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    max_retries : int
        Maximum number of retry attempts.
    rng_key : jax.random.PRNGKey | None
        Random key.

    Returns
    -------
    tuple[MCMCSamples, SamplingStats]
        Samples and statistics.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    last_error = None
    run_logger = with_context(logger, run=getattr(config, "run_id", None))

    for attempt in range(max_retries):
        attempt_num = attempt + 1
        attempt_logger = with_context(run_logger, attempt=attempt_num)
        attempt_start = time.perf_counter()
        attempt_logger.info(
            f"🔄 Attempt {attempt_num}/{max_retries}: starting NUTS "
            f"(chains={config.num_chains}, samples={config.num_samples})"
        )
        try:
            # Use different RNG key for each attempt
            attempt_key = jax.random.fold_in(rng_key, attempt)

            samples, stats = run_nuts_sampling(
                model=model,
                model_kwargs=model_kwargs,
                config=config,
                initial_values=initial_values,
                parameter_space=parameter_space,
                n_phi=n_phi,
                analysis_mode=analysis_mode,
                rng_key=attempt_key,
                progress_bar=attempt == 0,  # Only show progress on first attempt
            )

            # Check for excessive divergences
            divergence_rate = stats.num_divergent / (
                config.num_samples * config.num_chains
            )
            duration = time.perf_counter() - attempt_start
            if divergence_rate > 0.1:  # More than 10% divergences
                attempt_logger.warning(
                    f"🔄 Attempt {attempt_num}/{max_retries}: divergence_rate={divergence_rate:.1%}, retrying..."
                )
                last_error = RuntimeError(
                    f"High divergence rate: {divergence_rate:.1%}"
                )
                continue

            attempt_logger.info(
                f"✅ Attempt {attempt_num}/{max_retries} succeeded in {duration:.2f}s "
                f"(divergences={stats.num_divergent}, accept_prob={stats.accept_prob:.3f})"
            )
            return samples, stats

        except Exception as e:
            duration = time.perf_counter() - attempt_start
            attempt_logger.warning(
                f"❌ Attempt {attempt_num}/{max_retries} failed after {duration:.2f}s: {e}"
            )
            last_error = e

    run_logger.error(
        f"❌ MCMC sampling failed after {max_retries} attempts: {last_error}"
    )
    # All retries failed
    raise RuntimeError(
        f"MCMC sampling failed after {max_retries} attempts: {last_error}"
    )
