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
from homodyne.utils.logging import get_logger, with_context

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
) -> Callable:
    """Create initialization strategy for NUTS.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial values from config.
    param_names : list[str]
        Expected parameter names in order.
    use_init_to_value : bool
        If True, use init_to_value when values provided.

    Returns
    -------
    Callable
        NumPyro initialization function.
    """
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
    if initial_values is not None:
        full_init = build_init_values_dict(
            n_phi=n_phi,
            analysis_mode=analysis_mode,
            initial_values=initial_values,
            parameter_space=parameter_space,
        )
    else:
        full_init = None

    # Create init strategy
    init_strategy = create_init_strategy(full_init, param_names_with_sigma)

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

    try:
        mcmc.run(rng_key, **model_kwargs)
    except Exception as e:
        run_logger.error(f"MCMC sampling failed: {e}")
        raise RuntimeError(f"MCMC sampling failed: {e}") from e

    total_time = time.perf_counter() - start_time

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
