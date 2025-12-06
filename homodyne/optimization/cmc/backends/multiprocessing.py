"""Multiprocessing backend for CMC execution.

This module provides parallel MCMC execution using Python's
multiprocessing module for CPU-based parallelism.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from homodyne.optimization.cmc.backends.base import CMCBackend, combine_shard_samples
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)


def _run_shard_worker(
    shard_idx: int,
    shard_data: dict[str, Any],
    model_fn: Callable,
    config_dict: dict[str, Any],
    initial_values: dict[str, float] | None,
    parameter_space_dict: dict[str, Any],
    n_phi: int,
    analysis_mode: str,
) -> dict[str, Any]:
    """Worker function for processing a single shard.

    This runs in a separate process.

    Parameters
    ----------
    shard_idx : int
        Shard index for logging.
    shard_data : dict[str, Any]
        Shard data dictionary.
    model_fn : Callable
        NumPyro model function.
    config_dict : dict[str, Any]
        CMC configuration as dict.
    initial_values : dict[str, float] | None
        Initial parameter values.
    parameter_space_dict : dict[str, Any]
        Serialized parameter space.
    n_phi : int
        Number of phi angles in this shard.
    analysis_mode : str
        Analysis mode.

    Returns
    -------
    dict[str, Any]
        Serialized MCMCSamples.
    """
    import jax.numpy as jnp

    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.sampler import run_nuts_sampling

    # Reconstruct objects from dicts
    config = CMCConfig.from_dict(config_dict)

    # Reconstruct ParameterSpace
    # For now, we pass the config dict directly
    parameter_space = ParameterSpace.from_config(
        config_dict=parameter_space_dict,
        analysis_mode=analysis_mode,
    )

    # Create RNG key for this shard
    rng_key = jax.random.PRNGKey(42 + shard_idx)

    # Prepare model kwargs
    model_kwargs = {
        "data": jnp.array(shard_data["data"]),
        "t1": jnp.array(shard_data["t1"]),
        "t2": jnp.array(shard_data["t2"]),
        "phi": jnp.array(shard_data["phi"]),
        "phi_indices": jnp.array(shard_data["phi_indices"]),
        "q": shard_data["q"],
        "L": shard_data["L"],
        "dt": shard_data["dt"],
        "analysis_mode": analysis_mode,
        "parameter_space": parameter_space,
        "n_phi": n_phi,
        "noise_scale": shard_data.get("noise_scale", 0.1),
    }

    try:
        # Run sampling
        samples, stats = run_nuts_sampling(
            model=model_fn,
            model_kwargs=model_kwargs,
            config=config,
            initial_values=initial_values,
            parameter_space=parameter_space,
            n_phi=n_phi,
            analysis_mode=analysis_mode,
            rng_key=rng_key,
            progress_bar=False,  # Disable in worker
        )

        # Serialize for return
        return {
            "success": True,
            "shard_idx": shard_idx,
            "samples": {k: np.array(v) for k, v in samples.samples.items()},
            "param_names": samples.param_names,
            "n_chains": samples.n_chains,
            "n_samples": samples.n_samples,
            "extra_fields": {k: np.array(v) for k, v in samples.extra_fields.items()},
            "stats": {
                "warmup_time": stats.warmup_time,
                "sampling_time": stats.sampling_time,
                "total_time": stats.total_time,
                "num_divergent": stats.num_divergent,
            },
        }

    except Exception as e:
        logger.error(f"Shard {shard_idx} failed: {e}")
        return {
            "success": False,
            "shard_idx": shard_idx,
            "error": str(e),
        }


class MultiprocessingBackend(CMCBackend):
    """CMC backend using Python multiprocessing.

    Runs MCMC sampling in parallel across CPU cores using
    Python's multiprocessing module.
    """

    def __init__(
        self,
        n_workers: int | None = None,
        spawn_method: str = "spawn",
    ):
        """Initialize multiprocessing backend.

        Parameters
        ----------
        n_workers : int | None
            Number of worker processes. If None, uses CPU count.
        spawn_method : str
            Process start method: "spawn", "fork", or "forkserver".
        """
        if n_workers is None:
            # Use physical cores, leave some for main process
            n_workers = max(1, mp.cpu_count() - 2)

        self.n_workers = n_workers
        self.spawn_method = spawn_method

    def get_name(self) -> str:
        """Get backend name."""
        return f"multiprocessing({self.n_workers} workers)"

    def run(
        self,
        model: Callable,
        model_kwargs: dict[str, Any],
        config: CMCConfig,
        shards: list[PreparedData] | None = None,
        initial_values: dict[str, float] | None = None,
        parameter_space: ParameterSpace = None,
        analysis_mode: str = "static",
    ) -> MCMCSamples:
        """Run MCMC sampling across shards.

        Parameters
        ----------
        model : Callable
            NumPyro model function.
        model_kwargs : dict[str, Any]
            Common model arguments.
        config : CMCConfig
            CMC configuration.
        shards : list[PreparedData] | None
            Data shards.
        initial_values : dict[str, float] | None
            Initial parameter values.
        parameter_space : ParameterSpace
            Parameter space for priors.
        analysis_mode : str
            Analysis mode.

        Returns
        -------
        MCMCSamples
            Combined samples from all shards.
        """
        from homodyne.optimization.cmc.sampler import MCMCSamples, run_nuts_sampling

        if shards is None or len(shards) <= 1:
            # Single shard - run directly without multiprocessing
            logger.info("Running single-shard MCMC (no parallelization)")
            samples, stats = run_nuts_sampling(
                model=model,
                model_kwargs=model_kwargs,
                config=config,
                initial_values=initial_values,
                parameter_space=parameter_space,
                n_phi=model_kwargs.get("n_phi", 1),
                analysis_mode=analysis_mode,
                progress_bar=True,
            )
            return samples

        # Multiple shards - run in parallel
        n_shards = len(shards)
        logger.info(
            f"Running {n_shards} shards in parallel with {self.n_workers} workers"
        )

        # Prepare shard data for workers
        shard_data_list = []
        for shard in shards:
            shard_data_list.append({
                "data": np.array(shard.data),
                "t1": np.array(shard.t1),
                "t2": np.array(shard.t2),
                "phi": np.array(shard.phi),
                "phi_indices": np.array(shard.phi_indices),
                "q": model_kwargs["q"],
                "L": model_kwargs["L"],
                "dt": model_kwargs["dt"],
                "noise_scale": shard.noise_scale,
            })

        # Serialize config and parameter_space
        config_dict = config.to_dict()

        # Get parameter space config (need to serialize)
        if hasattr(parameter_space, "_config_dict"):
            ps_dict = parameter_space._config_dict
        else:
            # Fallback - use model_kwargs config if available
            ps_dict = model_kwargs.get("config_dict", {})

        # Run workers
        results = []

        # Use spawn context for clean process isolation
        ctx = mp.get_context(self.spawn_method)

        with ctx.Pool(processes=min(self.n_workers, n_shards)) as pool:
            # Submit all shards
            async_results = []
            for i, shard_data in enumerate(shard_data_list):
                async_result = pool.apply_async(
                    _run_shard_worker,
                    args=(
                        i,
                        shard_data,
                        model,
                        config_dict,
                        initial_values,
                        ps_dict,
                        shards[i].n_phi,
                        analysis_mode,
                    ),
                )
                async_results.append(async_result)

            # Collect results
            for async_result in async_results:
                try:
                    result = async_result.get(timeout=3600)  # 1 hour timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Worker failed: {e}")
                    results.append({"success": False, "error": str(e)})

        # Process results
        successful_samples = []
        for result in results:
            if result["success"]:
                # Reconstruct MCMCSamples
                samples = MCMCSamples(
                    samples=result["samples"],
                    param_names=result["param_names"],
                    n_chains=result["n_chains"],
                    n_samples=result["n_samples"],
                    extra_fields=result["extra_fields"],
                )
                successful_samples.append(samples)
            else:
                logger.warning(
                    f"Shard {result.get('shard_idx', '?')} failed: {result.get('error', 'unknown')}"
                )

        if not successful_samples:
            raise RuntimeError("All shards failed")

        # Check success rate
        success_rate = len(successful_samples) / n_shards
        if success_rate < config.min_success_rate:
            logger.warning(
                f"Success rate {success_rate:.1%} below threshold {config.min_success_rate:.1%}"
            )

        # Combine samples
        combined = combine_shard_samples(
            successful_samples,
            method=config.combination_method,
        )

        logger.info(
            f"Combined {len(successful_samples)}/{n_shards} successful shards"
        )

        return combined

    def is_available(self) -> bool:
        """Check if multiprocessing is available."""
        return True
