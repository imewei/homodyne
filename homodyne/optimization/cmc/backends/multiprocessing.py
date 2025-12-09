"""Multiprocessing backend for CMC execution.

This module provides parallel MCMC execution using Python's
multiprocessing module for CPU-based parallelism.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from tqdm import tqdm

from homodyne.optimization.cmc.backends.base import CMCBackend, combine_shard_samples
from homodyne.utils.logging import get_logger, with_context

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
    threads_per_worker: int = 2,
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
    threads_per_worker : int
        Number of threads for JAX/XLA in this worker process.

    Returns
    -------
    dict[str, Any]
        Serialized MCMCSamples.
    """
    import os

    # Limit threads BEFORE importing JAX to enable true parallelism across workers
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    import jax.numpy as jnp

    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.sampler import run_nuts_sampling

    start_time = time.perf_counter()
    worker_logger = get_logger(
        __name__,
        context={"run": config_dict.get("run_id"), "shard": shard_idx},
    )
    worker_logger.debug("Starting shard worker")

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

    # Prepare model kwargs - must match xpcs_model() signature
    model_kwargs = {
        "data": jnp.array(shard_data["data"]),
        "t1": jnp.array(shard_data["t1"]),
        "t2": jnp.array(shard_data["t2"]),
        "phi_unique": jnp.array(shard_data["phi_unique"]),
        "phi_indices": jnp.array(shard_data["phi_indices"]),
        "q": shard_data["q"],
        "L": shard_data["L"],
        "dt": shard_data["dt"],
        "time_grid": jnp.array(shard_data["time_grid"]) if shard_data.get("time_grid") is not None else None,
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

        duration = time.perf_counter() - start_time
        worker_logger.info(
            f"Shard {shard_idx} finished in {duration:.2f}s with "
            f"{samples.n_samples} samples per chain"
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
            "duration": duration,
            "stats": {
                "warmup_time": stats.warmup_time,
                "sampling_time": stats.sampling_time,
                "total_time": stats.total_time,
                "num_divergent": stats.num_divergent,
            },
        }

    except Exception as e:
        duration = time.perf_counter() - start_time
        worker_logger.error(f"Shard {shard_idx} failed after {duration:.2f}s: {e}")
        return {
            "success": False,
            "shard_idx": shard_idx,
            "error": str(e),
            "duration": duration,
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
        progress_bar: bool = True,
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
        progress_bar : bool
            Whether to show progress bar for shard completion.

        Returns
        -------
        MCMCSamples
            Combined samples from all shards.
        """
        from homodyne.optimization.cmc.sampler import MCMCSamples, run_nuts_sampling

        run_logger = with_context(
            logger,
            run=getattr(config, "run_id", None),
            backend="multiprocessing",
        )

        if shards is None or len(shards) <= 1:
            # Single shard - run directly without multiprocessing
            run_logger.info("Running single-shard MCMC (no parallelization)")
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
        actual_workers = min(self.n_workers, n_shards)

        # Calculate threads per worker to avoid over-subscription
        # Each worker gets a fair share of CPU threads
        total_threads = mp.cpu_count()
        threads_per_worker = max(1, total_threads // actual_workers)

        run_logger.info(
            f"Running {n_shards} shards in parallel with {actual_workers} workers "
            f"({threads_per_worker} threads each)"
        )

        # Prepare shard data for workers - must include all fields needed by xpcs_model()
        shard_data_list = []
        for shard in shards:
            shard_data_list.append(
                {
                    "data": np.array(shard.data),
                    "t1": np.array(shard.t1),
                    "t2": np.array(shard.t2),
                    "phi_unique": np.array(shard.phi_unique),
                    "phi_indices": np.array(shard.phi_indices),
                    "q": model_kwargs["q"],
                    "L": model_kwargs["L"],
                    "dt": model_kwargs["dt"],
                    "time_grid": np.array(model_kwargs["time_grid"]) if model_kwargs.get("time_grid") is not None else None,
                    "noise_scale": shard.noise_scale,
                }
            )

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

        with ctx.Pool(processes=actual_workers) as pool:
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
                        threads_per_worker,
                    ),
                )
                async_results.append(async_result)

            # Collect results with polling-based progress bar
            # This shows progress immediately instead of waiting for first result
            pbar = tqdm(
                total=n_shards,
                desc="CMC shards",
                disable=not progress_bar,
                unit="shard",
                position=0,
                leave=True,
                dynamic_ncols=True,
            )
            # Show that sampling is in progress
            pbar.set_postfix_str("sampling...")
            pbar.refresh()

            # Timeout for NUTS sampling - scale with number of shards
            # Each shard can take ~30-60 min for complex models, but they run in parallel
            # Set per-shard timeout of 2 hours, total timeout scales with parallelism
            per_shard_timeout = 7200  # 2 hours per shard
            # With parallel execution, total time ≈ per_shard_timeout * ceil(n_shards / n_workers)
            batches = (n_shards + actual_workers - 1) // actual_workers
            timeout_seconds = per_shard_timeout * batches
            run_logger.info(
                f"Timeout set to {timeout_seconds/3600:.1f} hours "
                f"({batches} batch(es) × {per_shard_timeout/3600:.1f}h per batch)"
            )
            poll_interval = 1.0  # Check every second
            start_time = time.time()

            # Track which results we've collected
            collected = [False] * n_shards
            completed_count = 0

            while completed_count < n_shards:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    # Timeout - collect whatever we have
                    for i, _async_result in enumerate(async_results):
                        if not collected[i]:
                            timeout_msg = (
                                f"Worker timed out after {timeout_seconds}s - shard likely has "
                                "too many data points for NUTS sampling. Consider reducing "
                                "max_points_per_shard in cmc config or using fewer angles."
                            )
                            run_logger.error(timeout_msg)
                            results.append({"success": False, "shard_idx": i, "error": timeout_msg})
                            pbar.update(1)
                            completed_count += 1
                    break

                # Poll each async result
                for i, async_result in enumerate(async_results):
                    if collected[i]:
                        continue

                    if async_result.ready():
                        try:
                            result = async_result.get(timeout=0)
                            results.append(result)
                            collected[i] = True
                            completed_count += 1
                            pbar.update(1)
                            if result.get("success"):
                                pbar.set_postfix(
                                    shard=result.get("shard_idx", "?"),
                                    time=f"{result.get('duration', 0):.1f}s",
                                )
                            else:
                                pbar.set_postfix(
                                    shard=result.get("shard_idx", "?"),
                                    status="failed",
                                )
                        except Exception as e:
                            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
                            run_logger.error(f"Worker {i} failed: {error_msg}")
                            results.append({"success": False, "shard_idx": i, "error": error_msg})
                            collected[i] = True
                            completed_count += 1
                            pbar.update(1)

                # Update elapsed time in progress bar
                if completed_count < n_shards:
                    mins, secs = divmod(int(elapsed), 60)
                    pbar.set_postfix_str(f"sampling... {mins}m{secs:02d}s")
                    time.sleep(poll_interval)

            pbar.close()

        # Process results
        successful_samples = []
        shard_timings: list[tuple[int | None, float | None]] = []
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
                shard_timings.append((result.get("shard_idx"), result.get("duration")))
            else:
                run_logger.warning(
                    f"Shard {result.get('shard_idx', '?')} failed: {result.get('error', 'unknown')}"
                )

        if not successful_samples:
            raise RuntimeError("All shards failed")

        # Check success rate
        success_rate = len(successful_samples) / n_shards
        if success_rate < config.min_success_rate:
            run_logger.warning(
                f"Success rate {success_rate:.1%} below threshold {config.min_success_rate:.1%}"
            )

        for shard_idx, duration in shard_timings:
            if shard_idx is not None and duration is not None:
                run_logger.debug(f"Shard {shard_idx} completed in {duration:.2f}s")

        # Combine samples
        combined = combine_shard_samples(
            successful_samples,
            method=config.combination_method,
        )

        run_logger.info(
            f"Combined {len(successful_samples)}/{n_shards} successful shards"
        )

        return combined

    def is_available(self) -> bool:
        """Check if multiprocessing is available."""
        return True
