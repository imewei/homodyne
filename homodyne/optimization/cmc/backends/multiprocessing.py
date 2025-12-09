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


def _run_shard_worker_with_queue(
    shard_idx: int,
    shard_data: dict[str, Any],
    model_fn: Callable,
    config_dict: dict[str, Any],
    initial_values: dict[str, float] | None,
    parameter_space_dict: dict[str, Any],
    n_phi: int,
    analysis_mode: str,
    threads_per_worker: int,
    result_queue: mp.Queue,
) -> None:
    """Worker function that puts result in a queue for proper timeout handling."""
    result = _run_shard_worker(
        shard_idx=shard_idx,
        shard_data=shard_data,
        model_fn=model_fn,
        config_dict=config_dict,
        initial_values=initial_values,
        parameter_space_dict=parameter_space_dict,
        n_phi=n_phi,
        analysis_mode=analysis_mode,
        threads_per_worker=threads_per_worker,
    )
    result_queue.put(result)


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

        # Multiple shards - run in parallel with per-shard timeout enforcement
        n_shards = len(shards)
        actual_workers = min(self.n_workers, n_shards)

        # Calculate threads per worker to avoid over-subscription
        total_threads = mp.cpu_count()
        threads_per_worker = max(1, total_threads // actual_workers)

        run_logger.info(
            f"Running {n_shards} shards in parallel with {actual_workers} workers "
            f"({threads_per_worker} threads each)"
        )

        # Per-shard timeout - enforced per individual process
        per_shard_timeout = config.per_shard_timeout  # Default: 7200s (2 hours)
        run_logger.info(
            f"Per-shard timeout: {per_shard_timeout/3600:.1f} hours "
            f"(processes will be terminated if exceeded)"
        )

        # Prepare shard data for workers
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

        if hasattr(parameter_space, "_config_dict"):
            ps_dict = parameter_space._config_dict
        else:
            ps_dict = model_kwargs.get("config_dict", {})

        # Use spawn context for clean process isolation
        ctx = mp.get_context(self.spawn_method)
        result_queue = ctx.Queue()

        # Track active processes: {shard_idx: (process, start_time)}
        active_processes: dict[int, tuple[mp.Process, float]] = {}
        pending_shards = list(range(n_shards))
        results = []
        completed_count = 0

        # Progress bar
        pbar = tqdm(
            total=n_shards,
            desc="CMC shards",
            disable=not progress_bar,
            unit="shard",
            position=0,
            leave=True,
            dynamic_ncols=True,
        )
        pbar.set_postfix_str("starting...")
        pbar.refresh()

        start_time = time.time()
        poll_interval = 2.0  # Check every 2 seconds

        try:
            while completed_count < n_shards:
                # Launch new processes up to max workers
                while len(active_processes) < actual_workers and pending_shards:
                    shard_idx = pending_shards.pop(0)
                    shard_data = shard_data_list[shard_idx]

                    process = ctx.Process(
                        target=_run_shard_worker_with_queue,
                        args=(
                            shard_idx,
                            shard_data,
                            model,
                            config_dict,
                            initial_values,
                            ps_dict,
                            shards[shard_idx].n_phi,
                            analysis_mode,
                            threads_per_worker,
                            result_queue,
                        ),
                    )
                    process.start()
                    active_processes[shard_idx] = (process, time.time())
                    run_logger.debug(f"Started shard {shard_idx} (pid={process.pid})")

                # Check for completed or timed-out processes
                for shard_idx, (process, proc_start_time) in list(active_processes.items()):
                    proc_elapsed = time.time() - proc_start_time

                    if not process.is_alive():
                        # Process finished - it should have put result in queue
                        process.join(timeout=1)
                        del active_processes[shard_idx]
                        run_logger.debug(f"Shard {shard_idx} process exited after {proc_elapsed:.1f}s")

                    elif proc_elapsed > per_shard_timeout:
                        # Per-shard timeout exceeded - terminate the process
                        run_logger.warning(
                            f"Shard {shard_idx} timed out after {proc_elapsed:.0f}s "
                            f"(limit: {per_shard_timeout}s), terminating process (pid={process.pid})"
                        )
                        process.terminate()
                        process.join(timeout=5)  # Give 5s to cleanup
                        if process.is_alive():
                            run_logger.warning(f"Shard {shard_idx} did not terminate, killing")
                            process.kill()
                            process.join(timeout=2)

                        del active_processes[shard_idx]
                        results.append({
                            "success": False,
                            "shard_idx": shard_idx,
                            "error": f"Timeout after {proc_elapsed:.0f}s (limit: {per_shard_timeout}s)",
                            "duration": proc_elapsed,
                        })
                        completed_count += 1
                        pbar.update(1)
                        pbar.set_postfix(shard=shard_idx, status="timeout")

                # Collect results from queue (non-blocking)
                while True:
                    try:
                        result = result_queue.get_nowait()
                        results.append(result)
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
                    except Exception:
                        # Queue empty
                        break

                # Update progress bar with elapsed time
                if completed_count < n_shards:
                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    hrs, mins = divmod(mins, 60)
                    if hrs > 0:
                        pbar.set_postfix_str(f"active={len(active_processes)} elapsed={hrs}h{mins:02d}m")
                    else:
                        pbar.set_postfix_str(f"active={len(active_processes)} elapsed={mins}m{secs:02d}s")
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            run_logger.warning("Interrupted - terminating all active processes")
            for shard_idx, (process, _) in active_processes.items():
                run_logger.debug(f"Terminating shard {shard_idx} (pid={process.pid})")
                process.terminate()
                process.join(timeout=2)
            raise

        finally:
            pbar.close()
            # Clean up any remaining active processes
            for shard_idx, (process, _) in list(active_processes.items()):
                if process.is_alive():
                    run_logger.warning(f"Cleaning up orphan process for shard {shard_idx}")
                    process.terminate()
                    process.join(timeout=2)

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
