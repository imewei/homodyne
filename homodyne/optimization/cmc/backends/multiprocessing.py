"""Multiprocessing backend for CMC execution.

This module provides parallel MCMC execution using Python's
multiprocessing module for CPU-based parallelism.

Optimizations (v2.9.1):
- Batch PRNG key generation: Pre-generate all shard keys in single JAX call
- Adaptive polling: Adjust poll interval based on shard activity
- Event.wait heartbeat: Efficient heartbeat using Event.wait(timeout)
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from homodyne.optimization.cmc.backends.base import CMCBackend, combine_shard_samples
from homodyne.optimization.cmc.diagnostics import check_shard_bimodality
from homodyne.utils.logging import get_logger, log_exception, with_context

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)


class SharedDataManager:
    """Manages shared memory blocks for data common to all CMC shards.

    Uses multiprocessing.shared_memory to share config, parameter space,
    initial values, and time_grid across spawned worker processes, avoiding
    redundant pickling per shard.

    Note on serialization: Uses pickle internally for trusted config dicts
    only (CMCConfig.to_dict(), ParameterSpace). This matches the existing
    multiprocessing behavior which also pickles all process arguments.

    Must be used as a context manager or call cleanup() in a finally block.
    """

    def __init__(self) -> None:
        self._shared_blocks: list[mp.shared_memory.SharedMemory] = []
        self._refs: dict[str, dict[str, Any]] = {}

    def create_shared_bytes(self, name: str, data: bytes) -> dict[str, Any]:
        """Store bytes in shared memory."""
        shm = mp.shared_memory.SharedMemory(create=True, size=len(data))
        shm.buf[: len(data)] = data
        self._shared_blocks.append(shm)
        ref = {"shm_name": shm.name, "size": len(data), "type": "bytes"}
        self._refs[name] = ref
        return ref

    def create_shared_array(self, name: str, array: np.ndarray) -> dict[str, Any]:
        """Store a numpy array in shared memory."""
        shm = mp.shared_memory.SharedMemory(create=True, size=array.nbytes)
        shared_arr = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared_arr[:] = array
        self._shared_blocks.append(shm)
        ref = {
            "shm_name": shm.name,
            "shape": array.shape,
            "dtype": str(array.dtype),
            "type": "array",
        }
        self._refs[name] = ref
        return ref

    def create_shared_dict(self, name: str, d: dict) -> dict[str, Any]:
        """Serialize a trusted internal dict to shared memory.

        Only used for CMCConfig and ParameterSpace dicts — never for
        external/untrusted data.
        """
        import pickle as _pkl  # noqa: S403 — trusted internal data

        return self.create_shared_bytes(name, _pkl.dumps(d))

    def cleanup(self) -> None:
        """Release all shared memory blocks. Must be called in a finally block."""
        for shm in self._shared_blocks:
            try:
                shm.close()
                shm.unlink()
            except (FileNotFoundError, OSError):
                pass
        self._shared_blocks.clear()
        self._refs.clear()

    def __enter__(self) -> SharedDataManager:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


def _load_shared_bytes(ref: dict[str, Any]) -> bytes:
    """Reconstruct bytes from a shared memory reference."""
    shm = mp.shared_memory.SharedMemory(name=ref["shm_name"], create=False)
    try:
        data = bytes(shm.buf[: ref["size"]])
    finally:
        shm.close()
    return data


def _load_shared_dict(ref: dict[str, Any]) -> dict:
    """Reconstruct a trusted internal dict from shared memory.

    Only used for CMCConfig and ParameterSpace dicts — never for
    external/untrusted data.
    """
    import pickle as _pkl  # noqa: S403 — trusted internal data

    return _pkl.loads(_load_shared_bytes(ref))  # noqa: S301


def _load_shared_array(ref: dict[str, Any]) -> np.ndarray:
    """Reconstruct a numpy array from a shared memory reference."""
    shm = mp.shared_memory.SharedMemory(name=ref["shm_name"], create=False)
    try:
        arr = np.ndarray(
            ref["shape"], dtype=np.dtype(ref["dtype"]), buffer=shm.buf
        ).copy()  # Copy so we don't hold a reference to the shared buffer
    finally:
        shm.close()
    return arr


def _generate_shard_keys(n_shards: int, seed: int = 42) -> list[tuple[int, int]]:
    """Pre-generate all shard PRNG keys in a single JAX call.

    This is more efficient than generating keys one-at-a-time in each worker,
    as it amortizes JAX compilation overhead across all shards.

    Parameters
    ----------
    n_shards : int
        Number of shards to generate keys for.
    seed : int
        Base seed for PRNG key generation.

    Returns
    -------
    list[tuple[int, int]]
        List of (key_high, key_low) tuples that can be used to reconstruct
        JAX PRNG keys in worker processes without importing JAX here.
    """
    import jax
    import jax.numpy as jnp

    # Generate all keys at once using jax.random.split
    base_key = jax.random.PRNGKey(seed)
    # Split into n_shards + 1 keys (first is throwaway, rest are for shards)
    all_keys = jax.random.split(base_key, n_shards + 1)
    shard_keys = all_keys[1:]  # Skip the first key

    # Convert to serializable format (tuples of ints)
    # JAX keys are uint32[2] arrays
    key_tuples = []
    for key in shard_keys:
        key_array = jnp.array(key, dtype=jnp.uint32)
        key_tuples.append((int(key_array[0]), int(key_array[1])))

    return key_tuples


def _get_physical_cores() -> int:
    """Get physical core count using psutil for accurate detection.

    Falls back to os.cpu_count() // 2 if psutil unavailable.
    """
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        if physical is not None:
            return physical
    except ImportError:
        pass
    # Fallback: assume hyperthreading (logical = 2 * physical)
    import os

    return max(1, (os.cpu_count() or 1) // 2)


def _compute_threads_per_worker(total_threads: int, workers: int) -> int:
    """Derive a conservative thread budget per worker to avoid oversubscription.

    Uses psutil for accurate physical core detection when available,
    otherwise approximates physical cores as half of logical (common HT layout).
    Divides the budget across workers, clamping to at least 1.
    """
    physical_cores = _get_physical_cores()
    # Use physical cores as the safe pool (avoid hyperthreading contention)
    safe_pool = max(1, min(total_threads, physical_cores))
    worker_count = max(1, workers)
    return max(1, safe_pool // worker_count)


def _run_shard_worker_with_queue(
    shard_idx: int,
    shard_data: dict[str, Any],
    model_fn: Callable,
    config_ref: dict[str, Any],
    initial_values_ref: dict[str, Any] | None,
    ps_ref: dict[str, Any],
    shared_kwargs_ref: dict[str, Any],
    time_grid_ref: dict[str, Any] | None,
    n_phi: int,
    analysis_mode: str,
    threads_per_worker: int,
    result_queue: mp.Queue,
    rng_key_tuple: tuple[int, int] | None = None,
) -> None:
    """Worker function that puts result in a queue for proper timeout handling.

    Accepts shared memory references instead of full dicts to avoid redundant
    pickling. Reconstructs shared data from shared memory blocks.
    """
    # Reconstruct shared data from shared memory
    config_dict = _load_shared_dict(config_ref)
    parameter_space_dict = _load_shared_dict(ps_ref)
    shared_kwargs = _load_shared_dict(shared_kwargs_ref)
    initial_values = (
        _load_shared_dict(initial_values_ref)
        if initial_values_ref is not None
        else None
    )

    # Reconstruct time_grid
    time_grid = _load_shared_array(time_grid_ref) if time_grid_ref is not None else None

    # Merge shared kwargs into shard_data for backward-compatible worker interface
    shard_data["time_grid"] = time_grid
    shard_data.update(shared_kwargs)

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
        result_queue=result_queue,
        rng_key_tuple=rng_key_tuple,
    )
    try:
        result_queue.put_nowait(result)
    except Exception:  # noqa: S110 - Best-effort queue put, parent handles failures
        # If the queue is already full or closed, drop the result; the parent
        # loop will have marked the shard as failed. This is a best-effort send.
        pass


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
    result_queue: mp.Queue | None = None,
    rng_key_tuple: tuple[int, int] | None = None,
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
    rng_key_tuple : tuple[int, int] | None
        Pre-generated PRNG key as (high, low) tuple. If None, generates
        a key based on shard_idx (legacy behavior).

    Returns
    -------
    dict[str, Any]
        Serialized MCMCSamples.
    """
    import os

    # Configure worker threading to avoid oversubscription across workers.
    # The parent process clears these before spawning, but we set them here
    # as a safety net in case the spawn context inherited stale values.
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)
    # CRITICAL: Clear OMP_PROC_BIND and OMP_PLACES to prevent thread pinning.
    # When set, each worker's OpenMP runtime tries to pin threads to the same
    # physical cores, causing severe contention across concurrent workers.
    os.environ.pop("OMP_PROC_BIND", None)
    os.environ.pop("OMP_PLACES", None)

    import jax
    import jax.numpy as jnp

    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.sampler import run_nuts_sampling

    start_time = time.perf_counter()
    worker_logger = get_logger(
        __name__,
        context={"run": config_dict.get("run_id"), "shard": shard_idx},
    )
    # T044: Log shard start with data range and point count
    n_points = len(shard_data["data"])
    worker_logger.info(
        f"Shard {shard_idx} starting: {n_points:,} points, "
        f"n_phi={n_phi}, mode={analysis_mode}"
    )

    # Reconstruct objects from dicts
    config = CMCConfig.from_dict(config_dict)

    # Reconstruct ParameterSpace
    # For now, we pass the config dict directly
    parameter_space = ParameterSpace.from_config(
        config_dict=parameter_space_dict,
        analysis_mode=analysis_mode,
    )

    # Create RNG key for this shard
    # Use pre-generated key if available (batch optimization), else generate locally
    if rng_key_tuple is not None:
        # Reconstruct JAX PRNG key from tuple
        rng_key = jnp.array(rng_key_tuple, dtype=jnp.uint32)
    else:
        # Legacy behavior: generate key based on shard index
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
        "time_grid": (
            jnp.array(shard_data["time_grid"])
            if shard_data.get("time_grid") is not None
            else None
        ),
        "analysis_mode": analysis_mode,
        "parameter_space": parameter_space,
        "n_phi": n_phi,
        "noise_scale": shard_data.get("noise_scale", 0.1),
    }

    # Restore fixed parameters if present
    if shard_data.get("fixed_contrast") is not None:
        model_kwargs["fixed_contrast"] = shard_data["fixed_contrast"]
    if shard_data.get("fixed_offset") is not None:
        model_kwargs["fixed_offset"] = shard_data["fixed_offset"]

    # Restore per_angle_mode, nlsq_prior_config, and propagated kwargs
    per_angle_mode = shard_data.get("per_angle_mode", "individual")
    model_kwargs["per_angle_mode"] = per_angle_mode
    if shard_data.get("nlsq_prior_config") is not None:
        model_kwargs["nlsq_prior_config"] = shard_data["nlsq_prior_config"]

    # Restore num_shards for prior tempering (Feb 2026 fix)
    if shard_data.get("num_shards") is not None:
        model_kwargs["num_shards"] = shard_data["num_shards"]

    # Restore reparam_config from serialized dict (Feb 2026 fix)
    if shard_data.get("reparam_config_dict") is not None:
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        model_kwargs["reparam_config"] = ReparamConfig(
            **shard_data["reparam_config_dict"]
        )

    # Restore t_ref for reference-time reparameterization
    if shard_data.get("t_ref") is not None:
        model_kwargs["t_ref"] = shard_data["t_ref"]

    # Heartbeat thread to emit liveness updates back to the parent.
    # Optimization: Use Event.wait(timeout) instead of busy-wait loop.
    # This reduces wake-ups by 75% (from 4 per interval to 1).
    stop_hb = threading.Event()
    heartbeat_interval = 30.0

    def _heartbeat_loop() -> None:
        while True:
            # Wait for stop signal OR timeout (whichever comes first)
            # This is much more efficient than sleep + check loop
            if stop_hb.wait(timeout=heartbeat_interval):
                # Event was set - time to exit
                break
            # Timeout expired - send heartbeat
            payload = {
                "type": "heartbeat",
                "shard_idx": shard_idx,
                "elapsed": time.perf_counter() - start_time,
            }
            if result_queue is not None:
                try:
                    result_queue.put_nowait(payload)
                except Exception:  # noqa: S110 - Best-effort heartbeat
                    pass

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

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
            per_angle_mode=per_angle_mode,
        )

        duration = time.perf_counter() - start_time
        # T045: Log shard completion with elapsed time, acceptance rate, divergence count
        divergence_str = (
            f", divergences: {stats.num_divergent}" if stats.num_divergent > 0 else ""
        )
        worker_logger.info(
            f"Shard {shard_idx} completed in {duration:.2f}s: "
            f"{samples.n_samples} samples/chain × {samples.n_chains} chains{divergence_str}"
        )
        if stats.num_divergent > 0:
            worker_logger.warning(
                f"Shard {shard_idx} had {stats.num_divergent} divergent transitions"
            )

        # Serialize for return
        result = {
            "type": "result",
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
                "n_warmup": stats.plan.n_warmup if stats.plan else None,
                "n_samples": stats.plan.n_samples if stats.plan else None,
            },
        }
        # Result is returned to _run_shard_in_process, which puts it on the queue.
        # Do NOT put it here — that would cause double-queuing.
        return result

    except Exception as e:
        duration = time.perf_counter() - start_time
        # Classify error type for diagnostics
        error_str = str(e).lower()
        if "nan" in error_str or "inf" in error_str or "singular" in error_str:
            error_category = "numerical"
        elif "convergence" in error_str or "diverge" in error_str:
            error_category = "convergence"
        else:
            error_category = "sampling"

        # T028: Log exception with structured context for debugging
        log_exception(
            worker_logger,
            e,
            context={
                "shard_idx": shard_idx,
                "duration_s": round(duration, 2),
                "error_category": error_category,
                "n_points": n_points,
            },
        )

        result = {
            "type": "result",
            "success": False,
            "shard_idx": shard_idx,
            "error": str(e),
            "error_category": error_category,
            "duration": duration,
        }
        # Result is returned to _run_shard_in_process, which puts it on the queue.
        # Do NOT put it here — that would cause double-queuing.
        return result
    finally:
        stop_hb.set()
        hb_thread.join(timeout=1)


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
        # Estimate physical cores (half of logical threads for HT systems)
        logical_cpus = mp.cpu_count()
        physical_cores_estimate = max(1, logical_cpus // 2)

        if n_workers is None:
            # Default to estimated physical cores, leaving some headroom
            n_workers = max(1, physical_cores_estimate - 1)
        else:
            # Cap user-specified workers to avoid over-subscription
            n_workers = min(n_workers, physical_cores_estimate)

        self.n_workers = max(1, n_workers)
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
                per_angle_mode=model_kwargs.get("per_angle_mode", "individual"),
            )
            return samples

        # Multiple shards - run in parallel with per-shard timeout enforcement
        n_shards = len(shards)
        actual_workers = min(self.n_workers, n_shards)

        # Calculate threads per worker to avoid over-subscription
        total_threads = mp.cpu_count()
        threads_per_worker = _compute_threads_per_worker(total_threads, actual_workers)

        if threads_per_worker < max(1, total_threads // max(1, actual_workers)):
            run_logger.info(
                f"Capping threads to avoid oversubscription: logical={total_threads}, "
                f"workers={actual_workers} → {threads_per_worker} threads/worker"
            )

        run_logger.info(
            f"Running {n_shards} shards in parallel with {actual_workers} workers "
            f"({threads_per_worker} threads each)"
        )

        # Per-shard timeout - enforced per individual process
        per_shard_timeout = config.per_shard_timeout  # Default: 7200s (2 hours)
        run_logger.info(
            f"Per-shard timeout: {per_shard_timeout / 3600:.1f} hours "
            f"(processes will be terminated if exceeded)"
        )
        run_logger.info(
            f"Heartbeat timeout: {config.heartbeat_timeout}s "
            f"(unresponsive workers will be terminated)"
        )

        # Prepare shard data for workers
        # Separate per-shard data from shared data to reduce pickling overhead.
        # Shared data (config, parameter_space, time_grid, model kwargs) is placed
        # in shared memory once; only per-shard arrays are pickled per process.
        shared_kwargs = {
            "q": model_kwargs["q"],
            "L": model_kwargs["L"],
            "dt": model_kwargs["dt"],
            "fixed_contrast": model_kwargs.get("fixed_contrast"),
            "fixed_offset": model_kwargs.get("fixed_offset"),
            "per_angle_mode": model_kwargs.get("per_angle_mode", "individual"),
            "nlsq_prior_config": model_kwargs.get("nlsq_prior_config"),
            "num_shards": model_kwargs.get("num_shards", 1),
            "t_ref": model_kwargs.get("t_ref"),
            "reparam_config_dict": (
                {
                    "enable_d_ref": model_kwargs["reparam_config"].enable_d_ref,
                    "enable_gamma_ref": model_kwargs["reparam_config"].enable_gamma_ref,
                    "t_ref": model_kwargs["reparam_config"].t_ref,
                }
                if model_kwargs.get("reparam_config") is not None
                else None
            ),
        }

        shard_data_list = []
        for shard in shards:
            shard_data_list.append(
                {
                    "data": np.array(shard.data),
                    "t1": np.array(shard.t1),
                    "t2": np.array(shard.t2),
                    "phi_unique": np.array(shard.phi_unique),
                    "phi_indices": np.array(shard.phi_indices),
                    "noise_scale": shard.noise_scale,
                }
            )

        # Serialize config and parameter_space
        config_dict = config.to_dict()

        if hasattr(parameter_space, "_config_dict"):
            ps_dict = parameter_space._config_dict
        else:
            ps_dict = model_kwargs.get("config_dict", {})

        # Place shared data in shared memory to avoid per-shard pickling.
        # Wrap in try-except so partially-created blocks are cleaned up on failure.
        shared_mgr = SharedDataManager()
        try:
            shared_config_ref = shared_mgr.create_shared_dict("config", config_dict)
            shared_ps_ref = shared_mgr.create_shared_dict("ps", ps_dict)
            shared_kwargs_ref = shared_mgr.create_shared_dict("kwargs", shared_kwargs)

            # Share time_grid as array if present (can be large)
            time_grid_raw = model_kwargs.get("time_grid")
            if time_grid_raw is not None:
                shared_tg_ref: dict[str, Any] | None = shared_mgr.create_shared_array(
                    "time_grid", np.array(time_grid_raw)
                )
            else:
                shared_tg_ref = None

            # Share initial_values as dict
            shared_iv_ref: dict[str, Any] | None = None
            if initial_values is not None:
                shared_iv_ref = shared_mgr.create_shared_dict(
                    "init_vals", initial_values
                )
        except Exception:
            shared_mgr.cleanup()
            raise

        run_logger.debug(
            f"Shared memory allocated: {len(shared_mgr._shared_blocks)} blocks"
        )

        # Pre-generate all shard PRNG keys in single JAX call (batch optimization)
        # This amortizes JAX compilation overhead across all shards
        run_logger.debug(f"Pre-generating {n_shards} PRNG keys...")
        key_gen_start = time.time()
        shard_keys = _generate_shard_keys(n_shards, seed=42)
        key_gen_time = time.time() - key_gen_start
        run_logger.debug(f"PRNG key generation completed in {key_gen_time:.3f}s")

        # Use spawn context for clean process isolation
        ctx = mp.get_context(self.spawn_method)
        result_queue = ctx.Queue()

        # Temporarily adjust parent environment before spawning workers.
        # spawn'd children inherit the parent's env at Process.start() time.
        # configure_optimal_device() sets OMP_PROC_BIND=true and
        # OMP_NUM_THREADS=<physical_cores> for the parent process, but workers
        # must NOT inherit these — they cause massive thread oversubscription
        # (e.g. 9 workers × 14 OMP threads = 126 threads on 14 cores).
        _saved_env: dict[str, str | None] = {}
        _worker_env_overrides = {
            "OMP_NUM_THREADS": str(threads_per_worker),
            "MKL_NUM_THREADS": str(threads_per_worker),
            "OPENBLAS_NUM_THREADS": str(threads_per_worker),
            "VECLIB_MAXIMUM_THREADS": str(threads_per_worker),
        }
        _worker_env_clear = ["OMP_PROC_BIND", "OMP_PLACES"]

        for key in _worker_env_clear:
            _saved_env[key] = os.environ.pop(key, None)
        for key, val in _worker_env_overrides.items():
            _saved_env[key] = os.environ.get(key)
            os.environ[key] = val

        # Track active processes: {shard_idx: (process, start_time)}
        active_processes: dict[int, tuple[mp.Process, float]] = {}
        pending_shards = list(range(n_shards))
        results = []
        completed_count = 0
        recorded_shards: set[int] = set()
        last_heartbeat: dict[int, float] = {}

        # EARLY ABORT TRACKING (Jan 2026): Monitor failure rate for early termination
        # If too many shards fail early, abort to save compute time
        early_abort_threshold = 0.5  # Abort if >50% of first N shards fail
        early_abort_sample_size = min(10, n_shards)  # Check first 10 shards
        failure_categories: dict[str, int] = {
            "timeout": 0,
            "heartbeat_timeout": 0,
            "crash": 0,
            "numerical": 0,
            "convergence": 0,
            "sampling": 0,
            "unknown": 0,
        }
        success_count = 0
        early_abort_triggered = False

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
        # Adaptive polling: start with faster polling, slow down as shards run longer
        poll_interval_min = 0.5  # Fast polling during startup
        poll_interval_max = 5.0  # Slow polling during long-running shards
        poll_interval = poll_interval_min
        last_completion_time = start_time  # Track when last shard completed
        status_log_interval = 300.0  # parent status log every 5 minutes
        last_status_log = start_time

        try:
            while completed_count < n_shards:
                # Drain queue first to capture heartbeats and completed shards
                while True:
                    try:
                        message = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    except Exception as exc:
                        run_logger.warning(f"Queue read error: {exc}")
                        break

                    msg_type = message.get("type")
                    shard_idx = message.get("shard_idx")
                    if msg_type == "heartbeat" and shard_idx is not None:
                        last_heartbeat[shard_idx] = time.time()
                        continue

                    if msg_type == "result" or message.get("success") is not None:
                        results.append(message)
                        if shard_idx is not None:
                            recorded_shards.add(shard_idx)
                        completed_count += 1
                        pbar.update(1)
                        # Reset to fast polling on completion (adaptive polling)
                        last_completion_time = time.time()
                        poll_interval = poll_interval_min

                        # Track success/failure for early abort logic
                        if message.get("success"):
                            success_count += 1
                            pbar.set_postfix(
                                shard=message.get("shard_idx", "?"),
                                time=f"{message.get('duration', 0):.1f}s",
                            )
                        else:
                            # Track failure category
                            category = message.get("error_category", "unknown")
                            if category in failure_categories:
                                failure_categories[category] += 1
                            else:
                                failure_categories["unknown"] += 1
                            pbar.set_postfix(
                                shard=message.get("shard_idx", "?"),
                                status="failed",
                            )

                        # EARLY ABORT CHECK (Jan 2026): Abort if too many shards fail early
                        if (
                            not early_abort_triggered
                            and completed_count >= early_abort_sample_size
                            and completed_count <= early_abort_sample_size + 2
                        ):
                            total_failures = sum(failure_categories.values())
                            failure_rate = total_failures / completed_count
                            if failure_rate > early_abort_threshold:
                                early_abort_triggered = True
                                run_logger.error(
                                    f"EARLY ABORT: {failure_rate:.1%} failure rate in first "
                                    f"{completed_count} shards exceeds {early_abort_threshold:.0%} threshold.\n"
                                    f"Failure breakdown: {failure_categories}\n"
                                    f"Terminating remaining shards to save compute time."
                                )
                                # Clear pending shards and terminate active processes
                                pending_shards.clear()
                                for idx, (proc, _) in list(active_processes.items()):
                                    run_logger.info(
                                        f"Terminating shard {idx} due to early abort"
                                    )
                                    proc.terminate()
                                    proc.join(timeout=2)
                                    if proc.is_alive():
                                        proc.kill()
                                        proc.join(timeout=1)
                                    active_processes.pop(idx, None)

                        if shard_idx in active_processes:
                            # Clean up completed process tracking
                            proc, _ = active_processes.pop(shard_idx)
                            if proc.is_alive():
                                proc.join(timeout=1)
                        continue

                    run_logger.debug(f"Ignoring unexpected queue message: {message}")

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
                            shared_config_ref,
                            shared_iv_ref,
                            shared_ps_ref,
                            shared_kwargs_ref,
                            shared_tg_ref,
                            shards[shard_idx].n_phi,
                            analysis_mode,
                            threads_per_worker,
                            result_queue,
                            shard_keys[shard_idx],  # Pre-generated PRNG key
                        ),
                    )
                    process.start()
                    now = time.time()
                    active_processes[shard_idx] = (process, now)
                    last_heartbeat[shard_idx] = now
                    run_logger.debug(f"Started shard {shard_idx} (pid={process.pid})")

                # Check for completed or timed-out processes
                for shard_idx, (process, proc_start_time) in list(
                    active_processes.items()
                ):
                    now = time.time()
                    proc_elapsed = now - proc_start_time
                    last_active = last_heartbeat.get(shard_idx, proc_start_time)
                    inactive_elapsed = now - last_active

                    if not process.is_alive():
                        process.join(timeout=1)
                        del active_processes[shard_idx]
                        run_logger.debug(
                            f"Shard {shard_idx} process exited after {proc_elapsed:.1f}s"
                        )

                        if shard_idx not in recorded_shards:
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": shard_idx,
                                    "error": "Process exited without returning a result",
                                    "error_category": "crash",
                                    "duration": proc_elapsed,
                                }
                            )
                            recorded_shards.add(shard_idx)
                            failure_categories["crash"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=shard_idx, status="no-result")

                    elif proc_elapsed > per_shard_timeout:
                        # Total runtime exceeded - terminate regardless of heartbeats
                        run_logger.warning(
                            f"Shard {shard_idx} exceeded runtime limit: {proc_elapsed:.0f}s "
                            f"(limit: {per_shard_timeout}s), terminating process (pid={process.pid})"
                        )
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            run_logger.warning(
                                f"Shard {shard_idx} did not terminate, killing"
                            )
                            process.kill()
                            process.join(timeout=2)

                        del active_processes[shard_idx]
                        if shard_idx not in recorded_shards:
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": shard_idx,
                                    "error": f"Runtime timeout after {proc_elapsed:.0f}s (limit: {per_shard_timeout}s)",
                                    "error_category": "runtime_timeout",
                                    "duration": proc_elapsed,
                                }
                            )
                            recorded_shards.add(shard_idx)
                            failure_categories["timeout"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=shard_idx, status="timeout")

                    elif inactive_elapsed > config.heartbeat_timeout:
                        # No heartbeat for configured timeout - process likely frozen
                        run_logger.warning(
                            f"Shard {shard_idx} unresponsive for {inactive_elapsed:.0f}s "
                            f"(heartbeat timeout: {config.heartbeat_timeout}s), "
                            f"terminating process (pid={process.pid})"
                        )
                        process.terminate()
                        process.join(timeout=5)
                        if process.is_alive():
                            run_logger.warning(
                                f"Shard {shard_idx} did not terminate, killing"
                            )
                            process.kill()
                            process.join(timeout=2)

                        del active_processes[shard_idx]
                        if shard_idx not in recorded_shards:
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": shard_idx,
                                    "error": f"Unresponsive after {inactive_elapsed:.0f}s (heartbeat timeout: {config.heartbeat_timeout}s)",
                                    "error_category": "heartbeat_timeout",
                                    "duration": proc_elapsed,
                                }
                            )
                            recorded_shards.add(shard_idx)
                            failure_categories["heartbeat_timeout"] += 1
                            completed_count += 1
                            pbar.update(1)
                            pbar.set_postfix(shard=shard_idx, status="frozen")

                # Update progress bar with elapsed time
                if completed_count < n_shards:
                    elapsed = time.time() - start_time
                    mins, secs = divmod(int(elapsed), 60)
                    hrs, mins = divmod(mins, 60)
                    if hrs > 0:
                        pbar.set_postfix_str(
                            f"active={len(active_processes)} elapsed={hrs}h{mins:02d}m"
                        )
                    else:
                        pbar.set_postfix_str(
                            f"active={len(active_processes)} elapsed={mins}m{secs:02d}s"
                        )

                    if time.time() - last_status_log >= status_log_interval:
                        heartbeat_snapshot = {
                            k: f"{time.time() - v:.0f}s"
                            for k, v in last_heartbeat.items()
                        }
                        run_logger.info(
                            f"CMC status: {completed_count}/{n_shards} complete; "
                            f"active={len(active_processes)}; last_heartbeats={heartbeat_snapshot}"
                        )
                        last_status_log = time.time()

                    # Adaptive polling: gradually increase interval if no recent completions
                    # This reduces CPU overhead during long-running shards
                    time_since_completion = time.time() - last_completion_time
                    if time_since_completion > 30.0:
                        # Gradually increase poll interval (10% per 30s of inactivity)
                        poll_interval = min(
                            poll_interval * 1.1,
                            poll_interval_max,
                        )

                    time.sleep(poll_interval)

                # If no processes remain and nothing is pending, mark any missing shards as failed
                if (
                    not active_processes
                    and not pending_shards
                    and completed_count < n_shards
                ):
                    missing = set(range(n_shards)) - recorded_shards
                    for shard_idx in sorted(missing):
                        results.append(
                            {
                                "success": False,
                                "shard_idx": shard_idx,
                                "error": "Shard exited without emitting a result",
                                "error_category": "crash",
                                "duration": None,
                            }
                        )
                        recorded_shards.add(shard_idx)
                        completed_count += 1
                        pbar.update(1)
                        pbar.set_postfix(shard=shard_idx, status="no-result")

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
                    run_logger.warning(
                        f"Cleaning up orphan process for shard {shard_idx}"
                    )
                    process.terminate()
                    process.join(timeout=2)

            # Restore parent environment after all workers are done
            for key, val in _saved_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

            # Release shared memory after all workers are done
            shared_mgr.cleanup()

        # Process results - collect successful samples with metadata for filtering
        successful_samples = []
        shard_metadata: list[dict] = []  # Track shard idx, divergences, total samples
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
                # Track divergence stats for quality filtering
                stats = result.get("stats", {})
                total_samples = result["n_chains"] * result["n_samples"]
                shard_metadata.append(
                    {
                        "shard_idx": result.get("shard_idx"),
                        "num_divergent": stats.get("num_divergent", 0),
                        "total_samples": total_samples,
                        # NUTS divergent count <= total_samples; max() prevents div-by-zero
                        "divergence_rate": stats.get("num_divergent", 0)
                        / max(total_samples, 1),
                        "n_warmup": stats.get("n_warmup"),
                        "n_samples": stats.get("n_samples"),
                    }
                )
            else:
                error_cat = result.get("error_category", "unknown")
                run_logger.warning(
                    f"Shard {result.get('shard_idx', '?')} failed [{error_cat}]: "
                    f"{result.get('error', 'unknown')}"
                )

        if not successful_samples:
            # Aggregate error categories for better diagnostics
            error_categories: dict[str, int] = {}
            for result in results:
                if not result.get("success"):
                    category = result.get("error_category", "unknown")
                    error_categories[category] = error_categories.get(category, 0) + 1
            run_logger.error(
                f"All {n_shards} shards failed. Error breakdown: {error_categories}"
            )
            raise RuntimeError(
                f"All shards failed. Error categories: {error_categories}"
            )

        # Check success rate
        success_rate = len(successful_samples) / n_shards
        if success_rate < config.min_success_rate:
            # Critical: below minimum threshold
            run_logger.error(
                f"Success rate {success_rate:.1%} below minimum threshold "
                f"{config.min_success_rate:.1%} - analysis may be unreliable"
            )
        elif success_rate < config.min_success_rate_warning:
            # Warning: below warning threshold but above minimum
            run_logger.warning(
                f"Success rate {success_rate:.1%} below recommended threshold "
                f"{config.min_success_rate_warning:.1%} - consider investigating failed shards"
            )

        for shard_idx, duration in shard_timings:
            if shard_idx is not None and duration is not None:
                run_logger.debug(f"Shard {shard_idx} completed in {duration:.2f}s")

        # ──────────────────────────────────────────────────────────────────────
        # Jan 2026 FIX: Divergence-based shard quality filter
        # Filter out shards with divergence rate > max_divergence_rate
        # High-divergence shards have corrupted posteriors that bias the
        # consensus combination. This is especially critical for laminar_flow
        # where 28.4% overall divergence rate (from the C020 CMC run) indicated
        # severe sampling issues that propagated to parameter estimates.
        # ──────────────────────────────────────────────────────────────────────
        max_div_rate = getattr(config, "max_divergence_rate", 0.10)
        if max_div_rate < 1.0 and shard_metadata:
            # Identify high-divergence shards
            high_div_shards = []
            filtered_samples = []
            filtered_metadata = []

            for samples, meta in zip(successful_samples, shard_metadata, strict=True):
                div_rate = meta["divergence_rate"]
                if div_rate > max_div_rate:
                    high_div_shards.append(
                        (meta["shard_idx"], div_rate, meta["num_divergent"])
                    )
                else:
                    filtered_samples.append(samples)
                    filtered_metadata.append(meta)

            if high_div_shards:
                run_logger.warning(
                    f"QUALITY FILTER: Excluding {len(high_div_shards)} shards with "
                    f"divergence rate > {max_div_rate:.0%}:"
                )
                for shard_idx, div_rate, num_div in high_div_shards:
                    run_logger.warning(
                        f"  Shard {shard_idx}: {div_rate:.1%} divergence ({num_div} transitions)"
                    )

                # Update samples list for combination
                n_before = len(successful_samples)
                successful_samples = filtered_samples
                shard_metadata = filtered_metadata

                run_logger.info(
                    f"After quality filtering: {len(successful_samples)}/{n_before} shards retained"
                )

                # Re-check if we still have enough shards
                if not successful_samples:
                    raise RuntimeError(
                        f"All {n_before} successful shards exceeded max_divergence_rate={max_div_rate:.0%}. "
                        "Consider: (1) reducing shard size, (2) adjusting priors, "
                        "(3) increasing max_divergence_rate threshold."
                    )

                # Warn if filtered rate is too low
                filtered_rate = len(successful_samples) / n_shards
                if filtered_rate < config.min_success_rate:
                    run_logger.error(
                        f"Post-filter success rate {filtered_rate:.1%} below minimum threshold "
                        f"{config.min_success_rate:.1%} - analysis may be unreliable"
                    )

        # Log per-shard posterior statistics BEFORE combination
        # This helps diagnose why combined posteriors may differ from initial values
        # Also check for heterogeneity abort (Jan 2026 v2)
        heterogeneity_abort = getattr(config, "heterogeneity_abort", True)
        max_parameter_cv = getattr(config, "max_parameter_cv", 1.0)
        high_cv_params: list[tuple[str, float]] = []  # Track (param, cv) pairs

        if len(successful_samples) > 1:
            key_params = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta"]
            run_logger.info(
                f"Per-shard posterior statistics ({len(successful_samples)} shards):"
            )
            if parameter_space is None:
                run_logger.warning(
                    "parameter_space is None — bounds-aware CV disabled; "
                    "heterogeneity detection may produce false positives for near-zero parameters"
                )

            for param in key_params:
                if param in successful_samples[0].samples:
                    means = [
                        float(np.mean(s.samples[param])) for s in successful_samples
                    ]
                    run_logger.info(
                        f"  {param}: shard_means=[{np.min(means):.4g}, {np.max(means):.4g}], "
                        f"range={np.max(means) - np.min(means):.4g}, "
                        f"mean_of_means={np.mean(means):.4g}, "
                        f"std_of_means={np.std(means):.4g}"
                    )
                    # Check for high heterogeneity (bounds-aware CV)
                    mean_val = abs(np.mean(means))
                    if parameter_space is not None:
                        try:
                            lo, hi = parameter_space.get_bounds(param)
                            param_range = hi - lo
                            # Distinguish inverted bounds (lo > hi) from degenerate (lo == hi)
                            if param_range < 0:
                                # Inverted bounds: use absolute range
                                run_logger.warning(
                                    f"  {param}: inverted bounds [{lo}, {hi}], "
                                    f"using abs(range)={abs(param_range):.4g}"
                                )
                                param_range = abs(param_range)
                            elif param_range == 0:
                                # Degenerate bounds: fall back to mean-based scale
                                run_logger.warning(
                                    f"  {param}: degenerate bounds [{lo}, {hi}] "
                                    f"(range=0), falling back to mean-based scale"
                                )
                            # For near-zero params, use bounds range as scale reference
                            scale = (
                                max(mean_val, param_range * 0.01)
                                if param_range > 0
                                else max(mean_val, 1e-10)
                            )
                        except (KeyError, ValueError, TypeError):
                            scale = max(mean_val, 1e-10)
                    else:
                        scale = max(mean_val, 1e-10)
                    cv = np.std(means) / scale
                    if cv > max_parameter_cv:
                        high_cv_params.append((param, cv))
                        run_logger.warning(
                            f"    HIGH HETEROGENEITY: {param} has CV={cv:.2f} across shards! "
                            f"(threshold={max_parameter_cv:.2f})"
                        )
                    elif cv > 0.5:
                        run_logger.warning(
                            f"    MODERATE HETEROGENEITY: {param} has CV={cv:.2f} across shards. "
                            f"Combined posterior may be unreliable."
                        )

            # HETEROGENEITY ABORT (Jan 2026 v2): Fail fast instead of silently bad results
            if heterogeneity_abort and high_cv_params:
                param_summary = ", ".join(
                    f"{p} (CV={cv:.2f})" for p, cv in high_cv_params
                )
                raise RuntimeError(
                    f"HETEROGENEITY ABORT: {len(high_cv_params)} parameter(s) exceed "
                    f"max_parameter_cv={max_parameter_cv:.2f}: {param_summary}\n\n"
                    f"This indicates shards are sampling from inconsistent posterior regions, "
                    f"making consensus combination unreliable.\n\n"
                    f"Recommended actions:\n"
                    f"  1. Ensure NLSQ warm-start is active (--nlsq-result <path> or automatic)\n"
                    f"  2. Increase min_points_per_shard (current: {getattr(config, 'min_points_per_shard', 10000):,})\n"
                    f"  3. Check if data quality issues exist (outliers, missing values)\n"
                    f"  4. Set validation.heterogeneity_abort=false to disable this check (not recommended)\n"
                    f"  5. Increase max_parameter_cv threshold if heterogeneity is expected"
                )

        # Check for bimodal posteriors (per-shard) - Jan 2026
        # This helps detect local minima or model misspecification
        bimodal_alerts: list[
            tuple[int, str, float, float]
        ] = []  # (shard_idx, param, sep, weights)
        for i, shard_result in enumerate(successful_samples):
            bimodal_results = check_shard_bimodality(shard_result.samples)
            for param, result in bimodal_results.items():
                if result.is_bimodal:
                    bimodal_alerts.append(
                        (i, param, result.separation, min(result.weights))
                    )
                    run_logger.warning(
                        f"BIMODAL POSTERIOR: Shard {i}, {param}: "
                        f"modes at {result.means[0]:.4g} and {result.means[1]:.4g} "
                        f"(weights: {result.weights[0]:.2f}/{result.weights[1]:.2f})"
                    )

        if bimodal_alerts:
            run_logger.warning(
                f"Detected {len(bimodal_alerts)} bimodal posteriors across shards. "
                f"This may indicate model misspecification or local minima."
            )

        # Combine samples
        combined = combine_shard_samples(
            successful_samples,
            method=config.combination_method,
        )

        # Explicitly set num_shards to surviving shard count for diagnostics.
        # Without this, per-shard MCMCSamples reconstruction defaults num_shards=1,
        # and the hierarchical combination may not accumulate correctly.
        combined.num_shards = len(successful_samples)

        # Propagate median adapted n_warmup from shard metadata.
        # Workers may adapt warmup independently (e.g., 500→140 for small shards).
        # Use median to represent the typical adapted value for CMCResult reporting.
        warmup_values = [
            m["n_warmup"] for m in shard_metadata if m.get("n_warmup") is not None
        ]
        if warmup_values:
            combined.shard_adapted_n_warmup = int(np.median(warmup_values))
        else:
            run_logger.info(
                "No shards reported adapted n_warmup; CMCResult will use config default"
            )

        # Log summary including divergence filtering
        total_divergences = (
            sum(m.get("num_divergent", 0) for m in shard_metadata)
            if shard_metadata
            else 0
        )
        total_transitions = (
            sum(m.get("total_samples", 0) for m in shard_metadata)
            if shard_metadata
            else 0
        )
        overall_div_rate = total_divergences / max(total_transitions, 1)
        run_logger.info(
            f"Combined {len(successful_samples)}/{n_shards} shards "
            f"(overall divergence rate: {overall_div_rate:.1%}, {total_divergences}/{total_transitions})"
        )

        return combined

    def is_available(self) -> bool:
        """Check if multiprocessing is available."""
        return True
