"""Multiprocessing backend for CMC execution.

This module provides parallel MCMC execution using Python's
multiprocessing module for CPU-based parallelism.

Optimizations (v2.9.1):
- Batch PRNG key generation: Pre-generate all shard keys in single JAX call
- Adaptive polling: Adjust poll interval based on shard activity
- Event.wait heartbeat: Efficient heartbeat using Event.wait(timeout)

Optimizations (v2.22.2):
- LPT scheduling: Dispatch highest-cost shards first (size + noise weighted)
- Per-shard shared memory: Shard arrays stored in shared memory (avoids pickle overhead)
- deque for pending shards: O(1) popleft instead of O(n) list.pop(0)
- JIT cache fix: Enable persistent compilation cache via jax.config.update (env var alone insufficient in JAX 0.8+, min_compile_time lowered to 0)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import multiprocessing.shared_memory
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from homodyne.optimization.cmc.backends.base import (
    CMCBackend,
    combine_shard_samples,
    combine_shard_samples_bimodal,
)
from homodyne.optimization.cmc.diagnostics import (
    check_shard_bimodality,
    cluster_shard_modes,
    summarize_cross_shard_bimodality,
)
from homodyne.utils.logging import get_logger, log_exception, with_context

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)

# Keys for per-shard numpy arrays stored in shared memory.
# Used by SharedDataManager.create_shared_shard_arrays() and _load_shared_shard_data().
_SHARD_ARRAY_KEYS = ("data", "t1", "t2", "phi_unique", "phi_indices")


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

    def create_shared_shard_arrays(
        self, shard_data_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Place per-shard numpy arrays into shared memory (packed format).

        Instead of creating one SharedMemory segment per array per shard
        (n_shards * 5 = thousands of file descriptors), this concatenates
        all shard arrays for each key into a single shared memory block.
        Only 5 SharedMemory segments are created regardless of shard count.

        Parameters
        ----------
        shard_data_list : list[dict[str, Any]]
            List of shard data dicts, each containing numpy arrays
            (data, t1, t2, phi_unique, phi_indices) and a scalar noise_scale.

        Returns
        -------
        list[dict[str, Any]]
            List of lightweight shard references (shm names + offsets).
            Each ref dict is small enough to serialize cheaply through spawn.
        """
        n_shards = len(shard_data_list)

        # For each array key, concatenate all shards into one block
        key_meta: dict[str, dict[str, Any]] = {}
        for key in _SHARD_ARRAY_KEYS:
            arrays = []
            sizes = []
            for sd in shard_data_list:
                arr = sd[key]
                if not isinstance(arr, np.ndarray):
                    arr = np.asarray(arr)
                arr = np.ascontiguousarray(arr.ravel())
                arrays.append(arr)
                sizes.append(arr.shape[0])

            combined = np.concatenate(arrays)
            shm = mp.shared_memory.SharedMemory(
                create=True, size=max(1, combined.nbytes)
            )
            shared_arr = np.ndarray(
                combined.shape, dtype=combined.dtype, buffer=shm.buf
            )
            shared_arr[:] = combined
            self._shared_blocks.append(shm)

            # Compute per-shard offsets via cumulative sum
            offsets = [0]
            for s in sizes[:-1]:
                offsets.append(offsets[-1] + s)

            key_meta[key] = {
                "shm_name": shm.name,
                "dtype": str(combined.dtype),
                "offsets": offsets,
                "sizes": sizes,
            }

        # Build per-shard refs that workers can slice from the packed blocks
        shard_refs: list[dict[str, Any]] = []
        for i in range(n_shards):
            ref: dict[str, Any] = {"noise_scale": shard_data_list[i]["noise_scale"]}
            for key in _SHARD_ARRAY_KEYS:
                meta = key_meta[key]
                ref[key] = {
                    "shm_name": meta["shm_name"],
                    "dtype": meta["dtype"],
                    "offset": meta["offsets"][i],
                    "size": meta["sizes"][i],
                }
            shard_refs.append(ref)

        return shard_refs

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

    return _pkl.loads(_load_shared_bytes(ref))  # noqa: S301  # nosec B301


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


def _load_shared_shard_data(shard_ref: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct per-shard data arrays from packed shared memory.

    Each array key maps to a single concatenated SharedMemory block shared
    across all shards.  The per-shard ref carries ``offset`` (element index)
    and ``size`` (element count) to slice this shard's portion.

    Parameters
    ----------
    shard_ref : dict[str, Any]
        Lightweight shard reference created by
        ``SharedDataManager.create_shared_shard_arrays``.

    Returns
    -------
    dict[str, Any]
        Shard data dict with numpy arrays (copied from shared memory)
        and scalar noise_scale.
    """
    shard_data: dict[str, Any] = {"noise_scale": shard_ref["noise_scale"]}

    for key in _SHARD_ARRAY_KEYS:
        arr_ref = shard_ref[key]
        shm = mp.shared_memory.SharedMemory(name=arr_ref["shm_name"], create=False)
        try:
            dtype = np.dtype(arr_ref["dtype"])
            offset = arr_ref["offset"]
            size = arr_ref["size"]
            # Map the full concatenated buffer, then slice this shard's region
            total_elements = len(shm.buf) // dtype.itemsize
            full_arr = np.ndarray((total_elements,), dtype=dtype, buffer=shm.buf)
            arr = full_arr[offset : offset + size].copy()
        finally:
            shm.close()
        shard_data[key] = arr

    return shard_data


def _generate_shard_keys(n_shards: int, seed: int = 42) -> list[tuple[int, ...]]:
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

    # Convert to serializable format (tuples of ints).
    # JAX ≤0.4.30 uses uint32[2]; JAX 0.4.31+ uses typed keys (key<fry>[]).
    # Flatten to raw uint32 array to handle both formats.
    key_tuples = []
    for key in shard_keys:
        raw = jax.random.key_data(key).flatten().astype(jnp.uint32)
        key_tuples.append(tuple(int(x) for x in raw))

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


def _compute_lpt_schedule(
    shard_data_list: list[dict[str, Any]],
) -> deque[int]:
    """Order shard indices by descending estimated cost (LPT heuristic).

    Cost = n_points * (1 + normalized_noise), where noise is linearly
    scaled to [0, 1] across shards.  Dispatching the most expensive
    shards first minimizes tail latency on identical parallel workers.

    Parameters
    ----------
    shard_data_list : list[dict[str, Any]]
        Shard dicts with ``"data"`` (array) and ``"noise_scale"`` (float).

    Returns
    -------
    deque[int]
        Shard indices sorted by descending cost.
    """
    n_shards = len(shard_data_list)
    sizes = [len(shard_data_list[i]["data"]) for i in range(n_shards)]
    noises = [shard_data_list[i]["noise_scale"] for i in range(n_shards)]

    max_noise = max(noises) if noises else 1.0
    min_noise = min(noises) if noises else 1.0
    noise_range = max_noise - min_noise

    if noise_range > 0:
        costs = [
            sizes[i] * (1.0 + (noises[i] - min_noise) / noise_range)
            for i in range(n_shards)
        ]
    else:
        costs = [float(s) for s in sizes]

    return deque(sorted(range(n_shards), key=lambda i: costs[i], reverse=True))


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
    shard_ref: dict[str, Any],
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
    rng_key_tuple: tuple[int, ...] | None = None,
) -> None:
    """Worker function that puts result in a queue for proper timeout handling.

    Accepts shared memory references instead of full dicts to avoid redundant
    pickling. Reconstructs shared data from shared memory blocks.

    Wraps all initialization and sampling in a top-level try/except to ensure
    that crashes during setup (imports, config reconstruction, model_kwargs)
    are captured and reported back to the parent via the result queue.
    """
    try:
        # Reconstruct per-shard arrays from shared memory (avoids pickle overhead)
        shard_data = _load_shared_shard_data(shard_ref)

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
        time_grid = (
            _load_shared_array(time_grid_ref) if time_grid_ref is not None else None
        )

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
    except Exception as e:
        # Catch crashes during initialization (shared memory, imports, config
        # reconstruction) that occur before _run_shard_worker's internal try/except.
        import traceback

        result = {
            "type": "result",
            "success": False,
            "shard_idx": shard_idx,
            "error": f"Worker initialization failed: {e}",
            "error_category": "init_crash",
            "traceback": traceback.format_exc(),
            "duration": 0.0,
        }

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
    rng_key_tuple: tuple[int, ...] | None = None,
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
    rng_key_tuple : tuple[int, ...] | None
        Pre-generated PRNG key as raw uint32 tuple. If None, generates
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

    # P0-1: Enable float64 precision and XLA device count BEFORE importing JAX.
    # Spawned workers don't inherit the parent's jax.config.x64_enabled state.
    # Without this, all JAX ops in workers run in float32, silently losing
    # precision for parameters spanning 6+ orders of magnitude (D0~1e4, gamma~1e-3).
    # P1-5: Ensure XLA_FLAGS propagates to spawned workers regardless of spawn
    # method (fork vs spawn). Parent sets JAX_ENABLE_X64 in homodyne/__init__.py
    # but spawn-mode workers start fresh processes and must re-set it.
    os.environ["JAX_ENABLE_X64"] = "true"
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(
            Path(os.path.expanduser("~/.cache/homodyne/jax_cache"))
        )
    # Unconditionally ensure device_count=4, stripping any stale value first.
    import re as _re

    _xla_flags = os.environ.get("XLA_FLAGS", "")
    _xla_flags = _re.sub(r"--xla_force_host_platform_device_count=\d+", "", _xla_flags)
    os.environ["XLA_FLAGS"] = (
        _xla_flags.strip() + " --xla_force_host_platform_device_count=4"
    )

    import jax

    jax.config.update("jax_enable_x64", True)

    # C3: Enable persistent compilation cache so subsequent workers reuse
    # compiled XLA programs from the first worker.  The env var alone is
    # insufficient in JAX 0.8+ — we must also call jax.config.update().
    # Additionally, CMC functions compile in 0.07-0.15s, below the default
    # 1.0s min_compile_time threshold, so we lower it to 0.
    _cache_dir = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        str(Path(os.path.expanduser("~/.cache/homodyne/jax_cache"))),
    )
    jax.config.update("jax_compilation_cache_dir", _cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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
        # Reconstruct JAX PRNG key from raw uint32 data (handles both
        # legacy uint32[2] and typed-key formats via key_data round-trip)
        rng_key = jax.random.wrap_key_data(jnp.array(rng_key_tuple, dtype=jnp.uint32))
    else:
        # Legacy behavior: generate key based on shard index
        rng_key = jax.random.PRNGKey(42 + shard_idx)

    # Prepare model kwargs - must match xpcs_model() signature
    # jnp.asarray avoids a copy when the source is already a contiguous ndarray.
    model_kwargs = {
        "data": jnp.asarray(shard_data["data"]),
        "t1": jnp.asarray(shard_data["t1"]),
        "t2": jnp.asarray(shard_data["t2"]),
        "phi_unique": jnp.asarray(shard_data["phi_unique"]),
        "phi_indices": jnp.asarray(shard_data["phi_indices"]),
        "q": shard_data["q"],
        "L": shard_data["L"],
        "dt": shard_data["dt"],
        "time_grid": (
            jnp.asarray(shard_data["time_grid"])
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

    # M1-worker: Free the numpy shard_data dict now that all values have been
    # extracted into model_kwargs (as JAX arrays) or as scalars.  This releases
    # the numpy copies of data/t1/t2/phi_indices/time_grid, which otherwise
    # stay alive for the entire sampling duration alongside their JAX twins.
    del shard_data

    # P0-1: Pre-compute scaling factors ONCE before NUTS starts.
    from homodyne.optimization.cmc.scaling import compute_scaling_factors

    model_kwargs["scalings"] = compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )

    # P0-2: Pre-compute physics prefactors (constant for entire shard).
    import math as _math

    _q = model_kwargs["q"]
    _L = model_kwargs["L"]
    _dt = model_kwargs["dt"]
    model_kwargs["wavevector_q_squared_half_dt"] = jnp.asarray(0.5 * (_q**2) * _dt)
    model_kwargs["sinc_prefactor"] = jnp.asarray(0.5 / _math.pi * _q * _L * _dt)

    # P1-3: Pre-compute point_idx array (constant for entire shard).
    model_kwargs["point_idx"] = jnp.arange(
        model_kwargs["phi_indices"].shape[0], dtype=jnp.int32
    )

    # D2: Pre-compute shard-constant quantities (time_safe + searchsorted indices)
    # once before NUTS starts.  Eliminates redundant work on every leapfrog step.
    try:
        from homodyne.core.physics_cmc import precompute_shard_grid

        _t1 = model_kwargs["t1"]
        _t2 = model_kwargs["t2"]
        _time_grid = model_kwargs.get("time_grid")
        _dt = model_kwargs.get("dt", 1e-3)
        if _time_grid is not None:
            model_kwargs["shard_grid"] = precompute_shard_grid(
                _time_grid, _t1, _t2, _dt
            )
    except (ImportError, ValueError, RuntimeError) as _exc:
        # Non-fatal: fall back to legacy compute_g1_total path in model.py
        worker_logger.warning(
            f"precompute_shard_grid failed (using legacy path): "
            f"{type(_exc).__name__}: {_exc}"
        )

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

        # M1-worker: Free shard input arrays now that sampling is done.
        # model_kwargs holds data/t1/t2/phi_indices/time_grid/shard_grid as
        # JAX arrays that are no longer needed.  Free them before serializing
        # the result, so peak memory during serialization is lower.
        model_kwargs.clear()

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


def _log_bimodality_summary(
    run_logger: Any,
    summary: dict[str, Any],
) -> None:
    """Log a structured cross-shard bimodality analysis.

    Parameters
    ----------
    run_logger
        Logger instance (supports .info()).
    summary : dict[str, Any]
        Output from summarize_cross_shard_bimodality().
    """
    per_param = summary.get("per_param", {})
    co_occurrence = summary.get("co_occurrence", {})
    n_detections = summary.get("n_detections", 0)
    n_shards = summary.get("n_shards", 0)

    if not per_param:
        run_logger.warning(
            f"Detected {n_detections} bimodal posteriors across shards, "
            f"but none exceeded the significance threshold."
        )
        return

    sep = "=" * 80
    dash = "-" * 80
    lines = [
        sep,
        f"BIMODALITY ANALYSIS ({n_detections} detections across {n_shards} shards)",
        sep,
        f"{'Parameter':<14} {'Bimodal%':>8}  {'Mode 1 (mean +/- std)':>24}  "
        f"{'Mode 2 (mean +/- std)':>24}  {'Sep.':>5}",
        dash,
    ]

    for param, stats in sorted(per_param.items()):
        pct = f"{stats['bimodal_fraction']:.1%}"
        m1 = f"{stats['lower_mean']:.3g} +/- {stats['lower_std']:.2g}"
        m2 = f"{stats['upper_mean']:.3g} +/- {stats['upper_std']:.2g}"
        sig = f"{stats['sep_significance']:.1f}x"
        lines.append(f"{param:<14} {pct:>8}  {m1:>24}  {m2:>24}  {sig:>5}")

    lines.append(dash)

    # Consensus impact section
    impact_lines: list[str] = []
    for param, stats in per_param.items():
        if stats["consensus_in_trough"]:
            impact_lines.append(
                f"  {param} consensus mean falls between modes (density trough)"
            )

    d0_alpha_frac = co_occurrence.get("d0_alpha_fraction")
    if d0_alpha_frac is not None and "D0" in per_param:
        impact_lines.append(
            f"  D0-alpha co-occurrence: {d0_alpha_frac:.0%} of D0-bimodal "
            f"shards also bimodal in alpha"
        )
        if d0_alpha_frac > 0.3:
            impact_lines.append(
                "  -> Likely parameter degeneracy: different (D0, alpha) "
                "pairs produce similar D(t)"
            )

    if impact_lines:
        lines.append("CONSENSUS IMPACT:")
        lines.extend(impact_lines)

    lines.append("GUIDANCE:")
    lines.append("  - NLSQ result likely converged to one mode; CMC captures both")
    lines.append(
        "  - Consider increasing shard size or using tighter NLSQ-informed priors"
    )
    lines.append(sep)

    for line in lines:
        run_logger.info(line)


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
            "global_phi_unique": model_kwargs.get("global_phi_unique"),
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
                    # np.asarray avoids a copy when shard arrays are already ndarrays.
                    "data": np.asarray(shard.data),
                    "t1": np.asarray(shard.t1),
                    "t2": np.asarray(shard.t2),
                    "phi_unique": np.asarray(shard.phi_unique),
                    "phi_indices": np.asarray(shard.phi_indices),
                    "noise_scale": shard.noise_scale,
                }
            )

        # shard_data_list is kept for LPT scheduling (size lookup).
        # Actual worker data will be served from shared memory (see below).

        # Serialize config and parameter_space
        config_dict = config.to_dict()

        if hasattr(parameter_space, "_config_dict"):
            ps_dict = parameter_space._config_dict
        else:
            ps_dict = model_kwargs.get("config_dict", {})
            if not ps_dict:
                run_logger.error(
                    "ParameterSpace._config_dict is absent and no 'config_dict' in "
                    "model_kwargs. Workers will reconstruct ParameterSpace from an "
                    "empty dict (default bounds). This may produce unconstrained or "
                    "incorrect NUTS proposals. Ensure ParameterSpace exposes "
                    "_config_dict or pass config_dict in model_kwargs."
                )

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

            # Place per-shard arrays in shared memory to avoid per-process
            # serialization overhead through spawn.  Each shard's 5 arrays
            # (data, t1, t2, phi_unique, phi_indices) are stored once;
            # workers reconstruct them via _load_shared_shard_data().
            shared_shard_refs = shared_mgr.create_shared_shard_arrays(shard_data_list)
        except Exception:
            shared_mgr.cleanup()
            raise

        # Sentinel variables for the finally block — must be defined before
        # try so that cleanup never hits NameError on early exceptions.
        _saved_env: dict[str, str | None] = {}
        active_processes: dict[int, tuple[mp.Process, float]] = {}
        pbar = None

        # All setup from here through the main loop is wrapped in try/finally
        # to ensure shared_mgr.cleanup() runs even if _generate_shard_keys(),
        # ctx.Queue(), or any other pre-loop setup raises.
        try:
            run_logger.debug(
                f"Shared memory allocated: {len(shared_mgr._shared_blocks)} blocks"
            )

            # Pre-generate all shard PRNG keys in single JAX call (batch optimization)
            # This amortizes JAX compilation overhead across all shards
            run_logger.debug(f"Pre-generating {n_shards} PRNG keys...")
            key_gen_start = time.time()
            shard_keys = _generate_shard_keys(n_shards, seed=config.seed)
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

            # LPT scheduling: dispatch highest-cost shards first to minimize
            # tail latency.  See _compute_lpt_schedule() for cost model.
            pending_shards = _compute_lpt_schedule(shard_data_list)
            if n_shards > 1:
                sizes = [len(sd["data"]) for sd in shard_data_list]
                noises = [sd["noise_scale"] for sd in shard_data_list]
                run_logger.debug(
                    f"LPT scheduling: shard sizes range "
                    f"[{min(sizes):,}, {max(sizes):,}], "
                    f"noise range [{min(noises):.4g}, {max(noises):.4g}], "
                    f"dispatching highest-cost first"
                )
            # M1-parent: Free per-shard numpy arrays now that they have been
            # copied into shared memory (via create_shared_shard_arrays above).
            del shard_data_list
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
            shards_launched = 0
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
                        # Guard against duplicate results from timed-out shards
                        # whose results arrive late via the queue.
                        if shard_idx is not None and shard_idx in recorded_shards:
                            run_logger.debug(
                                f"Ignoring duplicate result for shard {shard_idx}"
                            )
                            continue
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

                    if run_logger.isEnabledFor(logging.DEBUG):
                        run_logger.debug(
                            f"Ignoring unexpected queue message: {message}"
                        )

                # Launch new processes up to max workers
                while len(active_processes) < actual_workers and pending_shards:
                    shard_idx = pending_shards.popleft()

                    process = ctx.Process(
                        target=_run_shard_worker_with_queue,
                        args=(
                            shard_idx,
                            shared_shard_refs[shard_idx],
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
                    shards_launched += 1

                # Check for completed or timed-out processes
                for shard_idx, (process, proc_start_time) in list(
                    active_processes.items()
                ):
                    # Skip shards already recorded by queue drain (prevents
                    # double-counting when queue result arrives in the same
                    # loop iteration as the process exit detection).
                    if shard_idx in recorded_shards:
                        del active_processes[shard_idx]
                        continue

                    now = time.time()
                    proc_elapsed = now - proc_start_time
                    last_active = last_heartbeat.get(shard_idx, proc_start_time)
                    inactive_elapsed = now - last_active

                    if not process.is_alive():
                        process.join(timeout=1)
                        exit_code = process.exitcode
                        del active_processes[shard_idx]
                        run_logger.debug(
                            f"Shard {shard_idx} process exited after {proc_elapsed:.1f}s "
                            f"(exit_code={exit_code})"
                        )

                        if shard_idx not in recorded_shards:
                            # Build descriptive error with exit code context
                            if exit_code is not None and exit_code < 0:
                                import signal as _signal

                                sig_name = _signal.Signals(-exit_code).name
                                error_msg = (
                                    f"Process killed by signal {sig_name} "
                                    f"(exit_code={exit_code})"
                                )
                            elif exit_code is not None and exit_code > 0:
                                error_msg = (
                                    f"Process exited with error (exit_code={exit_code})"
                                )
                            else:
                                error_msg = "Process exited without returning a result"
                            results.append(
                                {
                                    "type": "result",
                                    "success": False,
                                    "shard_idx": shard_idx,
                                    "error": error_msg,
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
                                    "error_category": "timeout",
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

                    _now = time.time()
                    if _now - last_status_log >= status_log_interval:
                        # Only show heartbeats for active processes
                        active_heartbeats = {
                            k: f"{_now - last_heartbeat.get(k, _now):.0f}s"
                            for k in active_processes
                        }
                        run_logger.info(
                            f"CMC status: {completed_count}/{n_shards} complete; "
                            f"active={len(active_processes)}; "
                            f"launched={shards_launched}; "
                            f"heartbeats={active_heartbeats}"
                        )
                        last_status_log = _now

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
            if pbar is not None:
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
                if result.get("traceback"):
                    run_logger.debug(
                        f"Shard {result.get('shard_idx', '?')} traceback:\n"
                        f"{result['traceback']}"
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

        # Check success rate — warn first, then error for worse.
        # P2-A: Previously, warning (0.80) < min (0.90) made the elif unreachable.
        # Fixed: check warning threshold first (higher), then error threshold (lower).
        success_rate = len(successful_samples) / n_shards
        if success_rate < config.min_success_rate_warning:
            # Critical: below warning threshold (worst case)
            run_logger.error(
                f"Success rate {success_rate:.1%} below minimum threshold "
                f"{config.min_success_rate_warning:.1%} - analysis may be unreliable"
            )
        elif success_rate < config.min_success_rate:
            # Degraded: between warning and recommended thresholds
            run_logger.warning(
                f"Success rate {success_rate:.1%} below recommended threshold "
                f"{config.min_success_rate:.1%} - consider investigating failed shards"
            )

        valid_durations = [d for _, d in shard_timings if d is not None]
        if valid_durations:
            run_logger.debug(
                f"Shard timing summary: n={len(valid_durations)}, "
                f"min={min(valid_durations):.1f}s, max={max(valid_durations):.1f}s, "
                f"median={sorted(valid_durations)[len(valid_durations) // 2]:.1f}s"
            )

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
                    if not np.isfinite(cv):
                        # NaN/Inf CV means shard posteriors contain non-finite values
                        high_cv_params.append((param, float("inf")))
                        run_logger.warning(
                            f"    NON-FINITE CV: {param} has nan/inf CV "
                            f"(likely NaN samples in shard posteriors)"
                        )
                    elif cv > max_parameter_cv:
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
        bimodal_detections: list[dict[str, Any]] = []
        for i, shard_result in enumerate(successful_samples):
            bimodal_results = check_shard_bimodality(shard_result.samples)
            for param, result in bimodal_results.items():
                if result.is_bimodal:
                    bimodal_detections.append(
                        {
                            "shard": i,
                            "param": param,
                            "mode1": result.means[0],
                            "mode2": result.means[1],
                            "std1": result.stds[0],
                            "std2": result.stds[1],
                            "weights": result.weights,
                            "separation": result.separation,
                        }
                    )
                    run_logger.warning(
                        f"BIMODAL POSTERIOR: Shard {i}, {param}: "
                        f"modes at {result.means[0]:.4g} and {result.means[1]:.4g} "
                        f"(weights: {result.weights[0]:.2f}/{result.weights[1]:.2f})"
                    )

        if bimodal_detections:
            # Compute pre-combine consensus means from per-shard posteriors
            consensus_means: dict[str, float] = {}
            key_params = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta"]
            for param in key_params:
                if param in successful_samples[0].samples:
                    means = [
                        float(np.mean(s.samples[param])) for s in successful_samples
                    ]
                    consensus_means[param] = float(np.mean(means))

            bimodal_summary = summarize_cross_shard_bimodality(
                bimodal_detections,
                n_shards=len(successful_samples),
                consensus_means=consensus_means,
            )
            _log_bimodality_summary(run_logger, bimodal_summary)

            # Mode-aware consensus if significant bimodality detected
            if bimodal_summary["per_param"]:
                modal_params = sorted(bimodal_summary["per_param"].keys())
                # Get parameter bounds for range normalization
                param_bounds: dict[str, tuple[float, float]] = {}
                if parameter_space is not None:
                    for param in modal_params:
                        try:
                            param_bounds[param] = parameter_space.get_bounds(param)
                        except (KeyError, ValueError):
                            pass

                mode_assignments = cluster_shard_modes(
                    bimodal_detections=bimodal_detections,
                    successful_samples=successful_samples,
                    bimodal_summary=bimodal_summary,
                    param_bounds=param_bounds,
                )

                run_logger.info(
                    f"Mode-aware consensus: cluster sizes = "
                    f"{len(mode_assignments[0])}, {len(mode_assignments[1])}"
                )

                combined, bimodal_result = combine_shard_samples_bimodal(
                    shard_samples=successful_samples,
                    cluster_assignments=mode_assignments,
                    bimodal_detections=bimodal_detections,
                    modal_params=modal_params,
                    co_occurrence=bimodal_summary.get("co_occurrence", {}),
                    method=config.combination_method,
                )
                combined.bimodal_consensus = bimodal_result

                # Log mode summary
                for i, mode in enumerate(bimodal_result.modes):
                    mode_means = ", ".join(
                        f"{p}={mode.mean[p]:.4g}"
                        for p in modal_params
                        if p in mode.mean
                    )
                    run_logger.info(
                        f"  Mode {i}: weight={mode.weight:.2f}, "
                        f"n_shards={mode.n_shards}, {mode_means}"
                    )
            else:
                # Bimodal detections exist but below significance threshold
                combined = combine_shard_samples(
                    successful_samples,
                    method=config.combination_method,
                )
        else:
            # No bimodality detected — standard path
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
