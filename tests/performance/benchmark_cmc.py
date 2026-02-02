"""
CMC (Consensus Monte Carlo) Performance Benchmark Suite.

Enterprise-level profiling for MCMC sampling performance:
1. NUTS sampling throughput (samples/sec)
2. Shard worker timing breakdown
3. Memory usage per shard and total
4. Multiprocessing overhead analysis
5. Scaling behavior across shard counts

Usage:
    # Run quick profiling
    python -m tests.performance.benchmark_cmc --mode quick

    # Run standard profiling with flamegraph
    python -m tests.performance.benchmark_cmc --mode standard --flamegraph

    # Run enterprise audit
    python -m tests.performance.benchmark_cmc --mode enterprise

    # Run via pytest
    pytest tests/performance/benchmark_cmc.py -v --benchmark-only

Author: Homodyne Performance Team
Date: 2026-02-01
"""

from __future__ import annotations

import cProfile
import gc
import io
import json
import logging
import multiprocessing as mp
import os
import pstats
import tempfile
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Results
# ============================================================================


@dataclass
class TimingBreakdown:
    """Detailed timing breakdown for CMC phases."""

    # Phase timings (seconds)
    data_prep_s: float = 0.0
    shard_creation_s: float = 0.0
    model_compilation_s: float = 0.0
    warmup_sampling_s: float = 0.0
    production_sampling_s: float = 0.0
    consensus_aggregation_s: float = 0.0
    diagnostics_s: float = 0.0
    total_s: float = 0.0

    # Per-sample metrics
    samples_per_second: float = 0.0
    time_per_sample_ms: float = 0.0

    # Overhead analysis
    overhead_fraction: float = 0.0  # Non-sampling time / total

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "data_prep_s": self.data_prep_s,
            "shard_creation_s": self.shard_creation_s,
            "model_compilation_s": self.model_compilation_s,
            "warmup_sampling_s": self.warmup_sampling_s,
            "production_sampling_s": self.production_sampling_s,
            "consensus_aggregation_s": self.consensus_aggregation_s,
            "diagnostics_s": self.diagnostics_s,
            "total_s": self.total_s,
            "samples_per_second": self.samples_per_second,
            "time_per_sample_ms": self.time_per_sample_ms,
            "overhead_fraction": self.overhead_fraction,
        }


@dataclass
class MemoryProfile:
    """Memory usage profile."""

    baseline_mb: float = 0.0
    peak_mb: float = 0.0
    delta_mb: float = 0.0

    # Per-component breakdown
    data_arrays_mb: float = 0.0
    jax_buffers_mb: float = 0.0
    mcmc_samples_mb: float = 0.0
    worker_overhead_mb: float = 0.0

    # Allocation tracking
    n_allocations: int = 0
    top_allocations: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "baseline_mb": self.baseline_mb,
            "peak_mb": self.peak_mb,
            "delta_mb": self.delta_mb,
            "data_arrays_mb": self.data_arrays_mb,
            "jax_buffers_mb": self.jax_buffers_mb,
            "mcmc_samples_mb": self.mcmc_samples_mb,
            "worker_overhead_mb": self.worker_overhead_mb,
            "n_allocations": self.n_allocations,
            "top_allocations": self.top_allocations,
        }


@dataclass
class CPUProfile:
    """CPU profiling results."""

    # Hot functions (name, cumulative_time_s, call_count)
    hot_functions: list[tuple[str, float, int]] = field(default_factory=list)

    # Time in different modules
    time_by_module: dict[str, float] = field(default_factory=dict)

    # IPC and cache metrics (if available)
    instructions_per_cycle: float = 0.0
    cache_miss_rate: float = 0.0
    branch_misprediction_rate: float = 0.0


@dataclass
class CMCBenchmarkResult:
    """Complete CMC benchmark result."""

    # Dataset info
    n_points: int = 0
    n_shards: int = 0
    n_phi: int = 0
    analysis_mode: str = "static"
    shard_size: int = 0

    # MCMC config
    n_warmup: int = 0
    n_samples: int = 0
    n_chains: int = 0

    # Timing
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)

    # Memory
    memory: MemoryProfile = field(default_factory=MemoryProfile)

    # CPU profiling
    cpu_profile: CPUProfile = field(default_factory=CPUProfile)

    # Multiprocessing metrics
    n_workers: int = 0
    worker_utilization: float = 0.0
    ipc_overhead_s: float = 0.0

    # Convergence metrics
    mean_rhat: float = 0.0
    min_ess: float = 0.0
    divergence_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "n_points": self.n_points,
            "n_shards": self.n_shards,
            "n_phi": self.n_phi,
            "analysis_mode": self.analysis_mode,
            "shard_size": self.shard_size,
            "n_warmup": self.n_warmup,
            "n_samples": self.n_samples,
            "n_chains": self.n_chains,
            "timing": self.timing.to_dict(),
            "memory": self.memory.to_dict(),
            "n_workers": self.n_workers,
            "worker_utilization": self.worker_utilization,
            "ipc_overhead_s": self.ipc_overhead_s,
            "mean_rhat": self.mean_rhat,
            "min_ess": self.min_ess,
            "divergence_rate": self.divergence_rate,
        }


# ============================================================================
# Profiling Utilities
# ============================================================================


def get_memory_mb() -> float:
    """Get current process memory in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024**2)
    return 0.0


class MemoryTracker:
    """Context manager for tracking memory usage with tracemalloc."""

    def __init__(self, track_allocations: bool = True):
        self.track_allocations = track_allocations
        self.start_memory = 0.0
        self.peak_memory = 0.0
        self.allocations: list[tuple[str, int]] = []

    def __enter__(self) -> "MemoryTracker":
        gc.collect()
        self.start_memory = get_memory_mb()
        if self.track_allocations:
            tracemalloc.start()
        return self

    def __exit__(self, *args: Any) -> None:
        gc.collect()
        self.peak_memory = get_memory_mb()
        if self.track_allocations:
            snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()
            # Get top allocations
            stats = snapshot.statistics("lineno")[:20]
            self.allocations = [(str(stat.traceback), stat.size) for stat in stats]

    def get_profile(self) -> MemoryProfile:
        """Get memory profile."""
        return MemoryProfile(
            baseline_mb=self.start_memory,
            peak_mb=self.peak_memory,
            delta_mb=self.peak_memory - self.start_memory,
            n_allocations=len(self.allocations),
            top_allocations=self.allocations[:10],
        )


class CPUProfiler:
    """Context manager for CPU profiling with cProfile."""

    def __init__(self):
        self.profiler = cProfile.Profile()
        self.stats: pstats.Stats | None = None

    def __enter__(self) -> "CPUProfiler":
        self.profiler.enable()
        return self

    def __exit__(self, *args: Any) -> None:
        self.profiler.disable()
        stream = io.StringIO()
        self.stats = pstats.Stats(self.profiler, stream=stream)

    def get_hot_functions(self, n: int = 20) -> list[tuple[str, float, int]]:
        """Get top N functions by cumulative time."""
        if self.stats is None:
            return []

        self.stats.sort_stats("cumulative")
        results = []
        for (filename, line, func), (cc, nc, tt, ct, callers) in list(
            self.stats.stats.items()
        )[:n]:
            func_name = f"{filename}:{line}:{func}"
            results.append((func_name, ct, nc))
        return results

    def get_time_by_module(self) -> dict[str, float]:
        """Get time spent in each module."""
        if self.stats is None:
            return {}

        module_times: dict[str, float] = {}
        for (filename, _line, _func), (_cc, _nc, _tt, ct, _callers) in self.stats.stats.items():
            # Extract module from filename
            if "/" in filename:
                module = filename.split("/")[-2] if "/" in filename else filename
            else:
                module = "builtin"
            module_times[module] = module_times.get(module, 0.0) + ct
        return dict(sorted(module_times.items(), key=lambda x: -x[1])[:10])


class PhaseTimer:
    """Timer for tracking individual phases."""

    def __init__(self):
        self.phases: dict[str, float] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0.0

    def start(self, phase: str) -> None:
        """Start timing a phase."""
        if self._current_phase is not None:
            self.stop()
        self._current_phase = phase
        self._phase_start = time.perf_counter()

    def stop(self) -> float:
        """Stop timing current phase and return elapsed time."""
        if self._current_phase is None:
            return 0.0
        elapsed = time.perf_counter() - self._phase_start
        self.phases[self._current_phase] = elapsed
        self._current_phase = None
        return elapsed

    def get(self, phase: str) -> float:
        """Get time for a phase."""
        return self.phases.get(phase, 0.0)


# ============================================================================
# Synthetic Data Generation
# ============================================================================


def create_cmc_benchmark_data(
    n_phi: int = 12,
    n_t1: int = 50,
    n_t2: int = 50,
    mode: str = "static",
    seed: int = 42,
) -> dict[str, Any]:
    """Create synthetic XPCS data for CMC benchmarking.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    n_t1 : int
        Number of t1 time points.
    n_t2 : int
        Number of t2 time points.
    mode : str
        Analysis mode: "static" or "laminar_flow".
    seed : int
        Random seed.

    Returns
    -------
    dict
        Data dictionary with pooled arrays ready for CMC.
    """
    rng = np.random.default_rng(seed)

    # Time arrays
    t1_base = np.arange(n_t1, dtype=np.float64)
    t2_base = np.arange(n_t2, dtype=np.float64)
    phi_angles = np.linspace(0, np.pi, n_phi)

    # Create meshgrid
    T1, T2 = np.meshgrid(t1_base, t2_base, indexing="ij")

    # True parameters
    if mode == "laminar_flow":
        true_params = {
            "D0": 15000.0,
            "alpha": 0.7,
            "D_offset": 100.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.5,
            "gamma_dot_offset": 0.0001,
            "phi_0": 0.0,
        }
    else:
        true_params = {
            "D0": 15000.0,
            "alpha": 0.7,
            "D_offset": 100.0,
        }

    # Generate C2 data
    c2_3d = np.zeros((n_phi, n_t1, n_t2), dtype=np.float64)
    tau = np.abs(T1 - T2) + 1e-6
    contrast = 0.3
    offset = 1.0

    for i, phi in enumerate(phi_angles):
        D_eff = true_params["D0"] * (1 + 0.1 * np.sin(phi))
        decay = np.exp(-D_eff * tau / 100000)
        c2_3d[i] = offset + contrast * decay

    # Add noise
    c2_3d += rng.normal(0, 0.01, c2_3d.shape)

    # Pool data (flatten to 1D arrays as expected by fit_mcmc_jax)
    n_total = n_phi * n_t1 * n_t2
    c2_pooled = np.zeros(n_total, dtype=np.float64)
    t1_pooled = np.zeros(n_total, dtype=np.float64)
    t2_pooled = np.zeros(n_total, dtype=np.float64)
    phi_pooled = np.zeros(n_total, dtype=np.float64)

    idx = 0
    for i, phi in enumerate(phi_angles):
        for j in range(n_t1):
            for k in range(n_t2):
                c2_pooled[idx] = c2_3d[i, j, k]
                t1_pooled[idx] = t1_base[j]
                t2_pooled[idx] = t2_base[k]
                phi_pooled[idx] = phi
                idx += 1

    return {
        "c2_pooled": c2_pooled,
        "t1_pooled": t1_pooled,
        "t2_pooled": t2_pooled,
        "phi_pooled": phi_pooled,
        "phi_angles": phi_angles,
        "n_phi": n_phi,
        "n_t1": n_t1,
        "n_t2": n_t2,
        "n_total": n_total,
        "q": 0.01,
        "L": 1000.0,
        "dt": 1.0,
        "true_params": true_params,
        "mode": mode,
    }


def create_minimal_cmc_config(
    n_warmup: int = 100,
    n_samples: int = 200,
    n_chains: int = 1,
    max_points_per_shard: int = 5000,
) -> dict[str, Any]:
    """Create minimal CMC config for benchmarking."""
    return {
        "mcmc": {
            "num_warmup": n_warmup,
            "num_samples": n_samples,
            "num_chains": n_chains,
            "target_accept_prob": 0.8,
            "max_tree_depth": 8,
        },
        "sharding": {
            "max_points_per_shard": max_points_per_shard,
            "min_shards": 2,
        },
        "backend": {
            "type": "multiprocessing",
            "max_workers": min(4, mp.cpu_count()),
        },
        "validation": {
            "max_divergence_rate": 0.10,
        },
    }


# ============================================================================
# Individual Component Benchmarks
# ============================================================================


def benchmark_nuts_single_shard(
    data: dict[str, Any],
    config: dict[str, Any],
    n_runs: int = 3,
) -> tuple[float, float, dict[str, Any]]:
    """Benchmark NUTS sampling on a single shard.

    Returns
    -------
    tuple
        (mean_time_s, std_time_s, metrics_dict)
    """
    try:
        from homodyne.optimization.cmc.model import get_xpcs_model
        from homodyne.optimization.cmc.sampler import run_nuts_sampling
        from homodyne.optimization.cmc.config import CMCConfig
        from homodyne.config.parameter_space import ParameterSpace
    except ImportError as e:
        logger.warning(f"CMC imports failed: {e}")
        return 0.0, 0.0, {"error": str(e)}

    # Take a subset for single shard
    shard_size = min(config["sharding"]["max_points_per_shard"], data["n_total"])
    indices = np.arange(shard_size)

    shard_data = {
        "c2": data["c2_pooled"][indices],
        "t1": data["t1_pooled"][indices],
        "t2": data["t2_pooled"][indices],
        "phi": data["phi_pooled"][indices],
        "q": data["q"],
        "L": data["L"],
        "dt": data["dt"],
    }

    # Create model (use "auto" mode which is the default for CMC)
    mode = data["mode"]
    model = get_xpcs_model(per_angle_mode="auto")

    # Create config and parameter space
    cmc_config = CMCConfig.from_dict(config)
    param_space = ParameterSpace.from_defaults(mode)

    # Create phi_unique and phi_indices for the model
    phi_unique = np.unique(shard_data["phi"])
    phi_indices = np.searchsorted(phi_unique, shard_data["phi"])

    model_kwargs = {
        "data": jnp.array(shard_data["c2"]),
        "t1": jnp.array(shard_data["t1"]),
        "t2": jnp.array(shard_data["t2"]),
        "phi_unique": jnp.array(phi_unique),
        "phi_indices": jnp.array(phi_indices),
        "q": shard_data["q"],
        "L": shard_data["L"],
        "dt": shard_data["dt"],
        "n_phi": data["n_phi"],
        "analysis_mode": mode,
        "parameter_space": param_space,
    }

    times = []
    metrics: dict[str, Any] = {}

    for i in range(n_runs + 1):  # +1 for warmup
        gc.collect()
        rng_key = random.PRNGKey(42 + i)

        start = time.perf_counter()
        try:
            samples, stats = run_nuts_sampling(
                model=model,
                model_kwargs=model_kwargs,
                config=cmc_config,
                initial_values=None,
                parameter_space=param_space,
                n_phi=data["n_phi"],
                analysis_mode=mode,
                rng_key=rng_key,
                progress_bar=False,
            )
            elapsed = time.perf_counter() - start

            if i > 0:  # Skip warmup
                times.append(elapsed)
                metrics = {
                    "n_samples": cmc_config.num_samples,
                    "n_warmup": cmc_config.num_warmup,
                    "n_chains": cmc_config.num_chains,
                    "shard_size": shard_size,
                }

        except Exception as e:
            logger.error(f"NUTS sampling failed: {e}")
            if i > 0:
                times.append(float("inf"))

    if not times or all(t == float("inf") for t in times):
        return 0.0, 0.0, {"error": "All runs failed"}

    valid_times = [t for t in times if t != float("inf")]
    mean_time = np.mean(valid_times)
    std_time = np.std(valid_times) if len(valid_times) > 1 else 0.0

    # Calculate throughput
    total_samples = cmc_config.num_samples * cmc_config.num_chains
    metrics["samples_per_second"] = total_samples / mean_time if mean_time > 0 else 0
    metrics["time_per_sample_ms"] = (mean_time * 1000) / total_samples if total_samples > 0 else 0

    return mean_time, std_time, metrics


def benchmark_multiprocessing_overhead(
    n_shards: int = 10,
    shard_size: int = 1000,
) -> dict[str, float]:
    """Benchmark multiprocessing overhead (IPC, worker startup, etc.).

    NOTE: This benchmark uses cloudpickle for IPC measurement, which is the
    same serialization used by the multiprocessing backend. The test data
    is synthetic and trusted (generated within this benchmark).

    Returns
    -------
    dict
        Overhead metrics in seconds and percentages.
    """
    # Use cloudpickle (same as multiprocessing backend) for realistic IPC measurement
    import cloudpickle

    # Test data for IPC (synthetic, trusted data)
    test_data = {
        "c2": np.random.randn(shard_size).astype(np.float64),
        "t1": np.random.randn(shard_size).astype(np.float64),
        "t2": np.random.randn(shard_size).astype(np.float64),
        "phi": np.random.randn(shard_size).astype(np.float64),
    }

    # Benchmark serialization (cloudpickle)
    serialize_times = []
    for _ in range(10):
        start = time.perf_counter()
        serialized = cloudpickle.dumps(test_data)
        serialize_times.append(time.perf_counter() - start)

    # Benchmark deserialization
    deserialize_times = []
    for _ in range(10):
        start = time.perf_counter()
        _ = cloudpickle.loads(serialized)
        deserialize_times.append(time.perf_counter() - start)

    # Benchmark process pool startup
    startup_times = []
    n_workers = min(4, mp.cpu_count())
    for _ in range(3):
        start = time.perf_counter()
        with mp.Pool(n_workers) as pool:
            # Use a simple function instead of lambda (pickle-safe)
            pool.map(abs, range(10))
        startup_times.append(time.perf_counter() - start)

    # Benchmark queue throughput
    queue_times = []
    for _ in range(5):
        q = mp.Queue()
        start = time.perf_counter()
        for i in range(n_shards):
            q.put(test_data)
        for i in range(n_shards):
            q.get()
        queue_times.append(time.perf_counter() - start)

    return {
        "serialize_time_ms": np.mean(serialize_times) * 1000,
        "deserialize_time_ms": np.mean(deserialize_times) * 1000,
        "pool_startup_time_s": np.mean(startup_times),
        "queue_throughput_time_s": np.mean(queue_times),
        "total_ipc_per_shard_ms": (
            np.mean(serialize_times) + np.mean(deserialize_times)
        ) * 1000,
        "data_size_bytes": len(serialized),
    }


def benchmark_consensus_aggregation(
    n_shards: int = 100,
    n_samples: int = 500,
    n_params: int = 7,
) -> dict[str, float]:
    """Benchmark consensus aggregation step.

    Returns
    -------
    dict
        Aggregation timing metrics.
    """
    # Simulate shard posteriors
    rng = np.random.default_rng(42)
    shard_samples = [
        rng.normal(0, 1, (n_samples, n_params)) for _ in range(n_shards)
    ]
    shard_weights = np.ones(n_shards) / n_shards

    # Benchmark simple weighted average (current approach)
    weighted_times = []
    for _ in range(10):
        start = time.perf_counter()
        stacked = np.stack(shard_samples)
        weights_expanded = shard_weights[:, None, None]
        _ = np.sum(stacked * weights_expanded, axis=0)
        weighted_times.append(time.perf_counter() - start)

    # Benchmark covariance-weighted (more accurate but slower)
    cov_times = []
    for _ in range(5):
        start = time.perf_counter()
        # Compute per-shard covariance
        covs = [np.cov(samples.T) for samples in shard_samples]
        # This would be the full consensus calculation
        elapsed = time.perf_counter() - start
        cov_times.append(elapsed)

    return {
        "simple_weighted_ms": np.mean(weighted_times) * 1000,
        "covariance_weighted_ms": np.mean(cov_times) * 1000,
        "n_shards": n_shards,
        "n_samples": n_samples,
        "n_params": n_params,
    }


# ============================================================================
# Full CMC Benchmark
# ============================================================================


def run_cmc_benchmark(
    n_phi: int = 12,
    n_t1: int = 50,
    n_t2: int = 50,
    mode: str = "static",
    n_warmup: int = 100,
    n_samples: int = 200,
    profile_cpu: bool = True,
    profile_memory: bool = True,
) -> CMCBenchmarkResult:
    """Run full CMC benchmark with profiling.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    n_t1 : int
        Time dimension 1.
    n_t2 : int
        Time dimension 2.
    mode : str
        Analysis mode.
    n_warmup : int
        MCMC warmup samples.
    n_samples : int
        MCMC production samples.
    profile_cpu : bool
        Whether to run CPU profiling.
    profile_memory : bool
        Whether to run memory profiling.

    Returns
    -------
    CMCBenchmarkResult
        Complete benchmark results.
    """
    logger.info("=" * 70)
    logger.info("CMC PERFORMANCE BENCHMARK")
    logger.info("=" * 70)

    # Create data
    logger.info(f"\nDataset: {n_phi} phi × {n_t1} t1 × {n_t2} t2 = {n_phi * n_t1 * n_t2:,} points")
    logger.info(f"Mode: {mode}")

    data = create_cmc_benchmark_data(n_phi, n_t1, n_t2, mode)
    config = create_minimal_cmc_config(n_warmup, n_samples)

    result = CMCBenchmarkResult(
        n_points=data["n_total"],
        n_phi=n_phi,
        analysis_mode=mode,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=config["mcmc"]["num_chains"],
    )

    timer = PhaseTimer()

    # Phase 1: Single-shard NUTS benchmark
    logger.info("\n--- Phase 1: Single-Shard NUTS Benchmark ---")
    timer.start("nuts_single")

    if profile_cpu:
        with CPUProfiler() as cpu_prof:
            mean_time, std_time, nuts_metrics = benchmark_nuts_single_shard(data, config)
        result.cpu_profile.hot_functions = cpu_prof.get_hot_functions(15)
        result.cpu_profile.time_by_module = cpu_prof.get_time_by_module()
    else:
        mean_time, std_time, nuts_metrics = benchmark_nuts_single_shard(data, config)

    timer.stop()

    if "error" not in nuts_metrics:
        logger.info(f"NUTS single-shard: {mean_time:.2f}s ± {std_time:.2f}s")
        logger.info(f"Throughput: {nuts_metrics.get('samples_per_second', 0):.1f} samples/sec")
        logger.info(f"Time per sample: {nuts_metrics.get('time_per_sample_ms', 0):.1f} ms")
        result.timing.production_sampling_s = mean_time
        result.timing.samples_per_second = nuts_metrics.get("samples_per_second", 0)
        result.timing.time_per_sample_ms = nuts_metrics.get("time_per_sample_ms", 0)
        result.shard_size = nuts_metrics.get("shard_size", 0)
    else:
        logger.warning(f"NUTS benchmark failed: {nuts_metrics['error']}")

    # Phase 2: Multiprocessing overhead
    logger.info("\n--- Phase 2: Multiprocessing Overhead ---")
    timer.start("mp_overhead")

    mp_metrics = benchmark_multiprocessing_overhead(
        n_shards=data["n_total"] // config["sharding"]["max_points_per_shard"] + 1,
        shard_size=config["sharding"]["max_points_per_shard"],
    )

    timer.stop()
    logger.info(f"Serialize time: {mp_metrics['serialize_time_ms']:.2f} ms/shard")
    logger.info(f"Deserialize time: {mp_metrics['deserialize_time_ms']:.2f} ms/shard")
    logger.info(f"Pool startup: {mp_metrics['pool_startup_time_s']:.2f}s")
    logger.info(f"Queue throughput: {mp_metrics['queue_throughput_time_s']:.2f}s")

    result.ipc_overhead_s = mp_metrics["queue_throughput_time_s"]

    # Phase 3: Consensus aggregation
    logger.info("\n--- Phase 3: Consensus Aggregation ---")
    timer.start("consensus")

    n_estimated_shards = max(2, data["n_total"] // config["sharding"]["max_points_per_shard"])
    n_params = 7 if mode == "laminar_flow" else 3
    consensus_metrics = benchmark_consensus_aggregation(
        n_shards=n_estimated_shards,
        n_samples=n_samples,
        n_params=n_params,
    )

    timer.stop()
    logger.info(f"Simple weighted: {consensus_metrics['simple_weighted_ms']:.2f} ms")
    logger.info(f"Covariance weighted: {consensus_metrics['covariance_weighted_ms']:.2f} ms")

    result.timing.consensus_aggregation_s = consensus_metrics["simple_weighted_ms"] / 1000
    result.n_shards = n_estimated_shards

    # Phase 4: Memory profiling
    if profile_memory:
        logger.info("\n--- Phase 4: Memory Profiling ---")
        timer.start("memory")

        with MemoryTracker(track_allocations=True) as mem_tracker:
            # Re-run NUTS to capture memory
            benchmark_nuts_single_shard(data, config, n_runs=1)

        memory_profile = mem_tracker.get_profile()
        result.memory = memory_profile

        timer.stop()
        logger.info(f"Baseline memory: {memory_profile.baseline_mb:.1f} MB")
        logger.info(f"Peak memory: {memory_profile.peak_mb:.1f} MB")
        logger.info(f"Delta: {memory_profile.delta_mb:.1f} MB")

        if memory_profile.top_allocations:
            logger.info("Top allocations:")
            for alloc, size in memory_profile.top_allocations[:5]:
                logger.info(f"  {size / 1024:.1f} KB: {alloc[:80]}...")

    # Calculate totals
    result.timing.total_s = sum(timer.phases.values())

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total benchmark time: {result.timing.total_s:.2f}s")
    logger.info(f"NUTS sampling: {result.timing.production_sampling_s:.2f}s")
    logger.info(f"Throughput: {result.timing.samples_per_second:.1f} samples/sec")
    logger.info(f"Memory delta: {result.memory.delta_mb:.1f} MB")

    if result.cpu_profile.hot_functions:
        logger.info("\nHot Functions (top 5):")
        for func, time_s, calls in result.cpu_profile.hot_functions[:5]:
            logger.info(f"  {time_s:.3f}s ({calls:,} calls): {func[:60]}...")

    logger.info("=" * 70)

    return result


# ============================================================================
# Flamegraph Generation (requires py-spy)
# ============================================================================


def generate_flamegraph(
    output_path: Path,
    duration_s: int = 30,
) -> bool:
    """Generate flamegraph using py-spy.

    Requires py-spy to be installed: pip install py-spy

    Parameters
    ----------
    output_path : Path
        Output SVG path.
    duration_s : int
        Duration to profile.

    Returns
    -------
    bool
        True if successful.
    """
    import subprocess

    # Check if py-spy is available
    try:
        subprocess.run(["py-spy", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("py-spy not available. Install with: pip install py-spy")
        return False

    # Create a script to run
    script = f"""
import sys
sys.path.insert(0, '.')
from tests.performance.benchmark_cmc import run_cmc_benchmark
run_cmc_benchmark(n_phi=12, n_t1=50, n_t2=50, profile_cpu=False, profile_memory=False)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        cmd = [
            "py-spy",
            "record",
            "-o",
            str(output_path),
            "--format",
            "speedscope",
            "--duration",
            str(duration_s),
            "--",
            "python",
            script_path,
        ]
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Flamegraph saved to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"py-spy failed: {e}")
        return False
    finally:
        os.unlink(script_path)


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for CMC benchmarking."""
    import argparse

    parser = argparse.ArgumentParser(description="CMC Performance Benchmark")
    parser.add_argument(
        "--mode",
        choices=["quick", "standard", "enterprise"],
        default="standard",
        help="Profiling mode",
    )
    parser.add_argument(
        "--analysis",
        choices=["static", "laminar_flow"],
        default="static",
        help="Analysis mode",
    )
    parser.add_argument("--n-phi", type=int, default=12, help="Number of phi angles")
    parser.add_argument("--n-t1", type=int, default=50, help="Time dimension 1")
    parser.add_argument("--n-t2", type=int, default=50, help="Time dimension 2")
    parser.add_argument("--n-warmup", type=int, default=100, help="MCMC warmup samples")
    parser.add_argument("--n-samples", type=int, default=200, help="MCMC production samples")
    parser.add_argument("--flamegraph", action="store_true", help="Generate flamegraph")
    parser.add_argument("--output", type=str, default=None, help="Output path for results")

    args = parser.parse_args()

    # Mode-specific settings
    if args.mode == "quick":
        n_warmup, n_samples = 50, 100
        profile_cpu, profile_memory = False, False
    elif args.mode == "standard":
        n_warmup, n_samples = args.n_warmup, args.n_samples
        profile_cpu, profile_memory = True, True
    else:  # enterprise
        n_warmup, n_samples = 200, 500
        profile_cpu, profile_memory = True, True

    # Run benchmark
    result = run_cmc_benchmark(
        n_phi=args.n_phi,
        n_t1=args.n_t1,
        n_t2=args.n_t2,
        mode=args.analysis,
        n_warmup=n_warmup,
        n_samples=n_samples,
        profile_cpu=profile_cpu,
        profile_memory=profile_memory,
    )

    # Save results
    output_path = Path(args.output) if args.output else Path(__file__).parent / "cmc_benchmark_results.npz"
    np.savez(output_path, **result.to_dict())
    logger.info(f"\nResults saved to: {output_path}")

    # Generate flamegraph if requested
    if args.flamegraph:
        flamegraph_path = output_path.with_suffix(".svg")
        generate_flamegraph(flamegraph_path)


if __name__ == "__main__":
    main()
