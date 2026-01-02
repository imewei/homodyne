"""
Baseline Performance Benchmark for NLSQ Optimizations (Spec 006).

This module provides comprehensive baseline benchmarks for the NLSQ optimization
pipeline before implementing performance optimizations. Results are saved to
`baseline_results.npz` for comparison after optimizations are applied.

Benchmarks Include:
1. End-to-end NLSQ fitting time
2. Per-iteration timing breakdown
3. Residual function evaluation time
4. Memory usage measurements
5. Vectorization opportunity measurements

Usage:
    # Run benchmarks and save baseline
    python -m tests.performance.benchmark_baseline

    # Run via pytest
    pytest tests/performance/benchmark_baseline.py -v --benchmark-only

Author: Homodyne Performance Team
Date: 2026-01-01
Spec: 006-nlsq-performance
"""

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TimingResult:
    """Timing result for a single benchmark."""

    name: str
    mean_time_s: float
    std_time_s: float
    min_time_s: float
    max_time_s: float
    n_runs: int


@dataclass
class MemoryResult:
    """Memory measurement result."""

    name: str
    peak_memory_mb: float
    baseline_memory_mb: float
    delta_memory_mb: float


@dataclass
class BaselineResults:
    """Complete baseline benchmark results."""

    # Timing measurements
    total_fit_time_s: float
    residual_eval_time_s: float
    jacobian_eval_time_s: float
    per_iteration_time_s: float

    # Memory measurements
    peak_memory_mb: float
    baseline_memory_mb: float

    # Iteration counts
    n_iterations: int
    n_function_evals: int

    # Dataset characteristics
    n_points: int
    n_params: int
    n_phi: int

    # Fitted parameters (for accuracy validation)
    fitted_params: np.ndarray
    param_names: list[str]

    # Speedup targets (from spec)
    target_speedup_total: float = 2.0  # 50% reduction = 2x speedup
    target_speedup_residual: float = 1.5  # 15-20% per-iteration
    target_memory_reduction: float = 0.7  # 30% reduction

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "total_fit_time_s": self.total_fit_time_s,
            "residual_eval_time_s": self.residual_eval_time_s,
            "jacobian_eval_time_s": self.jacobian_eval_time_s,
            "per_iteration_time_s": self.per_iteration_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "baseline_memory_mb": self.baseline_memory_mb,
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "n_points": self.n_points,
            "n_params": self.n_params,
            "n_phi": self.n_phi,
            "fitted_params": self.fitted_params,
            "param_names": self.param_names,
            "target_speedup_total": self.target_speedup_total,
            "target_speedup_residual": self.target_speedup_residual,
            "target_memory_reduction": self.target_memory_reduction,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaselineResults":
        """Load from dictionary."""
        return cls(
            total_fit_time_s=d["total_fit_time_s"],
            residual_eval_time_s=d["residual_eval_time_s"],
            jacobian_eval_time_s=d["jacobian_eval_time_s"],
            per_iteration_time_s=d["per_iteration_time_s"],
            peak_memory_mb=d["peak_memory_mb"],
            baseline_memory_mb=d["baseline_memory_mb"],
            n_iterations=d["n_iterations"],
            n_function_evals=d["n_function_evals"],
            n_points=d["n_points"],
            n_params=d["n_params"],
            n_phi=d["n_phi"],
            fitted_params=d["fitted_params"],
            param_names=list(d["param_names"]),
            target_speedup_total=d.get("target_speedup_total", 2.0),
            target_speedup_residual=d.get("target_speedup_residual", 1.5),
            target_memory_reduction=d.get("target_memory_reduction", 0.7),
        )


# ============================================================================
# Utility Functions
# ============================================================================


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024**2)
    return 0.0


# Performance Optimization (Spec 006 - FR-006, T033-T035): Memory profiling utilities
class MemoryProfiler:
    """Track memory usage during optimization iterations.

    Performance Optimization (Spec 006 - T033):
    Provides detailed memory profiling for diagonal correction and
    optimization iterations to verify 30% memory reduction target.
    """

    def __init__(self):
        self.snapshots: list[tuple[str, float]] = []
        self.peak_memory_mb: float = 0.0
        self.baseline_memory_mb: float = 0.0

    def snapshot(self, label: str) -> float:
        """Take memory snapshot with label."""
        current_mb = get_memory_mb()
        self.snapshots.append((label, current_mb))
        if current_mb > self.peak_memory_mb:
            self.peak_memory_mb = current_mb
        return current_mb

    def set_baseline(self) -> float:
        """Set baseline memory (call before operation)."""
        gc.collect()
        self.baseline_memory_mb = get_memory_mb()
        self.snapshot("baseline")
        return self.baseline_memory_mb

    def get_peak_delta(self) -> float:
        """Get peak memory increase from baseline."""
        return self.peak_memory_mb - self.baseline_memory_mb

    def get_report(self) -> dict[str, Any]:
        """Get memory profiling report."""
        return {
            "baseline_mb": self.baseline_memory_mb,
            "peak_mb": self.peak_memory_mb,
            "peak_delta_mb": self.get_peak_delta(),
            "snapshots": self.snapshots,
        }


def benchmark_diagonal_correction_memory(data: dict) -> dict[str, Any]:
    """Benchmark memory usage for diagonal correction.

    Performance Optimization (Spec 006 - T034):
    Measures memory usage for batch diagonal correction to verify
    30% reduction target compared to sequential approach.
    """
    from homodyne.data.xpcs_loader import XPCSDataLoader

    c2_exp = data["c2_exp"]
    n_phi, n_t1, n_t2 = c2_exp.shape

    # Create a minimal config dict for XPCSDataLoader
    config_dict = {
        "experimental_data": {
            "data_folder_path": "/tmp",
            "data_file_name": "dummy.npz",
        },
        "analyzer_parameters": {
            "dt": 1.0,
            "start_frame": 1,
            "end_frame": 100,
        },
    }

    # Initialize loader (we just need its correction methods)
    try:
        loader = XPCSDataLoader(config_dict=config_dict)
    except Exception:
        # If initialization fails, measure memory for correction alone
        logger.warning("XPCSDataLoader init failed, using direct correction")
        return {"error": "loader_init_failed"}

    profiler = MemoryProfiler()

    # Measure sequential correction memory
    profiler.set_baseline()
    c2_seq_result = []
    for i in range(n_phi):
        c2_corrected = loader._correct_diagonal(c2_exp[i].copy())
        c2_seq_result.append(c2_corrected)
    c2_seq_result = np.array(c2_seq_result)
    seq_peak = profiler.snapshot("sequential_complete")
    seq_memory = profiler.get_peak_delta()
    del c2_seq_result
    gc.collect()

    # Measure batch correction memory
    profiler2 = MemoryProfiler()
    profiler2.set_baseline()
    c2_batch_result = loader._correct_diagonal_batch(c2_exp.copy())
    batch_peak = profiler2.snapshot("batch_complete")
    batch_memory = profiler2.get_peak_delta()
    del c2_batch_result
    gc.collect()

    # Calculate reduction
    memory_reduction = 1.0 - (batch_memory / seq_memory) if seq_memory > 0 else 0.0

    return {
        "sequential_peak_mb": seq_peak,
        "sequential_delta_mb": seq_memory,
        "batch_peak_mb": batch_peak,
        "batch_delta_mb": batch_memory,
        "memory_reduction_fraction": memory_reduction,
        "target_met": memory_reduction >= 0.30,
        "n_matrices": n_phi,
        "matrix_size": (n_t1, n_t2),
    }


def benchmark_optimization_memory(data: dict, config: dict) -> dict[str, Any]:
    """Benchmark memory usage during NLSQ optimization.

    Performance Optimization (Spec 006 - T035):
    Tracks memory usage during optimization to verify no unnecessary
    array copies occur during iterations.
    """
    try:
        from homodyne.optimization.nlsq import NLSQ_AVAILABLE, fit_nlsq_jax
    except ImportError:
        return {"error": "nlsq_not_available"}

    if not NLSQ_AVAILABLE:
        return {"error": "nlsq_not_available"}

    profiler = MemoryProfiler()
    profiler.set_baseline()
    profiler.snapshot("before_fit")

    try:
        result = fit_nlsq_jax(data, config)
        profiler.snapshot("after_fit")

        # Force garbage collection and measure
        gc.collect()
        profiler.snapshot("after_gc")

    except Exception as e:
        return {"error": str(e)}

    return {
        "fit_success": result.success if hasattr(result, "success") else False,
        "baseline_mb": profiler.baseline_memory_mb,
        "peak_mb": profiler.peak_memory_mb,
        "peak_delta_mb": profiler.get_peak_delta(),
        "snapshots": profiler.snapshots,
    }


def measure_time(func, *args, n_runs: int = 5, warmup: int = 1, **kwargs) -> TimingResult:
    """Measure execution time of a function.

    Parameters
    ----------
    func : callable
        Function to benchmark
    *args : Any
        Positional arguments for func
    n_runs : int
        Number of timing runs
    warmup : int
        Number of warmup runs (not timed)
    **kwargs : Any
        Keyword arguments for func

    Returns
    -------
    TimingResult
        Timing statistics
    """
    # Warmup
    for _ in range(warmup):
        result = func(*args, **kwargs)
        if JAX_AVAILABLE and hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    times = []
    for _ in range(n_runs):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if JAX_AVAILABLE and hasattr(result, "block_until_ready"):
            result.block_until_ready()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return TimingResult(
        name=func.__name__ if hasattr(func, "__name__") else "anonymous",
        mean_time_s=np.mean(times),
        std_time_s=np.std(times),
        min_time_s=np.min(times),
        max_time_s=np.max(times),
        n_runs=n_runs,
    )


def measure_memory(func, *args, **kwargs) -> tuple[Any, MemoryResult]:
    """Measure memory usage during function execution.

    Parameters
    ----------
    func : callable
        Function to measure
    *args, **kwargs
        Arguments for func

    Returns
    -------
    result : Any
        Function result
    memory : MemoryResult
        Memory measurements
    """
    gc.collect()
    baseline = get_memory_mb()

    result = func(*args, **kwargs)

    gc.collect()
    peak = get_memory_mb()

    return result, MemoryResult(
        name=func.__name__ if hasattr(func, "__name__") else "anonymous",
        peak_memory_mb=peak,
        baseline_memory_mb=baseline,
        delta_memory_mb=peak - baseline,
    )


# ============================================================================
# Synthetic Data Generation
# ============================================================================


def create_benchmark_dataset(
    n_phi: int = 23,
    n_t1: int = 100,
    n_t2: int = 100,
    mode: str = "laminar_flow",
    seed: int = 42,
) -> dict[str, Any]:
    """Create synthetic XPCS dataset for benchmarking.

    Parameters
    ----------
    n_phi : int
        Number of phi angles (default 23 for realistic benchmark)
    n_t1 : int
        Number of t1 time points
    n_t2 : int
        Number of t2 time points
    mode : str
        Analysis mode: "static" or "laminar_flow"
    seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Dataset dictionary with keys: c2_exp, sigma, phi_angles_list, t1, t2, etc.
    """
    rng = np.random.default_rng(seed)

    # Time arrays
    t1 = np.arange(n_t1, dtype=np.float64)
    t2 = np.arange(n_t2, dtype=np.float64)

    # Phi angles (radians)
    phi_angles = np.linspace(0, np.pi, n_phi)

    # True physical parameters
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

    # Create meshgrid
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")

    # Generate synthetic g2 data
    c2_exp = np.zeros((n_phi, n_t1, n_t2), dtype=np.float64)
    contrast_true = 0.3
    offset_true = 1.0

    # Simplified g2 model for benchmarking
    tau = np.abs(T1 - T2) + 1e-6
    for i, phi in enumerate(phi_angles):
        D_eff = true_params["D0"] * (1 + 0.1 * np.sin(phi))
        decay = np.exp(-D_eff * tau / 100000)
        c2_exp[i] = offset_true + contrast_true * decay

    # Add realistic noise
    noise_level = 0.01
    c2_exp += rng.normal(0, noise_level, c2_exp.shape)

    # Sigma (uncertainties)
    sigma = np.ones_like(c2_exp) * noise_level

    # Construct dataset
    data = {
        "c2_exp": c2_exp,
        "sigma": sigma,
        "phi_angles_list": phi_angles,
        "t1": t1,
        "t2": t2,
        "wavevector_q_list": np.array([0.01]),
        "L": 1000.0,
        "dt": 1.0,
        "true_params": true_params,
        "contrast_true": contrast_true,
        "offset_true": offset_true,
    }

    return data


def create_benchmark_config(mode: str = "laminar_flow") -> dict[str, Any]:
    """Create benchmark configuration.

    Parameters
    ----------
    mode : str
        Analysis mode: "static" or "laminar_flow"

    Returns
    -------
    dict
        Configuration dictionary
    """
    config = {
        "analysis": {
            "mode": mode,
        },
        "optimization": {
            "lsq": {
                "max_iterations": 100,
                "ftol": 1e-8,
                "gtol": 1e-8,
                "xtol": 1e-8,
            },
            "per_angle_scaling": True,
        },
        "initial_params": {
            "contrast": 0.3,
            "offset": 1.0,
            "D0": 10000.0,
            "alpha": 0.5,
            "D_offset": 50.0,
        },
    }

    if mode == "laminar_flow":
        config["initial_params"].update(
            {
                "gamma_dot_t0": 0.0005,
                "beta": 0.4,
                "gamma_dot_offset": 0.0001,
                "phi_0": 0.0,
            }
        )

    return config


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_residual_evaluation(data: dict, config: dict, n_runs: int = 10) -> TimingResult:
    """Benchmark residual function evaluation.

    This measures the time to compute residuals for a single parameter vector,
    which is the core computation in each optimization iteration.
    """
    try:
        from homodyne.optimization.nlsq.strategies.residual import (
            StratifiedResidualFunction,
        )
    except ImportError:
        logger.warning("NLSQ strategies not available, skipping residual benchmark")
        return TimingResult("residual_eval", 0.0, 0.0, 0.0, 0.0, 0)

    # Prepare stratified data
    mode = config["analysis"]["mode"]
    n_phi = len(data["phi_angles_list"])
    phi_angles = data["phi_angles_list"]
    t1 = data["t1"]
    t2 = data["t2"]
    c2_exp = data["c2_exp"]
    sigma = data["sigma"]
    q = data["wavevector_q_list"][0]
    L = data["L"]
    dt = data["dt"]

    if mode == "laminar_flow":
        physical_names = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_offset",
            "phi_0",
        ]
    else:
        physical_names = ["D0", "alpha", "D_offset"]

    # Create mock stratified data structure matching the expected interface
    class MockChunk:
        def __init__(self, phi, t1, t2, g2, q, L, dt):
            self.phi = phi
            self.t1 = t1
            self.t2 = t2
            self.g2 = g2
            self.q = q
            self.L = L
            self.dt = dt

    # Create meshgrid indices
    n_t1, n_t2 = len(t1), len(t2)
    t1_grid, t2_grid = np.meshgrid(t1, t2, indexing="ij")

    # Create chunks (2 chunks for typical dataset)
    n_chunks = 2
    n_t_pairs = n_t1 * n_t2
    t_pairs_per_chunk = n_t_pairs // n_chunks

    chunks = []
    for i in range(n_chunks):
        t_start = i * t_pairs_per_chunk
        t_end = min(t_start + t_pairs_per_chunk, n_t_pairs)

        chunk_phi = []
        chunk_t1 = []
        chunk_t2 = []
        chunk_g2 = []

        for phi_idx, phi_val in enumerate(phi_angles):
            # Flatten the (t1, t2) grid for this phi
            t1_slice = t1_grid.flatten()
            t2_slice = t2_grid.flatten()
            g2_slice = c2_exp[phi_idx].flatten()

            # Take the subset for this chunk
            chunk_phi.extend([phi_val] * (t_end - t_start))
            chunk_t1.extend(t1_slice[t_start:t_end])
            chunk_t2.extend(t2_slice[t_start:t_end])
            chunk_g2.extend(g2_slice[t_start:t_end])

        chunk = MockChunk(
            phi=np.array(chunk_phi),
            t1=np.array(chunk_t1),
            t2=np.array(chunk_t2),
            g2=np.array(chunk_g2),
            q=q,
            L=L,
            dt=dt,
        )
        chunks.append(chunk)

    class MockStratifiedData:
        def __init__(self, chunks, sigma):
            self.chunks = chunks
            self.sigma = sigma

    stratified_data = MockStratifiedData(chunks, sigma)

    # Create residual function
    residual_fn = StratifiedResidualFunction(
        stratified_data=stratified_data,
        per_angle_scaling=True,
        physical_param_names=physical_names,
        logger=logging.getLogger("benchmark"),
    )

    # Create test parameters
    n_scaling = 2 * n_phi
    n_physical = len(physical_names)
    n_params = n_scaling + n_physical

    # Random test parameters
    rng = np.random.default_rng(42)
    test_params = rng.uniform(0.1, 1.0, n_params)

    # Benchmark
    result = measure_time(residual_fn, test_params, n_runs=n_runs, warmup=2)
    result.name = "residual_eval"
    return result


def benchmark_full_fit(data: dict, config: dict, n_runs: int = 3) -> tuple[TimingResult, dict]:
    """Benchmark full NLSQ fit.

    Returns timing and the fit result for parameter validation.
    """
    try:
        from homodyne.optimization.nlsq import NLSQ_AVAILABLE, fit_nlsq_jax
    except ImportError:
        logger.warning("NLSQ not available, skipping full fit benchmark")
        return TimingResult("full_fit", 0.0, 0.0, 0.0, 0.0, 0), {}

    if not NLSQ_AVAILABLE:
        logger.warning("NLSQ not available, skipping full fit benchmark")
        return TimingResult("full_fit", 0.0, 0.0, 0.0, 0.0, 0), {}

    # Run fits and measure time
    times = []
    last_result = None

    for i in range(n_runs + 1):  # +1 for warmup
        gc.collect()
        start = time.perf_counter()

        try:
            result = fit_nlsq_jax(data, config)
            elapsed = time.perf_counter() - start

            if i > 0:  # Skip warmup
                times.append(elapsed)
                last_result = result
        except Exception as e:
            logger.error(f"Fit failed: {e}")
            if i > 0:
                times.append(float("inf"))

    if not times:
        return TimingResult("full_fit", 0.0, 0.0, 0.0, 0.0, 0), {}

    timing = TimingResult(
        name="full_fit",
        mean_time_s=np.mean(times),
        std_time_s=np.std(times),
        min_time_s=np.min(times),
        max_time_s=np.max(times),
        n_runs=len(times),
    )

    # Extract result info
    result_info = {}
    if last_result is not None:
        result_info = {
            "success": last_result.success,
            "n_iterations": getattr(last_result, "n_iterations", 0),
            "n_function_evals": getattr(last_result, "n_function_evals", 0),
            "params": last_result.params if hasattr(last_result, "params") else None,
            "param_names": (
                last_result.param_names if hasattr(last_result, "param_names") else []
            ),
        }

    return timing, result_info


def benchmark_vectorization_opportunity(data: dict) -> dict[str, float]:
    """Measure potential for vectorization improvements.

    Compares sequential vs vectorized operations that are optimization targets.
    """
    if not JAX_AVAILABLE:
        return {}

    n_phi = len(data["phi_angles_list"])
    t1 = jnp.asarray(data["t1"])
    t2 = jnp.asarray(data["t2"])
    phi = jnp.asarray(data["phi_angles_list"])

    # Create meshgrid
    T1, T2 = jnp.meshgrid(t1, t2, indexing="ij")

    # Benchmark: Sequential per-angle computation
    def compute_sequential():
        results = []
        for i in range(n_phi):
            # Simplified g2 computation per angle
            tau = jnp.abs(T1 - T2) + 1e-6
            D_eff = 15000.0 * (1 + 0.1 * jnp.sin(phi[i]))
            decay = jnp.exp(-D_eff * tau / 100000)
            results.append(1.0 + 0.3 * decay)
        return jnp.stack(results, axis=0)

    # Benchmark: Vectorized computation
    def compute_vectorized():
        tau = jnp.abs(T1 - T2) + 1e-6

        def compute_single(phi_val):
            D_eff = 15000.0 * (1 + 0.1 * jnp.sin(phi_val))
            decay = jnp.exp(-D_eff * tau / 100000)
            return 1.0 + 0.3 * decay

        return jax.vmap(compute_single)(phi)

    # Time both approaches
    seq_timing = measure_time(compute_sequential, n_runs=5, warmup=2)
    vec_timing = measure_time(compute_vectorized, n_runs=5, warmup=2)

    speedup = seq_timing.mean_time_s / vec_timing.mean_time_s if vec_timing.mean_time_s > 0 else 0

    return {
        "sequential_time_s": seq_timing.mean_time_s,
        "vectorized_time_s": vec_timing.mean_time_s,
        "vectorization_speedup": speedup,
        "n_phi": n_phi,
    }


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_baseline_benchmarks(
    n_phi: int = 23,
    n_t1: int = 100,
    n_t2: int = 100,
    mode: str = "laminar_flow",
    save_path: Path | None = None,
) -> BaselineResults:
    """Run complete baseline benchmark suite.

    Parameters
    ----------
    n_phi : int
        Number of phi angles
    n_t1 : int
        Number of t1 time points
    n_t2 : int
        Number of t2 time points
    mode : str
        Analysis mode
    save_path : Path, optional
        Path to save results

    Returns
    -------
    BaselineResults
        Complete baseline measurements
    """
    logger.info("=" * 70)
    logger.info("NLSQ Baseline Performance Benchmark (Spec 006)")
    logger.info("=" * 70)

    # Create dataset
    logger.info(f"\nCreating benchmark dataset: {n_phi} phi × {n_t1} t1 × {n_t2} t2")
    logger.info(f"Mode: {mode}")

    data = create_benchmark_dataset(n_phi=n_phi, n_t1=n_t1, n_t2=n_t2, mode=mode)
    config = create_benchmark_config(mode=mode)

    n_points = n_phi * n_t1 * n_t2
    logger.info(f"Total points: {n_points:,}")

    # Memory baseline
    baseline_memory = get_memory_mb()
    logger.info(f"Baseline memory: {baseline_memory:.1f} MB")

    # Benchmark 1: Residual evaluation
    logger.info("\n--- Benchmark 1: Residual Evaluation ---")
    residual_timing = benchmark_residual_evaluation(data, config)
    logger.info(f"Mean time: {residual_timing.mean_time_s * 1000:.2f} ms")
    logger.info(f"Std dev: {residual_timing.std_time_s * 1000:.2f} ms")

    # Benchmark 2: Full fit
    logger.info("\n--- Benchmark 2: Full NLSQ Fit ---")
    fit_timing, fit_result = benchmark_full_fit(data, config)
    logger.info(f"Mean time: {fit_timing.mean_time_s:.2f} s")
    logger.info(f"Std dev: {fit_timing.std_time_s:.2f} s")

    if fit_result:
        logger.info(f"Success: {fit_result.get('success', False)}")
        logger.info(f"Iterations: {fit_result.get('n_iterations', 'N/A')}")
        logger.info(f"Function evals: {fit_result.get('n_function_evals', 'N/A')}")

    # Benchmark 3: Vectorization opportunity
    logger.info("\n--- Benchmark 3: Vectorization Opportunity ---")
    vec_results = benchmark_vectorization_opportunity(data)
    if vec_results:
        logger.info(f"Sequential: {vec_results['sequential_time_s'] * 1000:.2f} ms")
        logger.info(f"Vectorized: {vec_results['vectorized_time_s'] * 1000:.2f} ms")
        logger.info(f"Speedup potential: {vec_results['vectorization_speedup']:.1f}x")

    # Benchmark 4: Memory profiling (T033-T035)
    logger.info("\n--- Benchmark 4: Diagonal Correction Memory (T034) ---")
    diag_memory = benchmark_diagonal_correction_memory(data)
    if "error" not in diag_memory:
        logger.info(f"Sequential delta: {diag_memory['sequential_delta_mb']:.1f} MB")
        logger.info(f"Batch delta: {diag_memory['batch_delta_mb']:.1f} MB")
        logger.info(f"Memory reduction: {diag_memory['memory_reduction_fraction'] * 100:.1f}%")
        logger.info(f"Target met (30%): {diag_memory['target_met']}")
    else:
        logger.warning(f"Diagonal memory benchmark skipped: {diag_memory['error']}")

    logger.info("\n--- Benchmark 5: Optimization Memory (T035) ---")
    opt_memory = benchmark_optimization_memory(data, config)
    if "error" not in opt_memory:
        logger.info(f"Baseline: {opt_memory['baseline_mb']:.1f} MB")
        logger.info(f"Peak: {opt_memory['peak_mb']:.1f} MB")
        logger.info(f"Peak delta: {opt_memory['peak_delta_mb']:.1f} MB")
    else:
        logger.warning(f"Optimization memory benchmark skipped: {opt_memory['error']}")

    # Peak memory
    peak_memory = get_memory_mb()
    logger.info("\n--- Memory Usage ---")
    logger.info(f"Peak memory: {peak_memory:.1f} MB")
    logger.info(f"Delta: {peak_memory - baseline_memory:.1f} MB")

    # Compute per-iteration time
    n_iterations = fit_result.get("n_iterations", 1)
    per_iteration_time = fit_timing.mean_time_s / n_iterations if n_iterations > 0 else 0

    # Determine param count
    n_params = 2 * n_phi + (7 if mode == "laminar_flow" else 3)

    # Build results
    results = BaselineResults(
        total_fit_time_s=fit_timing.mean_time_s,
        residual_eval_time_s=residual_timing.mean_time_s,
        jacobian_eval_time_s=residual_timing.mean_time_s * 2,  # Approx 2x residual
        per_iteration_time_s=per_iteration_time,
        peak_memory_mb=peak_memory,
        baseline_memory_mb=baseline_memory,
        n_iterations=n_iterations,
        n_function_evals=fit_result.get("n_function_evals", 0),
        n_points=n_points,
        n_params=n_params,
        n_phi=n_phi,
        fitted_params=np.array(fit_result.get("params", [])),
        param_names=fit_result.get("param_names", []),
    )

    # Save results
    if save_path is None:
        save_path = Path(__file__).parent / "baseline_results.npz"

    logger.info(f"\nSaving baseline to: {save_path}")
    np.savez(save_path, **results.to_dict())

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total fit time: {results.total_fit_time_s:.2f} s")
    logger.info(f"Per-iteration time: {results.per_iteration_time_s * 1000:.2f} ms")
    logger.info(f"Residual eval time: {results.residual_eval_time_s * 1000:.2f} ms")
    logger.info(f"Peak memory: {results.peak_memory_mb:.1f} MB")
    logger.info("\nOptimization Targets (from spec):")
    logger.info(f"  Total speedup target: {results.target_speedup_total}x (50% reduction)")
    logger.info(f"  Per-iteration target: {results.target_speedup_residual}x (15-20% reduction)")
    logger.info(f"  Memory reduction target: {(1 - results.target_memory_reduction) * 100:.0f}%")
    logger.info("=" * 70)

    return results


def load_baseline(path: Path | None = None) -> BaselineResults | None:
    """Load saved baseline results.

    Parameters
    ----------
    path : Path, optional
        Path to baseline results file

    Returns
    -------
    BaselineResults or None
        Loaded results, or None if file doesn't exist
    """
    if path is None:
        path = Path(__file__).parent / "baseline_results.npz"

    if not path.exists():
        logger.warning(f"Baseline not found at {path}")
        return None

    data = np.load(path, allow_pickle=True)
    return BaselineResults.from_dict(dict(data))


def compare_with_baseline(current: BaselineResults, baseline: BaselineResults) -> dict[str, Any]:
    """Compare current results with baseline.

    Parameters
    ----------
    current : BaselineResults
        Current benchmark results
    baseline : BaselineResults
        Baseline results to compare against

    Returns
    -------
    dict
        Comparison results with speedups and memory changes
    """
    comparison = {
        "total_speedup": baseline.total_fit_time_s / current.total_fit_time_s,
        "residual_speedup": baseline.residual_eval_time_s / current.residual_eval_time_s,
        "per_iteration_speedup": baseline.per_iteration_time_s / current.per_iteration_time_s,
        "memory_ratio": current.peak_memory_mb / baseline.peak_memory_mb,
        "targets": {
            "total_speedup_met": (
                baseline.total_fit_time_s / current.total_fit_time_s
            ) >= baseline.target_speedup_total,
            "residual_speedup_met": (
                baseline.residual_eval_time_s / current.residual_eval_time_s
            ) >= baseline.target_speedup_residual,
            "memory_reduction_met": (
                current.peak_memory_mb / baseline.peak_memory_mb
            ) <= baseline.target_memory_reduction,
        },
    }

    # Validate parameter accuracy
    if len(current.fitted_params) > 0 and len(baseline.fitted_params) > 0:
        try:
            np.testing.assert_allclose(
                current.fitted_params,
                baseline.fitted_params,
                rtol=1e-6,
                atol=1e-10,
            )
            comparison["parameters_match"] = True
        except AssertionError:
            comparison["parameters_match"] = False
            comparison["param_diff"] = np.abs(current.fitted_params - baseline.fitted_params)

    return comparison


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NLSQ Baseline Performance Benchmark")
    parser.add_argument("--n-phi", type=int, default=23, help="Number of phi angles")
    parser.add_argument("--n-t1", type=int, default=100, help="Number of t1 points")
    parser.add_argument("--n-t2", type=int, default=100, help="Number of t2 points")
    parser.add_argument(
        "--mode",
        type=str,
        default="laminar_flow",
        choices=["static", "laminar_flow"],
        help="Analysis mode",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results",
    )

    args = parser.parse_args()

    save_path = Path(args.output) if args.output else None

    results = run_baseline_benchmarks(
        n_phi=args.n_phi,
        n_t1=args.n_t1,
        n_t2=args.n_t2,
        mode=args.mode,
        save_path=save_path,
    )
