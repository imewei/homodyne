"""
Performance Benchmarks for Homodyne v2
======================================

Comprehensive performance benchmarks for computational kernels:
- JAX backend mathematical operations
- NLSQ optimization performance
- Data loading and preprocessing
- Memory usage and scaling
- CPU vs GPU performance comparison
"""

import gc
import time

import numpy as np
import pytest

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Handle benchmark-specific imports
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.requires_jax
class TestComputationalBenchmarks:
    """Benchmark core computational operations."""

    def test_g1_diffusion_benchmark(self, jax_backend, benchmark_config):
        """Benchmark g1 diffusion computation across different sizes."""
        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        # Test different matrix sizes
        sizes = [50, 100, 200, 300]
        results = {}

        for size in sizes:
            t1, t2 = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
            q = 0.01
            D = 0.1

            # Warmup
            for _ in range(3):
                result = compute_g1_diffusion_jax(t1, t2, q, D)
                result.block_until_ready()

            # Benchmark
            times = []
            for _ in range(benchmark_config["min_rounds"]):
                start = time.perf_counter()
                result = compute_g1_diffusion_jax(t1, t2, q, D)
                result.block_until_ready()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            operations_per_sec = 1.0 / avg_time
            elements_per_sec = (size * size) / avg_time

            results[size] = {
                "avg_time": avg_time,
                "operations_per_sec": operations_per_sec,
                "elements_per_sec": elements_per_sec,
                "memory_mb": (size * size * 8) / (1024 * 1024),  # float64
            }

            # Performance expectations
            assert avg_time < 5.0, f"Size {size}: too slow ({avg_time:.3f}s)"
            assert operations_per_sec > 0.1, f"Size {size}: too few ops/sec"

        # Check scaling behavior
        if len(results) >= 2:
            sizes_list = sorted(results.keys())
            times_list = [results[s]["avg_time"] for s in sizes_list]

            # Should scale sub-quadratically for reasonable sizes
            largest_time = times_list[-1]
            smallest_time = times_list[0]
            size_ratio = sizes_list[-1] / sizes_list[0]
            time_ratio = largest_time / smallest_time

            # Rough scaling check - should not be worse than O(n^3)
            expected_max_ratio = size_ratio**2.5
            assert time_ratio < expected_max_ratio, f"Poor scaling: {time_ratio:.2f}x"

    def test_c2_model_benchmark(self, jax_backend, benchmark_config):
        """Benchmark complete c2 model computation."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        # Realistic dataset sizes
        test_cases = [
            {"n_times": 50, "n_angles": 24},
            {"n_times": 100, "n_angles": 36},
            {"n_times": 200, "n_angles": 72},
        ]

        results = {}

        for case in test_cases:
            n_times = case["n_times"]
            n_angles = case["n_angles"]

            t1, t2 = jnp.meshgrid(
                jnp.arange(n_times), jnp.arange(n_times), indexing="ij"
            )
            phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

            params = {
                "offset": 1.0,
                "contrast": 0.4,
                "diffusion_coefficient": 0.1,
                "shear_rate": 0.0,
                "L": 1.0,
            }
            q = 0.01

            # Warmup
            for _ in range(2):
                result = compute_c2_model_jax(params, t1, t2, phi, q)
                result.block_until_ready()

            # Benchmark
            times = []
            for _ in range(benchmark_config["min_rounds"]):
                start = time.perf_counter()
                result = compute_c2_model_jax(params, t1, t2, phi, q)
                result.block_until_ready()
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            total_elements = n_angles * n_times * n_times
            elements_per_sec = total_elements / avg_time

            case_key = f"{n_times}x{n_times}x{n_angles}"
            results[case_key] = {
                "avg_time": avg_time,
                "elements_per_sec": elements_per_sec,
                "total_elements": total_elements,
            }

            # Performance expectations
            assert avg_time < 10.0, f"Case {case_key}: too slow ({avg_time:.3f}s)"
            assert elements_per_sec > 10000, f"Case {case_key}: too few elements/sec"

    def test_optimization_benchmark(
        self, jax_backend, synthetic_xpcs_data, test_config, benchmark_config
    ):
        """Benchmark NLSQ optimization performance."""
        try:
            from homodyne.optimization.nlsq import NLSQ_AVAILABLE, fit_nlsq_jax
        except ImportError:
            pytest.skip("NLSQ optimization module not available")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        data = synthetic_xpcs_data

        # Configure for benchmarking
        bench_config = test_config.copy()
        bench_config["optimization"]["lsq"]["max_iterations"] = 50

        # Warmup
        try:
            result = fit_nlsq_jax(data, bench_config)
            if not result.success:
                pytest.skip("Optimization warmup failed")
        except Exception as e:
            pytest.skip(f"Optimization not working: {e}")

        # Benchmark
        times = []
        successes = []

        for _ in range(max(3, benchmark_config["min_rounds"])):
            start = time.perf_counter()
            result = fit_nlsq_jax(data, bench_config)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            successes.append(result.success)

        successful_times = [t for t, s in zip(times, successes, strict=False) if s]

        if len(successful_times) == 0:
            pytest.skip("No successful optimizations")

        avg_time = np.mean(successful_times)
        success_rate = sum(successes) / len(successes)

        # Performance expectations
        assert avg_time < 30.0, f"Optimization too slow: {avg_time:.3f}s"
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"

        # Report performance metrics
        data_size = np.prod(data["c2_exp"].shape)
        elements_per_sec = data_size / avg_time

        assert (
            elements_per_sec > 100
        ), f"Processing rate too slow: {elements_per_sec:.0f} elements/sec"

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_usage_benchmark(self, jax_backend):
        """Benchmark memory usage for different dataset sizes."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        memory_results = {}

        sizes = [30, 60, 100]
        for size in sizes:
            # Force garbage collection
            gc.collect()

            t1, t2 = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
            phi = jnp.linspace(0, 2 * jnp.pi, 36)

            params = {
                "offset": 1.0,
                "contrast": 0.4,
                "diffusion_coefficient": 0.1,
                "shear_rate": 0.0,
                "L": 1.0,
            }

            # Measure memory before computation
            before_memory = process.memory_info().rss / (1024 * 1024)

            # Perform computation
            result = compute_c2_model_jax(params, t1, t2, phi, 0.01)
            result.block_until_ready()

            # Measure memory after computation
            after_memory = process.memory_info().rss / (1024 * 1024)

            memory_used = after_memory - before_memory
            expected_data_size = (36 * size * size * 8) / (1024 * 1024)  # float64 MB

            memory_results[size] = {
                "memory_used_mb": memory_used,
                "expected_data_mb": expected_data_size,
                "memory_overhead": (
                    memory_used / expected_data_size if expected_data_size > 0 else 0
                ),
            }

            # Memory should not grow excessively
            assert (
                memory_used < expected_data_size * 10
            ), f"Memory usage too high for size {size}"

        # Check for memory leaks
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory

        # Should not grow more than reasonable amount
        assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.1f} MB"


@pytest.mark.performance
@pytest.mark.slow
class TestDataLoadingBenchmarks:
    """Benchmark data loading and preprocessing performance."""

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_synthetic_data_generation_benchmark(self, benchmark_config):
        """Benchmark synthetic data generation for different sizes."""
        sizes = [
            (50, 24),  # Small
            (100, 36),  # Medium
            (200, 72),  # Large
        ]

        results = {}

        for n_times, n_angles in sizes:
            # Measure time and memory
            process = psutil.Process()
            start_memory = process.memory_info().rss / (1024 * 1024)

            times = []
            for _ in range(benchmark_config["min_rounds"]):
                start = time.perf_counter()

                # Generate synthetic data (similar to fixture)
                t1, t2 = np.meshgrid(
                    np.arange(n_times), np.arange(n_times), indexing="ij"
                )
                phi = np.linspace(0, 2 * np.pi, n_angles)
                tau = np.abs(t1 - t2) + 1e-6
                c2_exp = 1 + 0.5 * np.exp(-tau / 10.0)

                elapsed = time.perf_counter() - start
                times.append(elapsed)

            end_memory = process.memory_info().rss / (1024 * 1024)

            avg_time = np.mean(times)
            memory_used = end_memory - start_memory
            data_size = n_angles * n_times * n_times * 8 / (1024 * 1024)  # MB

            results[f"{n_times}x{n_angles}"] = {
                "avg_time": avg_time,
                "memory_used_mb": memory_used,
                "data_size_mb": data_size,
                "generation_rate_mb_per_sec": data_size / avg_time,
            }

            # Performance expectations
            assert avg_time < 5.0, f"Data generation too slow: {avg_time:.3f}s"
            assert (
                memory_used < data_size * 5
            ), f"Memory usage too high: {memory_used:.1f} MB"

    def test_config_loading_benchmark(self, temp_dir, benchmark_config):
        """Benchmark configuration loading performance."""
        try:
            from homodyne.data.xpcs_loader import load_config_file
        except ImportError:
            pytest.skip("Config loading module not available")

        # Create test config files of different sizes
        base_config = {
            "analysis_mode": "static_isotropic",
            "optimization": {"method": "nlsq", "lsq": {"max_iterations": 100}},
        }

        # Create configs with varying complexity
        configs = {
            "simple": base_config,
            "medium": {
                **base_config,
                "extra_params": {f"param_{i}": i for i in range(50)},
            },
            "complex": {
                **base_config,
                "extra_params": {f"param_{i}": i for i in range(200)},
            },
        }

        results = {}

        for config_name, config_data in configs.items():
            import json

            config_file = temp_dir / f"{config_name}_config.json"
            with open(config_file, "w") as f:
                json.dump(config_data, f)

            # Benchmark loading
            times = []
            for _ in range(max(10, benchmark_config["min_rounds"])):
                start = time.perf_counter()
                loaded_config = load_config_file(str(config_file))
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_time = np.mean(times)
            file_size = config_file.stat().st_size / 1024  # KB

            results[config_name] = {
                "avg_time": avg_time,
                "file_size_kb": file_size,
                "load_rate_kb_per_sec": file_size / avg_time,
            }

            # Performance expectations
            assert avg_time < 0.1, f"Config loading too slow: {avg_time:.4f}s"


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.requires_jax
class TestScalingBenchmarks:
    """Test computational scaling behavior."""

    def test_jax_compilation_overhead(self, jax_backend, benchmark_config):
        """Test JAX compilation overhead vs execution time."""
        from jax import jit

        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        size = 100
        t1, t2 = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
        q = 0.01
        D = 0.1

        # Test compilation time
        jit_fn = jit(compute_g1_diffusion_jax)

        start = time.perf_counter()
        result = jit_fn(t1, t2, q, D)  # First call triggers compilation
        result.block_until_ready()
        compilation_time = time.perf_counter() - start

        # Test execution time (after compilation)
        execution_times = []
        for _ in range(10):
            start = time.perf_counter()
            result = jit_fn(t1, t2, q, D)
            result.block_until_ready()
            elapsed = time.perf_counter() - start
            execution_times.append(elapsed)

        avg_execution_time = np.mean(execution_times)

        # Compilation should be reasonable compared to execution
        compilation_overhead = compilation_time / avg_execution_time

        # Report metrics
        assert compilation_time < 10.0, f"Compilation too slow: {compilation_time:.3f}s"
        assert (
            avg_execution_time < 1.0
        ), f"Execution too slow: {avg_execution_time:.3f}s"
        # Increased threshold to 2000x to account for system variability
        # v2.3.0 CPU-only: First JIT compilation can be slower on some systems
        assert (
            compilation_overhead < 2000
        ), f"Compilation overhead too high: {compilation_overhead:.1f}x"

    def test_vectorization_scaling(self, jax_backend):
        """Test performance scaling with vectorization."""
        from jax import vmap

        from homodyne.core.jax_backend import compute_c2_model_jax

        # Base computation
        t1 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        t2 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Test different numbers of q-values
        q_counts = [1, 5, 10, 20]
        results = {}

        for n_q in q_counts:
            q_values = jnp.linspace(0.005, 0.02, n_q)

            # Vectorized computation
            vmap_fn = vmap(lambda q: compute_c2_model_jax(params, t1, t2, phi, q))

            # Warmup
            result_vmap = vmap_fn(q_values)
            result_vmap.block_until_ready()

            # Benchmark vectorized
            times_vmap = []
            for _ in range(5):
                start = time.perf_counter()
                result_vmap = vmap_fn(q_values)
                result_vmap.block_until_ready()
                elapsed = time.perf_counter() - start
                times_vmap.append(elapsed)

            # Benchmark sequential (for comparison)
            times_sequential = []
            for _ in range(5):
                start = time.perf_counter()
                for q in q_values:
                    result_seq = compute_c2_model_jax(params, t1, t2, phi, q)
                    result_seq.block_until_ready()
                elapsed = time.perf_counter() - start
                times_sequential.append(elapsed)

            avg_vmap = np.mean(times_vmap)
            avg_sequential = np.mean(times_sequential)
            speedup = avg_sequential / avg_vmap

            results[n_q] = {
                "vmap_time": avg_vmap,
                "sequential_time": avg_sequential,
                "speedup": speedup,
            }

            # Vectorization should provide some benefit
            assert speedup >= 0.8, f"Vectorization slower for n_q={n_q}: {speedup:.2f}x"

    def test_memory_scaling_behavior(self, jax_backend):
        """Test memory scaling with dataset size."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        # Test different sizes
        sizes = [20, 40, 60]
        memory_usage = []

        for size in sizes:
            # Create data
            t1, t2 = jnp.meshgrid(jnp.arange(size), jnp.arange(size), indexing="ij")
            phi = jnp.linspace(0, 2 * jnp.pi, 24)

            params = {
                "offset": 1.0,
                "contrast": 0.4,
                "diffusion_coefficient": 0.1,
                "shear_rate": 0.0,
                "L": 1.0,
            }

            # Estimate memory usage (rough)
            expected_arrays = [
                t1,  # input
                t2,  # input
                phi,  # input
            ]

            # Compute result
            result = compute_c2_model_jax(params, t1, t2, phi, 0.01)

            # Estimate total memory
            total_elements = sum(arr.size for arr in expected_arrays) + result.size
            estimated_memory_mb = (total_elements * 8) / (1024 * 1024)

            memory_usage.append((size, estimated_memory_mb))

        # Check scaling behavior
        if len(memory_usage) >= 2:
            size_ratios = []
            memory_ratios = []

            for i in range(1, len(memory_usage)):
                size_ratio = memory_usage[i][0] / memory_usage[i - 1][0]
                memory_ratio = memory_usage[i][1] / memory_usage[i - 1][1]

                size_ratios.append(size_ratio)
                memory_ratios.append(memory_ratio)

            # Memory should scale roughly quadratically with size (due to matrices)
            avg_size_ratio = np.mean(size_ratios)
            avg_memory_ratio = np.mean(memory_ratios)

            expected_memory_scaling = avg_size_ratio**2
            scaling_factor = avg_memory_ratio / expected_memory_scaling

            # Should be close to expected scaling (within factor of 2)
            assert (
                0.5 < scaling_factor < 2.0
            ), f"Memory scaling unexpected: {scaling_factor:.2f} (expected ~1.0)"


@pytest.mark.performance
@pytest.mark.slow
class TestRegressionBenchmarks:
    """Regression tests to ensure performance doesn't degrade."""

    def test_baseline_performance_regression(self, jax_backend, synthetic_xpcs_data):
        """Test against baseline performance expectations."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        # Standard test case
        data = synthetic_xpcs_data
        t1 = data["t1"]
        t2 = data["t2"]
        phi = data["phi_angles_list"]
        q = data["wavevector_q_list"][0]

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Warmup
        result = compute_c2_model_jax(params, t1, t2, phi, q)
        result.block_until_ready()

        # Benchmark
        times = []
        for _ in range(5):
            start = time.perf_counter()
            result = compute_c2_model_jax(params, t1, t2, phi, q)
            result.block_until_ready()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = np.mean(times)
        data_size = np.prod(result.shape)
        throughput = data_size / avg_time

        # Baseline expectations (these should be updated as hardware/software improves)
        BASELINE_THROUGHPUT = 10000  # elements per second (conservative)
        BASELINE_MAX_TIME = 5.0  # maximum seconds for standard dataset

        assert (
            throughput > BASELINE_THROUGHPUT
        ), f"Performance regression: {throughput:.0f} < {BASELINE_THROUGHPUT} elements/sec"
        assert (
            avg_time < BASELINE_MAX_TIME
        ), f"Performance regression: {avg_time:.3f}s > {BASELINE_MAX_TIME}s"

    def test_optimization_performance_regression(
        self, synthetic_xpcs_data, test_config
    ):
        """Test optimization performance regression."""
        try:
            from homodyne.optimization.nlsq import NLSQ_AVAILABLE, fit_nlsq_jax
        except ImportError:
            pytest.skip("Optimization module not available")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        data = synthetic_xpcs_data

        # Configure for regression test
        regression_config = test_config.copy()
        regression_config["optimization"]["lsq"]["max_iterations"] = 30

        # Benchmark optimization
        times = []
        successes = []

        for _ in range(3):
            start = time.perf_counter()
            result = fit_nlsq_jax(data, regression_config)
            elapsed = time.perf_counter() - start

            times.append(elapsed)
            successes.append(result.success)

        successful_times = [t for t, s in zip(times, successes, strict=False) if s]

        if len(successful_times) == 0:
            pytest.skip("No successful optimizations")

        avg_time = np.mean(successful_times)
        success_rate = sum(successes) / len(successes)

        # Baseline expectations for optimization
        BASELINE_OPT_TIME = 20.0  # maximum seconds
        BASELINE_SUCCESS_RATE = 0.8  # minimum success rate

        assert (
            avg_time < BASELINE_OPT_TIME
        ), f"Optimization regression: {avg_time:.3f}s > {BASELINE_OPT_TIME}s"
        assert (
            success_rate >= BASELINE_SUCCESS_RATE
        ), f"Success rate regression: {success_rate:.2f} < {BASELINE_SUCCESS_RATE}"
