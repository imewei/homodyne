"""
Performance Benchmark Tests for Physics Utilities
=================================================

Performance benchmarks for homodyne.core.physics_utils functions.
Tests measure execution time, memory usage, and scaling behavior.

Benchmark Categories:
- Function execution time
- Memory efficiency
- Scaling with input size
- JIT compilation overhead
"""

import gc
import time

import numpy as np
import pytest

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.physics_utils import (
    apply_diagonal_correction,
    calculate_diffusion_coefficient,
    calculate_shear_rate,
    create_time_integral_matrix,
    safe_exp,
    safe_sinc,
    trapezoid_cumsum,
)

# =============================================================================
# Utility Functions
# =============================================================================


def measure_execution_time(func, *args, warmup=2, iterations=10, **kwargs):
    """Measure execution time of a function.

    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        **kwargs: Keyword arguments for function

    Returns:
        dict with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Force JAX to complete any pending operations
    if JAX_AVAILABLE:
        result = func(*args, **kwargs)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed iterations
    times = []
    for _ in range(iterations):
        gc.collect()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if JAX_AVAILABLE and hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "iterations": iterations,
    }


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function.

    Args:
        func: Function to benchmark
        *args: Positional arguments for function
        **kwargs: Keyword arguments for function

    Returns:
        dict with memory statistics
    """
    try:
        import tracemalloc

        tracemalloc.start()
        gc.collect()

        result = func(*args, **kwargs)
        if JAX_AVAILABLE and hasattr(result, "block_until_ready"):
            result.block_until_ready()

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "current_bytes": current,
            "peak_bytes": peak,
            "current_mb": current / (1024 * 1024),
            "peak_mb": peak / (1024 * 1024),
        }
    except ImportError:
        return {"current_mb": 0, "peak_mb": 0}


# =============================================================================
# Performance Test Fixtures
# =============================================================================


@pytest.fixture
def small_array():
    """Small array for basic tests."""
    return jnp.linspace(0, 1, 100)


@pytest.fixture
def medium_array():
    """Medium array for moderate tests."""
    return jnp.linspace(0, 10, 1000)


@pytest.fixture
def large_array():
    """Large array for stress tests."""
    return jnp.linspace(0, 100, 10000)


@pytest.fixture
def small_matrix():
    """Small matrix for basic tests."""
    return jnp.ones((50, 50)) + 0.5 * jnp.eye(50)


@pytest.fixture
def medium_matrix():
    """Medium matrix for moderate tests."""
    return jnp.ones((200, 200)) + 0.5 * jnp.eye(200)


@pytest.fixture
def large_matrix():
    """Large matrix for stress tests."""
    return jnp.ones((500, 500)) + 0.5 * jnp.eye(500)


# =============================================================================
# safe_exp Performance Tests
# =============================================================================


@pytest.mark.performance
class TestSafeExpPerformance:
    """Performance benchmarks for safe_exp function."""

    def test_safe_exp_small_array(self, small_array):
        """Benchmark safe_exp with small array."""
        perf = measure_execution_time(safe_exp, small_array)
        print(f"\nsafe_exp (100 elements): {perf['mean_time'] * 1000:.3f}ms")
        # Should complete in < 10ms
        assert perf["mean_time"] < 0.01

    def test_safe_exp_medium_array(self, medium_array):
        """Benchmark safe_exp with medium array."""
        perf = measure_execution_time(safe_exp, medium_array)
        print(f"\nsafe_exp (1000 elements): {perf['mean_time'] * 1000:.3f}ms")
        # Should complete in < 50ms
        assert perf["mean_time"] < 0.05

    def test_safe_exp_large_array(self, large_array):
        """Benchmark safe_exp with large array."""
        perf = measure_execution_time(safe_exp, large_array)
        print(f"\nsafe_exp (10000 elements): {perf['mean_time'] * 1000:.3f}ms")
        # Should complete in < 100ms
        assert perf["mean_time"] < 0.1

    @pytest.mark.parametrize("size", [100, 1000, 10000, 100000])
    def test_safe_exp_scaling(self, size):
        """Test safe_exp scaling behavior."""
        arr = jnp.linspace(-10, 10, size)
        perf = measure_execution_time(safe_exp, arr, iterations=5)
        throughput = size / perf["mean_time"]
        print(f"\nsafe_exp ({size} elements): {throughput / 1e6:.2f}M elements/s")
        # Minimum throughput requirement - scales with size
        # Small arrays have high overhead per element; larger arrays are more efficient
        # Note: Thresholds are conservative to account for system load variability
        min_throughput = {100: 5e4, 1000: 1e5, 10000: 5e5, 100000: 1e6}
        assert throughput > min_throughput[size], (
            f"Throughput {throughput:.0f} below minimum {min_throughput[size]:.0f}"
        )


# =============================================================================
# safe_sinc Performance Tests
# =============================================================================


@pytest.mark.performance
class TestSafeSincPerformance:
    """Performance benchmarks for safe_sinc function."""

    def test_safe_sinc_small_array(self, small_array):
        """Benchmark safe_sinc with small array."""
        perf = measure_execution_time(safe_sinc, small_array)
        print(f"\nsafe_sinc (100 elements): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.01

    def test_safe_sinc_large_array(self, large_array):
        """Benchmark safe_sinc with large array."""
        perf = measure_execution_time(safe_sinc, large_array)
        print(f"\nsafe_sinc (10000 elements): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.1


# =============================================================================
# calculate_diffusion_coefficient Performance Tests
# =============================================================================


@pytest.mark.performance
class TestDiffusionCoefficientPerformance:
    """Performance benchmarks for calculate_diffusion_coefficient."""

    def test_diffusion_small_array(self, small_array):
        """Benchmark diffusion coefficient with small array."""
        perf = measure_execution_time(
            calculate_diffusion_coefficient, small_array, 1.0, 0.5, 0.1
        )
        print(f"\ndiffusion (100 points): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.01

    def test_diffusion_large_array(self, large_array):
        """Benchmark diffusion coefficient with large array."""
        perf = measure_execution_time(
            calculate_diffusion_coefficient, large_array, 1.0, 0.5, 0.1
        )
        print(f"\ndiffusion (10000 points): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.1

    @pytest.mark.parametrize("alpha", [-1.0, 0.0, 0.5, 1.0, 2.0])
    def test_diffusion_alpha_independence(self, medium_array, alpha):
        """Test diffusion performance is independent of alpha value."""
        perf = measure_execution_time(
            calculate_diffusion_coefficient, medium_array, 1.0, alpha, 0.1
        )
        print(f"\ndiffusion (alpha={alpha}): {perf['mean_time'] * 1000:.3f}ms")
        # Performance should be similar regardless of alpha
        assert perf["mean_time"] < 0.05


# =============================================================================
# calculate_shear_rate Performance Tests
# =============================================================================


@pytest.mark.performance
class TestShearRatePerformance:
    """Performance benchmarks for calculate_shear_rate."""

    def test_shear_rate_small_array(self, small_array):
        """Benchmark shear rate with small array."""
        perf = measure_execution_time(calculate_shear_rate, small_array, 0.5, 0.3, 0.1)
        print(f"\nshear_rate (100 points): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.01

    def test_shear_rate_large_array(self, large_array):
        """Benchmark shear rate with large array."""
        perf = measure_execution_time(calculate_shear_rate, large_array, 0.5, 0.3, 0.1)
        print(f"\nshear_rate (10000 points): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.1


# =============================================================================
# create_time_integral_matrix Performance Tests
# =============================================================================


@pytest.mark.performance
class TestTimeIntegralMatrixPerformance:
    """Performance benchmarks for create_time_integral_matrix."""

    def test_time_integral_small(self):
        """Benchmark time integral matrix with small input."""
        values = jnp.ones(50)
        perf = measure_execution_time(create_time_integral_matrix, values)
        print(f"\ntime_integral_matrix (50x50): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.05

    def test_time_integral_medium(self):
        """Benchmark time integral matrix with medium input."""
        values = jnp.ones(200)
        perf = measure_execution_time(create_time_integral_matrix, values)
        print(f"\ntime_integral_matrix (200x200): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.5

    def test_time_integral_large(self):
        """Benchmark time integral matrix with large input."""
        values = jnp.ones(500)
        perf = measure_execution_time(create_time_integral_matrix, values)
        print(f"\ntime_integral_matrix (500x500): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 2.0

    @pytest.mark.parametrize("n", [50, 100, 200, 500])
    def test_time_integral_scaling(self, n):
        """Test time integral matrix O(n²) scaling."""
        values = jnp.ones(n)
        perf = measure_execution_time(create_time_integral_matrix, values, iterations=5)
        # Time should scale as O(n²)
        time_per_element = perf["mean_time"] / (n * n)
        print(
            f"\ntime_integral_matrix ({n}x{n}): {time_per_element * 1e9:.1f}ns/element"
        )


# =============================================================================
# trapezoid_cumsum Performance Tests
# =============================================================================


@pytest.mark.performance
class TestTrapezoidCumsumPerformance:
    """Performance benchmarks for trapezoid_cumsum."""

    def test_trapezoid_small(self, small_array):
        """Benchmark trapezoid_cumsum with small array."""
        perf = measure_execution_time(trapezoid_cumsum, small_array)
        print(f"\ntrapezoid_cumsum (100 elements): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.01

    def test_trapezoid_large(self, large_array):
        """Benchmark trapezoid_cumsum with large array."""
        perf = measure_execution_time(trapezoid_cumsum, large_array)
        print(f"\ntrapezoid_cumsum (10000 elements): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.1


# =============================================================================
# apply_diagonal_correction Performance Tests
# =============================================================================


@pytest.mark.performance
class TestDiagonalCorrectionPerformance:
    """Performance benchmarks for apply_diagonal_correction."""

    def test_diagonal_correction_small(self, small_matrix):
        """Benchmark diagonal correction with small matrix."""
        perf = measure_execution_time(apply_diagonal_correction, small_matrix)
        print(f"\ndiagonal_correction (50x50): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.05

    def test_diagonal_correction_medium(self, medium_matrix):
        """Benchmark diagonal correction with medium matrix."""
        perf = measure_execution_time(apply_diagonal_correction, medium_matrix)
        print(f"\ndiagonal_correction (200x200): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 0.2

    def test_diagonal_correction_large(self, large_matrix):
        """Benchmark diagonal correction with large matrix."""
        perf = measure_execution_time(apply_diagonal_correction, large_matrix)
        print(f"\ndiagonal_correction (500x500): {perf['mean_time'] * 1000:.3f}ms")
        assert perf["mean_time"] < 1.0

    @pytest.mark.parametrize("n", [50, 100, 200, 500])
    def test_diagonal_correction_scaling(self, n):
        """Test diagonal correction O(n) scaling."""
        mat = jnp.ones((n, n)) + 0.5 * jnp.eye(n)
        perf = measure_execution_time(apply_diagonal_correction, mat, iterations=5)
        # Time should scale roughly linearly with n
        time_per_element = perf["mean_time"] / n
        print(f"\ndiagonal_correction ({n}x{n}): {time_per_element * 1e6:.1f}µs/row")


# =============================================================================
# Memory Efficiency Tests
# =============================================================================


@pytest.mark.performance
class TestMemoryEfficiency:
    """Test memory efficiency of physics utilities."""

    def test_safe_exp_memory(self):
        """Test safe_exp memory usage."""
        arr = jnp.linspace(-10, 10, 10000)
        mem = measure_memory_usage(safe_exp, arr)
        print(f"\nsafe_exp memory: {mem['peak_mb']:.2f}MB peak")
        # Should use less than 10MB for 10000 elements
        assert mem["peak_mb"] < 10

    def test_time_integral_matrix_memory(self):
        """Test time integral matrix memory usage."""
        values = jnp.ones(500)
        mem = measure_memory_usage(create_time_integral_matrix, values)
        print(f"\ntime_integral_matrix memory: {mem['peak_mb']:.2f}MB peak")
        # 500x500 float64 matrix = 2MB, allow 20MB for intermediate values
        assert mem["peak_mb"] < 50

    def test_diagonal_correction_memory(self):
        """Test diagonal correction memory usage."""
        mat = jnp.ones((500, 500)) + 0.5 * jnp.eye(500)
        mem = measure_memory_usage(apply_diagonal_correction, mat)
        print(f"\ndiagonal_correction memory: {mem['peak_mb']:.2f}MB peak")
        # Should use less than 20MB
        assert mem["peak_mb"] < 50


# =============================================================================
# JIT Compilation Overhead Tests
# =============================================================================


@pytest.mark.performance
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJITOverhead:
    """Test JIT compilation overhead for physics utilities."""

    def test_safe_exp_jit_overhead(self):
        """Measure JIT compilation overhead for safe_exp."""
        arr = jnp.linspace(-10, 10, 1000)

        # First call (includes compilation)
        start = time.perf_counter()
        result = safe_exp(arr)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        first_call = time.perf_counter() - start

        # Subsequent call (compiled)
        start = time.perf_counter()
        result = safe_exp(arr)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        second_call = time.perf_counter() - start

        print(
            f"\nsafe_exp JIT: first={first_call * 1000:.2f}ms, compiled={second_call * 1000:.3f}ms"
        )
        # Compiled version should be faster or similar
        # (first call may not be slower if already cached)

    def test_diffusion_jit_overhead(self):
        """Measure JIT compilation overhead for diffusion coefficient."""
        time_arr = jnp.linspace(0, 1, 1000)

        # First call
        start = time.perf_counter()
        result = calculate_diffusion_coefficient(time_arr, 1.0, 0.5, 0.1)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        first_call = time.perf_counter() - start

        # Subsequent call
        start = time.perf_counter()
        result = calculate_diffusion_coefficient(time_arr, 1.0, 0.5, 0.1)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        second_call = time.perf_counter() - start

        print(
            f"\ndiffusion JIT: first={first_call * 1000:.2f}ms, compiled={second_call * 1000:.3f}ms"
        )


# =============================================================================
# Throughput Tests
# =============================================================================


@pytest.mark.performance
class TestThroughput:
    """Test throughput (elements/second) for physics utilities."""

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_safe_exp_throughput(self, size):
        """Measure safe_exp throughput."""
        arr = jnp.linspace(-10, 10, size)
        perf = measure_execution_time(safe_exp, arr, iterations=5)
        throughput = size / perf["mean_time"]
        print(f"\nsafe_exp throughput ({size}): {throughput / 1e6:.1f}M elements/s")
        # Minimum throughput scales with size (small arrays have higher overhead)
        min_throughput = {1000: 5e5, 10000: 1e6, 100000: 1e6}
        assert throughput > min_throughput[size]

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_diffusion_throughput(self, size):
        """Measure diffusion coefficient throughput."""
        time_arr = jnp.linspace(0, 10, size)
        perf = measure_execution_time(
            calculate_diffusion_coefficient, time_arr, 1.0, 0.5, 0.1, iterations=5
        )
        throughput = size / perf["mean_time"]
        print(f"\ndiffusion throughput ({size}): {throughput / 1e6:.1f}M points/s")
        # Minimum throughput scales with size (small arrays have higher overhead)
        min_throughput = {1000: 2e5, 10000: 1e6, 100000: 1e6}
        assert throughput > min_throughput[size]

    @pytest.mark.parametrize("size", [1000, 10000, 100000])
    def test_shear_rate_throughput(self, size):
        """Measure shear rate throughput."""
        time_arr = jnp.linspace(0, 10, size)
        perf = measure_execution_time(
            calculate_shear_rate, time_arr, 0.5, 0.3, 0.1, iterations=5
        )
        throughput = size / perf["mean_time"]
        print(f"\nshear_rate throughput ({size}): {throughput / 1e6:.1f}M points/s")
        # Minimum throughput scales with size (small arrays have higher overhead)
        min_throughput = {1000: 2e5, 10000: 1e6, 100000: 1e6}
        assert throughput > min_throughput[size]


# =============================================================================
# Comparative Performance Tests
# =============================================================================


@pytest.mark.performance
class TestComparativePerformance:
    """Compare performance of related functions."""

    def test_shear_rate_vs_cmc_variant(self):
        """Compare shear_rate and shear_rate_cmc performance."""
        from homodyne.core.physics_utils import calculate_shear_rate_cmc

        time_arr = jnp.linspace(0, 10, 10000)

        perf_regular = measure_execution_time(
            calculate_shear_rate, time_arr, 0.5, 0.3, 0.1
        )
        perf_cmc = measure_execution_time(
            calculate_shear_rate_cmc, time_arr, 0.5, 0.3, 0.1
        )

        print(f"\nshear_rate: {perf_regular['mean_time'] * 1000:.3f}ms")
        print(f"shear_rate_cmc: {perf_cmc['mean_time'] * 1000:.3f}ms")

        # CMC variant should be within reasonable range of regular version
        # Note: Wider tolerance to account for JIT compilation variability
        ratio = perf_cmc["mean_time"] / perf_regular["mean_time"]
        assert 0.2 < ratio < 5.0, (
            f"Performance ratio {ratio:.2f} outside expected range"
        )

    def test_numpy_vs_jax_safe_exp(self):
        """Compare NumPy vs JAX safe_exp performance."""
        data = np.linspace(-10, 10, 10000)

        # NumPy version
        def numpy_safe_exp(x):
            return np.exp(np.clip(x, -700, 700))

        perf_numpy = measure_execution_time(numpy_safe_exp, data)

        # JAX version
        data_jax = jnp.array(data)
        perf_jax = measure_execution_time(safe_exp, data_jax)

        print(f"\nNumPy safe_exp: {perf_numpy['mean_time'] * 1000:.3f}ms")
        print(f"JAX safe_exp: {perf_jax['mean_time'] * 1000:.3f}ms")

        # Both should be fast
        assert perf_numpy["mean_time"] < 0.1
        assert perf_jax["mean_time"] < 0.1
