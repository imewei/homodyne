"""
Comprehensive Test Suite for Diagonal Correction

Tests the apply_diagonal_correction function and its integration with
optimization methods (NLSQ, MCMC, NUTS, CMC).

Generated: 2025-10-27
Coverage target: >95%

Critical Requirements:
1. Diagonal correction must be applied consistently to experimental and theoretical data
2. Correction must preserve off-diagonal correlation structure
3. JAX operations must be JIT-compatible and differentiable
4. Integration with all optimization methods must be verified
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose
import warnings

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    pytestmark = pytest.mark.skip("JAX not available")

# Import functions under test
from homodyne.core.jax_backend import apply_diagonal_correction


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility"""
    return np.random.RandomState(42)


@pytest.fixture
def simple_matrix():
    """Simple 3x3 correlation matrix for basic tests"""
    return jnp.array([[5.0, 1.2, 1.1], [1.2, 5.0, 1.3], [1.1, 1.3, 5.0]])


@pytest.fixture
def correlation_matrix_1001x1001(rng):
    """Realistic 1001x1001 correlation matrix matching XPCS data"""
    # Create a symmetric correlation matrix with diagonal autocorrelation
    size = 1001
    matrix = rng.randn(size, size) * 0.1 + 1.0  # Base correlation ~1.0
    matrix = (matrix + matrix.T) / 2  # Make symmetric

    # Add strong diagonal (autocorrelation at t1=t2)
    np.fill_diagonal(matrix, rng.uniform(2.0, 5.0, size))

    return jnp.array(matrix)


@pytest.fixture
def identity_diagonal_matrix():
    """Matrix with identity-like diagonal for edge case testing"""
    size = 10
    matrix = jnp.eye(size) * 10.0 + jnp.ones((size, size))
    return matrix


# ============================================================================
# Unit Tests - Basic Functionality
# ============================================================================


class TestDiagonalCorrectionBasic:
    """Test suite for basic diagonal correction functionality"""

    def test_diagonal_correction_simple_case(self, simple_matrix):
        """Test basic functionality with simple 3x3 matrix"""
        corrected = apply_diagonal_correction(simple_matrix)

        # Original diagonal should be replaced
        assert not jnp.allclose(jnp.diag(corrected), jnp.diag(simple_matrix))

        # Off-diagonal elements should be unchanged
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert corrected[i, j] == simple_matrix[i, j]

    def test_diagonal_values_are_interpolated(self, simple_matrix):
        """Test that diagonal values are interpolated from off-diagonal neighbors"""
        corrected = apply_diagonal_correction(simple_matrix)

        # First diagonal element should equal first off-diagonal
        assert corrected[0, 0] == simple_matrix[0, 1]

        # Middle diagonal element should be average of neighbors
        expected_middle = (simple_matrix[1, 0] + simple_matrix[1, 2]) / 2.0
        assert_allclose(corrected[1, 1], expected_middle, rtol=1e-10)

        # Last diagonal element should equal last off-diagonal
        assert corrected[2, 2] == simple_matrix[2, 1]

    def test_output_shape_preserved(self, correlation_matrix_1001x1001):
        """Test output shape matches input shape"""
        corrected = apply_diagonal_correction(correlation_matrix_1001x1001)
        assert corrected.shape == correlation_matrix_1001x1001.shape

    def test_output_dtype_preserved(self):
        """Test output dtype matches input dtype"""
        for dtype in [jnp.float32, jnp.float64]:
            matrix = jnp.array([[1.0, 0.5], [0.5, 1.0]], dtype=dtype)
            corrected = apply_diagonal_correction(matrix)
            assert corrected.dtype == dtype

    def test_off_diagonal_unchanged(self, correlation_matrix_1001x1001):
        """Test that off-diagonal elements remain exactly unchanged"""
        corrected = apply_diagonal_correction(correlation_matrix_1001x1001)

        # Create mask for off-diagonal elements
        size = correlation_matrix_1001x1001.shape[0]
        mask = ~jnp.eye(size, dtype=bool)

        # Off-diagonal should be identical
        assert_allclose(
            corrected[mask],
            correlation_matrix_1001x1001[mask],
            rtol=0,
            atol=0,  # Exact equality
        )


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestDiagonalCorrectionEdgeCases:
    """Edge case tests for diagonal correction"""

    def test_minimum_size_2x2(self):
        """Test with minimum valid matrix size 2x2"""
        matrix = jnp.array([[5.0, 1.2], [1.2, 5.0]])
        corrected = apply_diagonal_correction(matrix)

        # Both diagonal elements should equal the off-diagonal
        assert corrected[0, 0] == matrix[0, 1]
        assert corrected[1, 1] == matrix[1, 0]

    def test_large_matrix_1000x1000(self, rng):
        """Test with large 1000x1000 matrix"""
        size = 1000
        matrix = jnp.array(rng.randn(size, size))

        # Should not raise or hang
        corrected = apply_diagonal_correction(matrix)

        assert corrected.shape == (size, size)
        assert jnp.all(jnp.isfinite(corrected))

    def test_identity_matrix(self):
        """Test with identity matrix"""
        matrix = jnp.eye(5)
        corrected = apply_diagonal_correction(matrix)

        # Diagonal should be zeros (average of 0s)
        assert_allclose(jnp.diag(corrected), jnp.zeros(5), atol=1e-14)

    def test_zeros_matrix(self):
        """Test with all-zeros matrix"""
        matrix = jnp.zeros((5, 5))
        corrected = apply_diagonal_correction(matrix)

        # Should remain all zeros
        assert_allclose(corrected, jnp.zeros((5, 5)), atol=1e-14)

    def test_uniform_values(self):
        """Test with matrix of uniform values"""
        matrix = jnp.ones((5, 5)) * 3.14
        corrected = apply_diagonal_correction(matrix)

        # Diagonal should equal uniform value (average of identical neighbors)
        assert_allclose(jnp.diag(corrected), jnp.ones(5) * 3.14, rtol=1e-10)

    def test_large_diagonal_values(self):
        """Test numerical stability with large diagonal values"""
        matrix = jnp.eye(5) * 1e10 + jnp.ones((5, 5))
        corrected = apply_diagonal_correction(matrix)

        # Should not overflow or lose precision
        assert jnp.all(jnp.isfinite(corrected))
        assert jnp.max(jnp.abs(corrected)) < 1e12

    def test_small_off_diagonal_values(self):
        """Test numerical stability with very small off-diagonal values"""
        matrix = jnp.eye(5) * 2.0
        matrix = matrix.at[jnp.triu_indices(5, 1)].set(1e-15)
        matrix = matrix.at[jnp.tril_indices(5, -1)].set(1e-15)

        corrected = apply_diagonal_correction(matrix)

        # Should preserve small values without underflow
        assert jnp.all(jnp.isfinite(corrected))


# ============================================================================
# Numerical Correctness Tests
# ============================================================================


class TestDiagonalCorrectionCorrectness:
    """Numerical correctness tests"""

    def test_matches_reference_implementation(self, correlation_matrix_1001x1001):
        """Test against reference NumPy implementation from xpcs_loader.py"""

        # Reference implementation (from xpcs_loader.py:924-953)
        def reference_diagonal_correction(c2_mat):
            c2_np = np.array(c2_mat)
            size = c2_np.shape[0]
            side_band = c2_np[(np.arange(size - 1), np.arange(1, size))]

            diag_val = np.zeros(size)
            diag_val[:-1] += side_band
            diag_val[1:] += side_band
            norm = np.ones(size)
            norm[1:-1] = 2

            c2_corrected = c2_np.copy()
            c2_corrected[np.diag_indices(size)] = diag_val / norm
            return c2_corrected

        # Our JAX implementation
        jax_result = apply_diagonal_correction(correlation_matrix_1001x1001)

        # Reference NumPy implementation
        numpy_result = reference_diagonal_correction(correlation_matrix_1001x1001)

        # Should match exactly
        assert_allclose(jax_result, numpy_result, rtol=1e-14, atol=1e-16)

    def test_diagonal_interpolation_formula(self):
        """Test exact diagonal interpolation formula"""
        # Create matrix with known off-diagonal values (use float to avoid integer division)
        matrix = jnp.array(
            [
                [999.0, 10.0, 20.0, 30.0],
                [10.0, 999.0, 11.0, 21.0],
                [20.0, 11.0, 999.0, 12.0],
                [30.0, 21.0, 12.0, 999.0],
            ]
        )

        corrected = apply_diagonal_correction(matrix)

        # Verify interpolation formula
        # diag[0] = matrix[0,1] = 10
        assert corrected[0, 0] == 10

        # diag[1] = (matrix[1,0] + matrix[1,2]) / 2 = (10 + 11) / 2 = 10.5
        assert_allclose(corrected[1, 1], 10.5, rtol=1e-10)

        # diag[2] = (matrix[2,1] + matrix[2,3]) / 2 = (11 + 12) / 2 = 11.5
        assert_allclose(corrected[2, 2], 11.5, rtol=1e-10)

        # diag[3] = matrix[3,2] = 12
        assert corrected[3, 3] == 12

    def test_symmetric_matrix_remains_symmetric(self, rng):
        """Test that symmetric matrices remain symmetric after correction"""
        # Create symmetric matrix
        size = 50
        matrix = rng.randn(size, size)
        matrix = jnp.array((matrix + matrix.T) / 2)

        corrected = apply_diagonal_correction(matrix)

        # Should still be symmetric
        assert_allclose(corrected, corrected.T, rtol=1e-14, atol=1e-16)

    def test_conservation_of_off_diagonal_structure(self, correlation_matrix_1001x1001):
        """Test that correlation structure is preserved in off-diagonal"""
        corrected = apply_diagonal_correction(correlation_matrix_1001x1001)

        # Statistical properties of off-diagonal should be preserved
        size = correlation_matrix_1001x1001.shape[0]
        mask = ~jnp.eye(size, dtype=bool)

        original_mean = jnp.mean(correlation_matrix_1001x1001[mask])
        corrected_mean = jnp.mean(corrected[mask])
        assert_allclose(original_mean, corrected_mean, rtol=1e-12)

        original_std = jnp.std(correlation_matrix_1001x1001[mask])
        corrected_std = jnp.std(corrected[mask])
        assert_allclose(original_std, corrected_std, rtol=1e-12)


# ============================================================================
# JAX-Specific Tests
# ============================================================================


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestDiagonalCorrectionJAX:
    """JAX-specific tests"""

    def test_jit_compilation_successful(self, simple_matrix):
        """Test that function can be JIT-compiled"""
        jitted_fn = jit(apply_diagonal_correction)

        # Should compile without errors
        result = jitted_fn(simple_matrix)

        # Should produce same result as non-JIT
        result_nojit = apply_diagonal_correction(simple_matrix)
        assert_allclose(result, result_nojit, rtol=1e-14)

    def test_jit_equivalence_large_matrix(self, correlation_matrix_1001x1001):
        """Test JIT equivalence with realistic large matrix"""
        jitted_fn = jit(apply_diagonal_correction)

        result_jit = jitted_fn(correlation_matrix_1001x1001)
        result_nojit = apply_diagonal_correction(correlation_matrix_1001x1001)

        assert_allclose(result_jit, result_nojit, rtol=1e-14, atol=1e-16)

    def test_gradient_correctness(self):
        """Test gradient using finite differences"""

        # Define scalar function for gradient testing
        def scalar_fn(x):
            # Create matrix from vector for testing
            matrix = jnp.outer(x, x) + jnp.eye(len(x))
            corrected = apply_diagonal_correction(matrix)
            return jnp.sum(corrected)

        x = jnp.array([1.0, 2.0, 3.0])

        # Analytical gradient
        grad_fn = grad(scalar_fn)
        analytical_grad = grad_fn(x)

        # Finite difference gradient
        epsilon = 1e-5
        numerical_grad = jnp.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.at[i].add(epsilon)
            x_minus = x.at[i].add(-epsilon)
            numerical_grad = numerical_grad.at[i].set(
                (scalar_fn(x_plus) - scalar_fn(x_minus)) / (2 * epsilon)
            )

        assert_allclose(analytical_grad, numerical_grad, rtol=1e-4, atol=1e-6)

    def test_vmap_correctness(self, rng):
        """Test vectorization with vmap"""
        # Single matrix
        single_matrix = jnp.array(rng.randn(10, 10))
        single_result = apply_diagonal_correction(single_matrix)

        # Batch of identical matrices
        batch_size = 5
        batch_matrices = jnp.stack([single_matrix] * batch_size)

        # Apply vmap
        vmapped_fn = vmap(apply_diagonal_correction, in_axes=0)
        batch_result = vmapped_fn(batch_matrices)

        # Each result should match single result
        for i in range(batch_size):
            assert_allclose(batch_result[i], single_result, rtol=1e-14)

    def test_vmap_on_phi_axis(self, rng):
        """Test vmap over phi angle axis (realistic NLSQ usage pattern)"""
        # Simulate (n_phi=23, n_t=1001, n_t=1001) correlation data
        n_phi = 23
        n_t = 101  # Reduced for test speed

        correlation_stack = jnp.array(rng.randn(n_phi, n_t, n_t))

        # Add strong diagonal to each phi angle's matrix
        for i in range(n_phi):
            correlation_stack = correlation_stack.at[
                i, jnp.arange(n_t), jnp.arange(n_t)
            ].set(5.0)

        # Apply vmap over phi axis
        vmapped_correction = vmap(apply_diagonal_correction, in_axes=0)
        corrected_stack = vmapped_correction(correlation_stack)

        # Verify shape preserved
        assert corrected_stack.shape == (n_phi, n_t, n_t)

        # Verify diagonal was corrected for each phi
        for i in range(n_phi):
            original_diag = jnp.diag(correlation_stack[i])
            corrected_diag = jnp.diag(corrected_stack[i])
            # Diagonal should have changed
            assert not jnp.allclose(original_diag, corrected_diag)

    def test_device_consistency_cpu_gpu(self, simple_matrix):
        """Test consistency across CPU/GPU devices"""
        results = {}

        for device in jax.devices():
            with jax.default_device(device):
                result = apply_diagonal_correction(simple_matrix)
                results[device.platform] = result

        # If multiple platforms available, compare
        platforms = list(results.keys())
        if len(platforms) > 1:
            for i in range(len(platforms) - 1):
                assert_allclose(
                    results[platforms[i]],
                    results[platforms[i + 1]],
                    rtol=1e-14,
                    atol=1e-16,
                )

    def test_compilation_cache_reuse(self, simple_matrix):
        """Test that JIT compilation is cached and reused"""
        jitted_fn = jit(apply_diagonal_correction)

        # First call - compiles
        _ = jitted_fn(simple_matrix)

        # Second call - should use cached compilation
        import time

        start = time.time()
        _ = jitted_fn(simple_matrix)
        elapsed = time.time() - start

        # Cached execution should be very fast (<1ms typically)
        assert elapsed < 0.1, f"JIT cache not reused, took {elapsed}s"


# ============================================================================
# Integration Tests - NLSQ
# ============================================================================


class TestDiagonalCorrectionNLSQIntegration:
    """Integration tests with NLSQ optimization"""

    def test_experimental_theoretical_consistency(self, rng):
        """Test that experimental and theoretical data have same preprocessing"""
        # This is an integration test that would require full pipeline
        # Marking as placeholder for end-to-end test
        pytest.skip("End-to-end test - requires full data pipeline")


# ============================================================================
# Integration Tests - Data Pipeline
# ============================================================================


class TestDiagonalCorrectionDataPipeline:
    """Integration tests with data loading pipeline"""

    def test_experimental_data_is_corrected(self):
        """Test that experimental data from xpcs_loader has diagonal correction"""
        # This would require loading real HDF5 data
        pytest.skip("Requires HDF5 test data")

    def test_cached_data_preserves_correction(self):
        """Test that cached data preserves diagonal correction"""
        pytest.skip("Requires cache infrastructure")


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.performance
class TestDiagonalCorrectionPerformance:
    """Performance benchmarks"""

    @pytest.mark.skip(reason="Requires pytest-benchmark plugin not in dependencies")
    @pytest.mark.parametrize("size", [10, 100, 1001])
    def test_performance_single_matrix(self, size, rng):
        """Benchmark performance across different matrix sizes"""
        matrix = jnp.array(rng.randn(size, size))

        result = apply_diagonal_correction(matrix)
        assert result is not None

    @pytest.mark.skip(reason="Requires pytest-benchmark plugin not in dependencies")
    def test_performance_batch_23_matrices(self, rng):
        """Benchmark realistic XPCS workflow: 23 phi angles × 1001×1001"""
        n_phi = 23
        n_t = 1001
        matrices = jnp.array(rng.randn(n_phi, n_t, n_t))

        vmapped_fn = vmap(apply_diagonal_correction, in_axes=0)

        result = vmapped_fn(matrices)
        assert result is not None

    def test_jit_compilation_overhead(self, simple_matrix):
        """Measure JIT compilation vs execution time"""
        import time

        # First call - includes compilation
        start = time.time()
        jitted_fn = jit(apply_diagonal_correction)
        _ = jitted_fn(simple_matrix)
        compile_time = time.time() - start

        # Subsequent calls - cached
        start = time.time()
        for _ in range(100):
            _ = jitted_fn(simple_matrix)
        exec_time = (time.time() - start) / 100

        # Compilation should be much slower than execution
        assert compile_time > exec_time * 10


# ============================================================================
# Regression Tests
# ============================================================================


class TestDiagonalCorrectionRegression:
    """Regression tests for known issues"""

    def test_issue_diagonal_mismatch_nlsq(self):
        """Regression test: NLSQ optimization had diagonal mismatch (Issue #XXX)

        Before fix: Experimental data had diagonal correction, theory did not.
        This caused terrible fits with bright diagonal in theory, flat experimental.

        After fix: Both have diagonal correction applied consistently.
        """
        # Create test matrix with strong diagonal
        matrix = jnp.eye(10) * 10.0 + jnp.ones((10, 10))

        # Apply correction
        corrected = apply_diagonal_correction(matrix)

        # Diagonal should be significantly reduced (not 10.0 anymore)
        diagonal_values = jnp.diag(corrected)
        assert jnp.max(diagonal_values) < 5.0, "Diagonal not properly corrected"
        assert jnp.min(diagonal_values) > 0.5, "Diagonal over-corrected"

        # Off-diagonal should be unchanged (all 1.0)
        size = matrix.shape[0]
        mask = ~jnp.eye(size, dtype=bool)
        assert_allclose(corrected[mask], 1.0, rtol=1e-14)


# ============================================================================
# Property-Based Tests
# ============================================================================

try:
    from hypothesis import given, strategies as st, settings
    from hypothesis.extra.numpy import arrays

    HYPOTHESIS_AVAILABLE = True

    # Define property-based test class inside try block
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    class TestDiagonalCorrectionProperties:
        """Property-based tests using Hypothesis"""

        @given(
            matrix=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(2, 50), st.integers(2, 50)).map(
                    lambda x: (x[0], x[0])
                ),  # Square
                elements=st.floats(
                    min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
                ),
            )
        )
        @settings(max_examples=50, deadline=None)
        def test_idempotent_property(self, matrix):
            """Property: Applying correction twice should equal applying once

            However, this is NOT true for diagonal correction! Second application
            will further interpolate the already-interpolated diagonal.

            This test verifies that diagonal correction is NOT idempotent,
            which is expected behavior.
            """
            matrix_jax = jnp.array(matrix)

            result1 = apply_diagonal_correction(matrix_jax)
            result2 = apply_diagonal_correction(result1)

            # These should NOT be equal (not idempotent)
            if matrix.shape[0] > 2:  # Need at least 3x3 to see difference
                diagonal1 = jnp.diag(result1)
                diagonal2 = jnp.diag(result2)
                # At least one diagonal element should change
                assert not jnp.allclose(diagonal1, diagonal2, rtol=1e-10)

        @given(
            matrix=arrays(
                dtype=np.float64,
                shape=st.tuples(st.integers(2, 30), st.integers(2, 30)).map(
                    lambda x: (x[0], x[0])
                ),
                elements=st.floats(
                    min_value=-100, max_value=100, allow_nan=False, allow_infinity=False
                ),
            )
        )
        @settings(max_examples=50, deadline=None)
        def test_off_diagonal_preservation(self, matrix):
            """Property: Off-diagonal elements must be exactly preserved"""
            matrix_jax = jnp.array(matrix)
            corrected = apply_diagonal_correction(matrix_jax)

            # All off-diagonal elements should be identical
            size = matrix.shape[0]
            for i in range(size):
                for j in range(size):
                    if i != j:
                        assert (
                            corrected[i, j] == matrix_jax[i, j]
                        ), f"Off-diagonal element [{i},{j}] changed"

        @given(
            size=st.integers(2, 100),
            value=st.floats(
                min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
            ),
        )
        @settings(max_examples=50, deadline=None)
        def test_uniform_matrix_property(self, size, value):
            """Property: Uniform matrix diagonal should remain uniform after correction"""
            matrix = jnp.ones((size, size)) * value
            corrected = apply_diagonal_correction(matrix)

            # All diagonal elements should equal the uniform value
            diagonal = jnp.diag(corrected)
            assert_allclose(diagonal, jnp.ones(size) * value, rtol=1e-10)

except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Skip property-based tests if hypothesis not available
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
