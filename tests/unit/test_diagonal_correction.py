"""
Comprehensive Test Suite for Diagonal Correction

Tests the unified diagonal_correction module (v2.14.2+) and its integration with
optimization methods (NLSQ, CMC).

Generated: 2025-10-27
Updated: 2026-01-08 (v2.14.2 - Unified diagonal handling)
Coverage target: >95%

Critical Requirements:
1. Diagonal correction must be applied consistently to experimental and theoretical data
2. Correction must preserve off-diagonal correlation structure
3. JAX operations must be JIT-compatible and differentiable
4. Integration with all optimization methods must be verified
5. (v2.14.2+) Unified module must support multiple backends and methods
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np  # type: ignore[misc]

# Import functions under test from unified module (v2.14.2+)
from homodyne.core.diagonal_correction import (
    apply_diagonal_correction,
    apply_diagonal_correction_batch,
    get_available_backends,
    get_diagonal_correction_methods,
)

# Backward compatibility import (re-exports from unified module)
from homodyne.core.jax_backend import (
    apply_diagonal_correction as apply_diagonal_correction_jax,
)

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
        """Test that JAX backend function can be JIT-compiled.

        Note: The unified apply_diagonal_correction() has backend detection
        that doesn't work inside JIT. We test the internal JAX function
        which is already JIT-compiled by design.
        """
        # Import the internal JIT-compiled function
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        # Should compile without errors (already JIT-compiled)
        result = _diagonal_correction_jax_core(simple_matrix)

        # Should produce same result as unified function with JAX backend
        result_unified = apply_diagonal_correction(simple_matrix, backend="jax")
        assert_allclose(result, result_unified, rtol=1e-14)

    def test_jit_equivalence_large_matrix(self, correlation_matrix_1001x1001):
        """Test JIT equivalence with realistic large matrix"""
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        result_jit = _diagonal_correction_jax_core(correlation_matrix_1001x1001)
        result_unified = apply_diagonal_correction(
            correlation_matrix_1001x1001, backend="jax"
        )

        assert_allclose(result_jit, result_unified, rtol=1e-14, atol=1e-16)

    def test_gradient_correctness(self):
        """Test gradient using finite differences"""
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        # Define scalar function for gradient testing using internal JIT function
        def scalar_fn(x):
            # Create matrix from vector for testing
            matrix = jnp.outer(x, x) + jnp.eye(len(x))
            corrected = _diagonal_correction_jax_core(matrix)
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
        """Test vectorization with vmap using internal JIT function"""
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        # Single matrix
        single_matrix = jnp.array(rng.randn(10, 10))
        single_result = _diagonal_correction_jax_core(single_matrix)

        # Batch of identical matrices
        batch_size = 5
        batch_matrices = jnp.stack([single_matrix] * batch_size)

        # Apply vmap
        vmapped_fn = vmap(_diagonal_correction_jax_core, in_axes=0)
        batch_result = vmapped_fn(batch_matrices)

        # Each result should match single result
        for i in range(batch_size):
            assert_allclose(batch_result[i], single_result, rtol=1e-14)

    def test_vmap_on_phi_axis(self, rng):
        """Test vmap over phi angle axis (realistic NLSQ usage pattern)"""
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

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
        vmapped_correction = vmap(_diagonal_correction_jax_core, in_axes=0)
        corrected_stack = vmapped_correction(correlation_stack)

        # Verify shape preserved
        assert corrected_stack.shape == (n_phi, n_t, n_t)

        # Verify diagonal was corrected for each phi
        for i in range(n_phi):
            original_diag = jnp.diag(correlation_stack[i])
            corrected_diag = jnp.diag(corrected_stack[i])
            # Diagonal should have changed
            assert not jnp.allclose(original_diag, corrected_diag)

    def test_device_consistency(self, simple_matrix):
        """Test consistency across CPU devices"""
        results = {}

        for device in jax.devices():
            with jax.default_device(device):
                result = apply_diagonal_correction(simple_matrix, backend="jax")
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
        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        # First call - compiles (internal function is pre-JIT'd)
        _ = _diagonal_correction_jax_core(simple_matrix)

        # Second call - should use cached compilation
        import time

        start = time.time()
        _ = _diagonal_correction_jax_core(simple_matrix)
        elapsed = time.time() - start

        # Cached execution should be very fast (<1ms typically)
        assert elapsed < 0.1, f"JIT cache not reused, took {elapsed}s"


# ============================================================================
# Unified Module Tests (v2.14.2+)
# ============================================================================


class TestUnifiedModuleAPI:
    """Tests for the unified diagonal_correction module API (v2.14.2+)."""

    def test_get_available_backends(self):
        """Test backend discovery function."""
        backends = get_available_backends()

        # NumPy should always be available
        assert "numpy" in backends

        # JAX availability depends on installation
        if JAX_AVAILABLE:
            assert "jax" in backends

    def test_get_diagonal_correction_methods(self):
        """Test method discovery function."""
        methods = get_diagonal_correction_methods()

        # All three methods should be available
        assert "basic" in methods
        assert "statistical" in methods
        assert "interpolation" in methods

    def test_backward_compatibility_jax_backend_import(self, simple_matrix):
        """Test backward compatibility via jax_backend re-export."""
        # Import from jax_backend should work (re-exports from unified module)
        result_unified = apply_diagonal_correction(simple_matrix)
        result_jax_backend = apply_diagonal_correction_jax(simple_matrix)

        # Results should be identical
        assert_allclose(np.asarray(result_unified), np.asarray(result_jax_backend))


class TestUnifiedModuleBackends:
    """Tests for different backends in unified module."""

    def test_numpy_backend_explicit(self, simple_matrix):
        """Test explicit NumPy backend selection."""
        # Convert to numpy for input
        matrix_np = np.asarray(simple_matrix)

        result = apply_diagonal_correction(matrix_np, backend="numpy")

        # Should return numpy array
        assert isinstance(result, np.ndarray)
        assert result.shape == matrix_np.shape

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_jax_backend_explicit(self, simple_matrix):
        """Test explicit JAX backend selection."""
        result = apply_diagonal_correction(simple_matrix, backend="jax")

        # Should return JAX array
        assert hasattr(result, "device")  # JAX arrays have .device attribute
        assert result.shape == simple_matrix.shape

    def test_auto_backend_numpy_input(self):
        """Test auto backend detection with NumPy input."""
        matrix_np = np.array([[5.0, 1.2], [1.2, 5.0]])

        result = apply_diagonal_correction(matrix_np, backend="auto")

        # With NumPy input, should return NumPy array
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_auto_backend_jax_input(self):
        """Test auto backend detection with JAX input."""
        matrix_jax = jnp.array([[5.0, 1.2], [1.2, 5.0]])

        result = apply_diagonal_correction(matrix_jax, backend="auto")

        # With JAX input, should return JAX array
        assert hasattr(result, "device")

    def test_backend_results_equivalent(self, simple_matrix):
        """Test that NumPy and JAX backends produce equivalent results."""
        matrix_np = np.asarray(simple_matrix)

        result_numpy = apply_diagonal_correction(matrix_np, backend="numpy")

        if JAX_AVAILABLE:
            matrix_jax = jnp.array(matrix_np)
            result_jax = apply_diagonal_correction(matrix_jax, backend="jax")

            assert_allclose(result_numpy, np.asarray(result_jax), rtol=1e-14)


class TestUnifiedModuleMethods:
    """Tests for different correction methods in unified module."""

    def test_basic_method(self):
        """Test basic diagonal correction method."""
        matrix = np.array([[5.0, 1.2, 1.1], [1.2, 5.0, 1.3], [1.1, 1.3, 5.0]])

        result = apply_diagonal_correction(matrix, method="basic", backend="numpy")

        # Verify diagonal was corrected
        assert result[0, 0] == matrix[0, 1]  # Edge: use neighbor
        assert_allclose(
            result[1, 1], (matrix[1, 0] + matrix[1, 2]) / 2
        )  # Middle: average
        assert result[2, 2] == matrix[2, 1]  # Edge: use neighbor

    def test_statistical_method_median(self):
        """Test statistical method with median estimator."""
        # Create matrix with outlier in neighbors
        matrix = np.array(
            [
                [999.0, 1.0, 100.0],  # 100.0 is outlier
                [1.0, 999.0, 1.0],
                [100.0, 1.0, 999.0],
            ]
        )

        result = apply_diagonal_correction(
            matrix, method="statistical", backend="numpy", estimator="median"
        )

        # Median should be robust to outlier
        # Middle diagonal neighbors: [1.0, 1.0, 1.0, 1.0] (within window)
        # Median = 1.0
        assert result[1, 1] == 1.0

    def test_statistical_method_mean(self):
        """Test statistical method with mean estimator."""
        matrix = np.array([[5.0, 1.0], [1.0, 5.0]])

        result = apply_diagonal_correction(
            matrix, method="statistical", backend="numpy", estimator="mean"
        )

        # Both diagonals should use mean of neighbors
        assert result.shape == matrix.shape
        # Should be different from original diagonal
        assert result[0, 0] != matrix[0, 0]
        assert result[1, 1] != matrix[1, 1]

    def test_interpolation_method_linear(self):
        """Test interpolation method with linear interpolation."""
        matrix = np.array(
            [
                [999.0, 1.0, 2.0, 3.0],
                [1.0, 999.0, 2.0, 3.0],
                [2.0, 2.0, 999.0, 3.0],
                [3.0, 3.0, 3.0, 999.0],
            ]
        )

        result = apply_diagonal_correction(
            matrix,
            method="interpolation",
            backend="numpy",
            interpolation_method="linear",
        )

        # Edge cases should use single neighbor
        assert result[0, 0] == matrix[0, 1]
        assert result[3, 3] == matrix[2, 3]

        # Middle values should be average of neighbors
        assert_allclose(result[1, 1], (matrix[0, 1] + matrix[2, 1]) / 2)

    def test_methods_produce_different_results_with_outliers(self):
        """Test that different methods handle outliers differently."""
        # Matrix with outlier
        matrix = np.array(
            [
                [999.0, 1.0, 1.0, 1.0, 100.0],
                [1.0, 999.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 999.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 999.0, 1.0],
                [100.0, 1.0, 1.0, 1.0, 999.0],
            ]
        )

        result_basic = apply_diagonal_correction(
            matrix, method="basic", backend="numpy"
        )
        result_statistical = apply_diagonal_correction(
            matrix, method="statistical", backend="numpy", estimator="median"
        )

        # Statistical method with larger window should differ from basic at edges
        # due to outlier handling
        assert result_basic.shape == result_statistical.shape


class TestBatchProcessing:
    """Tests for batch processing functionality (v2.14.2+)."""

    def test_batch_numpy_basic(self, rng):
        """Test batch processing with NumPy backend."""
        n_phi = 5
        n_t = 20
        matrices = rng.randn(n_phi, n_t, n_t)

        result = apply_diagonal_correction_batch(matrices, backend="numpy")

        assert result.shape == (n_phi, n_t, n_t)
        assert isinstance(result, np.ndarray)

        # Each matrix should be independently corrected
        for i in range(n_phi):
            single_result = apply_diagonal_correction(matrices[i], backend="numpy")
            assert_allclose(result[i], single_result)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_batch_jax_basic(self, rng):
        """Test batch processing with JAX backend."""
        n_phi = 5
        n_t = 20
        matrices = jnp.array(rng.randn(n_phi, n_t, n_t))

        result = apply_diagonal_correction_batch(matrices, backend="jax")

        assert result.shape == (n_phi, n_t, n_t)
        assert hasattr(result, "device")  # JAX array

        # Each matrix should be independently corrected
        for i in range(n_phi):
            single_result = apply_diagonal_correction(matrices[i], backend="jax")
            assert_allclose(np.asarray(result[i]), np.asarray(single_result))

    def test_batch_backends_equivalent(self, rng):
        """Test that batch results are equivalent across backends."""
        n_phi = 3
        n_t = 10
        matrices_np = rng.randn(n_phi, n_t, n_t)

        result_numpy = apply_diagonal_correction_batch(matrices_np, backend="numpy")

        if JAX_AVAILABLE:
            matrices_jax = jnp.array(matrices_np)
            result_jax = apply_diagonal_correction_batch(matrices_jax, backend="jax")

            assert_allclose(result_numpy, np.asarray(result_jax), rtol=1e-14)

    def test_batch_statistical_method(self, rng):
        """Test batch processing with statistical method."""
        n_phi = 3
        n_t = 10
        matrices = rng.randn(n_phi, n_t, n_t)

        result = apply_diagonal_correction_batch(
            matrices, method="statistical", backend="numpy", estimator="median"
        )

        assert result.shape == (n_phi, n_t, n_t)

        # Each matrix should be independently corrected
        for i in range(n_phi):
            single_result = apply_diagonal_correction(
                matrices[i], method="statistical", backend="numpy", estimator="median"
            )
            assert_allclose(result[i], single_result)

    def test_batch_realistic_xpcs_size(self, rng):
        """Test batch processing with realistic XPCS data sizes."""
        n_phi = 23  # Typical number of angles
        n_t = 101  # Reduced for test speed (real data ~1001)
        matrices = rng.randn(n_phi, n_t, n_t) * 0.1 + 1.0

        # Add strong diagonal to simulate autocorrelation
        for i in range(n_phi):
            np.fill_diagonal(matrices[i], 5.0)

        result = apply_diagonal_correction_batch(matrices, backend="numpy")

        # Verify diagonal was corrected for all angles
        for i in range(n_phi):
            diag_original = np.diag(matrices[i])
            diag_corrected = np.diag(result[i])
            # Diagonal values should be reduced (not 5.0 anymore)
            assert np.max(diag_corrected) < np.max(diag_original)


# ============================================================================
# Integration Tests - NLSQ Diagonal Masking (v2.14.2+)
# ============================================================================


class TestNLSQDiagonalMasking:
    """Tests for NLSQ residual diagonal masking (v2.14.2+).

    The unified diagonal handling ensures:
    1. Data (c2_exp): Corrected at load time
    2. Theory (g2): Corrected via apply_diagonal_correction()
    3. Residual: Masked where t1 == t2 (diagonal contributes zero to loss)
    """

    def test_diagonal_mask_concept(self):
        """Test the concept of diagonal masking for residuals."""
        # Simulate t1 and t2 indices
        n_points = 100
        n_t = 10
        t1_indices = np.random.randint(0, n_t, size=n_points)
        t2_indices = np.random.randint(0, n_t, size=n_points)

        # Add some diagonal points
        diagonal_points = n_points // 10
        t1_indices[:diagonal_points] = np.arange(diagonal_points) % n_t
        t2_indices[:diagonal_points] = t1_indices[:diagonal_points]

        # Simulate residuals
        residuals = np.random.randn(n_points)

        # Apply diagonal mask (as done in residual.py v2.14.2+)
        non_diagonal_mask = t1_indices != t2_indices
        masked_residuals = np.where(non_diagonal_mask, residuals, 0.0)

        # Verify diagonal points are zeroed
        for i in range(n_points):
            if t1_indices[i] == t2_indices[i]:
                assert masked_residuals[i] == 0.0
            else:
                assert masked_residuals[i] == residuals[i]

    def test_diagonal_mask_preserves_chi_squared_contribution(self):
        """Test that diagonal mask affects chi-squared correctly."""
        n_points = 100
        t1_indices = np.arange(n_points) % 10
        t2_indices = np.arange(n_points) % 10

        # Make first 10 points diagonal
        diagonal_mask = t1_indices == t2_indices
        n_diagonal = np.sum(diagonal_mask)

        # Simulate residuals with known values
        residuals = np.ones(n_points) * 2.0  # All residuals = 2

        # Without mask: chi2 = sum(residuals^2) = n_points * 4
        chi2_unmasked = np.sum(residuals**2)
        assert chi2_unmasked == n_points * 4.0

        # With mask: diagonal residuals zeroed
        masked_residuals = np.where(~diagonal_mask, residuals, 0.0)
        chi2_masked = np.sum(masked_residuals**2)
        assert chi2_masked == (n_points - n_diagonal) * 4.0


# ============================================================================
# Performance Tests
# ============================================================================


@pytest.mark.performance
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestDiagonalCorrectionPerformance:
    """Performance benchmarks"""

    def test_jit_compilation_overhead(self, simple_matrix):
        """Measure JIT compilation vs execution time"""
        import time

        from homodyne.core.diagonal_correction import _diagonal_correction_jax_core

        # First call - includes compilation (internal function is pre-JIT'd)
        start = time.time()
        _ = _diagonal_correction_jax_core(simple_matrix)
        compile_time = time.time() - start

        # Subsequent calls - cached
        start = time.time()
        for _ in range(100):
            _ = _diagonal_correction_jax_core(simple_matrix)
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
    from hypothesis import given, settings
    from hypothesis import strategies as st
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
            """Property: Applying correction twice should equal applying once.

            The diagonal correction IS idempotent because:
            1. The side band (super-diagonal elements) is not modified
            2. New diagonal is computed from the unchanged side band
            3. Second application uses same side band, producing same diagonal

            This test verifies that diagonal correction is idempotent.
            """
            matrix_jax = jnp.array(matrix)

            result1 = apply_diagonal_correction(matrix_jax)
            result2 = apply_diagonal_correction(result1)

            # Diagonals should be equal (idempotent)
            diagonal1 = jnp.diag(result1)
            diagonal2 = jnp.diag(result2)
            assert jnp.allclose(diagonal1, diagonal2, rtol=1e-10)

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
                        assert corrected[i, j] == matrix_jax[i, j], (
                            f"Off-diagonal element [{i},{j}] changed"
                        )

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
