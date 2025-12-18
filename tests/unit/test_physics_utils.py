"""
Unit Tests for homodyne.core.physics_utils Module
=================================================

Comprehensive tests for shared physics utility functions used by both
NLSQ (meshgrid) and CMC (element-wise) computational backends.

Test Coverage:
- safe_len: JAX-safe length function
- safe_exp: Overflow-protected exponential
- safe_sinc: Numerically stable sinc function
- calculate_diffusion_coefficient: Time-dependent diffusion D(t)
- calculate_shear_rate: Time-dependent shear rate γ̇(t)
- calculate_shear_rate_cmc: CMC variant with additional safety
- create_time_integral_matrix: Trapezoidal integration matrix
- trapezoid_cumsum: Cumulative trapezoid integral
- apply_diagonal_correction: Diagonal correction for correlation matrices
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

# JAX imports with fallback
try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.physics_utils import (
    EPS,
    PI,
    apply_diagonal_correction,
    calculate_diffusion_coefficient,
    calculate_shear_rate,
    calculate_shear_rate_cmc,
    create_time_integral_matrix,
    safe_exp,
    safe_len,
    safe_sinc,
    trapezoid_cumsum,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def time_array_small():
    """Small time array for basic tests."""
    return jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.fixture
def time_array_large():
    """Larger time array for comprehensive tests."""
    return jnp.linspace(0, 10, 100)


@pytest.fixture
def sample_correlation_matrix():
    """Sample 2D correlation matrix for diagonal correction tests."""
    # Create a symmetric matrix with known diagonal
    n = 5
    mat = np.ones((n, n)) * 1.1
    np.fill_diagonal(mat, 2.0)  # Diagonal has different value
    return jnp.array(mat)


# =============================================================================
# Tests for safe_len
# =============================================================================


class TestSafeLen:
    """Test suite for safe_len function."""

    def test_scalar_int(self):
        """Test safe_len with integer scalar."""
        assert safe_len(5) == 1

    def test_scalar_float(self):
        """Test safe_len with float scalar."""
        assert safe_len(3.14) == 1

    def test_numpy_array_1d(self):
        """Test safe_len with 1D numpy array."""
        arr = np.array([1, 2, 3, 4, 5])
        assert safe_len(arr) == 5

    def test_numpy_array_2d(self):
        """Test safe_len with 2D numpy array."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        assert safe_len(arr) == 3  # First dimension

    def test_numpy_scalar(self):
        """Test safe_len with numpy scalar (0-d array)."""
        arr = np.array(42)
        assert safe_len(arr) == 1

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_array_1d(self):
        """Test safe_len with 1D JAX array."""
        arr = jnp.array([1, 2, 3, 4])
        assert safe_len(arr) == 4

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_array_2d(self):
        """Test safe_len with 2D JAX array."""
        arr = jnp.ones((3, 4))
        assert safe_len(arr) == 3

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_scalar(self):
        """Test safe_len with JAX scalar."""
        arr = jnp.array(3.14)
        assert safe_len(arr) == 1

    def test_python_list(self):
        """Test safe_len with Python list."""
        assert safe_len([1, 2, 3]) == 3

    def test_python_tuple(self):
        """Test safe_len with Python tuple."""
        assert safe_len((1, 2, 3, 4)) == 4

    def test_empty_list(self):
        """Test safe_len with empty list."""
        assert safe_len([]) == 0

    def test_empty_array(self):
        """Test safe_len with empty array."""
        assert safe_len(np.array([])) == 0


# =============================================================================
# Tests for safe_exp
# =============================================================================


class TestSafeExp:
    """Test suite for safe_exp function."""

    def test_normal_values(self):
        """Test safe_exp with normal input values."""
        x = jnp.array([0.0, 1.0, 2.0])
        result = safe_exp(x)
        expected = jnp.exp(x)
        assert_allclose(result, expected, rtol=1e-10)

    def test_large_positive_values(self):
        """Test safe_exp clips large positive values to prevent overflow."""
        x = jnp.array([1000.0, 800.0, 750.0])
        result = safe_exp(x)
        # Should be clipped to exp(700) = ~1e304
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_large_negative_values(self):
        """Test safe_exp clips large negative values."""
        x = jnp.array([-1000.0, -800.0, -750.0])
        result = safe_exp(x)
        # Should be clipped to exp(-700) ≈ 0
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0)

    def test_custom_max_val(self):
        """Test safe_exp with custom max_val parameter."""
        x = jnp.array([100.0, 200.0, 300.0])
        result = safe_exp(x, max_val=50.0)
        expected = jnp.exp(jnp.array([50.0, 50.0, 50.0]))
        assert_allclose(result, expected, rtol=1e-10)

    def test_zero_input(self):
        """Test safe_exp with zero input."""
        x = jnp.array([0.0])
        result = safe_exp(x)
        assert_allclose(result, jnp.array([1.0]), rtol=1e-10)

    def test_preserves_shape(self):
        """Test that safe_exp preserves array shape."""
        x = jnp.ones((3, 4, 5))
        result = safe_exp(x)
        assert result.shape == x.shape

    def test_mixed_values(self):
        """Test safe_exp with mix of normal and extreme values."""
        x = jnp.array([-1000.0, 0.0, 1.0, 1000.0])
        result = safe_exp(x)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Tests for safe_sinc
# =============================================================================


class TestSafeSinc:
    """Test suite for safe_sinc (unnormalized sinc: sin(x)/x)."""

    def test_zero_input(self):
        """Test safe_sinc returns 1.0 at x=0."""
        x = jnp.array([0.0])
        result = safe_sinc(x)
        assert_allclose(result, jnp.array([1.0]), rtol=1e-10)

    def test_small_values(self):
        """Test safe_sinc near zero returns ~1.0."""
        x = jnp.array([1e-14, 1e-13, 1e-12])
        result = safe_sinc(x)
        assert_allclose(result, jnp.ones_like(x), rtol=1e-6)

    def test_pi_values(self):
        """Test safe_sinc at multiples of pi."""
        # sin(nπ)/nπ = 0 for n ≠ 0
        x = jnp.array([PI, 2 * PI, 3 * PI])
        result = safe_sinc(x)
        assert_allclose(result, jnp.zeros_like(x), atol=1e-10)

    def test_pi_half(self):
        """Test safe_sinc at π/2."""
        x = jnp.array([PI / 2])
        result = safe_sinc(x)
        expected = jnp.sin(PI / 2) / (PI / 2)  # = 2/π
        assert_allclose(result, expected, rtol=1e-10)

    def test_negative_values(self):
        """Test safe_sinc with negative values (should be same as positive)."""
        x_pos = jnp.array([1.0, 2.0, 3.0])
        x_neg = -x_pos
        result_pos = safe_sinc(x_pos)
        result_neg = safe_sinc(x_neg)
        assert_allclose(result_pos, result_neg, rtol=1e-10)

    def test_unnormalized_vs_normalized(self):
        """Verify this is unnormalized sinc (sin(x)/x), not normalized (sin(πx)/(πx))."""
        x = jnp.array([1.0])
        result = safe_sinc(x)
        unnormalized = jnp.sin(1.0) / 1.0
        normalized = jnp.sinc(1.0)  # numpy sinc is normalized
        # Our function should match unnormalized
        assert_allclose(result, unnormalized, rtol=1e-10)
        # And NOT match normalized
        assert not jnp.allclose(result, normalized)

    def test_array_operations(self):
        """Test safe_sinc works element-wise on arrays."""
        x = jnp.array([0.0, 0.5, 1.0, 2.0, PI])
        result = safe_sinc(x)
        assert result.shape == x.shape
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Tests for calculate_diffusion_coefficient
# =============================================================================


class TestCalculateDiffusionCoefficient:
    """Test suite for calculate_diffusion_coefficient function."""

    def test_constant_diffusion(self, time_array_small):
        """Test D(t) with alpha=0 gives constant D0 + D_offset."""
        D0, alpha, D_offset = 1.0, 0.0, 0.1
        result = calculate_diffusion_coefficient(time_array_small, D0, alpha, D_offset)
        # With alpha=0: D(t) = D0 * t^0 + D_offset = D0 + D_offset
        expected = D0 + D_offset
        assert_allclose(result, jnp.full_like(time_array_small, expected), rtol=1e-6)

    def test_linear_diffusion(self, time_array_small):
        """Test D(t) with alpha=1 gives linear dependence."""
        D0, alpha, D_offset = 1.0, 1.0, 0.0
        result = calculate_diffusion_coefficient(time_array_small, D0, alpha, D_offset)
        # With alpha=1: D(t) = D0 * t + D_offset (approximately, with epsilon)
        assert result[0] > 0  # Should be positive due to epsilon
        assert result[-1] > result[0]  # Should increase with time

    def test_negative_alpha_no_singularity(self, time_array_small):
        """Test that negative alpha doesn't cause singularity at t=0."""
        D0, alpha, D_offset = 1.0, -0.5, 0.0
        result = calculate_diffusion_coefficient(time_array_small, D0, alpha, D_offset)
        # Should be finite at t=0 due to epsilon protection
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_positive_output(self, time_array_large):
        """Test that output is always positive."""
        D0, alpha, D_offset = 0.5, 0.3, 0.01
        result = calculate_diffusion_coefficient(time_array_large, D0, alpha, D_offset)
        assert jnp.all(result >= 1e-10)

    def test_zero_D0(self, time_array_small):
        """Test with D0=0, result should be D_offset (clamped to minimum)."""
        D0, alpha, D_offset = 0.0, 1.0, 1.0
        result = calculate_diffusion_coefficient(time_array_small, D0, alpha, D_offset)
        assert jnp.all(result >= 1e-10)

    def test_anomalous_subdiffusion(self, time_array_large):
        """Test subdiffusive behavior (0 < alpha < 1)."""
        D0, alpha, D_offset = 1.0, 0.5, 0.0
        result = calculate_diffusion_coefficient(time_array_large, D0, alpha, D_offset)
        # D(t) = D0 * t^0.5 + D_offset, increases but slower than linear
        assert jnp.all(jnp.diff(result) > 0)  # Monotonically increasing

    def test_anomalous_superdiffusion(self, time_array_large):
        """Test superdiffusive behavior (alpha > 1)."""
        D0, alpha, D_offset = 1.0, 1.5, 0.0
        time_positive = time_array_large[time_array_large > 0]
        result = calculate_diffusion_coefficient(time_positive, D0, alpha, D_offset)
        # D(t) = D0 * t^1.5 increases faster than linear
        assert jnp.all(jnp.diff(result) > 0)


# =============================================================================
# Tests for calculate_shear_rate
# =============================================================================


class TestCalculateShearRate:
    """Test suite for calculate_shear_rate function."""

    def test_constant_shear(self, time_array_small):
        """Test γ̇(t) with beta=0 gives constant γ̇₀ + γ̇_offset."""
        gamma_dot_0, beta, gamma_dot_offset = 0.5, 0.0, 0.1
        result = calculate_shear_rate(
            time_array_small, gamma_dot_0, beta, gamma_dot_offset
        )
        expected = gamma_dot_0 + gamma_dot_offset
        # First element uses dt substitution, rest should be constant
        assert_allclose(result[1:], jnp.full(len(result) - 1, expected), rtol=1e-6)

    def test_linear_shear(self, time_array_small):
        """Test γ̇(t) with beta=1 gives linear dependence."""
        gamma_dot_0, beta, gamma_dot_offset = 1.0, 1.0, 0.0
        result = calculate_shear_rate(
            time_array_small, gamma_dot_0, beta, gamma_dot_offset
        )
        # Should increase with time
        assert result[-1] > result[0]

    def test_negative_beta_no_singularity(self, time_array_small):
        """Test that negative beta doesn't cause singularity at t=0."""
        gamma_dot_0, beta, gamma_dot_offset = 1.0, -0.5, 0.0
        result = calculate_shear_rate(
            time_array_small, gamma_dot_0, beta, gamma_dot_offset
        )
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_positive_output(self, time_array_large):
        """Test that output is always positive."""
        gamma_dot_0, beta, gamma_dot_offset = 0.5, 0.3, 0.01
        result = calculate_shear_rate(
            time_array_large, gamma_dot_0, beta, gamma_dot_offset
        )
        assert jnp.all(result >= 1e-10)

    def test_single_time_point(self):
        """Test with single time point uses fallback dt."""
        time_single = jnp.array([0.5])
        gamma_dot_0, beta, gamma_dot_offset = 1.0, 0.5, 0.0
        result = calculate_shear_rate(time_single, gamma_dot_0, beta, gamma_dot_offset)
        assert jnp.all(jnp.isfinite(result))
        assert result.shape == (1,)


# =============================================================================
# Tests for calculate_shear_rate_cmc
# =============================================================================


class TestCalculateShearRateCmc:
    """Test suite for calculate_shear_rate_cmc function."""

    def test_consecutive_zeros_handling(self):
        """Test CMC variant handles consecutive zeros (dt=0 case)."""
        # CMC element-wise data can have consecutive zeros
        time_array = jnp.array([0.0, 0.0, 0.1, 0.2])
        gamma_dot_0, beta, gamma_dot_offset = 1.0, -0.5, 0.0
        result = calculate_shear_rate_cmc(
            time_array, gamma_dot_0, beta, gamma_dot_offset
        )
        # Should be finite despite dt=0 due to maximum(dt, 1e-10)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result > 0)

    def test_matches_regular_for_normal_input(self, time_array_small):
        """Test CMC variant gives similar results to regular for normal input."""
        gamma_dot_0, beta, gamma_dot_offset = 0.5, 0.3, 0.1
        result_cmc = calculate_shear_rate_cmc(
            time_array_small, gamma_dot_0, beta, gamma_dot_offset
        )
        result_regular = calculate_shear_rate(
            time_array_small, gamma_dot_0, beta, gamma_dot_offset
        )
        # Should be very close for normal inputs
        assert_allclose(result_cmc, result_regular, rtol=1e-8)


# =============================================================================
# Tests for create_time_integral_matrix
# =============================================================================


class TestCreateTimeIntegralMatrix:
    """Test suite for create_time_integral_matrix function."""

    def test_output_shape(self):
        """Test output matrix has correct shape."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = create_time_integral_matrix(values)
        assert result.shape == (5, 5)

    def test_diagonal_is_zero(self):
        """Test diagonal elements are approximately zero."""
        values = jnp.array([1.0, 1.0, 1.0, 1.0])
        result = create_time_integral_matrix(values)
        # Diagonal should be sqrt(0 + epsilon) ≈ epsilon^0.5
        diagonal = jnp.diag(result)
        assert jnp.all(diagonal < 1e-9)

    def test_symmetry(self):
        """Test matrix is symmetric."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = create_time_integral_matrix(values)
        assert_allclose(result, result.T, rtol=1e-10)

    def test_non_negative(self):
        """Test all elements are non-negative."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = create_time_integral_matrix(values)
        assert jnp.all(result >= 0)

    def test_constant_input(self):
        """Test with constant input array."""
        values = jnp.ones(5)
        result = create_time_integral_matrix(values)
        # For constant f(t)=1, integral from i to j is |j-i|
        for i in range(5):
            for j in range(5):
                # Using trapezoidal rule: result[i,j] ≈ |i-j|
                expected = abs(i - j)
                assert_allclose(result[i, j], expected, atol=1e-9)

    def test_single_element(self):
        """Test with single element array."""
        values = jnp.array([3.0])
        result = create_time_integral_matrix(values)
        assert result.shape == (1, 1)

    def test_scalar_input_converted(self):
        """Test scalar input is converted to array."""
        values = jnp.array(5.0)  # Scalar
        result = create_time_integral_matrix(values)
        assert result.shape == (1, 1)

    def test_finite_output(self):
        """Test all output values are finite."""
        values = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0])
        result = create_time_integral_matrix(values)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Tests for trapezoid_cumsum
# =============================================================================


class TestTrapezoidCumsum:
    """Test suite for trapezoid_cumsum function."""

    def test_output_length(self):
        """Test output has same length as input."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = trapezoid_cumsum(values)
        assert result.shape == values.shape

    def test_starts_at_zero(self):
        """Test cumsum starts at zero."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = trapezoid_cumsum(values)
        assert result[0] == 0.0

    def test_constant_input(self):
        """Test with constant input."""
        values = jnp.ones(5)
        result = trapezoid_cumsum(values)
        # Trapezoidal cumsum of [1,1,1,1,1] = [0, 1, 2, 3, 4]
        expected = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert_allclose(result, expected, rtol=1e-10)

    def test_linear_input(self):
        """Test with linear input."""
        values = jnp.array([0.0, 1.0, 2.0, 3.0])
        result = trapezoid_cumsum(values)
        # trap_avg = [0.5, 1.5, 2.5]
        # cumsum = [0, 0.5, 2.0, 4.5]
        expected = jnp.array([0.0, 0.5, 2.0, 4.5])
        assert_allclose(result, expected, rtol=1e-10)

    def test_single_element(self):
        """Test with single element (uses direct cumsum)."""
        values = jnp.array([5.0])
        result = trapezoid_cumsum(values)
        assert_allclose(result, jnp.array([5.0]), rtol=1e-10)

    def test_preserves_dtype(self):
        """Test that dtype is preserved."""
        values = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        result = trapezoid_cumsum(values)
        assert result.dtype == jnp.float32


# =============================================================================
# Tests for apply_diagonal_correction
# =============================================================================


class TestApplyDiagonalCorrection:
    """Test suite for apply_diagonal_correction function."""

    def test_output_shape(self, sample_correlation_matrix):
        """Test output has same shape as input."""
        result = apply_diagonal_correction(sample_correlation_matrix)
        assert result.shape == sample_correlation_matrix.shape

    def test_diagonal_changed(self, sample_correlation_matrix):
        """Test that diagonal values are modified."""
        result = apply_diagonal_correction(sample_correlation_matrix)
        original_diag = jnp.diag(sample_correlation_matrix)
        new_diag = jnp.diag(result)
        # Diagonal should be changed
        assert not jnp.allclose(original_diag, new_diag)

    def test_off_diagonal_unchanged(self, sample_correlation_matrix):
        """Test that off-diagonal elements are unchanged."""
        result = apply_diagonal_correction(sample_correlation_matrix)
        n = sample_correlation_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j:
                    assert_allclose(
                        result[i, j],
                        sample_correlation_matrix[i, j],
                        rtol=1e-10,
                    )

    def test_edge_elements(self):
        """Test edge elements (first and last diagonal) are handled correctly."""
        # Create matrix where we know the expected result
        mat = jnp.array(
            [
                [5.0, 1.0, 1.0],
                [1.0, 5.0, 2.0],
                [1.0, 2.0, 5.0],
            ]
        )
        result = apply_diagonal_correction(mat)
        # diag[0] = side_band[0] = 1.0
        # diag[1] = (side_band[0] + side_band[1]) / 2 = (1.0 + 2.0) / 2 = 1.5
        # diag[2] = side_band[1] = 2.0
        expected_diag = jnp.array([1.0, 1.5, 2.0])
        assert_allclose(jnp.diag(result), expected_diag, rtol=1e-10)

    def test_symmetric_input_symmetric_output(self):
        """Test symmetric input produces symmetric output."""
        mat = jnp.array(
            [
                [2.0, 1.1, 1.2, 1.3],
                [1.1, 2.0, 1.4, 1.5],
                [1.2, 1.4, 2.0, 1.6],
                [1.3, 1.5, 1.6, 2.0],
            ]
        )
        result = apply_diagonal_correction(mat)
        assert_allclose(result, result.T, rtol=1e-10)

    def test_small_matrix(self):
        """Test with 2x2 matrix (minimum size)."""
        mat = jnp.array([[3.0, 1.0], [1.0, 3.0]])
        result = apply_diagonal_correction(mat)
        # Both diagonal elements should become side_band[0] = 1.0
        expected_diag = jnp.array([1.0, 1.0])
        assert_allclose(jnp.diag(result), expected_diag, rtol=1e-10)

    def test_finite_output(self):
        """Test output contains only finite values."""
        mat = jnp.ones((10, 10)) + jnp.diag(jnp.ones(10))
        result = apply_diagonal_correction(mat)
        assert jnp.all(jnp.isfinite(result))

    def test_realistic_correlation_matrix(self):
        """Test with a realistic correlation matrix shape."""
        # Create a matrix that looks like real XPCS data
        n = 100
        # Off-diagonal ~ 1.0, diagonal ~ 1.5 (autocorrelation peak)
        mat = jnp.ones((n, n)) + 0.5 * jnp.eye(n)
        result = apply_diagonal_correction(mat)

        # Diagonal should now be close to off-diagonal
        diagonal = jnp.diag(result)
        assert jnp.all(jnp.abs(diagonal - 1.0) < 0.01)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical stability of physics utility functions."""

    def test_safe_exp_extreme_values(self):
        """Test safe_exp handles extreme values without overflow."""
        x = jnp.array([-1e10, -1000, 0, 1000, 1e10])
        result = safe_exp(x)
        assert jnp.all(jnp.isfinite(result))

    def test_diffusion_coefficient_extreme_alpha(self):
        """Test diffusion coefficient with extreme alpha values."""
        time_array = jnp.linspace(0.1, 1.0, 10)  # Avoid t=0
        for alpha in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            result = calculate_diffusion_coefficient(time_array, 1.0, alpha, 0.1)
            assert jnp.all(jnp.isfinite(result))

    def test_shear_rate_extreme_beta(self):
        """Test shear rate with extreme beta values."""
        time_array = jnp.linspace(0.1, 1.0, 10)
        for beta in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            result = calculate_shear_rate(time_array, 1.0, beta, 0.1)
            assert jnp.all(jnp.isfinite(result))

    def test_time_integral_matrix_large_values(self):
        """Test time integral matrix with large input values."""
        values = jnp.array([1e6, 2e6, 3e6, 4e6])
        result = create_time_integral_matrix(values)
        assert jnp.all(jnp.isfinite(result))

    def test_time_integral_matrix_small_values(self):
        """Test time integral matrix with small input values."""
        values = jnp.array([1e-10, 2e-10, 3e-10, 4e-10])
        result = create_time_integral_matrix(values)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# JAX JIT Compatibility Tests
# =============================================================================


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJITCompatibility:
    """Test that functions work correctly with JAX JIT compilation."""

    def test_safe_exp_jit(self):
        """Test safe_exp is JIT-compatible."""
        # safe_exp is already decorated with @jit
        x = jnp.array([1.0, 2.0, 3.0])
        result = safe_exp(x)
        assert jnp.all(jnp.isfinite(result))

    def test_safe_sinc_jit(self):
        """Test safe_sinc is JIT-compatible."""
        x = jnp.array([0.0, 1.0, 2.0])
        result = safe_sinc(x)
        assert jnp.all(jnp.isfinite(result))

    def test_diffusion_coefficient_jit(self):
        """Test calculate_diffusion_coefficient is JIT-compatible."""
        time_array = jnp.linspace(0, 1, 10)
        result = calculate_diffusion_coefficient(time_array, 1.0, 0.5, 0.1)
        assert jnp.all(jnp.isfinite(result))

    def test_shear_rate_jit(self):
        """Test calculate_shear_rate is JIT-compatible."""
        time_array = jnp.linspace(0, 1, 10)
        result = calculate_shear_rate(time_array, 1.0, 0.5, 0.1)
        assert jnp.all(jnp.isfinite(result))

    def test_time_integral_matrix_jit(self):
        """Test create_time_integral_matrix is JIT-compatible."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = create_time_integral_matrix(values)
        assert jnp.all(jnp.isfinite(result))

    def test_diagonal_correction_jit(self):
        """Test apply_diagonal_correction is JIT-compatible."""
        mat = jnp.ones((5, 5)) + jnp.eye(5)
        result = apply_diagonal_correction(mat)
        assert jnp.all(jnp.isfinite(result))


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


class TestBackwardCompatibilityAliases:
    """Test backward compatibility aliases."""

    def test_diffusion_coefficient_alias(self):
        """Test _calculate_diffusion_coefficient_impl_jax alias exists."""
        from homodyne.core.physics_utils import (
            _calculate_diffusion_coefficient_impl_jax,
        )

        assert (
            _calculate_diffusion_coefficient_impl_jax is calculate_diffusion_coefficient
        )

    def test_shear_rate_alias(self):
        """Test _calculate_shear_rate_impl_jax alias exists."""
        from homodyne.core.physics_utils import _calculate_shear_rate_impl_jax

        assert _calculate_shear_rate_impl_jax is calculate_shear_rate

    def test_time_integral_matrix_alias(self):
        """Test _create_time_integral_matrix_impl_jax alias exists."""
        from homodyne.core.physics_utils import (
            _create_time_integral_matrix_impl_jax,
        )

        assert _create_time_integral_matrix_impl_jax is create_time_integral_matrix

    def test_trapezoid_cumsum_alias(self):
        """Test _trapezoid_cumsum alias exists."""
        from homodyne.core.physics_utils import _trapezoid_cumsum

        assert _trapezoid_cumsum is trapezoid_cumsum


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_pi_value(self):
        """Test PI constant is correct."""
        assert_allclose(float(PI), np.pi, rtol=1e-15)

    def test_eps_value(self):
        """Test EPS constant is appropriate."""
        assert EPS > 0
        assert EPS < 1e-10
