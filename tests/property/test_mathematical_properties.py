"""
Property-Based Tests for Mathematical Properties
===============================================

Hypothesis-based property tests for mathematical invariants and constraints:
- Physical bounds and constraints
- Symmetry properties
- Mathematical identities
- Numerical stability
- Edge case behavior
"""

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

# Handle Hypothesis imports
try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("Hypothesis not available", allow_module_level=True)


# Custom strategies for physical parameters
def physical_parameters():
    """Generate physically reasonable parameter sets."""
    return st.fixed_dictionaries(
        {
            "offset": st.floats(min_value=0.5, max_value=2.0),
            "contrast": st.floats(min_value=0.0, max_value=1.0),
            "diffusion_coefficient": st.floats(min_value=0.001, max_value=1.0),
            "shear_rate": st.floats(min_value=0.0, max_value=0.5),
            "L": st.floats(min_value=0.1, max_value=10.0),
        }
    )


def time_arrays(min_size=3, max_size=20):
    """Generate time arrays for correlation functions."""
    return st.integers(min_value=min_size, max_value=max_size).flatmap(
        lambda n: st.just(np.meshgrid(np.arange(n), np.arange(n), indexing="ij"))
    )


def angle_arrays(min_angles=6, max_angles=72):
    """Generate angle arrays."""
    return st.integers(min_value=min_angles, max_value=max_angles).map(
        lambda n: np.linspace(0, 2 * np.pi, n)
    )


def q_values():
    """Generate realistic q-values."""
    return st.floats(min_value=0.001, max_value=0.1)


@pytest.mark.property
@pytest.mark.requires_jax
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestJAXBackendProperties:
    """Property-based tests for JAX backend mathematical properties."""

    @given(
        params=physical_parameters(),
        q=q_values(),
        time_data=time_arrays(min_size=3, max_size=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_g1_diffusion_physical_bounds(self, params, q, time_data):
        """Test that g1_diffusion satisfies physical bounds."""
        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        t1, t2 = time_data
        t1 = jnp.array(t1)
        t2 = jnp.array(t2)
        D = params["diffusion_coefficient"]

        result = compute_g1_diffusion_jax(t1, t2, q, D)

        # Physical constraint: 0 ≤ g1 ≤ 1
        assert jnp.all(result >= 0.0), "g1_diffusion must be non-negative"
        assert jnp.all(result <= 1.0), "g1_diffusion must be ≤ 1"

        # Finite values only
        assert jnp.all(jnp.isfinite(result)), "g1_diffusion must be finite"

        # Diagonal elements should be 1 (t1 = t2 case)
        diagonal = jnp.diag(result)
        np.testing.assert_array_almost_equal(
            diagonal,
            jnp.ones_like(diagonal),
            decimal=10,
            err_msg="Diagonal elements should be 1",
        )

    @given(
        params=physical_parameters(),
        q=q_values(),
        angles=angle_arrays(min_angles=6, max_angles=24),
    )
    @settings(max_examples=30, deadline=None)
    def test_g1_shear_physical_bounds(self, params, q, angles):
        """Test that g1_shear satisfies physical bounds."""
        from homodyne.core.jax_backend import compute_g1_shear_jax

        phi = jnp.array(angles)
        # Simple time arrays
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])

        gamma_dot = params["shear_rate"]
        L = params["L"]

        result = compute_g1_shear_jax(phi, t1, t2, gamma_dot, q, L)

        # Physical constraint: 0 ≤ g1_shear ≤ 1
        assert jnp.all(result >= 0.0), "g1_shear must be non-negative"
        assert jnp.all(result <= 1.0), "g1_shear must be ≤ 1"

        # Finite values only
        assert jnp.all(jnp.isfinite(result)), "g1_shear must be finite"

        # Zero shear should give all ones
        if gamma_dot == 0.0:
            expected = jnp.ones_like(result)
            np.testing.assert_array_almost_equal(
                result, expected, decimal=8, err_msg="Zero shear should give all ones"
            )

    @given(
        params=physical_parameters(),
        q=q_values(),
        time_data=time_arrays(min_size=3, max_size=8),
        angles=angle_arrays(min_angles=6, max_angles=18),
    )
    @settings(max_examples=30, deadline=None)
    def test_c2_model_physical_bounds(self, params, q, time_data, angles):
        """Test that c2 model satisfies physical bounds."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        t1, t2 = time_data
        t1 = jnp.array(t1)
        t2 = jnp.array(t2)
        phi = jnp.array(angles)

        result = compute_c2_model_jax(params, t1, t2, phi, q)

        # Physical constraint: g2 ≥ 1 (fundamental bound)
        assert jnp.all(result >= 1.0 - 1e-10), "g2 must be ≥ 1"

        # Finite values only
        assert jnp.all(jnp.isfinite(result)), "g2 must be finite"

        # Diagonal elements should equal offset + contrast
        expected_diagonal = params["offset"] + params["contrast"]
        for angle_idx in range(result.shape[0]):
            diagonal = jnp.diag(result[angle_idx])
            np.testing.assert_array_almost_equal(
                diagonal,
                jnp.full_like(diagonal, expected_diagonal),
                decimal=6,
                err_msg=f"Diagonal elements wrong for angle {angle_idx}",
            )

    @given(params=physical_parameters(), time_data=time_arrays(min_size=3, max_size=10))
    @settings(max_examples=20, deadline=None)
    def test_g1_diffusion_symmetry(self, params, time_data):
        """Test g1_diffusion time-reversal symmetry."""
        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        t1, t2 = time_data
        t1 = jnp.array(t1)
        t2 = jnp.array(t2)
        D = params["diffusion_coefficient"]
        q = 0.01

        # Compute g1(t1, t2)
        result1 = compute_g1_diffusion_jax(t1, t2, q, D)

        # Compute g1(t2, t1) - should be same due to time-reversal symmetry
        result2 = compute_g1_diffusion_jax(t2, t1, q, D)

        np.testing.assert_array_almost_equal(
            result1,
            result2,
            decimal=10,
            err_msg="g1_diffusion should be symmetric in t1, t2",
        )

    @given(
        params1=physical_parameters(),
        params2=physical_parameters(),
        scale=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=20, deadline=None)
    def test_parameter_scaling_properties(self, params1, params2, scale):
        """Test parameter scaling properties."""
        from homodyne.core.jax_backend import compute_c2_model_jax

        # Simple test arrays
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0, jnp.pi])
        q = 0.01

        # Compute base result
        result1 = compute_c2_model_jax(params1, t1, t2, phi, q)

        # Scale contrast and test linearity
        scaled_params = params1.copy()
        scaled_params["contrast"] *= scale

        result_scaled = compute_c2_model_jax(scaled_params, t1, t2, phi, q)

        # The difference should scale linearly with contrast
        diff1 = result1 - params1["offset"]
        diff_scaled = result_scaled - params1["offset"]

        expected_scaling = diff1 * scale
        np.testing.assert_array_almost_equal(
            diff_scaled,
            expected_scaling,
            decimal=8,
            err_msg="Contrast scaling should be linear",
        )

    @given(params=physical_parameters(), q=q_values())
    @settings(max_examples=20, deadline=None)
    def test_diffusion_monotonicity(self, params, q):
        """Test that diffusion decreases monotonically with time."""
        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        # Create increasing time separations
        times = jnp.array([0, 1, 2, 3, 4, 5])
        t1_base = jnp.zeros_like(times)
        t2_varying = times

        D = params["diffusion_coefficient"]

        result = compute_g1_diffusion_jax(t1_base, t2_varying, q, D)

        # Should be monotonically decreasing (or at least non-increasing)
        for i in range(len(result) - 1):
            assert result[i] >= result[i + 1] - 1e-10, (
                f"g1_diffusion should decrease with time: {result[i]} >= {result[i + 1]}"
            )


@pytest.mark.property
@pytest.mark.requires_jax
class TestResidualProperties:
    """Property-based tests for residual and chi-squared functions."""

    @given(
        params=physical_parameters(),
        noise_scale=st.floats(min_value=0.001, max_value=0.1),
        q=q_values(),
    )
    @settings(max_examples=20, deadline=None)
    def test_residuals_zero_for_perfect_fit(self, params, noise_scale, q):
        """Test that residuals are zero for perfect model fit."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import compute_c2_model_jax, residuals_jax

        # Simple test case
        t1 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        t2 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

        # Generate perfect model data
        c2_exp = compute_c2_model_jax(params, t1, t2, phi, q)
        sigma = jnp.ones_like(c2_exp) * noise_scale

        # Compute residuals with same parameters
        residuals = residuals_jax(params, c2_exp, sigma, t1, t2, phi, q)

        # Residuals should be very close to zero
        max_residual = jnp.max(jnp.abs(residuals))
        assert max_residual < 1e-8, f"Perfect fit residuals too large: {max_residual}"

        # Mean residual should be close to zero
        mean_residual = jnp.mean(residuals)
        assert abs(mean_residual) < 1e-10, f"Mean residual not zero: {mean_residual}"

    @given(
        params=physical_parameters(),
        offset_scale=st.floats(min_value=0.01, max_value=0.5),
        noise_scale=st.floats(min_value=0.001, max_value=0.1),
    )
    @settings(max_examples=15, deadline=None)
    def test_chi_squared_properties(self, params, offset_scale, noise_scale):
        """Test chi-squared function properties."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import chi_squared_jax, compute_c2_model_jax

        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        # Generate model data
        c2_model = compute_c2_model_jax(params, t1, t2, phi, q)
        sigma = jnp.ones_like(c2_model) * noise_scale

        # Perfect fit case
        chi2_perfect = chi_squared_jax(params, c2_model, sigma, t1, t2, phi, q)
        assert chi2_perfect < 1e-12, "Perfect fit should have near-zero chi-squared"

        # Offset data to create imperfect fit
        c2_offset = c2_model + offset_scale
        chi2_offset = chi_squared_jax(params, c2_offset, sigma, t1, t2, phi, q)

        # Chi-squared should increase with worse fit
        assert chi2_offset > chi2_perfect, (
            "Imperfect fit should have higher chi-squared"
        )

        # Chi-squared should always be non-negative
        assert chi2_offset >= 0.0, "Chi-squared must be non-negative"

    @given(params=physical_parameters(), scale=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=15, deadline=None)
    def test_chi_squared_scaling_with_sigma(self, params, scale):
        """Test chi-squared scaling with uncertainty."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import chi_squared_jax, compute_c2_model_jax

        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        # Generate data with fixed offset
        c2_model = compute_c2_model_jax(params, t1, t2, phi, q)
        c2_exp = c2_model + 0.01  # Fixed offset

        # Test with different sigma values
        sigma1 = jnp.ones_like(c2_exp) * 0.02
        sigma2 = sigma1 * scale

        chi2_1 = chi_squared_jax(params, c2_exp, sigma1, t1, t2, phi, q)
        chi2_2 = chi_squared_jax(params, c2_exp, sigma2, t1, t2, phi, q)

        # Chi-squared should scale inversely with sigma^2
        expected_ratio = 1.0 / (scale**2)
        actual_ratio = chi2_2 / (chi2_1 + 1e-12)

        np.testing.assert_almost_equal(
            actual_ratio,
            expected_ratio,
            decimal=6,
            err_msg=f"Chi-squared sigma scaling wrong: {actual_ratio} != {expected_ratio}",
        )


@pytest.mark.property
class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability."""

    @given(
        small_values=st.floats(min_value=1e-12, max_value=1e-6),
        large_values=st.floats(min_value=1e6, max_value=1e12),
    )
    @settings(max_examples=10, deadline=None)
    def test_extreme_value_stability(self, small_values, large_values):
        """Test numerical stability with extreme values."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import compute_g1_diffusion_jax

        # Test with very small time differences
        t1 = jnp.array([[0.0, small_values], [small_values, 0.0]])
        t2 = jnp.array([[0.0, small_values], [small_values, 0.0]])

        result_small = compute_g1_diffusion_jax(t1, t2, 0.01, 0.1)

        # Should still be finite and within bounds
        assert jnp.all(jnp.isfinite(result_small)), (
            "Small values should give finite results"
        )
        assert jnp.all(result_small >= 0.0), "Small values should respect bounds"
        assert jnp.all(result_small <= 1.0), "Small values should respect bounds"

        # Test with large time differences (but reasonable D and q)
        assume(large_values < 1e10)  # Avoid overflow
        t1_large = jnp.array([[0.0, large_values], [large_values, 0.0]])
        t2_large = jnp.array([[0.0, large_values], [large_values, 0.0]])

        result_large = compute_g1_diffusion_jax(t1_large, t2_large, 0.01, 0.1)

        # Should decay to near zero but remain finite
        assert jnp.all(jnp.isfinite(result_large)), (
            "Large values should give finite results"
        )
        assert jnp.all(result_large >= 0.0), "Large values should respect bounds"

    @given(
        dimension=st.integers(min_value=2, max_value=50), params=physical_parameters()
    )
    @settings(max_examples=10, deadline=None)
    def test_scaling_with_matrix_size(self, dimension, params):
        """Test numerical stability with different matrix sizes."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import compute_c2_model_jax

        # Create matrices of different sizes
        t1, t2 = jnp.meshgrid(
            jnp.arange(dimension), jnp.arange(dimension), indexing="ij"
        )
        phi = jnp.linspace(0, 2 * jnp.pi, 8)  # Keep angles reasonable

        result = compute_c2_model_jax(params, t1, t2, phi, 0.01)

        # Should maintain numerical stability regardless of size
        assert jnp.all(jnp.isfinite(result)), (
            f"Size {dimension} should give finite results"
        )
        assert jnp.all(result >= 1.0 - 1e-10), (
            f"Size {dimension} should respect physical bounds"
        )

        # Shape should be correct
        expected_shape = (len(phi), dimension, dimension)
        assert result.shape == expected_shape, f"Shape mismatch for size {dimension}"

    @given(
        angles=st.lists(
            st.floats(min_value=0.0, max_value=2 * np.pi),
            min_size=3,
            max_size=100,
            unique=True,
        ).map(lambda x: sorted(x))
    )
    @settings(max_examples=10, deadline=None)
    def test_angle_array_stability(self, angles):
        """Test stability with different angle array configurations."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        from homodyne.core.jax_backend import compute_g1_shear_jax

        phi = jnp.array(angles)
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])

        result = compute_g1_shear_jax(phi, t1, t2, 0.1, 0.01, 1.0)

        # Should be stable regardless of angle array size/configuration
        assert jnp.all(jnp.isfinite(result)), "Angle arrays should give finite results"
        assert jnp.all(result >= 0.0), "Should respect lower bound"
        assert jnp.all(result <= 1.0), "Should respect upper bound"

        # Shape should be correct
        expected_shape = (len(angles), 2, 2)
        assert result.shape == expected_shape, "Shape should match angle array"
