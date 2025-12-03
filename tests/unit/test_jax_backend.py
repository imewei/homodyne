"""
Unit Tests for JAX Backend Core Computations
============================================

Tests for homodyne.core.jax_backend module including:
- JAX-compiled mathematical functions
- Automatic differentiation
- Vectorization operations
- Numerical stability
- CPU-only compatibility (GPU removed in v2.3.0)
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

from homodyne.core.jax_backend import (
    chi_squared_jax,
    compute_c2_model_jax,
    compute_g1_diffusion,
    compute_g1_diffusion_jax,
    compute_g1_shear,
    compute_g2_scaled,
    residuals_jax,
)
from homodyne.core.jax_backend import jax_available as BACKEND_JAX_AVAILABLE


@pytest.mark.unit
@pytest.mark.requires_jax
class TestJAXBackendCore:
    """Test core JAX computational functions."""

    def test_jax_availability(self):
        """Test JAX availability matches module state."""
        assert JAX_AVAILABLE == BACKEND_JAX_AVAILABLE

    def test_g1_diffusion_basic(self, jax_backend):
        """Test basic g1 diffusion computation."""
        t1 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        t2 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        q = 0.01
        D = 0.1

        result = compute_g1_diffusion(t1, t2, q, D)

        # Test shape preservation
        assert result.shape == t1.shape
        assert result.shape == t2.shape

        # Test physical constraints
        assert jnp.all(result >= 0.0), "g1_diffusion must be non-negative"
        assert jnp.all(result <= 1.0), "g1_diffusion must be <= 1.0"

        # Test diagonal elements (t1 = t2)
        diagonal = jnp.diag(result)
        expected_diagonal = jnp.ones(3)
        np.testing.assert_array_almost_equal(diagonal, expected_diagonal, decimal=6)

    def test_g1_diffusion_symmetry(self, jax_backend):
        """Test g1 diffusion symmetry properties."""
        n = 5
        t1, t2 = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        q = 0.015
        D = 0.05
        dt = 1.0  # Time step
        params = jnp.array([D, 0.0, 0.0])  # [D0, alpha, D_offset]

        result = compute_g1_diffusion(params, t1, t2, q, dt)

        # Test time-reversal symmetry: g1(t1,t2) = g1(t2,t1)
        result_transposed = compute_g1_diffusion(params, t2, t1, q, dt)
        np.testing.assert_array_almost_equal(result, result_transposed, decimal=8)

    def test_g1_shear_basic(self, jax_backend):
        """Test basic g1 shear computation."""
        phi = jnp.linspace(0, 2 * jnp.pi, 36)
        t1 = jnp.array([[0, 1, 2]])
        t2 = jnp.array([[0], [1], [2]])
        # Parameters: [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        params = jnp.array([1000.0, 0.5, 10.0, 0.1, 0.5, 0.01, 0.0])
        q = 0.01
        L = 1.0
        dt = 0.1

        result = compute_g1_shear(params, t1, t2, phi, q, L, dt)

        # Test shape
        expected_shape = (len(phi), 1, 1)
        assert result.shape == expected_shape

        # Test physical constraints
        assert jnp.all(result >= 0.0), "g1_shear must be non-negative"
        assert jnp.all(result <= 1.0), "g1_shear must be <= 1.0"

    def test_g1_shear_no_shear(self, jax_backend):
        """Test g1 shear with zero shear rate."""
        phi = jnp.linspace(0, 2 * jnp.pi, 24)
        t1 = jnp.ones((1, 5))
        t2 = jnp.ones((5, 1))
        # Parameters with zero shear: [D0, alpha, D_offset, gamma_dot_0=0, beta, gamma_dot_offset, phi0]
        params = jnp.array([1000.0, 0.5, 10.0, 0.0, 0.5, 0.0, 0.0])
        q = 0.01
        L = 1.0
        dt = 0.1

        result = compute_g1_shear(params, t1, t2, phi, q, L, dt)

        # With no shear, result should be all ones
        expected = jnp.ones_like(result)
        np.testing.assert_array_almost_equal(result, expected, decimal=10)

    def test_c2_model_basic(self, jax_backend):
        """Test basic c2 model computation."""
        n_times = 10
        n_angles = 12
        t1, t2 = jnp.meshgrid(jnp.arange(n_times), jnp.arange(n_times), indexing="ij")
        phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

        params = {
            "offset": 1.0,
            "contrast": 0.5,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }
        q = 0.01

        result = compute_c2_model_jax(params, t1, t2, phi, q)

        # Test shape
        expected_shape = (n_angles, n_times, n_times)
        assert result.shape == expected_shape

        # Test physical constraints
        assert jnp.all(result >= 1.0), "c2 must be >= 1.0 (physical minimum)"
        assert jnp.all(jnp.isfinite(result)), "c2 must be finite"

        # Test diagonal behavior
        diagonal_elements = result[:, jnp.arange(n_times), jnp.arange(n_times)]
        expected_diagonal = params["offset"] + params["contrast"]
        np.testing.assert_array_almost_equal(
            diagonal_elements,
            jnp.full_like(diagonal_elements, expected_diagonal),
            decimal=6,
        )

    def test_c2_model_parameter_sensitivity(self, jax_backend):
        """Test c2 model sensitivity to parameters."""
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0, jnp.pi])
        q = 0.01

        base_params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Test offset effect
        base_result = compute_c2_model_jax(base_params, t1, t2, phi, q)

        offset_params = base_params.copy()
        offset_params["offset"] = 1.2
        offset_result = compute_c2_model_jax(offset_params, t1, t2, phi, q)

        # Offset should shift entire result by 0.2
        expected_diff = 0.2
        actual_diff = jnp.mean(offset_result - base_result)
        np.testing.assert_almost_equal(actual_diff, expected_diff, decimal=6)

        # Test contrast effect
        contrast_params = base_params.copy()
        contrast_params["contrast"] = 0.6
        contrast_result = compute_c2_model_jax(contrast_params, t1, t2, phi, q)

        # Higher contrast should increase amplitude
        assert jnp.all(contrast_result >= base_result)

    def test_residuals_computation(self, jax_backend):
        """Test residuals computation."""
        # Create synthetic data
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Generate model data
        model = compute_c2_model_jax(params, t1, t2, phi, q)

        # Add some noise to create "experimental" data
        noise = jnp.array([[[0.01, -0.01], [-0.01, 0.01]]])
        c2_exp = model + noise
        sigma = jnp.ones_like(c2_exp) * 0.02

        residuals = residuals_jax(params, c2_exp, sigma, t1, t2, phi, q)

        # Test shape
        assert residuals.shape == c2_exp.shape

        # Test that residuals are reasonable
        assert jnp.all(jnp.abs(residuals) < 10.0), "Residuals too large"

    def test_chi_squared_computation(self, jax_backend):
        """Test chi-squared computation."""
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Perfect fit case
        model = compute_c2_model_jax(params, t1, t2, phi, q)
        sigma = jnp.ones_like(model) * 0.01

        chi2_perfect = chi_squared_jax(params, model, sigma, t1, t2, phi, q)

        # Perfect fit should have chi-squared ≈ 0
        assert chi2_perfect < 1e-10, "Perfect fit should have near-zero chi-squared"

        # Imperfect fit case
        c2_exp = model + 0.1  # Add constant offset
        chi2_imperfect = chi_squared_jax(params, c2_exp, sigma, t1, t2, phi, q)

        # Imperfect fit should have positive chi-squared
        assert chi2_imperfect > 0, "Imperfect fit should have positive chi-squared"
        assert chi2_imperfect > chi2_perfect, (
            "Imperfect fit should have higher chi-squared"
        )

    def test_jax_jit_compilation(self, jax_backend):
        """Test JAX JIT compilation works correctly."""
        from jax import jit

        # Test JIT compilation of c2 model
        jit_c2_model = jit(compute_c2_model_jax)

        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0, jnp.pi / 2])
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Regular computation
        result_regular = compute_c2_model_jax(params, t1, t2, phi, q)

        # JIT-compiled computation
        result_jit = jit_c2_model(params, t1, t2, phi, q)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_regular, result_jit, decimal=10)

    def test_automatic_differentiation(self, jax_backend):
        """Test automatic differentiation capabilities."""
        from jax import grad

        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        # Create a function that computes chi-squared for gradient testing
        def chi2_fn(param_array):
            params = {
                "offset": param_array[0],
                "contrast": param_array[1],
                "diffusion_coefficient": param_array[2],
                "shear_rate": 0.0,
                "L": 1.0,
            }

            # Synthetic data
            model = compute_c2_model_jax(params, t1, t2, phi, q)
            c2_exp = model + 0.01  # Add small noise
            sigma = jnp.ones_like(model) * 0.02

            return chi_squared_jax(params, c2_exp, sigma, t1, t2, phi, q)

        # Compute gradient
        grad_fn = grad(chi2_fn)
        param_values = jnp.array([1.0, 0.3, 0.1])
        gradient = grad_fn(param_values)

        # Test gradient shape and finiteness
        assert gradient.shape == param_values.shape
        assert jnp.all(jnp.isfinite(gradient)), "Gradient must be finite"

    def test_vectorization(self, jax_backend):
        """Test vectorization with vmap."""
        from jax import vmap

        # Test vectorization over multiple q values
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q_values = jnp.array([0.005, 0.01, 0.015])

        params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Vectorized computation
        vmap_c2 = vmap(
            lambda q: compute_c2_model_jax(params, t1, t2, phi, q), in_axes=0
        )
        result_vmap = vmap_c2(q_values)

        # Manual computation for comparison
        results_manual = []
        for q in q_values:
            result = compute_c2_model_jax(params, t1, t2, phi, q)
            results_manual.append(result)
        result_manual = jnp.stack(results_manual)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_vmap, result_manual, decimal=10)

        # Test shape
        expected_shape = (len(q_values), len(phi), t1.shape[0], t1.shape[1])
        assert result_vmap.shape == expected_shape


@pytest.mark.unit
class TestJAXBackendFallback:
    """Test JAX backend fallback behavior."""

    def test_fallback_availability(self):
        """Test that fallback functions are available when JAX is not."""
        # This test should always pass since we import fallbacks
        from homodyne.core.jax_backend import compute_c2_model_jax

        assert callable(compute_c2_model_jax)

    def test_numpy_compatibility(self, numpy_backend):
        """Test that functions work with numpy arrays."""
        t1 = np.array([[0, 1], [1, 0]])
        t2 = np.array([[0, 1], [1, 0]])
        phi = np.array([0.0])
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Should work with numpy arrays
        result = compute_c2_model_jax(params, t1, t2, phi, q)

        # Test basic properties
        assert hasattr(result, "shape")
        assert hasattr(result, "dtype")
        assert result.shape == (len(phi), t1.shape[0], t1.shape[1])


@pytest.mark.unit
@pytest.mark.property
class TestJAXBackendProperties:
    """Property-based tests for JAX backend mathematical properties."""

    @pytest.mark.requires_jax
    def test_g1_diffusion_monotonicity(self, jax_backend):
        """Test that g1 diffusion decreases with increasing time difference."""
        q = 0.01
        D = 0.1

        # Test monotonicity for increasing tau
        times = jnp.array([0, 1, 2, 3, 4])
        t1_base = jnp.zeros_like(times)
        t2_varying = times

        result = compute_g1_diffusion_jax(t1_base, t2_varying, q, D)

        # Should be monotonically decreasing
        assert jnp.all(result[:-1] >= result[1:]), (
            "g1_diffusion should decrease with tau"
        )

    @pytest.mark.requires_jax
    def test_c2_model_scaling(self, jax_backend):
        """Test c2 model scaling properties."""
        t1 = jnp.array([[0, 1], [1, 0]])
        t2 = jnp.array([[0, 1], [1, 0]])
        phi = jnp.array([0.0])
        q = 0.01

        base_params = {
            "offset": 1.0,
            "contrast": 0.5,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        base_result = compute_c2_model_jax(base_params, t1, t2, phi, q)

        # Test linear scaling with contrast
        for scale_factor in [0.5, 2.0, 1.5]:
            scaled_params = base_params.copy()
            scaled_params["contrast"] *= scale_factor

            scaled_result = compute_c2_model_jax(scaled_params, t1, t2, phi, q)

            # Check that scaling is correct
            expected_scaling = (scaled_result - base_params["offset"]) / (
                base_result - base_params["offset"]
            )
            np.testing.assert_array_almost_equal(
                expected_scaling,
                jnp.full_like(expected_scaling, scale_factor),
                decimal=6,
            )

    @pytest.mark.requires_jax
    def test_residuals_zero_mean(self, jax_backend):
        """Test that residuals have zero mean for perfect fit."""
        t1 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        t2 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        phi = jnp.linspace(0, 2 * jnp.pi, 8)
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.08,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Generate perfect model data
        c2_exp = compute_c2_model_jax(params, t1, t2, phi, q)
        sigma = jnp.ones_like(c2_exp) * 0.01

        # Compute residuals
        residuals = residuals_jax(params, c2_exp, sigma, t1, t2, phi, q)

        # Mean residual should be very close to zero
        mean_residual = jnp.mean(residuals)
        assert abs(mean_residual) < 1e-10, (
            "Mean residual should be zero for perfect fit"
        )


@pytest.mark.unit
@pytest.mark.requires_jax
class TestDispatcherMemory:
    """Test architectural fix for dispatcher memory allocation (Nov 2025)."""

    def test_meshgrid_mode_still_works(self, jax_backend):
        """Verify meshgrid mode still works for NLSQ (small arrays)."""
        # Small meshgrid for NLSQ (20x20 = 400 points, safe for memory)
        n = 20
        t1, t2 = jnp.meshgrid(
            jnp.linspace(0, 2, n), jnp.linspace(0, 2, n), indexing="ij"
        )
        phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])

        # Laminar flow parameters
        params = jnp.array([1000.0, 1.0, 10.0, 0.5, 1.0, 0.1, 0.0])
        q = 0.005
        L = 1.0
        dt = 0.1
        sinc_prefactor = 0.5 / jnp.pi * q * L * dt

        from homodyne.core.jax_backend import _compute_g1_shear_core

        # Call dispatcher (should use meshgrid JIT function)
        result = _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt)

        # Verify meshgrid mode was used (shape should be 3D)
        assert result.ndim == 3, "Meshgrid mode should return 3D array"
        assert result.shape == (3, 20, 20), f"Expected (3, 20, 20), got {result.shape}"

        # Verify physical constraints
        assert jnp.all(result >= 0.0), "g1_shear must be non-negative"
        assert jnp.all(result <= 1.0), "g1_shear must be <= 1.0"


@pytest.mark.unit
@pytest.mark.requires_jax
class TestParameterDependency:
    """
    Test that physics functions depend on input parameters.

    Historical Context:
    - During NLSQ zero-iteration debugging, we suspected parameters weren't
      affecting output, which would cause zero gradients
    - These tests verify parameters DO affect physics computations
    - Root cause was actually per-angle scaling incompatibility with chunking

    See: .ultra-think/ROOT_CAUSE_FOUND.md for full investigation
    """

    def test_g2_changes_with_different_parameters(self, jax_backend):
        """
        Test that changing parameters produces different g2 output.

        Note: Uses larger parameter changes because synthetic parameters
        may show weak dependence compared to real config parameters.
        """
        # Test setup
        t1 = np.linspace(0, 100, 51)
        t2 = np.linspace(0, 100, 51)
        phi = jnp.array([0.0])  # Single angle
        q = 0.005
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        # Laminar flow parameters with LARGE differences
        params1 = jnp.array([1000.0, 0.5, 10.0, 0.1, 0.5, 0.01, 0.0])
        params2 = jnp.array([10000.0, 1.5, 100.0, 0.5, 1.5, 0.05, 0.5])  # 10x larger

        # Compute g2 with both parameter sets
        g2_1 = compute_g2_scaled(
            params=params1,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )
        g2_2 = compute_g2_scaled(
            params=params2,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Verify outputs are different (very tolerant threshold for synthetic params)
        difference = jnp.abs(g2_2 - g2_1)
        max_diff = jnp.max(difference)

        # Parameters MUST affect output (regression test)
        # Very weak threshold because synthetic parameters show weak dependence
        assert max_diff > 1e-12, "Parameters must affect g2 output"

    def test_g2_gradient_is_nonzero(self, jax_backend):
        """Test that gradient of g2 with respect to parameters is non-zero."""
        # Test setup (smaller for gradient computation speed)
        t1 = np.linspace(0, 100, 51)
        t2 = np.linspace(0, 100, 51)
        phi = jnp.array([0.0])
        q = 0.005
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        # Laminar flow parameters
        params = jnp.array([1000.0, 0.5, 10.0, 0.1, 0.5, 0.01, 0.0])

        # Define loss function
        def loss_fn(params):
            g2 = compute_g2_scaled(
                params=params,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                contrast=contrast,
                offset=offset,
                dt=dt,
            )
            return jnp.sum(g2)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(params)

        # Verify gradient is non-zero
        gradient_norm = jnp.linalg.norm(gradient)
        assert gradient_norm > 1e-6, (
            f"Gradient norm {gradient_norm:.6e} is too small (should be >1e-6)"
        )
        assert jnp.all(jnp.isfinite(gradient)), "Gradient must be finite"

    # Note: More detailed parameter sensitivity tests with ACTUAL config parameters
    # are in test_parameter_gradients.py, which shows proper sensitivity with
    # realistic parameter values (gradient norm ~1972 vs synthetic params ~0.003)
