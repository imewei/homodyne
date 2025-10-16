"""
Unit tests for residual function creation in NLSQ wrapper.

Tests cover:
- Residual function signature and JAX compatibility (T012)
- Correct residual computation: (data - theory) / sigma
- Integration with homodyne physics models
"""

import jax.numpy as jnp
import numpy as np
import pytest

from homodyne.optimization.nlsq_wrapper import NLSQWrapper


class TestResidualFunctionCreation:
    """Test residual function creation (T012)."""

    def test_residual_function_signature(self):
        """
        Verify residual function has correct signature for NLSQ.

        NLSQ expects: f(xdata, *params) -> residuals
        """

        # Create mock XPCS data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, np.pi / 2, np.pi])
                self.t1 = np.linspace(0, 1, 10)
                self.t2 = np.linspace(0, 1, 10)
                self.g2 = np.random.rand(3, 10, 10)
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Create residual function
        residual_fn = wrapper._create_residual_function(
            mock_data, analysis_mode="static_isotropic"
        )

        # Verify it's callable
        assert callable(residual_fn), "Residual function must be callable"

        # Test function signature with dummy parameters
        # Static isotropic: [contrast, offset, D0, alpha, D_offset]
        n_params = 5
        xdata = np.arange(3 * 10 * 10, dtype=np.float64)  # Flattened size
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])

        # Call residual function
        residuals = residual_fn(xdata, *params)

        # Verify output shape matches flattened data
        assert residuals.shape == (
            300,
        ), f"Residuals should have shape (300,), got {residuals.shape}"
        assert isinstance(
            residuals, (np.ndarray, jnp.ndarray)
        ), "Residuals should be numpy or JAX array"

    def test_residual_computation_correctness(self):
        """
        Verify residual computation: residuals = (data - theory) / sigma.

        Test with known data where we can validate the computation.
        """
        # Create simple mock data with known structure
        n_phi, n_t1, n_t2 = 2, 3, 3

        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, np.pi])
                self.t1 = np.array([0.0, 0.5, 1.0])
                self.t2 = np.array([0.0, 0.5, 1.0])
                self.g2 = np.ones((n_phi, n_t1, n_t2)) * 1.5  # Constant data
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Create residual function
        residual_fn = wrapper._create_residual_function(
            mock_data, analysis_mode="static_isotropic"
        )

        # Call with parameters that should produce g2 ≈ 1.0
        # (contrast=0, offset=1.0 should give g2 = offset = 1.0)
        xdata = np.arange(n_phi * n_t1 * n_t2, dtype=np.float64)
        params = np.array([0.0, 1.0, 1000.0, 0.5, 10.0])

        residuals = residual_fn(xdata, *params)

        # Expected: (1.5 - 1.0) / 0.1 = 5.0 for all elements
        expected_residuals = np.ones(n_phi * n_t1 * n_t2) * 5.0

        np.testing.assert_allclose(
            residuals,
            expected_residuals,
            rtol=0.1,  # Allow 10% relative error due to g1 computation
            err_msg="Residuals don't match expected (data - theory) / sigma",
        )

    def test_residual_function_jax_jit_compatible(self):
        """
        Verify residual function is JAX JIT-compilable.

        Required for NLSQ optimization performance.
        """
        import jax

        # Create mock data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0])
                self.t1 = np.array([0.0, 1.0])
                self.t2 = np.array([0.0, 1.0])
                self.g2 = np.ones((1, 2, 2))
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        residual_fn = wrapper._create_residual_function(
            mock_data, analysis_mode="static_isotropic"
        )

        # Try to JIT compile (should not raise)
        try:
            jitted_fn = jax.jit(residual_fn)

            # Call JIT-compiled function
            xdata = jnp.arange(4, dtype=jnp.float64)
            params = jnp.array([0.5, 1.0, 1000.0, 0.5, 10.0])
            result = jitted_fn(xdata, *params)

            assert result.shape == (4,), "JIT-compiled function should work"
        except Exception as e:
            pytest.fail(f"Residual function should be JIT-compilable: {e}")

    def test_residual_function_laminar_flow_mode(self):
        """
        Verify residual function works with laminar flow analysis mode.

        Laminar flow: [contrast, offset, D0, alpha, D_offset,
                       gamma_dot_0, beta, gamma_dot_offset, phi0]
        """

        # Create mock data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, np.pi / 4, np.pi / 2])
                self.t1 = np.linspace(0, 1, 5)
                self.t2 = np.linspace(0, 1, 5)
                self.g2 = np.random.rand(3, 5, 5)
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Create residual function for laminar flow
        residual_fn = wrapper._create_residual_function(
            mock_data, analysis_mode="laminar_flow"
        )

        # Test with laminar flow parameters (9 total)
        xdata = np.arange(3 * 5 * 5, dtype=np.float64)
        params = np.array(
            [
                0.5,  # contrast
                1.0,  # offset
                1000.0,  # D0
                0.5,  # alpha
                10.0,  # D_offset
                1e-4,  # gamma_dot_0
                0.5,  # beta
                1e-5,  # gamma_dot_offset
                0.0,  # phi0
            ]
        )

        residuals = residual_fn(xdata, *params)

        assert residuals.shape == (
            75,
        ), f"Residuals should have shape (75,), got {residuals.shape}"
        assert not np.any(
            np.isnan(residuals)
        ), "Residuals should not contain NaN values"
        assert not np.any(
            np.isinf(residuals)
        ), "Residuals should not contain Inf values"

    def test_residual_function_missing_data_attributes(self):
        """Test graceful error handling for missing data attributes."""

        class IncompleteData:
            def __init__(self):
                self.phi = np.array([0.0])
                self.t1 = np.array([0.0])
                # Missing t2, g2, sigma, q, L

        wrapper = NLSQWrapper()

        with pytest.raises((AttributeError, ValueError)):
            wrapper._create_residual_function(
                IncompleteData(), analysis_mode="static_isotropic"
            )
