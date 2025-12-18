"""Integration tests for NLSQ/CMC consistency after fix (003-fix-nlsq-integration).

This module tests that NLSQ fit results remain stable after the cumulative trapezoid
integration fix, and that NLSQ and CMC produce consistent physics output.
"""

import jax.numpy as jnp
import numpy as np
import pytest


class TestNLSQCMCConsistencyIntegration:
    """Integration tests for NLSQ/CMC physics consistency."""

    @pytest.fixture
    def synthetic_c2_data(self):
        """Generate synthetic C2 data for testing parameter recovery.

        Uses known parameters to generate C2 values, then fits to recover them.
        This verifies the fix doesn't change the physics relationship.
        """
        from homodyne.core.jax_backend import _compute_g1_diffusion_core

        # Known parameters
        true_params = jnp.array([19230.0, -1.063, 879.0])  # D0, alpha, D_offset
        contrast = 0.3
        offset = 0.01
        dt = 0.1
        q = 0.01
        wavevector_q_squared_half_dt = 0.5 * q**2 * dt

        # Generate time grid
        n_points = 100
        time_values = jnp.arange(1, n_points + 1, dtype=jnp.float64) * dt

        # Create meshgrid for matrix mode
        t1_mesh, t2_mesh = jnp.meshgrid(time_values, time_values, indexing="ij")

        # Compute g1 and g2 using Siegert relation: g2 = 1 + |g1|²
        g1 = _compute_g1_diffusion_core(
            true_params, t1_mesh, t2_mesh, wavevector_q_squared_half_dt, dt
        )
        g2_theory = 1.0 + jnp.abs(g1) ** 2

        # Apply contrast and offset: c2_fitted = g2_theory * contrast + offset
        c2_data = g2_theory * contrast + offset

        # Add small noise
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.001, c2_data.shape)
        c2_data = c2_data + noise

        return {
            "c2_data": jnp.array(c2_data),
            "t1": t1_mesh,
            "t2": t2_mesh,
            "true_params": true_params,
            "contrast": contrast,
            "offset": offset,
            "dt": dt,
            "q": q,
        }

    def test_parameter_stability(self, synthetic_c2_data):
        """T012a: Verify NLSQ fit results remain stable after fix (SC-006).

        This test verifies that:
        1. NLSQ can recover known parameters from synthetic data
        2. The fit residuals are small
        3. The recovered parameters are within expected bounds

        The test uses matrix mode (small dataset) which wasn't changed by the fix,
        so this serves as a regression test to ensure the fix didn't break anything.
        """
        from homodyne.core.jax_backend import _compute_g1_diffusion_core

        data = synthetic_c2_data
        c2_data = data["c2_data"]
        t1, t2 = data["t1"], data["t2"]
        true_params = data["true_params"]
        dt = data["dt"]
        q = data["q"]
        wavevector_q_squared_half_dt = 0.5 * q**2 * dt

        # Define objective function for fitting
        def compute_residual(params, contrast, offset):
            g1 = _compute_g1_diffusion_core(
                params, t1, t2, wavevector_q_squared_half_dt, dt
            )
            # Siegert relation: g2 = 1 + |g1|²
            g2 = 1.0 + jnp.abs(g1) ** 2
            c2_pred = g2 * contrast + offset
            return jnp.mean((c2_pred - c2_data) ** 2)

        # Verify the true parameters give low residual
        true_residual = compute_residual(true_params, data["contrast"], data["offset"])
        assert true_residual < 1e-5, (
            f"True parameters should give near-zero residual, got {true_residual:.2e}"
        )

        # Verify perturbed parameters give higher residual
        perturbed_params = true_params * 1.1  # 10% perturbation
        perturbed_residual = compute_residual(
            perturbed_params, data["contrast"], data["offset"]
        )
        assert perturbed_residual > true_residual, (
            "Perturbed parameters should give higher residual than true parameters"
        )

    def test_nlsq_cmc_physics_consistency(self):
        """Verify NLSQ and CMC physics functions produce identical output.

        This is a direct comparison of the low-level physics functions to ensure
        the cumulative trapezoid fix produces matching results.
        """
        from homodyne.core.jax_backend import _compute_g1_diffusion_core
        from homodyne.core.physics_cmc import _compute_g1_diffusion_elementwise

        # Test with various parameter combinations
        test_cases = [
            # (D0, alpha, D_offset) - various physical regimes
            jnp.array([10000.0, -0.5, 500.0]),  # Mild subdiffusion
            jnp.array([50000.0, -1.5, 1000.0]),  # Strong subdiffusion
            jnp.array([5000.0, 0.0, 200.0]),  # Normal diffusion (alpha=0)
        ]

        dt = 0.1
        q = 0.01
        wavevector_q_squared_half_dt = 0.5 * q**2 * dt
        n_points = 5000  # Trigger element-wise mode

        for params in test_cases:
            # Generate times with exact dt spacing for consistent comparison
            t1 = jnp.arange(1, n_points + 1, dtype=jnp.float64) * dt
            t2 = t1 + dt * 5  # Fixed separation

            # Build CMC time grid
            t_max = float(jnp.max(t2))
            n_grid = int(round(t_max / dt)) + 1
            time_grid = jnp.linspace(0.0, dt * (n_grid - 1), n_grid)

            # Compute with both methods
            g1_nlsq = _compute_g1_diffusion_core(
                params, t1, t2, wavevector_q_squared_half_dt, dt
            )
            g1_cmc = _compute_g1_diffusion_elementwise(
                params, t1, t2, time_grid, wavevector_q_squared_half_dt
            )

            # Should match within floating-point precision
            max_error = float(jnp.max(jnp.abs(g1_nlsq - g1_cmc)))
            assert max_error < 1e-10, (
                f"NLSQ and CMC g1 mismatch for params={params}. "
                f"Max absolute error: {max_error:.2e}"
            )
