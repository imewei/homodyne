"""
Unit tests for parameter gradient computation with real config parameters.

This test suite verifies that gradients are computed correctly using actual
parameter values from configuration files, not synthetic test values.

Historical Context:
- During NLSQ zero-iteration debugging, we suspected zero gradients
- Tests with synthetic parameters showed weak gradients
- Tests with ACTUAL config parameters showed strong gradients (norm = 1972)
- This proved gradient computation works correctly with real parameters
- Root cause was per-angle scaling incompatibility with NLSQ chunking

See: .ultra-think/ROOT_CAUSE_FOUND.md for full investigation
"""

import pytest
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.jax_backend import compute_g2_scaled


@pytest.mark.unit
@pytest.mark.requires_jax
class TestGradientWithRealParameters:
    """Test gradient computation using realistic parameter values from configs."""

    def test_gradient_with_actual_config_parameters(self, jax_backend):
        """
        Test gradient computation with ACTUAL parameters from a real config file.

        This test uses the exact parameter values that were used in the
        laminar flow analysis that triggered the zero-iteration bug.
        """
        # ACTUAL values from config file (not synthetic!)
        # These are the exact parameters that showed gradient norm = 1972
        actual_params = jnp.array([
            57846.77371153954,      # D0
            -1.5050639269388213,    # alpha (NEGATIVE!)
            6454.490609318921,      # D_offset
            0.009160159749860066,   # gamma_dot_0
            -0.06976873166354103,   # beta (NEGATIVE!)
            -0.006662014902639189,  # gamma_dot_offset (NEGATIVE!)
            -12.736570086595812     # phi_0 (NEGATIVE!)
        ])

        # Real data setup matching actual analysis
        t1 = np.linspace(0, 100, 101)  # Reduced for test speed
        t2 = np.linspace(0, 100, 101)
        phi = jnp.array([0.0])
        q = 0.00532  # From real data
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        # Define loss function
        def loss_fn(params):
            g2 = compute_g2_scaled(
                params=params,
                t1=t1, t2=t2, phi=phi, q=q, L=L,
                contrast=contrast, offset=offset, dt=dt
            )
            return jnp.sum(g2)

        # Compute gradient
        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(actual_params)

        # Verify gradient is non-zero and large
        gradient_norm = jnp.linalg.norm(gradient)

        # With actual parameters, gradient norm was ~1972 (very large!)
        assert gradient_norm > 1.0, \
            f"Gradient norm {gradient_norm:.6e} should be >1.0 with real parameters"

        # Verify no parameters have zero gradient
        zero_grad_mask = jnp.abs(gradient) < 1e-10
        num_zero_grads = jnp.sum(zero_grad_mask)
        assert num_zero_grads == 0, \
            f"{num_zero_grads} parameters have zero gradient"

        # Verify all gradients are finite
        assert jnp.all(jnp.isfinite(gradient)), "All gradients must be finite"

    def test_parameter_perturbation_with_real_params(self, jax_backend):
        """Test that small perturbations to real parameters produce measurable changes."""
        # ACTUAL config parameters
        actual_params = jnp.array([
            57846.77371153954,
            -1.5050639269388213,
            6454.490609318921,
            0.009160159749860066,
            -0.06976873166354103,
            -0.006662014902639189,
            -12.736570086595812
        ])

        # Test setup
        t1 = np.linspace(0, 100, 51)
        t2 = np.linspace(0, 100, 51)
        phi = jnp.array([0.0])
        q = 0.00532
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        # Compute g2 with actual parameters
        g2_actual = compute_g2_scaled(
            params=actual_params,
            t1=t1, t2=t2, phi=phi, q=q, L=L,
            contrast=contrast, offset=offset, dt=dt
        )

        # Test 1% perturbation
        perturbed_params_1pct = actual_params * 1.01
        g2_1pct = compute_g2_scaled(
            params=perturbed_params_1pct,
            t1=t1, t2=t2, phi=phi, q=q, L=L,
            contrast=contrast, offset=offset, dt=dt
        )

        # Verify 1% change produces measurable difference
        diff_1pct = jnp.abs(g2_1pct - g2_actual)
        relative_diff_1pct = jnp.mean(diff_1pct) / jnp.mean(g2_actual)

        # Real parameters showed 0.1161% relative difference for 1% change
        assert relative_diff_1pct > 0.0001, \
            f"1% param change produced {relative_diff_1pct*100:.4f}% output change (should be >0.01%)"

        # Test 10% perturbation
        perturbed_params_10pct = actual_params * 1.10
        g2_10pct = compute_g2_scaled(
            params=perturbed_params_10pct,
            t1=t1, t2=t2, phi=phi, q=q, L=L,
            contrast=contrast, offset=offset, dt=dt
        )

        # Verify 10% change produces larger difference
        diff_10pct = jnp.abs(g2_10pct - g2_actual)
        relative_diff_10pct = jnp.mean(diff_10pct) / jnp.mean(g2_actual)

        # Real parameters showed 1.02% relative difference for 10% change
        assert relative_diff_10pct > 0.001, \
            f"10% param change produced {relative_diff_10pct*100:.4f}% output change (should be >0.1%)"

        # 10% change should produce larger effect than 1% change
        assert relative_diff_10pct > relative_diff_1pct, \
            "10% perturbation should have larger effect than 1% perturbation"

    def test_negative_parameters_handled_correctly(self, jax_backend):
        """
        Test that negative parameters (common in real configs) are handled correctly.

        Many real parameters are negative: alpha, beta, gamma_dot_offset, phi_0
        """
        # Parameters with negative values (from real config)
        params_with_negatives = jnp.array([
            57846.77,    # D0 (positive)
            -1.505,      # alpha (NEGATIVE)
            6454.49,     # D_offset (positive)
            0.00916,     # gamma_dot_0 (positive)
            -0.0698,     # beta (NEGATIVE)
            -0.00666,    # gamma_dot_offset (NEGATIVE)
            -12.737      # phi_0 (NEGATIVE)
        ])

        # Test setup
        t1 = np.linspace(0, 100, 51)
        t2 = np.linspace(0, 100, 51)
        phi = jnp.array([0.0])
        q = 0.00532
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        # Should not raise any errors
        g2 = compute_g2_scaled(
            params=params_with_negatives,
            t1=t1, t2=t2, phi=phi, q=q, L=L,
            contrast=contrast, offset=offset, dt=dt
        )

        # Verify output is physically reasonable
        assert g2.shape[0] == len(phi), "Output shape must match phi"
        assert jnp.all(jnp.isfinite(g2)), "Output must be finite with negative parameters"
        assert jnp.all(g2 >= 0), "g2 must be non-negative even with negative parameters"

        # Gradient computation should also work
        def loss_fn(params):
            g2 = compute_g2_scaled(
                params=params,
                t1=t1, t2=t2, phi=phi, q=q, L=L,
                contrast=contrast, offset=offset, dt=dt
            )
            return jnp.sum(g2)

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(params_with_negatives)

        assert jnp.all(jnp.isfinite(gradient)), "Gradient must be finite with negative parameters"

    def test_gradient_consistency_across_parameter_scales(self, jax_backend):
        """Test that gradient computation is consistent across different parameter scales."""
        # Test setup
        t1 = np.linspace(0, 100, 51)
        t2 = np.linspace(0, 100, 51)
        phi = jnp.array([0.0])
        q = 0.00532
        L = 2_000_000.0
        dt = 0.1
        contrast = 0.5
        offset = 1.0

        def compute_gradient_norm(params):
            def loss_fn(p):
                g2 = compute_g2_scaled(
                    params=p,
                    t1=t1, t2=t2, phi=phi, q=q, L=L,
                    contrast=contrast, offset=offset, dt=dt
                )
                return jnp.sum(g2)

            grad_fn = jax.grad(loss_fn)
            gradient = grad_fn(params)
            return jnp.linalg.norm(gradient)

        # Test with real config parameters (large scale: D0 ~ 58000)
        real_params = jnp.array([
            57846.77, -1.505, 6454.49, 0.00916, -0.0698, -0.00666, -12.737
        ])
        grad_norm_real = compute_gradient_norm(real_params)

        # Test with scaled-down parameters (moderate scale: D0 ~ 1000)
        moderate_params = jnp.array([
            1000.0, -1.0, 100.0, 0.01, -0.05, -0.01, -10.0
        ])
        grad_norm_moderate = compute_gradient_norm(moderate_params)

        # Both should have non-zero gradients (actual values will differ)
        assert grad_norm_real > 1.0, f"Real params gradient norm {grad_norm_real:.2f} should be >1.0"
        assert grad_norm_moderate > 1e-3, f"Moderate params gradient norm {grad_norm_moderate:.6e} should be >1e-3"


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v"])
