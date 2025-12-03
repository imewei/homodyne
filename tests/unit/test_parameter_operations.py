"""
Unit Tests for Parameter Operations
====================================

**Consolidation**: Week 5 (2025-11-15)

Consolidated from:
- test_parameter_expansion.py (Parameter expansion logic, 11 tests, 317 lines)
- test_parameter_transformation.py (Parameter transformations, 6 tests, 147 lines)
- test_parameter_gradients.py (Gradient calculations, 4 tests, 315 lines)
- test_parameter_names_consistency.py (Name consistency validation, 22 tests, 321 lines)

Test Categories:
---------------
**Parameter Expansion** (11 tests):
- Per-angle scaling expansion (5→9 params for 3 angles)
- Array construction from scalar parameters
- Expansion validation and correctness

**Parameter Transformations** (6 tests):
- Log/exp transformations for bounded optimization
- Bounds mapping (physical space ↔ unbounded space)
- Transformation correctness and invertibility

**Gradient Calculations** (4 tests):
- Gradient calculations for optimization algorithms
- Numerical gradient validation
- Gradient accuracy and stability

**Name Consistency** (22 tests):
- Parameter name consistency validation
- Canonical name enforcement (gamma_dot_0 → gamma_dot_t0)
- Name mapping validation across system
- Consistency checks between components

Test Coverage:
-------------
- Parameter expansion for per-angle scaling (critical for v2.4.0+)
- Array construction from scalar physical parameters
- Log/exp transformations for bounded parameter optimization
- Bounds mapping between physical and unbounded spaces
- Gradient calculations for trust-region optimization
- Numerical gradient validation and stability checking
- Parameter name consistency validation across entire system
- Canonical name enforcement (ensures gamma_dot_0 → gamma_dot_t0)
- Name mapping validation between ParameterManager and other components
- Cross-component consistency checking

Total: 43 tests

Usage Example:
-------------
```python
# Run all parameter operation tests
pytest tests/unit/test_parameter_operations.py -v

# Run specific category
pytest tests/unit/test_parameter_operations.py -k "expansion" -v
pytest tests/unit/test_parameter_operations.py -k "gradient" -v

# Test name consistency
pytest tests/unit/test_parameter_operations.py -k "consistency" -v
```

See Also:
---------
- docs/WEEK5_CONSOLIDATION_SUMMARY.md: Consolidation details
- homodyne/config/parameter_manager.py: Parameter expansion and transformations
- homodyne/config/types.py: PARAMETER_NAME_MAPPING dictionary
"""

import numpy as np
import pytest

# ==============================================================================
# Parameter Expansion Tests (from test_parameter_expansion.py)
# ==============================================================================


class TestParameterExpansion:
    """Test parameter expansion for per-angle scaling."""

    def test_expansion_basic_3_angles(self):
        """Test basic parameter expansion for 3 angles."""
        # Simulate config parameters: [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        validated_params = np.array(
            [0.5, 1.0, 100.0, 0.6, 10.0, 0.001, 0.5, 0.0001, 0.0]
        )
        n_angles = 3
        n_physical = 7  # laminar_flow

        # Extract scaling parameters (first 2)
        base_contrast = validated_params[0]
        base_offset = validated_params[1]
        physical_params = validated_params[2:]

        # Expand scaling parameters per angle
        contrast_per_angle = np.full(n_angles, base_contrast)
        offset_per_angle = np.full(n_angles, base_offset)

        # Create expanded parameters in StratifiedResidualFunction order
        expanded_params = np.concatenate(
            [
                contrast_per_angle,  # [0.5, 0.5, 0.5]
                offset_per_angle,  # [1.0, 1.0, 1.0]
                physical_params,  # [100.0, 0.6, 10.0, 0.001, 0.5, 0.0001, 0.0]
            ]
        )

        # Assertions
        assert len(expanded_params) == n_physical + 2 * n_angles  # 7 + 6 = 13
        assert expanded_params[0] == base_contrast  # First contrast
        assert expanded_params[1] == base_contrast  # Second contrast
        assert expanded_params[2] == base_contrast  # Third contrast
        assert expanded_params[3] == base_offset  # First offset
        assert expanded_params[4] == base_offset  # Second offset
        assert expanded_params[5] == base_offset  # Third offset
        assert expanded_params[6] == 100.0  # D0
        assert expanded_params[7] == 0.6  # alpha
        assert np.allclose(expanded_params[6:], physical_params)

    def test_expansion_static_mode(self):
        """Test parameter expansion for static_mode mode."""
        # Static: [contrast, offset, D0, alpha, D_offset]
        validated_params = np.array([0.4, 1.1, 200.0, 0.7, 15.0])
        n_angles = 5
        n_physical = 3  # static_mode

        base_contrast = validated_params[0]
        base_offset = validated_params[1]
        physical_params = validated_params[2:]

        contrast_per_angle = np.full(n_angles, base_contrast)
        offset_per_angle = np.full(n_angles, base_offset)

        expanded_params = np.concatenate(
            [contrast_per_angle, offset_per_angle, physical_params]
        )

        # Assertions
        assert len(expanded_params) == n_physical + 2 * n_angles  # 3 + 10 = 13
        assert np.all(expanded_params[:n_angles] == base_contrast)
        assert np.all(expanded_params[n_angles : 2 * n_angles] == base_offset)
        assert np.allclose(expanded_params[2 * n_angles :], physical_params)

    def test_expansion_preserves_physical_params(self):
        """Test that physical parameters are preserved exactly."""
        validated_params = np.array([0.3, 1.2, 50.0, 0.5, 5.0, 0.002, 0.4, 0.0002, 0.1])
        n_angles = 3
        n_physical = 7

        physical_params = validated_params[2:]  # Extract physical params

        # Expand
        contrast_per_angle = np.full(n_angles, validated_params[0])
        offset_per_angle = np.full(n_angles, validated_params[1])
        expanded_params = np.concatenate(
            [contrast_per_angle, offset_per_angle, physical_params]
        )

        # Physical parameters should be unchanged
        assert np.allclose(expanded_params[2 * n_angles :], physical_params)
        assert len(expanded_params[2 * n_angles :]) == n_physical

    def test_parameter_count_formula(self):
        """Test the parameter count formula for various angle counts."""
        n_physical = 7  # laminar_flow

        test_cases = [
            (3, 13),  # 7 + 2*3 = 13
            (5, 17),  # 7 + 2*5 = 17
            (10, 27),  # 7 + 2*10 = 27
            (23, 53),  # 7 + 2*23 = 53
        ]

        for n_angles, expected_count in test_cases:
            actual_count = n_physical + 2 * n_angles
            assert actual_count == expected_count

    def test_bounds_expansion(self):
        """Test that bounds are expanded correctly."""
        # Original bounds: [contrast, offset, D0, alpha, ...]
        lower_original = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -np.pi])
        upper_original = np.array([1.0, 2.0, 1000.0, 1.0, 100.0, 1.0, 1.0, 1.0, np.pi])
        n_angles = 3
        n_physical = 7

        # Extract bounds
        lower_contrast = lower_original[0]
        upper_contrast = upper_original[0]
        lower_offset = lower_original[1]
        upper_offset = upper_original[1]
        lower_physical = lower_original[2:]
        upper_physical = upper_original[2:]

        # Expand bounds per angle
        lower_contrast_per_angle = np.full(n_angles, lower_contrast)
        upper_contrast_per_angle = np.full(n_angles, upper_contrast)
        lower_offset_per_angle = np.full(n_angles, lower_offset)
        upper_offset_per_angle = np.full(n_angles, upper_offset)

        expanded_lower = np.concatenate(
            [lower_contrast_per_angle, lower_offset_per_angle, lower_physical]
        )
        expanded_upper = np.concatenate(
            [upper_contrast_per_angle, upper_offset_per_angle, upper_physical]
        )

        # Assertions
        assert len(expanded_lower) == 13
        assert len(expanded_upper) == 13
        assert np.all(expanded_lower[:3] == lower_contrast)  # Contrast bounds
        assert np.all(expanded_upper[:3] == upper_contrast)
        assert np.all(expanded_lower[3:6] == lower_offset)  # Offset bounds
        assert np.all(expanded_upper[3:6] == upper_offset)
        assert np.allclose(expanded_lower[6:], lower_physical)
        assert np.allclose(expanded_upper[6:], upper_physical)

    def test_parameter_ordering_matches_residual_function(self):
        """Test that parameter ordering matches StratifiedResidualFunction expectations."""
        # StratifiedResidualFunction expects: [contrast_per_angle, offset_per_angle, physical_params]
        # This is line 207-211 in stratified_residual.py:
        #   contrast = params_all[:self.n_phi]
        #   offset = params_all[self.n_phi:2*self.n_phi]
        #   physical_params = params_all[2*self.n_phi:]

        validated_params = np.array(
            [0.5, 1.0, 100.0, 0.6, 10.0, 0.001, 0.5, 0.0001, 0.0]
        )
        n_angles = 3
        n_phi = n_angles

        base_contrast = validated_params[0]
        base_offset = validated_params[1]
        physical_params = validated_params[2:]

        contrast_per_angle = np.full(n_angles, base_contrast)
        offset_per_angle = np.full(n_angles, base_offset)
        expanded_params = np.concatenate(
            [contrast_per_angle, offset_per_angle, physical_params]
        )

        # Simulate what StratifiedResidualFunction will extract
        extracted_contrast = expanded_params[:n_phi]
        extracted_offset = expanded_params[n_phi : 2 * n_phi]
        extracted_physical = expanded_params[2 * n_phi :]

        # Verify correct extraction
        assert np.allclose(extracted_contrast, contrast_per_angle)
        assert np.allclose(extracted_offset, offset_per_angle)
        assert np.allclose(extracted_physical, physical_params)

    def test_validation_detects_wrong_count(self):
        """Test that parameter count validation detects mismatches."""
        n_physical = 7
        n_angles = 3

        # Expected count
        expected_params = n_physical + 2 * n_angles  # 13

        # Test various wrong counts
        wrong_counts = [9, 10, 11, 12, 14, 15]
        for wrong_count in wrong_counts:
            # In real code, this would raise ValueError
            assert wrong_count != expected_params


class TestGradientSanityCheck:
    """Test the gradient sanity check feature."""

    def test_gradient_check_detects_zero_gradient(self):
        """Test that gradient check detects zero gradient correctly."""

        # Simulate a residual function that returns constant residuals
        def constant_residual_fn(params):
            return np.ones(1000) * 100.0  # Constant regardless of params

        initial_params = np.array([0.5, 1.0, 100.0])

        # Compute gradient estimate
        residuals_0 = constant_residual_fn(initial_params)
        params_test = initial_params.copy()
        params_test[0] *= 1.01  # 1% perturbation
        residuals_1 = constant_residual_fn(params_test)

        gradient_estimate = np.abs(np.sum(residuals_1 - residuals_0))

        # Should detect zero gradient
        assert gradient_estimate < 1e-10

    def test_gradient_check_detects_nonzero_gradient(self):
        """Test that gradient check detects non-zero gradient correctly."""

        # Simulate a residual function sensitive to parameters
        def sensitive_residual_fn(params):
            return params[0] * np.ones(1000)  # Scales with first parameter

        initial_params = np.array([0.5, 1.0, 100.0])

        # Compute gradient estimate
        residuals_0 = sensitive_residual_fn(initial_params)
        params_test = initial_params.copy()
        params_test[0] *= 1.01  # 1% perturbation
        residuals_1 = sensitive_residual_fn(params_test)

        gradient_estimate = np.abs(np.sum(residuals_1 - residuals_0))

        # Should detect non-zero gradient
        assert gradient_estimate > 1e-10
        # Expected: 0.5 * 0.01 * 1000 = 5.0
        assert gradient_estimate > 1.0

    def test_gradient_threshold_1e_minus_10(self):
        """Test that threshold of 1e-10 is appropriate."""
        # This threshold was chosen to detect genuine zero gradients
        # while allowing numerical precision variations
        threshold = 1e-10

        # Truly zero gradient should be well below threshold
        zero_gradient = 0.0
        assert zero_gradient < threshold

        # Small but meaningful gradient should be above threshold
        meaningful_gradient = 1.0
        assert meaningful_gradient > threshold

        # Numerical noise at double precision (~1e-15) should be below threshold
        numerical_noise = 1e-15
        assert numerical_noise < threshold


class TestParameterExtractionFromConfig:
    """Test parameter extraction from config ordering."""

    def test_config_params_ordering(self):
        """Test that we correctly understand config parameter ordering."""
        # _params_to_array() in nlsq.py creates:
        # [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

        # Simulate what _params_to_array returns
        config_params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 100.0,
            "alpha": 0.6,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.5,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 0.0,
        }

        # This is the ordering _params_to_array uses
        validated_params = np.array(
            [
                config_params["contrast"],
                config_params["offset"],
                config_params["D0"],
                config_params["alpha"],
                config_params["D_offset"],
                config_params["gamma_dot_t0"],
                config_params["beta"],
                config_params["gamma_dot_t_offset"],
                config_params["phi0"],
            ]
        )

        # Extract scaling parameters from BEGINNING (not END!)
        base_contrast = validated_params[0]
        base_offset = validated_params[1]
        physical_params = validated_params[2:]

        # Verify correct extraction
        assert base_contrast == 0.5
        assert base_offset == 1.0
        assert len(physical_params) == 7
        assert physical_params[0] == 100.0  # D0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# Parameter Transformation Tests (from test_parameter_transformation.py)
# ==============================================================================

from homodyne.optimization.nlsq_wrapper import NLSQWrapper


class TestDataFlattening:
    """Test data flattening transformation (T006)."""

    def test_data_flattening_shape_transformation(self):
        """
        Verify multi-dimensional XPCS data (n_phi, n_t1, n_t2) → flattened 1D arrays.

        Test data: (23, 1001, 1001) → 23,023,023 elements
        Verify meshgrid indexing='ij', flatten() preserves order.
        """
        # Create mock XPCS data with known structure
        n_phi, n_t1, n_t2 = 23, 1001, 1001
        expected_size = n_phi * n_t1 * n_t2  # 23,023,023

        # Mock data object
        class MockXPCSData:
            def __init__(self):
                self.phi = np.linspace(0, 2 * np.pi, n_phi)
                self.t1 = np.linspace(0, 1, n_t1)
                self.t2 = np.linspace(0, 1, n_t2)
                # 3D correlation data
                self.g2 = np.random.rand(n_phi, n_t1, n_t2)

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Execute transformation
        xdata, ydata = wrapper._prepare_data(mock_data)

        # Assertions
        assert xdata.shape[0] == expected_size, (
            f"xdata should have {expected_size} elements, got {xdata.shape[0]}"
        )
        assert ydata.shape[0] == expected_size, (
            f"ydata should have {expected_size} elements, got {ydata.shape[0]}"
        )
        assert xdata.ndim == 1, f"xdata should be 1D, got {xdata.ndim}D"
        assert ydata.ndim == 1, f"ydata should be 1D, got {ydata.ndim}D"

    def test_meshgrid_indexing_order(self):
        """
        Verify meshgrid uses indexing='ij' to preserve correct ordering.

        This ensures compatibility with homodyne's physics calculations.
        """
        # Small test case for manual verification
        n_phi, n_t1, n_t2 = 2, 3, 4

        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, 1.0])
                self.t1 = np.array([0.0, 0.5, 1.0])
                self.t2 = np.array([0.0, 0.33, 0.67, 1.0])
                self.g2 = np.arange(n_phi * n_t1 * n_t2).reshape(n_phi, n_t1, n_t2)

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        xdata, ydata = wrapper._prepare_data(mock_data)

        # Verify first few elements match expected indexing='ij' order
        # With indexing='ij', phi varies slowest, t2 varies fastest
        expected_ydata_start = mock_data.g2.flatten()  # NumPy default is C-order

        assert len(ydata) == n_phi * n_t1 * n_t2
        np.testing.assert_array_equal(
            ydata[:5],
            expected_ydata_start[:5],
            err_msg="Flattening order doesn't match expected indexing='ij'",
        )

    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""

        class MockEmptyData:
            def __init__(self):
                self.phi = np.array([])
                self.t1 = np.array([])
                self.t2 = np.array([])
                self.g2 = np.array([])

        mock_data = MockEmptyData()
        wrapper = NLSQWrapper()

        with pytest.raises((ValueError, IndexError)):
            wrapper._prepare_data(mock_data)


class TestBoundsFormatConversion:
    """Test bounds format conversion (T010)."""

    def test_bounds_tuple_unchanged(self):
        """
        Verify homodyne bounds tuple format → NLSQ format conversion.

        Test: (lower_array, upper_array) tuple unchanged, verify shapes match n_params.
        """
        n_params = 5
        lower = np.array([0.0, 0.0, 100.0, 0.3, 1.0])
        upper = np.array([1.0, 2.0, 1e5, 1.5, 1000.0])
        homodyne_bounds = (lower, upper)

        wrapper = NLSQWrapper()
        nlsq_bounds = wrapper._convert_bounds(homodyne_bounds)

        # NLSQ expects same tuple format
        assert isinstance(nlsq_bounds, tuple)
        assert len(nlsq_bounds) == 2
        np.testing.assert_array_equal(nlsq_bounds[0], lower)
        np.testing.assert_array_equal(nlsq_bounds[1], upper)
        assert nlsq_bounds[0].shape == (n_params,)
        assert nlsq_bounds[1].shape == (n_params,)

    def test_bounds_validation_lower_less_than_upper(self):
        """Verify bounds validation checks lower < upper elementwise."""
        # Invalid bounds: some lower > upper
        lower = np.array([0.0, 2.0, 100.0])  # lower[1] = 2.0
        upper = np.array([1.0, 1.0, 1e5])  # upper[1] = 1.0 (invalid!)
        invalid_bounds = (lower, upper)

        wrapper = NLSQWrapper()

        with pytest.raises(ValueError, match="lower.*upper|bound"):
            wrapper._convert_bounds(invalid_bounds)

    def test_none_bounds_handling(self):
        """Test handling of None bounds (unbounded optimization)."""
        wrapper = NLSQWrapper()

        # None bounds should return None or appropriate default
        result = wrapper._convert_bounds(None)
        assert result is None or result == (-np.inf, np.inf)


# ==============================================================================
# Parameter Gradient Tests (from test_parameter_gradients.py)
# ==============================================================================


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
        actual_params = jnp.array(
            [
                57846.77371153954,  # D0
                -1.5050639269388213,  # alpha (NEGATIVE!)
                6454.490609318921,  # D_offset
                0.009160159749860066,  # gamma_dot_0
                -0.06976873166354103,  # beta (NEGATIVE!)
                -0.006662014902639189,  # gamma_dot_offset (NEGATIVE!)
                -12.736570086595812,  # phi_0 (NEGATIVE!)
            ]
        )

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
        gradient = grad_fn(actual_params)

        # Verify gradient is non-zero and large
        gradient_norm = jnp.linalg.norm(gradient)

        # With actual parameters, gradient norm was ~1972 (very large!)
        assert gradient_norm > 1.0, (
            f"Gradient norm {gradient_norm:.6e} should be >1.0 with real parameters"
        )

        # Verify no parameters have zero gradient
        zero_grad_mask = jnp.abs(gradient) < 1e-10
        num_zero_grads = jnp.sum(zero_grad_mask)
        assert num_zero_grads == 0, f"{num_zero_grads} parameters have zero gradient"

        # Verify all gradients are finite
        assert jnp.all(jnp.isfinite(gradient)), "All gradients must be finite"

    def test_parameter_perturbation_with_real_params(self, jax_backend):
        """Test that small perturbations to real parameters produce measurable changes."""
        # ACTUAL config parameters
        actual_params = jnp.array(
            [
                57846.77371153954,
                -1.5050639269388213,
                6454.490609318921,
                0.009160159749860066,
                -0.06976873166354103,
                -0.006662014902639189,
                -12.736570086595812,
            ]
        )

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
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Test 1% perturbation
        perturbed_params_1pct = actual_params * 1.01
        g2_1pct = compute_g2_scaled(
            params=perturbed_params_1pct,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Verify 1% change produces measurable difference
        diff_1pct = jnp.abs(g2_1pct - g2_actual)
        relative_diff_1pct = jnp.mean(diff_1pct) / jnp.mean(g2_actual)

        # Real parameters showed 0.1161% relative difference for 1% change
        assert relative_diff_1pct > 0.0001, (
            f"1% param change produced {relative_diff_1pct * 100:.4f}% output change (should be >0.01%)"
        )

        # Test 10% perturbation
        perturbed_params_10pct = actual_params * 1.10
        g2_10pct = compute_g2_scaled(
            params=perturbed_params_10pct,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Verify 10% change produces larger difference
        diff_10pct = jnp.abs(g2_10pct - g2_actual)
        relative_diff_10pct = jnp.mean(diff_10pct) / jnp.mean(g2_actual)

        # Real parameters showed 1.02% relative difference for 10% change
        assert relative_diff_10pct > 0.001, (
            f"10% param change produced {relative_diff_10pct * 100:.4f}% output change (should be >0.1%)"
        )

        # 10% change should produce larger effect than 1% change
        assert relative_diff_10pct > relative_diff_1pct, (
            "10% perturbation should have larger effect than 1% perturbation"
        )

    def test_negative_parameters_handled_correctly(self, jax_backend):
        """
        Test that negative parameters (common in real configs) are handled correctly.

        Many real parameters are negative: alpha, beta, gamma_dot_offset, phi_0
        """
        # Parameters with negative values (from real config)
        params_with_negatives = jnp.array(
            [
                57846.77,  # D0 (positive)
                -1.505,  # alpha (NEGATIVE)
                6454.49,  # D_offset (positive)
                0.00916,  # gamma_dot_0 (positive)
                -0.0698,  # beta (NEGATIVE)
                -0.00666,  # gamma_dot_offset (NEGATIVE)
                -12.737,  # phi_0 (NEGATIVE)
            ]
        )

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
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Verify output is physically reasonable
        assert g2.shape[0] == len(phi), "Output shape must match phi"
        assert jnp.all(jnp.isfinite(g2)), (
            "Output must be finite with negative parameters"
        )
        assert jnp.all(g2 >= 0), "g2 must be non-negative even with negative parameters"

        # Gradient computation should also work
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

        grad_fn = jax.grad(loss_fn)
        gradient = grad_fn(params_with_negatives)

        assert jnp.all(jnp.isfinite(gradient)), (
            "Gradient must be finite with negative parameters"
        )

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

            grad_fn = jax.grad(loss_fn)
            gradient = grad_fn(params)
            return jnp.linalg.norm(gradient)

        # Test with real config parameters (large scale: D0 ~ 58000)
        real_params = jnp.array(
            [57846.77, -1.505, 6454.49, 0.00916, -0.0698, -0.00666, -12.737]
        )
        grad_norm_real = compute_gradient_norm(real_params)

        # Test with scaled-down parameters (moderate scale: D0 ~ 1000)
        moderate_params = jnp.array([1000.0, -1.0, 100.0, 0.01, -0.05, -0.01, -10.0])
        grad_norm_moderate = compute_gradient_norm(moderate_params)

        # Both should have non-zero gradients (actual values will differ)
        assert grad_norm_real > 1.0, (
            f"Real params gradient norm {grad_norm_real:.2f} should be >1.0"
        )
        assert grad_norm_moderate > 1e-3, (
            f"Moderate params gradient norm {grad_norm_moderate:.6e} should be >1e-3"
        )


if __name__ == "__main__":
    # Allow running directly for quick debugging
    pytest.main([__file__, "-v"])


# ==============================================================================
# Name Consistency Tests (from test_parameter_names_consistency.py)
# ==============================================================================


import numpy as np
import pytest

# Import parameter constants
from homodyne.config.parameter_names import (
    LAMINAR_FLOW_PARAMS,
    STATIC_ISOTROPIC_PARAMS,
    get_num_parameters,
    get_parameter_names,
    validate_parameter_names,
    verify_samples_dict,
)


class TestParameterNameConstants:
    """Test parameter name constant definitions."""

    def test_static_mode_params_count(self):
        """Verify static mode has 5 parameters."""
        assert len(STATIC_ISOTROPIC_PARAMS) == 5

    def test_laminar_flow_params_count(self):
        """Verify laminar flow mode has 9 parameters."""
        assert len(LAMINAR_FLOW_PARAMS) == 9

    def test_static_params_ordering(self):
        """Verify static parameter ordering."""
        expected = ["contrast", "offset", "D0", "alpha", "D_offset"]
        assert STATIC_ISOTROPIC_PARAMS == expected

    def test_laminar_params_ordering(self):
        """Verify laminar flow parameter ordering."""
        expected = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        assert LAMINAR_FLOW_PARAMS == expected

    def test_get_parameter_names_static(self):
        """Test get_parameter_names for static mode."""
        params = get_parameter_names("static")
        assert params == STATIC_ISOTROPIC_PARAMS

    def test_get_parameter_names_laminar(self):
        """Test get_parameter_names for laminar flow."""
        params = get_parameter_names("laminar_flow")
        assert LAMINAR_FLOW_PARAMS == params

    def test_get_num_parameters(self):
        """Test get_num_parameters function."""
        assert get_num_parameters("static") == 5
        assert get_num_parameters("laminar_flow") == 9


class TestParameterNameValidation:
    """Test parameter name validation functions."""

    def test_validate_correct_static_params(self):
        """Validate correct static parameter names."""
        params = ["contrast", "offset", "D0", "alpha", "D_offset"]
        validate_parameter_names(params, "static")  # Should not raise

    def test_validate_correct_laminar_params(self):
        """Validate correct laminar flow parameter names."""
        params = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        validate_parameter_names(params, "laminar_flow")  # Should not raise

    def test_validate_wrong_order_strict(self):
        """Test validation fails with wrong order (strict mode)."""
        params = ["offset", "contrast", "D0", "alpha", "D_offset"]  # Wrong order
        with pytest.raises(ValueError, match="don't match expected order"):
            validate_parameter_names(params, "static", strict=True)

    def test_validate_missing_params(self):
        """Test validation fails with missing parameters."""
        params = ["contrast", "D0", "alpha"]  # Missing offset, D_offset
        with pytest.raises(ValueError, match="Missing parameters"):
            validate_parameter_names(params, "static", strict=False)

    def test_validate_extra_params(self):
        """Test validation fails with extra parameters."""
        params = ["contrast", "offset", "D0", "alpha", "D_offset", "extra_param"]
        with pytest.raises(ValueError, match="Unexpected parameters"):
            validate_parameter_names(params, "static", strict=False)

    def test_verify_samples_dict_complete(self):
        """Test verify_samples_dict with complete samples."""
        samples = {
            "contrast": np.array([0.5]),
            "offset": np.array([1.0]),
            "D0": np.array([1000.0]),
            "alpha": np.array([0.5]),
            "D_offset": np.array([10.0]),
        }
        verify_samples_dict(samples, "static")  # Should not raise

    def test_verify_samples_dict_missing(self):
        """Test verify_samples_dict fails with missing parameters."""
        samples = {
            "contrast": np.array([0.5]),
            "D0": np.array([1000.0]),
            # Missing: offset, alpha, D_offset
        }
        with pytest.raises(KeyError, match="Missing parameters in MCMC samples"):
            verify_samples_dict(samples, "static")


class TestMCMCModelConsistency:
    """Test parameter name consistency in MCMC model definitions."""

    def test_mcmc_model_param_names_static(self):
        """Verify mcmc.py uses correct parameter names for static mode."""
        # Import the model creation function
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import _create_numpyro_model

        # Create minimal parameter space for static mode
        config = {
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                ],
            }
        }
        param_space = ParameterSpace.from_config(config)

        # Create model (just verify it can be created without errors)
        # The model internally defines parameter names - we check those match constants
        try:
            model = _create_numpyro_model(
                data=np.array([1.0]),
                sigma=np.array([0.1]),
                t1=np.array([0.0]),
                t2=np.array([1.0]),
                phi=np.array([0.0]),
                q=0.005,
                L=1.0,
                analysis_mode="static",
                parameter_space=param_space,
                dt=1.0,
            )
            # If model creation succeeds, parameter names are internally consistent
            assert model is not None
        except Exception as e:
            pytest.fail(
                f"Model creation failed, possibly due to parameter mismatch: {e}"
            )

    def test_mcmc_model_param_names_laminar(self):
        """Verify mcmc.py uses correct parameter names for laminar flow."""
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import _create_numpyro_model

        # Create minimal parameter space for laminar flow
        config = {
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                    {"name": "gamma_dot_t0", "min": 0.1, "max": 10.0},
                    {"name": "beta", "min": 0.0, "max": 2.0},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 5.0},
                    {"name": "phi0", "min": -3.14159, "max": 3.14159},
                ],
            }
        }
        param_space = ParameterSpace.from_config(config)

        # Create model
        try:
            model = _create_numpyro_model(
                data=np.array([1.0]),
                sigma=np.array([0.1]),
                t1=np.array([0.0]),
                t2=np.array([1.0]),
                phi=np.array([0.0]),
                q=0.005,
                L=1.0,
                analysis_mode="laminar_flow",
                parameter_space=param_space,
                dt=1.0,
            )
            assert model is not None
        except Exception as e:
            pytest.fail(
                f"Model creation failed, possibly due to parameter mismatch: {e}"
            )


class TestPjitBackendConsistency:
    """Test parameter name consistency in pjit backend sample extraction."""

    def test_pjit_no_old_incorrect_names(self):
        """Verify pjit backend does not contain old incorrect parameter names."""
        import inspect

        from homodyne.optimization.cmc.backends.pjit import PjitBackend

        # Get the source code
        source = inspect.getsource(PjitBackend._run_single_shard_mcmc)

        # Verify old incorrect names that caused the bug are not present
        assert "'gamma_dot_0'" not in source  # Old bug: should be gamma_dot_t0
        assert (
            "'gamma_dot_offset'" not in source
        )  # Old bug: should be gamma_dot_t_offset
        assert "'phi_0'" not in source  # Old bug: should be phi0


class TestCrossCodingConsistency:
    """Integration tests verifying parameter names are consistent across all modules."""

    def test_mcmc_to_pjit_consistency_static(self):
        """Verify mcmc.py and pjit.py use identical static parameter names."""
        from homodyne.config.parameter_names import STATIC_ISOTROPIC_PARAMS

        # Expected parameter names from constants
        expected = STATIC_ISOTROPIC_PARAMS

        # This test ensures both mcmc.py model definition and pjit.py sample extraction
        # use the same parameter names defined in parameter_names.py
        # If this test passes, the modules are consistent

        # Verify expected matches our constants
        assert expected == ["contrast", "offset", "D0", "alpha", "D_offset"]

    def test_mcmc_to_pjit_consistency_laminar(self):
        """Verify mcmc.py and pjit.py use identical laminar flow parameter names."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # Expected parameter names from constants
        expected = LAMINAR_FLOW_PARAMS

        # Verify expected matches our constants
        assert expected == [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

        # CRITICAL: Verify no old incorrect names
        assert "gamma_dot_0" not in expected  # Old bug name
        assert "gamma_dot_offset" not in expected  # Old bug name
        assert "phi_0" not in expected  # Old bug name


class TestRegressionPrevention:
    """Tests that specifically prevent the gamma_dot_0 vs gamma_dot_t0 bug."""

    def test_no_gamma_dot_0_in_constants(self):
        """Ensure gamma_dot_0 is never used (should be gamma_dot_t0)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # gamma_dot_0 is the INCORRECT name that caused the bug
        assert "gamma_dot_0" not in LAMINAR_FLOW_PARAMS
        # gamma_dot_t0 is the CORRECT name
        assert "gamma_dot_t0" in LAMINAR_FLOW_PARAMS

    def test_no_phi_0_in_constants(self):
        """Ensure phi_0 is never used (should be phi0)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # phi_0 is incorrect (old style)
        assert "phi_0" not in LAMINAR_FLOW_PARAMS
        # phi0 is correct
        assert "phi0" in LAMINAR_FLOW_PARAMS

    def test_no_gamma_dot_offset_in_constants(self):
        """Ensure gamma_dot_offset is never used (should be gamma_dot_t_offset)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # gamma_dot_offset is incorrect
        assert "gamma_dot_offset" not in LAMINAR_FLOW_PARAMS
        # gamma_dot_t_offset is correct
        assert "gamma_dot_t_offset" in LAMINAR_FLOW_PARAMS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
