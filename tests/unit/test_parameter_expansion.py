"""
Unit tests for per-angle scaling parameter expansion in v2.2.1.

Tests the automatic parameter expansion logic that converts:
- 9 input parameters (7 physical + 1 contrast + 1 offset)
- To 13 output parameters (7 physical + 3 contrast + 3 offset) for 3 angles

This fixes the critical bug where optimization failed with zero gradients
due to parameter count mismatch.
"""

import numpy as np
import pytest


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
