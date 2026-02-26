"""
Unit Tests for homodyne.core.physics Module
==========================================

Tests for parameter validation and ValidationResult dataclass.
"""

import numpy as np
import pytest

from homodyne.core.physics import (
    ValidationResult,
    validate_parameters,
    validate_parameters_detailed,
)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_valid_result(self):
        """Test valid ValidationResult."""
        result = ValidationResult(
            valid=True, parameters_checked=5, message="All parameters valid"
        )

        assert result.valid is True
        assert result.parameters_checked == 5
        assert result.message == "All parameters valid"
        assert len(result.violations) == 0
        assert "OK" in str(result)

    def test_invalid_result(self):
        """Test invalid ValidationResult."""
        result = ValidationResult(
            valid=False,
            violations=["D0 out of bounds", "alpha out of bounds"],
            parameters_checked=5,
            message="Validation failed",
        )

        assert result.valid is False
        assert result.parameters_checked == 5
        assert len(result.violations) == 2
        assert "FAIL" in str(result)
        assert "D0 out of bounds" in str(result)
        assert "alpha out of bounds" in str(result)

    def test_str_representation(self):
        """Test string representation formatting."""
        valid_result = ValidationResult(valid=True, message="Success")
        invalid_result = ValidationResult(
            valid=False, violations=["Error 1", "Error 2"], message="Failed"
        )

        assert str(valid_result) == "OK Success"
        assert "FAIL Failed" in str(invalid_result)
        assert "Error 1" in str(invalid_result)
        assert "Error 2" in str(invalid_result)


class TestValidateParametersDetailed:
    """Test validate_parameters_detailed function."""

    def test_all_valid_parameters(self):
        """Test validation with all parameters within bounds."""
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)

        assert result.valid is True
        assert result.parameters_checked == 5
        assert len(result.violations) == 0
        assert "valid" in result.message.lower()

    def test_parameter_above_bounds(self):
        """Test validation with parameter above upper bound."""
        params = np.array([1.5, 1.0, 1000.0, 0.5, 10.0])  # contrast > 1.0
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)

        assert result.valid is False
        assert result.parameters_checked == 5
        assert len(result.violations) == 1
        assert "contrast" in result.violations[0]
        assert "above bounds" in result.violations[0]

    def test_parameter_below_bounds(self):
        """Test validation with parameter below lower bound."""
        params = np.array([0.5, 1.0, 0.5, 0.5, 10.0])  # D0 < 1.0
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)

        assert result.valid is False
        assert result.parameters_checked == 5
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "below bounds" in result.violations[0]

    def test_multiple_violations(self):
        """Test validation with multiple parameters out of bounds."""
        params = np.array([1.5, -0.5, 0.5, 1.5, 150.0])
        # contrast > 1.0, offset < 0.0, D0 < 1.0, alpha > 1.0, D_offset > 100.0
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)

        assert result.valid is False
        assert result.parameters_checked == 5
        assert len(result.violations) == 5
        assert any("contrast" in v for v in result.violations)
        assert any("offset" in v for v in result.violations)
        assert any("D0" in v for v in result.violations)
        assert any("alpha" in v for v in result.violations)
        assert any("D_offset" in v for v in result.violations)

    def test_without_param_names(self):
        """Test validation without parameter names."""
        params = np.array([1.5, 1.0, 1000.0, 0.5, 10.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        result = validate_parameters_detailed(params, bounds)

        assert result.valid is False
        assert result.parameters_checked == 5
        assert len(result.violations) == 1
        assert "param_0" in result.violations[0]  # Should use index format "param_i"

    def test_violation_magnitude(self):
        """Test that violations include magnitude information."""
        params = np.array([1.5e6, 1.0, 1000.0, 0.5, 10.0])  # D0 >> max
        bounds = [(1.0, 1e6), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]
        param_names = ["D0", "offset", "D0_2", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)

        assert result.valid is False
        assert len(result.violations) == 1
        violation = result.violations[0]
        assert "D0" in violation
        assert "by" in violation.lower()  # Should include magnitude

    def test_boundary_tolerance(self):
        """Test tolerance at boundary values."""
        # Test with value just above upper bound but within tolerance
        params = np.array([1.0 + 1e-11, 1.0, 1000.0, 0.5, 10.0])  # Just above bound
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        # Default tolerance 1e-10 should pass (1e-11 < 1e-10)
        result = validate_parameters_detailed(params, bounds, tolerance=1e-10)
        assert result.valid is True

        # Stricter tolerance should fail
        result = validate_parameters_detailed(params, bounds, tolerance=1e-12)
        assert result.valid is False

    def test_parameter_count_mismatch(self):
        """Test validation with mismatched parameter and bounds counts."""
        params = np.array([0.5, 1.0, 1000.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        result = validate_parameters_detailed(params, bounds)

        # Should return invalid result with mismatch message
        assert result.valid is False
        assert result.parameters_checked == 0
        assert len(result.violations) == 1
        assert "mismatch" in result.violations[0].lower()
        assert "3" in result.violations[0]  # param count
        assert "5" in result.violations[0]  # bounds count


class TestValidateParametersLegacy:
    """Test legacy validate_parameters function for backward compatibility."""

    def test_valid_parameters(self):
        """Test legacy function with valid parameters."""
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        result = validate_parameters(params, bounds)
        assert result is True

    def test_invalid_parameters(self):
        """Test legacy function with invalid parameters."""
        params = np.array([1.5, 1.0, 1000.0, 0.5, 10.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        result = validate_parameters(params, bounds)
        assert result is False

    def test_backward_compatibility(self):
        """Test that legacy function works with existing code patterns."""
        # This test ensures the legacy API hasn't changed
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        bounds = [(0.0, 1.0), (0.0, 2.0), (1.0, 1e6), (0.0, 1.0), (0.0, 100.0)]

        # Should work with positional args only
        result = validate_parameters(params, bounds)
        assert isinstance(result, bool)

        # Should work with array-like inputs
        params_list = [0.5, 1.0, 1000.0, 0.5, 10.0]
        result = validate_parameters(params_list, bounds)
        assert isinstance(result, bool)


class TestValidationIntegration:
    """Integration tests for parameter validation."""

    def test_laminar_flow_parameters(self):
        """Test validation with realistic laminar flow parameters."""
        # Realistic parameter values from user config
        params = np.array(
            [
                0.5,  # contrast
                1.0,  # offset
                13930.8,  # D0
                -0.479,  # alpha
                49.298,  # D_offset
                9.65e-4,  # gamma_dot_t0
                0.5018,  # beta
                3.13e-5,  # gamma_dot_t_offset
                8.99e-2,  # phi0
            ]
        )
        bounds = [
            (0.0, 1.0),  # contrast
            (0.0, 10.0),  # offset
            (1.0, 1e6),  # D0
            (-2.0, 2.0),  # alpha
            (0.0, 1e6),  # D_offset
            (1e-10, 1.0),  # gamma_dot_t0
            (-2.0, 2.0),  # beta
            (1e-10, 1.0),  # gamma_dot_t_offset
            (-np.pi, np.pi),  # phi0
        ]
        param_names = [
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

        result = validate_parameters_detailed(params, bounds, param_names)
        assert result.valid is True
        assert result.parameters_checked == 9

    def test_static_mode_parameters(self):
        """Test validation with realistic static mode parameters."""
        params = np.array(
            [
                0.5,  # contrast
                1.0,  # offset
                1000.0,  # D0
                0.5,  # alpha
                10.0,  # D_offset
            ]
        )
        bounds = [
            (0.0, 1.0),  # contrast
            (0.0, 10.0),  # offset
            (1.0, 1e6),  # D0
            (-2.0, 2.0),  # alpha
            (0.0, 1e6),  # D_offset
        ]
        param_names = ["contrast", "offset", "D0", "alpha", "D_offset"]

        result = validate_parameters_detailed(params, bounds, param_names)
        assert result.valid is True
        assert result.parameters_checked == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
