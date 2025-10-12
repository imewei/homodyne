"""
Unit Tests for ParameterManager Physics Validation (Phase 4.4)
===============================================================

Tests for physics-based parameter constraint validation beyond simple bounds.
"""

import numpy as np
import pytest

from homodyne.config.parameter_manager import ConstraintSeverity, ParameterManager


class TestPhysicsValidationBasics:
    """Test basic physics validation functionality."""

    def test_validate_all_valid_parameters(self):
        """Test validation with physically reasonable parameters."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.5,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 0.5,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        # Should pass all checks (only INFO messages possible)
        assert len(result.violations) == 0
        assert result.parameters_checked == len(params)
        assert "successfully" in result.message.lower()

    def test_validate_empty_parameters(self):
        """Test validation with empty parameter dict."""
        pm = ParameterManager(None, "laminar_flow")
        params = {}

        result = pm.validate_physical_constraints(params)

        assert result.valid is True
        assert len(result.violations) == 0
        assert result.parameters_checked == 0


class TestErrorLevelViolations:
    """Test ERROR-level physics violations (physically impossible)."""

    def test_negative_diffusion_coefficient(self):
        """Test ERROR for negative D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "physically impossible" in result.violations[0].lower()
        assert "[error]" in result.violations[0]

    def test_zero_diffusion_coefficient(self):
        """Test ERROR for D0 = 0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 0.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "non-positive" in result.violations[0].lower()

    def test_negative_shear_rate(self):
        """Test ERROR for negative gamma_dot_t0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": -0.001}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "gamma_dot_t0" in result.violations[0]
        assert "negative shear rate" in result.violations[0].lower()
        assert "physically impossible" in result.violations[0].lower()

    def test_invalid_contrast_too_high(self):
        """Test ERROR for contrast > 1."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 1.5}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "contrast" in result.violations[0]
        assert "outside physical range" in result.violations[0].lower()

    def test_invalid_contrast_zero(self):
        """Test ERROR for contrast = 0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 0.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "contrast" in result.violations[0]

    def test_negative_offset(self):
        """Test ERROR for negative offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"offset": -0.5}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert "offset" in result.violations[0]
        assert "non-positive baseline" in result.violations[0].lower()

    def test_multiple_errors(self):
        """Test multiple ERROR violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0, "gamma_dot_t0": -0.001, "contrast": 2.0}

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 3
        # Check all three violations are present
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "gamma_dot_t0" in violation_str
        assert "contrast" in violation_str


class TestWarningLevelViolations:
    """Test WARNING-level physics violations (unusual but possible)."""

    def test_very_large_diffusion_coefficient(self):
        """Test WARNING for D0 > 1e7."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 5e7}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "extremely large" in result.violations[0].lower()
        assert "[warning]" in result.violations[0]

    def test_strongly_subdiffusive_alpha(self):
        """Test WARNING for alpha < -1.5."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": -1.8}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "alpha" in result.violations[0]
        assert "subdiffusive" in result.violations[0].lower()

    def test_strongly_superdiffusive_alpha(self):
        """Test WARNING for alpha > 1.0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": 1.5}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "alpha" in result.violations[0]
        assert "superdiffusive" in result.violations[0].lower()
        assert "ballistic" in result.violations[0].lower()

    def test_negative_d_offset(self):
        """Test WARNING for negative D_offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D_offset": -10.0}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "D_offset" in result.violations[0]
        assert "negative offset" in result.violations[0].lower()

    def test_very_high_shear_rate(self):
        """Test WARNING for gamma_dot_t0 > 1.0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": 5.0}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "gamma_dot_t0" in result.violations[0]
        assert "very high shear rate" in result.violations[0].lower()

    def test_beta_out_of_range(self):
        """Test WARNING for beta outside [-2, 2]."""
        pm = ParameterManager(None, "laminar_flow")
        params_low = {"beta": -2.5}
        params_high = {"beta": 2.5}

        result_low = pm.validate_physical_constraints(
            params_low, severity_level="warning"
        )
        result_high = pm.validate_physical_constraints(
            params_high, severity_level="warning"
        )

        assert result_low.valid is False
        assert result_high.valid is False
        assert "beta" in result_low.violations[0]
        assert "beta" in result_high.violations[0]

    def test_negative_gamma_dot_offset(self):
        """Test WARNING for negative gamma_dot_t_offset."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t_offset": -0.0001}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "gamma_dot_t_offset" in result.violations[0]

    def test_very_low_contrast(self):
        """Test WARNING for contrast < 0.1."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"contrast": 0.05}

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert "contrast" in result.violations[0]
        assert "very low contrast" in result.violations[0].lower()


class TestInfoLevelViolations:
    """Test INFO-level physics notifications (noteworthy but acceptable)."""

    def test_near_normal_diffusion(self):
        """Test INFO for alpha near zero (normal diffusion)."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"alpha": 0.05}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "alpha" in result.violations[0]
        assert "near-normal diffusion" in result.violations[0].lower()
        assert "[info]" in result.violations[0]

    def test_very_low_shear_rate(self):
        """Test INFO for gamma_dot_t0 < 1e-6."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"gamma_dot_t0": 1e-8}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert "gamma_dot_t0" in result.violations[0]
        assert "very low shear rate" in result.violations[0].lower()
        assert "quasi-static" in result.violations[0].lower()

    def test_angle_outside_pi_range(self):
        """Test INFO for phi0 outside [-π, π]."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"phi0": 4.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert "phi0" in result.violations[0]
        assert "will wrap" in result.violations[0].lower()


class TestCrossParameterConstraints:
    """Test cross-parameter physics constraints."""

    def test_d_offset_dominates_d0(self):
        """Test INFO when D_offset is large compared to D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "D_offset": 600.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D_offset" in result.violations[0]
        assert "60.0%" in result.violations[0]  # 600/1000 = 60%
        assert "overfitting" in result.violations[0].lower()

    def test_d_offset_reasonable_ratio(self):
        """Test no violation when D_offset is reasonable compared to D0."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "D_offset": 100.0}

        result = pm.validate_physical_constraints(params, severity_level="info")

        # Should pass - 100/1000 = 10% < 50%
        assert result.valid is True
        assert len(result.violations) == 0


class TestSeverityFiltering:
    """Test severity level filtering."""

    def test_error_level_filters_warnings(self):
        """Test that error level only shows ERROR violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="error")

        assert result.valid is False
        assert len(result.violations) == 1
        assert "D0" in result.violations[0]
        assert "[error]" in result.violations[0]

    def test_warning_level_shows_errors_and_warnings(self):
        """Test that warning level shows ERROR + WARNING."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        assert len(result.violations) == 2
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "alpha" in violation_str
        assert "gamma_dot_t0" not in violation_str

    def test_info_level_shows_all(self):
        """Test that info level shows all violations."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": -100.0,  # ERROR
            "alpha": 1.5,  # WARNING
            "gamma_dot_t0": 1e-8,  # INFO
        }

        result = pm.validate_physical_constraints(params, severity_level="info")

        assert result.valid is False
        assert len(result.violations) == 3
        violation_str = " ".join(result.violations)
        assert "D0" in violation_str
        assert "alpha" in violation_str
        assert "gamma_dot_t0" in violation_str


class TestValidationMessages:
    """Test validation result messages."""

    def test_success_message(self):
        """Test message for successful validation."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5}

        result = pm.validate_physical_constraints(params)

        assert result.valid is True
        assert "successfully" in result.message.lower()
        assert "2 parameters" in result.message.lower()

    def test_failure_message(self):
        """Test message for failed validation."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": -100.0}

        result = pm.validate_physical_constraints(params)

        assert result.valid is False
        assert "1 issue" in result.message.lower()

    def test_parameters_checked_count(self):
        """Test parameters_checked field."""
        pm = ParameterManager(None, "laminar_flow")
        params = {"D0": 1000.0, "alpha": 0.5, "gamma_dot_t0": 0.001}

        result = pm.validate_physical_constraints(params)

        assert result.parameters_checked == 3


class TestPhysicsValidationIntegration:
    """Integration tests for physics validation."""

    def test_static_mode_parameters(self):
        """Test validation for static mode parameters."""
        pm = ParameterManager(None, "static")
        params = {
            "D0": 1000.0,
            "alpha": -0.5,  # Subdiffusion common in static
            "D_offset": 10.0,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        # Should pass all checks
        assert result.valid is True

    def test_laminar_flow_parameters(self):
        """Test validation for laminar flow parameters."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.001,
            "beta": 0.3,
            "gamma_dot_t_offset": 0.0001,
            "phi0": 0.5,
            "contrast": 0.8,
            "offset": 1.0,
        }

        result = pm.validate_physical_constraints(params)

        assert result.valid is True

    def test_realistic_subdiffusion_scenario(self):
        """Test realistic subdiffusion scenario."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 500.0,
            "alpha": -0.7,  # Subdiffusion
            "D_offset": 5.0,
            "gamma_dot_t0": 0.0001,
            "beta": 0.5,
            "contrast": 0.6,
            "offset": 1.2,
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        # Should pass - subdiffusion is common
        assert result.valid is True

    def test_problematic_parameter_set(self):
        """Test detection of problematic parameter set."""
        pm = ParameterManager(None, "laminar_flow")
        params = {
            "D0": 100.0,
            "alpha": 0.5,
            "D_offset": 90.0,  # 90% of D0 - likely overfitting
            "gamma_dot_t0": 10.0,  # Very high
            "contrast": 0.05,  # Very low
        }

        result = pm.validate_physical_constraints(params, severity_level="warning")

        assert result.valid is False
        # Should have multiple warnings
        assert len(result.violations) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
