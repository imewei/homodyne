"""
Unit Tests for Advanced ParameterManager Features (Phase 4.3)
============================================================

Tests for fixed parameters, optimizable parameters, and TypedDict integration.
"""

import pytest

from homodyne.config.parameter_manager import ParameterManager
from homodyne.config.types import (
    LAMINAR_FLOW_PARAM_NAMES,
    PARAMETER_NAME_MAPPING,
    STATIC_PARAM_NAMES,
    BoundDict,
)


class TestFixedParameters:
    """Test fixed parameters functionality."""

    def test_get_fixed_parameters_none(self):
        """Test getting fixed parameters when none specified."""
        pm = ParameterManager(None, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        assert fixed == {}

    def test_get_fixed_parameters_from_config(self):
        """Test getting fixed parameters from configuration."""
        config = {
            "initial_parameters": {
                "fixed_parameters": {"D_offset": 10.0, "contrast": 0.5}
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        assert fixed == {"D_offset": 10.0, "contrast": 0.5}

    def test_get_fixed_parameters_invalid_type(self):
        """Test handling of invalid fixed_parameters type."""
        config = {"initial_parameters": {"fixed_parameters": ["not", "a", "dict"]}}
        pm = ParameterManager(config, "laminar_flow")
        fixed = pm.get_fixed_parameters()

        # Should return empty dict and log warning
        assert fixed == {}


class TestParameterActiveStatus:
    """Test is_parameter_active() functionality."""

    def test_parameter_active_no_config(self):
        """Test parameter active status without configuration."""
        pm = ParameterManager(None, "laminar_flow")

        # All default active parameters should be active
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("alpha") is True
        assert pm.is_parameter_active("gamma_dot_t0") is True

    def test_parameter_active_with_fixed(self):
        """Test parameter active status with fixed parameters."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "fixed_parameters": {"D_offset": 10.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # D0 and alpha are active
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("alpha") is True

        # D_offset is fixed (not active)
        assert pm.is_parameter_active("D_offset") is False

    def test_parameter_active_with_name_mapping(self):
        """Test parameter active status with name mapping."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "gamma_dot_0"],  # Config name
                "fixed_parameters": {"D0": 1000.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # D0 is fixed
        assert pm.is_parameter_active("D0") is False

        # gamma_dot_t0 is active (mapped from gamma_dot_0)
        assert pm.is_parameter_active("gamma_dot_t0") is True


class TestOptimizableParameters:
    """Test get_optimizable_parameters() functionality."""

    def test_optimizable_params_no_fixed(self):
        """Test optimizable parameters when nothing is fixed."""
        pm = ParameterManager(None, "static")
        optimizable = pm.get_optimizable_parameters()

        # All active parameters should be optimizable
        assert optimizable == ["D0", "alpha", "D_offset"]

    def test_optimizable_params_with_fixed(self):
        """Test optimizable parameters with some fixed."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_t0"],
                "fixed_parameters": {"D_offset": 10.0, "gamma_dot_t0": 1e-4},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        optimizable = pm.get_optimizable_parameters()

        # Only D0 and alpha should be optimizable
        assert optimizable == ["D0", "alpha"]
        assert len(optimizable) == 2

    def test_optimizable_params_all_fixed(self):
        """Test optimizable parameters when all are fixed."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "fixed_parameters": {"D0": 1000.0, "alpha": 0.5},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        optimizable = pm.get_optimizable_parameters()

        # No parameters should be optimizable
        assert optimizable == []


class TestTypeConstants:
    """Test type constants from types.py."""

    def test_static_param_names(self):
        """Test STATIC_PARAM_NAMES constant."""
        assert STATIC_PARAM_NAMES == ["D0", "alpha", "D_offset"]
        assert len(STATIC_PARAM_NAMES) == 3

    def test_laminar_flow_param_names(self):
        """Test LAMINAR_FLOW_PARAM_NAMES constant."""
        expected = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        assert LAMINAR_FLOW_PARAM_NAMES == expected
        assert len(LAMINAR_FLOW_PARAM_NAMES) == 7

    def test_parameter_name_mapping(self):
        """Test PARAMETER_NAME_MAPPING constant."""
        assert PARAMETER_NAME_MAPPING["gamma_dot_0"] == "gamma_dot_t0"
        assert PARAMETER_NAME_MAPPING["gamma_dot_offset"] == "gamma_dot_t_offset"
        assert PARAMETER_NAME_MAPPING["phi_0"] == "phi0"
        assert len(PARAMETER_NAME_MAPPING) == 3


class TestBoundDictType:
    """Test BoundDict TypedDict."""

    def test_bound_dict_structure(self):
        """Test that BoundDict has expected structure."""
        # This tests that the TypedDict is properly defined
        bound: BoundDict = {
            "name": "D0",
            "min": 1.0,
            "max": 1e6,
            "type": "Normal",
        }

        assert bound["name"] == "D0"
        assert bound["min"] == 1.0
        assert bound["max"] == 1e6
        assert bound["type"] == "Normal"


class TestReprWithFixedParams:
    """Test string representation with fixed parameters."""

    def test_repr_with_fixed_params(self):
        """Test repr includes fixed parameter count."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "fixed_parameters": {"D_offset": 10.0},
            }
        }
        pm = ParameterManager(config, "laminar_flow")
        repr_str = repr(pm)

        assert "ParameterManager" in repr_str
        assert "active_params=3" in repr_str
        assert "fixed_params=1" in repr_str
        assert "optimizable=2" in repr_str

    def test_repr_no_fixed_params(self):
        """Test repr with no fixed parameters."""
        pm = ParameterManager(None, "static")
        repr_str = repr(pm)

        assert "active_params=3" in repr_str
        assert "fixed_params=0" in repr_str
        assert "optimizable=3" in repr_str


class TestIntegrationScenarios:
    """Integration tests for advanced features."""

    def test_subset_optimization_workflow(self):
        """Test workflow with subset of parameters and fixed values."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                ],
                "fixed_parameters": {"beta": 0.5, "D_offset": 10.0},
            },
        }
        pm = ParameterManager(config, "laminar_flow")

        # Check active parameters
        active = pm.get_active_parameters()
        assert len(active) == 5

        # Check optimizable (active - fixed)
        optimizable = pm.get_optimizable_parameters()
        assert optimizable == ["D0", "alpha", "gamma_dot_t0"]
        assert len(optimizable) == 3

        # Check individual status
        assert pm.is_parameter_active("D0") is True
        assert pm.is_parameter_active("beta") is False
        assert pm.is_parameter_active("D_offset") is False

    def test_name_mapping_with_fixed_params(self):
        """Test that name mapping works with fixed parameters."""
        config = {
            "initial_parameters": {
                "parameter_names": ["D0", "gamma_dot_0", "phi_0"],
                "fixed_parameters": {"phi_0": 0.1},  # Config name
            }
        }
        pm = ParameterManager(config, "laminar_flow")

        # Active should use canonical names
        active = pm.get_active_parameters()
        assert "gamma_dot_t0" in active
        assert "phi0" in active

        # Fixed should work with config name
        fixed = pm.get_fixed_parameters()
        assert "phi_0" in fixed

        # is_parameter_active should handle canonical name
        assert pm.is_parameter_active("phi0") is False
        assert pm.is_parameter_active("gamma_dot_t0") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
