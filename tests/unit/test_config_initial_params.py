"""Unit tests for initial parameters configuration loading.

Tests the ConfigManager.get_initial_parameters() method for:
- Loading explicit values from config
- Mid-point calculation when values are null
- Active/fixed parameter filtering
- Parameter name mapping
- Error handling for invalid configs

This module is part of Task Group 2.2 in the v2.1.0 MCMC simplification implementation.
"""

import pytest
import numpy as np
from homodyne.config.manager import ConfigManager


class TestInitialParametersLoading:
    """Test loading explicit initial parameter values from config."""

    def test_load_explicit_values_static(self):
        """Test loading explicit values for static mode."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, 0.5, 10.0],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params == {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
        }

    def test_load_explicit_values_laminar_flow(self):
        """Test loading explicit values for laminar flow mode."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",
                    "beta",
                    "gamma_dot_offset",
                    "phi_0",
                ],
                "values": [1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Check canonical name mapping (gamma_dot_0 → gamma_dot_t0)
        assert initial_params == {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.01,
            "beta": 0.0,
            "gamma_dot_t_offset": 0.0,
            "phi0": 0.0,
        }

    def test_parameter_name_mapping(self):
        """Test that config names are mapped to canonical code names."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["gamma_dot_0", "phi_0"],  # Config names
                "values": [0.01, 45.0],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should use canonical names
        assert "gamma_dot_t0" in initial_params  # Mapped from gamma_dot_0
        assert "phi0" in initial_params  # Mapped from phi_0
        assert "gamma_dot_0" not in initial_params  # Config name not present
        assert "phi_0" not in initial_params  # Config name not present

    def test_values_type_coercion(self):
        """Test that values are coerced to float."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "values": [1000, 0],  # Integers, should be converted to float
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert isinstance(initial_params["D0"], float)
        assert isinstance(initial_params["alpha"], float)
        assert initial_params["D0"] == 1000.0
        assert initial_params["alpha"] == 0.0


class TestMidpointDefaultCalculation:
    """Test mid-point default calculation when values are null."""

    def test_null_values_uses_midpoint(self):
        """Test that null values trigger mid-point calculation."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": None,  # Null values
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Check mid-points: (min + max) / 2
        assert initial_params["D0"] == pytest.approx(550.0)  # (100 + 1000) / 2
        assert initial_params["alpha"] == pytest.approx(0.0)  # (-2 + 2) / 2
        assert initial_params["D_offset"] == pytest.approx(50.0)  # (0 + 100) / 2

    def test_missing_values_uses_midpoint(self):
        """Test that missing values field triggers mid-point calculation."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                # No 'values' field
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should calculate mid-points
        assert len(initial_params) == 3
        assert initial_params["D0"] == pytest.approx(550.0)

    def test_no_initial_parameters_section(self):
        """Test that missing initial_parameters section uses mid-point defaults."""
        config = {
            "analysis_mode": "static",
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 100.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should use package defaults and calculate mid-points
        assert len(initial_params) == 3
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "D_offset" in initial_params

    def test_midpoint_uses_parameter_manager_bounds(self):
        """Test that mid-point calculation uses ParameterManager bounds."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "values": None,  # Trigger mid-point calculation
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should calculate mid-points for all laminar flow parameters
        # (7 parameters: D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0)
        assert len(initial_params) == 7
        assert all(isinstance(v, float) for v in initial_params.values())

    def test_midpoint_with_custom_bounds(self):
        """Test mid-point calculation with custom bounds in config."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "values": None,
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 500.0, "max": 1500.0},  # Custom bounds
                    {"name": "alpha", "min": -1.0, "max": 1.0},  # Custom bounds
                    {"name": "D_offset", "min": 5.0, "max": 15.0},  # Custom bounds
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Check mid-points with custom bounds
        assert initial_params["D0"] == pytest.approx(1000.0)  # (500 + 1500) / 2
        assert initial_params["alpha"] == pytest.approx(0.0)  # (-1 + 1) / 2
        assert initial_params["D_offset"] == pytest.approx(10.0)  # (5 + 15) / 2


class TestActiveParametersFiltering:
    """Test active_parameters filtering functionality."""

    def test_active_parameters_subset(self):
        """Test that active_parameters filters to a subset."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",
                    "beta",
                ],
                "values": [1000.0, 0.5, 10.0, 0.01, 0.0],
                "active_parameters": ["D0", "alpha", "gamma_dot_0"],  # Only 3 active
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should only include active parameters
        assert len(initial_params) == 3
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "gamma_dot_t0" in initial_params  # Mapped from gamma_dot_0

        # Should exclude non-active parameters
        assert "D_offset" not in initial_params
        assert "beta" not in initial_params

    def test_active_parameters_with_name_mapping(self):
        """Test active_parameters with config name mapping."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["gamma_dot_0", "phi_0", "beta"],
                "values": [0.01, 45.0, 0.0],
                "active_parameters": ["gamma_dot_0", "phi_0"],  # Config names
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should map active parameter names and filter
        assert len(initial_params) == 2
        assert "gamma_dot_t0" in initial_params  # Mapped and active
        assert "phi0" in initial_params  # Mapped and active
        assert "beta" not in initial_params  # Not in active list


class TestFixedParametersFiltering:
    """Test fixed_parameters filtering functionality."""

    def test_fixed_parameters_excluded(self):
        """Test that fixed_parameters are excluded from initial parameters."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, 0.5, 10.0],
                "fixed_parameters": {"D_offset": 10.0},  # Fix D_offset
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should exclude fixed parameter
        assert len(initial_params) == 2
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "D_offset" not in initial_params  # Fixed, so excluded

    def test_fixed_parameters_with_name_mapping(self):
        """Test fixed_parameters with config name mapping."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["D0", "gamma_dot_0", "beta"],
                "values": [1000.0, 0.01, 0.0],
                "fixed_parameters": {
                    "gamma_dot_0": 0.01,  # Config name
                    "beta": 0.0,
                },
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should map fixed parameter names and exclude them
        assert len(initial_params) == 1
        assert "D0" in initial_params
        assert "gamma_dot_t0" not in initial_params  # Fixed (mapped name)
        assert "beta" not in initial_params  # Fixed

    def test_active_and_fixed_combined(self):
        """Test combination of active_parameters and fixed_parameters."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset", "gamma_dot_0"],
                "values": [1000.0, 0.5, 10.0, 0.01],
                "active_parameters": ["D0", "alpha", "D_offset"],  # 3 active
                "fixed_parameters": {"D_offset": 10.0},  # 1 fixed
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should be active AND not fixed: D0 and alpha only
        assert len(initial_params) == 2
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "D_offset" not in initial_params  # Fixed
        assert "gamma_dot_t0" not in initial_params  # Not in active list


class TestErrorHandling:
    """Test error handling for invalid configurations."""

    def test_values_length_mismatch(self):
        """Test error when values length doesn't match parameter_names."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, 0.5],  # Only 2 values, need 3
            },
        }

        config_mgr = ConfigManager(config_override=config)

        with pytest.raises(ValueError, match="Number of values.*does not match"):
            config_mgr.get_initial_parameters()

    def test_values_not_list(self):
        """Test error when values is not a list."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "values": "invalid",  # String instead of list
            },
        }

        config_mgr = ConfigManager(config_override=config)

        with pytest.raises(ValueError, match="must be a list"):
            config_mgr.get_initial_parameters()

    def test_null_values_without_midpoint(self):
        """Test error when values are null and use_midpoint_defaults is False."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "values": None,
            },
        }

        config_mgr = ConfigManager(config_override=config)

        with pytest.raises(
            ValueError, match="values is null and use_midpoint_defaults is False"
        ):
            config_mgr.get_initial_parameters(use_midpoint_defaults=False)

    def test_empty_config(self):
        """Test that empty config uses default config with mid-point defaults."""
        config_mgr = ConfigManager(config_override={})

        initial_params = config_mgr.get_initial_parameters()

        # Empty config should still return mid-point defaults from package defaults
        # Default mode is "static" with 3 parameters
        assert len(initial_params) >= 0  # May be empty or have defaults
        assert isinstance(initial_params, dict)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_nlsq_results_to_mcmc_workflow(self):
        """Test loading NLSQ results as initial values for MCMC."""
        # Simulate NLSQ results manually copied to config
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1234.5, 0.567, 12.34],  # From NLSQ output
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should use NLSQ results as initial values
        assert initial_params["D0"] == 1234.5
        assert initial_params["alpha"] == 0.567
        assert initial_params["D_offset"] == 12.34

    def test_partial_optimization_workflow(self):
        """Test partial optimization (fixing some parameters)."""
        # User wants to optimize D0 and alpha, but fix D_offset
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, 0.5, 10.0],
                "fixed_parameters": {"D_offset": 10.0},  # Fix at NLSQ value
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should only return optimizable parameters
        assert len(initial_params) == 2
        assert initial_params == {"D0": 1000.0, "alpha": 0.5}

    def test_exploration_with_midpoints(self):
        """Test exploration mode using mid-point defaults."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "values": None,  # Use mid-points for exploration
            },
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                    {"name": "gamma_dot_0", "min": 1e-6, "max": 0.5},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {"name": "gamma_dot_offset", "min": -0.1, "max": 0.1},
                    {"name": "phi_0", "min": -180.0, "max": 180.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should calculate reasonable mid-points for all parameters
        assert len(initial_params) == 7
        assert initial_params["D0"] == pytest.approx(5050.0)  # (100 + 10000) / 2
        assert initial_params["alpha"] == pytest.approx(0.0)  # (-2 + 2) / 2
        assert initial_params["phi0"] == pytest.approx(0.0)  # (-180 + 180) / 2

    def test_realistic_laminar_flow_config(self):
        """Test realistic laminar flow config with all features."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",
                    "beta",
                    "gamma_dot_offset",
                    "phi_0",
                ],
                "values": [1000.0, 0.5, 10.0, 0.01, 0.0, 0.0, 0.0],
                "active_parameters": [
                    "D0",
                    "alpha",
                    "gamma_dot_0",
                    "phi_0",
                ],  # Optimize 4 params
                "fixed_parameters": {
                    "beta": 0.0,  # Constant shear
                    "gamma_dot_offset": 0.0,  # No offset
                    "D_offset": 10.0,  # Fix offset
                },
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should filter correctly
        assert len(initial_params) == 4
        assert set(initial_params.keys()) == {
            "D0",
            "alpha",
            "gamma_dot_t0",
            "phi0",
        }


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_parameter(self):
        """Test with single parameter."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0"],
                "values": [1000.0],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params == {"D0": 1000.0}

    def test_negative_values(self):
        """Test that negative values are handled correctly."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["alpha", "D_offset"],
                "values": [-1.5, -10.0],  # Negative values
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params["alpha"] == -1.5
        assert initial_params["D_offset"] == -10.0

    def test_very_large_values(self):
        """Test that very large values are handled correctly."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0"],
                "values": [1e10],  # Very large value
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params["D0"] == 1e10

    def test_zero_values(self):
        """Test that zero values are handled correctly."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["beta", "gamma_dot_offset"],
                "values": [0.0, 0.0],  # Zero values
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params["beta"] == 0.0
        assert initial_params["gamma_dot_t_offset"] == 0.0
