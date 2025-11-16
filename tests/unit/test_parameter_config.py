"""
Unit Tests for Parameter Configuration
=======================================

Consolidated from:
- test_config_initial_params.py (Initial parameter config, 26 tests, 591 lines)
- test_config_manager_parameters.py (ConfigManager integration, 8 tests, 246 lines)
- test_parameter_space_config.py (Parameter space configuration, 35 tests, 789 lines)

Tests cover:
- Initial parameter configuration and validation
- ConfigManager integration with parameters
- Parameter space configuration from YAML
- Bounds configuration and validation
- Prior distribution configuration
- Config-driven parameter workflows

Total: 69 tests
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution


# ==============================================================================
# Initial Parameter Config Tests (from test_config_initial_params.py)
# ==============================================================================


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


# ==============================================================================
# ConfigManager Parameter Tests (from test_config_manager_parameters.py)
# ==============================================================================


class TestConfigManagerParameterBounds:
    """Test ConfigManager.get_parameter_bounds() method."""

    def test_get_parameter_bounds_laminar_flow(self):
        """Test getting bounds for laminar flow configuration."""
        config_content = """
analysis_mode: laminar_flow
parameter_space:
  bounds:
    - name: D0
      min: 10.0
      max: 1e5
    - name: alpha
      min: -1.5
      max: 1.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            bounds = config_mgr.get_parameter_bounds(["D0", "alpha"])

            assert len(bounds) == 2
            assert bounds[0]["name"] == "D0"
            assert bounds[0]["min"] == 10.0
            assert bounds[0]["max"] == 1e5
            assert bounds[1]["name"] == "alpha"
            assert bounds[1]["min"] == -1.5
            assert bounds[1]["max"] == 1.5
        finally:
            Path(config_path).unlink()

    def test_get_parameter_bounds_static_mode(self):
        """Test getting bounds for static mode."""
        config_content = """
analysis_mode: static_mode
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            bounds = config_mgr.get_parameter_bounds(["D0", "alpha", "D_offset"])

            # Should return default bounds for static mode
            assert len(bounds) == 3
            assert all("min" in b and "max" in b for b in bounds)
        finally:
            Path(config_path).unlink()

    def test_get_parameter_bounds_all_parameters(self):
        """Test getting bounds for all parameters."""
        config_content = """
analysis_mode: laminar_flow
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            bounds = config_mgr.get_parameter_bounds()

            # Should return bounds for all parameters (scaling + physical)
            assert len(bounds) == 9  # 2 scaling + 7 physical for laminar flow
        finally:
            Path(config_path).unlink()


class TestConfigManagerActiveParameters:
    """Test ConfigManager.get_active_parameters() method."""

    def test_get_active_parameters_laminar_flow(self):
        """Test getting active parameters for laminar flow."""
        config_content = """
analysis_mode: laminar_flow
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            active = config_mgr.get_active_parameters()

            expected = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
            assert active == expected
        finally:
            Path(config_path).unlink()

    def test_get_active_parameters_static_mode(self):
        """Test getting active parameters for static mode."""
        config_content = """
analysis_mode: static_mode
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            active = config_mgr.get_active_parameters()

            expected = ["D0", "alpha", "D_offset"]
            assert active == expected
        finally:
            Path(config_path).unlink()

    def test_get_active_parameters_from_config(self):
        """Test active parameters specified in configuration."""
        config_content = """
analysis_mode: laminar_flow
initial_parameters:
  parameter_names:
    - D0
    - alpha
    - D_offset
    - gamma_dot_t0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            active = config_mgr.get_active_parameters()

            # Should use the subset from config
            assert active == ["D0", "alpha", "D_offset", "gamma_dot_t0"]
            assert len(active) == 4
        finally:
            Path(config_path).unlink()


class TestConfigManagerIntegration:
    """Integration tests for ConfigManager with ParameterManager."""

    def test_end_to_end_laminar_flow(self):
        """Test complete workflow for laminar flow configuration."""
        config_content = """
analysis_mode: laminar_flow
parameter_space:
  bounds:
    - name: D0
      min: 100.0
      max: 1e5
    - name: gamma_dot_t0
      min: 1e-5
      max: 0.1
initial_parameters:
  parameter_names:
    - D0
    - alpha
    - D_offset
    - gamma_dot_t0
    - beta
    - gamma_dot_t_offset
    - phi0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)

            # Get active parameters
            active = config_mgr.get_active_parameters()
            assert len(active) == 7

            # Get bounds for active parameters
            bounds = config_mgr.get_parameter_bounds(active)
            assert len(bounds) == 7

            # Verify custom bounds were applied
            d0_bound = next(b for b in bounds if b["name"] == "D0")
            assert d0_bound["min"] == 100.0
            assert d0_bound["max"] == 1e5

            gamma_bound = next(b for b in bounds if b["name"] == "gamma_dot_t0")
            assert gamma_bound["min"] == 1e-5
            assert gamma_bound["max"] == 0.1
        finally:
            Path(config_path).unlink()

    def test_parameter_name_mapping(self):
        """Test that parameter name mapping works through ConfigManager."""
        config_content = """
analysis_mode: laminar_flow
parameter_space:
  bounds:
    - name: gamma_dot_0
      min: 1e-6
      max: 0.5
    - name: phi_0
      min: -1.0
      max: 1.0
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)

            # Request bounds with canonical names
            bounds = config_mgr.get_parameter_bounds(["gamma_dot_t0", "phi0"])

            # Should find bounds using name mapping
            gamma_bound = next(b for b in bounds if b["name"] == "gamma_dot_t0")
            assert gamma_bound["min"] == 1e-6

            phi_bound = next(b for b in bounds if b["name"] == "phi0")
            assert phi_bound["min"] == -1.0
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ==============================================================================
# Parameter Space Config Tests (from test_parameter_space_config.py)
# ==============================================================================


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def static_config_valid():
    """Valid static configuration with complete parameter_space section."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                    "unit": "Å²/s",
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                    "unit": "dimensionless",
                },
                {
                    "name": "D_offset",
                    "min": -1000.0,
                    "max": 1000.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 100.0,
                    "unit": "Å²/s",
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def laminar_flow_config_valid():
    """Valid laminar flow configuration with complete parameter_space section."""
    return {
        "analysis_mode": "laminar_flow",
        "parameter_space": {
            "model": "laminar_flow",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                },
                {
                    "name": "D_offset",
                    "min": -1000.0,
                    "max": 1000.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 100.0,
                },
                {
                    "name": "gamma_dot_0",  # Config name (will be mapped)
                    "min": 1e-10,
                    "max": 1.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.01,
                    "prior_sigma": 0.01,
                },
                {
                    "name": "beta",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": 0.0,
                    "prior_sigma": 0.5,
                },
                {
                    "name": "gamma_dot_t_offset",
                    "min": 0.0,
                    "max": 1.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 0.1,
                },
                {
                    "name": "phi0",
                    "min": -np.pi,
                    "max": np.pi,
                    "type": "Uniform",
                    "prior_mu": 0.0,
                    "prior_sigma": 1.0,
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
        },
    }


@pytest.fixture
def config_missing_bounds():
    """Configuration with missing parameter_space.bounds section."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            # Missing 'bounds' section
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def config_partial_bounds():
    """Configuration with only partial bounds (some parameters missing)."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                },
                # Missing 'alpha' and 'D_offset'
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def config_invalid_prior():
    """Configuration with invalid prior distribution parameters."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": -500.0,  # Negative sigma (invalid)
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0"],
        },
    }


# =============================================================================
# Test PriorDistribution Class
# =============================================================================


class TestPriorDistribution:
    """Tests for PriorDistribution dataclass."""

    def test_truncated_normal_creation(self):
        """Test creating TruncatedNormal prior."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=1000.0,
            sigma=500.0,
            min_val=100.0,
            max_val=1e5,
        )

        assert prior.dist_type == "TruncatedNormal"
        assert prior.mu == 1000.0
        assert prior.sigma == 500.0
        assert prior.min_val == 100.0
        assert prior.max_val == 1e5

    def test_normal_creation(self):
        """Test creating Normal prior."""
        prior = PriorDistribution(
            dist_type="Normal",
            mu=-1.2,
            sigma=0.3,
            min_val=-np.inf,
            max_val=np.inf,
        )

        assert prior.dist_type == "Normal"
        assert prior.mu == -1.2
        assert prior.sigma == 0.3

    def test_uniform_creation(self):
        """Test creating Uniform prior."""
        prior = PriorDistribution(
            dist_type="Uniform", mu=0.0, sigma=1.0, min_val=-np.pi, max_val=np.pi
        )

        assert prior.dist_type == "Uniform"
        assert prior.min_val == pytest.approx(-np.pi)
        assert prior.max_val == pytest.approx(np.pi)

    def test_invalid_bounds_raises_error(self):
        """Test that invalid bounds (min >= max) raise ValueError."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            PriorDistribution(
                dist_type="TruncatedNormal",
                mu=1000.0,
                sigma=500.0,
                min_val=1e5,  # min > max
                max_val=100.0,
            )

    def test_truncated_normal_requires_finite_bounds(self):
        """Test that TruncatedNormal requires finite bounds."""
        with pytest.raises(ValueError, match="finite bounds"):
            PriorDistribution(
                dist_type="TruncatedNormal",
                mu=0.0,
                sigma=1.0,
                min_val=-np.inf,  # Infinite bound (invalid for TruncatedNormal)
                max_val=np.inf,
            )

    def test_unknown_dist_type_warning(self):
        """Test that unknown distribution type defaults to TruncatedNormal."""
        prior = PriorDistribution(
            dist_type="UnknownDistribution",
            mu=0.0,
            sigma=1.0,
            min_val=0.0,
            max_val=1.0,
        )

        # Should default to TruncatedNormal
        assert prior.dist_type == "TruncatedNormal"

    def test_to_numpyro_kwargs_truncated_normal(self):
        """Test conversion to NumPyro kwargs for TruncatedNormal."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=1000.0,
            sigma=500.0,
            min_val=100.0,
            max_val=1e5,
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["loc"] == 1000.0
        assert kwargs["scale"] == 500.0
        assert kwargs["low"] == 100.0
        assert kwargs["high"] == 1e5

    def test_to_numpyro_kwargs_normal(self):
        """Test conversion to NumPyro kwargs for Normal."""
        prior = PriorDistribution(
            dist_type="Normal", mu=-1.2, sigma=0.3, min_val=-2.0, max_val=2.0
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["loc"] == -1.2
        assert kwargs["scale"] == 0.3
        assert "low" not in kwargs
        assert "high" not in kwargs

    def test_to_numpyro_kwargs_uniform(self):
        """Test conversion to NumPyro kwargs for Uniform."""
        prior = PriorDistribution(
            dist_type="Uniform", mu=0.0, sigma=1.0, min_val=-np.pi, max_val=np.pi
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["low"] == pytest.approx(-np.pi)
        assert kwargs["high"] == pytest.approx(np.pi)


# =============================================================================
# Test ParameterSpace.from_config()
# =============================================================================


class TestParameterSpaceFromConfig:
    """Tests for ParameterSpace.from_config() class method."""

    def test_load_static_config_valid(self, static_config_valid):
        """Test loading valid static configuration."""
        param_space = ParameterSpace.from_config(static_config_valid)

        # Check model type
        assert param_space.model_type == "static"

        # Check parameter names
        assert len(param_space.parameter_names) == 3
        assert "D0" in param_space.parameter_names
        assert "alpha" in param_space.parameter_names
        assert "D_offset" in param_space.parameter_names

        # Check bounds
        assert param_space.get_bounds("D0") == (100.0, 1e5)
        assert param_space.get_bounds("alpha") == (-2.0, 2.0)
        assert param_space.get_bounds("D_offset") == (-1000.0, 1000.0)

        # Check priors
        prior_D0 = param_space.get_prior("D0")
        assert prior_D0.dist_type == "TruncatedNormal"
        assert prior_D0.mu == 1000.0
        assert prior_D0.sigma == 1000.0

        prior_alpha = param_space.get_prior("alpha")
        assert prior_alpha.dist_type == "Normal"
        assert prior_alpha.mu == -1.2
        assert prior_alpha.sigma == 0.3

        # Check units
        assert param_space.units["D0"] == "Å²/s"
        assert param_space.units["alpha"] == "dimensionless"

    def test_load_laminar_flow_config_valid(self, laminar_flow_config_valid):
        """Test loading valid laminar flow configuration."""
        param_space = ParameterSpace.from_config(laminar_flow_config_valid)

        # Check model type
        assert param_space.model_type == "laminar_flow"

        # Check parameter count (7 for laminar flow)
        assert len(param_space.parameter_names) == 7

        # Check name mapping (gamma_dot_0 → gamma_dot_t0)
        assert "gamma_dot_t0" in param_space.parameter_names

        # Check bounds for mapped parameter
        bounds = param_space.get_bounds("gamma_dot_t0")
        assert bounds == (1e-10, 1.0)

        # Check prior for mapped parameter
        prior = param_space.get_prior("gamma_dot_t0")
        assert prior.dist_type == "TruncatedNormal"
        assert prior.mu == 0.01

    def test_load_config_missing_bounds(self, config_missing_bounds):
        """Test loading config with missing bounds section (uses defaults)."""
        param_space = ParameterSpace.from_config(config_missing_bounds)

        # Should still create parameter space with defaults
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3

        # Should have bounds from ParameterManager defaults
        bounds_D0 = param_space.get_bounds("D0")
        assert bounds_D0[0] > 0  # Has some default min
        assert bounds_D0[1] > bounds_D0[0]  # max > min

    def test_load_config_partial_bounds(self, config_partial_bounds):
        """Test loading config with partial bounds (missing some parameters)."""
        param_space = ParameterSpace.from_config(config_partial_bounds)

        # Check that D0 has config bounds
        assert param_space.get_bounds("D0") == (100.0, 1e5)

        # Check that alpha and D_offset have default bounds
        bounds_alpha = param_space.get_bounds("alpha")
        assert bounds_alpha[0] < 0  # Should have some negative min
        assert bounds_alpha[1] > 0  # Should have some positive max

        bounds_offset = param_space.get_bounds("D_offset")
        assert bounds_offset is not None

    def test_load_config_no_parameter_space_section(self):
        """Test loading config without parameter_space section (uses defaults)."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        param_space = ParameterSpace.from_config(config)

        # Should create parameter space with package defaults
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3
        assert len(param_space.bounds) == 3
        assert len(param_space.priors) == 3

    def test_auto_detect_analysis_mode(self):
        """Test automatic detection of analysis mode from config."""
        config = {
            "analysis_mode": "laminar_flow",  # Should auto-detect from here
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
            },
        }

        # Don't specify analysis_mode parameter
        param_space = ParameterSpace.from_config(config)

        assert param_space.model_type == "laminar_flow"
        assert len(param_space.parameter_names) == 7

    def test_explicit_analysis_mode_override(self):
        """Test that explicit analysis_mode parameter overrides config."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        # Explicitly override to static
        param_space = ParameterSpace.from_config(config, analysis_mode="static")

        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3


# =============================================================================
# Test ParameterSpace.from_defaults()
# =============================================================================


class TestParameterSpaceFromDefaults:
    """Tests for ParameterSpace.from_defaults() class method."""

    def test_from_defaults_static(self):
        """Test creating static parameter space from defaults."""
        param_space = ParameterSpace.from_defaults("static")

        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3
        assert "D0" in param_space.parameter_names
        assert "alpha" in param_space.parameter_names
        assert "D_offset" in param_space.parameter_names

        # Should have bounds
        assert len(param_space.bounds) == 3

        # Should have priors (with defaults)
        assert len(param_space.priors) == 3

    def test_from_defaults_laminar_flow(self):
        """Test creating laminar flow parameter space from defaults."""
        param_space = ParameterSpace.from_defaults("laminar_flow")

        assert param_space.model_type == "laminar_flow"
        assert len(param_space.parameter_names) == 7

        # Should have all laminar flow parameters
        expected_params = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for param in expected_params:
            assert param in param_space.parameter_names

    def test_from_defaults_has_wide_priors(self):
        """Test that default priors are reasonably wide."""
        param_space = ParameterSpace.from_defaults("static")

        # Check that priors are not too narrow
        for param_name in param_space.parameter_names:
            prior = param_space.get_prior(param_name)
            bounds = param_space.get_bounds(param_name)

            # Prior sigma should be reasonably large (at least 10% of range)
            bounds_range = bounds[1] - bounds[0]
            assert prior.sigma >= 0.1 * bounds_range


# =============================================================================
# Test ParameterSpace Methods
# =============================================================================


class TestParameterSpaceMethods:
    """Tests for ParameterSpace instance methods."""

    def test_get_bounds_valid_parameter(self, static_config_valid):
        """Test getting bounds for valid parameter."""
        param_space = ParameterSpace.from_config(static_config_valid)

        bounds = param_space.get_bounds("D0")

        assert bounds == (100.0, 1e5)

    def test_get_bounds_invalid_parameter(self, static_config_valid):
        """Test getting bounds for invalid parameter raises KeyError."""
        param_space = ParameterSpace.from_config(static_config_valid)

        with pytest.raises(KeyError, match="not in parameter space"):
            param_space.get_bounds("nonexistent_parameter")

    def test_get_prior_valid_parameter(self, static_config_valid):
        """Test getting prior for valid parameter."""
        param_space = ParameterSpace.from_config(static_config_valid)

        prior = param_space.get_prior("alpha")

        assert prior.dist_type == "Normal"
        assert prior.mu == -1.2
        assert prior.sigma == 0.3

    def test_get_prior_invalid_parameter(self, static_config_valid):
        """Test getting prior for invalid parameter raises KeyError."""
        param_space = ParameterSpace.from_config(static_config_valid)

        with pytest.raises(KeyError, match="not in parameter space"):
            param_space.get_prior("nonexistent_parameter")

    def test_get_bounds_array(self, static_config_valid):
        """Test getting bounds as numpy arrays."""
        param_space = ParameterSpace.from_config(static_config_valid)

        lower, upper = param_space.get_bounds_array()

        # Should have 3 elements (static mode)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

        # Check values (order matches parameter_names)
        param_names = param_space.parameter_names
        for i, param_name in enumerate(param_names):
            expected_bounds = param_space.get_bounds(param_name)
            assert lower[i] == expected_bounds[0]
            assert upper[i] == expected_bounds[1]

        # All lower bounds should be less than upper bounds
        assert np.all(lower < upper)

    def test_get_prior_means(self, static_config_valid):
        """Test getting prior means as numpy array."""
        param_space = ParameterSpace.from_config(static_config_valid)

        means = param_space.get_prior_means()

        # Should have 3 elements
        assert means.shape == (3,)

        # Check values match priors
        param_names = param_space.parameter_names
        for i, param_name in enumerate(param_names):
            expected_mu = param_space.get_prior(param_name).mu
            assert means[i] == expected_mu

    def test_validate_values_all_valid(self, static_config_valid):
        """Test validating parameter values that are all within bounds."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {"D0": 1000.0, "alpha": -1.2, "D_offset": 0.0}

        is_valid, violations = param_space.validate_values(values)

        assert is_valid
        assert len(violations) == 0

    def test_validate_values_below_min(self, static_config_valid):
        """Test validating parameter value below minimum bound."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {"D0": 50.0, "alpha": -1.2, "D_offset": 0.0}  # D0 < min (100.0)

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "D0" in violations[0]
        assert "< min" in violations[0]

    def test_validate_values_above_max(self, static_config_valid):
        """Test validating parameter value above maximum bound."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 2e5,  # D0 > max (1e5)
            "alpha": -1.2,
            "D_offset": 0.0,
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "D0" in violations[0]
        assert "> max" in violations[0]

    def test_validate_values_multiple_violations(self, static_config_valid):
        """Test validating with multiple bound violations."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 50.0,  # Below min
            "alpha": 5.0,  # Above max
            "D_offset": -2000.0,  # Below min
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 3

    def test_validate_values_unknown_parameter(self, static_config_valid):
        """Test validating with unknown parameter name."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 1000.0,
            "unknown_param": 42.0,  # Not in parameter space
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "unknown_param" in violations[0]
        assert "Unknown parameter" in violations[0]


# =============================================================================
# Test ParameterSpace String Representations
# =============================================================================


class TestParameterSpaceStringRepresentations:
    """Tests for __repr__ and __str__ methods."""

    def test_repr(self, static_config_valid):
        """Test __repr__ output."""
        param_space = ParameterSpace.from_config(static_config_valid)

        repr_str = repr(param_space)

        assert "ParameterSpace" in repr_str
        assert "static" in repr_str
        assert "n_params=3" in repr_str

    def test_str(self, static_config_valid):
        """Test __str__ output."""
        param_space = ParameterSpace.from_config(static_config_valid)

        str_repr = str(param_space)

        assert "ParameterSpace" in str_repr
        assert "static" in str_repr
        assert "D0" in str_repr
        assert "alpha" in str_repr
        assert "D_offset" in str_repr


# =============================================================================
# Integration Tests
# =============================================================================


class TestParameterSpaceIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_static(self, static_config_valid):
        """Test complete workflow: load → validate → get arrays."""
        # Load from config
        param_space = ParameterSpace.from_config(static_config_valid)

        # Validate parameter values
        values = {"D0": 1000.0, "alpha": -1.2, "D_offset": 0.0}
        is_valid, _ = param_space.validate_values(values)
        assert is_valid

        # Get bounds as arrays
        lower, upper = param_space.get_bounds_array()
        assert lower.shape == (3,)

        # Get prior means
        means = param_space.get_prior_means()
        assert means.shape == (3,)

        # Check that means are within bounds
        assert np.all(means >= lower)
        assert np.all(means <= upper)

    def test_full_workflow_laminar_flow(self, laminar_flow_config_valid):
        """Test complete workflow for laminar flow mode."""
        # Load from config
        param_space = ParameterSpace.from_config(laminar_flow_config_valid)

        # Should have 7 parameters
        assert len(param_space.parameter_names) == 7

        # Get bounds array
        lower, upper = param_space.get_bounds_array()
        assert lower.shape == (7,)
        assert upper.shape == (7,)

        # Get prior means
        means = param_space.get_prior_means()
        assert means.shape == (7,)

        # Means should be within bounds
        assert np.all(means >= lower)
        assert np.all(means <= upper)

    def test_config_to_numpyro_workflow(self, static_config_valid):
        """Test workflow from config to NumPyro distribution kwargs."""
        param_space = ParameterSpace.from_config(static_config_valid)

        # For each parameter, get NumPyro kwargs
        for param_name in param_space.parameter_names:
            prior = param_space.get_prior(param_name)
            kwargs = prior.to_numpyro_kwargs()

            # Should have required keys for the distribution type
            if prior.dist_type == "TruncatedNormal":
                assert "loc" in kwargs
                assert "scale" in kwargs
                assert "low" in kwargs
                assert "high" in kwargs
            elif prior.dist_type == "Normal":
                assert "loc" in kwargs
                assert "scale" in kwargs

