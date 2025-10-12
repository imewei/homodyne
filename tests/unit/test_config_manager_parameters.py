"""
Unit Tests for ConfigManager Parameter Methods (Phase 4.2)
=========================================================

Tests for parameter-related methods in ConfigManager that use ParameterManager.
"""

import tempfile
from pathlib import Path

import pytest

from homodyne.config.manager import ConfigManager


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
analysis_mode: static_isotropic
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
analysis_mode: static_isotropic
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
