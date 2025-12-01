"""Test loading of per-angle scaling parameters (contrast, offset) from config.

This test verifies the fix for the issue where initial contrast and offset values
from the per_angle_scaling section were not being loaded into MCMC initialization.
"""

import pytest
from homodyne.config.manager import ConfigManager


class TestPerAngleScalingInitialization:
    """Test suite for per-angle scaling parameter loading."""

    def test_load_single_angle_contrast_offset(self):
        """Test loading scalar contrast and offset for single-angle analysis."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, -1.5, 3.0],
                "per_angle_scaling": {
                    "contrast": [0.05],
                    "offset": [1.001]
                }
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100.0, "max": 100.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify physical parameters
        assert params["D0"] == 1000.0
        assert params["alpha"] == -1.5
        assert params["D_offset"] == 3.0

        # Verify scaling parameters from per_angle_scaling
        assert "contrast" in params
        assert "offset" in params
        assert params["contrast"] == 0.05
        assert params["offset"] == 1.001

    def test_load_multi_angle_contrast_offset(self):
        """Test loading per-angle contrast and offset for multi-angle analysis."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, -1.5, 3.0],
                "per_angle_scaling": {
                    "contrast": [0.05, 0.06, 0.04],
                    "offset": [1.001, 1.002, 1.000]
                }
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100.0, "max": 100.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify physical parameters
        assert params["D0"] == 1000.0
        assert params["alpha"] == -1.5
        assert params["D_offset"] == 3.0

        # Verify per-angle scaling parameters
        assert "contrast_0" in params
        assert "contrast_1" in params
        assert "contrast_2" in params
        assert params["contrast_0"] == 0.05
        assert params["contrast_1"] == 0.06
        assert params["contrast_2"] == 0.04

        assert "offset_0" in params
        assert "offset_1" in params
        assert "offset_2" in params
        assert params["offset_0"] == 1.001
        assert params["offset_1"] == 1.002
        assert params["offset_2"] == 1.000

    def test_missing_per_angle_scaling_section(self):
        """Test that config without per_angle_scaling still works."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, -1.5, 3.0]
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100.0, "max": 100.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify physical parameters
        assert params["D0"] == 1000.0
        assert params["alpha"] == -1.5
        assert params["D_offset"] == 3.0

        # Verify no contrast/offset were added
        assert "contrast" not in params
        assert "offset" not in params

    def test_partial_per_angle_scaling_contrast_only(self):
        """Test config with only contrast specified in per_angle_scaling."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, -1.5, 3.0],
                "per_angle_scaling": {
                    "contrast": [0.05]
                }
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100.0, "max": 100.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify physical parameters
        assert params["D0"] == 1000.0
        assert params["alpha"] == -1.5
        assert params["D_offset"] == 3.0

        # Verify contrast was added, but not offset
        assert "contrast" in params
        assert params["contrast"] == 0.05
        assert "offset" not in params

    def test_partial_per_angle_scaling_offset_only(self):
        """Test config with only offset specified in per_angle_scaling."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1000.0, -1.5, 3.0],
                "per_angle_scaling": {
                    "offset": [1.001]
                }
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100.0, "max": 100.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify physical parameters
        assert params["D0"] == 1000.0
        assert params["alpha"] == -1.5
        assert params["D_offset"] == 3.0

        # Verify offset was added, but not contrast
        assert "contrast" not in params
        assert "offset" in params
        assert params["offset"] == 1.001

    def test_realistic_simon_config_values(self):
        """Test with realistic values from Simon's configuration."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [16834.85791844444, -1.571278060789398, 3.0262561756929505],
                "per_angle_scaling": {
                    "contrast": [0.05014977988697604],
                    "offset": [1.001402641457075]
                }
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 100000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -100000.0, "max": 100000.0}
                ]
            }
        }

        config_mgr = ConfigManager(config_override=config)
        params = config_mgr.get_initial_parameters()

        # Verify all parameters match config exactly
        assert params["D0"] == pytest.approx(16834.85791844444)
        assert params["alpha"] == pytest.approx(-1.571278060789398)
        assert params["D_offset"] == pytest.approx(3.0262561756929505)
        assert params["contrast"] == pytest.approx(0.05014977988697604)
        assert params["offset"] == pytest.approx(1.001402641457075)
