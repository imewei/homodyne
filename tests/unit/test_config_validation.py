"""
Unit Tests for Configuration Validation
========================================

**Consolidation**: Week 6 (2025-11-15)
**Updated**: v3.0 CMC-only migration

Consolidated from:
- test_device_config.py (Hardware config, 5 tests)
- test_edge_cases_config_validation.py (Config validation edge cases, 34 tests)

Test Categories:
---------------
**Hardware Configuration** (5 tests):
- Hardware detection
- Platform-specific configurations (CPU, PBS, SLURM)
- Backend recommendation logic

**Config Validation Edge Cases** (34 tests):
- Malformed config file handling
- Parameter value validation edge cases
- Invalid data types and boundary conditions
- Missing config sections and graceful degradation

Test Coverage:
-------------
- Hardware detection for CMC sharding
- Platform-specific configurations and recommendations
- Malformed config file handling with clear error messages
- Parameter value validation: type checking, range validation, physics constraints
- Invalid data type detection and conversion attempts
- Missing config sections with intelligent fallback behavior
- Graceful degradation when config is incomplete

Note: NUTS/CMC auto-selection removed in v3.0 (CMC-only architecture)

Total: ~39 tests

Usage Example:
-------------
```python
# Run all config validation tests
pytest tests/unit/test_config_validation.py -v

# Run specific category
pytest tests/unit/test_config_validation.py -k "hardware" -v
pytest tests/unit/test_config_validation.py -k "edge" -v
```

See Also:
---------
- docs/migration/v3_cmc_only.md: CMC-only migration guide
- homodyne/device/config.py: Hardware detection
- homodyne/config/manager.py: ConfigManager validation logic
"""

import numpy as np
import pytest

from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

# ==============================================================================
# Config Validation Edge Case Tests (from test_edge_cases_config_validation.py)
# ==============================================================================
# Note: Hardware/CMC selection tests removed in v3.0 (CMC-only architecture)
# See test_hardware_detection.py for HardwareConfig tests


class TestMalformedConfigFiles:
    """Test handling of malformed configuration files."""

    def test_config_missing_analysis_mode(self):
        """Test config without analysis_mode section."""
        config = {
            "initial_parameters": {"parameter_names": ["D0"], "values": [1000.0]},
            # Missing analysis_mode
        }
        config_mgr = ConfigManager(config_override=config)
        # Should still work with defaults
        initial_params = config_mgr.get_initial_parameters()
        assert isinstance(initial_params, dict)

    def test_config_missing_initial_parameters_section(self):
        """Test config without initial_parameters section entirely."""
        config = {
            "analysis_mode": "static",
            # Missing initial_parameters section
            "parameter_space": {"bounds": {"D0": {"min": 100, "max": 5000}}},
        }
        config_mgr = ConfigManager(config_override=config)
        # Should return empty dict or defaults
        initial_params = config_mgr.get_initial_parameters()
        assert isinstance(initial_params, dict)

    def test_config_missing_parameter_space_section(self):
        """Test config without parameter_space section."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {"parameter_names": ["D0"], "values": [1000.0]},
            # Missing parameter_space section
        }
        # Should not raise, fallback to defaults
        ConfigManager(config_override=config)
        param_space = ParameterSpace.from_config(config, analysis_mode="static")
        assert param_space is not None

    def test_config_empty_dict(self):
        """Test with completely empty config."""
        config = {}
        config_mgr = ConfigManager(config_override=config)
        # Should handle gracefully
        initial_params = config_mgr.get_initial_parameters()
        assert isinstance(initial_params, dict)


class TestMissingRequiredFields:
    """Test handling of missing required fields in config."""

    def test_initial_parameters_missing_parameter_names(self):
        """Test initial_parameters without parameter_names."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                # Missing parameter_names
                "values": [1000.0, 0.5, 10.0]
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        # Should handle gracefully or return empty
        assert isinstance(initial_params, dict)

    def test_initial_parameters_missing_values(self):
        """Test initial_parameters without values."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                # Missing values
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        # Should handle gracefully
        assert isinstance(initial_params, dict)

    def test_mismatched_parameter_names_and_values_length(self):
        """Test when parameter_names and values have different lengths."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],  # 2 parameters
                "values": [1000.0, 0.5, 10.0],  # 3 values
            },
        }
        config_mgr = ConfigManager(config_override=config)
        # Should either raise or handle gracefully
        with pytest.raises((ValueError, IndexError, KeyError)):
            config_mgr.get_initial_parameters()

    def test_parameter_space_missing_bounds(self):
        """Test parameter_space without bounds section."""
        config = {
            "analysis_mode": "static",
            "parameter_space": {
                # Missing bounds section
                "priors": {"D0": {"type": "TruncatedNormal", "mu": 1000, "sigma": 100}}
            },
        }
        param_space = ParameterSpace.from_config(config, "static")
        # Should use defaults
        assert param_space is not None


class TestInvalidParameterValues:
    """Test handling of invalid parameter values."""

    def test_nan_parameter_values(self):
        """Test with NaN values in initial parameters."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [float("nan"), 0.5, 10.0],
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        # Should contain NaN (Python allows it, but user might use downstream)
        assert "D0" in initial_params
        assert np.isnan(initial_params["D0"])

    def test_infinity_parameter_values(self):
        """Test with infinity values."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [float("inf"), 0.5, 10.0],
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        assert "D0" in initial_params
        assert np.isinf(initial_params["D0"])

    def test_negative_d0_parameter_invalid(self):
        """Test that negative D0 values should be caught during validation."""
        param_space = ParameterSpace.from_defaults("static")
        initial_params = {"D0": -1000.0, "alpha": 0.5, "D_offset": 10.0}

        # Validation should catch this
        errors = param_space.validate_values(initial_params)
        assert len(errors) > 0
        assert any("D0" in str(e) for e in errors)

    def test_string_instead_of_float_parameter(self):
        """Test type mismatch: string instead of float."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": ["1000.0", 0.5, 10.0],  # First value is string
            },
        }
        config_mgr = ConfigManager(config_override=config)
        # Should attempt type coercion or raise error
        try:
            initial_params = config_mgr.get_initial_parameters()
            # If it succeeds, verify type conversion happened
            assert isinstance(initial_params["D0"], (int, float))
        except (ValueError, TypeError):
            # Type error is also acceptable
            pass

    def test_very_large_parameter_values(self):
        """Test with very large parameter values."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1e20, 0.5, 10.0],  # Very large D0
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        assert initial_params["D0"] == 1e20

    def test_very_small_positive_parameter_values(self):
        """Test with very small but positive parameter values."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [1e-20, 0.5, 10.0],  # Very small D0
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()
        assert initial_params["D0"] == 1e-20

    def test_zero_d0_parameter_invalid(self):
        """Test that zero D0 should be invalid (requires positive diffusion)."""
        param_space = ParameterSpace.from_defaults("static")
        initial_params = {"D0": 0.0, "alpha": 0.5, "D_offset": 10.0}

        errors = param_space.validate_values(initial_params)
        # Zero diffusion should fail validation
        assert len(errors) > 0


# Note: TestExtremeThresholdValues removed in v3.0 (CMC-only architecture)
# min_samples_for_cmc and memory_threshold_pct are no longer used


class TestPriorDistributionEdgeCases:
    """Test edge cases in prior distribution specification."""

    def test_truncated_normal_with_equal_bounds(self):
        """Test TruncatedNormal with min = max."""
        with pytest.raises(ValueError):
            # min_val == max_val should raise
            PriorDistribution(
                dist_type="TruncatedNormal",
                min_val=100.0,
                max_val=100.0,  # min == max
                mu=100.0,
                sigma=10.0,
            )

    def test_truncated_normal_inverted_bounds(self):
        """Test TruncatedNormal with min > max."""
        with pytest.raises(ValueError):
            PriorDistribution(
                dist_type="TruncatedNormal",
                min_val=1000.0,
                max_val=100.0,  # min > max (inverted)
                mu=500.0,
                sigma=10.0,
            )

    def test_truncated_normal_mu_outside_bounds(self):
        """Test TruncatedNormal with mu outside [min, max]."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            min_val=100.0,
            max_val=500.0,
            mu=1000.0,  # Outside bounds
            sigma=10.0,
        )
        # Should warn or accept with clipping
        assert prior is not None

    def test_negative_sigma(self):
        """Test that negative sigma is still allowed (dataclass allows it)."""
        # Dataclass allows negative sigma, but it's physically meaningless
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            min_val=100.0,
            max_val=500.0,
            mu=300.0,
            sigma=-10.0,  # Negative sigma unusual but not prohibited
        )
        assert prior is not None

    def test_zero_sigma(self):
        """Test zero sigma (degenerate distribution)."""
        # Zero sigma means all probability at mu
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            min_val=100.0,
            max_val=500.0,
            mu=300.0,
            sigma=0.0,  # Degenerate
        )
        # May be problematic for MCMC but should be allowed
        assert prior is not None

    def test_unknown_distribution_type(self):
        """Test unknown distribution type."""
        prior = PriorDistribution(
            dist_type="UnknownDistribution",  # Unknown type
            min_val=100.0,
            max_val=500.0,
            mu=300.0,
            sigma=10.0,
        )
        # Should warn and default to TruncatedNormal
        assert prior.dist_type == "TruncatedNormal"


class TestParameterSpaceEdgeCases:
    """Test edge cases in ParameterSpace."""

    def test_parameter_space_empty_config(self):
        """Test ParameterSpace with completely empty config."""
        config = {}
        param_space = ParameterSpace.from_config(config, "static")
        # Should use defaults
        assert param_space is not None
        # Check it has parameters by trying to access them
        bounds = param_space.get_bounds("D0")
        assert bounds is not None

    def test_parameter_space_null_bounds(self):
        """Test parameter with null bounds in config."""
        config = {
            "parameter_space": {
                "bounds": {
                    "D0": None,  # Null bounds
                    "alpha": {"min": 0.1, "max": 1.0},
                }
            }
        }
        param_space = ParameterSpace.from_config(config, "static")
        # Should fallback to default bounds
        assert param_space is not None
        # Check D0 has default bounds
        bounds = param_space.get_bounds("D0")
        assert bounds is not None

    def test_parameter_space_single_parameter(self):
        """Test ParameterSpace with only one parameter."""
        config = {"parameter_space": {"bounds": {"D0": {"min": 100, "max": 5000}}}}
        param_space = ParameterSpace.from_config(config, "static")
        # Should still work
        assert param_space is not None
        # Can get D0 bounds
        bounds = param_space.get_bounds("D0")
        assert bounds is not None


class TestComplexIntegrationEdgeCases:
    """Test complex edge case scenarios combining multiple issues."""

    def test_all_parameters_at_bounds_lower(self):
        """Test all parameters at their minimum bounds."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [100.0, 0.0, 0.0],  # All at minimum
            },
            "parameter_space": {
                "bounds": {
                    "D0": {"min": 100.0, "max": 5000.0},
                    "alpha": {"min": 0.0, "max": 1.0},
                    "D_offset": {"min": 0.0, "max": 1000.0},
                }
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params["D0"] == 100.0
        assert initial_params["alpha"] == 0.0
        assert initial_params["D_offset"] == 0.0

    def test_all_parameters_at_bounds_upper(self):
        """Test all parameters at their maximum bounds."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [5000.0, 1.0, 1000.0],  # All at maximum
            },
            "parameter_space": {
                "bounds": {
                    "D0": {"min": 100.0, "max": 5000.0},
                    "alpha": {"min": 0.0, "max": 1.0},
                    "D_offset": {"min": 0.0, "max": 1000.0},
                }
            },
        }
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        assert initial_params["D0"] == 5000.0
        assert initial_params["alpha"] == 1.0
        assert initial_params["D_offset"] == 1000.0

    def test_midpoint_with_infinite_bounds(self):
        """Test midpoint calculation with infinite bounds."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0"],
                "values": None,  # Will calculate midpoint
            },
            "parameter_space": {"bounds": {"D0": {"min": 100.0, "max": float("inf")}}},
        }
        config_mgr = ConfigManager(config_override=config)
        # Should handle infinite bounds gracefully
        initial_params = config_mgr.get_initial_parameters()
        # Midpoint with infinity might be inf or clamped
        assert isinstance(initial_params["D0"], (int, float, type(np.nan)))
