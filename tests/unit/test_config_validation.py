"""
Unit Tests for Configuration Validation
========================================

Consolidated from:
- test_device_config.py (Hardware config & CMC/NUTS selection, 10 tests, 307 lines)
- test_edge_cases_config_validation.py (Config validation edge cases, 34 tests, 513 lines)

Tests cover:
- Hardware detection and CMC/NUTS selection logic
- Dual-criteria OR logic: (num_samples >= 15) OR (memory > 30%)
- Platform-specific configurations
- Malformed config file handling
- Parameter value validation edge cases
- Threshold validation and extreme values
- Invalid data types and boundary conditions

Total: 44 tests
"""

import pytest
import numpy as np
import argparse

from homodyne.device.config import HardwareConfig, should_use_cmc
from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution
from homodyne.cli.args_parser import validate_args


# ==============================================================================
# Hardware Config & CMC Selection Tests (from test_device_config.py)
# ==============================================================================



class TestCMCSelectionLogic:
    """Test CMC selection logic with dual-criteria OR conditions."""

    @pytest.fixture
    def mock_hardware_config(self):
        """Create a mock hardware configuration for testing."""
        return HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=32.0,
            num_nodes=1,
            cores_per_node=14,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=14,
        )

    def test_dual_criteria_or_logic_parallelism_mode(self, mock_hardware_config):
        """
        Test dual-criteria OR logic: num_samples >= 15 triggers CMC (parallelism mode).

        Spec requirement: (num_samples >= 15) OR (memory > 30%) → CMC
        This test verifies the parallelism trigger.
        """
        # 20 samples with small dataset → CMC via parallelism trigger
        num_samples = 20
        dataset_size = 1_000_000  # Small enough to not trigger memory threshold

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            f"CMC should be selected for {num_samples} samples "
            f"(>= 15 threshold) even with small dataset"
        )

    def test_dual_criteria_or_logic_memory_mode(self, mock_hardware_config):
        """
        Test dual-criteria OR logic: memory > 30% triggers CMC (memory mode).

        Spec requirement: (num_samples >= 15) OR (memory > 30%) → CMC
        This test verifies the memory trigger.
        """
        # 5 samples (below threshold) but HUGE dataset → CMC via memory trigger
        num_samples = 5
        dataset_size = 50_000_000  # Large enough to trigger memory threshold

        # Calculate expected memory: 50M * 8 bytes * 30 / 1e9 ≈ 12 GB
        # 12 GB / 32 GB = 37.5% > 30% threshold → should trigger CMC

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            f"CMC should be selected for large dataset ({dataset_size:,} points) "
            f"even with only {num_samples} samples (< 15 threshold)"
        )

    def test_parallelism_trigger_exactly_at_threshold(self, mock_hardware_config):
        """
        Test parallelism trigger: exactly 15 samples should trigger CMC.

        Spec requirement: min_samples_for_cmc=15 (default)
        Boundary condition: exactly at threshold.
        """
        num_samples = 15  # Exactly at threshold
        dataset_size = 1_000_000

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert (
            result is True
        ), "CMC should be selected at exactly 15 samples (threshold boundary)"

    def test_nuts_selection_below_all_thresholds(self, mock_hardware_config):
        """
        Test NUTS selection: small samples AND small dataset → NUTS.

        Spec requirement: If all quad-criteria fail → NUTS

        NOTE: Must use dataset_size < 1M to avoid triggering Criterion 3
        (JAX Broadcasting Protection at 1M threshold)
        """
        num_samples = 10  # Below 15 threshold
        dataset_size = 500_000  # Small dataset (< 1M to avoid Criterion 3)

        # Calculate expected memory: 500K * 8 bytes * 30 / 1e9 ≈ 0.12 GB
        # 0.12 GB / 32 GB = 0.375% < 30% threshold → all quad-criteria fail

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, (
            f"NUTS should be selected for {num_samples} samples (< 15), "
            f"small dataset (< 30% memory), and dataset_size < 1M (no broadcasting protection)"
        )

    def test_threshold_configurability_custom_sample_threshold(
        self, mock_hardware_config
    ):
        """
        Test threshold configurability: custom min_samples_for_cmc.

        Spec requirement: Thresholds should be configurable from YAML.
        """
        num_samples = 25
        dataset_size = 1_000_000

        # Test with custom threshold of 30 samples
        result_high_threshold = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=30,  # Custom: higher than default 15
        )

        assert (
            result_high_threshold is False
        ), f"NUTS should be selected when {num_samples} < custom threshold (30)"

        # Test with custom threshold of 20 samples
        result_low_threshold = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=20,  # Custom: 25 >= 20
        )

        assert (
            result_low_threshold is True
        ), f"CMC should be selected when {num_samples} >= custom threshold (20)"

    def test_threshold_configurability_custom_memory_threshold(
        self, mock_hardware_config
    ):
        """
        Test threshold configurability: custom memory_threshold_pct.

        Spec requirement: Thresholds should be configurable from YAML.

        NOTE (Quad-Criteria): With large dataset (30M > 1M), Criterion 3
        (JAX Broadcasting Protection) and Criterion 4 (Large Dataset, Few Samples)
        will also trigger. This test verifies memory threshold configurability
        while acknowledging other criteria may override.
        """
        num_samples = 5  # Below sample threshold
        dataset_size = 30_000_000

        # Memory: 30M * 8 * 5 / 1e9 = 1.2 GB → 1.2/32 = 3.75% memory usage
        # Note: Adjusted num_samples from 30 to 5 to match test intent

        # Test with strict 1% threshold
        result_strict = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.01,  # 3.75% > 1% → trigger CMC (Criterion 2)
            min_samples_for_cmc=15,
        )

        assert (
            result_strict is True
        ), "CMC should be selected when memory (3.75%) > strict threshold (1%)"

        # Test with relaxed 10% threshold
        # NOTE: Criterion 3 and 4 will still trigger due to large dataset
        result_relaxed = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.10,  # 3.75% < 10% → Criterion 2 doesn't trigger
            min_samples_for_cmc=15,
        )

        # With quad-criteria, CMC will still be selected due to Criteria 3 and 4
        assert (
            result_relaxed is True
        ), "CMC selected due to Criterion 3 (JAX protection) and Criterion 4 (large dataset, few samples)"

    def test_memory_estimation_without_dataset_size(self, mock_hardware_config):
        """
        Test behavior when dataset_size is None (only sample-based decision).

        Spec requirement: Memory trigger only applies when dataset_size is provided.
        """
        num_samples = 10  # Below threshold

        # No dataset_size provided → only sample-based decision
        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=None,  # No memory estimation
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, (
            "NUTS should be selected when num_samples < threshold and "
            "dataset_size is None (no memory trigger)"
        )

    def test_realistic_xpcs_scenario_small_experiment(self, mock_hardware_config):
        """
        Test realistic XPCS scenario: small experiment (10 phi angles, 500K points).

        Real-world use case: Small dataset that should use NUTS.

        NOTE: Using 500K points (< 1M) to avoid Criterion 3 (JAX Broadcasting Protection).
        Real XPCS experiments with < 1M points are common for quick scans.
        """
        num_samples = 10  # 10 phi angles
        dataset_size = 500_000  # 500K points total (< 1M to avoid Criterion 3)

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, "Small XPCS experiment should use NUTS"

    def test_realistic_xpcs_scenario_medium_experiment(self, mock_hardware_config):
        """
        Test realistic XPCS scenario: medium experiment (20 phi angles, 10M points).

        Real-world use case: Parallelism-triggered CMC.
        """
        num_samples = 20  # 20 phi angles
        dataset_size = 10_000_000  # 10M points total

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            "Medium XPCS experiment with 20 phi angles should use CMC "
            "(parallelism mode)"
        )

    def test_realistic_xpcs_scenario_large_memory_experiment(
        self, mock_hardware_config
    ):
        """
        Test realistic XPCS scenario: few angles but huge data (5 phi, 50M points).

        Real-world use case: Memory-triggered CMC.
        """
        num_samples = 5  # Only 5 phi angles
        dataset_size = 50_000_000  # 50M points total

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            "Large memory XPCS experiment should use CMC (memory mode) "
            "even with only 5 phi angles"
        )


# ==============================================================================
# Config Validation Edge Case Tests (from test_edge_cases_config_validation.py)
# ==============================================================================



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
        config_mgr = ConfigManager(config_override=config)
        param_space = ParameterSpace.from_config(
            config, analysis_mode="static"
        )
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
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [-1000.0, 0.5, 10.0],  # Negative D0 is invalid
            },
        }
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


class TestExtremeThresholdValues:
    """Test extreme and invalid threshold values in config."""

    def test_negative_min_samples_for_cmc(self):
        """Test that negative min_samples_for_cmc is in config."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": -1,  # Invalid: negative
                    "memory_threshold_pct": 0.30,
                }
            }
        }
        # Config manager should load it (validation happens downstream)
        config_mgr = ConfigManager(config_override=config)
        # Just verify config loads without crashing
        assert config_mgr.config is not None

    def test_zero_min_samples_for_cmc(self):
        """Test that zero min_samples_for_cmc is in config."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 0,  # Edge case: zero
                    "memory_threshold_pct": 0.30,
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        assert config_mgr.config is not None

    def test_negative_memory_threshold(self):
        """Test that negative memory threshold is in config."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 15,
                    "memory_threshold_pct": -0.1,  # Invalid: negative
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        # Should load
        assert config_mgr.config is not None

    def test_memory_threshold_above_one(self):
        """Test that memory threshold > 1.0 is in config."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 15,
                    "memory_threshold_pct": 1.5,  # Invalid: > 1.0
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        # Should load
        assert config_mgr.config is not None

    def test_memory_threshold_exactly_one(self):
        """Test that memory threshold = 1.0 is in config."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 15,
                    "memory_threshold_pct": 1.0,  # Edge case: exactly 100%
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        assert config_mgr.config is not None

    def test_very_large_min_samples_for_cmc(self):
        """Test extremely large min_samples_for_cmc."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 1000000,  # Very large number
                    "memory_threshold_pct": 0.30,
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        # Should be valid (just makes CMC less likely to trigger)
        assert config_mgr.config is not None

    def test_memory_threshold_boundary_zero(self):
        """Test memory threshold = 0.0 (CMC never triggered by memory)."""
        config = {
            "optimization": {
                "mcmc": {
                    "min_samples_for_cmc": 15,
                    "memory_threshold_pct": 0.0,  # Edge case: 0%
                }
            }
        }
        config_mgr = ConfigManager(config_override=config)
        # Should be valid (memory criterion will always be false)
        assert config_mgr.config is not None


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

