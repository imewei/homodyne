"""Integration Tests for CLI Override → Config → Workflow

End-to-end integration tests verifying the complete CLI override workflow:

1. Load config from YAML file
2. Apply CLI overrides (parameter values, MCMC thresholds)
3. Verify final configuration state
4. Ensure integration with ParameterSpace and ParameterManager
5. Demonstrate realistic user workflows

Test Scenarios:
- NLSQ workflow with parameter overrides
- MCMC workflow with threshold overrides
- Combined parameter + threshold overrides
- Static mode vs laminar flow mode
- Partial overrides (some parameters from config, some from CLI)

Test Coverage:
- test_integration_* (6 tests): End-to-end workflows

Total: 6 tests
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import _apply_cli_overrides
from homodyne.config.manager import ConfigManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_config_file():
    """Create temporary YAML config file for testing."""
    config_dict = {
        "analysis_mode": "laminar_flow",
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
            "values": [1000.0, 0.5, 10.0, 0.001, 0.2, 0.0001, 0.0],
        },
        "parameter_space": {
            "bounds": [
                {"name": "D0", "min": 1.0, "max": 1e6},
                {"name": "alpha", "min": -2.0, "max": 2.0},
                {"name": "D_offset", "min": 0.0, "max": 1e5},
                {"name": "gamma_dot_t0", "min": 1e-5, "max": 1.0},
                {"name": "beta", "min": -2.0, "max": 2.0},
                {"name": "gamma_dot_t_offset", "min": -1.0, "max": 1.0},
                {"name": "phi0", "min": -3.14159, "max": 3.14159},
            ]
        },
        "optimization": {
            "mcmc": {
                "num_samples": 1000,
                "num_warmup": 500,
                "num_chains": 4,
                "min_samples_for_cmc": 15,
                "memory_threshold_pct": 0.30,
                "dense_mass": False,
            }
        },
        "hardware": {"force_cpu": False, "gpu_memory_fraction": 0.9},
    }

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_file = Path(f.name)

    yield temp_file

    # Cleanup
    temp_file.unlink()


@pytest.fixture
def parser():
    """Create argument parser."""
    return create_parser()


# ============================================================================
# Integration Tests: Full Workflow
# ============================================================================


def test_integration_load_config_and_override_parameters(temp_config_file, parser):
    """Test full workflow: load config → override parameters → verify final state."""
    # Step 1: Parse CLI arguments with overrides
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--initial-d0",
            "2000.0",
            "--initial-alpha",
            "0.75",
        ]
    )

    # Step 2: Load configuration
    config = ConfigManager(str(temp_config_file))

    # Step 3: Apply CLI overrides
    _apply_cli_overrides(config, args)

    # Step 4: Verify final configuration state
    param_names = config.config["initial_parameters"]["parameter_names"]
    param_values = config.config["initial_parameters"]["values"]

    # Find overridden parameters
    d0_idx = param_names.index("D0")
    alpha_idx = param_names.index("alpha")
    d_offset_idx = param_names.index("D_offset")

    # Check overrides applied
    assert param_values[d0_idx] == 2000.0  # Overridden
    assert param_values[alpha_idx] == 0.75  # Overridden
    assert param_values[d_offset_idx] == 10.0  # Not overridden (from config)

    # Step 5: Verify integration with ConfigManager.get_initial_parameters()
    initial_params = config.get_initial_parameters()

    assert initial_params["D0"] == 2000.0
    assert initial_params["alpha"] == 0.75
    assert initial_params["D_offset"] == 10.0


def test_integration_load_config_and_override_thresholds(temp_config_file, parser):
    """Test full workflow: load config → override MCMC thresholds → verify."""
    # Step 1: Parse CLI arguments with threshold overrides
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--method",
            "mcmc",
            "--min-samples-cmc",
            "25",
            "--memory-threshold-pct",
            "0.40",
        ]
    )

    # Step 2: Load configuration
    config = ConfigManager(str(temp_config_file))

    # Step 3: Apply CLI overrides
    _apply_cli_overrides(config, args)

    # Step 4: Verify threshold overrides
    mcmc_config = config.config["optimization"]["mcmc"]

    assert mcmc_config["min_samples_for_cmc"] == 25  # Overridden from 15
    assert mcmc_config["memory_threshold_pct"] == 0.40  # Overridden from 0.30


def test_integration_combined_parameter_and_threshold_overrides(
    temp_config_file, parser
):
    """Test combined workflow: override both parameters and thresholds."""
    # Step 1: Parse CLI with both types of overrides
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--method",
            "mcmc",
            "--initial-d0",
            "1500.0",
            "--initial-gamma-dot-t0",
            "0.005",
            "--min-samples-cmc",
            "20",
            "--dense-mass-matrix",
        ]
    )

    # Step 2: Load and apply
    config = ConfigManager(str(temp_config_file))
    _apply_cli_overrides(config, args)

    # Step 3: Verify parameter overrides
    param_names = config.config["initial_parameters"]["parameter_names"]
    param_values = config.config["initial_parameters"]["values"]

    d0_idx = param_names.index("D0")
    gamma_idx = param_names.index("gamma_dot_t0")

    assert param_values[d0_idx] == 1500.0
    assert param_values[gamma_idx] == 0.005

    # Step 4: Verify threshold overrides
    mcmc_config = config.config["optimization"]["mcmc"]

    assert mcmc_config["min_samples_for_cmc"] == 20
    assert mcmc_config["dense_mass"] is True


def test_integration_static_mode_with_overrides(temp_config_file, parser):
    """Test static mode workflow with parameter overrides."""
    # Step 1: Parse CLI for static mode
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--static-mode",
            "--initial-d0",
            "3000.0",
            "--initial-alpha",
            "-0.5",
            "--initial-d-offset",
            "50.0",
        ]
    )

    # Step 2: Load and apply
    config = ConfigManager(str(temp_config_file))
    _apply_cli_overrides(config, args)

    # Step 3: Verify analysis mode changed to static
    assert config.config["analysis_mode"] == "static_isotropic"

    # Step 4: Verify parameter overrides
    param_names = config.config["initial_parameters"]["parameter_names"]
    param_values = config.config["initial_parameters"]["values"]

    # Should have only 3 static parameters after override
    assert "D0" in param_names
    assert "alpha" in param_names
    assert "D_offset" in param_names

    d0_idx = param_names.index("D0")
    alpha_idx = param_names.index("alpha")
    d_offset_idx = param_names.index("D_offset")

    assert param_values[d0_idx] == 3000.0
    assert param_values[alpha_idx] == -0.5
    assert param_values[d_offset_idx] == 50.0


def test_integration_partial_override_preserves_config(temp_config_file, parser):
    """Test that partial override preserves non-overridden config values."""
    # Step 1: Override only 2 out of 7 parameters
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--initial-d0",
            "1200.0",
            "--initial-beta",
            "0.35",
        ]
    )

    # Step 2: Load and apply
    config = ConfigManager(str(temp_config_file))
    _apply_cli_overrides(config, args)

    # Step 3: Verify overridden parameters
    param_names = config.config["initial_parameters"]["parameter_names"]
    param_values = config.config["initial_parameters"]["values"]

    d0_idx = param_names.index("D0")
    beta_idx = param_names.index("beta")

    assert param_values[d0_idx] == 1200.0  # Overridden
    assert param_values[beta_idx] == 0.35  # Overridden

    # Step 4: Verify non-overridden parameters preserved
    alpha_idx = param_names.index("alpha")
    d_offset_idx = param_names.index("D_offset")
    gamma_idx = param_names.index("gamma_dot_t0")

    assert param_values[alpha_idx] == 0.5  # Original config value
    assert param_values[d_offset_idx] == 10.0  # Original config value
    assert param_values[gamma_idx] == 0.001  # Original config value


def test_integration_override_with_parameter_space(temp_config_file, parser):
    """Test that overrides integrate correctly with ParameterSpace."""
    from homodyne.core.fitting import ParameterSpace
    from homodyne.config.parameter_manager import ParameterManager

    # Step 1: Parse CLI with overrides
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--initial-d0",
            "2500.0",
            "--initial-alpha",
            "0.8",
        ]
    )

    # Step 2: Load and apply overrides
    config = ConfigManager(str(temp_config_file))
    _apply_cli_overrides(config, args)

    # Step 3: Create ParameterSpace with config_manager for bound override support
    param_space = ParameterSpace()
    param_space.config_manager = config

    # Step 4: Get bounds using ParameterManager
    param_manager = ParameterManager(config.config, config.config["analysis_mode"])
    bounds_dict = param_manager.get_parameter_bounds(["D0", "alpha"])

    # Verify bounds are loaded correctly
    assert bounds_dict[0]["name"] == "D0"
    assert bounds_dict[0]["min"] == 1.0
    assert bounds_dict[0]["max"] == 1e6
    assert bounds_dict[1]["name"] == "alpha"
    assert bounds_dict[1]["min"] == -2.0
    assert bounds_dict[1]["max"] == 2.0

    # Step 5: Verify that get_initial_parameters() returns overridden values
    initial_params = config.get_initial_parameters()

    assert initial_params["D0"] == 2500.0
    assert initial_params["alpha"] == 0.8

    # Step 6: Verify integration - initial values should be within bounds
    assert bounds_dict[0]["min"] <= initial_params["D0"] <= bounds_dict[0]["max"]
    assert bounds_dict[1]["min"] <= initial_params["alpha"] <= bounds_dict[1]["max"]


# ============================================================================
# Integration Test: Realistic User Workflows
# ============================================================================


def test_realistic_workflow_nlsq_to_mcmc_with_overrides(temp_config_file, parser):
    """Test realistic workflow: NLSQ → manual copy → MCMC with CLI overrides.

    User Story:
    1. User runs NLSQ with default config
    2. User gets results: D0=1234.5, alpha=0.567
    3. User wants to run MCMC but override D0 for exploration
    4. User runs: homodyne --config config.yaml --method mcmc --initial-d0 1500.0
    5. System should use: D0=1500.0 (CLI), alpha=0.567 (would be from config)
    """
    # Simulate: User has updated config with NLSQ results but wants to explore
    # different D0 value via CLI

    # Step 1: Parse CLI for MCMC with D0 override
    args = parser.parse_args(
        [
            "--config",
            str(temp_config_file),
            "--method",
            "mcmc",
            "--initial-d0",
            "1500.0",  # Explore different value
            "--min-samples-cmc",
            "20",  # Also adjust CMC threshold
        ]
    )

    # Step 2: Load config and apply overrides
    config = ConfigManager(str(temp_config_file))
    _apply_cli_overrides(config, args)

    # Step 3: Verify exploration parameters
    initial_params = config.get_initial_parameters()

    assert initial_params["D0"] == 1500.0  # CLI override
    assert initial_params["alpha"] == 0.5  # From config (NLSQ result in real scenario)

    # Step 4: Verify MCMC configuration
    mcmc_config = config.config["optimization"]["mcmc"]

    assert mcmc_config["min_samples_for_cmc"] == 20  # CLI override
    assert config.config["optimization"]["method"] == "mcmc"

    # This workflow demonstrates:
    # - User can quickly explore parameter space without editing config
    # - CLI overrides enable iterative analysis
    # - Priority works correctly: CLI > config > defaults
