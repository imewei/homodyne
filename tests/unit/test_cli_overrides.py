"""Unit Tests for CLI Parameter Override Mechanism

Tests the CLI override functionality added in v2.1 for parameter values
and MCMC thresholds. Verifies:

1. CLI arguments are correctly defined
2. Override priority: CLI args > config file > package defaults
3. Validation rejects invalid override values
4. Clear logging when overrides are applied
5. Integration with ConfigManager and ParameterManager

Test Coverage:
- test_cli_args_* (5 tests): Argument parsing and defaults
- test_override_priority_* (4 tests): Override precedence
- test_validation_* (5 tests): Input validation
- test_parameter_overrides_* (6 tests): Initial parameter overrides
- test_threshold_overrides_* (3 tests): MCMC threshold overrides
- test_logging_* (2 tests): Override logging

Total: 25+ tests
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from homodyne.cli.args_parser import create_parser, validate_args
from homodyne.cli.commands import _apply_cli_overrides
from homodyne.config.manager import ConfigManager


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def parser():
    """Create argument parser for testing."""
    return create_parser()


@pytest.fixture
def mock_config():
    """Create mock ConfigManager with default configuration."""
    config_dict = {
        "analysis_mode": "laminar_flow",
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
            "values": [1000.0, 0.5, 10.0],
        },
        "optimization": {
            "mcmc": {
                "min_samples_for_cmc": 15,
                "memory_threshold_pct": 0.30,
                "dense_mass": False,
            }
        },
    }
    mock = MagicMock(spec=ConfigManager)
    mock.config = config_dict
    return mock


# ============================================================================
# Test CLI Argument Parsing
# ============================================================================


def test_cli_args_parameter_overrides_exist(parser):
    """Test that all parameter override arguments exist."""
    # Parse empty args to get defaults
    args = parser.parse_args([])

    # Check that all override attributes exist
    assert hasattr(args, "initial_d0")
    assert hasattr(args, "initial_alpha")
    assert hasattr(args, "initial_d_offset")
    assert hasattr(args, "initial_gamma_dot_t0")
    assert hasattr(args, "initial_beta")
    assert hasattr(args, "initial_gamma_dot_offset")
    assert hasattr(args, "initial_phi0")
    assert hasattr(args, "min_samples_cmc")
    assert hasattr(args, "memory_threshold_pct")
    assert hasattr(args, "dense_mass_matrix")


def test_cli_args_defaults_are_none(parser):
    """Test that parameter override defaults are None."""
    args = parser.parse_args([])

    # All override arguments should default to None
    assert args.initial_d0 is None
    assert args.initial_alpha is None
    assert args.initial_d_offset is None
    assert args.initial_gamma_dot_t0 is None
    assert args.initial_beta is None
    assert args.initial_gamma_dot_offset is None
    assert args.initial_phi0 is None
    assert args.min_samples_cmc is None
    assert args.memory_threshold_pct is None


def test_cli_args_dense_mass_matrix_default_false(parser):
    """Test that dense_mass_matrix defaults to False."""
    args = parser.parse_args([])
    assert args.dense_mass_matrix is False


def test_cli_args_parse_parameter_overrides(parser):
    """Test parsing parameter override values from CLI."""
    args = parser.parse_args(
        [
            "--initial-d0",
            "1500.0",
            "--initial-alpha",
            "0.75",
            "--initial-d-offset",
            "20.0",
        ]
    )

    assert args.initial_d0 == 1500.0
    assert args.initial_alpha == 0.75
    assert args.initial_d_offset == 20.0


def test_cli_args_parse_threshold_overrides(parser):
    """Test parsing MCMC threshold overrides from CLI."""
    args = parser.parse_args(
        [
            "--min-samples-cmc",
            "20",
            "--memory-threshold-pct",
            "0.40",
        ]
    )

    assert args.min_samples_cmc == 20
    assert args.memory_threshold_pct == 0.40


# ============================================================================
# Test Validation
# ============================================================================


def test_validation_rejects_negative_d0(parser):
    """Test that validation rejects negative D0."""
    args = parser.parse_args(["--initial-d0", "-100.0"])
    assert not validate_args(args)


def test_validation_rejects_negative_min_samples(parser):
    """Test that validation rejects negative min_samples_for_cmc."""
    args = parser.parse_args(["--min-samples-cmc", "-5"])
    assert not validate_args(args)


def test_validation_rejects_invalid_memory_threshold_low(parser):
    """Test that validation rejects memory_threshold_pct < 0."""
    args = parser.parse_args(["--memory-threshold-pct", "-0.1"])
    assert not validate_args(args)


def test_validation_rejects_invalid_memory_threshold_high(parser):
    """Test that validation rejects memory_threshold_pct > 1."""
    args = parser.parse_args(["--memory-threshold-pct", "1.5"])
    assert not validate_args(args)


def test_validation_accepts_valid_overrides(parser):
    """Test that validation accepts all valid override values."""
    args = parser.parse_args(
        [
            "--initial-d0",
            "1500.0",
            "--initial-alpha",
            "0.75",
            "--initial-d-offset",
            "20.0",
            "--min-samples-cmc",
            "25",
            "--memory-threshold-pct",
            "0.40",
        ]
    )
    assert validate_args(args)


# ============================================================================
# Test Override Priority
# ============================================================================


def test_override_priority_cli_beats_config(mock_config):
    """Test that CLI overrides take precedence over config values."""
    # Mock args with CLI override
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=2000.0,  # Override from config 1000.0
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    # Apply overrides
    _apply_cli_overrides(mock_config, args)

    # Check that CLI override was applied
    param_values = mock_config.config["initial_parameters"]["values"]
    param_names = mock_config.config["initial_parameters"]["parameter_names"]

    # Find D0 index
    d0_idx = param_names.index("D0")
    assert param_values[d0_idx] == 2000.0  # CLI value, not config 1000.0


def test_override_priority_config_beats_default(mock_config):
    """Test that config values are used when no CLI override provided."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,  # No override
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    # Apply overrides
    _apply_cli_overrides(mock_config, args)

    # Check that config values are preserved
    param_values = mock_config.config["initial_parameters"]["values"]
    assert param_values == [1000.0, 0.5, 10.0]  # Original config values


def test_override_priority_threshold_cli_beats_config(mock_config):
    """Test that CLI threshold overrides beat config values."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="mcmc",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=25,  # Override from config 15
        memory_threshold_pct=0.40,  # Override from config 0.30
        dense_mass_matrix=False,
    )

    # Apply overrides
    _apply_cli_overrides(mock_config, args)

    # Check that CLI overrides were applied
    assert mock_config.config["optimization"]["mcmc"]["min_samples_for_cmc"] == 25
    assert mock_config.config["optimization"]["mcmc"]["memory_threshold_pct"] == 0.40


def test_override_priority_multiple_parameters(mock_config):
    """Test that multiple CLI parameter overrides work correctly."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=1500.0,
        initial_alpha=0.75,
        initial_d_offset=25.0,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    # Apply overrides
    _apply_cli_overrides(mock_config, args)

    # Check that all overrides were applied
    param_values = mock_config.config["initial_parameters"]["values"]
    param_names = mock_config.config["initial_parameters"]["parameter_names"]

    assert param_names == ["D0", "alpha", "D_offset"]
    assert param_values == [1500.0, 0.75, 25.0]


# ============================================================================
# Test Initial Parameter Overrides
# ============================================================================


def test_parameter_override_static_mode_d0(mock_config):
    """Test overriding D0 in static mode."""
    mock_config.config["analysis_mode"] = "static_isotropic"

    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=3000.0,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    param_names = mock_config.config["initial_parameters"]["parameter_names"]
    param_values = mock_config.config["initial_parameters"]["values"]

    d0_idx = param_names.index("D0")
    assert param_values[d0_idx] == 3000.0


def test_parameter_override_laminar_flow_gamma_dot(mock_config):
    """Test overriding gamma_dot_t0 in laminar flow mode."""
    # Add laminar flow parameters to config
    mock_config.config["initial_parameters"]["parameter_names"] = [
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]
    mock_config.config["initial_parameters"]["values"] = [
        1000.0,
        0.5,
        10.0,
        0.001,
        0.2,
        0.0001,
        0.0,
    ]

    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=0.005,  # Override
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    param_names = mock_config.config["initial_parameters"]["parameter_names"]
    param_values = mock_config.config["initial_parameters"]["values"]

    gamma_idx = param_names.index("gamma_dot_t0")
    assert param_values[gamma_idx] == 0.005


def test_parameter_override_creates_initial_parameters_section(mock_config):
    """Test that override creates initial_parameters section if missing."""
    # Remove initial_parameters section
    del mock_config.config["initial_parameters"]

    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=1500.0,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    # Check that initial_parameters was created
    assert "initial_parameters" in mock_config.config
    assert "parameter_names" in mock_config.config["initial_parameters"]
    assert "values" in mock_config.config["initial_parameters"]

    # Check D0 was set
    param_names = mock_config.config["initial_parameters"]["parameter_names"]
    param_values = mock_config.config["initial_parameters"]["values"]
    assert "D0" in param_names
    d0_idx = param_names.index("D0")
    assert param_values[d0_idx] == 1500.0


def test_parameter_override_handles_null_values(mock_config):
    """Test that override handles null values in config."""
    # Set values to None (null in YAML)
    mock_config.config["initial_parameters"]["values"] = None

    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=2000.0,
        initial_alpha=0.8,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    # Check that parameters were set
    param_names = mock_config.config["initial_parameters"]["parameter_names"]
    param_values = mock_config.config["initial_parameters"]["values"]

    assert "D0" in param_names
    assert "alpha" in param_names
    d0_idx = param_names.index("D0")
    alpha_idx = param_names.index("alpha")
    assert param_values[d0_idx] == 2000.0
    assert param_values[alpha_idx] == 0.8


def test_parameter_override_partial_override(mock_config):
    """Test that partial override preserves non-overridden values."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,  # Don't override
        initial_alpha=0.9,  # Override
        initial_d_offset=None,  # Don't override
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    param_names = mock_config.config["initial_parameters"]["parameter_names"]
    param_values = mock_config.config["initial_parameters"]["values"]

    # Check all three parameters
    assert param_names == ["D0", "alpha", "D_offset"]
    assert param_values[0] == 1000.0  # D0 unchanged
    assert param_values[1] == 0.9  # alpha overridden
    assert param_values[2] == 10.0  # D_offset unchanged


def test_parameter_override_all_seven_parameters(mock_config):
    """Test overriding all 7 laminar flow parameters."""
    # Set up laminar flow config
    mock_config.config["initial_parameters"]["parameter_names"] = [
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_t0",
        "beta",
        "gamma_dot_t_offset",
        "phi0",
    ]
    mock_config.config["initial_parameters"]["values"] = [
        1000.0,
        0.5,
        10.0,
        0.001,
        0.2,
        0.0001,
        0.0,
    ]

    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=1500.0,
        initial_alpha=0.75,
        initial_d_offset=20.0,
        initial_gamma_dot_t0=0.005,
        initial_beta=0.3,
        initial_gamma_dot_offset=0.0005,
        initial_phi0=0.5,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    param_values = mock_config.config["initial_parameters"]["values"]

    # All values should be overridden
    assert param_values == [1500.0, 0.75, 20.0, 0.005, 0.3, 0.0005, 0.5]


# ============================================================================
# Test MCMC Threshold Overrides
# ============================================================================


def test_threshold_override_min_samples_cmc(mock_config):
    """Test overriding min_samples_for_cmc threshold."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="mcmc",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=30,  # Override from 15
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    assert mock_config.config["optimization"]["mcmc"]["min_samples_for_cmc"] == 30


def test_threshold_override_memory_threshold(mock_config):
    """Test overriding memory_threshold_pct."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="mcmc",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=0.50,  # Override from 0.30
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    assert mock_config.config["optimization"]["mcmc"]["memory_threshold_pct"] == 0.50


def test_threshold_override_dense_mass_matrix(mock_config):
    """Test overriding dense_mass flag."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="mcmc",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=True,  # Override from False
    )

    _apply_cli_overrides(mock_config, args)

    assert mock_config.config["optimization"]["mcmc"]["dense_mass"] is True


# ============================================================================
# Test Logging
# ============================================================================


@patch("homodyne.cli.commands.logger")
def test_logging_parameter_override(mock_logger, mock_config):
    """Test that parameter overrides are logged clearly."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="nlsq",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=2000.0,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=None,
        memory_threshold_pct=None,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    # Check that logger.info was called with override message
    mock_logger.info.assert_any_call(
        "Overriding config D0=1000 with CLI value D0=2000"
    )


@patch("homodyne.cli.commands.logger")
def test_logging_threshold_override(mock_logger, mock_config):
    """Test that threshold overrides are logged clearly."""
    args = argparse.Namespace(
        data_file=None,
        static_mode=False,
        laminar_flow=False,
        method="mcmc",
        n_samples=None,
        n_warmup=None,
        n_chains=None,
        force_cpu=False,
        gpu_memory_fraction=0.9,
        initial_d0=None,
        initial_alpha=None,
        initial_d_offset=None,
        initial_gamma_dot_t0=None,
        initial_beta=None,
        initial_gamma_dot_offset=None,
        initial_phi0=None,
        min_samples_cmc=25,
        memory_threshold_pct=0.40,
        dense_mass_matrix=False,
    )

    _apply_cli_overrides(mock_config, args)

    # Check that logger.info was called for both threshold overrides
    mock_logger.info.assert_any_call(
        "Overriding config min_samples_for_cmc=15 with CLI value min_samples_for_cmc=25"
    )
    mock_logger.info.assert_any_call(
        "Overriding config memory_threshold_pct=0.30 with CLI value memory_threshold_pct=0.40"
    )
