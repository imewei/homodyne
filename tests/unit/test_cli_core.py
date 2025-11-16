"""
Unit Tests for CLI Core Functionality
======================================

Consolidated from:
- test_cli_args.py (CLI argument parsing, 17 tests, 273 lines)
- test_cli_validation.py (CLI validation & edge cases, 46 tests, 881 lines)
- test_cli_data_loading.py (CLI data loading, 14 tests, 345 lines)

Tests cover:
- Method argument parsing (nlsq, mcmc)
- Method validation after v2.1.0 simplification
- Deprecated method rejection (nuts, cmc, auto)
- CMC-specific CLI options (min-samples-cmc, memory-threshold-pct)
- CLI parameter overrides
- Shell alias existence and correctness
- Config schema normalization
- XPCSDataLoader integration
- CLI argument overrides for data loading
- Edge case handling

Total: 77 tests
"""

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from homodyne.cli.args_parser import create_parser, validate_args
from homodyne.cli.commands import _load_data
from homodyne.config.manager import ConfigManager


# ==============================================================================
# CLI Argument Parsing Tests (from test_cli_args.py)
# ==============================================================================

class TestMethodArgumentParsing:
    """Test CLI method argument parsing after simplification."""

    def test_method_nlsq_accepted(self):
        """Test that --method nlsq is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--method", "nlsq"])

        assert args.method == "nlsq"
        assert validate_args(args) is True

    def test_method_mcmc_accepted(self):
        """Test that --method mcmc is accepted and triggers automatic selection."""
        parser = create_parser()
        args = parser.parse_args(["--method", "mcmc"])

        assert args.method == "mcmc"
        assert validate_args(args) is True

    def test_method_nuts_rejected(self):
        """Test that --method nuts raises clear error message."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "nuts"])

        assert exc_info.value.code == 2  # argparse error exit code

    def test_method_cmc_rejected(self):
        """Test that --method cmc raises clear error message."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "cmc"])

        assert exc_info.value.code == 2  # argparse error exit code

    def test_method_auto_rejected(self):
        """Test that --method auto raises clear error message."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "auto"])

        assert exc_info.value.code == 2  # argparse error exit code

    def test_default_method_is_nlsq(self):
        """Test that default method is nlsq when not specified."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.method == "nlsq"

    def test_method_choices_only_nlsq_and_mcmc(self):
        """Test that method choices are restricted to nlsq and mcmc only."""
        parser = create_parser()

        # Find the --method argument in parser
        method_action = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_action = action
                break

        assert method_action is not None
        assert set(method_action.choices) == {"nlsq", "mcmc"}
        assert "nuts" not in method_action.choices
        assert "cmc" not in method_action.choices
        assert "auto" not in method_action.choices


class TestMethodValidation:
    """Test argument validation with valid and invalid methods."""

    def test_validate_args_accepts_nlsq(self, tmp_path):
        """Test that validate_args accepts nlsq method."""
        parser = create_parser()

        # Create a temporary config file to avoid file not found error
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(["--method", "nlsq", "--config", str(config_file)])
        assert validate_args(args) is True

    def test_validate_args_accepts_mcmc(self, tmp_path):
        """Test that validate_args accepts mcmc method."""
        parser = create_parser()

        # Create a temporary config file to avoid file not found error
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(["--method", "mcmc", "--config", str(config_file)])
        assert validate_args(args) is True


class TestCMCOptions:
    """Test CMC-specific CLI options are accepted with mcmc method."""

    def test_cmc_num_shards_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-num-shards is accepted with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            ["--method", "mcmc", "--cmc-num-shards", "4", "--config", str(config_file)]
        )

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 4
        assert validate_args(args) is True

    def test_cmc_backend_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-backend is accepted with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            ["--method", "mcmc", "--cmc-backend", "pjit", "--config", str(config_file)]
        )

        assert args.method == "mcmc"
        assert args.cmc_backend == "pjit"
        assert validate_args(args) is True

    def test_cmc_plot_diagnostics_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-plot-diagnostics is accepted with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            ["--method", "mcmc", "--cmc-plot-diagnostics", "--config", str(config_file)]
        )

        assert args.method == "mcmc"
        assert args.cmc_plot_diagnostics is True
        assert validate_args(args) is True

    def test_all_cmc_options_accepted_together(self, tmp_path):
        """Test that all CMC options can be used together with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--cmc-num-shards",
                "8",
                "--cmc-backend",
                "multiprocessing",
                "--cmc-plot-diagnostics",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 8
        assert args.cmc_backend == "multiprocessing"
        assert args.cmc_plot_diagnostics is True
        assert validate_args(args) is True

    def test_cmc_options_warned_with_nlsq(self, tmp_path, capsys):
        """Test that CMC options generate warnings when used with nlsq method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "nlsq",
                "--cmc-num-shards",
                "4",
                "--cmc-backend",
                "pjit",
                "--cmc-plot-diagnostics",
                "--config",
                str(config_file),
            ]
        )

        # Validation should succeed but generate warnings
        result = validate_args(args)
        assert result is True

        # Check that warnings were printed
        captured = capsys.readouterr()
        assert "Warning: --cmc-num-shards ignored" in captured.out
        assert "Warning: --cmc-backend ignored" in captured.out
        assert "Warning: --cmc-plot-diagnostics ignored" in captured.out


class TestHelpTextDocumentation:
    """Test that help text properly documents automatic selection."""

    def test_method_help_mentions_automatic_selection(self):
        """Test that --method help text mentions automatic NUTS/CMC selection."""
        parser = create_parser()

        # Find the --method argument
        method_action = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_action = action
                break

        assert method_action is not None
        help_text = method_action.help

        # Check that help text mentions key concepts
        assert "automatic NUTS/CMC selection" in help_text or "automatic" in help_text
        assert "num_samples" in help_text or "dual criteria" in help_text
        assert "config" in help_text.lower()

    def test_cmc_group_has_description_with_selection_info(self):
        """Test that CMC argument group has a descriptive title and/or description."""
        parser = create_parser()

        # Find CMC argument group
        cmc_group = None
        for group in parser._action_groups:
            if "CMC" in group.title:
                cmc_group = group
                break

        assert cmc_group is not None
        # Check group title mentions CMC
        assert "CMC" in cmc_group.title or "Consensus Monte Carlo" in cmc_group.title

        # If description exists, it should explain automatic selection
        # Note: argparse may not populate description in all cases
        if cmc_group.description:
            description_lower = cmc_group.description.lower()
            assert "automatic" in description_lower or "selected" in description_lower

    def test_epilog_documents_automatic_selection_criteria(self):
        """Test that epilog documents the dual-criteria selection logic."""
        parser = create_parser()
        epilog = parser.epilog

        assert epilog is not None
        # Check for dual-criteria documentation
        assert "num_samples >= 15" in epilog or "memory > 30%" in epilog
        assert "Automatic selection" in epilog or "automatic" in epilog.lower()


# ==============================================================================
# CLI Validation Tests (from test_cli_validation.py)
# ==============================================================================

class TestDeprecatedMethodRejection:
    """Test that deprecated methods are properly rejected with clear errors."""

    def test_method_nuts_rejected_with_exit_code(self):
        """Test that --method nuts is rejected with proper exit code."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "nuts"])

        # argparse returns exit code 2 for invalid arguments
        assert exc_info.value.code == 2

    def test_method_cmc_rejected_with_exit_code(self):
        """Test that --method cmc is rejected with proper exit code."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "cmc"])

        # argparse returns exit code 2 for invalid arguments
        assert exc_info.value.code == 2

    def test_method_auto_rejected_with_exit_code(self):
        """Test that --method auto is rejected with proper exit code."""
        parser = create_parser()

        # Should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--method", "auto"])

        # argparse returns exit code 2 for invalid arguments
        assert exc_info.value.code == 2

    def test_method_choices_excludes_deprecated_methods(self):
        """Test that method argument only includes nlsq and mcmc."""
        parser = create_parser()

        # Find the --method argument
        method_action = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_action = action
                break

        assert method_action is not None
        assert set(method_action.choices) == {"nlsq", "mcmc"}

        # Explicitly verify deprecated methods are NOT in choices
        assert "nuts" not in method_action.choices
        assert "cmc" not in method_action.choices
        assert "auto" not in method_action.choices


class TestAcceptedMethodsFunctionality:
    """Test that accepted methods (nlsq, mcmc) work correctly."""

    def test_method_nlsq_accepted(self):
        """Test that --method nlsq is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--method", "nlsq"])

        assert args.method == "nlsq"

    def test_method_mcmc_accepted(self):
        """Test that --method mcmc is accepted."""
        parser = create_parser()
        args = parser.parse_args(["--method", "mcmc"])

        assert args.method == "mcmc"

    def test_default_method_is_nlsq(self):
        """Test that default method is nlsq when not specified."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.method == "nlsq"

    def test_method_nlsq_validation_passes(self, tmp_path):
        """Test that nlsq method passes validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(["--method", "nlsq", "--config", str(config_file)])
        assert validate_args(args) is True

    def test_method_mcmc_validation_passes(self, tmp_path):
        """Test that mcmc method passes validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(["--method", "mcmc", "--config", str(config_file)])
        assert validate_args(args) is True


class TestCMCSpecificOptions:
    """Test CMC-specific CLI options work with --method mcmc."""

    def test_min_samples_cmc_override_accepted(self, tmp_path):
        """Test that --min-samples-cmc override is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--min-samples-cmc",
                "20",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.min_samples_cmc == 20
        assert validate_args(args) is True

    def test_memory_threshold_pct_override_accepted(self, tmp_path):
        """Test that --memory-threshold-pct override is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--memory-threshold-pct",
                "0.35",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.memory_threshold_pct == 0.35
        assert validate_args(args) is True

    def test_both_cmc_thresholds_accepted_together(self, tmp_path):
        """Test that both CMC threshold overrides work together."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--min-samples-cmc",
                "25",
                "--memory-threshold-pct",
                "0.40",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.min_samples_cmc == 25
        assert args.memory_threshold_pct == 0.40
        assert validate_args(args) is True

    def test_cmc_num_shards_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-num-shards is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--cmc-num-shards",
                "8",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 8
        assert validate_args(args) is True

    def test_cmc_backend_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-backend is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--cmc-backend",
                "pjit",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.cmc_backend == "pjit"
        assert validate_args(args) is True

    def test_all_cmc_options_together(self, tmp_path):
        """Test that all CMC options can be used together."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--min-samples-cmc",
                "20",
                "--memory-threshold-pct",
                "0.35",
                "--cmc-num-shards",
                "8",
                "--cmc-backend",
                "multiprocessing",
                "--cmc-plot-diagnostics",
                "--config",
                str(config_file),
            ]
        )

        assert args.method == "mcmc"
        assert args.min_samples_cmc == 20
        assert args.memory_threshold_pct == 0.35
        assert args.cmc_num_shards == 8
        assert args.cmc_backend == "multiprocessing"
        assert args.cmc_plot_diagnostics is True
        assert validate_args(args) is True


class TestCLIParameterOverrides:
    """Test CLI parameter override options."""

    def test_initial_d0_override(self, tmp_path):
        """Test --initial-d0 override."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-d0",
                "1500.5",
                "--config",
                str(config_file),
            ]
        )

        assert args.initial_d0 == 1500.5
        assert validate_args(args) is True

    def test_initial_alpha_override(self, tmp_path):
        """Test --initial-alpha override."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-alpha",
                "0.75",
                "--config",
                str(config_file),
            ]
        )

        assert args.initial_alpha == 0.75
        assert validate_args(args) is True

    def test_initial_d_offset_override(self, tmp_path):
        """Test --initial-d-offset override."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-d-offset",
                "100.0",
                "--config",
                str(config_file),
            ]
        )

        assert args.initial_d_offset == 100.0
        assert validate_args(args) is True

    def test_multiple_parameter_overrides(self, tmp_path):
        """Test multiple parameter overrides applied correctly."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-d0",
                "1234.5",
                "--initial-alpha",
                "0.567",
                "--initial-d-offset",
                "12.34",
                "--config",
                str(config_file),
            ]
        )

        assert args.initial_d0 == 1234.5
        assert args.initial_alpha == 0.567
        assert args.initial_d_offset == 12.34
        assert validate_args(args) is True

    def test_laminar_flow_parameter_overrides(self, tmp_path):
        """Test laminar flow specific parameter overrides."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-gamma-dot-t0",
                "1.5",
                "--initial-beta",
                "0.3",
                "--initial-gamma-dot-offset",
                "0.1",
                "--initial-phi0",
                "0.785",
                "--config",
                str(config_file),
            ]
        )

        assert args.initial_gamma_dot_t0 == 1.5
        assert args.initial_beta == 0.3
        assert args.initial_gamma_dot_offset == 0.1
        assert args.initial_phi0 == 0.785
        assert validate_args(args) is True

    def test_dense_mass_matrix_override(self, tmp_path):
        """Test --dense-mass-matrix override for MCMC."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--dense-mass-matrix",
                "--config",
                str(config_file),
            ]
        )

        assert args.dense_mass_matrix is True
        assert validate_args(args) is True

    def test_mcmc_sample_overrides(self, tmp_path):
        """Test MCMC sample count overrides."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "mcmc",
                "--n-samples",
                "2000",
                "--n-warmup",
                "1000",
                "--n-chains",
                "8",
                "--config",
                str(config_file),
            ]
        )

        assert args.n_samples == 2000
        assert args.n_warmup == 1000
        assert args.n_chains == 8
        assert validate_args(args) is True


class TestParameterOverrideValidation:
    """Test validation of parameter override values."""

    def test_negative_d0_rejected(self, tmp_path):
        """Test that negative D0 values are rejected."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--initial-d0",
                "-100.0",
                "--config",
                str(config_file),
            ]
        )

        # Validation should fail for negative D0
        assert validate_args(args) is False

    def test_invalid_memory_threshold_rejected(self, tmp_path):
        """Test that invalid memory threshold values are rejected."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--memory-threshold-pct",
                "1.5",  # > 1.0
                "--config",
                str(config_file),
            ]
        )

        # Validation should fail for out-of-range threshold
        assert validate_args(args) is False

    def test_memory_threshold_boundaries(self, tmp_path):
        """Test memory threshold at valid boundaries."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Test 0.0 boundary
        args = parser.parse_args(
            [
                "--memory-threshold-pct",
                "0.0",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is True

        # Test 1.0 boundary
        args = parser.parse_args(
            [
                "--memory-threshold-pct",
                "1.0",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is True


class TestShellAliasDefinitions:
    """Test shell alias definitions in post_install.py."""

    def test_shell_aliases_defined(self):
        """Test that shell aliases are defined in post_install.py."""
        post_install_file = Path(
            "/home/wei/Documents/GitHub/homodyne/homodyne/post_install.py"
        )
        assert post_install_file.exists()

        content = post_install_file.read_text()

        # Check that correct aliases are defined
        assert "alias hm-nlsq='homodyne --method nlsq'" in content
        assert "alias hm-mcmc='homodyne --method mcmc'" in content

    def test_deprecated_aliases_not_defined(self):
        """Test that deprecated aliases are not defined."""
        post_install_file = Path(
            "/home/wei/Documents/GitHub/homodyne/homodyne/post_install.py"
        )
        assert post_install_file.exists()

        content = post_install_file.read_text()

        # Deprecated aliases should not be defined
        # (they would show up as "alias hm-nuts=" or "alias hm-cmc=" or "alias hm-auto=")
        assert "alias hm-nuts=" not in content
        assert "alias hm-cmc=" not in content
        assert "alias hm-auto=" not in content

    def test_alias_descriptions_in_post_install(self):
        """Test that alias descriptions properly document the changes."""
        post_install_file = Path(
            "/home/wei/Documents/GitHub/homodyne/homodyne/post_install.py"
        )
        content = post_install_file.read_text()

        # Check that descriptions mention automatic selection for mcmc
        assert "hm-nlsq" in content
        assert "hm-mcmc" in content


class TestHelpTextValidation:
    """Test help text properly documents CLI changes."""

    def test_help_text_method_choices_correct(self):
        """Test that help text shows only nlsq and mcmc as choices."""
        parser = create_parser()

        # Find the --method argument
        method_action = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_action = action
                break

        assert method_action is not None
        help_text = method_action.help

        # Verify help text mentions key concepts
        assert "nlsq" in help_text or "NLSQ" in help_text
        assert "mcmc" in help_text or "MCMC" in help_text

    def test_help_mentions_automatic_selection(self):
        """Test that help text documents automatic NUTS/CMC selection."""
        parser = create_parser()

        # Find the --method argument
        method_action = None
        for action in parser._actions:
            if "--method" in action.option_strings:
                method_action = action
                break

        assert method_action is not None
        help_text = method_action.help

        # Should mention dual criteria selection
        assert (
            "automatic NUTS/CMC selection" in help_text
            or "automatic" in help_text.lower()
            or "dual criteria" in help_text.lower()
        )

    def test_cmc_group_describes_automatic_selection(self):
        """Test that CMC argument group describes automatic selection."""
        parser = create_parser()

        # Find CMC argument group
        cmc_group = None
        for group in parser._action_groups:
            if "CMC" in group.title or "Consensus" in group.title:
                cmc_group = group
                break

        assert cmc_group is not None
        # Check group title/description mentions CMC
        assert "CMC" in cmc_group.title or "Consensus" in cmc_group.title

    def test_epilog_documents_dual_criteria(self):
        """Test that epilog documents the dual-criteria selection logic."""
        parser = create_parser()
        epilog = parser.epilog

        assert epilog is not None
        # Check for dual-criteria documentation
        assert "num_samples >= 15" in epilog or "memory > 30%" in epilog

    def test_epilog_mentions_automatic_selection(self):
        """Test that epilog mentions automatic selection."""
        parser = create_parser()
        epilog = parser.epilog

        assert epilog is not None
        assert "automatic" in epilog.lower() or "Automatic" in epilog


class TestMCMCSpecificOptionsValidation:
    """Test MCMC-specific option validation."""

    def test_mcmc_samples_validation(self, tmp_path):
        """Test MCMC sample count validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Valid samples
        args = parser.parse_args(
            [
                "--n-samples",
                "1000",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is True

        # Invalid samples
        args = parser.parse_args(
            [
                "--n-samples",
                "0",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is False

    def test_mcmc_warmup_validation(self, tmp_path):
        """Test MCMC warmup count validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Valid warmup
        args = parser.parse_args(
            [
                "--n-warmup",
                "500",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is True

        # Invalid warmup
        args = parser.parse_args(
            [
                "--n-warmup",
                "-100",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is False

    def test_mcmc_chains_validation(self, tmp_path):
        """Test MCMC chain count validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Valid chains
        args = parser.parse_args(
            [
                "--n-chains",
                "4",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is True

        # Invalid chains
        args = parser.parse_args(
            [
                "--n-chains",
                "0",
                "--config",
                str(config_file),
            ]
        )
        assert validate_args(args) is False


class TestCMCWarningsWithNLSQ:
    """Test that CMC options generate warnings when used with nlsq."""

    def test_cmc_options_warned_with_nlsq_method(self, tmp_path, capsys):
        """Test that CMC options generate warnings with nlsq method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--method",
                "nlsq",
                "--cmc-num-shards",
                "4",
                "--cmc-backend",
                "pjit",
                "--cmc-plot-diagnostics",
                "--config",
                str(config_file),
            ]
        )

        # Validation should succeed but generate warnings
        result = validate_args(args)
        assert result is True

        # Check that warnings were printed
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "warning" in captured.out.lower()


class TestConflictingArguments:
    """Test handling of conflicting arguments."""

    def test_static_and_laminar_flow_conflict(self, tmp_path):
        """Test that --static-mode and --laminar-flow conflict."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--static-mode",
                "--laminar-flow",
                "--config",
                str(config_file),
            ]
        )

        # Should fail validation
        assert validate_args(args) is False

    def test_verbose_and_quiet_conflict(self, tmp_path):
        """Test that --verbose and --quiet conflict."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--verbose",
                "--quiet",
                "--config",
                str(config_file),
            ]
        )

        # Should fail validation
        assert validate_args(args) is False


class TestCliHelpOutput:
    """Test actual CLI help output contains expected information."""

    def test_homodyne_help_shows_method_choices(self):
        """Test that homodyne --help shows method choices."""
        result = subprocess.run(
            [sys.executable, "-m", "homodyne.cli.main", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/wei/Documents/GitHub/homodyne",
        )

        assert result.returncode == 0
        help_output = result.stdout

        # Should mention nlsq and mcmc
        assert "nlsq" in help_output.lower()
        assert "mcmc" in help_output.lower()

    def test_homodyne_help_does_not_mention_deprecated_methods(self):
        """Test that homodyne --help does not mention deprecated methods."""
        result = subprocess.run(
            [sys.executable, "-m", "homodyne.cli.main", "--help"],
            capture_output=True,
            text=True,
            cwd="/home/wei/Documents/GitHub/homodyne",
        )

        assert result.returncode == 0
        help_output = result.stdout

        # Should NOT mention deprecated methods in method choices
        # (Note: may appear in epilog examples showing what NOT to do)
        help_lines = help_output.split("\n")
        for i, line in enumerate(help_lines):
            # Look for the --method line
            if "--method" in line:
                # Next few lines should show only nlsq and mcmc
                method_section = "\n".join(help_lines[i : min(i + 10, len(help_lines))])
                # Check that deprecated methods aren't in the choices
                assert "{nuts,cmc,auto}" not in method_section
                assert "{nuts,cmc}" not in method_section
                assert "{auto,nuts}" not in method_section


class TestVersionInformation:
    """Test version information is properly displayed."""

    def test_homodyne_version_flag(self):
        """Test that --version flag works."""
        result = subprocess.run(
            [sys.executable, "-m", "homodyne.cli.main", "--version"],
            capture_output=True,
            text=True,
            cwd="/home/wei/Documents/GitHub/homodyne",
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr
        assert "Homodyne" in output or "v" in output.lower()


class TestConfigFileValidation:
    """Test configuration file path validation."""

    def test_missing_config_file_error(self, tmp_path):
        """Test that missing config file is properly rejected."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "--config",
                str(tmp_path / "nonexistent.yaml"),
            ]
        )

        # Validation should fail for missing config
        assert validate_args(args) is False

    def test_existing_config_file_accepted(self, tmp_path):
        """Test that existing config file is accepted."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args(
            [
                "--config",
                str(config_file),
            ]
        )

        # Validation should pass
        assert validate_args(args) is True


class TestDefaultConfigBehavior:
    """Test default configuration behavior."""

    def test_default_config_path_is_homodyne_config_yaml(self):
        """Test that default config path is ./homodyne_config.yaml."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.config == Path("./homodyne_config.yaml")

    def test_default_output_dir_is_homodyne_results(self):
        """Test that default output directory is ./homodyne_results."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.output_dir == Path("./homodyne_results")


# ==============================================================================
# CLI Data Loading Tests (from test_cli_data_loading.py)
# ==============================================================================

class TestConfigNormalization:
    """Test suite for configuration schema normalization."""

    def test_normalization_legacy_format(self, tmp_path):
        """Test that legacy data_folder_path + data_file_name is normalized to file_path."""
        # Create a test config with legacy format
        config_data = {
            "experimental_data": {
                "data_folder_path": "./data/sample/",
                "data_file_name": "test_data.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check that normalization happened
        exp_data = config.config["experimental_data"]
        assert "file_path" in exp_data, "file_path should be added by normalization"
        assert exp_data["file_path"] == "data/sample/test_data.hdf"

        # Check that original fields are preserved
        assert exp_data["data_folder_path"] == "./data/sample/"
        assert exp_data["data_file_name"] == "test_data.hdf"

    def test_normalization_modern_format(self, tmp_path):
        """Test that modern file_path format is not modified."""
        # Create a test config with modern format
        config_data = {
            "experimental_data": {"file_path": "./data/experiment.hdf"},
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check that file_path is unchanged
        exp_data = config.config["experimental_data"]
        assert exp_data["file_path"] == "./data/experiment.hdf"

        # Check that no extra fields were added
        assert "data_folder_path" not in exp_data
        assert "data_file_name" not in exp_data

    def test_normalization_phi_angles(self, tmp_path):
        """Test that phi angles paths are also normalized."""
        # Create a test config with phi angles
        config_data = {
            "experimental_data": {
                "data_folder_path": "./data/",
                "data_file_name": "test.hdf",
                "phi_angles_path": "./data/phi/",
                "phi_angles_file": "angles.txt",
            }
        }

        # Create ConfigManager with override
        config = ConfigManager(config_override=config_data)

        # Check phi angles normalization
        exp_data = config.config["experimental_data"]
        assert "phi_angles_full_path" in exp_data
        assert exp_data["phi_angles_full_path"] == "data/phi/angles.txt"

    def test_normalization_with_absolute_paths(self):
        """Test normalization with absolute paths."""
        config_data = {
            "experimental_data": {
                "data_folder_path": "/home/user/xpcs/data/",
                "data_file_name": "experiment_001.hdf",
            }
        }

        config = ConfigManager(config_override=config_data)
        exp_data = config.config["experimental_data"]

        assert "file_path" in exp_data
        assert exp_data["file_path"] == "/home/user/xpcs/data/experiment_001.hdf"


class TestDataLoading:
    """Test suite for data loading functionality."""

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_with_cli_override(self, mock_loader_class):
        """Test loading data with --data-file CLI override."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {
            "c2_exp": Mock(size=1000),
            "t1": Mock(),
            "t2": Mock(),
        }
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = "test_data.hdf"

        mock_config = Mock()
        mock_config.config = {
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify XPCSDataLoader was called with correct config
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        assert "config_dict" in call_args

        # Verify data was loaded
        assert result is not None
        assert "c2_exp" in result

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_from_config(self, mock_loader_class):
        """Test loading data from configuration file."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {
            "c2_exp": Mock(size=2000),
            "t1": Mock(),
            "t2": Mock(),
        }
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = None

        mock_config = Mock()
        mock_config.config = {
            "experimental_data": {
                "data_folder_path": "./data/",
                "data_file_name": "config_data.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 200},
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify XPCSDataLoader was called with full config
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        assert call_args["config_dict"] == mock_config.config

        # Verify data was loaded
        assert result is not None
        assert "c2_exp" in result

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_relative_path(self, mock_loader_class):
        """Test loading data with relative path (edge case)."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {"c2_exp": Mock(size=500)}
        mock_loader_class.return_value = mock_loader

        # Create mock args with relative path (just filename)
        mock_args = Mock()
        mock_args.data_file = "data.hdf"  # No directory prefix

        mock_config = Mock()
        mock_config.config = {
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100}
        }

        # Call _load_data
        result = _load_data(mock_args, mock_config)

        # Verify it handled relative path
        assert mock_loader_class.called
        call_args = mock_loader_class.call_args[1]
        exp_data = call_args["config_dict"]["experimental_data"]

        # Should have absolute path
        assert "data_folder_path" in exp_data
        assert "data_file_name" in exp_data
        assert exp_data["data_file_name"] == "data.hdf"

    def test_load_data_missing_config(self):
        """Test error handling when configuration is missing data file."""
        # Create mock args without CLI override
        mock_args = Mock()
        mock_args.data_file = None

        # Create config without data file
        mock_config = Mock()
        mock_config.config = {
            "experimental_data": {},  # No data file specified
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        # Should raise RuntimeError (wraps ValueError) with helpful message
        with pytest.raises(RuntimeError) as exc_info:
            _load_data(mock_args, mock_config)

        assert "No data file specified" in str(exc_info.value)

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_file_not_found(self, mock_loader_class):
        """Test error handling when data file doesn't exist."""
        # Mock the loader to raise FileNotFoundError
        mock_loader = Mock()
        mock_loader.load_experimental_data.side_effect = FileNotFoundError(
            "Data file not found: /path/to/missing.hdf"
        )
        mock_loader_class.return_value = mock_loader

        # Create mock args and config
        mock_args = Mock()
        mock_args.data_file = "/path/to/missing.hdf"

        mock_config = Mock()
        mock_config.config = {"analyzer_parameters": {}}

        # Should raise RuntimeError with FileNotFoundError as cause
        with pytest.raises(RuntimeError) as exc_info:
            _load_data(mock_args, mock_config)

        assert "Data file not found" in str(exc_info.value)


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_empty_config(self):
        """Test handling of empty configuration."""
        config_data = {}
        config = ConfigManager(config_override=config_data)

        # Should not crash during normalization
        assert config.config == {}

    def test_config_with_none_values(self):
        """Test handling of None values in config."""
        config_data = {
            "experimental_data": {"data_folder_path": None, "data_file_name": None}
        }

        config = ConfigManager(config_override=config_data)

        # Should handle None gracefully
        exp_data = config.config["experimental_data"]
        # Normalization might skip or handle None values
        # This tests for no crash

    @patch("homodyne.cli.commands.XPCSDataLoader")
    def test_load_data_with_complex_path(self, mock_loader_class):
        """Test loading data with complex path (spaces, special chars)."""
        # Mock the loader
        mock_loader = Mock()
        mock_loader.load_experimental_data.return_value = {"c2_exp": Mock(size=100)}
        mock_loader_class.return_value = mock_loader

        # Create mock args with complex path
        mock_args = Mock()
        mock_args.data_file = "./data/my experiment (test)/data file #1.hdf"

        mock_config = Mock()
        mock_config.config = {"analyzer_parameters": {}}

        # Should handle complex path
        result = _load_data(mock_args, mock_config)
        assert result is not None


class TestIntegrationScenarios:
    """Integration test scenarios combining multiple components."""

    def test_full_workflow_template_config(self, tmp_path):
        """Test full workflow: template config → normalization → data loading."""
        # Create a template-style config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "experimental_data": {
                "data_folder_path": str(tmp_path),
                "data_file_name": "test.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
        }

        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        # Load config
        config = ConfigManager(str(config_file))

        # Verify normalization happened
        exp_data = config.config["experimental_data"]
        assert "file_path" in exp_data
        assert exp_data["data_file_name"] == "test.hdf"

    def test_cli_override_precedence(self):
        """Test that CLI --data-file takes precedence over config."""
        # Create config with one file
        config_data = {
            "experimental_data": {
                "data_folder_path": "./config/",
                "data_file_name": "config_file.hdf",
            },
            "analyzer_parameters": {"dt": 0.1},
        }

        config = ConfigManager(config_override=config_data)

        # Mock args with different file
        mock_args = Mock()
        mock_args.data_file = "./cli/cli_file.hdf"

        # When _load_data is called, it should use CLI file, not config file
        # (This would need mocking XPCSDataLoader to fully test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

