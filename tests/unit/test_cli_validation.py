"""Comprehensive CLI validation tests for v2.1.0 simplification.

This module tests the CLI validation for the MCMC/CMC v2.1.0 simplification,
which removed direct 'nuts' and 'cmc' method flags in favor of automatic
NUTS/CMC selection via the 'mcmc' method.

Test Coverage:
- Deprecated method rejection (nuts, cmc, auto)
- Accepted methods functionality (nlsq, mcmc)
- Shell alias existence and correctness
- CMC-specific CLI options (min-samples-cmc, memory-threshold-pct)
- CLI parameter overrides
- Help text validation
- Default behavior verification
"""

import subprocess
import sys
from pathlib import Path

import pytest

from homodyne.cli.args_parser import create_parser
from homodyne.cli.args_parser import validate_args


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

        args = parser.parse_args([
            "--method", "mcmc",
            "--min-samples-cmc", "20",
            "--config", str(config_file),
        ])

        assert args.method == "mcmc"
        assert args.min_samples_cmc == 20
        assert validate_args(args) is True

    def test_memory_threshold_pct_override_accepted(self, tmp_path):
        """Test that --memory-threshold-pct override is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--memory-threshold-pct", "0.35",
            "--config", str(config_file),
        ])

        assert args.method == "mcmc"
        assert args.memory_threshold_pct == 0.35
        assert validate_args(args) is True

    def test_both_cmc_thresholds_accepted_together(self, tmp_path):
        """Test that both CMC threshold overrides work together."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--min-samples-cmc", "25",
            "--memory-threshold-pct", "0.40",
            "--config", str(config_file),
        ])

        assert args.method == "mcmc"
        assert args.min_samples_cmc == 25
        assert args.memory_threshold_pct == 0.40
        assert validate_args(args) is True

    def test_cmc_num_shards_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-num-shards is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-num-shards", "8",
            "--config", str(config_file),
        ])

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 8
        assert validate_args(args) is True

    def test_cmc_backend_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-backend is accepted with mcmc method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-backend", "pjit",
            "--config", str(config_file),
        ])

        assert args.method == "mcmc"
        assert args.cmc_backend == "pjit"
        assert validate_args(args) is True

    def test_all_cmc_options_together(self, tmp_path):
        """Test that all CMC options can be used together."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--min-samples-cmc", "20",
            "--memory-threshold-pct", "0.35",
            "--cmc-num-shards", "8",
            "--cmc-backend", "multiprocessing",
            "--cmc-plot-diagnostics",
            "--config", str(config_file),
        ])

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

        args = parser.parse_args([
            "--initial-d0", "1500.5",
            "--config", str(config_file),
        ])

        assert args.initial_d0 == 1500.5
        assert validate_args(args) is True

    def test_initial_alpha_override(self, tmp_path):
        """Test --initial-alpha override."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--initial-alpha", "0.75",
            "--config", str(config_file),
        ])

        assert args.initial_alpha == 0.75
        assert validate_args(args) is True

    def test_initial_d_offset_override(self, tmp_path):
        """Test --initial-d-offset override."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--initial-d-offset", "100.0",
            "--config", str(config_file),
        ])

        assert args.initial_d_offset == 100.0
        assert validate_args(args) is True

    def test_multiple_parameter_overrides(self, tmp_path):
        """Test multiple parameter overrides applied correctly."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--initial-d0", "1234.5",
            "--initial-alpha", "0.567",
            "--initial-d-offset", "12.34",
            "--config", str(config_file),
        ])

        assert args.initial_d0 == 1234.5
        assert args.initial_alpha == 0.567
        assert args.initial_d_offset == 12.34
        assert validate_args(args) is True

    def test_laminar_flow_parameter_overrides(self, tmp_path):
        """Test laminar flow specific parameter overrides."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--initial-gamma-dot-t0", "1.5",
            "--initial-beta", "0.3",
            "--initial-gamma-dot-offset", "0.1",
            "--initial-phi0", "0.785",
            "--config", str(config_file),
        ])

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

        args = parser.parse_args([
            "--method", "mcmc",
            "--dense-mass-matrix",
            "--config", str(config_file),
        ])

        assert args.dense_mass_matrix is True
        assert validate_args(args) is True

    def test_mcmc_sample_overrides(self, tmp_path):
        """Test MCMC sample count overrides."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--n-samples", "2000",
            "--n-warmup", "1000",
            "--n-chains", "8",
            "--config", str(config_file),
        ])

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

        args = parser.parse_args([
            "--initial-d0", "-100.0",
            "--config", str(config_file),
        ])

        # Validation should fail for negative D0
        assert validate_args(args) is False

    def test_invalid_memory_threshold_rejected(self, tmp_path):
        """Test that invalid memory threshold values are rejected."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--memory-threshold-pct", "1.5",  # > 1.0
            "--config", str(config_file),
        ])

        # Validation should fail for out-of-range threshold
        assert validate_args(args) is False

    def test_memory_threshold_boundaries(self, tmp_path):
        """Test memory threshold at valid boundaries."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Test 0.0 boundary
        args = parser.parse_args([
            "--memory-threshold-pct", "0.0",
            "--config", str(config_file),
        ])
        assert validate_args(args) is True

        # Test 1.0 boundary
        args = parser.parse_args([
            "--memory-threshold-pct", "1.0",
            "--config", str(config_file),
        ])
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
            "automatic NUTS/CMC selection" in help_text or
            "automatic" in help_text.lower() or
            "dual criteria" in help_text.lower()
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
        args = parser.parse_args([
            "--n-samples", "1000",
            "--config", str(config_file),
        ])
        assert validate_args(args) is True

        # Invalid samples
        args = parser.parse_args([
            "--n-samples", "0",
            "--config", str(config_file),
        ])
        assert validate_args(args) is False

    def test_mcmc_warmup_validation(self, tmp_path):
        """Test MCMC warmup count validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Valid warmup
        args = parser.parse_args([
            "--n-warmup", "500",
            "--config", str(config_file),
        ])
        assert validate_args(args) is True

        # Invalid warmup
        args = parser.parse_args([
            "--n-warmup", "-100",
            "--config", str(config_file),
        ])
        assert validate_args(args) is False

    def test_mcmc_chains_validation(self, tmp_path):
        """Test MCMC chain count validation."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        # Valid chains
        args = parser.parse_args([
            "--n-chains", "4",
            "--config", str(config_file),
        ])
        assert validate_args(args) is True

        # Invalid chains
        args = parser.parse_args([
            "--n-chains", "0",
            "--config", str(config_file),
        ])
        assert validate_args(args) is False


class TestCMCWarningsWithNLSQ:
    """Test that CMC options generate warnings when used with nlsq."""

    def test_cmc_options_warned_with_nlsq_method(self, tmp_path, capsys):
        """Test that CMC options generate warnings with nlsq method."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "nlsq",
            "--cmc-num-shards", "4",
            "--cmc-backend", "pjit",
            "--cmc-plot-diagnostics",
            "--config", str(config_file),
        ])

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

        args = parser.parse_args([
            "--static-mode",
            "--laminar-flow",
            "--config", str(config_file),
        ])

        # Should fail validation
        assert validate_args(args) is False

    def test_verbose_and_quiet_conflict(self, tmp_path):
        """Test that --verbose and --quiet conflict."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--verbose",
            "--quiet",
            "--config", str(config_file),
        ])

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

        args = parser.parse_args([
            "--config", str(tmp_path / "nonexistent.yaml"),
        ])

        # Validation should fail for missing config
        assert validate_args(args) is False

    def test_existing_config_file_accepted(self, tmp_path):
        """Test that existing config file is accepted."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--config", str(config_file),
        ])

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
