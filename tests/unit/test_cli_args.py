"""Test suite for CLI argument parsing and method validation.

This module tests the CLI argument parser, focusing on method selection
and validation after the removal of 'nuts' and 'cmc' as direct method choices.

Test Categories:
- Valid method arguments: nlsq and mcmc work correctly
- Invalid method arguments: nuts and cmc raise clear errors
- Method validation: argument validation rejects removed methods
- Backward compatibility: existing nlsq usage unchanged
- CMC options: CMC-specific options accepted with mcmc method
"""

import pytest

from homodyne.cli.args_parser import create_parser
from homodyne.cli.args_parser import validate_args


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

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-num-shards", "4",
            "--config", str(config_file)
        ])

        assert args.method == "mcmc"
        assert args.cmc_num_shards == 4
        assert validate_args(args) is True

    def test_cmc_backend_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-backend is accepted with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-backend", "pjit",
            "--config", str(config_file)
        ])

        assert args.method == "mcmc"
        assert args.cmc_backend == "pjit"
        assert validate_args(args) is True

    def test_cmc_plot_diagnostics_accepted_with_mcmc(self, tmp_path):
        """Test that --cmc-plot-diagnostics is accepted with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-plot-diagnostics",
            "--config", str(config_file)
        ])

        assert args.method == "mcmc"
        assert args.cmc_plot_diagnostics is True
        assert validate_args(args) is True

    def test_all_cmc_options_accepted_together(self, tmp_path):
        """Test that all CMC options can be used together with --method mcmc."""
        parser = create_parser()
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("# Test config")

        args = parser.parse_args([
            "--method", "mcmc",
            "--cmc-num-shards", "8",
            "--cmc-backend", "multiprocessing",
            "--cmc-plot-diagnostics",
            "--config", str(config_file)
        ])

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

        args = parser.parse_args([
            "--method", "nlsq",
            "--cmc-num-shards", "4",
            "--cmc-backend", "pjit",
            "--cmc-plot-diagnostics",
            "--config", str(config_file)
        ])

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
