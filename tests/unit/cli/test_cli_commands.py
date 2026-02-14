"""Unit tests for CLI command-line interface.

Tests for config generator, argument parser, and command execution.
"""

import argparse
from pathlib import Path

import pytest

from homodyne.cli.config_generator import create_parser, get_template_path


class TestConfigGeneratorParser:
    """Tests for homodyne-config argument parser."""

    def test_create_parser_returns_argparse_parser(self):
        """Test that create_parser returns ArgumentParser."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parser_has_mode_argument(self):
        """Test that parser includes --mode argument."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "static"])
        assert args.mode == "static"

    def test_parser_mode_choices(self):
        """Test that mode only accepts valid choices."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])

    def test_parser_accepts_static_mode(self):
        """Test parser accepts static mode."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "static"])
        assert args.mode == "static"

    def test_parser_accepts_laminar_flow_mode(self):
        """Test parser accepts laminar_flow mode."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "laminar_flow"])
        assert args.mode == "laminar_flow"

    def test_parser_has_output_argument(self):
        """Test that parser includes --output argument."""
        parser = create_parser()
        args = parser.parse_args(["--output", "test.yaml"])
        assert args.output == Path("test.yaml")

    def test_parser_output_converts_to_path(self):
        """Test that output argument is converted to Path."""
        parser = create_parser()
        args = parser.parse_args(["--output", "config/test.yaml"])
        assert isinstance(args.output, Path)

    def test_parser_has_interactive_flag(self):
        """Test that parser includes --interactive flag."""
        parser = create_parser()
        args = parser.parse_args(["--interactive"])
        assert args.interactive is True

    def test_parser_interactive_default_false(self):
        """Test that interactive defaults to False."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.interactive is False

    def test_parser_has_validate_argument(self):
        """Test that parser includes --validate argument."""
        parser = create_parser()
        args = parser.parse_args(["--validate", "my_config.yaml"])
        assert args.validate == Path("my_config.yaml")

    def test_parser_has_force_flag(self):
        """Test that parser includes --force flag."""
        parser = create_parser()
        args = parser.parse_args(["--force"])
        assert args.force is True

    def test_parser_force_default_false(self):
        """Test that force defaults to False."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.force is False

    def test_parser_short_mode_option(self):
        """Test that -m short option works for mode."""
        parser = create_parser()
        args = parser.parse_args(["-m", "static"])
        assert args.mode == "static"

    def test_parser_short_output_option(self):
        """Test that -o short option works for output."""
        parser = create_parser()
        args = parser.parse_args(["-o", "config.yaml"])
        assert args.output == Path("config.yaml")

    def test_parser_short_interactive_option(self):
        """Test that -i short option works for interactive."""
        parser = create_parser()
        args = parser.parse_args(["-i"])
        assert args.interactive is True

    def test_parser_short_validate_option(self):
        """Test that -v short option works for validate."""
        parser = create_parser()
        args = parser.parse_args(["-v", "test.yaml"])
        assert args.validate == Path("test.yaml")

    def test_parser_short_force_option(self):
        """Test that -f short option works for force."""
        parser = create_parser()
        args = parser.parse_args(["-f"])
        assert args.force is True

    def test_parser_combined_mode_output(self):
        """Test combining mode and output arguments."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "laminar_flow", "--output", "flow.yaml"])
        assert args.mode == "laminar_flow"
        assert args.output == Path("flow.yaml")

    def test_parser_combined_mode_force(self):
        """Test combining mode and force arguments."""
        parser = create_parser()
        args = parser.parse_args(["--mode", "static", "--force"])
        assert args.mode == "static"
        assert args.force is True

    def test_parser_no_arguments_succeeds(self):
        """Test parser with no arguments (all defaults)."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.mode is None
        assert args.output is None
        assert args.interactive is False
        assert args.validate is None
        assert args.force is False


class TestGetTemplatePath:
    """Tests for get_template_path function."""

    def test_get_template_path_for_static(self):
        """Test getting template path for static mode."""
        path = get_template_path("static")
        assert isinstance(path, Path)
        assert path.name == "homodyne_static.yaml"

    def test_get_template_path_for_laminar_flow(self):
        """Test getting template path for laminar_flow mode."""
        path = get_template_path("laminar_flow")
        assert isinstance(path, Path)
        assert path.name == "homodyne_laminar_flow.yaml"

    def test_get_template_path_templates_exist(self):
        """Test that template files actually exist."""
        static_path = get_template_path("static")
        flow_path = get_template_path("laminar_flow")
        assert static_path.exists(), f"Static template not found: {static_path}"
        assert flow_path.exists(), f"Flow template not found: {flow_path}"

    def test_get_template_path_invalid_mode_raises(self):
        """Test that invalid mode raises KeyError."""
        with pytest.raises(KeyError):
            get_template_path("invalid_mode")

    def test_get_template_path_returns_absolute_path(self):
        """Test that returned path is absolute."""
        path = get_template_path("static")
        assert path.is_absolute()

    def test_static_template_in_templates_directory(self):
        """Test that static template is in templates directory."""
        path = get_template_path("static")
        assert path.parent.name == "templates"

    def test_laminar_flow_template_in_templates_directory(self):
        """Test that laminar_flow template is in templates directory."""
        path = get_template_path("laminar_flow")
        assert path.parent.name == "templates"


class TestConfigGeneratorDefaults:
    """Tests for default argument values."""

    def test_default_mode_is_none(self):
        """Test that mode defaults to None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.mode is None

    def test_default_output_is_none(self):
        """Test that output defaults to None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.output is None

    def test_all_flags_false_by_default(self):
        """Test that all boolean flags default to False."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.interactive is False
        assert args.force is False

    def test_validate_default_none(self):
        """Test that validate defaults to None."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.validate is None
