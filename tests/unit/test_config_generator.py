"""Tests for config_generator YAML escaping and text-replacement fallback."""

from __future__ import annotations

import yaml

from homodyne.cli.config_generator import _yaml_escape_string

# --- Unit tests for _yaml_escape_string ---


class TestYamlEscapeString:
    """Test the YAML double-quote escape helper."""

    def test_plain_string_unchanged(self) -> None:
        assert _yaml_escape_string("hello/world.hdf") == "hello/world.hdf"

    def test_double_quote_escaped(self) -> None:
        assert _yaml_escape_string('say "hi"') == 'say \\"hi\\"'

    def test_backslash_escaped(self) -> None:
        assert _yaml_escape_string("C:\\Users\\data") == "C:\\\\Users\\\\data"

    def test_colon_preserved(self) -> None:
        # Colons are safe inside double-quoted YAML strings
        assert _yaml_escape_string("host: port") == "host: port"

    def test_hash_preserved(self) -> None:
        # Hashes are safe inside double-quoted YAML strings
        assert _yaml_escape_string("run #42") == "run #42"

    def test_combined_special_chars(self) -> None:
        result = _yaml_escape_string('path\\to\\"file#1": ok')
        assert result == 'path\\\\to\\\\\\"file#1\\": ok'


# --- Round-trip test for the fallback text-replacement path ---

# Minimal template mimicking the real template's placeholder strings
_MINI_TEMPLATE = """\
experimental_data:
  file_path: "./data/sample/experiment.hdf"

output:
  directory: "./results"
"""


def _apply_fallback_replacement(
    template: str,
    data_file: str,
    output_dir: str,
    sample_name: str,
    experiment_id: str,
) -> str:
    """Reproduce the text-replacement logic from interactive_builder's fallback path."""
    content = template

    safe_data_file = _yaml_escape_string(data_file)
    safe_output_dir = _yaml_escape_string(output_dir)
    safe_sample_name = _yaml_escape_string(sample_name)
    safe_experiment_id = _yaml_escape_string(experiment_id)

    content = content.replace(
        'file_path: "./data/sample/experiment.hdf"',
        f'file_path: "{safe_data_file}"',
    )
    content = content.replace(
        'directory: "./results"', f'directory: "{safe_output_dir}"'
    )

    if "sample_name:" not in content:
        content = content.replace(
            f'directory: "{safe_output_dir}"',
            f'directory: "{safe_output_dir}"\n  sample_name: "{safe_sample_name}"\n  experiment_id: "{safe_experiment_id}"',
        )

    return content


class TestFallbackYamlRoundTrip:
    """Test that the text-replacement fallback produces valid YAML with special characters."""

    def test_plain_values_roundtrip(self) -> None:
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE, "/data/exp.hdf", "/output", "sample_1", "exp_001"
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == "/data/exp.hdf"
        assert parsed["output"]["directory"] == "/output"
        assert parsed["output"]["sample_name"] == "sample_1"
        assert parsed["output"]["experiment_id"] == "exp_001"

    def test_colon_in_path_roundtrip(self) -> None:
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE,
            "/data/run: test.hdf",
            "/out: put",
            "sample: one",
            "exp: 1",
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == "/data/run: test.hdf"
        assert parsed["output"]["directory"] == "/out: put"
        assert parsed["output"]["sample_name"] == "sample: one"

    def test_hash_in_values_roundtrip(self) -> None:
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE,
            "/data/run#42.hdf",
            "/output#dir",
            "sample#1",
            "exp#001",
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == "/data/run#42.hdf"
        assert parsed["output"]["sample_name"] == "sample#1"

    def test_double_quote_in_values_roundtrip(self) -> None:
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE,
            '/data/"quoted".hdf',
            '/out/"put"',
            '"my_sample"',
            '"exp_001"',
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == '/data/"quoted".hdf'
        assert parsed["output"]["sample_name"] == '"my_sample"'

    def test_backslash_in_values_roundtrip(self) -> None:
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE,
            "C:\\Users\\data\\exp.hdf",
            "C:\\output",
            "sample\\1",
            "exp\\001",
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == "C:\\Users\\data\\exp.hdf"
        assert parsed["output"]["directory"] == "C:\\output"

    def test_combined_special_chars_roundtrip(self) -> None:
        """All special characters combined in a single value."""
        nasty_path = '/data/"run: #1"\\test.hdf'
        result = _apply_fallback_replacement(
            _MINI_TEMPLATE, nasty_path, "./output", "sample", "exp"
        )
        parsed = yaml.safe_load(result)
        assert parsed["experimental_data"]["file_path"] == nasty_path
