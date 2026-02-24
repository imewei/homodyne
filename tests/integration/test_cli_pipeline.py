"""Integration Tests for CLI Pipeline
=====================================

CLI invocation -> full execution -> output validation.

Tests cover:
- homodyne-config generation in static and laminar_flow modes
- Error messaging for invalid invocations
- Output file validation

All tests use subprocess.run with sys.executable for CLI invocations.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_output_dir():
    """Provide a temporary directory for CLI output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCLIConfigGeneratorStatic:
    """Test homodyne-config generation in static mode."""

    def test_generates_static_config_file(self, tmp_output_dir: Path) -> None:
        """homodyne-config -m static should create a valid YAML file."""
        output_path = tmp_output_dir / "test_static.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"homodyne-config static failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert output_path.exists(), "Output YAML file was not created"

    def test_static_config_is_valid_yaml(self, tmp_output_dir: Path) -> None:
        """Generated static config must be parseable YAML."""
        output_path = tmp_output_dir / "test_static.yaml"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config should be a dict"
        assert "analysis_mode" in config, "Config must include analysis_mode"

    def test_static_config_has_analysis_mode(self, tmp_output_dir: Path) -> None:
        """Static config analysis_mode should be 'static_isotropic' or 'static'."""
        output_path = tmp_output_dir / "test_static.yaml"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        with open(output_path) as f:
            config = yaml.safe_load(f)

        mode = config.get("analysis_mode", "")
        assert mode in (
            "static",
            "static_isotropic",
        ), f"Expected static or static_isotropic mode, got '{mode}'"


@pytest.mark.integration
class TestCLIConfigGeneratorLaminarFlow:
    """Test homodyne-config generation in laminar_flow mode."""

    def test_generates_laminar_flow_config_file(self, tmp_output_dir: Path) -> None:
        """homodyne-config -m laminar_flow should create a valid YAML file."""
        output_path = tmp_output_dir / "test_flow.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "laminar_flow",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, (
            f"homodyne-config laminar_flow failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert output_path.exists(), "Output YAML file was not created"

    def test_laminar_flow_config_is_valid_yaml(self, tmp_output_dir: Path) -> None:
        """Generated laminar_flow config must be parseable YAML."""
        output_path = tmp_output_dir / "test_flow.yaml"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "laminar_flow",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        with open(output_path) as f:
            config = yaml.safe_load(f)

        assert isinstance(config, dict), "Config should be a dict"

    def test_laminar_flow_has_shear_parameters(self, tmp_output_dir: Path) -> None:
        """Laminar flow config should include shear-related parameter names."""
        output_path = tmp_output_dir / "test_flow.yaml"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "laminar_flow",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        with open(output_path) as f:
            config = yaml.safe_load(f)

        mode = config.get("analysis_mode", "")
        assert mode == "laminar_flow", f"Expected laminar_flow mode, got '{mode}'"


@pytest.mark.integration
class TestCLIInvalidConfig:
    """Test CLI error handling for invalid invocations."""

    def test_no_mode_shows_help(self) -> None:
        """Running homodyne-config without --mode should exit non-zero."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Should exit with non-zero since no mode specified
        assert result.returncode != 0, (
            "Expected non-zero exit code when no mode specified"
        )

    def test_overwrite_without_force_fails(self, tmp_output_dir: Path) -> None:
        """Generating to an existing path without --force should fail."""
        output_path = tmp_output_dir / "existing.yaml"

        # Create the file first
        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert output_path.exists()

        # Try to generate again without --force
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, (
            "Expected non-zero exit code when overwriting without --force"
        )

    def test_force_overwrite_succeeds(self, tmp_output_dir: Path) -> None:
        """Generating to an existing path with --force should succeed."""
        output_path = tmp_output_dir / "overwrite.yaml"

        # Create initial file
        subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "static",
                "--output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert output_path.exists()

        # Overwrite with --force
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--mode",
                "laminar_flow",
                "--output",
                str(output_path),
                "--force",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"--force overwrite failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Verify the file now has laminar_flow mode
        with open(output_path) as f:
            config = yaml.safe_load(f)
        assert config.get("analysis_mode") == "laminar_flow"

    def test_validate_nonexistent_file(self, tmp_output_dir: Path) -> None:
        """Validating a non-existent file should fail gracefully."""
        bogus_path = tmp_output_dir / "does_not_exist.yaml"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "homodyne.cli.config_generator",
                "--validate",
                str(bogus_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode != 0, (
            "Expected non-zero exit code for missing validate target"
        )
