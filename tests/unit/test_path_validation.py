"""Unit tests for homodyne.utils.path_validation module.

Tests path validation utilities that prevent path traversal attacks
and ensure secure file operations.
"""

from pathlib import Path

import pytest

from homodyne.utils.path_validation import (
    PathValidationError,
    _sanitize_log_path,
    get_safe_output_dir,
    validate_plot_save_path,
    validate_save_path,
)


class TestPathValidationError:
    """Test PathValidationError exception class."""

    def test_is_subclass_of_value_error(self):
        """Test that PathValidationError inherits from ValueError."""
        assert issubclass(PathValidationError, ValueError)

    def test_can_raise_and_catch(self):
        """Test that exception can be raised and caught."""
        with pytest.raises(PathValidationError) as exc_info:
            raise PathValidationError("test error message")
        assert "test error message" in str(exc_info.value)


class TestValidateSavePath:
    """Test validate_save_path function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert validate_save_path(None) is None

    def test_valid_relative_path(self, tmp_path):
        """Test valid relative path is resolved."""
        result = validate_save_path(
            "test.txt",
            require_parent_exists=False,
            base_dir=tmp_path,
        )
        assert result is not None
        assert result.name == "test.txt"

    def test_valid_absolute_path(self, tmp_path):
        """Test valid absolute path is accepted."""
        abs_path = tmp_path / "test.txt"
        result = validate_save_path(
            str(abs_path),
            require_parent_exists=True,
        )
        assert result is not None
        assert result == abs_path

    def test_path_traversal_rejected(self):
        """Test that path traversal attempts are rejected."""
        with pytest.raises(PathValidationError, match="Path traversal"):
            validate_save_path("../../../etc/passwd")

        with pytest.raises(PathValidationError, match="Path traversal"):
            validate_save_path("subdir/../../../secret")

        with pytest.raises(PathValidationError, match="Path traversal"):
            validate_save_path("..\\..\\windows\\system32\\config")

    def test_absolute_path_rejected_when_not_allowed(self, tmp_path):
        """Test absolute path rejected when allow_absolute=False."""
        abs_path = tmp_path / "test.txt"
        with pytest.raises(PathValidationError, match="Absolute paths not allowed"):
            validate_save_path(str(abs_path), allow_absolute=False)

    def test_absolute_path_allowed_when_allowed(self, tmp_path):
        """Test absolute path accepted when allow_absolute=True."""
        abs_path = tmp_path / "test.txt"
        result = validate_save_path(
            str(abs_path),
            allow_absolute=True,
            require_parent_exists=True,
        )
        assert result == abs_path

    def test_extension_validation(self, tmp_path):
        """Test file extension validation."""
        # Valid extension
        result = validate_save_path(
            "test.png",
            allowed_extensions=(".png", ".jpg"),
            require_parent_exists=False,
            base_dir=tmp_path,
        )
        assert result is not None
        assert result.suffix == ".png"

        # Invalid extension
        with pytest.raises(ValueError, match="Invalid file extension"):
            validate_save_path(
                "test.exe",
                allowed_extensions=(".png", ".jpg"),
                require_parent_exists=False,
                base_dir=tmp_path,
            )

    def test_extension_case_insensitive(self, tmp_path):
        """Test that extension check is case-insensitive."""
        result = validate_save_path(
            "test.PNG",
            allowed_extensions=(".png", ".jpg"),
            require_parent_exists=False,
            base_dir=tmp_path,
        )
        assert result is not None

    def test_parent_must_exist_when_required(self, tmp_path):
        """Test parent directory must exist when required."""
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            validate_save_path(
                "nonexistent_dir/test.txt",
                require_parent_exists=True,
                base_dir=tmp_path,
            )

    def test_parent_not_required(self, tmp_path):
        """Test parent directory not checked when not required."""
        result = validate_save_path(
            "nonexistent_dir/test.txt",
            require_parent_exists=False,
            base_dir=tmp_path,
        )
        assert result is not None
        assert result.name == "test.txt"

    def test_parent_must_be_directory(self, tmp_path):
        """Test that parent path must be a directory."""
        # Create a file that we'll try to use as parent
        file_as_parent = tmp_path / "not_a_dir"
        file_as_parent.write_text("I'm a file")

        with pytest.raises(ValueError, match="Parent path is not a directory"):
            validate_save_path(
                str(file_as_parent / "subfile.txt"),
                require_parent_exists=True,
            )

    def test_base_dir_containment(self, tmp_path):
        """Test that resolved path stays within base_dir."""
        # This test creates a symlink outside the base_dir
        # to test containment check for relative paths
        outside_dir = tmp_path.parent / "outside_dir"
        outside_dir.mkdir(exist_ok=True)

        inside_dir = tmp_path / "inside"
        inside_dir.mkdir(exist_ok=True)

        # Create symlink pointing outside
        symlink = inside_dir / "escape_link"
        try:
            symlink.symlink_to(outside_dir)

            # Attempting to use symlink that escapes should fail
            with pytest.raises(PathValidationError, match="outside allowed directory"):
                validate_save_path(
                    "escape_link/secret.txt",
                    require_parent_exists=False,
                    base_dir=inside_dir,
                )
        except OSError:
            # Symlinks may not be supported on all systems
            pytest.skip("Symlinks not supported on this system")
        finally:
            # Cleanup
            if symlink.exists():
                symlink.unlink()
            if outside_dir.exists():
                outside_dir.rmdir()

    def test_accepts_path_object(self, tmp_path):
        """Test that Path objects are accepted."""
        path = tmp_path / "test.txt"
        result = validate_save_path(path, require_parent_exists=True)
        assert result == path


class TestValidatePlotSavePath:
    """Test validate_plot_save_path function."""

    def test_none_returns_none(self):
        """Test that None input returns None."""
        assert validate_plot_save_path(None) is None

    def test_valid_png_path(self, tmp_path):
        """Test valid PNG path is accepted."""
        result = validate_plot_save_path(
            tmp_path / "plot.png",
            require_parent_exists=True,
        )
        assert result is not None
        assert result.suffix == ".png"

    def test_valid_pdf_path(self, tmp_path):
        """Test valid PDF path is accepted."""
        result = validate_plot_save_path(
            tmp_path / "plot.pdf",
            require_parent_exists=True,
        )
        assert result is not None
        assert result.suffix == ".pdf"

    def test_valid_svg_path(self, tmp_path):
        """Test valid SVG path is accepted."""
        result = validate_plot_save_path(
            tmp_path / "plot.svg",
            require_parent_exists=True,
        )
        assert result is not None
        assert result.suffix == ".svg"

    def test_valid_eps_path(self, tmp_path):
        """Test valid EPS path is accepted."""
        result = validate_plot_save_path(
            tmp_path / "plot.eps",
            require_parent_exists=True,
        )
        assert result is not None
        assert result.suffix == ".eps"

    def test_valid_jpg_paths(self, tmp_path):
        """Test valid JPG/JPEG paths are accepted."""
        for ext in [".jpg", ".jpeg"]:
            result = validate_plot_save_path(
                tmp_path / f"plot{ext}",
                require_parent_exists=True,
            )
            assert result is not None

    def test_valid_tiff_paths(self, tmp_path):
        """Test valid TIFF paths are accepted."""
        for ext in [".tiff", ".tif"]:
            result = validate_plot_save_path(
                tmp_path / f"plot{ext}",
                require_parent_exists=True,
            )
            assert result is not None

    def test_invalid_extension_rejected(self, tmp_path):
        """Test that non-image extensions are rejected."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            validate_plot_save_path(
                tmp_path / "plot.txt",
                require_parent_exists=True,
            )

    def test_executable_extension_rejected(self, tmp_path):
        """Test that executable extensions are rejected."""
        with pytest.raises(ValueError, match="Invalid file extension"):
            validate_plot_save_path(
                tmp_path / "plot.exe",
                require_parent_exists=True,
            )

    def test_path_traversal_rejected(self):
        """Test path traversal is rejected."""
        with pytest.raises(PathValidationError, match="Path traversal"):
            validate_plot_save_path("../../../etc/plot.png")


class TestSanitizeLogPath:
    """Test _sanitize_log_path function."""

    def test_short_path_unchanged(self):
        """Test short paths pass through unchanged."""
        result = _sanitize_log_path("/short/path.txt")
        assert result == "/short/path.txt"

    def test_newline_escaped(self):
        """Test newlines are escaped to prevent log injection."""
        result = _sanitize_log_path("path\nwith\nnewlines")
        assert "\n" not in result
        assert "\\n" in result

    def test_carriage_return_escaped(self):
        """Test carriage returns are escaped."""
        result = _sanitize_log_path("path\rwith\rreturns")
        assert "\r" not in result
        assert "\\r" in result

    def test_long_path_truncated(self):
        """Test that long paths are truncated."""
        long_path = "/very/long/path/" + "a" * 100
        result = _sanitize_log_path(long_path, max_length=50)
        assert len(result) <= 50 + 3  # Allows for "..."
        assert "..." in result

    def test_truncation_preserves_start_and_end(self):
        """Test truncation shows beginning and end of path."""
        long_path = "START" + "x" * 100 + "END"
        result = _sanitize_log_path(long_path, max_length=50)
        assert result.startswith("START")
        assert result.endswith("END")

    def test_custom_max_length(self):
        """Test custom max_length parameter."""
        path = "a" * 30
        result = _sanitize_log_path(path, max_length=20)
        assert "..." in result


class TestGetSafeOutputDir:
    """Test get_safe_output_dir function."""

    def test_none_creates_default_subdir(self, tmp_path, monkeypatch):
        """Test that None creates default subdirectory in cwd."""
        monkeypatch.chdir(tmp_path)
        result = get_safe_output_dir(None)
        assert result.name == "homodyne_output"
        assert result.exists()
        assert result.is_dir()

    def test_custom_default_subdir(self, tmp_path, monkeypatch):
        """Test custom default subdirectory name."""
        monkeypatch.chdir(tmp_path)
        result = get_safe_output_dir(None, default_subdir="custom_output")
        assert result.name == "custom_output"
        assert result.exists()

    def test_existing_directory_returned(self, tmp_path):
        """Test existing directory is returned as-is."""
        existing = tmp_path / "existing_dir"
        existing.mkdir()
        result = get_safe_output_dir(existing)
        assert result == existing

    def test_new_directory_created(self, tmp_path):
        """Test new directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_output_dir"
        assert not new_dir.exists()
        result = get_safe_output_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_nested_directory_created(self, tmp_path):
        """Test nested directories are created."""
        nested = tmp_path / "level1" / "level2" / "level3"
        assert not nested.exists()
        result = get_safe_output_dir(nested)
        assert result.exists()
        assert result.is_dir()

    def test_path_traversal_rejected(self, tmp_path):
        """Test path traversal is rejected."""
        with pytest.raises(PathValidationError, match="Path traversal"):
            get_safe_output_dir(tmp_path / ".." / "escape")

    def test_file_as_dir_rejected(self, tmp_path):
        """Test that existing file raises error."""
        file_path = tmp_path / "i_am_a_file"
        file_path.write_text("content")
        with pytest.raises(PathValidationError, match="not a directory"):
            get_safe_output_dir(file_path)

    def test_accepts_string_path(self, tmp_path):
        """Test that string paths are accepted."""
        result = get_safe_output_dir(str(tmp_path / "string_path_dir"))
        assert result.exists()
        assert result.is_dir()


class TestPathValidationIntegration:
    """Integration tests for path validation."""

    def test_validate_then_write(self, tmp_path):
        """Test that validated paths can be written to."""
        validated = validate_save_path(
            tmp_path / "output.txt",
            require_parent_exists=True,
        )
        assert validated is not None

        # Should be able to write to validated path
        validated.write_text("test content")
        assert validated.read_text() == "test content"

    def test_plot_path_validation_flow(self, tmp_path):
        """Test complete plot save path validation flow."""
        # Get output directory
        output_dir = get_safe_output_dir(tmp_path / "plots")

        # Validate plot path within output directory
        plot_path = validate_plot_save_path(
            output_dir / "figure.png",
            require_parent_exists=True,
        )
        assert plot_path is not None
        assert plot_path.parent.exists()

    def test_security_edge_cases(self, tmp_path):
        """Test various security edge cases."""
        # Null bytes (should be handled by Path)
        with pytest.raises((PathValidationError, ValueError)):
            validate_save_path("file\x00.txt", base_dir=tmp_path)

        # Unicode normalization attacks
        # These should either work or raise clear errors
        validate_save_path(
            "normal_unicode.txt",
            require_parent_exists=False,
            base_dir=tmp_path,
        )

    def test_windows_path_separator(self, tmp_path):
        """Test Windows-style path separators are handled."""
        # Windows backslashes with traversal
        with pytest.raises(PathValidationError, match="Path traversal"):
            validate_save_path("..\\..\\secret.txt", base_dir=tmp_path)
