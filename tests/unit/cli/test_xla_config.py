"""Tests for XLA configuration CLI."""

from homodyne.cli.xla_config import VALID_MODES, detect_optimal_devices, set_mode


class TestSetMode:
    """Tests for set_mode function."""

    def test_valid_mode_writes_config(self, tmp_path, monkeypatch):
        """Valid mode should write config file and return True."""
        config_dir = tmp_path / "etc" / "homodyne"
        config_file = config_dir / "xla_mode"
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
        monkeypatch.setattr(
            "homodyne.cli.xla_config._get_config_file", lambda: config_file
        )

        result = set_mode("cmc")

        assert result is True
        assert config_file.exists()
        assert config_file.read_text().strip() == "cmc"

    def test_invalid_mode_returns_false(self, tmp_path, monkeypatch):
        """Invalid mode should return False without writing."""
        config_file = tmp_path / "etc" / "homodyne" / "xla_mode"
        monkeypatch.setattr(
            "homodyne.cli.xla_config._get_config_file", lambda: config_file
        )

        result = set_mode("invalid_mode")

        assert result is False
        assert not config_file.exists()

    def test_venv_activation_script_detection(self, tmp_path, monkeypatch):
        """Should write config to per-venv path when VIRTUAL_ENV is set."""
        monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path))
        config_file = tmp_path / "etc" / "homodyne" / "xla_mode"
        monkeypatch.setattr(
            "homodyne.cli.xla_config._get_config_file", lambda: config_file
        )

        result = set_mode("nlsq")
        assert result is True
        assert config_file.read_text().strip() == "nlsq"

    def test_all_valid_modes_accepted(self, tmp_path, monkeypatch):
        """All modes in VALID_MODES should be accepted."""
        config_file = tmp_path / "etc" / "homodyne" / "xla_mode"
        monkeypatch.setattr(
            "homodyne.cli.xla_config._get_config_file", lambda: config_file
        )

        for mode in VALID_MODES:
            assert set_mode(mode) is True


class TestDetectOptimalDevices:
    """Tests for detect_optimal_devices."""

    def test_returns_positive_integer(self):
        """Should always return a positive integer."""
        result = detect_optimal_devices()
        assert isinstance(result, int)
        assert result > 0

    def test_result_in_expected_range(self):
        """Should return 2, 4, 6, or 8."""
        result = detect_optimal_devices()
        assert result in (2, 4, 6, 8)
