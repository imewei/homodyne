"""Tests for CMC I/O module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.io import (
    load_samples_npz,
    save_samples_npz,
    samples_to_arviz,
)


class TestLoadSamplesNpz:
    """Tests for load_samples_npz function."""

    def test_path_validation_nonexistent(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_samples_npz(Path("/nonexistent/path/samples.npz"))

    def test_path_validation_wrong_extension(self, tmp_path):
        """Test loading non-npz file raises ValueError."""
        wrong_file = tmp_path / "samples.txt"
        wrong_file.write_text("not npz data")

        with pytest.raises(ValueError, match=".npz"):
            load_samples_npz(wrong_file)

    def test_path_validation_directory(self, tmp_path):
        """Test loading directory raises ValueError."""
        dir_path = tmp_path / "samples.npz"
        dir_path.mkdir()

        with pytest.raises(ValueError, match="not a regular file"):
            load_samples_npz(dir_path)

    def test_path_resolution(self, tmp_path):
        """Test path with .. is resolved."""
        # Create valid npz file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        npz_path = tmp_path / "samples.npz"

        # Create minimal valid npz
        np.savez(
            npz_path,
            schema_version=np.array([1, 0]),
            posterior_samples=np.random.randn(2, 10, 3),
            param_names=np.array(["a", "b", "c"]),
            r_hat=np.array([1.0, 1.0, 1.0]),
            ess_bulk=np.array([100.0, 100.0, 100.0]),
            ess_tail=np.array([100.0, 100.0, 100.0]),
            divergences=np.array([0]),
            analysis_mode=np.array(["static"]),
            n_phi=np.array([1]),
            n_chains=np.array([2]),
            n_samples=np.array([10]),
        )

        # Load with relative path containing ..
        relative_path = subdir / ".." / "samples.npz"
        data = load_samples_npz(relative_path)

        assert data["schema_version"] == (1, 0)


class TestSaveSamplesNpz:
    """Tests for save_samples_npz function."""

    @pytest.fixture
    def mock_result(self):
        """Create mock CMCResult for testing."""

        class MockResult:
            param_names = ["contrast_0", "D0", "alpha"]
            samples = {
                "contrast_0": np.random.randn(4, 100),
                "D0": np.random.randn(4, 100) * 100 + 1000,
                "alpha": np.random.randn(4, 100) * 0.1,
            }
            r_hat = {"contrast_0": 1.01, "D0": 1.02, "alpha": 1.01}
            ess_bulk = {"contrast_0": 500.0, "D0": 600.0, "alpha": 550.0}
            ess_tail = {"contrast_0": 400.0, "D0": 500.0, "alpha": 450.0}
            divergences = 0
            analysis_mode = "static"
            n_chains = 4
            n_samples = 100

            def get_samples_array(self):
                return np.stack(
                    [self.samples[name] for name in self.param_names], axis=-1
                )

        return MockResult()

    def test_save_creates_file(self, mock_result, tmp_path):
        """Test save creates npz file."""
        output_path = tmp_path / "samples.npz"

        save_samples_npz(mock_result, output_path)

        assert output_path.exists()

    def test_save_contains_required_fields(self, mock_result, tmp_path):
        """Test saved file contains all required fields."""
        output_path = tmp_path / "samples.npz"
        save_samples_npz(mock_result, output_path)

        data = np.load(output_path, allow_pickle=True)

        required = [
            "schema_version",
            "posterior_samples",
            "param_names",
            "r_hat",
            "ess_bulk",
            "ess_tail",
            "divergences",
            "analysis_mode",
            "n_phi",
            "n_chains",
            "n_samples",
        ]

        for field in required:
            assert field in data.files, f"Missing field: {field}"

    def test_save_samples_shape(self, mock_result, tmp_path):
        """Test posterior_samples has correct shape."""
        output_path = tmp_path / "samples.npz"
        save_samples_npz(mock_result, output_path)

        data = np.load(output_path, allow_pickle=True)
        samples = data["posterior_samples"]

        # Shape should be (n_chains, n_samples, n_params)
        assert samples.shape == (4, 100, 3)


class TestSamplesToArviz:
    """Tests for samples_to_arviz function."""

    def test_conversion_basic(self):
        """Test basic conversion to ArviZ."""
        samples_data = {
            "schema_version": (1, 0),
            "posterior_samples": np.random.randn(4, 100, 3),
            "param_names": ["a", "b", "c"],
            "r_hat": np.array([1.0, 1.0, 1.0]),
            "ess_bulk": np.array([100.0, 100.0, 100.0]),
            "ess_tail": np.array([100.0, 100.0, 100.0]),
            "divergences": np.array([0]),
            "analysis_mode": "static",
            "n_phi": 1,
            "n_chains": 4,
            "n_samples": 100,
        }

        idata = samples_to_arviz(samples_data)

        # Should have posterior group
        assert hasattr(idata, "posterior")

        # Should have all parameters
        assert "a" in idata.posterior.data_vars
        assert "b" in idata.posterior.data_vars
        assert "c" in idata.posterior.data_vars

    def test_conversion_param_shapes(self):
        """Test converted parameters have correct shapes."""
        n_chains, n_samples, n_params = 2, 50, 2
        samples_data = {
            "schema_version": (1, 0),
            "posterior_samples": np.random.randn(n_chains, n_samples, n_params),
            "param_names": ["x", "y"],
            "r_hat": np.array([1.0, 1.0]),
            "ess_bulk": np.array([100.0, 100.0]),
            "ess_tail": np.array([100.0, 100.0]),
            "divergences": np.array([0]),
            "analysis_mode": "static",
            "n_phi": 1,
            "n_chains": n_chains,
            "n_samples": n_samples,
        }

        idata = samples_to_arviz(samples_data)

        # Each parameter should have shape (chain, draw)
        assert idata.posterior["x"].shape == (n_chains, n_samples)
        assert idata.posterior["y"].shape == (n_chains, n_samples)


class TestRoundTrip:
    """Tests for save/load round-trip consistency."""

    @pytest.fixture
    def mock_result(self):
        """Create mock result with known values."""

        class MockResult:
            param_names = ["contrast_0", "D0"]
            samples = {
                "contrast_0": np.array([[0.5, 0.51, 0.49], [0.48, 0.52, 0.50]]),
                "D0": np.array([[1000.0, 1001.0, 999.0], [998.0, 1002.0, 1000.0]]),
            }
            r_hat = {"contrast_0": 1.001, "D0": 1.002}
            ess_bulk = {"contrast_0": 500.0, "D0": 600.0}
            ess_tail = {"contrast_0": 400.0, "D0": 500.0}
            divergences = 2
            analysis_mode = "static"
            n_chains = 2
            n_samples = 3

            def get_samples_array(self):
                return np.stack(
                    [self.samples[name] for name in self.param_names], axis=-1
                )

        return MockResult()

    def test_round_trip_preserves_data(self, mock_result, tmp_path):
        """Test save then load preserves all data."""
        output_path = tmp_path / "samples.npz"

        save_samples_npz(mock_result, output_path)
        loaded = load_samples_npz(output_path)

        # Check metadata
        assert loaded["schema_version"] == (1, 0)
        assert loaded["analysis_mode"] == "static"
        assert loaded["n_chains"] == 2
        assert loaded["n_samples"] == 3

        # Check samples shape
        assert loaded["posterior_samples"].shape == (2, 3, 2)

        # Check param names
        assert loaded["param_names"] == ["contrast_0", "D0"]
