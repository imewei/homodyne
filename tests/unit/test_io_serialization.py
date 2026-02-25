"""Unit tests for I/O serialization utilities.

Tests for JSON serialization, NLSQ writers, and MCMC writers.
Ensures correct conversion of numpy arrays, nested structures, and NaN/Inf handling.
"""

import json

import numpy as np
import pytest

from homodyne.io.json_utils import json_safe, json_serializer
from homodyne.io.nlsq_writers import save_nlsq_json_files, save_nlsq_npz_file


class TestJsonSafe:
    """Tests for json_safe function."""

    def test_converts_numpy_arrays_to_lists(self):
        """Test that numpy arrays are converted to lists."""
        arr = np.array([1, 2, 3])
        result = json_safe(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_converts_nested_dicts(self):
        """Test handling of nested dictionaries with numpy arrays."""
        data = {
            "arr": np.array([1.0, 2.0]),
            "val": np.float64(3.14),
            "nested": {"inner": np.array([4, 5])},
        }
        result = json_safe(data)
        assert result["arr"] == [1.0, 2.0]
        assert result["val"] == 3.14
        assert result["nested"]["inner"] == [4, 5]

    def test_converts_numpy_scalar_types(self):
        """Test conversion of numpy scalar types."""
        assert json_safe(np.int64(42)) == 42
        assert json_safe(np.float32(3.14)) == pytest.approx(3.14, abs=1e-5)
        assert json_safe(np.bool_(True)) is True

    def test_handles_nested_lists(self):
        """Test handling of nested lists with numpy types."""
        data = [np.array([1, 2]), [np.float64(3.0), np.int32(4)]]
        result = json_safe(data)
        assert result == [[1, 2], [3.0, 4]]

    def test_handles_nan_inf(self):
        """Test that NaN/Inf are sanitized for JSON compatibility.

        JSON spec does not support NaN, Inf, or -Inf. They are converted to:
        - NaN → None
        - Inf → "Infinity"
        - -Inf → "-Infinity"
        """
        data = {
            "nan_val": np.nan,
            "inf_val": np.inf,
            "ninf_val": -np.inf,
            "arr": np.array([1.0, np.nan, np.inf]),
        }
        result = json_safe(data)
        assert result["nan_val"] is None
        assert result["inf_val"] == "Infinity"
        assert result["ninf_val"] == "-Infinity"
        assert len(result["arr"]) == 3
        assert result["arr"][0] == 1.0
        assert result["arr"][1] is None  # NaN → None
        assert result["arr"][2] == "Infinity"  # Inf → "Infinity"

    def test_handles_empty_arrays(self):
        """Test handling of empty arrays."""
        result = json_safe(np.array([]))
        assert result == []

    def test_preserves_regular_python_types(self):
        """Test that regular Python types are preserved."""
        data = {"int": 42, "float": 3.14, "str": "test", "bool": True, "none": None}
        result = json_safe(data)
        assert result == data


class TestJsonSerializer:
    """Tests for json_serializer function."""

    def test_serializes_numpy_arrays(self):
        """Test serialization of numpy arrays with json.dumps."""
        data = {"arr": np.array([1, 2, 3])}
        result = json.dumps(data, default=json_serializer)
        parsed = json.loads(result)
        assert parsed["arr"] == [1, 2, 3]

    def test_serializes_numpy_scalars(self):
        """Test serialization of numpy scalar types."""
        data = {
            "int": np.int64(42),
            "float": np.float32(3.14),
            "bool": np.bool_(True),
        }
        result = json.dumps(data, default=json_serializer)
        parsed = json.loads(result)
        assert parsed["int"] == 42
        assert parsed["bool"] is True

    def test_fallback_to_string_for_unknown_types(self):
        """Test that unknown types are converted to string."""

        class CustomObject:
            def __str__(self):
                return "custom"

        data = {"obj": CustomObject()}
        result = json.dumps(data, default=json_serializer)
        parsed = json.loads(result)
        assert parsed["obj"] == "custom"


class TestNlsqWriters:
    """Tests for NLSQ writer functions."""

    def test_save_nlsq_json_files_creates_three_files(self, tmp_path):
        """Test that save_nlsq_json_files creates 3 JSON files."""
        param_dict = {"D0": {"value": 1000.0, "uncertainty": 50.0}}
        analysis_dict = {"method": "nlsq", "fit_quality": {"reduced_chi2": 1.2}}
        convergence_dict = {"status": "success", "iterations": 10}

        save_nlsq_json_files(param_dict, analysis_dict, convergence_dict, tmp_path)

        assert (tmp_path / "parameters.json").exists()
        assert (tmp_path / "analysis_results_nlsq.json").exists()
        assert (tmp_path / "convergence_metrics.json").exists()

    def test_save_nlsq_json_files_correct_content(self, tmp_path):
        """Test that JSON files contain correct content."""
        param_dict = {"D0": {"value": 1000.0}}
        analysis_dict = {"method": "nlsq"}
        convergence_dict = {"status": "success"}

        save_nlsq_json_files(param_dict, analysis_dict, convergence_dict, tmp_path)

        with open(tmp_path / "parameters.json") as f:
            loaded_params = json.load(f)
            assert loaded_params == param_dict

        with open(tmp_path / "analysis_results_nlsq.json") as f:
            loaded_analysis = json.load(f)
            assert loaded_analysis == analysis_dict

    def test_save_nlsq_npz_file_creates_file(self, tmp_path):
        """Test that save_nlsq_npz_file creates NPZ file."""
        phi_angles = np.array([0.0, 30.0, 60.0])
        n_angles, n_t1, n_t2 = 3, 10, 10
        c2_exp = np.random.randn(n_angles, n_t1, n_t2)
        c2_raw = np.random.randn(n_angles, n_t1, n_t2)
        c2_scaled = np.random.randn(n_angles, n_t1, n_t2)
        per_angle_scaling = np.ones((n_angles, 2))
        per_angle_scaling_solver = np.ones((n_angles, 2))
        residuals = np.random.randn(n_angles, n_t1, n_t2)
        residuals_norm = np.random.randn(n_angles, n_t1, n_t2)
        t1 = np.logspace(-3, 1, n_t1)
        t2 = np.logspace(-3, 1, n_t2)
        q = 0.05

        save_nlsq_npz_file(
            phi_angles,
            c2_exp,
            c2_raw,
            c2_scaled,
            None,
            per_angle_scaling,
            per_angle_scaling_solver,
            residuals,
            residuals_norm,
            t1,
            t2,
            q,
            tmp_path,
        )

        assert (tmp_path / "fitted_data.npz").exists()

    def test_save_nlsq_npz_file_correct_arrays(self, tmp_path):
        """Test that NPZ file contains correct arrays."""
        phi_angles = np.array([0.0, 30.0])
        n_angles, n_t1, n_t2 = 2, 5, 5
        c2_exp = np.ones((n_angles, n_t1, n_t2))
        c2_raw = np.ones((n_angles, n_t1, n_t2)) * 2
        c2_scaled = np.ones((n_angles, n_t1, n_t2)) * 3
        per_angle_scaling = np.array([[1.0, 0.5], [1.0, 0.5]])
        per_angle_scaling_solver = np.array([[1.0, 0.5], [1.0, 0.5]])
        residuals = np.zeros((n_angles, n_t1, n_t2))
        residuals_norm = np.zeros((n_angles, n_t1, n_t2))
        t1 = np.linspace(0.1, 1.0, n_t1)
        t2 = np.linspace(0.1, 1.0, n_t2)
        q = 0.05

        save_nlsq_npz_file(
            phi_angles,
            c2_exp,
            c2_raw,
            c2_scaled,
            None,
            per_angle_scaling,
            per_angle_scaling_solver,
            residuals,
            residuals_norm,
            t1,
            t2,
            q,
            tmp_path,
        )

        data = np.load(tmp_path / "fitted_data.npz")
        assert np.allclose(data["phi_angles"], phi_angles)
        assert np.allclose(data["c2_exp"], c2_exp)
        assert np.allclose(data["t1"], t1)
        assert data["q"].item() == pytest.approx(q)

    def test_save_nlsq_npz_file_with_c2_solver(self, tmp_path):
        """Test NPZ file with optional c2_solver array."""
        phi_angles = np.array([0.0])
        n_angles, n_t1, n_t2 = 1, 5, 5
        c2_exp = np.ones((n_angles, n_t1, n_t2))
        c2_raw = np.ones((n_angles, n_t1, n_t2))
        c2_scaled = np.ones((n_angles, n_t1, n_t2))
        c2_solver = np.ones((n_angles, n_t1, n_t2)) * 4
        per_angle_scaling = np.ones((n_angles, 2))
        per_angle_scaling_solver = np.ones((n_angles, 2))
        residuals = np.zeros((n_angles, n_t1, n_t2))
        residuals_norm = np.zeros((n_angles, n_t1, n_t2))
        t1 = np.linspace(0.1, 1.0, n_t1)
        t2 = np.linspace(0.1, 1.0, n_t2)
        q = 0.05

        save_nlsq_npz_file(
            phi_angles,
            c2_exp,
            c2_raw,
            c2_scaled,
            c2_solver,
            per_angle_scaling,
            per_angle_scaling_solver,
            residuals,
            residuals_norm,
            t1,
            t2,
            q,
            tmp_path,
        )

        data = np.load(tmp_path / "fitted_data.npz")
        assert "c2_solver_scaled" in data
        assert np.allclose(data["c2_solver_scaled"], c2_solver)
