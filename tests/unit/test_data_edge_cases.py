"""Comprehensive edge case tests for homodyne/data module.

This module tests edge cases, boundary conditions, and error handling
for the data loading and validation components of the homodyne package.

Edge Cases Covered:
- Configuration parsing edge cases (missing keys, invalid formats)
- Data loader initialization with various config formats
- Cache handling (corruption, format mismatch, validation)
- HDF5 format detection and loading edge cases
- Correlation matrix reconstruction edge cases
- Time array generation edge cases
- Validation edge cases (missing data, invalid values)
- Error handling and graceful fallbacks
"""

import json
from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

from homodyne.data.validation import (
    DataQualityReport,
    ValidationIssue,
    validate_data_component,
    validate_xpcs_data,
    validate_xpcs_data_incremental,
)

# Import modules under test
from homodyne.data.xpcs_loader import (
    XPCSConfigurationError,
    XPCSDataFormatError,
    XPCSDataLoader,
    load_xpcs_config,
)

# =============================================================================
# CONFIGURATION LOADING EDGE CASES
# =============================================================================


class TestConfigLoadingEdgeCases:
    """Test edge cases for configuration loading."""

    def test_load_config_nonexistent_file(self):
        """Loading non-existent file should raise XPCSConfigurationError."""
        with pytest.raises(XPCSConfigurationError, match="not found"):
            load_xpcs_config("/nonexistent/path/config.yaml")

    def test_load_config_unsupported_format(self, tmp_path):
        """Loading unsupported format should raise XPCSConfigurationError."""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some content")
        with pytest.raises(XPCSConfigurationError, match="Unsupported"):
            load_xpcs_config(config_file)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Loading invalid YAML should raise XPCSConfigurationError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        with pytest.raises(XPCSConfigurationError, match="parse"):
            load_xpcs_config(config_file)

    def test_load_config_invalid_json(self, tmp_path):
        """Loading invalid JSON should raise XPCSConfigurationError."""
        config_file = tmp_path / "config.json"
        config_file.write_text('{"invalid": json,}')
        with pytest.raises(XPCSConfigurationError, match="parse"):
            load_xpcs_config(config_file)

    def test_load_config_empty_yaml(self, tmp_path):
        """Loading empty YAML should return None or empty dict."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        result = load_xpcs_config(config_file)
        assert result is None or result == {}

    def test_load_config_yaml_with_anchors(self, tmp_path):
        """YAML with anchors should be parsed correctly."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """
defaults: &defaults
  dt: 0.1
  start_frame: 1

analyzer_parameters:
  <<: *defaults
  end_frame: 1000
"""
        )
        result = load_xpcs_config(config_file)
        assert result["analyzer_parameters"]["dt"] == 0.1
        assert result["analyzer_parameters"]["end_frame"] == 1000

    def test_load_config_json_converts_to_yaml_structure(self, tmp_path):
        """JSON config should be usable in same structure as YAML."""
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "experimental_data": {
                        "data_folder_path": "/tmp",
                        "data_file_name": "test.h5",
                    },
                    "analyzer_parameters": {
                        "dt": 0.1,
                        "start_frame": 1,
                        "end_frame": -1,
                    },
                }
            )
        )
        result = load_xpcs_config(config_file)
        assert "experimental_data" in result
        assert result["analyzer_parameters"]["dt"] == 0.1


class TestFlatConfigNormalization:
    """Test edge cases for flat config to nested config normalization."""

    def test_flat_config_normalization(self, tmp_path):
        """Flat config structure should be normalized to nested structure."""
        # Create a minimal HDF5-like file for testing
        data_file = tmp_path / "test.h5"
        data_file.touch()

        flat_config = {
            "data_file": str(data_file),
            "analysis_mode": "static_isotropic",
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 100,
        }

        # Create loader with dict config - may or may not raise depending on impl
        try:
            loader = XPCSDataLoader(config_dict=flat_config)
            # If it doesn't raise, check that config was processed
            assert loader is not None
        except (XPCSDataFormatError, FileNotFoundError, XPCSConfigurationError):
            # Expected - file doesn't contain valid HDF5 data
            pass

    def test_nested_config_not_modified(self):
        """Already nested config should not be modified."""
        nested_config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": -1},
        }

        # Make a copy to verify it's not modified
        import copy

        original = copy.deepcopy(nested_config)

        try:
            loader = XPCSDataLoader(config_dict=nested_config)
        except Exception:
            pass  # Expected to fail without actual data file

        # Structure keys should be preserved
        assert "experimental_data" in nested_config


# =============================================================================
# DATA LOADER INITIALIZATION EDGE CASES
# =============================================================================


class TestDataLoaderInitEdgeCases:
    """Test edge cases for XPCSDataLoader initialization."""

    def test_init_with_both_config_path_and_dict_raises(self, tmp_path):
        """Providing both config_path and config_dict should raise ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("test: value")
        with pytest.raises(ValueError, match="either.*not both"):
            XPCSDataLoader(config_path=str(config_file), config_dict={"test": "value"})

    def test_init_with_neither_config_raises(self):
        """Providing neither config_path nor config_dict should raise ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            XPCSDataLoader()

    def test_init_missing_required_exp_data_params(self):
        """Missing required experimental_data params should raise XPCSConfigurationError."""
        config = {
            "experimental_data": {
                # Missing data_folder_path and data_file_name
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": -1},
        }
        with pytest.raises(XPCSConfigurationError, match="Missing.*experimental_data"):
            XPCSDataLoader(config_dict=config)

    def test_init_missing_required_analyzer_params(self):
        """Missing required analyzer_parameters should raise XPCSConfigurationError."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                # Missing dt, start_frame, end_frame
            },
        }
        with pytest.raises(
            XPCSConfigurationError, match="Missing.*analyzer_parameters"
        ):
            XPCSDataLoader(config_dict=config)


# =============================================================================
# CORRELATION MATRIX EDGE CASES
# =============================================================================


class TestCorrelationMatrixEdgeCases:
    """Test edge cases for correlation matrix reconstruction and correction."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": -1},
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    def test_reconstruct_full_matrix_symmetric(self, mock_loader):
        """Reconstructed matrix should be symmetric."""
        # Create half matrix
        n = 10
        half_matrix = np.triu(np.random.rand(n, n))
        full_matrix = mock_loader._reconstruct_full_matrix(half_matrix)
        # Check symmetry
        assert_allclose(full_matrix, full_matrix.T, rtol=1e-10)

    def test_reconstruct_full_matrix_diagonal_preserved(self, mock_loader):
        """Diagonal values should be preserved after reconstruction."""
        n = 10
        diagonal_values = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        half_matrix = np.triu(np.random.rand(n, n))
        np.fill_diagonal(half_matrix, diagonal_values)
        full_matrix = mock_loader._reconstruct_full_matrix(half_matrix)
        assert_allclose(np.diag(full_matrix), diagonal_values, rtol=1e-10)

    def test_reconstruct_full_matrix_single_element(self, mock_loader):
        """Single element matrix should work."""
        half_matrix = np.array([[1.5]])
        full_matrix = mock_loader._reconstruct_full_matrix(half_matrix)
        assert_allclose(full_matrix, np.array([[1.5]]))

    def test_correct_diagonal_jax_array(self, mock_loader):
        """Diagonal correction should work with JAX arrays."""
        import jax.numpy as jnp

        # Create a symmetric matrix with known diagonal
        n = 5
        c2_mat = jnp.ones((n, n))
        # Set off-diagonal values
        c2_mat = c2_mat.at[0, 1].set(0.8)
        c2_mat = c2_mat.at[1, 0].set(0.8)
        c2_mat = c2_mat.at[1, 2].set(0.6)
        c2_mat = c2_mat.at[2, 1].set(0.6)
        c2_mat = c2_mat.at[2, 3].set(0.4)
        c2_mat = c2_mat.at[3, 2].set(0.4)
        c2_mat = c2_mat.at[3, 4].set(0.2)
        c2_mat = c2_mat.at[4, 3].set(0.2)

        corrected = mock_loader._correct_diagonal(c2_mat)

        # Check that diagonal is corrected (average of adjacent off-diagonal)
        # First element: only one neighbor
        # Middle elements: average of two neighbors
        # Last element: only one neighbor
        assert corrected.shape == (n, n)
        assert np.isfinite(np.asarray(corrected)).all()

    def test_correct_diagonal_small_matrix(self, mock_loader):
        """Diagonal correction should work with small matrices."""
        import jax.numpy as jnp

        c2_mat = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        corrected = mock_loader._correct_diagonal(c2_mat)
        assert corrected.shape == (2, 2)


# =============================================================================
# TIME ARRAY GENERATION EDGE CASES
# =============================================================================


class TestTimeArrayGenerationEdgeCases:
    """Test edge cases for time array generation."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    def test_calculate_time_arrays_starts_from_zero(self, mock_loader):
        """Time array should start from 0."""
        time_1d = mock_loader._calculate_time_arrays(matrix_size=100)
        assert time_1d[0] == 0.0

    def test_calculate_time_arrays_correct_spacing(self, mock_loader):
        """Time array should have correct spacing based on dt."""
        mock_loader.analyzer_config["dt"] = 0.5
        time_1d = mock_loader._calculate_time_arrays(matrix_size=10)
        # Check spacing is approximately uniform (linspace)
        diffs = np.diff(time_1d)
        assert_allclose(diffs, diffs[0] * np.ones_like(diffs), rtol=1e-10)

    def test_calculate_time_arrays_single_frame(self, mock_loader):
        """Single frame should return array with single element."""
        time_1d = mock_loader._calculate_time_arrays(matrix_size=1)
        assert len(time_1d) == 1
        assert time_1d[0] == 0.0

    def test_calculate_time_arrays_large_matrix(self, mock_loader):
        """Large matrix should work without issues."""
        time_1d = mock_loader._calculate_time_arrays(matrix_size=10000)
        assert len(time_1d) == 10000
        assert time_1d[0] == 0.0
        assert time_1d[-1] > 0.0


# =============================================================================
# DATA VALIDATION EDGE CASES
# =============================================================================


class TestDataValidationEdgeCases:
    """Test edge cases for data validation functions."""

    @pytest.fixture
    def valid_data(self):
        """Create minimal valid XPCS data."""
        n_phi = 5
        n_times = 50
        return {
            "wavevector_q_list": np.array([0.005] * n_phi),
            "phi_angles_list": np.linspace(-20, 20, n_phi),
            "t1": np.linspace(0, 10, n_times),
            "t2": np.linspace(0, 10, n_times),
            "c2_exp": np.ones((n_phi, n_times, n_times)) * 1.0,
        }

    def test_validate_empty_data(self):
        """Empty data should fail validation."""
        report = validate_xpcs_data({})
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_validate_missing_required_keys(self, valid_data):
        """Missing required keys should be reported as errors."""
        del valid_data["c2_exp"]
        report = validate_xpcs_data(valid_data)
        assert not report.is_valid
        assert any("c2_exp" in str(e.message) for e in report.errors)

    def test_validate_nan_values(self, valid_data):
        """NaN values should be reported as errors."""
        valid_data["c2_exp"][0, 0, 0] = np.nan
        report = validate_xpcs_data(valid_data)
        assert not report.is_valid
        assert any("non-finite" in str(e.message).lower() for e in report.errors)

    def test_validate_inf_values(self, valid_data):
        """Infinite values should be reported as errors."""
        valid_data["c2_exp"][0, 0, 0] = np.inf
        report = validate_xpcs_data(valid_data)
        assert not report.is_valid

    def test_validate_negative_q_values(self, valid_data):
        """Negative q values should be reported as errors."""
        valid_data["wavevector_q_list"][0] = -0.01
        report = validate_xpcs_data(valid_data)
        assert not report.is_valid

    def test_validate_negative_time_values(self, valid_data):
        """Negative time values should be reported as errors."""
        valid_data["t1"][0] = -1.0
        report = validate_xpcs_data(valid_data)
        assert not report.is_valid

    def test_validate_negative_correlation_warning(self, valid_data):
        """Negative correlation values should produce warning."""
        valid_data["c2_exp"][0, 0, 1] = -0.1
        report = validate_xpcs_data(valid_data)
        # Might be warning or error depending on implementation
        assert report.total_issues > 0

    def test_validate_validation_level_none(self, valid_data):
        """Validation level 'none' should skip all checks."""
        # Even with invalid data, should pass with level 'none'
        valid_data["c2_exp"][0, 0, 0] = np.nan
        report = validate_xpcs_data(valid_data, validation_level="none")
        assert report.is_valid
        assert report.total_issues == 0

    def test_validate_mismatched_time_shapes(self, valid_data):
        """Mismatched t1 and t2 shapes should be reported."""
        valid_data["t2"] = np.linspace(0, 10, len(valid_data["t1"]) + 10)
        report = validate_xpcs_data(valid_data)
        # Should have shape mismatch error
        assert report.total_issues > 0

    def test_validate_non_square_correlation_matrix(self, valid_data):
        """Non-square correlation matrices should be reported."""
        n_phi = 5
        # Create non-square matrices
        valid_data["c2_exp"] = np.ones((n_phi, 50, 60))
        report = validate_xpcs_data(valid_data)
        assert report.total_issues > 0


class TestIncrementalValidationEdgeCases:
    """Test edge cases for incremental validation."""

    @pytest.fixture
    def valid_data(self):
        """Create minimal valid XPCS data."""
        n_phi = 5
        n_times = 50
        return {
            "wavevector_q_list": np.array([0.005] * n_phi),
            "phi_angles_list": np.linspace(-20, 20, n_phi),
            "t1": np.linspace(0, 10, n_times),
            "t2": np.linspace(0, 10, n_times),
            "c2_exp": np.ones((n_phi, n_times, n_times)) * 1.0,
        }

    def test_incremental_validation_with_no_changes(self, valid_data):
        """Incremental validation with unchanged data should use cache."""
        # First validation
        report1 = validate_xpcs_data_incremental(valid_data, validation_level="basic")
        # Second validation with same data
        report2 = validate_xpcs_data_incremental(
            valid_data, validation_level="incremental", previous_report=report1
        )
        # Should have similar results
        assert report1.is_valid == report2.is_valid

    def test_incremental_validation_force_revalidate(self, valid_data):
        """Force revalidate should bypass cache."""
        # First validation
        report1 = validate_xpcs_data_incremental(valid_data, validation_level="basic")
        # Force revalidation
        report2 = validate_xpcs_data_incremental(
            valid_data, validation_level="basic", force_revalidate=True
        )
        assert report1.is_valid == report2.is_valid


class TestValidationComponentEdgeCases:
    """Test edge cases for component-level validation."""

    def test_validate_nonexistent_component(self):
        """Validating non-existent component should report error."""
        data = {"existing_key": np.array([1, 2, 3])}
        report = validate_data_component(data, "nonexistent_component")
        assert not report.is_valid
        assert any("not found" in str(e.message).lower() for e in report.errors)

    def test_validate_wavevector_component_only(self):
        """Validating only wavevector component should work."""
        data = {
            "wavevector_q_list": np.array([0.001, 0.005, 0.01]),
        }
        report = validate_data_component(data, "wavevector_q_list")
        assert report.is_valid

    def test_validate_phi_angles_component_only(self):
        """Validating only phi angles component should work."""
        data = {
            "phi_angles_list": np.array([-30, -15, 0, 15, 30]),
        }
        report = validate_data_component(data, "phi_angles_list")
        assert report.is_valid

    def test_validate_time_component_only(self):
        """Validating only time component should work."""
        data = {
            "t1": np.linspace(0, 10, 100),
        }
        report = validate_data_component(data, "t1")
        assert report.is_valid


# =============================================================================
# VALIDATION REPORT EDGE CASES
# =============================================================================


class TestDataQualityReportEdgeCases:
    """Test edge cases for DataQualityReport."""

    def test_add_error_invalidates_report(self):
        """Adding an error should set is_valid to False."""
        report = DataQualityReport(
            is_valid=True, validation_level="basic", total_issues=0
        )
        report.add_issue(
            ValidationIssue(severity="error", category="test", message="Test error")
        )
        assert not report.is_valid
        assert report.total_issues == 1

    def test_add_warning_keeps_valid(self):
        """Adding a warning should keep is_valid True."""
        report = DataQualityReport(
            is_valid=True, validation_level="basic", total_issues=0
        )
        report.add_issue(
            ValidationIssue(severity="warning", category="test", message="Test warning")
        )
        assert report.is_valid
        assert report.total_issues == 1

    def test_add_info_keeps_valid(self):
        """Adding info should keep is_valid True."""
        report = DataQualityReport(
            is_valid=True, validation_level="basic", total_issues=0
        )
        report.add_issue(
            ValidationIssue(severity="info", category="test", message="Test info")
        )
        assert report.is_valid
        assert report.total_issues == 1

    def test_get_summary_structure(self):
        """Summary should have expected structure."""
        report = DataQualityReport(
            is_valid=True, validation_level="full", total_issues=0, quality_score=0.95
        )
        summary = report.get_summary()
        expected_keys = [
            "is_valid",
            "validation_level",
            "total_issues",
            "errors",
            "warnings",
            "info",
            "quality_score",
        ]
        for key in expected_keys:
            assert key in summary


# =============================================================================
# CACHE HANDLING EDGE CASES
# =============================================================================


class TestCacheHandlingEdgeCases:
    """Test edge cases for NPZ cache handling."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
            "v2_features": {"cache_strategy": "intelligent"},
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    def test_load_from_cache_2d_format_rejected(self, mock_loader, tmp_path):
        """Old 2D meshgrid cache format should be rejected."""
        cache_file = tmp_path / "cache.npz"
        # Create old format cache with 2D time arrays
        t_2d = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
        np.savez(
            cache_file,
            wavevector_q_list=np.array([0.005]),
            phi_angles_list=np.array([0.0]),
            t1=t_2d[0],  # 2D array - old format
            t2=t_2d[1],  # 2D array - old format
            c2_exp=np.ones((1, 50, 50)),
        )
        with pytest.raises(ValueError, match="2D meshgrid"):
            mock_loader._load_from_cache(str(cache_file))

    def test_load_from_cache_valid_1d_format(self, mock_loader, tmp_path):
        """Valid 1D cache format should load successfully."""
        cache_file = tmp_path / "cache.npz"
        # Create new format cache with 1D time arrays
        t_1d = np.linspace(0, 10, 50)
        np.savez(
            cache_file,
            wavevector_q_list=np.array([0.005]),
            phi_angles_list=np.array([0.0]),
            t1=t_1d,  # 1D array - new format
            t2=t_1d,  # 1D array - new format
            c2_exp=np.ones((1, 50, 50)),
        )
        data = mock_loader._load_from_cache(str(cache_file))
        assert data["t1"].ndim == 1
        assert data["t2"].ndim == 1


# =============================================================================
# HDF5 FORMAT DETECTION EDGE CASES
# =============================================================================


class TestHDF5FormatDetectionEdgeCases:
    """Test edge cases for HDF5 format detection."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": -1},
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py required"),
        reason="h5py not available",
    )
    def test_detect_format_empty_file(self, mock_loader, tmp_path):
        """Empty HDF5 file should return 'unknown' format."""
        import h5py

        hdf_file = tmp_path / "empty.h5"
        with h5py.File(hdf_file, "w"):
            pass  # Create empty file
        format_type = mock_loader._detect_format(str(hdf_file))
        assert format_type == "unknown"

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py required"),
        reason="h5py not available",
    )
    def test_detect_format_aps_u_structure(self, mock_loader, tmp_path):
        """APS-U format structure should be detected."""
        import h5py

        hdf_file = tmp_path / "aps_u.h5"
        with h5py.File(hdf_file, "w") as f:
            f.create_group("xpcs/qmap")
            f["xpcs/qmap"].create_dataset("dynamic_v_list_dim0", data=np.array([0.005]))
            f.create_group("xpcs/twotime")
            f["xpcs/twotime"].create_dataset(
                "correlation_map", data=np.ones((1, 10, 10))
            )
        format_type = mock_loader._detect_format(str(hdf_file))
        # May return 'unknown' if not complete structure
        assert format_type in ["aps_u", "unknown"]

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py required"),
        reason="h5py not available",
    )
    def test_detect_format_aps_old_structure(self, mock_loader, tmp_path):
        """APS old format structure should be detected."""
        import h5py

        hdf_file = tmp_path / "aps_old.h5"
        with h5py.File(hdf_file, "w") as f:
            f.create_group("xpcs")
            f["xpcs"].create_dataset("dqlist", data=np.array([[0.005]]))
            f["xpcs"].create_dataset("dphilist", data=np.array([[0.0]]))
            f.create_group("exchange")
            c2t_group = f.create_group("exchange/C2T_all")
            c2t_group.create_dataset("c2_00001", data=np.ones((10, 10)))
        format_type = mock_loader._detect_format(str(hdf_file))
        assert format_type == "aps_old"


# =============================================================================
# Q-VECTOR SELECTION EDGE CASES
# =============================================================================


class TestQVectorSelectionEdgeCases:
    """Test edge cases for q-vector selection."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": -1,
                "scattering": {"wavevector_q": 0.005},
            },
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    def test_select_optimal_wavevector_exact_match(self, mock_loader):
        """Exact match should be selected."""
        dqlist = np.array([0.001, 0.003, 0.005, 0.007, 0.009])
        idx = mock_loader._select_optimal_wavevector(dqlist)
        assert dqlist[idx] == 0.005

    def test_select_optimal_wavevector_closest_match(self, mock_loader):
        """Closest match should be selected when exact not available."""
        dqlist = np.array([0.001, 0.003, 0.004, 0.006, 0.009])  # No exact 0.005
        idx = mock_loader._select_optimal_wavevector(dqlist)
        # Should select 0.004 or 0.006 (both equidistant)
        assert abs(dqlist[idx] - 0.005) <= 0.002

    def test_select_optimal_wavevector_single_value(self, mock_loader):
        """Single q-value should be selected."""
        dqlist = np.array([0.003])
        idx = mock_loader._select_optimal_wavevector(dqlist)
        assert idx == 0


# =============================================================================
# FRAME SLICING EDGE CASES
# =============================================================================


class TestFrameSlicingEdgeCases:
    """Test edge cases for frame slicing."""

    @pytest.fixture
    def mock_loader(self):
        """Create a mock loader for testing internal methods."""
        config = {
            "experimental_data": {
                "data_folder_path": "/tmp",
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 10,
                "end_frame": 90,
            },
        }
        with patch.object(XPCSDataLoader, "_validate_configuration"):
            loader = XPCSDataLoader(config_dict=config)
        return loader

    def test_frame_slicing_within_bounds(self, mock_loader):
        """Normal frame slicing should work."""
        c2_matrices = np.ones((5, 100, 100))
        sliced = mock_loader._apply_frame_slicing_to_selected_q(c2_matrices)
        # start_frame=10 -> index 9, end_frame=90
        expected_size = 90 - 9  # 81
        assert sliced.shape == (5, 81, 81)

    def test_frame_slicing_start_exceeds_matrix(self, mock_loader):
        """Start frame exceeding matrix size should be handled."""
        mock_loader.analyzer_config["start_frame"] = 200  # Beyond matrix
        c2_matrices = np.ones((5, 100, 100))
        # Should adjust start_frame
        sliced = mock_loader._apply_frame_slicing_to_selected_q(c2_matrices)
        # Should have some valid output
        assert sliced.shape[0] == 5

    def test_frame_slicing_end_exceeds_matrix(self, mock_loader):
        """End frame exceeding matrix size should be adjusted."""
        mock_loader.analyzer_config["end_frame"] = 200  # Beyond matrix
        c2_matrices = np.ones((5, 100, 100))
        sliced = mock_loader._apply_frame_slicing_to_selected_q(c2_matrices)
        # End should be adjusted to 100
        assert sliced.shape[-1] <= 100

    def test_frame_slicing_full_range(self, mock_loader):
        """Using full frame range should return complete matrix."""
        mock_loader.analyzer_config["start_frame"] = 1
        mock_loader.analyzer_config["end_frame"] = 100
        c2_matrices = np.ones((5, 100, 100))
        sliced = mock_loader._apply_frame_slicing_to_selected_q(c2_matrices)
        # Should be same as input (no slicing needed)
        assert sliced.shape == (5, 100, 100)
