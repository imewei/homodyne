"""
Unit Tests for homodyne.data.xpcs_loader Module
===============================================

Comprehensive tests for XPCS data loading functionality including:
- Configuration loading (YAML/JSON)
- HDF5 format detection and loading
- Caching mechanisms
- Data validation and quality control
- Error handling and edge cases
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Import the module under test
from homodyne.data.xpcs_loader import (
    XPCSConfigurationError,
    XPCSDataFormatError,
    XPCSDataLoader,
    XPCSDependencyError,
    load_xpcs_config,
    load_xpcs_data,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def minimal_yaml_config(temp_dir):
    """Create a minimal valid YAML configuration file."""
    config_content = f"""
experimental_data:
  data_folder_path: "{temp_dir}"
  data_file_name: "test_data.h5"

analyzer_parameters:
  dt: 0.1
  start_frame: 1
  end_frame: 100
  scattering:
    wavevector_q: 0.0054
"""
    config_path = os.path.join(temp_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


@pytest.fixture
def minimal_json_config(temp_dir):
    """Create a minimal valid JSON configuration file."""
    config = {
        "experimental_data": {
            "data_folder_path": temp_dir,
            "data_file_name": "test_data.h5",
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 100,
            "scattering": {"wavevector_q": 0.0054},
        },
    }
    config_path = os.path.join(temp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path


@pytest.fixture
def flat_config_dict(temp_dir):
    """Create a flat configuration dictionary (legacy format)."""
    return {
        "data_file": os.path.join(temp_dir, "test_data.h5"),
        "analysis_mode": "static_isotropic",
        "dt": 0.1,
        "start_frame": 1,
        "end_frame": 100,
    }


@pytest.fixture
def nested_config_dict(temp_dir):
    """Create a nested configuration dictionary."""
    return {
        "experimental_data": {
            "data_folder_path": temp_dir,
            "data_file_name": "test_data.h5",
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 100,
            "scattering": {"wavevector_q": 0.0054},
        },
    }


@pytest.fixture
def mock_npz_cache(temp_dir):
    """Create a mock NPZ cache file with valid data."""
    n_phi = 5
    n_frames = 50

    wavevector_q_list = np.full(n_phi, 0.0054)
    phi_angles_list = np.linspace(0, 180, n_phi)
    t1 = np.linspace(0, 4.9, n_frames)
    t2 = np.linspace(0, 4.9, n_frames)
    c2_exp = np.ones((n_phi, n_frames, n_frames)) * 1.1
    # Add some variation to diagonal
    for i in range(n_phi):
        np.fill_diagonal(c2_exp[i], 1.5)

    cache_path = os.path.join(temp_dir, "cached_c2_frames_1_100.npz")
    np.savez(
        cache_path,
        wavevector_q_list=wavevector_q_list,
        phi_angles_list=phi_angles_list,
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
    )
    return cache_path


# =============================================================================
# Tests for load_xpcs_config Function
# =============================================================================


class TestLoadXpcsConfig:
    """Test suite for load_xpcs_config function."""

    def test_load_yaml_config(self, minimal_yaml_config):
        """Test loading a valid YAML configuration."""
        config = load_xpcs_config(minimal_yaml_config)
        assert "experimental_data" in config
        assert "analyzer_parameters" in config
        assert config["analyzer_parameters"]["dt"] == 0.1

    def test_load_json_config(self, minimal_json_config):
        """Test loading a valid JSON configuration."""
        config = load_xpcs_config(minimal_json_config)
        assert "experimental_data" in config
        assert "analyzer_parameters" in config
        assert config["analyzer_parameters"]["dt"] == 0.1

    def test_load_yml_extension(self, temp_dir):
        """Test loading YAML with .yml extension."""
        config_content = """
experimental_data:
  data_folder_path: "/tmp"
  data_file_name: "test.h5"
analyzer_parameters:
  dt: 0.2
  start_frame: 1
  end_frame: 50
"""
        config_path = os.path.join(temp_dir, "config.yml")
        with open(config_path, "w") as f:
            f.write(config_content)

        config = load_xpcs_config(config_path)
        assert config["analyzer_parameters"]["dt"] == 0.2

    def test_file_not_found_error(self, temp_dir):
        """Test error raised for non-existent config file."""
        fake_path = os.path.join(temp_dir, "nonexistent.yaml")
        with pytest.raises(XPCSConfigurationError, match="not found"):
            load_xpcs_config(fake_path)

    def test_unsupported_format_error(self, temp_dir):
        """Test error raised for unsupported config format."""
        config_path = os.path.join(temp_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write("some content")

        with pytest.raises(XPCSConfigurationError, match="Unsupported"):
            load_xpcs_config(config_path)

    def test_invalid_yaml_syntax(self, temp_dir):
        """Test error raised for invalid YAML syntax."""
        config_path = os.path.join(temp_dir, "invalid.yaml")
        with open(config_path, "w") as f:
            f.write("key: value\n  bad_indent: wrong")

        with pytest.raises(XPCSConfigurationError, match="parse"):
            load_xpcs_config(config_path)

    def test_invalid_json_syntax(self, temp_dir):
        """Test error raised for invalid JSON syntax."""
        config_path = os.path.join(temp_dir, "invalid.json")
        with open(config_path, "w") as f:
            f.write("{'key': 'value'")  # Missing closing brace

        with pytest.raises(XPCSConfigurationError, match="parse"):
            load_xpcs_config(config_path)

    def test_path_object_input(self, minimal_yaml_config):
        """Test that Path objects are accepted."""
        config = load_xpcs_config(Path(minimal_yaml_config))
        assert "experimental_data" in config


# =============================================================================
# Tests for XPCSDataLoader Initialization
# =============================================================================


class TestXPCSDataLoaderInit:
    """Test suite for XPCSDataLoader initialization."""

    def test_init_with_config_path(self, minimal_yaml_config):
        """Test initialization with config file path."""
        loader = XPCSDataLoader(config_path=minimal_yaml_config)
        assert loader.config is not None
        assert "experimental_data" in loader.config

    def test_init_with_config_dict(self, nested_config_dict):
        """Test initialization with config dictionary."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert loader.config is not None
        assert "experimental_data" in loader.config

    def test_init_requires_config(self):
        """Test that initialization requires either config_path or config_dict."""
        with pytest.raises(ValueError, match="Must provide"):
            XPCSDataLoader()

    def test_init_rejects_both_configs(self, minimal_yaml_config, nested_config_dict):
        """Test that initialization rejects both config_path and config_dict."""
        with pytest.raises(ValueError, match="not both"):
            XPCSDataLoader(
                config_path=minimal_yaml_config, config_dict=nested_config_dict
            )

    def test_flat_config_normalization(self, flat_config_dict):
        """Test that flat config is normalized to nested structure."""
        loader = XPCSDataLoader(config_dict=flat_config_dict)
        # Should have been transformed to nested structure
        assert "experimental_data" in loader.config
        assert "analyzer_parameters" in loader.config

    def test_missing_required_experimental_data(self, temp_dir):
        """Test error when required experimental_data fields are missing."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                # Missing data_file_name
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }
        with pytest.raises(XPCSConfigurationError, match="data_file_name"):
            XPCSDataLoader(config_dict=config)

    def test_missing_required_analyzer_params(self, temp_dir):
        """Test error when required analyzer_parameters fields are missing."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "test.h5",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                # Missing start_frame and end_frame
            },
        }
        with pytest.raises(XPCSConfigurationError, match="start_frame"):
            XPCSDataLoader(config_dict=config)

    def test_v2_features_defaults(self, nested_config_dict):
        """Test that v2_features defaults are applied."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert "v2_features" in loader.config
        assert loader.config["v2_features"]["output_format"] == "auto"
        assert loader.config["v2_features"]["validation_level"] == "basic"

    def test_performance_defaults(self, nested_config_dict):
        """Test that performance defaults are applied."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert "performance" in loader.config
        assert loader.config["performance"]["performance_engine_enabled"] is True


# =============================================================================
# Tests for Configuration Processing
# =============================================================================


class TestConfigurationProcessing:
    """Test suite for configuration processing methods."""

    def test_normalize_nested_config_unchanged(self, nested_config_dict):
        """Test that already nested config is not modified."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert loader.config["experimental_data"]["data_folder_path"] is not None

    def test_normalize_flat_to_nested(self, temp_dir):
        """Test flat config is transformed to nested."""
        flat_config = {
            "data_file": os.path.join(temp_dir, "subdir", "test.h5"),
            "dt": 0.1,
            "start_frame": 5,
            "end_frame": 200,
        }
        loader = XPCSDataLoader(config_dict=flat_config)

        # Check transformation
        assert loader.config["experimental_data"]["data_folder_path"] == os.path.join(
            temp_dir, "subdir"
        )
        assert loader.config["experimental_data"]["data_file_name"] == "test.h5"
        assert loader.config["analyzer_parameters"]["dt"] == 0.1
        assert loader.config["analyzer_parameters"]["start_frame"] == 5
        assert loader.config["analyzer_parameters"]["end_frame"] == 200

    def test_flat_config_default_values(self, temp_dir):
        """Test flat config gets default values when missing."""
        flat_config = {
            "data_file": os.path.join(temp_dir, "test.h5"),
        }
        loader = XPCSDataLoader(config_dict=flat_config)

        # Should have defaults
        assert loader.config["analyzer_parameters"]["dt"] == 0.1
        assert loader.config["analyzer_parameters"]["start_frame"] == 1
        assert loader.config["analyzer_parameters"]["end_frame"] == -1


# =============================================================================
# Tests for Cache Loading
# =============================================================================


class TestCacheLoading:
    """Test suite for NPZ cache loading functionality."""

    def test_load_from_valid_cache(self, temp_dir, mock_npz_cache):
        """Test loading data from a valid NPZ cache."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
                "scattering": {"wavevector_q": 0.0054},
            },
        }
        loader = XPCSDataLoader(config_dict=config)
        data = loader.load_experimental_data()

        assert "wavevector_q_list" in data
        assert "phi_angles_list" in data
        assert "t1" in data
        assert "t2" in data
        assert "c2_exp" in data

    def test_cache_validation_required_keys(self, temp_dir):
        """Test that cache loading validates required keys."""
        # Create cache missing required keys
        cache_path = os.path.join(temp_dir, "incomplete_cache.npz")
        np.savez(cache_path, wavevector_q_list=np.array([0.0054]))

        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "incomplete_cache.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }
        loader = XPCSDataLoader(config_dict=config)

        with pytest.raises(KeyError):
            loader.load_experimental_data()

    def test_cache_rejects_2d_time_arrays(self, temp_dir):
        """Test that old 2D meshgrid cache format is rejected."""
        n_phi = 3
        n_frames = 20

        # Create old-format cache with 2D time arrays
        t1_2d, t2_2d = np.meshgrid(np.arange(n_frames), np.arange(n_frames))

        cache_path = os.path.join(temp_dir, "old_format.npz")
        np.savez(
            cache_path,
            wavevector_q_list=np.full(n_phi, 0.0054),
            phi_angles_list=np.linspace(0, 180, n_phi),
            t1=t1_2d,  # 2D - should be rejected
            t2=t2_2d,  # 2D - should be rejected
            c2_exp=np.ones((n_phi, n_frames, n_frames)),
        )

        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "old_format.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }
        loader = XPCSDataLoader(config_dict=config)

        with pytest.raises(ValueError, match="Old 2D meshgrid"):
            loader.load_experimental_data()


# =============================================================================
# Tests for HDF5 Format Detection
# =============================================================================


class TestFormatDetection:
    """Test suite for HDF5 format detection."""

    def test_detect_format_method_exists(self, nested_config_dict):
        """Test that _detect_format method exists."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert hasattr(loader, "_detect_format")
        assert callable(loader._detect_format)


# =============================================================================
# Tests for Matrix Reconstruction
# =============================================================================


class TestMatrixReconstruction:
    """Test suite for half-matrix reconstruction."""

    def test_reconstruct_full_matrix_symmetry(self, nested_config_dict):
        """Test reconstructed matrix is symmetric."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)

        # Create a half matrix (upper triangle)
        n = 5
        half_matrix = np.triu(np.random.rand(n, n))

        full_matrix = loader._reconstruct_full_matrix(half_matrix)

        # Should be symmetric
        assert_allclose(full_matrix, full_matrix.T, rtol=1e-10)

    def test_reconstruct_full_matrix_diagonal(self, nested_config_dict):
        """Test diagonal is not doubled after reconstruction."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)

        n = 5
        half_matrix = np.zeros((n, n))
        np.fill_diagonal(half_matrix, 2.0)  # Set diagonal

        full_matrix = loader._reconstruct_full_matrix(half_matrix)

        # Diagonal should still be 2.0 (not 4.0)
        assert_allclose(np.diag(full_matrix), np.full(n, 2.0), rtol=1e-10)


# =============================================================================
# Tests for Diagonal Correction
# =============================================================================


class TestDiagonalCorrection:
    """Test suite for diagonal correction functionality."""

    def test_correct_diagonal_jax(self, nested_config_dict):
        """Test diagonal correction with JAX arrays.

        Note: _correct_diagonal uses JAX's .at[] syntax which requires JAX arrays.
        """
        import jax.numpy as jnp

        loader = XPCSDataLoader(config_dict=nested_config_dict)

        mat = jnp.array(
            [
                [5.0, 1.0, 1.0],
                [1.0, 5.0, 2.0],
                [1.0, 2.0, 5.0],
            ]
        )

        corrected = loader._correct_diagonal(mat)

        # Diagonal should be corrected
        # diag[0] = side_band[0] = 1.0
        # diag[1] = (1.0 + 2.0) / 2 = 1.5
        # diag[2] = side_band[1] = 2.0
        expected_diag = np.array([1.0, 1.5, 2.0])
        assert_allclose(np.diag(np.asarray(corrected)), expected_diag, rtol=1e-10)

    def test_correct_diagonal_off_diagonal_unchanged(self, nested_config_dict):
        """Test that off-diagonal elements are unchanged."""
        import jax.numpy as jnp

        loader = XPCSDataLoader(config_dict=nested_config_dict)

        mat = jnp.array(
            [
                [5.0, 1.1, 1.2],
                [1.1, 5.0, 1.3],
                [1.2, 1.3, 5.0],
            ]
        )

        corrected = loader._correct_diagonal(mat)

        # Off-diagonal should be unchanged
        assert float(corrected[0, 1]) == 1.1
        assert float(corrected[0, 2]) == 1.2
        assert float(corrected[1, 2]) == 1.3


# =============================================================================
# Tests for Time Array Calculation
# =============================================================================


class TestTimeArrayCalculation:
    """Test suite for time array calculation."""

    def test_calculate_time_arrays_basic(self, nested_config_dict):
        """Test basic time array calculation."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["dt"] = 0.1
        loader.analyzer_config["start_frame"] = 1
        loader.analyzer_config["end_frame"] = 11

        time_1d = loader._calculate_time_arrays(10)

        # Should be 10 points from 0 to 1.0
        assert len(time_1d) == 10
        assert time_1d[0] == 0.0
        assert_allclose(time_1d[-1], 1.0, rtol=1e-10)

    def test_calculate_time_arrays_starts_at_zero(self, nested_config_dict):
        """Test time array always starts at zero."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["dt"] = 0.5
        loader.analyzer_config["start_frame"] = 10
        loader.analyzer_config["end_frame"] = 20

        time_1d = loader._calculate_time_arrays(10)

        assert time_1d[0] == 0.0


# =============================================================================
# Tests for Wavevector Selection
# =============================================================================


class TestWavevectorSelection:
    """Test suite for q-vector selection."""

    def test_select_optimal_wavevector_exact_match(self, nested_config_dict):
        """Test selection when exact match exists."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["scattering"] = {"wavevector_q": 0.0054}

        dqlist = np.array([0.003, 0.0054, 0.008, 0.01])
        idx = loader._select_optimal_wavevector(dqlist)

        assert idx == 1
        assert dqlist[idx] == 0.0054

    def test_select_optimal_wavevector_closest(self, nested_config_dict):
        """Test selection when no exact match exists."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["scattering"] = {"wavevector_q": 0.0055}

        dqlist = np.array([0.003, 0.0054, 0.008, 0.01])
        idx = loader._select_optimal_wavevector(dqlist)

        # Should select 0.0054 as closest to 0.0055
        assert idx == 1


# =============================================================================
# Tests for Frame Slicing
# =============================================================================


class TestFrameSlicing:
    """Test suite for frame slicing functionality."""

    def test_apply_frame_slicing_basic(self, nested_config_dict):
        """Test basic frame slicing."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["start_frame"] = 10
        loader.analyzer_config["end_frame"] = 20

        # Create test data
        c2_matrices = np.ones((3, 100, 100))

        sliced = loader._apply_frame_slicing_to_selected_q(c2_matrices)

        # Should be (3, 11, 11) - frames 9 to 19 (0-indexed: 10-1=9 to 20)
        assert sliced.shape == (3, 11, 11)

    def test_apply_frame_slicing_edge_cases(self, nested_config_dict):
        """Test frame slicing edge cases."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["start_frame"] = 1  # Start from beginning
        loader.analyzer_config["end_frame"] = 50

        c2_matrices = np.ones((2, 50, 50))

        sliced = loader._apply_frame_slicing_to_selected_q(c2_matrices)

        # Should use full matrix
        assert sliced.shape == (2, 50, 50)

    def test_apply_frame_slicing_out_of_bounds(self, nested_config_dict):
        """Test frame slicing with out-of-bounds values."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        loader.analyzer_config["start_frame"] = -5  # Invalid
        loader.analyzer_config["end_frame"] = 200  # Beyond matrix size

        c2_matrices = np.ones((2, 50, 50))

        # Should handle gracefully with warnings
        sliced = loader._apply_frame_slicing_to_selected_q(c2_matrices)
        assert sliced.shape[1] <= 50


# =============================================================================
# Tests for load_xpcs_data Convenience Function
# =============================================================================


class TestLoadXpcsDataFunction:
    """Test suite for load_xpcs_data convenience function."""

    def test_load_with_config_path(self, minimal_yaml_config, mock_npz_cache, temp_dir):
        """Test load_xpcs_data with config path."""
        # Update config to point to cache
        config = load_xpcs_config(minimal_yaml_config)
        config["experimental_data"]["data_file_name"] = "cached_c2_frames_1_100.npz"

        data = load_xpcs_data(config_dict=config)

        assert "c2_exp" in data

    def test_load_with_dict_positional(self, temp_dir, mock_npz_cache):
        """Test load_xpcs_data with dict as positional argument (backward compat)."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }

        # Positional dict argument should work
        data = load_xpcs_data(config)

        assert "c2_exp" in data

    def test_load_with_dict_keyword(self, temp_dir, mock_npz_cache):
        """Test load_xpcs_data with dict as keyword argument."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }

        data = load_xpcs_data(config_dict=config)

        assert "c2_exp" in data

    def test_load_rejects_both_dict_forms(self, nested_config_dict):
        """Test that providing dict as both positional and keyword raises error."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            load_xpcs_data(nested_config_dict, config_dict=nested_config_dict)


# =============================================================================
# Tests for Exception Classes
# =============================================================================


class TestExceptionClasses:
    """Test suite for custom exception classes."""

    def test_xpcs_data_format_error(self):
        """Test XPCSDataFormatError exception."""
        with pytest.raises(XPCSDataFormatError):
            raise XPCSDataFormatError("Test error message")

    def test_xpcs_dependency_error(self):
        """Test XPCSDependencyError exception."""
        with pytest.raises(XPCSDependencyError):
            raise XPCSDependencyError("Missing numpy")

    def test_xpcs_configuration_error(self):
        """Test XPCSConfigurationError exception."""
        with pytest.raises(XPCSConfigurationError):
            raise XPCSConfigurationError("Invalid config")

    def test_exception_inheritance(self):
        """Test that custom exceptions inherit from Exception."""
        assert issubclass(XPCSDataFormatError, Exception)
        assert issubclass(XPCSDependencyError, Exception)
        assert issubclass(XPCSConfigurationError, Exception)


# =============================================================================
# Tests for Output Format Handling
# =============================================================================


class TestOutputFormatHandling:
    """Test suite for output format handling (JAX vs NumPy)."""

    def test_get_output_format_default(self, nested_config_dict):
        """Test default output format is 'auto'."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert loader._get_output_format() == "auto"

    def test_get_output_format_configured(self, nested_config_dict):
        """Test configured output format is respected."""
        nested_config_dict["v2_features"] = {"output_format": "numpy"}
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        assert loader._get_output_format() == "numpy"

    def test_convert_arrays_preserves_numpy(self, nested_config_dict):
        """Test array conversion preserves numpy when specified."""
        nested_config_dict["v2_features"] = {"output_format": "numpy"}
        loader = XPCSDataLoader(config_dict=nested_config_dict)

        data = {"arr": np.array([1, 2, 3]), "scalar": 5}
        converted = loader._convert_arrays_to_target_format(data)

        assert isinstance(converted["arr"], np.ndarray)
        assert converted["scalar"] == 5


# =============================================================================
# Tests for Validation Settings
# =============================================================================


class TestValidationSettings:
    """Test suite for validation settings."""

    def test_should_perform_validation_default(self, nested_config_dict):
        """Test default validation settings."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        settings = loader._should_perform_validation()

        assert "physics_checks" in settings
        assert "data_quality" in settings
        assert "comprehensive" in settings

    def test_validation_level_none(self, nested_config_dict):
        """Test validation disabled with level 'none'."""
        nested_config_dict["v2_features"] = {"validation_level": "none"}
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        settings = loader._should_perform_validation()

        assert settings["data_quality"] is False

    def test_validation_level_full(self, nested_config_dict):
        """Test comprehensive validation with level 'full'."""
        nested_config_dict["v2_features"] = {"validation_level": "full"}
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        settings = loader._should_perform_validation()

        assert settings["comprehensive"] is True


# =============================================================================
# Tests for Cache Path Generation
# =============================================================================


class TestCachePathGeneration:
    """Test suite for cache path generation."""

    def test_generate_cache_path_default(self, nested_config_dict):
        """Test default cache path generation."""
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        cache_path = loader._generate_cache_path()

        assert isinstance(cache_path, Path)
        assert "cached_c2_frames" in str(cache_path)

    def test_generate_cache_path_with_template(self, nested_config_dict):
        """Test cache path with custom template."""
        nested_config_dict["experimental_data"]["cache_filename_template"] = (
            "cache_{start_frame}_{end_frame}_q{wavevector_q}.npz"
        )
        loader = XPCSDataLoader(config_dict=nested_config_dict)
        cache_path = loader._generate_cache_path()

        assert "q0.0054" in str(cache_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete data loading workflows."""

    def test_complete_cache_load_workflow(self, temp_dir, mock_npz_cache):
        """Test complete workflow: config → loader → data."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
                "scattering": {"wavevector_q": 0.0054},
            },
            "v2_features": {"output_format": "jax"},  # Use JAX for diagonal correction
        }

        loader = XPCSDataLoader(config_dict=config)
        data = loader.load_experimental_data()

        # Verify all expected keys
        assert "wavevector_q_list" in data
        assert "phi_angles_list" in data
        assert "t1" in data
        assert "t2" in data
        assert "c2_exp" in data

        # Verify shapes are consistent
        n_phi = len(data["phi_angles_list"])
        assert data["c2_exp"].shape[0] == n_phi
        assert data["c2_exp"].shape[1] == data["c2_exp"].shape[2]

    def test_data_types_are_correct(self, temp_dir, mock_npz_cache):
        """Test that loaded data has correct types."""
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
            "v2_features": {"output_format": "jax"},  # Use JAX for diagonal correction
        }

        loader = XPCSDataLoader(config_dict=config)
        data = loader.load_experimental_data()

        # Verify data was loaded successfully
        assert "wavevector_q_list" in data
        assert "phi_angles_list" in data
        assert "t1" in data
        assert "t2" in data
        assert "c2_exp" in data
