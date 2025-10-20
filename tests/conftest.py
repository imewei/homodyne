"""
Pytest Configuration and Fixtures for Homodyne v2
==================================================

Shared fixtures, configuration, and test utilities for the entire test suite.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Handle JAX imports gracefully
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = np

# Handle optional dependencies
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "gpu: GPU acceleration tests")
    config.addinivalue_line("markers", "mcmc: MCMC statistical tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Slow tests (> 5 seconds)")
    config.addinivalue_line("markers", "requires_jax: Requires JAX installation")
    config.addinivalue_line("markers", "requires_gpu: Requires GPU hardware")


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location and requirements."""
    for item in items:
        # Mark tests based on directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)
        elif "gpu" in str(item.fspath):
            item.add_marker(pytest.mark.gpu)
            item.add_marker(pytest.mark.requires_gpu)
        elif "mcmc" in str(item.fspath):
            item.add_marker(pytest.mark.mcmc)
            item.add_marker(pytest.mark.slow)

        # Mark JAX-dependent tests
        if hasattr(item, "fixturenames") and any(
            "jax" in name for name in item.fixturenames
        ):
            item.add_marker(pytest.mark.requires_jax)


# ============================================================================
# Core Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def jax_backend():
    """JAX backend configuration for tests."""
    if not JAX_AVAILABLE:
        pytest.skip("JAX not available")

    # Configure JAX for testing
    original_platform = os.environ.get("JAX_PLATFORM_NAME", "")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"  # Use CPU for consistent testing

    yield jax

    # Restore original configuration
    if original_platform:
        os.environ["JAX_PLATFORM_NAME"] = original_platform
    else:
        os.environ.pop("JAX_PLATFORM_NAME", None)


@pytest.fixture
def numpy_backend():
    """NumPy backend for fallback tests."""
    return np


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config():
    """Basic test configuration dictionary."""
    return {
        "analysis_mode": "static_isotropic",
        "optimization": {
            "method": "nlsq",
            "lsq": {"max_iterations": 100, "tolerance": 1e-6},
        },
        "hardware": {"force_cpu": True, "gpu_memory_fraction": 0.8},
        "output": {"save_plots": False, "verbose": False},
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture
def synthetic_xpcs_data():
    """Generate synthetic XPCS data for testing."""
    n_times = 50
    n_angles = 36

    # Time arrays
    t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")

    # Angle array
    phi = np.linspace(0, 2 * np.pi, n_angles)

    # Generate synthetic correlation function
    tau = np.abs(t1 - t2) + 1e-6
    c2_exp = 1 + 0.5 * np.exp(-tau / 10.0)

    # Add realistic noise
    np.random.seed(42)  # Reproducible
    noise = 0.01 * np.random.randn(*c2_exp.shape)
    c2_exp += noise

    return {
        "t1": t1,
        "t2": t2,
        "phi_angles_list": phi,
        "c2_exp": c2_exp,
        "wavevector_q_list": np.array([0.01]),
        "sigma": np.ones_like(c2_exp) * 0.01,
    }


@pytest.fixture
def small_xpcs_data():
    """Small XPCS dataset for fast tests."""
    n_times = 10
    n_angles = 12

    t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")

    phi = np.linspace(0, 2 * np.pi, n_angles)
    tau = np.abs(t1 - t2) + 1e-6
    c2_exp = 1 + 0.3 * np.exp(-tau / 5.0)

    return {
        "t1": t1,
        "t2": t2,
        "phi_angles_list": phi,
        "c2_exp": c2_exp,
        "wavevector_q_list": np.array([0.015]),
        "sigma": np.ones_like(c2_exp) * 0.02,
    }


@pytest.fixture
def large_xpcs_data():
    """Large XPCS dataset for performance tests."""
    n_times = 200
    n_angles = 72

    t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")

    phi = np.linspace(0, 2 * np.pi, n_angles)
    tau = np.abs(t1 - t2) + 1e-6
    c2_exp = 1 + 0.4 * np.exp(-tau / 15.0)

    # Add structured noise
    np.random.seed(123)
    noise = 0.005 * np.random.randn(*c2_exp.shape)
    c2_exp += noise

    return {
        "t1": t1,
        "t2": t2,
        "phi_angles_list": phi,
        "c2_exp": c2_exp,
        "wavevector_q_list": np.array([0.008]),
        "sigma": np.ones_like(c2_exp) * 0.005,
    }


@pytest.fixture
def test_parameters():
    """Standard test parameters for optimization."""
    return {
        "offset": 1.0,
        "contrast": 0.5,
        "diffusion_coefficient": 0.1,
        "shear_rate": 0.0,
        "L": 1.0,
    }


# ============================================================================
# File Format Fixtures
# ============================================================================


@pytest.fixture
def mock_hdf5_file(temp_dir):
    """Create a mock HDF5 file for testing."""
    if not HAS_H5PY:
        pytest.skip("h5py not available")

    file_path = temp_dir / "test_data.h5"

    with h5py.File(file_path, "w") as f:
        # Mock APS format structure
        grp = f.create_group("exchange")
        grp.create_dataset("correlation", data=np.random.random((50, 50, 36)))
        grp.create_dataset("phi_angles", data=np.linspace(0, 2 * np.pi, 36))
        grp.create_dataset("wavevector_q", data=np.array([0.01]))
        grp.create_dataset("time_grid", data=np.arange(50))

    yield file_path


@pytest.fixture
def mock_yaml_config(temp_dir):
    """Create a mock YAML configuration file."""
    if not HAS_YAML:
        pytest.skip("yaml not available")

    config_path = temp_dir / "test_config.yaml"
    config = {
        "data_file": "test_data.h5",
        "analysis_mode": "static_isotropic",
        "output_directory": str(temp_dir),
        "optimization": {"method": "nlsq", "lsq": {"max_iterations": 50}},
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    yield config_path


# ============================================================================
# GPU Test Fixtures
# ============================================================================


@pytest.fixture
def gpu_available():
    """Check if GPU is available for testing."""
    if not JAX_AVAILABLE:
        return False

    try:
        gpu_devices = jax.devices("gpu")
        return len(gpu_devices) > 0
    except:
        return False


@pytest.fixture
def force_cpu():
    """Force CPU execution for consistent testing."""
    if JAX_AVAILABLE:
        original_platform = os.environ.get("JAX_PLATFORM_NAME", "")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

        yield

        if original_platform:
            os.environ["JAX_PLATFORM_NAME"] = original_platform
        else:
            os.environ.pop("JAX_PLATFORM_NAME", None)
    else:
        yield


# ============================================================================
# Mathematical Validation Fixtures
# ============================================================================


@pytest.fixture
def tolerance_params():
    """Standard tolerance parameters for numerical tests."""
    return {
        "rtol": 1e-6,
        "atol": 1e-8,
        "optimization_tol": 1e-6,
        "mcmc_tol": 0.05,  # Higher tolerance for stochastic methods
    }


@pytest.fixture
def physics_validation():
    """Physics validation parameters for correlation functions."""
    return {
        "min_correlation": 1.0,  # g2 >= 1.0
        "max_correlation": 5.0,  # Reasonable upper bound
        "symmetry_tol": 1e-6,  # Correlation matrix symmetry
        "causality_tol": 1e-6,  # Time-ordering constraints
    }


# ============================================================================
# Performance Test Fixtures
# ============================================================================


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        "min_rounds": 3,
        "max_time": 30.0,  # Maximum 30 seconds per benchmark
        "warmup_rounds": 1,
        "disable_gc": True,
    }


# ============================================================================
# Test Utilities
# ============================================================================


class TestUtils:
    """Utility functions for tests."""

    @staticmethod
    def assert_correlation_physics(c2_data, tolerance=1e-6):
        """Assert that correlation data satisfies physical constraints."""
        # g2 >= 1.0 (physical minimum)
        assert np.all(c2_data >= 1.0 - tolerance), "Correlation below physical minimum"

        # Finite values only
        assert np.all(np.isfinite(c2_data)), "Non-finite correlation values"

    @staticmethod
    def assert_array_properties(arr, expected_shape=None, dtype=None):
        """Assert basic array properties."""
        assert arr is not None, "Array is None"
        assert hasattr(arr, "shape"), "Object is not array-like"

        if expected_shape is not None:
            assert (
                arr.shape == expected_shape
            ), f"Shape mismatch: {arr.shape} != {expected_shape}"

        if dtype is not None:
            assert arr.dtype == dtype, f"Dtype mismatch: {arr.dtype} != {dtype}"

    @staticmethod
    def assert_optimization_result(result, expected_params=None, tolerance=0.1):
        """Assert optimization result validity."""
        assert hasattr(result, "parameters"), "Result missing parameters"
        assert hasattr(result, "chi_squared"), "Result missing chi_squared"

        if expected_params is not None:
            for key, expected_val in expected_params.items():
                if key in result.parameters:
                    actual_val = result.parameters[key]
                    assert (
                        abs(actual_val - expected_val) < tolerance
                    ), f"Parameter {key}: {actual_val} != {expected_val} ± {tolerance}"


@pytest.fixture
def test_utils():
    """Test utility functions."""
    return TestUtils
