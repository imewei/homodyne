"""
Pytest Configuration and Fixtures for Homodyne
==================================================

Shared fixtures, configuration, and test utilities for the entire test suite.
"""

import os
import sys
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

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    az = None


# ============================================================================
# Platform Detection
# ============================================================================


def is_linux():
    """Check if running on Linux."""
    return sys.platform.startswith("linux")


def is_macos():
    """Check if running on macOS."""
    return sys.platform == "darwin"


def is_windows():
    """Check if running on Windows."""
    return sys.platform == "win32"


# Skip decorators for platform-specific tests
skip_if_not_linux = pytest.mark.skipif(not is_linux(), reason="Test requires Linux OS")

skip_if_windows = pytest.mark.skipif(
    is_windows(), reason="Test not supported on Windows"
)

skip_if_macos = pytest.mark.skipif(is_macos(), reason="Test not supported on macOS")


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests for workflows")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "mcmc: MCMC statistical tests")
    config.addinivalue_line("markers", "property: Property-based tests")
    config.addinivalue_line("markers", "slow: Slow tests (> 5 seconds)")
    config.addinivalue_line("markers", "requires_jax: Requires JAX installation")
    config.addinivalue_line("markers", "linux: Requires Linux OS")
    config.addinivalue_line("markers", "arviz: ArviZ integration tests (v2.4.1+)")
    config.addinivalue_line(
        "markers", "visualization: Plotting and visualization tests"
    )


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


@pytest.fixture(autouse=True, scope="function")
def cleanup_jax_state():
    """Clear JAX cache between tests to prevent contamination.

    This fixture automatically runs after each test to ensure clean state.
    Addresses test isolation issues identified in v2.1.0 testing.
    """
    yield

    # Force garbage collection
    import gc

    gc.collect()

    # Clear JAX compilation cache if available
    # CRITICAL FIX (Nov 10, 2025): jax.clear_backends() removed in JAX 0.4.0+
    # Use jax.clear_caches() for JAX 0.8.0 compatibility
    if JAX_AVAILABLE:
        try:
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        except Exception:
            pass


@pytest.fixture(autouse=True, scope="function")
def reset_config_state():
    """Reset global configuration state between tests.

    Prevents config pollution from one test affecting subsequent tests.
    Clears cached ConfigManager instances and parameter spaces.
    """
    yield

    # Clear cached config managers
    try:
        from homodyne.config import manager

        if hasattr(manager, "_cache"):
            manager._cache.clear()
    except (ImportError, AttributeError):
        pass

    # Clear parameter space cache
    try:
        from homodyne.config import parameter_space

        if hasattr(parameter_space, "_cache"):
            parameter_space._cache.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture(scope="session")
def numpy_backend():
    """NumPy backend for fallback tests.

    Note: Session-scoped as numpy module reference is constant.
    """
    return np


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="module")
def test_config():
    """Basic test configuration dictionary.

    Note: Module-scoped as config values are read-only during tests.
    """
    return {
        "analysis_mode": "static",
        "optimization": {
            "method": "nlsq",
            "lsq": {"max_iterations": 100, "tolerance": 1e-6},
        },
        "hardware": {"force_cpu": True},  # GPU support removed in v2.3.0
        "output": {"save_plots": False, "verbose": False},
    }


# ============================================================================
# Test Data Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def synthetic_xpcs_data():
    """Generate synthetic XPCS data for testing.

    Note: Module-scoped for memory efficiency (avoids 200MB+ recreation per test).
    """
    n_times = 50
    n_angles = 36

    # Time arrays
    t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")

    # Angle array
    phi = np.linspace(0, 2 * np.pi, n_angles)

    # Generate synthetic correlation function
    tau = np.abs(t1 - t2) + 1e-6
    c2_base = 1 + 0.5 * np.exp(-tau / 10.0)

    # Create 3D array (n_angles, n_times, n_times) by repeating for each angle
    c2_exp = np.tile(c2_base, (n_angles, 1, 1))

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
        "dt": 0.1,  # Time step in seconds (required for physics calculations)
    }


@pytest.fixture(scope="module")
def small_xpcs_data():
    """Small XPCS dataset for fast tests.

    Note: Module-scoped for memory efficiency.
    """
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
        "dt": 0.1,  # Time step in seconds (required for physics calculations)
    }


@pytest.fixture(scope="module")
def large_xpcs_data():
    """Large XPCS dataset for performance tests.

    Note: Module-scoped to prevent 200MB+ memory waste per test.
    """
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


@pytest.fixture(scope="module")
def test_parameters():
    """Standard test parameters for optimization.

    Note: Module-scoped as values are immutable and reused across tests.
    """
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
    """Create a mock HDF5 file for testing in APS old format.

    APS old format structure:
    - xpcs/dqlist: (1, N) array of q-vectors
    - xpcs/dphilist: (1, N) array of phi angles
    - exchange/C2T_all/{0000, 0001, ...}: half-matrix correlation data
    """
    if not HAS_H5PY:
        pytest.skip("h5py not available")

    file_path = temp_dir / "test_data.h5"

    with h5py.File(file_path, "w") as f:
        # Create APS old format structure
        # Required keys for detection: xpcs/dqlist, xpcs/dphilist, exchange/C2T_all
        xpcs_grp = f.create_group("xpcs")

        # Create dqlist and dphilist with shape (1, N)
        n_angles = 36
        xpcs_grp.create_dataset(
            "dqlist", data=np.array([[0.01] * n_angles])
        )  # Shape (1, 36)
        xpcs_grp.create_dataset(
            "dphilist", data=np.array([np.linspace(0, 2 * np.pi, n_angles)])
        )  # Shape (1, 36)

        # Create exchange/C2T_all group with correlation matrices
        exchange_grp = f.create_group("exchange")
        c2t_grp = exchange_grp.create_group("C2T_all")

        # Create correlation matrices (half matrices as per APS old format)
        # APS "half matrix" is actually a square matrix (NxN) where full = half + half.T
        # Reconstruction: c2_full = c2_half + c2_half.T; c2_full[diag] /= 2
        n_times = 50

        for i in range(n_angles):
            # Create square half matrix (upper triangular part matters)
            half_matrix = (
                np.triu(np.random.random((n_times, n_times))) + 1.0
            )  # Values > 1.0
            c2t_grp.create_dataset(f"{i:04d}", data=half_matrix)

    yield file_path


@pytest.fixture
def mock_yaml_config(temp_dir):
    """Create a mock YAML configuration file."""
    if not HAS_YAML:
        pytest.skip("yaml not available")

    config_path = temp_dir / "test_config.yaml"
    config = {
        "data_file": "test_data.h5",
        "analysis_mode": "static",
        "output_directory": str(temp_dir),
        "optimization": {"method": "nlsq", "lsq": {"max_iterations": 50}},
    }

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    yield config_path


@pytest.fixture
def platform_info():
    """Get current platform information."""
    return {
        "is_linux": is_linux(),
        "is_macos": is_macos(),
        "is_windows": is_windows(),
        "platform": sys.platform,
    }


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


# ============================================================================
# Phase 1: Per-Angle Scaling Fixtures for NumPyro MCMC
# ============================================================================
# Critical for v2.4.0: NumPyro requires strict parameter ordering
# Order: [contrast_0, ..., contrast_N, offset_0, ..., offset_N, physical_params...]


def validate_numpyro_param_order(param_names, n_angles, mode):
    """Validate parameter ordering matches NumPyro expectations.

    NumPyro samples parameters in strict order:
    1. Per-angle contrast: contrast_0, contrast_1, ..., contrast_{n_angles-1}
    2. Per-angle offset: offset_0, offset_1, ..., offset_{n_angles-1}
    3. Physical parameters: D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0

    Args:
        param_names: List of parameter names to validate
        n_angles: Number of angles
        mode: Analysis mode ("static" or "laminar_flow")

    Raises:
        ValueError: If parameter ordering is incorrect

    Returns:
        bool: True if ordering is valid
    """
    if not param_names:
        raise ValueError("param_names cannot be empty")

    if n_angles < 1:
        raise ValueError(f"n_angles must be >= 1, got {n_angles}")

    if mode not in ("static", "laminar_flow"):
        raise ValueError(f"mode must be 'static' or 'laminar_flow', got {mode}")

    idx = 0

    # Check per-angle contrast parameters
    for angle_idx in range(n_angles):
        expected_name = f"contrast_{angle_idx}"
        if idx >= len(param_names) or param_names[idx] != expected_name:
            raise ValueError(
                f"Expected parameter '{expected_name}' at position {idx}, "
                f"but got '{param_names[idx] if idx < len(param_names) else 'END'}'"
            )
        idx += 1

    # Check per-angle offset parameters
    for angle_idx in range(n_angles):
        expected_name = f"offset_{angle_idx}"
        if idx >= len(param_names) or param_names[idx] != expected_name:
            raise ValueError(
                f"Expected parameter '{expected_name}' at position {idx}, "
                f"but got '{param_names[idx] if idx < len(param_names) else 'END'}'"
            )
        idx += 1

    # Check physical parameters in order
    if mode == "static":
        expected_physical = ["D0", "alpha", "D_offset"]
    elif mode == "laminar_flow":
        expected_physical = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

    for expected_name in expected_physical:
        if idx >= len(param_names) or param_names[idx] != expected_name:
            raise ValueError(
                f"Expected physical parameter '{expected_name}' at position {idx}, "
                f"but got '{param_names[idx] if idx < len(param_names) else 'END'}'"
            )
        idx += 1

    if idx != len(param_names):
        raise ValueError(
            f"Expected {idx} parameters total, but got {len(param_names)}. "
            f"Extra parameters: {param_names[idx:]}"
        )

    return True


@pytest.fixture(scope="module")
def per_angle_params_static():
    """Per-angle scaling parameters for static isotropic mode.

    Returns dict with correctly ordered parameters for NumPyro MCMC:
    [contrast_0, contrast_1, contrast_2, offset_0, offset_1, offset_2,
     D0, alpha, D_offset]

    Default n_angles=3 for typical use cases.

    Yields:
        dict with keys:
        - 'params': np.array of shape (9,) with parameter values
        - 'param_names': list of parameter names in NumPyro order
        - 'n_angles': 3
        - 'n_physical': 3
    """
    n_angles = 3
    n_physical = 3

    # Create parameter names in NumPyro order
    param_names = (
        [f"contrast_{i}" for i in range(n_angles)]
        + [f"offset_{i}" for i in range(n_angles)]
        + ["D0", "alpha", "D_offset"]
    )

    # Validate ordering
    validate_numpyro_param_order(param_names, n_angles, "static")

    # Create parameter values in correct order
    contrast_values = np.array([0.5, 0.5, 0.5])
    offset_values = np.array([1.0, 1.0, 1.0])
    physical_values = np.array([1000.0, 0.5, 10.0])

    params = np.concatenate([contrast_values, offset_values, physical_values])

    yield {
        "params": params,
        "param_names": param_names,
        "n_angles": n_angles,
        "n_physical": n_physical,
    }


@pytest.fixture(scope="module")
def per_angle_params_laminar():
    """Per-angle scaling parameters for laminar flow mode.

    Returns dict with correctly ordered parameters for NumPyro MCMC:
    [contrast_0, contrast_1, contrast_2, offset_0, offset_1, offset_2,
     D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    Default n_angles=3 for typical use cases.

    Yields:
        dict with keys:
        - 'params': np.array of shape (13,) with parameter values
        - 'param_names': list of parameter names in NumPyro order
        - 'n_angles': 3
        - 'n_physical': 7
    """
    n_angles = 3
    n_physical = 7

    # Create parameter names in NumPyro order
    param_names = (
        [f"contrast_{i}" for i in range(n_angles)]
        + [f"offset_{i}" for i in range(n_angles)]
        + [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
    )

    # Validate ordering
    validate_numpyro_param_order(param_names, n_angles, "laminar_flow")

    # Create parameter values in correct order
    contrast_values = np.array([0.5, 0.5, 0.5])
    offset_values = np.array([1.0, 1.0, 1.0])
    physical_values = np.array([1000.0, 0.5, 10.0, 100.0, 0.3, 5.0, 0.0])

    params = np.concatenate([contrast_values, offset_values, physical_values])

    yield {
        "params": params,
        "param_names": param_names,
        "n_angles": n_angles,
        "n_physical": n_physical,
    }


@pytest.fixture(scope="module")
def per_angle_bounds_static():
    """Per-angle bounds for static isotropic mode.

    Returns tuple (lower_bounds, upper_bounds) as np.arrays with proper scaling:
    - Contrast bounds: [0.0, 1.0] for each angle
    - Offset bounds: [0.5, 1.5] for each angle (MCMC stability from v2.2.1)
    - Physical bounds: D0 [100.0, 1e5], alpha [0.3, 1.5], D_offset [1.0, 1000.0]

    For 3 angles:
    - Lower: [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 100.0, 0.3, 1.0]
    - Upper: [1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1e5, 1.5, 1000.0]

    Yields:
        tuple: (lower_bounds, upper_bounds) both as np.arrays
    """
    n_angles = 3

    # Contrast bounds: [0.0, 1.0] per angle
    contrast_lower = np.zeros(n_angles)
    contrast_upper = np.ones(n_angles)

    # Offset bounds: [0.5, 1.5] per angle (MCMC stability)
    offset_lower = np.full(n_angles, 0.5)
    offset_upper = np.full(n_angles, 1.5)

    # Physical bounds for static mode
    # D0: [100.0, 1e5], alpha: [0.3, 1.5], D_offset: [1.0, 1000.0]
    physical_lower = np.array([100.0, 0.3, 1.0])
    physical_upper = np.array([1e5, 1.5, 1000.0])

    # Concatenate in NumPyro order: contrast, offset, physical
    lower_bounds = np.concatenate([contrast_lower, offset_lower, physical_lower])
    upper_bounds = np.concatenate([contrast_upper, offset_upper, physical_upper])

    yield (lower_bounds, upper_bounds)


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
            assert arr.shape == expected_shape, (
                f"Shape mismatch: {arr.shape} != {expected_shape}"
            )

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
                    assert abs(actual_val - expected_val) < tolerance, (
                        f"Parameter {key}: {actual_val} != {expected_val} ± {tolerance}"
                    )


@pytest.fixture
def test_utils():
    """Test utility functions."""
    return TestUtils


# ============================================================================
# MCMC/CMC Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def mcmc_config_minimal():
    """Mock MCMC configuration with minimal settings.

    Returns a simple dict-like object with MCMC parameters for testing.
    Properties: num_warmup, num_samples, num_chains
    """

    class MCMCConfig:
        """Simple MCMC configuration for testing."""

        def __init__(self):
            self.num_warmup = 100
            self.num_samples = 200
            self.num_chains = 2

        def __getitem__(self, key):
            """Dict-like access for compatibility."""
            return getattr(self, key)

        def __setitem__(self, key, value):
            """Dict-like setting for compatibility."""
            setattr(self, key, value)

        def get(self, key, default=None):
            """Dict-like get method."""
            return getattr(self, key, default)

    return MCMCConfig()


@pytest.fixture(scope="module")
def cmc_shard_config():
    """CMC sharding configuration.

    Returns a dict-like object with CMC-specific sharding parameters.
    Properties: n_shards, shard_strategy, min_points_per_shard
    """

    class CMCShardConfig:
        """CMC sharding configuration for testing."""

        def __init__(self):
            self.n_shards = 4
            self.shard_strategy = "stratified"
            self.min_points_per_shard = 100

        def __getitem__(self, key):
            """Dict-like access for compatibility."""
            return getattr(self, key)

        def __setitem__(self, key, value):
            """Dict-like setting for compatibility."""
            setattr(self, key, value)

        def get(self, key, default=None):
            """Dict-like get method."""
            return getattr(self, key, default)

    return CMCShardConfig()


@pytest.fixture(scope="function")
def mock_mcmc_samples():
    """Generate mock MCMC samples with reproducible randomness.

    Returns a callable that generates mock MCMC samples with keys:
    - 'samples': (n_chains, n_samples, n_params) array
    - 'divergences': (n_chains, n_samples) boolean array
    - 'accept_prob': (n_chains, n_samples) probability array
    """

    def generate_samples(n_samples=100, n_params=9, n_chains=2, seed=42):
        """Generate reproducible mock MCMC samples.

        Args:
            n_samples: Number of samples per chain
            n_params: Number of parameters per sample
            n_chains: Number of independent chains
            seed: Random seed for reproducibility

        Returns:
            Dictionary with 'samples', 'divergences', and 'accept_prob' keys
        """
        rng = np.random.default_rng(seed)

        # Generate samples: (n_chains, n_samples, n_params)
        # Samples around reasonable parameter values (0.5-2.0 range)
        samples = rng.normal(1.0, 0.3, size=(n_chains, n_samples, n_params))
        samples = np.clip(samples, 0.1, 5.0)  # Physical bounds

        # Generate divergences: some chains diverge occasionally
        divergences = rng.random((n_chains, n_samples)) < 0.05  # 5% divergence rate

        # Generate acceptance probabilities: realistic (70-95%)
        accept_prob = rng.uniform(0.7, 0.95, size=(n_chains, n_samples))

        return {
            "samples": samples,
            "divergences": divergences,
            "accept_prob": accept_prob,
        }

    return generate_samples


@pytest.fixture(scope="module")
def numpyro_init_values_validator():
    """Validate NumPyro initialization values against expected parameter names.

    Returns a callable that validates init_values dict against required parameters.
    Checks that all parameters are present and in correct order for NumPyro sampling.

    Raises:
        AssertionError: If validation fails with helpful message
    """

    def validate_init_values(init_values, expected_params=None, n_angles=None):
        """Validate NumPyro initialization values dictionary.

        Args:
            init_values: Dictionary of parameter names -> values
            expected_params: List of expected parameter names (optional)
            n_angles: Number of angles for auto-generating per-angle params

        Returns:
            True if valid

        Raises:
            AssertionError: If validation fails with helpful error message
        """
        if not isinstance(init_values, dict):
            raise AssertionError(
                f"init_values must be dict, got {type(init_values).__name__}"
            )

        # If explicit expected params provided, validate against them
        if expected_params is not None:
            actual_keys = set(init_values.keys())
            expected_keys = set(expected_params)

            missing = expected_keys - actual_keys
            if missing:
                raise AssertionError(
                    f"Missing parameters: {sorted(missing)}. "
                    f"Present: {sorted(actual_keys)}"
                )

            extra = actual_keys - expected_keys
            if extra:
                raise AssertionError(
                    f"Unexpected parameters: {sorted(extra)}. "
                    f"Expected: {sorted(expected_keys)}"
                )

            # Validate parameter ordering matches expected order
            actual_order = list(init_values.keys())
            if actual_order != expected_params:
                raise AssertionError(
                    f"Parameter ordering mismatch. "
                    f"Expected: {expected_params}, "
                    f"Got: {actual_order}"
                )

        # If n_angles provided, validate per-angle structure
        if n_angles is not None:
            # Check for per-angle contrast and offset parameters
            contrast_keys = [k for k in init_values.keys() if k.startswith("contrast_")]
            offset_keys = [k for k in init_values.keys() if k.startswith("offset_")]

            if len(contrast_keys) != n_angles:
                raise AssertionError(
                    f"Expected {n_angles} contrast parameters, "
                    f"found {len(contrast_keys)}"
                )

            if len(offset_keys) != n_angles:
                raise AssertionError(
                    f"Expected {n_angles} offset parameters, found {len(offset_keys)}"
                )

            # Validate that per-angle indices are 0 to n_angles-1
            expected_indices = set(range(n_angles))
            actual_contrast_indices = set()
            actual_offset_indices = set()

            for key in contrast_keys:
                try:
                    idx = int(key.split("_")[1])
                    actual_contrast_indices.add(idx)
                except (IndexError, ValueError) as exc:
                    raise AssertionError(
                        f"Invalid contrast parameter name format: {key}"
                    ) from exc

            for key in offset_keys:
                try:
                    idx = int(key.split("_")[1])
                    actual_offset_indices.add(idx)
                except (IndexError, ValueError) as exc:
                    raise AssertionError(
                        f"Invalid offset parameter name format: {key}"
                    ) from exc

            if actual_contrast_indices != expected_indices:
                raise AssertionError(
                    f"Contrast indices mismatch. Expected {expected_indices}, "
                    f"got {actual_contrast_indices}"
                )

            if actual_offset_indices != expected_indices:
                raise AssertionError(
                    f"Offset indices mismatch. Expected {expected_indices}, "
                    f"got {actual_offset_indices}"
                )

        # Validate all values are numeric and finite
        for key, value in init_values.items():
            if not isinstance(value, (int, float, np.number)):
                raise AssertionError(
                    f"Parameter {key} has non-numeric value: {type(value).__name__}"
                )

            if isinstance(value, float) and not np.isfinite(value):
                raise AssertionError(f"Parameter {key} has non-finite value: {value}")

        return True

    return validate_init_values


# ============================================================================
# NumericalValidator Exception-Based Fixtures
# ============================================================================
# v2.4.0 API change: NumericalValidator now uses exception-based validation
# instead of return-value based pattern. These fixtures support tests that
# need to verify numerical validation behavior using try/except patterns.


@pytest.fixture(scope="module")
def numerical_error_types():
    """Map exception type names to their classes for numerical validation testing.

    Returns a dictionary mapping error type names to exception classes that can be
    raised during numerical validation. Provides graceful degradation if imports fail.

    Returns:
        dict: Mapping of error type names to exception classes:
            - "NumericalError": NLSQNumericalError
            - "OptimizationError": NLSQOptimizationError
            - "ConvergenceError": NLSQConvergenceError
              (only if import succeeds; otherwise returns empty dict)

    Examples:
        >>> with pytest.raises(error_types["NumericalError"]):
        ...     validator.validate_gradients(np.array([np.nan, 1.0]))
    """
    error_types = {}

    try:
        from homodyne.optimization.exceptions import (
            NLSQConvergenceError,
            NLSQNumericalError,
            NLSQOptimizationError,
        )

        error_types["NumericalError"] = NLSQNumericalError
        error_types["OptimizationError"] = NLSQOptimizationError
        error_types["ConvergenceError"] = NLSQConvergenceError
    except ImportError:
        # Graceful degradation if exceptions module not available
        pass

    return error_types


@pytest.fixture(scope="function")
def mock_numerical_validator():
    """Create a mock NumericalValidator using exception-based API.

    Returns a mock validator object that uses the exception-based validation API
    (v2.4.0+). Provides .validate_values(), .validate_gradients(), and
    .validate_parameters() methods that raise custom exceptions on failure.

    Returns:
        object: Mock validator with these methods:
            - validate_values(data): Raises exception for NaN/Inf values
            - validate_gradients(gradients): Raises exception for non-finite gradients
            - validate_parameters(params, bounds): Raises exception for invalid params
            - enable(): Re-enable validation
            - disable(): Disable validation

    Examples:
        >>> with pytest.raises(Exception):
        ...     validator.validate_gradients(np.array([np.nan, 1.0]))
        >>> validator.validate_parameters(np.array([0.5, 1.0]), bounds)
        >>> # No exception raised on valid parameters
    """
    try:
        from homodyne.optimization.exceptions import NLSQNumericalError
    except ImportError:
        # Fallback: define minimal exception locally
        class NLSQNumericalError(Exception):
            """Fallback numerical error exception."""

            def __init__(
                self,
                message,
                detection_point=None,
                invalid_values=None,
                error_context=None,
            ):
                super().__init__(message)
                self.detection_point = detection_point
                self.invalid_values = invalid_values or []
                self.error_context = error_context or {}

    class MockNumericalValidator:
        """Mock validator using exception-based API."""

        def __init__(self):
            self.enable_validation = True
            self.bounds = None

        def validate_values(self, data):
            """Validate data array for NaN/Inf values.

            Parameters
            ----------
            data : array-like
                Values to validate

            Raises
            ------
            NLSQNumericalError
                If data contains NaN or Inf values
            """
            if not self.enable_validation:
                return

            data_array = np.asarray(data)

            if not np.all(np.isfinite(data_array)):
                invalid_mask = ~np.isfinite(data_array)
                invalid_indices = np.where(invalid_mask)[0]

                invalid_values = [
                    f"data[{int(idx)}]={float(data_array.flat[idx])}"
                    for idx in invalid_indices[:5]
                ]

                raise NLSQNumericalError(
                    f"Non-finite values detected at {len(invalid_indices)} locations.",
                    detection_point="values",
                    invalid_values=invalid_values,
                    error_context={"n_invalid": int(len(invalid_indices))},
                )

        def validate_gradients(self, gradients):
            """Validate gradient array for NaN/Inf values.

            Parameters
            ----------
            gradients : array-like
                Gradient values to validate

            Raises
            ------
            NLSQNumericalError
                If gradients contain NaN or Inf values
            """
            if not self.enable_validation:
                return

            grad_array = np.asarray(gradients)

            if not np.all(np.isfinite(grad_array)):
                invalid_mask = ~np.isfinite(grad_array)
                invalid_indices = np.where(invalid_mask)[0]

                invalid_values = [
                    f"grad[{int(idx)}]={float(grad_array[idx])}"
                    for idx in invalid_indices[:5]
                ]

                raise NLSQNumericalError(
                    f"Non-finite gradients detected at {len(invalid_indices)} locations.",
                    detection_point="gradient",
                    invalid_values=invalid_values,
                    error_context={"n_invalid": int(len(invalid_indices))},
                )

        def validate_parameters(self, parameters, bounds=None):
            """Validate parameters for NaN/Inf and bounds violations.

            Parameters
            ----------
            parameters : array-like
                Parameter values to validate
            bounds : tuple of np.ndarray, optional
                Parameter bounds (lower, upper) for validation

            Raises
            ------
            NLSQNumericalError
                If parameters contain NaN/Inf or violate bounds
            """
            if not self.enable_validation:
                return

            param_array = np.asarray(parameters)

            # Check for NaN/Inf
            if not np.all(np.isfinite(param_array)):
                invalid_mask = ~np.isfinite(param_array)
                invalid_indices = np.where(invalid_mask)[0]

                invalid_values = [
                    f"param[{int(idx)}]={float(param_array[idx])}"
                    for idx in invalid_indices[:5]
                ]

                raise NLSQNumericalError(
                    f"Non-finite parameters detected at {len(invalid_indices)} locations.",
                    detection_point="parameter",
                    invalid_values=invalid_values,
                    error_context={"n_invalid": int(len(invalid_indices))},
                )

            # Check bounds if provided
            bounds_to_check = bounds or self.bounds
            if bounds_to_check is not None:
                lower, upper = bounds_to_check
                lower = np.asarray(lower)
                upper = np.asarray(upper)

                violations_lower = param_array < lower
                violations_upper = param_array > upper

                if np.any(violations_lower) or np.any(violations_upper):
                    n_violations = int(
                        np.sum(violations_lower) + np.sum(violations_upper)
                    )

                    violation_info = []
                    for i in range(len(param_array)):
                        if violations_lower[i]:
                            violation_info.append(
                                f"param[{i}]={float(param_array[i])} < lower={float(lower[i])}"
                            )
                        elif violations_upper[i]:
                            violation_info.append(
                                f"param[{i}]={float(param_array[i])} > upper={float(upper[i])}"
                            )

                    raise NLSQNumericalError(
                        f"Parameter bounds violations at {n_violations} locations.",
                        detection_point="parameter_bounds",
                        invalid_values=violation_info[:5],
                        error_context={"n_violations": n_violations},
                    )

        def validate_loss(self, loss_value):
            """Validate loss value for NaN/Inf.

            Parameters
            ----------
            loss_value : scalar
                Loss value to validate

            Raises
            ------
            NLSQNumericalError
                If loss is NaN or Inf
            """
            if not self.enable_validation:
                return

            loss_scalar = float(loss_value)

            if not np.isfinite(loss_scalar):
                raise NLSQNumericalError(
                    f"Non-finite loss value: {loss_scalar}",
                    detection_point="loss",
                    invalid_values=[f"loss={loss_scalar}"],
                    error_context={"loss_value": loss_scalar},
                )

        def set_bounds(self, bounds):
            """Update parameter bounds for validation.

            Parameters
            ----------
            bounds : tuple of np.ndarray
                New parameter bounds (lower, upper)
            """
            self.bounds = bounds

        def enable(self):
            """Enable validation."""
            self.enable_validation = True

        def disable(self):
            """Disable validation."""
            self.enable_validation = False

    return MockNumericalValidator()


@pytest.fixture(scope="function")
def numerical_validation_context():
    """Context manager for capturing numerical validation results.

    Returns a context manager that wraps code blocks containing numerical
    validation operations. Captures whether validation passed (no exception)
    or failed (exception raised), enabling test assertions on validation outcomes.

    Returns:
        callable: Context manager factory function

    Usage:
        >>> with numerical_validation_context() as result:
        ...     validator.validate_parameters(params, bounds)
        ... assert result.passed

    Examples:
        Test validation success:
        >>> with numerical_validation_context() as result:
        ...     validator.validate_parameters(np.array([0.5, 1.0]), bounds)
        ... assert result.passed
        ... assert result.exception is None

        Test validation failure:
        >>> with numerical_validation_context() as result:
        ...     validator.validate_parameters(np.array([np.nan, 1.0]), bounds)
        ... assert not result.passed
        ... assert result.exception is not None
    """
    from contextlib import contextmanager

    @contextmanager
    def validation_context():
        """Context manager capturing validation state.

        Yields:
            object: Result object with attributes:
                - passed: bool, True if no exception raised
                - exception: Exception instance if one was raised, None otherwise
                - detection_point: str if exception has detection_point attribute
                - error_context: dict if exception has error_context attribute
        """

        class ValidationResult:
            """Capture validation outcome."""

            def __init__(self):
                self.passed = True
                self.exception = None
                self.detection_point = None
                self.error_context = None

        result = ValidationResult()

        try:
            yield result
        except Exception as e:
            # Capture exception details
            result.passed = False
            result.exception = e

            # Capture additional context if available
            if hasattr(e, "detection_point"):
                result.detection_point = e.detection_point
            if hasattr(e, "error_context"):
                result.error_context = e.error_context

            # Re-raise to allow test to assert on exception
            raise

    return validation_context


# ============================================================================
# Scientific Computing Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def physics_bounds():
    """Physics bounds for parameter validation in XPCS analysis.

    Returns a dict with physically reasonable bounds for all parameters used in
    homodyne correlation function modeling. These bounds represent typical ranges
    for XPCS experiments and are used for parameter recovery validation.

    Returns:
        dict: Parameter bounds with keys:
            - 'D0': [100.0, 1e5] - Diffusion coefficient (nm²/s)
            - 'alpha': [0.3, 1.5] - Anomalous diffusion exponent
            - 'D_offset': [1.0, 1000.0] - Diffusion offset
            - 'gamma_dot_t0': [1.0, 10000.0] - Shear rate
            - 'beta': [0.1, 2.0] - Shear exponent
            - 'gamma_dot_t_offset': [-100.0, 100.0] - Shear offset (can be negative)
            - 'phi0': [-np.pi, np.pi] - Phase offset
            - 'contrast': [0.0, 1.0] - Per-angle contrast
            - 'offset': [0.5, 1.5] - Per-angle offset (MCMC stability from v2.2.1)

    Examples:
        >>> bounds = physics_bounds
        >>> assert bounds['D0'] == [100.0, 1e5]
        >>> assert bounds['alpha'] == [0.3, 1.5]
    """
    return {
        "D0": [100.0, 1e5],
        "alpha": [0.3, 1.5],
        "D_offset": [1.0, 1000.0],
        "gamma_dot_t0": [1.0, 10000.0],
        "beta": [0.1, 2.0],
        "gamma_dot_t_offset": [-100.0, 100.0],
        "phi0": [-np.pi, np.pi],
        "contrast": [0.0, 1.0],
        "offset": [0.5, 1.5],
    }


@pytest.fixture(scope="module")
def physics_tolerances():
    """Tolerance values for physics validation in XPCS analysis.

    Returns a dict with tolerance values used to validate physical constraints
    on correlation functions and parameter recovery in homodyne fitting.

    Returns:
        dict: Tolerance values with keys:
            - 'relative_error': 0.05 - Relative error tolerance (5%) for parameter recovery
            - 'absolute_error': 1e-6 - Absolute error tolerance for near-zero values
            - 'g2_min': 1.0 - Lower bound for g2 correlation function
            - 'g2_max': 2.0 - Upper bound for g2 correlation function

    Examples:
        >>> tolerances = physics_tolerances
        >>> assert tolerances['relative_error'] == 0.05
        >>> assert tolerances['g2_min'] == 1.0
    """
    return {
        "relative_error": 0.05,
        "absolute_error": 1e-6,
        "g2_min": 1.0,
        "g2_max": 2.0,
    }


@pytest.fixture(scope="function")
def physics_validation_helpers():
    """Helper functions for physics assertions in XPCS tests.

    Returns a dict of callable helper functions for validating physics constraints
    on correlation functions, time symmetry, parameter recovery, and correlation decay.

    Returns:
        dict: Helper functions with keys:
            - 'validate_g2_bounds': Assert 1.0 <= g2 <= 2.0
            - 'validate_time_symmetry': Assert g2(t1,t2) == g2(t2,t1)
            - 'validate_parameter_recovery': Assert parameter recovery within tolerance
            - 'validate_correlation_decay': Assert g2 decays to 1.0 as |t1-t2| increases

    Examples:
        >>> helpers = physics_validation_helpers
        >>> g2 = np.linspace(1.0, 1.5, 100)
        >>> helpers['validate_g2_bounds'](g2)  # No exception if valid

    Notes:
        All validators raise AssertionError with descriptive messages on failure.
    """

    def validate_g2_bounds(g2_array, g2_min=1.0, g2_max=2.0):
        """Assert 1.0 <= g2 <= 2.0 for correlation function values.

        Parameters:
            g2_array: Array of g2 correlation function values
            g2_min: Minimum physical bound (default 1.0)
            g2_max: Maximum physical bound (default 2.0)

        Raises:
            AssertionError: If any g2 value violates bounds
        """
        g2_arr = np.asarray(g2_array)

        assert np.all(g2_arr >= g2_min), (
            f"g2 values below minimum {g2_min}: "
            f"min={np.min(g2_arr):.6f}, max={np.max(g2_arr):.6f}"
        )

        assert np.all(g2_arr <= g2_max), (
            f"g2 values above maximum {g2_max}: "
            f"min={np.min(g2_arr):.6f}, max={np.max(g2_arr):.6f}"
        )

    def validate_time_symmetry(g2_phi, t1, t2, tolerance=1e-6):
        """Assert time symmetry: g2(t1,t2) == g2(t2,t1) for correlation matrix.

        The correlation function should be symmetric about the diagonal
        when evaluated at swapped time indices.

        Parameters:
            g2_phi: Correlation matrix of shape (n_times, n_times)
            t1: Time array for first dimension
            t2: Time array for second dimension
            tolerance: Tolerance for symmetry check (default 1e-6)

        Raises:
            AssertionError: If symmetry violated beyond tolerance
        """
        g2_arr = np.asarray(g2_phi)

        # Check that g2(t1,t2) == g2(t2,t1)
        g2_transposed = g2_arr.T

        max_diff = np.max(np.abs(g2_arr - g2_transposed))

        assert max_diff <= tolerance, (
            f"Time symmetry violated: max difference = {max_diff:.2e}, "
            f"tolerance = {tolerance:.2e}"
        )

    def validate_parameter_recovery(true_params, fitted_params, tolerance=0.05):
        """Assert parameter recovery within relative tolerance.

        Parameters:
            true_params: True parameter values (dict or array)
            fitted_params: Fitted parameter values (dict or array)
            tolerance: Relative error tolerance (default 0.05 = 5%)

        Raises:
            AssertionError: If recovery error exceeds tolerance
        """
        # Handle both dict and array inputs
        if isinstance(true_params, dict):
            true_arr = np.array([true_params[k] for k in sorted(true_params.keys())])
            fitted_arr = np.array(
                [fitted_params[k] for k in sorted(fitted_params.keys())]
            )
        else:
            true_arr = np.asarray(true_params)
            fitted_arr = np.asarray(fitted_params)

        # Calculate relative error, avoiding division by zero
        relative_errors = []
        for true_val, fitted_val in zip(true_arr, fitted_arr, strict=False):
            if abs(true_val) < 1e-10:
                # For near-zero values, use absolute error
                error = abs(fitted_val)
            else:
                error = abs((fitted_val - true_val) / true_val)

            relative_errors.append(error)

        max_error = max(relative_errors)

        assert max_error <= tolerance, (
            f"Parameter recovery failed: max relative error = {max_error:.4f}, "
            f"tolerance = {tolerance:.4f}"
        )

    def validate_correlation_decay(g2, t1, t2, tolerance=1e-3):
        """Assert g2 decays to 1.0 as |t1-t2| increases.

        The correlation function should approach 1.0 (no correlation)
        for large time differences.

        Parameters:
            g2: Correlation matrix of shape (n_times, n_times)
            t1: Time array for first dimension
            t2: Time array for second dimension
            tolerance: Tolerance for 1.0 at large times (default 1e-3)

        Raises:
            AssertionError: If decay to 1.0 not observed
        """
        g2_arr = np.asarray(g2)

        # Evaluate at corners (large time differences)
        # Top-left: large negative t1-t2
        corner_tl = g2_arr[0, -1]

        # Bottom-right: large positive t1-t2
        corner_br = g2_arr[-1, 0]

        # Both should be close to 1.0
        assert abs(corner_tl - 1.0) <= tolerance, (
            f"Correlation at (t1_min, t2_max) = {corner_tl:.6f} "
            f"not decayed to 1.0 ± {tolerance}"
        )

        assert abs(corner_br - 1.0) <= tolerance, (
            f"Correlation at (t1_max, t2_min) = {corner_br:.6f} "
            f"not decayed to 1.0 ± {tolerance}"
        )

    return {
        "validate_g2_bounds": validate_g2_bounds,
        "validate_time_symmetry": validate_time_symmetry,
        "validate_parameter_recovery": validate_parameter_recovery,
        "validate_correlation_decay": validate_correlation_decay,
    }


@pytest.fixture(scope="module")
def typical_xpcs_time_scales():
    """Typical time scales for XPCS experiments.

    Returns a dict with time scale parameters for generating realistic XPCS
    time arrays spanning multiple decades from microseconds to thousands of seconds.

    Returns:
        dict: Time scale parameters with keys:
            - 't_min': 0.001 - Minimum time (1 ms)
            - 't_max': 1000.0 - Maximum time (1000 s)
            - 'n_decades': 6 - Log-spaced points per decade

    Examples:
        >>> scales = typical_xpcs_time_scales
        >>> t_min = scales['t_min']
        >>> t_max = scales['t_max']
        >>> n_decades = scales['n_decades']
        >>> # Generate log-spaced time array
        >>> times = np.logspace(np.log10(t_min), np.log10(t_max),
        ...                      n_decades * 6)

    Notes:
        These are typical ranges for XPCS at synchrotrons like the APS.
        Actual experiments may span different ranges depending on sample
        dynamics and beam intensity.
    """
    return {
        "t_min": 0.001,
        "t_max": 1000.0,
        "n_decades": 6,
    }
