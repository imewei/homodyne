"""
Pytest Configuration and Fixtures for Homodyne
==================================================

Shared fixtures, configuration, and test utilities for the entire test suite.
"""

import os
import sys
import tempfile
from pathlib import Path

# Float64 must be enabled BEFORE the first JAX import (project rule #8).
os.environ.setdefault("JAX_ENABLE_X64", "1")

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
        elif "cmc" in str(item.fspath):
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
    """Yield-only fixture kept for potential future test isolation needs.

    JAX JIT caches are keyed by (function_id, shape, dtype) and do not
    cause cross-test contamination.  Clearing them after every test
    forces recompilation (~113 ms per JIT function) and added ~285 s
    (28 %) to the full suite via gc.collect() + jax.clear_caches().
    Removed in Feb 2026 cleanup.
    """
    yield


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
# MCMC/CMC Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def mock_mcmc_samples():
    """Generate mock MCMC samples with reproducible randomness.

    Returns a callable that generates mock MCMC samples with keys:
    - 'samples': (n_chains, n_samples, n_params) array
    - 'divergences': (n_chains, n_samples) boolean array
    - 'accept_prob': (n_chains, n_samples) probability array
    """

    def generate_samples(n_samples=100, n_params=9, n_chains=2, seed=42):
        """Generate reproducible mock MCMC samples with AR(1) autocorrelation.

        Args:
            n_samples: Number of samples per chain
            n_params: Number of parameters per sample
            n_chains: Number of independent chains
            seed: Random seed for reproducibility

        Returns:
            Dictionary with 'samples', 'divergences', and 'accept_prob' keys
        """
        rng = np.random.default_rng(seed)

        # Generate correlated samples using AR(1) process
        rho = 0.8  # autocorrelation coefficient
        noise = rng.standard_normal((n_chains, n_samples, n_params))
        samples = np.zeros_like(noise)
        samples[:, 0, :] = noise[:, 0, :]
        for i in range(1, n_samples):
            samples[:, i, :] = (
                rho * samples[:, i - 1, :] + np.sqrt(1 - rho**2) * noise[:, i, :]
            )

        # Adjust mean and scale to reasonable parameter values (0.5-2.0 range)
        samples = samples * 0.3 + 1.0
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


# Resolve NLSQNumericalError once at import time (used by MockNumericalValidator).
try:
    from homodyne.optimization.exceptions import (
        NLSQNumericalError as _NLSQNumericalError,
    )  # noqa: I001
except ImportError:

    class _NLSQNumericalError(Exception):  # type: ignore[no-redef]
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
    """Mock validator using exception-based API (v2.4.0+).

    Defined at module level so CPython compiles the class once rather than
    re-creating it for each of the 41 test invocations.
    """

    def __init__(self):
        self.enable_validation = True
        self.bounds = None

    def validate_values(self, data):
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
            raise _NLSQNumericalError(
                f"Non-finite values detected at {len(invalid_indices)} locations.",
                detection_point="values",
                invalid_values=invalid_values,
                error_context={"n_invalid": int(len(invalid_indices))},
            )

    def validate_gradients(self, gradients):
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
            raise _NLSQNumericalError(
                f"Non-finite gradients detected at {len(invalid_indices)} locations.",
                detection_point="gradient",
                invalid_values=invalid_values,
                error_context={"n_invalid": int(len(invalid_indices))},
            )

    def validate_parameters(self, parameters, bounds=None):
        if not self.enable_validation:
            return
        param_array = np.asarray(parameters)
        if not np.all(np.isfinite(param_array)):
            invalid_mask = ~np.isfinite(param_array)
            invalid_indices = np.where(invalid_mask)[0]
            invalid_values = [
                f"param[{int(idx)}]={float(param_array[idx])}"
                for idx in invalid_indices[:5]
            ]
            raise _NLSQNumericalError(
                f"Non-finite parameters detected at {len(invalid_indices)} locations.",
                detection_point="parameter",
                invalid_values=invalid_values,
                error_context={"n_invalid": int(len(invalid_indices))},
            )
        bounds_to_check = bounds or self.bounds
        if bounds_to_check is not None:
            lower, upper = (
                np.asarray(bounds_to_check[0]),
                np.asarray(bounds_to_check[1]),
            )
            violations_lower = param_array < lower
            violations_upper = param_array > upper
            if np.any(violations_lower) or np.any(violations_upper):
                n_violations = int(np.sum(violations_lower) + np.sum(violations_upper))
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
                raise _NLSQNumericalError(
                    f"Parameter bounds violations at {n_violations} locations.",
                    detection_point="parameter_bounds",
                    invalid_values=violation_info[:5],
                    error_context={"n_violations": n_violations},
                )

    def validate_loss(self, loss_value):
        if not self.enable_validation:
            return
        loss_scalar = float(loss_value)
        if not np.isfinite(loss_scalar):
            raise _NLSQNumericalError(
                f"Non-finite loss value: {loss_scalar}",
                detection_point="loss",
                invalid_values=[f"loss={loss_scalar}"],
                error_context={"loss_value": loss_scalar},
            )

    def set_bounds(self, bounds):
        self.bounds = bounds

    def enable(self):
        self.enable_validation = True

    def disable(self):
        self.enable_validation = False


@pytest.fixture(scope="function")
def mock_numerical_validator():
    """Fresh MockNumericalValidator instance per test."""
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
