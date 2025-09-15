"""
Utilities and Helper Functions for JAX Fallback Testing
======================================================

Provides common utilities, test data generators, and validation functions
for comprehensive JAX fallback testing across the homodyne codebase.

This module supports the comprehensive testing suite by providing:
- Realistic XPCS test data generation
- Performance measurement utilities
- Mock objects for testing different scenarios
- Validation helpers for scientific accuracy
- Memory and resource monitoring
"""

import contextlib
import os
import pickle
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np

# Constants for realistic XPCS testing
XPCS_REALISTIC_RANGES = {
    "diffusion_coefficients": {
        "D0_min": 1e-3,  # Å²/s - minimum realistic diffusion
        "D0_max": 1e6,  # Å²/s - maximum realistic diffusion
        "D0_typical": 100.0,  # Å²/s - typical experimental value
    },
    "time_scales": {
        "t_min": 1e-6,  # seconds - fastest correlation times
        "t_max": 1e3,  # seconds - slowest correlation times
        "t_typical": 1.0,  # seconds - typical measurement time
    },
    "scattering_parameters": {
        "q_min": 1e-4,  # Å⁻¹ - minimum scattering vector
        "q_max": 1e-1,  # Å⁻¹ - maximum scattering vector
        "q_typical": 0.01,  # Å⁻¹ - typical XPCS measurement
        "L_min": 100.0,  # mm - minimum sample-detector distance
        "L_max": 5000.0,  # mm - maximum sample-detector distance
        "L_typical": 1000.0,  # mm - typical experimental setup
    },
    "shear_parameters": {
        "gamma_dot_min": 1e-4,  # s⁻¹ - minimum shear rate
        "gamma_dot_max": 1e3,  # s⁻¹ - maximum shear rate
        "gamma_dot_typical": 1.0,  # s⁻¹ - typical shear flow
        "phi_min": -180.0,  # degrees - minimum angle
        "phi_max": 180.0,  # degrees - maximum angle
    },
    "experimental_parameters": {
        "contrast_min": 0.1,  # Minimum contrast parameter
        "contrast_max": 1.0,  # Maximum contrast parameter
        "contrast_typical": 0.8,  # Typical contrast value
        "offset_min": 0.9,  # Minimum baseline offset
        "offset_max": 1.1,  # Maximum baseline offset
        "offset_typical": 1.0,  # Typical baseline
    },
}


@dataclass
class XPCSTestConfiguration:
    """Configuration for XPCS testing scenarios."""

    mode: str  # 'static_isotropic', 'static_anisotropic', 'laminar_flow'
    n_params: int
    parameter_names: List[str]
    realistic_ranges: Dict[str, Tuple[float, float]]
    typical_values: List[float]
    test_conditions: Dict[str, Any] = field(default_factory=dict)

    def generate_test_parameters(
        self, n_sets: int = 5, include_extremes: bool = True
    ) -> List[np.ndarray]:
        """Generate realistic test parameter sets."""
        param_sets = []

        # Always include typical values
        param_sets.append(np.array(self.typical_values))

        # Generate random realistic values
        for _ in range(n_sets - 1):
            params = []
            for name in self.parameter_names:
                if name in self.realistic_ranges:
                    min_val, max_val = self.realistic_ranges[name]
                    # Log-uniform distribution for parameters that span orders of magnitude
                    if max_val / min_val > 100:
                        log_min, log_max = np.log10(min_val), np.log10(max_val)
                        param_val = 10 ** (np.random.uniform(log_min, log_max))
                    else:
                        param_val = np.random.uniform(min_val, max_val)
                    params.append(param_val)
                else:
                    # Default to small random variation around 1.0
                    params.append(np.random.uniform(0.1, 10.0))
            param_sets.append(np.array(params))

        # Add extreme values if requested
        if include_extremes:
            # Minimum extreme
            min_params = []
            max_params = []
            for name in self.parameter_names:
                if name in self.realistic_ranges:
                    min_val, max_val = self.realistic_ranges[name]
                    min_params.append(min_val)
                    max_params.append(max_val)
                else:
                    min_params.append(0.01)
                    max_params.append(100.0)
            param_sets.extend([np.array(min_params), np.array(max_params)])

        return param_sets


# Pre-defined test configurations
STATIC_ISOTROPIC_CONFIG = XPCSTestConfiguration(
    mode="static_isotropic",
    n_params=3,
    parameter_names=["D0", "alpha", "D_offset"],
    realistic_ranges={"D0": (1e-3, 1e6), "alpha": (-2.0, 2.0), "D_offset": (0.0, 1e4)},
    typical_values=[100.0, 0.0, 10.0],
)

STATIC_ANISOTROPIC_CONFIG = XPCSTestConfiguration(
    mode="static_anisotropic",
    n_params=3,
    parameter_names=["D0", "alpha", "D_offset"],
    realistic_ranges={"D0": (1e-3, 1e6), "alpha": (-2.0, 2.0), "D_offset": (0.0, 1e4)},
    typical_values=[100.0, 0.0, 10.0],
    test_conditions={"multiple_angles": True, "angle_filtering": True},
)

LAMINAR_FLOW_CONFIG = XPCSTestConfiguration(
    mode="laminar_flow",
    n_params=7,
    parameter_names=[
        "D0",
        "alpha",
        "D_offset",
        "gamma_dot_0",
        "beta",
        "gamma_dot_offset",
        "phi0",
    ],
    realistic_ranges={
        "D0": (1e-3, 1e6),
        "alpha": (-2.0, 2.0),
        "D_offset": (0.0, 1e4),
        "gamma_dot_0": (1e-4, 1e3),
        "beta": (-2.0, 2.0),
        "gamma_dot_offset": (0.0, 1e2),
        "phi0": (-180.0, 180.0),
    },
    typical_values=[100.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0],
)


@dataclass
class PerformanceBenchmark:
    """Container for performance benchmark results."""

    test_name: str
    jax_time: Optional[float] = None
    numpy_time: Optional[float] = None
    performance_ratio: Optional[float] = None
    memory_usage_jax: Optional[float] = None
    memory_usage_numpy: Optional[float] = None
    function_calls: int = 0
    success: bool = False
    error_message: Optional[str] = None

    def calculate_metrics(self):
        """Calculate derived performance metrics."""
        if self.jax_time and self.numpy_time:
            self.performance_ratio = self.numpy_time / self.jax_time

    def is_acceptable_performance(self, max_slowdown: float = 100.0) -> bool:
        """Check if performance is within acceptable range."""
        if self.performance_ratio is None:
            return True  # Cannot assess without JAX comparison
        return self.performance_ratio <= max_slowdown


class MockJAXEnvironment:
    """Mock JAX environment for testing fallback scenarios."""

    def __init__(self, scenario: str = "jax_unavailable"):
        """
        Initialize mock environment.

        Args:
            scenario: 'jax_unavailable', 'jax_import_error', 'jax_partial_failure'
        """
        self.scenario = scenario
        self.original_modules = {}

    def __enter__(self):
        """Enter mock environment."""
        if self.scenario == "jax_unavailable":
            # Remove JAX from sys.modules
            jax_modules = [
                name for name in sys.modules.keys() if name.startswith("jax")
            ]
            for module_name in jax_modules:
                if module_name in sys.modules:
                    self.original_modules[module_name] = sys.modules[module_name]
                    del sys.modules[module_name]

        elif self.scenario == "jax_import_error":
            # Mock failed JAX import
            def mock_import(*args, **kwargs):
                raise ImportError("JAX not available")

            self.original_import = __builtins__.__import__
            __builtins__.__import__ = mock_import

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit mock environment."""
        if self.scenario == "jax_unavailable":
            # Restore original modules
            for module_name, module in self.original_modules.items():
                sys.modules[module_name] = module

        elif self.scenario == "jax_import_error":
            # Restore original import
            __builtins__.__import__ = self.original_import


def generate_realistic_xpcs_data(
    config: XPCSTestConfiguration,
    n_time_points: int = 50,
    n_angles: int = 10,
    noise_level: float = 0.05,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate realistic XPCS experimental data for testing.

    Args:
        config: Test configuration defining the XPCS mode
        n_time_points: Number of time correlation points
        n_angles: Number of scattering angles
        noise_level: Relative noise level (0.05 = 5% noise)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing synthetic experimental data
    """
    np.random.seed(random_seed)

    # Generate time grids
    t_max = XPCS_REALISTIC_RANGES["time_scales"]["t_typical"]
    t1 = np.logspace(-3, np.log10(t_max), n_time_points)
    t2 = t1 + np.logspace(-4, np.log10(t_max / 10), n_time_points)

    # Generate angle grid
    if config.mode == "static_isotropic":
        phi = np.array([0.0])  # Single angle for isotropic
    else:
        phi = np.linspace(-90, 90, n_angles)

    # Scattering parameters
    q = XPCS_REALISTIC_RANGES["scattering_parameters"]["q_typical"]
    L = XPCS_REALISTIC_RANGES["scattering_parameters"]["L_typical"]

    # Use typical parameters to generate "ground truth" data
    true_params = np.array(config.typical_values)

    # Simulate theoretical g2 function (simplified)
    theory_g2 = np.ones((len(t1), len(phi)))

    # Add parameter-dependent modulation
    D0, alpha, D_offset = true_params[0], true_params[1], true_params[2]

    for i, (t1_val, t2_val) in enumerate(zip(t1, t2)):
        dt = abs(t2_val - t1_val)

        # Diffusion contribution
        if abs(alpha + 1) > 1e-12:
            diffusion_integral = D0 * dt ** (alpha + 1) / (alpha + 1) + D_offset * dt
        else:
            diffusion_integral = D0 * np.log(dt + 1e-12) + D_offset * dt

        g1_diff = np.exp(-0.5 * q**2 * diffusion_integral)

        # Shear contribution (if applicable)
        if config.n_params == 7:
            gamma_dot_0, beta, gamma_dot_offset, phi0 = true_params[3:7]

            # Shear integral
            if abs(beta + 1) > 1e-12:
                shear_integral = (
                    gamma_dot_0 * dt ** (beta + 1) / (beta + 1) + gamma_dot_offset * dt
                )
            else:
                shear_integral = (
                    gamma_dot_0 * np.log(dt + 1e-12) + gamma_dot_offset * dt
                )

            # Phase and sinc calculation
            for j, phi_val in enumerate(phi):
                phase = (
                    (q * L / (2 * np.pi))
                    * np.cos(np.deg2rad(phi0 - phi_val))
                    * shear_integral
                )
                sinc_val = (
                    np.sin(np.pi * phase) / (np.pi * phase + 1e-12)
                    if abs(phase) > 1e-12
                    else 1.0
                )
                g1_shear = sinc_val**2

                # Combined g1 and g2
                g1_total = g1_diff * g1_shear
                contrast = XPCS_REALISTIC_RANGES["experimental_parameters"][
                    "contrast_typical"
                ]
                offset = XPCS_REALISTIC_RANGES["experimental_parameters"][
                    "offset_typical"
                ]
                theory_g2[i, j] = offset + contrast * g1_total**2
        else:
            # Static case
            contrast = XPCS_REALISTIC_RANGES["experimental_parameters"][
                "contrast_typical"
            ]
            offset = XPCS_REALISTIC_RANGES["experimental_parameters"]["offset_typical"]
            theory_g2[i, :] = offset + contrast * g1_diff**2

    # Add realistic noise
    noise = noise_level * theory_g2 * np.random.randn(*theory_g2.shape)
    experimental_g2 = theory_g2 + noise

    # Generate uncertainties
    sigma = noise_level * theory_g2

    return {
        "g2_data": experimental_g2,
        "g2_theory": theory_g2,
        "sigma": sigma,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "q": q,
        "L": L,
        "true_parameters": true_params,
        "noise_level": noise_level,
        "config": config,
    }


def measure_memory_usage():
    """Measure current memory usage (approximate)."""
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    except ImportError:
        # Fallback to system-dependent methods
        try:
            import resource
            # Different systems have different units for ru_maxrss
            import sys

            if sys.platform == "darwin":  # macOS
                return (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
                )  # bytes to MB
            else:  # Linux
                return (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                )  # KB to MB
        except:
            return None  # Cannot measure memory


class PerformanceTimer:
    """Context manager for precise performance timing."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time

    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_time if self.elapsed_time is not None else 0.0


def benchmark_function_performance(
    func: Callable,
    args: Tuple,
    kwargs: Optional[Dict] = None,
    n_iterations: int = 10,
    warmup_iterations: int = 2,
) -> Dict[str, float]:
    """
    Benchmark function performance with statistical analysis.

    Args:
        func: Function to benchmark
        args: Function arguments
        kwargs: Function keyword arguments
        n_iterations: Number of timing iterations
        warmup_iterations: Number of warmup runs

    Returns:
        Dictionary with timing statistics
    """
    if kwargs is None:
        kwargs = {}

    # Warmup runs
    for _ in range(warmup_iterations):
        try:
            _ = func(*args, **kwargs)
        except Exception:
            pass

    # Timed runs
    times = []
    memory_before = measure_memory_usage()

    for _ in range(n_iterations):
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "mean_time": np.inf,
                "std_time": np.inf,
                "min_time": np.inf,
                "max_time": np.inf,
                "memory_delta": None,
            }

    memory_after = measure_memory_usage()
    memory_delta = (
        memory_after - memory_before if memory_before and memory_after else None
    )

    return {
        "success": True,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "times": times,
        "memory_delta": memory_delta,
    }


def validate_scientific_accuracy(
    computed_result: np.ndarray,
    reference_result: np.ndarray,
    tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Validate scientific accuracy of computed results.

    Args:
        computed_result: Result from computation
        reference_result: Reference/expected result
        tolerance: Absolute tolerance
        relative_tolerance: Relative tolerance

    Returns:
        Dictionary with validation results
    """
    computed = np.asarray(computed_result)
    reference = np.asarray(reference_result)

    if computed.shape != reference.shape:
        return {
            "valid": False,
            "error": f"Shape mismatch: {computed.shape} vs {reference.shape}",
            "absolute_error": np.inf,
            "relative_error": np.inf,
            "max_error": np.inf,
        }

    # Calculate errors
    absolute_error = np.abs(computed - reference)
    relative_error = absolute_error / (np.abs(reference) + 1e-15)

    max_absolute_error = np.max(absolute_error)
    max_relative_error = np.max(relative_error)

    # Check validity
    absolute_valid = max_absolute_error <= tolerance
    relative_valid = max_relative_error <= relative_tolerance

    return {
        "valid": absolute_valid or relative_valid,
        "absolute_valid": absolute_valid,
        "relative_valid": relative_valid,
        "absolute_error": absolute_error,
        "relative_error": relative_error,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "mean_absolute_error": np.mean(absolute_error),
        "mean_relative_error": np.mean(relative_error),
        "rmse": np.sqrt(np.mean(absolute_error**2)),
    }


class FallbackTestReporter:
    """Comprehensive test reporter for fallback scenarios."""

    def __init__(self):
        self.test_results = []
        self.performance_benchmarks = []
        self.accuracy_validations = []

    def add_test_result(self, test_name: str, result: Dict[str, Any]):
        """Add test result to the report."""
        self.test_results.append(
            {"test_name": test_name, "timestamp": time.time(), **result}
        )

    def add_performance_benchmark(self, benchmark: PerformanceBenchmark):
        """Add performance benchmark to the report."""
        self.performance_benchmarks.append(benchmark)

    def add_accuracy_validation(self, validation_name: str, validation: Dict[str, Any]):
        """Add accuracy validation to the report."""
        self.accuracy_validations.append(
            {"validation_name": validation_name, "timestamp": time.time(), **validation}
        )

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary report."""
        report = []
        report.append("=" * 60)
        report.append("JAX FALLBACK TESTING SUMMARY REPORT")
        report.append("=" * 60)

        # Test results summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("success", False))
        report.append(f"\nTest Results: {passed_tests}/{total_tests} passed")

        if total_tests > 0:
            failure_rate = (total_tests - passed_tests) / total_tests * 100
            report.append(f"Failure Rate: {failure_rate:.1f}%")

        # Performance summary
        if self.performance_benchmarks:
            report.append(
                f"\nPerformance Benchmarks: {len(self.performance_benchmarks)} tests"
            )

            successful_benchmarks = [
                b for b in self.performance_benchmarks if b.success
            ]
            if successful_benchmarks:
                ratios = [
                    b.performance_ratio
                    for b in successful_benchmarks
                    if b.performance_ratio
                ]
                if ratios:
                    mean_ratio = np.mean(ratios)
                    max_ratio = np.max(ratios)
                    report.append(f"Mean Performance Ratio: {mean_ratio:.1f}x slower")
                    report.append(f"Max Performance Ratio: {max_ratio:.1f}x slower")

                    acceptable = sum(1 for r in ratios if r <= 100)
                    report.append(
                        f"Acceptable Performance: {acceptable}/{len(ratios)} tests"
                    )

        # Accuracy summary
        if self.accuracy_validations:
            report.append(
                f"\nAccuracy Validations: {len(self.accuracy_validations)} tests"
            )

            valid_results = sum(
                1 for v in self.accuracy_validations if v.get("valid", False)
            )
            report.append(
                f"Accuracy Valid: {valid_results}/{len(self.accuracy_validations)} tests"
            )

            if len(self.accuracy_validations) > 0:
                accuracy_rate = valid_results / len(self.accuracy_validations) * 100
                report.append(f"Accuracy Rate: {accuracy_rate:.1f}%")

        # Failed tests details
        failed_tests = [r for r in self.test_results if not r.get("success", False)]
        if failed_tests:
            report.append(f"\nFAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                report.append(
                    f"  - {test['test_name']}: {test.get('error_message', 'Unknown error')}"
                )

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_detailed_report(self, filename: str):
        """Save detailed report to file."""
        report_data = {
            "summary": self.generate_summary_report(),
            "test_results": self.test_results,
            "performance_benchmarks": [
                {
                    "test_name": b.test_name,
                    "jax_time": b.jax_time,
                    "numpy_time": b.numpy_time,
                    "performance_ratio": b.performance_ratio,
                    "success": b.success,
                    "error_message": b.error_message,
                }
                for b in self.performance_benchmarks
            ],
            "accuracy_validations": self.accuracy_validations,
        }

        with open(filename, "wb") as f:
            pickle.dump(report_data, f)


# Export main utilities
__all__ = [
    "STATIC_ISOTROPIC_CONFIG",
    "STATIC_ANISOTROPIC_CONFIG",
    "LAMINAR_FLOW_CONFIG",
    "XPCSTestConfiguration",
    "PerformanceBenchmark",
    "MockJAXEnvironment",
    "generate_realistic_xpcs_data",
    "measure_memory_usage",
    "PerformanceTimer",
    "benchmark_function_performance",
    "validate_scientific_accuracy",
    "FallbackTestReporter",
    "XPCS_REALISTIC_RANGES",
]
