"""
Test Runner and Configuration for Homodyne v2
==============================================

Comprehensive test runner with different test profiles:
- Quick tests for development
- Full test suite for CI/CD
- Performance benchmarks
- Statistical validation tests
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


class HomodyneTestRunner:
    """Test runner for Homodyne v2 test suite."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize test runner."""
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.test_dir = self.base_dir / "tests"

    def run_quick_tests(self) -> int:
        """Run quick tests for development."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "unit"),
            "-m",
            "not slow",
            "-x",  # Stop on first failure
            "--tb=short",
            "--durations=5",
        ]
        return subprocess.call(cmd)

    def run_unit_tests(self) -> int:
        """Run unit tests only."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "unit"),
            "--cov=homodyne.core",
            "--cov=homodyne.optimization",
            "--cov=homodyne.data",
            "--tb=short",
        ]
        return subprocess.call(cmd)

    def run_integration_tests(self) -> int:
        """Run integration tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "integration"),
            "--tb=short",
            "--maxfail=5",
        ]
        return subprocess.call(cmd)

    def run_performance_tests(self) -> int:
        """Run performance and benchmark tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "performance"),
            "-m",
            "performance",
            "--benchmark-only",
            "--benchmark-sort=mean",
            "--benchmark-json=benchmark_results.json",
        ]
        return subprocess.call(cmd)

    def run_mcmc_tests(self) -> int:
        """Run MCMC statistical tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "mcmc"),
            "-m",
            "mcmc",
            "--tb=short",
            "--maxfail=3",
        ]
        return subprocess.call(cmd)

    def run_property_tests(self) -> int:
        """Run property-based tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "property"),
            "-m",
            "property",
            "--tb=short",
        ]
        return subprocess.call(cmd)

    def run_api_tests(self) -> int:
        """Run API compatibility tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "api"),
            "-m",
            "api",
            "--tb=short",
        ]
        return subprocess.call(cmd)

    def run_full_suite(self) -> int:
        """Run complete test suite."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir),
            "--cov=homodyne",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-fail-under=75",
            "--html=test_report.html",
            "--self-contained-html",
            "--maxfail=20",
        ]
        return subprocess.call(cmd)

    def run_ci_tests(self) -> int:
        """Run tests suitable for CI/CD."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir),
            "--cov=homodyne",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term",
            "--junit-xml=test_results.xml",
            "--tb=short",
            "--maxfail=10",
        ]
        return subprocess.call(cmd)

    def run_regression_tests(self) -> int:
        """Run regression tests."""
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir / "performance"),
            str(self.test_dir / "integration"),
            "-m",
            "not slow",
            "--tb=short",
        ]
        return subprocess.call(cmd)

    def run_custom_tests(self, test_args: list[str]) -> int:
        """Run custom test command."""
        cmd = ["python", "-m", "pytest"] + test_args
        return subprocess.call(cmd)

    def check_test_environment(self) -> dict[str, Any]:
        """Check test environment and dependencies."""
        env_info = {
            "python_version": sys.version,
            "test_directory": str(self.test_dir),
            "dependencies": {},
        }

        # Check core dependencies
        dependencies = [
            "numpy",
            "jax",
            "pytest",
            "hypothesis",
            "scipy",
            "h5py",
            "yaml",
            "arviz",
        ]

        for dep in dependencies:
            try:
                module = __import__(dep)
                env_info["dependencies"][dep] = getattr(
                    module, "__version__", "unknown"
                )
            except ImportError:
                env_info["dependencies"][dep] = "not_installed"

        # Check JAX backend
        try:
            import jax

            env_info["jax_backend"] = jax.default_backend()
            env_info["jax_devices"] = [str(d) for d in jax.devices()]
        except ImportError:
            env_info["jax_backend"] = "not_available"
            env_info["jax_devices"] = []

        return env_info

    def print_environment_info(self) -> None:
        """Print test environment information."""
        print("=" * 60)
        print("Homodyne v2 Test Environment")
        print("=" * 60)

        env_info = self.check_test_environment()

        print(f"Python Version: {env_info['python_version']}")
        print(f"Test Directory: {env_info['test_directory']}")
        print(f"JAX Backend: {env_info.get('jax_backend', 'unknown')}")

        print("\nDependencies:")
        for dep, version in env_info["dependencies"].items():
            status = "✓" if version != "not_installed" else "✗"
            print(f"  {status} {dep}: {version}")

        print("=" * 60)


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="Homodyne v2 Test Runner")

    parser.add_argument(
        "test_type",
        choices=[
            "quick",
            "unit",
            "integration",
            "performance",
            "mcmc",
            "property",
            "api",
            "full",
            "ci",
            "regression",
            "custom",
            "env",
        ],
        help="Type of tests to run",
    )

    parser.add_argument(
        "test_args", nargs="*", help="Additional test arguments (for custom mode)"
    )

    parser.add_argument("--base-dir", type=Path, help="Base directory for tests")

    args = parser.parse_args()

    runner = HomodyneTestRunner(base_dir=args.base_dir)

    if args.test_type == "env":
        runner.print_environment_info()
        return 0

    # Run appropriate test suite
    test_methods = {
        "quick": runner.run_quick_tests,
        "unit": runner.run_unit_tests,
        "integration": runner.run_integration_tests,
        "performance": runner.run_performance_tests,
        "mcmc": runner.run_mcmc_tests,
        "property": runner.run_property_tests,
        "api": runner.run_api_tests,
        "full": runner.run_full_suite,
        "ci": runner.run_ci_tests,
        "regression": runner.run_regression_tests,
        "custom": lambda: runner.run_custom_tests(args.test_args),
    }

    method = test_methods.get(args.test_type)
    if method:
        print(f"\nRunning {args.test_type} tests...")
        return method()
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
