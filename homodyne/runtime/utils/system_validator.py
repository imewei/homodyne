#!/usr/bin/env python3
"""Homodyne System Validator
==========================

Comprehensive testing and validation system for Homodyne installation,
including dependency versions, JAX configuration, NLSQ integration,
configuration system, data pipeline, shell completion, and overall system health.

Version: 2.3.0 (CPU-only architecture, GPU validation removed)
"""

import json
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of a system validation check."""

    name: str
    success: bool
    message: str
    details: dict[str, Any] | None = None
    execution_time: float = 0.0
    warnings: list[str] | None = None
    # Enhanced fields for v3.0
    severity: str = "info"  # "critical", "warning", "info"
    remediation: list[str] | None = None  # Suggested fixes
    error_code: str | None = None  # For programmatic handling


class SystemValidator:
    """Comprehensive system validation for homodyne installation."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[ValidationResult] = []
        self.environment_info: dict[str, Any] = {}

    def log(self, message: str, level: str = "info") -> None:
        """Log message if verbose."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level.upper()}: {message}")

    def run_command(self, cmd: list[str], timeout: int = 30) -> tuple[bool, str, str]:
        """Run shell command and return success, stdout, stderr."""
        try:
            resolved = shutil.which(cmd[0])
            if resolved:
                cmd = [resolved] + cmd[1:]
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return False, "", str(e)

    def test_environment_detection(self) -> ValidationResult:
        """Test environment detection and basic setup."""
        start_time = time.perf_counter()

        try:
            # Gather environment info
            self.environment_info = {
                "platform": os.uname().sysname,
                "python_version": sys.version.split()[0],
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "virtual_env": os.environ.get("VIRTUAL_ENV"),
                "shell": os.environ.get("SHELL", "").split("/")[-1],
                "path_dirs": len(os.environ.get("PATH", "").split(":")),
            }

            # Check if in virtual environment
            is_venv = (
                hasattr(sys, "real_prefix")
                or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
                or os.environ.get("CONDA_DEFAULT_ENV") is not None
            )

            warnings = []
            if not is_venv:
                warnings.append("Not running in a virtual environment")

            execution_time = time.perf_counter() - start_time

            return ValidationResult(
                name="Environment Detection",
                success=True,
                message=f"Detected: {self.environment_info['platform']}, "
                f"Python {self.environment_info['python_version']}, "
                f"Shell: {self.environment_info['shell']}",
                details=self.environment_info,
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Environment Detection",
                success=False,
                message=f"Failed to detect environment: {e}",
                execution_time=execution_time,
            )

    def test_homodyne_installation(self) -> ValidationResult:
        """Test homodyne package installation."""
        start_time = time.perf_counter()

        try:
            # Check if homodyne commands exist
            commands = [
                "homodyne",
                "homodyne-config",
                "homodyne-post-install",
                "homodyne-cleanup",
                "homodyne-validate",
            ]
            found_commands = []
            missing_commands = []

            for cmd in commands:
                if shutil.which(cmd):
                    found_commands.append(cmd)
                else:
                    missing_commands.append(cmd)

            # Test basic command execution
            success, stdout, stderr = self.run_command(["homodyne", "--help"])
            if not success:
                execution_time = time.perf_counter() - start_time
                return ValidationResult(
                    name="Homodyne Installation",
                    success=False,
                    message="homodyne --help failed",
                    details={"stdout": stdout, "stderr": stderr},
                    execution_time=execution_time,
                )

            # Check if help output looks correct
            if "homodyne scattering analysis" not in stdout.lower():
                execution_time = time.perf_counter() - start_time
                return ValidationResult(
                    name="Homodyne Installation",
                    success=False,
                    message="homodyne help output doesn't look correct",
                    details={"help_output": stdout[:200]},
                    execution_time=execution_time,
                )

            execution_time = time.perf_counter() - start_time
            warnings = []
            if missing_commands:
                warnings.append(f"Missing commands: {', '.join(missing_commands)}")

            return ValidationResult(
                name="Homodyne Installation",
                success=True,
                message=f"Found {len(found_commands)}/{len(commands)} commands",
                details={
                    "found_commands": found_commands,
                    "missing_commands": missing_commands,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Homodyne Installation",
                success=False,
                message=f"Installation test failed: {e}",
                execution_time=execution_time,
            )

    def test_shell_completion(self) -> ValidationResult:
        """Test shell completion system."""
        start_time = time.perf_counter()

        try:
            venv_path = Path(sys.prefix)
            completion_files = []
            missing_files = []

            # Check for completion files (multiple possible configurations)
            potential_files = [
                # Simple completion mode
                venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
                venv_path / "etc" / "zsh" / "homodyne-completion.zsh",
                # Advanced completion mode
                venv_path
                / "etc"
                / "conda"
                / "activate.d"
                / "homodyne-advanced-completion.sh",
                # Future fish support
                venv_path / "share" / "fish" / "vendor_completions.d" / "homodyne.fish",
            ]

            # Also check for the main completion source file
            try:
                import homodyne

                homodyne_src_dir = Path(homodyne.__file__).parent.parent
                main_completion_file = (
                    homodyne_src_dir
                    / "homodyne"
                    / "runtime"
                    / "shell"
                    / "completion.sh"
                )
                if main_completion_file.exists():
                    potential_files.append(main_completion_file)
            except ImportError:
                pass

            for file_path in potential_files:
                if file_path.exists():
                    completion_files.append(str(file_path))
                else:
                    missing_files.append(str(file_path))

            # Test if activation scripts work
            alias_test_passed = False
            working_aliases = 0
            if completion_files:
                # Try to source a completion file using the appropriate shell
                try:
                    # Determine which shell and completion file to use
                    # Prioritize matching current shell environment
                    current_shell = self.environment_info.get("shell", "bash")
                    use_zsh = current_shell == "zsh" and any(
                        "zsh" in str(f) for f in completion_files
                    )
                    shell_cmd = "zsh" if use_zsh else "bash"

                    # Find the appropriate completion file
                    target_completion_file = completion_files[0]  # default
                    if use_zsh:
                        # Look for zsh-specific completion file
                        for f in completion_files:
                            if "zsh" in str(f):
                                target_completion_file = f
                                break

                    # Create a test script to source completion (force reload for zsh)
                    # Use shlex.quote() to prevent path injection in the inline script.
                    quoted_path = shlex.quote(target_completion_file)
                    if use_zsh:
                        test_script = f"""#!/usr/bin/env zsh
# Force reload by unsetting the loaded flag
unset _HOMODYNE_ZSH_COMPLETION_LOADED
source {quoted_path} 2>/dev/null || exit 1
# Test if core aliases were created
alias hm >/dev/null 2>&1 && echo "core_alias_works" || echo "core_alias_missing"
alias hconfig >/dev/null 2>&1 && echo "config_alias_works" || echo "config_alias_missing"
alias hexp >/dev/null 2>&1 && echo "plot_alias_works" || echo "plot_alias_missing"
alias hc-iso >/dev/null 2>&1 && echo "shortcut_alias_works" || echo "shortcut_alias_missing"
"""
                    else:
                        test_script = f"""#!/bin/bash
source {quoted_path} 2>/dev/null || exit 1
# Test if core aliases were created
alias hm >/dev/null 2>&1 && echo "core_alias_works" || echo "core_alias_missing"
alias hconfig >/dev/null 2>&1 && echo "config_alias_works" || echo "config_alias_missing"
alias hexp >/dev/null 2>&1 && echo "plot_alias_works" || echo "plot_alias_missing"
alias hc-iso >/dev/null 2>&1 && echo "shortcut_alias_works" || echo "shortcut_alias_missing"
"""

                    success, stdout, stderr = self.run_command(
                        [shell_cmd, "-c", test_script.strip()],
                    )
                    # Count how many alias categories work
                    alias_counts = {
                        "core": "core_alias_works" in stdout,
                        "config": "config_alias_works" in stdout,
                        "plot": "plot_alias_works" in stdout,
                        "shortcut": "shortcut_alias_works" in stdout,
                    }
                    working_aliases = sum(alias_counts.values())
                    alias_test_passed = (
                        working_aliases >= 2
                    )  # At least core and config should work
                except Exception:
                    logger.debug(
                        "Shell alias validation failed; continuing without alias checks"
                    )

            execution_time = time.perf_counter() - start_time
            warnings = []
            # Only warn about critical missing files (not all potential files)
            critical_missing = []
            for file_path in potential_files:
                if not file_path.exists():
                    # Only warn about files that should definitely exist
                    if "homodyne-completion.sh" in str(
                        file_path,
                    ) or "homodyne-advanced-completion.sh" in str(file_path):
                        critical_missing.append(file_path)
                    elif "completion.sh" in str(file_path):  # Main completion source
                        critical_missing.append(file_path)

            if critical_missing:
                warnings.append(
                    "Run 'homodyne-post-install --shell <your_shell>' to install completion",
                )
            if not alias_test_passed and completion_files:
                if working_aliases == 0:
                    warnings.append(
                        "No aliases working - try 'homodyne-post-install --shell zsh' to reinstall",
                    )
                elif working_aliases < 4:
                    warnings.append(
                        f"Only {working_aliases}/4 alias categories working - may need shell restart",
                    )

            success = len(completion_files) > 0
            message = f"Found {len(completion_files)} completion files"
            if alias_test_passed:
                message += f" ({working_aliases}/4 alias categories working)"
            elif working_aliases > 0:
                message += (
                    f" ({working_aliases}/4 alias categories working, some issues)"
                )
            elif completion_files:
                message += " (aliases not working)"

            return ValidationResult(
                name="Shell Completion",
                success=success,
                message=message,
                details={
                    "found_files": completion_files,
                    "missing_files": missing_files,
                    "alias_test_passed": alias_test_passed,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Shell Completion",
                success=False,
                message=f"Shell completion test failed: {e}",
                execution_time=execution_time,
            )

    def test_integration(self) -> ValidationResult:
        """Test integration between components."""
        start_time = time.perf_counter()

        try:
            # Test if post-install script works
            success, stdout, stderr = self.run_command(
                ["python", "-c", "from homodyne.post_install import main"],
            )

            post_install_works = success

            # Test cleanup script
            success, stdout, stderr = self.run_command(
                ["python", "-c", "from homodyne.uninstall_scripts import main"],
            )

            cleanup_works = success

            # Test import of main modules
            import_tests: dict[str, bool | str] = {}
            modules = [
                "homodyne.run_homodyne",
                "homodyne.create_config",
                "homodyne.post_install",
                "homodyne.uninstall_scripts",
            ]

            for module in modules:
                try:
                    __import__(module)
                    import_tests[module] = True
                except Exception as e:
                    import_tests[module] = f"Import failed: {e}"

            execution_time = time.perf_counter() - start_time

            success_count = sum(1 for v in import_tests.values() if v is True)
            total_count = len(import_tests)

            success = (
                post_install_works and cleanup_works and success_count == total_count
            )

            warnings = []
            if not post_install_works:
                warnings.append("Post-install script has issues")
            if not cleanup_works:
                warnings.append("Cleanup script has issues")
            if success_count < total_count:
                failed = [k for k, v in import_tests.items() if v is not True]
                warnings.append(f"Module import failures: {len(failed)}")

            return ValidationResult(
                name="Integration",
                success=success,
                message=f"Module imports: {success_count}/{total_count}",
                details={
                    "post_install_works": post_install_works,
                    "cleanup_works": cleanup_works,
                    "import_tests": import_tests,
                },
                execution_time=execution_time,
                warnings=warnings,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Integration",
                success=False,
                message=f"Integration test failed: {e}",
                execution_time=execution_time,
            )

    def test_dependency_versions(self) -> ValidationResult:
        """Test that all required dependencies have correct versions."""
        start_time = time.perf_counter()

        try:
            from importlib.metadata import version

            # Required versions per CLAUDE.md v3.0
            requirements = {
                "jax": {"operator": "==", "version": "0.8.0"},
                "jaxlib": {"operator": "==", "version": "0.8.0"},
                "nlsq": {"operator": ">=", "version": "0.1.0"},
                "numpyro": {
                    "operator": ">=",
                    "version": "0.18.0",
                    "upper": "0.20.0",
                },
                "numpy": {"operator": ">=", "version": "2.0.0", "upper": "3.0.0"},
                "scipy": {"operator": ">=", "version": "1.14.0"},
                "h5py": {"operator": ">=", "version": "3.10.0"},
                "pyyaml": {"operator": ">=", "version": "6.0.0"},
                "matplotlib": {"operator": ">=", "version": "3.8.0"},
            }

            results = {}
            warnings = []
            errors = []
            remediation = []

            for package, constraints in requirements.items():
                try:
                    installed_version = version(package)
                    results[package] = installed_version

                    # Parse versions for comparison
                    from packaging.version import parse as parse_version

                    installed = parse_version(installed_version)
                    required = parse_version(constraints["version"])

                    # Check version constraints
                    operator = constraints["operator"]
                    if operator == "==":
                        if installed != required:
                            errors.append(
                                f"{package}: expected {constraints['version']}, "
                                f"found {installed_version}",
                            )
                    elif operator == ">=":
                        if installed < required:
                            errors.append(
                                f"{package}: expected >={constraints['version']}, "
                                f"found {installed_version}",
                            )
                        # Check upper bound if specified
                        if "upper" in constraints:
                            upper = parse_version(constraints["upper"])
                            if installed >= upper:
                                warnings.append(
                                    f"{package}: version {installed_version} may be "
                                    f"incompatible (expected <{constraints['upper']})",
                                )

                except Exception:
                    errors.append(f"{package}: not installed or version check failed")
                    results[package] = "NOT INSTALLED"

            # CRITICAL: Check JAX/jaxlib version match
            if "jax" in results and "jaxlib" in results:
                if results["jax"] != results["jaxlib"]:
                    errors.append(
                        f"JAX/jaxlib version mismatch: jax={results['jax']}, "
                        f"jaxlib={results['jaxlib']}. These must match exactly!",
                    )
                    remediation.append(
                        "pip install jax==0.8.0 jaxlib==0.8.0  # Fix version mismatch",
                    )

            # Add general remediation if errors found
            if errors and not remediation:
                remediation.append(
                    "pip install jax==0.8.0 jaxlib==0.8.0 nlsq>=0.1.0",
                )
                remediation.append("pip install 'numpyro>=0.18.0,<0.20.0'")
                remediation.append("pip install 'numpy>=2.0.0,<3.0.0'")

            execution_time = time.perf_counter() - start_time
            success = len(errors) == 0

            message = f"Validated {len(results)} packages"
            if errors:
                message = f"Found {len(errors)} version issues"

            return ValidationResult(
                name="Dependency Versions",
                success=success,
                message=message,
                details=results,
                execution_time=execution_time,
                warnings=warnings if warnings else None,
                severity="critical" if errors else "info",
                remediation=remediation if remediation else None,
                error_code="EDEP_001" if errors else None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Dependency Versions",
                success=False,
                message=f"Version check failed: {e}",
                execution_time=execution_time,
                severity="critical",
                error_code="EDEP_000",
            )

    def test_jax_installation(self) -> ValidationResult:
        """Test JAX installation and platform compatibility."""
        start_time = time.perf_counter()

        try:
            import jax
            import jax.numpy as jnp

            # Ensure environment_info is populated
            if not self.environment_info:
                self.test_environment_detection()

            platform = self.environment_info.get("platform", "Unknown")
            warnings: list[str] = []
            remediation: list[str] = []
            details: dict[str, Any] = {}

            # Test 1: JAX imports successfully
            details["jax_version"] = jax.__version__
            details["jax_import"] = "success"

            # Test 2: Enumerate devices
            try:
                devices = jax.devices()
                cpu_devices = [d for d in devices if "cpu" in str(d).lower()]
                gpu_devices = [
                    d
                    for d in devices
                    if "gpu" in str(d).lower() or "cuda" in str(d).lower()
                ]

                details["total_devices"] = len(devices)
                details["cpu_devices"] = len(cpu_devices)
                details["gpu_devices"] = len(gpu_devices)
                details["devices"] = [str(d) for d in devices]

            except Exception as e:
                details["device_enumeration_error"] = str(e)
                warnings.append(f"Device enumeration failed: {e}")

            # Test 3: Platform-specific validation
            if platform not in ["Linux", "Darwin", "Windows"]:
                warnings.append(f"Unknown platform: {platform}")

            # Note: GPU support removed in v2.3.0 - Homodyne is CPU-only
            # All platforms now use CPU-only JAX configuration

            # Test 4: Test JIT compilation
            try:

                @jax.jit
                def test_fn(x: Any) -> Any:
                    return x**2

                result = test_fn(jnp.array([1.0, 2.0, 3.0]))
                expected = jnp.array([1.0, 4.0, 9.0])

                if jnp.allclose(result, expected):
                    details["jit_compilation"] = "success"
                else:
                    details["jit_compilation"] = "failed"
                    warnings.append("JIT compilation produced incorrect results")

            except Exception as e:
                details["jit_compilation"] = "failed"
                warnings.append(f"JIT compilation test failed: {e}")

            execution_time = time.perf_counter() - start_time

            # Build message
            message = f"JAX {jax.__version__}: "
            if details.get("cpu_devices", 0) > 0:
                message += f"{details['cpu_devices']} CPU device(s)"
            if details.get("gpu_devices", 0) > 0:
                message += f", {details['gpu_devices']} GPU device(s)"
            if details.get("jit_compilation") == "success":
                message += ", JIT works"

            return ValidationResult(
                name="JAX Installation",
                success=True,
                message=message,
                details=details,
                execution_time=execution_time,
                warnings=warnings if warnings else None,
                severity="info" if not warnings else "warning",
                remediation=remediation if remediation else None,
            )

        except ImportError as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="JAX Installation",
                success=False,
                message=f"JAX not installed: {e}",
                execution_time=execution_time,
                severity="critical",
                remediation=[
                    "pip install jax==0.8.0 jaxlib==0.8.0  # CPU-only (all platforms)",
                    "pip install jax[cuda12-local]==0.8.0  # GPU (Linux only, after CPU install)",
                ],
                error_code="EJAX_001",
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="JAX Installation",
                success=False,
                message=f"JAX validation failed: {e}",
                execution_time=execution_time,
                severity="critical",
                error_code="EJAX_000",
            )

    def test_nlsq_integration(self) -> ValidationResult:
        """Test NLSQ optimization engine integration."""
        start_time = time.perf_counter()

        try:
            import nlsq

            warnings = []
            remediation: list[str] = []
            details: dict[str, Any] = {}

            # Test 1: NLSQ version check
            try:
                nlsq_version = nlsq.__version__
                details["nlsq_version"] = nlsq_version

                from packaging.version import parse as parse_version

                installed = parse_version(nlsq_version)
                required = parse_version("0.1.0")

                if installed < required:
                    warnings.append(
                        f"NLSQ version {nlsq_version} is below recommended "
                        f"minimum 0.1.0",
                    )
                    remediation.append("pip install --upgrade nlsq>=0.1.0")

                details["version_check"] = (
                    "pass" if installed >= required else "warning"
                )

            except AttributeError:
                details["nlsq_version"] = "unknown"
                warnings.append("Could not determine NLSQ version")

            # Test 2: Core NLSQ functions import
            try:
                from nlsq import curve_fit, curve_fit_large  # noqa: F401 - Import test

                details["core_functions"] = "available"
            except ImportError as e:
                details["core_functions"] = "missing"
                warnings.append(f"NLSQ core functions not available: {e}")
                remediation.append("pip install --upgrade nlsq>=0.1.0")

            # Test 3: AdaptiveHybridStreamingOptimizer availability (v0.3.2+ feature)
            # Note: The old StreamingOptimizer was removed in NLSQ 0.4.0
            streaming_available = False
            try:
                from nlsq import (  # noqa: F401 - Import test
                    AdaptiveHybridStreamingOptimizer,
                )

                details["hybrid_streaming_optimizer"] = "available"
                streaming_available = True
            except ImportError:
                details["hybrid_streaming_optimizer"] = "not available"
                # Only warn if version is recent enough to expect it
                if "nlsq_version" in details:
                    try:
                        version = parse_version(details["nlsq_version"])
                        if version >= parse_version("0.3.2"):
                            warnings.append(
                                "AdaptiveHybridStreamingOptimizer not available despite "
                                "NLSQ >= 0.3.2",
                            )
                    except Exception as exc:
                        logger.debug(
                            "Version parsing for hybrid streaming optimizer failed: %s",
                            exc,
                        )

            # Test 4: Homodyne NLSQ integration
            homodyne_integration_ok = True
            try:
                from homodyne.optimization.nlsq.wrapper import NLSQWrapper  # noqa: F401

                details["nlsq_wrapper"] = "available"
            except ImportError as e:
                details["nlsq_wrapper"] = "missing"
                warnings.append(f"Homodyne NLSQWrapper not available: {e}")
                homodyne_integration_ok = False

            # Test 5: Unified memory-based strategy selection (v2.13.0+)
            try:
                from homodyne.optimization.nlsq.memory import (  # noqa: F401
                    NLSQStrategy,
                    select_nlsq_strategy,
                )

                details["strategy_selection"] = "available"
            except ImportError as e:
                details["strategy_selection"] = "missing"
                warnings.append(f"Strategy selection not available: {e}")
                homodyne_integration_ok = False

            # Test 6: Checkpoint manager (if streaming available)
            if streaming_available:
                try:
                    from homodyne.optimization.checkpoint_manager import (  # noqa: F401
                        CheckpointManager,
                    )

                    details["checkpoint_manager"] = "available"
                except ImportError:
                    details["checkpoint_manager"] = "missing"
                    warnings.append(
                        "CheckpointManager not available (needed for streaming)",
                    )

            execution_time = time.perf_counter() - start_time

            # Build message
            message = f"NLSQ {details.get('nlsq_version', 'unknown')}"
            if streaming_available:
                message += " with AdaptiveHybridStreamingOptimizer"
            if homodyne_integration_ok:
                message += ", homodyne integration OK"

            # Success if basic imports work
            success = details.get("core_functions") == "available"

            return ValidationResult(
                name="NLSQ Integration",
                success=success,
                message=message,
                details=details,
                execution_time=execution_time,
                warnings=warnings if warnings else None,
                severity="warning" if warnings else "info",
                remediation=remediation if remediation else None,
            )

        except ImportError as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="NLSQ Integration",
                success=False,
                message=f"NLSQ not installed: {e}",
                execution_time=execution_time,
                severity="critical",
                remediation=[
                    "pip install nlsq>=0.1.0  # Required for optimization",
                    "pip install nlsq>=0.1.5  # Recommended for streaming support",
                ],
                error_code="ENLSQ_001",
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="NLSQ Integration",
                success=False,
                message=f"NLSQ validation failed: {e}",
                execution_time=execution_time,
                severity="critical",
                error_code="ENLSQ_000",
            )

    def test_config_system(self) -> ValidationResult:
        """Test configuration system functionality."""
        start_time = time.perf_counter()

        try:
            warnings = []
            remediation: list[str] = []
            details: dict[str, Any] = {}

            # Test 1: ConfigManager import
            try:
                from homodyne.config.manager import ConfigManager

                details["config_manager"] = "available"
            except ImportError as e:
                details["config_manager"] = "missing"
                return ValidationResult(
                    name="Config System",
                    success=False,
                    message=f"ConfigManager not available: {e}",
                    execution_time=time.perf_counter() - start_time,
                    severity="critical",
                    error_code="ECONFIG_001",
                )

            # Test 2: ParameterManager import
            try:
                from homodyne.config.parameter_manager import (  # noqa: F401
                    ParameterManager,
                )

                details["parameter_manager"] = "available"
            except ImportError as e:
                details["parameter_manager"] = "missing"
                warnings.append(f"ParameterManager not available: {e}")

            # Test 3: Check template directory exists
            try:
                import homodyne

                homodyne_path = Path(homodyne.__file__).parent
                template_dir = homodyne_path / "config" / "templates"

                if template_dir.exists():
                    details["template_directory"] = str(template_dir)
                    details["template_dir_exists"] = True

                    # Count template files
                    template_files = list(template_dir.glob("*.yaml"))
                    details["template_count"] = len(template_files)
                    details["template_files"] = [f.name for f in template_files]

                    # Expected templates per CLAUDE.md
                    expected_templates = [
                        "homodyne_streaming_config.yaml",
                    ]

                    missing_templates = [
                        t
                        for t in expected_templates
                        if t not in details["template_files"]
                    ]
                    if missing_templates:
                        warnings.append(
                            f"Missing expected templates: {', '.join(missing_templates)}",
                        )
                else:
                    details["template_dir_exists"] = False
                    warnings.append(f"Template directory not found: {template_dir}")

            except Exception as e:
                details["template_check_error"] = str(e)
                warnings.append(f"Template directory check failed: {e}")

            # Test 4: Try loading a template config (if available)
            config_load_ok = False
            if details.get("template_count", 0) > 0:
                try:
                    # Try to load the first template
                    template_file = template_dir / details["template_files"][0]
                    config_mgr = ConfigManager(str(template_file))
                    details["template_load_test"] = "success"
                    config_load_ok = True

                    # Test parameter bounds retrieval
                    try:
                        _bounds = config_mgr.get_parameter_bounds()  # noqa: F841
                        details["parameter_bounds_retrieval"] = "success"
                    except Exception as e:
                        details["parameter_bounds_retrieval"] = "failed"
                        warnings.append(f"Parameter bounds retrieval failed: {e}")

                except Exception as e:
                    details["template_load_test"] = "failed"
                    warnings.append(f"Template loading failed: {e}")

            # Test 5: Check for deprecated config paths
            deprecated_paths = [
                "performance.subsampling",
                "optimization_performance.time_subsampling",
            ]
            details["deprecated_paths_checked"] = deprecated_paths

            execution_time = time.perf_counter() - start_time

            # Build message
            message = "ConfigManager available"
            if details.get("template_count", 0) > 0:
                message += f", {details['template_count']} template(s) found"
            if config_load_ok:
                message += ", config loading works"

            success = details.get("config_manager") == "available" and details.get(
                "template_dir_exists", False
            )

            return ValidationResult(
                name="Config System",
                success=success,
                message=message,
                details=details,
                execution_time=execution_time,
                warnings=warnings if warnings else None,
                severity="warning" if warnings else "info",
                remediation=remediation if remediation else None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Config System",
                success=False,
                message=f"Config system validation failed: {e}",
                execution_time=execution_time,
                severity="critical",
                error_code="ECONFIG_000",
            )

    def test_data_pipeline(self) -> ValidationResult:
        """Test data loading pipeline functionality."""
        start_time = time.perf_counter()

        try:
            warnings = []
            remediation: list[str] = []
            details: dict[str, Any] = {}

            # Test 1: h5py import and functionality
            try:
                import h5py

                details["h5py_version"] = h5py.__version__
                details["h5py_import"] = "success"

                # Test basic HDF5 operations (create temp file).
                # Use mkstemp + explicit unlink instead of NamedTemporaryFile(delete=True)
                # to avoid a TOCTOU race: h5py needs the file closed before it can
                # open it on some platforms, and delete=True removes it on __exit__.
                import tempfile

                try:
                    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".hdf5")
                    try:
                        os.close(tmp_fd)
                        with h5py.File(tmp_name, "w") as f:
                            f.create_dataset("test", data=[1, 2, 3])
                        with h5py.File(tmp_name, "r") as f:
                            _ = f["test"][:]
                        details["hdf5_readwrite"] = "success"
                    finally:
                        try:
                            os.unlink(tmp_name)
                        except OSError:
                            pass
                except Exception as e:
                    details["hdf5_readwrite"] = "failed"
                    warnings.append(f"HDF5 file operations failed: {e}")

            except ImportError as e:
                details["h5py_import"] = "missing"
                return ValidationResult(
                    name="Data Pipeline",
                    success=False,
                    message=f"h5py not installed: {e}",
                    execution_time=time.perf_counter() - start_time,
                    severity="critical",
                    remediation=["pip install h5py>=3.10.0"],
                    error_code="EDATA_001",
                )

            # Test 2: XPCSDataLoader import
            try:
                from homodyne.data.xpcs_loader import XPCSDataLoader  # noqa: F401

                details["xpcs_dataloader"] = "available"
            except ImportError as e:
                details["xpcs_dataloader"] = "missing"
                warnings.append(f"XPCSDataLoader not available: {e}")

            # Test 3: Phi filtering import
            try:
                from homodyne.data.angle_filtering import (
                    normalize_angle_to_symmetric_range,
                )

                details["phi_filtering"] = "available"

                # Test angle normalization function
                try:
                    test_angle = float(normalize_angle_to_symmetric_range(270.0))
                    if abs(test_angle - (-90.0)) < 1e-6:
                        details["phi_filtering_test"] = "success"
                    else:
                        details["phi_filtering_test"] = "unexpected_result"
                        warnings.append(
                            f"Phi filtering test produced unexpected result: {test_angle}",
                        )
                except Exception as e:
                    details["phi_filtering_test"] = "failed"
                    warnings.append(f"Phi filtering test failed: {e}")

            except ImportError as e:
                details["phi_filtering"] = "missing"
                warnings.append(f"Phi filtering not available: {e}")

            # Test 4: Memory manager import
            try:
                from homodyne.data.memory_manager import (  # noqa: F401
                    AdvancedMemoryManager,
                )

                details["memory_manager"] = "available"
            except ImportError as e:
                details["memory_manager"] = "missing"
                warnings.append(f"Memory manager not available: {e}")

            # Test 5: Data preprocessing import
            try:
                from homodyne.data.preprocessing import (
                    preprocess_xpcs_data,  # noqa: F401
                )

                details["preprocessing"] = "available"
            except ImportError as e:
                details["preprocessing"] = "missing"
                warnings.append(f"Data preprocessing not available: {e}")

            # Test 6: Check for example data files (optional)
            try:
                import homodyne

                homodyne_path = Path(homodyne.__file__).parent.parent
                scripts_dir = homodyne_path / "scripts"
                tests_dir = homodyne_path / "tests"

                test_data_locations = []
                if scripts_dir.exists():
                    hdf5_files = list(scripts_dir.rglob("*.hdf5")) + list(
                        scripts_dir.rglob("*.hdf"),
                    )
                    if hdf5_files:
                        test_data_locations.append(
                            f"scripts/ ({len(hdf5_files)} files)"
                        )

                if tests_dir.exists():
                    hdf5_files = list(tests_dir.rglob("*.hdf5")) + list(
                        tests_dir.rglob("*.hdf"),
                    )
                    if hdf5_files:
                        test_data_locations.append(f"tests/ ({len(hdf5_files)} files)")

                if test_data_locations:
                    details["test_data_available"] = ", ".join(test_data_locations)
                else:
                    details["test_data_available"] = "none found"

            except Exception as e:
                details["test_data_check_error"] = str(e)

            execution_time = time.perf_counter() - start_time

            # Build message
            message = f"h5py {details.get('h5py_version', 'unknown')}"
            components = []
            if details.get("xpcs_dataloader") == "available":
                components.append("XPCSDataLoader")
            if details.get("phi_filtering") == "available":
                components.append("phi filtering")
            if details.get("memory_manager") == "available":
                components.append("memory manager")

            if components:
                message += f", {', '.join(components)} available"

            success = (
                details.get("h5py_import") == "success"
                and details.get("hdf5_readwrite") == "success"
                and details.get("xpcs_dataloader") == "available"
            )

            return ValidationResult(
                name="Data Pipeline",
                success=success,
                message=message,
                details=details,
                execution_time=execution_time,
                warnings=warnings if warnings else None,
                severity="warning" if warnings else "info",
                remediation=remediation if remediation else None,
            )

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return ValidationResult(
                name="Data Pipeline",
                success=False,
                message=f"Data pipeline validation failed: {e}",
                execution_time=execution_time,
                severity="critical",
                error_code="EDATA_000",
            )

    def run_all_tests(self) -> dict[str, ValidationResult]:
        """Run all system tests (9 tests total)."""
        tests = [
            self.test_environment_detection,
            self.test_dependency_versions,
            self.test_jax_installation,
            self.test_nlsq_integration,
            self.test_config_system,
            self.test_data_pipeline,
            self.test_homodyne_installation,
            self.test_shell_completion,
            self.test_integration,
        ]

        results = {}

        for test_func in tests:
            self.log(f"Running {test_func.__name__}...")
            result = test_func()
            results[result.name] = result
            self.results.append(result)

            status = "[PASS]" if result.success else "[FAIL]"
            self.log(f"{status}: {result.name} - {result.message}")

            if result.warnings:
                for warning in result.warnings:
                    self.log(f"[WARN] {warning}")

        return results

    def run_quick_tests(self) -> dict[str, ValidationResult]:
        """Run quick critical tests only (for fast CI/CD feedback)."""
        # Quick tests: only run critical dependency and JAX validation
        # Skips: homodyne installation, shell completion, GPU setup, integration
        quick_tests = [
            self.test_environment_detection,
            self.test_dependency_versions,
            self.test_jax_installation,
        ]

        results = {}

        for test_func in quick_tests:
            self.log(f"Running {test_func.__name__}...")
            result = test_func()
            results[result.name] = result
            self.results.append(result)

            status = "[PASS]" if result.success else "[FAIL]"
            self.log(f"{status}: {result.name} - {result.message}")

            if result.warnings:
                for warning in result.warnings:
                    self.log(f"[WARN] {warning}")

        return results

    def calculate_health_score(self) -> int:
        """Calculate overall system health score (0-100) across 9 validation tests.

        Weight Distribution (totaling 100%):
        - Dependency Versions: 20% (critical - exact JAX/jaxlib match required)
        - JAX Installation: 20% (critical - core computation engine)
        - NLSQ Integration: 15% (important - primary optimization method)
        - Config System: 10% (important - configuration management)
        - Data Pipeline: 10% (important - HDF5 data loading)
        - Homodyne Installation: 10% (important - CLI commands available)
        - Environment Detection: 5% (baseline - platform/Python/shell)
        - Shell Completion: 5% (convenience - aliases and completion)
        - Integration: 5% (optional - cross-component integration)

        Note: GPU validation removed in v2.3.0 (CPU-only architecture).
        Integration weight increased from 2% to 5% to absorb GPU test weight.
        """
        if not self.results:
            return 0

        # Weight factors for different test categories (9 tests, totaling 100%)
        weights = {
            "Environment Detection": 5,
            "Dependency Versions": 20,
            "JAX Installation": 20,
            "NLSQ Integration": 15,
            "Config System": 10,
            "Data Pipeline": 10,
            "Homodyne Installation": 10,
            "Shell Completion": 5,
            "Integration": 5,  # Increased from 2% (absorbed GPU test weight)
        }

        total_weight = 0
        earned_score = 0.0

        for result in self.results:
            weight = weights.get(result.name, 5)
            total_weight += weight

            if result.success:
                # Full points for success
                earned_score += weight
            elif result.warnings and not result.success:
                # Partial credit if test "passed" but has warnings
                earned_score += weight * 0.5
            # else: 0 points for failure

            # Deduct points for warnings even on success
            if result.warnings:
                warning_penalty = min(len(result.warnings) * 2, weight * 0.2)
                earned_score -= warning_penalty

        if total_weight == 0:
            return 0

        score = int((earned_score / total_weight) * 100)
        return max(0, min(100, score))  # Clamp to 0-100

    def generate_report(self) -> str:
        """Generate comprehensive validation report."""
        if not self.results:
            return "No tests have been run."

        report = []

        # Header
        report.append("=" * 80)
        report.append("HOMODYNE SYSTEM VALIDATION REPORT")
        report.append("=" * 80)

        # Health Score
        health_score = self.calculate_health_score()
        health_status = (
            "Excellent"
            if health_score >= 90
            else (
                "Good"
                if health_score >= 70
                else "Fair"
                if health_score >= 50
                else "Poor"
            )
        )
        report.append(
            f"\nHealth Score: {health_score}/100 ({health_status})"
        )

        # Summary
        passed = sum(1 for r in self.results if r.success)
        total = len(self.results)
        report.append(f"Summary: {passed}/{total} tests passed")

        if passed == total:
            report.append("All systems operational!")
        else:
            report.append("[WARN] Some issues detected - see details below")

        # Environment info
        if self.environment_info:
            report.append("\nEnvironment:")
            for key, value in self.environment_info.items():
                report.append(f"   {key}: {value}")

        # Test results
        report.append("\nTest Results:")
        report.append("-" * 40)

        for result in self.results:
            status = "[PASS]" if result.success else "[FAIL]"
            report.append(f"\n{status} {result.name}")
            report.append(f"   Message: {result.message}")
            report.append(f"   Time: {result.execution_time:.3f}s")

            if result.warnings:
                report.append("   Warnings:")
                for warning in result.warnings:
                    report.append(f"     [WARN] {warning}")

            if result.remediation:
                report.append("   Remediation:")
                for fix in result.remediation:
                    report.append(f"     [FIX] {fix}")

            if result.details and self.verbose:
                report.append("   Details:")
                for key, value in result.details.items():
                    if isinstance(value, list | dict):
                        report.append(
                            f"     {key}: {len(value) if isinstance(value, list) else 'dict'} items",
                        )
                    else:
                        report.append(f"     {key}: {value}")

        # Recommendations
        report.append("\nRecommendations:")

        failed_tests = [r for r in self.results if not r.success]
        if failed_tests:
            report.append("   Fix failed tests:")
            for test in failed_tests:
                report.append(f"     - {test.name}: {test.message}")

        warnings_count = sum(len(r.warnings or []) for r in self.results)
        if warnings_count > 0:
            report.append(
                f"   Address {warnings_count} warnings for optimal performance",
            )

        if passed == total:
            report.append("   Your homodyne installation is ready!")
            report.append("   Check documentation for usage examples")

        report.append("\n" + "=" * 80)

        return "\n".join(report)


def main() -> None:
    """Main function for system validation CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Homodyne System Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-validate              # Quick validation
  homodyne-validate --verbose    # Detailed output
  homodyne-validate --json       # JSON output for automation
        """,
    )

    # Note: --verbose argument removed to avoid conflicts with main CLI\n    # Verbose mode controlled internally via logging level
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (critical tests only for fast CI/CD)",
    )
    parser.add_argument(
        "--test",
        choices=[
            "env",
            "deps",
            "jax",
            "nlsq",
            "config",
            "data",
            "install",
            "completion",
            "integration",
        ],
        help="Run specific test only",
    )

    args = parser.parse_args()

    validator = SystemValidator(verbose=True)  # Always verbose for validation tool

    if args.test:
        # Run specific test
        test_map = {
            "env": validator.test_environment_detection,
            "deps": validator.test_dependency_versions,
            "jax": validator.test_jax_installation,
            "nlsq": validator.test_nlsq_integration,
            "config": validator.test_config_system,
            "data": validator.test_data_pipeline,
            "install": validator.test_homodyne_installation,
            "completion": validator.test_shell_completion,
            "integration": validator.test_integration,
        }

        result = test_map[args.test]()

        if args.json:
            print(json.dumps(asdict(result), indent=2))
        else:
            status = "[PASS]" if result.success else "[FAIL]"
            print(f"{status} {result.name}: {result.message}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"[WARN] {warning}")
            if result.remediation:
                print("\n🔧 Remediation:")
                for fix in result.remediation:
                    print(f"   {fix}")

        sys.exit(0 if result.success else 1)
    elif args.quick:
        # Run quick tests only
        results = validator.run_quick_tests()

        if args.json:
            json_results = {name: asdict(result) for name, result in results.items()}
            print(json.dumps(json_results, indent=2))
        else:
            print(validator.generate_report())

        # Exit with error code if any test failed
        sys.exit(0 if all(r.success for r in results.values()) else 1)
    else:
        # Run all tests
        results = validator.run_all_tests()

        if args.json:
            json_results = {name: asdict(result) for name, result in results.items()}
            print(json.dumps(json_results, indent=2))
        else:
            print(validator.generate_report())

        # Exit with error code if any test failed
        sys.exit(0 if all(r.success for r in results.values()) else 1)


if __name__ == "__main__":
    main()
