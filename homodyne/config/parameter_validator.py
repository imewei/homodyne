"""
CLI Parameter Validation for Homodyne v2
========================================

Comprehensive parameter validation integrating with CLI validators
and providing configuration-specific validation logic.

Key Features:
- Mode-specific parameter validation (3 vs 7 parameters)
- Range checking and physics constraints
- Cross-parameter consistency validation
- User-friendly error messages and suggestions
- Integration with CLI validation pipeline
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import psutil

from homodyne.utils.logging import get_logger

# Try to import GPU libraries for hardware detection
try:
    import GPUtil

    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import pynvml

    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import jax

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Parameter validation result container.

    Attributes:
        is_valid: Whether validation passed
        errors: List of validation errors
        warnings: List of validation warnings
        suggestions: List of improvement suggestions
        info: List of informational messages
        hardware_info: Dict with hardware detection results
    """

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    info: List[str]
    hardware_info: Dict[str, Any]


class ParameterValidator:
    """
    Configuration parameter validator with mode-aware validation.

    Provides comprehensive validation of analysis parameters with
    physics-based constraints and cross-parameter consistency checking.
    """

    def __init__(self):
        """Initialize parameter validator."""
        self.physics_constraints = self._get_physics_constraints()
        self.mode_parameter_counts = {
            "static_isotropic": 3,
            "static_anisotropic": 3,
            "laminar_flow": 7,
        }
        # Cache hardware information for performance
        self._hardware_cache = None
        self._gpu_info_cache = None

    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate complete configuration dictionary.

        Args:
            config: Configuration to validate

        Returns:
            Validation result with errors/warnings
        """
        logger.debug("Validating configuration parameters")

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        # Validate core configuration structure
        self._validate_config_structure(config, result)

        # Validate analysis mode
        self._validate_analysis_mode(config, result)

        # Validate optimization parameters
        self._validate_optimization_params(config, result)

        # Validate hardware settings
        self._validate_hardware_params(config, result)

        # Validate data parameters
        self._validate_data_params(config, result)

        # Validate cross-parameter consistency
        self._validate_parameter_consistency(config, result)

        # Validate multi-method configurations
        self._validate_multi_method_workflow(config, result)

        # Validate HPC configurations if present
        self._validate_hpc_configuration(config, result)

        # Validate advanced scenarios
        self._validate_advanced_scenarios(config, result)

        # Detect and validate hardware configuration
        self._detect_and_validate_hardware(config, result)

        # Set final validation status
        result.is_valid = len(result.errors) == 0

        if result.is_valid:
            logger.debug("✓ Configuration validation passed")
        else:
            logger.debug(
                f"✗ Configuration validation failed: {len(result.errors)} errors"
            )

        return result

    def validate_analysis_parameters(
        self, mode: str, parameters: Dict[str, float]
    ) -> ValidationResult:
        """
        Validate analysis-specific parameters for given mode.

        Args:
            mode: Analysis mode
            parameters: Parameter dictionary to validate

        Returns:
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        # Check parameter count
        expected_count = self.mode_parameter_counts.get(mode, 3)
        actual_count = len(parameters)

        if actual_count != expected_count:
            result.errors.append(
                f"Parameter count mismatch for {mode}: expected {expected_count}, got {actual_count}"
            )

        # Validate individual parameters based on mode
        if mode in ["static_isotropic", "static_anisotropic"]:
            self._validate_static_parameters(parameters, result)
        elif mode == "laminar_flow":
            self._validate_flow_parameters(parameters, result)

        # Enhanced physics validation
        self._validate_enhanced_physics_constraints(parameters, mode, result)

        result.is_valid = len(result.errors) == 0
        return result

    def _validate_config_structure(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate basic configuration structure.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        required_sections = ["analysis_mode", "optimization"]

        for section in required_sections:
            if section not in config:
                result.errors.append(
                    f"Missing required configuration section: {section}"
                )

        # Check optimization subsections
        if "optimization" in config:
            opt_config = config["optimization"]
            if not isinstance(opt_config, dict):
                result.errors.append("optimization section must be a dictionary")
            else:
                required_opt_sections = ["vi", "mcmc"]
                for opt_section in required_opt_sections:
                    if opt_section not in opt_config:
                        result.warnings.append(
                            f"Missing optimization section: {opt_section}"
                        )

    def _validate_analysis_mode(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate analysis mode specification.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        mode = config.get("analysis_mode")

        if not mode:
            result.errors.append("analysis_mode is required")
            return

        valid_modes = list(self.mode_parameter_counts.keys()) + ["auto-detect"]

        if mode not in valid_modes:
            result.errors.append(
                f"Invalid analysis_mode '{mode}'. Valid options: {valid_modes}"
            )

        # Mode-specific validation hints
        if mode == "auto-detect":
            result.suggestions.append(
                "Consider specifying explicit mode for reproducible analysis"
            )

    def _validate_optimization_params(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate optimization parameters.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        opt_config = config.get("optimization", {})

        # Validate VI parameters
        self._validate_vi_params(opt_config.get("vi", {}), result)

        # Validate MCMC parameters
        self._validate_mcmc_params(opt_config.get("mcmc", {}), result)

        # Validate hybrid parameters
        if "hybrid" in opt_config:
            self._validate_hybrid_params(opt_config["hybrid"], result)

    def _validate_vi_params(
        self, vi_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate Variational Inference parameters.

        Args:
            vi_config: VI configuration dictionary
            result: Validation result to update
        """
        # Iterations validation
        n_iterations = vi_config.get("n_iterations", 2000)
        if not isinstance(n_iterations, int) or n_iterations <= 0:
            result.errors.append("VI n_iterations must be positive integer")
        elif n_iterations < 100:
            result.warnings.append(
                "VI n_iterations < 100 may be insufficient for convergence"
            )
        elif n_iterations > 10000:
            result.warnings.append("VI n_iterations > 10000 may be unnecessarily long")

        # Learning rate validation
        learning_rate = vi_config.get("learning_rate", 0.01)
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            result.errors.append("VI learning_rate must be positive number")
        elif learning_rate > 0.1:
            result.warnings.append("VI learning_rate > 0.1 may cause instability")
        elif learning_rate < 0.001:
            result.warnings.append("VI learning_rate < 0.001 may converge slowly")

        # Convergence tolerance validation
        conv_tol = vi_config.get("convergence_tol", 1e-6)
        if not isinstance(conv_tol, (int, float)) or conv_tol <= 0:
            result.errors.append("VI convergence_tol must be positive number")
        elif conv_tol > 1e-3:
            result.warnings.append(
                "VI convergence_tol > 1e-3 may terminate prematurely"
            )

    def _validate_mcmc_params(
        self, mcmc_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate MCMC parameters.

        Args:
            mcmc_config: MCMC configuration dictionary
            result: Validation result to update
        """
        # Samples validation
        n_samples = mcmc_config.get("n_samples", 1000)
        if not isinstance(n_samples, int) or n_samples <= 0:
            result.errors.append("MCMC n_samples must be positive integer")
        elif n_samples < 500:
            result.warnings.append(
                "MCMC n_samples < 500 may provide poor posterior estimates"
            )

        # Warmup validation
        n_warmup = mcmc_config.get("n_warmup", 1000)
        if not isinstance(n_warmup, int) or n_warmup <= 0:
            result.errors.append("MCMC n_warmup must be positive integer")
        elif n_warmup < n_samples * 0.5:
            result.warnings.append("MCMC n_warmup should be at least 50% of n_samples")

        # Chains validation
        n_chains = mcmc_config.get("n_chains", 4)
        if not isinstance(n_chains, int) or n_chains <= 0:
            result.errors.append("MCMC n_chains must be positive integer")
        elif n_chains < 2:
            result.warnings.append("MCMC n_chains < 2 prevents convergence diagnostics")
        elif n_chains > 8:
            result.warnings.append(
                "MCMC n_chains > 8 may not improve results significantly"
            )

        # Target accept probability validation
        target_accept = mcmc_config.get("target_accept_prob", 0.8)
        if not isinstance(target_accept, (int, float)) or not (0 < target_accept < 1):
            result.errors.append("MCMC target_accept_prob must be between 0 and 1")
        elif target_accept < 0.5:
            result.warnings.append(
                "MCMC target_accept_prob < 0.5 may indicate poor mixing"
            )
        elif target_accept > 0.95:
            result.warnings.append(
                "MCMC target_accept_prob > 0.95 may be unnecessarily conservative"
            )

    def _validate_hybrid_params(
        self, hybrid_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate hybrid method parameters.

        Args:
            hybrid_config: Hybrid configuration dictionary
            result: Validation result to update
        """
        # VI initialization flag
        use_vi_init = hybrid_config.get("use_vi_init", True)
        if not isinstance(use_vi_init, bool):
            result.errors.append("Hybrid use_vi_init must be boolean")

        # Convergence threshold
        conv_threshold = hybrid_config.get("convergence_threshold", 0.1)
        if not isinstance(conv_threshold, (int, float)) or conv_threshold <= 0:
            result.errors.append("Hybrid convergence_threshold must be positive number")
        elif conv_threshold > 1.0:
            result.warnings.append(
                "Hybrid convergence_threshold > 1.0 may be too lenient"
            )

    def _validate_hardware_params(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate hardware configuration parameters.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        hardware_config = config.get("hardware", {})

        # GPU memory fraction
        gpu_memory = hardware_config.get("gpu_memory_fraction", 0.8)
        if not isinstance(gpu_memory, (int, float)) or not (0 < gpu_memory <= 1):
            result.errors.append("gpu_memory_fraction must be between 0 and 1")
        elif gpu_memory > 0.95:
            result.warnings.append("gpu_memory_fraction > 0.95 may cause memory issues")

        # Force CPU flag
        force_cpu = hardware_config.get("force_cpu", False)
        if not isinstance(force_cpu, bool):
            result.errors.append("force_cpu must be boolean")

    def _validate_data_params(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate data-related parameters.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        data_config = config.get("data", {})

        # File path validation
        file_path = data_config.get("file_path")
        if file_path:
            from pathlib import Path

            if not Path(file_path).exists():
                result.errors.append(f"Data file not found: {file_path}")

        # Dataset optimization flag
        dataset_opt = data_config.get("dataset_optimization", True)
        if not isinstance(dataset_opt, bool):
            result.errors.append("dataset_optimization must be boolean")

        # Custom phi angles validation
        custom_phi = data_config.get("custom_phi_angles")
        if custom_phi is not None:
            self._validate_phi_angles(custom_phi, result)

    def _validate_phi_angles(self, phi_angles: Any, result: ValidationResult) -> None:
        """
        Validate phi angles specification.

        Args:
            phi_angles: Phi angles to validate
            result: Validation result to update
        """
        if isinstance(phi_angles, str):
            # Parse comma-separated string
            try:
                angles = [float(x.strip()) for x in phi_angles.split(",") if x.strip()]
            except ValueError:
                result.errors.append(
                    "Invalid phi angles format - must be comma-separated numbers"
                )
                return
        elif isinstance(phi_angles, (list, tuple)):
            angles = phi_angles
        else:
            result.errors.append("phi_angles must be string, list, or tuple")
            return

        # Check angle values
        for angle in angles:
            if not isinstance(angle, (int, float)):
                result.errors.append("All phi angles must be numeric")
                continue

            # Use heuristic to detect if angles are in degrees or radians
            max_angle = max(abs(a) for a in angles if isinstance(a, (int, float)))
            if max_angle > 2 * np.pi:  # Likely degrees
                if angle < 0 or angle > 360:
                    result.warnings.append(
                        f"Phi angle {angle}° outside typical range [0°, 360°]"
                    )
            else:  # Likely radians
                if angle < 0 or angle > 2 * np.pi:
                    result.warnings.append(
                        f"Phi angle {angle} rad outside typical range [0, 2π] rad"
                    )

    def _validate_parameter_consistency(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate cross-parameter consistency.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        # Check VI-MCMC parameter consistency for hybrid method
        opt_config = config.get("optimization", {})
        vi_config = opt_config.get("vi", {})
        mcmc_config = opt_config.get("mcmc", {})

        vi_iters = vi_config.get("n_iterations", 2000)
        mcmc_samples = mcmc_config.get("n_samples", 1000)

        # Warn if VI iterations are much larger than MCMC samples
        if vi_iters > mcmc_samples * 5:
            result.suggestions.append(
                "Consider reducing VI iterations or increasing MCMC samples for balanced hybrid analysis"
            )

        # Check mode-specific requirements
        mode = config.get("analysis_mode")
        data_config = config.get("data", {})

        if mode == "static_isotropic" and data_config.get("custom_phi_angles"):
            angles_str = data_config["custom_phi_angles"]
            if isinstance(angles_str, str) and "," in angles_str:
                result.warnings.append(
                    "Multiple phi angles specified for isotropic analysis - only first will be used"
                )

    def _validate_static_parameters(
        self, params: Dict[str, float], result: ValidationResult
    ) -> None:
        """
        Validate static analysis parameters.

        Args:
            params: Parameter dictionary
            result: Validation result to update
        """
        required_params = ["D0", "alpha", "D_offset"]

        for param in required_params:
            if param not in params:
                result.errors.append(f"Missing required static parameter: {param}")
                continue

            value = params[param]
            constraint = self.physics_constraints.get(param, {})

            min_val = constraint.get("min")
            max_val = constraint.get("max")

            if min_val is not None and value < min_val:
                result.errors.append(f"{param} = {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                result.warnings.append(
                    f"{param} = {value} above typical maximum {max_val}"
                )

    def _validate_flow_parameters(
        self, params: Dict[str, float], result: ValidationResult
    ) -> None:
        """
        Validate laminar flow parameters.

        Args:
            params: Parameter dictionary
            result: Validation result to update
        """
        # Static parameters
        self._validate_static_parameters(params, result)

        # Flow-specific parameters
        flow_params = ["gamma_dot_0", "beta", "gamma_dot_offset", "phi_0"]

        for param in flow_params:
            if param not in params:
                result.errors.append(f"Missing required flow parameter: {param}")
                continue

            value = params[param]
            constraint = self.physics_constraints.get(param, {})

            min_val = constraint.get("min")
            max_val = constraint.get("max")

            if min_val is not None and value < min_val:
                result.errors.append(f"{param} = {value} below minimum {min_val}")
            if max_val is not None and value > max_val:
                result.warnings.append(
                    f"{param} = {value} above typical maximum {max_val}"
                )

    def _validate_enhanced_physics_constraints(
        self, params: Dict[str, float], mode: str, result: ValidationResult
    ) -> None:
        """
        Enhanced physics constraint validation with detailed parameter analysis.

        Args:
            params: Parameter dictionary
            mode: Analysis mode
            result: Validation result to update
        """
        constraints = self._get_physics_constraints()

        for param_name, value in params.items():
            if param_name not in constraints:
                result.warnings.append(
                    f"Unknown parameter '{param_name}' - skipping validation"
                )
                continue

            constraint = constraints[param_name]

            # Check hard bounds
            if value < constraint["min"]:
                result.errors.append(
                    f"{param_name} = {value} {constraint['units']} below physical minimum "
                    f"({constraint['min']} {constraint['units']})"
                )

            if value > constraint["max"]:
                result.errors.append(
                    f"{param_name} = {value} {constraint['units']} above physical maximum "
                    f"({constraint['max']} {constraint['units']})"
                )

            # Check typical ranges
            if "typical_min" in constraint and value < constraint["typical_min"]:
                result.warnings.append(
                    f"{param_name} = {value} {constraint['units']} below typical range "
                    f"({constraint['typical_min']} - {constraint['typical_max']} {constraint['units']}). "
                    f"This may indicate unusual physics or measurement conditions."
                )

            if "typical_max" in constraint and value > constraint["typical_max"]:
                result.warnings.append(
                    f"{param_name} = {value} {constraint['units']} above typical range "
                    f"({constraint['typical_min']} - {constraint['typical_max']} {constraint['units']}). "
                    f"Verify this is expected for your system."
                )

        # Mode-specific physics validation
        if mode == "laminar_flow":
            self._validate_flow_physics_consistency(params, result)

        # Cross-parameter physics validation
        self._validate_cross_parameter_physics(params, result)

    def _validate_flow_physics_consistency(
        self, params: Dict[str, float], result: ValidationResult
    ) -> None:
        """
        Validate physics consistency for laminar flow parameters.

        Args:
            params: Parameter dictionary
            result: Validation result to update
        """
        # Check shear rate vs diffusion consistency
        D0 = params.get("D0")
        gamma_dot_0 = params.get("gamma_dot_0")

        if D0 is not None and gamma_dot_0 is not None:
            # Peclet number estimation
            # Pe ~ gamma_dot * L^2 / D, typical L ~ 1 micron
            typical_length = 1e4  # Angstroms
            peclet_estimate = gamma_dot_0 * typical_length**2 / D0

            if peclet_estimate > 1000:
                result.warnings.append(
                    f"High estimated Peclet number ({peclet_estimate:.1f}). "
                    "Strong flow may dominate over diffusion. Verify this is expected."
                )
            elif peclet_estimate < 0.01:
                result.warnings.append(
                    f"Low estimated Peclet number ({peclet_estimate:.3f}). "
                    "Diffusion dominates over flow. Consider static analysis mode."
                )

        # Check power law exponent consistency
        alpha = params.get("alpha")
        beta = params.get("beta")

        if alpha is not None and beta is not None:
            if abs(alpha - beta) > 3.0:
                result.warnings.append(
                    f"Large difference between diffusion exponent (α={alpha:.2f}) "
                    f"and shear exponent (β={beta:.2f}). Verify independent evolution is expected."
                )

    def _validate_cross_parameter_physics(
        self, params: Dict[str, float], result: ValidationResult
    ) -> None:
        """
        Validate cross-parameter physics relationships.

        Args:
            params: Parameter dictionary
            result: Validation result to update
        """
        # Validate diffusion parameter relationships
        D0 = params.get("D0")
        D_offset = params.get("D_offset")
        alpha = params.get("alpha")

        if D0 is not None and D_offset is not None:
            if abs(D_offset) > D0 * 10:
                result.warnings.append(
                    f"Large diffusion offset (D_offset={D_offset:.2e}) compared to D0={D0:.2e}. "
                    "This may indicate model instability."
                )

        # Check for anomalous diffusion consistency
        if alpha is not None:
            if alpha < -2.5:
                result.warnings.append(
                    f"Very negative anomalous exponent (α={alpha:.2f}) indicates "
                    "super-diffusive behavior. Verify this is physically expected."
                )
            elif alpha > 2.5:
                result.warnings.append(
                    f"Very positive anomalous exponent (α={alpha:.2f}) indicates "
                    "extreme sub-diffusive behavior. Check for artifacts."
                )

        # Validate scaling parameter relationships
        contrast = params.get("contrast")
        offset = params.get("offset")

        if contrast is not None and offset is not None:
            if contrast + offset > 3.0:
                result.warnings.append(
                    f"High correlation maximum (contrast={contrast:.3f} + offset={offset:.3f} = {contrast + offset:.3f}). "
                    "Verify data normalization is correct."
                )

            if contrast / (contrast + offset) < 0.1:
                result.warnings.append(
                    f"Very low contrast ratio ({contrast / (contrast + offset):.3f}). "
                    "This may indicate weak correlation signal or normalization issues."
                )

    def _validate_multi_method_workflow(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate multi-method optimization workflows (VI → MCMC → Hybrid).

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        opt_config = config.get("optimization_config", {})

        # Check for hybrid method configuration
        if "hybrid" in opt_config:
            self._validate_hybrid_workflow(opt_config, result)

        # Validate method compatibility
        self._validate_method_compatibility(opt_config, result)

        # Check resource allocation consistency across methods
        self._validate_resource_allocation_consistency(opt_config, result)

    def _validate_hybrid_workflow(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate hybrid VI → MCMC workflow configuration.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        hybrid_config = opt_config.get("hybrid", {})
        vi_config = opt_config.get("variational", {})
        mcmc_config = opt_config.get("mcmc_sampling", {})

        if not vi_config and hybrid_config.get("use_vi_init", True):
            result.errors.append(
                "Hybrid method requires VI initialization but VI config is missing"
            )

        if not mcmc_config:
            result.errors.append("Hybrid method requires MCMC config but it is missing")

        # Check VI-MCMC parameter compatibility
        vi_iters = vi_config.get("n_iterations", 2000)
        mcmc_samples = mcmc_config.get("draws", 3000)

        if vi_iters > mcmc_samples * 2:
            result.warnings.append(
                f"VI iterations ({vi_iters}) much larger than MCMC samples ({mcmc_samples}). "
                "Consider reducing VI iterations for balanced workflow."
            )

        # Check convergence thresholds consistency
        vi_conv_tol = vi_config.get("convergence_tol", 1e-6)
        hybrid_conv_threshold = hybrid_config.get("convergence_threshold", 0.1)

        if hybrid_conv_threshold < vi_conv_tol * 100:
            result.warnings.append(
                f"Hybrid convergence threshold ({hybrid_conv_threshold}) may be too strict "
                f"compared to VI convergence tolerance ({vi_conv_tol})"
            )

    def _validate_method_compatibility(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate compatibility between different optimization methods.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        enabled_methods = set()

        if opt_config.get("classical_optimization", {}).get("enabled", True):
            enabled_methods.add("classical")

        if opt_config.get("robust_optimization", {}).get("enabled", True):
            enabled_methods.add("robust")

        if opt_config.get("mcmc_sampling", {}).get("enabled", True):
            enabled_methods.add("mcmc")

        if "hybrid" in opt_config:
            enabled_methods.add("hybrid")

        # Check for conflicting configurations
        if "hybrid" in enabled_methods and len(enabled_methods) > 2:
            result.warnings.append(
                "Hybrid method enabled with multiple other methods. "
                "Consider using hybrid OR other methods, not both."
            )

        # Validate method-specific requirements
        if "robust" in enabled_methods:
            self._validate_robust_method_requirements(opt_config, result)

        if "mcmc" in enabled_methods:
            self._validate_mcmc_method_requirements(opt_config, result)

    def _validate_resource_allocation_consistency(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate consistent resource allocation across methods.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        # Check CPU core allocation
        mcmc_cores = opt_config.get("mcmc_sampling", {}).get("cores", 4)
        batch_parallel = (
            opt_config.get("classical_optimization", {})
            .get("batch_processing", {})
            .get("max_parallel_runs", 4)
        )

        total_cores = psutil.cpu_count(logical=False)
        requested_cores = max(mcmc_cores, batch_parallel)

        if requested_cores > total_cores:
            result.warnings.append(
                f"Requested cores ({requested_cores}) exceed available physical cores ({total_cores}). "
                "Performance may be suboptimal."
            )

        # Check memory consistency
        self._validate_memory_allocation(opt_config, result)

    def _validate_memory_allocation(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate memory allocation across different methods.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        available_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Estimate memory requirements for different methods
        mcmc_chains = opt_config.get("mcmc_sampling", {}).get("chains", 4)
        mcmc_samples = opt_config.get("mcmc_sampling", {}).get("draws", 3000)
        estimated_mcmc_memory = (
            mcmc_chains * mcmc_samples * 8 / (1024**3)
        )  # rough estimate

        batch_runs = (
            opt_config.get("classical_optimization", {})
            .get("batch_processing", {})
            .get("max_parallel_runs", 4)
        )
        estimated_batch_memory = batch_runs * 0.5  # rough estimate per run

        total_estimated = estimated_mcmc_memory + estimated_batch_memory

        if total_estimated > available_memory_gb * 0.8:
            result.warnings.append(
                f"Estimated memory usage ({total_estimated:.1f} GB) may exceed available memory "
                f"({available_memory_gb:.1f} GB). Consider reducing batch size or MCMC parameters."
            )

    def _validate_robust_method_requirements(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate requirements for robust optimization methods.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        robust_config = opt_config.get("robust_optimization", {})

        # Check solver availability
        preferred_solver = robust_config.get("preferred_solver", "CLARABEL")
        if preferred_solver == "GUROBI" and not self._check_gurobi_availability():
            result.warnings.append(
                "Gurobi solver preferred but not available. Will fall back to open-source solvers."
            )

        # Validate uncertainty parameters
        uncertainty_radius = robust_config.get("uncertainty_radius", 0.03)
        if uncertainty_radius > 0.2:
            result.warnings.append(
                f"Large uncertainty radius ({uncertainty_radius}) may lead to overly conservative results"
            )

    def _validate_mcmc_method_requirements(
        self, opt_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate requirements for MCMC methods.

        Args:
            opt_config: Optimization configuration
            result: Validation result to update
        """
        mcmc_config = opt_config.get("mcmc_sampling", {})

        # Check backend configuration
        backend_specific = mcmc_config.get("backend_specific", {})
        if "gpu_backend" in backend_specific and not self._check_jax_availability():
            result.warnings.append(
                "GPU backend requested but JAX not available. Will fall back to CPU backend."
            )

        # Validate convergence diagnostics
        chains = mcmc_config.get("chains", 4)
        if chains < 2:
            result.errors.append(
                "MCMC requires at least 2 chains for convergence diagnostics"
            )

    def _validate_hpc_configuration(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate HPC-specific configuration settings.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        hpc_config = config.get("hpc_settings", {})

        if not hpc_config:
            result.info.append("No HPC configuration detected - running in local mode")
            return

        # Validate PBS Professional settings
        if "pbs" in hpc_config:
            self._validate_pbs_configuration(hpc_config["pbs"], result)

        # Validate SLURM settings
        if "slurm" in hpc_config:
            self._validate_slurm_configuration(hpc_config["slurm"], result)

        # Validate distributed computing settings
        if "distributed" in hpc_config:
            self._validate_distributed_configuration(hpc_config["distributed"], result)

    def _validate_pbs_configuration(
        self, pbs_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate PBS Professional job parameters.

        Args:
            pbs_config: PBS configuration dictionary
            result: Validation result to update
        """
        # Check resource requests
        nodes = pbs_config.get("nodes", 1)
        ppn = pbs_config.get("ppn", 1)  # processors per node
        mem = pbs_config.get("mem", "4gb")
        walltime = pbs_config.get("walltime", "24:00:00")

        # Validate resource consistency
        total_cores = nodes * ppn
        if total_cores > 128:  # typical HPC limit
            result.warnings.append(
                f"Requesting {total_cores} cores may exceed typical job limits"
            )

        # Parse memory request
        try:
            mem_value, mem_unit = self._parse_memory_spec(mem)
            if mem_unit.lower() == "gb" and mem_value < 2:
                result.warnings.append(
                    f"Low memory request ({mem}) may be insufficient for analysis"
                )
        except ValueError:
            result.errors.append(f"Invalid memory specification: {mem}")

        # Validate walltime format
        if not self._validate_time_format(walltime):
            result.errors.append(
                f"Invalid walltime format: {walltime}. Use HH:MM:SS or DD:HH:MM:SS"
            )

    def _validate_slurm_configuration(
        self, slurm_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate SLURM job parameters.

        Args:
            slurm_config: SLURM configuration dictionary
            result: Validation result to update
        """
        # Similar validation logic for SLURM
        ntasks = slurm_config.get("ntasks", 1)
        cpus_per_task = slurm_config.get("cpus-per-task", 1)
        mem_per_cpu = slurm_config.get("mem-per-cpu", "2G")
        time_limit = slurm_config.get("time", "24:00:00")

        total_cores = ntasks * cpus_per_task

        # Validate partition if specified
        partition = slurm_config.get("partition")
        if partition and partition not in ["cpu", "gpu", "debug", "normal", "long"]:
            result.warnings.append(
                f"Unknown SLURM partition '{partition}'. Verify it exists on your system."
            )

    def _validate_distributed_configuration(
        self, dist_config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate distributed computing configuration.

        Args:
            dist_config: Distributed configuration dictionary
            result: Validation result to update
        """
        if "mpi" in dist_config:
            mpi_config = dist_config["mpi"]
            processes = mpi_config.get("processes", 1)

            if processes > 1 and not self._check_mpi_availability():
                result.errors.append(
                    "MPI configuration specified but MPI not available on system"
                )

    def _validate_advanced_scenarios(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate advanced analysis scenarios.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        # Validate large dataset scenarios
        self._validate_large_dataset_configuration(config, result)

        # Validate batch processing scenarios
        self._validate_batch_processing_configuration(config, result)

        # Validate complex phi angle filtering
        self._validate_complex_phi_filtering(config, result)

    def _validate_large_dataset_configuration(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate configuration for large dataset processing.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        analyzer_params = config.get("analyzer_parameters", {})
        temporal = analyzer_params.get("temporal", {})

        start_frame = temporal.get("start_frame", 0)
        end_frame = temporal.get("end_frame", 1000)
        frame_count = end_frame - start_frame

        # Estimate memory requirements for large datasets
        if frame_count > 100000:  # Very large dataset
            result.info.append(f"Large dataset detected: {frame_count:,} frames")

            # Check memory settings
            perf_settings = config.get("performance_settings", {})
            memory_limit = perf_settings.get("memory_management", {}).get(
                "memory_limit_gb", 16
            )

            estimated_memory = frame_count * 8 * 1024 / (1024**3)  # rough estimate

            if estimated_memory > memory_limit:
                result.warnings.append(
                    f"Estimated memory usage ({estimated_memory:.1f} GB) may exceed limit ({memory_limit} GB). "
                    "Consider enabling low_memory_mode or reducing frame range."
                )

            # Suggest performance optimizations
            if not perf_settings.get("caching", {}).get("enable_disk_cache", True):
                result.suggestions.append(
                    "Enable disk caching for large datasets to improve performance"
                )

    def _validate_batch_processing_configuration(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate batch processing setup.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        batch_config = config.get("batch_processing", {})

        if not batch_config:
            return

        # Validate batch size and parallelization
        batch_size = batch_config.get("batch_size", 10)
        parallel_batches = batch_config.get("parallel_batches", 2)

        total_parallel = batch_size * parallel_batches
        available_cores = psutil.cpu_count(logical=True)

        if total_parallel > available_cores * 2:
            result.warnings.append(
                f"Batch parallelization ({total_parallel}) may exceed system capacity ({available_cores} cores)"
            )

        # Check output directory capacity
        output_dir = batch_config.get("output_directory", "./batch_results")
        if not self._check_directory_writable(output_dir):
            result.errors.append(f"Batch output directory not writable: {output_dir}")

    def _validate_complex_phi_filtering(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate complex phi angle filtering configurations.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        opt_config = config.get("optimization_config", {})
        angle_filtering = opt_config.get("angle_filtering", {})

        if not angle_filtering.get("enabled", False):
            return

        target_ranges = angle_filtering.get("target_ranges", [])

        if not target_ranges:
            result.warnings.append(
                "Angle filtering enabled but no target ranges specified"
            )
            return

        # Validate range specifications
        total_coverage = 0
        for i, range_spec in enumerate(target_ranges):
            if not isinstance(range_spec, dict):
                result.errors.append(f"Invalid angle range specification at index {i}")
                continue

            min_angle = range_spec.get("min_angle")
            max_angle = range_spec.get("max_angle")

            if min_angle is None or max_angle is None:
                result.errors.append(f"Missing min_angle or max_angle in range {i}")
                continue

            if min_angle >= max_angle:
                result.errors.append(
                    f"Invalid angle range {i}: min ({min_angle}) >= max ({max_angle})"
                )

            range_width = max_angle - min_angle
            total_coverage += range_width

        # Check coverage efficiency
        if total_coverage < 30:  # Less than 30 degrees total
            result.warnings.append(
                f"Limited angle coverage ({total_coverage:.1f}°) may reduce analysis quality"
            )

        if len(target_ranges) > 10:
            result.suggestions.append(
                f"Many angle ranges ({len(target_ranges)}) specified. Consider consolidating for efficiency."
            )

    def _detect_and_validate_hardware(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Detect available hardware and validate configuration compatibility.

        Args:
            config: Configuration dictionary
            result: Validation result to update
        """
        # Detect system hardware
        hardware_info = self._detect_system_hardware()
        result.hardware_info = hardware_info

        # Validate GPU configuration if specified
        hardware_config = config.get("hardware", {})
        if "gpu_memory_fraction" in hardware_config or "force_cpu" in hardware_config:
            self._validate_gpu_configuration(hardware_config, hardware_info, result)

        # Check JAX/CUDA compatibility
        self._validate_jax_cuda_compatibility(hardware_info, result)

    def _detect_system_hardware(self) -> Dict[str, Any]:
        """
        Detect available system hardware.

        Returns:
            Dictionary with hardware information
        """
        if self._hardware_cache is not None:
            return self._hardware_cache

        hardware_info = {
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "gpu_available": False,
            "gpu_info": [],
            "jax_available": HAS_JAX,
            "cuda_available": False,
        }

        # Detect GPU information
        gpu_info = self._detect_gpu_info()
        hardware_info.update(gpu_info)

        # Check CUDA availability
        hardware_info["cuda_available"] = self._check_cuda_availability()

        self._hardware_cache = hardware_info
        return hardware_info

    def _detect_gpu_info(self) -> Dict[str, Any]:
        """
        Detect GPU information using multiple methods.

        Returns:
            Dictionary with GPU information
        """
        if self._gpu_info_cache is not None:
            return self._gpu_info_cache

        gpu_info = {"gpu_available": False, "gpu_info": [], "total_gpu_memory_gb": 0}

        # Try GPUtil first
        if HAS_GPUTIL:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info["gpu_available"] = True
                    for gpu in gpus:
                        gpu_data = {
                            "id": gpu.id,
                            "name": gpu.name,
                            "memory_total_mb": gpu.memoryTotal,
                            "memory_used_mb": gpu.memoryUsed,
                            "memory_free_mb": gpu.memoryFree,
                            "utilization": gpu.load * 100,
                            "temperature": gpu.temperature,
                        }
                        gpu_info["gpu_info"].append(gpu_data)
                        gpu_info["total_gpu_memory_gb"] += gpu.memoryTotal / 1024
            except Exception as e:
                logger.debug(f"GPUtil detection failed: {e}")

        # Try pynvml as fallback
        if not gpu_info["gpu_available"] and HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()

                if device_count > 0:
                    gpu_info["gpu_available"] = True
                    for i in range(device_count):
                        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                        name = pynvml.nvmlDeviceGetName(handle).decode()
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_data = {
                            "id": i,
                            "name": name,
                            "memory_total_mb": mem_info.total // (1024 * 1024),
                            "memory_used_mb": mem_info.used // (1024 * 1024),
                            "memory_free_mb": mem_info.free // (1024 * 1024),
                        }
                        gpu_info["gpu_info"].append(gpu_data)
                        gpu_info["total_gpu_memory_gb"] += mem_info.total / (1024**3)

                pynvml.nvmlShutdown()
            except Exception as e:
                logger.debug(f"pynvml detection failed: {e}")

        # Try nvidia-smi as final fallback
        if not gpu_info["gpu_available"]:
            gpu_info.update(self._detect_gpu_via_nvidia_smi())

        self._gpu_info_cache = gpu_info
        return gpu_info

    def _detect_gpu_via_nvidia_smi(self) -> Dict[str, Any]:
        """
        Detect GPU using nvidia-smi command.

        Returns:
            Dictionary with GPU information
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                gpu_info = {
                    "gpu_available": True,
                    "gpu_info": [],
                    "total_gpu_memory_gb": 0,
                }

                for i, line in enumerate(result.stdout.strip().split("\n")):
                    if line:
                        parts = line.split(", ")
                        if len(parts) >= 4:
                            name, total_mb, used_mb, free_mb = parts[:4]
                            gpu_data = {
                                "id": i,
                                "name": name,
                                "memory_total_mb": int(total_mb),
                                "memory_used_mb": int(used_mb),
                                "memory_free_mb": int(free_mb),
                            }
                            gpu_info["gpu_info"].append(gpu_data)
                            gpu_info["total_gpu_memory_gb"] += int(total_mb) / 1024

                return gpu_info
        except Exception as e:
            logger.debug(f"nvidia-smi detection failed: {e}")

        return {"gpu_available": False, "gpu_info": [], "total_gpu_memory_gb": 0}

    def _validate_gpu_configuration(
        self,
        hardware_config: Dict[str, Any],
        hardware_info: Dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """
        Validate GPU configuration against detected hardware.

        Args:
            hardware_config: Hardware configuration
            hardware_info: Detected hardware information
            result: Validation result to update
        """
        gpu_memory_fraction = hardware_config.get("gpu_memory_fraction", 0.8)
        force_cpu = hardware_config.get("force_cpu", False)

        if force_cpu:
            result.info.append("GPU disabled by force_cpu setting")
            return

        if not hardware_info.get("gpu_available", False):
            result.warnings.append(
                "GPU configuration specified but no GPU detected. Will fall back to CPU."
            )
            return

        # Validate memory fraction against available GPU memory
        total_gpu_memory = hardware_info.get("total_gpu_memory_gb", 0)
        if total_gpu_memory > 0:
            requested_memory = total_gpu_memory * gpu_memory_fraction

            # Check current GPU usage
            gpu_info = hardware_info.get("gpu_info", [])
            if gpu_info:
                for gpu in gpu_info:
                    free_memory_gb = gpu.get("memory_free_mb", 0) / 1024
                    if requested_memory > free_memory_gb:
                        result.warnings.append(
                            f"Requested GPU memory ({requested_memory:.1f} GB) may exceed "
                            f"available memory on GPU {gpu['id']} ({free_memory_gb:.1f} GB free)"
                        )

            result.info.append(
                f"GPU memory allocation: {requested_memory:.1f} GB / {total_gpu_memory:.1f} GB "
                f"({gpu_memory_fraction:.1%})"
            )

    def _validate_jax_cuda_compatibility(
        self, hardware_info: Dict[str, Any], result: ValidationResult
    ) -> None:
        """
        Validate JAX and CUDA compatibility.

        Args:
            hardware_info: Hardware information
            result: Validation result to update
        """
        if not hardware_info.get("jax_available", False):
            result.info.append("JAX not available - using numpy backend")
            return

        if not hardware_info.get("gpu_available", False):
            result.info.append("JAX available but no GPU detected - using CPU backend")
            return

        # Check JAX GPU backend availability
        if HAS_JAX:
            try:
                devices = jax.devices("gpu")
                if devices:
                    result.info.append(
                        f"JAX GPU backend available: {len(devices)} device(s)"
                    )
                else:
                    result.warnings.append(
                        "JAX available but no GPU devices detected by JAX"
                    )
            except Exception as e:
                result.warnings.append(f"JAX GPU detection failed: {e}")

    # Helper methods for hardware detection
    def _check_gurobi_availability(self) -> bool:
        """Check if Gurobi solver is available."""
        try:
            import gurobipy

            return True
        except ImportError:
            return False

    def _check_jax_availability(self) -> bool:
        """Check if JAX is available."""
        return HAS_JAX

    def _check_cuda_availability(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_mpi_availability(self) -> bool:
        """Check if MPI is available."""
        return shutil.which("mpirun") is not None or shutil.which("mpiexec") is not None

    def _parse_memory_spec(self, mem_spec: str) -> Tuple[float, str]:
        """Parse memory specification like '4gb' or '8GB'."""
        import re

        match = re.match(r"(\d+(?:\.\d+)?)([a-zA-Z]+)", mem_spec.lower())
        if match:
            value, unit = match.groups()
            return float(value), unit
        raise ValueError(f"Invalid memory specification: {mem_spec}")

    def _validate_time_format(self, time_str: str) -> bool:
        """Validate HPC time format (HH:MM:SS or DD:HH:MM:SS)."""
        import re

        patterns = [
            r"^\d{1,2}:\d{2}:\d{2}$",  # HH:MM:SS
            r"^\d{1,3}:\d{2}:\d{2}:\d{2}$",  # DD:HH:MM:SS
        ]
        return any(re.match(pattern, time_str) for pattern in patterns)

    def _check_directory_writable(self, directory: str) -> bool:
        """Check if directory exists and is writable."""
        try:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            return os.access(path, os.W_OK)
        except Exception:
            return False

    def _get_physics_constraints(self) -> Dict[str, Dict[str, float]]:
        """
        Get enhanced physics-based parameter constraints with realistic bounds.

        Returns:
            Dictionary of parameter constraints with extended physics validation
        """
        return {
            # Diffusion parameters
            "D0": {
                "min": 1e-3,  # Consistent with physics.py DIFFUSION_MIN
                "max": 1e5,   # Consistent with physics.py DIFFUSION_MAX
                "typical_min": 10.0,
                "typical_max": 10000.0,
                "units": "Å²/s",
                "description": "Reference diffusion coefficient",
            },
            "alpha": {
                "min": -10.0,  # Consistent with physics.py ALPHA_MIN
                "max": 10.0,   # Consistent with physics.py ALPHA_MAX
                "typical_min": -2.0,
                "typical_max": 2.0,
                "units": "dimensionless",
                "description": "Anomalous diffusion exponent",
            },
            "D_offset": {
                "min": -1e5,  # Consistent with physics.py DIFFUSION_OFFSET_MIN
                "max": 1e5,   # Consistent with physics.py DIFFUSION_OFFSET_MAX
                "typical_min": -1000.0,
                "typical_max": 1000.0,
                "units": "Å²/s",
                "description": "Baseline diffusion coefficient",
            },
            # Flow parameters (laminar flow mode)
            "gamma_dot_0": {
                "min": 1e-6,  # Consistent with typical shear rate bounds
                "max": 1.0,   # Consistent with typical shear rate bounds
                "typical_min": 1e-3,
                "typical_max": 1e-1,
                "units": "s⁻¹",
                "description": "Reference shear rate amplitude",
            },
            "beta": {
                "min": -10.0,  # Consistent with physics.py BETA_MIN
                "max": 10.0,   # Consistent with physics.py BETA_MAX
                "typical_min": -2.0,
                "typical_max": 2.0,
                "units": "dimensionless",
                "description": "Shear rate power-law exponent",
            },
            "gamma_dot_offset": {
                "min": -1.0,  # Consistent with physics.py SHEAR_OFFSET_MIN
                "max": 1.0,   # Consistent with physics.py SHEAR_OFFSET_MAX
                "typical_min": -0.1,
                "typical_max": 0.1,
                "units": "s⁻¹",
                "description": "Baseline shear rate offset",
            },
            "phi_0": {
                "min": -30.0,  # Consistent with physics.py ANGLE_MIN
                "max": 30.0,   # Consistent with physics.py ANGLE_MAX
                "typical_min": -10.0,
                "typical_max": 10.0,
                "units": "degrees",
                "description": "Angular phase offset",
            },
            # Scaling parameters
            "contrast": {
                "min": 0.001,  # Very low but non-zero contrast
                "max": 2.0,  # Allow super-unity for certain systems
                "typical_min": 0.01,
                "typical_max": 1.0,
                "units": "dimensionless",
                "description": "Correlation function contrast",
            },
            "offset": {
                "min": 0.5,  # Minimum baseline
                "max": 3.0,  # Allow high baseline
                "typical_min": 0.9,
                "typical_max": 2.0,
                "units": "dimensionless",
                "description": "Correlation function offset",
            },
        }
