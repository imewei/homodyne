"""
Mode-Aware Configuration Loading for Homodyne v2
===============================================

Handles mode-specific parameter loading and configuration management.
Provides utilities for detecting analysis modes and loading appropriate
parameter structures.

Supported Modes:
- Static Isotropic: 3 physics params, single angle (no filtering)
- Static Anisotropic: 3 physics params, multiple angles with filtering
- Laminar Flow: 7 physics params, multiple angles with filtering

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from .manager import ConfigManager
    from .parameters import (AnalysisMode, CombinedParameters,
                             PhysicsParameters, ScalingParameters,
                             create_parameters_from_config,
                             validate_parameter_configuration)
except ImportError:
    # Handle relative imports during development
    from manager import ConfigManager
    from parameters import (AnalysisMode, CombinedParameters,
                            PhysicsParameters, ScalingParameters,
                            create_parameters_from_config,
                            validate_parameter_configuration)

# V2 integration
try:
    from homodyne.utils.logging import get_logger

    HAS_V2_LOGGING = True
except ImportError:
    import logging

    HAS_V2_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class ModeDetectionError(Exception):
    """Raised when analysis mode cannot be determined."""

    pass


class ParameterCountError(Exception):
    """Raised when parameter count doesn't match expected mode."""

    pass


def detect_analysis_mode(config: ConfigManager) -> AnalysisMode:
    """
    Detect analysis mode from configuration.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager instance

    Returns
    -------
    AnalysisMode
        Detected analysis mode

    Raises
    ------
    ModeDetectionError
        If mode cannot be determined from configuration
    """
    try:
        # Use ConfigManager's built-in mode detection
        mode_str = config.get_analysis_mode()

        mode_mapping = {
            "static_isotropic": AnalysisMode.STATIC_ISOTROPIC,
            "static_anisotropic": AnalysisMode.STATIC_ANISOTROPIC,
            "laminar_flow": AnalysisMode.LAMINAR_FLOW,
        }

        if mode_str in mode_mapping:
            return mode_mapping[mode_str]
        else:
            raise ModeDetectionError(f"Unknown analysis mode: {mode_str}")

    except Exception as e:
        logger.error(f"Failed to detect analysis mode: {e}")
        raise ModeDetectionError(
            f"Could not determine analysis mode from configuration: {e}"
        )


def validate_parameter_count_for_mode(parameter_count: int, mode: AnalysisMode) -> None:
    """
    Validate that parameter count matches expected count for mode.

    Parameters
    ----------
    parameter_count : int
        Number of parameters in configuration
    mode : AnalysisMode
        Analysis mode

    Raises
    ------
    ParameterCountError
        If parameter count doesn't match mode expectations
    """
    expected_counts = {
        AnalysisMode.STATIC_ISOTROPIC: 3,
        AnalysisMode.STATIC_ANISOTROPIC: 3,
        AnalysisMode.LAMINAR_FLOW: 7,
    }

    expected = expected_counts[mode]
    if parameter_count != expected:
        raise ParameterCountError(
            f"Parameter count mismatch for {mode.value}: expected {expected}, got {parameter_count}"
        )


def load_mode_specific_parameters(
    config: ConfigManager, mode: Optional[AnalysisMode] = None
) -> CombinedParameters:
    """
    Load parameters from configuration based on analysis mode.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager instance
    mode : AnalysisMode, optional
        Override analysis mode. If None, mode is detected from config.

    Returns
    -------
    CombinedParameters
        Combined parameter structure with physics and scaling parameters

    Raises
    ------
    ModeDetectionError
        If analysis mode cannot be determined
    ParameterCountError
        If parameter count doesn't match mode expectations
    """
    # Detect mode if not provided
    if mode is None:
        mode = detect_analysis_mode(config)

    logger.info(f"Loading parameters for {mode.value} mode")

    # Get parameter configuration
    initial_params = config.get("initial_parameters", default={})
    values = initial_params.get("values", [])
    parameter_names = initial_params.get("parameter_names", [])

    # Validate parameter count
    if values:
        validate_parameter_count_for_mode(len(values), mode)

    # Determine number of angles for scaling parameters
    n_angles = _determine_angle_count(config, mode)

    # Create combined parameters
    config_dict = config.config or {}
    combined_params = create_parameters_from_config(config_dict, mode, n_angles)

    logger.info(
        f"Loaded {combined_params.physics.get_parameter_count()} physics parameters"
    )
    logger.info(
        f"Scaling parameters: {combined_params.scaling.get_total_scaling_parameters()} ({n_angles} angles)"
    )

    return combined_params


def _determine_angle_count(config: ConfigManager, mode: AnalysisMode) -> int:
    """
    Determine number of angles for scaling parameters.

    For static isotropic mode, always returns 1 (single angle).
    For other modes, attempts to determine from data or defaults to reasonable value.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager instance
    mode : AnalysisMode
        Analysis mode

    Returns
    -------
    int
        Number of angles for scaling parameters
    """
    if mode == AnalysisMode.STATIC_ISOTROPIC:
        return 1  # Single angle, no filtering

    # For anisotropic/laminar flow modes, try to get from config or use default
    # This would typically be determined from the actual data file during loading
    # For configuration purposes, we use a reasonable default

    # Check if angle filtering is configured
    if config.is_angle_filtering_enabled():
        angle_ranges = config.get_target_angle_ranges()
        if angle_ranges:
            # Estimate based on angle ranges - this is approximate
            total_range = sum(max_a - min_a for min_a, max_a in angle_ranges)
            # Assume ~5 degree resolution
            estimated_angles = max(1, int(total_range / 5))
            logger.debug(f"Estimated {estimated_angles} angles from filtering config")
            return min(estimated_angles, 72)  # Cap at reasonable maximum

    # Default angle counts by mode
    defaults = {
        AnalysisMode.STATIC_ISOTROPIC: 1,
        AnalysisMode.STATIC_ANISOTROPIC: 36,  # Reasonable default
        AnalysisMode.LAMINAR_FLOW: 36,
    }

    return defaults.get(mode, 36)


def get_mode_specific_defaults(mode: AnalysisMode) -> Dict[str, Any]:
    """
    Get mode-specific default configuration.

    Parameters
    ----------
    mode : AnalysisMode
        Analysis mode

    Returns
    -------
    dict
        Mode-specific default configuration
    """
    base_defaults = {
        "analysis_settings": {
            "model_description": {
                "static_case": (
                    "g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt) = g₁_diff(t₁,t₂), g₂(t₁,t₂) = [g₁(t₁,t₂)]²"
                ),
                "laminar_flow_case": (
                    "g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂) where "
                    "g₁_shear = [sinc(Φ)]² and Φ = (1/2π)qL cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'"
                ),
            }
        }
    }

    if mode == AnalysisMode.STATIC_ISOTROPIC:
        return {
            **base_defaults,
            "analysis_settings": {
                **base_defaults["analysis_settings"],
                "static_mode": True,
                "static_submode": "isotropic",
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "optimization_config": {
                "angle_filtering": {
                    "enabled": False,  # Disabled for isotropic
                    "fallback_to_all_angles": True,
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1e6, "type": "Normal"},
                    {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {"name": "D_offset", "min": -100.0, "max": 100.0, "type": "Normal"},
                ]
            },
        }

    elif mode == AnalysisMode.STATIC_ANISOTROPIC:
        return {
            **base_defaults,
            "analysis_settings": {
                **base_defaults["analysis_settings"],
                "static_mode": True,
                "static_submode": "anisotropic",
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0],
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -10.0, "max_angle": 10.0},
                        {"min_angle": 170.0, "max_angle": 190.0},
                    ],
                    "fallback_to_all_angles": True,
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1e6, "type": "Normal"},
                    {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {"name": "D_offset", "min": -100.0, "max": 100.0, "type": "Normal"},
                ]
            },
        }

    elif mode == AnalysisMode.LAMINAR_FLOW:
        return {
            **base_defaults,
            "analysis_settings": {
                **base_defaults["analysis_settings"],
                "static_mode": False,
            },
            "initial_parameters": {
                "values": [100.0, 0.0, 10.0, 0.01, 0.0, 0.0, 0.0],
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
            },
            "optimization_config": {
                "angle_filtering": {
                    "enabled": True,
                    "target_ranges": [
                        {"min_angle": -10.0, "max_angle": 10.0},
                        {"min_angle": 170.0, "max_angle": 190.0},
                    ],
                    "fallback_to_all_angles": True,
                }
            },
            "parameter_space": {
                "bounds": [
                    {"name": "D0", "min": 1.0, "max": 1e6, "type": "Normal"},
                    {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {"name": "D_offset", "min": -100.0, "max": 100.0, "type": "Normal"},
                    {"name": "gamma_dot_t0", "min": 1e-6, "max": 1.0, "type": "Normal"},
                    {"name": "beta", "min": -2.0, "max": 2.0, "type": "Normal"},
                    {
                        "name": "gamma_dot_t_offset",
                        "min": -1e-2,
                        "max": 1e-2,
                        "type": "Normal",
                    },
                    {"name": "phi0", "min": -10.0, "max": 10.0, "type": "Normal"},
                ]
            },
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_mode_aware_config(
    mode: AnalysisMode, base_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create mode-aware configuration by merging mode defaults with base configuration.

    Parameters
    ----------
    mode : AnalysisMode
        Analysis mode
    base_config : dict, optional
        Base configuration to merge with mode defaults

    Returns
    -------
    dict
        Complete mode-aware configuration
    """
    mode_defaults = get_mode_specific_defaults(mode)

    if base_config is None:
        return mode_defaults

    # Deep merge base_config with mode_defaults
    def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    return deep_merge(mode_defaults, base_config)


def validate_mode_consistency(config: ConfigManager) -> Dict[str, Any]:
    """
    Validate that configuration is consistent with detected mode.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager instance

    Returns
    -------
    dict
        Validation result with 'valid', 'errors', 'warnings', 'mode_info'
    """
    errors = []
    warnings = []

    try:
        # Detect mode
        mode = detect_analysis_mode(config)
        mode_info = {
            "detected_mode": mode.value,
            "static_mode": config.is_static_mode_enabled(),
            "static_submode": config.get_static_submode(),
            "angle_filtering_enabled": config.is_angle_filtering_enabled(),
        }

        # Validate parameter count
        param_count = config.get_effective_parameter_count()
        expected_count = (
            3
            if mode in [AnalysisMode.STATIC_ISOTROPIC, AnalysisMode.STATIC_ANISOTROPIC]
            else 7
        )

        if param_count != expected_count:
            errors.append(
                f"Parameter count mismatch: {param_count} != {expected_count} for {mode.value}"
            )

        # Validate angle filtering settings
        if (
            mode == AnalysisMode.STATIC_ISOTROPIC
            and config.is_angle_filtering_enabled()
        ):
            warnings.append(
                "Angle filtering enabled for static isotropic mode (will be ignored)"
            )
        elif (
            mode in [AnalysisMode.STATIC_ANISOTROPIC, AnalysisMode.LAMINAR_FLOW]
            and not config.is_angle_filtering_enabled()
        ):
            warnings.append(
                f"Angle filtering disabled for {mode.value} mode (may use all angles)"
            )

        # Validate parameter configuration
        config_dict = config.config or {}
        param_validation = validate_parameter_configuration(config_dict)
        if not param_validation["valid"]:
            errors.extend(param_validation["errors"])
        warnings.extend(param_validation["warnings"])

        mode_info.update(param_validation["parameter_info"])

    except Exception as e:
        errors.append(f"Mode validation failed: {str(e)}")
        mode_info = {"error": str(e)}

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "mode_info": mode_info,
    }


def get_mode_summary(config: ConfigManager) -> Dict[str, Any]:
    """
    Get comprehensive summary of mode configuration.

    Parameters
    ----------
    config : ConfigManager
        Configuration manager instance

    Returns
    -------
    dict
        Mode configuration summary
    """
    try:
        mode = detect_analysis_mode(config)
        combined_params = load_mode_specific_parameters(config, mode)

        return {
            "mode": mode.value,
            "parameter_structure": combined_params.get_parameter_structure_summary(),
            "angle_filtering": {
                "enabled": config.is_angle_filtering_enabled(),
                "target_ranges": (
                    config.get_target_angle_ranges()
                    if config.is_angle_filtering_enabled()
                    else []
                ),
                "fallback_to_all": config.should_fallback_to_all_angles(),
            },
            "configuration": {
                "static_mode": config.is_static_mode_enabled(),
                "static_submode": config.get_static_submode(),
                "effective_param_count": config.get_effective_parameter_count(),
                "active_parameters": config.get_active_parameters(),
            },
        }
    except Exception as e:
        return {
            "error": f"Failed to generate mode summary: {str(e)}",
            "mode": "unknown",
        }
