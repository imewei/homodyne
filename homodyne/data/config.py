"""
Configuration System for XPCS Data Loading
==========================================

YAML-first configuration system with JSON support for XPCS data loading.
Provides configuration validation, schema definitions, and format conversion utilities.

This module supports:
- YAML configuration loading and validation
- JSON configuration support
- Configuration schema validation
- Migration utilities from JSON to YAML
- Integration with modern configuration management

Configuration Structure:
- experimental_data: File paths and data parameters
- analyzer_parameters: Analysis settings (time, frames)
- enhanced_features: Enhanced features and optimizations
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass

# Handle YAML dependency
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# V2 logging integration
try:
    from homodyne.utils.logging import get_logger
    HAS_V2_LOGGING = True
except ImportError:
    import logging
    HAS_V2_LOGGING = False
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    missing_optional: List[str]

class XPCSConfigurationError(Exception):
    """Raised when XPCS configuration is invalid."""
    pass

# Configuration schema definitions
XPCS_CONFIG_SCHEMA = {
    "experimental_data": {
        "required": {
            "data_folder_path": str,
            "data_file_name": str,
        },
        "optional": {
            "phi_angles_path": str,
            "cache_file_path": str,
            "cache_filename_template": str,
            "cache_compression": bool,
            "apply_diagonal_correction": bool,
        },
        "defaults": {
            "phi_angles_path": "./output/",
            "cache_file_path": None,  # Will use data_folder_path if None
            "cache_filename_template": "cached_c2_frames_{start_frame}_{end_frame}.npz",
            "cache_compression": True,
            "apply_diagonal_correction": True,
        }
    },
    "data_filtering": {
        "required": {},
        "optional": {
            "enabled": bool,
            "q_range": dict,
            "phi_range": dict,
            "quality_threshold": (int, float),
            "frame_filtering": dict,
            "combine_criteria": str,
            "fallback_on_empty": bool,
            "validation_level": str,
        },
        "defaults": {
            "enabled": False,
            "q_range": {},
            "phi_range": {},
            "quality_threshold": None,
            "frame_filtering": {},
            "combine_criteria": "AND",  # "AND", "OR"
            "fallback_on_empty": True,
            "validation_level": "basic",  # "basic", "strict"
        }
    },
    "analyzer_parameters": {
        "required": {
            "dt": (int, float),
            "start_frame": int,
            "end_frame": int,
        },
        "optional": {
            "frame_step": int,
            "time_unit": str,
        },
        "defaults": {
            "frame_step": 1,
            "time_unit": "seconds",
        }
    },
    "v2_features": {
        "required": {},
        "optional": {
            "output_format": str,
            "validation_level": str,
            "performance_optimization": bool,
            "physics_validation": bool,
            "cache_strategy": str,
            "parallel_processing": bool,
            "gpu_acceleration": bool,
        },
        "defaults": {
            "output_format": "auto",  # "numpy", "jax", "auto"
            "validation_level": "basic",  # "none", "basic", "full"
            "performance_optimization": True,
            "physics_validation": False,
            "cache_strategy": "intelligent",  # "none", "simple", "intelligent"
            "parallel_processing": False,
            "gpu_acceleration": False,
        }
    }
}

def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        XPCSConfigurationError: If YAML loading fails
    """
    if not HAS_YAML:
        raise XPCSConfigurationError("PyYAML required for YAML configuration files")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise XPCSConfigurationError(f"Empty or invalid YAML file: {config_path}")
        
        logger.debug(f"Loaded YAML configuration from: {config_path}")
        return config
        
    except yaml.YAMLError as e:
        raise XPCSConfigurationError(f"Failed to parse YAML configuration {config_path}: {e}")

def load_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load JSON configuration file with automatic YAML conversion.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        XPCSConfigurationError: If JSON loading fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.debug(f"Loaded JSON configuration from: {config_path}")
        logger.info("Consider migrating to YAML format for improved readability")
        return config
        
    except json.JSONDecodeError as e:
        raise XPCSConfigurationError(f"Failed to parse JSON configuration {config_path}: {e}")

def validate_config_schema(config: Dict[str, Any], 
                          schema: Dict[str, Any] = None) -> ConfigValidationResult:
    """
    Validate configuration against schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema to validate against (defaults to XPCS_CONFIG_SCHEMA)
        
    Returns:
        Validation result with errors and warnings
    """
    if schema is None:
        schema = XPCS_CONFIG_SCHEMA
    
    errors = []
    warnings = []
    missing_optional = []
    
    for section_name, section_schema in schema.items():
        if section_name not in config:
            if section_schema.get("required"):
                errors.append(f"Missing required configuration section: {section_name}")
            else:
                warnings.append(f"Missing optional configuration section: {section_name}")
            continue
        
        section_config = config[section_name]
        
        # Check required parameters
        for param_name, param_type in section_schema.get("required", {}).items():
            if param_name not in section_config:
                errors.append(f"Missing required parameter: {section_name}.{param_name}")
            else:
                value = section_config[param_name]
                if isinstance(param_type, tuple):
                    # Multiple allowed types
                    if not any(isinstance(value, t) for t in param_type):
                        errors.append(f"Parameter {section_name}.{param_name} has wrong type: "
                                    f"expected {param_type}, got {type(value)}")
                else:
                    if not isinstance(value, param_type):
                        errors.append(f"Parameter {section_name}.{param_name} has wrong type: "
                                    f"expected {param_type}, got {type(value)}")
        
        # Check optional parameters
        for param_name, param_type in section_schema.get("optional", {}).items():
            if param_name not in section_config:
                missing_optional.append(f"{section_name}.{param_name}")
            else:
                value = section_config[param_name]
                if isinstance(param_type, tuple):
                    if not any(isinstance(value, t) for t in param_type):
                        warnings.append(f"Parameter {section_name}.{param_name} has unexpected type: "
                                      f"expected {param_type}, got {type(value)}")
                else:
                    if not isinstance(value, param_type):
                        warnings.append(f"Parameter {section_name}.{param_name} has unexpected type: "
                                      f"expected {param_type}, got {type(value)}")
    
    # Validate specific parameter values
    errors.extend(_validate_parameter_values(config))
    warnings.extend(_validate_parameter_warnings(config))
    
    return ConfigValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        missing_optional=missing_optional
    )

def _validate_parameter_values(config: Dict[str, Any]) -> List[str]:
    """Validate specific parameter value constraints."""
    errors = []
    
    # Validate experimental_data parameters
    exp_data = config.get("experimental_data", {})
    
    # Check file paths
    data_folder = exp_data.get("data_folder_path", "")
    if data_folder and not os.path.exists(data_folder):
        errors.append(f"Data folder does not exist: {data_folder}")
    
    data_file = exp_data.get("data_file_name", "")
    if data_folder and data_file:
        full_path = os.path.join(data_folder, data_file)
        if not os.path.exists(full_path):
            errors.append(f"Data file does not exist: {full_path}")
    
    # Validate analyzer_parameters
    analyzer = config.get("analyzer_parameters", {})
    
    start_frame = analyzer.get("start_frame")
    end_frame = analyzer.get("end_frame")
    if start_frame is not None and end_frame is not None:
        if start_frame >= end_frame:
            errors.append(f"start_frame ({start_frame}) must be less than end_frame ({end_frame})")
        if start_frame < 1:
            errors.append(f"start_frame ({start_frame}) must be >= 1")
    
    dt = analyzer.get("dt")
    if dt is not None and dt <= 0:
        errors.append(f"dt ({dt}) must be positive")
    
    # Validate data_filtering parameters
    data_filtering = config.get("data_filtering", {})
    if data_filtering.get("enabled", False):
        # Validate q_range
        q_range = data_filtering.get("q_range", {})
        if q_range:
            q_min = q_range.get("min")
            q_max = q_range.get("max")
            if q_min is not None and q_min <= 0:
                errors.append(f"q_range.min ({q_min}) must be positive")
            if q_max is not None and q_max <= 0:
                errors.append(f"q_range.max ({q_max}) must be positive")
            if q_min is not None and q_max is not None and q_min >= q_max:
                errors.append(f"q_range.min ({q_min}) must be less than q_range.max ({q_max})")
        
        # Validate phi_range
        phi_range = data_filtering.get("phi_range", {})
        if phi_range:
            phi_min = phi_range.get("min")
            phi_max = phi_range.get("max")
            if phi_min is not None and phi_max is not None and phi_min >= phi_max:
                errors.append(f"phi_range.min ({phi_min}) must be less than phi_range.max ({phi_max})")
            if phi_min is not None and not (-360 <= phi_min <= 360):
                errors.append(f"phi_range.min ({phi_min}) should be in range [-360, 360]")
            if phi_max is not None and not (-360 <= phi_max <= 360):
                errors.append(f"phi_range.max ({phi_max}) should be in range [-360, 360]")
        
        # Validate quality_threshold
        quality_threshold = data_filtering.get("quality_threshold")
        if quality_threshold is not None and quality_threshold <= 0:
            errors.append(f"quality_threshold ({quality_threshold}) must be positive")
        
        # Validate combine_criteria
        combine_criteria = data_filtering.get("combine_criteria", "AND")
        if combine_criteria not in ["AND", "OR"]:
            errors.append(f"combine_criteria must be one of: AND, OR (got: {combine_criteria})")
        
        # Validate validation_level
        validation_level = data_filtering.get("validation_level", "basic")
        if validation_level not in ["basic", "strict"]:
            errors.append(f"data_filtering.validation_level must be one of: basic, strict (got: {validation_level})")
    
    # Validate v2_features parameters
    v2_features = config.get("v2_features", {})
    
    output_format = v2_features.get("output_format", "auto")
    if output_format not in ["numpy", "jax", "auto"]:
        errors.append(f"output_format must be one of: numpy, jax, auto (got: {output_format})")
    
    validation_level = v2_features.get("validation_level", "basic")
    if validation_level not in ["none", "basic", "full"]:
        errors.append(f"validation_level must be one of: none, basic, full (got: {validation_level})")
    
    cache_strategy = v2_features.get("cache_strategy", "intelligent")
    if cache_strategy not in ["none", "simple", "intelligent"]:
        errors.append(f"cache_strategy must be one of: none, simple, intelligent (got: {cache_strategy})")
    
    return errors

def _validate_parameter_warnings(config: Dict[str, Any]) -> List[str]:
    """Generate warnings for parameter values that may cause issues."""
    warnings = []
    
    analyzer = config.get("analyzer_parameters", {})
    
    # Warn about very large frame ranges
    start_frame = analyzer.get("start_frame", 1)
    end_frame = analyzer.get("end_frame", 1000)
    frame_count = end_frame - start_frame + 1
    
    if frame_count > 10000:
        warnings.append(f"Large frame range ({frame_count} frames) may result in long processing time")
    
    # Warn about very small dt values
    dt = analyzer.get("dt")
    if dt is not None and dt < 1e-6:
        warnings.append(f"Very small dt value ({dt}) - check time units")
    
    return warnings

def apply_config_defaults(config: Dict[str, Any], 
                         schema: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Apply default values to configuration.
    
    Args:
        config: Configuration dictionary
        schema: Schema with default values (defaults to XPCS_CONFIG_SCHEMA)
        
    Returns:
        Configuration with defaults applied
    """
    if schema is None:
        schema = XPCS_CONFIG_SCHEMA
    
    config_with_defaults = config.copy()
    
    for section_name, section_schema in schema.items():
        if section_name not in config_with_defaults:
            config_with_defaults[section_name] = {}
        
        section_config = config_with_defaults[section_name]
        defaults = section_schema.get("defaults", {})
        
        for param_name, default_value in defaults.items():
            if param_name not in section_config:
                # Special handling for cache_file_path default
                if param_name == "cache_file_path" and default_value is None:
                    data_folder = config_with_defaults.get("experimental_data", {}).get("data_folder_path")
                    if data_folder:
                        section_config[param_name] = data_folder
                else:
                    section_config[param_name] = default_value
                logger.debug(f"Applied default for {section_name}.{param_name}: {default_value}")
    
    return config_with_defaults

def migrate_json_to_yaml_config(json_config: Dict[str, Any],
                               yaml_output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Migrate JSON configuration to YAML format.
    
    Args:
        json_config: JSON configuration dictionary
        yaml_output_path: Optional path to save YAML configuration
        
    Returns:
        YAML configuration dictionary
    """
    # Structure is already suitable, so just copy
    # In the future, this could include more sophisticated transformations
    yaml_config = json_config.copy()
    
    # Add v2 features section if not present
    if "v2_features" not in yaml_config:
        yaml_config["v2_features"] = XPCS_CONFIG_SCHEMA["v2_features"]["defaults"].copy()
    
    # Apply defaults
    yaml_config = apply_config_defaults(yaml_config)
    
    if yaml_output_path:
        save_yaml_config(yaml_config, yaml_output_path)
    
    logger.info("Migrated JSON configuration to YAML format")
    return yaml_config

def save_yaml_config(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save YAML file
        
    Raises:
        XPCSConfigurationError: If YAML saving fails
    """
    if not HAS_YAML:
        raise XPCSConfigurationError("PyYAML required to save YAML configuration files")
    
    output_path = Path(output_path)
    
    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"Saved YAML configuration to: {output_path}")
        
    except Exception as e:
        raise XPCSConfigurationError(f"Failed to save YAML configuration to {output_path}: {e}")

def create_example_yaml_config(output_path: Union[str, Path],
                              data_folder: str = "/path/to/data",
                              data_file: str = "experiment.hdf") -> None:
    """
    Create an example YAML configuration file.
    
    Args:
        output_path: Path to save example configuration
        data_folder: Example data folder path
        data_file: Example data file name
    """
    example_config = {
        "experimental_data": {
            "data_folder_path": data_folder,
            "data_file_name": data_file,
            "phi_angles_path": "./output/",
            "cache_file_path": "./cache/",
            "cache_filename_template": "cached_c2_frames_{start_frame}_{end_frame}.npz",
            "cache_compression": True,
            "apply_diagonal_correction": True
        },
        "analyzer_parameters": {
            "dt": 0.001,
            "start_frame": 1,
            "end_frame": 1000,
            "time_unit": "seconds"
        },
        "v2_features": {
            "output_format": "auto",
            "validation_level": "basic",
            "performance_optimization": True,
            "physics_validation": False,
            "cache_strategy": "intelligent",
            "parallel_processing": False,
            "gpu_acceleration": False
        }
    }
    
    save_yaml_config(example_config, output_path)
    logger.info(f"Created example YAML configuration: {output_path}")

# Export main functions
__all__ = [
    "load_yaml_config",
    "load_json_config", 
    "validate_config_schema",
    "apply_config_defaults",
    "migrate_json_to_yaml_config",
    "save_yaml_config",
    "create_example_yaml_config",
    "ConfigValidationResult",
    "XPCSConfigurationError",
    "XPCS_CONFIG_SCHEMA"
]