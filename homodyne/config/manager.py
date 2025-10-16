"""Minimal Configuration Management for Homodyne v2
===============================================

Simplified configuration system with preserved API compatibility.
Provides essential YAML/JSON loading with the same interface as the original
ConfigManager while removing complex features not needed for core functionality.
"""

import json
from pathlib import Path
from typing import Any

# Handle YAML dependency
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Import minimal logging
try:
    from homodyne.utils.logging import get_logger

    HAS_LOGGING = True
except ImportError:
    import logging

    HAS_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class ConfigManager:
    """Minimal configuration manager for homodyne v2 scattering analysis.

    Provides simplified configuration loading with preserved API compatibility.

    Key Features:
    - YAML/JSON configuration file loading
    - Compatible .config attribute access
    - Preserved constructor signature
    - Graceful fallback to defaults

    Usage:
        config_manager = ConfigManager('my_config.yaml')
        data = config_manager.config
    """

    def __init__(
        self,
        config_file: str = "homodyne_config.yaml",
        config_override: dict[str, Any] | None = None,
    ):
        """Initialize configuration manager.

        Parameters
        ----------
        config_file : str
            Path to YAML/JSON configuration file
        config_override : dict, optional
            Override configuration data instead of loading from file
        """
        self.config_file = config_file
        self.config: dict[str, Any] | None = None

        # Cache for ParameterManager to avoid repeated instantiation
        self._cached_param_manager: Any | None = None

        if config_override is not None:
            self.config = config_override.copy()
            logger.info("Configuration loaded from override data")
        else:
            self.load_config()

        # Normalize schema for backward compatibility
        self._normalize_schema()

    def load_config(self) -> None:
        """Load and parse YAML/JSON configuration file.

        Supports both YAML and JSON formats with graceful fallback
        to default configuration if loading fails.
        """
        try:
            if self.config_file is None:
                raise ValueError("Configuration file path cannot be None")

            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}",
                )

            # Determine file format and load accordingly
            file_extension = config_path.suffix.lower()

            # Use 8KB buffering for improved I/O performance on large config files
            with open(config_path, buffering=8192, encoding="utf-8") as f:
                if file_extension in [".yaml", ".yml"] and HAS_YAML:
                    self.config = yaml.safe_load(f)
                elif file_extension == ".json":
                    self.config = json.load(f)
                elif HAS_YAML:
                    # Try YAML first for unknown extensions
                    content = f.read()
                    try:
                        self.config = yaml.safe_load(content)
                    except yaml.YAMLError:
                        # Fallback to JSON
                        self.config = json.loads(content)
                else:
                    # Only JSON available
                    self.config = json.load(f)

            logger.info(f"Configuration loaded from: {self.config_file}")

            # Display version information if available
            if isinstance(self.config, dict) and "metadata" in self.config:
                version = self.config["metadata"].get("config_version", "Unknown")
                logger.info(f"Configuration version: {version}")

            # Optional validation (can be disabled via environment variable)
            import os

            if os.environ.get("HOMODYNE_VALIDATE_CONFIG", "true").lower() == "true":
                self._validate_config()

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
        except Exception as e:
            # Handle YAML errors and other exceptions
            error_type = (
                "YAML parsing"
                if HAS_YAML and "yaml" in str(type(e)).lower()
                else "Configuration parsing"
            )
            logger.error(f"{error_type} error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()

    def _get_default_config(self) -> dict[str, Any]:
        """Get default configuration structure.

        Returns minimal configuration that supports basic analysis modes.
        """
        return {
            "metadata": {
                "config_version": "2.1",
                "description": "Default minimal configuration",
            },
            "analysis_mode": "static_isotropic",
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": -1,
            },
            "experimental_data": {
                "file_path": None,
                "cache_directory": "./cache",
                "use_caching": True,
            },
            "optimization": {
                "method": "nlsq",
                "lsq": {
                    "max_iterations": 10000,
                    "tolerance": 1e-8,
                    "method": "trf",
                },
                "mcmc": {
                    "n_samples": 1000,
                    "n_warmup": 1000,
                    "n_chains": 4,
                    "target_accept_prob": 0.8,
                },
            },
            "hardware": {
                "force_cpu": False,
                "gpu_memory_fraction": 0.8,
            },
            "output": {
                "formats": ["yaml", "npz"],
                "include_diagnostics": True,
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "console": {"enabled": True},
                "file": {"enabled": False},
            },
        }

    def get_config(self) -> dict[str, Any]:
        """Get the current configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Current configuration dictionary
        """
        return self.config

    def update_config(self, key: str, value: Any) -> None:
        """Update a configuration value using dot notation.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation like 'optimization.method')
        value : Any
            New value to set
        """
        keys = key.split(".")
        config_ref = self.config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]

        # Set the value
        config_ref[keys[-1]] = value

    def is_static_mode_enabled(self) -> bool:
        """Check if static analysis mode is enabled."""
        if not self.config:
            return True
        analysis_mode = self.config.get("analysis_mode", "static_isotropic")
        return "static" in analysis_mode.lower()

    def get_target_angle_ranges(self) -> dict[str, Any]:
        """Get angle filtering ranges."""
        if not self.config:
            return {"enabled": False}

        optimization = self.config.get("optimization", {})
        angle_filtering = optimization.get("angle_filtering", {})
        return angle_filtering

    def _get_parameter_manager(self):
        """Get or create cached ParameterManager.

        This avoids creating a new ParameterManager on every config access,
        providing ~14x speedup for repeated parameter queries.

        Returns
        -------
        ParameterManager
            Cached ParameterManager instance
        """
        if self._cached_param_manager is None:
            from homodyne.config.parameter_manager import ParameterManager

            # Determine analysis mode
            analysis_mode = "laminar_flow"
            if self.is_static_mode_enabled():
                analysis_mode = "static"

            # Create and cache ParameterManager
            self._cached_param_manager = ParameterManager(self.config, analysis_mode)
            logger.debug(f"Created cached ParameterManager for mode: {analysis_mode}")

        return self._cached_param_manager

    def get_parameter_bounds(
        self,
        parameter_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get parameter bounds from configuration (cached).

        Uses cached ParameterManager internally for improved performance.

        Parameters
        ----------
        parameter_names : list of str, optional
            List of parameter names to get bounds for. If None, returns bounds
            for all parameters in the current analysis mode.

        Returns
        -------
        list of dict
            List of bound dictionaries with keys: 'name', 'min', 'max', 'type'

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> bounds = config_mgr.get_parameter_bounds(["D0", "alpha"])
        >>> bounds[0]
        {'min': 1.0, 'max': 1000000.0, 'name': 'D0', 'type': 'Normal'}

        Notes
        -----
        This method uses a cached ParameterManager for ~14x speedup on repeated calls.
        """
        return self._get_parameter_manager().get_parameter_bounds(parameter_names)

    def get_active_parameters(self) -> list[str]:
        """Get list of active (physical) parameters from configuration (cached).

        Uses cached ParameterManager internally for improved performance.

        Returns
        -------
        list of str
            List of parameter names to be optimized. Falls back to mode-appropriate
            parameters if not specified in config.

        Examples
        --------
        >>> config_mgr = ConfigManager("config.yaml")
        >>> config_mgr.get_active_parameters()
        ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']

        Notes
        -----
        This method uses a cached ParameterManager for ~14x speedup on repeated calls.
        """
        return self._get_parameter_manager().get_active_parameters()

    def _validate_config(self) -> None:
        """Lightweight configuration validation.

        Checks for required sections and valid values.
        Can be disabled by setting HOMODYNE_VALIDATE_CONFIG=false environment variable.
        """
        if not self.config:
            logger.warning("Configuration is empty")
            return

        # Check for required sections
        required_sections = ["analysis_mode"]
        for section in required_sections:
            if section not in self.config:
                logger.warning(f"Missing recommended section: {section}")

        # Validate analysis_mode value
        valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        mode = self.config.get("analysis_mode", "")
        if mode and mode not in valid_modes:
            logger.warning(
                f"Unknown analysis_mode: '{mode}'. Valid modes: {valid_modes}",
            )

        logger.debug("Configuration validation completed")

    def _normalize_schema(self) -> None:
        """Normalize configuration schema for backward compatibility.

        Handles multiple configuration format versions by converting
        legacy formats to modern standardized formats transparently.
        """
        if not self.config:
            return

        self._normalize_experimental_data()
        # Future: add other normalizations here

    def _normalize_experimental_data(self) -> None:
        """Normalize experimental_data section.

        Supports two formats:
        1. Template/Legacy: data_folder_path + data_file_name
        2. Modern: file_path

        The normalization adds the missing format while preserving
        the original fields for backward compatibility.
        """
        if "experimental_data" not in self.config:
            return

        from pathlib import Path

        exp_data = self.config["experimental_data"]

        # Handle legacy composite format (data_folder_path + data_file_name)
        if "data_folder_path" in exp_data and "data_file_name" in exp_data:
            folder_path = exp_data["data_folder_path"]
            filename = exp_data["data_file_name"]

            # Skip normalization if either value is None
            if folder_path is None or filename is None:
                logger.debug(
                    "Skipping normalization: data_folder_path or data_file_name is None",
                )
                return

            folder = Path(folder_path)

            # Resolve relative paths for consistency
            # Note: Keep as-is if already absolute to preserve user intent
            file_path = folder / filename

            # Add modern format while preserving legacy fields
            exp_data["file_path"] = str(file_path)
            logger.info(
                f"Normalized legacy config format:\n"
                f"   {folder} + {filename}\n"
                f"   → file_path: {file_path}",
            )

        # Handle phi angles similarly
        if "phi_angles_path" in exp_data and "phi_angles_file" in exp_data:
            phi_folder = Path(exp_data["phi_angles_path"])
            phi_file = exp_data["phi_angles_file"]
            phi_path = phi_folder / phi_file

            # Add combined path for convenience
            exp_data["phi_angles_full_path"] = str(phi_path)
            logger.debug(f"Normalized phi angles path: {phi_path}")


def load_xpcs_config(config_path: str) -> dict[str, Any]:
    """Load XPCS configuration from file.

    Convenience function for loading configuration files.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    manager = ConfigManager(config_path)
    return manager.config
