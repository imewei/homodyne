"""
Minimal Configuration Management for Homodyne v2
===============================================

Simplified configuration system with preserved API compatibility.
Provides essential YAML/JSON loading with the same interface as the original
ConfigManager while removing complex features not needed for core functionality.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

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
    """
    Minimal configuration manager for homodyne v2 scattering analysis.

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
        config_override: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize configuration manager.

        Parameters
        ----------
        config_file : str
            Path to YAML/JSON configuration file
        config_override : dict, optional
            Override configuration data instead of loading from file
        """
        self.config_file = config_file
        self.config: Optional[Dict[str, Any]] = None

        if config_override is not None:
            self.config = config_override.copy()
            logger.info("Configuration loaded from override data")
        else:
            self.load_config()

    def load_config(self) -> None:
        """
        Load and parse YAML/JSON configuration file.

        Supports both YAML and JSON formats with graceful fallback
        to default configuration if loading fails.
        """
        try:
            if self.config_file is None:
                raise ValueError("Configuration file path cannot be None")

            config_path = Path(self.config_file)
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_file}"
                )

            # Determine file format and load accordingly
            file_extension = config_path.suffix.lower()

            with open(config_path, "r", encoding="utf-8") as f:
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

        except (yaml.YAMLError if HAS_YAML else Exception) as e:
            logger.error(f"YAML parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            logger.info("Using default configuration...")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration structure.

        Returns minimal configuration that supports basic analysis modes.
        """
        return {
            "metadata": {
                "config_version": "2.1",
                "description": "Default minimal configuration"
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

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration dictionary.

        Returns
        -------
        Dict[str, Any]
            Current configuration dictionary
        """
        return self.config

    def update_config(self, key: str, value: Any) -> None:
        """
        Update a configuration value using dot notation.

        Parameters
        ----------
        key : str
            Configuration key (supports dot notation like 'optimization.method')
        value : Any
            New value to set
        """
        keys = key.split('.')
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

    def get_target_angle_ranges(self) -> Dict[str, Any]:
        """Get angle filtering ranges."""
        if not self.config:
            return {"enabled": False}

        optimization = self.config.get("optimization", {})
        angle_filtering = optimization.get("angle_filtering", {})
        return angle_filtering


def load_xpcs_config(config_path: str) -> Dict[str, Any]:
    """
    Load XPCS configuration from file.

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