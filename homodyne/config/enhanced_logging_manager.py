"""
Enhanced Logging Configuration Manager for Homodyne v2
=====================================================

Extended configuration management with comprehensive support for the enhanced
logging system including JAX-specific, scientific computing, distributed
computing, advanced debugging, and production monitoring configurations.

This module extends the base ConfigManager with enhanced logging capabilities
while supporting existing configurations.

Key Features:
- Enhanced logging configuration parsing and validation
- JAX-specific logging settings management
- Scientific computing logging contexts configuration
- Distributed computing logging coordination settings
- Advanced debugging configuration management
- Production monitoring and alerting configuration
- Environment variable override support
- Configuration migration and upgrade utilities

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's original design
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import base configuration manager
try:
    from .manager import ConfigManager as BaseConfigManager
    from .manager import logger as base_logger

    HAS_BASE_MANAGER = True
except ImportError:
    HAS_BASE_MANAGER = False
    BaseConfigManager = object
    base_logger = logging.getLogger(__name__)

# Import enhanced logging utilities
try:
    from homodyne.utils.advanced_debugging import clear_advanced_debugging_data
    from homodyne.utils.distributed_logging import \
        get_distributed_logger_manager
    from homodyne.utils.jax_logging import clear_jax_tracking_data
    from homodyne.utils.logging import configure_logging_from_json, get_logger
    from homodyne.utils.production_monitoring import \
        get_production_monitoring_stats
    from homodyne.utils.scientific_logging import \
        clear_scientific_tracking_data

    HAS_ENHANCED_LOGGING = True
except ImportError:
    HAS_ENHANCED_LOGGING = False


@dataclass
class EnhancedLoggingConfig:
    """Configuration structure for enhanced logging system."""

    # Core logging settings
    enabled: bool = True
    level: str = "INFO"

    # Console logging
    console_enabled: bool = True
    console_level: str = "INFO"
    console_format: str = "detailed"
    console_colors: bool = True

    # File logging
    file_enabled: bool = True
    file_level: str = "DEBUG"
    file_path: str = "~/.homodyne/logs/"
    file_name: str = "homodyne.log"
    file_max_size_mb: int = 50
    file_backup_count: int = 10

    # Performance logging
    performance_enabled: bool = True
    performance_level: str = "INFO"
    performance_filename: str = "performance.log"
    performance_threshold_seconds: float = 0.1

    # Module-specific levels
    module_levels: Dict[str, str] = field(default_factory=dict)

    # JAX-specific settings
    jax_enabled: bool = True
    jax_compilation_logging: bool = True
    jax_memory_tracking: bool = True
    jax_gradient_logging: bool = True
    jax_device_logging: bool = True

    # Scientific computing settings
    scientific_enabled: bool = True
    scientific_data_validation: bool = True
    scientific_physics_validation: bool = True
    scientific_correlation_monitoring: bool = True
    scientific_fitting_tracking: bool = True
    scientific_numerical_stability: bool = True

    # Distributed computing settings
    distributed_enabled: bool = False
    distributed_resource_monitoring: bool = True
    distributed_mpi_integration: bool = True
    distributed_hierarchical_logging: bool = True

    # Advanced debugging settings
    debugging_enabled: bool = True
    debugging_error_recovery: bool = True
    debugging_numerical_stability: bool = True
    debugging_performance_anomalies: bool = True
    debugging_memory_leak_detection: bool = False

    # Production monitoring settings
    production_enabled: bool = False
    production_health_checks: bool = True
    production_alerting: bool = True
    production_metrics_collection: bool = True
    production_performance_baselines: bool = True


class EnhancedLoggingConfigManager:
    """Enhanced configuration manager with comprehensive logging support."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        base_manager: Optional[BaseConfigManager] = None,
    ):
        """
        Initialize enhanced logging configuration manager.

        Args:
            config_path: Path to configuration file
            base_manager: Optional base ConfigManager instance
        """
        self.logger = get_logger(__name__) if HAS_ENHANCED_LOGGING else base_logger

        # Initialize base configuration manager
        if base_manager:
            self.base_manager = base_manager
        elif HAS_BASE_MANAGER and config_path:
            self.base_manager = BaseConfigManager(config_path)
        else:
            self.base_manager = None
            self.logger.warning("Base configuration manager not available")

        # Enhanced logging configuration
        self.enhanced_config = EnhancedLoggingConfig()

        # Load enhanced configuration if available
        if self.base_manager and hasattr(self.base_manager, "config"):
            self._load_enhanced_logging_config()

    def _load_enhanced_logging_config(self):
        """Load enhanced logging configuration from base config."""
        if not self.base_manager or not hasattr(self.base_manager, "config"):
            return

        config = self.base_manager.config
        if not isinstance(config, dict):
            return

        logging_config = config.get("logging", {})

        # Core logging settings
        self.enhanced_config.enabled = logging_config.get("enabled", True)
        self.enhanced_config.level = logging_config.get("level", "INFO")

        # Console settings
        console_config = logging_config.get("console", {})
        self.enhanced_config.console_enabled = console_config.get("enabled", True)
        self.enhanced_config.console_level = console_config.get("level", "INFO")
        self.enhanced_config.console_format = console_config.get("format", "detailed")
        self.enhanced_config.console_colors = console_config.get("colors", True)

        # File settings
        file_config = logging_config.get("file", {})
        self.enhanced_config.file_enabled = file_config.get("enabled", True)
        self.enhanced_config.file_level = file_config.get("level", "DEBUG")
        self.enhanced_config.file_path = file_config.get("path", "~/.homodyne/logs/")
        self.enhanced_config.file_name = file_config.get("filename", "homodyne.log")
        self.enhanced_config.file_max_size_mb = file_config.get("max_size_mb", 50)
        self.enhanced_config.file_backup_count = file_config.get("backup_count", 10)

        # Performance settings
        perf_config = logging_config.get("performance", {})
        self.enhanced_config.performance_enabled = perf_config.get("enabled", True)
        self.enhanced_config.performance_level = perf_config.get("level", "INFO")
        self.enhanced_config.performance_filename = perf_config.get(
            "filename", "performance.log"
        )
        self.enhanced_config.performance_threshold_seconds = perf_config.get(
            "threshold_seconds", 0.1
        )

        # Module-specific levels
        self.enhanced_config.module_levels = logging_config.get("modules", {})

        # JAX-specific settings
        jax_config = logging_config.get("jax", {})
        self.enhanced_config.jax_enabled = jax_config.get("enabled", True)

        jax_compilation = jax_config.get("compilation", {})
        self.enhanced_config.jax_compilation_logging = jax_compilation.get(
            "enabled", True
        )

        jax_memory = jax_config.get("memory", {})
        self.enhanced_config.jax_memory_tracking = jax_memory.get("enabled", True)

        jax_gradients = jax_config.get("gradients", {})
        self.enhanced_config.jax_gradient_logging = jax_gradients.get("enabled", True)

        jax_devices = jax_config.get("devices", {})
        self.enhanced_config.jax_device_logging = jax_devices.get("enabled", True)

        # Scientific computing settings
        sci_config = logging_config.get("scientific", {})
        self.enhanced_config.scientific_enabled = sci_config.get("enabled", True)

        sci_data = sci_config.get("data_loading", {})
        self.enhanced_config.scientific_data_validation = sci_data.get(
            "validate_data_quality", True
        )

        sci_physics = sci_config.get("physics_validation", {})
        self.enhanced_config.scientific_physics_validation = sci_physics.get(
            "enabled", True
        )

        sci_correlation = sci_config.get("correlation_computation", {})
        self.enhanced_config.scientific_correlation_monitoring = sci_correlation.get(
            "enabled", True
        )

        sci_fitting = sci_config.get("model_fitting", {})
        self.enhanced_config.scientific_fitting_tracking = sci_fitting.get(
            "track_fitting_progress", True
        )

        sci_numerical = sci_config.get("numerical_stability", {})
        self.enhanced_config.scientific_numerical_stability = sci_numerical.get(
            "enabled", True
        )

        # Distributed computing settings
        dist_config = logging_config.get("distributed", {})
        self.enhanced_config.distributed_enabled = dist_config.get("enabled", False)

        dist_resource = dist_config.get("resource_monitoring", {})
        self.enhanced_config.distributed_resource_monitoring = dist_resource.get(
            "enabled", True
        )

        dist_mpi = dist_config.get("mpi_integration", {})
        self.enhanced_config.distributed_mpi_integration = dist_mpi.get("enabled", True)

        dist_hierarchical = dist_config.get("hierarchical_logging", {})
        self.enhanced_config.distributed_hierarchical_logging = dist_hierarchical.get(
            "enabled", True
        )

        # Advanced debugging settings
        debug_config = logging_config.get("advanced_debugging", {})
        self.enhanced_config.debugging_enabled = debug_config.get("enabled", True)

        debug_recovery = debug_config.get("error_recovery", {})
        self.enhanced_config.debugging_error_recovery = debug_recovery.get(
            "enabled", True
        )

        debug_stability = debug_config.get("numerical_stability", {})
        self.enhanced_config.debugging_numerical_stability = debug_stability.get(
            "enabled", True
        )

        debug_anomalies = debug_config.get("performance_anomalies", {})
        self.enhanced_config.debugging_performance_anomalies = debug_anomalies.get(
            "enabled", True
        )

        debug_memory = debug_config.get("memory_leak_detection", {})
        self.enhanced_config.debugging_memory_leak_detection = debug_memory.get(
            "enabled", False
        )

        # Production monitoring settings
        prod_config = logging_config.get("production", {})
        self.enhanced_config.production_enabled = prod_config.get("enabled", False)

        prod_health = prod_config.get("health_checks", {})
        self.enhanced_config.production_health_checks = prod_health.get("enabled", True)

        prod_alerting = prod_config.get("alerting", {})
        self.enhanced_config.production_alerting = prod_alerting.get("enabled", True)

        prod_metrics = prod_config.get("metrics_collection", {})
        self.enhanced_config.production_metrics_collection = prod_metrics.get(
            "enabled", True
        )

        prod_baselines = prod_config.get("performance_baselines", {})
        self.enhanced_config.production_performance_baselines = prod_baselines.get(
            "enabled", True
        )

        self.logger.info("Enhanced logging configuration loaded successfully")

    def apply_environment_overrides(self):
        """Apply environment variable overrides to logging configuration."""
        env_mappings = {
            "HOMODYNE_LOG_LEVEL": "level",
            "HOMODYNE_LOG_CONSOLE_ENABLED": "console_enabled",
            "HOMODYNE_LOG_FILE_ENABLED": "file_enabled",
            "HOMODYNE_LOG_JAX_ENABLED": "jax_enabled",
            "HOMODYNE_LOG_SCIENTIFIC_ENABLED": "scientific_enabled",
            "HOMODYNE_LOG_DISTRIBUTED_ENABLED": "distributed_enabled",
            "HOMODYNE_LOG_PRODUCTION_ENABLED": "production_enabled",
            "HOMODYNE_LOG_DEBUG_ENABLED": "debugging_enabled",
        }

        for env_var, attr_name in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert string values to appropriate types
                if attr_name.endswith("_enabled") or attr_name in [
                    "console_enabled",
                    "file_enabled",
                ]:
                    converted_value = value.lower() in ("true", "1", "yes", "on")
                else:
                    converted_value = value

                setattr(self.enhanced_config, attr_name, converted_value)
                self.logger.debug(
                    f"Environment override: {attr_name} = {converted_value}"
                )

    def configure_enhanced_logging(self):
        """Configure the enhanced logging system based on loaded configuration."""
        if not HAS_ENHANCED_LOGGING:
            self.logger.warning("Enhanced logging utilities not available")
            return

        # Apply environment overrides first
        self.apply_environment_overrides()

        # Configure base logging if available
        if self.base_manager and hasattr(self.base_manager, "config"):
            try:
                configure_logging_from_json(self.base_manager.config)
                self.logger.info("Base logging system configured")
            except Exception as e:
                self.logger.error(f"Failed to configure base logging: {e}")

        # Configure distributed logging if enabled
        if self.enhanced_config.distributed_enabled:
            try:
                distributed_manager = get_distributed_logger_manager()
                self.logger.info("Distributed logging system configured")
            except Exception as e:
                self.logger.error(f"Failed to configure distributed logging: {e}")

        # Log configuration summary
        self._log_configuration_summary()

    def _log_configuration_summary(self):
        """Log a summary of the active configuration."""
        summary = [
            f"Enhanced logging system active:",
            f"  Core logging: {'enabled' if self.enhanced_config.enabled else 'disabled'}",
            f"  Console: {'enabled' if self.enhanced_config.console_enabled else 'disabled'} ({self.enhanced_config.console_level})",
            f"  File: {'enabled' if self.enhanced_config.file_enabled else 'disabled'} ({self.enhanced_config.file_level})",
            f"  JAX logging: {'enabled' if self.enhanced_config.jax_enabled else 'disabled'}",
            f"  Scientific logging: {'enabled' if self.enhanced_config.scientific_enabled else 'disabled'}",
            f"  Distributed logging: {'enabled' if self.enhanced_config.distributed_enabled else 'disabled'}",
            f"  Advanced debugging: {'enabled' if self.enhanced_config.debugging_enabled else 'disabled'}",
            f"  Production monitoring: {'enabled' if self.enhanced_config.production_enabled else 'disabled'}",
        ]

        for line in summary:
            self.logger.info(line)

    def validate_enhanced_configuration(self) -> List[str]:
        """Validate enhanced logging configuration and return any issues."""
        issues = []

        # Check log levels
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        if self.enhanced_config.level.upper() not in valid_levels:
            issues.append(f"Invalid log level: {self.enhanced_config.level}")

        if self.enhanced_config.console_level.upper() not in valid_levels:
            issues.append(
                f"Invalid console log level: {self.enhanced_config.console_level}"
            )

        if self.enhanced_config.file_level.upper() not in valid_levels:
            issues.append(f"Invalid file log level: {self.enhanced_config.file_level}")

        # Check file settings
        if self.enhanced_config.file_enabled:
            if self.enhanced_config.file_max_size_mb <= 0:
                issues.append("File max size must be positive")

            if self.enhanced_config.file_backup_count < 0:
                issues.append("File backup count cannot be negative")

            # Check if log directory is writable
            try:
                log_dir = Path(self.enhanced_config.file_path).expanduser()
                if not log_dir.exists():
                    log_dir.mkdir(parents=True, exist_ok=True)

                test_file = log_dir / "config_validation_test.tmp"
                test_file.write_text("test")
                test_file.unlink()

            except Exception as e:
                issues.append(f"Log directory not writable: {e}")

        # Check performance threshold
        if self.enhanced_config.performance_threshold_seconds < 0:
            issues.append("Performance threshold cannot be negative")

        # Validate module log levels
        for module, level in self.enhanced_config.module_levels.items():
            if level.upper() not in valid_levels:
                issues.append(f"Invalid log level for module {module}: {level}")

        return issues

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective enhanced logging configuration as dictionary."""
        return {
            "core": {
                "enabled": self.enhanced_config.enabled,
                "level": self.enhanced_config.level,
            },
            "console": {
                "enabled": self.enhanced_config.console_enabled,
                "level": self.enhanced_config.console_level,
                "format": self.enhanced_config.console_format,
                "colors": self.enhanced_config.console_colors,
            },
            "file": {
                "enabled": self.enhanced_config.file_enabled,
                "level": self.enhanced_config.file_level,
                "path": self.enhanced_config.file_path,
                "filename": self.enhanced_config.file_name,
                "max_size_mb": self.enhanced_config.file_max_size_mb,
                "backup_count": self.enhanced_config.file_backup_count,
            },
            "performance": {
                "enabled": self.enhanced_config.performance_enabled,
                "level": self.enhanced_config.performance_level,
                "filename": self.enhanced_config.performance_filename,
                "threshold_seconds": self.enhanced_config.performance_threshold_seconds,
            },
            "modules": self.enhanced_config.module_levels,
            "jax": {
                "enabled": self.enhanced_config.jax_enabled,
                "compilation_logging": self.enhanced_config.jax_compilation_logging,
                "memory_tracking": self.enhanced_config.jax_memory_tracking,
                "gradient_logging": self.enhanced_config.jax_gradient_logging,
                "device_logging": self.enhanced_config.jax_device_logging,
            },
            "scientific": {
                "enabled": self.enhanced_config.scientific_enabled,
                "data_validation": self.enhanced_config.scientific_data_validation,
                "physics_validation": self.enhanced_config.scientific_physics_validation,
                "correlation_monitoring": self.enhanced_config.scientific_correlation_monitoring,
                "fitting_tracking": self.enhanced_config.scientific_fitting_tracking,
                "numerical_stability": self.enhanced_config.scientific_numerical_stability,
            },
            "distributed": {
                "enabled": self.enhanced_config.distributed_enabled,
                "resource_monitoring": self.enhanced_config.distributed_resource_monitoring,
                "mpi_integration": self.enhanced_config.distributed_mpi_integration,
                "hierarchical_logging": self.enhanced_config.distributed_hierarchical_logging,
            },
            "debugging": {
                "enabled": self.enhanced_config.debugging_enabled,
                "error_recovery": self.enhanced_config.debugging_error_recovery,
                "numerical_stability": self.enhanced_config.debugging_numerical_stability,
                "performance_anomalies": self.enhanced_config.debugging_performance_anomalies,
                "memory_leak_detection": self.enhanced_config.debugging_memory_leak_detection,
            },
            "production": {
                "enabled": self.enhanced_config.production_enabled,
                "health_checks": self.enhanced_config.production_health_checks,
                "alerting": self.enhanced_config.production_alerting,
                "metrics_collection": self.enhanced_config.production_metrics_collection,
                "performance_baselines": self.enhanced_config.production_performance_baselines,
            },
        }

    def clear_logging_data(self):
        """Clear all logging data from enhanced utilities."""
        if not HAS_ENHANCED_LOGGING:
            return

        try:
            clear_jax_tracking_data()
            clear_scientific_tracking_data()
            clear_advanced_debugging_data()
            self.logger.info("Enhanced logging data cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear logging data: {e}")

    def get_logging_statistics(self) -> Dict[str, Any]:
        """Get comprehensive logging statistics."""
        stats = {"configuration_summary": self.get_effective_config()}

        if HAS_ENHANCED_LOGGING:
            try:
                stats["production_monitoring"] = get_production_monitoring_stats()
            except Exception as e:
                stats["production_monitoring_error"] = str(e)

        return stats

    def export_configuration(self, filepath: Optional[Path] = None) -> str:
        """Export the current enhanced logging configuration."""
        import json
        from datetime import datetime

        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = Path(f"homodyne_enhanced_logging_config_{timestamp}.json")

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "configuration": self.get_effective_config(),
            "validation_issues": self.validate_enhanced_configuration(),
            "statistics": self.get_logging_statistics(),
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info(f"Enhanced logging configuration exported to: {filepath}")
        return str(filepath)


def create_enhanced_config_manager(
    config_path: Union[str, Path],
) -> EnhancedLoggingConfigManager:
    """
    Create and configure an enhanced logging configuration manager.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configured EnhancedLoggingConfigManager instance
    """
    logger = (
        get_logger(__name__) if HAS_ENHANCED_LOGGING else logging.getLogger(__name__)
    )

    try:
        # Create base manager if available
        base_manager = None
        if HAS_BASE_MANAGER:
            base_manager = BaseConfigManager(config_path)

        # Create enhanced manager
        enhanced_manager = EnhancedLoggingConfigManager(
            config_path=config_path, base_manager=base_manager
        )

        # Configure logging system
        enhanced_manager.configure_enhanced_logging()

        # Validate configuration
        issues = enhanced_manager.validate_enhanced_configuration()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
        else:
            logger.info("Enhanced logging configuration validation passed")

        return enhanced_manager

    except Exception as e:
        logger.error(f"Failed to create enhanced config manager: {e}")
        raise


def get_default_enhanced_config() -> EnhancedLoggingConfig:
    """Get default enhanced logging configuration."""
    return EnhancedLoggingConfig()
