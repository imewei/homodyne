"""
Configuration Health Monitor and Validation System for Homodyne v2
================================================================

Comprehensive health monitoring and validation system for homodyne configurations.
Provides deep validation, performance health checks, system diagnostics, and
continuous monitoring capabilities.

Key Features:
- Multi-level configuration validation (syntax, semantics, physics)
- Performance health monitoring and bottleneck detection
- System resource checks and recommendations
- Configuration quality scoring and optimization suggestions
- Automated health reports and trend analysis
- Real-time configuration monitoring during analysis
- Predictive issue detection and early warnings

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from .enhanced_console_output import (EnhancedConsoleLogger,
                                          ValidationFeedbackSystem)
    from .exceptions import (AnalysisModeError, ConfigurationFileError,
                             ErrorContext, FixSuggestion,
                             HomodyneConfigurationError,
                             ParameterValidationError)
    from .smart_recovery import SmartRecoveryEngine

    HAS_CONFIG_SYSTEM = True
except ImportError:
    HAS_CONFIG_SYSTEM = False
    HomodyneConfigurationError = Exception

try:
    from homodyne.utils.logging import get_logger

    HAS_UTILS_LOGGING = True
except ImportError:
    import logging

    HAS_UTILS_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""

    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"  # 70-89%
    WARNING = "warning"  # 50-69%
    CRITICAL = "critical"  # 0-49%


class ValidationLevel(Enum):
    """Validation level enumeration."""

    BASIC = "basic"  # Essential checks only
    STANDARD = "standard"  # Standard validation
    COMPREHENSIVE = "comprehensive"  # Thorough validation
    STRICT = "strict"  # Strictest validation


@dataclass
class HealthMetric:
    """Individual health metric."""

    name: str
    value: float  # 0-100 score
    status: HealthStatus
    message: str
    suggestions: List[str] = field(default_factory=list)
    impact: str = "low"  # low, medium, high, critical
    category: str = "general"  # configuration, performance, system, physics


@dataclass
class HealthReport:
    """Comprehensive health report."""

    overall_score: float
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_file: Optional[str] = None
    system_info: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def get_metrics_by_category(self, category: str) -> List[HealthMetric]:
        """Get metrics filtered by category."""
        return [m for m in self.metrics if m.category == category]

    def get_critical_issues(self) -> List[HealthMetric]:
        """Get critical issues that need immediate attention."""
        return [
            m
            for m in self.metrics
            if m.status == HealthStatus.CRITICAL or m.impact == "critical"
        ]

    def get_warnings(self) -> List[HealthMetric]:
        """Get warning-level issues."""
        return [m for m in self.metrics if m.status == HealthStatus.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert health report to dictionary."""
        return {
            "overall_score": self.overall_score,
            "overall_status": self.overall_status.value,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "status": m.status.value,
                    "message": m.message,
                    "suggestions": m.suggestions,
                    "impact": m.impact,
                    "category": m.category,
                }
                for m in self.metrics
            ],
            "timestamp": self.timestamp,
            "config_file": self.config_file,
            "system_info": self.system_info,
            "recommendations": self.recommendations,
        }


class ConfigurationHealthMonitor:
    """Comprehensive configuration health monitoring system."""

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        logger: Optional[EnhancedConsoleLogger] = None,
    ):
        self.validation_level = validation_level
        self.logger = logger
        self.feedback_system = (
            ValidationFeedbackSystem(logger) if HAS_CONFIG_SYSTEM else None
        )

        # Health metrics cache
        self.metrics_history: List[HealthReport] = []
        self.max_history = 100

        # System information
        self.system_info = self._collect_system_info()

        # Validation rules
        self.validation_rules = self._load_validation_rules()

        # Performance baselines
        self.performance_baselines = self._load_performance_baselines()

        logger.info("Configuration health monitor initialized")

    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect system information for health assessment."""
        try:
            info = {
                "platform": platform.system(),
                "platform_version": platform.release(),
                "python_version": sys.version.split()[0],
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": (
                    psutil.disk_usage("/").percent
                    if os.name != "nt"
                    else psutil.disk_usage("C:").percent
                ),
                "timestamp": datetime.now().isoformat(),
            }

            # Check for GPU availability
            try:
                import jax

                devices = jax.devices()
                info["jax_devices"] = [str(d) for d in devices]
                info["gpu_available"] = any("gpu" in str(d).lower() for d in devices)
                info["tpu_available"] = any("tpu" in str(d).lower() for d in devices)
            except ImportError:
                info["jax_devices"] = []
                info["gpu_available"] = False
                info["tpu_available"] = False

            # Check for CUDA
            try:
                import subprocess

                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                info["cuda_available"] = result.returncode == 0
            except (FileNotFoundError, subprocess.SubprocessError):
                info["cuda_available"] = False

            return info

        except Exception as e:
            logger.warning(f"Could not collect complete system info: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules based on validation level."""
        base_rules = {
            "parameter_ranges": {
                "wavevector_q": (1e-6, 10.0),
                "dt": (1e-6, 3600.0),
                "start_frame": (1, 1000000),
                "end_frame": (1, 1000000),
                "stator_rotor_gap": (1.0, 1e9),
                "D0": (1e-3, 1e8),
                "alpha": (-5.0, 5.0),
                "D_offset": (-1e6, 1e6),
                "gamma_dot_t0": (1e-9, 1e3),
                "beta": (-5.0, 5.0),
                "gamma_dot_t_offset": (-1e3, 1e3),
                "phi0": (-180.0, 180.0),
            },
            "frame_limits": {
                "minimum_frames": 10,
                "recommended_minimum": 100,
                "warning_above": 10000,
                "maximum_reasonable": 50000,
            },
            "physics_constraints": {
                "q_typical_min": 0.001,
                "q_typical_max": 0.1,
                "dt_typical_min": 0.001,
                "dt_typical_max": 10.0,
                "D0_colloidal_range": (1.0, 10000.0),
                "alpha_reasonable": (-2.0, 2.0),
            },
        }

        if self.validation_level == ValidationLevel.STRICT:
            # Tighter constraints for strict validation
            base_rules["parameter_ranges"]["alpha"] = (-2.5, 2.5)
            base_rules["parameter_ranges"]["beta"] = (-2.5, 2.5)
            base_rules["frame_limits"]["minimum_frames"] = 50
            base_rules["frame_limits"]["recommended_minimum"] = 500
        elif self.validation_level == ValidationLevel.BASIC:
            # Relaxed constraints for basic validation
            base_rules["frame_limits"]["minimum_frames"] = 5
            base_rules["frame_limits"]["recommended_minimum"] = 50

        return base_rules

    def _load_performance_baselines(self) -> Dict[str, Any]:
        """Load performance baselines for health assessment."""
        return {
            "memory_usage": {
                "low": 2.0,  # GB
                "medium": 8.0,  # GB
                "high": 16.0,  # GB
                "critical": 32.0,  # GB
            },
            "frame_processing_time": {
                "fast": 0.001,  # seconds per frame
                "medium": 0.01,  # seconds per frame
                "slow": 0.1,  # seconds per frame
                "critical": 1.0,  # seconds per frame
            },
            "optimization_iterations": {
                "fast": 100,
                "medium": 500,
                "slow": 2000,
                "critical": 10000,
            },
        }

    def perform_health_check(self, config_path: Union[str, Path]) -> HealthReport:
        """Perform comprehensive health check on configuration."""
        config_path = Path(config_path)

        if self.logger:
            self.logger.header(f"Configuration Health Check")
            self.logger.file_path(config_path)

        if self.feedback_system:
            self.feedback_system.start_validation(str(config_path))

        metrics = []

        try:
            # Load and parse configuration
            config_data = self._load_configuration(config_path)

            # 1. Syntax and structure validation
            syntax_metrics = self._validate_syntax_and_structure(
                config_data, config_path
            )
            metrics.extend(syntax_metrics)

            # 2. Parameter validation
            param_metrics = self._validate_parameters(config_data)
            metrics.extend(param_metrics)

            # 3. Physics validation
            physics_metrics = self._validate_physics_constraints(config_data)
            metrics.extend(physics_metrics)

            # 4. Performance assessment
            perf_metrics = self._assess_performance_configuration(config_data)
            metrics.extend(perf_metrics)

            # 5. System compatibility check
            system_metrics = self._check_system_compatibility(config_data)
            metrics.extend(system_metrics)

            # 6. Security and best practices
            if self.validation_level in [
                ValidationLevel.COMPREHENSIVE,
                ValidationLevel.STRICT,
            ]:
                security_metrics = self._check_security_and_best_practices(
                    config_data, config_path
                )
                metrics.extend(security_metrics)

            # Calculate overall score
            overall_score = self._calculate_overall_score(metrics)
            overall_status = self._score_to_status(overall_score)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, config_data)

            # Create health report
            report = HealthReport(
                overall_score=overall_score,
                overall_status=overall_status,
                metrics=metrics,
                config_file=str(config_path),
                system_info=self.system_info,
                recommendations=recommendations,
            )

            # Add to history
            self.metrics_history.append(report)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)

            # Show feedback
            if self.feedback_system:
                self._show_health_feedback(report)

            return report

        except Exception as e:
            logger.error(f"Health check failed: {e}")

            # Create error report
            error_metric = HealthMetric(
                name="health_check_error",
                value=0.0,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                suggestions=[
                    "Fix configuration file errors",
                    "Check file permissions",
                    "Validate YAML/JSON syntax",
                ],
                impact="critical",
                category="system",
            )

            return HealthReport(
                overall_score=0.0,
                overall_status=HealthStatus.CRITICAL,
                metrics=[error_metric],
                config_file=str(config_path),
                system_info=self.system_info,
                recommendations=["Fix critical errors before proceeding"],
            )

    def _load_configuration(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration file with error handling."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            raise (
                ConfigurationFileError(str(config_path), e) if HAS_CONFIG_SYSTEM else e
            )

    def _validate_syntax_and_structure(
        self, config_data: Dict[str, Any], config_path: Path
    ) -> List[HealthMetric]:
        """Validate configuration syntax and structure."""
        metrics = []

        if self.feedback_system:
            self.feedback_system.validate_section(
                "syntax_structure", True, "Configuration loaded successfully"
            )

        # Check required sections
        required_sections = [
            "analyzer_parameters",
            "experimental_data",
            "analysis_settings",
            "initial_parameters",
        ]
        missing_sections = [s for s in required_sections if s not in config_data]

        if missing_sections:
            metrics.append(
                HealthMetric(
                    name="missing_required_sections",
                    value=max(0, 100 - len(missing_sections) * 25),
                    status=(
                        HealthStatus.CRITICAL
                        if len(missing_sections) > 2
                        else HealthStatus.WARNING
                    ),
                    message=f"Missing required sections: {', '.join(missing_sections)}",
                    suggestions=[f"Add missing section: {s}" for s in missing_sections],
                    impact="high",
                    category="configuration",
                )
            )
        else:
            metrics.append(
                HealthMetric(
                    name="required_sections_present",
                    value=100.0,
                    status=HealthStatus.EXCELLENT,
                    message="All required sections are present",
                    category="configuration",
                )
            )

        # Check configuration structure depth and complexity
        max_depth = self._calculate_config_depth(config_data)
        total_keys = self._count_config_keys(config_data)

        complexity_score = min(100, max(0, 120 - max_depth * 10 - total_keys * 0.5))

        if complexity_score > 80:
            complexity_status = HealthStatus.EXCELLENT
            complexity_msg = "Configuration has good structure and complexity"
        elif complexity_score > 60:
            complexity_status = HealthStatus.GOOD
            complexity_msg = "Configuration is moderately complex"
        elif complexity_score > 40:
            complexity_status = HealthStatus.WARNING
            complexity_msg = "Configuration is quite complex - consider simplification"
        else:
            complexity_status = HealthStatus.CRITICAL
            complexity_msg = (
                "Configuration is overly complex and may be hard to maintain"
            )

        metrics.append(
            HealthMetric(
                name="configuration_complexity",
                value=complexity_score,
                status=complexity_status,
                message=f"{complexity_msg} (depth: {max_depth}, keys: {total_keys})",
                suggestions=(
                    [
                        "Consider breaking down complex sections",
                        "Use templates for common patterns",
                    ]
                    if complexity_score < 60
                    else []
                ),
                impact="medium",
                category="configuration",
            )
        )

        return metrics

    def _validate_parameters(self, config_data: Dict[str, Any]) -> List[HealthMetric]:
        """Validate parameter values and ranges."""
        metrics = []

        # Get analyzer parameters
        analyzer_params = config_data.get("analyzer_parameters", {})

        # Validate temporal parameters
        temporal = analyzer_params.get("temporal", {})
        dt = temporal.get("dt", 0.1)
        start_frame = temporal.get("start_frame", 1)
        end_frame = temporal.get("end_frame", 100)

        param_scores = []

        # Check dt
        dt_range = self.validation_rules["parameter_ranges"]["dt"]
        if dt_range[0] <= dt <= dt_range[1]:
            dt_score = 100.0
            dt_status = HealthStatus.EXCELLENT
            dt_msg = f"Time step dt = {dt}s is valid"
            dt_suggestions = []
        else:
            dt_score = 20.0
            dt_status = HealthStatus.CRITICAL
            dt_msg = f"Time step dt = {dt}s is outside valid range {dt_range}"
            dt_suggestions = [f"Set dt between {dt_range[0]} and {dt_range[1]} seconds"]

        param_scores.append(dt_score)
        metrics.append(
            HealthMetric(
                name="time_step_validation",
                value=dt_score,
                status=dt_status,
                message=dt_msg,
                suggestions=dt_suggestions,
                impact="high" if dt_score < 50 else "low",
                category="physics",
            )
        )

        # Check frame range
        frame_rules = self.validation_rules["frame_limits"]
        frame_count = end_frame - start_frame

        if frame_count < frame_rules["minimum_frames"]:
            frame_score = 20.0
            frame_status = HealthStatus.CRITICAL
            frame_msg = f"Too few frames ({frame_count}) for reliable analysis"
            frame_suggestions = [
                f"Use at least {frame_rules['recommended_minimum']} frames"
            ]
        elif frame_count < frame_rules["recommended_minimum"]:
            frame_score = 60.0
            frame_status = HealthStatus.WARNING
            frame_msg = f"Frame count ({frame_count}) is below recommended minimum"
            frame_suggestions = [
                f"Consider using at least {frame_rules['recommended_minimum']} frames for better statistics"
            ]
        elif frame_count > frame_rules["warning_above"]:
            frame_score = 70.0
            frame_status = HealthStatus.WARNING
            frame_msg = (
                f"Large frame count ({frame_count}) may increase computation time"
            )
            frame_suggestions = [
                "Consider reducing frame count for faster analysis",
                "Ensure sufficient computational resources",
            ]
        else:
            frame_score = 100.0
            frame_status = HealthStatus.EXCELLENT
            frame_msg = f"Frame count ({frame_count}) is optimal for analysis"
            frame_suggestions = []

        param_scores.append(frame_score)
        metrics.append(
            HealthMetric(
                name="frame_count_validation",
                value=frame_score,
                status=frame_status,
                message=frame_msg,
                suggestions=frame_suggestions,
                impact="high" if frame_score < 50 else "medium",
                category="performance",
            )
        )

        # Validate scattering parameters
        scattering = analyzer_params.get("scattering", {})
        q = scattering.get("wavevector_q", 0.0054)

        q_range = self.validation_rules["parameter_ranges"]["wavevector_q"]
        physics_constraints = self.validation_rules["physics_constraints"]

        if not (q_range[0] <= q <= q_range[1]):
            q_score = 10.0
            q_status = HealthStatus.CRITICAL
            q_msg = f"Wavevector q = {q} Å⁻¹ is outside valid range"
            q_suggestions = [f"Set q between {q_range[0]} and {q_range[1]} Å⁻¹"]
        elif not (
            physics_constraints["q_typical_min"]
            <= q
            <= physics_constraints["q_typical_max"]
        ):
            q_score = 70.0
            q_status = HealthStatus.WARNING
            q_msg = f"Wavevector q = {q} Å⁻¹ is outside typical XPCS range"
            q_suggestions = [
                f"Typical XPCS range is {physics_constraints['q_typical_min']}-{physics_constraints['q_typical_max']} Å⁻¹"
            ]
        else:
            q_score = 100.0
            q_status = HealthStatus.EXCELLENT
            q_msg = f"Wavevector q = {q} Å⁻¹ is in typical XPCS range"
            q_suggestions = []

        param_scores.append(q_score)
        metrics.append(
            HealthMetric(
                name="wavevector_validation",
                value=q_score,
                status=q_status,
                message=q_msg,
                suggestions=q_suggestions,
                impact="high" if q_score < 50 else "low",
                category="physics",
            )
        )

        # Validate initial parameters
        initial_params = config_data.get("initial_parameters", {})
        param_values = initial_params.get("values", [])
        param_names = initial_params.get("parameter_names", [])

        if len(param_values) != len(param_names):
            param_consistency_score = 20.0
            param_consistency_status = HealthStatus.CRITICAL
            param_consistency_msg = f"Parameter count mismatch: {len(param_values)} values vs {len(param_names)} names"
            param_consistency_suggestions = [
                "Ensure parameter values and names arrays have the same length"
            ]
        else:
            param_consistency_score = 100.0
            param_consistency_status = HealthStatus.EXCELLENT
            param_consistency_msg = (
                f"Parameter arrays are consistent ({len(param_values)} parameters)"
            )
            param_consistency_suggestions = []

        param_scores.append(param_consistency_score)
        metrics.append(
            HealthMetric(
                name="parameter_consistency",
                value=param_consistency_score,
                status=param_consistency_status,
                message=param_consistency_msg,
                suggestions=param_consistency_suggestions,
                impact="critical" if param_consistency_score < 50 else "low",
                category="configuration",
            )
        )

        return metrics

    def _validate_physics_constraints(
        self, config_data: Dict[str, Any]
    ) -> List[HealthMetric]:
        """Validate physics constraints and model consistency."""
        metrics = []

        # Get analysis mode
        analysis_settings = config_data.get("analysis_settings", {})
        static_mode = analysis_settings.get("static_mode", True)
        static_submode = analysis_settings.get("static_submode", "anisotropic")

        # Check mode-parameter consistency
        initial_params = config_data.get("initial_parameters", {})
        param_count = len(initial_params.get("values", []))

        expected_count = 3 if static_mode else 7

        if param_count == expected_count:
            mode_score = 100.0
            mode_status = HealthStatus.EXCELLENT
            mode_msg = f"Parameter count ({param_count}) matches analysis mode"
            mode_suggestions = []
        else:
            mode_score = 20.0
            mode_status = HealthStatus.CRITICAL
            mode_msg = f"Parameter count ({param_count}) doesn't match analysis mode (expected {expected_count})"
            mode_suggestions = [
                f"Use {expected_count} parameters for current mode",
                "Change analysis mode to match parameter count",
            ]

        metrics.append(
            HealthMetric(
                name="mode_parameter_consistency",
                value=mode_score,
                status=mode_status,
                message=mode_msg,
                suggestions=mode_suggestions,
                impact="critical" if mode_score < 50 else "low",
                category="physics",
            )
        )

        # Validate parameter physical reasonableness
        param_values = initial_params.get("values", [])
        param_names = initial_params.get("parameter_names", [])

        physics_scores = []

        for i, (name, value) in enumerate(zip(param_names, param_values)):
            if name in self.validation_rules["parameter_ranges"]:
                param_range = self.validation_rules["parameter_ranges"][name]

                if param_range[0] <= value <= param_range[1]:
                    param_score = 100.0
                    param_status = HealthStatus.EXCELLENT
                    param_msg = f"Parameter {name} = {value} is within bounds"
                    param_suggestions = []
                else:
                    param_score = 20.0
                    param_status = HealthStatus.CRITICAL
                    param_msg = (
                        f"Parameter {name} = {value} is outside bounds {param_range}"
                    )
                    param_suggestions = [
                        f"Set {name} between {param_range[0]} and {param_range[1]}"
                    ]

                physics_scores.append(param_score)

                if self.feedback_system:
                    self.feedback_system.show_parameter_validation(
                        name,
                        value,
                        param_score > 50,
                        param_range if param_score <= 50 else None,
                    )

                if param_score < 100:  # Only add metrics for problematic parameters
                    metrics.append(
                        HealthMetric(
                            name=f"parameter_{name}_validation",
                            value=param_score,
                            status=param_status,
                            message=param_msg,
                            suggestions=param_suggestions,
                            impact="high" if param_score < 50 else "medium",
                            category="physics",
                        )
                    )

        # Overall physics validation score
        if physics_scores:
            overall_physics_score = sum(physics_scores) / len(physics_scores)
            overall_physics_status = self._score_to_status(overall_physics_score)

            metrics.append(
                HealthMetric(
                    name="overall_physics_validation",
                    value=overall_physics_score,
                    status=overall_physics_status,
                    message=f"Overall physics validation score: {overall_physics_score:.1f}%",
                    suggestions=(
                        []
                        if overall_physics_score > 80
                        else ["Review parameter values for physical reasonableness"]
                    ),
                    impact="high" if overall_physics_score < 70 else "medium",
                    category="physics",
                )
            )

        return metrics

    def _assess_performance_configuration(
        self, config_data: Dict[str, Any]
    ) -> List[HealthMetric]:
        """Assess performance-related configuration settings."""
        metrics = []

        # Check frame count impact on performance
        analyzer_params = config_data.get("analyzer_parameters", {})
        temporal = analyzer_params.get("temporal", {})
        start_frame = temporal.get("start_frame", 1)
        end_frame = temporal.get("end_frame", 100)
        frame_count = end_frame - start_frame

        # Performance impact scoring
        if frame_count < 1000:
            perf_score = 100.0
            perf_status = HealthStatus.EXCELLENT
            perf_msg = "Low frame count - excellent performance expected"
            perf_suggestions = []
        elif frame_count < 5000:
            perf_score = 80.0
            perf_status = HealthStatus.GOOD
            perf_msg = "Moderate frame count - good performance expected"
            perf_suggestions = []
        elif frame_count < 20000:
            perf_score = 60.0
            perf_status = HealthStatus.WARNING
            perf_msg = "High frame count - may impact performance"
            perf_suggestions = [
                "Consider reducing frame count for faster analysis",
                "Enable performance optimizations",
            ]
        else:
            perf_score = 30.0
            perf_status = HealthStatus.CRITICAL
            perf_msg = "Very high frame count - significant performance impact expected"
            perf_suggestions = [
                "Strongly consider reducing frame count",
                "Enable GPU acceleration if available",
                "Use aggressive caching",
            ]

        metrics.append(
            HealthMetric(
                name="frame_count_performance_impact",
                value=perf_score,
                status=perf_status,
                message=perf_msg,
                suggestions=perf_suggestions,
                impact="medium" if perf_score > 50 else "high",
                category="performance",
            )
        )

        # Check optimization settings
        opt_config = config_data.get("optimization_config", {})

        # Angle filtering performance impact
        angle_filtering = opt_config.get("angle_filtering", {})
        filtering_enabled = angle_filtering.get("enabled", True)

        analysis_settings = config_data.get("analysis_settings", {})
        static_submode = analysis_settings.get("static_submode", "anisotropic")

        if static_submode == "isotropic" and filtering_enabled:
            angle_score = 70.0
            angle_status = HealthStatus.WARNING
            angle_msg = (
                "Angle filtering enabled for isotropic mode (unnecessary overhead)"
            )
            angle_suggestions = [
                "Disable angle filtering for isotropic mode to improve performance"
            ]
        elif static_submode != "isotropic" and not filtering_enabled:
            angle_score = 80.0
            angle_status = HealthStatus.GOOD
            angle_msg = "Angle filtering disabled - faster but may reduce accuracy"
            angle_suggestions = [
                "Consider enabling angle filtering for better analysis quality"
            ]
        else:
            angle_score = 100.0
            angle_status = HealthStatus.EXCELLENT
            angle_msg = "Angle filtering configuration is optimal"
            angle_suggestions = []

        metrics.append(
            HealthMetric(
                name="angle_filtering_optimization",
                value=angle_score,
                status=angle_status,
                message=angle_msg,
                suggestions=angle_suggestions,
                impact="low",
                category="performance",
            )
        )

        # Check v2 features performance settings
        v2_features = config_data.get("v2_features", {})

        performance_features = [
            ("performance_optimization", "performance optimizations"),
            ("parallel_processing", "parallel processing"),
            ("gpu_acceleration", "GPU acceleration"),
        ]

        enabled_features = []
        disabled_features = []

        for feature, description in performance_features:
            if v2_features.get(feature, False):
                enabled_features.append(description)
            else:
                disabled_features.append(description)

        feature_score = len(enabled_features) / len(performance_features) * 100
        feature_status = self._score_to_status(feature_score)

        if feature_score == 100:
            feature_msg = "All performance features are enabled"
            feature_suggestions = []
        elif feature_score > 60:
            feature_msg = f"Most performance features enabled ({len(enabled_features)}/{len(performance_features)})"
            feature_suggestions = [f"Consider enabling: {', '.join(disabled_features)}"]
        else:
            feature_msg = f"Few performance features enabled ({len(enabled_features)}/{len(performance_features)})"
            feature_suggestions = [
                f"Enable performance features: {', '.join(disabled_features)}"
            ]

        metrics.append(
            HealthMetric(
                name="performance_features",
                value=feature_score,
                status=feature_status,
                message=feature_msg,
                suggestions=feature_suggestions,
                impact="medium",
                category="performance",
            )
        )

        return metrics

    def _check_system_compatibility(
        self, config_data: Dict[str, Any]
    ) -> List[HealthMetric]:
        """Check system compatibility and resource requirements."""
        metrics = []

        # Memory requirements assessment
        analyzer_params = config_data.get("analyzer_parameters", {})
        temporal = analyzer_params.get("temporal", {})
        frame_count = temporal.get("end_frame", 100) - temporal.get("start_frame", 1)

        # Rough memory estimation (MB)
        estimated_memory = frame_count * 0.5  # ~0.5 MB per frame (rough estimate)
        available_memory = self.system_info.get("memory_available", 0) / (1024**3)  # GB

        memory_usage_ratio = (
            estimated_memory / (available_memory * 1024)
            if available_memory > 0
            else 1.0
        )

        if memory_usage_ratio < 0.1:
            memory_score = 100.0
            memory_status = HealthStatus.EXCELLENT
            memory_msg = (
                f"Estimated memory usage: {estimated_memory:.1f}MB (plenty available)"
            )
            memory_suggestions = []
        elif memory_usage_ratio < 0.3:
            memory_score = 80.0
            memory_status = HealthStatus.GOOD
            memory_msg = f"Estimated memory usage: {estimated_memory:.1f}MB (adequate)"
            memory_suggestions = []
        elif memory_usage_ratio < 0.7:
            memory_score = 60.0
            memory_status = HealthStatus.WARNING
            memory_msg = (
                f"Estimated memory usage: {estimated_memory:.1f}MB (may be tight)"
            )
            memory_suggestions = [
                "Monitor memory usage during analysis",
                "Consider reducing frame count if needed",
            ]
        else:
            memory_score = 20.0
            memory_status = HealthStatus.CRITICAL
            memory_msg = f"Estimated memory usage: {estimated_memory:.1f}MB (likely insufficient)"
            memory_suggestions = [
                "Reduce frame count to lower memory requirements",
                "Enable aggressive caching to disk",
                "Add more system memory if possible",
            ]

        metrics.append(
            HealthMetric(
                name="memory_requirements",
                value=memory_score,
                status=memory_status,
                message=memory_msg,
                suggestions=memory_suggestions,
                impact="high" if memory_score < 50 else "medium",
                category="system",
            )
        )

        # GPU compatibility check
        v2_features = config_data.get("v2_features", {})
        gpu_requested = v2_features.get("gpu_acceleration", False)
        gpu_available = self.system_info.get("gpu_available", False)

        if gpu_requested and gpu_available:
            gpu_score = 100.0
            gpu_status = HealthStatus.EXCELLENT
            gpu_msg = "GPU acceleration requested and available"
            gpu_suggestions = []
        elif gpu_requested and not gpu_available:
            gpu_score = 30.0
            gpu_status = HealthStatus.WARNING
            gpu_msg = "GPU acceleration requested but not available"
            gpu_suggestions = [
                "Install JAX with GPU support",
                "Check CUDA installation",
                "Disable GPU acceleration if not available",
            ]
        elif not gpu_requested and gpu_available:
            gpu_score = 80.0
            gpu_status = HealthStatus.GOOD
            gpu_msg = "GPU available but not enabled - missing performance opportunity"
            gpu_suggestions = ["Enable GPU acceleration for better performance"]
        else:
            gpu_score = 70.0
            gpu_status = HealthStatus.GOOD
            gpu_msg = "GPU acceleration not requested (CPU-only analysis)"
            gpu_suggestions = []

        metrics.append(
            HealthMetric(
                name="gpu_compatibility",
                value=gpu_score,
                status=gpu_status,
                message=gpu_msg,
                suggestions=gpu_suggestions,
                impact="low",
                category="system",
            )
        )

        # CPU utilization check
        cpu_percent = self.system_info.get("cpu_percent", 0)
        cpu_count = self.system_info.get("cpu_count", 1)

        parallel_processing = v2_features.get("parallel_processing", False)

        if parallel_processing:
            if cpu_count >= 4:
                cpu_score = 100.0
                cpu_status = HealthStatus.EXCELLENT
                cpu_msg = f"Parallel processing enabled with {cpu_count} CPU cores"
                cpu_suggestions = []
            else:
                cpu_score = 70.0
                cpu_status = HealthStatus.WARNING
                cpu_msg = (
                    f"Parallel processing enabled but only {cpu_count} cores available"
                )
                cpu_suggestions = [
                    "Consider disabling parallel processing on single-core systems"
                ]
        else:
            if cpu_count >= 4:
                cpu_score = 80.0
                cpu_status = HealthStatus.GOOD
                cpu_msg = (
                    f"Parallel processing disabled with {cpu_count} cores available"
                )
                cpu_suggestions = ["Enable parallel processing for better performance"]
            else:
                cpu_score = 90.0
                cpu_status = HealthStatus.EXCELLENT
                cpu_msg = (
                    f"Single-threaded processing appropriate for {cpu_count} cores"
                )
                cpu_suggestions = []

        metrics.append(
            HealthMetric(
                name="cpu_utilization",
                value=cpu_score,
                status=cpu_status,
                message=cpu_msg,
                suggestions=cpu_suggestions,
                impact="medium",
                category="system",
            )
        )

        return metrics

    def _check_security_and_best_practices(
        self, config_data: Dict[str, Any], config_path: Path
    ) -> List[HealthMetric]:
        """Check security and best practices compliance."""
        metrics = []

        # Check for hardcoded sensitive paths
        sensitive_patterns = ["/home/", "C:\\Users\\", "password", "secret", "key"]
        config_str = json.dumps(config_data, indent=2)

        found_patterns = []
        for pattern in sensitive_patterns:
            if pattern.lower() in config_str.lower():
                found_patterns.append(pattern)

        if found_patterns:
            security_score = max(20, 100 - len(found_patterns) * 20)
            security_status = (
                HealthStatus.WARNING if security_score > 50 else HealthStatus.CRITICAL
            )
            security_msg = f"Potentially sensitive information detected: {', '.join(found_patterns)}"
            security_suggestions = [
                "Use relative paths instead of absolute paths",
                "Remove any hardcoded credentials or sensitive information",
                "Consider using environment variables for sensitive data",
            ]
        else:
            security_score = 100.0
            security_status = HealthStatus.EXCELLENT
            security_msg = "No obvious security issues detected"
            security_suggestions = []

        metrics.append(
            HealthMetric(
                name="security_check",
                value=security_score,
                status=security_status,
                message=security_msg,
                suggestions=security_suggestions,
                impact="medium" if security_score < 80 else "low",
                category="security",
            )
        )

        # Check configuration file permissions
        try:
            file_stat = config_path.stat()
            file_mode = oct(file_stat.st_mode)[-3:]

            if file_mode == "644" or file_mode == "664":
                perm_score = 100.0
                perm_status = HealthStatus.EXCELLENT
                perm_msg = f"File permissions are appropriate ({file_mode})"
                perm_suggestions = []
            elif file_mode[2] in ["6", "7"]:  # World writable
                perm_score = 30.0
                perm_status = HealthStatus.WARNING
                perm_msg = f"File is world-writable ({file_mode})"
                perm_suggestions = ["Remove world write permissions: chmod 644"]
            else:
                perm_score = 80.0
                perm_status = HealthStatus.GOOD
                perm_msg = f"File permissions: {file_mode}"
                perm_suggestions = []

            metrics.append(
                HealthMetric(
                    name="file_permissions",
                    value=perm_score,
                    status=perm_status,
                    message=perm_msg,
                    suggestions=perm_suggestions,
                    impact="low",
                    category="security",
                )
            )

        except Exception as e:
            logger.warning(f"Could not check file permissions: {e}")

        return metrics

    def _calculate_config_depth(
        self, config_data: Dict[str, Any], current_depth: int = 1
    ) -> int:
        """Calculate maximum depth of configuration structure."""
        max_depth = current_depth

        if isinstance(config_data, dict):
            for value in config_data.values():
                if isinstance(value, (dict, list)):
                    depth = self._calculate_config_depth(value, current_depth + 1)
                    max_depth = max(max_depth, depth)
        elif isinstance(config_data, list):
            for item in config_data:
                if isinstance(item, (dict, list)):
                    depth = self._calculate_config_depth(item, current_depth + 1)
                    max_depth = max(max_depth, depth)

        return max_depth

    def _count_config_keys(self, config_data: Dict[str, Any]) -> int:
        """Count total number of keys in configuration."""
        count = 0

        if isinstance(config_data, dict):
            count += len(config_data)
            for value in config_data.values():
                if isinstance(value, (dict, list)):
                    count += self._count_config_keys(value)
        elif isinstance(config_data, list):
            for item in config_data:
                if isinstance(item, (dict, list)):
                    count += self._count_config_keys(item)

        return count

    def _calculate_overall_score(self, metrics: List[HealthMetric]) -> float:
        """Calculate overall health score from individual metrics."""
        if not metrics:
            return 0.0

        # Weight metrics by impact
        impact_weights = {"critical": 5.0, "high": 3.0, "medium": 2.0, "low": 1.0}

        total_weighted_score = 0.0
        total_weight = 0.0

        for metric in metrics:
            weight = impact_weights.get(metric.impact, 1.0)
            total_weighted_score += metric.value * weight
            total_weight += weight

        return total_weighted_score / total_weight if total_weight > 0 else 0.0

    def _score_to_status(self, score: float) -> HealthStatus:
        """Convert numeric score to health status."""
        if score >= 90:
            return HealthStatus.EXCELLENT
        elif score >= 70:
            return HealthStatus.GOOD
        elif score >= 50:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _generate_recommendations(
        self, metrics: List[HealthMetric], config_data: Dict[str, Any]
    ) -> List[str]:
        """Generate overall recommendations based on health metrics."""
        recommendations = []

        # Critical issues first
        critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
        if critical_metrics:
            recommendations.append("🚨 Address critical issues immediately:")
            for metric in critical_metrics[:3]:  # Top 3 critical issues
                recommendations.append(f"  • {metric.message}")
                if metric.suggestions:
                    recommendations.append(f"    → {metric.suggestions[0]}")

        # Performance optimization suggestions
        perf_metrics = [
            m
            for m in metrics
            if m.category == "performance" and m.status != HealthStatus.EXCELLENT
        ]
        if perf_metrics:
            recommendations.append("🚀 Performance optimization opportunities:")
            for metric in perf_metrics[:2]:  # Top 2 performance improvements
                if metric.suggestions:
                    recommendations.append(f"  • {metric.suggestions[0]}")

        # Configuration improvement suggestions
        config_metrics = [
            m
            for m in metrics
            if m.category == "configuration" and m.status == HealthStatus.WARNING
        ]
        if config_metrics:
            recommendations.append("⚙️ Configuration improvements:")
            for metric in config_metrics[:2]:  # Top 2 config improvements
                if metric.suggestions:
                    recommendations.append(f"  • {metric.suggestions[0]}")

        # Overall recommendations
        overall_score = self._calculate_overall_score(metrics)
        if overall_score < 70:
            recommendations.append(
                "📋 Consider using the interactive configuration builder for guidance"
            )
            recommendations.append("🔧 Run automated recovery to fix common issues")
        elif overall_score < 90:
            recommendations.append(
                "✨ Configuration is good - small tweaks could make it excellent"
            )

        return recommendations

    def _show_health_feedback(self, report: HealthReport):
        """Show health feedback using the validation feedback system."""
        if not self.feedback_system:
            return

        # Show critical issues
        critical_issues = report.get_critical_issues()
        for issue in critical_issues:
            self.feedback_system.add_error(
                issue.message, issue.suggestions[0] if issue.suggestions else None
            )

        # Show warnings
        warnings = report.get_warnings()
        for warning in warnings:
            self.feedback_system.add_warning(
                warning.message, warning.suggestions[0] if warning.suggestions else None
            )

        # Show performance notes
        perf_metrics = report.get_metrics_by_category("performance")
        for metric in perf_metrics:
            if metric.status == HealthStatus.EXCELLENT:
                self.feedback_system.add_performance_note(metric.message)

        # Show final results
        final_result = self.feedback_system.finish_validation()
        return final_result

    def generate_health_report_file(
        self, report: HealthReport, output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Generate a detailed health report file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"homodyne_health_report_{timestamp}.json"

        output_path = Path(output_path)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2, default=str)

            logger.info(f"Health report saved to: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to save health report: {e}")
            raise

    def continuous_monitoring(
        self, config_path: Union[str, Path], check_interval: int = 300
    ) -> None:
        """Continuously monitor configuration health."""
        config_path = Path(config_path)

        if self.logger:
            self.logger.info(f"Starting continuous monitoring of {config_path}")
            self.logger.info(f"Check interval: {check_interval} seconds")

        last_modified = 0

        try:
            while True:
                try:
                    # Check if file has been modified
                    current_modified = config_path.stat().st_mtime

                    if current_modified != last_modified:
                        if self.logger:
                            self.logger.info(
                                "Configuration file changed - running health check"
                            )

                        report = self.perform_health_check(config_path)

                        # Alert on critical issues
                        critical_issues = report.get_critical_issues()
                        if critical_issues:
                            if self.logger:
                                self.logger.error(
                                    f"Found {len(critical_issues)} critical issues!"
                                )
                                for issue in critical_issues:
                                    self.logger.error(f"  • {issue.message}")

                        last_modified = current_modified

                    time.sleep(check_interval)

                except KeyboardInterrupt:
                    if self.logger:
                        self.logger.info("Continuous monitoring stopped by user")
                    break
                except Exception as e:
                    logger.error(f"Error during continuous monitoring: {e}")
                    time.sleep(check_interval)  # Continue monitoring despite errors

        except Exception as e:
            logger.error(f"Continuous monitoring failed: {e}")
            raise


# Factory functions
def create_health_monitor(
    validation_level: ValidationLevel = ValidationLevel.STANDARD,
    logger: Optional[EnhancedConsoleLogger] = None,
) -> ConfigurationHealthMonitor:
    """Create a configuration health monitor."""
    return ConfigurationHealthMonitor(validation_level=validation_level, logger=logger)


def quick_health_check(config_path: Union[str, Path]) -> HealthReport:
    """Perform a quick health check on a configuration file."""
    monitor = create_health_monitor(ValidationLevel.BASIC)
    return monitor.perform_health_check(config_path)


def comprehensive_health_check(config_path: Union[str, Path]) -> HealthReport:
    """Perform a comprehensive health check on a configuration file."""
    monitor = create_health_monitor(ValidationLevel.COMPREHENSIVE)
    return monitor.perform_health_check(config_path)


# CLI integration
def main_health_check():
    """Main function for CLI health check."""
    import argparse

    parser = argparse.ArgumentParser(description="Homodyne Configuration Health Check")
    parser.add_argument("config", help="Configuration file to check")
    parser.add_argument(
        "--level",
        choices=["basic", "standard", "comprehensive", "strict"],
        default="standard",
        help="Validation level",
    )
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    parser.add_argument(
        "--continuous", "-c", action="store_true", help="Enable continuous monitoring"
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=300, help="Monitoring interval in seconds"
    )

    args = parser.parse_args()

    # Create health monitor
    validation_level = ValidationLevel(args.level)
    monitor = create_health_monitor(validation_level)

    if args.continuous:
        monitor.continuous_monitoring(args.config, args.interval)
    else:
        # Single health check
        report = monitor.perform_health_check(args.config)

        print(f"\n{'=' * 60}")
        print(f"Configuration Health Report")
        print(f"{'=' * 60}")
        print(f"Overall Score: {report.overall_score:.1f}/100")
        print(f"Status: {report.overall_status.value.upper()}")

        critical_issues = report.get_critical_issues()
        if critical_issues:
            print(f"\n🚨 Critical Issues ({len(critical_issues)}):")
            for issue in critical_issues:
                print(f"  • {issue.message}")

        warnings = report.get_warnings()
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:5]:  # Show top 5 warnings
                print(f"  • {warning.message}")

        if report.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in report.recommendations[:5]:  # Show top 5 recommendations
                print(f"  {rec}")

        # Save detailed report if requested
        if args.output:
            report_file = monitor.generate_health_report_file(report, args.output)
            print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main_health_check()
