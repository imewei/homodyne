"""Fit quality validation for NLSQ results (T056).

Provides post-optimization quality checks with configurable thresholds.
Logs warnings for potential issues but does not raise exceptions.

Usage:
    >>> from homodyne.optimization.nlsq.validation.fit_quality import (
    ...     FitQualityConfig,
    ...     validate_fit_quality,
    ... )
    >>> config = FitQualityConfig(reduced_chi_squared_threshold=10.0)
    >>> report = validate_fit_quality(result, bounds=bounds, config=config)
    >>> if not report.passed:
    ...     print(f"Warnings: {report.warnings}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FitQualityConfig:
    """Configuration for fit quality validation.

    Attributes
    ----------
    enable : bool
        Whether to enable quality validation. Default: True.
    reduced_chi_squared_threshold : float
        Warn if reduced chi-squared exceeds this. Default: 10.0.
    warn_on_max_restarts : bool
        Warn if CMA-ES reached max_restarts. Default: True.
    warn_on_bounds_hit : bool
        Warn if physical parameters hit bounds. Default: True.
    warn_on_convergence_failure : bool
        Warn if convergence_status indicates failure. Default: True.
    bounds_tolerance : float
        Tolerance for "at bounds" detection. Default: 1e-9.
    """

    enable: bool = True
    reduced_chi_squared_threshold: float = 10.0
    warn_on_max_restarts: bool = True
    warn_on_bounds_hit: bool = True
    warn_on_convergence_failure: bool = True
    bounds_tolerance: float = 1e-9


@dataclass
class FitQualityReport:
    """Report from fit quality validation.

    Attributes
    ----------
    passed : bool
        True if no warnings were generated.
    warnings : list[str]
        List of warning messages.
    checks_performed : dict[str, bool]
        Which checks were performed and their pass/fail status.
    """

    passed: bool = True
    warnings: list[str] = field(default_factory=list)
    checks_performed: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving in results."""
        return {
            "quality_validation_passed": self.passed,
            "quality_warnings": self.warnings,
            "quality_checks": self.checks_performed,
        }


def _classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    atol: float = 1e-9,
) -> list[str]:
    """Classify each parameter's status relative to bounds.

    Parameters
    ----------
    values : np.ndarray
        Parameter values.
    lower : np.ndarray
        Lower bounds.
    upper : np.ndarray
        Upper bounds.
    atol : float
        Absolute tolerance for "at bound" detection.

    Returns
    -------
    list[str]
        Status for each parameter: "active", "at_lower_bound", or "at_upper_bound".
    """
    statuses = []
    for i, (val, lb, ub) in enumerate(zip(values, lower, upper)):
        if abs(val - lb) < atol:
            statuses.append("at_lower_bound")
        elif abs(val - ub) < atol:
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def _is_physical_param(label: str) -> bool:
    """Check if a parameter label is for a physical parameter (not per-angle scaling).

    Physical parameters include: D0, alpha, D_offset, gamma_dot_t0, beta,
    gamma_dot_t_offset, phi0.

    Per-angle scaling parameters are: contrast[*], offset[*].

    Parameters
    ----------
    label : str
        Parameter label.

    Returns
    -------
    bool
        True if this is a physical parameter.
    """
    # Per-angle scaling patterns to exclude
    scaling_patterns = ["contrast[", "offset[", "contrast_", "offset_"]
    for pattern in scaling_patterns:
        if pattern in label.lower():
            return False
    return True


def validate_fit_quality(
    result: Any,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    config: FitQualityConfig | None = None,
    param_labels: list[str] | None = None,
) -> FitQualityReport:
    """Validate fit quality and log warnings.

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result.
    bounds : tuple[np.ndarray, np.ndarray] | None
        Parameter bounds (lower, upper) for bounds checking.
    config : FitQualityConfig | None
        Validation configuration. Uses defaults if None.
    param_labels : list[str] | None
        Parameter labels for identifying physical vs scaling params.

    Returns
    -------
    FitQualityReport
        Validation report with warnings and check results.
    """
    if config is None:
        config = FitQualityConfig()

    if not config.enable:
        return FitQualityReport(passed=True, checks_performed={"enabled": False})

    report = FitQualityReport()

    # Check 1: Reduced chi-squared threshold
    reduced_chi_squared = getattr(result, "reduced_chi_squared", None)
    if reduced_chi_squared is not None:
        passed = reduced_chi_squared <= config.reduced_chi_squared_threshold
        report.checks_performed["reduced_chi_squared"] = passed

        if not passed:
            warning = (
                f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                f"({config.reduced_chi_squared_threshold}). Consider reviewing fit quality."
            )
            report.warnings.append(warning)
            logger.warning(f"[FitQuality] {warning}")
            report.passed = False

    # Check 2: CMA-ES max_restarts convergence
    if config.warn_on_max_restarts:
        device_info = getattr(result, "device_info", {}) or {}
        convergence_reason = device_info.get("convergence_reason", "")

        if convergence_reason == "max_restarts":
            report.checks_performed["cmaes_convergence"] = False
            warning = (
                "CMA-ES reached maximum restarts without convergence. "
                "Consider increasing max_restarts or adjusting sigma."
            )
            report.warnings.append(warning)
            logger.warning(f"[FitQuality] {warning}")
            report.passed = False
        elif convergence_reason:
            # CMA-ES was used and converged normally
            report.checks_performed["cmaes_convergence"] = True

    # Check 3: Physical parameters at bounds
    if config.warn_on_bounds_hit and bounds is not None:
        params = getattr(result, "parameters", None)
        if params is not None and len(params) > 0:
            lower, upper = bounds
            if len(params) == len(lower) == len(upper):
                statuses = _classify_parameter_status(
                    params, lower, upper, config.bounds_tolerance
                )

                # Identify physical parameters at bounds (exclude per-angle scaling)
                at_bounds = []
                for i, status in enumerate(statuses):
                    if status in ("at_lower_bound", "at_upper_bound"):
                        # Get label if available
                        label = (
                            param_labels[i]
                            if param_labels and i < len(param_labels)
                            else f"param[{i}]"
                        )

                        # Skip per-angle scaling parameters
                        if not _is_physical_param(label):
                            continue

                        at_bounds.append((label, status))

                report.checks_performed["physical_bounds"] = len(at_bounds) == 0

                if at_bounds:
                    params_str = ", ".join(
                        f"{label} ({status})" for label, status in at_bounds
                    )
                    warning = (
                        f"Physical parameters at bounds: {params_str}. "
                        "Consider expanding bounds or reviewing initial parameters."
                    )
                    report.warnings.append(warning)
                    logger.warning(f"[FitQuality] {warning}")
                    report.passed = False

    # Check 4: Convergence status
    if config.warn_on_convergence_failure:
        status = getattr(result, "convergence_status", "")
        failed_statuses = {"max_iter", "failed", "diverged", "max_iterations"}

        if status:
            passed = status.lower() not in failed_statuses
            report.checks_performed["convergence_status"] = passed

            if not passed:
                warning = f"Optimization did not converge successfully (status: {status})."
                report.warnings.append(warning)
                logger.warning(f"[FitQuality] {warning}")
                report.passed = False

    # Log summary
    if report.passed:
        logger.info("[FitQuality] All quality checks passed")
    else:
        logger.warning(
            f"[FitQuality] {len(report.warnings)} quality warning(s) generated"
        )

    return report


__all__ = [
    "FitQualityConfig",
    "FitQualityReport",
    "validate_fit_quality",
]
