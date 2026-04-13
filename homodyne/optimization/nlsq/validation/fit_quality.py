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
    chi2_good_threshold : float
        Reduced chi-squared below which fit is classified as "good". Default: 2.0.
    chi2_acceptable_threshold : float
        Reduced chi-squared below which fit is classified as "acceptable". Default: 5.0.
    min_parameter_significance : float
        Minimum parameter/uncertainty ratio for significance. Default: 2.0.
    max_condition_number : float
        Maximum covariance matrix condition number. Default: 1e12.
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
    chi2_good_threshold: float = 2.0
    chi2_acceptable_threshold: float = 5.0
    min_parameter_significance: float = 2.0
    max_condition_number: float = 1e12
    warn_on_max_restarts: bool = True
    warn_on_bounds_hit: bool = True
    warn_on_convergence_failure: bool = True
    bounds_tolerance: float = 1e-9

    @classmethod
    def from_validation_config(
        cls, validation_config: dict[str, Any] | None
    ) -> FitQualityConfig:
        """Create FitQualityConfig from an NLSQValidationConfig dict.

        Parameters
        ----------
        validation_config : dict or None
            Dictionary with keys from NLSQValidationConfig TypedDict.
            If None, returns defaults.

        Returns
        -------
        FitQualityConfig
            Configuration with values from the dict, falling back to defaults.
        """
        if validation_config is None:
            return cls()
        return cls(
            chi2_good_threshold=validation_config.get("chi2_good_threshold", 2.0),
            chi2_acceptable_threshold=validation_config.get(
                "chi2_acceptable_threshold", 5.0
            ),
            min_parameter_significance=validation_config.get(
                "min_parameter_significance", 2.0
            ),
            max_condition_number=validation_config.get("max_condition_number", 1e12),
        )


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
    for _i, (val, lb, ub) in enumerate(zip(values, lower, upper, strict=False)):
        if abs(val - lb) < atol * (1.0 + abs(lb)):
            statuses.append("at_lower_bound")
        elif abs(val - ub) < atol * (1.0 + abs(ub)):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def _is_physical_param(label: str) -> bool:
    """Check if a parameter label is for a physical parameter (not per-angle scaling).

    Derives scaling identity from the ``is_scaling`` flag on
    :class:`~homodyne.config.parameter_registry.ParameterInfo`.

    Parameters
    ----------
    label : str
        Parameter label (e.g. ``"D0"``, ``"contrast_0"``).

    Returns
    -------
    bool
        True if this is a physical parameter.
    """
    from homodyne.config.parameter_registry import ParameterRegistry

    scaling = ParameterRegistry().scaling_names
    return not any(
        label.startswith(f"{s}_") or label.startswith(f"{s}[") or label == s
        for s in scaling
    )


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
            sigma_is_default = getattr(result, "sigma_is_default", False)
            if sigma_is_default:
                warning = (
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}), but sigma was not provided "
                    f"(using default 0.01). Chi-squared is not physically meaningful "
                    f"without experimental uncertainties. Fit quality should be assessed "
                    f"by inspecting residuals directly."
                )
            else:
                warning = (
                    f"Reduced chi-squared ({reduced_chi_squared:.4g}) exceeds threshold "
                    f"({config.reduced_chi_squared_threshold}). Consider reviewing fit quality."
                )
            report.warnings.append(warning)
            logger.warning(f"[FitQuality] {warning}")
            report.passed = False

        # Classify fit quality using configurable thresholds
        if reduced_chi_squared <= config.chi2_good_threshold:
            report.checks_performed["chi2_quality"] = True
            logger.info(
                "[FitQuality] Chi-squared quality: good (%.4g <= %.4g)",
                reduced_chi_squared,
                config.chi2_good_threshold,
            )
        elif reduced_chi_squared <= config.chi2_acceptable_threshold:
            report.checks_performed["chi2_quality"] = True
            logger.info(
                "[FitQuality] Chi-squared quality: acceptable (%.4g <= %.4g)",
                reduced_chi_squared,
                config.chi2_acceptable_threshold,
            )
        else:
            report.checks_performed["chi2_quality"] = False
            logger.warning(
                "[FitQuality] Chi-squared quality: poor (%.4g > %.4g)",
                reduced_chi_squared,
                config.chi2_acceptable_threshold,
            )

    # Check 1b: Parameter significance (parameter / uncertainty ratio)
    params = getattr(result, "parameters", None)
    uncertainties = getattr(result, "uncertainties", None)
    if params is not None and uncertainties is not None:
        try:
            params_arr = np.asarray(params, dtype=np.float64)
            uncert_arr = np.asarray(uncertainties, dtype=np.float64)
            if params_arr.shape == uncert_arr.shape and len(params_arr) > 0:
                finite_mask = np.isfinite(uncert_arr) & (uncert_arr > 0)
                if np.any(finite_mask):
                    significance = (
                        np.abs(params_arr[finite_mask]) / uncert_arr[finite_mask]
                    )
                    insignificant = significance < config.min_parameter_significance
                    if np.any(insignificant):
                        n_insig = int(np.sum(insignificant))
                        report.checks_performed["parameter_significance"] = False
                        warning = (
                            f"{n_insig} parameter(s) below significance threshold "
                            f"(|param/uncertainty| < {config.min_parameter_significance}). "
                            f"These parameters may be poorly constrained."
                        )
                        report.warnings.append(warning)
                        logger.warning(f"[FitQuality] {warning}")
                    else:
                        report.checks_performed["parameter_significance"] = True
        except (TypeError, ValueError):
            pass

    # Check 1c: Covariance matrix condition number
    pcov = getattr(result, "covariance", None)
    if pcov is None:
        pcov = getattr(result, "pcov", None)
    if pcov is not None:
        try:
            pcov_arr = np.asarray(pcov, dtype=np.float64)
            if pcov_arr.ndim == 2 and pcov_arr.shape[0] == pcov_arr.shape[1]:
                cond = np.linalg.cond(pcov_arr)
                if np.isfinite(cond):
                    if cond > config.max_condition_number:
                        report.checks_performed["condition_number"] = False
                        warning = (
                            f"Covariance matrix condition number ({cond:.2e}) exceeds "
                            f"threshold ({config.max_condition_number:.2e}). "
                            f"Parameters may be highly correlated or poorly determined."
                        )
                        report.warnings.append(warning)
                        logger.warning(f"[FitQuality] {warning}")
                        report.passed = False
                    else:
                        report.checks_performed["condition_number"] = True
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass

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
                warning = (
                    f"Optimization did not converge successfully (status: {status})."
                )
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
