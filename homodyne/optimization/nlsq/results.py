"""NLSQ optimization result classes.

This module extracts result dataclasses from nlsq_wrapper.py
to reduce file size and improve maintainability.

Extracted from nlsq_wrapper.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from homodyne.optimization.nlsq.strategies.chunking import StratificationDiagnostics


@dataclass
class FunctionEvaluationCounter:
    """Wraps a callable and counts invocations.

    Useful for tracking the number of function evaluations during optimization.
    """

    fn: Callable[..., Any]
    count: int = 0

    def __call__(self, *args, **kwargs):
        """Call the wrapped function and increment count."""
        self.count += 1
        return self.fn(*args, **kwargs)


@dataclass
class OptimizationResult:
    """Complete optimization result with fit quality metrics and diagnostics.

    Attributes
    ----------
    parameters : np.ndarray
        Converged parameter values.
    uncertainties : np.ndarray
        Standard deviations from covariance matrix diagonal.
    covariance : np.ndarray
        Full parameter covariance matrix.
    chi_squared : float
        Sum of squared residuals.
    reduced_chi_squared : float
        chi_squared / (n_data - n_params).
    convergence_status : str
        'converged', 'max_iter', or 'failed'.
    iterations : int
        Number of optimization iterations.
    execution_time : float
        Wall-clock execution time in seconds.
    device_info : dict[str, Any]
        Device used for computation (CPU details).
    recovery_actions : list[str]
        List of error recovery actions taken.
    quality_flag : str
        'good', 'marginal', or 'poor'.
    streaming_diagnostics : dict[str, Any] | None
        Enhanced diagnostics for streaming optimization.
    stratification_diagnostics : StratificationDiagnostics | None
        Diagnostics for angle-stratified chunking.
    nlsq_diagnostics : dict[str, Any] | None
        Additional NLSQ-specific diagnostics.
    """

    parameters: np.ndarray
    uncertainties: np.ndarray
    covariance: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    convergence_status: str
    iterations: int
    execution_time: float
    device_info: dict[str, Any]
    recovery_actions: list[str] = field(default_factory=list)
    quality_flag: str = "good"
    streaming_diagnostics: dict[str, Any] | None = None
    stratification_diagnostics: StratificationDiagnostics | None = None
    nlsq_diagnostics: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        """Return True if optimization converged (backward compatibility)."""
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Return descriptive message about optimization outcome."""
        if self.convergence_status == "converged":
            return f"Optimization converged successfully. chi2={self.chi_squared:.6f}"
        elif self.convergence_status == "max_iter":
            return "Optimization stopped: maximum iterations reached"
        else:
            return f"Optimization failed: {self.convergence_status}"


# =============================================================================
# T010: FallbackInfo dataclass for adapter-to-wrapper fallback tracking
# =============================================================================
@dataclass
class FallbackInfo:
    """Tracks fallback from NLSQAdapter to NLSQWrapper.

    Included in OptimizationResult.device_info when fallback occurs.

    Attributes:
        fallback_occurred: True if fallback was triggered
        adapter_used: "NLSQAdapter" or "NLSQWrapper"
        adapter_error: Error message if adapter failed (None if succeeded)
        wrapper_error: Error message if wrapper also failed (None otherwise)

    States:

    * NLSQAdapter + fallback_occurred=False + adapter_error=None: Adapter succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="...": Fallback succeeded
    * NLSQWrapper + fallback_occurred=True + adapter_error="..." + wrapper_error="...": Both failed
    """

    fallback_occurred: bool
    adapter_used: str  # "NLSQAdapter" or "NLSQWrapper"
    adapter_error: str | None = None
    wrapper_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for inclusion in device_info."""
        return {
            "fallback_occurred": self.fallback_occurred,
            "adapter_used": self.adapter_used,
            "adapter_error": self.adapter_error,
            "wrapper_error": self.wrapper_error,
        }


@dataclass
class UseSequentialOptimization:
    """Marker indicating sequential per-angle optimization should be used.

    This is returned by _apply_stratification_if_needed when conditions require
    sequential per-angle optimization as a fallback strategy.

    Attributes
    ----------
    data : Any
        Original XPCS data object.
    reason : str
        Why sequential optimization is needed.
    """

    data: Any
    reason: str


def build_parameter_labels(
    per_angle_scaling: bool,
    n_phi: int,
    physical_param_names: list[str],
) -> list[str]:
    """Build parameter labels for optimization result.

    Parameters
    ----------
    per_angle_scaling : bool
        Whether per-angle scaling is enabled.
    n_phi : int
        Number of phi angles.
    physical_param_names : list[str]
        List of physical parameter names.

    Returns
    -------
    list[str]
        List of parameter labels.
    """
    labels: list[str] = []
    if per_angle_scaling:
        labels.extend([f"contrast[{i}]" for i in range(n_phi)])
        labels.extend([f"offset[{i}]" for i in range(n_phi)])
    labels.extend(physical_param_names)
    return labels


def classify_parameter_status(
    values: np.ndarray,
    lower: np.ndarray | None,
    upper: np.ndarray | None,
    atol: float = 1e-9,
) -> list[str]:
    """Classify parameter status relative to bounds.

    Parameters
    ----------
    values : np.ndarray
        Parameter values.
    lower : np.ndarray | None
        Lower bounds.
    upper : np.ndarray | None
        Upper bounds.
    atol : float
        Absolute tolerance for bound comparison.

    Returns
    -------
    list[str]
        Status for each parameter: 'active', 'at_lower_bound', or 'at_upper_bound'.
    """
    if lower is None or upper is None:
        return ["active"] * len(values)

    statuses: list[str] = []
    for value, lo, hi in zip(values, lower, upper, strict=False):
        if np.isclose(value, lo, atol=atol * (1.0 + abs(lo))):
            statuses.append("at_lower_bound")
        elif np.isclose(value, hi, atol=atol * (1.0 + abs(hi))):
            statuses.append("at_upper_bound")
        else:
            statuses.append("active")
    return statuses


def sample_xdata(xdata: np.ndarray, max_points: int) -> np.ndarray:
    """Sample xdata array for Jacobian computation.

    Parameters
    ----------
    xdata : np.ndarray
        Full xdata array.
    max_points : int
        Maximum number of points to sample.

    Returns
    -------
    np.ndarray
        Sampled xdata array.
    """
    if max_points <= 0 or xdata.size <= max_points:
        return xdata
    indices = np.linspace(0, xdata.size - 1, max_points, dtype=np.int64)
    return xdata[indices]
