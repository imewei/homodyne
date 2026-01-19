"""Result validation for NLSQ optimization (T080).

Extracted from wrapper.py as part of architecture refactoring.
"""

from __future__ import annotations

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class ResultValidator:
    """Validator for NLSQ optimization results.

    Validates optimized parameters, covariance matrices, and result consistency.
    """

    def __init__(self, strict_mode: bool = False):
        """Initialize ResultValidator.

        Parameters
        ----------
        strict_mode : bool, optional
            If True, raise errors on validation failures.
            If False, log warnings but continue.
        """
        self.strict_mode = strict_mode
        self._validation_warnings: list[str] = []

    def validate_all(
        self,
        params: np.ndarray,
        covariance: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        chi_squared: float | None = None,
    ) -> bool:
        """Validate all result components.

        Parameters
        ----------
        params : np.ndarray
            Optimized parameter values
        covariance : np.ndarray | None
            Parameter covariance matrix
        bounds : tuple[np.ndarray, np.ndarray] | None
            Parameter bounds (lower, upper)
        chi_squared : float | None, optional
            Chi-squared value for quality check

        Returns
        -------
        bool
            True if all validation passes, False otherwise
        """
        self._validation_warnings = []

        # Validate optimized params
        if not validate_optimized_params(params, bounds):
            self._validation_warnings.append("Optimized parameters outside bounds")

        # Validate covariance
        if covariance is not None:
            if not validate_covariance(covariance, len(params)):
                self._validation_warnings.append("Covariance matrix invalid")

        # Check result consistency
        if chi_squared is not None:
            if not validate_result_consistency(params, chi_squared):
                self._validation_warnings.append("Result consistency check failed")

        if self._validation_warnings:
            if self.strict_mode:
                raise ValueError(
                    f"Result validation failed: {'; '.join(self._validation_warnings)}"
                )
            else:
                for warning in self._validation_warnings:
                    logger.warning(f"Result validation warning: {warning}")
                return False

        return True

    @property
    def validation_warnings(self) -> list[str]:
        """Get list of validation warnings from last validate_all() call."""
        return self._validation_warnings.copy()


def validate_optimized_params(
    params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    tolerance: float = 1e-10,
) -> bool:
    """Validate that optimized parameters are finite and within bounds.

    Parameters
    ----------
    params : np.ndarray
        Optimized parameter values
    bounds : tuple[np.ndarray, np.ndarray] | None
        (lower, upper) bounds arrays
    tolerance : float, optional
        Tolerance for boundary violations

    Returns
    -------
    bool
        True if params are valid
    """
    # Check for NaN/Inf
    if not np.all(np.isfinite(params)):
        nan_count = np.sum(np.isnan(params))
        inf_count = np.sum(np.isinf(params))
        logger.warning(f"Optimized params contain {nan_count} NaN, {inf_count} Inf")
        return False

    if bounds is None:
        return True

    lower, upper = bounds

    # Check bounds with tolerance
    below_lower = params < (lower - tolerance)
    above_upper = params > (upper + tolerance)

    if np.any(below_lower) or np.any(above_upper):
        violations = []
        if np.any(below_lower):
            indices = np.where(below_lower)[0]
            violations.append(f"below lower at {indices.tolist()}")
        if np.any(above_upper):
            indices = np.where(above_upper)[0]
            violations.append(f"above upper at {indices.tolist()}")
        logger.warning(f"Params outside bounds: {', '.join(violations)}")
        return False

    return True


def validate_covariance(covariance: np.ndarray, n_params: int) -> bool:
    """Validate covariance matrix properties.

    Parameters
    ----------
    covariance : np.ndarray
        Parameter covariance matrix
    n_params : int
        Expected number of parameters

    Returns
    -------
    bool
        True if covariance is valid
    """
    # Check shape
    if covariance.shape != (n_params, n_params):
        logger.warning(
            f"Covariance shape {covariance.shape} != expected ({n_params}, {n_params})"
        )
        return False

    # Check for NaN/Inf
    if not np.all(np.isfinite(covariance)):
        nan_count = np.sum(np.isnan(covariance))
        inf_count = np.sum(np.isinf(covariance))
        logger.warning(f"Covariance contains {nan_count} NaN, {inf_count} Inf")
        return False

    # Check symmetry (with tolerance for numerical errors)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        max_diff = np.max(np.abs(covariance - covariance.T))
        logger.warning(f"Covariance not symmetric, max diff={max_diff:.2e}")
        return False

    # Check positive semi-definiteness (diagonal elements should be non-negative)
    diag = np.diag(covariance)
    if np.any(diag < 0):
        neg_indices = np.where(diag < 0)[0]
        logger.warning(
            f"Covariance has negative diagonal at indices: {neg_indices.tolist()}"
        )
        return False

    return True


def validate_result_consistency(
    params: np.ndarray,
    chi_squared: float,
) -> bool:
    """Validate consistency of optimization result.

    Parameters
    ----------
    params : np.ndarray
        Optimized parameter values
    chi_squared : float
        Chi-squared value

    Returns
    -------
    bool
        True if result is consistent
    """
    # Check chi-squared is finite and non-negative
    if not np.isfinite(chi_squared):
        logger.warning(f"Chi-squared is not finite: {chi_squared}")
        return False

    if chi_squared < 0:
        logger.warning(f"Chi-squared is negative: {chi_squared}")
        return False

    # Warn if chi-squared is suspiciously low (might indicate overfitting)
    if chi_squared < 1e-15:
        logger.warning(f"Chi-squared suspiciously low: {chi_squared:.2e}")

    # Warn if chi-squared is very high (might indicate poor fit)
    if chi_squared > 1e10:
        logger.warning(f"Chi-squared very high: {chi_squared:.2e}")

    return True


__all__ = [
    "ResultValidator",
    "validate_covariance",
    "validate_optimized_params",
    "validate_result_consistency",
]
