"""Input validation for NLSQ optimization (T079).

Extracted from wrapper.py as part of architecture refactoring.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class InputValidator:
    """Validator for NLSQ optimization input data.

    Validates input arrays, bounds, initial parameters, and configuration
    before optimization begins.
    """

    def __init__(self, strict_mode: bool = True):
        """Initialize InputValidator.

        Parameters
        ----------
        strict_mode : bool, optional
            If True, raise errors on validation failures.
            If False, log warnings but continue.
        """
        self.strict_mode = strict_mode
        self._validation_errors: list[str] = []

    def validate_all(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> bool:
        """Validate all input data.

        Parameters
        ----------
        xdata : np.ndarray
            Independent variable data (t1, t2, phi)
        ydata : np.ndarray
            Dependent variable data (g2 values)
        initial_params : np.ndarray
            Initial parameter guess
        bounds : tuple[np.ndarray, np.ndarray] | None
            Parameter bounds (lower, upper)

        Returns
        -------
        bool
            True if all validation passes, False otherwise
        """
        self._validation_errors = []

        # Check array dimensions
        if not validate_array_dimensions(xdata, ydata):
            self._validation_errors.append(
                f"Array dimension mismatch: xdata.shape[0]={len(xdata)}, ydata.shape[0]={len(ydata)}"
            )

        # Check for NaN/Inf
        if not validate_no_nan_inf(xdata, "xdata"):
            self._validation_errors.append("xdata contains NaN or Inf values")
        if not validate_no_nan_inf(ydata, "ydata"):
            self._validation_errors.append("ydata contains NaN or Inf values")
        if not validate_no_nan_inf(initial_params, "initial_params"):
            self._validation_errors.append("initial_params contains NaN or Inf values")

        # Check bounds consistency
        if bounds is not None:
            if not validate_bounds_consistency(bounds, initial_params):
                self._validation_errors.append("Bounds are inconsistent with initial parameters")

        # Check initial params
        if not validate_initial_params(initial_params, bounds):
            self._validation_errors.append("Initial parameters outside bounds")

        if self._validation_errors:
            if self.strict_mode:
                raise ValueError(f"Input validation failed: {'; '.join(self._validation_errors)}")
            else:
                for error in self._validation_errors:
                    logger.warning(f"Input validation warning: {error}")
                return False

        return True

    @property
    def validation_errors(self) -> list[str]:
        """Get list of validation errors from last validate_all() call."""
        return self._validation_errors.copy()


def validate_array_dimensions(xdata: np.ndarray, ydata: np.ndarray) -> bool:
    """Validate that xdata and ydata have compatible dimensions.

    Parameters
    ----------
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data

    Returns
    -------
    bool
        True if dimensions are compatible
    """
    if len(xdata) == 0:
        logger.warning("xdata is empty")
        return False

    if len(ydata) == 0:
        logger.warning("ydata is empty")
        return False

    if len(xdata) != len(ydata):
        logger.warning(f"Array length mismatch: xdata={len(xdata)}, ydata={len(ydata)}")
        return False

    return True


def validate_no_nan_inf(arr: np.ndarray, name: str) -> bool:
    """Validate that array contains no NaN or Inf values.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate
    name : str
        Name for logging

    Returns
    -------
    bool
        True if array contains only finite values
    """
    if not np.all(np.isfinite(arr)):
        nan_count = np.sum(np.isnan(arr))
        inf_count = np.sum(np.isinf(arr))
        logger.warning(f"{name} contains {nan_count} NaN and {inf_count} Inf values")
        return False
    return True


def validate_bounds_consistency(
    bounds: tuple[np.ndarray, np.ndarray],
    initial_params: np.ndarray,
) -> bool:
    """Validate that bounds are consistent.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        (lower, upper) bounds arrays
    initial_params : np.ndarray
        Initial parameter values

    Returns
    -------
    bool
        True if bounds are consistent
    """
    lower, upper = bounds

    # Check bounds arrays have same length as params
    if len(lower) != len(initial_params):
        logger.warning(f"Lower bounds length {len(lower)} != params length {len(initial_params)}")
        return False
    if len(upper) != len(initial_params):
        logger.warning(f"Upper bounds length {len(upper)} != params length {len(initial_params)}")
        return False

    # Check lower <= upper
    if not np.all(lower <= upper):
        violations = np.where(lower > upper)[0]
        logger.warning(f"Lower > upper at indices: {violations}")
        return False

    return True


def validate_initial_params(
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
) -> bool:
    """Validate that initial parameters are within bounds.

    Parameters
    ----------
    initial_params : np.ndarray
        Initial parameter values
    bounds : tuple[np.ndarray, np.ndarray] | None
        (lower, upper) bounds arrays, or None for unbounded

    Returns
    -------
    bool
        True if params are within bounds
    """
    if bounds is None:
        return True

    lower, upper = bounds

    # Check within bounds
    below_lower = initial_params < lower
    above_upper = initial_params > upper

    if np.any(below_lower):
        indices = np.where(below_lower)[0]
        logger.warning(f"Params below lower bound at indices: {indices}")
        return False

    if np.any(above_upper):
        indices = np.where(above_upper)[0]
        logger.warning(f"Params above upper bound at indices: {indices}")
        return False

    return True


__all__ = [
    "InputValidator",
    "validate_array_dimensions",
    "validate_bounds_consistency",
    "validate_initial_params",
    "validate_no_nan_inf",
]
