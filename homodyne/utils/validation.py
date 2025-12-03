"""Validation utilities for common array and parameter checks.

This module provides centralized validation functions to reduce code
duplication across the codebase.

Extracted as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_array_not_none(
    arr: Any,
    name: str,
    context: str | None = None,
) -> np.ndarray:
    """Validate array is not None and convert to numpy array.

    Parameters
    ----------
    arr : Any
        Input array or array-like.
    name : str
        Name of the array for error messages.
    context : str, optional
        Additional context for error messages.

    Returns
    -------
    np.ndarray
        Validated numpy array.

    Raises
    ------
    ValueError
        If arr is None.
    """
    if arr is None:
        ctx = f" for {context}" if context else ""
        raise ValueError(f"{name} cannot be None{ctx}")
    return np.asarray(arr)


def validate_array_not_empty(
    arr: Any,
    name: str,
    context: str | None = None,
    allow_none: bool = False,
) -> np.ndarray | None:
    """Validate array is not None and not empty.

    Parameters
    ----------
    arr : Any
        Input array or array-like.
    name : str
        Name of the array for error messages.
    context : str, optional
        Additional context for error messages.
    allow_none : bool
        If True, None values are allowed and returned as-is.

    Returns
    -------
    np.ndarray | None
        Validated numpy array or None if allow_none=True and arr is None.

    Raises
    ------
    ValueError
        If arr is None (when allow_none=False) or empty.
    """
    if arr is None:
        if allow_none:
            return None
        ctx = f" for {context}" if context else ""
        raise ValueError(f"{name} cannot be None{ctx}")

    arr = np.asarray(arr)
    if arr.size == 0:
        ctx = f" for {context}" if context else ""
        raise ValueError(f"{name} cannot be empty{ctx}")

    return arr


def validate_positive_scalar(
    value: float | int | None,
    name: str,
    context: str | None = None,
    allow_none: bool = False,
    allow_zero: bool = False,
) -> float | None:
    """Validate scalar is positive.

    Parameters
    ----------
    value : float | int | None
        Input scalar value.
    name : str
        Name of the parameter for error messages.
    context : str, optional
        Additional context for error messages.
    allow_none : bool
        If True, None values are allowed.
    allow_zero : bool
        If True, zero is allowed.

    Returns
    -------
    float | None
        Validated float value or None if allow_none=True.

    Raises
    ------
    ValueError
        If value is invalid.
    """
    if value is None:
        if allow_none:
            return None
        ctx = f" for {context}" if context else ""
        raise ValueError(f"{name} cannot be None{ctx}")

    value = float(value)
    ctx = f" for {context}" if context else ""

    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be non-negative{ctx}, got {value}")
    else:
        if value <= 0:
            raise ValueError(f"{name} must be positive{ctx}, got {value}")

    return value


def validate_in_bounds(
    value: float,
    lower: float,
    upper: float,
    name: str,
    context: str | None = None,
    inclusive: bool = True,
) -> float:
    """Validate value is within bounds.

    Parameters
    ----------
    value : float
        Value to validate.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    name : str
        Name of the parameter for error messages.
    context : str, optional
        Additional context for error messages.
    inclusive : bool
        If True, bounds are inclusive.

    Returns
    -------
    float
        Validated value.

    Raises
    ------
    ValueError
        If value is out of bounds.
    """
    ctx = f" for {context}" if context else ""

    if inclusive:
        if value < lower or value > upper:
            raise ValueError(
                f"{name} must be in [{lower}, {upper}]{ctx}, got {value}"
            )
    else:
        if value <= lower or value >= upper:
            raise ValueError(
                f"{name} must be in ({lower}, {upper}){ctx}, got {value}"
            )

    return value


def validate_array_shapes_match(
    arrays: dict[str, np.ndarray],
    context: str | None = None,
) -> None:
    """Validate multiple arrays have matching shapes.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Dictionary mapping array names to arrays.
    context : str, optional
        Additional context for error messages.

    Raises
    ------
    ValueError
        If array shapes don't match.
    """
    if len(arrays) < 2:
        return

    names = list(arrays.keys())
    shapes = [arr.shape for arr in arrays.values()]
    reference_shape = shapes[0]

    for name, shape in zip(names[1:], shapes[1:], strict=False):
        if shape != reference_shape:
            ctx = f" for {context}" if context else ""
            raise ValueError(
                f"Array shape mismatch{ctx}: "
                f"{names[0]} has shape {reference_shape}, "
                f"but {name} has shape {shape}"
            )


def validate_required_params(
    params: dict[str, Any],
    required: list[str],
    context: str | None = None,
) -> None:
    """Validate all required parameters are present and not None.

    Parameters
    ----------
    params : dict[str, Any]
        Dictionary of parameters.
    required : list[str]
        List of required parameter names.
    context : str, optional
        Additional context for error messages.

    Raises
    ------
    ValueError
        If any required parameter is missing or None.
    """
    ctx = f" for {context}" if context else ""
    missing = [name for name in required if params.get(name) is None]

    if missing:
        raise ValueError(
            f"Required parameters cannot be None{ctx}: {', '.join(missing)}"
        )
