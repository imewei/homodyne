"""Shared configuration utilities for NLSQ optimization.

.. deprecated:: 2.14.0
    This module is deprecated. Use ``homodyne.optimization.nlsq.config`` instead:
    - safe_float, safe_int, safe_bool are now in config.py

This module provides reusable helper functions for safe type conversion
in configuration classes, eliminating duplication across:
- gradient_monitor.py
- hierarchical.py
- adaptive_regularization.py
- (and any future config classes)

Part of Architecture Refactoring v2.9.1.
"""

from __future__ import annotations

import warnings

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Issue deprecation warning on module import
warnings.warn(
    "homodyne.optimization.nlsq.config_utils is deprecated since v2.14.0. "
    "Use homodyne.optimization.nlsq.config (safe_float, safe_int, safe_bool) instead.",
    DeprecationWarning,
    stacklevel=2,
)


def safe_float(value, default: float) -> float:
    """Convert value to float safely, returning default on failure.

    Parameters
    ----------
    value : Any
        Value to convert to float.
    default : float
        Default value to return if conversion fails.

    Returns
    -------
    float
        Converted float value or default.

    Examples
    --------
    >>> safe_float("3.14", 0.0)
    3.14
    >>> safe_float(None, 1.0)
    1.0
    >>> safe_float("invalid", 2.5)
    2.5
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to float, using default {default}")
        return default


def safe_int(value, default: int) -> int:
    """Convert value to int safely, returning default on failure.

    Parameters
    ----------
    value : Any
        Value to convert to int.
    default : int
        Default value to return if conversion fails.

    Returns
    -------
    int
        Converted int value or default.

    Examples
    --------
    >>> safe_int("42", 0)
    42
    >>> safe_int(None, 10)
    10
    >>> safe_int("invalid", 5)
    5
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to int, using default {default}")
        return default


def safe_bool(value, default: bool) -> bool:
    """Convert value to bool safely, returning default on failure.

    Handles string values like "true", "false", "1", "0".

    Parameters
    ----------
    value : Any
        Value to convert to bool.
    default : bool
        Default value to return if conversion fails.

    Returns
    -------
    bool
        Converted bool value or default.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "1", "yes", "on"):
            return True
        if lower in ("false", "0", "no", "off"):
            return False
    try:
        return bool(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to bool, using default {default}")
        return default
