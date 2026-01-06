"""Validation utilities for NLSQ optimization.

This subpackage contains validation logic extracted from wrapper.py
as part of architecture refactoring (FR-011).
"""

from homodyne.optimization.nlsq.validation.input_validator import (
    InputValidator,
    validate_array_dimensions,
    validate_bounds_consistency,
    validate_initial_params,
    validate_no_nan_inf,
)
from homodyne.optimization.nlsq.validation.result_validator import (
    ResultValidator,
    validate_covariance,
    validate_optimized_params,
    validate_result_consistency,
)

__all__ = [
    # Input validation
    "InputValidator",
    "validate_array_dimensions",
    "validate_bounds_consistency",
    "validate_initial_params",
    "validate_no_nan_inf",
    # Result validation
    "ResultValidator",
    "validate_covariance",
    "validate_optimized_params",
    "validate_result_consistency",
]
