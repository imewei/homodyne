"""Minimal utilities for the homodyne package.

Essential utility functions with preserved API compatibility.
"""

from homodyne.utils.logging import (
    configure_logging,
    get_logger,
    log_calls,
    log_operation,
    log_performance,
    with_context,
)
from homodyne.utils.path_validation import (
    PathValidationError,
    get_safe_output_dir,
    validate_plot_save_path,
    validate_save_path,
)
from homodyne.utils.validation import (
    validate_array_not_empty,
    validate_array_not_none,
    validate_array_shapes_match,
    validate_in_bounds,
    validate_positive_scalar,
    validate_required_params,
)

__all__ = [
    # Logging utilities
    "get_logger",
    "configure_logging",
    "with_context",
    "log_performance",
    "log_calls",
    "log_operation",
    # Path validation utilities
    "PathValidationError",
    "validate_save_path",
    "validate_plot_save_path",
    "get_safe_output_dir",
    # Array/param validation utilities
    "validate_array_not_none",
    "validate_array_not_empty",
    "validate_positive_scalar",
    "validate_in_bounds",
    "validate_array_shapes_match",
    "validate_required_params",
]
