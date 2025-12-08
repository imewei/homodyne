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
]
