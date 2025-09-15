"""
Utilities for Homodyne v2
=========================

Enhanced logging system and mathematical utilities for the JAX-based backend.

This module provides comprehensive logging functionality and enhanced utilities for
the v2 architecture.
"""

from homodyne.utils.debug import (CallTracker, PerformanceProfiler,
                                  debug_calls, debug_context)
# Logging system
from homodyne.utils.logging import (LoggerManager, configure_logging_from_json,
                                    get_logger, log_calls, log_operation,
                                    log_performance, validate_logging_config)

# Additional utilities for JAX backend (to be implemented)
# from homodyne.math import (
#     safe_jnp_functions,
#     numerical_stability,
#     parameter_transformations,
# )

# from homodyne.io import (
#     save_results_v2,
#     load_results_v2,
#     export_results_format,
# )

__all__ = [
    # Logging system
    "get_logger",
    "log_performance",
    "log_calls",
    "log_operation",
    "configure_logging_from_json",
    "validate_logging_config",
    "LoggerManager",
    # Debug utilities
    "debug_calls",
    "debug_context",
    "CallTracker",
    "PerformanceProfiler",
    # Math utilities (to be implemented)
    # "safe_jnp_functions",
    # "numerical_stability",
    # "parameter_transformations",
    # I/O utilities (to be implemented)
    # "save_results_v2",
    # "load_results_v2",
    # "export_results_format",
]
