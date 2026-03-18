"""Runtime utilities for homodyne package.

This module provides:
- System validation utilities
- Shell completion scripts
- XLA activation scripts

Example:
    >>> from homodyne.runtime import run_validation
    >>> run_validation()
"""

from homodyne.runtime.shell import (
    get_completion_script,
    get_xla_config_script,
)
from homodyne.runtime.utils import (
    SystemValidator,
    ValidationResult,
    run_validation,
)

__all__ = [
    # Validation
    "SystemValidator",
    "ValidationResult",
    "run_validation",
    # Shell
    "get_completion_script",
    "get_xla_config_script",
]
