"""
Command Line Interface for Homodyne v2
======================================

JAX-accelerated XPCS analysis with modern CLI design.
Provides enhanced performance with modern CLI features.

Main Entry Points:
- main.py: Primary CLI entry point
- args_parser.py: Argument parsing and help
- commands.py: Command routing and dispatch
- validators.py: Input validation and error handling

Usage:
    homodyne [options]
    python -m homodyne [options]
"""

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.cli.main import main
from homodyne.cli.validators import ValidationError, validate_args

__all__ = [
    "main",
    "create_parser",
    "dispatch_command",
    "validate_args",
    "ValidationError",
]
