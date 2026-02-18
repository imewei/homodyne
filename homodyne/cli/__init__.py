"""Command Line Interface for Homodyne
======================================

Simplified CLI for JAX-first homodyne scattering analysis.
Provides essential commands for optimization with minimal complexity.

Usage:
    homodyne --method nlsq --config config.yaml
    homodyne --method cmc --config config.yaml --nlsq-result results/
"""

from homodyne.cli.args_parser import create_parser, validate_args
from homodyne.cli.commands import dispatch_command
from homodyne.cli.main import main

__all__ = [
    "main",
    "create_parser",
    "dispatch_command",
    "validate_args",
]
