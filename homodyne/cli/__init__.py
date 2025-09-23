"""
Command Line Interface for Homodyne v2
======================================

Simplified CLI for JAX-first homodyne scattering analysis.
Provides essential commands for optimization with minimal complexity.

Usage:
    homodyne --method nlsq --config config.yaml
    homodyne --method mcmc --data-file data.h5
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