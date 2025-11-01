"""Minimal configuration system for the homodyne package.

Provides essential configuration management with preserved API compatibility.
"""

from homodyne.config.manager import ConfigManager, load_xpcs_config
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

__all__ = [
    "ConfigManager",
    "load_xpcs_config",
    "ParameterSpace",
    "PriorDistribution",
]
