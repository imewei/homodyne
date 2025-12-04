"""Minimal configuration system for the homodyne package.

Provides essential configuration management with preserved API compatibility.
"""

from homodyne.config.manager import ConfigManager, load_xpcs_config
from homodyne.config.parameter_registry import (
    ParameterInfo,
    ParameterRegistry,
    get_all_param_names,
    get_bounds,
    get_defaults,
    get_param_names,
    get_registry,
)
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

__all__ = [
    # Configuration management
    "ConfigManager",
    "load_xpcs_config",
    # Parameter registry (v2.4.1+)
    "ParameterInfo",
    "ParameterRegistry",
    "get_registry",
    "get_param_names",
    "get_all_param_names",
    "get_bounds",
    "get_defaults",
    # Parameter space
    "ParameterSpace",
    "PriorDistribution",
]
