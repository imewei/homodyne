"""Minimal configuration system for the homodyne package.

Provides essential configuration management with preserved API compatibility.
"""

from homodyne.config.manager import ConfigManager, load_xpcs_config

__all__ = [
    "ConfigManager",
    "load_xpcs_config",
]
