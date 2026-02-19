"""Homodyne: JAX-First XPCS Analysis Package
========================================

A high-performance package for X-ray Photon Correlation Spectroscopy (XPCS) analysis
using JAX-accelerated optimization methods for HPC and supercomputer environments.

Key Features:
- JAX-first computational architecture (CPU-only v2.3.0+)
- NLSQ trust-region nonlinear least squares (primary optimization)
- NumPyro/BlackJAX MCMC sampling (secondary optimization, CMC-only)
- HPC-optimized for 36/128-core CPU nodes
- Preserved API compatibility with validated components

Core Equation:
    c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²

Analysis Modes:
- Static Isotropic: 3 parameters [D₀, α, D_offset]
- Laminar Flow: 7 parameters [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]

Quick Start:
    >>> from homodyne.optimization import fit_nlsq_jax
    >>> from homodyne.data import load_xpcs_data
    >>> from homodyne.config import ConfigManager
    >>>
    >>> # Load data and configuration
    >>> data = load_xpcs_data("config.yaml")
    >>> config = ConfigManager("config.yaml")
    >>>
    >>> # Run optimization
    >>> result = fit_nlsq_jax(data, config)
    >>> print(f"Parameters: {result.parameters}")

Authors: Wei Chen, Hongrui He (Argonne National Laboratory)
"""

# Standard library imports
import logging
import os
import sys
import warnings

# ============================================================================
# Third-Party Deprecation Warning Filters (MUST be set before imports)
# ============================================================================
# NumPyro 0.19.0 uses deprecated JAX 0.8.2+ internals (xla_pmap_p)
# Filter before any imports that might trigger NumPyro loading
# See: https://github.com/pyro-ppl/numpyro/releases (awaiting upstream fix)
warnings.filterwarnings(
    "ignore",
    message="jax.interpreters.pxla.xla_pmap_p is deprecated",
    category=DeprecationWarning,
)

# ============================================================================
# JAX CPU Device Configuration (MUST be set before JAX import)
# ============================================================================
# Configure JAX to use multiple CPU devices for parallel MCMC chains
# This MUST be set before JAX/XLA is initialized (import time)
# Default: 4 devices for parallel MCMC, can be overridden by user

# Default XLA flags for homodyne:
# - xla_force_host_platform_device_count=4: Enable parallel MCMC chains
# - xla_disable_hlo_passes=constant_folding: Prevent slow compilation warnings
#   for large datasets (23M+ points) where data arrays are captured in JIT
#   closures (e.g., CMA-ES fitness functions). This avoids XLA's
#   slow_operation_alarm (> 1s) during gather operations on large constants.
#   Performance impact: minimal (< 5ms difference per call). (v2.17.0+)
_DEFAULT_XLA_FLAGS = [
    "--xla_force_host_platform_device_count=4",
    "--xla_disable_hlo_passes=constant_folding",
]

# P2-A: Set JAX_ENABLE_X64 explicitly rather than relying on nlsq import side-effect.
# JAX must be in float64 for parameters spanning 6+ orders of magnitude (D0~1e4, gamma~1e-3).
# This env var must be set BEFORE the first JAX import (any import that triggers jax init).
os.environ.setdefault("JAX_ENABLE_X64", "1")

if "XLA_FLAGS" not in os.environ:
    # No existing XLA_FLAGS, set defaults
    os.environ["XLA_FLAGS"] = " ".join(_DEFAULT_XLA_FLAGS)
else:
    # XLA_FLAGS exists, append any missing default flags
    existing = os.environ["XLA_FLAGS"]
    flags_to_add = []
    for flag in _DEFAULT_XLA_FLAGS:
        flag_name = flag.split("=")[0]
        if flag_name not in existing:
            flags_to_add.append(flag)
    if flags_to_add:
        os.environ["XLA_FLAGS"] += " " + " ".join(flags_to_add)

# Suppress NLSQ GPU warnings (v2.3.0 is CPU-only)
os.environ.setdefault("NLSQ_SKIP_GPU_CHECK", "1")

# Suppress JAX backend warnings and messages (CPU-only in v2.3.0)
# - TPU backend warnings (not available on standard systems)
# - GPU fallback warnings (expected behavior for CPU-only installation)
# - Backend initialization INFO messages
# IMPORTANT: Don't set JAX_PLATFORMS - let JAX auto-detect available backend

# Suppress JAX backend logs (set to ERROR to hide GPU fallback warnings)
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

# Version handling
try:
    from homodyne._version import __version__
except ImportError:
    __version__ = "2.22.1"

# ---------------------------------------------------------------------------
# Lazy public API
# ---------------------------------------------------------------------------
# Submodules (homodyne.data, .core, .optimization, .config, .device, .cli)
# import JAX, NumPyro, ArviZ, h5py, and other heavy dependencies.  Loading
# them eagerly adds 3-6 s to *every* `import homodyne` — including CLI
# invocations that only need argument parsing.
#
# The __getattr__ hook below delays each import until the attribute is first
# accessed (e.g. `homodyne.fit_nlsq_jax`).  Subsequent accesses resolve
# directly via the module __dict__ at zero cost.
#
# The _LAZY_IMPORTS dict maps exported names to (submodule, attr) pairs.
# _MODULE_SENTINELS maps each module flag (HAS_DATA etc.) to its submodule
# path so that `HAS_*` reflects real availability when first queried.

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # data
    "XPCSDataLoader":           ("homodyne.data",         "XPCSDataLoader"),
    "load_xpcs_data":           ("homodyne.data",         "load_xpcs_data"),
    # core
    "ParameterSpace":           ("homodyne.core",         "ParameterSpace"),
    "ScaledFittingEngine":      ("homodyne.core",         "ScaledFittingEngine"),
    "TheoryEngine":             ("homodyne.core",         "TheoryEngine"),
    "compute_g2_scaled":        ("homodyne.core",         "compute_g2_scaled"),
    # optimization
    "fit_mcmc_jax":             ("homodyne.optimization", "fit_mcmc_jax"),
    "fit_nlsq_jax":             ("homodyne.optimization", "fit_nlsq_jax"),
    "get_optimization_info":    ("homodyne.optimization", "get_optimization_info"),
    # config
    "ConfigManager":            ("homodyne.config",       "ConfigManager"),
    # device
    "configure_optimal_device": ("homodyne.device",       "configure_optimal_device"),
    "get_device_status":        ("homodyne.device",       "get_device_status"),
    # cli
    "cli_main":                 ("homodyne.cli",          "main"),
}

# Sentinel names that trigger module-level HAS_* evaluation on first access.
_MODULE_FLAGS: dict[str, tuple[str, str]] = {
    "HAS_DATA":         ("homodyne.data",         "XPCSDataLoader"),
    "HAS_CORE":         ("homodyne.core",         "ParameterSpace"),
    "HAS_OPTIMIZATION": ("homodyne.optimization", "fit_nlsq_jax"),
    "HAS_CONFIG":       ("homodyne.config",       "ConfigManager"),
    "HAS_DEVICE":       ("homodyne.device",       "configure_optimal_device"),
    "HAS_CLI":          ("homodyne.cli",          "main"),
}


def __getattr__(name: str) -> object:
    """Lazy-load public API symbols and HAS_* flags on first access."""
    import importlib

    # Lazy submodule attribute
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        try:
            mod = importlib.import_module(module_path)
            value = getattr(mod, attr)
        except (ImportError, AttributeError):
            raise AttributeError(
                f"homodyne.{name} is not available "
                f"(could not import {module_path}.{attr})"
            ) from None
        globals()[name] = value  # cache in module dict for future access
        return value

    # Lazy HAS_* flags
    if name in _MODULE_FLAGS:
        module_path, attr = _MODULE_FLAGS[name]
        try:
            mod = importlib.import_module(module_path)
            getattr(mod, attr)  # ensure the symbol actually exists
            value: bool = True
        except (ImportError, AttributeError):
            value = False
        globals()[name] = value
        return value

    raise AttributeError(f"module 'homodyne' has no attribute {name!r}")


# __features__ is populated lazily via get_package_info() or direct access.
# It is NOT pre-populated here to avoid triggering eager imports.

# Main exports — include all known names; unavailable ones raise AttributeError
# at access time with a helpful message.
__all__ = [
    "__version__",
    "get_package_info",
    # data
    "XPCSDataLoader", "load_xpcs_data",
    # core
    "ParameterSpace", "ScaledFittingEngine", "TheoryEngine", "compute_g2_scaled",
    # optimization
    "fit_nlsq_jax", "fit_mcmc_jax", "get_optimization_info",
    # config
    "ConfigManager",
    # device
    "configure_optimal_device", "get_device_status",
    # cli
    "cli_main",
    # flags (resolved lazily)
    "HAS_DATA", "HAS_CORE", "HAS_OPTIMIZATION", "HAS_CONFIG", "HAS_DEVICE", "HAS_CLI",
]


def get_package_info() -> dict:
    """Get comprehensive package information and status.

    Returns
    -------
    dict
        Package information including version, features, and dependencies
    """
    import importlib

    # Resolve HAS_* lazily (triggers __getattr__ if not yet cached)
    import homodyne as _self
    has_data = getattr(_self, "HAS_DATA", False)
    has_core = getattr(_self, "HAS_CORE", False)
    has_optimization = getattr(_self, "HAS_OPTIMIZATION", False)
    has_config = getattr(_self, "HAS_CONFIG", False)
    has_device = getattr(_self, "HAS_DEVICE", False)
    has_cli = getattr(_self, "HAS_CLI", False)

    # Check JAX availability
    try:
        importlib.import_module("jax")
        jax_available = True
    except ImportError:
        jax_available = False

    features = {
        "data_loading": has_data,
        "core_physics": has_core,
        "optimization": has_optimization,
        "configuration": has_config,
        "device_optimization": has_device,
        "cli_interface": has_cli,
        "jax_acceleration": jax_available,
    }

    info = {
        "version": __version__,
        "features": features,
        "modules_available": {
            "data": has_data,
            "core": has_core,
            "optimization": has_optimization,
            "config": has_config,
            "device": has_device,
            "cli": has_cli,
        },
        "dependencies": {},
        "recommendations": [],
    }

    # Check key dependencies
    dependencies = {
        "jax": False,
        "nlsq": False,
        "numpyro": False,
        "blackjax": False,
        "h5py": False,
        "yaml": False,
        "numpy": False,
        "scipy": False,
    }

    for dep in dependencies:
        try:
            importlib.import_module(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    info["dependencies"] = dependencies

    # Generate recommendations
    recommendations: list[str] = []
    if not dependencies["jax"]:
        recommendations.append("Install JAX for acceleration: pip install jax")

    if not dependencies["nlsq"]:
        recommendations.append(
            "Install NLSQ for optimization: pip install nlsq",
        )

    if not dependencies["numpyro"] and not dependencies["blackjax"]:
        recommendations.append(
            "Install NumPyro or BlackJAX for MCMC: pip install numpyro blackjax",
        )

    if not dependencies["h5py"]:
        recommendations.append("Install h5py for HDF5 data: pip install h5py")

    # System recommendations (CPU-only in v2.3.0)
    if jax_available:
        recommendations.append(
            "CPU acceleration configured - good performance expected",
        )
    else:
        recommendations.append("Install JAX for optimal performance")

    info["recommendations"] = recommendations

    return info


def _check_installation() -> None:
    """Check installation status and provide helpful messages (CPU-only in v2.3.0)."""
    import homodyne as _self
    if not getattr(_self, "HAS_OPTIMIZATION", False):
        print(
            "Warning: Optimization modules not available. Core functionality limited.",
        )
        print("Install with: pip install homodyne[all]")

    try:
        import importlib
        importlib.import_module("jax")
    except ImportError:
        print("Note: JAX not available. Install with: pip install jax")


# Run installation check on import (only in interactive mode)
if hasattr(sys, "ps1"):  # Interactive mode
    _check_installation()
