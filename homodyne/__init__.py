"""Homodyne: JAX-First XPCS Analysis Package
========================================

A high-performance package for X-ray Photon Correlation Spectroscopy (XPCS) analysis
using JAX-accelerated optimization methods for HPC and supercomputer environments.

Key Features:
- JAX-first computational architecture
- NLSQ trust-region nonlinear least squares (primary optimization)
- NumPyro/BlackJAX MCMC sampling (secondary optimization)
- System CUDA integration with jax[local]
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

# Suppress JAX TPU backend warnings before any JAX imports
# TPU is not available on standard systems and creates log noise
# IMPORTANT: Don't set JAX_PLATFORMS here - it forces backend selection order
# Instead, let JAX auto-detect optimal backend (GPU > CPU)

# Suppress JAX backend initialization INFO messages
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)

# Version handling
try:
    from homodyne._version import version as __version__
except ImportError:
    __version__ = "2.0.0-dev"

# Core imports with graceful fallback
try:
    from homodyne.data import XPCSDataLoader, load_xpcs_data

    HAS_DATA = True
except ImportError:
    HAS_DATA = False

try:
    from homodyne.core import (
        ParameterSpace,
        ScaledFittingEngine,
        TheoryEngine,
        compute_g2_scaled,
    )

    HAS_CORE = True
except ImportError:
    HAS_CORE = False

try:
    from homodyne.optimization import fit_mcmc_jax, fit_nlsq_jax, get_optimization_info

    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

try:
    from homodyne.config import ConfigManager

    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

try:
    from homodyne.device import configure_optimal_device, get_device_status

    HAS_DEVICE = True
except ImportError:
    HAS_DEVICE = False

try:
    from homodyne.cli import main as cli_main

    HAS_CLI = True
except ImportError:
    HAS_CLI = False

# Package feature information
__features__ = {
    "data_loading": HAS_DATA,
    "core_physics": HAS_CORE,
    "optimization": HAS_OPTIMIZATION,
    "configuration": HAS_CONFIG,
    "device_optimization": HAS_DEVICE,
    "cli_interface": HAS_CLI,
    "jax_acceleration": None,  # Will be set below
    "gpu_acceleration": None,  # Will be set below
}

# Check JAX availability
try:
    import jax

    __features__["jax_acceleration"] = True
    try:
        gpu_devices = jax.devices("gpu")
        __features__["gpu_acceleration"] = len(gpu_devices) > 0
    except (RuntimeError, AttributeError, ValueError):
        __features__["gpu_acceleration"] = False
except ImportError:
    __features__["jax_acceleration"] = False
    __features__["gpu_acceleration"] = False

# Main exports (only available components)
__all__ = ["__version__", "__features__", "get_package_info"]

# Add available components to exports
if HAS_DATA:
    __all__.extend(["load_xpcs_data", "XPCSDataLoader"])

if HAS_CORE:
    __all__.extend(
        ["compute_g2_scaled", "TheoryEngine", "ScaledFittingEngine", "ParameterSpace"],
    )

if HAS_OPTIMIZATION:
    __all__.extend(["fit_nlsq_jax", "fit_mcmc_jax", "get_optimization_info"])

if HAS_CONFIG:
    __all__.extend(["ConfigManager"])

if HAS_DEVICE:
    __all__.extend(["configure_optimal_device", "get_device_status"])

if HAS_CLI:
    __all__.extend(["cli_main"])


def get_package_info() -> dict:
    """Get comprehensive package information and status.

    Returns
    -------
    dict
        Package information including version, features, and dependencies
    """
    info = {
        "version": __version__,
        "features": __features__.copy(),
        "modules_available": {
            "data": HAS_DATA,
            "core": HAS_CORE,
            "optimization": HAS_OPTIMIZATION,
            "config": HAS_CONFIG,
            "device": HAS_DEVICE,
            "cli": HAS_CLI,
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
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False

    info["dependencies"] = dependencies

    # Generate recommendations
    if not dependencies["jax"]:
        info["recommendations"].append("Install JAX for acceleration: pip install jax")

    if not dependencies["nlsq"]:
        info["recommendations"].append(
            "Install NLSQ for optimization: pip install nlsq",
        )

    if not dependencies["numpyro"] and not dependencies["blackjax"]:
        info["recommendations"].append(
            "Install NumPyro or BlackJAX for MCMC: pip install numpyro blackjax",
        )

    if __features__["jax_acceleration"] and not __features__["gpu_acceleration"]:
        info["recommendations"].append(
            "For GPU acceleration: pip install jax[local] (uses system CUDA)",
        )

    if not dependencies["h5py"]:
        info["recommendations"].append("Install h5py for HDF5 data: pip install h5py")

    # System recommendations
    if __features__["jax_acceleration"]:
        if __features__["gpu_acceleration"]:
            info["recommendations"].append(
                "✓ GPU acceleration available - excellent performance expected",
            )
        else:
            info["recommendations"].append(
                "CPU acceleration configured - good performance expected",
            )
    else:
        info["recommendations"].append("Install JAX for optimal performance")

    return info


def _check_installation():
    """Check installation status and provide helpful messages."""
    if not HAS_OPTIMIZATION:
        print(
            "Warning: Optimization modules not available. Core functionality limited.",
        )
        print("Install with: pip install homodyne[all]")

    if not __features__["jax_acceleration"]:
        print("Note: JAX not available. Install with: pip install jax")

    if __features__["jax_acceleration"] and not __features__["gpu_acceleration"]:
        print("Note: For GPU acceleration, install: pip install jax[local]")


# Run installation check on import (only in interactive mode)
if hasattr(sys, "ps1"):  # Interactive mode
    _check_installation()
