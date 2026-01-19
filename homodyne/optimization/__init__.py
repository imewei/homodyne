"""JAX-First Optimization for Homodyne.4
==========================================

Simplified optimization system using NLSQ package (primary) and CMC
(high-accuracy Bayesian) for robust parameter estimation in homodyne analysis.

This module implements the streamlined optimization philosophy:
1. NLSQ as primary method (fast, reliable parameter estimation)
2. CMC (NumPyro/NUTS) for uncertainty quantification
3. Unified homodyne model: c2_fitted = c2_theory * contrast + offset

Key Features:
- NLSQ trust-region optimization (Levenberg-Marquardt) as foundation
- CMC: Fresh reimplementation with ArviZ-native output
- CPU-primary architecture (GPU removed in v2.3.0)
- Dataset size-aware optimization strategies

Performance Comparison:
- NLSQ: Fast, reliable parameter estimation
- CMC: Full posterior sampling, publication-quality uncertainty

Note: Legacy mcmc/ package removed in v3.0. CMC is the sole MCMC backend.
"""

from __future__ import annotations

from typing import Any

# Import submodules as attributes for hasattr() checks
# These imports expose the submodule packages even if their contents fail to import
from homodyne.optimization import nlsq

# Handle NLSQ imports with intelligent fallback
try:
    from homodyne.optimization.nlsq import (  # Chunking; Residual; Sequential
        MultiStartConfig,
        MultiStartResult,
        NLSQResult,
        NLSQWrapper,
        OptimizationResult,
        StratificationDiagnostics,
        StratifiedResidualFunction,
        StratifiedResidualFunctionJIT,
        create_angle_stratified_data,
        create_angle_stratified_indices,
        create_stratified_residual_function,
        fit_nlsq_jax,
        fit_nlsq_multistart,
        optimize_per_angle_sequential,
        should_use_stratification,
    )
    # NOTE: DatasetSizeStrategy, OptimizationStrategy, estimate_memory_requirements
    # removed from public API in v2.12.0. Use NLSQ's WorkflowSelector instead.

    NLSQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NLSQ optimization: {e}")
    fit_nlsq_jax = None  # type: ignore[assignment]
    fit_nlsq_multistart = None  # type: ignore[assignment]
    MultiStartConfig = None  # type: ignore[assignment,misc]
    MultiStartResult = None  # type: ignore[assignment,misc]
    NLSQResult = None  # type: ignore[assignment,misc]
    NLSQWrapper = None  # type: ignore[assignment,misc]
    OptimizationResult = None  # type: ignore[assignment,misc]
    StratificationDiagnostics = None  # type: ignore[assignment,misc]
    create_angle_stratified_data = None  # type: ignore[assignment]
    create_angle_stratified_indices = None  # type: ignore[assignment]
    should_use_stratification = None  # type: ignore[assignment]
    StratifiedResidualFunction = None  # type: ignore[assignment,misc]
    StratifiedResidualFunctionJIT = None  # type: ignore[assignment,misc]
    create_stratified_residual_function = None  # type: ignore[assignment]
    optimize_per_angle_sequential = None  # type: ignore[assignment]
    NLSQ_AVAILABLE = False

# Handle CMC imports (NO FALLBACK to legacy mcmc - it's removed)
# Try to import the cmc module for hasattr() checks
try:
    from homodyne.optimization import cmc

    CMC_SUBMODULE_AVAILABLE = True
except ImportError:
    cmc = None  # type: ignore[assignment]
    CMC_SUBMODULE_AVAILABLE = False

try:
    from homodyne.optimization.cmc import (
        CMCConfig,
        CMCResult,
        fit_mcmc_jax,
    )

    # Aliases for backward compatibility
    MCMCResult = CMCResult

    # CMC uses NumPyro/JAX
    MCMC_JAX_AVAILABLE = True
    NUMPYRO_AVAILABLE = True
    BLACKJAX_AVAILABLE = True
    MCMC_AVAILABLE = True

except ImportError as e:
    print(f"Warning: Could not import CMC optimization: {e}")
    fit_mcmc_jax = None  # type: ignore[assignment]
    CMCConfig = None  # type: ignore[assignment,misc]
    CMCResult = None  # type: ignore[assignment,misc]
    MCMCResult = None  # type: ignore[misc,assignment]
    MCMC_JAX_AVAILABLE = False
    NUMPYRO_AVAILABLE = False
    BLACKJAX_AVAILABLE = False
    MCMC_AVAILABLE = False

# Module status
OPTIMIZATION_STATUS = {
    "nlsq_available": NLSQ_AVAILABLE,
    "mcmc_available": MCMC_AVAILABLE,
    "cmc_available": MCMC_AVAILABLE,  # CMC is the MCMC backend
    "jax_available": MCMC_JAX_AVAILABLE if MCMC_AVAILABLE else False,
    "numpyro_available": NUMPYRO_AVAILABLE if MCMC_AVAILABLE else False,
    "blackjax_available": BLACKJAX_AVAILABLE if MCMC_AVAILABLE else False,
}

# Primary API functions
__all__ = [
    # Primary optimization methods
    "fit_nlsq_jax",  # NLSQ trust-region (PRIMARY)
    "fit_nlsq_multistart",  # Multi-start NLSQ (v2.6.0)
    "fit_mcmc_jax",  # CMC NumPyro/NUTS (SECONDARY)
    # Result classes
    "NLSQResult",
    "MultiStartConfig",
    "MultiStartResult",
    "CMCResult",  # CMC result class
    "MCMCResult",  # Alias for backward compatibility
    "CMCConfig",  # CMC configuration
    # NLSQ components
    "NLSQWrapper",
    "OptimizationResult",
    "StratificationDiagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "should_use_stratification",
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    "optimize_per_angle_sequential",
    # NOTE: DatasetSizeStrategy, OptimizationStrategy, estimate_memory_requirements
    # removed from public API in v2.12.0. Use NLSQ's WorkflowSelector instead.
    # Status information
    "OPTIMIZATION_STATUS",
    "NLSQ_AVAILABLE",
    "MCMC_AVAILABLE",
    # Submodules
    "nlsq",
    "cmc",
]


def get_optimization_info() -> dict[str, Any]:
    """Get information about available optimization methods.

    Returns
    -------
    dict
        Dictionary with availability status and recommendations
    """
    info: dict[str, Any] = {
        "status": OPTIMIZATION_STATUS.copy(),
        "primary_method": "nlsq" if NLSQ_AVAILABLE else None,
        "secondary_method": "cmc" if MCMC_AVAILABLE else None,
        "recommendations": [],
    }

    if NLSQ_AVAILABLE:
        info["recommendations"].append(
            "Use fit_nlsq_jax() for fast, reliable parameter estimation",
        )

    if MCMC_AVAILABLE:
        info["recommendations"].append(
            "Use fit_mcmc_jax() for uncertainty quantification (CMC)",
        )

    if not NLSQ_AVAILABLE and not MCMC_AVAILABLE:
        info["recommendations"].append(
            "Install NLSQ and NumPyro for optimization capabilities",
        )

    return info
