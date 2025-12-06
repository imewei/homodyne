"""JAX-First Optimization for Homodyne v2.4
==========================================

Simplified optimization system using NLSQ package (primary) and CMC v3.0
(high-accuracy Bayesian) for robust parameter estimation in homodyne analysis.

This module implements the streamlined optimization philosophy:
1. NLSQ as primary method (fast, reliable parameter estimation)
2. CMC v3.0 (NumPyro/NUTS) for uncertainty quantification
3. Unified homodyne model: c2_fitted = c2_theory * contrast + offset

Key Features:
- NLSQ trust-region optimization (Levenberg-Marquardt) as foundation
- CMC v3.0: Fresh reimplementation with ArviZ-native output
- CPU-primary architecture (GPU removed in v2.3.0)
- Dataset size-aware optimization strategies

Performance Comparison:
- NLSQ: Fast, reliable parameter estimation
- CMC: Full posterior sampling, publication-quality uncertainty

Note: Legacy mcmc/ package removed in v3.0. CMC v3.0 is the sole MCMC backend.
"""

# Handle NLSQ imports with intelligent fallback
try:
    from homodyne.optimization.nlsq import (
        NLSQResult,
        NLSQWrapper,
        OptimizationResult,
        fit_nlsq_jax,
        # Strategies
        DatasetSizeStrategy,
        OptimizationStrategy,
        estimate_memory_requirements,
        # Chunking
        StratificationDiagnostics,
        create_angle_stratified_data,
        create_angle_stratified_indices,
        should_use_stratification,
        # Residual
        StratifiedResidualFunction,
        StratifiedResidualFunctionJIT,
        create_stratified_residual_function,
        # Sequential
        optimize_per_angle_sequential,
    )

    NLSQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NLSQ optimization: {e}")
    fit_nlsq_jax = None
    NLSQResult = None
    NLSQWrapper = None
    OptimizationResult = None
    DatasetSizeStrategy = None
    OptimizationStrategy = None
    estimate_memory_requirements = None
    StratificationDiagnostics = None
    create_angle_stratified_data = None
    create_angle_stratified_indices = None
    should_use_stratification = None
    StratifiedResidualFunction = None
    StratifiedResidualFunctionJIT = None
    create_stratified_residual_function = None
    optimize_per_angle_sequential = None
    NLSQ_AVAILABLE = False

# Handle CMC v3.0 imports (NO FALLBACK to legacy mcmc - it's removed)
try:
    from homodyne.optimization.cmc import (
        CMCConfig,
        CMCResult,
        fit_mcmc_jax,
    )

    # Aliases for backward compatibility
    MCMCResult = CMCResult

    # CMC v3.0 uses NumPyro/JAX
    MCMC_JAX_AVAILABLE = True
    NUMPYRO_AVAILABLE = True
    BLACKJAX_AVAILABLE = True
    MCMC_AVAILABLE = True

except ImportError as e:
    print(f"Warning: Could not import CMC optimization: {e}")
    fit_mcmc_jax = None
    CMCConfig = None
    CMCResult = None
    MCMCResult = None
    MCMC_JAX_AVAILABLE = False
    NUMPYRO_AVAILABLE = False
    BLACKJAX_AVAILABLE = False
    MCMC_AVAILABLE = False

# Module status
OPTIMIZATION_STATUS = {
    "nlsq_available": NLSQ_AVAILABLE,
    "mcmc_available": MCMC_AVAILABLE,
    "cmc_available": MCMC_AVAILABLE,  # CMC v3.0 is the MCMC backend
    "jax_available": MCMC_JAX_AVAILABLE if MCMC_AVAILABLE else False,
    "numpyro_available": NUMPYRO_AVAILABLE if MCMC_AVAILABLE else False,
    "blackjax_available": BLACKJAX_AVAILABLE if MCMC_AVAILABLE else False,
}

# Primary API functions
__all__ = [
    # Primary optimization methods
    "fit_nlsq_jax",  # NLSQ trust-region (PRIMARY)
    "fit_mcmc_jax",  # CMC v3.0 NumPyro/NUTS (SECONDARY)
    # Result classes
    "NLSQResult",
    "CMCResult",  # CMC v3.0 result class
    "MCMCResult",  # Alias for backward compatibility
    "CMCConfig",  # CMC configuration
    # NLSQ components
    "NLSQWrapper",
    "OptimizationResult",
    "DatasetSizeStrategy",
    "OptimizationStrategy",
    "estimate_memory_requirements",
    "StratificationDiagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "should_use_stratification",
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    "optimize_per_angle_sequential",
    # Status information
    "OPTIMIZATION_STATUS",
    "NLSQ_AVAILABLE",
    "MCMC_AVAILABLE",
]


def get_optimization_info():
    """Get information about available optimization methods.

    Returns
    -------
    dict
        Dictionary with availability status and recommendations
    """
    info = {
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
            "Use fit_mcmc_jax() for uncertainty quantification (CMC v3.0)",
        )

    if not NLSQ_AVAILABLE and not MCMC_AVAILABLE:
        info["recommendations"].append(
            "Install NLSQ and NumPyro for optimization capabilities",
        )

    return info
