"""JAX-First Optimization for Homodyne v2
======================================

Simplified optimization system using NLSQ package (primary) and MCMC+JAX
(high-accuracy) for robust parameter estimation in homodyne analysis.

This module implements the streamlined optimization philosophy:
1. NLSQ as primary method (fast, reliable parameter estimation)
2. MCMC+JAX (NumPyro/BlackJAX) for uncertainty quantification
3. Unified homodyne model: c2_fitted = c2_theory * contrast + offset

Key Features:
- NLSQ trust-region optimization (Levenberg-Marquardt) as foundation
- JAX-accelerated MCMC for uncertainty quantification
- CPU-primary, GPU-optional architecture
- Dataset size-aware optimization strategies

Performance Comparison:
- NLSQ: Fast, reliable parameter estimation
- MCMC+JAX: Full posterior sampling, highest accuracy
"""

# Handle NLSQ imports with intelligent fallback
try:
    from homodyne.optimization.nlsq import NLSQResult, fit_nlsq_jax

    NLSQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import NLSQ optimization: {e}")
    fit_nlsq_jax = None
    NLSQResult = None
    NLSQ_AVAILABLE = False

# Handle MCMC imports with intelligent fallback
try:
    from homodyne.optimization.mcmc import (
        BLACKJAX_AVAILABLE,
        NUMPYRO_AVAILABLE,
        MCMCResult,
        fit_mcmc_jax,
    )
    from homodyne.optimization.mcmc import JAX_AVAILABLE as MCMC_JAX_AVAILABLE

    MCMC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import MCMC optimization: {e}")
    fit_mcmc_jax = None
    MCMCResult = None
    MCMC_JAX_AVAILABLE = False
    NUMPYRO_AVAILABLE = False
    BLACKJAX_AVAILABLE = False
    MCMC_AVAILABLE = False

# Module status
OPTIMIZATION_STATUS = {
    "nlsq_available": NLSQ_AVAILABLE,
    "mcmc_available": MCMC_AVAILABLE,
    "jax_available": MCMC_JAX_AVAILABLE if MCMC_AVAILABLE else False,
    "numpyro_available": NUMPYRO_AVAILABLE if MCMC_AVAILABLE else False,
    "blackjax_available": BLACKJAX_AVAILABLE if MCMC_AVAILABLE else False,
}

# Primary API functions
__all__ = [
    # Primary optimization methods
    "fit_nlsq_jax",  # NLSQ trust-region (PRIMARY)
    "fit_mcmc_jax",  # NumPyro/BlackJAX NUTS (SECONDARY)
    # Result classes
    "NLSQResult",
    "MCMCResult",
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
        "secondary_method": "mcmc" if MCMC_AVAILABLE else None,
        "recommendations": [],
    }

    if NLSQ_AVAILABLE:
        info["recommendations"].append(
            "Use fit_nlsq_jax() for fast, reliable parameter estimation",
        )

    if MCMC_AVAILABLE:
        info["recommendations"].append(
            "Use fit_mcmc_jax() for uncertainty quantification and publication-quality analysis",
        )

    if not NLSQ_AVAILABLE and not MCMC_AVAILABLE:
        info["recommendations"].append(
            "Install NLSQ and NumPyro/BlackJAX for optimization capabilities",
        )

    return info
