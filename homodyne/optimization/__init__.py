"""
LSQ and MCMC+JAX Optimization for Homodyne v2
=============================================

Reliable optimization system using LSQ (primary) and MCMC+JAX
(high-accuracy) for robust parameter estimation in homodyne analysis.

This module implements the streamlined optimization philosophy:
1. LSQ as primary method (fast, reliable parameter estimation)
2. MCMC+JAX (NumPyro/BlackJAX) for uncertainty quantification
3. Hybrid LSQ→MCMC for comprehensive analysis
4. Unified homodyne model: c2_fitted = c2_theory * contrast + offset

Key Features:
- Robust least squares optimization as foundation
- JAX-accelerated MCMC for uncertainty quantification
- Hybrid pipeline combining speed and accuracy
- CPU-primary, GPU-optional architecture
- Dataset size-aware optimization strategies

Performance Comparison:
- LSQ: Fast, reliable parameter estimation
- MCMC+JAX: Full posterior sampling, highest accuracy
- Hybrid: Best balance of speed and statistical rigor
"""

# Handle imports with intelligent fallback
try:
    from homodyne.optimization.lsq_wrapper import fit_homodyne_lsq
    LSQ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import LSQ optimization: {e}")
    fit_homodyne_lsq = None
    LSQ_AVAILABLE = False

try:
    from homodyne.optimization.hybrid import fit_hybrid_lsq_mcmc, HybridResult, optimize_hybrid
    HYBRID_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Hybrid optimization: {e}")
    fit_hybrid_lsq_mcmc = None
    HybridResult = None
    optimize_hybrid = None
    HYBRID_AVAILABLE = False

try:
    from homodyne.optimization.mcmc import \
        HAS_NUMPY_GRADIENTS as MCMC_HAS_NUMPY_GRADIENTS
    from homodyne.optimization.mcmc import JAX_AVAILABLE as MCMC_JAX_AVAILABLE
    from homodyne.optimization.mcmc import fit_mcmc_jax  # Primary API function
    from homodyne.optimization.mcmc import MCMCJAXSampler, MCMCResult

    MCMC_AVAILABLE = True
    MCMC_FALLBACK_MODE = (
        "numpy_mh"
        if not MCMC_JAX_AVAILABLE and MCMC_HAS_NUMPY_GRADIENTS
        else ("unavailable" if not MCMC_JAX_AVAILABLE else "jax")
    )
except ImportError as e:
    print(f"Warning: Could not import MCMC optimization: {e}")
    MCMCJAXSampler = None
    MCMCResult = None
    fit_mcmc_jax = None
    MCMC_AVAILABLE = False
    MCMC_FALLBACK_MODE = "unavailable"

# Base result classes (always available)
try:
    from homodyne.optimization.base_result import (BaseOptimizationResult,
                                                  LSQResult,
                                                  create_result)
    BASE_RESULT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import base result classes: {e}")
    BaseOptimizationResult = None
    LSQResult = None
    create_result = None
    BASE_RESULT_AVAILABLE = False

try:
    from homodyne.data.optimization import (DatasetInfo, DatasetOptimizer,
                                            ProcessingStrategy,
                                            create_dataset_optimizer,
                                            optimize_for_method)

    DATASET_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import dataset optimization: {e}")
    DatasetOptimizer = None
    optimize_for_method = None
    DatasetInfo = None
    ProcessingStrategy = None
    create_dataset_optimizer = None
    DATASET_OPTIMIZATION_AVAILABLE = False

# Direct solver has been removed - functionality now in lsq_wrapper.py
DirectLeastSquaresSolver = None
DirectSolverConfig = None
fit_homodyne_direct = None
DIRECT_SOLVER_AVAILABLE = False


# Main API functions for homodyne optimization
def fit_homodyne_lsq_api(
    data, sigma, t1, t2, phi, q, L, analysis_mode="laminar_flow", **kwargs
):
    """
    Fit homodyne data using Least Squares optimization.

    Fast, reliable parameter estimation using robust least squares methods.
    This is the primary optimization method for routine homodyne analysis.

    Args:
        data, sigma: Experimental data and uncertainties
        t1, t2, phi: Time and angle grids
        q, L: Experimental parameters
        analysis_mode: Analysis mode
        **kwargs: Additional LSQ parameters

    Returns:
        LSQResult with fitted parameters and statistics

    Raises:
        ImportError: If LSQ optimization is not available
    """
    if not LSQ_AVAILABLE:
        raise ImportError(
            "LSQ optimization not available. Please install dependencies."
        )

    return fit_homodyne_lsq(data, sigma, t1, t2, phi, q, L, analysis_mode, **kwargs)


def fit_homodyne_hybrid(
    data, sigma, t1, t2, phi, q, L, analysis_mode="laminar_flow", **kwargs
):
    """
    Fit homodyne data using Hybrid LSQ→MCMC optimization.

    Combines LSQ speed with MCMC accuracy for comprehensive analysis.
    This is the recommended method for publication-quality results.

    Args:
        data, sigma: Experimental data and uncertainties
        t1, t2, phi: Time and angle grids
        q, L: Experimental parameters
        analysis_mode: Analysis mode
        **kwargs: Additional hybrid parameters

    Returns:
        HybridResult with LSQ and MCMC results and recommendations

    Raises:
        ImportError: If hybrid optimization is not available
    """
    if not HYBRID_AVAILABLE:
        raise ImportError(
            "Hybrid optimization not available. Please install dependencies."
        )

    return fit_hybrid_lsq_mcmc(data, sigma, t1, t2, phi, q, L, analysis_mode, **kwargs)


def fit_homodyne_mcmc(
    data, sigma, t1, t2, phi, q, L, analysis_mode="laminar_flow", **kwargs
):
    """
    Fit homodyne data using MCMC with intelligent fallbacks.

    Uses NumPyro/BlackJAX for full posterior sampling with unified homodyne model.
    Recommended for critical analysis requiring detailed uncertainty quantification.

    Performance modes (automatic selection):
    - JAX+NumPyro/BlackJAX: Full NUTS sampling (fastest, best diagnostics)
    - NumPy+Metropolis-Hastings: Fallback MCMC (50-200x slower but accurate)

    Args:
        data, sigma: Experimental data and uncertainties
        t1, t2, phi: Time and angle grids
        q, L: Experimental parameters
        analysis_mode: Analysis mode
        **kwargs: Additional MCMC parameters

    Returns:
        MCMCResult with posterior samples and uncertainties

    Raises:
        ImportError: If no MCMC backend is available
    """
    if not MCMC_AVAILABLE:
        raise ImportError(
            "MCMC not available. Please install dependencies:\n"
            "- For best performance: pip install jax jaxlib numpyro\n"
            "- Alternative: pip install jax jaxlib blackjax\n"
            "- For fallback mode: homodyne with numpy_gradients module"
        )

    # Provide user guidance on performance mode
    if MCMC_FALLBACK_MODE == "numpy_mh":
        print(
            "INFO: Using NumPy+Metropolis-Hastings fallback (50-200x slower than JAX+NUTS)"
        )
    elif MCMC_FALLBACK_MODE == "jax" and "numpyro" not in str(kwargs):
        print("INFO: Using JAX-accelerated MCMC for optimal performance")

    return fit_mcmc_jax(data, sigma, t1, t2, phi, q, L, analysis_mode, **kwargs)


__all__ = [
    # Primary API (LSQ, MCMC, and Hybrid)
    "fit_homodyne_lsq_api",  # Main LSQ fitting function
    "fit_homodyne_mcmc",     # Main MCMC+JAX fitting function
    "fit_homodyne_hybrid",   # Main Hybrid LSQ→MCMC fitting function
    # Direct access to LSQ
    "fit_homodyne_lsq",
    # Direct access to MCMC+JAX
    "MCMCJAXSampler",
    "MCMCResult",
    "fit_mcmc_jax",
    # Direct access to Hybrid
    "fit_hybrid_lsq_mcmc",
    "HybridResult",
    "optimize_hybrid",
    # Unified result classes (consolidated from base_result.py)
    "BaseOptimizationResult",
    "LSQResult",
    "create_result",
    # Dataset optimization
    "DatasetOptimizer",
    "optimize_for_method",
    "DatasetInfo",
    "ProcessingStrategy",
    "create_dataset_optimizer",
    # Availability flags and fallback modes
    "LSQ_AVAILABLE",
    "HYBRID_AVAILABLE",
    "MCMC_AVAILABLE",
    "DATASET_OPTIMIZATION_AVAILABLE",
    "DIRECT_SOLVER_AVAILABLE",
    "MCMC_FALLBACK_MODE",
]
