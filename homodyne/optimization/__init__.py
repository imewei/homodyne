"""
VI+JAX and MCMC+JAX Only: Streamlined Optimization for Homodyne v2
===================================================================

Streamlined optimization system using only VI+JAX (primary) and MCMC+JAX
(high-accuracy) for efficient parameter estimation in homodyne analysis.

This module implements the simplified optimization philosophy:
1. VI+JAX as primary method (10-100x faster, KL divergence minimization)
2. MCMC+JAX (NumPyro/BlackJAX) as refinement for critical analysis
3. Unified homodyne model: c2_fitted = c2_theory * contrast + offset
4. Both methods minimize: Exp - Fitted

Key Features:
- Complete removal of classical/robust optimization methods
- JAX-accelerated computations with automatic differentiation
- Unified parameter space with specified bounds and priors
- CPU-primary, GPU-optional architecture
- Dataset size-aware optimization strategies

Performance Comparison:
- VI+JAX: 10-100x faster, good uncertainty quantification
- MCMC+JAX: Full posterior sampling, highest accuracy
"""

# Handle imports with intelligent fallback
try:
    from homodyne.optimization.variational import \
        HAS_NUMPY_GRADIENTS as VI_HAS_NUMPY_GRADIENTS
    from homodyne.optimization.variational import \
        JAX_AVAILABLE as VI_JAX_AVAILABLE
    from homodyne.optimization.variational import \
        fit_vi_jax  # Primary API function
    from homodyne.optimization.variational import (VariationalFamilies,
                                                   VariationalInferenceJAX,
                                                   VIResult)

    VI_AVAILABLE = True
    VI_FALLBACK_MODE = (
        "numpy_gradients"
        if not VI_JAX_AVAILABLE and VI_HAS_NUMPY_GRADIENTS
        else ("simple" if not VI_JAX_AVAILABLE else "jax")
    )
except ImportError as e:
    print(f"Warning: Could not import VI optimization: {e}")
    VariationalInferenceJAX = None
    VIResult = None
    VariationalFamilies = None
    fit_vi_jax = None
    VI_AVAILABLE = False
    VI_FALLBACK_MODE = "unavailable"

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


# Main API functions (replace all classical methods)
def fit_homodyne_vi(
    data, sigma, t1, t2, phi, q, L, analysis_mode="laminar_flow", **kwargs
):
    """
    Fit homodyne data using Variational Inference with intelligent fallbacks.

    This is the main fitting function that replaces all classical optimization.
    Uses KL divergence minimization with unified homodyne model.

    Performance modes (automatic selection):
    - JAX mode: Full JAX acceleration (fastest, 100x speedup)
    - NumPy+Gradients mode: Numerical gradients fallback (10-50x slower but accurate)
    - Simple mode: Least squares approximation (fast but limited accuracy)

    Args:
        data, sigma: Experimental data and uncertainties
        t1, t2, phi: Time and angle grids
        q, L: Experimental parameters
        analysis_mode: Analysis mode
        **kwargs: Additional VI parameters

    Returns:
        VIResult with optimized parameters and uncertainties

    Raises:
        ImportError: If no optimization backend is available
    """
    if not VI_AVAILABLE:
        raise ImportError(
            "Variational Inference not available. Please install dependencies:\n"
            "- For best performance: pip install jax jaxlib\n"
            "- For fallback mode: homodyne with numpy_gradients module"
        )

    # Provide user guidance on performance mode
    if VI_FALLBACK_MODE == "numpy_gradients":
        print("INFO: Using NumPy+numerical gradients fallback (10-50x slower than JAX)")
    elif VI_FALLBACK_MODE == "simple":
        print(
            "WARNING: Using simple fallback mode - limited accuracy. Consider installing JAX."
        )

    return fit_vi_jax(data, sigma, t1, t2, phi, q, L, analysis_mode, **kwargs)


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
    # Primary API (replaces all classical methods)
    "fit_homodyne_vi",  # Main VI+JAX fitting function
    "fit_homodyne_mcmc",  # Main MCMC+JAX fitting function
    # Direct access to VI+JAX
    "VariationalInferenceJAX",
    "VIResult",
    "VariationalFamilies",
    "fit_vi_jax",
    # Direct access to MCMC+JAX
    "MCMCJAXSampler",
    "MCMCResult",
    "fit_mcmc_jax",
    # Dataset optimization
    "DatasetOptimizer",
    "optimize_for_method",
    "DatasetInfo",
    "ProcessingStrategy",
    "create_dataset_optimizer",
    # Availability flags and fallback modes
    "VI_AVAILABLE",
    "MCMC_AVAILABLE",
    "DATASET_OPTIMIZATION_AVAILABLE",
    "VI_FALLBACK_MODE",
    "MCMC_FALLBACK_MODE",
]
