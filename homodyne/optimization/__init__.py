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

# Import MCMC extraction modules for public API
# These modules were extracted from _create_numpyro_model (Dec 2025)
# to reduce cyclomatic complexity and improve maintainability.
try:
    from homodyne.optimization.mcmc.data_prep import (
        compute_phi_mapping,
        get_target_dtype,
        normalize_array,
        normalize_scalar,
        prepare_mcmc_arrays,
        validate_array_shapes,
    )
    from homodyne.optimization.mcmc.priors import (
        DIST_TYPE_MAP,
        auto_convert_to_bounded_distribution,
        cast_dist_kwargs,
        create_beta_scaled_distribution,
        get_prior_spec_with_fallback,
        prepare_truncated_normal_kwargs,
        sample_parameter,
        sample_scaling_parameters,
    )
    from homodyne.optimization.mcmc.scaling import (
        apply_global_scaling,
        apply_per_angle_scaling,
        apply_scaling_to_theory,
        compute_phi_indices,
        extract_per_point_theory,
        prepare_phi_mapping,
        select_scaling_per_point,
        validate_c2_fitted_shape,
    )
    from homodyne.optimization.mcmc.single_angle import (
        build_log_d0_prior_config,
        estimate_single_angle_scaling,
        is_single_angle_static,
        sample_log_d0,
    )

    MCMC_EXTRACTION_AVAILABLE = True
except ImportError:
    MCMC_EXTRACTION_AVAILABLE = False


# Primary API functions
__all__ = [
    # Primary optimization methods
    "fit_nlsq_jax",  # NLSQ trust-region (PRIMARY)
    "fit_mcmc_jax",  # NumPyro/BlackJAX NUTS (SECONDARY)
    # Result classes
    "NLSQResult",
    "MCMCResult",
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
    "MCMC_EXTRACTION_AVAILABLE",
    # MCMC data preparation utilities
    "get_target_dtype",
    "normalize_array",
    "normalize_scalar",
    "prepare_mcmc_arrays",
    "validate_array_shapes",
    "compute_phi_mapping",
    # MCMC prior utilities
    "DIST_TYPE_MAP",
    "get_prior_spec_with_fallback",
    "auto_convert_to_bounded_distribution",
    "cast_dist_kwargs",
    "prepare_truncated_normal_kwargs",
    "create_beta_scaled_distribution",
    "sample_parameter",
    "sample_scaling_parameters",
    # MCMC scaling utilities
    "prepare_phi_mapping",
    "compute_phi_indices",
    "extract_per_point_theory",
    "apply_per_angle_scaling",
    "apply_global_scaling",
    "select_scaling_per_point",
    "validate_c2_fitted_shape",
    "apply_scaling_to_theory",
    # MCMC single-angle utilities (v2.4.1+: simplified, no tier system)
    "estimate_single_angle_scaling",
    "sample_log_d0",
    "is_single_angle_static",
    "build_log_d0_prior_config",
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
