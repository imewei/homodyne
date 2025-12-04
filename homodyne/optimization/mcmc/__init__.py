"""MCMC Optimization Subpackage for Homodyne.

This subpackage contains all MCMC (Markov Chain Monte Carlo) optimization
components organized into logical modules.

Structure:
- core.py: Main fit_mcmc_jax function and MCMC constants
- data_prep.py: Data preparation utilities for NumPyro models
- priors.py: Prior distribution utilities
- scaling.py: Scaling utilities for per-angle/global scaling
- single_angle.py: Single-angle MCMC utilities (log-space D0 sampling)
- cmc/: Consensus Monte Carlo subpackage
  - coordinator.py: CMC workflow orchestration
  - sharding.py: Data sharding strategies
  - result.py: MCMCResult dataclass
  - backends/: Backend implementations (multiprocessing, PBS)

Note (v2.4.1): The tier/surrogate system has been removed. Single-angle datasets
now sample ALL 5 parameters (D0, alpha, D_offset, contrast, offset) using
log-space D0 sampling for better MCMC geometry.
"""

from homodyne.optimization.mcmc.core import (
    BLACKJAX_AVAILABLE,
    JAX_AVAILABLE,
    NUMPYRO_AVAILABLE,
    _calculate_midpoint_defaults,
    _create_numpyro_model,
    _estimate_single_angle_scaling,
    _evaluate_convergence_thresholds,
    _format_init_params_for_chains,
    _get_mcmc_config,
    _prepare_phi_mapping,
    _process_posterior_samples,
    _run_numpyro_sampling,
    _sample_single_angle_log_d0,
    fit_mcmc_jax,
)
from homodyne.optimization.mcmc.cmc.result import MCMCResult
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

__all__ = [
    # Core
    "fit_mcmc_jax",
    "JAX_AVAILABLE",
    "NUMPYRO_AVAILABLE",
    "BLACKJAX_AVAILABLE",
    "_create_numpyro_model",
    "_calculate_midpoint_defaults",
    "_estimate_single_angle_scaling",
    "_evaluate_convergence_thresholds",
    "_format_init_params_for_chains",
    "_get_mcmc_config",
    "_prepare_phi_mapping",
    "_process_posterior_samples",
    "_run_numpyro_sampling",
    "_sample_single_angle_log_d0",
    # Result
    "MCMCResult",
    # Data preparation
    "get_target_dtype",
    "normalize_array",
    "normalize_scalar",
    "prepare_mcmc_arrays",
    "validate_array_shapes",
    "compute_phi_mapping",
    # Priors
    "DIST_TYPE_MAP",
    "get_prior_spec_with_fallback",
    "auto_convert_to_bounded_distribution",
    "cast_dist_kwargs",
    "prepare_truncated_normal_kwargs",
    "create_beta_scaled_distribution",
    "sample_parameter",
    "sample_scaling_parameters",
    # Scaling
    "prepare_phi_mapping",
    "compute_phi_indices",
    "extract_per_point_theory",
    "apply_per_angle_scaling",
    "apply_global_scaling",
    "select_scaling_per_point",
    "validate_c2_fitted_shape",
    "apply_scaling_to_theory",
    # Single-angle
    "estimate_single_angle_scaling",
    "sample_log_d0",
    "is_single_angle_static",
    "build_log_d0_prior_config",
]
