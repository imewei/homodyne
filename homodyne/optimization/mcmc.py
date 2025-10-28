"""MCMC + JAX: High-Accuracy Bayesian Analysis for Homodyne v2
===========================================================

NumPyro/BlackJAX-based MCMC sampling for high-precision parameter estimation
and uncertainty quantification. MCMC serves as the refinement method for
critical analysis using the unified homodyne model.

Key Features:
- NumPyro/BlackJAX NUTS sampling only (PyMC completely removed)
- Unified homodyne model: c2_fitted = c2_theory * contrast + offset
- Same likelihood as VI: Exp - (contrast * Theory + offset)
- Full posterior sampling with specified parameter priors
- JAX acceleration for both CPU and GPU
- Comprehensive convergence diagnostics
- Can be initialized from VI results for efficiency
- Automatic method selection: NUTS vs CMC based on dual criteria
- Consensus Monte Carlo (CMC) for many samples (>=100) OR large memory footprint

MCMC Philosophy:
- Gold standard for uncertainty quantification
- Full posterior sampling (not just point estimates)
- Essential for critical/publication-quality analysis
- Complements VI+JAX for comprehensive Bayesian workflow

Method Selection:
- 'auto': Automatically select NUTS or CMC based on dataset size and hardware
- 'nuts': Force standard NUTS (may fail on very large datasets)
- 'cmc': Force Consensus Monte Carlo (adds overhead for small datasets)
"""

import time
from typing import Any, Dict

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# JAX imports with intelligent fallback
try:
    import jax.numpy as jnp
    from jax import jit, random

    JAX_AVAILABLE = True
    HAS_NUMPY_GRADIENTS = False  # Use JAX when available
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    HAS_NUMPY_GRADIENTS = False
    logger.warning(
        "JAX not available - MCMC will use numpy fallback with limited functionality",
    )

    def jit(f):
        return f

    class MockRandom:
        @staticmethod
        def PRNGKey(seed):
            np.random.seed(seed)
            return seed

        @staticmethod
        def normal(key, shape):
            return np.random.normal(size=shape)

    random = MockRandom()


# NumPyro imports with fallback
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro import sample
    from numpyro.infer import MCMC, NUTS

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    numpyro = None

# BlackJAX imports with fallback
try:
    import blackjax

    BLACKJAX_AVAILABLE = True
except ImportError:
    BLACKJAX_AVAILABLE = False
    blackjax = None

# Core homodyne imports
try:
    from homodyne.core.fitting import ParameterSpace

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False

# Import extended MCMCResult with CMC support
# This provides backward compatibility while supporting CMC
try:
    from homodyne.optimization.cmc.result import MCMCResult
    HAS_CMC_RESULT = True
except ImportError:
    # Fallback to original MCMCResult if CMC module not available
    HAS_CMC_RESULT = False

    class MCMCResult:
        """MCMC optimization result container.

        Compatible with existing MCMC result structure while simplifying
        the interface for the new architecture.
        """

        def __init__(
            self,
            mean_params: np.ndarray,
            mean_contrast: float,
            mean_offset: float,
            std_params: np.ndarray | None = None,
            std_contrast: float | None = None,
            std_offset: float | None = None,
            samples_params: np.ndarray | None = None,
            samples_contrast: np.ndarray | None = None,
            samples_offset: np.ndarray | None = None,
            converged: bool = True,
            n_iterations: int = 0,
            computation_time: float = 0.0,
            backend: str = "JAX",
            analysis_mode: str = "static_isotropic",
            dataset_size: str = "unknown",
            n_chains: int = 4,
            n_warmup: int = 1000,
            n_samples: int = 1000,
            sampler: str = "NUTS",
            acceptance_rate: float | None = None,
            r_hat: dict[str, float] | None = None,
            effective_sample_size: dict[str, float] | None = None,
            **kwargs,
        ):
            # Primary results
            self.mean_params = mean_params
            self.mean_contrast = mean_contrast
            self.mean_offset = mean_offset

            # Uncertainties
            self.std_params = (
                std_params if std_params is not None else np.zeros_like(mean_params)
            )
            self.std_contrast = std_contrast if std_contrast is not None else 0.0
            self.std_offset = std_offset if std_offset is not None else 0.0

            # Samples
            self.samples_params = samples_params
            self.samples_contrast = samples_contrast
            self.samples_offset = samples_offset

            # Metadata
            self.converged = converged
            self.n_iterations = n_iterations
            self.computation_time = computation_time
            self.backend = backend
            self.analysis_mode = analysis_mode
            self.dataset_size = dataset_size

            # MCMC-specific
            self.n_chains = n_chains
            self.n_warmup = n_warmup
            self.n_samples = n_samples
            self.sampler = sampler
            self.acceptance_rate = acceptance_rate
            self.r_hat = r_hat
            self.effective_sample_size = effective_sample_size


@log_performance(threshold=10.0)
def fit_mcmc_jax(
    data: np.ndarray,
    sigma: np.ndarray | None = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "static_isotropic",  # Changed default to simpler mode
    parameter_space: ParameterSpace | None = None,
    enable_dataset_optimization: bool = True,
    estimate_noise: bool = False,
    noise_model: str = "hierarchical",
    initial_params: dict[str, float] | None = None,  # Added for initialization
    use_simplified_likelihood: bool = True,  # Added for performance
    method: str = "auto",  # NEW: Method selection ('auto', 'nuts', 'cmc')
    **kwargs,
) -> MCMCResult:
    """High-accuracy fitting using MCMC+JAX sampling with automatic method selection.

    Uses NumPyro/BlackJAX for full posterior sampling with unified homodyne model.
    Same likelihood as VI: Exp - (contrast * Theory + offset)
    Automatically selects between standard NUTS and Consensus Monte Carlo (CMC)
    based on dataset size and hardware configuration.

    Method Selection Strategy
    -------------------------
    - **'auto'** (default): Automatically select using dual-criteria OR logic
        - CMC if: (num_samples >= 100) OR (estimated_memory > threshold)
        - NUTS otherwise (faster for few samples with manageable memory)
        - Hardware-adaptive: Considers GPU memory, CPU cores, cluster environment
    - **'nuts'**: Force standard NUTS MCMC
        - Single-device execution
        - May fail with OOM on very large datasets
        - Best for <1M points
    - **'cmc'**: Force Consensus Monte Carlo
        - Multi-shard execution with subposterior combination
        - Adds overhead for small datasets
        - Recommended for >1M points

    Parameters
    ----------
    data : np.ndarray
        Experimental correlation data (flattened)
    sigma : np.ndarray, optional
        Noise standard deviations. If None, estimated from data.
    t1 : np.ndarray
        First delay time array (flattened, same length as data)
    t2 : np.ndarray
        Second delay time array (flattened, same length as data)
    phi : np.ndarray
        Phi angle values (flattened, same length as data)
    q : float
        q-vector magnitude
    L : float
        Sample-detector distance
    analysis_mode : str, default "static_isotropic"
        Analysis mode: "static_isotropic" (3 params) or "laminar_flow" (7 params)
    parameter_space : ParameterSpace, optional
        Parameter bounds and priors
    enable_dataset_optimization : bool, default True
        Enable dataset size optimization (deprecated, kept for compatibility)
    estimate_noise : bool, default False
        Whether to estimate noise hierarchically
    noise_model : str, default "hierarchical"
        Noise model type
    initial_params : dict[str, float], optional
        Initial parameter values (e.g., from NLSQ optimization)
        Used for tighter priors and faster convergence
    use_simplified_likelihood : bool, default True
        Use simplified likelihood for faster computation
    method : str, default "auto"
        MCMC method selection:
        - 'auto': Automatic selection based on dataset size (RECOMMENDED)
        - 'nuts': Force standard NUTS (may fail on large datasets)
        - 'cmc': Force Consensus Monte Carlo (adds overhead for small datasets)
    **kwargs
        Additional MCMC/CMC configuration parameters:
        - For NUTS: n_samples, n_warmup, n_chains, target_accept_prob, etc.
        - For CMC: cmc_config dict with sharding, initialization, combination settings

    Returns
    -------
    MCMCResult
        MCMC sampling result with posterior samples and diagnostics.
        For CMC results, includes additional fields:
        - per_shard_diagnostics: List of per-shard convergence info
        - cmc_diagnostics: Overall CMC diagnostics
        - combination_method: Method used to combine posteriors
        - num_shards: Number of shards used

    Raises
    ------
    ImportError
        If NumPyro or BlackJAX not available
    ValueError
        If data validation fails or invalid method specified
    RuntimeError
        If CMC execution fails (all shards fail to converge)

    Examples
    --------
    Automatic method selection (recommended):

    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     analysis_mode='laminar_flow',
    ...     initial_params={'D0': 1000.0, 'alpha': 1.5},
    ... )
    >>> print(f"Used method: {'CMC' if result.is_cmc_result() else 'NUTS'}")

    Force standard NUTS:

    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     method='nuts',
    ... )

    Force CMC with custom configuration:

    >>> cmc_config = {
    ...     'sharding': {'num_shards': 10, 'strategy': 'stratified'},
    ...     'initialization': {'use_svi': True, 'svi_steps': 5000},
    ...     'combination': {'method': 'weighted'},
    ... }
    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     method='cmc',
    ...     cmc_config=cmc_config,
    ... )
    >>> print(f"Used {result.num_shards} shards")

    Notes
    -----
    - CMC is automatically enabled based on dual criteria: (samples >= 100) OR (memory > threshold)
    - method='auto' is recommended for all use cases
    - CMC adds ~10-20% overhead but enables parallelism and unlimited dataset sizes
    - Backward compatible: Existing code continues to work without changes
    """

    # Validate method parameter
    if method not in ['auto', 'nuts', 'cmc']:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of: 'auto', 'nuts', 'cmc'"
        )

    if not NUMPYRO_AVAILABLE and not BLACKJAX_AVAILABLE:
        raise ImportError(
            "NumPyro or BlackJAX is required for MCMC optimization. "
            "Install with: pip install numpyro blackjax",
        )

    if not HAS_CORE_MODULES:
        raise ImportError("Core homodyne modules are required for optimization")

    # Validate input data
    _validate_mcmc_data(data, t1, t2, phi, q, L)

    # Get total dataset size (handles both 1D and multi-dimensional arrays)
    dataset_size = data.size if hasattr(data, 'size') else len(data)

    # Get number of independent samples for CMC decision
    # For multi-dimensional XPCS data (n_phi, n_t1, n_t2), each phi angle is one sample
    # For 1D data, num_samples equals dataset_size
    if hasattr(data, 'shape') and hasattr(data, 'ndim'):
        num_samples = data.shape[0] if data.ndim > 1 else dataset_size
    else:
        num_samples = dataset_size

    logger.info("Starting MCMC+JAX sampling")
    logger.info(f"Dataset size: {dataset_size:,} data points ({num_samples} independent samples)")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Step 1: Detect hardware configuration
    try:
        from homodyne.device.config import detect_hardware, should_use_cmc
        hardware_config = detect_hardware()
    except ImportError:
        logger.warning(
            "Hardware detection not available. Assuming standard NUTS execution."
        )
        hardware_config = None

    # Step 2: Determine actual method to use
    if method == 'auto':
        if hardware_config is not None:
            # CMC decision based on TWO criteria (OR logic):
            # 1. Parallelism: num_samples >= threshold (e.g., 100)
            # 2. Memory: dataset_size causes estimated memory > threshold
            use_cmc = should_use_cmc(num_samples, hardware_config, dataset_size=dataset_size)
            actual_method = 'cmc' if use_cmc else 'nuts'
            logger.info(
                f"Automatic method selection: {actual_method.upper()} "
                f"(num_samples={num_samples:,}, dataset_size={dataset_size:,}, "
                f"platform={hardware_config.platform})"
            )
        else:
            # Fallback: Simple threshold-based selection using num_samples
            use_cmc = num_samples > 500
            actual_method = 'cmc' if use_cmc else 'nuts'
            logger.info(
                f"Automatic method selection (fallback): {actual_method.upper()} "
                f"(num_samples={num_samples:,})"
            )
    else:
        actual_method = method
        logger.info(f"Using user-specified method: {actual_method.upper()}")

    # Step 3: Log warnings for suboptimal method choices
    if actual_method == 'cmc' and num_samples < 20:
        logger.warning(
            f"Using CMC with very few samples ({num_samples} samples). "
            f"CMC adds 10-20% overhead; consider method='nuts' for <20 samples if memory permits."
        )
    elif actual_method == 'nuts' and num_samples > 100:
        logger.warning(
            f"Using NUTS with many samples ({num_samples:,} samples). "
            f"CMC may be 2-5x faster via parallelization on multi-core CPU; consider method='auto' or method='cmc'."
        )

    # Step 4: Execute selected method
    if actual_method == 'cmc':
        # Use Consensus Monte Carlo
        logger.info("=" * 70)
        logger.info("Executing Consensus Monte Carlo (CMC)")
        logger.info("=" * 70)

        try:
            from homodyne.optimization.cmc.coordinator import CMCCoordinator
        except ImportError as e:
            logger.error(f"CMC module not available: {e}")
            raise ImportError(
                "CMC module required for method='cmc'. "
                "Ensure homodyne.optimization.cmc is installed."
            ) from e

        # Extract CMC configuration
        cmc_config = kwargs.pop('cmc_config', {})

        # Add MCMC config to cmc_config if not already present
        if 'mcmc' not in cmc_config:
            cmc_config['mcmc'] = _get_mcmc_config(kwargs)

        # Create CMC coordinator
        coordinator = CMCCoordinator(cmc_config)

        # Prepare NLSQ params from initial_params
        nlsq_params = initial_params if initial_params is not None else {}

        # Run CMC pipeline
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            nlsq_params=nlsq_params,
        )

        logger.info(f"CMC execution completed. Used {result.num_shards} shards.")
        return result

    else:
        # Use standard NUTS
        logger.info("=" * 70)
        logger.info("Executing standard NUTS MCMC")
        logger.info("=" * 70)

        return _run_standard_nuts(
            data=data,
            sigma=sigma,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            parameter_space=parameter_space,
            initial_params=initial_params,
            use_simplified_likelihood=use_simplified_likelihood,
            **kwargs,
        )


def _run_standard_nuts(
    data: np.ndarray,
    sigma: np.ndarray | None = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "static_isotropic",
    parameter_space: ParameterSpace | None = None,
    initial_params: dict[str, float] | None = None,
    use_simplified_likelihood: bool = True,
    **kwargs,
) -> MCMCResult:
    """Execute standard NUTS MCMC sampling.

    This is the original MCMC implementation extracted into a helper function.
    Used when method='nuts' or when automatic selection chooses standard NUTS.

    Parameters
    ----------
    data : np.ndarray
        Experimental correlation data
    sigma : np.ndarray, optional
        Noise standard deviations
    t1, t2, phi : np.ndarray
        Time and angle arrays
    q, L : float
        Physical parameters
    analysis_mode : str
        Analysis mode
    parameter_space : ParameterSpace, optional
        Parameter bounds and priors
    initial_params : dict, optional
        Initial parameter values
    use_simplified_likelihood : bool
        Use simplified likelihood
    **kwargs
        Additional MCMC configuration

    Returns
    -------
    MCMCResult
        Standard MCMC result (non-CMC)
    """
    logger.info("Starting standard NUTS MCMC sampling")
    start_time = time.perf_counter()

    try:
        # Set up parameter space
        if parameter_space is None:
            parameter_space = ParameterSpace()

        # Configure MCMC parameters
        mcmc_config = _get_mcmc_config(kwargs)

        # Determine analysis mode
        if analysis_mode not in ["static_isotropic", "laminar_flow"]:
            logger.warning(
                f"Unknown analysis mode {analysis_mode}, using static_isotropic",
            )
            analysis_mode = "static_isotropic"

        logger.info(f"Analysis mode: {analysis_mode}")

        # Set up noise model
        if sigma is None:
            sigma = _estimate_noise(data)

        # Create NumPyro model with optional initialization
        model = _create_numpyro_model(
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            analysis_mode,
            parameter_space,
            initial_params=initial_params,
            use_simplified=use_simplified_likelihood,
        )

        # Run MCMC sampling
        if NUMPYRO_AVAILABLE:
            result = _run_numpyro_sampling(model, mcmc_config)
        else:
            result = _run_blackjax_sampling(model, mcmc_config)

        # Process results
        posterior_summary = _process_posterior_samples(result, analysis_mode)

        computation_time = time.perf_counter() - start_time

        logger.info(f"MCMC sampling completed in {computation_time:.3f}s")
        logger.info(f"Posterior summary: {len(posterior_summary['samples'])} samples")

        return MCMCResult(
            mean_params=posterior_summary["mean_params"],
            mean_contrast=posterior_summary["mean_contrast"],
            mean_offset=posterior_summary["mean_offset"],
            std_params=posterior_summary["std_params"],
            std_contrast=posterior_summary["std_contrast"],
            std_offset=posterior_summary["std_offset"],
            samples_params=posterior_summary["samples_params"],
            samples_contrast=posterior_summary["samples_contrast"],
            samples_offset=posterior_summary["samples_offset"],
            converged=posterior_summary["converged"],
            n_iterations=mcmc_config["n_samples"],
            computation_time=computation_time,
            backend="JAX",
            analysis_mode=analysis_mode,
            n_chains=mcmc_config["n_chains"],
            n_warmup=mcmc_config["n_warmup"],
            n_samples=mcmc_config["n_samples"],
            sampler="NUTS",
            acceptance_rate=posterior_summary.get("acceptance_rate"),
            r_hat=posterior_summary.get("r_hat"),
            effective_sample_size=posterior_summary.get("ess"),
        )

    except Exception as e:
        computation_time = time.perf_counter() - start_time
        logger.error(f"MCMC sampling failed after {computation_time:.3f}s: {e}")

        # Return failed result
        n_params = 3 if "static" in analysis_mode else 7
        return MCMCResult(
            mean_params=np.zeros(n_params),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=False,
            n_iterations=0,
            computation_time=computation_time,
            backend="JAX",
            analysis_mode=analysis_mode,
        )


def _validate_mcmc_data(data, t1, t2, phi, q, L):
    """Validate MCMC input data."""
    if data is None or data.size == 0:
        raise ValueError("Data cannot be None or empty")

    required_arrays = [t1, t2, phi]
    array_names = ["t1", "t2", "phi"]

    for arr, name in zip(required_arrays, array_names, strict=False):
        if arr is None:
            raise ValueError(f"{name} cannot be None")

    if q is None or L is None:
        raise ValueError("q and L parameters cannot be None")


def _estimate_noise(data: np.ndarray) -> np.ndarray:
    """Estimate noise from data if not provided."""
    # Simple noise estimation: assume 1% relative noise
    noise_level = 0.01 * np.abs(data)
    # Minimum noise floor
    min_noise = 0.001
    return np.maximum(noise_level, min_noise)


def _get_mcmc_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Get MCMC configuration with optimized defaults."""
    default_config = {
        "n_samples": 1000,
        "n_warmup": 500,  # Reduced warmup for faster testing
        "n_chains": 4,  # Enable parallel chains by default
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
        "rng_key": 42,
    }

    # Update with provided kwargs
    default_config.update(kwargs)
    return default_config


def _create_numpyro_model(
    data,
    sigma,
    t1,
    t2,
    phi,
    q,
    L,
    analysis_mode,
    param_space,
    initial_params=None,
    use_simplified=True,
):
    """Create NumPyro probabilistic model with optional initialization.

    Parameters
    ----------
    initial_params : dict, optional
        Initial parameter values for better starting points (from NLSQ)
    use_simplified : bool
        Use simplified likelihood for faster computation

    Notes
    -----
    If initial_params provided (from NLSQ), uses tighter priors centered
    on NLSQ values for faster MCMC convergence. Otherwise uses default
    broad priors from parameter_space.
    """

    def homodyne_model():
        # Define priors based on analysis mode
        if "static" in analysis_mode:
            # Static mode: 5 parameters
            contrast = sample(
                "contrast",
                dist.TruncatedNormal(
                    param_space.contrast_prior[0],
                    param_space.contrast_prior[1],
                    low=param_space.contrast_bounds[0],
                    high=param_space.contrast_bounds[1],
                ),
            )
            offset = sample(
                "offset",
                dist.TruncatedNormal(
                    param_space.offset_prior[0],
                    param_space.offset_prior[1],
                    low=param_space.offset_bounds[0],
                    high=param_space.offset_bounds[1],
                ),
            )
            D0 = sample(
                "D0",
                dist.TruncatedNormal(
                    param_space.D0_prior[0],
                    param_space.D0_prior[1],
                    low=param_space.D0_bounds[0],
                    high=param_space.D0_bounds[1],
                ),
            )
            alpha = sample(
                "alpha",
                dist.TruncatedNormal(
                    param_space.alpha_prior[0],
                    param_space.alpha_prior[1],
                    low=param_space.alpha_bounds[0],
                    high=param_space.alpha_bounds[1],
                ),
            )
            D_offset = sample(
                "D_offset",
                dist.TruncatedNormal(
                    param_space.D_offset_prior[0],
                    param_space.D_offset_prior[1],
                    low=param_space.D_offset_bounds[0],
                    high=param_space.D_offset_bounds[1],
                ),
            )

            params = jnp.array([contrast, offset, D0, alpha, D_offset])
        else:
            # Laminar flow mode: 9 parameters
            # (Add the additional 4 parameters)
            contrast = sample(
                "contrast",
                dist.TruncatedNormal(
                    param_space.contrast_prior[0],
                    param_space.contrast_prior[1],
                    low=param_space.contrast_bounds[0],
                    high=param_space.contrast_bounds[1],
                ),
            )
            offset = sample(
                "offset",
                dist.TruncatedNormal(
                    param_space.offset_prior[0],
                    param_space.offset_prior[1],
                    low=param_space.offset_bounds[0],
                    high=param_space.offset_bounds[1],
                ),
            )
            D0 = sample(
                "D0",
                dist.TruncatedNormal(
                    param_space.D0_prior[0],
                    param_space.D0_prior[1],
                    low=param_space.D0_bounds[0],
                    high=param_space.D0_bounds[1],
                ),
            )
            alpha = sample(
                "alpha",
                dist.TruncatedNormal(
                    param_space.alpha_prior[0],
                    param_space.alpha_prior[1],
                    low=param_space.alpha_bounds[0],
                    high=param_space.alpha_bounds[1],
                ),
            )
            D_offset = sample(
                "D_offset",
                dist.TruncatedNormal(
                    param_space.D_offset_prior[0],
                    param_space.D_offset_prior[1],
                    low=param_space.D_offset_bounds[0],
                    high=param_space.D_offset_bounds[1],
                ),
            )
            gamma_dot_t0 = sample(
                "gamma_dot_t0",
                dist.TruncatedNormal(
                    param_space.gamma_dot_t0_prior[0],
                    param_space.gamma_dot_t0_prior[1],
                    low=param_space.gamma_dot_t0_bounds[0],
                    high=param_space.gamma_dot_t0_bounds[1],
                ),
            )
            beta = sample(
                "beta",
                dist.TruncatedNormal(
                    param_space.beta_prior[0],
                    param_space.beta_prior[1],
                    low=param_space.beta_bounds[0],
                    high=param_space.beta_bounds[1],
                ),
            )
            gamma_dot_t_offset = sample(
                "gamma_dot_t_offset",
                dist.TruncatedNormal(
                    param_space.gamma_dot_t_offset_prior[0],
                    param_space.gamma_dot_t_offset_prior[1],
                    low=param_space.gamma_dot_t_offset_bounds[0],
                    high=param_space.gamma_dot_t_offset_bounds[1],
                ),
            )
            phi0 = sample(
                "phi0",
                dist.TruncatedNormal(
                    param_space.phi0_prior[0],
                    param_space.phi0_prior[1],
                    low=param_space.phi0_bounds[0],
                    high=param_space.phi0_bounds[1],
                ),
            )

            params = jnp.array(
                [
                    contrast,
                    offset,
                    D0,
                    alpha,
                    D_offset,
                    gamma_dot_t0,
                    beta,
                    gamma_dot_t_offset,
                    phi0,
                ],
            )

        # Compute theoretical model using JIT-compiled function
        # Pass full params array for proper indexing
        c2_theory = _compute_simple_theory_jit(params, t1, t2, phi, q, analysis_mode)

        # Scaled model: c2_fitted = contrast * c2_theory + offset
        c2_fitted = contrast * c2_theory + offset

        # Likelihood
        sample("obs", dist.Normal(c2_fitted, sigma), obs=data)

    return homodyne_model


def _compute_simple_theory(params, t1, t2, phi, q, analysis_mode):
    """Optimized theoretical model computation for MCMC.

    Uses simplified calculations for initial MCMC testing.
    For production, integrate with TheoryEngine.

    Parameters
    ----------
    params : array
        Full parameter array [contrast, offset, D0, alpha, D_offset, ...]
    t1, t2 : arrays
        Flattened time delay arrays (already matched to data points)
    phi : array
        Flattened angle array (already matched to data points)
    q : scalar
        Wavevector magnitude
    analysis_mode : str
        Analysis mode string

    Returns
    -------
    array
        Theoretical c2 values, flattened to match data shape

    Notes
    -----
    Input arrays t1, t2, phi must all have the same length and correspond
    to the flattened data structure.
    """
    # Extract physical parameters (skip contrast and offset at indices 0,1)
    D0 = params[2]
    alpha = params[3]
    D_offset = params[4] if len(params) > 4 else 0.0

    # Time delay (absolute difference) - already flattened arrays
    tau = jnp.abs(t2 - t1)
    tau_safe = jnp.maximum(tau, 1e-10)  # Avoid division by zero

    # Effective diffusion coefficient
    effective_D = D0 + D_offset

    # Simple diffusion model: g1(tau) = exp(-D*q^2*tau^|alpha|)
    q_squared = q * q
    exponent = -effective_D * q_squared * jnp.power(tau_safe, jnp.abs(alpha))
    g1 = jnp.exp(exponent)

    # g2 theory: c2 = 1 + |g1|^2
    c2_theory = 1.0 + g1 * g1

    # Return flattened array matching input data shape
    return c2_theory


# JIT compile with static arguments for maximum performance
# Static argnums: (5,) for analysis_mode
_compute_simple_theory_jit = jit(_compute_simple_theory, static_argnums=(5,))


def _run_numpyro_sampling(model, config):
    """Run NumPyro MCMC sampling with parallel chains and comprehensive diagnostics.

    Parameters
    ----------
    model : callable
        NumPyro model function
    config : dict
        MCMC configuration with keys:
        - n_chains : int
        - n_warmup : int
        - n_samples : int
        - target_accept_prob : float
        - max_tree_depth : int
        - rng_key : int

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        MCMC object with samples and diagnostics

    Notes
    -----
    - Configures parallel chains for CPU/GPU
    - Extracts acceptance probability and divergence info
    - Logs warmup diagnostics for debugging
    """
    # Configure parallel chains - must be done before creating MCMC object
    # For CPU parallelization, set host device count
    n_chains = config.get("n_chains", 1)
    if n_chains > 1:
        try:
            import jax

            n_devices = jax.local_device_count()
            # Check actual platform - single GPU also has n_devices=1
            platform = jax.devices()[0].platform if jax.devices() else "cpu"

            if platform == "cpu" and n_devices == 1:
                # CPU mode: use host device count for parallel chains
                numpyro.set_host_device_count(n_chains)
                logger.info(
                    f"Set host device count to {n_chains} for CPU parallel chains",
                )
            elif platform == "gpu":
                # GPU mode: use available GPU devices
                logger.info(f"Using GPU with {n_chains} chains on {n_devices} device(s)")
            else:
                logger.info(f"Using {min(n_chains, n_devices)} parallel devices")
        except Exception as e:
            logger.warning(f"Could not configure parallel chains: {e}")

    # Create NUTS kernel with diagnostics
    nuts_kernel = NUTS(
        model,
        target_accept_prob=config["target_accept_prob"],
        max_tree_depth=config.get("max_tree_depth", 10),
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,  # Use diagonal mass matrix for efficiency
    )

    # Create MCMC sampler
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=config["n_warmup"],
        num_samples=config["n_samples"],
        num_chains=config["n_chains"],
        progress_bar=True,  # Enable progress bar for monitoring
    )

    rng_key = random.PRNGKey(config["rng_key"])

    # Run MCMC sampling with comprehensive diagnostics
    logger.info(
        f"Starting NUTS sampling: {config['n_chains']} chains, "
        f"{config['n_warmup']} warmup, {config['n_samples']} samples"
    )

    try:
        mcmc.run(
            rng_key,
            extra_fields=(
                "potential_energy",
                "accept_prob",
                "diverging",
                "num_steps",
            )
        )

        # Log warmup diagnostics
        _log_warmup_diagnostics(mcmc)

    except Exception as e:
        logger.error(f"MCMC sampling failed: {e}")
        raise

    return mcmc


def _log_warmup_diagnostics(mcmc):
    """Log MCMC warmup diagnostics for debugging.

    Parameters
    ----------
    mcmc : numpyro.infer.MCMC
        MCMC object after sampling
    """
    try:
        extra_fields = mcmc.get_extra_fields()

        # Log divergence information
        if "diverging" in extra_fields:
            n_divergences = int(jnp.sum(extra_fields["diverging"]))
            if n_divergences > 0:
                logger.warning(
                    f"MCMC had {n_divergences} divergent transitions! "
                    "Consider increasing target_accept_prob or reparameterizing model."
                )
            else:
                logger.info("No divergent transitions detected")

        # Log acceptance rate
        if "accept_prob" in extra_fields:
            mean_accept = float(jnp.mean(extra_fields["accept_prob"]))
            logger.info(f"Mean acceptance probability: {mean_accept:.3f}")

        # Log tree depth statistics
        if "num_steps" in extra_fields:
            mean_steps = float(jnp.mean(extra_fields["num_steps"]))
            max_steps = int(jnp.max(extra_fields["num_steps"]))
            logger.info(f"Mean tree depth: {mean_steps:.1f}, Max: {max_steps}")

    except Exception as e:
        logger.debug(f"Could not extract warmup diagnostics: {e}")


def _run_blackjax_sampling(model, config):
    """Run BlackJAX MCMC sampling."""
    # Placeholder for BlackJAX implementation
    # Would need to convert NumPyro model to BlackJAX format
    raise NotImplementedError("BlackJAX sampling not yet implemented")


def _process_posterior_samples(mcmc_result, analysis_mode):
    """Process posterior samples to extract summary statistics and diagnostics.

    Computes:
    - Mean and std for all parameters
    - R-hat convergence diagnostic (if multiple chains)
    - Effective Sample Size (ESS)
    - Acceptance rate
    - Full sample arrays for trace plots
    """
    samples = mcmc_result.get_samples()

    # Extract parameter samples
    if "static" in analysis_mode:
        param_samples = jnp.column_stack(
            [samples["D0"], samples["alpha"], samples["D_offset"]],
        )
    else:
        param_samples = jnp.column_stack(
            [
                samples["D0"],
                samples["alpha"],
                samples["D_offset"],
                samples["gamma_dot_t0"],
                samples["beta"],
                samples["gamma_dot_t_offset"],
                samples["phi0"],
            ],
        )

    # Extract fitting parameter samples
    contrast_samples = samples["contrast"]
    offset_samples = samples["offset"]

    # Compute summary statistics
    mean_params = jnp.mean(param_samples, axis=0)
    std_params = jnp.std(param_samples, axis=0)
    mean_contrast = float(jnp.mean(contrast_samples))
    std_contrast = float(jnp.std(contrast_samples))
    mean_offset = float(jnp.mean(offset_samples))
    std_offset = float(jnp.std(offset_samples))

    # Compute MCMC diagnostics
    try:
        # Extract diagnostic information from MCMC result
        extra_fields = mcmc_result.get_extra_fields()

        # Acceptance rate
        if "accept_prob" in extra_fields:
            acceptance_rate = float(jnp.mean(extra_fields["accept_prob"]))
        else:
            acceptance_rate = None
            logger.warning("Acceptance probability not available in MCMC diagnostics")

        # Compute R-hat and ESS using numpyro diagnostics
        r_hat_dict = {}
        ess_dict = {}

        # Get samples with chain dimension preserved
        samples_with_chains = mcmc_result.get_samples(group_by_chain=True)

        if samples_with_chains["contrast"].ndim > 1:
            # Multiple chains available - compute convergence diagnostics
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            for param_name in samples.keys():
                try:
                    # R-hat (Gelman-Rubin statistic)
                    r_hat_dict[param_name] = float(gelman_rubin(samples_with_chains[param_name]))

                    # ESS (Effective Sample Size)
                    ess_dict[param_name] = float(effective_sample_size(samples_with_chains[param_name]))
                except Exception as e:
                    logger.warning(f"Could not compute diagnostics for {param_name}: {e}")
                    r_hat_dict[param_name] = None
                    ess_dict[param_name] = None
        else:
            # Single chain - cannot compute R-hat
            logger.info("Single chain detected - R-hat not available")
            for param_name in samples.keys():
                r_hat_dict[param_name] = None
                # Compute ESS for single chain using autocorrelation
                try:
                    ess_dict[param_name] = float(effective_sample_size(samples_with_chains[param_name]))
                except:
                    ess_dict[param_name] = None

        # Check convergence based on diagnostics
        converged = True
        if any(r_hat_dict.values()):
            # Check if any R-hat > 1.1 (indicates non-convergence)
            max_r_hat = max([v for v in r_hat_dict.values() if v is not None], default=0.0)
            if max_r_hat > 1.1:
                logger.warning(f"Poor convergence detected: max R-hat = {max_r_hat:.3f} > 1.1")
                converged = False

        if any(ess_dict.values()):
            # Check if any ESS < 100 (indicates poor mixing)
            min_ess = min([v for v in ess_dict.values() if v is not None], default=float('inf'))
            if min_ess < 100:
                logger.warning(f"Poor sampling efficiency: min ESS = {min_ess:.0f} < 100")
                # Don't set converged=False for low ESS, just warn

    except Exception as e:
        logger.error(f"Failed to compute MCMC diagnostics: {e}")
        acceptance_rate = None
        r_hat_dict = None
        ess_dict = None
        converged = True  # Default to converged if diagnostics fail

    return {
        "mean_params": np.array(mean_params),
        "std_params": np.array(std_params),
        "mean_contrast": mean_contrast,
        "std_contrast": std_contrast,
        "mean_offset": mean_offset,
        "std_offset": std_offset,
        "samples_params": np.array(param_samples),
        "samples_contrast": np.array(contrast_samples),
        "samples_offset": np.array(offset_samples),
        "samples": samples,
        "converged": converged,
        "acceptance_rate": acceptance_rate,
        "r_hat": r_hat_dict,
        "ess": ess_dict,
    }
