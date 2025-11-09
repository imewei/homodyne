"""MCMC + JAX: High-Accuracy Bayesian Analysis for Homodyne v2.1
================================================================

NumPyro/BlackJAX-based MCMC sampling for high-precision parameter estimation
and uncertainty quantification with automatic NUTS/CMC selection.

Key Features
------------
- **Automatic NUTS/CMC Selection**: Tri-criteria OR logic based on dataset characteristics
  - Criterion 1 (Parallelism): num_samples >= min_samples_for_cmc (default: 15)
  - Criterion 2 (Memory): estimated_memory > memory_threshold_pct (default: 30%)
  - Criterion 3 (Large Dataset): dataset_size > large_dataset_threshold (default: 1M)
  - Decision: CMC if (Criterion 1 OR Criterion 2 OR Criterion 3), otherwise NUTS

- **Configuration-Driven Parameter Management**: All parameters loaded from YAML config
  - parameter_space: Bounds and prior distributions
  - initial_values: Starting points for MCMC chains (e.g., from NLSQ results)
  - Automatic fallback to mid-point of bounds if values not specified

- **Full Posterior Sampling**: NumPyro/BlackJAX NUTS with comprehensive diagnostics
  - Unified homodyne model: c2_fitted = contrast * c2_theory + offset
  - Physics-informed priors from ParameterSpace
  - Convergence diagnostics: R-hat, ESS, acceptance rate
  - Auto-retry mechanism with different random seeds (max 3 retries)

- **JAX Acceleration**: Transparent CPU/GPU execution with JIT compilation
  - Single-device NUTS for small datasets (<1M points)
  - Multi-shard CMC for large datasets or many samples
  - Hardware-adaptive selection using HardwareConfig

Workflow
--------
**Recommended: Manual NLSQ → MCMC Workflow**
1. Run NLSQ optimization to get point estimates
2. Manually copy best-fit results to config YAML: `initial_parameters.values`
3. Run MCMC with `--method mcmc` (automatic NUTS/CMC selection)
4. MCMC uses config-loaded values for faster convergence

**Configuration Structure (YAML)**
```yaml
optimization:
  mcmc:
    min_samples_for_cmc: 15        # Parallelism threshold
    memory_threshold_pct: 0.30     # Memory threshold (30%)
    dense_mass_matrix: false       # Diagonal (fast) vs full covariance (accurate)

initial_parameters:
  parameter_names: [D0, alpha, D_offset]
  values: [1234.5, 0.567, 12.34]  # From NLSQ results (manual copy)

parameter_space:
  bounds:
    D0: {min: 100.0, max: 5000.0}
    alpha: {min: 0.1, max: 2.0}
    D_offset: {min: 0.1, max: 100.0}
  priors:
    D0: {type: TruncatedNormal, mu: 1000.0, sigma: 500.0}
    alpha: {type: TruncatedNormal, mu: 1.0, sigma: 0.3}
    D_offset: {type: TruncatedNormal, mu: 10.0, sigma: 5.0}
```

MCMC Philosophy
---------------
- Gold standard for uncertainty quantification
- Full posterior sampling (not just point estimates)
- Essential for critical/publication-quality analysis
- Complements NLSQ for comprehensive Bayesian workflow

Automatic Selection Logic
--------------------------
**NUTS (Single-Device):**
- Fast for small datasets (<1M points)
- Low overhead, single-device execution
- Selected when: ALL criteria fail (num_samples < 15) AND (memory < 30%) AND (dataset_size <= 1M)

**CMC (Multi-Shard):**
- Parallelized for CPU cores or large memory requirements
- ~10-20% overhead but enables unlimited dataset sizes
- Selected when: (num_samples >= 15) OR (memory >= 30%) OR (dataset_size > 1M)

**Examples:**
- 50 phi angles (num_samples=50) → CMC (parallelism criterion)
- 5 phi angles but 10M points (memory>30%) → CMC (memory criterion)
- 3 phi angles, 3M pooled points → CMC (large dataset criterion, JAX broadcasting protection)
- 10 phi angles, 100k points (memory<30%) → NUTS (all criteria fail, minimal overhead)
"""

from __future__ import annotations

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
    from homodyne.core.fitting import ParameterSpace as LegacyParameterSpace
    from homodyne.core.physics_cmc import compute_g1_diffusion, compute_g1_total

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False
    # Fallback if physics_cmc not available
    compute_g1_diffusion = None
    compute_g1_total = None

# New config-driven ParameterSpace (v2.1)
try:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

    HAS_PARAMETER_SPACE = True
except ImportError:
    HAS_PARAMETER_SPACE = False
    ParameterSpace = LegacyParameterSpace if HAS_CORE_MODULES else None
    PriorDistribution = None

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


def _calculate_midpoint_defaults(parameter_space: ParameterSpace) -> dict[str, float]:
    """Calculate mid-point defaults for initial parameter values.

    Used when initial_values is None or null in config. Calculates the mid-point
    between min and max bounds for each parameter.

    Parameters
    ----------
    parameter_space : ParameterSpace
        Parameter space containing bounds

    Returns
    -------
    dict[str, float]
        Dictionary mapping parameter names to mid-point values
    """
    initial_values = {}

    for param_name in parameter_space.parameter_names:
        # Skip contrast and offset (scaling parameters, not physical)
        if param_name in ["contrast", "offset"]:
            continue

        bounds = parameter_space.get_bounds(param_name)
        if bounds is not None:
            min_val, max_val = bounds
            midpoint = (min_val + max_val) / 2.0
            initial_values[param_name] = midpoint

    return initial_values


@log_performance(threshold=10.0)
def fit_mcmc_jax(
    data: np.ndarray,
    sigma: np.ndarray | None = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "static_isotropic",
    parameter_space: ParameterSpace | None = None,
    initial_values: dict[str, float] | None = None,
    enable_dataset_optimization: bool = True,
    estimate_noise: bool = False,
    noise_model: str = "hierarchical",
    use_simplified_likelihood: bool = True,
    **kwargs,
) -> MCMCResult:
    """High-accuracy Bayesian parameter estimation using MCMC with automatic NUTS/CMC selection.

    Performs full posterior sampling using NumPyro/BlackJAX with the unified homodyne
    correlation model. Automatically selects between standard NUTS (single-device) and
    Consensus Monte Carlo (multi-shard parallelization) based on dual-criteria OR logic:
    (num_samples >= min_samples_for_cmc) OR (estimated_memory > memory_threshold_pct).

    Configuration-Driven Parameter Management
    ------------------------------------------
    Parameters and priors are loaded from YAML configuration files via:
    - **parameter_space**: Bounds and prior distributions (from `parameter_space` YAML section)
    - **initial_values**: Starting points for MCMC chains (from `initial_parameters.values` YAML section)

    If not provided, defaults are loaded from package configuration:
    - Mid-point of parameter bounds for initial values
    - Physics-informed TruncatedNormal priors with wide distributions

    Manual NLSQ → MCMC Workflow:
    1. Run NLSQ optimization to get point estimates
    2. Manually copy best-fit results to config YAML: `initial_parameters.values`
    3. Run MCMC with initialized values for faster convergence

    Automatic NUTS/CMC Selection
    -----------------------------
    Selection uses dual-criteria OR logic (configurable via YAML):

    **Criterion 1 - Parallelism**: `num_samples >= min_samples_for_cmc` (default: 15)
    - Many independent samples (e.g., 50 phi angles) → CMC for CPU parallelization
    - Achieves ~3x speedup on 14-core CPU with 50 samples

    **Criterion 2 - Memory**: `estimated_memory > memory_threshold_pct` (default: 0.30)
    - Large datasets approaching OOM threshold → CMC for memory management
    - Prevents out-of-memory failures on datasets >1M points

    **Decision**: CMC if (Criterion 1 OR Criterion 2), otherwise NUTS

    Configuration (in YAML):
    ```yaml
    optimization:
      mcmc:
        min_samples_for_cmc: 15        # Parallelism threshold
        memory_threshold_pct: 0.30     # Memory threshold (30%)
        dense_mass_matrix: false       # Diagonal (fast) vs full covariance (accurate)
    ```

    Parameters
    ----------
    data : np.ndarray
        Experimental correlation data (flattened or 2D array)
    sigma : np.ndarray, optional
        Noise standard deviations. If None, estimated as 1% of data.
    t1 : np.ndarray
        First delay time array (same shape as data)
    t2 : np.ndarray
        Second delay time array (same shape as data)
    phi : np.ndarray
        Azimuthal angle values (same shape as data)
    q : float
        Scattering wavevector magnitude [Å⁻¹]
    L : float
        Sample-detector distance [Å] (required for laminar_flow mode)
    analysis_mode : str, default "static_isotropic"
        Physics model selection:
        - "static_isotropic": Diffusion-only (3 physical params)
        - "laminar_flow": Diffusion + shear flow (7 physical params)
    parameter_space : ParameterSpace, optional
        Parameter bounds and prior distributions. If None, loaded from config or defaults.
        Loaded from YAML: `parameter_space` section
    initial_values : dict[str, float], optional
        Initial parameter values for MCMC chains (e.g., from NLSQ results).
        If None, defaults to mid-point of parameter bounds.
        Loaded from YAML: `initial_parameters.values` section
        Example: {'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}
    enable_dataset_optimization : bool, default True
        Deprecated parameter, kept for backward compatibility. No effect.
    estimate_noise : bool, default False
        Enable hierarchical noise estimation (experimental feature)
    noise_model : str, default "hierarchical"
        Noise model type for hierarchical estimation
    use_simplified_likelihood : bool, default True
        Use simplified likelihood for faster computation
    **kwargs
        Additional MCMC configuration parameters:
        - n_samples : int, default 1000
            Number of posterior samples per chain
        - n_warmup : int, default 500
            Number of warmup iterations for adaptation
        - n_chains : int, default 4
            Number of parallel MCMC chains
        - target_accept_prob : float, default 0.8
            Target acceptance probability for step size adaptation
        - max_tree_depth : int, default 10
            Maximum NUTS tree depth
        - rng_key : int, default 42
            Random seed for reproducibility
        - min_samples_for_cmc : int, default 15
            Parallelism threshold for CMC selection (from YAML config)
        - memory_threshold_pct : float, default 0.30
            Memory threshold (30%) for CMC selection (from YAML config)
        - dense_mass_matrix : bool, default False
            Use full covariance (True) vs diagonal (False) mass matrix
        - cmc_config : dict, optional
            CMC-specific configuration (sharding, backend, diagnostics)

    Returns
    -------
    MCMCResult
        MCMC sampling result with full posterior samples and diagnostics.

        Standard NUTS results:
        - mean_params, std_params : Parameter means and standard deviations
        - samples_params : Full posterior samples (n_samples × n_params)
        - r_hat : Gelman-Rubin convergence diagnostic (per parameter)
        - effective_sample_size : ESS diagnostic (per parameter)
        - acceptance_rate : Mean acceptance probability

        CMC results (additional fields):
        - per_shard_diagnostics : List of per-shard convergence info
        - cmc_diagnostics : Overall CMC diagnostics
        - combination_method : Method used to combine posteriors
        - num_shards : Number of shards used for parallelization

    Raises
    ------
    ImportError
        If NumPyro or BlackJAX not available
    ValueError
        If data validation fails (None/empty data, missing arrays, invalid parameters)
    RuntimeError
        If MCMC fails to converge after auto-retry attempts (max 3 retries)

    Examples
    --------
    **Example 1: Config-driven workflow with automatic selection**

    >>> from homodyne.config import ConfigManager
    >>> from homodyne.config.parameter_space import ParameterSpace
    >>>
    >>> # Load configuration from YAML
    >>> config = ConfigManager.from_yaml("config.yaml")
    >>> param_space = ParameterSpace.from_config(config.to_dict())
    >>> initial_vals = config.get_initial_parameters()
    >>>
    >>> # Run MCMC with automatic NUTS/CMC selection
    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     analysis_mode='laminar_flow',
    ...     parameter_space=param_space,
    ...     initial_values=initial_vals,
    ... )
    >>> print(f"Used method: {'CMC' if hasattr(result, 'num_shards') else 'NUTS'}")
    >>> print(f"Convergence: R-hat={result.r_hat}, ESS={result.effective_sample_size}")

    **Example 2: Manual NLSQ → MCMC workflow**

    >>> # Step 1: Run NLSQ for point estimates
    >>> nlsq_result = fit_nlsq_jax(data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5)
    >>> print(f"NLSQ results: D0={nlsq_result.params[0]:.2f}, alpha={nlsq_result.params[1]:.3f}")
    >>>
    >>> # Step 2: Manually update config.yaml with NLSQ results
    >>> # Edit initial_parameters.values:
    >>> #   values: [1234.5, 0.567, 12.34]  # From NLSQ output
    >>>
    >>> # Step 3: Run MCMC with initialized values from config
    >>> config = ConfigManager.from_yaml("config.yaml")  # Reload after manual edit
    >>> initial_vals = config.get_initial_parameters()
    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     initial_values=initial_vals,  # From NLSQ via config
    ... )

    **Example 3: CLI override of automatic selection thresholds**

    >>> # Override config thresholds via CLI arguments
    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     min_samples_for_cmc=20,      # Increase parallelism threshold
    ...     memory_threshold_pct=0.40,   # Increase memory threshold
    ... )

    **Example 4: Dense mass matrix for difficult posteriors**

    >>> # Use full covariance mass matrix (slower but more accurate)
    >>> result = fit_mcmc_jax(
    ...     data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5,
    ...     dense_mass_matrix=True,
    ...     n_warmup=1000,  # More warmup for mass matrix adaptation
    ... )

    Notes
    -----
    **Automatic Selection Logic:**
    - Dual-criteria OR logic ensures optimal performance across all scenarios
    - Parallelism criterion: Many samples (e.g., 50 phi angles) trigger CMC for speedup
    - Memory criterion: Large datasets (>30% memory) trigger CMC to prevent OOM
    - Hardware-adaptive: Uses `HardwareConfig` for GPU memory, CPU cores, cluster detection

    **Configuration Structure:**
    - All parameters loaded from YAML configuration files
    - Three-tier priority: CLI args > config file > package defaults
    - See YAML templates: homodyne_static.yaml, homodyne_laminar_flow.yaml

    **Convergence Diagnostics:**
    - Auto-retry mechanism: Up to 3 attempts with different random seeds
    - R-hat < 1.1 indicates good convergence (Gelman-Rubin diagnostic)
    - ESS > 100 indicates sufficient effective sample size
    - Acceptance rate ~0.7-0.9 indicates good step size tuning

    **Performance:**
    - NUTS: Fast for small datasets (<1M points), single-device execution
    - CMC: Parallelized for large datasets or many samples, ~10-20% overhead
    - Dense mass matrix: 2-3x slower but better for correlated parameters

    See Also
    --------
    ConfigManager.get_initial_parameters : Load initial values from config
    ParameterSpace.from_config : Load parameter space from config
    fit_nlsq_jax : NLSQ optimization for point estimates
    """

    # Validate dependencies
    if not NUMPYRO_AVAILABLE and not BLACKJAX_AVAILABLE:
        raise ImportError(
            "NumPyro or BlackJAX is required for MCMC optimization. "
            "Install with: pip install numpyro blackjax",
        )

    if not HAS_CORE_MODULES:
        raise ImportError("Core homodyne modules are required for optimization")

    # Validate input data
    _validate_mcmc_data(data, t1, t2, phi, q, L)

    # =========================================================================
    # CONFIG-DRIVEN PARAMETER LOADING
    # =========================================================================
    # Load parameter_space and initial_values from config if not provided
    # This implements three-tier priority: CLI args > config file > package defaults
    # - parameter_space: Bounds and prior distributions (from YAML parameter_space section)
    # - initial_values: Starting points for MCMC chains (from YAML initial_parameters.values)

    # Extract config dict once from kwargs (avoid duplicate pop)
    config_dict = kwargs.pop("config", None)

    # Step 1: Load parameter_space from config if None
    # Contains parameter bounds and prior distributions for Bayesian sampling
    if parameter_space is None:
        logger.info("Loading parameter_space from config (not provided as argument)")
        try:
            if config_dict is not None:
                # Load ParameterSpace from config dict
                parameter_space = ParameterSpace.from_config(config_dict)
                logger.info(
                    f"Loaded parameter_space from config: "
                    f"model={parameter_space.model_type}, "
                    f"num_params={len(parameter_space.parameter_names)}"
                )
            else:
                # Fallback to package defaults
                logger.info("No config provided, using package default parameter_space")
                parameter_space = ParameterSpace.from_defaults(analysis_mode)

        except Exception as e:
            logger.warning(
                f"Failed to load parameter_space from config: {e}. "
                "Using package defaults."
            )
            parameter_space = ParameterSpace.from_defaults(analysis_mode)
    else:
        logger.info(
            f"Using provided parameter_space: "
            f"model={parameter_space.model_type}, "
            f"num_params={len(parameter_space.parameter_names)}"
        )

    # Step 2: Load initial_values from config if None
    if initial_values is None:
        logger.info("Loading initial_values from config (not provided as argument)")
        try:
            if config_dict is not None:
                # Load initial values from config using ConfigManager
                from homodyne.config.manager import ConfigManager

                config_mgr = ConfigManager(config_override=config_dict)
                initial_values = config_mgr.get_initial_parameters()

                if initial_values:
                    logger.info(
                        f"Loaded initial_values from config: "
                        f"{list(initial_values.keys())} = "
                        f"{[f'{v:.4g}' for v in initial_values.values()]}"
                    )
                else:
                    # Calculate mid-point defaults
                    logger.info(
                        "Config has null initial_values, calculating mid-point defaults"
                    )
                    initial_values = _calculate_midpoint_defaults(parameter_space)
                    logger.info(
                        f"Using mid-point defaults: "
                        f"{list(initial_values.keys())} = "
                        f"{[f'{v:.4g}' for v in initial_values.values()]}"
                    )
            else:
                # No config, calculate mid-point defaults
                logger.info(
                    "No config provided, calculating mid-point defaults for initial_values"
                )
                initial_values = _calculate_midpoint_defaults(parameter_space)
                logger.info(
                    f"Using mid-point defaults: "
                    f"{list(initial_values.keys())} = "
                    f"{[f'{v:.4g}' for v in initial_values.values()]}"
                )

        except Exception as e:
            logger.warning(
                f"Failed to load initial_values from config: {e}. "
                "Using mid-point defaults."
            )
            initial_values = _calculate_midpoint_defaults(parameter_space)
    else:
        logger.info(
            f"Using provided initial_values: "
            f"{list(initial_values.keys())} = "
            f"{[f'{v:.4g}' for v in initial_values.values()]}"
        )

    # Step 3: Validate loaded parameters
    # Verify initial_values are within parameter_space bounds
    if initial_values is not None:
        is_valid, violations = parameter_space.validate_values(initial_values)
        if not is_valid:
            raise ValueError(
                f"Initial parameter values violate bounds:\n" + "\n".join(violations)
            )
        logger.info(
            "Initial parameter values validated successfully (all within bounds)"
        )

    # Get total dataset size (handles both 1D and multi-dimensional arrays)
    dataset_size = data.size if hasattr(data, "size") else len(data)

    # Get number of independent samples for NUTS/CMC decision
    # For multi-dimensional XPCS data (n_phi, n_t1, n_t2), each phi angle is one sample
    # For 1D flattened data, count unique phi angles from phi parameter
    if hasattr(data, "shape") and hasattr(data, "ndim"):
        if data.ndim > 1:
            # Multi-dimensional: first dimension is number of samples (phi angles)
            num_samples = data.shape[0]
        else:
            # 1D flattened data: count unique phi angles from phi parameter
            import numpy as np

            num_samples = len(np.unique(phi)) if phi is not None else dataset_size
    else:
        # Fallback for non-array data: count unique phi angles if available
        import numpy as np

        num_samples = len(np.unique(phi)) if phi is not None else dataset_size

    logger.info("Starting MCMC+JAX sampling")
    logger.info(
        f"Dataset size: {dataset_size:,} data points ({num_samples} independent samples)"
    )
    logger.info(f"Analysis mode: {analysis_mode}")

    # =========================================================================
    # AUTOMATIC NUTS/CMC SELECTION - DUAL-CRITERIA OR LOGIC
    # =========================================================================
    # Step 1: Detect hardware configuration
    # Hardware detection provides memory information for dual-criteria decision
    try:
        from homodyne.device.config import detect_hardware, should_use_cmc

        hardware_config = detect_hardware()
    except ImportError:
        logger.warning(
            "Hardware detection not available. Using simple threshold-based selection."
        )
        hardware_config = None

    # Step 2: Extract configurable thresholds from kwargs (with defaults)
    # These thresholds are loaded from YAML config: optimization.mcmc.min_samples_for_cmc
    # Users can override via CLI: --min-samples-cmc, --memory-threshold-pct, --large-dataset-threshold
    min_samples_for_cmc = kwargs.pop("min_samples_for_cmc", 15)
    memory_threshold_pct = kwargs.pop("memory_threshold_pct", 0.30)
    large_dataset_threshold = kwargs.pop("large_dataset_threshold", 1_000_000)

    # Step 3: Automatic NUTS/CMC selection using tri-criteria OR logic
    # Criterion 1 (Parallelism): num_samples >= min_samples_for_cmc (default: 15)
    #   - Many independent samples (e.g., 50 phi angles) → CMC for CPU parallelization
    #   - Achieves ~3x speedup on multi-core CPUs with many samples
    # Criterion 2 (Memory): estimated_memory > memory_threshold_pct (default: 30%)
    #   - Large datasets approaching OOM threshold → CMC for memory management
    #   - Prevents out-of-memory failures on datasets >1M points
    # Criterion 3 (Large Dataset): dataset_size > large_dataset_threshold (default: 1M)
    #   - Very large pooled datasets → CMC to prevent JAX broadcasting overflow
    #   - Critical for pooled data causing (3M, 3M, 3M) array creation
    # Decision: use_cmc = (Criterion 1 OR Criterion 2 OR Criterion 3)
    #   - Any criterion triggers CMC (OR logic, not AND)
    if hardware_config is not None:
        # Hardware detection available - use full tri-criteria logic
        use_cmc = should_use_cmc(
            num_samples,
            hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=min_samples_for_cmc,
            memory_threshold_pct=memory_threshold_pct,
            large_dataset_threshold=large_dataset_threshold,
        )
        actual_method = "cmc" if use_cmc else "nuts"
        logger.info(
            f"Automatic selection: {actual_method.upper()} "
            f"(num_samples={num_samples:,}, dataset_size={dataset_size:,}, "
            f"thresholds: min_samples={min_samples_for_cmc}, memory={memory_threshold_pct:.1%}, "
            f"platform={hardware_config.platform})"
        )
    else:
        # Fallback: Simple threshold-based selection using num_samples only
        use_cmc = num_samples >= min_samples_for_cmc
        actual_method = "cmc" if use_cmc else "nuts"
        logger.info(
            f"Automatic selection (fallback): {actual_method.upper()} "
            f"(num_samples={num_samples:,}, min_samples_for_cmc={min_samples_for_cmc})"
        )

    # Step 4: Log warnings for edge cases
    if actual_method == "cmc" and num_samples < min_samples_for_cmc:
        logger.warning(
            f"Using CMC with very few samples ({num_samples} samples). "
            f"CMC adds 10-20% overhead; NUTS is faster for <{min_samples_for_cmc} samples if memory permits. "
            f"(Likely triggered by memory criterion: estimated_memory > {memory_threshold_pct:.1%})"
        )
    elif actual_method == "nuts" and num_samples >= min_samples_for_cmc:
        logger.info(
            f"Using NUTS with {num_samples:,} samples. "
            f"CMC may provide additional parallelization on multi-core CPU."
        )

    # Step 5: Execute selected method
    if actual_method == "cmc":
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
        cmc_config = kwargs.pop("cmc_config", {})

        # Add MCMC config to cmc_config if not already present
        if "mcmc" not in cmc_config:
            cmc_config["mcmc"] = _get_mcmc_config(kwargs)

        # Create CMC coordinator
        coordinator = CMCCoordinator(cmc_config)

        # Run CMC pipeline with config-driven parameters
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            parameter_space=parameter_space,
            initial_values=initial_values,
        )

        logger.info(f"CMC execution completed. Used {result.num_shards} shards.")
        return result

    else:
        # Use standard NUTS with automatic retry on convergence failure
        logger.info("=" * 70)
        logger.info("Executing standard NUTS MCMC")
        logger.info("=" * 70)

        # Implement auto-retry mechanism for poor convergence
        max_retries = 3
        for attempt in range(max_retries):
            # Use different random seed for each retry
            retry_kwargs = kwargs.copy()
            if attempt > 0:
                # Change random seed for retry attempts
                base_seed = kwargs.get("rng_key", 42)
                new_seed = base_seed + attempt * 1000
                retry_kwargs["rng_key"] = new_seed

                # Enhanced logging with diagnostics from previous attempt
                # Get diagnostics from previous attempt (if available)
                if "result" in locals():
                    max_rhat = (
                        max(
                            [v for v in result.r_hat.values() if v is not None],
                            default=0.0,
                        )
                        if result.r_hat is not None
                        else 0.0
                    )
                    min_ess = (
                        min(
                            [
                                v
                                for v in result.effective_sample_size.values()
                                if v is not None
                            ],
                            default=0.0,
                        )
                        if result.effective_sample_size is not None
                        else 0.0
                    )
                    logger.warning(
                        f"🔄 Retry {attempt}/{max_retries - 1} - Convergence poor "
                        f"(R-hat={max_rhat:.3f}, ESS={min_ess:.0f}). "
                        f"Changing random seed to {new_seed}..."
                    )
                else:
                    logger.warning(
                        f"🔄 Retry {attempt}/{max_retries - 1} with new random seed "
                        f"(seed={new_seed})"
                    )

            result = _run_standard_nuts(
                data=data,
                sigma=sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                analysis_mode=analysis_mode,
                parameter_space=parameter_space,
                initial_values=initial_values,
                use_simplified_likelihood=use_simplified_likelihood,
                **retry_kwargs,
            )

            # Check convergence quality
            if result.converged:
                # Check R-hat if available
                poor_rhat = False
                if result.r_hat is not None:
                    max_rhat = max(
                        [v for v in result.r_hat.values() if v is not None], default=0.0
                    )
                    if max_rhat > 1.1:
                        poor_rhat = True
                        logger.warning(
                            f"Poor convergence detected: max R-hat = {max_rhat:.3f} > 1.1"
                        )

                # Check ESS if available
                poor_ess = False
                if result.effective_sample_size is not None:
                    min_ess = min(
                        [
                            v
                            for v in result.effective_sample_size.values()
                            if v is not None
                        ],
                        default=float("inf"),
                    )
                    if min_ess < 100:
                        poor_ess = True
                        logger.warning(
                            f"Low effective sample size: min ESS = {min_ess:.0f} < 100"
                        )

                # Retry if convergence is poor and we have retries left
                if (poor_rhat or poor_ess) and attempt < max_retries - 1:
                    logger.warning(
                        f"Convergence quality poor. Retrying with different random seed..."
                    )
                    continue  # Retry
                else:
                    # Either convergence is good, or we're out of retries
                    if poor_rhat or poor_ess:
                        # All retries failed - provide actionable suggestions
                        logger.warning(
                            f"❌ All {max_retries} retry attempts failed. "
                            f"Final diagnostics: R-hat={max_rhat:.3f}, ESS={min_ess:.0f}. "
                            f"Consider: (1) Increase n_warmup/n_samples, "
                            f"(2) Reparameterize model, (3) Check data quality"
                        )
                    elif attempt > 0:
                        # Retry was successful - show improved diagnostics
                        logger.info(
                            f"✅ Retry {attempt} successful - Convergence improved "
                            f"(R-hat={max_rhat:.3f}, ESS={min_ess:.0f})"
                        )
                    return result
            else:
                # Convergence failed completely
                if attempt < max_retries - 1:
                    logger.warning(
                        f"🔄 Retry {attempt + 1}/{max_retries - 1} - MCMC did not converge. "
                        f"Retrying with different random seed..."
                    )
                    continue
                else:
                    logger.error(
                        f"❌ All {max_retries} retry attempts failed - MCMC did not converge. "
                        f"Consider: (1) Increase n_warmup/n_samples, "
                        f"(2) Reparameterize model, (3) Check data quality"
                    )
                    return result

        # Should never reach here, but return last result as fallback
        return result


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
    initial_values: dict[str, float] | None = None,
    use_simplified_likelihood: bool = True,
    **kwargs,
) -> MCMCResult:
    """Execute standard NUTS MCMC sampling.

    Internal helper function for single-device NUTS execution.
    Used when automatic selection chooses NUTS based on dual-criteria logic.

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
        Parameter bounds and priors (loaded from config)
    initial_values : dict, optional
        Initial parameter values for MCMC chains (loaded from config)
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

        # =====================================================================
        # PRE-COMPUTATION BEFORE JIT TRACING
        # =====================================================================
        # Pre-compute dt and phi_unique BEFORE JIT compilation to avoid JAX concretization errors
        # This fixes: "Abstract tracer value encountered where concrete value is expected"
        # when jnp.unique() is called during NumPyro's JIT tracing

        # Pre-compute dt
        dt_computed = None
        if t1 is not None:
            import numpy as np  # Use numpy (not jax.numpy) for pre-computation

            if t1.ndim == 2:
                time_array = np.asarray(t1[:, 0] if t1.shape[1] > 0 else t1[0, :])
            else:
                time_array = np.asarray(t1)
            # Estimate from first two unique time points
            unique_times = np.unique(time_array)
            if len(unique_times) > 1:
                dt_computed = float(unique_times[1] - unique_times[0])
            else:
                dt_computed = 1.0  # Fallback
            logger.debug(f"Pre-computed dt = {dt_computed:.6f} s for MCMC model")

        # Pre-compute phi_unique (CRITICAL FIX for JAX concretization error)
        # CMC pooled data replicates phi for each time point (e.g., 3 angles × 100K points = 300K array)
        # We need unique phi values for broadcasting, but jnp.unique() doesn't work during JIT tracing
        # Extract unique values HERE (non-JIT context) and pass to model as closure variable
        phi_unique = None
        if phi is not None and "laminar" in analysis_mode.lower():
            import numpy as np  # Use numpy (not jax.numpy) for pre-computation

            phi_unique = np.unique(np.asarray(phi))
            logger.debug(
                f"Pre-computed phi_unique: {len(phi_unique)} unique angles from {len(phi)} total values "
                f"(reduction: {len(phi) / len(phi_unique):.1f}x)"
            )

        # Create NumPyro model using physics-informed priors from ParameterSpace
        # Pass phi_unique (if available) instead of full phi array to avoid JIT tracing jnp.unique()
        # Also pass phi_full for per-angle contrast/offset scaling
        model = _create_numpyro_model(
            data,
            sigma,
            t1,
            t2,
            (
                phi_unique if phi_unique is not None else phi
            ),  # Use pre-computed unique values
            q,
            L,
            analysis_mode,
            parameter_space,
            use_simplified=use_simplified_likelihood,
            dt=dt_computed,
            phi_full=phi,  # Full replicated phi array for per-angle scaling mapping
            per_angle_scaling=True,  # Enable per-angle contrast/offset
        )

        # Log initial values if provided (for MCMC chain initialization)
        if initial_values is not None:
            logger.info(
                f"Initializing MCMC chains with values: "
                f"{list(initial_values.keys())} = "
                f"{[f'{v:.4g}' for v in initial_values.values()]}"
            )
        else:
            logger.info("MCMC chains will use default initialization (NumPyro random)")

        # Run MCMC sampling
        if NUMPYRO_AVAILABLE:
            result = _run_numpyro_sampling(model, mcmc_config, initial_values)
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
    """Get MCMC configuration with optimized defaults.

    Parameters
    ----------
    kwargs : dict
        Configuration overrides from YAML config or function arguments.
        Supported keys:
        - n_samples : int
            Number of MCMC samples per chain
        - n_warmup : int
            Number of warmup samples for NUTS adaptation
        - n_chains : int
            Number of parallel MCMC chains
        - target_accept_prob : float
            Target acceptance probability for NUTS (0.8 recommended)
        - max_tree_depth : int
            Maximum NUTS tree depth (10 default)
        - dense_mass_matrix : bool
            Use full covariance mass matrix (slower, more accurate)
        - rng_key : int
            Random seed for reproducibility

    Returns
    -------
    dict
        Complete MCMC configuration dictionary

    Notes
    -----
    **Mass Matrix Options:**

    - **dense_mass_matrix=False (default)**: Uses diagonal mass matrix
      - Faster: O(d) complexity for d parameters
      - Less memory: stores only d diagonal elements
      - Good for: weakly correlated posteriors, >10 parameters
      - Recommended for: production use, large parameter spaces

    - **dense_mass_matrix=True**: Uses full covariance mass matrix
      - Slower: O(d²) complexity for matrix operations
      - More memory: stores d×d matrix elements
      - Better for: highly correlated posteriors, <10 parameters
      - Recommended for: research, difficult convergence cases

    The mass matrix is learned during warmup via adaptation and affects
    the proposal distribution geometry in HMC/NUTS sampling.
    """
    default_config = {
        "n_samples": 1000,
        "n_warmup": 500,  # Reduced warmup for faster testing
        "n_chains": 4,  # Enable parallel chains by default
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
        "dense_mass_matrix": False,  # Diagonal mass matrix (faster)
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
    parameter_space,
    use_simplified=True,
    dt=None,
    phi_full=None,
    per_angle_scaling=True,  # Default True: Physically correct per-angle scaling
):
    """Create NumPyro probabilistic model using config-driven priors.

    This function creates a NumPyro model with prior distributions dynamically
    generated from the ParameterSpace configuration. It supports multiple
    distribution types (Normal, TruncatedNormal, Uniform, LogNormal) and
    automatically handles parameter ordering for different analysis modes.

    **NEW (Nov 2025): Per-angle contrast and offset parameters**

    When per_angle_scaling=True, contrast and offset are sampled as arrays of
    shape (n_phi,) instead of scalars, allowing different scaling parameters
    for each phi angle. This requires phi_full to create the mapping from
    data points to phi angles.

    Parameters
    ----------
    data : array
        Experimental correlation data to fit
    sigma : array
        Noise standard deviations for likelihood
    t1, t2 : array
        Time delay arrays
    phi : array
        Azimuthal angle array (pre-computed unique values for laminar_flow)
    q : float
        Wavevector magnitude
    L : float
        Sample-detector distance (for laminar_flow)
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'
    parameter_space : ParameterSpace
        Parameter space configuration with bounds and prior distributions
        loaded from YAML config file. Contains:
        - parameter_names: list of parameter names
        - bounds: dict mapping param_name -> (min, max)
        - priors: dict mapping param_name -> PriorDistribution
    use_simplified : bool, default=True
        Use simplified likelihood for faster computation
    dt : float, optional
        Time step (pre-computed to avoid JAX concretization errors)
    phi_full : array, optional
        Full replicated phi array matching data length (required for per_angle_scaling)
        Example: [0, 0, 0, ..., 60, 60, 60, ..., 120, 120, 120, ...]
        If None and per_angle_scaling=True, will use phi for mapping (assumes pre-sorted data)
    per_angle_scaling : bool, default=True
        If True (default), sample contrast and offset as arrays of shape (n_phi,) for per-angle scaling.
        This is the physically correct behavior as each scattering angle can have different
        optical properties and detector responses.

        If False, use legacy behavior with scalar contrast and offset (shared across all angles).
        This mode is provided for backward compatibility testing only and is not recommended
        for production analysis.

    Returns
    -------
    callable
        NumPyro model function for MCMC sampling

    Notes
    -----
    **CRITICAL FIX (Nov 2025): Pre-computed phi_unique**

    The phi parameter passed to this function should be pre-computed unique values
    (not the full replicated array) when using laminar_flow mode. This avoids
    JAX concretization errors when NumPyro JIT-traces the model.

    The caller (_run_standard_nuts) must pre-compute:
    ```python
    phi_unique = np.unique(np.asarray(phi))
    model = _create_numpyro_model(..., phi=phi_unique, ...)
    ```

    This ensures jnp.unique() is never called during JIT tracing, which would
    cause: "Abstract tracer value encountered where concrete value is expected"

    **Config-Driven Prior Creation:**

    Priors are created dynamically from the ParameterSpace object, which is
    loaded from the YAML config file. Example config structure:

    .. code-block:: yaml

        parameter_space:
          model: static
          bounds:
            - name: D0
              min: 100.0
              max: 100000.0
              prior_mu: 1000.0
              prior_sigma: 1000.0
              type: TruncatedNormal
            - name: alpha
              min: -2.0
              max: 2.0
              prior_mu: -1.2
              prior_sigma: 0.3
              type: Normal

    **Supported Prior Distribution Types:**

    - **TruncatedNormal**: Normal distribution truncated to [min, max] bounds
      (recommended for most parameters with finite support)
    - **Normal**: Standard normal distribution (use for unbounded parameters)
    - **Uniform**: Uniform distribution over [min, max] (non-informative)
    - **LogNormal**: Log-normal distribution (for strictly positive parameters)

    **Prior Selection Guidelines:**

    - Use TruncatedNormal for physical parameters with hard bounds (D0, alpha)
    - Use Normal for parameters without physical constraints
    - Use Uniform for completely non-informative priors
    - Use LogNormal for strictly positive scale parameters (rare in XPCS)

    **Physics-Informed Defaults:**

    If config does not specify priors, the ParameterSpace class provides
    scientifically validated defaults based on XPCS domain knowledge:
    - D0: TruncatedNormal centered at mid-range with wide sigma
    - alpha: TruncatedNormal centered at -1.2 (typical homodyne value)
    - Flow parameters: TruncatedNormal with physically reasonable ranges

    **Parameter Ordering:**

    The function automatically orders parameters correctly for the physics
    engine based on analysis_mode:
    - static_isotropic: [contrast, offset, D0, alpha, D_offset]
    - laminar_flow: [contrast, offset, D0, alpha, D_offset,
                     gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    Examples
    --------
    >>> # Load parameter space from config
    >>> from homodyne.config.parameter_space import ParameterSpace
    >>> param_space = ParameterSpace.from_config(config_dict)
    >>>
    >>> # Pre-compute phi_unique (CRITICAL for laminar_flow)
    >>> phi_unique = np.unique(np.asarray(phi))
    >>>
    >>> # Create model
    >>> model = _create_numpyro_model(
    ...     data, sigma, t1, t2, phi_unique, q, L,
    ...     analysis_mode='laminar_flow',
    ...     parameter_space=param_space
    ... )
    >>>
    >>> # Use in MCMC sampling
    >>> from numpyro.infer import MCMC, NUTS
    >>> nuts_kernel = NUTS(model)
    >>> mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
    >>> mcmc.run(jax.random.PRNGKey(0))

    See Also
    --------
    homodyne.config.parameter_space.ParameterSpace : Parameter space configuration
    homodyne.config.parameter_space.PriorDistribution : Prior distribution specs
    """
    # Map distribution type names to NumPyro distribution classes
    DIST_TYPE_MAP = {
        "Normal": dist.Normal,
        "TruncatedNormal": dist.TruncatedNormal,
        "Uniform": dist.Uniform,
        "LogNormal": dist.LogNormal,
    }

    # =========================================================================
    # PRE-COMPUTE PHI VALUES BEFORE JIT TRACING (CRITICAL FIX)
    # =========================================================================
    # CRITICAL FIX (Nov 2025): Pre-compute phi_unique BEFORE model function definition
    # to avoid JAX concretization error during NumPyro's JIT tracing.
    #
    # The model function will be JIT-compiled by NumPyro, so any values that need to be
    # concrete (actual values, not abstract tracers) must be computed BEFORE the model
    # function is defined. These values are captured by the closure.
    #
    # Using numpy (not jax.numpy) ensures we get concrete values:
    phi_array_for_mapping = phi_full if phi_full is not None else phi
    phi_unique_for_sampling = np.unique(np.asarray(phi))  # numpy, not jnp!
    n_phi = len(phi_unique_for_sampling)  # Python int, not JAX tracer

    def homodyne_model():
        """Inner NumPyro model function with config-driven priors."""
        # Import parameter name constants to ensure consistency
        from homodyne.config.parameter_names import get_parameter_names

        # Determine parameter list based on analysis mode
        # Note: Scaling parameters (contrast, offset) are always first
        # Using centralized parameter names from homodyne.config.parameter_names
        # to prevent mismatches with sample extraction logic
        param_names_ordered = get_parameter_names(analysis_mode)

        # =====================================================================
        # DYNAMIC PRIOR SAMPLING FROM CONFIG
        # =====================================================================
        # NEW (Nov 2025): Per-angle contrast and offset sampling
        # Note: phi_array_for_mapping, phi_unique_for_sampling, and n_phi are
        # pre-computed above (before model definition) and captured by closure

        # Sample parameters dynamically using config-driven priors
        # Loop through all parameters in correct order (scaling + physics)
        sampled_params = []
        for i, param_name in enumerate(param_names_ordered):
            # Get prior distribution from parameter space (loaded from YAML config)
            # Handle missing scaling parameters (contrast, offset) with fallback defaults
            try:
                prior_spec = parameter_space.get_prior(param_name)
            except KeyError:
                # Scaling parameters (contrast, offset) might not be in config
                # Use sensible defaults based on XPCS physics
                if param_name == "contrast":
                    # Contrast typically in [0, 1], centered at 0.5
                    prior_spec = PriorDistribution(
                        dist_type="TruncatedNormal",
                        mu=0.5,
                        sigma=0.2,
                        min_val=0.0,
                        max_val=1.0,
                    )
                elif param_name == "offset":
                    # Offset typically near 1.0 for c2
                    prior_spec = PriorDistribution(
                        dist_type="TruncatedNormal",
                        mu=1.0,
                        sigma=0.2,
                        min_val=0.5,
                        max_val=1.5,
                    )
                else:
                    # Unexpected missing parameter - raise error
                    raise KeyError(
                        f"Parameter '{param_name}' not found in parameter_space "
                        f"and no default available. Available parameters: "
                        f"{list(parameter_space.priors.keys())}"
                    )

            # Get NumPyro distribution class
            dist_class = DIST_TYPE_MAP.get(
                prior_spec.dist_type,
                dist.TruncatedNormal,  # Safe fallback
            )

            # Get distribution kwargs
            dist_kwargs = prior_spec.to_numpyro_kwargs()

            # PER-ANGLE SAMPLING: contrast and offset as separate parameters per phi angle
            if per_angle_scaling and param_name in ["contrast", "offset"]:
                # Sample separate parameters for each phi angle
                # This ensures clean sample extraction and backward compatibility
                param_values = []
                for phi_idx in range(n_phi):
                    # Name: contrast_0, contrast_1, ..., offset_0, offset_1, ...
                    param_name_phi = f"{param_name}_{phi_idx}"
                    param_value_phi = sample(param_name_phi, dist_class(**dist_kwargs))
                    param_values.append(param_value_phi)
                # Stack into array for per-point indexing
                param_value = jnp.array(param_values)
                sampled_params.append(param_value)
            else:
                # Sample scalar parameter (standard behavior for physics params)
                param_value = sample(param_name, dist_class(**dist_kwargs))
                sampled_params.append(param_value)

        # Extract contrast and offset for scaling (always first two parameters)
        # These are now arrays of shape (n_phi,) if per_angle_scaling=True
        contrast = sampled_params[0]
        offset = sampled_params[1]

        # Build params array for physics computation (physics params only, no scaling)
        # Physics parameters start at index 2 (after contrast and offset)
        physics_params = sampled_params[2:]
        params = jnp.array(physics_params)

        # For backward compatibility with physics functions that expect full params array,
        # we need to prepend mean values of contrast and offset
        if per_angle_scaling:
            # Use mean values for physics computation (theory doesn't need per-angle scaling)
            contrast_mean = jnp.mean(contrast)
            offset_mean = jnp.mean(offset)
            params_full = jnp.concatenate(
                [jnp.array([contrast_mean, offset_mean]), params]
            )
        else:
            # Standard behavior: contrast and offset are scalars
            params_full = jnp.concatenate([jnp.array([contrast, offset]), params])

        # =====================================================================
        # THEORETICAL MODEL COMPUTATION
        # =====================================================================
        # Compute theoretical model using JIT-compiled function with proper physics
        # CRITICAL: Ensure t1, t2, phi, sigma arrays match data size to avoid memory issues
        # For pooled data (4,600 samples), these should be 1D arrays of length 4,600
        # NOT the original meshgrids (1,002,001 elements) which would cause OOM

        # Verify array sizes match to prevent memory explosion during MCMC
        data_size = data.shape[0] if hasattr(data, "shape") else len(data)
        t1_size = t1.shape[0] if hasattr(t1, "shape") else len(t1)

        if t1_size != data_size:
            # Arrays are mismatched - this indicates closure is capturing wrong arrays
            # This is the root cause of the 778GB SVI OOM error
            import warnings

            warnings.warn(
                f"Model closure array size mismatch: data={data_size}, t1={t1_size}. "
                f"This will cause OOM during SVI. Ensure pooled_data has correctly sized arrays.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Pass full params array for proper indexing, plus L and dt for correct physics
        # CRITICAL FIX (Nov 2025): Use replicated phi for flattened element-wise data
        # For large datasets (CMC shards with len > 2000), data is flattened element-wise:
        # each data point is paired with corresponding t1, t2, phi values.
        # Physics functions need ALL arrays to have same length to avoid meshgrid creation.
        # For small datasets (len <= 2000), use unique phi values and let physics create meshgrid.
        phi_for_theory = phi_array_for_mapping if len(t1) > 2000 else phi
        c2_theory = _compute_simple_theory_jit(
            params_full, t1, t2, phi_for_theory, q, analysis_mode, L, dt
        )

        # PER-ANGLE SCALING: Apply different contrast/offset for each phi angle
        if per_angle_scaling:
            # Create mapping from data points to phi angle indices
            # phi_array_for_mapping contains the full replicated phi array matching data length
            # phi_unique_for_sampling contains the unique phi values
            #
            # Example:
            #   phi_array_for_mapping = [0, 0, 0, ..., 60, 60, 60, ..., 120, 120, 120]  (300K elements)
            #   phi_unique_for_sampling = [0, 60, 120]  (3 elements)
            #   phi_indices = [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]  (300K elements)
            #
            # Use searchsorted to find the index of each phi value in the unique array
            phi_array_for_mapping_jax = jnp.atleast_1d(phi_array_for_mapping)
            phi_indices = jnp.searchsorted(
                phi_unique_for_sampling, phi_array_for_mapping_jax
            )

            # Select the appropriate contrast and offset for each data point
            # contrast: shape (n_phi,) → contrast_per_point: shape (n_data_points,)
            # offset: shape (n_phi,) → offset_per_point: shape (n_data_points,)
            contrast_per_point = contrast[phi_indices]
            offset_per_point = offset[phi_indices]

            # Apply per-angle scaling
            c2_fitted = contrast_per_point * c2_theory + offset_per_point
        else:
            # LEGACY BEHAVIOR: Global contrast and offset (shared across all angles)
            c2_fitted = contrast * c2_theory + offset

        # Likelihood
        sample("obs", dist.Normal(c2_fitted, sigma), obs=data)

    return homodyne_model


def _compute_simple_theory(params, t1, t2, phi, q, analysis_mode, L=None, dt=None):
    """Compute theoretical c2 using proper physics from core.jax_backend.

    Uses the SAME physics as NLSQ to ensure consistency between methods.
    For laminar_flow: includes diffusion + shear effects via compute_g1_total().
    For static: uses diffusion-only via compute_g1_diffusion().

    Parameters
    ----------
    params : array
        Full parameter array [contrast, offset, D0, alpha, D_offset, ...]
    t1, t2 : arrays
        Time delay arrays
    phi : array
        Angle array (pre-computed unique values for laminar_flow)
    q : scalar
        Wavevector magnitude
    analysis_mode : str
        Analysis mode: 'static_isotropic' or 'laminar_flow'
    L : float, optional
        Sample-detector distance for laminar_flow mode [Å]
    dt : float, optional
        Time step [s] - estimated from t1 if not provided

    Returns
    -------
    array
        Theoretical c2 values

    Notes
    -----
    **CRITICAL FIX (Nov 2025): Pre-computed phi_unique**

    The phi parameter must be pre-computed unique values (not replicated array)
    when using laminar_flow mode. This avoids JAX concretization errors during
    NumPyro's JIT tracing when compute_g1_total() is called.

    The caller must ensure phi contains only unique angles:
    ```python
    phi_unique = np.unique(np.asarray(phi))
    c2_theory = _compute_simple_theory(..., phi=phi_unique, ...)
    ```

    This function now uses the proper physics from core.jax_backend to match
    NLSQ optimization, fixing the NaN ELBO issue caused by physics mismatch.
    """
    # dt must be provided (pre-computed before JIT compilation)
    # Using jnp.unique() here would cause JAX concretization error during JIT tracing
    if dt is None:
        # Fallback for backward compatibility, but this should not happen in normal use
        dt = 1.0
        logger.warning(
            "dt not provided to _compute_simple_theory, using fallback value 1.0"
        )

    # Extract physical parameters (skip contrast and offset at indices 0,1)
    # params = [contrast, offset, D0, alpha, D_offset, ...]
    phys_params = params[2:]  # Physical parameters only

    # Compute g1 using proper physics from core.jax_backend
    if "laminar" in analysis_mode.lower():
        # Laminar flow: use full physics (diffusion + shear)
        if compute_g1_total is None:
            raise ImportError("compute_g1_total not available from core.jax_backend")
        if L is None:
            L = 2000000.0  # Default: 200 µm in Angstroms
            logger.debug(f"Using default L = {L:.1f} Å for laminar_flow model")

        # CRITICAL: phi here should be pre-computed unique values
        # compute_g1_total will NOT call jnp.unique() since phi is already unique
        # This is handled by the pre-computation in _run_standard_nuts()
        g1 = compute_g1_total(
            phys_params,  # [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            t1,
            t2,
            phi,  # Pre-computed unique values (not replicated array)
            q,
            L,
            dt,
        )
    else:
        # Static mode: use diffusion-only physics
        if compute_g1_diffusion is None:
            raise ImportError(
                "compute_g1_diffusion not available from core.jax_backend"
            )

        g1 = compute_g1_diffusion(
            phys_params,  # [D0, alpha, D_offset]
            t1,
            t2,
            q,
            dt,
        )

    # g2 theory: c2 = 1 + |g1|^2
    c2_theory = 1.0 + g1 * g1

    return c2_theory


# ARCHITECTURAL FIX (Nov 2025): Remove @jit to prevent 80GB OOM
# DO NOT JIT THIS FUNCTION - it calls dispatchers with Python conditionals
# NumPyro's internal tracing would compile both branches causing 80GB meshgrid allocation
# The inner element-wise/meshgrid functions are already JIT-compiled for performance
_compute_simple_theory_jit = _compute_simple_theory  # No JIT wrapper


def _run_numpyro_sampling(model, config, initial_values=None):
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
    initial_values : dict[str, float], optional
        Initial parameter values for MCMC chains (e.g., from NLSQ results).
        If provided, chains will be initialized near these values.
        If None, NumPyro uses random initialization from priors.

    Returns
    -------
    mcmc : numpyro.infer.MCMC
        MCMC object with samples and diagnostics

    Notes
    -----
    - Configures parallel chains for CPU/GPU
    - Extracts acceptance probability and divergence info
    - Logs warmup diagnostics for debugging
    - Initial values improve convergence when starting from good point estimates
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
                logger.info(
                    f"Using GPU with {n_chains} chains on {n_devices} device(s)"
                )
            else:
                logger.info(f"Using {min(n_chains, n_devices)} parallel devices")
        except Exception as e:
            logger.warning(f"Could not configure parallel chains: {e}")

    # Create NUTS kernel with diagnostics and config-driven mass matrix
    dense_mass_matrix = config.get("dense_mass_matrix", False)
    nuts_kernel = NUTS(
        model,
        target_accept_prob=config["target_accept_prob"],
        max_tree_depth=config.get("max_tree_depth", 10),
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=dense_mass_matrix,  # Config-driven: False=diagonal, True=full covariance
    )

    # Log mass matrix configuration
    mass_type = "full covariance" if dense_mass_matrix else "diagonal"
    logger.info(f"NUTS mass matrix: {mass_type} (dense_mass={dense_mass_matrix})")

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
            ),
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

    # Extract fitting parameter samples (per-angle scaling ONLY)
    # BREAKING CHANGE (Nov 2025): Legacy scalar contrast/offset removed - not physically meaningful
    # MCMC MUST use per-angle scaling: contrast_0, contrast_1, ..., offset_0, offset_1, ...
    per_angle_scaling_detected = "contrast_0" in samples

    if not per_angle_scaling_detected:
        # ERROR: Per-angle scaling parameters not found
        available_keys = list(samples.keys())
        logger.error(
            "Per-angle scaling parameters not found in MCMC samples. "
            "Expected 'contrast_0', 'contrast_1', ... but found: %s. "
            "Legacy scalar contrast/offset is no longer supported as it is not physically meaningful.",
            available_keys
        )
        raise ValueError(
            "Per-angle scaling MCMC sampling failed: 'contrast_0' parameter not found. "
            "This indicates the NumPyro model did not use per-angle scaling. "
            f"Available parameters: {available_keys}"
        )

    # PER-ANGLE SCALING: Extract all contrast_i and offset_i parameters
    contrast_keys = sorted([k for k in samples.keys() if k.startswith("contrast_")])
    offset_keys = sorted([k for k in samples.keys() if k.startswith("offset_")])

    if not contrast_keys or not offset_keys:
        logger.error(
            "Incomplete per-angle parameters: contrast_keys=%s, offset_keys=%s",
            contrast_keys, offset_keys
        )
        raise ValueError(
            f"Incomplete per-angle scaling parameters. "
            f"Found {len(contrast_keys)} contrast parameters and {len(offset_keys)} offset parameters. "
            f"Expected matching counts for all phi angles."
        )

    # Stack samples: shape (n_samples, n_angles)
    contrast_samples_per_angle = jnp.stack(
        [samples[k] for k in contrast_keys], axis=1
    )
    offset_samples_per_angle = jnp.stack([samples[k] for k in offset_keys], axis=1)

    # Flatten across angles for global statistics
    # This gives us all samples from all angles: shape (n_samples * n_angles,)
    contrast_samples = contrast_samples_per_angle.flatten()
    offset_samples = offset_samples_per_angle.flatten()

    logger.debug(
        f"Per-angle scaling validated: {len(contrast_keys)} angles, "
        f"{contrast_samples_per_angle.shape[0]} samples per angle, "
        f"total {len(contrast_samples)} contrast samples"
    )

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

        # Check dimensionality using a representative parameter (handle per-angle scaling)
        # Use the first available contrast parameter or D0 as reference
        ref_param_key = "contrast_0" if per_angle_scaling_detected else "contrast"
        if ref_param_key not in samples_with_chains:
            ref_param_key = "D0"  # Fallback to physics parameter

        if samples_with_chains[ref_param_key].ndim > 1:
            # Multiple chains available - compute convergence diagnostics
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            for param_name in samples.keys():
                try:
                    # R-hat (Gelman-Rubin statistic)
                    r_hat_dict[param_name] = float(
                        gelman_rubin(samples_with_chains[param_name])
                    )

                    # ESS (Effective Sample Size)
                    ess_dict[param_name] = float(
                        effective_sample_size(samples_with_chains[param_name])
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute diagnostics for {param_name}: {e}"
                    )
                    r_hat_dict[param_name] = None
                    ess_dict[param_name] = None
        else:
            # Single chain - cannot compute R-hat
            logger.info("Single chain detected - R-hat not available")
            for param_name in samples.keys():
                r_hat_dict[param_name] = None
                # Compute ESS for single chain using autocorrelation
                try:
                    ess_dict[param_name] = float(
                        effective_sample_size(samples_with_chains[param_name])
                    )
                except:
                    ess_dict[param_name] = None

        # Check convergence based on diagnostics
        converged = True
        if any(r_hat_dict.values()):
            # Check if any R-hat > 1.1 (indicates non-convergence)
            max_r_hat = max(
                [v for v in r_hat_dict.values() if v is not None], default=0.0
            )
            if max_r_hat > 1.1:
                logger.warning(
                    f"Poor convergence detected: max R-hat = {max_r_hat:.3f} > 1.1"
                )
                converged = False

        if any(ess_dict.values()):
            # Check if any ESS < 100 (indicates poor mixing)
            min_ess = min(
                [v for v in ess_dict.values() if v is not None], default=float("inf")
            )
            if min_ess < 100:
                logger.warning(
                    f"Poor sampling efficiency: min ESS = {min_ess:.0f} < 100"
                )
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
