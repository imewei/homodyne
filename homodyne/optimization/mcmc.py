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

- **JAX Acceleration**: CPU-only execution with JIT compilation (v2.3.0+)
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

import os
import time
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

_FIXED_CONTRAST_RANGE = (0.01, 0.2)
_FIXED_OFFSET_RANGE = (0.9, 1.1)


def _get_physical_param_order(analysis_mode: str) -> list[str]:
    """Return canonical ordering of physical parameters for diagnostics."""

    mode = (analysis_mode or "").lower()
    if "laminar" in mode:
        return [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
    return ["D0", "alpha", "D_offset"]


def _estimate_single_angle_scaling(data: Any) -> tuple[float, float]:
    """Estimate deterministic contrast/offset for phi_count==1 fallback."""

    try:
        data_arr = np.asarray(data).astype(float, copy=False).ravel()
    except Exception:  # noqa: BLE001 - fallback for unexpected dtypes
        data_arr = np.asarray(data, dtype=float).ravel()

    finite = data_arr[np.isfinite(data_arr)]
    if finite.size == 0:
        return 0.5, 1.0

    low = float(np.percentile(finite, 1.0))
    high = float(np.percentile(finite, 99.0))
    span = max(high - low, 1e-4)

    contrast = 0.8 * span
    contrast = float(np.clip(contrast, *_FIXED_CONTRAST_RANGE))
    offset = float(np.clip(low, *_FIXED_OFFSET_RANGE))
    return contrast, offset


def _build_single_angle_surrogate_settings(
    parameter_space: ParameterSpace | None, tier: str
) -> dict[str, Any]:
    """Construct surrogate configuration for phi_count==1."""

    if parameter_space is None or PriorDistribution is None:
        return {}

    tier_normalized = (tier or "2").strip()
    # Tier 1: NLSQ-friendly mode (keeps all params, no log-space sampling)
    # Tier 2-4: Difficult cases (drop D_offset, use log-space D0)
    if tier_normalized not in {"1", "2", "3", "4"}:
        tier_normalized = "2"

    try:
        d0_bounds = parameter_space.get_bounds("D0")
        d0_prior = parameter_space.get_prior("D0")
    except KeyError:
        return {}

    # Tier 1 uses simplified setup - no log-space configuration needed
    if tier_normalized == "1":
        nuts_overrides = {
            "target_accept_prob": 0.95,  # Less aggressive than tier 2
            "max_tree_depth": 10,
            "n_warmup": 1000,
        }
        diagnostic_thresholds = {
            "focus_params": ["D0", "alpha", "D_offset"],
            "min_ess": 20.0,
            "max_rhat": 1.2,
        }
        return {
            "tier": "1",
            "drop_d_offset": False,  # Keep all 3 parameters
            "sample_log_d0": False,  # Use standard linear-space sampling
            "fixed_d_offset": None,  # Sample D_offset, don't fix it
            "alpha_prior_override": None,  # Use default alpha prior
            "fixed_alpha": None,
            "fixed_d0_value": None,
            "disable_reparam": True,
            "nuts_overrides": nuts_overrides,
            "diagnostic_thresholds": diagnostic_thresholds,
        }

    min_d0 = max(d0_bounds[0], 1e-6)
    max_d0 = max(d0_bounds[1], min_d0 * 10.0)
    if tier_normalized == "4":
        constrained_max = max(min_d0 * 50.0, min_d0 * 2.0)
        max_d0 = min(max_d0, constrained_max)

    log_low = float(np.log(min_d0))
    log_high = float(np.log(max_d0))

    prior_mu_clamped = float(np.clip(d0_prior.mu, min_d0, max_d0))
    if tier_normalized == "4":
        prior_mu_clamped = min(prior_mu_clamped, max_d0 * 0.6)
    log_loc = float(np.log(max(prior_mu_clamped, 1e-6)))
    log_scale = 0.5

    alpha_prior_override: PriorDistribution | None = None
    fixed_alpha = None

    log_scale = 0.5
    trust_radius = None
    sample_log_d0 = True

    if tier_normalized == "2":
        alpha_prior_override = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=-1.0,
            sigma=0.15,
            min_val=-1.5,
            max_val=0.5,
        )
        log_scale = 0.35
    elif tier_normalized == "3":
        fixed_alpha = -1.2
        log_scale = 0.3
    elif tier_normalized == "4":
        # Tier-4: Keep log-space sampling but with tighter bounds
        # Avoid deterministic clamp (fixed_d0_value) to enable proper MCMC
        fixed_alpha = -1.2
        log_scale = 0.12
        trust_radius = None  # Remove trust_radius - not needed with ExpTransform
        sample_log_d0 = True  # Changed from False - use log-space sampling

    nuts_overrides = {
        "target_accept_prob": 0.99 if tier_normalized == "2" else 0.995,
        "max_tree_depth": 8 if tier_normalized == "2" else 6,
        "n_warmup": 1500 if tier_normalized == "2" else 2000,
    }
    if tier_normalized == "4":
        nuts_overrides.update({"max_tree_depth": 6, "n_warmup": 2000})

    diagnostic_thresholds = {
        "focus_params": ["D0", "alpha"],
        "min_ess": 25.0 if tier_normalized == "2" else 40.0,
        "max_rhat": 1.2,
    }
    if tier_normalized == "4":
        diagnostic_thresholds["min_ess"] = 50.0

    return {
        "tier": tier_normalized,
        "drop_d_offset": True,
        "sample_log_d0": sample_log_d0,
        "log_d0_prior": {
            "loc": log_loc,
            "scale": log_scale,
            "low": log_low,
            "high": log_high,
            "trust_radius": trust_radius,
        },
        "alpha_prior_override": alpha_prior_override,
        "fixed_alpha": fixed_alpha,
        "fixed_d_offset": 0.0,
        "fixed_d0_value": None,  # Removed tier-4 deterministic clamp
        "disable_reparam": True,
        "nuts_overrides": nuts_overrides,
        "diagnostic_thresholds": diagnostic_thresholds,
    }


def _sample_single_angle_log_d0(
    prior_cfg: dict[str, float], target_dtype
) -> jnp.ndarray:
    """Sample D0 via truncated Normal in log-space with ExpTransform.

    For single-angle static/static_isotropic models (n_phi==1), this function
    samples D0 using a truncated Normal distribution in log-space, then
    automatically transforms to linear space via ExpTransform. This provides
    better MCMC sampling geometry compared to linear-space sampling, as
    diffusion coefficients naturally span multiple orders of magnitude.

    The sampled parameter is named 'D0' (not 'log_D0') to maintain API
    consistency with multi-angle paths.

    Parameters
    ----------
    prior_cfg : dict
        Prior configuration with keys:
        - 'loc': Mean of log-D0 (e.g., np.log(1000) ≈ 6.9)
        - 'scale': Standard deviation in log-space (e.g., 0.5)
        - 'low': Lower bound in log-space (e.g., np.log(100) ≈ 4.6)
        - 'high': Upper bound in log-space (e.g., np.log(10000) ≈ 9.2)
        - 'trust_radius': Optional additional clipping radius (for tier-4)
    target_dtype : jnp.dtype
        Target JAX dtype (float32 or float64)

    Returns
    -------
    jnp.ndarray
        D0 value in linear space (automatically exp-transformed)

    Notes
    -----
    **Log-space sampling advantages:**
    - More efficient exploration of scale parameters spanning orders of magnitude
    - Better MCMC geometry (symmetric proposals in log-space)
    - Improved ESS and R-hat convergence diagnostics
    - Natural handling of positivity constraint

    **Implementation:**
    Uses NumPyro's TransformedDistribution with ExpTransform to sample
    in log-space and automatically convert to linear space. The latent
    variable is sampled from TruncatedNormal(loc, scale, low, high) in
    log-space, then exp-transformed to produce D0.

    **Multi-angle compatibility:**
    This function is ONLY used when n_phi==1. Multi-angle paths use
    the standard linear TruncatedNormal sampling.
    """

    loc_value = float(prior_cfg.get("loc", 0.0))
    scale_value = max(float(prior_cfg.get("scale", 1.0)), 1e-6)
    low_value = float(prior_cfg.get("low", loc_value - 5.0))
    high_value = float(prior_cfg.get("high", loc_value + 5.0))

    # Ensure proper bounds with small epsilon for numerical stability
    _interval_width = max(high_value - low_value, 1e-6)  # noqa: F841 - Validated bounds
    eps = 1e-6

    loc = jnp.asarray(loc_value, dtype=target_dtype)
    scale = jnp.asarray(scale_value, dtype=target_dtype)
    low = jnp.asarray(low_value + eps, dtype=target_dtype)
    high = jnp.asarray(high_value - eps, dtype=target_dtype)

    # Create truncated Normal distribution in log-space
    log_space_dist = dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    # Apply ExpTransform to get linear-space D0
    # This creates: D0 = exp(log_D0) where log_D0 ~ TruncatedNormal(...)
    from numpyro.distributions.transforms import ExpTransform

    d0_dist = dist.TransformedDistribution(log_space_dist, ExpTransform())

    # Sample D0 directly (already in linear space due to transform)
    # Keep parameter name as 'D0' for API consistency
    d0_value = sample("D0", d0_dist)

    # Emit the log-space latent as a deterministic node for diagnostics
    # This allows post-hoc analysis of the log-space sampling
    log_d0_value = jnp.log(d0_value)
    deterministic("log_D0_latent", log_d0_value)

    return d0_value


# JAX imports with intelligent fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random

    JAX_AVAILABLE = True
    HAS_NUMPY_GRADIENTS = False  # Use JAX when available
except ImportError:
    JAX_AVAILABLE = False
    jax = None
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
    from numpyro import deterministic, prng_key, sample  # noqa: F401 - prng_key
    from numpyro.distributions import transforms as dist_transforms
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
    from homodyne.optimization.cmc.bypass import evaluate_cmc_bypass
    from homodyne.optimization.cmc.result import MCMCResult

    HAS_CMC_RESULT = True
except ImportError:
    # Fallback to original MCMCResult if CMC module not available
    HAS_CMC_RESULT = False
    evaluate_cmc_bypass = None

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
            analysis_mode: str = "static",
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

    class _DummyBypassDecision:
        should_bypass = False
        reason = None
        triggered_by = None
        mode = "force_cmc"

    def evaluate_cmc_bypass(*_, **__):  # type: ignore[override]  # noqa: F811
        return _DummyBypassDecision()


# Import per-phi initialization utilities
try:
    from homodyne.optimization.initialization.per_phi_initializer import (
        PerPhiInitConfig,
        build_per_phi_initial_values,
    )

    HAS_PER_PHI_INIT = True
except ImportError:
    HAS_PER_PHI_INIT = False
    PerPhiInitConfig = None
    build_per_phi_initial_values = None


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


def _prepare_phi_mapping(
    phi_array: np.ndarray | None,
    *,
    data_size: int,
    n_phi: int,
    phi_unique_np: np.ndarray,
    target_dtype,
) -> jnp.ndarray | None:
    """Ensure phi mapping array matches flattened data size.

    The NumPyro model needs a 1D array whose length matches the pooled
    data array so that each data point can be mapped back to its source
    phi angle when applying per-angle contrast/offset scaling. Some
    callers (notably older CLI code paths) accidentally pass only the
    list of unique phi angles rather than the full replicated mapping.
    This helper auto-expands that list to the expected length when the
    relationship between ``data`` and ``phi`` is unambiguous.

    Parameters
    ----------
    phi_array : np.ndarray | None
        Raw phi array provided by the caller (may already be replicated).
    data_size : int
        Flattened data length used by the likelihood.
    n_phi : int
        Number of unique phi angles present in the dataset.
    phi_unique_np : np.ndarray
        Array of unique phi values computed outside of JIT tracing.
    target_dtype : jnp.dtype
        Target dtype for JAX computations (float64 by default).

    Returns
    -------
    jnp.ndarray | None
        Phi mapping array with ``data_size`` elements when inference is
        possible, otherwise the original array (mismatch will be
        validated downstream).
    """

    if phi_array is None:
        return None

    phi_mapping = jnp.ravel(jnp.asarray(phi_array, dtype=target_dtype))
    phi_length = int(phi_mapping.size)

    # Nothing to do if either length is zero or already matches data size
    if data_size <= 0 or phi_length == data_size:
        return phi_mapping

    # Auto-expand when we only received the list of unique angles but can
    # infer how many data points belong to each angle deterministically.
    if (
        n_phi > 0
        and phi_length == n_phi
        and data_size % n_phi == 0
        and phi_unique_np.size == n_phi
    ):
        points_per_angle = data_size // n_phi
        expanded = np.repeat(phi_unique_np, points_per_angle)
        logger.warning(
            "phi array length (%d) does not match data size (%d). "
            "Auto-expanding by repeating each of the %d unique angles for "
            "%d points. Update caller to pass the flattened phi array.",
            phi_length,
            data_size,
            n_phi,
            points_per_angle,
        )
        return jnp.asarray(expanded, dtype=target_dtype)

    return phi_mapping


@log_performance(threshold=10.0)
def fit_mcmc_jax(
    data: np.ndarray,
    sigma: np.ndarray | None = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "static",
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
    analysis_mode : str, default "static"
        Physics model selection:
        - "static": Diffusion-only (3 physical params)
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

    # Resolve stable prior fallback knob (kwargs > config > default False)
    env_stable_prior = os.environ.get("HOMODYNE_STABLE_PRIOR", "0") == "1"
    stable_prior_flag = kwargs.get("stable_prior_fallback")
    if stable_prior_flag is None:
        if config_dict is not None:
            stable_prior_flag = (
                config_dict.get("optimization", {})
                .get("mcmc", {})
                .get("stable_prior_fallback", False)
            )
        else:
            stable_prior_flag = False

    kwargs["stable_prior_fallback"] = bool(stable_prior_flag or env_stable_prior)

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
                "Initial parameter values violate bounds:\n" + "\n".join(violations)
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

        bypass_decision = evaluate_cmc_bypass(
            cmc_config,
            num_samples=num_samples,
            dataset_size=dataset_size,
            phi=phi,
        )

        if bypass_decision.should_bypass:
            reason = bypass_decision.reason or "heuristic bypass triggered"
            logger.warning(
                "Bypassing CMC (%s): %s",
                bypass_decision.triggered_by or bypass_decision.mode,
                reason,
            )
            actual_method = "nuts"
        else:
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

    if actual_method != "cmc":
        # Use standard NUTS with automatic retry on convergence failure
        logger.info("=" * 70)
        logger.info("Executing standard NUTS MCMC")
        logger.info("=" * 70)

        # Implement auto-retry mechanism for poor convergence
        max_retries = int(kwargs.get("n_retries") or kwargs.get("max_retries") or 3)
        default_max_rhat = float(kwargs.get("max_rhat_threshold", 1.1))
        default_min_ess = float(kwargs.get("min_ess_threshold", 100))
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
                    prev_eval = _evaluate_convergence_thresholds(
                        result,
                        default_max_rhat,
                        default_min_ess,
                    )
                    prev_rhat = prev_eval.get("max_rhat_observed")
                    prev_ess = prev_eval.get("min_ess_observed")
                    rhat_text = (
                        f"{prev_rhat:.3f}"
                        if (prev_rhat is not None and np.isfinite(prev_rhat))
                        else "N/A"
                    )
                    ess_text = (
                        f"{prev_ess:.0f}"
                        if (prev_ess is not None and np.isfinite(prev_ess))
                        else "N/A"
                    )
                    logger.warning(
                        f"🔄 Retry {attempt}/{max_retries - 1} - Convergence poor "
                        f"(R-hat={rhat_text}, ESS={ess_text}). "
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
                eval_report = _evaluate_convergence_thresholds(
                    result,
                    default_max_rhat,
                    default_min_ess,
                )
                poor_rhat = eval_report["poor_rhat"]
                poor_ess = eval_report["poor_ess"]
                max_rhat = eval_report.get("max_rhat_observed")
                min_ess = eval_report.get("min_ess_observed")
                thresholds = eval_report.get("thresholds", {})
                rhat_text = (
                    f"{max_rhat:.3f}"
                    if (max_rhat is not None and np.isfinite(max_rhat))
                    else "N/A"
                )
                ess_text = (
                    f"{min_ess:.0f}"
                    if (min_ess is not None and np.isfinite(min_ess))
                    else "N/A"
                )
                if poor_rhat:
                    logger.warning(
                        f"Poor convergence detected: max R-hat = {rhat_text} > "
                        f"{thresholds.get('max_rhat', default_max_rhat):.3f}"
                    )
                if poor_ess:
                    logger.warning(
                        f"Low effective sample size: min ESS = {ess_text} < "
                        f"{thresholds.get('min_ess', default_min_ess):.0f}"
                    )

                # Retry if convergence is poor and we have retries left
                if (poor_rhat or poor_ess) and attempt < max_retries - 1:
                    logger.warning(
                        "Convergence quality poor. Retrying with different random seed..."
                    )
                    continue  # Retry
                else:
                    # Either convergence is good, or we're out of retries
                    if poor_rhat or poor_ess:
                        # All retries failed - provide actionable suggestions
                        logger.warning(
                            f"❌ All {max_retries} retry attempts failed. "
                            f"Final diagnostics: R-hat={rhat_text}, ESS={ess_text}. "
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
    analysis_mode: str = "static",
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
        import numpy as _np

        # Set up parameter space
        if parameter_space is None:
            parameter_space = ParameterSpace()

        # Determine analysis mode (normalize shorthand names)
        # Map common shortcuts to canonical names
        mode_mapping = {
            "static_isotropic": "static",  # Legacy compatibility: map old name to new
            "laminar": "laminar_flow",
        }
        analysis_mode = mode_mapping.get(analysis_mode, analysis_mode)

        if analysis_mode not in ["static", "laminar_flow"]:
            logger.warning(
                f"Unknown analysis mode {analysis_mode}, using static",
            )
            analysis_mode = "static"

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
            if t1.ndim == 2:
                time_array = _np.asarray(t1[:, 0] if t1.shape[1] > 0 else t1[0, :])
            else:
                time_array = _np.asarray(t1)
            # Estimate from first two unique time points
            unique_times = _np.unique(time_array)
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
        if phi is not None:
            phi_unique = _np.unique(_np.asarray(phi))
            logger.debug(
                f"Pre-computed phi_unique: {len(phi_unique)} unique angles from {len(phi)} total values "
                f"(reduction: {len(phi) / len(phi_unique):.1f}x)"
            )

        n_unique_phi = len(phi_unique) if phi_unique is not None else 0
        single_angle_static = (
            analysis_mode.lower().startswith("static") and n_unique_phi == 1
        )
        per_angle_scaling_enabled = not single_angle_static
        single_angle_geom_priors = None
        t_reference_value = None
        single_angle_scaling_override: dict[str, float] | None = None
        single_angle_surrogate_cfg: dict[str, Any] | None = None
        surrogate_tier_env = os.environ.get("HOMODYNE_SINGLE_ANGLE_TIER", "2")
        user_scaling_override = kwargs.pop("fixed_scaling_overrides", None)

        # Extract user-provided contrast/offset from initial_values if present
        # This allows configured values to override data-derived percentiles
        if user_scaling_override is None and initial_values is not None:
            extracted_scaling = {}
            if "contrast" in initial_values:
                extracted_scaling["contrast"] = float(initial_values["contrast"])
            if "offset" in initial_values:
                extracted_scaling["offset"] = float(initial_values["offset"])
            if extracted_scaling:
                user_scaling_override = extracted_scaling
                logger.info(
                    "Using contrast/offset from initial_values: %s",
                    ", ".join(f"{k}={v:.4g}" for k, v in extracted_scaling.items()),
                )

        if single_angle_static:
            logger.info(
                "Single-angle static dataset detected → disabling per-angle scaling and enabling stabilized priors",
            )
            if not kwargs.get("stable_prior_fallback", False):
                kwargs["stable_prior_fallback"] = True
            try:
                t1_array = _np.asarray(t1)
                positive = t1_array[t1_array > 0]
                if positive.size:
                    t_reference_value = float(_np.median(positive))
                elif t1_array.size:
                    t_reference_value = float(_np.mean(_np.abs(t1_array)))
            except Exception:  # noqa: BLE001 - heuristic fallback
                t_reference_value = None

            contrast_fixed, offset_fixed = _estimate_single_angle_scaling(data)
            single_angle_scaling_override = {
                "contrast": contrast_fixed,
                "offset": offset_fixed,
            }
            logger.info(
                "Single-angle fallback: fixing contrast=%.4f, offset=%.4f based on data percentiles",
                contrast_fixed,
                offset_fixed,
            )

        if user_scaling_override:
            single_angle_scaling_override = {
                key: float(value) for key, value in user_scaling_override.items()
            }
            logger.info(
                "Single-angle fallback: using user-provided scaling overrides %s",
                ", ".join(
                    f"{name}={val:.4g}"
                    for name, val in single_angle_scaling_override.items()
                ),
            )

        stable_prior_config = bool(kwargs.get("stable_prior_fallback", False))
        stable_prior_env = os.environ.get("HOMODYNE_STABLE_PRIOR", "0") == "1"
        stable_prior_enabled = stable_prior_config or stable_prior_env

        if single_angle_static and parameter_space is not None:
            parameter_space = parameter_space.with_single_angle_stabilization(
                enable_beta_fallback=stable_prior_enabled,
            )
            logger.debug(
                "Applied single-angle stabilization priors (beta_fallback=%s)",
                stable_prior_enabled,
            )
            single_angle_geom_priors = (
                parameter_space.get_single_angle_geometry_config()
            )
            if t_reference_value is not None:
                single_angle_geom_priors["t_reference"] = t_reference_value
            single_angle_surrogate_cfg = _build_single_angle_surrogate_settings(
                parameter_space, surrogate_tier_env
            )
            if single_angle_surrogate_cfg:
                logger.info(
                    "Single-angle surrogate tier %s active (drop_d_offset=%s)",
                    single_angle_surrogate_cfg.get("tier"),
                    single_angle_surrogate_cfg.get("drop_d_offset"),
                )
                if single_angle_surrogate_cfg.get("drop_d_offset"):
                    # Preserve D_offset value from initial_values before dropping it
                    # This allows MCMC to use the NLSQ-fitted value instead of hardcoded 0.0
                    if initial_values is not None and "D_offset" in initial_values:
                        d_offset_from_init = initial_values.pop("D_offset")
                        # Update surrogate config to use the initial value
                        single_angle_surrogate_cfg["fixed_d_offset"] = float(
                            d_offset_from_init
                        )
                        logger.info(
                            "Using D_offset=%.4f from initial_values for single-angle surrogate (not sampled)",
                            d_offset_from_init,
                        )
                    else:
                        # Fallback to default if no initial value provided
                        if initial_values is not None:
                            initial_values.pop("D_offset", None)
                    parameter_space = parameter_space.drop_parameters({"D_offset"})
                alpha_override = single_angle_surrogate_cfg.get("alpha_prior_override")
                if alpha_override is not None:
                    parameter_space = parameter_space.with_prior_overrides(
                        {"alpha": alpha_override}
                    )
                if (
                    single_angle_surrogate_cfg.get("fixed_alpha") is not None
                    and initial_values is not None
                ):
                    initial_values.pop("alpha", None)

        per_angle_param_count = (
            len(phi_unique) * 2
            if phi_unique is not None and per_angle_scaling_enabled
            else 0
        )
        base_param_count = (
            len(parameter_space.parameter_names) if parameter_space else 0
        )
        override_param_count = len(single_angle_scaling_override or {})
        total_param_count = (
            max(0, base_param_count - override_param_count) + per_angle_param_count
        )

        # Configure MCMC parameters
        config_kwargs = dict(kwargs)
        config_kwargs.setdefault("n_params", total_param_count)
        user_config_overrides = {
            "n_warmup": config_kwargs.get("n_warmup") is not None,
        }
        mcmc_config = _get_mcmc_config(config_kwargs)
        if single_angle_surrogate_cfg:
            nuts_overrides = single_angle_surrogate_cfg.get("nuts_overrides") or {}
            if "target_accept_prob" in nuts_overrides:
                mcmc_config["target_accept_prob"] = max(
                    mcmc_config["target_accept_prob"],
                    nuts_overrides["target_accept_prob"],
                )
            if "max_tree_depth" in nuts_overrides:
                mcmc_config["max_tree_depth"] = min(
                    mcmc_config["max_tree_depth"], nuts_overrides["max_tree_depth"]
                )
            if "n_warmup" in nuts_overrides and not user_config_overrides["n_warmup"]:
                mcmc_config["n_warmup"] = max(
                    mcmc_config["n_warmup"], nuts_overrides["n_warmup"]
                )
            logger.info(
                "Single-angle NUTS overrides applied: target_accept=%.3f, max_tree_depth=%d, n_warmup=%d",
                mcmc_config["target_accept_prob"],
                mcmc_config["max_tree_depth"],
                mcmc_config["n_warmup"],
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

        # Only add reparameterization parameters if not disabled by surrogate config
        # Tier 1 and others with disable_reparam=True skip this initialization
        reparam_disabled = (
            single_angle_surrogate_cfg
            and single_angle_surrogate_cfg.get("disable_reparam", False)
        )
        if single_angle_static and initial_values is not None and not reparam_disabled:
            d0_init = initial_values.get("D0")
            d_offset_init = initial_values.get("D_offset")
            if d0_init is not None and d_offset_init is not None:
                center_value = d0_init + d_offset_init
                if not np.isfinite(center_value) or center_value <= 0:
                    center_value = max(1.0, abs(d0_init))
                log_center_init = float(np.log(max(center_value, 1e-6)))
                frac = d0_init / max(center_value, 1e-6)
                frac = float(np.clip(frac, 1e-6, 5.0))
                delta_raw_init = float(np.log(np.expm1(frac)))
                initial_values.setdefault("log_D_center", log_center_init)
                initial_values.setdefault("delta_raw", delta_raw_init)

        logger.debug(
            "Stable prior fallback: enabled=%s (config=%s, env=%s)",
            stable_prior_enabled,
            stable_prior_config,
            stable_prior_env,
        )

        def _build_model(space: ParameterSpace):
            return _create_numpyro_model(
                data,
                sigma,
                t1,
                t2,
                (phi_unique if phi_unique is not None else phi),
                q,
                L,
                analysis_mode,
                space,
                use_simplified=use_simplified_likelihood,
                dt=dt_computed,
                phi_full=phi,
                per_angle_scaling=per_angle_scaling_enabled,
                single_angle_reparam_config=single_angle_geom_priors,
                fixed_scaling_overrides=single_angle_scaling_override,
                single_angle_surrogate_config=single_angle_surrogate_cfg,
            )

        def _execute_sampling(space: ParameterSpace):
            model_local = _build_model(space)
            if NUMPYRO_AVAILABLE:
                return _run_numpyro_sampling(
                    model_local,
                    mcmc_config,
                    initial_values,
                    parameter_space=space,
                    phi_unique=phi_unique,
                    per_angle_scaling=per_angle_scaling_enabled,
                )
            return _run_blackjax_sampling(model_local, mcmc_config)

        retry_trigger = "Cannot find valid initial parameters"
        try:
            result = _execute_sampling(parameter_space)
        except RuntimeError as exc:
            message = str(exc)
            initialize_failure = retry_trigger in message
            if stable_prior_enabled and initialize_failure:
                logger.warning(
                    "initialize_model failed (%s). Retrying with BetaScaled priors...",
                    message,
                )
                beta_space = parameter_space.convert_to_beta_scaled_priors()
                result = _execute_sampling(beta_space)
                logger.info("BetaScaled retry succeeded")
            else:
                raise

        if result is None:
            raise RuntimeError("MCMC sampling did not return a result")

        # Process results
        surrogate_thresholds = None
        if single_angle_surrogate_cfg:
            diag_cfg = single_angle_surrogate_cfg.get("diagnostic_thresholds") or {}
            surrogate_thresholds = {
                "active": True,
                "focus_params": diag_cfg.get("focus_params") or ["D0", "alpha"],
                "min_ess": float(diag_cfg.get("min_ess", 25.0)),
                "max_rhat": float(diag_cfg.get("max_rhat", 1.2)),
            }

        deterministic_param_overrides: set[str] = set()
        if single_angle_scaling_override:
            deterministic_param_overrides.update(single_angle_scaling_override.keys())
        if single_angle_surrogate_cfg:
            if (
                single_angle_surrogate_cfg.get("drop_d_offset")
                or single_angle_surrogate_cfg.get("fixed_d_offset") is not None
            ):
                deterministic_param_overrides.add("D_offset")
            if single_angle_surrogate_cfg.get("fixed_alpha") is not None:
                deterministic_param_overrides.add("alpha")
            if single_angle_surrogate_cfg.get("fixed_d0_value") is not None:
                deterministic_param_overrides.add("D0")

        diag_settings = {
            "max_rhat": mcmc_config.get("max_rhat_threshold", 1.1),
            "min_ess": mcmc_config.get("min_ess_threshold", 100),
            "check_hmc_diagnostics": mcmc_config.get("check_hmc_diagnostics", True),
            "single_angle_surrogate": single_angle_surrogate_cfg,
            "scaling_overrides": single_angle_scaling_override or {},
            "expected_params": _get_physical_param_order(analysis_mode),
            "deterministic_params": sorted(deterministic_param_overrides),
            "surrogate_thresholds": surrogate_thresholds,
        }

        posterior_summary = _process_posterior_samples(
            result, analysis_mode, diag_settings
        )

        computation_time = time.perf_counter() - start_time

        logger.info(f"MCMC sampling completed in {computation_time:.3f}s")
        # Get actual sample count from one of the parameter arrays (not dict key count)
        sample_count = len(next(iter(posterior_summary["samples"].values())))
        param_count = len(posterior_summary["samples"])
        logger.info(
            f"Posterior summary: {sample_count} samples × {param_count} parameters"
        )

        result_obj = MCMCResult(
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

        diag_summary = posterior_summary.get("diagnostic_summary", {})
        result_obj.diagnostic_summary = diag_summary
        result_obj.deterministic_params = diag_summary.get("deterministic_params", [])
        result_obj.surrogate_thresholds = surrogate_thresholds

        return result_obj

    except Exception as e:
        computation_time = time.perf_counter() - start_time
        import traceback

        logger.error(f"MCMC sampling failed after {computation_time:.3f}s: {str(e)}")
        logger.debug(f"Full traceback:\n{traceback.format_exc()}")

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
    adaptive_param_count = kwargs.get("n_params")
    default_warmup = 1000
    if isinstance(adaptive_param_count, int) and adaptive_param_count > 0:
        default_warmup = max(default_warmup, adaptive_param_count * 200)

    default_config = {
        "n_samples": 2000,
        "n_warmup": default_warmup,
        "n_chains": 4,
        "target_accept_prob": 0.9,
        "max_tree_depth": 12,
        "dense_mass_matrix": "auto",
        "rng_key": 42,
        "init_jitter_scale": 0.02,
        "max_retries": 3,
    }

    # Update with provided kwargs (support both snake_case variants)
    config = default_config.copy()
    config.update(kwargs)

    dense_setting = config.get("dense_mass_matrix", "auto")
    if isinstance(dense_setting, str) and dense_setting.lower() == "auto":
        if isinstance(adaptive_param_count, int) and adaptive_param_count > 0:
            config["dense_mass_matrix"] = adaptive_param_count <= 10
        else:
            config["dense_mass_matrix"] = False

    min_ess_threshold = config.pop("min_ess", None)
    max_rhat_threshold = config.pop("max_rhat", None)
    if min_ess_threshold is not None:
        config["min_ess_threshold"] = min_ess_threshold
    if max_rhat_threshold is not None:
        config["max_rhat_threshold"] = max_rhat_threshold
    if "min_ess_threshold" not in config:
        config["min_ess_threshold"] = 100
    if "max_rhat_threshold" not in config:
        config["max_rhat_threshold"] = 1.1

    if "n_retries" in config and "max_retries" not in config:
        config["max_retries"] = config["n_retries"]

    if "check_hmc_diagnostics" not in config:
        config["check_hmc_diagnostics"] = kwargs.get("check_hmc_diagnostics", True)

    # Mirror short-form keys to long-form equivalents so that downstream
    # components (CMC coordinator) that expect num_* receive the overrides.
    config["num_samples"] = config.get("num_samples", config["n_samples"])
    config["num_warmup"] = config.get("num_warmup", config["n_warmup"])
    config["num_chains"] = config.get("num_chains", config["n_chains"])

    return config


def _evaluate_convergence_thresholds(
    result: Any, default_max_rhat: float = 1.1, default_min_ess: float = 100.0
) -> dict[str, Any]:
    """Assess convergence metrics with surrogate-aware thresholds."""

    diag_summary = getattr(result, "diagnostic_summary", {}) or {}
    deterministic_params = set(diag_summary.get("deterministic_params") or [])
    per_param_stats = diag_summary.get("per_param_stats") or {}
    surrogate_ctx = diag_summary.get("surrogate_thresholds") or {}

    def _collect_stat(values: list[float], reducer, default_value):
        finite_vals = [v for v in values if v is not None and np.isfinite(v)]
        return reducer(finite_vals) if finite_vals else default_value

    per_param_items = list(per_param_stats.items())
    max_rhat = _collect_stat(
        [
            stats.get("r_hat")
            for name, stats in per_param_items
            if not stats.get("deterministic")
        ],
        max,
        None,
    )
    min_ess = _collect_stat(
        [
            stats.get("ess")
            for name, stats in per_param_items
            if not stats.get("deterministic")
        ],
        min,
        None,
    )

    thresholds = {
        "mode": "default",
        "max_rhat": default_max_rhat,
        "min_ess": default_min_ess,
    }
    poor_rhat = False
    poor_ess = False
    focus_details: dict[str, dict[str, Any]] = {}

    if surrogate_ctx.get("active"):
        thresholds.update(
            {
                "mode": "single_angle_surrogate",
                "max_rhat": float(surrogate_ctx.get("max_rhat", default_max_rhat)),
                "min_ess": float(surrogate_ctx.get("min_ess", default_min_ess)),
            }
        )
        focus_params = (
            surrogate_ctx.get("focus_params") or diag_summary.get("focus_params") or []
        )
        for name in focus_params:
            stats = per_param_stats.get(name, {})
            focus_details[name] = stats
            if stats.get("deterministic"):
                continue
            rhat_val = stats.get("r_hat")
            ess_val = stats.get("ess")
            if rhat_val is not None and rhat_val > thresholds["max_rhat"]:
                poor_rhat = True
            if ess_val is None or ess_val < thresholds["min_ess"]:
                poor_ess = True
    else:
        if max_rhat is not None and max_rhat > thresholds["max_rhat"]:
            poor_rhat = True
        if min_ess is not None and min_ess < thresholds["min_ess"]:
            poor_ess = True

    return {
        "poor_rhat": poor_rhat,
        "poor_ess": poor_ess,
        "max_rhat_observed": max_rhat,
        "min_ess_observed": min_ess,
        "thresholds": thresholds,
        "deterministic_params": deterministic_params,
        "focus_param_details": focus_details,
    }


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
    single_angle_reparam_config: dict[str, float] | None = None,
    fixed_scaling_overrides: dict[str, float] | None = None,
    single_angle_surrogate_config: dict[str, Any] | None = None,
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
        Analysis mode: 'static' or 'laminar_flow'
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

    single_angle_reparam_config : dict, optional
        When provided (phi_count == 1 static mode), enables the centered diffusion
        reparameterization using keys emitted by
        ``ParameterSpace.get_single_angle_geometry_config``.
    single_angle_surrogate_config : dict, optional
        Configuration for reduced-dimension single-angle surrogate models.
        Allows deterministic/fixed parameters and log-space sampling when
        ``phi_count == 1`` without affecting multi-angle execution paths.
    fixed_scaling_overrides : dict, optional
        Deterministic values for scaling parameters (contrast/offset) used when
        single-angle fallback removes those degrees of freedom. When provided,
        the corresponding parameters are emitted as deterministic nodes instead
        of sampled latents, so NUTS no longer explores those dimensions.

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
    - static: [contrast, offset, D0, alpha, D_offset]
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
    dtype_flag = os.environ.get("HOMODYNE_MCMC_DTYPE", "float64").lower()
    if dtype_flag in {"float32", "fp32", "32", "single"}:
        target_dtype = jnp.float32
    elif dtype_flag in {"bfloat16", "bf16"}:
        target_dtype = jnp.bfloat16
    else:
        target_dtype = jnp.float64

    reparam_cfg = {}
    surrogate_cfg = {}
    use_single_angle_reparam = False
    log_center_loc = 8.0
    log_center_scale = 1.0
    delta_loc = 0.0
    delta_scale = 1.0
    delta_floor_value = 1e-3
    t_ref_candidate = None

    def _normalize_array(value, name, allow_none=False):
        if value is None:
            if allow_none:
                return None
            raise ValueError(f"{name} cannot be None for _create_numpyro_model")
        return jnp.asarray(value, dtype=target_dtype)

    def _normalize_scalar(value, allow_none=False):
        if value is None:
            if allow_none:
                return None
            raise ValueError("Scalar value required for _create_numpyro_model")
        return float(jnp.asarray(value, dtype=target_dtype))

    # Capture phi uniqueness using numpy before converting to JAX arrays
    phi_unique_np = np.unique(np.asarray(phi))
    n_phi = len(phi_unique_np)
    analysis_lower = (analysis_mode or "").lower()
    single_angle_static_mode = analysis_lower.startswith("static") and n_phi == 1

    reparam_cfg = dict(single_angle_reparam_config or {})
    surrogate_cfg = single_angle_surrogate_config or {}
    if single_angle_static_mode:
        reparam_cfg["enabled"] = False
    use_single_angle_reparam = bool(
        reparam_cfg.get("enabled")
    ) and not surrogate_cfg.get("disable_reparam", False)
    log_center_loc = float(reparam_cfg.get("log_center_loc", 8.0))
    log_center_scale = max(0.1, float(reparam_cfg.get("log_center_scale", 1.0)))
    delta_loc = float(reparam_cfg.get("delta_loc", 0.0))
    delta_scale = max(0.25, float(reparam_cfg.get("delta_scale", 1.0)))
    delta_floor_value = float(reparam_cfg.get("delta_floor", 1e-3))
    t_ref_candidate = reparam_cfg.get("t_reference")
    if not (
        isinstance(t_ref_candidate, (int, float))
        and np.isfinite(t_ref_candidate)
        and t_ref_candidate > 0
    ):
        fallback_dt = dt if dt is not None else 1.0
        if (
            isinstance(fallback_dt, (int, float))
            and np.isfinite(fallback_dt)
            and fallback_dt > 0
        ):
            t_ref_candidate = float(fallback_dt)
        else:
            t_ref_candidate = 1.0

    data = _normalize_array(data, "data")
    sigma = _normalize_array(sigma, "sigma")
    t1 = _normalize_array(t1, "t1")
    t2 = _normalize_array(t2, "t2")
    phi = _normalize_array(phi, "phi")
    phi_full = _normalize_array(phi_full, "phi_full", allow_none=True)
    phi_array_for_mapping = phi_full if phi_full is not None else phi
    phi_unique_for_sampling = jnp.asarray(phi_unique_np, dtype=target_dtype)
    q = _normalize_scalar(q)
    L = _normalize_scalar(L)
    dt = _normalize_scalar(dt, allow_none=True)

    dtype_debug_enabled = os.environ.get("HOMODYNE_DEBUG_INIT", "0") not in (
        "0",
        "",
        "false",
        "False",
    )
    if dtype_debug_enabled:
        logger.info(
            "MCMC dtype normalization: target=%s data=%s sigma=%s t1=%s t2=%s phi=%s",
            target_dtype,
            data.dtype,
            sigma.dtype,
            t1.dtype,
            t2.dtype,
            phi.dtype,
        )

    # Map distribution type names to NumPyro distribution classes
    DIST_TYPE_MAP = {
        "Normal": dist.Normal,
        "TruncatedNormal": dist.TruncatedNormal,
        "Uniform": dist.Uniform,
        "LogNormal": dist.LogNormal,
        "BetaScaled": dist.Beta,
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

    scaling_overrides = fixed_scaling_overrides or {}

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
        sampled_values: dict[str, jnp.ndarray] = {}
        reparam_active = use_single_angle_reparam and analysis_mode.lower().startswith(
            "static"
        )
        single_angle_latents: dict[str, jnp.ndarray] = {}
        if reparam_active:
            log_center_site = sample(
                "log_D_center",
                dist.Normal(
                    loc=jnp.asarray(log_center_loc, dtype=target_dtype),
                    scale=jnp.asarray(log_center_scale, dtype=target_dtype),
                ),
            )
            delta_raw_site = sample(
                "delta_raw",
                dist.Normal(
                    loc=jnp.asarray(delta_loc, dtype=target_dtype),
                    scale=jnp.asarray(delta_scale, dtype=target_dtype),
                ),
            )
            single_angle_latents = {
                "log_center": jnp.asarray(log_center_site, dtype=target_dtype),
                "delta_raw": jnp.asarray(delta_raw_site, dtype=target_dtype),
                "delta_floor": jnp.asarray(delta_floor_value, dtype=target_dtype),
                "t_reference": jnp.asarray(
                    max(t_ref_candidate, 1e-6), dtype=target_dtype
                ),
            }

        for i, param_name in enumerate(param_names_ordered):
            if surrogate_cfg.get("fixed_d0_value") is not None and param_name == "D0":
                value = jnp.asarray(surrogate_cfg["fixed_d0_value"], dtype=target_dtype)
                sampled_values[param_name] = value
                deterministic("D0", value)
                continue
            if (
                surrogate_cfg.get("sample_log_d0")
                and param_name == "D0"
                and surrogate_cfg.get("log_d0_prior") is not None
            ):
                log_prior = surrogate_cfg["log_d0_prior"]
                # _sample_single_angle_log_d0 now returns D0 in linear space
                # (already exp-transformed via ExpTransform)
                d0_value = _sample_single_angle_log_d0(log_prior, target_dtype)
                sampled_values[param_name] = d0_value
                # No need for additional deterministic node - already emitted in function
                continue
            if surrogate_cfg.get("fixed_alpha") is not None and param_name == "alpha":
                alpha_value = jnp.asarray(
                    surrogate_cfg["fixed_alpha"], dtype=target_dtype
                )
                sampled_values[param_name] = alpha_value
                deterministic("alpha", alpha_value)
                continue
            if (
                surrogate_cfg.get("fixed_d_offset") is not None
                and param_name == "D_offset"
            ):
                d_offset_value = jnp.asarray(
                    surrogate_cfg["fixed_d_offset"], dtype=target_dtype
                )
                sampled_values[param_name] = d_offset_value
                deterministic("D_offset", d_offset_value)
                continue
            if param_name in scaling_overrides and not per_angle_scaling:
                value = jnp.asarray(scaling_overrides[param_name], dtype=target_dtype)
                sampled_values[param_name] = value
                deterministic(param_name, value)
                continue
            if reparam_active and param_name in {"D0", "D_offset"}:
                sampled_values[param_name] = None
                continue
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

            if (
                surrogate_cfg.get("alpha_prior_override") is not None
                and param_name == "alpha"
            ):
                prior_spec = surrogate_cfg["alpha_prior_override"]

            # Get NumPyro distribution class
            dist_class = DIST_TYPE_MAP.get(
                prior_spec.dist_type,
                dist.TruncatedNormal,  # Safe fallback
            )

            # CRITICAL FIX (Nov 2025): Auto-convert unbounded distributions to bounded
            # If config specifies Normal/LogNormal with bounds, convert to Truncated version
            # This prevents NumPyro from sampling extreme values that cause physics NaN/inf
            if hasattr(prior_spec, "min_val") and hasattr(prior_spec, "max_val"):
                if prior_spec.min_val is not None and prior_spec.max_val is not None:
                    # Bounds are specified - use truncated distribution regardless of type
                    if prior_spec.dist_type == "Normal":
                        dist_class = dist.TruncatedNormal
                        logger.debug(
                            f"Auto-converted {param_name} from Normal to TruncatedNormal "
                            f"with bounds [{prior_spec.min_val}, {prior_spec.max_val}]"
                        )
                    elif prior_spec.dist_type == "LogNormal":
                        # LogNormal is inherently positive, truncate at bounds
                        dist_class = (
                            dist.TruncatedNormal
                        )  # Use TruncatedNormal as fallback
                        logger.debug(
                            f"Converted {param_name} from LogNormal to TruncatedNormal "
                            f"with bounds [{prior_spec.min_val}, {prior_spec.max_val}]"
                        )

            # Get distribution kwargs
            dist_kwargs = prior_spec.to_numpyro_kwargs()

            def _cast_dist_value(value):
                if isinstance(value, (int, float, np.number)):
                    return jnp.asarray(value, dtype=target_dtype)
                if isinstance(value, np.ndarray):
                    return jnp.asarray(value, dtype=target_dtype)
                return value

            dist_kwargs = {
                key: _cast_dist_value(val) for key, val in dist_kwargs.items()
            }
            # Ensure bounds stay as JAX arrays to avoid tracer conversion errors
            if "low" in dist_kwargs and dist_kwargs["low"] is not None:
                dist_kwargs["low"] = jnp.asarray(dist_kwargs["low"], dtype=target_dtype)
            if "high" in dist_kwargs and dist_kwargs["high"] is not None:
                dist_kwargs["high"] = jnp.asarray(
                    dist_kwargs["high"], dtype=target_dtype
                )

            # Tier 1 uses standard linear-space sampling, no log-space conversion needed
            # Only use alternative log-space sampling for cases without surrogate log sampling
            tier_allows_log_fallback = surrogate_cfg.get("tier") != "1"
            single_angle_log_sampling = (
                single_angle_static_mode
                and param_name == "D0"
                and parameter_space is not None
                and not surrogate_cfg.get("sample_log_d0")
                and surrogate_cfg.get("fixed_d0_value") is None
                and not reparam_active
                and tier_allows_log_fallback  # Don't use for tier 1
            )
            if single_angle_log_sampling:
                d0_bounds = parameter_space.get_bounds("D0")

                def _extract_scalar(name: str, fallback: float):
                    """Extract value from dist_kwargs, keeping as JAX array to avoid tracer errors."""
                    value = dist_kwargs.get(name)
                    if value is None:
                        return jnp.asarray(fallback, dtype=target_dtype)
                    # Don't call float() - keep as JAX array to avoid tracer concretization
                    return jnp.asarray(value, dtype=target_dtype).reshape(())

                linear_loc = _extract_scalar("loc", d0_bounds[0])
                linear_scale = jnp.abs(
                    _extract_scalar("scale", (d0_bounds[1] - d0_bounds[0]) / 4.0)
                )
                linear_low = jnp.maximum(_extract_scalar("low", d0_bounds[0]), 1e-6)
                linear_high = jnp.maximum(
                    _extract_scalar("high", d0_bounds[1]), linear_low + 1e-6
                )

                # Convert to Python scalars for log_prior_cfg (static configuration, not traced)
                # Use .item() to extract concrete values outside of traced region
                import numpy as _np

                log_prior_cfg = {
                    "loc": float(_np.log(max(float(linear_loc), 1e-6))),
                    "scale": float(
                        _np.clip(
                            float(linear_scale) / max(float(linear_loc), 1e-6),
                            1e-6,
                            5.0,
                        )
                    ),
                    "low": float(_np.log(max(float(linear_low), 1e-6))),
                    "high": float(
                        _np.log(max(float(linear_high), float(linear_low) + 1e-6))
                    ),
                }

                d0_value = _sample_single_angle_log_d0(log_prior_cfg, target_dtype)
                sampled_values[param_name] = d0_value
                continue

            dist_instance = None
            if prior_spec.dist_type == "BetaScaled":
                low = jnp.asarray(dist_kwargs.pop("low"), dtype=target_dtype)
                high = jnp.asarray(dist_kwargs.pop("high"), dtype=target_dtype)
                scale = jnp.maximum(
                    high - low,
                    jnp.asarray(jnp.finfo(target_dtype).tiny, dtype=target_dtype),
                )
                base_dist = dist.Beta(**dist_kwargs)
                transform = dist_transforms.AffineTransform(loc=low, scale=scale)
                dist_instance = dist.TransformedDistribution(base_dist, transform)
                if dtype_debug_enabled and JAX_AVAILABLE:
                    jax.debug.print(
                        "Parameter '{name}': BetaScaled support=[{low}, {high}]",
                        name=param_name,
                        low=low,
                        high=high,
                    )

            if dist_instance is not None:

                def _sample_from_instance(site_name: str) -> jnp.ndarray:
                    return jnp.asarray(
                        sample(site_name, dist_instance), dtype=target_dtype
                    )

                if per_angle_scaling and param_name in ["contrast", "offset"]:
                    param_values = [
                        _sample_from_instance(f"{param_name}_{phi_idx}")
                        for phi_idx in range(n_phi)
                    ]
                    sampled_values[param_name] = jnp.stack(param_values, axis=0)
                else:
                    sampled_values[param_name] = _sample_from_instance(param_name)
                continue

            if dist_class is dist.TruncatedNormal:
                truncated_keys = ("loc", "scale", "low", "high")

                def _ensure_tensor(val_name: str) -> jnp.ndarray:
                    value = dist_kwargs.get(val_name)
                    if value is None:
                        return None
                    return jnp.asarray(value, dtype=target_dtype)

                truncated_tensors = {
                    key: _ensure_tensor(key)
                    for key in truncated_keys
                    if dist_kwargs.get(key) is not None
                }

                shapes = [tuple(tensor.shape) for tensor in truncated_tensors.values()]
                broadcast_shape = ()
                for shape in shapes:
                    broadcast_shape = np.broadcast_shapes(broadcast_shape, shape)

                def _broadcast(value: jnp.ndarray) -> jnp.ndarray:
                    if value is None:
                        return value
                    if broadcast_shape == ():
                        return jnp.asarray(value, dtype=target_dtype).reshape(())
                    if value.shape == broadcast_shape:
                        return value
                    return jnp.broadcast_to(value, broadcast_shape)

                tiny = jnp.asarray(jnp.finfo(target_dtype).tiny, dtype=target_dtype)
                if "scale" in truncated_tensors:
                    truncated_tensors["scale"] = jnp.maximum(
                        jnp.abs(truncated_tensors["scale"]), tiny
                    )
                if "low" in truncated_tensors and "high" in truncated_tensors:
                    min_interval = tiny
                    truncated_tensors["high"] = jnp.maximum(
                        truncated_tensors["high"],
                        truncated_tensors["low"] + min_interval,
                    )

                for key, tensor in truncated_tensors.items():
                    if tensor is None:
                        continue
                    dist_kwargs[key] = _broadcast(tensor)

                if dtype_debug_enabled and JAX_AVAILABLE:
                    jax.debug.print(
                        "Parameter '{name}': TruncatedNormal support=[{low}, {high}] dtype={dtype} shape={shape}",
                        name=param_name,
                        low=dist_kwargs.get("low"),
                        high=dist_kwargs.get("high"),
                        dtype=str(target_dtype),
                        shape=broadcast_shape,
                    )

            # PER-ANGLE SAMPLING: contrast and offset as separate parameters per phi angle
            if per_angle_scaling and param_name in ["contrast", "offset"]:
                param_values = []
                for phi_idx in range(n_phi):
                    param_name_phi = f"{param_name}_{phi_idx}"
                    param_value_phi = sample(param_name_phi, dist_class(**dist_kwargs))
                    param_values.append(
                        jnp.asarray(param_value_phi, dtype=target_dtype)
                    )
                sampled_values[param_name] = jnp.array(param_values, dtype=target_dtype)
            else:
                param_value = sample(param_name, dist_class(**dist_kwargs))
                param_value = jnp.asarray(param_value, dtype=target_dtype)
                sampled_values[param_name] = param_value

        if reparam_active:
            alpha_value = sampled_values.get("alpha")
            if alpha_value is None:
                raise ValueError(
                    "Single-angle reparameterization requires alpha parameter to be present"
                )
            d_center = jnp.exp(single_angle_latents["log_center"])
            delta_rel = (
                jax.nn.softplus(single_angle_latents["delta_raw"])
                + single_angle_latents["delta_floor"]
            )
            reference_time = jnp.maximum(
                single_angle_latents["t_reference"],
                jnp.asarray(1e-6, dtype=target_dtype),
            )
            alpha_clamped = jnp.clip(alpha_value, -10.0, 10.0)
            denom = jnp.power(reference_time, alpha_clamped)
            denom = jnp.maximum(denom, jnp.asarray(1e-6, dtype=target_dtype))
            d0_reparam = d_center * delta_rel / denom
            d_offset_reparam = d_center * (1.0 - delta_rel)
            deterministic("D0", d0_reparam)
            deterministic("D_offset", d_offset_reparam)
            sampled_values["D0"] = d0_reparam
            sampled_values["D_offset"] = d_offset_reparam

        contrast = sampled_values["contrast"]
        offset = sampled_values["offset"]

        physics_params: list[jnp.ndarray] = []
        for name in param_names_ordered[2:]:
            value = sampled_values[name]
            physics_params.append(jnp.reshape(value, ()))
        params = jnp.stack(physics_params).astype(target_dtype)

        # For backward compatibility with physics functions that expect full params array,
        # we need to prepend mean values of contrast and offset
        if per_angle_scaling:
            # Use mean values for physics computation (theory doesn't need per-angle scaling)
            contrast_mean = jnp.mean(contrast, dtype=target_dtype)
            offset_mean = jnp.mean(offset, dtype=target_dtype)
            params_full = jnp.concatenate(
                [
                    jnp.array([contrast_mean, offset_mean], dtype=target_dtype),
                    params,
                ]
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
        t2_size = t2.shape[0] if hasattr(t2, "shape") else len(t2)
        sigma_size = sigma.shape[0] if hasattr(sigma, "shape") else len(sigma)

        # CRITICAL: All arrays must have the same size
        if not (t1_size == t2_size == data_size == sigma_size):
            import warnings

            warnings.warn(
                f"Model closure array size mismatch: data={data_size}, t1={t1_size}, "
                f"t2={t2_size}, sigma={sigma_size}. This will cause indexing errors.",
                RuntimeWarning,
                stacklevel=2,
            )

        # CRITICAL: phi_array_for_mapping must also match data size for per-angle scaling
        if per_angle_scaling:
            phi_mapping_for_scaling = phi_array_for_mapping
            phi_mapping_for_scaling = _prepare_phi_mapping(
                phi_mapping_for_scaling,
                data_size=data_size,
                n_phi=n_phi,
                phi_unique_np=phi_unique_np,
                target_dtype=target_dtype,
            )

            if phi_mapping_for_scaling is None:
                raise ValueError(
                    "Per-angle scaling requires phi values, but phi array is None"
                )

            phi_mapping_size = len(phi_mapping_for_scaling)
            if phi_mapping_size != data_size:
                import warnings

                warnings.warn(
                    f"Per-angle scaling size mismatch: phi_array_for_mapping={phi_mapping_size}, "
                    f"data={data_size}. This will cause phi indexing errors.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Pass full params array for proper indexing, plus L and dt for correct physics
        # CRITICAL FIX (Nov 2025): physics_cmc.py ALWAYS requires unique phi values
        #
        # physics_cmc.py is specifically designed for CMC element-wise data:
        # - t1/t2 are already element-wise paired (handled internally)
        # - phi must be UNIQUE values for broadcasting: (n_phi_unique, n_points)
        # - Using replicated phi creates (2M × 2M) matrix = 35TB allocation!
        #
        # The phi_array_for_mapping (replicated) is ONLY used for per-angle scaling
        # indexing after c2_theory is computed, NOT for physics computation.
        phi_for_theory = phi  # Already normalized to target dtype

        # DIAGNOSTIC: Log parameter values AND data before physics computation
        # This helps identify which sampled values cause NaN/inf in c2_theory
        def log_params_and_data():
            """Log parameter values and data stats for debugging."""
            import jax

            # params_full has shape (n_params_total,) = (2 + n_physics_params,)
            # For laminar_flow: [contrast_mean, offset_mean, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            jax.debug.print(
                "MCMC params: contrast={c:.4f}, offset={o:.4f}, D0={d0:.2e}, alpha={a:.4f}, D_offset={doff:.2e}, "
                "gamma_dot_t0={g0:.4e}, beta={b:.4f}, gamma_dot_t_offset={goff:.4e}, phi0={p0:.4f}",
                c=params_full[0],
                o=params_full[1],
                d0=params_full[2],
                a=params_full[3],
                doff=params_full[4],
                g0=params_full[5],
                b=params_full[6],
                goff=params_full[7],
                p0=params_full[8],
            )
            jax.debug.print(
                "Input data: t1 range=[{t1_min:.6e}, {t1_max:.6e}], t2 range=[{t2_min:.6e}, {t2_max:.6e}], "
                "t1 has_zero={t1z}, t1 has_neg={t1n}",
                t1_min=jnp.min(t1),
                t1_max=jnp.max(t1),
                t2_min=jnp.min(t2),
                t2_max=jnp.max(t2),
                t1z=jnp.any(t1 == 0.0),
                t1n=jnp.any(t1 < 0.0),
            )
            jax.debug.print(
                "t1[0:10]={t1_sample}, t2[0:10]={t2_sample}",
                t1_sample=t1[:10],
                t2_sample=t2[:10],
            )

        # DISABLED: Excessive output during initialization
        # log_params_and_data()

        c2_theory = _compute_simple_theory_jit(
            params_full, t1, t2, phi_for_theory, q, analysis_mode, L, dt
        )

        # DIAGNOSTIC: Check c2_theory immediately after computation
        def check_c2_theory():
            """Check for NaN/inf in c2_theory and log diagnostics."""
            import jax

            has_nan = jnp.any(jnp.isnan(c2_theory))
            has_inf = jnp.any(jnp.isinf(c2_theory))
            jax.debug.print(
                "c2_theory: shape={shape}, has_nan={nan}, has_inf={inf}, range=[{mn:.6e}, {mx:.6e}]",
                shape=c2_theory.shape,
                nan=has_nan,
                inf=has_inf,
                mn=jnp.nanmin(c2_theory),
                mx=jnp.nanmax(c2_theory),
            )

        # DISABLED: Excessive output during initialization
        # check_c2_theory()

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
            # CRITICAL FIX (Nov 2025): Use nearest-neighbor matching instead of searchsorted
            #
            # PROBLEM with searchsorted:
            # - searchsorted finds INSERTION POINTS to maintain sorted order, NOT nearest matches
            # - Example: phi_unique=[-174.20, -163.56, -154.49], value=-163.57
            #   searchsorted returns 2 (insert after -163.56), but nearest match is index 1
            # - This maps data points to WRONG phi angle rows in c2_theory
            # - Wrong theory values → numerical mismatch → NaN propagation → validation error
            #
            # SOLUTION: argmin-based nearest-neighbor matching
            # - Handles floating-point mismatches (e.g., -154.48506165 vs -154.485)
            # - Always finds the CLOSEST phi value, not insertion point
            # - Memory: (n_data × n_phi × 8 bytes) ≈ 368 MB for 2M points × 23 angles
            phi_array_for_mapping_jax = jnp.atleast_1d(phi_mapping_for_scaling)

            # Compute pairwise distances: shape (n_data, n_phi)
            # Broadcasting: phi_array[:,None] - phi_unique[None,:] creates distance matrix
            distances = jnp.abs(
                phi_array_for_mapping_jax[:, None] - phi_unique_for_sampling[None, :]
            )

            # Find index of nearest phi for each data point: shape (n_data,)
            phi_indices = jnp.argmin(distances, axis=1)

            # Select the appropriate contrast and offset for each data point
            # contrast: shape (n_phi,) → contrast_per_point: shape (n_data_points,)
            # offset: shape (n_phi,) → offset_per_point: shape (n_data_points,)

            # DIAGNOSTIC: Log phi_indices stats before indexing
            def log_phi_indices():
                import jax

                # CRITICAL: Do NOT use jnp.unique() here - causes JAX concretization error during JIT tracing
                jax.debug.print(
                    "phi_indices: shape={shape}, min={mn}, max={mx}",
                    shape=phi_indices.shape,
                    mn=jnp.min(phi_indices),
                    mx=jnp.max(phi_indices),
                )
                jax.debug.print(
                    "contrast array: shape={shape}, values={vals}",
                    shape=contrast.shape,
                    vals=contrast,
                )

            # DISABLED: Excessive output during initialization
            # log_phi_indices()

            contrast_per_point = contrast[phi_indices]
            offset_per_point = offset[phi_indices]

            # CRITICAL FIX (Nov 10, 2025): Handle both 1D and 2D c2_theory
            # - laminar_flow mode: c2_theory is 2D (n_phi, n_data) - angle-dependent
            # - static mode: c2_theory is 1D (n_data,) - angle-independent
            #
            # For static mode, c2_theory is the same for all angles (diffusion only),
            # so we don't need phi indexing - just use it directly.
            # Only the contrast/offset scaling is per-angle.
            n_data_points = phi_indices.shape[0]

            if c2_theory.ndim == 2:
                # laminar_flow mode: 2D theory (n_phi, n_data)
                # Extract the appropriate angle's row for each data point
                # Advanced indexing: c2_theory[phi_indices, range(len)] → shape (n_data,)
                c2_theory_per_point = c2_theory[phi_indices, jnp.arange(n_data_points)]
            else:
                # static mode: 1D theory (n_data,)
                # Theory is angle-independent (same for all phi)
                # No indexing needed - use directly
                c2_theory_per_point = c2_theory

            # DIAGNOSTIC: Check c2_theory_per_point extraction
            def check_c2_extraction():
                """Check if c2_theory extraction is working correctly."""
                import jax

                jax.debug.print(
                    "c2_theory_per_point: shape={shape}, has_nan={nan}, has_inf={inf}, range=[{mn:.6e}, {mx:.6e}]",
                    shape=c2_theory_per_point.shape,
                    nan=jnp.any(jnp.isnan(c2_theory_per_point)),
                    inf=jnp.any(jnp.isinf(c2_theory_per_point)),
                    mn=jnp.nanmin(c2_theory_per_point),
                    mx=jnp.nanmax(c2_theory_per_point),
                )
                jax.debug.print(
                    "c2_theory_per_point[0:10]={sample}",
                    sample=c2_theory_per_point[:10],
                )
                # Also check the raw c2_theory for comparison (handle both 1D and 2D)
                if c2_theory.ndim == 2:
                    # laminar_flow: 2D c2_theory
                    jax.debug.print(
                        "c2_theory[0, 0:10]={row0}, c2_theory[0, -10:]={row0_end}",
                        row0=c2_theory[0, :10],
                        row0_end=c2_theory[0, -10:],
                    )
                    jax.debug.print(
                        "c2_theory[0, middle]={mid}, unique_vals_approx={uniq}",
                        mid=c2_theory[0, n_data_points // 2 : n_data_points // 2 + 10],
                        uniq=jnp.array(
                            [
                                jnp.min(c2_theory),
                                jnp.max(c2_theory),
                                jnp.mean(c2_theory),
                            ]
                        ),
                    )
                else:
                    # static: 1D c2_theory
                    jax.debug.print(
                        "c2_theory[0:10]={start}, c2_theory[-10:]={end}",
                        start=c2_theory[:10],
                        end=c2_theory[-10:],
                    )
                    jax.debug.print(
                        "c2_theory[middle]={mid}, unique_vals_approx={uniq}",
                        mid=c2_theory[n_data_points // 2 : n_data_points // 2 + 10],
                        uniq=jnp.array(
                            [
                                jnp.min(c2_theory),
                                jnp.max(c2_theory),
                                jnp.mean(c2_theory),
                            ]
                        ),
                    )

            # DISABLED: Excessive output during initialization
            # check_c2_extraction()

            # Apply per-angle scaling to flattened c2_theory
            # CRITICAL FIX: c2_theory = 1 + g1², so g1² = c2_theory - 1
            # Correct physics: c2_fitted = offset + contrast * g1²
            #                            = offset + contrast * (c2_theory - 1)
            # WRONG (previous): c2_fitted = contrast * c2_theory + offset
            #                              = contrast * (1 + g1²) + offset
            #                              = contrast + contrast*g1² + offset  ← Extra "contrast" term!
            c2_theory_for_likelihood = c2_theory_per_point
            g1_squared = c2_theory_for_likelihood - 1.0
            c2_fitted = offset_per_point + contrast_per_point * g1_squared
        else:
            # LEGACY BEHAVIOR: Global contrast and offset (shared across all angles)
            c2_theory_for_likelihood = c2_theory
            g1_squared = c2_theory_for_likelihood - 1.0
            c2_fitted = offset + contrast * g1_squared

        c2_fitted_raw = c2_fitted

        # CRITICAL VALIDATION: Ensure c2_fitted matches data shape before sampling
        # If shapes mismatch, NumPyro will fail with "invalid loc parameter" error
        # This catches bugs in phi_indices indexing or c2_theory shape issues
        expected_shape = data.shape
        if c2_fitted.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in MCMC model: c2_fitted.shape={c2_fitted.shape} "
                f"but data.shape={expected_shape}. This indicates a bug in "
                f"per-angle scaling indexing or c2_theory computation."
            )

        # NOTE: Python 'if' statements with JAX arrays don't work during JIT tracing
        # NumPyro will automatically handle NaN/inf by rejecting those samples during MCMC
        # The jax.debug.print statements above already log has_nan/has_inf status for diagnostics

        # DIAGNOSTIC: Check sigma and c2_fitted before likelihood sampling
        def check_likelihood_inputs():
            """Check sigma and c2_fitted for issues that would cause likelihood failure."""
            import jax

            jax.debug.print(
                "Likelihood inputs: c2_fitted shape={cf_shape}, sigma shape={s_shape}",
                cf_shape=c2_fitted.shape,
                s_shape=sigma.shape,
            )
            jax.debug.print(
                "c2_fitted: has_nan={cf_nan}, has_inf={cf_inf}, range=[{cf_min:.6e}, {cf_max:.6e}], mean={cf_mean:.6f}",
                cf_nan=jnp.any(jnp.isnan(c2_fitted)),
                cf_inf=jnp.any(jnp.isinf(c2_fitted)),
                cf_min=jnp.nanmin(c2_fitted),
                cf_max=jnp.nanmax(c2_fitted),
                cf_mean=jnp.mean(c2_fitted),
            )
            # Sample values from c2_fitted for detailed inspection
            jax.debug.print("c2_fitted[0:10]={sample}", sample=c2_fitted[:10])
            jax.debug.print("data[0:10]={sample}", sample=data[:10])
            jax.debug.print(
                "sigma: has_zero={s_zero}, has_neg={s_neg}, has_nan={s_nan}, has_inf={s_inf}, range=[{s_min:.6e}, {s_max:.6e}]",
                s_zero=jnp.any(sigma == 0.0),
                s_neg=jnp.any(sigma < 0.0),
                s_nan=jnp.any(jnp.isnan(sigma)),
                s_inf=jnp.any(jnp.isinf(sigma)),
                s_min=jnp.nanmin(sigma),
                s_max=jnp.nanmax(sigma),
            )

        # DISABLED: Excessive output during initialization
        # check_likelihood_inputs()

        # ------------------------------------------------------------------
        # Likelihood guards + targeted diagnostics (behind HOMODYNE_DEBUG_INIT)
        # ------------------------------------------------------------------

        def _ensure_finite(values):
            """Replace non-finite entries with the mean of finite values (or 1.0)."""
            finite_mask = jnp.isfinite(values)
            finite_sum = jnp.sum(
                jnp.where(finite_mask, values, 0.0), dtype=values.dtype
            )
            finite_count = jnp.sum(finite_mask)
            finite_count_safe = jnp.maximum(
                jnp.asarray(finite_count, dtype=values.dtype),
                jnp.asarray(1.0, dtype=values.dtype),
            )
            safe_mean = jnp.where(
                finite_count > 0,
                finite_sum / finite_count_safe,
                jnp.asarray(1.0, dtype=values.dtype),
            )
            sanitized = jnp.where(finite_mask, values, safe_mean)
            invalid_count = jnp.sum(~finite_mask)
            return sanitized, invalid_count, finite_mask

        def _ensure_positive_sigma(values):
            sigma_arr = jnp.asarray(values)
            eps = jnp.asarray(1e-12, dtype=sigma_arr.dtype)
            nan_mask = ~jnp.isfinite(sigma_arr)
            nonpos_mask = sigma_arr <= eps
            invalid_mask = jnp.logical_or(nan_mask, nonpos_mask)
            sigma_safe = jnp.where(invalid_mask, eps, sigma_arr)
            invalid_count = jnp.sum(invalid_mask)
            return sigma_safe, invalid_count, invalid_mask

        sigma_safe, sigma_invalid_count, _ = _ensure_positive_sigma(sigma)
        c2_theory_safe, c2_theory_invalid_count, _ = _ensure_finite(
            c2_theory_for_likelihood
        )

        if per_angle_scaling:
            c2_fitted_from_theory = (
                contrast_per_point * c2_theory_safe + offset_per_point
            )
        else:
            c2_fitted_from_theory = contrast * c2_theory_safe + offset

        c2_fitted_safe, c2_fitted_invalid_count, _ = _ensure_finite(
            c2_fitted_from_theory
        )

        def log_obs_site_stats():
            """Emit lightweight diagnostics for the obs site under debug flag."""
            debug_flag = os.environ.get("HOMODYNE_DEBUG_INIT", "0").lower()
            diagnostics_enabled = debug_flag not in {"", "0", "false"}
            if not diagnostics_enabled:
                return

            import jax

            raw_theory = c2_theory_for_likelihood
            raw_fitted = c2_fitted_raw

            jax.debug.print(
                "OBS DEBUG :: sigma invalid={n_bad}, clamp_min={mn:.6e}, clamp_max={mx:.6e}",
                n_bad=sigma_invalid_count,
                mn=jnp.nanmin(sigma_safe),
                mx=jnp.nanmax(sigma_safe),
            )
            jax.debug.print(
                "OBS DEBUG :: c2_theory raw_has_nan={raw_nan}, raw_has_inf={raw_inf}, "
                "sanitized_invalid={n_bad}",
                raw_nan=jnp.any(jnp.isnan(raw_theory)),
                raw_inf=jnp.any(jnp.isinf(raw_theory)),
                n_bad=c2_theory_invalid_count,
            )
            jax.debug.print(
                "OBS DEBUG :: c2_fitted raw_has_nan={raw_nan}, raw_has_inf={raw_inf}, "
                "sanitized_invalid={n_bad}",
                raw_nan=jnp.any(jnp.isnan(raw_fitted)),
                raw_inf=jnp.any(jnp.isinf(raw_fitted)),
                n_bad=c2_fitted_invalid_count,
            )
            jax.debug.print(
                "OBS DEBUG :: c2_fitted_safe range=[{mn:.6e}, {mx:.6e}], mean={mean:.6e}",
                mn=jnp.nanmin(c2_fitted_safe),
                mx=jnp.nanmax(c2_fitted_safe),
                mean=jnp.nanmean(c2_fitted_safe),
            )

        # DISABLED: Excessive output during initialization
        # log_obs_site_stats()

        # Likelihood with guarded inputs
        sample("obs", dist.Normal(c2_fitted_safe, sigma_safe), obs=data)

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
        Analysis mode: 'static' or 'laminar_flow'
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


def _format_init_params_for_chains(
    initial_values: dict[str, float] | None,
    n_chains: int,
    jitter_scale: float,
    rng_key,
    parameter_space: ParameterSpace | None = None,
):
    """Broadcast initial parameters across chains with optional jitter."""
    if initial_values is None:
        return None, rng_key

    jitter_scale = max(float(jitter_scale or 0.0), 0.0)
    formatted: dict[str, Any] = {}
    current_key = rng_key

    for param, value in initial_values.items():
        base = jnp.full((n_chains,), float(value), dtype=jnp.float64)

        if jitter_scale > 0.0 and np.isfinite(value):
            current_key, subkey = random.split(current_key)
            perturb = jitter_scale * random.normal(subkey, shape=(n_chains,))
            if value != 0.0:
                base = base * (1.0 + perturb)
            else:
                base = base + perturb

        if parameter_space is not None:
            try:
                lower, upper = parameter_space.get_bounds(param)
            except KeyError:
                lower = upper = None
            if lower is not None and upper is not None:
                epsilon = 1e-9 * max(1.0, abs(upper - lower))
                base = jnp.clip(base, lower + epsilon, upper - epsilon)

        formatted[param] = base

    return formatted, current_key


def _run_numpyro_sampling(
    model,
    config,
    initial_values=None,
    parameter_space=None,
    phi_unique=None,
    per_angle_scaling: bool = True,
):
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
    parameter_space : ParameterSpace, optional
        Parameter space defining priors and parameter bounds
    phi_unique : np.ndarray, optional
        Unique phi angles for per-angle parameter expansion
    per_angle_scaling : bool, default True
        Whether the NumPyro model samples per-angle contrast/offset parameters. Disabled
        automatically for single-angle static datasets to prevent over-parameterization.

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
    - Parameters are used directly without validation (user responsible for physical validity)
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

            if platform == "cpu":
                # CPU mode: check if XLA_FLAGS configured multiple devices
                # XLA_FLAGS should be set in homodyne/__init__.py before JAX import
                if n_devices >= n_chains:
                    logger.info(
                        f"Using {n_devices} CPU devices for {n_chains} parallel MCMC chains",
                    )
                elif n_devices == 1:
                    logger.warning(
                        f"Only 1 CPU device detected, but {n_chains} chains requested. "
                        f"XLA_FLAGS may not be set correctly. "
                        f"Set environment variable before running homodyne:\n"
                        f"  export XLA_FLAGS='--xla_force_host_platform_device_count={n_chains}'\n"
                        f"Falling back to num_chains=1."
                    )
                    n_chains = 1
                    config["n_chains"] = 1  # Update config to reflect fallback
                else:
                    logger.info(
                        f"Using {n_devices} CPU devices for {n_chains} parallel chains "
                        f"(fewer devices than requested)"
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

    if (
        not per_angle_scaling
        and initial_values is not None
        and parameter_space is not None
    ):
        # Ensure scalar contrast/offset exist for deterministic initialization
        if "contrast" not in initial_values:
            try:
                contrast_midpoint = sum(parameter_space.get_bounds("contrast")) / 2.0
                initial_values["contrast"] = parameter_space.clamp_to_open_interval(
                    "contrast", contrast_midpoint, epsilon=1e-6
                )
            except KeyError:
                initial_values["contrast"] = 0.5
        if "offset" not in initial_values:
            try:
                offset_midpoint = sum(parameter_space.get_bounds("offset")) / 2.0
                initial_values["offset"] = parameter_space.clamp_to_open_interval(
                    "offset", offset_midpoint, epsilon=1e-6
                )
            except KeyError:
                initial_values["offset"] = 1.0
    if per_angle_scaling and initial_values is not None and parameter_space is not None:
        logger.info(
            f"Expanding initial_values for per-angle scaling: {list(initial_values.keys())}"
        )

        if phi_unique is not None:
            n_phi = len(phi_unique)
        else:
            inferred = [
                name
                for name in initial_values
                if name.startswith("contrast_") or name.startswith("offset_")
            ]
            n_phi = len(
                {name.split("_")[-1] for name in inferred if name.count("_") == 1}
            )

        if n_phi > 0:
            for phi_idx in range(n_phi):
                contrast_key = f"contrast_{phi_idx}"
                offset_key = f"offset_{phi_idx}"

                if contrast_key not in initial_values:
                    try:
                        contrast_midpoint = (
                            sum(parameter_space.get_bounds("contrast")) / 2.0
                        )
                        initial_values[contrast_key] = (
                            parameter_space.clamp_to_open_interval(
                                "contrast", contrast_midpoint, epsilon=1e-6
                            )
                        )
                    except KeyError:
                        initial_values[contrast_key] = 0.5

                if offset_key not in initial_values:
                    try:
                        offset_midpoint = (
                            sum(parameter_space.get_bounds("offset")) / 2.0
                        )
                        initial_values[offset_key] = (
                            parameter_space.clamp_to_open_interval(
                                "offset", offset_midpoint, epsilon=1e-6
                            )
                        )
                    except KeyError:
                        initial_values[offset_key] = 1.0

            removed = False
            if initial_values.pop("contrast", None) is not None:
                removed = True
            if initial_values.pop("offset", None) is not None:
                removed = True

            if removed:
                logger.info(
                    "Removed base contrast/offset entries after per-angle expansion"
                )

            logger.info(
                f"Ensured {n_phi} per-angle parameters for contrast and offset in initial_values"
            )
        else:
            logger.debug("No per-angle expansion needed (n_phi=0)")
    elif not per_angle_scaling and initial_values is not None:
        logger.debug(
            "Per-angle scaling disabled; using scalar contrast/offset initial values"
        )
    elif initial_values is not None:
        logger.debug(
            "ParameterSpace not available; skipping per-angle initial value expansion"
        )

    # Use parameters directly without validation/repair
    # Per user request: Do not limit parameter space or auto-correct for numerical instability
    if initial_values is not None and parameter_space is not None:
        logger.info(
            f"Using init parameters directly (no validation), {len(initial_values)} parameters ready for NUTS initialization"
        )
    elif initial_values is None:
        logger.info("Using NumPyro default initialization (sampling from priors)")
    else:
        logger.info(f"Using initial_values with {len(initial_values)} parameters")

    # CRITICAL FIX: NumPyro init_params requires arrays of shape (num_chains, ...)
    # Convert scalar initial values to broadcasted arrays for each chain
    init_params_formatted, rng_key = _format_init_params_for_chains(
        initial_values,
        config["n_chains"],
        config.get("init_jitter_scale", 0.0),
        rng_key,
        parameter_space,
    )
    if init_params_formatted is not None:
        logger.debug(
            f"Formatted init_params for {config['n_chains']} chains: {list(init_params_formatted.keys())}"
        )

    try:
        # Pass initial_values to mcmc.run() for chain initialization
        # NumPyro will use these as starting points for all chains
        mcmc.run(
            rng_key,
            init_params=init_params_formatted,  # Initialize chains at these parameter values (broadc asted)
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
        error_msg = str(e)

        # Check if this is the "not enough devices" error
        if "not enough devices" in error_msg.lower() and config["n_chains"] > 1:
            logger.warning(
                f"Parallel chains failed ({config['n_chains']} chains requested but only 1 device available). "
                f"Retrying with num_chains=1 (sequential execution)..."
            )

            # Recreate MCMC with num_chains=1
            mcmc = MCMC(
                nuts_kernel,
                num_warmup=config["n_warmup"],
                num_samples=config["n_samples"],
                num_chains=1,  # Fall back to single chain
                progress_bar=True,
            )

            # Retry with single chain - reformat init_params for 1 chain
            init_params_single = None
            if initial_values is not None:
                import jax.numpy as jnp

                init_params_single = {
                    param: jnp.full((1,), value, dtype=jnp.float64)
                    for param, value in initial_values.items()
                }

            mcmc.run(
                rng_key,
                init_params=init_params_single,
                extra_fields=(
                    "potential_energy",
                    "accept_prob",
                    "diverging",
                    "num_steps",
                ),
            )

            logger.info("MCMC completed successfully with num_chains=1")
            _log_warmup_diagnostics(mcmc)
        else:
            # Log full error details including traceback for debugging
            import traceback

            logger.error(f"MCMC sampling failed: {str(e)}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
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
            total_transitions = extra_fields["diverging"].size
            if n_divergences > 0:
                divergence_pct = (n_divergences / max(total_transitions, 1)) * 100.0
                logger.warning(
                    f"MCMC had {n_divergences} divergent transitions ({divergence_pct:.1f}%). "
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
            max_hits = float(jnp.mean(extra_fields["num_steps"] == max_steps)) * 100.0
            logger.info(
                f"Mean tree depth: {mean_steps:.1f}, Max: {max_steps} (chains hit max {max_hits:.1f}% of the time)"
            )

        # Log adaptation state (step size + mass matrix conditioning)
        last_state = getattr(mcmc, "last_state", None)
        adapt_state = getattr(last_state, "adapt_state", None)
        if adapt_state is not None:
            try:
                step_size = float(np.asarray(adapt_state.step_size))
                logger.info(f"Final step size after adaptation: {step_size:.4e}")
            except Exception:
                logger.debug("Could not read adapted step size")

            try:
                inv_mass = np.asarray(adapt_state.inverse_mass_matrix)
                if inv_mass.ndim == 1:
                    positive = inv_mass[inv_mass > 0]
                    if positive.size:
                        ratio = float(np.max(positive) / np.min(positive))
                        logger.info(
                            f"Mass matrix diagonal range: [{np.min(positive):.3e}, {np.max(positive):.3e}] (ratio={ratio:.2e})"
                        )
                        if ratio > 1e4:
                            logger.warning(
                                "Mass matrix diagonal is ill-conditioned. "
                                "Consider rescaling parameters or enabling dense mass matrix."
                            )
                else:
                    eigenvalues = np.linalg.eigvalsh(inv_mass)
                    positive = eigenvalues[eigenvalues > 0]
                    if positive.size:
                        cond_number = float(np.max(positive) / np.min(positive))
                        logger.info(
                            f"Mass matrix condition number: {cond_number:.2e} (dense)"
                        )
                        if cond_number > 1e4:
                            logger.warning(
                                "Dense mass matrix is ill-conditioned. Consider reparameterization or stronger priors."
                            )
            except Exception as exc:
                logger.debug(f"Could not inspect mass matrix conditioning: {exc}")

    except Exception as e:
        logger.debug(f"Could not extract warmup diagnostics: {e}")


def _run_blackjax_sampling(model, config):
    """Run BlackJAX MCMC sampling."""
    # Placeholder for BlackJAX implementation
    # Would need to convert NumPyro model to BlackJAX format
    raise NotImplementedError("BlackJAX sampling not yet implemented")


def _process_posterior_samples(
    mcmc_result, analysis_mode, diagnostic_settings: dict[str, Any] | None = None
):
    """Process posterior samples to extract summary statistics and diagnostics."""
    diagnostic_settings = diagnostic_settings or {}
    max_rhat_threshold = float(diagnostic_settings.get("max_rhat", 1.1))
    min_ess_threshold = float(diagnostic_settings.get("min_ess", 100))
    check_hmc = bool(diagnostic_settings.get("check_hmc_diagnostics", True))

    raw_samples = mcmc_result.get_samples()
    samples: dict[str, Any] = {}
    invalid_params: list[str] = []
    scaling_overrides = diagnostic_settings.get("scaling_overrides") or {}
    surrogate_cfg = diagnostic_settings.get("single_angle_surrogate") or {}
    deterministic_params: set[str] = set(
        diagnostic_settings.get("deterministic_params") or []
    )
    deterministic_params.update(scaling_overrides.keys())
    if (
        surrogate_cfg.get("drop_d_offset")
        or surrogate_cfg.get("fixed_d_offset") is not None
    ):
        deterministic_params.add("D_offset")
    if surrogate_cfg.get("fixed_alpha") is not None:
        deterministic_params.add("alpha")
    constructed_deterministic: set[str] = set()

    for param_name, param_values in raw_samples.items():
        arr_np = np.asarray(param_values)
        finite_mask = np.isfinite(arr_np)
        if not finite_mask.all():
            invalid_count = arr_np.size - np.count_nonzero(finite_mask)
            invalid_params.append(f"{param_name} ({invalid_count} invalid)")
            if np.any(finite_mask):
                replacement = float(np.median(arr_np[finite_mask]))
            else:
                replacement = 0.0
            arr_np = np.where(finite_mask, arr_np, replacement)
        samples[param_name] = jnp.asarray(arr_np)

    sample_count = len(next(iter(samples.values()))) if samples else 0
    expected_params = diagnostic_settings.get(
        "expected_params"
    ) or _get_physical_param_order(analysis_mode)

    # Handle log-space D0 sampling (legacy and new implementations)
    if "D0" not in samples:
        # New implementation (ExpTransform): log_D0_latent is emitted as deterministic
        if "log_D0_latent" in samples:
            samples["D0"] = jnp.exp(samples["log_D0_latent"])
        # Legacy implementation: single_angle_log_d0_value
        elif "single_angle_log_d0_value" in samples:
            samples["D0"] = jnp.exp(samples["single_angle_log_d0_value"])

    if "D_offset" not in samples and surrogate_cfg.get("fixed_d_offset") is not None:
        fixed_offset = float(surrogate_cfg["fixed_d_offset"])
        if sample_count:
            samples["D_offset"] = jnp.full(
                (sample_count,), fixed_offset, dtype=jnp.float64
            )
            constructed_deterministic.add("D_offset")

    def _ensure_param_array(name: str) -> jnp.ndarray | None:
        arr = samples.get(name)
        if arr is None:
            return None
        return jnp.asarray(arr)

    # Extract parameter samples using expected order but skipping missing entries.
    param_arrays: list[jnp.ndarray] = []
    used_param_names: list[str] = []
    for param_name in expected_params:
        arr = _ensure_param_array(param_name)
        filled_constant = False
        if arr is None:
            if (
                param_name == "D_offset"
                and surrogate_cfg.get("fixed_d_offset") is not None
            ):
                fixed_offset = float(surrogate_cfg["fixed_d_offset"])
                arr = jnp.full((sample_count,), fixed_offset, dtype=jnp.float64)
                samples[param_name] = arr
                filled_constant = True
            else:
                logger.debug("Skipping missing parameter '%s' in summary", param_name)
                continue
        param_arrays.append(arr)
        used_param_names.append(param_name)
        if filled_constant:
            constructed_deterministic.add(param_name)

    if param_arrays:
        param_samples = jnp.column_stack(param_arrays)
    else:
        param_samples = jnp.empty((sample_count, 0), dtype=jnp.float64)

    per_angle_scaling_detected = any(k.startswith("contrast_") for k in samples)

    if per_angle_scaling_detected:
        contrast_keys = sorted([k for k in samples.keys() if k.startswith("contrast_")])
        offset_keys = sorted([k for k in samples.keys() if k.startswith("offset_")])

        if not contrast_keys or not offset_keys:
            logger.error(
                "Incomplete per-angle parameters: contrast_keys=%s, offset_keys=%s",
                contrast_keys,
                offset_keys,
            )
            raise ValueError(
                f"Incomplete per-angle scaling parameters. "
                f"Found {len(contrast_keys)} contrast parameters and {len(offset_keys)} offset parameters. "
                f"Expected matching counts for all phi angles."
            )

        contrast_samples_per_angle = jnp.stack(
            [samples[k] for k in contrast_keys], axis=1
        )
        offset_samples_per_angle = jnp.stack([samples[k] for k in offset_keys], axis=1)

        contrast_samples = contrast_samples_per_angle.flatten()
        offset_samples = offset_samples_per_angle.flatten()

        logger.debug(
            f"Per-angle scaling validated: {len(contrast_keys)} angles, "
            f"{contrast_samples_per_angle.shape[0]} samples per angle, total {len(contrast_samples)} contrast samples",
        )
    else:
        contrast_samples = samples.get("contrast")
        offset_samples = samples.get("offset")
        if contrast_samples is None or offset_samples is None:
            available_keys = list(samples.keys())
            logger.error(
                "Scalar contrast/offset parameters not found in MCMC samples. Available keys: %s",
                available_keys,
            )
            raise ValueError(
                "MCMC sampling failed: Expected scalar 'contrast'/'offset' parameters for single-angle mode."
            )
        logger.debug(
            "Scalar scaling mode detected: using %s contrast samples and %s offset samples",
            len(contrast_samples),
            len(offset_samples),
        )
        constructed_deterministic.update(
            [
                name
                for name in ["contrast", "offset"]
                if name in scaling_overrides or name in deterministic_params
            ]
        )

    # Compute summary statistics
    mean_params = jnp.mean(param_samples, axis=0) if param_arrays else jnp.array([])
    std_params = jnp.std(param_samples, axis=0) if param_arrays else jnp.array([])
    mean_contrast = float(jnp.mean(contrast_samples))
    std_contrast = float(jnp.std(contrast_samples))
    mean_offset = float(jnp.mean(offset_samples))
    std_offset = float(jnp.std(offset_samples))

    # Compute MCMC diagnostics
    try:
        extra_fields = mcmc_result.get_extra_fields()
    except Exception:
        extra_fields = {}

    acceptance_rate = None
    if "accept_prob" in extra_fields:
        acceptance_rate = float(jnp.mean(extra_fields["accept_prob"]))
    else:
        logger.warning("Acceptance probability not available in MCMC diagnostics")

    diagnostics_blocked = bool(invalid_params)
    if diagnostics_blocked:
        logger.error(
            "Detected invalid MCMC samples: %s. Skipping convergence diagnostics.",
            ", ".join(invalid_params),
        )

    deterministic_params.update(constructed_deterministic)

    r_hat_dict = dict.fromkeys(samples.keys())
    ess_dict = dict.fromkeys(samples.keys())
    per_param_stats: dict[str, dict[str, Any]] = {
        name: {"deterministic": name in deterministic_params} for name in samples.keys()
    }
    converged = not diagnostics_blocked

    try:
        samples_with_chains = (
            mcmc_result.get_samples(group_by_chain=True)
            if not diagnostics_blocked
            else {}
        )
    except Exception:
        samples_with_chains = {}

    def _samples_with_chain_dim(param_name: str):
        if param_name in samples_with_chains:
            return samples_with_chains[param_name]
        value = samples.get(param_name)
        if value is None:
            return None
        return jnp.reshape(value, (1,) + value.shape)

    chain_example = None
    if samples_with_chains:
        chain_example = next(iter(samples_with_chains.values()))
    elif samples:
        first = next(iter(samples.values()))
        chain_example = jnp.reshape(first, (1,) + first.shape)

    num_chains = chain_example.shape[0] if chain_example is not None else 0
    multi_chain = num_chains > 1

    if diagnostics_blocked:
        pass
    elif not check_hmc:
        logger.info(
            "HMC diagnostics disabled by configuration; treating run as converged"
        )
    else:
        try:
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            if not multi_chain:
                logger.info(
                    "Single chain detected - R-hat not available; computing ESS only"
                )

            for param_name in samples.keys():
                arr = _samples_with_chain_dim(param_name)
                if arr is None:
                    per_param_stats[param_name]["r_hat"] = None
                    per_param_stats[param_name]["ess"] = None
                    continue
                if param_name in deterministic_params:
                    r_hat_dict[param_name] = None
                    ess_dict[param_name] = None
                    per_param_stats[param_name]["r_hat"] = None
                    per_param_stats[param_name]["ess"] = None
                    continue
                try:
                    if multi_chain:
                        r_hat_dict[param_name] = float(gelman_rubin(arr))
                        ess_val = float(effective_sample_size(arr))
                    else:
                        ess_val = float(effective_sample_size(arr))
                    if not np.isfinite(ess_val) or ess_val < 0:
                        ess_val = 0.0
                    ess_dict[param_name] = ess_val
                    per_param_stats[param_name]["r_hat"] = r_hat_dict.get(param_name)
                    per_param_stats[param_name]["ess"] = ess_val
                except Exception as exc:
                    logger.warning(
                        f"Could not compute diagnostics for {param_name}: {exc}"
                    )
                    r_hat_dict[param_name] = None
                    ess_dict[param_name] = ess_dict.get(param_name)
                    per_param_stats[param_name]["r_hat"] = None
                    per_param_stats[param_name]["ess"] = ess_dict.get(param_name)

            diag_rhat_values = [v for v in r_hat_dict.values() if v is not None]
            if diag_rhat_values:
                max_r_hat = max(diag_rhat_values)
                if max_r_hat > max_rhat_threshold:
                    logger.warning(
                        f"Poor convergence detected: max R-hat = {max_r_hat:.3f} > {max_rhat_threshold:.3f}"
                    )
                    converged = False

            diag_ess_values = [v for v in ess_dict.values() if v is not None]
            if diag_ess_values:
                min_ess = min(diag_ess_values)
                if min_ess < min_ess_threshold:
                    logger.warning(
                        f"Poor sampling efficiency: min ESS = {min_ess:.0f} < {min_ess_threshold:.0f}"
                    )
                    converged = False

        except Exception as e:
            logger.error(f"Failed to compute MCMC diagnostics: {e}")
            converged = False

    for stats in per_param_stats.values():
        stats.setdefault("r_hat", None)
        stats.setdefault("ess", None)

    diagnostic_summary = {
        "deterministic_params": sorted(deterministic_params),
        "per_param_stats": per_param_stats,
        "multi_chain": multi_chain,
        "surrogate_thresholds": diagnostic_settings.get("surrogate_thresholds"),
        "focus_params": (
            (diagnostic_settings.get("surrogate_thresholds") or {}).get("focus_params")
        ),
    }

    return {
        "mean_params": np.array(mean_params),
        "std_params": np.array(std_params),
        "param_names": used_param_names,
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
        "diagnostic_summary": diagnostic_summary,
        "deterministic_params": diagnostic_summary["deterministic_params"],
        "multi_chain": multi_chain,
    }
