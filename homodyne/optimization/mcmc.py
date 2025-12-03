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
from typing import Any, Dict, Tuple

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


def _estimate_single_angle_scaling(data: Any) -> Tuple[float, float]:
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
    interval_width = max(high_value - low_value, 1e-6)
    eps = 1e-6

    loc = jnp.asarray(loc_value, dtype=target_dtype)
    scale = jnp.asarray(scale_value, dtype=target_dtype)
    low = jnp.asarray(low_value + eps, dtype=target_dtype)
    high = jnp.asarray(high_value - eps, dtype=target_dtype)

    # Create truncated Normal distribution in log-space
    log_space_dist = dist.TruncatedNormal(
        loc=loc,
        scale=scale,
        low=low,
        high=high
    )

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
    from numpyro.distributions import transforms as dist_transforms
    from numpyro import sample, deterministic, prng_key
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




# ---------------------------------------------------------------------------
# CMC-only entrypoint (overrides legacy mixed-mode definitions above)
# ---------------------------------------------------------------------------
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
    seed_base: int = 0,
    **kwargs,
) -> MCMCResult:
    """CMC-only Bayesian fitting using the CMC coordinator."""

    if initial_values is not None and parameter_space is not None:
        is_valid, violations = parameter_space.validate_values(initial_values)
        if not is_valid:
            raise ValueError(
                f"Initial parameter values violate bounds:\n" + "\n".join(violations)
            )
        logger.info("Initial parameter values validated successfully (all within bounds)")

    dataset_size = data.size if hasattr(data, "size") else len(data)
    logger.info(f"Dataset size: {dataset_size:,} data points")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Per-phi initial values: config-first then percentile fallback
    if phi is not None:
        if initial_values is None:
            initial_values = {}
        has_per_phi = any(k.startswith("contrast_") or k.startswith("offset_") for k in initial_values)
        if not has_per_phi:
            cfg = PerPhiInitConfig(seed_base=int(seed_base))
            per_phi_vals, _ = build_per_phi_initial_values(
                phi=np.asarray(phi),
                g2=data,
                config_per_phi=initial_values.get("per_phi") if isinstance(initial_values, dict) else None,
                cfg=cfg,
            )
            initial_values.update(per_phi_vals)

    from homodyne.optimization.cmc.coordinator import CMCCoordinator

    cmc_config = kwargs.pop("cmc_config", {})
    if "mcmc" not in cmc_config:
        cmc_config["mcmc"] = _get_mcmc_config(kwargs)

    coordinator = CMCCoordinator(cmc_config)
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
        seed_base=seed_base,
    )

    logger.info(f"CMC execution completed. Used {result.num_shards} shard(s).")
    return result
