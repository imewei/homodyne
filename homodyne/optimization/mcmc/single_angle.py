"""Single-angle surrogate mode utilities for NumPyro MCMC models.

This module extracts single-angle handling logic from _create_numpyro_model
and fit_mcmc_jax to reduce cyclomatic complexity and improve maintainability.

For single-angle datasets (phi_count == 1), special handling is needed:
- Disable per-angle scaling (redundant with single angle)
- Use log-space D0 sampling for better MCMC geometry
- Apply surrogate tiers for varying difficulty levels
- Fix certain parameters to reduce dimensionality

Extracted from mcmc.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro import deterministic, sample
from numpyro.distributions.transforms import ExpTransform

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

logger = get_logger(__name__)

# Scaling parameter clamp ranges for single-angle fallback
_FIXED_CONTRAST_RANGE = (0.1, 0.9)
_FIXED_OFFSET_RANGE = (0.7, 1.3)


def estimate_single_angle_scaling(data: Any) -> tuple[float, float]:
    """Estimate deterministic contrast/offset for phi_count==1 fallback.

    For single-angle datasets, we estimate fixed scaling parameters from
    data statistics since per-angle scaling would be redundant.

    Parameters
    ----------
    data : array-like
        Experimental correlation data.

    Returns
    -------
    tuple[float, float]
        (contrast, offset) - Estimated scaling parameters.

    Notes
    -----
    Uses 1st and 99th percentiles to estimate the data range,
    then derives contrast and offset from that range.
    """
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


def build_surrogate_settings(
    parameter_space: "ParameterSpace | None",
    tier: str,
) -> dict[str, Any]:
    """Construct surrogate configuration for phi_count==1.

    Different tiers provide varying levels of simplification:
    - Tier 1: NLSQ-friendly mode (keeps all params, no log-space sampling)
    - Tier 2: Drop D_offset, use log-space D0, tighten alpha prior
    - Tier 3: Fix alpha to -1.2, tighter log-D0 scale
    - Tier 4: Tightest constraints for difficult cases

    Parameters
    ----------
    parameter_space : ParameterSpace | None
        Parameter space with prior configurations.
    tier : str
        Surrogate tier ("1", "2", "3", or "4").

    Returns
    -------
    dict[str, Any]
        Surrogate configuration with keys:
        - tier: Normalized tier string
        - drop_d_offset: Whether to fix D_offset to 0
        - sample_log_d0: Whether to use log-space D0 sampling
        - log_d0_prior: Configuration for log-space sampling
        - alpha_prior_override: Custom alpha prior (if any)
        - fixed_alpha: Fixed alpha value (if any)
        - fixed_d_offset: Fixed D_offset value (if any)
        - fixed_d0_value: Fixed D0 value (if any)
        - disable_reparam: Whether to disable reparameterization
        - nuts_overrides: NUTS configuration overrides
        - diagnostic_thresholds: Convergence thresholds
    """
    # Import here to avoid circular imports
    try:
        from homodyne.config.parameter_space import PriorDistribution as PD
    except ImportError:
        PD = None

    if parameter_space is None or PD is None:
        return {}

    tier_normalized = (tier or "2").strip()
    if tier_normalized not in {"1", "2", "3", "4"}:
        tier_normalized = "2"

    try:
        d0_bounds = parameter_space.get_bounds("D0")
        d0_prior = parameter_space.get_prior("D0")
    except KeyError:
        return {}

    # Tier 1: Simplified setup - no log-space configuration needed
    if tier_normalized == "1":
        return _build_tier1_settings()

    # Tiers 2-4: Use log-space D0 sampling with varying constraints
    return _build_tier2_4_settings(
        tier_normalized,
        d0_bounds,
        d0_prior,
        PD,
    )


def _build_tier1_settings() -> dict[str, Any]:
    """Build Tier 1 surrogate settings (NLSQ-friendly mode)."""
    nuts_overrides = {
        "target_accept_prob": 0.95,
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
        "drop_d_offset": False,
        "sample_log_d0": False,
        "fixed_d_offset": None,
        "alpha_prior_override": None,
        "fixed_alpha": None,
        "fixed_d0_value": None,
        "disable_reparam": True,
        "nuts_overrides": nuts_overrides,
        "diagnostic_thresholds": diagnostic_thresholds,
    }


def _build_tier2_4_settings(
    tier: str,
    d0_bounds: tuple[float, float],
    d0_prior: "PriorDistribution",
    PD: type,
) -> dict[str, Any]:
    """Build Tier 2-4 surrogate settings with log-space D0 sampling."""
    min_d0 = max(d0_bounds[0], 1e-6)
    max_d0 = max(d0_bounds[1], min_d0 * 10.0)
    if tier == "4":
        constrained_max = max(min_d0 * 50.0, min_d0 * 2.0)
        max_d0 = min(max_d0, constrained_max)

    log_low = float(np.log(min_d0))
    log_high = float(np.log(max_d0))

    prior_mu_clamped = float(np.clip(d0_prior.mu, min_d0, max_d0))
    if tier == "4":
        prior_mu_clamped = min(prior_mu_clamped, max_d0 * 0.6)
    log_loc = float(np.log(max(prior_mu_clamped, 1e-6)))

    # Tier-specific settings
    alpha_prior_override = None
    fixed_alpha = None
    log_scale = 0.5
    sample_log_d0 = True

    if tier == "2":
        alpha_prior_override = PD(
            dist_type="TruncatedNormal",
            mu=-1.0,
            sigma=0.15,
            min_val=-1.5,
            max_val=0.5,
        )
        log_scale = 0.35
    elif tier == "3":
        fixed_alpha = -1.2
        log_scale = 0.3
    elif tier == "4":
        fixed_alpha = -1.2
        log_scale = 0.12
        sample_log_d0 = True

    # NUTS overrides
    nuts_overrides = {
        "target_accept_prob": 0.99 if tier == "2" else 0.995,
        "max_tree_depth": 8 if tier == "2" else 6,
        "n_warmup": 1500 if tier == "2" else 2000,
    }
    if tier == "4":
        nuts_overrides.update({"max_tree_depth": 6, "n_warmup": 2000})

    # Diagnostic thresholds
    diagnostic_thresholds = {
        "focus_params": ["D0", "alpha"],
        "min_ess": 25.0 if tier == "2" else 40.0,
        "max_rhat": 1.2,
    }
    if tier == "4":
        diagnostic_thresholds["min_ess"] = 50.0

    return {
        "tier": tier,
        "drop_d_offset": True,
        "sample_log_d0": sample_log_d0,
        "log_d0_prior": {
            "loc": log_loc,
            "scale": log_scale,
            "low": log_low,
            "high": log_high,
            "trust_radius": None,
        },
        "alpha_prior_override": alpha_prior_override,
        "fixed_alpha": fixed_alpha,
        "fixed_d_offset": 0.0,
        "fixed_d0_value": None,
        "disable_reparam": True,
        "nuts_overrides": nuts_overrides,
        "diagnostic_thresholds": diagnostic_thresholds,
    }


def sample_log_d0(
    prior_cfg: dict[str, float],
    target_dtype: Any,
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
        - 'loc': Mean of log-D0 (e.g., np.log(1000) approx 6.9)
        - 'scale': Standard deviation in log-space (e.g., 0.5)
        - 'low': Lower bound in log-space (e.g., np.log(100) approx 4.6)
        - 'high': Upper bound in log-space (e.g., np.log(10000) approx 9.2)
        - 'trust_radius': Optional additional clipping radius (for tier-4)
    target_dtype : dtype
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
    eps = 1e-6

    loc = jnp.asarray(loc_value, dtype=target_dtype)
    scale = jnp.asarray(scale_value, dtype=target_dtype)
    low = jnp.asarray(low_value + eps, dtype=target_dtype)
    high = jnp.asarray(high_value - eps, dtype=target_dtype)

    # Create truncated Normal distribution in log-space
    log_space_dist = dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

    # Apply ExpTransform to get linear-space D0
    # This creates: D0 = exp(log_D0) where log_D0 ~ TruncatedNormal(...)
    d0_dist = dist.TransformedDistribution(log_space_dist, ExpTransform())

    # Sample D0 directly (already in linear space due to transform)
    # Keep parameter name as 'D0' for API consistency
    d0_value = sample("D0", d0_dist)

    # Emit the log-space latent as a deterministic node for diagnostics
    # This allows post-hoc analysis of the log-space sampling
    log_d0_value = jnp.log(d0_value)
    deterministic("log_D0_latent", log_d0_value)

    return d0_value


def is_single_angle_static(analysis_mode: str, n_phi: int) -> bool:
    """Check if dataset is single-angle static mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode ("static", "static_isotropic", "laminar_flow", etc.)
    n_phi : int
        Number of unique phi angles.

    Returns
    -------
    bool
        True if single-angle static mode.
    """
    return analysis_mode.lower().startswith("static") and n_phi == 1


def should_use_log_sampling(
    single_angle_static_mode: bool,
    param_name: str,
    surrogate_cfg: dict[str, Any],
    reparam_active: bool,
) -> bool:
    """Determine if log-space D0 sampling should be used.

    Parameters
    ----------
    single_angle_static_mode : bool
        Whether in single-angle static mode.
    param_name : str
        Parameter name being sampled.
    surrogate_cfg : dict
        Surrogate configuration.
    reparam_active : bool
        Whether reparameterization is active.

    Returns
    -------
    bool
        True if log-space sampling should be used for this parameter.
    """
    if param_name != "D0":
        return False

    if not single_angle_static_mode:
        return False

    # Check if already handled by surrogate config
    if surrogate_cfg.get("sample_log_d0"):
        return False

    if surrogate_cfg.get("fixed_d0_value") is not None:
        return False

    if reparam_active:
        return False

    # Tier 1 doesn't use log fallback
    tier_allows_log_fallback = surrogate_cfg.get("tier") != "1"
    return tier_allows_log_fallback


def build_log_d0_fallback_prior(
    d0_bounds: tuple[float, float],
    d0_prior: "PriorDistribution",
    target_dtype: Any,
) -> dict[str, float]:
    """Build log-space prior configuration for D0 fallback.

    Parameters
    ----------
    d0_bounds : tuple[float, float]
        (min, max) bounds for D0.
    d0_prior : PriorDistribution
        Prior distribution specification for D0.
    target_dtype : dtype
        Target JAX dtype.

    Returns
    -------
    dict[str, float]
        Log-space prior configuration for sample_log_d0().
    """
    d0_min = max(d0_bounds[0], 1e-6)
    d0_max = max(d0_bounds[1], d0_min * 10.0)
    prior_mu_val = float(d0_prior.mu) if hasattr(d0_prior, "mu") else 1000.0
    prior_mu_clamp = max(d0_min, min(d0_max, prior_mu_val))

    log_loc_val = float(np.log(prior_mu_clamp))
    log_scale_val = 0.5
    log_low = float(np.log(d0_min))
    log_high = float(np.log(d0_max))

    return {
        "loc": log_loc_val,
        "scale": log_scale_val,
        "low": log_low,
        "high": log_high,
    }


def apply_surrogate_parameter_overrides(
    sampled_values: dict[str, jnp.ndarray],
    surrogate_cfg: dict[str, Any],
    target_dtype: Any,
) -> None:
    """Apply surrogate configuration parameter overrides in-place.

    Parameters
    ----------
    sampled_values : dict
        Dictionary of sampled parameter values (modified in-place).
    surrogate_cfg : dict
        Surrogate configuration.
    target_dtype : dtype
        Target JAX dtype.
    """
    # Fixed D0 value
    if surrogate_cfg.get("fixed_d0_value") is not None:
        value = jnp.asarray(surrogate_cfg["fixed_d0_value"], dtype=target_dtype)
        deterministic("D0", value)
        sampled_values["D0"] = value

    # Fixed alpha value
    if surrogate_cfg.get("fixed_alpha") is not None:
        value = jnp.asarray(surrogate_cfg["fixed_alpha"], dtype=target_dtype)
        deterministic("alpha", value)
        sampled_values["alpha"] = value

    # Fixed D_offset value
    if surrogate_cfg.get("fixed_d_offset") is not None:
        value = jnp.asarray(surrogate_cfg["fixed_d_offset"], dtype=target_dtype)
        deterministic("D_offset", value)
        sampled_values["D_offset"] = value


def get_surrogate_diagnostic_thresholds(
    surrogate_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract diagnostic thresholds from surrogate configuration.

    Parameters
    ----------
    surrogate_cfg : dict
        Surrogate configuration.

    Returns
    -------
    dict | None
        Diagnostic thresholds or None if not configured.
    """
    if not surrogate_cfg:
        return None

    diag_cfg = surrogate_cfg.get("diagnostic_thresholds") or {}
    if not diag_cfg:
        return None

    return {
        "active": True,
        "focus_params": diag_cfg.get("focus_params", []),
        "min_ess": float(diag_cfg.get("min_ess", 50.0)),
        "max_rhat": float(diag_cfg.get("max_rhat", 1.1)),
    }


def get_deterministic_param_overrides(
    scaling_override: dict[str, float] | None,
    surrogate_cfg: dict[str, Any] | None,
) -> set[str]:
    """Get set of parameter names that are fixed (not sampled).

    Parameters
    ----------
    scaling_override : dict | None
        Fixed scaling overrides.
    surrogate_cfg : dict | None
        Surrogate configuration.

    Returns
    -------
    set[str]
        Set of parameter names that are deterministic (fixed).
    """
    result = set()

    if scaling_override:
        result.update(scaling_override.keys())

    if surrogate_cfg:
        if (
            surrogate_cfg.get("drop_d_offset")
            or surrogate_cfg.get("fixed_d_offset") is not None
        ):
            result.add("D_offset")
        if surrogate_cfg.get("fixed_alpha") is not None:
            result.add("alpha")
        if surrogate_cfg.get("fixed_d0_value") is not None:
            result.add("D0")

    return result
