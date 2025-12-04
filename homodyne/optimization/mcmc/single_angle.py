"""Single-angle utilities for NumPyro MCMC models.

This module provides utilities for single-angle XPCS datasets (phi_count == 1).

For single-angle datasets:
- Use log-space D0 sampling for better MCMC geometry (D0 spans orders of magnitude)
- Sample ALL 5 parameters: D0, alpha, D_offset, contrast, offset
- No parameters are fixed or dropped

For multi-angle datasets (N angles):
- Sample 3 + 2N parameters: D0, alpha, D_offset + N*(contrast_i, offset_i)

Simplified from v2.4.1 tier system (Dec 2025).
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
    from homodyne.config.parameter_space import PriorDistribution

logger = get_logger(__name__)


def estimate_single_angle_scaling(data: Any) -> tuple[float, float]:
    """Estimate initial contrast/offset from data statistics.

    For single-angle datasets, this provides reasonable starting values
    for contrast and offset parameters based on the data range.

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

    # Estimate contrast from data range, clamp to reasonable values
    contrast = 0.8 * span
    contrast = float(np.clip(contrast, 0.01, 0.9))
    # Estimate offset from data minimum, clamp to reasonable values
    offset = float(np.clip(low, 0.7, 1.3))
    return contrast, offset


def sample_log_d0(
    prior_cfg: dict[str, float],
    target_dtype: Any,
) -> jnp.ndarray:
    """Sample D0 via truncated Normal in log-space with ExpTransform.

    For single-angle datasets, this function samples D0 using a truncated
    Normal distribution in log-space, then automatically transforms to
    linear space via ExpTransform. This provides better MCMC sampling
    geometry compared to linear-space sampling, as diffusion coefficients
    naturally span multiple orders of magnitude.

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
        Analysis mode ("static", "laminar_flow", etc.)
    n_phi : int
        Number of unique phi angles.

    Returns
    -------
    bool
        True if single-angle static mode.
    """
    return analysis_mode.lower().startswith("static") and n_phi == 1


def build_log_d0_prior_config(
    d0_bounds: tuple[float, float],
    d0_prior: "PriorDistribution",
) -> dict[str, float]:
    """Build log-space prior configuration for D0 sampling.

    Parameters
    ----------
    d0_bounds : tuple[float, float]
        (min, max) bounds for D0.
    d0_prior : PriorDistribution
        Prior distribution specification for D0.

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
    log_scale_val = 0.5  # Reasonable default for log-space exploration
    log_low = float(np.log(d0_min))
    log_high = float(np.log(d0_max))

    return {
        "loc": log_loc_val,
        "scale": log_scale_val,
        "low": log_low,
        "high": log_high,
    }
