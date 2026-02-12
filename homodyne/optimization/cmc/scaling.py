"""Parameter Scaling for MCMC Gradient Balancing.

This module implements non-centered reparameterization to balance gradient
scales across parameters with vastly different magnitudes.

The Problem:
------------
In the CMC model, parameters span many orders of magnitude:
- D0: ~10^4 (diffusion coefficient)
- alpha: ~10^0 (exponent)
- gamma_dot_t0: ~10^-3 (shear rate)
- contrast: ~10^-1 (optical scaling)

When NUTS samples these parameters directly, gradients are dominated by
large-scale parameters (D0), causing the sampler to effectively ignore
small-scale parameters. This leads to 0% acceptance rate.

The Solution:
-------------
Non-centered reparameterization transforms each parameter to unit scale:

    P_z ~ Normal(0, 1)           # Sample in normalized space
    P = center + scale × P_z     # Transform to original space
    P = smooth_bound(P, low, high)  # Smoothly enforce bounds

Where:
- center = (low + high) / 2  or  prior_mu
- scale = (high - low) / 4   or  prior_sigma

This ensures ALL gradients have similar magnitude, enabling balanced MCMC
exploration.

CRITICAL - Lessons Learned (Dec 2025):
--------------------------------------
Hard clipping (jnp.clip) introduces non-smooth behavior at the bounds.
In practice this can lead to poor HMC/NUTS adaptation (especially when chains
push against bounds during warmup), including near-zero acceptance.

To avoid this, Homodyne uses a smooth bounded transform based on tanh:

    smooth_bound(x; low, high) = mid + half * tanh((x - mid) / half)

This maps ℝ → (low, high) smoothly while behaving approximately like the
identity mapping in the middle of the interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


@dataclass
class ParameterScaling:
    """Scaling parameters for a single parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    center : float
        Center value for transformation (typically prior mean or bounds midpoint).
    scale : float
        Scale factor for transformation (typically prior std or bounds/4).
    low : float
        Lower bound for clipping.
    high : float
        Upper bound for clipping.
    use_log_space : bool
        Reserved for future use. Homodyne currently uses purely linear
        z-space scaling with smooth bounding for all parameters.
    """

    name: str
    center: float
    scale: float
    low: float
    high: float
    use_log_space: bool = False

    def _smooth_bound(
        self, raw: jnp.ndarray, low: float, high: float, eps: float = 1e-12
    ) -> jnp.ndarray:
        """Smoothly bound a value to (low, high) using tanh.

        Maps ℝ → (low, high) and remains differentiable everywhere.
        """
        mid = 0.5 * (low + high)
        half = 0.5 * (high - low)
        # Avoid division by zero on degenerate bounds.
        half_safe = jnp.where(half > 0.0, half, eps)
        return mid + half_safe * jnp.tanh((raw - mid) / half_safe)

    def _smooth_bound_inverse(
        self, value: float, low: float, high: float, eps: float = 1e-12
    ) -> float:
        """Inverse of _smooth_bound for initialization.

        This is used only to map initial values from original-space to z-space.
        Values at/over the bounds are projected slightly into the interior to
        keep the inverse finite.
        """
        import numpy as np

        mid = 0.5 * (low + high)
        half = 0.5 * (high - low)
        half_safe = half if half > 0.0 else eps
        y = (float(value) - mid) / half_safe
        y = float(np.clip(y, -1.0 + 1e-6, 1.0 - 1e-6))
        return mid + half_safe * float(np.arctanh(y))

    def to_normalized(self, value: float) -> float:
        """Transform from original to normalized space.

        Uses the analytic inverse of the smooth bounding transform to recover
        the underlying affine value prior to normalization.
        """
        # NOTE: use_log_space is intentionally ignored for now.
        raw = self._smooth_bound_inverse(value, self.low, self.high)
        scale = self.scale if self.scale != 0.0 else 1.0
        return float((raw - self.center) / scale)

    def to_original(self, z_value: jnp.ndarray) -> jnp.ndarray:
        """Transform from normalized to original space with smooth bounding."""
        # NOTE: use_log_space is intentionally ignored for now.
        raw = self.center + self.scale * z_value
        return self._smooth_bound(raw, self.low, self.high)


def compute_scaling_factors(
    parameter_space: ParameterSpace,
    n_phi: int,
    analysis_mode: str,
) -> dict[str, ParameterScaling]:
    """Compute scaling factors for all parameters.

    Parameters
    ----------
    parameter_space : ParameterSpace
        Parameter space with bounds and priors.
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").

    Returns
    -------
    dict[str, ParameterScaling]
        Scaling factors for each parameter.
    """
    scalings = {}

    # Per-angle parameters
    for i in range(n_phi):
        for base_name in ["contrast", "offset"]:
            param_name = f"{base_name}_{i}"
            low, high = parameter_space.get_bounds(base_name)

            # Try to get prior, fall back to bounds-based scaling
            try:
                prior = parameter_space.get_prior(base_name)
                center = prior.mu if hasattr(prior, "mu") else (low + high) / 2
                scale = prior.sigma if hasattr(prior, "sigma") else (high - low) / 4
            except KeyError:
                # No prior defined, use bounds midpoint and 1/4 range
                center = (low + high) / 2
                scale = (high - low) / 4

            # Ensure scale is positive and reasonable
            scale = max(scale, (high - low) / 10, 1e-6)

            scalings[param_name] = ParameterScaling(
                name=param_name,
                center=center,
                scale=scale,
                low=low,
                high=high,
            )

    # Physical parameters (always present)
    physical_params = ["D0", "alpha", "D_offset"]
    if analysis_mode == "laminar_flow":
        physical_params.extend(["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"])

    # GRADIENT BALANCING (Dec 2025):
    # Use purely linear z-space scaling for all parameters, then apply a smooth
    # bounding transform (tanh-based) to respect parameter bounds without hard
    # clipping.

    for param_name in physical_params:
        try:
            low, high = parameter_space.get_bounds(param_name)
        except KeyError:
            logger.warning(f"Parameter {param_name} not in parameter_space, skipping")
            continue

        # Always use linear scaling (no log-space)
        try:
            prior = parameter_space.get_prior(param_name)
            center = prior.mu if hasattr(prior, "mu") else (low + high) / 2
            scale = prior.sigma if hasattr(prior, "sigma") else (high - low) / 4
        except KeyError:
            center = (low + high) / 2
            scale = (high - low) / 4

        # Ensure scale is positive and reasonable
        scale = max(scale, (high - low) / 10, 1e-6)

        scalings[param_name] = ParameterScaling(
            name=param_name,
            center=center,
            scale=scale,
            low=low,
            high=high,
            use_log_space=False,  # Always linear
        )

    return scalings


def sample_scaled_parameter(
    name: str,
    scaling: ParameterScaling,
    initial_z: float | None = None,
    prior_scale: float = 1.0,
) -> jnp.ndarray:
    """Sample a parameter in normalized space and transform to original.

    Parameters
    ----------
    name : str
        Parameter name (used for NumPyro site name).
    scaling : ParameterScaling
        Scaling parameters.
    initial_z : float | None
        Initial value in normalized space (for initialization).
    prior_scale : float
        Prior tempering scale factor. For CMC with K shards, set to sqrt(K)
        to implement prior^(1/K) tempering (Scott et al. 2016). The z-space
        prior Normal(0, 1) becomes Normal(0, prior_scale), effectively
        widening the prior so the combined posterior across K shards has
        the correct single-prior contribution.

    Returns
    -------
    jnp.ndarray
        Parameter value in original space.
    """
    # Sample in normalized space
    # prior_scale > 1.0 widens the prior for CMC prior tempering
    z = numpyro.sample(
        f"{name}_z",
        dist.Normal(0.0, prior_scale),
    )

    # Transform to original space with smooth bounds
    value = scaling.to_original(z)

    # Register the transformed value as deterministic for output
    numpyro.deterministic(name, value)

    return value


def log_scaling_factors(scalings: dict[str, ParameterScaling]) -> None:
    """Log scaling factors for debugging.

    Parameters
    ----------
    scalings : dict[str, ParameterScaling]
        Scaling factors.
    """
    logger.info("Parameter scaling factors for gradient balancing:")
    for name, s in scalings.items():
        logger.debug(
            f"  {name}: center={s.center:.4g}, scale={s.scale:.4g}, "
            f"bounds=[{s.low:.4g}, {s.high:.4g}]"
        )


def transform_initial_values_to_z(
    initial_values: dict[str, float] | None,
    scalings: dict[str, ParameterScaling],
) -> dict[str, float]:
    """Transform initial values from original to normalized space.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial values in original space.
    scalings : dict[str, ParameterScaling]
        Scaling factors.

    Returns
    -------
    dict[str, float]
        Initial values in normalized (z) space.
    """
    if initial_values is None:
        return {}

    z_values = {}
    for name, scaling in scalings.items():
        if name in initial_values:
            original_value = initial_values[name]
            z_value = scaling.to_normalized(original_value)
            z_values[f"{name}_z"] = z_value

    return z_values


def transform_samples_from_z(
    samples: dict[str, jnp.ndarray],
    scalings: dict[str, ParameterScaling],
) -> dict[str, jnp.ndarray]:
    """Transform samples from normalized to original space.

    Parameters
    ----------
    samples : dict[str, jnp.ndarray]
        Samples in normalized space (keys ending with "_z").
    scalings : dict[str, ParameterScaling]
        Scaling factors.

    Returns
    -------
    dict[str, jnp.ndarray]
        Samples in original space.
    """
    original_samples = {}

    for name, scaling in scalings.items():
        z_name = f"{name}_z"
        if z_name in samples:
            z_samples = samples[z_name]
            original_samples[name] = scaling.to_original(z_samples)

    return original_samples
