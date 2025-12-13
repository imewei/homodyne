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
    P = clip(P, low, high)       # Enforce bounds

Where:
- center = (low + high) / 2  or  prior_mu
- scale = (high - low) / 4   or  prior_sigma

This ensures ALL gradients have similar magnitude, enabling balanced MCMC
exploration.

CRITICAL - Lessons Learned (Dec 2025):
--------------------------------------
DO NOT use log-space transformations (use_log_space=True) for parameters.

While log-space seems intuitive for large-scale parameters like D0, it creates
a Jacobian amplification problem:
- Log-space: ∂D0/∂z = D0 × log_scale ≈ 17000
- Linear: ∂gamma_offset/∂z = 0.02

This ~10^6:1 gradient ratio causes 0% NUTS acceptance rate because proposals
that are reasonable for small-gradient parameters overshoot large-gradient ones.

The fix is to use purely LINEAR z-space scaling for ALL parameters. The scale
differences are handled by NUTS dense mass matrix adaptation during warmup.
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
        If True, sample in log-space for better gradient balancing.
        Used for large-scale parameters like D0, D_offset.
    """

    name: str
    center: float
    scale: float
    low: float
    high: float
    use_log_space: bool = False

    def to_normalized(self, value: float) -> float:
        """Transform from original to normalized space."""
        import numpy as np

        if self.use_log_space:
            # Transform to log-space first, then normalize
            # Ensure value is positive for log
            safe_value = max(value, 1e-10)
            log_value = float(np.log(safe_value))
            return (log_value - self.center) / self.scale
        return (value - self.center) / self.scale

    def to_original(self, z_value: jnp.ndarray) -> jnp.ndarray:
        """Transform from normalized to original space with clipping."""
        if self.use_log_space:
            # Transform from z-space to log-space, then exp to original
            log_value = self.center + self.scale * z_value
            # Clip log_value to avoid overflow (exp(700) is near float max)
            log_clipped = jnp.clip(log_value, -700.0, 700.0)
            raw = jnp.exp(log_clipped)
            return jnp.clip(raw, self.low, self.high)
        raw = self.center + self.scale * z_value
        return jnp.clip(raw, self.low, self.high)


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

    # GRADIENT BALANCING FIX (Dec 2025, revised):
    # Use PURELY LINEAR z-space scaling for all parameters.
    # The original log-space approach for D0/D_offset created Jacobian
    # imbalance: ∂D0/∂z = D0 × log_scale ≈ 17000 vs ∂gamma_offset/∂z = 0.02.
    # This 10^6:1 ratio caused 0% NUTS acceptance rate.
    #
    # Linear z-space ensures consistent Jacobian: ∂param/∂z = scale for all.
    # The scale differences (0.02 to 20000) are handled by NUTS dense mass
    # matrix adaptation, which learns the parameter covariance during warmup.

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

    Returns
    -------
    jnp.ndarray
        Parameter value in original space.
    """
    # Sample in normalized space (unit scale)
    # Use TruncatedNormal to softly encourage values within ~3 sigma of center
    z = numpyro.sample(
        f"{name}_z",
        dist.Normal(0.0, 1.0),
    )

    # Transform to original space with clipping
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
