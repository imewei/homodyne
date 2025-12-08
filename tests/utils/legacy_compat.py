"""Legacy compatibility functions for tests.

These functions provide backward-compatible wrappers around the modern API
for use in tests only. They should NOT be used in production code.

Migration from legacy to modern API:
- compute_c2_model_jax -> compute_g2_scaled
- residuals_jax -> (manual computation)
- chi_squared_jax -> compute_chi_squared
- compute_g1_diffusion_jax -> compute_g1_diffusion
"""

from __future__ import annotations

import jax.numpy as jnp

from homodyne.core.jax_backend import (
    EPS,
    compute_g1_diffusion,
    compute_g1_shear,
    compute_g2_scaled,
    safe_len,
)


def compute_c2_model_jax(
    params: dict,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """Legacy wrapper for compute_g2_scaled() - for test compatibility.

    Args:
        params: Parameter dictionary with keys like 'offset', 'contrast',
                'diffusion_coefficient', etc.
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude

    Returns:
        g2 correlation function
    """
    # Extract parameters from dict with defaults
    contrast = params.get("contrast", 0.5)
    offset = params.get("offset", 1.0)
    L = params.get("L", 1.0)

    # Convert parameter dict to array format
    D0 = params.get("diffusion_coefficient", params.get("D0", 1000.0))
    alpha = params.get("alpha", 0.5)
    D_offset = params.get("D_offset", 10.0)

    param_array = jnp.array([D0, alpha, D_offset])

    # Estimate dt from time array
    if t1.ndim == 2:
        time_array = t1[:, 0]
    else:
        time_array = t1
    dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    return compute_g2_scaled(
        params=param_array,
        t1=t1,
        t2=t2,
        phi=phi,
        q=q,
        L=L,
        contrast=contrast,
        offset=offset,
        dt=dt,
    )


def residuals_jax(
    params: dict,
    c2_exp: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """Legacy function: compute residuals (data - model) / sigma."""
    c2_model = compute_c2_model_jax(params, t1, t2, phi, q)
    return (c2_exp - c2_model) / (sigma + EPS)


def chi_squared_jax(
    params: dict,
    c2_exp: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
) -> float:
    """Legacy function: compute chi-squared goodness of fit."""
    residuals = residuals_jax(params, c2_exp, sigma, t1, t2, phi, q)
    return jnp.sum(residuals**2)


def compute_g1_diffusion_jax(
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    q: float,
    D: float,
) -> jnp.ndarray:
    """Legacy function: compute g1 diffusion factor."""
    # Create params array [D0, alpha, D_offset] with alpha=0.5
    params = jnp.array([D, 0.5, 0.0])

    # Estimate dt
    if t1.ndim == 2:
        time_array = t1[:, 0]
    elif t1.ndim == 1:
        time_array = t1
    else:
        time_array = t1.flatten()

    dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    return compute_g1_diffusion(params, t1, t2, q, dt)


def compute_g1_shear_jax(
    phi: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    gamma_dot: float,
    q: float,
    L: float,
) -> jnp.ndarray:
    """Legacy function: compute g1 shear factor.

    Maps legacy API: compute_g1_shear_jax(phi, t1, t2, gamma_dot, q, L)
    to modern API: compute_g1_shear(params, t1, t2, phi, q, L, dt)

    Args:
        phi: Scattering angles
        t1, t2: Time arrays
        gamma_dot: Shear rate (legacy single value)
        q: Scattering wave vector magnitude
        L: Sample-detector distance

    Returns:
        Shear contribution to g1 correlation function
    """
    # Create full laminar flow params array
    # [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
    # Using default diffusion params and mapping gamma_dot to gamma_dot_t0
    params = jnp.array([1000.0, 0.5, 0.0, gamma_dot, 0.0, 0.0, 0.0])

    # Estimate dt
    if t1.ndim == 2:
        time_array = t1[:, 0]
    elif t1.ndim == 1:
        time_array = t1
    else:
        time_array = t1.flatten()

    dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    return compute_g1_shear(params, t1, t2, phi, q, L, dt)
