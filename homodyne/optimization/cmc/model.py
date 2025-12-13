"""NumPyro model for XPCS C2 correlation function.

This module defines the probabilistic model for Bayesian inference
of XPCS parameters using NumPyro.

CRITICAL: Parameter sampling order must match:
1. Per-angle contrast: contrast_0, contrast_1, ...
2. Per-angle offset: offset_0, offset_1, ...
3. Physical parameters: D0, alpha, D_offset, [gamma_dot_t0, ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from homodyne.core.physics_cmc import compute_g1_total
from homodyne.optimization.cmc.priors import build_prior
from homodyne.optimization.cmc.scaling import (
    compute_scaling_factors,
    sample_scaled_parameter,
)
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


def xpcs_model(
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
    analysis_mode: str,
    parameter_space: ParameterSpace,
    n_phi: int,
    time_grid: jnp.ndarray | None = None,
    noise_scale: float = 0.1,
) -> None:
    """NumPyro model for XPCS two-time correlation function.

    This model uses the EXACT same physics as NLSQ via compute_g1_total.

    CRITICAL: Parameters are sampled in this EXACT order to ensure
    init_to_value() works correctly:
    1. contrast_0, contrast_1, ..., contrast_{n_phi-1}
    2. offset_0, offset_1, ..., offset_{n_phi-1}
    3. D0, alpha, D_offset (static)
    4. gamma_dot_t0, beta, gamma_dot_t_offset, phi0 (laminar_flow only)

    Parameters
    ----------
    data : jnp.ndarray
        Observed C2 correlation data, shape (n_total,).
    t1 : jnp.ndarray
        Time coordinates t1, shape (n_total,).
    t2 : jnp.ndarray
        Time coordinates t2, shape (n_total,).
    phi_unique : jnp.ndarray
        Unique phi angles, shape (n_phi,).
    phi_indices : jnp.ndarray
        Index into per-angle arrays for each point, shape (n_total,).
    q : float
        Wavevector magnitude.
    L : float
        Stator-rotor gap length (nm).
    dt : float
        Time step.
    analysis_mode : str
        Analysis mode: "static" or "laminar_flow".
    parameter_space : ParameterSpace
        Parameter space with bounds and priors.
    n_phi : int
        Number of unique phi angles.
    noise_scale : float
        Initial estimate of observation noise.
    """
    # =========================================================================
    # 1. Sample per-angle CONTRAST parameters (FIRST)
    # =========================================================================
    contrasts = []
    for i in range(n_phi):
        contrast_prior = build_prior("contrast", parameter_space)
        c_i = numpyro.sample(f"contrast_{i}", contrast_prior)
        contrasts.append(c_i)
    contrast_arr = jnp.array(contrasts)

    # =========================================================================
    # 2. Sample per-angle OFFSET parameters (SECOND)
    # =========================================================================
    offsets = []
    for i in range(n_phi):
        offset_prior = build_prior("offset", parameter_space)
        o_i = numpyro.sample(f"offset_{i}", offset_prior)
        offsets.append(o_i)
    offset_arr = jnp.array(offsets)

    # =========================================================================
    # 3. Sample PHYSICAL parameters (THIRD)
    # =========================================================================
    # Always sample these in canonical order
    D0 = numpyro.sample("D0", build_prior("D0", parameter_space))
    alpha = numpyro.sample("alpha", build_prior("alpha", parameter_space))
    D_offset = numpyro.sample("D_offset", build_prior("D_offset", parameter_space))

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = numpyro.sample(
            "gamma_dot_t0", build_prior("gamma_dot_t0", parameter_space)
        )
        beta = numpyro.sample("beta", build_prior("beta", parameter_space))
        gamma_dot_t_offset = numpyro.sample(
            "gamma_dot_t_offset", build_prior("gamma_dot_t_offset", parameter_space)
        )
        phi0 = numpyro.sample("phi0", build_prior("phi0", parameter_space))

        # Build parameter vector for physics model
        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        # Static mode
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 4. Compute theoretical g1 using EXACT same physics as NLSQ
    # =========================================================================
    # Note: compute_g1_total infers mode from params array length (3=static, 7=laminar)
    # IMPORTANT: phi_unique must be UNIQUE to avoid n_points^2 blowup (see physics_cmc docs)
    g1_all_phi = compute_g1_total(
        params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
    )  # shape: (n_phi, n_points)

    # Map each pooled data point to its phi row to keep a 1D vector aligned with data
    point_idx = jnp.arange(phi_indices.shape[0], dtype=phi_indices.dtype)
    g1_per_point = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 5. Apply per-angle scaling to get C2
    # =========================================================================
    # c2 = contrast * g1^2 + offset
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    c2_theory_raw = contrast_per_point * g1_per_point**2 + offset_per_point

    # =========================================================================
    # 5b. Numerical stability safeguard
    # =========================================================================
    # Replace NaN/Inf with bounded values to prevent likelihood becoming -inf
    # This allows sampling to continue while flagging problematic regions
    c2_theory = jnp.where(
        jnp.isfinite(c2_theory_raw),
        c2_theory_raw,
        jnp.ones_like(c2_theory_raw),  # Replace NaN/Inf with 1.0 (neutral C2 value)
    )

    # Expose numerical health as deterministic for diagnostics
    n_nan = jnp.sum(~jnp.isfinite(c2_theory_raw))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 6. Likelihood with noise model
    # =========================================================================
    # Sample observation noise
    # MCMC-SAFE FIX: Use 3x multiplier on noise_scale to allow larger sigma values.
    # The original noise_scale from data variance is often too tight, causing NUTS
    # to reject all proposals because the sigma prior strongly prefers small values
    # while the data needs larger sigma to account for model-data mismatch.
    # The 3x multiplier allows sigma to explore a wider range during MCMC warmup.
    sigma_scale = noise_scale * 3.0  # MCMC-safe multiplier
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    # Observation likelihood
    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def validate_model_output(
    c2_theory: jnp.ndarray,
    params: jnp.ndarray,
) -> bool:
    """Validate that model output is physically reasonable.

    Parameters
    ----------
    c2_theory : jnp.ndarray
        Theoretical C2 values.
    params : jnp.ndarray
        Parameter values.

    Returns
    -------
    bool
        True if output is valid.
    """
    # Check for NaN/inf
    if jnp.any(jnp.isnan(c2_theory)) or jnp.any(jnp.isinf(c2_theory)):
        return False

    # Check physical range (C2 should be roughly [0, 2] with some tolerance)
    if jnp.any(c2_theory < -1.0) or jnp.any(c2_theory > 10.0):
        return False

    return True


def get_model_param_count(n_phi: int, analysis_mode: str) -> int:
    """Get total number of sampled parameters.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.

    Returns
    -------
    int
        Total number of parameters (including sigma).
    """
    # Per-angle parameters
    n_params = n_phi * 2  # contrast + offset

    # Physical parameters
    if analysis_mode == "laminar_flow":
        n_params += (
            7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
        )
    else:
        n_params += 3  # D0, alpha, D_offset

    # Noise parameter
    n_params += 1  # sigma

    return n_params


def xpcs_model_scaled(
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    phi_indices: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
    analysis_mode: str,
    parameter_space: ParameterSpace,
    n_phi: int,
    time_grid: jnp.ndarray | None = None,
    noise_scale: float = 0.1,
) -> None:
    """NumPyro model with non-centered parameterization for gradient balancing.

    This model samples all parameters in normalized (z) space where z ~ N(0,1),
    then transforms to original space: P = center + scale * z. This ensures
    all gradient magnitudes are balanced, solving the 0% acceptance rate issue
    caused by D0 (~10^4) dominating gradients over gamma_dot_t0 (~10^-3).

    The physics computation is identical to xpcs_model, only the sampling
    space is transformed.

    Parameters
    ----------
    data : jnp.ndarray
        Observed C2 correlation data, shape (n_total,).
    t1, t2 : jnp.ndarray
        Time coordinates, shape (n_total,).
    phi_unique : jnp.ndarray
        Unique phi angles, shape (n_phi,).
    phi_indices : jnp.ndarray
        Index into per-angle arrays for each point, shape (n_total,).
    q : float
        Wavevector magnitude.
    L : float
        Stator-rotor gap length (nm).
    dt : float
        Time step.
    analysis_mode : str
        Analysis mode: "static" or "laminar_flow".
    parameter_space : ParameterSpace
        Parameter space with bounds and priors.
    n_phi : int
        Number of unique phi angles.
    noise_scale : float
        Initial estimate of observation noise.
    """
    # =========================================================================
    # 0. Compute scaling factors for all parameters
    # =========================================================================
    scalings = compute_scaling_factors(parameter_space, n_phi, analysis_mode)

    # =========================================================================
    # 1. Sample per-angle CONTRAST parameters in z-space (FIRST)
    # =========================================================================
    contrasts = []
    for i in range(n_phi):
        c_i = sample_scaled_parameter(f"contrast_{i}", scalings[f"contrast_{i}"])
        contrasts.append(c_i)
    contrast_arr = jnp.array(contrasts)

    # =========================================================================
    # 2. Sample per-angle OFFSET parameters in z-space (SECOND)
    # =========================================================================
    offsets = []
    for i in range(n_phi):
        o_i = sample_scaled_parameter(f"offset_{i}", scalings[f"offset_{i}"])
        offsets.append(o_i)
    offset_arr = jnp.array(offsets)

    # =========================================================================
    # 3. Sample PHYSICAL parameters in z-space (THIRD)
    # =========================================================================
    D0 = sample_scaled_parameter("D0", scalings["D0"])
    alpha = sample_scaled_parameter("alpha", scalings["alpha"])
    D_offset = sample_scaled_parameter("D_offset", scalings["D_offset"])

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = sample_scaled_parameter("gamma_dot_t0", scalings["gamma_dot_t0"])
        beta = sample_scaled_parameter("beta", scalings["beta"])
        gamma_dot_t_offset = sample_scaled_parameter(
            "gamma_dot_t_offset", scalings["gamma_dot_t_offset"]
        )
        phi0 = sample_scaled_parameter("phi0", scalings["phi0"])

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 4. Compute theoretical g1 using EXACT same physics as NLSQ
    # =========================================================================
    g1_all_phi = compute_g1_total(
        params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
    )

    point_idx = jnp.arange(phi_indices.shape[0], dtype=phi_indices.dtype)
    g1_per_point = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 5. Apply per-angle scaling to get C2
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    c2_theory_raw = contrast_per_point * g1_per_point**2 + offset_per_point

    # Numerical stability safeguard
    c2_theory = jnp.where(
        jnp.isfinite(c2_theory_raw),
        c2_theory_raw,
        jnp.ones_like(c2_theory_raw),
    )

    n_nan = jnp.sum(~jnp.isfinite(c2_theory_raw))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 6. Likelihood with noise model
    # =========================================================================
    sigma_scale = noise_scale * 3.0
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)
