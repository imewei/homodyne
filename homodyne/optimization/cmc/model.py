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
    c2_theory = contrast_per_point * g1_per_point**2 + offset_per_point

    # =========================================================================
    # 6. Likelihood with noise model
    # =========================================================================
    # Sample observation noise
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=noise_scale))

    # Observation likelihood
    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def xpcs_model_single_chain(
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
    use_log_d0: bool = True,
) -> None:
    """NumPyro model optimized for single-angle sampling.

    Uses log-space sampling for D0 to improve convergence stability.

    Parameters
    ----------
    data : jnp.ndarray
        Observed C2 correlation data.
    t1, t2 : jnp.ndarray
        Time coordinates.
    phi_unique : jnp.ndarray
        Unique phi angles used for physics evaluation.
    phi_indices : jnp.ndarray
        Mapping from pooled data points to phi_unique rows.
    q, L, dt : float
        Physics parameters.
    analysis_mode : str
        Analysis mode.
    parameter_space : ParameterSpace
        Parameter bounds/priors.
    n_phi : int
        Number of phi angles.
    noise_scale : float
        Initial noise estimate.
    use_log_d0 : bool
        If True, sample log(D0) for better convergence.
    """
    # Sample contrast and offset
    contrasts = []
    for i in range(n_phi):
        c_i = numpyro.sample(f"contrast_{i}", build_prior("contrast", parameter_space))
        contrasts.append(c_i)
    contrast_arr = jnp.array(contrasts)

    offsets = []
    for i in range(n_phi):
        o_i = numpyro.sample(f"offset_{i}", build_prior("offset", parameter_space))
        offsets.append(o_i)
    offset_arr = jnp.array(offsets)

    # Sample D0 (optionally in log space for stability)
    if use_log_d0:
        bounds = parameter_space.get_bounds("D0")
        log_D0 = numpyro.sample(
            "log_D0",
            dist.Uniform(jnp.log(bounds[0]), jnp.log(bounds[1])),
        )
        D0 = numpyro.deterministic("D0", jnp.exp(log_D0))
    else:
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
        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # Compute physics
    # Note: compute_g1_total infers mode from params array length (3=static, 7=laminar)
    g1_all_phi = compute_g1_total(
        params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
    )
    point_idx = jnp.arange(phi_indices.shape[0], dtype=phi_indices.dtype)
    g1_per_point = g1_all_phi[phi_indices, point_idx]

    # Apply scaling
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    c2_theory = contrast_per_point * g1_per_point**2 + offset_per_point

    # Likelihood
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=noise_scale))
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
