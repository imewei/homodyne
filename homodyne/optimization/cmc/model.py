"""NumPyro model for XPCS C2 correlation function.

This module defines the probabilistic model for Bayesian inference
of XPCS parameters using NumPyro.

CRITICAL: Parameter sampling order must match:
1. Per-angle contrast: contrast_0, contrast_1, ... (individual mode only)
2. Per-angle offset: offset_0, offset_1, ... (individual mode only)
3. Physical parameters: D0, alpha, D_offset, [gamma_dot_t0, ...]

Per-Angle Modes (v2.18.0+):
- "individual": Independent contrast + offset per angle (2*n_phi + n_physical + 1 params)
- "constant": Fixed per-angle contrast/offset from quantile estimation (n_physical + 1 params)
- "auto": Selects based on n_phi threshold (constant if n_phi >= 3, else individual)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from homodyne.core.physics_cmc import (
    ShardGrid,
    compute_g1_total,
    compute_g1_total_with_precomputed,
)
from homodyne.optimization.cmc.scaling import (
    compute_scaling_factors,
    sample_scaled_parameter,
)
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc.reparameterization import ReparamConfig

logger = get_logger(__name__)


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


def get_model_param_count(
    n_phi: int, analysis_mode: str, per_angle_mode: str = "individual"
) -> int:
    """Get total number of sampled parameters.

    Parameters
    ----------
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    per_angle_mode : str
        Per-angle scaling mode: "individual", "auto", or "constant".

    Returns
    -------
    int
        Total number of parameters (including sigma).

    Notes
    -----
    Mode semantics (same as NLSQ):
    - individual mode: 2*n_phi (contrast + offset) + physical + sigma
    - auto mode: 2 (averaged contrast + offset, SAMPLED) + physical + sigma
    - constant mode: 0 per-angle (FIXED from quantiles) + physical + sigma
    """
    # Per-angle parameters depend on mode
    if per_angle_mode == "constant":
        n_params = 0  # No per-angle params sampled (fixed from quantile estimation)
    elif per_angle_mode == "auto":
        n_params = 2  # Single averaged contrast + offset (SAMPLED)
    else:
        n_params = n_phi * 2  # contrast_0..n + offset_0..n

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
    num_shards: int = 1,
    shard_grid: ShardGrid | None = None,
    **kwargs,
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
    # 0. Compute scaling factors and prior tempering scale
    # =========================================================================
    # P0-1: Use pre-computed scalings from model_kwargs (avoids ~50K Python
    # allocations per NUTS leapfrog step). Fallback for backward compat.
    scalings = kwargs.get("scalings") or compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )
    prior_scale = math.sqrt(num_shards)

    # =========================================================================
    # 1. Sample per-angle CONTRAST parameters in z-space (FIRST)
    # =========================================================================
    contrasts = []
    for i in range(n_phi):
        c_i = sample_scaled_parameter(
            f"contrast_{i}", scalings[f"contrast_{i}"], prior_scale=prior_scale
        )
        contrasts.append(c_i)
    contrast_arr = jnp.array(contrasts)

    # =========================================================================
    # 2. Sample per-angle OFFSET parameters in z-space (SECOND)
    # =========================================================================
    offsets = []
    for i in range(n_phi):
        o_i = sample_scaled_parameter(
            f"offset_{i}", scalings[f"offset_{i}"], prior_scale=prior_scale
        )
        offsets.append(o_i)
    offset_arr = jnp.array(offsets)

    # =========================================================================
    # 3. Sample PHYSICAL parameters in z-space (THIRD, with prior tempering)
    # =========================================================================
    D0 = sample_scaled_parameter("D0", scalings["D0"], prior_scale=prior_scale)
    alpha = sample_scaled_parameter("alpha", scalings["alpha"], prior_scale=prior_scale)
    D_offset = sample_scaled_parameter(
        "D_offset", scalings["D_offset"], prior_scale=prior_scale
    )

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = sample_scaled_parameter(
            "gamma_dot_t0", scalings["gamma_dot_t0"], prior_scale=prior_scale
        )
        beta = sample_scaled_parameter(
            "beta", scalings["beta"], prior_scale=prior_scale
        )
        gamma_dot_t_offset = sample_scaled_parameter(
            "gamma_dot_t_offset",
            scalings["gamma_dot_t_offset"],
            prior_scale=prior_scale,
        )
        phi0 = sample_scaled_parameter(
            "phi0", scalings["phi0"], prior_scale=prior_scale
        )

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 4. Compute theoretical g1 using EXACT same physics as NLSQ
    # =========================================================================
    # P0-2: Use pre-computed wavevector constants from model_kwargs.
    wq2hdt = kwargs.get("wavevector_q_squared_half_dt")
    sp = kwargs.get("sinc_prefactor")
    if shard_grid is not None:
        if wq2hdt is None:
            wq2hdt = jnp.asarray(0.5 * (q**2) * dt)
        if sp is None:
            sp = jnp.asarray(0.5 / math.pi * q * L * dt)
        g1_all_phi = compute_g1_total_with_precomputed(
            params, phi_unique, shard_grid, wq2hdt, sp
        )
    else:
        g1_all_phi = compute_g1_total(
            params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
        )

    # P1-3: Use pre-computed point_idx from model_kwargs.
    point_idx = kwargs.get("point_idx")
    if point_idx is None:
        point_idx = jnp.arange(phi_indices.shape[0], dtype=jnp.int32)
    g1_per_point = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 5. Apply per-angle scaling to get C2
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    # P1-1: Sanitize g1 BEFORE squaring to prevent NaN gradient contamination.
    g1_per_point = jnp.where(jnp.isfinite(g1_per_point), g1_per_point, 1e-10)
    c2_theory = contrast_per_point * g1_per_point**2 + offset_per_point

    n_nan = jnp.sum(~jnp.isfinite(c2_theory))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 6. Likelihood with noise model (tighter sigma prior for precision)
    # =========================================================================
    # Jan 2026: Reduced from 3.0x to 1.5x for tighter precision
    # Prior tempering: widen sigma prior by sqrt(K)
    sigma_scale = noise_scale * 1.5 * prior_scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def xpcs_model_constant(
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
    fixed_contrast: jnp.ndarray | None = None,
    fixed_offset: jnp.ndarray | None = None,
    num_shards: int = 1,
    shard_grid: ShardGrid | None = None,
    **kwargs,
) -> None:
    """NumPyro model with FIXED per-angle scaling (anti-degeneracy constant mode).

    This model uses FIXED per-angle contrast/offset values estimated from
    quantile analysis of the raw data. These values are NOT sampled, reducing
    the parameter space to only physical parameters + sigma.

    This matches NLSQ's anti-degeneracy constant mode and prevents parameter
    absorption degeneracy where per-angle params absorb physical signals.

    Parameter count comparison (laminar_flow, n_phi=23):
    - individual mode: 54 params (46 per-angle + 7 physical + 1 sigma)
    - constant mode: 8 params (7 physical + 1 sigma)

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
    fixed_contrast : jnp.ndarray, optional
        Fixed per-angle contrast values, shape (n_phi,).
        Estimated from quantile analysis. Required for constant mode.
    fixed_offset : jnp.ndarray, optional
        Fixed per-angle offset values, shape (n_phi,).
        Estimated from quantile analysis. Required for constant mode.
    """
    # =========================================================================
    # 0. Validate fixed scaling arrays
    # =========================================================================
    if fixed_contrast is None or fixed_offset is None:
        raise ValueError(
            "xpcs_model_constant requires fixed_contrast and fixed_offset arrays. "
            "These should be estimated from quantile analysis before calling."
        )

    # P0-4: Remap fixed_contrast/fixed_offset from global angle ordering to shard-local.
    # fixed_contrast is indexed 0..global_n_phi-1 (built from full dataset),
    # but phi_indices in each shard are 0..shard_n_phi-1 (recomputed per shard).
    # Without remapping, shard-local index 0 fetches global angle 0's contrast
    # instead of the correct global angle for this shard's first angle.
    global_phi_unique = kwargs.get("global_phi_unique", None)
    if global_phi_unique is not None and phi_unique is not None:
        # Map each shard-local phi_unique entry to its global index
        # Use nearest-neighbor argmin instead of searchsorted to handle float
        # precision differences between shard-local and global phi values
        # (consistent with extract_phi_info which uses argmin for n_phi <= 256).
        global_indices = jnp.argmin(
            jnp.abs(phi_unique[:, None] - global_phi_unique[None, :]), axis=1
        )
        contrast_arr = fixed_contrast[global_indices]  # shape: (shard_n_phi,)
        offset_arr = fixed_offset[global_indices]  # shape: (shard_n_phi,)
    else:
        # Fallback: use directly (single-shard or matching phi ordering)
        contrast_arr = fixed_contrast
        offset_arr = fixed_offset

    # =========================================================================
    # 1. Compute scaling factors and prior tempering scale
    # =========================================================================
    # P0-1: Use pre-computed scalings from model_kwargs.
    scalings = kwargs.get("scalings") or compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )
    prior_scale = math.sqrt(num_shards)

    # =========================================================================
    # 2. Sample PHYSICAL parameters in z-space (with prior tempering)
    # =========================================================================
    D0 = sample_scaled_parameter("D0", scalings["D0"], prior_scale=prior_scale)
    alpha = sample_scaled_parameter("alpha", scalings["alpha"], prior_scale=prior_scale)
    D_offset = sample_scaled_parameter(
        "D_offset", scalings["D_offset"], prior_scale=prior_scale
    )

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = sample_scaled_parameter(
            "gamma_dot_t0", scalings["gamma_dot_t0"], prior_scale=prior_scale
        )
        beta = sample_scaled_parameter(
            "beta", scalings["beta"], prior_scale=prior_scale
        )
        gamma_dot_t_offset = sample_scaled_parameter(
            "gamma_dot_t_offset",
            scalings["gamma_dot_t_offset"],
            prior_scale=prior_scale,
        )
        phi0 = sample_scaled_parameter(
            "phi0", scalings["phi0"], prior_scale=prior_scale
        )

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 3. Compute theoretical g1 using EXACT same physics as NLSQ
    # =========================================================================
    # P0-2: Use pre-computed wavevector constants from model_kwargs.
    wq2hdt = kwargs.get("wavevector_q_squared_half_dt")
    sp = kwargs.get("sinc_prefactor")
    if shard_grid is not None:
        if wq2hdt is None:
            wq2hdt = jnp.asarray(0.5 * (q**2) * dt)
        if sp is None:
            sp = jnp.asarray(0.5 / math.pi * q * L * dt)
        g1_all_phi = compute_g1_total_with_precomputed(
            params, phi_unique, shard_grid, wq2hdt, sp
        )
    else:
        g1_all_phi = compute_g1_total(
            params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
        )

    # P1-3: Use pre-computed point_idx from model_kwargs.
    point_idx = kwargs.get("point_idx")
    if point_idx is None:
        point_idx = jnp.arange(phi_indices.shape[0], dtype=jnp.int32)
    g1_per_point = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 4. Apply FIXED per-angle scaling to get C2
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    # P1-1: Sanitize g1 BEFORE squaring to prevent NaN gradient contamination.
    g1_per_point = jnp.where(jnp.isfinite(g1_per_point), g1_per_point, 1e-10)
    c2_theory = contrast_per_point * g1_per_point**2 + offset_per_point

    n_nan = jnp.sum(~jnp.isfinite(c2_theory))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 5. Likelihood with noise model (tighter sigma prior for precision)
    # =========================================================================
    # Prior tempering: widen sigma prior by sqrt(K)
    sigma_scale = noise_scale * 1.5 * prior_scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def xpcs_model_averaged(
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
    fixed_contrast: jnp.ndarray | None = None,
    fixed_offset: jnp.ndarray | None = None,
    nlsq_prior_config: dict | None = None,
    num_shards: int = 1,
    shard_grid: ShardGrid | None = None,
    **kwargs,
) -> None:
    """NumPyro model with SAMPLED averaged per-angle scaling (auto mode).

    This model samples a SINGLE contrast and SINGLE offset value, then broadcasts
    them to all phi angles. This matches NLSQ's auto/constant mode behavior where
    the averaged scaling parameters are optimized (not fixed).

    Parameter count comparison (laminar_flow, n_phi=23):
    - individual mode: 54 params (46 per-angle + 7 physical + 1 sigma)
    - auto mode (this): 10 params (2 averaged scaling + 7 physical + 1 sigma)
    - constant mode: 8 params (7 physical + 1 sigma, scaling FIXED)

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
    fixed_contrast : jnp.ndarray, optional
        Ignored in this model. Present for API compatibility.
    fixed_offset : jnp.ndarray, optional
        Ignored in this model. Present for API compatibility.
    """
    # =========================================================================
    # 0. Compute scaling factors and prior tempering scale
    # =========================================================================
    # P0-1: Use pre-computed scalings from model_kwargs.
    scalings = kwargs.get("scalings") or compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )

    # Prior tempering (Scott et al. 2016): widen priors by sqrt(K) so that
    # the combined posterior across K shards has exactly one prior contribution.
    # num_shards=1 → prior_scale=1.0 (no tempering, single-shard behavior).
    prior_scale = math.sqrt(num_shards)

    # =========================================================================
    # 1. Sample SINGLE averaged contrast and offset (SAMPLED, not fixed)
    # =========================================================================
    # Use contrast_0 and offset_0 scaling as representative for the averaged values
    contrast = sample_scaled_parameter(
        "contrast", scalings["contrast_0"], prior_scale=prior_scale
    )
    offset = sample_scaled_parameter(
        "offset", scalings["offset_0"], prior_scale=prior_scale
    )

    # Broadcast to all angles
    contrast_arr = jnp.full(n_phi, contrast)
    offset_arr = jnp.full(n_phi, offset)

    # =========================================================================
    # 2. Sample PHYSICAL parameters in z-space
    #    When NLSQ-informed priors are available, use TruncatedNormal centered
    #    on NLSQ estimates instead of the default scaled Normal priors.
    #    Prior tempering is applied via width_factor scaling for NLSQ priors,
    #    or via prior_scale for z-space priors.
    # =========================================================================
    def _sample_param(name: str) -> jnp.ndarray:
        """Sample a parameter, using NLSQ-informed prior if available."""
        if nlsq_prior_config is not None and name in nlsq_prior_config.get(
            "values", {}
        ):
            from homodyne.optimization.cmc.priors import build_nlsq_informed_prior

            # P1-3: Do NOT temper NLSQ-informed priors. The NLSQ estimate is
            # data-driven (not a vague base prior), so tempering by sqrt(K) would
            # make the prior effectively uniform for large shard counts, defeating
            # the warm-start. Only non-NLSQ z-space priors should be tempered.
            base_width = nlsq_prior_config.get("width_factor", 2.0)

            prior = build_nlsq_informed_prior(
                param_name=name,
                nlsq_value=nlsq_prior_config["values"][name],
                nlsq_std=nlsq_prior_config.get("uncertainties", {}).get(name)
                if nlsq_prior_config.get("uncertainties")
                else None,
                bounds=parameter_space.get_bounds(name),
                width_factor=base_width,
            )
            return numpyro.sample(name, prior)
        return sample_scaled_parameter(name, scalings[name], prior_scale=prior_scale)

    D0 = _sample_param("D0")
    alpha = _sample_param("alpha")
    D_offset = _sample_param("D_offset")

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = _sample_param("gamma_dot_t0")
        beta = _sample_param("beta")
        gamma_dot_t_offset = _sample_param("gamma_dot_t_offset")
        phi0 = _sample_param("phi0")

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 3. Compute theoretical C2
    # =========================================================================
    # P0-2: Use pre-computed wavevector constants from model_kwargs.
    wq2hdt = kwargs.get("wavevector_q_squared_half_dt")
    sp = kwargs.get("sinc_prefactor")
    if wq2hdt is None:
        wq2hdt = jnp.asarray(0.5 * (q**2) * dt)
    if sp is None:
        sp = jnp.asarray(0.5 / math.pi * q * L * dt)

    if shard_grid is not None:
        g1_all_phi = compute_g1_total_with_precomputed(
            params, phi_unique, shard_grid, wq2hdt, sp
        )
    else:
        g1_all_phi = compute_g1_total(
            params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
        )

    # P1-3: Use pre-computed point_idx from model_kwargs.
    point_idx = kwargs.get("point_idx")
    if point_idx is None:
        point_idx = jnp.arange(phi_indices.shape[0], dtype=jnp.int32)
    g1 = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 4. Apply per-point contrast and offset
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    # P1-1: Sanitize g1 BEFORE squaring to prevent NaN gradient contamination.
    g1 = jnp.where(jnp.isfinite(g1), g1, 1e-10)
    c2_theory = offset_per_point + contrast_per_point * g1**2

    n_nan = jnp.sum(~jnp.isfinite(c2_theory))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 5. Likelihood with noise model (tighter sigma prior for precision)
    # =========================================================================
    # Jan 2026: Reduced from 3.0x to 1.5x to prevent sigma from absorbing
    # systematic errors and inflating uncertainty estimates
    # Prior tempering: widen sigma prior by sqrt(K)
    sigma_scale = noise_scale * 1.5 * prior_scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def xpcs_model_constant_averaged(
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
    fixed_contrast: jnp.ndarray | None = None,
    fixed_offset: jnp.ndarray | None = None,
    nlsq_prior_config: dict | None = None,
    num_shards: int = 1,
    shard_grid: ShardGrid | None = None,
    **kwargs,
) -> None:
    """NumPyro model with FIXED averaged per-angle scaling (NLSQ parity mode).

    This model uses FIXED contrast/offset values that are the AVERAGE of per-angle
    estimates. These values are NOT sampled, providing exact parity with NLSQ's
    "auto" mode behavior.

    CRITICAL (Jan 2026): This mode fixes the parameter shift issue where CMC's
    "auto" mode (xpcs_model_averaged) samples contrast/offset, introducing extra
    uncertainty that biases physical parameters. By using FIXED averaged values,
    the physical parameter posteriors should match NLSQ estimates.

    Parameter count comparison (laminar_flow):
    - individual mode: 54 params (46 per-angle + 7 physical + 1 sigma)
    - auto mode (xpcs_model_averaged): 10 params (2 sampled scaling + 7 physical + 1 sigma)
    - constant mode (xpcs_model_constant): 8 params (7 physical + 1 sigma, per-angle fixed)
    - constant_averaged mode (this): 8 params (7 physical + 1 sigma, averaged fixed)

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
    fixed_contrast : jnp.ndarray
        Fixed per-angle contrast values, shape (n_phi,). Will be averaged.
    fixed_offset : jnp.ndarray
        Fixed per-angle offset values, shape (n_phi,). Will be averaged.
    """
    # =========================================================================
    # 0. Compute AVERAGED fixed scaling (NLSQ parity)
    # =========================================================================
    if fixed_contrast is None or fixed_offset is None:
        raise ValueError(
            "xpcs_model_constant_averaged requires fixed_contrast and fixed_offset arrays. "
            "These should be estimated from quantile analysis before calling."
        )

    # P0-3: Remap fixed arrays from global to shard-local angles before averaging.
    # Without this, shards with a subset of angles would average over absent angles,
    # biasing the scaling toward irrelevant angle contrasts.
    global_phi_unique = kwargs.get("global_phi_unique", None)
    if global_phi_unique is not None and phi_unique is not None:
        # Use nearest-neighbor argmin instead of searchsorted to handle float
        # precision differences (consistent with xpcs_model_constant).
        global_indices = jnp.argmin(
            jnp.abs(phi_unique[:, None] - global_phi_unique[None, :]), axis=1
        )
        shard_contrast = fixed_contrast[global_indices]
        shard_offset = fixed_offset[global_indices]
    else:
        shard_contrast = fixed_contrast
        shard_offset = fixed_offset

    # Average the shard-local per-angle values and broadcast to all angles
    avg_contrast = jnp.mean(shard_contrast)
    avg_offset = jnp.mean(shard_offset)
    contrast_arr = jnp.full(n_phi, avg_contrast)
    offset_arr = jnp.full(n_phi, avg_offset)

    # Log the averaged values (stored as deterministics for diagnostics)
    numpyro.deterministic("fixed_contrast_mean", avg_contrast)
    numpyro.deterministic("fixed_offset_mean", avg_offset)

    # =========================================================================
    # 1. Compute scaling factors and prior tempering scale
    # =========================================================================
    # P0-1: Use pre-computed scalings from model_kwargs.
    scalings = kwargs.get("scalings") or compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )
    prior_scale = math.sqrt(num_shards)

    # =========================================================================
    # 2. Sample PHYSICAL parameters in z-space (8 params total: 7 physical + sigma)
    #    When NLSQ-informed priors are available, use TruncatedNormal centered
    #    on NLSQ estimates instead of the default scaled Normal priors.
    #    Prior tempering applied via width_factor or prior_scale.
    # =========================================================================
    def _sample_param(name: str) -> jnp.ndarray:
        """Sample a parameter, using NLSQ-informed prior if available."""
        if nlsq_prior_config is not None and name in nlsq_prior_config.get(
            "values", {}
        ):
            from homodyne.optimization.cmc.priors import build_nlsq_informed_prior

            base_width = nlsq_prior_config.get("width_factor", 2.0)

            prior = build_nlsq_informed_prior(
                param_name=name,
                nlsq_value=nlsq_prior_config["values"][name],
                nlsq_std=nlsq_prior_config.get("uncertainties", {}).get(name)
                if nlsq_prior_config.get("uncertainties")
                else None,
                bounds=parameter_space.get_bounds(name),
                width_factor=base_width,
            )
            return numpyro.sample(name, prior)
        return sample_scaled_parameter(name, scalings[name], prior_scale=prior_scale)

    D0 = _sample_param("D0")
    alpha = _sample_param("alpha")
    D_offset = _sample_param("D_offset")

    if analysis_mode == "laminar_flow":
        gamma_dot_t0 = _sample_param("gamma_dot_t0")
        beta = _sample_param("beta")
        gamma_dot_t_offset = _sample_param("gamma_dot_t_offset")
        phi0 = _sample_param("phi0")

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 3. Compute theoretical C2
    # =========================================================================
    # P0-2: Use pre-computed wavevector constants from model_kwargs.
    wq2hdt = kwargs.get("wavevector_q_squared_half_dt")
    sp = kwargs.get("sinc_prefactor")
    if wq2hdt is None:
        wq2hdt = jnp.asarray(0.5 * (q**2) * dt)
    if sp is None:
        sp = jnp.asarray(0.5 / math.pi * q * L * dt)

    if shard_grid is not None:
        g1_all_phi = compute_g1_total_with_precomputed(
            params, phi_unique, shard_grid, wq2hdt, sp
        )
    else:
        g1_all_phi = compute_g1_total(
            params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
        )

    # P1-3: Use pre-computed point_idx from model_kwargs.
    point_idx = kwargs.get("point_idx")
    if point_idx is None:
        point_idx = jnp.arange(phi_indices.shape[0], dtype=jnp.int32)
    g1 = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 4. Apply FIXED averaged scaling to get C2
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    # P1-1: Sanitize g1 BEFORE squaring to prevent NaN gradient contamination.
    g1 = jnp.where(jnp.isfinite(g1), g1, 1e-10)
    c2_theory = offset_per_point + contrast_per_point * g1**2

    n_nan = jnp.sum(~jnp.isfinite(c2_theory))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 5. Likelihood with noise model (tighter sigma prior for precision)
    # =========================================================================
    # Jan 2026: Use tighter sigma prior (1.5x vs 3.0x noise_scale) for better precision
    # Prior tempering: widen sigma prior by sqrt(K)
    sigma_scale = noise_scale * 1.5 * prior_scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def xpcs_model_reparameterized(
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
    fixed_contrast: jnp.ndarray | None = None,
    fixed_offset: jnp.ndarray | None = None,
    reparam_config: ReparamConfig | None = None,
    nlsq_prior_config: dict | None = None,
    num_shards: int = 1,
    t_ref: float = 1.0,
    shard_grid: ShardGrid | None = None,
    **kwargs,
) -> None:
    """NumPyro model with reference-time reparameterized sampling space.

    This model transforms correlated parameters to orthogonal sampling space:
    - D0, alpha → log_D_ref, alpha where D_ref = D0 * t_ref^alpha (decorrelates)
    - D_offset → D_offset_frac = D_offset / (D_ref + D_offset)
    - gamma_dot_t0, beta → log_gamma_ref, beta where gamma_ref = gamma_dot_t0 * t_ref^beta

    The original physics parameters (D0, D_offset, gamma_dot_t0) are computed
    as deterministic transforms and included in the trace for output.

    Parameters
    ----------
    reparam_config : ReparamConfig, optional
        Reparameterization configuration. If None, uses defaults.
    nlsq_prior_config : dict, optional
        NLSQ-informed prior configuration with keys:
        - "values": dict of NLSQ parameter estimates
        - "uncertainties": dict of NLSQ standard errors
        - "width_factor": prior width multiplier
        - "reparam_values": dict of reparameterized NLSQ values (log_D_ref, etc.)
        - "reparam_uncertainties": dict of reparameterized uncertainties
    t_ref : float
        Reference time for reparameterization (default: 1.0).
    [Other parameters same as xpcs_model_averaged]
    """
    from homodyne.optimization.cmc.reparameterization import ReparamConfig

    if reparam_config is None:
        reparam_config = ReparamConfig()

    # Use t_ref from reparam_config if set, otherwise from kwarg
    effective_t_ref = reparam_config.t_ref if reparam_config.t_ref != 1.0 else t_ref

    # =========================================================================
    # 0. Compute scaling factors and prior tempering scale
    # =========================================================================
    # P0-1: Use pre-computed scalings from model_kwargs.
    scalings = kwargs.get("scalings") or compute_scaling_factors(
        parameter_space, n_phi, analysis_mode
    )
    prior_scale = math.sqrt(num_shards)

    # =========================================================================
    # 1. Sample SINGLE averaged contrast and offset (same as auto mode)
    # =========================================================================
    contrast = sample_scaled_parameter(
        "contrast", scalings["contrast_0"], prior_scale=prior_scale
    )
    offset = sample_scaled_parameter(
        "offset", scalings["offset_0"], prior_scale=prior_scale
    )

    contrast_arr = jnp.full(n_phi, contrast)
    offset_arr = jnp.full(n_phi, offset)

    # =========================================================================
    # Helper: sample parameter with NLSQ-informed prior if available
    # =========================================================================
    def _sample_param(name: str) -> jnp.ndarray:
        """Sample a parameter, using NLSQ-informed prior if available."""
        if nlsq_prior_config is not None and name in nlsq_prior_config.get(
            "values", {}
        ):
            from homodyne.optimization.cmc.priors import build_nlsq_informed_prior

            # P1-3: Do NOT temper NLSQ-informed priors (see averaged model).
            base_width = nlsq_prior_config.get("width_factor", 2.0)

            prior = build_nlsq_informed_prior(
                param_name=name,
                nlsq_value=nlsq_prior_config["values"][name],
                nlsq_std=nlsq_prior_config.get("uncertainties", {}).get(name)
                if nlsq_prior_config.get("uncertainties")
                else None,
                bounds=parameter_space.get_bounds(name),
                width_factor=base_width,
            )
            return numpyro.sample(name, prior)
        return sample_scaled_parameter(name, scalings[name], prior_scale=prior_scale)

    # =========================================================================
    # 2. Sample REPARAMETERIZED physical parameters
    # =========================================================================
    # Alpha is sampled directly (not reparameterized, nearly orthogonal to D_ref)
    alpha = _sample_param("alpha")

    # Initialize reparam dicts unconditionally so both enable_d_ref and
    # enable_gamma_ref branches can access them without NameError.
    reparam_vals = (
        nlsq_prior_config.get("reparam_values", {}) if nlsq_prior_config else {}
    )
    reparam_uncs = (
        nlsq_prior_config.get("reparam_uncertainties", {}) if nlsq_prior_config else {}
    )

    if reparam_config.enable_d_ref:
        # --- Reference-time diffusion reparameterization ---
        # Sample log_D_ref where D_ref = D0 * t_ref^alpha (well-constrained by data)

        log_D_ref_loc = reparam_vals.get("log_D_ref", 10.0)  # ~exp(10) ≈ 22K
        log_D_ref_scale = reparam_uncs.get("log_D_ref", 1.0)
        # Prior tempering: widen by sqrt(K), capped at 10 log-units.
        # D0 = exp(log_D_ref) * t_ref^(-alpha); a ±10 deviation in log space
        # spans exp(-10)..exp(10) ≈ 4.5e-5..22026 relative to the reference,
        # covering any physically plausible diffusion coefficient. Without a
        # cap, large K (e.g. 1000) gives scale≈31.6 log-units, spanning 27
        # orders of magnitude and destroying NUTS warmup efficiency.
        log_D_ref_scale_tempered = min(log_D_ref_scale * prior_scale, 10.0)
        log_D_ref = numpyro.sample(
            "log_D_ref",
            dist.Normal(loc=log_D_ref_loc, scale=log_D_ref_scale_tempered),
        )
        D_ref = jnp.exp(log_D_ref)

        # D_offset_frac in [0, 0.5]
        frac_loc = reparam_vals.get("D_offset_frac", 0.05)
        frac_scale = reparam_uncs.get("D_offset_frac", 0.1)
        # P0-2: Cap tempered scale at half the support width [0, 0.5].
        # Unbounded priors (Normal) benefit from sqrt(K) tempering, but for a
        # TruncatedNormal with tight bounds, an over-wide scale makes the prior
        # nearly uniform, losing NLSQ warm-start information.
        frac_scale_tempered = jnp.minimum(frac_scale * prior_scale, 0.24)
        D_offset_frac = numpyro.sample(
            "D_offset_frac",
            dist.TruncatedNormal(
                loc=frac_loc, scale=frac_scale_tempered, low=0.0, high=0.5
            ),
        )

        # Recover physics parameters as deterministics
        D0 = numpyro.deterministic("D0", D_ref * effective_t_ref ** (-alpha))
        _denom = 1 - D_offset_frac
        D_offset = numpyro.deterministic(
            "D_offset",
            D_ref * D_offset_frac / jnp.where(_denom > 1e-10, _denom, 1e-10),
        )
    else:
        # Standard sampling
        D0 = _sample_param("D0")
        D_offset = _sample_param("D_offset")

    if analysis_mode == "laminar_flow":
        # Beta is sampled directly (nearly orthogonal to gamma_ref)
        beta = _sample_param("beta")

        if reparam_config.enable_gamma_ref:
            # --- Reference-time shear reparameterization ---
            # Sample log_gamma_ref where gamma_ref = gamma_dot_t0 * t_ref^beta
            log_gamma_ref_loc = reparam_vals.get("log_gamma_ref", -5.0)
            log_gamma_ref_scale = reparam_uncs.get("log_gamma_ref", 1.0)
            # Cap at 10 log-units (same rationale as log_D_ref_scale_tempered above).
            log_gamma_ref_scale_tempered = min(log_gamma_ref_scale * prior_scale, 10.0)
            log_gamma_ref = numpyro.sample(
                "log_gamma_ref",
                dist.Normal(
                    loc=log_gamma_ref_loc, scale=log_gamma_ref_scale_tempered
                ),
            )
            # Recover gamma_dot_t0 = exp(log_gamma_ref) * t_ref^(-beta)
            gamma_dot_t0 = numpyro.deterministic(
                "gamma_dot_t0", jnp.exp(log_gamma_ref) * effective_t_ref ** (-beta)
            )
        else:
            gamma_dot_t0 = _sample_param("gamma_dot_t0")

        gamma_dot_t_offset = _sample_param("gamma_dot_t_offset")
        phi0 = _sample_param("phi0")

        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        )
    else:
        params = jnp.array([D0, alpha, D_offset])

    # =========================================================================
    # 3. Compute theoretical C2
    # =========================================================================
    # P0-2: Use pre-computed wavevector constants from model_kwargs.
    wq2hdt = kwargs.get("wavevector_q_squared_half_dt")
    sp = kwargs.get("sinc_prefactor")
    if wq2hdt is None:
        wq2hdt = jnp.asarray(0.5 * (q**2) * dt)
    if sp is None:
        sp = jnp.asarray(0.5 / math.pi * q * L * dt)

    if shard_grid is not None:
        g1_all_phi = compute_g1_total_with_precomputed(
            params, phi_unique, shard_grid, wq2hdt, sp
        )
    else:
        g1_all_phi = compute_g1_total(
            params, t1, t2, phi_unique, q, L, dt, time_grid=time_grid
        )

    # P1-3: Use pre-computed point_idx from model_kwargs.
    point_idx = kwargs.get("point_idx")
    if point_idx is None:
        point_idx = jnp.arange(phi_indices.shape[0], dtype=jnp.int32)
    g1 = g1_all_phi[phi_indices, point_idx]

    # =========================================================================
    # 4. Apply per-point contrast and offset
    # =========================================================================
    contrast_per_point = contrast_arr[phi_indices]
    offset_per_point = offset_arr[phi_indices]
    # P1-1: Sanitize g1 BEFORE squaring to prevent NaN gradient contamination.
    g1 = jnp.where(jnp.isfinite(g1), g1, 1e-10)
    c2_theory = offset_per_point + contrast_per_point * g1**2

    n_nan = jnp.sum(~jnp.isfinite(c2_theory))
    numpyro.deterministic("n_numerical_issues", n_nan)

    # =========================================================================
    # 5. Likelihood with noise model (with prior tempering)
    # =========================================================================
    sigma_scale = noise_scale * 1.5 * prior_scale
    sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))

    numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)


def get_xpcs_model(
    per_angle_mode: str = "individual", use_reparameterization: bool = False
):
    """Get the appropriate NumPyro model function for the given per-angle mode.

    Parameters
    ----------
    per_angle_mode : str
        Per-angle scaling mode: "individual", "auto", "constant", or "constant_averaged".
    use_reparameterization : bool
        If True and per_angle_mode is "auto", use reparameterized model for
        better sampling of correlated parameters (D_total instead of D0/D_offset,
        log_gamma_dot_t0 instead of gamma_dot_t0).

    Returns
    -------
    callable
        NumPyro model function.

    Notes
    -----
    Mode semantics (same as NLSQ):

    - individual: Uses xpcs_model_scaled which samples per-angle contrast/offset
      (n_phi*2 + 7 physical + 1 sigma params for laminar_flow).
    - auto: Uses xpcs_model_averaged which samples SINGLE averaged contrast/offset
      (2 averaged + 7 physical + 1 sigma = 10 params for laminar_flow).
      If use_reparameterization=True, uses xpcs_model_reparameterized instead.
    - constant: Uses xpcs_model_constant which requires fixed_contrast/fixed_offset
      arrays (NOT sampled, 7 physical + 1 sigma = 8 params for laminar_flow).
    - constant_averaged: Uses xpcs_model_constant_averaged with FIXED averaged scaling
      (NOT sampled, 7 physical + 1 sigma = 8 params). Provides exact NLSQ parity.
    """
    if per_angle_mode == "auto":
        if use_reparameterization:
            logger.info(
                "CMC: Using reparameterized auto mode model "
                "(log_D_ref + log_gamma_ref sampling)"
            )
            return xpcs_model_reparameterized
        else:
            logger.info(
                "CMC: Using auto mode model (sampled averaged scaling, 10 params)"
            )
            return xpcs_model_averaged
    elif per_angle_mode == "constant":
        logger.info(
            "CMC: Using constant mode model (fixed per-angle scaling, 8 params)"
        )
        return xpcs_model_constant
    elif per_angle_mode == "constant_averaged":
        logger.info(
            "CMC: Using constant_averaged mode model (fixed averaged scaling, 8 params, NLSQ parity)"
        )
        return xpcs_model_constant_averaged
    elif per_angle_mode == "individual":
        logger.info("CMC: Using individual mode model (sampled per-angle scaling)")
        return xpcs_model_scaled
    else:
        # T3-2: Reject unsupported modes (e.g., "fourier") instead of silently
        # falling through to individual mode with wrong parameterization.
        raise ValueError(
            f"Unsupported CMC per_angle_mode: '{per_angle_mode}'. "
            f"Valid modes: 'auto', 'constant', 'constant_averaged', 'individual'. "
            f"NLSQ 'fourier' mode has no CMC equivalent."
        )
