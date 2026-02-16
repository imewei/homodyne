"""CMC Physics Backend - Element-wise Computations Only
===========================================================

This module provides element-wise physics computations specifically for
Consensus Monte Carlo (CMC) analysis. It contains ONLY element-wise functions
and has NO meshgrid computations to prevent 80GB OOM errors during MCMC.

Key Design Principles:
- Element-wise computations for paired (t1[i], t2[i]) arrays
- No dispatchers, no Python if statements
- Direct JIT-compiled functions only
- Completely independent from NLSQ physics

Architecture Fix (Nov 2025):
- Separated from meshgrid-based NLSQ computations
- Prevents NumPyro NUTS from compiling unused meshgrid branches
- Eliminates 80GB memory allocation during CMC MCMC

Physical Model:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²
g₁_total = g₁_diffusion × g₁_shear

Usage:
  from homodyne.core.physics_cmc import compute_g1_diffusion, compute_g1_total
"""

from functools import partial

import jax.numpy as jnp
from jax import jit

from homodyne.core.physics_utils import (
    PI,
    safe_exp,
    safe_len,
    safe_sinc,
)
from homodyne.core.physics_utils import (
    calculate_diffusion_coefficient as _calculate_diffusion_coefficient_impl_jax,
)
from homodyne.core.physics_utils import (
    calculate_shear_rate_cmc as _calculate_shear_rate_impl_jax,
)
from homodyne.core.physics_utils import (
    trapezoid_cumsum as _trapezoid_cumsum,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@partial(jit, static_argnums=(4,))
def _compute_g1_diffusion_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    time_grid: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
) -> jnp.ndarray:
    """Element-wise diffusion computation for CMC shards using trapezoidal cumulative sums.

    Mirrors the NLSQ cumulative trapezoid logic: build a cumulative integral on the
    1D time grid, then take the absolute difference for each (t1, t2) pair.

    Notes:
        * t1/t2 are physical times (seconds) coming from the loader:
          t = dt * (frame_index - start_frame). They are NOT frame indices.
        * The smooth absolute difference comes from |cumsum[idx2]-cumsum[idx1]|,
          approximating ∫_{t1}^{t2} D(t) dt using all interior trapezoids.
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Build diffusion on the 1D grid and cumulative trapezoid (no dt scaling here)
    D_grid = _calculate_diffusion_coefficient_impl_jax(time_grid, D0, alpha, D_offset)
    D_cumsum = _trapezoid_cumsum(D_grid)

    # Map t1/t2 onto grid indices (time_grid is sorted, uniform)
    max_index = time_grid.shape[0] - 1
    idx1 = jnp.clip(jnp.searchsorted(time_grid, t1, side="left"), 0, max_index)
    idx2 = jnp.clip(jnp.searchsorted(time_grid, t2, side="left"), 0, max_index)

    # Trapezoidal integral steps between indices with smooth abs for stability
    epsilon_abs = 1e-20
    integral_steps = jnp.sqrt((D_cumsum[idx2] - D_cumsum[idx1]) ** 2 + epsilon_abs)

    # Compute g1 using log-space for numerical stability
    # CRITICAL FIX (Dec 2025): Use -700 clip (same as NLSQ) instead of -100
    # The -100 clip caused gradient vanishing for NUTS MCMC:
    # - When log_g1 < -100, gradient = 0 (clipping removes gradient flow)
    # - NUTS uses gradients to guide proposals, so zero gradients = stuck sampling
    # - exp(-700) ≈ 10^-304 is near machine epsilon but still differentiable
    log_g1 = -wavevector_q_squared_half_dt * integral_steps
    log_g1_clipped = jnp.clip(log_g1, -700.0, 0.0)
    g1_diffusion = safe_exp(log_g1_clipped)

    g1_safe = jnp.minimum(g1_diffusion, 1.0)

    return g1_safe  # Shape: (n_points,)


@jit
def _compute_g1_shear_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    sinc_prefactor: float,
    time_grid: jnp.ndarray,
) -> jnp.ndarray:
    """Element-wise shear computation for CMC shards.

    Computes g1_shear for paired (t1[i], t2[i]) points across unique phi angles.
    This is the ONLY shear function for CMC - no meshgrid mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        phi_unique: UNIQUE scattering angles (1D array, pre-filtered for unique values only)
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        time_grid: 1D physical time grid (seconds) used for cumulative trapezoid
            integration; |cumsum[idx2]-cumsum[idx1]| spans all interior intervals.

    Returns:
        Shear contribution to g1 (2D array: (n_unique_phi, n_points))
    """
    # Check params length - if < 7, we're in static mode (no shear)
    if safe_len(params) < 7:
        # Return ones for all unique phi angles and time combinations (g1_shear = 1)
        n_phi_unique = safe_len(phi_unique)
        n_points = safe_len(t1)
        return jnp.ones((n_phi_unique, n_points))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # Build shear rate on the 1D grid and cumulative trapezoid (no dt scaling here)
    gamma_grid = _calculate_shear_rate_impl_jax(
        time_grid,
        gamma_dot_0,
        beta,
        gamma_dot_offset,
    )
    gamma_cumsum = _trapezoid_cumsum(gamma_grid)

    # Map t1/t2 onto grid indices (time_grid is sorted, uniform)
    max_index = time_grid.shape[0] - 1
    idx1 = jnp.clip(jnp.searchsorted(time_grid, t1, side="left"), 0, max_index)
    idx2 = jnp.clip(jnp.searchsorted(time_grid, t2, side="left"), 0, max_index)

    # Trapezoidal integration using cumulative sums with smooth abs for stability
    epsilon_abs = 1e-20
    gamma_integral_elementwise = jnp.sqrt(
        (gamma_cumsum[idx2] - gamma_cumsum[idx1]) ** 2 + epsilon_abs
    )

    # phi_unique is already filtered to unique values by caller (compute_g1_total)
    # No need for jnp.unique() here (causes JAX concretization error during JIT)
    n_phi_unique = safe_len(phi_unique)

    # Compute phase for unique phi angles (vectorized over unique phi)
    angle_diff = jnp.deg2rad(phi0 - phi_unique)  # shape: (n_unique_phi,)
    cos_term = jnp.cos(angle_diff)  # shape: (n_unique_phi,)

    # Broadcast: cos_term (n_unique_phi,) × gamma_integral_elementwise (n_points,) → (n_unique_phi, n_points)
    # Example: 3 unique angles × 100K points = 300K elements (2.4 MB), NOT 300K×100K (80 GB)
    phase = sinc_prefactor * cos_term[:, None] * gamma_integral_elementwise[None, :]

    # Compute sinc² values
    sinc_val = safe_sinc(phase)
    g1_shear: jnp.ndarray = sinc_val**2

    return g1_shear  # Shape: (n_unique_phi, n_points)


@partial(jit, static_argnums=(5, 6))
def _compute_g1_total_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    time_grid: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
) -> jnp.ndarray:
    """Element-wise total g1 computation for CMC shards.

    Computes g1_total = g1_diffusion × g1_shear for paired (t1[i], t2[i]) points.
    This is the ONLY g1_total function for CMC - no meshgrid mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        phi_unique: UNIQUE scattering angles (1D array, pre-filtered)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt

    Returns:
        Total g1 correlation function (2D array: (n_unique_phi, n_points))
    """
    # Compute diffusion contribution: shape (n_points,)
    g1_diff = _compute_g1_diffusion_elementwise(
        params, t1, t2, time_grid, wavevector_q_squared_half_dt
    )

    # Compute shear contribution: shape (n_unique_phi, n_points)
    g1_shear = _compute_g1_shear_elementwise(
        params, t1, t2, phi_unique, sinc_prefactor, time_grid
    )

    # Broadcast g1_diff from (n_points,) to (n_phi, n_points)
    n_phi = g1_shear.shape[0]
    g1_diff_broadcasted = jnp.broadcast_to(
        g1_diff[None, :],
        (n_phi, g1_diff.shape[0]),
    )

    # Multiply: g₁_total[phi, i] = g₁_diffusion[i] × g₁_shear[phi, i]
    g1_total = g1_diff_broadcasted * g1_shear

    # Apply positive-only constraint with minimum threshold for numerical stability
    epsilon = 1e-10
    g1_bounded = jnp.maximum(g1_total, epsilon)

    return g1_bounded


# =============================================================================
# PUBLIC API FUNCTIONS (CMC ONLY)
# =============================================================================


def compute_g1_diffusion(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    q: float,
    dt: float | None = None,
    time_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute g1 diffusion for CMC element-wise paired arrays.

    CMC-specific function - expects 1D paired arrays (t1[i], t2[i]).
    No meshgrid expansion, no dispatchers, direct element-wise computation only.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        q: Scattering wave vector magnitude
        dt: Time step from configuration (REQUIRED for correct physics)
        time_grid: Optional 1D time grid; if None, inferred from t1/t2 and dt.

    Returns:
        Diffusion contribution to g1 (1D array)
    """
    # Validate inputs
    if t1.ndim != 1 or t2.ndim != 1:
        raise ValueError(
            f"CMC physics expects 1D paired arrays, got t1.ndim={t1.ndim}, t2.ndim={t2.ndim}"
        )

    if len(t1) != len(t2):
        raise ValueError(
            f"CMC physics expects paired arrays of same length, got len(t1)={len(t1)}, len(t2)={len(t2)}"
        )

    dt_value: float
    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        time_array = t1
        dt_value = (
            float(time_array[1] - time_array[0]) if safe_len(time_array) > 1 else 1.0
        )
    else:
        dt_value = dt

    # Build/infer time grid (prefer provided grid to avoid JIT shape issues)
    if time_grid is None:
        # Infer maximum time and construct grid from dt
        import numpy as np

        max_time = float(np.max(np.concatenate([np.asarray(t1), np.asarray(t2)])))
        n_time = int(round(max_time / dt_value)) + 1
        time_grid = jnp.linspace(0.0, dt_value * (n_time - 1), n_time)
    else:
        time_grid = jnp.asarray(time_grid)

    # Compute the pre-computed factor using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value

    result: jnp.ndarray = _compute_g1_diffusion_elementwise(
        params, t1, t2, time_grid, wavevector_q_squared_half_dt
    )
    return result


def compute_g1_total(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
    time_grid: jnp.ndarray | None = None,
    _debug: bool = False,
) -> jnp.ndarray:
    """Compute total g1 for CMC element-wise paired arrays.

    CMC-specific function - expects 1D paired arrays (t1[i], t2[i]).
    No meshgrid expansion, no dispatchers, direct element-wise computation only.

    **CRITICAL (Nov 2025): phi Parameter Requirements**

    This function expects phi to contain UNIQUE scattering angles only.
    When calling from MCMC backends, the caller MUST pre-compute unique values::

        # CORRECT: Pre-compute unique phi before calling
        phi_unique = np.unique(np.asarray(phi))
        g1 = compute_g1_total(params, t1, t2, phi_unique, q, L, dt)

        # INCORRECT: Passing replicated phi array causes memory explosion
        # phi_replicated = [0, 0, 0, ..., 90, 90, 90, ...]  # 300K elements
        # g1 = compute_g1_total(..., phi_replicated, ...)  # BAD

    **Why Pre-computation is Required:**

    - CMC pooled data replicates phi for each time point (e.g., 3 angles × 100K points = 300K array)
    - Computing unique values inside this function with ``jnp.unique()`` causes JAX concretization errors
      when called from NumPyro's JIT-traced MCMC model
    - Pre-computing unique values in non-JIT context (e.g., in ``mcmc.py:_run_standard_nuts()``)
      avoids this issue and reduces memory from 80GB to 2.4MB

    **Backward Compatibility:**

    For non-MCMC use cases where phi is already unique or contains few duplicates,
    this function will still work correctly. The internal ``jnp.unique()`` call has been
    removed as of Nov 2025, so callers must ensure phi contains unique values.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        phi: Scattering angles (MUST be unique values, not replicated array)
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)
        time_grid: Optional 1D time grid (preferred). If None, inferred from t1/t2.
        _debug: If True, log detailed debug information (default: False)

    Returns:
        Total g1 correlation function (2D array: (n_phi, n_points))

    Raises:
        ValueError: If t1/t2 are not 1D paired arrays of same length

    Notes:
        - For MCMC use: Pre-compute phi_unique = np.unique(phi) before calling
        - For direct use: Ensure phi contains only unique scattering angles
        - Memory scaling: (n_unique_phi, n_points) NOT (n_replicated_phi, n_points)
    """
    # Validate inputs
    if t1.ndim != 1 or t2.ndim != 1:
        raise ValueError(
            f"CMC physics expects 1D paired arrays, got t1.ndim={t1.ndim}, t2.ndim={t2.ndim}"
        )

    if len(t1) != len(t2):
        raise ValueError(
            f"CMC physics expects paired arrays of same length, got len(t1)={len(t1)}, len(t2)={len(t2)}"
        )

    # Compute the physics factors using configuration dt
    # IMPORTANT: Config dt value will OVERRIDE this default
    # Default dt = 0.001s if not in config (APS-U standard XPCS frame rate: 1ms)
    dt_value = dt if dt is not None else 0.001

    if time_grid is None:
        # Infer maximum time and construct grid from dt_value
        import numpy as np

        dt_safe = float(dt_value)
        max_time = float(np.max(np.concatenate([np.asarray(t1), np.asarray(t2)])))
        n_time = int(round(max_time / dt_safe)) + 1
        time_grid = jnp.linspace(0.0, dt_safe * (n_time - 1), n_time)
        if _debug:
            logger.warning(
                f"[CMC DEBUG] time_grid inferred: n_time={n_time}, max_time={max_time:.4g}, dt={dt_safe:.6g}"
            )
    else:
        time_grid = jnp.asarray(time_grid)

    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
    sinc_prefactor = 0.5 / PI * q * L * dt_value

    # DEBUG logging for CMC physics diagnosis
    if _debug:
        import numpy as np

        t1_np = np.asarray(t1)
        t2_np = np.asarray(t2)
        time_grid_np = np.asarray(time_grid)
        params_np = np.asarray(params)

        logger.warning(
            f"[CMC DEBUG] compute_g1_total called:\n"
            f"  q={q:.6g}, L={L:.6g}, dt={dt_value:.6g}\n"
            f"  wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6g}\n"
            f"  sinc_prefactor={sinc_prefactor:.6g}\n"
            f"  params={params_np}\n"
            f"  t1: shape={t1_np.shape}, range=[{t1_np.min():.4g}, {t1_np.max():.4g}]\n"
            f"  t2: shape={t2_np.shape}, range=[{t2_np.min():.4g}, {t2_np.max():.4g}]\n"
            f"  time_grid: shape={time_grid_np.shape}, range=[{time_grid_np.min():.4g}, {time_grid_np.max():.4g}]\n"
            f"  phi: {np.asarray(phi)}"
        )

    # CRITICAL FIX (Nov 2025): Removed jnp.unique() call to prevent JAX concretization error
    # The caller MUST pre-compute unique phi values before calling this function
    # This is especially critical for MCMC backends where NumPyro JIT-traces this function
    #
    # OLD CODE (REMOVED):
    # phi_unique = jnp.unique(jnp.atleast_1d(phi))  # ❌ Causes JAX concretization error in MCMC
    #
    # NEW CODE:
    # Assume phi is already unique (caller's responsibility to pre-compute)
    phi_unique = jnp.atleast_1d(phi)

    # Log warning if phi appears to have duplicates (only when size is suspicious)
    # This is a soft check - we can't call np.unique() here without breaking JIT
    if hasattr(phi, "shape") and len(phi.shape) == 1:
        # If phi has many elements, it might contain duplicates
        # But we can't check without calling unique(), so just trust the caller
        pass

    result: jnp.ndarray = _compute_g1_total_elementwise(
        params,
        t1,
        t2,
        phi_unique,
        time_grid,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
    )
    return result
