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

Performance Optimization (Feb 2026) — D2: Cumsum split + searchsorted dedup:
- Pre-compute shard-constant quantities (time_safe, idx1, idx2) once per shard
  instead of recomputing them on every NUTS leapfrog step.
- `time_safe`: floor applied to time_grid to avoid t^alpha singularity at t=0.
  Fixed per shard since time_grid is shard-constant.
- `idx1, idx2`: searchsorted results mapping t1/t2 → time_grid indices.
  Fixed per shard since time_grid, t1, t2 are all shard-constant.
  Previously computed independently in diffusion AND shear functions (2× work).
  Now computed once in `precompute_shard_grid` and threaded down to both.
- Expected speedup: 2-5× CMC wall time for laminar_flow mode.

Public API additions:
  precompute_shard_grid(time_grid, t1, t2, dt) → ShardGrid
  compute_g1_total_with_precomputed(params, phi_unique, shard_grid, wq_dt, sinc_pre)

Physical Model:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²
g₁_total = g₁_diffusion × g₁_shear

Usage:
  from homodyne.core.physics_cmc import compute_g1_diffusion, compute_g1_total
  from homodyne.core.physics_cmc import precompute_shard_grid, compute_g1_total_with_precomputed
"""

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import jit

from homodyne.core.physics_utils import (
    calculate_diffusion_coefficient as _calc_diff,
)
from homodyne.core.physics_utils import (
    calculate_shear_rate_cmc as _calc_shear,
)
from homodyne.core.physics_utils import (
    safe_sinc,
)
from homodyne.core.physics_utils import (
    trapezoid_cumsum as _trapezoid_cumsum,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# SHARD-CONSTANT PRE-COMPUTED GRID
# =============================================================================


class ShardGrid(NamedTuple):
    """Pre-computed shard-constant arrays.

    These quantities depend only on (time_grid, t1, t2, dt) — all of which
    are fixed for the lifetime of a shard. Pre-computing them outside the
    NUTS hot path avoids redundant work on every leapfrog step.

    Attributes
    ----------
    time_safe : jnp.ndarray, shape (G,)
        ``jnp.where(time_grid > epsilon, time_grid, epsilon)`` where
        epsilon = max(dt/2, 1e-8). Uses gradient-safe ``jnp.where`` floor
        per CLAUDE.md Rule 7. Avoids the t=0 singularity for t^alpha /
        t^beta power laws. Fixed per shard.
    idx1 : jnp.ndarray, shape (N,), dtype int
        ``searchsorted(time_grid, t1)`` clipped to [0, G-1].
        Maps each t1 observation to its nearest time_grid index.
    idx2 : jnp.ndarray, shape (N,), dtype int
        ``searchsorted(time_grid, t2)`` clipped to [0, G-1].
        Maps each t2 observation to its nearest time_grid index.
    dt_safe : float
        Time step used to build time_safe epsilon. Stored for reference.
    """

    time_safe: jnp.ndarray
    idx1: jnp.ndarray
    idx2: jnp.ndarray
    dt_safe: float


def precompute_shard_grid(
    time_grid: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    dt: float | None,
) -> ShardGrid:
    """Pre-compute shard-constant quantities for the CMC physics hot path.

    Call this ONCE per shard (outside the NUTS sampling loop) and pass the
    returned ``ShardGrid`` to ``compute_g1_total_with_precomputed``.

    Parameters
    ----------
    time_grid : jnp.ndarray, shape (G,)
        1D time grid used for trapezoidal cumulative integration.
    t1 : jnp.ndarray, shape (N,)
        Observed t1 time coordinates for this shard.
    t2 : jnp.ndarray, shape (N,)
        Observed t2 time coordinates for this shard.
    dt : float
        Time step (seconds). Used to set the t=0 singularity floor:
        ``epsilon = max(dt / 2, 1e-8)``.

    Returns
    -------
    ShardGrid
        Pre-computed (time_safe, idx1, idx2) ready for NUTS.
    """
    if dt is None:
        logger.warning(
            "CMC precompute_shard_grid: dt not provided; falling back to 0.001 s (1 kHz). "
            "Pass dt explicitly for correct physics factors."
        )
    dt_safe = float(dt) if dt is not None else 1e-3

    if dt_safe <= 0:
        raise ValueError(
            f"precompute_shard_grid: dt must be positive, got {dt_safe}. "
            "Check shard_config['dt'] — frame interval cannot be zero or negative."
        )

    # Build time_safe: apply singularity floor once
    # This is identical to what calculate_diffusion_coefficient and
    # calculate_shear_rate_cmc do internally on every leapfrog step.
    # P2-R6-05: Use jnp.where (not jnp.maximum) for consistency with project
    # gradient-safe floor convention (CLAUDE.md critical rule #7). Although
    # time_safe here is shard-constant and not differentiated through, the
    # jnp.where form is used throughout physics_utils.py and prevents
    # accidental regression if this path is ever made differentiable.
    epsilon = max(dt_safe * 0.5, 1e-8)
    time_safe = jnp.where(time_grid > epsilon, time_grid, epsilon)

    # Build searchsorted indices: shard-constant because time_grid, t1, t2 are fixed
    max_index = time_grid.shape[0] - 1
    idx1 = jnp.clip(jnp.searchsorted(time_grid, t1, side="left"), 0, max_index)
    idx2 = jnp.clip(jnp.searchsorted(time_grid, t2, side="left"), 0, max_index)

    return ShardGrid(time_safe=time_safe, idx1=idx1, idx2=idx2, dt_safe=dt_safe)


# =============================================================================
# OPTIMIZED INNER KERNELS (accept pre-computed idx + time_safe)
# =============================================================================


@jit
def _compute_g1_diffusion_from_idx(
    params: jnp.ndarray,
    idx1: jnp.ndarray,
    idx2: jnp.ndarray,
    time_safe: jnp.ndarray,
    wavevector_q_squared_half_dt: jnp.ndarray,
) -> jnp.ndarray:
    """Diffusion g1 using pre-computed indices and time_safe.

    Replaces ``_compute_g1_diffusion_elementwise`` in the NUTS hot path.
    Skips ``searchsorted`` and the ``jnp.where`` singularity floor —
    both are shard-constant and were pre-computed in ``precompute_shard_grid``.

    Parameters
    ----------
    params : (3,) or (7,)
        Physical parameters: [D0, alpha, D_offset, ...]
    idx1, idx2 : (N,) int
        Pre-computed ``searchsorted`` indices for t1/t2 → time_grid.
    time_safe : (G,)
        ``jnp.where(time_grid > epsilon, time_grid, epsilon)`` — pre-computed per shard.
    wavevector_q_squared_half_dt : scalar
        Pre-computed factor 0.5 * q² * dt.

    Returns
    -------
    jnp.ndarray, shape (N,)
        Diffusion contribution to g1.
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Compute D(t) on the safe time grid (param-dependent: varies each leapfrog step)
    # Use jnp.where instead of jnp.maximum to preserve gradients when D(t) → 0.
    # jnp.maximum kills gradients (d/dx = 0) below the threshold, causing NUTS
    # divergences from zero momentum updates during leapfrog integration.
    D_raw = D0 * (time_safe**alpha) + D_offset
    D_grid = jnp.where(D_raw > 1e-10, D_raw, 1e-10)
    D_cumsum = _trapezoid_cumsum(D_grid)

    # Index into cumsum using pre-computed indices (no searchsorted here)
    epsilon_abs = 1e-12
    integral_steps = jnp.sqrt((D_cumsum[idx2] - D_cumsum[idx1]) ** 2 + epsilon_abs)

    log_g1 = -wavevector_q_squared_half_dt * integral_steps
    log_g1_clipped = jnp.clip(log_g1, -700.0, 0.0)
    # log_g1_clipped is already in [-700, 0] — jnp.exp is safe (no overflow risk).
    return jnp.exp(log_g1_clipped)  # shape: (N,)


@jit
def _compute_g1_shear_from_idx(
    params: jnp.ndarray,
    idx1: jnp.ndarray,
    idx2: jnp.ndarray,
    time_safe: jnp.ndarray,
    phi_unique: jnp.ndarray,
    sinc_prefactor: jnp.ndarray,
) -> jnp.ndarray:
    """Shear g1 using pre-computed indices and time_safe.

    Replaces ``_compute_g1_shear_elementwise`` in the NUTS hot path.
    Skips ``searchsorted`` and the singularity floor — both pre-computed.

    Parameters
    ----------
    params : (7,)
        Physical parameters: [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
    idx1, idx2 : (N,) int
        Pre-computed ``searchsorted`` indices.
    time_safe : (G,)
        Pre-computed safe time grid.
    phi_unique : (P,)
        Unique scattering angles (degrees). Must be pre-deduplicated by caller.
    sinc_prefactor : scalar
        Pre-computed factor 0.5/π * q * L * dt.

    Returns
    -------
    jnp.ndarray, shape (P, N)
        Shear contribution to g1.
    """
    # Static mode (no shear): return ones
    if params.shape[0] < 7:
        n_phi_unique = phi_unique.shape[0]
        n_points = idx1.shape[0]
        return jnp.ones((n_phi_unique, n_points))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # Compute gamma(t) on the safe time grid.
    # P2-C: time_safe already has all t=0 elements floored by precompute_shard_grid
    # (epsilon >= 1e-8 > 0), so no additional guard is needed here.
    # Use jnp.where instead of jnp.maximum to preserve gradients when γ̇(t) → 0.
    gamma_raw = gamma_dot_0 * (time_safe**beta) + gamma_dot_offset
    gamma_grid = jnp.where(gamma_raw > 1e-10, gamma_raw, 1e-10)
    gamma_cumsum = _trapezoid_cumsum(gamma_grid)

    # Index into cumsum using pre-computed indices (no searchsorted here)
    epsilon_abs = 1e-12
    gamma_integral = jnp.sqrt(
        (gamma_cumsum[idx2] - gamma_cumsum[idx1]) ** 2 + epsilon_abs
    )

    # Vectorised over unique phi angles
    angle_diff = jnp.deg2rad(phi0 - phi_unique)  # (P,)
    cos_term = jnp.cos(angle_diff)  # (P,)
    phase = sinc_prefactor * cos_term[:, None] * gamma_integral[None, :]  # (P, N)

    sinc_val = safe_sinc(phase)
    result: jnp.ndarray = sinc_val**2  # (P, N)
    return result


@jit
def _compute_g1_total_with_precomputed(
    params: jnp.ndarray,
    phi_unique: jnp.ndarray,
    time_safe: jnp.ndarray,
    idx1: jnp.ndarray,
    idx2: jnp.ndarray,
    wavevector_q_squared_half_dt: jnp.ndarray,
    sinc_prefactor: jnp.ndarray,
) -> jnp.ndarray:
    """Total g1 using pre-computed shard-constant quantities.

    This is the NUTS hot-path kernel.  All quantities that depend only on the
    shard (time_safe, idx1, idx2) are pre-computed once by
    ``precompute_shard_grid`` and passed in, eliminating O(G) work that was
    previously repeated on every leapfrog step.

    The only O(G) work that remains per step is ``jnp.cumsum`` over the
    param-dependent ``D_grid`` / ``gamma_grid`` — unavoidable since D0/alpha
    and gamma_dot_0/beta are sampled parameters.

    Parameters
    ----------
    params : (3,) or (7,)
    phi_unique : (P,)
    time_safe : (G,)   — pre-computed, shard-constant
    idx1, idx2 : (N,)  — pre-computed, shard-constant
    wavevector_q_squared_half_dt : scalar
    sinc_prefactor : scalar

    Returns
    -------
    jnp.ndarray, shape (P, N)
    """
    # Diffusion: shape (N,)
    g1_diff = _compute_g1_diffusion_from_idx(
        params, idx1, idx2, time_safe, wavevector_q_squared_half_dt
    )

    # Shear: shape (P, N)
    g1_shear = _compute_g1_shear_from_idx(
        params, idx1, idx2, time_safe, phi_unique, sinc_prefactor
    )

    # Broadcast and multiply: (P, N)
    n_phi = g1_shear.shape[0]
    g1_diff_broadcasted = jnp.broadcast_to(g1_diff[None, :], (n_phi, g1_diff.shape[0]))
    g1_total = g1_diff_broadcasted * g1_shear
    # P1-A: Gradient-safe lower floor — consistent with physics_nlsq.py and jax_backend.py.
    # g1_shear (sinc²) is exactly zero at Phi=n*pi; without this floor, d(g2)/dparams=0
    # at those points, stalling NUTS leapfrog.
    epsilon = 1e-10
    return jnp.where(g1_total > epsilon, g1_total, epsilon)  # type: ignore[no-any-return]


# =============================================================================
# LEGACY INNER KERNELS (kept for backward compat with non-precomputed callers)
# =============================================================================


@jit
def _compute_g1_diffusion_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    time_grid: jnp.ndarray,
    wavevector_q_squared_half_dt: jnp.ndarray,
) -> jnp.ndarray:
    """Element-wise diffusion computation for CMC shards using trapezoidal cumulative sums.

    Mirrors the NLSQ cumulative trapezoid logic: build a cumulative integral on the
    1D time grid, then take the absolute difference for each (t1, t2) pair.

    Notes:
        * t1/t2 are physical times (seconds) coming from the loader:
          t = dt * (frame_index - start_frame). They are NOT frame indices.
        * The smooth absolute difference comes from |cumsum[idx2]-cumsum[idx1]|,
          approximating ∫_{t1}^{t2} D(t) dt using all interior trapezoids.

    .. deprecated::
        Prefer ``_compute_g1_diffusion_from_idx`` + ``precompute_shard_grid`` in
        NUTS hot paths to avoid repeating ``searchsorted`` every leapfrog step.
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Build diffusion on the 1D grid and cumulative trapezoid (no dt scaling here)
    D_grid = _calc_diff(time_grid, D0, alpha, D_offset)
    D_cumsum = _trapezoid_cumsum(D_grid)

    # Map t1/t2 onto grid indices (time_grid is sorted, uniform)
    max_index = time_grid.shape[0] - 1
    idx1 = jnp.clip(jnp.searchsorted(time_grid, t1, side="left"), 0, max_index)
    idx2 = jnp.clip(jnp.searchsorted(time_grid, t2, side="left"), 0, max_index)

    # Trapezoidal integral steps between indices with smooth abs for stability.
    # epsilon_abs must be large enough for float32 precision (>= 1e-12).
    # Previous value 1e-20 was below float32 machine epsilon, making the
    # smooth-abs non-functional in float32 (x² + 1e-20 == x² due to underflow).
    epsilon_abs = 1e-12
    integral_steps = jnp.sqrt((D_cumsum[idx2] - D_cumsum[idx1]) ** 2 + epsilon_abs)

    # Compute g1 using log-space for numerical stability
    # CRITICAL FIX (Dec 2025): Use -700 clip (same as NLSQ) instead of -100
    # The -100 clip caused gradient vanishing for NUTS MCMC:
    # - When log_g1 < -100, gradient = 0 (clipping removes gradient flow)
    # - NUTS uses gradients to guide proposals, so zero gradients = stuck sampling
    # - exp(-700) ≈ 10^-304 is near machine epsilon but still differentiable
    log_g1 = -wavevector_q_squared_half_dt * integral_steps
    log_g1_clipped = jnp.clip(log_g1, -700.0, 0.0)
    # log_g1_clipped is already in [-700, 0] — jnp.exp is safe (no overflow risk).
    g1_diffusion = jnp.exp(log_g1_clipped)

    # P1-2: Removed jnp.minimum(g1_diffusion, 1.0) — the log-space clip above
    # (jnp.clip(log_g1, -700, 0)) already guarantees g1 = exp(log_g1) ≤ 1.0.
    # The hard min killed gradients at g1=1.0 (diagonal elements), harming NUTS.
    result: jnp.ndarray = g1_diffusion  # Shape: (n_points,)
    return result


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

    .. deprecated::
        Prefer ``_compute_g1_shear_from_idx`` + ``precompute_shard_grid`` in
        NUTS hot paths to avoid repeating ``searchsorted`` every leapfrog step.
    """
    # Check params length - if < 7, we're in static mode (no shear)
    if params.shape[0] < 7:
        # Return ones for all unique phi angles and time combinations (g1_shear = 1)
        n_phi_unique = phi_unique.shape[0]
        n_points = t1.shape[0]
        return jnp.ones((n_phi_unique, n_points))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # Build shear rate on the 1D grid and cumulative trapezoid (no dt scaling here)
    gamma_grid = _calc_shear(
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

    # Trapezoidal integration using cumulative sums with smooth abs for stability.
    # epsilon_abs=1e-12: float32/float64-safe (see P0-2 fix in diffusion path).
    epsilon_abs = 1e-12
    gamma_integral_elementwise = jnp.sqrt(
        (gamma_cumsum[idx2] - gamma_cumsum[idx1]) ** 2 + epsilon_abs
    )

    # phi_unique is already filtered to unique values by caller (compute_g1_total).
    # No jnp.unique() here — causes JAX concretization error during JIT.

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


@jit
def _compute_g1_total_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    time_grid: jnp.ndarray,
    wavevector_q_squared_half_dt: jnp.ndarray,
    sinc_prefactor: jnp.ndarray,
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
    # P1-B: Gradient-safe lower floor — consistent with all other g1_total implementations.
    epsilon = 1e-10
    return jnp.where(g1_total > epsilon, g1_total, epsilon)  # type: ignore[no-any-return]


# =============================================================================
# PUBLIC API FUNCTIONS (CMC ONLY)
# =============================================================================


def compute_g1_total_with_precomputed(
    params: jnp.ndarray,
    phi_unique: jnp.ndarray,
    shard_grid: ShardGrid,
    wavevector_q_squared_half_dt: jnp.ndarray,
    sinc_prefactor: jnp.ndarray,
) -> jnp.ndarray:
    """Compute total g1 using pre-computed shard-constant quantities.

    **Preferred hot-path API for NUTS sampling.**  Call ``precompute_shard_grid``
    once per shard (outside the model function) and pass the result here.
    Avoids repeating ``searchsorted`` and the singularity floor on every
    leapfrog step.

    Parameters
    ----------
    params : jnp.ndarray, shape (3,) or (7,)
        Physical parameters [D0, alpha, D_offset, ...].
    phi_unique : jnp.ndarray, shape (P,)
        Unique scattering angles (degrees). Must be pre-deduplicated.
    shard_grid : ShardGrid
        Pre-computed grid from ``precompute_shard_grid``.
    wavevector_q_squared_half_dt : scalar
        Pre-computed factor 0.5 * q² * dt.
    sinc_prefactor : scalar
        Pre-computed factor 0.5/π * q * L * dt.

    Returns
    -------
    jnp.ndarray, shape (P, N)
        Total g1 correlation function.
    """
    result: jnp.ndarray = _compute_g1_total_with_precomputed(
        params,
        jnp.atleast_1d(phi_unique),
        shard_grid.time_safe,
        shard_grid.idx1,
        shard_grid.idx2,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
    )
    return result


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
        # FALLBACK: Do NOT estimate dt from t1[1]-t1[0] — CMC paired data is
        # not monotonic (shards are shuffled), so the difference between
        # consecutive elements is meaningless. Fall back to 1e-3 s (1 kHz).
        logger.warning(
            "CMC compute_g1_diffusion: dt not provided; falling back to 0.001 s (1 kHz). "
            "Pass dt explicitly for correct physics factors."
        )
        dt_value = 1e-3
    else:
        dt_value = dt

    # Build/infer time grid (prefer provided grid to avoid JIT shape issues)
    if time_grid is None:
        # Infer maximum time and construct grid from dt
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
    dt: float | None,
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
    # P2-R6-04: Warn when dt is not provided — physics factors are dt-dependent.
    # Fallback 0.001s = APS-U standard XPCS frame rate (1 kHz), but callers
    # should always supply dt explicitly for reproducible results.
    if dt is None:
        logger.warning(
            "CMC compute_g1_total: dt not provided; falling back to 0.001 s (1 kHz). "
            "Pass dt explicitly for correct physics factors."
        )
    dt_value = dt if dt is not None else 0.001

    if time_grid is None:
        # Infer maximum time and construct grid from dt_value
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

    # Use math.pi (a plain Python float) rather than jnp.pi (a JAX array)
    # to keep these values as concrete scalars.  Do NOT wrap in float():
    # under pmap tracing, closure variables that flow through JAX operations
    # can be abstract tracers, and float() on a tracer raises
    # ConcretizationTypeError.  jnp.asarray is safe under any transform.
    wavevector_q_squared_half_dt = jnp.asarray(0.5 * (q**2) * dt_value)
    sinc_prefactor = jnp.asarray(0.5 / math.pi * q * L * dt_value)

    # DEBUG logging for CMC physics diagnosis
    if _debug:
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
