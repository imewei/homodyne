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

import jax
import jax.numpy as jnp
from jax import jit

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Physical and mathematical constants
PI = jnp.pi
EPS = 1e-12  # Numerical stability epsilon


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_len(obj):
    """JAX-safe length function that handles scalars, arrays, and JAX objects.

    Args:
        obj: Any object that might have a length or shape

    Returns:
        int: Length of the object, or 1 for scalars
    """
    # Handle JAX arrays and numpy arrays with shape attribute
    if hasattr(obj, "shape"):
        if obj.shape == () or len(obj.shape) == 0:
            # Scalar (0-dimensional array)
            return 1
        else:
            # Array - return first dimension size
            return obj.shape[0]

    # Handle objects with __len__ method (lists, tuples, etc.)
    if hasattr(obj, "__len__"):
        try:
            return len(obj)
        except TypeError:
            # This catches "len() of unsized object" errors
            return 1

    # Handle scalars (int, float, etc.)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        # Iterable but not string/bytes
        try:
            return len(list(obj))
        except (TypeError, ValueError):
            return 1

    # Default case: treat as scalar
    return 1


@jit
def safe_exp(x: jnp.ndarray, max_val: float = 700.0) -> jnp.ndarray:
    """Safe exponential to prevent overflow."""
    return jnp.exp(jnp.clip(x, -max_val, max_val))


@jit
def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    """Safe UNNORMALIZED sinc function: sin(x) / x (NOT sin(πx) / (πx)).

    This matches the reference implementation which uses sin(arg) / arg directly.
    The phase argument already includes all necessary scaling factors.
    """
    return jnp.where(jnp.abs(x) > EPS, jnp.sin(x) / x, 1.0)


# =============================================================================
# PHYSICS HELPER FUNCTIONS
# =============================================================================


@jit
def _calculate_diffusion_coefficient_impl_jax(
    time_array: jnp.ndarray,
    D0: float,
    alpha: float,
    D_offset: float,
) -> jnp.ndarray:
    """Calculate time-dependent diffusion coefficient using discrete evaluation.

    Follows reference v1 implementation: D_t[i] = D0 * (time_array[i] ** alpha) + D_offset
    Physical constraint: D(t) should be positive and finite

    Args:
        time_array: Array of time points
        D0: Diffusion coefficient amplitude
        alpha: Anomalous diffusion exponent
        D_offset: Baseline diffusion offset

    Returns:
        D(t) evaluated at each time point with physical bounds applied
    """
    # CRITICAL FIX: Add epsilon to prevent t=0 with negative alpha causing Inf/NaN gradients
    # When alpha < 0: t^alpha = 1/t^|alpha|, so t=0 → infinity
    # Adding epsilon ensures numerical stability: (t+ε)^alpha is always finite
    epsilon = 1e-10
    time_safe = time_array + epsilon

    # Compute diffusion coefficient
    D_t = D0 * (time_safe**alpha) + D_offset

    # Ensure positive values
    return jnp.maximum(D_t, 1e-10)


@jit
def _calculate_shear_rate_impl_jax(
    time_array: jnp.ndarray,
    gamma_dot_0: float,
    beta: float,
    gamma_dot_offset: float,
) -> jnp.ndarray:
    """Calculate time-dependent shear rate using discrete evaluation.

    Follows reference v1 implementation: γ̇_t[i] = γ̇₀ * (time_array[i] ** β) + γ̇_offset

    Args:
        time_array: Array of time points
        gamma_dot_0: Shear rate amplitude
        beta: Shear rate exponent
        gamma_dot_offset: Baseline shear rate offset

    Returns:
        γ̇(t) evaluated at each time point
    """
    # CRITICAL FIX: Replace t=0 with dt to prevent singularity when beta < 0
    # When beta < 0: t^beta = 1/t^|beta|, so t=0 → infinity
    # Strategy: Replace only the first element (t=0) with dt, leave others unchanged
    # This ensures smooth continuity: γ̇(dt), γ̇(dt), γ̇(2dt), ...

    # Infer dt from time grid
    if safe_len(time_array) > 1:
        dt = jnp.abs(time_array[1] - time_array[0])
    else:
        dt = 1e-3  # Fallback for single time point

    # Replace t=0 with dt: where(time_array == 0, dt, time_array)
    # This avoids discontinuity since both t[0] and t[1] map to dt
    time_safe = jnp.where(time_array == 0.0, dt, time_array)

    gamma_t = gamma_dot_0 * (time_safe**beta) + gamma_dot_offset
    # Ensure positive values with numerical stability floor
    return jnp.maximum(gamma_t, 1e-10)


# =============================================================================
# ELEMENT-WISE CORE PHYSICS (CMC ONLY)
# =============================================================================


@jit
def _compute_g1_diffusion_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
) -> jnp.ndarray:
    """Element-wise diffusion computation for CMC shards.

    Computes g1_diffusion for paired (t1[i], t2[i]) points using trapezoidal integration.
    This is the ONLY diffusion function for CMC - no meshgrid mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt

    Returns:
        Diffusion contribution to g1 (1D array, same length as t1/t2)
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Element-wise computation for each paired (t1[i], t2[i]) point
    # Compute integral from t1[i] to t2[i] for each i
    D_t1 = _calculate_diffusion_coefficient_impl_jax(t1, D0, alpha, D_offset)
    D_t2 = _calculate_diffusion_coefficient_impl_jax(t2, D0, alpha, D_offset)

    # Trapezoidal integration: ∫D(t)dt ≈ |t2-t1| * (D(t1) + D(t2)) / 2
    D_integral_elementwise = jnp.abs(t2 - t1) * (D_t1 + D_t2) / 2.0

    # Compute g1 using log-space for numerical stability
    log_g1 = -wavevector_q_squared_half_dt * D_integral_elementwise
    log_g1_clipped = jnp.clip(log_g1, -100.0, 0.0)
    g1_diffusion = jnp.exp(log_g1_clipped)

    return g1_diffusion  # Shape: (n_points,)


@jit
def _compute_g1_shear_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    sinc_prefactor: float,
) -> jnp.ndarray:
    """Element-wise shear computation for CMC shards.

    Computes g1_shear for paired (t1[i], t2[i]) points across unique phi angles.
    This is the ONLY shear function for CMC - no meshgrid mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        phi_unique: UNIQUE scattering angles (1D array, pre-filtered for unique values only)
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt

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

    # Element-wise computation for each paired (t1[i], t2[i]) point
    # Compute integral from t1[i] to t2[i] for each i
    gamma_t1 = _calculate_shear_rate_impl_jax(
        t1,
        gamma_dot_0,
        beta,
        gamma_dot_offset,
    )
    gamma_t2 = _calculate_shear_rate_impl_jax(
        t2,
        gamma_dot_0,
        beta,
        gamma_dot_offset,
    )

    # Trapezoidal integration: ∫γ̇(t)dt ≈ |t2-t1| * (γ̇(t1) + γ̇(t2)) / 2
    gamma_integral_elementwise = jnp.abs(t2 - t1) * (gamma_t1 + gamma_t2) / 2.0

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
    g1_shear = sinc_val**2

    return g1_shear  # Shape: (n_unique_phi, n_points)


@jit
def _compute_g1_total_elementwise(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi_unique: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
) -> jnp.ndarray:
    """Element-wise total g1 computation for CMC shards.

    Computes g1_total = g1_diffusion × g1_shear for paired (t1[i], t2[i]) points.
    This is the ONLY g1_total function for CMC - no meshgrid mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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
        params, t1, t2, wavevector_q_squared_half_dt
    )

    # Compute shear contribution: shape (n_unique_phi, n_points)
    g1_shear = _compute_g1_shear_elementwise(params, t1, t2, phi_unique, sinc_prefactor)

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
    dt: float = None,
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

    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factor using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

    return _compute_g1_diffusion_elementwise(
        params, t1, t2, wavevector_q_squared_half_dt
    )


def compute_g1_total(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
) -> jnp.ndarray:
    """Compute total g1 for CMC element-wise paired arrays.

    CMC-specific function - expects 1D paired arrays (t1[i], t2[i]).
    No meshgrid expansion, no dispatchers, direct element-wise computation only.

    **CRITICAL (Nov 2025): phi Parameter Requirements**

    This function expects phi to contain UNIQUE scattering angles only.
    When calling from MCMC backends, the caller MUST pre-compute unique values:

    ```python
    # CORRECT: Pre-compute unique phi before calling
    phi_unique = np.unique(np.asarray(phi))
    g1 = compute_g1_total(params, t1, t2, phi_unique, q, L, dt)

    # INCORRECT: Passing replicated phi array causes memory explosion
    # phi_replicated = [0, 0, 0, ..., 90, 90, 90, ...]  # 300K elements
    # g1 = compute_g1_total(..., phi_replicated, ...)  # BAD: Will call jnp.unique() during JIT
    ```

    **Why Pre-computation is Required:**

    - CMC pooled data replicates phi for each time point (e.g., 3 angles × 100K points = 300K array)
    - Computing unique values inside this function with `jnp.unique()` causes JAX concretization errors
      when called from NumPyro's JIT-traced MCMC model
    - Pre-computing unique values in non-JIT context (e.g., in `mcmc.py:_run_standard_nuts()`)
      avoids this issue and reduces memory from 80GB to 2.4MB

    **Backward Compatibility:**

    For non-MCMC use cases where phi is already unique or contains few duplicates,
    this function will still work correctly. The internal `jnp.unique()` call has been
    removed as of Nov 2025, so callers must ensure phi contains unique values.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1: Time array (1D, element-wise paired with t2)
        t2: Time array (1D, element-wise paired with t1)
        phi: Scattering angles (MUST be unique values, not replicated array)
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)

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
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

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

    return _compute_g1_total_elementwise(
        params,
        t1,
        t2,
        phi_unique,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
    )
