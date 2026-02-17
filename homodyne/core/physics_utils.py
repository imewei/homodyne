"""Shared Physics Utility Functions for Homodyne
================================================

This module provides common utility functions and physics helpers used by
both NLSQ (meshgrid) and CMC (element-wise) computational backends.

These functions were consolidated from:
- jax_backend.py
- physics_nlsq.py
- physics_cmc.py

to eliminate code duplication and ensure consistent behavior across backends.

Key Functions:
- safe_len: JAX-safe length function for scalars and arrays
- safe_exp: Overflow-protected exponential
- safe_sinc: Numerically stable unnormalized sinc function
- _calculate_diffusion_coefficient_impl_jax: Time-dependent diffusion D(t)
- _calculate_shear_rate_impl_jax: Time-dependent shear rate γ̇(t)
- _create_time_integral_matrix_impl_jax: Trapezoidal cumulative integral matrix
"""

import jax.numpy as jnp
from jax import jit

# Physical and mathematical constants
PI = jnp.pi
EPS = 1e-12  # Numerical stability epsilon


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def safe_len(obj: object) -> int:
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
            return int(obj.shape[0])

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
    """Safe exponential to prevent overflow.

    Args:
        x: Input array
        max_val: Maximum absolute value to clip to (default 700.0)

    Returns:
        exp(clip(x, -max_val, max_val))
    """
    return jnp.exp(jnp.clip(x, -max_val, max_val))


@jit
def safe_sinc(x: jnp.ndarray) -> jnp.ndarray:
    r"""Safe UNNORMALIZED sinc function: sin(x) / x (NOT sin(πx) / (πx)).

    This matches the reference implementation which uses sin(arg) / arg directly.
    The phase argument already includes all necessary scaling factors.

    P2-4: Uses a Taylor expansion near zero (1 - x²/6 + x⁴/120) for smooth
    gradient continuity. The old hard switch from sin(x)/x to 1.0 at ``|x|``\=EPS
    created a gradient discontinuity that caused spurious NUTS rejections near
    gamma_dot_t0 ≈ 0.

    Args:
        x: Input array

    Returns:
        sin(x)/x for ``|x|`` >= 1e-4, Taylor approximation for ``|x|`` < 1e-4
    """
    x2 = x * x
    near_zero = 1.0 - x2 / 6.0 + x2 * x2 / 120.0
    far = jnp.sin(x) / jnp.where(jnp.abs(x) > EPS, x, 1.0)  # avoid div/0
    return jnp.where(jnp.abs(x) < 1e-4, near_zero, far)


# =============================================================================
# PHYSICS HELPER FUNCTIONS
# =============================================================================


@jit
def calculate_diffusion_coefficient(
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
    # CRITICAL FIX: Replace near-zero values to prevent t=0 with negative alpha causing Inf/NaN
    # When alpha < 0: t^alpha = 1/t^|alpha|, so t=0 → infinity
    # Using jnp.maximum (not addition) to only affect near-zero values
    # Use dt/2 to preserve monotonicity: D(dt/2) < D(dt) for alpha > 0
    if time_array.shape[0] > 1:
        dt_inferred = jnp.abs(time_array[1] - time_array[0])
        epsilon = jnp.maximum(dt_inferred * 0.5, 1e-8)
    else:
        epsilon = 1e-3  # type: ignore[assignment]
    time_safe = jnp.maximum(time_array, epsilon)

    # Compute diffusion coefficient
    D_t = D0 * (time_safe**alpha) + D_offset

    # Ensure positive values
    return jnp.maximum(D_t, 1e-10)


@jit
def calculate_shear_rate(
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
    if time_array.shape[0] > 1:
        dt: jnp.ndarray | float = jnp.abs(time_array[1] - time_array[0])
    else:
        dt = 1e-3  # Fallback for single time point

    # Replace t=0 with dt: where(time_array == 0, dt, time_array)
    # This avoids discontinuity since both t[0] and t[1] map to dt
    time_safe = jnp.where(time_array == 0.0, dt, time_array)

    gamma_t = gamma_dot_0 * (time_safe**beta) + gamma_dot_offset
    # Ensure positive values with numerical stability floor
    return jnp.maximum(gamma_t, 1e-10)


@jit
def calculate_shear_rate_cmc(
    time_array: jnp.ndarray,
    gamma_dot_0: float,
    beta: float,
    gamma_dot_offset: float,
) -> jnp.ndarray:
    """Calculate time-dependent shear rate for CMC (element-wise) computations.

    This variant includes an additional safety check for consecutive zeros
    in CMC element-wise data where dt could be zero.

    Args:
        time_array: Array of time points
        gamma_dot_0: Shear rate amplitude
        beta: Shear rate exponent
        gamma_dot_offset: Baseline shear rate offset

    Returns:
        γ̇(t) evaluated at each time point
    """
    # Infer dt from time grid
    if time_array.shape[0] > 1:
        dt: jnp.ndarray | float = jnp.abs(time_array[1] - time_array[0])
        # CRITICAL FIX: Ensure dt > 0 to prevent 0^(negative beta) = infinity
        # CMC element-wise data can have consecutive zeros: t[0]=0, t[1]=0 → dt=0
        # This causes NaN when beta < 0 in gamma_t = gamma_dot_0 * (time_safe**beta)
        dt = jnp.maximum(
            dt, 1e-5
        )  # Minimum 1e-5 for numerical stability with negative beta
    else:
        dt = 1e-3  # Fallback for single time point

    # Replace t=0 with dt
    time_safe = jnp.where(time_array == 0.0, dt, time_array)

    gamma_t = gamma_dot_0 * (time_safe**beta) + gamma_dot_offset
    return jnp.maximum(gamma_t, 1e-10)


@jit
def create_time_integral_matrix(
    time_dependent_array: jnp.ndarray,
) -> jnp.ndarray:
    """Create time integral matrix using trapezoidal numerical integration.

    RESTORED (Nov 2025): Back to working implementation from homodyne-analysis/kernels.py
    The dt scaling happens in wavevector_q_squared_half_dt, NOT in this cumsum.

    Algorithm (from working version):

    1. Trapezoidal integration: cumsum[i] = Sum(k=0 to i-1) 0.5 * (f[k] + f[k+1])
    2. Compute difference matrix: matrix[i,j] = abs(cumsum[i] - cumsum[j])
    3. The dt factor is applied via wavevector_q_squared_half_dt = 0.5 * q^2 * dt

    This gives: matrix[i,j] = number of integration steps.
    Actual integral: dt * matrix[i,j] approximates the integral from 0 to abs(ti-tj) of f(t') dt'

    Benefits over simple cumsum:
    - Reduces oscillations from discretization by ~50%
    - Second-order accuracy (O(dt²)) vs. first-order (O(dt))
    - Eliminates checkerboard artifacts in diagonal-corrected results

    Args:
        time_dependent_array: f(t) evaluated at discrete time points

    Returns:
        Time integral matrix (in units of integration steps)
    """
    # Handle scalar input by converting to array
    time_dependent_array = jnp.atleast_1d(time_dependent_array)
    n = safe_len(time_dependent_array)

    # Step 1: Improved cumulative integration using trapezoidal rule
    # Trapezoidal: ∫f(t)dt ≈ dt × Σ(1/2)(f[i] + f[i+1])
    # The dt scaling happens in wavevector_q_squared_half_dt, not here
    if n > 1:
        # Compute trapezoidal averages: 0.5 * (f[i] + f[i+1])
        trap_avg = 0.5 * (time_dependent_array[:-1] + time_dependent_array[1:])

        # Cumulative sum of trapezoidal averages (NO dt scaling)
        cumsum_trap = jnp.cumsum(trap_avg)

        # Prepend 0 for initial condition: cumsum[0] = 0
        cumsum = jnp.concatenate([jnp.array([0.0]), cumsum_trap])
    else:
        # Single point: just use direct cumsum
        cumsum = jnp.cumsum(time_dependent_array)

    # Step 2: Create difference matrix
    # matrix[i,j] = |cumsum[i] - cumsum[j]| (number of integration steps)
    cumsum_i = cumsum[:, None]  # Shape: (n, 1)
    cumsum_j = cumsum[None, :]  # Shape: (1, n)
    diff = cumsum_i - cumsum_j

    # CRITICAL FIX: Use smooth approximation of abs() for gradient stability
    # jnp.abs() has undefined gradient at x=0, causing NaN in backpropagation
    # The diagonal of diff matrix is exactly 0 (cumsum[i] - cumsum[i] = 0)
    # Solution: sqrt(x² + ε) ≈ |x| but is differentiable everywhere
    # P0-2: epsilon=1e-12 (was 1e-20, below float32 machine epsilon ~1.2e-7).
    epsilon = 1e-12
    matrix = jnp.sqrt(diff**2 + epsilon)  # Shape: (n, n)

    return matrix


def trapezoid_cumsum(values: jnp.ndarray) -> jnp.ndarray:
    """Cumulative trapezoid integral without dt scaling (dt is applied outside).

    Returns cumsum so that ``cumsum[j] - cumsum[i]`` equals the trapezoidal sum
    over all intervals between indices ``i`` and ``j``. The caller applies a
    smooth absolute value to that difference when mapping each (t1, t2) pair,
    keeping gradients well-behaved at zero-length intervals.

    This is used by the CMC element-wise computations.

    Args:
        values: 1D array of values to integrate

    Returns:
        Cumulative trapezoidal sums
    """
    if safe_len(values) > 1:
        trap_avg = 0.5 * (values[:-1] + values[1:])
        cumsum_trap = jnp.cumsum(trap_avg)
        return jnp.concatenate([jnp.array([0.0], dtype=values.dtype), cumsum_trap])
    return jnp.cumsum(values)


# =============================================================================
# DIAGONAL CORRECTION
# =============================================================================
# Re-export from unified diagonal_correction module for backward compatibility.
# See homodyne/core/diagonal_correction.py for the canonical implementation.

from homodyne.core.diagonal_correction import (  # noqa: E402
    apply_diagonal_correction,  # noqa: F401
    apply_diagonal_correction_batch,  # noqa: F401
)

# =============================================================================
# BACKWARD COMPATIBILITY ALIASES
# =============================================================================

# These aliases maintain backward compatibility with existing code
_calculate_diffusion_coefficient_impl_jax = calculate_diffusion_coefficient
_calculate_shear_rate_impl_jax = calculate_shear_rate
_create_time_integral_matrix_impl_jax = create_time_integral_matrix
_trapezoid_cumsum = trapezoid_cumsum
