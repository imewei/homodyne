"""NLSQ Physics Backend - Meshgrid Computations Only
======================================================

This module provides meshgrid physics computations specifically for
Nonlinear Least Squares (NLSQ) optimization. It contains ONLY meshgrid functions
and has NO element-wise paired array computations.

Key Design Principles:
- Meshgrid computations for 2D correlation matrices
- No element-wise modes, no dispatchers
- Direct JIT-compiled functions only
- Completely independent from CMC physics

Architecture Fix (Nov 2025):
- Separated from element-wise CMC computations
- Prevents NLSQ from including CMC-specific code
- Clear separation of concerns

Physical Model:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²
g₁_total = g₁_diffusion × g₁_shear

Usage:
  from homodyne.core.jax_backend import compute_g2_scaled, apply_diagonal_correction
"""

import jax.numpy as jnp
from jax import jit

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Physical and mathematical constants
PI = jnp.pi
EPS = 1e-12  # Numerical stability epsilon

# JAX availability flag (for backward compatibility)
jax_available = True


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


@jit
def _create_time_integral_matrix_impl_jax(
    time_dependent_array: jnp.ndarray,
) -> jnp.ndarray:
    """Create time integral matrix using trapezoidal numerical integration.

    RESTORED (Nov 2025): Back to working implementation from homodyne-analysis/kernels.py
    The dt scaling happens in wavevector_q_squared_half_dt, NOT in this cumsum.

    Algorithm (from working version):
    1. Trapezoidal integration: cumsum[i] = Σ(k=0 to i-1) 0.5 * (f[k] + f[k+1])
    2. Compute difference matrix: matrix[i,j] = |cumsum[i] - cumsum[j]|
    3. The dt factor is applied via wavevector_q_squared_half_dt = 0.5 * q² * dt

    This gives: matrix[i,j] = number of integration steps
    Actual integral: dt * matrix[i,j] ≈ ∫₀^|tᵢ-tⱼ| f(t') dt'

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
    epsilon = 1e-20
    matrix = jnp.sqrt(diff**2 + epsilon)  # Shape: (n, n)

    return matrix


# =============================================================================
# MESHGRID CORE PHYSICS (NLSQ ONLY)
# =============================================================================


@jit
def _compute_g1_diffusion_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid diffusion computation for NLSQ optimization.

    Computes g1_diffusion for 2D meshgrid (all combinations of t1[i], t2[j]).
    This is the ONLY diffusion function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        t2: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        dt: Time step [seconds] - used ONLY for wavevector_q_squared_half_dt calculation

    Returns:
        Diffusion contribution to g1 (2D array: (n_times, n_times))

    Note:
        The data loader (xpcs_loader.py) converts frame indices to physical time:
        time_1d = np.linspace(0, dt * (end_frame - start_frame), matrix_size)
        So t1/t2 arrays contain physical time [0.0, 0.1, 0.2, ...], NOT frame indices.
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # Extract time array (t1 and t2 should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
        # So extract first COLUMN to get unique t1 values
        time_array = t1[:, 0]  # Extract first column for unique t1 values (in seconds)
    elif t1.ndim == 0:
        # Handle 0-dimensional (scalar) input
        time_array = jnp.atleast_1d(t1)
    else:
        # Handle 1D and other cases
        time_array = jnp.atleast_1d(t1)

    # ✅ CRITICAL FIX (Nov 11, 2025): time_array is ALREADY physical time in seconds
    # Data loader converts: time_1d = np.linspace(0, dt*(end-start), size)
    # Result: time_array = [0.0, 0.1, 0.2, ...] seconds (NOT frame indices!)
    # DO NOT multiply by dt - that would cause 10× time scale error!

    # Calculate D(t) at each time point (time_array already in seconds)
    D_t = _calculate_diffusion_coefficient_impl_jax(time_array, D0, alpha, D_offset)

    # Create diffusion integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
    D_integral = _create_time_integral_matrix_impl_jax(D_t)

    # Compute g1 correlation using log-space for numerical stability
    # This matches reference: g1 = exp(-wavevector_q_squared_half_dt * D_integral)
    #
    # LOG-SPACE CALCULATION FIX (Oct 2025):
    # Computing in log-space preserves precision across full dynamic range.
    # Old approach: clip(g1, 1e-10, 1.0) caused artificial plateaus (~16% of data)
    # New approach: clip in log-space, then exp() - no artificial plateaus
    log_g1 = -wavevector_q_squared_half_dt * D_integral

    # Clip in log-space to prevent numerical overflow/underflow
    # -700 → exp(-700) ≈ 1e-304 (near machine precision)
    # 0 → exp(0) = 1.0 (maximum physical value)
    log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)

    # Compute exponential with safeguards (safe_exp handles edge cases)
    g1_result = safe_exp(log_g1_bounded)

    # Apply ONLY upper bound (g1 ≤ 1.0 is physical constraint)
    # No lower bound clipping - preserves full precision down to machine epsilon
    # This eliminates artificial plateaus from overly aggressive clipping
    g1_safe = jnp.minimum(g1_result, 1.0)

    return g1_safe


@jit
def _compute_g1_shear_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid shear computation for NLSQ optimization.

    Computes g1_shear for 2D meshgrid (all combinations of t1[i], t2[j])
    across all phi angles. This is the ONLY shear function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        t2: Time meshgrid (2D) or 1D time array (PHYSICAL TIME in seconds)
        phi: Scattering angles (1D array)
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        dt: Time step [seconds] - used ONLY for sinc_prefactor calculation

    Returns:
        Shear contribution to g1 (3D array: (n_phi, n_times, n_times))

    Note:
        The data loader (xpcs_loader.py) converts frame indices to physical time.
        So t1/t2 arrays contain physical time [0.0, 0.1, 0.2, ...], NOT frame indices.
    """
    # Check params length - if < 7, we're in static mode (no shear)
    if safe_len(params) < 7:
        # Return ones for all phi angles and time combinations (g1_shear = 1)
        phi_array = jnp.atleast_1d(phi)
        n_phi = safe_len(phi_array)
        if t1.ndim == 2:
            n_times = t1.shape[0]
            return jnp.ones((n_phi, n_times, n_times))
        else:
            n_times = safe_len(t1)
            return jnp.ones((n_phi, n_times, n_times))

    gamma_dot_0, beta, gamma_dot_offset, phi0 = (
        params[3],
        params[4],
        params[5],
        params[6],
    )

    # Extract time array from t1 and t2 (should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
        # So extract first COLUMN to get unique t1 values
        time_array = t1[:, 0]  # Extract first column for unique t1 values (in seconds)
    elif t1.ndim == 0:
        # Handle 0-dimensional (scalar) input
        time_array = jnp.atleast_1d(t1)
    else:
        # Handle 1D and other cases
        time_array = jnp.atleast_1d(t1)

    # ✅ CRITICAL FIX (Nov 11, 2025): time_array is ALREADY physical time in seconds
    # Data loader converts: time_1d = np.linspace(0, dt*(end-start), size)
    # Result: time_array = [0.0, 0.1, 0.2, ...] seconds (NOT frame indices!)
    # DO NOT multiply by dt - that would cause 10× time scale error!

    # Calculate γ̇(t) at each time point (time_array already in seconds)
    gamma_t = _calculate_shear_rate_impl_jax(
        time_array,
        gamma_dot_0,
        beta,
        gamma_dot_offset,
    )

    # Create shear integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫γ̇(t)dt from i to j|
    # Create shear integral matrix using cumulative sums
    gamma_integral = _create_time_integral_matrix_impl_jax(gamma_t)

    # Fix phi shape if it has extra dimensions
    # Handle case where phi might be (1, 1, 1, 23) instead of (23,) or other malformed shapes
    phi = jnp.asarray(phi)  # Ensure it's a JAX array

    # Remove all leading singleton dimensions and flatten to 1D
    while phi.ndim > 1:
        if phi.ndim == 4 and phi.shape[:3] == (1, 1, 1):
            # Handle specific case (1, 1, 1, N) -> (N,)
            phi = jnp.squeeze(phi, axis=(0, 1, 2))
        elif phi.ndim > 1:
            # Handle any other multi-dimensional case by squeezing all singleton dims
            phi = jnp.squeeze(phi)
            # If squeezing didn't reduce dimensions, flatten
            if phi.ndim > 1:
                phi = phi.flatten()
                break

    # Compute sinc² for each phi angle using pre-computed factor (vectorized)
    phi_array = jnp.atleast_1d(phi)
    n_phi = safe_len(phi_array)
    n_times = safe_len(time_array)

    # Vectorized computation: compute all phi angles at once
    # angle_diff shape: (n_phi,)
    angle_diff = jnp.deg2rad(phi0 - phi_array)  # Use phi_array for consistency
    cos_term = jnp.cos(angle_diff)  # shape: (n_phi,)

    # Broadcast: prefactor shape (n_phi,), gamma_integral shape (n_times, n_times)
    # Need to expand prefactor to (n_phi, 1, 1) for proper broadcasting
    prefactor = sinc_prefactor * cos_term[:, None, None]  # shape: (n_phi, 1, 1)

    # Ensure gamma_integral has the expected 2D shape
    if gamma_integral.ndim != 2:
        raise ValueError(
            f"gamma_integral should be 2D, got shape {gamma_integral.shape}",
        )

    # Compute phase matrix for all phi angles: shape (n_phi, n_times, n_times)
    try:
        phase = (
            prefactor * gamma_integral
        )  # Broadcast: (n_phi, 1, 1) * (n_times, n_times)
    except Exception as e:
        # Enhanced error message for debugging
        raise ValueError(
            f"Broadcasting error in _compute_g1_shear_meshgrid: "
            f"prefactor.shape={prefactor.shape}, gamma_integral.shape={gamma_integral.shape}. "
            f"Original error: {e}",
        ) from e

    # Compute sinc² values: [sinc(Φ)]² for all phi angles
    sinc_val = safe_sinc(phase)
    sinc2_result = sinc_val**2

    return sinc2_result


@jit
def _compute_g1_total_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid total g1 computation for NLSQ optimization.

    Computes g1_total = g1_diffusion × g1_shear for 2D meshgrid.
    This is the ONLY g1_total function for NLSQ - no element-wise mode.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1: Time meshgrid (2D) or 1D time array (FRAME INDICES)
        t2: Time meshgrid (2D) or 1D time array (FRAME INDICES)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        dt: Time step per frame [seconds] - for frame→time conversion

    Returns:
        Total g1 correlation function (3D array: (n_phi, n_times, n_times))
    """
    # Compute diffusion contribution: shape (n_times, n_times)
    g1_diff = _compute_g1_diffusion_meshgrid(
        params, t1, t2, wavevector_q_squared_half_dt, dt
    )

    # Compute shear contribution: shape (n_phi, n_times, n_times)
    g1_shear = _compute_g1_shear_meshgrid(params, t1, t2, phi, sinc_prefactor, dt)

    # Broadcast g1_diff from (n_times, n_times) to (n_phi, n_times, n_times)
    n_phi = g1_shear.shape[0]
    g1_diff_broadcasted = jnp.broadcast_to(
        g1_diff[None, :, :],
        (n_phi, g1_diff.shape[0], g1_diff.shape[1]),
    )

    # Multiply: g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]
    try:
        g1_total = g1_diff_broadcasted * g1_shear
    except Exception as e:
        # Enhanced error message for debugging
        raise ValueError(
            f"Broadcasting error in _compute_g1_total_meshgrid: "
            f"g1_diff_broadcasted.shape={g1_diff_broadcasted.shape}, g1_shear.shape={g1_shear.shape}. "
            f"Original error: {e}",
        ) from e

    # Apply physical bounds for g1: (0, 2]
    # Lower bound: epsilon (effectively 0) for numerical stability
    # Upper bound: 2.0 (loose bound allowing for experimental variations beyond theoretical 1.0)
    # ✅ UPDATED (Nov 11, 2025): Loosened bounds to g1 ∈ (0, 2] for fitting flexibility
    epsilon = 1e-10
    g1_bounded = jnp.clip(g1_total, epsilon, 2.0)

    return g1_bounded


@jit
def _compute_g2_scaled_meshgrid(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Meshgrid g2 computation for NLSQ optimization.

    Core homodyne equation: g₂ = offset + contrast × [g₁]²

    The homodyne scattering equation is g₂ = 1 + β×g₁², where the baseline "1"
    is the constant background. In our implementation, this baseline is included
    in the offset parameter (offset ≈ 1.0 for physical measurements).

    For theoretical fits: Use offset=1.0, contrast=1.0 to get g₂ = 1 + g₁²
    For experimental fits: offset and contrast are free parameters centered around 1.0 and 0.5

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation (FRAME INDICES)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt
        contrast: Contrast parameter (β in literature) - typically [0, 1]
        offset: Baseline level (includes the "1" from physics) - typically ~1.0
        dt: Time step per frame [seconds] - for frame→time conversion

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    g1 = _compute_g1_total_meshgrid(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        dt,
    )

    # Homodyne physics: g₂ = offset + contrast × [g₁]²
    # The baseline "1" is included in the offset parameter (offset ≈ 1.0 for physical data)
    # NOTE: Bounds enforcement is handled by NLSQ optimization bounds, NOT by clipping here.
    # Clipping parameters inside the physics function creates a mismatch between:
    # - The parameter values NLSQ stores (unclipped)
    # - The parameter values actually used in computation (clipped)
    # This causes visualization to use different values than optimization used!
    g2 = offset + contrast * g1**2

    # Apply physical bounds: 0.5 < g2 ≤ 2.5
    # Updated bounds (Nov 11, 2025) to reflect realistic homodyne detection range:
    # - Lower bound 0.5: Allows for significant negative offset deviations
    # - Upper bound 2.5: Theoretical maximum for g₂ = 1 + 1×1² = 2, plus 25% headroom
    # - Physical constraint: 0.5 ≤ g2 ≤ 2.5 for homodyne detection
    g2_bounded = jnp.clip(g2, 0.5, 2.5)

    return g2_bounded


@jit
def apply_diagonal_correction(c2_mat: jnp.ndarray) -> jnp.ndarray:
    """Apply diagonal correction to two-time correlation matrix.

    This function replaces the diagonal elements (t₁=t₂) with interpolated values
    from adjacent off-diagonal elements. This removes the bright autocorrelation peak
    and isolates the cross-correlation dynamics.

    Based on pyXPCSViewer's correct_diagonal_c2 function. This is a critical
    preprocessing step that MUST be applied consistently to both experimental data
    and theoretical model predictions during optimization.

    Algorithm:
    1. Extract side band: elements at (i, i+1) for i=0..N-2
    2. Compute diagonal values as average of adjacent off-diagonals:
       - diag[0] = side_band[0] (edge case)
       - diag[i] = (side_band[i-1] + side_band[i]) / 2 for i=1..N-2
       - diag[N-1] = side_band[N-2] (edge case)
    3. Replace diagonal with computed values

    Args:
        c2_mat: Two-time correlation matrix with shape (N, N)
                Must be square matrix with N ≥ 2

    Returns:
        Corrected correlation matrix with interpolated diagonal

    Example:
        >>> c2 = jnp.array([[5.0, 1.2, 1.1],
        ...                 [1.2, 5.0, 1.3],
        ...                 [1.1, 1.3, 5.0]])
        >>> c2_corrected = apply_diagonal_correction(c2)
        >>> # Diagonal now contains interpolated values, not 5.0

    References:
        - pyXPCSViewer: https://github.com/AdvancedPhotonSource/pyXPCSViewer
        - XPCS Analysis: He et al. PNAS 2024, doi:10.1073/pnas.2401162121
    """
    size = c2_mat.shape[0]

    # Extract side band: off-diagonal elements adjacent to main diagonal
    # side_band[i] = c2_mat[i, i+1] for i in range(size-1)
    indices_i = jnp.arange(size - 1)
    indices_j = jnp.arange(1, size)
    side_band = c2_mat[indices_i, indices_j]  # Shape: (size-1,)

    # Compute diagonal values as average of adjacent off-diagonal elements
    # This implementation matches xpcs_loader.py:924-953 but uses pure JAX ops
    diag_val = jnp.zeros(size)

    # Add left neighbors: diag_val[:-1] += side_band
    diag_val = diag_val.at[:-1].add(side_band)

    # Add right neighbors: diag_val[1:] += side_band
    diag_val = diag_val.at[1:].add(side_band)

    # Normalize by number of neighbors (1 for edges, 2 for middle)
    norm = jnp.ones(size)
    norm = norm.at[1:-1].set(2.0)  # Middle elements have 2 neighbors

    diag_val = diag_val / norm

    # Replace diagonal with computed values using JAX immutable array operations
    diag_indices = jnp.diag_indices(size)
    c2_corrected = c2_mat.at[diag_indices].set(diag_val)

    return c2_corrected


# =============================================================================
# PUBLIC API FUNCTIONS (NLSQ ONLY)
# =============================================================================


def compute_g2_scaled(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """Compute g2 for NLSQ using meshgrid 2D matrices.

    NLSQ-specific function - creates meshgrid for 2D correlation matrices.
    No element-wise mode, direct meshgrid computation only.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation (1D arrays will be converted to meshgrid)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Normal time vectors: create 2D meshgrids for all (t1[i], t2[j]) pairs
        # CRITICAL: Must match caller's convention: t1_grid, t2_grid = meshgrid(t, t, 'ij')
        t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    # Compute the physics factors using configuration dt
    # IMPORTANT: Config dt value will OVERRIDE this default
    # Default dt = 0.001s if not in config (APS-U standard XPCS frame rate: 1ms)
    dt_value = dt if dt is not None else 0.001
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt_value
    sinc_prefactor = 0.5 / PI * q * L * dt_value

    return _compute_g2_scaled_meshgrid(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        contrast,
        offset,
        dt_value,
    )


@jit
def compute_g2_scaled_with_factors(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    contrast: float,
    offset: float,
    dt: float,
) -> jnp.ndarray:
    """JIT-optimized g2 computation using pre-computed physics factors.

    This is the hybrid architecture functional core - accepts pre-computed
    factors directly, avoiding runtime computation. Suitable for use with
    HomodyneModel where factors are computed once at initialization.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids for correlation calculation (FRAME INDICES)
        phi: Scattering angles [degrees]
        wavevector_q_squared_half_dt: Pre-computed factor (0.5 * q² * dt)
        sinc_prefactor: Pre-computed factor (q * L * dt / 2π)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step per frame [seconds] - for frame→time conversion

    Returns:
        g2 correlation function with scaled fitting

    Note:
        This function is JIT-compiled for maximum performance.
        Use with HomodyneModel for best results.
        For NLSQ backend, simply calls _compute_g2_scaled_meshgrid.
    """
    # Directly call meshgrid implementation
    return _compute_g2_scaled_meshgrid(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        contrast,
        offset,
        dt,
    )
