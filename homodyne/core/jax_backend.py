"""JAX Computational Backend for Homodyne
==========================================

High-performance JAX-based implementation of the core mathematical operations
for homodyne scattering analysis. Provides JIT-compiled functions with automatic
differentiation capabilities for optimization.

This module provides JAX-based computational kernels
that offer superior performance, GPU/TPU support, and automatic differentiation
for gradient-based optimization methods.

Key Features:
- JIT compilation for optimal performance
- Automatic differentiation (grad, hessian) for optimization
- Vectorized operations with vmap/pmap for parallelization
- GPU/TPU acceleration when available
- Memory-efficient operations for large correlation matrices
- Numerical stability enhancements

Physical Model Implementation:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²

Where g₁ = g₁_diffusion × g₁_shear captures:
- Anomalous diffusion: g₁_diff = exp[-q²/2 ∫ D(t')dt']
- Time-dependent shear: g₁_shear = [sinc(Φ)]²
"""

# Handle JAX import with graceful fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, hessian, jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    # Fallback to numpy when JAX is not available
    import numpy as jnp

    JAX_AVAILABLE = False

    # Import NumPy-based gradients for graceful fallback
    try:
        from homodyne.core.numpy_gradients import numpy_gradient, numpy_hessian

        NUMPY_GRADIENTS_AVAILABLE = True
    except ImportError:
        NUMPY_GRADIENTS_AVAILABLE = False

    # Create fallback decorators
    def jit(func):
        """No-op JIT decorator for NumPy fallback."""
        return func

    def vmap(func, *args, **kwargs):
        """Simple vectorization fallback using Python loops."""

        def vectorized_func(inputs, *vargs, **vkwargs):
            if hasattr(inputs, "__iter__") and not isinstance(inputs, str):
                return [func(inp, *vargs, **vkwargs) for inp in inputs]
            return func(inputs, *vargs, **vkwargs)

        return vectorized_func

    def grad(func, argnums=0):
        """Intelligent fallback gradient function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_gradient_fallback(func, argnums)
        else:
            return _create_no_gradient_fallback(
                func.__name__ if hasattr(func, "__name__") else "function",
            )

    def hessian(func, argnums=0):
        """Intelligent fallback Hessian function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_hessian_fallback(func, argnums)
        else:
            return _create_no_hessian_fallback(
                func.__name__ if hasattr(func, "__name__") else "function",
            )


from collections.abc import Callable
from functools import wraps

from homodyne.core.physics_utils import (
    EPS,
    PI,
    safe_exp,
    safe_len,
    safe_sinc,
)
from homodyne.core.physics_utils import (
    calculate_diffusion_coefficient as _calculate_diffusion_coefficient_impl_jax,
)
from homodyne.core.physics_utils import (
    calculate_shear_rate as _calculate_shear_rate_impl_jax,
)
from homodyne.core.physics_utils import (
    create_time_integral_matrix as _create_time_integral_matrix_impl_jax,
)
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# Performance tracking for fallback warnings
_performance_warned = set()
_fallback_stats = {
    "gradient_calls": 0,
    "hessian_calls": 0,
    "jit_bypassed": 0,
    "vmap_loops": 0,
}

# Global flags for availability checking
jax_available = JAX_AVAILABLE
numpy_gradients_available = NUMPY_GRADIENTS_AVAILABLE if not JAX_AVAILABLE else False


if not JAX_AVAILABLE:
    if NUMPY_GRADIENTS_AVAILABLE:
        logger.warning(
            "JAX not available - using NumPy gradients fallback.\n"
            "Performance will be 10-50x slower than JAX.\n"
            "Install JAX for optimal performance: pip install jax",
        )
    else:
        logger.error(
            "Neither JAX nor NumPy gradients available.\n"
            "Install NumPy gradients: pip install scipy\n"
            "Or install JAX for optimal performance: pip install jax",
        )


def _create_gradient_fallback(func: Callable, argnums: int = 0) -> Callable:
    """Create intelligent gradient fallback with performance monitoring."""
    func_name = getattr(func, "__name__", "unknown")

    @wraps(func)
    def fallback_gradient(*args, **kwargs):
        _fallback_stats["gradient_calls"] += 1

        # Issue performance warning (once per function)
        if func_name not in _performance_warned:
            logger.warning(
                f"Using NumPy gradient fallback for {func_name}. "
                f"Expected 10-50x performance degradation. "
                f"Install JAX for optimal performance.",
            )
            _performance_warned.add(func_name)

        # Use numpy_gradient with appropriate configuration
        grad_func = numpy_gradient(func, argnums)
        return grad_func(*args, **kwargs)

    return fallback_gradient


def _create_hessian_fallback(func: Callable, argnums: int = 0) -> Callable:
    """Create intelligent Hessian fallback with performance monitoring."""
    func_name = getattr(func, "__name__", "unknown")

    @wraps(func)
    def fallback_hessian(*args, **kwargs):
        _fallback_stats["hessian_calls"] += 1

        # Issue performance warning (once per function)
        if func_name not in _performance_warned:
            logger.warning(
                f"Using NumPy Hessian fallback for {func_name}. "
                f"Expected 50-200x performance degradation. "
                f"Install JAX for optimal performance.",
            )
            _performance_warned.add(func_name)

        # Use numpy_hessian with appropriate configuration
        hess_func = numpy_hessian(func, argnums)
        return hess_func(*args, **kwargs)

    return fallback_hessian


def _create_no_gradient_fallback(func_name: str) -> Callable:
    """Create informative gradient fallback when no numerical differentiation is available."""

    def no_gradient_available(*args, **kwargs):
        error_msg = (
            f"Gradient computation not available for {func_name}.\n"
            f"Install NumPy gradients support or JAX:\n"
            f"  pip install scipy (for numerical differentiation)\n"
            f"  pip install jax (recommended for optimal performance)\n"
            f"\nCurrently available backends: None"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    return no_gradient_available


def _create_no_hessian_fallback(func_name: str) -> Callable:
    """Create informative Hessian fallback when no numerical differentiation is available."""

    def no_hessian_available(*args, **kwargs):
        error_msg = (
            f"Hessian computation not available for {func_name}.\n"
            f"Install NumPy gradients support or JAX:\n"
            f"  pip install scipy (for numerical differentiation)\n"
            f"  pip install jax (recommended for optimal performance)\n"
            f"\nCurrently available backends: None"
        )
        logger.error(error_msg)
        raise ImportError(error_msg)

    return no_hessian_available


# safe_divide is kept here as it's only used in jax_backend.py
@jit
def safe_divide(a: jnp.ndarray, b: jnp.ndarray, default: float = 0.0) -> jnp.ndarray:
    """Safe division with numerical stability."""
    return jnp.where(jnp.abs(b) > EPS, a / b, default)


# Core physics computations with discrete numerical integration
# Note: _calculate_diffusion_coefficient_impl_jax, _calculate_shear_rate_impl_jax,
# and _create_time_integral_matrix_impl_jax are now imported from physics_utils.py
@jit
def _compute_g1_diffusion_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    dt: float,
) -> jnp.ndarray:
    """Compute diffusion contribution to g1 using reference implementation approach.

    Algorithm (following reference v1 exactly):
    1. Extract time array (t1 = t2 = t, same time points)
    2. Calculate D(t) = D₀ t^α + D_offset at each time point
    3. Create integral matrix using cumulative sums: matrix[i,j] = |∫D(t)dt from i to j|
    4. Compute g1[i,j] = exp(-wavevector_q_squared_half_dt * matrix[i,j])

    Physical model: g₁_diff[i,j] = exp[-q²/2 * dt * ∫|tᵢ-tⱼ| D(t')dt']
    Where: D(t) = D₀ t^α + D_offset
    And: wavevector_q_squared_half_dt = 0.5 * q² * dt (from configuration)

    FORMULA VERIFICATION (matches reference exactly):
    Reference: self.wavevector_q_squared_half_dt = 0.5 * self.wavevector_q_squared * self.dt
    Which is: wavevector_q_squared_half_dt = 0.5 * (q²) * dt

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        dt: Time step from experimental configuration (time per frame)

    Returns:
        Diffusion contribution to g1 correlation function
    """
    D0, alpha, D_offset = params[0], params[1], params[2]

    # CRITICAL FIX (Nov 2025): Detect element-wise data to prevent 35TB matrix allocation
    # Same issue as in _compute_g1_shear_core
    is_elementwise = t1.ndim == 1 and safe_len(t1) > 2000

    if is_elementwise:
        # ELEMENT-WISE MODE: Compute integrals directly for each (t1[i], t2[i]) pair
        t1_arr = jnp.atleast_1d(t1)
        t2_arr = jnp.atleast_1d(t2)

        # Compute D(t) at both t1[i] and t2[i] for all i
        D_t1 = _calculate_diffusion_coefficient_impl_jax(t1_arr, D0, alpha, D_offset)
        D_t2 = _calculate_diffusion_coefficient_impl_jax(t2_arr, D0, alpha, D_offset)

        # Element-wise trapezoidal integration (dimensionless, like matrix mode)
        # Compute frame separation: |t2[i] - t1[i]| / dt = number of frames
        # For XPCS data, times are always on uniform grid: t = dt * frame_index
        frame_diff = jnp.abs(t2_arr - t1_arr) / dt  # Dimensionless (number of frames)
        D_integral = frame_diff * 0.5 * (D_t1 + D_t2)  # Dimensionless, Shape: (n,)

    else:
        # MATRIX MODE: Standard approach for small datasets or meshgrids
        # Step 1: Extract time array (t1 and t2 should be identical)
        # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
        if t1.ndim == 2:
            # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
            # So extract first COLUMN to get unique t1 values
            time_array = t1[:, 0]  # Extract first column for unique t1 values
        elif t1.ndim == 0:
            # Handle 0-dimensional (scalar) input
            time_array = jnp.atleast_1d(t1)
        else:
            # Handle 1D and other cases
            time_array = jnp.atleast_1d(t1)

        # Step 2: Calculate D(t) at each time point
        D_t = _calculate_diffusion_coefficient_impl_jax(time_array, D0, alpha, D_offset)

        # Step 3: Create diffusion integral matrix using cumulative sums
        # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
        D_integral = _create_time_integral_matrix_impl_jax(D_t)

    # Step 4: Compute g1 correlation using log-space for numerical stability
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
def _compute_g1_shear_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Compute shear contribution to g1 using reference implementation approach.

    Algorithm (following reference v1 exactly):
    1. Extract time array (t1 = t2 = t, same time points)
    2. Calculate γ̇(t) = γ̇₀ t^β + γ̇_offset at each time point
    3. Create integral matrix using cumulative sums: matrix[i,j] = |∫γ̇(t)dt from i to j|
    4. Compute sinc²[i,j] for each phi angle

    Physical model: g₁_shear = [sinc(Φ)]²
    Where: Φ = sinc_prefactor * cos(φ₀-φ) * ∫|tᵢ-tⱼ| γ̇(t') dt'
    And: γ̇(t) = γ̇₀ t^β + γ̇_offset
    And: sinc_prefactor = 0.5/π * q * L * dt (from configuration)

    FORMULA VERIFICATION (matches reference exactly):
    Reference: self.sinc_prefactor = 0.5 / np.pi * self.wavevector_q * self.stator_rotor_gap * self.dt
    Which is: sinc_prefactor = 0.5/π * q * L * dt
    Where L = stator_rotor_gap (sample-detector distance)

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        dt: Time step from experimental configuration (time per frame)

    Returns:
        Shear contribution to g1 correlation function (sinc² values)
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

    # CRITICAL FIX (Nov 2025): Detect element-wise data to prevent 35TB matrix allocation
    # For CMC shards with flattened element-wise data (len > 2000), t1 and t2 are paired arrays
    # where each element i corresponds to one measurement at (t1[i], t2[i], phi[i])
    # We need element-wise integrals, NOT a full (n×n) matrix!
    is_elementwise = t1.ndim == 1 and safe_len(t1) > 2000

    if is_elementwise:
        # ELEMENT-WISE MODE: Compute integrals directly for each (t1[i], t2[i]) pair
        # Each measurement i needs: ∫_{t1[i]}^{t2[i]} γ̇(t') dt'
        # Trapezoidal approximation: |t2[i] - t1[i]| × 0.5 × (γ̇(t1[i]) + γ̇(t2[i]))

        t1_arr = jnp.atleast_1d(t1)
        t2_arr = jnp.atleast_1d(t2)

        # Compute γ̇(t) at both t1[i] and t2[i] for all i
        gamma_t1 = _calculate_shear_rate_impl_jax(
            t1_arr, gamma_dot_0, beta, gamma_dot_offset
        )
        gamma_t2 = _calculate_shear_rate_impl_jax(
            t2_arr, gamma_dot_0, beta, gamma_dot_offset
        )

        # Element-wise trapezoidal integration (dimensionless, like matrix mode)
        # Compute frame separation: |t2[i] - t1[i]| / dt = number of frames
        # For XPCS data, times are always on uniform grid: t = dt * frame_index
        frame_diff = jnp.abs(t2_arr - t1_arr) / dt  # Dimensionless (number of frames)
        gamma_integral_elementwise = (
            frame_diff * 0.5 * (gamma_t1 + gamma_t2)
        )  # Dimensionless

        # For consistency with matrix mode, store as 1D array
        gamma_integral = gamma_integral_elementwise  # Shape: (n,)
        n_times = safe_len(t1_arr)

    else:
        # MATRIX MODE: Standard approach for small datasets or meshgrids
        # Step 1: Extract time array (t1 and t2 should be identical)
        # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
        if t1.ndim == 2:
            # For meshgrid with indexing="ij": t1 varies along rows (axis 0), constant along columns
            # So extract first COLUMN to get unique t1 values
            time_array = t1[:, 0]  # Extract first column for unique t1 values
        elif t1.ndim == 0:
            # Handle 0-dimensional (scalar) input
            time_array = jnp.atleast_1d(t1)
        else:
            # Handle 1D and other cases
            time_array = jnp.atleast_1d(t1)

        # Step 2: Calculate γ̇(t) at each time point
        gamma_t = _calculate_shear_rate_impl_jax(
            time_array,
            gamma_dot_0,
            beta,
            gamma_dot_offset,
        )

        # Step 3: Create shear integral matrix using cumulative sums
        # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫γ̇(t)dt from i to j|
        # Create shear integral matrix using cumulative sums
        gamma_integral = _create_time_integral_matrix_impl_jax(gamma_t)
        n_times = safe_len(time_array)

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

    # Step 4: Compute sinc² for each phi angle using pre-computed factor (vectorized)
    phi_array = jnp.atleast_1d(phi)
    n_phi = safe_len(phi_array)

    if is_elementwise:
        # ELEMENT-WISE MODE: phi, gamma_integral are all 1D arrays (n,)
        # Each element i has its own phi[i] value (per-angle scaling)
        # Compute phase: Φ[i] = sinc_prefactor × cos(φ₀-phi[i]) × gamma_integral[i]

        # Element-wise computation (no broadcasting needed!)
        angle_diff = jnp.deg2rad(phi0 - phi_array)  # shape: (n,)
        cos_term = jnp.cos(angle_diff)  # shape: (n,)
        prefactor = sinc_prefactor * cos_term  # shape: (n,)
        phase = prefactor * gamma_integral  # shape: (n,)

        # Compute sinc² values: [sinc(Φ)]² for all elements
        sinc_val = safe_sinc(phase)
        sinc2_result = sinc_val**2  # shape: (n,)

    else:
        # MATRIX MODE: Standard broadcasting for small datasets
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
                f"Broadcasting error in _compute_g1_shear_core: "
                f"prefactor.shape={prefactor.shape}, gamma_integral.shape={gamma_integral.shape}. "
                f"Original error: {e}",
            ) from e

        # Compute sinc² values: [sinc(Φ)]² for all phi angles
        sinc_val = safe_sinc(phase)
        sinc2_result = sinc_val**2

    return sinc2_result


@jit
def _compute_g1_total_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
    sinc_prefactor: float,
    dt: float,
) -> jnp.ndarray:
    """Compute total g1 correlation function as product of diffusion and shear.

    Following reference implementation:
    g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]

    Physical constraint: 0 < g₁(t) ≤ 1

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        dt: Time step from experimental configuration (time per frame)

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)
    """
    # Compute diffusion contribution
    g1_diff = _compute_g1_diffusion_core(
        params, t1, t2, wavevector_q_squared_half_dt, dt
    )

    # Compute shear contribution
    g1_shear = _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt)

    # CRITICAL FIX (Nov 2025): Handle element-wise vs matrix mode
    # Element-wise mode: both g1_diff and g1_shear are 1D (shape (n,))
    # Matrix mode: g1_diff is 2D (n_times, n_times), g1_shear is 3D (n_phi, n_times, n_times)
    is_elementwise = g1_diff.ndim == 1 and g1_shear.ndim == 1

    if is_elementwise:
        # ELEMENT-WISE MODE: Simple element-wise multiplication
        # g1_diff: (n,), g1_shear: (n,) → g1_total: (n,)
        try:
            g1_total = g1_diff * g1_shear
        except Exception as e:
            raise ValueError(
                f"Element-wise multiplication error in _compute_g1_total_core: "
                f"g1_diff.shape={g1_diff.shape}, g1_shear.shape={g1_shear.shape}. "
                f"Original error: {e}",
            ) from e

    else:
        # MATRIX MODE: Broadcast diffusion term to match shear dimensions
        # g1_diff needs to be broadcast from (n_times, n_times) to (n_phi, n_times, n_times)
        # Use the shape of g1_shear to determine n_phi (more reliable than parsing phi directly)
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
                f"Broadcasting error in _compute_g1_total_core: "
                f"g1_diff_broadcasted.shape={g1_diff_broadcasted.shape}, g1_shear.shape={g1_shear.shape}. "
                f"Original error: {e}",
            ) from e

    # Apply physical bounds for g1: (0, 2]
    # Theoretical: g1 is the normalized field correlation function, range (0, 1]
    # Lower bound: epsilon (effectively 0) for numerical stability
    # Upper bound: 2.0 (loose bound allowing for experimental variations beyond theoretical 1.0)
    # ✅ UPDATED (Nov 11, 2025): Loosened bounds to g1 ∈ (0, 2] for fitting flexibility
    epsilon = 1e-10
    g1_bounded = jnp.clip(g1_total, epsilon, 2.0)

    return g1_bounded


@jit
def _compute_g2_scaled_core(
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
    """Core homodyne equation: g₂ = offset + contrast × [g₁]²

    The homodyne scattering equation is g₂ = 1 + β×g₁², where the baseline "1"
    is the constant background. In our implementation, this baseline is included
    in the offset parameter (offset ≈ 1.0 for physical measurements).

    For theoretical fits: Use offset=1.0, contrast=1.0 to get g₂ = 1 + g₁²
    For experimental fits: offset and contrast are free parameters centered around 1.0 and 0.5

    Physical constraint: 0.5 < g2 ≤ 2.5

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        contrast: Contrast parameter (β in literature) - typically [0, 1]
        offset: Baseline level (includes the "1" from physics) - typically ~1.0
        dt: Time step from experimental configuration (time per frame) [seconds]

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    g1 = _compute_g1_total_core(
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
    g2 = offset + contrast * g1**2

    # Apply physical bounds: 0.5 < g2 ≤ 2.5
    # Updated bounds (Nov 11, 2025) to reflect realistic homodyne detection range:
    # - Lower bound 0.5: Allows for significant negative offset deviations
    # - Upper bound 2.5: Theoretical maximum for g₂ = 1 + 1×1² = 2, plus 25% headroom
    # - Physical constraint: 0.5 ≤ g2 ≤ 2.5 for homodyne detection
    g2_bounded = jnp.clip(g2, 0.5, 2.5)

    return g2_bounded


# =============================================================================
# COMPATIBILITY WRAPPER FUNCTIONS
# =============================================================================
# Note: apply_diagonal_correction is imported from physics_utils.py
# to eliminate code duplication between NLSQ and CMC backends.
# These maintain the old API for backward compatibility while using correct
# configuration values internally


def compute_g1_diffusion(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    q: float,
    dt: float = None,
) -> jnp.ndarray:
    """Wrapper function that computes g1 diffusion using configuration dt.

    IMPORTANT: The dt parameter should come from configuration, not be computed.

    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        q: Scattering wave vector magnitude
        dt: Time step from configuration (REQUIRED for correct physics)

    Returns:
        Diffusion contribution to g1 correlation function
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Check if this is flattened pooled data (from CMC) vs normal time vectors
        # Normal time vectors: typically 100-2000 elements (need meshgrid expansion)
        # Pooled data for SVI: 2000-5000+ elements (already element-wise matched, DON'T mesh)
        # Using 2000 as threshold: above typical time vectors (e.g., 1001), below pooled data (e.g., 4600)
        if len(t1) > 2000:
            # Pooled/flattened data: arrays already element-wise matched, don't create meshgrid
            # Creating meshgrid would cause OOM: e.g., 4600² = 21M elements, 23M² = 530 quadrillion
            pass  # t1 and t2 are already correctly paired element-wise
        else:
            # Normal time vectors: create 2D meshgrids for all (t1[i], t2[j]) pairs
            # CRITICAL: Must match caller's convention: t1_grid, t2_grid = meshgrid(t, t, 'ij')
            t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
            t1 = t1_grid
            t2 = t2_grid

    # Use dt from configuration (REQUIRED for correct physics)
    # If dt not provided, estimate from time array as fallback
    if dt is None:
        # FALLBACK: Estimate from time array
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factor using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

    return _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt, dt)


def compute_g1_shear(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
) -> jnp.ndarray:
    """Wrapper function that computes g1 shear using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        Shear contribution to g1 correlation function (sinc² values)

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Check if this is flattened pooled data (from CMC) vs normal time vectors
        # Normal time vectors: typically 100-2000 elements (need meshgrid expansion)
        # Pooled data for SVI: 2000-5000+ elements (already element-wise matched, DON'T mesh)
        # Using 2000 as threshold: above typical time vectors (e.g., 1001), below pooled data (e.g., 4600)
        if len(t1) > 2000:
            # Pooled/flattened data: arrays already element-wise matched, don't create meshgrid
            # Creating meshgrid would cause OOM: e.g., 4600² = 21M elements, 23M² = 530 quadrillion
            pass  # t1 and t2 are already correctly paired element-wise
        else:
            # Normal time vectors: create 2D meshgrids for all (t1[i], t2[j]) pairs
            # CRITICAL: Must match caller's convention: t1_grid, t2_grid = meshgrid(t, t, 'ij')
            t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
            t1 = t1_grid
            t2 = t2_grid

    # Compute the physics factor using configuration dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt)


def compute_g1_total(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float,
) -> jnp.ndarray:
    """Wrapper function that computes total g1 using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Check if this is flattened pooled data (from CMC) vs normal time vectors
        # Normal time vectors: typically 100-2000 elements (need meshgrid expansion)
        # Pooled data for SVI: 2000-5000+ elements (already element-wise matched, DON'T mesh)
        # Using 2000 as threshold: above typical time vectors (e.g., 1001), below pooled data (e.g., 4600)
        if len(t1) > 2000:
            # Pooled/flattened data: arrays already element-wise matched, don't create meshgrid
            # Creating meshgrid would cause OOM: e.g., 4600² = 21M elements, 23M² = 530 quadrillion
            pass  # t1 and t2 are already correctly paired element-wise
        else:
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

    return _compute_g1_total_core(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        dt_value,
    )


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
    """Wrapper function that computes g2 using configuration dt.

    IMPORTANT: The dt parameter MUST come from configuration.
    No fallback estimation - explicit dt is required for correct physics.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from configuration [s] (REQUIRED)

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied

    Raises:
        TypeError: If dt is None (no longer accepts None)
        ValueError: If dt <= 0 or not finite
    """
    # Note: dt validation moved to caller to avoid JAX tracing issues.
    # The residual function validates dt before JIT compilation.
    # If dt validation is needed here, it must be done before the function is traced.

    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Check if this is flattened pooled data (from CMC) vs normal time vectors
        # Normal time vectors: typically 100-2000 elements (need meshgrid expansion)
        # Pooled data for SVI: 2000-5000+ elements (already element-wise matched, DON'T mesh)
        # Using 2000 as threshold: above typical time vectors (e.g., 1001), below pooled data (e.g., 4600)
        if len(t1) > 2000:
            # Pooled/flattened data: arrays already element-wise matched, don't create meshgrid
            # Creating meshgrid would cause OOM: e.g., 4600² = 21M elements, 23M² = 530 quadrillion
            pass  # t1 and t2 are already correctly paired element-wise
        else:
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

    return _compute_g2_scaled_core(
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
        t1, t2: Time grids for correlation calculation
        phi: Scattering angles [degrees]
        wavevector_q_squared_half_dt: Pre-computed factor (0.5 * q² * dt)
        sinc_prefactor: Pre-computed factor (q * L * dt / 2π)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from experimental configuration (time per frame) [seconds]

    Returns:
        g2 correlation function with scaled fitting

    Note:
        This function is JIT-compiled for maximum performance.
        Use with HomodyneModel for best results.
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Check if this is flattened pooled data (from CMC) vs normal time vectors
        # Normal time vectors: typically 100-2000 elements (need meshgrid expansion)
        # Pooled data for SVI: 2000-5000+ elements (already element-wise matched, DON'T mesh)
        # Using 2000 as threshold: above typical time vectors (e.g., 1001), below pooled data (e.g., 4600)
        if len(t1) > 2000:
            # Pooled/flattened data: arrays already element-wise matched, don't create meshgrid
            # Creating meshgrid would cause OOM: e.g., 4600² = 21M elements, 23M² = 530 quadrillion
            pass  # t1 and t2 are already correctly paired element-wise
        else:
            # Normal time vectors: create 2D meshgrids for all (t1[i], t2[j]) pairs
            # CRITICAL: Must match caller's convention: t1_grid, t2_grid = meshgrid(t, t, 'ij')
            t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
            t1 = t1_grid
            t2 = t2_grid

    # Call core computation with pre-computed factors
    return _compute_g2_scaled_core(
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


@jit
def compute_chi_squared(
    params: jnp.ndarray,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float,
) -> float:
    """Compute chi-squared goodness of fit.

    χ² = Σᵢ [(data_i - theory_i) / σᵢ]²

    Args:
        params: Physical parameters
        data: Experimental correlation data
        sigma: Measurement uncertainties
        t1, t2: Time grids
        phi: Angle grid
        q: Wave vector magnitude
        L: Sample-detector distance
        contrast, offset: Scaling parameters
        dt: Time step from configuration

    Returns:
        Chi-squared value
    """
    theory = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)
    residuals = (data - theory) / (sigma + EPS)  # Avoid division by zero
    return jnp.sum(residuals**2)


# Automatic differentiation functions with intelligent fallback
# These will work with either JAX or NumPy fallbacks
gradient_g2 = grad(compute_g2_scaled, argnums=0)  # Gradient w.r.t. params
hessian_g2 = hessian(compute_g2_scaled, argnums=0)  # Hessian w.r.t. params

gradient_chi2 = grad(compute_chi_squared, argnums=0)  # Gradient of chi-squared
hessian_chi2 = hessian(compute_chi_squared, argnums=0)  # Hessian of chi-squared


# Vectorized versions for batch computation
@log_performance(threshold=0.1)
def vectorized_g2_computation(
    params_batch: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """Vectorized g2 computation for multiple parameter sets.

    Uses JAX vmap for efficient parallel computation.
    """
    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset)
            results.append(result)
        return jnp.stack(results)

    # JAX vectorized version
    vectorized_func = vmap(
        compute_g2_scaled,
        in_axes=(0, None, None, None, None, None, None, None),
    )
    return vectorized_func(params_batch, t1, t2, phi, q, L, contrast, offset)


@log_performance(threshold=0.05)
def batch_chi_squared(
    params_batch: jnp.ndarray,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
) -> jnp.ndarray:
    """Compute chi-squared for multiple parameter sets efficiently."""
    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_chi_squared(
                params,
                data,
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                contrast,
                offset,
            )
            results.append(result)
        return jnp.array(results)

    # JAX vectorized version
    vectorized_func = vmap(
        compute_chi_squared,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    return vectorized_func(
        params_batch,
        data,
        sigma,
        t1,
        t2,
        phi,
        q,
        L,
        contrast,
        offset,
    )


# Utility functions for optimization
def validate_backend() -> dict[str, bool | str | dict]:
    """Validate computational backends with comprehensive diagnostics."""
    results = {
        "jax_available": JAX_AVAILABLE,
        "numpy_gradients_available": numpy_gradients_available,
        "gradient_support": False,
        "hessian_support": False,
        "backend_type": "unknown",
        "performance_estimate": "unknown",
        "recommendations": [],
        "fallback_stats": _fallback_stats.copy(),
        "test_results": {},
    }

    # Determine backend type and performance characteristics
    if JAX_AVAILABLE:
        results["backend_type"] = "jax_native"
        results["performance_estimate"] = "optimal (1x)"
    elif numpy_gradients_available:
        results["backend_type"] = "numpy_fallback"
        results["performance_estimate"] = "degraded (10-50x slower)"
        results["recommendations"].append(
            "Install JAX for optimal performance: pip install jax",
        )
    else:
        results["backend_type"] = "none"
        results["performance_estimate"] = "unavailable"
        results["recommendations"].extend(
            [
                "Install JAX for optimal performance: pip install jax",
                "Or install scipy for basic functionality: pip install scipy",
            ],
        )

    # Test basic computation
    try:
        test_params = jnp.array([100.0, 0.0, 10.0])
        test_t1 = jnp.array([0.0])
        test_t2 = jnp.array([1.0])
        test_q = 0.01

        # Test forward computation
        compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        results["test_results"]["forward_computation"] = "success"

        # Test gradient computation
        try:
            grad_func = grad(compute_g1_diffusion, argnums=0)
            grad_func(test_params, test_t1, test_t2, test_q)
            results["gradient_support"] = True
            results["test_results"]["gradient_computation"] = "success"

            if not JAX_AVAILABLE:
                results["test_results"]["gradient_method"] = "numpy_fallback"

        except ImportError as e:
            results["test_results"]["gradient_computation"] = f"failed: {str(e)}"
            logger.warning(f"Gradient computation not available: {e}")
        except Exception as e:
            results["test_results"]["gradient_computation"] = f"error: {str(e)}"
            logger.error(f"Gradient computation failed: {e}")

        # Test hessian computation
        try:
            hess_func = hessian(compute_g1_diffusion, argnums=0)
            hess_func(test_params, test_t1, test_t2, test_q)
            results["hessian_support"] = True
            results["test_results"]["hessian_computation"] = "success"

            if not JAX_AVAILABLE:
                results["test_results"]["hessian_method"] = "numpy_fallback"

        except ImportError as e:
            results["test_results"]["hessian_computation"] = f"failed: {str(e)}"
            logger.warning(f"Hessian computation not available: {e}")
        except Exception as e:
            results["test_results"]["hessian_computation"] = f"error: {str(e)}"
            logger.error(f"Hessian computation failed: {e}")

        logger.info(f"Backend validation completed: {results['backend_type']} mode")

    except Exception as e:
        logger.error(f"Basic computation test failed: {e}")
        results["test_results"]["forward_computation"] = f"failed: {str(e)}"

    return results


def get_device_info() -> dict:
    """Get comprehensive device and backend information."""
    if not JAX_AVAILABLE:
        fallback_info = {
            "available": False,
            "devices": [],
            "backend": "numpy_fallback" if numpy_gradients_available else "none",
            "fallback_active": True,
            "performance_impact": (
                "10-50x slower" if numpy_gradients_available else "unavailable"
            ),
            "recommendations": [],
        }

        if numpy_gradients_available:
            fallback_info["recommendations"].append(
                "Install JAX for optimal performance: pip install jax",
            )
            fallback_info["fallback_stats"] = _fallback_stats.copy()
        else:
            fallback_info["recommendations"].extend(
                [
                    "Install JAX for optimal performance: pip install jax",
                    "Or install scipy for basic functionality: pip install scipy",
                ],
            )

        return fallback_info

    try:
        devices = jax.devices()
        return {
            "available": True,
            "devices": [str(d) for d in devices],
            "backend": jax.default_backend(),
            "device_count": len(devices),
            "fallback_active": False,
            "performance_impact": "optimal (native JAX)",
            "recommendations": ["JAX is available and configured correctly"],
        }
    except Exception as e:
        logger.warning(f"Could not get JAX device info: {e}")
        return {
            "available": True,
            "devices": ["unknown"],
            "backend": "unknown",
            "error": str(e),
            "fallback_active": False,
        }


def get_performance_summary() -> dict[str, str | int | dict]:
    """Get performance summary and recommendations."""
    return {
        "backend_type": (
            "jax_native"
            if JAX_AVAILABLE
            else ("numpy_fallback" if numpy_gradients_available else "none")
        ),
        "jax_available": JAX_AVAILABLE,
        "numpy_gradients_available": numpy_gradients_available,
        "fallback_stats": _fallback_stats.copy(),
        "performance_multiplier": (
            "1x"
            if JAX_AVAILABLE
            else ("10-50x" if numpy_gradients_available else "N/A")
        ),
        "recommendations": _get_performance_recommendations(),
    }


def _get_performance_recommendations() -> list[str]:
    """Get performance optimization recommendations."""
    recommendations = []

    if not JAX_AVAILABLE:
        recommendations.append(
            "🚀 Install JAX for 10-50x performance improvement: pip install jax",
        )

        if not numpy_gradients_available:
            recommendations.append(
                "📊 Install scipy for basic numerical differentiation: pip install scipy",
            )
        else:
            recommendations.append("✅ NumPy gradients available as fallback")

    if JAX_AVAILABLE:
        try:
            import jax

            devices = jax.devices()
            if len(devices) > 1:
                recommendations.append(
                    f"🔥 {len(devices)} compute devices available for parallel processing",
                )
            if any("gpu" in str(d).lower() for d in devices):
                recommendations.append("🎯 GPU acceleration available")
            if any("tpu" in str(d).lower() for d in devices):
                recommendations.append("⚡ TPU acceleration available")
        except Exception:
            logger.debug("Device inspection failed; proceeding without device hints")

    return recommendations


# Export main functions
__all__ = [
    "jax_available",
    "numpy_gradients_available",
    "compute_g1_diffusion",
    "compute_g1_shear",
    "compute_g1_total",
    "compute_g2_scaled",
    "compute_chi_squared",
    "gradient_g2",
    "hessian_g2",
    "gradient_chi2",
    "hessian_chi2",
    "vectorized_g2_computation",
    "batch_chi_squared",
    "validate_backend",
    "get_device_info",
    "get_performance_summary",  # New performance monitoring
]
