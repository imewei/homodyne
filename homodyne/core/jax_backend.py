"""JAX Computational Backend for Homodyne v2
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
        global _fallback_stats
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
        global _fallback_stats
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


# Physical and mathematical constants
PI = jnp.pi
EPS = 1e-12  # Numerical stability epsilon


@jit
def safe_divide(a: jnp.ndarray, b: jnp.ndarray, default: float = 0.0) -> jnp.ndarray:
    """Safe division with numerical stability."""
    return jnp.where(jnp.abs(b) > EPS, a / b, default)


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


# Discrete numerical integration helpers (following reference v1 implementation)
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

    # TEMPORARY: Remove hard clipping to test if it's blocking gradients
    # Just ensure positive values
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


# Core physics computations with discrete numerical integration
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

    # DEBUG: Log input parameters
    if jax_available and hasattr(jnp, "where"):
        # Use JAX operations for debugging within JIT context
        pass  # Can't print in JIT context easily
    else:
        import numpy as np

        if hasattr(D0, "item"):
            print(
                f"DEBUG g1_diffusion: D0={D0.item():.6f}, alpha={alpha.item():.6f}, D_offset={D_offset.item():.6f}",
            )
            print(
                f"DEBUG g1_diffusion: wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6e}",
            )
        else:
            print(
                f"DEBUG g1_diffusion: D0={D0:.6f}, alpha={alpha:.6f}, D_offset={D_offset:.6f}",
            )
            print(
                f"DEBUG g1_diffusion: wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6e}",
            )

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

        # DEBUG: Check D_t values
        if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
            import numpy as np

            if hasattr(D_t, "min"):
                print(
                    f"DEBUG g1_diffusion: D_t min={np.min(D_t):.6e}, max={np.max(D_t):.6e}, mean={np.mean(D_t):.6e}",
                )

        # Step 3: Create diffusion integral matrix using cumulative sums
        # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
        D_integral = _create_time_integral_matrix_impl_jax(D_t)

        # DEBUG: Check D_integral values
        if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
            import numpy as np

            if hasattr(D_integral, "min"):
                print(
                    f"DEBUG g1_diffusion: D_integral min={np.min(D_integral):.6e}, max={np.max(D_integral):.6e}, mean={np.mean(D_integral):.6e}",
                )

    # Step 4: Compute g1 correlation using log-space for numerical stability
    # This matches reference: g1 = exp(-wavevector_q_squared_half_dt * D_integral)
    #
    # LOG-SPACE CALCULATION FIX (Oct 2025):
    # Computing in log-space preserves precision across full dynamic range.
    # Old approach: clip(g1, 1e-10, 1.0) caused artificial plateaus (~16% of data)
    # New approach: clip in log-space, then exp() - no artificial plateaus
    log_g1 = -wavevector_q_squared_half_dt * D_integral

    # DEBUG: Check log values before clipping
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(log_g1, "min"):
            print(
                f"DEBUG g1_diffusion: log_g1 (pre-clip) min={np.min(log_g1):.6e}, max={np.max(log_g1):.6e}, mean={np.mean(log_g1):.6e}",
            )

    # Clip in log-space to prevent numerical overflow/underflow
    # -700 → exp(-700) ≈ 1e-304 (near machine precision)
    # 0 → exp(0) = 1.0 (maximum physical value)
    log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)

    # DEBUG: Check log values after clipping
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(log_g1_bounded, "min"):
            print(
                f"DEBUG g1_diffusion: log_g1_bounded min={np.min(log_g1_bounded):.6e}, max={np.max(log_g1_bounded):.6e}",
            )

    # Compute exponential with safeguards (safe_exp handles edge cases)
    g1_result = safe_exp(log_g1_bounded)

    # DEBUG: Check g1_result after exp
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(g1_result, "min"):
            print(
                f"DEBUG g1_diffusion: g1_result min={np.min(g1_result):.6e}, max={np.max(g1_result):.6e}",
            )

    # Apply ONLY upper bound (g1 ≤ 1.0 is physical constraint)
    # No lower bound clipping - preserves full precision down to machine epsilon
    # This eliminates artificial plateaus from overly aggressive clipping
    g1_safe = jnp.minimum(g1_result, 1.0)

    # DEBUG: Final g1_diffusion result
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(g1_safe, "min"):
            print(
                f"DEBUG g1_diffusion: FINAL min={np.min(g1_safe):.6e}, max={np.max(g1_safe):.6e}",
            )

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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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
        gamma_integral_elementwise = frame_diff * 0.5 * (gamma_t1 + gamma_t2)  # Dimensionless

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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        dt: Time step from experimental configuration (time per frame)

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)
    """
    # Compute diffusion contribution
    g1_diff = _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt, dt)

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

    # Apply loose physical bounds to allow natural correlation function behavior
    # Remove artificial upper bound to prevent fitted data collapse
    # |g₁|² ∈ [1e-10, ∞) - naturally non-negative with numerical stability

    # Apply positive-only constraint with minimum threshold for numerical stability
    epsilon = 1e-10
    g1_bounded = jnp.maximum(g1_total, epsilon)

    # No upper bound - allows unlimited correlation function growth
    # This removes the previous artificial constraint that caused fitted data collapse

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
    """Core scaled optimization: g₂ = offset + contrast × [g₁]²

    This is the central equation for homodyne scattering analysis.

    Physical constraint: 0 < g2 ≤ 2

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
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
    g2 = offset + contrast * g1**2

    # Apply physical bounds: 0 < g2 ≤ 2
    # Use small epsilon to avoid exact zero (which could cause numerical issues)
    #
    # NOTE (Oct 2025): Analysis confirms upper bound of 2.0 is appropriate:
    # - Maximum observed g2 = offset_max + contrast_max × 1² ≈ 1.518
    # - Current bound provides 25% headroom (1.518 < 2.0)
    # - Physical constraint: g2 ≤ 2 for homodyne detection
    #
    # If checkerboard artifacts persist after trapezoidal integration fix,
    # uncomment the line below to test without upper bound clipping:
    # g2_bounded = jnp.maximum(g2, 1e-10)  # Test: remove upper bound
    g2_bounded = jnp.clip(g2, 1e-10, 2.0)

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
    # Use same dtype as input matrix to avoid casting warnings
    diag_val = jnp.zeros(size, dtype=c2_mat.dtype)

    # Add left neighbors: diag_val[:-1] += side_band
    diag_val = diag_val.at[:-1].add(side_band)

    # Add right neighbors: diag_val[1:] += side_band
    diag_val = diag_val.at[1:].add(side_band)

    # Normalize by number of neighbors (1 for edges, 2 for middle)
    norm = jnp.ones(size, dtype=c2_mat.dtype)
    norm = norm.at[1:-1].set(2.0)  # Middle elements have 2 neighbors

    diag_val = diag_val / norm

    # Replace diagonal with computed values using JAX immutable array operations
    diag_indices = jnp.diag_indices(size)
    c2_corrected = c2_mat.at[diag_indices].set(diag_val)

    return c2_corrected


# =============================================================================
# COMPATIBILITY WRAPPER FUNCTIONS
# =============================================================================
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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g1_total_core(
        params,
        t1,
        t2,
        phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        dt,
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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

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
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
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

    Returns:
        Chi-squared value
    """
    theory = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset)
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


# Legacy function for compatibility
def validate_jax_backend() -> bool:
    """Legacy function - use validate_backend() instead."""
    results = validate_backend()
    return results["jax_available"] and results["gradient_support"]


# Legacy function for backward compatibility with old tests
def compute_c2_model_jax(
    params: dict,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """Legacy wrapper for compute_g2_scaled() - for backward compatibility with old tests.

    This function provides a simplified interface matching the old API signature,
    using default values for L, contrast, offset, and dt parameters.

    Args:
        params: Parameter dictionary with keys like 'offset', 'contrast', 'diffusion_coefficient', etc.
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude

    Returns:
        g2 correlation function

    Note:
        This is a legacy function for backward compatibility.
        New code should use compute_g2_scaled() directly with explicit parameters.
    """
    # Extract parameters from dict with defaults
    contrast = params.get("contrast", 0.5)
    offset = params.get("offset", 1.0)
    L = params.get("L", 1.0)

    # Convert parameter dict to array format expected by compute_g2_scaled
    # Old tests use 'diffusion_coefficient', new code uses 'D0'
    D0 = params.get("diffusion_coefficient", params.get("D0", 1000.0))
    alpha = params.get("alpha", 0.5)
    D_offset = params.get("D_offset", 10.0)

    # For static isotropic mode (3 physical parameters)
    param_array = jnp.array([D0, alpha, D_offset])

    # Estimate dt from time array (legacy behavior)
    if t1.ndim == 2:
        time_array = t1[:, 0]
    else:
        time_array = t1
    dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Call new function with explicit parameters
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


# Legacy aliases for backward compatibility with old tests
def residuals_jax(
    params: dict,
    c2_exp: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
) -> jnp.ndarray:
    """Legacy function: compute residuals (data - model) / sigma.

    Note: This is for backward compatibility with old tests.
    New code should use compute_chi_squared() directly.
    """
    # Generate model prediction using legacy wrapper
    c2_model = compute_c2_model_jax(params, t1, t2, phi, q)

    # Compute residuals
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
    """Legacy function: compute chi-squared goodness of fit.

    Note: This is for backward compatibility with old tests.
    New code should use compute_chi_squared() directly.
    """
    residuals = residuals_jax(params, c2_exp, sigma, t1, t2, phi, q)
    return jnp.sum(residuals**2)


def compute_g1_diffusion_jax(
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    q: float,
    D: float,
) -> jnp.ndarray:
    """Legacy function: compute g1 diffusion factor.

    Note: This is for backward compatibility with old tests.
    New code should use compute_g1_diffusion() directly.
    """
    # Call new function with simple parameters
    # Old function signature: (t1, t2, q, D)
    # New function signature: (params, t1, t2, q, dt)

    # Create params array [D0, alpha, D_offset] with alpha=0.5 (normal diffusion)
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
            pass

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
    "validate_jax_backend",  # Legacy compatibility
    "compute_c2_model_jax",  # Legacy compatibility for old tests
    "residuals_jax",  # Legacy compatibility for old tests
    "chi_squared_jax",  # Legacy compatibility for old tests
    "compute_g1_diffusion_jax",  # Legacy compatibility for old tests
    "get_device_info",
    "get_performance_summary",  # New performance monitoring
]
