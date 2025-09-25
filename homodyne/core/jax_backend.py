"""
JAX Computational Backend for Homodyne v2
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
    from jax import grad, hessian, jit, random, vmap
    from jax.scipy import special

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
                func.__name__ if hasattr(func, "__name__") else "function"
            )

    def hessian(func, argnums=0):
        """Intelligent fallback Hessian function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_hessian_fallback(func, argnums)
        else:
            return _create_no_hessian_fallback(
                func.__name__ if hasattr(func, "__name__") else "function"
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
    """
    JAX-safe length function that handles scalars, arrays, and JAX objects.

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
            "Install JAX for optimal performance: pip install jax"
        )
    else:
        logger.error(
            "Neither JAX nor NumPy gradients available.\n"
            "Install NumPy gradients: pip install scipy\n"
            "Or install JAX for optimal performance: pip install jax"
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
                f"Install JAX for optimal performance."
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
                f"Install JAX for optimal performance."
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
    """Safe sinc function with numerical stability at x=0."""
    return jnp.where(jnp.abs(x) > EPS, jnp.sin(PI * x) / (PI * x), 1.0)


# Discrete numerical integration helpers (following reference v1 implementation)
@jit
def _calculate_diffusion_coefficient_impl_jax(
    time_array: jnp.ndarray, D0: float, alpha: float, D_offset: float
) -> jnp.ndarray:
    """
    Calculate time-dependent diffusion coefficient using discrete evaluation.

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
    # Robust calculation with bounds checking to prevent numerical overflow
    # Apply bounds on the exponential term to prevent extreme values
    power_term = jnp.clip(time_array**alpha, 1e-20, 1e5)  # Prevent extreme power values
    D_t = D0 * power_term + D_offset

    # Apply physical bounds: D(t) should be positive and not too large
    # Hard bounds enforce 1.0 < D_t < 1e6 for numerical stability (updated to match DIFFUSION_MAX)
    D_t_bounded = jnp.clip(D_t, 1.0, 1e6)

    return D_t_bounded


@jit
def _calculate_shear_rate_impl_jax(
    time_array: jnp.ndarray, gamma_dot_0: float, beta: float, gamma_dot_offset: float
) -> jnp.ndarray:
    """
    Calculate time-dependent shear rate using discrete evaluation.

    Follows reference v1 implementation: γ̇_t[i] = γ̇₀ * (time_array[i] ** β) + γ̇_offset

    Args:
        time_array: Array of time points
        gamma_dot_0: Shear rate amplitude
        beta: Shear rate exponent
        gamma_dot_offset: Baseline shear rate offset

    Returns:
        γ̇(t) evaluated at each time point
    """
    gamma_t = gamma_dot_0 * (time_array**beta) + gamma_dot_offset
    # Apply hard bounds: 1e-5 < gamma_t < 1 for numerical stability and physical constraints
    return jnp.clip(gamma_t, 1e-5, 1.0)


@jit
def _create_time_integral_matrix_impl_jax(
    time_dependent_array: jnp.ndarray,
) -> jnp.ndarray:
    """
    Create time integral matrix using discrete numerical integration.

    Follows reference v1 implementation algorithm:
    1. Calculate cumulative sum: cumsum[i] = ∫₀^tᵢ f(t') dt' ≈ dt * sum(f[0:i])
    2. Compute difference matrix: matrix[i,j] = |cumsum[i] - cumsum[j]|

    This gives: matrix[i,j] ≈ ∫₀^|tᵢ-tⱼ| f(t') dt'

    Args:
        time_dependent_array: f(t) evaluated at discrete time points

    Returns:
        Time integral matrix for correlation calculations
    """
    # Handle scalar input by converting to array
    time_dependent_array = jnp.atleast_1d(time_dependent_array)
    safe_len(time_dependent_array)

    # Step 1: Discrete cumulative integration
    # This approximates ∫₀^tᵢ f(t') dt' using cumulative sum
    cumsum = jnp.cumsum(time_dependent_array)

    # Step 2: Create difference matrix
    # matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ ∫₀^|tᵢ-tⱼ| f(t') dt'
    cumsum_i = cumsum[:, None]  # Shape: (n, 1)
    cumsum_j = cumsum[None, :]  # Shape: (1, n)
    matrix = jnp.abs(cumsum_i - cumsum_j)  # Shape: (n, n)

    return matrix


# Core physics computations with discrete numerical integration
@jit
def _compute_g1_diffusion_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    wavevector_q_squared_half_dt: float,
) -> jnp.ndarray:
    """
    Compute diffusion contribution to g1 using reference implementation approach.

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
                f"DEBUG g1_diffusion: D0={D0.item():.6f}, alpha={alpha.item():.6f}, D_offset={D_offset.item():.6f}"
            )
            print(
                f"DEBUG g1_diffusion: wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6e}"
            )
        else:
            print(
                f"DEBUG g1_diffusion: D0={D0:.6f}, alpha={alpha:.6f}, D_offset={D_offset:.6f}"
            )
            print(
                f"DEBUG g1_diffusion: wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6e}"
            )

    # Step 1: Extract time array (t1 and t2 should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid: t1 varies along columns, so extract first row
        time_array = t1[0, :]  # Extract first row for unique t1 values
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
                f"DEBUG g1_diffusion: D_t min={np.min(D_t):.6e}, max={np.max(D_t):.6e}, mean={np.mean(D_t):.6e}"
            )

    # Step 3: Create diffusion integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫D(t)dt from i to j|
    D_integral = _create_time_integral_matrix_impl_jax(D_t)

    # DEBUG: Check D_integral values
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(D_integral, "min"):
            print(
                f"DEBUG g1_diffusion: D_integral min={np.min(D_integral):.6e}, max={np.max(D_integral):.6e}, mean={np.mean(D_integral):.6e}"
            )

    # Step 4: Compute g1 correlation using pre-computed factor
    # This matches reference: g1 = exp(-wavevector_q_squared_half_dt * D_integral)
    exponent = -wavevector_q_squared_half_dt * D_integral

    # DEBUG: Check exponent values before bounding
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(exponent, "min"):
            print(
                f"DEBUG g1_diffusion: exponent (pre-clip) min={np.min(exponent):.6e}, max={np.max(exponent):.6e}, mean={np.mean(exponent):.6e}"
            )

    # Apply bounds to prevent numerical issues
    exponent_bounded = jnp.clip(exponent, -700.0, 0.0)  # Physical bounds

    # DEBUG: Check exponent values after bounding
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(exponent_bounded, "min"):
            print(
                f"DEBUG g1_diffusion: exponent_bounded min={np.min(exponent_bounded):.6e}, max={np.max(exponent_bounded):.6e}"
            )

    # Compute exponential with safeguards
    g1_result = safe_exp(exponent_bounded)

    # DEBUG: Check g1_result before final clipping
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(g1_result, "min"):
            print(
                f"DEBUG g1_diffusion: g1_result (pre-clip) min={np.min(g1_result):.6e}, max={np.max(g1_result):.6e}"
            )

    # Apply physical bounds: 0 < g1 ≤ 1
    g1_safe = jnp.clip(g1_result, 1e-10, 1.0)

    # DEBUG: Final g1_diffusion result
    if not jax_available or not hasattr(jnp, "where"):  # Outside JIT
        import numpy as np

        if hasattr(g1_safe, "min"):
            print(
                f"DEBUG g1_diffusion: FINAL min={np.min(g1_safe):.6e}, max={np.max(g1_safe):.6e}"
            )

    return g1_safe


@jit
def _compute_g1_shear_core(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    sinc_prefactor: float,
) -> jnp.ndarray:
    """
    Compute shear contribution to g1 using reference implementation approach.

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

    Returns:
        Shear contribution to g1 correlation function (sinc² values)
    """
    if safe_len(params) < 7:  # Static mode - no shear
        # Return ones for all phi angles and time combinations
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

    # Step 1: Extract time array (t1 and t2 should be identical)
    # Handle all dimensionality cases: 0D (scalar), 1D arrays, and 2D meshgrids
    if t1.ndim == 2:
        # For meshgrid: t1 varies along columns, so extract first row
        time_array = t1[0, :]  # Extract first row for unique t1 values
    elif t1.ndim == 0:
        # Handle 0-dimensional (scalar) input
        time_array = jnp.atleast_1d(t1)
    else:
        # Handle 1D and other cases
        time_array = jnp.atleast_1d(t1)

    # Step 2: Calculate γ̇(t) at each time point
    gamma_t = _calculate_shear_rate_impl_jax(
        time_array, gamma_dot_0, beta, gamma_dot_offset
    )

    # Step 3: Create shear integral matrix using cumulative sums
    # This gives matrix[i,j] = |cumsum[i] - cumsum[j]| ≈ |∫γ̇(t)dt from i to j|
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

    # Step 4: Compute sinc² for each phi angle using pre-computed factor (vectorized)
    phi_array = jnp.atleast_1d(phi)
    n_phi = safe_len(phi_array)
    n_times = safe_len(time_array)

    # Debug: Log shapes for troubleshooting broadcasting issues
    import os

    if os.environ.get("HOMODYNE_DEBUG_SHAPES") == "1":
        print("DEBUG _compute_g1_shear_core shapes:")
        print(f"  phi_array.shape: {phi_array.shape}")
        print(f"  gamma_integral.shape: {gamma_integral.shape}")
        print(f"  n_phi: {n_phi}, n_times: {n_times}")

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
            f"gamma_integral should be 2D, got shape {gamma_integral.shape}"
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
            f"Original error: {e}"
        )

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
) -> jnp.ndarray:
    """
    Compute total g1 correlation function as product of diffusion and shear.

    Following reference implementation:
    g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]

    Physical constraint: 0 < g₁(t) ≤ 1

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        wavevector_q_squared_half_dt: Pre-computed factor 0.5 * q² * dt from configuration
        sinc_prefactor: Pre-computed factor 0.5/π * q * L * dt from configuration

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)
    """
    # Compute diffusion contribution: shape (n_times, n_times)
    g1_diff = _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt)

    # Compute shear contribution: shape (n_phi, n_times, n_times)
    g1_shear = _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor)

    # Broadcast diffusion term to match shear dimensions
    # g1_diff needs to be broadcast from (n_times, n_times) to (n_phi, n_times, n_times)
    # Use the shape of g1_shear to determine n_phi (more reliable than parsing phi directly)
    n_phi = g1_shear.shape[0]
    g1_diff_broadcasted = jnp.broadcast_to(
        g1_diff[None, :, :], (n_phi, g1_diff.shape[0], g1_diff.shape[1])
    )

    # Multiply: g₁_total[phi, i, j] = g₁_diffusion[i, j] × g₁_shear[phi, i, j]
    try:
        g1_total = g1_diff_broadcasted * g1_shear
    except Exception as e:
        # Enhanced error message for debugging
        raise ValueError(
            f"Broadcasting error in _compute_g1_total_core: "
            f"g1_diff_broadcasted.shape={g1_diff_broadcasted.shape}, g1_shear.shape={g1_shear.shape}. "
            f"Original error: {e}"
        )

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
) -> jnp.ndarray:
    """
    Core scaled optimization: g₂ = offset + contrast × [g₁]²

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

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    g1 = _compute_g1_total_core(
        params, t1, t2, phi, wavevector_q_squared_half_dt, sinc_prefactor
    )
    g2 = offset + contrast * g1**2

    # Apply physical bounds: 0 < g2 ≤ 2
    # Use small epsilon to avoid exact zero (which could cause numerical issues)
    g2_bounded = jnp.clip(g2, 1e-10, 2.0)

    return g2_bounded


# =============================================================================
# COMPATIBILITY WRAPPER FUNCTIONS
# =============================================================================
# These maintain the old API for backward compatibility while using correct
# configuration values internally


def compute_g1_diffusion(
    params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, q: float, dt: float = None
) -> jnp.ndarray:
    """
    Wrapper function that computes g1 diffusion using configuration dt.

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
        # Create 2D meshgrids from 1D arrays
        t2_grid, t1_grid = jnp.meshgrid(t2, t1, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factor using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

    return _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt)


def compute_g1_shear(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float = None,
) -> jnp.ndarray:
    """
    Wrapper function that computes g1 shear using configuration dt.

    IMPORTANT: The dt parameter should come from configuration, not be computed.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration (REQUIRED for correct physics)

    Returns:
        Shear contribution to g1 correlation function (sinc² values)
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Create 2D meshgrids from 1D arrays
        t2_grid, t1_grid = jnp.meshgrid(t2, t1, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factor using configuration dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor)


def compute_g1_total(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    dt: float = None,
) -> jnp.ndarray:
    """
    Wrapper function that computes total g1 using configuration dt.

    IMPORTANT: The dt parameter should come from configuration, not be computed.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time grids (should be identical: t1 = t2 = t)
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        dt: Time step from configuration (REQUIRED for correct physics)

    Returns:
        Total g1 correlation function with shape (n_phi, n_times, n_times)
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Create 2D meshgrids from 1D arrays
        t2_grid, t1_grid = jnp.meshgrid(t2, t1, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factors using configuration dt
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    # DEBUG: Log input parameters before calling JIT function
    import os

    if os.environ.get("HOMODYNE_DEBUG_G1") == "1":
        import numpy as np

        params_np = np.asarray(params) if hasattr(params, "__array__") else params
        print(f"DEBUG compute_g1_total: params shape={params_np.shape}")
        print(
            f"DEBUG compute_g1_total: D0={params_np[0]:.6e}, alpha={params_np[1]:.6f}, D_offset={params_np[2]:.6e}"
        )
        print(
            f"DEBUG compute_g1_total: gamma_dot_0={params_np[3]:.6e}, beta={params_np[4]:.6f}, gamma_dot_offset={params_np[5]:.6e}"
        )
        print(f"DEBUG compute_g1_total: phi0={params_np[6]:.6f}")
        print(f"DEBUG compute_g1_total: q={q:.6e}, L={L:.6e}, dt={dt:.6e}")
        print(
            f"DEBUG compute_g1_total: wavevector_q_squared_half_dt={wavevector_q_squared_half_dt:.6e}"
        )
        print(f"DEBUG compute_g1_total: sinc_prefactor={sinc_prefactor:.6e}")

    result = _compute_g1_total_core(
        params, t1, t2, phi, wavevector_q_squared_half_dt, sinc_prefactor
    )

    # DEBUG: Check result
    if os.environ.get("HOMODYNE_DEBUG_G1") == "1":
        import numpy as np

        result_np = np.asarray(result) if hasattr(result, "__array__") else result
        print(f"DEBUG compute_g1_total: result shape={result_np.shape}")
        print(
            f"DEBUG compute_g1_total: result min={np.min(result_np):.6e}, max={np.max(result_np):.6e}, mean={np.mean(result_np):.6e}"
        )

    return result


def compute_g2_scaled(
    params: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    contrast: float,
    offset: float,
    dt: float = None,
) -> jnp.ndarray:
    """
    Wrapper function that computes g2 using configuration dt.

    IMPORTANT: The dt parameter should come from configuration, not be computed.

    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance (stator_rotor_gap)
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        dt: Time step from configuration (REQUIRED for correct physics)

    Returns:
        g2 correlation function with scaled fitting and physical bounds applied
    """
    # Handle 1D time arrays by creating meshgrids
    if t1.ndim == 1 and t2.ndim == 1:
        # Create 2D meshgrids from 1D arrays
        t2_grid, t1_grid = jnp.meshgrid(t2, t1, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    if dt is None:
        # FALLBACK: Estimate from time array (NOT RECOMMENDED)
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute the pre-computed factors using configuration dt
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
    """
    Compute chi-squared goodness of fit.

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
    """
    Vectorized g2 computation for multiple parameter sets.

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
        compute_g2_scaled, in_axes=(0, None, None, None, None, None, None, None)
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
    """
    Compute chi-squared for multiple parameter sets efficiently.
    """
    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_chi_squared(
                params, data, sigma, t1, t2, phi, q, L, contrast, offset
            )
            results.append(result)
        return jnp.array(results)

    # JAX vectorized version
    vectorized_func = vmap(
        compute_chi_squared,
        in_axes=(0, None, None, None, None, None, None, None, None, None),
    )
    return vectorized_func(
        params_batch, data, sigma, t1, t2, phi, q, L, contrast, offset
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
            "Install JAX for optimal performance: pip install jax"
        )
    else:
        results["backend_type"] = "none"
        results["performance_estimate"] = "unavailable"
        results["recommendations"].extend(
            [
                "Install JAX for optimal performance: pip install jax",
                "Or install scipy for basic functionality: pip install scipy",
            ]
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
                "Install JAX for optimal performance: pip install jax"
            )
            fallback_info["fallback_stats"] = _fallback_stats.copy()
        else:
            fallback_info["recommendations"].extend(
                [
                    "Install JAX for optimal performance: pip install jax",
                    "Or install scipy for basic functionality: pip install scipy",
                ]
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
            "🚀 Install JAX for 10-50x performance improvement: pip install jax"
        )

        if not numpy_gradients_available:
            recommendations.append(
                "📊 Install scipy for basic numerical differentiation: pip install scipy"
            )
        else:
            recommendations.append("✅ NumPy gradients available as fallback")

    if JAX_AVAILABLE:
        try:
            import jax

            devices = jax.devices()
            if len(devices) > 1:
                recommendations.append(
                    f"🔥 {len(devices)} compute devices available for parallel processing"
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
    "get_device_info",
    "get_performance_summary",  # New performance monitoring
]
