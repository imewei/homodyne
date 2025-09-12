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
    from jax import jit, vmap, grad, hessian, random
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
            if hasattr(inputs, '__iter__') and not isinstance(inputs, str):
                return [func(inp, *vargs, **vkwargs) for inp in inputs]
            return func(inputs, *vargs, **vkwargs)
        return vectorized_func
    
    def grad(func, argnums=0):
        """Intelligent fallback gradient function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_gradient_fallback(func, argnums)
        else:
            return _create_no_gradient_fallback(func.__name__ if hasattr(func, '__name__') else 'function')
    
    def hessian(func, argnums=0):
        """Intelligent fallback Hessian function with performance warnings."""
        if NUMPY_GRADIENTS_AVAILABLE:
            return _create_hessian_fallback(func, argnums)
        else:
            return _create_no_hessian_fallback(func.__name__ if hasattr(func, '__name__') else 'function')

import numpy as np
import warnings
from typing import Tuple, Union, Optional, Dict, Callable, List
from functools import wraps
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# Performance tracking for fallback warnings
_performance_warned = set()
_fallback_stats = {
    'gradient_calls': 0,
    'hessian_calls': 0,
    'jit_bypassed': 0,
    'vmap_loops': 0
}

# Global flags for availability checking
jax_available = JAX_AVAILABLE
numpy_gradients_available = NUMPY_GRADIENTS_AVAILABLE if not JAX_AVAILABLE else False

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
    func_name = getattr(func, '__name__', 'unknown')
    
    @wraps(func)
    def fallback_gradient(*args, **kwargs):
        global _fallback_stats
        _fallback_stats['gradient_calls'] += 1
        
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
    func_name = getattr(func, '__name__', 'unknown')
    
    @wraps(func)
    def fallback_hessian(*args, **kwargs):
        global _fallback_stats
        _fallback_stats['hessian_calls'] += 1
        
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

# Core physics computations
@jit
def compute_g1_diffusion(params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, 
                        q: float) -> jnp.ndarray:
    """
    Compute diffusion contribution to g1 correlation function.
    
    Physical model: g₁_diff = exp[-q²/2 ∫|t₂-t₁| D(t')dt']
    Where: D(t) = D₀ t^α + D_offset
    
    Args:
        params: Physical parameters [D0, alpha, D_offset, ...]
        t1, t2: Time points for correlation calculation
        q: Scattering wave vector magnitude
        
    Returns:
        Diffusion contribution to g1 correlation function
    """
    D0, alpha, D_offset = params[0], params[1], params[2]
    
    # Time difference
    dt = jnp.abs(t2 - t1)
    
    # Integrate D(t) over time interval
    # ∫₀^dt D(t') dt' = D₀ * t^(α+1)/(α+1) + D_offset * t
    alpha_plus_1 = alpha + 1
    
    # Handle special case α = -1 (logarithmic diffusion)
    diffusion_integral = jnp.where(
        jnp.abs(alpha_plus_1) > EPS,
        D0 * dt**(alpha_plus_1) / alpha_plus_1 + D_offset * dt,
        D0 * jnp.log(dt + EPS) + D_offset * dt
    )
    
    # Exponential decay: exp[-q²/2 * integral]
    exponent = -0.5 * q**2 * diffusion_integral
    return safe_exp(exponent)

@jit
def compute_g1_shear(params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray,
                    phi: jnp.ndarray, q: float, L: float) -> jnp.ndarray:
    """
    Compute shear contribution to g1 correlation function.
    
    Physical model: g₁_shear = [sinc(Φ)]²
    Where: Φ = (qL/2π) cos(φ₀-φ) ∫|t₂-t₁| γ̇(t') dt'
    And: γ̇(t) = γ̇₀ t^β + γ̇_offset
    
    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time points for correlation calculation  
        phi: Scattering angles
        q: Scattering wave vector magnitude
        L: Sample-detector distance
        
    Returns:
        Shear contribution to g1 correlation function
    """
    if len(params) < 7:  # Static mode - no shear
        return jnp.ones_like(phi)
    
    gamma_dot_0, beta, gamma_dot_offset, phi0 = params[3], params[4], params[5], params[6]
    
    # Time difference
    dt = jnp.abs(t2 - t1)
    
    # Integrate γ̇(t) over time interval
    # ∫₀^dt γ̇(t') dt' = γ̇₀ * t^(β+1)/(β+1) + γ̇_offset * t
    beta_plus_1 = beta + 1
    
    # Handle special case β = -1 (logarithmic shear)
    shear_integral = jnp.where(
        jnp.abs(beta_plus_1) > EPS,
        gamma_dot_0 * dt**(beta_plus_1) / beta_plus_1 + gamma_dot_offset * dt,
        gamma_dot_0 * jnp.log(dt + EPS) + gamma_dot_offset * dt
    )
    
    # Phase calculation: Φ = (qL/2π) cos(φ₀-φ) * shear_integral
    angle_diff = jnp.deg2rad(phi0 - phi)  # Convert degrees to radians
    cos_term = jnp.cos(angle_diff)
    phase = (q * L / (2 * PI)) * cos_term * shear_integral
    
    # Sinc squared: [sinc(Φ)]²
    sinc_val = safe_sinc(phase)
    return sinc_val**2

@jit
def compute_g1_total(params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray,
                    phi: jnp.ndarray, q: float, L: float) -> jnp.ndarray:
    """
    Compute total g1 correlation function as product of diffusion and shear.
    
    g₁_total = g₁_diffusion × g₁_shear
    
    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles  
        q: Scattering wave vector magnitude
        L: Sample-detector distance
        
    Returns:
        Total g1 correlation function
    """
    g1_diff = compute_g1_diffusion(params, t1, t2, q)
    g1_shear = compute_g1_shear(params, t1, t2, phi, q, L)
    
    # Broadcast diffusion term to match shear dimensions if needed
    return g1_diff * g1_shear

@jit  
def compute_g2_scaled(params: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray,
                     phi: jnp.ndarray, q: float, L: float, 
                     contrast: float, offset: float) -> jnp.ndarray:
    """
    Core scaled optimization: g₂ = offset + contrast × [g₁]²
    
    This is the central equation for homodyne scattering analysis.
    
    Args:
        params: Physical parameters [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        t1, t2: Time points for correlation calculation
        phi: Scattering angles
        q: Scattering wave vector magnitude  
        L: Sample-detector distance
        contrast: Contrast parameter (β in literature)
        offset: Baseline offset
        
    Returns:
        g2 correlation function with scaled fitting
    """
    g1 = compute_g1_total(params, t1, t2, phi, q, L)
    return offset + contrast * g1**2

@jit
def compute_chi_squared(params: jnp.ndarray, data: jnp.ndarray, 
                       sigma: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray,
                       phi: jnp.ndarray, q: float, L: float,
                       contrast: float, offset: float) -> float:
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
def vectorized_g2_computation(params_batch: jnp.ndarray, 
                             t1: jnp.ndarray, t2: jnp.ndarray,
                             phi: jnp.ndarray, q: float, L: float,
                             contrast: float, offset: float) -> jnp.ndarray:
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
    vectorized_func = vmap(compute_g2_scaled, in_axes=(0, None, None, None, None, None, None, None))
    return vectorized_func(params_batch, t1, t2, phi, q, L, contrast, offset)

@log_performance(threshold=0.05) 
def batch_chi_squared(params_batch: jnp.ndarray, data: jnp.ndarray,
                     sigma: jnp.ndarray, t1: jnp.ndarray, t2: jnp.ndarray, 
                     phi: jnp.ndarray, q: float, L: float,
                     contrast: float, offset: float) -> jnp.ndarray:
    """
    Compute chi-squared for multiple parameter sets efficiently.
    """
    if not JAX_AVAILABLE:
        logger.warning("JAX not available - using slower numpy fallback")
        # Simple loop fallback
        results = []
        for params in params_batch:
            result = compute_chi_squared(params, data, sigma, t1, t2, phi, q, L, contrast, offset)
            results.append(result)
        return jnp.array(results)
    
    # JAX vectorized version
    vectorized_func = vmap(compute_chi_squared, in_axes=(0, None, None, None, None, None, None, None, None, None))
    return vectorized_func(params_batch, data, sigma, t1, t2, phi, q, L, contrast, offset)

# Utility functions for optimization
def validate_backend() -> Dict[str, Union[bool, str, Dict]]:
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
        "test_results": {}
    }
    
    # Determine backend type and performance characteristics
    if JAX_AVAILABLE:
        results["backend_type"] = "jax_native"
        results["performance_estimate"] = "optimal (1x)"
    elif numpy_gradients_available:
        results["backend_type"] = "numpy_fallback"
        results["performance_estimate"] = "degraded (10-50x slower)"
        results["recommendations"].append("Install JAX for optimal performance: pip install jax")
    else:
        results["backend_type"] = "none"
        results["performance_estimate"] = "unavailable"
        results["recommendations"].extend([
            "Install JAX for optimal performance: pip install jax",
            "Or install scipy for basic functionality: pip install scipy"
        ])
    
    # Test basic computation
    try:
        test_params = jnp.array([100.0, 0.0, 10.0])
        test_t1 = jnp.array([0.0])
        test_t2 = jnp.array([1.0])
        test_q = 0.01
        
        # Test forward computation
        result = compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        results["test_results"]["forward_computation"] = "success"
        
        # Test gradient computation
        try:
            grad_func = grad(compute_g1_diffusion, argnums=0)
            grad_result = grad_func(test_params, test_t1, test_t2, test_q)
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
            hess_result = hess_func(test_params, test_t1, test_t2, test_q)
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
            "performance_impact": "10-50x slower" if numpy_gradients_available else "unavailable",
            "recommendations": []
        }
        
        if numpy_gradients_available:
            fallback_info["recommendations"].append("Install JAX for optimal performance: pip install jax")
            fallback_info["fallback_stats"] = _fallback_stats.copy()
        else:
            fallback_info["recommendations"].extend([
                "Install JAX for optimal performance: pip install jax",
                "Or install scipy for basic functionality: pip install scipy"
            ])
        
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
            "recommendations": ["JAX is available and configured correctly"]
        }
    except Exception as e:
        logger.warning(f"Could not get JAX device info: {e}")
        return {
            "available": True, 
            "devices": ["unknown"], 
            "backend": "unknown",
            "error": str(e),
            "fallback_active": False
        }

def get_performance_summary() -> Dict[str, Union[str, int, Dict]]:
    """Get performance summary and recommendations."""
    return {
        "backend_type": "jax_native" if JAX_AVAILABLE else ("numpy_fallback" if numpy_gradients_available else "none"),
        "jax_available": JAX_AVAILABLE,
        "numpy_gradients_available": numpy_gradients_available,
        "fallback_stats": _fallback_stats.copy(),
        "performance_multiplier": "1x" if JAX_AVAILABLE else ("10-50x" if numpy_gradients_available else "N/A"),
        "recommendations": _get_performance_recommendations()
    }

def _get_performance_recommendations() -> List[str]:
    """Get performance optimization recommendations."""
    recommendations = []
    
    if not JAX_AVAILABLE:
        recommendations.append("🚀 Install JAX for 10-50x performance improvement: pip install jax")
        
        if not numpy_gradients_available:
            recommendations.append("📊 Install scipy for basic numerical differentiation: pip install scipy")
        else:
            recommendations.append("✅ NumPy gradients available as fallback")
    
    if JAX_AVAILABLE:
        try:
            import jax
            devices = jax.devices()
            if len(devices) > 1:
                recommendations.append(f"🔥 {len(devices)} compute devices available for parallel processing")
            if any('gpu' in str(d).lower() for d in devices):
                recommendations.append("🎯 GPU acceleration available")
            if any('tpu' in str(d).lower() for d in devices):
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