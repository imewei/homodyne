"""
Hybrid NumPyro Models for Noise Estimation in Homodyne v2
========================================================

This module implements sophisticated hybrid NumPyro models that combine Adam optimization
for noise parameters with method-specific physics parameter optimization strategies.

Key Innovation:
- Unified NumPyro probabilistic framework
- Adam optimization for noise parameters (fast, scalable)
- Method-specific physics parameter handling
- Full uncertainty propagation for noise estimates

Model Variants:
1. VI Noise Models: Noise-only estimation for VI pipeline
2. MCMC Hybrid Models: Full joint estimation for MCMC pipeline  
3. Three noise types: hierarchical, per_angle, adaptive

Author: Homodyne v2 Development Team
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# JAX and NumPyro imports with intelligent fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    logger.error("JAX not available - Hybrid NumPyro models require JAX")

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro import sample
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    logger.error("NumPyro not available - Cannot use hybrid noise estimation")

from homodyne.core.fitting import UnifiedHomodyneEngine


def create_vi_noise_model(
    engine: UnifiedHomodyneEngine,
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    noise_type: str = "hierarchical"
) -> Callable:
    """
    Create NumPyro model for noise-only estimation in VI pipeline.
    
    This model is optimized for fast Adam-based estimation of noise parameters only.
    Physics parameters are NOT included - they will be handled by subsequent VI.
    
    Args:
        engine: Unified homodyne engine
        data: Experimental correlation data [n_phi, n_t1, n_t2] or flattened
        t1, t2: Time grids
        phi: Angle grid  
        q, L: Experimental parameters
        noise_type: "hierarchical", "per_angle", or "adaptive"
        
    Returns:
        NumPyro model function for noise estimation
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro required for hybrid noise models")
        
    logger.debug(f"Creating VI noise model: {noise_type}")
    
    def vi_noise_model(*args, **kwargs):
        """Noise-only model for fast Adam optimization."""

        if noise_type == "hierarchical":
            # Single global noise parameter
            sigma = sample("sigma",
                          dist.Gamma(concentration=2.0, rate=20.0))

            # Simplified likelihood using data residuals
            # Use deviation from mean as proxy for noise level
            data_centered = data - jnp.mean(data)

            # Subsample for computational efficiency with large datasets
            data_flat = data_centered.flatten()
            max_points = 10000  # Use at most 10k points for noise estimation
            if data_flat.size > max_points:
                # Random subsample without replacement
                import jax
                key = jax.random.PRNGKey(42)
                indices = jax.random.choice(key, data_flat.size, shape=(max_points,), replace=False)
                data_subsample = data_flat[indices]
            else:
                data_subsample = data_flat

            sample("obs",
                  dist.Normal(0.0, sigma),
                  obs=data_subsample)
            
        elif noise_type == "per_angle":
            # Different noise for each phi angle
            n_phi = len(phi)
            sigma_angles = sample("sigma_angles",
                                 dist.Gamma(concentration=2.0, rate=20.0),
                                 sample_shape=(n_phi,))

            # Per-angle likelihood with subsampling
            import jax
            max_points_per_angle = 500  # Subsample each angle
            for i in range(n_phi):
                if data.ndim == 3:
                    angle_data = data[i].flatten()
                else:
                    # Handle different data shapes
                    n_points_per_angle = data.size // n_phi
                    start_idx = i * n_points_per_angle
                    end_idx = (i + 1) * n_points_per_angle
                    angle_data = data.flatten()[start_idx:end_idx]

                angle_centered = angle_data - jnp.mean(angle_data)

                # Subsample if needed
                if angle_centered.size > max_points_per_angle:
                    key = jax.random.PRNGKey(42 + i)
                    indices = jax.random.choice(key, angle_centered.size,
                                              shape=(max_points_per_angle,), replace=False)
                    angle_subsample = angle_centered[indices]
                else:
                    angle_subsample = angle_centered

                sample(f"obs_angle_{i}",
                      dist.Normal(0.0, sigma_angles[i]),
                      obs=angle_subsample)
            
            
        elif noise_type == "adaptive":
            # Heteroscedastic noise model
            sigma_base = sample("sigma_base",
                               dist.Gamma(concentration=2.0, rate=20.0))
            sigma_scale = sample("sigma_scale",
                                dist.Beta(concentration1=2.0, concentration0=5.0))

            # Adaptive noise depends on signal strength
            # Use absolute deviation as signal strength proxy
            signal_strength = jnp.abs(data - jnp.mean(data))
            sigma_adaptive = sigma_base * (1.0 + sigma_scale * signal_strength)

            # Flatten for likelihood
            data_flat = data.flatten()
            sigma_flat = sigma_adaptive.flatten()
            data_centered = data_flat - jnp.mean(data_flat)

            # Subsample for efficiency
            max_points = 10000
            if data_centered.size > max_points:
                import jax
                key = jax.random.PRNGKey(42)
                indices = jax.random.choice(key, data_centered.size,
                                          shape=(max_points,), replace=False)
                data_subsample = data_centered[indices]
                sigma_subsample = sigma_flat[indices]
            else:
                data_subsample = data_centered
                sigma_subsample = sigma_flat

            sample("obs",
                  dist.Normal(0.0, sigma_subsample),
                  obs=data_subsample)
            
            
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
    
    return vi_noise_model


def create_mcmc_hybrid_model(
    engine: UnifiedHomodyneEngine,
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
    q: float,
    L: float,
    noise_type: str = "hierarchical"
) -> Callable:
    """
    Create full hybrid NumPyro model for joint noise+physics estimation in MCMC pipeline.
    
    This model includes both noise parameters (Adam-initialized) and physics parameters
    (NUTS-sampled) for complete Bayesian treatment with full parameter correlations.
    
    Args:
        engine: Unified homodyne engine
        data: Experimental correlation data
        t1, t2: Time grids
        phi: Angle grid
        q, L: Experimental parameters
        noise_type: "hierarchical", "per_angle", or "adaptive"
        
    Returns:
        NumPyro model function for joint estimation
    """
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro required for hybrid noise models")
        
    logger.debug(f"Creating MCMC hybrid model: {noise_type}")
    
    def mcmc_hybrid_model(*args, **kwargs):
        """Full joint model: noise + physics + scaling parameters."""
        
        # PART 1: Noise parameters (Adam-initialized, can be refined by NUTS)
        if noise_type == "hierarchical":
            sigma = sample("sigma", 
                          dist.Gamma(concentration=2.0, rate=20.0))
        elif noise_type == "per_angle":
            n_phi = len(phi)
            sigma = sample("sigma_angles",
                          dist.Gamma(concentration=2.0, rate=20.0),
                          sample_shape=(n_phi,))
        elif noise_type == "adaptive":
            sigma_base = sample("sigma_base",
                               dist.Gamma(concentration=2.0, rate=20.0))
            sigma_scale = sample("sigma_scale",
                                dist.Beta(concentration1=2.0, concentration0=5.0))
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}")
        
        # PART 2: Physics parameters (NUTS sampling)
        params = []
        param_priors = engine.param_priors
        param_bounds = engine.param_bounds
        
        for i, (prior, bounds) in enumerate(zip(param_priors, param_bounds)):
            mu_prior, sigma_prior = prior
            lower, upper = bounds
            
            # Use appropriate prior distribution
            if bounds in [(-2.0, 2.0), (-100.0, 100.0), (-10.0, 10.0), (-1e5, 1e5)]:
                # Unbounded parameters - use Normal
                param_i = sample(f"param_{i}", 
                               dist.Normal(mu_prior, sigma_prior))
            else:
                # Bounded parameters - use TruncatedNormal
                param_i = sample(f"param_{i}",
                               dist.TruncatedNormal(mu_prior, sigma_prior, 
                                                  low=lower, high=upper))
            params.append(param_i)
            
        params = jnp.array(params)
        
        # PART 3: Scaling parameters (NUTS sampling)
        contrast_prior = engine.parameter_space.contrast_prior
        contrast_bounds = engine.parameter_space.contrast_bounds
        contrast = sample("contrast",
                         dist.TruncatedNormal(contrast_prior[0], contrast_prior[1],
                                            low=contrast_bounds[0], high=contrast_bounds[1]))
        
        offset_prior = engine.parameter_space.offset_prior
        offset_bounds = engine.parameter_space.offset_bounds
        offset = sample("offset",
                       dist.TruncatedNormal(offset_prior[0], offset_prior[1],
                                          low=offset_bounds[0], high=offset_bounds[1]))
        
        # PART 4: Theory computation
        g1_theory = engine.theory_engine.compute_g1(params, t1, t2, phi, q, L)
        g1_squared = g1_theory ** 2
        theory_fitted = contrast * g1_squared + offset
        
        # PART 5: Likelihood with appropriate noise model
        if noise_type == "hierarchical":
            # Single global noise
            sample("obs", dist.Normal(theory_fitted, sigma), obs=data)
            noise_return = sigma
            
        elif noise_type == "per_angle":
            # Per-angle noise
            n_phi = len(phi)
            if data.ndim == 3:
                # Standard [n_phi, n_t1, n_t2] format
                for i in range(n_phi):
                    sample(f"obs_angle_{i}",
                          dist.Normal(theory_fitted[i], sigma[i]),
                          obs=data[i])
            else:
                # Flattened data - need to reshape appropriately
                data_reshaped = data.reshape((n_phi, -1))
                theory_reshaped = theory_fitted.reshape((n_phi, -1))
                for i in range(n_phi):
                    sample(f"obs_angle_{i}",
                          dist.Normal(theory_reshaped[i], sigma[i]),
                          obs=data_reshaped[i])
            
        elif noise_type == "adaptive":
            # Heteroscedastic noise
            sigma_full = sigma_base * (1.0 + sigma_scale * jnp.abs(theory_fitted))
            sample("obs", dist.Normal(theory_fitted, sigma_full), obs=data)
        
    
    return mcmc_hybrid_model


def validate_hybrid_model_inputs(
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray, 
    phi: jnp.ndarray,
    q: float,
    L: float,
    noise_type: str
) -> None:
    """
    Validate inputs for hybrid NumPyro models.
    
    Args:
        data: Experimental correlation data
        t1, t2: Time grids
        phi: Angle grid
        q, L: Experimental parameters
        noise_type: Noise model type
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not JAX_AVAILABLE:
        raise ImportError("JAX required for hybrid NumPyro models")
        
    if not NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro required for hybrid NumPyro models")
        
    # Validate noise type
    valid_noise_types = ["hierarchical", "per_angle", "adaptive"]
    if noise_type not in valid_noise_types:
        raise ValueError(f"noise_type must be one of {valid_noise_types}, got {noise_type}")
    
    # Validate data shapes
    if data.size == 0:
        raise ValueError("Data array is empty")
        
    if len(t1) == 0 or len(t2) == 0:
        raise ValueError("Time arrays cannot be empty")
        
    if len(phi) == 0:
        raise ValueError("Phi array cannot be empty")
    
    # Validate experimental parameters
    if q <= 0:
        raise ValueError("Scattering vector q must be positive")
        
    if L <= 0:
        raise ValueError("Sample thickness L must be positive")
    
    # Per-angle model specific validation
    if noise_type == "per_angle":
        if data.ndim == 3 and data.shape[0] != len(phi):
            raise ValueError(f"Data first dimension {data.shape[0]} must match phi length {len(phi)}")
        elif data.ndim != 3 and data.size % len(phi) != 0:
            raise ValueError(f"Data size {data.size} must be divisible by phi length {len(phi)}")
    
    logger.debug(f"Hybrid model inputs validated for {noise_type} noise model")


def get_noise_model_info(noise_type: str) -> Dict[str, Any]:
    """
    Get information about noise model characteristics.
    
    Args:
        noise_type: Type of noise model
        
    Returns:
        Dictionary with model information
    """
    info = {
        "hierarchical": {
            "description": "Single global noise parameter",
            "parameters": ["sigma"],
            "complexity": "low",
            "recommended_for": "isotropic data, fast estimation",
            "adam_steps": 300,
            "mcmc_adaptation": "standard"
        },
        "per_angle": {
            "description": "Independent noise for each phi angle", 
            "parameters": ["sigma_angles[n_phi]"],
            "complexity": "medium",
            "recommended_for": "anisotropic data, multi-angle analysis",
            "adam_steps": 500,
            "mcmc_adaptation": "extended"
        },
        "adaptive": {
            "description": "Heteroscedastic noise scaling with signal",
            "parameters": ["sigma_base", "sigma_scale"],
            "complexity": "high", 
            "recommended_for": "varying signal quality, advanced analysis",
            "adam_steps": 750,
            "mcmc_adaptation": "careful"
        }
    }
    
    return info.get(noise_type, {"description": "Unknown noise model"})


# Export key functions
__all__ = [
    "create_vi_noise_model",
    "create_mcmc_hybrid_model", 
    "validate_hybrid_model_inputs",
    "get_noise_model_info"
]