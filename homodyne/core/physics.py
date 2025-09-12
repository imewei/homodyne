"""
Physical Constants and Parameter Validation for Homodyne v2
===========================================================

Centralized physical constants, parameter bounds, and validation functions
for homodyne scattering analysis. Provides reference values and constraints
based on experimental physics and numerical stability requirements.

This module establishes the physical framework for all model computations
and ensures parameter values remain within reasonable bounds for stable
numerical computation.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

class PhysicsConstants:
    """
    Physical constants and reference values for XPCS analysis.
    
    These values are based on typical synchrotron X-ray scattering
    experiments and provide reasonable defaults for most analyses.
    """
    
    # X-ray wavelengths (Angstroms)
    WAVELENGTH_CU_KA = 1.54  # Copper K-alpha
    WAVELENGTH_8KEV = 1.55   # ~8 keV synchrotron
    WAVELENGTH_12KEV = 1.03  # ~12 keV synchrotron
    WAVELENGTH_15KEV = 0.83  # ~15 keV synchrotron
    
    # Typical q-ranges (inverse Angstroms)
    Q_MIN_TYPICAL = 1e-4
    Q_MAX_TYPICAL = 1.0
    
    # Time scales (seconds)
    TIME_MIN_XPCS = 1e-6     # Microsecond resolution
    TIME_MAX_XPCS = 1e3      # Kilosecond measurements
    
    # Diffusion coefficient ranges (Å²/s)
    DIFFUSION_MIN = 1e-3     # Very slow diffusion
    DIFFUSION_MAX = 1e6      # Very fast diffusion
    DIFFUSION_TYPICAL = 100.0
    
    # Shear rate ranges (s⁻¹)
    SHEAR_RATE_MIN = 1e-4    # Very slow shear
    SHEAR_RATE_MAX = 1e3     # Very fast shear
    SHEAR_RATE_TYPICAL = 1.0
    
    # Angular ranges (degrees)
    ANGLE_MIN = -180.0
    ANGLE_MAX = 180.0
    
    # Numerical stability
    EPS = 1e-12              # Avoid division by zero
    MAX_EXP_ARG = 700.0      # Prevent exponential overflow
    MIN_POSITIVE = 1e-100    # Minimum positive value
    
    # Physical parameter bounds
    ALPHA_MIN = -2.0         # Minimum diffusion exponent
    ALPHA_MAX = 2.0          # Maximum diffusion exponent
    BETA_MIN = -2.0          # Minimum shear exponent
    BETA_MAX = 2.0           # Maximum shear exponent

def parameter_bounds() -> Dict[str, List[Tuple[float, float]]]:
    """
    Get standard parameter bounds for all model types.
    
    Returns:
        Dictionary mapping model types to parameter bounds
    """
    return {
        "diffusion": [
            (PhysicsConstants.DIFFUSION_MIN, PhysicsConstants.DIFFUSION_MAX),  # D0
            (PhysicsConstants.ALPHA_MIN, PhysicsConstants.ALPHA_MAX),          # alpha  
            (0.0, PhysicsConstants.DIFFUSION_MAX/10),                          # D_offset
        ],
        
        "shear": [
            (PhysicsConstants.SHEAR_RATE_MIN, PhysicsConstants.SHEAR_RATE_MAX), # gamma_dot_0
            (PhysicsConstants.BETA_MIN, PhysicsConstants.BETA_MAX),             # beta
            (0.0, PhysicsConstants.SHEAR_RATE_MAX/10),                          # gamma_dot_offset
            (PhysicsConstants.ANGLE_MIN, PhysicsConstants.ANGLE_MAX),           # phi0
        ],
        
        "combined": [
            # Diffusion parameters
            (PhysicsConstants.DIFFUSION_MIN, PhysicsConstants.DIFFUSION_MAX),  # D0
            (PhysicsConstants.ALPHA_MIN, PhysicsConstants.ALPHA_MAX),          # alpha
            (0.0, PhysicsConstants.DIFFUSION_MAX/10),                          # D_offset
            # Shear parameters  
            (PhysicsConstants.SHEAR_RATE_MIN, PhysicsConstants.SHEAR_RATE_MAX), # gamma_dot_0
            (PhysicsConstants.BETA_MIN, PhysicsConstants.BETA_MAX),             # beta
            (0.0, PhysicsConstants.SHEAR_RATE_MAX/10),                          # gamma_dot_offset
            (PhysicsConstants.ANGLE_MIN, PhysicsConstants.ANGLE_MAX),           # phi0
        ]
    }

def validate_parameters(params: np.ndarray, bounds: List[Tuple[float, float]], 
                       tolerance: float = 1e-10) -> bool:
    """
    Validate parameter values against bounds with tolerance.
    
    Args:
        params: Parameter array to validate
        bounds: List of (min, max) tuples for each parameter
        tolerance: Tolerance for bounds checking
        
    Returns:
        True if all parameters are within bounds, False otherwise
    """
    if len(params) != len(bounds):
        logger.warning(f"Parameter count mismatch: got {len(params)}, expected {len(bounds)}")
        return False
    
    for i, (param, (min_val, max_val)) in enumerate(zip(params, bounds)):
        if not (min_val - tolerance <= param <= max_val + tolerance):
            logger.warning(f"Parameter {i} out of bounds: {param} not in [{min_val}, {max_val}]")
            return False
    
    return True

def clip_parameters(params: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
    """
    Clip parameters to stay within bounds.
    
    Args:
        params: Parameter array to clip
        bounds: List of (min, max) tuples for each parameter
        
    Returns:
        Clipped parameter array
    """
    if len(params) != len(bounds):
        raise ValueError(f"Parameter count mismatch: got {len(params)}, expected {len(bounds)}")
    
    clipped = np.zeros_like(params)
    for i, (param, (min_val, max_val)) in enumerate(zip(params, bounds)):
        clipped[i] = np.clip(param, min_val, max_val)
        
        if abs(clipped[i] - param) > 1e-10:
            logger.debug(f"Clipped parameter {i}: {param} → {clipped[i]}")
    
    return clipped

def get_default_parameters(model_type: str) -> np.ndarray:
    """
    Get sensible default parameters for a model type.
    
    Args:
        model_type: "diffusion", "shear", or "combined"
        
    Returns:
        Array of default parameter values
    """
    defaults = {
        "diffusion": np.array([
            PhysicsConstants.DIFFUSION_TYPICAL,  # D0 = 100 Å²/s  
            0.0,                                 # alpha = 0 (normal diffusion)
            PhysicsConstants.DIFFUSION_TYPICAL/10  # D_offset = 10 Å²/s
        ]),
        
        "shear": np.array([
            PhysicsConstants.SHEAR_RATE_TYPICAL,  # gamma_dot_0 = 1 s⁻¹
            0.0,                                  # beta = 0 (constant shear)
            0.0,                                  # gamma_dot_offset = 0
            0.0                                   # phi0 = 0 degrees
        ]),
        
        "combined": np.array([
            # Diffusion defaults
            PhysicsConstants.DIFFUSION_TYPICAL,   # D0 = 100 Å²/s
            0.0,                                  # alpha = 0 
            PhysicsConstants.DIFFUSION_TYPICAL/10, # D_offset = 10 Å²/s
            # Shear defaults
            PhysicsConstants.SHEAR_RATE_TYPICAL,  # gamma_dot_0 = 1 s⁻¹ 
            0.0,                                  # beta = 0
            0.0,                                  # gamma_dot_offset = 0
            0.0                                   # phi0 = 0 degrees
        ])
    }
    
    if model_type not in defaults:
        raise ValueError(f"Unknown model type '{model_type}'. Must be one of {list(defaults.keys())}")
    
    return defaults[model_type]

def validate_experimental_setup(q: float, L: float, wavelength: float = None) -> bool:
    """
    Validate experimental setup parameters for physical reasonableness.
    
    Args:
        q: Scattering wave vector magnitude (Å⁻¹)
        L: Sample-detector distance (mm)
        wavelength: X-ray wavelength (Å), optional
        
    Returns:
        True if setup is physically reasonable
    """
    # Check q-range
    if not (PhysicsConstants.Q_MIN_TYPICAL <= q <= PhysicsConstants.Q_MAX_TYPICAL):
        logger.warning(f"q-value {q:.2e} Å⁻¹ outside typical range "
                      f"[{PhysicsConstants.Q_MIN_TYPICAL:.2e}, {PhysicsConstants.Q_MAX_TYPICAL:.2e}]")
        return False
    
    # Check detector distance (typical range: 10mm to 10m)
    if not (10.0 <= L <= 10000.0):
        logger.warning(f"Sample-detector distance {L:.1f} mm outside reasonable range [10, 10000]")
        return False
    
    # Check wavelength if provided
    if wavelength is not None:
        if not (0.1 <= wavelength <= 10.0):
            logger.warning(f"X-ray wavelength {wavelength:.2f} Å outside reasonable range [0.1, 10.0]")
            return False
    
    return True

def estimate_correlation_time(D0: float, alpha: float, q: float) -> float:
    """
    Estimate characteristic correlation time for diffusion process.
    
    For normal diffusion (alpha=0): τ ≈ 1/(q²D₀)
    For anomalous diffusion: scaling is more complex
    
    Args:
        D0: Reference diffusion coefficient (Å²/s)
        alpha: Diffusion exponent
        q: Scattering wave vector (Å⁻¹)
        
    Returns:
        Estimated correlation time (seconds)
    """
    if alpha == 0.0:
        # Normal diffusion
        return 1.0 / (q**2 * D0) if D0 > 0 else np.inf
    else:
        # Anomalous diffusion - approximate scaling
        # This is a rough estimate for experimental planning
        base_time = 1.0 / (q**2 * D0) if D0 > 0 else np.inf
        return base_time * (1.0 + abs(alpha))  # Rough correction

def get_parameter_info(model_type: str) -> Dict[str, Any]:
    """
    Get comprehensive parameter information for a model type.
    
    Args:
        model_type: "diffusion", "shear", or "combined"
        
    Returns:
        Dictionary with parameter names, bounds, defaults, and descriptions
    """
    info = {
        "diffusion": {
            "names": ["D0", "alpha", "D_offset"],
            "descriptions": [
                "Reference diffusion coefficient (Å²/s)",
                "Diffusion time-dependence exponent (-)",
                "Baseline diffusion coefficient (Å²/s)"
            ],
            "physical_meaning": [
                "Characteristic mobility scale",
                "0=normal, >0=super-diffusion, <0=sub-diffusion", 
                "Residual diffusion at t=0"
            ]
        },
        
        "shear": {
            "names": ["gamma_dot_0", "beta", "gamma_dot_offset", "phi0"],
            "descriptions": [
                "Reference shear rate (s⁻¹)",
                "Shear rate time-dependence exponent (-)",
                "Baseline shear rate (s⁻¹)",
                "Flow direction angle (degrees)"
            ],
            "physical_meaning": [
                "Characteristic shear rate scale",
                "0=constant, >0=accelerating, <0=decelerating",
                "Residual shear rate at t=0", 
                "Preferred flow direction"
            ]
        },
        
        "combined": {
            "names": ["D0", "alpha", "D_offset", "gamma_dot_0", "beta", "gamma_dot_offset", "phi0"],
            "descriptions": [
                "Reference diffusion coefficient (Å²/s)",
                "Diffusion time-dependence exponent (-)",
                "Baseline diffusion coefficient (Å²/s)",
                "Reference shear rate (s⁻¹)",
                "Shear rate time-dependence exponent (-)", 
                "Baseline shear rate (s⁻¹)",
                "Flow direction angle (degrees)"
            ],
            "physical_meaning": [
                "Characteristic mobility scale",
                "0=normal, >0=super-diffusion, <0=sub-diffusion",
                "Residual diffusion at t=0",
                "Characteristic shear rate scale", 
                "0=constant, >0=accelerating, <0=decelerating",
                "Residual shear rate at t=0",
                "Preferred flow direction"
            ]
        }
    }
    
    if model_type not in info:
        raise ValueError(f"Unknown model type '{model_type}'")
    
    # Add common information
    result = info[model_type].copy()
    result.update({
        "bounds": parameter_bounds()[model_type],
        "defaults": get_default_parameters(model_type).tolist(),
        "n_parameters": len(info[model_type]["names"])
    })
    
    return result

# Export main functions and constants
__all__ = [
    "PhysicsConstants",
    "parameter_bounds",
    "validate_parameters", 
    "clip_parameters",
    "get_default_parameters",
    "validate_experimental_setup",
    "estimate_correlation_time",
    "get_parameter_info",
]