"""
Core Computation Engine for Homodyne v2
========================================

JAX-first computational backend providing high-performance implementations
of the physical models and mathematical operations for homodyne scattering analysis.

Key Components:
- jax_backend: JAX computational primitives and JIT-compiled functions
- models: Physical models (diffusion, shear, combined)
- theory: Theoretical computations (g1, g2 calculations) 
- fitting: Core fitting engine (Fitted = contrast × Theory + offset)
- physics: Physical constants and utility functions

Performance Features:
- JIT compilation for optimal performance
- Automatic differentiation for gradient-based optimization
- Vectorized operations for parallel computation
- GPU/TPU acceleration when available
- Memory-efficient handling of large correlation matrices
"""

from homodyne.core.jax_backend import (
    jax_available,
    compute_g1_diffusion,
    compute_g1_shear,
    compute_g2_scaled,
    gradient_g2,
    hessian_g2,
)

from homodyne.core.models import (
    DiffusionModel,
    ShearModel, 
    CombinedModel,
    PhysicsModelBase,
)

from homodyne.core.theory import (
    TheoryEngine,
    compute_g2_theory,
    compute_chi2_theory,
)

from homodyne.core.fitting import (
    ScaledFittingEngine,
    FitResult,
)

from homodyne.core.physics import (
    PhysicsConstants,
    validate_parameters,
    parameter_bounds,
)

__all__ = [
    # JAX backend
    "jax_available",
    "compute_g1_diffusion", 
    "compute_g1_shear",
    "compute_g2_scaled",
    "gradient_g2",
    "hessian_g2",
    
    # Models
    "DiffusionModel",
    "ShearModel",
    "CombinedModel", 
    "PhysicsModelBase",
    
    # Theory computation
    "TheoryEngine",
    "compute_g2_theory",
    "compute_chi2_theory",
    
    # Fitting engine
    "ScaledFittingEngine",
    "FitResult",
    
    # Physics utilities
    "PhysicsConstants",
    "validate_parameters",
    "parameter_bounds",
]