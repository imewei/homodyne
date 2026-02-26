"""Core Computation Engine for Homodyne
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
- CPU-optimized multi-core computation
- Memory-efficient handling of large correlation matrices
"""

from homodyne.core.fitting import FitResult, ParameterSpace, ScaledFittingEngine
from homodyne.core.homodyne_model import HomodyneModel
from homodyne.core.jax_backend import (
    compute_g1_diffusion,
    compute_g1_shear,
    compute_g2_scaled,
    compute_g2_scaled_with_factors,
    gradient_g2,
    hessian_g2,
    jax_available,
)
from homodyne.core.models import (
    CombinedModel,
    DiffusionModel,
    PhysicsModelBase,
    ShearModel,
)
from homodyne.core.physics import (
    PhysicsConstants,
    parameter_bounds,
    validate_parameters,
)
from homodyne.core.physics_factors import (
    PhysicsFactors,
    create_physics_factors_from_config_dict,
)
from homodyne.core.theory import TheoryEngine, compute_chi2_theory, compute_g2_theory

__all__ = [
    # JAX backend
    "jax_available",
    "compute_g1_diffusion",
    "compute_g1_shear",
    "compute_g2_scaled",
    "compute_g2_scaled_with_factors",
    "gradient_g2",
    "hessian_g2",
    # Models
    "DiffusionModel",
    "ShearModel",
    "CombinedModel",
    "PhysicsModelBase",
    "HomodyneModel",
    # Theory computation
    "TheoryEngine",
    "compute_g2_theory",
    "compute_chi2_theory",
    # Fitting engine
    "ScaledFittingEngine",
    "FitResult",
    "ParameterSpace",
    # Physics utilities
    "PhysicsConstants",
    "validate_parameters",
    "parameter_bounds",
    # Physics factors
    "PhysicsFactors",
    "create_physics_factors_from_config_dict",
]
