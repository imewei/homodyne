"""
Homodyne Scattering Analysis Package v2
========================================

High-performance Python package for analyzing homodyne scattering in X-ray Photon
Correlation Spectroscopy (XPCS) under nonequilibrium conditions. 

Completely rebuilt with JAX-first architecture for modern HPC environments.

Key Features:
- JAX-powered computation: 10-50x performance improvements with GPU/TPU support
- Simplified optimization: VI (primary) → MCMC (optional) pipeline  
- HPC integration: Native PBS Professional and distributed computing
- Extensible architecture: Plugin system for models and optimizers

Core Analysis Modes:
- Static Isotropic (3 parameters): D₀, α, D_offset
- Static Anisotropic (3 parameters): D₀, α, D_offset with angle filtering  
- Laminar Flow (7 parameters): + γ̇₀, β, γ̇_offset, φ₀

Optimization Methods:
- Variational Inference (VI): Fast approximate posterior with uncertainty
- MCMC: Full posterior characterization with JAX-accelerated sampling
- Hybrid: VI → MCMC pipeline for best of both worlds

Physical Model:
g₂(φ,t₁,t₂) = offset + contrast × [g₁(φ,t₁,t₂)]²

Where g₁ captures the interplay between:
- Anomalous diffusion: D(t) = D₀ t^α + D_offset
- Time-dependent shear: γ̇(t) = γ̇₀ t^β + γ̇_offset

Reference:
H. He et al., "Transport coefficient approach for characterizing nonequilibrium
dynamics in soft matter", PNAS 121 (31) e2401162121 (2024).

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

# Version information
__version__ = "2.0.0-alpha"
__author__ = "Wei Chen, Hongrui He"  
__email__ = "wchen@anl.gov"
__institution__ = "Argonne National Laboratory"

# Core v2 API - New JAX-first implementation
from homodyne.core.models import DiffusionModel, ShearModel, CombinedModel
from homodyne.optimization.variational import VariationalInferenceJAX
from homodyne.optimization.mcmc import MCMCJAXSampler
from homodyne.optimization.hybrid import HybridOptimizer

# Configuration and workflow management
from homodyne.config.manager import ConfigManager
from homodyne.workflows.pipeline import AnalysisPipeline

# Data loading with YAML configuration
from homodyne.data.xpcs_loader import XPCSDataLoader, load_xpcs_data

# Utilities (preserve existing logging system)
from homodyne.utils.logging import get_logger


__all__ = [
    # Core v2 API
    "DiffusionModel", 
    "ShearModel",
    "CombinedModel",
    "VariationalInferenceJAX",
    "MCMCJAXSampler", 
    "HybridOptimizer",
    "ConfigManager",
    "AnalysisPipeline",
    "XPCSDataLoader",
    "load_xpcs_data",
    "get_logger",
    
]

# Package metadata
__all_methods__ = ["classical", "robust", "mcmc", "all", "vi", "hybrid"]
__analysis_modes__ = ["static_isotropic", "static_anisotropic", "laminar_flow"]