"""NLSQ Optimization Subpackage for Homodyne.

This subpackage contains all NLSQ (Non-Linear Least Squares) optimization
components organized into logical modules.

Structure:
- core.py: Main fit_nlsq_jax function and NLSQResult class
- wrapper.py: NLSQWrapper adapter class
- jacobian.py: Jacobian computation utilities
- results.py: OptimizationResult and related dataclasses
- transforms.py: Parameter transformation utilities
- strategies/: Optimization strategy modules
  - selection.py: DatasetSizeStrategy and OptimizationStrategy
  - chunking.py: Angle-stratified chunking for large datasets
  - residual.py: StratifiedResidualFunction for per-angle optimization
  - residual_jit.py: JIT-compiled stratified residual function
  - sequential.py: Sequential per-angle optimization
"""

from homodyne.optimization.nlsq.core import (
    JAX_AVAILABLE,
    NLSQ_AVAILABLE,
    NLSQResult,
    _get_param_names,
    fit_nlsq_jax,
)
from homodyne.optimization.nlsq.results import (
    FunctionEvaluationCounter,
    OptimizationResult,
)
from homodyne.optimization.nlsq.strategies.chunking import (
    StratificationDiagnostics,
    analyze_angle_distribution,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_stratification_memory,
    format_diagnostics_report,
    should_use_stratification,
)
from homodyne.optimization.nlsq.strategies.residual import (
    StratifiedResidualFunction,
    create_stratified_residual_function,
)
from homodyne.optimization.nlsq.strategies.residual_jit import (
    StratifiedResidualFunctionJIT,
)
from homodyne.optimization.nlsq.strategies.selection import (
    DatasetSizeStrategy,
    OptimizationStrategy,
    estimate_memory_requirements,
)
from homodyne.optimization.nlsq.strategies.sequential import (
    JAC_SAMPLE_SIZE,
    optimize_per_angle_sequential,
)
from homodyne.optimization.nlsq.wrapper import NLSQWrapper

__all__ = [
    # Core
    "fit_nlsq_jax",
    "NLSQResult",
    "JAX_AVAILABLE",
    "NLSQ_AVAILABLE",
    # Wrapper
    "NLSQWrapper",
    "OptimizationResult",
    "FunctionEvaluationCounter",
    # Strategies
    "DatasetSizeStrategy",
    "OptimizationStrategy",
    "estimate_memory_requirements",
    # Chunking
    "StratificationDiagnostics",
    "analyze_angle_distribution",
    "compute_stratification_diagnostics",
    "create_angle_stratified_data",
    "create_angle_stratified_indices",
    "estimate_stratification_memory",
    "format_diagnostics_report",
    "should_use_stratification",
    # Residual
    "StratifiedResidualFunction",
    "StratifiedResidualFunctionJIT",
    "create_stratified_residual_function",
    # Sequential
    "JAC_SAMPLE_SIZE",
    "optimize_per_angle_sequential",
]
