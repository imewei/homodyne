"""NLSQ Optimization Subpackage for Homodyne.

This subpackage contains all NLSQ (Non-Linear Least Squares) optimization
components organized into logical modules.

Structure:
- core.py: Main fit_nlsq_jax function and NLSQResult class
- wrapper.py: NLSQWrapper adapter class
- jacobian.py: Jacobian computation utilities
- results.py: OptimizationResult and related dataclasses
- transforms.py: Parameter transformation utilities
- data_prep.py: Data preparation utilities
- result_builder.py: Result building utilities
- fit_computation.py: Fit computation utilities
- multistart.py: Multi-start optimization with LHS (v2.6.0)
- strategies/: Optimization strategy modules
  - selection.py: DatasetSizeStrategy and OptimizationStrategy
  - chunking.py: Angle-stratified chunking for large datasets
  - residual.py: StratifiedResidualFunction for per-angle optimization
  - residual_jit.py: JIT-compiled stratified residual function
  - sequential.py: Sequential per-angle optimization
  - executors.py: Strategy pattern executors
"""

from homodyne.optimization.nlsq.core import (
    JAX_AVAILABLE,
    NLSQ_AVAILABLE,
    NLSQResult,
    _get_param_names,
    fit_nlsq_jax,
    fit_nlsq_multistart,
)

# New refactored modules (Dec 2025)
from homodyne.optimization.nlsq.data_prep import (
    ExpandedParameters,
    PreparedData,
    build_parameter_labels,
    classify_parameter_status,
    convert_bounds_to_nlsq_format,
    expand_per_angle_parameters,
    validate_bounds,
    validate_initial_params,
)
from homodyne.optimization.nlsq.fit_computation import (
    compute_theoretical_fits,
    extract_parameters_from_result,
    get_physical_param_count,
    normalize_analysis_mode,
)

# Multi-start optimization (v2.6.0)
# NOTE: Subsampling is explicitly NOT supported per project requirements.
# Numerical precision and reproducibility take priority over computational speed.
from homodyne.optimization.nlsq.multistart import (
    MultiStartConfig,
    MultiStartResult,
    SingleStartResult,
    check_zero_volume_bounds,
    detect_degeneracy,
    generate_lhs_starts,
    generate_random_starts,
    include_custom_starts,
    run_multistart_nlsq,
    screen_starts,
    validate_n_starts_for_lhs,
)
from homodyne.optimization.nlsq.result_builder import (
    QualityMetrics,
    ResultBuilder,
    compute_quality_metrics,
    compute_uncertainties,
    determine_convergence_status,
    normalize_nlsq_result,
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
from homodyne.optimization.nlsq.strategies.executors import (
    ExecutionResult,
    LargeDatasetExecutor,
    OptimizationExecutor,
    StandardExecutor,
    StreamingExecutor,
    get_executor,
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
    "fit_nlsq_multistart",
    "NLSQResult",
    "JAX_AVAILABLE",
    "NLSQ_AVAILABLE",
    "_get_param_names",
    # Multi-start (v2.6.0)
    # NOTE: No subsampling - numerical precision takes priority
    "MultiStartConfig",
    "MultiStartResult",
    "SingleStartResult",
    "generate_lhs_starts",
    "generate_random_starts",
    "screen_starts",
    "detect_degeneracy",
    "run_multistart_nlsq",
    "include_custom_starts",
    "check_zero_volume_bounds",
    "validate_n_starts_for_lhs",
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
    # Data Preparation (new in Dec 2025)
    "PreparedData",
    "ExpandedParameters",
    "expand_per_angle_parameters",
    "validate_bounds",
    "validate_initial_params",
    "convert_bounds_to_nlsq_format",
    "build_parameter_labels",
    "classify_parameter_status",
    # Result Building (new in Dec 2025)
    "QualityMetrics",
    "ResultBuilder",
    "compute_quality_metrics",
    "compute_uncertainties",
    "normalize_nlsq_result",
    "determine_convergence_status",
    # Fit Computation (new in Dec 2025)
    "compute_theoretical_fits",
    "normalize_analysis_mode",
    "get_physical_param_count",
    "extract_parameters_from_result",
    # Executors (new in Dec 2025)
    "ExecutionResult",
    "OptimizationExecutor",
    "StandardExecutor",
    "LargeDatasetExecutor",
    "StreamingExecutor",
    "get_executor",
]
