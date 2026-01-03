"""NLSQ Optimization Subpackage for Homodyne.

This subpackage contains all NLSQ (Non-Linear Least Squares) optimization
components organized into logical modules.

Structure:
- core.py: Main fit_nlsq_jax function and NLSQResult class
- wrapper.py: NLSQWrapper adapter class (legacy)
- adapter.py: NLSQAdapter using NLSQ's CurveFit class (v2.11.0+)
- memory.py: Memory management utilities (extracted Jan 2026)
- parameter_utils.py: Parameter utilities (extracted Jan 2026)
- jacobian.py: Jacobian computation utilities
- results.py: OptimizationResult and related dataclasses
- transforms.py: Parameter transformation utilities
- data_prep.py: Data preparation utilities
- result_builder.py: Result building utilities
- fit_computation.py: Fit computation utilities
- multistart.py: Multi-start optimization with LHS (v2.6.0)
- anti_degeneracy_adapter.py: NLSQ integration for anti-degeneracy (v2.11.0+)
- strategies/: Optimization strategy modules
  - selection.py: DatasetSizeStrategy and OptimizationStrategy
  - chunking.py: Angle-stratified chunking for large datasets
  - residual.py: StratifiedResidualFunction for per-angle optimization
  - residual_jit.py: JIT-compiled stratified residual function
  - sequential.py: Sequential per-angle optimization
  - executors.py: Strategy pattern executors

NLSQ Integration (v2.11.0+):
- Uses NLSQ's CurveFit class for JIT compilation caching
- Leverages WorkflowSelector for automatic strategy selection
- Integrates with MultiStartOrchestrator for global optimization
- Anti-degeneracy layers remain in homodyne (physics-specific)
"""

# =============================================================================
# NLSQ Package Imports (v0.4+)
# These provide the new unified API with CurveFit class and WorkflowSelector
# =============================================================================

# Core NLSQ imports (always available)
try:
    from nlsq import CurveFit, curve_fit

    NLSQ_CURVEFIT_AVAILABLE = True
except ImportError:
    CurveFit = None  # type: ignore[assignment, misc]
    curve_fit = None  # type: ignore[assignment]
    NLSQ_CURVEFIT_AVAILABLE = False

# Workflow selection (NLSQ v0.4+)
try:
    from nlsq.core.workflow import (
        DatasetSizeTier as NLSQDatasetSizeTier,
    )
    from nlsq.core.workflow import (
        OptimizationGoal,
        WorkflowSelector,
        WorkflowTier,
    )

    NLSQ_WORKFLOW_AVAILABLE = True
except ImportError:
    WorkflowSelector = None  # type: ignore[assignment, misc]
    WorkflowTier = None  # type: ignore[assignment, misc]
    OptimizationGoal = None  # type: ignore[assignment, misc]
    NLSQDatasetSizeTier = None  # type: ignore[assignment, misc]
    NLSQ_WORKFLOW_AVAILABLE = False

# Global optimization (NLSQ v0.4+)
try:
    from nlsq.global_optimization import (
        GlobalOptimizationConfig,
        MultiStartOrchestrator,
    )

    NLSQ_GLOBAL_OPT_AVAILABLE = True
except ImportError:
    GlobalOptimizationConfig = None  # type: ignore[assignment, misc]
    MultiStartOrchestrator = None  # type: ignore[assignment, misc]
    NLSQ_GLOBAL_OPT_AVAILABLE = False

# Stability and recovery (NLSQ v0.4+)
try:
    from nlsq.stability import (
        NumericalStabilityGuard,
    )
    from nlsq.stability import (
        OptimizationRecovery as NLSQOptimizationRecovery,
    )

    NLSQ_STABILITY_AVAILABLE = True
except ImportError:
    NumericalStabilityGuard = None  # type: ignore[assignment, misc]
    NLSQOptimizationRecovery = None  # type: ignore[assignment, misc]
    NLSQ_STABILITY_AVAILABLE = False

# Caching and memory management (NLSQ v0.4+)
try:
    from nlsq.caching import MemoryManager as NLSQMemoryManager
    from nlsq.caching import get_memory_manager

    NLSQ_CACHING_AVAILABLE = True
except ImportError:
    NLSQMemoryManager = None  # type: ignore[assignment, misc]
    get_memory_manager = None  # type: ignore[assignment]
    NLSQ_CACHING_AVAILABLE = False

# Result types (NLSQ v0.4+)
try:
    from nlsq.result import CurveFitResult

    NLSQ_RESULT_AVAILABLE = True
except ImportError:
    CurveFitResult = None  # type: ignore[assignment, misc]
    NLSQ_RESULT_AVAILABLE = False

# Streaming optimizer (NLSQ v0.3.2+)
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    NLSQ_STREAMING_AVAILABLE = True
except ImportError:
    AdaptiveHybridStreamingOptimizer = None  # type: ignore[assignment, misc]
    HybridStreamingConfig = None  # type: ignore[assignment, misc]
    NLSQ_STREAMING_AVAILABLE = False

# =============================================================================
# Homodyne NLSQ Module Imports
# =============================================================================

# NLSQAdapter using CurveFit class (v2.11.0+)
from homodyne.optimization.nlsq.adapter import (
    AdapterConfig,
    NLSQAdapter,
    clear_model_cache,
    get_adapter,
    get_cache_stats,
    get_or_create_model,
    is_adapter_available,
)

# Anti-degeneracy defense system (v2.9.0)
from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)
from homodyne.optimization.nlsq.config import (
    HybridRecoveryConfig,
    NLSQConfig,
)
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

# Memory management utilities (extracted Jan 2026)
from homodyne.optimization.nlsq.memory import (
    DEFAULT_MEMORY_FRACTION,
    FALLBACK_THRESHOLD_GB,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    get_adaptive_memory_threshold,
    should_use_streaming,
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
from homodyne.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

# Parameter utilities (extracted Jan 2026)
from homodyne.optimization.nlsq.parameter_utils import (
    build_parameter_labels as build_parameter_labels_utils,
)
from homodyne.optimization.nlsq.parameter_utils import (
    classify_parameter_status as classify_parameter_status_utils,
)
from homodyne.optimization.nlsq.parameter_utils import (
    compute_consistent_per_angle_init,
    compute_jacobian_stats,
    sample_xdata,
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
    # NLSQ Package Integration (v2.11.0+)
    # Core NLSQ classes
    "CurveFit",
    "curve_fit",
    "NLSQ_CURVEFIT_AVAILABLE",
    # Workflow selection
    "WorkflowSelector",
    "WorkflowTier",
    "OptimizationGoal",
    "NLSQDatasetSizeTier",
    "NLSQ_WORKFLOW_AVAILABLE",
    # Global optimization
    "GlobalOptimizationConfig",
    "MultiStartOrchestrator",
    "NLSQ_GLOBAL_OPT_AVAILABLE",
    # Stability and recovery
    "NumericalStabilityGuard",
    "NLSQOptimizationRecovery",
    "NLSQ_STABILITY_AVAILABLE",
    # Caching
    "NLSQMemoryManager",
    "get_memory_manager",
    "NLSQ_CACHING_AVAILABLE",
    # Result types
    "CurveFitResult",
    "NLSQ_RESULT_AVAILABLE",
    # Streaming
    "AdaptiveHybridStreamingOptimizer",
    "HybridStreamingConfig",
    "NLSQ_STREAMING_AVAILABLE",
    # Homodyne Core
    "fit_nlsq_jax",
    "fit_nlsq_multistart",
    "NLSQResult",
    "JAX_AVAILABLE",
    "NLSQ_AVAILABLE",
    "_get_param_names",
    # Configuration (v2.11.0+)
    "NLSQConfig",
    "HybridRecoveryConfig",
    # Anti-degeneracy defense system (v2.9.0)
    "AntiDegeneracyConfig",
    "AntiDegeneracyController",
    "ParameterIndexMapper",
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
    # Wrapper (legacy)
    "NLSQWrapper",
    "OptimizationResult",
    "FunctionEvaluationCounter",
    # Adapter (v2.11.0+ - recommended)
    "NLSQAdapter",
    "AdapterConfig",
    "get_adapter",
    "is_adapter_available",
    # Model caching (v2.11.0+)
    "get_or_create_model",
    "clear_model_cache",
    "get_cache_stats",
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
    # Memory management (extracted Jan 2026)
    "DEFAULT_MEMORY_FRACTION",
    "FALLBACK_THRESHOLD_GB",
    "detect_total_system_memory",
    "estimate_peak_memory_gb",
    "get_adaptive_memory_threshold",
    "should_use_streaming",
    # Parameter utilities (extracted Jan 2026)
    "build_parameter_labels_utils",
    "classify_parameter_status_utils",
    "compute_consistent_per_angle_init",
    "compute_jacobian_stats",
    "sample_xdata",
]
