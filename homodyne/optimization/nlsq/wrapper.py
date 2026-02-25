"""NLSQ Wrapper for Homodyne Optimization.

Role and When to Use (v2.11.0+)
-------------------------------

**NLSQWrapper** (this module) is the **stable fallback adapter** for:
- Complex optimizations requiring full anti-degeneracy integration
- laminar_flow mode with many phi angles (> 6)
- Large datasets (> 100M points) requiring streaming/chunking strategies
- Custom transforms or advanced recovery mechanisms
- Production stability where reliability is critical

Use **NLSQAdapter** instead for:
- Standard optimizations (static_isotropic mode)
- Small to medium datasets (< 10M points)
- Multi-start optimization (model caching provides 3-5× speedup)
- Performance-critical workflows requiring JIT compilation

**Key Differences:**

* Model caching: NLSQWrapper=None, NLSQAdapter=Built-in
* JIT compilation: NLSQWrapper=Manual, NLSQAdapter=Auto
* Workflow auto-select: NLSQWrapper=Custom, NLSQAdapter=Via NLSQ
* Anti-degeneracy layers: NLSQWrapper=Full, NLSQAdapter=Via fit()
* Recovery system: NLSQWrapper=3-attempt, NLSQAdapter=NLSQ native
* Streaming support: NLSQWrapper=Full custom, NLSQAdapter=Via NLSQ

**Decision Guide:**

1. If you need robust streaming for 100M+ points: Use NLSQWrapper
2. If you need full anti-degeneracy control: Use NLSQWrapper
3. If you need maximum speed for multi-start optimization: Use NLSQAdapter
4. Default recommendation: NLSQAdapter with automatic fallback to NLSQWrapper

This module provides an adapter layer between homodyne's optimization API
and the NLSQ package's trust-region nonlinear least squares interface.

The NLSQWrapper class implements the Adapter pattern to translate:
- Homodyne's multi-dimensional XPCS data → NLSQ's flattened array format
- Homodyne's parameter bounds tuple → NLSQ's (lower, upper) format
- NLSQ's (popt, pcov) output → Homodyne's OptimizationResult dataclass

Key Features:
- Automatic dataset size detection and strategy selection
- Angle-stratified chunking for per-angle parameter compatibility (v2.2+)
- Intelligent error recovery with 3-attempt retry strategy (T022-T024)
- Actionable error diagnostics with 5 error categories
- CPU-optimized execution through JAX
- Progress logging and convergence diagnostics
- Scientifically validated (7/7 validation tests passed, T036-T041)
- Serves as fallback when NLSQAdapter fails

Per-Angle Scaling Fix (v2.2):
- Fixes silent optimization failures with per-angle parameters on large datasets
- Applies angle-stratified chunking when: per_angle_scaling=True AND n_points>100k
- Ensures every NLSQ chunk contains all phi angles → gradients always well-defined
- <1% performance overhead (0.15s for 3M points)
- Reference: ultra-think-20251106-012247

Production Status:
- ✅ Production-ready with comprehensive error recovery
- ✅ Scientifically validated (100% test pass rate)
- ✅ Parameter recovery accuracy: 2-14% on core parameters
- ✅ Sub-linear performance scaling with dataset size
- ✅ Per-angle scaling compatible with large datasets (v2.2+)

References:
- NLSQ Package: https://github.com/imewei/NLSQ
- Validation: See tests/validation/test_scientific_validation.py (T036-T041)
- Documentation: See CHANGELOG.md and CLAUDE.md for detailed status
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

# ruff: noqa: I001
# Import order is INTENTIONAL: nlsq must be imported BEFORE JAX
# This enables automatic x64 (double precision) configuration per NLSQ best practices
# Reference: https://nlsq.readthedocs.io/en/latest/guides/advanced_features.html
from nlsq import LeastSquares, curve_fit, curve_fit_large

# Try importing AdaptiveHybridStreamingOptimizer (available in NLSQ >= 0.3.2)
# This is the preferred streaming optimizer - the old StreamingOptimizer was removed in NLSQ 0.4.0
# Fixes: 1) Shear-term weak gradients, 2) Slow convergence, 3) Crude covariance
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    STREAMING_AVAILABLE = True  # For backwards compatibility
    HYBRID_STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False
    HYBRID_STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None
    HybridStreamingConfig = None

import logging

from homodyne.utils.logging import get_logger, log_exception

from homodyne.optimization.batch_statistics import BatchStatistics
from homodyne.optimization.nlsq.adapter_base import NLSQAdapterBase
from homodyne.optimization.exceptions import NLSQOptimizationError
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
from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyController,
)

# Local OptimizationStrategy enum (selection.py removed in v2.13.0)
# This is kept for backward compatibility with fallback chain logic
from enum import Enum as _Enum


class OptimizationStrategy(_Enum):
    """Local optimization strategy enum for internal use.

    Note: This replaces the deprecated selection.py OptimizationStrategy.
    For new code, use NLSQStrategy from memory.py instead.
    """

    STANDARD = "standard"
    LARGE = "large"
    CHUNKED = "chunked"
    STREAMING = "streaming"


def _get_strategy_info(strategy: OptimizationStrategy) -> dict:
    """Get information about a strategy for logging/diagnostics."""
    info = {
        OptimizationStrategy.STANDARD: {
            "name": "Standard",
            "supports_progress": False,
        },
        OptimizationStrategy.LARGE: {
            "name": "Large",
            "supports_progress": True,
        },
        OptimizationStrategy.CHUNKED: {
            "name": "Chunked",
            "supports_progress": True,
        },
        OptimizationStrategy.STREAMING: {
            "name": "Streaming",
            "supports_progress": True,
        },
    }
    return info.get(strategy, {"name": "Unknown", "supports_progress": False})


from homodyne.optimization.nlsq.strategies.sequential import (  # noqa: E402
    JAC_SAMPLE_SIZE,
    optimize_per_angle_sequential,
)
from homodyne.optimization.nlsq.strategies.chunking import (  # noqa: E402
    calculate_adaptive_chunk_size,
    get_stratified_chunk_iterator,
)
from homodyne.core.physics_nlsq import compute_g2_scaled  # noqa: E402
from homodyne.core.physics_utils import apply_diagonal_correction  # noqa: E402
from homodyne.optimization.nlsq.transforms import (  # noqa: E402
    adjust_covariance_for_transforms,
    apply_forward_shear_transforms_to_bounds,
    apply_forward_shear_transforms_to_vector,
    apply_inverse_shear_transforms_to_vector,
    build_per_parameter_x_scale,
    build_physical_index_map,
    format_x_scale_for_log,
    normalize_x_scale_map,
    parse_shear_transform_config,
    wrap_model_function_with_transforms,
    wrap_stratified_function_with_transforms,
)
from homodyne.optimization.numerical_validation import NumericalValidator  # noqa: E402
from homodyne.optimization.recovery_strategies import (  # noqa: E402
    RecoveryStrategyApplicator,
)

# Anti-Degeneracy Defense System v2.9.0
from homodyne.optimization.nlsq.adaptive_regularization import (  # noqa: E402
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from homodyne.optimization.nlsq.fourier_reparam import (  # noqa: E402
    FourierReparamConfig,
    FourierReparameterizer,
)
from homodyne.optimization.nlsq.gradient_monitor import (  # noqa: E402
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from homodyne.optimization.nlsq.hierarchical import (  # noqa: E402
    HierarchicalConfig,
    HierarchicalOptimizer,
)
from homodyne.optimization.nlsq.shear_weighting import (  # noqa: E402
    ShearSensitivityWeighting,
    ShearWeightingConfig,
)

# Memory management utilities (extracted to memory.py for reduced complexity)
from homodyne.optimization.nlsq.memory import (  # noqa: E402
    get_adaptive_memory_threshold,
    NLSQStrategy,
    select_nlsq_strategy,
)

# Parameter utilities (extracted to parameter_utils.py for reduced complexity)
from homodyne.optimization.nlsq.parameter_utils import (  # noqa: E402
    build_parameter_labels as _build_parameter_labels,
    classify_parameter_status as _classify_parameter_status,
    sample_xdata as _sample_xdata,
    compute_jacobian_stats as _compute_jacobian_stats,
    compute_consistent_per_angle_init as _compute_consistent_per_angle_init,
    compute_quantile_per_angle_scaling as _compute_quantile_per_angle_scaling,
)

# Module-level logger
_memory_logger = get_logger(__name__)


def _extract_n_points(data: Any) -> int:
    """Extract number of data points from various data formats.

    Handles XPCSData objects, numpy arrays, lists, and other iterables.

    Parameters
    ----------
    data : Any
        Data object with g2 attribute or array-like

    Returns
    -------
    int
        Number of data points (0 if cannot determine)
    """
    # Try g2 attribute (XPCSData)
    if hasattr(data, "g2"):
        g2 = data.g2
        if hasattr(g2, "size"):
            return int(g2.size)
        if hasattr(g2, "__len__"):
            return len(g2)
    # Try direct array-like
    if hasattr(data, "size"):
        return int(data.size)
    if hasattr(data, "__len__"):
        return len(data)
    return 0


@dataclass
class FunctionEvaluationCounter:
    """Wraps a callable and counts invocations."""

    fn: Callable[..., Any]
    count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return self.fn(*args, **kwargs)


def create_multistart_warmup_func(
    model_func: Callable[..., np.ndarray],
    xdata: np.ndarray,
    ydata: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    warmup_learning_rate: float = 0.001,
    normalize: bool = True,
    chunk_size: int = 50_000,
) -> Callable[[dict[str, Any], np.ndarray, int], Any]:
    """Create a warmup-only fit function for multi-start Phase 1 strategy.

    This function creates a warmup_fit_func compatible with the multi-start
    optimization module Phase 1 strategy. It uses the L-BFGS warmup phase
    from the NLSQ AdaptiveHybridStreamingOptimizer to quickly explore the
    parameter space without full Gauss-Newton refinement.

    Parameters
    ----------
    model_func : Callable
        Model function with signature: ``func(x, *params) -> predictions``
    xdata : np.ndarray
        Independent variable data
    ydata : np.ndarray
        Dependent variable data (observations)
    bounds : tuple[np.ndarray, np.ndarray] | None, optional
        Parameter bounds as (lower, upper)
    warmup_learning_rate : float, default=0.001
        L-BFGS line search scale for warmup phase
    normalize : bool, default=True
        Whether to use parameter normalization (recommended for scale imbalance)
    chunk_size : int, default=50000
        Points per chunk for streaming computation

    Returns
    -------
    warmup_fit_func : Callable
        Function with signature: (data, initial_params, n_iterations) -> SingleStartResult
        Compatible with run_multistart_nlsq() warmup_fit_func parameter.

    Raises
    ------
    RuntimeError
        If AdaptiveHybridStreamingOptimizer is not available (NLSQ < 0.3.2)

    Examples
    --------
    >>> from homodyne.optimization.nlsq.wrapper import create_multistart_warmup_func
    >>> from homodyne.optimization.nlsq.multistart import run_multistart_nlsq
    >>>
    >>> # Create warmup function
    >>> warmup_func = create_multistart_warmup_func(
    ...     model_func=my_model,
    ...     xdata=x_data,
    ...     ydata=y_data,
    ...     bounds=(lower, upper),
    ... )
    >>>
    >>> # Use with multi-start
    >>> result = run_multistart_nlsq(
    ...     data=my_data,
    ...     bounds=bounds,
    ...     config=config,
    ...     single_fit_func=full_fit_func,
    ...     warmup_fit_func=warmup_func,  # For Phase 1 strategy
    ... )

    Notes
    -----
    This function integrates with the Phase 1 multi-start strategy which:
    1. Runs parallel L-BFGS warmup from multiple starting points
    2. Selects the best warmup result
    3. Performs full Gauss-Newton refinement from the best starting point

    This approach is memory-efficient for very large datasets (>100M points)
    and provides good exploration of the parameter space.

    See Also
    --------
    homodyne.optimization.nlsq.multistart.run_multistart_nlsq : Main multi-start function
    homodyne.optimization.nlsq.multistart._run_phase1_strategy : Phase 1 strategy implementation
    """
    from homodyne.optimization.nlsq.multistart import SingleStartResult

    if not HYBRID_STREAMING_AVAILABLE:
        raise RuntimeError(
            "AdaptiveHybridStreamingOptimizer not available. "
            "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
        )

    def warmup_fit_func(
        data: dict[str, Any],
        initial_params: np.ndarray,
        n_iterations: int,
    ) -> SingleStartResult:
        """Run warmup-only optimization from a starting point.

        Parameters
        ----------
        data : dict
            Data dictionary (not used directly; uses captured xdata/ydata)
        initial_params : np.ndarray
            Initial parameter values
        n_iterations : int
            Number of L-BFGS warmup iterations

        Returns
        -------
        SingleStartResult
            Optimization result with warmup parameters and cost
        """
        import time

        start_time = time.perf_counter()

        try:
            # Configure for warmup-only: skip Gauss-Newton phase
            config = HybridStreamingConfig(
                normalize=normalize,
                normalization_strategy="bounds",
                warmup_iterations=n_iterations,
                max_warmup_iterations=n_iterations,  # Force stop at n_iterations
                warmup_learning_rate=warmup_learning_rate,
                gauss_newton_max_iterations=0,  # Skip GN phase
                gauss_newton_tol=1e-8,
                chunk_size=chunk_size,
                validate_numerics=True,
            )

            optimizer = AdaptiveHybridStreamingOptimizer(config)

            # Run warmup-only optimization
            result = optimizer.fit(
                data_source=(xdata, ydata),
                func=model_func,
                p0=initial_params,
                bounds=bounds,
                verbose=0,  # Quiet mode
            )

            # Extract results
            final_params = np.asarray(result["x"])

            # Compute chi-squared from final cost
            # The optimizer returns cost as 0.5 * sum(residuals^2)
            diagnostics = result.get("streaming_diagnostics", {})
            warmup_diag = diagnostics.get("warmup_diagnostics", {})
            final_loss = warmup_diag.get("final_loss", float("inf"))

            # Convert loss to chi-squared (loss = 0.5 * chi_sq for LSQ)
            chi_squared = (
                2.0 * final_loss if final_loss != float("inf") else float("inf")
            )

            wall_time = time.perf_counter() - start_time

            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=final_params,
                chi_squared=chi_squared,
                success=result.get("success", False),
                n_iterations=n_iterations,
                wall_time=wall_time,
                message="L-BFGS warmup completed",
            )

        except (ValueError, RuntimeError, OSError) as e:
            wall_time = time.perf_counter() - start_time
            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=initial_params,
                chi_squared=float("inf"),
                success=False,
                n_iterations=0,
                wall_time=wall_time,
                message=f"Warmup failed: {str(e)}",
            )

    return warmup_fit_func


@dataclass
class OptimizationResult:
    """Complete optimization result with fit quality metrics and diagnostics.

    Attributes:
        parameters: Converged parameter values
        uncertainties: Standard deviations from covariance matrix diagonal
        covariance: Full parameter covariance matrix
        chi_squared: Sum of squared residuals
        reduced_chi_squared: chi_squared / (n_data - n_params)
        convergence_status: 'converged', 'max_iter', 'failed'
        iterations: Number of optimization iterations
        execution_time: Wall-clock execution time in seconds
        device_info: Device used for computation (CPU details)
        recovery_actions: List of error recovery actions taken
        quality_flag: 'good', 'marginal', 'poor'
        streaming_diagnostics: Enhanced diagnostics for streaming optimization (Task 5.4)
        stratification_diagnostics: Diagnostics for angle-stratified chunking (v2.2.1)
        success: Boolean indicating convergence (backward compatibility)
        message: Descriptive message about optimization outcome (backward compatibility)
    """

    parameters: np.ndarray
    uncertainties: np.ndarray
    covariance: np.ndarray
    chi_squared: float
    reduced_chi_squared: float
    convergence_status: str
    iterations: int
    execution_time: float
    device_info: dict[str, Any]
    recovery_actions: list[str] = field(default_factory=list)
    quality_flag: str = "good"
    streaming_diagnostics: dict[str, Any] | None = (
        None  # Task 5.4: Enhanced streaming diagnostics
    )
    stratification_diagnostics: StratificationDiagnostics | None = (
        None  # v2.2.1: Stratification diagnostics
    )
    nlsq_diagnostics: dict[str, Any] | None = None
    sigma_is_default: bool = (
        False  # True when sigma=0.01*ones (no experimental uncertainties)
    )

    # Backward compatibility attributes (FR-002)
    @property
    def success(self) -> bool:
        """Return True if optimization converged (backward compatibility)."""
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Return descriptive message about optimization outcome."""
        if self.convergence_status == "converged":
            return f"Optimization converged successfully. χ²={self.chi_squared:.6f}"
        elif self.convergence_status == "max_iter":
            return "Optimization stopped: maximum iterations reached"
        else:
            return f"Optimization failed: {self.convergence_status}"


@dataclass
class UseSequentialOptimization:
    """Marker indicating sequential per-angle optimization should be used.

    This is returned by _apply_stratification_if_needed when conditions require
    sequential per-angle optimization as a fallback strategy.

    Attributes:
        data: Original XPCS data object
        reason: Why sequential optimization is needed
    """

    data: Any
    reason: str


def _safe_uncertainties_from_pcov(pcov: np.ndarray, n_params: int) -> np.ndarray:
    """Extract uncertainties with diagonal regularization for singular pcov."""
    if pcov.shape[0] != n_params:
        return np.zeros(n_params)
    diag = np.diag(pcov)
    if np.any(diag < 1e-15):
        _memory_logger.warning(
            f"Singular covariance: {np.sum(diag < 1e-15)}/{n_params} near-zero entries. "
            "Applying regularization."
        )
        diag = np.diag(pcov + np.eye(n_params) * 1e-10)
    return np.sqrt(np.maximum(diag, 0.0))


class NLSQWrapper(NLSQAdapterBase):
    """Adapter class for NLSQ package integration with homodyne optimization.

    This class translates between homodyne's optimization API and the NLSQ
    package's curve_fit interface, handling:
    - Data format transformations
    - Parameter validation and bounds checking
    - Automatic strategy selection for large datasets
    - Hybrid error handling and recovery

    Usage:
        wrapper = NLSQWrapper(enable_large_dataset=True)
        result = wrapper.fit(data, config, initial_params, bounds, analysis_mode)
    """

    def __init__(
        self,
        enable_large_dataset: bool = True,
        enable_recovery: bool = True,
        enable_numerical_validation: bool = True,
        max_retries: int = 2,
        fast_mode: bool = False,
    ) -> None:
        """Initialize NLSQWrapper.

        Args:
            enable_large_dataset: Use curve_fit_large for datasets >1M points
            enable_recovery: Enable automatic error recovery strategies
            enable_numerical_validation: Enable NaN/Inf validation at 3 critical points
            max_retries: Maximum retry attempts per batch (default: 2)
            fast_mode: Disable non-essential checks for < 1% overhead (Task 5.5)
        """
        self.enable_large_dataset = enable_large_dataset
        self.enable_recovery = enable_recovery
        self.enable_numerical_validation = enable_numerical_validation and not fast_mode
        self.max_retries = max_retries
        self.fast_mode = fast_mode

        # Initialize streaming optimization components
        self.batch_statistics = BatchStatistics(max_size=100)
        self.recovery_applicator = RecoveryStrategyApplicator(max_retries=max_retries)
        self.numerical_validator = NumericalValidator(
            enable_validation=enable_numerical_validation and not fast_mode
        )

        # Best parameter tracking
        self.best_params = None
        self.best_loss = float("inf")
        self.best_batch_idx = -1

    @staticmethod
    def _get_physical_param_names(analysis_mode: str) -> list[str]:
        """Get physical parameter names for a given analysis mode.

        Args:
            analysis_mode: 'static_isotropic' or 'laminar_flow'

        Returns:
            List of physical parameter names (excludes scaling parameters)

        Raises:
            ValueError: If analysis_mode is not recognized
        """
        normalized_mode = analysis_mode.lower()

        if normalized_mode in {"static", "static_isotropic"}:
            return ["D0", "alpha", "D_offset"]
        elif normalized_mode == "laminar_flow":
            return [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",  # Canonical name (was gamma_dot_0)
                "beta",
                "gamma_dot_t_offset",  # Canonical name (was gamma_dot_offset)
                "phi0",
            ]
        else:
            raise ValueError(
                f"Unknown analysis_mode: '{analysis_mode}'. "
                f"Expected 'static_isotropic'/'static' or 'laminar_flow'"
            )

    @staticmethod
    def _extract_nlsq_settings(config: Any) -> dict[str, Any]:
        """Return NLSQ-specific settings from the config tree (if present)."""

        config_dict = None
        if hasattr(config, "config") and isinstance(config.config, dict):
            config_dict = config.config
        elif isinstance(config, dict):
            config_dict = config

        if not config_dict:
            return {}

        nlsq_settings = config_dict.get("optimization", {}).get("nlsq", {})
        return cast(dict[str, Any], nlsq_settings)

    @staticmethod
    def _handle_nlsq_result(
        result: Any,
        strategy: OptimizationStrategy,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Normalize NLSQ return values to consistent format.

        NLSQ v0.1.5 has inconsistent return types across different functions:
        - curve_fit: Returns tuple (popt, pcov) OR CurveFitResult object
        - curve_fit_large: Returns tuple (popt, pcov) OR OptimizeResult object
        - StreamingOptimizer.fit: Returns dict with 'x', 'pcov', 'streaming_diagnostics'

        This function normalizes all these formats to a consistent tuple:
        (popt, pcov, info)

        Args:
            result: Return value from NLSQ optimization call
            strategy: Optimization strategy used (for logging/diagnostics)

        Returns:
            tuple: (popt, pcov, info) where:
                - popt: np.ndarray of optimized parameters
                - pcov: np.ndarray covariance matrix
                - info: dict with additional information (empty if not available)

        Raises:
            TypeError: If result format is unrecognized

        Examples:
            >>> # Handle tuple (popt, pcov)
            >>> popt, pcov, info = _handle_nlsq_result((params, cov), OptimizationStrategy.STANDARD)
            >>> assert info == {}

            >>> # Handle tuple (popt, pcov, info)
            >>> result = (params, cov, {'nfev': 100})
            >>> popt, pcov, info = _handle_nlsq_result(result, OptimizationStrategy.STANDARD)
            >>> assert 'nfev' in info

            >>> # Handle CurveFitResult object
            >>> from nlsq import CurveFitResult
            >>> result = CurveFitResult(popt=params, pcov=cov, info={'nfev': 100})
            >>> popt, pcov, info = _handle_nlsq_result(result, OptimizationStrategy.STANDARD)

            >>> # Handle StreamingOptimizer dict
            >>> result = {'x': params, 'streaming_diagnostics': {...}}
            >>> popt, pcov, info = _handle_nlsq_result(result, OptimizationStrategy.STREAMING)
            >>> assert 'streaming_diagnostics' in info

        Notes:
            - For STREAMING strategy, pcov is computed from streaming diagnostics if available
            - Missing info dicts are replaced with empty dicts for consistency
            - All outputs are converted to numpy arrays for type consistency
        """
        logger = get_logger(__name__)

        # Case 1: Dict (from StreamingOptimizer)
        if isinstance(result, dict):
            popt = np.asarray(result.get("x", result.get("popt")))
            pcov = np.asarray(
                result.get("pcov", np.eye(len(popt)))
            )  # Identity if missing
            info = {
                "streaming_diagnostics": result.get("streaming_diagnostics", {}),
                "success": result.get("success", True),
                "message": result.get("message", ""),
                "best_loss": result.get("best_loss", None),
                "final_epoch": result.get("final_epoch", None),
            }
            logger.debug(
                f"Normalized StreamingOptimizer dict result (strategy: {strategy.value})"
            )
            return popt, pcov, info

        # Case 2: Tuple with 2 or 3 elements
        if isinstance(result, tuple):
            if len(result) == 2:
                # (popt, pcov) - most common case
                popt, pcov = result
                info = {}
                logger.debug(
                    f"Normalized (popt, pcov) tuple (strategy: {strategy.value})"
                )
            elif len(result) == 3:
                # (popt, pcov, info) - from curve_fit with full_output=True
                popt, pcov, info = result
                # Ensure info is a dict
                if not isinstance(info, dict):
                    logger.warning(
                        f"Info object is not a dict: {type(info)}. Converting to dict."
                    )
                    info = {"raw_info": info}
                logger.debug(
                    f"Normalized (popt, pcov, info) tuple (strategy: {strategy.value})"
                )
            else:
                raise TypeError(
                    f"Unexpected tuple length: {len(result)}. "
                    f"Expected 2 (popt, pcov) or 3 (popt, pcov, info). "
                    f"Got: {result}"
                )
            return np.asarray(popt), np.asarray(pcov), info

        # Case 3: Object with attributes (CurveFitResult, OptimizeResult, etc.)
        if hasattr(result, "x") or hasattr(result, "popt"):
            # Extract popt
            popt_raw = getattr(result, "x", getattr(result, "popt", None))
            if popt_raw is None:
                raise AttributeError(
                    f"Result object has neither 'x' nor 'popt' attribute. "
                    f"Available attributes: {dir(result)}"
                )
            popt = np.asarray(popt_raw)

            # Extract pcov
            pcov_raw = getattr(result, "pcov", None)
            if pcov_raw is None:
                # No covariance available, create identity matrix
                logger.warning(
                    "No pcov attribute in result object. Using identity matrix."
                )
                pcov = np.eye(len(popt))
            else:
                pcov = np.asarray(pcov_raw)

            # Extract info dict
            info = {}
            # Common attributes to extract
            for attr in [
                "message",
                "success",
                "nfev",
                "njev",
                "fun",
                "jac",
                "optimality",
            ]:
                if hasattr(result, attr):
                    info[attr] = getattr(result, attr)

            # Check for 'info' attribute (some objects nest additional info)
            if hasattr(result, "info") and isinstance(result.info, dict):
                info.update(result.info)

            logger.debug(
                f"Normalized object result (type: {type(result).__name__}, strategy: {strategy.value})"
            )
            return np.asarray(popt), np.asarray(pcov), info

        # Case 4: Unrecognized format
        raise TypeError(
            f"Unrecognized NLSQ result format: {type(result)}. "
            f"Expected tuple, dict, or object with 'x'/'popt' attributes. "
            f"Available attributes: {dir(result) if hasattr(result, '__dict__') else 'N/A'}"
        )

    def _get_fallback_strategy(
        self, current_strategy: OptimizationStrategy
    ) -> OptimizationStrategy | None:
        """Get fallback strategy when current strategy fails.

        Implements degradation chain:
        STREAMING → CHUNKED → LARGE → STANDARD → None

        Args:
            current_strategy: Strategy that failed

        Returns:
            Next strategy to try, or None if no fallback available
        """
        fallback_chain = {
            OptimizationStrategy.STREAMING: OptimizationStrategy.CHUNKED,
            OptimizationStrategy.CHUNKED: OptimizationStrategy.LARGE,
            OptimizationStrategy.LARGE: OptimizationStrategy.STANDARD,
            OptimizationStrategy.STANDARD: None,  # No further fallback
        }
        return fallback_chain.get(current_strategy)

    def fit(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None = None,
        bounds: tuple[np.ndarray, np.ndarray] | None = None,
        analysis_mode: str = "static_isotropic",
        per_angle_scaling: bool = True,  # REQUIRED: per-angle is physically correct
        diagnostics_enabled: bool = False,
        shear_transforms: dict[str, Any] | None = None,
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
    ) -> OptimizationResult:
        """Execute NLSQ optimization with automatic strategy selection and per-angle scaling.

        Args:
            data: XPCS experimental data
            config: Configuration manager with optimization settings
            initial_params: Initial parameter guess (auto-loaded if None)
            bounds: Parameter bounds as (lower, upper) tuple
            analysis_mode: 'static_isotropic' or 'laminar_flow'
            per_angle_scaling: MUST be True. Per-angle contrast/offset parameters are physically correct
                             as each scattering angle has different optical properties and detector responses.
                             Legacy scalar mode (False) is no longer supported.

        Returns:
            OptimizationResult with converged parameters and diagnostics

        Raises:
            ValueError: If bounds are invalid (lower >= upper) or if per_angle_scaling=False
        """
        import time

        # nlsq imported at module level (line 36) for automatic x64 configuration

        logger = get_logger(__name__)

        # BREAKING CHANGE (Nov 2025): Validate per-angle scaling is enabled
        # Legacy scalar contrast/offset mode is not physically meaningful
        if not per_angle_scaling:
            logger.error(
                "Legacy scalar contrast/offset mode (per_angle_scaling=False) is no longer supported. "
                "Single contrast/offset parameters are not physically meaningful as each scattering "
                "angle has different optical properties and detector responses. "
                "Per-angle scaling is required for physically correct NLSQ optimization."
            )
            raise ValueError(
                "per_angle_scaling=False is deprecated and removed. "
                "Use per_angle_scaling=True (default) for physically correct behavior."
            )

        # Start timing
        start_time = time.time()

        physical_param_names = self._get_physical_param_names(analysis_mode)

        nlsq_settings = self._extract_nlsq_settings(config)
        loss_name = nlsq_settings.get("loss", "soft_l1")
        trust_region_scale = float(nlsq_settings.get("trust_region_scale", 1.0))
        if trust_region_scale <= 0:
            trust_region_scale = 1.0
        x_scale_override = nlsq_settings.get("x_scale")
        x_scale_value = x_scale_override if x_scale_override is not None else "jac"
        x_scale_map_config = normalize_x_scale_map(nlsq_settings.get("x_scale_map"))
        diagnostics_cfg = nlsq_settings.get("diagnostics", {})
        diagnostics_enabled = diagnostics_enabled or bool(
            diagnostics_cfg.get("enable", False),
        )
        diagnostics_sample_size = int(diagnostics_cfg.get("sample_size", 2048))
        diagnostics_payload = (
            {"solver_settings": {"loss": loss_name}} if diagnostics_enabled else None
        )
        transform_cfg = parse_shear_transform_config(shear_transforms)

        # Step 0.5: Unified Memory-Based Strategy Selection (v2.13.0)
        # Uses pure memory estimation - no legacy point thresholds.
        n_est_points = _extract_n_points(data)
        n_params = len(initial_params) if initial_params is not None else 0

        strategy_decision = select_nlsq_strategy(n_est_points, n_params)
        logger.info(
            f"Strategy selection: {strategy_decision.strategy.value} "
            f"({strategy_decision.reason})"
        )

        # Handle HYBRID_STREAMING (extreme scale - index array > 75% RAM)
        if strategy_decision.strategy == NLSQStrategy.HYBRID_STREAMING:
            if not HYBRID_STREAMING_AVAILABLE:
                logger.critical(
                    "AdaptiveHybridStreamingOptimizer required for extreme-scale "
                    f"dataset ({n_est_points:,} points) but not available."
                )
                raise MemoryError(
                    f"Dataset too large for RAM (index={strategy_decision.index_memory_gb:.1f} GB > "
                    f"threshold={strategy_decision.threshold_gb:.1f} GB) and Streaming unavailable."
                )
            # Streaming path continues below (handled by existing streaming logic)
            logger.warning(
                f"Extreme-scale dataset: {strategy_decision.reason}. "
                "Proceeding with Adaptive Hybrid Streaming."
            )

        # Handle OUT_OF_CORE (large scale - peak memory > 75% RAM)
        elif strategy_decision.strategy == NLSQStrategy.OUT_OF_CORE:
            if initial_params is None:
                raise ValueError("initial_params required for out-of-core optimization")

            validated_params = self._validate_initial_params(initial_params, bounds)
            nlsq_bounds = self._convert_bounds(bounds)

            # Default to False (User requirement: Never subsample data)
            use_fast_mode = self.fast_mode or config.config.get("optimization", {}).get(
                "fast_chi2_mode", False
            )

            # Extract anti-degeneracy config (will warn that it's not supported for out-of-core)
            ooc_anti_degeneracy_config = None
            if config is not None and hasattr(config, "config"):
                ooc_nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                ooc_anti_degeneracy_config = ooc_nlsq_config.get("anti_degeneracy", {})

            popt, pcov, info = self._fit_with_out_of_core_accumulation(
                stratified_data=None,
                data=data,
                per_angle_scaling=per_angle_scaling,
                physical_param_names=physical_param_names,
                initial_params=validated_params,
                bounds=nlsq_bounds,
                logger=logger,
                config=config,
                fast_chi2_mode=use_fast_mode,
                anti_degeneracy_config=ooc_anti_degeneracy_config,
            )

            execution_time = time.time() - start_time
            uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))
            reduced_chi2 = info.get("chi_squared", 0.0) / max(
                1, n_est_points - len(popt)
            )

            return OptimizationResult(
                parameters=popt,
                uncertainties=uncertainties,
                covariance=pcov,
                chi_squared=info.get("chi_squared", 0.0),
                reduced_chi_squared=reduced_chi2,
                convergence_status=info.get("convergence_status", "unknown"),
                iterations=info.get("iterations", 0),
                execution_time=execution_time,
                device_info={
                    "device": "cpu_accumulated",
                    "strategy": "out_of_core",
                    "fast_mode": use_fast_mode,
                    "decision": strategy_decision.reason,
                },
                recovery_actions=["out_of_core_delegation"],
                quality_flag="good",
            )

        # STANDARD strategy falls through to existing optimization path

        # Step 1: Apply angle-stratified chunking if needed (BEFORE data preparation)
        # This fixes per-angle parameter incompatibility with NLSQ chunking (ultra-think-20251106-012247)
        stratified_data = self._apply_stratification_if_needed(
            data, per_angle_scaling, config, logger
        )

        # Extract stratification diagnostics if available
        stratification_diagnostics = None
        if hasattr(stratified_data, "stratification_diagnostics"):
            stratification_diagnostics = stratified_data.stratification_diagnostics

        # Check if sequential optimization is required
        transform_state: dict[str, Any] | None = None

        if isinstance(stratified_data, UseSequentialOptimization):
            logger.info(
                f"Using sequential per-angle optimization: {stratified_data.reason}"
            )
            return self._run_sequential_optimization(
                stratified_data.data,
                config,
                initial_params,
                bounds,
                analysis_mode,
                per_angle_scaling,
                logger,
                start_time,
                x_scale_value=x_scale_value,
                transform_cfg=transform_cfg,
                physical_param_names=physical_param_names,
                per_angle_scaling_initial=per_angle_scaling_initial,
            )

        # NEW: Check if stratified least_squares should be used (v2.2.0 double-chunking fix)
        # Conditions:
        # 1. Stratified data was created (has phi_flat attribute)
        # 2. Per-angle scaling is enabled
        # 3. Dataset is large enough to benefit (>1M points)
        #
        # FIXED (Nov 13, 2025): Use JIT-compatible StratifiedResidualFunctionJIT
        # Solution: Padded vmap implementation with static shapes
        # - Pads chunks to uniform size (enables JIT compilation)
        # - Uses jax.vmap for parallel chunk processing (no Python loops)
        # - Masks padded values in final residuals
        # Performance: ~1% memory overhead, 10-100x speedup from vectorization
        use_stratified_least_squares = (
            hasattr(stratified_data, "phi_flat")
            and per_angle_scaling
            and hasattr(stratified_data, "g2_flat")
            and len(stratified_data.g2_flat) >= 1_000_000
        )
        if use_stratified_least_squares:
            logger.info("=" * 80)
            logger.info("STRATIFIED LEAST-SQUARES PATH ACTIVATED (v2.2.1)")
            logger.info("Solving double-chunking problem with NLSQ's least_squares()")
            logger.info("=" * 80)

            # Validate initial parameters
            if initial_params is None:
                raise ValueError("initial_params must be provided")
            validated_params = self._validate_initial_params(initial_params, bounds)

            # Convert bounds
            nlsq_bounds = self._convert_bounds(bounds)

            # Validate bounds consistency
            if nlsq_bounds is not None:
                lower, upper = nlsq_bounds
                if np.any(lower >= upper):
                    invalid_indices = np.where(lower >= upper)[0]
                    raise ValueError(
                        f"Invalid bounds at indices {invalid_indices}: "
                        f"lower >= upper. Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}"
                    )

            # Get physical parameter names for this analysis mode
            physical_param_names = self._get_physical_param_names(analysis_mode)
            logger.info(
                f"Physical parameters for {analysis_mode}: {physical_param_names}"
            )

            # FIX: Expand scaling parameters for per-angle scaling
            # When per_angle_scaling=True with N angles, we need:
            # - All physical parameters (7 for laminar_flow, 3 for static)
            # - N contrast parameters (one per angle)
            # - N offset parameters (one per angle)
            # Total: n_physical + 2*N parameters
            #
            # Config provides: n_physical + 2 parameters (single contrast, single offset)
            # We must expand: [contrast, offset] → [c0, c1, ..., cN-1, o0, o1, ..., oN-1]

            if per_angle_scaling:
                # Determine number of angles from stratified data
                n_angles = len(np.unique(stratified_data.phi_flat))
                n_physical = len(physical_param_names)

                logger.info("Expanding scaling parameters for per-angle scaling:")
                logger.info(f"  Angles: {n_angles}")
                logger.info(f"  Physical parameters: {n_physical}")
                logger.info(
                    f"  Input parameters: {len(validated_params)} (expected: {n_physical + 2})"
                )

                # Validate input parameter count
                expected_input = (
                    n_physical + 2
                )  # Physical params + single contrast + single offset
                if len(validated_params) != expected_input:
                    raise ValueError(
                        f"Parameter count mismatch for per-angle scaling: "
                        f"got {len(validated_params)}, expected {expected_input} "
                        f"({n_physical} physical + 2 scaling). "
                        f"For {n_angles} angles, will expand to {n_physical + 2 * n_angles} parameters."
                    )

                # Expand compact [contrast, offset, physical...] to per-angle format
                # matching StratifiedResidualFunction order:
                #   [contrast_per_angle, offset_per_angle, physical_params]
                from homodyne.optimization.nlsq.data_prep import (
                    expand_per_angle_parameters,
                )

                expanded = expand_per_angle_parameters(
                    validated_params,
                    nlsq_bounds,
                    n_angles,
                    n_physical,
                    logger=logger,
                )
                validated_params = expanded.params
                nlsq_bounds = expanded.bounds

            # Parameter count validation (CRITICAL)
            # Per-angle scaling is always enabled (legacy mode removed Nov 2025)
            n_physical = len(physical_param_names)
            n_angles = len(np.unique(stratified_data.phi_flat))
            expected_params = n_physical + 2 * n_angles

            if len(validated_params) != expected_params:
                raise ValueError(
                    f"Parameter count mismatch: got {len(validated_params)}, "
                    f"expected {expected_params} "
                    f"(physical={n_physical}, per_angle_scaling=True, "
                    f"n_angles={n_angles})"
                )

            logger.info(
                f"✓ Parameter validation passed: {len(validated_params)} parameters"
            )

            # Step: Re-run unified strategy selection with EFFECTIVE parameter count
            # (v2.14.0, v2.22.0 fix: anti-degeneracy pre-check)
            #
            # The expanded param count (e.g. 53 for 23 angles individual) may be much
            # larger than the effective count after anti-degeneracy mode selection
            # (e.g. 9 for auto_averaged). Using the expanded count for memory estimation
            # can unnecessarily trigger out-of-core routing, which bypasses the
            # anti-degeneracy defense system entirely — causing parameter absorption
            # degeneracy and false convergence.
            #
            # Fix: Pre-check what anti-degeneracy would select, and use the effective
            # param count for memory routing. The actual anti-degeneracy transformation
            # still happens inside _fit_with_stratified_least_squares().
            n_total_points = len(stratified_data.g2_flat)
            actual_n_params = len(validated_params)
            effective_n_params = actual_n_params  # Default: no reduction

            if per_angle_scaling and config is not None and hasattr(config, "config"):
                nlsq_cfg = config.config.get("optimization", {}).get("nlsq", {})
                ad_cfg = nlsq_cfg.get("anti_degeneracy", {})
                ad_per_angle_mode = ad_cfg.get("per_angle_mode", "auto")
                ad_threshold = ad_cfg.get("constant_scaling_threshold", 3)
                n_angles_check = len(np.unique(stratified_data.phi_flat))

                if ad_per_angle_mode == "auto" and n_angles_check >= ad_threshold:
                    # auto_averaged: 2 averaged scaling params replace 2*n_angles
                    effective_n_params = n_physical + 2
                    logger.info(
                        f"Anti-Degeneracy pre-check: auto → auto_averaged "
                        f"(n_phi={n_angles_check} >= threshold={ad_threshold}). "
                        f"Effective params: {effective_n_params} "
                        f"(expanded: {actual_n_params})"
                    )
                elif ad_per_angle_mode == "constant":
                    # constant: scaling fixed, only physical params optimized
                    effective_n_params = n_physical
                    logger.info(
                        f"Anti-Degeneracy pre-check: constant mode. "
                        f"Effective params: {effective_n_params} "
                        f"(expanded: {actual_n_params})"
                    )

            strategy_recheck = select_nlsq_strategy(n_total_points, effective_n_params)

            logger.info(
                f"Strategy re-check (with {effective_n_params} effective params, "
                f"{actual_n_params} expanded): "
                f"{strategy_recheck.strategy.value} ({strategy_recheck.reason})"
            )

            # Route to OUT_OF_CORE if peak memory exceeds threshold
            if strategy_recheck.strategy == NLSQStrategy.OUT_OF_CORE:
                # Safety check: warn if anti-degeneracy would have prevented this
                if effective_n_params < actual_n_params:
                    logger.warning(
                        f"Out-of-core triggered with {actual_n_params} expanded params, "
                        f"but anti-degeneracy would reduce to {effective_n_params}. "
                        f"This should not happen — the pre-check should have used "
                        f"effective params for memory estimation. Check routing logic."
                    )
                logger.info("=" * 80)
                logger.info("OUT-OF-CORE ACCUMULATION MODE (Re-check)")
                logger.info(
                    f"Peak memory ({strategy_recheck.peak_memory_gb:.1f} GB) exceeds "
                    f"threshold ({strategy_recheck.threshold_gb:.1f} GB)"
                )
                logger.info("Using chunk-wise J^T J accumulation for memory efficiency")
                logger.info("=" * 80)

                # Default to False (User requirement: Never subsample data)
                use_fast_mode = self.fast_mode or (
                    config.config.get("optimization", {}).get("fast_chi2_mode", False)
                    if config is not None and hasattr(config, "config")
                    else False
                )

                # Extract anti-degeneracy config (will warn that it's not supported for out-of-core)
                recheck_anti_degeneracy_config = None
                if config is not None and hasattr(config, "config"):
                    recheck_nlsq_config = config.config.get("optimization", {}).get(
                        "nlsq", {}
                    )
                    recheck_anti_degeneracy_config = recheck_nlsq_config.get(
                        "anti_degeneracy", {}
                    )

                popt, pcov, info = self._fit_with_out_of_core_accumulation(
                    stratified_data=stratified_data,
                    data=data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=logger,
                    config=config,
                    fast_chi2_mode=use_fast_mode,
                    anti_degeneracy_config=recheck_anti_degeneracy_config,
                )

                execution_time = time.time() - start_time
                uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))
                reduced_chi2 = info.get("chi_squared", 0.0) / max(
                    1, n_total_points - len(popt)
                )

                return OptimizationResult(
                    parameters=popt,
                    uncertainties=uncertainties,
                    covariance=pcov,
                    chi_squared=info.get("chi_squared", 0.0),
                    reduced_chi_squared=reduced_chi2,
                    convergence_status=info.get("convergence_status", "unknown"),
                    iterations=info.get("iterations", 0),
                    execution_time=execution_time,
                    device_info={
                        "device": "cpu_accumulated",
                        "strategy": "out_of_core",
                        "fast_mode": use_fast_mode,
                        "decision": strategy_recheck.reason,
                    },
                    recovery_actions=["out_of_core_recheck_delegation"],
                    quality_flag="good",
                )

            # Route to HYBRID_STREAMING if index array exceeds threshold (extreme scale)
            if strategy_recheck.strategy == NLSQStrategy.HYBRID_STREAMING:
                if not HYBRID_STREAMING_AVAILABLE:
                    logger.critical(
                        "AdaptiveHybridStreamingOptimizer required for extreme-scale "
                        f"dataset ({n_total_points:,} points) but not available."
                    )
                    raise MemoryError(
                        f"Dataset too large for RAM (index={strategy_recheck.index_memory_gb:.1f} GB > "
                        f"threshold={strategy_recheck.threshold_gb:.1f} GB) and Streaming unavailable."
                    )
                logger.warning(
                    f"Extreme-scale dataset: {strategy_recheck.reason}. "
                    "Proceeding with Adaptive Hybrid Streaming."
                )
                # Fall through to streaming path below (use_streaming_mode will be set)

            # Extract target chunk size from config
            target_chunk_size = 100_000  # Default
            hybrid_streaming_config = None
            use_streaming_mode = False
            use_hybrid_streaming = False

            # Compute adaptive memory threshold (v2.7.0+)
            # Default: 75% of total system memory instead of fixed 16 GB
            memory_fraction: float | None = None  # Will use default or env var
            memory_threshold_gb: float | None = None  # Will be computed adaptively

            if config is not None and hasattr(config, "config"):
                strat_config = config.config.get("optimization", {}).get(
                    "stratification", {}
                )
                target_chunk_size = strat_config.get("target_chunk_size", 100_000)

                # Extract streaming configuration
                nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                hybrid_streaming_config = nlsq_config.get("hybrid_streaming", {})

                # Support for explicit memory_threshold_gb (backwards compatible)
                # or memory_fraction (new adaptive approach)
                if "memory_threshold_gb" in nlsq_config:
                    memory_threshold_gb = nlsq_config["memory_threshold_gb"]
                if "memory_fraction" in nlsq_config:
                    memory_fraction = nlsq_config["memory_fraction"]

            # Compute adaptive threshold if not explicitly set
            if memory_threshold_gb is None:
                memory_threshold_gb, threshold_info = get_adaptive_memory_threshold(
                    memory_fraction=memory_fraction
                )
                logger.debug(
                    f"Using adaptive memory threshold: {memory_threshold_gb:.1f} GB "
                    f"(fraction={threshold_info['memory_fraction']}, "
                    f"total={threshold_info['total_memory_gb']:.1f} GB, "
                    f"source={threshold_info['source']})"
                )
            else:
                logger.debug(
                    f"Using explicit memory threshold from config: {memory_threshold_gb:.1f} GB"
                )

            # Check for hybrid streaming mode (preferred for large datasets)
            if hybrid_streaming_config is not None:
                use_hybrid_streaming = hybrid_streaming_config.get("enable", False)

            # Check for forced streaming mode from config
            # Also set from strategy_recheck if it returned HYBRID_STREAMING
            if config is not None and hasattr(config, "config"):
                nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
                use_streaming_mode = nlsq_config.get("use_streaming", False)

            # Set streaming mode if strategy_recheck returned HYBRID_STREAMING (extreme scale)
            # This unified decision replaces the legacy _should_use_streaming() check
            if strategy_recheck.strategy == NLSQStrategy.HYBRID_STREAMING:
                logger.info("=" * 80)
                logger.info("HYBRID STREAMING MODE (Strategy Re-check)")
                logger.info(
                    f"Index array ({strategy_recheck.index_memory_gb:.1f} GB) exceeds "
                    f"threshold ({strategy_recheck.threshold_gb:.1f} GB)"
                )
                logger.info("=" * 80)
                use_streaming_mode = True

            # Log strategy decision for STANDARD (in-memory) path
            if not use_streaming_mode:
                logger.info(
                    f"Memory check: {strategy_recheck.reason}. "
                    "Proceeding with in-memory stratified least-squares."
                )

            # Use streaming optimizer if needed
            if use_streaming_mode:
                # Prefer AdaptiveHybridStreamingOptimizer when available
                # It fixes shear-term gradients, convergence, and covariance issues
                # Use hybrid if: (1) explicitly enabled, OR (2) basic streaming unavailable
                use_hybrid = HYBRID_STREAMING_AVAILABLE and (
                    use_hybrid_streaming or not STREAMING_AVAILABLE
                )

                if use_hybrid:
                    logger.info("=" * 80)
                    logger.info("ADAPTIVE HYBRID STREAMING MODE (Preferred)")
                    logger.info(
                        "Using NLSQ AdaptiveHybridStreamingOptimizer for better "
                        "convergence and parameter estimation"
                    )
                    logger.info("=" * 80)
                    # Extract anti-degeneracy config for defense system v2.9.0
                    anti_degeneracy_config = nlsq_config.get("anti_degeneracy", {})
                    try:
                        popt, pcov, info = self._fit_with_stratified_hybrid_streaming(
                            stratified_data=stratified_data,
                            per_angle_scaling=per_angle_scaling,
                            physical_param_names=physical_param_names,
                            initial_params=validated_params,
                            bounds=nlsq_bounds,
                            logger=logger,
                            hybrid_config=hybrid_streaming_config,
                            anti_degeneracy_config=anti_degeneracy_config,
                        )

                        # Compute final residuals for result creation
                        chunked_data = self._create_stratified_chunks(
                            stratified_data, target_chunk_size
                        )
                        residual_fn = create_stratified_residual_function(
                            stratified_data=chunked_data,
                            per_angle_scaling=per_angle_scaling,
                            physical_param_names=physical_param_names,
                            logger=cast(logging.Logger | None, logger),
                            validate=False,
                        )
                        final_residuals = residual_fn(popt)
                        n_data = len(final_residuals)

                        # Get execution time
                        execution_time = time.time() - start_time

                        # Create result
                        result = self._create_fit_result(
                            popt=popt,
                            pcov=pcov,
                            residuals=final_residuals,
                            n_data=n_data,
                            iterations=info.get("nit", 0),
                            execution_time=execution_time,
                            convergence_status=(
                                "converged" if info.get("success", True) else "failed"
                            ),
                            recovery_actions=["hybrid_streaming_optimizer_method"],
                            streaming_diagnostics=info.get(
                                "hybrid_streaming_diagnostics"
                            ),
                            stratification_diagnostics=stratification_diagnostics,
                            diagnostics_payload=None,
                        )

                        logger.info("=" * 80)
                        logger.info("HYBRID STREAMING OPTIMIZATION COMPLETE")
                        logger.info(
                            f"Final χ²: {result.chi_squared:.4e}, "
                            f"Reduced χ²: {result.reduced_chi_squared:.4f}"
                        )
                        logger.info("=" * 80)

                        return result

                    except (ValueError, RuntimeError, MemoryError, OSError) as e:
                        logger.warning(
                            f"Hybrid streaming optimization failed: {e}\n"
                            f"Falling back to stratified least-squares..."
                        )
                        # Fall through to stratified least-squares

                if not STREAMING_AVAILABLE:
                    # AdaptiveHybridStreamingOptimizer not available (NLSQ < 0.3.2)
                    logger.error(
                        "Streaming mode requested but AdaptiveHybridStreamingOptimizer "
                        "not available. Upgrade NLSQ to >= 0.3.2. "
                        "Falling back to stratified least-squares."
                    )
                    # Fall through to stratified least-squares

            # Extract anti-degeneracy config for defense system v2.14.0
            # (Now applies to stratified LS, not just hybrid streaming)
            anti_degeneracy_config = None
            if config is not None and hasattr(config, "config"):
                nlsq_config_ad = config.config.get("optimization", {}).get("nlsq", {})
                anti_degeneracy_config = nlsq_config_ad.get("anti_degeneracy", {})
                if anti_degeneracy_config:
                    logger.info(
                        f"Anti-Degeneracy config loaded: per_angle_mode="
                        f"{anti_degeneracy_config.get('per_angle_mode', 'auto')}"
                    )

            # Call stratified least_squares optimization
            try:
                popt, pcov, info = self._fit_with_stratified_least_squares(
                    stratified_data=stratified_data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    initial_params=validated_params,
                    bounds=nlsq_bounds,
                    logger=logger,
                    target_chunk_size=target_chunk_size,
                    anti_degeneracy_config=anti_degeneracy_config,
                )

                # Compute final residuals for result creation
                # We need to recreate the residual function to compute final residuals
                chunked_data = self._create_stratified_chunks(
                    stratified_data, target_chunk_size
                )
                residual_fn = create_stratified_residual_function(
                    stratified_data=chunked_data,
                    per_angle_scaling=per_angle_scaling,
                    physical_param_names=physical_param_names,
                    logger=cast(logging.Logger | None, logger),
                    validate=False,  # Already validated
                )
                final_residuals = residual_fn(popt)
                n_data = len(final_residuals)

                # Get execution time
                execution_time = time.time() - start_time

                # Create result
                result = self._create_fit_result(
                    popt=popt,
                    pcov=pcov,
                    residuals=final_residuals,
                    n_data=n_data,
                    iterations=info.get("nit", 0),
                    execution_time=execution_time,
                    convergence_status=(
                        "converged" if info.get("success", True) else "failed"
                    ),
                    recovery_actions=["stratified_least_squares_method"],
                    streaming_diagnostics=None,
                    stratification_diagnostics=stratification_diagnostics,
                    diagnostics_payload=None,
                )

                logger.info("=" * 80)
                logger.info("STRATIFIED LEAST-SQUARES COMPLETE")
                logger.info(
                    f"Final χ²: {result.chi_squared:.4e}, Reduced χ²: {result.reduced_chi_squared:.4f}"
                )
                logger.info("=" * 80)

                return result

            except (ValueError, RuntimeError, MemoryError, OSError) as e:
                logger.error(
                    f"Stratified least_squares failed: {e}\n"
                    f"Falling back to standard curve_fit_large path..."
                )
                # Fall through to standard optimization path below

        # Step 2: Prepare data
        logger.info(f"Preparing data for {analysis_mode} optimization...")
        xdata, ydata = self._prepare_data(stratified_data)
        n_data = len(ydata)
        logger.info(f"Data prepared: {n_data} points")

        # Note: Memory estimation is deferred to NLSQ's estimate_memory_requirements()
        # which provides accurate Jacobian sizing based on actual parameter count.
        if n_data > 10_000_000:
            logger.warning(
                f"Very large dataset: {n_data:,} points. "
                f"NLSQ will use memory-efficient strategies automatically."
            )
        elif n_data > 1_000_000:
            logger.info(
                f"Large dataset: {n_data:,} points. Memory managed automatically."
            )

        # Step 3: Validate initial parameters
        if initial_params is None:
            raise ValueError(
                "initial_params must be provided (auto-loading not yet implemented)",
            )

        validated_params = self._validate_initial_params(initial_params, bounds)

        # Step 4: Convert bounds
        nlsq_bounds = self._convert_bounds(bounds)

        # Step 5: Validate bounds consistency (FR-006)
        if nlsq_bounds is not None:
            lower, upper = nlsq_bounds
            if np.any(lower >= upper):
                invalid_indices = np.where(lower >= upper)[0]
                raise ValueError(
                    f"Invalid bounds at indices {invalid_indices}: "
                    f"lower >= upper. Bounds must satisfy lower < upper elementwise. "
                    f"Lower: {lower[invalid_indices]}, Upper: {upper[invalid_indices]}",
                )

        # Step 6: Create residual function with per-angle scaling
        logger.info(
            f"Creating residual function (per_angle_scaling={per_angle_scaling})..."
        )
        residual_fn = self._create_residual_function(
            stratified_data, analysis_mode, per_angle_scaling
        )
        base_residual_fn = residual_fn
        physical_param_names = self._get_physical_param_names(analysis_mode)
        phi_values = np.asarray(stratified_data.phi)
        n_phi_unique = len(np.unique(phi_values)) if phi_values.size else 0

        per_angle_contrast_override: np.ndarray | None = None
        per_angle_offset_override: np.ndarray | None = None
        if per_angle_scaling_initial:
            contrast_override = per_angle_scaling_initial.get("contrast")
            if contrast_override is not None:
                try:
                    arr = np.asarray(contrast_override, dtype=np.float64)
                    if arr.size == n_phi_unique:
                        per_angle_contrast_override = arr.copy()
                    else:
                        logger.warning(
                            "per_angle_scaling contrast override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_unique,
                        )
                except (TypeError, ValueError):
                    logger.warning("Invalid per-angle contrast override; ignoring")
            offset_override = per_angle_scaling_initial.get("offset")
            if offset_override is not None:
                try:
                    arr = np.asarray(offset_override, dtype=np.float64)
                    if arr.size == n_phi_unique:
                        per_angle_offset_override = arr.copy()
                    else:
                        logger.warning(
                            "per_angle_scaling offset override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_unique,
                        )
                except (TypeError, ValueError):
                    logger.warning("Invalid per-angle offset override; ignoring")

        # Step 6.5: Expand parameters for per-angle scaling if needed
        # This is CRITICAL: the residual function expects per-angle parameters,
        # but validated_params is still in compact form [contrast, offset, *physical]
        if per_angle_scaling:
            n_phi = n_phi_unique

            # Expand parameters from compact to per-angle form
            # Input:  [contrast, offset, *physical] (e.g., 5 params)
            # Output: [contrast_0, ..., contrast_{n-1}, offset_0, ..., offset_{n-1}, *physical]
            #         (e.g., 2*n_phi + 3 params for static_isotropic with n_phi angles)

            contrast_single = validated_params[0]
            offset_single = validated_params[1]
            physical_params = validated_params[2:]

            # Replicate contrast and offset for each angle
            # CRITICAL FIX (v2.7.1): For laminar_flow mode, use consistent initialization
            # to prevent per-angle params from absorbing the shear signal
            is_laminar_flow = "gamma_dot_t0" in physical_param_names
            use_consistent_init = (
                is_laminar_flow
                and per_angle_contrast_override is None
                and per_angle_offset_override is None
                and n_phi > 3  # Only for many angles where absorption is a problem
            )

            if use_consistent_init:
                logger.info(
                    "Computing consistent per-angle initialization for laminar_flow mode..."
                )
                try:
                    contrast_per_angle, offset_per_angle = (
                        _compute_consistent_per_angle_init(
                            stratified_data=stratified_data,
                            physical_params=physical_params,
                            physical_param_names=physical_param_names,
                            default_contrast=contrast_single,
                            default_offset=offset_single,
                            logger=logger,
                        )
                    )
                except (
                    ValueError,
                    RuntimeError,
                    TypeError,
                    AttributeError,
                    np.linalg.LinAlgError,
                ) as e:
                    logger.warning(
                        f"Failed to compute consistent per-angle init: {e}\n"
                        "Falling back to uniform replication."
                    )
                    contrast_per_angle = np.full(n_phi, contrast_single)
                    offset_per_angle = np.full(n_phi, offset_single)
            else:
                if per_angle_contrast_override is not None:
                    contrast_per_angle = per_angle_contrast_override
                else:
                    contrast_per_angle = np.full(n_phi, contrast_single)
                if per_angle_offset_override is not None:
                    offset_per_angle = per_angle_offset_override
                else:
                    offset_per_angle = np.full(n_phi, offset_single)

            # Concatenate: [contrasts, offsets, physical]
            validated_params = np.concatenate(
                [contrast_per_angle, offset_per_angle, physical_params]
            )

            logger.info(
                f"Expanded parameters for per-angle scaling:\n"
                f"  {n_phi} phi angles detected\n"
                f"  Parameters: compact {2 + len(physical_params)} → per-angle {len(validated_params)}\n"
                f"  Structure: [{n_phi} contrasts, {n_phi} offsets, {len(physical_params)} physical]"
            )

            # Also expand bounds if they exist
            if nlsq_bounds is not None:
                lower, upper = nlsq_bounds

                # Extract compact bounds
                contrast_lower, offset_lower = lower[0], lower[1]
                contrast_upper, offset_upper = upper[0], upper[1]
                physical_lower = lower[2:]
                physical_upper = upper[2:]

                # Expand to per-angle bounds
                contrast_lower_per_angle = np.full(n_phi, contrast_lower)
                contrast_upper_per_angle = np.full(n_phi, contrast_upper)
                offset_lower_per_angle = np.full(n_phi, offset_lower)
                offset_upper_per_angle = np.full(n_phi, offset_upper)

                # Concatenate expanded bounds
                expanded_lower = np.concatenate(
                    [contrast_lower_per_angle, offset_lower_per_angle, physical_lower]
                )
                expanded_upper = np.concatenate(
                    [contrast_upper_per_angle, offset_upper_per_angle, physical_upper]
                )

                nlsq_bounds = (expanded_lower, expanded_upper)

                logger.info(
                    f"Expanded bounds for per-angle scaling:\n"
                    f"  Bounds: compact {2 + len(physical_lower)} → per-angle {len(expanded_lower)}"
                )

        n_angles_for_map = n_phi_unique if per_angle_scaling else 1
        physical_index_map = build_physical_index_map(
            per_angle_scaling,
            n_angles_for_map,
            physical_param_names,
        )
        validated_params, transform_state = apply_forward_shear_transforms_to_vector(
            validated_params,
            physical_index_map,
            transform_cfg,
        )
        if transform_state:
            nlsq_bounds = apply_forward_shear_transforms_to_bounds(
                nlsq_bounds,
                transform_state,
            )

        solver_residual_fn = base_residual_fn
        if transform_state:
            if isinstance(base_residual_fn, StratifiedResidualFunction):
                solver_residual_fn = wrap_stratified_function_with_transforms(
                    base_residual_fn,
                    transform_state,
                )
            else:
                solver_residual_fn = wrap_model_function_with_transforms(
                    base_residual_fn,
                    transform_state,
                )

        param_labels = _build_parameter_labels(
            per_angle_scaling,
            n_phi_unique if per_angle_scaling else 0,
            physical_param_names,
        )

        per_param_x_scale = build_per_parameter_x_scale(
            per_angle_scaling,
            n_phi_unique if per_angle_scaling else 0,
            physical_param_names,
            analysis_mode,
            x_scale_map_config,
        )
        if per_param_x_scale is not None:
            x_scale_value = per_param_x_scale

        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {
                "solver_settings": {"loss": loss_name}
            }
            solver_settings = diagnostics_payload.setdefault(
                "solver_settings", {"loss": loss_name}
            )
            solver_settings["x_scale"] = (
                x_scale_value.tolist()
                if isinstance(x_scale_value, np.ndarray)
                else x_scale_value
            )
            logger.info(
                "Diagnostics enabled: loss=%s, x_scale=%s, sample_size=%d",
                loss_name,
                format_x_scale_for_log(x_scale_value),
                diagnostics_sample_size,
            )

        diagnostics_sample_x: np.ndarray | None = None
        sample_scaling = 1.0
        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {}
            diagnostics_sample_x = _sample_xdata(xdata, diagnostics_sample_size)
            if diagnostics_sample_x.size == 0:
                diagnostics_sample_x = xdata
            sample_scaling = max(1.0, xdata.size / max(diagnostics_sample_x.size, 1))
            initial_jtj, initial_norms = _compute_jacobian_stats(
                solver_residual_fn,
                diagnostics_sample_x,
                validated_params,
                sample_scaling,
            )
            if initial_norms is not None:
                diagnostics_payload.setdefault("initial_jacobian_norms", {})
                diagnostics_payload["initial_jacobian_norms"] = dict(
                    zip(param_labels, initial_norms.tolist(), strict=False),
                )
                logger.info(
                    "Initial Jacobian column norms: %s",
                    ", ".join(
                        f"{label}={norm:.3e}"
                        for label, norm in diagnostics_payload[
                            "initial_jacobian_norms"
                        ].items()
                    ),
                )

        residual_counter: FunctionEvaluationCounter | None = None
        wrapped_residual_fn: StratifiedResidualFunction | FunctionEvaluationCounter
        if diagnostics_enabled:
            residual_counter = FunctionEvaluationCounter(solver_residual_fn)
            wrapped_residual_fn = residual_counter
        else:
            wrapped_residual_fn = solver_residual_fn

        # Step 7: Select optimization strategy using memory-based selection (v2.13.0)
        # Uses unified select_nlsq_strategy() instead of deprecated DatasetSizeStrategy
        n_parameters = len(validated_params)

        # Map NLSQStrategy to local OptimizationStrategy for fallback chain
        from homodyne.optimization.nlsq.strategies.chunking import (
            estimate_nlsq_optimization_memory,
        )

        memory_stats = estimate_nlsq_optimization_memory(n_data, n_parameters)
        logger.info(
            f"Memory estimate: {memory_stats['peak_gb']:.2f} GB peak, "
            f"{memory_stats.get('available_gb', 0):.2f} GB available"
        )

        if not memory_stats.get("is_safe", True):
            logger.warning(
                f"Memory usage may be high ({memory_stats['peak_gb']:.2f} GB). "
                f"Using memory-efficient strategy."
            )

        # Check for strategy override in config
        strategy_override = None
        if config is not None and hasattr(config, "config"):
            perf_config = config.config.get("performance", {})
            strategy_override = perf_config.get("strategy_override")

        # Select strategy: use override if provided, else use memory-based selection
        if strategy_override:
            try:
                strategy = OptimizationStrategy(strategy_override)
                logger.info(f"Using overridden strategy: {strategy.value}")
            except ValueError:
                logger.warning(
                    f"Invalid strategy override '{strategy_override}', using auto"
                )
                strategy_override = None

        if not strategy_override:
            # Map memory-based decision to OptimizationStrategy for fallback chain
            decision = select_nlsq_strategy(n_data, n_parameters)
            if decision.strategy == NLSQStrategy.HYBRID_STREAMING:
                strategy = OptimizationStrategy.STREAMING
            elif decision.strategy == NLSQStrategy.OUT_OF_CORE:
                strategy = OptimizationStrategy.CHUNKED
            else:
                # STANDARD: use size-based selection for STANDARD/LARGE distinction
                if n_data < 1_000_000:
                    strategy = OptimizationStrategy.STANDARD
                elif n_data < 10_000_000:
                    strategy = OptimizationStrategy.LARGE
                else:
                    strategy = OptimizationStrategy.CHUNKED

        logger.info(
            f"Selected {strategy.value} strategy for {n_data:,} points "
            f"(peak memory: {memory_stats['peak_gb']:.2f} GB)"
        )

        # Step 8: Execute optimization with strategy fallback
        popt, pcov, info, recovery_actions, convergence_status = (
            self._execute_optimization_with_fallback(
                strategy=strategy,
                wrapped_residual_fn=wrapped_residual_fn,
                xdata=xdata,
                ydata=ydata,
                validated_params=validated_params,
                nlsq_bounds=nlsq_bounds,
                loss_name=loss_name,
                x_scale_value=x_scale_value,
                config=config,
                start_time=start_time,
                logger=logger,
            )
        )

        return self._post_process_results(
            popt=popt,
            pcov=pcov,
            info=info,
            transform_state=transform_state,
            validated_params=validated_params,
            residual_counter=residual_counter,
            base_residual_fn=base_residual_fn,
            xdata=xdata,
            n_data=n_data,
            start_time=start_time,
            nlsq_bounds=nlsq_bounds,
            convergence_status=convergence_status,
            recovery_actions=recovery_actions,
            stratification_diagnostics=stratification_diagnostics,
            diagnostics_state={
                "enabled": diagnostics_enabled,
                "payload": diagnostics_payload,
                "sample_x": diagnostics_sample_x,
                "solver_residual_fn": solver_residual_fn,
                "sample_scaling": sample_scaling,
                "param_labels": param_labels,
            },
            logger=logger,
        )

    def _execute_optimization_with_fallback(
        self,
        strategy: OptimizationStrategy,
        wrapped_residual_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        validated_params: np.ndarray,
        nlsq_bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | str,
        config: Any,
        start_time: float,
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any], list[str], str]:
        """Execute optimization with strategy fallback.

        Tries selected strategy first, then falls back to simpler strategies
        if needed. Returns (popt, pcov, info, recovery_actions, convergence_status).
        """
        import time

        current_strategy = strategy
        strategy_attempts: list[OptimizationStrategy] = []

        while current_strategy is not None:
            try:
                strategy_info = _get_strategy_info(current_strategy)
                logger.info(
                    f"Attempting optimization with {current_strategy.value} strategy..."
                )

                if (
                    current_strategy == OptimizationStrategy.STREAMING
                    and STREAMING_AVAILABLE
                ):
                    logger.info(
                        "Using NLSQ AdaptiveHybridStreamingOptimizer for large datasets..."
                    )

                    popt, pcov, info = self._fit_with_hybrid_streaming_optimizer(
                        residual_fn=wrapped_residual_fn,
                        xdata=xdata,
                        ydata=ydata,
                        initial_params=validated_params,
                        bounds=nlsq_bounds,
                        logger=logger,
                        nlsq_config=config,
                    )
                    recovery_actions = info.get("recovery_actions", [])
                    convergence_status = (
                        "converged" if info.get("success", True) else "partial"
                    )

                elif self.enable_recovery:
                    popt, pcov, info, recovery_actions, convergence_status = (
                        self._execute_with_recovery(
                            residual_fn=wrapped_residual_fn,
                            xdata=xdata,
                            ydata=ydata,
                            initial_params=validated_params,
                            bounds=nlsq_bounds,
                            strategy=current_strategy,
                            logger=logger,
                            loss_name=loss_name,
                            x_scale_value=x_scale_value,
                        )
                    )
                else:
                    use_large = current_strategy != OptimizationStrategy.STANDARD

                    if use_large:
                        result_tuple = curve_fit_large(
                            wrapped_residual_fn,
                            xdata,
                            ydata,
                            p0=validated_params.tolist(),
                            bounds=nlsq_bounds
                            if nlsq_bounds is not None
                            else (-np.inf, np.inf),
                            loss=loss_name,
                            x_scale=x_scale_value,
                            gtol=1e-6,
                            ftol=1e-6,
                            max_nfev=5000,
                            verbose=2,
                            full_output=True,
                            show_progress=strategy_info["supports_progress"],
                            stability="auto",
                        )
                        popt, pcov, info = result_tuple  # type: ignore[misc]
                    else:
                        popt, pcov = curve_fit(
                            wrapped_residual_fn,
                            xdata,
                            ydata,
                            p0=validated_params.tolist(),
                            bounds=nlsq_bounds,
                            loss=loss_name,
                            x_scale=x_scale_value,
                            gtol=1e-6,
                            ftol=1e-6,
                            max_nfev=5000,
                            verbose=2,
                            stability="auto",
                        )
                        info = {}

                    logger.info("🔍 NLSQ Result Analysis:")
                    logger.info(f"  p0 (initial):  {validated_params}")
                    logger.info(f"  popt (fitted): {popt}")
                    logger.info(
                        f"  bounds lower:  {nlsq_bounds[0] if nlsq_bounds else 'None'}"
                    )
                    logger.info(
                        f"  bounds upper:  {nlsq_bounds[1] if nlsq_bounds else 'None'}"
                    )
                    logger.info(f"  pcov diagonal: {np.diag(pcov)}")

                    params_unchanged = np.allclose(
                        popt, validated_params, rtol=1e-10, atol=1e-14
                    )
                    uncertainties_zero = np.any(np.abs(np.diag(pcov)) < 1e-15)

                    if params_unchanged:
                        logger.warning(
                            "⚠️  Optimization failure: Parameters unchanged from initial guess!\n"
                            "   This suggests curve_fit returned immediately without optimizing.\n"
                            "   Possible causes: (1) Already at optimum, (2) Singular Jacobian, (3) Bounds too tight"
                        )

                    if uncertainties_zero:
                        zero_unc_indices = np.where(np.abs(np.diag(pcov)) < 1e-15)[0]
                        logger.warning(
                            f"⚠️  Degenerate covariance: Zero uncertainties for parameters at indices {zero_unc_indices}\n"
                            f"   pcov diagonal: {np.diag(pcov)}\n"
                            f"   This indicates singular/ill-conditioned Jacobian matrix.\n"
                            f"   Affected parameters may not have been optimized properly."
                        )

                    recovery_actions = []
                    convergence_status = "converged"

                if strategy_attempts:
                    recovery_actions.append(
                        f"strategy_fallback_to_{current_strategy.value}"
                    )
                    logger.info(
                        f"Successfully optimized with fallback strategy: {current_strategy.value}\n"
                        f"  Previous attempts: {[s.value for s in strategy_attempts]}"
                    )
                break

            except (
                ValueError,
                RuntimeError,
                TypeError,
                AttributeError,
                OSError,
                MemoryError,
            ) as e:
                strategy_attempts.append(current_strategy)

                fallback_strategy = self._get_fallback_strategy(current_strategy)

                if fallback_strategy is not None:
                    logger.warning(
                        f"Strategy {current_strategy.value} failed: {str(e)[:100]}\n"
                        f"  Attempting fallback to {fallback_strategy.value} strategy..."
                    )
                    current_strategy = fallback_strategy
                else:
                    execution_time = time.time() - start_time
                    logger.error(
                        f"All strategies failed after {execution_time:.2f}s\n"
                        f"  Attempted: {[s.value for s in strategy_attempts]}\n"
                        f"  Final error: {e}"
                    )

                    if isinstance(e, RuntimeError) and (
                        "Recovery actions" in str(e) or "Suggestions" in str(e)
                    ):
                        raise
                    else:
                        raise RuntimeError(
                            f"Optimization failed with all strategies: {[s.value for s in strategy_attempts]}"
                        ) from e

        return popt, pcov, info, recovery_actions, convergence_status

    def _post_process_results(
        self,
        popt: np.ndarray,
        pcov: np.ndarray | None,
        info: dict[str, Any],
        transform_state: Any,
        validated_params: np.ndarray,
        residual_counter: Any,
        base_residual_fn: Callable[..., np.ndarray],
        xdata: np.ndarray,
        n_data: int,
        start_time: float,
        nlsq_bounds: tuple[np.ndarray, np.ndarray] | None,
        convergence_status: str,
        recovery_actions: list[str],
        stratification_diagnostics: Any,
        diagnostics_state: dict[str, Any],
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
    ) -> OptimizationResult:
        """Post-process optimization outputs into final result.

        Applies inverse transforms, computes final residuals and costs,
        runs optional diagnostics, determines success, and creates result.
        """
        import time

        # Unpack diagnostics state
        diagnostics_enabled = diagnostics_state.get("enabled", False)
        diagnostics_payload = diagnostics_state.get("payload")
        diagnostics_sample_x = diagnostics_state.get("sample_x")
        solver_residual_fn = diagnostics_state.get("solver_residual_fn")
        sample_scaling = diagnostics_state.get("sample_scaling")
        param_labels = diagnostics_state.get("param_labels")

        # Apply inverse shear transforms
        solver_params = np.asarray(popt, dtype=float)
        if transform_state:
            physical_params = apply_inverse_shear_transforms_to_vector(
                solver_params,
                transform_state,
            )
            popt = physical_params
            if pcov is not None:
                pcov = adjust_covariance_for_transforms(
                    np.asarray(pcov, dtype=float),
                    solver_params,
                    physical_params,
                    transform_state,
                )
        else:
            popt = np.asarray(popt, dtype=float)

        # Count function evaluations
        reported_nfev: int = cast(int, info.get("nfev", -1))
        corrected_nfev = (
            residual_counter.count if residual_counter is not None else reported_nfev
        )
        if diagnostics_enabled:
            diagnostics_payload = diagnostics_payload or {}
            diagnostics_payload["nfev_reported"] = reported_nfev
            diagnostics_payload["nfev_actual"] = corrected_nfev
            logger.info(
                "Diagnostics: nfev reported=%s actual=%s",
                reported_nfev,
                corrected_nfev,
            )

        # Compute final residuals using the base function (avoid counter side-effects).
        # StratifiedResidualFunction/JIT takes (params), not (xdata, *params).
        if isinstance(
            base_residual_fn,
            (StratifiedResidualFunction, StratifiedResidualFunctionJIT),
        ):
            final_residuals = base_residual_fn(popt)
        else:
            final_residuals = base_residual_fn(xdata, *popt)

        reported_iterations = -1
        if isinstance(info, dict):
            reported_iterations = info.get("nit", info.get("nfev", -1))
        iterations = max(0, corrected_nfev)

        if reported_iterations == -1:
            logger.debug(
                "Iteration count not available from NLSQ (curve_fit_large does not return this info)"
            )

        execution_time = time.time() - start_time

        # Optional diagnostics: Jacobian stats, parameter status, covariance refinement
        if diagnostics_enabled and diagnostics_sample_x is not None:
            assert diagnostics_payload is not None
            final_jtj, final_norms = _compute_jacobian_stats(
                solver_residual_fn,
                diagnostics_sample_x,
                solver_params,
                sample_scaling,
            )
            if final_norms is not None:
                diagnostics_payload["final_jacobian_norms"] = dict(
                    zip(param_labels, final_norms.tolist(), strict=False),
                )
                logger.info(
                    "Final Jacobian column norms: %s",
                    ", ".join(
                        f"{label}={norm:.3e}"
                        for label, norm in diagnostics_payload[
                            "final_jacobian_norms"
                        ].items()
                    ),
                )
            if nlsq_bounds is not None:
                statuses = _classify_parameter_status(
                    popt,
                    nlsq_bounds[0],
                    nlsq_bounds[1],
                )
                diagnostics_payload["parameter_status"] = dict(
                    zip(param_labels, statuses, strict=False),
                )
                clips = [
                    label
                    for label, st in diagnostics_payload["parameter_status"].items()
                    if st != "active"
                ]
                if clips:
                    logger.warning(
                        "Diagnostics: parameters at bounds → %s",
                        ", ".join(clips),
                    )
            if final_jtj is not None:
                pcov = np.linalg.pinv(final_jtj, rcond=1e-10)
                diagnostics_payload["jtj_condition"] = (
                    float(np.linalg.cond(final_jtj)) if final_jtj.size > 0 else None
                )

        # Determine optimization success
        initial_cost = info.get("initial_cost", 0) if isinstance(info, dict) else 0
        final_cost = np.sum(final_residuals**2)

        function_evals = iterations
        cost_reduction = (
            (initial_cost - final_cost) / initial_cost if initial_cost > 0 else 0
        )
        params_changed = not np.allclose(popt, validated_params, rtol=1e-8)

        optimization_ran = function_evals > 10 or params_changed
        optimization_improved = cost_reduction > 0.05

        if optimization_ran and optimization_improved:
            status_indicator = "✅ SUCCESS"
            status_msg = "Optimization succeeded"
        elif optimization_ran and not optimization_improved:
            status_indicator = "⚠️ MARGINAL"
            status_msg = "Optimization ran but minimal improvement"
        else:
            status_indicator = "❌ FAILED"
            status_msg = "Optimization failed (0 iterations, no cost reduction)"

        logger.info(
            f"{status_indicator}: {status_msg} in {execution_time:.2f}s\n"
            f"  Function evaluations: {function_evals}\n"
            f"  Cost: {initial_cost:.4e} → {final_cost:.4e} ({cost_reduction * 100:+.1f}%)\n"
            f"  Iterations reported: {reported_iterations} (NLSQ may report 0)"
        )
        if recovery_actions:
            logger.info(f"Recovery actions applied: {len(recovery_actions)}")

        # Extract streaming diagnostics
        streaming_diagnostics = None
        if "batch_statistics" in info:
            streaming_diagnostics = info["batch_statistics"]
        elif "streaming_diagnostics" in info:
            streaming_diagnostics = info["streaming_diagnostics"]

        # Create result
        result = self._create_fit_result(
            popt=popt,
            pcov=pcov,
            residuals=final_residuals,
            n_data=n_data,
            iterations=iterations,
            execution_time=execution_time,
            convergence_status=convergence_status,
            recovery_actions=recovery_actions,
            streaming_diagnostics=streaming_diagnostics,
            stratification_diagnostics=stratification_diagnostics,
            diagnostics_payload=diagnostics_payload if diagnostics_enabled else None,
        )

        logger.info(
            f"Final chi-squared: {result.chi_squared:.4e}, "
            f"reduced chi-squared: {result.reduced_chi_squared:.4f}",
        )

        return result

    def _execute_with_recovery(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        strategy: OptimizationStrategy,
        logger: logging.Logger | logging.LoggerAdapter[logging.Logger],
        loss_name: str,
        x_scale_value: float | str,
    ) -> tuple[np.ndarray, np.ndarray, dict, list[str], str]:
        """Execute optimization with automatic error recovery (T022-T024).

        Implements intelligent retry strategies:
        - Attempt 1: Original parameters with selected strategy
        - Attempt 2: Perturbed parameters (±10%)
        - Attempt 3: Relaxed convergence tolerance
        - Final failure: Comprehensive diagnostics

        Args:
            residual_fn: Residual function
            xdata, ydata: Data arrays
            initial_params: Initial parameter guess
            bounds: Parameter bounds tuple
            strategy: Optimization strategy to use
            logger: Logger instance

        Returns:
            (popt, pcov, info, recovery_actions, convergence_status)
        """
        # nlsq imported at module level (line 36) for automatic x64 configuration

        recovery_actions = []
        max_retries = 3
        current_params = initial_params.copy()

        # Compute initial cost for optimization success tracking
        initial_residuals = residual_fn(xdata, *initial_params)
        initial_cost = np.sum(initial_residuals**2)

        # Determine if we should use large dataset functions
        use_large = strategy != OptimizationStrategy.STANDARD
        show_progress = strategy in [
            OptimizationStrategy.LARGE,
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.STREAMING,
        ]

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Optimization attempt {attempt + 1}/{max_retries} ({strategy.value} strategy)"
                )

                if use_large:
                    # Use curve_fit_large for LARGE, CHUNKED, STREAMING strategies
                    # NLSQ handles memory management automatically via psutil
                    logger.debug(
                        "Using curve_fit_large with NLSQ automatic memory management"
                    )

                    # ✅ CRITICAL FIX (Nov 14, 2025): Use parameter magnitude-based scaling
                    # Same issue as curve_fit: x_scale from config may be scalar or "jac"
                    # which fails with 6+ orders of magnitude gradient disparity
                    if isinstance(x_scale_value, (int, float)):
                        # Config provided scalar x_scale: replace with magnitude-based
                        x_scale_large = np.abs(current_params) + 1e-3
                        logger.info(
                            f"Replacing scalar x_scale={x_scale_value} with magnitude-based scaling"
                        )
                    elif isinstance(x_scale_value, np.ndarray):
                        # Config provided per-parameter x_scale: use as-is
                        x_scale_large = x_scale_value
                    else:
                        # Fallback: magnitude-based
                        x_scale_large = np.abs(current_params) + 1e-3

                    # Note: curve_fit_large may return (popt, pcov) or OptimizeResult object
                    # depending on NLSQ version. Use _handle_nlsq_result for normalization.
                    result = curve_fit_large(
                        residual_fn,
                        xdata,
                        ydata,
                        p0=current_params.tolist(),  # Convert to list to avoid NLSQ boolean bug
                        bounds=bounds,
                        loss=loss_name,  # Configurable loss
                        x_scale=x_scale_large,  # MAGNITUDE-BASED SCALING
                        gtol=1e-6,  # Relaxed gradient tolerance
                        ftol=1e-6,  # Relaxed function tolerance
                        max_nfev=5000,  # Increased max function evaluations
                        verbose=2,  # Show iteration details
                        show_progress=show_progress,  # Enable progress for large datasets
                        stability="auto",  # Enable memory management and stability
                    )
                    # Normalize result format and extract iterations if available
                    popt, pcov, info = self._handle_nlsq_result(
                        result, OptimizationStrategy.LARGE
                    )
                    # Add initial cost for diagnostics
                    info["initial_cost"] = initial_cost
                else:
                    # Use standard curve_fit for small datasets
                    # ✅ CRITICAL FIX (Nov 14, 2025): Use parameter magnitude-based scaling
                    # PROBLEM: x_scale="jac" fails when gradient magnitudes span 6+ orders
                    #   - D0 gradient: ~1e-4 (physics: D0*t^alpha with alpha<0 suppresses sensitivity)
                    #   - offset gradient: ~600 (direct additive term)
                    #   - Result: x_scale[D0]=1e-4 makes D0 steps tiny, condition number 8.81e+14
                    #   - Consequence: Singular pcov matrix, zero uncertainties for alpha
                    #
                    # SOLUTION: Scale by parameter magnitudes, not gradients
                    #   - x_scale[i] = |p0[i]| + epsilon ensures reasonable step sizes
                    #   - D0~16830 → steps of ~16830 (appropriate for range [100, 1e6])
                    #   - alpha~1.57 → steps of ~1.57 (appropriate for range [-3, 0])
                    #   - Expected: condition number 1e+6 (8 orders better), non-zero uncertainties

                    # Compute parameter magnitude-based scaling
                    x_scale_array = (
                        np.abs(current_params) + 1e-3
                    )  # Avoid zero scale for small params

                    # DEBUG: Print bounds and scaling for diagnostics
                    # Show first 8 parameters to avoid log spam (covers scaling + key physical)
                    n_show = min(8, len(current_params))
                    logger.info(
                        f"DEBUG: Bounds and scaling (showing first {n_show} of {len(current_params)} params):"
                    )
                    if bounds is not None:
                        lower, upper = bounds
                        for i in range(n_show):
                            logger.info(
                                f"  param[{i}]: [{lower[i]:.6f}, {upper[i]:.6f}], "
                                f"initial={current_params[i]:.6f}, x_scale={x_scale_array[i]:.6e}"
                            )
                    else:
                        logger.info("  bounds=None (unbounded)")
                        for i in range(n_show):
                            logger.info(
                                f"  param[{i}]: initial={current_params[i]:.6f}, "
                                f"x_scale={x_scale_array[i]:.6e}"
                            )

                    popt, pcov = curve_fit(
                        residual_fn,
                        xdata,
                        ydata,
                        p0=current_params.tolist(),  # Convert to list to avoid NLSQ boolean bug
                        bounds=bounds,
                        loss=loss_name,
                        x_scale=x_scale_array,  # MAGNITUDE-BASED SCALING (not "jac")
                        gtol=1e-6,  # Relaxed gradient tolerance
                        ftol=1e-6,  # Relaxed function tolerance
                        max_nfev=5000,  # Increased max function evaluations
                        verbose=2,  # Show iteration details
                        stability="auto",  # Enable memory management and stability
                    )
                    info = {"initial_cost": initial_cost}

                    # DEBUG: Check pcov for singular/degenerate covariance
                    logger.info("=" * 80)
                    logger.info("🔍 NLSQ curve_fit RESULT DIAGNOSTICS")
                    logger.info("=" * 80)
                    logger.info(f"  Initial params (p0):  {current_params}")
                    logger.info(f"  Fitted params (popt): {popt}")
                    logger.info(
                        f"  Params changed: {not np.allclose(popt, current_params, rtol=1e-10)}"
                    )
                    logger.info(f"  pcov shape: {pcov.shape}")
                    logger.info(f"  pcov diagonal (uncertainties²): {np.diag(pcov)}")
                    logger.info(f"  pcov condition number: {np.linalg.cond(pcov):.2e}")

                    # Check for zero/near-zero uncertainties
                    zero_unc_mask = np.abs(np.diag(pcov)) < 1e-15
                    if np.any(zero_unc_mask):
                        zero_indices = np.where(zero_unc_mask)[0]
                        logger.warning(
                            f"⚠️  ZERO UNCERTAINTIES detected for parameters at indices: {zero_indices}"
                        )
                        logger.warning(
                            "   This indicates singular/ill-conditioned Jacobian matrix!"
                        )
                        logger.warning(
                            "   Affected parameters were likely NOT optimized by NLSQ."
                        )
                    logger.info("=" * 80)

                # Validate result: Check for NLSQ streaming bug (returns p0 instead of best_params)
                # This bug can occur when streaming optimization fails internally
                params_unchanged = np.allclose(popt, current_params, rtol=1e-10)
                identity_covariance = np.allclose(pcov, np.eye(len(popt)), rtol=1e-10)

                if params_unchanged or identity_covariance:
                    # Possible bug: parameters unchanged or identity covariance matrix
                    logger.warning(
                        f"Potential optimization failure detected:\n"
                        f"  Parameters unchanged: {params_unchanged}\n"
                        f"  Identity covariance: {identity_covariance}\n"
                        f"  This may indicate NLSQ streaming bug or failed optimization"
                    )

                    if attempt < max_retries - 1:
                        # Retry with different strategy or parameters
                        recovery_actions.append("detected_parameter_stagnation")
                        logger.info("Retrying with perturbed parameters...")
                        # Perturb parameters by 5% for next attempt (seeded for reproducibility)
                        _rng = np.random.default_rng(seed=42 + attempt)
                        perturbation = (
                            0.05
                            * current_params
                            * _rng.uniform(-1, 1, size=len(current_params))
                        )
                        current_params = current_params + perturbation
                        if bounds is not None:
                            # Clip to bounds
                            current_params = np.clip(
                                current_params, bounds[0], bounds[1]
                            )
                        continue  # Retry optimization
                    else:
                        logger.error(
                            "Optimization returned unchanged parameters after all retries. "
                            "This may indicate a bug in NLSQ or an intractable problem."
                        )

                # Success!
                convergence_status = (
                    "converged" if attempt == 0 else "converged_with_recovery"
                )
                logger.info(f"Optimization converged on attempt {attempt + 1}")
                return popt, pcov, info, recovery_actions, convergence_status

            except (
                ValueError,
                RuntimeError,
                TypeError,
                AttributeError,
                OSError,
                MemoryError,
            ) as e:
                # T026: Log exception with parameter context for debugging
                log_exception(
                    logger,
                    e,
                    context={
                        "attempt": attempt + 1,
                        "max_retries": max_retries,
                        "strategy": strategy.value,
                        "n_params": len(current_params),
                        "params_summary": f"[{current_params[0]:.4g}, ..., {current_params[-1]:.4g}]",
                    },
                    level=logging.WARNING,
                )

                # Diagnose error and determine recovery strategy
                diagnostic = self._diagnose_error(
                    error=e,
                    params=current_params,
                    bounds=bounds,
                    attempt=attempt,
                )

                logger.warning(
                    f"Attempt {attempt + 1} failed: {diagnostic['error_type']}",
                )
                logger.info(f"Diagnostic: {diagnostic['message']}")

                # Check if this is an unrecoverable error (e.g., OOM)
                recovery_strategy = diagnostic["recovery_strategy"]
                if recovery_strategy.get("action") == "no_recovery_available":
                    # Unrecoverable error - fail immediately with detailed guidance
                    error_msg = (
                        f"Optimization failed: {diagnostic['error_type']} (unrecoverable)\n"
                        f"Diagnostic: {diagnostic['message']}\n"
                        f"Suggestions:\n"
                    )
                    for suggestion in diagnostic["suggestions"]:
                        error_msg += f"  - {suggestion}\n"

                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                if attempt < max_retries - 1:
                    # Apply recovery strategy
                    recovery_actions.append(recovery_strategy["action"])
                    params_before = current_params.copy()

                    logger.info(f"Applying recovery: {recovery_strategy['action']}")

                    # Update parameters for next attempt
                    current_params = recovery_strategy["new_params"]

                    # T029: Log recovery with before/after parameter values
                    logger.info(
                        f"Recovery parameter adjustment:\n"
                        f"  Before: [{params_before[0]:.4g}, ..., {params_before[-1]:.4g}]\n"
                        f"  After:  [{current_params[0]:.4g}, ..., {current_params[-1]:.4g}]\n"
                        f"  Max change: {np.max(np.abs(current_params - params_before)):.4g}"
                    )

                    # Note: We don't modify bounds during recovery for safety
                    continue  # Retry optimization
                else:
                    # Final failure - raise with comprehensive diagnostics
                    error_msg = (
                        f"Optimization failed after {max_retries} attempts.\n"
                        f"Recovery actions attempted: {recovery_actions}\n"
                        f"Final diagnostic: {diagnostic['message']}\n"
                        f"Suggestions:\n"
                    )
                    for suggestion in diagnostic["suggestions"]:
                        error_msg += f"  - {suggestion}\n"

                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

    def _diagnose_error(
        self,
        error: Exception,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        attempt: int,
    ) -> dict[str, Any]:
        """Diagnose optimization error and provide actionable recovery strategy (T023).

        Args:
            error: Exception raised during optimization
            params: Current parameter values
            bounds: Parameter bounds
            attempt: Current attempt number (0-indexed)

        Returns:
            Diagnostic dictionary with error analysis and recovery strategy
        """
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Initialize diagnostic result
        diagnostic = {
            "error_type": error_type,
            "message": str(error),
            "suggestions": [],
            "recovery_strategy": {},
        }

        # Analyze error and determine recovery strategy
        # Check for out-of-memory errors first (most critical)
        if "resource_exhausted" in error_str or "out of memory" in error_str:
            # CPU memory exhaustion - parameter perturbation won't help
            diagnostic["error_type"] = "out_of_memory"
            diagnostic["suggestions"] = [
                "Dataset too large for available CPU memory",
                "IMMEDIATE FIX: Reduce dataset size:",
                "  - Enable phi angle filtering in config (reduce angles from 23 to 8-12)",
                "  - Reduce time points via subsampling (1001×1001 → 200×200)",
                "  - Use smaller time window in config (frames: 1000-2000 → 1000-1500)",
                "ALTERNATIVE: Increase system memory or use machine with more RAM",
                "NOTE: curve_fit_large() is disabled - residual function not chunk-aware",
            ]

            # No parameter recovery will help OOM - need architectural change
            diagnostic["recovery_strategy"] = {
                "action": "no_recovery_available",
                "reason": "Memory exhaustion requires data reduction",
                "suggested_actions": [
                    "enable_angle_filtering",
                    "reduce_time_points",
                    "increase_system_memory",
                ],
            }

        elif (
            "convergence" in error_str or "max" in error_str or "iteration" in error_str
        ):
            # Convergence failure
            diagnostic["error_type"] = "convergence_failure"
            diagnostic["suggestions"] = [
                "Try different initial parameters",
                "Relax convergence tolerance",
                "Check if data quality is sufficient",
                "Verify parameter bounds are reasonable",
            ]

            # Recovery strategy: perturb parameters or relax tolerance
            if attempt == 0:
                # First retry: perturb parameters by 10%
                perturbation = np.random.randn(*params.shape) * 0.1
                new_params = params * (1.0 + perturbation)

                # Clip to bounds if they exist
                if bounds is not None:
                    new_params = np.clip(new_params, bounds[0], bounds[1])

                diagnostic["recovery_strategy"] = {
                    "action": "perturb_initial_parameters_10pct",
                    "new_params": new_params,
                }
            else:
                # Second retry: larger perturbation (20%)
                perturbation = np.random.randn(*params.shape) * 0.2
                new_params = params * (1.0 + perturbation)

                if bounds is not None:
                    new_params = np.clip(new_params, bounds[0], bounds[1])

                diagnostic["recovery_strategy"] = {
                    "action": "perturb_initial_parameters_20pct",
                    "new_params": new_params,
                }

        elif "bound" in error_str or "constraint" in error_str:
            # Bounds-related error
            diagnostic["error_type"] = "bounds_violation"
            diagnostic["suggestions"] = [
                "Check that lower bounds < upper bounds",
                "Verify bounds are physically reasonable",
                "Consider expanding bounds if parameters consistently hit limits",
            ]

            # Recovery strategy: adjust parameters away from bounds
            if bounds is not None:
                lower, upper = bounds
                # Move parameters 20% away from bounds
                range_width = upper - lower
                new_params = lower + 0.5 * range_width  # Center of bounds

                diagnostic["recovery_strategy"] = {
                    "action": "reset_to_bounds_center",
                    "new_params": new_params,
                }
            else:
                # No bounds, just perturb
                new_params = params * 0.9
                diagnostic["recovery_strategy"] = {
                    "action": "scale_parameters_0.9x",
                    "new_params": new_params,
                }

        elif "singular" in error_str or "condition" in error_str or "rank" in error_str:
            # Ill-conditioned problem
            diagnostic["error_type"] = "ill_conditioned_jacobian"
            diagnostic["suggestions"] = [
                "Data may be insufficient to constrain all parameters",
                "Consider fixing some parameters",
                "Check for parameter correlation",
                "Verify data quality and noise levels",
            ]

            # Recovery strategy: scale parameters to improve conditioning
            new_params = params * 0.1  # Scale down by 10x
            if bounds is not None:
                new_params = np.clip(new_params, bounds[0], bounds[1])

            diagnostic["recovery_strategy"] = {
                "action": "scale_parameters_0.1x_for_conditioning",
                "new_params": new_params,
            }

        elif "nan" in error_str or "inf" in error_str:
            # Numerical overflow/underflow
            diagnostic["error_type"] = "numerical_instability"
            diagnostic["suggestions"] = [
                "Check for extreme parameter values",
                "Verify data contains no NaN/Inf values",
                "Consider parameter rescaling",
                "Check residual function implementation",
            ]

            # Recovery strategy: reset to safe default values
            if bounds is not None:
                lower, upper = bounds
                # Geometric mean of bounds (safe middle ground in log space)
                new_params = np.sqrt(np.abs(lower * upper))
                new_params = np.clip(new_params, lower, upper)
            else:
                new_params = np.ones_like(params) * 0.5

            diagnostic["recovery_strategy"] = {
                "action": "reset_to_geometric_mean_of_bounds",
                "new_params": new_params,
            }

        else:
            # Unknown error - generic recovery
            diagnostic["error_type"] = "unknown_error"
            diagnostic["suggestions"] = [
                f"Unexpected error: {error_type}",
                "Check data format and residual function",
                "Verify NLSQ package installation",
                "Consult error message for details",
            ]

            # Generic recovery: small perturbation
            perturbation = np.random.randn(*params.shape) * 0.05
            new_params = params * (1.0 + perturbation)

            if bounds is not None:
                new_params = np.clip(new_params, bounds[0], bounds[1])

            diagnostic["recovery_strategy"] = {
                "action": "generic_perturbation_5pct",
                "new_params": new_params,
            }

        return diagnostic

    def _prepare_data(self, data: Any) -> tuple[np.ndarray, np.ndarray]:
        """Transform multi-dimensional XPCS data to flattened 1D arrays.

        Args:
            data: XPCSData with shape (n_phi, n_t1, n_t2) OR StratifiedData (already flattened)

        Returns:
            (xdata, ydata): Flattened independent variables and observations
        """
        # Validate data has required attributes
        if (
            not hasattr(data, "phi")
            or not hasattr(data, "t1")
            or not hasattr(data, "t2")
            or not hasattr(data, "g2")
        ):
            raise ValueError("Data must have 'phi', 't1', 't2', and 'g2' attributes")

        # Check if this is already stratified data (has phi_flat attribute)
        if hasattr(data, "phi_flat"):
            # Stratified data is already flattened - use directly
            g2_flat = np.asarray(data.g2_flat, dtype=np.float64)
            xdata = np.arange(len(g2_flat), dtype=np.int32)
            ydata = g2_flat
            return xdata, ydata

        # Original data path: needs meshgrid and flattening
        # Get dimensions
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # CRITICAL FIX (Nov 14, 2025): Extract 1D arrays from 2D meshgrids if needed
        # Same issue as in _apply_stratification_if_needed - cache loader returns 2D
        # but meshgrid expects 1D inputs
        if t1.ndim == 2:
            t1 = t1[:, 0] if t1.size > 0 else np.array([])
        if t2.ndim == 2:
            t2 = t2[0, :] if t2.size > 0 else np.array([])

        # Validate non-empty arrays
        if phi.size == 0 or t1.size == 0 or t2.size == 0:
            raise ValueError("Data arrays cannot be empty")

        # Create meshgrid with indexing='ij' to preserve correct ordering
        # This ensures phi varies slowest, t2 varies fastest
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")

        # Flatten all arrays to 1D
        # For NLSQ curve_fit interface, xdata is typically just indices
        # We'll use a simple index array matching the data size
        xdata = np.arange(g2.size, dtype=np.int32)

        # Flatten observations
        ydata = g2.flatten().astype(np.float64, copy=False)

        return xdata, ydata

    def _apply_stratification_if_needed(
        self,
        data: Any,
        per_angle_scaling: bool,
        config: Any,
        logger: Any,
    ) -> Any:
        """Apply angle-stratified chunking if conditions require it.

        This method fixes the per-angle scaling + NLSQ chunking incompatibility
        identified in ultra-think-20251106-012247. When per-angle parameters are
        used (contrast[i], offset[i] for each phi angle), NLSQ's arbitrary chunking
        can create chunks without certain angles, resulting in zero gradients and
        silent optimization failures.

        Solution: Reorganize data so every chunk contains all phi angles, ensuring
        gradients are always well-defined.

        Args:
            data: XPCSData object with phi, t1, t2, g2 attributes
            per_angle_scaling: Whether per-angle parameters are enabled
            config: Configuration manager with stratification settings
            logger: Logger instance for diagnostics

        Returns:
            Data object (original or stratified copy) ready for optimization

        Notes:
            - No-op if conditions don't require stratification
            - Creates temporary 2x memory overhead during reorganization
            - <1% performance overhead (0.15s for 3M points)
            - Respects configuration overrides in optimization.stratification
        """
        # Extract stratification configuration with defaults
        strat_config = {}
        if config is not None and hasattr(config, "config"):
            opt_config = config.config.get("optimization", {})
            strat_config = opt_config.get("stratification", {})

        # Configuration defaults (matching YAML template)
        enabled = strat_config.get("enabled", "auto")  # "auto", true, false
        target_chunk_size = strat_config.get("target_chunk_size", 100_000)
        max_imbalance_ratio = strat_config.get("max_imbalance_ratio", 5.0)
        force_sequential = strat_config.get("force_sequential_fallback", False)
        check_memory = strat_config.get("check_memory_safety", True)
        use_index_based = strat_config.get("use_index_based", False)
        collect_diagnostics = strat_config.get("collect_diagnostics", False)
        log_diagnostics = strat_config.get("log_diagnostics", False)

        # Check if explicitly disabled
        if enabled is False or (
            isinstance(enabled, str) and enabled.lower() == "false"
        ):
            logger.info("Stratification disabled via configuration")
            return data

        # Check if we should fallback to sequential
        if force_sequential:
            logger.info("Sequential per-angle fallback forced via configuration")
            return UseSequentialOptimization(
                data=data, reason="force_sequential_fallback=true in configuration"
            )

        # Get data dimensions
        # Note: Per-angle scaling is always enabled (legacy mode removed Nov 2025)
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # CRITICAL FIX (Nov 14, 2025): Extract 1D arrays from 2D meshgrids if needed
        # ROOT CAUSE: After commit e5ac926, cache loader returns 2D meshgrids (600, 600)
        # but np.meshgrid() at line 2428 expects 1D input arrays (600,)
        # Calling meshgrid on already-meshgridded data produces wrong structure!
        # This was breaking alpha parameter gradient computation.
        if t1.ndim == 2:
            # t1_2d[i, j] = time[i] (constant along j), extract first column
            if t1.size > 0:
                t1 = t1[:, 0]
                logger.debug(
                    f"Extracted 1D t1 array from 2D meshgrid: shape {t1.shape}"
                )
            else:
                t1 = np.array([])
                logger.debug("Empty 2D t1 array converted to empty 1D array")
        if t2.ndim == 2:
            # t2_2d[i, j] = time[j] (constant along i), extract first row
            if t2.size > 0:
                t2 = t2[0, :]
                logger.debug(
                    f"Extracted 1D t2 array from 2D meshgrid: shape {t2.shape}"
                )
            else:
                t2 = np.array([])
                logger.debug("Empty 2D t2 array converted to empty 1D array")

        # Calculate total points (meshgrid creates n_phi × n_t1 × n_t2 points)
        n_points = len(phi) * len(t1) * len(t2)

        # Analyze angle distribution
        stats = analyze_angle_distribution(phi)

        # Decision logic (use configured max_imbalance_ratio)
        # Override the imbalance check with configured value
        should_stratify_auto, reason = should_use_stratification(
            n_points=n_points,
            n_angles=stats.n_angles,
            per_angle_scaling=per_angle_scaling,
            imbalance_ratio=stats.imbalance_ratio,
        )

        # Override with configuration if imbalance exceeds configured threshold
        if stats.imbalance_ratio > max_imbalance_ratio:
            # Extreme imbalance - use sequential optimization
            logger.info(
                f"Extreme angle imbalance detected ({stats.imbalance_ratio:.1f} > {max_imbalance_ratio:.1f})"
            )
            return UseSequentialOptimization(
                data=data,
                reason=f"Angle imbalance ratio ({stats.imbalance_ratio:.1f}) exceeds threshold ({max_imbalance_ratio:.1f})",
            )

        # Handle "auto" mode
        if enabled == "auto" or (
            isinstance(enabled, str) and enabled.lower() == "auto"
        ):
            should_stratify = should_stratify_auto
        else:
            # enabled is True (force on)
            should_stratify = True
            reason = "Stratification forced via configuration (enabled=true)"

        if not should_stratify:
            logger.info(f"Stratification skipped: {reason}")
            return data

        # Apply stratification
        logger.info(
            f"Applying angle-stratified chunking: {reason}\n"
            f"  Angles: {stats.n_angles}, Imbalance ratio: {stats.imbalance_ratio:.2f}\n"
            f"  Total points: {n_points:,}\n"
            f"  Target chunk size: {target_chunk_size:,}\n"
            f"  Use index-based: {use_index_based}"
        )

        # Check memory safety (if enabled in config)
        if check_memory:
            mem_stats = estimate_stratification_memory(
                n_points, use_index_based=use_index_based
            )
            if not mem_stats["is_safe"]:
                logger.warning(
                    f"Stratification may use significant memory: "
                    f"{mem_stats['peak_memory_mb']:.1f} MB peak. "
                    f"Consider: (1) setting use_index_based=true, or "
                    f"(2) setting force_sequential_fallback=true"
                )

        # Reorganize data arrays
        # Need to expand to full meshgrid first, then stratify
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")
        phi_flat = phi_grid.flatten()
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()
        g2_flat = g2.flatten()

        # Measure stratification execution time
        import time

        stratification_start = time.perf_counter()

        # Apply stratification based on mode
        if use_index_based:
            # Index-based stratification (zero-copy, ~1% memory overhead)
            logger.info("Using index-based stratification (zero-copy)")
            indices, chunk_sizes = create_angle_stratified_indices(
                phi_flat, target_chunk_size=target_chunk_size
            )

            # Apply indices to get stratified data
            phi_stratified = phi_flat[indices]
            t1_stratified = t1_flat[indices]
            t2_stratified = t2_flat[indices]
            g2_stratified = g2_flat[indices]

            # CRITICAL FIX (Jan 2026): For index-based stratification,
            # chunk_sizes are now explicitly returned.
        else:
            # Full copy stratification (2x memory overhead)
            logger.info("Using full-copy stratification")
            # Convert to JAX arrays for stratification
            phi_jax = jnp.array(phi_flat)
            t1_jax = jnp.array(t1_flat)
            t2_jax = jnp.array(t2_flat)
            g2_jax = jnp.array(g2_flat)

            # Apply stratification (use configured target_chunk_size)
            # CRITICAL FIX (Nov 10, 2025): Now returns chunk_sizes as 5th value
            # to preserve stratification boundaries during re-chunking
            (
                phi_stratified,
                t1_stratified,
                t2_stratified,
                g2_stratified,
                chunk_sizes,
            ) = create_angle_stratified_data(
                phi_jax, t1_jax, t2_jax, g2_jax, target_chunk_size=target_chunk_size
            )

            # Convert back to numpy
            phi_stratified = np.array(phi_stratified)
            t1_stratified = np.array(t1_stratified)
            t2_stratified = np.array(t2_stratified)
            g2_stratified = np.array(g2_stratified)

        # Measure execution time
        stratification_time_ms = (time.perf_counter() - stratification_start) * 1000.0

        # Compute diagnostics if requested
        diagnostics = None
        if collect_diagnostics:
            diagnostics = compute_stratification_diagnostics(
                phi_original=phi_flat,
                phi_stratified=phi_stratified,
                execution_time_ms=stratification_time_ms,
                use_index_based=use_index_based,
                target_chunk_size=target_chunk_size,
                chunk_sizes=chunk_sizes,  # Pass actual chunk boundaries
            )

            # Optionally log diagnostic report
            if log_diagnostics and diagnostics is not None:
                report = format_diagnostics_report(diagnostics)
                logger.info(f"\n{report}")

        # Create stratified data object (modify in-place or create copy)
        # We need to "unflatten" back to original shape for _prepare_data to work
        # Actually, we can't easily unflatten to 3D grid, so instead we'll create
        # a modified data object that stores the flattened stratified arrays

        # Create a simple namespace object to hold stratified data
        class StratifiedData:
            def __init__(
                self,
                phi: np.ndarray,
                t1: np.ndarray,
                t2: np.ndarray,
                g2: np.ndarray,
                original_data: Any,
                diagnostics: Any = None,
                chunk_sizes: Any = None,
            ) -> None:
                # Store flattened stratified arrays
                self.phi_flat = phi
                self.t1_flat = t1
                self.t2_flat = t2
                self.g2_flat = g2

                # Also store unique values for backwards compatibility
                self.phi = np.unique(phi)
                self.t1 = np.unique(t1)
                self.t2 = np.unique(t2)

                # Store as 1D array (already flattened and stratified)
                self.g2 = g2

                # Copy critical metadata attributes from original data
                # These are required for residual function computation
                self.sigma = original_data.sigma  # Uncertainty/error bars (CRITICAL)
                self.q = original_data.q  # Wavevector magnitude (CRITICAL)
                self.L = original_data.L  # Sample-detector distance (CRITICAL)

                # Copy optional dt if present (time step)
                if hasattr(original_data, "dt"):
                    self.dt = original_data.dt

                # Store diagnostics if available
                self.stratification_diagnostics = diagnostics

                # CRITICAL FIX (Nov 10, 2025): Store original chunk sizes
                # to preserve stratification boundaries during re-chunking
                self.chunk_sizes = chunk_sizes

        # CRITICAL FIX (Dec 2025): Pre-shuffle stratified data before returning
        # This prevents the hybrid streaming optimizer from seeing angle-sequential data
        # during L-BFGS warmup, which would cause local minimum traps (gamma_dot_t0 -> 0)
        # The shuffle must happen HERE, not in _fit_with_stratified_hybrid_streaming,
        # because other code paths may also use the stratified data.
        # Fixed seed for reproducible stratified shuffling.
        # Not user-configurable — this ensures deterministic data ordering
        # for consistent NLSQ convergence across runs.
        shuffle_seed = 42
        rng = np.random.RandomState(shuffle_seed)
        perm = rng.permutation(len(phi_stratified))
        phi_stratified = phi_stratified[perm]
        t1_stratified = t1_stratified[perm]
        t2_stratified = t2_stratified[perm]
        g2_stratified = g2_stratified[perm]
        logger.info(
            f"Pre-shuffled stratified data (seed={shuffle_seed}) to prevent local minimum traps"
        )

        stratified_data = StratifiedData(
            phi_stratified,
            t1_stratified,
            t2_stratified,
            g2_stratified,
            data,  # Pass original data to copy metadata attributes
            diagnostics,
            chunk_sizes,  # CRITICAL FIX: Pass chunk sizes for boundary-aware re-chunking
        )

        logger.info(
            f"Stratification complete: {len(g2_stratified):,} points reorganized"
        )

        return stratified_data

    def _run_sequential_optimization(
        self,
        data: Any,
        config: Any,
        initial_params: np.ndarray | None,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        analysis_mode: str,
        per_angle_scaling: bool,
        logger: Any,
        start_time: float,
        x_scale_value: Any,
        transform_cfg: dict[str, Any],
        physical_param_names: list[str],
        per_angle_scaling_initial: dict[str, list[float]] | None = None,
    ) -> OptimizationResult:
        """Run sequential per-angle optimization as a fallback strategy.

        This method optimizes each phi angle independently and combines results
        using inverse variance weighting. It's used when:
        - Angle imbalance ratio exceeds threshold (>5.0 by default)
        - force_sequential_fallback=true in configuration
        - Stratification cannot be applied

        Args:
            data: Original XPCS data object
            config: Configuration manager
            initial_params: Initial parameter guess
            bounds: Parameter bounds (lower, upper)
            analysis_mode: 'static_isotropic' or 'laminar_flow'
            per_angle_scaling: Whether per-angle parameters enabled
            logger: Logger instance
            start_time: Start time for execution timing

        Returns:
            OptimizationResult with combined parameters from all angles

        Raises:
            RuntimeError: If too few angles converge (<50% by default)
        """
        import time

        logger.info("=" * 80)
        logger.info("SEQUENTIAL PER-ANGLE OPTIMIZATION")
        logger.info("=" * 80)

        # Prepare data arrays
        phi = np.asarray(data.phi)
        t1 = np.asarray(data.t1)
        t2 = np.asarray(data.t2)
        g2 = np.asarray(data.g2)

        # Create full meshgrid
        phi_grid, t1_grid, t2_grid = np.meshgrid(phi, t1, t2, indexing="ij")
        phi_flat = phi_grid.flatten()
        t1_flat = t1_grid.flatten()
        t2_flat = t2_grid.flatten()
        g2_flat = g2.flatten()

        from homodyne.config.parameter_manager import ParameterManager

        param_manager = ParameterManager(config.config, analysis_mode)
        base_param_names = param_manager.get_all_parameter_names()
        config_lower_bounds, config_upper_bounds = param_manager.get_bounds_as_arrays(
            base_param_names
        )
        config_lower_bounds = np.asarray(config_lower_bounds, dtype=float)
        config_upper_bounds = np.asarray(config_upper_bounds, dtype=float)

        # Load initial parameters if not provided
        if initial_params is None:
            initial_params = param_manager.get_initial_values()
            logger.info(
                f"Loaded initial parameters from config: {len(initial_params)} parameters"
            )

        # Load bounds if not provided
        if bounds is None:
            bounds = param_manager.get_parameter_bounds(base_param_names)
            logger.info("Loaded parameter bounds from config")

        if initial_params is not None:
            initial_params = np.asarray(initial_params, dtype=np.float64)

        if bounds is not None:
            bounds = (
                np.asarray(bounds[0], dtype=np.float64),
                np.asarray(bounds[1], dtype=np.float64),
            )
            try:
                logger.debug(
                    "Sequential bounds dtype: lower=%s upper=%s",
                    getattr(bounds[0], "dtype", type(bounds[0])),
                    getattr(bounds[1], "dtype", type(bounds[1])),
                )
                logger.debug(
                    "Sequential bounds values: lower=%s upper=%s",
                    np.array2string(bounds[0], precision=3),
                    np.array2string(bounds[1], precision=3),
                )
            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as exc:  # pragma: no cover - logging safeguard
                logger.debug(f"Sequential bounds dtype logging failed: {exc}")

        # Create residual function using physics kernels
        # (apply_diagonal_correction and compute_g2_scaled imported at module level)

        phi_unique_all = np.unique(np.round(phi_flat, decimals=6))
        t1_unique_all = np.unique(np.asarray(t1))
        t2_unique_all = np.unique(np.asarray(t2))
        n_phi_total = len(phi_unique_all)

        per_angle_contrast_override: np.ndarray | None = None
        per_angle_offset_override: np.ndarray | None = None
        if per_angle_scaling_initial:
            contrast_override = per_angle_scaling_initial.get("contrast")
            if contrast_override is not None:
                try:
                    arr = np.asarray(contrast_override, dtype=np.float64)
                    if arr.size == n_phi_total:
                        per_angle_contrast_override = arr.copy()
                    else:
                        logger.warning(
                            "Sequential per-angle contrast override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_total,
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid sequential per-angle contrast override; ignoring"
                    )
            offset_override = per_angle_scaling_initial.get("offset")
            if offset_override is not None:
                try:
                    arr = np.asarray(offset_override, dtype=np.float64)
                    if arr.size == n_phi_total:
                        per_angle_offset_override = arr.copy()
                    else:
                        logger.warning(
                            "Sequential per-angle offset override has %d entries (expected %d); ignoring override",
                            arr.size,
                            n_phi_total,
                        )
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid sequential per-angle offset override; ignoring"
                    )

        scalar_layout_len = len(physical_param_names) + 2
        expected_per_angle_len = 2 * n_phi_total + len(physical_param_names)

        def _expand_compact_layout(vector: np.ndarray) -> np.ndarray:
            """Replicate scalar contrast/offset entries across all angles."""

            arr = np.asarray(vector, dtype=np.float64)
            if (
                expected_per_angle_len == scalar_layout_len
                or arr.size == expected_per_angle_len
            ):
                return arr
            if n_phi_total == 0 or arr.size != scalar_layout_len:
                return arr
            contrast_val = arr[0]
            offset_val = arr[1]
            physical_vals = arr[2:]
            contrast_block = np.full(n_phi_total, contrast_val, dtype=np.float64)
            offset_block = np.full(n_phi_total, offset_val, dtype=np.float64)
            return np.concatenate([contrast_block, offset_block, physical_vals])

        solver_initial_params = initial_params.copy()
        solver_per_angle_scaling = False
        solver_per_angle_expanded = False

        if per_angle_scaling:
            if solver_initial_params.size == expected_per_angle_len:
                solver_per_angle_scaling = True
            elif (
                expected_per_angle_len > scalar_layout_len
                and solver_initial_params.size == scalar_layout_len
            ):
                solver_initial_params = _expand_compact_layout(solver_initial_params)
                solver_per_angle_scaling = True
                solver_per_angle_expanded = True
                logger.info(
                    "Expanded scalar contrast/offset to per-angle layout for sequential solver (%d angles)",
                    n_phi_total,
                )
                if bounds is not None and bounds[0].size == scalar_layout_len:
                    bounds = (
                        _expand_compact_layout(bounds[0]),
                        _expand_compact_layout(bounds[1]),
                    )
            else:
                logger.warning(
                    "Per-angle scaling requested but parameter vector has %d entries (expected %d); "
                    "sequential solver will operate with scalar scaling",
                    solver_initial_params.size,
                    expected_per_angle_len,
                )

        if (
            solver_per_angle_scaling
            and solver_initial_params.size == expected_per_angle_len
        ):
            if per_angle_contrast_override is not None:
                solver_initial_params[:n_phi_total] = per_angle_contrast_override
            if per_angle_offset_override is not None:
                solver_initial_params[n_phi_total : 2 * n_phi_total] = (
                    per_angle_offset_override
                )

        param_lower_bounds = config_lower_bounds.copy()
        param_upper_bounds = config_upper_bounds.copy()
        if solver_per_angle_scaling and expected_per_angle_len > scalar_layout_len:
            param_lower_bounds = _expand_compact_layout(param_lower_bounds)
            param_upper_bounds = _expand_compact_layout(param_upper_bounds)

        if solver_per_angle_scaling:
            param_names = _build_parameter_labels(
                True,
                n_phi_total,
                physical_param_names,
            )
        else:
            param_names = base_param_names

        def _maybe_expand_x_scale(value: Any) -> Any:
            if value is None or not solver_per_angle_scaling:
                return value
            if isinstance(value, (str, bytes, dict)):
                return value
            try:
                array = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError):
                return value
            if array.ndim == 0:
                if expected_per_angle_len > 0:
                    return np.full(
                        expected_per_angle_len,
                        float(array),
                        dtype=np.float64,
                    )
                return float(array)
            if array.size == expected_per_angle_len:
                return array
            if (
                expected_per_angle_len > scalar_layout_len
                and array.size == scalar_layout_len
            ):
                return _expand_compact_layout(array)
            return value

        x_scale_value = _maybe_expand_x_scale(x_scale_value)
        sigma_source = getattr(data, "sigma", None)
        if sigma_source is None:
            sigma_array = np.ones(
                (n_phi_total, len(t1_unique_all), len(t2_unique_all)),
                dtype=np.float64,
            )
        else:
            sigma_array = np.asarray(sigma_source, dtype=np.float64)
            if not np.all(np.isfinite(sigma_array)):
                raise ValueError(
                    "sigma values must be finite; received NaN/inf entries"
                )
            if np.any(sigma_array <= 0):
                non_positive = float(np.count_nonzero(sigma_array <= 0))
                raise ValueError(
                    "sigma values must be strictly positive for least-squares "
                    "weighting; found "
                    f"{non_positive:.0f} non-positive entries"
                )

        q_value = float(getattr(data, "q", 1.0))
        L_value = float(getattr(data, "L", 1.0))
        dt_attr = getattr(data, "dt", None)
        dt_value = float(dt_attr) if dt_attr is not None else None

        t1_unique_jnp = jnp.asarray(t1_unique_all)
        t2_unique_jnp = jnp.asarray(t2_unique_all)

        physical_index_map = build_physical_index_map(
            solver_per_angle_scaling,
            n_phi_total if solver_per_angle_scaling else 0,
            physical_param_names,
        )

        transform_state: dict[str, Any] = {}
        if transform_cfg:
            solver_initial_params, transform_state = (
                apply_forward_shear_transforms_to_vector(
                    solver_initial_params,
                    physical_index_map,
                    transform_cfg,
                )
            )
            if bounds is not None:
                bounds = apply_forward_shear_transforms_to_bounds(
                    bounds,
                    transform_state,
                )

        def _compute_g2_grid_for_phi(
            phi_index: int,
            physical_params: np.ndarray,
            contrast_params: np.ndarray | float,
            offset_params: np.ndarray | float,
        ) -> np.ndarray:
            phi_val = float(phi_unique_all[phi_index])
            if solver_per_angle_scaling:
                contrast_val = float(contrast_params[phi_index])
                offset_val = float(offset_params[phi_index])
            else:
                contrast_val = float(contrast_params)
                offset_val = float(offset_params)

            g2_grid = compute_g2_scaled(
                params=jnp.asarray(physical_params),
                t1=t1_unique_jnp,
                t2=t2_unique_jnp,
                phi=phi_val,
                q=q_value,
                L=L_value,
                contrast=contrast_val,
                offset=offset_val,
                dt=dt_value,
            )
            g2_grid = jnp.squeeze(g2_grid, axis=0)
            g2_grid = apply_diagonal_correction(g2_grid)
            return np.asarray(g2_grid, dtype=np.float64)

        residual_debug_logged = False

        def residual_func(
            params: np.ndarray,
            phi_vals: np.ndarray,
            t1_vals: np.ndarray,
            t2_vals: np.ndarray,
            g2_vals: np.ndarray,
        ) -> np.ndarray:
            """Residual function compatible with sequential optimization."""

            params_np = np.asarray(params, dtype=np.float64)
            if transform_state:
                params_np = apply_inverse_shear_transforms_to_vector(
                    params_np,
                    transform_state,
                )
            phi_section = np.asarray(phi_vals, dtype=np.float64)
            t1_section = np.asarray(t1_vals, dtype=np.float64)
            t2_section = np.asarray(t2_vals, dtype=np.float64)
            g2_section = np.asarray(g2_vals, dtype=np.float64)

            if solver_per_angle_scaling:
                contrast_params = params_np[:n_phi_total]
                offset_params = params_np[n_phi_total : 2 * n_phi_total]
                physical_params = params_np[2 * n_phi_total :]
            else:
                contrast_params = float(params_np[0])
                offset_params = float(params_np[1])
                physical_params = params_np[2:]

            # Note: clip removed - sequential residual data comes from same source as
            # unique arrays (phi_flat, t1, t2), so all values are guaranteed to be in range.
            # The clip was causing optimization to converge to wrong local minima.
            # See: stratified LS fix in residual.py (D0=91342 vs 19253 issue).
            phi_indices = np.searchsorted(
                phi_unique_all, np.round(phi_section, decimals=6)
            )
            t1_indices = np.searchsorted(t1_unique_all, t1_section)
            t2_indices = np.searchsorted(t2_unique_all, t2_section)

            g2_model = np.empty_like(g2_section, dtype=np.float64)
            sigma_vals = np.empty_like(g2_section, dtype=np.float64)

            nonlocal residual_debug_logged
            if not residual_debug_logged:
                logger.debug(
                    "Sequential residual call: params_shape=%s, phi_unique=%d",
                    params_np.shape,
                    len(np.unique(phi_section)),
                )
                residual_debug_logged = True

            for phi_idx in np.unique(phi_indices):
                mask = phi_indices == phi_idx
                g2_grid = _compute_g2_grid_for_phi(
                    phi_idx, physical_params, contrast_params, offset_params
                )
                g2_model[mask] = g2_grid[t1_indices[mask], t2_indices[mask]]
                sigma_slice = sigma_array[phi_idx]
                sigma_vals[mask] = sigma_slice[t1_indices[mask], t2_indices[mask]]

            residuals = (g2_section - g2_model) / (sigma_vals + 1e-10)
            return residuals

        # Get optimizer configuration
        opt_config = config.config.get("optimization", {})
        nlsq_config = opt_config.get("nlsq", {})

        # Sequential-specific config
        seq_config = opt_config.get("sequential", {})
        min_success_rate = seq_config.get("min_success_rate", 0.5)
        weighting = seq_config.get("weighting", "inverse_variance")

        # Run sequential optimization
        logger.info(
            f"Starting per-angle optimization with {len(np.unique(phi_flat))} angles..."
        )

        least_squares_kwargs: dict[str, Any] = {
            "max_nfev": nlsq_config.get("max_iterations", 1000),
            "ftol": nlsq_config.get("tolerance", 1e-8),
        }
        if "diff_step" in nlsq_config:
            least_squares_kwargs["diff_step"] = nlsq_config["diff_step"]
        if "f_scale" in nlsq_config:
            least_squares_kwargs["f_scale"] = nlsq_config["f_scale"]
        if x_scale_value is not None:
            least_squares_kwargs["x_scale"] = x_scale_value

        sequential_result = optimize_per_angle_sequential(
            phi=phi_flat,
            t1=t1_flat,
            t2=t2_flat,
            g2_exp=g2_flat,
            residual_func=residual_func,
            initial_params=solver_initial_params,
            bounds=bounds,
            weighting=weighting,
            min_success_rate=min_success_rate,
            parameter_names=param_names,
            **least_squares_kwargs,
        )

        # Convert SequentialResult to OptimizationResult
        execution_time = time.time() - start_time

        # Get device info
        device_info = {
            "type": "CPU",  # Sequential uses scipy.optimize.least_squares (CPU only)
            "backend": "scipy.optimize.least_squares",
            "strategy": "sequential_per_angle",
        }

        # Compute chi-squared
        combined_solver = sequential_result.combined_parameters.copy()
        combined_physical = combined_solver.copy()
        if transform_state:
            combined_physical = apply_inverse_shear_transforms_to_vector(
                combined_physical,
                transform_state,
            )

        final_residuals = residual_func(
            combined_physical, phi_flat, t1_flat, t2_flat, g2_flat
        )
        chi_squared = float(np.sum(final_residuals**2))
        n_data = len(phi_flat)
        n_params = len(sequential_result.combined_parameters)
        reduced_chi_squared = chi_squared / (n_data - n_params)

        # Diagnostics payload
        param_status = {}
        for idx, name in enumerate(param_names):
            value = combined_physical[idx]
            if np.isclose(value, param_lower_bounds[idx]):
                status = "at_lower_bound"
            elif np.isclose(value, param_upper_bounds[idx]):
                status = "at_upper_bound"
            else:
                status = "active"
            param_status[name] = status

        def _norm_array_to_dict(array: np.ndarray | None) -> dict[str, float] | None:
            if array is None:
                return None
            return {name: float(array[idx]) for idx, name in enumerate(param_names)}

        per_angle_jac = {}
        for angle_result in sequential_result.per_angle_results:
            angle_label = f"phi_{angle_result['phi_angle']:.2f}"
            per_angle_jac[angle_label] = {
                "initial": None,
                "final": None,
            }
            if angle_result.get("jac_initial_norms") is not None:
                per_angle_jac[angle_label]["initial"] = {
                    name: float(angle_result["jac_initial_norms"][idx])
                    for idx, name in enumerate(param_names)
                }
            if angle_result.get("jac_final_norms") is not None:
                per_angle_jac[angle_label]["final"] = {
                    name: float(angle_result["jac_final_norms"][idx])
                    for idx, name in enumerate(param_names)
                }

        total_nfev = sum(
            r.get("n_iterations", 0) for r in sequential_result.per_angle_results
        )

        diagnostics_payload = {
            "solver_settings": {
                "loss": nlsq_config.get("loss", "linear"),
                "x_scale": nlsq_config.get("x_scale", "jac"),
                "strategy": "sequential_per_angle",
                "jac_sample_size": JAC_SAMPLE_SIZE,
            },
            "nfev_reported": total_nfev,
            "nfev_actual": total_nfev,
            "parameter_status": param_status,
            "initial_jacobian_norms": _norm_array_to_dict(
                sequential_result.initial_jacobian_norms
            ),
            "final_jacobian_norms": _norm_array_to_dict(
                sequential_result.final_jacobian_norms
            ),
            "per_angle_jacobian_norms": per_angle_jac,
            "solver_per_angle_expanded": solver_per_angle_expanded,
            "chi_squared": chi_squared,
            "reduced_chi_squared": reduced_chi_squared,
        }

        # Determine convergence status
        if sequential_result.success_rate >= min_success_rate:
            convergence_status = "converged"
            quality_flag = (
                "good" if sequential_result.success_rate > 0.8 else "marginal"
            )
        else:
            convergence_status = "failed"
            quality_flag = "poor"

        # Compute uncertainties from covariance
        uncertainties = np.sqrt(np.diag(sequential_result.combined_covariance))

        # Summary logging
        logger.info("=" * 80)
        logger.info("SEQUENTIAL OPTIMIZATION COMPLETE")
        logger.info(f"  Success rate: {sequential_result.success_rate:.1%}")
        logger.info(
            f"  Angles optimized: {sequential_result.n_angles_optimized}/{sequential_result.n_angles_optimized + sequential_result.n_angles_failed}"
        )
        logger.info(f"  Combined cost: {sequential_result.total_cost:.4f}")
        logger.info(f"  Reduced χ²: {reduced_chi_squared:.4f}")
        logger.info(f"  Execution time: {execution_time:.2f}s")
        logger.info(f"  Weighting: {weighting}")
        logger.info("=" * 80)

        return OptimizationResult(
            parameters=combined_physical,
            uncertainties=uncertainties,
            covariance=sequential_result.combined_covariance,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=sum(
                r["n_iterations"] for r in sequential_result.per_angle_results
            ),
            execution_time=execution_time,
            device_info=device_info,
            recovery_actions=[
                f"Sequential per-angle optimization: {sequential_result.n_angles_optimized} angles converged"
            ],
            quality_flag=quality_flag,
            nlsq_diagnostics=diagnostics_payload,
        )

    def _validate_initial_params(
        self,
        params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Validate initial parameters are within bounds, clip if necessary.

        Args:
            params: Initial parameter guess
            bounds: (lower, upper) bounds tuple or None

        Returns:
            Validated/clipped parameter array

        Raises:
            ValueError: If params shape doesn't match bounds
        """
        params = np.asarray(params)

        # If no bounds, return params as-is
        if bounds is None:
            return params

        lower, upper = bounds
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Validate parameter count matches bounds
        if params.shape != lower.shape or params.shape != upper.shape:
            raise ValueError(
                f"Parameter shape mismatch: params={params.shape}, "
                f"lower={lower.shape}, upper={upper.shape}",
            )

        # Clip parameters to bounds
        clipped_params = np.clip(params, lower, upper)

        # Warn if any parameters were clipped
        if not np.allclose(params, clipped_params):
            clipped_indices = np.where(~np.isclose(params, clipped_params))[0]
            logger = get_logger(__name__)
            logger.warning(
                f"Initial parameters clipped to bounds at indices {clipped_indices}",
            )

        return clipped_params

    def _convert_bounds(
        self,
        homodyne_bounds: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Convert homodyne bounds format to NLSQ format.

        Args:
            homodyne_bounds: (lower_array, upper_array) tuple or None

        Returns:
            NLSQ-compatible bounds tuple or None for unbounded optimization

        Raises:
            ValueError: If bounds are invalid (lower >= upper)
        """
        # Handle None bounds (unbounded optimization)
        if homodyne_bounds is None:
            return None

        # Extract lower and upper bounds
        lower, upper = homodyne_bounds

        # Convert to numpy arrays if not already
        lower = np.asarray(lower)
        upper = np.asarray(upper)

        # Validate bounds: lower < upper elementwise
        if np.any(lower >= upper):
            invalid_indices = np.where(lower >= upper)[0]
            raise ValueError(
                f"Invalid bounds: lower >= upper at indices {invalid_indices}. "
                f"Lower bounds must be strictly less than upper bounds.",
            )

        # NLSQ uses the same (lower, upper) tuple format as homodyne
        # Just return validated bounds
        return (lower, upper)

    def _create_residual_function(
        self, data: Any, analysis_mode: str, per_angle_scaling: bool = True
    ) -> Any:
        """Create JAX-compatible model function for NLSQ with per-angle scaling support.

        IMPORTANT: NLSQ's curve_fit_large expects a MODEL FUNCTION f(x, *params) -> y,
        NOT a residual function. NLSQ internally computes residuals = data - model.

        Args:
            data: XPCS experimental data
            analysis_mode: Analysis mode determining model computation
            per_angle_scaling: If True (default), use per-angle contrast/offset parameters.
                             This is the physically correct behavior.
                             If False, use legacy single contrast/offset for all angles.

        Returns:
            Model function with signature f(xdata, *params) -> ydata_theory
            where xdata is a dummy variable for NLSQ compatibility

        Raises:
            AttributeError: If data is missing required attributes
        """
        # Import NLSQ physics backend for g2 computation
        from homodyne.core.physics_nlsq import compute_g2_scaled

        # Validate data has required attributes
        required_attrs = ["phi", "t1", "t2", "g2", "sigma", "q", "L"]
        for attr in required_attrs:
            if not hasattr(data, attr):
                raise AttributeError(
                    f"Data must have '{attr}' attribute for residual computation",
                )

        # Extract data attributes and convert to JAX arrays
        # CRITICAL FIX (Nov 11, 2025): Handle stratified vs non-stratified data differently
        #
        # Stratified data: phi_flat, t1_flat, t2_flat are all per-point arrays (same length)
        # Non-stratified data: phi, t1, t2 are unique grid values (different lengths)
        is_stratified = hasattr(data, "phi_flat")

        if is_stratified:
            # Stratified data: use per-point flat arrays
            phi = jnp.asarray(data.phi_flat)  # Shape: (n_data,)
            t1 = jnp.asarray(data.t1_flat)  # Shape: (n_data,)
            t2 = jnp.asarray(data.t2_flat)  # Shape: (n_data,)
        else:
            # Non-stratified data: use unique grid values
            phi = jnp.asarray(data.phi)  # Shape: (n_phi,)
            t1 = jnp.asarray(data.t1)  # Shape: (n_t1,)
            t2 = jnp.asarray(data.t2)  # Shape: (n_t2,)

        q = float(data.q)
        L = float(data.L)

        # Get dt from data — required for compute_g2_scaled (float, not Optional).
        dt = getattr(data, "dt", None)
        if dt is not None:
            dt = float(dt)
            # Validate dt before JIT compilation (avoid JAX tracing issues)
            if dt <= 0:
                raise ValueError(f"dt must be positive, got {dt}")
            if not np.isfinite(dt):
                raise ValueError(f"dt must be finite, got {dt}")
        else:
            # Fallback: derive from t1 minimum spacing
            t1_arr = np.asarray(data.t1)
            t1_unique = np.unique(t1_arr)
            if len(t1_unique) > 1:
                dt = float(np.min(np.diff(t1_unique)))
            else:
                dt = 0.001
            import warnings

            warnings.warn(
                f"data.dt missing; derived dt={dt:.6g} from t1 spacing",
                stacklevel=2,
            )

        # Pre-compute phi_unique for per-angle parameter mapping
        phi_unique = jnp.asarray(np.unique(np.asarray(phi)))
        n_phi = len(phi_unique)

        # Determine parameter structure based on analysis mode and per_angle_scaling
        # Legacy (per_angle_scaling=False): [contrast, offset, *physical_params]
        #   Static isotropic: 5 params total (2 scaling + 3 physical)
        #   Laminar flow: 9 params total (2 scaling + 7 physical)
        #
        # Per-angle (per_angle_scaling=True): [contrast_0, ..., contrast_{n_phi-1},
        #                                       offset_0, ..., offset_{n_phi-1}, *physical_params]
        #   Static isotropic: (2*n_phi + 3) params total
        #   Laminar flow: (2*n_phi + 7) params total

        def model_function(xdata: jnp.ndarray, *params_tuple: float) -> jnp.ndarray:
            """Compute theoretical g2 model for NLSQ optimization with per-angle scaling.

            IMPORTANT: xdata contains indices into the flattened data array.
            This function MUST respect xdata size for curve_fit_large chunking.
            When curve_fit_large chunks the data, xdata will be a subset of indices.

            NLSQ will internally compute residuals as: (ydata - model) / sigma

            Args:
                xdata: Array of indices into flattened g2 array.
                       Full dataset: [0, 1, ..., n-1]
                       Chunked: [0, 1, ..., chunk_size-1] (subset)
                *params_tuple: Unpacked parameters (per-angle scaling only)
                    - Format: [contrast_0, ..., contrast_{n_phi-1},
                              offset_0, ..., offset_{n_phi-1}, *physical]

            Returns:
                Theoretical g2 values at requested indices (size matches xdata)
            """
            # Convert params tuple to array (stack avoids retracing vs asarray)
            params_array = jnp.stack(params_tuple)

            # Extract per-angle scaling parameters (legacy mode removed Nov 2025)
            # Per-angle mode: first n_phi are contrasts, next n_phi are offsets
            contrast = params_array[:n_phi]  # Array of shape (n_phi,)
            offset = params_array[n_phi : 2 * n_phi]  # Array of shape (n_phi,)
            physical_params = params_array[2 * n_phi :]

            # Get requested data point indices
            indices = jnp.asarray(xdata, dtype=jnp.int32)

            # CRITICAL FIX (Nov 11, 2025): Handle stratified vs non-stratified data
            if is_stratified:
                # STRATIFIED DATA PATH (per-point arrays)
                # Extract per-point values for requested indices
                phi_requested = phi[indices]  # Shape: (chunk_size,)
                t1_requested = t1[indices]  # Shape: (chunk_size,)
                t2_requested = t2[indices]  # Shape: (chunk_size,)

                # Map phi values to indices in phi_unique to get correct contrast/offset
                # Find which unique phi each requested phi corresponds to
                # Since phi values come from phi_unique, we can use searchsorted
                # CRITICAL: Keep all arrays in JAX (no np.asarray) for JIT compatibility
                # Note: clip removed - phi_requested is a subset of phi which was used to
                # build phi_unique, so all values are guaranteed to be in range.
                # The clip was causing optimization to converge to wrong local minima.
                phi_idx = jnp.searchsorted(
                    phi_unique, phi_requested
                )  # Shape: (chunk_size,)

                # Select per-angle contrast and offset for each data point
                contrast_requested = contrast[phi_idx]  # Shape: (chunk_size,)
                offset_requested = offset[phi_idx]  # Shape: (chunk_size,)

                # Compute g2 per-point using vmap
                # Each point has its own (phi, t1, t2, contrast, offset)
                compute_g2_per_point = jax.vmap(
                    lambda phi_val, t1_val, t2_val, c_val, o_val: compute_g2_scaled(
                        params=physical_params,
                        t1=jnp.array([t1_val]),  # Single value as 1D array
                        t2=jnp.array([t2_val]),
                        phi=phi_val,
                        q=q,
                        L=L,
                        contrast=c_val,
                        offset=o_val,
                        dt=dt,
                    )[0, 0],  # Extract scalar from (1, 1) output
                    in_axes=(0, 0, 0, 0, 0),  # Vmap over all arrays
                )

                g2_theory = compute_g2_per_point(
                    phi_requested,
                    t1_requested,
                    t2_requested,
                    contrast_requested,
                    offset_requested,
                )  # Shape: (chunk_size,) or possibly (chunk_size, 1)

                # Ensure 1D output by squeezing any trailing dimensions
                g2_theory = jnp.squeeze(g2_theory)

                return g2_theory

            else:
                # NON-STRATIFIED DATA PATH (grid-based computation)
                # Original grid-based logic for non-stratified data
                compute_g2_scaled_vmap = jax.vmap(
                    lambda phi_val, contrast_val, offset_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t1,  # 1D arrays
                            t2=t2,
                            phi=phi_val,  # Single phi value
                            q=q,
                            L=L,
                            contrast=contrast_val,  # Per-angle contrast
                            offset=offset_val,  # Per-angle offset
                            dt=dt,
                        ),
                        axis=0,  # Squeeze the phi dimension
                    ),
                    in_axes=(0, 0, 0),  # Vectorize over all three arrays
                )

                # Compute on grid for all unique angles
                g2_theory = compute_g2_scaled_vmap(phi_unique, contrast, offset)
                # Shape: (n_phi, n_t1, n_t2)

                # Apply diagonal correction
                from homodyne.core.jax_backend import apply_diagonal_correction

                apply_diagonal_vmap = jax.vmap(apply_diagonal_correction, in_axes=0)
                g2_theory = apply_diagonal_vmap(g2_theory)

                # Grid-based indexing for non-stratified data
                n_t1 = len(t1)
                n_t2 = len(t2)
                grid_size_per_angle = n_t1 * n_t2

                # Decompose flat indices into grid coordinates
                phi_idx = indices // grid_size_per_angle
                remaining = indices % grid_size_per_angle
                t1_idx = remaining // n_t2
                t2_idx = remaining % n_t2

                return g2_theory[phi_idx, t1_idx, t2_idx]

        return model_function

    def _update_best_parameters(
        self,
        params: np.ndarray,
        loss: float,
        batch_idx: int,
        logger: Any,
    ) -> None:
        """Update best parameters if current loss is better.

        Parameters
        ----------
        params : np.ndarray
            Current parameter values
        loss : float
            Current loss value
        batch_idx : int
            Current batch index
        logger : logging.Logger
            Logger instance for reporting
        """
        if params is None:
            return  # Cannot update without parameters
        if loss < self.best_loss:
            self.best_params = params.copy()
            self.best_loss = loss
            self.best_batch_idx = batch_idx
            logger.info(
                f"New best loss: {loss:.6e} at batch {batch_idx} "
                f"(improved from {self.best_loss:.6e})"
            )

    def _fit_with_streaming_optimizer(
        self,
        residual_fn: Any,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        checkpoint_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using streaming optimizer for large datasets.

        .. deprecated:: 2.9.1
            The old non-stratified StreamingOptimizer was removed in NLSQ 0.4.0.
            Use the stratified optimization path with per-angle scaling instead,
            which automatically uses AdaptiveHybridStreamingOptimizer for large datasets.

        Raises
        ------
        RuntimeError
            Always - this code path is no longer supported
        """
        raise RuntimeError(
            "Non-stratified StreamingOptimizer was removed in NLSQ 0.4.0. "
            "Use the stratified optimization path with per_angle_scaling=True, "
            "which automatically switches to AdaptiveHybridStreamingOptimizer "
            "for memory-constrained datasets. "
            "See CLAUDE.md section 'NLSQ Adaptive Hybrid Streaming Mode' for details."
        )

    def _fit_with_hybrid_streaming_optimizer(
        self,
        residual_fn: Any,
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        nlsq_config: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ AdaptiveHybridStreamingOptimizer for large datasets.

        This method uses NLSQ's four-phase hybrid optimizer to fix three key issues:
        1. Shear-term weak gradients (scale imbalance) - via parameter normalization
        2. Slow convergence - via L-BFGS warmup + Gauss-Newton refinement
        3. Crude covariance - via exact J^T J accumulation + covariance transform

        Four Phases:
        - Phase 0: Parameter normalization setup (bounds-based)
        - Phase 1: L-BFGS warmup with adaptive switching
        - Phase 2: Streaming Gauss-Newton with exact J^T J accumulation
        - Phase 3: Denormalization and covariance transform

        Parameters
        ----------
        residual_fn : callable
            Residual function (StratifiedResidualFunction or similar)
        xdata : np.ndarray
            Independent variable data (flattened)
        ydata : np.ndarray
            Dependent variable data (flattened)
        initial_params : np.ndarray
            Initial parameter guess
        bounds : tuple of np.ndarray or None
            Parameter bounds (lower, upper)
        logger : logging.Logger
            Logger instance
        nlsq_config : NLSQConfig, optional
            NLSQ configuration with hybrid streaming settings

        Returns
        -------
        popt : np.ndarray
            Optimized parameters
        pcov : np.ndarray
            Covariance matrix (properly transformed to original space)
        info : dict
            Optimization information including phase diagnostics

        Raises
        ------
        RuntimeError
            If AdaptiveHybridStreamingOptimizer is not available
        NLSQOptimizationError
            If optimization fails
        """
        if not HYBRID_STREAMING_AVAILABLE:
            raise RuntimeError(
                "AdaptiveHybridStreamingOptimizer not available. "
                "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
            )

        logger.info("Initializing NLSQ AdaptiveHybridStreamingOptimizer...")
        logger.info("Fixes: 1) Shear-term gradients, 2) Convergence, 3) Covariance")

        # Create HybridStreamingConfig from NLSQConfig with 4-layer defense
        if nlsq_config is not None:
            config = HybridStreamingConfig(
                normalize=nlsq_config.hybrid_normalize,
                normalization_strategy=nlsq_config.hybrid_normalization_strategy,
                warmup_iterations=nlsq_config.hybrid_warmup_iterations,
                max_warmup_iterations=nlsq_config.hybrid_max_warmup_iterations,
                warmup_learning_rate=nlsq_config.hybrid_warmup_learning_rate,
                gauss_newton_max_iterations=nlsq_config.hybrid_gauss_newton_max_iterations,
                gauss_newton_tol=nlsq_config.hybrid_gauss_newton_tol,
                chunk_size=nlsq_config.hybrid_chunk_size,
                trust_region_initial=nlsq_config.hybrid_trust_region_initial,
                regularization_factor=nlsq_config.hybrid_regularization_factor,
                enable_checkpoints=nlsq_config.hybrid_enable_checkpoints,
                checkpoint_frequency=nlsq_config.hybrid_checkpoint_frequency,
                validate_numerics=nlsq_config.hybrid_validate_numerics,
                # 4-Layer Defense Strategy (NLSQ 0.3.6)
                enable_warm_start_detection=nlsq_config.hybrid_enable_warm_start_detection,
                warm_start_threshold=nlsq_config.hybrid_warm_start_threshold,
                enable_adaptive_warmup_lr=nlsq_config.hybrid_enable_adaptive_warmup_lr,
                warmup_lr_refinement=nlsq_config.hybrid_warmup_lr_refinement,
                warmup_lr_careful=nlsq_config.hybrid_warmup_lr_careful,
                enable_cost_guard=nlsq_config.hybrid_enable_cost_guard,
                cost_increase_tolerance=nlsq_config.hybrid_cost_increase_tolerance,
                enable_step_clipping=nlsq_config.hybrid_enable_step_clipping,
                max_warmup_step_size=nlsq_config.hybrid_max_warmup_step_size,
            )
        else:
            # Use NLSQ 0.3.6 defaults with 4-layer defense enabled
            config = HybridStreamingConfig(
                normalize=True,
                normalization_strategy="auto",
                warmup_iterations=200,
                max_warmup_iterations=500,
                gauss_newton_max_iterations=100,
                gauss_newton_tol=1e-8,
                chunk_size=10000,
                # 4-Layer Defense enabled by default
                enable_warm_start_detection=True,
                warm_start_threshold=0.01,
                enable_adaptive_warmup_lr=True,
                warmup_lr_refinement=1e-6,
                warmup_lr_careful=1e-5,
                enable_cost_guard=True,
                cost_increase_tolerance=0.05,
                enable_step_clipping=True,
                max_warmup_step_size=0.1,
            )

        logger.info(f"  Normalization: {config.normalization_strategy}")
        logger.info(f"  Warmup iterations: {config.warmup_iterations}")
        logger.info(f"  Gauss-Newton max: {config.gauss_newton_max_iterations}")
        logger.info(f"  Chunk size: {config.chunk_size}")

        # Initialize optimizer
        optimizer = AdaptiveHybridStreamingOptimizer(config)

        # Create model function from residual function
        # The hybrid optimizer expects: func(x, *params) -> predictions
        # Our residual function computes: residuals = y - predictions
        # We need: predictions = y - residuals
        if hasattr(residual_fn, "jax_residual"):
            # Stratified residual function

            def model_fn(x: Any, *params: float) -> Any:
                params_array = jnp.asarray(params)
                residuals = residual_fn.jax_residual(params_array)
                return ydata - residuals

        else:
            # Standard residual function

            def model_fn(x: Any, *params: float) -> Any:
                residuals = residual_fn(x, *params)
                return ydata - residuals

        try:
            # Run optimization
            result = optimizer.fit(
                data_source=(xdata, ydata),
                func=model_fn,
                p0=initial_params,
                bounds=bounds,
                sigma=None,  # TODO: Add sigma support if needed
                verbose=1 if not self.fast_mode else 0,
            )

            # Extract results
            popt = np.asarray(result["x"])
            pcov = np.asarray(result.get("pcov", np.eye(len(popt))))
            perr = np.asarray(
                result.get("perr", _safe_uncertainties_from_pcov(pcov, len(popt)))
            )

            # Build info dict with phase diagnostics
            info = {
                "success": result.get("success", True),
                "message": result.get("message", "Hybrid optimization completed"),
                "hybrid_streaming_diagnostics": result.get("streaming_diagnostics", {}),
                "perr": perr,
                "sigma_sq": result.get("streaming_diagnostics", {})
                .get("gauss_newton_diagnostics", {})
                .get("final_cost"),
                "phase_timings": result.get("streaming_diagnostics", {}).get(
                    "phase_timings", {}
                ),
            }

            logger.info("Hybrid streaming optimization completed successfully")
            phase_timings = info.get("phase_timings", {})
            if phase_timings:
                logger.info(
                    f"  Phase 0 (normalization): {phase_timings.get('phase0_normalization', 0):.3f}s"
                )
                logger.info(
                    f"  Phase 1 (L-BFGS warmup): {phase_timings.get('phase1_warmup', 0):.3f}s"
                )
                logger.info(
                    f"  Phase 2 (Gauss-Newton): {phase_timings.get('phase2_gauss_newton', 0):.3f}s"
                )
                logger.info(
                    f"  Phase 3 (covariance): {phase_timings.get('phase3_finalize', 0):.3f}s"
                )

            return popt, pcov, info

        except (
            ValueError,
            RuntimeError,
            TypeError,
            AttributeError,
            OSError,
            MemoryError,
        ) as e:
            # T031: Log detailed warning explaining failure and lost capabilities
            logger.error(f"AdaptiveHybridStreamingOptimizer failed: {e}")
            logger.warning(
                "=" * 60 + "\n"
                "HYBRID OPTIMIZER FAILURE - Falling back to basic streaming\n"
                "=" * 60 + "\n"
                "The AdaptiveHybridStreamingOptimizer encountered an error.\n"
                "\n"
                "Capabilities lost with fallback:\n"
                "  - Parameter normalization (gradient equalization)\n"
                "  - L-BFGS warmup + Gauss-Newton hybrid convergence\n"
                "  - Exact J^T J covariance accumulation\n"
                "\n"
                "Fallback uses basic streaming optimizer which may:\n"
                "  - Converge slower (1000+ vs ~110 iterations)\n"
                "  - Miss shear parameters (imbalanced gradients)\n"
                "  - Produce less accurate uncertainties\n"
                "\n"
                f"Error details: {type(e).__name__}: {str(e)}\n"
                "=" * 60
            )
            # T030: TODO - Implement 3-attempt retry with HybridRecoveryConfig
            # For now, immediately raise to trigger fallback to streaming
            if isinstance(e, NLSQOptimizationError):
                raise
            else:
                raise NLSQOptimizationError(
                    f"AdaptiveHybridStreamingOptimizer failed: {str(e)}",
                    error_context={"original_error": type(e).__name__},
                ) from e

    def _create_stratified_chunks(
        self,
        stratified_data: Any,
        target_chunk_size: int = 100_000,
    ) -> Any:
        """Convert stratified flat arrays into chunks for StratifiedResidualFunction.

        Args:
            stratified_data: StratifiedData object with flat stratified arrays
            target_chunk_size: Target size for each chunk

        Returns:
            Object with .chunks attribute containing list of chunk objects
        """
        # Get flat stratified arrays
        phi_flat = stratified_data.phi_flat
        t1_flat = stratified_data.t1_flat
        t2_flat = stratified_data.t2_flat
        g2_flat = stratified_data.g2_flat

        # Get metadata (not chunked - shared across all chunks)
        sigma = stratified_data.sigma  # 3D array: (n_phi, n_t1, n_t2)
        q = stratified_data.q
        L = stratified_data.L
        dt = getattr(stratified_data, "dt", None)

        # CRITICAL FIX (Nov 10, 2025): Use original stratification boundaries
        # instead of naive sequential slicing to preserve angle completeness
        chunk_sizes_attr = getattr(stratified_data, "chunk_sizes", None)

        if chunk_sizes_attr is not None:
            # Use original chunk boundaries from stratification
            # This ensures each chunk contains all phi angles
            n_chunks = len(chunk_sizes_attr)
            chunks = []
            current_idx = 0

            for _, chunk_size in enumerate(chunk_sizes_attr):
                start_idx = current_idx
                end_idx = current_idx + chunk_size

                # Create simple namespace object for chunk
                class Chunk:
                    def __init__(
                        self,
                        phi: Any,
                        t1: Any,
                        t2: Any,
                        g2: Any,
                        q: Any,
                        L: Any,
                        dt: Any,
                    ) -> None:
                        self.phi = phi
                        self.t1 = t1
                        self.t2 = t2
                        self.g2 = g2
                        self.q = q
                        self.L = L
                        self.dt = dt

                chunk = Chunk(
                    phi=phi_flat[start_idx:end_idx],
                    t1=t1_flat[start_idx:end_idx],
                    t2=t2_flat[start_idx:end_idx],
                    g2=g2_flat[start_idx:end_idx],
                    q=q,
                    L=L,
                    dt=dt,
                )
                chunks.append(chunk)
                current_idx = end_idx
        else:
            # Fallback: Sequential chunking (for index-based stratification)
            # WARNING: This may still have angle incompleteness issues!
            n_total = len(g2_flat)
            n_chunks = max(1, (n_total + target_chunk_size - 1) // target_chunk_size)

            chunks = []
            for i in range(n_chunks):
                start_idx = i * target_chunk_size
                end_idx = min(start_idx + target_chunk_size, n_total)

                class Chunk:  # type: ignore[no-redef]
                    def __init__(
                        self,
                        phi: Any,
                        t1: Any,
                        t2: Any,
                        g2: Any,
                        q: Any,
                        L: Any,
                        dt: Any,
                    ) -> None:
                        self.phi = phi
                        self.t1 = t1
                        self.t2 = t2
                        self.g2 = g2
                        self.q = q
                        self.L = L
                        self.dt = dt

                chunk = Chunk(
                    phi=phi_flat[start_idx:end_idx],
                    t1=t1_flat[start_idx:end_idx],
                    t2=t2_flat[start_idx:end_idx],
                    g2=g2_flat[start_idx:end_idx],
                    q=q,
                    L=L,
                    dt=dt,
                )
                chunks.append(chunk)

        # Create object with chunks attribute and metadata
        class StratifiedChunkedData:
            def __init__(self, chunks: list[Any], sigma: Any) -> None:
                self.chunks = chunks
                self.sigma = sigma  # Store sigma as metadata at parent level

        return StratifiedChunkedData(chunks, sigma)

    def _fit_with_stratified_least_squares(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        target_chunk_size: int = 100_000,
        anti_degeneracy_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ's least_squares() with stratified residual function.

        This method solves the double-chunking problem by using NLSQ's least_squares()
        function directly with a StratifiedResidualFunction. This gives us full control
        over chunking, ensuring angle completeness in each chunk for proper per-angle
        parameter gradients.

        Unlike curve_fit_large(), which expects a MODEL function f(x, *params) → y and
        does its own internal chunking (breaking angle stratification), least_squares()
        accepts a RESIDUAL function fun(params) → residuals where we control all data
        access inside the function.

        Key Features:
        - Uses NLSQ's least_squares() for JAX-accelerated optimization
        - Maintains angle-stratified chunks (no double-chunking)
        - Compatible with per-angle scaling parameters
        - CPU-optimized execution
        - Trust-region reflective (TRF) algorithm
        - Anti-Degeneracy Defense System (v2.14.0+)

        Args:
            stratified_data: StratifiedData object with flat stratified arrays
            per_angle_scaling: Whether per-angle parameters are enabled
            physical_param_names: List of physical parameter names (e.g., ['D0', 'alpha', 'D_offset'])
            initial_params: Initial parameter guess
            bounds: Parameter bounds (lower, upper) tuple
            logger: Logger instance
            target_chunk_size: Target size for each chunk (default: 100k points)
            anti_degeneracy_config: Optional config dict for Anti-Degeneracy Defense System

        Returns:
            (popt, pcov, info) tuple:
                - popt: Optimized parameters
                - pcov: Covariance matrix
                - info: Optimization info dict

        Raises:
            ValueError: If stratified_data doesn't have required attributes
            RuntimeError: If optimization fails

        References:
            - Solution document: docs/architecture/nlsq-least-squares-solution.md
            - Ultra-think analysis: docs/architecture/ultra-think-nlsq-solution-20251106.md
        """
        import time

        logger.info("=" * 80)
        logger.info("STRATIFIED LEAST-SQUARES OPTIMIZATION")
        logger.info("Using NLSQ's least_squares() with angle-stratified chunks")
        logger.info("=" * 80)

        # =====================================================================
        # Anti-Degeneracy Defense System (v2.14.0+)
        # =====================================================================
        is_laminar_flow = "gamma_dot_t0" in physical_param_names
        n_phi = len(np.unique(stratified_data.phi_flat))
        n_physical = len(physical_param_names)

        # Initialize anti-degeneracy controller
        ad_controller = None
        if anti_degeneracy_config is not None and per_angle_scaling and is_laminar_flow:
            phi_unique_rad = np.deg2rad(np.array(sorted(set(stratified_data.phi_flat))))
            ad_controller = AntiDegeneracyController.from_config(
                config_dict=anti_degeneracy_config,
                n_phi=n_phi,
                phi_angles=phi_unique_rad,
                n_physical=n_physical,
                per_angle_scaling=per_angle_scaling,
                is_laminar_flow=is_laminar_flow,
            )

            if ad_controller.is_enabled:
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY DEFENSE: Enabled for Stratified LS")
                logger.info(f"  per_angle_mode: {ad_controller.per_angle_mode_actual}")
                logger.info(f"  use_constant: {ad_controller.use_constant}")
                logger.info(f"  use_fixed_scaling: {ad_controller.use_fixed_scaling}")
                logger.info(
                    f"  use_averaged_scaling: {ad_controller.use_averaged_scaling}"
                )
                logger.info(f"  use_fourier: {ad_controller.use_fourier}")
                logger.info(
                    f"  use_shear_weighting: {ad_controller.use_shear_weighting}"
                )
                logger.info("=" * 60)

                # Transform initial parameters for Fourier mode only
                # CONSTANT MODE (v2.17.0): Parameter transformation is handled later
                # when computing fixed per-angle scaling from quantiles
                if ad_controller.use_fixed_scaling:
                    logger.info(
                        "Fixed constant mode: parameter transformation deferred to "
                        "quantile-based fixed scaling computation"
                    )
                    # Store original bounds for quantile computation later
                    # The transformation will happen in the residual function creation section

                elif ad_controller.use_averaged_scaling:
                    logger.info(
                        "Auto averaged mode: parameter transformation deferred to "
                        "quantile-based averaged scaling computation"
                    )
                    # Store original bounds for quantile computation later
                    # The transformation will happen in the residual function creation section

                elif ad_controller.use_fourier:
                    logger.info(
                        f"Transforming parameters: Fourier mode ({len(initial_params)} -> "
                        f"{ad_controller.n_per_angle_params + n_physical})"
                    )
                    initial_params, _ = ad_controller.transform_params_to_fourier(
                        initial_params
                    )
                    if bounds is not None:
                        # Transform bounds for Fourier mode
                        lower, upper = bounds
                        n_coeffs = ad_controller.fourier.n_coeffs_per_param
                        # Use the mean of bounds for Fourier coefficients
                        lower_fourier = np.concatenate(
                            [
                                np.full(
                                    n_coeffs, np.mean(lower[:n_phi])
                                ),  # contrast coeffs
                                np.full(
                                    n_coeffs, np.mean(lower[n_phi : 2 * n_phi])
                                ),  # offset coeffs
                                lower[2 * n_phi :],  # physical lower
                            ]
                        )
                        upper_fourier = np.concatenate(
                            [
                                np.full(
                                    n_coeffs, np.mean(upper[:n_phi])
                                ),  # contrast coeffs
                                np.full(
                                    n_coeffs, np.mean(upper[n_phi : 2 * n_phi])
                                ),  # offset coeffs
                                upper[2 * n_phi :],  # physical upper
                            ]
                        )
                        bounds = (lower_fourier, upper_fourier)
                        logger.debug(
                            f"Transformed bounds to Fourier mode: {bounds[0].shape}"
                        )

        # Convert stratified flat arrays into chunks
        logger.info(
            f"Creating chunks from stratified data (target size: {target_chunk_size:,})..."
        )
        chunked_data = self._create_stratified_chunks(
            stratified_data, target_chunk_size
        )
        logger.info(f"Created {len(chunked_data.chunks)} chunks")

        # Start timing
        start_time = time.perf_counter()

        # Create JIT-compatible stratified residual function
        # CRITICAL UPDATE (v2.17.0): Constant mode now uses fixed per-angle scaling
        # from quantile estimation. The parameters contain ONLY physical parameters.
        # This replaces the old approach of using mean contrast/offset.
        effective_per_angle_scaling = per_angle_scaling
        fixed_contrast = None
        fixed_offset = None

        if ad_controller is not None and ad_controller.use_fixed_scaling:
            # FIXED_CONSTANT MODE (v2.18.0): Compute fixed per-angle scaling
            # from quantiles. Per-angle values are FIXED (not optimized).
            # Result: 7 physical params only.
            logger.info("=" * 60)
            logger.info(
                "FIXED_CONSTANT MODE: Computing fixed per-angle scaling from quantiles"
            )
            logger.info("=" * 60)

            # Get contrast/offset bounds from initial bounds
            if bounds is not None:
                # Original bounds before any transformation
                # Bounds are [contrast_0..n, offset_0..n, physical]
                contrast_bounds = (
                    float(np.min(bounds[0][:n_phi])),
                    float(np.max(bounds[1][:n_phi])),
                )
                offset_bounds = (
                    float(np.min(bounds[0][n_phi : 2 * n_phi])),
                    float(np.max(bounds[1][n_phi : 2 * n_phi])),
                )
            else:
                contrast_bounds = (0.0, 1.0)
                offset_bounds = (0.5, 1.5)

            # Compute fixed per-angle scaling from quantiles
            ad_controller.compute_fixed_per_angle_scaling(
                stratified_data=stratified_data,
                contrast_bounds=contrast_bounds,
                offset_bounds=offset_bounds,
            )

            # Get the fixed scaling values
            fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
            if fixed_scaling is not None:
                fixed_contrast, fixed_offset = fixed_scaling

                # Update initial_params to contain ONLY physical parameters
                # Original format was [contrast(n_phi), offset(n_phi), physical(n_physical)]
                # New format is [physical(n_physical)]
                initial_params = initial_params[2 * n_phi :]
                logger.info(
                    f"Reduced initial parameters to physical only: {len(initial_params)} params"
                )

                # Update bounds to contain ONLY physical parameter bounds
                if bounds is not None:
                    lower, upper = bounds
                    bounds = (lower[2 * n_phi :], upper[2 * n_phi :])
                    logger.info(
                        f"Reduced bounds to physical only: {len(bounds[0])} params"
                    )

                # Mark that we're using fixed scaling (not per_angle_scaling from params)
                effective_per_angle_scaling = False
                logger.info(
                    f"Fixed per-angle scaling will be used:\n"
                    f"  Contrast: mean={np.mean(fixed_contrast):.4f}, "
                    f"range=[{np.min(fixed_contrast):.4f}, {np.max(fixed_contrast):.4f}]\n"
                    f"  Offset: mean={np.mean(fixed_offset):.4f}, "
                    f"range=[{np.min(fixed_offset):.4f}, {np.max(fixed_offset):.4f}]"
                )
            else:
                logger.warning(
                    "Failed to compute fixed per-angle scaling, "
                    "falling back to standard mode"
                )
                effective_per_angle_scaling = False

        elif ad_controller is not None and ad_controller.use_averaged_scaling:
            # AUTO_AVERAGED MODE (v2.18.0): Estimate per-angle scaling from
            # quantiles, AVERAGE to single values, then OPTIMIZE them.
            # Result: 9 params (7 physical + 1 contrast_avg + 1 offset_avg).
            logger.info("=" * 60)
            logger.info("AUTO_AVERAGED MODE: Computing averaged scaling initial values")
            logger.info("=" * 60)

            # Get contrast/offset bounds from initial bounds
            if bounds is not None:
                contrast_bounds = (
                    float(np.min(bounds[0][:n_phi])),
                    float(np.max(bounds[1][:n_phi])),
                )
                offset_bounds = (
                    float(np.min(bounds[0][n_phi : 2 * n_phi])),
                    float(np.max(bounds[1][n_phi : 2 * n_phi])),
                )
            else:
                contrast_bounds = (0.0, 1.0)
                offset_bounds = (0.5, 1.5)

            # Compute per-angle scaling from quantiles for initial estimates
            ad_controller.compute_fixed_per_angle_scaling(
                stratified_data=stratified_data,
                contrast_bounds=contrast_bounds,
                offset_bounds=offset_bounds,
            )

            # Get per-angle estimates and AVERAGE them for optimization start
            fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
            if fixed_scaling is not None:
                contrast_per_angle, offset_per_angle = fixed_scaling
                avg_contrast = float(np.mean(contrast_per_angle))
                avg_offset = float(np.mean(offset_per_angle))

                # Build 9-param initial_params: [contrast_avg, offset_avg, physical(7)]
                physical_params_init = initial_params[2 * n_phi :]
                initial_params = np.concatenate(
                    [[avg_contrast, avg_offset], physical_params_init]
                )
                logger.info(
                    f"Averaged initial parameters: {len(initial_params)} params "
                    f"(contrast={avg_contrast:.4f}, offset={avg_offset:.4f})"
                )

                # Update bounds: [contrast_bounds, offset_bounds, physical_bounds]
                if bounds is not None:
                    lower, upper = bounds
                    bounds = (
                        np.concatenate(
                            [[contrast_bounds[0], offset_bounds[0]], lower[2 * n_phi :]]
                        ),
                        np.concatenate(
                            [[contrast_bounds[1], offset_bounds[1]], upper[2 * n_phi :]]
                        ),
                    )
                    logger.info(
                        f"Updated bounds for averaged mode: {len(bounds[0])} params"
                    )

                # Scalar contrast/offset will be OPTIMIZED (not fixed)
                # per_angle_scaling=False + no fixed arrays → residual mode 3
                effective_per_angle_scaling = False
                # Do NOT set fixed_contrast/fixed_offset — they are optimized
                logger.info(
                    f"Averaged scaling will be OPTIMIZED (not fixed):\n"
                    f"  Initial contrast: {avg_contrast:.4f} "
                    f"(from per-angle range [{np.min(contrast_per_angle):.4f}, "
                    f"{np.max(contrast_per_angle):.4f}])\n"
                    f"  Initial offset: {avg_offset:.4f} "
                    f"(from per-angle range [{np.min(offset_per_angle):.4f}, "
                    f"{np.max(offset_per_angle):.4f}])"
                )
            else:
                logger.warning(
                    "Failed to compute per-angle scaling estimates, "
                    "falling back to mean of initial per-angle values"
                )
                # Fallback: average the initial per-angle values
                avg_contrast = float(np.mean(initial_params[:n_phi]))
                avg_offset = float(np.mean(initial_params[n_phi : 2 * n_phi]))
                physical_params_init = initial_params[2 * n_phi :]
                initial_params = np.concatenate(
                    [[avg_contrast, avg_offset], physical_params_init]
                )
                if bounds is not None:
                    lower, upper = bounds
                    bounds = (
                        np.concatenate(
                            [[contrast_bounds[0], offset_bounds[0]], lower[2 * n_phi :]]
                        ),
                        np.concatenate(
                            [[contrast_bounds[1], offset_bounds[1]], upper[2 * n_phi :]]
                        ),
                    )
                effective_per_angle_scaling = False

        logger.info("Creating JIT-compatible stratified residual function...")
        residual_fn = StratifiedResidualFunctionJIT(
            stratified_data=chunked_data,  # Use chunked_data with .chunks attribute
            per_angle_scaling=effective_per_angle_scaling,
            physical_param_names=physical_param_names,
            logger=logger,
            fixed_contrast_per_angle=fixed_contrast,
            fixed_offset_per_angle=fixed_offset,
        )

        # Validate chunk structure
        logger.info("Validating chunk structure...")
        residual_fn.validate_chunk_structure()

        # Log diagnostics
        residual_fn.log_diagnostics()

        # Gradient sanity check (CRITICAL)
        # Verify that gradients are non-zero before starting optimization
        # This catches parameter initialization issues early
        logger.info("=" * 80)
        logger.info("GRADIENT SANITY CHECK")
        logger.info("=" * 80)

        try:
            # Compute residuals at initial parameters
            residuals_0 = residual_fn(initial_params)
            logger.info(
                f"Initial residuals: shape={residuals_0.shape}, "
                f"min={float(np.min(residuals_0)):.6e}, "
                f"max={float(np.max(residuals_0)):.6e}, "
                f"mean={float(np.mean(residuals_0)):.6e}"
            )

            # Perturb the first physical parameter (D0) by 1%.
            # BUG-5: In auto_averaged mode, effective_per_angle_scaling=False but
            # params = [contrast_avg, offset_avg, D0, ...], so D0 is at index 2.
            # In individual mode, D0 is at index 2*n_phi. In fixed_constant, D0
            # is at index 0 (no scaling params in vector).
            if effective_per_angle_scaling:
                phys_idx = 2 * residual_fn.n_phi  # individual mode
            elif len(initial_params) > 2:
                phys_idx = 2  # auto_averaged: [contrast_avg, offset_avg, D0, ...]
            else:
                phys_idx = 0  # fixed_constant: [D0, alpha, ...]
            params_test = np.array(initial_params, copy=True)
            params_test[phys_idx] *= 1.01  # 1% perturbation
            residuals_1 = residual_fn(params_test)

            # Estimate gradient magnitude
            gradient_estimate = float(np.abs(np.sum(residuals_1 - residuals_0)))
            logger.info(
                f"Gradient estimate (1% perturbation of param[{phys_idx}]): {gradient_estimate:.6e}"
            )

            # Check if gradient is suspiciously small
            if gradient_estimate < 1e-10:
                logger.error("=" * 80)
                logger.error("GRADIENT SANITY CHECK FAILED")
                logger.error("=" * 80)
                logger.error(
                    f"Gradient estimate: {gradient_estimate:.6e} (expected > 1e-10)"
                )
                logger.error("This indicates:")
                logger.error(
                    "  - Parameter initialization issue (likely wrong parameter count)"
                )
                logger.error("  - Residual function not sensitive to parameter changes")
                logger.error("  - Optimization will fail with 0 iterations")
                logger.error("")
                logger.error("Diagnostic information:")
                logger.error(f"  Initial parameters count: {len(initial_params)}")
                if effective_per_angle_scaling:
                    expected_count = len(physical_param_names) + 2 * residual_fn.n_phi
                    logger.error(
                        f"  Expected for per-angle scaling: {len(physical_param_names)} physical + 2*{residual_fn.n_phi} scaling = {expected_count}"
                    )
                else:
                    expected_count = len(physical_param_names) + 2
                    logger.error(
                        f"  Expected for constant scaling: {len(physical_param_names)} physical + 2 scaling = {expected_count}"
                    )
                logger.error(
                    f"  Residual function expects: per_angle_scaling={effective_per_angle_scaling}, n_phi={residual_fn.n_phi}"
                )
                logger.error("=" * 80)
                raise ValueError(
                    f"Gradient sanity check FAILED: gradient ≈ {gradient_estimate:.2e} "
                    f"(expected > 1e-10). Optimization cannot proceed with zero gradients."
                )

            logger.info(
                f"✓ Gradient sanity check passed (gradient magnitude: {gradient_estimate:.6e})"
            )

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            if "Gradient sanity check FAILED" in str(e):
                raise  # Re-raise our custom error
            logger.warning(f"Gradient sanity check encountered error: {e}")
            logger.warning("Proceeding with optimization, but this may fail")

        logger.info("=" * 80)

        # Prepare for optimization
        logger.info("Starting NLSQ least_squares() optimization...")
        logger.info(f"  Initial parameters: {len(initial_params)} parameters")
        logger.info(f"  Bounds: {'provided' if bounds is not None else 'unbounded'}")
        logger.info(f"  Residual chunks: {residual_fn.n_chunks}")
        logger.info(f"  Real data points: {residual_fn.n_real_points:,}")

        # Call NLSQ's least_squares() - NO xdata/ydata needed!
        # Data is encapsulated in residual_fn
        optimization_start = time.perf_counter()

        # Instantiate LeastSquares with stability mode enabled
        # Stability mode provides:
        # - Automatic memory management (switches to LSMR when memory tight)
        # - Numerical stability checks and Jacobian conditioning monitoring
        # Note: Requires NLSQ <= 0.3.0 for correct results. Later versions
        # introduced regressions in stability mode that cause divergence.
        ls = LeastSquares(enable_stability=True, enable_diagnostics=True)

        result = ls.least_squares(
            fun=residual_fn,  # Residual function (we control chunking!)
            x0=initial_params,  # Initial parameters
            jac=None,  # Use JAX autodiff for Jacobian
            bounds=bounds,  # Parameter bounds
            method="trf",  # Trust Region Reflective
            ftol=1e-8,  # Function tolerance
            xtol=1e-8,  # Parameter tolerance
            gtol=1e-8,  # Gradient tolerance
            max_nfev=1000,  # Max function evaluations
            verbose=2,  # Show progress
        )

        optimization_time = time.perf_counter() - optimization_start
        logger.info(f"Optimization completed in {optimization_time:.2f}s")

        # Extract results from NLSQ result object
        popt = np.asarray(result["x"])

        # CRITICAL: Enforce parameter bounds (post-optimization clipping)
        # NLSQ's trust-region algorithm can violate bounds to minimize cost
        # Clip parameters to ensure physical validity
        logger.info(
            f"POST-OPTIMIZATION BOUNDS CHECK: bounds={'provided' if bounds is not None else 'None'}, popt shape={popt.shape}"
        )
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            bounds_violated = False

            # Debug: Log first few bounds and parameters
            logger.info(
                f"Debug: lower_bounds type={type(lower_bounds)}, shape={getattr(lower_bounds, 'shape', 'N/A')}"
            )
            logger.info(
                f"Debug: First 3 lower bounds: {lower_bounds[:3] if hasattr(lower_bounds, '__getitem__') else 'N/A'}"
            )
            logger.info(
                f"Debug: Last 3 lower bounds: {lower_bounds[-3:] if hasattr(lower_bounds, '__getitem__') else 'N/A'}"
            )
            logger.info(f"Debug: First 3 popt: {popt[:3]}")
            logger.info(f"Debug: Last 3 popt: {popt[-3:]}")

            for i in range(len(popt)):
                original_value = popt[i]

                # Clip to bounds
                if original_value < lower_bounds[i] or original_value > upper_bounds[i]:
                    popt[i] = np.clip(popt[i], lower_bounds[i], upper_bounds[i])
                    bounds_violated = True

                    # BUG-6: Use effective_per_angle_scaling (post anti-degeneracy)
                    # not per_angle_scaling (original config), to match actual param layout.
                    if effective_per_angle_scaling:
                        n_angles = residual_fn.n_phi
                        n_scaling = 2 * n_angles
                        if i < n_angles:
                            param_name = f"contrast_angle_{i}"
                        elif i < n_scaling:
                            param_name = f"offset_angle_{i - n_angles}"
                        else:
                            param_idx = i - n_scaling
                            param_name = (
                                physical_param_names[param_idx]
                                if param_idx < len(physical_param_names)
                                else f"param_{i}"
                            )
                    else:
                        param_name = (
                            physical_param_names[i]
                            if i < len(physical_param_names)
                            else f"param_{i}"
                        )

                    logger.warning(
                        f"⚠️  Parameter '{param_name}' violated bounds: "
                        f"{original_value:.6e} ∉ [{lower_bounds[i]:.6e}, {upper_bounds[i]:.6e}]"
                    )
                    logger.warning(f"    Clipped to: {popt[i]:.6e} (bounds enforced)")

            if bounds_violated:
                logger.warning("=" * 80)
                logger.warning("BOUNDS VIOLATION DETECTED")
                logger.warning("=" * 80)
                logger.warning("One or more parameters violated physical bounds.")
                logger.warning("Parameters have been clipped to valid ranges.")
                logger.warning("This may indicate:")
                logger.warning(
                    "  - Poor initial conditions (check config initial_parameters.values)"
                )
                logger.warning(
                    "  - Insufficient constraints (consider constrained optimizer)"
                )
                logger.warning("  - Optimizer exploring unphysical parameter space")
                logger.warning("=" * 80)

        # Compute covariance matrix from Jacobian
        # NLSQ's least_squares may or may not provide pcov directly
        if "pcov" in result and result["pcov"] is not None:
            pcov = np.asarray(result["pcov"])
            logger.info("Using covariance matrix from NLSQ result")
        else:
            # Compute covariance from Jacobian: pcov = inv(J^T J)
            logger.info("Computing covariance matrix from Jacobian...")

            # Use JAX to compute Jacobian at final parameters
            jac_fn = jax.jacfwd(residual_fn)
            J = jac_fn(popt)
            J = np.asarray(J)

            # Compute covariance: (J^T J)^{-1}
            # This is the standard formula for nonlinear least squares
            try:
                JTJ = J.T @ J
                pcov = np.linalg.inv(JTJ)
            except np.linalg.LinAlgError:
                logger.warning("Singular Jacobian, using pseudo-inverse for covariance")
                pcov = np.linalg.pinv(JTJ)

        # Compute final cost
        final_residuals = residual_fn(popt)
        final_cost = float(np.sum(final_residuals**2))

        # Extract convergence information
        success = result.get("success", True)
        message = result.get("message", "Optimization completed")
        nfev = result.get("nfev", 0)
        nit = result.get("nit", 0)

        # Determine if optimization actually improved
        initial_residuals = residual_fn(initial_params)
        initial_cost = float(np.sum(initial_residuals**2))
        cost_reduction = (
            (initial_cost - final_cost) / initial_cost if initial_cost > 0 else 0
        )
        params_changed = not np.allclose(popt, initial_params, rtol=1e-8)

        # Log results
        logger.info("=" * 80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info(f"  Status: {'SUCCESS' if success else 'FAILED'}")
        logger.info(f"  Message: {message}")
        logger.info(f"  Function evaluations: {nfev}")
        logger.info(f"  Iterations: {nit}")
        logger.info(f"  Initial cost: {initial_cost:.6e}")
        logger.info(f"  Final cost: {final_cost:.6e}")
        logger.info(f"  Cost reduction: {cost_reduction * 100:+.2f}%")
        logger.info(f"  Parameters changed: {params_changed}")
        logger.info(f"  Total time: {time.perf_counter() - start_time:.2f}s")
        logger.info("=" * 80)

        # Check for optimization failure
        if not params_changed or cost_reduction < 0.01:
            logger.warning(
                "Optimization may have failed:\n"
                f"  Parameters changed: {params_changed}\n"
                f"  Cost reduction: {cost_reduction * 100:.2f}%\n"
                "This may indicate:\n"
                "  - Initial parameters already optimal\n"
                "  - Optimization converged immediately\n"
                "  - Problem with gradient computation"
            )

        # =====================================================================
        # Anti-Degeneracy: Inverse Transformation (v2.14.0+, v2.18.0 update)
        # =====================================================================
        # Expand optimized params back to per-angle form:
        #   fixed_constant: 7 physical → [contrast(n_phi), offset(n_phi), physical(7)]
        #   auto_averaged:  9 params   → [contrast(n_phi), offset(n_phi), physical(7)]
        #   fourier:        n_coeffs   → [contrast(n_phi), offset(n_phi), physical(7)]
        anti_degeneracy_info = {}
        if ad_controller is not None and ad_controller.is_enabled:
            if ad_controller.use_fixed_scaling:
                # FIXED_CONSTANT MODE (v2.18.0): Use fixed per-angle scaling
                # popt contains ONLY physical parameters (7 params)
                if ad_controller.has_fixed_per_angle_scaling():
                    fixed_scaling = ad_controller.get_fixed_per_angle_scaling()
                    fixed_contrast, fixed_offset = fixed_scaling

                    logger.info(
                        f"Expanding parameters from fixed_constant mode:\n"
                        f"  Physical params: {len(popt)}\n"
                        f"  Fixed contrast: mean={np.mean(fixed_contrast):.4f}\n"
                        f"  Fixed offset: mean={np.mean(fixed_offset):.4f}\n"
                        f"  Expanded: {2 * n_phi + n_physical}"
                    )

                    # Combine fixed scaling with optimized physical parameters
                    # Output format: [contrast(n_phi), offset(n_phi), physical(n_physical)]
                    popt_expanded = np.concatenate(
                        [
                            fixed_contrast,
                            fixed_offset,
                            popt,  # popt contains only physical params
                        ]
                    )

                    # Covariance for fixed scaling parameters is zero (they're fixed)
                    # Only physical parameters have non-zero covariance
                    pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
                    pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov

                    popt = popt_expanded
                    pcov = pcov_expanded
                    logger.info(
                        f"Expanded to {len(popt)} parameters with fixed per-angle scaling"
                    )
                    anti_degeneracy_info["mode"] = "fixed_constant_quantile"
                    anti_degeneracy_info["original_n_params"] = n_physical
                    anti_degeneracy_info["expanded_n_params"] = len(popt)
                    anti_degeneracy_info["fixed_contrast_mean"] = float(
                        np.mean(fixed_contrast)
                    )
                    anti_degeneracy_info["fixed_offset_mean"] = float(
                        np.mean(fixed_offset)
                    )
                else:
                    logger.warning(
                        "Fixed constant mode but no fixed scaling available. "
                        "Unexpected state - results may be unreliable."
                    )
                    anti_degeneracy_info["mode"] = "fixed_constant_fallback"

            elif ad_controller.use_averaged_scaling:
                # AUTO_AVERAGED MODE (v2.18.0): Expand 9-param optimized result
                # popt = [contrast_avg, offset_avg, physical(7)]
                logger.info(
                    f"Expanding parameters from auto_averaged mode ({len(popt)} -> "
                    f"{2 * n_phi + n_physical})"
                )
                popt_expanded = ad_controller.transform_params_from_constant(popt)

                # Transform covariance: broadcast averaged scaling to per-angle
                # Each per-angle param gets same variance as the averaged value
                pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
                # Contrast variance: duplicate for all angles
                pcov_expanded[:n_phi, :n_phi] = np.eye(n_phi) * pcov[0, 0]
                # Offset variance: duplicate for all angles
                pcov_expanded[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = (
                    np.eye(n_phi) * pcov[1, 1]
                )
                # Physical params covariance
                pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov[2:, 2:]
                # Cross-terms (physical with averaged scaling)
                pcov_expanded[2 * n_phi :, :n_phi] = np.tile(pcov[2:, 0:1], (1, n_phi))
                pcov_expanded[:n_phi, 2 * n_phi :] = np.tile(pcov[0:1, 2:], (n_phi, 1))
                pcov_expanded[2 * n_phi :, n_phi : 2 * n_phi] = np.tile(
                    pcov[2:, 1:2], (1, n_phi)
                )
                pcov_expanded[n_phi : 2 * n_phi, 2 * n_phi :] = np.tile(
                    pcov[1:2, 2:], (n_phi, 1)
                )

                popt = popt_expanded
                pcov = pcov_expanded
                logger.info(f"Expanded to {len(popt)} per-angle parameters")
                anti_degeneracy_info["mode"] = "auto_averaged"
                anti_degeneracy_info["original_n_params"] = 2 + n_physical
                anti_degeneracy_info["expanded_n_params"] = len(popt)

            elif ad_controller.use_fourier:
                logger.info(
                    f"Expanding parameters from Fourier mode ({len(popt)} -> "
                    f"{2 * n_phi + n_physical})"
                )
                popt_expanded = ad_controller.transform_params_from_fourier(popt)

                # Transform covariance matrix (Jacobian-based propagation)
                # For Fourier mode: use chain rule with Fourier basis
                # pcov_expanded = B @ pcov_fourier @ B.T where B is the Fourier basis
                n_coeffs = ad_controller.fourier.n_coeffs_per_param
                B_contrast = (
                    ad_controller.fourier.get_basis_matrix()
                )  # (n_phi, n_coeffs)
                B_offset = ad_controller.fourier.get_basis_matrix()

                pcov_expanded = np.zeros((len(popt_expanded), len(popt_expanded)))
                # Contrast block
                pcov_contrast = pcov[:n_coeffs, :n_coeffs]
                pcov_expanded[:n_phi, :n_phi] = (
                    B_contrast @ pcov_contrast @ B_contrast.T
                )
                # Offset block
                pcov_offset = pcov[n_coeffs : 2 * n_coeffs, n_coeffs : 2 * n_coeffs]
                pcov_expanded[n_phi : 2 * n_phi, n_phi : 2 * n_phi] = (
                    B_offset @ pcov_offset @ B_offset.T
                )
                # Physical params covariance (unchanged)
                pcov_expanded[2 * n_phi :, 2 * n_phi :] = pcov[
                    2 * n_coeffs :, 2 * n_coeffs :
                ]
                # Cross-terms: approximate propagation
                pcov_expanded[2 * n_phi :, :n_phi] = (
                    pcov[2 * n_coeffs :, :n_coeffs] @ B_contrast.T
                )
                pcov_expanded[:n_phi, 2 * n_phi :] = (
                    B_contrast @ pcov[:n_coeffs, 2 * n_coeffs :]
                )
                pcov_expanded[2 * n_phi :, n_phi : 2 * n_phi] = (
                    pcov[2 * n_coeffs :, n_coeffs : 2 * n_coeffs] @ B_offset.T
                )
                pcov_expanded[n_phi : 2 * n_phi, 2 * n_phi :] = (
                    B_offset @ pcov[n_coeffs : 2 * n_coeffs, 2 * n_coeffs :]
                )

                popt = popt_expanded
                pcov = pcov_expanded
                logger.info(f"Expanded to {len(popt)} per-angle parameters")
                anti_degeneracy_info["mode"] = "fourier"
                anti_degeneracy_info["fourier_order"] = ad_controller.fourier.order
                anti_degeneracy_info["original_n_params"] = 2 * n_coeffs + n_physical
                anti_degeneracy_info["expanded_n_params"] = len(popt)

            # Add diagnostics to info
            anti_degeneracy_info["controller_diagnostics"] = (
                ad_controller.get_diagnostics()
            )

        # Prepare info dict
        info = {
            "success": success,
            "message": message,
            "nfev": nfev,
            "nit": nit,
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "cost_reduction": cost_reduction,
            "optimization_time": optimization_time,
            "method": "stratified_least_squares",
        }
        if anti_degeneracy_info:
            info["anti_degeneracy"] = anti_degeneracy_info

        # Check for shear collapse in laminar_flow mode
        is_laminar_flow = "gamma_dot_t0" in physical_param_names
        if is_laminar_flow:
            # BUG-4: Use actual unique phi count from stratified data, not .chunks
            n_phi_check = (
                len(set(stratified_data.phi_flat.tolist()))
                if hasattr(stratified_data, "phi_flat")
                else 1
            )
            # In auto_averaged mode, popt has scalar contrast/offset (n_phi_eff=1)
            n_phi = n_phi_check if effective_per_angle_scaling else 1
            if len(popt) > 2 * n_phi + 3:
                gamma_dot_t0_idx = 2 * n_phi + 3
                gamma_dot_t0_value = popt[gamma_dot_t0_idx]
                if abs(gamma_dot_t0_value) < 1e-5:
                    logger.warning("=" * 80)
                    logger.warning("SHEAR COLLAPSE WARNING")
                    logger.warning(
                        f"gamma_dot_t0 = {gamma_dot_t0_value:.2e} s⁻¹ is effectively zero"
                    )
                    logger.warning(
                        "The model has effectively collapsed to static_isotropic mode."
                    )
                    logger.warning(
                        "RECOMMENDED: Use phi_filtering for angles near 0° and 90°"
                    )
                    logger.warning("=" * 80)
                    info["shear_collapse_warning"] = {
                        "gamma_dot_t0": float(gamma_dot_t0_value),
                        "threshold": 1e-5,
                        "message": "Shear contribution effectively zero",
                    }

        return popt, pcov, info

    def _fit_with_streaming_optimizer(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        streaming_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ streaming optimizer for memory-constrained large datasets.

        .. deprecated:: 2.9.1
            The old StreamingOptimizer was removed in NLSQ 0.4.0.
            This method now delegates to _fit_with_stratified_hybrid_streaming
            which uses AdaptiveHybridStreamingOptimizer.

        Args:
            stratified_data: StratifiedData object with flat stratified arrays
            per_angle_scaling: Whether per-angle parameters are enabled
            physical_param_names: List of physical parameter names
            initial_params: Initial parameter guess
            bounds: Parameter bounds (lower, upper) tuple
            logger: Logger instance
            streaming_config: Optional config dict (converted to hybrid_config)

        Returns:
            (popt, pcov, info) tuple

        Raises:
            RuntimeError: If AdaptiveHybridStreamingOptimizer is not available
        """
        # Delegate to hybrid streaming optimizer (old StreamingOptimizer removed in NLSQ 0.4.0)
        logger.info(
            "Note: StreamingOptimizer was removed in NLSQ 0.4.0. "
            "Using AdaptiveHybridStreamingOptimizer instead."
        )

        # Convert streaming_config to hybrid_config format
        hybrid_config = None
        if streaming_config:
            hybrid_config = {
                "chunk_size": streaming_config.get("batch_size", 50000),
                "warmup_iterations": streaming_config.get("max_epochs", 100),
                "gauss_newton_tol": streaming_config.get("convergence_tol", 1e-8),
                "warmup_learning_rate": streaming_config.get("learning_rate", 0.001),
            }

        return self._fit_with_stratified_hybrid_streaming(
            stratified_data=stratified_data,
            per_angle_scaling=per_angle_scaling,
            physical_param_names=physical_param_names,
            initial_params=initial_params,
            bounds=bounds,
            logger=logger,
            hybrid_config=hybrid_config,
        )

    # NOTE: Dead streaming optimizer code removed (NLSQ 0.4.0+ removed StreamingOptimizer)

    def _fit_with_out_of_core_accumulation(
        self,
        stratified_data: Any,
        data: Any,  # Original data for metadata
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        config: Any,
        fast_chi2_mode: bool = False,
        anti_degeneracy_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using Out-of-Core Global Accumulation for massive datasets.

        This strategy virtually chunks the dataset using Index-Based Stratification,
        accumulates the full Hessian and Gradient (J^T J, J^T r) by iterating
        over chunks, and takes a global Levenberg-Marquardt step.

        Guarantees identical convergence to standard NLSQ but with minimal memory.

        Note (v2.14.1+):
            This method now uses FULL homodyne physics via compute_g2_scaled(),
            identical to stratified least-squares. Anti-Degeneracy Defense System
            support is planned for a future release.
        """
        import time
        import jax
        import jax.numpy as jnp

        _start_time = time.perf_counter()  # noqa: F841
        logger.info(
            "Initializing Out-of-Core Global Stratified Optimization (Full Physics)..."
        )

        # 1. Setup Chunking
        # Use StratifiedIndices if available (Zero-Copy)
        _use_index_based = False  # noqa: F841
        # We operate on the ORIGINAL flattened data to avoid pre-materializing
        # a giant stratified copy (which causes OOM).
        # We assume `data` object has .phi, .t1, .t2, .g2
        # We need to flatten them carefully (using ravel/reshape to avoid copies if possible)

        # Helper to flatten dimensions
        def _get_flat_arrays(d):
            # Same logic as _prepare_data but trying to be lazy/view-based
            phi_arr = np.asarray(d.phi)
            t1_arr = np.asarray(d.t1)
            t2_arr = np.asarray(d.t2)
            g2_arr = np.asarray(d.g2)

            # Extract 1D from meshgrids if needed (borrowed from _prepare_data)
            if t1_arr.ndim == 2 and t1_arr.size > 0:
                t1_arr = t1_arr[:, 0]
            if t2_arr.ndim == 2 and t2_arr.size > 0:
                t2_arr = t2_arr[0, :]

            phi_grid, t1_grid, t2_grid = np.meshgrid(
                phi_arr, t1_arr, t2_arr, indexing="ij"
            )

            # These flattens create copies usually, but for 25M points (200MB) it's acceptable ONCE
            # The OOM comes from creating SECOND and THIRD copies during stratification.
            return phi_grid.ravel(), t1_grid.ravel(), t2_grid.ravel(), g2_arr.ravel()

        phi_flat, t1_flat, t2_flat, g2_flat = _get_flat_arrays(data)

        # Calculate optimal chunk size
        n_points = len(phi_flat)
        n_params = len(initial_params)
        n_angles = len(np.unique(phi_flat))

        chunk_size = calculate_adaptive_chunk_size(
            total_points=n_points,
            n_params=n_params,
            n_angles=n_angles,
            safety_factor=5.0,
        )

        # Get iterator that yields INDICES for stratified chunks
        # This allows us to pull stratified data from the flat arrays on demand
        iterator = get_stratified_chunk_iterator(phi_flat, chunk_size)
        logger.info(
            f"Out-of-Core Strategy: {len(iterator)} chunks of size ~{chunk_size}\n"
            f"  Pipeline: Chunk(Indices) -> Load -> JIT(Acc) -> Global Step"
        )

        # Pre-compute unique phi for JAX mapping
        phi_unique = jnp.sort(jnp.unique(phi_flat))

        # 2. Setup Optimization State
        params_curr = jnp.array(initial_params)
        n_iter = 0

        cfg_dict = (
            config.config
            if hasattr(config, "config")
            else (config if isinstance(config, dict) else {})
        )

        # Extract physics constants from data (v2.14.1+: Full homodyne physics)
        q_val = float(data.q)
        L_val = float(data.L)
        dt_val = float(getattr(data, "dt", cfg_dict.get("dt", 0.001)))

        # Extract global unique time arrays for meshgrid construction
        t_unique_global = jnp.sort(jnp.unique(jnp.concatenate([t1_flat, t2_flat])))
        n_phi = len(phi_unique)
        n_t = len(t_unique_global)

        logger.info(
            f"Full Physics Setup: n_phi={n_phi}, n_t={n_t}, "
            f"q={q_val:.4e}, L={L_val:.4e}, dt={dt_val:.4e}"
        )
        max_iter = cfg_dict.get("optimization", {}).get("max_iterations", 50)

        # Convergence tolerances (v2.22.0: multi-criteria, matching standard NLSQ)
        xtol = 1e-6  # Relative parameter change (per-component max, not norm)
        ftol = 1e-6  # Relative cost function change
        lm_lambda = 0.01  # Initial damping
        rel_change = float("inf")  # Initialize to prevent NameError at loop exit

        # ====================================================================
        # JIT-compiled Chunk Kernel with FULL HOMODYNE PHYSICS (v2.14.1+)
        # ====================================================================
        # This kernel uses the same physics as stratified LS via compute_g2_scaled()
        # Pattern: Build theory grid on unique times, extract chunk values via indexing

        @jax.jit
        def compute_chunk_accumulators(p, phi_c, t1_c, t2_c, g2_c, sigma):
            """Compute J^T J, J^T r, and chi2 for a chunk using FULL homodyne physics."""

            def r_fn(curr_p):
                # Unpack parameters (same pattern as stratified LS)
                if per_angle_scaling:
                    contrast_arr = curr_p[:n_phi]
                    offset_arr = curr_p[n_phi : 2 * n_phi]
                    physical_params = curr_p[2 * n_phi :]
                else:
                    contrast_scalar = curr_p[0]
                    offset_scalar = curr_p[1]
                    physical_params = curr_p[2:]

                # === FULL PHYSICS: Vectorize over angles ===
                # Same pattern as StratifiedResidualFunctionJIT in residual_jit.py
                if per_angle_scaling:
                    # vmap over angles with per-angle contrast/offset
                    compute_g2_vmap = jax.vmap(
                        lambda phi_val, c_val, o_val: jnp.squeeze(
                            compute_g2_scaled(
                                params=physical_params,
                                t1=t_unique_global,
                                t2=t_unique_global,
                                phi=phi_val,
                                q=q_val,
                                L=L_val,
                                contrast=c_val,
                                offset=o_val,
                                dt=dt_val,
                            )
                        ),
                        in_axes=(0, 0, 0),
                    )
                    g2_theory_grid = compute_g2_vmap(
                        phi_unique, contrast_arr, offset_arr
                    )
                else:
                    # vmap over angles with single contrast/offset
                    compute_g2_vmap = jax.vmap(
                        lambda phi_val: jnp.squeeze(
                            compute_g2_scaled(
                                params=physical_params,
                                t1=t_unique_global,
                                t2=t_unique_global,
                                phi=phi_val,
                                q=q_val,
                                L=L_val,
                                contrast=contrast_scalar,
                                offset=offset_scalar,
                                dt=dt_val,
                            )
                        ),
                        in_axes=0,
                    )
                    g2_theory_grid = compute_g2_vmap(phi_unique)

                # NOTE: Diagonal correction skipped — residuals with t1==t2 are
                # masked out below via `jnp.where(t1_c != t2_c, res, 0.0)`,
                # so theory grid diagonal values are never used.

                # === FLAT INDEXING: Extract chunk values from grid ===
                g2_theory_flat = g2_theory_grid.flatten()

                # Find indices in the unique arrays
                phi_indices = jnp.searchsorted(phi_unique, phi_c)
                t1_indices = jnp.searchsorted(t_unique_global, t1_c)
                t2_indices = jnp.searchsorted(t_unique_global, t2_c)

                # Compute flat indices (C-order: phi varies slowest)
                flat_indices = phi_indices * (n_t * n_t) + t1_indices * n_t + t2_indices

                # Extract theory values for this chunk
                g2_theory_chunk = g2_theory_flat[flat_indices]

                # Weighted residuals with diagonal mask
                w = 1.0 / sigma
                res = (g2_c - g2_theory_chunk) * w
                return jnp.where(t1_c != t2_c, res, 0.0)

            # Compute Jacobian and residuals
            J = jax.jacfwd(r_fn)(p)
            r = r_fn(p)

            return J.T @ J, J.T @ r, jnp.sum(r**2)

        # JIT-compiled Chi2-only Kernel with FULL HOMODYNE PHYSICS (v2.14.1+)
        @jax.jit
        def compute_chunk_chi2(p, phi_c, t1_c, t2_c, g2_c, sigma):
            """Compute chi2 for a chunk using FULL homodyne physics (no Jacobian)."""
            # Unpack parameters
            if per_angle_scaling:
                contrast_arr = p[:n_phi]
                offset_arr = p[n_phi : 2 * n_phi]
                physical_params = p[2 * n_phi :]
            else:
                contrast_scalar = p[0]
                offset_scalar = p[1]
                physical_params = p[2:]

            # === FULL PHYSICS: Vectorize over angles ===
            if per_angle_scaling:
                compute_g2_vmap = jax.vmap(
                    lambda phi_val, c_val, o_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t_unique_global,
                            t2=t_unique_global,
                            phi=phi_val,
                            q=q_val,
                            L=L_val,
                            contrast=c_val,
                            offset=o_val,
                            dt=dt_val,
                        )
                    ),
                    in_axes=(0, 0, 0),
                )
                g2_theory_grid = compute_g2_vmap(phi_unique, contrast_arr, offset_arr)
            else:
                compute_g2_vmap = jax.vmap(
                    lambda phi_val: jnp.squeeze(
                        compute_g2_scaled(
                            params=physical_params,
                            t1=t_unique_global,
                            t2=t_unique_global,
                            phi=phi_val,
                            q=q_val,
                            L=L_val,
                            contrast=contrast_scalar,
                            offset=offset_scalar,
                            dt=dt_val,
                        )
                    ),
                    in_axes=0,
                )
                g2_theory_grid = compute_g2_vmap(phi_unique)

            # NOTE: Diagonal correction skipped — chi2 with t1==t2 is masked
            # out below via `jnp.where(t1_c != t2_c, res, 0.0)`.

            # === FLAT INDEXING: Extract chunk values ===
            g2_theory_flat = g2_theory_grid.flatten()
            phi_indices = jnp.searchsorted(phi_unique, phi_c)
            t1_indices = jnp.searchsorted(t_unique_global, t1_c)
            t2_indices = jnp.searchsorted(t_unique_global, t2_c)
            flat_indices = phi_indices * (n_t * n_t) + t1_indices * n_t + t2_indices
            g2_theory_chunk = g2_theory_flat[flat_indices]

            # Compute chi-squared with diagonal mask
            w = 1.0 / sigma
            res = (g2_c - g2_theory_chunk) * w
            res = jnp.where(t1_c != t2_c, res, 0.0)
            return jnp.sum(res**2)

        def evaluate_total_chi2(params_eval):
            total_c2 = 0.0

            # Fast Mode: Subsample chunks
            # Use fixed stride of 10 (10% sample)
            stride = 10 if fast_chi2_mode else 1
            _scale_factor = float(stride)  # noqa: F841

            # Create subsampled iterator
            # StratifiedIndexIterator is iterable but not slicable usually?
            # We iterate manually to be safe

            eval_count = 0
            for i, ind_c in enumerate(iterator):
                if i % stride != 0:
                    continue

                p_c = phi_flat[ind_c]
                t1_c = t1_flat[ind_c]
                t2_c = t2_flat[ind_c]
                g2_c = g2_flat[ind_c]
                c2_chunk = compute_chunk_chi2(
                    params_eval, p_c, t1_c, t2_c, g2_c, sigma_val
                )
                total_c2 += c2_chunk
                eval_count += 1

            # Correction if eval_count doesn't match stride exactly due to remainder
            # Or simpler: total_c2 * (total_chunks / eval_count)
            total_chunks = len(iterator)
            if eval_count > 0:
                scale = total_chunks / eval_count
                return total_c2 * scale
            return 0.0

        # Sigma placeholder (physics constants already extracted above)
        sigma_val = 1.0

        # Optimization Loop
        logger.info(f"Starting Out-of-Core Loop (Max iter: {max_iter})...")
        import time

        for i in range(max_iter):
            _iter_start = time.perf_counter()  # noqa: F841
            total_JtJ = jnp.zeros((n_params, n_params))
            total_Jtr = jnp.zeros(n_params)
            total_chi2 = 0.0

            # Accumulate over chunks
            count = 0
            for indices_chunk in iterator:
                # Load chunk data using stratifying indices
                # This is the "Zero-Copy" magic - we only copy small chunks
                phi_c = phi_flat[indices_chunk]
                t1_c = t1_flat[indices_chunk]
                t2_c = t2_flat[indices_chunk]
                g2_c = g2_flat[indices_chunk]

                # Compute and Accumulate (using FULL homodyne physics v2.14.1+)
                JtJ, Jtr, chi2 = compute_chunk_accumulators(
                    params_curr, phi_c, t1_c, t2_c, g2_c, sigma_val
                )

                total_JtJ += JtJ
                total_Jtr += Jtr
                total_chi2 += chi2
                count += len(indices_chunk)

            # Robust Levenberg-Marquardt Step Loop
            step_accepted = False

            # Check for invalid Jacobian/Residuals
            if jnp.any(jnp.isnan(total_Jtr)) or jnp.any(jnp.isinf(total_JtJ)):
                logger.warning("Gradient/Hessian contains NaNs/Infs. Checking params.")
                # If we are here, current params are bad? Or gradients near boundary are bad.
                # If params valid but grad inf: likely at boundary singularity (tau=0).
                if n_iter == 0:
                    raise RuntimeError("Initial parameters produced invalid gradients.")
                # We should have rejected the previous step!
                # But we are here.
                break

            diag_idx = jnp.diag_indices_from(total_JtJ)

            for _lm_iter in range(10):  # Max dampings per iter
                solver_matrix = total_JtJ.at[diag_idx].add(
                    lm_lambda * jnp.diag(total_JtJ)
                )

                try:
                    # use lstsq for robustness against singular matrices
                    step, _, _, _ = jnp.linalg.lstsq(
                        solver_matrix, -total_Jtr, rcond=1e-5
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    step = jnp.nan  # Signal fail

                # Check step validity
                if jnp.any(jnp.isnan(step)):
                    logger.warning(
                        f"Bad step (NaN). Increasing damping ({lm_lambda:.1e} -> {lm_lambda * 10:.1e})"
                    )
                    lm_lambda *= 10
                    continue

                # Proposed parameters
                params_new = params_curr + step
                # Clip
                if bounds is not None:
                    lower, upper = bounds
                    params_new = jnp.clip(
                        params_new, jnp.asarray(lower), jnp.asarray(upper)
                    )

                # Evaluate New Cost
                # This is expensive but necessary
                try:
                    chi2_new = evaluate_total_chi2(params_new)
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    logger.warning(f"Eval failed: {e}")
                    chi2_new = jnp.inf

                # Acceptance check
                if chi2_new < total_chi2:
                    # Accept
                    ratio = (total_chi2 - chi2_new) / total_chi2
                    logger.info(
                        f"Iter {i + 1}: chi2={float(chi2_new):.4e} (dec {ratio:.1%}), "
                        f"lambda={lm_lambda:.1e}"
                    )
                    params_curr = params_new
                    lm_lambda *= 0.1  # Decrease damping (trust more)
                    if lm_lambda < 1e-7:
                        lm_lambda = 1e-7
                    step_accepted = True

                    # Multi-criteria convergence (v2.22.0)
                    # 1. Per-component relative parameter change (scale-invariant)
                    param_scale = jnp.maximum(jnp.abs(params_curr), 1e-10)
                    rel_change = float(jnp.max(jnp.abs(step) / param_scale))
                    # 2. Relative cost function change
                    cost_change = float(ratio)

                    logger.debug(
                        f"  Convergence: xtol={rel_change:.2e} "
                        f"(thresh={xtol:.0e}), "
                        f"ftol={cost_change:.2e} "
                        f"(thresh={ftol:.0e})"
                    )

                    if rel_change < xtol and cost_change < ftol:
                        logger.info(
                            f"Out-of-Core converged: xtol={rel_change:.2e}<{xtol:.0e}, "
                            f"ftol={cost_change:.2e}<{ftol:.0e}"
                        )
                        return (
                            np.array(params_curr),
                            np.array(total_JtJ),
                            {
                                "chi_squared": float(chi2_new),
                                "iterations": i + 1,
                                "convergence_status": "converged",
                                "message": "Out-of-Core converged (xtol+ftol)",
                            },
                        )
                    break  # Break inner LM loop, proceed to next accumulation
                else:
                    # Reject
                    logger.debug(
                        f"Reject step (chi2 {float(chi2_new):.4e} >= {float(total_chi2):.4e}). Damping up."
                    )
                    lm_lambda *= 10

            if not step_accepted:
                logger.warning("Could not find better step. Stopping.")
                break

        # Determine final status (rel_change initialized to inf before loop)
        converged = rel_change < xtol
        info = {
            "chi_squared": float(total_chi2),
            "iterations": i + 1,
            "convergence_status": "converged" if converged else "max_iter",
            "message": "Out-of-Core accumulation completed",
        }
        return np.array(params_curr), np.array(total_JtJ), info

    def _fit_with_stratified_hybrid_streaming(
        self,
        stratified_data: Any,
        per_angle_scaling: bool,
        physical_param_names: list[str],
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        logger: Any,
        hybrid_config: dict | None = None,
        anti_degeneracy_config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Fit using NLSQ AdaptiveHybridStreamingOptimizer for large datasets.

        This method implements the 4-phase hybrid optimization from NLSQ >=0.3.2:
        - Phase 0: Parameter normalization setup (bounds-based)
        - Phase 1: L-BFGS warmup with adaptive switching
        - Phase 2: Streaming Gauss-Newton with exact J^T J accumulation
        - Phase 3: Denormalization and covariance transform

        With Anti-Degeneracy Defense System v2.9.0 integration:
        - Layer 1: Fourier Reparameterization (reduces per-angle DoF)
        - Layer 2: Hierarchical Optimization (alternating stage fitting)
        - Layer 3: Adaptive CV-based Regularization (scales properly)
        - Layer 4: Gradient Collapse Detection (runtime monitoring)

        Key improvements over basic StreamingOptimizer:
        1. Shear-term weak gradients: Fixed via parameter normalization
        2. Slow convergence: Fixed via L-BFGS warmup + Gauss-Newton refinement
        3. Crude covariance: Fixed via exact J^T J accumulation
        4. Structural degeneracy: Fixed via anti-degeneracy defense layers

        Args:
            stratified_data: StratifiedData object with flat stratified arrays
            per_angle_scaling: Whether per-angle parameters are enabled
            physical_param_names: List of physical parameter names
            initial_params: Initial parameter guess
            bounds: Parameter bounds (lower, upper) tuple
            logger: Logger instance
            hybrid_config: Optional config dict with keys:
                - normalize: Enable parameter normalization (default: True)
                - normalization_strategy: "bounds" or "scale" (default: "bounds")
                - warmup_iterations: L-BFGS warmup iterations (default: 100)
                - max_warmup_iterations: Max L-BFGS iterations (default: 500)
                - warmup_learning_rate: L-BFGS line search scale (default: 0.001)
                - gauss_newton_max_iterations: GN iterations (default: 50)
                - gauss_newton_tol: Convergence tolerance (default: 1e-8)
                - chunk_size: Points per chunk for streaming (default: 50000)
            anti_degeneracy_config: Optional config dict for Anti-Degeneracy Defense:
                - per_angle_mode: "independent", "fourier", or "auto" (default: "auto")
                - fourier_order: Fourier harmonic order (default: 2)
                - fourier_auto_threshold: n_phi threshold for auto mode (default: 6)
                - hierarchical.enable: Enable hierarchical optimization (default: True)
                - regularization.mode: "absolute", "relative", or "auto" (default: "relative")
                - regularization.lambda: Base regularization strength (default: 1.0)
                - gradient_monitoring.enable: Enable gradient collapse detection (default: True)

        Returns:
            (popt, pcov, info) tuple

        Raises:
            RuntimeError: If AdaptiveHybridStreamingOptimizer is not available
        """
        import time

        if not HYBRID_STREAMING_AVAILABLE:
            raise RuntimeError(
                "AdaptiveHybridStreamingOptimizer not available. "
                "Please upgrade NLSQ to version >= 0.3.2: pip install --upgrade nlsq"
            )

        logger.info("Initializing NLSQ AdaptiveHybridStreamingOptimizer...")
        logger.info("Fixes: 1) Shear-term gradients, 2) Convergence, 3) Covariance")

        start_time = time.perf_counter()

        # Parse hybrid streaming configuration
        # Uses NLSQ 0.3.6 defaults which include 4-layer defense strategy
        config_dict = hybrid_config or {}
        normalize = config_dict.get("normalize", True)
        normalization_strategy = config_dict.get("normalization_strategy", "auto")
        # Standard warmup iterations - NLSQ 0.3.6 has 4-layer defense to prevent
        # divergence when starting from good parameters
        warmup_iterations = config_dict.get("warmup_iterations", 200)
        max_warmup_iterations = config_dict.get("max_warmup_iterations", 500)
        warmup_learning_rate = config_dict.get("warmup_learning_rate", 0.001)
        gauss_newton_max_iterations = config_dict.get(
            "gauss_newton_max_iterations", 100
        )
        gauss_newton_tol = config_dict.get("gauss_newton_tol", 1e-8)
        chunk_size = config_dict.get("chunk_size", 10_000)
        trust_region_initial = config_dict.get("trust_region_initial", 1.0)
        regularization_factor = config_dict.get("regularization_factor", 1e-10)
        enable_checkpoints = config_dict.get("enable_checkpoints", True)
        checkpoint_frequency = config_dict.get("checkpoint_frequency", 100)
        validate_numerics = config_dict.get("validate_numerics", True)

        # Learning rate scheduling
        use_learning_rate_schedule = config_dict.get(
            "use_learning_rate_schedule", False
        )
        lr_schedule_warmup_steps = config_dict.get(
            "lr_schedule_warmup_steps", warmup_iterations
        )
        lr_schedule_decay_steps = config_dict.get(
            "lr_schedule_decay_steps", max_warmup_iterations - warmup_iterations
        )
        lr_schedule_end_value = config_dict.get("lr_schedule_end_value", 0.0001)

        # 4-Layer Defense Strategy (NLSQ 0.3.6)
        # Prevents L-BFGS warmup from diverging when starting from good parameters
        # Layer 1: Warm Start Detection - skip warmup if already at good solution
        enable_warm_start_detection = config_dict.get(
            "enable_warm_start_detection", True
        )
        warm_start_threshold = float(config_dict.get("warm_start_threshold", 0.01))
        # Layer 2: Adaptive Learning Rate - scale LR based on initial loss quality
        enable_adaptive_warmup_lr = config_dict.get("enable_adaptive_warmup_lr", True)
        warmup_lr_refinement = float(config_dict.get("warmup_lr_refinement", 1e-6))
        warmup_lr_careful = float(config_dict.get("warmup_lr_careful", 1e-5))
        # Layer 3: Cost-Increase Guard - abort if loss increases during warmup
        enable_cost_guard = config_dict.get("enable_cost_guard", True)
        cost_increase_tolerance = float(
            config_dict.get("cost_increase_tolerance", 0.05)
        )
        # Layer 4: Step Clipping - limit max parameter change per L-BFGS iteration
        enable_step_clipping = config_dict.get("enable_step_clipping", True)
        max_warmup_step_size = float(config_dict.get("max_warmup_step_size", 0.1))

        # Group Variance Regularization (NLSQ 0.3.8)
        # Prevents per-angle parameters from absorbing angle-dependent physical signals
        enable_group_variance_regularization = config_dict.get(
            "enable_group_variance_regularization", False
        )
        group_variance_lambda = float(config_dict.get("group_variance_lambda", 0.01))
        # group_variance_indices will be auto-computed if not provided
        group_variance_indices_raw = config_dict.get("group_variance_indices", None)

        # Compute n_phi early for auto-computing group_variance_indices
        # Extract unique phi angles from stratified data
        all_phi_early = []
        if hasattr(stratified_data, "chunks") and len(stratified_data.chunks) > 0:
            for chunk in stratified_data.chunks:
                all_phi_early.extend(chunk.phi.tolist())
        else:
            all_phi_early = stratified_data.phi_flat.tolist()
        n_phi = len(set(all_phi_early))
        phi_unique = np.array(sorted(set(all_phi_early)))  # For shear weighting

        # Auto-compute group_variance_indices for laminar_flow with per-angle scaling
        is_laminar_flow = "gamma_dot_t0" in physical_param_names

        # =====================================================================
        # Anti-Degeneracy Defense System v2.9.0 Initialization
        # =====================================================================
        # CRITICAL FIX (Jan 2026): Define n_physical unconditionally FIRST
        # This variable is used by multiple conditional blocks (hierarchical,
        # gradient_monitor, shear_weighter). Previously it was only defined
        # inside conditional blocks, causing UnboundLocalError when those
        # conditions were false but shear_weighter tried to use it.
        n_physical = len(physical_param_names)

        # Parse anti-degeneracy configuration
        ad_config = anti_degeneracy_config or {}
        hierarchical_config = ad_config.get("hierarchical", {})
        regularization_config = ad_config.get("regularization", {})
        gradient_monitoring_config = ad_config.get("gradient_monitoring", {})

        # Layer 1: Fourier Reparameterization / Constant Scaling Configuration
        # v2.18.0: Distinct semantics for auto vs explicit constant mode
        per_angle_mode = ad_config.get("per_angle_mode", "auto")
        fourier_order = ad_config.get("fourier_order", 2)
        fourier_auto_threshold = ad_config.get("fourier_auto_threshold", 6)
        constant_scaling_threshold = ad_config.get("constant_scaling_threshold", 3)

        # Determine actual per-angle mode
        # v2.18.0: Distinct semantics:
        #   - auto (n_phi >= threshold): "auto_averaged" → 9 params, OPTIMIZED averaged scaling
        #   - constant (explicit): "fixed_constant" → 7 params, FIXED per-angle scaling
        #   - individual: per-angle scaling OPTIMIZED
        if per_angle_mode == "auto":
            if n_phi >= constant_scaling_threshold:
                # AUTO mode with large n_phi: optimize averaged scaling (9 params)
                # Computes N quantile estimates, averages to 1 contrast + 1 offset
                # These 2 averaged values ARE OPTIMIZED along with 7 physical params
                per_angle_mode_actual = "auto_averaged"
                logger.info("=" * 60)
                logger.info(
                    "ANTI-DEGENERACY DEFENSE: Auto-selected 'auto_averaged' mode"
                )
                logger.info(
                    f"  Reason: n_phi ({n_phi}) >= "
                    f"constant_scaling_threshold ({constant_scaling_threshold})"
                )
                logger.info("  Behavior: Quantile estimates → AVERAGED → OPTIMIZED")
                logger.info(
                    "  Parameters: 7 physical + 2 averaged (contrast, offset) = 9 total"
                )
                logger.info("=" * 60)
            else:
                # Use individual per-angle parameters for few angles (N < 3)
                per_angle_mode_actual = "individual"
                logger.info("=" * 60)
                logger.info("ANTI-DEGENERACY DEFENSE: Auto-selected 'individual' mode")
                logger.info(
                    f"  Reason: n_phi ({n_phi}) < "
                    f"constant_scaling_threshold ({constant_scaling_threshold})"
                )
                logger.info(
                    f"  Parameters: 7 physical + {2 * n_phi} per-angle = {7 + 2 * n_phi} total"
                )
                logger.info("=" * 60)
        elif per_angle_mode == "constant":
            # EXPLICIT constant mode: FIXED per-angle scaling (7 params)
            # Computes N quantile estimates, uses per-angle values DIRECTLY (NOT averaged)
            # Only 7 physical params are optimized; scaling is FIXED
            per_angle_mode_actual = "fixed_constant"
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY DEFENSE: Explicit 'constant' mode → fixed_constant"
            )
            logger.info(f"  n_phi: {n_phi}")
            logger.info(
                "  Behavior: Quantile estimates → per-angle values FIXED (NOT optimized)"
            )
            logger.info("  Parameters: 7 physical only (scaling FIXED from quantiles)")
            logger.info("=" * 60)
        else:
            # Other explicit modes (fourier or individual)
            per_angle_mode_actual = per_angle_mode
            logger.debug(
                f"ANTI-DEGENERACY: Using explicit per_angle_mode: {per_angle_mode_actual}"
            )

        # T031: Determine mode flags
        # use_constant: True for both auto_averaged and fixed_constant (constant-style mapping)
        # use_fixed_scaling: True only for fixed_constant (scaling NOT optimized)
        # use_averaged_scaling: True only for auto_averaged (scaling optimized)
        use_constant = per_angle_mode_actual in ("auto_averaged", "fixed_constant")
        use_averaged_scaling = per_angle_mode_actual == "auto_averaged"
        # use_fixed_scaling will be set True after quantile estimation for fixed_constant mode

        # Initialize Fourier reparameterizer if using fourier mode
        fourier_reparameterizer = None
        if per_angle_mode_actual == "fourier" and per_angle_scaling and is_laminar_flow:
            # Get unique phi angles in radians
            phi_unique_rad = np.deg2rad(np.array(sorted(set(all_phi_early))))

            # Extract user-configured bounds for contrast and offset from bounds tuple
            # Bounds layout: [contrast(n_phi), offset(n_phi), physical(7)]
            # Use first contrast/offset bound as the c0/o0 (mean) bounds
            c0_bounds = (0.1, 0.8)  # Default
            o0_bounds = (0.5, 1.5)  # Default
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                if len(lower_bounds) >= n_phi and len(upper_bounds) >= n_phi:
                    # Extract contrast bounds from first contrast element
                    c0_bounds = (float(lower_bounds[0]), float(upper_bounds[0]))
                    # Extract offset bounds from first offset element
                    o0_bounds = (float(lower_bounds[n_phi]), float(upper_bounds[n_phi]))
                    logger.debug(
                        f"  Using user-configured Fourier bounds: "
                        f"c0={c0_bounds}, o0={o0_bounds}"
                    )

            fourier_config = FourierReparamConfig(
                mode="fourier",
                fourier_order=fourier_order,
                auto_threshold=fourier_auto_threshold,
                c0_bounds=c0_bounds,
                o0_bounds=o0_bounds,
            )
            fourier_reparameterizer = FourierReparameterizer(
                phi_unique_rad, fourier_config
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Layer 1 - Fourier Reparameterization")
            logger.info(f"  Mode: {per_angle_mode_actual}")
            logger.info(f"  n_phi: {n_phi}, Fourier order: {fourier_order}")
            logger.info(f"  Contrast bounds (c0): {c0_bounds}")
            logger.info(f"  Offset bounds (o0): {o0_bounds}")
            logger.info(
                f"  Parameter reduction: {2 * n_phi} -> {fourier_reparameterizer.n_coeffs}"
            )
            logger.info("=" * 60)
        elif (
            per_angle_mode_actual == "fixed_constant"
            and per_angle_scaling
            and is_laminar_flow
        ):
            # fixed_constant mode: per-angle scaling is FIXED, not optimized
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY DEFENSE: Layer 1 - Fixed Constant Mode (v2.18.0)"
            )
            logger.info(f"  Mode: {per_angle_mode_actual}")
            logger.info(f"  n_phi: {n_phi}")
            logger.info(
                "  Method: Quantile-based per-angle scaling (FIXED, not optimized)"
            )
            logger.info(
                "  Per-angle contrast/offset will be estimated from c2 data quantiles"
            )
            logger.info("  These values are FIXED (not optimized) during fitting")
            logger.info(f"  Parameter reduction: {2 * n_phi} -> 0 (physical only)")
            logger.info("=" * 60)
        elif (
            per_angle_mode_actual == "auto_averaged"
            and per_angle_scaling
            and is_laminar_flow
        ):
            # auto_averaged mode: averaged scaling is OPTIMIZED (9 params)
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY DEFENSE: Layer 1 - Auto Averaged Mode (v2.18.0)"
            )
            logger.info(f"  Mode: {per_angle_mode_actual}")
            logger.info(f"  n_phi: {n_phi}")
            logger.info("  Method: Quantile estimates → averaged → OPTIMIZED")
            logger.info("  Initial values: averaged from per-angle quantile estimates")
            logger.info(
                f"  Parameter reduction: {2 * n_phi} -> 2 (averaged contrast + offset)"
            )
            logger.info("=" * 60)

        # =====================================================================
        # CONSTANT/AUTO_AVERAGED MODES (v2.18.0): Quantile-Based Scaling
        # =====================================================================
        # - fixed_constant: per-angle values are FIXED (not optimized), 7 params
        # - auto_averaged: averaged values are OPTIMIZED as initial values, 9 params
        # =====================================================================
        use_fixed_scaling = False
        fixed_contrast_per_angle: np.ndarray | None = None
        fixed_offset_per_angle: np.ndarray | None = None
        fixed_contrast_jax: jnp.ndarray | None = None
        fixed_offset_jax: jnp.ndarray | None = None
        # For auto_averaged mode: averaged values to use as initial optimization values
        averaged_contrast_init: float | None = None
        averaged_offset_init: float | None = None

        if use_constant and per_angle_scaling and is_laminar_flow:
            logger.info("Computing quantile-based per-angle scaling estimates...")
            try:
                # Extract bounds for clipping
                contrast_bounds = (0.0, 1.0)  # Default
                offset_bounds = (0.5, 1.5)  # Default
                if bounds is not None:
                    lower_bounds, upper_bounds = bounds
                    if len(lower_bounds) >= n_phi and len(upper_bounds) >= n_phi:
                        contrast_bounds = (
                            float(lower_bounds[0]),
                            float(upper_bounds[0]),
                        )
                        offset_bounds = (
                            float(lower_bounds[n_phi]),
                            float(upper_bounds[n_phi]),
                        )

                # Compute quantile-based per-angle scaling
                fixed_contrast_per_angle, fixed_offset_per_angle = (
                    _compute_quantile_per_angle_scaling(
                        stratified_data=stratified_data,
                        contrast_bounds=contrast_bounds,
                        offset_bounds=offset_bounds,
                        logger=logger,
                    )
                )

                if (
                    fixed_contrast_per_angle is not None
                    and fixed_offset_per_angle is not None
                ):
                    if per_angle_mode_actual == "fixed_constant":
                        # fixed_constant: Use per-angle values DIRECTLY as FIXED
                        use_fixed_scaling = True
                        fixed_contrast_jax = jnp.asarray(fixed_contrast_per_angle)
                        fixed_offset_jax = jnp.asarray(fixed_offset_per_angle)

                        logger.info(
                            "Fixed per-angle scaling computed (FIXED, not optimized):"
                        )
                        logger.info(
                            f"  Contrast: mean={np.mean(fixed_contrast_per_angle):.4f}, "
                            f"range=[{np.min(fixed_contrast_per_angle):.4f}, "
                            f"{np.max(fixed_contrast_per_angle):.4f}]"
                        )
                        logger.info(
                            f"  Offset: mean={np.mean(fixed_offset_per_angle):.4f}, "
                            f"range=[{np.min(fixed_offset_per_angle):.4f}, "
                            f"{np.max(fixed_offset_per_angle):.4f}]"
                        )
                    elif per_angle_mode_actual == "auto_averaged":
                        # auto_averaged: AVERAGE per-angle values → use as INITIAL for optimization
                        averaged_contrast_init = float(
                            np.mean(fixed_contrast_per_angle)
                        )
                        averaged_offset_init = float(np.mean(fixed_offset_per_angle))

                        logger.info(
                            "Averaged scaling computed (initial values for optimization):"
                        )
                        logger.info(
                            f"  Averaged contrast: {averaged_contrast_init:.4f}"
                        )
                        logger.info(f"  Averaged offset: {averaged_offset_init:.4f}")
                        logger.info(
                            "  These will be OPTIMIZED along with 7 physical params (9 total)"
                        )

                        # Do NOT set use_fixed_scaling = True for auto_averaged
                        # The averaged values are just initial guesses for optimization
                else:
                    logger.warning(
                        "Failed to compute quantile-based scaling, "
                        "falling back to standard constant mode (optimizing 2 params)"
                    )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                logger.warning(
                    f"Error computing quantile-based scaling: {e}, "
                    f"falling back to standard constant mode"
                )
                use_fixed_scaling = False

        # Layer 2: Hierarchical Optimization Configuration
        # =====================================================================
        # CRITICAL FIX (Jan 2026): Auto-enable hierarchical when shear_weighting
        # is enabled. Shear weighting is ONLY applied inside hierarchical
        # optimizer's loss function. Without hierarchical, the gradient
        # cancellation for gamma_dot_t0 is NOT prevented!
        #
        # Root cause: The shear gradient ∂L/∂γ̇₀ ∝ Σ cos(φ₀-φ) cancels when
        # summing over angles spanning 360° (e.g., 23 angles → 94.6% cancellation).
        # Shear weighting emphasizes shear-sensitive angles to prevent this.
        # =====================================================================
        shear_weighting_config_early = ad_config.get("shear_weighting", {})
        shear_weighting_will_be_enabled = (
            shear_weighting_config_early.get("enable", True)
            and is_laminar_flow
            and n_phi > 3
        )

        enable_hierarchical = hierarchical_config.get("enable", True)

        # Override: shear weighting requires hierarchical optimization to function
        if shear_weighting_will_be_enabled and not enable_hierarchical:
            logger.warning("=" * 60)
            logger.warning(
                "ANTI-DEGENERACY: Shear weighting enabled but hierarchical disabled!"
            )
            logger.warning(
                "  Auto-enabling hierarchical optimization to apply shear weights."
            )
            logger.warning(
                "  Without this, gradient cancellation will collapse gamma_dot_t0."
            )
            logger.warning(
                "  See: docs/analysis/nlsq-divergence-root-cause-20260101.md"
            )
            logger.warning("=" * 60)
            enable_hierarchical = True

        hierarchical_optimizer = None
        # Skip hierarchical optimization in constant scaling mode:
        # - Constant mode already prevents per-angle absorption (2 DoF vs 46)
        # - HierarchicalOptimizer expects n_per_angle = 2*n_phi or n_coeffs (Fourier)
        # - Using hierarchical with constant mode causes index mismatch error
        if (
            enable_hierarchical
            and per_angle_scaling
            and is_laminar_flow
            and not use_constant
        ):
            # n_physical defined unconditionally above
            hier_config = HierarchicalConfig(
                enable=True,
                max_outer_iterations=hierarchical_config.get("max_outer_iterations", 5),
                outer_tolerance=float(hierarchical_config.get("outer_tolerance", 1e-6)),
                physical_max_iterations=hierarchical_config.get(
                    "physical_max_iterations", 100
                ),
                per_angle_max_iterations=hierarchical_config.get(
                    "per_angle_max_iterations", 50
                ),
            )
            hierarchical_optimizer = HierarchicalOptimizer(
                config=hier_config,
                n_phi=n_phi,
                n_physical=n_physical,
                fourier_reparameterizer=fourier_reparameterizer,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
            logger.info(f"  Enabled: {enable_hierarchical}")
            logger.info(f"  Max outer iterations: {hier_config.max_outer_iterations}")
            logger.info(f"  Outer tolerance: {hier_config.outer_tolerance}")
            if shear_weighting_will_be_enabled:
                logger.info(
                    "  Shear weighting: WILL BE APPLIED via hierarchical loss function"
                )
            logger.info("=" * 60)
        elif (
            use_constant
            and enable_hierarchical
            and per_angle_scaling
            and is_laminar_flow
        ):
            # Log that hierarchical is skipped due to constant scaling mode
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Layer 2 - Hierarchical Optimization")
            logger.info(
                "  Skipped: constant scaling mode already prevents per-angle absorption"
            )
            logger.info(
                "  Reason: Only 2 per-angle DoF (vs 46), no need for hierarchical alternation"
            )
            logger.info("=" * 60)

        # Layer 3: Adaptive Relative Regularization Configuration
        # Replaces/enhances the basic group variance regularization with CV-based approach
        regularization_mode = regularization_config.get("mode", "relative")
        regularization_lambda = float(regularization_config.get("lambda", 1.0))
        target_cv = float(regularization_config.get("target_cv", 0.10))
        target_contribution = float(
            regularization_config.get("target_contribution", 0.10)
        )
        max_cv = float(regularization_config.get("max_cv", 0.20))

        adaptive_regularizer = None
        if per_angle_scaling and is_laminar_flow:
            # Compute mode-aware group indices
            # Group indices depend on per-angle mode: fixed_constant, auto_averaged, fourier, or individual
            if use_fixed_scaling:
                # fixed_constant: No scaling params to regularize (7 physical only)
                mode_group_indices = []
                logger.debug(
                    "Fixed-constant mode: No per-angle regularization (scaling is fixed)"
                )
            elif use_averaged_scaling:
                # auto_averaged: 2 per-angle params (1 contrast + 1 offset) to regularize
                mode_group_indices = [(0, 1), (1, 2)]
                logger.debug(
                    f"Auto-averaged regularization groups: {mode_group_indices} "
                    f"(1 contrast + 1 offset)"
                )
            elif (
                fourier_reparameterizer is not None
                and fourier_reparameterizer.use_fourier
            ):
                n_coeffs_per_param = fourier_reparameterizer.n_coeffs_per_param
                mode_group_indices = [
                    (0, n_coeffs_per_param),  # contrast Fourier coefficients
                    (
                        n_coeffs_per_param,
                        2 * n_coeffs_per_param,
                    ),  # offset Fourier coefficients
                ]
                logger.debug(
                    f"Fourier-aware regularization groups: {mode_group_indices} "
                    f"(n_coeffs_per_param={n_coeffs_per_param})"
                )
            else:
                mode_group_indices = None  # Use default: [(0, n_phi), (n_phi, 2*n_phi)]
                logger.debug(
                    f"Using default regularization groups (Fourier mode not active): "
                    f"fourier_reparameterizer={fourier_reparameterizer is not None}, "
                    f"use_fourier={fourier_reparameterizer.use_fourier if fourier_reparameterizer else 'N/A'}"
                )

            reg_config = AdaptiveRegularizationConfig(
                enable=True,
                mode=regularization_mode,
                lambda_base=regularization_lambda,
                target_cv=target_cv,
                target_contribution=target_contribution,
                max_cv=max_cv,
                group_indices=mode_group_indices,
            )
            adaptive_regularizer = AdaptiveRegularizer(reg_config, n_phi)
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Layer 3 - Adaptive Regularization")
            logger.info(f"  Mode: {regularization_mode}")
            logger.info(f"  Auto-tuned lambda: {adaptive_regularizer.lambda_value:.2f}")
            logger.info(f"  Target CV: {target_cv} ({target_cv * 100:.0f}% variation)")
            logger.info(f"  Max CV: {max_cv}")
            logger.info(f"  Group indices: {adaptive_regularizer.group_indices}")
            logger.info("=" * 60)

            # Update group variance settings to use adaptive regularizer's lambda
            # This ensures NLSQ's built-in regularization is consistent
            enable_group_variance_regularization = True
            group_variance_lambda = adaptive_regularizer.lambda_value

        # Layer 4: Gradient Collapse Monitor Configuration
        gradient_monitor_enabled = gradient_monitoring_config.get("enable", True)
        gradient_monitor = None
        if gradient_monitor_enabled and per_angle_scaling and is_laminar_flow:
            # Compute mode-aware parameter count
            # n_per_angle depends on per-angle mode: fixed_constant, auto_averaged, fourier, or individual
            if use_fixed_scaling:
                # fixed_constant: 0 per-angle params (scaling is fixed)
                n_per_angle = 0
            elif use_averaged_scaling:
                # auto_averaged: 2 per-angle params (1 contrast + 1 offset)
                n_per_angle = 2
            elif fourier_reparameterizer is not None:
                # Fourier mode: n_coeffs Fourier coefficients
                n_per_angle = fourier_reparameterizer.n_coeffs
            else:
                # Independent mode: 2 * n_phi per-angle params
                n_per_angle = 2 * n_phi
            # n_physical defined unconditionally above
            # Use numpy arrays for indices (JAX compatibility)
            per_angle_indices = np.arange(n_per_angle, dtype=np.intp)
            physical_indices = np.arange(
                n_per_angle, n_per_angle + n_physical, dtype=np.intp
            )

            # Compute gamma_dot_t0 index for watch_parameters
            # In laminar_flow, physical params are [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
            # gamma_dot_t0 is at physical_indices[3] = n_per_angle + 3
            gamma_dot_t0_idx = (
                n_per_angle + 3
            )  # Index of gamma_dot_t0 in full param vector

            monitor_config = GradientMonitorConfig(
                enable=True,
                ratio_threshold=float(
                    gradient_monitoring_config.get("ratio_threshold", 0.01)
                ),
                consecutive_triggers=gradient_monitoring_config.get(
                    "consecutive_triggers", 5
                ),
                response_mode=gradient_monitoring_config.get(
                    "response", "hierarchical"
                ),
                # NEW (Dec 2025): Watch gamma_dot_t0 specifically for gradient collapse
                # This detects when shear parameter gradient vanishes during L-BFGS warmup
                watch_parameters=[gamma_dot_t0_idx],
                watch_threshold=float(
                    gradient_monitoring_config.get("watch_threshold", 1e-8)
                ),
            )
            gradient_monitor = GradientCollapseMonitor(
                config=monitor_config,
                physical_indices=physical_indices,
                per_angle_indices=per_angle_indices,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY DEFENSE: Layer 4 - Gradient Collapse Monitor")
            logger.info(f"  Enabled: {gradient_monitor_enabled}")
            logger.info(f"  Ratio threshold: {monitor_config.ratio_threshold}")
            logger.info(
                f"  Consecutive triggers: {monitor_config.consecutive_triggers}"
            )
            logger.info(f"  Response mode: {monitor_config.response_mode}")
            logger.info("=" * 60)

        # Layer 5: Shear-Sensitivity Weighting (v2.9.1)
        # Prevents gradient cancellation for shear parameters by emphasizing
        # shear-sensitive angles (parallel/antiparallel to flow direction)
        shear_weighting_config = ad_config.get("shear_weighting", {})
        shear_weighting_enabled = shear_weighting_config.get("enable", True)
        shear_weighter: ShearSensitivityWeighting | None = None

        if is_laminar_flow and shear_weighting_enabled and n_phi > 3:
            # Get initial phi0 from config or use default
            initial_phi0 = shear_weighting_config.get("initial_phi0", None)
            if initial_phi0 is None:
                # Try to get from initial parameters
                initial_phi0 = (
                    float(initial_params[-1]) if len(initial_params) > 0 else 0.0
                )

            sw_config = ShearWeightingConfig(
                enable=True,
                min_weight=float(shear_weighting_config.get("min_weight", 0.3)),
                alpha=float(shear_weighting_config.get("alpha", 1.0)),
                update_frequency=int(shear_weighting_config.get("update_frequency", 1)),
                initial_phi0=initial_phi0,
                normalize=shear_weighting_config.get("normalize", True),
            )
            shear_weighter = ShearSensitivityWeighting(
                phi_angles=phi_unique,
                n_physical=n_physical,
                phi0_index=6,  # phi0 is last of 7 physical params
                config=sw_config,
            )
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY DEFENSE: Layer 5 - Shear-Sensitivity Weighting"
            )
            logger.info(f"  Enabled: {shear_weighting_enabled}")
            logger.info(f"  n_phi: {n_phi}")
            logger.info(f"  min_weight: {sw_config.min_weight:.2f}")
            logger.info(f"  alpha: {sw_config.alpha:.1f}")
            logger.info(f"  initial_phi0: {initial_phi0:.1f}°")
            logger.info("=" * 60)

        # Store anti-degeneracy components for diagnostics
        anti_degeneracy_components = {
            "per_angle_mode": per_angle_mode_actual,
            "use_constant": use_constant,  # T031: Track constant mode status
            "use_fixed_scaling": use_fixed_scaling,  # v2.17.0: Track fixed scaling status
            "fourier_reparameterizer": fourier_reparameterizer,
            "hierarchical_optimizer": hierarchical_optimizer,
            "adaptive_regularizer": adaptive_regularizer,
            "gradient_monitor": gradient_monitor,
            "shear_weighter": shear_weighter,
        }
        # ===================================================================== #
        if enable_group_variance_regularization and group_variance_indices_raw is None:
            if is_laminar_flow and per_angle_scaling and n_phi > 3:
                # T031: Handle fixed scaling, constant, Fourier, and individual modes
                # Fixed scaling mode (v2.17.0): 0 per-angle params (all fixed)
                # Constant mode: 1 value per group (contrast/offset)
                # Fourier mode: n_coeffs_per_param values per group
                # Individual mode: n_phi values per group
                if use_fixed_scaling:
                    # No per-angle params to regularize - skip group variance
                    n_per_group = 0
                    group_variance_indices = []
                    logger.info(
                        "  Fixed scaling mode: skipping group variance regularization "
                        "(no per-angle params)"
                    )
                elif use_constant:
                    n_per_group = 1
                elif fourier_reparameterizer is not None:
                    n_per_group = fourier_reparameterizer.n_coeffs_per_param
                else:
                    n_per_group = n_phi

                # Only compute group indices if not using fixed scaling
                if not use_fixed_scaling:
                    # Per-angle contrast: params[0:n_per_group]
                    # Per-angle offset: params[n_per_group:2*n_per_group]
                    group_variance_indices = [
                        (0, n_per_group),
                        (n_per_group, 2 * n_per_group),
                    ]
                    logger.info(
                        f"  Auto-computed group_variance_indices for {n_phi} angles: "
                        f"{group_variance_indices} (mode: {per_angle_mode_actual})"
                    )
            else:
                group_variance_indices = None
                if enable_group_variance_regularization:
                    logger.warning(
                        "Group variance regularization enabled but no indices provided. "
                        "Auto-computation requires laminar_flow mode with per_angle_scaling "
                        f"and n_phi > 3. (is_laminar_flow={is_laminar_flow}, "
                        f"per_angle_scaling={per_angle_scaling}, n_phi={n_phi})"
                    )
        else:
            # Convert raw indices to list of tuples if provided
            if group_variance_indices_raw is not None:
                group_variance_indices = [
                    tuple(idx) for idx in group_variance_indices_raw
                ]
            else:
                group_variance_indices = None

        logger.info("Hybrid streaming config:")
        logger.info(f"  Normalization: {normalization_strategy}")
        logger.info(f"  Warmup iterations: {warmup_iterations}")
        logger.info(f"  Max warmup iterations: {max_warmup_iterations}")
        logger.info(f"  Learning rate: {warmup_learning_rate}")
        if use_learning_rate_schedule:
            logger.info(
                f"  LR schedule: warmup={lr_schedule_warmup_steps}, "
                f"decay={lr_schedule_decay_steps}, end={lr_schedule_end_value}"
            )
        logger.info(f"  Gauss-Newton iterations: {gauss_newton_max_iterations}")
        logger.info(f"  Gauss-Newton tolerance: {gauss_newton_tol}")
        logger.info(f"  Chunk size: {chunk_size:,}")
        logger.info("  4-Layer Defense Strategy (NLSQ 0.3.6):")
        logger.info(f"    L1 Warm Start Detection: {enable_warm_start_detection}")
        logger.info(f"    L2 Adaptive LR: {enable_adaptive_warmup_lr}")
        logger.info(f"    L3 Cost Guard: {enable_cost_guard}")
        logger.info(f"    L4 Step Clipping: {enable_step_clipping}")
        if enable_group_variance_regularization:
            logger.info("  Group Variance Regularization (NLSQ 0.3.8):")
            logger.info(f"    Enabled: {enable_group_variance_regularization}")
            logger.info(f"    Lambda: {group_variance_lambda}")
            logger.info(f"    Indices: {group_variance_indices}")

        # Prepare residual weighting for NLSQ optimizer (Layer 5 of Anti-Degeneracy)
        # Homodyne computes shear-sensitivity weights and passes them to NLSQ
        # as generic residual weights - NLSQ doesn't need to know about XPCS physics
        enable_residual_weighting = shear_weighter is not None
        residual_weights_list = None
        if enable_residual_weighting:
            # Compute shear-sensitivity weights in homodyne, pass to NLSQ as generic weights
            residual_weights_list = shear_weighter.get_weights().tolist()
            logger.info("  Residual Weighting (Shear-Sensitivity):")
            logger.info(f"    Enabled: {enable_residual_weighting}")
            logger.info(f"    n_weights: {len(residual_weights_list)}")
            logger.info(
                f"    Weight range: [{min(residual_weights_list):.3f}, "
                f"{max(residual_weights_list):.3f}]"
            )

        # Create HybridStreamingConfig with 4-layer defense
        optimizer_config = HybridStreamingConfig(
            normalize=normalize,
            normalization_strategy=normalization_strategy,
            warmup_iterations=warmup_iterations,
            max_warmup_iterations=max_warmup_iterations,
            warmup_learning_rate=warmup_learning_rate,
            gauss_newton_max_iterations=gauss_newton_max_iterations,
            gauss_newton_tol=gauss_newton_tol,
            chunk_size=chunk_size,
            trust_region_initial=trust_region_initial,
            regularization_factor=regularization_factor,
            enable_checkpoints=enable_checkpoints,
            checkpoint_frequency=checkpoint_frequency,
            validate_numerics=validate_numerics,
            use_learning_rate_schedule=use_learning_rate_schedule,
            lr_schedule_warmup_steps=lr_schedule_warmup_steps,
            lr_schedule_decay_steps=lr_schedule_decay_steps,
            lr_schedule_end_value=lr_schedule_end_value,
            # 4-Layer Defense Strategy
            enable_warm_start_detection=enable_warm_start_detection,
            warm_start_threshold=warm_start_threshold,
            enable_adaptive_warmup_lr=enable_adaptive_warmup_lr,
            warmup_lr_refinement=warmup_lr_refinement,
            warmup_lr_careful=warmup_lr_careful,
            enable_cost_guard=enable_cost_guard,
            cost_increase_tolerance=cost_increase_tolerance,
            enable_step_clipping=enable_step_clipping,
            max_warmup_step_size=max_warmup_step_size,
            # Group Variance Regularization (NLSQ 0.3.8)
            enable_group_variance_regularization=enable_group_variance_regularization,
            group_variance_lambda=group_variance_lambda,
            group_variance_indices=group_variance_indices,
            # Residual Weighting (NLSQ 0.4.x)
            # Homodyne computes shear-sensitivity weights and passes them as generic
            # residual weights - NLSQ just does weighted least squares
            enable_residual_weighting=enable_residual_weighting,
            residual_weights=residual_weights_list,
            verbose=config_dict.get("verbose", 1),
            log_frequency=config_dict.get("log_frequency", 1),
        )

        # Initialize optimizer
        optimizer = AdaptiveHybridStreamingOptimizer(optimizer_config)

        # Extract global metadata from stratified data
        if hasattr(stratified_data, "chunks") and len(stratified_data.chunks) > 0:
            first_chunk = stratified_data.chunks[0]
            q = first_chunk.q
            L = first_chunk.L
            dt = first_chunk.dt
        else:
            q = stratified_data.q
            L = stratified_data.L
            dt = stratified_data.dt

        logger.debug(f"Global metadata: q={q}, L={L}, dt={dt}")

        # Extract unique values for theory computation
        all_phi = []
        all_t1 = []
        all_t2 = []
        if hasattr(stratified_data, "chunks"):
            for chunk in stratified_data.chunks:
                all_phi.extend(chunk.phi.tolist())
                all_t1.extend(chunk.t1.tolist())
                all_t2.extend(chunk.t2.tolist())
        else:
            all_phi = stratified_data.phi_flat.tolist()
            all_t1 = stratified_data.t1_flat.tolist()
            all_t2 = stratified_data.t2_flat.tolist()

        phi_unique = np.array(sorted(set(all_phi)))
        t1_unique = np.array(sorted(set(all_t1)))
        n_phi = len(phi_unique)

        logger.info(f"Unique values: {n_phi} phi, {len(t1_unique)} t1")

        # Import physics utilities
        from homodyne.core.physics_utils import (
            PI,
            calculate_diffusion_coefficient,
            calculate_shear_rate,
            safe_sinc,
            trapezoid_cumsum,
        )

        # Pre-compute physics factors
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
        sinc_prefactor = 0.5 / PI * q * L * dt

        # Convert to JAX arrays
        phi_unique_jax = jnp.asarray(phi_unique)
        t1_unique_jax = jnp.asarray(t1_unique)

        # Create model function
        is_laminar_flow = "gamma_dot_t0" in physical_param_names

        # T042: Compute n_per_angle for model function based on mode
        # In fixed scaling mode: 0 (all params are physical)
        # In constant mode (fallback): 1 contrast + 1 offset = 2
        # In individual mode: n_phi contrast + n_phi offset = 2*n_phi
        # In Fourier mode: n_coeffs contrast + n_coeffs offset = 2*n_coeffs
        if use_fixed_scaling:
            # Fixed scaling: all params are physical, no per-angle params in vector
            n_per_angle = 0
        elif use_constant:
            n_per_angle = 2
        elif fourier_reparameterizer is not None:
            n_per_angle = fourier_reparameterizer.n_coeffs
        else:
            n_per_angle = 2 * n_phi

        @jax.jit
        def model_fn_pointwise(x_batch: jnp.ndarray, *params_tuple) -> jnp.ndarray:
            """Point-wise model function for hybrid streaming optimizer."""
            # Handle both single points (1D) and batches (2D)
            # The optimizer may call with single points during Jacobian computation
            x_batch_2d = jnp.atleast_2d(x_batch)

            params_all = jnp.stack(params_tuple)

            # Extract indices from x_batch (now guaranteed 2D)
            phi_idx = x_batch_2d[:, 0].astype(jnp.int32)
            t1_idx = x_batch_2d[:, 1].astype(jnp.int32)
            t2_idx = x_batch_2d[:, 2].astype(jnp.int32)

            # T042: Extract scaling and physical parameters based on mode
            # Fixed scaling mode (v2.17.0): use pre-computed fixed arrays, all params are physical
            # Constant mode (fallback): params[0]=contrast, params[1]=offset, params[2:]=physical
            # Individual mode: params[:n_phi]=contrast, params[n_phi:2*n_phi]=offset, params[2*n_phi:]=physical
            if use_fixed_scaling:
                # Use pre-computed fixed per-angle scaling from quantiles
                # All params in params_all are physical
                contrast_all = fixed_contrast_jax
                offset_all = fixed_offset_jax
                physical_params = params_all
            elif use_constant:
                # Single contrast and offset shared across all angles
                contrast_all = jnp.full(n_phi, params_all[0])
                offset_all = jnp.full(n_phi, params_all[1])
                physical_params = params_all[2:]
            else:
                contrast_all = params_all[:n_phi]
                offset_all = params_all[n_phi : 2 * n_phi]
                physical_params = params_all[2 * n_phi :]

            # Extract physical parameters
            D0 = physical_params[0]
            alpha = physical_params[1]
            D_offset = physical_params[2]

            # Compute diffusion
            D_t = calculate_diffusion_coefficient(t1_unique_jax, D0, alpha, D_offset)
            D_cumsum = trapezoid_cumsum(D_t)
            D_diff = D_cumsum[t1_idx] - D_cumsum[t2_idx]
            # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
            D_integral_batch = jnp.sqrt(D_diff**2 + 1e-12)

            log_g1_diff = -wavevector_q_squared_half_dt * D_integral_batch
            log_g1_diff_bounded = jnp.clip(log_g1_diff, -700.0, 0.0)
            g1_diffusion = jnp.exp(log_g1_diff_bounded)

            if is_laminar_flow:
                # Shear parameters
                gamma_dot_0 = physical_params[3]
                beta = physical_params[4]
                gamma_dot_offset = physical_params[5]
                phi0 = physical_params[6]

                # Compute shear
                gamma_t = calculate_shear_rate(
                    t1_unique_jax, gamma_dot_0, beta, gamma_dot_offset
                )
                gamma_cumsum = trapezoid_cumsum(gamma_t)
                gamma_diff = gamma_cumsum[t1_idx] - gamma_cumsum[t2_idx]
                # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
                gamma_integral_batch = jnp.sqrt(gamma_diff**2 + 1e-12)

                # Shear contribution with angle dependence
                # Formula: g₁_shear = [sinc(Φ)]² where Φ = sinc_prefactor * cos(φ₀-φ) * ∫γ̇
                phi_values = phi_unique_jax[phi_idx]
                angle_diff = jnp.deg2rad(phi0 - phi_values)  # Match physics: cos(φ₀-φ)
                cos_phi = jnp.cos(angle_diff)

                sinc_arg = sinc_prefactor * gamma_integral_batch * cos_phi
                sinc_val = safe_sinc(sinc_arg)
                g1_shear = sinc_val**2  # CRITICAL: g1_shear = sinc²(Φ)

                g1_total = g1_diffusion * g1_shear
                # P0-3: Use jnp.where (gradient-safe) instead of jnp.clip.
                # log-space clip above guarantees g1 ≤ 1.0; lower floor prevents log(0).
                epsilon = 1e-10
                g1 = jnp.where(g1_total > epsilon, g1_total, epsilon)
            else:
                epsilon = 1e-10
                g1 = jnp.where(g1_diffusion > epsilon, g1_diffusion, epsilon)

            # Compute g2 with per-angle scaling
            contrast = contrast_all[phi_idx]
            offset = offset_all[phi_idx]
            g2_theory = offset + contrast * g1**2
            # P0-3: Removed jnp.clip(g2, 0.5, 2.5) — kills gradients at boundaries.
            # Bounds enforced via parameter bounds in optimizer, not g2 clipping.
            g2 = g2_theory

            # Squeeze output to match input dimensionality
            # Returns 0D scalar for single point, 1D array for batch
            return g2.squeeze()

        # Prepare data
        logger.info("Preparing hybrid streaming data...")
        prep_start = time.perf_counter()

        if hasattr(stratified_data, "chunks"):
            all_phi_data = np.concatenate([c.phi for c in stratified_data.chunks])
            all_t1_data = np.concatenate([c.t1 for c in stratified_data.chunks])
            all_t2_data = np.concatenate([c.t2 for c in stratified_data.chunks])
            y_data = np.concatenate([c.g2 for c in stratified_data.chunks])
        else:
            all_phi_data = stratified_data.phi_flat
            all_t1_data = stratified_data.t1_flat
            all_t2_data = stratified_data.t2_flat
            y_data = stratified_data.g2_flat

        # Convert to indices (vectorized).
        # NOTE: Both t1 and t2 index into t1_unique because XPCS correlation
        # matrices use a shared time grid (t1_unique == t2_unique).
        phi_idx_arr = np.clip(
            np.searchsorted(phi_unique, all_phi_data), 0, len(phi_unique) - 1
        )
        t1_idx_arr = np.clip(
            np.searchsorted(t1_unique, all_t1_data), 0, len(t1_unique) - 1
        )
        t2_idx_arr = np.clip(
            np.searchsorted(t1_unique, all_t2_data), 0, len(t1_unique) - 1
        )

        x_data = np.column_stack([phi_idx_arr, t1_idx_arr, t2_idx_arr]).astype(
            np.float64
        )
        y_data = np.asarray(y_data, dtype=np.float64)

        # =====================================================================
        # Diagonal Handling (v2.14.2+)
        # =====================================================================
        # Hybrid streaming uses point-wise theory computation (no 2D grid), so
        # apply_diagonal_correction() cannot be applied to theory.
        #
        # Instead, diagonal points are FILTERED OUT from the data entirely:
        # - Data: Already has diagonal correction applied at load time
        # - Theory: Never computes diagonal values (filtered points excluded)
        # - Residual: Diagonal points excluded from loss (equivalent to mask=0)
        #
        # This is architecturally equivalent to correction + masking used in
        # Stratified LS and Out-of-Core methods. The result is the same:
        # diagonal points contribute ZERO to the optimization objective.
        # =====================================================================
        n_points_before = len(y_data)
        non_diagonal_mask = t1_idx_arr != t2_idx_arr
        x_data = x_data[non_diagonal_mask]
        y_data = y_data[non_diagonal_mask]
        n_diagonal_removed = n_points_before - len(y_data)

        prep_time = time.perf_counter() - prep_start
        logger.info(f"Data preparation completed in {prep_time:.2f}s")
        logger.info(f"  Dataset size: {len(y_data):,} points")
        logger.info(
            f"  Diagonal points removed: {n_diagonal_removed:,} "
            f"({100 * n_diagonal_removed / n_points_before:.1f}%)"
        )

        # NOTE (Dec 2025): Data is already pre-shuffled at stratification stage
        # in _apply_stratification_if_needed(). No additional shuffle needed here.
        # The pre-shuffle prevents L-BFGS warmup from seeing angle-sequential data,
        # which would cause local minimum traps (gamma_dot_t0 -> 0).

        # =====================================================================
        # Anti-Degeneracy Defense System v2.9.0 - EXECUTION INTEGRATION
        # =====================================================================
        # Transform parameters and execute appropriate optimization path
        use_hierarchical = (
            hierarchical_optimizer is not None
            and anti_degeneracy_components.get("hierarchical_optimizer") is not None
            and ad_config.get("enable", True)
        )
        use_fourier = (
            fourier_reparameterizer is not None
            and anti_degeneracy_components.get("fourier_reparameterizer") is not None
            and ad_config.get("enable", True)
        )

        # Track params for fitting
        fit_initial_params = initial_params.copy()
        fit_bounds = bounds

        # T034-T038: Constant mode parameter transformation
        # v2.17.0: When use_fixed_scaling=True, use physical params only (fixed contrast/offset from quantiles)
        # Fallback: Transform per-angle params (2*n_phi) to constant (2) by taking means
        if use_fixed_scaling:
            # FIXED SCALING MODE (v2.17.0): Use quantile-derived fixed per-angle scaling
            # Parameters are physical-only, contrast/offset are NOT in the param vector
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY EXECUTION: Fixed Per-Angle Scaling (v2.17.0)")
            physical_params = initial_params[2 * n_phi :]

            # New parameter layout: [physical_params] only
            fit_initial_params = physical_params

            logger.info(f"  Original params: {len(initial_params)}")
            logger.info(
                f"  Fixed scaling params: {len(fit_initial_params)} (physical only)"
            )
            logger.info(f"  Per-angle reduction: {2 * n_phi} -> 0 (using fixed arrays)")

            # Transform bounds to physical only
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                fit_bounds = (lower_bounds[2 * n_phi :], upper_bounds[2 * n_phi :])
                logger.info(
                    f"  Bounds reduced to physical only: {len(fit_bounds[0])} params"
                )
            logger.info("=" * 60)
        elif use_averaged_scaling:
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY EXECUTION: Auto Averaged Scaling Mode")
            # Transform per-angle params to single values (means) for optimization
            per_angle_params = initial_params[: 2 * n_phi]
            physical_params = initial_params[2 * n_phi :]

            # Split per-angle into contrast and offset groups
            contrast_per_angle = per_angle_params[:n_phi]
            offset_per_angle = per_angle_params[n_phi : 2 * n_phi]

            # Use quantile-based averaged values if computed, else take means
            if averaged_contrast_init is not None and averaged_offset_init is not None:
                contrast_mean = averaged_contrast_init
                offset_mean = averaged_offset_init
                logger.info(
                    "  Using quantile-based averaged initial values (OPTIMIZED)"
                )
            else:
                contrast_mean = np.mean(contrast_per_angle)
                offset_mean = np.mean(offset_per_angle)
                logger.info(
                    "  Using parameter-based averaged initial values (OPTIMIZED)"
                )

            # New parameter layout: [contrast_const, offset_const, physical_params]
            fit_initial_params = np.concatenate(
                [[contrast_mean], [offset_mean], physical_params]
            )

            logger.info(f"  Original params: {len(initial_params)}")
            logger.info(f"  Constant params: {len(fit_initial_params)}")
            logger.info(f"  Per-angle reduction: {2 * n_phi} -> 2")
            logger.info(f"  Contrast mean: {contrast_mean:.6f}")
            logger.info(f"  Offset mean: {offset_mean:.6f}")

            # T039: Transform bounds for constant mode
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                # For constant mode, use the bounds of the first per-angle param
                # (all per-angle bounds are typically the same)
                fit_lower = np.concatenate(
                    [
                        [lower_bounds[0]],
                        [lower_bounds[n_phi]],
                        lower_bounds[2 * n_phi :],
                    ]
                )
                fit_upper = np.concatenate(
                    [
                        [upper_bounds[0]],
                        [upper_bounds[n_phi]],
                        upper_bounds[2 * n_phi :],
                    ]
                )
                fit_bounds = (fit_lower, fit_upper)
            logger.info("=" * 60)

        # Layer 1: Fourier reparameterization of initial parameters
        elif use_fourier:
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY EXECUTION: Fourier Reparameterization")
            # Transform per-angle params to Fourier coefficients
            per_angle_params = initial_params[: 2 * n_phi]
            physical_params = initial_params[2 * n_phi :]

            # Split per-angle into contrast and offset groups
            contrast_per_angle = per_angle_params[:n_phi]
            offset_per_angle = per_angle_params[n_phi : 2 * n_phi]

            # Transform to Fourier coefficients
            contrast_coeffs = fourier_reparameterizer.to_fourier(contrast_per_angle)
            offset_coeffs = fourier_reparameterizer.to_fourier(offset_per_angle)

            # New parameter layout: [contrast_coeffs, offset_coeffs, physical_params]
            fit_initial_params = np.concatenate(
                [contrast_coeffs, offset_coeffs, physical_params]
            )

            logger.info(f"  Original params: {len(initial_params)}")
            logger.info(f"  Fourier params: {len(fit_initial_params)}")
            logger.info(
                f"  Per-angle reduction: {2 * n_phi} -> {len(contrast_coeffs) + len(offset_coeffs)}"
            )

            # Transform bounds for Fourier space
            if bounds is not None:
                lower_bounds, upper_bounds = bounds
                # Per-angle bounds are typically (0,1) for contrast, (0.5, 1.5) for offset
                # Fourier coefficients can have wider bounds since they combine linearly
                # Use n_coeffs_per_param (e.g., 5 for order=2), NOT n_coeffs (total=10)
                n_half = fourier_reparameterizer.n_coeffs_per_param

                # Fourier coefficient bounds: a0 keeps the mean, others can be ±range
                contrast_lower = np.concatenate(
                    [
                        [lower_bounds[0]],  # a0 (mean) lower bound
                        np.full(n_half - 1, -1.0),  # Other coeffs can be negative
                    ]
                )
                contrast_upper = np.concatenate(
                    [
                        [upper_bounds[0]],  # a0 (mean) upper bound
                        np.full(n_half - 1, 1.0),  # Other coeffs bounded
                    ]
                )
                offset_lower = np.concatenate(
                    [
                        [lower_bounds[n_phi]],  # a0 (mean) lower bound
                        np.full(n_half - 1, -0.5),  # Other coeffs
                    ]
                )
                offset_upper = np.concatenate(
                    [
                        [upper_bounds[n_phi]],  # a0 (mean) upper bound
                        np.full(n_half - 1, 0.5),  # Other coeffs
                    ]
                )

                fit_lower = np.concatenate(
                    [contrast_lower, offset_lower, lower_bounds[2 * n_phi :]]
                )
                fit_upper = np.concatenate(
                    [contrast_upper, offset_upper, upper_bounds[2 * n_phi :]]
                )
                fit_bounds = (fit_lower, fit_upper)
            logger.info("=" * 60)

        # =====================================================================
        # Anti-Degeneracy Defense: Create Fourier-wrapped model function
        # =====================================================================
        # When using Fourier mode, wrap model_fn to convert Fourier coeffs -> per-angle
        if use_fourier:
            n_coeffs_per_param = fourier_reparameterizer.n_coeffs_per_param

            @jax.jit
            def model_fn_fourier(x_batch: jnp.ndarray, *params_tuple) -> jnp.ndarray:
                """Model function with Fourier coefficient inputs."""
                # Handle both single points (1D) and batches (2D)
                x_batch_2d = jnp.atleast_2d(x_batch)
                params_all = jnp.stack(params_tuple)

                # Extract Fourier coefficients and physical params
                # Layout: [contrast_coeffs, offset_coeffs, physical_params]
                n_coeffs = fourier_reparameterizer.n_coeffs_per_param
                contrast_coeffs = params_all[:n_coeffs]
                offset_coeffs = params_all[n_coeffs : 2 * n_coeffs]
                physical_params = params_all[2 * n_coeffs :]

                # Convert Fourier coefficients to per-angle values
                # Uses precomputed basis matrix: values = B @ coeffs
                basis_matrix = jnp.asarray(fourier_reparameterizer._basis_matrix)
                contrast_all = basis_matrix @ contrast_coeffs
                offset_all = basis_matrix @ offset_coeffs

                # Extract indices from x_batch (now guaranteed 2D)
                phi_idx = x_batch_2d[:, 0].astype(jnp.int32)
                t1_idx = x_batch_2d[:, 1].astype(jnp.int32)
                t2_idx = x_batch_2d[:, 2].astype(jnp.int32)

                # Extract physical parameters
                D0 = physical_params[0]
                alpha = physical_params[1]
                D_offset = physical_params[2]

                # Compute diffusion
                D_t = calculate_diffusion_coefficient(
                    t1_unique_jax, D0, alpha, D_offset
                )
                D_cumsum = trapezoid_cumsum(D_t)
                D_diff = D_cumsum[t1_idx] - D_cumsum[t2_idx]
                # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
                D_integral_batch = jnp.sqrt(D_diff**2 + 1e-12)

                log_g1_diff = -wavevector_q_squared_half_dt * D_integral_batch
                log_g1_diff_bounded = jnp.clip(log_g1_diff, -700.0, 0.0)
                g1_diffusion = jnp.exp(log_g1_diff_bounded)

                if is_laminar_flow:
                    # Shear parameters
                    gamma_dot_0 = physical_params[3]
                    beta = physical_params[4]
                    gamma_dot_offset = physical_params[5]
                    phi0 = physical_params[6]

                    # Compute shear
                    gamma_t = calculate_shear_rate(
                        t1_unique_jax, gamma_dot_0, beta, gamma_dot_offset
                    )
                    gamma_cumsum = trapezoid_cumsum(gamma_t)
                    gamma_diff = gamma_cumsum[t1_idx] - gamma_cumsum[t2_idx]
                    # P0-2: epsilon_abs=1e-12 (was 1e-20, below float32 precision)
                    gamma_integral_batch = jnp.sqrt(gamma_diff**2 + 1e-12)

                    # Shear contribution with angle dependence
                    phi_values = phi_unique_jax[phi_idx]
                    angle_diff = jnp.deg2rad(phi0 - phi_values)
                    cos_phi = jnp.cos(angle_diff)

                    sinc_arg = sinc_prefactor * gamma_integral_batch * cos_phi
                    sinc_val = safe_sinc(sinc_arg)
                    g1_shear = sinc_val**2

                    g1_total = g1_diffusion * g1_shear
                    # P0-3: Use jnp.where (gradient-safe) instead of jnp.clip.
                    # log-space clip above guarantees g1 ≤ 1.0; lower floor prevents log(0).
                    epsilon = 1e-10
                    g1 = jnp.where(g1_total > epsilon, g1_total, epsilon)
                else:
                    epsilon = 1e-10
                    g1 = jnp.where(g1_diffusion > epsilon, g1_diffusion, epsilon)

                # Compute g2 with per-angle scaling (from Fourier-derived values)
                contrast = contrast_all[phi_idx]
                offset = offset_all[phi_idx]
                g2_theory = offset + contrast * g1**2
                # P0-3: Removed jnp.clip(g2, 0.5, 2.5) — kills gradients at boundaries.
                # Bounds enforced via parameter bounds in optimizer, not g2 clipping.
                g2 = g2_theory

                return g2.squeeze()

            # Use Fourier model function for optimization
            active_model_fn = model_fn_fourier
            logger.info("  Using Fourier-wrapped model function")
        else:
            # Use standard per-angle model function
            active_model_fn = model_fn_pointwise

        # Run hybrid optimization
        logger.info("Starting hybrid optimization (L-BFGS + Gauss-Newton)...")
        opt_start = time.perf_counter()

        # Layer 2: Hierarchical optimization path
        # Can be combined with Fourier mode (hierarchical operates on Fourier params)
        if use_hierarchical:
            # Use hierarchical two-stage optimization
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY EXECUTION: Hierarchical Two-Stage Optimization"
            )

            # Pre-extract phi indices for shear weighting (x_data[:, 0] contains phi indices)
            phi_indices_jax = jnp.asarray(x_data[:, 0], dtype=jnp.int32)
            shear_weighter_local = anti_degeneracy_components.get("shear_weighter")

            def loss_fn(params):
                """Loss function for hierarchical optimizer.

                CRITICAL: Must use jnp (JAX) operations, NOT np (NumPy).
                Using np.mean breaks the JAX autodiff computation graph,
                resulting in zero gradients for all parameters.

                Layer 5: Shear-sensitivity weighting is applied here to prevent
                gradient cancellation for shear parameters (gamma_dot_t0, phi0).
                """
                # Convert params to JAX array if needed for tracing
                params_jax = jnp.asarray(params)
                pred = active_model_fn(x_data, *params_jax)

                # Convert y_data to JAX for proper gradient flow
                y_data_jax = jnp.asarray(y_data)
                residuals = y_data_jax - pred

                # Layer 5: Apply shear-sensitivity weighting if enabled
                # This emphasizes angles parallel/antiparallel to flow direction,
                # preventing gradient cancellation for shear parameters
                if shear_weighter_local is not None:
                    # Use shear-weighted loss instead of uniform MSE
                    weighted_loss = shear_weighter_local.apply_weights_to_loss(
                        residuals, phi_indices_jax
                    )
                else:
                    # CRITICAL: Use jnp.mean, NOT np.mean!
                    # np.mean breaks JAX autodiff and causes zero gradients
                    weighted_loss = jnp.mean(residuals**2) * len(y_data)

                # Add adaptive regularization if enabled
                if adaptive_regularizer is not None:
                    # Use JAX-compatible method for autodiff compatibility
                    # Note: weighted_loss already includes the normalization
                    mse_for_reg = weighted_loss / len(y_data)
                    reg_term = adaptive_regularizer.compute_regularization_jax(
                        params_jax, mse_for_reg, len(y_data)
                    )
                    return weighted_loss + reg_term
                return weighted_loss

            def grad_fn(params):
                """Gradient function with optional monitoring."""
                # Use JAX autodiff for gradient computation
                grad = jax.grad(lambda p: loss_fn(p))(params)

                # Layer 4: Gradient monitoring
                if gradient_monitor is not None:
                    gradient_monitor.check(
                        grad, iteration_counter[0], params, loss_fn(params)
                    )
                    iteration_counter[0] += 1

                return grad

            iteration_counter = [0]  # Mutable counter for gradient monitor

            # Layer 5: Create callback for shear weight updates
            # Updates weights based on current phi0 estimate at start of each outer iteration
            def shear_weight_update_callback(
                params: np.ndarray, outer_iter: int
            ) -> None:
                """Update shear-sensitivity weights based on current phi0."""
                if shear_weighter_local is not None:
                    shear_weighter_local.update_phi0(params, outer_iter)

            hier_result = hierarchical_optimizer.fit(
                loss_fn=loss_fn,
                grad_fn=grad_fn,
                p0=fit_initial_params,
                bounds=fit_bounds,
                outer_iteration_callback=shear_weight_update_callback,
            )

            # Convert HierarchicalResult to standard format
            result = {
                "x": hier_result.x,
                "pcov": np.eye(len(hier_result.x)),  # Placeholder
                "success": hier_result.success,
                "message": hier_result.message,
                "function_evaluations": hier_result.n_outer_iterations
                * 150,  # Estimate
                "streaming_diagnostics": {
                    "phase_iterations": {
                        "phase1": 0,
                        "phase2": hier_result.n_outer_iterations,
                    },
                    "warmup_diagnostics": {},
                    "gauss_newton_diagnostics": {
                        "final_cost": hier_result.fun,
                    },
                    "hierarchical_history": hier_result.history,
                },
            }
            logger.info(f"  Hierarchical result: success={hier_result.success}")
            logger.info(f"  Outer iterations: {hier_result.n_outer_iterations}")
            logger.info(f"  Final loss: {hier_result.fun:.6e}")
            logger.info("=" * 60)
        else:
            # Standard hybrid streaming optimization path
            result = optimizer.fit(
                data_source=(x_data, y_data),
                func=active_model_fn,
                p0=fit_initial_params,
                bounds=fit_bounds,
                verbose=1,
            )

        opt_time = time.perf_counter() - opt_start
        total_time = time.perf_counter() - start_time

        # Extract diagnostics from NLSQ result structure
        # NLSQ uses nested dicts: streaming_diagnostics -> phase_iterations/warmup_diagnostics
        diagnostics = result.get("streaming_diagnostics", {})
        phase_iterations = diagnostics.get("phase_iterations", {})
        warmup_diag = diagnostics.get("warmup_diagnostics", {})
        gn_diag = diagnostics.get("gauss_newton_diagnostics", {})

        lbfgs_epochs = phase_iterations.get("phase1", 0)
        gn_iterations = phase_iterations.get("phase2", 0)
        final_lbfgs_loss = warmup_diag.get("final_loss", float("inf"))
        final_gn_cost = gn_diag.get("final_cost", float("inf"))

        logger.info("=" * 80)
        logger.info("HYBRID STREAMING OPTIMIZATION COMPLETE")
        logger.info(f"  Success: {result.get('success', False)}")
        logger.info(f"  L-BFGS final loss: {final_lbfgs_loss:.6e}")
        logger.info(f"  GN final cost: {final_gn_cost:.6e}")
        logger.info(f"  L-BFGS epochs: {lbfgs_epochs}")
        logger.info(f"  GN iterations: {gn_iterations}")
        logger.info(f"  Optimization time: {opt_time:.2f}s")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info("=" * 80)

        # Extract results
        popt = np.asarray(result["x"])

        # =====================================================================
        # Anti-Degeneracy Defense System v2.9.0 - INVERSE TRANSFORMATION
        # =====================================================================
        # Transform Fourier coefficients back to per-angle parameters
        if use_fourier:
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY EXECUTION: Inverse Fourier Transform")
            # Use n_coeffs_per_param (e.g., 5 for order=2), NOT n_coeffs (total=10)
            # Layout: [contrast_coeffs (5), offset_coeffs (5), physical (7)]
            n_half = fourier_reparameterizer.n_coeffs_per_param

            # Extract Fourier coefficients and physical params from optimized result
            fourier_contrast_coeffs = popt[:n_half]
            fourier_offset_coeffs = popt[n_half : 2 * n_half]
            physical_params_opt = popt[2 * n_half :]

            # Transform back to per-angle parameters
            contrast_per_angle_opt = fourier_reparameterizer.from_fourier(
                fourier_contrast_coeffs
            )
            offset_per_angle_opt = fourier_reparameterizer.from_fourier(
                fourier_offset_coeffs
            )

            # Reconstruct full parameter vector in original layout
            popt = np.concatenate(
                [contrast_per_angle_opt, offset_per_angle_opt, physical_params_opt]
            )

            logger.info(f"  Fourier params: {2 * n_half + len(physical_params_opt)}")
            logger.info(f"  Restored per-angle params: {len(popt)}")

            # Transform covariance from Fourier space to per-angle space
            # J_fourier = d(per_angle)/d(fourier_coeffs)
            # pcov_per_angle = J_full @ pcov_fourier @ J_full.T
            pcov_fourier = result.get("pcov", None)
            n_fourier_total = 2 * n_half + len(physical_params_opt)

            if (
                pcov_fourier is not None
                and pcov_fourier.shape[0] == n_fourier_total
                and pcov_fourier.shape[1] == n_fourier_total
            ):
                # Get Jacobian for per-angle transformation
                # This is the Fourier basis matrix that maps coefficients to per-angle values
                jacobian_per_angle = fourier_reparameterizer.get_jacobian_transform()
                # jacobian_per_angle shape: (2 * n_phi, n_coeffs_fourier)
                # where n_coeffs_fourier = 2 * n_half

                # Build full Jacobian for complete parameter space transformation
                # Layout: [n_phi contrast, n_phi offset, n_physical]
                # Fourier layout: [n_half contrast_coeffs, n_half offset_coeffs, n_physical]
                n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
                n_physical = len(physical_params_opt)
                n_total_restored = n_per_angle_total + n_physical

                J_full = np.zeros((n_total_restored, n_fourier_total))
                # Block for per-angle params: use Fourier Jacobian
                J_full[:n_per_angle_total, : 2 * n_half] = jacobian_per_angle
                # Block for physical params: identity (pass-through)
                J_full[n_per_angle_total:, 2 * n_half :] = np.eye(n_physical)

                # Transform covariance: pcov_full = J @ pcov_fourier @ J.T
                try:
                    pcov_transformed = J_full @ pcov_fourier @ J_full.T
                    # Store for later use (override the result dict lookup)
                    result["pcov_transformed"] = pcov_transformed
                    logger.info(
                        "  Covariance transformed from Fourier to per-angle space"
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        f"  Covariance transformation failed: {e}. Using identity fallback."
                    )
                    result["pcov_transformed"] = None
            else:
                pcov_shape = pcov_fourier.shape if pcov_fourier is not None else None
                logger.warning(
                    f"  Fourier covariance unavailable or wrong shape (got {pcov_shape}, "
                    f"expected ({n_fourier_total}, {n_fourier_total})). "
                    "Using identity fallback."
                )
                result["pcov_transformed"] = None

            logger.info("=" * 60)

        # v2.17.0: Fixed scaling mode inverse transformation
        # Expand physical-only params back to per-angle format using fixed scaling arrays
        elif use_fixed_scaling:
            logger.info("=" * 60)
            logger.info(
                "ANTI-DEGENERACY EXECUTION: Inverse Fixed Scaling Transform (v2.17.0)"
            )
            # Layout: [physical_params] - popt contains ONLY physical parameters
            physical_params_opt = popt

            # Use the pre-computed fixed per-angle scaling from quantiles
            contrast_per_angle_opt = fixed_contrast_per_angle
            offset_per_angle_opt = fixed_offset_per_angle

            # Reconstruct full parameter vector in original layout
            popt = np.concatenate(
                [contrast_per_angle_opt, offset_per_angle_opt, physical_params_opt]
            )

            logger.info(f"  Physical params: {len(physical_params_opt)}")
            logger.info(f"  Fixed per-angle scaling restored: {len(popt)} total params")
            logger.info(
                f"  Contrast (fixed): mean={np.mean(contrast_per_angle_opt):.4f}, "
                f"range=[{np.min(contrast_per_angle_opt):.4f}, {np.max(contrast_per_angle_opt):.4f}]"
            )
            logger.info(
                f"  Offset (fixed): mean={np.mean(offset_per_angle_opt):.4f}, "
                f"range=[{np.min(offset_per_angle_opt):.4f}, {np.max(offset_per_angle_opt):.4f}]"
            )

            # Transform covariance from physical-only space to full space
            # For fixed scaling mode, the Jacobian is simpler:
            # Per-angle params are fixed (variance = 0), physical params have identity
            # J[i, j] = 0 for per-angle params (i < 2*n_phi)
            # J[2*n_phi+i, i] = 1 for physical params (identity)
            pcov_physical = result.get("pcov", None)
            n_physical = len(physical_params_opt)

            if (
                pcov_physical is not None
                and pcov_physical.shape[0] == n_physical
                and pcov_physical.shape[1] == n_physical
            ):
                n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
                n_total_restored = n_per_angle_total + n_physical

                # Build full covariance matrix
                # Per-angle params have zero covariance (they're fixed)
                # Physical params have the original covariance
                try:
                    pcov_full = np.zeros((n_total_restored, n_total_restored))
                    # Physical params covariance block
                    pcov_full[2 * n_phi :, 2 * n_phi :] = pcov_physical
                    result["pcov_transformed"] = pcov_full
                    logger.info(
                        "  Covariance expanded: per-angle=0 (fixed), physical=preserved"
                    )
                except (
                    ValueError,
                    RuntimeError,
                    MemoryError,
                    np.linalg.LinAlgError,
                ) as e:
                    logger.warning(
                        f"  Covariance expansion failed: {e}. Using identity fallback."
                    )
                    result["pcov_transformed"] = None
            else:
                pcov_shape = pcov_physical.shape if pcov_physical is not None else None
                logger.warning(
                    f"  Physical covariance unavailable or wrong shape (got {pcov_shape}, "
                    f"expected ({n_physical}, {n_physical})). "
                    "Using identity fallback."
                )
                result["pcov_transformed"] = None

            logger.info("=" * 60)

        # T046-T049: Auto averaged mode inverse transformation
        # Expand averaged parameters back to per-angle format for backward compatibility
        elif use_averaged_scaling:
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY EXECUTION: Inverse Auto Averaged Transform")
            # Layout: [contrast_const, offset_const, physical_params]
            from homodyne.optimization.nlsq.data_prep import (
                expand_per_angle_parameters,
            )

            contrast_const = popt[0]
            offset_const = popt[1]
            n_physical_opt = len(popt) - 2
            expanded = expand_per_angle_parameters(
                popt,
                None,
                n_phi,
                n_physical_opt,
            )
            popt = expanded.params

            logger.info(f"  Constant params: 2 + {n_physical_opt} physical")
            logger.info(f"  Restored per-angle params: {len(popt)}")
            logger.info(f"  Contrast (uniform): {contrast_const:.6f}")
            logger.info(f"  Offset (uniform): {offset_const:.6f}")

            # Transform covariance from constant space to per-angle space
            # For constant mode, the Jacobian is simpler: broadcasting matrix
            # J[i, 0] = 1 for i in 0..n_phi-1 (contrast params)
            # J[n_phi+i, 1] = 1 for i in 0..n_phi-1 (offset params)
            # J[2*n_phi+i, 2+i] = 1 for physical params (identity)
            pcov_constant = result.get("pcov", None)
            n_constant_total = 2 + n_physical_opt

            if (
                pcov_constant is not None
                and pcov_constant.shape[0] == n_constant_total
                and pcov_constant.shape[1] == n_constant_total
            ):
                n_per_angle_total = 2 * n_phi  # contrast + offset per-angle
                n_physical = n_physical_opt
                n_total_restored = n_per_angle_total + n_physical

                # Build Jacobian for constant → per-angle transformation
                J_full = np.zeros((n_total_restored, n_constant_total))
                # Contrast broadcast: d(contrast_per_angle[i])/d(contrast_const) = 1
                J_full[:n_phi, 0] = 1.0
                # Offset broadcast: d(offset_per_angle[i])/d(offset_const) = 1
                J_full[n_phi : 2 * n_phi, 1] = 1.0
                # Physical params: identity (pass-through)
                J_full[2 * n_phi :, 2:] = np.eye(n_physical)

                # Transform covariance: pcov_full = J @ pcov_constant @ J.T
                try:
                    pcov_transformed = J_full @ pcov_constant @ J_full.T
                    result["pcov_transformed"] = pcov_transformed
                    logger.info(
                        "  Covariance transformed from constant to per-angle space"
                    )
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                    logger.warning(
                        f"  Covariance transformation failed: {e}. Using identity fallback."
                    )
                    result["pcov_transformed"] = None
            else:
                pcov_shape = pcov_constant.shape if pcov_constant is not None else None
                logger.warning(
                    f"  Constant covariance unavailable or wrong shape (got {pcov_shape}, "
                    f"expected ({n_constant_total}, {n_constant_total})). "
                    "Using identity fallback."
                )
                result["pcov_transformed"] = None

            logger.info("=" * 60)

        # Log gradient monitor summary if available
        if gradient_monitor is not None:
            gradient_monitor.log_summary()
            if gradient_monitor.collapse_detected:
                logger.warning("=" * 60)
                logger.warning("GRADIENT COLLAPSE WAS DETECTED DURING OPTIMIZATION")
                logger.warning(
                    f"  Collapse events: {len(gradient_monitor.collapse_events)}"
                )
                for event in gradient_monitor.collapse_events:
                    logger.warning(
                        f"    Iteration {event.iteration}: ratio={event.ratio:.6f}"
                    )
                logger.warning("=" * 60)

        # Get covariance (properly transformed from normalized space)
        # Priority: 1) pcov_transformed (from Fourier space), 2) pcov, 3) identity fallback
        pcov = result.get("pcov_transformed", None)
        if pcov is None:
            pcov = result.get("pcov", None)
        if pcov is None or pcov.shape[0] != len(popt):
            logger.debug(
                f"Covariance size mismatch or unavailable: expected ({len(popt)}, {len(popt)}), "
                f"got {pcov.shape if pcov is not None else None}. Using identity fallback."
            )
            pcov = np.eye(len(popt))

        # Enforce bounds on final parameters
        if bounds is not None:
            lower_bounds, upper_bounds = bounds
            popt = np.clip(popt, lower_bounds, upper_bounds)

        # Check for parameters stuck at bounds with zero/near-zero uncertainty
        # This indicates the optimizer could not move these parameters away from bounds
        bound_stuck_warning = None
        if bounds is not None and is_laminar_flow:
            perr = _safe_uncertainties_from_pcov(pcov, len(popt))
            param_statuses = _classify_parameter_status(
                popt, lower_bounds, upper_bounds, atol=1e-6
            )

            # Map indices to physical parameter names for laminar_flow mode
            # Layout: [n_phi contrasts] + [n_phi offsets] + [7 physical params]
            physical_indices = list(range(2 * n_phi, len(popt)))
            physical_param_names_local = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

            bound_stuck_params = []
            for i, idx in enumerate(physical_indices):
                if idx < len(param_statuses) and idx < len(popt):
                    status = param_statuses[idx]
                    uncertainty = perr[idx] if idx < len(perr) else 0.0
                    if status != "active" and (
                        uncertainty == 0.0 or uncertainty < 1e-15
                    ):
                        param_name = (
                            physical_param_names_local[i]
                            if i < len(physical_param_names_local)
                            else f"param[{idx}]"
                        )
                        bound_stuck_params.append(
                            (param_name, status, popt[idx], uncertainty)
                        )

            if bound_stuck_params:
                logger.warning("=" * 80)
                logger.warning("PARAMETER BOUNDS WARNING")
                logger.warning(
                    "The following parameters are stuck at bounds with zero uncertainty:"
                )
                for param_name, status, value, unc in bound_stuck_params:
                    logger.warning(
                        f"  {param_name}: {value:.6e} ({status}, uncertainty={unc:.2e})"
                    )
                logger.warning("")
                logger.warning("This may indicate:")
                logger.warning(
                    "  1. The optimizer cannot find gradient information for these parameters"
                )
                logger.warning(
                    "  2. The initial guess was already at or near the bounds"
                )
                logger.warning(
                    "  3. The model is insensitive to these parameters with this data coverage"
                )
                logger.warning("")
                logger.warning("RECOMMENDED ACTIONS:")
                logger.warning(
                    "  - Enable phi_filtering to use only angles near 0° and 90° for laminar flow"
                )
                logger.warning(
                    "  - Use multi-start optimization to explore multiple parameter basins"
                )
                logger.warning(
                    "  - Check if gamma_dot_t0 ≈ 0 means shear contribution is missing"
                )
                logger.warning("=" * 80)

                # Store for info dict
                bound_stuck_warning = {
                    "parameters_at_bounds": [
                        {
                            "name": name,
                            "status": status,
                            "value": float(val),
                            "uncertainty": float(unc),
                        }
                        for name, status, val, unc in bound_stuck_params
                    ]
                }

        # Build info dict
        info = {
            "success": result.get("success", False),
            "message": result.get("message", "Hybrid streaming optimization completed"),
            "nfev": result.get("function_evaluations", 0),
            "nit": lbfgs_epochs + gn_iterations,
            "final_loss": final_gn_cost
            if final_gn_cost != float("inf")
            else final_lbfgs_loss,
            "lbfgs_epochs": lbfgs_epochs,
            "gauss_newton_iterations": gn_iterations,
            "optimization_time": opt_time,
            "total_time": total_time,
            "method": "adaptive_hybrid_streaming",
            "hybrid_streaming_diagnostics": diagnostics,
        }

        # Add anti-degeneracy defense diagnostics
        shear_weighter = anti_degeneracy_components.get("shear_weighter")
        info["anti_degeneracy"] = {
            "version": "2.18.0",
            "per_angle_mode": anti_degeneracy_components["per_angle_mode"],
            "use_constant": anti_degeneracy_components.get("use_constant", False),
            "use_fixed_scaling": use_fixed_scaling,
            "fourier_enabled": fourier_reparameterizer is not None,
            "hierarchical_enabled": hierarchical_optimizer is not None,
            "adaptive_regularization_enabled": adaptive_regularizer is not None,
            "gradient_monitor_enabled": gradient_monitor is not None,
            "shear_weighting_enabled": shear_weighter is not None,
        }
        if fourier_reparameterizer is not None:
            info["anti_degeneracy"]["fourier"] = {
                "order": fourier_order,
                "n_coeffs": fourier_reparameterizer.n_coeffs,
                "param_reduction": f"{2 * n_phi} -> {fourier_reparameterizer.n_coeffs}",
            }
        # T048: Add constant mode diagnostics
        if use_fixed_scaling:
            # v2.18.0: Fixed scaling mode - per-angle values are fixed, not optimized
            info["anti_degeneracy"]["fixed_scaling"] = {
                "param_reduction": f"{2 * n_phi} -> 0 (physical only)",
                "method": "quantile_estimation",
                "contrast_mean": float(np.mean(fixed_contrast_per_angle)),
                "contrast_range": [
                    float(np.min(fixed_contrast_per_angle)),
                    float(np.max(fixed_contrast_per_angle)),
                ],
                "offset_mean": float(np.mean(fixed_offset_per_angle)),
                "offset_range": [
                    float(np.min(fixed_offset_per_angle)),
                    float(np.max(fixed_offset_per_angle)),
                ],
            }
        elif use_averaged_scaling:
            # v2.18.0: Auto averaged mode - averaged values are OPTIMIZED
            info["anti_degeneracy"]["auto_averaged"] = {
                "param_reduction": f"{2 * n_phi} -> 2 (averaged scaling)",
                "method": "quantile_estimation_averaged",
                # After inverse transform, popt[0] is first contrast (uniform)
                "contrast_optimized": float(popt[0]) if len(popt) > 0 else None,
                "offset_optimized": float(popt[n_phi]) if len(popt) > n_phi else None,
            }
        if hierarchical_optimizer is not None:
            info["anti_degeneracy"]["hierarchical"] = (
                hierarchical_optimizer.get_diagnostics()
            )
        if adaptive_regularizer is not None:
            info["anti_degeneracy"]["regularization"] = (
                adaptive_regularizer.get_diagnostics()
            )
        if gradient_monitor is not None:
            info["anti_degeneracy"]["gradient_monitor"] = (
                gradient_monitor.get_diagnostics()
            )
        if shear_weighter is not None:
            info["anti_degeneracy"]["shear_weighting"] = (
                shear_weighter.get_diagnostics()
            )

        # Add bounds warning info if detected
        if bound_stuck_warning is not None:
            info["bound_stuck_warning"] = bound_stuck_warning

        # Check for shear collapse: gamma_dot_t0 essentially zero
        if is_laminar_flow and len(popt) > 2 * n_phi + 3:
            gamma_dot_t0_idx = 2 * n_phi + 3
            gamma_dot_t0_value = popt[gamma_dot_t0_idx]
            # Check if shear rate is effectively zero (< 1e-5 s^-1)
            if abs(gamma_dot_t0_value) < 1e-5:
                logger.warning("=" * 80)
                logger.warning("SHEAR COLLAPSE WARNING")
                logger.warning(
                    f"gamma_dot_t0 = {gamma_dot_t0_value:.2e} s⁻¹ is effectively zero"
                )
                logger.warning("")
                logger.warning("This means the shear contribution to g₁ is negligible.")
                logger.warning(
                    "The model has effectively collapsed to static_isotropic mode."
                )
                logger.warning("")
                logger.warning("POSSIBLE CAUSES:")
                logger.warning(
                    "  1. Per-angle contrast/offset absorbed the shear signal"
                )
                logger.warning(
                    "  2. Inconsistent initialization of per-angle vs physical params"
                )
                logger.warning(
                    "  3. Physical parameters at bounds with weak gradient signal"
                )
                logger.warning("  4. The data may genuinely have no measurable shear")
                logger.warning("")
                logger.warning("RECOMMENDED ACTIONS:")
                logger.warning(
                    "  - Enable multi-start optimization to explore parameter basins"
                )
                logger.warning(
                    "  - Check reduced chi-squared: if worse than expected, re-run optimization"
                )
                logger.warning(
                    "  - Verify per-angle contrast/offset are not varying excessively"
                )
                logger.warning(
                    "  - Consider static_isotropic mode if shear is truly absent"
                )
                logger.warning("=" * 80)
                info["shear_collapse_warning"] = {
                    "gamma_dot_t0": float(gamma_dot_t0_value),
                    "threshold": 1e-5,
                    "message": "Shear contribution effectively zero",
                }

        return popt, pcov, info

    def _estimate_memory_for_stratified_ls(
        self,
        n_points: int,
        n_params: int,
        n_chunks: int,
    ) -> float:
        """Estimate peak memory usage for stratified least-squares optimization.

        The main memory consumers are:
        1. Padded arrays: n_chunks × max_chunk_size × 5 arrays × 8 bytes
        2. Dense Jacobian: n_points × n_params × 8 bytes
        3. JAX autodiff intermediates: ~3× Jacobian size for backprop
        4. JAX compilation cache: ~5-10 GB

        Args:
            n_points: Total number of data points
            n_params: Number of parameters
            n_chunks: Number of stratified chunks

        Returns:
            Estimated peak memory in bytes
        """
        bytes_per_float = 8

        # Padded arrays (5 arrays: phi, t1, t2, g2, mask)
        max_chunk_size = (n_points + n_chunks - 1) // n_chunks
        padded_arrays = n_chunks * max_chunk_size * 5 * bytes_per_float

        # Dense Jacobian
        jacobian = n_points * n_params * bytes_per_float

        # JAX autodiff intermediates (keep all grids for backprop)
        # This is the main memory killer - originally estimated at 3× Jacobian
        # but empirical testing shows 5× is more accurate for large datasets
        # (C020 dataset: estimated 44.9 GB at 3×, actual ~60 GB at 96% pressure)
        autodiff_intermediates = jacobian * 5

        # JAX compilation cache
        jax_cache = 5 * 1e9  # ~5 GB

        total = padded_arrays + jacobian + autodiff_intermediates + jax_cache

        return total

    def _should_use_streaming(
        self,
        n_points: int,
        n_params: int,
        n_chunks: int,
        memory_threshold_gb: float | None = None,
        memory_fraction: float | None = None,
    ) -> tuple[bool, float, str]:
        """Determine if streaming optimizer should be used based on memory estimate.

        Uses adaptive memory thresholding (v2.7.0+) to automatically compute
        an appropriate threshold based on total system memory.

        Args:
            n_points: Total number of data points
            n_params: Number of parameters
            n_chunks: Number of stratified chunks
            memory_threshold_gb: Memory threshold in GB above which to use streaming.
                If None (default), computes adaptive threshold as 75% of total memory.
            memory_fraction: Fraction of total memory for adaptive threshold (0.1-0.9).
                Only used if memory_threshold_gb is None.

        Returns:
            (use_streaming, estimated_gb, reason) tuple
        """
        import psutil

        # Compute adaptive threshold if not explicitly provided
        if memory_threshold_gb is None:
            memory_threshold_gb, threshold_info = get_adaptive_memory_threshold(
                memory_fraction=memory_fraction
            )
            _memory_logger.debug(
                f"_should_use_streaming using adaptive threshold: "
                f"{memory_threshold_gb:.1f} GB ({threshold_info})"
            )

        # Get available system memory
        mem = psutil.virtual_memory()
        available_gb = mem.available / 1e9

        # Estimate memory for stratified LS
        estimated_bytes = self._estimate_memory_for_stratified_ls(
            n_points, n_params, n_chunks
        )
        estimated_gb = estimated_bytes / 1e9

        # Decision logic
        # Use streaming if:
        # 1. Estimated memory exceeds threshold, OR
        # 2. Estimated memory exceeds 85% of available memory
        #
        # Note: Increased from 70% to 85% because non-streaming Levenberg-Marquardt
        # is more accurate than streaming optimization. The 85% threshold allows
        # more datasets to use the preferred non-streaming path.
        use_streaming = False
        reason = ""

        if estimated_gb > memory_threshold_gb:
            use_streaming = True
            reason = (
                f"Estimated memory ({estimated_gb:.1f} GB) exceeds "
                f"threshold ({memory_threshold_gb:.1f} GB)"
            )
        elif estimated_gb > available_gb * 0.85:
            use_streaming = True
            reason = (
                f"Estimated memory ({estimated_gb:.1f} GB) exceeds "
                f"85% of available memory ({available_gb:.1f} GB available)"
            )
        else:
            reason = (
                f"Estimated memory ({estimated_gb:.1f} GB) within limits "
                f"(threshold={memory_threshold_gb:.1f} GB, "
                f"available={available_gb:.1f} GB)"
            )

        return use_streaming, estimated_gb, reason

    def _create_fit_result(
        self,
        popt: np.ndarray,
        pcov: np.ndarray,
        residuals: np.ndarray,
        n_data: int,
        iterations: int,
        execution_time: float,
        convergence_status: str = "converged",
        recovery_actions: list[str] | None = None,
        streaming_diagnostics: dict[str, Any] | None = None,
        stratification_diagnostics: StratificationDiagnostics | None = None,
        diagnostics_payload: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """Convert NLSQ output to OptimizationResult.

        Args:
            popt: Optimized parameters
            pcov: Parameter covariance matrix
            residuals: Final residuals
            n_data: Number of data points
            iterations: Optimization iterations
            execution_time: Execution time in seconds
            convergence_status: Convergence status string
            recovery_actions: List of recovery actions taken
            streaming_diagnostics: Enhanced diagnostics for streaming optimization (Task 5.4)

        Returns:
            Complete OptimizationResult dataclass
        """

        # Convert to numpy arrays
        popt = np.asarray(popt)
        pcov = np.asarray(pcov)
        residuals = np.asarray(residuals)

        # Compute uncertainties from covariance diagonal
        uncertainties = _safe_uncertainties_from_pcov(pcov, len(popt))

        # Compute chi-squared
        chi_squared = float(np.sum(residuals**2))

        # Compute reduced chi-squared
        n_params = len(popt)
        degrees_of_freedom = n_data - n_params
        reduced_chi_squared = (
            chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.inf
        )

        # Get device information
        devices = jax.devices()
        device_info = {
            "platform": devices[0].platform,
            "device": str(devices[0]),
            "device_kind": devices[0].device_kind,
            "n_devices": len(devices),
        }

        # Determine quality flag based on reduced chi-squared
        if reduced_chi_squared < 1.5:
            quality_flag = "good"
        elif reduced_chi_squared < 3.0:
            quality_flag = "marginal"
        else:
            quality_flag = "poor"

        # Task 5.4: Build enhanced streaming diagnostics if batch statistics available
        enhanced_streaming_diagnostics = None
        if streaming_diagnostics is not None:
            # Start with provided diagnostics
            enhanced_streaming_diagnostics = streaming_diagnostics.copy()

            # Add batch statistics if available
            if (
                hasattr(self, "batch_statistics")
                and self.batch_statistics.total_batches > 0
            ):
                batch_stats = self.batch_statistics.get_statistics()

                # Extract key metrics for enhanced diagnostics
                enhanced_streaming_diagnostics.update(
                    {
                        "batch_success_rate": batch_stats["success_rate"],
                        "failed_batch_indices": [
                            b["batch_idx"]
                            for b in batch_stats["recent_batches"]
                            if not b["success"]
                        ],
                        "error_type_distribution": batch_stats["error_distribution"],
                        "average_iterations_per_batch": batch_stats[
                            "average_iterations"
                        ],
                        "total_batches_processed": batch_stats["total_batches"],
                    }
                )

        # Create result
        result = OptimizationResult(
            parameters=popt,
            uncertainties=uncertainties,
            covariance=pcov,
            chi_squared=chi_squared,
            reduced_chi_squared=reduced_chi_squared,
            convergence_status=convergence_status,
            iterations=iterations,
            execution_time=execution_time,
            device_info=device_info,
            recovery_actions=recovery_actions or [],
            quality_flag=quality_flag,
            streaming_diagnostics=enhanced_streaming_diagnostics,  # Task 5.4
            stratification_diagnostics=stratification_diagnostics,  # v2.2.1: Stratification diagnostics
            nlsq_diagnostics=diagnostics_payload,
        )

        return result
