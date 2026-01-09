"""Optimization Strategy Executors for NLSQ.

This module implements the Strategy pattern for different optimization approaches,
enabling cleaner code organization and easier testing.

Extracted from wrapper.py as part of refactoring (Dec 2025).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

# Module-level logger for threshold warnings
_module_logger = get_logger(__name__)

# T040: Configurable threshold for slow operations (default 10s per FR-020)
SLOW_OPERATION_THRESHOLD_S = 10.0

# Import NLSQ functions
from nlsq import curve_fit, curve_fit_large

# Try importing AdaptiveHybridStreamingOptimizer (available in NLSQ >= 0.3.2)
# The old StreamingOptimizer was removed in NLSQ 0.4.0
try:
    from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

    STREAMING_AVAILABLE = True  # For backwards compatibility
except ImportError:
    STREAMING_AVAILABLE = False
    AdaptiveHybridStreamingOptimizer = None
    HybridStreamingConfig = None


@dataclass
class ExecutionResult:
    """Result from optimization execution.

    Attributes:
        popt: Optimized parameters
        pcov: Parameter covariance matrix
        info: Additional optimization information
        recovery_actions: List of recovery actions taken
        convergence_status: 'converged', 'partial', or 'failed'
    """

    popt: np.ndarray
    pcov: np.ndarray
    info: dict[str, Any]
    recovery_actions: list[str]
    convergence_status: str


class OptimizationExecutor(ABC):
    """Abstract base class for optimization strategy executors.

    Implements the Strategy pattern for different optimization approaches.
    Each concrete implementation handles a specific optimization method.
    """

    @abstractmethod
    def execute(
        self,
        residual_fn: Callable[..., Any],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | np.ndarray | str,
        logger: Any,
    ) -> ExecutionResult:
        """Execute optimization with the specific strategy.

        Args:
            residual_fn: Residual function to minimize
            xdata: Independent variable data
            ydata: Dependent variable data (observations)
            initial_params: Initial parameter guess
            bounds: Parameter bounds as (lower, upper) tuple
            loss_name: Loss function name (e.g., 'soft_l1')
            x_scale_value: Parameter scaling for trust region
            logger: Logger instance

        Returns:
            ExecutionResult with optimized parameters and diagnostics
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        pass

    @property
    @abstractmethod
    def supports_progress(self) -> bool:
        """Whether this strategy supports progress bars."""
        pass


class StandardExecutor(OptimizationExecutor):
    """Standard curve_fit optimization for small datasets (<1M points).

    Uses scipy.optimize.curve_fit through the NLSQ wrapper.
    Fast for small datasets, but doesn't handle large datasets efficiently.
    """

    @property
    def name(self) -> str:
        return "standard"

    @property
    def supports_progress(self) -> bool:
        return False

    def execute(
        self,
        residual_fn: Callable[..., Any],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | np.ndarray | str,
        logger: Any,
    ) -> ExecutionResult:
        """Execute standard curve_fit optimization."""
        logger.debug("Using standard curve_fit")
        _start_time = time.perf_counter()

        # Use parameter magnitude-based scaling for numerical stability
        if isinstance(x_scale_value, (int, float)) or (
            isinstance(x_scale_value, str) and x_scale_value == "jac"
        ):
            x_scale = np.abs(initial_params) + 1e-3
        elif isinstance(x_scale_value, np.ndarray):
            x_scale = x_scale_value
        else:
            x_scale = np.abs(initial_params) + 1e-3

        try:
            popt, pcov = curve_fit(
                residual_fn,
                xdata,
                ydata,
                p0=initial_params.tolist(),
                bounds=bounds,
                loss=loss_name,
                x_scale=x_scale,
                gtol=1e-6,
                ftol=1e-6,
                max_nfev=5000,
                verbose=2,
                stability="auto",  # Enable memory management and stability
            )

            _duration = time.perf_counter() - _start_time
            # T040: Log operations exceeding threshold at DEBUG level
            if _duration > SLOW_OPERATION_THRESHOLD_S:
                logger.debug(
                    f"[SLOW_OP] standard curve_fit took {_duration:.2f}s "
                    f"(threshold: {SLOW_OPERATION_THRESHOLD_S}s)"
                )

            info = {"success": True, "strategy": "standard", "duration_s": _duration}
            return ExecutionResult(
                popt=np.asarray(popt),
                pcov=np.asarray(pcov),
                info=info,
                recovery_actions=[],
                convergence_status="converged",
            )

        except Exception as e:
            logger.error(f"Standard optimization failed: {e}")
            raise


class LargeDatasetExecutor(OptimizationExecutor):
    """Large dataset optimization using curve_fit_large.

    Uses NLSQ's memory-efficient curve_fit_large function for datasets
    that exceed memory limits with standard curve_fit.
    """

    @property
    def name(self) -> str:
        return "large"

    @property
    def supports_progress(self) -> bool:
        return True

    def execute(
        self,
        residual_fn: Callable[..., Any],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | np.ndarray | str,
        logger: Any,
    ) -> ExecutionResult:
        """Execute large dataset optimization."""
        logger.debug("Using curve_fit_large for memory-efficient optimization")

        # Use parameter magnitude-based scaling
        if isinstance(x_scale_value, (int, float)):
            x_scale = np.abs(initial_params) + 1e-3
            logger.info(
                f"Replacing scalar x_scale={x_scale_value} with magnitude-based scaling"
            )
        elif isinstance(x_scale_value, np.ndarray):
            x_scale = x_scale_value
        else:
            x_scale = np.abs(initial_params) + 1e-3

        # Convert bounds to list format to avoid numpy array comparison issues in nlsq
        if bounds is not None:
            bounds_for_nlsq = (bounds[0].tolist(), bounds[1].tolist())
        else:
            bounds_for_nlsq = (-np.inf, np.inf)

        try:
            result = curve_fit_large(
                residual_fn,
                xdata,
                ydata,
                p0=initial_params.tolist(),
                bounds=bounds_for_nlsq,
                loss=loss_name,
                x_scale=x_scale,
                gtol=1e-6,
                ftol=1e-6,
                max_nfev=5000,
                verbose=2,
                show_progress=True,
                stability="auto",  # Enable memory management and stability
            )

            # Handle different return formats from curve_fit_large
            if isinstance(result, tuple) and len(result) >= 2:
                popt, pcov = result[0], result[1]
                info = result[2] if len(result) > 2 else {}
            else:
                # OptimizeResult object
                popt = result.x
                pcov = getattr(result, "pcov", np.zeros((len(popt), len(popt))))
                info = {"nfev": getattr(result, "nfev", 0)}

            info["success"] = True
            info["strategy"] = "large"

            return ExecutionResult(
                popt=np.asarray(popt),
                pcov=np.asarray(pcov),
                info=info,
                recovery_actions=[],
                convergence_status="converged",
            )

        except Exception as e:
            logger.error(f"Large dataset optimization failed: {e}")
            raise


class StreamingExecutor(OptimizationExecutor):
    """Streaming optimization for unlimited dataset sizes.

    Uses NLSQ's AdaptiveHybridStreamingOptimizer for datasets that are too large
    to fit in memory. Supports checkpointing and recovery.

    .. note:: The old StreamingOptimizer was removed in NLSQ 0.4.0.
        This executor now uses AdaptiveHybridStreamingOptimizer which provides
        better convergence and parameter estimation.
    """

    def __init__(self, checkpoint_config: dict[str, Any] | None = None):
        """Initialize streaming executor.

        Args:
            checkpoint_config: Configuration for checkpointing and hybrid streaming
        """
        self.checkpoint_config = checkpoint_config or {}

    @property
    def name(self) -> str:
        return "streaming"

    @property
    def supports_progress(self) -> bool:
        return True

    def execute(
        self,
        residual_fn: Callable[..., Any],
        xdata: np.ndarray,
        ydata: np.ndarray,
        initial_params: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray] | None,
        loss_name: str,
        x_scale_value: float | np.ndarray | str,
        logger: Any,
    ) -> ExecutionResult:
        """Execute streaming optimization using AdaptiveHybridStreamingOptimizer."""
        if not STREAMING_AVAILABLE:
            raise RuntimeError(
                "AdaptiveHybridStreamingOptimizer not available. Upgrade NLSQ to >= 0.3.2"
            )

        logger.info("Using NLSQ AdaptiveHybridStreamingOptimizer for large dataset...")

        # Configure hybrid streaming
        config = HybridStreamingConfig(
            chunk_size=self.checkpoint_config.get("chunk_size", 50000),
            warmup_iterations=self.checkpoint_config.get("warmup_iterations", 100),
            max_warmup_iterations=self.checkpoint_config.get(
                "max_warmup_iterations", 500
            ),
            gauss_newton_max_iterations=self.checkpoint_config.get(
                "gauss_newton_max_iterations", 50
            ),
            gauss_newton_tol=self.checkpoint_config.get("gauss_newton_tol", 1e-8),
            normalize=self.checkpoint_config.get("normalize", True),
            normalization_strategy=self.checkpoint_config.get(
                "normalization_strategy", "bounds"
            ),
        )

        optimizer = AdaptiveHybridStreamingOptimizer(config)

        try:
            result = optimizer.fit(
                residual_fn,
                xdata,
                ydata,
                p0=initial_params,
                bounds=bounds,
            )

            info = {
                "success": result.get("success", True),
                "strategy": "hybrid_streaming",
                "iterations": result.get("nit", 0),
                "streaming_diagnostics": result.get("streaming_diagnostics", {}),
            }

            popt = np.asarray(result["x"])
            pcov = result.get("pcov", np.zeros((len(popt), len(popt))))

            return ExecutionResult(
                popt=popt,
                pcov=np.asarray(pcov),
                info=info,
                recovery_actions=[],
                convergence_status="converged" if info["success"] else "partial",
            )

        except Exception as e:
            logger.error(f"Hybrid streaming optimization failed: {e}")
            raise


def get_executor(
    strategy_name: str,
    checkpoint_config: dict[str, Any] | None = None,
) -> OptimizationExecutor:
    """Factory function to get the appropriate executor.

    Args:
        strategy_name: Name of strategy ('standard', 'large', 'streaming')
        checkpoint_config: Configuration for streaming checkpoints

    Returns:
        OptimizationExecutor instance for the strategy

    Raises:
        ValueError: If strategy name is unknown
    """
    executors = {
        "standard": StandardExecutor,
        "large": LargeDatasetExecutor,
        "chunked": LargeDatasetExecutor,  # Same as large
        "streaming": lambda: StreamingExecutor(checkpoint_config),
    }

    if strategy_name not in executors:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. Available: {list(executors.keys())}"
        )

    executor_class = executors[strategy_name]
    if callable(executor_class) and not isinstance(executor_class, type):
        return executor_class()
    return executor_class()
