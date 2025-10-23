"""Dataset Size Strategy Selection for NLSQ Optimization.

This module implements intelligent strategy selection for NLSQ optimization based on
dataset size and available memory. It automatically chooses between:
- STANDARD: curve_fit for <1M points
- LARGE: curve_fit_large for 1M-10M points
- CHUNKED: chunked processing with progress for 10M-100M points
- STREAMING: streaming optimization for >100M points

The strategy selection is based on NLSQ best practices:
https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """NLSQ optimization strategy based on dataset size.

    Attributes
    ----------
    STANDARD : str
        Standard curve_fit for small datasets (<1M points)
    LARGE : str
        curve_fit_large with default settings (1M-10M points)
    CHUNKED : str
        Chunked processing with progress monitoring (10M-100M points)
    STREAMING : str
        Streaming optimization for unlimited data (>100M points)
    """

    STANDARD = "standard"
    LARGE = "large"
    CHUNKED = "chunked"
    STREAMING = "streaming"


class DatasetSizeStrategy:
    """Intelligent strategy selection based on dataset size and memory.

    This class implements the tiered strategy approach recommended by NLSQ:
    - Estimates memory requirements before fitting
    - Selects appropriate optimization strategy
    - Provides memory-based adjustments
    - Supports configuration overrides

    Size Thresholds
    ---------------
    - SMALL: < 1M points → STANDARD strategy (curve_fit)
    - MEDIUM: 1M - 10M points → LARGE strategy (curve_fit_large)
    - LARGE: 10M - 100M points → CHUNKED strategy (with progress)
    - XLARGE: > 100M points → STREAMING strategy (unlimited data)

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with strategy settings

    Attributes
    ----------
    THRESHOLD_SMALL : int
        Upper bound for STANDARD strategy (default: 1_000_000)
    THRESHOLD_MEDIUM : int
        Upper bound for LARGE strategy (default: 10_000_000)
    THRESHOLD_LARGE : int
        Upper bound for CHUNKED strategy (default: 100_000_000)

    Examples
    --------
    >>> strategy_selector = DatasetSizeStrategy()
    >>> strategy = strategy_selector.select_strategy(n_points=5_000_000)
    >>> print(strategy)
    OptimizationStrategy.LARGE

    >>> # With memory-based adjustment
    >>> strategy = strategy_selector.select_strategy(
    ...     n_points=50_000_000,
    ...     n_parameters=9,
    ...     check_memory=True
    ... )
    """

    # Size thresholds (class constants)
    THRESHOLD_SMALL = 1_000_000      # 1M points
    THRESHOLD_MEDIUM = 10_000_000    # 10M points
    THRESHOLD_LARGE = 100_000_000    # 100M points

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize strategy selector.

        Parameters
        ----------
        config : dict, optional
            Configuration with optional overrides:
            - strategy_override: Force specific strategy
            - memory_limit_gb: Custom memory limit
            - enable_progress: Enable/disable progress bars
        """
        self.config = config or {}
        self._override = self.config.get("strategy_override")
        self._memory_limit_gb = self.config.get("memory_limit_gb")
        self._enable_progress = self.config.get("enable_progress", True)

        if self._override:
            logger.info(f"Strategy override enabled: {self._override}")

    def select_strategy(
        self,
        n_points: int,
        n_parameters: int = 9,
        check_memory: bool = True,
    ) -> OptimizationStrategy:
        """Select optimal NLSQ strategy based on dataset size and memory.

        Parameters
        ----------
        n_points : int
            Total number of data points
        n_parameters : int, optional
            Number of parameters to optimize (default: 9 for laminar_flow)
        check_memory : bool, optional
            Whether to check available memory (default: True)

        Returns
        -------
        OptimizationStrategy
            Selected optimization strategy

        Notes
        -----
        If memory check is enabled and estimated memory usage exceeds available
        memory, the strategy may be adjusted to a more memory-efficient approach.

        Examples
        --------
        >>> selector = DatasetSizeStrategy()
        >>> strategy = selector.select_strategy(500_000)
        >>> print(strategy)
        OptimizationStrategy.STANDARD
        """
        # Check for override
        if self._override:
            try:
                return OptimizationStrategy(self._override)
            except ValueError:
                logger.warning(
                    f"Invalid strategy override '{self._override}', "
                    f"using automatic selection"
                )

        # Size-based selection
        if n_points < self.THRESHOLD_SMALL:
            strategy = OptimizationStrategy.STANDARD
            logger.debug(
                f"Selected STANDARD strategy for {n_points:,} points "
                f"(< {self.THRESHOLD_SMALL:,})"
            )
        elif n_points < self.THRESHOLD_MEDIUM:
            strategy = OptimizationStrategy.LARGE
            logger.debug(
                f"Selected LARGE strategy for {n_points:,} points "
                f"({self.THRESHOLD_SMALL:,} - {self.THRESHOLD_MEDIUM:,})"
            )
        elif n_points < self.THRESHOLD_LARGE:
            strategy = OptimizationStrategy.CHUNKED
            logger.debug(
                f"Selected CHUNKED strategy for {n_points:,} points "
                f"({self.THRESHOLD_MEDIUM:,} - {self.THRESHOLD_LARGE:,})"
            )
        else:
            strategy = OptimizationStrategy.STREAMING
            logger.debug(
                f"Selected STREAMING strategy for {n_points:,} points "
                f"(> {self.THRESHOLD_LARGE:,})"
            )

        # Memory-based adjustment
        if check_memory:
            strategy = self._adjust_for_memory(
                strategy, n_points, n_parameters
            )

        return strategy

    def _adjust_for_memory(
        self,
        strategy: OptimizationStrategy,
        n_points: int,
        n_parameters: int,
    ) -> OptimizationStrategy:
        """Adjust strategy based on available memory.

        Parameters
        ----------
        strategy : OptimizationStrategy
            Initially selected strategy
        n_points : int
            Number of data points
        n_parameters : int
            Number of parameters

        Returns
        -------
        OptimizationStrategy
            Potentially adjusted strategy
        """
        try:
            # Estimate memory requirements
            estimated_gb = self._estimate_memory_gb(n_points, n_parameters)
            available_gb = self._get_available_memory_gb()

            logger.debug(
                f"Memory estimate: {estimated_gb:.2f} GB required, "
                f"{available_gb:.2f} GB available"
            )

            # Check if we need to upgrade to more memory-efficient strategy
            if estimated_gb > available_gb * 0.7:  # 70% safety margin
                logger.warning(
                    f"Estimated memory ({estimated_gb:.2f} GB) exceeds "
                    f"70% of available ({available_gb:.2f} GB), "
                    f"considering strategy adjustment"
                )

                # Upgrade to more memory-efficient strategy
                if strategy == OptimizationStrategy.STANDARD:
                    logger.info("Upgrading STANDARD → LARGE for memory efficiency")
                    return OptimizationStrategy.LARGE
                elif strategy == OptimizationStrategy.LARGE:
                    logger.info("Upgrading LARGE → CHUNKED for memory efficiency")
                    return OptimizationStrategy.CHUNKED
                elif strategy == OptimizationStrategy.CHUNKED:
                    logger.info("Upgrading CHUNKED → STREAMING for memory efficiency")
                    return OptimizationStrategy.STREAMING

        except Exception as e:
            logger.warning(f"Failed to adjust strategy for memory: {e}")

        return strategy

    def _estimate_memory_gb(self, n_points: int, n_parameters: int) -> float:
        """Estimate memory requirements in GB.

        Parameters
        ----------
        n_points : int
            Number of data points
        n_parameters : int
            Number of parameters

        Returns
        -------
        float
            Estimated memory in GB

        Notes
        -----
        This is a conservative estimate based on:
        - Data storage (float64)
        - Jacobian matrix (n_points × n_parameters)
        - Hessian approximation (n_parameters × n_parameters)
        - Temporary arrays and overhead (2x multiplier)
        """
        bytes_per_float = 8  # float64

        # Data arrays
        data_memory = n_points * bytes_per_float

        # Jacobian matrix
        jacobian_memory = n_points * n_parameters * bytes_per_float

        # Hessian and other parameter-space arrays
        hessian_memory = n_parameters * n_parameters * bytes_per_float

        # Total with 2x overhead for temporaries
        total_bytes = (data_memory + jacobian_memory + hessian_memory) * 2

        return total_bytes / (1024**3)  # Convert to GB

    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB.

        Returns
        -------
        float
            Available memory in GB

        Notes
        -----
        Uses psutil to query system memory. Returns available memory,
        which is memory that can be given instantly to processes without
        swapping to disk.
        """
        if self._memory_limit_gb is not None:
            return self._memory_limit_gb

        memory = psutil.virtual_memory()
        return memory.available / (1024**3)

    def build_streaming_config(
        self,
        n_points: int,
        n_parameters: int = 9,
        checkpoint_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Build optimized StreamingOptimizer configuration.

        Creates configuration for NLSQ StreamingOptimizer with optimal batch size,
        checkpoint settings, and fault tolerance based on dataset size and available memory.

        Parameters
        ----------
        n_points : int
            Total number of data points
        n_parameters : int, optional
            Number of parameters to optimize (default: 9)
        checkpoint_config : dict, optional
            Checkpoint configuration override. Keys:
            - enable_checkpoints: bool
            - checkpoint_dir: str
            - checkpoint_frequency: int
            - resume_from_checkpoint: bool
            - keep_last_checkpoints: int
            - enable_fault_tolerance: bool
            - max_retries_per_batch: int
            - min_success_rate: float

        Returns
        -------
        dict
            StreamingOptimizer configuration with keys:
            - batch_size: Optimal batch size based on memory
            - max_epochs: Number of epochs (default: 10)
            - enable_checkpoints: Whether to enable checkpointing
            - checkpoint_dir: Directory for checkpoints
            - checkpoint_frequency: Save every N batches
            - resume_from_checkpoint: Auto-resume from latest
            - enable_fault_tolerance: Enable validation and retry
            - validate_numerics: Enable NaN/Inf checks
            - min_success_rate: Minimum batch success rate
            - max_retries_per_batch: Maximum retry attempts

        Examples
        --------
        >>> selector = DatasetSizeStrategy()
        >>> config = selector.build_streaming_config(
        ...     n_points=200_000_000,
        ...     n_parameters=9
        ... )
        >>> print(config['batch_size'])
        10000
        """
        checkpoint_config = checkpoint_config or {}

        # Calculate optimal batch size based on available memory
        available_gb = self._get_available_memory_gb()
        batch_size = self._calculate_optimal_batch_size(
            available_gb, n_parameters
        )

        # Extract checkpoint settings with defaults
        enable_checkpoints = checkpoint_config.get('enable_checkpoints', False)
        checkpoint_dir = checkpoint_config.get('checkpoint_dir', './checkpoints')
        checkpoint_frequency = checkpoint_config.get('checkpoint_frequency', 10)
        resume_from_checkpoint = checkpoint_config.get('resume_from_checkpoint', True)
        keep_last_n = checkpoint_config.get('keep_last_checkpoints', 3)

        # Extract fault tolerance settings
        enable_fault_tolerance = checkpoint_config.get('enable_fault_tolerance', True)
        max_retries_per_batch = checkpoint_config.get('max_retries_per_batch', 2)
        min_success_rate = checkpoint_config.get('min_success_rate', 0.5)

        logger.info(
            f"Building streaming config for {n_points:,} points: "
            f"batch_size={batch_size:,}, checkpoints={'enabled' if enable_checkpoints else 'disabled'}"
        )

        return {
            # Batch processing
            "batch_size": batch_size,
            "max_epochs": 10,

            # Checkpoint management
            "enable_checkpoints": enable_checkpoints,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint_frequency": checkpoint_frequency if enable_checkpoints else 0,
            "resume_from_checkpoint": resume_from_checkpoint,
            "keep_last_checkpoints": keep_last_n,

            # Fault tolerance
            "enable_fault_tolerance": enable_fault_tolerance,
            "validate_numerics": enable_fault_tolerance,
            "min_success_rate": min_success_rate,
            "max_retries_per_batch": max_retries_per_batch,
        }

    def _calculate_optimal_batch_size(
        self,
        available_memory_gb: float,
        n_parameters: int,
    ) -> int:
        """Calculate optimal batch size based on available memory.

        Parameters
        ----------
        available_memory_gb : float
            Available system memory in GB
        n_parameters : int
            Number of parameters to optimize

        Returns
        -------
        int
            Optimal batch size (bounded between 1,000 and 100,000)

        Notes
        -----
        Batch size is calculated to use ~10% of available memory to leave
        headroom for other operations. Typical batch sizes:
        - 1 GB available → 10,000 points
        - 8 GB available → 50,000 points
        - 32 GB available → 100,000 points (capped)
        """
        bytes_per_float = 8  # float64
        target_memory_gb = available_memory_gb * 0.1  # Use 10% of available

        # Estimate points per GB: data + Jacobian
        memory_per_point = bytes_per_float * (1 + n_parameters)
        target_bytes = target_memory_gb * (1024**3)
        batch_size = int(target_bytes / memory_per_point)

        # Bound between min and max
        MIN_BATCH_SIZE = 1_000
        MAX_BATCH_SIZE = 100_000
        batch_size = max(MIN_BATCH_SIZE, min(batch_size, MAX_BATCH_SIZE))

        # Round to nearest 1000 for cleaner numbers
        batch_size = round(batch_size / 1000) * 1000

        logger.debug(
            f"Calculated batch size: {batch_size:,} points "
            f"(available memory: {available_memory_gb:.1f} GB, "
            f"parameters: {n_parameters})"
        )

        return batch_size

    def get_strategy_info(self, strategy: OptimizationStrategy) -> dict[str, Any]:
        """Get information about a specific strategy.

        Parameters
        ----------
        strategy : OptimizationStrategy
            Strategy to query

        Returns
        -------
        dict
            Strategy information including:
            - name: Strategy name
            - description: Human-readable description
            - use_case: When to use this strategy
            - nlsq_function: Corresponding NLSQ function name
            - supports_progress: Whether progress bars are supported
        """
        info = {
            OptimizationStrategy.STANDARD: {
                "name": "Standard",
                "description": "Standard curve_fit for small datasets",
                "use_case": "< 1M points, fast convergence",
                "nlsq_function": "curve_fit",
                "supports_progress": False,
            },
            OptimizationStrategy.LARGE: {
                "name": "Large",
                "description": "curve_fit_large with default settings",
                "use_case": "1M - 10M points, memory-efficient",
                "nlsq_function": "curve_fit_large",
                "supports_progress": True,
            },
            OptimizationStrategy.CHUNKED: {
                "name": "Chunked",
                "description": "Chunked processing with progress monitoring",
                "use_case": "10M - 100M points, memory-safe with progress",
                "nlsq_function": "curve_fit_large (chunked)",
                "supports_progress": True,
            },
            OptimizationStrategy.STREAMING: {
                "name": "Streaming",
                "description": "Streaming optimization for unlimited data",
                "use_case": "> 100M points, processes 100% of data",
                "nlsq_function": "fit_streaming",
                "supports_progress": True,
            },
        }

        return info[strategy]


def estimate_memory_requirements(
    n_points: int,
    n_parameters: int,
) -> dict[str, Any]:
    """Estimate memory requirements for NLSQ optimization.

    This is a convenience function that wraps NLSQ's estimate_memory_requirements
    with additional XPCS-specific context.

    Parameters
    ----------
    n_points : int
        Total number of data points
    n_parameters : int
        Number of parameters to optimize

    Returns
    -------
    dict
        Memory statistics including:
        - total_memory_estimate_gb: Total estimated memory usage
        - available_memory_gb: Available system memory
        - memory_safe: Whether optimization is safe to proceed
        - recommended_strategy: Suggested optimization strategy

    Examples
    --------
    >>> stats = estimate_memory_requirements(5_000_000, 9)
    >>> print(f"Estimated memory: {stats['total_memory_estimate_gb']:.2f} GB")
    >>> print(f"Recommended: {stats['recommended_strategy']}")
    """
    selector = DatasetSizeStrategy()

    estimated_gb = selector._estimate_memory_gb(n_points, n_parameters)
    available_gb = selector._get_available_memory_gb()
    memory_safe = estimated_gb < available_gb * 0.7

    # Get recommended strategy
    strategy = selector.select_strategy(
        n_points=n_points,
        n_parameters=n_parameters,
        check_memory=True,
    )

    return {
        "total_memory_estimate_gb": estimated_gb,
        "available_memory_gb": available_gb,
        "memory_safe": memory_safe,
        "recommended_strategy": strategy.value,
        "strategy_info": selector.get_strategy_info(strategy),
    }
