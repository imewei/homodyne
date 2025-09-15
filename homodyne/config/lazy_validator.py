"""
Lazy Validation System for Homodyne v2 Configuration
===================================================

Intelligent lazy validation system that defers expensive validation checks
until they are actually needed, with priority-based ordering and user-configurable
validation levels.

Key Features:
- Deferred validation with intelligent triggering
- Priority-based validation ordering (critical → important → optional)
- User-configurable validation levels (fast, standard, thorough)
- Async validation with progress tracking
- Validation result caching and memoization
- Memory-efficient validation pipelines
"""

import asyncio
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import (Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple,
                    Union)

from homodyne.config.parameter_validator import (ParameterValidator,
                                                 ValidationResult)
from homodyne.config.performance_cache import get_validation_cache
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationPriority(Enum):
    """Validation priority levels."""

    CRITICAL = 1  # Must be validated before any processing
    HIGH = 2  # Should be validated early
    MEDIUM = 3  # Can be deferred until needed
    LOW = 4  # Optional validation
    BACKGROUND = 5  # Can run in background


class ValidationLevel(Enum):
    """Validation thoroughness levels."""

    FAST = "fast"  # Essential checks only
    STANDARD = "standard"  # Standard validation suite
    THOROUGH = "thorough"  # Comprehensive validation
    EXHAUSTIVE = "exhaustive"  # All possible checks


@dataclass
class ValidationTask:
    """
    Individual validation task with metadata.

    Attributes:
        name: Task identifier
        validator: Validation function
        priority: Task priority level
        dependencies: List of task names this depends on
        timeout_seconds: Maximum execution time
        cache_key: Optional cache key for results
        description: Human-readable description
        estimated_time_ms: Estimated execution time
        memory_requirement_mb: Estimated memory requirement
    """

    name: str
    validator: Callable[[Dict[str, Any]], ValidationResult]
    priority: ValidationPriority
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: float = 30.0
    cache_key: Optional[str] = None
    description: str = ""
    estimated_time_ms: float = 100.0
    memory_requirement_mb: float = 10.0

    def __post_init__(self):
        if not self.description:
            self.description = f"Validation task: {self.name}"


@dataclass
class ValidationProgress:
    """
    Validation progress tracking.

    Attributes:
        total_tasks: Total number of validation tasks
        completed_tasks: Number of completed tasks
        current_task: Currently executing task name
        elapsed_time_ms: Total elapsed time
        estimated_remaining_ms: Estimated remaining time
        errors_count: Number of validation errors
        warnings_count: Number of validation warnings
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    current_task: str = ""
    elapsed_time_ms: float = 0.0
    estimated_remaining_ms: float = 0.0
    errors_count: int = 0
    warnings_count: int = 0

    @property
    def progress_fraction(self) -> float:
        """Get progress as fraction (0.0 to 1.0)."""
        return self.completed_tasks / max(1, self.total_tasks)

    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        return self.progress_fraction * 100.0


class LazyValidator:
    """
    Lazy validation system with intelligent task scheduling.

    Manages validation tasks with priority-based execution, caching,
    and resource-aware scheduling.
    """

    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        max_concurrent_tasks: int = 4,
        enable_caching: bool = True,
    ):
        """
        Initialize lazy validator.

        Args:
            validation_level: Default validation thoroughness
            max_concurrent_tasks: Maximum concurrent validation tasks
            enable_caching: Enable validation result caching
        """
        self.validation_level = validation_level
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_caching = enable_caching

        # Task management
        self._registered_tasks: Dict[str, ValidationTask] = {}
        self._task_results: Dict[str, ValidationResult] = {}
        self._task_completion_times: Dict[str, float] = {}

        # Progress tracking
        self._progress_callbacks: List[Callable[[ValidationProgress], None]] = []
        self._current_progress = ValidationProgress()

        # Caching
        if enable_caching:
            self._cache = get_validation_cache()
        else:
            self._cache = None

        # Resource tracking
        self._active_tasks: set = set()
        self._task_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

        logger.debug(
            f"Lazy validator initialized: level={validation_level.value}, "
            f"concurrent_tasks={max_concurrent_tasks}, caching={'enabled' if enable_caching else 'disabled'}"
        )

        # Register built-in validation tasks
        self._register_builtin_tasks()

    def register_task(self, task: ValidationTask) -> None:
        """
        Register a validation task.

        Args:
            task: Validation task to register
        """
        self._registered_tasks[task.name] = task
        logger.debug(
            f"Registered validation task: {task.name} (priority: {task.priority.name})"
        )

    def set_validation_level(self, level: ValidationLevel) -> None:
        """
        Set validation thoroughness level.

        Args:
            level: New validation level
        """
        self.validation_level = level
        logger.info(f"Validation level set to: {level.value}")

    def add_progress_callback(
        self, callback: Callable[[ValidationProgress], None]
    ) -> None:
        """
        Add progress callback function.

        Args:
            callback: Function to call with validation progress
        """
        self._progress_callbacks.append(callback)

    async def validate_async(
        self,
        config: Dict[str, Any],
        required_tasks: Optional[List[str]] = None,
        skip_optional: bool = False,
    ) -> ValidationResult:
        """
        Perform lazy validation asynchronously.

        Args:
            config: Configuration to validate
            required_tasks: Specific tasks to run (None for auto-selection)
            skip_optional: Skip optional validation tasks

        Returns:
            Aggregated validation result
        """
        start_time = time.perf_counter()
        logger.info(
            f"Starting async lazy validation (level: {self.validation_level.value})"
        )

        # Select tasks to run
        selected_tasks = self._select_tasks_for_level(required_tasks, skip_optional)

        # Initialize progress tracking
        self._current_progress = ValidationProgress(
            total_tasks=len(selected_tasks), elapsed_time_ms=0.0
        )
        self._notify_progress()

        # Execute tasks with dependency resolution
        task_results = await self._execute_tasks_with_dependencies(
            config, selected_tasks
        )

        # Aggregate results
        final_result = self._aggregate_results(task_results)

        # Update timing
        total_time = (time.perf_counter() - start_time) * 1000
        final_result.validation_time_ms = total_time

        self._current_progress.elapsed_time_ms = total_time
        self._current_progress.completed_tasks = len(selected_tasks)
        self._notify_progress()

        logger.info(
            f"Lazy validation completed: {len(selected_tasks)} tasks, "
            f"{final_result.errors_count} errors, {final_result.warnings_count} warnings, "
            f"{total_time:.1f}ms"
        )

        return final_result

    def validate_sync(
        self,
        config: Dict[str, Any],
        required_tasks: Optional[List[str]] = None,
        skip_optional: bool = False,
    ) -> ValidationResult:
        """
        Perform lazy validation synchronously.

        Args:
            config: Configuration to validate
            required_tasks: Specific tasks to run
            skip_optional: Skip optional validation tasks

        Returns:
            Aggregated validation result
        """
        # Run async validation in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.validate_async(config, required_tasks, skip_optional)
        )

    def get_task_estimate(self, tasks: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Get time estimates for validation tasks.

        Args:
            tasks: Specific tasks to estimate (None for auto-selection)

        Returns:
            Dictionary with time estimates
        """
        if tasks is None:
            tasks = list(self._registered_tasks.keys())

        estimates = {}
        for task_name in tasks:
            if task_name in self._registered_tasks:
                task = self._registered_tasks[task_name]

                # Use historical data if available
                if task_name in self._task_completion_times:
                    estimates[task_name] = self._task_completion_times[task_name]
                else:
                    estimates[task_name] = task.estimated_time_ms

        return estimates

    def validate_on_demand(
        self, config: Dict[str, Any], task_name: str
    ) -> ValidationResult:
        """
        Validate specific task on demand.

        Args:
            config: Configuration to validate
            task_name: Name of task to execute

        Returns:
            Validation result for the specific task
        """
        if task_name not in self._registered_tasks:
            return ValidationResult(
                is_valid=False,
                errors=[f"Unknown validation task: {task_name}"],
                warnings=[],
                suggestions=[],
                info=[],
                hardware_info={},
            )

        # Check cache first
        if self._cache:
            cache_key = f"{task_name}:{hash(str(config))}"
            cached_result, cache_hit = self._cache.get_validation_result(
                {"task": task_name, "config": config}
            )
            if cache_hit and cached_result:
                logger.debug(f"Using cached result for task: {task_name}")
                return cached_result

        # Execute task
        task = self._registered_tasks[task_name]
        start_time = time.perf_counter()

        try:
            logger.debug(f"Executing on-demand validation: {task_name}")
            result = task.validator(config)

            execution_time = (time.perf_counter() - start_time) * 1000
            self._task_completion_times[task_name] = execution_time

            # Cache result if enabled
            if self._cache:
                self._cache.cache_validation_result(
                    {"task": task_name, "config": config}, result.__dict__
                )

            logger.debug(f"Task {task_name} completed in {execution_time:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Task {task_name} failed: {str(e)}"],
                warnings=[],
                suggestions=[],
                info=[],
                hardware_info={},
            )

    def _select_tasks_for_level(
        self, required_tasks: Optional[List[str]] = None, skip_optional: bool = False
    ) -> List[str]:
        """Select tasks based on validation level and requirements."""
        if required_tasks is not None:
            return [task for task in required_tasks if task in self._registered_tasks]

        selected = []

        # Priority-based selection based on validation level
        priority_thresholds = {
            ValidationLevel.FAST: [ValidationPriority.CRITICAL],
            ValidationLevel.STANDARD: [
                ValidationPriority.CRITICAL,
                ValidationPriority.HIGH,
            ],
            ValidationLevel.THOROUGH: [
                ValidationPriority.CRITICAL,
                ValidationPriority.HIGH,
                ValidationPriority.MEDIUM,
            ],
            ValidationLevel.EXHAUSTIVE: [
                ValidationPriority.CRITICAL,
                ValidationPriority.HIGH,
                ValidationPriority.MEDIUM,
                ValidationPriority.LOW,
            ],
        }

        allowed_priorities = priority_thresholds.get(
            self.validation_level, [ValidationPriority.CRITICAL]
        )

        for task_name, task in self._registered_tasks.items():
            if task.priority in allowed_priorities:
                if skip_optional and task.priority == ValidationPriority.LOW:
                    continue
                selected.append(task_name)

        logger.debug(
            f"Selected {len(selected)} tasks for level {self.validation_level.value}"
        )
        return selected

    async def _execute_tasks_with_dependencies(
        self, config: Dict[str, Any], task_names: List[str]
    ) -> Dict[str, ValidationResult]:
        """Execute tasks respecting dependencies."""
        results = {}
        remaining_tasks = set(task_names)
        executing_tasks = set()

        while remaining_tasks or executing_tasks:
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = []
            for task_name in remaining_tasks:
                task = self._registered_tasks[task_name]
                if all(dep in results for dep in task.dependencies):
                    ready_tasks.append(task_name)

            # Limit concurrent tasks
            available_slots = self.max_concurrent_tasks - len(executing_tasks)
            tasks_to_start = ready_tasks[:available_slots]

            # Start new tasks
            task_futures = {}
            for task_name in tasks_to_start:
                future = asyncio.create_task(
                    self._execute_task_async(config, task_name)
                )
                task_futures[future] = task_name
                executing_tasks.add(task_name)
                remaining_tasks.remove(task_name)

                # Update progress
                self._current_progress.current_task = task_name
                self._notify_progress()

            # Wait for at least one task to complete
            if task_futures:
                done, pending = await asyncio.wait(
                    task_futures.keys(), return_when=asyncio.FIRST_COMPLETED
                )

                for future in done:
                    task_name = task_futures[future]
                    try:
                        result = await future
                        results[task_name] = result

                        # Update progress
                        self._current_progress.completed_tasks += 1
                        if not result.is_valid:
                            self._current_progress.errors_count += len(result.errors)
                            self._current_progress.warnings_count += len(
                                result.warnings
                            )

                    except Exception as e:
                        logger.error(f"Task {task_name} failed: {e}")
                        results[task_name] = ValidationResult(
                            is_valid=False,
                            errors=[f"Task execution failed: {str(e)}"],
                            warnings=[],
                            suggestions=[],
                            info=[],
                            hardware_info={},
                        )
                        self._current_progress.errors_count += 1

                    executing_tasks.remove(task_name)
                    self._notify_progress()
            else:
                # No tasks ready and none executing - check for circular dependencies
                if remaining_tasks:
                    logger.error(
                        f"Circular dependency detected in tasks: {remaining_tasks}"
                    )
                    # Execute remaining tasks without dependency checking
                    for task_name in remaining_tasks:
                        result = await self._execute_task_async(config, task_name)
                        results[task_name] = result
                    break

        return results

    async def _execute_task_async(
        self, config: Dict[str, Any], task_name: str
    ) -> ValidationResult:
        """Execute a single task asynchronously."""
        task = self._registered_tasks[task_name]

        # Check cache first
        if self._cache:
            cache_key = f"{task_name}:{hash(str(config))}"
            cached_result, cache_hit = self._cache.get_validation_result(
                {"task": task_name, "config": config}
            )
            if cache_hit and cached_result:
                logger.debug(f"Using cached result for task: {task_name}")
                return ValidationResult(**cached_result)

        # Execute task in thread pool
        start_time = time.perf_counter()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._task_executor, task.validator, config
            )

            execution_time = (time.perf_counter() - start_time) * 1000
            self._task_completion_times[task_name] = execution_time

            # Cache result if enabled
            if self._cache:
                self._cache.cache_validation_result(
                    {"task": task_name, "config": config}, result.__dict__
                )

            logger.debug(f"Task {task_name} completed in {execution_time:.1f}ms")
            return result

        except Exception as e:
            logger.error(f"Task {task_name} failed: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Task {task_name} failed: {str(e)}"],
                warnings=[],
                suggestions=[],
                info=[],
                hardware_info={},
            )

    def _aggregate_results(
        self, task_results: Dict[str, ValidationResult]
    ) -> ValidationResult:
        """Aggregate multiple validation results."""
        all_errors = []
        all_warnings = []
        all_suggestions = []
        all_info = []
        combined_hardware_info = {}

        is_valid = True

        for task_name, result in task_results.items():
            if not result.is_valid:
                is_valid = False

            # Add task context to messages
            all_errors.extend([f"[{task_name}] {error}" for error in result.errors])
            all_warnings.extend(
                [f"[{task_name}] {warning}" for warning in result.warnings]
            )
            all_suggestions.extend(
                [f"[{task_name}] {suggestion}" for suggestion in result.suggestions]
            )
            all_info.extend([f"[{task_name}] {info}" for info in result.info])

            # Merge hardware info
            if hasattr(result, "hardware_info") and result.hardware_info:
                combined_hardware_info.update(result.hardware_info)

        aggregated = ValidationResult(
            is_valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            suggestions=all_suggestions,
            info=all_info,
            hardware_info=combined_hardware_info,
        )

        # Add summary statistics
        aggregated.errors_count = len(all_errors)
        aggregated.warnings_count = len(all_warnings)
        aggregated.tasks_executed = len(task_results)

        return aggregated

    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self._current_progress)
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")

    def _register_builtin_tasks(self) -> None:
        """Register built-in validation tasks."""

        # Critical validation tasks
        self.register_task(
            ValidationTask(
                name="config_structure",
                validator=self._validate_config_structure,
                priority=ValidationPriority.CRITICAL,
                description="Validate basic configuration structure",
                estimated_time_ms=10.0,
            )
        )

        self.register_task(
            ValidationTask(
                name="analysis_mode",
                validator=self._validate_analysis_mode,
                priority=ValidationPriority.CRITICAL,
                description="Validate analysis mode specification",
                estimated_time_ms=5.0,
            )
        )

        # High priority tasks
        self.register_task(
            ValidationTask(
                name="optimization_params",
                validator=self._validate_optimization_params,
                priority=ValidationPriority.HIGH,
                dependencies=["config_structure"],
                description="Validate optimization parameters",
                estimated_time_ms=50.0,
            )
        )

        self.register_task(
            ValidationTask(
                name="hardware_compatibility",
                validator=self._validate_hardware_compatibility,
                priority=ValidationPriority.HIGH,
                description="Validate hardware compatibility",
                estimated_time_ms=100.0,
            )
        )

        # Medium priority tasks
        self.register_task(
            ValidationTask(
                name="parameter_bounds",
                validator=self._validate_parameter_bounds,
                priority=ValidationPriority.MEDIUM,
                dependencies=["analysis_mode"],
                description="Validate parameter bounds and constraints",
                estimated_time_ms=30.0,
            )
        )

        self.register_task(
            ValidationTask(
                name="data_consistency",
                validator=self._validate_data_consistency,
                priority=ValidationPriority.MEDIUM,
                description="Validate data file consistency",
                estimated_time_ms=200.0,
            )
        )

        # Low priority / optional tasks
        self.register_task(
            ValidationTask(
                name="performance_optimization",
                validator=self._validate_performance_settings,
                priority=ValidationPriority.LOW,
                description="Validate performance optimization settings",
                estimated_time_ms=20.0,
            )
        )

        self.register_task(
            ValidationTask(
                name="advanced_features",
                validator=self._validate_advanced_features,
                priority=ValidationPriority.LOW,
                dependencies=["config_structure", "analysis_mode"],
                description="Validate advanced configuration features",
                estimated_time_ms=75.0,
            )
        )

    # Built-in validation task implementations
    def _validate_config_structure(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate basic configuration structure."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        # Check required sections
        required_sections = ["analysis_mode"]
        for section in required_sections:
            if section not in config:
                result.errors.append(f"Missing required section: {section}")
                result.is_valid = False

        # Check configuration format
        if not isinstance(config, dict):
            result.errors.append("Configuration must be a dictionary")
            result.is_valid = False

        result.info.append("Basic configuration structure validation completed")
        return result

    def _validate_analysis_mode(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate analysis mode specification."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        mode = config.get("analysis_mode")
        valid_modes = [
            "static_isotropic",
            "static_anisotropic",
            "laminar_flow",
            "auto-detect",
        ]

        if not mode:
            result.errors.append("analysis_mode is required")
            result.is_valid = False
        elif mode not in valid_modes:
            result.errors.append(
                f"Invalid analysis_mode '{mode}'. Valid options: {valid_modes}"
            )
            result.is_valid = False

        if mode == "auto-detect":
            result.suggestions.append(
                "Consider specifying explicit mode for reproducible analysis"
            )

        result.info.append(f"Analysis mode validation completed: {mode}")
        return result

    def _validate_optimization_params(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate optimization parameters."""
        # Delegate to existing parameter validator
        validator = ParameterValidator()
        return validator.validate_config(config)

    def _validate_hardware_compatibility(
        self, config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate hardware compatibility."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        hardware_config = config.get("hardware", {})

        # Check GPU settings
        if "gpu_memory_fraction" in hardware_config:
            gpu_memory = hardware_config["gpu_memory_fraction"]
            if not isinstance(gpu_memory, (int, float)) or not (0 < gpu_memory <= 1):
                result.errors.append("gpu_memory_fraction must be between 0 and 1")
                result.is_valid = False

        result.info.append("Hardware compatibility validation completed")
        return result

    def _validate_parameter_bounds(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate parameter bounds."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        param_space = config.get("parameter_space", {})
        bounds = param_space.get("bounds", [])

        for i, bound in enumerate(bounds):
            if not isinstance(bound, dict):
                result.errors.append(f"Parameter bound {i} must be a dictionary")
                result.is_valid = False
                continue

            required_fields = ["name", "min", "max"]
            for field in required_fields:
                if field not in bound:
                    result.errors.append(
                        f"Parameter bound {i} missing required field: {field}"
                    )
                    result.is_valid = False

        result.info.append(
            f"Parameter bounds validation completed: {len(bounds)} bounds checked"
        )
        return result

    def _validate_data_consistency(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate data file consistency."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        data_config = config.get("experimental_data", {})
        data_path = data_config.get("data_file_path")

        if data_path:
            from pathlib import Path

            if not Path(data_path).exists():
                result.warnings.append(f"Data file not found: {data_path}")

        result.info.append("Data consistency validation completed")
        return result

    def _validate_performance_settings(
        self, config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate performance settings."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        perf_config = config.get("performance_settings", {})

        # Check thread count
        num_threads = perf_config.get("num_threads")
        if num_threads is not None:
            import psutil

            max_threads = psutil.cpu_count()
            if num_threads > max_threads:
                result.warnings.append(
                    f"num_threads ({num_threads}) exceeds CPU count ({max_threads})"
                )

        result.info.append("Performance settings validation completed")
        return result

    def _validate_advanced_features(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate advanced configuration features."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            info=[],
            hardware_info={},
        )

        v2_features = config.get("v2_features", {})

        # Check cache strategy
        cache_strategy = v2_features.get("cache_strategy")
        if cache_strategy and cache_strategy not in [
            "intelligent",
            "aggressive",
            "conservative",
            "disabled",
        ]:
            result.warnings.append(f"Unknown cache strategy: {cache_strategy}")

        result.info.append("Advanced features validation completed")
        return result


# Global lazy validator instance
_global_lazy_validator: Optional[LazyValidator] = None


def get_lazy_validator(
    validation_level: Optional[ValidationLevel] = None,
) -> LazyValidator:
    """Get global lazy validator instance."""
    global _global_lazy_validator

    if _global_lazy_validator is None:
        if validation_level is None:
            validation_level = ValidationLevel.STANDARD

        _global_lazy_validator = LazyValidator(
            validation_level=validation_level,
            max_concurrent_tasks=4,
            enable_caching=True,
        )

        logger.info(
            f"Global lazy validator initialized: level={validation_level.value}"
        )

    return _global_lazy_validator


def set_global_validation_level(level: ValidationLevel) -> None:
    """Set global validation level."""
    validator = get_lazy_validator()
    validator.set_validation_level(level)
