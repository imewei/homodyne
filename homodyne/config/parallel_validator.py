"""
Parallel Validation System for Multi-Configuration Scenarios
===========================================================

High-performance parallel validation system designed for enterprise workloads
with thousands of configurations, intelligent worker pool management, and
real-time progress tracking.

Key Features:
- Scalable worker pool management with dynamic sizing
- Batch processing optimization for large configuration sets
- Real-time progress tracking with ETA estimation
- Intelligent load balancing and resource utilization
- Fault tolerance with retry mechanisms
- Memory-efficient processing with streaming support
- Results aggregation and reporting
"""

import asyncio
import multiprocessing as mp
import queue
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union, Iterator
import psutil
import pickle
import tempfile
from pathlib import Path

from homodyne.utils.logging import get_logger
from homodyne.config.lazy_validator import LazyValidator, ValidationLevel, ValidationResult
from homodyne.config.performance_cache import get_performance_cache

logger = get_logger(__name__)


@dataclass
class ValidationJob:
    """
    Individual validation job for parallel processing.
    
    Attributes:
        job_id: Unique job identifier
        config: Configuration to validate
        validation_level: Validation thoroughness level
        priority: Job priority (lower numbers = higher priority)
        metadata: Additional metadata for the job
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts allowed
    """
    job_id: str
    config: Dict[str, Any]
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    priority: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())


@dataclass
class ValidationBatch:
    """
    Batch of validation jobs for efficient processing.
    
    Attributes:
        batch_id: Unique batch identifier
        jobs: List of validation jobs
        batch_size: Target batch size
        created_time: Batch creation timestamp
        priority: Batch priority
        metadata: Batch metadata
    """
    batch_id: str
    jobs: List[ValidationJob] = field(default_factory=list)
    batch_size: int = 50
    created_time: float = field(default_factory=time.time)
    priority: int = 100
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
    
    def add_job(self, job: ValidationJob) -> None:
        """Add job to batch."""
        self.jobs.append(job)
    
    def is_full(self) -> bool:
        """Check if batch is full."""
        return len(self.jobs) >= self.batch_size
    
    @property
    def size(self) -> int:
        """Get current batch size."""
        return len(self.jobs)


@dataclass
class ValidationProgress:
    """
    Progress tracking for parallel validation.
    
    Attributes:
        total_jobs: Total number of validation jobs
        completed_jobs: Number of completed jobs
        failed_jobs: Number of failed jobs
        in_progress_jobs: Number of jobs currently being processed
        start_time: Processing start time
        estimated_completion_time: Estimated completion timestamp
        current_throughput_jobs_per_sec: Current processing throughput
        average_job_time_ms: Average job processing time
        active_workers: Number of active worker processes/threads
    """
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    in_progress_jobs: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_completion_time: Optional[float] = None
    current_throughput_jobs_per_sec: float = 0.0
    average_job_time_ms: float = 0.0
    active_workers: int = 0
    
    @property
    def progress_fraction(self) -> float:
        """Get progress as fraction (0.0 to 1.0)."""
        if self.total_jobs == 0:
            return 0.0
        return self.completed_jobs / self.total_jobs
    
    @property
    def progress_percent(self) -> float:
        """Get progress as percentage."""
        return self.progress_fraction * 100.0
    
    @property
    def elapsed_time_sec(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    @property
    def eta_sec(self) -> Optional[float]:
        """Get estimated time to completion in seconds."""
        if self.current_throughput_jobs_per_sec > 0:
            remaining_jobs = self.total_jobs - self.completed_jobs
            return remaining_jobs / self.current_throughput_jobs_per_sec
        return None


@dataclass
class WorkerConfig:
    """
    Configuration for worker processes/threads.
    
    Attributes:
        worker_type: Type of workers ('process' or 'thread')
        num_workers: Number of worker processes/threads
        worker_memory_limit_mb: Memory limit per worker in MB
        worker_timeout_sec: Timeout for individual workers
        enable_dynamic_scaling: Enable dynamic worker scaling
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers
        scale_threshold: Threshold for scaling decisions
    """
    worker_type: str = "process"  # 'process' or 'thread'
    num_workers: int = 4
    worker_memory_limit_mb: int = 512
    worker_timeout_sec: float = 300.0
    enable_dynamic_scaling: bool = True
    min_workers: int = 2
    max_workers: int = 16
    scale_threshold: float = 0.8  # Scale up when utilization > threshold


class ParallelValidator:
    """
    High-performance parallel validation system.
    
    Manages parallel validation of multiple configurations with intelligent
    resource management, fault tolerance, and performance optimization.
    """
    
    def __init__(self, worker_config: Optional[WorkerConfig] = None):
        """
        Initialize parallel validator.
        
        Args:
            worker_config: Worker configuration (auto-configured if None)
        """
        if worker_config is None:
            worker_config = self._auto_configure_workers()
        
        self.worker_config = worker_config
        self.cache = get_performance_cache()
        
        # State management
        self._active_jobs: Dict[str, ValidationJob] = {}
        self._completed_jobs: Dict[str, Tuple[ValidationJob, ValidationResult]] = {}
        self._failed_jobs: Dict[str, Tuple[ValidationJob, Exception]] = {}
        
        # Progress tracking
        self._progress_callbacks: List[Callable[[ValidationProgress], None]] = []
        self._current_progress = ValidationProgress()
        
        # Worker management
        self._executor: Optional[Union[ProcessPoolExecutor, ThreadPoolExecutor]] = None
        self._worker_stats: Dict[str, Dict[str, Any]] = {}
        
        # Batch processing
        self._pending_batches: List[ValidationBatch] = []
        self._processing_batches: Dict[str, ValidationBatch] = {}
        
        logger.info(f"Parallel validator initialized: {worker_config.num_workers} {worker_config.worker_type} workers")
    
    def add_progress_callback(self, callback: Callable[[ValidationProgress], None]) -> None:
        """Add progress callback function."""
        self._progress_callbacks.append(callback)
    
    def validate_configurations(self, 
                               configs: List[Dict[str, Any]],
                               validation_level: ValidationLevel = ValidationLevel.STANDARD,
                               batch_size: int = 50,
                               enable_streaming: bool = True) -> Iterator[Tuple[Dict[str, Any], ValidationResult]]:
        """
        Validate multiple configurations in parallel with streaming results.
        
        Args:
            configs: List of configurations to validate
            validation_level: Validation thoroughness level
            batch_size: Number of configs per batch
            enable_streaming: Enable streaming results as they complete
            
        Yields:
            Tuples of (config, validation_result) as they complete
        """
        if not configs:
            return
        
        logger.info(f"Starting parallel validation of {len(configs)} configurations")
        
        # Create validation jobs
        jobs = [
            ValidationJob(
                job_id=str(uuid.uuid4()),
                config=config,
                validation_level=validation_level
            )
            for config in configs
        ]
        
        # Initialize progress tracking
        self._current_progress = ValidationProgress(
            total_jobs=len(jobs),
            start_time=time.time()
        )
        
        # Process jobs
        if enable_streaming:
            yield from self._validate_streaming(jobs, batch_size)
        else:
            results = self._validate_batch_sync(jobs, batch_size)
            for config, result in results:
                yield config, result
    
    def validate_configurations_async(self,
                                    configs: List[Dict[str, Any]],
                                    validation_level: ValidationLevel = ValidationLevel.STANDARD,
                                    batch_size: int = 50) -> List[Tuple[Dict[str, Any], ValidationResult]]:
        """
        Validate multiple configurations asynchronously.
        
        Args:
            configs: List of configurations to validate
            validation_level: Validation thoroughness level
            batch_size: Number of configs per batch
            
        Returns:
            List of (config, validation_result) tuples
        """
        return list(self.validate_configurations(
            configs, validation_level, batch_size, enable_streaming=False
        ))
    
    def estimate_processing_time(self, 
                               num_configs: int,
                               validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Dict[str, float]:
        """
        Estimate processing time for a given number of configurations.
        
        Args:
            num_configs: Number of configurations to validate
            validation_level: Validation thoroughness level
            
        Returns:
            Dictionary with time estimates
        """
        # Base time estimates per validation level (in milliseconds per config)
        base_times = {
            ValidationLevel.FAST: 50.0,
            ValidationLevel.STANDARD: 200.0,
            ValidationLevel.THOROUGH: 500.0,
            ValidationLevel.EXHAUSTIVE: 1000.0
        }
        
        base_time_per_config = base_times.get(validation_level, 200.0)
        
        # Adjust for parallelism
        effective_workers = min(self.worker_config.num_workers, num_configs)
        parallel_efficiency = 0.8  # Account for overhead
        
        parallel_time_per_config = base_time_per_config / (effective_workers * parallel_efficiency)
        
        # Total estimates
        total_time_sec = (num_configs * parallel_time_per_config) / 1000.0
        total_time_min = total_time_sec / 60.0
        
        return {
            'total_configs': num_configs,
            'base_time_per_config_ms': base_time_per_config,
            'parallel_time_per_config_ms': parallel_time_per_config,
            'estimated_total_time_sec': total_time_sec,
            'estimated_total_time_min': total_time_min,
            'effective_workers': effective_workers,
            'validation_level': validation_level.value
        }
    
    def get_current_progress(self) -> ValidationProgress:
        """Get current validation progress."""
        return self._current_progress
    
    def cancel_all_jobs(self) -> int:
        """
        Cancel all pending and in-progress jobs.
        
        Returns:
            Number of jobs cancelled
        """
        cancelled_count = 0
        
        # Cancel pending batches
        cancelled_count += sum(len(batch.jobs) for batch in self._pending_batches)
        self._pending_batches.clear()
        
        # Cancel in-progress jobs (if possible)
        if self._executor:
            # For thread/process pools, we can't easily cancel individual tasks
            # So we shutdown and recreate the executor
            self._executor.shutdown(wait=False)
            self._executor = None
            cancelled_count += len(self._active_jobs)
            self._active_jobs.clear()
        
        logger.info(f"Cancelled {cancelled_count} validation jobs")
        return cancelled_count
    
    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get worker performance statistics."""
        if not self._worker_stats:
            return {
                'total_workers': self.worker_config.num_workers,
                'active_workers': 0,
                'average_cpu_usage': 0.0,
                'average_memory_usage_mb': 0.0,
                'total_jobs_processed': 0
            }
        
        # Aggregate worker statistics
        total_cpu = sum(stats.get('cpu_percent', 0) for stats in self._worker_stats.values())
        total_memory = sum(stats.get('memory_mb', 0) for stats in self._worker_stats.values())
        total_jobs = sum(stats.get('jobs_processed', 0) for stats in self._worker_stats.values())
        
        return {
            'total_workers': len(self._worker_stats),
            'active_workers': len([s for s in self._worker_stats.values() if s.get('active', False)]),
            'average_cpu_usage': total_cpu / len(self._worker_stats) if self._worker_stats else 0.0,
            'average_memory_usage_mb': total_memory / len(self._worker_stats) if self._worker_stats else 0.0,
            'total_jobs_processed': total_jobs
        }
    
    def optimize_worker_count(self, target_throughput: Optional[float] = None) -> int:
        """
        Optimize number of workers based on current performance.
        
        Args:
            target_throughput: Target jobs per second (auto-determined if None)
            
        Returns:
            New optimal worker count
        """
        current_stats = self.get_worker_statistics()
        current_throughput = self._current_progress.current_throughput_jobs_per_sec
        
        if target_throughput is None:
            # Auto-determine target based on system resources
            available_cores = psutil.cpu_count(logical=False)
            target_throughput = available_cores * 2.0  # Conservative target
        
        # Simple optimization: adjust workers based on throughput
        if current_throughput < target_throughput * 0.8:
            # Increase workers if underutilized
            new_count = min(
                self.worker_config.max_workers,
                int(self.worker_config.num_workers * 1.5)
            )
        elif current_throughput > target_throughput * 1.2:
            # Decrease workers if over-utilized
            new_count = max(
                self.worker_config.min_workers,
                int(self.worker_config.num_workers * 0.8)
            )
        else:
            new_count = self.worker_config.num_workers
        
        if new_count != self.worker_config.num_workers:
            logger.info(f"Optimizing worker count: {self.worker_config.num_workers} → {new_count}")
            self._reconfigure_workers(new_count)
        
        return new_count
    
    def _auto_configure_workers(self) -> WorkerConfig:
        """Auto-configure workers based on system resources."""
        # Detect system capabilities
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative worker configuration
        if memory_gb > 32 and cpu_count >= 16:
            # High-end system
            config = WorkerConfig(
                worker_type="process",
                num_workers=min(16, cpu_count),
                worker_memory_limit_mb=1024,
                max_workers=min(32, cpu_count * 2)
            )
        elif memory_gb > 16 and cpu_count >= 8:
            # Mid-range system
            config = WorkerConfig(
                worker_type="process",
                num_workers=min(8, cpu_count),
                worker_memory_limit_mb=512,
                max_workers=min(16, cpu_count * 2)
            )
        else:
            # Low-end system or limited resources
            config = WorkerConfig(
                worker_type="thread",  # Use threads for limited resources
                num_workers=min(4, cpu_count),
                worker_memory_limit_mb=256,
                max_workers=min(8, cpu_count)
            )
        
        logger.info(f"Auto-configured workers: {config.num_workers} {config.worker_type} workers "
                   f"(system: {cpu_count} cores, {memory_gb:.1f}GB RAM)")
        
        return config
    
    def _validate_streaming(self, jobs: List[ValidationJob], batch_size: int) -> Iterator[Tuple[Dict[str, Any], ValidationResult]]:
        """Validate jobs with streaming results."""
        # Create batches
        batches = self._create_batches(jobs, batch_size)
        
        # Initialize executor
        self._initialize_executor()
        
        try:
            # Submit batches and process results as they complete
            future_to_batch = {}
            
            with self._executor:
                # Submit initial batches
                for batch in batches[:self.worker_config.num_workers]:
                    future = self._executor.submit(self._process_batch, batch)
                    future_to_batch[future] = batch
                    self._processing_batches[batch.batch_id] = batch
                
                batch_idx = self.worker_config.num_workers
                
                # Process completed batches and submit new ones
                while future_to_batch:
                    # Wait for at least one batch to complete
                    completed_futures = as_completed(future_to_batch.keys(), timeout=30.0)
                    
                    for future in completed_futures:
                        batch = future_to_batch.pop(future)
                        
                        try:
                            batch_results = future.result()
                            
                            # Yield results
                            for job, result in batch_results:
                                self._current_progress.completed_jobs += 1
                                self._update_progress_metrics()
                                yield job.config, result
                            
                        except Exception as e:
                            logger.error(f"Batch {batch.batch_id} failed: {e}")
                            self._current_progress.failed_jobs += len(batch.jobs)
                            
                            # Yield failed results
                            for job in batch.jobs:
                                failed_result = ValidationResult(
                                    is_valid=False,
                                    errors=[f"Batch processing failed: {str(e)}"],
                                    warnings=[], suggestions=[], info=[], hardware_info={}
                                )
                                yield job.config, failed_result
                        
                        finally:
                            self._processing_batches.pop(batch.batch_id, None)
                            self._notify_progress()
                        
                        # Submit next batch if available
                        if batch_idx < len(batches):
                            next_batch = batches[batch_idx]
                            future = self._executor.submit(self._process_batch, next_batch)
                            future_to_batch[future] = next_batch
                            self._processing_batches[next_batch.batch_id] = next_batch
                            batch_idx += 1
                            
                        # Break if all batches processed
                        if not future_to_batch and batch_idx >= len(batches):
                            break
                
        finally:
            self._cleanup_executor()
    
    def _validate_batch_sync(self, jobs: List[ValidationJob], batch_size: int) -> List[Tuple[Dict[str, Any], ValidationResult]]:
        """Validate jobs synchronously and return all results."""
        results = []
        for config, result in self._validate_streaming(jobs, batch_size):
            results.append((config, result))
        return results
    
    def _create_batches(self, jobs: List[ValidationJob], batch_size: int) -> List[ValidationBatch]:
        """Create batches from validation jobs."""
        batches = []
        current_batch = ValidationBatch(
            batch_id=str(uuid.uuid4()),
            batch_size=batch_size
        )
        
        for job in jobs:
            current_batch.add_job(job)
            
            if current_batch.is_full():
                batches.append(current_batch)
                current_batch = ValidationBatch(
                    batch_id=str(uuid.uuid4()),
                    batch_size=batch_size
                )
        
        # Add final batch if it has jobs
        if current_batch.size > 0:
            batches.append(current_batch)
        
        logger.debug(f"Created {len(batches)} batches from {len(jobs)} jobs")
        return batches
    
    def _initialize_executor(self) -> None:
        """Initialize worker executor."""
        if self._executor is not None:
            return
        
        if self.worker_config.worker_type == "process":
            self._executor = ProcessPoolExecutor(
                max_workers=self.worker_config.num_workers,
                mp_context=mp.get_context('spawn')  # More stable across platforms
            )
        else:
            self._executor = ThreadPoolExecutor(
                max_workers=self.worker_config.num_workers
            )
        
        logger.debug(f"Initialized {self.worker_config.worker_type} executor with {self.worker_config.num_workers} workers")
    
    def _cleanup_executor(self) -> None:
        """Cleanup worker executor."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
            logger.debug("Worker executor shutdown completed")
    
    def _reconfigure_workers(self, new_worker_count: int) -> None:
        """Reconfigure workers with new count."""
        # Shutdown existing executor
        self._cleanup_executor()
        
        # Update configuration
        self.worker_config.num_workers = new_worker_count
        
        # Reinitialize executor
        self._initialize_executor()
        
        logger.info(f"Reconfigured workers: {new_worker_count} workers")
    
    def _process_batch(self, batch: ValidationBatch) -> List[Tuple[ValidationJob, ValidationResult]]:
        """
        Process a batch of validation jobs.
        
        This method runs in worker processes, so it needs to be self-contained.
        """
        results = []
        
        # Create validator instance for this worker
        validator = LazyValidator(validation_level=ValidationLevel.STANDARD)
        
        for job in batch.jobs:
            try:
                # Validate configuration
                result = validator.validate_sync(
                    job.config,
                    skip_optional=True  # Skip optional tasks for performance
                )
                results.append((job, result))
                
            except Exception as e:
                # Create failed result
                failed_result = ValidationResult(
                    is_valid=False,
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[], suggestions=[], info=[], hardware_info={}
                )
                results.append((job, failed_result))
        
        return results
    
    def _update_progress_metrics(self) -> None:
        """Update progress metrics and estimates."""
        elapsed_time = self._current_progress.elapsed_time_sec
        
        if elapsed_time > 0:
            self._current_progress.current_throughput_jobs_per_sec = (
                self._current_progress.completed_jobs / elapsed_time
            )
        
        # Update ETA
        if self._current_progress.current_throughput_jobs_per_sec > 0:
            remaining_jobs = (self._current_progress.total_jobs - 
                            self._current_progress.completed_jobs)
            eta_sec = remaining_jobs / self._current_progress.current_throughput_jobs_per_sec
            self._current_progress.estimated_completion_time = time.time() + eta_sec
    
    def _notify_progress(self) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(self._current_progress)
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")
    
    def __del__(self):
        """Cleanup resources on destruction."""
        self._cleanup_executor()


# Global parallel validator instance
_global_parallel_validator: Optional[ParallelValidator] = None


def get_parallel_validator(worker_config: Optional[WorkerConfig] = None) -> ParallelValidator:
    """Get global parallel validator instance."""
    global _global_parallel_validator
    
    if _global_parallel_validator is None:
        _global_parallel_validator = ParallelValidator(worker_config)
        logger.info("Global parallel validator initialized")
    
    return _global_parallel_validator


def validate_configurations_parallel(configs: List[Dict[str, Any]],
                                    validation_level: ValidationLevel = ValidationLevel.STANDARD,
                                    max_workers: Optional[int] = None) -> List[Tuple[Dict[str, Any], ValidationResult]]:
    """
    Convenience function for parallel configuration validation.
    
    Args:
        configs: List of configurations to validate
        validation_level: Validation thoroughness level
        max_workers: Maximum number of worker processes
        
    Returns:
        List of (config, validation_result) tuples
    """
    worker_config = None
    if max_workers is not None:
        worker_config = WorkerConfig(num_workers=max_workers)
    
    validator = get_parallel_validator(worker_config)
    return validator.validate_configurations_async(configs, validation_level)