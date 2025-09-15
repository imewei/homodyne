"""
Performance Monitoring and Profiling for Homodyne v2 Configuration
===================================================================

Comprehensive performance monitoring and profiling system for configuration
operations, providing detailed insights into bottlenecks and optimization
opportunities for enterprise workloads.

Key Features:
- Real-time performance monitoring with metrics collection
- Detailed profiling of configuration operations
- Bottleneck identification and optimization recommendations
- Performance regression testing and baseline management
- Resource utilization tracking (CPU, memory, I/O)
- Performance analytics and reporting
- Automated performance tuning suggestions
"""

import cProfile
import io
import json
import pickle
import pstats
import threading
import time
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import psutil

try:
    import line_profiler

    HAS_LINE_PROFILER = True
except ImportError:
    HAS_LINE_PROFILER = False
    line_profiler = None

try:
    import memory_profiler

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    memory_profiler = None

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for configuration operations.

    Attributes:
        operation_name: Name of the operation being measured
        execution_time_ms: Total execution time in milliseconds
        cpu_time_ms: CPU time in milliseconds
        memory_peak_mb: Peak memory usage in MB
        memory_delta_mb: Change in memory usage in MB
        function_calls: Number of function calls
        io_operations: Number of I/O operations
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        timestamp: Operation timestamp
        metadata: Additional metadata
    """

    operation_name: str
    execution_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    function_calls: int = 0
    io_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfileData:
    """
    Detailed profiling data for performance analysis.

    Attributes:
        function_stats: Per-function performance statistics
        memory_trace: Memory allocation trace
        bottlenecks: Identified performance bottlenecks
        recommendations: Performance optimization recommendations
        execution_trace: Detailed execution trace
    """

    function_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    memory_trace: List[Dict[str, Any]] = field(default_factory=list)
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceBaseline:
    """
    Performance baseline for regression testing.

    Attributes:
        baseline_name: Name of the baseline
        operation_metrics: Baseline metrics for operations
        system_info: System information when baseline was created
        created_timestamp: Baseline creation timestamp
        version: Configuration system version
    """

    baseline_name: str
    operation_metrics: Dict[str, PerformanceMetrics]
    system_info: Dict[str, Any] = field(default_factory=dict)
    created_timestamp: float = field(default_factory=time.time)
    version: str = "v2.0"


class PerformanceProfiler:
    """
    Comprehensive performance profiling system.

    Provides detailed profiling capabilities for configuration operations
    with real-time monitoring and analysis.
    """

    def __init__(
        self,
        enable_memory_profiling: bool = True,
        enable_line_profiling: bool = False,
        metrics_history_size: int = 1000,
    ):
        """
        Initialize performance profiler.

        Args:
            enable_memory_profiling: Enable memory usage profiling
            enable_line_profiling: Enable line-by-line profiling
            metrics_history_size: Maximum number of metrics to keep in history
        """
        self.enable_memory_profiling = enable_memory_profiling
        self.enable_line_profiling = enable_line_profiling and HAS_LINE_PROFILER

        # Metrics storage
        self._metrics_history: deque = deque(maxlen=metrics_history_size)
        self._operation_stats: Dict[str, List[PerformanceMetrics]] = defaultdict(list)

        # Profiling state
        self._active_profiles: Dict[str, Dict[str, Any]] = {}
        self._profile_lock = threading.Lock()

        # Baseline management
        self._baselines: Dict[str, PerformanceBaseline] = {}

        # System monitoring
        self._system_monitor_active = False
        self._system_metrics: deque = deque(maxlen=100)

        # Initialize memory profiling if requested
        if self.enable_memory_profiling:
            try:
                tracemalloc.start()
                logger.debug("Memory profiling enabled")
            except Exception as e:
                logger.warning(f"Failed to enable memory profiling: {e}")
                self.enable_memory_profiling = False

        logger.info(
            f"Performance profiler initialized: memory_profiling={self.enable_memory_profiling}, "
            f"line_profiling={self.enable_line_profiling}"
        )

    @contextmanager
    def profile_operation(
        self, operation_name: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for profiling configuration operations.

        Args:
            operation_name: Name of the operation to profile
            metadata: Additional metadata to store
        """
        if metadata is None:
            metadata = {}

        profile_id = f"{operation_name}_{int(time.time() * 1000)}"

        # Initialize profiling
        profiler = cProfile.Profile()
        initial_memory = self._get_memory_usage()
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()

        # Start system monitoring for this operation
        self._start_operation_monitoring(profile_id)

        try:
            profiler.enable()
            yield profile_id
        finally:
            profiler.disable()

            # Calculate metrics
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            final_memory = self._get_memory_usage()

            # Stop monitoring
            self._stop_operation_monitoring(profile_id)

            # Create performance metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time_ms=(end_time - start_time) * 1000,
                cpu_time_ms=(end_cpu_time - start_cpu_time) * 1000,
                memory_peak_mb=final_memory.get("peak_mb", 0),
                memory_delta_mb=final_memory.get("current_mb", 0)
                - initial_memory.get("current_mb", 0),
                metadata=metadata,
            )

            # Analyze profiler results
            self._analyze_profile_results(profiler, metrics)

            # Store metrics
            self._store_metrics(metrics)

            logger.debug(
                f"Operation profiled: {operation_name} "
                f"({metrics.execution_time_ms:.1f}ms, {metrics.memory_delta_mb:.1f}MB delta)"
            )

    def profile_function(
        self, func: Callable, *args, **kwargs
    ) -> Tuple[Any, PerformanceMetrics]:
        """
        Profile a specific function call.

        Args:
            func: Function to profile
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Tuple of (function_result, performance_metrics)
        """
        operation_name = f"{func.__module__}.{func.__name__}"

        with self.profile_operation(operation_name) as profile_id:
            result = func(*args, **kwargs)

        # Get the metrics for this operation
        latest_metrics = self._metrics_history[-1] if self._metrics_history else None

        return result, latest_metrics

    def start_continuous_monitoring(self, interval: float = 1.0) -> None:
        """
        Start continuous system performance monitoring.

        Args:
            interval: Monitoring interval in seconds
        """
        if self._system_monitor_active:
            return

        self._system_monitor_active = True

        def monitor_loop():
            while self._system_monitor_active:
                try:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_info = psutil.virtual_memory()

                    system_metrics = {
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_info.percent,
                        "memory_available_gb": memory_info.available / (1024**3),
                        "active_profiles": len(self._active_profiles),
                    }

                    self._system_metrics.append(system_metrics)

                    time.sleep(interval)

                except Exception as e:
                    logger.debug(f"System monitoring error: {e}")

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        logger.info(f"Continuous monitoring started with {interval}s interval")

    def stop_continuous_monitoring(self) -> None:
        """Stop continuous system performance monitoring."""
        self._system_monitor_active = False
        logger.info("Continuous monitoring stopped")

    def get_operation_statistics(
        self, operation_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get performance statistics for operations.

        Args:
            operation_name: Specific operation name (None for all operations)

        Returns:
            Performance statistics
        """
        if operation_name:
            if operation_name not in self._operation_stats:
                return {"error": f"No statistics found for operation: {operation_name}"}

            metrics_list = self._operation_stats[operation_name]
            return self._calculate_operation_statistics(operation_name, metrics_list)

        # Return statistics for all operations
        all_stats = {}
        for op_name, metrics_list in self._operation_stats.items():
            all_stats[op_name] = self._calculate_operation_statistics(
                op_name, metrics_list
            )

        return all_stats

    def identify_bottlenecks(
        self, operation_name: Optional[str] = None, threshold_percentile: float = 95.0
    ) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks.

        Args:
            operation_name: Specific operation to analyze (None for all)
            threshold_percentile: Percentile threshold for bottleneck identification

        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []

        stats = self.get_operation_statistics(operation_name)

        if operation_name:
            bottlenecks.extend(
                self._analyze_operation_bottlenecks(
                    operation_name, stats, threshold_percentile
                )
            )
        else:
            for op_name, op_stats in stats.items():
                if isinstance(op_stats, dict) and "error" not in op_stats:
                    bottlenecks.extend(
                        self._analyze_operation_bottlenecks(
                            op_name, op_stats, threshold_percentile
                        )
                    )

        # Sort bottlenecks by severity
        bottlenecks.sort(key=lambda x: x.get("severity_score", 0), reverse=True)

        return bottlenecks

    def generate_performance_report(
        self, output_file: Optional[Path] = None, include_detailed_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.

        Args:
            output_file: Optional file to save the report
            include_detailed_analysis: Include detailed profiling analysis

        Returns:
            Performance report dictionary
        """
        report = {
            "report_generated": time.time(),
            "profiler_config": {
                "memory_profiling_enabled": self.enable_memory_profiling,
                "line_profiling_enabled": self.enable_line_profiling,
                "metrics_history_size": len(self._metrics_history),
            },
            "operation_statistics": self.get_operation_statistics(),
            "identified_bottlenecks": self.identify_bottlenecks(),
            "system_metrics_summary": self._get_system_metrics_summary(),
            "performance_trends": self._analyze_performance_trends(),
            "optimization_recommendations": self._generate_optimization_recommendations(),
        }

        if include_detailed_analysis:
            report["detailed_analysis"] = self._generate_detailed_analysis()

        # Save to file if requested
        if output_file:
            try:
                with open(output_file, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Performance report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save performance report: {e}")

        return report

    def create_baseline(self, baseline_name: str) -> None:
        """
        Create performance baseline for regression testing.

        Args:
            baseline_name: Name for the baseline
        """
        # Get current operation metrics
        baseline_metrics = {}
        for operation_name, metrics_list in self._operation_stats.items():
            if metrics_list:
                # Use median metrics as baseline
                sorted_metrics = sorted(metrics_list, key=lambda m: m.execution_time_ms)
                median_idx = len(sorted_metrics) // 2
                baseline_metrics[operation_name] = sorted_metrics[median_idx]

        # Get system information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "platform": psutil.os.name,
        }

        # Create baseline
        baseline = PerformanceBaseline(
            baseline_name=baseline_name,
            operation_metrics=baseline_metrics,
            system_info=system_info,
        )

        self._baselines[baseline_name] = baseline
        logger.info(f"Performance baseline created: {baseline_name}")

    def compare_to_baseline(self, baseline_name: str) -> Dict[str, Any]:
        """
        Compare current performance to baseline.

        Args:
            baseline_name: Name of baseline to compare against

        Returns:
            Comparison results
        """
        if baseline_name not in self._baselines:
            return {"error": f"Baseline not found: {baseline_name}"}

        baseline = self._baselines[baseline_name]
        current_stats = self.get_operation_statistics()

        comparison = {
            "baseline_name": baseline_name,
            "baseline_created": baseline.created_timestamp,
            "comparison_timestamp": time.time(),
            "operation_comparisons": {},
            "overall_regression": False,
            "regression_operations": [],
        }

        for operation_name, baseline_metrics in baseline.operation_metrics.items():
            if operation_name in current_stats:
                current_stats_op = current_stats[operation_name]

                # Compare execution time
                baseline_time = baseline_metrics.execution_time_ms
                current_mean_time = current_stats_op.get("execution_time_ms_mean", 0)

                time_regression_pct = (
                    ((current_mean_time - baseline_time) / baseline_time * 100)
                    if baseline_time > 0
                    else 0
                )

                # Compare memory usage
                baseline_memory = baseline_metrics.memory_peak_mb
                current_mean_memory = current_stats_op.get("memory_peak_mb_mean", 0)

                memory_regression_pct = (
                    ((current_mean_memory - baseline_memory) / baseline_memory * 100)
                    if baseline_memory > 0
                    else 0
                )

                op_comparison = {
                    "baseline_execution_time_ms": baseline_time,
                    "current_execution_time_ms": current_mean_time,
                    "time_regression_percent": time_regression_pct,
                    "baseline_memory_mb": baseline_memory,
                    "current_memory_mb": current_mean_memory,
                    "memory_regression_percent": memory_regression_pct,
                    "has_regression": time_regression_pct > 20.0
                    or memory_regression_pct > 20.0,  # 20% threshold
                }

                comparison["operation_comparisons"][operation_name] = op_comparison

                if op_comparison["has_regression"]:
                    comparison["overall_regression"] = True
                    comparison["regression_operations"].append(operation_name)

        return comparison

    def save_baselines(self, file_path: Path) -> None:
        """Save baselines to file."""
        try:
            with open(file_path, "wb") as f:
                pickle.dump(self._baselines, f)
            logger.info(f"Baselines saved to: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    def load_baselines(self, file_path: Path) -> None:
        """Load baselines from file."""
        try:
            with open(file_path, "rb") as f:
                self._baselines = pickle.load(f)
            logger.info(f"Baselines loaded from: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load baselines: {e}")

    def clear_metrics_history(self) -> None:
        """Clear all stored metrics and history."""
        self._metrics_history.clear()
        self._operation_stats.clear()
        logger.info("Performance metrics history cleared")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            result = {
                "current_mb": memory_info.rss / (1024**2),
                "peak_mb": memory_info.rss / (1024**2),  # fallback
            }

            if self.enable_memory_profiling and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
                result["peak_mb"] = peak / (1024**2)

            return result

        except Exception as e:
            logger.debug(f"Failed to get memory usage: {e}")
            return {"current_mb": 0.0, "peak_mb": 0.0}

    def _analyze_profile_results(
        self, profiler: cProfile.Profile, metrics: PerformanceMetrics
    ) -> None:
        """Analyze cProfile results and update metrics."""
        try:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")

            # Count function calls
            metrics.function_calls = ps.total_calls

            # Extract top functions for analysis
            top_functions = []
            ps.print_stats(10)  # Top 10 functions

            # Store detailed profiling data if requested
            profile_data = {
                "total_calls": ps.total_calls,
                "primitive_calls": ps.prim_calls,
                "total_time": ps.total_tt,
            }

            metrics.metadata["profile_data"] = profile_data

        except Exception as e:
            logger.debug(f"Failed to analyze profile results: {e}")

    def _store_metrics(self, metrics: PerformanceMetrics) -> None:
        """Store performance metrics."""
        with self._profile_lock:
            self._metrics_history.append(metrics)
            self._operation_stats[metrics.operation_name].append(metrics)

    def _start_operation_monitoring(self, profile_id: str) -> None:
        """Start monitoring for a specific operation."""
        with self._profile_lock:
            self._active_profiles[profile_id] = {
                "start_time": time.time(),
                "initial_memory": self._get_memory_usage(),
            }

    def _stop_operation_monitoring(self, profile_id: str) -> None:
        """Stop monitoring for a specific operation."""
        with self._profile_lock:
            self._active_profiles.pop(profile_id, None)

    def _calculate_operation_statistics(
        self, operation_name: str, metrics_list: List[PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Calculate statistics for an operation."""
        if not metrics_list:
            return {}

        execution_times = [m.execution_time_ms for m in metrics_list]
        memory_peaks = [m.memory_peak_mb for m in metrics_list]
        memory_deltas = [m.memory_delta_mb for m in metrics_list]

        return {
            "operation_name": operation_name,
            "total_executions": len(metrics_list),
            "execution_time_ms_mean": sum(execution_times) / len(execution_times),
            "execution_time_ms_min": min(execution_times),
            "execution_time_ms_max": max(execution_times),
            "execution_time_ms_percentile_95": sorted(execution_times)[
                int(len(execution_times) * 0.95)
            ],
            "memory_peak_mb_mean": sum(memory_peaks) / len(memory_peaks),
            "memory_peak_mb_max": max(memory_peaks),
            "memory_delta_mb_mean": sum(memory_deltas) / len(memory_deltas),
            "function_calls_mean": sum(m.function_calls for m in metrics_list)
            / len(metrics_list),
        }

    def _analyze_operation_bottlenecks(
        self, operation_name: str, stats: Dict[str, Any], threshold_percentile: float
    ) -> List[Dict[str, Any]]:
        """Analyze bottlenecks for a specific operation."""
        bottlenecks = []

        # Check execution time bottlenecks
        mean_time = stats.get("execution_time_ms_mean", 0)
        p95_time = stats.get("execution_time_ms_percentile_95", 0)
        max_time = stats.get("execution_time_ms_max", 0)

        if p95_time > mean_time * 2:  # P95 is more than 2x the mean
            bottlenecks.append(
                {
                    "operation": operation_name,
                    "type": "execution_time_variability",
                    "description": f"High execution time variability (P95: {p95_time:.1f}ms, Mean: {mean_time:.1f}ms)",
                    "severity_score": (p95_time / mean_time) * 10,
                    "recommendation": "Investigate causes of execution time spikes",
                }
            )

        if mean_time > 1000:  # Operations taking more than 1 second on average
            bottlenecks.append(
                {
                    "operation": operation_name,
                    "type": "slow_execution",
                    "description": f"Slow average execution time ({mean_time:.1f}ms)",
                    "severity_score": mean_time / 100,  # Score based on seconds
                    "recommendation": "Optimize critical path or consider caching",
                }
            )

        # Check memory bottlenecks
        memory_peak = stats.get("memory_peak_mb_max", 0)
        if memory_peak > 1000:  # Peak memory usage > 1GB
            bottlenecks.append(
                {
                    "operation": operation_name,
                    "type": "high_memory_usage",
                    "description": f"High peak memory usage ({memory_peak:.1f}MB)",
                    "severity_score": memory_peak / 100,
                    "recommendation": "Consider streaming processing or memory optimization",
                }
            )

        return bottlenecks

    def _get_system_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of system metrics."""
        if not self._system_metrics:
            return {}

        cpu_values = [m["cpu_percent"] for m in self._system_metrics]
        memory_values = [m["memory_percent"] for m in self._system_metrics]

        return {
            "monitoring_duration_minutes": (
                time.time() - self._system_metrics[0]["timestamp"]
            )
            / 60,
            "cpu_usage_mean": sum(cpu_values) / len(cpu_values),
            "cpu_usage_max": max(cpu_values),
            "memory_usage_mean": sum(memory_values) / len(memory_values),
            "memory_usage_max": max(memory_values),
            "samples_collected": len(self._system_metrics),
        }

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if len(self._metrics_history) < 10:
            return {
                "insufficient_data": "Need at least 10 measurements for trend analysis"
            }

        # Analyze recent vs older performance
        recent_metrics = list(self._metrics_history)[-10:]  # Last 10 measurements
        older_metrics = (
            list(self._metrics_history)[:-10] if len(self._metrics_history) > 10 else []
        )

        if not older_metrics:
            return {"trend": "insufficient_historical_data"}

        recent_avg_time = sum(m.execution_time_ms for m in recent_metrics) / len(
            recent_metrics
        )
        older_avg_time = sum(m.execution_time_ms for m in older_metrics) / len(
            older_metrics
        )

        trend_direction = (
            "improving" if recent_avg_time < older_avg_time else "degrading"
        )
        trend_magnitude = abs(recent_avg_time - older_avg_time) / older_avg_time * 100

        return {
            "trend_direction": trend_direction,
            "trend_magnitude_percent": trend_magnitude,
            "recent_avg_time_ms": recent_avg_time,
            "historical_avg_time_ms": older_avg_time,
        }

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []

        # Analyze bottlenecks
        bottlenecks = self.identify_bottlenecks()

        if any(b["type"] == "slow_execution" for b in bottlenecks):
            recommendations.append(
                "Consider implementing caching for frequently accessed configurations"
            )
            recommendations.append(
                "Optimize critical path algorithms or use parallel processing"
            )

        if any(b["type"] == "high_memory_usage" for b in bottlenecks):
            recommendations.append("Implement streaming processing for large datasets")
            recommendations.append(
                "Use memory-efficient data structures and garbage collection tuning"
            )

        if any(b["type"] == "execution_time_variability" for b in bottlenecks):
            recommendations.append("Investigate and eliminate performance spikes")
            recommendations.append(
                "Consider using performance budgets and SLA monitoring"
            )

        # System-level recommendations
        system_summary = self._get_system_metrics_summary()
        if system_summary.get("cpu_usage_max", 0) > 90:
            recommendations.append("Consider CPU optimization or scaling to more cores")

        if system_summary.get("memory_usage_max", 0) > 90:
            recommendations.append(
                "Consider memory optimization or increasing available RAM"
            )

        return recommendations

    def _generate_detailed_analysis(self) -> Dict[str, Any]:
        """Generate detailed performance analysis."""
        return {
            "function_call_analysis": self._analyze_function_calls(),
            "memory_allocation_patterns": self._analyze_memory_patterns(),
            "io_operation_analysis": self._analyze_io_operations(),
            "cache_efficiency_analysis": self._analyze_cache_efficiency(),
        }

    def _analyze_function_calls(self) -> Dict[str, Any]:
        """Analyze function call patterns."""
        if not self._metrics_history:
            return {}

        total_calls = sum(m.function_calls for m in self._metrics_history)
        avg_calls = total_calls / len(self._metrics_history)

        return {
            "total_function_calls": total_calls,
            "average_calls_per_operation": avg_calls,
            "call_efficiency_score": min(
                100, max(0, 100 - (avg_calls / 1000) * 10)
            ),  # Penalty for excessive calls
        }

    def _analyze_memory_patterns(self) -> Dict[str, Any]:
        """Analyze memory allocation patterns."""
        if not self._metrics_history:
            return {}

        memory_deltas = [m.memory_delta_mb for m in self._metrics_history]
        memory_peaks = [m.memory_peak_mb for m in self._metrics_history]

        return {
            "average_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "max_memory_peak_mb": max(memory_peaks),
            "memory_leak_indicator": sum(1 for delta in memory_deltas if delta > 10)
            / len(memory_deltas),  # Fraction with high deltas
        }

    def _analyze_io_operations(self) -> Dict[str, Any]:
        """Analyze I/O operation patterns."""
        if not self._metrics_history:
            return {}

        total_io = sum(m.io_operations for m in self._metrics_history)
        avg_io = total_io / len(self._metrics_history)

        return {
            "total_io_operations": total_io,
            "average_io_per_operation": avg_io,
            "io_efficiency_score": min(100, max(0, 100 - (avg_io / 100) * 10)),
        }

    def _analyze_cache_efficiency(self) -> Dict[str, Any]:
        """Analyze cache efficiency patterns."""
        if not self._metrics_history:
            return {}

        total_hits = sum(m.cache_hits for m in self._metrics_history)
        total_misses = sum(m.cache_misses for m in self._metrics_history)
        total_requests = total_hits + total_misses

        if total_requests == 0:
            return {"cache_hit_rate": 0.0, "efficiency_score": 0}

        hit_rate = total_hits / total_requests

        return {
            "total_cache_hits": total_hits,
            "total_cache_misses": total_misses,
            "cache_hit_rate": hit_rate,
            "efficiency_score": hit_rate * 100,
        }


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler(
    enable_memory_profiling: bool = True, enable_line_profiling: bool = False
) -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler

    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(
            enable_memory_profiling=enable_memory_profiling,
            enable_line_profiling=enable_line_profiling,
        )
        logger.info("Global performance profiler initialized")

    return _global_profiler


@contextmanager
def profile_configuration_operation(
    operation_name: str, metadata: Optional[Dict[str, Any]] = None
):
    """Context manager for profiling configuration operations."""
    profiler = get_performance_profiler()
    with profiler.profile_operation(operation_name, metadata) as profile_id:
        yield profile_id
