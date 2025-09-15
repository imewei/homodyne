"""
Comprehensive Test Suite for Performance Optimizations
======================================================

Test suite for validating performance optimizations in the Homodyne v2
configuration system, including benchmarks, regression tests, and
stress tests for enterprise workloads.

Key Features:
- Performance benchmark tests for all optimization components
- Regression testing against performance baselines
- Stress testing with large datasets and high concurrency
- Memory usage validation and leak detection
- Cache effectiveness testing
- Parallel validation performance testing
- Real-world scenario simulation
"""

import asyncio
import gc
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.config.lazy_validator import (LazyValidator, ValidationLevel,
                                            ValidationPriority, ValidationTask)
from homodyne.config.memory_optimizer import MemoryMonitor
from homodyne.config.memory_optimizer import \
    StreamingConfig as MemStreamingConfig
from homodyne.config.memory_optimizer import StreamingProcessor
from homodyne.config.optimized_mode_resolver import (AnalysisMetrics,
                                                     OptimizedModeResolver,
                                                     StreamingConfig)
from homodyne.config.parallel_validator import (ParallelValidator,
                                                ValidationJob, WorkerConfig)
# Import the performance optimization modules
from homodyne.config.performance_cache import (PerformanceCache,
                                               ValidationResultCache,
                                               get_performance_cache)
from homodyne.config.performance_profiler import (PerformanceMetrics,
                                                  PerformanceProfiler)


class TestPerformanceCache:
    """Test suite for performance caching system."""

    def test_cache_initialization(self):
        """Test cache initialization with different configurations."""
        cache = PerformanceCache(max_memory_mb=128, max_entries=1000)

        assert cache.max_memory_bytes == 128 * 1024 * 1024
        assert cache.max_entries == 1000

        stats = cache.get_stats()
        assert stats.total_entries == 0
        assert stats.memory_usage_bytes == 0
        assert stats.hit_rate == 0.0

    def test_basic_cache_operations(self):
        """Test basic cache put/get operations."""
        cache = PerformanceCache(max_memory_mb=64, max_entries=100)

        # Test cache miss
        value, hit = cache.get("test_key")
        assert value is None
        assert hit is False

        # Test cache put and hit
        test_data = {"config": "test", "validation": True}
        cache.put("test_key", test_data, content_hash="test_hash")

        value, hit = cache.get("test_key")
        assert value == test_data
        assert hit is True

        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total_entries == 1

    def test_cache_invalidation(self):
        """Test cache invalidation strategies."""
        cache = PerformanceCache(max_memory_mb=64, max_entries=100)

        # Add test entries
        cache.put("key1", "value1", content_hash="hash1")
        cache.put("key2", "value2", content_hash="hash1")  # Same hash
        cache.put("key3", "value3", content_hash="hash2")

        # Test single key invalidation
        result = cache.invalidate("key1")
        assert result is True

        value, hit = cache.get("key1")
        assert hit is False

        # Test content hash invalidation
        invalidated = cache.invalidate_by_content_hash("hash1")
        assert invalidated == 1  # Only key2 should remain with hash1

        value, hit = cache.get("key2")
        assert hit is False

    def test_cache_memory_optimization(self):
        """Test cache memory optimization and eviction."""
        # Use small cache to trigger optimization
        cache = PerformanceCache(max_memory_mb=1, max_entries=10)

        # Fill cache beyond limits
        large_data = {"data": "x" * 10000}  # ~10KB per entry
        for i in range(20):
            cache.put(f"key_{i}", large_data.copy())

        # Trigger optimization
        stats = cache.optimize_memory_usage()

        assert stats["entries_removed"] > 0
        assert stats["memory_freed"] > 0

        cache_stats = cache.get_stats()
        assert cache_stats.total_entries <= 10  # Should respect max_entries

    def test_cache_warming(self):
        """Test cache warming functionality."""
        cache = PerformanceCache(max_memory_mb=64, max_entries=100)

        # Create test configurations for warming
        warm_configs = [
            {"analysis_mode": "static_isotropic", "param1": i} for i in range(10)
        ]

        warmed_count = cache.warm_cache(warm_configs)
        assert warmed_count == 10

        stats = cache.get_stats()
        assert stats.total_entries == 10

    @pytest.mark.performance
    def test_cache_performance_benchmark(self):
        """Benchmark cache performance under load."""
        cache = PerformanceCache(max_memory_mb=256, max_entries=10000)

        # Benchmark cache operations
        num_operations = 1000
        test_data = {"config": "benchmark_data", "size": 100}

        start_time = time.perf_counter()

        # Write operations
        for i in range(num_operations):
            cache.put(f"bench_key_{i}", test_data.copy())

        write_time = time.perf_counter() - start_time

        # Read operations
        start_time = time.perf_counter()
        hits = 0
        for i in range(num_operations):
            _, hit = cache.get(f"bench_key_{i}")
            if hit:
                hits += 1

        read_time = time.perf_counter() - start_time

        # Performance assertions
        assert write_time < 1.0  # Should complete writes in < 1 second
        assert read_time < 0.5  # Should complete reads in < 0.5 seconds
        assert hits == num_operations  # All should be cache hits

        print(
            f"Cache benchmark: {num_operations} writes in {write_time:.3f}s, "
            f"{num_operations} reads in {read_time:.3f}s"
        )


class TestOptimizedModeResolver:
    """Test suite for optimized mode resolution."""

    def test_streaming_config_initialization(self):
        """Test streaming configuration initialization."""
        config = StreamingConfig(chunk_size=50000, max_memory_mb=512, num_workers=4)

        resolver = OptimizedModeResolver(config)
        assert resolver.streaming_config.chunk_size == 50000
        assert resolver.streaming_config.max_memory_mb == 512
        assert resolver.streaming_config.num_workers == 4

    def test_small_phi_array_analysis(self):
        """Test analysis of small phi angle arrays."""
        resolver = OptimizedModeResolver()

        # Test single angle
        single_angle = np.array([0.0])
        result = resolver.analyze_large_phi_array(single_angle, detailed_analysis=False)

        assert result["mode_suggestion"] == "static_isotropic"
        assert result["confidence"] > 0.9

        # Test multiple angles
        multi_angles = np.array([0.0, np.pi / 2, np.pi])
        result = resolver.analyze_large_phi_array(multi_angles, detailed_analysis=True)

        assert result["mode_suggestion"] in ["static_anisotropic", "laminar_flow"]
        assert "processing_time_ms" in result

    @pytest.mark.performance
    def test_large_phi_array_performance(self):
        """Test performance with very large phi angle arrays."""
        resolver = OptimizedModeResolver()

        # Create large phi angle array
        large_array = np.random.uniform(0, 2 * np.pi, size=1000000)  # 1M angles

        start_time = time.perf_counter()
        result = resolver.analyze_large_phi_array(large_array, detailed_analysis=False)
        processing_time = time.perf_counter() - start_time

        assert processing_time < 5.0  # Should complete in < 5 seconds
        assert result["mode_suggestion"] is not None
        assert "processing_time_ms" in result

        print(
            f"Large array analysis: {len(large_array):,} angles in {processing_time:.3f}s"
        )

    def test_batch_mode_resolution(self):
        """Test batch mode resolution."""
        resolver = OptimizedModeResolver()

        # Create test configuration-data pairs
        config_data_pairs = [
            (
                {"analysis_mode": "auto-detect"},
                {"phi_angles": np.random.uniform(0, 2 * np.pi, size=1000)},
            )
            for _ in range(10)
        ]

        start_time = time.perf_counter()
        results = resolver.batch_resolve_modes(config_data_pairs, max_workers=2)
        batch_time = time.perf_counter() - start_time

        assert len(results) == 10
        assert batch_time < 2.0  # Should complete batch in < 2 seconds

        # Verify all results are valid
        for mode, metrics in results:
            assert mode in ["static_isotropic", "static_anisotropic", "laminar_flow"]
            assert isinstance(metrics, AnalysisMetrics)

    def test_memory_optimization(self):
        """Test memory optimization during large dataset processing."""
        resolver = OptimizedModeResolver()

        initial_memory = resolver.optimize_memory_usage()

        # Process large dataset
        very_large_array = np.random.uniform(0, 2 * np.pi, size=5000000)  # 5M angles
        result = resolver.analyze_large_phi_array(very_large_array)

        # Check memory optimization
        final_stats = resolver.optimize_memory_usage()
        assert (
            final_stats["memory_freed_mb"] >= 0
        )  # Should not increase memory usage significantly


class TestLazyValidator:
    """Test suite for lazy validation system."""

    def test_validation_level_configuration(self):
        """Test validation level configuration."""
        validator = LazyValidator(validation_level=ValidationLevel.FAST)

        assert validator.validation_level == ValidationLevel.FAST

        # Test level change
        validator.set_validation_level(ValidationLevel.THOROUGH)
        assert validator.validation_level == ValidationLevel.THOROUGH

    def test_task_registration(self):
        """Test validation task registration."""
        validator = LazyValidator()

        def dummy_validator(config):
            return Mock(
                is_valid=True,
                errors=[],
                warnings=[],
                suggestions=[],
                info=[],
                hardware_info={},
            )

        task = ValidationTask(
            name="test_task",
            validator=dummy_validator,
            priority=ValidationPriority.HIGH,
            estimated_time_ms=100.0,
        )

        validator.register_task(task)
        assert "test_task" in validator._registered_tasks

    def test_sync_validation(self):
        """Test synchronous validation."""
        validator = LazyValidator(validation_level=ValidationLevel.FAST)

        test_config = {
            "analysis_mode": "static_isotropic",
            "optimization": {"vi": {}, "mcmc": {}},
        }

        result = validator.validate_sync(test_config, skip_optional=True)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "warnings")
        assert hasattr(result, "tasks_executed")

    @pytest.mark.asyncio
    async def test_async_validation(self):
        """Test asynchronous validation."""
        validator = LazyValidator(validation_level=ValidationLevel.STANDARD)

        test_config = {
            "analysis_mode": "laminar_flow",
            "optimization": {"vi": {"n_iterations": 1000}, "mcmc": {"n_samples": 500}},
        }

        result = await validator.validate_async(test_config)

        assert hasattr(result, "is_valid")
        assert hasattr(result, "validation_time_ms")
        assert result.validation_time_ms > 0

    def test_on_demand_validation(self):
        """Test on-demand task validation."""
        validator = LazyValidator()

        test_config = {"analysis_mode": "static_anisotropic"}

        result = validator.validate_on_demand(test_config, "analysis_mode")

        assert hasattr(result, "is_valid")
        # Should be fast for simple validation
        assert hasattr(result, "errors")

    def test_validation_estimates(self):
        """Test validation time estimates."""
        validator = LazyValidator()

        estimates = validator.get_task_estimate()

        assert isinstance(estimates, dict)
        assert len(estimates) > 0

        # All estimates should be positive
        for task_name, estimate_ms in estimates.items():
            assert estimate_ms > 0


class TestParallelValidator:
    """Test suite for parallel validation system."""

    def test_worker_configuration(self):
        """Test worker configuration."""
        config = WorkerConfig(worker_type="thread", num_workers=4, max_workers=8)

        validator = ParallelValidator(config)
        assert validator.worker_config.num_workers == 4
        assert validator.worker_config.worker_type == "thread"

    def test_parallel_configuration_validation(self):
        """Test parallel validation of multiple configurations."""
        validator = ParallelValidator()

        # Create test configurations
        test_configs = [
            {"analysis_mode": "static_isotropic", "config_id": i} for i in range(20)
        ]

        start_time = time.perf_counter()
        results = list(
            validator.validate_configurations(
                test_configs,
                validation_level=ValidationLevel.FAST,
                batch_size=5,
                enable_streaming=False,
            )
        )
        parallel_time = time.perf_counter() - start_time

        assert len(results) == 20
        assert parallel_time < 5.0  # Should complete in reasonable time

        # Verify all results
        for config, result in results:
            assert "config_id" in config
            assert hasattr(result, "is_valid")

    @pytest.mark.performance
    def test_parallel_performance_vs_serial(self):
        """Test parallel vs serial performance."""
        parallel_validator = ParallelValidator(
            WorkerConfig(num_workers=4, worker_type="thread")
        )

        serial_validator = ParallelValidator(
            WorkerConfig(num_workers=1, worker_type="thread")
        )

        test_configs = [
            {"analysis_mode": "laminar_flow", "config_id": i}
            for i in range(50)  # Enough configs to see parallel benefit
        ]

        # Test parallel processing
        start_time = time.perf_counter()
        parallel_results = list(
            parallel_validator.validate_configurations(
                test_configs, validation_level=ValidationLevel.FAST
            )
        )
        parallel_time = time.perf_counter() - start_time

        # Test serial processing
        start_time = time.perf_counter()
        serial_results = list(
            serial_validator.validate_configurations(
                test_configs, validation_level=ValidationLevel.FAST
            )
        )
        serial_time = time.perf_counter() - start_time

        # Parallel should be faster (with some tolerance for overhead)
        speedup_ratio = serial_time / parallel_time
        assert speedup_ratio > 1.5  # Should be at least 1.5x faster

        print(
            f"Parallel speedup: {speedup_ratio:.2f}x ({serial_time:.3f}s → {parallel_time:.3f}s)"
        )

    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        validator = ParallelValidator()

        estimates = validator.estimate_processing_time(
            num_configs=100, validation_level=ValidationLevel.STANDARD
        )

        assert "total_configs" in estimates
        assert "estimated_total_time_sec" in estimates
        assert estimates["total_configs"] == 100
        assert estimates["estimated_total_time_sec"] > 0

    def test_worker_optimization(self):
        """Test dynamic worker optimization."""
        validator = ParallelValidator()
        initial_workers = validator.worker_config.num_workers

        # Test worker count optimization
        new_worker_count = validator.optimize_worker_count(target_throughput=10.0)

        assert isinstance(new_worker_count, int)
        assert new_worker_count >= validator.worker_config.min_workers
        assert new_worker_count <= validator.worker_config.max_workers


class TestMemoryOptimizer:
    """Test suite for memory optimization system."""

    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor(alert_threshold=0.8, critical_threshold=0.9)

        assert monitor.alert_threshold == 0.8
        assert monitor.critical_threshold == 0.9
        assert not monitor._monitoring

    def test_memory_statistics(self):
        """Test memory statistics collection."""
        monitor = MemoryMonitor()

        stats = monitor.get_current_stats()

        assert stats.total_memory_gb > 0
        assert stats.available_memory_gb > 0
        assert stats.process_memory_gb > 0
        assert 0 <= stats.memory_utilization <= 1.0

    def test_garbage_collection_optimization(self):
        """Test garbage collection optimization."""
        monitor = MemoryMonitor()

        # Create some objects to collect
        large_objects = [list(range(10000)) for _ in range(100)]
        del large_objects

        freed_objects = monitor.force_gc()
        assert freed_objects >= 0

        optimization_stats = monitor.optimize_memory()
        assert "memory_freed_gb" in optimization_stats
        assert "freed_objects" in optimization_stats

    def test_streaming_processor_initialization(self):
        """Test streaming processor initialization."""
        config = MemStreamingConfig(
            chunk_size=1000, max_memory_usage_gb=2.0, adaptive_sizing=True
        )

        processor = StreamingProcessor(config)
        assert processor.config.chunk_size == 1000
        assert processor.config.max_memory_usage_gb == 2.0
        assert processor.config.adaptive_sizing is True

    def test_streaming_configuration_processing(self):
        """Test streaming configuration processing."""
        processor = StreamingProcessor()

        def simple_processor(configs):
            return [{"processed": True, "config_id": c.get("id")} for c in configs]

        test_configs = [{"id": i, "data": f"config_{i}"} for i in range(100)]

        results = list(processor.stream_configurations(test_configs, simple_processor))

        assert len(results) == 100
        for result in results:
            assert result["processed"] is True
            assert "config_id" in result

    @pytest.mark.performance
    def test_memory_efficient_large_dataset(self):
        """Test memory-efficient processing of large datasets."""
        processor = StreamingProcessor(
            MemStreamingConfig(chunk_size=1000, max_memory_usage_gb=1.0)
        )

        def memory_intensive_processor(configs):
            # Simulate memory-intensive processing
            processed = []
            for config in configs:
                # Create temporary large data structure
                temp_data = list(range(1000))
                processed.append(
                    {"config_id": config.get("id"), "result": len(temp_data)}
                )
            return processed

        # Create large dataset
        large_configs = [{"id": i} for i in range(10000)]

        start_time = time.perf_counter()
        initial_memory = processor.monitor.get_current_stats().process_memory_gb

        results = list(
            processor.stream_configurations(large_configs, memory_intensive_processor)
        )

        final_memory = processor.monitor.get_current_stats().process_memory_gb
        processing_time = time.perf_counter() - start_time

        assert len(results) == 10000
        assert processing_time < 30.0  # Should complete in reasonable time

        # Memory usage should not grow excessively
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1.0  # Should not grow by more than 1GB

        print(
            f"Large dataset processing: {len(large_configs):,} configs in {processing_time:.3f}s, "
            f"memory growth: {memory_growth:.3f}GB"
        )


class TestPerformanceProfiler:
    """Test suite for performance profiling system."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(
            enable_memory_profiling=True, enable_line_profiling=False
        )

        assert profiler.enable_memory_profiling is True
        assert profiler.enable_line_profiling is False

    def test_operation_profiling(self):
        """Test operation profiling."""
        profiler = PerformanceProfiler()

        # Profile a simple operation
        with profiler.profile_operation("test_operation", {"test": True}):
            time.sleep(0.1)  # Simulate work
            result = sum(range(1000))  # Some computation

        # Check metrics were collected
        assert len(profiler._metrics_history) > 0

        latest_metrics = profiler._metrics_history[-1]
        assert latest_metrics.operation_name == "test_operation"
        assert latest_metrics.execution_time_ms >= 100  # At least 100ms due to sleep
        assert latest_metrics.function_calls > 0

    def test_function_profiling(self):
        """Test function-specific profiling."""
        profiler = PerformanceProfiler()

        def test_function(n):
            return sum(range(n))

        result, metrics = profiler.profile_function(test_function, 10000)

        assert result == sum(range(10000))
        assert metrics is not None
        assert metrics.execution_time_ms > 0

    def test_bottleneck_identification(self):
        """Test bottleneck identification."""
        profiler = PerformanceProfiler()

        # Create some slow operations
        with profiler.profile_operation("slow_operation"):
            time.sleep(0.5)  # Simulate slow operation

        with profiler.profile_operation("fast_operation"):
            time.sleep(0.01)  # Simulate fast operation

        bottlenecks = profiler.identify_bottlenecks()

        # Should identify the slow operation as a bottleneck
        slow_bottlenecks = [b for b in bottlenecks if "slow" in b.get("operation", "")]
        assert len(slow_bottlenecks) > 0

    def test_baseline_creation_and_comparison(self):
        """Test performance baseline management."""
        profiler = PerformanceProfiler()

        # Create some baseline operations
        for i in range(5):
            with profiler.profile_operation("baseline_operation"):
                time.sleep(0.1)
                result = sum(range(1000))

        # Create baseline
        profiler.create_baseline("test_baseline")
        assert "test_baseline" in profiler._baselines

        # Run more operations (simulate regression)
        for i in range(3):
            with profiler.profile_operation("baseline_operation"):
                time.sleep(0.2)  # Slower operations
                result = sum(range(2000))

        # Compare to baseline
        comparison = profiler.compare_to_baseline("test_baseline")

        assert "baseline_name" in comparison
        assert "operation_comparisons" in comparison

        if "baseline_operation" in comparison["operation_comparisons"]:
            op_comparison = comparison["operation_comparisons"]["baseline_operation"]
            # Should detect regression (slower execution)
            assert op_comparison.get("time_regression_percent", 0) > 0

    def test_performance_report_generation(self):
        """Test comprehensive performance report generation."""
        profiler = PerformanceProfiler()

        # Generate some activity
        operations = ["config_validation", "mode_resolution", "parameter_checking"]
        for op in operations:
            with profiler.profile_operation(op):
                time.sleep(0.05)
                result = sum(range(500))

        # Generate report
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            report_file = Path(f.name)

        try:
            report = profiler.generate_performance_report(
                output_file=report_file, include_detailed_analysis=True
            )

            # Verify report structure
            assert "operation_statistics" in report
            assert "identified_bottlenecks" in report
            assert "optimization_recommendations" in report
            assert "detailed_analysis" in report

            # Verify file was created
            assert report_file.exists()

        finally:
            if report_file.exists():
                report_file.unlink()


class TestIntegrationScenarios:
    """Integration tests for combined performance optimizations."""

    @pytest.mark.performance
    @pytest.mark.integration
    def test_end_to_end_performance_scenario(self):
        """Test end-to-end performance with all optimizations enabled."""
        # Initialize all components
        cache = PerformanceCache(max_memory_mb=256, max_entries=1000)
        resolver = OptimizedModeResolver()
        validator = LazyValidator(validation_level=ValidationLevel.STANDARD)
        parallel_validator = ParallelValidator()
        profiler = PerformanceProfiler()

        # Create realistic test scenario
        test_configs = []
        for i in range(100):
            config = {
                "analysis_mode": "auto-detect",
                "config_id": i,
                "analyzer_parameters": {
                    "temporal": {"dt": 0.001, "start_frame": 1, "end_frame": 1000},
                    "scattering": {"wavevector_q": 0.0054},
                },
                "optimization": {
                    "vi": {"n_iterations": 1000, "learning_rate": 0.01},
                    "mcmc": {"n_samples": 500, "n_chains": 2},
                },
            }
            test_configs.append(config)

        # Profile the entire end-to-end process
        with profiler.profile_operation(
            "end_to_end_processing", {"num_configs": len(test_configs)}
        ):
            # Process configurations in parallel with all optimizations
            results = list(
                parallel_validator.validate_configurations(
                    test_configs,
                    validation_level=ValidationLevel.STANDARD,
                    batch_size=10,
                )
            )

        # Verify results
        assert len(results) == len(test_configs)

        # Generate performance report
        report = profiler.generate_performance_report(include_detailed_analysis=False)

        # Performance assertions
        end_to_end_stats = None
        for op_name, op_stats in report["operation_statistics"].items():
            if "end_to_end" in op_name:
                end_to_end_stats = op_stats
                break

        if end_to_end_stats:
            # Should complete processing in reasonable time
            mean_time = end_to_end_stats.get("execution_time_ms_mean", float("inf"))
            assert mean_time < 30000  # Less than 30 seconds for 100 configs

            print(
                f"End-to-end performance: {len(test_configs)} configs processed in {mean_time:.0f}ms"
            )

        # Check cache effectiveness
        cache_stats = cache.get_stats()
        print(
            f"Cache performance: {cache_stats.hits} hits, {cache_stats.misses} misses, "
            f"hit rate: {cache_stats.hit_rate:.2%}"
        )

    @pytest.mark.stress
    def test_stress_test_large_scale(self):
        """Stress test with large-scale processing."""
        # Create stress test configuration
        streaming_config = MemStreamingConfig(
            chunk_size=500, max_memory_usage_gb=2.0, adaptive_sizing=True
        )

        processor = StreamingProcessor(streaming_config)

        # Create large-scale test data
        large_configs = [
            {
                "config_id": i,
                "analysis_mode": [
                    "static_isotropic",
                    "static_anisotropic",
                    "laminar_flow",
                ][i % 3],
                "data_size": 10000 + (i % 5000),  # Variable data sizes
                "phi_angles": list(
                    np.random.uniform(0, 2 * np.pi, size=100 + (i % 50))
                ),
            }
            for i in range(10000)  # 10K configurations
        ]

        def stress_processor(configs):
            results = []
            for config in configs:
                # Simulate intensive processing
                phi_angles = config.get("phi_angles", [])
                if len(phi_angles) > 75:
                    suggested_mode = "laminar_flow"
                elif len(phi_angles) > 25:
                    suggested_mode = "static_anisotropic"
                else:
                    suggested_mode = "static_isotropic"

                results.append(
                    {
                        "config_id": config["config_id"],
                        "suggested_mode": suggested_mode,
                        "processed": True,
                    }
                )
            return results

        start_time = time.perf_counter()
        initial_memory = processor.monitor.get_current_stats().process_memory_gb

        with processor.memory_managed_processing():
            results = list(
                processor.stream_configurations(large_configs, stress_processor)
            )

        final_memory = processor.monitor.get_current_stats().process_memory_gb
        processing_time = time.perf_counter() - start_time
        memory_growth = final_memory - initial_memory

        # Stress test assertions
        assert len(results) == 10000
        assert processing_time < 120.0  # Should complete in < 2 minutes
        assert memory_growth < 2.0  # Should not grow memory excessively

        throughput = len(results) / processing_time

        print(
            f"Stress test results: {len(results):,} configs in {processing_time:.1f}s "
            f"({throughput:.1f} configs/sec), memory growth: {memory_growth:.3f}GB"
        )

        # Should achieve reasonable throughput
        assert throughput > 50  # At least 50 configs per second


# Benchmark fixtures
@pytest.fixture
def benchmark_configs():
    """Generate benchmark configurations."""
    return [
        {
            "config_id": i,
            "analysis_mode": "auto-detect",
            "analyzer_parameters": {
                "temporal": {
                    "dt": 0.001,
                    "start_frame": 1,
                    "end_frame": 1000 + i * 100,
                },
                "scattering": {"wavevector_q": 0.005 + i * 0.0001},
            },
            "optimization": {
                "vi": {"n_iterations": 1000 + i * 100},
                "mcmc": {"n_samples": 500 + i * 50},
            },
        }
        for i in range(50)
    ]


# Performance test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.timeout(300),  # 5 minute timeout for performance tests
]


if __name__ == "__main__":
    # Run specific test suites
    import sys

    if len(sys.argv) > 1:
        test_suite = sys.argv[1]
        if test_suite == "cache":
            pytest.main(["-v", "TestPerformanceCache"])
        elif test_suite == "resolver":
            pytest.main(["-v", "TestOptimizedModeResolver"])
        elif test_suite == "validator":
            pytest.main(["-v", "TestLazyValidator"])
        elif test_suite == "parallel":
            pytest.main(["-v", "TestParallelValidator"])
        elif test_suite == "memory":
            pytest.main(["-v", "TestMemoryOptimizer"])
        elif test_suite == "profiler":
            pytest.main(["-v", "TestPerformanceProfiler"])
        elif test_suite == "integration":
            pytest.main(["-v", "TestIntegrationScenarios"])
        elif test_suite == "stress":
            pytest.main(
                ["-v", "TestIntegrationScenarios::test_stress_test_large_scale"]
            )
        else:
            print(f"Unknown test suite: {test_suite}")
            sys.exit(1)
    else:
        # Run all tests
        pytest.main(["-v", __file__])
