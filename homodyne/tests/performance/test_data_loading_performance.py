"""
Performance Validation Test Suite for Enhanced Data Loading System
==================================================================

Comprehensive performance testing and benchmarking for the enhanced data loading
system. Validates performance improvements, memory optimization, scalability,
and regression testing.

Key Testing Areas:
- Memory usage profiling for large datasets  
- Timing benchmarks for all major operations
- Scalability validation across different dataset sizes
- Resource utilization monitoring (CPU, memory, I/O)
- Performance regression testing
- Cache effectiveness validation
"""

import os
import sys
import pytest
import numpy as np
import time
import psutil
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import threading
import concurrent.futures
import gc
from contextlib import contextmanager
import warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Performance monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Memory profiling
try:
    from memory_profiler import profile
    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False

# Test data generator
from homodyne.tests.data.synthetic_data_generator import (
    SyntheticDataGenerator, SyntheticDatasetConfig, 
    DatasetSize, DataQuality, DatasetFormat,
    generate_test_dataset_suite
)

# Core homodyne imports
try:
    from homodyne.data.xpcs_loader import XPCSLoader
    from homodyne.data.filtering_utils import FilteringEngine, FilterConfig
    from homodyne.data.preprocessing import PreprocessingPipeline, PreprocessingConfig
    from homodyne.data.quality_controller import QualityController, QualityControlConfig
    from homodyne.data.performance_engine import PerformanceEngine, PerformanceConfig, OptimizationLevel
    from homodyne.data.memory_manager import MemoryManager, MemoryConfig, MemoryStrategy
    HOMODYNE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import homodyne modules: {e}")
    HOMODYNE_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    operation_name: str
    execution_time: float
    peak_memory_mb: float
    average_memory_mb: float
    cpu_percent: float
    io_read_mb: float
    io_write_mb: float
    dataset_size_mb: float
    throughput_mbps: float = field(init=False)
    
    def __post_init__(self):
        if self.execution_time > 0 and self.dataset_size_mb > 0:
            self.throughput_mbps = self.dataset_size_mb / self.execution_time
        else:
            self.throughput_mbps = 0.0


@contextmanager
def performance_monitor(operation_name: str, dataset_size_mb: float = 0.0):
    """Context manager for monitoring performance metrics"""
    if not HAS_PSUTIL:
        yield None
        return
        
    # Initial measurements
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024**2  # MB
    initial_io = process.io_counters()
    start_time = time.time()
    
    # Monitor during execution
    memory_samples = []
    cpu_samples = []
    
    def monitor_thread():
        while not monitor_thread.stop_event.is_set():
            try:
                memory_samples.append(process.memory_info().rss / 1024**2)
                cpu_samples.append(process.cpu_percent())
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
    
    monitor_thread.stop_event = threading.Event()
    monitor_thread_obj = threading.Thread(target=monitor_thread)
    monitor_thread_obj.start()
    
    try:
        yield None
    finally:
        # Stop monitoring
        end_time = time.time()
        monitor_thread.stop_event.set()
        monitor_thread_obj.join(timeout=1.0)
        
        # Final measurements
        final_io = process.io_counters()
        execution_time = end_time - start_time
        
        # Calculate metrics
        peak_memory = max(memory_samples) if memory_samples else initial_memory
        avg_memory = np.mean(memory_samples) if memory_samples else initial_memory
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0.0
        
        io_read_mb = (final_io.read_bytes - initial_io.read_bytes) / 1024**2
        io_write_mb = (final_io.write_bytes - initial_io.write_bytes) / 1024**2
        
        # Store results in global registry (for test access)
        if not hasattr(performance_monitor, 'results'):
            performance_monitor.results = {}
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            cpu_percent=avg_cpu,
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
            dataset_size_mb=dataset_size_mb
        )
        
        performance_monitor.results[operation_name] = metrics


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    @classmethod
    def setup_class(cls):
        """Set up performance test environment"""
        if not HAS_PSUTIL or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_performance_test_"))
        
        # Generate performance test datasets
        print("Generating performance test datasets...")
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
        
        # Performance baselines (these would be updated over time)
        cls.performance_baselines = {
            'load_small_dataset': {'time': 1.0, 'memory': 100},      # 1s, 100MB
            'load_medium_dataset': {'time': 5.0, 'memory': 500},     # 5s, 500MB  
            'load_large_dataset': {'time': 20.0, 'memory': 2000},    # 20s, 2GB
            'filter_small_dataset': {'time': 0.5, 'memory': 50},     # 0.5s, 50MB
            'preprocess_small_dataset': {'time': 2.0, 'memory': 200}, # 2s, 200MB
            'full_pipeline_small': {'time': 5.0, 'memory': 300},     # 5s, 300MB
        }
        
        # Clear any existing performance results
        if hasattr(performance_monitor, 'results'):
            performance_monitor.results.clear()
    
    @classmethod
    def teardown_class(cls):
        """Clean up performance test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_data_loading_performance(self):
        """Benchmark data loading performance across different sizes"""
        test_cases = [
            ('small', 'integration_small'),
            ('medium', 'performance_medium'), 
            ('large', 'performance_large')
        ]
        
        results = {}
        
        for size_name, dataset_key in test_cases:
            if dataset_key not in self.test_datasets:
                print(f"Skipping {size_name} dataset - not available")
                continue
                
            dataset_path = self.test_datasets[dataset_key]
            dataset_size_mb = dataset_path.stat().st_size / 1024**2
            
            operation_name = f"load_{size_name}_dataset"
            
            with performance_monitor(operation_name, dataset_size_mb):
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                # Verify data was loaded
                assert data is not None
                assert 'g2' in data
            
            # Check performance against baseline
            metrics = performance_monitor.results[operation_name]
            baseline = self.performance_baselines.get(operation_name, {})
            
            if 'time' in baseline:
                time_ratio = metrics.execution_time / baseline['time']
                assert time_ratio < 2.0, f"Loading {size_name} dataset too slow: {metrics.execution_time:.2f}s vs {baseline['time']}s baseline"
                
            if 'memory' in baseline:
                memory_ratio = metrics.peak_memory_mb / baseline['memory']
                assert memory_ratio < 2.0, f"Loading {size_name} dataset uses too much memory: {metrics.peak_memory_mb:.1f}MB vs {baseline['memory']}MB baseline"
            
            results[size_name] = metrics
            print(f"✓ {size_name.capitalize()} dataset: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB, {metrics.throughput_mbps:.1f}MB/s")
        
        # Verify scalability - loading time should scale sub-linearly with size
        if 'small' in results and 'large' in results:
            small_metrics = results['small']
            large_metrics = results['large']
            
            size_ratio = large_metrics.dataset_size_mb / small_metrics.dataset_size_mb
            time_ratio = large_metrics.execution_time / small_metrics.execution_time
            
            # Time should scale better than O(n) - allow up to O(n^1.5)
            scaling_factor = time_ratio / (size_ratio ** 1.5)
            assert scaling_factor < 1.0, f"Loading time scales poorly: {time_ratio:.2f}x time for {size_ratio:.2f}x size"
            
            print(f"✓ Loading scalability: {time_ratio:.2f}x time for {size_ratio:.2f}x size")
    
    def test_filtering_performance(self):
        """Benchmark filtering performance and efficiency"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        loader = XPCSLoader(dataset_path)
        data = loader.load_data()
        
        # Test different filtering configurations
        filter_configs = [
            ('basic_filtering', FilterConfig()),
            ('aggressive_filtering', FilterConfig(
                q_range={'min': 1e-4, 'max': 5e-3},
                quality_threshold=0.9,
                enable_parallel=True
            )),
            ('quality_filtering', FilterConfig(
                quality_threshold=0.95,
                enable_advanced_quality=True
            ))
        ]
        
        for config_name, filter_config in filter_configs:
            operation_name = f"filter_{config_name}"
            
            with performance_monitor(operation_name, dataset_size_mb):
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                assert filtered_data is not None
                assert 'g2' in filtered_data
            
            metrics = performance_monitor.results[operation_name]
            
            # Filtering should be fast
            assert metrics.execution_time < 5.0, f"Filtering too slow: {metrics.execution_time:.2f}s"
            
            # Should not use excessive memory
            memory_overhead = metrics.peak_memory_mb / dataset_size_mb
            assert memory_overhead < 5.0, f"Filtering memory overhead too high: {memory_overhead:.1f}x"
            
            print(f"✓ {config_name}: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB")
    
    def test_preprocessing_performance(self):
        """Benchmark preprocessing pipeline performance"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        loader = XPCSLoader(dataset_path)
        data = loader.load_data()
        
        # Test different preprocessing configurations
        from homodyne.data.preprocessing import PreprocessingStage
        
        preprocessing_configs = [
            ('minimal', PreprocessingConfig(
                pipeline_stages=[PreprocessingStage.DIAGONAL_CORRECTION]
            )),
            ('standard', PreprocessingConfig(
                pipeline_stages=[
                    PreprocessingStage.DIAGONAL_CORRECTION,
                    PreprocessingStage.NORMALIZATION
                ]
            )),
            ('comprehensive', PreprocessingConfig(
                pipeline_stages=[
                    PreprocessingStage.DIAGONAL_CORRECTION,
                    PreprocessingStage.NORMALIZATION,
                    PreprocessingStage.NOISE_REDUCTION
                ],
                enable_chunked_processing=True
            ))
        ]
        
        for config_name, preprocessing_config in preprocessing_configs:
            operation_name = f"preprocess_{config_name}"
            
            with performance_monitor(operation_name, dataset_size_mb):
                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
                processed_data = preprocessing_pipeline.process_data(data)
                
                assert processed_data is not None
                assert 'g2' in processed_data
            
            metrics = performance_monitor.results[operation_name]
            
            # Preprocessing performance should be reasonable
            stage_count = len(preprocessing_config.pipeline_stages)
            max_time = stage_count * 2.0  # 2 seconds per stage
            assert metrics.execution_time < max_time, f"Preprocessing too slow: {metrics.execution_time:.2f}s for {stage_count} stages"
            
            print(f"✓ {config_name} preprocessing: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB")
    
    def test_memory_optimization_effectiveness(self):
        """Test memory optimization strategies"""
        if 'performance_medium' not in self.test_datasets:
            pytest.skip("Medium dataset not available")
            
        dataset_path = self.test_datasets['performance_medium']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        # Test different memory strategies
        memory_strategies = [
            ('conservative', MemoryStrategy.CONSERVATIVE),
            ('balanced', MemoryStrategy.BALANCED),
            ('aggressive', MemoryStrategy.AGGRESSIVE)
        ]
        
        results = {}
        
        for strategy_name, strategy in memory_strategies:
            operation_name = f"memory_{strategy_name}"
            
            memory_config = MemoryConfig(
                strategy=strategy,
                enable_memory_mapping=True,
                max_memory_usage=1024**3  # 1GB limit
            )
            
            with performance_monitor(operation_name, dataset_size_mb):
                memory_manager = MemoryManager(memory_config)
                
                with memory_manager.memory_context():
                    loader = XPCSLoader(dataset_path)
                    data = loader.load_data()
                    
                    # Apply some processing to test memory management
                    filter_config = FilterConfig()
                    filtering_engine = FilteringEngine(filter_config)
                    filtered_data = filtering_engine.apply_filtering(data)
                    
                    assert filtered_data is not None
            
            metrics = performance_monitor.results[operation_name]
            results[strategy_name] = metrics
            
            # Should respect memory limits
            memory_limit_mb = 1024  # 1GB
            assert metrics.peak_memory_mb < memory_limit_mb * 1.2, f"{strategy_name} strategy exceeded memory limit: {metrics.peak_memory_mb:.1f}MB"
            
            print(f"✓ {strategy_name} strategy: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB")
        
        # Conservative should use least memory, aggressive should be fastest
        if 'conservative' in results and 'aggressive' in results:
            conservative = results['conservative']
            aggressive = results['aggressive']
            
            # Conservative should use less memory
            assert conservative.peak_memory_mb <= aggressive.peak_memory_mb, "Conservative strategy should use less memory"
            
            # Aggressive should be faster (or at least not significantly slower)
            time_ratio = aggressive.execution_time / conservative.execution_time
            assert time_ratio < 1.5, f"Aggressive strategy should not be much slower: {time_ratio:.2f}x"
            
            print(f"✓ Memory strategy comparison: Conservative={conservative.peak_memory_mb:.1f}MB, Aggressive={aggressive.peak_memory_mb:.1f}MB")
    
    def test_parallel_processing_scalability(self):
        """Test parallel processing performance scaling"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        loader = XPCSLoader(dataset_path)
        data = loader.load_data()
        
        # Test different parallel configurations
        thread_counts = [1, 2, 4]  # Single, dual, quad
        results = {}
        
        for thread_count in thread_counts:
            operation_name = f"parallel_{thread_count}threads"
            
            with performance_monitor(operation_name, dataset_size_mb):
                filter_config = FilterConfig(
                    enable_parallel=thread_count > 1,
                    max_threads=thread_count
                )
                filtering_engine = FilteringEngine(filter_config)
                
                preprocessing_config = PreprocessingConfig(
                    enable_parallel_processing=thread_count > 1,
                    max_threads=thread_count
                )
                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
                
                # Apply both filtering and preprocessing
                filtered_data = filtering_engine.apply_filtering(data)
                processed_data = preprocessing_pipeline.process_data(filtered_data)
                
                assert processed_data is not None
            
            metrics = performance_monitor.results[operation_name]
            results[thread_count] = metrics
            
            print(f"✓ {thread_count} threads: {metrics.execution_time:.2f}s, CPU: {metrics.cpu_percent:.1f}%")
        
        # Verify parallel processing provides benefit
        if 1 in results and 4 in results:
            single_thread = results[1]
            quad_thread = results[4]
            
            speedup = single_thread.execution_time / quad_thread.execution_time
            efficiency = speedup / 4.0  # Efficiency = speedup / thread_count
            
            # Should see some speedup (at least 20% efficiency)
            assert efficiency > 0.2, f"Parallel processing efficiency too low: {efficiency:.2f}"
            
            print(f"✓ Parallel speedup: {speedup:.2f}x with 4 threads (efficiency: {efficiency:.2f})")
    
    def test_cache_effectiveness(self):
        """Test caching system effectiveness"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        # First load (cold cache)
        operation_name_cold = "cache_cold_load"
        with performance_monitor(operation_name_cold, dataset_size_mb):
            loader = XPCSLoader(dataset_path)
            loader.clear_cache()  # Ensure cold start
            data_cold = loader.load_data()
            assert data_cold is not None
        
        # Second load (warm cache)
        operation_name_warm = "cache_warm_load"  
        with performance_monitor(operation_name_warm, dataset_size_mb):
            loader = XPCSLoader(dataset_path)
            data_warm = loader.load_data()
            assert data_warm is not None
        
        cold_metrics = performance_monitor.results[operation_name_cold]
        warm_metrics = performance_monitor.results[operation_name_warm]
        
        # Warm load should be significantly faster
        speedup = cold_metrics.execution_time / warm_metrics.execution_time
        assert speedup > 2.0, f"Cache speedup insufficient: {speedup:.2f}x"
        
        # Warm load should use less I/O
        io_reduction = (cold_metrics.io_read_mb - warm_metrics.io_read_mb) / cold_metrics.io_read_mb
        assert io_reduction > 0.5, f"Cache I/O reduction insufficient: {io_reduction:.2f}"
        
        print(f"✓ Cache effectiveness: {speedup:.2f}x speedup, {io_reduction:.1%} I/O reduction")
    
    def test_full_pipeline_performance(self):
        """Benchmark full pipeline end-to-end performance"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        operation_name = "full_pipeline_small"
        
        with performance_monitor(operation_name, dataset_size_mb):
            # Complete pipeline
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            
            filter_config = FilterConfig(enable_parallel=True)
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            
            from homodyne.data.preprocessing import PreprocessingStage
            preprocessing_config = PreprocessingConfig(
                pipeline_stages=[
                    PreprocessingStage.DIAGONAL_CORRECTION,
                    PreprocessingStage.NORMALIZATION
                ]
            )
            preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            processed_data = preprocessing_pipeline.process_data(filtered_data)
            
            qc_config = QualityControlConfig()
            quality_controller = QualityController(qc_config)
            final_data, quality_report = quality_controller.assess_and_improve_data(processed_data)
            
            assert final_data is not None
            assert quality_report is not None
        
        metrics = performance_monitor.results[operation_name]
        baseline = self.performance_baselines[operation_name]
        
        # Check against baseline
        time_ratio = metrics.execution_time / baseline['time']
        memory_ratio = metrics.peak_memory_mb / baseline['memory']
        
        assert time_ratio < 1.5, f"Full pipeline too slow: {metrics.execution_time:.2f}s vs {baseline['time']}s baseline"
        assert memory_ratio < 1.5, f"Full pipeline uses too much memory: {metrics.peak_memory_mb:.1f}MB vs {baseline['memory']}MB baseline"
        
        print(f"✓ Full pipeline: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB, {metrics.throughput_mbps:.1f}MB/s")


class TestScalabilityValidation:
    """Scalability testing across different dataset sizes and conditions"""
    
    @classmethod
    def setup_class(cls):
        """Set up scalability tests"""
        if not HAS_PSUTIL or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_scalability_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
    
    @classmethod 
    def teardown_class(cls):
        """Clean up scalability test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_memory_scalability_large_datasets(self):
        """Test memory usage scaling with large datasets"""
        available_sizes = []
        for size in ['small', 'medium', 'large']:
            dataset_key = f"performance_{size}" if size != 'small' else 'integration_small'
            if dataset_key in self.test_datasets:
                available_sizes.append((size, dataset_key))
        
        if len(available_sizes) < 2:
            pytest.skip("Insufficient datasets for scalability testing")
        
        memory_results = {}
        
        for size_name, dataset_key in available_sizes:
            dataset_path = self.test_datasets[dataset_key]
            dataset_size_mb = dataset_path.stat().st_size / 1024**2
            
            operation_name = f"scalability_memory_{size_name}"
            
            with performance_monitor(operation_name, dataset_size_mb):
                memory_config = MemoryConfig(
                    strategy=MemoryStrategy.BALANCED,
                    enable_memory_mapping=True
                )
                memory_manager = MemoryManager(memory_config)
                
                with memory_manager.memory_context():
                    loader = XPCSLoader(dataset_path)
                    data = loader.load_data()
                    
                    # Apply basic processing
                    filter_config = FilterConfig()
                    filtering_engine = FilteringEngine(filter_config)
                    filtered_data = filtering_engine.apply_filtering(data)
            
            metrics = performance_monitor.results[operation_name]
            memory_efficiency = dataset_size_mb / metrics.peak_memory_mb
            
            memory_results[size_name] = {
                'dataset_size_mb': dataset_size_mb,
                'peak_memory_mb': metrics.peak_memory_mb,
                'efficiency': memory_efficiency,
                'execution_time': metrics.execution_time
            }
            
            print(f"✓ {size_name}: {dataset_size_mb:.1f}MB dataset → {metrics.peak_memory_mb:.1f}MB memory (efficiency: {memory_efficiency:.2f})")
        
        # Verify memory efficiency doesn't degrade severely with size
        if len(memory_results) >= 2:
            sizes = sorted(memory_results.keys(), key=lambda x: memory_results[x]['dataset_size_mb'])
            smallest = memory_results[sizes[0]]
            largest = memory_results[sizes[-1]]
            
            efficiency_degradation = smallest['efficiency'] / largest['efficiency']
            assert efficiency_degradation < 3.0, f"Memory efficiency degrades too much: {efficiency_degradation:.2f}x"
            
            print(f"✓ Memory scalability: Efficiency degradation {efficiency_degradation:.2f}x from {sizes[0]} to {sizes[-1]}")
    
    def test_processing_time_scalability(self):
        """Test processing time scaling with dataset complexity"""
        complexity_configs = [
            ('simple', {
                'filtering': FilterConfig(),
                'preprocessing': PreprocessingConfig(
                    pipeline_stages=['diagonal_correction']
                )
            }),
            ('moderate', {
                'filtering': FilterConfig(quality_threshold=0.8),
                'preprocessing': PreprocessingConfig(
                    pipeline_stages=['diagonal_correction', 'normalization']
                )
            }),
            ('complex', {
                'filtering': FilterConfig(
                    quality_threshold=0.9,
                    enable_advanced_quality=True
                ),
                'preprocessing': PreprocessingConfig(
                    pipeline_stages=['diagonal_correction', 'normalization', 'noise_reduction']
                )
            })
        ]
        
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small'] 
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        timing_results = {}
        
        for complexity_name, configs in complexity_configs:
            operation_name = f"complexity_{complexity_name}"
            
            with performance_monitor(operation_name, dataset_size_mb):
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                filtering_engine = FilteringEngine(configs['filtering'])
                filtered_data = filtering_engine.apply_filtering(data)
                
                preprocessing_pipeline = PreprocessingPipeline(configs['preprocessing'])
                processed_data = preprocessing_pipeline.process_data(filtered_data)
                
                assert processed_data is not None
            
            metrics = performance_monitor.results[operation_name]
            timing_results[complexity_name] = metrics.execution_time
            
            print(f"✓ {complexity_name} complexity: {metrics.execution_time:.2f}s")
        
        # Verify time scaling is reasonable
        if len(timing_results) >= 2:
            simple_time = timing_results.get('simple', 0)
            complex_time = timing_results.get('complex', timing_results[max(timing_results.keys())])
            
            if simple_time > 0:
                complexity_overhead = complex_time / simple_time
                assert complexity_overhead < 5.0, f"Complexity overhead too high: {complexity_overhead:.2f}x"
                
                print(f"✓ Complexity scaling: {complexity_overhead:.2f}x overhead for complex processing")
    
    def test_concurrent_load_scalability(self):
        """Test scalability under concurrent load"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        concurrent_levels = [1, 2, 4]
        results = {}
        
        for num_concurrent in concurrent_levels:
            operation_name = f"concurrent_{num_concurrent}"
            
            def load_and_process():
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                filter_config = FilterConfig()
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                return filtered_data is not None
            
            with performance_monitor(operation_name, 0):
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                    futures = [executor.submit(load_and_process) for _ in range(num_concurrent)]
                    concurrent.futures.wait(futures)
                    
                    # Verify all completed successfully
                    assert all(future.result() for future in futures)
            
            metrics = performance_monitor.results[operation_name]
            results[num_concurrent] = metrics
            
            print(f"✓ {num_concurrent} concurrent: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB")
        
        # Verify concurrent performance doesn't degrade too severely
        if 1 in results and 4 in results:
            single = results[1]
            quad = results[4]
            
            time_degradation = quad.execution_time / single.execution_time
            memory_increase = quad.peak_memory_mb / single.peak_memory_mb
            
            # Should not be more than 2x slower or use more than 4x memory
            assert time_degradation < 2.0, f"Concurrent performance degrades too much: {time_degradation:.2f}x"
            assert memory_increase < 4.0, f"Concurrent memory usage too high: {memory_increase:.2f}x"
            
            print(f"✓ Concurrent scaling: {time_degradation:.2f}x time, {memory_increase:.2f}x memory for 4x concurrency")


class TestPerformanceRegression:
    """Performance regression testing"""
    
    PERFORMANCE_REGRESSION_TOLERANCE = 1.2  # Allow 20% performance degradation
    
    @classmethod
    def setup_class(cls):
        """Set up regression testing"""
        if not HAS_PSUTIL or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_regression_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
        
        # Load performance baselines (in real implementation, these would be from a file)
        cls.baseline_file = cls.test_dir / "performance_baselines.json"
        cls.load_baselines()
    
    @classmethod
    def teardown_class(cls):
        """Clean up and save updated baselines"""
        cls.save_baselines()
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def load_baselines(cls):
        """Load performance baselines"""
        import json
        
        # Default baselines (would be loaded from file in real implementation)
        cls.baselines = {
            'load_small_dataset_v2': {'time': 0.5, 'memory': 50, 'version': '2.0.0'},
            'filter_small_dataset_v2': {'time': 0.2, 'memory': 25, 'version': '2.0.0'},
            'preprocess_small_dataset_v2': {'time': 1.0, 'memory': 100, 'version': '2.0.0'},
            'full_pipeline_v2': {'time': 3.0, 'memory': 200, 'version': '2.0.0'}
        }
        
        if cls.baseline_file.exists():
            try:
                with open(cls.baseline_file, 'r') as f:
                    stored_baselines = json.load(f)
                cls.baselines.update(stored_baselines)
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")
    
    @classmethod
    def save_baselines(cls):
        """Save updated baselines"""
        import json
        
        if hasattr(performance_monitor, 'results'):
            # Update baselines with current results (if better)
            for operation_name, metrics in performance_monitor.results.items():
                baseline_key = f"{operation_name}_v2"
                
                current_baseline = cls.baselines.get(baseline_key, {})
                
                # Update if this is better performance or no baseline exists
                if (not current_baseline or 
                    metrics.execution_time < current_baseline.get('time', float('inf'))):
                    
                    cls.baselines[baseline_key] = {
                        'time': metrics.execution_time,
                        'memory': metrics.peak_memory_mb,
                        'version': '2.0.0',
                        'updated': time.time()
                    }
        
        try:
            with open(cls.baseline_file, 'w') as f:
                json.dump(cls.baselines, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baselines: {e}")
    
    def test_loading_regression(self):
        """Test for performance regression in data loading"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        operation_name = "load_small_dataset_v2"
        
        with performance_monitor(operation_name, dataset_size_mb):
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            assert data is not None
        
        metrics = performance_monitor.results[operation_name]
        baseline = self.baselines.get(operation_name, {})
        
        if baseline:
            time_ratio = metrics.execution_time / baseline['time']
            memory_ratio = metrics.peak_memory_mb / baseline['memory']
            
            assert time_ratio < self.PERFORMANCE_REGRESSION_TOLERANCE, \
                f"Loading time regression: {time_ratio:.2f}x slower than baseline"
            assert memory_ratio < self.PERFORMANCE_REGRESSION_TOLERANCE, \
                f"Loading memory regression: {memory_ratio:.2f}x more memory than baseline"
            
            print(f"✓ Loading regression check: {time_ratio:.2f}x time, {memory_ratio:.2f}x memory vs baseline")
        else:
            print("⚠ No baseline available for loading regression test")
    
    def test_full_pipeline_regression(self):
        """Test for regression in full pipeline performance"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        dataset_size_mb = dataset_path.stat().st_size / 1024**2
        
        operation_name = "full_pipeline_v2"
        
        with performance_monitor(operation_name, dataset_size_mb):
            # Execute full enhanced pipeline
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            
            filter_config = FilterConfig(enable_parallel=True)
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            
            from homodyne.data.preprocessing import PreprocessingStage
            preprocessing_config = PreprocessingConfig(
                pipeline_stages=[PreprocessingStage.DIAGONAL_CORRECTION, PreprocessingStage.NORMALIZATION],
                enable_chunked_processing=True
            )
            preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            processed_data = preprocessing_pipeline.process_data(filtered_data)
            
            qc_config = QualityControlConfig()
            quality_controller = QualityController(qc_config)
            final_data, quality_report = quality_controller.assess_and_improve_data(processed_data)
            
            assert final_data is not None
            assert quality_report is not None
        
        metrics = performance_monitor.results[operation_name]
        baseline = self.baselines.get(operation_name, {})
        
        if baseline:
            time_ratio = metrics.execution_time / baseline['time']
            memory_ratio = metrics.peak_memory_mb / baseline['memory']
            
            assert time_ratio < self.PERFORMANCE_REGRESSION_TOLERANCE, \
                f"Pipeline time regression: {time_ratio:.2f}x slower than baseline"
            assert memory_ratio < self.PERFORMANCE_REGRESSION_TOLERANCE, \
                f"Pipeline memory regression: {memory_ratio:.2f}x more memory than baseline"
            
            print(f"✓ Pipeline regression check: {time_ratio:.2f}x time, {memory_ratio:.2f}x memory vs baseline")
        else:
            print("⚠ No baseline available for pipeline regression test")


def test_performance_reporting():
    """Generate performance report from test results"""
    if not hasattr(performance_monitor, 'results'):
        pytest.skip("No performance results available")
        
    print("\n" + "="*80)
    print("PERFORMANCE TEST RESULTS SUMMARY")
    print("="*80)
    
    results = performance_monitor.results
    
    # Summary statistics
    total_operations = len(results)
    total_time = sum(metrics.execution_time for metrics in results.values())
    avg_throughput = np.mean([metrics.throughput_mbps for metrics in results.values() 
                             if metrics.throughput_mbps > 0])
    
    print(f"Total operations tested: {total_operations}")
    print(f"Total execution time: {total_time:.2f}s")
    print(f"Average throughput: {avg_throughput:.1f} MB/s")
    print()
    
    # Top performers
    print("TOP PERFORMERS (by throughput):")
    sorted_by_throughput = sorted(results.items(), 
                                 key=lambda x: x[1].throughput_mbps, 
                                 reverse=True)
    
    for i, (operation, metrics) in enumerate(sorted_by_throughput[:5]):
        if metrics.throughput_mbps > 0:
            print(f"  {i+1}. {operation}: {metrics.throughput_mbps:.1f} MB/s")
    print()
    
    # Memory efficiency
    print("MEMORY EFFICIENCY:")
    for operation, metrics in results.items():
        if metrics.dataset_size_mb > 0:
            efficiency = metrics.dataset_size_mb / metrics.peak_memory_mb
            print(f"  {operation}: {efficiency:.2f} (dataset: {metrics.dataset_size_mb:.1f}MB, peak: {metrics.peak_memory_mb:.1f}MB)")
    
    print("="*80)
    
    # Performance criteria check
    performance_issues = []
    for operation, metrics in results.items():
        if metrics.execution_time > 30.0:  # Slow operations
            performance_issues.append(f"{operation} is slow: {metrics.execution_time:.2f}s")
        if metrics.peak_memory_mb > 2000:  # High memory usage
            performance_issues.append(f"{operation} uses excessive memory: {metrics.peak_memory_mb:.1f}MB")
        if metrics.dataset_size_mb > 0 and metrics.throughput_mbps < 5:  # Low throughput
            performance_issues.append(f"{operation} has low throughput: {metrics.throughput_mbps:.1f}MB/s")
    
    if performance_issues:
        print("PERFORMANCE ISSUES DETECTED:")
        for issue in performance_issues:
            print(f"  ⚠ {issue}")
    else:
        print("✓ All performance criteria passed")
    
    print("="*80)


if __name__ == "__main__":
    # Run performance tests when executed directly
    pytest.main([__file__, "-v", "--tb=short", "-s"])
    
    # Generate performance report
    test_performance_reporting()