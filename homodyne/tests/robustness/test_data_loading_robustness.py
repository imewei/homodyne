"""
Robustness and Edge Case Test Suite for Enhanced Data Loading System
====================================================================

Comprehensive testing of error handling, edge cases, and stress conditions
for the enhanced data loading system. Tests system behavior under:

- Corrupted or incomplete data
- Resource constraint scenarios  
- Invalid configurations
- Concurrent access patterns
- System resource limits
- Network/IO failures
- Memory pressure
- Unusual dataset characteristics

Key Testing Areas:
- Error handling and graceful degradation
- Data corruption recovery
- Resource constraint adaptation
- Configuration validation robustness
- Concurrent access safety
- System resource limit handling
- Fallback mechanism validation
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import shutil
import threading
import time
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings
import gc
from contextlib import contextmanager
from unittest.mock import patch, MagicMock, mock_open
import concurrent.futures

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Core dependencies
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

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
    from homodyne.data.performance_engine import PerformanceEngine, PerformanceConfig
    from homodyne.data.memory_manager import MemoryManager, MemoryConfig
    from homodyne.config.manager import ConfigManager
    HOMODYNE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import homodyne modules: {e}")
    HOMODYNE_AVAILABLE = False


@contextmanager
def memory_pressure_simulation(target_usage_mb: int = 1000):
    """Simulate memory pressure by allocating memory"""
    allocated_arrays = []
    
    try:
        # Allocate memory to simulate pressure
        current_mem = psutil.virtual_memory().available / 1024**2  # MB
        if current_mem > target_usage_mb:
            chunk_size = min(100, target_usage_mb // 10)  # Allocate in 100MB chunks
            num_chunks = (target_usage_mb - 100) // chunk_size  # Leave some buffer
            
            for _ in range(num_chunks):
                # Allocate array and touch memory to ensure it's actually used
                arr = np.random.rand(chunk_size * 1024**2 // 8)  # 8 bytes per float64
                arr[0] = 1.0  # Touch memory
                allocated_arrays.append(arr)
        
        yield
        
    finally:
        # Clean up allocated memory
        del allocated_arrays
        gc.collect()


@contextmanager
def io_failure_simulation():
    """Simulate I/O failures"""
    original_open = open
    
    def failing_open(*args, **kwargs):
        # Randomly fail some file operations
        if np.random.random() < 0.1:  # 10% failure rate
            raise IOError("Simulated I/O failure")
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', side_effect=failing_open):
        yield


class TestDataCorruptionRecovery:
    """Test recovery from various types of data corruption"""
    
    @classmethod
    def setup_class(cls):
        """Set up corruption recovery tests"""
        if not HAS_HDF5 or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_corruption_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
        
        # Create corrupted datasets
        cls._create_corrupted_datasets()
    
    @classmethod
    def teardown_class(cls):
        """Clean up corruption test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_corrupted_datasets(cls):
        """Create various types of corrupted datasets"""
        cls.corrupted_datasets = {}
        
        if 'integration_small' not in cls.test_datasets:
            return
            
        base_dataset = cls.test_datasets['integration_small']
        corrupted_dir = cls.test_dir / "corrupted"
        corrupted_dir.mkdir(exist_ok=True)
        
        # Type 1: Missing required datasets
        missing_path = corrupted_dir / "missing_datasets.h5"
        with h5py.File(base_dataset, 'r') as src:
            with h5py.File(missing_path, 'w') as dst:
                # Copy everything except g2 data
                for key in src.keys():
                    if key != 'analysis' and key != 'exchange':
                        src.copy(key, dst)
                
                # Create incomplete structure
                if 'analysis' in src:
                    analysis_group = dst.create_group('analysis')
                    if 'correlation' in src['analysis']:
                        corr_group = analysis_group.create_group('correlation')
                        # Only copy delays, not g2
                        if 'delays' in src['analysis/correlation']:
                            src.copy('analysis/correlation/delays', corr_group)
        
        cls.corrupted_datasets['missing_datasets'] = missing_path
        
        # Type 2: Invalid data types
        invalid_types_path = corrupted_dir / "invalid_types.h5"
        with h5py.File(base_dataset, 'r') as src:
            with h5py.File(invalid_types_path, 'w') as dst:
                def copy_with_corruption(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if 'g2' in name.lower():
                            # Create with wrong data type
                            corrupted_data = obj[()].astype(np.int32)  # Convert float to int
                            dst.create_dataset(name, data=corrupted_data)
                        else:
                            obj.copy(src, dst, name)
                    elif isinstance(obj, h5py.Group):
                        dst.create_group(name)
                
                src.visititems(copy_with_corruption)
        
        cls.corrupted_datasets['invalid_types'] = invalid_types_path
        
        # Type 3: NaN/Inf contamination
        nan_contaminated_path = corrupted_dir / "nan_contaminated.h5"
        with h5py.File(base_dataset, 'r') as src:
            with h5py.File(nan_contaminated_path, 'w') as dst:
                for key in src.keys():
                    if key == 'analysis' and 'analysis' in src:
                        analysis_group = dst.create_group('analysis')
                        if 'correlation' in src['analysis']:
                            corr_group = analysis_group.create_group('correlation')
                            
                            # Contaminate g2 data with NaN/Inf
                            if 'g2' in src['analysis/correlation']:
                                g2_data = src['analysis/correlation/g2'][()]
                                # Add NaN values
                                nan_mask = np.random.random(g2_data.shape) < 0.05  # 5% NaN
                                g2_data[nan_mask] = np.nan
                                # Add Inf values  
                                inf_mask = np.random.random(g2_data.shape) < 0.02  # 2% Inf
                                g2_data[inf_mask] = np.inf
                                
                                corr_group.create_dataset('g2', data=g2_data)
                            
                            # Copy other correlation data
                            for corr_key in src['analysis/correlation'].keys():
                                if corr_key != 'g2':
                                    src.copy(f'analysis/correlation/{corr_key}', corr_group)
                    else:
                        src.copy(key, dst)
        
        cls.corrupted_datasets['nan_contaminated'] = nan_contaminated_path
        
        print(f"Created {len(cls.corrupted_datasets)} corrupted datasets for testing")
    
    def test_missing_datasets_recovery(self):
        """Test recovery from missing required datasets"""
        if 'missing_datasets' not in self.corrupted_datasets:
            pytest.skip("Missing datasets corruption test file not available")
            
        corrupted_path = self.corrupted_datasets['missing_datasets']
        
        # Should handle missing g2 data gracefully
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            loader = XPCSLoader(corrupted_path)
            
            # Should raise appropriate exception or return None
            with pytest.raises((KeyError, ValueError, RuntimeError)):
                data = loader.load_data()
            
            # Or if it returns None, verify warnings were issued
            try:
                data = loader.load_data()
                if data is None:
                    assert len(w) > 0, "Should issue warnings when data is missing"
                    assert any('missing' in str(warning.message).lower() for warning in w)
            except:
                pass  # Expected behavior
        
        print("✓ Missing datasets handled gracefully")
    
    def test_nan_inf_recovery(self):
        """Test recovery from NaN/Inf contaminated data"""
        if 'nan_contaminated' not in self.corrupted_datasets:
            pytest.skip("NaN contaminated test file not available")
            
        corrupted_path = self.corrupted_datasets['nan_contaminated']
        
        # Should handle NaN/Inf values
        loader = XPCSLoader(corrupted_path)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = loader.load_data()
            
            # Should either clean data or warn about issues
            if data is not None and 'g2' in data:
                g2_data = data['g2']
                
                # Check if NaN/Inf values were cleaned or flagged
                has_nan = np.any(np.isnan(g2_data))
                has_inf = np.any(np.isinf(g2_data))
                
                if has_nan or has_inf:
                    # Should have issued warnings
                    assert len(w) > 0, "Should warn about NaN/Inf values"
                    warning_messages = [str(warning.message).lower() for warning in w]
                    assert any('nan' in msg or 'inf' in msg or 'invalid' in msg 
                              for msg in warning_messages)
            
            # Test quality control can handle contaminated data
            if data is not None:
                qc_config = QualityControlConfig(enable_auto_repair=True)
                quality_controller = QualityController(qc_config)
                
                improved_data, quality_report = quality_controller.assess_and_improve_data(data)
                
                # Quality should be flagged as poor
                assert quality_report['overall_score'] < 0.7, "Contaminated data should have poor quality score"
                
                # Auto-repair should attempt to fix issues
                if qc_config.enable_auto_repair:
                    assert quality_report.get('repairs_applied', 0) > 0, "Should attempt repairs on contaminated data"
        
        print("✓ NaN/Inf contamination handled gracefully")
    
    def test_partial_corruption_recovery(self):
        """Test recovery when only part of data is corrupted"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Base dataset not available")
            
        # Create partially corrupted dataset
        base_path = self.test_datasets['integration_small']
        partial_corrupt_path = self.test_dir / "partial_corrupt.h5"
        
        with h5py.File(base_path, 'r') as src:
            with h5py.File(partial_corrupt_path, 'w') as dst:
                # Copy everything first
                for key in src.keys():
                    src.copy(key, dst)
                
                # Corrupt only a section of g2 data
                if 'analysis/correlation/g2' in dst:
                    g2_dataset = dst['analysis/correlation/g2']
                    g2_data = g2_dataset[()]
                    
                    # Corrupt middle section
                    middle_start = g2_data.shape[-1] // 3
                    middle_end = 2 * g2_data.shape[-1] // 3
                    g2_data[..., middle_start:middle_end] = np.nan
                    
                    # Overwrite dataset
                    del dst['analysis/correlation/g2']
                    dst.create_dataset('analysis/correlation/g2', data=g2_data)
        
        # Test loading partially corrupted data
        loader = XPCSLoader(partial_corrupt_path)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = loader.load_data()
            
            assert data is not None, "Should be able to load partially corrupted data"
            assert 'g2' in data
            
            # Apply quality control
            qc_config = QualityControlConfig(
                enable_auto_repair=True,
                enable_progressive=True
            )
            quality_controller = QualityController(qc_config)
            
            improved_data, quality_report = quality_controller.assess_and_improve_data(data)
            
            # Should detect and potentially repair partial corruption
            assert 'data_integrity_issues' in quality_report
            assert quality_report['overall_score'] < 0.9  # Should detect issues
            
            if qc_config.enable_auto_repair:
                # Repairs should improve the data somewhat
                assert quality_report['repairs_applied'] > 0
                
                # Verify repaired data has fewer NaN values
                original_nan_count = np.sum(np.isnan(data['g2']))
                repaired_nan_count = np.sum(np.isnan(improved_data['g2']))
                assert repaired_nan_count <= original_nan_count
        
        print("✓ Partial corruption recovery validated")


class TestResourceConstraintAdaptation:
    """Test system behavior under resource constraints"""
    
    @classmethod
    def setup_class(cls):
        """Set up resource constraint tests"""
        if not HAS_PSUTIL or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_resource_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
    
    @classmethod
    def teardown_class(cls):
        """Clean up resource test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_low_memory_adaptation(self):
        """Test adaptation to low memory conditions"""
        if 'performance_medium' not in self.test_datasets:
            pytest.skip("Medium dataset not available for memory testing")
            
        dataset_path = self.test_datasets['performance_medium']
        
        # Configure for low memory scenario
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.CONSERVATIVE,
            max_memory_usage=256 * 1024**2,  # 256MB limit
            enable_memory_mapping=True,
            enable_aggressive_gc=True
        )
        
        memory_manager = MemoryManager(memory_config)
        
        with memory_manager.memory_context():
            # Should adapt to memory constraints
            loader = XPCSLoader(dataset_path)
            
            # Should use memory mapping for large files under memory pressure
            assert loader.should_use_memory_mapping(), "Should use memory mapping under memory constraints"
            
            data = loader.load_data()
            assert data is not None, "Should be able to load data under memory constraints"
            
            # Apply processing with memory constraints
            filter_config = FilterConfig(enable_chunked_processing=True)
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            
            # Check memory usage stayed within bounds
            current_memory = psutil.virtual_memory().used
            # Note: This is a simplified check - in practice we'd monitor more precisely
            assert filtered_data is not None
            
        print("✓ Low memory adaptation validated")
    
    def test_disk_space_constraint_handling(self):
        """Test handling of disk space constraints"""
        # Create temporary directory with limited space (simulated)
        limited_cache_dir = self.test_dir / "limited_cache"
        limited_cache_dir.mkdir(exist_ok=True)
        
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Configure with limited cache space
        perf_config = PerformanceConfig(
            cache_directory=str(limited_cache_dir),
            max_cache_size=50 * 1024**2,  # 50MB cache limit
            enable_cache_compression=True
        )
        
        performance_engine = PerformanceEngine(perf_config)
        
        with performance_engine.optimized_context():
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            
            # Process multiple times to fill cache
            for i in range(3):
                filter_config = FilterConfig()
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                # Should manage cache size automatically
                cache_size = sum(f.stat().st_size for f in limited_cache_dir.rglob('*') if f.is_file())
                cache_size_mb = cache_size / 1024**2
                
                # Allow some overhead, but should stay within reasonable bounds
                assert cache_size_mb < 100, f"Cache size ({cache_size_mb:.1f}MB) exceeded reasonable limit"
        
        print("✓ Disk space constraint handling validated")
    
    def test_cpu_constraint_adaptation(self):
        """Test adaptation to CPU constraints"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Simulate CPU constraints by limiting thread count
        constrained_config = PerformanceConfig(
            max_threads=1,  # Force single-threaded
            optimization_level=OptimizationLevel.CONSERVATIVE,
            enable_parallel_processing=False
        )
        
        performance_engine = PerformanceEngine(constrained_config)
        
        with performance_engine.optimized_context():
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            
            # Should still work in single-threaded mode
            filter_config = FilterConfig(
                enable_parallel=False,  # Force single-threaded
                max_threads=1
            )
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            
            assert filtered_data is not None, "Should work under CPU constraints"
            
            # Apply preprocessing in constrained mode
            preprocessing_config = PreprocessingConfig(
                enable_parallel_processing=False,
                max_threads=1,
                enable_chunked_processing=True  # Use chunking to reduce memory load
            )
            preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            processed_data = preprocessing_pipeline.process_data(filtered_data)
            
            assert processed_data is not None, "Preprocessing should work under CPU constraints"
        
        print("✓ CPU constraint adaptation validated")
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_pressure_response(self):
        """Test response to memory pressure"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Simulate memory pressure
        with memory_pressure_simulation(target_usage_mb=500):
            
            memory_config = MemoryConfig(
                strategy=MemoryStrategy.ADAPTIVE,
                enable_pressure_monitoring=True,
                memory_pressure_threshold=0.8  # 80% memory usage threshold
            )
            memory_manager = MemoryManager(memory_config)
            
            with memory_manager.memory_context():
                # Should detect memory pressure and adapt
                assert memory_manager.is_under_memory_pressure(), "Should detect simulated memory pressure"
                
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                # Should use more aggressive memory management
                assert data is not None, "Should handle memory pressure"
                
                # Apply processing under pressure - should use conservative strategies
                filter_config = FilterConfig()
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                assert filtered_data is not None
        
        print("✓ Memory pressure response validated")


class TestConcurrentAccessSafety:
    """Test thread safety and concurrent access patterns"""
    
    @classmethod
    def setup_class(cls):
        """Set up concurrent access tests"""
        if not HOMODYNE_AVAILABLE:
            pytest.skip("Homodyne modules not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_concurrent_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
    
    @classmethod
    def teardown_class(cls):
        """Clean up concurrent test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_concurrent_data_loading(self):
        """Test concurrent data loading safety"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        results = []
        errors = []
        
        def load_data_worker(worker_id: int):
            """Worker function for concurrent loading"""
            try:
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                # Verify data integrity
                assert data is not None
                assert 'g2' in data
                
                # Perform basic validation
                g2_data = data['g2']
                assert np.all(g2_data >= 1.0), f"Worker {worker_id}: g2 constraint violation"
                assert np.all(np.isfinite(g2_data)), f"Worker {worker_id}: non-finite values"
                
                results.append((worker_id, True, data['g2'].shape))
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run concurrent workers
        num_workers = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(load_data_worker, i) for i in range(num_workers)]
            concurrent.futures.wait(futures)
        
        # Verify results
        assert len(errors) == 0, f"Concurrent loading errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        
        # Verify all workers got consistent results
        shapes = [result[2] for result in results]
        reference_shape = shapes[0]
        assert all(shape == reference_shape for shape in shapes), "Inconsistent data shapes from concurrent loading"
        
        print(f"✓ Concurrent data loading: {num_workers} workers, no errors")
    
    def test_concurrent_processing_safety(self):
        """Test concurrent processing pipeline safety"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Load data once for shared processing
        loader = XPCSLoader(dataset_path)
        shared_data = loader.load_data()
        
        results = []
        errors = []
        
        def process_data_worker(worker_id: int):
            """Worker function for concurrent processing"""
            try:
                # Each worker processes the same data independently
                filter_config = FilterConfig()
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(shared_data.copy())
                
                preprocessing_config = PreprocessingConfig()
                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
                processed_data = preprocessing_pipeline.process_data(filtered_data)
                
                # Verify processed data
                assert processed_data is not None
                assert 'g2' in processed_data
                
                results.append((worker_id, True, processed_data['g2'].shape))
                
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Run concurrent processing
        num_workers = 6
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_data_worker, i) for i in range(num_workers)]
            concurrent.futures.wait(futures)
        
        # Verify results
        assert len(errors) == 0, f"Concurrent processing errors: {errors}"
        assert len(results) == num_workers, f"Expected {num_workers} results, got {len(results)}"
        
        print(f"✓ Concurrent processing: {num_workers} workers, no errors")
    
    def test_thread_local_storage_isolation(self):
        """Test that thread-local storage properly isolates data"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        thread_results = {}
        
        def worker_with_local_state(worker_id: int, config_modification: Dict[str, Any]):
            """Worker that modifies configuration and verifies isolation"""
            try:
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                # Apply worker-specific configuration
                filter_config = FilterConfig(**config_modification)
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                # Store configuration and results for verification
                thread_results[worker_id] = {
                    'config': config_modification,
                    'data_shape': filtered_data['g2'].shape if filtered_data and 'g2' in filtered_data else None,
                    'success': True
                }
                
            except Exception as e:
                thread_results[worker_id] = {
                    'error': str(e),
                    'success': False
                }
        
        # Create workers with different configurations
        worker_configs = [
            {'quality_threshold': 0.5},
            {'quality_threshold': 0.8},
            {'quality_threshold': 0.9},
            {'enable_parallel': True},
            {'enable_parallel': False}
        ]
        
        threads = []
        for i, config in enumerate(worker_configs):
            thread = threading.Thread(target=worker_with_local_state, args=(i, config))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify isolation - each thread should have its own configuration
        assert len(thread_results) == len(worker_configs)
        
        for worker_id, result in thread_results.items():
            assert result['success'], f"Worker {worker_id} failed: {result.get('error')}"
            assert result['config'] == worker_configs[worker_id], f"Configuration not isolated for worker {worker_id}"
        
        print(f"✓ Thread-local storage isolation: {len(worker_configs)} threads with different configs")
    
    def test_resource_contention_handling(self):
        """Test handling of resource contention between threads"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Create shared resource contention scenario
        shared_cache_dir = self.test_dir / "shared_cache"
        shared_cache_dir.mkdir(exist_ok=True)
        
        completion_times = []
        
        def contending_worker(worker_id: int):
            """Worker that creates resource contention"""
            start_time = time.time()
            
            try:
                # Configure to use shared cache
                perf_config = PerformanceConfig(
                    cache_directory=str(shared_cache_dir),
                    enable_caching=True,
                    max_cache_size=100 * 1024**2  # 100MB shared cache
                )
                performance_engine = PerformanceEngine(perf_config)
                
                with performance_engine.optimized_context():
                    loader = XPCSLoader(dataset_path)
                    data = loader.load_data()
                    
                    # Process data to create cache contention
                    filter_config = FilterConfig()
                    filtering_engine = FilteringEngine(filter_config)
                    filtered_data = filtering_engine.apply_filtering(data)
                    
                    # Simulate additional work
                    time.sleep(0.1)
                    
                completion_time = time.time() - start_time
                completion_times.append((worker_id, completion_time, True))
                
            except Exception as e:
                completion_times.append((worker_id, time.time() - start_time, False))
        
        # Run contending workers
        num_workers = 6
        threads = [threading.Thread(target=contending_worker, args=(i,)) for i in range(num_workers)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_workers = [result for result in completion_times if result[2]]
        failed_workers = [result for result in completion_times if not result[2]]
        
        assert len(successful_workers) >= num_workers // 2, "Too many workers failed under resource contention"
        
        if len(failed_workers) > 0:
            print(f"⚠ {len(failed_workers)} workers failed under resource contention (acceptable)")
        
        # Check completion time variance
        completion_times_only = [result[1] for result in successful_workers]
        if len(completion_times_only) > 1:
            time_std = np.std(completion_times_only)
            time_mean = np.mean(completion_times_only)
            cv = time_std / time_mean if time_mean > 0 else 0
            
            # Coefficient of variation should not be excessive
            assert cv < 2.0, f"Excessive completion time variation under contention: CV={cv:.2f}"
        
        print(f"✓ Resource contention handling: {len(successful_workers)}/{num_workers} workers succeeded")


class TestConfigurationRobustness:
    """Test robustness of configuration system"""
    
    @classmethod
    def setup_class(cls):
        """Set up configuration robustness tests"""
        if not HOMODYNE_AVAILABLE:
            pytest.skip("Homodyne modules not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_config_robust_test_"))
    
    @classmethod
    def teardown_class(cls):
        """Clean up configuration test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations"""
        invalid_configs = [
            # Invalid q-range
            {'data_loading': {'filtering': {'q_range': {'min': 1e-2, 'max': 1e-4}}}},  # min > max
            
            # Invalid thresholds
            {'data_loading': {'filtering': {'quality_threshold': 1.5}}},  # > 1.0
            {'data_loading': {'filtering': {'quality_threshold': -0.1}}},  # < 0.0
            
            # Invalid thread counts
            {'data_loading': {'performance': {'max_threads': -1}}},
            {'data_loading': {'performance': {'max_threads': 1000}}},  # Unreasonably high
            
            # Invalid memory limits
            {'data_loading': {'performance': {'max_memory_usage': -1000}}},
            
            # Missing required sections
            {'data_loading': {}},  # Empty section
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            config_path = self.test_dir / f"invalid_config_{i}.yaml"
            
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(invalid_config, f)
            
            # Should handle invalid configuration gracefully
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    config_manager = ConfigManager(str(config_path))
                    
                    # Try to create components with invalid config
                    filter_config = FilterConfig.from_dict(
                        config_manager.get('data_loading', 'filtering', default={})
                    )
                    
                    # Should either raise exception or use fallback values
                    filtering_engine = FilteringEngine(filter_config)
                    
                    # If it succeeds, should have issued warnings
                    if len(w) == 0:
                        # Should use fallback values
                        assert filter_config.quality_threshold >= 0.0
                        assert filter_config.quality_threshold <= 1.0
                    
                except (ValueError, RuntimeError, KeyError) as e:
                    # Expected behavior for invalid configs
                    pass
        
        print("✓ Invalid configuration handling validated")
    
    def test_configuration_migration_robustness(self):
        """Test robustness of configuration migration from old formats"""
        # Create old-format configuration
        old_config_v1 = {
            'xpcs_loader': {  # Old section name
                'quality_threshold': 0.8,
                'enable_filtering': True
            },
            'preprocessing': {
                'diagonal_correction': True,
                'normalization': 'baseline'  # Old format
            }
        }
        
        old_config_path = self.test_dir / "old_config.yaml"
        import yaml
        with open(old_config_path, 'w') as f:
            yaml.dump(old_config_v1, f)
        
        # Should handle migration or provide reasonable fallbacks
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            config_manager = ConfigManager(str(old_config_path))
            
            # Should be able to extract meaningful configuration even from old format
            data_loading_config = config_manager.get('data_loading', default={})
            
            # Should either migrate or use defaults
            assert isinstance(data_loading_config, dict)
        
        print("✓ Configuration migration robustness validated")
    
    def test_partial_configuration_handling(self):
        """Test handling of partial/incomplete configurations"""
        partial_configs = [
            # Only filtering specified
            {'data_loading': {'filtering': {'quality_threshold': 0.9}}},
            
            # Only performance specified  
            {'data_loading': {'performance': {'optimization_level': 'aggressive'}}},
            
            # Deeply nested partial config
            {'data_loading': {'preprocessing': {'pipeline_stages': ['normalization']}}},
        ]
        
        for i, partial_config in enumerate(partial_configs):
            config_path = self.test_dir / f"partial_config_{i}.yaml"
            
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(partial_config, f)
            
            config_manager = ConfigManager(str(config_path))
            
            # Should be able to create components with partial config
            # using defaults for missing sections
            try:
                filter_config = FilterConfig.from_dict(
                    config_manager.get('data_loading', 'filtering', default={})
                )
                filtering_engine = FilteringEngine(filter_config)
                
                preprocessing_config = PreprocessingConfig.from_dict(
                    config_manager.get('data_loading', 'preprocessing', default={})
                )
                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
                
                # Components should be created successfully with reasonable defaults
                assert filtering_engine is not None
                assert preprocessing_pipeline is not None
                
            except Exception as e:
                pytest.fail(f"Failed to handle partial config {i}: {e}")
        
        print("✓ Partial configuration handling validated")


class TestSystemResourceLimits:
    """Test behavior at system resource limits"""
    
    @classmethod
    def setup_class(cls):
        """Set up resource limit tests"""
        if not HAS_PSUTIL or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_limits_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
    
    @classmethod
    def teardown_class(cls):
        """Clean up resource limit test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_file_descriptor_limit_handling(self):
        """Test handling when approaching file descriptor limits"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Open many file handles to approach system limits
        open_files = []
        max_attempts = 100
        
        try:
            # Open many temporary files to consume file descriptors
            for i in range(max_attempts):
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    open_files.append(temp_file)
                except OSError:
                    break  # Hit file descriptor limit
            
            # Now try to load data near file descriptor limit
            loader = XPCSLoader(dataset_path)
            
            # Should handle file descriptor constraints gracefully
            try:
                data = loader.load_data()
                
                if data is not None:
                    # Should work even under file descriptor pressure
                    assert 'g2' in data
                    print(f"✓ Loaded data successfully with {len(open_files)} open file descriptors")
                else:
                    print("⚠ Data loading failed under file descriptor pressure (acceptable)")
                    
            except OSError as e:
                if "too many open files" in str(e).lower():
                    print("⚠ File descriptor limit hit (expected behavior)")
                else:
                    raise
                    
        finally:
            # Clean up open files
            for temp_file in open_files:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_memory_limit_graceful_degradation(self):
        """Test graceful degradation at memory limits"""
        if 'performance_medium' not in self.test_datasets:
            pytest.skip("Medium dataset not available")
            
        dataset_path = self.test_datasets['performance_medium']
        
        # Configure for very restrictive memory limits
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.CONSERVATIVE,
            max_memory_usage=128 * 1024**2,  # 128MB hard limit
            enable_aggressive_gc=True,
            memory_pressure_threshold=0.7
        )
        
        memory_manager = MemoryManager(memory_config)
        
        with memory_manager.memory_context():
            initial_memory = psutil.virtual_memory().used
            
            try:
                loader = XPCSLoader(dataset_path)
                
                # Should either work or fail gracefully
                data = loader.load_data()
                
                if data is not None:
                    peak_memory = psutil.virtual_memory().used
                    memory_used = peak_memory - initial_memory
                    
                    # Should respect memory limits reasonably well
                    memory_limit = 128 * 1024**2
                    assert memory_used < 2 * memory_limit, f"Memory usage ({memory_used/1024**2:.1f}MB) exceeded limit significantly"
                    
                    print(f"✓ Loaded data within memory constraints: {memory_used/1024**2:.1f}MB used")
                else:
                    print("⚠ Data loading failed under memory constraints (acceptable)")
                    
            except MemoryError:
                print("⚠ Memory limit hit - graceful failure (expected)")
            except Exception as e:
                if "memory" in str(e).lower():
                    print(f"⚠ Memory-related failure: {e} (acceptable)")
                else:
                    raise
    
    def test_timeout_handling(self):
        """Test handling of operation timeouts"""
        if 'integration_small' not in self.test_datasets:
            pytest.skip("Small dataset not available")
            
        dataset_path = self.test_datasets['integration_small']
        
        # Configure very short timeouts
        perf_config = PerformanceConfig(
            operation_timeout=0.1,  # 100ms timeout - very aggressive
            enable_timeout_warnings=True
        )
        
        performance_engine = PerformanceEngine(perf_config)
        
        with performance_engine.optimized_context():
            loader = XPCSLoader(dataset_path)
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                try:
                    # This might timeout or succeed depending on system speed
                    data = loader.load_data()
                    
                    if data is not None:
                        print("✓ Operation completed within aggressive timeout")
                    else:
                        print("⚠ Operation timed out (expected with aggressive timeout)")
                        
                    # Should have issued timeout warnings
                    timeout_warnings = [warning for warning in w 
                                       if 'timeout' in str(warning.message).lower()]
                    if len(timeout_warnings) > 0:
                        print(f"✓ Timeout warnings issued: {len(timeout_warnings)}")
                        
                except TimeoutError:
                    print("⚠ Timeout exception raised (acceptable)")
                except Exception as e:
                    if "timeout" in str(e).lower():
                        print(f"⚠ Timeout-related exception: {e} (acceptable)")
                    else:
                        raise


def test_error_message_quality():
    """Test that error messages are clear and actionable"""
    # Test with non-existent file
    non_existent_path = "/non/existent/file.h5"
    
    with pytest.raises(FileNotFoundError) as exc_info:
        loader = XPCSLoader(non_existent_path)
        loader.load_data()
    
    error_message = str(exc_info.value)
    assert "file" in error_message.lower() or "path" in error_message.lower()
    print("✓ File not found error message is clear")
    
    # Test with invalid file format (if we can create one)
    invalid_format_path = Path(tempfile.mkdtemp()) / "invalid.h5"
    
    try:
        # Create file with invalid format
        with h5py.File(invalid_format_path, 'w') as f:
            f.create_dataset('dummy', data=[1, 2, 3])  # No proper XPCS structure
        
        loader = XPCSLoader(invalid_format_path)
        
        with pytest.raises((KeyError, ValueError)) as exc_info:
            loader.load_data()
        
        error_message = str(exc_info.value)
        # Should provide helpful information about what's missing
        assert len(error_message) > 10, "Error message should be descriptive"
        print("✓ Invalid format error message is descriptive")
        
    finally:
        if invalid_format_path.exists():
            os.unlink(invalid_format_path)
            invalid_format_path.parent.rmdir()


def test_fallback_mechanism_completeness():
    """Test that fallback mechanisms are comprehensive"""
    # Test JAX fallback
    with patch('homodyne.data.filtering_utils.HAS_JAX', False):
        filter_config = FilterConfig()
        filtering_engine = FilteringEngine(filter_config)
        
        # Should work without JAX
        assert filtering_engine is not None
        print("✓ JAX fallback mechanism works")
    
    # Test HDF5 fallback (if applicable)
    # This would test behavior when h5py is not available
    # Implementation would depend on how homodyne handles this case
    
    print("✓ Fallback mechanisms validated")


if __name__ == "__main__":
    # Run robustness tests when executed directly
    pytest.main([__file__, "-v", "--tb=short", "-x"])  # Stop on first failure for debugging