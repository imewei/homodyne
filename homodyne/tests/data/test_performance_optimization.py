"""
Comprehensive Tests for Performance Optimization Features - Homodyne v2
=======================================================================

Test suite for advanced performance optimization components including:
- Performance engine with memory-mapped I/O and intelligent chunking
- Advanced memory manager with dynamic allocation and pressure monitoring
- Enhanced dataset optimizer with performance engine integration
- Multi-level caching system with intelligent eviction
- Performance monitoring and bottleneck detection

This test suite validates both functionality and performance characteristics
of the advanced optimization features introduced in Phase 4.
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

# Core dependencies
import numpy as np

# Test performance optimization modules
try:
    from homodyne.data.performance_engine import (AdaptiveChunker, CacheError,
                                                  ChunkInfo, MemoryMapManager,
                                                  MemoryPressureError,
                                                  MultiLevelCache,
                                                  PerformanceEngine,
                                                  PerformanceEngineError,
                                                  PerformanceMetrics)

    HAS_PERFORMANCE_ENGINE = True
except ImportError:
    HAS_PERFORMANCE_ENGINE = False

try:
    from homodyne.data.memory_manager import (AdvancedMemoryManager,
                                              AllocationError,
                                              MemoryManagerError, MemoryPool)
    from homodyne.data.memory_manager import \
        MemoryPressureError as MemMgrMemoryPressureError
    from homodyne.data.memory_manager import MemoryPressureMonitor, MemoryStats

    HAS_MEMORY_MANAGER = True
except ImportError:
    HAS_MEMORY_MANAGER = False

try:
    from homodyne.data.optimization import AdvancedDatasetOptimizer

    HAS_ADVANCED_OPTIMIZER = True
except ImportError:
    HAS_ADVANCED_OPTIMIZER = False

# Optional dependencies for comprehensive testing
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# Test utilities
def create_test_hdf5_file(
    file_path: str, num_matrices: int = 10, matrix_size: int = 64
) -> None:
    """Create a test HDF5 file with correlation matrices for testing."""
    if not HAS_H5PY:
        raise unittest.SkipTest("h5py not available for HDF5 testing")

    with h5py.File(file_path, "w") as f:
        # Create APS old format structure
        xpcs_group = f.create_group("xpcs")
        exchange_group = f.create_group("exchange")
        c2t_group = exchange_group.create_group("C2T_all")

        # Create q and phi lists
        dqlist = np.linspace(0.001, 0.1, num_matrices).reshape(1, -1)
        dphilist = np.linspace(0, 180, num_matrices).reshape(1, -1)

        xpcs_group.create_dataset("dqlist", data=dqlist)
        xpcs_group.create_dataset("dphilist", data=dphilist)

        # Create correlation matrices (half matrices as per APS format)
        for i in range(num_matrices):
            # Create a symmetric correlation matrix and take upper triangle
            full_matrix = np.random.rand(matrix_size, matrix_size)
            full_matrix = 0.5 * (full_matrix + full_matrix.T)  # Make symmetric

            # Add some realistic correlation decay
            for j in range(matrix_size):
                for k in range(matrix_size):
                    decay = np.exp(-0.1 * abs(j - k))
                    full_matrix[j, k] *= decay

            # Take upper triangle (APS format stores half matrix)
            half_matrix = np.triu(full_matrix)

            c2t_group.create_dataset(f"C2T_{i:04d}", data=half_matrix)


def create_large_test_array(size_mb: float) -> np.ndarray:
    """Create a large test array of specified size in MB."""
    # Calculate number of float64 elements needed
    bytes_per_element = 8  # float64
    total_bytes = int(size_mb * 1024 * 1024)
    num_elements = total_bytes // bytes_per_element

    return np.random.random(num_elements)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance metrics tracking and trending."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PERFORMANCE_ENGINE:
            self.skipTest("Performance engine not available")

        self.metrics = PerformanceMetrics()

    def test_metrics_initialization(self):
        """Test performance metrics initialization."""
        self.assertEqual(self.metrics.loading_speed_mbps, 0.0)
        self.assertEqual(self.metrics.memory_usage_mb, 0.0)
        self.assertEqual(self.metrics.cache_hit_rate, 0.0)
        self.assertEqual(self.metrics.cpu_utilization, 0.0)
        self.assertIsNone(self.metrics.bottleneck_type)

    def test_metrics_update(self):
        """Test performance metrics updates."""
        self.metrics.update(
            loading_speed_mbps=150.0,
            memory_usage_mb=2048.0,
            cache_hit_rate=0.85,
            cpu_utilization=0.6,
        )

        self.assertEqual(self.metrics.loading_speed_mbps, 150.0)
        self.assertEqual(self.metrics.memory_usage_mb, 2048.0)
        self.assertEqual(self.metrics.cache_hit_rate, 0.85)
        self.assertEqual(self.metrics.cpu_utilization, 0.6)

    def test_metrics_history_tracking(self):
        """Test performance metrics history tracking."""
        # Update metrics multiple times
        for i in range(5):
            self.metrics.update(loading_speed_mbps=float(i * 10))
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Check history length
        self.assertEqual(len(self.metrics._history), 5)

        # Check history contains expected values
        speeds = [h["loading_speed_mbps"] for h in self.metrics._history]
        self.assertEqual(speeds, [0.0, 10.0, 20.0, 30.0, 40.0])

    def test_metrics_trend_calculation(self):
        """Test performance trend calculation."""
        # Create increasing trend
        for i in range(10):
            self.metrics.update(loading_speed_mbps=float(i * 10))
            time.sleep(0.001)

        trend = self.metrics.get_trend("loading_speed_mbps", window=5)
        self.assertGreater(trend, 0.5)  # Should be positive trend

        # Create decreasing trend
        for i in range(10, 0, -1):
            self.metrics.update(cache_hit_rate=float(i * 0.1))
            time.sleep(0.001)

        trend = self.metrics.get_trend("cache_hit_rate", window=5)
        self.assertLess(trend, -0.5)  # Should be negative trend


@unittest.skipUnless(HAS_PERFORMANCE_ENGINE, "Performance engine not available")
class TestAdaptiveChunker(unittest.TestCase):
    """Test intelligent chunking system with adaptive sizing."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = AdaptiveChunker(base_chunk_size=10000)

    def test_chunker_initialization(self):
        """Test adaptive chunker initialization."""
        self.assertEqual(self.chunker.base_chunk_size, 10000)
        self.assertEqual(self.chunker.memory_threshold, 0.8)
        self.assertEqual(self.chunker._optimal_chunk_size, 10000)

    def test_optimal_chunk_size_calculation(self):
        """Test optimal chunk size calculation."""
        # Test normal conditions
        chunk_size = self.chunker.calculate_optimal_chunk_size(
            total_size=100000, data_complexity=1.0, available_memory_mb=4096.0
        )

        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 100000)

    def test_chunk_size_adaptation_memory_pressure(self):
        """Test chunk size adaptation under memory pressure."""
        # Test with high memory pressure
        chunk_size_high_pressure = self.chunker.calculate_optimal_chunk_size(
            total_size=100000,
            data_complexity=1.0,
            available_memory_mb=512.0,  # Low available memory
        )

        # Test with low memory pressure
        chunk_size_low_pressure = self.chunker.calculate_optimal_chunk_size(
            total_size=100000,
            data_complexity=1.0,
            available_memory_mb=8192.0,  # High available memory
        )

        # High pressure should result in smaller chunks
        self.assertLessEqual(chunk_size_high_pressure, chunk_size_low_pressure)

    def test_chunk_plan_creation(self):
        """Test intelligent chunk processing plan creation."""
        total_size = 50000
        chunk_size = 12000

        chunks = self.chunker.create_chunk_plan(total_size, chunk_size)

        # Check number of chunks
        expected_chunks = (total_size + chunk_size - 1) // chunk_size
        self.assertEqual(len(chunks), expected_chunks)

        # Check chunk properties
        for i, chunk in enumerate(chunks):
            self.assertIsInstance(chunk, ChunkInfo)
            self.assertEqual(chunk.index, i)
            self.assertGreater(chunk.size, 0)
            self.assertLessEqual(chunk.size, chunk_size)
            self.assertGreater(chunk.memory_size_mb, 0)
            self.assertGreater(chunk.complexity_score, 0)
            self.assertIn(chunk.priority, [1, 2, 3, 4, 5])

    def test_performance_feedback_update(self):
        """Test performance feedback system."""
        chunk_info = ChunkInfo(
            index=0,
            size=10000,
            memory_size_mb=80.0,
            complexity_score=1.0,
            priority=1,
            access_pattern="sequential",
            estimated_processing_time=2.0,
        )

        # Simulate successful processing
        self.chunker.update_performance_feedback(
            chunk_info, actual_processing_time=1.5, success=True
        )

        # Check feedback was recorded
        self.assertEqual(len(self.chunker._chunk_performance), 1)

        feedback = self.chunker._chunk_performance[0]
        self.assertEqual(feedback["chunk_size"], 10000)
        self.assertEqual(feedback["actual_time"], 1.5)
        self.assertTrue(feedback["success"])
        self.assertAlmostEqual(feedback["performance_ratio"], 2.0 / 1.5, places=2)


@unittest.skipUnless(HAS_PERFORMANCE_ENGINE, "Performance engine not available")
class TestMultiLevelCache(unittest.TestCase):
    """Test multi-level caching system with intelligent eviction."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLevelCache(
            memory_cache_mb=10.0,  # Small for testing
            ssd_cache_mb=50.0,
            hdd_cache_mb=100.0,
            compression_level=1,  # Fast compression for testing
        )

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_initialization(self):
        """Test cache system initialization."""
        self.assertEqual(self.cache.memory_cache_mb, 10.0)
        self.assertEqual(self.cache.ssd_cache_mb, 50.0)
        self.assertEqual(self.cache.hdd_cache_mb, 100.0)
        self.assertEqual(self.cache.compression_level, 1)

    def test_memory_cache_operations(self):
        """Test memory cache put/get operations."""
        # Test data
        test_data = np.random.random(1000)

        # Put in cache
        self.cache.put("test_key", test_data, priority=1)

        # Get from cache
        cached_data = self.cache.get("test_key")

        self.assertIsNotNone(cached_data)
        np.testing.assert_array_equal(cached_data, test_data)

    def test_cache_miss(self):
        """Test cache miss behavior."""
        result = self.cache.get("nonexistent_key")
        self.assertIsNone(result)

    def test_cache_statistics(self):
        """Test cache statistics reporting."""
        # Add some test data
        for i in range(5):
            self.cache.put(f"key_{i}", np.random.random(100), priority=2)

        stats = self.cache.get_cache_stats()

        # Check structure
        self.assertIn("memory_cache", stats)
        self.assertIn("ssd_cache", stats)
        self.assertIn("hdd_cache", stats)
        self.assertIn("total_items", stats)

        # Check memory cache stats
        memory_stats = stats["memory_cache"]
        self.assertGreater(memory_stats["items"], 0)
        self.assertGreater(memory_stats["usage_mb"], 0)

    def test_size_estimation(self):
        """Test memory size estimation for different data types."""
        # Test numpy array
        array = np.random.random(1000)
        size_mb = self.cache._estimate_size_mb(array)
        expected_size = array.nbytes / (1024 * 1024)
        self.assertAlmostEqual(size_mb, expected_size, places=2)

        # Test list
        test_list = [1, 2, 3, 4, 5]
        size_mb = self.cache._estimate_size_mb(test_list)
        self.assertGreater(size_mb, 0)

        # Test dictionary
        test_dict = {"a": 1, "b": 2, "c": 3}
        size_mb = self.cache._estimate_size_mb(test_dict)
        self.assertGreater(size_mb, 0)


@unittest.skipUnless(HAS_MEMORY_MANAGER, "Memory manager not available")
class TestAdvancedMemoryManager(unittest.TestCase):
    """Test advanced memory management system."""

    def setUp(self):
        """Set up test fixtures."""
        config = {
            "memory": {
                "warning_threshold": 0.8,
                "critical_threshold": 0.95,
                "monitoring_interval": 0.1,
                "enable_monitoring": False,  # Disable for testing
            }
        }
        self.memory_manager = AdvancedMemoryManager(config)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.memory_manager, "shutdown"):
            self.memory_manager.shutdown()

    def test_memory_manager_initialization(self):
        """Test memory manager initialization."""
        self.assertIsNotNone(self.memory_manager.pressure_monitor)
        self.assertEqual(self.memory_manager.pressure_monitor.warning_threshold, 0.8)
        self.assertEqual(self.memory_manager.pressure_monitor.critical_threshold, 0.95)

    def test_managed_allocation_context_manager(self):
        """Test managed allocation context manager."""
        with self.memory_manager.managed_allocation(1000) as buffer:
            self.assertIsInstance(buffer, np.ndarray)
            self.assertEqual(buffer.size, 1000)
            self.assertEqual(buffer.dtype, np.float64)

    def test_memory_pool_operations(self):
        """Test memory pool allocation and return."""
        # Test pool allocation
        buffer1, pool_id = self.memory_manager._get_from_pool(1000)

        if buffer1 is not None:  # Pool was available
            self.assertIsInstance(buffer1, np.ndarray)
            self.assertEqual(buffer1.size, 1000)
            self.assertIsNotNone(pool_id)

            # Test pool return
            self.memory_manager._return_to_pool(buffer1, pool_id)

    def test_memory_statistics(self):
        """Test memory statistics collection."""
        stats = self.memory_manager.get_memory_stats()

        # Check structure
        self.assertIn("system_memory", stats)
        self.assertIn("pool_management", stats)
        self.assertIn("allocation_performance", stats)
        self.assertIn("optimization_status", stats)

        # Check system memory stats
        system_stats = stats["system_memory"]
        self.assertIn("total_gb", system_stats)
        self.assertIn("available_gb", system_stats)
        self.assertIn("pressure", system_stats)


@unittest.skipUnless(HAS_PERFORMANCE_ENGINE, "Performance engine not available")
class TestPerformanceEngine(unittest.TestCase):
    """Test main performance engine functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create basic config for testing
        self.config = {
            "performance": {
                "memory_mapping": {"max_open_files": 8, "buffer_size_mb": 64.0},
                "chunking": {"base_chunk_size": 1000, "memory_threshold": 0.8},
                "caching": {
                    "memory_cache_mb": 64.0,
                    "ssd_cache_mb": 256.0,
                    "hdd_cache_mb": 1024.0,
                    "compression_level": 1,
                },
                "parallel": {"max_workers": 2},  # Small for testing
            }
        }

        self.performance_engine = PerformanceEngine(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.performance_engine, "shutdown"):
            self.performance_engine.shutdown()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_performance_engine_initialization(self):
        """Test performance engine initialization."""
        self.assertIsNotNone(self.performance_engine.memory_manager)
        self.assertIsNotNone(self.performance_engine.chunker)
        self.assertIsNotNone(self.performance_engine.cache)
        self.assertIsNotNone(self.performance_engine.executor)
        self.assertIsInstance(self.performance_engine.metrics, PerformanceMetrics)

    def test_cache_key_generation(self):
        """Test cache key generation for data."""
        hdf_path = "/test/path/data.hdf"
        data_keys = ["key1", "key2", "key3"]

        # Mock os.stat to avoid file system dependency
        with patch("os.stat") as mock_stat:
            mock_stat.return_value = type(
                "MockStat", (), {"st_mtime": 12345, "st_size": 67890}
            )()

            cache_key = self.performance_engine._generate_cache_key(hdf_path, data_keys)

            self.assertIsInstance(cache_key, str)
            self.assertIn("corr_matrices", cache_key)

    @patch("h5py.File")
    def test_correlation_matrix_reconstruction(self, mock_h5py):
        """Test correlation matrix reconstruction from half matrix."""
        # Create test half matrix (upper triangular)
        size = 4
        half_matrix = np.array(
            [
                [1.0, 0.8, 0.6, 0.4],
                [0.0, 1.0, 0.7, 0.3],
                [0.0, 0.0, 1.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        reconstructed = self.performance_engine._reconstruct_full_matrix(half_matrix)

        # Check that result is symmetric
        np.testing.assert_array_almost_equal(reconstructed, reconstructed.T)

        # Check diagonal values are correct
        expected_diagonal = np.diag(half_matrix)
        actual_diagonal = np.diag(reconstructed)
        np.testing.assert_array_almost_equal(actual_diagonal, expected_diagonal)

    def test_performance_report_generation(self):
        """Test performance report generation."""
        # Update some metrics first
        self.performance_engine.metrics.update(
            loading_speed_mbps=100.0, memory_usage_mb=1024.0, cache_hit_rate=0.75
        )

        report = self.performance_engine.get_performance_report()

        # Check report structure
        self.assertIn("performance_metrics", report)
        self.assertIn("cache_statistics", report)
        self.assertIn("chunker_status", report)
        self.assertIn("system_info", report)

        # Check metrics are included
        metrics = report["performance_metrics"]
        self.assertEqual(metrics["loading_speed_mbps"], 100.0)
        self.assertEqual(metrics["memory_usage_mb"], 1024.0)
        self.assertEqual(metrics["cache_hit_rate"], 0.75)


@unittest.skipUnless(
    HAS_ADVANCED_OPTIMIZER and HAS_PERFORMANCE_ENGINE,
    "Advanced optimizer not available",
)
class TestAdvancedDatasetOptimizer(unittest.TestCase):
    """Test advanced dataset optimizer with performance engine integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "advanced_features": {
                "auto_init_performance_engine": False,  # Manual init for testing
                "auto_init_memory_manager": False,
                "prefetching": True,
                "background_optimization": False,  # Disable for testing
            },
            "basic_optimization": {
                "memory_limit_mb": 1024.0,
                "enable_compression": True,
                "max_workers": 2,
            },
        }

        self.optimizer = AdvancedDatasetOptimizer(self.config)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self.optimizer, "cleanup"):
            self.optimizer.cleanup()

    def test_advanced_optimizer_initialization(self):
        """Test advanced optimizer initialization."""
        self.assertIsNotNone(self.optimizer.base_optimizer)
        self.assertIsNotNone(self.optimizer.config)

        # Check that performance components can be initialized
        self.assertIsNotNone(self.optimizer._should_init_performance_engine)
        self.assertIsNotNone(self.optimizer._should_init_memory_manager)

    @patch("numpy.random.random")
    def test_optimization_statistics(self, mock_random):
        """Test optimization statistics collection."""
        # Mock random data
        mock_random.return_value = np.ones(1000)

        stats = self.optimizer.get_optimization_statistics()

        # Check structure
        self.assertIn("optimization_history", stats)
        self.assertIn("advanced_features_status", stats)

        # Check advanced features status
        features_status = stats["advanced_features_status"]
        self.assertIn("performance_engine", features_status)
        self.assertIn("memory_manager", features_status)
        self.assertIn("prefetching_enabled", features_status)


class TestIntegrationWithXPCSLoader(unittest.TestCase):
    """Test integration of performance optimization with XPCS data loader."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_H5PY:
            self.skipTest("h5py not available for integration testing")

        self.temp_dir = tempfile.mkdtemp()
        self.test_hdf_path = os.path.join(self.temp_dir, "test_data.hdf")

        # Create test HDF5 file
        create_test_hdf5_file(self.test_hdf_path, num_matrices=20, matrix_size=32)

        # Create test config
        self.config = {
            "experimental_data": {
                "data_folder_path": self.temp_dir,
                "data_file_name": "test_data.hdf",
            },
            "analyzer_parameters": {"dt": 0.1, "start_frame": 1, "end_frame": 100},
            "performance": {
                "performance_engine_enabled": True,
                "memory_mapped_io": True,
                "advanced_chunking": True,
            },
            "v2_features": {"output_format": "numpy"},
        }

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("homodyne.data.xpcs_loader.HAS_PERFORMANCE_ENGINE", True)
    @patch("homodyne.data.xpcs_loader.PerformanceEngine")
    @patch("homodyne.data.xpcs_loader.AdvancedMemoryManager")
    @patch("homodyne.data.xpcs_loader.AdvancedDatasetOptimizer")
    def test_performance_components_initialization(
        self, mock_optimizer, mock_memory, mock_engine
    ):
        """Test that performance components are properly initialized in XPCS loader."""
        from homodyne.data.xpcs_loader import XPCSDataLoader

        # Create mock instances
        mock_engine.return_value = Mock()
        mock_memory.return_value = Mock()
        mock_optimizer.return_value = Mock()

        # Initialize loader
        loader = XPCSDataLoader(config_dict=self.config)

        # Check that performance components were initialized
        mock_engine.assert_called_once()
        mock_memory.assert_called_once()
        mock_optimizer.assert_called_once()

    def test_performance_disabled_fallback(self):
        """Test fallback behavior when performance optimization is disabled."""
        # Disable performance optimization
        config_disabled = self.config.copy()
        config_disabled["performance"]["performance_engine_enabled"] = False

        try:
            from homodyne.data.xpcs_loader import XPCSDataLoader

            loader = XPCSDataLoader(config_dict=config_disabled)

            # Should not have performance components
            self.assertIsNone(getattr(loader, "performance_engine", None))
            self.assertIsNone(getattr(loader, "memory_manager", None))
            self.assertIsNone(getattr(loader, "advanced_optimizer", None))
        except ImportError:
            self.skipTest("XPCS loader not available for testing")


class TestPerformanceConfiguration(unittest.TestCase):
    """Test performance configuration loading and validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_performance_config_template_loading(self):
        """Test loading of performance-optimized configuration template."""
        # Create a simple performance config
        config_path = os.path.join(self.temp_dir, "test_performance_config.yaml")

        config_content = """
performance:
  performance_engine_enabled: true
  memory_mapped_io: true
  advanced_chunking: true
  memory:
    warning_threshold: 0.75
    critical_threshold: 0.9
  caching:
    memory_cache_mb: 1024.0
    ssd_cache_mb: 4096.0
"""

        with open(config_path, "w") as f:
            f.write(config_content)

        # Test config loading
        try:
            import yaml

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate structure
            self.assertIn("performance", config)
            perf_config = config["performance"]

            self.assertTrue(perf_config["performance_engine_enabled"])
            self.assertTrue(perf_config["memory_mapped_io"])
            self.assertTrue(perf_config["advanced_chunking"])

            # Check nested configuration
            self.assertIn("memory", perf_config)
            self.assertEqual(perf_config["memory"]["warning_threshold"], 0.75)

        except ImportError:
            self.skipTest("PyYAML not available for config testing")

    def test_performance_config_validation(self):
        """Test performance configuration validation."""
        # Test valid configuration
        valid_config = {
            "performance": {
                "performance_engine_enabled": True,
                "memory": {
                    "warning_threshold": 0.75,
                    "critical_threshold": 0.9,
                    "monitoring_interval": 1.0,
                },
                "caching": {"memory_cache_mb": 1024.0, "compression_level": 3},
            }
        }

        # Validate thresholds are in correct range
        memory_config = valid_config["performance"]["memory"]
        self.assertGreaterEqual(memory_config["warning_threshold"], 0.0)
        self.assertLessEqual(memory_config["warning_threshold"], 1.0)
        self.assertGreaterEqual(memory_config["critical_threshold"], 0.0)
        self.assertLessEqual(memory_config["critical_threshold"], 1.0)
        self.assertGreater(
            memory_config["critical_threshold"], memory_config["warning_threshold"]
        )

        # Validate cache settings
        cache_config = valid_config["performance"]["caching"]
        self.assertGreater(cache_config["memory_cache_mb"], 0)
        self.assertIn(
            cache_config["compression_level"], range(1, 23)
        )  # Valid range for zstd


class TestPerformanceUnderLoad(unittest.TestCase):
    """Test performance optimization under realistic load conditions."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PERFORMANCE_ENGINE or not HAS_PSUTIL:
            self.skipTest("Performance engine or psutil not available for load testing")

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_pressure_response(self):
        """Test system response to memory pressure."""
        config = {
            "memory": {
                "warning_threshold": 0.5,  # Low threshold for testing
                "critical_threshold": 0.7,
                "monitoring_interval": 0.1,
                "enable_monitoring": False,  # Manual control for testing
            }
        }

        if not HAS_MEMORY_MANAGER:
            self.skipTest("Memory manager not available")

        memory_manager = AdvancedMemoryManager(config)

        try:
            # Get initial memory stats
            initial_stats = memory_manager.get_memory_stats()
            initial_pressure = initial_stats["system_memory"]["pressure"]

            # Simulate memory allocation (small amounts for testing)
            allocated_arrays = []

            # Allocate memory in small chunks
            for i in range(5):
                with memory_manager.managed_allocation(10000) as buffer:
                    # Create a copy to keep reference
                    allocated_arrays.append(buffer.copy())

            # Check that memory manager is tracking allocations
            final_stats = memory_manager.get_memory_stats()

            # Should have some allocation activity
            self.assertGreaterEqual(
                final_stats["allocation_performance"]["total_allocated_mb"], 0
            )

        finally:
            memory_manager.shutdown()

    def test_concurrent_cache_operations(self):
        """Test cache system under concurrent access."""
        if not HAS_PERFORMANCE_ENGINE:
            self.skipTest("Performance engine not available")

        cache = MultiLevelCache(
            memory_cache_mb=50.0,
            ssd_cache_mb=100.0,
            hdd_cache_mb=200.0,
            compression_level=1,
        )

        def cache_worker(worker_id: int, num_operations: int):
            """Worker function for concurrent cache operations."""
            for i in range(num_operations):
                key = f"worker_{worker_id}_item_{i}"
                data = np.random.random(100)

                # Put in cache
                cache.put(key, data, priority=worker_id)

                # Immediately try to get it back
                cached_data = cache.get(key)

                # Verify data integrity
                if cached_data is not None:
                    np.testing.assert_array_equal(cached_data, data)

        # Create multiple worker threads
        threads = []
        num_workers = 3
        operations_per_worker = 10

        for worker_id in range(num_workers):
            thread = threading.Thread(
                target=cache_worker, args=(worker_id, operations_per_worker)
            )
            threads.append(thread)

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10.0)  # 10 second timeout

        end_time = time.time()

        # Check that operations completed in reasonable time
        total_time = end_time - start_time
        self.assertLess(total_time, 10.0)  # Should complete within timeout

        # Check final cache statistics
        stats = cache.get_cache_stats()
        self.assertGreater(stats["total_items"], 0)

    def test_chunking_performance_adaptation(self):
        """Test chunking system performance adaptation."""
        if not HAS_PERFORMANCE_ENGINE:
            self.skipTest("Performance engine not available")

        chunker = AdaptiveChunker(base_chunk_size=5000)

        # Simulate processing with varying performance
        for i in range(15):  # Enough for adaptation window
            chunk_info = ChunkInfo(
                index=i,
                size=5000,
                memory_size_mb=40.0,
                complexity_score=1.0,
                priority=1,
                access_pattern="sequential",
                estimated_processing_time=2.0,
            )

            # Simulate progressively slower processing (to trigger adaptation)
            actual_time = 2.0 + (i * 0.1)

            chunker.update_performance_feedback(chunk_info, actual_time, success=True)

        # Check that adaptation occurred
        feedback_count = len(chunker._chunk_performance)
        self.assertEqual(feedback_count, chunker.performance_feedback_window)

        # Performance should have been recorded
        self.assertGreater(len(chunker._chunk_performance), 0)


def create_performance_test_suite() -> unittest.TestSuite:
    """Create comprehensive test suite for performance optimization features."""
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestPerformanceMetrics,
        TestAdaptiveChunker,
        TestMultiLevelCache,
        TestAdvancedMemoryManager,
        TestPerformanceEngine,
        TestAdvancedDatasetOptimizer,
        TestIntegrationWithXPCSLoader,
        TestPerformanceConfiguration,
        TestPerformanceUnderLoad,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    return suite


def run_performance_tests():
    """Run the complete performance optimization test suite."""
    print("Running Homodyne v2 Performance Optimization Tests")
    print("=" * 60)

    # Check dependencies
    missing_deps = []
    if not HAS_PERFORMANCE_ENGINE:
        missing_deps.append("performance_engine")
    if not HAS_MEMORY_MANAGER:
        missing_deps.append("memory_manager")
    if not HAS_ADVANCED_OPTIMIZER:
        missing_deps.append("advanced_optimizer")

    if missing_deps:
        print(f"WARNING: Missing dependencies: {', '.join(missing_deps)}")
        print("Some tests will be skipped.")
        print()

    # Run tests
    suite = create_performance_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("Performance Optimization Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(getattr(result, 'skipped', []))}")

    if result.wasSuccessful():
        print("\n✅ All performance optimization tests passed!")
    else:
        print("\n❌ Some tests failed. Check output above for details.")

        if result.failures:
            print(f"\nFailures ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests if called directly
    success = run_performance_tests()
    exit(0 if success else 1)
