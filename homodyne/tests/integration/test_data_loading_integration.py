"""
Master Integration Test Suite for Enhanced Data Loading System
==============================================================

Comprehensive end-to-end testing of the complete enhanced data loading pipeline:
- Config-based filtering (Subagent 1: filtering_utils.py)
- Advanced preprocessing (Subagent 2: preprocessing.py) 
- Data quality control (Subagent 3: quality_controller.py)
- Performance optimization (Subagent 4: performance_engine.py, memory_manager.py)

This test suite validates that all components work together correctly and
that the enhanced system provides the expected functionality and performance.
"""

import os
import sys
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import time
import psutil
import warnings
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Core dependencies
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

# JAX integration
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False

# Test data generator
from homodyne.tests.data.synthetic_data_generator import (
    SyntheticDataGenerator, SyntheticDatasetConfig, 
    DatasetSize, DataQuality, DatasetFormat,
    generate_test_dataset_suite
)

# Core homodyne imports
try:
    from homodyne.data.xpcs_loader import XPCSLoader
    from homodyne.data.filtering_utils import (
        FilteringEngine, FilterConfig, FilterType,
        QRangeFilter, QualityFilter, FrameFilter, PhiFilter
    )
    from homodyne.data.preprocessing import (
        PreprocessingPipeline, PreprocessingConfig, PreprocessingStage,
        DiagonalCorrectionMethod, NormalizationMethod, NoiseReductionMethod
    )
    from homodyne.data.quality_controller import (
        QualityController, QualityControlConfig, QualityThreshold,
        QualityStage, QualityAction
    )
    from homodyne.data.performance_engine import (
        PerformanceEngine, PerformanceConfig, OptimizationLevel
    )
    from homodyne.data.memory_manager import (
        MemoryManager, MemoryConfig, MemoryStrategy
    )
    from homodyne.config.manager import ConfigManager
    HOMODYNE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import homodyne modules: {e}")
    HOMODYNE_AVAILABLE = False


class TestDataLoadingIntegration:
    """Master integration test suite for enhanced data loading"""
    
    @classmethod
    def setup_class(cls):
        """Set up test class with synthetic datasets"""
        if not HAS_HDF5:
            pytest.skip("h5py not available")
        if not HOMODYNE_AVAILABLE:
            pytest.skip("Homodyne modules not available")
            
        # Create temporary directory for test data
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_integration_test_"))
        
        # Generate test datasets
        print("Generating synthetic test datasets...")
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
        
        # Create test configurations
        cls._create_test_configurations()
        
    @classmethod
    def teardown_class(cls):
        """Clean up test data"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            
    @classmethod
    def _create_test_configurations(cls):
        """Create test configurations for different scenarios"""
        cls.config_dir = cls.test_dir / "configs"
        cls.config_dir.mkdir(exist_ok=True)
        
        # Basic configuration
        basic_config = {
            'data_loading': {
                'enhanced_features': {
                    'enable_filtering': True,
                    'enable_preprocessing': True,
                    'enable_quality_control': True,
                    'enable_performance_optimization': True
                },
                'filtering': {
                    'q_range': {'min': 1e-4, 'max': 1e-2, 'units': 'm^-1'},
                    'quality_threshold': 0.8,
                    'frame_selection': {'method': 'auto'},
                    'phi_filtering': {'enable': True, 'method': 'adaptive'}
                },
                'preprocessing': {
                    'pipeline_stages': ['diagonal_correction', 'normalization', 'noise_reduction'],
                    'diagonal_correction': {'method': 'statistical', 'threshold': 2.0},
                    'normalization': {'method': 'baseline', 'preserve_physics': True},
                    'noise_reduction': {'method': 'median', 'kernel_size': 3}
                },
                'quality_control': {
                    'enable_progressive': True,
                    'enable_auto_repair': True,
                    'thresholds': {
                        'signal_to_noise': 5.0,
                        'data_completeness': 0.9,
                        'baseline_stability': 0.95
                    }
                },
                'performance': {
                    'optimization_level': 'adaptive',
                    'enable_memory_mapping': True,
                    'enable_caching': True,
                    'parallel_processing': True,
                    'chunk_size': 'auto'
                }
            }
        }
        
        # Save configurations
        import yaml
        with open(cls.config_dir / "basic_config.yaml", 'w') as f:
            yaml.dump(basic_config, f)
            
        # Minimal configuration (fallback testing)
        minimal_config = {
            'data_loading': {
                'enhanced_features': {
                    'enable_filtering': False,
                    'enable_preprocessing': False, 
                    'enable_quality_control': False,
                    'enable_performance_optimization': False
                }
            }
        }
        
        with open(cls.config_dir / "minimal_config.yaml", 'w') as f:
            yaml.dump(minimal_config, f)
            
        # Performance-focused configuration
        performance_config = {
            'data_loading': {
                'enhanced_features': {
                    'enable_filtering': True,
                    'enable_preprocessing': True,
                    'enable_quality_control': True,
                    'enable_performance_optimization': True
                },
                'performance': {
                    'optimization_level': 'aggressive',
                    'enable_memory_mapping': True,
                    'enable_caching': True,
                    'parallel_processing': True,
                    'memory_strategy': 'aggressive',
                    'chunk_size': 'large'
                }
            }
        }
        
        with open(cls.config_dir / "performance_config.yaml", 'w') as f:
            yaml.dump(performance_config, f)
    
    def test_basic_integration_pipeline(self):
        """Test basic integration of all enhanced components"""
        # Use small good-quality dataset for reliable testing
        dataset_path = self.test_datasets['integration_small']
        config_path = self.config_dir / "basic_config.yaml"
        
        # Initialize ConfigManager
        config_manager = ConfigManager(str(config_path))
        
        # Test complete pipeline: Load → Filter → Preprocess → Quality Control
        loader = XPCSLoader(dataset_path)
        
        # Load raw data
        raw_data = loader.load_data()
        assert raw_data is not None
        assert 'g2' in raw_data
        assert 'delays' in raw_data
        
        # Apply filtering
        filter_config = FilterConfig.from_dict(
            config_manager.get('data_loading', 'filtering', default={})
        )
        filtering_engine = FilteringEngine(filter_config)
        filtered_data = filtering_engine.apply_filtering(raw_data)
        
        # Verify filtering preserved data integrity
        assert 'g2' in filtered_data
        assert filtered_data['g2'].shape[-1] == raw_data['g2'].shape[-1]  # Time delays preserved
        
        # Apply preprocessing
        preprocessing_config = PreprocessingConfig.from_dict(
            config_manager.get('data_loading', 'preprocessing', default={})
        )
        preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        preprocessed_data = preprocessing_pipeline.process_data(filtered_data)
        
        # Verify preprocessing preserved physics
        assert 'g2' in preprocessed_data
        assert np.all(preprocessed_data['g2'] >= 1.0)  # g2 ≥ 1 (physics constraint)
        
        # Apply quality control
        qc_config = QualityControlConfig.from_dict(
            config_manager.get('data_loading', 'quality_control', default={})
        )
        quality_controller = QualityController(qc_config)
        final_data, quality_report = quality_controller.assess_and_improve_data(preprocessed_data)
        
        # Verify quality control results
        assert quality_report is not None
        assert 'overall_score' in quality_report
        assert quality_report['overall_score'] >= 0.0
        assert quality_report['overall_score'] <= 1.0
        
        print("✓ Basic integration pipeline completed successfully")
    
    def test_performance_optimization_integration(self):
        """Test performance optimization integration with data loading"""
        dataset_path = self.test_datasets['performance_medium']
        config_path = self.config_dir / "performance_config.yaml"
        
        # Initialize performance-optimized system
        config_manager = ConfigManager(str(config_path))
        
        # Initialize performance engine and memory manager
        perf_config = PerformanceConfig.from_dict(
            config_manager.get('data_loading', 'performance', default={})
        )
        performance_engine = PerformanceEngine(perf_config)
        
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.AGGRESSIVE,
            enable_memory_mapping=True,
            chunk_size='auto'
        )
        memory_manager = MemoryManager(memory_config)
        
        # Monitor system resources before and after
        initial_memory = psutil.virtual_memory().used
        start_time = time.time()
        
        # Load and process data with performance optimization
        with performance_engine.optimized_context():
            with memory_manager.memory_context():
                loader = XPCSLoader(dataset_path)
                loader.configure_performance(performance_engine)
                
                data = loader.load_data()
                
                # Apply full pipeline with performance monitoring
                filter_config = FilterConfig(enable_parallel=True)
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                preprocessing_config = PreprocessingConfig(enable_chunked_processing=True)
                preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
                processed_data = preprocessing_pipeline.process_data(filtered_data)
        
        # Check performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        peak_memory = psutil.virtual_memory().used
        memory_usage = peak_memory - initial_memory
        
        # Verify performance optimization worked
        assert processing_time < 60.0  # Should complete within 60 seconds
        assert memory_usage < 2 * 1024**3  # Should use less than 2GB additional memory
        
        # Verify data integrity maintained
        assert processed_data is not None
        assert 'g2' in processed_data
        
        print(f"✓ Performance optimization integration: {processing_time:.2f}s, {memory_usage/1024**2:.1f}MB")
    
    def test_cross_format_consistency(self):
        """Test consistency between APS old and APS-U formats"""
        aps_old_path = self.test_datasets.get('integration_aps_old')
        aps_u_path = self.test_datasets.get('integration_small')  # APS-U format
        
        if not (aps_old_path and aps_u_path):
            pytest.skip("Both APS formats not available")
        
        config_path = self.config_dir / "basic_config.yaml"
        config_manager = ConfigManager(str(config_path))
        
        # Process both formats identically
        results = {}
        for format_name, dataset_path in [('aps_old', aps_old_path), ('aps_u', aps_u_path)]:
            loader = XPCSLoader(dataset_path)
            data = loader.load_data()
            
            # Apply consistent processing
            filter_config = FilterConfig.from_dict(
                config_manager.get('data_loading', 'filtering', default={})
            )
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            
            preprocessing_config = PreprocessingConfig.from_dict(
                config_manager.get('data_loading', 'preprocessing', default={})
            )
            preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            processed_data = preprocessing_pipeline.process_data(filtered_data)
            
            results[format_name] = processed_data
        
        # Compare results for consistency
        aps_old_g2 = results['aps_old']['g2']
        aps_u_g2 = results['aps_u']['g2']
        
        # Should have same basic structure and physics constraints
        assert aps_old_g2.ndim == aps_u_g2.ndim
        assert np.all(aps_old_g2 >= 1.0)
        assert np.all(aps_u_g2 >= 1.0)
        
        # Both should produce reasonable correlation functions
        assert np.all(aps_old_g2[:, :, 0] > aps_old_g2[:, :, -1])  # Decay with time
        assert np.all(aps_u_g2[:, :, 0] > aps_u_g2[:, :, -1])      # Decay with time
        
        print("✓ Cross-format consistency validated")
    
    def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation"""
        # Test with corrupted dataset
        if 'edge_case_corrupted' not in self.test_datasets:
            pytest.skip("Corrupted test dataset not available")
            
        dataset_path = self.test_datasets['edge_case_corrupted']
        config_path = self.config_dir / "basic_config.yaml"
        
        config_manager = ConfigManager(str(config_path))
        
        # Should handle corrupted data gracefully
        loader = XPCSLoader(dataset_path)
        
        # Load data - should succeed but with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data = loader.load_data()
            
            # Should have warnings about data quality
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any('quality' in msg.lower() or 'corrupt' in msg.lower() 
                      for msg in warning_messages)
        
        # Data should still be loaded (with NaN values)
        assert data is not None
        assert 'g2' in data
        
        # Apply quality control - should detect issues
        qc_config = QualityControlConfig(
            enable_auto_repair=True,
            enable_progressive=True
        )
        quality_controller = QualityController(qc_config)
        
        improved_data, quality_report = quality_controller.assess_and_improve_data(data)
        
        # Quality report should indicate issues
        assert quality_report['overall_score'] < 0.8  # Poor quality detected
        assert 'data_integrity_issues' in quality_report
        
        # Auto-repair should improve data somewhat
        if qc_config.enable_auto_repair:
            assert quality_report['repairs_applied'] > 0
        
        print("✓ Error handling and recovery validated")
    
    def test_fallback_behavior(self):
        """Test fallback behavior when advanced features fail"""
        dataset_path = self.test_datasets['unit_test_good']
        config_path = self.config_dir / "minimal_config.yaml"  # Minimal config
        
        # Test loading with minimal configuration (fallback mode)
        config_manager = ConfigManager(str(config_path))
        loader = XPCSLoader(dataset_path)
        
        # Should work even without enhanced features
        data = loader.load_data()
        assert data is not None
        assert 'g2' in data
        
        # Test JAX fallback
        with patch('homodyne.data.filtering_utils.HAS_JAX', False):
            filter_config = FilterConfig()
            filtering_engine = FilteringEngine(filter_config)
            filtered_data = filtering_engine.apply_filtering(data)
            assert filtered_data is not None
        
        # Test preprocessing fallback
        with patch('homodyne.data.preprocessing.HAS_JAX', False):
            preprocessing_config = PreprocessingConfig(
                pipeline_stages=[PreprocessingStage.DIAGONAL_CORRECTION]
            )
            preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
            processed_data = preprocessing_pipeline.process_data(data)
            assert processed_data is not None
        
        print("✓ Fallback behavior validated")
    
    def test_memory_efficiency_large_datasets(self):
        """Test memory efficiency with large datasets"""
        if 'performance_large' not in self.test_datasets:
            pytest.skip("Large test dataset not available")
            
        dataset_path = self.test_datasets['performance_large']
        config_path = self.config_dir / "performance_config.yaml"
        
        # Configure for memory efficiency
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.CONSERVATIVE,
            enable_memory_mapping=True,
            max_memory_usage=1024**3  # 1GB limit
        )
        memory_manager = MemoryManager(memory_config)
        
        initial_memory = psutil.virtual_memory().used
        
        with memory_manager.memory_context():
            loader = XPCSLoader(dataset_path)
            
            # Should use memory mapping for large files
            assert loader.should_use_memory_mapping()
            
            data = loader.load_data()
            assert data is not None
            
            current_memory = psutil.virtual_memory().used
            memory_increase = current_memory - initial_memory
            
            # Should not exceed memory limit significantly
            assert memory_increase < 1.5 * 1024**3  # 1.5GB max (some overhead allowed)
        
        print(f"✓ Memory efficiency validated: {memory_increase/1024**2:.1f}MB used")
    
    def test_configuration_validation(self):
        """Test configuration validation and parameter combinations"""
        # Test invalid configuration
        invalid_config = {
            'data_loading': {
                'filtering': {
                    'q_range': {'min': 1e-2, 'max': 1e-4}  # Invalid: min > max
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(invalid_config, f)
            invalid_config_path = f.name
        
        try:
            # Should raise validation error
            with pytest.raises((ValueError, RuntimeError)):
                config_manager = ConfigManager(invalid_config_path)
                filter_config = FilterConfig.from_dict(
                    config_manager.get('data_loading', 'filtering', default={})
                )
                # Validation should fail when creating filtering engine
                filtering_engine = FilteringEngine(filter_config)
        finally:
            os.unlink(invalid_config_path)
        
        print("✓ Configuration validation working")
    
    def test_progress_reporting(self):
        """Test progress reporting during long operations"""
        dataset_path = self.test_datasets['integration_small']
        config_path = self.config_dir / "basic_config.yaml"
        
        config_manager = ConfigManager(str(config_path))
        
        # Create mock progress callback
        progress_updates = []
        def progress_callback(stage: str, progress: float, message: str = ""):
            progress_updates.append((stage, progress, message))
        
        # Test with progress reporting
        loader = XPCSLoader(dataset_path)
        loader.set_progress_callback(progress_callback)
        
        data = loader.load_data()
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Progress should be in valid range
        for stage, progress, message in progress_updates:
            assert 0.0 <= progress <= 1.0
            assert isinstance(stage, str)
        
        # Should have start and completion updates
        assert any(progress == 0.0 for _, progress, _ in progress_updates)
        assert any(progress == 1.0 for _, progress, _ in progress_updates)
        
        print(f"✓ Progress reporting validated: {len(progress_updates)} updates")
    
    def test_concurrent_access(self):
        """Test thread safety and concurrent access"""
        import threading
        import concurrent.futures
        
        dataset_path = self.test_datasets['integration_small'] 
        config_path = self.config_dir / "basic_config.yaml"
        
        results = []
        errors = []
        
        def load_and_process():
            """Function to run in parallel threads"""
            try:
                config_manager = ConfigManager(str(config_path))
                loader = XPCSLoader(dataset_path)
                
                data = loader.load_data()
                
                # Quick processing to test thread safety
                filter_config = FilterConfig()
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
                
                results.append(filtered_data)
                
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads concurrently
        num_threads = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(load_and_process) for _ in range(num_threads)]
            concurrent.futures.wait(futures)
        
        # Check results
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == num_threads
        
        # All results should be valid and similar
        for result in results:
            assert result is not None
            assert 'g2' in result
        
        print(f"✓ Concurrent access validated: {num_threads} threads")
    
    def test_data_integrity_round_trip(self):
        """Test that processed data maintains scientific validity"""
        dataset_path = self.test_datasets['unit_test_perfect']  # Use perfect data
        config_path = self.config_dir / "basic_config.yaml"
        
        config_manager = ConfigManager(str(config_path))
        loader = XPCSLoader(dataset_path)
        
        # Load original data
        original_data = loader.load_data()
        original_g2 = original_data['g2']
        
        # Apply full processing pipeline
        filter_config = FilterConfig.from_dict(
            config_manager.get('data_loading', 'filtering', default={})
        )
        filtering_engine = FilteringEngine(filter_config)
        filtered_data = filtering_engine.apply_filtering(original_data)
        
        preprocessing_config = PreprocessingConfig.from_dict(
            config_manager.get('data_loading', 'preprocessing', default={})
        )
        preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        processed_data = preprocessing_pipeline.process_data(filtered_data)
        
        processed_g2 = processed_data['g2']
        
        # Physics constraints should be maintained
        assert np.all(processed_g2 >= 1.0), "g2 should be ≥ 1 (Siegert relation)"
        
        # Correlation decay should be preserved
        for q_idx in range(processed_g2.shape[0]):
            for phi_idx in range(processed_g2.shape[1]):
                g2_trace = processed_g2[q_idx, phi_idx, :]
                # Should generally decay from short to long times
                assert g2_trace[0] >= g2_trace[-1], f"Correlation should decay for q={q_idx}, phi={phi_idx}"
                
                # Should not have extreme outliers
                assert np.all(g2_trace < 10.0), "g2 should not have extreme values"
        
        # Statistical properties should be reasonable
        mean_g2 = np.mean(processed_g2)
        std_g2 = np.std(processed_g2)
        
        assert 1.0 < mean_g2 < 3.0, f"Mean g2 ({mean_g2:.3f}) should be reasonable"
        assert std_g2 < 2.0, f"g2 standard deviation ({std_g2:.3f}) should not be excessive"
        
        print("✓ Data integrity and physics preservation validated")


class TestComponentInteractions:
    """Test interactions between specific components"""
    
    @classmethod
    def setup_class(cls):
        """Set up component interaction tests"""
        if not HAS_HDF5 or not HOMODYNE_AVAILABLE:
            pytest.skip("Required dependencies not available")
            
        cls.test_dir = Path(tempfile.mkdtemp(prefix="homodyne_component_test_"))
        cls.test_datasets = generate_test_dataset_suite(cls.test_dir / "datasets")
    
    @classmethod
    def teardown_class(cls):
        """Clean up"""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
    
    def test_filtering_preprocessing_interaction(self):
        """Test interaction between filtering and preprocessing"""
        dataset_path = self.test_datasets['integration_small']
        
        loader = XPCSLoader(dataset_path)
        data = loader.load_data()
        
        # Apply aggressive filtering
        filter_config = FilterConfig(
            q_range={'min': 5e-4, 'max': 5e-3},  # Narrow range
            quality_threshold=0.9  # High quality threshold
        )
        filtering_engine = FilteringEngine(filter_config)
        filtered_data = filtering_engine.apply_filtering(data)
        
        # Preprocessing should adapt to filtered data
        preprocessing_config = PreprocessingConfig(
            pipeline_stages=[
                PreprocessingStage.DIAGONAL_CORRECTION,
                PreprocessingStage.NORMALIZATION,
                PreprocessingStage.NOISE_REDUCTION
            ],
            adapt_to_filtered_data=True
        )
        preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        processed_data = preprocessing_pipeline.process_data(filtered_data)
        
        # Verify preprocessing adapted correctly
        assert processed_data is not None
        assert 'g2' in processed_data
        
        # Should maintain filtered q-range
        q_values = processed_data.get('q', [])
        if len(q_values) > 0:
            assert np.all(q_values >= filter_config.q_range['min'])
            assert np.all(q_values <= filter_config.q_range['max'])
        
        print("✓ Filtering-preprocessing interaction validated")
    
    def test_quality_control_preprocessing_feedback(self):
        """Test quality control feedback to preprocessing"""
        dataset_path = self.test_datasets.get('edge_case_noisy', 
                                              self.test_datasets['integration_small'])
        
        loader = XPCSLoader(dataset_path)
        data = loader.load_data()
        
        # Quality control with feedback enabled
        qc_config = QualityControlConfig(
            enable_progressive=True,
            enable_feedback=True,
            thresholds={
                'signal_to_noise': 8.0,  # High threshold
                'baseline_stability': 0.95
            }
        )
        quality_controller = QualityController(qc_config)
        
        # Initial assessment
        initial_assessment = quality_controller.assess_data_quality(data)
        
        # Preprocessing with quality feedback
        preprocessing_config = PreprocessingConfig(
            pipeline_stages=[PreprocessingStage.NOISE_REDUCTION],
            quality_feedback=True
        )
        preprocessing_pipeline = PreprocessingPipeline(preprocessing_config)
        
        # Should adjust processing based on quality assessment
        processed_data = preprocessing_pipeline.process_data_with_quality_feedback(
            data, initial_assessment
        )
        
        # Re-assess quality
        final_assessment = quality_controller.assess_data_quality(processed_data)
        
        # Quality should improve or maintain
        assert final_assessment['overall_score'] >= initial_assessment['overall_score']
        
        print("✓ Quality control-preprocessing feedback validated")
    
    def test_performance_memory_coordination(self):
        """Test coordination between performance engine and memory manager"""
        if 'performance_medium' not in self.test_datasets:
            pytest.skip("Medium dataset not available")
            
        dataset_path = self.test_datasets['performance_medium']
        
        # Configure coordinated performance and memory management
        perf_config = PerformanceConfig(
            optimization_level=OptimizationLevel.AGGRESSIVE,
            enable_parallel_processing=True,
            coordinate_with_memory_manager=True
        )
        
        memory_config = MemoryConfig(
            strategy=MemoryStrategy.ADAPTIVE,
            coordinate_with_performance_engine=True
        )
        
        performance_engine = PerformanceEngine(perf_config)
        memory_manager = MemoryManager(memory_config)
        
        # Should coordinate chunk sizes and processing strategies
        memory_manager.register_performance_engine(performance_engine)
        performance_engine.register_memory_manager(memory_manager)
        
        initial_memory = psutil.virtual_memory().used
        
        with performance_engine.optimized_context():
            with memory_manager.memory_context():
                loader = XPCSLoader(dataset_path)
                data = loader.load_data()
                
                # Should use coordinated processing
                assert performance_engine.is_memory_coordinated()
                assert memory_manager.is_performance_coordinated()
                
                # Process with coordination
                filter_config = FilterConfig(enable_parallel=True)
                filtering_engine = FilteringEngine(filter_config)
                filtered_data = filtering_engine.apply_filtering(data)
        
        peak_memory = psutil.virtual_memory().used
        memory_usage = peak_memory - initial_memory
        
        # Coordinated operation should be memory-efficient
        assert memory_usage < 1.5 * 1024**3  # 1.5GB limit
        
        print(f"✓ Performance-memory coordination validated: {memory_usage/1024**2:.1f}MB")


# Test utilities and fixtures
@pytest.fixture(scope="session")
def test_datasets():
    """Provide test datasets for all tests"""
    test_dir = Path(tempfile.mkdtemp(prefix="homodyne_fixture_test_"))
    datasets = generate_test_dataset_suite(test_dir / "datasets")
    
    yield datasets
    
    # Cleanup
    shutil.rmtree(test_dir)


@pytest.fixture
def basic_config():
    """Provide basic configuration for tests"""
    return {
        'data_loading': {
            'enhanced_features': {
                'enable_filtering': True,
                'enable_preprocessing': True,
                'enable_quality_control': True,
                'enable_performance_optimization': True
            }
        }
    }


def test_suite_completeness():
    """Verify test suite covers all major components"""
    required_components = [
        'filtering_utils',
        'preprocessing', 
        'quality_controller',
        'performance_engine',
        'memory_manager'
    ]
    
    # This test ensures we have integration tests for all components
    # In actual implementation, we would inspect the test methods
    # to ensure coverage
    
    test_methods = [method for method in dir(TestDataLoadingIntegration) 
                   if method.startswith('test_')]
    
    assert len(test_methods) >= 10, f"Should have at least 10 integration tests, found {len(test_methods)}"
    
    # Check for key test categories
    key_test_patterns = [
        'integration',
        'performance', 
        'error',
        'fallback',
        'memory',
        'configuration'
    ]
    
    for pattern in key_test_patterns:
        matching_tests = [method for method in test_methods 
                         if pattern in method.lower()]
        assert len(matching_tests) > 0, f"Missing tests for {pattern}"
    
    print("✓ Test suite completeness validated")


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v", "--tb=short"])