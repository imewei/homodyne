"""
Tests for Dataset Size Optimization System
==========================================

Comprehensive tests for the dataset optimization functionality that ensures
memory-efficient processing strategies for different dataset sizes.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the modules to test
try:
    from homodyne.data.optimization import (
        DatasetOptimizer, 
        DatasetInfo, 
        ProcessingStrategy,
        optimize_for_method,
        create_dataset_optimizer
    )
    from homodyne.optimization import fit_vi_jax, fit_mcmc_jax
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Dataset optimization not available")
class TestDatasetOptimizer:
    """Test the DatasetOptimizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = DatasetOptimizer(memory_limit_mb=1024, enable_compression=True)
        
        # Create test datasets of different sizes
        self.small_data = np.random.randn(100, 50)  # ~40KB
        self.medium_data = np.random.randn(2000, 500)  # ~8MB  
        self.large_data = np.random.randn(5000, 2000)  # ~80MB
        
        self.sigma = np.ones_like(self.small_data) * 0.1
        self.t1 = np.linspace(0, 1, self.small_data.shape[1])
        self.t2 = np.linspace(0, 1, self.small_data.shape[1])
        self.phi = np.linspace(0, 2*np.pi, self.small_data.shape[0])
    
    def test_dataset_analyzer_small(self):
        """Test dataset analysis for small datasets."""
        dataset_info = self.optimizer.analyze_dataset(self.small_data, self.sigma)
        
        assert dataset_info.category == "small"
        assert dataset_info.size == self.small_data.size
        assert dataset_info.memory_usage_mb < 1.0  # Should be small
        assert not dataset_info.use_progressive_loading
        assert dataset_info.recommended_chunk_size == dataset_info.size  # No chunking needed
    
    def test_dataset_analyzer_medium(self):
        """Test dataset analysis for medium datasets."""
        dataset_info = self.optimizer.analyze_dataset(self.medium_data)
        
        assert dataset_info.category == "medium"
        assert dataset_info.size == self.medium_data.size
        assert dataset_info.use_progressive_loading
        assert dataset_info.recommended_chunk_size < dataset_info.size  # Should chunk
    
    def test_dataset_analyzer_large(self):
        """Test dataset analysis for large datasets."""
        dataset_info = self.optimizer.analyze_dataset(self.large_data)
        
        assert dataset_info.category == "large"
        assert dataset_info.size == self.large_data.size
        assert dataset_info.use_progressive_loading
        assert dataset_info.recommended_chunk_size < dataset_info.size  # Should chunk heavily
    
    def test_processing_strategy_vi(self):
        """Test processing strategy for VI method."""
        dataset_info = self.optimizer.analyze_dataset(self.medium_data)
        strategy = self.optimizer.get_processing_strategy(dataset_info, "vi")
        
        assert isinstance(strategy, ProcessingStrategy)
        assert strategy.chunk_size > 0
        assert strategy.batch_size > 0
        assert "vi" in str(strategy.jax_config.get("jax_enable_x64", "false")).lower()
        assert strategy.jax_config.get("xla_python_client_mem_fraction") == "0.8"
    
    def test_processing_strategy_mcmc(self):
        """Test processing strategy for MCMC method."""
        dataset_info = self.optimizer.analyze_dataset(self.medium_data)
        strategy = self.optimizer.get_processing_strategy(dataset_info, "mcmc")
        
        assert isinstance(strategy, ProcessingStrategy)
        assert strategy.chunk_size > 0
        assert strategy.batch_size > 0
        assert "true" in str(strategy.jax_config.get("jax_enable_x64", "false")).lower()
        assert strategy.jax_config.get("xla_python_client_mem_fraction") == "0.7"
    
    def test_chunked_iterator(self):
        """Test chunked iterator functionality."""
        chunk_size = 1000
        chunks = list(self.optimizer.create_chunked_iterator(
            self.medium_data, self.sigma, self.t1, self.t2, self.phi, chunk_size
        ))
        
        assert len(chunks) > 1  # Should create multiple chunks
        
        # Verify chunk structure
        for data_chunk, sigma_chunk, t1_chunk, t2_chunk, phi_chunk in chunks:
            assert data_chunk.shape[0] <= chunk_size
            assert sigma_chunk.shape == data_chunk.shape
            assert len(t1_chunk) > 0
            assert len(t2_chunk) > 0
            assert len(phi_chunk) > 0
    
    def test_memory_limit_enforcement(self):
        """Test that memory limits are respected."""
        # Create optimizer with very low memory limit
        low_memory_optimizer = DatasetOptimizer(memory_limit_mb=1.0)
        
        dataset_info = low_memory_optimizer.analyze_dataset(self.large_data)
        strategy = low_memory_optimizer.get_processing_strategy(dataset_info, "vi")
        
        # Should have very small chunk size due to memory constraint
        assert strategy.chunk_size < self.large_data.size // 10
    
    def test_strategy_caching(self):
        """Test that processing strategies are cached."""
        dataset_info = self.optimizer.analyze_dataset(self.medium_data)
        
        # Get strategy twice
        strategy1 = self.optimizer.get_processing_strategy(dataset_info, "vi")
        strategy2 = self.optimizer.get_processing_strategy(dataset_info, "vi")
        
        # Should be the same object (cached)
        assert strategy1 is strategy2
    
    def test_time_estimation(self):
        """Test processing time estimation."""
        dataset_info = self.optimizer.analyze_dataset(self.medium_data)
        
        vi_times = self.optimizer.estimate_processing_time(dataset_info, "vi")
        mcmc_times = self.optimizer.estimate_processing_time(dataset_info, "mcmc")
        
        # VI should be faster than MCMC
        assert vi_times["estimated_seconds"] < mcmc_times["estimated_seconds"]
        
        # Check that all time fields are present
        for times in [vi_times, mcmc_times]:
            assert "estimated_seconds" in times
            assert "estimated_minutes" in times
            assert "effective_rate" in times
            assert "efficiency" in times


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Dataset optimization not available")
class TestDatasetOptimizationIntegration:
    """Test integration with VI and MCMC optimization methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create small test dataset for fast testing
        self.data = np.random.randn(50, 20)
        self.sigma = np.ones_like(self.data) * 0.1
        self.t1 = np.linspace(0, 1, self.data.shape[1])
        self.t2 = np.linspace(0, 1, self.data.shape[1])
        self.phi = np.linspace(0, 2*np.pi, self.data.shape[0])
        self.q = 1.0
        self.L = 1.0
    
    def test_optimize_for_method_vi(self):
        """Test the optimize_for_method function with VI."""
        config = optimize_for_method(
            self.data, self.sigma, self.t1, self.t2, self.phi, method="vi"
        )
        
        assert "dataset_info" in config
        assert "strategy" in config
        assert config["dataset_info"].category == "small"
    
    def test_optimize_for_method_mcmc(self):
        """Test the optimize_for_method function with MCMC."""
        config = optimize_for_method(
            self.data, self.sigma, self.t1, self.t2, self.phi, method="mcmc"
        )
        
        assert "dataset_info" in config
        assert "strategy" in config
        assert config["dataset_info"].category == "small"
        assert config["strategy"].jax_config.get("jax_enable_x64") == "true"
    
    @patch('homodyne.optimization.variational.VariationalInferenceJAX')
    def test_vi_with_dataset_optimization(self, mock_vi_class):
        """Test VI fitting with dataset optimization enabled."""
        mock_vi_instance = MagicMock()
        mock_vi_class.return_value = mock_vi_instance
        
        # Mock the VI result
        from homodyne.optimization.variational import VIResult
        mock_result = MagicMock(spec=VIResult)
        mock_vi_instance.fit_vi_jax.return_value = mock_result
        
        try:
            result = fit_vi_jax(
                self.data, self.sigma, self.t1, self.t2, self.phi, 
                self.q, self.L, enable_dataset_optimization=True
            )
            
            # Verify that VI was called with optimization parameters
            mock_vi_instance.fit_vi_jax.assert_called_once()
            
            # Check that the result has dataset size information
            assert hasattr(result, 'dataset_size')
            
        except ImportError:
            pytest.skip("VI+JAX not available")
    
    @patch('homodyne.optimization.mcmc.MCMCJAXSampler')
    def test_mcmc_with_dataset_optimization(self, mock_mcmc_class):
        """Test MCMC fitting with dataset optimization enabled."""
        mock_mcmc_instance = MagicMock()
        mock_mcmc_class.return_value = mock_mcmc_instance
        
        # Mock the MCMC result
        from homodyne.optimization.mcmc import MCMCResult
        mock_result = MagicMock(spec=MCMCResult)
        mock_mcmc_instance.fit_mcmc_jax.return_value = mock_result
        
        try:
            result = fit_mcmc_jax(
                self.data, self.sigma, self.t1, self.t2, self.phi,
                self.q, self.L, enable_dataset_optimization=True
            )
            
            # Verify that MCMC was called with optimization parameters
            mock_mcmc_instance.fit_mcmc_jax.assert_called_once()
            
            # Check that the result has dataset size information
            assert hasattr(result, 'dataset_size')
            
        except ImportError:
            pytest.skip("MCMC+JAX not available")


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Dataset optimization not available")
class TestDatasetSizeCategories:
    """Test dataset size categorization."""
    
    def test_small_dataset_category(self):
        """Test small dataset categorization."""
        from homodyne.core.fitting import DatasetSize
        
        assert DatasetSize.categorize(100) == DatasetSize.SMALL
        assert DatasetSize.categorize(999_999) == DatasetSize.SMALL
    
    def test_medium_dataset_category(self):
        """Test medium dataset categorization.""" 
        from homodyne.core.fitting import DatasetSize
        
        assert DatasetSize.categorize(1_000_000) == DatasetSize.MEDIUM
        assert DatasetSize.categorize(5_000_000) == DatasetSize.MEDIUM
        assert DatasetSize.categorize(9_999_999) == DatasetSize.MEDIUM
    
    def test_large_dataset_category(self):
        """Test large dataset categorization."""
        from homodyne.core.fitting import DatasetSize
        
        assert DatasetSize.categorize(10_000_000) == DatasetSize.LARGE
        assert DatasetSize.categorize(100_000_000) == DatasetSize.LARGE


@pytest.mark.skipif(not OPTIMIZATION_AVAILABLE, reason="Dataset optimization not available") 
class TestConvenienceFunctions:
    """Test convenience functions for dataset optimization."""
    
    def test_create_dataset_optimizer(self):
        """Test the create_dataset_optimizer convenience function."""
        optimizer = create_dataset_optimizer(memory_limit_mb=512)
        
        assert isinstance(optimizer, DatasetOptimizer)
        assert optimizer.memory_limit_mb == 512
    
    def test_create_dataset_optimizer_defaults(self):
        """Test create_dataset_optimizer with default parameters."""
        optimizer = create_dataset_optimizer()
        
        assert isinstance(optimizer, DatasetOptimizer)
        assert optimizer.memory_limit_mb == 4096.0  # Default value


if __name__ == "__main__":
    pytest.main([__file__])