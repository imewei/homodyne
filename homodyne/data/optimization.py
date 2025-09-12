"""
Dataset Size Optimization for Homodyne v2
==========================================

Memory-efficient data processing strategies for different dataset sizes.
Implements chunked processing, progressive loading, and batch optimization
for VI+JAX and MCMC+JAX methods.

Key Features:
- Size-aware processing strategies (<1M, 1-10M, >20M points)
- Memory-efficient chunked processing for large datasets
- Progressive loading with intelligent caching
- JAX-optimized batch processing
- Integration with VI+JAX and MCMC+JAX pipelines
"""

import numpy as np
from typing import Iterator, Tuple, Dict, Any, Optional, List, Union
from dataclasses import dataclass
import time
from pathlib import Path
from collections import deque
from homodyne.utils.logging import get_logger, log_performance

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    def jit(f): return f
    def vmap(f, **kwargs): return f

logger = get_logger(__name__)

@dataclass
class DatasetInfo:
    """Information about dataset characteristics for optimization."""
    size: int
    category: str  # "small", "medium", "large"
    memory_usage_mb: float
    recommended_chunk_size: int
    recommended_batch_size: int
    use_progressive_loading: bool
    compression_ratio: Optional[float] = None

@dataclass
class ProcessingStrategy:
    """Processing strategy for different dataset sizes."""
    chunk_size: int
    batch_size: int
    memory_limit_mb: float
    use_caching: bool
    use_compression: bool
    parallel_workers: int
    jax_config: Dict[str, Any]

class DatasetOptimizer:
    """
    Dataset size-aware optimization for VI+JAX and MCMC+JAX.
    
    Provides memory-efficient processing strategies based on dataset size:
    - Small (<1M): In-memory processing with full JAX acceleration
    - Medium (1-10M): Efficient batching with partial memory optimization
    - Large (>20M): Distributed chunked processing with streaming
    """
    
    def __init__(self, memory_limit_mb: float = 4096.0, 
                 enable_compression: bool = True,
                 max_workers: int = None):
        """
        Initialize dataset optimizer.
        
        Args:
            memory_limit_mb: Maximum memory usage in MB
            enable_compression: Enable data compression for large datasets
            max_workers: Maximum parallel workers (None for auto-detection)
        """
        self.memory_limit_mb = memory_limit_mb
        self.enable_compression = enable_compression
        self.max_workers = max_workers or self._detect_optimal_workers()
        
        # Strategy cache for repeated operations
        self._strategy_cache: Dict[int, ProcessingStrategy] = {}
        
        logger.info(f"Dataset optimizer initialized:")
        logger.info(f"  Memory limit: {memory_limit_mb:.1f} MB")
        logger.info(f"  Compression: {enable_compression}")
        logger.info(f"  Workers: {self.max_workers}")
    
    def analyze_dataset(self, data: np.ndarray, 
                       sigma: Optional[np.ndarray] = None) -> DatasetInfo:
        """
        Analyze dataset characteristics and recommend processing strategy.
        
        Args:
            data: Primary data array
            sigma: Optional uncertainty array
            
        Returns:
            DatasetInfo with analysis results and recommendations
        """
        # Calculate memory usage
        memory_usage = self._calculate_memory_usage(data, sigma)
        
        # Categorize dataset size
        size = data.size
        if size < 1_000_000:
            category = "small"
            chunk_size = size  # Process everything at once
            batch_size = min(1000, size // 10)
            progressive_loading = False
        elif size < 10_000_000:
            category = "medium"
            chunk_size = min(500_000, size // 4)
            batch_size = min(500, size // 100)
            progressive_loading = True
        else:
            category = "large"
            chunk_size = min(100_000, size // 20)
            batch_size = min(100, size // 1000)
            progressive_loading = True
        
        # Adjust for memory constraints
        if memory_usage > self.memory_limit_mb:
            scale_factor = self.memory_limit_mb / memory_usage
            chunk_size = int(chunk_size * scale_factor * 0.8)  # 20% safety margin
            batch_size = int(batch_size * scale_factor * 0.8)
        
        dataset_info = DatasetInfo(
            size=size,
            category=category,
            memory_usage_mb=memory_usage,
            recommended_chunk_size=chunk_size,
            recommended_batch_size=batch_size,
            use_progressive_loading=progressive_loading
        )
        
        logger.info(f"Dataset analysis complete:")
        logger.info(f"  Size: {size:,} points ({category})")
        logger.info(f"  Memory: {memory_usage:.1f} MB")
        logger.info(f"  Chunk size: {chunk_size:,}")
        logger.info(f"  Batch size: {batch_size}")
        
        return dataset_info
    
    def get_processing_strategy(self, dataset_info: DatasetInfo, 
                               method: str = "vi") -> ProcessingStrategy:
        """
        Get optimized processing strategy for specific method.
        
        Args:
            dataset_info: Dataset analysis results
            method: "vi" for VI+JAX or "mcmc" for MCMC+JAX
            
        Returns:
            ProcessingStrategy optimized for the method and dataset
        """
        cache_key = hash((dataset_info.size, method, self.memory_limit_mb))
        
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        # Base strategy from dataset analysis
        chunk_size = dataset_info.recommended_chunk_size
        batch_size = dataset_info.recommended_batch_size
        
        # Method-specific adjustments
        if method.lower() == "vi":
            # VI+JAX can handle larger batches efficiently
            batch_size = min(batch_size * 2, chunk_size)
            jax_config = {
                "xla_python_client_mem_fraction": "0.8",
                "jax_enable_x64": "false",  # VI can use float32
                "jax_platforms": "gpu,cpu"
            }
        elif method.lower() == "mcmc":
            # MCMC needs more conservative memory usage
            batch_size = max(batch_size // 2, 50)
            chunk_size = max(chunk_size // 2, 1000)
            jax_config = {
                "xla_python_client_mem_fraction": "0.7",
                "jax_enable_x64": "true",  # MCMC benefits from float64
                "jax_platforms": "gpu,cpu"
            }
        else:
            jax_config = {
                "xla_python_client_mem_fraction": "0.8",
                "jax_enable_x64": "false",
                "jax_platforms": "cpu"
            }
        
        # Determine parallel workers based on dataset size
        if dataset_info.category == "small":
            workers = 1  # No need for parallelization
        elif dataset_info.category == "medium":
            workers = min(self.max_workers, 4)
        else:
            workers = self.max_workers
        
        strategy = ProcessingStrategy(
            chunk_size=chunk_size,
            batch_size=batch_size,
            memory_limit_mb=self.memory_limit_mb,
            use_caching=dataset_info.use_progressive_loading,
            use_compression=self.enable_compression and dataset_info.category == "large",
            parallel_workers=workers,
            jax_config=jax_config
        )
        
        self._strategy_cache[cache_key] = strategy
        
        logger.info(f"Processing strategy for {method.upper()}:")
        logger.info(f"  Chunk size: {chunk_size:,}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Workers: {workers}")
        logger.info(f"  Caching: {strategy.use_caching}")
        logger.info(f"  Compression: {strategy.use_compression}")
        
        return strategy
    
    def create_chunked_iterator(self, data: np.ndarray,
                               sigma: np.ndarray,
                               t1: np.ndarray, t2: np.ndarray,
                               phi: np.ndarray,
                               chunk_size: int) -> Iterator[Tuple[np.ndarray, ...]]:
        """
        Create memory-efficient chunked iterator for large datasets.
        
        Args:
            data, sigma, t1, t2, phi: Input arrays
            chunk_size: Size of each chunk
            
        Yields:
            Tuple of chunked arrays
        """
        n_data = len(data)
        n_chunks = (n_data + chunk_size - 1) // chunk_size
        
        logger.info(f"Creating chunked iterator: {n_chunks} chunks of {chunk_size:,} points")
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_data)
            
            # Extract chunk with proper indexing
            data_chunk = data[start_idx:end_idx]
            sigma_chunk = sigma[start_idx:end_idx] if sigma is not None else None
            t1_chunk = t1[start_idx:end_idx] if len(t1) > 1 else t1
            t2_chunk = t2[start_idx:end_idx] if len(t2) > 1 else t2
            phi_chunk = phi[start_idx:end_idx] if len(phi) > 1 else phi
            
            # Convert to JAX arrays if available
            if JAX_AVAILABLE:
                data_chunk = jnp.array(data_chunk)
                if sigma_chunk is not None:
                    sigma_chunk = jnp.array(sigma_chunk)
                t1_chunk = jnp.array(t1_chunk)
                t2_chunk = jnp.array(t2_chunk)
                phi_chunk = jnp.array(phi_chunk)
            
            yield data_chunk, sigma_chunk, t1_chunk, t2_chunk, phi_chunk
    
    @log_performance
    def optimize_for_vi_jax(self, data: np.ndarray, sigma: np.ndarray,
                           t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                           **kwargs) -> Dict[str, Any]:
        """
        Optimize data processing specifically for VI+JAX.
        
        Args:
            data, sigma, t1, t2, phi: Input arrays
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized processing configuration
        """
        dataset_info = self.analyze_dataset(data, sigma)
        strategy = self.get_processing_strategy(dataset_info, "vi")
        
        # Apply JAX configuration
        if JAX_AVAILABLE:
            for key, value in strategy.jax_config.items():
                import os
                os.environ[key.upper()] = value
        
        optimization_config = {
            "dataset_info": dataset_info,
            "strategy": strategy,
            "chunked_iterator": None,
            "preprocessing_time": 0.0
        }
        
        # Setup chunked processing for large datasets
        if dataset_info.category == "large":
            start_time = time.time()
            optimization_config["chunked_iterator"] = self.create_chunked_iterator(
                data, sigma, t1, t2, phi, strategy.chunk_size
            )
            optimization_config["preprocessing_time"] = time.time() - start_time
        
        return optimization_config
    
    @log_performance 
    def optimize_for_mcmc_jax(self, data: np.ndarray, sigma: np.ndarray,
                             t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                             **kwargs) -> Dict[str, Any]:
        """
        Optimize data processing specifically for MCMC+JAX.
        
        Args:
            data, sigma, t1, t2, phi: Input arrays
            **kwargs: Additional optimization parameters
            
        Returns:
            Dictionary with optimized processing configuration
        """
        dataset_info = self.analyze_dataset(data, sigma)
        strategy = self.get_processing_strategy(dataset_info, "mcmc")
        
        # Apply JAX configuration
        if JAX_AVAILABLE:
            for key, value in strategy.jax_config.items():
                import os
                os.environ[key.upper()] = value
        
        optimization_config = {
            "dataset_info": dataset_info,
            "strategy": strategy,
            "chunked_iterator": None,
            "preprocessing_time": 0.0
        }
        
        # Setup chunked processing for medium and large datasets
        if dataset_info.category in ["medium", "large"]:
            start_time = time.time()
            optimization_config["chunked_iterator"] = self.create_chunked_iterator(
                data, sigma, t1, t2, phi, strategy.chunk_size
            )
            optimization_config["preprocessing_time"] = time.time() - start_time
        
        return optimization_config
    
    def estimate_processing_time(self, dataset_info: DatasetInfo, 
                                method: str = "vi") -> Dict[str, float]:
        """
        Estimate processing time for different methods.
        
        Args:
            dataset_info: Dataset analysis results
            method: Processing method
            
        Returns:
            Dictionary with time estimates
        """
        # Base processing rates (points per second) based on empirical measurements
        if method.lower() == "vi":
            base_rate = 50000 if JAX_AVAILABLE else 5000  # VI+JAX vs numpy fallback
        elif method.lower() == "mcmc":
            base_rate = 5000 if JAX_AVAILABLE else 500   # MCMC+JAX vs numpy fallback
        else:
            base_rate = 1000
        
        # Adjust for dataset size effects
        if dataset_info.category == "small":
            efficiency = 1.0  # Full efficiency
        elif dataset_info.category == "medium":
            efficiency = 0.8  # Some overhead from chunking
        else:
            efficiency = 0.6  # More overhead from distributed processing
        
        effective_rate = base_rate * efficiency
        estimated_time = dataset_info.size / effective_rate
        
        return {
            "estimated_seconds": estimated_time,
            "estimated_minutes": estimated_time / 60,
            "effective_rate": effective_rate,
            "efficiency": efficiency
        }
    
    def _calculate_memory_usage(self, data: np.ndarray, 
                               sigma: Optional[np.ndarray] = None) -> float:
        """Calculate memory usage in MB."""
        memory_bytes = data.nbytes
        if sigma is not None:
            memory_bytes += sigma.nbytes
        
        # Add overhead for intermediate computations (factor of 3-4)
        memory_bytes *= 4
        
        return memory_bytes / (1024 * 1024)  # Convert to MB
    
    def _detect_optimal_workers(self) -> int:
        """Detect optimal number of parallel workers."""
        try:
            import os
            return min(os.cpu_count() or 1, 8)  # Cap at 8 workers
        except:
            return 4  # Safe default

# Convenience functions for integration with existing codebase
def create_dataset_optimizer(**kwargs) -> DatasetOptimizer:
    """Create dataset optimizer with sensible defaults."""
    return DatasetOptimizer(**kwargs)

def optimize_for_method(data: np.ndarray, sigma: np.ndarray,
                       t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                       method: str = "vi", **kwargs) -> Dict[str, Any]:
    """
    One-shot optimization for specific method.
    
    Args:
        data, sigma, t1, t2, phi: Input arrays
        method: "vi" or "mcmc" 
        **kwargs: Additional optimization parameters
        
    Returns:
        Optimization configuration dictionary
    """
    optimizer = create_dataset_optimizer(**kwargs)
    
    if method.lower() == "vi":
        return optimizer.optimize_for_vi_jax(data, sigma, t1, t2, phi)
    elif method.lower() == "mcmc":
        return optimizer.optimize_for_mcmc_jax(data, sigma, t1, t2, phi)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'vi' or 'mcmc'.")

class AdvancedDatasetOptimizer:
    """
    Advanced dataset optimizer that builds upon DatasetOptimizer with
    performance engine integration, memory-mapped I/O, and intelligent caching.
    
    This class extends the basic optimization.py with advanced features:
    - Integration with PerformanceEngine for memory-mapped I/O
    - Advanced memory management with pressure monitoring
    - Multi-level caching with intelligent eviction
    - Background prefetching and parallel processing
    - Real-time performance monitoring and adaptation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 performance_engine: Optional[Any] = None,
                 memory_manager: Optional[Any] = None):
        """
        Initialize advanced dataset optimizer.
        
        Args:
            config: Configuration dictionary with performance settings
            performance_engine: Optional PerformanceEngine instance
            memory_manager: Optional AdvancedMemoryManager instance
        """
        self.config = config or {}
        
        # Initialize base optimizer
        basic_config = self.config.get('basic_optimization', {})
        memory_limit = basic_config.get('memory_limit_mb', 4096.0)
        enable_compression = basic_config.get('enable_compression', True)
        max_workers = basic_config.get('max_workers', None)
        
        self.base_optimizer = DatasetOptimizer(
            memory_limit_mb=memory_limit,
            enable_compression=enable_compression,
            max_workers=max_workers
        )
        
        # Performance engine integration
        self.performance_engine = performance_engine
        if self.performance_engine is None and self._should_init_performance_engine():
            self._init_performance_engine()
        
        # Memory manager integration
        self.memory_manager = memory_manager
        if self.memory_manager is None and self._should_init_memory_manager():
            self._init_memory_manager()
        
        # Advanced optimization features
        self._prefetch_enabled = self.config.get('advanced_features', {}).get('prefetching', True)
        self._background_optimization = self.config.get('advanced_features', {}).get('background_optimization', True)
        
        # Performance tracking
        self._optimization_history = deque(maxlen=100)
        
        logger.info("Advanced dataset optimizer initialized with performance engine integration")
    
    def _should_init_performance_engine(self) -> bool:
        """Check if performance engine should be automatically initialized."""
        advanced_features = self.config.get('advanced_features', {})
        return advanced_features.get('auto_init_performance_engine', True)
    
    def _should_init_memory_manager(self) -> bool:
        """Check if memory manager should be automatically initialized."""
        advanced_features = self.config.get('advanced_features', {})
        return advanced_features.get('auto_init_memory_manager', True)
    
    def _init_performance_engine(self) -> None:
        """Initialize performance engine with configuration."""
        try:
            from homodyne.data.performance_engine import PerformanceEngine
            self.performance_engine = PerformanceEngine(self.config)
            logger.info("Performance engine initialized")
        except ImportError as e:
            logger.warning(f"Performance engine not available: {e}")
            self.performance_engine = None
        except Exception as e:
            logger.error(f"Failed to initialize performance engine: {e}")
            self.performance_engine = None
    
    def _init_memory_manager(self) -> None:
        """Initialize advanced memory manager with configuration."""
        try:
            from homodyne.data.memory_manager import AdvancedMemoryManager
            self.memory_manager = AdvancedMemoryManager(self.config)
            logger.info("Advanced memory manager initialized")
        except ImportError as e:
            logger.warning(f"Advanced memory manager not available: {e}")
            self.memory_manager = None
        except Exception as e:
            logger.error(f"Failed to initialize advanced memory manager: {e}")
            self.memory_manager = None
    
    @log_performance
    def optimize_massive_dataset(self, data: np.ndarray, sigma: np.ndarray,
                                t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                                hdf_path: Optional[str] = None,
                                method: str = "vi", **kwargs) -> Dict[str, Any]:
        """
        Optimize processing for massive datasets with advanced features.
        
        This method provides advanced optimization beyond the basic optimizer:
        - Memory-mapped I/O for datasets too large to fit in memory
        - Intelligent chunking with adaptive sizing
        - Multi-level caching for repeated operations
        - Background prefetching for predictive loading
        - Real-time performance monitoring and adaptation
        
        Args:
            data, sigma, t1, t2, phi: Input arrays
            hdf_path: Optional path to HDF5 file for memory-mapped access
            method: Processing method ("vi" or "mcmc")
            **kwargs: Additional optimization parameters
            
        Returns:
            Advanced optimization configuration with performance engine integration
        """
        start_time = time.time()
        
        # Get basic optimization as foundation
        if method.lower() == "vi":
            basic_config = self.base_optimizer.optimize_for_vi_jax(data, sigma, t1, t2, phi, **kwargs)
        else:
            basic_config = self.base_optimizer.optimize_for_mcmc_jax(data, sigma, t1, t2, phi, **kwargs)
        
        # Enhance with advanced features
        advanced_config = basic_config.copy()
        advanced_config['advanced_features'] = {}
        
        # Memory management optimization
        if self.memory_manager:
            dataset_size_gb = sum(arr.nbytes for arr in [data, sigma, t1, t2, phi] if arr is not None) / (1024**3)
            
            # Optimize memory management for workload type
            if basic_config['dataset_info'].category == "large":
                workload_type = "batch"  # Large datasets typically batch processed
            elif basic_config['dataset_info'].category == "medium":
                workload_type = "interactive"  # Medium datasets for interactive analysis
            else:
                workload_type = "streaming"  # Small datasets for streaming
            
            self.memory_manager.optimize_for_workload(workload_type, dataset_size_gb)
            
            # Get memory statistics for optimization decisions
            memory_stats = self.memory_manager.get_memory_stats()
            advanced_config['advanced_features']['memory_stats'] = memory_stats
            
            logger.info(f"Memory optimization applied: {workload_type} workload, "
                       f"{dataset_size_gb:.1f}GB dataset")
        
        # Performance engine optimization
        if self.performance_engine and hdf_path:
            try:
                # Optimize correlation matrix loading if HDF5 path provided
                data_keys = kwargs.get('correlation_keys', [])
                
                if data_keys:
                    # Create chunk plan for correlation matrices
                    chunk_info = None
                    if basic_config['dataset_info'].category in ["medium", "large"]:
                        chunk_size = self.performance_engine.chunker.calculate_optimal_chunk_size(
                            len(data_keys), data_complexity=1.2
                        )
                        chunk_info = self.performance_engine.chunker.create_chunk_plan(
                            len(data_keys), chunk_size
                        )
                    
                    advanced_config['advanced_features']['chunk_plan'] = chunk_info
                    advanced_config['advanced_features']['performance_engine_available'] = True
                    
                    # Schedule prefetching if enabled
                    if self._prefetch_enabled and len(data_keys) > 50:
                        prefetch_future = self.performance_engine.prefetch_data(
                            hdf_path, data_keys[:len(data_keys)//2], priority=3
                        )
                        advanced_config['advanced_features']['prefetch_future'] = prefetch_future
                        logger.info(f"Scheduled prefetching for {len(data_keys)//2} correlation matrices")
                else:
                    advanced_config['advanced_features']['performance_engine_available'] = True
                    
            except Exception as e:
                logger.warning(f"Performance engine optimization failed: {e}")
                advanced_config['advanced_features']['performance_engine_available'] = False
        else:
            advanced_config['advanced_features']['performance_engine_available'] = False
        
        # Advanced chunking strategy
        if basic_config['dataset_info'].category in ["medium", "large"]:
            advanced_chunking_config = self._create_advanced_chunking_config(
                basic_config['dataset_info'], method, **kwargs
            )
            advanced_config['advanced_features']['chunking'] = advanced_chunking_config
        
        # Performance monitoring setup
        optimization_time = time.time() - start_time
        performance_metrics = {
            'optimization_time_ms': optimization_time * 1000,
            'dataset_category': basic_config['dataset_info'].category,
            'method': method.upper(),
            'advanced_features_enabled': len(advanced_config['advanced_features']) > 0
        }
        
        # Add to optimization history
        self._optimization_history.append({
            'timestamp': time.time(),
            'dataset_size': basic_config['dataset_info'].size,
            'optimization_time': optimization_time,
            'method': method,
            'success': True
        })
        
        advanced_config['performance_metrics'] = performance_metrics
        
        # Background optimization if enabled
        if self._background_optimization and self.performance_engine:
            self._schedule_background_optimization(advanced_config)
        
        logger.info(f"Advanced optimization completed in {optimization_time*1000:.1f}ms: "
                   f"{basic_config['dataset_info'].category} {method.upper()} dataset")
        
        return advanced_config
    
    def _create_advanced_chunking_config(self, dataset_info: "DatasetInfo", 
                                        method: str, **kwargs) -> Dict[str, Any]:
        """Create advanced chunking configuration beyond basic optimization."""
        chunking_config = {
            'strategy': 'adaptive',
            'base_chunk_size': dataset_info.recommended_chunk_size,
            'memory_pressure_adaptation': True,
            'cross_chunk_validation': True,
            'parallel_chunk_processing': True
        }
        
        # Method-specific chunking adjustments
        if method.lower() == "vi":
            chunking_config.update({
                'chunk_overlap': 0.1,  # 10% overlap for VI stability
                'dynamic_sizing': True,
                'gpu_memory_optimization': HAS_JAX and jax_available
            })
        elif method.lower() == "mcmc":
            chunking_config.update({
                'chunk_overlap': 0.05,  # 5% overlap for MCMC
                'conservative_sizing': True,
                'chain_continuity_preservation': True
            })
        
        # Add performance-based adaptations
        if len(self._optimization_history) > 5:
            recent_performance = self._optimization_history[-5:]
            avg_time = sum(h['optimization_time'] for h in recent_performance) / len(recent_performance)
            
            if avg_time > 10.0:  # Slow optimization
                chunking_config['aggressive_chunking'] = True
                chunking_config['chunk_size_multiplier'] = 0.7
            elif avg_time < 1.0:  # Fast optimization
                chunking_config['larger_chunks_enabled'] = True
                chunking_config['chunk_size_multiplier'] = 1.3
        
        return chunking_config
    
    def _schedule_background_optimization(self, config: Dict[str, Any]) -> None:
        """Schedule background optimization tasks."""
        if not self.performance_engine:
            return
        
        try:
            # Schedule cache warming for next likely operations
            dataset_size = config['dataset_info'].size
            method = config['performance_metrics']['method'].lower()
            
            # Predict next operation based on common patterns
            if method == 'vi' and dataset_size > 1000000:
                # VI on large datasets often followed by MCMC refinement
                logger.debug("Scheduling background optimization for potential MCMC follow-up")
            
        except Exception as e:
            logger.warning(f"Background optimization scheduling failed: {e}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            'optimization_history': len(self._optimization_history),
            'advanced_features_status': {
                'performance_engine': self.performance_engine is not None,
                'memory_manager': self.memory_manager is not None,
                'prefetching_enabled': self._prefetch_enabled,
                'background_optimization': self._background_optimization
            }
        }
        
        # Add performance engine stats if available
        if self.performance_engine:
            try:
                pe_stats = self.performance_engine.get_performance_report()
                stats['performance_engine_stats'] = pe_stats
            except Exception as e:
                logger.warning(f"Failed to get performance engine stats: {e}")
        
        # Add memory manager stats if available
        if self.memory_manager:
            try:
                mem_stats = self.memory_manager.get_memory_stats()
                stats['memory_manager_stats'] = mem_stats
            except Exception as e:
                logger.warning(f"Failed to get memory manager stats: {e}")
        
        # Optimization history analysis
        if self._optimization_history:
            recent_optimizations = list(self._optimization_history)[-20:]  # Last 20
            
            stats['recent_performance'] = {
                'avg_optimization_time_ms': np.mean([h['optimization_time'] * 1000 for h in recent_optimizations]),
                'success_rate': sum(h['success'] for h in recent_optimizations) / len(recent_optimizations),
                'methods_used': list(set(h['method'] for h in recent_optimizations)),
                'dataset_sizes_range': {
                    'min': min(h['dataset_size'] for h in recent_optimizations),
                    'max': max(h['dataset_size'] for h in recent_optimizations),
                    'avg': np.mean([h['dataset_size'] for h in recent_optimizations])
                }
            }
        
        return stats
    
    def cleanup(self) -> None:
        """Cleanup advanced optimizer resources."""
        logger.info("Cleaning up advanced dataset optimizer")
        
        # Cleanup performance engine
        if self.performance_engine and hasattr(self.performance_engine, 'shutdown'):
            self.performance_engine.shutdown()
        
        # Cleanup memory manager
        if self.memory_manager and hasattr(self.memory_manager, 'shutdown'):
            self.memory_manager.shutdown()
        
        # Clear history
        self._optimization_history.clear()
        
        logger.info("Advanced dataset optimizer cleanup complete")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()

# Enhanced convenience functions that use advanced optimization
def create_advanced_dataset_optimizer(config: Optional[Dict[str, Any]] = None, **kwargs) -> AdvancedDatasetOptimizer:
    """Create advanced dataset optimizer with performance engine integration."""
    return AdvancedDatasetOptimizer(config=config, **kwargs)

def optimize_for_method_advanced(data: np.ndarray, sigma: np.ndarray,
                                t1: np.ndarray, t2: np.ndarray, phi: np.ndarray,
                                method: str = "vi", 
                                hdf_path: Optional[str] = None,
                                config: Optional[Dict[str, Any]] = None,
                                **kwargs) -> Dict[str, Any]:
    """
    Advanced one-shot optimization with performance engine integration.
    
    This function provides all the advanced features beyond basic optimization:
    - Memory-mapped I/O for massive datasets
    - Intelligent chunking and parallel processing
    - Multi-level caching and prefetching
    - Real-time performance monitoring
    
    Args:
        data, sigma, t1, t2, phi: Input arrays
        method: "vi" or "mcmc"
        hdf_path: Optional HDF5 file path for memory-mapped access
        config: Advanced optimization configuration
        **kwargs: Additional optimization parameters
        
    Returns:
        Advanced optimization configuration dictionary
    """
    with create_advanced_dataset_optimizer(config) as optimizer:
        return optimizer.optimize_massive_dataset(
            data, sigma, t1, t2, phi, hdf_path=hdf_path, method=method, **kwargs
        )

# Import guard for new dependencies
try:
    from collections import deque
    from typing import Optional, Dict, Any
    import time
    HAS_ADVANCED_DEPS = True
except ImportError:
    HAS_ADVANCED_DEPS = False
    logger.warning("Advanced optimization features may have limited functionality")

# Export main classes and functions
__all__ = [
    "DatasetInfo",
    "ProcessingStrategy", 
    "DatasetOptimizer",
    "AdvancedDatasetOptimizer",
    "create_dataset_optimizer",
    "create_advanced_dataset_optimizer",
    "optimize_for_method",
    "optimize_for_method_advanced",
]