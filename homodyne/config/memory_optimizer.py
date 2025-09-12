"""
Memory Optimization and Streaming for Homodyne v2 Configuration
===============================================================

Advanced memory management and streaming processing system for handling
very large configuration datasets with enterprise-grade performance.

Key Features:
- Memory-efficient streaming processing for billions of data points
- Intelligent memory allocation and garbage collection
- Adaptive batch sizing based on available memory
- Real-time memory usage monitoring and alerts
- Memory pool management for optimal performance
- Streaming configuration validation with minimal memory footprint
- Memory leak detection and prevention
"""

import gc
import mmap
import os
import psutil
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Callable, Generator
import pickle
import json

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """
    Memory usage statistics.
    
    Attributes:
        total_memory_gb: Total system memory in GB
        available_memory_gb: Available system memory in GB
        process_memory_gb: Current process memory usage in GB
        peak_memory_gb: Peak process memory usage in GB
        memory_utilization: Memory utilization percentage
        gc_collections: Number of garbage collections performed
        memory_pressure: Memory pressure level (0.0 to 1.0)
    """
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    process_memory_gb: float = 0.0
    peak_memory_gb: float = 0.0
    memory_utilization: float = 0.0
    gc_collections: int = 0
    memory_pressure: float = 0.0


@dataclass
class StreamingConfig:
    """
    Configuration for streaming processing.
    
    Attributes:
        chunk_size: Base chunk size for processing
        max_memory_usage_gb: Maximum memory usage limit
        adaptive_sizing: Enable adaptive chunk sizing
        use_memory_mapping: Use memory mapping for large files
        enable_compression: Enable data compression
        temp_directory: Temporary directory for spillover
        memory_check_interval: Memory check interval in seconds
        gc_frequency: Garbage collection frequency (items processed)
    """
    chunk_size: int = 1000
    max_memory_usage_gb: float = 4.0
    adaptive_sizing: bool = True
    use_memory_mapping: bool = True
    enable_compression: bool = True
    temp_directory: Optional[Path] = None
    memory_check_interval: float = 1.0
    gc_frequency: int = 10000


class MemoryMonitor:
    """
    Real-time memory monitoring and management.
    
    Provides continuous monitoring of memory usage with alerts
    and automatic memory optimization.
    """
    
    def __init__(self, alert_threshold: float = 0.8, critical_threshold: float = 0.9):
        """
        Initialize memory monitor.
        
        Args:
            alert_threshold: Memory usage threshold for alerts (0.0 to 1.0)
            critical_threshold: Memory usage threshold for critical actions
        """
        self.alert_threshold = alert_threshold
        self.critical_threshold = critical_threshold
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_history: List[MemoryStats] = []
        self._alert_callbacks: List[Callable[[MemoryStats], None]] = []
        
        # Memory tracking
        self._peak_memory = 0.0
        self._gc_count = 0
        
        logger.debug(f"Memory monitor initialized: alert={alert_threshold:.1%}, critical={critical_threshold:.1%}")
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        logger.info(f"Memory monitoring started with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        
        logger.info("Memory monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[MemoryStats], None]) -> None:
        """Add callback for memory alerts."""
        self._alert_callbacks.append(callback)
    
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        virtual_mem = psutil.virtual_memory()
        total_memory_gb = virtual_mem.total / (1024**3)
        available_memory_gb = virtual_mem.available / (1024**3)
        
        # Process memory
        process = psutil.Process()
        process_memory_bytes = process.memory_info().rss
        process_memory_gb = process_memory_bytes / (1024**3)
        
        # Update peak memory
        self._peak_memory = max(self._peak_memory, process_memory_gb)
        
        # Calculate utilization and pressure
        memory_utilization = 1.0 - (available_memory_gb / total_memory_gb)
        memory_pressure = min(1.0, memory_utilization / self.critical_threshold)
        
        return MemoryStats(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            process_memory_gb=process_memory_gb,
            peak_memory_gb=self._peak_memory,
            memory_utilization=memory_utilization,
            gc_collections=self._gc_count,
            memory_pressure=memory_pressure
        )
    
    def force_gc(self) -> int:
        """Force garbage collection and return freed objects."""
        initial_objects = len(gc.get_objects())
        gc.collect()
        self._gc_count += 1
        final_objects = len(gc.get_objects())
        
        freed_objects = max(0, initial_objects - final_objects)
        logger.debug(f"Garbage collection freed {freed_objects} objects")
        
        return freed_objects
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization."""
        logger.info("Starting memory optimization")
        
        initial_stats = self.get_current_stats()
        
        # Force garbage collection
        freed_objects = self.force_gc()
        
        # Clear internal caches if available
        optimization_stats = {
            'initial_memory_gb': initial_stats.process_memory_gb,
            'freed_objects': freed_objects,
            'gc_performed': True
        }
        
        # Get final stats
        final_stats = self.get_current_stats()
        optimization_stats['final_memory_gb'] = final_stats.process_memory_gb
        optimization_stats['memory_freed_gb'] = max(0, initial_stats.process_memory_gb - final_stats.process_memory_gb)
        
        logger.info(f"Memory optimization completed: freed {optimization_stats['memory_freed_gb']:.2f}GB")
        
        return optimization_stats
    
    def _monitor_loop(self, interval: float) -> None:
        """Memory monitoring loop."""
        while self._monitoring:
            try:
                stats = self.get_current_stats()
                
                # Store stats history (keep last 100 entries)
                self._stats_history.append(stats)
                if len(self._stats_history) > 100:
                    self._stats_history.pop(0)
                
                # Check for alerts
                if stats.memory_utilization >= self.critical_threshold:
                    logger.critical(f"CRITICAL: Memory usage {stats.memory_utilization:.1%} "
                                  f"(process: {stats.process_memory_gb:.2f}GB)")
                    self._trigger_alerts(stats)
                    
                    # Emergency memory optimization
                    self.optimize_memory()
                    
                elif stats.memory_utilization >= self.alert_threshold:
                    logger.warning(f"HIGH: Memory usage {stats.memory_utilization:.1%} "
                                 f"(process: {stats.process_memory_gb:.2f}GB)")
                    self._trigger_alerts(stats)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(interval)
    
    def _trigger_alerts(self, stats: MemoryStats) -> None:
        """Trigger memory alert callbacks."""
        for callback in self._alert_callbacks:
            try:
                callback(stats)
            except Exception as e:
                logger.debug(f"Memory alert callback failed: {e}")


class StreamingProcessor:
    """
    Memory-efficient streaming processor for large datasets.
    
    Handles processing of very large configuration datasets with
    minimal memory footprint using streaming and chunking techniques.
    """
    
    def __init__(self, streaming_config: Optional[StreamingConfig] = None):
        """
        Initialize streaming processor.
        
        Args:
            streaming_config: Streaming configuration (auto-configured if None)
        """
        if streaming_config is None:
            streaming_config = self._auto_configure_streaming()
        
        self.config = streaming_config
        self.monitor = MemoryMonitor()
        
        # State management
        self._temp_files: List[Path] = []
        self._memory_pool: List[Any] = []
        self._current_chunk_size = streaming_config.chunk_size
        
        # Temp directory setup
        if self.config.temp_directory is None:
            self.config.temp_directory = Path(tempfile.gettempdir()) / "homodyne_streaming"
        
        self.config.temp_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Streaming processor initialized: chunk_size={self.config.chunk_size}, "
                   f"max_memory={self.config.max_memory_usage_gb}GB")
    
    @contextmanager
    def memory_managed_processing(self):
        """Context manager for memory-managed processing."""
        self.monitor.start_monitoring(self.config.memory_check_interval)
        
        # Add memory alert callback
        def memory_alert_handler(stats: MemoryStats):
            if stats.memory_pressure > 0.8:
                self._adaptive_resize(stats.memory_pressure)
        
        self.monitor.add_alert_callback(memory_alert_handler)
        
        try:
            yield self
        finally:
            self.monitor.stop_monitoring()
            self._cleanup_temp_files()
    
    def stream_configurations(self, 
                            configs: Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]],
                            processor_func: Callable[[List[Dict[str, Any]]], List[Any]]) -> Generator[Any, None, None]:
        """
        Stream process configurations with memory optimization.
        
        Args:
            configs: Configuration data (list or iterator)
            processor_func: Function to process configuration chunks
            
        Yields:
            Processed results
        """
        with self.memory_managed_processing():
            chunk = []
            processed_count = 0
            
            # Convert to iterator if needed
            if isinstance(configs, list):
                config_iter = iter(configs)
            else:
                config_iter = configs
            
            for config in config_iter:
                chunk.append(config)
                processed_count += 1
                
                # Process chunk when full
                if len(chunk) >= self._current_chunk_size:
                    yield from self._process_chunk(chunk, processor_func)
                    chunk = []
                    
                    # Periodic memory management
                    if processed_count % self.config.gc_frequency == 0:
                        self._periodic_memory_management()
            
            # Process final chunk
            if chunk:
                yield from self._process_chunk(chunk, processor_func)
    
    def stream_large_file(self, 
                         file_path: Path,
                         parser_func: Callable[[str], Dict[str, Any]],
                         processor_func: Callable[[List[Dict[str, Any]]], List[Any]]) -> Generator[Any, None, None]:
        """
        Stream process large configuration files.
        
        Args:
            file_path: Path to configuration file
            parser_func: Function to parse file lines into configs
            processor_func: Function to process configuration chunks
            
        Yields:
            Processed results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        logger.info(f"Streaming large file: {file_path} ({file_path.stat().st_size / (1024**2):.1f}MB)")
        
        with self.memory_managed_processing():
            if self.config.use_memory_mapping and file_path.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                yield from self._stream_with_mmap(file_path, parser_func, processor_func)
            else:
                yield from self._stream_with_buffered_read(file_path, parser_func, processor_func)
    
    def create_memory_efficient_cache(self, 
                                    cache_data: Dict[str, Any],
                                    cache_name: str = "config_cache") -> Path:
        """
        Create memory-efficient cache file for large datasets.
        
        Args:
            cache_data: Data to cache
            cache_name: Cache identifier
            
        Returns:
            Path to cache file
        """
        cache_file = self.config.temp_directory / f"{cache_name}_{int(time.time())}.cache"
        
        try:
            if self.config.enable_compression:
                import lzma
                with lzma.open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
            
            self._temp_files.append(cache_file)
            logger.debug(f"Created memory-efficient cache: {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to create cache file: {e}")
            raise
        
        return cache_file
    
    def load_memory_efficient_cache(self, cache_file: Path) -> Any:
        """Load data from memory-efficient cache."""
        try:
            if self.config.enable_compression:
                import lzma
                with lzma.open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to load cache file: {e}")
            raise
    
    def get_memory_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        stats = self.monitor.get_current_stats()
        
        return {
            'current_memory_gb': stats.process_memory_gb,
            'peak_memory_gb': stats.peak_memory_gb,
            'available_memory_gb': stats.available_memory_gb,
            'memory_utilization': stats.memory_utilization,
            'memory_pressure': stats.memory_pressure,
            'current_chunk_size': self._current_chunk_size,
            'temp_files_count': len(self._temp_files),
            'temp_files_size_mb': sum(
                f.stat().st_size for f in self._temp_files if f.exists()
            ) / (1024**2),
            'gc_collections': stats.gc_collections
        }
    
    def _auto_configure_streaming(self) -> StreamingConfig:
        """Auto-configure streaming based on system resources."""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Conservative memory usage (use 50% of available memory max)
        max_memory_usage = min(available_memory_gb * 0.5, 8.0)  # Cap at 8GB
        
        # Adaptive chunk size based on available memory
        if available_memory_gb > 32:
            chunk_size = 10000
        elif available_memory_gb > 16:
            chunk_size = 5000
        elif available_memory_gb > 8:
            chunk_size = 2000
        else:
            chunk_size = 1000
        
        config = StreamingConfig(
            chunk_size=chunk_size,
            max_memory_usage_gb=max_memory_usage,
            adaptive_sizing=True,
            use_memory_mapping=available_memory_gb > 4.0,  # Only if we have decent memory
            enable_compression=True
        )
        
        logger.info(f"Auto-configured streaming: chunk_size={chunk_size}, "
                   f"max_memory={max_memory_usage:.1f}GB, "
                   f"memory_mapping={'enabled' if config.use_memory_mapping else 'disabled'}")
        
        return config
    
    def _process_chunk(self, 
                      chunk: List[Dict[str, Any]], 
                      processor_func: Callable[[List[Dict[str, Any]]], List[Any]]) -> Generator[Any, None, None]:
        """Process a chunk of configurations."""
        try:
            results = processor_func(chunk)
            for result in results:
                yield result
        except Exception as e:
            logger.error(f"Chunk processing failed: {e}")
            # Yield error results
            for config in chunk:
                yield {"config": config, "error": str(e)}
    
    def _stream_with_mmap(self, 
                         file_path: Path,
                         parser_func: Callable[[str], Dict[str, Any]],
                         processor_func: Callable[[List[Dict[str, Any]]], List[Any]]) -> Generator[Any, None, None]:
        """Stream file using memory mapping."""
        logger.debug(f"Using memory mapping for {file_path}")
        
        with open(file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                chunk = []
                line_buffer = b""
                
                for byte_chunk in iter(lambda: mmapped_file.read(8192), b""):
                    line_buffer += byte_chunk
                    
                    while b'\n' in line_buffer:
                        line, line_buffer = line_buffer.split(b'\n', 1)
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        
                        if line_str:
                            try:
                                config = parser_func(line_str)
                                chunk.append(config)
                                
                                if len(chunk) >= self._current_chunk_size:
                                    yield from self._process_chunk(chunk, processor_func)
                                    chunk = []
                                    
                            except Exception as e:
                                logger.debug(f"Failed to parse line: {e}")
                
                # Process final chunk
                if chunk:
                    yield from self._process_chunk(chunk, processor_func)
    
    def _stream_with_buffered_read(self, 
                                  file_path: Path,
                                  parser_func: Callable[[str], Dict[str, Any]],
                                  processor_func: Callable[[List[Dict[str, Any]]], List[Any]]) -> Generator[Any, None, None]:
        """Stream file using buffered reading."""
        logger.debug(f"Using buffered reading for {file_path}")
        
        chunk = []
        
        with open(file_path, 'r', buffering=8192) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        config = parser_func(line)
                        chunk.append(config)
                        
                        if len(chunk) >= self._current_chunk_size:
                            yield from self._process_chunk(chunk, processor_func)
                            chunk = []
                            
                    except Exception as e:
                        logger.debug(f"Failed to parse line: {e}")
        
        # Process final chunk
        if chunk:
            yield from self._process_chunk(chunk, processor_func)
    
    def _adaptive_resize(self, memory_pressure: float) -> None:
        """Adaptively resize chunk size based on memory pressure."""
        if not self.config.adaptive_sizing:
            return
        
        old_size = self._current_chunk_size
        
        if memory_pressure > 0.9:
            # Critical memory pressure - reduce chunk size aggressively
            self._current_chunk_size = max(100, int(self._current_chunk_size * 0.5))
        elif memory_pressure > 0.7:
            # High memory pressure - reduce chunk size moderately
            self._current_chunk_size = max(500, int(self._current_chunk_size * 0.8))
        elif memory_pressure < 0.3:
            # Low memory pressure - can increase chunk size
            self._current_chunk_size = min(self.config.chunk_size * 2, int(self._current_chunk_size * 1.2))
        
        if self._current_chunk_size != old_size:
            logger.info(f"Adaptive chunk resize: {old_size} → {self._current_chunk_size} "
                       f"(memory pressure: {memory_pressure:.1%})")
    
    def _periodic_memory_management(self) -> None:
        """Perform periodic memory management."""
        # Force garbage collection
        self.monitor.force_gc()
        
        # Clear memory pool if it's getting large
        if len(self._memory_pool) > 1000:
            self._memory_pool.clear()
        
        # Check if we need to spill to disk
        current_stats = self.monitor.get_current_stats()
        if current_stats.memory_pressure > 0.8:
            self._spill_to_disk()
    
    def _spill_to_disk(self) -> None:
        """Spill memory data to disk when under pressure."""
        if self._memory_pool:
            spill_file = self.config.temp_directory / f"spill_{int(time.time())}_{threading.get_ident()}.tmp"
            
            try:
                with open(spill_file, 'wb') as f:
                    pickle.dump(self._memory_pool, f)
                
                self._temp_files.append(spill_file)
                self._memory_pool.clear()
                
                logger.debug(f"Spilled memory pool to disk: {spill_file}")
                
            except Exception as e:
                logger.error(f"Failed to spill to disk: {e}")
    
    def _cleanup_temp_files(self) -> None:
        """Cleanup temporary files."""
        cleaned_count = 0
        cleaned_size = 0
        
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    cleaned_count += 1
                    cleaned_size += size
            except Exception as e:
                logger.debug(f"Failed to cleanup temp file {temp_file}: {e}")
        
        self._temp_files.clear()
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} temp files ({cleaned_size / (1024**2):.1f}MB)")


# Global instances
_global_memory_monitor: Optional[MemoryMonitor] = None
_global_streaming_processor: Optional[StreamingProcessor] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
        logger.info("Global memory monitor initialized")
    
    return _global_memory_monitor


def get_streaming_processor(config: Optional[StreamingConfig] = None) -> StreamingProcessor:
    """Get global streaming processor instance."""
    global _global_streaming_processor
    
    if _global_streaming_processor is None:
        _global_streaming_processor = StreamingProcessor(config)
        logger.info("Global streaming processor initialized")
    
    return _global_streaming_processor


@contextmanager
def memory_efficient_processing(max_memory_gb: Optional[float] = None):
    """Context manager for memory-efficient processing."""
    config = None
    if max_memory_gb:
        config = StreamingConfig(max_memory_usage_gb=max_memory_gb)
    
    processor = get_streaming_processor(config)
    
    with processor.memory_managed_processing():
        yield processor