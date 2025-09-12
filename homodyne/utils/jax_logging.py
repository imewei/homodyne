"""
JAX-Specific Logging Utilities for Homodyne v2
==============================================

Enhanced logging utilities specifically designed for JAX operations in the 
homodyne scientific computing framework. Provides specialized logging for:
- JIT compilation monitoring and timing
- GPU/TPU memory usage tracking  
- Gradient computation logging
- JAX transformation debugging
- Device placement and memory management

This module extends the base logging system with JAX-aware functionality
while integrating with the existing homodyne logging infrastructure.
"""

import functools
import time
import threading
import weakref
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable, List, Union, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import tree_util
    from jax._src.lib import xla_bridge
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Memory profiling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .logging import get_logger, log_performance, log_operation
from .debug import MemoryTracker, CallTracker


@dataclass
class JITCompilationEvent:
    """Information about a JIT compilation event."""
    function_name: str
    compilation_time: float
    input_shape: Optional[str]
    device: str
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None
    compilation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def memory_delta_mb(self) -> Optional[float]:
        if self.memory_before is not None and self.memory_after is not None:
            return (self.memory_after - self.memory_before) / 1024 / 1024
        return None


@dataclass 
class JAXMemorySnapshot:
    """JAX-specific memory usage snapshot."""
    device: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    timestamp: float = field(default_factory=time.time)
    operation_context: Optional[str] = None
    
    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / 1024 / 1024
    
    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / 1024 / 1024
    
    @property
    def peak_allocated_mb(self) -> float:
        return self.peak_allocated_bytes / 1024 / 1024


class JAXCompilationTracker:
    """Track JIT compilation events and performance."""
    
    def __init__(self, max_events: int = 500):
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._compilation_counts: Dict[str, int] = defaultdict(int)
        self._compilation_times: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def record_compilation(self, event: JITCompilationEvent):
        """Record a JIT compilation event."""
        with self._lock:
            self._events.append(event)
            self._compilation_counts[event.function_name] += 1
            self._compilation_times[event.function_name].append(event.compilation_time)
            
            # Keep compilation times list bounded
            if len(self._compilation_times[event.function_name]) > 50:
                self._compilation_times[event.function_name] = \
                    self._compilation_times[event.function_name][-25:]
    
    def get_compilation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get compilation statistics for all tracked functions."""
        with self._lock:
            stats = {}
            for func_name, count in self._compilation_counts.items():
                times = self._compilation_times.get(func_name, [])
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    total_time = sum(times)
                else:
                    avg_time = min_time = max_time = total_time = 0
                
                stats[func_name] = {
                    'compilation_count': count,
                    'avg_compilation_time': avg_time,
                    'min_compilation_time': min_time,
                    'max_compilation_time': max_time,
                    'total_compilation_time': total_time
                }
            
            return stats
    
    def get_recent_compilations(self, n: int = 10) -> List[JITCompilationEvent]:
        """Get the most recent N compilation events."""
        with self._lock:
            return list(self._events)[-n:]
    
    def clear(self):
        """Clear all compilation tracking data."""
        with self._lock:
            self._events.clear()
            self._compilation_counts.clear()
            self._compilation_times.clear()


class JAXMemoryTracker:
    """Track JAX device memory usage over time."""
    
    def __init__(self, max_snapshots: int = 1000):
        self.max_snapshots = max_snapshots
        self._snapshots: deque = deque(maxlen=max_snapshots)
        self._lock = threading.Lock()
    
    def take_snapshot(self, operation_context: Optional[str] = None):
        """Take a memory snapshot for all JAX devices."""
        if not HAS_JAX:
            return
        
        try:
            devices = jax.devices()
            for device in devices:
                # Get memory info if available
                if hasattr(device, 'memory_stats'):
                    memory_stats = device.memory_stats()
                    snapshot = JAXMemorySnapshot(
                        device=str(device),
                        allocated_bytes=memory_stats.get('bytes_in_use', 0),
                        reserved_bytes=memory_stats.get('bytes_reserved', 0),
                        peak_allocated_bytes=memory_stats.get('peak_bytes_in_use', 0),
                        operation_context=operation_context
                    )
                    
                    with self._lock:
                        self._snapshots.append(snapshot)
                        
        except Exception as e:
            logger = get_logger(__name__)
            logger.debug(f"Failed to take JAX memory snapshot: {e}")
    
    def get_peak_usage(self) -> Dict[str, JAXMemorySnapshot]:
        """Get peak memory usage per device."""
        with self._lock:
            if not self._snapshots:
                return {}
            
            peak_by_device = {}
            for snapshot in self._snapshots:
                device = snapshot.device
                if device not in peak_by_device or \
                   snapshot.allocated_bytes > peak_by_device[device].allocated_bytes:
                    peak_by_device[device] = snapshot
            
            return peak_by_device
    
    def get_current_usage(self) -> Dict[str, JAXMemorySnapshot]:
        """Get current memory usage per device."""
        # Take fresh snapshots
        self.take_snapshot("current_usage_query")
        
        with self._lock:
            if not self._snapshots:
                return {}
            
            # Get most recent snapshot per device
            current_by_device = {}
            for snapshot in reversed(self._snapshots):
                if snapshot.device not in current_by_device:
                    current_by_device[snapshot.device] = snapshot
            
            return current_by_device
    
    def get_usage_timeline(self, device: Optional[str] = None, 
                          window_size: int = 100) -> List[JAXMemorySnapshot]:
        """Get memory usage timeline for a specific device or all devices."""
        with self._lock:
            snapshots = list(self._snapshots)[-window_size:]
            if device:
                snapshots = [s for s in snapshots if s.device == device]
            return snapshots
    
    def clear(self):
        """Clear all memory snapshots."""
        with self._lock:
            self._snapshots.clear()


# Global trackers
_jit_tracker = JAXCompilationTracker()
_memory_tracker = JAXMemoryTracker()


def log_jit_compilation(logger: Optional = None,
                       track_memory: bool = True,
                       log_threshold_seconds: float = 0.5):
    """
    Decorator to log JAX JIT compilation events.
    
    Args:
        logger: Logger instance to use
        track_memory: Whether to track memory usage during compilation
        log_threshold_seconds: Only log compilations taking longer than threshold
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        # Store original function for tracking
        original_func = func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Check if this is a JIT compilation (first call or cache miss)
            start_time = time.perf_counter()
            
            # Track memory before if enabled
            memory_before = None
            if track_memory:
                _memory_tracker.take_snapshot(f"before_jit_{func_name}")
                current_usage = _memory_tracker.get_current_usage()
                if current_usage:
                    memory_before = sum(s.allocated_bytes for s in current_usage.values())
            
            # Get device info
            try:
                device = str(jax.devices()[0]) if HAS_JAX else "cpu"
            except:
                device = "unknown"
            
            # Get input shape info
            input_shape = None
            if HAS_JAX and args:
                try:
                    # Try to get shape info from first array argument
                    for arg in args:
                        if hasattr(arg, 'shape'):
                            input_shape = str(arg.shape)
                            break
                        elif hasattr(arg, '__len__'):
                            input_shape = f"seq_len_{len(arg)}"
                            break
                except:
                    input_shape = "unknown"
            
            try:
                result = func(*args, **kwargs)
                compilation_time = time.perf_counter() - start_time
                
                # Track memory after if enabled  
                memory_after = None
                if track_memory:
                    _memory_tracker.take_snapshot(f"after_jit_{func_name}")
                    current_usage = _memory_tracker.get_current_usage()
                    if current_usage:
                        memory_after = sum(s.allocated_bytes for s in current_usage.values())
                
                # Only log if compilation took significant time
                if compilation_time >= log_threshold_seconds:
                    # Create compilation event
                    event = JITCompilationEvent(
                        function_name=func_name,
                        compilation_time=compilation_time,
                        input_shape=input_shape,
                        device=device,
                        memory_before=memory_before,
                        memory_after=memory_after
                    )
                    
                    # Record in tracker
                    _jit_tracker.record_compilation(event)
                    
                    # Log the compilation
                    log_msg = f"JIT compiled {func_name} in {compilation_time:.3f}s on {device}"
                    if input_shape:
                        log_msg += f" (input: {input_shape})"
                    if event.memory_delta_mb is not None:
                        log_msg += f" [Memory Δ: {event.memory_delta_mb:+.2f}MB]"
                    
                    logger.info(log_msg)
                
                return result
                
            except Exception as e:
                compilation_time = time.perf_counter() - start_time
                logger.error(f"JIT compilation failed for {func_name} after {compilation_time:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


@contextmanager
def jax_operation_context(operation_name: str,
                         logger: Optional = None,
                         track_memory: bool = True,
                         log_device_placement: bool = True):
    """
    Context manager for JAX operations with comprehensive logging.
    
    Args:
        operation_name: Name of the JAX operation
        logger: Logger instance to use
        track_memory: Whether to track memory usage
        log_device_placement: Whether to log device placement info
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.perf_counter()
    
    # Log operation start
    logger.debug(f"JAX operation started: {operation_name}")
    
    # Track memory before
    if track_memory:
        _memory_tracker.take_snapshot(f"start_{operation_name}")
    
    # Log device info
    device_info = ""
    if log_device_placement and HAS_JAX:
        try:
            devices = jax.devices()
            device_info = f" [Devices: {', '.join(str(d) for d in devices)}]"
        except:
            device_info = " [Device info unavailable]"
    
    try:
        yield logger
        
        # Success logging
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        success_msg = f"JAX operation completed: {operation_name} [{duration:.3f}s]{device_info}"
        
        # Track memory after
        if track_memory:
            _memory_tracker.take_snapshot(f"end_{operation_name}")
            peak_usage = _memory_tracker.get_peak_usage()
            if peak_usage:
                total_peak_mb = sum(s.peak_allocated_mb for s in peak_usage.values())
                success_msg += f" [Peak GPU Memory: {total_peak_mb:.1f}MB]"
        
        logger.debug(success_msg)
        
    except Exception as e:
        # Error logging
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        error_msg = f"JAX operation failed: {operation_name} [{duration:.3f}s]: {e}{device_info}"
        
        if track_memory:
            _memory_tracker.take_snapshot(f"error_{operation_name}")
        
        logger.error(error_msg)
        raise


def log_gradient_computation(logger: Optional = None,
                           include_grad_norm: bool = True,
                           norm_threshold: float = 1e-6):
    """
    Decorator to log gradient computation operations.
    
    Args:
        logger: Logger instance to use
        include_grad_norm: Whether to compute and log gradient norms
        norm_threshold: Log warning if gradient norm is below threshold
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            with jax_operation_context(f"gradient_{func_name}", logger) as op_logger:
                result = func(*args, **kwargs)
                
                # Analyze gradient if possible
                if include_grad_norm and HAS_JAX and result is not None:
                    try:
                        # Calculate gradient norm
                        if hasattr(result, '__iter__'):
                            # Multiple gradients
                            grad_norms = []
                            for i, grad in enumerate(result):
                                if hasattr(grad, 'shape'):
                                    norm = jnp.linalg.norm(grad)
                                    grad_norms.append(float(norm))
                                    
                                    if norm < norm_threshold:
                                        op_logger.warning(
                                            f"Small gradient norm in {func_name}[{i}]: {norm:.2e}"
                                        )
                            
                            if grad_norms:
                                total_norm = sum(grad_norms)
                                op_logger.debug(
                                    f"Gradient norms for {func_name}: "
                                    f"individual={grad_norms}, total={total_norm:.2e}"
                                )
                        
                        elif hasattr(result, 'shape'):
                            # Single gradient
                            norm = float(jnp.linalg.norm(result))
                            if norm < norm_threshold:
                                op_logger.warning(
                                    f"Small gradient norm in {func_name}: {norm:.2e}"
                                )
                            else:
                                op_logger.debug(f"Gradient norm for {func_name}: {norm:.2e}")
                                
                    except Exception as e:
                        op_logger.debug(f"Could not analyze gradients for {func_name}: {e}")
                
                return result
        
        return wrapper
    return decorator


def get_jax_compilation_stats() -> Dict[str, Any]:
    """Get comprehensive JAX compilation statistics."""
    return {
        'compilation_stats': _jit_tracker.get_compilation_stats(),
        'recent_compilations': [
            {
                'function': event.function_name,
                'compilation_time': event.compilation_time,
                'device': event.device,
                'input_shape': event.input_shape,
                'memory_delta_mb': event.memory_delta_mb,
                'timestamp': event.timestamp
            }
            for event in _jit_tracker.get_recent_compilations()
        ],
        'memory_stats': {
            'peak_usage': {
                device: {
                    'allocated_mb': snapshot.allocated_mb,
                    'reserved_mb': snapshot.reserved_mb,
                    'peak_allocated_mb': snapshot.peak_allocated_mb
                }
                for device, snapshot in _memory_tracker.get_peak_usage().items()
            },
            'current_usage': {
                device: {
                    'allocated_mb': snapshot.allocated_mb,
                    'reserved_mb': snapshot.reserved_mb
                }
                for device, snapshot in _memory_tracker.get_current_usage().items()
            }
        }
    }


def clear_jax_tracking_data():
    """Clear all JAX tracking data."""
    _jit_tracker.clear()
    _memory_tracker.clear()


def dump_jax_compilation_report(filepath: Optional[str] = None) -> str:
    """Dump a comprehensive JAX compilation report."""
    import json
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'jax_available': HAS_JAX,
        'jax_compilation_stats': get_jax_compilation_stats()
    }
    
    if HAS_JAX:
        try:
            report['jax_version'] = jax.__version__
            report['jax_devices'] = [str(d) for d in jax.devices()]
            report['jax_backend'] = str(jax.default_backend())
        except:
            pass
    
    report_json = json.dumps(report, indent=2, default=str)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(report_json)
    
    return report_json


# Utility functions for common JAX logging patterns
def log_array_stats(array, name: str, logger: Optional = None):
    """Log statistics about a JAX/NumPy array."""
    if logger is None:
        logger = get_logger()
    
    if not hasattr(array, 'shape'):
        logger.debug(f"{name}: not an array")
        return
    
    try:
        if HAS_JAX:
            # Use JAX functions
            stats = {
                'shape': array.shape,
                'dtype': array.dtype,
                'size': array.size,
                'mean': float(jnp.mean(array)),
                'std': float(jnp.std(array)),
                'min': float(jnp.min(array)),
                'max': float(jnp.max(array))
            }
        else:
            # Fallback to numpy
            import numpy as np
            stats = {
                'shape': array.shape,
                'dtype': array.dtype,
                'size': array.size,
                'mean': float(np.mean(array)),
                'std': float(np.std(array)),
                'min': float(np.min(array)),
                'max': float(np.max(array))
            }
        
        logger.debug(f"{name} stats: {stats}")
        
        # Check for potential issues
        if not jnp.isfinite(array).all():
            logger.warning(f"{name} contains non-finite values (NaN/Inf)")
        
        if stats['std'] == 0:
            logger.info(f"{name} has zero variance (constant array)")
            
    except Exception as e:
        logger.warning(f"Could not compute stats for {name}: {e}")