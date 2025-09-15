"""
Debugging utilities for the homodyne package.

This module provides decorators and utilities for debugging, profiling,
and monitoring the homodyne package functionality.
"""

import functools
import gc
import inspect
import logging
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# Optional dependency for memory monitoring
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .logging import get_logger


@dataclass
class CallInfo:
    """Information about a function call."""

    function_name: str
    module_name: str
    args: tuple
    kwargs: dict
    start_time: float
    end_time: Optional[float] = None
    result: Any = None
    exception: Optional[Exception] = None
    memory_before: Optional[int] = None
    memory_after: Optional[int] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

    @property
    def memory_delta(self) -> Optional[int]:
        if self.memory_before is not None and self.memory_after is not None:
            return self.memory_after - self.memory_before
        return None


class CallTracker:
    """Thread-safe tracker for function calls."""

    def __init__(self, max_calls: int = 1000):
        self.max_calls = max_calls
        self._calls: deque = deque(maxlen=max_calls)
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._call_times: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_call(self, call_info: CallInfo):
        """Add a call to the tracker."""
        with self._lock:
            self._calls.append(call_info)
            func_key = f"{call_info.module_name}.{call_info.function_name}"
            self._call_counts[func_key] += 1

            if call_info.duration is not None:
                self._call_times[func_key].append(call_info.duration)
                # Keep only recent times to avoid memory issues
                if len(self._call_times[func_key]) > 100:
                    self._call_times[func_key] = self._call_times[func_key][-50:]

    def get_recent_calls(self, n: int = 10) -> List[CallInfo]:
        """Get the most recent N calls."""
        with self._lock:
            return list(self._calls)[-n:]

    def get_call_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked calls."""
        with self._lock:
            stats = {}
            for func_key, count in self._call_counts.items():
                times = self._call_times.get(func_key, [])
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                else:
                    avg_time = min_time = max_time = 0

                stats[func_key] = {
                    "call_count": count,
                    "avg_duration": avg_time,
                    "min_duration": min_time,
                    "max_duration": max_time,
                    "total_duration": sum(times),
                }

            return stats

    def clear(self):
        """Clear all tracked calls."""
        with self._lock:
            self._calls.clear()
            self._call_counts.clear()
            self._call_times.clear()


# Global call tracker
_call_tracker = CallTracker()


def debug_calls(
    logger: Optional[logging.Logger] = None,
    include_args: bool = True,
    include_result: bool = False,
    include_stack: bool = False,
    track_memory: bool = False,
):
    """
    Decorator for debugging function calls with detailed logging.

    Args:
        logger: Logger to use. If None, creates one for the module.
        include_args: Whether to log function arguments.
        include_result: Whether to log function return value.
        include_stack: Whether to include call stack in logs.
        track_memory: Whether to track memory usage.
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__qualname__
            module_name = func.__module__

            # Get memory before if tracking
            memory_before = None
            if track_memory and HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss
                except Exception:
                    pass

            # Create call info
            call_info = CallInfo(
                function_name=func_name,
                module_name=module_name,
                args=args if include_args else (),
                kwargs=kwargs if include_args else {},
                start_time=time.perf_counter(),
                memory_before=memory_before,
            )

            # Log function entry
            entry_msg = f"ENTER {module_name}.{func_name}"
            if include_args and (args or kwargs):
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                entry_msg += f"({all_args})"

            if include_stack:
                stack = traceback.format_stack()[:-1]  # Exclude current frame
                entry_msg += f"\nCall stack:\n{''.join(stack[-5:])}"  # Last 5 frames

            logger.debug(entry_msg)

            try:
                result = func(*args, **kwargs)
                call_info.result = result
                call_info.end_time = time.perf_counter()

                # Get memory after if tracking
                if track_memory and HAS_PSUTIL:
                    try:
                        process = psutil.Process()
                        call_info.memory_after = process.memory_info().rss
                    except Exception:
                        pass

                # Log function exit
                exit_msg = f"EXIT  {module_name}.{func_name} [Duration: {call_info.duration:.3f}s]"
                if include_result:
                    exit_msg += f" -> {repr(result)}"
                if track_memory and call_info.memory_delta is not None:
                    exit_msg += (
                        f" [Memory Δ: {call_info.memory_delta / 1024 / 1024:.2f}MB]"
                    )

                logger.debug(exit_msg)

                # Track the call
                _call_tracker.add_call(call_info)

                return result

            except Exception as e:
                call_info.exception = e
                call_info.end_time = time.perf_counter()

                if track_memory and HAS_PSUTIL:
                    try:
                        process = psutil.Process()
                        call_info.memory_after = process.memory_info().rss
                    except Exception:
                        pass

                error_msg = f"ERROR {module_name}.{func_name} [Duration: {call_info.duration:.3f}s]: {e}"
                if track_memory and call_info.memory_delta is not None:
                    error_msg += (
                        f" [Memory Δ: {call_info.memory_delta / 1024 / 1024:.2f}MB]"
                    )

                logger.error(error_msg)
                logger.debug(traceback.format_exc())

                # Track the failed call
                _call_tracker.add_call(call_info)

                raise

        return wrapper

    return decorator


class MemoryTracker:
    """Track memory usage over time."""

    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self._samples: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()

    def sample(self, label: str = ""):
        """Take a memory sample."""
        if not HAS_PSUTIL:
            return

        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            with self._lock:
                self._samples.append(
                    {
                        "timestamp": time.time(),
                        "label": label,
                        "rss": memory_info.rss,
                        "vms": memory_info.vms,
                        "percent": process.memory_percent(),
                    }
                )
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to sample memory: {e}")

    def get_peak_usage(self) -> Dict[str, Any]:
        """Get peak memory usage."""
        with self._lock:
            if not self._samples:
                return {}

            peak_rss = max(self._samples, key=lambda x: x["rss"])
            peak_vms = max(self._samples, key=lambda x: x["vms"])

            return {
                "peak_rss": peak_rss,
                "peak_vms": peak_vms,
                "current": self._samples[-1] if self._samples else None,
            }

    def get_usage_trend(self, window_size: int = 50) -> List[Dict[str, Any]]:
        """Get recent memory usage trend."""
        with self._lock:
            return list(self._samples)[-window_size:]

    def clear(self):
        """Clear all samples."""
        with self._lock:
            self._samples.clear()


# Global memory tracker
_memory_tracker = MemoryTracker()


def profile_memory(
    logger: Optional[logging.Logger] = None,
    sample_interval: float = 1.0,
    log_threshold_mb: float = 100.0,
):
    """
    Decorator to profile memory usage of a function.

    Args:
        logger: Logger to use. If None, creates one for the module.
        sample_interval: Interval between memory samples (seconds).
        log_threshold_mb: Log warning if memory usage exceeds this (MB).
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Sample memory before
            _memory_tracker.sample(f"before_{func_name}")
            start_memory = 0
            if HAS_PSUTIL:
                peak_usage = _memory_tracker.get_peak_usage()
                if peak_usage.get("current"):
                    start_memory = peak_usage["current"].get("rss", 0)

            # Run function
            try:
                result = func(*args, **kwargs)

                # Sample memory after
                _memory_tracker.sample(f"after_{func_name}")
                end_memory = 0
                if HAS_PSUTIL:
                    peak_usage = _memory_tracker.get_peak_usage()
                    if peak_usage.get("current"):
                        end_memory = peak_usage["current"].get("rss", 0)

                # Calculate memory delta
                memory_delta_mb = (end_memory - start_memory) / 1024 / 1024

                # Log if significant memory usage
                if abs(memory_delta_mb) > log_threshold_mb:
                    logger.warning(
                        f"Memory usage in {func_name}: {memory_delta_mb:+.2f}MB "
                        f"(before: {start_memory / 1024 / 1024:.2f}MB, "
                        f"after: {end_memory / 1024 / 1024:.2f}MB)"
                    )
                else:
                    logger.debug(
                        f"Memory delta for {func_name}: {memory_delta_mb:+.2f}MB"
                    )

                return result

            except Exception as e:
                # Sample memory on error too
                _memory_tracker.sample(f"error_{func_name}")
                logger.error(
                    f"Memory profiling interrupted by exception in {func_name}: {e}"
                )
                raise

        return wrapper

    return decorator


class PerformanceProfiler:
    """Detailed performance profiler."""

    def __init__(self):
        self._profiles: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def start_profile(self, name: str):
        """Start profiling a named operation."""
        with self._lock:
            self._profiles[name] = {
                "start_time": time.perf_counter(),
                "start_cpu_time": time.process_time(),
                "checkpoints": [],
            }

    def checkpoint(self, name: str, label: str):
        """Add a checkpoint to a profile."""
        with self._lock:
            if name in self._profiles:
                now = time.perf_counter()
                self._profiles[name]["checkpoints"].append(
                    {
                        "label": label,
                        "time": now,
                        "elapsed": now - self._profiles[name]["start_time"],
                    }
                )

    def end_profile(self, name: str) -> Dict[str, Any]:
        """End profiling and return results."""
        with self._lock:
            if name not in self._profiles:
                return {}

            profile = self._profiles.pop(name)
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()

            return {
                "total_time": end_time - profile["start_time"],
                "cpu_time": end_cpu_time - profile["start_cpu_time"],
                "checkpoints": profile["checkpoints"],
                "efficiency": (end_cpu_time - profile["start_cpu_time"])
                / (end_time - profile["start_time"]),
            }


# Global performance profiler
_performance_profiler = PerformanceProfiler()


def profile_performance(
    logger: Optional[logging.Logger] = None, detailed: bool = False
):
    """
    Decorator for detailed performance profiling.

    Args:
        logger: Logger to use. If None, creates one for the module.
        detailed: Whether to include detailed CPU/memory information.
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Start profiling
            start_time = time.perf_counter()
            start_cpu = time.process_time()

            if detailed:
                _performance_profiler.start_profile(func_name)
                gc.collect()  # Clean up before measuring
                start_objects = len(gc.get_objects())

            try:
                result = func(*args, **kwargs)

                # End profiling
                end_time = time.perf_counter()
                end_cpu = time.process_time()
                wall_time = end_time - start_time
                cpu_time = end_cpu - start_cpu

                # Basic performance log
                logger.info(
                    f"Performance {func_name}: Wall={wall_time:.3f}s, CPU={cpu_time:.3f}s"
                )

                if detailed:
                    gc.collect()
                    end_objects = len(gc.get_objects())
                    object_delta = end_objects - start_objects

                    profile_data = _performance_profiler.end_profile(func_name)

                    logger.debug(
                        f"Detailed performance {func_name}: "
                        f"Efficiency={cpu_time / wall_time:.2%}, "
                        f"Objects Δ={object_delta:+d}"
                    )

                return result

            except Exception as e:
                end_time = time.perf_counter()
                wall_time = end_time - start_time
                logger.error(
                    f"Performance {func_name} failed after {wall_time:.3f}s: {e}"
                )

                if detailed:
                    _performance_profiler.end_profile(func_name)

                raise

        return wrapper

    return decorator


@contextmanager
def debug_context(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    track_memory: bool = True,
    track_performance: bool = True,
):
    """
    Context manager for debugging operations.

    Args:
        operation_name: Name of the operation.
        logger: Logger to use.
        track_memory: Whether to track memory usage.
        track_performance: Whether to track performance.
    """
    if logger is None:
        logger = get_logger()

    start_time = time.perf_counter()
    start_memory = None

    if track_memory:
        _memory_tracker.sample(f"start_{operation_name}")
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except Exception:
                pass

    if track_performance:
        _performance_profiler.start_profile(operation_name)

    logger.debug(f"DEBUG CONTEXT START: {operation_name}")

    try:
        yield logger

        # Success logging
        end_time = time.perf_counter()
        duration = end_time - start_time

        success_msg = f"DEBUG CONTEXT END: {operation_name} [Duration: {duration:.3f}s]"

        if track_memory:
            _memory_tracker.sample(f"end_{operation_name}")
            if HAS_PSUTIL:
                try:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss
                    if start_memory:
                        memory_delta = (end_memory - start_memory) / 1024 / 1024
                        success_msg += f" [Memory Δ: {memory_delta:+.2f}MB]"
                except Exception:
                    pass

        logger.debug(success_msg)

        if track_performance:
            profile_data = _performance_profiler.end_profile(operation_name)
            if profile_data:
                logger.debug(
                    f"Performance profile for {operation_name}: {profile_data}"
                )

    except Exception as e:
        # Error logging
        end_time = time.perf_counter()
        duration = end_time - start_time

        error_msg = (
            f"DEBUG CONTEXT ERROR: {operation_name} [Duration: {duration:.3f}s]: {e}"
        )

        if track_memory:
            _memory_tracker.sample(f"error_{operation_name}")

        logger.error(error_msg)
        logger.debug(traceback.format_exc())

        if track_performance:
            _performance_profiler.end_profile(operation_name)

        raise


def get_debug_stats() -> Dict[str, Any]:
    """Get comprehensive debugging statistics."""
    return {
        "call_stats": _call_tracker.get_call_stats(),
        "recent_calls": [
            {
                "function": f"{call.module_name}.{call.function_name}",
                "duration": call.duration,
                "memory_delta": call.memory_delta,
                "had_exception": call.exception is not None,
            }
            for call in _call_tracker.get_recent_calls()
        ],
        "memory_stats": _memory_tracker.get_peak_usage(),
        "memory_trend": _memory_tracker.get_usage_trend(20),
    }


def clear_debug_data():
    """Clear all debugging data."""
    _call_tracker.clear()
    _memory_tracker.clear()


def dump_debug_report(filepath: Optional[str] = None) -> str:
    """Dump a comprehensive debug report."""
    import json
    from datetime import datetime

    report = {
        "timestamp": datetime.now().isoformat(),
        "debug_stats": get_debug_stats(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform,
        },
    }

    if HAS_PSUTIL:
        try:
            process = psutil.Process()
            report["system_info"].update(
                {
                    "memory_info": dict(process.memory_info()._asdict()),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                }
            )
        except Exception:
            pass

    report_json = json.dumps(report, indent=2, default=str)

    if filepath:
        with open(filepath, "w") as f:
            f.write(report_json)

    return report_json
