"""
Minimal logging infrastructure for the homodyne package.

Provides simplified logging with preserved API compatibility for the rebuild.
This module maintains exact function signatures from the original implementation
while removing complex features not essential for core functionality.
"""

import functools
import inspect
import logging
import time
from contextlib import contextmanager
from typing import Any, Optional


class MinimalLogger:
    """Simplified logger manager for the homodyne package."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._configured = False
        self._root_logger_name = "homodyne"
        self._initialized = True

    def configure(self, level: str = "INFO"):
        """Configure basic logging."""
        if self._configured:
            return

        # Set up root logger
        root_logger = logging.getLogger(self._root_logger_name)
        root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Add console handler if none exists
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

        self._configured = True

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with hierarchical naming."""

        # Ensure we have a fully qualified name
        if not name.startswith(self._root_logger_name):
            if name == "__main__":
                full_name = f"{self._root_logger_name}.main"
            else:
                # Extract module path from name
                if "." in name and name.startswith("homodyne"):
                    full_name = name
                else:
                    full_name = f"{self._root_logger_name}.{name}"
        else:
            full_name = name

        # Configure if not already done
        if not self._configured:
            self.configure()

        return logging.getLogger(full_name)


# Global logger manager instance
_logger_manager = MinimalLogger()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with automatic naming.

    Args:
        name: Logger name. If None, uses caller's module name.

    Returns:
        Configured logger instance.
    """
    if name is None:
        # Auto-discover caller's module
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                name = module_name
        finally:
            del frame

    return _logger_manager.get_logger(name or "unknown")


def log_calls(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
):
    """
    Decorator to log function calls.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        include_args: Whether to log function arguments.
        include_result: Whether to log function return value.
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # Log function entry
            if include_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                logger.log(level, f"Calling {func_name}({all_args})")
            else:
                logger.log(level, f"Calling {func_name}")

            try:
                result = func(*args, **kwargs)

                # Log function exit
                if include_result:
                    logger.log(level, f"Completed {func_name} -> {repr(result)}")
                else:
                    logger.log(level, f"Completed {func_name}")

                return result

            except Exception as e:
                logger.log(logging.ERROR, f"Exception in {func_name}: {e}")
                raise

        return wrapper

    return decorator


def log_performance(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    threshold: float = 0.1,
):
    """
    Decorator to log function performance.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        threshold: Minimum duration (seconds) to log.
    """

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            func_name = f"{func.__module__}.{func.__qualname__}"

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                if duration >= threshold:
                    logger.log(level, f"Performance: {func_name} completed in {duration:.3f}s")

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.log(
                    logging.ERROR,
                    f"Performance: {func_name} failed after {duration:.3f}s: {e}",
                )
                raise

        return wrapper

    return decorator


@contextmanager
def log_operation(
    operation_name: str,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
):
    """
    Context manager for logging operations.

    Args:
        operation_name: Name of the operation.
        logger: Logger to use. If None, creates one for caller's module.
        level: Logging level to use.
    """
    if logger is None:
        logger = get_logger()

    logger.log(level, f"Starting operation: {operation_name}")
    start_time = time.perf_counter()

    try:
        yield logger
        duration = time.perf_counter() - start_time
        logger.log(level, f"Completed operation: {operation_name} in {duration:.3f}s")
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.log(
            logging.ERROR,
            f"Failed operation: {operation_name} after {duration:.3f}s: {e}",
        )
        raise


# Configure default logging on import
_logger_manager.configure()