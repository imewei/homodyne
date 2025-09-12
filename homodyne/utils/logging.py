"""
Comprehensive logging infrastructure for the homodyne package.

This module provides a sophisticated logging system with auto-discovery,
hierarchical naming, performance monitoring, and extensible formatting.
"""

import logging
import logging.handlers
import os
import sys
import time
import functools
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Callable, Any, Union, List
from datetime import datetime
import traceback
import inspect


class HomodyneFormatter(logging.Formatter):
    """Enhanced formatter with context-aware formatting."""
    
    def __init__(self, include_function: bool = True, include_module: bool = True):
        self.include_function = include_function
        self.include_module = include_module
        super().__init__()
    
    def format(self, record):
        # Add custom attributes
        if not hasattr(record, 'module_name'):
            record.module_name = record.name.split('.')[-1] if '.' in record.name else record.name
        
        if not hasattr(record, 'function_name'):
            record.function_name = getattr(record, 'funcName', 'unknown')
        
        # Choose format based on log level
        if record.levelno >= logging.ERROR:
            fmt = self._error_format()
        elif record.levelno >= logging.WARNING:
            fmt = self._warning_format()
        elif record.levelno >= logging.INFO:
            fmt = self._info_format()
        else:
            fmt = self._debug_format()
        
        formatter = logging.Formatter(fmt)
        return formatter.format(record)
    
    def _error_format(self):
        return "%(asctime)s | ERROR | %(name)s.%(function_name)s:%(lineno)d | %(message)s"
    
    def _warning_format(self):
        return "%(asctime)s | WARN  | %(module_name)s.%(function_name)s | %(message)s"
    
    def _info_format(self):
        if self.include_module and self.include_function:
            return "%(asctime)s | INFO  | %(module_name)s.%(function_name)s | %(message)s"
        elif self.include_module:
            return "%(asctime)s | INFO  | %(module_name)s | %(message)s"
        else:
            return "%(asctime)s | INFO  | %(message)s"
    
    def _debug_format(self):
        return "%(asctime)s | DEBUG | %(name)s.%(function_name)s:%(lineno)d | %(message)s"


class ConsoleFormatter(HomodyneFormatter):
    """Colorized console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        colored_record = logging.makeLogRecord(record.__dict__)
        level_color = self.COLORS.get(record.levelname, '')
        reset_color = self.COLORS['RESET']
        
        # Color the level name
        colored_record.levelname = f"{level_color}{record.levelname}{reset_color}"
        
        return super().format(colored_record)


class ContextFilter(logging.Filter):
    """Filter that adds contextual information to log records."""
    
    def filter(self, record):
        # Add thread information
        record.thread_name = threading.current_thread().name
        
        # Add caller information if not already present
        if not hasattr(record, 'caller_module'):
            frame = inspect.currentframe()
            try:
                # Go up the stack to find the actual caller
                while frame:
                    frame = frame.f_back
                    if frame and frame.f_code.co_filename != __file__:
                        record.caller_module = os.path.basename(frame.f_code.co_filename)
                        record.caller_function = frame.f_code.co_name
                        record.caller_line = frame.f_lineno
                        break
            finally:
                del frame
        
        return True


class PerformanceFilter(logging.Filter):
    """Filter for performance-related logging."""
    
    def __init__(self, min_duration: float = 0.1):
        super().__init__()
        self.min_duration = min_duration
    
    def filter(self, record):
        # Only log performance records that exceed minimum duration
        if hasattr(record, 'duration') and record.duration < self.min_duration:
            return False
        return True


class LoggerManager:
    """Central manager for all loggers in the homodyne package."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._loggers: Dict[str, logging.Logger] = {}
        self._handlers: List[logging.Handler] = []
        self._custom_handlers: Dict[str, Callable] = {}
        self._root_logger_name = 'homodyne'
        self._configured = False
        self._log_dir = None
        self._initialized = True
    
    def configure_from_json(self, config: Dict[str, Any]):
        """
        Configure logging from JSON configuration dictionary.
        
        Args:
            config: Dictionary containing logging configuration
        """
        if self._configured:
            return
            
        logging_config = config.get('logging', {})
        
        # Check if logging is enabled
        if not logging_config.get('enabled', True):
            return
            
        # Extract configuration values
        global_level = self._parse_log_level(logging_config.get('level', 'INFO'))
        
        # Console configuration
        console_config = logging_config.get('console', {})
        console_enabled = console_config.get('enabled', True)
        console_level = self._parse_log_level(console_config.get('level', global_level))
        console_format = console_config.get('format', 'detailed')
        console_colors = console_config.get('colors', True)
        
        # File configuration
        file_config = logging_config.get('file', {})
        file_enabled = file_config.get('enabled', True)
        file_level = self._parse_log_level(file_config.get('level', 'DEBUG'))
        log_path = file_config.get('path', '~/.homodyne/logs/')
        log_filename = file_config.get('filename', 'homodyne.log')
        max_size_mb = file_config.get('max_size_mb', 10)
        backup_count = file_config.get('backup_count', 5)
        file_format = file_config.get('format', 'detailed')
        
        # Performance configuration
        perf_config = logging_config.get('performance', {})
        perf_enabled = perf_config.get('enabled', True)
        perf_level = self._parse_log_level(perf_config.get('level', 'INFO'))
        perf_filename = perf_config.get('filename', 'performance.log')
        perf_threshold = perf_config.get('threshold_seconds', 0.1)
        
        # Module-specific levels
        module_levels = logging_config.get('modules', {})
        
        # Debug configuration
        debug_config = logging_config.get('debug', {})
        
        # Set up log directory
        log_dir = Path(log_path).expanduser()
        self._log_dir = log_dir
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger(self._root_logger_name)
        root_logger.setLevel(global_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level)
            
            if console_colors and console_format == 'detailed':
                console_formatter = ConsoleFormatter(include_function=True, include_module=True)
            elif console_format == 'detailed':
                console_formatter = HomodyneFormatter(include_function=True, include_module=True)
            else:
                console_formatter = HomodyneFormatter(include_function=False, include_module=True)
            
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(ContextFilter())
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)
        
        # Add file handler
        if file_enabled:
            log_file = self._log_dir / log_filename
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size_mb * 1024 * 1024,
                backupCount=backup_count
            )
            file_handler.setLevel(file_level)
            
            file_formatter = HomodyneFormatter(include_function=True, include_module=True)
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(ContextFilter())
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)
            
            # Add separate performance log
            if perf_enabled:
                perf_file = self._log_dir / perf_filename
                perf_handler = logging.handlers.RotatingFileHandler(
                    perf_file,
                    maxBytes=max_size_mb * 1024 * 1024,
                    backupCount=backup_count
                )
                perf_handler.setLevel(perf_level)
                perf_handler.addFilter(PerformanceFilter(perf_threshold))
                
                perf_formatter = logging.Formatter(
                    '%(asctime)s | PERF | %(name)s | %(message)s | Duration: %(duration).3fs'
                )
                perf_handler.setFormatter(perf_formatter)
                self._handlers.append(perf_handler)
        
        # Apply module-specific log levels
        for module_name, level in module_levels.items():
            if module_name and level:
                module_level = self._parse_log_level(level)
                module_logger = logging.getLogger(module_name)
                module_logger.setLevel(module_level)
        
        self._configured = True
    
    def _parse_log_level(self, level: Union[str, int]) -> int:
        """Parse log level from string or int."""
        if isinstance(level, int):
            return level
        
        level_str = str(level).upper()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO, 
            'WARNING': logging.WARNING,
            'WARN': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        
        return level_map.get(level_str, logging.INFO)

    def configure(self, 
                  level: Union[str, int] = logging.INFO,
                  log_dir: Optional[Union[str, Path]] = None,
                  console_output: bool = True,
                  file_output: bool = True,
                  max_file_size: int = 10 * 1024 * 1024,  # 10MB
                  backup_count: int = 5,
                  format_style: str = 'enhanced'):
        """Configure the logging system with legacy parameters."""
        
        if self._configured:
            return
        
        # Set up log directory
        if log_dir:
            self._log_dir = Path(log_dir)
        else:
            self._log_dir = Path.home() / '.homodyne' / 'logs'
        
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger(self._root_logger_name)
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Add console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            
            if format_style == 'enhanced':
                console_formatter = ConsoleFormatter(include_function=True, include_module=True)
            else:
                console_formatter = HomodyneFormatter(include_function=False, include_module=True)
            
            console_handler.setFormatter(console_formatter)
            console_handler.addFilter(ContextFilter())
            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)
        
        # Add file handler
        if file_output:
            log_file = self._log_dir / 'homodyne.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all messages
            
            file_formatter = HomodyneFormatter(include_function=True, include_module=True)
            file_handler.setFormatter(file_formatter)
            file_handler.addFilter(ContextFilter())
            root_logger.addHandler(file_handler)
            self._handlers.append(file_handler)
            
            # Add separate performance log
            perf_file = self._log_dir / 'performance.log'
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=max_file_size,
                backupCount=backup_count
            )
            perf_handler.setLevel(logging.DEBUG)
            perf_handler.addFilter(PerformanceFilter())
            
            perf_formatter = logging.Formatter(
                '%(asctime)s | PERF | %(name)s | %(message)s | Duration: %(duration).3fs'
            )
            perf_handler.setFormatter(perf_formatter)
            self._handlers.append(perf_handler)
        
        self._configured = True
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with hierarchical naming."""
        
        # Ensure we have a fully qualified name
        if not name.startswith(self._root_logger_name):
            if name == '__main__':
                full_name = f"{self._root_logger_name}.main"
            else:
                # Extract module path from name
                if '.' in name and name.startswith('homodyne'):
                    full_name = name
                else:
                    full_name = f"{self._root_logger_name}.{name}"
        else:
            full_name = name
        
        # Return existing logger or create new one
        if full_name in self._loggers:
            return self._loggers[full_name]
        
        logger = logging.getLogger(full_name)
        
        # Set up logger properties
        if not self._configured:
            self.configure()
        
        # Store reference
        self._loggers[full_name] = logger
        
        return logger
    
    def register_handler(self, name: str, handler_factory: Callable):
        """Register a custom handler factory."""
        self._custom_handlers[name] = handler_factory
    
    def add_handler(self, handler: logging.Handler, logger_name: Optional[str] = None):
        """Add a handler to specified logger or root logger."""
        if logger_name:
            logger = self.get_logger(logger_name)
        else:
            logger = logging.getLogger(self._root_logger_name)
        
        logger.addHandler(handler)
        self._handlers.append(handler)
    
    def get_log_directory(self) -> Path:
        """Get the configured log directory."""
        return self._log_dir
    
    def list_loggers(self) -> List[str]:
        """List all registered loggers."""
        return list(self._loggers.keys())
    
    def set_level(self, level: Union[str, int], logger_name: Optional[str] = None):
        """Set logging level for specified logger or all loggers."""
        if logger_name:
            logger = self.get_logger(logger_name)
            logger.setLevel(level)
        else:
            # Set level for root logger and all handlers
            root_logger = logging.getLogger(self._root_logger_name)
            root_logger.setLevel(level)
            for handler in self._handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(level)


# Global logger manager instance
_logger_manager = LoggerManager()


def configure_logging(**kwargs):
    """Configure the global logging system."""
    _logger_manager.configure(**kwargs)


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
                module_name = caller_frame.f_globals.get('__name__', 'unknown')
                name = module_name
        finally:
            del frame
    
    return _logger_manager.get_logger(name or 'unknown')


def log_calls(logger: Optional[logging.Logger] = None, 
              level: int = logging.DEBUG,
              include_args: bool = False,
              include_result: bool = False):
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
                args_str = ', '.join([repr(arg) for arg in args])
                kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ', '.join(filter(None, [args_str, kwargs_str]))
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


def log_performance(logger: Optional[logging.Logger] = None,
                   level: int = logging.INFO,
                   threshold: float = 0.1):
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
                    # Create a log record with duration attribute for PerformanceFilter
                    record = logger.makeRecord(
                        logger.name, level, func.__code__.co_filename,
                        func.__code__.co_firstlineno, 
                        f"Performance: {func_name} completed",
                        (), None, func=func.__name__
                    )
                    record.duration = duration
                    logger.handle(record)
                
                return result
            
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.log(logging.ERROR, 
                          f"Performance: {func_name} failed after {duration:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


@contextmanager
def log_operation(operation_name: str, 
                  logger: Optional[logging.Logger] = None,
                  level: int = logging.INFO):
    """
    Context manager for logging operations.
    
    Args:
        operation_name: Name of the operation.
        logger: Logger to use. If None, creates one for caller's module.
        level: Logging level to use.
    """
    if logger is None:
        logger = get_logger()
    
    start_time = time.perf_counter()
    logger.log(level, f"Starting operation: {operation_name}")
    
    try:
        yield logger
        duration = time.perf_counter() - start_time
        logger.log(level, f"Completed operation: {operation_name} in {duration:.3f}s")
    
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.log(logging.ERROR, 
                  f"Failed operation: {operation_name} after {duration:.3f}s: {e}")
        raise


class LoggingPlugin:
    """Base class for logging plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    def create_handler(self) -> logging.Handler:
        """Create and return a custom handler."""
        raise NotImplementedError
    
    def configure(self, **kwargs):
        """Configure the plugin with options."""
        pass


def register_plugin(plugin: LoggingPlugin, logger_name: Optional[str] = None):
    """Register a logging plugin."""
    handler = plugin.create_handler()
    _logger_manager.add_handler(handler, logger_name)


# Example plugin implementations
class SlackPlugin(LoggingPlugin):
    """Plugin for sending logs to Slack."""
    
    def __init__(self, webhook_url: str):
        super().__init__("slack")
        self.webhook_url = webhook_url
    
    def create_handler(self) -> logging.Handler:
        # This would implement actual Slack integration
        # For now, return a NullHandler as placeholder
        return logging.NullHandler()


class DatabasePlugin(LoggingPlugin):
    """Plugin for logging to database."""
    
    def __init__(self, connection_string: str):
        super().__init__("database")
        self.connection_string = connection_string
    
    def create_handler(self) -> logging.Handler:
        # This would implement actual database integration
        # For now, return a NullHandler as placeholder
        return logging.NullHandler()


# Convenience functions for environment-aware configuration
def configure_development():
    """Configure logging for development environment."""
    configure_logging(
        level=logging.DEBUG,
        console_output=True,
        file_output=True,
        format_style='enhanced'
    )


def configure_production():
    """Configure logging for production environment."""
    configure_logging(
        level=logging.INFO,
        console_output=False,
        file_output=True,
        format_style='standard'
    )


def configure_testing():
    """Configure logging for testing environment."""
    configure_logging(
        level=logging.WARNING,
        console_output=False,
        file_output=False
    )


# Auto-configuration based on environment
def auto_configure():
    """Automatically configure logging based on environment."""
    env = os.getenv('HOMODYNE_ENV', 'development').lower()
    
    if env == 'production':
        configure_production()
    elif env == 'testing':
        configure_testing()
    else:
        configure_development()


# JSON Configuration Support Functions
def configure_logging_from_json(config: Dict[str, Any]):
    """
    Configure logging from a JSON configuration dictionary.
    
    Args:
        config: Dictionary containing JSON configuration with logging section
    """
    _logger_manager.configure_from_json(config)


def load_and_apply_logging_config(config_path: str = None, config_dict: Dict[str, Any] = None):
    """
    Load JSON configuration and apply logging settings.
    
    Args:
        config_path: Path to JSON configuration file
        config_dict: Configuration dictionary (alternative to config_path)
        
    Returns:
        Configuration dictionary that was loaded/used
    """
    if config_path and config_dict:
        raise ValueError("Provide either config_path or config_dict, not both")
    
    if config_path:
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
    elif config_dict:
        config = config_dict
    else:
        raise ValueError("Must provide either config_path or config_dict")
    
    # Apply logging configuration
    configure_logging_from_json(config)
    
    return config


def validate_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate logging configuration and return any errors or warnings.
    
    Args:
        config: Configuration dictionary containing logging section
        
    Returns:
        Dictionary with validation results: {'errors': [...], 'warnings': [...]}
    """
    errors = []
    warnings = []
    
    logging_config = config.get('logging', {})
    
    if not logging_config:
        warnings.append("No logging configuration found, using defaults")
        return {'errors': errors, 'warnings': warnings}
    
    # Validate log levels
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    global_level = logging_config.get('level', 'INFO')
    if isinstance(global_level, str) and global_level.upper() not in valid_levels:
        errors.append(f"Invalid global log level: {global_level}")
    
    console_config = logging_config.get('console', {})
    if console_config.get('enabled', True):
        console_level = console_config.get('level', global_level)
        if isinstance(console_level, str) and console_level.upper() not in valid_levels:
            errors.append(f"Invalid console log level: {console_level}")
    
    file_config = logging_config.get('file', {})
    if file_config.get('enabled', True):
        file_level = file_config.get('level', 'DEBUG')
        if isinstance(file_level, str) and file_level.upper() not in valid_levels:
            errors.append(f"Invalid file log level: {file_level}")
        
        # Validate file settings
        max_size = file_config.get('max_size_mb', 10)
        if not isinstance(max_size, (int, float)) or max_size <= 0:
            errors.append(f"Invalid max_size_mb: {max_size} (must be positive number)")
        
        backup_count = file_config.get('backup_count', 5)
        if not isinstance(backup_count, int) or backup_count < 0:
            errors.append(f"Invalid backup_count: {backup_count} (must be non-negative integer)")
    
    # Validate module levels
    module_levels = logging_config.get('modules', {})
    for module_name, level in module_levels.items():
        if isinstance(level, str) and level.upper() not in valid_levels:
            warnings.append(f"Invalid log level for module {module_name}: {level}")
    
    # Validate performance configuration
    perf_config = logging_config.get('performance', {})
    if perf_config.get('enabled', True):
        threshold = perf_config.get('threshold_seconds', 0.1)
        if not isinstance(threshold, (int, float)) or threshold < 0:
            warnings.append(f"Invalid performance threshold: {threshold} (should be non-negative number)")
    
    return {'errors': errors, 'warnings': warnings}


def get_effective_logging_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the effective logging configuration with all defaults applied.
    
    Args:
        config: Configuration dictionary containing logging section
        
    Returns:
        Complete logging configuration with all defaults applied
    """
    logging_config = config.get('logging', {})
    
    # Apply defaults
    effective_config = {
        'enabled': logging_config.get('enabled', True),
        'level': logging_config.get('level', 'INFO'),
        'console': {
            'enabled': logging_config.get('console', {}).get('enabled', True),
            'level': logging_config.get('console', {}).get('level', logging_config.get('level', 'INFO')),
            'format': logging_config.get('console', {}).get('format', 'detailed'),
            'colors': logging_config.get('console', {}).get('colors', True)
        },
        'file': {
            'enabled': logging_config.get('file', {}).get('enabled', True),
            'level': logging_config.get('file', {}).get('level', 'DEBUG'),
            'path': logging_config.get('file', {}).get('path', '~/.homodyne/logs/'),
            'filename': logging_config.get('file', {}).get('filename', 'homodyne.log'),
            'max_size_mb': logging_config.get('file', {}).get('max_size_mb', 10),
            'backup_count': logging_config.get('file', {}).get('backup_count', 5),
            'format': logging_config.get('file', {}).get('format', 'detailed')
        },
        'performance': {
            'enabled': logging_config.get('performance', {}).get('enabled', True),
            'level': logging_config.get('performance', {}).get('level', 'INFO'),
            'filename': logging_config.get('performance', {}).get('filename', 'performance.log'),
            'threshold_seconds': logging_config.get('performance', {}).get('threshold_seconds', 0.1)
        },
        'modules': logging_config.get('modules', {}),
        'debug': {
            'trace_calls': logging_config.get('debug', {}).get('trace_calls', False),
            'track_memory': logging_config.get('debug', {}).get('track_memory', False),
            'profile_performance': logging_config.get('debug', {}).get('profile_performance', True)
        }
    }
    
    return effective_config


def reset_logging_configuration():
    """Reset logging configuration to allow reconfiguration."""
    global _logger_manager
    _logger_manager._configured = False
    
    # Clear all handlers from root logger
    root_logger = logging.getLogger(_logger_manager._root_logger_name)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    _logger_manager._handlers.clear()


# Environment variable support for logging configuration
def apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to logging configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with environment overrides applied
    """
    import os
    
    # Make a copy to avoid modifying the original
    config = config.copy()
    logging_config = config.get('logging', {}).copy()
    
    # Environment variable mappings
    env_mappings = {
        'HOMODYNE_LOG_LEVEL': ('level',),
        'HOMODYNE_LOG_CONSOLE_ENABLED': ('console', 'enabled'),
        'HOMODYNE_LOG_CONSOLE_LEVEL': ('console', 'level'),
        'HOMODYNE_LOG_FILE_ENABLED': ('file', 'enabled'),
        'HOMODYNE_LOG_FILE_LEVEL': ('file', 'level'),
        'HOMODYNE_LOG_FILE_PATH': ('file', 'path'),
        'HOMODYNE_LOG_PERFORMANCE_ENABLED': ('performance', 'enabled'),
        'HOMODYNE_LOG_PERFORMANCE_THRESHOLD': ('performance', 'threshold_seconds'),
        'HOMODYNE_LOG_DEBUG_TRACE': ('debug', 'trace_calls'),
        'HOMODYNE_LOG_DEBUG_MEMORY': ('debug', 'track_memory'),
        'HOMODYNE_LOG_DEBUG_PROFILE': ('debug', 'profile_performance')
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the nested config location
            current = logging_config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            # Convert value to appropriate type
            final_key = config_path[-1]
            if final_key in ['enabled', 'trace_calls', 'track_memory', 'profile_performance']:
                current[final_key] = value.lower() in ('true', '1', 'yes', 'on')
            elif final_key in ['threshold_seconds']:
                try:
                    current[final_key] = float(value)
                except ValueError:
                    pass  # Keep original value if conversion fails
            else:
                current[final_key] = value
    
    config['logging'] = logging_config
    return config


# Initialize with auto-configuration if not already configured
if not _logger_manager._configured:
    try:
        auto_configure()
    except Exception:
        # Fallback to basic configuration
        configure_logging(level=logging.INFO, console_output=True, file_output=False)