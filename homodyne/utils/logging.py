"""Structured logging utilities for the homodyne package.

Provides a lightweight but flexible logging system that matches the CMC
reimplementation requirements: contextual log prefixes, configurable console
and rotating file handlers, and helpers for performance monitoring.
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar

# Type variables for decorators
F = TypeVar("F", bound=Callable[..., Any])

# Type alias for logger types
LoggerType = logging.Logger | logging.LoggerAdapter[logging.Logger]

DEFAULT_FORMAT_DETAILED = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_FORMAT_SIMPLE = "%(levelname)-8s | %(message)s"


def _resolve_level(level: str | int | None) -> int | None:
    """Convert string/int log level to logging level constant."""
    if level is None:
        return None
    if isinstance(level, int):
        return level
    return getattr(logging, str(level).upper(), logging.INFO)


class _ColorFormatter(logging.Formatter):
    """Optional ANSI color formatter for console logging."""

    COLOR_MAP = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, datefmt: str | None, use_color: bool) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color and original_levelname in self.COLOR_MAP:
            record.levelname = (
                f"{self.COLOR_MAP[original_levelname]}{original_levelname}{self.RESET}"
            )
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class _ContextAdapter(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with structured context."""

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        if not self.extra:
            return msg, kwargs

        context_parts = [
            f"{key}={value}"
            for key, value in self.extra.items()
            if value is not None and value != ""
        ]
        if context_parts:
            msg = f"[{' '.join(context_parts)}] {msg}"
        return msg, kwargs


class MinimalLogger:
    """Configurable logger manager for the homodyne package."""

    _instance: MinimalLogger | None = None
    _initialized: bool
    _configured: bool
    _root_logger_name: str

    def __new__(cls) -> MinimalLogger:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self._configured = False
        self._root_logger_name = "homodyne"
        self._initialized = True

    @staticmethod
    def _build_formatter(
        format_name: str = "detailed",
        use_color: bool = False,
    ) -> logging.Formatter:
        fmt = (
            DEFAULT_FORMAT_SIMPLE
            if format_name == "simple"
            else DEFAULT_FORMAT_DETAILED
        )
        return _ColorFormatter(
            fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S", use_color=use_color
        )

    def _clear_managed_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            if getattr(handler, "_homodyne_managed", False):
                logger.removeHandler(handler)
                handler.close()

    def configure(
        self,
        level: str | int = "INFO",
        *,
        console_level: str | int | None = None,
        console_format: str = "detailed",
        console_colors: bool = False,
        file_path: str | Path | None = None,
        file_level: str | int | None = None,
        max_size_mb: int = 10,
        backup_count: int = 5,
        module_levels: Mapping[str, str | int] | None = None,
        force: bool = False,
    ) -> Path | None:
        """Configure homodyne logging.

        Returns the file path if a file handler is created.
        """
        root_logger = logging.getLogger(self._root_logger_name)

        if force:
            self._clear_managed_handlers(root_logger)

        root_level_candidates = [_resolve_level(level)]
        if console_level is not False:
            root_level_candidates.append(_resolve_level(console_level))
        if file_level is not False:
            root_level_candidates.append(_resolve_level(file_level))
        root_level = min(lvl for lvl in root_level_candidates if lvl is not None)
        root_logger.setLevel(root_level)

        # Console handler
        console_handler: logging.Handler | None = None
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler,
                logging.FileHandler,
            ):
                console_handler = handler
                break

        if console_level is not False:
            if console_handler is None:
                console_handler = logging.StreamHandler()
                console_handler._homodyne_managed = True  # type: ignore[attr-defined]
                root_logger.addHandler(console_handler)
            console_handler.setLevel(_resolve_level(console_level) or root_level)
            console_handler.setFormatter(
                self._build_formatter(console_format, use_color=console_colors)
            )

        # File handler
        created_file: Path | None = None
        if file_path:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            created_file = file_path

            max_bytes = int(max_size_mb * 1024 * 1024)
            if max_bytes > 0:
                file_handler: logging.Handler = RotatingFileHandler(
                    file_path,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                )
            else:
                file_handler = logging.FileHandler(file_path)
            file_handler._homodyne_managed = True  # type: ignore[attr-defined]
            file_handler.setLevel(_resolve_level(file_level) or root_level)
            file_handler.setFormatter(
                logging.Formatter(
                    DEFAULT_FORMAT_DETAILED,
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            root_logger.addHandler(file_handler)

        # Module-specific overrides
        if module_levels:
            for module_name, module_level in module_levels.items():
                logging.getLogger(module_name).setLevel(
                    _resolve_level(module_level) or root_level
                )

        self._configured = True
        return created_file

    def configure_from_dict(
        self,
        logging_config: Mapping[str, Any] | None,
        *,
        verbose: bool = False,
        quiet: bool = False,
        output_dir: Path | str | None = None,
        run_id: str | None = None,
    ) -> Path | None:
        """Configure logging from a `logging:` config section."""
        if not logging_config or not logging_config.get("enabled", True):
            return None

        level = logging_config.get("level", "INFO")

        console_cfg: Mapping[str, Any] = logging_config.get("console", {}) or {}
        file_cfg: Mapping[str, Any] = logging_config.get("file", {}) or {}

        console_enabled = console_cfg.get("enabled", True)
        console_level: str | int | None = (
            console_cfg.get("level", level) if console_enabled else False
        )
        if console_enabled:
            if quiet:
                console_level = "ERROR"
            elif verbose:
                console_level = "DEBUG"

        file_path: Path | None = None
        if file_cfg.get("enabled", False):
            if "path" in file_cfg:
                base_dir = Path(file_cfg.get("path", "./logs/"))
                if not base_dir.is_absolute():
                    base_dir = base_dir.resolve()
            else:
                base_dir = Path(output_dir) / "logs" if output_dir else Path("./logs")
                base_dir = base_dir.resolve()
            filename = file_cfg.get("filename", "homodyne_analysis.log")
            run_suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            if "{run_id}" in filename:
                filename = filename.format(run_id=run_suffix)
            else:
                stem = Path(filename).stem or "homodyne_analysis"
                suffix = Path(filename).suffix or ".log"
                filename = f"{stem}_{run_suffix}{suffix}"
            file_path = base_dir / filename

        return self.configure(
            level=level,
            console_level=console_level,
            console_format=console_cfg.get("format", "detailed"),
            console_colors=bool(console_cfg.get("colors", False)),
            file_path=file_path,
            file_level=file_cfg.get("level", "DEBUG"),
            max_size_mb=int(file_cfg.get("max_size_mb", 10)),
            backup_count=int(file_cfg.get("backup_count", 5)),
            module_levels=logging_config.get("modules"),
            force=True,
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger with hierarchical naming."""
        if not name.startswith(self._root_logger_name):
            if name == "__main__":
                full_name = f"{self._root_logger_name}.main"
            elif "." in name and name.startswith("homodyne"):
                full_name = name
            else:
                full_name = f"{self._root_logger_name}.{name}"
        else:
            full_name = name

        if not self._configured:
            self.configure()

        return logging.getLogger(full_name)


# Global logger manager instance
_logger_manager = MinimalLogger()


def configure_logging(
    logging_config: Mapping[str, Any] | None,
    *,
    verbose: bool = False,
    quiet: bool = False,
    output_dir: Path | str | None = None,
    run_id: str | None = None,
) -> Path | None:
    """Public helper to configure logging from config + CLI flags."""
    return _logger_manager.configure_from_dict(
        logging_config,
        verbose=verbose,
        quiet=quiet,
        output_dir=output_dir,
        run_id=run_id,
    )


def get_logger(
    name: str | None = None,
    *,
    context: Mapping[str, Any] | None = None,
) -> logging.Logger | logging.LoggerAdapter[logging.Logger]:
    """Get a logger instance with automatic naming and optional context."""
    if name is None:
        frame = inspect.currentframe()
        try:
            if frame is not None and frame.f_back is not None:
                name = frame.f_back.f_globals.get("__name__", "unknown")
        finally:
            del frame

    base_logger = _logger_manager.get_logger(name or "unknown")
    if context:
        return _ContextAdapter(base_logger, dict(context))
    return base_logger


def with_context(
    logger: logging.Logger, **context: Any
) -> logging.LoggerAdapter[logging.Logger]:
    """Attach contextual prefix to an existing logger."""
    return _ContextAdapter(logger, {k: v for k, v in context.items() if v is not None})


def log_calls(
    logger: LoggerType | None = None,
    level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function calls.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        include_args: Whether to log function arguments.
        include_result: Whether to log function return value.
    """
    resolved_logger: LoggerType | None = logger

    def decorator(func: F) -> F:
        nonlocal resolved_logger
        if resolved_logger is None:
            resolved_logger = get_logger(func.__module__)  # type: ignore[attr-defined]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = f"{func.__module__}.{func.__qualname__}"  # type: ignore[attr-defined]
            assert resolved_logger is not None  # For type narrowing

            # Log function entry
            if include_args:
                args_str = ", ".join([repr(arg) for arg in args])
                kwargs_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                resolved_logger.log(level, f"Calling {func_name}({all_args})")
            else:
                resolved_logger.log(level, f"Calling {func_name}")

            try:
                result = func(*args, **kwargs)

                # Log function exit
                if include_result:
                    resolved_logger.log(level, f"Completed {func_name} -> {repr(result)}")
                else:
                    resolved_logger.log(level, f"Completed {func_name}")

                return result

            except Exception as e:
                resolved_logger.log(logging.ERROR, f"Exception in {func_name}: {e}")
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def log_performance(
    logger: LoggerType | None = None,
    level: int = logging.INFO,
    threshold: float = 0.1,
) -> Callable[[F], F]:
    """Decorator to log function performance.

    Args:
        logger: Logger to use. If None, creates one for the module.
        level: Logging level to use.
        threshold: Minimum duration (seconds) to log.
    """
    resolved_logger: LoggerType | None = logger

    def decorator(func: F) -> F:
        nonlocal resolved_logger
        if resolved_logger is None:
            resolved_logger = get_logger(func.__module__)  # type: ignore[attr-defined]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            func_name = f"{func.__module__}.{func.__qualname__}"  # type: ignore[attr-defined]
            assert resolved_logger is not None  # For type narrowing

            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                if duration >= threshold:
                    resolved_logger.log(
                        level,
                        f"Performance: {func_name} completed in {duration:.3f}s",
                    )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                resolved_logger.log(
                    logging.ERROR,
                    f"Performance: {func_name} failed after {duration:.3f}s: {e}",
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


@contextmanager
def log_operation(
    operation_name: str,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> Generator[logging.Logger | logging.LoggerAdapter[logging.Logger], None, None]:
    """Context manager for logging operations.

    Args:
        operation_name: Name of the operation.
        logger: Logger to use. If None, creates one for caller's module.
        level: Logging level to use.
    """
    resolved_logger = get_logger() if logger is None else logger

    resolved_logger.log(level, f"Starting operation: {operation_name}")
    start_time = time.perf_counter()

    try:
        yield resolved_logger
        duration = time.perf_counter() - start_time
        resolved_logger.log(
            level, f"Completed operation: {operation_name} in {duration:.3f}s"
        )
    except Exception as e:
        duration = time.perf_counter() - start_time
        resolved_logger.log(
            logging.ERROR,
            f"Failed operation: {operation_name} after {duration:.3f}s: {e}",
        )
        raise


# Configure default logging on import
_logger_manager.configure()
