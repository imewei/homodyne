"""
Enhanced Console Output and Logging System for Homodyne v2
=========================================================

Advanced logging system with color-coded output, progress indicators,
structured logging hierarchies, and user-friendly console experiences.

Key Features:
- Color-coded console output for different message types
- Progress bars and indicators for long-running operations
- Hierarchical logging with configurable verbosity levels
- Real-time configuration validation feedback
- Context-aware debug information for developers
- Performance monitoring with visual indicators
- Interactive error recovery prompts

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import colorama
    from colorama import Back, Fore, Style

    colorama.init()  # Initialize colorama for Windows support
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

    # Fallback color codes for systems without colorama
    class Fore:
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        CYAN = "\033[96m"
        WHITE = "\033[97m"
        RESET = "\033[0m"

    class Style:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        RESET_ALL = "\033[0m"

    class Back:
        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
        RESET = "\033[49m"


try:
    from homodyne.utils.logging import get_logger

    HAS_UTILS_LOGGING = True
except ImportError:
    import logging

    HAS_UTILS_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


@dataclass
class ConsoleTheme:
    """Color theme for console output."""

    # Message type colors
    error: str = Fore.RED
    warning: str = Fore.YELLOW
    info: str = Fore.BLUE
    success: str = Fore.GREEN
    debug: str = Fore.MAGENTA

    # Component colors
    header: str = Fore.CYAN + Style.BRIGHT
    section: str = Fore.BLUE + Style.BRIGHT
    parameter: str = Fore.YELLOW
    value: str = Fore.GREEN
    file_path: str = Fore.CYAN

    # Progress indicators
    progress_bar: str = Fore.GREEN
    progress_percent: str = Fore.CYAN
    spinner: str = Fore.YELLOW

    # Special formatting
    emphasis: str = Style.BRIGHT
    dim: str = Style.DIM
    reset: str = Style.RESET_ALL


class ProgressIndicator:
    """Progress indicator with multiple display modes."""

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "",
        style: str = "bar",
        show_eta: bool = True,
    ):
        self.total = total
        self.current = 0
        self.description = description
        self.style = style  # "bar", "spinner", "dots", "percentage"
        self.show_eta = show_eta
        self.start_time = time.time()
        self.last_update = 0
        self.theme = ConsoleTheme()

        # Spinner characters
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0

        # Dots for indeterminate progress
        self.dots_count = 0
        self.dots_max = 3

        # Check if stdout supports progress updates
        self.supports_updates = (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and os.getenv("TERM") != "dumb"
        )

    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress indicator."""
        self.current += increment
        if message:
            self.description = message

        current_time = time.time()
        # Throttle updates to avoid excessive output
        if current_time - self.last_update < 0.1:  # Update at most every 100ms
            return

        self.last_update = current_time

        if self.supports_updates:
            self._render_progress()
        else:
            # Fallback for non-interactive terminals
            if self.total:
                percent = (self.current / self.total) * 100
                print(
                    f"{self.description}: {percent:.1f}% ({self.current}/{self.total})"
                )

    def _render_progress(self):
        """Render progress indicator to console."""
        if not self.supports_updates:
            return

        # Clear current line
        print("\r", end="")

        if self.style == "bar" and self.total:
            self._render_progress_bar()
        elif self.style == "spinner":
            self._render_spinner()
        elif self.style == "dots":
            self._render_dots()
        elif self.style == "percentage" and self.total:
            self._render_percentage()
        else:
            # Fallback to simple counter
            print(f"{self.description}: {self.current}", end="")

        sys.stdout.flush()

    def _render_progress_bar(self):
        """Render progress bar."""
        if not self.total:
            return

        percent = min(100, (self.current / self.total) * 100)
        bar_width = 40
        filled_width = int(bar_width * percent / 100)

        bar = "█" * filled_width + "░" * (bar_width - filled_width)

        # Calculate ETA
        eta_str = ""
        if self.show_eta and self.current > 0:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f" ETA: {self._format_time(remaining)}"

        # Color the progress bar
        colored_bar = f"{self.theme.progress_bar}{bar}{self.theme.reset}"
        percent_str = f"{self.theme.progress_percent}{percent:5.1f}%{self.theme.reset}"

        print(
            f"{self.description}: {colored_bar} {percent_str} ({self.current}/{self.total}){eta_str}",
            end="",
        )

    def _render_spinner(self):
        """Render spinner animation."""
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        spinner = self.spinner_chars[self.spinner_index]

        elapsed_str = self._format_time(time.time() - self.start_time)
        print(
            f"{self.theme.spinner}{spinner}{self.theme.reset} {self.description} ({elapsed_str})",
            end="",
        )

    def _render_dots(self):
        """Render dots animation."""
        self.dots_count = (self.dots_count + 1) % (self.dots_max + 1)
        dots = "." * self.dots_count + " " * (self.dots_max - self.dots_count)

        elapsed_str = self._format_time(time.time() - self.start_time)
        print(f"{self.description}{dots} ({elapsed_str})", end="")

    def _render_percentage(self):
        """Render percentage only."""
        percent = (self.current / self.total) * 100
        print(
            f"{self.description}: {self.theme.progress_percent}{percent:.1f}%{self.theme.reset} ({self.current}/{self.total})",
            end="",
        )

    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def finish(self, message: Optional[str] = None):
        """Finish progress indicator."""
        if self.supports_updates:
            print("\r", end="")  # Clear current line

        final_message = message or f"{self.description}: Complete"
        elapsed = self._format_time(time.time() - self.start_time)

        print(f"{self.theme.success}✓{self.theme.reset} {final_message} ({elapsed})")


class EnhancedConsoleLogger:
    """Enhanced console logger with colors and structured output."""

    def __init__(
        self,
        theme: Optional[ConsoleTheme] = None,
        show_timestamps: bool = True,
        show_levels: bool = True,
        indent_level: int = 0,
    ):
        self.theme = theme or ConsoleTheme()
        self.show_timestamps = show_timestamps
        self.show_levels = show_levels
        self.indent_level = indent_level
        self.indent_str = "  " * indent_level

        # Check if terminal supports colors
        self.use_colors = (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and os.getenv("TERM") != "dumb"
            and os.getenv("NO_COLOR") != "1"
        )

        # Message type mapping
        self.level_colors = {
            "ERROR": self.theme.error,
            "WARNING": self.theme.warning,
            "INFO": self.theme.info,
            "DEBUG": self.theme.debug,
            "SUCCESS": self.theme.success,
        }

        # Unicode icons for message types
        self.level_icons = {
            "ERROR": "❌",
            "WARNING": "⚠️ ",
            "INFO": "ℹ️ ",
            "DEBUG": "🔍",
            "SUCCESS": "✅",
        }

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color}{text}{self.theme.reset}"
        return text

    def _format_timestamp(self) -> str:
        """Format timestamp for log messages."""
        if not self.show_timestamps:
            return ""

        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{self._colorize(timestamp, self.theme.dim)}] "

    def _format_level(self, level: str) -> str:
        """Format log level with colors and icons."""
        if not self.show_levels:
            return ""

        icon = self.level_icons.get(level, "")
        color = self.level_colors.get(level, "")

        colored_level = self._colorize(f"{level:<7}", color)
        return f"{icon}{colored_level} "

    def _print_message(self, level: str, message: str, **kwargs):
        """Print formatted log message."""
        timestamp = self._format_timestamp()
        level_str = self._format_level(level)
        indent = self.indent_str

        # Handle multiline messages
        lines = str(message).split("\n")
        for i, line in enumerate(lines):
            if i == 0:
                print(f"{timestamp}{level_str}{indent}{line}")
            else:
                # Subsequent lines get extra indentation
                extra_indent = " " * (len(level_str) - 3) if self.show_levels else ""
                print(f"{timestamp}{extra_indent}{indent}  {line}")

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._print_message("ERROR", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._print_message("WARNING", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._print_message("INFO", message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._print_message("DEBUG", message, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success message."""
        self._print_message("SUCCESS", message, **kwargs)

    def header(self, text: str, width: int = 60, char: str = "="):
        """Print formatted header."""
        print()
        print(self._colorize(char * width, self.theme.header))
        centered = text.center(width)
        print(self._colorize(centered, self.theme.header))
        print(self._colorize(char * width, self.theme.header))
        print()

    def section(self, text: str, width: int = 50, char: str = "-"):
        """Print formatted section header."""
        print()
        colored_text = self._colorize(f"📋 {text}", self.theme.section)
        print(colored_text)
        print(self._colorize(char * (len(text) + 3), self.theme.section))

    def parameter(self, name: str, value: Any, description: Optional[str] = None):
        """Print parameter with formatted name and value."""
        colored_name = self._colorize(name, self.theme.parameter)
        colored_value = self._colorize(str(value), self.theme.value)

        line = f"{self.indent_str}• {colored_name}: {colored_value}"

        if description:
            line += f" ({description})"

        print(line)

    def file_path(self, path: Union[str, Path], description: Optional[str] = None):
        """Print file path with special formatting."""
        colored_path = self._colorize(str(path), self.theme.file_path)

        if description:
            print(f"{self.indent_str}📁 {description}: {colored_path}")
        else:
            print(f"{self.indent_str}📁 {colored_path}")

    def validation_result(self, is_valid: bool, message: str):
        """Print validation result with appropriate styling."""
        if is_valid:
            icon = "✅"
            color = self.theme.success
        else:
            icon = "❌"
            color = self.theme.error

        colored_message = self._colorize(message, color)
        print(f"{self.indent_str}{icon} {colored_message}")

    def progress_update(
        self, step: str, current: int, total: int, details: Optional[str] = None
    ):
        """Print progress update in a consistent format."""
        percent = (current / total * 100) if total > 0 else 0
        progress_str = self._colorize(
            f"[{current}/{total}]", self.theme.progress_percent
        )
        step_str = self._colorize(step, self.theme.info)

        line = f"{progress_str} {step_str}"
        if details:
            line += f" - {details}"

        print(line)

    def create_child_logger(
        self, additional_indent: int = 1
    ) -> "EnhancedConsoleLogger":
        """Create child logger with increased indentation."""
        return EnhancedConsoleLogger(
            theme=self.theme,
            show_timestamps=self.show_timestamps,
            show_levels=self.show_levels,
            indent_level=self.indent_level + additional_indent,
        )


class ConfigurationProgressTracker:
    """Tracks progress through configuration operations with detailed feedback."""

    def __init__(self, logger: Optional[EnhancedConsoleLogger] = None):
        self.logger = logger or EnhancedConsoleLogger()
        self.operations = []
        self.current_operation = 0
        self.start_time = time.time()

        # Operation categories with estimated durations
        self.operation_estimates = {
            "load_config": 2,  # seconds
            "validate_structure": 3,
            "validate_parameters": 5,
            "validate_physics": 4,
            "optimize_settings": 3,
            "save_config": 1,
            "create_backup": 2,
            "recovery_attempt": 10,
            "template_generation": 3,
        }

    def set_operations(self, operations: List[str]):
        """Set the list of operations to track."""
        self.operations = operations
        self.current_operation = 0

        # Calculate total estimated time
        total_estimate = sum(self.operation_estimates.get(op, 5) for op in operations)

        self.logger.header("Configuration Processing")
        self.logger.info(
            f"Starting {len(operations)} operations (estimated: {total_estimate}s)"
        )

        for i, op in enumerate(operations, 1):
            estimate = self.operation_estimates.get(op, 5)
            self.logger.info(f"  {i}. {op.replace('_', ' ').title()} (~{estimate}s)")

        print()

    @contextmanager
    def operation(self, operation_name: str, details: Optional[str] = None):
        """Context manager for tracking individual operations."""
        if operation_name not in self.operations:
            self.operations.append(operation_name)

        op_index = self.operations.index(operation_name)
        self.current_operation = op_index

        # Start operation
        start_time = time.time()
        progress_str = f"[{op_index + 1}/{len(self.operations)}]"

        self.logger.info(
            f"{progress_str} Starting: {operation_name.replace('_', ' ').title()}"
        )
        if details:
            child_logger = self.logger.create_child_logger()
            child_logger.info(details)

        # Create progress indicator for longer operations
        estimated_time = self.operation_estimates.get(operation_name, 5)
        show_progress = estimated_time > 3

        progress = None
        if show_progress:
            progress = ProgressIndicator(
                description=f"  {operation_name.replace('_', ' ').title()}",
                style="spinner",
                show_eta=False,
            )

        try:
            yield progress

            # Success
            elapsed = time.time() - start_time
            self.logger.success(
                f"{progress_str} Completed: {operation_name.replace('_', ' ').title()} ({elapsed:.1f}s)"
            )

        except Exception as e:
            # Failure
            elapsed = time.time() - start_time
            self.logger.error(
                f"{progress_str} Failed: {operation_name.replace('_', ' ').title()} - {str(e)} ({elapsed:.1f}s)"
            )
            raise

        finally:
            if progress:
                progress.finish()

    def summary(self, success: bool = True, errors: Optional[List[str]] = None):
        """Print operation summary."""
        total_time = time.time() - self.start_time

        print()
        if success:
            self.logger.header("✅ Configuration Processing Complete", char="=")
            self.logger.success(
                f"All {len(self.operations)} operations completed successfully"
            )
        else:
            self.logger.header("❌ Configuration Processing Failed", char="=")
            self.logger.error(
                f"Processing failed after {self.current_operation}/{len(self.operations)} operations"
            )

            if errors:
                self.logger.error("Errors encountered:")
                child_logger = self.logger.create_child_logger()
                for error in errors:
                    child_logger.error(error)

        self.logger.info(f"Total time: {total_time:.1f} seconds")

        # Performance analysis
        if success and total_time > 30:
            self.logger.warning("Processing took longer than expected")
            self.logger.info(
                "Consider using performance optimizations or simpler configuration"
            )
        elif success and total_time < 5:
            self.logger.success(
                "Very fast processing - configuration is well optimized!"
            )


class ValidationFeedbackSystem:
    """Provides real-time feedback during configuration validation."""

    def __init__(self, logger: Optional[EnhancedConsoleLogger] = None):
        self.logger = logger or EnhancedConsoleLogger()
        self.errors = []
        self.warnings = []
        self.suggestions = []
        self.performance_notes = []

    def start_validation(self, config_name: str):
        """Start validation process."""
        self.logger.section("Configuration Validation")
        self.logger.info(f"Validating configuration: {config_name}")
        self.logger.file_path(config_name)
        print()

    def validate_section(
        self, section_name: str, is_valid: bool, details: Optional[str] = None
    ):
        """Report validation result for a configuration section."""
        self.logger.validation_result(is_valid, f"Section '{section_name}'")

        if details:
            child_logger = self.logger.create_child_logger()
            if is_valid:
                child_logger.info(details)
            else:
                child_logger.error(details)

    def add_error(self, message: str, suggestion: Optional[str] = None):
        """Add validation error with optional suggestion."""
        self.errors.append(message)
        self.logger.error(message)

        if suggestion:
            child_logger = self.logger.create_child_logger()
            child_logger.info(f"💡 Suggestion: {suggestion}")
            self.suggestions.append(suggestion)

    def add_warning(self, message: str, suggestion: Optional[str] = None):
        """Add validation warning with optional suggestion."""
        self.warnings.append(message)
        self.logger.warning(message)

        if suggestion:
            child_logger = self.logger.create_child_logger()
            child_logger.info(f"💡 Suggestion: {suggestion}")
            self.suggestions.append(suggestion)

    def add_performance_note(self, message: str):
        """Add performance-related note."""
        self.performance_notes.append(message)
        self.logger.info(f"🚀 Performance: {message}")

    def show_parameter_validation(
        self,
        param_name: str,
        value: Any,
        is_valid: bool,
        expected_range: Optional[tuple] = None,
    ):
        """Show parameter validation with detailed feedback."""
        status_icon = "✅" if is_valid else "❌"
        colored_name = self.logger._colorize(param_name, self.logger.theme.parameter)
        colored_value = self.logger._colorize(str(value), self.logger.theme.value)

        line = f"  {status_icon} {colored_name}: {colored_value}"

        if expected_range and not is_valid:
            min_val, max_val = expected_range
            line += f" (expected: {min_val} - {max_val})"

        print(line)

    def finish_validation(self) -> Dict[str, Any]:
        """Finish validation and show summary."""
        total_issues = len(self.errors) + len(self.warnings)
        is_valid = len(self.errors) == 0

        print()
        self.logger.section("Validation Summary")

        if is_valid:
            if total_issues == 0:
                self.logger.success("Configuration is perfect! ✨")
                score = 100
            else:
                self.logger.success(
                    f"Configuration is valid with {len(self.warnings)} warnings"
                )
                score = max(70, 100 - len(self.warnings) * 10)
        else:
            self.logger.error(
                f"Configuration is invalid - {len(self.errors)} errors found"
            )
            score = max(0, 50 - len(self.errors) * 10)

        # Show score with color coding
        if score >= 90:
            score_color = self.logger.theme.success
        elif score >= 70:
            score_color = self.logger.theme.warning
        else:
            score_color = self.logger.theme.error

        score_str = self.logger._colorize(f"{score}/100", score_color)
        self.logger.info(f"Configuration quality score: {score_str}")

        # Show suggestions if any
        if self.suggestions:
            print()
            self.logger.info(f"💡 {len(self.suggestions)} improvement suggestions:")
            child_logger = self.logger.create_child_logger()
            for suggestion in self.suggestions:
                child_logger.info(f"• {suggestion}")

        # Show performance notes if any
        if self.performance_notes:
            print()
            self.logger.info(f"🚀 {len(self.performance_notes)} performance notes:")
            child_logger = self.logger.create_child_logger()
            for note in self.performance_notes:
                child_logger.info(f"• {note}")

        return {
            "is_valid": is_valid,
            "score": score,
            "errors": self.errors,
            "warnings": self.warnings,
            "suggestions": self.suggestions,
            "performance_notes": self.performance_notes,
        }


class InteractivePromptSystem:
    """Enhanced interactive prompts with better UX."""

    def __init__(self, logger: Optional[EnhancedConsoleLogger] = None):
        self.logger = logger or EnhancedConsoleLogger()

    def ask_choice(
        self,
        prompt: str,
        choices: List[str],
        default: Optional[str] = None,
        descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Ask user to choose from options with enhanced formatting."""
        print()
        self.logger.info(prompt)

        for i, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            desc = descriptions.get(choice, "") if descriptions else ""

            colored_choice = self.logger._colorize(choice, self.logger.theme.value)
            line = f"  {i}. {colored_choice}{marker}"

            if desc:
                line += f" - {desc}"

            print(line)

        while True:
            try:
                response = input(
                    f"\n{self.logger._colorize('Choice', self.logger.theme.parameter)} [1-{len(choices)}]: "
                ).strip()

                if not response and default:
                    return default

                choice_num = int(response)
                if 1 <= choice_num <= len(choices):
                    selected = choices[choice_num - 1]
                    self.logger.success(f"Selected: {selected}")
                    return selected
                else:
                    self.logger.error(
                        f"Please enter a number between 1 and {len(choices)}"
                    )

            except ValueError:
                self.logger.error("Please enter a valid number")
            except KeyboardInterrupt:
                print()
                self.logger.warning("Operation cancelled by user")
                raise

    def ask_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Ask yes/no question with clear formatting."""
        default_str = "Y/n" if default else "y/N"
        colored_prompt = self.logger._colorize(prompt, self.logger.theme.info)

        while True:
            try:
                response = input(f"{colored_prompt} [{default_str}]: ").strip().lower()

                if not response:
                    return default

                if response in ["y", "yes", "true", "1"]:
                    return True
                elif response in ["n", "no", "false", "0"]:
                    return False
                else:
                    self.logger.error("Please enter 'y' for yes or 'n' for no")

            except KeyboardInterrupt:
                print()
                self.logger.warning("Operation cancelled by user")
                raise

    def ask_value(
        self,
        prompt: str,
        value_type: type = str,
        default: Any = None,
        validator: Optional[Callable] = None,
        description: Optional[str] = None,
    ) -> Any:
        """Ask for a value with type checking and validation."""
        default_str = f" (default: {default})" if default is not None else ""
        type_hint = f" [{value_type.__name__}]" if value_type != str else ""

        colored_prompt = self.logger._colorize(prompt, self.logger.theme.parameter)
        full_prompt = f"{colored_prompt}{type_hint}{default_str}"

        if description:
            self.logger.info(f"💬 {description}")

        while True:
            try:
                response = input(f"{full_prompt}: ").strip()

                if not response and default is not None:
                    return default

                # Type conversion
                if value_type == bool:
                    value = response.lower() in ["true", "yes", "y", "1"]
                elif value_type == int:
                    value = int(response)
                elif value_type == float:
                    value = float(response)
                else:
                    value = response

                # Validation
                if validator and not validator(value):
                    self.logger.error("Invalid value. Please try again.")
                    continue

                # Show accepted value
                colored_value = self.logger._colorize(
                    str(value), self.logger.theme.success
                )
                print(f"  → {colored_value}")

                return value

            except ValueError:
                self.logger.error(f"Please enter a valid {value_type.__name__}")
            except KeyboardInterrupt:
                print()
                self.logger.warning("Operation cancelled by user")
                raise

    def confirm_action(
        self, action: str, details: Optional[str] = None, default: bool = False
    ) -> bool:
        """Ask for confirmation before performing an action."""
        print()
        self.logger.warning(f"⚠️  About to: {action}")

        if details:
            child_logger = self.logger.create_child_logger()
            child_logger.info(details)

        return self.ask_yes_no("Are you sure?", default=default)


# Factory functions for creating enhanced console components
def create_console_logger(
    theme: Optional[ConsoleTheme] = None, **kwargs
) -> EnhancedConsoleLogger:
    """Create an enhanced console logger."""
    return EnhancedConsoleLogger(theme=theme, **kwargs)


def create_progress_tracker(
    logger: Optional[EnhancedConsoleLogger] = None,
) -> ConfigurationProgressTracker:
    """Create a configuration progress tracker."""
    return ConfigurationProgressTracker(logger=logger)


def create_validation_feedback() -> ValidationFeedbackSystem:
    """Create a validation feedback system."""
    return ValidationFeedbackSystem()


def create_interactive_prompts() -> InteractivePromptSystem:
    """Create an interactive prompt system."""
    return InteractivePromptSystem()


# Context managers for enhanced output
@contextmanager
def console_operation(name: str, logger: Optional[EnhancedConsoleLogger] = None):
    """Context manager for console operations with timing."""
    if logger is None:
        logger = EnhancedConsoleLogger()

    start_time = time.time()
    logger.info(f"Starting: {name}")

    try:
        yield logger
        elapsed = time.time() - start_time
        logger.success(f"Completed: {name} ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Failed: {name} - {str(e)} ({elapsed:.1f}s)")
        raise


@contextmanager
def progress_context(
    total: Optional[int] = None, description: str = "", style: str = "bar"
) -> ProgressIndicator:
    """Context manager for progress indicators."""
    progress = ProgressIndicator(total=total, description=description, style=style)

    try:
        yield progress
        progress.finish()
    except Exception as e:
        progress.finish(f"Failed: {str(e)}")
        raise
