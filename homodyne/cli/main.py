"""
Main CLI Entry Point for Homodyne v2
====================================

Command-line interface for homodyne scattering analysis with JAX-accelerated
optimization methods with enhanced performance.

Entry point for console script: homodyne [args]
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.cli.validators import validate_args
from homodyne.utils.logging import get_logger


def check_python_version() -> None:
    """Check Python version requirement."""
    if sys.version_info < (3, 12):
        print(
            "Error: Python 3.12 or higher is required for homodyne-analysis.",
            file=sys.stderr,
        )
        print(
            f"Current Python version: {sys.version_info[0]}.{sys.version_info[1]}",
            file=sys.stderr,
        )
        sys.exit(1)


def setup_logging(args) -> None:
    """Configure logging based on CLI arguments."""
    # Map log_level string to logging constants
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    # Get log level from args (fallback to INFO if not specified)
    log_level = log_level_map.get(getattr(args, 'log_level', 'INFO'), logging.INFO)

    # Configure handlers list
    handlers = []

    # Console handler (unless --no-console)
    if not getattr(args, 'no_console', False):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)

    # File handler (if --log-file specified)
    if getattr(args, "log_file", False):
        # Create logs directory in output_dir
        log_dir = Path(args.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / "homodyne.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all messages
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(name)s.%(funcName)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger with handlers
    logging.basicConfig(level=log_level, handlers=handlers, force=True)


def print_banner() -> None:
    """Print Homodyne v2 banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     Homodyne v2 - JAX-Accelerated XPCS Analysis              ║
║                                                                               ║
║  High-performance X-ray Photon Correlation Spectroscopy analysis             ║
║  for nonequilibrium systems with 10-50x speedup over classical methods       ║
║                                                                               ║
║  Methods: VI (primary), MCMC (high-accuracy), Hybrid (VI→MCMC pipeline)      ║
║  Modes: Static Isotropic/Anisotropic (3 params), Laminar Flow (7 params)     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main() -> int:
    """
    Main CLI entry point for homodyne scattering analysis.

    Provides complete interface for XPCS analysis under nonequilibrium
    conditions with JAX-accelerated optimization methods.

    Returns:
        Exit code: 0 for success, 1 for error, 2 for invalid arguments
    """
    try:
        # Check Python version requirement
        check_python_version()

        # Create argument parser and parse arguments
        parser = create_parser()
        args = parser.parse_args()

        # Setup logging based on arguments
        setup_logging(args)
        logger = get_logger(__name__)

        # Print banner unless console output is disabled
        if not getattr(args, 'no_console', False):
            print_banner()

        logger.info("Starting Homodyne v2 analysis")

        # Log file logging status
        if getattr(args, "log_file", False):
            log_file_path = Path(args.output_dir) / "logs" / "homodyne.log"
            logger.info(f"📝 File logging enabled: {log_file_path}")

        logger.debug(f"Command line arguments: {vars(args)}")

        # Validate arguments
        try:
            validate_args(args)
            logger.debug("✓ Arguments validated successfully")
        except ValueError as e:
            logger.error(f"❌ Invalid arguments: {e}")
            return 2

        # Dispatch to appropriate command handler
        exit_code = dispatch_command(args)

        if exit_code == 0:
            logger.info("✓ Homodyne analysis completed successfully")
        else:
            logger.error(f"❌ Analysis failed with exit code {exit_code}")

        return exit_code

    except KeyboardInterrupt:
        print("\n\n❌ Analysis interrupted by user", file=sys.stderr)
        return 1
    except SystemExit as e:
        return e.code if e.code is not None else 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        if "--verbose" in sys.argv or "--debug" in sys.argv:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
