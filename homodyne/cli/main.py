"""
Minimal CLI Entry Point for Homodyne v2
=======================================

Simplified command-line interface for homodyne scattering analysis with
JAX-first optimization methods.

Entry point for console script: homodyne [args]
"""

import sys
from pathlib import Path
from typing import Optional

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def check_python_version() -> None:
    """Check Python version requirement."""
    if sys.version_info < (3, 10):
        print(
            "Error: Python 3.10 or higher is required for homodyne.",
            file=sys.stderr,
        )
        print(
            f"Current Python version: {sys.version_info[0]}.{sys.version_info[1]}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    """
    Main CLI entry point.

    Processes command-line arguments and dispatches to appropriate command handler.
    """
    try:
        # Check Python version
        check_python_version()

        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()

        # Configure logging level
        if hasattr(args, 'verbose') and args.verbose:
            import logging
            logging.getLogger('homodyne').setLevel(logging.DEBUG)

        # Log startup
        logger.info("Starting homodyne analysis...")
        logger.debug(f"Arguments: {vars(args)}")

        # Dispatch command
        result = dispatch_command(args)

        # Handle result
        if result and result.get('success', False):
            logger.info("Analysis completed successfully")
            sys.exit(0)
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'Command failed'
            logger.error(f"Analysis failed: {error_msg}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()