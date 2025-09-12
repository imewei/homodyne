"""
Main CLI Entry Point for Homodyne v2
====================================

Command-line interface for homodyne scattering analysis with JAX-accelerated
optimization methods with enhanced performance.

Entry point for console script: homodyne [args]
"""

import sys
import logging
from pathlib import Path
from typing import Optional

from homodyne.utils.logging import get_logger
from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.cli.validators import validate_args


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
    if args.verbose:
        log_level = logging.DEBUG
    elif args.quiet:
        log_level = logging.ERROR
    else:
        log_level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


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
        
        # Print banner unless quiet mode
        if not args.quiet:
            print_banner()
        
        logger.info("Starting Homodyne v2 analysis")
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
        if '--verbose' in sys.argv or '--debug' in sys.argv:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())