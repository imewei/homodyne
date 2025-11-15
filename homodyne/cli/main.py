"""Minimal CLI Entry Point for Homodyne v2
=======================================

Simplified command-line interface for homodyne scattering analysis with
JAX-first optimization methods.

Entry point for console script: homodyne [args]
"""

import os
import sys

# ============================================================================
# JAX CPU Device Configuration (MUST be set before JAX import)
# ============================================================================
# Configure JAX to use multiple CPU devices for parallel MCMC chains
# This MUST be set before JAX/XLA is initialized (import time)
# Default: 4 devices for parallel MCMC, can be overridden by user
if "XLA_FLAGS" not in os.environ:
    # No existing XLA_FLAGS, set default
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
elif "xla_force_host_platform_device_count" not in os.environ["XLA_FLAGS"]:
    # XLA_FLAGS exists but doesn't specify device count, append it
    os.environ["XLA_FLAGS"] += " --xla_force_host_platform_device_count=4"
# else: User has already configured device count, respect their setting

# Suppress NLSQ GPU warnings (v2.3.0 is CPU-only)
os.environ.setdefault("NLSQ_SKIP_GPU_CHECK", "1")

# Suppress JAX backend logs (set to ERROR to hide GPU fallback warnings)
# This must be done before any imports that trigger JAX initialization
import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
logging.getLogger("jax._src.compiler").setLevel(logging.ERROR)

# Note: GPU support removed in v2.3.0 (CPU-only)

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def check_python_version() -> None:
    """Check Python version requirement."""


def main() -> None:
    """Main CLI entry point.

    Processes command-line arguments and dispatches to appropriate command handler.
    """
    try:
        # Check Python version
        check_python_version()

        # Parse arguments
        parser = create_parser()
        args = parser.parse_args()

        # Configure logging level
        if hasattr(args, "verbose") and args.verbose:
            import logging

            logging.getLogger("homodyne").setLevel(logging.DEBUG)

        # Log startup
        logger.info("Starting homodyne analysis...")
        logger.debug(f"Arguments: {vars(args)}")

        # Dispatch command (device configuration happens inside dispatch_command)
        # Note: GPU status is checked and logged during device configuration,
        # not here at startup, to avoid premature/inaccurate warnings
        result = dispatch_command(args)

        # Handle result
        if result and result.get("success", False):
            logger.info("Analysis completed successfully")
            sys.exit(0)
        else:
            error_msg = (
                result.get("error", "Unknown error") if result else "Command failed"
            )
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
