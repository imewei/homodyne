"""Minimal CLI Entry Point for Homodyne v2
=======================================

Simplified command-line interface for homodyne scattering analysis with
JAX-first optimization methods.

Entry point for console script: homodyne [args]
"""

import os
import sys

# Disable JAX GPU autotuning BEFORE any JAX imports
# This prevents hanging issues with gemm_fusion_autotuner
os.environ.setdefault(
    "XLA_FLAGS",
    (" --xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"),
)

from homodyne.cli.args_parser import create_parser
from homodyne.cli.commands import dispatch_command
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def check_python_version() -> None:
    """Check Python version requirement."""


def check_gpu_availability() -> None:
    """Check if GPU is available but not being used by JAX.

    Prints a helpful warning if:
    - NVIDIA GPU hardware is detected (nvidia-smi works)
    - But JAX is running in CPU-only mode

    This helps users realize they can enable GPU acceleration for 20-100x speedup.
    """
    try:
        import subprocess

        # Check if nvidia-smi detects GPU hardware
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            # No GPU hardware detected, no warning needed
            return

        gpu_names = result.stdout.strip().split("\n")
        if not gpu_names or not gpu_names[0]:
            return

        # GPU hardware detected, now check if JAX is using it
        try:
            import jax
            devices = jax.devices()

            if devices and devices[0].platform == "gpu":
                # GPU is being used, all good
                return

            # GPU hardware present but JAX using CPU - print warning
            logger.warning(
                "⚠️  GPU Acceleration Not Enabled\n"
                f"   Detected: {gpu_names[0]}\n"
                "   JAX is using CPU-only mode\n"
                "\n"
                "   For 20-100x speedup on large datasets:\n"
                "   1. cd /path/to/homodyne/repo\n"
                "   2. make install-jax-gpu\n"
                "\n"
                "   Requires: Linux + CUDA 12.1-12.9 installed\n"
                "   See: https://github.com/yourusername/homodyne#gpu-acceleration"
            )

        except ImportError:
            # JAX not available, skip check
            pass

    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        # nvidia-smi not found or failed, no GPU present
        pass
    except Exception:
        # Any other error, silently continue (don't disrupt startup)
        pass


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
