#!/usr/bin/env python3
"""Configure XLA_FLAGS for Homodyne optimization workflows."""

import argparse
import os
import sys
from pathlib import Path

VALID_MODES = ["cmc", "cmc-hpc", "nlsq", "auto"]
CONFIG_FILE = Path.home() / ".homodyne_xla_mode"


def detect_optimal_devices():
    """Detect optimal XLA device count based on CPU cores."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or 4
    except ImportError:
        cores = os.cpu_count() or 4

    # Same logic as shell scripts
    if cores <= 7:
        return 2
    elif cores <= 15:
        return 4
    elif cores <= 35:
        return 6
    else:
        return 8


def set_mode(mode: str) -> bool:
    """Save XLA mode preference."""
    if mode not in VALID_MODES:
        print(f"Error: Invalid mode '{mode}'", file=sys.stderr)
        print(f"Valid modes: {', '.join(VALID_MODES)}", file=sys.stderr)
        return False

    try:
        CONFIG_FILE.write_text(mode + "\n")
    except OSError as e:
        print(f"Error: Cannot write XLA mode config to {CONFIG_FILE}: {e}", file=sys.stderr)
        return False
    print(f"✓ XLA mode set to: {mode}")

    # Show what this means
    if mode == "cmc":
        print("  → 4 CPU devices for parallel CMC chains")
    elif mode == "cmc-hpc":
        print("  → 8 CPU devices for HPC clusters (36+ cores)")
    elif mode == "nlsq":
        print("  → 1 CPU device (NLSQ doesn't need parallelism)")
    elif mode == "auto":
        devices = detect_optimal_devices()
        cores = os.cpu_count() or 4
        print(f"  → Auto-detect: {devices} devices (detected {cores} CPU cores)")

    # Check if activation scripts exist before suggesting them
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        activation_dir = Path(venv) / "etc" / "homodyne" / "activation"
        bash_script = activation_dir / "xla_config.bash"
        fish_script = activation_dir / "xla_config.fish"
        if bash_script.exists() or fish_script.exists():
            print("\nReload your shell or run:")
            if bash_script.exists():
                print(f"  source {bash_script}  # bash/zsh")
            if fish_script.exists():
                print(f"  source {fish_script}  # fish")
        else:
            print(
                "\nActivation scripts not found. Run 'homodyne-post-install -i' first."
            )
    else:
        print(
            "\nNo active virtualenv detected. Activate your venv, then run 'homodyne-post-install -i'."
        )
    return True


def show_config():
    """Display current XLA configuration."""
    # Read current mode
    if CONFIG_FILE.exists():
        mode = CONFIG_FILE.read_text().strip()
    else:
        mode = "cmc (default)"

    # Get current XLA_FLAGS
    xla_flags = os.environ.get("XLA_FLAGS", "Not set")

    print("Current XLA Configuration:")
    print(f"  Mode: {mode}")
    print(f"  XLA_FLAGS: {xla_flags}")
    print(f"  Config file: {CONFIG_FILE}")

    # Show JAX devices if available
    try:
        import jax

        devices = jax.devices()
        print(f"  JAX devices: {len(devices)} ({devices[0].platform})")
        for i, dev in enumerate(devices):
            print(f"    [{i}] {dev}")
    except ImportError:
        print("  JAX: Not installed")
    except Exception as e:
        print(f"  JAX devices: Error - {e}")


def main():
    """Main entry point for homodyne-config-xla."""
    parser = argparse.ArgumentParser(
        description="Configure XLA_FLAGS for Homodyne workflows",
        epilog="""
Examples:
  homodyne-config-xla --mode cmc         # 4 devices for CMC
  homodyne-config-xla --mode auto        # Auto-detect based on CPU
  homodyne-config-xla --show             # Show current configuration

Modes:
  cmc        4 devices (multi-core workstations)
  cmc-hpc    8 devices (HPC with 36+ cores)
  nlsq       1 device (NLSQ-only workflows)
  auto       Auto-detect based on CPU cores
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        help="Set XLA mode (cmc, cmc-hpc, nlsq, auto)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )

    args = parser.parse_args()

    if args.show:
        show_config()
    elif args.mode:
        set_mode(args.mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
