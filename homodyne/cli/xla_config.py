#!/usr/bin/env python3
"""Configure XLA_FLAGS for Homodyne optimization workflows."""

import argparse
import os
import sys
from pathlib import Path

VALID_MODES = ["cmc", "cmc-hpc", "nlsq", "auto"]


def _get_config_file() -> Path:
    """Get the XLA mode config file path (per-venv > XDG > legacy).

    Uses the same resolution logic as post_install.get_xla_mode_path().
    """
    # Prefer per-environment config
    venv = os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX")
    if venv:
        return Path(venv) / "etc" / "homodyne" / "xla_mode"

    # Fall back to XDG config directory
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "")
    if not xdg_config:
        xdg_config = str(Path.home() / ".config")
    return Path(xdg_config) / "homodyne" / "xla_mode"


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

    config_file = _get_config_file()
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(mode + "\n", encoding="utf-8")
    except OSError as e:
        print(
            f"Error: Cannot write XLA mode config to {config_file}: {e}",
            file=sys.stderr,
        )
        return False
    print(f"OK: XLA mode set to: {mode}")

    # Show what this means
    if mode == "cmc":
        print("  -> 4 CPU devices for parallel CMC chains")
    elif mode == "cmc-hpc":
        print("  -> 8 CPU devices for HPC clusters (36+ cores)")
    elif mode == "nlsq":
        print("  -> 1 CPU device (NLSQ doesn't need parallelism)")
    elif mode == "auto":
        devices = detect_optimal_devices()
        cores = os.cpu_count() or 4
        print(f"  -> Auto-detect: {devices} devices (detected {cores} CPU cores)")

    # Suggest reactivation to pick up the new mode
    print("\nDeactivate and reactivate your venv to apply the new mode.")
    return True


def show_config():
    """Display current XLA configuration."""
    config_file = _get_config_file()

    # Read current mode
    if config_file.exists():
        try:
            mode = config_file.read_text(encoding="utf-8").strip()
        except OSError:
            mode = "cmc (default, config unreadable)"
    else:
        mode = "cmc (default)"

    # Get current XLA_FLAGS
    xla_flags = os.environ.get("XLA_FLAGS", "Not set")

    print("Current XLA Configuration:")
    print(f"  Mode: {mode}")
    print(f"  XLA_FLAGS: {xla_flags}")
    print(f"  Config file: {config_file}")

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


def main() -> None:
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
