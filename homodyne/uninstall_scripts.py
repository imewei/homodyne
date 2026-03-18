"""Cleanup utilities for homodyne package.

This module provides utilities to remove:
- Shell completion files
- XLA configuration files
- Activation script modifications

CLI Entry Point: homodyne-cleanup
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple


class CleanupTarget(NamedTuple):
    """A file or directory to clean up."""

    path: Path
    description: str
    exists: bool


def get_venv_path() -> Path | None:
    """Get the virtual environment path if in one.

    Returns:
        Path to venv or None if not in a virtual environment.
    """
    if sys.prefix != sys.base_prefix:
        return Path(sys.prefix)

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        return Path(venv)

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)

    return None


def find_cleanup_targets() -> list[CleanupTarget]:
    """Find all homodyne-related files that can be cleaned up.

    Returns:
        List of CleanupTarget objects.
    """
    targets: list[CleanupTarget] = []

    # XLA mode configuration -- check all possible locations
    home = Path.home()

    # New location: per-env or XDG
    from homodyne.post_install import get_xla_mode_path

    xla_mode_file = get_xla_mode_path()
    targets.append(
        CleanupTarget(
            path=xla_mode_file,
            description="XLA mode configuration",
            exists=xla_mode_file.exists(),
        )
    )

    # Legacy location
    legacy_xla = home / ".homodyne_xla_mode"
    if legacy_xla.exists():
        targets.append(
            CleanupTarget(
                path=legacy_xla,
                description="XLA mode configuration (legacy)",
                exists=True,
            )
        )

    # Virtual environment files
    venv_path = get_venv_path()
    if venv_path:
        # Bash completion
        bash_completion = venv_path / "etc" / "bash_completion.d" / "homodyne"
        targets.append(
            CleanupTarget(
                path=bash_completion,
                description="Bash completion script",
                exists=bash_completion.exists(),
            )
        )

        # Zsh completion
        zsh_completion = venv_path / "etc" / "zsh" / "homodyne-completion.zsh"
        targets.append(
            CleanupTarget(
                path=zsh_completion,
                description="Zsh completion script",
                exists=zsh_completion.exists(),
            )
        )

        # Fish completion
        fish_completion = (
            venv_path / "share" / "fish" / "vendor_completions.d" / "homodyne.fish"
        )
        targets.append(
            CleanupTarget(
                path=fish_completion,
                description="Fish completion script",
                exists=fish_completion.exists(),
            )
        )

        # Check for empty parent directories
        for completion in [bash_completion, zsh_completion, fish_completion]:
            if completion.parent.exists() and not any(completion.parent.iterdir()):
                targets.append(
                    CleanupTarget(
                        path=completion.parent,
                        description=f"Empty directory: {completion.parent.name}",
                        exists=True,
                    )
                )

    # User-level completion directories
    local_bash = (
        home / ".local" / "share" / "bash-completion" / "completions" / "homodyne"
    )
    targets.append(
        CleanupTarget(
            path=local_bash,
            description="User bash completion",
            exists=local_bash.exists(),
        )
    )

    return targets


def cleanup_completion_files(
    dry_run: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Remove shell completion files.

    Args:
        dry_run: If True, don't actually delete files.
        verbose: Print detailed output.

    Returns:
        List of paths that were (or would be) removed.
    """
    removed: list[Path] = []
    targets = find_cleanup_targets()

    for target in targets:
        if not target.exists:
            continue

        if "completion" not in target.description.lower():
            continue

        if verbose:
            action = "Would remove" if dry_run else "Removing"
            print(f"{action}: {target.path} ({target.description})")

        if not dry_run:
            try:
                if target.path.is_dir():
                    target.path.rmdir()
                else:
                    target.path.unlink()
                removed.append(target.path)
            except OSError as e:
                if verbose:
                    print(f"  Failed: {e}")
        else:
            removed.append(target.path)

    return removed


def cleanup_xla_config(
    dry_run: bool = False,
    verbose: bool = False,
) -> list[Path]:
    """Remove XLA configuration files.

    Args:
        dry_run: If True, don't actually delete files.
        verbose: Print detailed output.

    Returns:
        List of paths that were (or would be) removed.
    """
    removed: list[Path] = []

    # XLA mode file -- remove from all locations
    from homodyne.post_install import get_xla_mode_path

    for xla_mode_file in [get_xla_mode_path(), Path.home() / ".homodyne_xla_mode"]:
        if xla_mode_file.exists():
            if verbose:
                action = "Would remove" if dry_run else "Removing"
                print(f"{action}: {xla_mode_file} (XLA mode configuration)")

            if not dry_run:
                try:
                    xla_mode_file.unlink()
                    removed.append(xla_mode_file)
                    # Clean up empty parent dir
                    parent = xla_mode_file.parent
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                except OSError as e:
                    if verbose:
                        print(f"  Failed: {e}")
            else:
                removed.append(xla_mode_file)

    return removed


def _remove_homodyne_blocks(content: str, end_marker: str) -> str:
    """Remove homodyne-injected blocks from activate script content.

    Uses a line-by-line state machine instead of regex to correctly handle
    nested if/fi or if/end structures within injected blocks.

    Args:
        content: Full text of the activate script.
        end_marker: Block terminator: "fi" for bash/zsh, "end" for fish.

    Returns:
        Content with all homodyne blocks removed.
    """
    _HOMODYNE_HEADERS = (
        "# Homodyne shell completion (auto-added by homodyne-post-install)",
        "# Homodyne XLA configuration (auto-added by homodyne-post-install)",
        "# homodyne XLA configuration",
    )

    lines = content.split("\n")
    result: list[str] = []
    skipping = False
    depth = 0

    for line in lines:
        stripped = line.strip()

        if not skipping:
            if stripped in _HOMODYNE_HEADERS:
                skipping = True
                depth = 0
                continue
            result.append(line)
        else:
            if end_marker == "fi":
                if stripped.startswith("if ") or stripped.startswith("if\t"):
                    depth += 1
                elif (
                    stripped == end_marker
                    or stripped.startswith("fi ")
                    or stripped.startswith("fi;")
                ):
                    depth -= 1
                    if depth <= 0:
                        skipping = False
                        continue
            else:
                if stripped in ("if", "for", "while") or any(
                    stripped.startswith(kw) for kw in ("if ", "for ", "while ")
                ):
                    depth += 1
                elif stripped == end_marker:
                    depth -= 1
                    if depth <= 0:
                        skipping = False
                        continue

    return "\n".join(result)


def cleanup_xla_activation_scripts(
    dry_run: bool = False,
    verbose: bool = False,
) -> bool:
    """Remove XLA configuration from venv activation scripts.

    Args:
        dry_run: If True, don't actually modify files.
        verbose: Print detailed output.

    Returns:
        True if any modifications were made.
    """
    modified = False
    venv_path = get_venv_path()

    if not venv_path:
        return False

    # (script_name, end_marker) pairs
    scripts = [
        ("activate", "fi"),
        ("activate.fish", "end"),
    ]

    for script_name, end_marker in scripts:
        script_path = venv_path / "bin" / script_name
        if not script_path.exists():
            continue

        content = script_path.read_text(encoding="utf-8")
        if "homodyne XLA" not in content and "Homodyne" not in content:
            continue

        if verbose:
            action = "Would modify" if dry_run else "Modifying"
            print(f"{action}: {script_path} (removing homodyne config)")

        if not dry_run:
            new_content = _remove_homodyne_blocks(content, end_marker)
            if new_content != content:
                try:
                    script_path.write_text(new_content, encoding="utf-8")
                    modified = True
                except OSError as e:
                    if verbose:
                        print(f"  Failed: {e}")
        else:
            modified = True

    return modified


def show_dry_run(verbose: bool = True) -> None:
    """Show what would be removed without actually removing anything.

    Args:
        verbose: Print detailed output.
    """
    print("Dry run - showing what would be removed:")
    print("-" * 50)

    targets = find_cleanup_targets()
    existing = [t for t in targets if t.exists]

    if not existing:
        print("No homodyne files found to clean up.")
        return

    for target in existing:
        print(f"  {target.path}")
        if verbose:
            print(f"    ({target.description})")

    # Check activation scripts
    venv_path = get_venv_path()
    if venv_path:
        activate = venv_path / "bin" / "activate"
        if activate.exists() and "homodyne XLA" in activate.read_text(encoding="utf-8"):
            print(f"  {activate} (would modify)")
            if verbose:
                print("    (remove XLA configuration block)")

        fish_activate = venv_path / "bin" / "activate.fish"
        if fish_activate.exists() and "homodyne XLA" in fish_activate.read_text(
            encoding="utf-8"
        ):
            print(f"  {fish_activate} (would modify)")
            if verbose:
                print("    (remove XLA configuration block)")


def interactive_cleanup() -> None:
    """Run interactive cleanup process."""
    print("=" * 60)
    print("Homodyne Cleanup")
    print("=" * 60)
    print()

    targets = find_cleanup_targets()
    existing = [t for t in targets if t.exists]

    if not existing:
        print("No homodyne files found to clean up.")
        return

    print("The following files were found:")
    for target in existing:
        print(f"  - {target.path}")
        print(f"    ({target.description})")
    print()

    # Check activation scripts
    venv_path = get_venv_path()
    has_activation_mods = False
    if venv_path:
        activate = venv_path / "bin" / "activate"
        if activate.exists() and "homodyne XLA" in activate.read_text(encoding="utf-8"):
            print(f"  - {activate} contains XLA configuration")
            has_activation_mods = True

    print()
    response = input("Remove all homodyne files? [y/N]: ").strip().lower()

    if response != "y":
        print("Cleanup cancelled.")
        return

    # Perform cleanup
    print("\nCleaning up...")

    removed = cleanup_completion_files(verbose=True)
    removed.extend(cleanup_xla_config(verbose=True))

    if has_activation_mods:
        cleanup_xla_activation_scripts(verbose=True)

    print()
    print(f"Removed {len(removed)} file(s).")
    print("Cleanup complete!")


def main() -> int:
    """CLI entry point for homodyne-cleanup."""
    parser = argparse.ArgumentParser(
        description="Clean up homodyne installation files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-cleanup                  # Interactive cleanup
  homodyne-cleanup --dry-run        # Show what would be removed
  homodyne-cleanup --force          # Remove without confirmation
""",
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be removed without removing",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Remove without confirmation",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive cleanup (default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.dry_run:
        show_dry_run(args.verbose)
        return 0

    if args.force:
        # Non-interactive removal
        removed = cleanup_completion_files(verbose=args.verbose)
        removed.extend(cleanup_xla_config(verbose=args.verbose))
        cleanup_xla_activation_scripts(verbose=args.verbose)

        if args.verbose:
            print(f"\nRemoved {len(removed)} file(s).")
        return 0

    # Default: interactive mode
    interactive_cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
