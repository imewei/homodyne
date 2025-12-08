#!/usr/bin/env python3
"""Homodyne Cleanup - Remove Shell Completion System
====================================================

This script removes homodyne-related files from virtual environments
that were installed by homodyne-post-install but are not automatically
tracked by pip uninstall.

Removes:
- Shell completion scripts (bash, zsh, fish)
- Activation scripts (homodyne-activate)
- GPU acceleration setup files (JAX with CUDA support)
- Conda activation hooks
- Environment aliases and shortcuts

Supports: conda, mamba, uv, venv, virtualenv
Architecture: JAX-first with automatic GPU detection

Usage:
    homodyne-cleanup
    homodyne-cleanup --interactive
    homodyne-cleanup --dry-run
"""

import argparse
import os
import platform
import sys
from pathlib import Path


def is_virtual_environment():
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
        or os.environ.get("MAMBA_ROOT_PREFIX") is not None
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def cleanup_completion_files():
    """Remove shell completion files from virtual environment."""
    venv_path = Path(sys.prefix)
    removed_files = []

    # Bash completion
    bash_files = [
        venv_path / "etc" / "bash_completion.d" / "homodyne-completion.bash",
        venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
        venv_path / "etc" / "conda" / "activate.d" / "homodyne-advanced-completion.sh",
    ]

    # Zsh completion
    zsh_files = [
        venv_path / "etc" / "zsh" / "homodyne-completion.zsh",
        venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
    ]

    # Fish completion
    fish_files = [
        venv_path / "share" / "fish" / "vendor_completions.d" / "homodyne.fish",
    ]

    all_completion_files = bash_files + zsh_files + fish_files

    for file_path in all_completion_files:
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(("Shell completion", file_path.name))
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_path.name}: {e}")

    return removed_files


def cleanup_gpu_files():
    """Remove GPU acceleration files from virtual environment."""
    venv_path = Path(sys.prefix)
    removed_files = []

    gpu_files = [
        venv_path / "etc" / "homodyne" / "gpu" / "gpu_activation.sh",
        venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu.sh",
    ]

    for file_path in gpu_files:
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(("GPU setup", file_path.name))
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_path.name}: {e}")

    return removed_files


def cleanup_advanced_features():
    """Remove advanced features CLI commands and activation scripts."""
    venv_path = Path(sys.prefix)
    removed_files = []

    # Advanced features CLI commands and activation scripts
    cli_commands = [
        venv_path / "bin" / "homodyne-gpu-optimize",
        venv_path / "bin" / "homodyne-validate",
        venv_path / "bin" / "homodyne-activate",  # Bash/Zsh activation
        venv_path / "bin" / "homodyne-activate.fish",  # Fish activation
    ]

    for file_path in cli_commands:
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(("Activation/CLI", file_path.name))
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_path.name}: {e}")

    return removed_files


def cleanup_xla_config():
    """Remove XLA configuration file from user's home directory."""
    removed_files = []

    xla_config_file = Path.home() / ".homodyne_xla_mode"
    if xla_config_file.exists():
        try:
            xla_config_file.unlink()
            removed_files.append(("XLA config", xla_config_file.name))
        except Exception as e:
            print(f"   ⚠️  Failed to remove {xla_config_file.name}: {e}")

    return removed_files


def cleanup_xla_activation_scripts():
    """Remove XLA activation scripts from virtual environment."""
    venv_path = Path(sys.prefix)
    removed_files = []

    xla_activation_files = [
        venv_path / "etc" / "homodyne" / "activation" / "xla_config.bash",
        venv_path / "etc" / "homodyne" / "activation" / "xla_config.fish",
    ]

    for file_path in xla_activation_files:
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(("XLA activation", file_path.name))
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_path.name}: {e}")

    return removed_files


def cleanup_old_system_files():
    """Remove old modular system files if they exist."""
    venv_path = Path(sys.prefix)
    removed_files = []

    # Old system files to remove
    old_files = [
        venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu-activate.sh",
        venv_path / "etc" / "conda" / "deactivate.d" / "homodyne-gpu-deactivate.sh",
        venv_path / "etc" / "homodyne" / "gpu_activation.sh",
        venv_path / "etc" / "homodyne" / "homodyne_aliases.sh",
        venv_path / "etc" / "homodyne" / "homodyne_completion.zsh",
        venv_path / "etc" / "homodyne" / "homodyne_config.sh",
        # Windows batch files
        venv_path / "Scripts" / "hm.bat",
        venv_path / "Scripts" / "hc.bat",
        venv_path / "Scripts" / "hr.bat",
        venv_path / "Scripts" / "ha.bat",
    ]

    for file_path in old_files:
        if file_path.exists():
            try:
                file_path.unlink()
                removed_files.append(("Old system file", file_path.name))
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_path.name}: {e}")

    return removed_files


def interactive_cleanup():
    """Interactive cleanup allowing user to choose what to remove."""
    try:
        print("\n🧹 Homodyne Interactive Cleanup")
        print("Choose what to remove:")
        print()

        # Shell completion
        print("1. Shell Completion")
        print("   - Removes bash/zsh/fish completion scripts")
        print("   - Removes completion activation scripts")

        remove_completion = (
            input("   Remove shell completion? [y/N]: ").lower().startswith("y")
        )

        # GPU setup (Linux only)
        remove_gpu = False
        if platform.system() == "Linux":
            print("\n2. GPU Acceleration Setup (JAX with CUDA)")
            print("   - Removes GPU activation scripts")
            print("   - Removes CUDA environment configuration")
            print("   - GPU works automatically via JAX device detection")

            remove_gpu = (
                input("   Remove GPU setup files? [y/N]: ").lower().startswith("y")
            )

        # Advanced features
        print("\n3. Advanced Features & Activation Scripts")
        print("   - Removes homodyne-gpu-optimize CLI command")
        print("   - Removes homodyne-validate CLI command")
        print("   - Removes homodyne-activate scripts (for uv/venv/virtualenv)")

        remove_advanced = (
            input("   Remove advanced features & activation scripts? [y/N]: ")
            .lower()
            .startswith("y")
        )

        # Old system files
        print("\n4. Old System Files")
        print("   - Removes legacy completion system files")
        print("   - Recommended for clean upgrade")

        remove_old = input("   Remove old system files? [Y/n]: ").lower() != "n"

        # XLA configuration
        print("\n5. XLA Configuration")
        print("   - Removes ~/.homodyne_xla_mode config file")
        print("   - Removes $VIRTUAL_ENV/etc/homodyne/activation/ scripts")
        print("   - Can be regenerated with: homodyne-post-install --interactive")

        remove_xla = input("   Remove XLA configuration? [y/N]: ").lower().startswith("y")

        # Perform cleanup
        all_removed = []

        if remove_completion:
            all_removed.extend(cleanup_completion_files())

        if remove_gpu:
            all_removed.extend(cleanup_gpu_files())

        if remove_advanced:
            all_removed.extend(cleanup_advanced_features())

        if remove_old:
            all_removed.extend(cleanup_old_system_files())

        if remove_xla:
            all_removed.extend(cleanup_xla_config())
            all_removed.extend(cleanup_xla_activation_scripts())

        return all_removed

    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Interactive cleanup cancelled by user")
        return []


def cleanup_all_files():
    """Remove all homodyne files from virtual environment."""
    system = platform.system()
    is_venv = is_virtual_environment()

    print("═" * 70)
    print("🧹 Homodyne Cleanup - Shell Completion System")
    print("═" * 70)
    print(f"🖥️  Platform: {system}")
    print(f"📦 Environment: {'Virtual Environment' if is_venv else 'System Python'}")

    if not is_venv:
        print("\n⚠️  Not running in a virtual environment")
        print("   Cleanup only works in virtual environments")
        print("   System installations don't create extra files")
        return False

    print(f"\n🔍 Scanning for homodyne files in: {sys.prefix}")

    try:
        # Clean up all types of files
        all_removed = []
        all_removed.extend(cleanup_completion_files())
        all_removed.extend(cleanup_gpu_files())
        all_removed.extend(cleanup_advanced_features())
        all_removed.extend(cleanup_old_system_files())
        all_removed.extend(cleanup_xla_config())
        all_removed.extend(cleanup_xla_activation_scripts())

        # Clean up empty directories
        venv_path = Path(sys.prefix)
        directories_to_clean = [
            venv_path / "etc" / "homodyne" / "gpu",
            venv_path / "etc" / "homodyne" / "activation",
            venv_path / "etc" / "homodyne",
            venv_path / "etc" / "zsh",
            venv_path / "share" / "fish" / "vendor_completions.d",
            venv_path / "etc" / "bash_completion.d",
        ]

        for dir_path in directories_to_clean:
            if dir_path.exists() and dir_path.is_dir():
                try:
                    # Only remove if empty
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        all_removed.append(("Empty directory", dir_path.name))
                except Exception as exc:
                    print(f"   ⚠️ Could not remove {dir_path}: {exc}")

        print("\n📊 Cleanup Summary:")
        if all_removed:
            print(f"   ✅ Successfully removed {len(all_removed)} items:")
            for file_type, name in all_removed:
                print(f"      • {file_type}: {name}")
            print("   🔄 Restart shell or reactivate environment to complete cleanup")
        else:
            print("   ✨ No homodyne files found to remove")
            print("   📝 Environment is already clean")

        return True

    except Exception as e:
        print(f"❌ Failed to clean up files: {e}")
        return False


def show_dry_run():
    """Show what would be removed without actually removing anything."""
    system = platform.system()
    is_venv = is_virtual_environment()

    print("═" * 70)
    print("🧹 Homodyne Cleanup - DRY RUN")
    print("═" * 70)
    print(f"🖥️  Platform: {system}")
    print(f"📦 Environment: {'Virtual Environment' if is_venv else 'System Python'}")

    if not is_venv:
        print("\n⚠️  Not running in a virtual environment")
        print("   Cleanup only works in virtual environments")
        print("   System installations don't create extra files")
        return False

    print(f"\n🔍 Would scan for homodyne files in: {sys.prefix}")

    # Simulate finding files without removing them
    venv_path = Path(sys.prefix)
    files_to_remove = []

    # Check for completion files
    completion_files = [
        (
            venv_path / "etc" / "bash_completion.d" / "homodyne-completion.bash",
            "Shell completion",
            "homodyne-completion.bash",
        ),
        (
            venv_path / "etc" / "conda" / "activate.d" / "homodyne-completion.sh",
            "Shell completion",
            "homodyne-completion.sh",
        ),
        (
            venv_path
            / "etc"
            / "conda"
            / "activate.d"
            / "homodyne-advanced-completion.sh",
            "Shell completion",
            "homodyne-advanced-completion.sh",
        ),
        (
            venv_path / "etc" / "zsh" / "homodyne-completion.zsh",
            "Shell completion",
            "homodyne-completion.zsh",
        ),
        (
            venv_path / "share" / "fish" / "vendor_completions.d" / "homodyne.fish",
            "Shell completion",
            "homodyne.fish",
        ),
    ]

    # Check for GPU files
    gpu_files = [
        (
            venv_path / "etc" / "homodyne" / "gpu" / "gpu_activation.sh",
            "GPU setup",
            "gpu_activation.sh",
        ),
        (
            venv_path / "etc" / "conda" / "activate.d" / "homodyne-gpu.sh",
            "GPU setup",
            "homodyne-gpu.sh",
        ),
    ]

    # Check for advanced features
    advanced_files = [
        (
            venv_path / "bin" / "homodyne-gpu-optimize",
            "Advanced features",
            "homodyne-gpu-optimize",
        ),
        (
            venv_path / "bin" / "homodyne-validate",
            "Advanced features",
            "homodyne-validate",
        ),
    ]

    # XLA config file and activation scripts
    xla_files = [
        (Path.home() / ".homodyne_xla_mode", "XLA config", ".homodyne_xla_mode"),
        (venv_path / "etc" / "homodyne" / "activation" / "xla_config.bash", "XLA activation", "xla_config.bash"),
        (venv_path / "etc" / "homodyne" / "activation" / "xla_config.fish", "XLA activation", "xla_config.fish"),
    ]

    all_files = completion_files + gpu_files + advanced_files + xla_files

    for file_path, file_type, name in all_files:
        if file_path.exists():
            files_to_remove.append((file_type, name))

    print("\n📊 Dry Run Results:")
    if files_to_remove:
        print(f"   📋 Would remove {len(files_to_remove)} items:")
        for file_type, name in files_to_remove:
            print(f"      • {file_type}: {name}")
        print("   🔄 Would clean up empty directories")
        print("\n💡 To actually remove these files, run without --dry-run")
    else:
        print("   ✨ No homodyne files found to remove")
        print("   📝 Environment is already clean")

    return True


def main():
    """Main cleanup routine for shell completion and GPU setup."""
    args = parse_args()

    try:
        if args.dry_run:
            success = show_dry_run()
        elif args.interactive:
            print("═" * 70)
            print("🧹 Homodyne Interactive Cleanup")
            print("═" * 70)

            if not is_virtual_environment():
                print("\n⚠️  Not in a virtual environment - nothing to clean")
                return 0

            removed_files = interactive_cleanup()

            print("\n📊 Cleanup Results:")
            if removed_files:
                print(f"   ✅ Removed {len(removed_files)} items:")
                for file_type, name in removed_files:
                    print(f"      • {file_type}: {name}")
            else:
                print("   ✨ No files were removed")

            success = True
        else:
            # Add confirmation prompt for non-interactive cleanup
            if not args.force:
                print(
                    "\n⚠️  This will remove all homodyne shell completion and setup files:",
                )
                print(
                    "   • Shell completion scripts and aliases (hm, hconfig, hm-nlsq, etc.)",
                )
                print("   • Activation scripts (homodyne-activate)")
                print("   • GPU acceleration setup files")
                print("   • Advanced features CLI commands")
                print("   • All conda activation hooks")
                print("   • XLA configuration (~/.homodyne_xla_mode)")
                print("   • XLA activation scripts ($VIRTUAL_ENV/etc/homodyne/activation/)")
                print("\n💡 To restore these files later, run:")
                print("   homodyne-post-install --interactive")
                print()

                try:
                    confirm = (
                        input("🤔 Are you sure you want to proceed? [y/N]: ")
                        .strip()
                        .lower()
                    )
                    if not confirm.startswith("y"):
                        print("🚫 Cleanup cancelled by user")
                        return 0
                except (KeyboardInterrupt, EOFError):
                    print("\n🚫 Cleanup cancelled by user")
                    return 0

            success = cleanup_all_files()

        print("\n" + "═" * 70)
        if success:
            print("✅ Homodyne cleanup completed!")
            if not args.interactive:
                print("\n💡 What was cleaned:")
                print("   ├─ Shell completion scripts (bash/zsh/fish)")
                print("   ├─ Activation scripts (homodyne-activate)")
                print("   ├─ GPU acceleration setup (JAX with CUDA)")
                print("   ├─ Conda activation hooks")
                print("   ├─ XLA configuration (~/.homodyne_xla_mode)")
                print("   ├─ XLA activation scripts ($VIRTUAL_ENV/etc/homodyne/activation/)")
                print("   └─ Legacy system files")
            print("\n🔄 Next steps:")
            print("   • Restart your shell session")
            print("   • Or reactivate your virtual environment")
            print("   • Run 'pip uninstall homodyne' to complete removal")
            print("\n🔧 To restore shell completion and setup files:")
            print("   homodyne-post-install --interactive")
        else:
            print("⚠️  Cleanup had some issues")
            print("\n💡 Troubleshooting:")
            print("   • Make sure you're in a virtual environment")
            print("   • Try: homodyne-cleanup --interactive")
            print("   • Check file permissions if needed")
        print("═" * 70)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⚠️  Cleanup cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during cleanup: {e}")
        print("💡 Please report this issue if it persists")
        return 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="homodyne-cleanup",
        description="Remove Homodyne shell completion system files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-cleanup                   # Remove all homodyne files (with confirmation)
  homodyne-cleanup --force           # Remove all homodyne files (no confirmation)
  homodyne-cleanup --interactive     # Choose what to remove
  homodyne-cleanup --dry-run         # Show what would be removed

This script removes homodyne-related files that were installed by
homodyne-post-install. Run this BEFORE 'pip uninstall homodyne'.

IMPORTANT: To restore files after cleanup, run:
  homodyne-post-install --interactive

Files removed:
  • Shell completion scripts (bash/zsh/fish)
  • Activation scripts (homodyne-activate)
  • GPU acceleration setup (JAX with CUDA)
  • Conda activation hooks
  • XLA configuration (~/.homodyne_xla_mode)
  • XLA activation scripts ($VIRTUAL_ENV/etc/homodyne/activation/)
  • Legacy system files

Supports: conda, mamba, uv, venv, virtualenv
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive cleanup - choose what to remove",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be removed without actually removing",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip confirmation prompt and force cleanup",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
