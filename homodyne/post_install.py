#!/usr/bin/env python3
"""Post-installation hook for Homodyne with Optional Shell Completion System
========================================================================

This script provides optional setup for:
1. Shell completion system (bash, zsh, fish) - user choice
2. Virtual environment integration (conda, mamba, venv, virtualenv)

Version 2.3.0: CPU-only architecture (GPU support removed)

Features:
- Safe completion scripts that don't interfere with system commands
- Cross-platform support: bash, zsh, fish
- Optional installation - user can choose what to install
- Easy removal with homodyne-cleanup
- Robust error handling and graceful degradation
"""

import argparse
import os
import platform
import sys
from pathlib import Path


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def detect_shell_type() -> str:
    """Detect the current shell type."""
    shell = os.environ.get("SHELL", "")
    if "zsh" in shell:
        return "zsh"
    elif "bash" in shell:
        return "bash"
    elif "fish" in shell:
        return "fish"
    else:
        return "bash"  # Default fallback


def is_virtual_environment() -> bool:
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
        or os.environ.get("MAMBA_ROOT_PREFIX") is not None
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def is_conda_environment(venv_path: Path) -> bool:
    """Check if the environment is a conda/mamba environment."""
    # Check for conda directory structure
    conda_meta = venv_path / "conda-meta"
    # Check if path contains conda/mamba/miniforge/mambaforge
    path_indicators = ["conda", "mamba", "miniforge", "mambaforge"]
    return conda_meta.exists() or any(
        indicator in str(venv_path).lower() for indicator in path_indicators
    )


def create_unified_zsh_completion(venv_path: Path) -> Path:
    """Create the unified zsh completion file."""
    zsh_dir = venv_path / "etc" / "zsh"
    zsh_dir.mkdir(parents=True, exist_ok=True)

    completion_file = zsh_dir / "homodyne-completion.zsh"
    completion_content = """#!/usr/bin/env zsh
# Homodyne Zsh aliases

# Only load if not already loaded
if [[ -z "$_HOMODYNE_ZSH_COMPLETION_LOADED" ]]; then
    export _HOMODYNE_ZSH_COMPLETION_LOADED=1

    # Base command aliases
    alias hm='homodyne'
    alias hconfig='homodyne-config'

    # Method aliases (hm- prefix)
    alias hm-nlsq='homodyne --method nlsq'    # NLSQ trust-region optimization (primary)
    alias hm-cmc='homodyne --method cmc'    # Consensus Monte Carlo

    # Config mode aliases (hc- prefix)
    alias hc-stat='homodyne-config --mode static'         # Generate static mode config
    alias hc-flow='homodyne-config --mode laminar_flow'   # Generate laminar_flow mode config

    # Plotting shortcuts
    alias hexp='homodyne --plot-experimental-data'
    alias hsim='homodyne --plot-simulated-data'

    # Helper function
    homodyne_help() {
        echo "Homodyne command shortcuts:"
        echo "  hm       = homodyne"
        echo "  hconfig  = homodyne-config"
        echo ""
        echo "Method shortcuts (hm- prefix):"
        echo "  hm-nlsq  = homodyne --method nlsq  # NLSQ trust-region (primary)"
        echo "  hm-cmc  = homodyne --method cmc  # Consensus Monte Carlo"
        echo ""
        echo "Config mode shortcuts (hc- prefix):"
        echo "  hc-stat  = homodyne-config --mode static"
        echo "  hc-flow  = homodyne-config --mode laminar_flow"
        echo ""
        echo "Plotting shortcuts:"
        echo "  hexp     = homodyne --plot-experimental-data"
        echo "  hsim     = homodyne --plot-simulated-data"
    }
fi"""

    completion_file.write_text(completion_content)
    return completion_file


def integrate_with_venv_activate(venv_path: Path) -> list[Path]:
    """Integrate homodyne completion with virtual environment's activate scripts.

    This modifies the venv's own activate scripts (not user's global shell configs)
    so completion is automatically available when the environment is activated.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of modified activation script paths
    """
    bin_dir = venv_path / "bin"
    modified_scripts = []

    # Bash/Zsh activation integration
    bash_activate = bin_dir / "activate"
    if bash_activate.exists():
        try:
            content = bash_activate.read_text()

            # Check if already integrated
            if "homodyne-completion" not in content:
                # Append homodyne completion sourcing
                completion_code = """
# Homodyne shell completion (auto-added by homodyne-post-install)
if [[ -f "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh" ]]; then
    source "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh"
fi
"""
                with bash_activate.open("a") as f:
                    f.write(completion_code)

                modified_scripts.append(bash_activate)
                print(f"   ✅ Integrated with {bash_activate.name}")
            else:
                print(f"   ℹ️  Already integrated with {bash_activate.name}")

        except Exception as e:
            print(f"   ⚠️  Could not integrate with {bash_activate.name}: {e}")

    # Fish activation integration
    fish_activate = bin_dir / "activate.fish"
    if fish_activate.exists():
        try:
            content = fish_activate.read_text()

            # Check if already integrated
            if "homodyne-completion" not in content:
                # Append homodyne completion sourcing
                completion_code = """
# Homodyne shell completion (auto-added by homodyne-post-install)
if test -f "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh"
    source "$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh"
end
"""
                with fish_activate.open("a") as f:
                    f.write(completion_code)

                modified_scripts.append(fish_activate)
                print(f"   ✅ Integrated with {fish_activate.name}")
            else:
                print(f"   ℹ️  Already integrated with {fish_activate.name}")

        except Exception as e:
            print(f"   ⚠️  Could not integrate with {fish_activate.name}: {e}")

    return modified_scripts


def create_xla_activation_scripts(venv_path: Path) -> list[Path]:
    """Create XLA_FLAGS activation scripts for bash/zsh and fish.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of created XLA activation script paths
    """
    created_scripts: list[Path] = []

    try:
        # Find homodyne package to copy activation scripts
        import homodyne

        homodyne_pkg = Path(homodyne.__file__).parent

        # Source XLA activation scripts
        xla_bash_src = (
            homodyne_pkg / "runtime" / "shell" / "activation" / "xla_config.bash"
        )
        xla_fish_src = (
            homodyne_pkg / "runtime" / "shell" / "activation" / "xla_config.fish"
        )

        if not xla_bash_src.exists() or not xla_fish_src.exists():
            print("   ⚠️  XLA activation scripts not found in homodyne package")
            return created_scripts

        # Create activation directory
        activation_dir = venv_path / "etc" / "homodyne" / "activation"
        activation_dir.mkdir(parents=True, exist_ok=True)

        # Copy bash/zsh activation script
        xla_bash_dest = activation_dir / "xla_config.bash"
        xla_bash_dest.write_text(xla_bash_src.read_text())
        xla_bash_dest.chmod(0o644)
        created_scripts.append(xla_bash_dest)

        # Copy fish activation script
        xla_fish_dest = activation_dir / "xla_config.fish"
        xla_fish_dest.write_text(xla_fish_src.read_text())
        xla_fish_dest.chmod(0o644)
        created_scripts.append(xla_fish_dest)

        print(f"   ✅ Created XLA activation scripts in {activation_dir}")

    except Exception as e:
        print(f"   ⚠️  Could not create XLA activation scripts: {e}")

    return created_scripts


def integrate_xla_with_venv_activate(venv_path: Path) -> list[Path]:
    """Integrate XLA_FLAGS configuration with virtual environment's activate scripts.

    Parameters
    ----------
    venv_path : Path
        Path to virtual environment

    Returns
    -------
    list[Path]
        List of modified activation script paths
    """
    bin_dir = venv_path / "bin"
    modified_scripts = []

    # First create the XLA activation scripts
    create_xla_activation_scripts(venv_path)

    # Bash/Zsh activation integration
    bash_activate = bin_dir / "activate"
    if bash_activate.exists():
        try:
            content = bash_activate.read_text()

            # Check if already integrated
            if "homodyne XLA configuration" not in content:
                # Append XLA configuration sourcing
                xla_code = """
# Homodyne XLA configuration (auto-added by homodyne-post-install)
if [[ -f "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash" ]]; then
    source "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash"
fi
"""
                with bash_activate.open("a") as f:
                    f.write(xla_code)

                modified_scripts.append(bash_activate)
                print(f"   ✅ Integrated XLA config with {bash_activate.name}")
            else:
                print(f"   ℹ️  XLA config already integrated with {bash_activate.name}")

        except Exception as e:
            print(
                f"   ⚠️  Could not integrate XLA config with {bash_activate.name}: {e}"
            )

    # Fish activation integration
    fish_activate = bin_dir / "activate.fish"
    if fish_activate.exists():
        try:
            content = fish_activate.read_text()

            # Check if already integrated
            if "homodyne XLA configuration" not in content:
                # Append XLA configuration sourcing
                xla_code = """
# Homodyne XLA configuration (auto-added by homodyne-post-install)
if test -f "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.fish"
    source "$VIRTUAL_ENV/etc/homodyne/activation/xla_config.fish"
end
"""
                with fish_activate.open("a") as f:
                    f.write(xla_code)

                modified_scripts.append(fish_activate)
                print(f"   ✅ Integrated XLA config with {fish_activate.name}")
            else:
                print(f"   ℹ️  XLA config already integrated with {fish_activate.name}")

        except Exception as e:
            print(
                f"   ⚠️  Could not integrate XLA config with {fish_activate.name}: {e}"
            )

    return modified_scripts


def configure_xla_mode(xla_mode: str | None = None) -> bool:
    """Configure XLA_FLAGS mode for the current environment.

    Parameters
    ----------
    xla_mode : str, optional
        XLA mode to configure (cmc, cmc-hpc, nlsq, auto)
        If None, prompts user interactively

    Returns
    -------
    bool
        True if configuration succeeded, False otherwise
    """
    from homodyne.cli.xla_config import VALID_MODES, set_mode

    if xla_mode is None:
        print("\n🔧 XLA Configuration")
        print("Select XLA device mode:")
        print("  cmc        - 4 devices (multi-core workstations)")
        print("  cmc-hpc    - 8 devices (HPC with 36+ cores)")
        print("  nlsq       - 1 device (NLSQ-only workflows)")
        print("  auto       - Auto-detect based on CPU cores")

        xla_mode = input("\nXLA mode [cmc]: ").strip() or "cmc"

    if xla_mode not in VALID_MODES:
        print(f"   ❌ Invalid XLA mode: {xla_mode}")
        print(f"   Valid modes: {', '.join(VALID_MODES)}")
        return False

    try:
        if set_mode(xla_mode):
            print("   ✅ XLA mode configured successfully")
            return True
        else:
            print("   ❌ Failed to configure XLA mode")
            return False
    except Exception as e:
        print(f"   ❌ XLA configuration failed: {e}")
        return False


def install_shell_completion(shell_type: str | None = None, force: bool = False) -> bool:
    """Install unified shell completion system.

    Integrates completion directly into the virtual environment's activate scripts.
    Does NOT modify user's global shell configuration files (~/.bashrc, ~/.zshrc).

    Parameters
    ----------
    shell_type : str, optional
        Shell type (bash, zsh, fish). Auto-detected if not provided.
    force : bool, default False
        Force installation even if not in virtual environment
    """
    if not is_virtual_environment() and not force:
        print("⚠️  Shell completion recommended only in virtual environments")
        return False

    venv_path = Path(sys.prefix)

    try:
        # Create unified zsh completion (works for most shells)
        create_unified_zsh_completion(venv_path)

        # Environment-specific setup
        is_conda = is_conda_environment(venv_path)

        if is_conda:
            # Create conda activation script (automatic on conda activate)
            activate_dir = venv_path / "etc" / "conda" / "activate.d"
            activate_dir.mkdir(parents=True, exist_ok=True)

            # Create XLA activation scripts
            create_xla_activation_scripts(venv_path)

            completion_script = activate_dir / "homodyne-completion.sh"
            completion_content = f"""#!/bin/bash
# Homodyne completion activation

# Zsh completion
if [[ -n "$ZSH_VERSION" ]] && [[ -f "{venv_path}/etc/zsh/homodyne-completion.zsh" ]]; then
    source "{venv_path}/etc/zsh/homodyne-completion.zsh"
fi

# Bash completion
if [[ -n "$BASH_VERSION" ]] && [[ -f "{venv_path}/etc/zsh/homodyne-completion.zsh" ]]; then
    source "{venv_path}/etc/zsh/homodyne-completion.zsh"
fi

# XLA configuration
if [[ -f "{venv_path}/etc/homodyne/activation/xla_config.bash" ]]; then
    source "{venv_path}/etc/homodyne/activation/xla_config.bash"
fi
"""
            completion_script.write_text(completion_content)
            completion_script.chmod(0o755)

            print("✅ Shell completion installed (conda/mamba)")
            print("   • Auto-activated on environment activation")
            print("   • Aliases: hm, hconfig, hm-nlsq, hm-cmc")
            print("   • Config: hc-stat, hc-flow")
            print("   • Plotting: hexp, hsim")
            print()
            print(
                "📋 Completion activates automatically when you activate this environment"
            )
        else:
            # Integrate with venv/uv/virtualenv activation scripts
            modified_scripts = integrate_with_venv_activate(venv_path)

            # Also integrate XLA configuration
            xla_modified = integrate_xla_with_venv_activate(venv_path)
            modified_scripts.extend(xla_modified)

            print("✅ Shell completion installed (uv/venv/virtualenv)")
            print("   • Aliases: hm, hconfig, hm-nlsq, hm-cmc")
            print("   • Config: hc-stat, hc-flow")
            print("   • Plotting: hexp, hsim")
            print("   • XLA_FLAGS: Auto-configured on activation")
            print()
            print(
                "📋 Completion activates automatically when you activate this environment"
            )
            print(f"   Modified: {', '.join(s.name for s in modified_scripts)}")
            print()
            print("💡 To test:")
            print("   1. Deactivate and reactivate this environment")
            print("   2. Try: hm --help")

        return True

    except Exception as e:
        print(f"❌ Shell completion installation failed: {e}")
        return False


# GPU acceleration removed in v2.3.0 - function removed for CPU-only architecture


def install_macos_shell_completion() -> None:
    """Install shell completion for macOS."""
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    config_dir = venv_path / "etc" / "homodyne"
    config_dir.mkdir(parents=True, exist_ok=True)

    # NOTE (Dec 2025): macOS shell aliases feature is not yet implemented.
    # This function is a placeholder for future macOS shell completion support.
    # Users can manually add aliases to their ~/.zshrc or ~/.bashrc:
    #
    #   alias hm='homodyne --method cmc'
    #   alias hconfig='homodyne --config'
    #
    # See docs/README.md for CLI usage documentation.
    pass


def install_advanced_features() -> bool:
    """Install advanced features: completion caching and system validation."""
    print("🚀 Installing Advanced Features...")

    try:
        venv_path = Path(sys.prefix)

        # Find the homodyne source directory
        try:
            import homodyne

            homodyne_src_dir = Path(homodyne.__file__).parent.parent
        except ImportError:
            print("❌ Homodyne package not found")
            return False

        # Check if advanced features files exist (CPU-only)
        required_files = [
            homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh",
            homodyne_src_dir / "homodyne" / "runtime" / "utils" / "system_validator.py",
        ]

        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print(
                f"⚠️  Advanced features files not found: {[f.name for f in missing_files]}",
            )
            print("   Run from development environment or upgrade to latest version")
            return False

        # Install CLI commands for advanced features
        bin_dir = venv_path / "bin"

        # System validator command
        validator_cmd = bin_dir / "homodyne-validate"
        validator_content = f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, "{homodyne_src_dir / "homodyne" / "runtime" / "utils"}")
from system_validator import main
if __name__ == "__main__":
    main()
"""
        validator_cmd.write_text(validator_content)
        validator_cmd.chmod(0o755)

        # Install advanced completion if conda environment
        if is_conda_environment(venv_path):
            activate_dir = venv_path / "etc" / "conda" / "activate.d"
            activate_dir.mkdir(parents=True, exist_ok=True)

            completion_script = activate_dir / "homodyne-advanced-completion.sh"
            completion_content = f"""#!/bin/bash
# Advanced homodyne completion (CPU-only v2.3.0)
if [[ -f "{homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh"}" ]]; then
    source "{homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh"}"
fi
"""
            completion_script.write_text(completion_content)
            completion_script.chmod(0o755)

        print("✅ Advanced features installed successfully")
        print("   • homodyne-validate - Comprehensive system validation")
        print("   • Advanced completion - Context-aware shell completion")

        return True

    except Exception as e:
        print(f"❌ Advanced features installation failed: {e}")
        return False


def interactive_setup() -> tuple[bool, list[str]]:
    """Interactive setup allowing user to choose what to install (CPU-only in v2.3.0)."""
    print("\n🔧 Homodyne Optional Setup (v2.3.0 - CPU-only)")
    print("Choose what to install (you can run this again later):")
    print()

    # Shell completion
    print("1. Shell Completion (bash/zsh/fish)")
    print("   - Adds tab completion for homodyne commands")
    print("   - Adds convenient aliases (hm, hconfig, hm-nlsq, hm-cmc)")
    print("   - Safe: doesn't interfere with system commands")

    install_completion = (
        input("   Install shell completion? [y/N]: ").lower().startswith("y")
    )

    shell_type = None
    if install_completion:
        print("\n   Detected shells:")
        current_shell = os.environ.get("SHELL", "").split("/")[-1]
        if current_shell:
            print(f"   - Current: {current_shell}")
        print("   - Available: bash, zsh, fish")

        shell_input = input(f"   Shell type [{current_shell or 'bash'}]: ").strip()
        shell_type = (
            shell_input
            if shell_input in ["bash", "zsh", "fish"]
            else (current_shell or "bash")
        )

    # XLA configuration
    print("\n2. XLA_FLAGS Configuration")
    print("   - Auto-configures JAX CPU devices on venv activation")
    print("   - Modes: cmc (4 devices), cmc-hpc (8 devices), nlsq (1 device), auto")
    print("   - Essential for MCMC parallelization and optimal CPU usage")

    configure_xla = input("   Configure XLA_FLAGS? [Y/n]: ").lower() != "n"

    xla_mode = None
    if configure_xla:
        xla_mode = None  # Will be prompted in configure_xla_mode()

    # Advanced features
    print("\n3. Advanced Features")
    print("   - Smart shell completion with config file caching")
    print("   - System validation (homodyne-validate command)")
    print("   - Adds homodyne-validate command")

    install_advanced = (
        input("   Install advanced features? [y/N]: ").lower().startswith("y")
    )

    # Perform installations
    results = []

    if install_completion and shell_type:
        if install_shell_completion(shell_type):
            results.append(f"✅ {shell_type.title()} completion")
        else:
            results.append(f"❌ {shell_type.title()} completion failed")

    if configure_xla:
        if configure_xla_mode(xla_mode):
            results.append("✅ XLA configuration")
        else:
            results.append("❌ XLA configuration failed")

    if install_advanced:
        if install_advanced_features():
            results.append("✅ Advanced features")
        else:
            results.append("❌ Advanced features failed")

    return len([r for r in results if r.startswith("✅")]) > 0, results


def show_installation_summary(interactive_results: list[str] | None = None) -> None:
    """Show installation summary with available commands."""
    print("\n🚀 Quick Start Commands:")
    print("   homodyne --method nlsq --config config.yaml")
    print(
        "   homodyne --method cmc --config config.yaml  # Automatic NUTS/CMC selection"
    )
    print("   homodyne-config --mode static -o my_config.yaml")

    print("\n⚡ Available Shortcuts (after shell restart):")
    print("   Base commands:")
    print("     hm       = homodyne")
    print("     hconfig  = homodyne-config")
    print("\n   Method shortcuts (hm- prefix):")
    print("     hm-nlsq  = homodyne --method nlsq  # NLSQ trust-region (primary)")
    print("     hm-cmc  = homodyne --method cmc  # Consensus Monte Carlo")
    print("\n   Config mode shortcuts (hc- prefix):")
    print("     hc-stat  = homodyne-config --mode static")
    print("     hc-flow  = homodyne-config --mode laminar_flow")

    print("\n📖 Help:")
    print("   homodyne --help")
    print("   homodyne-config --help")
    print("   homodyne_help               # View all shortcuts")


def main() -> int:
    """Main post-installation routine with optional shell completion system."""
    args = parse_args()

    print("═" * 70)
    print("🔧 Homodyne Post-Installation Setup")
    print("═" * 70)

    # Detect environment and platform
    is_venv = is_virtual_environment()
    system = platform.system()

    print(f"🖥️  Platform: {system}")
    print(f"📦 Environment: {'Virtual Environment' if is_venv else 'System Python'}")

    if not is_venv and not args.force:
        print("\n⚠️  Virtual environment recommended for optimal setup")
        print("   Run in conda/mamba/venv for full functionality")
        print("   Use --force to install anyway")
        print("\n💡 Basic usage (no setup needed):")
        print("   homodyne --help")
        print("   homodyne-config --help")
        return 0

    if args.interactive:
        success, results = interactive_setup()
        print("\n" + "═" * 70)
        if success:
            print("✅ Setup completed!")
            for result in results:
                print(f"   {result}")
            print("\n💡 Next steps:")
            print("   1. Restart your shell or reactivate environment")
            print("   2. Test: homodyne --help")
            print("   3. Try shortcuts: hm --help")
        else:
            print("⚠️  Setup completed with issues")
            for result in results:
                print(f"   {result}")
        print("═" * 70)
        return 0 if success else 1

    # Non-interactive mode - install based on arguments
    results = []
    success = True

    # Determine what to install
    if args.shell or (not args.shell and not args.advanced and not args.xla_mode):
        # Install shell completion by default or if specified
        print("\n📝 Installing shell completion...")
        shell_type = args.shell if args.shell else None
        if install_shell_completion(shell_type, force=args.force):
            results.append("✅ Shell completion")
        else:
            results.append("❌ Shell completion failed")
            success = False

    if args.xla_mode:
        # Configure XLA if requested
        print("\n🔧 Configuring XLA_FLAGS...")
        if configure_xla_mode(args.xla_mode):
            results.append("✅ XLA configuration")
        else:
            results.append("❌ XLA configuration failed")
            success = False

    if args.advanced:
        # Install advanced features if requested
        print("\n🚀 Installing Advanced Features...")
        if install_advanced_features():
            results.append("✅ Advanced features")
        else:
            results.append("❌ Advanced features failed")
            success = False

    print("\n" + "═" * 70)
    if results:
        print("Installation results:")
        for result in results:
            print(f"   {result}")

    if success:
        print("\n✅ Setup completed!")
        print("\n💡 Next steps:")
        print("   1. Restart shell or reactivate environment:")
        print("      conda deactivate && conda activate $CONDA_DEFAULT_ENV")
        print("   2. Test commands:")
        print("      hm --help  # Should work after reactivation")
    else:
        print("\n⚠️  Setup had some issues")
        print("   Try: homodyne-post-install --interactive")
    print("═" * 70)

    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="homodyne-post-install",
        description="Set up optional Homodyne shell completion (v2.3.0 - CPU-only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-post-install --interactive              # Interactive setup (recommended)
  homodyne-post-install                            # Install shell completion only
  homodyne-post-install --xla-mode auto            # Configure XLA with auto-detection
  homodyne-post-install --shell bash --xla-mode cmc   # Shell completion + XLA
  homodyne-post-install --force                    # Force install outside venv

The script provides optional installation of:
- Shell completion (bash/zsh/fish) with safe aliases
- XLA_FLAGS auto-configuration for JAX CPU devices
- Advanced features: completion caching, system validation
- Virtual environment integration

Version 2.3.0: GPU support removed - CPU-only architecture
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive setup - choose what to install",
    )

    parser.add_argument(
        "--shell",
        choices=["bash", "zsh", "fish"],
        help="Specify shell type for completion",
    )

    parser.add_argument(
        "--xla-mode",
        choices=["cmc", "cmc-hpc", "nlsq", "auto"],
        help="Configure XLA_FLAGS mode (cmc, cmc-hpc, nlsq, auto)",
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Install advanced features (completion caching, system validation)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force setup even if not in virtual environment",
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
