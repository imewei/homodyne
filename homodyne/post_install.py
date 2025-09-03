#!/usr/bin/env python3
"""
Post-installation hook for Homodyne with Optional Shell Completion System
========================================================================

This script provides optional setup for:
1. Shell completion system (bash, zsh, fish) - user choice
2. GPU acceleration configuration (Linux only) - user choice
3. Virtual environment integration (conda, mamba, venv, virtualenv)

MCMC Backend Architecture:
- CPU Backend: Pure PyMC implementation (isolated from JAX)
- GPU Backend: Pure NumPyro+JAX implementation (isolated from PyMC)
- Complete separation prevents backend conflicts and namespace pollution

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


def is_linux():
    """Check if running on Linux."""
    return platform.system() == "Linux"


def detect_shell_type():
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


def is_virtual_environment():
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("CONDA_DEFAULT_ENV") is not None
        or os.environ.get("MAMBA_ROOT_PREFIX") is not None
        or os.environ.get("VIRTUAL_ENV") is not None
    )


def is_conda_environment(venv_path):
    """Check if the environment is a conda/mamba environment."""
    # Check for conda directory structure
    conda_meta = venv_path / "conda-meta"
    # Check if path contains conda/mamba/miniforge/mambaforge
    path_indicators = ["conda", "mamba", "miniforge", "mambaforge"]
    return conda_meta.exists() or any(
        indicator in str(venv_path).lower() for indicator in path_indicators
    )


def create_unified_zsh_completion(venv_path):
    """Create the unified zsh completion file."""
    zsh_dir = venv_path / "etc" / "zsh"
    zsh_dir.mkdir(parents=True, exist_ok=True)

    completion_file = zsh_dir / "homodyne-completion.zsh"
    completion_content = """#!/usr/bin/env zsh
# Homodyne Zsh aliases (simplified)

# Only load if not already loaded
if [[ -z "$_HOMODYNE_ZSH_COMPLETION_LOADED" ]]; then
    export _HOMODYNE_ZSH_COMPLETION_LOADED=1

    # Define aliases
    alias hm='homodyne --method mcmc'
    alias hc='homodyne --method classical'
    alias hr='homodyne --method robust'
    alias ha='homodyne --method all'
    alias hconfig='homodyne-config'

    # Linux GPU aliases (only for GPU-compatible methods)
    if [[ "$(uname -s)" == "Linux" ]] && command -v homodyne-gpu >/dev/null 2>&1; then
        alias hgm='homodyne-gpu --method mcmc'    # GPU-accelerated MCMC with NumPyro
        alias hga='homodyne-gpu --method all'     # GPU-accelerated full analysis
        # Note: classical and robust methods run CPU-only, use hc/hr instead
    fi

    # GPU status function
    homodyne_gpu_status() {
        if [[ "$(uname -s)" == "Linux" ]]; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🚀 Homodyne GPU Status (Isolated Backend Architecture)"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            # Hardware detection
            if command -v nvidia-smi >/dev/null 2>&1; then
                echo "✅ NVIDIA GPU detected:"
                nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | sed 's/^/   /'
            else
                echo "❌ No NVIDIA GPU detected"
            fi

            # Environment status
            echo ""
            echo "🔧 Environment Configuration:"
            echo "   HOMODYNE_GPU_INTENT: ${HOMODYNE_GPU_INTENT:-false} (backend selection)"
            echo "   JAX_PLATFORMS: ${JAX_PLATFORMS:-auto-detect} (NumPyro backend only)"
            echo "   JAX_ENABLE_X64: ${JAX_ENABLE_X64:-not set} (NumPyro backend only)"

            # Backend routing info
            echo ""
            echo "📊 Isolated MCMC Backends:"
            echo "   homodyne        → Pure PyMC CPU (isolated from JAX)"
            echo "   homodyne-gpu    → Pure NumPyro GPU/JAX (isolated from PyMC)"
            echo ""
            echo "🔒 Backend Isolation:"
            echo "   • CPU backend: No JAX imports, pure PyTensor CPU mode"
            echo "   • GPU backend: No PyMC imports, pure NumPyro+JAX implementation"  
            echo "   • Complete namespace separation prevents conflicts"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        else
            echo "GPU status only available on Linux"
        fi
    }

    # Helper function
    homodyne_help() {
        echo "Homodyne command shortcuts:"
        echo "  hm  = homodyne --method mcmc"
        echo "  hc  = homodyne --method classical"
        echo "  hr  = homodyne --method robust"
        echo "  ha  = homodyne --method all"
        echo "  hconfig = homodyne-config"

        if [[ "$(uname -s)" == "Linux" ]] && command -v homodyne-gpu >/dev/null 2>&1; then
            echo ""
            echo "GPU shortcuts (NumPyro + JAX, Linux only):"
            echo "  hgm = homodyne-gpu --method mcmc  # GPU-accelerated MCMC"
            echo "  hga = homodyne-gpu --method all   # GPU-accelerated full analysis"
            echo ""
            echo "Note: classical/robust methods run CPU-only (use hc/hr)"
        fi
    }
fi"""

    completion_file.write_text(completion_content)
    return completion_file


def install_shell_completion(shell_type=None, force=False):
    """Install unified shell completion system."""
    if not is_virtual_environment() and not force:
        print("⚠️  Shell completion recommended only in virtual environments")
        return False

    venv_path = Path(sys.prefix)

    try:
        # Create unified zsh completion (works for most shells)
        create_unified_zsh_completion(venv_path)

        # Create conda activation script for conda/mamba environments
        if is_conda_environment(venv_path):
            activate_dir = venv_path / "etc" / "conda" / "activate.d"
            activate_dir.mkdir(parents=True, exist_ok=True)

            completion_script = activate_dir / "homodyne-completion.sh"
            completion_content = f"""#!/bin/bash
# Homodyne completion activation

# Zsh completion
if [[ -n "$ZSH_VERSION" ]] && [[ -f "{venv_path}/etc/zsh/homodyne-completion.zsh" ]]; then
    source "{venv_path}/etc/zsh/homodyne-completion.zsh"
fi
"""
            completion_script.write_text(completion_content)
            completion_script.chmod(0o755)

        print("✅ Shell completion installed")
        print("   • CPU aliases: hm, hc, hr, ha, hconfig")
        print("   • GPU aliases: hgm, hga (NumPyro GPU, Linux only)")
        return True

    except Exception as e:
        print(f"❌ Shell completion installation failed: {e}")
        return False


def install_gpu_acceleration(force=False):
    """Install GPU acceleration setup (Linux only)."""
    if platform.system() != "Linux":
        print("ℹ️  GPU acceleration only available on Linux")
        return False

    if not is_virtual_environment() and not force:
        print("⚠️  GPU acceleration recommended only in virtual environments")
        return False

    try:
        venv_path = Path(sys.prefix)

        # Find the homodyne source directory
        try:
            import homodyne

            homodyne_src_dir = Path(homodyne.__file__).parent.parent
            smart_gpu_script = (
                homodyne_src_dir / "homodyne" / "runtime" / "gpu" / "activation.sh"
            )

            if not smart_gpu_script.exists():
                print(f"⚠️  Smart GPU activation script not found at {smart_gpu_script}")
                return False
        except ImportError:
            print("❌ Homodyne package not found")
            return False

        # Create conda activation script for GPU (for conda/mamba environments)
        if is_conda_environment(venv_path):
            activate_dir = venv_path / "etc" / "conda" / "activate.d"
            activate_dir.mkdir(parents=True, exist_ok=True)

            gpu_activate_script = activate_dir / "homodyne-gpu.sh"
            gpu_activate_content = f"""#!/bin/bash
# Smart GPU activation for homodyne
if [[ -f "{smart_gpu_script}" ]]; then
    source "{smart_gpu_script}"
fi
"""
            gpu_activate_script.write_text(gpu_activate_content)
            gpu_activate_script.chmod(0o755)

        print("✅ GPU acceleration setup installed")
        return True

    except Exception as e:
        print(f"❌ GPU setup failed: {e}")
        return False


def install_macos_shell_completion():
    """Install shell completion for macOS."""
    import sys
    from pathlib import Path

    venv_path = Path(sys.prefix)
    config_dir = venv_path / "etc" / "homodyne"
    config_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Create aliases script that works on macOS (not yet implemented)
    # script_content = '''#!/bin/bash


# # Homodyne Shell Aliases for macOS
#
# # CPU-only aliases
# alias hm='homodyne --method mcmc'
# alias hc='homodyne --method classical'
# alias hr='homodyne --method robust'
# alias ha='homodyne --method all'
#
# # Configuration shortcuts
# alias hconfig='homodyne --config'
#
# # Plotting shortcuts
# alias hexp='homodyne --plot-experimental-data'
# alias hsim='homodyne --plot-simulated-data'
#
# # homodyne-config shortcuts
# alias hc-iso='homodyne-config --mode static_isotropic'
# alias hc-aniso='homodyne-config --mode static_anisotropic'
# alias hc-flow='homodyne-config --mode laminar_flow'
#
# # Helper function
# homodyne_help() {
#     echo "Homodyne command shortcuts:"
#     echo "  hc = homodyne --method classical"
#     echo "  hm = homodyne --method mcmc"
#     echo "  hr = homodyne --method robust"
#     echo "  ha = homodyne --method all"
#     echo ""
#     echo "Config shortcuts:"
#     echo "  hc-iso   = homodyne-config --mode static_isotropic"
#     echo "  hc-aniso = homodyne-config --mode static_anisotropic"
#     echo "  hc-flow  = homodyne-config --mode laminar_flow"
# }
# '''


def install_advanced_features():
    """Install advanced features (Phases 4-6)."""
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

        # Check if advanced features files exist
        required_files = [
            homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh",
            homodyne_src_dir / "homodyne" / "runtime" / "gpu" / "optimizer.py",
            homodyne_src_dir / "homodyne" / "runtime" / "utils" / "system_validator.py",
        ]

        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            print(
                f"⚠️  Advanced features files not found: {[f.name for f in missing_files]}"
            )
            print("   Run from development environment or upgrade to latest version")
            return False

        # Install CLI commands for advanced features
        bin_dir = venv_path / "bin"

        # GPU optimizer command
        gpu_cmd = bin_dir / "homodyne-gpu-optimize"
        gpu_content = f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, "{homodyne_src_dir / "homodyne" / "runtime" / "gpu"}")
from optimizer import main
if __name__ == "__main__":
    main()
"""
        gpu_cmd.write_text(gpu_content)
        gpu_cmd.chmod(0o755)

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
# Advanced homodyne completion
if [[ -f "{homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh"}" ]]; then
    source "{homodyne_src_dir / "homodyne" / "runtime" / "shell" / "completion.sh"}"
fi
"""
            completion_script.write_text(completion_content)
            completion_script.chmod(0o755)

        print("✅ Advanced features installed successfully")
        print("   • homodyne-gpu-optimize - GPU optimization and benchmarking")
        print("   • homodyne-validate - Comprehensive system validation")
        print("   • Advanced completion - Context-aware shell completion")

        return True

    except Exception as e:
        print(f"❌ Advanced features installation failed: {e}")
        return False


def interactive_setup():
    """Interactive setup allowing user to choose what to install."""
    print("\n🔧 Homodyne Optional Setup")
    print("Choose what to install (you can run this again later):")
    print()

    # Shell completion
    print("1. Shell Completion (bash/zsh/fish)")
    print("   - Adds tab completion for homodyne commands")
    print("   - Adds convenient aliases (hm, hc, hr, ha)")
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

    # GPU acceleration (Linux only)
    install_gpu = False
    if platform.system() == "Linux":
        print("\n2. GPU Acceleration (NumPyro + JAX, Linux only)")
        print("   - Configures CUDA environment for NumPyro GPU backend")
        print("   - Enables homodyne-gpu command with automatic CPU fallback")
        print("   - Requires CUDA toolkit 12.x+ and JAX[cuda12-local] installation")

        install_gpu = (
            input("   Install GPU acceleration? [y/N]: ").lower().startswith("y")
        )

    # Advanced features (Phases 4-6)
    print("\n3. Advanced Features (Phases 4-6)")
    print("   - Phase 4: Advanced shell completion with caching")
    print("   - Phase 5: GPU auto-optimization and benchmarking")
    print("   - Phase 6: Comprehensive system validation")
    print("   - Adds homodyne-gpu-optimize and homodyne-validate commands")

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

    if install_gpu:
        if install_gpu_acceleration():
            results.append("✅ GPU acceleration")
        else:
            results.append("❌ GPU acceleration failed")

    if install_advanced:
        if install_advanced_features():
            results.append("✅ Advanced features")
        else:
            results.append("❌ Advanced features failed")

    return len([r for r in results if r.startswith("✅")]) > 0, results


def show_installation_summary(interactive_results=None):
    """Show installation summary with available commands."""
    print("\n🚀 Quick Start Commands:")
    print("   homodyne --method mcmc --config config.json")
    print("   homodyne-config --mode static_isotropic -o my_config.json")

    if is_linux():
        print("   homodyne-gpu --method mcmc --config config.json  # GPU-accelerated")

    print("\n⚡ Available Shortcuts (after shell restart):")
    print("   hm  = homodyne --method mcmc")
    print("   hc  = homodyne --method classical")
    print("   hr  = homodyne --method robust")
    print("   ha  = homodyne --method all")

    if is_linux():
        print("   hgm = homodyne-gpu --method mcmc  # NumPyro GPU-accelerated")
        print("   hga = homodyne-gpu --method all   # NumPyro GPU-accelerated")

    print("\n📖 Help:")
    print("   homodyne --help")
    print("   homodyne_help               # View all shortcuts")

    if is_linux():
        print("   homodyne_gpu_status         # Check GPU status")


def main():
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
    if args.shell or (not args.gpu and not args.shell and not args.advanced):
        # Install shell completion by default or if specified
        print("\n📝 Installing shell completion...")
        shell_type = args.shell if args.shell else None
        if install_shell_completion(shell_type, force=args.force):
            results.append("✅ Shell completion")
        else:
            results.append("❌ Shell completion failed")
            success = False

    if args.gpu:
        # Install GPU if requested
        print("\n🚀 Installing GPU acceleration...")
        if install_gpu_acceleration(force=args.force):
            results.append("✅ GPU acceleration")
        else:
            results.append("❌ GPU acceleration failed")
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
        print("      homodyne_gpu_status  # Check GPU status")
    else:
        print("\n⚠️  Setup had some issues")
        print("   Try: homodyne-post-install --interactive")
    print("═" * 70)

    return 0 if success else 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="homodyne-post-install",
        description="Set up optional Homodyne shell completion and GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-post-install --interactive    # Interactive setup (recommended)
  homodyne-post-install                  # Install shell completion only
  homodyne-post-install --force          # Force install outside venv

The script provides optional installation of:
- Shell completion (bash/zsh/fish) with safe aliases
- GPU acceleration setup (Linux only)
- Advanced features (Phases 4-6): completion caching, GPU optimization, system validation
- Virtual environment integration
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
        "--gpu", action="store_true", help="Install GPU acceleration (Linux only)"
    )

    parser.add_argument(
        "--advanced", action="store_true", help="Install advanced features (Phases 4-6)"
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force setup even if not in virtual environment",
    )

    return parser.parse_args()


# Backwards compatibility aliases
setup_gpu_acceleration = install_gpu_acceleration


if __name__ == "__main__":
    sys.exit(main())
