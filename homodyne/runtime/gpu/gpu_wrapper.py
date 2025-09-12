#!/usr/bin/env python3
"""
GPU Wrapper for Homodyne Analysis
==================================

Command-line wrapper that automatically activates GPU support before running homodyne.
This provides a seamless way to use homodyne with GPU acceleration without manual
environment setup.

The wrapper handles the common cuSPARSE loading issues by:
1. Setting up NVIDIA library paths from pip-installed packages
2. Configuring JAX environment variables for GPU detection
3. Cleaning conflicting system CUDA paths
4. Falling back to activation script if direct setup fails

Usage:
    homodyne-gpu --config my_config.json --method mcmc
    homodyne-gpu --method all --output results/

For troubleshooting GPU setup issues, see homodyne/runtime/README.md or run homodyne-validate --test gpu.

Author: Claude Code Assistant
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path


def find_activation_script():
    """Find the GPU activation script in common locations."""
    # Check virtual environment config directory first (preferred)
    try:
        import site

        venv_config_dir = (
            Path(site.getsitepackages()[0]).parent.parent / "etc" / "homodyne"
        )
        venv_script = venv_config_dir / "gpu_activation.sh"
        if venv_script.exists():
            return venv_script
    except (ImportError, IndexError):
        pass

    # Check user config directory (legacy/fallback)
    config_dir = Path.home() / ".config" / "homodyne"
    config_script = config_dir / "gpu_activation.sh"
    if config_script.exists():
        return config_script

    # Check current directory
    current_dir = Path.cwd()
    script_path = current_dir / "activate_gpu.sh"
    if script_path.exists():
        return script_path

    # Check homodyne package directory
    try:
        import homodyne

        package_dir = Path(homodyne.__file__).parent.parent
        script_path = package_dir / "activate_gpu.sh"
        if script_path.exists():
            return script_path
    except ImportError:
        pass

    return None


def setup_gpu_environment(quiet=False):
    """
    Configure GPU environment using system CUDA and cuDNN.

    Sets up environment variables for JAX GPU acceleration using:
    - System CUDA 12.6 at /usr/local/cuda
    - System cuDNN 9.12 at /usr/lib/x86_64-linux-gnu
    - jax[cuda12-local] package integration

    Args:
        quiet: If True, suppress console output

    Returns:
        bool: True if GPU environment configured successfully, False otherwise
    """
    try:
        import platform

        # Platform requirement check
        if platform.system() != "Linux":
            if not quiet:
                print(
                    f"homodyne-gpu: GPU acceleration requires Linux (detected: {platform.system()})"
                )
                print("Falling back to CPU-only mode")
            return False

        # Verify system CUDA installation
        cuda_root = "/usr/local/cuda"
        if not os.path.exists(cuda_root):
            if not quiet:
                print("homodyne-gpu: System CUDA not found at /usr/local/cuda")
                print("Please install CUDA Toolkit 12.x from NVIDIA")
            return False

        # Configure system CUDA environment
        os.environ["CUDA_ROOT"] = cuda_root
        os.environ["CUDA_HOME"] = cuda_root

        # Add CUDA binaries to PATH
        current_path = os.environ.get("PATH", "")
        cuda_bin = os.path.join(cuda_root, "bin")
        if cuda_bin not in current_path:
            os.environ["PATH"] = f"{cuda_bin}:{current_path}"

        # Configure library paths for system CUDA + system cuDNN
        cuda_lib = os.path.join(cuda_root, "lib64")
        cudnn_lib = "/usr/lib/x86_64-linux-gnu"
        system_lib = "/lib/x86_64-linux-gnu"

        # Build comprehensive library path
        lib_paths = [cuda_lib, cudnn_lib, system_lib]
        current_lib_path = os.environ.get("LD_LIBRARY_PATH", "")

        # Add new paths to existing library path
        new_lib_path = ":".join(lib_paths)
        if current_lib_path:
            os.environ["LD_LIBRARY_PATH"] = f"{new_lib_path}:{current_lib_path}"
        else:
            os.environ["LD_LIBRARY_PATH"] = new_lib_path

        # JAX configuration for system CUDA
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_root}"
        os.environ["JAX_PLATFORMS"] = ""  # Enable GPU detection

        if not quiet:
            print("homodyne-gpu: System CUDA environment configured")
            print(f"  • CUDA: {cuda_root}")
            print(f"  • cuDNN: {cudnn_lib}")
            print("  • JAX: jax[cuda12-local] integration")

        return True

    except Exception as e:
        if not quiet:
            print(f"homodyne-gpu: Error configuring GPU environment: {e}")
            print("Falling back to CPU-only mode")
        return False


def activate_gpu(quiet=False):
    """Activate GPU support by setting up environment directly or using activation script."""
    # Try direct environment setup first
    if setup_gpu_environment(quiet=quiet):
        return

    # Fall back to activation script method
    script_path = find_activation_script()
    if not script_path:
        return

    try:
        # Validate script path to prevent shell injection
        script_path = Path(script_path).resolve()
        if not script_path.exists() or not script_path.is_file():
            return

        # Source the activation script and capture environment changes safely
        result = subprocess.run(
            ["/bin/bash", "-c", f"source {shlex.quote(str(script_path))} && env"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            # Parse environment variables and update current environment
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
    except Exception:
        # Silently continue if GPU activation fails
        pass


def main():
    """Main entry point for homodyne-gpu wrapper."""
    import platform

    # Check if quiet mode is requested
    quiet_mode = "--quiet" in sys.argv

    # Check if method is compatible with GPU wrapper
    if "--method" in sys.argv:
        method_idx = sys.argv.index("--method") + 1
        if method_idx < len(sys.argv):
            method = sys.argv[method_idx]
            if method in ["classical", "robust"]:
                if not quiet_mode:
                    print(
                        f"homodyne-gpu: GPU acceleration not needed for --method {method}"
                    )
                    print("Classical and robust methods run on CPU only")
                    print(f"Use 'homodyne --method {method}' instead")
                sys.exit(1)

    # Check platform requirement (only print if not quiet)
    if platform.system() != "Linux":
        if not quiet_mode:
            print(f"homodyne-gpu: GPU requires Linux (detected: {platform.system()})")
            print(
                "System CUDA integration requires Linux with CUDA 12.6+ and cuDNN 9.12+"
            )
            print("Falling back to standard homodyne command (CPU-only)")
            print()
    else:
        if not quiet_mode:
            print("homodyne-gpu: Configuring system CUDA integration on Linux...")

            # Check system CUDA installation
            if not os.path.exists("/usr/local/cuda"):
                print("⚠️  System CUDA not found at /usr/local/cuda")
                print("   Please install CUDA Toolkit 12.x from NVIDIA")
                print("   Falling back to CPU-only mode")
            else:
                # Check JAX installation
                try:
                    import jax

                    print(f"✅ JAX {jax.__version__} available for GPU acceleration")
                except ImportError:
                    print(
                        "⚠️  JAX not installed - install with: pip install jax[cuda12-local]"
                    )

            # Check if GPU auto-activation is available in virtual environment
            import site

            try:
                venv_config_dir = (
                    Path(site.getsitepackages()[0]).parent.parent / "etc" / "homodyne"
                )
                if venv_config_dir.exists():
                    print(f"✅ Virtual environment GPU integration: {venv_config_dir}")
                else:
                    print("💡 Tip: Install virtual environment integration:")
                    print("   python scripts/install_gpu_autoload.py")
            except (IndexError, AttributeError):
                print("⚠️  Virtual environment detection failed")

    # Set GPU intent flag to signal that user explicitly wants GPU acceleration
    os.environ["HOMODYNE_GPU_INTENT"] = "true"

    # Activate GPU support (will handle platform checks internally)
    activate_gpu(quiet=quiet_mode)

    # Import and run homodyne main function
    try:
        from homodyne.run_homodyne import main as homodyne_main

        homodyne_main()
    except ImportError:
        print("Error: homodyne package not found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
