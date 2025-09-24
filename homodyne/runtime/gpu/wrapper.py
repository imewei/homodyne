"""
GPU Environment Setup for Homodyne v2
=====================================

GPU environment setup and activation functions for VI+JAX and MCMC+JAX.
Optimized wrapper focused on JAX GPU acceleration.

Key Features:
- JAX-specific GPU environment setup
- CUDA detection and configuration
- Memory fraction optimization
- CPU-primary, GPU-optional architecture
"""

import logging
import os
import platform
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_gpu_environment(quiet: bool = False) -> bool:
    """
    Configure GPU environment for JAX acceleration.

    Sets up environment variables for JAX GPU acceleration using:
    - System CUDA installation
    - JAX-specific optimizations
    - Memory fraction management

    Args:
        quiet: If True, suppress console output

    Returns:
        bool: True if GPU environment configured successfully
    """
    try:
        # Platform requirement check
        if platform.system() != "Linux":
            if not quiet:
                print(
                    f"GPU acceleration requires Linux (detected: {platform.system()})"
                )
            return False

        # Verify system CUDA installation
        cuda_paths = ["/usr/local/cuda", "/opt/cuda", os.environ.get("CUDA_HOME", "")]
        cuda_root = None

        for cuda_path in cuda_paths:
            if cuda_path and Path(cuda_path).exists():
                cuda_root = cuda_path
                break

        if not cuda_root:
            if not quiet:
                print("System CUDA not found - checking common locations")
                print("Please install CUDA Toolkit 12.x from NVIDIA")
            return False

        # Configure CUDA environment
        os.environ["CUDA_ROOT"] = cuda_root
        os.environ["CUDA_HOME"] = cuda_root

        # Add CUDA binaries to PATH
        current_path = os.environ.get("PATH", "")
        cuda_bin = os.path.join(cuda_root, "bin")
        if cuda_bin not in current_path:
            os.environ["PATH"] = f"{cuda_bin}:{current_path}"

        # Configure library paths
        cuda_lib = os.path.join(cuda_root, "lib64")
        current_lib_path = os.environ.get("LD_LIBRARY_PATH", "")

        if cuda_lib not in current_lib_path:
            if current_lib_path:
                os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}:{current_lib_path}"
            else:
                os.environ["LD_LIBRARY_PATH"] = cuda_lib

        # JAX-specific configuration
        os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={cuda_root}"
        os.environ["JAX_PLATFORMS"] = "gpu,cpu"  # Prefer GPU, fallback to CPU

        # Set conservative memory fraction
        if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

        if not quiet:
            print("JAX GPU environment configured")
            print(f"  CUDA: {cuda_root}")
            print("  JAX: GPU-primary, CPU-fallback")

        return True

    except Exception as e:
        if not quiet:
            print(f"Error configuring GPU environment: {e}")
        logger.exception("GPU setup failed")
        return False


def activate_gpu(quiet: bool = False) -> bool:
    """
    Activate GPU support for JAX.

    Simplified activation focused on JAX GPU acceleration.

    Args:
        quiet: If True, suppress console output

    Returns:
        bool: True if GPU activated successfully
    """
    # Check if already configured
    if os.environ.get("HOMODYNE_GPU_ACTIVATED") == "1":
        if not quiet:
            print(" GPU already activated")
        return True

    # Try to set up GPU environment
    success = setup_gpu_environment(quiet=quiet)

    if success:
        os.environ["HOMODYNE_GPU_ACTIVATED"] = "1"

        if not quiet:
            print(" JAX GPU environment activated")

            # Try to import JAX and test GPU
            try:
                import jax

                devices = jax.devices()
                gpu_devices = [
                    d
                    for d in devices
                    if "gpu" in str(d).lower() or "cuda" in str(d).lower()
                ]

                if gpu_devices:
                    print(f" JAX found {len(gpu_devices)} GPU device(s)")
                else:
                    print("�  JAX GPU devices not found - using CPU")

            except ImportError:
                print(
                    "�  JAX not available - install with: pip install jax[cuda12-local]"
                )
    else:
        if not quiet:
            print("9  Using CPU-only mode")
        os.environ["JAX_PLATFORMS"] = "cpu"

    return success


def get_gpu_status() -> dict[str, Any]:
    """
    Get current GPU status and configuration.

    Returns:
        Dict with GPU status information
    """
    status = {
        "activated": os.environ.get("HOMODYNE_GPU_ACTIVATED") == "1",
        "cuda_home": os.environ.get("CUDA_HOME"),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", "cpu"),
        "memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8"),
        "devices": [],
        "jax_available": False,
        "gpu_available": False,
    }

    # Check JAX status
    try:
        import jax

        status["jax_available"] = True
        devices = jax.devices()
        status["devices"] = [str(d) for d in devices]
        status["gpu_available"] = any(
            "gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices
        )
    except ImportError:
        pass

    return status


def print_gpu_status():
    """Print detailed GPU status information."""
    print("=� Homodyne v2 GPU Status")
    print("=" * 40)

    status = get_gpu_status()

    # Activation status
    if status["activated"]:
        print(" GPU Environment: Activated")
    else:
        print("L GPU Environment: Not Activated")

    # CUDA status
    if status["cuda_home"]:
        print(f" CUDA Path: {status['cuda_home']}")
    else:
        print("L CUDA: Not configured")

    # JAX status
    print(f"JAX Available: {'' if status['jax_available'] else 'L'}")
    print(f"JAX Platforms: {status['jax_platforms']}")
    print(f"Memory Fraction: {status['memory_fraction']}")

    if status["devices"]:
        print("\nJAX Devices:")
        for i, device in enumerate(status["devices"]):
            print(f"  {i}: {device}")

    if status["gpu_available"]:
        print(" GPU acceleration available")
    else:
        print("=� Using CPU-only mode")

    print("=" * 40)


def configure_for_vi_jax(memory_fraction: float = 0.8, enable_x64: bool = False):
    """
    Configure environment specifically for VI+JAX workloads.

    Args:
        memory_fraction: GPU memory fraction to use
        enable_x64: Whether to enable 64-bit precision
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    os.environ["JAX_ENABLE_X64"] = str(enable_x64).lower()

    # VI-specific optimizations
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_triton_softmax_fusion=true" not in xla_flags:
        xla_flags += " --xla_gpu_enable_triton_softmax_fusion=true"
    if "--xla_gpu_enable_latency_hiding_scheduler=true" not in xla_flags:
        xla_flags += " --xla_gpu_enable_latency_hiding_scheduler=true"

    os.environ["XLA_FLAGS"] = xla_flags.strip()


def configure_for_mcmc_jax(memory_fraction: float = 0.7, enable_x64: bool = True):
    """
    Configure environment specifically for MCMC+JAX workloads.

    Args:
        memory_fraction: GPU memory fraction to use (lower for MCMC chains)
        enable_x64: Whether to enable 64-bit precision (recommended for MCMC)
    """
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
    os.environ["JAX_ENABLE_X64"] = str(enable_x64).lower()

    # MCMC-specific optimizations (more conservative)
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_enable_async_collectives=true" not in xla_flags:
        xla_flags += " --xla_gpu_enable_async_collectives=true"

    os.environ["XLA_FLAGS"] = xla_flags.strip()


# Convenience functions
def ensure_gpu_ready(method: str = "vi") -> bool:
    """
    Ensure GPU is ready for the specified method.

    Args:
        method: "vi" for VI+JAX or "mcmc" for MCMC+JAX

    Returns:
        bool: True if GPU is ready
    """
    if not activate_gpu(quiet=True):
        return False

    if method.lower() == "vi":
        configure_for_vi_jax()
    elif method.lower() == "mcmc":
        configure_for_mcmc_jax()

    return True


def get_recommended_settings() -> dict[str, Any]:
    """Get recommended settings based on available hardware."""
    try:
        from .optimizer import GPUOptimizer

        optimizer = GPUOptimizer()

        if not optimizer.load_optimization_cache():
            optimizer.detect_gpu_hardware()
            optimizer.determine_optimal_settings()
            optimizer.save_optimization_cache()

        return optimizer.optimal_settings
    except ImportError:
        return {
            "use_gpu": False,
            "vi_batch_size": 1000,
            "mcmc_batch_size": 500,
            "memory_fraction": 0.8,
        }


# Export main functions
__all__ = [
    "setup_gpu_environment",
    "activate_gpu",
    "get_gpu_status",
    "print_gpu_status",
    "configure_for_vi_jax",
    "configure_for_mcmc_jax",
    "ensure_gpu_ready",
    "get_recommended_settings",
]
