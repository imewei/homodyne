"""
GPU Runtime Support for Homodyne v2
===================================

GPU acceleration and optimization support for JAX-based computations.
Implements JAX-specific optimizations for enhanced performance on NVIDIA GPUs.

This module provides:
- GPU detection and activation
- JAX device management
- Memory optimization
- Performance benchmarking
- CUDA 12.1+ support
"""

# Import new GPU activation module
try:
    from homodyne.runtime.gpu.activation import (
        GPUActivator,
        activate_gpu,
        benchmark_gpu,
        get_activator,
        get_gpu_status,
    )

    GPU_AVAILABLE = True
except ImportError as e:
    GPU_AVAILABLE = False
    _import_error = str(e)

    # Stub functions for graceful fallback
    def activate_gpu(*args, **kwargs):
        raise ImportError(f"GPU activation not available: {_import_error}")

    def get_gpu_status():
        return {"available": False, "error": _import_error}

    def benchmark_gpu():
        raise ImportError(f"GPU benchmark not available: {_import_error}")

    class GPUActivator:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"GPUActivator not available: {_import_error}")


# Legacy imports for compatibility
try:
    from homodyne.runtime.gpu.optimizer import GPUOptimizer

    OPTIMIZER_AVAILABLE = True
except ImportError:
    GPUOptimizer = None
    OPTIMIZER_AVAILABLE = False

try:
    from homodyne.runtime.gpu.wrapper import setup_gpu_environment

    WRAPPER_AVAILABLE = True
except ImportError:
    setup_gpu_environment = None
    WRAPPER_AVAILABLE = False

__all__ = [
    # New primary exports
    "GPUActivator",
    "activate_gpu",
    "get_gpu_status",
    "benchmark_gpu",
    "get_activator",
    "GPU_AVAILABLE",
    # Legacy exports
    "GPUOptimizer",
    "setup_gpu_environment",
    "OPTIMIZER_AVAILABLE",
    "WRAPPER_AVAILABLE",
]
