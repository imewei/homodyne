"""
GPU Runtime Support for Homodyne v2
===================================

GPU acceleration and optimization support for VI+JAX and MCMC+JAX methods.
Implements JAX-specific optimizations for enhanced performance.

This module provides:
- GPU detection and benchmarking
- JAX device management
- Memory optimization
- Performance tuning
"""

# Handle imports with graceful fallback
try:
    from homodyne.runtime.gpu.optimizer import GPUOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    GPUOptimizer = None
    OPTIMIZER_AVAILABLE = False

try:
    from homodyne.runtime.gpu.wrapper import activate_gpu, setup_gpu_environment
    WRAPPER_AVAILABLE = True
except ImportError:
    activate_gpu = None
    setup_gpu_environment = None
    WRAPPER_AVAILABLE = False

__all__ = [
    "GPUOptimizer",
    "activate_gpu",
    "setup_gpu_environment",
    "OPTIMIZER_AVAILABLE",
    "WRAPPER_AVAILABLE",
]