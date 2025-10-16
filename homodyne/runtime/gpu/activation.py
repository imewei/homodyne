"""
GPU Activation and Configuration for JAX
=========================================

This module provides GPU setup, validation, and optimization for JAX-based
homodyne computations on NVIDIA GPUs with CUDA 12.1+.

Features:
- Automatic GPU detection and validation
- Memory management and allocation strategies
- Performance optimization settings
- Multi-GPU support
- Graceful CPU fallback
"""

import os
import subprocess
import warnings
from typing import Any

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import default_backend

    JAX_AVAILABLE = True

    # Access config properly for newer JAX versions
    try:
        from jax.config import config as jax_config
    except ImportError:
        # For JAX 0.4+, config is accessed differently
        jax_config = jax.config
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None
    warnings.warn(
        "JAX not available. GPU activation requires JAX installation.", stacklevel=2
    )

# Optional imports
try:
    import pynvml

    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUActivator:
    """GPU activation and configuration for JAX computations."""

    def __init__(self, verbose: bool = True):
        """
        Initialize GPU activator.

        Parameters
        ----------
        verbose : bool
            Whether to print status messages
        """
        self.verbose = verbose
        self.gpu_info = {}
        self.is_activated = False
        self.cuda_version = None
        self.driver_version = None

    def activate(
        self,
        memory_fraction: float = 0.9,
        force_gpu: bool = False,
        gpu_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Activate and configure GPU for JAX.

        Parameters
        ----------
        memory_fraction : float
            Fraction of GPU memory to allocate (0.0-1.0)
        force_gpu : bool
            If True, fail if GPU is not available
        gpu_id : int, optional
            Specific GPU ID to use (for multi-GPU systems)

        Returns
        -------
        Dict[str, Any]
            GPU configuration and status information
        """
        if not JAX_AVAILABLE:
            if force_gpu:
                raise RuntimeError(
                    "JAX is required for GPU activation. Install with: pip install jax[cuda12-local]"
                )
            return {"status": "failed", "reason": "JAX not installed"}

        # Check CUDA availability
        self._check_cuda_installation()

        # Detect GPUs
        gpu_devices = self._detect_gpus()

        if not gpu_devices:
            if force_gpu:
                raise RuntimeError(
                    "No GPU devices found. Check CUDA installation and drivers."
                )
            if self.verbose:
                print("⚠️ No GPU detected, using CPU backend")
            return {"status": "cpu_fallback", "devices": ["cpu"]}

        # Select GPU
        if gpu_id is not None:
            if gpu_id >= len(gpu_devices):
                raise ValueError(
                    f"GPU {gpu_id} not found. Available: 0-{len(gpu_devices) - 1}"
                )
            selected_device = gpu_devices[gpu_id]
        else:
            selected_device = gpu_devices[0]

        # Configure memory allocation
        self._configure_memory(memory_fraction)

        # Set performance options
        self._configure_performance()

        # Validate setup
        validation = self._validate_gpu_setup(selected_device)

        if validation["success"]:
            self.is_activated = True
            if self.verbose:
                self._print_activation_summary(validation)

        return validation

    def _check_cuda_installation(self) -> None:
        """Check CUDA installation and version."""
        try:
            # Check CUDA version
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "release" in line:
                        self.cuda_version = (
                            line.split("release")[-1].strip().split(",")[0]
                        )
                        break

            # Check driver version
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                self.driver_version = result.stdout.strip()

        except FileNotFoundError:
            if self.verbose:
                print("⚠️ CUDA tools not found in PATH")

    def _detect_gpus(self) -> list:
        """Detect available GPU devices."""
        try:
            gpu_devices = jax.devices("gpu")
            return gpu_devices
        except (RuntimeError, ValueError, AttributeError):
            return []

    def _configure_memory(self, memory_fraction: float) -> None:
        """
        Configure GPU memory allocation.

        Parameters
        ----------
        memory_fraction : float
            Fraction of GPU memory to allocate
        """
        if not 0.0 < memory_fraction <= 1.0:
            raise ValueError(
                f"memory_fraction must be in (0, 1], got {memory_fraction}"
            )

        # Set XLA memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

        # Configure JAX memory preallocation
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

        if self.verbose:
            print(f"📊 GPU memory allocation: {memory_fraction * 100:.0f}%")

    def _configure_performance(self) -> None:
        """Configure JAX/XLA performance settings."""
        # Enable persistent compilation cache
        if JAX_AVAILABLE:
            try:
                # Try different ways to set config for different JAX versions
                if hasattr(jax_config, "update"):
                    jax_config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
                    jax_config.update("jax_persistent_cache_min_compile_time_secs", 0)
                else:
                    # For newer JAX versions, use environment variables
                    os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
            except (AttributeError, KeyError, ValueError):
                # Fallback to environment variables
                os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

        # Performance flags
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_enable_triton_softmax_fusion=true "
            "--xla_gpu_triton_gemm_any=true "
            "--xla_gpu_enable_async_collectives=true "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
            "--xla_gpu_enable_highest_priority_async_stream=true"
        )

        # CUDA-specific optimizations
        os.environ["TF_CUDNN_DETERMINISM"] = "0"  # Faster non-deterministic ops
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"

        # NCCL settings for multi-GPU
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    def _validate_gpu_setup(self, device) -> dict[str, Any]:
        """
        Validate GPU setup with test computation.

        Parameters
        ----------
        device : Device
            JAX device to validate

        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        validation = {
            "success": False,
            "device": str(device),
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "backend": default_backend(),
            "gpu_info": {},
        }

        try:
            # Test computation
            test_array = jnp.ones((1000, 1000))
            result = jnp.dot(test_array, test_array)
            result.block_until_ready()

            # Get GPU info if pynvml available
            if PYNVML_AVAILABLE:
                validation["gpu_info"] = self._get_gpu_info()

            validation["success"] = True
            validation["test_passed"] = True

        except Exception as e:
            validation["error"] = str(e)

        return validation

    def _get_gpu_info(self) -> dict[str, Any]:
        """Get detailed GPU information using pynvml."""
        if not PYNVML_AVAILABLE:
            return {}

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            info = {}
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                info[f"gpu_{i}"] = {
                    "name": pynvml.nvmlDeviceGetName(handle).decode(),
                    "memory_total": pynvml.nvmlDeviceGetMemoryInfo(handle).total
                    // (1024**2),  # MB
                    "memory_free": pynvml.nvmlDeviceGetMemoryInfo(handle).free
                    // (1024**2),  # MB
                    "temperature": pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    "power": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,  # Watts
                }

            pynvml.nvmlShutdown()
            return info

        except Exception as e:
            return {"error": str(e)}

    def _print_activation_summary(self, validation: dict[str, Any]) -> None:
        """Print GPU activation summary."""
        print("\n" + "=" * 50)
        print("🚀 GPU Activation Successful!")
        print("=" * 50)
        print(f"✅ Device: {validation['device']}")
        print(f"✅ Backend: {validation['backend']}")
        print(f"✅ CUDA Version: {validation['cuda_version']}")
        print(f"✅ Driver Version: {validation['driver_version']}")

        if validation.get("gpu_info"):
            for gpu_id, info in validation["gpu_info"].items():
                if isinstance(info, dict) and "name" in info:
                    print(f"\n📊 {gpu_id.upper()} Status:")
                    print(f"  • Model: {info['name']}")
                    print(
                        f"  • Memory: {info['memory_free']}/{info['memory_total']} MB free"
                    )
                    print(f"  • Temperature: {info['temperature']}°C")
                    print(f"  • Power: {info['power']:.1f}W")

        print("=" * 50 + "\n")

    def deactivate(self) -> None:
        """Deactivate GPU and cleanup resources."""
        # Clear memory
        if JAX_AVAILABLE and jax is not None:
            try:
                # Try to clear memory if method exists
                for device in jax.devices("gpu"):
                    if hasattr(device, "_clear_memory"):
                        device._clear_memory()
            except (RuntimeError, AttributeError, ValueError):
                # If clear_memory doesn't exist or fails, just pass
                pass

        # Reset environment variables
        env_vars = [
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_FLAGS",
        ]
        for var in env_vars:
            os.environ.pop(var, None)

        self.is_activated = False
        if self.verbose:
            print("🔌 GPU deactivated")


def activate_gpu(
    memory_fraction: float = 0.9,
    force_gpu: bool = False,
    gpu_id: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to activate GPU for JAX.

    Parameters
    ----------
    memory_fraction : float
        Fraction of GPU memory to allocate (0.0-1.0)
    force_gpu : bool
        If True, fail if GPU is not available
    gpu_id : int, optional
        Specific GPU ID to use
    verbose : bool
        Whether to print status messages

    Returns
    -------
    Dict[str, Any]
        GPU configuration and status

    Examples
    --------
    >>> # Basic activation
    >>> status = activate_gpu()

    >>> # Force GPU with 80% memory
    >>> status = activate_gpu(memory_fraction=0.8, force_gpu=True)

    >>> # Use specific GPU
    >>> status = activate_gpu(gpu_id=1)
    """
    activator = GPUActivator(verbose=verbose)
    return activator.activate(memory_fraction, force_gpu, gpu_id)


def get_gpu_status() -> dict[str, Any]:
    """
    Get current GPU status without activation.

    Returns
    -------
    Dict[str, Any]
        Current GPU status information
    """
    status = {
        "jax_available": JAX_AVAILABLE,
        "devices": [],
        "cuda_version": None,
        "driver_version": None,
    }

    if JAX_AVAILABLE:
        try:
            status["devices"] = [str(d) for d in jax.devices()]
            status["backend"] = default_backend()
        except (RuntimeError, AttributeError, ValueError):
            pass

    # Get CUDA info
    try:
        result = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line:
                    status["cuda_version"] = (
                        line.split("release")[-1].strip().split(",")[0]
                    )
                    break

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            status["driver_version"] = result.stdout.strip()

    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    return status


def benchmark_gpu() -> dict[str, float]:
    """
    Run GPU benchmark tests.

    Returns
    -------
    Dict[str, float]
        Benchmark results (operations per second)
    """
    if not JAX_AVAILABLE or not jax.devices("gpu"):
        return {"error": "GPU not available"}

    results = {}

    # Matrix multiplication benchmark
    sizes = [1000, 2000, 4000]
    for size in sizes:
        import time

        # Warmup
        A = jnp.ones((size, size))
        B = jnp.ones((size, size))
        C = jnp.dot(A, B)
        C.block_until_ready()

        # Benchmark
        start = time.perf_counter()
        for _ in range(10):
            C = jnp.dot(A, B)
            C.block_until_ready()
        elapsed = time.perf_counter() - start

        gflops = (2 * size**3 * 10) / (elapsed * 1e9)
        results[f"matmul_{size}x{size}_gflops"] = gflops

    # FFT benchmark
    for size in [1024, 4096, 16384]:
        data = jnp.ones(size, dtype=jnp.complex64)

        # Warmup
        result = jnp.fft.fft(data)
        result.block_until_ready()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            result = jnp.fft.fft(data)
            result.block_until_ready()
        elapsed = time.perf_counter() - start

        ops_per_sec = 100 / elapsed
        results[f"fft_{size}_ops_per_sec"] = ops_per_sec

    return results


# Module-level activation for convenience
_gpu_activator = None


def get_activator() -> GPUActivator:
    """Get or create the global GPU activator instance."""
    global _gpu_activator
    if _gpu_activator is None:
        _gpu_activator = GPUActivator()
    return _gpu_activator
