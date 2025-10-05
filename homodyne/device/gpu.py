"""
System CUDA Integration for Homodyne v2
=======================================

GPU acceleration using system CUDA installation with jax[local].
Provides CUDA detection, configuration, and optimization for HPC/supercomputer
environments where CUDA is pre-installed system-wide.

Key Features:
- System CUDA detection and version validation
- JAX configuration for system CUDA libraries
- GPU memory management and optimization
- Multi-GPU support for HPC clusters
- Graceful fallback to CPU when GPU unavailable

System CUDA Benefits:
- Uses existing HPC CUDA installations
- Avoids version conflicts with system libraries
- Leverages optimized system CUDA configurations
- Compatible with HPC module systems
"""

import os
import subprocess

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None


def detect_system_cuda() -> dict[str, any]:
    """
    Detect system CUDA installation and capabilities.

    Returns
    -------
    dict
        CUDA system information including version, devices, and capabilities
    """
    cuda_info = {
        "cuda_available": False,
        "cuda_version": None,
        "driver_version": None,
        "devices": [],
        "device_count": 0,
        "memory_total": 0,
        "compute_capability": None,
        "cuda_home": None,
        "nvcc_available": False,
        "jax_cuda_compatible": False,
    }

    try:
        # Check for CUDA installation
        cuda_info["cuda_home"] = os.environ.get("CUDA_HOME") or os.environ.get(
            "CUDA_PATH"
        )

        # Try to detect CUDA via nvcc
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                cuda_info["nvcc_available"] = True
                output = result.stdout

                # Parse CUDA version
                for line in output.split("\n"):
                    if "release" in line and "V" in line:
                        version_part = line.split("V")[1].split(",")[0]
                        cuda_info["cuda_version"] = version_part.strip()
                        break

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.debug("nvcc not found or not accessible")

        # Try to detect via nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                cuda_info["cuda_available"] = True
                lines = result.stdout.strip().split("\n")

                for i, line in enumerate(lines):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            cuda_info["devices"].append(
                                {
                                    "id": i,
                                    "name": parts[0],
                                    "memory_mb": (
                                        int(parts[1]) if parts[1].isdigit() else 0
                                    ),
                                    "compute_capability": parts[2],
                                }
                            )

                cuda_info["device_count"] = len(cuda_info["devices"])
                if cuda_info["devices"]:
                    cuda_info["memory_total"] = sum(
                        d["memory_mb"] for d in cuda_info["devices"]
                    )
                    cuda_info["compute_capability"] = cuda_info["devices"][0][
                        "compute_capability"
                    ]

        except (
            subprocess.SubprocessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            logger.debug("nvidia-smi not found or not accessible")

        # Check JAX CUDA compatibility
        if JAX_AVAILABLE and cuda_info["cuda_available"]:
            try:
                # Test if JAX can detect CUDA devices
                os.environ["JAX_PLATFORM_NAME"] = "gpu"
                devices = jax.devices("gpu")
                if devices:
                    cuda_info["jax_cuda_compatible"] = True
                    logger.info(f"JAX detected {len(devices)} CUDA device(s)")
            except Exception as e:
                logger.debug(f"JAX CUDA detection failed: {e}")
                # Reset to CPU if GPU detection fails
                os.environ["JAX_PLATFORM_NAME"] = "cpu"

    except Exception as e:
        logger.warning(f"CUDA detection failed: {e}")

    return cuda_info


def configure_system_cuda(
    device_id: int | None = None,
    memory_fraction: float = 0.9,
    enable_preallocation: bool = False,
) -> dict[str, any]:
    """
    Configure JAX to use system CUDA installation.

    Parameters
    ----------
    device_id : int, optional
        Specific GPU device ID to use. If None, uses default device.
    memory_fraction : float, default 0.9
        Fraction of GPU memory to allocate (0.1 to 1.0)
    enable_preallocation : bool, default False
        Whether to preallocate GPU memory

    Returns
    -------
    dict
        Configuration summary and GPU status
    """
    logger.info("Configuring system CUDA for JAX")

    config_summary = {
        "cuda_configured": False,
        "fallback_to_cpu": True,
        "device_info": None,
        "memory_config": None,
        "error": None,
    }

    try:
        # Detect system CUDA
        cuda_info = detect_system_cuda()

        if not cuda_info["cuda_available"]:
            logger.warning("No CUDA devices detected, falling back to CPU")
            return _configure_cpu_fallback(config_summary)

        if not JAX_AVAILABLE:
            logger.error("JAX not available, cannot configure CUDA")
            config_summary["error"] = "JAX not available"
            return config_summary

        # Configure JAX for GPU
        os.environ["JAX_PLATFORM_NAME"] = "gpu"

        # Set memory fraction
        memory_fraction = max(0.1, min(1.0, memory_fraction))
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)

        if enable_preallocation:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
        else:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

        # Try to initialize JAX with GPU
        try:
            devices = jax.devices("gpu")
            if not devices:
                raise RuntimeError("No JAX GPU devices available")

            # Select specific device if requested
            if device_id is not None:
                if device_id < len(devices):
                    selected_device = devices[device_id]
                    jax.config.update("jax_default_device", selected_device)
                    logger.info(f"Using GPU device {device_id}: {selected_device}")
                else:
                    logger.warning(
                        f"Device ID {device_id} not available, using default"
                    )

            # Verify GPU functionality with a simple test
            _test_gpu_functionality()

            config_summary.update(
                {
                    "cuda_configured": True,
                    "fallback_to_cpu": False,
                    "device_info": cuda_info,
                    "memory_config": {
                        "memory_fraction": memory_fraction,
                        "preallocation_enabled": enable_preallocation,
                    },
                    "active_devices": len(devices),
                    "default_device": str(jax.devices()[0]),
                }
            )

            logger.info(
                f"✓ System CUDA configured successfully with {len(devices)} device(s)"
            )
            logger.info(
                f"Memory fraction: {memory_fraction:.1%}, "
                f"Preallocation: {'enabled' if enable_preallocation else 'disabled'}"
            )

        except Exception as e:
            logger.error(f"JAX GPU initialization failed: {e}")
            return _configure_cpu_fallback(config_summary, error=str(e))

    except Exception as e:
        logger.error(f"System CUDA configuration failed: {e}")
        config_summary["error"] = str(e)
        return _configure_cpu_fallback(config_summary, error=str(e))

    return config_summary


def _configure_cpu_fallback(config_summary: dict, error: str | None = None) -> dict:
    """Configure CPU fallback when GPU is unavailable."""
    logger.info("Configuring CPU fallback")

    try:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        if JAX_AVAILABLE:
            jax.config.update("jax_default_device", jax.devices("cpu")[0])

        config_summary.update(
            {"cuda_configured": False, "fallback_to_cpu": True, "cpu_configured": True}
        )

        if error:
            config_summary["error"] = error

        logger.info("✓ CPU fallback configured successfully")

    except Exception as e:
        logger.error(f"CPU fallback configuration failed: {e}")
        config_summary["error"] = f"CPU fallback failed: {e}"

    return config_summary


def _test_gpu_functionality() -> None:
    """Test basic GPU functionality with JAX."""
    if not JAX_AVAILABLE:
        return

    try:
        # Simple computation test
        @jax.jit
        def test_computation(x):
            return jnp.sum(x**2)

        test_array = jnp.ones((1000, 1000))
        result = test_computation(test_array)
        result.block_until_ready()  # Ensure computation completes

        logger.debug("GPU functionality test passed")

    except Exception as e:
        raise RuntimeError(f"GPU functionality test failed: {e}") from e


def get_gpu_memory_info() -> dict[str, any]:
    """
    Get current GPU memory usage information.

    Returns
    -------
    dict
        Memory usage information for all GPUs
    """
    memory_info = {
        "total_memory_mb": 0,
        "available_memory_mb": 0,
        "used_memory_mb": 0,
        "devices": [],
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")

            for i, line in enumerate(lines):
                if line.strip():
                    parts = [int(p.strip()) for p in line.split(",")]
                    if len(parts) >= 3:
                        total, used, free = parts
                        memory_info["devices"].append(
                            {
                                "device_id": i,
                                "total_mb": total,
                                "used_mb": used,
                                "free_mb": free,
                                "usage_percent": (
                                    (used / total) * 100 if total > 0 else 0
                                ),
                            }
                        )

            # Calculate totals
            memory_info["total_memory_mb"] = sum(
                d["total_mb"] for d in memory_info["devices"]
            )
            memory_info["used_memory_mb"] = sum(
                d["used_mb"] for d in memory_info["devices"]
            )
            memory_info["available_memory_mb"] = sum(
                d["free_mb"] for d in memory_info["devices"]
            )

    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Could not query GPU memory information")

    return memory_info


def optimize_gpu_memory(
    data_size: int,
    available_memory_mb: int | None = None,
    safety_factor: float = 0.8,
) -> dict[str, int]:
    """
    Calculate optimal memory usage strategy for GPU processing.

    Parameters
    ----------
    data_size : int
        Size of data to process
    available_memory_mb : int, optional
        Available GPU memory in MB. If None, auto-detects.
    safety_factor : float, default 0.8
        Safety factor for memory allocation (0.1 to 1.0)

    Returns
    -------
    dict
        Memory optimization strategy with batch sizes and chunks
    """
    if available_memory_mb is None:
        memory_info = get_gpu_memory_info()
        available_memory_mb = memory_info["available_memory_mb"]

    if available_memory_mb <= 0:
        logger.warning("No GPU memory information available")
        return {"batch_size": 1000, "num_chunks": max(1, data_size // 1000)}

    # Estimate memory usage per data point (heuristic)
    memory_per_point_mb = 0.01  # 10 KB per data point (conservative)
    usable_memory_mb = available_memory_mb * safety_factor

    # Calculate optimal batch size
    optimal_batch_size = int(usable_memory_mb / memory_per_point_mb)
    optimal_batch_size = max(min(optimal_batch_size, data_size), 100)

    # Calculate number of chunks needed
    num_chunks = max(1, (data_size + optimal_batch_size - 1) // optimal_batch_size)

    optimization_strategy = {
        "batch_size": optimal_batch_size,
        "num_chunks": num_chunks,
        "estimated_memory_usage_mb": optimal_batch_size * memory_per_point_mb,
        "available_memory_mb": available_memory_mb,
        "safety_factor": safety_factor,
    }

    logger.info(
        f"GPU memory optimization: batch_size={optimal_batch_size}, "
        f"chunks={num_chunks}, memory={optimization_strategy['estimated_memory_usage_mb']:.1f}MB"
    )

    return optimization_strategy


def benchmark_gpu_performance(
    test_sizes: list[int] = None, num_iterations: int = 3
) -> dict[str, any]:
    """
    Benchmark GPU performance for optimization planning.

    Parameters
    ----------
    test_sizes : list of int
        Sizes of test computations
    num_iterations : int, default 3
        Number of benchmark iterations per test size

    Returns
    -------
    dict
        Benchmark results with timing and throughput information
    """
    if test_sizes is None:
        test_sizes = [1000, 5000, 10000]
    logger.info("Running GPU performance benchmark")

    if not JAX_AVAILABLE:
        return {"error": "JAX not available for benchmarking"}

    try:
        devices = jax.devices("gpu")
        if not devices:
            return {"error": "No GPU devices available for benchmarking"}
    except (RuntimeError, ImportError, Exception) as e:
        return {"error": f"GPU not accessible for benchmarking: {e}"}

    import time

    import numpy as np

    results = {
        "device_info": str(devices[0]),
        "test_results": {},
        "peak_throughput": 0,
        "memory_bandwidth": 0,
    }

    @jax.jit
    def benchmark_computation(x):
        """Simulate typical XPCS computation on GPU."""
        y = jnp.fft.fft2(x)
        z = jnp.abs(y) ** 2
        return jnp.sum(z)

    for test_size in test_sizes:
        logger.info(f"Benchmarking with size {test_size}x{test_size}")

        # Warm up JIT compilation
        warm_up_array = jnp.ones((100, 100))
        _ = benchmark_computation(warm_up_array)

        times = []
        for _i in range(num_iterations):
            # Generate test data
            test_array = jnp.array(np.random.randn(test_size, test_size))

            start_time = time.perf_counter()
            result = benchmark_computation(test_array)
            result.block_until_ready()  # Ensure GPU computation completes
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        throughput = (test_size * test_size) / mean_time  # Operations per second

        results["test_results"][test_size] = {
            "mean_time": mean_time,
            "std_time": std_time,
            "throughput": throughput,
            "times": times,
        }

        results["peak_throughput"] = max(results["peak_throughput"], throughput)

        logger.info(
            f"Size {test_size}: {mean_time:.4f}s ± {std_time:.4f}s "
            f"({throughput:.0f} ops/sec)"
        )

    # Estimate memory bandwidth (rough calculation)
    if results["test_results"]:
        largest_test = max(results["test_results"].keys())
        largest_result = results["test_results"][largest_test]
        # Estimate: 4 bytes per float32, read + write operations
        bytes_processed = largest_test * largest_test * 4 * 2
        results["memory_bandwidth"] = (
            bytes_processed / largest_result["mean_time"] / 1e9
        )  # GB/s

    logger.info(
        f"Benchmark completed. Peak throughput: {results['peak_throughput']:.0f} ops/sec"
    )
    logger.info(f"Estimated memory bandwidth: {results['memory_bandwidth']:.1f} GB/s")

    return results


def validate_cuda_installation() -> dict[str, any]:
    """
    Validate system CUDA installation for homodyne compatibility.

    Returns
    -------
    dict
        Validation results with recommendations
    """
    logger.info("Validating system CUDA installation")

    validation = {
        "cuda_detected": False,
        "version_compatible": False,
        "jax_compatible": False,
        "performance_ready": False,
        "recommendations": [],
        "warnings": [],
        "errors": [],
    }

    try:
        # Check CUDA detection
        cuda_info = detect_system_cuda()
        validation["cuda_detected"] = cuda_info["cuda_available"]

        if not validation["cuda_detected"]:
            validation["errors"].append("No CUDA installation detected")
            validation["recommendations"].append(
                "Install CUDA toolkit or load CUDA module on HPC system"
            )
            return validation

        # Check version compatibility
        if cuda_info["cuda_version"]:
            major_version = float(cuda_info["cuda_version"].split(".")[0])
            if major_version >= 11.0:
                validation["version_compatible"] = True
            else:
                validation["warnings"].append(
                    f"CUDA {cuda_info['cuda_version']} detected. "
                    "CUDA 11.0+ recommended for optimal JAX compatibility"
                )

        # Check JAX compatibility
        if JAX_AVAILABLE:
            try:
                config_result = configure_system_cuda()
                validation["jax_compatible"] = config_result["cuda_configured"]

                if not validation["jax_compatible"]:
                    validation["errors"].append("JAX cannot access CUDA devices")
                    validation["recommendations"].append(
                        "Install jax[local] and ensure system CUDA is in PATH"
                    )
            except Exception as e:
                validation["errors"].append(f"JAX CUDA test failed: {e}")
        else:
            validation["errors"].append("JAX not available")
            validation["recommendations"].append(
                "Install JAX with: pip install jax[local]"
            )

        # Performance validation
        if validation["jax_compatible"] and cuda_info["device_count"] > 0:
            memory_info = get_gpu_memory_info()
            total_memory_gb = memory_info["total_memory_mb"] / 1024

            if total_memory_gb >= 8:
                validation["performance_ready"] = True
            else:
                validation["warnings"].append(
                    f"Only {total_memory_gb:.1f}GB GPU memory available. "
                    "8GB+ recommended for large datasets"
                )

        # Generate final recommendations
        if validation["cuda_detected"] and validation["jax_compatible"]:
            validation["recommendations"].append(
                "System CUDA configuration validated successfully"
            )
        elif validation["cuda_detected"]:
            validation["recommendations"].append(
                "CUDA detected but JAX integration needs attention"
            )

    except Exception as e:
        validation["errors"].append(f"Validation failed: {e}")

    return validation
