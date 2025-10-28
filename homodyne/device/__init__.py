"""Device Optimization Module for Homodyne v2
===========================================

HPC/GPU device optimization with system CUDA integration.
Provides intelligent device detection, configuration, and optimization
for high-performance computing environments.

Key Features:
- Automatic device detection and optimal configuration
- HPC CPU optimization for 36/128-core nodes
- System CUDA integration with jax[local]
- Graceful fallback from GPU to CPU
- Performance benchmarking and optimization

Usage:
    from homodyne.device import configure_optimal_device
    config = configure_optimal_device()
"""

import logging
import os

# Suppress JAX TPU initialization warnings before importing JAX
# TPU backend is not available on standard systems and creates noise
# IMPORTANT: Don't set JAX_PLATFORMS - let JAX auto-select optimal backend

# Suppress JAX INFO-level logs for backend initialization
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logging.getLogger("jax._src.compiler").setLevel(logging.WARNING)

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Import device-specific modules with fallback
try:
    from homodyne.device.cpu import (
        benchmark_cpu_performance,
        configure_cpu_hpc,
        detect_cpu_info,
        get_optimal_batch_size,
    )

    HAS_CPU_MODULE = True
except ImportError as e:
    logger.warning(f"CPU optimization module not available: {e}")
    HAS_CPU_MODULE = False

try:
    from homodyne.device.gpu import (
        benchmark_gpu_performance,
        configure_system_cuda,
        detect_system_cuda,
        get_gpu_memory_info,
        optimize_gpu_memory,
        validate_cuda_installation,
    )

    HAS_GPU_MODULE = True
except ImportError as e:
    logger.warning(f"GPU optimization module not available: {e}")
    HAS_GPU_MODULE = False


def configure_optimal_device(
    prefer_gpu: bool = True,
    gpu_memory_fraction: float = 0.9,
    cpu_threads: int | None = None,
    force_cpu: bool = False,
) -> dict[str, any]:
    """Automatically configure the optimal device for homodyne analysis.

    Attempts GPU configuration first (if available), then falls back to
    optimized CPU configuration for HPC environments.

    Parameters
    ----------
    prefer_gpu : bool, default True
        Whether to prefer GPU over CPU when both are available
    gpu_memory_fraction : float, default 0.9
        Fraction of GPU memory to allocate
    cpu_threads : int, optional
        Number of CPU threads to use. If None, auto-detects optimal count.
    force_cpu : bool, default False
        Force CPU-only mode, skip GPU detection

    Returns
    -------
    dict
        Device configuration summary with performance hints
    """
    logger.info("Configuring optimal device for homodyne analysis")

    config_result = {
        "device_type": "unknown",
        "configuration_successful": False,
        "performance_ready": False,
        "recommendations": [],
        "warnings": [],
        "device_info": {},
    }

    # Force CPU mode if requested
    if force_cpu:
        logger.info("CPU-only mode requested, skipping GPU detection")
        return _configure_cpu_optimal(config_result, cpu_threads)

    # Try GPU configuration first (if preferred and available)
    if prefer_gpu and HAS_GPU_MODULE:
        try:
            logger.info("Attempting GPU configuration...")
            gpu_config = configure_system_cuda(memory_fraction=gpu_memory_fraction)

            if gpu_config["cuda_configured"]:
                config_result.update(
                    {
                        "device_type": "gpu",
                        "configuration_successful": True,
                        "performance_ready": True,
                        "device_info": gpu_config,
                        "recommendations": [
                            "GPU acceleration configured successfully",
                            f"Using {gpu_memory_fraction:.0%} of GPU memory",
                        ],
                    },
                )

                logger.info("✓ GPU configuration successful")
                return config_result

            else:
                logger.info("GPU configuration failed, falling back to CPU")
                if "error" in gpu_config:
                    config_result["warnings"].append(
                        f"GPU setup failed: {gpu_config['error']}",
                    )

        except Exception as e:
            logger.warning(f"GPU configuration attempt failed: {e}")
            config_result["warnings"].append(f"GPU configuration error: {e}")

    # Configure CPU (either as fallback or primary choice)
    return _configure_cpu_optimal(config_result, cpu_threads)


def _configure_cpu_optimal(config_result: dict, cpu_threads: int | None) -> dict:
    """Configure optimal CPU settings."""
    logger.info("Configuring CPU optimization...")

    try:
        if HAS_CPU_MODULE:
            cpu_config = configure_cpu_hpc(
                num_threads=cpu_threads,
                enable_hyperthreading=False,  # Usually better for HPC
                numa_policy="auto",
                memory_optimization="standard",
            )

            config_result.update(
                {
                    "device_type": "cpu",
                    "configuration_successful": True,
                    "performance_ready": True,
                    "device_info": cpu_config,
                    "recommendations": [
                        f"CPU optimization configured for {cpu_config['threads_configured']} threads",
                        "Consider GPU acceleration for larger datasets",
                    ],
                },
            )

            logger.info(
                f"✓ CPU configuration successful with {cpu_config['threads_configured']} threads",
            )

        else:
            # Minimal CPU configuration if module not available
            import multiprocessing
            import os

            num_cores = multiprocessing.cpu_count()
            os.environ["OMP_NUM_THREADS"] = str(num_cores)
            os.environ["JAX_PLATFORM_NAME"] = "cpu"

            config_result.update(
                {
                    "device_type": "cpu",
                    "configuration_successful": True,
                    "performance_ready": False,
                    "recommendations": [
                        f"Basic CPU configuration with {num_cores} cores",
                        "Install psutil for advanced CPU optimization",
                    ],
                },
            )

            logger.info("✓ Basic CPU configuration completed")

    except Exception as e:
        logger.error(f"CPU configuration failed: {e}")
        config_result.update(
            {
                "device_type": "cpu",
                "configuration_successful": False,
                "performance_ready": False,
                "warnings": [f"CPU configuration failed: {e}"],
            },
        )

    return config_result


def get_device_status() -> dict[str, any]:
    """Get current device status and capabilities.

    Returns
    -------
    dict
        Comprehensive device status information
    """
    status = {
        "timestamp": None,
        "cpu_info": {},
        "gpu_info": {},
        "recommendations": [],
        "performance_estimate": "unknown",
    }

    try:
        import datetime

        status["timestamp"] = datetime.datetime.now().isoformat()

        # Get CPU information
        if HAS_CPU_MODULE:
            status["cpu_info"] = detect_cpu_info()
        else:
            import multiprocessing

            status["cpu_info"] = {
                "logical_cores": multiprocessing.cpu_count(),
                "optimization_available": False,
            }

        # Get GPU information
        if HAS_GPU_MODULE:
            status["gpu_info"] = detect_system_cuda()
        else:
            status["gpu_info"] = {
                "cuda_available": False,
                "optimization_available": False,
            }

        # Generate performance estimate
        if status["gpu_info"].get("cuda_available", False):
            status["performance_estimate"] = "high"
            status["recommendations"].append(
                "GPU acceleration available - excellent performance expected",
            )
        elif status["cpu_info"].get("physical_cores", 0) >= 16:
            status["performance_estimate"] = "medium-high"
            status["recommendations"].append(
                "Multi-core CPU detected - good performance expected",
            )
        else:
            status["performance_estimate"] = "medium"
            status["recommendations"].append(
                "Consider upgrading hardware for better performance",
            )

    except Exception as e:
        logger.error(f"Device status check failed: {e}")
        status["error"] = str(e)

    return status


def is_gpu_active() -> bool:
    """Check if JAX is currently configured to use GPU.

    Returns
    -------
    bool
        True if GPU is active, False otherwise
    """
    try:
        import jax

        devices = jax.devices()
        return len(devices) > 0 and devices[0].platform == "gpu"
    except Exception:
        return False


def switch_to_cpu(num_threads: int | None = None) -> dict[str, any]:
    """Dynamically switch JAX from GPU to CPU execution.

    This function reconfigures JAX to use CPU instead of GPU, useful for
    recovering from GPU out-of-memory errors during optimization.

    Parameters
    ----------
    num_threads : int, optional
        Number of CPU threads to use. If None, auto-detects optimal count.

    Returns
    -------
    dict
        CPU configuration result

    Notes
    -----
    This function should be called before retrying failed GPU operations.
    JAX device allocation is persistent within a Python session, so this
    provides a way to switch devices without restarting.
    """
    logger.info("Switching from GPU to CPU execution...")

    try:
        import os

        import jax

        # Force JAX to use CPU
        jax.config.update("jax_platform_name", "cpu")

        # Configure CPU threading
        if num_threads is None:
            import multiprocessing

            num_threads = multiprocessing.cpu_count()

        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={num_threads}"
        )

        # Verify switch
        devices = jax.devices()
        if devices[0].platform == "cpu":
            logger.info(
                f"✓ Successfully switched to CPU with {num_threads} threads",
            )
            return {
                "success": True,
                "device_type": "cpu",
                "num_threads": num_threads,
                "devices": str(devices),
            }
        else:
            logger.warning(
                f"CPU switch may not be complete - device platform: {devices[0].platform}",
            )
            return {
                "success": False,
                "device_type": devices[0].platform,
                "warning": "Switch initiated but device still shows as GPU",
            }

    except Exception as e:
        logger.error(f"Failed to switch to CPU: {e}")
        return {"success": False, "error": str(e)}


def benchmark_device_performance(
    device_type: str | None = None,
    test_size: int = 5000,
) -> dict[str, any]:
    """Benchmark device performance for optimization planning.

    Parameters
    ----------
    device_type : str, optional
        Device type to benchmark ('cpu', 'gpu', or None for auto-detect)
    test_size : int, default 5000
        Size of benchmark computation

    Returns
    -------
    dict
        Benchmark results with performance metrics
    """
    logger.info(f"Benchmarking device performance (test_size={test_size})")

    benchmark_results = {
        "device_type": device_type or "auto",
        "test_size": test_size,
        "results": {},
        "recommendations": [],
    }

    try:
        # Benchmark CPU if available or requested
        if (device_type in [None, "cpu"]) and HAS_CPU_MODULE:
            logger.info("Running CPU benchmark...")
            cpu_results = benchmark_cpu_performance(test_size=test_size)
            benchmark_results["results"]["cpu"] = cpu_results

        # Benchmark GPU if available or requested
        if (device_type in [None, "gpu"]) and HAS_GPU_MODULE:
            logger.info("Running GPU benchmark...")
            gpu_results = benchmark_gpu_performance(test_sizes=[test_size])
            benchmark_results["results"]["gpu"] = gpu_results

        # Generate recommendations based on results
        if (
            "gpu" in benchmark_results["results"]
            and "cpu" in benchmark_results["results"]
        ):
            gpu_time = (
                benchmark_results["results"]["gpu"]
                .get("test_results", {})
                .get(test_size, {})
                .get("mean_time", float("inf"))
            )
            cpu_time = benchmark_results["results"]["cpu"].get(
                "numpy_mean_time",
                float("inf"),
            )

            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time if gpu_time > 0 else 1
                benchmark_results["recommendations"].append(
                    f"GPU acceleration provides {speedup:.1f}x speedup over CPU",
                )
            else:
                benchmark_results["recommendations"].append(
                    "CPU performance competitive with GPU for this problem size",
                )

    except Exception as e:
        logger.error(f"Device benchmarking failed: {e}")
        benchmark_results["error"] = str(e)

    return benchmark_results


# Main exports
__all__ = [
    # Primary device configuration
    "configure_optimal_device",
    # Device information
    "get_device_status",
    "benchmark_device_performance",
    # Device switching utilities
    "is_gpu_active",
    "switch_to_cpu",
    # CPU-specific (if available)
    "configure_cpu_hpc" if HAS_CPU_MODULE else None,
    "detect_cpu_info" if HAS_CPU_MODULE else None,
    # GPU-specific (if available)
    "configure_system_cuda" if HAS_GPU_MODULE else None,
    "detect_system_cuda" if HAS_GPU_MODULE else None,
    "validate_cuda_installation" if HAS_GPU_MODULE else None,
    # Status flags
    "HAS_CPU_MODULE",
    "HAS_GPU_MODULE",
]

# Remove None values from __all__
__all__ = [item for item in __all__ if item is not None]
