"""
GPU Optimizer for Homodyne v2 - VI+JAX and MCMC+JAX
====================================================

Intelligent GPU detection, benchmarking, and optimization system specifically
designed for VI+JAX and MCMC+JAX workloads with enhancements for the unified homodyne model.

Key Features:
- JAX-specific GPU optimization
- VI+JAX and MCMC+JAX workload profiling
- Memory management for large datasets
- CPU-primary, GPU-optional architecture
- Hardware-aware parameter recommendations
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# JAX imports with fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    Intelligent GPU optimization for VI+JAX and MCMC+JAX methods.

    Implements JAX-specific enhancements for
    the unified homodyne model workloads.
    """

    def __init__(self):
        self.gpu_info = {}
        self.optimal_settings = {}
        self.cache_file = (
            Path.home() / ".cache" / "homodyne" / "gpu_optimization_v2.json"
        )
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)

    def detect_gpu_hardware(self) -> Dict[str, Any]:
        """Detect GPU hardware and JAX integration capabilities."""
        info: Dict[str, Any] = {
            "available": False,
            "cuda_available": False,
            "devices": [],
            "cuda_version": None,
            "driver_version": None,
            "compute_capability": [],
            "jax_gpu_available": False,
            "jax_devices": [],
        }

        # Check for NVIDIA GPU
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                info["available"] = True
                for line in result.stdout.strip().split("\n"):
                    parts = line.split(", ")
                    if len(parts) >= 4:
                        info["devices"].append(
                            {
                                "name": parts[0],
                                "memory_mb": int(parts[1].replace(" MiB", "")),
                                "compute_capability": parts[2],
                                "driver_version": parts[3],
                            }
                        )
                        info["compute_capability"].append(parts[2])
                        info["driver_version"] = parts[3]
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass

        # Check CUDA installation
        cuda_paths = ["/usr/local/cuda", "/opt/cuda", os.environ.get("CUDA_HOME", "")]
        for cuda_path in cuda_paths:
            if cuda_path and Path(cuda_path).exists():
                info["cuda_available"] = True
                # Get CUDA version
                nvcc_path = Path(cuda_path) / "bin" / "nvcc"
                if nvcc_path.exists():
                    try:
                        result = subprocess.run(
                            [str(nvcc_path), "--version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if "release" in result.stdout:
                            for line in result.stdout.split("\n"):
                                if "release" in line:
                                    version = (
                                        line.split("release")[1].split(",")[0].strip()
                                    )
                                    info["cuda_version"] = version
                                    break
                    except subprocess.SubprocessError:
                        pass
                break

        # Check JAX GPU support
        if JAX_AVAILABLE:
            try:
                devices = jax.devices()
                info["jax_devices"] = [str(d) for d in devices]
                info["jax_gpu_available"] = any(
                    "gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices
                )
            except Exception:
                info["jax_devices"] = []
                info["jax_gpu_available"] = False
        else:
            info["jax_devices"] = []
            info["jax_gpu_available"] = False

        self.gpu_info = info
        return info

    def benchmark_jax_workloads(
        self, matrix_sizes: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark JAX operations typical for VI+JAX and MCMC+JAX workloads.

        Tests operations similar to those in the unified homodyne model:
        - Matrix operations (parameter covariance)
        - Gradient computations (VI optimization)
        - Likelihood evaluations (both VI and MCMC)
        """
        if matrix_sizes is None:
            matrix_sizes = [100, 500, 1000, 2000, 5000]  # Larger sizes for homodyne

        benchmarks: Dict[str, Dict] = {
            "matrix_operations": {},
            "gradient_computation": {},
            "likelihood_evaluation": {},
            "memory_bandwidth": {},
        }

        if not JAX_AVAILABLE or not self.gpu_info.get("jax_gpu_available"):
            logger.warning("JAX GPU not available - skipping benchmarks")
            return benchmarks

        try:
            # Get GPU device
            devices = jax.devices()
            gpu_device = next(
                (
                    d
                    for d in devices
                    if "gpu" in str(d).lower() or "cuda" in str(d).lower()
                ),
                None,
            )

            if gpu_device:
                logger.info(f"Benchmarking on device: {gpu_device}")

                for size in matrix_sizes:
                    # Test matrix operations (similar to covariance computations in VI)
                    @jit
                    def matrix_ops(x):
                        return jnp.dot(x, x.T) + jnp.eye(size) * 0.01

                    x = jax.device_put(jnp.ones((size, size)), gpu_device)

                    # Warmup
                    _ = matrix_ops(x).block_until_ready()

                    # Benchmark matrix operations
                    start = time.perf_counter()
                    for _ in range(10):
                        _ = matrix_ops(x).block_until_ready()
                    elapsed = (time.perf_counter() - start) / 10

                    benchmarks["matrix_operations"][size] = {
                        "time_ms": elapsed * 1000,
                        "gflops": (2 * size**3 + size) / (elapsed * 1e9),
                    }

                    # Test gradient computation (similar to VI ELBO gradients)
                    @jit
                    def loss_function(params):
                        return jnp.sum(params**2) + jnp.sum(jnp.sin(params))

                    grad_fn = jit(jax.grad(loss_function))
                    params = jax.device_put(jnp.ones(size), gpu_device)

                    # Warmup
                    _ = grad_fn(params).block_until_ready()

                    # Benchmark gradient computation
                    start = time.perf_counter()
                    for _ in range(10):
                        _ = grad_fn(params).block_until_ready()
                    elapsed = (time.perf_counter() - start) / 10

                    benchmarks["gradient_computation"][size] = {
                        "time_ms": elapsed * 1000,
                        "throughput": size / elapsed,  # params/second
                    }

                    # Test likelihood evaluation (similar to homodyne model)
                    @jit
                    def likelihood_eval(data, theory, sigma):
                        residuals = (data - theory) / sigma
                        return -0.5 * jnp.sum(residuals**2)

                    data = jax.device_put(jnp.ones(size), gpu_device)
                    theory = jax.device_put(jnp.ones(size) * 0.8, gpu_device)
                    sigma = jax.device_put(jnp.ones(size) * 0.1, gpu_device)

                    # Warmup
                    _ = likelihood_eval(data, theory, sigma).block_until_ready()

                    # Benchmark likelihood evaluation
                    start = time.perf_counter()
                    for _ in range(100):  # More iterations for smaller operation
                        _ = likelihood_eval(data, theory, sigma).block_until_ready()
                    elapsed = (time.perf_counter() - start) / 100

                    benchmarks["likelihood_evaluation"][size] = {
                        "time_ms": elapsed * 1000,
                        "evaluations_per_sec": 1.0 / elapsed,
                    }

                # Memory bandwidth test (important for large datasets)
                large_size = 10000  # Reduced from 20000 to avoid memory issues
                x_large = jax.device_put(
                    jnp.ones((large_size, large_size), dtype=jnp.float32), gpu_device
                )

                @jit
                def memory_test(x):
                    return x * 1.01 + 0.001  # Simple memory-bound operation

                # Warmup
                _ = memory_test(x_large).block_until_ready()

                start = time.perf_counter()
                for _ in range(5):
                    _ = memory_test(x_large).block_until_ready()
                elapsed = (time.perf_counter() - start) / 5

                # Memory bandwidth in GB/s (read + write)
                bytes_transferred = 2 * x_large.nbytes
                benchmarks["memory_bandwidth"]["gb_per_sec"] = bytes_transferred / (
                    elapsed * 1e9
                )
                benchmarks["memory_bandwidth"]["test_size_mb"] = x_large.nbytes / (
                    1024**2
                )

        except Exception as e:
            logger.warning(f"JAX GPU benchmarking failed: {e}")

        return benchmarks

    def determine_optimal_settings(self) -> Dict[str, Any]:
        """
        Determine optimal settings for VI+JAX and MCMC+JAX workloads.

        Considers dataset sizes, memory requirements, and JAX-specific optimizations.
        """
        settings: Dict[str, Any] = {
            "use_gpu": False,
            "xla_flags": [],
            "jax_settings": {},
            "vi_batch_size": 1000,
            "mcmc_batch_size": 500,
            "memory_fraction": 0.9,
            "dataset_size_recommendations": {},
        }

        if not self.gpu_info.get("jax_gpu_available"):
            logger.info("GPU not available - using CPU-optimized settings")
            settings["dataset_size_recommendations"] = {
                "small": {"method": "VI+JAX", "batch_size": 10000},
                "medium": {"method": "VI+JAX", "batch_size": 5000, "chunking": True},
                "large": {
                    "method": "VI+JAX",
                    "batch_size": 2000,
                    "chunking": True,
                    "warning": "Consider GPU for large datasets",
                },
            }
            return settings

        settings["use_gpu"] = True

        # GPU-specific optimizations
        if self.gpu_info.get("devices"):
            device = self.gpu_info["devices"][0]
            memory_mb = device.get("memory_mb", 8192)
            compute_cap = device.get("compute_capability", "7.0")

            # Memory management based on GPU memory
            if memory_mb < 4096:  # <4GB
                settings["memory_fraction"] = 0.6
                settings["vi_batch_size"] = 500
                settings["mcmc_batch_size"] = 100
            elif memory_mb < 8192:  # 4-8GB
                settings["memory_fraction"] = 0.7
                settings["vi_batch_size"] = 1000
                settings["mcmc_batch_size"] = 250
            elif memory_mb < 16384:  # 8-16GB
                settings["memory_fraction"] = 0.8
                settings["vi_batch_size"] = 2000
                settings["mcmc_batch_size"] = 500
            else:  # >16GB
                settings["memory_fraction"] = 0.9
                settings["vi_batch_size"] = 5000
                settings["mcmc_batch_size"] = 1000

            # JAX-specific XLA optimizations
            cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
            settings["xla_flags"] = [
                f"--xla_gpu_cuda_data_dir={cuda_home}",
                "--xla_gpu_enable_triton_softmax_fusion=true",
                "--xla_gpu_triton_gemm_any=true",
                "--xla_gpu_enable_async_collectives=true",
                "--xla_gpu_enable_latency_hiding_scheduler=true",
            ]

            # Compute capability specific optimizations
            try:
                compute_cap_float = float(compute_cap)
                if compute_cap_float >= 8.0:  # Ampere and newer (A100, RTX 30xx+)
                    settings["xla_flags"].append("--xla_gpu_enable_triton_gemm=true")
                    settings["jax_settings"]["jax_enable_x64"] = True
                    settings["xla_flags"].append("--xla_gpu_enable_triton=true")
                elif compute_cap_float >= 7.0:  # Volta/Turing (V100, RTX 20xx)
                    settings["jax_settings"][
                        "jax_enable_x64"
                    ] = False  # Float32 for performance
            except (ValueError, TypeError):
                # Fallback for invalid compute capability
                settings["jax_settings"]["jax_enable_x64"] = False

            # Dataset size recommendations for GPU
            settings["dataset_size_recommendations"] = {
                "small": {
                    "method": "VI+JAX",
                    "batch_size": settings["vi_batch_size"],
                    "gpu_utilization": "low",
                    "memory_strategy": "full_memory",
                },
                "medium": {
                    "method": "VI+JAX or MCMC+JAX",
                    "vi_batch_size": settings["vi_batch_size"],
                    "mcmc_batch_size": settings["mcmc_batch_size"],
                    "gpu_utilization": "medium",
                    "memory_strategy": "chunked_processing",
                },
                "large": {
                    "method": "VI+JAX (primary), MCMC+JAX (if needed)",
                    "vi_batch_size": settings["vi_batch_size"] // 2,
                    "mcmc_batch_size": settings["mcmc_batch_size"] // 2,
                    "gpu_utilization": "high",
                    "memory_strategy": "streaming_chunks",
                    "recommendation": "Use VI+JAX for exploration, MCMC+JAX for final analysis",
                },
            }

        self.optimal_settings = settings
        return settings

    def save_optimization_cache(self):
        """Save optimization results to cache."""
        cache_data = {
            "timestamp": time.time(),
            "gpu_info": self.gpu_info,
            "optimal_settings": self.optimal_settings,
            "homodyne_version": "v2",
            "jax_available": JAX_AVAILABLE,
        }

        try:
            with open(self.cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save GPU optimization cache: {e}")

    def load_optimization_cache(self) -> bool:
        """Load cached optimization if recent and valid."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file) as f:
                    cache_data = json.load(f)

                # Check if cache is recent (7 days) and from v2
                cache_age = time.time() - cache_data.get("timestamp", 0)
                if (
                    cache_age < 7 * 24 * 3600
                    and cache_data.get("homodyne_version") == "v2"
                    and cache_data.get("jax_available") == JAX_AVAILABLE
                ):
                    self.gpu_info = cache_data.get("gpu_info", {})
                    self.optimal_settings = cache_data.get("optimal_settings", {})
                    return True
        except Exception:
            pass

        return False

    def apply_optimal_settings(self):
        """Apply optimal settings to environment for JAX."""
        if not self.optimal_settings.get("use_gpu"):
            print("ℹ️  GPU optimization not available - using CPU-optimized settings")
            os.environ["JAX_PLATFORMS"] = "cpu"
            return

        # Set JAX GPU settings
        os.environ["JAX_PLATFORMS"] = "gpu,cpu"  # Prefer GPU, fallback to CPU

        # Set XLA flags
        if self.optimal_settings.get("xla_flags"):
            os.environ["XLA_FLAGS"] = " ".join(self.optimal_settings["xla_flags"])

        # Set JAX-specific settings
        for key, value in self.optimal_settings.get("jax_settings", {}).items():
            os.environ[key.upper()] = str(value)

        # Set memory fraction
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(
            self.optimal_settings["memory_fraction"]
        )

        print("✅ JAX GPU optimization applied:")
        print(f"   Memory fraction: {self.optimal_settings['memory_fraction']}")
        print(f"   VI batch size: {self.optimal_settings['vi_batch_size']}")
        print(f"   MCMC batch size: {self.optimal_settings['mcmc_batch_size']}")

    def generate_report(self) -> str:
        """Generate detailed GPU optimization report for homodyne v2."""
        report = []
        report.append("=" * 70)
        report.append("🚀 Homodyne v2 GPU Optimization Report")
        report.append("=" * 70)

        # Hardware detection
        report.append("\n📊 Hardware Detection:")
        if self.gpu_info.get("devices"):
            for i, device in enumerate(self.gpu_info["devices"]):
                report.append(f"   GPU {i}: {device['name']}")
                report.append(f"      Memory: {device['memory_mb']:,} MB")
                report.append(
                    f"      Compute Capability: {device['compute_capability']}"
                )
                report.append(f"      Driver: {device['driver_version']}")
        else:
            report.append("   No NVIDIA GPU detected")

        # CUDA and JAX status
        report.append(
            f"\n   CUDA Available: {self.gpu_info.get('cuda_available', False)}"
        )
        if self.gpu_info.get("cuda_version"):
            report.append(f"   CUDA Version: {self.gpu_info['cuda_version']}")

        report.append(f"   JAX Available: {JAX_AVAILABLE}")
        report.append(
            f"   JAX GPU Support: {self.gpu_info.get('jax_gpu_available', False)}"
        )

        # Optimization recommendations
        report.append("\n⚙️  Optimization Settings:")
        if self.optimal_settings.get("use_gpu"):
            report.append("   ✅ GPU acceleration recommended for VI+JAX and MCMC+JAX")
            report.append(
                f"   Memory fraction: {self.optimal_settings['memory_fraction']}"
            )
            report.append(
                f"   VI batch size: {self.optimal_settings['vi_batch_size']:,}"
            )
            report.append(
                f"   MCMC batch size: {self.optimal_settings['mcmc_batch_size']:,}"
            )
        else:
            report.append("   💻 CPU-only mode recommended")
            report.append("   VI+JAX will use CPU with NumPy fallback")

        # Dataset size recommendations
        if "dataset_size_recommendations" in self.optimal_settings:
            report.append("\n📏 Dataset Size Recommendations:")
            for size_cat, rec in self.optimal_settings[
                "dataset_size_recommendations"
            ].items():
                report.append(f"   {size_cat.upper()}:")
                report.append(f"      Method: {rec['method']}")
                if "vi_batch_size" in rec:
                    report.append(f"      VI batch size: {rec['vi_batch_size']:,}")
                if "mcmc_batch_size" in rec:
                    report.append(f"      MCMC batch size: {rec['mcmc_batch_size']:,}")
                if "recommendation" in rec:
                    report.append(f"      Note: {rec['recommendation']}")

        report.append("\n" + "=" * 70)
        return "\n".join(report)


def main():
    """Main function for GPU optimization CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Optimizer for Homodyne v2 - VI+JAX and MCMC+JAX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  homodyne-gpu-optimize                    # Auto-detect and optimize
  homodyne-gpu-optimize --benchmark        # Run JAX benchmarks
  homodyne-gpu-optimize --apply            # Apply optimal settings
  homodyne-gpu-optimize --report           # Generate detailed report
  homodyne-gpu-optimize --force            # Force re-detection
        """,
    )

    parser.add_argument(
        "--benchmark", action="store_true", help="Run JAX GPU benchmarks"
    )
    parser.add_argument("--apply", action="store_true", help="Apply optimal settings")
    parser.add_argument(
        "--report", action="store_true", help="Generate optimization report"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-detection (ignore cache)"
    )

    args = parser.parse_args()

    optimizer = GPUOptimizer()

    # Load cache or detect hardware
    if not args.force and optimizer.load_optimization_cache():
        print("📦 Loaded cached GPU optimization")
    else:
        print("🔍 Detecting GPU hardware for JAX...")
        optimizer.detect_gpu_hardware()

        if args.benchmark and optimizer.gpu_info.get("jax_gpu_available"):
            print("⏱️  Running JAX workload benchmarks...")
            benchmarks = optimizer.benchmark_jax_workloads()

            if benchmarks.get("matrix_operations"):
                print("\n📊 JAX Benchmark Results:")
                for size, result in benchmarks["matrix_operations"].items():
                    print(
                        f"   Matrix {size}x{size}: {result['time_ms']:.2f}ms ({result['gflops']:.1f} GFLOPS)"
                    )

                if benchmarks.get("memory_bandwidth", {}).get("gb_per_sec"):
                    print(
                        f"   Memory Bandwidth: {benchmarks['memory_bandwidth']['gb_per_sec']:.1f} GB/s"
                    )

        optimizer.determine_optimal_settings()
        optimizer.save_optimization_cache()

    if args.apply:
        optimizer.apply_optimal_settings()

    if args.report or not any([args.benchmark, args.apply]):
        print(optimizer.generate_report())


if __name__ == "__main__":
    main()
