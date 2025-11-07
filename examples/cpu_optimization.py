#!/usr/bin/env python
"""
CPU-Optimized NLSQ Analysis for HPC Systems
=============================================

This example demonstrates best practices for CPU-optimized XPCS analysis using
Homodyne v2.3 on HPC clusters and multi-core personal computers.

Key Features:
- Multi-core CPU thread management (optimal for 36/128-core HPC nodes)
- HPC cluster submission examples (Slurm, PBS)
- CPU performance benchmarking and monitoring
- NUMA-aware configuration for large systems
- JAX XLA CPU-specific optimizations
- Memory-efficient processing strategies

Who Should Use This:
- HPC cluster users with CPU-only access
- Personal computer users with many cores (8-128 cores)
- Research groups needing reliable, reproducible XPCS analysis
- Systems without GPU support or insufficient GPU memory

Performance Targets:
- 36-core HPC node: ~1-3 hours for 10M point analysis
- 128-core HPC node: ~30-60 minutes for 10M point analysis
- 16-core personal computer: ~4-8 hours for 10M point analysis

Note:
GPU support was removed in v2.3.0 to simplify maintenance and focus on
reliable CPU execution. v2.2.x remains available for GPU users.
"""

import os
import time
from pathlib import Path

import numpy as np

# Homodyne imports
from homodyne.config import ConfigManager
from homodyne.device import (
    configure_cpu_hpc,
    detect_cpu_info,
    get_optimal_batch_size,
)
from homodyne.optimization import fit_nlsq_jax


def detect_hpc_environment():
    """
    Detect HPC cluster environment and optimize configuration.

    Returns:
        dict: Environment information including cores, memory, cluster type
    """
    print("🔍 Detecting HPC Environment...")
    print("-" * 60)

    # Detect CPU information
    cpu_info = detect_cpu_info()

    print(f"CPU Architecture: {cpu_info['architecture']}")
    print(f"CPU Brand: {cpu_info.get('cpu_brand', 'Unknown')}")
    print(f"Physical cores: {cpu_info['physical_cores']}")
    print(f"Logical CPUs: {cpu_info['logical_cores']}")

    # Check for NUMA architecture (common on HPC)
    numa_nodes = cpu_info.get("numa_nodes", 1)
    if numa_nodes > 1:
        print(f"NUMA nodes: {numa_nodes}")
        print("⚠️ NUMA detected - multi-node memory access will be slower")

    # Detect CPU architecture optimizations
    supports_avx512 = cpu_info.get("supports_avx512", False)
    supports_avx = cpu_info.get("supports_avx", False)

    if supports_avx512:
        print("✓ AVX-512 support detected - optimal XLA compilation")
    elif supports_avx:
        print("✓ AVX-2 support detected - good XLA compilation")

    # Check cluster environment
    cluster_type = "local"
    if os.environ.get("SLURM_JOB_ID"):
        cluster_type = "Slurm"
    elif os.environ.get("PBS_JOBID"):
        cluster_type = "PBS"
    elif os.environ.get("LSB_JOBID"):
        cluster_type = "LSF"

    print(f"Cluster type: {cluster_type}")

    return {
        "cores_physical": cpu_info["physical_cores"],
        "cores_logical": cpu_info["logical_cores"],
        "has_numa": numa_nodes > 1,
        "has_avx512": supports_avx512,
        "cluster_type": cluster_type,
    }


def configure_cpu_optimal(cpu_info):
    """
    Configure CPU optimization based on detected environment.

    Args:
        cpu_info (dict): CPU information from detect_hpc_environment()

    Returns:
        dict: Recommended configuration for optimal CPU performance
    """
    print("\n⚙️ Configuring CPU Optimization...")
    print("-" * 60)

    # Configure optimal HPC CPU settings
    # Use num_threads parameter (leaves 1-2 cores for OS)
    optimal_threads = max(1, cpu_info["cores_logical"] - 2)
    config = configure_cpu_hpc(
        num_threads=optimal_threads,
        enable_hyperthreading=False,
        numa_policy='auto' if cpu_info.get("has_numa", False) else 'single',
        memory_optimization='standard',
    )

    print(f"OMP_NUM_THREADS: {config.get('omp_num_threads', optimal_threads)}")
    print(f"Thread configuration: {config}")

    # Set JAX XLA environment variables for CPU optimization
    # These flags enable CPU-specific optimizations in JAX/XLA
    xla_flags = []

    # Enable CPU-optimized code generation
    xla_flags.append("--xla_cpu_enable_fast_math=true")

    # Use the CPU backend's parallel threading
    xla_flags.append("--xla_cpu_multi_thread_eigen=true")

    # Optimize for Eigen tensor operations
    xla_flags.append("--xla_force_host_platform_device_count=1")

    # Auto-tune for the CPU
    xla_flags.append("--xla_cpu_prefer_vector_width=256")  # AVX2
    if cpu_info["has_avx512"]:
        xla_flags.append("--xla_cpu_prefer_vector_width=512")  # AVX-512

    os.environ["XLA_FLAGS"] = " ".join(xla_flags)
    os.environ["OMP_NUM_THREADS"] = str(optimal_threads)

    print(f"\n✓ JAX XLA Flags configured for CPU optimization:")
    print(f"  XLA_FLAGS={os.environ['XLA_FLAGS']}")
    print(f"\n✓ OpenMP threads configured:")
    print(f"  OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    return config


def generate_synthetic_xpcs_data(n_times=100, n_angles=12):
    """
    Generate synthetic XPCS data for demonstration.

    Args:
        n_times (int): Number of time points
        n_angles (int): Number of phi angles

    Returns:
        dict: Synthetic XPCS dataset
    """
    print(f"\n📊 Generating synthetic XPCS data...")
    print(f"   Time points: {n_times}, Phi angles: {n_angles}")

    # Create time arrays
    t1, t2 = np.meshgrid(
        np.arange(n_times, dtype=np.float32),
        np.arange(n_times, dtype=np.float32),
        indexing="ij",
    )

    # Create phi angle array
    phi = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Generate synthetic correlation function: c2(t1,t2,phi) = 1 + contrast*[c1(t1,t2,phi)]^2
    # Simplified model: exponential decay with angle dependence
    tau = np.abs(t1 - t2) + 1e-6  # Avoid division by zero
    q = 0.01  # Wave vector

    # Tau-dependent decay
    decay = np.exp(-tau / (10.0 + np.random.rand() * 5.0))

    # Angle-dependent contrast variation
    angle_dependence = np.zeros((n_angles, n_times, n_times))
    for i, angle in enumerate(phi):
        # Anisotropic effect: contrast varies with angle
        contrast = 0.5 + 0.2 * np.cos(2 * angle)
        c1_squared = (0.3 + contrast * 0.2 * decay) ** 2
        angle_dependence[i] = 1.0 + c1_squared

    # Add measurement noise
    sigma = np.ones_like(angle_dependence) * 0.01
    c2_exp = angle_dependence + sigma * np.random.randn(*angle_dependence.shape)

    return {
        "t1": t1,
        "t2": t2,
        "phi_angles_list": phi,
        "c2_exp": c2_exp,
        "wavevector_q_list": np.array([q]),
        "sigma": sigma,
        "data_size": n_times * n_angles * n_times,
    }


def run_cpu_optimized_analysis(data, cpu_info):
    """
    Run CPU-optimized NLSQ analysis.

    Args:
        data (dict): Synthetic XPCS dataset
        cpu_info (dict): CPU environment information

    Returns:
        dict: Optimization results with timing
    """
    print("\n⚡ Running CPU-Optimized NLSQ Analysis...")
    print("-" * 60)

    # Determine optimal batch size based on dataset size
    optimal_batch_size = get_optimal_batch_size(
        data_size=data["data_size"],
        available_memory_gb=None,
        target_memory_usage=0.7,
    )
    print(f"Optimal batch size: {optimal_batch_size:,} points per batch")

    # Create configuration for CPU optimization
    config = ConfigManager(
        config_override={
            "analysis_mode": "static_isotropic",
            "optimization": {
                "method": "nlsq",
                "lsq": {
                    "max_iterations": 100,
                    "tolerance": 1e-8,
                },
            },
            "performance": {
                "strategy_override": None,  # Auto-select based on data size
                "enable_progress": True,
            },
        }
    )

    # Time the optimization
    start_time = time.perf_counter()

    try:
        result = fit_nlsq_jax(data, config)
        elapsed = time.perf_counter() - start_time

        print(f"\n✅ NLSQ Optimization Completed!")
        print(f"   Time elapsed: {elapsed:.3f}s")
        print(f"   Data points: {data['data_size']:,}")
        print(f"   Throughput: {data['data_size'] / elapsed:,.0f} points/sec")

        if hasattr(result, "parameters"):
            print(f"\n   Fitted parameters:")
            for key, val in result.parameters.items():
                print(f"   - {key}: {val:.6f}")

        if hasattr(result, "chi_squared"):
            print(f"\n   Chi-squared: {result.chi_squared:.6f}")

        return {
            "success": True,
            "result": result,
            "elapsed_time": elapsed,
            "throughput": data["data_size"] / elapsed,
        }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n❌ NLSQ Optimization Failed!")
        print(f"   Error: {e}")
        print(f"   Time spent: {elapsed:.3f}s")

        return {
            "success": False,
            "error": str(e),
            "elapsed_time": elapsed,
        }


def benchmark_and_report():
    """Report CPU performance capabilities."""
    print("\n📈 CPU Performance Report...")
    print("-" * 60)

    cpu_info = detect_cpu_info()

    print(f"Physical cores: {cpu_info['physical_cores']}")
    print(f"Logical cores (with hyperthreading): {cpu_info['logical_cores']}")

    # Estimate theoretical performance
    base_ghz = 3.0  # Conservative estimate for modern CPUs
    cores = cpu_info['physical_cores']
    estimated_ghz = cores * base_ghz

    print(f"\nEstimated peak performance:")
    print(f"  Base frequency: ~{base_ghz} GHz per core")
    print(f"  Total: ~{estimated_ghz:.1f} GHz aggregate")
    print(f"  Note: Actual performance varies with workload and thermal conditions")

    return {"cores": cores, "estimated_ghz": estimated_ghz}


def print_hpc_submission_examples():
    """Print example HPC job submission scripts."""
    print("\n📋 HPC Job Submission Examples")
    print("=" * 60)

    print("\nExample 1: Slurm Cluster Submission")
    print("-" * 60)
    print("""#!/bin/bash
#SBATCH --job-name=homodyne_xpcs
#SBATCH --nodes=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --partition=cpu

# Load required modules
module load python/3.12
module load openblas/0.3.18

# Activate conda environment
conda activate homodyne

# Set CPU optimization flags
export OMP_NUM_THREADS=36
export OMP_PLACES=cores
export OMP_PROC_BIND=tight
export XLA_FLAGS="--xla_cpu_enable_fast_math=true --xla_cpu_multi_thread_eigen=true"

# Run Homodyne analysis
homodyne --config config.yaml --verbose

# Optional: Run multiple configurations in parallel
# for config in configs/*.yaml; do
#   homodyne --config "$config" &
# done
# wait
""")

    print("\nExample 2: PBS Cluster Submission")
    print("-" * 60)
    print("""#!/bin/bash
#PBS -N homodyne_xpcs
#PBS -l select=1:ncpus=128:mem=256gb
#PBS -l walltime=02:00:00
#PBS -q cpu

# Load required modules
module load python/3.12
module load intel-compiler

# Activate virtual environment
source ~/venv/homodyne/bin/activate

# Set CPU optimization flags
export OMP_NUM_THREADS=128
export KMP_AFFINITY=granularity=fine,compact
export XLA_FLAGS="--xla_cpu_enable_fast_math=true --xla_cpu_multi_thread_eigen=true"

# Change to work directory
cd "$PBS_O_WORKDIR"

# Run Homodyne analysis
homodyne --config config.yaml --verbose

# Generate report
echo "Analysis completed at $(date)" >> analysis.log
""")

    print("\nExample 3: Local Multi-Core Execution (Personal Computer)")
    print("-" * 60)
    print("""#!/bin/bash
# For a 16-core personal computer

# Set CPU optimization flags
export OMP_NUM_THREADS=14  # Leave 2 cores for OS
export OMP_PLACES=cores
export OMP_PROC_BIND=tight
export XLA_FLAGS="--xla_cpu_enable_fast_math=true --xla_cpu_multi_thread_eigen=true"

# Run Homodyne analysis
homodyne --config config.yaml --verbose

# Monitor performance
echo "CPU Usage:"
top -b -n 1 | grep "Cpu(s)"
echo "Memory Usage:"
free -h
""")


def main():
    """Main CPU optimization demonstration."""
    print("\n" + "=" * 60)
    print(" CPU-Optimized Homodyne v2.3 Analysis Example")
    print("=" * 60)
    print("\nGPU support removed in v2.3.0")
    print("Focus: Multi-core CPU optimization for HPC clusters")

    # Step 1: Detect HPC environment
    cpu_info = detect_hpc_environment()

    # Step 2: Configure CPU optimization
    config = configure_cpu_optimal(cpu_info)

    # Step 3: Report CPU performance
    benchmark_results = benchmark_and_report()

    # Step 4: Generate synthetic data
    data = generate_synthetic_xpcs_data(n_times=50, n_angles=8)

    # Step 5: Run CPU-optimized analysis
    results = run_cpu_optimized_analysis(data, cpu_info)

    # Step 6: Show HPC submission examples
    print_hpc_submission_examples()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    if results["success"]:
        print(f"✅ Analysis successful!")
        print(f"   Throughput: {results['throughput']:,.0f} points/sec")
        print(f"   Time: {results['elapsed_time']:.3f}s")
    else:
        print(f"❌ Analysis failed: {results['error']}")

    print("\nRecommended Configuration for Your System:")
    print(f"- Cores to use: {min(cpu_info['cores_logical'] - 1, 36)}")
    print(f"- Memory-efficient batch size: {get_optimal_batch_size(data['data_size']):,} points")
    print(f"- Cluster type: {cpu_info['cluster_type']}")

    print("\n✅ CPU-Optimized Example Completed!")
    print("\nNext steps:")
    print("1. Adjust OMP_NUM_THREADS based on your system")
    print("2. Submit to HPC cluster using provided scripts")
    print("3. Monitor performance with system tools (top, nvidia-smi, etc.)")
    print("4. Tune batch size for your specific hardware")


if __name__ == "__main__":
    main()
