#!/usr/bin/env python
"""Benchmark oneDNN Performance for XPCS Analysis

This script benchmarks the performance impact of Intel oneDNN optimizations
on typical XPCS analysis workloads. It compares CPU performance with and
without oneDNN enabled.

Expected Results:
- XPCS workloads are element-wise dominated (exp, sin, cos, sqrt)
- oneDNN is optimized for matrix operations (GEMM, convolutions)
- Expected performance difference: minimal (< 5% improvement)
- Only Intel CPUs support oneDNN

Usage:
    python benchmark_onednn.py

Output:
    Performance comparison with oneDNN enabled vs disabled
    Recommendation on whether to enable oneDNN for your system
"""

import time

import numpy as np

# Homodyne imports
from homodyne.device import configure_cpu_hpc, detect_cpu_info


def run_benchmark(enable_onednn: bool, num_iterations: int = 5) -> dict:
    """Run CPU benchmark with or without oneDNN.

    Args:
        enable_onednn: Whether to enable Intel oneDNN optimizations
        num_iterations: Number of benchmark iterations

    Returns:
        Benchmark results with timing information
    """
    print(f"\n{'=' * 60}")
    print(f"Benchmarking with oneDNN: {enable_onednn}")
    print("=" * 60)

    # Configure CPU with oneDNN setting
    config = configure_cpu_hpc(
        num_threads=None,  # Auto-detect
        enable_hyperthreading=False,
        numa_policy="auto",
        memory_optimization="standard",
        enable_onednn=enable_onednn,
    )

    print(f"Configuration: {config.get('threads_configured')} threads")
    print(f"oneDNN status: {config.get('onednn_enabled', False)}")

    # Import JAX after configuration (XLA_FLAGS must be set before JAX import)
    import jax
    import jax.numpy as jnp

    print(f"JAX devices: {jax.devices()}")
    print(f"XLA_FLAGS: {config}")

    # Simulate typical XPCS operations
    test_size = 1000
    results = {"timings": []}

    print(f"\nRunning {num_iterations} iterations...")

    @jax.jit
    def xpcs_like_computation(t1, t2, phi, params):
        """Simulate typical XPCS correlation function computation.

        This includes:
        - Meshgrid operations
        - Element-wise exp, sqrt, sin, cos operations
        - Cumulative sums
        - Broadcasting
        """
        # Time difference matrix
        jnp.abs(t1[:, None] - t2[None, :])

        # Anomalous diffusion term: exp(-q²/2 * ∫D(t)dt)
        D0, alpha, D_offset = params[0], params[1], params[2]
        D_t = D0 * jnp.power(jnp.arange(len(t1)) + 1.0, alpha) + D_offset
        D_integral = jnp.cumsum(D_t)
        D_diff = jnp.abs(D_integral[:, None] - D_integral[None, :])
        g1_diff = jnp.exp(-0.5 * 0.01**2 * D_diff)

        # Shear flow term: sinc²(phase)
        gamma_dot_0, beta = params[3], params[4]
        gamma_t = gamma_dot_0 * jnp.power(jnp.arange(len(t1)) + 1.0, beta)
        gamma_integral = jnp.cumsum(gamma_t)
        gamma_diff = jnp.abs(gamma_integral[:, None] - gamma_integral[None, :])

        # Angular dependence
        phi_reshaped = phi[:, None, None]
        phase = 0.5 / jnp.pi * 0.01 * 1.0 * gamma_diff * jnp.cos(phi_reshaped)
        sinc_term = jnp.where(jnp.abs(phase) > 1e-10, jnp.sin(phase) / phase, 1.0)
        g1_shear = sinc_term**2

        # Total g1
        g1_total = g1_diff[None, :, :] * g1_shear

        # g2 = 1 + contrast * g1²
        contrast = 0.5
        g2 = 1.0 + contrast * g1_total**2

        return jnp.sum(g2)

    # Prepare test data
    t1 = jnp.linspace(0, 100, test_size)
    t2 = jnp.linspace(0, 100, test_size)
    phi = jnp.linspace(0, 2 * jnp.pi, 12)
    params = jnp.array([1000.0, 0.5, 10.0, 100.0, 0.3])

    # Warm-up JIT compilation
    print("Warming up JIT compilation...")
    _ = xpcs_like_computation(t1[:100], t2[:100], phi, params)

    # Run benchmarks
    print("Running benchmarks...")
    for i in range(num_iterations):
        start_time = time.perf_counter()

        result = xpcs_like_computation(t1, t2, phi, params)
        result.block_until_ready()  # Ensure computation completes

        elapsed = time.perf_counter() - start_time
        results["timings"].append(elapsed)
        print(f"  Iteration {i + 1}/{num_iterations}: {elapsed:.4f}s")

    # Calculate statistics
    results["mean_time"] = np.mean(results["timings"])
    results["std_time"] = np.std(results["timings"])
    results["min_time"] = np.min(results["timings"])

    print("\nResults:")
    print(f"  Mean time: {results['mean_time']:.4f}s ± {results['std_time']:.4f}s")
    print(f"  Best time: {results['min_time']:.4f}s")

    return results


def main():
    """Main benchmark function."""
    print("\n" + "=" * 60)
    print(" Intel oneDNN Performance Benchmark for XPCS Analysis")
    print("=" * 60)

    # Detect CPU
    cpu_info = detect_cpu_info()
    print("\nCPU Information:")
    print(f"  Brand: {cpu_info.get('cpu_brand', 'Unknown')}")
    print(f"  Physical cores: {cpu_info['physical_cores']}")
    print(f"  Logical cores: {cpu_info['logical_cores']}")
    print(f"  AVX-512 support: {cpu_info.get('supports_avx512', False)}")

    # Check if Intel CPU
    if "Intel" not in cpu_info.get("cpu_brand", ""):
        print("\n⚠️  WARNING: oneDNN is only supported on Intel CPUs.")
        print("   Your CPU is not Intel, so oneDNN will be skipped.")
        print("   Exiting benchmark.")
        return

    print("\n" + "=" * 60)
    print("Running benchmarks (this may take a few minutes)...")
    print("=" * 60)

    # Run without oneDNN
    results_without = run_benchmark(enable_onednn=False, num_iterations=5)

    # Run with oneDNN
    results_with = run_benchmark(enable_onednn=True, num_iterations=5)

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    speedup = results_without["mean_time"] / results_with["mean_time"]
    improvement_pct = (
        1 - results_with["mean_time"] / results_without["mean_time"]
    ) * 100

    print(
        f"\nWithout oneDNN: {results_without['mean_time']:.4f}s ± {results_without['std_time']:.4f}s"
    )
    print(
        f"With oneDNN:    {results_with['mean_time']:.4f}s ± {results_with['std_time']:.4f}s"
    )
    print(f"\nSpeedup:        {speedup:.3f}x")
    print(f"Improvement:    {improvement_pct:+.2f}%")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if improvement_pct > 10:
        print("\n✅ ENABLE oneDNN")
        print(
            f"   Significant performance improvement detected ({improvement_pct:+.2f}%)"
        )
        print("   Set enable_onednn=True in configure_cpu_hpc()")
    elif improvement_pct > 5:
        print("\n⚠️  CONSIDER oneDNN")
        print(f"   Moderate performance improvement detected ({improvement_pct:+.2f}%)")
        print("   May be worth enabling for large-scale production runs")
    elif improvement_pct > -5:
        print("\n❌ DO NOT ENABLE oneDNN")
        print(f"   Minimal or no performance impact ({improvement_pct:+.2f}%)")
        print("   Keep default configuration (oneDNN disabled)")
    else:
        print("\n❌ DO NOT ENABLE oneDNN")
        print(f"   Performance DEGRADATION detected ({improvement_pct:+.2f}%)")
        print("   oneDNN may be incompatible with your workload")

    print("\nNote:")
    print("  XPCS analysis is dominated by element-wise operations (exp, sin, cos)")
    print("  oneDNN is optimized for matrix operations (GEMM, convolutions)")
    print("  Expected improvement for XPCS: minimal (< 5%)")
    print("  Your specific results may vary based on dataset characteristics")

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
