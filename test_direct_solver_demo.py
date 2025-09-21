#!/usr/bin/env python
"""
Demonstration of JAX-based Direct Classical Least Squares Solver
================================================================

This script demonstrates the high-performance JAX-accelerated direct solver
for classical least squares fitting with the Normal Equation approach.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from homodyne.optimization import (
    DirectLeastSquaresSolver,
    DirectSolverConfig,
    fit_homodyne_direct
)

def demo_basic_fitting():
    """Demonstrate basic fitting with contrast and offset."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Least Squares Fitting")
    print("="*60)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 10000
    theory = np.random.randn(n_samples)

    # True parameters
    contrast_true = 0.5
    offset_true = 1.0

    # Add noise
    noise_level = 0.1
    data = theory * contrast_true + offset_true + noise_level * np.random.randn(n_samples)

    # Fit using direct solver
    print(f"\nFitting {n_samples:,} data points...")
    solver = DirectLeastSquaresSolver()

    start_time = time.time()
    result = solver.fit(data, theory)
    fit_time = time.time() - start_time

    # Print results
    print(f"\nResults:")
    print(f"  True contrast:    {contrast_true:.4f}")
    print(f"  Fitted contrast:  {result.contrast:.4f}")
    print(f"  Error:            {abs(result.contrast - contrast_true):.4f}")
    print(f"\n  True offset:      {offset_true:.4f}")
    print(f"  Fitted offset:    {result.offset:.4f}")
    print(f"  Error:            {abs(result.offset - offset_true):.4f}")
    print(f"\nFit statistics:")
    print(f"  Chi-squared:      {result.chi_squared:.2f}")
    print(f"  Reduced chi²:     {result.reduced_chi_squared:.4f}")
    print(f"  Residual std:     {result.residual_std:.4f}")
    print(f"  Backend:          {result.backend}")
    print(f"  Fit time:         {fit_time:.4f} seconds")

    return result, data, theory


def demo_dataset_sizes():
    """Demonstrate performance across different dataset sizes."""
    print("\n" + "="*60)
    print("DEMO 2: Performance Across Dataset Sizes")
    print("="*60)

    sizes = [1000, 10000, 100000, 1000000, 5000000]
    times_direct = []
    times_numpy = []

    solver = DirectLeastSquaresSolver()

    for size in sizes:
        print(f"\nTesting with {size:,} data points...")

        # Generate data
        theory = np.random.randn(size).astype(np.float32)
        data = 0.5 * theory + 1.0 + 0.01 * np.random.randn(size)

        # Time direct solver
        start = time.time()
        result = solver.fit(data, theory)
        time_direct = time.time() - start
        times_direct.append(time_direct)

        # Time NumPy lstsq for comparison
        start = time.time()
        A = np.column_stack([theory, np.ones(size)])
        params_numpy = np.linalg.lstsq(A, data, rcond=None)[0]
        time_numpy = time.time() - start
        times_numpy.append(time_numpy)

        speedup = time_numpy / time_direct
        print(f"  Direct solver: {time_direct:.4f}s")
        print(f"  NumPy lstsq:   {time_numpy:.4f}s")
        print(f"  Speedup:       {speedup:.2f}x")
        print(f"  Dataset size:  {result.dataset_size}")

    return sizes, times_direct, times_numpy


def demo_chunked_processing():
    """Demonstrate chunked processing for large datasets."""
    print("\n" + "="*60)
    print("DEMO 3: Chunked Processing for Large Datasets")
    print("="*60)

    # Create a large dataset
    n_samples = 12_000_000  # 12M points (LARGE category)
    print(f"\nGenerating {n_samples:,} data points...")

    theory = np.random.randn(n_samples).astype(np.float32)
    data = 0.7 * theory + 0.3 + 0.01 * np.random.randn(n_samples).astype(np.float32)

    # Configure solver for large datasets
    config = DirectSolverConfig(
        chunk_size_large=1000,  # Small chunks for memory efficiency
        use_parameter_space_bounds=True
    )

    solver = DirectLeastSquaresSolver(config=config)

    print("Fitting with chunked processing...")
    start = time.time()
    result = solver.fit(data, theory)
    fit_time = time.time() - start

    print(f"\nResults for {n_samples:,} points:")
    print(f"  Contrast:      {result.contrast:.4f} (true: 0.7)")
    print(f"  Offset:        {result.offset:.4f} (true: 0.3)")
    print(f"  Dataset size:  {result.dataset_size}")
    print(f"  Fit time:      {fit_time:.4f} seconds")
    print(f"  Points/second: {n_samples/fit_time:,.0f}")

    return result


def demo_weighted_least_squares():
    """Demonstrate weighted least squares with uncertainties."""
    print("\n" + "="*60)
    print("DEMO 4: Weighted Least Squares")
    print("="*60)

    n_samples = 10000
    theory = np.random.randn(n_samples)

    # Generate heteroscedastic data (varying noise)
    sigma = np.abs(0.1 + 0.2 * np.abs(theory))  # Noise increases with |theory|
    noise = sigma * np.random.randn(n_samples)
    data = 0.6 * theory + 0.4 + noise

    solver = DirectLeastSquaresSolver()

    # Unweighted fit
    print("\nUnweighted fit:")
    result_unweighted = solver.fit(data, theory)
    print(f"  Contrast: {result_unweighted.contrast:.4f} (true: 0.6)")
    print(f"  Offset:   {result_unweighted.offset:.4f} (true: 0.4)")

    # Weighted fit
    print("\nWeighted fit (using sigma):")
    result_weighted = solver.fit(data, theory, sigma=sigma)
    print(f"  Contrast: {result_weighted.contrast:.4f} (true: 0.6)")
    print(f"  Offset:   {result_weighted.offset:.4f} (true: 0.4)")

    print("\nWeighted fit should be more accurate!")

    return result_unweighted, result_weighted


def demo_benchmark():
    """Run benchmark across multiple dataset sizes."""
    print("\n" + "="*60)
    print("DEMO 5: Performance Benchmark")
    print("="*60)

    solver = DirectLeastSquaresSolver()

    # Run benchmark
    data_sizes = [1000, 10000, 100000, 1000000]
    results = solver.benchmark(data_sizes=data_sizes, n_trials=3)

    print("\nBenchmark Results:")
    print("-" * 40)
    print("Size        Mean Time   Std Time    Backend")
    print("-" * 40)
    for size, stats in results.items():
        print(f"{size:<10} {stats['mean_time']:.4f}s    "
              f"±{stats['std_time']:.4f}s   {stats['backend']}")

    return results


def main():
    """Run all demonstrations."""
    print("\n" + "="*80)
    print(" JAX Classical Least Squares Solver Demonstration")
    print("="*80)

    # Check JAX availability
    try:
        import jax
        print(f"\n✓ JAX is available (version {jax.__version__})")
        print(f"  Devices: {jax.devices()}")
    except ImportError:
        print("\n✗ JAX not available - using NumPy fallback")
        print("  Install JAX for 10-100x speedup: pip install jax jaxlib")

    # Run demonstrations
    demo_basic_fitting()
    demo_dataset_sizes()
    demo_chunked_processing()
    demo_weighted_least_squares()
    demo_benchmark()

    print("\n" + "="*80)
    print(" Demonstration Complete!")
    print("="*80)
    print("\nKey Takeaways:")
    print("• JAX provides significant speedup for large datasets")
    print("• Automatic chunking handles datasets of any size")
    print("• Weighted least squares improves accuracy with known uncertainties")
    print("• Direct solution is non-iterative and always converges")
    print("• Seamless integration with existing homodyne architecture")


if __name__ == "__main__":
    main()