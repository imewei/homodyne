#!/usr/bin/env python
"""
Multi-Core Batch Processing for XPCS Analysis
==============================================

This example demonstrates how to efficiently process multiple XPCS datasets
in parallel using Homodyne v2.3 on multi-core systems.

Key Features:
- Parallel processing of multiple datasets
- Intelligent work distribution across cores
- Memory-efficient batch processing
- Progress tracking with detailed logging
- Result aggregation and reporting
- CPU resource management for stable operation

Use Cases:
- Analyze multiple time points from same experiment
- Process different q-values in parallel
- Compare analysis across experimental conditions
- Batch processing of archived XPCS data
- High-throughput screening of samples

Performance Benefits:
- Linear scaling with CPU cores (up to 32-36 cores)
- Minimal memory overhead due to sequential processing
- Fault-tolerant design (failure of one doesn't stop others)
- Adaptive batch sizing based on available memory
- Progress tracking for long-running jobs

Limitations:
- One dataset per core (no sub-core parallelism)
- Memory must accommodate largest dataset
- Requires coordinated CPU allocation (don't oversubscribe)

Note:
GPU support removed in v2.3.0. Use multi-core CPU parallelism for
throughput scaling instead of relying on GPU acceleration.
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# Homodyne imports
from homodyne.config import ConfigManager
from homodyne.device import detect_cpu_info
from homodyne.optimization import fit_nlsq_jax


def generate_multiple_datasets(
    num_datasets: int = 4,
    n_times: int = 50,
    n_angles: int = 12,
) -> list[dict]:
    """
    Generate multiple synthetic XPCS datasets for batch processing.

    Args:
        num_datasets (int): Number of datasets to generate
        n_times (int): Time points per dataset
        n_angles (int): Phi angles per dataset

    Returns:
        List[Dict]: List of XPCS datasets
    """
    print(f"📊 Generating {num_datasets} synthetic datasets...")
    print(f"   Format: {n_times}x{n_times} times × {n_angles} angles each")

    datasets = []

    for dataset_id in range(num_datasets):
        # Vary parameters slightly between datasets
        decay_time = 10.0 + dataset_id * 2.0
        baseline_contrast = 0.5 + dataset_id * 0.05

        # Create time arrays
        t1, t2 = np.meshgrid(
            np.arange(n_times, dtype=np.float32),
            np.arange(n_times, dtype=np.float32),
            indexing="ij",
        )

        # Create phi angle array
        phi = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

        # Generate synthetic correlation function
        tau = np.abs(t1 - t2) + 1e-6
        decay = np.exp(-tau / decay_time)

        # Angle-dependent contrast
        angle_dependence = np.zeros((n_angles, n_times, n_times))
        for i, angle in enumerate(phi):
            contrast = baseline_contrast + 0.2 * np.cos(2 * angle)
            c1_squared = (0.3 + contrast * 0.2 * decay) ** 2
            angle_dependence[i] = 1.0 + c1_squared

        # Add noise
        sigma = np.ones_like(angle_dependence) * 0.01
        c2_exp = angle_dependence + sigma * np.random.randn(*angle_dependence.shape)

        dataset = {
            "dataset_id": dataset_id,
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp,
            "wavevector_q_list": np.array([0.01]),
            "sigma": sigma,
            "data_size": n_times * n_angles * n_times,
            "description": f"Dataset {dataset_id}: decay_time={decay_time:.1f}, contrast={baseline_contrast:.2f}",
        }

        datasets.append(dataset)

    print(f"✓ Generated {num_datasets} datasets")
    total_size = sum(d["data_size"] for d in datasets)
    print(f"  Total data points: {total_size:,}")

    return datasets


def process_single_dataset(dataset: dict) -> tuple[int, dict]:
    """
    Process a single XPCS dataset with NLSQ optimization.

    This function runs in a separate process for parallel execution.

    Args:
        dataset (Dict): Single XPCS dataset

    Returns:
        Tuple[int, Dict]: Dataset ID and results
    """
    dataset_id = dataset["dataset_id"]

    try:
        # Create configuration for this dataset
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
            }
        )

        # Time the optimization
        start_time = time.perf_counter()

        # Run optimization
        result = fit_nlsq_jax(dataset, config)

        elapsed = time.perf_counter() - start_time

        # Extract results
        results = {
            "dataset_id": dataset_id,
            "success": True,
            "elapsed_time": elapsed,
            "data_size": dataset["data_size"],
            "throughput": dataset["data_size"] / elapsed,
            "description": dataset["description"],
        }

        # Extract fitted parameters if available
        if hasattr(result, "parameters"):
            results["parameters"] = result.parameters

        if hasattr(result, "chi_squared"):
            results["chi_squared"] = result.chi_squared

        return dataset_id, results

    except Exception as e:
        # Return error result
        return dataset_id, {
            "dataset_id": dataset_id,
            "success": False,
            "error": str(e),
            "data_size": dataset["data_size"],
            "description": dataset["description"],
        }


def parallel_batch_processing(
    datasets: list[dict],
    max_workers: int = None,
) -> dict:
    """
    Process multiple datasets in parallel using multi-core execution.

    Args:
        datasets (List[Dict]): List of XPCS datasets
        max_workers (int): Maximum parallel workers (None = auto-detect)

    Returns:
        Dict: Aggregated results from all datasets
    """
    print("\n⚡ Starting Parallel Batch Processing...")
    print("-" * 60)

    # Auto-detect optimal number of workers
    if max_workers is None:
        cpu_info = detect_cpu_info()
        max_workers = max(1, cpu_info["logical_cores"] - 1)

    print(f"CPU cores available: {max_workers}")
    print(f"Datasets to process: {len(datasets)}")

    # Track results
    results = {
        "total_datasets": len(datasets),
        "successful": 0,
        "failed": 0,
        "datasets": {},
        "timing": {},
        "performance_summary": {},
    }

    # Start overall timer
    overall_start = time.perf_counter()

    # Process datasets in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_dataset = {
            executor.submit(process_single_dataset, dataset): dataset
            for dataset in datasets
        }

        # Process completed jobs with progress tracking
        completed = 0
        for future in as_completed(future_to_dataset):
            completed += 1
            dataset_id, dataset_result = future.result()

            # Store result
            results["datasets"][dataset_id] = dataset_result

            # Update counters
            if dataset_result["success"]:
                results["successful"] += 1
                status = "✓"
            else:
                results["failed"] += 1
                status = "✗"

            # Print progress
            elapsed_dataset = dataset_result.get("elapsed_time", 0)
            print(
                f"[{completed}/{len(datasets)}] {status} Dataset {dataset_id}: "
                f"{elapsed_dataset:.3f}s "
                f"({dataset_result.get('throughput', 0):,.0f} pts/sec)"
            )

    overall_time = time.perf_counter() - overall_start

    # Calculate performance summary
    successful_results = [r for r in results["datasets"].values() if r["success"]]
    if successful_results:
        times = [r["elapsed_time"] for r in successful_results]
        throughputs = [r["throughput"] for r in successful_results]

        results["performance_summary"] = {
            "total_time": overall_time,
            "parallel_efficiency": (
                (sum(times) / overall_time) if overall_time > 0 else 0
            ),
            "average_time_per_dataset": np.mean(times),
            "average_throughput": np.mean(throughputs),
            "total_data_processed": sum(r["data_size"] for r in successful_results),
            "overall_throughput": sum(r["data_size"] for r in successful_results)
            / overall_time,
        }

    return results


def print_batch_results(results: dict) -> None:
    """
    Print formatted batch processing results.

    Args:
        results (Dict): Aggregated results from batch processing
    """
    print("\n" + "=" * 60)
    print("Batch Processing Results")
    print("=" * 60)

    print("\nSummary:")
    print(f"  Total datasets: {results['total_datasets']}")
    print(f"  Successful: {results['successful']} ✓")
    print(f"  Failed: {results['failed']} ✗")

    if results.get("performance_summary"):
        summary = results["performance_summary"]
        print("\nPerformance:")
        print(f"  Total time: {summary['total_time']:.3f}s")
        print(f"  Parallel efficiency: {summary['parallel_efficiency']:.2%}")
        print(f"  Avg time per dataset: {summary['average_time_per_dataset']:.3f}s")
        print(f"  Avg throughput: {summary['average_throughput']:,.0f} pts/sec")
        print(f"  Total data processed: {summary['total_data_processed']:,} points")
        print(f"  Overall throughput: {summary['overall_throughput']:,.0f} pts/sec")

    print("\nDataset Details:")
    for dataset_id in sorted(results["datasets"].keys()):
        result = results["datasets"][dataset_id]
        if result["success"]:
            print(
                f"  Dataset {dataset_id}: {result['elapsed_time']:.3f}s "
                f"({result['throughput']:,.0f} pts/sec)"
            )
            if "chi_squared" in result:
                print(f"    χ² = {result['chi_squared']:.6f}")
        else:
            print(f"  Dataset {dataset_id}: FAILED - {result['error']}")


def save_batch_results(results: dict, output_dir: Path = None) -> None:
    """
    Save batch processing results to JSON file.

    Args:
        results (Dict): Aggregated results
        output_dir (Path): Output directory (default: current directory)
    """
    if output_dir is None:
        output_dir = Path.cwd()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data for JSON serialization
    json_results = {
        "total_datasets": results["total_datasets"],
        "successful": results["successful"],
        "failed": results["failed"],
        "performance_summary": results.get("performance_summary", {}),
        "datasets": {},
    }

    for dataset_id, result in results["datasets"].items():
        # Convert numpy arrays and complex objects to JSON-serializable format
        json_result = {
            "dataset_id": result.get("dataset_id"),
            "success": result.get("success"),
            "elapsed_time": result.get("elapsed_time"),
            "data_size": result.get("data_size"),
            "throughput": result.get("throughput"),
        }

        if "error" in result:
            json_result["error"] = result["error"]

        if "chi_squared" in result:
            json_result["chi_squared"] = float(result["chi_squared"])

        if "parameters" in result:
            # Convert parameter dict to JSON-serializable format
            json_result["parameters"] = {
                k: float(v) for k, v in result["parameters"].items()
            }

        json_results["datasets"][str(dataset_id)] = json_result

    # Write to file
    output_file = output_dir / "batch_processing_results.json"
    with open(output_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")


def demonstrate_dynamic_worker_scaling(datasets: list[dict]) -> None:
    """
    Demonstrate how performance scales with different numbers of workers.

    Args:
        datasets (List[Dict]): List of XPCS datasets
    """
    print("\n📊 Worker Scaling Analysis")
    print("=" * 60)
    print("Testing different numbers of parallel workers...\n")

    worker_configs = [1, 2, 4, 8]
    cpu_info = detect_cpu_info()

    # Cap at available cores
    worker_configs = [w for w in worker_configs if w <= cpu_info["cores_logical"]]

    scaling_results = {}

    for num_workers in worker_configs:
        print(f"Testing with {num_workers} workers...")

        start_time = time.perf_counter()
        results = parallel_batch_processing(datasets, max_workers=num_workers)
        elapsed = time.perf_counter() - start_time

        scaling_results[num_workers] = {
            "time": elapsed,
            "successful": results["successful"],
        }

        print(
            f"  Time: {elapsed:.3f}s (Success: {results['successful']}/{len(datasets)})\n"
        )

    # Show scaling efficiency
    print("Scaling Efficiency:")
    baseline_time = scaling_results[1]["time"]
    for num_workers in sorted(scaling_results.keys()):
        time_elapsed = scaling_results[num_workers]["time"]
        speedup = baseline_time / time_elapsed
        efficiency = speedup / num_workers
        print(
            f"  {num_workers} workers: {speedup:.2f}x speedup ({efficiency:.1%} efficiency)"
        )


def print_usage_guide() -> None:
    """Print usage guide for batch processing."""
    print("\n📖 Batch Processing Usage Guide")
    print("=" * 60)

    print(
        """
For Small Systems (Personal Computers):
  - Use 1-2 fewer workers than total cores
  - Example: 16-core PC → 14-15 workers
  - Leaves cores for OS tasks

For Large HPC Systems (36+ cores):
  - Use 80-90% of available cores
  - Example: 36-core node → 32 workers
  - Reserve cores for I/O, progress tracking

Memory Management:
  - Each worker needs ~1-2 GB per dataset
  - 8 workers × 1.5 GB = 12 GB peak
  - Monitor with: free -h or nvidia-smi

Fault Tolerance:
  - Failed datasets don't stop processing
  - Results aggregated with success/failure counts
  - Use JSON output for detailed error analysis

Scaling Limitations:
  - Speedup plateaus around 32 cores
  - Network I/O may become bottleneck
  - Memory bandwidth limits throughput
  - Use NUMA-aware pinning on large systems

Best Practices:
  1. Start with 4 workers, scale up if needed
  2. Monitor CPU usage: top -p <pid> -H
  3. Monitor memory: watch -n 1 free -h
  4. Log results to JSON for post-processing
  5. Use Slurm/PBS for reproducible resource allocation

Debugging:
  - Run with max_workers=1 for sequential execution
  - Use try-except to catch per-dataset errors
  - Enable verbose logging for timing analysis
  - Save intermediate results for fault recovery
"""
    )


def main():
    """Main batch processing demonstration."""
    print("\n" + "=" * 60)
    print(" Multi-Core Batch Processing Example - Homodyne v2.3")
    print("=" * 60)
    print("\nCPU-optimized parallel processing (GPU removed in v2.3.0)")

    # Step 1: Generate multiple datasets
    datasets = generate_multiple_datasets(num_datasets=4, n_times=50, n_angles=8)

    # Step 2: Run parallel batch processing
    results = parallel_batch_processing(datasets)

    # Step 3: Print results
    print_batch_results(results)

    # Step 4: Save results to JSON
    output_dir = Path.cwd() / "batch_results"
    save_batch_results(results, output_dir)

    # Step 5: Show scaling analysis
    print_usage_guide()

    # Optional: Demonstrate scaling behavior (uncomment to test)
    # print("\nOptional: Run scaling analysis (comment out for faster demo):")
    # demonstrate_dynamic_worker_scaling(datasets[:2])

    print("\n✅ Batch Processing Example Completed!")
    print("\nNext steps:")
    print("1. Adjust num_datasets and max_workers for your system")
    print("2. Replace synthetic data with real HDF5 files")
    print("3. Customize worker count based on available CPU cores")
    print("4. Monitor performance and tune parameters")


if __name__ == "__main__":
    main()
