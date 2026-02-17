#!/usr/bin/env python3
"""CMC Performance Baseline Script.

This script captures baseline performance metrics for CMC analysis
to enable before/after comparison during optimization work.

Usage:
    python scripts/benchmarks/cmc_baseline.py --dataset small
    python scripts/benchmarks/cmc_baseline.py --dataset medium --output results.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class BaselineMetrics:
    """Container for CMC baseline metrics."""

    timestamp: str
    dataset_size: str
    n_points: int
    n_phi: int
    analysis_mode: str
    n_shards: int
    n_chains: int
    num_warmup: int
    num_samples: int
    per_shard_timeout: int
    total_runtime_seconds: float
    estimated_runtime_seconds: float
    runtime_accuracy_percent: float
    shards_succeeded: int
    shards_failed: int
    success_rate: float
    convergence_status: str
    max_r_hat: float
    min_ess: float
    num_divergences: int
    error_categories: dict[str, int]


def generate_synthetic_data(
    n_points: int = 100000,
    n_phi: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic XPCS-like data for benchmarking."""
    rng = np.random.default_rng(seed)

    # Generate time grid
    n_t = int(np.sqrt(n_points / n_phi))
    t_vals = np.linspace(0, 50, n_t)
    t1_grid, t2_grid = np.meshgrid(t_vals, t_vals, indexing="ij")
    t1_base = t1_grid.ravel()
    t2_base = t2_grid.ravel()

    # Expand for multiple phi angles
    phi_angles = np.linspace(0, np.pi / 2, n_phi)
    t1_list = []
    t2_list = []
    phi_list = []

    for phi_val in phi_angles:
        t1_list.append(t1_base)
        t2_list.append(t2_base)
        phi_list.append(np.full_like(t1_base, phi_val))

    t1 = np.concatenate(t1_list)
    t2 = np.concatenate(t2_list)
    phi = np.concatenate(phi_list)

    # Generate synthetic C2 correlation data
    tau = np.abs(t2 - t1)
    # Simple exponential decay model
    data = np.exp(-tau / 10.0) + 0.1 * rng.standard_normal(len(tau))
    data = np.clip(data, 0.0, 2.0)

    return data, t1, t2, phi


def run_baseline_benchmark(
    dataset_size: str = "small",
    analysis_mode: str = "static",
    output_path: Path | None = None,
) -> BaselineMetrics:
    """Run a baseline benchmark and capture metrics."""
    from homodyne.config.parameter_space import ParameterSpace
    from homodyne.optimization.cmc import fit_mcmc_jax
    from homodyne.optimization.cmc.config import CMCConfig

    # Dataset size presets
    size_presets = {
        "tiny": {"n_points": 10000, "n_phi": 1},
        "small": {"n_points": 50000, "n_phi": 3},
        "medium": {"n_points": 200000, "n_phi": 3},
        "large": {"n_points": 1000000, "n_phi": 3},
    }

    if dataset_size not in size_presets:
        raise ValueError(f"Unknown dataset size: {dataset_size}")

    preset = size_presets[dataset_size]
    print(f"Generating {dataset_size} synthetic dataset...")
    data, t1, t2, phi = generate_synthetic_data(
        n_points=preset["n_points"],
        n_phi=preset["n_phi"],
    )

    # Configure CMC
    cmc_config_dict = {
        "enable": True,
        "sharding": {"strategy": "stratified", "max_points_per_shard": 50000},
        "per_shard_mcmc": {
            "num_warmup": 100,  # Reduced for benchmark
            "num_samples": 200,
            "num_chains": 2,
        },
        "per_shard_timeout": 300,  # 5 min for benchmark
    }

    config = CMCConfig.from_dict(cmc_config_dict)

    # Create parameter space
    ps_config = {
        "D0": {"min": 1e8, "max": 1e12, "initial": 1e10},
        "alpha": {"min": -1.0, "max": 0.0, "initial": -0.5},
        "D_offset": {"min": 0.0, "max": 1e11, "initial": 1e9},
    }
    parameter_space = ParameterSpace.from_config(ps_config, analysis_mode)

    initial_values = {"D0": 1e10, "alpha": -0.5, "D_offset": 1e9}

    print(f"Running CMC benchmark (mode={analysis_mode})...")
    start_time = time.perf_counter()

    try:
        result = fit_mcmc_jax(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.01,
            L=2_000_000.0,
            analysis_mode=analysis_mode,
            cmc_config=cmc_config_dict,
            initial_values=initial_values,
            parameter_space=parameter_space,
            progress_bar=True,
        )
        success = True
    except Exception as e:
        print(f"CMC failed: {e}")
        success = False
        result = None

    total_runtime = time.perf_counter() - start_time

    # Extract metrics
    metrics = BaselineMetrics(
        timestamp=datetime.now().isoformat(),
        dataset_size=dataset_size,
        n_points=len(data),
        n_phi=preset["n_phi"],
        analysis_mode=analysis_mode,
        n_shards=preset["n_phi"],  # Stratified sharding
        n_chains=config.num_chains,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        per_shard_timeout=config.per_shard_timeout,
        total_runtime_seconds=total_runtime,
        estimated_runtime_seconds=0.0,  # Would need to parse from logs
        runtime_accuracy_percent=0.0,
        shards_succeeded=preset["n_phi"] if success else 0,
        shards_failed=0 if success else preset["n_phi"],
        success_rate=1.0 if success else 0.0,
        convergence_status=result.convergence_status if result else "failed",
        max_r_hat=max(result.r_hat.values()) if result else float("nan"),
        min_ess=min(result.ess_bulk.values()) if result else float("nan"),
        num_divergences=result.divergences if result else 0,
        error_categories={},
    )

    print(f"\n{'=' * 60}")
    print("BASELINE METRICS")
    print(f"{'=' * 60}")
    print(f"Dataset: {dataset_size} ({metrics.n_points:,} points, {metrics.n_phi} phi)")
    print(f"Runtime: {metrics.total_runtime_seconds:.1f}s")
    print(f"Convergence: {metrics.convergence_status}")
    print(f"R-hat (max): {metrics.max_r_hat:.3f}")
    print(f"ESS (min): {metrics.min_ess:.0f}")
    print(f"Divergences: {metrics.num_divergences}")
    print(f"{'=' * 60}\n")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"Metrics saved to {output_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="CMC Performance Baseline")
    parser.add_argument(
        "--dataset",
        choices=["tiny", "small", "medium", "large"],
        default="small",
        help="Dataset size preset",
    )
    parser.add_argument(
        "--mode",
        choices=["static", "laminar_flow"],
        default="static",
        help="Analysis mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file path",
    )

    args = parser.parse_args()

    run_baseline_benchmark(
        dataset_size=args.dataset,
        analysis_mode=args.mode,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
