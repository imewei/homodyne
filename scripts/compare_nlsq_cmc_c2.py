#!/usr/bin/env python3
"""Compare NLSQ C2 heatmaps with CMC physics predictions.

This script:
1. Loads NLSQ best-fit parameters and theoretical C2
2. Computes C2 using CMC physics with the same parameters
3. Compares the two to verify physics consistency
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from homodyne.core.physics_cmc import compute_g1_total as cmc_compute_g1


def load_nlsq_results(results_dir: str) -> dict:
    """Load NLSQ results from directory."""
    import json

    results_dir = Path(results_dir)

    # Load parameters
    with open(results_dir / "parameters.json") as f:
        params_data = json.load(f)

    # Load fitted data
    fitted_data = np.load(results_dir / "fitted_data.npz", allow_pickle=True)

    return {
        "params": params_data["parameters"],
        "c2_theoretical": fitted_data["c2_theoretical_scaled"],
        "c2_exp": fitted_data["c2_exp"],
        "per_angle_scaling": fitted_data["per_angle_scaling_solver"],
        "phi_angles": fitted_data["phi_angles"],
        "t1": fitted_data["t1"],
        "t2": fitted_data["t2"],
        "q": fitted_data["q"][0],
    }


def compute_c2_cmc(
    params: np.ndarray,
    t: np.ndarray,
    phi_unique: np.ndarray,
    q: float,
    L: float,
    dt: float,
    contrasts: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Compute C2 using CMC physics.

    Returns C2 heatmap with shape (n_phi, n_times, n_times).
    """
    # CRITICAL FIX (Dec 2025): Use proper dt-based time grid, NOT data length
    # The physics integration depends on having the correct grid density
    # With dt=0.1s and t_max=100s, we need 1001 points, not len(t) which may be subsampled
    n_times = len(t)  # For reshaping output
    t_max = float(t[-1])
    n_time_points = int(round(t_max / dt)) + 1
    time_grid = jnp.linspace(0.0, t_max, n_time_points)

    # Create meshgrid for all time pairs
    t1_grid, t2_grid = np.meshgrid(t, t, indexing="ij")
    t1_flat = t1_grid.flatten()
    t2_flat = t2_grid.flatten()

    # Compute g1 using CMC physics
    g1_all_phi = cmc_compute_g1(
        jnp.array(params),
        jnp.array(t1_flat),
        jnp.array(t2_flat),
        jnp.array(phi_unique),
        float(q),
        float(L),
        float(dt),
        time_grid=time_grid,
    )

    # g1_all_phi shape: (n_phi, n_points)
    g1_np = np.asarray(g1_all_phi)

    # Reshape to (n_phi, n_times, n_times)
    n_phi = len(phi_unique)
    g1_reshaped = g1_np.reshape(n_phi, n_times, n_times)

    # Apply C2 = contrast × g1² + offset for each phi
    c2_all = []
    for i in range(n_phi):
        c2_phi = contrasts[i] * g1_reshaped[i] ** 2 + offsets[i]
        c2_all.append(c2_phi)

    return np.stack(c2_all, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Compare NLSQ and CMC C2 heatmaps")
    parser.add_argument(
        "--nlsq-dir",
        type=str,
        default="/home/wei/Documents/Projects/data/C020/homodyne_results/nlsq",
        help="NLSQ results directory",
    )
    parser.add_argument(
        "--L",
        type=float,
        default=2000000,
        help="Stator-rotor gap [Angstrom]",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step [seconds]",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/nlsq_vs_cmc_comparison.png",
        help="Output file path",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=100,
        help="Subsample time points for faster computation (use 0 for full data)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("NLSQ vs CMC Physics Comparison")
    print("=" * 70)

    # Load NLSQ results
    print("\nLoading NLSQ results...")
    nlsq = load_nlsq_results(args.nlsq_dir)

    # Extract parameters
    params = nlsq["params"]
    print(f"\nNLSQ Parameters:")
    print(f"  D0 = {params['D0']['value']:.4f}")
    print(f"  alpha = {params['alpha']['value']:.4f}")
    print(f"  D_offset = {params['D_offset']['value']:.4f}")
    print(f"  gamma_dot_t0 = {params['gamma_dot_t0']['value']:.6f}")
    print(f"  beta = {params['beta']['value']:.4f}")
    print(f"  gamma_dot_t_offset = {params['gamma_dot_t_offset']['value']:.6f}")
    print(f"  phi0 = {params['phi0']['value']:.4f}")

    # Build parameter array
    param_array = np.array(
        [
            params["D0"]["value"],
            params["alpha"]["value"],
            params["D_offset"]["value"],
            params["gamma_dot_t0"]["value"],
            params["beta"]["value"],
            params["gamma_dot_t_offset"]["value"],
            params["phi0"]["value"],
        ]
    )

    # Get data arrays
    t = nlsq["t1"]
    phi_angles = nlsq["phi_angles"]
    q = nlsq["q"]
    per_angle_scaling = nlsq["per_angle_scaling"]
    c2_nlsq = nlsq["c2_theoretical"]

    print(f"\nData shape:")
    print(f"  n_times = {len(t)}")
    print(f"  phi_angles = {phi_angles}")
    print(f"  q = {q}")
    print(f"  dt = {args.dt}")
    print(f"  L = {args.L}")

    print(f"\nPer-angle scaling:")
    for i, phi in enumerate(phi_angles):
        print(
            f"  phi={phi:7.3f}°: contrast={per_angle_scaling[i,0]:.6f}, "
            f"offset={per_angle_scaling[i,1]:.6f}"
        )

    # Subsample for faster computation
    if args.subsample > 0 and len(t) > args.subsample:
        step = len(t) // args.subsample
        t_sub = t[::step]
        c2_nlsq_sub = c2_nlsq[:, ::step, ::step]
        print(f"\nSubsampling: {len(t)} → {len(t_sub)} time points")
    else:
        t_sub = t
        c2_nlsq_sub = c2_nlsq
        print(f"\nUsing full data: {len(t)} time points")

    # Compute C2 using CMC physics
    print("\nComputing C2 using CMC physics...")
    contrasts = per_angle_scaling[:, 0]
    offsets = per_angle_scaling[:, 1]

    c2_cmc = compute_c2_cmc(
        param_array,
        t_sub,
        phi_angles,
        q,
        args.L,
        args.dt,
        contrasts,
        offsets,
    )

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    diff = c2_nlsq_sub - c2_cmc
    rel_diff = np.abs(diff) / np.maximum(np.abs(c2_nlsq_sub), 1e-10)

    print(f"\nGlobal Statistics:")
    print(f"  NLSQ C2 range: [{c2_nlsq_sub.min():.6g}, {c2_nlsq_sub.max():.6g}]")
    print(f"  CMC C2 range:  [{c2_cmc.min():.6g}, {c2_cmc.max():.6g}]")
    print(f"  Absolute diff: min={diff.min():.6g}, max={diff.max():.6g}")
    print(f"  Relative diff: mean={rel_diff.mean():.2%}, max={rel_diff.max():.2%}")

    print(f"\nPer-angle Statistics:")
    for i, phi in enumerate(phi_angles):
        diff_i = c2_nlsq_sub[i] - c2_cmc[i]
        rel_i = np.abs(diff_i) / np.maximum(np.abs(c2_nlsq_sub[i]), 1e-10)
        print(
            f"  phi={phi:7.3f}°: "
            f"NLSQ=[{c2_nlsq_sub[i].min():.4g}, {c2_nlsq_sub[i].max():.4g}], "
            f"CMC=[{c2_cmc[i].min():.4g}, {c2_cmc[i].max():.4g}], "
            f"rel_diff_mean={rel_i.mean():.2%}"
        )

    # Plot comparison
    n_phi = len(phi_angles)
    fig, axes = plt.subplots(4, n_phi, figsize=(5 * n_phi, 16))

    t_extent = [0, t_sub[-1], 0, t_sub[-1]]

    for i, phi in enumerate(phi_angles):
        # NLSQ theoretical
        im0 = axes[0, i].imshow(
            c2_nlsq_sub[i], origin="lower", extent=t_extent, cmap="viridis"
        )
        axes[0, i].set_title(f"NLSQ C2 (phi={phi:.1f}°)")
        axes[0, i].set_xlabel("t2 (s)")
        axes[0, i].set_ylabel("t1 (s)")
        plt.colorbar(im0, ax=axes[0, i])

        # CMC predicted
        im1 = axes[1, i].imshow(
            c2_cmc[i], origin="lower", extent=t_extent, cmap="viridis"
        )
        axes[1, i].set_title(f"CMC C2 (phi={phi:.1f}°)")
        axes[1, i].set_xlabel("t2 (s)")
        axes[1, i].set_ylabel("t1 (s)")
        plt.colorbar(im1, ax=axes[1, i])

        # Absolute difference
        im2 = axes[2, i].imshow(
            diff[i], origin="lower", extent=t_extent, cmap="RdBu_r"
        )
        axes[2, i].set_title(f"Diff (NLSQ - CMC)")
        axes[2, i].set_xlabel("t2 (s)")
        axes[2, i].set_ylabel("t1 (s)")
        plt.colorbar(im2, ax=axes[2, i])

        # Relative difference
        im3 = axes[3, i].imshow(
            rel_diff[i] * 100, origin="lower", extent=t_extent, cmap="Reds", vmax=1.0
        )
        axes[3, i].set_title(f"Rel. Diff (%) - max={rel_diff[i].max()*100:.2f}%")
        axes[3, i].set_xlabel("t2 (s)")
        axes[3, i].set_ylabel("t1 (s)")
        plt.colorbar(im3, ax=axes[3, i], label="%")

    plt.suptitle(
        f"NLSQ vs CMC Physics Comparison\n"
        f"D0={param_array[0]:.1f}, alpha={param_array[1]:.3f}, "
        f"gamma_dot_t0={param_array[3]:.4f}, beta={param_array[4]:.3f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison to: {args.output}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if rel_diff.max() < 0.01:
        print("✓ NLSQ and CMC physics are CONSISTENT (max relative diff < 1%)")
    elif rel_diff.max() < 0.05:
        print(
            f"⚠ NLSQ and CMC physics show MINOR differences "
            f"(max relative diff = {rel_diff.max():.1%})"
        )
    else:
        print(
            f"✗ NLSQ and CMC physics show SIGNIFICANT differences "
            f"(max relative diff = {rel_diff.max():.1%})"
        )
        print("\nPossible causes:")
        print("  1. Different integration approaches (single trapezoid vs cumulative)")
        print("  2. Different handling of t=0 singularity when alpha < 0")
        print("  3. Different time grid construction")


if __name__ == "__main__":
    main()
