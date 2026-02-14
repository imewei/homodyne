#!/usr/bin/env python3
"""Diagnostic script to compare CMC vs NLSQ physics implementations.

This script computes C2 heatmaps using:
1. NLSQ physics (jax_backend.py element-wise mode)
2. CMC physics (physics_cmc.py)

Using identical fixed parameters to verify the physics models are consistent.

Key Finding: The integration approaches differ!
- NLSQ: Single trapezoid approximation for each (t1, t2) pair
- CMC: Cumulative trapezoid sum over time grid, then lookup

This can cause different C2 values especially when alpha ≠ 0 or beta ≠ 0.
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

from homodyne.core.jax_backend import compute_g2_scaled as nlsq_compute_g2  # noqa: E402
from homodyne.core.physics_cmc import compute_g1_total as cmc_compute_g1  # noqa: E402


def create_time_grid(n_times: int, dt: float) -> np.ndarray:
    """Create uniform time grid."""
    return np.arange(n_times) * dt


def compute_c2_nlsq(
    params: np.ndarray,
    t: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    dt: float,
    contrasts: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Compute C2 using NLSQ physics (meshgrid mode).

    Returns C2 heatmap with shape (n_phi, n_times, n_times).
    """
    # NLSQ uses meshgrid approach for small arrays
    # Create meshgrids
    t1_grid, t2_grid = np.meshgrid(t, t, indexing="ij")

    # Compute for each phi angle with its contrast/offset
    c2_all = []
    for _i, (phi_val, contrast, offset) in enumerate(zip(phi, contrasts, offsets, strict=True)):
        c2_phi = nlsq_compute_g2(
            jnp.array(params),
            jnp.array(t1_grid),
            jnp.array(t2_grid),
            jnp.array([phi_val]),
            float(q),
            float(L),
            float(contrast),
            float(offset),
            float(dt),
        )
        c2_all.append(np.asarray(c2_phi[0]))  # Remove phi dimension

    return np.stack(c2_all, axis=0)


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
    """Compute C2 using CMC physics (element-wise mode).

    Returns C2 heatmap with shape (n_phi, n_times, n_times).
    """
    n_times = len(t)

    # Create time grid for CMC (required)
    time_grid = jnp.linspace(0.0, t[-1], n_times)

    # CMC computes g1 for all phi at once: shape (n_phi, n_points)
    # For C2 heatmap, we need meshgrid expansion
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


def compare_integration_methods(
    D0: float, alpha: float, D_offset: float, dt: float, n_steps: int
) -> dict:
    """Compare integration methods for diffusion integral.

    For integral from t=0 to t=n_steps*dt of D(t) dt where D(t) = D0*t^alpha + D_offset

    Returns dict with:
    - nlsq_method: Single trapezoid approximation
    - cmc_method: Cumulative trapezoid sum
    - exact: Analytical solution (when possible)
    - relative_error: Between methods
    """
    t_end = n_steps * dt

    # Time grid
    t_grid = np.linspace(0, t_end, n_steps + 1)
    epsilon = 1e-10

    # D(t) values with epsilon to handle t=0
    D_values = D0 * (t_grid + epsilon) ** alpha + D_offset

    # NLSQ method: Single trapezoid
    # frame_diff = n_steps, D_integral = n_steps * 0.5 * (D(0) + D(t_end))
    D_t0 = D0 * epsilon**alpha + D_offset
    D_tend = D0 * (t_end + epsilon) ** alpha + D_offset
    nlsq_integral = n_steps * 0.5 * (D_t0 + D_tend)

    # CMC method: Cumulative trapezoid
    # Sum of 0.5 * (D[i] + D[i+1]) for i = 0 to n_steps-1
    trap_avg = 0.5 * (D_values[:-1] + D_values[1:])
    cmc_integral = np.sum(trap_avg)

    # Analytical solution (for alpha != -1)
    # ∫ (D0*t^α + D_offset) dt = D0*t^(α+1)/(α+1) + D_offset*t
    if abs(alpha + 1) > 1e-10:
        # Use t+epsilon for consistency
        exact_end = D0 * (t_end + epsilon) ** (alpha + 1) / (alpha + 1) + D_offset * (
            t_end + epsilon
        )
        exact_start = D0 * epsilon ** (alpha + 1) / (alpha + 1) + D_offset * epsilon
        exact_integral = (exact_end - exact_start) / dt  # Normalize to "steps"
    else:
        exact_integral = None

    relative_error = abs(nlsq_integral - cmc_integral) / max(
        abs(nlsq_integral), abs(cmc_integral), 1e-10
    )

    return {
        "nlsq_method": nlsq_integral,
        "cmc_method": cmc_integral,
        "exact": exact_integral,
        "relative_error": relative_error,
        "n_steps": n_steps,
        "alpha": alpha,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare CMC vs NLSQ physics")
    parser.add_argument(
        "--params",
        type=str,
        default="19230,-1.063,879,0.078,-0.61,-0.0089,5.8",
        help="Comma-separated physical parameters: D0,alpha,D_offset,gamma_dot_t0,beta,gamma_dot_t_offset,phi0",
    )
    parser.add_argument("--n-times", type=int, default=100, help="Number of time points")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step (seconds)")
    parser.add_argument("--q", type=float, default=0.01, help="Wave vector magnitude")
    parser.add_argument("--L", type=float, default=1e6, help="Stator-rotor gap (Angstrom)")
    parser.add_argument(
        "--phi",
        type=str,
        default="0,45,90",
        help="Comma-separated phi angles (degrees)",
    )
    parser.add_argument(
        "--contrasts",
        type=str,
        default="0.5,0.5,0.5",
        help="Comma-separated contrast values",
    )
    parser.add_argument(
        "--offsets",
        type=str,
        default="1.0,1.0,1.0",
        help="Comma-separated offset values",
    )
    parser.add_argument(
        "--output", type=str, default="cmc_vs_nlsq_comparison.png", help="Output file"
    )
    parser.add_argument(
        "--integration-test",
        action="store_true",
        help="Run integration method comparison test",
    )

    args = parser.parse_args()

    # Parse parameters
    params = np.array([float(x) for x in args.params.split(",")])
    phi = np.array([float(x) for x in args.phi.split(",")])
    contrasts = np.array([float(x) for x in args.contrasts.split(",")])
    offsets = np.array([float(x) for x in args.offsets.split(",")])

    print("=" * 60)
    print("CMC vs NLSQ Physics Comparison")
    print("=" * 60)
    print(f"Parameters: {params}")
    print(f"  D0={params[0]:.4g}, alpha={params[1]:.4g}, D_offset={params[2]:.4g}")
    if len(params) >= 7:
        print(
            f"  gamma_dot_t0={params[3]:.4g}, beta={params[4]:.4g}, "
            f"gamma_dot_t_offset={params[5]:.4g}, phi0={params[6]:.4g}"
        )
    print(f"n_times={args.n_times}, dt={args.dt}, q={args.q}, L={args.L}")
    print(f"phi angles: {phi}")
    print(f"contrasts: {contrasts}")
    print(f"offsets: {offsets}")
    print()

    # Run integration test first
    if args.integration_test or True:  # Always run
        print("Integration Method Comparison Test")
        print("-" * 40)
        alpha = params[1]
        for n_steps in [10, 50, 100, 500]:
            result = compare_integration_methods(
                params[0], alpha, params[2], args.dt, n_steps
            )
            print(
                f"n_steps={n_steps:4d}: "
                f"NLSQ={result['nlsq_method']:12.4g}, "
                f"CMC={result['cmc_method']:12.4g}, "
                f"rel_err={result['relative_error']:.2e}"
            )
        print()

    # Create time grid
    t = create_time_grid(args.n_times, args.dt)

    # Compute C2 using both methods
    print("Computing C2 using NLSQ physics...")
    c2_nlsq = compute_c2_nlsq(
        params, t, phi, args.q, args.L, args.dt, contrasts, offsets
    )

    print("Computing C2 using CMC physics...")
    c2_cmc = compute_c2_cmc(
        params, t, phi, args.q, args.L, args.dt, contrasts, offsets
    )

    # Compare
    print()
    print("Comparison Results")
    print("-" * 40)
    diff = c2_nlsq - c2_cmc
    rel_diff = np.abs(diff) / np.maximum(np.abs(c2_nlsq), 1e-10)

    print(f"C2 NLSQ: min={c2_nlsq.min():.6g}, max={c2_nlsq.max():.6g}")
    print(f"C2 CMC:  min={c2_cmc.min():.6g}, max={c2_cmc.max():.6g}")
    print(f"Absolute diff: min={diff.min():.6g}, max={diff.max():.6g}")
    print(f"Relative diff: mean={rel_diff.mean():.6g}, max={rel_diff.max():.6g}")
    print()

    # Per-angle statistics
    for i, phi_val in enumerate(phi):
        diff_i = c2_nlsq[i] - c2_cmc[i]
        rel_i = np.abs(diff_i) / np.maximum(np.abs(c2_nlsq[i]), 1e-10)
        print(
            f"phi={phi_val:5.1f}°: "
            f"abs_diff_max={np.abs(diff_i).max():.4g}, "
            f"rel_diff_mean={rel_i.mean():.4g}"
        )

    # Plot comparison
    fig, axes = plt.subplots(3, len(phi), figsize=(5 * len(phi), 12))
    if len(phi) == 1:
        axes = axes.reshape(-1, 1)

    for i, phi_val in enumerate(phi):
        # NLSQ
        im0 = axes[0, i].imshow(
            c2_nlsq[i], origin="lower", extent=[0, t[-1], 0, t[-1]], cmap="viridis"
        )
        axes[0, i].set_title(f"NLSQ C2 (phi={phi_val}°)")
        axes[0, i].set_xlabel("t2 (s)")
        axes[0, i].set_ylabel("t1 (s)")
        plt.colorbar(im0, ax=axes[0, i])

        # CMC
        im1 = axes[1, i].imshow(
            c2_cmc[i], origin="lower", extent=[0, t[-1], 0, t[-1]], cmap="viridis"
        )
        axes[1, i].set_title(f"CMC C2 (phi={phi_val}°)")
        axes[1, i].set_xlabel("t2 (s)")
        axes[1, i].set_ylabel("t1 (s)")
        plt.colorbar(im1, ax=axes[1, i])

        # Difference
        im2 = axes[2, i].imshow(
            c2_nlsq[i] - c2_cmc[i],
            origin="lower",
            extent=[0, t[-1], 0, t[-1]],
            cmap="RdBu_r",
        )
        axes[2, i].set_title("Difference (NLSQ - CMC)")
        axes[2, i].set_xlabel("t2 (s)")
        axes[2, i].set_ylabel("t1 (s)")
        plt.colorbar(im2, ax=axes[2, i])

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot to: {args.output}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if rel_diff.max() < 0.01:
        print("✓ CMC and NLSQ physics are consistent (max relative diff < 1%)")
    elif rel_diff.max() < 0.1:
        print(
            f"⚠ CMC and NLSQ physics show moderate differences "
            f"(max relative diff = {rel_diff.max():.1%})"
        )
    else:
        print(
            f"✗ CMC and NLSQ physics show SIGNIFICANT differences "
            f"(max relative diff = {rel_diff.max():.1%})"
        )
        print()
        print("Root cause: Different integration approaches!")
        print("  NLSQ: Single trapezoid for each (t1, t2) pair")
        print("  CMC:  Cumulative trapezoid sum over time grid")
        print()
        print("For D(t) = D0*t^alpha + D_offset with alpha != 0,")
        print("these methods give different integral values.")


if __name__ == "__main__":
    main()
