#!/usr/bin/env python3
"""Print basic diagonal overlay statistics for solver vs post-hoc C₂ surfaces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from homodyne.viz.diagnostics import compute_diagonal_overlay_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare experimental, solver-evaluated, and post-hoc C₂ diagonals "
            "using a fitted_data.npz produced by save_nlsq_results."
        ),
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to fitted_data.npz",
    )
    parser.add_argument(
        "--phi-index",
        type=int,
        default=0,
        help="Angle index to inspect (default: 0)",
    )
    args = parser.parse_args()

    data = np.load(args.npz_path)
    solver = data.get("c2_solver_scaled")
    if solver is None:
        raise SystemExit(
            "This NPZ does not contain c2_solver_scaled. Re-run save_nlsq_results on the latest build."
        )

    posthoc = data["c2_theoretical_scaled"]
    raw = data["c2_exp"]

    overlay = compute_diagonal_overlay_stats(
        raw,
        solver,
        posthoc,
        phi_index=args.phi_index,
    )

    print(f"φ index        : {overlay.phi_index}")
    print(f"Raw variance   : {overlay.raw_variance:.6e}")
    print(f"Solver variance: {overlay.solver_variance:.6e}")
    print(f"Post-hoc var   : {overlay.posthoc_variance:.6e}")
    print(f"Solver RMSE    : {overlay.solver_rmse:.6e}")
    print(f"Post-hoc RMSE  : {overlay.posthoc_rmse:.6e}")


if __name__ == "__main__":
    main()
