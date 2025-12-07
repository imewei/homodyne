#!/usr/bin/env python3
"""
Gradient Diagnostics CLI Tool
==============================

Analyzes NLSQ optimization results to detect gradient imbalance and provides
recommendations for parameter scaling (x_scale_map).

Usage:
------
    # Analyze results from default location
    python scripts/diagnose_gradients.py

    # Analyze specific result directory
    python scripts/diagnose_gradients.py --results-dir /path/to/homodyne_results/nlsq

    # Save recommended config updates
    python scripts/diagnose_gradients.py --output config_updates.yaml

Author: Homodyne Development Team
Date: 2025-11-13
Version: 1.0.0
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]


@dataclass
class NLSQData:
    """Lightweight container for NLSQ fitted data."""

    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2: np.ndarray
    q: float
    L: float
    dt: float
    per_angle_scaling_solver: np.ndarray | None = None


@dataclass
class NLSQConfig:
    """Placeholder for config data required by diagnostics."""

    raw: Dict[str, Any] | None = None


def load_nlsq_results(results_dir: Path) -> Tuple[Dict[str, float], NLSQData, NLSQConfig, str]:
    """Load NLSQ results from directory."""
    # Load parameters
    param_file = results_dir / "parameters.json"
    if not param_file.exists():
        raise FileNotFoundError(f"Parameters file not found: {param_file}")

    with open(param_file) as f:
        param_data = json.load(f)

    # Extract parameter values
    parameters = {
        name: data["value"] for name, data in param_data["parameters"].items()
    }

    # Load fitted data
    npz_file = results_dir / "fitted_data.npz"
    if not npz_file.exists():
        raise FileNotFoundError(f"Fitted data file not found: {npz_file}")

    npz_data = np.load(npz_file)

    per_angle_scaling = (
        npz_data["per_angle_scaling_solver"]
        if "per_angle_scaling_solver" in npz_data
        else None
    )
    data = NLSQData(
        phi=npz_data["phi_angles"],
        t1=npz_data["t1"],
        t2=npz_data["t2"],
        g2=npz_data["c2_exp"],
        q=float(npz_data["q"][0]),
        L=2_000_000.0,  # Default
        dt=float(npz_data["t1"][1] - npz_data["t1"][0]),
        per_angle_scaling_solver=per_angle_scaling
        if isinstance(per_angle_scaling, np.ndarray)
        else None,
    )

    # Load analysis results to get analysis_mode
    analysis_file = results_dir / "analysis_results_nlsq.json"
    analysis_mode = "laminar_flow"  # Default
    if analysis_file.exists():
        with open(analysis_file) as f:
            analysis_data = json.load(f)
            analysis_mode = analysis_data.get("analysis_mode", "laminar_flow")

    config = NLSQConfig()

    return parameters, data, config, analysis_mode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose gradient imbalance in NLSQ optimization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze results in default location
  python scripts/diagnose_gradients.py

  # Analyze specific directory
  python scripts/diagnose_gradients.py --results-dir ./homodyne_results/nlsq

  # Save recommended config
  python scripts/diagnose_gradients.py --output config_update.yaml

For more information, see:
  docs/troubleshooting/gradient_imbalance.md
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./homodyne_results/nlsq"),
        help="Path to NLSQ results directory (default: ./homodyne_results/nlsq)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Save recommended x_scale_map to YAML file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Gradient ratio threshold for warning (default: 10.0)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show recommendations",
    )

    args = parser.parse_args()

    # Check results directory exists
    if not args.results_dir.exists():
        print(
            f"Error: Results directory not found: {args.results_dir}", file=sys.stderr
        )
        print("\nPlease run NLSQ optimization first:", file=sys.stderr)
        print("  homodyne --config config.yaml --method nlsq", file=sys.stderr)
        sys.exit(1)

    # Load results
    try:
        print(f"Loading results from: {args.results_dir}")
        parameters, data, config, analysis_mode = load_nlsq_results(args.results_dir)
        print(f"  Analysis mode: {analysis_mode}")
        print(f"  Parameters: {list(parameters.keys())}")
        print()
    except Exception as e:
        print(f"Error loading results: {e}", file=sys.stderr)
        sys.exit(1)

    # Run gradient diagnostics
    try:
        from homodyne.optimization.gradient_diagnostics import (
            diagnose_gradient_imbalance,
            print_gradient_report,
        )

        if not args.quiet:
            print_gradient_report(parameters, data, config, analysis_mode)
        else:
            # Just print recommendations
            diag = diagnose_gradient_imbalance(
                parameters, data, config, analysis_mode, threshold=args.threshold
            )
            print(diag["recommendations"]["summary"])

        # Save output if requested
        if args.output:
            from homodyne.optimization.gradient_diagnostics import (
                compute_optimal_x_scale,
            )

            x_scale_map = compute_optimal_x_scale(
                parameters, data, config, analysis_mode
            )

            output_data = {
                "optimization": {
                    "nlsq": {
                        "x_scale_map": {k: float(v) for k, v in x_scale_map.items()}
                    }
                }
            }

            with open(args.output, "w") as f:
                yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

            print(f"\n✓ Saved recommended x_scale_map to: {args.output}")
            print("\nTo apply these fixes:")
            print(f"  1. Merge {args.output} into your configuration file")
            print("  2. Re-run optimization:")
            print("     homodyne --config your_config.yaml --method nlsq")
            print("  3. Verify improved fit quality in c2 heatmaps")

    except ImportError as e:
        print(
            f"Error: Could not import gradient diagnostics module: {e}", file=sys.stderr
        )
        print("\nMake sure homodyne is properly installed:", file=sys.stderr)
        print("  pip install -e .", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during gradient analysis: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
