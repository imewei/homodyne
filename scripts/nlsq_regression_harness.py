#!/usr/bin/env python3
"""
NLSQ Regression Test Harness

Compares NLSQ v0.3.0 behavior vs current HEAD to identify divergence.

Usage:
    1. With installed v0.3.0:
       python scripts/nlsq_regression_harness.py

    2. With local HEAD (override):
       NLSQ_PATH=/home/wei/Documents/GitHub/NLSQ python scripts/nlsq_regression_harness.py

    3. Both versions comparison:
       python scripts/nlsq_regression_harness.py --compare
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Suppress JAX warnings about TPU/GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Allow override of NLSQ path for testing different versions
NLSQ_PATH = os.environ.get("NLSQ_PATH")
if NLSQ_PATH:
    sys.path.insert(0, NLSQ_PATH)
    print(f"Using NLSQ from: {NLSQ_PATH}")

# Import after path adjustment
import jax.numpy as jnp  # noqa: E402
from nlsq import __version__ as nlsq_version  # noqa: E402
from nlsq import curve_fit  # noqa: E402

# Add homodyne to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_deterministic_data(
    seed: int = 42,
    n_phi: int = 5,
    n_t1: int = 15,
    n_t2: int = 15,
    noise_level: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate deterministic synthetic data using simplified model.

    Uses the same simplified exponential decay model as the fitting function
    to ensure clean optimizer behavior testing.
    """
    np.random.seed(seed)

    # Known ground truth
    ground_truth = {
        "D0": 100.0,  # Scaled for numerical stability
        "alpha": 0.5,
        "D_offset": 1.0,  # Scaled for numerical stability
        "contrast": 0.5,
        "offset": 1.0,
    }

    q = 0.01

    # Create coordinate arrays
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    t1 = np.linspace(0.1, 1.0, n_t1)
    t2 = np.linspace(0.1, 1.0, n_t2)

    # Generate theoretical data using simplified model
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")

    xdata_list = []
    ydata_list = []
    sigma_list = []

    for i, phi_val in enumerate(phi):
        n_points = n_t1 * n_t2
        t1_flat = T1.ravel()
        t2_flat = T2.ravel()

        # Simplified model: g2 = offset + contrast * exp(-D_eff * (t1 + t2))
        tau = np.sqrt(t1_flat * t2_flat)
        D_eff = (
            ground_truth["D0"] * np.power(tau + 0.01, ground_truth["alpha"] - 1.0)
            + ground_truth["D_offset"]
        )
        decay = np.exp(-D_eff * (t1_flat + t2_flat) * q * q)
        g2_theoretical = ground_truth["offset"] + ground_truth["contrast"] * decay

        # Add noise
        sigma = np.abs(g2_theoretical) * noise_level + 1e-6
        noise = np.random.normal(0, 1, n_points) * sigma
        g2_noisy = g2_theoretical + noise

        xdata_list.append(
            np.column_stack(
                [
                    t1_flat,
                    t2_flat,
                    np.full(n_points, phi_val),
                    np.full(n_points, i),  # phi index for per-angle params
                ]
            )
        )
        ydata_list.append(g2_noisy)
        sigma_list.append(sigma)

    xdata = np.vstack(xdata_list)
    ydata = np.hstack(ydata_list)
    sigma = np.hstack(sigma_list)

    # NLSQ expects xdata shape (n_features, n_points), so transpose
    xdata = xdata.T

    metadata = {
        "ground_truth": ground_truth,
        "phi": phi,
        "t1": t1,
        "t2": t2,
        "q": q,
        "n_phi": n_phi,
        "n_points": len(ydata),
    }

    return xdata, ydata, sigma, metadata


def create_model_function(
    q: float, L: float, dt: float, n_phi: int
) -> Callable[..., Any]:
    """Create model function for curve_fit.

    Uses a simple exponential decay model as a proxy for XPCS g2 to enable
    fast testing of NLSQ optimizer behavior without full physics computation.
    """

    def model(xdata: Any, *params: Any) -> Any:
        """Model: per-angle (contrast, offset) + physical params (D0, alpha, D_offset).

        Simplified model: g2 = offset + contrast * exp(-D_eff * (t1 + t2))
        where D_eff = D0 * tau^(alpha-1) + D_offset, tau = sqrt(t1*t2)
        """
        # Parse params: [contrast_0..N, offset_0..N, D0, alpha, D_offset]
        params_arr = jnp.array(params)
        contrasts = params_arr[:n_phi]
        offsets = params_arr[n_phi : 2 * n_phi]
        D0 = params_arr[2 * n_phi]
        alpha = params_arr[2 * n_phi + 1]
        D_offset = params_arr[2 * n_phi + 2]

        # xdata is (4, n_points) - NLSQ transposed format
        # Rows: [t1, t2, phi, phi_idx]
        xdata_arr = jnp.asarray(xdata)
        t1 = xdata_arr[0, :]
        t2 = xdata_arr[1, :]
        phi_idx = xdata_arr[3, :].astype(jnp.int32)

        # Get per-angle parameters
        contrast = contrasts[phi_idx]
        offset = offsets[phi_idx]

        # Effective time scale
        tau = jnp.sqrt(t1 * t2)

        # Effective diffusion coefficient (simplified model)
        D_eff = D0 * jnp.power(tau + 0.01, alpha - 1.0) + D_offset

        # Decay factor
        decay = jnp.exp(-D_eff * (t1 + t2) * q * q)

        # g2 = offset + contrast * decay
        g2 = offset + contrast * decay

        return g2

    return model


def run_nlsq_fit(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: np.ndarray,
    metadata: dict,
    verbose: bool = True,
) -> dict:
    """Run NLSQ fitting and return results."""
    n_phi = metadata["n_phi"]
    ground_truth = metadata["ground_truth"]

    # Create model function
    model = create_model_function(
        q=metadata["q"],
        L=1.0,  # Not used in simplified model
        dt=0.1,  # Not used in simplified model
        n_phi=n_phi,
    )

    # Initial parameters (10% perturbation from truth)
    p0_list: list[float] = []
    # Per-angle contrasts
    for _ in range(n_phi):
        p0_list.append(ground_truth["contrast"] * 1.1)
    # Per-angle offsets
    for _ in range(n_phi):
        p0_list.append(ground_truth["offset"] * 1.05)
    # Physical params
    p0_list.extend(
        [
            ground_truth["D0"] * 1.15,
            ground_truth["alpha"] * 0.9,
            ground_truth["D_offset"] * 1.2,
        ]
    )
    p0 = np.array(p0_list)

    # Bounds (adjusted for scaled parameters)
    lower = []
    upper = []
    # Contrasts
    for _ in range(n_phi):
        lower.append(0.1)
        upper.append(0.9)
    # Offsets
    for _ in range(n_phi):
        lower.append(0.5)
        upper.append(1.5)
    # Physical params (scaled)
    lower.extend([10.0, 0.1, 0.1])  # D0, alpha, D_offset
    upper.extend([1000.0, 1.0, 10.0])

    bounds = (np.array(lower), np.array(upper))

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"NLSQ Version: {nlsq_version}")
        print(f"Data points: {len(ydata)}")
        print(f"Parameters: {len(p0)}")
        print(f"Initial D0: {p0[2 * n_phi]:.2f}")
        print(f"Initial alpha: {p0[2 * n_phi + 1]:.3f}")
        print(f"Initial D_offset: {p0[2 * n_phi + 2]:.2f}")
        print(f"{'=' * 60}")

    # Get stability mode from environment (default: "auto" like homodyne)
    stability_mode = os.environ.get("NLSQ_STABILITY", "auto")
    stability_mode_val: str | bool = stability_mode
    if stability_mode.lower() == "false":
        stability_mode_val = False

    # Run curve_fit
    start_time = time.time()
    try:
        result = curve_fit(
            f=model,
            xdata=xdata,
            ydata=ydata,
            p0=p0,
            sigma=sigma,
            bounds=bounds,
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            max_nfev=500,
            stability=stability_mode_val,  # type: ignore[arg-type]  # Use environment-specified stability mode
            verbose=2 if verbose else 0,
        )

        # Handle different return types
        if isinstance(result, tuple):
            popt, pcov = result[:2]
        else:
            popt = np.array(result.get("popt", result.get("x", p0)))
            pcov = np.array(result.get("pcov", np.eye(len(p0))))

        elapsed = time.time() - start_time
        success = True
        message = "Converged"

    except Exception as e:
        elapsed = time.time() - start_time
        popt = p0.copy()
        pcov = np.eye(len(p0))
        success = False
        message = str(e)

    # Extract recovered parameters
    recovered = {
        "contrast": np.mean(popt[:n_phi]),
        "offset": np.mean(popt[n_phi : 2 * n_phi]),
        "D0": popt[2 * n_phi],
        "alpha": popt[2 * n_phi + 1],
        "D_offset": popt[2 * n_phi + 2],
    }

    # Compute errors
    errors = {}
    for key in ground_truth:
        true_val = ground_truth[key]
        rec_val = recovered[key]
        errors[key] = abs(rec_val - true_val) / true_val * 100

    # Compute final residual
    if success:
        final_pred = model(xdata, *popt)
        residuals = ydata - final_pred
        chi_squared = np.sum((residuals / sigma) ** 2)
        reduced_chi_squared = chi_squared / (len(ydata) - len(popt))
    else:
        chi_squared = np.inf
        reduced_chi_squared = np.inf

    return {
        "nlsq_version": nlsq_version,
        "success": success,
        "message": message,
        "popt": popt,
        "pcov": pcov,
        "recovered": recovered,
        "ground_truth": ground_truth,
        "errors_pct": errors,
        "chi_squared": chi_squared,
        "reduced_chi_squared": reduced_chi_squared,
        "elapsed_time": elapsed,
    }


def print_results(result: dict) -> None:
    """Print formatted results."""
    print(f"\n{'=' * 60}")
    print(f"RESULTS (NLSQ {result['nlsq_version']})")
    print(f"{'=' * 60}")
    print(f"Success: {result['success']}")
    print(f"Message: {result['message']}")
    print(f"Time: {result['elapsed_time']:.2f}s")
    print(f"Chi-squared: {result['chi_squared']:.6f}")
    print(f"Reduced chi-squared: {result['reduced_chi_squared']:.6f}")
    print()
    print("Parameter Recovery:")
    print(f"  {'Param':<12} {'True':>12} {'Recovered':>12} {'Error %':>10}")
    print(f"  {'-' * 48}")
    for key in result["ground_truth"]:
        true_val = result["ground_truth"][key]
        rec_val = result["recovered"][key]
        err = result["errors_pct"][key]
        status = "OK" if err < 15 else "POOR" if err < 50 else "FAIL"
        print(f"  {key:<12} {true_val:>12.4f} {rec_val:>12.4f} {err:>10.2f}% {status}")
    print(f"{'=' * 60}")


def main() -> None:
    print("NLSQ Regression Test Harness")
    print(f"Python: {sys.version}")
    print(f"NLSQ Version: {nlsq_version}")

    # Generate deterministic data
    print("\nGenerating synthetic XPCS data...")
    xdata, ydata, sigma, metadata = generate_deterministic_data(
        seed=42,
        n_phi=5,
        n_t1=10,
        n_t2=10,
        noise_level=0.02,
    )

    print(f"Data shape: {xdata.shape}")
    print(f"Ground truth: {metadata['ground_truth']}")

    # Run fitting
    print("\nRunning NLSQ fitting...")
    result = run_nlsq_fit(xdata, ydata, sigma, metadata, verbose=True)

    # Print results
    print_results(result)

    # Save results for comparison
    output_file = (
        Path(__file__).parent / f"nlsq_result_{nlsq_version.replace('.', '_')}.npz"
    )
    np.savez(
        output_file,
        popt=result["popt"],
        pcov=result["pcov"],
        chi_squared=result["chi_squared"],
        reduced_chi_squared=result["reduced_chi_squared"],
        errors_pct=result["errors_pct"],
        nlsq_version=nlsq_version,
    )
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
