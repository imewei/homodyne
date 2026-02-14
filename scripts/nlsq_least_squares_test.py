#!/usr/bin/env python3
"""
Test LeastSquares class directly (what homodyne uses).

This tests enable_stability=True behavior between v0.3.0 and HEAD.
"""

import os
import sys

# Allow NLSQ path override
NLSQ_PATH = os.environ.get("NLSQ_PATH")
if NLSQ_PATH:
    sys.path.insert(0, NLSQ_PATH)
    print(f"Using NLSQ from: {NLSQ_PATH}")

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from nlsq import LeastSquares  # noqa: E402
from nlsq import __version__ as nlsq_version  # noqa: E402

print(f"NLSQ Version: {nlsq_version}")

# Parameters
n_phi = 5
n_points_per_phi = 100
n_params = 2 * n_phi + 3  # 5 contrasts + 5 offsets + D0, alpha, D_offset
q = 0.01

# Ground truth
ground_truth = {
    "D0": 100.0,
    "alpha": 0.5,
    "D_offset": 1.0,
    "contrast": 0.5,
    "offset": 1.0,
}

# Generate synthetic data
np.random.seed(42)
t1 = np.linspace(0.1, 1.0, 10)
t2 = np.linspace(0.1, 1.0, 10)
T1, T2 = np.meshgrid(t1, t2, indexing='ij')

xdata_list = []
ydata_list = []

for i in range(n_phi):
    phi_val = 2 * np.pi * i / n_phi
    t1_flat = T1.ravel()
    t2_flat = T2.ravel()

    tau = np.sqrt(t1_flat * t2_flat)
    D_eff = ground_truth["D0"] * np.power(tau + 0.01, ground_truth["alpha"] - 1.0) + ground_truth["D_offset"]
    decay = np.exp(-D_eff * (t1_flat + t2_flat) * q * q)
    g2_theoretical = ground_truth["offset"] + ground_truth["contrast"] * decay

    sigma = np.abs(g2_theoretical) * 0.02 + 1e-6
    noise = np.random.normal(0, 1, len(t1_flat)) * sigma
    g2_noisy = g2_theoretical + noise

    xdata_list.append(np.column_stack([
        t1_flat, t2_flat,
        np.full(len(t1_flat), phi_val),
        np.full(len(t1_flat), i),
    ]))
    ydata_list.append(g2_noisy)

xdata = jnp.array(np.vstack(xdata_list))  # (n_total, 4)
ydata = jnp.array(np.hstack(ydata_list))  # (n_total,)

# Initial guess
p0 = np.array([0.55] * n_phi + [1.05] * n_phi + [115.0, 0.45, 1.2])
bounds = (
    np.array([0.1] * n_phi + [0.5] * n_phi + [10.0, 0.1, 0.1]),
    np.array([0.9] * n_phi + [1.5] * n_phi + [1000.0, 1.0, 10.0]),
)


def model_func(xdata_chunk, *params):
    """Model function for LeastSquares."""
    params_arr = jnp.array(params)
    contrasts = params_arr[:n_phi]
    offsets = params_arr[n_phi:2*n_phi]
    D0 = params_arr[2*n_phi]
    alpha = params_arr[2*n_phi + 1]
    D_offset = params_arr[2*n_phi + 2]

    t1 = xdata_chunk[:, 0]
    t2 = xdata_chunk[:, 1]
    phi_idx = xdata_chunk[:, 3].astype(jnp.int32)

    contrast = contrasts[phi_idx]
    offset = offsets[phi_idx]

    tau = jnp.sqrt(t1 * t2)
    D_eff = D0 * jnp.power(tau + 0.01, alpha - 1.0) + D_offset
    decay = jnp.exp(-D_eff * (t1 + t2) * q * q)
    g2 = offset + contrast * decay
    return g2


# Test with enable_stability=True (what homodyne uses)
print("\n" + "=" * 60)
print("Testing LeastSquares with enable_stability=True")
print("=" * 60)

# Note: enable_diagnostics=False to avoid API incompatibility between versions
ls = LeastSquares(enable_stability=True, enable_diagnostics=False)

result = ls.least_squares(
    fun=model_func,
    x0=p0,
    xdata=xdata,
    ydata=ydata,
    bounds=bounds,
    method="trf",
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-8,
    max_nfev=500,
    verbose=2,
)

popt = np.array(result["x"])
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"NLSQ Version: {nlsq_version}")
print(f"Success: {result.get('success', 'N/A')}")
print(f"Cost: {result.get('cost', 'N/A')}")
print(f"D0: {popt[2*n_phi]:.4f} (true: {ground_truth['D0']})")
print(f"alpha: {popt[2*n_phi + 1]:.4f} (true: {ground_truth['alpha']})")
print(f"D_offset: {popt[2*n_phi + 2]:.4f} (true: {ground_truth['D_offset']})")
print(f"Contrast mean: {np.mean(popt[:n_phi]):.4f} (true: {ground_truth['contrast']})")
print(f"Offset mean: {np.mean(popt[n_phi:2*n_phi]):.4f} (true: {ground_truth['offset']})")
print("=" * 60)
