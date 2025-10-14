#!/usr/bin/env python3
"""
Debug script to diagnose simulated data plotting issues.
Compare computed values with expected physics.
"""

import numpy as np
import jax.numpy as jnp
from homodyne.core.models import CombinedModel
from homodyne.config.manager import ConfigManager

# Load config
config_mgr = ConfigManager("/home/wei/Documents/Projects/data/C020/homodyne_laminar_flow_config.yaml")
config_mgr.load_config()  # Load the config (stores in self.config)
config = config_mgr.config  # Get the config dict

# Extract parameters
initial_params = config["initial_parameters"]
params_dict = dict(zip(initial_params["parameter_names"], initial_params["values"]))

params = jnp.array([
    params_dict["D0"],
    params_dict["alpha"],
    params_dict["D_offset"],
    params_dict["gamma_dot_0"],
    params_dict["beta"],
    params_dict["gamma_dot_offset"],
    params_dict["phi_0"],
])

print("="*80)
print("PARAMETER VALUES")
print("="*80)
for name, val in params_dict.items():
    print(f"{name:20s} = {val:12.6e}")

# Physics constants
analyzer_params = config["analyzer_parameters"]
dt = analyzer_params["dt"]
start_frame = analyzer_params["start_frame"]
end_frame = analyzer_params["end_frame"]
q = analyzer_params["scattering"]["wavevector_q"]
L = analyzer_params["geometry"]["stator_rotor_gap"]

print("\n" + "="*80)
print("PHYSICS CONSTANTS")
print("="*80)
print(f"{'dt':20s} = {dt:12.6e} s")
print(f"{'wavevector_q':20s} = {q:12.6e} Å⁻¹")
print(f"{'stator_rotor_gap':20s} = {L:12.6e} Å  ({L/10000:.1f} μm)")
print(f"{'start_frame':20s} = {start_frame}")
print(f"{'end_frame':20s} = {end_frame}")

# Pre-computed factors
wavevector_q_squared_half_dt = 0.5 * (q ** 2) * dt
sinc_prefactor = 0.5 / np.pi * q * L * dt

print("\n" + "="*80)
print("PRE-COMPUTED FACTORS")
print("="*80)
print(f"{'q²dt/2':20s} = {wavevector_q_squared_half_dt:12.6e}")
print(f"{'qLdt/2π':20s} = {sinc_prefactor:12.6e}")

# Create time array (small for testing)
n_time = 20
time_max = dt * (n_time - 1)
t_vals = jnp.linspace(0, time_max, n_time)
t1_grid, t2_grid = jnp.meshgrid(t_vals, t_vals, indexing="ij")

print("\n" + "="*80)
print("TIME GRID")
print("="*80)
print(f"{'n_time':20s} = {n_time}")
print(f"{'time_max':20s} = {time_max:.6f} s")
print(f"{'t_vals[0]':20s} = {t_vals[0]:.6f} s")
print(f"{'t_vals[-1]':20s} = {t_vals[-1]:.6f} s")
print(f"{'dt_actual':20s} = {float(t_vals[1] - t_vals[0]):.6f} s")

# Create model and compute C2
model = CombinedModel("laminar_flow")

phi_angles = jnp.array([0.0, 45.0, 90.0, 135.0])
contrast = 0.5
offset = 1.0

print("\n" + "="*80)
print("COMPUTING C2 FOR DIFFERENT PHI ANGLES")
print("="*80)

for phi_val in phi_angles:
    phi_array = jnp.array([phi_val])

    # Compute phase factor for this angle
    phi0 = params_dict["phi_0"]
    angle_diff = np.deg2rad(phi0 - phi_val)
    cos_term = np.cos(angle_diff)

    print(f"\nφ = {phi_val:6.1f}°:")
    print(f"  φ0 - φ = {phi0 - phi_val:8.3f}°")
    print(f"  cos(φ0 - φ) = {cos_term:8.6f}")
    print(f"  phase_prefactor = sinc_prefactor × cos = {sinc_prefactor * cos_term:12.6e}")

    # Compute C2
    c2 = model.compute_g2(
        params,
        t1_grid,
        t2_grid,
        phi_array,
        q,
        L,
        contrast,
        offset,
        dt,
    )

    c2_result = np.array(c2[0])

    print(f"  C2 shape:  {c2_result.shape}")
    print(f"  C2 min:    {c2_result.min():.6f}")
    print(f"  C2 max:    {c2_result.max():.6f}")
    print(f"  C2 mean:   {c2_result.mean():.6f}")
    print(f"  C2[0,0]:   {c2_result[0,0]:.6f} (diagonal)")
    print(f"  C2[10,10]: {c2_result[10,10]:.6f} (mid-diagonal)")
    print(f"  C2[0,10]:  {c2_result[0,10]:.6f} (off-diagonal)")

print("\n" + "="*80)
print("EXPECTED BEHAVIOR")
print("="*80)
print("For φ0 ≈ 0° (flow direction):")
print("  - φ = 0°:   cos ≈ 1   → large phase → strong decorrelation → narrow band")
print("  - φ = 45°:  cos ≈ 0.7 → medium phase → medium decorrelation")
print("  - φ = 90°:  cos ≈ 0   → zero phase → no shear → full diffusion correlation")
print("  - φ = 135°: cos ≈ -0.7 → medium phase → medium decorrelation")
print("\nIf all angles except 90° show very weak signal, check:")
print("  1. Is sinc_prefactor too large? (should be ~1e-6 to 1e-3 typically)")
print("  2. Are gamma_integral values unreasonably large?")
print("  3. Units consistency (all Angstroms, all seconds)?")
