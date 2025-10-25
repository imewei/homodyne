#!/usr/bin/env python
"""
Example 2: Laminar Flow NLSQ with Angle Filtering

This example demonstrates NLSQ optimization for a laminar flow system
with angle filtering to reduce parameter count.

Key concepts:
- 7+2n parameters (7 physical + 2 per angle)
- Angle filtering reduces 'n' to number of filtered ranges
- Extracting flow parameters (γ̇₀, β, γ̇_offset)
- Suitable for flowing systems with anisotropic dynamics

Parameter counting:
- Laminar flow: 7 physical [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi_0]
- Per-angle scaling: 2 parameters [contrast, offset] per angle range
- With 2 filtered ranges: 7 + 2*2 = 11 total parameters

Learning outcomes:
- Configuring angle filtering
- Understanding per-angle scaling
- Extracting flow parameters
- Analyzing anisotropic dynamics
"""

from pathlib import Path

CONFIG_LAMINAR_FLOW = """
# ==============================================================================
# LAMINAR FLOW NLSQ OPTIMIZATION EXAMPLE
# ==============================================================================
#
# Parameter count: 7 physical + 2 per angle range = 7 + 2m
# Example with 2 angle ranges: 7 + 2×2 = 11 total parameters
#
# Physical parameters:
#   - D₀: Diffusion coefficient [Å²/s]
#   - α: Diffusion power-law exponent [dimensionless]
#   - D_offset: Diffusion offset [Å²/s]
#   - γ̇₀: Initial shear rate [s⁻¹]
#   - β: Shear power-law exponent [dimensionless]
#   - γ̇_offset: Shear rate offset [s⁻¹]
#   - φ₀: Initial angle [degrees]
#

experimental_data:
  file_path: "./data/sample/experiment.hdf"

parameter_space:
  model: "laminar_flow"

  bounds:
    # Diffusion parameters
    - name: D0
      min: 100.0
      max: 1e5

    - name: alpha
      min: 0.0
      max: 2.0

    - name: D_offset
      min: -100.0
      max: 100.0

    # Flow parameters
    - name: gamma_dot_0    # Maps to gamma_dot_t0 in code
      min: 1e-6
      max: 0.5

    - name: beta
      min: 0.0
      max: 2.0

    - name: gamma_dot_offset
      min: -0.1
      max: 0.1

    - name: phi_0          # Maps to phi0 in code
      min: -180.0
      max: 180.0

initial_parameters:
  parameter_names:
    - D0
    - alpha
    - D_offset
    - gamma_dot_0
    - beta
    - gamma_dot_offset
    - phi_0

  # Optimize critical flow parameters
  active_parameters:
    - D0
    - gamma_dot_0
    - beta
    - phi_0

optimization:
  method: "nlsq"

  nlsq:
    max_iterations: 150
    tolerance: 1e-8
    trust_region_scale: 1.0

# Angle filtering: reduces parameter count
phi_filtering:
  enabled: true

  target_ranges:
    # Near-parallel to flow direction
    - min_angle: -10.0
      max_angle: 10.0
      description: "Parallel to flow (0°)"

    # Near-perpendicular to flow direction
    - min_angle: 85.0
      max_angle: 95.0
      description: "Perpendicular to flow (90°)"

  # With 2 filtered ranges:
  # Total parameters = 7 + 2*2 = 11
  # vs. without filtering: 7 + 2*n_all_angles (potentially 7+2*10=27)

performance:
  strategy_override: null
  enable_progress: true
  device:
    preferred_device: "auto"

output:
  directory: "./results_laminar_flow"
"""


def main():
    """Run laminar flow NLSQ example."""
    print("Laminar Flow NLSQ with Angle Filtering Example")
    print("=" * 60)

    # Create configuration file
    config_path = Path("homodyne_config_laminar_flow.yaml")
    config_path.write_text(CONFIG_LAMINAR_FLOW)
    print(f"✓ Created configuration: {config_path}")

    print("\nKey configuration parameters:")
    print("\nPhysical parameters (7):")
    print("  - D0: Diffusion coefficient [Å²/s]")
    print("  - alpha: Diffusion exponent")
    print("  - D_offset: Diffusion offset")
    print("  - gamma_dot_0: Initial shear rate [s⁻¹]")
    print("  - beta: Shear exponent")
    print("  - gamma_dot_offset: Shear offset")
    print("  - phi_0: Initial angle [degrees]")

    print("\nAngle filtering:")
    print("  Enabled with 2 ranges:")
    print("    1. Near 0° (parallel to flow)")
    print("    2. Near 90° (perpendicular to flow)")

    print("\nParameter counting:")
    print("  Without filtering: 7 + 2*n_angles")
    print("    Example: 10 angles → 7 + 2*10 = 27 parameters")
    print("\n  With filtering to 2 ranges: 7 + 2*2 = 11 parameters")
    print("  Benefit: Faster convergence, more constrained solution")

    print("\nPer-angle parameters (contrast, offset):")
    print("  - Amplitude scaling for each angle range")
    print("  - Baseline offset for each angle range")

    print("\nTo run the analysis:")
    print(f"  homodyne --config {config_path}")

    print("\nExpected output:")
    print("  - Flow parameters: γ̇₀, β, γ̇_offset")
    print("  - Per-angle analysis for 0° and 90° regions")
    print("  - Angle-dependent dynamics")

    print("\n✓ Example configuration ready!")
    print("\nNext steps:")
    print("  1. Prepare your HDF5 data file")
    print("  2. Update 'file_path' in the configuration")
    print("  3. Run: homodyne --config homodyne_config_laminar_flow.yaml")
    print("  4. Examine results in results_laminar_flow/")


if __name__ == "__main__":
    main()
