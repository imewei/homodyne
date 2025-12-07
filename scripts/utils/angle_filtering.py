#!/usr/bin/env python
"""
Example 7: Angle Filtering for Anisotropic Analysis

Demonstrates phi angle filtering for analyzing direction-dependent dynamics.

Key concepts:
- Angle filtering reduces parameter count from 3+2n to 3+2m
- m = number of filtered angle ranges (<<  n = total angles)
- Improves convergence by reducing parameters
- Handles wrap-around at ±180° boundary automatically
"""

from pathlib import Path

CONFIG_ANGLE_FILTERING = """
experimental_data:
  file_path: "./data/sample/experiment.hdf"

parameter_space:
  model: "laminar_flow"
  bounds:
    - name: D0
      min: 100.0
      max: 1e5
    - name: alpha
      min: 0.0
      max: 2.0
    - name: D_offset
      min: -100.0
      max: 100.0
    - name: gamma_dot_0
      min: 1e-6
      max: 0.5
    - name: beta
      min: 0.0
      max: 2.0
    - name: gamma_dot_offset
      min: -0.1
      max: 0.1
    - name: phi_0
      min: -180.0
      max: 180.0

initial_parameters:
  parameter_names: [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi_0]
  active_parameters: [D0, gamma_dot_0, beta, phi_0]

optimization:
  method: "nlsq"
  nlsq:
    max_iterations: 100

# Angle filtering for anisotropic analysis
phi_filtering:
  enabled: true

  target_ranges:
    # Parallel to flow direction
    - min_angle: -10.0
      max_angle: 10.0
      description: "Parallel to flow (0°)"

    # Perpendicular to flow direction
    - min_angle: 85.0
      max_angle: 95.0
      description: "Perpendicular to flow (90°)"

  # Parameter counting with these 2 ranges:
  # Total = 7 physical + 2*2 = 11 parameters
  # vs. without filtering: 7 + 2*n_all_angles (e.g., 7 + 2*10 = 27)

performance:
  device:
    preferred_device: "auto"

output:
  directory: "./results_angle_filtering"
"""


def main() -> None:
    """Run angle filtering example."""
    print("Angle Filtering for Anisotropic Analysis Example")
    print("=" * 60)

    config_path: Path = Path("homodyne_config_angle_filtering.yaml")
    config_path.write_text(CONFIG_ANGLE_FILTERING)
    print(f"✓ Created configuration: {config_path}")

    print("\nAngle Filtering Configuration:")
    print("  Enabled: True")
    print("  Ranges: 2")
    print("    1. [-10°, 10°] - Parallel to flow")
    print("    2. [85°, 95°] - Perpendicular to flow")

    print("\nParameter count reduction:")
    print("  Without filtering:")
    print("    If data has 10 angles → 7 + 2*10 = 27 parameters")
    print("\n  With 2 filtered ranges:")
    print("    7 + 2*2 = 11 parameters")
    print("\n  Benefit: ~60% reduction in parameters")

    print("\nAngle normalization:")
    print("  All angles normalized to [-180°, 180°]")
    print("  Wrap-around handled automatically")
    print("  Example: 350° → -10°")
    print("  Range [170°, -170°] spans ±180° boundary correctly")

    print("\nAnalysis benefits:")
    print("  - Faster convergence")
    print("  - More stable optimization")
    print("  - Separates anisotropic components")
    print("  - Improves parameter constraints")

    print("\nIntegration points:")
    print("  - Applied before NLSQ optimization")
    print("  - Applied before MCMC sampling")
    print("  - Applied in plotting functions")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\nExpected output:")
    print("  - Separate analysis for each angle range")
    print("  - Angle-specific contrast and offset")
    print("  - Flow parameters per range")

    print("\n✓ Angle filtering configuration ready!")


if __name__ == "__main__":
    main()
