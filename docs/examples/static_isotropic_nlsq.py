#!/usr/bin/env python
"""
Example 1: Static Isotropic NLSQ Optimization

This example demonstrates NLSQ optimization for a static isotropic system.

Key concepts:
- 3+2n parameters (3 physical + 2 per angle)
- NLSQ trust-region optimization
- Suitable for small to medium datasets (< 100M points)

Parameter counting:
- Static isotropic: 3 physical [D0, alpha, D_offset]
- Per-angle scaling: 2 parameters [contrast, offset] per angle
- Example with 3 angles: 3 + 2*3 = 9 total parameters

Learning outcomes:
- How to set up static isotropic configuration
- Interpreting NLSQ convergence metrics
- Reading parameter uncertainties
"""

import json
from pathlib import Path

# Create sample configuration for static isotropic analysis
CONFIG_STATIC_ISOTROPIC = """
# ==============================================================================
# STATIC ISOTROPIC NLSQ OPTIMIZATION EXAMPLE
# ==============================================================================
#
# Parameter count: 3 physical + 2 per angle = 3 + 2n
# Example: 3 angles → 3 + 2×3 = 9 total parameters
#

experimental_data:
  file_path: "./data/sample/experiment.hdf"

parameter_space:
  model: "static_isotropic"

  bounds:
    # Diffusion coefficient D₀ [Å²/s]
    - name: D0
      min: 100.0
      max: 1e5

    # Anomalous diffusion exponent α [dimensionless]
    # 0 = normal, <1 = subdiffusion, >1 = superdiffusion
    - name: alpha
      min: 0.0
      max: 2.0

    # Diffusion offset [Å²/s]
    - name: D_offset
      min: -100.0
      max: 100.0

initial_parameters:
  parameter_names:
    - D0
    - alpha
    - D_offset

  # Optional: optimize only subset
  active_parameters:
    - D0
    - alpha

  # Optional: fix specific parameters
  fixed_parameters:
    D_offset: 10.0

optimization:
  method: "nlsq"

  nlsq:
    max_iterations: 100
    tolerance: 1e-8
    trust_region_scale: 1.0

phi_filtering:
  enabled: false

performance:
  strategy_override: null
  enable_progress: true
  device:
    preferred_device: "auto"

output:
  directory: "./results_static_isotropic"
"""


def main():
    """Run static isotropic NLSQ example."""
    print("Static Isotropic NLSQ Optimization Example")
    print("=" * 60)

    # Create configuration file
    config_path = Path("homodyne_config_static_isotropic.yaml")
    config_path.write_text(CONFIG_STATIC_ISOTROPIC)
    print(f"✓ Created configuration: {config_path}")

    print("\nKey configuration parameters:")
    print("  Model: static_isotropic")
    print("  Physical parameters: D0, alpha, D_offset (3)")
    print("  Per-angle parameters: contrast, offset per angle (2n)")
    print("  Total: 3 + 2n parameters")
    print("\nWithout angle filtering:")
    print("  - If data has 10 phi angles → 3 + 2*10 = 23 parameters")

    print("\nWith angle filtering (if enabled):")
    print("  - If filtered to 2 angle ranges → 3 + 2*2 = 7 parameters")
    print("  - Reduces parameter count for better convergence")

    print("\nTo run the analysis:")
    print(f"  homodyne --config {config_path}")

    print("\nExpected output files:")
    print("  - results_static_isotropic/nlsq/parameters.json")
    print("  - results_static_isotropic/nlsq/fitted_data.npz")
    print("  - results_static_isotropic/nlsq/analysis_results_nlsq.json")
    print("  - results_static_isotropic/nlsq/convergence_metrics.json")

    print("\nParameters learned:")
    print("  D0: Diffusion coefficient (Å²/s)")
    print("  alpha: Power-law exponent (0=normal, <1=subdiffusion)")
    print("  D_offset: Diffusion offset term (Å²/s)")
    print("  contrast[i]: Amplitude for angle i")
    print("  offset[i]: Baseline for angle i")

    print("\n✓ Example configuration ready!")
    print("\nNext steps:")
    print("  1. Prepare your HDF5 data file")
    print("  2. Update 'file_path' in the configuration")
    print("  3. Run: homodyne --config homodyne_config_static_isotropic.yaml")
    print("  4. Examine results in results_static_isotropic/")


if __name__ == "__main__":
    main()
