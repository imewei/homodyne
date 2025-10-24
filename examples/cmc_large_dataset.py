#!/usr/bin/env python
"""
Example 4: Large Dataset with CMC (Covariance Matrix Combination)

Demonstrates CMC for analyzing large datasets (1M-100M points).

Key concepts:
- Parallel optimization across angles
- Covariance matrix combination
- Diagonal correction for better estimates
- Suitable for 1M-100M point datasets
"""

from pathlib import Path

CONFIG_CMC = """
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

optimization:
  method: "nlsq"

  nlsq:
    max_iterations: 100

  cmc:
    enable: true
    backend: "jax"              # GPU-accelerated
    diagonal_correction: true   # Better covariance estimates

phi_filtering:
  enabled: false

performance:
  strategy_override: null
  device:
    preferred_device: "auto"
    gpu_memory_fraction: 0.9

output:
  directory: "./results_cmc"
"""


def main():
    """Run CMC large dataset example."""
    print("CMC Large Dataset Analysis Example")
    print("=" * 60)

    config_path = Path("homodyne_config_cmc.yaml")
    config_path.write_text(CONFIG_CMC)
    print(f"✓ Created configuration: {config_path}")

    print("\nCMC Configuration:")
    print("  Backend: JAX (GPU-accelerated)")
    print("  Diagonal correction: Enabled")
    print("  Dataset size: Recommended 1M-100M points")

    print("\nCMC approach:")
    print("  1. Optimize parameters for each phi angle in parallel")
    print("  2. Combine covariance matrices from all angles")
    print("  3. Compute combined parameter estimate")
    print("  4. Apply diagonal correction if enabled")

    print("\nExpected output:")
    print("  - Combined parameter estimates")
    print("  - Covariance matrices")
    print("  - Per-angle results")
    print("  - Speedup vs standard NLSQ")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\n✓ CMC configuration ready!")


if __name__ == "__main__":
    main()
