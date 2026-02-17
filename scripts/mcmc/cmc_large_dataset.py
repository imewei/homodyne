#!/usr/bin/env python
"""
Example 4: Large Dataset with Automatic CMC Selection (v2.1.0)

Demonstrates automatic CMC selection when analyzing large datasets or many samples.

Key concepts:
- Automatic NUTS/CMC selection (no manual method selection required)
- CMC triggered by: many samples (>= 15) OR large memory footprint (>= 30%)
- Parallel optimization across angles (when CMC is selected)
- Covariance matrix combination
- Diagonal correction for better estimates
- Suitable for 1M-100M point datasets OR 15+ angles
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
  method: "mcmc"              # Automatic NUTS/CMC selection

  mcmc:
    num_warmup: 1000
    num_samples: 2000
    min_samples_for_cmc: 15         # CMC if num_samples >= 15
    memory_threshold_pct: 0.30      # CMC if memory > 30%
    dense_mass_matrix: false        # Diagonal mass matrix for large datasets

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


def main() -> None:
    """Run automatic CMC selection example."""
    print("Large Dataset with Automatic CMC Selection (v2.1.0)")
    print("=" * 60)

    config_path = Path("homodyne_config_cmc.yaml")
    config_path.write_text(CONFIG_CMC)
    print(f"✓ Created configuration: {config_path}")

    print("\nAutomatic CMC Selection (v2.1.0):")
    print("  Criterion 1: num_samples >= 15 → CMC for parallelism")
    print("    Example: 23 phi angles trigger CMC for ~1.4x speedup")
    print("  Criterion 2: memory > 30% → CMC for memory management")
    print("    Example: 100M+ data points trigger CMC to avoid OOM")

    print("\nMCMC Configuration:")
    print("  Method: mcmc (automatic NUTS/CMC selection)")
    print("  Warmup samples: 1000")
    print("  Posterior samples: 2000")
    print("  Dataset size: Recommended 1M-100M points OR 15+ angles")

    print("\nCMC Approach (when triggered):")
    print("  1. Sample parameters for each phi angle in parallel")
    print("  2. Combine posterior samples from all angles")
    print("  3. Compute combined parameter estimates")
    print("  4. Apply diagonal mass matrix (efficiency for large datasets)")

    print("\nExpected output:")
    print("  - Posterior parameter estimates")
    print("  - Covariance matrices")
    print("  - Per-angle results (if CMC triggered)")
    print("  - Convergence diagnostics (R-hat, ESS)")
    print("  - Speedup vs NUTS (if CMC triggered)")

    print("\nMigration note (v2.0 → v2.1):")
    print("  - Removed explicit 'cmc' method (use 'mcmc' with automatic selection)")
    print("  - Removed 'initialization' section (use ParameterSpace for priors)")
    print("  - Thresholds configurable: min_samples_for_cmc, memory_threshold_pct")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\n✓ CMC configuration ready!")


if __name__ == "__main__":
    main()
