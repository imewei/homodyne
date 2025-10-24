#!/usr/bin/env python
"""
Example 3: MCMC Uncertainty Quantification

Demonstrates MCMC sampling with NumPyro for obtaining posterior distributions
and uncertainty estimates.

Key concepts:
- MCMC sampling vs NLSQ point estimates
- NumPyro NUTS sampler for efficient sampling
- Convergence diagnostics (R-hat, ESS)
- Posterior distributions and credible intervals

Configuration example:
  num_warmup: 1000
  num_samples: 2000
  num_chains: 4
  backend: "numpyro"
"""

from pathlib import Path

CONFIG_MCMC = """
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
  active_parameters: [D0, gamma_dot_0, beta]

optimization:
  method: "mcmc"

  mcmc:
    num_warmup: 1000           # NUTS warmup samples
    num_samples: 2000          # Posterior samples
    num_chains: 4              # Parallel chains
    progress_bar: true         # Show progress
    backend: "numpyro"         # or "blackjax"

phi_filtering:
  enabled: true
  target_ranges:
    - min_angle: -10.0
      max_angle: 10.0

performance:
  device:
    preferred_device: "auto"

output:
  directory: "./results_mcmc"
"""


def main():
    """Run MCMC uncertainty example."""
    print("MCMC Uncertainty Quantification Example")
    print("=" * 60)

    config_path = Path("homodyne_config_mcmc.yaml")
    config_path.write_text(CONFIG_MCMC)
    print(f"✓ Created configuration: {config_path}")

    print("\nMCMC Configuration:")
    print("  Warmup samples: 1000 (tuning phase)")
    print("  Posterior samples: 2000 (for inference)")
    print("  Parallel chains: 4 (for convergence diagnostics)")
    print("  Sampler: NUTS (No-U-Turn Sampler)")
    print("  Backend: NumPyro (with progress bars)")

    print("\nExpected output:")
    print("  - parameters.json: Mean, median, std of posterior")
    print("  - corner_plot.png: Joint posterior distributions")
    print("  - trace_plots.png: MCMC chain traces")
    print("  - mcmc_diagnostics.json: R-hat, ESS values")

    print("\nConvergence diagnostics:")
    print("  R-hat: Should be < 1.01 (indicates convergence)")
    print("  ESS: Effective sample size (>>1 means efficient sampling)")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\n✓ MCMC configuration ready!")


if __name__ == "__main__":
    main()
