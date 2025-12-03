#!/usr/bin/env python
"""
Example 3: MCMC Uncertainty Quantification (v3.0 CMC-only)

Demonstrates MCMC sampling using CMC (Consensus Monte Carlo) framework for
obtaining posterior distributions and uncertainty estimates.

Key concepts (v3.0 CMC-only architecture):
- All MCMC runs use CMC framework
- CMC uses NUTS as per-shard sampler internally
- Single-shard CMC (num_shards=1) equivalent to legacy NUTS behavior
- Automatic sharding based on hardware and dataset size
- Convergence diagnostics (R-hat, ESS)
- Posterior distributions and credible intervals
- Physics-informed priors from ParameterSpace (no initialization needed)

Configuration example:
  num_warmup: 1000
  num_samples: 2000
  num_chains: 4
  backend: "numpyro"
  cmc_num_shards: auto        # Auto-determined based on hardware
  cmc_backend: multiprocessing
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
    num_warmup: 1000                # NUTS warmup samples per shard
    num_samples: 2000               # Posterior samples per shard
    num_chains: 4                   # Chains per shard
    progress_bar: true              # Show progress
    backend: "numpyro"              # or "blackjax"
    dense_mass_matrix: false        # Use diagonal mass matrix

  cmc:
    num_shards: auto                # Auto-determined from hardware
    backend: multiprocessing        # or pjit, pbs
    sharding_strategy: stratified   # Per-phi stratified sharding

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
    print("MCMC Uncertainty Quantification Example (v3.0 CMC-only)")
    print("=" * 60)

    config_path = Path("homodyne_config_mcmc.yaml")
    config_path.write_text(CONFIG_MCMC)
    print(f"Created configuration: {config_path}")

    print("\nMCMC Configuration (v3.0 CMC-only):")
    print("  Warmup samples: 1000 (tuning phase per shard)")
    print("  Posterior samples: 2000 (per shard)")
    print("  Parallel chains: 4 (per shard)")
    print("  CMC backend: multiprocessing")
    print("  Backend: NumPyro (with progress bars)")

    print("\nCMC Sharding (v3.0):")
    print("  - All MCMC runs use CMC framework")
    print("  - Single-shard CMC equivalent to legacy NUTS")
    print("  - Shards determined automatically from hardware")
    print("  - Stratified sharding preserves per-phi structure")

    print("\nExpected output:")
    print("  - parameters.json: Mean, median, std of posterior")
    print("  - corner_plot.png: Joint posterior distributions")
    print("  - trace_plots.png: MCMC chain traces")
    print("  - mcmc_diagnostics.json: R-hat, ESS values")

    print("\nConvergence diagnostics:")
    print("  R-hat: Should be < 1.01 (indicates convergence)")
    print("  ESS: Effective sample size (>>1 means efficient sampling)")

    print("\nMigration note (v2.x -> v3.0):")
    print("  - CMC-only architecture (NUTS runs inside CMC shards)")
    print("  - Removed: min_samples_for_cmc, memory_threshold_pct")
    print("  - Added: cmc.num_shards, cmc.backend, cmc.sharding_strategy")
    print("  - See docs/migration/v3_cmc_only.md for details")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\nMCMC configuration ready!")


if __name__ == "__main__":
    main()
