#!/usr/bin/env python3
"""Demonstration of Task Group 9: MCMC Integration with CMC.

This script demonstrates the new automatic method selection in fit_mcmc_jax()
which chooses between standard NUTS and Consensus Monte Carlo (CMC) based on
dataset size and hardware configuration.

Author: Claude Code (Task Group 9)
Date: 2025-10-24
"""

import numpy as np


def demo_automatic_selection() -> None:
    """Demonstrate automatic method selection (v2.1.0)."""
    print("=" * 70)
    print("DEMO 1: Automatic Method Selection (v2.1.0)")
    print("=" * 70)

    # Create small synthetic dataset (10k points)
    np.random.seed(42)
    np.random.randn(10_000) * 0.1 + 1.0
    np.random.rand(10_000) * 10
    np.random.rand(10_000) * 10
    np.random.rand(10_000) * 360 - 180

    print("\nSmall dataset: 10,000 points")
    print("Expected: Automatic selection will choose NUTS")
    print("Reason: < 15 samples AND < 30% memory\n")

    # Note: This would normally run MCMC, but for demo we're just showing the API
    # In production, you would call:
    # from homodyne.config.parameter_space import ParameterSpace
    # parameter_space = ParameterSpace.from_config(config_dict)
    # result = fit_mcmc_jax(
    #     data=data_small,
    #     t1=t1,
    #     t2=t2,
    #     phi=phi,
    #     q=0.01,
    #     L=3.5,
    #     parameter_space=parameter_space,
    #     initial_values=initial_values,
    #     # No method= parameter (automatic selection)
    # )
    # print(f"Method used: {'CMC' if result.is_cmc_result() else 'NUTS'}")

    print("✅ API call signature (v2.1.0):")
    print("   result = fit_mcmc_jax(")
    print("       data, t1, t2, phi, q, L,")
    print("       parameter_space=parameter_space,")
    print("       initial_values=initial_values,")
    print("   )")
    print("   # Automatically selects NUTS for small datasets")


def demo_forced_nuts() -> None:
    """Demonstrate NUTS method (v2.1.0: automatic selection)."""
    print("\n" + "=" * 70)
    print("DEMO 2: NUTS Method (v2.1.0: Automatic Selection)")
    print("=" * 70)

    print("\nWhen NUTS is used:")
    print("  - Criteria: num_samples < 15 AND memory < 30%")
    print("  - Typical: Small experiments (< 15 angles) with < 100M points")
    print("\n✅ API call signature (v2.1.0):")
    print("   result = fit_mcmc_jax(")
    print("       data, t1, t2, phi, q, L,")
    print("       parameter_space=parameter_space,")
    print("       initial_values=initial_values,")
    print("   )")
    print("   # Automatic: NUTS for small datasets")


def demo_forced_cmc() -> None:
    """Demonstrate CMC method (v2.1.0: automatic selection)."""
    print("\n" + "=" * 70)
    print("DEMO 3: CMC Method (v2.1.0: Automatic Selection)")
    print("=" * 70)

    print("\nWhen CMC is used:")
    print("  - Criterion 1: num_samples >= 15 (parallelism)")
    print("    Example: 20+ phi angles → ~1.4x speedup")
    print("  - Criterion 2: memory > 30% (OOM prevention)")
    print("    Example: 100M+ points → avoid OOM")

    print("\nCMC Configuration (v2.1.0):")

    print("  - min_samples_for_cmc: 15 (configurable)")
    print("  - memory_threshold_pct: 0.30 (configurable)")
    print("  - dense_mass_matrix: false (efficiency)")

    print("\n✅ API call signature (v2.1.0):")
    print("   result = fit_mcmc_jax(")
    print("       data, t1, t2, phi, q, L,")
    print("       parameter_space=parameter_space,")
    print("       initial_values=initial_values,")
    print("   )")
    print("   # Automatic: CMC when >=15 samples OR >=30% memory")


def demo_backward_compatibility() -> None:
    """Demonstrate backward compatibility (v2.1.0)."""
    print("\n" + "=" * 70)
    print("DEMO 4: Backward Compatibility (v2.1.0)")
    print("=" * 70)

    print("\nVersion 2.0 API signature (deprecated):")
    print("  ❌ result = fit_mcmc_jax(data, t1, t2, phi, q, L, method='nuts')")
    print("  ❌ result = fit_mcmc_jax(data, ..., method='cmc')")
    print("  ❌ result = fit_mcmc_jax(data, ..., initial_params={})")

    print("\nVersion 2.1 API signature (current):")
    print("✅ result = fit_mcmc_jax(")
    print("       data, t1, t2, phi, q, L,")
    print("       parameter_space=parameter_space,")
    print("       initial_values=initial_values,")
    print("   )")
    print("\nChanges required for v2.0 → v2.1 migration:")
    print("  1. Remove 'method' parameter (automatic selection)")
    print("  2. Rename 'initial_params' to 'initial_values'")
    print("  3. Add 'parameter_space' from ParameterSpace.from_config()")
    print("  4. Remove 'mcmc.initialization' from YAML config")
    print("  5. Add 'min_samples_for_cmc', 'memory_threshold_pct' to YAML")


def demo_result_inspection() -> None:
    """Demonstrate how to inspect results."""
    print("\n" + "=" * 70)
    print("DEMO 5: Inspecting Results")
    print("=" * 70)

    print("\nCheck which method was used:")
    print("✅ if result.is_cmc_result():")
    print("       print(f'Used CMC with {result.num_shards} shards')")
    print("       print(f'Combination: {result.combination_method}')")
    print(
        "       print(f'Converged: {result.cmc_diagnostics[\"n_shards_converged\"]}')"
    )
    print("   else:")
    print("       print('Used standard NUTS')")
    print("       print(f'Acceptance rate: {result.acceptance_rate}')")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print(
        "║"
        + "  v2.1.0: MCMC with Automatic NUTS/CMC Selection - Demonstrations  ".center(
            68
        )
        + "║"
    )
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    demo_automatic_selection()
    demo_forced_nuts()
    demo_forced_cmc()
    demo_backward_compatibility()
    demo_result_inspection()

    print("\n" + "=" * 70)
    print("v2.1.0 Features & Changes")
    print("=" * 70)
    print("\n✅ Automatic method selection (NUTS/CMC)")
    print("✅ Dual-criteria OR logic: samples >= 15 OR memory > 30%")
    print("✅ Removed explicit 'method' parameter")
    print("✅ New API: parameter_space, initial_values")
    print("✅ Removed 'mcmc.initialization' from YAML")
    print("✅ Added 'min_samples_for_cmc', 'memory_threshold_pct'")
    print("✅ Hardware-adaptive thresholds (configurable)")
    print("✅ Extended MCMCResult with CMC fields")
    print("\n" + "=" * 70)
    print("Migration Status: v2.0 → v2.1.0 (Breaking Changes)")
    print("=" * 70)
    print("\nSee docs/migration/v2.0-to-v2.1.md for detailed upgrade guide")
    print("\n✨ v2.1.0 Ready for production use.\n")
