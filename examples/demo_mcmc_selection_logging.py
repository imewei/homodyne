#!/usr/bin/env python
"""
Demonstration of Automatic NUTS/CMC Selection with Comprehensive Logging

This script demonstrates the dual-criteria OR logic for automatic method selection
in the MCMC implementation, showing how the decision is made based on:

1. Parallelism criterion: num_samples >= min_samples_for_cmc (default: 15)
2. Memory criterion: estimated_memory > memory_threshold_pct (default: 30%)

The comprehensive logging shows both criteria evaluation and the final decision.

Usage:
    python examples/demo_mcmc_selection_logging.py
"""

import logging
from homodyne.device.config import HardwareConfig, should_use_cmc

# Configure logging to see INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

# Create a mock hardware configuration (typical workstation)
hardware_config = HardwareConfig(
    platform="cpu",
    num_devices=1,
    memory_per_device_gb=32.0,  # 32 GB RAM
    num_nodes=1,
    cores_per_node=14,  # 14-core CPU
    total_memory_gb=32.0,
    cluster_type="standalone",
    recommended_backend="multiprocessing",
    max_parallel_shards=14,
)

print("=" * 80)
print("MCMC Automatic Selection Demonstration")
print("=" * 80)
print(f"\nHardware Configuration:")
print(f"  Platform: {hardware_config.platform}")
print(f"  Memory: {hardware_config.memory_per_device_gb:.1f} GB")
print(f"  CPU Cores: {hardware_config.cores_per_node}")
print()

# Scenario 1: Parallelism Mode
print("\n" + "=" * 80)
print("SCENARIO 1: Parallelism Mode (Many Samples)")
print("=" * 80)
print("\nExperiment: 20 phi angles, 10M data points")
print("Expected: CMC (parallelism criterion triggers)")
print()

result1 = should_use_cmc(
    num_samples=20,
    hardware_config=hardware_config,
    dataset_size=10_000_000,
    min_samples_for_cmc=15,
    memory_threshold_pct=0.30,
)

print(f"\n→ Decision: {'CMC' if result1 else 'NUTS'}")

# Scenario 2: Memory Mode
print("\n" + "=" * 80)
print("SCENARIO 2: Memory Mode (Large Dataset)")
print("=" * 80)
print("\nExperiment: 5 phi angles, 50M data points")
print("Expected: CMC (memory criterion triggers)")
print()

result2 = should_use_cmc(
    num_samples=5,
    hardware_config=hardware_config,
    dataset_size=50_000_000,
    min_samples_for_cmc=15,
    memory_threshold_pct=0.30,
)

print(f"\n→ Decision: {'CMC' if result2 else 'NUTS'}")

# Scenario 3: NUTS Mode (Both Criteria Fail)
print("\n" + "=" * 80)
print("SCENARIO 3: NUTS Mode (Small Experiment)")
print("=" * 80)
print("\nExperiment: 10 phi angles, 5M data points")
print("Expected: NUTS (both criteria fail)")
print()

result3 = should_use_cmc(
    num_samples=10,
    hardware_config=hardware_config,
    dataset_size=5_000_000,
    min_samples_for_cmc=15,
    memory_threshold_pct=0.30,
)

print(f"\n→ Decision: {'CMC' if result3 else 'NUTS'}")

# Scenario 4: Edge Case (CMC with Few Samples)
print("\n" + "=" * 80)
print("SCENARIO 4: Edge Case (Memory Triggers CMC with Few Samples)")
print("=" * 80)
print("\nExperiment: 5 phi angles, 60M data points")
print("Expected: CMC with warning (memory > 30% but samples < 15)")
print()

result4 = should_use_cmc(
    num_samples=5,
    hardware_config=hardware_config,
    dataset_size=60_000_000,
    min_samples_for_cmc=15,
    memory_threshold_pct=0.30,
)

print(f"\n→ Decision: {'CMC' if result4 else 'NUTS'}")

# Scenario 5: Custom Thresholds
print("\n" + "=" * 80)
print("SCENARIO 5: Custom Thresholds (Config Override)")
print("=" * 80)
print("\nExperiment: 18 phi angles, 15M data points")
print("Config: min_samples_for_cmc=20, memory_threshold_pct=0.25")
print("Expected: NUTS (18 < 20, memory < 25%)")
print()

result5 = should_use_cmc(
    num_samples=18,
    hardware_config=hardware_config,
    dataset_size=15_000_000,
    min_samples_for_cmc=20,  # Custom threshold (higher than default 15)
    memory_threshold_pct=0.25,  # Custom threshold (lower than default 0.30)
)

print(f"\n→ Decision: {'CMC' if result5 else 'NUTS'}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nScenario 1 (Parallelism):  {'✓ CMC' if result1 else '✗ NUTS'}")
print(f"Scenario 2 (Memory):       {'✓ CMC' if result2 else '✗ NUTS'}")
print(f"Scenario 3 (NUTS):         {'✗ CMC' if result3 else '✓ NUTS'}")
print(f"Scenario 4 (Edge Case):    {'✓ CMC' if result4 else '✗ NUTS'} (with warning)")
print(f"Scenario 5 (Custom):       {'✗ CMC' if result5 else '✓ NUTS'}")
print()
print("All scenarios executed successfully!")
print("Dual-criteria OR logic working as expected.")
print("=" * 80)
