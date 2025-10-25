#!/usr/bin/env python
"""
Example 5: Streaming Optimization for 100M+ Points

Demonstrates streaming optimization for very large datasets
with constant memory footprint.

Key concepts:
- StreamingOptimizer for unlimited dataset sizes
- Constant memory usage
- Checkpoint/resume capability
- Automatic error recovery and retry logic
- Suitable for > 100M point datasets
"""

from pathlib import Path

CONFIG_STREAMING = """
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

  streaming:
    enable_checkpoints: true
    checkpoint_dir: "./checkpoints"
    checkpoint_frequency: 10      # Save every N batches
    resume_from_checkpoint: true   # Resume from last checkpoint
    keep_last_checkpoints: 3       # Keep last 3 checkpoints
    enable_fault_tolerance: true   # Enable error recovery
    max_retries_per_batch: 2       # Retry attempts
    min_success_rate: 0.5          # Minimum batch success rate

performance:
  strategy_override: "streaming"   # Force streaming mode
  memory_limit_gb: null            # Auto-detect
  enable_progress: true

output:
  directory: "./results_streaming"
"""


def main():
    """Run streaming optimization example."""
    print("Streaming Optimization (100M+ Points) Example")
    print("=" * 60)

    config_path = Path("homodyne_config_streaming.yaml")
    config_path.write_text(CONFIG_STREAMING)
    print(f"✓ Created configuration: {config_path}")

    print("\nStreaming Configuration:")
    print("  Strategy: Streaming (unlimited data)")
    print("  Checkpoints: Enabled (HDF5 format)")
    print("  Checkpoint frequency: Every 10 batches")
    print("  Resume capability: Enabled")
    print("  Fault tolerance: Enabled")

    print("\nKey features:")
    print("  - Constant memory usage regardless of data size")
    print("  - Intelligent batch sizing (1K-100K points)")
    print("  - Checkpoint/resume for long-running jobs")
    print("  - Automatic error recovery")
    print("  - Batch-level monitoring")

    print("\nCheckpoint directory structure:")
    print("  ./checkpoints/")
    print("  ├── checkpoint_batch_10.h5")
    print("  ├── checkpoint_batch_20.h5")
    print("  └── checkpoint_batch_30.h5")

    print("\nBatch-level success monitoring:")
    print("  - Tracks success rate across batches")
    print("  - Triggers error recovery if needed")
    print("  - Maintains best parameters found so far")

    print("\nExpected performance:")
    print("  - Overhead: < 5% with full fault tolerance")
    print("  - Checkpoint saves: < 2 seconds per save")
    print("  - Validated up to 1B points")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\n✓ Streaming configuration ready!")


if __name__ == "__main__":
    main()
