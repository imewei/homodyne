#!/usr/bin/env python
"""
Example 6: GPU Acceleration

Demonstrates GPU acceleration for performance improvement (Linux only).

Key concepts:
- Automatic GPU detection and usage
- GPU memory management
- Performance benchmarking
- CPU fallback on unavailable GPU

Requirements:
- Linux x86_64 or aarch64
- CUDA 12.1-12.9 (pre-installed)
- NVIDIA driver >= 525
"""

from pathlib import Path

CONFIG_GPU = """
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

performance:
  device:
    preferred_device: "gpu"     # Force GPU if available
    gpu_memory_fraction: 0.9    # Use 90% of GPU memory

output:
  directory: "./results_gpu"
"""


def main():
    """Run GPU acceleration example."""
    print("GPU Acceleration Example (Linux Only)")
    print("=" * 60)

    config_path = Path("homodyne_config_gpu.yaml")
    config_path.write_text(CONFIG_GPU)
    print(f"✓ Created configuration: {config_path}")

    print("\nGPU Requirements:")
    print("  - OS: Linux x86_64 or aarch64 only")
    print("  - CUDA: 12.1-12.9 (pre-installed on system)")
    print("  - NVIDIA driver: >= 525")
    print("  - JAX: Installed with CUDA support")

    print("\nGPU Configuration:")
    print("  Device: GPU (auto-selected if available)")
    print("  GPU memory: 90% of available")
    print("  Fallback: CPU if GPU unavailable")

    print("\nVerification steps:")
    print("  1. Check CUDA: nvcc --version")
    print("  2. Check GPU: nvidia-smi")
    print('  3. Check JAX: python -c "import jax; print(jax.devices())"')
    print("  4. Check homodyne: make gpu-check")

    print("\nExpected speedup:")
    print("  - Small datasets (< 10M): CPU often faster")
    print("  - Large datasets (> 10M): GPU 2-10x faster")
    print("  - Very large (> 100M): GPU 5-20x faster")

    print("\nTo run:")
    print(f"  homodyne --config {config_path}")

    print("\nBenchmarking performance:")
    print("  from homodyne.device import benchmark_device_performance")
    print("  results = benchmark_device_performance()")
    print("  print(results)")

    print("\n✓ GPU configuration ready!")


if __name__ == "__main__":
    main()
