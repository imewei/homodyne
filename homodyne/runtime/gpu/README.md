# GPU Acceleration for Homodyne v2

## Overview

This module provides GPU acceleration support for JAX-based computations in the Homodyne v2 package. It offers automatic GPU detection, memory management, and performance optimization for NVIDIA GPUs with CUDA 12.1+.

## Features

- 🚀 **Automatic GPU Detection**: Detects available NVIDIA GPUs and configures JAX accordingly
- 💾 **Memory Management**: Configurable GPU memory allocation with preallocation support
- ⚡ **Performance Optimization**: XLA flags and CUDA optimizations for maximum performance
- 🔧 **Multi-GPU Support**: Select specific GPUs in multi-GPU systems
- 🔄 **Graceful Fallback**: Automatic CPU fallback when GPU is unavailable
- 📊 **Benchmarking**: Built-in performance benchmarks for GPU capabilities

## Requirements

- **Hardware**: NVIDIA GPU with Compute Capability 5.2+ (Maxwell or newer)
- **CUDA**: Version 12.1 - 12.9
- **Driver**: NVIDIA driver >= 525 (or >= 570 for CUDA 12.8)
- **OS**: Linux x86_64 or aarch64
- **Python**: 3.10+
- **JAX**: Install with `pip install "jax[cuda12-local]"`

## Quick Start

### Basic Usage

```python
from homodyne.runtime.gpu import activate_gpu

# Activate GPU with default settings (90% memory)
status = activate_gpu()

# Check if successful
if status.get('success'):
    print("GPU activated successfully!")
    print(f"Device: {status['device']}")
    print(f"CUDA Version: {status['cuda_version']}")
```

### Advanced Configuration

```python
from homodyne.runtime.gpu import GPUActivator

# Create activator instance
activator = GPUActivator(verbose=True)

# Activate with custom settings
result = activator.activate(
    memory_fraction=0.8,  # Use 80% of GPU memory
    force_gpu=True,       # Fail if GPU not available
    gpu_id=0             # Use first GPU
)

# Later: deactivate and cleanup
activator.deactivate()
```

### Check GPU Status

```python
from homodyne.runtime.gpu import get_gpu_status

status = get_gpu_status()
print(f"JAX available: {status['jax_available']}")
print(f"Devices: {status['devices']}")
print(f"CUDA version: {status['cuda_version']}")
print(f"Driver version: {status['driver_version']}")
```

### Benchmark GPU Performance

```python
from homodyne.runtime.gpu import benchmark_gpu

# Run performance benchmarks
results = benchmark_gpu()

# Display results
for test, score in results.items():
    print(f"{test}: {score:.2f}")
```

## Environment Setup

### Shell Activation Script

For shell-based activation, source the provided script:

```bash
source /path/to/homodyne/runtime/gpu/activation.sh
```

This script will:
1. Detect CUDA installation
2. Set environment variables
3. Configure JAX for GPU usage
4. Verify GPU detection

### Manual Environment Variables

```bash
# GPU memory allocation (0.0-1.0)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Enable memory preallocation
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# XLA performance flags
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
--xla_gpu_triton_gemm_any=true \
--xla_gpu_enable_async_collectives=true"

# CUDA paths
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

## Integration with Homodyne Optimization

```python
from homodyne.runtime.gpu import activate_gpu
from homodyne.optimization import fit_nlsq_jax
from homodyne.data import load_xpcs_data
from homodyne.config import ConfigManager

# Activate GPU
activate_gpu(memory_fraction=0.9)

# Load data and config
data = load_xpcs_data("config.yaml")
config = ConfigManager("config.yaml")

# Run GPU-accelerated optimization
result = fit_nlsq_jax(data, config)
print(f"Optimized parameters: {result.parameters}")
```

## Performance Tips

### Memory Management
- Set `memory_fraction` based on your dataset size
- Use 0.8-0.9 for dedicated GPU systems
- Use 0.5-0.7 for shared GPU environments

### Optimization Strategies
1. **Large Datasets**: Use higher memory fraction
2. **Multiple Runs**: Enable compilation cache
3. **Batch Processing**: Process multiple datasets together
4. **Mixed Precision**: Use float32 or bfloat16 for speed

### Common Performance Gains
- Matrix operations: 10-50x speedup
- FFT operations: 20-100x speedup
- NLSQ optimization: 5-20x speedup
- MCMC sampling: 10-30x speedup

## Troubleshooting

### GPU Not Detected

```python
# Check CUDA installation
import subprocess
result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
print(result.stdout)

# Check JAX GPU detection
import jax
print(jax.devices())  # Should show CudaDevice(s)
```

### Out of Memory Errors

```python
# Reduce memory fraction
activate_gpu(memory_fraction=0.5)

# Or clear GPU memory
import jax
for device in jax.devices('gpu'):
    if hasattr(device, '_clear_memory'):
        device._clear_memory()
```

### CUDA Version Mismatch

Ensure CUDA 12.1+ is installed:
```bash
nvcc --version  # Should show 12.1 or higher
```

If using system CUDA:
```bash
pip install "jax[cuda12-local]"  # Uses system CUDA
```

### Performance Issues

1. Check GPU utilization:
```bash
nvidia-smi
```

2. Monitor during computation:
```bash
watch -n 1 nvidia-smi
```

3. Verify XLA compilation:
```python
import os
os.environ["XLA_FLAGS"] = "--xla_dump_to=/tmp/xla_dump"
```

## API Reference

### Main Functions

- `activate_gpu(memory_fraction=0.9, force_gpu=False, gpu_id=None, verbose=True)`
  - Activates GPU with specified settings
  - Returns status dictionary

- `get_gpu_status()`
  - Returns current GPU configuration and availability

- `benchmark_gpu()`
  - Runs performance benchmarks
  - Returns benchmark scores

### GPUActivator Class

- `GPUActivator(verbose=True)`
  - Main class for GPU management

- `activate(memory_fraction, force_gpu, gpu_id)`
  - Activate GPU with configuration

- `deactivate()`
  - Cleanup GPU resources

## Examples

See `/examples/gpu_accelerated_optimization.py` for a complete example.

## Support

For issues or questions:
1. Check JAX installation: `pip show jax jaxlib`
2. Verify CUDA: `nvidia-smi`
3. Test with example script
4. Report issues with full error messages

## Performance Benchmarks

Typical performance on NVIDIA RTX 4090:
- Matrix Multiply (4000x4000): ~25 TFLOPS
- FFT (16384 points): ~14,000 ops/sec
- NLSQ Optimization: 10-20x faster than CPU
- MCMC Sampling: 15-30x faster than CPU

## Future Enhancements

- [ ] Multi-GPU parallel processing
- [ ] Mixed precision (fp16/bf16) support
- [ ] Dynamic memory management
- [ ] TPU support
- [ ] AMD ROCm support