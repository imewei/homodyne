# Homodyne v2.3 Examples

This directory contains practical examples demonstrating how to use Homodyne for XPCS analysis.

**Note:** GPU support has been removed in v2.3.0. All examples are CPU-optimized. See [GPU Removal Notice](#gpu-removal-notice) below.

## Quick Start

Each example is a standalone script that demonstrates specific analysis patterns:

```bash
# Run any example directly
python examples/static_isotropic_nlsq.py
python examples/cpu_optimization.py
python examples/multi_core_batch_processing.py
```

Or use the Makefile:

```bash
make run-example  # Runs the current default example
```

## Example Index

### CPU-Optimized Examples (v2.3.0+)

#### 1. `cpu_optimization.py` (NEW)
**Best for:** HPC cluster users, multi-core personal computers

Demonstrates CPU-optimized XPCS analysis with:
- Multi-core CPU thread management
- HPC cluster configuration (Slurm, PBS, LSF)
- CPU performance benchmarking
- NUMA-aware optimization for large systems
- JAX XLA CPU-specific optimizations
- Example job submission scripts

**Key Topics:**
- Detecting HPC environment and CPU capabilities
- Configuring OpenMP threads and XLA flags
- Batch sizing for optimal throughput
- Example Slurm/PBS job scripts

**Run it:**
```bash
python examples/cpu_optimization.py
```

**Expected output:**
- CPU information detection
- Performance benchmarking results
- Example HPC submission scripts
- Configuration recommendations

---

#### 2. `multi_core_batch_processing.py` (NEW)
**Best for:** Processing multiple datasets efficiently, batch analysis

Demonstrates parallel processing patterns with:
- Multi-core batch processing of multiple datasets
- Intelligent work distribution across cores
- Memory-efficient parallel execution
- Progress tracking for long-running jobs
- Result aggregation and JSON output
- Scaling analysis and performance profiling

**Key Topics:**
- Parallel dataset processing with ProcessPoolExecutor
- Adaptive worker scaling based on CPU cores
- Performance monitoring and reporting
- Fault-tolerant batch execution
- Memory management for parallel workloads

**Run it:**
```bash
python examples/multi_core_batch_processing.py
```

**Expected output:**
- Parallel processing progress
- Per-dataset timing and throughput
- Performance summary and scaling analysis
- JSON results file: `batch_processing_results.json`

---

### Standard NLSQ Examples

#### 3. `static_isotropic_nlsq.py`
**Best for:** First-time users, simple isotropic systems

Single dataset NLSQ optimization for static isotropic systems:
- 3 physical parameters (D₀, α, D_offset)
- Per-angle scaling parameters
- Configuration setup and parameter bounds

---

#### 4. `laminar_flow_nlsq.py`
**Best for:** Laminar flow analysis, shear-induced alignment

NLSQ optimization for laminar flow systems:
- 7 physical parameters (includes flow parameters)
- Anisotropic diffusion analysis
- Flow effect on diffusion

---

### MCMC & Uncertainty Quantification

#### 5. `mcmc_uncertainty.py`
**Best for:** Uncertainty quantification, posterior distributions

Bayesian MCMC sampling with NumPyro:
- NUTS sampler for small datasets
- CMC parallel sampling for large datasets
- Posterior distribution analysis
- Posterior predictive checks

---

#### 6. `mcmc_integration_demo.py`
**Best for:** Combining NLSQ results with MCMC, workflow demonstrations

Integrated workflow showing:
- NLSQ optimization → MCMC initialization
- Manual parameter transfer between methods
- Posterior sampling with informed priors

---

### Advanced Examples

#### 7. `cmc_large_dataset.py`
**Best for:** Large datasets (>15 samples), CMC parallelization

CMC (Correlated Monte Carlo) parallel sampling:
- Many-core parallelization strategies
- Memory management for large datasets
- Convergence monitoring

---

#### 8. `streaming_100m_points.py`
**Best for:** Very large datasets (100M+ points), memory constraints

Streaming optimization for massive datasets:
- Chunked data processing
- Memory-efficient streaming strategies
- Checkpoint/resume capability

---

#### 9. `angle_filtering.py`
**Best for:** Selective angle analysis, data filtering

Phi angle filtering and selection:
- Filtering by angle range
- Weighted angle filtering
- Selective analysis

---

## GPU Removal Notice

**Version 2.3.0: GPU Support Removed**

GPU acceleration support has been removed in v2.3.0 to:
- Simplify maintenance and reduce failure modes
- Focus on reliable CPU-only execution
- Remove GPU memory limitations encountered with large datasets
- Encourage multi-core CPU optimization instead

### Migration Guide

**If you were using GPU acceleration (v2.2.x):**

1. **Option 1: Stay on v2.2.x**
   - v2.2.x continues to support GPU
   - Install with: `pip install 'homodyne<2.3'`

2. **Option 2: Upgrade to v2.3 with CPU optimization**
   - Remove GPU-related config options
   - Use CPU optimization examples
   - Leverage multi-core parallelism (see `multi_core_batch_processing.py`)
   - Typical performance: 1-3 hours for 10M point analysis on 36-core HPC

3. **Option 3: Use JAX GPU capabilities**
   - JAX still has native GPU support
   - Homodyne uses CPU-only configuration by default
   - See JAX documentation for manual GPU configuration

### Removed Features

The following GPU-specific features were removed:

- `--force-cpu` CLI flag (CPU is now the only option)
- `--gpu-memory-fraction` CLI flag
- `homodyne.runtime.gpu` module
- `homodyne.device.gpu` module
- GPU configuration keys in YAML
- GPU-related Makefile targets (`install-jax-gpu`, `gpu-check`, `test-gpu`)
- GPU system validation tests

### CPU Alternatives

For improved throughput on CPU systems:

1. **Multi-Core Parallelism** (see `multi_core_batch_processing.py`)
   - Process multiple datasets in parallel
   - Linear scaling up to ~36 cores

2. **HPC Cluster Submission** (see `cpu_optimization.py`)
   - Run on dedicated HPC nodes
   - Automatic thread management

3. **Batch Processing with Slurm/PBS**
   - Submit jobs to queue systems
   - Example scripts included

## Configuration & Setup

### Basic Configuration

All examples use configuration overrides. To use YAML configuration files:

```bash
homodyne --config my_config.yaml
```

See `homodyne/config/templates/` for example YAML templates:
- `homodyne_static.yaml` - Static isotropic analysis
- `homodyne_laminar_flow.yaml` - Laminar flow analysis
- `homodyne_master_template.yaml` - Full template with all options

### CPU Optimization Flags

For multi-core optimization, set environment variables:

```bash
# Auto-detect optimal thread count
export OMP_NUM_THREADS=<num_cores_minus_1>

# Enable XLA CPU optimizations
export XLA_FLAGS="--xla_cpu_enable_fast_math=true --xla_cpu_multi_thread_eigen=true"

# Run analysis
homodyne --config config.yaml
```

### HPC Cluster Setup

Slurm example:
```bash
#SBATCH --cpus-per-task=36
#SBATCH --mem=128G

export OMP_NUM_THREADS=36
homodyne --config config.yaml
```

PBS example:
```bash
#PBS -l select=1:ncpus=128:mem=256gb

export OMP_NUM_THREADS=128
homodyne --config config.yaml
```

## Running Examples

### Individual Examples

```bash
# Static isotropic analysis
python examples/static_isotropic_nlsq.py

# CPU-optimized analysis with HPC tips
python examples/cpu_optimization.py

# Parallel batch processing
python examples/multi_core_batch_processing.py

# Other analyses
python examples/laminar_flow_nlsq.py
python examples/mcmc_uncertainty.py
```

### Using Makefile

```bash
# Run default example
make run-example

# Run all tests
make test

# Run example with verbose output
python examples/static_isotropic_nlsq.py --verbose
```

## Expected Output

Each example creates output in its current directory:

- **Static/Laminar NLSQ:** `results_<mode>/` directory
- **MCMC Examples:** HDF5 results file with posterior samples
- **Batch Processing:** `batch_results/batch_processing_results.json`
- **CPU Optimization:** Console output with recommendations

## Performance Metrics

Typical performance on different systems (NLSQ optimization):

| System | Configuration | 100k Points | 1M Points | 10M Points |
|--------|---------------|-------------|-----------|------------|
| Laptop | 8 core i7 | <1s | 5-10s | 1-2 min |
| Workstation | 16 core Ryzen | <1s | 2-5s | 30-60s |
| HPC 36-core | Xeon E5 | <1s | 1-3s | 10-20s |
| HPC 128-core | EPYC | <1s | 1s | 5-10s |

*Timing varies by algorithm complexity and number of angles*

## Troubleshooting

### Common Issues

**ImportError: Cannot import homodyne**
```bash
# Install in development mode
pip install -e /path/to/homodyne[dev]
```

**Example runs very slowly**
```bash
# Check CPU thread count
python -c "import os; print(f'OMP_NUM_THREADS={os.environ.get(\"OMP_NUM_THREADS\", \"not set\")}')"

# Set optimal for your system
export OMP_NUM_THREADS=14  # For 16-core system
```

**Out of memory errors**
```bash
# Use streaming strategy for very large datasets
# See streaming_100m_points.py example
```

**Parallel batch processing slower than sequential**
```bash
# Check if max_workers is set correctly
# Too many workers causes context switching overhead
# Rule of thumb: num_workers = num_cores - 1 or - 2
```

## Next Steps

1. **Try the examples** in order: `static_isotropic_nlsq.py` → `cpu_optimization.py` → `multi_core_batch_processing.py`

2. **Adapt to your data:** Replace synthetic data with real HDF5 files

3. **Optimize for your system:** Adjust thread counts and batch sizes

4. **Scale up:** Use HPC cluster submission for production analysis

5. **Add post-processing:** Extend examples with custom plotting and analysis

## Documentation

For more information:
- Main documentation: `../docs/`
- API reference: `../docs/api/`
- Configuration guide: `../homodyne/config/templates/`
- CLAUDE.md: Development notes and architecture

## Support

For issues or questions:
- Check example docstrings for detailed explanations
- Review CLAUDE.md for architecture and known issues
- Run system validation: `python -m homodyne.runtime.utils.system_validator`
- Check GPU removal migration guide (see above)
