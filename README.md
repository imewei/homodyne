# Homodyne v2.1: JAX-First XPCS Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.1.0-green.svg)](#)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://homodyne.readthedocs.io)
[![ReadTheDocs](https://readthedocs.org/projects/homodyne/badge/?version=latest)](https://homodyne.readthedocs.io/en/latest/)
[![GitHub Actions](https://github.com/imewei/homodyne/actions/workflows/docs.yml/badge.svg)](https://github.com/imewei/homodyne/actions/workflows/docs.yml)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

## 🎉 v2.2.0 New Features

**Angle-Stratified Chunking** - Automatic fix for per-angle scaling on large datasets (>1M points)

Key improvements:
- **Automatic activation**: No configuration required - activates when `per_angle_scaling=True` AND `n_points>=100k`
- **Zero regressions**: 100% backward compatible with existing configurations
- **Performance**: <1% overhead (0.15s for 3M points), sub-linear O(n^1.01) scaling
- **Reliability**: Fixes silent optimization failures caused by arbitrary chunking
- **Fallbacks**: Sequential per-angle optimization for extreme angle imbalance (>5.0 ratio)

See [v2.2 Release Notes](docs/releases/v2.2-stratification-release-notes.md) for complete details.

## ⚠️ v2.1.0 Breaking Changes

**If upgrading from v2.0.x, please read the [Migration Guide](docs/migration/v2.0-to-v2.1.md).**

Key changes:
- **CLI**: Only `--method nlsq` and `--method mcmc` supported (automatic NUTS/CMC selection)
- **Config**: Removed `mcmc.initialization` section; added `min_samples_for_cmc`, `memory_threshold_pct`
- **API**: Updated `fit_mcmc_jax()` signature with new `parameter_space` and `initial_values` parameters

**High-performance JAX-first package for X-ray Photon Correlation Spectroscopy (XPCS)
analysis**, implementing the theoretical framework from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing
transport properties in flowing soft matter systems through time-dependent intensity
correlation functions.

📚 **[Read the Full Documentation](https://homodyne.readthedocs.io)** | **[Quick Start Guide](https://homodyne.readthedocs.io/en/latest/user-guide/quickstart.html)** | **[API Reference](https://homodyne.readthedocs.io/en/latest/api-reference/index.html)**

A completely rebuilt homodyne package with JAX-first architecture, optimized for HPC and
supercomputer environments.

## Architecture Overview

This rebuild achieves **~70% complexity reduction** while maintaining **100% API
compatibility** for all validated components:

### Preserved Components

- **`data/`** - Complete XPCS data loading infrastructure (11 files preserved)
- **`core/`** - Validated physics models and JAX backend (7 files preserved)

### New Components

- **`optimization/`** - JAX-first optimization with **NLSQ trust-region solver** +
  NumPyro/BlackJAX MCMC
  - ✅ **Scientifically validated** (7/7 validation tests passed)
  - ✅ **Production-ready** with intelligent error recovery
- **`device/`** - HPC CPU optimization for multi-core systems
- **`config/`** - Streamlined YAML-only configuration system with parameter management
- **`utils/`** - Minimal logging with preserved API signatures
- **`cli/`** - Essential command-line interface

## Key Features

### JAX-First Computational Engine

- **Primary**: NLSQ trust-region nonlinear least squares (JAX-native optimizer)
- **Secondary**: NumPyro/BlackJAX NUTS sampling for uncertainty quantification
- **Core Equation**: `c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²`

### HPC CPU Optimization (v2.3.0+)

- **CPU-only architecture** for 36/128-core HPC nodes
- Intelligent thread allocation and NUMA-aware configuration
- Memory-efficient processing for large datasets
- Optimized for multi-core personal computers and CPU clusters

### Intelligent Resource Allocation

- **CPU processing** for all operations (optimization, data loading, plotting)
- **Parallel CPU workers** for efficient batch processing
- **Memory management** optimized for large datasets without GPU constraints

### Consensus Monte Carlo (CMC) for Large-Scale Bayesian Inference

**New in v2.0**: Scalable Bayesian uncertainty quantification with **dual-criteria automatic selection**

**Automatic NUTS/CMC Selection** (v2.1.0 - October 2025):
- **Sample-based parallelism**: Triggers when `num_samples >= 15` (e.g., 20+ phi angles)
  - Optimized for multi-core CPU workloads (14+ cores)
  - Example: 20 samples on 14-core CPU → ~1.4x speedup via CMC
- **Memory management**: Triggers when `memory usage > 30%` of available memory
  - Conservative OOM prevention for large datasets
  - Example: 100M+ points → automatic data sharding
- **Simplified CLI**: Only `--method mcmc` needed (automatic NUTS/CMC selection)
- **No initialization required**: Physics-informed priors from `ParameterSpace`

**Key Features:**
- **Dual-criteria OR logic**: Triggers on EITHER many samples OR large memory footprint
- **Hardware-adaptive**: Automatic backend selection (pjit/multiprocessing/PBS/Slurm)
- **Linear speedup**: Perfect parallelization across CPU cores or cluster nodes
- **Memory efficient**: Each shard fits in available memory with 40% safety margin
- **Production-ready**: Comprehensive validation, fault tolerance, and convergence diagnostics

**Quick Example:**

```python
from homodyne.optimization.mcmc import fit_mcmc_jax
from homodyne.config.parameter_space import ParameterSpace

# Example 1: Many samples (23 angles) → CMC for parallelism
parameter_space = ParameterSpace.from_config(config_dict)
result = fit_mcmc_jax(
    data=data['c2'],
    t1=data['t1'], t2=data['t2'], phi=data['phi'],  # 23 angles
    q=0.0054, L=2000000,
    analysis_mode='static_isotropic',
    parameter_space=parameter_space,
    initial_values={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
    # Automatic NUTS/CMC selection: ≥15 samples OR ≥30% memory → CMC
)

# Example 2: Few samples, huge dataset → CMC for memory
result = fit_mcmc_jax(
    data=huge_data['c2'],  # 100M+ points
    t1=huge_data['t1'], t2=huge_data['t2'], phi=huge_data['phi'],  # 2 angles
    parameter_space=parameter_space,
    # Automatic selection: 2 samples < 15 but memory > 30% → CMC for OOM prevention
)

# Check which method was used
if result.is_cmc_result():
    print(f"✓ CMC used with {result.num_shards} shards")
    print(f"  Reason: {result.cmc_trigger_reason}")  # 'parallelism' or 'memory'
```

**Performance:**

| Scenario | Samples | Data Size | Method | Runtime | Speedup |
|----------|---------|-----------|--------|---------|---------|
| Multi-core CPU (14 cores) | 23 | 50M | CMC | ~40 min | 1.4x |
| Multi-core CPU (14 cores) | 23 | 50M | NUTS | ~60 min | baseline |
| HPC CPU (36 cores) | 5 | 200M | CMC | ~2 hours | 1.5x |
| HPC CPU (36 cores) | 5 | 200M | NUTS | ~3 hours | baseline |

**Documentation:**
- Architecture Guide: [`docs/architecture/cmc-dual-mode-strategy.md`](docs/architecture/cmc-dual-mode-strategy.md)
- Quick Reference: [`docs/architecture/cmc-decision-quick-reference.md`](docs/architecture/cmc-decision-quick-reference.md)
- User Guide: [`docs/advanced-topics/cmc-large-datasets.rst`](docs/advanced-topics/cmc-large-datasets.rst)
- MCMC Guide: [`docs/advanced-topics/mcmc-uncertainty.rst`](docs/advanced-topics/mcmc-uncertainty.rst)

## Platform Support

### CPU-Only Architecture (All Platforms) ✅

**v2.3.0+**: GPU support removed - CPU-optimized for all platforms

- **Linux**: Full support (HPC clusters recommended)
- **macOS**: Full support
- **Windows**: Full support
- **Python**: 3.12+

**For GPU users**: Use Homodyne v2.2.1 (last GPU-supporting version)

## Installation

### Quick Install (All Platforms)

```bash
# CPU-only installation (v2.3.0+)
# Works on Linux, macOS, Windows
pip install homodyne
```

This installs Homodyne with CPU-optimized JAX 0.8.0, suitable for:
- Development and prototyping
- Datasets up to 100M points on multi-core CPUs
- HPC clusters with 36-128 CPU cores

### Migration from GPU to CPU-Only

**If upgrading from v2.2.x (GPU version):**

```bash
# Uninstall GPU JAX
pip uninstall -y jax jaxlib

# Install CPU-only Homodyne v2.3.0+
pip install homodyne

# Verify CPU devices
python -c "import jax; print('Devices:', jax.devices())"
# Expected: [CpuDevice(id=0)]
```

### For GPU Users

**GPU support removed in v2.3.0.** If you need GPU acceleration:

1. **Stay on v2.2.1** (last GPU-supporting version):
   ```bash
   pip install homodyne==2.2.1
   ```

2. **See migration guide**: [`docs/migration/v2.2-to-v2.3-gpu-removal.md`](docs/migration/v2.2-to-v2.3-gpu-removal.md)

### System Validation

After installation, verify your system:

```bash
python -m homodyne.runtime.utils.system_validator --quick
```

Expected output: 9 tests passing (90-100% health score)

### Development Installation

```bash
# CPU-only installation (all platforms)
pip install homodyne[dev]

# Or from source
git clone https://github.com/your-org/homodyne.git
cd homodyne
pip install -e ".[dev]"
```

### HPC Environment Setup

For HPC clusters with multi-core CPUs:

```bash
# Load Python module (site-specific)
module load python/3.12

# Create virtual environment
python -m venv homodyne-env
source homodyne-env/bin/activate

# Install homodyne
pip install homodyne[dev]

# Configure CPU threads for your HPC node
export OMP_NUM_THREADS=34  # Reserve 2 cores for OS on 36-core node
```

### Makefile Shortcuts (Development)

```bash
make dev               # Install dev environment (CPU-only)
make test              # Run core test suite
make quality           # Format, lint, type-check
```

## Quick Start

### Command Line Interface

```bash
# NLSQ optimization (default method)
homodyne --method nlsq --config config.yaml

# MCMC sampling for uncertainty quantification (v2.1.0: automatic NUTS/CMC selection)
homodyne --method mcmc --config config.yaml

# Force CPU-only computation
homodyne --method nlsq --force-cpu

# Custom output directory
homodyne --method nlsq --output-dir ./results
```

### Manual NLSQ → MCMC Workflow (v2.1.0)

**Step-by-step process:**

```bash
# 1. Run NLSQ first
homodyne --method nlsq --config config.yaml

# 2. Copy best-fit results from output:
#    D0: 1234.5 ± 45.6
#    alpha: 0.567 ± 0.012
#    D_offset: 12.34 ± 1.23

# 3. Update config.yaml:
#    initial_parameters:
#      values: [1234.5, 0.567, 12.34]

# 4. Run MCMC with initialized parameters
homodyne --method mcmc --config config.yaml
```

### Python API

```python
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.data import load_xpcs_data
from homodyne.config import ConfigManager, ParameterSpace

# Load data and configuration
data = load_xpcs_data("config.yaml")
config = ConfigManager("config.yaml")

# Primary: NLSQ optimization (JAX-native trust-region solver)
result = fit_nlsq_jax(data, config)
print(f"Parameters: {result.parameters}")
print(f"Chi-squared: {result.chi_squared:.4f}")
print(f"Convergence: {result.convergence_status}")

# Secondary: MCMC sampling for uncertainty quantification (v2.1: automatic NUTS/CMC selection)
parameter_space = ParameterSpace.from_config(config.to_dict())
initial_values = config.get_initial_parameters()

mcmc_result = fit_mcmc_jax(
    data=data,
    parameter_space=parameter_space,
    initial_values=initial_values,
    # Automatic NUTS/CMC selection based on dataset characteristics
)
print(f"Posterior means: {mcmc_result.mean_params}")
```

### Device Configuration

```python
from homodyne.device import configure_optimal_device

# Auto-detect and configure optimal device
device_config = configure_optimal_device()

# Configure CPU-only device (v2.3.0+)
cpu_config = configure_optimal_device()
```

### Interactive Example

For a complete interactive tutorial, see the
[NLSQ Optimization Example Notebook](examples/nlsq_optimization_example.ipynb):

```bash
# Launch Jupyter and open the example notebook
jupyter notebook examples/nlsq_optimization_example.ipynb
```

The notebook covers:

- Synthetic XPCS data generation with ground truth parameters
- NLSQ optimization workflow with parameter recovery
- Fit quality visualization and residual analysis
- Parameter uncertainty quantification
- Error recovery demonstration

## Shell Completion & CLI Tools

Homodyne provides four CLI commands and intelligent shell completion for faster workflows.

### Available Commands

- **`homodyne`** - Run XPCS analysis (NLSQ/MCMC)
- **`homodyne-config`** - Generate and validate configuration files
- **`homodyne-post-install`** - Install shell completion (bash/zsh/fish)
- **`homodyne-cleanup`** - Remove shell completion scripts

### Quick Shell Completion Setup

Install intelligent tab completion with aliases:

```bash
# Interactive setup (recommended)
homodyne-post-install --interactive

# Or quick one-liner
homodyne-post-install --shell $(basename $SHELL)
```

**Conda/Mamba users:** Completion auto-activates with your environment - no extra setup needed!

**uv/venv/virtualenv users:** Add activation to your shell RC file:

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.bashrc
source ~/.bashrc
```

**Supported shells:** bash, zsh, fish (PowerShell not supported)

### Convenient Aliases

After installing completion, use short aliases:

```bash
hm-nlsq --config config.yaml           # homodyne --method nlsq
hm-mcmc --config config.yaml           # homodyne --method mcmc

hc-stat --output static.yaml           # homodyne-config --mode static
hc-flow --output flow.yaml             # homodyne-config --mode laminar_flow

hconfig --validate my_config.yaml      # homodyne-config --validate
```

**Smart completion examples:**

```bash
homodyne --config <TAB>        # Shows *.yaml files
homodyne --method <TAB>        # Shows: nlsq, mcmc (v2.1.0: nuts/cmc removed)
hm-nlsq --<TAB>                # Shows all available options
```

**See documentation:** [Shell Completion Guide](https://homodyne.readthedocs.io/en/latest/user-guide/shell-completion.html)

## Analysis Modes

### Static Isotropic (3 parameters)

- Parameters: `[D₀, α, D_offset]`
- Fast analysis for isotropic systems
- Ideal for quick parameter estimation

### Laminar Flow (7 parameters)

- Parameters: `[D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]`
- Full anisotropic analysis with shear flow
- Comprehensive characterization of complex systems

## Configuration

The package uses YAML-based configuration with preserved template compatibility:

```yaml
# config.yaml
analysis_mode: "static_isotropic"
experimental_data:
  file_path: "data.h5"
optimization:
  method: "nlsq"
  lsq:
    max_iterations: 10000
    tolerance: 1e-8
hardware:
  force_cpu: false
  gpu_memory_fraction: 0.9
```

## Performance Characteristics

### Optimization Methods

| Method | Speed | Accuracy | Use Case | |--------|-------|----------|----------| |
**NLSQ** | Fast | Excellent | Production workflows, real-time analysis | | **MCMC** |
Slower | Excellent | Publication-quality, uncertainty quantification |

### Validated Performance Benchmarks

Based on comprehensive scientific validation (T036-T041):

| Dataset Size | Points | Optimization Time | Throughput | Convergence |
|--------------|--------|-------------------|------------|-------------| | Small | 500 |
1.6s | 317 pts/s | 100% | | Medium | 4,000 | 1.5s | 2,758 pts/s | 100% | | Large | 9,375
| 1.6s | 5,977 pts/s | 100% |

**Key Performance Features**:

- ✅ **Sub-linear time scaling**: Near-constant execution time across dataset sizes
- ✅ **Parameter recovery accuracy**: 2-14% error on core parameters
- ✅ **Numerical stability**: \<4% parameter deviation across initial conditions
- ✅ **Physics compliance**: 100% constraint satisfaction rate

### Adaptive Subsampling for Large Datasets

Homodyne v2 includes **intelligent two-layer subsampling** to handle very large datasets
(>50M points) while preserving XPCS correlation structure and minimizing accuracy loss.

**Two-Layer Defense System**:

- **Layer 1 (Homodyne)**: Physics-aware logarithmic subsampling

  - Threshold: 50M points
  - Reduction: 2-4x adaptive (conservative)
  - Sampling: Logarithmic (dense at short times, sparse at long times)
  - Preserves: XPCS correlation decay structure

- **Layer 2 (NLSQ)**: Memory fallback with uniform sampling

  - Threshold: 150M points
  - Reduction: 2x maximum
  - Sampling: Uniform (preserves time ordering)
  - Activates: Only if Layer 1 insufficient

**Subsampling Behavior by Dataset Size**:

```text
Dataset Size | Layer 1 (Homodyne) | Layer 2 (NLSQ)    | Total Reduction | Final Size
-------------|--------------------|--------------------|-----------------|------------
23M          | Not triggered      | Not triggered      | 1x (none)       | 23M
100M         | 100M → 50M (2x)    | Not triggered      | 2x              | 50M
200M         | 200M → 50M (4x)    | Not triggered      | 4x              | 50M
500M         | 500M → 125M (4x)   | 125M → 62.5M (2x)  | 8x              | 62.5M
1000M        | 1000M → 250M (4x)  | 250M → 125M (2x)   | 8x              | 125M
```

**Configuration**:

```yaml
performance:
  subsampling:
    enabled: true              # Enable adaptive subsampling
    trigger_threshold_points: 50000000  # 50M threshold
    max_reduction_factor: 4    # Maximum 4x reduction (conservative)
    method: "logarithmic"      # Preserves XPCS structure
    preserve_edges: true       # Keep t_min and t_max
```

**Key Benefits**:

- ✅ **Minimal accuracy loss**: 2-4x reduction (vs 10x in aggressive methods)
- ✅ **Physics-aware**: Logarithmic sampling preserves correlation decay
- ✅ **Automatic activation**: Only triggers for large datasets
- ✅ **Transparent operation**: No user intervention required
- ✅ **Backward compatible**: Existing configs work without changes

## Pipeline Architecture: CPU-Optimized (v2.3.0+)

The NLSQ analysis pipeline uses **JAX-accelerated CPU processing** for all operations, optimized for multi-core systems and HPC clusters.

### Computational Stages

**All stages run on CPU** with JAX JIT compilation:

1. **Configuration Loading**: YAML parsing with PyYAML (\<1s)
2. **Data Loading**: HDF5 file reading with h5py + NumPy (2-3s for cached data)
3. **Data Validation**: Statistical quality checks with NumPy + SciPy (\<1s)
4. **Angle Filtering**: Array slicing and indexing (\<1ms)
5. **NLSQ Optimization**: JIT-compiled JAX physics functions (30-60s on 14-core CPU)
   - Residual calculations called hundreds of times per iteration
   - Data volume: 3 angles × 1001 × 1001 = 3M+ points per iteration
   - Automatic parallelization across CPU cores
6. **Theoretical Fit Computation**: JAX backend with per-angle scaling (~2-3s on CPU)
7. **Result Saving**: JSON serialization + NPZ compression (~1s)
8. **Parallel Plotting**: 20 parallel workers using Datashader (CPU-optimized, ~12s)

### Performance Summary (14-Core CPU)

| Stage | Backend | Duration | Notes |
|-------|---------|----------|-------|
| Config Loading | PyYAML | <1s | I/O bound |
| Data Loading | h5py+NumPy | ~2s | I/O bound |
| Data Validation | NumPy+SciPy | <1s | Fast validation |
| Angle Filtering | NumPy | <1ms | Array operations |
| **NLSQ Optimization** | **JAX JIT** | **30-60s** | **Multi-core parallel** |
| **Theoretical Fits** | **JAX JIT** | **~2-3s** | **CPU-accelerated** |
| Result Saving | json+npz | ~1s | I/O bound |
| Plotting (Workers) | Datashader | ~12s | Parallel workers |

**Total Runtime**: ~50-80 seconds (14-core CPU)
**HPC Speedup**: ~2-3x faster on 36-core nodes

### Key Architectural Insights

- **JAX JIT Compilation**: All compute-intensive operations use JAX's JIT compiler for CPU optimization
- **Multi-Core Parallelization**: Automatic thread distribution across available CPU cores
- **Memory Efficiency**: Optimized memory usage for datasets up to 100M+ points
- **Parallel Visualization**: CPU workers efficiently handle batch plotting without memory constraints

## System Requirements (v2.3.0+)

### Minimum

- Python 3.12+
- NumPy >= 2.0, SciPy >= 1.14
- 4+ CPU cores
- 8+ GB RAM

### Recommended

- Python 3.12+
- 14+ CPU cores (desktop/workstation)
- 16+ GB RAM
- Linux, macOS, or Windows

### HPC Environment

- 36/128-core CPU nodes
- 64+ GB RAM for large datasets
- NUMA-aware configuration
- High-speed storage (NVMe/parallel filesystem)

## Dependencies (v2.3.0 - CPU-Only)

### Required Dependencies

- `jax==0.8.0` - JAX CPU framework (exact version required)
- `jaxlib==0.8.0` - JAX XLA library (must match JAX version)
- `numpy>=2.0.0,<3.0.0` - Core numerical operations (NumPy 2.x series)
- `scipy>=1.14.0,<2.0.0` - Scientific computing
- `nlsq>=0.1.0` - **NLSQ trust-region optimizer** for JAX-native nonlinear least squares
- `numpyro>=0.18.0,<0.20.0` - MCMC with built-in progress bars
- `h5py>=3.10.0,<4.0.0` - HDF5 file support
- `pyyaml>=6.0.2` - YAML configuration

### Optional Dependencies

- `blackjax>=1.2.0,<2.0.0` - Alternative MCMC backend
- `psutil>=6.0.0` - Enhanced CPU optimization
- `tqdm>=4.65.0` - Progress bars for NLSQ optimization
- `matplotlib>=3.8.0,<4.0.0` - Plotting capabilities

### Version Notes

- **JAX 0.8.0**: CPU-only, exact version match required for stability
- **NumPy 2.x**: Required for JAX 0.8.0 compatibility
- **Python 3.12+**: Minimum version for all dependencies

## Testing

Run the API compatibility test:

```bash
python test_api_compatibility.py
```

This validates:

- All essential imports work
- API function signatures are preserved
- Core functionality is accessible
- Configuration system works
- Device optimization works
- CLI interface is functional

## Migration from v1

> **Note:** v2.0 uses the NLSQ package for trust-region optimization. For details on the
> v2.0 release, see [CHANGELOG.md](CHANGELOG.md#200---2025-10-12).

The v2 rebuild maintains full API compatibility for validated components:

```python
# These imports work identically in v1 and v2
from homodyne.data import load_xpcs_data, XPCSDataLoader
from homodyne.core import compute_g2_scaled, TheoryEngine
from homodyne.config import ConfigManager

# New in v2: JAX-first optimization
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.device import configure_optimal_device
```

Existing YAML configuration files work without modification.

## HPC CPU Optimization (v2.3.0+)

### Multi-Core CPU Configuration

Homodyne v2.3.0+ is optimized for multi-core CPUs on HPC clusters:

- **36-core nodes**: Reserve 2 cores for OS, use 34 for computation
- **128-core nodes**: Reserve 4-8 cores for OS, use remaining for computation
- **NUMA awareness**: Automatic thread affinity for large HPC nodes
- **Memory efficiency**: Optimized for datasets up to 100M+ points

### HPC Slurm Job Example

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=04:00:00
#SBATCH --mem=64GB

# Load Python module
module load python/3.12

# Activate environment
source homodyne-env/bin/activate

# Configure CPU threads (reserve 2 for OS)
export OMP_NUM_THREADS=34
export JAX_NUM_THREADS=34

# Run analysis
homodyne --config laminar_flow.yaml --method nlsq
```

### CPU Performance Optimization

```python
# examples/cpu_optimization.py
import os
import psutil

# Reserve cores for system
cpu_count = psutil.cpu_count(logical=False)
optimal_threads = max(1, cpu_count - 2)

os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
os.environ['JAX_NUM_THREADS'] = str(optimal_threads)
```

See [`examples/cpu_optimization.py`](examples/cpu_optimization.py) for comprehensive HPC setup guide.

### Progress Tracking

- **NLSQ**: Optional progress bars with `tqdm` installation
- **MCMC**: Automatic progress bars via NumPyro (built-in)
- Enable verbose mode in config for detailed output

## Production Readiness

### Error Recovery Mechanisms (T022-T024)

The NLSQ optimizer includes **intelligent error recovery** for production reliability:

**Automatic Retry Strategy**:

- 3-attempt recovery with intelligent parameter perturbation
- Convergence failure → perturb parameters 10-20%
- Bounds violation → reset to bounds center
- Ill-conditioned Jacobian → scale parameters 0.1x
- Numerical instability → geometric mean reset

**Actionable Diagnostics**:

- Categorized error analysis (5 error types)
- User-actionable suggestions
- Comprehensive logging of recovery actions

**Usage**:

```python
# Enabled by default
result = fit_nlsq_jax(data, config)  # enable_recovery=True

# Disable for debugging
from homodyne.optimization.nlsq_wrapper import NLSQWrapper
wrapper = NLSQWrapper(enable_recovery=False)
```

### Scientific Validation (T036-T041)

The NLSQ implementation has been **scientifically validated** through comprehensive
testing:

**Validation Results**: ✅ **7/7 tests passed (100% success rate)**

- ✅ **T036**: Ground truth parameter recovery (easy/medium/hard cases)
- ✅ **T037**: Numerical stability (5 initial conditions)
- ✅ **T038**: Performance benchmarks (3 dataset sizes)
- ✅ **T039**: Error recovery validation
- ✅ **T040**: Physics validation (6 constraints)
- ✅ **T041**: Validation report generation

**Production Status**: ✅ **APPROVED for scientific research and production deployment**

**Documentation**: See `SCIENTIFIC_VALIDATION_REPORT.md` and
`PRODUCTION_READINESS_REPORT.md` for detailed analysis.

## Authors

- Wei Chen (weichen@anl.gov) - Argonne National Laboratory
- Hongrui He (hhe@anl.gov) - Argonne National Laboratory

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite both:

**Software:**

```bibtex
@software{chen2025homodyne,
  title={Homodyne v2: JAX-First XPCS Analysis Package},
  author={Chen, Wei and He, Hongrui},
  year={2025},
  organization={Argonne National Laboratory}
}
```

**Theory Paper:**

```bibtex
@article{he2024pnas,
  title={Theoretical framework for characterizing transport properties in flowing soft matter},
  author={He, Hongrui and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  year={2024},
  doi={10.1073/pnas.2401162121}
}
```
