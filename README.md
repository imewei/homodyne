# Homodyne v2.1: JAX-First XPCS Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.1.0-green.svg)](#)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://homodyne.readthedocs.io)
[![ReadTheDocs](https://readthedocs.org/projects/homodyne/badge/?version=latest)](https://homodyne.readthedocs.io/en/latest/)
[![GitHub Actions](https://github.com/imewei/homodyne/actions/workflows/docs.yml/badge.svg)](https://github.com/imewei/homodyne/actions/workflows/docs.yml)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

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
- **`device/`** - HPC/GPU optimization with system CUDA integration
- **`config/`** - Streamlined YAML-only configuration system with parameter management
- **`utils/`** - Minimal logging with preserved API signatures
- **`cli/`** - Essential command-line interface

## Key Features

### JAX-First Computational Engine

- **Primary**: NLSQ trust-region nonlinear least squares (JAX-native optimizer)
- **Secondary**: NumPyro/BlackJAX NUTS sampling for uncertainty quantification
- **Core Equation**: `c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²`

### System CUDA Integration

- Uses `jax[cuda12-local]` for system CUDA libraries
- Avoids version conflicts with existing HPC CUDA installations
- Graceful fallback to optimized CPU computation

### HPC Optimization

- CPU-primary design for 36/128-core HPC nodes
- Intelligent thread allocation and NUMA-aware configuration
- Memory-efficient processing for large datasets

### Intelligent Resource Allocation

- **GPU acceleration** for compute-intensive physics (NLSQ optimization, theoretical
  fits)
- **CPU processing** for I/O operations (data loading, result saving, plotting)
- **Exclusive GPU allocation** for main optimization process (90% memory)
- **CPU-only workers** for parallel plotting to prevent GPU memory exhaustion

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
| Multi-core CPU | 23 | 50M | CMC | ~40 min | 1.4x |
| Multi-core CPU | 23 | 50M | NUTS | ~60 min | baseline |
| Single GPU | 5 | 200M | CMC | ~2 hours | Avoids OOM |
| Single GPU | 5 | 200M | NUTS | OOM ❌ | N/A |

**Documentation:**
- Architecture Guide: [`docs/architecture/cmc-dual-mode-strategy.md`](docs/architecture/cmc-dual-mode-strategy.md)
- Quick Reference: [`docs/architecture/cmc-decision-quick-reference.md`](docs/architecture/cmc-decision-quick-reference.md)
- User Guide: [`docs/advanced-topics/cmc-large-datasets.rst`](docs/advanced-topics/cmc-large-datasets.rst)
- MCMC Guide: [`docs/advanced-topics/mcmc-uncertainty.rst`](docs/advanced-topics/mcmc-uncertainty.rst)

## Platform Support

### CPU-Only (All Platforms) ✅

- **Linux**: Full support
- **macOS**: Full support
- **Windows**: Full support
- **Python**: 3.12+

Default installation provides CPU-only JAX, which works on all platforms:

```bash
pip install homodyne
```

### GPU Acceleration (Linux Only) ⚡

- **Supported**: Linux x86_64 or aarch64 **only**
- **Not supported**: Windows, macOS
- **Requirements**:
  - NVIDIA GPU with CUDA Compute Capability 6.0+
  - CUDA 12.1-12.9 (pre-installed on system)
  - NVIDIA driver >= 525

## Installation

### Quick Install (CPU-Only, All Platforms)

```bash
# Works on Linux, macOS, Windows
pip install homodyne
```

This installs homodyne with CPU-only JAX (suitable for development and small datasets).

### GPU Installation (Linux Only) ⚡

**Performance Impact**: 20-100x speedup for large datasets (>1M points)

#### Option 1: Quick Install via Makefile (Recommended)

From the homodyne repository:

```bash
git clone https://github.com/your-org/homodyne.git
cd homodyne
make install-jax-gpu  # Automatically handles uninstall + GPU install
```

This single command:
- Uninstalls CPU-only JAX
- Installs GPU-enabled JAX with CUDA 12 support
- Verifies GPU detection

#### Option 2: Manual Installation

For GPU-accelerated computation on Linux systems with CUDA 12+:

```bash
# Step 1: Uninstall CPU-only version
pip uninstall -y jax jaxlib

# Step 2: Install JAX with CUDA support
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Step 3: Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Should show: [cuda(id=0)] instead of [CpuDevice(id=0)]
```

**Why separate installation?** JAX with CUDA support is Linux-specific and requires system CUDA 12.1-12.9 pre-installed. Separating the installation avoids dependency conflicts on macOS/Windows.

#### GPU Troubleshooting

**Issue**: Warning "An NVIDIA GPU may be present... but a CUDA-enabled jaxlib is not installed"

**Solution**:
```bash
# 1. Check GPU hardware
nvidia-smi  # Should show your GPU

# 2. Check CUDA version
nvcc --version  # Should show CUDA 12.1-12.9

# 3. Reinstall JAX with GPU support
pip uninstall -y jax jaxlib
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# 4. Verify JAX detects GPU
python -c "import jax; print(jax.devices())"
# Expected: [cuda(id=0)]
# If still showing [CpuDevice(id=0)], check CUDA installation
```

**Issue**: ImportError or CUDA library not found

**Solution**:
```bash
# Set CUDA library path (add to ~/.bashrc for permanent fix)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Or use homodyne's activation script
source homodyne/runtime/gpu/activation.sh
```

**Issue**: Out of memory errors during optimization

**Solution**: Reduce GPU memory fraction in config:
```yaml
performance:
  gpu_memory_fraction: 0.7  # Reduce from default 0.9
```

### Development Installation

```bash
# CPU-only (all platforms)
pip install homodyne[dev]

# Or from source
git clone https://github.com/your-org/homodyne.git
cd homodyne
pip install -e ".[dev]"

# For GPU support on Linux: run this first
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0
```

### HPC Environment Setup

```bash
# First, load system CUDA modules (site-specific)
module load cuda/12.2
module load cudnn/9.8

# Install JAX with CUDA support
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Install homodyne
pip install homodyne[dev]
```

### Makefile Shortcuts (Development)

```bash
make dev               # Install dev environment (CPU-only)
make install-jax-gpu   # Install JAX with GPU support (Linux only)
make gpu-check         # Verify GPU setup
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

# Force CPU optimization
cpu_config = configure_optimal_device(force_cpu=True)

# Configure GPU with specific memory fraction
gpu_config = configure_optimal_device(
    prefer_gpu=True,
    gpu_memory_fraction=0.8
)
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

## Pipeline Architecture: GPU vs CPU Usage

The NLSQ analysis pipeline uses **GPU acceleration for physics computations**
(optimization and theoretical fits) while **CPU for everything else** (data I/O,
validation, plotting). This optimal resource allocation prevents GPU memory conflicts
while maximizing performance.

### Computational Stages

**CPU-Only Stages** (I/O and validation):

- **Configuration Loading**: YAML parsing with PyYAML
- **Data Loading**: HDF5 file reading with h5py + NumPy (2-3s for cached data)
- **Data Validation**: Statistical quality checks with NumPy + SciPy (\<1s)
- **Angle Filtering**: Array slicing and indexing (\<1ms)

**GPU-Accelerated Stages** ⚡ (compute-intensive):

**1. NLSQ Optimization** (Primary GPU usage, ~6s):

- JIT-compiled JAX physics functions (`compute_g2_scaled_core`, `compute_g1_diffusion`,
  `compute_g1_shear`)
- Residual calculations called hundreds of times per iteration
- Data volume: 3 angles × 1001 × 1001 = 3M+ points per iteration
- Performance: **6s on GPU vs 30-60s on CPU** (5-10x speedup)
- Memory: **90% GPU allocation** (exclusive to main process)

**2. Theoretical Fit Computation** (~1s):

- Uses same JAX backend as optimization
- Sequential computation for all 23 angles (not just filtered subset)
- Per-angle scaling via least squares (find optimal contrast/offset)

**CPU-Only Stages** (Output and visualization):

**3. Result Saving & Plotting** (I/O bound, GPU provides no benefit):

- **Result Saving**: JSON serialization + NPZ compression (~1s)
- **Plotting Workers**: **Forced to CPU** to prevent GPU memory conflicts
  - 20 parallel workers using Datashader (CPU-optimized rasterization)
  - Environment variables set: `JAX_PLATFORMS="cpu"`, `CUDA_VISIBLE_DEVICES="-1"`
  - Reason: Main process uses 90% GPU memory; workers would cause
    `CUDA_ERROR_OUT_OF_MEMORY`
  - Performance: 23 plots in ~12s (5-10x faster than pure matplotlib)

### Performance Summary

| Stage | Device | Backend | Duration | GPU Benefit |
|-------|--------|---------|----------|-------------| | Config Loading | CPU | PyYAML |
\<1s | None (I/O bound) | | Data Loading | CPU | h5py+NumPy | ~2s | None (I/O bound) | |
Data Validation | CPU | NumPy+SciPy | \<1s | None (not critical) | | Angle Filtering |
CPU | NumPy | \<1ms | None (trivial) | | **NLSQ Optimization** | **GPU** ⚡ | **JAX JIT**
| **6s** | **5-10x speedup** | | **Theoretical Fits** | **GPU** ⚡ | **JAX JIT** |
**~1s** | **3-5x speedup** | | Result Saving | CPU | json+npz | ~1s | None (I/O bound) |
| Plotting (Workers) | CPU 🔒 | Datashader | ~12s | None (forced CPU) |

**Total GPU Time**: ~7 seconds (optimization + theoretical fits) **Total CPU Time**: ~17
seconds (all other stages) **Peak GPU Memory**: 90% allocation (main process only)
**Overall Speedup**: 5-10x for compute-intensive parts; ~24s total vs ~70-100s without
GPU

### Key Architectural Insights

- **GPU Acceleration Where It Matters**: Only the most compute-intensive stages (NLSQ
  optimization and theoretical fits) use GPU, achieving 5-10x speedup for ~7 seconds of
  GPU time. This focused allocation maximizes performance impact.

- **CPU for I/O and Visualization**: All data operations (loading, validation, saving)
  remain on CPU where they are I/O bound. Parallel plotting is forced to CPU-only
  workers to prevent GPU memory conflicts while leveraging Datashader's efficient CPU
  rasterization.

- **Resource Isolation Strategy**: Main process maintains exclusive GPU access (90%
  memory) for optimization. Worker processes are restricted to CPU via environment
  variables (`JAX_PLATFORMS="cpu"`, `CUDA_VISIBLE_DEVICES="-1"`), preventing CUDA OOM
  errors while enabling parallel visualization.

## System Requirements

### Minimum

- Python 3.10+
- NumPy >= 1.25, SciPy >= 1.11
- 4+ CPU cores
- Linux for GPU support

### Recommended

- Python 3.11+
- 16+ CPU cores or GPU
- 16+ GB RAM
- CUDA 12.1-12.9 pre-installed (for GPU)

### HPC Environment

- 36/128-core CPU nodes
- System CUDA 12.1-12.9 (for GPU acceleration)
- cuDNN 9.8+
- NVIDIA Driver >= 525 (or >= 570 for CUDA 12.8)
- 64+ GB RAM for large datasets

## Dependencies (2025 Latest Stable)

### Required Dependencies

- `numpy>=2.0.0` - Core numerical operations (NumPy 2.x series)
- `scipy>=1.14.0` - Scientific computing
- `jax>=0.7.2` - JAX acceleration framework
- `jaxlib>=0.7.2` - JAX XLA library (must match JAX version)
- `nlsq>=0.1.0` - **NLSQ trust-region optimizer** for JAX-native nonlinear least squares
- `numpyro>=0.19.0` - MCMC with built-in progress bars
- `h5py>=3.8.0` - HDF5 file support
- `pyyaml>=6.0` - YAML configuration

### Optional Dependencies

- `jax[cuda12-local]>=0.7.2` - System CUDA integration (Linux only)
- `blackjax>=1.2.5` - Alternative MCMC backend
- `psutil>=5.9.0` - Enhanced CPU optimization
- `tqdm>=4.65.0` - Progress bars for NLSQ optimization
- `matplotlib>=3.6.0` - Plotting capabilities

### GPU/System CUDA Notes

- **Platform**: Linux x86_64 or aarch64 only
- **CUDA**: Pre-installed 12.1-12.9 required
- **Why cuda12-local**: Uses system CUDA, avoids HPC conflicts
- **Alternative**: `jax[cuda12]` bundles CUDA (not recommended for HPC)

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

## HPC and System CUDA Setup

### Why System CUDA (`jax[cuda12-local]`)?

- **Avoids Conflicts**: Uses existing HPC CUDA installation
- **Version Compatibility**: Works with site-specific CUDA modules
- **Smaller Package**: Doesn't bundle CUDA libraries
- **Module System**: Compatible with HPC module environments

### HPC Quick Setup

```bash
# 1. Load system modules (site-specific commands)
module load cuda/12.2
module load cudnn/9.8

# 2. Verify CUDA installation
nvcc --version  # Should show CUDA 12.1+
nvidia-smi      # Should show driver >= 525

# 3. Set environment variables if needed
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH

# 4. Install homodyne with system CUDA
pip install --upgrade "jax[cuda12-local]==0.7.2" jaxlib==0.7.2 \
            nlsq>=0.1.0 numpyro==0.19.0 h5py pyyaml

# 5. Verify GPU detection
python -c "import jax; print(jax.devices())"
# Should show: [cuda:0], [cuda:1], etc.
```

### Troubleshooting GPU/CUDA Issues

**"CUDA not found" or "No GPU detected":**

```bash
# Check CUDA paths
which nvcc
echo $LD_LIBRARY_PATH

# Ensure paths are set
export PATH=/path/to/cuda/bin:$PATH
export LD_LIBRARY_PATH=/path/to/cuda/lib64:$LD_LIBRARY_PATH
```

**"CUDA version mismatch":**

- Requires CUDA 12.1-12.9 (CUDA 11.x no longer supported)
- Check with: `nvcc --version`
- Update driver if needed: minimum 525 (or 570 for CUDA 12.8)

**"Missing cuDNN":**

- Install cuDNN 9.8+ or load via module system
- Verify with: `ls $LD_LIBRARY_PATH | grep cudnn`

**Platform limitations:**

- GPU support is Linux-only (x86_64 or aarch64)
- Windows users: Use WSL2 with Linux installation
- macOS: CPU-only (no CUDA support)

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
