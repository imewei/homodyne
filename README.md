# Homodyne.4: CPU-Optimized JAX-First XPCS Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-2.4.1-green.svg)](#)
[![Documentation](https://img.shields.io/badge/docs-sphinx-blue.svg)](https://homodyne.readthedocs.io)
[![ReadTheDocs](https://readthedocs.org/projects/homodyne/badge/?version=latest)](https://homodyne.readthedocs.io/en/latest/)
[![GitHub Actions](https://github.com/imewei/homodyne/actions/workflows/docs.yml/badge.svg)](https://github.com/imewei/homodyne/actions/workflows/docs.yml)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

## ⚠️ **BREAKING CHANGES: v2.4.x**

### v2.4.1 - CMC-Only MCMC Architecture

**MCMC now always uses Consensus Monte Carlo (CMC)** - NUTS auto-selection removed.

**Key Changes:**

- **CMC mandatory**: All MCMC runs use CMC; single-shard runs still use NUTS internally
- **Removed CLI flags**: `--min-samples-cmc`, `--memory-threshold-pct` (deprecated)
- **Per-phi initialization**: Initial values derived from config or per-phi percentiles
- **Migration guide**: See [CMC-Only Migration](docs/migration/v3_cmc_only.md)

### v2.4.0 - Per-Angle Scaling Mandatory

**Per-angle scaling is now mandatory** - Legacy scalar `per_angle_scaling=False`
removed.

**Key Changes:**

- **Breaking**: Remove `per_angle_scaling` parameter or set to `True`
- **Impact**: 3 angles: 5 params → 9 params `[c₀,c₁,c₂, o₀,o₁,o₂, D0,α,D_offset]`
- **Rationale**: Per-angle mode is physically correct for heterogeneous samples

### v2.3.0 - GPU Support Removed

**v2.3.0 transitions to CPU-only architecture** - All GPU acceleration support has been
removed.

**Key Changes:**

- **Rationale**: Simplify maintenance, focus on reliable HPC CPU optimization
- **Impact**: Removed 9 GPU API functions, GPU-specific CLI flags, GPU examples
- **For GPU users**: Stay on **v2.2.1** (last GPU-supporting version, available on PyPI)
- **Migration guide**: See
  [v2.2-to-v2.3 GPU Removal Guide](docs/migration/v2.2-to-v2.3-gpu-removal.md)

## 🎉 v2.2.1 Critical Fix

**Parameter Expansion for Per-Angle Scaling** - Resolves silent NLSQ optimization
failures

Key fixes:

- **Automatic parameter expansion**: 9 → 13 parameters for 3 angles (7 physical + 3×2
  scaling)
- **Gradient sanity check**: Pre-optimization validation detects zero-gradient issues
- **Stratified least-squares**: Direct NLSQ integration with StratifiedResidualFunction
- **Performance**: 93.15% cost reduction, 113 function evaluations (vs 0 before fix)

## 🎉 v2.2.0 Feature Release

**Angle-Stratified Chunking** - Automatic fix for per-angle scaling on large datasets

Key improvements:

- **Automatic activation**: No configuration required - activates when
  `per_angle_scaling=True` AND `n_points>=100k`
- **Zero regressions**: 100% backward compatible with existing configurations
- **Performance**: \<1% overhead (0.15s for 3M points), sub-linear O(n^1.01) scaling
- **Reliability**: Fixes silent optimization failures caused by arbitrary chunking
- **Fallbacks**: Sequential per-angle optimization for extreme angle imbalance (>5.0
  ratio)

See [v2.2 Release Notes](docs/releases/v2.2-stratification-release-notes.md) for
complete details.

**High-performance JAX-first package for X-ray Photon Correlation Spectroscopy (XPCS)
analysis**, implementing the theoretical framework from
[He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing
transport properties in flowing soft matter systems through time-dependent intensity
correlation functions.

📚 **[Read the Full Documentation](https://homodyne.readthedocs.io)** |
**[Quick Start Guide](https://homodyne.readthedocs.io/en/latest/user-guide/quickstart.html)**
|
**[API Reference](https://homodyne.readthedocs.io/en/latest/api-reference/index.html)**

A completely rebuilt homodyne package with JAX-first architecture, optimized for HPC and
supercomputer environments.

## Homodyne Correlation Model

```
c2(φ, t1, t2) = offset + contrast * [c1(φ, t1, t2)]^2
c1(φ, t1, t2) = c1_diff(t1, t2) * c1_shear(φ, t1, t2)

c1_diff(t1, t2) = exp[-(q^2 / 2) * ∫|t2 - t1| D(t') dt']
c1_shear(φ, t1, t2) = [sinc(Φ(φ, t1, t2))]^2
Φ(φ, t1, t2) = (1 / 2π) * q * L * cos(φ0 - φ) * ∫|t2 - t1| γ̇(t') dt'

D(t) = D0 * t^α + D_offset
γ̇(t) = γ̇0 * t^β + γ̇_offset
```

Parameter sets:
- Static (3): D0, α, D_offset (shear terms ignored, φ0 = 0)
- Laminar flow (7): D0, α, D_offset, γ̇0, β, γ̇_offset, φ0

Experimental parameters: q (Å⁻¹), L (Å), φ (deg), dt (s/frame); contrast/offset are per-angle.

## Architecture Overview

This rebuild achieves **~70% complexity reduction** while maintaining **100% API
compatibility** for all validated components:

### Preserved Components

- **`data/`** - Complete XPCS data loading infrastructure (11 files preserved)
- **`core/`** - Validated physics models and JAX backend (11 files including shared utilities)

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

**v2.4.1+**: CMC-only architecture - all MCMC runs use Consensus Monte Carlo.

**CMC-Only Architecture** (v2.4.1 - December 2025):

- **CMC mandatory**: All MCMC runs use CMC with automatic sharding
- **Single-shard NUTS**: Small datasets run as single-shard CMC (NUTS internally)
- **Per-phi initialization**: Initial values from config or per-phi percentiles
- **Simplified CLI**: Only `--method cmc` needed (CMC always used)

**Key Features:**

- **Z-space gradient balancing**: Non-centered parameterization for stable MCMC sampling
- **Data-driven initialization**: Automatic contrast/offset estimation from C2 data
- **Hardware-adaptive**: Automatic backend selection (pjit/multiprocessing/PBS/Slurm)
- **Linear speedup**: Perfect parallelization across CPU cores or cluster nodes
- **Memory efficient**: Each shard fits in available memory with 40% safety margin
- **Production-ready**: Comprehensive validation, fault tolerance, and convergence
  diagnostics

**Quick Example:**

```python
from homodyne.optimization.mcmc import fit_mcmc_jax
from homodyne.config.parameter_space import ParameterSpace

# All MCMC runs use CMC (v2.4.1+)
parameter_space = ParameterSpace.from_config(config_dict)
result = fit_mcmc_jax(
    data=data['c2'],
    t1=data['t1'], t2=data['t2'], phi=data['phi'],
    q=0.0054, L=2000000,
    analysis_mode='static',
    parameter_space=parameter_space,
    initial_values={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
)

# CMC is always used
print(f"✓ CMC used with {result.num_shards} shards")
```

**Performance:**

| Scenario | Shards | Data Size | Runtime | Speedup |
|----------|--------|-----------|---------|---------| | Multi-core CPU (14 cores) | 4 |
50M | ~40 min | 1.4x | | HPC CPU (36 cores) | 8 | 200M | ~2 hours | 1.5x | |
Single-shard (small data) | 1 | 5M | ~10 min | baseline |

**Documentation:**

- Architecture Guide:
  [`docs/architecture/cmc-dual-mode-strategy.md`](docs/architecture/cmc-dual-mode-strategy.md)
- Quick Reference:
  [`docs/architecture/cmc-decision-quick-reference.md`](docs/architecture/cmc-decision-quick-reference.md)
- User Guide:
  [`docs/advanced-topics/cmc-large-datasets.rst`](docs/advanced-topics/cmc-large-datasets.rst)
- MCMC Guide:
  [`docs/advanced-topics/mcmc-uncertainty.rst`](docs/advanced-topics/mcmc-uncertainty.rst)

### Result Artifacts & Diagnostics (v2.3.1)

- `fitted_data.npz` now stores both the **solver-evaluated** surface
  (`c2_solver_scaled`) and the legacy **post-hoc** surface, plus the original per-angle
  contrast/offset pairs (`per_angle_scaling_solver`). Existing consumers that rely on
  `c2_theoretical_scaled` continue to work unchanged.
- Plotting defaults to the solver surface and supports adaptive color scaling via
  `output.plots.color_scale` (`mode: legacy|adaptive`, optional percentiles/fixed
  ranges). Set `output.plots.fit_surface` to `"posthoc"` to retain the previous behavior
  or pin `[1.0, 1.5]` via `pin_legacy_range: true`.
- Use `scripts/nlsq/overlay_solver_vs_posthoc.py` (or the helper in
  `homodyne.viz.diagnostics`) to print baseline oscillation stats and overlay
  solver/post-hoc diagonals for any saved `fitted_data.npz`.

## Platform Support

### CPU-Only Architecture (All Platforms) ✅

**v2.3.0+**: GPU support removed - CPU-optimized for all platforms

- **Linux**: Full support (HPC clusters recommended)
- **macOS**: Full support
- **Windows**: Full support
- **Python**: 3.12+

**For GPU users**: Use Homodyne.2.1 (last GPU-supporting version)

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

# Install CPU-only Homodyne.3.0+
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

1. **See migration guide**:
   [`docs/migration/v2.2-to-v2.3-gpu-removal.md`](docs/migration/v2.2-to-v2.3-gpu-removal.md)

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

# CMC sampling for uncertainty quantification (v2.4.1+: CMC-only architecture)
homodyne --method cmc --config config.yaml

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

# 4. Run CMC with initialized parameters
homodyne --method cmc --config config.yaml
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
[NLSQ Optimization Example Notebook](scripts/notebooks/nlsq_optimization_example.ipynb):

```bash
# Launch Jupyter and open the example notebook
jupyter notebook scripts/notebooks/nlsq_optimization_example.ipynb
```

The notebook covers:

- Synthetic XPCS data generation with ground truth parameters
- NLSQ optimization workflow with parameter recovery
- Fit quality visualization and residual analysis
- Parameter uncertainty quantification
- Error recovery demonstration

## Shell Completion & CLI Tools

Homodyne provides four CLI commands and intelligent shell completion for faster
workflows.

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

**Conda/Mamba users:** Completion auto-activates with your environment - no extra setup
needed!

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
hm-cmc --config config.yaml            # homodyne --method cmc

hc-stat --output static.yaml           # homodyne-config --mode static
hc-flow --output flow.yaml             # homodyne-config --mode laminar_flow

hconfig --validate my_config.yaml      # homodyne-config --validate
```

**Smart completion examples:**

```bash
homodyne --config <TAB>        # Shows *.yaml files
homodyne --method <TAB>        # Shows: nlsq, cmc
hm-nlsq --<TAB>                # Shows all available options
```

**See documentation:**
[Shell Completion Guide](https://homodyne.readthedocs.io/en/latest/user-guide/shell-completion.html)

## XLA Configuration

Homodyne includes automatic XLA_FLAGS configuration that optimizes JAX CPU device
allocation for MCMC and NLSQ workflows.

### Quick Setup

```bash
# Interactive setup (recommended)
homodyne-post-install --interactive

# Or quick one-liner
homodyne-post-install --xla-mode auto  # Auto-detect optimal device count
homodyne-post-install --xla-mode mcmc  # Configure for MCMC (4 devices)
```

### Configuration Modes

| Mode | Devices | Best For | Hardware | |------|---------|----------|----------| |
**mcmc** | 4 | Multi-core workstations, parallel MCMC chains | 8-15 CPU cores | |
**mcmc-hpc** | 8 | HPC clusters with many CPU cores | 36+ CPU cores | | **nlsq** | 1 |
NLSQ-only workflows, memory-constrained systems | Any CPU | | **auto** | 2-8 | Automatic
detection based on CPU core count | Auto-adaptive |

**Auto mode detection logic:**

```text
CPU Cores    →    Devices
≤ 7 cores    →    2 devices  (small workstations)
8-15 cores   →    4 devices  (medium workstations)
16-35 cores  →    6 devices  (large workstations)
36+ cores    →    8 devices  (HPC nodes)
```

### Managing XLA Configuration

```bash
# Set XLA mode
homodyne-config-xla --mode auto

# Show current configuration
homodyne-config-xla --show

# Example output:
#   Current XLA Configuration:
#     Mode: auto
#     XLA_FLAGS: --xla_force_host_platform_device_count=6
#     JAX devices: 6 (cpu)
```

### How It Works

1. **Configuration Storage**: Your selected mode is saved to `~/.homodyne_xla_mode`
1. **Automatic Activation**: XLA_FLAGS is set automatically when you activate your
   virtual environment
1. **JAX Detection**: JAX automatically creates the configured number of CPU devices

**Conda/Mamba** (automatic):

```bash
conda activate myenv  # XLA_FLAGS auto-configured
echo $XLA_FLAGS
# Output: --xla_force_host_platform_device_count=6
```

**uv/venv/virtualenv** (requires shell RC update):

```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.bashrc
source ~/.bashrc
```

### Performance Impact

| Workflow | Device Count | Hardware | Performance |
|----------|--------------|----------|-------------| | MCMC (4 chains) | 4 devices |
14-core CPU | 1.4x speedup | | MCMC (8 chains) | 8 devices | 36-core HPC | 1.8x speedup
| | NLSQ optimization | 1 device | Any CPU | Optimal (no overhead) | | Auto mode | 2-8
devices | Adapts to CPU | Automatic optimization |

### Best Practices

**For MCMC workflows:**

```bash
homodyne-config-xla --mode mcmc      # Typical workstations
homodyne-config-xla --mode mcmc-hpc  # HPC clusters (36+ cores)
homodyne --method cmc --config config.yaml
```

**For NLSQ workflows:**

```bash
homodyne-config-xla --mode nlsq  # Optimal single-device performance
homodyne --method nlsq --config config.yaml
```

**For mixed workflows:**

```bash
homodyne-config-xla --mode auto  # Adapts to hardware automatically
```

### Advanced Features

**Manual override** (temporary):

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=2"
source venv/bin/activate  # Respects your manual setting
```

**Per-environment configuration**:

```bash
export HOMODYNE_XLA_MODE=nlsq  # Takes precedence over ~/.homodyne_xla_mode
```

**Verbose mode**:

```bash
export HOMODYNE_VERBOSE=1
source venv/bin/activate
# Output: [homodyne] XLA: auto mode → 6 devices (detected 20 CPU cores)
```

**See documentation:**
[XLA Configuration Guide](https://homodyne.readthedocs.io/en/latest/user-guide/shell-completion.html#xla-configuration-system)

## Analysis Modes

### Static Isotropic (3 parameters)

- Parameters: `[D₀, α, D_offset]`
- Fast analysis for isotropic systems
- Ideal for quick parameter estimation

### Laminar Flow (7 parameters)

- Parameters: `[D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]`
- Full anisotropic analysis with shear flow
- Comprehensive characterization of complex systems

## Parameter Constraints

### Physical Parameters

Default bounds for NLSQ optimization and MCMC priors (updated Nov 15, 2025):

| Parameter | Min | Max | Units | Physical Meaning | Notes |
|-----------|-----|-----|-------|------------------|-------| | **D0** | 1×10² | 1×10⁵ |
Å²/s | Diffusion coefficient prefactor | Typical colloidal range | | **alpha** | -2.0 |
2.0 | - | Diffusion time exponent | Anomalous diffusion | | **D_offset** | -1×10⁵ |
1×10⁵ | Å²/s | Diffusion baseline correction | **Negative for jammed systems** | |
**gamma_dot_t0** | 1×10⁻⁶ | 0.5 | s⁻¹ | Initial shear rate | Laminar flow only | |
**beta** | -2.0 | 2.0 | - | Shear rate time exponent | Laminar flow only | |
**gamma_dot_t_offset** | -0.1 | 0.1 | s⁻¹ | Shear rate baseline correction | Laminar
flow only | | **phi0** | -10 | 10 | degrees | Initial flow angle | **Uses degrees, not
radians** |

### Scaling Parameters

| Parameter | Min | Max | Physical Meaning | Notes |
|-----------|-----|-----|------------------|-------| | **contrast** | 0.0 | 1.0 |
Visibility parameter | Homodyne detection efficiency | | **offset** | 0.5 | 1.5 |
Baseline level | ±50% from theoretical g2=1.0 |

### Correlation Function Constraints

Physics-enforced constraints applied during optimization:

| Function | Min | Max | Notes | |----------|-----|-----|-------| | **g1 (c1)** | 0.0 |
1.0 | Normalized correlation function<br>Log-space clipping: `log(g1) ∈ [-700, 0]` | |
**g2 (c2)** | 0.5 | 2.5 | Experimental range with headroom<br>Theoretical: g2 = 1 +
contrast × g1² |

**Important Notes:**

- **D_offset** can be negative for arrested/jammed systems (caging, jamming transitions)
- **phi0** uses degrees throughout the codebase (templates, physics modules)
- **gamma_dot_t_offset** allows negative values (baseline correction)
- All bounds align with template files: `homodyne_static.yaml`,
  `homodyne_laminar_flow.yaml`
- User configs override these default bounds (no breaking changes)

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

Homodyne includes **intelligent two-layer subsampling** to handle very large datasets
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

The NLSQ analysis pipeline uses **JAX-accelerated CPU processing** for all operations,
optimized for multi-core systems and HPC clusters.

### Computational Stages

**All stages run on CPU** with JAX JIT compilation:

1. **Configuration Loading**: YAML parsing with PyYAML (\<1s)
1. **Data Loading**: HDF5 file reading with h5py + NumPy (2-3s for cached data)
1. **Data Validation**: Statistical quality checks with NumPy + SciPy (\<1s)
1. **Angle Filtering**: Array slicing and indexing (\<1ms)
1. **NLSQ Optimization**: JIT-compiled JAX physics functions (30-60s on 14-core CPU)
   - Residual calculations called hundreds of times per iteration
   - Data volume: 3 angles × 1001 × 1001 = 3M+ points per iteration
   - Automatic parallelization across CPU cores
1. **Theoretical Fit Computation**: JAX backend with per-angle scaling (~2-3s on CPU)
1. **Result Saving**: JSON serialization + NPZ compression (~1s)
1. **Parallel Plotting**: 20 parallel workers using Datashader (CPU-optimized, ~12s)

### Performance Summary (14-Core CPU)

| Stage | Backend | Duration | Notes | |-------|---------|----------|-------| | Config
Loading | PyYAML | \<1s | I/O bound | | Data Loading | h5py+NumPy | ~2s | I/O bound | |
Data Validation | NumPy+SciPy | \<1s | Fast validation | | Angle Filtering | NumPy |
\<1ms | Array operations | | **NLSQ Optimization** | **JAX JIT** | **30-60s** |
**Multi-core parallel** | | **Theoretical Fits** | **JAX JIT** | **~2-3s** |
**CPU-accelerated** | | Result Saving | json+npz | ~1s | I/O bound | | Plotting
(Workers) | Datashader | ~12s | Parallel workers |

**Total Runtime**: ~50-80 seconds (14-core CPU) **HPC Speedup**: ~2-3x faster on 36-core
nodes

### Key Architectural Insights

- **JAX JIT Compilation**: All compute-intensive operations use JAX's JIT compiler for
  CPU optimization
- **Multi-Core Parallelization**: Automatic thread distribution across available CPU
  cores
- **Memory Efficiency**: Optimized memory usage for datasets up to 100M+ points
- **Parallel Visualization**: CPU workers efficiently handle batch plotting without
  memory constraints

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

Homodyne.3.0+ is optimized for multi-core CPUs on HPC clusters:

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
# scripts/nlsq/cpu_optimization.py
import os
import psutil

# Reserve cores for system
cpu_count = psutil.cpu_count(logical=False)
optimal_threads = max(1, cpu_count - 2)

os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
os.environ['JAX_NUM_THREADS'] = str(optimal_threads)
```

See [`scripts/nlsq/cpu_optimization.py`](scripts/nlsq/cpu_optimization.py) for
comprehensive HPC setup guide.

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
  title={Homodyne: JAX-First XPCS Analysis Package},
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
