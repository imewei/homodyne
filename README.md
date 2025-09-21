# Homodyne Scattering Analysis Package

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![PyPI version](https://badge.fury.io/py/homodyne-analysis.svg)](https://badge.fury.io/py/homodyne-analysis)
[![PyPI Downloads](https://img.shields.io/pypi/dm/homodyne-analysis)](https://pypi.org/project/homodyne-analysis/)
[![Documentation Status](https://readthedocs.org/projects/homodyne/badge/?version=latest)](https://homodyne.readthedocs.io/en/latest/?badge=latest)
[![CI Status](https://github.com/imewei/homodyne/workflows/CI/badge.svg)](https://github.com/imewei/homodyne/actions)
[![codecov](https://codecov.io/gh/imewei/homodyne/branch/main/graph/badge.svg)](https://codecov.io/gh/imewei/homodyne)
[![DOI](https://zenodo.org/badge/DOI/10.1073/pnas.2401162121.svg)](https://doi.org/10.1073/pnas.2401162121)

**High-performance Python package for X-ray Photon Correlation Spectroscopy (XPCS) analysis under nonequilibrium conditions.**

Implements the theoretical framework from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121) for characterizing transport properties in flowing soft matter systems through time-dependent intensity correlation functions.

---

## ✨ Key Features

### 🔬 **Advanced Analysis Capabilities**
- **Three analysis modes**: Static Isotropic (3 params), Static Anisotropic (3 params), Laminar Flow (7 params)
- **Multiple optimization methods**: Classical (Nelder-Mead, Gurobi), Robust (Wasserstein DRO, Scenario-based, Ellipsoidal), Bayesian MCMC (NUTS)
- **Noise-resistant analysis**: Robust optimization for measurement uncertainty and outlier resistance
- **Scientific accuracy**: Automatic $g_2 = \\text{offset} + \\text{contrast} \\times g_1$ fitting

### ⚡ **High Performance**
- **Numba JIT compilation**: 3-5x speedup with comprehensive warmup
- **Isolated MCMC backends**: Completely separated PyMC CPU and NumPyro GPU implementations prevent conflicts
- **Smart GPU acceleration**: Automatic backend selection with CPU fallback
- **Vectorized operations**: Optimized memory usage and batch processing
- **Performance monitoring**: Built-in benchmarking and optimization tools

### 🛠️ **Unified System Integration**
- **One-command setup**: `homodyne-post-install --shell zsh --gpu --advanced`
- **Cross-platform shell completion**: Smart caching and environment integration
- **Advanced tools**: GPU optimization, system validation, performance monitoring
- **Comprehensive testing**: 65+ test files with 500+ tests and automated CI/CD

### 📊 **Enhanced User Experience**
- **Interactive configuration**: `homodyne-config` with guided setup
- **Flexible logging**: Verbose, quiet, and file-only logging modes
- **Rich visualizations**: Correlation heatmaps, 3D surfaces, diagnostic plots
- **Extensive documentation**: Complete user guides, API reference, and tutorials

---

## 🚀 Quick Start

### Installation & Setup

```bash
# Complete installation with all features
pip install homodyne-analysis[all]

# Unified system setup
homodyne-post-install --shell zsh --gpu --advanced

# Validate installation
homodyne-validate
```

### Basic Usage

```bash
# Create configuration
homodyne-config --mode laminar_flow --sample my_sample

# Run analysis with smart aliases (Linux only - see Setup section)
hm --config config.json          # MCMC with smart GPU detection
ha --config config.json          # All methods with intelligent selection
hc --config config.json          # Classical optimization
hr --config config.json          # Robust optimization

# System tools (Linux/GPU)
gpu-status               # Check GPU status
homodyne-validate        # System validation
```

### Python API

```python
from homodyne import HomodyneAnalysisCore, ConfigManager

config = ConfigManager("config.json")
analysis = HomodyneAnalysisCore(config)

# Run different optimization methods
results = analysis.optimize_classical()  # Fast classical methods
results = analysis.optimize_robust()     # Noise-resistant methods
results = analysis.optimize_all()        # Comprehensive analysis
```

---

## 📦 Installation Options

### Core Installation

```bash
# Standard installation with essential features
pip install homodyne-analysis[performance,mcmc,robust]

# Development installation
pip install homodyne-analysis[all]

# GPU acceleration (Linux + NVIDIA)
pip install homodyne-analysis[jax]
```

### Optional Dependencies

| Feature | Command | Includes |
|---------|---------|----------|
| **Performance** | `[performance]` | Numba JIT, JAX acceleration, profiling tools |
| **MCMC CPU Backend** | `[mcmc]` | Pure PyMC CPU backend (isolated from JAX) |
| **MCMC GPU Backend** | `[mcmc-gpu]` | Pure NumPyro/JAX GPU backend (isolated from PyMC) |
| **MCMC All Backends** | `[mcmc-all]` | Both isolated CPU and GPU backends |
| **Robust Optimization** | `[robust]` | CVXPY for noise-resistant methods |
| **Commercial Solvers** | `[gurobi]` | Gurobi optimization (requires license) |
| **Development** | `[dev]` | Testing, docs, quality tools, pre-commit |
| **All Features** | `[all]` | Complete installation |

### GPU Acceleration Setup

```bash
# Install with GPU support
pip install homodyne-analysis[jax]

# Smart GPU setup (Linux + NVIDIA)
homodyne-post-install --shell zsh --gpu --advanced

# Test GPU configuration
homodyne-validate --test gpu
gpu-bench                    # Performance benchmark
homodyne-gpu-optimize        # Hardware optimization
```

---

## 🔬 Analysis Modes

| Mode | Parameters | Use Case | Speed | Command |
|------|------------|----------|-------|---------|
| **Static Isotropic** | 3 | Isotropic systems, fastest analysis | ⭐⭐⭐ | `--static-isotropic` |
| **Static Anisotropic** | 3 | Static with angular dependencies | ⭐⭐ | `--static-anisotropic` |
| **Laminar Flow** | 7 | Flow & shear analysis, full physics | ⭐ | `--laminar-flow` |

### Physical Models

**Static Modes**: $g_1(t_1,t_2) = \\exp(-q^2 \\int\_{t_1}^{t_2} D(t)dt)$
- Parameters: $D_0$ (diffusion), $\\alpha$ (time exponent), $D\_{\\text{offset}}$ (baseline)

**Laminar Flow Mode**: $g_1(t_1,t_2) = g\_{1,\\text{diff}}(t_1,t_2) \\times g\_{1,\\text{shear}}(t_1,t_2)$
- Additional: $\\dot{\\gamma}\_0$, $\\beta$, $\\dot{\\gamma}\_{\\text{offset}}$ (shear), $\\phi_0$ (angular offset)

### Standardized Physical Parameter Bounds

The package enforces standardized physical bounds for all parameters across all optimization methods:

- **D₀**: [1.0, 1e6] Å²/s - Characteristic diffusion coefficient
- **α**: [-10, 10] dimensionless - Time-dependent diffusion exponent
- **D_offset**: [-1e5, 1e5] Å²/s - Diffusion baseline offset
- **γ̇₀**: [1e-5, 1.0] s⁻¹ - Characteristic shear rate
- **β**: [-10, 10] dimensionless - Time-dependent shear exponent
- **γ̇_offset**: [-1.0, 1.0] s⁻¹ - Shear rate baseline offset
- **φ₀**: [-30°, 30°] degrees - Angular offset between shear direction and scattering vector
- **contrast**: [0.01, 1.0] dimensionless - Correlation amplitude
- **offset**: [0.0, 2.0] dimensionless - Correlation baseline

These bounds are physically motivated and ensure stable optimization across all methods (Classical, Robust, MCMC). Bounds can be overridden via configuration files for specialized applications.

---

## 🛠️ Configuration & Usage

### Configuration Generation

```bash
# Interactive configuration
homodyne-config --mode static_isotropic --sample protein_01

# Template with metadata
homodyne-config --mode laminar_flow --sample microgel \
  --author "Your Name" --experiment "Shear dynamics"
```

### Analysis Commands

```bash
# Basic analysis
homodyne --config config.json --method classical
homodyne --config config.json --method robust      # Noise-resistant
homodyne --config config.json --method mcmc        # Bayesian analysis
homodyne --config config.json --method all         # Comprehensive

# Data visualization
homodyne --plot-experimental-data --config config.json
homodyne --plot-simulated-data --config config.json

# Logging control
homodyne --config config.json --verbose    # Debug information
homodyne --config config.json --quiet      # File logging only
```

### Isolated MCMC Backend Commands

```bash
# CPU Backend (Pure PyMC - isolated from JAX)
homodyne --config config.json --method mcmc

# GPU Backend (Pure NumPyro/JAX - isolated from PyMC)
homodyne-gpu --config config.json --method mcmc

# Smart aliases (automatic backend selection - Linux only)
hm --config config.json          # Smart MCMC with backend isolation
ha --config config.json          # All methods with optimization
hc --config config.json          # Classical methods only
hr --config config.json          # Robust methods only
```

---

## 📊 Results & Output

### Directory Structure

```
./homodyne_results/
├── homodyne_analysis_results.json     # Main results summary
├── run.log                            # Execution log
├── classical/                         # Classical methods
│   ├── nelder_mead/
│   │   ├── parameters.json
│   │   ├── fitted_data.npz
│   │   └── c2_heatmaps_*.png
│   └── gurobi/                        # If available
├── robust/                            # Robust methods
│   ├── wasserstein/
│   ├── scenario/
│   └── ellipsoidal/
├── mcmc/                              # MCMC results
│   ├── mcmc_summary.json
│   ├── mcmc_trace.nc
│   ├── fitted_data.npz
│   ├── c2_heatmaps_*.png
│   ├── mcmc_corner_plot.png
│   ├── mcmc_trace_plots.png
│   └── mcmc_convergence_diagnostics.png
└── diagnostic_summary.png             # Cross-method comparison
```

### Data Files

Each method directory contains:
- **`parameters.json`**: Human-readable results with uncertainties
- **`fitted_data.npz`**: Complete numerical data (experimental, fitted, residuals)
- **Visualization plots**: Correlation heatmaps, 3D surfaces, diagnostics

---

## ⚡ Performance & Optimization

### Optimization Methods

| Method | Algorithm | Speed | Use Case |
|--------|-----------|-------|----------|
| **Classical** | Nelder-Mead, Gurobi | Minutes | Exploratory analysis |
| **Robust** | Wasserstein DRO, Scenario, Ellipsoidal | 2-5x slower | Noisy data |
| **MCMC CPU** | NUTS (Pure PyMC) | Hours | CPU-only uncertainty quantification |
| **MCMC GPU** | NUTS (Pure NumPyro/JAX) | 2-10x faster | GPU-accelerated uncertainty quantification |

### Performance Features

- **JIT Compilation**: 3-5x speedup with Numba
- **Isolated Backend Architecture**: Prevents PyTensor/JAX conflicts, enables optimal performance
- **GPU Acceleration**: Pure NumPyro/JAX backend for MCMC with automatic CPU fallback
- **Vectorized Operations**: Optimized batch processing
- **Memory Management**: Efficient caching and cleanup
- **Parallel Testing**: Multi-core test execution

### Environment Optimization

```bash
# Threading optimization
export OMP_NUM_THREADS=8
export NUMBA_DISABLE_INTEL_SVML=1

# GPU optimization (Linux)
export JAX_ENABLE_X64=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

---

## 🧪 Testing & Quality

### Testing Framework

```bash
# Quick development tests
pytest -c pytest-quick.ini

# Full test suite with coverage
pytest -c pytest-full.ini

# Specific test categories
pytest -m "fast"                # Quick tests
pytest -m "not slow"            # Skip slow tests
pytest homodyne/tests/unit       # Unit tests only
```

### Test Categories

- **Unit Tests** (20 files): Core component testing
- **Integration Tests** (8 files): Component interactions
- **System Tests** (16 files): CLI and GPU functionality
- **MCMC Tests**: Isolated CPU/GPU backend validation
- **Performance Tests**: Regression detection
- **Backend Isolation Tests**: Verify complete separation of PyMC and NumPyro

### Code Quality

- ✅ **Black**: 100% formatted (88-char line length)
- ✅ **isort**: Import organization
- ✅ **Ruff**: Fast linting with auto-fixes
- ✅ **Bandit**: 0 security issues
- ✅ **Pre-commit hooks**: Automated quality checks

---

## 🔧 Advanced Features

### System Validation

```bash
# Complete system check
homodyne-validate

# Component-specific testing
homodyne-validate --test gpu
homodyne-validate --test completion
homodyne-validate --json            # CI/CD output
```

### GPU Optimization

```bash
# Hardware analysis & optimization
homodyne-gpu-optimize --report
homodyne-gpu-optimize --benchmark --apply

# Performance monitoring
gpu-bench                           # Quick benchmark
nvidia-smi dmon -i 0 -s pucvmet -d 1  # Real-time monitoring
```

### Shell Integration

```bash
# One-command setup
homodyne-post-install --shell zsh --gpu --advanced

# Interactive setup
homodyne-post-install --interactive

# Smart completion
homodyne --config <TAB>             # File completion
homodyne --method <TAB>             # Method completion

# Smart aliases (created on Linux, manual setup required on macOS/Windows)
# hm, ha, hc, hr - see post-install output for details
```

---

## 🧹 Uninstallation

```bash
# Step 1: Clean up environment scripts (important!)
homodyne-cleanup --interactive

# Step 2: Remove package
pip uninstall homodyne-analysis

# Step 3: Verify cleanup
homodyne-validate 2>/dev/null || echo "✅ Successfully uninstalled"
```

**⚠️ Always run cleanup first** - the `homodyne-cleanup` command removes environment scripts that `pip uninstall` cannot track.

---

## 📚 Documentation

### Quick Access

| Topic | Link | Description |
|-------|------|-------------|
| **Getting Started** | [Quickstart](docs/user-guide/quickstart.rst) | 5-minute tutorial |
| **CLI Commands** | [CLI_REFERENCE.md](CLI_REFERENCE.md) | Complete command reference |
| **Configuration** | [Configuration Guide](docs/user-guide/configuration.rst) | Setup and templates |
| **API Usage** | [API_REFERENCE.md](API_REFERENCE.md) | Python API documentation |
| **Testing** | [Testing Guide](homodyne/tests/README.md) | Test framework documentation |
| **Runtime System** | [Runtime Guide](homodyne/runtime/README.md) | Shell completion & GPU setup |

### Complete Documentation

- **Primary Site**: https://homodyne.readthedocs.io/
- **User Guides**: Installation, quickstart, configuration, tutorials
- **Developer Resources**: Architecture, contributing, performance, troubleshooting
- **API Reference**: Core analysis, optimization methods, utilities

---

## 🔬 Theoretical Background

The package implements three key equations for correlation functions in nonequilibrium systems:

**Full Nonequilibrium Laminar Flow (Eq. 13)**:
$$c_2(\\vec{q}, t_1, t_2) = 1 + \\beta\\left[e^{-q^2\\int J(t)dt}\\right] \\times \\text{sinc}^2\\left[\\frac{1}{2\\pi} qh \\int\\dot{\\gamma}(t)\\cos(\\phi(t))dt\\right]$$

**Key Parameters**:
- $\\vec{q}$: scattering wavevector [Å⁻¹]
- $h$: gap between stator and rotor [Å]
- $\\phi(t)$: angle between shear/flow direction and $\\vec{q}$ [degrees]
- $\\dot{\\gamma}(t)$: time-dependent shear rate [s⁻¹]
- $D(t)$: time-dependent diffusion coefficient [Å²/s]

---

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@article{he2024transport,
  title={Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter},
  author={He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh and Chen, Wei},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={31},
  pages={e2401162121},
  year={2024},
  publisher={National Academy of Sciences},
  doi={10.1073/pnas.2401162121}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/developer-guide/contributing.md) for development workflow and standards.

### Development Setup

```bash
git clone https://github.com/imewei/homodyne.git
cd homodyne
pip install -e .[all]

# Setup development environment
homodyne-post-install --shell zsh --gpu --advanced

# Run tests
pytest -c pytest-quick.ini        # Quick tests
pytest -c pytest-full.ini         # Full suite

# Code quality checks
black homodyne/                    # Format code
ruff check homodyne/              # Linting
bandit -r homodyne/               # Security scan
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Authors**: Wei Chen, Hongrui He (Argonne National Laboratory)