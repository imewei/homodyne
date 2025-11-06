# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Start

**Homodyne v2.1** is a JAX-first high-performance package for X-ray Photon Correlation Spectroscopy (XPCS) analysis. It implements the theoretical framework from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121).

**Core Equation:** `c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²`

**Version:** 2.1.0 | **Python:** 3.12+ | **JAX:** 0.8.0

## Essential Links

- [Development Commands](#development-commands) - Daily workflow commands
- [Architecture](#architecture) - System design overview
- [System Validation](#system-validation) - Health check tool
- [Common Issues](#known-issues) - Troubleshooting guide

______________________________________________________________________

## Table of Contents

1. [Recent Updates](#recent-updates-october-28-2025)
2. [Version 2.0.0 Highlights](#version-200-highlights)
3. [Development Commands](#development-commands)
4. [Repository Structure](#repository-structure)
5. [Architecture](#architecture)
6. [System Validation](#system-validation)
7. [CLI Usage](#command-line-interface)
8. [Testing Strategy](#testing-strategy)
9. [Common Tasks](#common-development-tasks)
10. [Known Issues](#known-issues)
11. [Dependencies](#dependencies)

______________________________________________________________________

## Recent Updates (October 31, 2025)

**Version 2.1.0: Simplified MCMC Implementation** (October 31, 2025)
- **Breaking Change**: Removed `--method nuts` and `--method cmc` CLI flags
- Only `--method nlsq` and `--method mcmc` supported (75% reduction in CLI complexity)
- Automatic NUTS/CMC selection: `(num_samples >= 15) OR (memory > 30%)` → CMC
- Removed automatic NLSQ/SVI initialization from MCMC (manual workflow required)
- Removed `mcmc.initialization` section from all YAML templates
- Added configurable thresholds: `min_samples_for_cmc: 15`, `memory_threshold_pct: 0.30`
- Implemented auto-retry mechanism with convergence failures (max 3 retries)
- See [Migration Guide](docs/migration/v2.0-to-v2.1.md) for upgrade instructions

**CMC Performance Optimization** (October 28, 2025)
- Optimized CMC selection thresholds for CPU parallelism
- `min_samples_for_cmc`: 100 → 20 → **15** (more aggressive parallelism)
- `memory_threshold_pct`: 0.50 → 0.40 → **0.30** (more conservative OOM prevention)
- **Impact**: 20-sample experiment on 14-core CPU now triggers CMC for ~1.4x speedup

**Architecture Documentation** (October 28, 2025)
- New comprehensive architecture documentation section
- `docs/architecture/cmc-dual-mode-strategy.md` - CMC design (3,500+ words)
- `docs/architecture/nuts-chain-parallelization.md` - NUTS chains (4,000+ words)
- Quick reference guides for both CMC and NUTS strategies

**CMC Dual-Criteria Logic** (October 28, 2025)
- Implemented OR logic: `(num_samples >= 15) OR (memory > 30%)`
- **Use Case 1**: Parallelism - many samples (e.g., 50 phi angles → 3x speedup)
- **Use Case 2**: Memory management - large datasets (avoid OOM errors)
- Hardware-adaptive decision making for optimal performance

**MCMC Result Saving** (October 28, 2025)
- Comprehensive result saving for both NUTS and CMC methods
- HDF5 format with posterior samples, diagnostics, and metadata
- Visualization support with trace plots, corner plots, autocorrelation

______________________________________________________________________

## Version 2.0.0 Highlights

**Release Date:** October 2025 | **Status:** Production Ready

### Core Features

**System Validation Tool** (October 24, 2025)
- 10 automated tests with weighted health scoring (0-100)
- Quick mode: ~0.15s (vs ~6.5s full validation)
- Error codes: EDEP_001, EJAX_001, ENLSQ_001, ECONFIG_001, EDATA_001
- CI/CD ready with JSON output
- File: `homodyne/runtime/utils/system_validator.py`

**Cross-Platform JAX Support** (October 21, 2025)
- CPU-first installation (all platforms: Linux, macOS, Windows)
- Optional GPU on Linux (CUDA 12.1-12.9)
- Platform auto-detection in Makefile
- Follows PyTorch installation model

**NLSQ Optimization Engine** (October 22, 2025)
- StreamingOptimizer for unlimited datasets (>100M points)
- Automatic strategy selection: STANDARD → LARGE → CHUNKED → STREAMING
- Checkpoint/resume capability with HDF5 compression
- 5 error-specific recovery strategies
- 155+ tests, 100% backward compatibility

**Shell Completion & CLI Tools** (October 24, 2025)
- 4 commands: `homodyne`, `homodyne-config`, `homodyne-post-install`, `homodyne-cleanup`
- Intelligent completion for bash/zsh/fish
- Environment detection: conda vs venv/uv/virtualenv
- Convenient aliases: `hm-nlsq`, `hm-mcmc`, `hc-stat`, `hconfig`

### Scientific Validation

**Results:** 100% test pass rate (7/7 tests)
- Ground truth recovery: 1.88-14.23% error (XPCS standards)
- Numerical stability: χ² consistency 0.00% across 5 initial conditions
- Performance: Sub-linear scaling (0.92x time for 8x data)
- Physics validation: 100% constraint satisfaction

### Breaking Changes from v1.x

- Removed JAXFit optimization engine → replaced with NLSQ
- Removed VI (Variational Inference) → use NLSQ or MCMC
- Removed `performance.subsampling` config → NLSQ handles automatically
- New module: `homodyne.device` for CPU/GPU management

### Breaking Changes from v2.0 → v2.1

**CLI Method Flags** (Hard Break)
- Removed `--method nuts` and `--method cmc` (use `--method mcmc` only)
- Automatic NUTS/CMC selection based on dataset characteristics
- No deprecation warnings (acknowledged breaking change)

**YAML Configuration Structure**
- Removed entire `mcmc.initialization` section from templates
- Removed fields: `run_nlsq_init`, `use_svi`, `svi_steps`, `svi_timeout`
- Added: `min_samples_for_cmc: 15` to `optimization.mcmc`
- Added: `memory_threshold_pct: 0.30` to `optimization.mcmc`
- Added: `dense_mass_matrix: false` to `optimization.mcmc`
- `initial_parameters.values` structure unchanged (backward compatible)

**MCMC Initialization Behavior**
- No automatic NLSQ/SVI initialization before MCMC
- Manual workflow required: Run NLSQ → copy results → update YAML → run MCMC
- Physics-informed priors from `ParameterSpace` used directly
- Complete separation between NLSQ and MCMC methods

**Migration Required**: See [Migration Guide](docs/migration/v2.0-to-v2.1.md) for step-by-step upgrade instructions

______________________________________________________________________

## Development Commands

### Quick Reference

```bash
# Testing
make test              # Core tests
make test-all          # All tests + coverage
make quality           # Format + lint + type-check

# Installation
make dev               # CPU-only (all platforms)
make install-jax-gpu   # GPU support (Linux + CUDA 12.1-12.9 only)

# Validation
python -m homodyne.runtime.utils.system_validator           # Full (~6.5s)
python -m homodyne.runtime.utils.system_validator --quick   # Quick (~0.15s)

# Documentation
make docs              # Build Sphinx docs
make clean-all         # Clean artifacts
```

### Testing Commands

```bash
make test-unit         # Unit tests only
make test-integration  # Integration tests
make test-performance  # Benchmarks
make test-nlsq         # NLSQ optimization
make test-mcmc         # MCMC statistical validation
make test-gpu          # GPU validation (Linux only)
```

### Code Quality

```bash
make format            # Auto-format (black + ruff)
make lint              # Run ruff linting
pre-commit run --all-files  # Run all pre-commit hooks
```

### System Checks

```bash
make deps-check        # Check dependencies
make gpu-check         # Check GPU/CUDA (Linux only)
```

______________________________________________________________________

## Repository Structure

```
homodyne/
├── homodyne/
│   ├── cli/                    # Command-line interface
│   │   ├── main.py             # Entry point
│   │   ├── args_parser.py      # Argument parsing
│   │   ├── commands.py         # Command implementations
│   │   └── config_generator.py # Interactive config builder
│   ├── config/                 # Configuration system
│   │   ├── manager.py          # YAML/JSON config loading
│   │   ├── parameter_manager.py # Parameter bounds & validation
│   │   ├── types.py            # TypedDict definitions
│   │   └── templates/          # 4 YAML templates
│   ├── core/                   # JAX physics engine
│   │   ├── jax_backend.py      # compute_g1, compute_g2, chi_squared
│   │   ├── physics.py          # Physical models
│   │   ├── models.py           # Object-oriented wrappers
│   │   ├── fitting.py          # ScaledFittingEngine
│   │   └── theory.py           # TheoryEngine
│   ├── data/                   # Data pipeline
│   │   ├── xpcs_loader.py      # HDF5 loading (XPCSDataLoader)
│   │   ├── preprocessing.py    # Data preparation
│   │   ├── phi_filtering.py    # Angular filtering
│   │   └── memory_manager.py   # Memory-efficient handling
│   ├── device/                 # CPU/GPU management
│   │   ├── __init__.py         # configure_optimal_device()
│   │   ├── gpu.py              # GPU/CUDA configuration
│   │   └── cpu.py              # HPC CPU optimization
│   ├── optimization/           # NLSQ & MCMC
│   │   ├── nlsq_wrapper.py     # NLSQ integration (423 lines)
│   │   ├── strategy.py         # Dataset strategy selection
│   │   ├── checkpoint_manager.py # HDF5 checkpoints
│   │   ├── mcmc.py             # NumPyro/BlackJAX MCMC
│   │   └── exceptions.py       # Custom exceptions
│   ├── runtime/                # Runtime features
│   │   ├── shell/              # Shell completion
│   │   ├── gpu/                # GPU activation scripts
│   │   └── utils/              # system_validator.py (1,577 lines)
│   └── utils/                  # Logging, progress
├── tests/
│   ├── unit/                   # Function-level tests
│   ├── integration/            # End-to-end workflows
│   ├── performance/            # Benchmarks
│   ├── mcmc/                   # Statistical validation
│   ├── gpu/                    # GPU/CUDA tests
│   ├── api/                    # Backward compatibility
│   └── factories/              # Test data generators
├── docs/                       # Sphinx documentation
│   ├── architecture/           # Architecture documentation (NEW)
│   │   ├── cmc-dual-mode-strategy.md          # CMC design (3,500+ words)
│   │   ├── cmc-decision-quick-reference.md    # CMC quick reference
│   │   ├── nuts-chain-parallelization.md      # NUTS execution modes (4,000+ words)
│   │   └── nuts-chain-parallelization-quick-reference.md  # NUTS quick reference
│   └── ...                     # User guides, API docs, advanced topics
├── examples/                   # Example scripts
├── pyproject.toml              # Package metadata
├── Makefile                    # Development commands
└── CLAUDE.md                   # This file
```

______________________________________________________________________

## Architecture

### JAX-First Design

**Primary Optimization:** NLSQ trust-region solver
- Levenberg-Marquardt algorithm via NLSQ package
- JIT-compiled, CPU/GPU transparent execution
- Strategy selection: STANDARD → LARGE → CHUNKED → STREAMING

**Secondary Optimization:** NumPyro/BlackJAX MCMC
- Automatic NUTS/CMC selection: `(num_samples >= 15) OR (memory > 30%)` → CMC
- NUTS sampling for uncertainty quantification (small datasets)
- CMC parallelization for many samples or large memory requirements
- Physics-informed priors from `ParameterSpace` (no initialization needed)
- Auto-retry mechanism with convergence failures (max 3 retries)
- Built-in progress tracking

**Device Management:**
- `configure_optimal_device()` - Auto-detect optimal device
- Graceful GPU→CPU fallback
- HPC CPU optimization (36/128-core nodes)

### Analysis Modes

**Static Isotropic:**
- Physical parameters (3): `[D₀, α, D_offset]`
- Total with scaling (5): `[contrast, offset, D₀, α, D_offset]`

**Laminar Flow:**
- Physical parameters (7): `[D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]`
- Total with scaling (9): All 7 + `[contrast, offset]`

**Note:** Optimization automatically adds scaling parameters (contrast=0.5, offset=1.0).

### Data Loading (Three-Tier)

1. **ConfigManager** (`config/manager.py`) - YAML/JSON loading
2. **XPCSDataLoader** (`data/xpcs_loader.py`) - HDF5 data (APS old/new formats)
3. **CLI Commands** (`cli/commands.py`) - Workflow orchestration

### Parameter Management

Centralized via `ParameterManager` class:
- Automatic name mapping (config → code: `gamma_dot_0` → `gamma_dot_t0`)
- Config bounds override support
- Physics constraint validation (ERROR, WARNING, INFO)
- Performance caching (~10-100x speedup)

**Usage:**
```python
from homodyne.config.parameter_manager import ParameterManager
pm = ParameterManager(config_dict, "laminar_flow")
bounds = pm.get_parameter_bounds(["D0", "alpha"])
```

### Critical Performance Paths

1. **Residual Calculation** (`core/jax_backend.py:compute_residuals`) - JIT-compiled, called repeatedly
2. **G2 Computation** (`core/jax_backend.py:compute_g2_scaled`) - Vectorized over phi angles, GPU-accelerated
3. **Memory Management** (`data/memory_manager.py`) - Data chunking, efficient JAX arrays

### GPU Support

**Requirements:**
- Linux x86_64 or aarch64 only
- CUDA 12.1-12.9 (system pre-installed)
- NVIDIA driver >= 525

**Installation:**
```bash
# Load system CUDA
module load cuda/12.2

# Install JAX with CUDA
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Install homodyne
pip install homodyne[dev]
```

**Auto-fallback to CPU:**
- Windows or macOS
- CUDA not available
- GPU initialization fails

______________________________________________________________________

## System Validation

### Quick Start

```bash
# Full validation (recommended after installation)
python -m homodyne.runtime.utils.system_validator

# Quick validation (CI/CD)
python -m homodyne.runtime.utils.system_validator --quick

# Test specific component
python -m homodyne.runtime.utils.system_validator --test jax

# JSON output
python -m homodyne.runtime.utils.system_validator --json
```

### 10 Validation Tests

| Test | Weight | Type | Runtime | Checks |
|------|--------|------|---------|--------|
| Dependency Versions | 20% | Critical | 0.006s | JAX==0.8.0, jaxlib==0.8.0 exact match |
| JAX Installation | 20% | Critical | 0.14s | Import, devices, CUDA detection |
| NLSQ Integration | 15% | Important | 0.001s | curve_fit, StreamingOptimizer |
| Config System | 10% | Important | 0.005s | ConfigManager, templates |
| Data Pipeline | 10% | Important | 0.003s | h5py, XPCSDataLoader, phi filtering |
| Homodyne Installation | 10% | Important | 2.1s | 4 commands available |
| Environment Detection | 5% | Baseline | 0.0003s | Platform, Python, venv |
| Shell Completion | 5% | Convenience | 0.004s | Completion scripts, aliases |
| GPU Setup | 3% | Optional | 0.026s | NVIDIA driver, CUDA home |
| Integration | 2% | Optional | 4.2s | Module imports, scripts |

### Health Score

- 🟢 **90-100**: Excellent (production ready)
- 🟡 **70-89**: Good (minor issues, functional)
- 🟠 **50-69**: Fair (significant issues)
- 🔴 **0-49**: Poor (critical failures)

### Common Failures

**JAX/jaxlib version mismatch:**
```bash
pip install jax==0.8.0 jaxlib==0.8.0
```

**Missing NLSQ StreamingOptimizer:**
```bash
pip install --upgrade nlsq>=0.1.5
```

**Missing config templates:**
```bash
pip install --force-reinstall homodyne
```

______________________________________________________________________

## Command-Line Interface

### Basic Usage

```bash
# Standard analysis
homodyne --config config.yaml

# Override data file
homodyne --config config.yaml --data-file experiment.hdf

# Select method (v2.1.0: only nlsq and mcmc supported)
homodyne --config config.yaml --method nlsq   # Default: fast optimization
homodyne --config config.yaml --method mcmc   # Bayesian uncertainty quantification
                                              # Automatic NUTS/CMC selection

# With visualization
homodyne --config config.yaml --plot-experimental-data

# Logging control
homodyne --config config.yaml --verbose       # DEBUG level
homodyne --config config.yaml --quiet         # Errors only
```

### Manual NLSQ → MCMC Workflow (v2.1.0)

**Step-by-step process for using NLSQ results to initialize MCMC:**

```bash
# 1. Run NLSQ analysis first to get point estimates
homodyne --config config.yaml --method nlsq

# 2. Manually copy best-fit results from NLSQ output
#    Example output:
#    Best-fit parameters:
#      D0: 1234.5 ± 45.6
#      alpha: 0.567 ± 0.012
#      D_offset: 12.34 ± 1.23

# 3. Update config.yaml with NLSQ results
#    Edit initial_parameters.values:
#      parameter_names: [D0, alpha, D_offset]
#      values: [1234.5, 0.567, 12.34]  # From NLSQ output

# 4. Run MCMC analysis with initialized parameters
homodyne --config config.yaml --method mcmc

# 5. Automatic selection decides NUTS vs CMC based on:
#    - (num_samples >= 15) OR (memory > 30%) → CMC
#    - Otherwise → NUTS
```

**Note:** No automatic initialization in v2.1.0. Manual workflow ensures transparency and user control over parameter transfer between methods.

### Configuration Generator

```bash
# Interactive builder
homodyne-config --interactive

# From template
homodyne-config --mode static --output config.yaml
homodyne-config --mode laminar_flow --output flow.yaml

# Validate
homodyne-config --validate my_config.yaml
```

### Shell Completion

```bash
# Install (interactive)
homodyne-post-install --interactive

# Install for specific shell
homodyne-post-install --shell zsh   # or bash, fish

# Cleanup
homodyne-cleanup --interactive
```

### Aliases (After Installing Completion)

```bash
hm-nlsq --config config.yaml           # homodyne --method nlsq
hm-mcmc --config config.yaml           # homodyne --method mcmc
hc-stat --output static.yaml           # homodyne-config --mode static
hc-flow --output flow.yaml             # homodyne-config --mode laminar_flow
```

______________________________________________________________________

## Testing Strategy

### Test Organization

```bash
tests/
├── unit/                # Function-level (28 parameter tests, 41 strategy tests)
├── integration/         # End-to-end workflows
├── performance/         # Benchmarks
├── mcmc/               # Statistical validation
├── gpu/                # GPU/CUDA (Linux only)
├── api/                # Backward compatibility
├── property/           # Mathematical invariants
└── factories/          # Test data generators
```

### Running Tests

```bash
make test              # Core tests
make test-all          # All tests + coverage
make test-unit         # Unit only
make test-integration  # Integration only
make test-performance  # Benchmarks
make test-nlsq         # NLSQ specific
make test-mcmc         # MCMC specific
make test-gpu          # GPU validation (Linux only)
```

### Test Coverage Highlights

- **155+ NLSQ tests**: Strategy selection (41), integration (32), comprehensive (82+)
- **72 angle filtering tests**: Unit, integration, performance, normalization
- **95 parameter management tests**: Core (28), caching (10), advanced (17), physics (32), integration (8)

______________________________________________________________________

## Common Development Tasks

### Adding a New Optimization Method

1. Create file in `homodyne/optimization/`
2. Match interface pattern from `nlsq_wrapper.py`
3. Use residual functions from `core/jax_backend.py`
4. Add unit tests: `tests/unit/test_optimization_*.py`
5. Add integration tests: `tests/integration/test_workflows.py`
6. If MCMC-based: add to `tests/mcmc/`

### Modifying Physics Models

1. Update JAX functions in `core/jax_backend.py`
2. Ensure JIT compatibility (no Python control flow)
3. Update wrappers in `core/models.py`
4. Test gradient/hessian computations

### Debugging Performance

```bash
# JAX compilation logs
JAX_LOG_COMPILES=1 python script.py

# Profiling
make profile-nlsq
make profile-mcmc

# GPU monitoring
nvidia-smi -l 1

# Full JAX traces
JAX_TRACEBACK_FILTERING=off python script.py
```

### Debugging GPU/JAX

```bash
# Device status
python -c "from homodyne.device import get_device_status; print(get_device_status())"

# Benchmark
python -c "from homodyne.device import benchmark_device_performance; print(benchmark_device_performance())"

# CUDA validation
python -c "import jax; print(jax.devices())"
```

### API Compatibility

**Backward Compatible (v1.x → v2.0):**
- All `homodyne.data` functions/classes
- All `homodyne.core` validated components
- YAML configuration format

**New in v2.0:**
- `homodyne.optimization.fit_nlsq_jax()`
- `homodyne.optimization.fit_mcmc_jax()`
- `homodyne.device.configure_optimal_device()`
- `homodyne.device.get_device_status()`
- `homodyne.device.benchmark_device_performance()`
- `homodyne._version.__version__`

**Removed in v2.0:**
- JAXFit optimization engine
- VI (Variational Inference)
- `performance.subsampling` config section

______________________________________________________________________

## Known Issues

### NLSQ Optimization

**Issue:** `curve_fit_large()` returns `(popt, pcov)`, not `(popt, pcov, info)`

**Fix:**
```python
# ❌ WRONG:
popt, pcov, info = curve_fit_large(...)

# ✅ CORRECT:
popt, pcov = curve_fit_large(...)
info = {}  # Empty dict for consistency
```

**Issue:** Silent failure with 0 iterations

**Quick checks:**
```python
# Identity covariance matrix?
if np.allclose(pcov, np.eye(len(popt))):
    print("Fallback triggered")

# Zero iterations?
if result.nit == 0:
    print("Optimization didn't run")

# Parameters unchanged?
if np.allclose(popt, initial_params):
    print("No optimization occurred")
```

**Issue:** Per-angle scaling incompatible with NLSQ chunking for large datasets (>1M points)

**Symptoms:**
- NLSQ returns 0 iterations with unchanged parameters
- `first-order optimality 0.00e+00` (gradient = 0)
- Final cost = initial cost (no improvement)
- Analysis completes but with unoptimized results
- Only affects datasets >1M points (triggers LARGE/CHUNKED strategy)

**Root Cause (VERIFIED 2025-01-06):**
Per-angle scaling architecture is fundamentally incompatible with NLSQ's chunking implementation:

1. **Per-angle scaling** adds 2×n_phi parameters (contrast[i], offset[i] for each phi angle)
   - Example: 3 angles → 6 scaling params + 7 physics params = 13 total
2. **NLSQ chunking** splits 3M points into 11 chunks of 300k points each
3. **Problem**: Each chunk may not contain all phi angles
4. **Result**: Gradient w.r.t. angle-specific parameters is zero for chunks missing those angles
5. **Failure**: NLSQ's gradient aggregation across chunks fails with partial angle coverage

**Proof:**
```bash
# With per_angle_scaling=True (default):
Function evaluations: 1, gradient: 0.00, NO OPTIMIZATION ❌

# With per_angle_scaling=False:
Function evaluations: 53-127, gradient: 9.22-8290, WORKS ✅
Initial cost: 2202 → Final cost: 508
Reduced chi-squared: 1.39 (good fit)
```

**Immediate Workaround:**
Disable per-angle scaling for large datasets by modifying the fit call:
```python
# In your analysis script:
from homodyne.optimization.nlsq import fit_nlsq_jax
result = fit_nlsq_jax(data, config, per_angle_scaling=False)
```

**Alternative Workarounds:**
1. **Reduce dataset size** below 1M points via phi angle filtering (forces STANDARD strategy)
2. **Force STANDARD strategy** with `strategy_override: "standard"` (no chunking, but memory intensive)
3. **Use MCMC** instead of NLSQ (handles per-angle scaling correctly)

**Long-term Solutions:**
1. **Fix NLSQ chunking** to handle per-angle parameters correctly (requires NLSQ library changes)
2. **Restructure data** so each chunk contains all phi angles (pre-process before optimization)
3. **Use alternative optimizer** like scipy.optimize.least_squares (loses GPU acceleration)

**Note:** Per-angle scaling is physically correct and recommended for small datasets (<1M points). The issue only affects large datasets requiring chunking.

**See:** `docs/troubleshooting/nlsq-zero-iterations-investigation.md` for investigation timeline

### Plotting

**Issue:** Two-time correlation diagonal orientation

**Fix:**
```python
# ❌ WRONG:
ax.imshow(c2, origin='lower', extent=[t1[0], t1[-1], t2[0], t2[-1]])

# ✅ CORRECT:
ax.imshow(c2.T, origin='lower', extent=[t1[0], t1[-1], t2[0], t2[-1]])
```

**Physical validation:** Correct plots show bright diagonal from bottom-left to top-right.

### Documentation Resources

- Silent failures: `docs/troubleshooting/silent-failure-diagnosis.md`
- Zero-iteration investigation: `docs/troubleshooting/nlsq-zero-iterations-investigation.md`
- Plotting issues: `docs/troubleshooting/imshow-transpose-pitfalls.md`
- NLSQ docs: https://nlsq.readthedocs.io/en/latest/
- Performance guide: https://nlsq.readthedocs.io/en/latest/guides/performance_guide.html

______________________________________________________________________

## Dependencies

### Core Stack (2025)

**Required:**
- **JAX 0.8.0 + jaxlib 0.8.0** (exact match required)
  - Default: CPU-only (all platforms)
  - Linux + CUDA 12.1-12.9: Optional GPU via `jax[cuda12-local]`
- **nlsq >= 0.1.0** - Trust-region optimization
- **NumPyro >= 0.18.0, <0.20.0** - MCMC with progress bars
- **BlackJAX >= 1.2.0, <2.0.0** - Alternative MCMC backend
- **NumPy >= 2.0.0, <3.0.0** - NumPy 2.x series
- **SciPy >= 1.14.0, <2.0.0**
- **h5py >= 3.10.0, <4.0.0** - HDF5 data files
- **PyYAML >= 6.0.2** - Configuration
- **matplotlib >= 3.8.0, <4.0.0** - Plotting
- **psutil >= 6.0.0** - System utilities

### Platform Support

- **Python**: 3.12+ (all platforms)
- **CPU**: Linux, macOS, Windows (full support)
- **GPU**: Linux only (CUDA 12.1-12.9)

### Development

- pytest >= 8.3.0, <9.0.0
- black >= 25.0.0, <26.0.0
- ruff >= 0.13.0, <0.14.0
- mypy >= 1.18.0, <2.0.0
- pre-commit >= 4.0.0, <5.0.0

### Configuration Files

- `pyproject.toml` - Package metadata, tool configs
- `.pre-commit-config.yaml` - Black, ruff, mypy, bandit
- Requires: `python >= 3.12`
- Float32 recommended for MCMC and Homodyne supports both float32 and float64