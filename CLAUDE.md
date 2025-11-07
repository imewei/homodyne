# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Start

**Homodyne v2.3** is a CPU-optimized high-performance package for X-ray Photon Correlation Spectroscopy (XPCS) analysis. It implements the theoretical framework from [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121).

**Core Equation:** `c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²`

**Version:** 2.3.0 | **Python:** 3.12+ | **JAX:** 0.8.0 (CPU-only)

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

## Recent Updates (November 7, 2025)

**Version 2.3.0: GPU Support Removed - CPU-Only Architecture** (November 7, 2025)
- **Breaking Change**: All GPU acceleration support removed
- **Hard break**: No migration path - choose v2.2.1 (GPU) or v2.3.0 (CPU-only)
- **Rationale**: Simplify maintenance, focus on reliable HPC CPU optimization
- **CPU-only JAX**: All installation now uses CPU-only JAX 0.8.0
- **Removed features**:
  - CLI flags: `--force-cpu`, `--gpu-memory-fraction` (deleted)
  - Configuration keys: `gpu_memory_fraction`, `force_cpu`, `cuda_device_id` (gracefully ignored)
  - API functions: `configure_system_cuda`, `detect_system_cuda`, `get_gpu_memory_info`, etc. (9 functions removed)
  - Makefile targets: `install-jax-gpu`, `gpu-check`, `test-gpu` (deleted)
  - Example files: `gpu_accelerated_optimization.py`, `gpu_acceleration.py` (deleted)
  - System validation: GPU Setup test removed (9 tests now, was 10)
  - Runtime module: `homodyne/runtime/gpu/` directory deleted
  - Device module: `homodyne/device/gpu.py` deleted
- **New CPU-focused examples**:
  - `examples/cpu_optimization.py` - HPC multi-core CPU usage patterns
  - `examples/multi_core_batch_processing.py` - Parallel CPU workflows
- **Migration guide**: See [v2.2-to-v2.3 GPU Removal Guide](docs/migration/v2.2-to-v2.3-gpu-removal.md)
- **For GPU users**: Stay on v2.2.1 (maintained, available on PyPI)
- **For CPU users**: Upgrade to v2.3.0 (recommended, simpler, more reliable)

______________________________________________________________________

## Previous Updates

**Version 2.2.1: Per-Angle Scaling Parameter Fix** (November 6, 2025)
- **Critical Fix**: Resolves parameter initialization bug for per-angle scaling with large datasets
- **Root Cause**: Parameter count mismatch (9 provided, 13 needed for 3 angles) + incorrect parameter ordering
- **Solution**: Automatic parameter expansion (9 → 13) with correct ordering for StratifiedResidualFunction
- **Gradient Sanity Check**: New pre-optimization validation detects zero-gradient issues before wasting time
- **Uses**: NLSQ's `least_squares()` directly with StratifiedResidualFunction
- **Full NLSQ power**: JAX-accelerated, CPU-optimized, trust-region optimization
- **Automatic activation**: Activates for datasets ≥1M points with per-angle scaling
- **Validation**: Comprehensive parameter count and gradient checks before optimization
- **Test coverage**: 20/20 unit tests passing for StratifiedResidualFunction
- **Performance**: 93.15% cost reduction, 113 function evaluations (vs 0 before fix)
- **Graceful fallback**: Falls back to curve_fit_large if stratified least_squares fails

**How It Works:**

v2.2.1 fixes the root cause of silent optimization failures with per-angle scaling:

1. **Parameter Expansion** (automatic):
   - Config provides: 9 parameters (7 physical + contrast + offset)
   - Per-angle scaling needs: 13 parameters for 3 angles (7 physical + 3×contrast + 3×offset)
   - **Solution**: Automatically expand `[contrast, offset]` → `[c₀, c₁, c₂, o₀, o₁, o₂]`
   - Correct ordering: `[scaling_params, physical_params]` (matches StratifiedResidualFunction)

2. **Gradient Sanity Check** (pre-optimization validation):
   - Computes residuals at initial parameters
   - Perturbs first parameter by 1% and recomputes residuals
   - Estimates gradient magnitude from residual change
   - **Fails fast** if gradient < 1e-10 (prevents wasting time on broken optimization)
   - Provides detailed diagnostics for debugging

3. **StratifiedResidualFunction**: Wraps residual computation with angle-aware chunking
   - Each chunk processes independently with JIT compilation
   - All chunks contain all angles (validated on initialization)
   - Computes weighted residuals: `(g2_obs - g2_theory) / sigma`
   - Uses NLSQ's `least_squares()` instead of `curve_fit_large()` for full control

4. **Automatic Activation**: When all criteria met:
   - Stratified data created (v2.2.0 stratification)
   - Per-angle scaling enabled
   - Dataset ≥ 1M points
   - Gradient sanity check passes

**Technical Details:**
```python
# Parameter expansion (automatic in nlsq_wrapper.py)
validated_params = [contrast, offset, D0, alpha, ...]  # 9 params from config
n_angles = 3

# Expand scaling parameters per angle
contrast_per_angle = [contrast] * n_angles  # [c, c, c]
offset_per_angle = [offset] * n_angles      # [o, o, o]

# Create parameters in StratifiedResidualFunction order
expanded_params = [*contrast_per_angle, *offset_per_angle, D0, alpha, ...]  # 13 params

# Gradient sanity check (before optimization)
residuals_0 = residual_fn(expanded_params)
params_test = expanded_params.copy()
params_test[0] *= 1.01  # 1% perturbation
residuals_1 = residual_fn(params_test)
gradient_estimate = abs(sum(residuals_1 - residuals_0))

if gradient_estimate < 1e-10:
    raise ValueError("Zero gradient detected - optimization cannot proceed")
# ✅ Gradient check passed: 5.614e+03

# Optimization with stratified residual function
residual_fn = StratifiedResidualFunction(stratified_data, ...)
result = least_squares(fun=residual_fn, x0=expanded_params, bounds=bounds, ...)
# ✅ 113 function evaluations, 93.15% cost reduction
```

**Files:**
- `homodyne/optimization/stratified_residual.py` - StratifiedResidualFunction class (496 lines)
- `homodyne/optimization/nlsq_wrapper.py` - Parameter expansion + gradient check (lines 500-595, 2506-2559)
- `tests/unit/test_stratified_residual.py` - Comprehensive unit tests (20 tests)
- `docs/architecture/nlsq-least-squares-solution.md` - Technical documentation

**Validation Results** (3M points, 3 angles, CPU):
- Gradient sanity check: ✅ PASSED (5.614e+03)
- Function evaluations: 113 (vs 1 before fix)
- Cost reduction: 93.15% (vs 0% before fix)
- Optimization time: 146.77s (14-core CPU)
- Status: SUCCESS

**Status:** ✅ **Production Ready** - Resolves all silent optimization failures with per-angle scaling.

______________________________________________________________________

**Version 2.2.0: Angle-Stratified Chunking** (November 6, 2025)
- **Critical Fix**: Per-angle scaling now works correctly with large datasets (>1M points)
- Implemented angle-stratified chunking to ensure all chunks contain all phi angles
- **Impact**: Resolves silent NLSQ optimization failures (0 iterations, gradient=0)
- **Performance**: <1% overhead (<0.5s for 3M points), 2x memory peak (temporary)
- **Automatic activation**: No configuration changes required for existing workflows
- **Configurable**: Full control via `optimization.stratification` section in YAML
- **Test coverage**: 47/47 tests passing (unit + integration + performance)
- **Backward compatible**: Zero breaking changes, existing configs work unchanged
- See [Release Notes](docs/releases/v2.2-stratification-release-notes.md) for details

**Configuration:**
```yaml
optimization:
  stratification:
    enabled: "auto"                  # "auto" | true | false
    target_chunk_size: 100000        # Points per chunk
    max_imbalance_ratio: 5.0        # Balance threshold
    check_memory_safety: true        # Memory safety checks
```

**Activation Criteria** (all must be true):
- Dataset ≥ 100k points
- Per-angle scaling enabled (`per_angle_scaling=True`)
- Angle distribution balanced (imbalance ratio ≤ 5.0)
- Not explicitly disabled (`enabled != false`)

______________________________________________________________________

## Previous Updates (October 31, 2025)

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
- CPU-only installation (all platforms: Linux, macOS, Windows)
- Platform auto-detection in Makefile
- **Note**: GPU support removed in v2.3.0 (see migration guide)

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
- New module: `homodyne.device` for CPU management

### Breaking Changes from v2.2 → v2.3

**GPU Support Removal** (Hard Break in v2.3.0)
- Removed all GPU acceleration (see [Migration Guide](docs/migration/v2.2-to-v2.3-gpu-removal.md))
- Removed CLI flags: `--force-cpu`, `--gpu-memory-fraction`
- Removed config keys: `gpu_memory_fraction`, `force_cpu`, `cuda_device_id`
- Removed 9 GPU API functions from `homodyne.device`
- Removed 3 Makefile targets: `install-jax-gpu`, `gpu-check`, `test-gpu`
- Removed GPU example files, runtime modules, and device code
- **For GPU users**: Stay on v2.2.1 (last GPU-supporting version)
- **For CPU users**: Upgrade to v2.3.0 (recommended)

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
│   │   └── templates/          # 3 YAML templates
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
│   ├── device/                 # CPU management (v2.3.0: GPU removed)
│   │   ├── __init__.py         # configure_optimal_device()
│   │   └── cpu.py              # HPC CPU optimization
│   ├── optimization/           # NLSQ & MCMC
│   │   ├── nlsq_wrapper.py     # NLSQ integration (CPU-optimized)
│   │   ├── strategy.py         # Dataset strategy selection
│   │   ├── checkpoint_manager.py # HDF5 checkpoints
│   │   ├── mcmc.py             # NumPyro/BlackJAX MCMC
│   │   └── exceptions.py       # Custom exceptions
│   ├── runtime/                # Runtime features
│   │   ├── shell/              # Shell completion
│   │   └── utils/              # system_validator.py
│   └── utils/                  # Logging, progress
├── tests/
│   ├── unit/                   # Function-level tests
│   ├── integration/            # End-to-end workflows
│   ├── performance/            # Benchmarks
│   ├── mcmc/                   # Statistical validation
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

### CPU-Optimized JAX Design (v2.3.0)

**Primary Optimization:** NLSQ trust-region solver
- Levenberg-Marquardt algorithm via NLSQ package
- JIT-compiled, CPU-optimized execution
- Strategy selection: STANDARD → LARGE → CHUNKED → STREAMING

**Secondary Optimization:** NumPyro/BlackJAX MCMC
- Automatic NUTS/CMC selection: `(num_samples >= 15) OR (memory > 30%)` → CMC
- NUTS sampling for uncertainty quantification (small datasets)
- CMC parallelization for many samples or large memory requirements
- Physics-informed priors from `ParameterSpace` (no initialization needed)
- Auto-retry mechanism with convergence failures (max 3 retries)
- Built-in progress tracking

**Device Management:**
- `configure_optimal_device()` - CPU-only device configuration
- HPC CPU optimization (36/128-core nodes)
- Multi-core parallelization for MCMC and batch processing

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
2. **G2 Computation** (`core/jax_backend.py:compute_g2_scaled`) - Vectorized over phi angles, CPU-optimized
3. **Memory Management** (`data/memory_manager.py`) - Data chunking, efficient JAX arrays

### CPU Optimization Best Practices (v2.3.0)

**Multi-Core Configuration:**
```python
import os
import psutil

# Reserve 2 cores for OS
cpu_count = psutil.cpu_count(logical=False)
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={cpu_count - 2}'
```

**HPC Cluster Setup (Slurm Example):**
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=02:00:00

export OMP_NUM_THREADS=34  # Reserve 2 for OS
homodyne --config config.yaml
```

**Performance Tips:**
- Use multi-core CPUs (14+ cores recommended)
- Configure thread pools for parallel processing
- Use NUMA-aware configuration on large HPC nodes
- See `examples/cpu_optimization.py` for comprehensive guide

**Note:** GPU support removed in v2.3.0. For GPU features, use v2.2.1 (last GPU-supporting version).

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

### 9 Validation Tests (v2.3.0: GPU test removed)

| Test | Weight | Type | Runtime | Checks |
|------|--------|------|---------|--------|
| Dependency Versions | 20% | Critical | 0.006s | JAX==0.8.0, jaxlib==0.8.0 exact match (CPU-only) |
| JAX Installation | 20% | Critical | 0.14s | Import, CPU devices |
| NLSQ Integration | 15% | Important | 0.001s | curve_fit, StreamingOptimizer |
| Config System | 10% | Important | 0.005s | ConfigManager, templates |
| Data Pipeline | 10% | Important | 0.003s | h5py, XPCSDataLoader, phi filtering |
| Homodyne Installation | 10% | Important | 2.1s | 4 commands available |
| Environment Detection | 7% | Baseline | 0.0003s | Platform, Python, venv |
| Shell Completion | 6% | Convenience | 0.004s | Completion scripts, aliases |
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
```

### Test Coverage Highlights

- **186+ NLSQ tests**: Strategy selection (41), integration (32), stratified residual (31), comprehensive (82+)
- **72 angle filtering tests**: Unit, integration, performance, normalization
- **95 parameter management tests**: Core (28), caching (10), advanced (17), physics (32), integration (8)
- **31 stratified residual tests**: Initialization, validation, residual computation, diagnostics, edge cases

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

# CPU monitoring
htop  # Interactive CPU monitor
top -H -p $(pgrep -f homodyne)  # Thread-level CPU usage

# Full JAX traces
JAX_TRACEBACK_FILTERING=off python script.py
```

### Debugging CPU/JAX (v2.3.0)

```bash
# Device status (always CPU)
python -c "from homodyne.device import get_device_status; print(get_device_status())"

# CPU benchmark
python -c "from homodyne.device import benchmark_cpu_performance; print(benchmark_cpu_performance())"

# JAX CPU devices
python -c "import jax; print(jax.devices())"
# Expected: [CpuDevice(id=0)]

# Check CPU thread configuration
python -c "import os; print(f'XLA_FLAGS: {os.environ.get(\"XLA_FLAGS\", \"not set\")}')"
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

**Issue:** ✅ **COMPLETELY RESOLVED in v2.2.1** - Silent optimization failures with per-angle scaling on large datasets (>1M points)

**Status:** This issue has been completely resolved in Homodyne v2.2.1 with parameter expansion and gradient validation.

**Root Cause (Discovered November 6, 2025):**
The original v2.2.0/v2.2.1 failures were NOT due to double-chunking, but due to:
1. **Parameter count mismatch**: Config provided 9 parameters, but per-angle scaling with 3 angles needs 13
2. **Incorrect parameter ordering**: Parameters extracted in wrong order from config
3. **Silent failure**: Zero gradients caused immediate termination with no clear error

**Complete Solution (v2.2.1):**
Three-layer fix resolves all silent optimization failures:
- ✅ **Automatic parameter expansion**: 9 → 13 parameters (7 physical + 3×2 scaling)
- ✅ **Correct parameter ordering**: Matches StratifiedResidualFunction expectations
- ✅ **Gradient sanity check**: Pre-optimization validation detects zero-gradient issues
- ✅ **Stratified least-squares**: Uses NLSQ's `least_squares()` with StratifiedResidualFunction
- ✅ **Full NLSQ power**: JAX-accelerated, CPU-optimized, trust-region optimization
- ✅ **Graceful fallback**: Falls back to curve_fit_large if needed

**For Current v2.2.1 Users:**
No action needed - per-angle scaling works perfectly on all dataset sizes. The fixes activate automatically when:
- Dataset ≥ 1M points
- Per-angle scaling enabled
- Stratified data created (v2.2.0 stratification)

**Validation Results** (3M points, 3 angles):
```
Before fix (v2.2.0):
- Function evaluations: 1
- Cost reduction: 0%
- Parameters changed: False
- Gradient: 0.0 (zero!)

After fix (v2.2.1):
- Gradient sanity check: PASSED (5.614e+03)
- Function evaluations: 113
- Cost reduction: 93.15%
- Parameters changed: True
- Optimization time: 146.77s (CPU)
```

**Technical Implementation:**
```python
# Parameter expansion (automatic)
validated_params = [contrast, offset, D0, ...]  # 9 from config
expanded_params = [c0, c1, c2, o0, o1, o2, D0, ...]  # 13 expanded

# Gradient sanity check (before optimization)
if gradient_estimate < 1e-10:
    raise ValueError("Zero gradient - optimization cannot proceed")

# Optimization with stratified residual function
residual_fn = StratifiedResidualFunction(stratified_data, ...)
result = least_squares(fun=residual_fn, x0=expanded_params, ...)
```

See:
- Parameter expansion: `homodyne/optimization/nlsq_wrapper.py` lines 500-595
- Gradient check: `homodyne/optimization/nlsq_wrapper.py` lines 2506-2559
- Implementation: `homodyne/optimization/stratified_residual.py`
- Tests: `tests/unit/test_stratified_residual.py` (20 tests passing)

**Historical Context (v2.1.x and earlier):**
Prior to v2.2.0, this was a critical issue where:
- NLSQ chunking split data arbitrarily without angle awareness
- Chunks missing certain angles had zero gradient for those angle parameters
- Result: Silent optimization failure with 0 iterations

**Upgrade Recommendation:**
Users on v2.2.0 or earlier experiencing per-angle scaling issues should upgrade to v2.2.1:
```bash
pip install --upgrade homodyne
```

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

### Core Stack (v2.3.0 - CPU-only)

**Required:**
- **JAX 0.8.0 + jaxlib 0.8.0** (exact match required, CPU-only)
  - Installation: `pip install jax==0.8.0 jaxlib==0.8.0`
  - **GPU support removed in v2.3.0** (use v2.2.1 for GPU)
- **nlsq >= 0.1.0** - Trust-region optimization
- **NumPyro >= 0.18.0, <0.20.0** - MCMC with progress bars
- **BlackJAX >= 1.2.0, <2.0.0** - Alternative MCMC backend
- **NumPy >= 2.0.0, <3.0.0** - NumPy 2.x series
- **SciPy >= 1.14.0, <2.0.0**
- **h5py >= 3.10.0, <4.0.0** - HDF5 data files
- **PyYAML >= 6.0.2** - Configuration
- **matplotlib >= 3.8.0, <4.0.0** - Plotting
- **psutil >= 6.0.0** - System utilities

### Platform Support (v2.3.0)

- **Python**: 3.12+ (all platforms)
- **CPU**: Linux, macOS, Windows (full support, multi-core optimized)
- **GPU**: Not supported (removed in v2.3.0)

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