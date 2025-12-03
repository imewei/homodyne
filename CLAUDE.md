# CLAUDE.md

**Homodyne v2.4** - CPU-optimized XPCS analysis implementing [He et al. PNAS 2024](https://doi.org/10.1073/pnas.2401162121)

**Core Equation:** `c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²`
**Version:** 2.4.0 | **Python:** 3.12+ | **JAX:** 0.8.0 (CPU-only)

## Quick Reference

| Section | Purpose |
|---------|---------|
| [Architecture](#architecture) | System design, optimization flow |
| [Development](#development-commands) | Testing, validation, debugging |
| [CLI Usage](#cli-usage) | Command patterns, workflows |
| [Known Issues](#known-issues) | Troubleshooting guide |

______________________________________________________________________

## Recent Updates

### v2.4.1 (Dec 2025) - CMC-Only Architecture & Single-Angle Fixes

- **MCMC**: CMC-only path (NUTS auto-selection removed for simplicity)
- **Single-Angle**: Log-space D0 sampling for convergence stability
- **Per-Phi Init**: New `homodyne/optimization/initialization/per_phi_initializer.py`
- **Linting**: Fixed F401/F841/E402/F811 issues across homodyne/

### v2.4.0 (Nov 2025) - Per-Angle Scaling Mandatory

- **Breaking**: Legacy scalar `per_angle_scaling=False` removed
- **Required**: Per-angle mode only (physically correct)
- **Migration**: Remove `per_angle_scaling` parameter or set to `True`
- **Impact**: 3 angles: 5 params → 9 params `[c₀,c₁,c₂, o₀,o₁,o₂, D0,α,D_offset]`

### v2.3.0 (Nov 2025) - CPU-Only Architecture

- **Breaking**: GPU support removed (use v2.2.1 for GPU)
- **Rationale**: Simplify maintenance, reliable HPC CPU optimization
- **New**: CPU examples, HPC multi-core patterns
- **Migration**: [v2.2-to-v2.3 Guide](docs/migration/v2.2-to-v2.3-gpu-removal.md)

### v2.2.1 (Nov 2025) - MCMC Stability Fixes

- **Critical Fixes**:
  - OOM prevention: Worker memory 4.0GB → 5.5GB, physical cores only
  - Parameter expansion: 9 → 13 params (per-angle scaling)
  - Offset bounds: [0.5, 2.5] → [0.5, 1.5] (MCMC stability)
- **Status**: Production ready, all tests passing

______________________________________________________________________

## Architecture

### Optimization Stack

1. **Primary**: NLSQ trust-region (Levenberg-Marquardt)
   - JIT-compiled, CPU-optimized
   - Auto strategy: STANDARD → LARGE → CHUNKED → STREAMING

2. **Secondary**: NumPyro/BlackJAX MCMC (CMC-only in v2.4.1+)
   - Physics-informed priors, auto-retry (max 3)
   - Single-angle: Log-space D0 sampling for stability

### Analysis Modes

**Static Isotropic** (3 params):
```
Physical: [D₀, α, D_offset]
With scaling: [contrast, offset, D₀, α, D_offset]
```

**Laminar Flow** (7 params):
```
Physical: [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
With scaling: [contrast, offset, D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
```

**Per-Angle Scaling** (v2.4.0 mandatory):
```
3 angles: [c₀,c₁,c₂, o₀,o₁,o₂, ...physical_params]
```

### Data Pipeline

```
ConfigManager → XPCSDataLoader → CLI Commands
   (YAML)         (HDF5 APS)      (orchestration)
```

### Parameter Management

**ParameterManager** (`config/parameter_manager.py`):
- Name mapping: `gamma_dot_0` → `gamma_dot_t0`
- Bounds validation, caching (~10-100x speedup)

**Critical Paths**:
1. `core/jax_backend.py:compute_residuals` (JIT, called repeatedly)
2. `core/jax_backend.py:compute_g2_scaled` (vectorized, CPU-optimized)
3. `data/memory_manager.py` (chunking, JAX arrays)

______________________________________________________________________

## Development Commands

### Essential

```bash
make test              # Core tests
make test-all          # All + coverage
make quality           # Format + lint + type-check
make dev               # Install CPU-only
make clean             # Clean build artifacts, cache, node_modules

# System validation
python -m homodyne.runtime.utils.system_validator --quick  # ~0.15s
```

### Testing

```bash
make test-unit         # Unit tests
make test-integration  # End-to-end
make test-nlsq         # NLSQ optimization
make test-mcmc         # MCMC validation
```

### Debugging

```bash
# JAX compilation
JAX_LOG_COMPILES=1 python script.py

# CPU monitoring
htop
top -H -p $(pgrep -f homodyne)

# Device status
python -c "from homodyne.device import get_device_status; print(get_device_status())"
```

______________________________________________________________________

## System Validation

**9 Tests** (v2.3.0: GPU removed):
- Dependency Versions (20%) - JAX==0.8.0 exact match
- JAX Installation (20%) - CPU devices
- NLSQ Integration (15%) - curve_fit, StreamingOptimizer
- Config/Data/Install (30%) - Templates, h5py, commands
- Shell/Integration (15%) - Completion, imports

**Health Score**:
- 🟢 90-100: Production ready
- 🟡 70-89: Functional (minor issues)
- 🔴 <70: Critical failures

**Common Fixes**:
```bash
pip install jax==0.8.0 jaxlib==0.8.0  # Version mismatch
pip install --upgrade nlsq>=0.1.5     # Missing StreamingOptimizer
```

______________________________________________________________________

## CLI Usage

### Basic Commands

```bash
# Standard analysis
homodyne --config config.yaml

# Method selection
homodyne --config config.yaml --method nlsq   # Fast optimization
homodyne --config config.yaml --method mcmc   # Bayesian UQ (CMC)

# Visualization
homodyne --config config.yaml --plot-experimental-data

# Logging
homodyne --config config.yaml --verbose  # DEBUG
homodyne --config config.yaml --quiet    # Errors only
```

### NLSQ → MCMC Workflow

```bash
# 1. Run NLSQ
homodyne --config config.yaml --method nlsq

# 2. Copy best-fit from output
#    D0: 1234.5 ± 45.6, alpha: 0.567 ± 0.012, ...

# 3. Update config initial_parameters.values
#    values: [1234.5, 0.567, ...]

# 4. Run MCMC
homodyne --config config.yaml --method mcmc
```

### Configuration

```bash
homodyne-config --interactive            # Interactive builder
homodyne-config --mode static            # From template
homodyne-config --validate config.yaml   # Validate

# Shell completion
homodyne-post-install --interactive
```

______________________________________________________________________

## Repository Structure

```
homodyne/
├── cli/           # main.py, args_parser.py, commands.py, config_generator.py
├── config/        # manager.py, parameter_manager.py, types.py, templates/
├── core/          # jax_backend.py, physics.py, models.py, fitting.py
├── data/          # xpcs_loader.py, preprocessing.py, phi_filtering.py
├── device/        # cpu.py (v2.3.0: GPU removed)
├── optimization/  # nlsq_wrapper.py, mcmc.py, strategy.py, checkpoint_manager.py
│   ├── cmc/       # coordinator.py, sharding.py, backends/
│   └── initialization/  # per_phi_initializer.py (v2.4.1)
├── runtime/       # shell/, utils/system_validator.py
└── utils/         # Logging, progress

tests/
├── unit/          # Function-level (186+ NLSQ, 95 param, 72 angle filtering)
├── integration/   # End-to-end workflows
├── performance/   # Benchmarks
├── mcmc/          # Statistical validation
└── factories/     # Test data generators
```

______________________________________________________________________

## Known Issues

### NLSQ Optimization

**Issue**: `curve_fit_large()` returns `(popt, pcov)`, not `(popt, pcov, info)`
```python
# ✅ CORRECT:
popt, pcov = curve_fit_large(...)
info = {}  # Empty dict for consistency
```

**Issue**: Silent failure (0 iterations)
- ✅ **RESOLVED in v2.2.1**: Auto parameter expansion, gradient sanity check
- Activates: Dataset ≥ 1M, per-angle scaling, stratified data
- Fix: `homodyne/optimization/nlsq_wrapper.py` lines 500-595, 2506-2559

### MCMC Initialization

**CRITICAL**: Parameter ordering requirement for per-angle scaling
- ✅ **RESOLVED in v2.4.0 (Nov 14, 2025)**: Parameter ordering fixed in coordinator
- **Root Cause**: NumPyro's `init_to_value()` requires parameters in EXACT order as model samples them
- **Symptom**: "Cannot find valid initial parameters" error during MCMC initialization
- **Error Location**: `homodyne/optimization/cmc/backends/multiprocessing.py:179-215`

**Required Parameter Ordering**:
```python
# NumPyro model samples parameters in THIS ORDER:
# 1. Per-angle contrast params: contrast_0, contrast_1, ..., contrast_{n_phi-1}
# 2. Per-angle offset params:   offset_0, offset_1, ..., offset_{n_phi-1}
# 3. Physical parameters:        D0, alpha, D_offset, gamma_dot_t0, ...

# Example for 3 angles (laminar_flow):
correct_order = [
    'contrast_0', 'contrast_1', 'contrast_2',  # Per-angle contrast FIRST
    'offset_0', 'offset_1', 'offset_2',        # Per-angle offset SECOND
    'D0', 'alpha', 'D_offset',                 # Physical params LAST
    'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0'
]
```

**Fix Locations**:
- Coordinator: `homodyne/optimization/cmc/coordinator.py:485-550` (parameter expansion)
- Worker validation: `homodyne/optimization/cmc/backends/multiprocessing.py:875-913` (assertion)
- Model sampling: `homodyne/optimization/mcmc.py:1706-1720` (per-angle iteration)

**Developer Notes**:
- Python dict insertion order matters for NumPyro initialization (Python 3.7+)
- Worker validates parameter ordering at runtime (raises ValueError if wrong)
- If coordinator sends wrong order, NumPyro assigns wrong values (e.g., D0 → contrast_0)
- This causes physics model to produce NaN/inf, triggering initialization failure

### Plotting

**Issue**: Two-time correlation diagonal orientation
```python
# ✅ CORRECT:
ax.imshow(c2.T, origin='lower', extent=[t1[0], t1[-1], t2[0], t2[-1]])
```

### Resources

- Silent failures: `docs/troubleshooting/silent-failure-diagnosis.md`
- NLSQ docs: https://nlsq.readthedocs.io/en/latest/
- CMC-only migration: `docs/migration/v3_cmc_only.md`

______________________________________________________________________

## Dependencies

### Core (v2.4.0 - CPU-only)

- **JAX 0.8.0 + jaxlib 0.8.0** (exact match, CPU-only)
- **nlsq ≥ 0.1.0** - Trust-region optimization
- **NumPyro ≥ 0.18.0, <0.20.0** - MCMC
- **BlackJAX ≥ 1.2.0, <2.0.0** - MCMC backend
- **NumPy ≥ 2.0.0, <3.0.0**
- **SciPy ≥ 1.14.0, <2.0.0**
- **h5py ≥ 3.10.0, <4.0.0**
- **PyYAML ≥ 6.0.2**
- **matplotlib ≥ 3.8.0, <4.0.0**
- **psutil ≥ 6.0.0**

### Platform Support

- **Python**: 3.12+ (all platforms)
- **CPU**: Linux, macOS, Windows (HPC-optimized: 14+ cores)
- **GPU**: Not supported (use v2.2.1)

### Development

- pytest ≥ 8.3.0, black ≥ 25.0.0, ruff ≥ 0.13.0, mypy ≥ 1.18.0

______________________________________________________________________

## Configuration

### Canonical Parameter Names

**Static Isotropic** (3 params):
```python
['D0', 'alpha', 'D_offset']
```

**Laminar Flow** (7 params):
```python
['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta', 'gamma_dot_t_offset', 'phi0']
```

**All Parameters** (with scaling):
```python
['contrast', 'offset', 'D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
 'gamma_dot_t_offset', 'phi0']
```

### Important Notes

- Float32 recommended for MCMC
- Homodyne supports both float32 and float64
- **Don't question initial parameter values/bounds - they are verified and physically correct**
- Per-angle scaling mandatory in v2.4.0
- CPU-only in v2.3.0+ (GPU support removed)
- All C2 heatmap plots now use adaptive color scaling with the conditional logic: vmin = max(1.0, c2_min) # Use data_min if >= 1.0, else clamp to 1.0; vmax = min(1.6, c2_max) # Use data_max if <= 1.6, else clamp to 1.6
- Only two modes, "static" and "laminar_flow", are available in the current codebase. No "static_isotropic" and "static_anisotropic" modes exist in the current codebase.
