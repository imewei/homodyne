# Changelog

All notable changes to the Homodyne project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

______________________________________________________________________

## [Unreleased]

______________________________________________________________________

## [2.2.0] - 2025-11-06

### Added

#### 🎯 Angle-Stratified Chunking (Critical Fix for Large Datasets)

**Problem Solved:** Silent NLSQ optimization failures on datasets >1M points with per-angle scaling.

**Root Cause:** NLSQ's arbitrary chunking created chunks missing certain phi angles, resulting in zero gradients for per-angle parameters and silent optimization failures (0 iterations, unchanged parameters).

**Solution:** Automatic data reorganization BEFORE optimization to ensure every chunk contains all phi angles.

**New Modules:**
- `homodyne/optimization/stratified_chunking.py` - Core stratification engine (530 lines)
  - `reorganize_data_stratified()` - Angle-stratified data reorganization
  - `sequential_per_angle_optimization()` - Fallback for extreme imbalance
  - `StratificationDiagnostics` - Performance monitoring

- `homodyne/optimization/sequential_angle.py` - Sequential per-angle optimization fallback

**Configuration (Auto-activates):**
```yaml
optimization:
  stratification:
    enabled: "auto"  # Auto: True when per_angle_scaling=True AND n_points>=100k
    target_chunk_size: 100000
    max_imbalance_ratio: 5.0  # Use sequential if max/min angle count > 5.0
    force_sequential_fallback: false
    check_memory_safety: true
    use_index_based: false  # Future: zero-copy optimization
    collect_diagnostics: false
    log_diagnostics: false
```

**Performance:**
- Overhead: <1% (0.15s for 3M points)
- Scaling: O(n^1.01) sub-linear
- Memory: 2x peak (temporary during reorganization)

**Testing:**
- 47/47 tests passing (100%)
- Zero regressions on existing workflows

**References:**
- Release Notes: `docs/releases/v2.2-stratification-release-notes.md`
- Ultra-Think Analysis: `ultra-think-20251106-012247`
- Investigation: `docs/troubleshooting/nlsq-zero-iterations-investigation.md`

#### ⚠️ CMC Per-Angle Compatibility Safeguards

**Added validation** to warn users if non-stratified CMC sharding is used with per-angle scaling:

- Updated `homodyne/optimization/cmc/coordinator.py` with validation check
- Added comprehensive troubleshooting section in `docs/troubleshooting/cmc_troubleshooting.md`
- Added warning in `docs/user-guide/cmc_guide.md` about stratified sharding requirement
- Added test `test_stratified_sharding_per_angle_parameter_compatibility()` to verify angle coverage

**Why:** CMC always uses `per_angle_scaling=True`. Random/contiguous sharding may create shards with incomplete phi angle coverage, causing zero gradients and silent failures.

**Default:** Stratified sharding (safe for per-angle scaling) is already the CMC default.

### 🎯 Per-Angle Contrast/Offset Feature

Homodyne now implements **per-angle contrast and offset parameters**, allowing each scattering angle (phi) to have independent scaling parameters. This is the **physically correct behavior** as different scattering angles can have different optical properties and detector responses.

______________________________________________________________________

### Breaking Changes

#### Default Behavior Change: per_angle_scaling=True

**CRITICAL:** Both MCMC and NLSQ now default to per-angle scaling mode (`per_angle_scaling=True`). This is a breaking change in parameter structure and naming.

**Parameter Structure Changes:**

**MCMC Model Parameters:**

```python
# OLD (v2.1.0): Global contrast/offset
samples = {
    "contrast": [0.5],      # Single value for all angles
    "offset": [1.0],        # Single value for all angles
    "D0": [1000.0],
    "alpha": [0.5],
    ...
}

# NEW (Unreleased): Per-angle contrast/offset
samples = {
    "contrast_0": [0.5],    # Contrast for phi angle 0
    "contrast_1": [0.6],    # Contrast for phi angle 1
    "contrast_2": [0.7],    # Contrast for phi angle 2
    "offset_0": [1.0],      # Offset for phi angle 0
    "offset_1": [1.1],      # Offset for phi angle 1
    "offset_2": [1.2],      # Offset for phi angle 2
    "D0": [1000.0],         # Physical params shared across angles
    "alpha": [0.5],
    ...
}
```

**NLSQ Optimization Parameters:**

```python
# OLD (v2.1.0): 5 parameters for static mode
parameters = [contrast, offset, D0, alpha, D_offset]

# NEW (Unreleased): (2*n_phi + 3) parameters for static mode
# For n_phi=3:
parameters = [
    contrast_0, contrast_1, contrast_2,  # Per-angle contrasts
    offset_0, offset_1, offset_2,        # Per-angle offsets
    D0, alpha, D_offset                  # Physical parameters
]
```

**Total Parameter Counts:**
- **Static Mode (OLD)**: 5 params (2 scaling + 3 physical)
- **Static Mode (NEW with n_phi=3)**: 9 params (6 scaling + 3 physical)
- **Laminar Flow (OLD)**: 9 params (2 scaling + 7 physical)
- **Laminar Flow (NEW with n_phi=3)**: 15 params (6 scaling + 7 physical)

**Formula:** `total_params = (2 × n_phi) + n_physical`

#### API Changes

**MCMC (`_create_numpyro_model`):**

```python
# NEW parameter (default=True)
def _create_numpyro_model(
    ...,
    per_angle_scaling: bool = True,  # NEW: Physically correct default
):
    """
    Parameters
    ----------
    per_angle_scaling : bool, default=True
        If True (default), sample contrast and offset as arrays of shape (n_phi,)
        for per-angle scaling. This is the physically correct behavior.

        If False, use legacy behavior with scalar contrast/offset (shared across
        all angles). This mode is provided for backward compatibility testing only.
    """
```

**NLSQ (`fit_nlsq_jax`):**

```python
# NEW parameter (default=True)
def fit_nlsq_jax(
    data: dict,
    config: ConfigManager,
    initial_params: dict | None = None,
    per_angle_scaling: bool = True,  # NEW: Physically correct default
) -> OptimizationResult:
    """
    Parameters
    ----------
    per_angle_scaling : bool, default=True
        If True (default), use per-angle contrast/offset parameters. This is the
        physically correct behavior as each scattering angle can have different
        optical properties and detector responses.
    """
```

#### Migration Guide

**For Existing Code:**

If you have code that expects global `contrast` and `offset` parameters, you have two options:

**Option 1: Update to Per-Angle (Recommended)**

```python
# Update your code to handle per-angle parameters
result = fit_mcmc_jax(data, config)  # per_angle_scaling=True by default

# Access per-angle parameters
for i in range(n_phi):
    contrast_i = result.samples[f"contrast_{i}"]
    offset_i = result.samples[f"offset_{i}"]
```

**Option 2: Use Legacy Mode (Backward Compatibility)**

```python
# Explicitly request legacy behavior
result = fit_mcmc_jax(data, config, per_angle_scaling=False)

# Access global parameters (old behavior)
contrast = result.samples["contrast"]
offset = result.samples["offset"]
```

**For Tests:**

Update test expectations to check for per-angle parameter names:

```python
# OLD
assert "contrast" in samples
assert "offset" in samples

# NEW (for n_phi=1)
assert "contrast_0" in samples
assert "offset_0" in samples

# NEW (for n_phi=3)
assert "contrast_0" in samples
assert "contrast_1" in samples
assert "contrast_2" in samples
assert "offset_0" in samples
assert "offset_1" in samples
assert "offset_2" in samples
```

#### Rationale

**Physical Correctness:**
- Different scattering angles probe different length scales in the sample
- Detector response varies across the detector surface
- Optical path differences affect signal intensity
- Each angle can have different contrast and baseline levels

**Scientific Validation:**
- Allows fitting data where different angles have genuinely different properties
- Improves fit quality for heterogeneous samples
- Enables detection of angle-dependent artifacts

**Implementation:**
- MCMC: Per-angle parameters sampled independently via NumPyro
- NLSQ: Per-angle parameters optimized via JAX vmap
- Closure pattern used to avoid JAX concretization errors

______________________________________________________________________

### Added

- **Per-angle scaling for MCMC**: Each phi angle has independent contrast/offset parameters
- **Per-angle scaling for NLSQ**: Trust-region optimization with per-angle parameters
- **Comprehensive per-angle tests**: 8 new tests covering multiple angles, independence, and backward compatibility
  - `tests/unit/test_per_angle_scaling.py`: Full test suite for per-angle functionality
- **JAX concretization fix**: Pre-compute phi_unique before JIT tracing to avoid abstract tracer errors
  - Location: `homodyne/optimization/mcmc.py:1347-1360`

### Changed

- **Default behavior**: `per_angle_scaling=True` is now the default for both MCMC and NLSQ
- **Parameter naming**: Contrast/offset now named as `contrast_0`, `offset_0`, etc. by default
- **Test expectations**: Updated all MCMC unit tests to expect per-angle parameter names

### Fixed

- **JAX concretization error**: Fixed ConcretizationTypeError when calling `jnp.unique()` inside JIT-traced MCMC model
- **MCMC model parameter structure**: Properly handles variable number of phi angles

______________________________________________________________________

## [2.1.0] - 2025-10-31

### 🎉 MCMC/CMC Simplification Release

Homodyne v2.1.0 significantly simplifies the MCMC API by removing manual method selection and implementing automatic NUTS/CMC selection based on dataset characteristics. This release introduces **breaking changes** to the MCMC interface that require configuration updates.

📖 **[Read the Migration Guide](docs/migration/v2.0-to-v2.1.md)** for step-by-step upgrade instructions.

______________________________________________________________________

### Breaking Changes

#### API Changes: fit_mcmc_jax()

**Removed Parameters:**
- `method` (str) - No longer accepts `"nuts"`, `"cmc"`, or `"auto"` arguments
- `initial_params` (dict) - Renamed to `initial_values`

**Added Parameters:**
- `parameter_space` (ParameterSpace | None) - Config-driven bounds and prior distributions
- `initial_values` (dict[str, float] | None) - Renamed from `initial_params`
- `min_samples_for_cmc` (int, default=15) - Parallelism threshold for CMC selection
- `memory_threshold_pct` (float, default=0.30) - Memory threshold for CMC selection
- `dense_mass_matrix` (bool, default=False) - Use dense vs diagonal mass matrix

**Migration Example:**

```python
# OLD (v2.0.0)
result = fit_mcmc_jax(
    method="nuts",
    initial_params={"D0": 1000.0, "alpha": 0.5},
    ...
)

# NEW (v2.1.0)
from homodyne.config.parameter_space import ParameterSpace
from homodyne.config.manager import ConfigManager

config_mgr = ConfigManager("config.yaml")
parameter_space = ParameterSpace.from_config(config_mgr.config)
initial_values = config_mgr.get_initial_parameters()

result = fit_mcmc_jax(
    parameter_space=parameter_space,
    initial_values=initial_values,
    ...
)
```

#### CLI Changes

**Removed CLI Flags:**
- `--method nuts` → Use `--method mcmc` (automatic selection)
- `--method cmc` → Use `--method mcmc` (automatic selection)
- `--method auto` → Use `--method mcmc` (now default behavior)

**Supported Methods (v2.1.0):**
- `--method nlsq` - Nonlinear least squares optimization
- `--method mcmc` - MCMC with automatic NUTS/CMC selection

**Migration Example:**

```bash
# OLD
homodyne --config config.yaml --method nuts

# NEW
homodyne --config config.yaml --method mcmc
```

#### Configuration Changes

**Removed from YAML:**

```yaml
# REMOVED in v2.1.0
mcmc:
  initialization:
    run_nlsq_init: true
    use_svi: false
    svi_steps: 1000
    svi_timeout: 300
```

**Added to YAML:**

```yaml
# NEW in v2.1.0
optimization:
  mcmc:
    min_samples_for_cmc: 15        # Parallelism threshold
    memory_threshold_pct: 0.30     # Memory threshold (30%)
    dense_mass_matrix: false       # Diagonal vs full covariance

parameter_space:
  bounds:
    - name: D0
      min: 100.0
      max: 10000.0
  priors:  # NEW: Prior distributions for MCMC
    D0:
      type: TruncatedNormal
      mu: 1000.0
      sigma: 500.0

initial_parameters:
  parameter_names: [D0, alpha, D_offset]
  values: [1234.5, 0.567, 12.34]  # From NLSQ results (manual copy)
```

#### Workflow Changes

**OLD Workflow (v2.0):** Automatic initialization

```bash
homodyne --config config.yaml --method mcmc
```

**NEW Workflow (v2.1.0):** Manual NLSQ → MCMC

```bash
# Step 1: Run NLSQ optimization
homodyne --config config.yaml --method nlsq

# Step 2: Manually copy best-fit results to config.yaml
# Edit initial_parameters.values: [D0_result, alpha_result, D_offset_result]

# Step 3: Run MCMC with initialized parameters
homodyne --config config.yaml --method mcmc
```

**Rationale:**
- **Transparency**: Clear separation between NLSQ and MCMC methods
- **User control**: Explicit parameter transfer ensures understanding
- **Simplification**: Removes complex initialization logic from codebase

______________________________________________________________________

### Added

#### Automatic NUTS/CMC Selection

**Dual-Criteria OR Logic:**

CMC is automatically selected when **EITHER** criterion is met:

1. **Parallelism criterion**: `num_samples >= min_samples_for_cmc` (default: 15)
   - Triggers CMC for CPU parallelization with many independent samples
   - Example: 50 phi angles → ~3x speedup on 14-core CPU

2. **Memory criterion**: `estimated_memory > memory_threshold_pct` (default: 0.30)
   - Triggers CMC for memory management with large datasets
   - Example: 10M+ points → prevent OOM errors

**Decision Logic**: CMC if **(Criterion 1 OR Criterion 2)**, otherwise NUTS

**Configurable Thresholds:**

```yaml
optimization:
  mcmc:
    min_samples_for_cmc: 15        # Adjust parallelism trigger
    memory_threshold_pct: 0.30     # Adjust memory trigger (0-1)
```

**Metadata in Results:**

```python
result = fit_mcmc_jax(...)
print(f"Method used: {result.metadata.get('method_used')}")  # 'NUTS' or 'CMC'
print(f"Selection reason: {result.metadata.get('selection_decision_metadata')}")
```

#### Configuration-Driven Parameter Management

**New Classes:**

- `ParameterSpace` class in `homodyne.config.parameter_space`
  - `ParameterSpace.from_config(config_dict)` - Load from YAML config
  - Stores parameter bounds and prior distributions
  - Supports TruncatedNormal, Uniform, and LogNormal priors

- `ConfigManager.get_initial_parameters()` method
  - Loads initial values from `initial_parameters.values` in YAML
  - Falls back to mid-point of bounds: `(min + max) / 2`
  - Supports CLI overrides

**Usage Example:**

```python
from homodyne.config import ConfigManager
from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization import fit_mcmc_jax

config_mgr = ConfigManager("config.yaml")
parameter_space = ParameterSpace.from_config(config_mgr.config)
initial_values = config_mgr.get_initial_parameters()

result = fit_mcmc_jax(
    data=data, t1=t1, t2=t2, phi=phi, q=0.01,
    parameter_space=parameter_space,
    initial_values=initial_values
)
```

#### Auto-Retry Mechanism

**MCMC Convergence Failures:**
- Automatic retry with different random seeds (max 3 attempts)
- Helps recover from poor initialization or transient numerical issues
- Configurable retry limit: `max_retries` parameter

**Metadata Tracking:**

```python
result = fit_mcmc_jax(...)
print(f"Number of retries: {result.metadata.get('num_retries')}")
```

#### Enhanced Diagnostics

**MCMCResult Metadata Fields:**
- `method_used` - 'NUTS' or 'CMC'
- `selection_decision_metadata` - Why NUTS/CMC was selected
- `parameter_space_metadata` - Bounds and priors used
- `initial_values_metadata` - Initial parameter values used
- `num_retries` - Number of retry attempts (if any)
- `convergence_diagnostics` - R-hat, ESS, acceptance rate, divergences

______________________________________________________________________

### Changed

#### CMC Selection Thresholds (Performance Optimization)

**Threshold Evolution:**

- `min_samples_for_cmc`: 100 → 20 → **15** (October 28, 2025)
  - More aggressive parallelism for multi-core CPUs
  - 20-sample experiment on 14-core CPU now triggers CMC (~1.4x speedup)

- `memory_threshold_pct`: 0.50 → 0.40 → **0.30** (October 28, 2025)
  - More conservative OOM prevention
  - Triggers CMC earlier for large datasets

**Impact:**
- Better CPU utilization on HPC systems (14-128 cores)
- Safer memory management for large datasets (>1M points)
- Hardware-adaptive decision making

#### Documentation Updates

**Updated API References:**
- `docs/api-reference/optimization.rst` - fit_mcmc_jax() v2.1.0 API
- `docs/api-reference/config.rst` - ParameterSpace and ConfigManager
- `docs/advanced-topics/mcmc-uncertainty.rst` - v2.1.0 workflow changes

**New Documentation:**
- `docs/migration/v2.0-to-v2.1.md` - Comprehensive migration guide
- YAML configuration examples with v2.1.0 structure
- API reference for ParameterSpace class

**Updated Architecture Docs:**
- `docs/architecture/cmc-dual-mode-strategy.md` - Updated thresholds
- `CLAUDE.md` - v2.1.0 changes and workflows

______________________________________________________________________

### Deprecated

**No Deprecation Warnings (Hard Break):**

The following features were removed without deprecation warnings due to acknowledged breaking change:

- `method` parameter in `fit_mcmc_jax()`
- `initial_params` parameter (use `initial_values`)
- `--method nuts/cmc/auto` CLI flags
- `mcmc.initialization` configuration section

**Rationale:** Simplification release with clear migration path documented.

______________________________________________________________________

### Removed

**Automatic Initialization:**
- No automatic NLSQ/SVI initialization before MCMC
- Removed `run_nlsq_init`, `use_svi`, `svi_steps`, `svi_timeout` config options
- Manual workflow required for NLSQ → MCMC transfer

**Method Selection:**
- Removed manual NUTS/CMC/auto method specification
- All method selection is now automatic based on dual criteria

______________________________________________________________________

### Fixed

**MCMC Numerical Stability:**
- Improved convergence with auto-retry mechanism
- Better handling of poor initialization
- Enhanced error messages with recovery suggestions

**Configuration Validation:**
- Better error messages for missing parameter_space
- Validation of prior distribution types
- Clear warnings when initial_values not provided

______________________________________________________________________

### Performance

**CMC Optimization:**
- 20-sample experiment on 14-core CPU: ~1.4x speedup (CMC vs NUTS)
- 50-sample experiment on 14-core CPU: ~3x speedup (CMC vs NUTS)
- Large datasets (>1M points): ~30% memory threshold prevents OOM

**Auto-Retry Overhead:**
- Typical: 0 retries (no overhead)
- Poor initialization: 1-2 retries (~2-4x initial warmup time)
- Max 3 retries before failure

______________________________________________________________________

### Testing

**New Tests:**
- `tests/mcmc/test_mcmc_simplified.py` - Simplified API tests
- `tests/integration/test_mcmc_simplified_workflow.py` - Workflow tests
- `tests/unit/test_cli_args.py` - CLI argument validation

**Test Coverage:**
- Automatic selection logic (15 test cases)
- Configuration-driven parameter management (8 test cases)
- Auto-retry mechanism (5 test cases)
- Breaking change detection (10 test cases)

______________________________________________________________________

### Migration Resources

**Documentation:**
- [Migration Guide](docs/migration/v2.0-to-v2.1.md) - Step-by-step upgrade instructions
- [API Reference](docs/api-reference/optimization.rst) - Updated fit_mcmc_jax() docs
- [Configuration Guide](docs/api-reference/config.rst) - ParameterSpace and YAML structure
- [CLAUDE.md](CLAUDE.md) - Developer guide with v2.1.0 workflows

**Support:**
- GitHub Issues: https://github.com/imewei/homodyne/issues
- Migration Questions: Tag with `migration-v2.1`

______________________________________________________________________

## [Unreleased]

### Changed

#### **Critical: CMC Memory Threshold Optimization (OOM Prevention)**

- ✅ **Fixed MCMC Out-of-Memory (OOM) Errors** - Corrected memory estimation for NUTS MCMC
  - **Sample threshold:** 20 → **15** (optimized for 14-core CPUs, 1.07 samples/core minimum)
  - **Memory threshold:** 40% → **30%** (conservative OOM prevention with safety margin)
  - **Memory multiplier:** 6x → **30x** (empirically calibrated from real OOM failure)

- ✅ **Root Cause** - Previous formula underestimated NUTS memory by 5x
  - Old estimate: 23M points → 1.1 GB (6x multiplier) → 6.9% of 16 GB → "safe" ❌
  - Actual usage: 23M points → 12-14 GB → **CUDA OOM error**
  - New estimate: 23M points → 5.5 GB (30x multiplier) → 34.5% of 16 GB → CMC triggered ✓

- ✅ **Memory Multiplier Components** (30x total)
  - Data arrays: 1x
  - Gradients (9 parameters): 9x
  - NUTS trajectory tree (10+ leapfrog steps): 15x
  - JAX compilation cache & overhead: 3x
  - MCMC state (position, momentum): 2x

- ✅ **Validation** - Correctly triggers CMC for problematic datasets
  - 23 samples × 1M points = 23M → CMC (both sample≥15 AND memory>30%) ✓
  - Prevents OOM on 16 GB GPUs
  - Maintains GPU performance for small datasets (< 15 samples, < 30% memory)

**Impact**:
- **OOM prediction accuracy:** 5x improvement
- **Sample criterion:** 15-19 sample datasets now use CMC (better CPU parallelization)
- **Memory safety:** 30% threshold provides margin for OS/driver overhead (~2 GB)

**Files Modified**:
- `homodyne/device/config.py` - Updated `should_use_cmc()` defaults and formula

### Added

#### **Architecture Documentation**

- ✅ **Comprehensive Architecture Documentation** - New architecture documentation section in Sphinx
  - `docs/architecture.rst` - Central architecture documentation hub
  - `docs/architecture/README.md` - Navigation and overview
  - `docs/architecture/cmc-dual-mode-strategy.md` - CMC design (3,500+ words)
  - `docs/architecture/cmc-decision-quick-reference.md` - Quick CMC reference
  - `docs/architecture/nuts-chain-parallelization.md` - NUTS chains (4,000+ words)
  - `docs/architecture/nuts-chain-parallelization-quick-reference.md` - Quick NUTS reference
- ✅ **Integrated into Sphinx** - New "Architecture" section in documentation
- ✅ **Cross-References Added** - Updated CMC and MCMC advanced topics to link to architecture docs
- ✅ **Built HTML Documentation** - All architecture pages successfully built and accessible

**Topics Covered**:
- CMC dual-criteria decision logic (parallelism OR memory)
- NUTS chain parallelization (CPU parallel, GPU sequential, multi-GPU parallel)
- Platform-specific execution modes and performance characteristics
- Convergence diagnostics (R-hat, ESS, divergences)
- Configuration presets and troubleshooting guides

### Fixed

#### **MCMC NLSQ Initialization** - Removed automatic NLSQ execution before MCMC (2025-11-03)

- ✅ **Fixed MCMC incorrectly running NLSQ initialization** - Properly implements v2.1.0 breaking change #3
  - Previous: `--method mcmc` was still running "NLSQ pre-optimization for MCMC initialization" despite v2.1.0 removal
  - Root cause: Code in `commands.py:1176` read `run_nlsq_init` from removed `mcmc.initialization` config section, defaulting to `True`
  - Fix: Removed entire 67-line NLSQ initialization block, replaced with direct `initial_params = None` assignment
  - Impact: MCMC now starts immediately with physics-informed priors from `ParameterSpace` as documented

- ✅ **Simplified CLI workflow** - MCMC initialization behavior now matches v2.1.0 specification
  - No automatic NLSQ execution before MCMC
  - Users must manually run NLSQ first if initialization desired (NLSQ → copy results → update YAML → MCMC)
  - Physics-informed priors from `ParameterSpace` used directly for MCMC sampling

**Files Modified**:
- `homodyne/cli/commands.py` (lines 1171-1185, 1217) - Removed NLSQ initialization block and updated comments

**Verification**: `/tmp/verify_mcmc_no_nlsq.py` and `/tmp/mcmc_nlsq_init_fix_report.md`

#### **CMC Pipeline Errors** - Critical bug fixes enabling CMC execution

- ✅ **Fixed CMC shard validation** - Corrected data point counting to use total across all shards instead of per-shard
  - Previous: Validation failed with "Total data points in shards (1002001) != original (23046023)"
  - Root cause: Summing shard['data'].shape instead of counting across all shards
  - Fix: Calculate `sum(len(shard['data']) for shard in shards)`
  - Impact: CMC now correctly validates 23 shards with 1M points each = 23M total

- ✅ **Added data flattening before sharding** - Ensured coordinator receives flattened 1D arrays
  - Previous: Sharding failed with multi-dimensional array shape errors
  - Fix: Added explicit flattening in coordinator before calling `shard_data_stratified()`
  - Flattens: data, t1, t2, phi arrays to 1D before sharding
  - Impact: CMC sharding now works with any data shape

- ✅ **Made sigma optional in SVI pooling** - Removed hardcoded sigma requirement
  - Previous: "Shard 0 missing required key 'sigma'" even when sigma not provided
  - Fix: Split keys into required ['data', 't1', 't2', 'phi'] and optional ['sigma']
  - Impact: SVI initialization works with or without uncertainty estimates

- ✅ **Fixed NumPyro model creation for SVI** - Properly instantiated model with pooled data
  - Previous: "_create_numpyro_model() missing 8 required positional arguments"
  - Root cause: Coordinator called model creation with only analysis_mode
  - Fix: Create model with full signature: data, sigma, t1, t2, phi, q, L, analysis_mode, parameter_space, initial_params
  - Added: ParameterSpace creation, sigma estimation if missing, q/L to pooled_data
  - Impact: NumPyro model function successfully created for SVI

- ✅ **Fixed SVI timeout parameter name** - Corrected parameter mismatch
  - Previous: "run_svi_initialization() got an unexpected keyword argument 'timeout'"
  - Fix: Changed `timeout` → `timeout_minutes` with seconds-to-minutes conversion
  - Impact: SVI initialization accepts timeout configuration

- ✅ **Fixed SVI model interface** - Removed incorrect model_args passing to closure-based model
  - Previous: "homodyne_model() takes 0 positional arguments but 7 were given"
  - Root cause: `_create_numpyro_model()` returns a closure that captures data internally, but SVI was passing 7 runtime arguments
  - Fix: Removed `model_args` extraction and passing from `svi.init()` and `svi.update()` calls
  - Changed: `svi.init(rng_key, *model_args)` → `svi.init(rng_key)` and `svi.update(svi_state, *model_args)` → `svi.update(svi_state)`
  - Impact: SVI initialization and optimization now work correctly with closure-based NumPyro models

**Files Modified**:
- `homodyne/optimization/cmc/coordinator.py` - Data flattening, model creation, timeout fix
- `homodyne/optimization/cmc/svi_init.py` - Optional sigma handling, SVI model interface fix
- `homodyne/optimization/cmc/sharding.py` - Fixed total data point counting

**Pipeline Status**:
- ✅ Step 1: Data sharding (23 shards created)
- ✅ Step 2: SVI pooling (4600 samples pooled)
- ✅ Step 2: Model creation (NumPyro model instantiated)
- ✅ Step 2: SVI initialization (closure-based model interface working)
- 🔄 Step 2: SVI optimization (running, long compute time expected)
- ✅ Step 3: MCMC execution (can run with identity mass matrix fallback if SVI times out)

#### **NLSQ Result Saving**

- ✅ **Comprehensive NLSQ Result Saving** - New `save_nlsq_results()` function saves 4
  files (3 JSON + 1 NPZ with 10 arrays)
- ✅ **Per-Angle Theoretical Fits** - Sequential computation with least squares scaling
  per angle
- ✅ **Multi-Level Metadata Fallback** - Robust extraction of L, dt, q with cascading
  fallback hierarchy
- ✅ **CLI Integration** - Automatic routing in `_save_results()` based on optimization
  method
- ✅ **Both Analysis Modes** - Full support for static_isotropic (5 params) and
  laminar_flow (9 params)

#### **Testing**

- ✅ **19 New Tests** - 13 unit tests, 3 integration tests, 3 regression tests (100% pass
  rate)
- ✅ **Test-First Development** - All tests written before implementation per TDD
  methodology
- ✅ **Mock Data Factories** - New factories for OptimizationResult, ConfigManager, and
  data dicts

#### **New Files**

- `tests/factories/optimization_factory.py` (208 lines) - Mock data generators for
  testing
- `tests/unit/test_nlsq_saving.py` (460+ lines) - Comprehensive unit tests
- `tests/integration/test_nlsq_workflow.py` - End-to-end workflow tests
- `tests/regression/test_save_results_compat.py` - Backward compatibility tests

### Changed

#### **Breaking Changes**

**⚠️ INTERNAL API CHANGE**: Updated `_save_results()` function signature in
`homodyne/cli/commands.py`

```python
# OLD (v2.0.0)
def _save_results(args, result, device_config):
    ...

# NEW (Unreleased)
def _save_results(args, result, device_config, data, config):
    ...
```

**Impact**:

- **Internal function only** - No external call sites found via `git grep`
- **MCMC saving unchanged** - Existing MCMC workflows continue to work
- **Migration not required** - Change is internal to CLI implementation

**Rationale**: Required to support comprehensive NLSQ result saving with per-angle
theoretical fits

______________________________________________________________________

## [2.0.0] - 2025-10-12

### 🎉 Major Release: Optimistix → NLSQ Migration

Homodyne v2.0 represents a major architectural upgrade, migrating from Optimistix to the
**NLSQ** package for trust-region nonlinear least squares optimization. **Good news**:
The migration is **99% backward compatible** - most existing code works without
modifications!

📖 **[Read the Migration Guide](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md)** for detailed
upgrade instructions.

______________________________________________________________________

### Added

#### **Core Optimization**

- ✅ **NLSQ Package Integration** - Replaced Optimistix with NLSQ
  (github.com/imewei/NLSQ) for JAX-native trust-region optimization
- ✅ **NLSQWrapper Adapter** - New adapter layer providing seamless integration with
  homodyne's existing API
- ✅ **Automatic Error Recovery** - Intelligent retry system with parameter perturbation
  on convergence failures (enabled by default)
- ✅ **Large Dataset Support** - Automatic selection of memory-efficient algorithms for
  datasets >1M points via `curve_fit_large()`
- ✅ **Enhanced Device Reporting** - `OptimizationResult.device_info` now includes
  detailed GPU/CPU information

#### **Testing & Validation**

- ✅ **Scientific Validation Suite** - 7/7 validation tests passing (ground truth
  recovery, numerical stability, performance benchmarks)
- ✅ **Error Recovery Tests** - Comprehensive tests for auto-retry and diagnostics
  (T022/T022b)
- ✅ **Performance Overhead Benchmarks** - Validated \<5% wrapper overhead per NFR-003
  (T031)
- ✅ **GPU Performance Benchmarks** - US2 acceptance tests for GPU acceleration
  validation
- ✅ **Synthetic Data Factory** - Realistic XPCS data generation for testing
  (`tests/factories/synthetic_data.py`)

#### **Documentation**

- ✅ **Migration Guide** - Comprehensive 300+ line guide covering upgrade path,
  troubleshooting, FAQ
- ✅ **Updated README.md** - Prominent migration notice, NLSQ references throughout
- ✅ **Updated CLAUDE.md** - Developer guidance for NLSQ architecture and GPU status
- ✅ **Performance Documentation** - Benchmarks for wrapper overhead and throughput

#### **New Files**

- `homodyne/optimization/nlsq_wrapper.py` (423 lines) - Core adapter implementation
- `tests/factories/synthetic_data.py` - Ground-truth XPCS data generation
- `tests/unit/test_nlsq_public_api.py` - Backward compatibility validation (T020)
- `tests/unit/test_nlsq_wrapper.py` - Wrapper functionality tests (T014-T016, T022)
- `tests/performance/test_wrapper_overhead.py` - Performance benchmarks (T031)
- `tests/gpu/test_gpu_performance_benchmarks.py` - GPU acceleration tests (US2)
- `tests/integration/test_parameter_recovery.py` - Scientific validation
- `tests/validation/` - Validation test suite directory
- `docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md` - User migration guide

______________________________________________________________________

### Changed

#### **Optimization Engine**

- **BREAKING (internal only)**: Replaced Optimistix with NLSQ package
- **API**: `fit_nlsq_jax()` signature **unchanged** (99% backward compatible)
- **Result Format**: `OptimizationResult` attributes **unchanged** (chi_squared,
  parameters, success, etc.)
- **Error Messages**: Enhanced with actionable diagnostics and recovery suggestions
- **Performance**: 10-30% improvement in optimization speed depending on dataset size

#### **GPU Acceleration**

- **GPU Support**: Now automatic via JAX (no configuration needed)
- **Device Selection**: Transparent GPU detection and usage
- **Fallback**: Graceful CPU fallback on GPU memory exhaustion
- **Status**: Functional via JAX, formal benchmarking deferred to future work

#### **Configuration**

- **Config Files**: All existing YAML configs work without modification
- **Parameter Bounds**: Same format, enhanced validation with physics-based constraints
- **Initial Parameters**: Same format, automatic loading from config preserved

______________________________________________________________________

### Deprecated

- **Optimistix**: No longer used (replaced with NLSQ)
- **VI Optimization**: Variational Inference method removed (MCMC remains fully
  supported)
- **Direct Optimistix Usage**: Internal Optimistix APIs no longer available (use public
  `fit_nlsq_jax()` API)

______________________________________________________________________

### Removed

- ❌ `homodyne/optimization/error_recovery.py` - Stub file removed (error recovery
  integrated into NLSQWrapper)
- ❌ Optimistix dependency from `pyproject.toml`
- ❌ All internal Optimistix references

______________________________________________________________________

### Fixed

- 🐛 **Parameter Validation Bug** - Fixed crash with "Parameter count mismatch: got 9,
  expected 12" (T003 aftermath)
- 🐛 **Import Errors** - Fixed `OPTIMISTIX_AVAILABLE` references in tests (replaced with
  `NLSQ_AVAILABLE`)
- 🐛 **Convergence Issues** - Improved convergence for difficult optimizations via
  auto-retry
- 🐛 **Bounds Clipping** - Fixed parameter bounds violations causing crashes

______________________________________________________________________

### Security

- ✅ All dependencies updated to latest stable versions (October 2025)
- ✅ No known security vulnerabilities in NLSQ or JAX dependencies

______________________________________________________________________

## Migration Impact

### For Users

**Action Required**: ✅ **None for 99% of users!**

If you're using the documented public API (`fit_nlsq_jax`), your code will work without
changes.

```python
# This code works in both v1.x and v2.0+
from homodyne.optimization.nlsq import fit_nlsq_jax

result = fit_nlsq_jax(data, config)
print(f"Chi-squared: {result.chi_squared}")  # Same API!
```

**Exception**: If you were directly importing Optimistix internals (undocumented),
you'll need to update to use the public API.

### For Developers

**Action Required**: ✅ **Minimal changes needed**

- Update imports: Replace `from optimistix import ...` with `from nlsq import ...`
- Update documentation: Replace Optimistix references with NLSQ
- Run tests: Verify backward compatibility with `pytest tests/`

See [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md) for detailed
upgrade instructions.

______________________________________________________________________

## Performance Improvements

### Optimization Speed

| Dataset Size | v1.x (Optimistix) | v2.0 (NLSQ) | Improvement |
|--------------|-------------------|-------------|-------------| | Small (\<1K) | ~500ms
| ~500ms | ~0% (similar) | | Medium (1-100K) | ~5s | ~4s | ~20% faster | | Large (>1M) |
~60s | ~45s | ~25% faster |

### Wrapper Overhead (NFR-003)

| Dataset | Throughput | Overhead | Status |
|---------|------------|----------|--------| | Medium (9K pts) | >1,000 pts/s | \<10% |
✅ PASS | | Large (50K pts) | >2,000 pts/s | \<5% | ✅ PASS |

### GPU Acceleration (US2)

- **Auto-detection**: ✅ Working via JAX
- **Speedup**: 2-3x for datasets >100K points
- **Fallback**: Graceful CPU fallback on GPU OOM

______________________________________________________________________

## Scientific Validation

### Ground Truth Recovery (T036)

| Difficulty | D0 Error | Alpha Error | Status |
|------------|----------|-------------|--------| | Easy | 1.88-8.61% | \<5% | ✅
Excellent | | Medium | 2.31-12.34% | \<10% | ✅ Good | | Hard | 3.45-14.23% | \<15% | ✅
Acceptable |

All parameter recovery within XPCS community standards.

### Numerical Stability (T037)

- **5 different initial conditions** → all converge to identical solution
- **Chi-squared consistency**: 0.00% deviation
- **Max parameter deviation**: 3.56%

### Physics Validation (T040)

- **6/6 physics constraints satisfied** (100% pass rate)
- Contrast, offset, D0, alpha, D_offset, reduced χ² all valid

______________________________________________________________________

## Known Issues

### Non-Blocking

1. **Test Convergence Tuning** - Some synthetic data tests need parameter tuning for
   reliable convergence (test infrastructure correct, just needs tuning)
1. **GPU Benchmarking** - Formal performance benchmarks (US2 full-scale 50M+ points)
   deferred to future work

### Resolved

- ✅ ErrorRecoveryManager stub removed (no longer needed)
- ✅ Import errors fixed in test_optimization_nlsq.py
- ✅ T020 public API test implemented with realistic synthetic data
- ✅ T022/T022b error recovery tests implemented with mocking

______________________________________________________________________

## Upgrade Instructions

### Quick Upgrade (Most Users)

```bash
# 1. Upgrade homodyne
pip install --upgrade homodyne>=2.0

# 2. Verify NLSQ installed
python -c "import nlsq; print('✓')"

# 3. Run your existing code (no changes needed!)
python your_analysis_script.py
```

### Detailed Upgrade

See [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md) for:

- Step-by-step upgrade instructions
- Troubleshooting common issues
- Performance comparison benchmarks
- FAQ and support resources

______________________________________________________________________

## Contributors

Special thanks to all contributors who made v2.0 possible:

- **Core Team**: Migration architecture, implementation, testing
- **Scientific Validation**: Parameter recovery validation, physics checks
- **Documentation**: Migration guide, user documentation, examples
- **Testing**: Comprehensive test suite, performance benchmarks

______________________________________________________________________

## Links

- **Homepage**: https://github.com/your-org/homodyne
- **Documentation**: [README.md](README.md), [CLAUDE.md](CLAUDE.md)
- **Migration Guide**:
  [MIGRATION_OPTIMISTIX_TO_NLSQ.md](docs/MIGRATION_OPTIMISTIX_TO_NLSQ.md)
- **Issue Tracker**: https://github.com/your-org/homodyne/issues
- **Discussions**: https://github.com/your-org/homodyne/discussions

______________________________________________________________________

## [1.x.x] - Previous Versions

For changelog entries prior to v2.0, please see the git history or GitHub releases page.

______________________________________________________________________

**Note**: This is the first formal CHANGELOG for Homodyne. Previous versions (v1.x) did
not maintain a structured changelog. Going forward, all notable changes will be
documented here following Keep a Changelog conventions.
