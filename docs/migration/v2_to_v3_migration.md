# Migration Guide: Homodyne v2.0 → v3.0

**From:** v2.0.0 (NLSQ Native Large Dataset Handling)
**To:** v3.0.0 (NLSQ API Alignment with StreamingOptimizer)
**Date:** October 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Breaking Changes Summary](#breaking-changes-summary)
3. [Configuration Migration](#configuration-migration)
4. [Code Migration Examples](#code-migration-examples)
5. [Removed Features](#removed-features)
6. [New Features](#new-features)
7. [Upgrade Checklist](#upgrade-checklist)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Homodyne v3.0 introduces NLSQ `StreamingOptimizer` support for unlimited dataset sizes with enhanced fault tolerance and checkpoint/resume capabilities. The migration primarily involves removing deprecated configuration sections and adopting new streaming features.

**Migration Difficulty:** LOW - Most users only need to remove deprecated config sections.

**Timeline Estimate:**
- **Simple configs:** 5-10 minutes (remove `subsampling` section)
- **Advanced users:** 30-60 minutes (adopt streaming features)
- **Custom workflows:** 1-2 hours (update Python API usage)

**Backward Compatibility:** Results are 100% backward compatible - no changes to physics or optimization quality.

---

## Breaking Changes Summary

### 1. Removed: `performance.subsampling` Configuration Section

**Status:** ❌ REMOVED in v3.0
**Impact:** HIGH (affects configs using subsampling)
**Migration:** Delete section (NLSQ now handles large datasets natively)

**Reason for Removal:** NLSQ v0.1.5 provides native large dataset handling through `curve_fit_large` and `StreamingOptimizer`, eliminating the need for custom subsampling logic.

### 2. Removed: `optimization_performance.time_subsampling` (Legacy)

**Status:** ❌ REMOVED in v2.1, confirmed removed in v3.0
**Impact:** LOW (deprecated since v2.1)
**Migration:** Delete if still present

### 3. StreamingOptimizer Required for > 100M Points

**Status:** ⚠️ NEW REQUIREMENT in v3.0
**Impact:** MEDIUM (automatic for most users)
**Migration:** Enable checkpoints for fault tolerance (optional but recommended)

### 4. New: `streaming_diagnostics` Field in OptimizationResult

**Status:** ✅ ADDED in v3.0
**Impact:** LOW (additive, backward compatible)
**Migration:** Optional - use if you need batch-level diagnostics

---

## Configuration Migration

### Step 1: Remove Deprecated `performance.subsampling`

#### Before (v2.0):

```yaml
# ❌ OLD v2.0 configuration
performance:
  subsampling:
    enabled: true
    max_time_points: 100000
    strategy: "uniform"
    seed: 42
```

#### After (v3.0):

```yaml
# ✅ NEW v3.0 configuration
# Simply delete the entire 'subsampling' section
# NLSQ handles large datasets natively

performance:
  # Optional: Strategy override (rarely needed)
  strategy_override: null  # Let homodyne choose automatically

  # Optional: Custom memory limit
  memory_limit_gb: null  # Auto-detect available memory

  # Optional: Progress bars
  enable_progress: true  # Show progress for long optimizations
```

**What Changed:**
- Homodyne now processes **100% of data** using NLSQ's native strategies
- Strategy selection is automatic based on dataset size
- No data loss from subsampling

### Step 2: Add Streaming Configuration (Optional, Recommended for > 100M Points)

#### New in v3.0:

```yaml
# ✅ NEW: Streaming configuration for large datasets
optimization:
  streaming:
    # Checkpoints (recommended for long optimizations)
    enable_checkpoints: true
    checkpoint_dir: "./checkpoints"
    checkpoint_frequency: 10  # Save every 10 batches
    resume_from_checkpoint: true
    keep_last_checkpoints: 3

    # Fault tolerance (recommended for production)
    enable_fault_tolerance: true
    validate_numerics: true
    min_success_rate: 0.5  # Minimum 50% batch success
    max_retries_per_batch: 2
```

**When to Add:**
- Datasets > 100M points (STREAMING mode)
- Long-running optimizations (> 30 minutes)
- Production pipelines requiring robustness
- HPC environments with job time limits

### Step 3: Update Strategy Override (If Used)

#### Before (v2.0):

```yaml
# ❌ OLD: Manual strategy selection
performance:
  optimization_strategy: "large_dataset"  # Old name
```

#### After (v3.0):

```yaml
# ✅ NEW: Strategy override with new names
performance:
  strategy_override: "streaming"  # 'standard' | 'large' | 'chunked' | 'streaming'
```

**Strategy Names Changed:**
- `standard` → `standard` (unchanged)
- `large_dataset` → `large` (renamed)
- `chunked_processing` → `chunked` (renamed)
- NEW: `streaming` (for > 100M points)

---

## Code Migration Examples

### Example 1: Basic Configuration File

#### Before (v2.0):

```yaml
# homodyne_config_v2.yaml
experimental_data:
  file_path: "./data/large_experiment.hdf"

initial_parameters:
  parameter_names: ["D0", "alpha", "D_offset"]
  values: [1000.0, -1.2, 0.0]

parameter_space:
  bounds:
    - name: D0
      min: 1.0
      max: 100000.0
    # ... other bounds

# ❌ OLD: Remove this section
performance:
  subsampling:
    enabled: true
    max_time_points: 100000
    strategy: "uniform"
```

#### After (v3.0):

```yaml
# homodyne_config_v3.yaml
experimental_data:
  file_path: "./data/large_experiment.hdf"

initial_parameters:
  parameter_names: ["D0", "alpha", "D_offset"]
  values: [1000.0, -1.2, 0.0]

parameter_space:
  bounds:
    - name: D0
      min: 1.0
      max: 100000.0
    # ... other bounds

# ✅ NEW: Optional performance tuning
performance:
  enable_progress: true  # Show progress bars

# ✅ NEW: Optional streaming configuration (for > 100M points)
optimization:
  streaming:
    enable_checkpoints: true
    checkpoint_dir: "./checkpoints"
    checkpoint_frequency: 10
```

### Example 2: Python API Migration

#### Before (v2.0):

```python
# ❌ OLD v2.0 code
from homodyne.optimization.nlsq_wrapper import NLSQWrapper

wrapper = NLSQWrapper(
    enable_large_dataset=True,
    enable_recovery=True,
)

result = wrapper.fit(
    model_func=model,
    xdata=xdata,
    ydata=ydata,
    p0=p0,
    bounds=bounds,
)

# Access results
print(f"Parameters: {result.parameters}")
print(f"Chi-squared: {result.chi_squared}")
# No streaming diagnostics available in v2.0
```

#### After (v3.0):

```python
# ✅ NEW v3.0 code
from homodyne.optimization.nlsq_wrapper import NLSQWrapper

# Optional: Enable fast mode for production
wrapper = NLSQWrapper(
    enable_large_dataset=True,
    enable_recovery=True,
    fast_mode=False,  # NEW: Disable for development
)

# Optional: Pass streaming config for > 100M points
config = {
    'optimization': {
        'streaming': {
            'enable_checkpoints': True,
            'checkpoint_dir': './checkpoints',
        }
    }
}

result = wrapper.fit(
    model_func=model,
    xdata=xdata,
    ydata=ydata,
    p0=p0,
    bounds=bounds,
    config=config,  # NEW: Optional streaming config
)

# Access results (backward compatible)
print(f"Parameters: {result.parameters}")
print(f"Chi-squared: {result.chi_squared}")

# NEW: Access streaming diagnostics (if STREAMING mode)
if result.streaming_diagnostics:
    diag = result.streaming_diagnostics
    print(f"Batch success rate: {diag['batch_success_rate']:.1%}")
    print(f"Failed batches: {diag['failed_batch_indices']}")
    print(f"Error distribution: {diag['error_type_distribution']}")
```

### Example 3: Custom Strategy Selection

#### Before (v2.0):

```python
# ❌ OLD v2.0: No custom strategy selection API
# Users had to rely on automatic selection only
```

#### After (v3.0):

```python
# ✅ NEW v3.0: Custom strategy selection
from homodyne.optimization.strategy import DatasetSizeStrategy

selector = DatasetSizeStrategy()

# Automatic selection (recommended)
strategy = selector.select_strategy(
    n_points=50_000_000,
    n_parameters=5,
)
print(strategy)  # OptimizationStrategy.CHUNKED

# Manual override (advanced)
strategy = selector.select_strategy(
    n_points=50_000_000,
    n_parameters=5,
    strategy_override='streaming',  # Force STREAMING mode
)
print(strategy)  # OptimizationStrategy.STREAMING

# Build optimized streaming config
config = selector.build_streaming_config(
    n_points=200_000_000,
    n_parameters=9,
    checkpoint_config={
        'enable_checkpoints': True,
        'checkpoint_dir': './checkpoints',
    },
)
print(f"Optimal batch size: {config['batch_size']}")
```

### Example 4: Checkpoint Management (New in v3.0)

```python
# ✅ NEW v3.0: Checkpoint management API
from homodyne.optimization.checkpoint_manager import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="./checkpoints",
    checkpoint_frequency=10,
    keep_last_n=3,
)

# Find and resume from latest checkpoint
latest = manager.find_latest_checkpoint()
if latest:
    data = manager.load_checkpoint(latest)
    print(f"Resuming from batch {data['batch_idx']}")
    initial_params = data['parameters']
else:
    print("No checkpoint found, starting fresh")
    initial_params = p0

# ... run optimization ...

# Cleanup old checkpoints
deleted = manager.cleanup_old_checkpoints()
print(f"Deleted {len(deleted)} old checkpoints")
```

---

## Removed Features

### 1. Custom Subsampling Logic

**Removed:** All subsampling-related code in `homodyne.optimization`

**Reason:** NLSQ v0.1.5 provides superior native large dataset handling

**Migration:** No action needed - NLSQ handles this automatically

**What You Gain:**
- 100% data utilization (no sampling)
- Better convergence (full dataset)
- Simpler configuration (no subsampling parameters)

### 2. Legacy `enable_sampling` Parameter

**Removed:** `enable_sampling` parameter in optimization functions

**Reason:** Deprecated in v2.1, fully removed in v3.0

**Migration:** Remove any references to `enable_sampling`

#### Before:
```python
# ❌ OLD
result = wrapper.fit(..., enable_sampling=True)  # Deprecated
```

#### After:
```python
# ✅ NEW
result = wrapper.fit(...)  # Sampling is automatic
```

### 3. Subsampling Seed Configuration

**Removed:** `performance.subsampling.seed` configuration

**Reason:** No longer needed without subsampling

**Migration:** Remove from config files

---

## New Features

### 1. StreamingOptimizer Support

**What's New:** Process unlimited dataset sizes with constant memory

**Benefits:**
- Datasets > 100M points supported
- Constant memory usage (~2 GB regardless of size)
- Checkpoint/resume for long optimizations
- Batch-level fault tolerance

**How to Use:**
```yaml
optimization:
  streaming:
    enable_checkpoints: true
    checkpoint_dir: "./checkpoints"
```

See: `/docs/guides/streaming_optimizer_usage.md`

### 2. Enhanced Diagnostics

**What's New:** `streaming_diagnostics` field in OptimizationResult

**Provides:**
- Batch success rate
- Failed batch indices
- Error type distribution
- Average iterations per batch

**How to Access:**
```python
if result.streaming_diagnostics:
    print(f"Success rate: {result.streaming_diagnostics['batch_success_rate']}")
```

### 3. Fast Mode

**What's New:** `fast_mode` parameter for < 1% overhead

**Benefits:**
- Minimal performance impact in production
- Skips non-essential numerical validation
- Preserves error recovery

**How to Use:**
```python
wrapper = NLSQWrapper(fast_mode=True)
```

**Recommendation:** Use in production after development/testing phase

### 4. Automatic Batch Size Optimization

**What's New:** Intelligent batch sizing based on available memory

**Benefits:**
- No manual batch size tuning
- Optimal memory usage (10% of available RAM)
- Bounded (1k-100k points) for safety

**How it Works:**
```python
# Automatic (recommended)
config = selector.build_streaming_config(
    n_points=200_000_000,
    n_parameters=9,
)
# Calculates optimal batch_size based on RAM and Jacobian size
```

### 5. Checkpoint/Resume Capability

**What's New:** HDF5-based checkpoints for fault tolerance

**Features:**
- Save/resume optimization state
- Checksum validation for integrity
- Automatic cleanup (keep last N)
- Target save time < 2 seconds

**How to Use:**
```yaml
optimization:
  streaming:
    enable_checkpoints: true
    checkpoint_dir: "./checkpoints"
    checkpoint_frequency: 10
    resume_from_checkpoint: true
```

### 6. Numerical Validation at 3 Critical Points

**What's New:** Detect NaN/Inf early at gradient, parameter, and loss calculation points

**Benefits:**
- Early error detection
- Targeted recovery strategies
- Can disable via `fast_mode`

**How it Works:** Automatic - just enable recovery:
```python
wrapper = NLSQWrapper(enable_numerical_validation=True)
```

---

## Upgrade Checklist

### Pre-Upgrade

- [ ] **Backup current configuration files** (`cp config.yaml config_v2_backup.yaml`)
- [ ] **Document current workflow** (note any custom scripts)
- [ ] **Check NLSQ version** (`pip show nlsq` → should be >= 0.1.5)
- [ ] **Review breaking changes** (see above)

### During Upgrade

- [ ] **Update homodyne** (`pip install --upgrade homodyne`)
- [ ] **Verify installation** (`homodyne --version` → should show 3.0.0+)
- [ ] **Update configuration files:**
  - [ ] Remove `performance.subsampling` section
  - [ ] Remove `optimization_performance.time_subsampling` if present
  - [ ] Add `optimization.streaming` section (optional, for > 100M points)
- [ ] **Update Python scripts** (if using Python API):
  - [ ] Remove `enable_sampling` parameters
  - [ ] Add `config` parameter for streaming settings
  - [ ] Update to new strategy names if using override
- [ ] **Test with small dataset first** (validate before running large jobs)

### Post-Upgrade Validation

- [ ] **Run test optimization** on a known dataset
- [ ] **Compare results** with v2.0 (should be identical within numerical precision)
- [ ] **Check for deprecation warnings** in logs
- [ ] **Verify checkpoint functionality** (if enabled)
- [ ] **Validate streaming mode** (if using > 100M points)
- [ ] **Update documentation** for your team/workflow

### Recommended Actions

- [ ] **Enable streaming checkpoints** for production workflows
- [ ] **Adopt fast mode** for validated pipelines
- [ ] **Monitor batch statistics** for long-running jobs
- [ ] **Set up checkpoint cleanup** to manage disk space
- [ ] **Review new diagnostics** for optimization insights

---

## Troubleshooting

### Issue 1: "Unknown configuration key: performance.subsampling"

**Symptom:** Warning or error about unknown `subsampling` key

**Cause:** Deprecated configuration section still present

**Solution:**
```bash
# Remove the entire subsampling section from your config
sed -i '/subsampling:/,/seed:/d' your_config.yaml

# Or manually delete:
# performance:
#   subsampling:  # DELETE THIS ENTIRE BLOCK
#     enabled: true
#     max_time_points: 100000
#     strategy: "uniform"
#     seed: 42
```

### Issue 2: Results Different from v2.0

**Symptom:** Parameters or chi-squared values differ from v2.0

**Likely Causes:**
1. **Expected:** v2.0 used subsampling (partial data), v3.0 uses full data
2. **Expected:** Random initialization differences
3. **Unexpected:** Bug - please report

**Validation:**
```python
# Compare on same subsample (for apples-to-apples comparison)
# Run v2.0 and v3.0 on identical subset:
# Both should produce identical results within numerical precision (< 1e-6)
```

**If results are truly different:**
- Check that you're using the same initial parameters
- Verify bounds are identical
- Ensure same dataset (not subsampled vs full)
- Report issue to GitHub with reproducible example

### Issue 3: Memory Error with STREAMING Mode

**Symptom:** `MemoryError` despite using STREAMING mode

**Causes:**
- Batch size too large for available memory
- Model function has memory leak
- JAX compilation overhead

**Solutions:**
```yaml
# Force smaller memory limit
performance:
  memory_limit_gb: 4.0  # Use only 4 GB for batches

# Or reduce checkpoint frequency
optimization:
  streaming:
    checkpoint_frequency: 50  # Save less often
```

```python
# Clear JAX caches
import jax
jax.clear_caches()
```

### Issue 4: Checkpoint Not Found When Resuming

**Symptom:** "No checkpoint found" despite previous run

**Causes:**
- Checkpoint directory path mismatch
- Checkpoints deleted or moved
- `resume_from_checkpoint: false` in config

**Solutions:**
```yaml
# Ensure consistent paths across runs
optimization:
  streaming:
    checkpoint_dir: "./checkpoints"  # Use absolute path if needed
    resume_from_checkpoint: true     # Must be enabled
```

```bash
# Verify checkpoint files exist
ls -lh ./checkpoints/
# Should show homodyne_state_batch_*.h5 files
```

### Issue 5: Slow Performance After Upgrade

**Symptom:** v3.0 is slower than v2.0 for same dataset

**Likely Causes:**
1. **Full data vs subsample:** v3.0 processes 100% of data (expected slower, but better results)
2. **Numerical validation enabled:** ~0.5% overhead
3. **Checkpoints enabled:** ~2% overhead with default frequency

**Solutions:**
```python
# Enable fast mode for production
wrapper = NLSQWrapper(fast_mode=True)
```

```yaml
# Adjust checkpoint frequency
optimization:
  streaming:
    checkpoint_frequency: 50  # Less frequent saves
```

**Expected Performance:**
- v3.0 may be 10-25% slower due to processing full dataset
- Trade-off: Better convergence and no data loss
- Use fast mode to recover most overhead

### Issue 6: Deprecation Warnings in Logs

**Symptom:** Warnings about deprecated configuration keys

**Example Warning:**
```
WARNING: Configuration key 'performance.subsampling' is deprecated.
Please remove this section from your configuration file.
NLSQ now handles large datasets natively.
```

**Solution:** Remove the deprecated section as indicated in the warning

**Safe to Ignore:** These warnings don't affect functionality, but should be addressed

---

## Migration Support

### Getting Help

1. **Documentation:**
   - StreamingOptimizer Usage: `/docs/guides/streaming_optimizer_usage.md`
   - API Reference: `/docs/api/optimization.md`
   - Performance Tuning: `/docs/guides/performance_tuning.md`

2. **Community:**
   - GitHub Issues: https://github.com/your-org/homodyne/issues
   - Discussions: https://github.com/your-org/homodyne/discussions

3. **Contact:**
   - Open an issue with `[migration]` tag
   - Provide config file and error messages
   - Include homodyne and NLSQ versions

### Reporting Migration Issues

Please include:
- Homodyne version (v2.0 → v3.0)
- NLSQ version (`pip show nlsq`)
- Configuration file (sanitized)
- Error messages (full traceback)
- Expected vs actual behavior

---

## Summary

**Key Takeaways:**
1. ✅ Remove `performance.subsampling` from configs
2. ✅ NLSQ now processes 100% of data (no subsampling)
3. ✅ StreamingOptimizer automatically enabled for > 100M points
4. ✅ Enable checkpoints for fault tolerance (optional but recommended)
5. ✅ Use `fast_mode` in production for minimal overhead
6. ✅ Results are backward compatible (better, actually, due to full data)

**Migration is straightforward** for most users - primarily involves removing deprecated configuration sections.

**Upgrade Benefits:**
- 100% data utilization (no sampling loss)
- Unlimited dataset sizes (constant memory)
- Enhanced fault tolerance (checkpoints + recovery)
- Better diagnostics (batch statistics)
- Faster production workflows (fast mode)

---

**Last Updated:** October 22, 2025
**Migration Guide Version:** 1.0
**Covers:** Homodyne v2.0.0 → v3.0.0
