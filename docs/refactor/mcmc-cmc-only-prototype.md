# MCMC CMC-Only Refactor Prototype

**Goal:** Refactor `mcmc.py` to be CMC-only. Remove auto-selection heuristics, standalone NUTS runner/retries, surrogate blocks. All MCMC entrypoints route through CMC coordinator; `shard=1` behaves like legacy NUTS inside CMC.

## Executive Summary

### What Changes
1. **`fit_mcmc_jax()`**: Becomes thin wrapper that always calls CMC coordinator
2. **`should_use_cmc()` in device/config.py**: DELETED - no longer needed
3. **`_run_standard_nuts()`**: DELETED - NUTS runs inside CMC workers only
4. **`bypass.py`**: DELETED - no bypass logic needed
5. **Single-angle surrogates**: Moved INTO CMC worker (shard-level handling)

### What Stays
- **`CMCCoordinator.run_cmc()`**: Unchanged API, handles all MCMC
- **`_worker_function`**: Still runs NUTS per-shard (no change to worker logic)
- **`_create_numpyro_model()`**: Unchanged, used by workers
- **NLSQ stack**: Completely untouched

### Single-Shard Path (Critical)
When `num_shards=1`:
- CMC coordinator creates 1 shard with ALL data
- Worker runs NUTS on that shard (same as legacy standalone NUTS)
- No consensus combination (just returns shard result directly)
- Overhead: ~5-10% vs standalone NUTS (pool creation, serialization)

---

## Architecture After Refactor

```
fit_mcmc_jax()
    ↓
CMCCoordinator.run_cmc()
    ↓
[Calculate num_shards]  ← Always ≥1, single-shard for small datasets
    ↓
shard_data_stratified()
    ↓
backend.run_parallel_mcmc()  ← Even 1 shard uses this
    ↓
_worker_function()  ← NUTS runs HERE (per-shard)
    ↓
combine_subposteriors()  ← Identity for single shard
    ↓
MCMCResult
```

---

## File Changes

### 1. `homodyne/optimization/mcmc.py`

#### Removals (~1200 lines)

```python
# DELETE: Lines 1027-1099 - Automatic NUTS/CMC selection logic
# DELETE: Lines 1100-1156 - CMC branch with bypass check
# DELETE: Lines 1158-1296 - NUTS branch with retry logic
# DELETE: Lines 1299-1801 - _run_standard_nuts() entire function
# DELETE: Lines 1929-1998 - _evaluate_convergence_thresholds()

# Keep but simplify:
# - Lines 628-863 - fit_mcmc_jax() signature and docstring (update to reflect CMC-only)
# - Lines 865-999 - Config loading (parameter_space, initial_values)
```

#### New Simplified `fit_mcmc_jax()` (~150 lines)

```python
@log_performance(threshold=10.0)
def fit_mcmc_jax(
    data: np.ndarray,
    sigma: np.ndarray | None = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "static",
    parameter_space: ParameterSpace | None = None,
    initial_values: dict[str, float] | None = None,
    **kwargs,
) -> MCMCResult:
    """Bayesian parameter estimation using CMC-based MCMC.

    All MCMC runs through CMC coordinator. For small datasets/single-angle,
    CMC uses num_shards=1 which is equivalent to standalone NUTS.

    Parameters
    ----------
    [Same as before, remove 'method' since it's always CMC]

    Notes
    -----
    **Single-Shard Behavior:**
    When dataset is small or has single phi angle, CMC automatically uses
    num_shards=1, which runs NUTS on the full dataset (no sharding overhead
    except pool creation ~1-2s).
    """
    # Validate dependencies
    if not NUMPYRO_AVAILABLE and not BLACKJAX_AVAILABLE:
        raise ImportError(
            "NumPyro or BlackJAX is required for MCMC optimization."
        )
    if not HAS_CORE_MODULES:
        raise ImportError("Core homodyne modules are required")

    # Validate input data
    _validate_mcmc_data(data, t1, t2, phi, q, L)

    # === CONFIG-DRIVEN PARAMETER LOADING (unchanged) ===
    config_dict = kwargs.pop("config", None)

    # Load parameter_space from config if None
    if parameter_space is None:
        # [Same logic as before - lines 906-933]
        ...

    # Load initial_values from config if None
    if initial_values is None:
        # [Same logic as before - lines 936-981]
        ...

    # Validate loaded parameters
    if initial_values is not None:
        is_valid, violations = parameter_space.validate_values(initial_values)
        if not is_valid:
            raise ValueError(f"Initial parameter values violate bounds:\n" + "\n".join(violations))

    # === ALWAYS USE CMC (no selection logic) ===
    dataset_size = data.size if hasattr(data, "size") else len(data)
    num_samples = len(np.unique(phi)) if phi is not None else dataset_size

    logger.info("Starting MCMC via CMC coordinator")
    logger.info(f"Dataset: {dataset_size:,} points, {num_samples} unique phi angles")
    logger.info(f"Analysis mode: {analysis_mode}")

    try:
        from homodyne.optimization.cmc.coordinator import CMCCoordinator
    except ImportError as e:
        raise ImportError("CMC module required for MCMC.") from e

    # Build CMC configuration
    cmc_config = kwargs.pop("cmc_config", {})
    if "mcmc" not in cmc_config:
        cmc_config["mcmc"] = _get_mcmc_config(kwargs)

    # Force single-shard for single-angle data (no parallelism benefit)
    if num_samples == 1:
        cmc_config.setdefault("cmc", {}).setdefault("sharding", {})["num_shards"] = 1
        logger.info("Single-angle data: forcing num_shards=1")

    # Create coordinator and run
    coordinator = CMCCoordinator(cmc_config)
    result = coordinator.run_cmc(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        q=q,
        L=L,
        analysis_mode=analysis_mode,
        parameter_space=parameter_space,
        initial_values=initial_values,
    )

    logger.info(f"CMC completed: {result.num_shards} shard(s), converged={result.converged}")
    return result
```

### 2. `homodyne/device/config.py`

#### DELETE: `should_use_cmc()` function (lines 278-517)

The entire function is no longer needed. Only keep:
- `HardwareConfig` dataclass (used by CMC coordinator for backend selection)
- `detect_hardware()` (used by CMC coordinator)

```python
# BEFORE: Public API
__all__ = [
    "HardwareConfig",
    "detect_hardware",
    "should_use_cmc",  # DELETE
]

# AFTER: Public API
__all__ = [
    "HardwareConfig",
    "detect_hardware",
]
```

### 3. `homodyne/optimization/cmc/bypass.py`

#### DELETE ENTIRE FILE

```python
# This file is no longer needed - CMC is always used
# Remove import from mcmc.py:
# - from homodyne.optimization.cmc.bypass import evaluate_cmc_bypass
```

### 4. `homodyne/optimization/cmc/__init__.py`

```python
# BEFORE
from homodyne.optimization.cmc.bypass import evaluate_cmc_bypass, CMCBypassDecision
from homodyne.optimization.cmc.coordinator import CMCCoordinator
...

# AFTER - remove bypass exports
from homodyne.optimization.cmc.coordinator import CMCCoordinator
...
```

### 5. `homodyne/optimization/cmc/coordinator.py`

#### Add: Single-shard identity combination

```python
# In run_cmc(), after Step 3 (parallel MCMC):

# =====================================================================
# Step 3: Combine subposteriors (or identity for single shard)
# =====================================================================
if len(shard_results) == 1:
    logger.info("Single shard - using identity combination (no consensus needed)")
    combined_posterior = {
        "samples": shard_results[0]["samples"],
        "mean": np.mean(shard_results[0]["samples"], axis=0),
        "cov": np.cov(shard_results[0]["samples"], rowvar=False),
        "method": "identity",  # Mark as single-shard
    }
    combination_time = 0.0
else:
    # Existing combination logic
    combination_start = time.time()
    combined_posterior = combine_subposteriors(...)
    combination_time = time.time() - combination_start
```

#### Update: Minimum shard count logic

```python
def _calculate_num_shards(self, dataset_size: int) -> int:
    """Calculate optimal number of shards.

    Always returns at least 1 (CMC-only architecture).
    """
    # Allow user override
    user_num_shards = self.config.get("cmc", {}).get("sharding", {}).get("num_shards")
    if user_num_shards is not None:
        return max(1, user_num_shards)  # Enforce minimum of 1

    # Calculate automatically - same logic as before
    # BUT: Remove any "use NUTS instead" returns - always return ≥1
    ...

    # Ensure at least 1 shard
    return max(1, calculated_shards)
```

### 6. `homodyne/optimization/cmc/backends/multiprocessing.py`

#### Move single-angle surrogate handling INTO worker

The single-angle surrogate configuration (tier 1-4, log-space D0, etc.) currently lives in `_run_standard_nuts()`. This needs to move into `_worker_function()`:

```python
def _worker_function(args: tuple) -> Dict[str, Any]:
    """Worker function - executes NUTS on a single shard."""
    # ... existing setup ...

    # === SINGLE-ANGLE SURROGATE HANDLING (moved from _run_standard_nuts) ===
    phi_unique = np.unique(shard["phi"])
    n_unique_phi = len(phi_unique)
    single_angle_static = (
        analysis_mode.lower().startswith("static") and n_unique_phi == 1
    )

    if single_angle_static:
        # Apply surrogate configuration
        # [Move lines 1443-1576 from mcmc.py here]
        stable_prior_enabled = mcmc_config.get("stable_prior_fallback", False)
        surrogate_tier = os.environ.get("HOMODYNE_SINGLE_ANGLE_TIER", "2")

        # Build surrogate settings
        from homodyne.optimization.mcmc import _build_single_angle_surrogate_settings
        single_angle_surrogate_cfg = _build_single_angle_surrogate_settings(
            parameter_space, surrogate_tier
        )

        # Apply parameter space modifications
        if single_angle_surrogate_cfg.get("drop_d_offset"):
            parameter_space = parameter_space.drop_parameters({"D_offset"})
            # ... rest of surrogate handling

    # ... rest of worker function unchanged ...
```

---

## API Changes Summary

| Before | After | Notes |
|--------|-------|-------|
| `fit_mcmc_jax(..., method='mcmc')` | `fit_mcmc_jax(...)` | Always CMC, no method param |
| `should_use_cmc(num_samples, hw)` | DELETED | No selection logic |
| `evaluate_cmc_bypass(config, ...)` | DELETED | No bypass logic |
| `_run_standard_nuts(...)` | DELETED | NUTS only in workers |
| Auto-retry in fit_mcmc_jax | DELETED | Retry logic in worker if needed |

### Configuration Changes

```yaml
# BEFORE (v2.4)
optimization:
  mcmc:
    min_samples_for_cmc: 15      # DELETE
    memory_threshold_pct: 0.30   # DELETE
    large_dataset_threshold: 1M  # DELETE
  cmc:
    bypass_mode: auto            # DELETE
    bypass_thresholds:           # DELETE entire section

# AFTER (v3.0)
optimization:
  mcmc:
    # Only MCMC sampling config
    n_samples: 2000
    n_warmup: 1000
    n_chains: 4
    target_accept_prob: 0.9
  cmc:
    sharding:
      strategy: stratified
      num_shards: auto  # or explicit int, minimum 1
      target_shard_size_cpu: 2_000_000
    combination:
      method: weighted
      fallback_enabled: true
```

---

## Pitfalls and Edge Cases

### 1. Single-Shard Path Must Handle All Cases

**Problem:** When `num_shards=1`, the single worker must handle:
- Single-angle static datasets (surrogate tiers)
- Multi-angle static datasets
- Laminar flow datasets
- All per-angle scaling variations

**Solution:** Move ALL preprocessing logic that was in `_run_standard_nuts()` into `_worker_function()`. The worker becomes the "universal NUTS executor".

### 2. Pool Creation Overhead for Single Shard

**Problem:** Creating multiprocessing.Pool for 1 worker adds ~1-2s overhead.

**Solutions:**
1. Accept overhead (simplest, consistent architecture)
2. Special-case `num_shards=1` to skip pool (hybrid, adds complexity)
3. Use ThreadPoolExecutor for single shard (avoids spawn cost)

**Recommendation:** Accept overhead. The architectural simplicity is worth 1-2s on small datasets.

### 3. Retry Logic Migration

**Before:** `fit_mcmc_jax` had 3 retries with different seeds.

**After options:**
1. Move retry logic INTO worker (per-shard retry)
2. Move retry logic INTO coordinator (re-run failed shards)
3. Remove retry, rely on CMC combination to be robust

**Recommendation:** Option 2 - Coordinator-level retry for failed shards. This is more natural for CMC architecture.

### 4. Diagnostics and Convergence Thresholds

**Before:** `_evaluate_convergence_thresholds()` in mcmc.py evaluated result.

**After:** This logic should move to:
- `CMCCoordinator._basic_validation()` (already exists, expand it)
- OR return in MCMCResult and let caller evaluate

### 5. Import Cleanup

Many imports in mcmc.py become unused:
```python
# DELETE these imports
from homodyne.optimization.cmc.bypass import evaluate_cmc_bypass
from homodyne.device.config import should_use_cmc

# These stay (used by workers)
from homodyne.device.config import detect_hardware, HardwareConfig
```

---

## Migration Path

### Phase 1: Prototype (This Document)
- Document all changes
- Validate single-shard path works

### Phase 2: Implementation
1. Delete `bypass.py`
2. Delete `should_use_cmc()` from device/config.py
3. Refactor `fit_mcmc_jax()` to always use CMC
4. Move surrogate logic to worker
5. Add single-shard identity combination

### Phase 3: Testing
- Unit tests for single-shard path
- Integration tests for migration from v2.4
- Performance benchmarks (single-shard overhead)

### Phase 4: Documentation
- Update CLAUDE.md
- Update CLI help strings
- Deprecation notices for removed config options

---

## Diff Summary

| File | Lines Added | Lines Removed | Net Change |
|------|-------------|---------------|------------|
| mcmc.py | ~150 | ~1200 | -1050 |
| device/config.py | 0 | ~240 | -240 |
| cmc/bypass.py | 0 | 183 | -183 (delete) |
| cmc/coordinator.py | ~20 | 0 | +20 |
| cmc/backends/multiprocessing.py | ~100 | 0 | +100 |
| **Total** | ~270 | ~1623 | **-1353** |

Net reduction of ~1350 lines while maintaining all functionality.
