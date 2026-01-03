# Homodyne Deprecation Removal Plan

**Created:** 2026-01-02
**Target Version:** v2.12.0 (full cleanup), v3.0 (final removals)
**Scope:** Remove all deprecated code, dead code, and legacy compatibility layers

---

## Executive Summary

The homodyne codebase contains 40+ deprecated items across 6 categories. This plan provides a phased approach to remove them safely while maintaining backward compatibility where needed.

| Phase | Scope | Files Modified | Risk | Timeline |
|-------|-------|----------------|------|----------|
| 1 | Quick wins (no dependencies) | 2 | None | Immediate |
| 2 | Test migration | 5 | Low | 1 week |
| 3 | CMC modernization | 5 | Low | v2.12.0 |
| 4 | Strategy class removal | 3 | Medium | v2.12.0 |
| 5 | Final cleanup | 2 | Low | v3.0 |

---

## Phase 1: Immediate Removal (No Dependencies)

### 1.1 Remove `should_use_streaming()` Function

**File:** `homodyne/optimization/nlsq/memory.py`
**Lines:** 259-327
**Deprecated:** v2.11.0
**Replacement:** `nlsq.core.workflow.WorkflowSelector` (NLSQ v0.4+)

**Action:** Delete the entire function. No production code calls it.

```python
# DELETE: Lines 259-327 (should_use_streaming function)
# KEEP: get_adaptive_memory_threshold() - still used by NLSQWrapper
```

**Update exports in:**
- `homodyne/optimization/nlsq/__init__.py` - remove from `__all__`
- `homodyne/optimization/__init__.py` - remove from `__all__`

### 1.2 Remove Deprecated Config Path Validation

**File:** `homodyne/runtime/utils/system_validator.py`
**Lines:** ~926-929

**Action:** Remove validation for:
- `performance.subsampling`
- `optimization_performance.time_subsampling`

These paths are never read by production code; only validation warnings exist.

### 1.3 Remove Deprecated Config Warning Logic

**File:** `homodyne/cli/commands.py`
**Lines:** 261-290

**Action:** Remove warning blocks for:
- `performance.subsampling` (lines 274-280)
- `optimization_performance.time_subsampling` (lines 285-290)

**Testing:** Run `make test` to verify no regressions.

---

## Phase 2: Legacy Test Compatibility Layer Migration

### 2.1 Overview

**File to delete:** `tests/utils/legacy_compat.py` (167 lines)

**Test files requiring migration:**

| File | Locations | Functions Used |
|------|-----------|----------------|
| `tests/unit/test_jax_backend.py` | Multiple | All 5 legacy functions |
| `tests/performance/test_benchmarks.py` | 8 | `compute_c2_model_jax`, `compute_g1_diffusion_jax` |
| `tests/unit/test_nlsq_core.py` | 3 | `compute_c2_model_jax` |
| `tests/integration/test_workflows.py` | 1 | `compute_g1_diffusion_jax` |

### 2.2 Migration Mapping

| Legacy Function | Modern Replacement |
|-----------------|-------------------|
| `compute_c2_model_jax()` | `compute_g2_scaled()` from `homodyne.core.fitting` |
| `residuals_jax()` | Manual: `data - compute_g2_scaled()` |
| `chi_squared_jax()` | `compute_chi_squared()` from `homodyne.core.fitting` |
| `compute_g1_diffusion_jax()` | `compute_g1_diffusion()` from `homodyne.core.theory` |
| `compute_g1_shear_jax()` | `compute_g1_shear()` from `homodyne.core.theory` |

### 2.3 Migration Steps

1. **For each test file:**
   - Replace imports from `tests.utils.legacy_compat` with imports from `homodyne.core`
   - Update function calls to match new signatures
   - Run tests to verify behavior unchanged

2. **After all migrations complete:**
   - Delete `tests/utils/legacy_compat.py`
   - Remove from any `conftest.py` fixtures

**Testing:** Run full test suite: `make test-all`

---

## Phase 3: CMC Combination Method Modernization

### 3.1 Problem

Default CMC combination method is `"weighted_gaussian"`, which is mathematically incorrect. The correct method is `"consensus_mc"` (Scott et al., 2016).

### 3.2 Files to Modify

| File | Change |
|------|--------|
| `homodyne/config/manager.py` (line 741) | Change default to `"consensus_mc"` |
| `homodyne/optimization/cmc/config.py` (line 229) | Update default |
| `homodyne/optimization/cmc/backends/base.py` | Add deprecation warning to legacy methods |

### 3.3 Deprecation Warnings

Add to `combine_shard_samples()`:

```python
if method == "weighted_gaussian":
    warnings.warn(
        "combination_method='weighted_gaussian' is deprecated since v2.12.0 "
        "and will be removed in v3.0. Use 'consensus_mc' instead.",
        DeprecationWarning,
        stacklevel=2
    )
elif method == "simple_average":
    warnings.warn(
        "combination_method='simple_average' is deprecated since v2.12.0 "
        "and will be removed in v3.0. Use 'consensus_mc' instead.",
        DeprecationWarning,
        stacklevel=2
    )
```

### 3.4 Migration Guide

Document in CHANGELOG.md:
- v2.12.0: Default changed from `"weighted_gaussian"` to `"consensus_mc"`
- v3.0: `"weighted_gaussian"` and `"simple_average"` removed

---

## Phase 4: DatasetSizeStrategy Removal

### 4.1 Overview

**Class:** `DatasetSizeStrategy`
**File:** `homodyne/optimization/nlsq/strategies/selection.py` (lines 61-583)
**Deprecated:** v2.11.0
**Replacement:** `nlsq.core.workflow.WorkflowSelector` (NLSQ v0.4+)

### 4.2 Current State

- Already has deprecation warning in `__init__()` (v2.11.0)
- Extensive test coverage: `tests/unit/test_nlsq_streaming.py` (1,884 lines)
- Exported in public API via `__all__`

### 4.3 Removal Steps

1. **v2.12.0:**
   - Remove from public exports (`__init__.py` files)
   - Rename file to `_deprecated_selection.py`
   - Keep internal for any remaining wrapper compatibility

2. **v2.13.0:**
   - Delete `_deprecated_selection.py`
   - Delete or archive `tests/unit/test_nlsq_streaming.py`
   - Remove all related helper functions:
     - `estimate_memory_requirements()`
     - `OptimizationStrategy` enum

### 4.4 Functions to Remove

From `homodyne/optimization/nlsq/strategies/selection.py`:

| Function | Lines | Action |
|----------|-------|--------|
| `DatasetSizeStrategy` class | 61-520 | Delete in v2.13.0 |
| `OptimizationStrategy` enum | 17-35 | Delete in v2.13.0 |
| `estimate_memory_requirements()` | 540-583 | Delete in v2.13.0 |

---

## Phase 5: Final Cleanup (v3.0)

### 5.1 Remove Deprecated CMC Methods

From `homodyne/optimization/cmc/backends/base.py`:
- Remove `"weighted_gaussian"` handling
- Remove `"simple_average"` handling
- Keep only `"consensus_mc"`

### 5.2 Remove Legacy Parameter Name Mappings

From `homodyne/optimization/nlsq/transforms.py`:
- Remove `gamma_dot_0` → `gamma_dot_t0` mapping
- Other legacy mappings if no longer needed

### 5.3 Consider NLSQWrapper Retirement

**Current role:** Fallback for NLSQAdapter when adapter fails

**Analysis needed:**
- Review failure rates of NLSQAdapter
- Determine if all NLSQWrapper features are available in NLSQ v1.0+
- If adapter is robust, consider retiring wrapper entirely

**Recommendation:** Keep for v3.0, reassess for v4.0

---

## Already Removed (Reference)

The following were removed in earlier versions and do not need action:

| Item | Removed In | Notes |
|------|------------|-------|
| GPU support | v2.3.0 | Entire `homodyne/runtime/gpu/` deleted |
| `--force-cpu` CLI flag | v2.3.0 | CPU-only now |
| `--gpu-memory-fraction` CLI flag | v2.3.0 | No GPU support |
| VI optimization method | v2.1.0 | Only NLSQ/MCMC remain |
| MCMC `method` parameter | v2.1.0 | Auto-selection now |
| `mcmc.initialization` config | v2.1.0 | Manual workflow |
| Scalar contrast/offset mode | v2.4.0 | Per-angle mandatory |
| `StreamingOptimizer` | NLSQ 0.4.0 | Use `AdaptiveHybridStreamingOptimizer` |
| Subsampling code | v2.5.0+ | Never to be re-added |

---

## Verification Checklist

### After Phase 1:
- [ ] `make test` passes
- [ ] No `should_use_streaming` in codebase (except comments)
- [ ] No deprecated config path warnings

### After Phase 2:
- [ ] `make test-all` passes
- [ ] `tests/utils/legacy_compat.py` deleted
- [ ] All test imports use modern API

### After Phase 3:
- [ ] CMC tests pass with new default
- [ ] Deprecation warnings fire for legacy methods
- [ ] CHANGELOG updated

### After Phase 4:
- [ ] `DatasetSizeStrategy` removed from exports
- [ ] Old streaming tests archived/deleted
- [ ] No public API breakage

### After Phase 5:
- [ ] Only `consensus_mc` CMC method exists
- [ ] No legacy parameter mappings
- [ ] Clean codebase with no deprecation warnings

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking user scripts | Phased deprecation with warnings first |
| Test failures | Migrate tests before removing code |
| Hidden dependencies | Grep verification before each deletion |
| NLSQ version mismatch | Document minimum NLSQ version requirement |

---

## Appendix: Files to Modify (Complete List)

### Phase 1 (2 files):
1. `homodyne/optimization/nlsq/memory.py`
2. `homodyne/cli/commands.py`

### Phase 2 (5 files):
1. `tests/unit/test_jax_backend.py`
2. `tests/performance/test_benchmarks.py`
3. `tests/unit/test_nlsq_core.py`
4. `tests/integration/test_workflows.py`
5. `tests/utils/legacy_compat.py` (delete)

### Phase 3 (5 files):
1. `homodyne/config/manager.py`
2. `homodyne/optimization/cmc/config.py`
3. `homodyne/optimization/cmc/backends/base.py`
4. `homodyne/optimization/cmc/backends/pjit.py`
5. `homodyne/optimization/cmc/backends/pbs.py`

### Phase 4 (3 files):
1. `homodyne/optimization/nlsq/strategies/selection.py`
2. `homodyne/optimization/nlsq/__init__.py`
3. `tests/unit/test_nlsq_streaming.py` (delete/archive)

### Phase 5 (2 files):
1. `homodyne/optimization/cmc/backends/base.py`
2. `homodyne/optimization/nlsq/transforms.py`

**Total:** ~17 files modified/deleted across all phases
