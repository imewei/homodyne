# Fix: Single-Angle MCMC Initialization Logic

**Date**: 2025-12-01
**Issue**: Incorrect logic for handling contrast/offset in single-angle MCMC analysis
**Fix**: Implement correct priority: config initial_values → data-derived fallback (both as starting points, NOT fixed)

---

## The Problem

### User's Correct Understanding

> "The logic here should Use initial_values as MCMC starting points (0.05015, 1.001). If the initial_values are not available in the configuration files, we should use Data-derived from percentiles (0.0406, 1.0002) as MCMC starting points. This is true for both NUTS and CMC."

### What the Code Was Doing (WRONG)

**Before Fix**:
1. Compute data-derived values: `contrast=0.0406, offset=1.0002`
2. Create `single_angle_scaling_override = {'contrast': 0.0406, 'offset': 1.0002}`
3. If user provided `initial_values`, override with those (but still as "overrides")
4. Pass `scaling_overrides` to NumPyro model
5. Model creates **deterministic nodes** (fixes parameters, doesn't sample them)

**Problems**:
- ❌ Contrast/offset were **FIXED** (deterministic), not **SAMPLED**
- ❌ Called them "overrides" when they should be "initial values"
- ❌ Confusing priority logic
- ❌ No clear distinction between "starting point" and "fixed parameter"

---

## The Fix

### New Logic (CORRECT)

**File**: `homodyne/optimization/mcmc.py:1459-1523`

```python
if single_angle_static:
    # Compute data-derived estimates as fallback
    contrast_data_derived, offset_data_derived = _estimate_single_angle_scaling(data)
    # contrast_data_derived = 0.0406, offset_data_derived = 1.0002

    # Check if user provided initial_values for contrast/offset
    has_contrast_init = initial_values is not None and "contrast" in initial_values
    has_offset_init = initial_values is not None and "offset" in initial_values

    # Priority 1: Use config initial_values if available
    if has_contrast_init:
        # User provided contrast=0.05015 in config
        logger.info(
            "Using config initial_values contrast=%.4f as MCMC starting point",
            initial_values["contrast"]  # 0.05015
        )
    else:
        # Priority 2: Fallback to data-derived
        initial_values["contrast"] = contrast_data_derived  # 0.0406
        logger.info(
            "Using data-derived contrast=%.4f as MCMC starting point (no initial_values provided)",
            contrast_data_derived
        )

    # Same logic for offset
    if has_offset_init:
        logger.info(
            "Using config initial_values offset=%.4f as MCMC starting point",
            initial_values["offset"]  # 1.001
        )
    else:
        initial_values["offset"] = offset_data_derived  # 1.0002
        logger.info(
            "Using data-derived offset=%.4f as MCMC starting point (no initial_values provided)",
            offset_data_derived
        )

    # CRITICAL: Do NOT create scaling_overrides!
    # We want contrast/offset to be SAMPLED, not FIXED
    single_angle_scaling_override = None
```

### Key Changes

1. **Priority is clear**: config `initial_values` → data-derived fallback
2. **Both are starting points**: Neither creates deterministic nodes
3. **No scaling_overrides**: Set to `None` for single-angle data
4. **Clear logging**: Explicitly states "MCMC starting point"
5. **Sampling enabled**: Contrast/offset will be sampled with priors

---

## Behavior Comparison

### Case 1: User Provides initial_values in Config

**Config**:
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset', 'contrast', 'offset']
  values: [16830.0, -1.571, 3.026, 0.05015, 1.001]
```

**Before Fix**:
```
1. Compute data-derived: contrast=0.0406, offset=1.0002
2. Create scaling_override = {'contrast': 0.0406, 'offset': 1.0002}
3. Override with user values: {'contrast': 0.05015, 'offset': 1.001}
4. Pass to model as fixed_scaling_overrides
5. Model creates deterministic nodes (FIXED, not sampled!)
6. Result: Contrast/offset NOT fitted ❌
```

**After Fix**:
```
1. Compute data-derived: contrast=0.0406, offset=1.0002 (for logging)
2. Check: has_contrast_init=True, has_offset_init=True
3. Use config values: initial_values = {'contrast': 0.05015, 'offset': 1.001}
4. Log: "Using config initial_values contrast=0.05015 as MCMC starting point"
5. single_angle_scaling_override = None (no deterministic nodes!)
6. Pass initial_values to NUTS
7. Model samples contrast/offset with priors, starting from 0.05015, 1.001
8. Result: Contrast/offset FITTED ✅
```

---

### Case 2: User Does NOT Provide initial_values

**Config**:
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset']
  values: [16830.0, -1.571, 3.026]
  # No contrast/offset!
```

**Before Fix**:
```
1. Compute data-derived: contrast=0.0406, offset=1.0002
2. Create scaling_override = {'contrast': 0.0406, 'offset': 1.0002}
3. No user override (initial_values doesn't have contrast/offset)
4. Pass to model as fixed_scaling_overrides
5. Model creates deterministic nodes (FIXED!)
6. Result: Contrast/offset fixed to 0.0406, 1.0002 ❌
```

**After Fix**:
```
1. Compute data-derived: contrast=0.0406, offset=1.0002
2. Check: has_contrast_init=False, has_offset_init=False
3. Add to initial_values: {'contrast': 0.0406, 'offset': 1.0002}
4. Log: "Using data-derived contrast=0.0406 as MCMC starting point"
5. single_angle_scaling_override = None (no deterministic nodes!)
6. Pass initial_values to NUTS
7. Model samples contrast/offset with priors, starting from 0.0406, 1.0002
8. Result: Contrast/offset FITTED ✅
```

---

## Log Messages

### Before Fix

```
INFO | Single-angle fallback: fixing contrast=0.0406, offset=1.0002 based on data percentiles
INFO | Single-angle fallback: using user-provided scaling overrides contrast=0.05015, offset=1.001
```
→ Misleading: Says "fixing" (implies deterministic) and "overrides" (confusing)

### After Fix

**With config initial_values**:
```
DEBUG | Data-derived scaling estimates: contrast=0.0406, offset=1.0002 (from 1st/99th percentile)
INFO  | Using config initial_values contrast=0.05015 as MCMC starting point
INFO  | Using config initial_values offset=1.001 as MCMC starting point
```

**Without config initial_values**:
```
DEBUG | Data-derived scaling estimates: contrast=0.0406, offset=1.0002 (from 1st/99th percentile)
INFO  | Using data-derived contrast=0.0406 as MCMC starting point (no initial_values provided)
INFO  | Using data-derived offset=1.0002 as MCMC starting point (no initial_values provided)
```

→ Clear: Explicitly states "MCMC starting point", no confusion about deterministic vs. sampled

---

## Edge Case: Explicitly Fixing Parameters (Rare)

If a user WANTS to fix contrast/offset (not recommended), they can still do it:

```python
result = fit_mcmc_jax(
    data, ...,
    initial_values={'D0': 16830, 'alpha': -1.571, 'D_offset': 3.026},
    fixed_scaling_overrides={'contrast': 0.05015, 'offset': 1.001},  # Explicitly fix!
)
```

**After Fix**:
```
WARNING | Explicitly fixing contrast/offset as deterministic parameters: contrast=0.05015, offset=1.001
WARNING | These parameters will NOT be sampled! Use initial_values instead if you want them fitted.
```

This makes it clear that the user is intentionally fixing these parameters.

---

## Impact on CMC

The fix applies to **both NUTS and CMC** because:

1. CMC uses the same `initial_values` mechanism
2. CMC coordinator also receives `scaling_overrides`
3. Setting `single_angle_scaling_override = None` ensures CMC samples contrast/offset
4. Data-derived fallback works for both methods

**CMC-specific note**: The CMC coordinator expands `initial_values` for per-angle scaling correctly, so this fix is fully compatible.

---

## Testing

### Expected Behavior After Fix

**With your config** (`contrast=0.05015, offset=1.001`):

```bash
homodyne --config config.yaml --method mcmc
```

**Expected log**:
```
INFO  | Single-angle static dataset detected → disabling per-angle scaling...
DEBUG | Data-derived scaling estimates: contrast=0.0406, offset=1.0002 (from 1st/99th percentile)
INFO  | Using config initial_values contrast=0.05015 as MCMC starting point
INFO  | Using config initial_values offset=1.001 as MCMC starting point
INFO  | Initializing MCMC chains with values: ['D0', 'alpha', 'D_offset', 'contrast', 'offset'] = ['1.683e+04', '-1.571', '3.026', '0.05015', '1.001']
```

**MCMC sampling**:
- Starts at: `contrast=0.05015, offset=1.001`
- Samples using: `TruncatedNormal(0.5, 0.2, [0, 1])` and `TruncatedNormal(1.0, 0.2, [0.5, 1.5])`
- Fitted values: Posterior means (e.g., `contrast=0.0495±0.0034, offset=1.001±0.0012`)

**No more**:
- ❌ Deterministic nodes for contrast/offset
- ❌ "Single-angle fallback: fixing..." messages
- ❌ Garbage values from failed deterministic initialization

---

## Summary

### What Changed

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Priority** | Data-derived → user override | Config initial_values → data-derived fallback |
| **Mechanism** | scaling_overrides (deterministic) | initial_values (starting points) |
| **Contrast/offset** | FIXED (not sampled) | SAMPLED (with priors) |
| **Log message** | "fixing... overrides..." | "MCMC starting point..." |
| **Applies to** | NUTS only (CMC unclear) | Both NUTS and CMC |

### User Requirements Met

✅ **Requirement 1**: "Use initial_values as MCMC starting points (0.05015, 1.001)"
✅ **Requirement 2**: "If initial_values not available, use data-derived as starting points"
✅ **Requirement 3**: "True for both NUTS and CMC"

### Files Modified

- **`homodyne/optimization/mcmc.py`**: Lines 1459-1523

### Breaking Changes

**None** - This is a bug fix that implements the correct behavior. Users who were relying on the buggy behavior (fixed contrast/offset) can still achieve it by explicitly passing `fixed_scaling_overrides` parameter.

---

## Code Location Reference

**Main fix**: `homodyne/optimization/mcmc.py:1459-1523`

```python
# Line 1467: Compute data-derived estimates
contrast_data_derived, offset_data_derived = _estimate_single_angle_scaling(data)

# Lines 1475-1476: Check if config provides initial values
has_contrast_init = initial_values is not None and "contrast" in initial_values
has_offset_init = initial_values is not None and "offset" in initial_values

# Lines 1478-1501: Priority logic (config → data-derived)
if has_contrast_init:
    logger.info("Using config initial_values contrast=%.4f as MCMC starting point", ...)
else:
    initial_values["contrast"] = contrast_data_derived
    logger.info("Using data-derived contrast=%.4f as MCMC starting point...", ...)

# Line 1506: Critical - DO NOT create scaling_overrides!
single_angle_scaling_override = None
```

**User override handling**: `homodyne/optimization/mcmc.py:1510-1523`

```python
# Only if user EXPLICITLY passes fixed_scaling_overrides parameter
if user_scaling_override:
    single_angle_scaling_override = {...}
    logger.warning("Explicitly fixing contrast/offset as deterministic parameters...")
    logger.warning("These parameters will NOT be sampled! ...")
```
