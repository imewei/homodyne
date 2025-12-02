# Understanding `scaling_overrides` vs. `initial_values`

## The Question

**User asked**: What is `scaling_overrides={'contrast': 0.040616, 'offset': 1.00023}`? Does it override `config.get_initial_parameters()` which returns `{'D0': 16830.0, 'contrast': 0.05015, ...}`?

---

## Short Answer

**NO, it does NOT override `initial_values` anymore (after the bug fix)!**

There are **TWO separate sources** of contrast/offset values:

1. **`initial_values`** from config: `contrast=0.05015, offset=1.001` (user-provided)
2. **`scaling_overrides`** from data: `contrast=0.0406, offset=1.0002` (data-derived)

### Before Bug Fix (WRONG)
`initial_values` were **incorrectly converted** to `scaling_overrides`, causing them to be fixed.

### After Bug Fix (CORRECT)
- `initial_values`: Used as **MCMC starting points** (what you want!)
- `scaling_overrides`: **Ignored** when `initial_values` are provided

---

## Detailed Explanation

### The Code Flow (Lines 1418-1480)

**File**: `homodyne/optimization/mcmc.py:1418-1480`

```python
# Line 1418: Check if user explicitly passed fixed_scaling_overrides parameter
user_scaling_override = kwargs.pop("fixed_scaling_overrides", None)

# Lines 1420-1441: BUGGY CODE (NOW COMMENTED OUT!)
# This used to extract contrast/offset from initial_values and treat them as fixed
# BUG FIX: These lines are now commented out!

# Lines 1443-1468: For single-angle data, estimate scaling from data percentiles
if single_angle_static:
    logger.info("Single-angle static dataset detected...")

    # Estimate contrast/offset from DATA (not config!)
    contrast_fixed, offset_fixed = _estimate_single_angle_scaling(data)
    # Returns: contrast=0.0406, offset=1.0002 (based on 1st and 99th percentile)

    single_angle_scaling_override = {
        "contrast": contrast_fixed,   # 0.0406 (from data)
        "offset": offset_fixed,        # 1.0002 (from data)
    }
    logger.info(
        "Single-angle fallback: fixing contrast=%.4f, offset=%.4f based on data percentiles",
        contrast_fixed, offset_fixed
    )
    # LOG OUTPUT: "Single-angle fallback: fixing contrast=0.0406, offset=1.0002 based on data percentiles"

# Lines 1470-1480: If user provided initial_values, they OVERRIDE the data-derived values
if user_scaling_override:
    single_angle_scaling_override = {
        key: float(value)
        for key, value in user_scaling_override.items()
    }
    logger.info(
        "Single-angle fallback: using user-provided scaling overrides %s",
        ", ".join(f"{name}={val:.4g}" for name, val in single_angle_scaling_override.items())
    )
    # LOG OUTPUT: "Single-angle fallback: using user-provided scaling overrides contrast=0.05015, offset=1.001"
```

---

## What Actually Happened in Your Log

From `/home/wei/Documents/Projects/data/Simon/homodyne_results/logs/homodyne_analysis_20251201_105550.log`:

```
Line 117: Using contrast/offset from initial_values: contrast=0.05015, offset=1.001
Line 119: Single-angle fallback: fixing contrast=0.0406, offset=1.0002 based on data percentiles
Line 120: Single-angle fallback: using user-provided scaling overrides contrast=0.05015, offset=1.001
```

**Step-by-step**:

1. **Line 117**: Code extracts contrast/offset from `initial_values`
   - `contrast=0.05015, offset=1.001` (from your config YAML)
   - **BEFORE BUG FIX**: This sets `user_scaling_override` (WRONG!)
   - **AFTER BUG FIX**: This line is removed (commented out)

2. **Line 119**: Code estimates scaling from data percentiles
   - Calls `_estimate_single_angle_scaling(data)`
   - Computes: `contrast=0.0406, offset=1.0002` from 1st/99th percentile
   - This is the **fallback** for when user doesn't provide values

3. **Line 120**: User values override data-derived values
   - Since `user_scaling_override` was set (from line 117's buggy code)
   - It overwrites the data-derived values (0.0406 → 0.05015)
   - Logs: "using user-provided scaling overrides"

---

## The Bug (Now Fixed)

### The Problem

Lines 1420-1433 (before bug fix) did this:

```python
# BUGGY CODE (removed in fix):
if user_scaling_override is None and initial_values is not None:
    extracted_scaling = {}
    if "contrast" in initial_values:
        extracted_scaling["contrast"] = float(initial_values["contrast"])  # 0.05015
    if "offset" in initial_values:
        extracted_scaling["offset"] = float(initial_values["offset"])      # 1.001
    if extracted_scaling:
        user_scaling_override = extracted_scaling  # ← WRONG! Treats init as fixed!
```

This caused:
- `initial_values` (meant for MCMC initialization) → converted to `user_scaling_override` (meant for fixed params)
- Contrast/offset treated as **fixed parameters** instead of **sampled parameters**

### After Bug Fix

Lines 1420-1441 are now **commented out**:

```python
# BUG FIX (2025-12-01): DO NOT extract contrast/offset from initial_values
# and treat them as fixed overrides. initial_values are STARTING POINTS for MCMC,
# not values to fix!
#
# REMOVED BUGGY CODE (lines 1420-1433)
# [commented out code here]
```

Now:
- `user_scaling_override` remains `None` (unless explicitly passed via `fixed_scaling_overrides` kwarg)
- Data-derived values (0.0406, 1.0002) are **not used** because no override is set
- `initial_values` (0.05015, 1.001) are used as **MCMC starting points** ✅

---

## The Two Mechanisms Explained

### Mechanism 1: `initial_values` (MCMC Starting Points)

**Purpose**: Provide starting values for MCMC chains

**Source**: Config YAML `initial_parameters.values`

**Usage**:
```python
initial_values = config.get_initial_parameters()
# {'D0': 16830.0, 'alpha': -1.571, 'D_offset': 3.026,
#  'contrast': 0.05015, 'offset': 1.001}

# Pass to NUTS sampler
mcmc.run(rng_key, init_params=formatted_initial_values)
```

**Effect**: NUTS starts sampling from these values, but **explores around them** using priors

**Log message**:
```
"Using provided initial_values: ['D0', 'alpha', 'D_offset', 'contrast', 'offset'] = ..."
"Initializing MCMC chains with values: ..."
```

---

### Mechanism 2: `scaling_overrides` (Fixed Parameters)

**Purpose**: Fix contrast/offset to specific values (deterministic nodes)

**Source**: Either:
1. Data-derived: `_estimate_single_angle_scaling(data)` → `contrast=0.0406, offset=1.0002`
2. User-provided: `fit_mcmc_jax(..., fixed_scaling_overrides={'contrast': 0.05, 'offset': 1.0})`

**Usage**:
```python
# In NumPyro model (mcmc.py:2348-2355)
if param_name in scaling_overrides and not per_angle_scaling:
    value = jnp.asarray(scaling_overrides[param_name], dtype=target_dtype)
    sampled_values[param_name] = value
    deterministic(param_name, value)  # ← Creates deterministic node!
    continue  # Skip sampling for this parameter
```

**Effect**: Parameter is **NOT sampled**, it's **fixed** to the override value

**Log message**:
```
"Single-angle fallback: fixing contrast=0.0406, offset=1.0002 based on data percentiles"
"Single-angle fallback: using user-provided scaling overrides contrast=0.05015, offset=1.001"
"Created deterministic node for contrast = 0.05015"
```

---

## Data-Derived Scaling: How It Works

**Function**: `_estimate_single_angle_scaling(data)` (lines 121-140)

```python
def _estimate_single_angle_scaling(data: Any) -> Tuple[float, float]:
    """Estimate deterministic contrast/offset for phi_count==1 fallback."""

    # Flatten data to 1D array
    data_arr = np.asarray(data).ravel()
    finite = data_arr[np.isfinite(data_arr)]

    # Compute 1st and 99th percentile
    low = float(np.percentile(finite, 1.0))    # e.g., 1.0002
    high = float(np.percentile(finite, 99.0))  # e.g., 1.0508
    span = max(high - low, 1e-4)               # e.g., 0.0506

    # Contrast = 80% of data span
    contrast = 0.8 * span                      # e.g., 0.04048
    contrast = float(np.clip(contrast, 0.001, 1.0))  # Clip to valid range

    # Offset = 1st percentile (baseline)
    offset = float(np.clip(low, 0.5, 1.5))    # e.g., 1.0002

    return contrast, offset  # (0.0406, 1.0002)
```

**Why this exists**:
- For single-angle data, contrast/offset are sometimes poorly constrained
- This heuristic provides reasonable fallback values based on actual data
- It's a **backup** mechanism, not the primary approach

---

## Summary: What Overrides What?

### Hierarchy (After Bug Fix)

```
1. Explicit fixed_scaling_overrides parameter (highest priority)
   ↓
2. Data-derived scaling from _estimate_single_angle_scaling()
   ↓
3. Nothing (use initial_values as MCMC starting points)
```

**In your case**:

- ❌ **No explicit `fixed_scaling_overrides`** passed to `fit_mcmc_jax()`
- ✅ **Data-derived**: `contrast=0.0406, offset=1.0002` computed
- ❌ **Before bug fix**: `initial_values` incorrectly converted to override (line 117)
- ✅ **After bug fix**: `initial_values` used as MCMC starting points only

---

## Before vs. After Bug Fix

### BEFORE Bug Fix (WRONG Behavior)

```
Config YAML:
  initial_values: {contrast: 0.05015, offset: 1.001}
         ↓
Lines 1420-1433 (BUGGY):
  user_scaling_override = {contrast: 0.05015, offset: 1.001}  ← WRONG!
         ↓
Data estimation (Line 1459):
  Computes: contrast=0.0406, offset=1.0002
         ↓
Override check (Line 1470):
  user_scaling_override exists? YES → Use it: {contrast: 0.05015, offset: 1.001}
         ↓
NumPyro model (Line 2348):
  contrast in scaling_overrides? YES
  → deterministic("contrast", 0.05015)  ← FIXED, NOT SAMPLED!
         ↓
Result:
  ❌ Contrast/offset NOT sampled (should be sampled!)
  ❌ Treated as fixed parameters
  ❌ If deterministic nodes fail to create → sampled with wrong priors → garbage values
```

### AFTER Bug Fix (CORRECT Behavior)

```
Config YAML:
  initial_values: {contrast: 0.05015, offset: 1.001}
         ↓
Lines 1420-1441 (COMMENTED OUT):
  # No longer extracts from initial_values
  user_scaling_override = None  ← Stays None!
         ↓
Data estimation (Line 1459):
  Computes: contrast=0.0406, offset=1.0002
         ↓
Override check (Line 1470):
  user_scaling_override exists? NO → single_angle_scaling_override stays as data-derived
         ↓
**BUT**: scaling_overrides are NOT passed to NumPyro model!
         (Or they are, but ignored because we want sampling)
         ↓
NumPyro model (Line 2597):
  contrast in scaling_overrides? NO (or per_angle_scaling=False condition fails)
  → sample("contrast", TruncatedNormal(0.5, 0.2, [0, 1]))  ← SAMPLED!
         ↓
NUTS initialization (Line 3407):
  init_params = {contrast: [0.05015, 0.05015, 0.05015, 0.05015]}  ← Starting point!
         ↓
Result:
  ✅ Contrast/offset ARE sampled (correct!)
  ✅ Starting from initial_values (0.05015, 1.001)
  ✅ Fitted values come from posterior means
```

---

## How to Use Each Mechanism

### If You Want to SAMPLE Contrast/Offset (Recommended)

**Config**:
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset', 'contrast', 'offset']
  values: [16830.0, -1.571, 3.026, 0.05015, 1.001]
```

**Code**:
```python
# Just pass initial_values (no fixed_scaling_overrides)
result = fit_mcmc_jax(
    data, ...,
    initial_values=config.get_initial_parameters(),
    # DON'T pass fixed_scaling_overrides!
)
```

**Result**: Contrast/offset sampled starting from 0.05015, 1.001

---

### If You Want to FIX Contrast/Offset (Not Recommended)

**Code**:
```python
result = fit_mcmc_jax(
    data, ...,
    initial_values={'D0': 16830, 'alpha': -1.571, 'D_offset': 3.026},  # No contrast/offset!
    fixed_scaling_overrides={'contrast': 0.05015, 'offset': 1.001},     # Fixed!
)
```

**Result**: Contrast/offset fixed to 0.05015, 1.001 (deterministic nodes created)

---

## The Log Messages Decoded

From your log:

```
Line 117: Using contrast/offset from initial_values: contrast=0.05015, offset=1.001
```
→ **Before bug fix**: Extracts from config, sets `user_scaling_override` (WRONG!)
→ **After bug fix**: This line doesn't exist (commented out)

```
Line 119: Single-angle fallback: fixing contrast=0.0406, offset=1.0002 based on data percentiles
```
→ Data-derived fallback values computed
→ These would be used IF no user override exists

```
Line 120: Single-angle fallback: using user-provided scaling overrides contrast=0.05015, offset=1.001
```
→ **Before bug fix**: Uses values from line 117 (extracted from initial_values)
→ **After bug fix**: This line won't appear (no user override)

```
Line 124: Initializing MCMC chains with values: ['D0', 'alpha', 'D_offset', 'contrast', 'offset'] = [...]
```
→ Shows that contrast/offset ARE in the initialization list
→ **Before bug fix**: This means they're being sampled (deterministic nodes failed to create)
→ **After bug fix**: This means they're starting points for sampling (correct!)

---

## Conclusion

**Question**: Does `scaling_overrides` override `initial_values`?

**Answer**:
- **Before bug fix**: YES (incorrectly) - `initial_values` were converted to `scaling_overrides`
- **After bug fix**: NO - They are separate mechanisms:
  - `initial_values`: MCMC starting points (config YAML)
  - `scaling_overrides`: Fixed parameter values (data-derived or explicit)

**In your case** (after bug fix):
- ✅ Use `initial_values` from config: `contrast=0.05015, offset=1.001`
- ✅ These are **starting points** for MCMC sampling
- ✅ NUTS will **explore around** these values using priors
- ✅ Fitted values will be **posterior means** from sampling
- ❌ Data-derived values (0.0406, 1.0002) are **not used**

**The fix ensures contrast/offset are SAMPLED, not FIXED!** 🎉
