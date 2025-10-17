# NLSQ Chunking Fix Plan

**Date:** October 17, 2025 **Issue:** curve_fit_large returns identity covariance
because all chunks fail silently **Root Cause:** Model function returns fixed 23M output
regardless of chunk size → shape mismatch

______________________________________________________________________

## Root Cause Analysis

### The Bug

**Location:** `homodyne/optimization/nlsq_wrapper.py` lines 862-916 (`model_function`)

**Problem:**

```python
def model_function(xdata: jnp.ndarray, *params_tuple) -> jnp.ndarray:
    # Model ignores xdata and always computes full output
    g2_theory = compute_g2_scaled_vmap(phi)  # Full (n_phi, n_t1, n_t2)
    g2_theory_flat = g2_theory.flatten()     # Always 23M points
    return g2_theory_flat                     # ❌ WRONG SIZE
```

**When curve_fit_large chunks:** -Input: `x_chunk` (1M indices), `y_chunk` (1M data
points)

- Model called: `model(x_chunk, *params)`
- Model returns: 23M points (ignores x_chunk size)
- Residual computation: `y_chunk - model_output` → (1M,) - (23M,) → **Shape mismatch!**
- Result: All chunks fail → success_rate = 0% → identity covariance fallback

### Why It Happens

`curve_fit_large` → `LargeDatasetFitter._fit_chunked()` → processes data in chunks:

1. Creates chunks: `(x_chunk, y_chunk, chunk_idx)` via `DataChunker.create_chunks()`
1. Calls `curve_fit(f, x_chunk, y_chunk, ...)` for each chunk
1. Expects `f(x_chunk, *params)` to return array matching `y_chunk.shape`
1. Our model returns fixed 23M array → **FAILS**

______________________________________________________________________

## The Fix

### Solution: Make model function respect xdata indices

**Change model_function to:**

```python
def model_function(xdata: jnp.ndarray, *params_tuple) -> jnp.ndarray:
    """Compute theoretical g2 model for NLSQ optimization.

    IMPORTANT: Must respect xdata size for curve_fit_large chunking.
    xdata contains indices into the flattened data array.
    """
    # Convert params tuple to array
    params_array = jnp.array(params_tuple)
    contrast = params_array[0]
    offset = params_array[1]
    physical_params = params_array[2:]

    # Compute theoretical g2 for all phi angles
    compute_g2_scaled_vmap = jax.vmap(
        lambda phi_val: jnp.squeeze(
            compute_g2_scaled(
                params=physical_params,
                t1=t1,
                t2=t2,
                phi=phi_val,
                q=q,
                L=L,
                contrast=contrast,
                offset=offset,
                dt=dt,
            ),
            axis=0,
        ),
        in_axes=0,
    )

    g2_theory = compute_g2_scaled_vmap(phi)
    g2_theory_flat = g2_theory.flatten()

    # ✅ FIX: Return only requested indices
    # xdata contains integer indices into flattened array
    # For full dataset: xdata = [0, 1, 2, ..., 23M-1]
    # For chunk: xdata = [0, 1, 2, ..., 1M-1] (subset)
    indices = xdata.astype(jnp.int32)
    return g2_theory_flat[indices]
```

### Key Changes

1. **Line 914 (new):** `indices = xdata.astype(jnp.int32)`
1. **Line 915 (new):** `return g2_theory_flat[indices]` instead of
   `return g2_theory_flat`
1. **Added docstring note** about respecting xdata size for chunking

______________________________________________________________________

## Implementation

### File to Modify

**`homodyne/optimization/nlsq_wrapper.py`**

### Exact Code Changes

**BEFORE (line 913-916):**

```python
    # Flatten theory to match flattened data (NLSQ expects 1D output)
    g2_theory_flat = g2_theory.flatten()

    return g2_theory_flat
```

**AFTER:**

```python
    # Flatten theory to match flattened data (NLSQ expects 1D output)
    g2_theory_flat = g2_theory.flatten()

    # CRITICAL FIX for curve_fit_large chunking:
    # xdata contains indices into the flattened array.
    # When curve_fit_large chunks the data, it passes subset indices.
    # We must return only those requested points to match ydata chunk size.
    # For full dataset: xdata = [0, 1, ..., n-1] returns all points
    # For chunk: xdata = [0, 1, ..., chunk_size-1] returns subset
    indices = xdata.astype(jnp.int32)
    return g2_theory_flat[indices]
```

______________________________________________________________________

## Testing Plan

### Test 1: Verify Fix with Diagnostic Script

Create test to verify indexed output matches chunk size:

```python
# test_model_chunking.py
import jax.numpy as jnp
import numpy as np

# Simulate model_function with fix
def model_with_fix(xdata, g2_full):
    """Model that respects xdata indices."""
    indices = xdata.astype(jnp.int32)
    return g2_full[indices]

# Test with full dataset
n_total = 23_046_023
g2_full = jnp.ones(n_total)  # Simulated full output

# Test 1: Full dataset (no chunking)
xdata_full = jnp.arange(n_total)
output_full = model_with_fix(xdata_full, g2_full)
assert output_full.shape == (n_total,), f"Full: Expected {n_total}, got {output_full.shape}"
print(f"✓ Full dataset test passed: {output_full.shape}")

# Test 2: Chunk 1 (first 1M points)
chunk_size = 1_000_000
xdata_chunk1 = jnp.arange(chunk_size)
output_chunk1 = model_with_fix(xdata_chunk1, g2_full)
assert output_chunk1.shape == (chunk_size,), f"Chunk1: Expected {chunk_size}, got {output_chunk1.shape}"
print(f"✓ Chunk 1 test passed: {output_chunk1.shape}")

# Test 3: Chunk 2 (second 1M points)
xdata_chunk2 = jnp.arange(chunk_size, 2 * chunk_size)
output_chunk2 = model_with_fix(xdata_chunk2, g2_full)
assert output_chunk2.shape == (chunk_size,), f"Chunk2: Expected {chunk_size}, got {output_chunk2.shape}"
print(f"✓ Chunk 2 test passed: {output_chunk2.shape}")

# Test 4: Last partial chunk
xdata_last = jnp.arange(23_000_000, n_total)
output_last = model_with_fix(xdata_last, g2_full)
assert output_last.shape == (46_023,), f"Last chunk: Expected 46023, got {output_last.shape}"
print(f"✓ Last chunk test passed: {output_last.shape}")

print("\n✅ All chunking tests passed!")
```

### Test 2: Full Integration Test with Real Data

```bash
cd /home/wei/Documents/Projects/data/C020
homodyne --config homodyne_laminar_flow_config.yaml \
         --method nlsq \
         --output-dir homodyne_results_fixed

# Expected results:
# 1. Parameters should change from initial values
# 2. Uncertainties should NOT all be 1.0
# 3. Log should show chunk processing (if NLSQ logging enabled)
# 4. Optimization should complete in 5-10 minutes
```

### Test 3: Verify Covariance Matrix

```python
# check_results.py
import json
import numpy as np

with open('homodyne_results_fixed/nlsq/parameters.json') as f:
    results = json.load(f)

uncertainties = [p['uncertainty'] for p in results['parameters'].values()]

print(f"Uncertainties: {uncertainties}")
print(f"All equal to 1.0? {all(u == 1.0 for u in uncertainties)}")
print(f"Identity matrix fallback? {all(u == 1.0 for u in uncertainties)}")

if not all(u == 1.0 for u in uncertainties):
    print("✅ Fix successful! Covariance computed properly.")
else:
    print("❌ Fix failed. Still using identity matrix.")
```

______________________________________________________________________

## Expected Behavior After Fix

### Before Fix:

- ❌ All chunks fail silently due to shape mismatch
- ❌ success_rate = 0%
- ❌ Fallback to identity covariance (all uncertainties = 1.0)
- ❌ Parameters unchanged from initial values
- ❌ Fast execution (2.6s - no real optimization)

### After Fix:

- ✅ Chunks process successfully
- ✅ success_rate > 50%
- ✅ Proper covariance matrix computed
- ✅ Varied uncertainties (not all 1.0)
- ✅ Parameters optimized from initial values
- ✅ Slower execution (5-10 min - actual optimization running)
- ✅ Log shows chunk progress (if NLSQ logging connected)

______________________________________________________________________

## Memory Considerations

**Question:** Does computing full 23M array defeat memory savings?

**Answer:** Partially, but acceptable:

1. **Forward pass:** Still computes 23M points (no memory savings)

   - Unavoidable given our data structure (all phi angles coupled)
   - JAX handles this efficiently with GPU memory

1. **Jacobian computation:** NLSQ chunks this (memory savings!)

   - Jacobian size: 23M × 9 params = 207M elements = ~1.7GB
   - Chunking reduces peak memory by processing 1M points at a time
   - This is the PRIMARY memory bottleneck NLSQ solves

1. **Net benefit:** Major memory savings from Jacobian chunking outweigh forward pass
   overhead

**Alternative (if memory still issues):**

- Reduce dataset size via phi angle filtering (23M → 8M)
- Enable NLSQ sampling for datasets > 100M points

______________________________________________________________________

## Validation Checklist

After implementing fix:

- [ ] Code compiles without errors
- [ ] Black formatting passes
- [ ] Ruff linting passes
- [ ] Test 1 (chunking logic) passes
- [ ] Test 2 (full integration) completes without errors
- [ ] Test 3 (covariance check) shows non-identity matrix
- [ ] Parameters change from initial values
- [ ] Chi-squared is reasonable (< 2 × data points)
- [ ] Execution time is realistic (5-10 min, not 2.6s)

______________________________________________________________________

## Alternative Fixes (if primary fix insufficient)

### Fix 2: Connect NLSQ Logging to Homodyne

Add after curve_fit_large call in nlsq_wrapper.py:

```python
# After line 412
popt, pcov = curve_fit_large(...)

# Add diagnostic logging
import logging
logger = logging.getLogger(__name__)

if np.allclose(pcov, np.eye(len(popt))):
    logger.error(
        "❌ curve_fit_large returned identity covariance matrix!\n"
        "   This indicates chunked fitting failed.\n"
        "   Possible causes:\n"
        "   1. Model function shape mismatch with chunks\n"
        "   2. All chunks failed during processing\n"
        "   3. NLSQ internal error\n"
        f"   Parameters unchanged: {np.allclose(popt, p0)}"
    )
```

### Fix 3: Force Single-Chunk Mode for Testing

Temporarily disable chunking to verify non-chunked path works:

```python
# In nlsq_wrapper.py line 215
use_large = False  # Force curve_fit instead of curve_fit_large
```

______________________________________________________________________

## Sign-Off

**Diagnosed By:** Claude Code **Root Cause:** Model ignores xdata size, returns fixed
23M array **Fix:** Index model output by xdata to respect chunk size **Files Modified:**
`homodyne/optimization/nlsq_wrapper.py` (2 lines changed) **Testing:** 3-stage
validation plan provided **Status:** ✅ Ready for implementation and testing

______________________________________________________________________

*End of Fix Plan*
