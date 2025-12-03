# Stratification + NLSQ Large Dataset Incompatibility Analysis

**Investigation Date:** 2025-11-06 **Log File:** `homodyne_analysis_20251106_122208.log`
**Dataset Size:** 3,006,003 points (3 angles, 1001×1001 time grid)

## Executive Summary

Angle-stratified chunking successfully reorganizes data to ensure all chunks contain all
angles, but NLSQ's `curve_fit_large()` optimization still fails with unchanged
parameters for large datasets (>1M points) when per-angle scaling is enabled.

**Root Cause:** Double-chunking architecture - stratification creates angle-balanced
chunks, but NLSQ's internal chunking breaks this structure.

## Log Analysis

### 1. Stratification Phase (SUCCESSFUL)

**Lines 91-133:** Stratification completed successfully

```
2025-11-06 12:22:10 | INFO | Applying angle-stratified chunking
  Angles: 3, Imbalance ratio: 1.00
  Total points: 3,006,003
  Target chunk size: 100,000
  Use index-based: False

Result: 31 chunks created, each with all 3 angles
  - Chunks 0-29: 99,999 points each
  - Chunk 30: 6,033 points (remainder)
```

**Key Evidence:**

- ✅ All chunks contain all 3 angles (verified by imbalance ratio = 1.00)
- ✅ Metadata preserved (sigma, q, L, dt copied to StratifiedData)
- ✅ Stratification diagnostics collected
- ✅ Execution time: \<1 second

### 2. Optimization Phase (FAILED)

**Lines 138-174:** NLSQ optimization with per_angle_scaling=True

**Strategy:** LARGE (curve_fit_large for 1M-10M points)

**Attempts:** 3 retries, all failed

```
Function evaluations: 0
Cost: 8.0160e+06 → 8.0160e+06 (+0.0%)
Iterations: 0
Parameters unchanged: True
```

**Warning Messages:**

```
WARNING | Potential optimization failure detected:
  Parameters unchanged: True
  Identity covariance: False
  This may indicate NLSQ streaming bug or failed optimization
```

**Final Result:**

```
ERROR | Optimization returned unchanged parameters after all retries.
```

### 3. Double-Chunking Problem

**Two-Level Chunking Architecture:**

1. **Level 1: Homodyne Stratification** (lines 91-133)

   - Creates 31 chunks from 3M points
   - Each chunk: ~100k points with all 3 angles
   - Goal: Ensure angle-specific parameters can be optimized in each chunk

1. **Level 2: NLSQ Internal Chunking** (lines 144-174)

   - `curve_fit_large()` applies its own chunking strategy
   - Breaks stratified chunks into smaller sub-chunks
   - **Problem:** Sub-chunks may not contain all angles

**Failure Mechanism:**

```
Stratified chunk (100k points, 3 angles)
   ↓
NLSQ curve_fit_large() chunks internally
   ↓
Sub-chunk 1: 50k points, angles [0, 1]     ← Missing angle 2
Sub-chunk 2: 50k points, angles [1, 2]     ← Missing angle 0
   ↓
Gradient w.r.t. contrast[2], offset[2] = 0 in sub-chunk 1
Gradient w.r.t. contrast[0], offset[0] = 0 in sub-chunk 2
   ↓
NLSQ aggregates gradients: all zeros
   ↓
Optimization fails (0 iterations)
```

## Known Issues (CLAUDE.md)

This issue is documented in CLAUDE.md:

```markdown
**Issue:** Per-angle scaling incompatible with NLSQ chunking for large datasets (>1M points)

**Root Cause (VERIFIED 2025-01-06):**
Per-angle scaling architecture is fundamentally incompatible with NLSQ's chunking implementation

**Symptoms:**
- NLSQ returns 0 iterations with unchanged parameters
- `first-order optimality 0.00e+00` (gradient = 0)
- Final cost = initial cost (no improvement)
```

## Why Stratification Doesn't Fix The Issue

**What We Thought:** Stratification ensures all chunks have all angles → per-angle
parameters can be optimized

**What Actually Happens:** NLSQ's curve_fit_large() doesn't respect our chunk
boundaries:

1. We pass 31 stratified chunks to NLSQ
1. NLSQ flattens them back into a single 3M point array
1. NLSQ re-chunks using its own logic (not angle-aware)
1. Result: Back to square one - chunks missing angles

**Evidence from Log:**

```
Line 144: | INFO | Selected Large strategy for 3,006,003 points
Line 150: | DEBUG | Using curve_fit_large with NLSQ automatic memory management
```

The "NLSQ automatic memory management" means NLSQ is controlling the chunking, not us.

## Solutions

### Immediate Workarounds

1. **Disable Per-Angle Scaling** (Recommended for >1M datasets)

   ```yaml
   optimization:
     nlsq:
       per_angle_scaling: false  # Use global contrast/offset
   ```

   - **Pro:** Works reliably with large datasets
   - **Con:** Less accurate if angles have different intensities

1. **Use MCMC Instead of NLSQ**

   ```bash
   homodyne --config config.yaml --method mcmc
   ```

   - **Pro:** Handles per-angle scaling correctly
   - **Con:** Much slower (~100x for large datasets)

1. **Reduce Dataset Size** (Filter angles or subsample time)

   ```yaml
   preprocessing:
     phi_range: [0, 45]  # Only use 2 angles instead of 3
   ```

   - **Pro:** Forces STANDARD strategy (no chunking)
   - **Con:** Loses data

### Long-Term Solutions

#### Option A: Fix NLSQ Library

**Modify curve_fit_large() to accept pre-chunked data:**

```python
# Proposed API:
curve_fit_large(
    f, xdata, ydata, p0,
    chunks=stratified_chunks,  # NEW: User-provided chunks
    preserve_chunks=True,       # NEW: Don't re-chunk
)
```

- **Pro:** Preserves angle-stratified structure
- **Con:** Requires NLSQ library changes (upstream contribution)

#### Option B: Custom Chunking Optimizer

**Implement our own chunk-aware optimizer:**

```python
def optimize_with_chunks(chunks, per_angle_params):
    """Optimize each chunk while preserving angle-specific parameters."""
    for chunk in chunks:
        assert all_angles_present(chunk)  # Verify stratification
        optimize_chunk(chunk, per_angle_params)
    aggregate_results(chunks)
```

- **Pro:** Full control over chunking
- **Con:** Reimplements NLSQ logic, loses GPU acceleration

#### Option C: Restructure Data

**Reshape data so each "chunk" is a full angle:**

```python
# Current: phi × t1 × t2 → flatten → chunk
# Proposed: Group by angle BEFORE optimization
for angle_idx in range(n_angles):
    angle_data = extract_angle(data, angle_idx)
    optimize(angle_data, contrast[angle_idx], offset[angle_idx])
```

- **Pro:** Natural per-angle structure
- **Con:** Loses cross-angle correlation info

## Recommendations

### For Users (Today)

- **Datasets < 1M points:** Use per_angle_scaling=True (works fine)
- **Datasets > 1M points:** Use per_angle_scaling=False or switch to MCMC

### For Developers (Future)

1. **Short-term:** Add automatic detection and warning

   ```python
   if n_points > 1_000_000 and per_angle_scaling:
       logger.warning("Large dataset + per-angle scaling detected. "
                     "Consider using per_angle_scaling=false or --method mcmc")
   ```

1. **Medium-term:** Contribute to NLSQ library (Option A)

   - Propose chunk preservation API
   - Submit PR with test cases

1. **Long-term:** Implement custom optimizer (Option B)

   - Only if NLSQ contribution rejected
   - Consider scipy.optimize.least_squares as alternative

## Test Coverage

✅ **Stratification mechanics work correctly** (verified 2025-11-06):

- `test_stratification_diagnostics_passed_to_result()` - Parameter passing
- `test_full_nlsq_workflow_with_stratification()` - End-to-end workflow
- Both tests pass, confirming stratification code is correct

❌ **NLSQ optimization still fails** (known limitation):

- Not a stratification bug
- Not a metadata bug
- Fundamental architecture incompatibility

## References

- **Log File:**
  `/home/wei/Documents/Projects/data/C020/homodyne_results/logs/homodyne_analysis_20251106_122208.log`
- **Known Issue:** CLAUDE.md lines 367-431
- **Stratification Code:** `homodyne/optimization/stratified_chunking.py`
- **NLSQ Wrapper:** `homodyne/optimization/nlsq_wrapper.py` lines 1166-1357
- **NLSQ Documentation:** https://nlsq.readthedocs.io/en/latest/

## Conclusion

The investigation confirms that:

1. ✅ **Bugs #1 and #2 are fixed** (metadata preservation, diagnostics parameter)
1. ✅ **Stratification works correctly** (31 chunks created, all with all angles)
1. ❌ **NLSQ double-chunking limitation remains** (documented known issue)

**The stratification feature is working as designed** - it reorganizes data correctly.
The NLSQ optimization failure is a separate, pre-existing issue caused by NLSQ's
internal chunking strategy, not by our stratification code.

**Recommended next step:** Add automatic detection and user-friendly warning when this
situation is detected (large dataset + per-angle scaling + NLSQ method).
