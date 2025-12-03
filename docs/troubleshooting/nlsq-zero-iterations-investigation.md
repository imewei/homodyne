# NLSQ Zero Iterations Investigation (2025-11-05)

## Executive Summary

**Status**: Unresolved - Deep architectural incompatibility identified **Symptom**: NLSQ
optimization consistently returns 0 iterations with unchanged parameters **Impact**:
Laminar flow analysis with >1M points cannot be optimized **Root Cause**: JAX JIT
compilation incompatibility with model function architecture

______________________________________________________________________

## Problem Statement

NLSQ optimization for laminar flow analysis (3,006,003 points, 9 parameters) fails with:

- **0 iterations** across all retry attempts
- **Parameters unchanged** from initial values
- **Gradient = 0.00** (first-order optimality immediately satisfied)
- **Final cost = Initial cost** (no improvement)
- **Analysis completes** but with un-optimized parameters

______________________________________________________________________

## Investigation Timeline

### Initial Diagnosis (Original Bug Report)

- **Symptom**: `TracerArrayConversionError` during optimization
- **Location**: `nlsq_wrapper.py:1301-1302`
- **Cause**: Dynamic array indexing `g2_theory_flat[xdata.astype(int)]` with JAX tracer

```python
# ❌ PROBLEMATIC CODE:
indices = xdata.astype(jnp.int32)
return g2_theory_flat[indices]  # Tracer cannot be converted to int
```

### Attempted Fix #1: Return Full Array

- **Approach**: Remove dynamic indexing, return full array
- **Result**: ✗ Shape mismatch error from `curve_fit_large`
- **Error**: `Model output (3M points) != xdata size (100 points)`
- **Lesson**: NLSQ's chunking expects chunk-sized output

### Attempted Fix #2: Static Slicing `[:chunk_size]`

- **Approach**: Return first N elements where N = len(xdata)
- **Result**: ✗ Still 0 iterations, gradient = 0.00
- **Observation**: Returns wrong data (first chunk for all chunks)
- **Lesson**: Model needs to know WHICH chunk, not just chunk size

### Attempted Fix #3: Force STANDARD Strategy

- **Approach**: Disable chunking via `strategy_override: "standard"`
- **Result**: ✗ Still 0 iterations, gradient = 0.00
- **Observation**: Even non-chunked `curve_fit` has same issue
- **Lesson**: Problem is deeper than chunking - JIT itself is problematic

### Attempted Fix #4: Return Full Array (Standard Strategy)

- **Approach**: Combine full array return + standard strategy
- **Result**: ✗ Still 0 iterations, gradient = 0.00
- **Observation**: `gtol` termination on first evaluation
- **Lesson**: Gradient computation is fundamentally broken

______________________________________________________________________

## Root Cause Analysis

### The Core Problem

The model function architecture is **fundamentally incompatible** with NLSQ's JIT
compilation:

```python
def model_function(xdata, *params):
    # 1. Compute theory for ALL data (3M points)
    g2_theory_flat = compute_full_theory(params)  # (3,006,003,)

    # 2. NLSQ expects: return ONLY requested chunk
    #    But xdata is a JAX tracer during JIT compilation
    #    Cannot determine which data to return without concrete indices

    # 3. Options (all fail):
    #    ❌ g2_theory_flat[xdata]  → TracerArrayConversionError
    #    ❌ g2_theory_flat[:len(xdata)]  → Returns wrong chunk
    #    ❌ return g2_theory_flat  → Shape mismatch or zero gradient
```

### Why Gradient is Zero

NLSQ reports `first-order optimality 0.00e+00` because:

1. **Model returns constant**: Full array return means model output doesn't vary
   correctly with chunk requests
1. **Gradient computation fails**: JAX can't compute gradients when model/data size
   mismatch
1. **JIT compilation artifact**: Traced computations produce degenerate gradients

### Evidence

All test runs show identical symptoms:

```
Function evaluations 1, initial cost 1.1330e+06, final cost 1.1330e+06
first-order optimality 0.00e+00
Convergence: reason=`gtol` termination condition is satisfied
iterations=None
Final χ² = 8.0160e+06 (unchanged across all runs)
```

______________________________________________________________________

## Architectural Analysis

### Current Model Structure

```
Data Flow:
  xdata (indices) → model_function → ydata (values)

Model Computes:
  1. Theory for ALL phi angles (3 angles)
  2. Theory for ALL time points (1001 × 1001)
  3. Flatten to (3,006,003,) array
  4. ??? Return subset ??? → IMPOSSIBLE with JAX tracer
```

### Why This Fails with JAX JIT

1. **NLSQ JIT-compiles model**: `xdata` becomes symbolic tracer
1. **Dynamic indexing forbidden**: `array[tracer]` → TracerArrayConversionError
1. **Chunk identification impossible**: Can't determine which chunk without concrete
   indices
1. **Full array breaks gradients**: Returning full array when chunk requested = gradient
   mismatch

______________________________________________________________________

## Proposed Solutions

### Option 1: Disable JIT Compilation ⚠️ Performance Impact

```python
# In NLSQ wrapper, pass jit=False to optimizer
# WARNING: May significantly slow down optimization
result = nlsq.curve_fit(model_fn, xdata, ydata, p0=p0, jit=False)
```

**Pros**: Avoids tracer issues entirely **Cons**: Orders of magnitude slower, defeats
purpose of JAX

### Option 2: Restructure Model to be Chunk-Aware ⚙️ Major Refactor

```python
def chunk_aware_model(xdata, *params):
    # xdata now encodes (phi_idx, t1_idx, t2_idx) tuples
    # Compute ONLY requested data points, not full array
    phi_indices, t1_indices, t2_indices = decode_xdata(xdata)
    g2_values = compute_g2_for_indices(params, phi_indices, t1_indices, t2_indices)
    return g2_values
```

**Pros**: Truly chunk-compatible, efficient **Cons**: Requires complete rewrite of data
pipeline

### Option 3: Alternative Optimizer 🔄 Library Change

```python
# Use scipy.optimize.least_squares or custom optimizer
# without JAX JIT dependency
from scipy.optimize import least_squares

result = least_squares(residual_fn, p0, bounds=bounds, method='trf')
```

**Pros**: Avoids JAX JIT issues entirely **Cons**: Loses GPU acceleration, may be slower

### Option 4: Manual Gradient Descent 🛠️ Custom Implementation

```python
# Implement Levenberg-Marquardt manually with JAX autodiff
# but without JIT compilation of model function
```

**Pros**: Full control over gradient computation **Cons**: Significant development
effort, maintenance burden

______________________________________________________________________

## Recommended Next Steps

### Short-term (Immediate)

1. **Document limitation** in CLAUDE.md known issues
1. **Add warning** when laminar_flow + large dataset detected
1. **Suggest phi angle filtering** to reduce dataset size below 1M points

### Medium-term (1-2 weeks)

1. **Test Option 3**: Evaluate scipy.optimize.least_squares performance
1. **Benchmark**: Compare scipy vs NLSQ (with jit=False) vs current state
1. **Prototype**: Chunk-aware architecture proof-of-concept

### Long-term (1-2 months)

1. **Implement Option 2**: Full chunk-aware model restructure
1. **Add tests**: Integration tests for large datasets
1. **Performance**: Profile and optimize new architecture

______________________________________________________________________

## Workarounds for Users

### Workaround 1: Reduce Dataset Size

```yaml
# In config.yaml
phi_filtering:
  enabled: true
  target_ranges:
    - {min_angle: -5, max_angle: 5}    # Fewer angles
    - {min_angle: 85, max_angle: 95}

data:
  time_range: [500, 1500]  # Fewer time points (1000 → 500)
```

**Target**: < 1M points to avoid LARGE strategy

### Workaround 2: Force STANDARD Strategy (Current Approach)

```yaml
# In config.yaml
performance:
  strategy_override: "standard"  # Use curve_fit instead of curve_fit_large
```

**Note**: Still has zero-iteration issue but completes analysis

### Workaround 3: Use Static Isotropic Mode

```bash
# Laminar flow has 9 parameters, static has 5
# Smaller parameter space may help
homodyne --config config.yaml --mode static_isotropic
```

______________________________________________________________________

## Test Cases for Future Solutions

### Test 1: Small Dataset (Should Work)

- **Size**: < 1M points (e.g., 1 angle × 1001 × 1001 = 1,001,001)
- **Expected**: Optimization runs, iterations > 0
- **Status**: ⚠️ Currently fails (needs verification)

### Test 2: Large Dataset (Currently Fails)

- **Size**: 3M points (3 angles × 1001 × 1001)
- **Expected**: Optimization runs with chunking
- **Status**: ❌ Fails with 0 iterations

### Test 3: Standard Strategy Override

- **Size**: 3M points with `strategy_override: "standard"`
- **Expected**: Optimization runs without chunking
- **Status**: ❌ Fails with 0 iterations (gradient=0)

______________________________________________________________________

## References

- **Original Bug**:
  `/home/wei/Documents/Projects/data/C020/homodyne_results/logs/homodyne_analysis_20251104_173024.log`
- **Investigation Logs**: `homodyne_analysis_20251105_*.log`
- **NLSQ Docs**: https://nlsq.readthedocs.io/en/latest/
- **JAX Tracer Error**:
  https://jax.readthedocs.io/en/latest/errors.html#jax.errors.TracerArrayConversionError

______________________________________________________________________

## Conclusion

This issue represents a **fundamental architectural incompatibility** between:

1. Homodyne's "compute-full-then-index" model structure
1. NLSQ's JIT-compiled chunking expectations
1. JAX's tracer-based JIT compilation

**No simple fix exists**. Resolution requires either:

- Major architectural refactor (Option 2)
- Alternative optimizer (Option 3)
- Accepting performance loss (Option 1)

Until resolved, users must work around by reducing dataset sizes or accepting
unoptimized results.

______________________________________________________________________

**Last Updated**: 2025-11-05 **Investigated By**: Claude Code Debugging Session
**Status**: Documented, Awaiting Architectural Decision
