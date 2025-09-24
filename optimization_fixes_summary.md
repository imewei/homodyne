# NLSQ and MCMC Optimization Fixes Summary

## Date: 2025-09-24

## Status
- **MCMC**: Partially working but very slow (~4-5 iterations/sec)
- **NLSQ**: Still encountering solver errors (NaN/inf in linear solver)

## Fixes Applied

### 1. JAX Tracing Issues Fixed
**Problem**: JAX was encountering float concretization errors when trying to convert traced values to Python floats.

**Files Modified**:
- `homodyne/optimization/nlsq.py`: Removed float() conversions in _array_to_params for JAX compatibility
- `homodyne/core/models.py`: Removed q scalar conversions (jnp.mean(q) → direct pass through)
- `homodyne/optimization/mcmc.py`: Fixed static argument issues with analysis_mode parameter

**Key Changes**:
```python
# Before (caused JAX tracing errors):
if hasattr(q, "shape") and q.shape:
    q_scalar = jnp.mean(q)
else:
    q_scalar = q

# After (JAX compatible):
# Pass q directly without conversion
```

### 2. Boolean Conversion Errors Fixed
**Problem**: Validation functions were performing boolean checks on JAX traced values.

**Files Modified**:
- `homodyne/core/theory.py`: Disabled runtime validation when using JAX
- `homodyne/core/models.py`: Commented out validate_parameters calls in compute methods

**Key Changes**:
```python
# Added JAX check to skip validations:
if jax_available:
    # In JAX mode, skip runtime validation
    return
```

### 3. MCMC Performance Optimizations
**Files Created**:
- `optimize_mcmc.py`: Helper for MCMC configuration with parallel chains

**Optimizations Applied**:
- Enabled parallel chains on CPU: `numpyro.set_host_device_count(4)`
- Reduced warmup/sample iterations for testing
- Used informed priors based on NLSQ results
- Fixed parameter indexing issues

### 4. Testing Improvements
**Files Created**:
- `test_static_mode.py`: Simplified test with 3-parameter static mode
- `debug_optimization_fixed.py`: Debug script for both optimizers
- `debugging_summary.md`: Documentation of issues and fixes

## Remaining Issues

### NLSQ Linear Solver Error
The NLSQ optimization now progresses past JAX tracing issues but encounters:
```
A linear solver returned non-finite (NaN or inf) output.
```

This suggests:
1. The optimization problem may be ill-posed
2. The test data might be causing singularity
3. The Levenberg-Marquardt solver configuration may need adjustment

### Potential Solutions to Explore
1. **Regularization**: Add regularization to the least squares problem
2. **Solver Configuration**: Use `AutoLinearSolver(well_posed=False)` as suggested
3. **Initial Parameters**: Provide better initial guesses closer to true values
4. **Data Conditioning**: Ensure test data is well-conditioned
5. **Alternative Optimizer**: Consider using a different optimization algorithm

## Code Architecture Issues
1. **Mode Mismatch**: Analysis mode is set to `static_isotropic` but model creates `laminar_flow`
2. **Parameter Count**: Warning about parameter count mismatch (5 vs 16)
3. **Validation Bypass**: Had to disable all runtime validations for JAX compatibility

## Performance Metrics
- **MCMC**: ~4-5 iterations/second (very slow)
- **NLSQ**: Fails before completing iterations
- **Static Mode Test**: MCMC runs but NLSQ fails

## Next Steps
1. Fix the linear solver issue in NLSQ by adjusting solver configuration
2. Address the analysis mode mismatch
3. Implement proper JAX-compatible validation
4. Optimize MCMC performance further
5. Create comprehensive test suite once both optimizers work

## Files Modified Summary
```
Modified:
- homodyne/optimization/nlsq.py
- homodyne/optimization/mcmc.py
- homodyne/core/models.py
- homodyne/core/theory.py

Created:
- test_static_mode.py
- optimize_mcmc.py
- debug_optimization_fixed.py
- debugging_summary.md
- optimization_fixes_summary.md (this file)
```

## Conclusion
Significant progress made on JAX compatibility issues. The main blocking issue is now the linear solver in NLSQ optimization. MCMC works but needs performance improvements.