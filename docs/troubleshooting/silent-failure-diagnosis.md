# Diagnosing Silent Optimization Failures

**Symptom**: Optimization completes with 0 iterations, parameters unchanged, identity
covariance matrix

This guide provides a systematic approach to diagnosing optimization failures that
complete without errors but produce incorrect results.

## Quick Diagnosis Checklist

- [ ] Check iteration count in logs (should be > 0)
- [ ] Inspect covariance matrix (all 1.0 = identity = fallback)
- [ ] Verify parameters changed from initial values
- [ ] Check chi-squared progression (should improve)
- [ ] Examine execution time (too fast = suspicious)

## Common Symptoms

### 1. Identity Covariance Matrix

```json
{
  "parameters": {
    "contrast": {"value": 0.5, "uncertainty": 1.0},
    "offset": {"value": 1.0, "uncertainty": 1.0},
    "D0": {"value": 1324.1, "uncertainty": 1.0}
  }
}
```

**All uncertainties = 1.0** indicates fallback behavior, not real uncertainty estimates.

### 2. Zero Iterations

```
INFO: Optimization completed in 2.62s, 0 iterations
```

Real optimization should take minutes and show iteration progress.

### 3. Parameters Unchanged

Initial and final parameters are identical, suggesting no optimization occurred.

## Diagnostic Approach

### Step 1: Verify Gradient Computation

Create a test script to compute gradient norm:

```python
import jax
import jax.numpy as jnp

# Load your data and initial parameters
params = jnp.array([0.5, 1.0, 1324.1, 0.5, 10.0, ...])

# Compute residual at initial point
residuals = your_residual_function(params)

# Compute gradient
gradient = jax.grad(lambda p: jnp.sum(residuals**2))(params)
gradient_norm = jnp.linalg.norm(gradient)

print(f"Gradient norm: {gradient_norm:.6e}")
print(f"gtol threshold: 1e-6")

if gradient_norm > 1e-6:
    print("✓ Gradient is non-zero, optimization should proceed")
else:
    print("✗ Gradient too small, at local minimum")
```

**Expected**: Gradient norm should be >> gtol (typically > 1e-3 for initial guess)

### Step 2: Test with Smaller Dataset

```python
# Reduce dataset size to isolate large-dataset issues
data_small = {
    "phi_angles_list": data["phi_angles_list"][:5],  # First 5 angles only
    "c2_exp": data["c2_exp"][:5],
    # ... other fields
}

result = nlsq_wrapper.fit(data_small, config)
```

If small dataset works but large fails → large-dataset-specific issue (memory, chunking,
etc.)

### Step 3: Check API Compatibility

Verify return value unpacking matches NLSQ API:

```python
# For curve_fit (standard):
try:
    popt, pcov, info = curve_fit(..., full_output=True)
    print("✓ curve_fit with full_output works")
except ValueError as e:
    print(f"✗ Unpacking error: {e}")

# For curve_fit_large:
try:
    popt, pcov = curve_fit_large(...)  # Returns only 2 values!
    info = {}  # Must create manually
    print("✓ curve_fit_large unpacking correct")
except ValueError as e:
    print(f"✗ Unpacking error: {e}")
```

### Step 4: Enable Verbose Logging

```python
# In your optimization call:
result = nlsq_wrapper.fit(
    data,
    config,
    verbose=2  # Maximum verbosity
)

# Check logs for:
# - Chunk processing messages
# - Success rates
# - Internal errors or warnings
```

## Common Root Causes

### 1. curve_fit_large Fallback Behavior

**Symptom**: Fast completion (seconds), identity covariance, 0 iterations

**Cause**: Internal chunking failure causing LargeDatasetFitter to return incomplete
result object:

- Missing `result.popt` attribute
- Missing `result.pcov` attribute
- curve_fit_large falls back to identity matrix

**Diagnosis**:

```python
# Add diagnostic logging:
popt, pcov = curve_fit_large(...)

if np.allclose(pcov, np.eye(len(popt))):
    logger.error("Identity covariance detected - chunking likely failed")
```

**Fix**: Check model function compatibility with chunking (see next section)

### 2. Model Function Chunking Incompatibility

**Symptom**: Shape mismatch errors in logs, all chunks fail

**Cause**: Model function ignores `xdata` chunk size:

```python
# ❌ WRONG: Returns fixed size regardless of xdata
def model_function(xdata, *params):
    full_output = compute_all_23M_points(params)
    return full_output  # Always 23M, ignores xdata size

# ✅ CORRECT: Respects xdata indices
def model_function(xdata, *params):
    full_output = compute_all_23M_points(params)
    indices = xdata.astype(jnp.int32)
    return full_output[indices]  # Returns only requested points
```

**Diagnosis**:

- Look for "shape mismatch" in logs
- Check if model output size matches ydata chunk size
- Test with NLSQ shape validation (available in v0.1.4+)

**Fix**: Make model function respect `xdata` as indices into output array

### 3. API Incompatibility (Unpacking Mismatch)

**Symptom**: `ValueError: not enough values to unpack (expected 3, got 2)`

**Cause**: Using curve_fit unpacking pattern with curve_fit_large:

```python
# ❌ WRONG for curve_fit_large:
popt, pcov, info = curve_fit_large(...)  # Returns only 2 values!

# ✅ CORRECT:
popt, pcov = curve_fit_large(...)
info = {}  # Create manually if needed
```

**Fix**: Use 2-value unpacking for curve_fit_large, create empty info dict for
compatibility

### 4. Immediate Convergence (Rare)

**Symptom**: Reduced chi-squared ≈ 1.0, but suspicious given initial guess

**Cause**: Initial parameters accidentally optimal (or data/model mismatch)

**Diagnosis**:

```python
# Perturb initial parameters and check chi-squared
import numpy as np

chi2_initial = compute_chi_squared(params_initial, data)

params_perturbed = params_initial * (1 + 0.2 * np.random.randn(len(params_initial)))
chi2_perturbed = compute_chi_squared(params_perturbed, data)

if chi2_perturbed >> chi2_initial:
    print("Initial guess is suspiciously good")
else:
    print("Chi-squared landscape is flat")
```

**Fix**:

- Try different initial guess
- Verify data quality
- Check parameter bounds aren't too tight

## Recovery Strategies

### For curve_fit_large Fallback:

1. **Switch to standard curve_fit** (if memory allows):

   ```python
   # In nlsq_wrapper.py:
   use_large = False  # Force curve_fit instead
   ```

1. **Reduce dataset size** via angle filtering:

   ```yaml
   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
   ```

1. **Enable NLSQ sampling** (for >100M points):

   ```python
   curve_fit_large(..., sampling_fraction=0.1)  # Use 10% of data
   ```

### For Model Chunking Issues:

1. **Add shape validation** before optimization:

   ```python
   # Test model output size
   test_xdata = jnp.arange(100)
   test_output = model_function(test_xdata, *initial_params)
   assert test_output.shape[0] == 100, "Model must respect xdata size"
   ```

1. **Fix model indexing**:

   ```python
   def model_function(xdata, *params):
       full_output = compute_theory(params)
       return full_output[xdata.astype(jnp.int32)]  # Index by xdata
   ```

### For API Incompatibility:

1. **Check NLSQ API version**:

   ```python
   import nlsq
   print(nlsq.__version__)  # Ensure >= 0.1.3
   ```

1. **Use correct unpacking**:

   ```python
   if use_large:
       popt, pcov = curve_fit_large(...)
       info = {}
   else:
       popt, pcov, info = curve_fit(..., full_output=True)
   ```

## Prevention Best Practices

1. **Always validate model function** before optimization:

   - Test with small dataset first
   - Verify output shape matches input
   - Check gradient is non-zero

1. **Enable verbose logging**:

   ```python
   result = optimizer.fit(data, config, verbose=2)
   ```

1. **Check result quality**:

   ```python
   # After optimization:
   if np.allclose(pcov, np.eye(len(popt))):
       logger.warning("Identity covariance - optimization may have failed")

   if result.nit == 0:
       logger.warning("Zero iterations - optimization didn't run")

   if np.allclose(popt, initial_params):
       logger.warning("Parameters unchanged - no optimization occurred")
   ```

1. **Use NLSQ best practices**:

   - Import nlsq before JAX (for automatic x64)
   - Enable XLA preallocation
   - Set appropriate memory limits
   - Disable traceback filtering for debugging

## References

- **Full diagnostic report**:
  `docs/archive/2025-10-nlsq-integration/NLSQ_0_ITERATIONS_DIAGNOSIS.md`
- **Fix implementation**: Search git log for "Oct 17, 2025"
- **NLSQ documentation**: https://nlsq.readthedocs.io/en/latest/
- **Related guide**: `silent-failure-diagnosis.md` (this file)

## Example Diagnostic Session

```bash
# 1. Check logs for symptoms
grep "iterations" logs/homodyne_analysis_*.log
grep "Identity covariance" logs/homodyne_analysis_*.log

# 2. Verify gradient
python test_gradient_diagnostic.py

# 3. Test with small dataset
homodyne --config config.yaml --method nlsq --output-dir test_small \
         --override "phi_filtering.enabled=true"

# 4. Enable verbose logging
homodyne --config config.yaml --method nlsq --verbose --output-dir test_verbose

# 5. Review results
python check_results.py test_verbose/nlsq/parameters.json
```

______________________________________________________________________

**Last Updated**: November 17, 2025 **Applies to**: homodyne v2.x with NLSQ >= 0.1.3
