# NLSQ and MCMC NUTS Implementation Debug Report

## Current Status

Both optimization methods are installed and partially functional but have implementation issues that need fixing.

## Identified Issues

### 1. NLSQ (Optimistix) Issues

**Primary Issue**: JAX Concretization Error
```
Abstract tracer value encountered where concrete value is expected
```

**Root Cause**: The `q` parameter is being passed as a traced value but needs to be concrete for certain operations.

**Solution**:
- Ensure `q` is extracted from data properly and converted to a scalar
- Mark certain arguments as static in JIT compilation

### 2. MCMC (NumPyro) Issues

**Primary Issue**: Static Argument Error
```
Error interpreting argument to <function _compute_simple_theory> as an abstract array
```

**Root Cause**: `analysis_mode` string is being traced by JAX but should be static.

**Solution**:
- Mark `analysis_mode` as static in the JIT-compiled function
- Fix the likelihood function to handle parameters correctly

## Step-by-Step Debugging Instructions

### Step 1: Fix NLSQ Implementation

1. Check the residual function in `nlsq.py`:
```bash
grep -n "def.*residual" homodyne/optimization/nlsq.py
```

2. Ensure proper scalar extraction for `q`:
```python
# In the residual function
q_scalar = float(data.get('wavevector_q_list', [0.0054])[0])
```

3. Fix attribute name:
```python
# Change 'iterations' to 'n_iterations' in NLSQResult
```

### Step 2: Fix MCMC Implementation

1. Check the likelihood function:
```bash
grep -n "def.*likelihood\|def.*_compute_simple_theory" homodyne/optimization/mcmc.py
```

2. Add static_argnums for JIT compilation:
```python
@jax.jit(static_argnums=(7,))  # Mark analysis_mode as static
def _compute_simple_theory(params, t1, t2, phi, q, L, contrast, analysis_mode):
    ...
```

### Step 3: Test Individual Components

1. Test JAX backend directly:
```python
import jax.numpy as jnp
from homodyne.core.jax_backend import compute_g2_scaled

# Test with simple inputs
params = jnp.array([10000.0, -1.5, 100.0])
t1 = jnp.ones((10, 10))
t2 = jnp.ones((10, 10))
phi = jnp.array([0.0])
q = 0.0054  # Ensure this is a scalar

result = compute_g2_scaled(params, t1, t2, phi, q, 100.0)
print(f"Result shape: {result.shape}")
```

### Step 4: Verify Dependencies

```python
import optimistix
import numpyro
import blackjax
import jax

print(f"JAX devices: {jax.devices()}")
print(f"Optimistix available: {hasattr(optimistix, 'LevenbergMarquardt')}")
print(f"NumPyro NUTS: {hasattr(numpyro.infer, 'NUTS')}")
```

## Quick Fixes

### Fix 1: NLSQ Scalar Conversion
```python
# In homodyne/optimization/nlsq.py, around line 300-350
def _create_residual_function(self, data, config):
    # Extract q as concrete scalar
    q_list = data.get('wavevector_q_list', [0.0054])
    q_scalar = float(q_list[0]) if hasattr(q_list, '__getitem__') else float(q_list)

    # ... rest of function
```

### Fix 2: MCMC Static Arguments
```python
# In homodyne/optimization/mcmc.py, around line 400-450
# Use partial to fix static arguments
from functools import partial

theory_fn = partial(_compute_simple_theory, analysis_mode=self.analysis_mode)
```

## Testing Protocol

1. **Unit Test JAX Backend**:
```bash
python -m pytest tests/unit/test_jax_backend.py -v
```

2. **Test NLSQ Separately**:
```bash
python -c "
from homodyne.optimization.nlsq import fit_nlsq_jax
# Minimal test here
"
```

3. **Test MCMC Separately**:
```bash
python -c "
from homodyne.optimization.mcmc import fit_mcmc_jax
# Minimal test here
"
```

## Performance Optimization

1. **Enable GPU for parallel chains**:
```python
import numpyro
numpyro.set_host_device_count(4)  # For CPU parallel chains
```

2. **JIT Compilation**:
- Ensure all inner loops are JIT-compiled
- Avoid Python loops inside JIT functions

3. **Memory Management**:
- Use chunked processing for large datasets
- Clear JAX cache periodically: `jax.clear_caches()`

## Monitoring and Logs

Check logs for detailed error information:
```bash
# Enable debug logging
export HOMODYNE_LOG_LEVEL=DEBUG
python debug_optimization_fixed.py

# Check JAX debugging
export JAX_DEBUG_NANS=1
export JAX_TRACEBACK_FILTERING=off
```

## Next Steps

1. Apply the fixes to `nlsq.py` and `mcmc.py`
2. Run the debug script again
3. Verify convergence and parameter recovery
4. Run full test suite
5. Profile performance

## References

- [JAX Concretization Errors](https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError)
- [NumPyro MCMC Guide](https://num.pyro.ai/en/latest/mcmc.html)
- [Optimistix Documentation](https://docs.kidger.site/optimistix/)