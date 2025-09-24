# NLSQ and MCMC Implementation Debug Summary

## Status After Fixes

### NLSQ (Optimistix)
- **Partial Fix Applied**: Extracted q parameter as concrete scalar outside residual function
- **Remaining Issue**: Float concretization error still occurring, likely in theory engine
- **Next Steps**:
  1. Check `theory_engine.compute_g2()` for float conversions
  2. Ensure all scalar parameters are pre-extracted
  3. Consider using `jnp.asarray()` instead of `float()` conversions

### MCMC (NumPyro)
- **Fixed**: Static argument issue with `analysis_mode`
- **Fixed**: Parameter indexing in simplified theory function
- **Performance Issue**: Very slow convergence (3-4 iterations/sec)
- **Next Steps**:
  1. Simplify the likelihood function for testing
  2. Reduce the number of parameters for initial tests
  3. Use better initial values from NLSQ results

## Code Changes Applied

### 1. NLSQ Fix (homodyne/optimization/nlsq.py)
```python
# Line 315-319: Extract q outside residual function
q_list = data.get("wavevector_q_list", [0.0054])
q_scalar = float(q_list[0]) if len(q_list) > 0 else 0.0054
L_scalar = 100.0

# Line 362-363: Use pre-extracted scalars
c2_theory = theory_engine.compute_g2(
    params_array_physical,
    data["t1"],
    data["t2"],
    data["phi_angles_list"],
    q_scalar,  # Use pre-extracted scalar
    L_scalar,  # Use pre-extracted scalar
    contrast,
    offset,
)
```

### 2. MCMC Fix (homodyne/optimization/mcmc.py)
```python
# Line 564: JIT compile with static argument
_compute_simple_theory = jit(_compute_simple_theory, static_argnums=(5,))

# Line 552-553: Fixed parameter indexing
D0 = params[0]  # Was params[2]
alpha = params[1]  # Was params[3]
```

## Performance Recommendations

### For NLSQ:
1. **Trace the error**: Add debug prints to locate exact float() call
2. **Use JAX-native operations**: Replace all float() with jnp.asarray()
3. **Test with simpler model**: Start with static mode (3 params) instead of laminar flow (7 params)

### For MCMC:
1. **Optimize sampling**:
   ```python
   # Enable parallel chains on CPU
   import numpyro
   numpyro.set_host_device_count(4)
   ```

2. **Simplify initial testing**:
   - Use fewer warmup/sample iterations (50/100)
   - Test with static mode first
   - Use tighter priors based on true values

3. **Profile the bottleneck**:
   ```python
   import jax.profiler
   jax.profiler.start_trace("/tmp/jax-trace")
   # Run MCMC
   jax.profiler.stop_trace()
   ```

## Test Commands

### Quick Test (Static Mode)
```python
# Test with simpler static mode (3 parameters)
config_dict = {
    'analysis_mode': 'static_isotropic',  # Simpler mode
    'optimization': {
        'nlsq': {'max_iterations': 50},
        'mcmc': {'n_warmup': 50, 'n_samples': 100}
    }
}
```

### Debug Mode
```bash
# Enable JAX debugging
export JAX_DEBUG_NANS=1
export JAX_TRACEBACK_FILTERING=off
export JAX_DISABLE_JIT=1  # Temporarily disable JIT to find errors

python debug_optimization_fixed.py
```

## Next Debugging Steps

1. **Locate float() error in NLSQ**:
   ```python
   # Add to theory_engine.compute_g2
   print(f"q type: {type(q)}, value: {q}")
   print(f"L type: {type(L)}, value: {L}")
   ```

2. **Profile MCMC bottleneck**:
   ```python
   # Use simpler likelihood for testing
   def simple_likelihood(params):
       # Direct computation without theory engine
       return jnp.sum((data - model)**2)
   ```

3. **Test components individually**:
   ```python
   # Test JAX backend directly
   from homodyne.core.jax_backend import compute_g2_scaled
   result = compute_g2_scaled(params, t1, t2, phi, 0.0054, 100.0)
   ```

## Conclusion

The implementations are close to working but need:
1. **NLSQ**: Complete removal of float() conversions in the computation chain
2. **MCMC**: Performance optimization and simplified testing
3. **Both**: Better initial parameter values and bounds

The architecture is sound - these are JAX-specific implementation details that need adjustment.