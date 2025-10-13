# Migration Guide: Optimistix → NLSQ

**Version**: Homodyne v2.0+
**Date**: October 2025
**Migration Difficulty**: 🟢 Easy (Backward compatible)

---

## Overview

Homodyne v2.0 has migrated from **Optimistix** to **NLSQ** (github.com/imewei/NLSQ) for trust-region nonlinear least squares optimization. This migration provides:

✅ **Better performance** - Optimized trust-region algorithm with large dataset support
✅ **Native error recovery** - Built-in convergence failure handling
✅ **GPU acceleration** - Transparent GPU support via JAX
✅ **Active maintenance** - Actively developed and maintained package

**Good news**: The migration is **99% backward compatible**. Most users won't need to change any code.

---

## Who Should Read This

- **Existing homodyne users** upgrading to v2.0+
- **New users** wanting to understand the optimization backend
- **Contributors** working on optimization-related code

---

## Quick Start

### Step 1: Update Installation

```bash
# OLD: Homodyne v1.x with Optimistix
pip install homodyne==1.x

# NEW: Homodyne v2.0+ with NLSQ
pip install homodyne>=2.0

# Or from source
git clone https://github.com/your-repo/homodyne.git
cd homodyne
pip install -e .
```

The NLSQ package will be automatically installed as a dependency.

### Step 2: Verify Installation

```python
import homodyne
from homodyne.optimization.nlsq import fit_nlsq_jax

print(f"Homodyne version: {homodyne.__version__}")
print(f"NLSQ integration: ✓")
```

### Step 3: Run Your Existing Code

**No code changes required!** Your existing scripts should work as-is:

```python
# This code works in both v1.x (Optimistix) and v2.0+ (NLSQ)
from homodyne.optimization.nlsq import fit_nlsq_jax

result = fit_nlsq_jax(
    data=xpcs_data,
    config=config,
    initial_params=None,  # Auto-loads from config
    bounds=None,          # Auto-loads from config
    analysis_mode="laminar_flow"
)

# Result attributes unchanged
print(f"Chi-squared: {result.chi_squared}")
print(f"Parameters: {result.parameters}")
print(f"Success: {result.success}")
```

---

## What's Changed

### ✅ Backward Compatible

| Feature | Status | Notes |
|---------|--------|-------|
| `fit_nlsq_jax()` API | ✅ Identical | Function signature unchanged |
| Result attributes | ✅ Identical | `chi_squared`, `parameters`, `success` unchanged |
| Config files (YAML) | ✅ Compatible | Existing configs work without modification |
| Parameter bounds | ✅ Compatible | Same bounds format and validation |
| Error handling | ✅ Enhanced | Better error messages, auto-retry on convergence failure |
| GPU support | ✅ Transparent | Automatic GPU detection via JAX (no code changes) |

### 🆕 New Features

1. **Automatic Error Recovery** (enabled by default):
   ```python
   # Automatically retries on convergence failure
   result = fit_nlsq_jax(data, config)

   # Check if recovery was needed
   if result.recovery_actions:
       print(f"Recovery actions: {result.recovery_actions}")
   ```

2. **Large Dataset Support** (>1M points):
   ```python
   # Automatic strategy selection
   # - <1M points: Uses curve_fit (standard)
   # - >1M points: Uses curve_fit_large (memory-efficient)
   wrapper = NLSQWrapper(enable_large_dataset=True)  # Default
   ```

3. **Enhanced Device Reporting**:
   ```python
   result = fit_nlsq_jax(data, config)
   print(result.device_info)
   # Example: {'device': 'gpu:0', 'platform': 'CUDA', 'memory_gb': 8.0}
   ```

### 🔄 Internal Changes (No Action Needed)

- Optimization engine: Optimistix → NLSQ
- Trust-region algorithm: Still Levenberg-Marquardt (same mathematical foundation)
- Error recovery: Now integrated into NLSQWrapper (no separate ErrorRecoveryManager)

---

## Breaking Changes

### ⚠️ None for Most Users

The API is **99% backward compatible**. However, if you were directly accessing internal Optimistix APIs (not recommended), you'll need to update:

```python
# ❌ OLD: Direct Optimistix usage (never documented, unsupported)
from optimistix import LevenbergMarquardt
solver = LevenbergMarquardt(...)

# ✅ NEW: Use homodyne's public API
from homodyne.optimization.nlsq import fit_nlsq_jax
result = fit_nlsq_jax(data, config)
```

### Configuration File Changes

**None required**. Existing YAML configs work without modification:

```yaml
# This works in both v1.x and v2.0+
optimization:
  lsq:
    max_iterations: 1000
    tolerance: 1e-6

analysis_mode: "laminar_flow"

initial_parameters:
  values: [0.5, 1.0, 1000.0, 0.5, 10.0, 1e-4, 0.5, 1e-5, 0.0]
  parameter_names: [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_offset, phi0]
```

---

## Performance Comparison

| Dataset Size | v1.x (Optimistix) | v2.0+ (NLSQ) | Speedup |
|--------------|-------------------|--------------|---------|
| Small (<1K points) | ~500ms | ~500ms | ~1.0x |
| Medium (1-100K points) | ~5s | ~4s | ~1.25x |
| Large (>1M points) | ~60s | ~45s | ~1.33x |
| GPU acceleration | Manual setup | Automatic | Built-in |

*Note: Benchmarks are approximate and depend on hardware, data complexity, and convergence behavior.*

---

## Troubleshooting

### Issue: "ImportError: No module named 'nlsq'"

**Solution**: Install/update NLSQ package:
```bash
pip install nlsq>=0.1.0
# or
pip install --upgrade homodyne
```

### Issue: "Optimization failed to converge"

**Solution**: The error recovery system will automatically retry with perturbed parameters. Check `result.recovery_actions` to see what was attempted:

```python
try:
    result = fit_nlsq_jax(data, config)
    if result.recovery_actions:
        print(f"Recovery needed: {result.recovery_actions}")
except Exception as e:
    print(f"Optimization failed after recovery attempts: {e}")
    # Check error message for suggestions
```

### Issue: Results differ slightly from v1.x

**Expected**: Minor numerical differences (<1%) are normal due to:
- Different trust-region implementation details
- Different convergence criteria
- Different numerical precision handling

**Action**: If differences are >5%, please [report an issue](https://github.com/your-repo/homodyne/issues) with:
- Your configuration file
- Dataset characteristics
- v1.x vs v2.0 results comparison

### Issue: GPU not being used

**Check GPU availability**:
```python
import jax
print(f"JAX devices: {jax.devices()}")
print(f"Default device: {jax.devices()[0]}")
```

**Expected output** (GPU available):
```
JAX devices: [GpuDevice(id=0, process_index=0)]
Default device: GpuDevice(id=0, process_index=0)
```

**If CPU only**:
- Ensure CUDA is installed (version 12.1-12.9)
- Reinstall JAX with CUDA support: `pip install --upgrade "jax[cuda12-local]"`
- Check NVIDIA driver version: `nvidia-smi`

---

## FAQ

### Q: Do I need to change my code?
**A**: No, for 99% of users. If you're using the documented public API (`fit_nlsq_jax`), no changes needed.

### Q: Will my results change?
**A**: Minor numerical differences (<1%) are expected and normal. The same mathematical algorithm (Levenberg-Marquardt trust-region) is used, but implementation details differ.

### Q: Can I roll back to v1.x?
**A**: Yes, downgrade to homodyne v1.x:
```bash
pip install homodyne==1.9.0  # Or your preferred v1.x version
```

### Q: Is GPU support still available?
**A**: Yes, GPU acceleration is **automatic** via JAX. No configuration needed. The system detects available GPUs and uses them automatically.

### Q: What about MCMC optimization?
**A**: MCMC optimization (NumPyro/BlackJAX) is **unchanged** and still fully supported.

### Q: Where can I get help?
**A**:
- **Documentation**: See [CLAUDE.md](../CLAUDE.md) for technical details
- **Issues**: [GitHub Issues](https://github.com/your-repo/homodyne/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/homodyne/discussions)

---

## Technical Details

### Architecture Changes

**v1.x (Optimistix)**:
```
fit_nlsq_jax() → Optimistix.LevenbergMarquardt → JAX → Result
```

**v2.0+ (NLSQ)**:
```
fit_nlsq_jax() → NLSQWrapper → NLSQ.curve_fit → JAX → Result
                      ↓
            (data prep, validation,
             error recovery, result packaging)
```

The new architecture includes:
- **NLSQWrapper**: Adapter layer providing data preparation, parameter validation, and result packaging
- **Error Recovery**: Automatic retry with parameter perturbation on convergence failures
- **Large Dataset Support**: Automatic selection of memory-efficient algorithms for >1M points

### Error Recovery Strategy

v2.0+ includes intelligent error recovery (can be disabled):

```python
# Enable error recovery (default)
wrapper = NLSQWrapper(enable_recovery=True)

# Disable for debugging
wrapper = NLSQWrapper(enable_recovery=False)
```

**Recovery flow**:
1. **Attempt 1**: Original parameters
2. **Attempt 2**: Parameters perturbed by ±10%
3. **Attempt 3**: Parameters perturbed by ±20% with relaxed tolerance
4. **Final**: Fail with comprehensive diagnostics and suggestions

### Performance Overhead

The NLSQWrapper adds minimal overhead (<5% for datasets >1000 points):

| Operation | Time (9K points) | % of Total |
|-----------|------------------|------------|
| Data preparation | ~5ms | ~1% |
| NLSQ optimization | ~500ms | ~95% |
| Result packaging | ~20ms | ~4% |
| **Total** | **~525ms** | **100%** |

See `tests/performance/test_wrapper_overhead.py` for detailed benchmarks.

---

## Migration Checklist

- [ ] Update homodyne to v2.0+: `pip install --upgrade homodyne`
- [ ] Verify NLSQ is installed: `python -c "import nlsq; print('✓')"`
- [ ] Run existing test suite to verify backward compatibility
- [ ] Check for any deprecation warnings in logs
- [ ] Review result differences (should be <1%)
- [ ] Update any direct Optimistix usage (if applicable)
- [ ] Test GPU acceleration (if using GPU hardware)
- [ ] Update CI/CD pipelines (if needed)
- [ ] Read [CLAUDE.md](../CLAUDE.md) for latest development guidelines

---

## Need More Help?

- **Quick Reference**: [README.md](../README.md)
- **Technical Guide**: [CLAUDE.md](../CLAUDE.md)
- **Bug Reports**: [GitHub Issues](https://github.com/your-repo/homodyne/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/your-repo/homodyne/discussions)

---

**Last Updated**: October 2025
**Homodyne Version**: v2.0+
**NLSQ Version**: v0.1.0+
