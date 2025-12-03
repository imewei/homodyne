# Consensus Monte Carlo Troubleshooting Guide

**Version:** 2.0+ **Last Updated:** 2025-10-24

______________________________________________________________________

## Table of Contents

1. [Common Error Messages](#common-error-messages)
1. [Convergence Issues](#convergence-issues)
1. [Performance Problems](#performance-problems)
1. [Memory Errors](#memory-errors)
1. [Backend-Specific Issues](#backend-specific-issues)
1. [Configuration Errors](#configuration-errors)
1. [Data Quality Issues](#data-quality-issues)
1. [Diagnostic Interpretation](#diagnostic-interpretation)
1. [Debug Mode](#debug-mode)
1. [Getting Help](#getting-help)

______________________________________________________________________

## Common Error Messages

### Error: "All shards failed to converge"

**Full Message:**

```
RuntimeError: All shards failed to converge. Cannot combine posteriors.
```

**Cause:**

- Poor initialization (bad initial parameters)
- Data quality issues (NaN, Inf, outliers)
- Too few warmup iterations
- Inappropriate priors

**Solutions:**

**1. Check initial parameters**

```python
# Run NLSQ first to get good initial values
from homodyne.optimization.nlsq import fit_nlsq_jax

nlsq_result = fit_nlsq_jax(
    data=c2_exp, t1=t1, t2=t2, phi=phi,
    q=q, L=L,
    analysis_mode='static_isotropic',
)

# Use NLSQ results for MCMC initialization
initial_params = nlsq_result.best_fit_parameters

result = fit_mcmc_jax(
    data=c2_exp, ...,
    initial_params=initial_params,  # Good starting point
    method='cmc',
)
```

**2. Increase warmup iterations**

```yaml
# In configuration file
per_shard_mcmc:
  num_warmup: 1000  # Up from default 500
  num_samples: 2000
```

**3. Check data quality**

```python
import numpy as np

# Check for NaN/Inf
assert not np.any(np.isnan(data)), "Data contains NaN values"
assert not np.any(np.isinf(data)), "Data contains Inf values"

# Check data range
print(f"Data range: [{data.min():.6f}, {data.max():.6f}]")
print(f"Data mean: {data.mean():.6f}, std: {data.std():.6f}")

# Look for outliers
outliers = np.abs(data - data.mean()) > 5 * data.std()
print(f"Outliers: {outliers.sum()} / {len(data)} ({100*outliers.sum()/len(data):.2f}%)")
```

______________________________________________________________________

### Error: "High KL divergence between shards"

**Full Message:**

```
WARNING: Max KL divergence (7.3) exceeds threshold (2.0)
```

**Cause:**

- Multi-modal posterior distribution
- Non-representative shards (poor sharding strategy)
- Data heterogeneity across time/angle bins
- Convergence to different local optima

**Solutions:**

**1. Use stratified sharding**

```yaml
# Ensure representative shards
sharding:
  strategy: stratified  # NOT 'random' or 'contiguous'
  num_shards: auto
```

**2. Switch to simple averaging**

```yaml
# More robust to multi-modal posteriors
combination:
  method: simple_average  # Instead of 'weighted_gaussian'
  fallback_enabled: true
```

**3. Increase shard size (fewer shards)**

```yaml
# Larger shards are more representative
sharding:
  num_shards: 8  # Down from auto (maybe 32)
  max_points_per_shard: 2000000  # 2M points per shard
```

**4. Relax validation threshold (lenient mode)**

```yaml
# For exploratory analysis
validation:
  strict_mode: false  # Log warnings but don't fail
  max_between_shard_kl: 5.0  # Relax threshold
```

______________________________________________________________________

### Error: "CUDA out of memory"

**Full Message:**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GiB (GPU 0; 15.78 GiB total capacity; 14.12 GiB already allocated; 896.00 MiB free; 14.32 GiB reserved in total by PyTorch)
```

**Cause:**

- Shard size too large for GPU memory
- Multiple chains per shard
- Memory leak in long-running MCMC

**Solutions:**

**1. Reduce shard size**

```yaml
sharding:
  max_points_per_shard: 500000  # Down from 1M
  num_shards: auto  # Will create more shards automatically
```

**2. Use 1 chain per shard**

```yaml
per_shard_mcmc:
  num_chains: 1  # Down from 4
  num_samples: 2000  # Keep same for quality
```

**3. Switch to CPU backend**

```yaml
backend:
  name: multiprocessing  # CPU instead of GPU (pjit)
```

**4. Clear GPU cache between shards**

```python
import jax
import gc

# In custom backend implementation
for shard in shards:
    result = run_mcmc(shard, ...)
    jax.clear_caches()  # Clear JAX compilation cache
    gc.collect()         # Python garbage collection
```

______________________________________________________________________

### Error: "SVI initialization timeout"

**Full Message:**

```
WARNING: SVI initialization timed out after 900 seconds
```

**Cause:**

- Too many SVI steps
- Large dataset for pooling
- Slow convergence

**Solutions:**

**1. Reduce SVI steps**

```yaml
initialization:
  method: svi
  svi_steps: 5000  # Down from 20000
  svi_timeout: 600  # 10 minutes max
```

**2. Use NLSQ initialization instead**

```yaml
initialization:
  method: nlsq  # Skip SVI, use NLSQ results
```

**3. Use identity initialization (fallback)**

```yaml
initialization:
  method: identity  # Fastest but slower MCMC convergence
```

______________________________________________________________________

## Convergence Issues

### Low Effective Sample Size (ESS < 100)

**Symptom:**

```json
{
  "shard_id": 3,
  "ess": 42.3,
  "acceptance_rate": 0.92
}
```

**Diagnosis:**

Low ESS indicates high autocorrelation in MCMC samples (poor mixing).

**Solutions:**

**1. Increase number of samples**

```yaml
per_shard_mcmc:
  num_samples: 5000  # Up from 2000
```

**2. Improve initialization**

```yaml
initialization:
  method: svi
  svi_steps: 10000  # Better init → better mixing
```

**3. Check acceptance rate**

```python
# Ideal acceptance rate: 0.65-0.85
if result.acceptance_rate < 0.5:
    print("Acceptance too low, NUTS struggling")
elif result.acceptance_rate > 0.95:
    print("Acceptance too high, NUTS taking tiny steps")
```

______________________________________________________________________

### High R-hat (> 1.1)

**Symptom:**

```json
{
  "shard_id": 5,
  "r_hat": 1.34,
  "converged": false
}
```

**Diagnosis:**

R-hat > 1.1 indicates chains haven't converged to same distribution.

**Solutions:**

**1. Increase warmup**

```yaml
per_shard_mcmc:
  num_warmup: 2000  # Up from 500
```

**2. Run multiple chains per shard**

```yaml
per_shard_mcmc:
  num_chains: 4  # Up from 1
  # Note: Slower but better convergence diagnostics
```

**3. Check initial parameters**

```python
# Verify initial params are reasonable
print(f"Initial D0: {initial_params['D0']}")  # Should be ~1e3-1e5
print(f"Initial alpha: {initial_params['alpha']}")  # Should be 0.5-2.0
```

______________________________________________________________________

## Performance Problems

### CMC slower than expected

**Expected:** Linear speedup with number of devices **Actual:** 2x slower than standard
NUTS

**Diagnosis:**

Check CMC overhead breakdown:

```python
# Enable detailed logging
import logging
logging.getLogger('homodyne.optimization.cmc').setLevel(logging.DEBUG)

result = fit_mcmc_jax(data, ..., method='cmc')

# Check logs for timing:
# - Sharding time
# - SVI initialization time
# - MCMC execution time (per shard)
# - Combination time
```

**Solutions:**

**1. Reduce SVI overhead**

```yaml
initialization:
  method: nlsq  # Skip SVI (30-60s overhead)
```

**2. Use faster sharding strategy**

```yaml
sharding:
  strategy: random  # Faster than stratified
```

**3. Optimize shard size**

```yaml
# Balance: Too small → overhead, Too large → OOM
sharding:
  num_shards: 16  # Manual tuning for your hardware
```

______________________________________________________________________

### SVI initialization too slow (> 5 minutes)

**Solutions:**

**1. Reduce pooled samples**

```yaml
initialization:
  samples_per_shard: 100  # Down from 200
```

**2. Reduce SVI steps**

```yaml
initialization:
  svi_steps: 3000  # Down from 5000
```

**3. Increase learning rate**

```yaml
initialization:
  svi_learning_rate: 0.005  # Up from 0.001 (faster convergence)
```

______________________________________________________________________

## Memory Errors

### Python process killed (OOM)

**Symptom:**

```
Killed
```

No error message, process just terminates.

**Diagnosis:**

System OOM killer terminated the process.

**Solutions:**

**1. Monitor memory usage**

```python
import psutil

process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1e9:.2f} GB")
```

**2. Reduce memory footprint**

```yaml
sharding:
  max_points_per_shard: 500000  # Smaller shards

per_shard_mcmc:
  num_samples: 1000  # Fewer samples
  num_chains: 1      # Single chain

backend:
  enable_checkpoints: false  # Disable checkpoint overhead
```

**3. Use chunked processing**

```python
# Process shards in batches
config['backend']['max_parallel_shards'] = 4  # Process 4 at a time
```

______________________________________________________________________

## Backend-Specific Issues

### PjitBackend: "No GPU available"

**Solution:**

```python
# Verify GPU detection
import jax
print(f"JAX devices: {jax.devices()}")
# Expected: [cuda(id=0)] or [gpu(id=0)]
# Actual: [cpu(id=0)] → GPU not detected

# Check CUDA installation
python -c "import jaxlib; print(jaxlib.__version__)"
# Should show version with 'cuda' in name

# Reinstall JAX with GPU support
pip uninstall -y jax jaxlib
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0
```

______________________________________________________________________

### MultiprocessingBackend: "Pickling error"

**Error:**

```
TypeError: cannot pickle 'jax.interpreters.xla.DeviceArray' object
```

**Cause:**

JAX arrays can't be pickled for multiprocessing.

**Solution:**

Convert to NumPy arrays before multiprocessing:

```python
# In backend implementation
import numpy as np

shard['data'] = np.array(shard['data'])  # JAX → NumPy
shard['t1'] = np.array(shard['t1'])
```

______________________________________________________________________

### PBSBackend: "Job submission failed"

**Error:**

```
qsub: Job rejected by all possible destinations
```

**Diagnosis:**

Check PBS configuration:

```bash
# Check queue availability
qstat -Q

# Check node availability
pbsnodes -a

# Verify project allocation
qmgr -c "list queue batch"
```

**Solution:**

Update PBS configuration:

```yaml
pbs:
  project_name: "correct_project_name"  # Check with sysadmin
  queue: "batch"  # Or "debug", "gpu", etc.
  walltime: "01:00:00"  # Reduce if needed
  cores_per_node: 36  # Match your cluster
```

______________________________________________________________________

## Configuration Errors

### Error: "Invalid sharding strategy"

**Message:**

```
ValueError: Invalid sharding strategy: 'stratify'. Must be one of: ['stratified', 'random', 'contiguous']
```

**Solution:**

Fix typo in configuration:

```yaml
sharding:
  strategy: stratified  # NOT 'stratify'
```

______________________________________________________________________

### Error: "min_success_rate must be between 0 and 1"

**Message:**

```
ValueError: min_success_rate must be in range [0.0, 1.0], got 90
```

**Solution:**

Use decimal fraction, not percentage:

```yaml
combination:
  min_success_rate: 0.90  # NOT 90
```

______________________________________________________________________

## Data Quality Issues

### Unexpected parameter estimates

**Symptom:**

```
D0 = 1234567.89  # Way too high
alpha = 0.001    # Way too low
```

**Diagnosis:**

Check data preprocessing:

```python
# 1. Check data normalization
c2_mean = np.mean(data)
print(f"Mean c2: {c2_mean}")  # Should be ~1.0-1.5

# 2. Check for scaling issues
if c2_mean > 10:
    print("Data not normalized, apply background subtraction")

# 3. Verify time arrays
print(f"t1 range: [{t1.min()}, {t1.max()}]")
print(f"t2 range: [{t2.min()}, {t2.max()}]")
```

**Solutions:**

**1. Apply proper normalization**

```python
from homodyne.data.preprocessing import normalize_c2

c2_normalized = normalize_c2(c2_raw, background=1.0)
```

**2. Check parameter bounds**

```yaml
parameter_space:
  bounds:
    - name: D0
      min: 100.0      # Reasonable lower bound
      max: 100000.0   # Reasonable upper bound
```

### Per-Angle Scaling with Non-Stratified Sharding

**⚠️ CRITICAL: CMC Sharding Strategy Requirement**

**Symptom:**

- Parameters unchanged from initial values
- Zero gradients in optimization
- "Convergence failure" warnings for some/all shards
- Identical initial and final parameter estimates

**Root Cause:**

CMC always uses **per-angle scaling** (separate `contrast[i]` and `offset[i]` for each
phi angle). This requires that **every shard contains data from all phi angles**.
Non-stratified sharding (random, contiguous) may create shards with incomplete phi angle
coverage, causing:

1. Missing angles in shard → Zero gradient for that angle's parameters
1. MCMC sampler cannot update parameters with zero gradients
1. Silent optimization failure (completes but returns initial values)

**Technical Details:**

This is the same root cause as the NLSQ chunking issue documented in
`docs/troubleshooting/nlsq-zero-iterations-investigation.md`:

```
Per-angle parameters: contrast[0], offset[0], contrast[1], offset[1], ...
                               ↓
Shard missing phi[0] → grad(contrast[0]) = 0, grad(offset[0]) = 0
                               ↓
MCMC sampler cannot move in zero-gradient directions
                               ↓
Parameters unchanged → silent failure
```

**Solution:**

**ALWAYS use stratified sharding for CMC (default):**

```yaml
cmc:
  sharding:
    strategy: stratified  # REQUIRED for per-angle scaling
    num_shards: auto
```

**Verification:**

Check that your configuration doesn't override the default:

```python
# ❌ WRONG - May cause per-angle failures
cmc:
  sharding:
    strategy: random      # Dangerous with per-angle scaling

# ❌ WRONG - May cause per-angle failures
cmc:
  sharding:
    strategy: contiguous  # Dangerous with per-angle scaling

# ✅ CORRECT - Safe for per-angle scaling
cmc:
  sharding:
    strategy: stratified  # Default, explicitly specified
```

**References:**

- Ultra-Think Analysis: `ultra-think-20251106-012247`
- NLSQ Investigation: `docs/troubleshooting/nlsq-zero-iterations-investigation.md`
- Code: `homodyne/optimization/cmc/backends/multiprocessing.py:444`
  (per_angle_scaling=True)

______________________________________________________________________

## Diagnostic Interpretation

### Understanding Per-Shard Diagnostics

```json
{
  "shard_id": 0,
  "n_samples": 2000,
  "converged": true,
  "acceptance_rate": 0.82,
  "r_hat": 1.02,
  "ess": 1543.2
}
```

**Interpretation:**

- `converged: true` - ✅ Shard passed convergence criteria
- `acceptance_rate: 0.82` - ✅ Ideal range (0.65-0.85)
- `r_hat: 1.02` - ✅ Below threshold (< 1.1)
- `ess: 1543.2` - ✅ Well above minimum (> 100)

**Red Flags:**

- `converged: false` - ❌ Shard failed
- `acceptance_rate < 0.5` or `> 0.95` - ⚠️ NUTS struggling
- `r_hat > 1.1` - ❌ Not converged
- `ess < 100` - ⚠️ High autocorrelation

______________________________________________________________________

### Understanding CMC Diagnostics

```json
{
  "combination_success": true,
  "n_shards_converged": 14,
  "n_shards_total": 16,
  "convergence_rate": 0.875,
  "combination_time": 12.3,
  "max_kl_divergence": 1.42
}
```

**Interpretation:**

- `convergence_rate: 0.875` - ✅ 87.5% shards converged (> 80% is good)
- `max_kl_divergence: 1.42` - ✅ Below threshold (< 2.0)
- `combination_time: 12.3` - ✅ Fast combination (< 30s)

**Red Flags:**

- `convergence_rate < 0.8` - ⚠️ Too many failed shards
- `max_kl_divergence > 5.0` - ❌ Shards diverged
- `combination_success: false` - ❌ Combination failed

______________________________________________________________________

## Debug Mode

### Enable detailed logging

```python
import logging

# Set CMC modules to DEBUG level
logging.getLogger('homodyne.optimization.cmc').setLevel(logging.DEBUG)
logging.getLogger('homodyne.optimization.mcmc').setLevel(logging.DEBUG)

# Run CMC
result = fit_mcmc_jax(data, ..., method='cmc')
```

### Inspect intermediate results

```python
# Create coordinator directly for more control
from homodyne.optimization.cmc.coordinator import CMCCoordinator

coordinator = CMCCoordinator(config)

# Run step-by-step
shards = coordinator._create_shards(data, ...)
print(f"Created {len(shards)} shards")

init_params, inv_mass_matrix = coordinator._run_svi(shards, ...)
print(f"SVI converged: {inv_mass_matrix is not None}")

# Continue with remaining steps...
```

### Save diagnostics to file

```python
import json

# After CMC completion
diagnostics = {
    'per_shard': result.per_shard_diagnostics,
    'cmc': result.cmc_diagnostics,
    'num_shards': result.num_shards,
    'combination_method': result.combination_method,
}

with open('cmc_diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=2)
```

______________________________________________________________________

## Getting Help

### Before Opening an Issue

**Collect diagnostic information:**

```python
# 1. System information
import platform
import jax
import numpyro

print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python: {platform.python_version()}")
print(f"JAX: {jax.__version__}")
print(f"NumPyro: {numpyro.__version__}")
print(f"JAX devices: {jax.devices()}")

# 2. homodyne version
import homodyne
print(f"homodyne: {homodyne.__version__}")

# 3. CMC configuration
from homodyne.config.manager import ConfigManager
config_mgr = ConfigManager("config.yaml")
cmc_config = config_mgr.get_cmc_config()
print(f"CMC config: {cmc_config}")

# 4. CMC diagnostics (if available)
if result.is_cmc_result():
    print(f"CMC diagnostics: {result.cmc_diagnostics}")
```

### Where to Get Help

**GitHub Issues:**

- Bug reports: https://github.com/your-org/homodyne/issues
- Feature requests: https://github.com/your-org/homodyne/issues

**Discussion Forum:**

- Questions: https://github.com/your-org/homodyne/discussions
- Show & Tell: https://github.com/your-org/homodyne/discussions/categories/show-and-tell

**Email Support:**

- Critical issues: support@homodyne-xpcs.org
- Security issues: security@homodyne-xpcs.org

### Issue Template

````markdown
**Environment:**
- homodyne version: 2.0.0
- Platform: Linux, Python 3.12
- JAX: 0.8.0 (GPU)
- Hardware: NVIDIA A100 80GB

**Problem:**
[Clear description of the issue]

**Configuration:**
```yaml
# Paste relevant config sections
````

**Error Message:**

```
# Paste full error traceback
```

**Steps to Reproduce:**

1. Load data: `data = load_xpcs_data(...)`
1. Run CMC: `result = fit_mcmc_jax(data, ..., method='cmc')`
1. Error occurs at...

**Expected Behavior:** [What should happen]

**Actual Behavior:** [What actually happens]

**Diagnostics:**

```json
# Paste CMC diagnostics if available
```

```

---

## Summary

**Quick Troubleshooting Checklist:**

- [ ] Check data quality (no NaN/Inf, proper normalization)
- [ ] Verify initial parameters (run NLSQ first)
- [ ] Check hardware detection (`jax.devices()`)
- [ ] Review configuration (valid values, no typos)
- [ ] Monitor memory usage (reduce shard size if needed)
- [ ] Enable debug logging for detailed diagnostics
- [ ] Check per-shard diagnostics (R-hat, ESS, acceptance rate)
- [ ] Verify KL divergence between shards (< 2.0 is good)
- [ ] Try lenient validation mode for exploratory analysis
- [ ] Collect diagnostic information before opening issue

**Most Common Fixes:**

1. **All shards fail** → Run NLSQ first for initialization
2. **High KL** → Use stratified sharding + simple averaging
3. **OOM** → Reduce `max_points_per_shard`
4. **Slow** → Reduce `svi_steps` or use `method: nlsq`
5. **Low ESS** → Increase `num_samples` or improve initialization

For detailed information, see:
- User Guide: `docs/user-guide/cmc_guide.md`
- API Reference: `docs/api-reference/cmc_api.md`
- Migration Guide: `docs/migration/v3_cmc_migration.md`
```
