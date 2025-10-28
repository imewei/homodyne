# NUTS Chain Parallelization - Quick Reference

**Last Updated**: October 28, 2025

## TL;DR

**Default**: 4 chains for convergence diagnostics (R-hat, ESS)

| Your System | Execution | Performance |
|-------------|-----------|-------------|
| **Single GPU (16GB)** | Sequential | 4× single-chain time |
| **14-core CPU** | Parallel | ~1.1× single-chain time |

**Trade-off**: Single GPU chains are sequential but provide critical convergence diagnostics worth the 4× time cost.

---

## Platform Behavior

| Platform | Chains | Mode | Speed | Diagnostics |
|----------|--------|------|-------|-------------|
| **CPU** | 4 | ✅ Parallel | ~1.1× | ✅ Full |
| **1 GPU** | 4 | ⚠️ Sequential | 4× | ✅ Full |
| **4 GPUs** | 4 | ✅ Parallel | ~1.1× | ✅ Full |

---

## Configuration Presets

### Fast (Development)
```yaml
mcmc:
  n_chains: 1      # 4× faster
  n_samples: 500
  n_warmup: 250
```
**Use when**: Quick iteration, don't need diagnostics

### Balanced (Default)
```yaml
mcmc:
  n_chains: 4      # Good diagnostics
  n_samples: 1000
  n_warmup: 500
```
**Use when**: Production analysis, need reliability

### High-Quality
```yaml
mcmc:
  n_chains: 4      # Full diagnostics
  n_samples: 2000  # More samples
  n_warmup: 1000   # Better adaptation
```
**Use when**: Publication-quality results

---

## Why 4 Chains on 1 GPU?

**Even though sequential, you get:**

1. ✅ **R-hat diagnostic** - Detects non-convergence
2. ✅ **ESS (Effective Sample Size)** - True independent sample count
3. ✅ **Divergence detection** - Identifies problematic regions
4. ✅ **Better uncertainty** - Cross-chain variance

**Without multiple chains:**
- ❌ Cannot detect if stuck in local mode
- ❌ Cannot compute R-hat (no between-chain variance)
- ❌ Limited divergence detection
- ⚠️ May report false confidence in results

---

## Memory Usage

**Single GPU (Sequential)**:
```
Memory = dataset_size × 8 bytes × 6
Example: 23M points × 8 × 6 = 1.1 GB
```
Only 1 chain on GPU at a time → Memory-efficient

**CPU (Parallel)**:
```
Memory = dataset_size × 8 bytes × 6 × n_chains
Example: 23M × 8 × 6 × 4 = 4.4 GB
```
All chains in RAM simultaneously → Need more memory

---

## When to Reduce Chains

**Use `n_chains: 1` when:**
- Quick development testing
- Time-critical results needed
- You know sampling works (prior successful runs)
- Using CMC instead (CMC handles parallelism differently)

**Use `n_chains: 2` when:**
- Need basic convergence check
- Computational budget limited
- Compromise between speed and diagnostics

**Keep `n_chains: 4` when:**
- Production analysis
- First time analyzing dataset
- Need publication-quality results
- Convergence is uncertain

---

## Convergence Diagnostics

**R-hat (Gelman-Rubin)**:
```python
R-hat < 1.01  → ✅ Converged
R-hat > 1.01  → ❌ Need more samples
```

**ESS (Effective Sample Size)**:
```python
ESS > 400 per parameter  → ✅ Good
ESS < 400 per parameter  → ⚠️ Increase n_samples
```

**Divergences**:
```python
Divergences < 1%  → ✅ Acceptable
Divergences > 1%  → ⚠️ Reparameterize or tune
```

---

## NUTS vs CMC

| Criterion | NUTS (4 chains) | CMC |
|-----------|-----------------|-----|
| **Best for** | < 100 samples | ≥ 100 samples or huge data |
| **Single GPU speed** | 4× single-chain | Parallel shards |
| **Diagnostics** | Excellent (R-hat, ESS) | Basic |
| **Memory** | Dataset × 1 | Dataset / num_shards |
| **Setup** | Simple | More complex |

---

## Configuration File

**Location**: `homodyne_laminar_flow_config.yaml`

```yaml
mcmc:
  method: auto  # or 'nuts' or 'cmc'

  # Chain configuration
  n_chains: 4        # Number of chains (default: 4)
  n_samples: 1000    # Samples per chain (default: 1000)
  n_warmup: 500      # Warmup per chain (default: 500)

  # NUTS tuning (advanced)
  target_accept_prob: 0.8    # Default: 0.8
  max_tree_depth: 10         # Default: 10
  rng_key: 42                # Random seed
```

---

## Example Execution Times

**Dataset**: 23 phi × 1M points each (23M total)

| Configuration | GPU Time | CPU Time (14 cores) |
|---------------|----------|---------------------|
| 1 chain | 3 min | 12 min |
| 2 chains | 6 min | 14 min |
| 4 chains | 12 min | 15 min |
| 8 chains | 24 min | 18 min |

---

## Code Reference

**Implementation**: `homodyne/optimization/mcmc.py`

**Key Functions**:
- `_get_mcmc_config()` - Line 629 (defaults)
- `_run_numpyro_sampling()` - Line 895 (parallelization)

**Log Messages**:
```
# CPU parallelization
Set host device count to 4 for CPU parallel chains

# GPU (sequential)
Using GPU with 4 chains on 1 device(s)

# Execution
Starting NUTS sampling: 4 chains, 500 warmup, 1000 samples
```

---

## Common Issues

**Issue**: Chains finish at different speeds

**Explanation**: Normal - different random initializations lead to different tree depths

**Action**: None needed (each chain independently explores)

---

**Issue**: Out of memory with 4 chains on CPU

**Solution**:
```yaml
n_chains: 2  # Reduce memory by 50%
```

---

**Issue**: Chains not converging (R-hat > 1.01)

**Solutions**:
```yaml
# Option 1: More samples
n_samples: 2000
n_warmup: 1000

# Option 2: More chains
n_chains: 8

# Option 3: Better priors (use NLSQ initialization)
# Already done automatically in homodyne
```

---

## See Also

- Full documentation: `docs/architecture/nuts-chain-parallelization.md`
- CMC decision logic: `docs/architecture/cmc-decision-quick-reference.md`
- Implementation: `homodyne/optimization/mcmc.py:895-985`
