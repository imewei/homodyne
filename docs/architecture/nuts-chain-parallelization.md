# NUTS Chain Parallelization

**Last Updated**: October 28, 2025

## Overview

NumPyro's NUTS (No-U-Turn Sampler) implementation in homodyne supports **multiple parallel chains** for improved convergence diagnostics and uncertainty quantification. This document explains how chain parallelization works across different hardware platforms and how to configure it for optimal performance.

---

## Table of Contents

1. [Quick Summary](#quick-summary)
2. [Default Configuration](#default-configuration)
3. [Platform-Specific Behavior](#platform-specific-behavior)
4. [Performance Characteristics](#performance-characteristics)
5. [Why Multiple Chains?](#why-multiple-chains)
6. [Configuration Options](#configuration-options)
7. [Comparison: NUTS vs CMC](#comparison-nuts-vs-cmc)
8. [Recommendations](#recommendations)
9. [Implementation Details](#implementation-details)
10. [References](#references)

---

## Quick Summary

**NUTS runs multiple chains for convergence diagnostics:**

| Platform | Default Chains | Execution Mode | Performance |
|----------|---------------|----------------|-------------|
| **CPU** | 4 | ✅ **Parallel** | 4 chains run simultaneously |
| **Single GPU** | 4 | ⚠️ **Sequential** | 4 chains share GPU, run one at a time |
| **Multi-GPU (N)** | 4 | ✅ **Parallel** | Chains distributed across N GPUs |

**Key Insight**: Even though chains run sequentially on single GPU, running 4 chains provides critical convergence diagnostics (R-hat, ESS) that detect sampling problems.

---

## Default Configuration

**File**: `homodyne/optimization/mcmc.py:629-642`

```python
def _get_mcmc_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Get MCMC configuration with optimized defaults."""
    default_config = {
        "n_samples": 1000,       # Posterior samples per chain
        "n_warmup": 500,         # Warmup iterations per chain
        "n_chains": 4,           # Number of parallel chains
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
        "rng_key": 42,
    }

    # Update with provided kwargs
    default_config.update(kwargs)
    return default_config
```

**Total Iterations per Chain**: `n_warmup + n_samples = 500 + 1000 = 1500`
**Total Posterior Samples**: `n_chains × n_samples = 4 × 1000 = 4000`

---

## Platform-Specific Behavior

### CPU Parallelization

**Implementation** (`mcmc.py:933-938`):
```python
if platform == "cpu" and n_devices == 1:
    # CPU mode: use host device count for parallel chains
    numpyro.set_host_device_count(n_chains)
    logger.info(
        f"Set host device count to {n_chains} for CPU parallel chains",
    )
```

**How It Works**:
1. NumPyro creates `n_chains` virtual devices on CPU
2. JAX distributes chains across available CPU cores
3. Each chain runs independently with different random seed
4. Chains execute **truly in parallel** using multiprocessing

**Performance**:
- **4 chains on 14-core CPU**: Near-linear speedup (chains run simultaneously)
- **Total time**: ~1.1× single-chain time (small coordination overhead)
- **Memory**: 4× single-chain memory usage (each chain duplicates data)

**Example Log Output**:
```
Set host device count to 4 for CPU parallel chains
Starting NUTS sampling: 4 chains, 500 warmup, 1000 samples
```

### Single GPU Execution

**Implementation** (`mcmc.py:939-941`):
```python
elif platform == "gpu":
    # GPU mode: use available GPU devices
    logger.info(f"Using GPU with {n_chains} chains on {n_devices} device(s)")
```

**How It Works**:
1. All 4 chains share the **same GPU**
2. Chains execute **sequentially** (one at a time)
3. Each chain gets full GPU acceleration
4. JAX automatically handles scheduling

**Performance**:
- **4 chains on 1 GPU**: 4× single-chain time (sequential execution)
- **GPU utilization**: 100% during each chain (efficient use)
- **Memory**: Only 1 chain's data on GPU at a time (memory-efficient)

**Why Sequential?**:
- Single GPU cannot execute multiple MCMC chains simultaneously
- Each chain requires full GPU resources for matrix operations
- Sequential execution ensures maximum per-chain performance

**Example Log Output**:
```
Using GPU with 4 chains on 1 device(s)
Starting NUTS sampling: 4 chains, 500 warmup, 1000 samples
chain 1 |██████████| 1500/1500 [00:45<00:00, 33.3it/s]
chain 2 |██████████| 1500/1500 [00:44<00:00, 34.1it/s]
chain 3 |██████████| 1500/1500 [00:45<00:00, 33.3it/s]
chain 4 |██████████| 1500/1500 [00:44<00:00, 34.1it/s]
```

### Multi-GPU Execution

**Implementation** (`mcmc.py:939-943`):
```python
elif platform == "gpu":
    logger.info(f"Using GPU with {n_chains} chains on {n_devices} device(s)")
else:
    logger.info(f"Using {min(n_chains, n_devices)} parallel devices")
```

**How It Works**:
1. NumPyro distributes chains across available GPUs
2. **4 chains on 4 GPUs**: Each chain gets dedicated GPU
3. **4 chains on 2 GPUs**: 2 chains per GPU (sequential per GPU)
4. Chains execute **in parallel** across GPUs

**Performance**:
- **4 chains on 4 GPUs**: ~1.1× single-chain time (true parallelism)
- **4 chains on 2 GPUs**: ~2.2× single-chain time (2 chains per GPU)
- **Scaling**: Near-linear with number of GPUs

---

## Performance Characteristics

### Execution Time

| Configuration | Total Time | Speedup | Notes |
|---------------|------------|---------|-------|
| **1 chain, 1 GPU** | T | 1× | Baseline |
| **4 chains, 1 GPU** | ~4T | 0.25× | Sequential, but with diagnostics |
| **4 chains, 14-core CPU** | ~1.1T | 0.91× | Near-linear parallel speedup |
| **4 chains, 4 GPUs** | ~1.1T | 0.91× | True parallel execution |

### Memory Usage

| Platform | Configuration | Memory per Chain | Total Memory |
|----------|---------------|------------------|--------------|
| **CPU** | 4 chains | M | ~4M (all in RAM) |
| **Single GPU** | 4 chains | M | ~M (sequential, 1 at a time) |
| **4 GPUs** | 4 chains | M | ~4M (1 chain per GPU) |

Where M = dataset size × 8 bytes × 6 (data + gradients + MCMC state)

### Convergence Quality

| Chains | R-hat | ESS | Divergences | Quality |
|--------|-------|-----|-------------|---------|
| **1** | ❌ N/A | Limited | Hard to detect | Poor diagnostics |
| **2** | ⚠️ Rough | Better | Can detect | Basic diagnostics |
| **4** | ✅ Reliable | Best | Clear detection | Excellent diagnostics |

---

## Why Multiple Chains?

Even though chains run **sequentially on single GPU**, running 4 chains provides critical benefits:

### 1. Convergence Diagnostics (R-hat Statistic)

**Definition**: Measures between-chain vs within-chain variance
```python
# Gelman-Rubin R-hat statistic
# R-hat ≈ 1.0 → converged
# R-hat > 1.01 → not converged
```

**Why It Matters**:
- **Single chain**: Cannot detect if sampler is stuck in local mode
- **Multiple chains**: Compare independent explorations of parameter space
- **R-hat < 1.01**: High confidence in convergence
- **R-hat > 1.01**: Warning that more sampling needed

**Example**:
```python
# Good convergence
r_hat = {'D0': 1.003, 'alpha': 1.001, 'D_offset': 1.002}

# Poor convergence (need more samples)
r_hat = {'D0': 1.045, 'alpha': 1.089, 'D_offset': 1.023}
```

### 2. Effective Sample Size (ESS)

**Definition**: Number of independent samples after accounting for autocorrelation

**Why It Matters**:
- MCMC samples are **correlated** (not independent)
- ESS tells you true number of independent samples
- **High ESS**: Good mixing, efficient exploration
- **Low ESS**: Poor mixing, need more samples

**Example**:
```python
# Good mixing
ess = {'D0': 3500, 'alpha': 3200, 'D_offset': 3800}  # Out of 4000 total

# Poor mixing (high autocorrelation)
ess = {'D0': 450, 'alpha': 380, 'D_offset': 520}  # Out of 4000 total
```

### 3. Divergence Detection

**Definition**: NUTS detects regions where Hamiltonian dynamics fail

**Why It Matters**:
- **Divergences**: Indicate problematic posterior geometry
- **Multiple chains**: Better sampling of divergent regions
- **Actionable**: Reparameterize model or adjust step size

**Example**:
```
chain 1: 2 divergences
chain 2: 0 divergences
chain 3: 1 divergence
chain 4: 3 divergences
Total: 6/6000 (0.1%) → acceptable
```

### 4. Better Uncertainty Quantification

**Cross-Chain Variance**:
```python
# Single chain: Only within-chain variance
std_single = np.std(chain1_samples)

# Multiple chains: Within + between chain variance
std_multi = np.sqrt(within_chain_var + between_chain_var)
# More conservative, more realistic uncertainty
```

---

## Configuration Options

### Via YAML Configuration

**File**: `homodyne_laminar_flow_config.yaml`

```yaml
mcmc:
  method: auto  # or 'nuts' or 'cmc'

  # Chain configuration
  n_chains: 4        # Number of parallel chains (default: 4)
  n_samples: 1000    # Posterior samples per chain (default: 1000)
  n_warmup: 500      # Warmup iterations per chain (default: 500)

  # NUTS-specific tuning
  target_accept_prob: 0.8   # Target acceptance rate (default: 0.8)
  max_tree_depth: 10        # Maximum NUTS tree depth (default: 10)
  rng_key: 42               # Random seed (default: 42)
```

### Common Configurations

#### Fast Execution (Single GPU)
```yaml
mcmc:
  n_chains: 1       # No convergence diagnostics, but 4× faster
  n_samples: 1000
  n_warmup: 500
```

#### Balanced (Default)
```yaml
mcmc:
  n_chains: 4       # Good diagnostics, reasonable speed
  n_samples: 1000
  n_warmup: 500
```

#### High-Quality Sampling
```yaml
mcmc:
  n_chains: 4       # Full diagnostics
  n_samples: 2000   # More posterior samples
  n_warmup: 1000    # Longer warmup for better adaptation
```

#### CPU Optimization
```yaml
mcmc:
  n_chains: 8       # Utilize more CPU cores (if available)
  n_samples: 1000
  n_warmup: 500
```

---

## Comparison: NUTS vs CMC

### Execution Strategy

| Method | Chains | Parallelism | Use Case |
|--------|--------|-------------|----------|
| **NUTS** | 4 | Sequential (1 GPU) / Parallel (CPU) | Small-medium datasets, need diagnostics |
| **CMC** | N shards | Parallel (multi-sample) | Large datasets (>100 samples) or huge data |

### When to Use Each

**Use NUTS when**:
- Dataset has **< 100 independent samples** (phi angles)
- You need **convergence diagnostics** (R-hat, ESS)
- Dataset fits in GPU memory (< 50% of available)
- You want **simple, reliable sampling**

**Use CMC when**:
- Dataset has **≥ 100 independent samples** (many phi angles)
- Dataset requires **> 50% of GPU memory** (memory-constrained)
- You need **parallel execution** for speed
- You're comfortable with **more complex setup**

### Memory Requirements

**NUTS** (sequential on 1 GPU):
```python
memory_nuts = dataset_size × 8 bytes × 6 (MCMC overhead)
# Example: 23M points × 8 × 6 = 1.1 GB
```

**CMC** (parallel shards):
```python
memory_cmc = (dataset_size / num_shards) × 8 × 6
# Example: 200M points / 4 shards = 50M per shard × 8 × 6 = 2.4 GB per shard
```

---

## Recommendations

### General Guidelines

| Scenario | Recommendation | Configuration |
|----------|----------------|---------------|
| **Development/Testing** | Single chain | `n_chains: 1` |
| **Production Analysis** | Multiple chains | `n_chains: 4` (default) |
| **High-Quality Results** | More samples | `n_chains: 4, n_samples: 2000` |
| **CPU-Only System** | Many chains | `n_chains: 8` (or more) |
| **Time-Critical** | Fewer chains | `n_chains: 2` |

### Platform-Specific

#### Single GPU (Your System)

**Typical Dataset (23 phi × 1M points each)**:
```yaml
mcmc:
  method: auto       # Will select NUTS
  n_chains: 4        # Good diagnostics (4× execution time)
  n_samples: 1000
  n_warmup: 500
```

**Fast Development Iteration**:
```yaml
mcmc:
  method: nuts
  n_chains: 1        # 4× faster, no diagnostics
  n_samples: 500     # Fewer samples for quick results
  n_warmup: 250
```

**Production-Quality Results**:
```yaml
mcmc:
  method: nuts
  n_chains: 4        # Full diagnostics
  n_samples: 2000    # More posterior samples
  n_warmup: 1000     # Better adaptation
```

#### Multi-Core CPU

**Recommended**:
```yaml
mcmc:
  method: nuts
  n_chains: 8        # Utilize 8 cores (or more if available)
  n_samples: 1000
  n_warmup: 500
```

**Performance**: Near-linear scaling with core count

#### Multi-GPU (4 GPUs)

**Recommended**:
```yaml
mcmc:
  method: nuts
  n_chains: 4        # 1 chain per GPU (optimal)
  n_samples: 2000    # More samples (fast with parallelism)
  n_warmup: 1000
```

**Or use CMC** for even better parallelism:
```yaml
mcmc:
  method: cmc
  cmc:
    backend:
      type: pjit     # Multi-GPU backend
```

---

## Implementation Details

### Code Location

**File**: `homodyne/optimization/mcmc.py`

**Key Functions**:
- `_get_mcmc_config()` - Lines 629-642 (default configuration)
- `_run_numpyro_sampling()` - Lines 895-985 (chain parallelization logic)
- `_run_standard_nuts()` - Lines 464-601 (NUTS execution wrapper)

### Chain Parallelization Logic

**Platform Detection** (`mcmc.py:926-945`):
```python
n_chains = config.get("n_chains", 1)
if n_chains > 1:
    import jax

    n_devices = jax.local_device_count()
    platform = jax.devices()[0].platform if jax.devices() else "cpu"

    if platform == "cpu" and n_devices == 1:
        # CPU mode: set host device count for parallel chains
        numpyro.set_host_device_count(n_chains)
        logger.info(f"Set host device count to {n_chains} for CPU parallel chains")

    elif platform == "gpu":
        # GPU mode: chains share GPU (sequential) or distributed (multi-GPU)
        logger.info(f"Using GPU with {n_chains} chains on {n_devices} device(s)")
```

### MCMC Sampler Creation

**NUTS Kernel** (`mcmc.py:948-955`):
```python
nuts_kernel = NUTS(
    model,
    target_accept_prob=config["target_accept_prob"],  # 0.8
    max_tree_depth=config.get("max_tree_depth", 10),
    adapt_step_size=True,      # Automatic step size tuning
    adapt_mass_matrix=True,    # Automatic mass matrix adaptation
    dense_mass=False,          # Diagonal mass (more efficient)
)
```

**MCMC Sampler** (`mcmc.py:958-964`):
```python
mcmc = MCMC(
    nuts_kernel,
    num_warmup=config["n_warmup"],      # 500
    num_samples=config["n_samples"],    # 1000
    num_chains=config["n_chains"],      # 4
    progress_bar=True,                  # Show progress
)
```

### Execution and Diagnostics

**Run Sampling** (`mcmc.py:969-980`):
```python
logger.info(
    f"Starting NUTS sampling: {config['n_chains']} chains, "
    f"{config['n_warmup']} warmup, {config['n_samples']} samples"
)

mcmc.run(rng_key)

# Print diagnostics
mcmc.print_summary()

# Extract samples and diagnostics
samples = mcmc.get_samples()
diagnostics = {
    'r_hat': az.rhat(mcmc.get_samples(group_by_chain=True)),
    'ess': az.ess(mcmc.get_samples(group_by_chain=True)),
    'divergences': mcmc.get_extra_fields()['diverging'],
}
```

---

## References

### Internal Documentation
- `homodyne/optimization/mcmc.py` - MCMC implementation
- `homodyne/device/config.py` - Hardware detection and CMC decision
- `docs/architecture/cmc-dual-mode-strategy.md` - CMC vs NUTS comparison

### External Resources
- **NumPyro MCMC Guide**: https://num.pyro.ai/en/stable/mcmc.html
- **NUTS Paper**: Hoffman & Gelman (2014), "The No-U-Turn Sampler"
- **Convergence Diagnostics**: Gelman et al., "Bayesian Data Analysis" (3rd ed.)
- **R-hat Statistic**: Vehtari et al. (2021), "Rank-Normalization, Folding, and Localization"

### Key Concepts
- **NUTS**: No-U-Turn Sampler (adaptive Hamiltonian Monte Carlo)
- **R-hat**: Gelman-Rubin convergence diagnostic
- **ESS**: Effective Sample Size (accounting for autocorrelation)
- **Divergences**: Numerical instabilities in Hamiltonian dynamics
- **Warmup**: Adaptation phase (not included in posterior samples)

---

**Document Status**: Living document. Update when implementation changes.

**Next Steps**:
1. Add examples with real XPCS data
2. Document common convergence issues and solutions
3. Add performance benchmarks across platforms
4. Create troubleshooting guide for sampling problems
