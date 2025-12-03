# CMC Decision Logic - Quick Reference

**Last Updated**: December 2, 2025 (v2.4.1)

> **v2.4.1 Update**: CMC is now mandatory for all MCMC runs. The NUTS auto-selection
> logic has been removed. Single-shard runs still use NUTS internally.

## CMC-Only Architecture (v2.4.1+)

```python
# v2.4.1+: CMC is always used
use_cmc = True  # No decision logic needed
```

## Sharding Scenarios

| Scenario | Samples | Dataset Size | Shards | Notes |
|----------|---------|--------------|--------|-------|
| Small data | 3 phi | 5M points | 1 | Single-shard CMC (NUTS internally) |
| Medium data | 23 phi | 50M points | 4-8 | Multi-shard CMC |
| Large data | 5 phi | 200M points | 8+ | Memory-optimized sharding |

## Memory Calculation

```python
# Estimate MCMC memory requirement per shard
estimated_memory_gb = (dataset_size × 8 bytes × 6) / 1e9

# Where multiplier 6 accounts for:
# - Original data: 1×
# - Gradients: 2×
# - MCMC state: 3×
```

## Configuration (v2.4.1+)

```yaml
# In YAML config (method selection removed in v2.4.1)
mcmc:
  sharding:
    num_shards: auto  # Automatic shard count
    seed_base: 42     # RNG seed for reproducibility
```

## Code Reference

**Note**: `should_use_cmc()` is deprecated and always returns `True` in v2.4.1+.

```python
from homodyne.optimization.mcmc import fit_mcmc_jax

# All MCMC runs use CMC automatically
result = fit_mcmc_jax(data, config)
print(f"Shards used: {result.num_shards}")
```

## Current Limitation

**Data-level sharding NOT YET IMPLEMENTED**

When memory-triggered CMC runs with few samples (e.g., 2 phi), it still uses
sample-level sharding, which provides NO memory benefit. Future enhancement will
implement data-level sharding (splitting time points within samples).

**Workaround**: Use NLSQ optimization (JAX-based, memory-efficient) instead of MCMC for
large datasets with few samples.

## See Also

- Full documentation: `docs/architecture/cmc-dual-mode-strategy.md`
- Implementation: `homodyne/device/config.py:312-418`
- Tests: `/home/wei/Desktop/test_cmc_decision.py`
