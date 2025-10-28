# CMC Decision Logic - Quick Reference

**Last Updated**: October 28, 2025

## When Does CMC Trigger? (OR Logic)

```python
use_cmc = (num_samples >= 100) OR (estimated_memory > 50% of available)
```

## Three Scenarios

| Scenario | Samples | Dataset Size | Memory | Result | Reason |
|----------|---------|--------------|--------|--------|--------|
| 1. Small | 23 phi | 23M points | 1.1 GB (7%) | **NUTS** | Neither threshold met |
| 2. Many Samples | 200 phi | 10M points | 0.5 GB (3%) | **CMC** | 200 >= 100 samples |
| 3. Huge Data | 2 phi | 200M points | 9.6 GB (60%) | **CMC** | 60% > 50% memory |

## Memory Calculation

```python
# Estimate MCMC memory requirement
estimated_memory_gb = (dataset_size × 8 bytes × 6) / 1e9

# Where multiplier 6 accounts for:
# - Original data: 1×
# - Gradients: 2×
# - MCMC state: 3×
```

## Manual Override

```yaml
# In YAML config
mcmc:
  method: cmc  # Force CMC
  # OR
  method: nuts  # Force NUTS
  # OR
  method: auto  # Automatic (default)
```

## Code Reference

**Function**: `homodyne/device/config.py:should_use_cmc()`

```python
from homodyne.device.config import detect_hardware, should_use_cmc

hw = detect_hardware()

# Typical XPCS scenario: 2 phi × 100M each
use_cmc = should_use_cmc(
    num_samples=2,
    hardware_config=hw,
    dataset_size=200_000_000,  # Total points
    memory_threshold_pct=0.5,  # 50% threshold
    min_samples_for_cmc=100    # Parallelism threshold
)
# Result: True (memory mode triggered)
```

## Current Limitation

**Data-level sharding NOT YET IMPLEMENTED**

When memory-triggered CMC runs with few samples (e.g., 2 phi), it still uses sample-level sharding, which provides NO memory benefit. Future enhancement will implement data-level sharding (splitting time points within samples).

**Workaround**: Use NLSQ optimization (JAX-based, memory-efficient) instead of MCMC for large datasets with few samples.

## See Also

- Full documentation: `docs/architecture/cmc-dual-mode-strategy.md`
- Implementation: `homodyne/device/config.py:312-418`
- Tests: `/home/wei/Desktop/test_cmc_decision.py`
