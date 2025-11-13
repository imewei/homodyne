# CMC RNG Seeding and Initialization Behavior

**Version:** 2.4.0+
**Last Updated:** 2025-11-13
**Status:** Production

---

## Overview

This document describes the RNG seeding strategy for CMC parallel execution and the auto-initialization behavior for per-angle scaling parameters when manual values are not provided.

---

## Per-Shard RNG Seeding

### Deterministic Seeding Strategy

Each CMC shard receives a **deterministic RNG seed** derived from its shard index:

```python
# In multiprocessing.py worker function (line 708-717)
rng_seed = shard_idx  # Shard 0 → seed 0, Shard 1 → seed 1, etc.
rng_key = jax.random.PRNGKey(rng_seed)

worker_logger.info(
    f"Multiprocessing shard {shard_idx}: Starting MCMC with RNG seed={rng_seed}"
)

mcmc.run(rng_key)
```

### Benefits

- **Reproducibility**: Same shard index always produces same samples (given same data)
- **Observability**: Explicit seed logging enables debugging and validation
- **Isolation**: Each shard's RNG stream is independent (no cross-contamination)

### Logging Output

```
INFO: Multiprocessing shard 0: Starting MCMC with RNG seed=0
INFO: Multiprocessing shard 1: Starting MCMC with RNG seed=1
INFO: Multiprocessing shard 2: Starting MCMC with RNG seed=2
...
```

### Configuration

**Current behavior:** Seeds are automatically derived from shard index (not user-configurable).

**Future extension:** To support custom seed offsets, add to config:

```yaml
optimization:
  mcmc:
    cmc:
      rng_seed_offset: 42  # Optional: offset all shard seeds by 42
                           # Shard 0 → seed 42, Shard 1 → seed 43, etc.
```

---

## Per-Angle Scaling Auto-Initialization

### Open-Interval Clamp with Auto-Init Fallback

When per-angle scaling parameters (contrast/offset) are not provided in `initial_parameters.values`, they are **automatically estimated from data statistics** with a robust open-interval clamping strategy.

### Algorithm (lines 605-669 in multiprocessing.py)

#### Step 1: Filter Invalid Data

```python
# Remove physically impossible values (c2 correlation must be >= 1.0)
valid_mask = shard["data"] >= 1.0
data_valid = shard["data"][valid_mask]

# Fallback if >50% invalid
if len(data_valid) < 0.5 * len(shard["data"]):
    logger.warning(f"Shard {shard_idx}: >50% invalid data, using full dataset")
    data_valid = shard["data"]
```

#### Step 2: Compute Robust Statistics

```python
# Use percentiles to avoid outliers
data_p05 = float(np.percentile(data_valid, 5))   # 5th percentile
data_p95 = float(np.percentile(data_valid, 95))  # 95th percentile
data_mean = float(np.mean(data_valid))
```

#### Step 3: Estimate Scaling Parameters

Physical model: `c2_fitted = offset + contrast × c2_theory`

Where `c2_theory ∈ [1.0, 2.0]` (decorrelated → fully correlated)

```python
# At c2_theory=1.0: c2_fitted ≈ data_p05
# At c2_theory=2.0: c2_fitted ≈ data_p95
# Solving: contrast = (data_p95 - data_p05), offset = data_p05 - contrast

estimated_contrast = max(0.01, data_p95 - data_p05)  # Open-interval minimum
estimated_offset = max(0.5, data_p05 - estimated_contrast)  # Physical minimum
```

#### Step 4: Apply to All Angles

```python
for phi_idx in range(len(phi_unique)):
    init_param_values[f"contrast_{phi_idx}"] = estimated_contrast
    init_param_values[f"offset_{phi_idx}"] = estimated_offset

# Remove scalar contrast/offset if present
if "contrast" in init_param_values:
    del init_param_values["contrast"]
if "offset" in init_param_values:
    del init_param_values["offset"]
```

### Clamping Strategy: Open-Interval Bounds

| Parameter | Hard Minimum | Auto-Init Minimum | Justification |
|-----------|--------------|-------------------|---------------|
| `contrast` | 0.0 (inclusive) | **0.01 (exclusive)** | Numerical stability: avoid zero gradients |
| `offset` | -∞ (unbounded) | **0.5 (exclusive)** | Physical constraint: c2 ≥ 1.0 at full decorrelation |

**Why open-interval?**

- **Prevents edge-case failures**: `contrast=0.0` causes zero gradient, breaking NUTS initialization
- **Physically motivated**: Real experimental data has non-zero contrast and offset ≥ 1.0
- **Robust initialization**: Ensures NumPyro can find valid starting point within 10 attempts

### Logging Output

```
INFO: Multiprocessing shard 0: Data statistics (robust): p05=1.0234, p95=1.4567, mean=1.2345
INFO: Multiprocessing shard 0: Estimated per-angle scaling: contrast=0.4333, offset=0.5901
INFO: Multiprocessing shard 0: Added 3 per-angle scaling parameters (contrast and offset) to init_param_values for NUTS initialization
INFO: Multiprocessing shard 0: init_param_values (first 10): {'D0': 1000.0, 'alpha': 0.5, 'D_offset': 10.0, 'gamma_dot_t0': 0.01, 'beta': 0.0, 'gamma_dot_t_offset': 0.0, 'phi0': 0.0, 'contrast_0': 0.4333, 'offset_0': 0.5901, 'contrast_1': 0.4333}
```

### Configuration Example

To **override auto-initialization** with manual values, add to config:

```yaml
initial_parameters:
  parameter_names:
    - D0
    - alpha
    - D_offset
    # ... (physical parameters)
    - contrast_0  # Manual per-angle contrast (3 angles)
    - contrast_1
    - contrast_2
    - offset_0    # Manual per-angle offset (3 angles)
    - offset_1
    - offset_2

  values:
    - 1000.0      # D0
    - 0.5         # alpha
    - 10.0        # D_offset
    # ... (physical parameters)
    - 0.45        # contrast_0 (manual value)
    - 0.50        # contrast_1
    - 0.48        # contrast_2
    - 0.95        # offset_0 (manual value)
    - 1.05        # offset_1
    - 1.00        # offset_2
```

**If not provided:** Auto-initialization estimates from data (recommended for first runs).

---

## Best Practices

### For Reproducibility

1. **Check seed logs**: Verify each shard uses expected seed
2. **Document shard count**: CMC shard count affects seed sequence
3. **Save RNG state**: Future extension could save/restore full RNG state

### For Initialization

1. **First run:** Let auto-init estimate from data (fast, data-driven)
2. **Refinement:** Use NLSQ results to manually set `initial_parameters.values`
3. **Validation:** Check logs for estimated contrast/offset values (should be reasonable)

### Common Issues

**Issue:** "Cannot find valid initial parameters" error

**Cause:** Auto-init estimates are outside prior bounds (rare with robust percentiles)

**Solution:** Manually set `initial_parameters.values` with physically reasonable values

---

**Issue:** Large variation in per-angle contrast/offset across shards

**Cause:** Dataset has strong angle-dependent properties or outliers

**Solution:** Use stratified sharding (`strategy: "stratified"`) to balance angles across shards

---

## Technical References

- **RNG seeding implementation**: `homodyne/optimization/cmc/backends/multiprocessing.py:708-717`
- **Auto-init implementation**: `homodyne/optimization/cmc/backends/multiprocessing.py:605-669`
- **Open-interval clamp bounds**: Lines 632, 633
- **Robust statistics**: Percentile-based (5th, 95th) to avoid outliers

---

## Version History

- **v2.4.0** (2025-11-13): Added per-shard RNG seed logging and documented auto-init behavior
- **v2.2.1** (2025-11-10): Introduced data-driven auto-init with open-interval clamps
- **v2.1.0** (2025-10-31): Removed automatic NLSQ/SVI initialization (manual workflow)

---

**Next Steps:**

- Add `rng_seed_offset` config option for custom seed control
- Consider saving/restoring RNG state for exact reproducibility across runs
- Add unit tests for auto-init edge cases (all invalid data, single angle, etc.)
