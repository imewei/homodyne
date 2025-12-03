# CMC Dual-Mode Strategy

**Status**: CMC-only architecture IMPLEMENTED ✓ (v2.4.1)

**Last Updated**: December 2, 2025 (v2.4.1)

> **Note**: As of v2.4.1, MCMC always uses CMC. The NUTS auto-selection logic described
> in earlier versions has been removed. Single-shard runs still use NUTS internally.

## Overview

Consensus Monte Carlo (CMC) is the mandatory MCMC backend in Homodyne v2.4.1+. All MCMC
runs use CMC with automatic sharding based on dataset characteristics and hardware.

## Table of Contents

1. [Dual-Criteria Decision Logic (IMPLEMENTED)](#dual-criteria-decision-logic-implemented)
1. [Current Sharding Strategy (IMPLEMENTED)](#current-sharding-strategy-implemented)
1. [Future Enhancement: Dual-Mode Sharding (PLANNED)](#future-enhancement-dual-mode-sharding-planned)
1. [Implementation Details](#implementation-details)
1. [Testing](#testing)
1. [References](#references)

______________________________________________________________________

## Dual-Criteria Decision Logic (IMPLEMENTED)

### Use Cases

CMC addresses two distinct computational challenges:

#### Use Case 1: Parallelism (Many Independent Samples)

- **Trigger**: `num_samples >= min_samples_for_cmc` (default: **15** in v2.1.0)
- **Scenario**: 20 phi angles × 10M points each
- **Sharding**: Split samples (phi angles) across shards
- **Benefit**: Parallel MCMC chains, faster convergence
- **Example**: 20 phi → 4 shards × 5 phi each on 14-core CPU

#### Use Case 2: Memory Management (Few Samples, Huge Data)

- **Trigger**: `dataset_size` causes estimated memory > **30%** threshold (v2.1.0)
- **Scenario**: 5 phi angles × 50M+ points each
- **Sharding**: Keep all samples in each shard, split data points (NOT YET IMPLEMENTED)
- **Benefit**: Avoid OOM errors, enable large dataset analysis
- **Example**: 5 phi × 50M → 4 shards × 5 phi × 12.5M points each

### Decision Logic (OR Condition)

**Use CMC if EITHER:**

1. `num_samples >= min_samples_for_cmc` (parallelism mode), OR
1. `estimated_memory_gb > threshold × available_memory` (memory mode)

**Otherwise:** Use standard NUTS

### Memory Estimation Formula

```python
estimated_memory_gb = (dataset_size × 8 bytes × 30) / 1e9
```

**Multiplier Breakdown (v2.1.0 - empirically calibrated):**

- Original data: 1×
- Gradients for 9 params: 9×
- NUTS trajectory storage: 15×
- JAX overhead + MCMC state: 5×

**Default Threshold**: **30%** of available GPU/CPU memory (v2.1.0, conservative for OOM
prevention)

### Implementation

**File**: `homodyne/device/config.py`

**Function**:
`should_use_cmc(num_samples, hardware_config, dataset_size=None, memory_threshold_pct=0.30, min_samples_for_cmc=15)`
(v2.1.0)

**Example (v2.1.0)**:

```python
from homodyne.device.config import detect_hardware, should_use_cmc

hw = detect_hardware()

# Case 1: Few samples, small data → NUTS
use_cmc = should_use_cmc(10, hw, dataset_size=5_000_000)
# Result: False (10 < 15, estimated memory < 30%)

# Case 2: Many samples → CMC (parallelism)
use_cmc = should_use_cmc(20, hw, dataset_size=10_000_000)
# Result: True (20 >= 15 samples, parallelism mode)

# Case 3: Few samples, HUGE data → CMC (memory)
use_cmc = should_use_cmc(5, hw, dataset_size=50_000_000)
# Result: True (12 GB = 37.5% > 30% threshold, memory mode)
```

______________________________________________________________________

## Current Sharding Strategy (IMPLEMENTED)

### Sample-Level Sharding Only

**Current Implementation**: CMC coordinator shards by **independent samples** (phi
angles) only.

**How It Works**:

```python
# For 200 phi angles with 10 shards:
shard_size = 200 // 10 = 20 phi per shard

# Each shard processes:
# - Shard 1: phi[0:20]   × all time points
# - Shard 2: phi[20:40]  × all time points
# - ...
# - Shard 10: phi[180:200] × all time points
```

**Pros**:

- Simple, well-tested implementation
- Natural parallelism for XPCS (phi angles are independent)
- Works perfectly for Use Case 1 (many samples)

**Cons**:

- Cannot split data within samples for Use Case 2
- Memory-triggered CMC currently triggers sample sharding, which doesn't help if you
  only have 2 phi angles

______________________________________________________________________

## Future Enhancement: Dual-Mode Sharding (PLANNED)

### Problem Statement

**Current Limitation**: When CMC is triggered by memory (Use Case 2), the system still
uses sample-level sharding. For datasets with few samples (e.g., 2 phi) but massive data
(100M+ points), this provides NO memory benefit because each shard still gets all data
points.

**Example Failure**:

```
Dataset: 2 phi × 100M points = 200M total
Memory requirement: 9.6 GB (triggers CMC via memory threshold)
Current sharding: 2 shards × 1 phi × 100M points each
Problem: Each shard still needs 4.8 GB (doesn't solve OOM issue!)
```

### Proposed Solution: Adaptive Sharding Strategy

Implement **two distinct sharding modes** based on which condition triggered CMC:

#### Mode 1: Sample-Level Sharding (Parallelism Mode)

- **Triggered by**: `num_samples >= min_samples_for_cmc`
- **Strategy**: Split samples (phi angles) across shards
- **Implementation**: CURRENT (already exists)
- **Example**: 200 phi × 1M → 10 shards × 20 phi × 1M

#### Mode 2: Data-Level Sharding (Memory Mode)

- **Triggered by**: Memory threshold exceeded
- **Strategy**: Split data points (time indices) within each sample
- **Implementation**: FUTURE ENHANCEMENT
- **Example**: 2 phi × 100M → 4 shards × 2 phi × 25M points

#### Hybrid Mode: Combined Sharding

- **Triggered by**: BOTH conditions met
- **Strategy**: Split both samples AND data
- **Implementation**: FUTURE ENHANCEMENT (low priority)
- **Example**: 200 phi × 100M → 40 shards × 5 phi × 25M points

### Technical Design (Data-Level Sharding)

#### 1. Shard Configuration

```python
@dataclass
class CMCShardConfig:
    """Configuration for CMC sharding strategy."""
    mode: Literal["sample", "data", "hybrid"]
    num_shards: int

    # Sample-level sharding
    samples_per_shard: Optional[int] = None
    sample_indices: Optional[List[np.ndarray]] = None

    # Data-level sharding
    data_points_per_shard: Optional[int] = None
    data_indices: Optional[List[np.ndarray]] = None
```

#### 2. Data Chunking

```python
def create_data_level_shards(
    data: np.ndarray,  # Shape: (n_phi, n_t1, n_t2)
    num_shards: int,
) -> List[np.ndarray]:
    """Split data along time dimensions while keeping all samples.

    Returns
    -------
    shards : List[np.ndarray]
        Each shard has shape (n_phi, n_t1_chunk, n_t2_chunk)
        All phi angles preserved in each shard
    """
    n_phi, n_t1, n_t2 = data.shape
    chunk_size = (n_t1 * n_t2) // num_shards

    # Flatten time dimensions
    data_flat = data.reshape(n_phi, -1)  # (n_phi, n_t1*n_t2)

    shards = []
    for i in range(num_shards):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_shards - 1 else n_t1 * n_t2

        shard_data = data_flat[:, start_idx:end_idx]
        shards.append(shard_data)

    return shards
```

#### 3. MCMC Execution per Shard

```python
# Each shard runs MCMC on its data chunk
def run_mcmc_on_data_shard(
    shard_data: np.ndarray,  # (n_phi, n_time_chunk)
    shard_indices: np.ndarray,  # Which time indices this shard covers
    model_params: Dict,
) -> MCMCResult:
    """Run MCMC on a data chunk.

    Key Difference from Sample Sharding:
    - All phi angles present (n_phi unchanged)
    - Subset of time points (n_time_chunk < n_time_total)
    - Log-likelihood computed only on this chunk's data
    """
    # Compute log-likelihood on chunk
    def log_likelihood_chunk(params):
        # Generate theory for full time range
        theory_full = compute_theory(params, all_time_points)

        # Extract only this shard's indices
        theory_chunk = theory_full[..., shard_indices]

        # Compare with observed data chunk
        return gaussian_log_likelihood(shard_data, theory_chunk)

    # Run MCMC
    result = run_nuts(log_likelihood_chunk, ...)
    return result
```

#### 4. Consensus Combination

```python
def combine_data_level_results(
    shard_results: List[MCMCResult],
    shard_weights: Optional[np.ndarray] = None,
) -> MCMCResult:
    """Combine MCMC results from data-level shards.

    Weighting Strategy:
    - Equal weights if shards have similar sizes
    - Weighted by data points if shard sizes differ

    Returns
    -------
    consensus_result : MCMCResult
        Combined parameter estimates and uncertainties
    """
    if shard_weights is None:
        # Equal weighting
        shard_weights = np.ones(len(shard_results)) / len(shard_results)

    # Weighted average of posterior means
    combined_params = np.average(
        [r.posterior_mean for r in shard_results],
        weights=shard_weights,
        axis=0
    )

    # Weighted variance (accounts for within-shard + between-shard variance)
    within_var = np.average(
        [r.posterior_var for r in shard_results],
        weights=shard_weights,
        axis=0
    )
    between_var = np.var(
        [r.posterior_mean for r in shard_results],
        axis=0
    )
    combined_var = within_var + between_var

    return MCMCResult(
        posterior_mean=combined_params,
        posterior_var=combined_var,
        ...
    )
```

### Validation Requirements

Before merging data-level sharding, validate:

1. **Convergence**: Data-level shards converge to same parameters
1. **Uncertainty Quantification**: Combined variance properly accounts for sharding
1. **Memory Reduction**: Actual memory usage reduced proportionally
1. **Performance**: Overhead from sharding < benefit from parallelism
1. **Edge Cases**: Single-shard fallback, unequal shard sizes

______________________________________________________________________

## Implementation Details

### Files Modified (Current Implementation)

1. **`homodyne/device/config.py`** (lines 312-418)

   - Function: `should_use_cmc()`
   - Added `dataset_size` parameter
   - Implemented dual-criteria OR logic
   - Memory estimation and threshold checking

1. **`homodyne/optimization/mcmc.py`** (lines 336-373)

   - Added `num_samples` calculation from data shape
   - Pass `dataset_size` to `should_use_cmc()`
   - Enhanced logging for transparency

1. **`homodyne/optimization/cmc/backends/selection.py`** (lines 150-158)

   - Fixed 'auto' backend selection handling
   - Added explicit logging for 'auto' mode

### Files to Modify (Future Enhancement)

1. **`homodyne/optimization/cmc/coordinator.py`**

   - Add `CMCShardConfig` dataclass
   - Implement `create_data_level_shards()`
   - Add sharding mode selection logic
   - Update shard validation for data-level shards

1. **`homodyne/optimization/cmc/consensus.py`**

   - Implement `combine_data_level_results()`
   - Add weighted variance calculation
   - Handle mixed sharding modes

1. **`homodyne/optimization/mcmc.py`**

   - Pass sharding mode hint to CMC coordinator
   - Add configuration option for manual mode override

### Configuration Options (Future)

Add to YAML config:

```yaml
mcmc:
  # v2.1.0: Automatic NUTS/CMC selection thresholds
  min_samples_for_cmc: 15              # Parallelism threshold (default: 15)
  memory_threshold_pct: 0.30           # Memory management threshold (default: 30%)

  cmc:
    sharding:
      mode: auto  # Options: auto, sample, data, hybrid

      # Advanced options (optional)
      force_data_sharding: false
      max_shard_size_gb: 2.0
```

______________________________________________________________________

## Testing

### Current Tests (Implemented)

**Test Script**: `/home/wei/Desktop/test_cmc_decision.py`

**Test Cases**:

1. Few samples (2 phi), small data (1M) → NUTS ✓
1. Many samples (200 phi), moderate data (10M) → CMC (parallelism) ✓
1. Few samples (2 phi), huge data (200M) → CMC (memory) ✓
1. Moderate samples (23 phi), no dataset_size → NUTS ✓

**Test Results**: ALL TESTS PASSED

### Future Tests (Planned)

1. **Unit Tests** for data-level sharding:

   ```python
   def test_data_level_shard_creation():
       """Verify data chunks preserve all samples."""
       data = np.random.rand(2, 1000, 1000)  # 2 phi × 1M points
       shards = create_data_level_shards(data, num_shards=4)

       # Each shard should have all phi angles
       assert all(s.shape[0] == 2 for s in shards)

       # Total data points should be preserved
       assert sum(s.size for s in shards) == data.size
   ```

1. **Integration Tests** for MCMC convergence:

   ```python
   def test_data_sharding_convergence():
       """Verify data-level shards converge to same parameters."""
       # Generate synthetic data
       true_params = [0.5, 1.2, 0.01]
       data = generate_xpcs_data(true_params, n_phi=2, n_points=10_000_000)

       # Run with data-level sharding
       result_sharded = fit_mcmc_jax(data, method='cmc', sharding_mode='data')

       # Run without sharding (on subset for speed)
       result_full = fit_mcmc_jax(data[:, :1000], method='nuts')

       # Parameters should match within uncertainty
       np.testing.assert_allclose(
           result_sharded.params,
           result_full.params,
           atol=3 * np.sqrt(result_sharded.variance)
       )
   ```

1. **Memory Tests**:

   ```python
   def test_memory_reduction():
       """Verify actual memory usage reduced with data sharding."""
       import psutil
       process = psutil.Process()

       # Baseline memory
       mem_before = process.memory_info().rss

       # Run with data sharding (should fit in memory)
       data = np.random.rand(2, 100_000_000)  # 2 phi × 100M
       result = fit_mcmc_jax(data, method='cmc', sharding_mode='data', num_shards=8)

       mem_peak = process.memory_info().rss
       mem_used_gb = (mem_peak - mem_before) / 1e9

       # Should use < 50% of available memory
       assert mem_used_gb < hw.memory_per_device_gb * 0.5
   ```

______________________________________________________________________

## References

### Internal Documentation

- `homodyne/device/config.py` - Hardware detection and CMC decision logic
- `homodyne/optimization/mcmc.py` - MCMC fitting with automatic method selection
- `homodyne/optimization/cmc/coordinator.py` - CMC workflow orchestration
- `docs/troubleshooting/cmc-decision-flowchart.md` - Visual decision tree (TODO)

### Scientific Background

- Consensus Monte Carlo (Scott et al., 2016): "Bayes and big data: The consensus Monte
  Carlo algorithm"
- XPCS Theory (He et al., 2024): https://doi.org/10.1073/pnas.2401162121

### Implementation Notes

- Current implementation prioritizes sample-level sharding (well-tested, natural for
  XPCS)
- Data-level sharding requires careful validation of likelihood factorization
- Hybrid mode (both sample AND data sharding) is low priority—rare use case

______________________________________________________________________

**Document Status**: Living document. Update when implementation progresses.

**Next Steps**:

1. Validate current dual-criteria logic with real XPCS datasets
1. Design detailed API for data-level sharding
1. Implement prototype and benchmark memory usage
1. Add comprehensive tests before merging
