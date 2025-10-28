# Architecture Documentation

**Last Updated**: October 28, 2025

This directory contains architectural documentation for the homodyne XPCS analysis package, focusing on advanced topics like CMC (Consensus Monte Carlo), NUTS parallelization, and hardware-adaptive optimization.

---

## Quick Navigation

### CMC (Consensus Monte Carlo)

**Dual-Criteria Decision Logic**:
- 📘 [Full Documentation](cmc-dual-mode-strategy.md) - Comprehensive design doc (3,500+ words)
- 📋 [Quick Reference](cmc-decision-quick-reference.md) - Lookup table and examples

**Key Topics**:
- When CMC triggers (parallelism OR memory threshold)
- Memory estimation formula
- Current implementation vs future enhancements
- Sample-level vs data-level sharding (planned)

---

### NUTS Chain Parallelization

**Chain Execution Strategy**:
- 📘 [Full Documentation](nuts-chain-parallelization.md) - Complete guide to chain parallelism
- 📋 [Quick Reference](nuts-chain-parallelization-quick-reference.md) - Configuration presets and examples

**Key Topics**:
- Sequential (1 GPU) vs parallel (CPU/multi-GPU) execution
- Why 4 chains even on single GPU
- Convergence diagnostics (R-hat, ESS, divergences)
- Performance characteristics across platforms

---

### Backend Selection

**CMC Backend Strategy**:
- 📄 [Backend Selection Logic](../optimization/cmc/backends/selection.py) - Source code
- Topics covered in CMC documentation above

**Backends**:
- `pjit` - GPU/JAX backend (single or multi-GPU)
- `multiprocessing` - CPU parallel backend
- `pbs` - HPC cluster backend (PBS scheduler)
- `auto` - Automatic hardware-based selection

---

## Document Hierarchy

```
docs/architecture/
├── README.md                                    # This file
├── cmc-dual-mode-strategy.md                   # CMC comprehensive guide
├── cmc-decision-quick-reference.md             # CMC quick lookup
├── nuts-chain-parallelization.md               # NUTS comprehensive guide
└── nuts-chain-parallelization-quick-reference.md  # NUTS quick lookup
```

---

## Key Concepts

### CMC (Consensus Monte Carlo)

**Purpose**: Distributed MCMC for large datasets or many samples

**Triggering Conditions** (OR logic):
1. **Parallelism**: `num_samples >= 100` (many phi angles)
2. **Memory**: `dataset_size` exceeds 50% of available memory

**Example**:
```python
# 2 phi × 100M each = 200M total
# Memory: 9.6 GB = 60% of 16 GB → CMC triggered
use_cmc = should_use_cmc(num_samples=2, hw, dataset_size=200_000_000)
# Result: True (memory threshold exceeded)
```

### NUTS Chains

**Purpose**: Multiple independent MCMC chains for convergence diagnostics

**Default Configuration**:
- 4 chains (enables R-hat, ESS diagnostics)
- Sequential on single GPU (4× time)
- Parallel on CPU (1.1× time)

**Why Multiple Chains**:
- ✅ R-hat statistic (convergence detection)
- ✅ Effective Sample Size (true independent samples)
- ✅ Divergence detection (problematic regions)
- ✅ Better uncertainty quantification

---

## Decision Trees

### Method Selection (Automatic)

```
Dataset Analysis
├─ num_samples >= 100?
│  ├─ YES → CMC (parallelism mode)
│  └─ NO → Check memory
│     ├─ dataset_size > 50% memory?
│     │  ├─ YES → CMC (memory mode)
│     │  └─ NO → NUTS
│     └─ dataset_size unknown?
│        └─ NUTS
```

### Chain Configuration

```
Hardware Platform
├─ CPU (multi-core)?
│  └─ Use 4-8 chains (parallel)
├─ Single GPU?
│  └─ Use 4 chains (sequential, but with diagnostics)
└─ Multi-GPU (N)?
   └─ Use N chains (1 per GPU, parallel)
```

---

## Performance Summary

### Execution Time

| Method | Configuration | Single GPU | CPU (14-core) | Multi-GPU (4) |
|--------|---------------|------------|---------------|---------------|
| **NUTS** | 1 chain | T | 4T | T |
| **NUTS** | 4 chains | 4T | 1.1T | 1.1T |
| **CMC** | 4 shards | 1.5T* | 1.2T | 1.1T |

*Depends on shard size and overhead

### Memory Usage

| Method | Configuration | Peak Memory |
|--------|---------------|-------------|
| **NUTS** | 4 chains, 1 GPU | M (sequential, 1 at a time) |
| **NUTS** | 4 chains, CPU | 4M (all in RAM) |
| **CMC** | 4 shards | M/4 per shard |

Where M = dataset_size × 8 bytes × 6 (MCMC overhead)

---

## Configuration Examples

### Standard Analysis (Your Setup)

**Dataset**: 23 phi × 1M points each (23M total)

```yaml
# Auto-selection will choose NUTS
# (23 samples < 100, memory OK)
mcmc:
  method: auto
  n_chains: 4      # Good diagnostics
  n_samples: 1000
  n_warmup: 500
```

### Large Sample Count

**Dataset**: 200 phi × 1M points each

```yaml
# Auto-selection will choose CMC
# (200 samples >= 100 threshold)
mcmc:
  method: auto  # → CMC
  cmc:
    backend:
      type: auto  # → pjit (single GPU)
```

### Huge Dataset, Few Samples

**Dataset**: 2 phi × 100M points each (200M total)

```yaml
# Auto-selection will choose CMC
# (Memory: 9.6 GB = 60% > 50%)
mcmc:
  method: auto  # → CMC
  cmc:
    backend:
      type: auto  # → pjit
```

**Current Limitation**: Data-level sharding not yet implemented, so CMC won't reduce memory per shard. Workaround: Use NLSQ instead.

---

## Implementation Status

### ✅ Implemented

1. **Dual-Criteria CMC Decision**
   - Parallelism-based triggering (num_samples)
   - Memory-based triggering (dataset_size)
   - OR logic (either condition triggers CMC)

2. **NUTS Chain Parallelization**
   - CPU parallel chains (via `set_host_device_count`)
   - GPU sequential chains (automatic)
   - Multi-GPU distribution

3. **Backend Selection**
   - Auto-detection (hardware-based)
   - Manual override support
   - Platform compatibility validation

### 🚧 Planned

1. **Data-Level Sharding** (v2.1.0)
   - Split time points within samples
   - Enable memory reduction for few-sample scenarios
   - Adaptive sharding mode selection

2. **Hybrid Sharding** (v3.0.0)
   - Both sample AND data sharding
   - Automatic shard size optimization
   - Custom sharding strategies

---

## Troubleshooting

### CMC Issues

**Problem**: CMC triggered but memory not reduced
- **Cause**: Data-level sharding not implemented
- **Solution**: Use NLSQ for large datasets with few samples
- **Future**: Data-level sharding in v2.1.0

**Problem**: Backend 'auto' error
- **Cause**: Fixed in current version
- **Solution**: Update to latest code

### NUTS Issues

**Problem**: Chains not converging (R-hat > 1.01)
- **Solution 1**: Increase samples (`n_samples: 2000`)
- **Solution 2**: Increase warmup (`n_warmup: 1000`)
- **Solution 3**: Use NLSQ initialization (automatic)

**Problem**: Out of memory with 4 chains on CPU
- **Solution**: Reduce chains (`n_chains: 2`)

**Problem**: Slow execution on single GPU
- **Explanation**: Sequential execution is expected
- **Solution**: Use `n_chains: 1` for faster results (no diagnostics)

---

## Related Documentation

### User Guides
- [Configuration Templates](../configuration-templates/) - YAML examples
- [CLI Usage](../user-guide/cli-usage.md) - Command-line interface
- [Optimization Methods](../guides/optimization-methods.md) - NLSQ vs MCMC

### Developer Guides
- [Contributing](../developer-guide/contributing.md) - Development workflow
- [Testing](../developer-guide/testing.md) - Test suite organization

### Theoretical Background
- [XPCS Theory](../theoretical-framework/xpcs-theory.md) - Physics equations
- [Consensus Monte Carlo](https://arxiv.org/abs/1407.5628) - Scott et al. (2016)
- [NUTS Algorithm](https://arxiv.org/abs/1111.4246) - Hoffman & Gelman (2014)

---

## Changelog

**October 28, 2025**:
- ✅ Implemented dual-criteria CMC decision logic
- ✅ Fixed backend='auto' handling
- ✅ Added memory-based CMC triggering
- ✅ Documented NUTS chain parallelization
- ✅ Created comprehensive architecture docs

**Next Steps**:
- Implement data-level sharding for CMC
- Add performance benchmarks
- Create visual decision trees
- Add troubleshooting examples

---

## Questions?

**For Users**:
- Quick lookup: Use quick reference guides
- Detailed info: Read full documentation
- Configuration help: See YAML examples in `/configuration-templates`

**For Developers**:
- Implementation: Check source code references in docs
- Contributing: See `/developer-guide/contributing.md`
- Testing: Run test suite (`make test-all`)

---

**Document Status**: Living documentation. Updates as implementation progresses.
