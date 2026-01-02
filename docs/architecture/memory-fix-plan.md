# Memory Issue Fix Plan: NLSQ Large Dataset Optimization

**Status: IMPLEMENTED (Option 1 - Streaming Optimizer)**

> **Note (Jan 2026)**: This is a historical planning document. The original
> `StreamingOptimizer` referenced here was removed in NLSQ 0.4.0 and replaced
> with `AdaptiveHybridStreamingOptimizer`. See CLAUDE.md for current usage.

---

## Executive Summary

The homodyne NLSQ optimization fails with OOM (77% → 99.9%) when fitting 23M data points with 53 parameters. The root cause is the combination of:
1. Padded array materialization (1.8 GB static)
2. JAX autodiff keeping all intermediate grids for Jacobian computation (22+ GB)
3. NLSQ least_squares() computing dense Jacobian for all 23M residuals

This document proposes **three fix options** with implementation details.

---

## Root Cause Analysis

### Memory Timeline (from log)
```
21:01:20 - Optimization starts (77.4% = ~48GB used)
21:01:43 - Memory pressure warning (87.9% = 55GB)
21:02:15 - Critical (91.7% = 5.2GB available)
21:02:17 - Critical (97.4% = 1.7GB available)
21:02:38 - OOM (99.9% = 0.1GB available)
```

### Memory Breakdown

| Component | Estimated | Actual | Notes |
|-----------|-----------|--------|-------|
| Data arrays | 702 MB | 702 MB | 23M × 8 bytes × 4 arrays |
| Padded arrays | 1.8 GB | 1.8 GB | (231, 99981) × 5 arrays × 8 bytes |
| Theory grids (per eval) | 425 MB | 425 MB | 231 chunks × 23 angles × 10K grid |
| Jacobian (dense) | 9.7 GB | 9.7 GB | 23M × 53 × 8 bytes |
| Jacobian autodiff intermediates | - | **22+ GB** | **MISSING FROM ESTIMATE** |
| JAX compilation cache | - | 5-10 GB | **MISSING FROM ESTIMATE** |

**The problem**: JAX's autodiff for Jacobian computation keeps ALL intermediate theory grids in memory for backpropagation. With 53 parameters and 231 chunks, this creates ~22 GB of temporary buffers that cannot be freed during optimization.

---

## Fix Options

### Option 1: Use Streaming Optimizer in Homodyne (Recommended)

**Impact**: Changes homodyne's NLSQ wrapper to use NLSQ's streaming mode
**Effort**: Medium
**Memory**: ~50KB per batch (vs 30+ GB current)
**Convergence**: Slower (gradient descent) but memory-bounded

#### Implementation

**File**: `homodyne/optimization/nlsq/wrapper.py`

Add new method `_fit_with_streaming_optimizer()`:

```python
def _fit_with_streaming_optimizer(
    self,
    stratified_data: Any,
    per_angle_scaling: bool,
    physical_param_names: list[str],
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None,
    logger: Any,
    batch_size: int = 10_000,
    max_epochs: int = 50,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Use NLSQ streaming optimizer for unlimited dataset size.

    This method uses mini-batch gradient descent instead of full Jacobian
    computation, enabling fitting of datasets that don't fit in memory.
    """
    from nlsq.streaming_optimizer import StreamingOptimizer, StreamingConfig

    # Create streaming config
    config = StreamingConfig(
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=0.001,
        optimizer="adam",
        enable_fault_tolerance=True,
        validate_numerics=True,
        min_success_rate=0.5,
        checkpoint_frequency=1000,
    )

    # Create data generator (yields batches, never loads full dataset)
    def data_generator():
        for chunk in stratified_data.chunks:
            # Yield mini-batches from each chunk
            n = len(chunk.phi)
            for i in range(0, n, batch_size):
                end = min(i + batch_size, n)
                yield {
                    'phi': chunk.phi[i:end],
                    't1': chunk.t1[i:end],
                    't2': chunk.t2[i:end],
                    'g2': chunk.g2[i:end],
                    'q': chunk.q,
                    'L': chunk.L,
                    'dt': chunk.dt,
                }

    # Create residual function for streaming
    def streaming_residual_fn(params, batch):
        # Compute theory and residuals for single batch
        # ... (implementation similar to current _compute_single_chunk_residuals)
        pass

    optimizer = StreamingOptimizer(config)
    result = optimizer.fit(
        data_source=data_generator,
        func=streaming_residual_fn,
        p0=initial_params,
        bounds=bounds,
    )

    return result['x'], result.get('covariance'), result
```

**Selection logic** in `fit()`:
```python
# Add memory threshold check
n_points = len(stratified_data.g2)
estimated_memory = n_points * 53 * 8 * 3  # Jacobian + intermediates

if estimated_memory > available_memory * 0.5:
    logger.info("Dataset too large for least_squares(), using streaming optimizer")
    return self._fit_with_streaming_optimizer(...)
else:
    return self._fit_with_stratified_least_squares(...)
```

---

### Option 2: Add Memory-Bounded Jacobian to NLSQ (Alternative)

**Impact**: Changes NLSQ to compute Jacobian in chunks
**Effort**: High
**Memory**: Configurable (e.g., 4GB Jacobian chunks)
**Convergence**: Same as current (Levenberg-Marquardt)

#### Implementation

**File**: `/home/wei/Documents/GitHub/NLSQ/nlsq/least_squares.py`

Add chunked Jacobian computation:

```python
def compute_jacobian_chunked(
    residual_fn: Callable,
    params: jnp.ndarray,
    n_residuals: int,
    chunk_size: int = 500_000,
    memory_limit_gb: float = 4.0,
) -> jnp.ndarray:
    """Compute Jacobian in memory-bounded chunks.

    Instead of computing the full n_residuals × n_params Jacobian at once,
    compute it in chunks and concatenate. This reduces peak memory from
    O(n_residuals × n_params) to O(chunk_size × n_params).
    """
    n_params = len(params)

    # Adjust chunk size based on memory limit
    bytes_per_element = 8  # float64
    max_elements = int(memory_limit_gb * 1e9 / bytes_per_element)
    chunk_size = min(chunk_size, max_elements // n_params)

    jacobian_chunks = []

    for start in range(0, n_residuals, chunk_size):
        end = min(start + chunk_size, n_residuals)

        # Create sliced residual function
        def sliced_residual(p):
            full_residuals = residual_fn(p)
            return full_residuals[start:end]

        # Compute Jacobian for this chunk
        chunk_jac = jax.jacrev(sliced_residual)(params)
        jacobian_chunks.append(np.asarray(chunk_jac))

        # Force garbage collection between chunks
        del chunk_jac
        gc.collect()

    return np.concatenate(jacobian_chunks, axis=0)
```

**Integration** in `LevenbergMarquardt.step()`:
```python
if memory_bounded and n_residuals > threshold:
    jacobian = compute_jacobian_chunked(
        residual_fn, params, n_residuals,
        memory_limit_gb=self.config.memory_limit_gb
    )
else:
    jacobian = jax.jacrev(residual_fn)(params)
```

---

### Option 3: Hybrid Approach (Best)

Combine streaming for initial optimization with memory-bounded least_squares for refinement.

#### Phase 1: Streaming Coarse Optimization
- Use streaming optimizer for 2-5 epochs
- Gets close to solution with minimal memory
- Typical: 2-5 minutes

#### Phase 2: Memory-Bounded Refinement
- Use chunked Jacobian least_squares()
- Better convergence from Newton-based method
- Process 5M points per chunk (5 chunks for 23M)
- Typical: 5-10 minutes

**Total**: ~10-15 minutes with <8GB peak memory

---

## Recommended Implementation Order

### Phase 1: Quick Fix (Homodyne-side)

1. Add streaming optimizer option to `NLSQWrapper`
2. Add memory estimation before optimization
3. Auto-select streaming when estimated memory > threshold
4. Update CLAUDE.md with streaming configuration

**Files to modify**:
- `homodyne/optimization/nlsq/wrapper.py`
- `homodyne/optimization/nlsq/core.py` (add streaming config)
- `homodyne/config/defaults/nlsq_defaults.yaml` (add streaming params)

### Phase 2: Performance Improvement (NLSQ-side)

1. Add `compute_jacobian_chunked()` to NLSQ
2. Integrate into `LevenbergMarquardt` algorithm
3. Add memory_limit configuration
4. Benchmark against streaming on 23M dataset

**Files to modify**:
- `/home/wei/Documents/GitHub/NLSQ/nlsq/least_squares.py`
- `/home/wei/Documents/GitHub/NLSQ/nlsq/config.py`
- `/home/wei/Documents/GitHub/NLSQ/nlsq/memory_manager.py`

### Phase 3: Hybrid Mode (Optional)

1. Implement two-phase optimization
2. Add seamless transition from streaming to LM
3. Auto-tune based on convergence metrics

---

## Configuration Recommendations

For 23M-point datasets on 64GB system:

```yaml
optimization:
  nlsq:
    # Memory threshold for streaming mode (bytes)
    streaming_threshold: 8_000_000_000  # 8GB

    # Streaming optimizer settings
    streaming:
      batch_size: 10_000
      max_epochs: 50
      learning_rate: 0.001
      optimizer: "adam"
      checkpoint_frequency: 1000

    # Memory-bounded Jacobian settings (if using Option 2)
    memory_bounded:
      enabled: true
      jacobian_chunk_size: 500_000
      memory_limit_gb: 4.0
```

---

## Memory Estimates After Fix

| Option | Peak Memory | Time (23M points) |
|--------|-------------|-------------------|
| Current (broken) | >62 GB | OOM |
| Option 1 (streaming) | ~2 GB | 15-30 min |
| Option 2 (chunked Jacobian) | ~8 GB | 10-15 min |
| Option 3 (hybrid) | ~6 GB | 10-15 min |

---

## Testing Plan

1. **Unit test**: Streaming optimizer with synthetic XPCS data (10K points)
2. **Integration test**: Full pipeline with cached C020 data (23M points)
3. **Memory profiling**: Verify peak memory stays below threshold
4. **Convergence test**: Compare parameter estimates across methods
5. **Performance benchmark**: Time comparison on various dataset sizes

---

## Decision Matrix

| Factor | Option 1 (Streaming) | Option 2 (Chunked Jac) | Option 3 (Hybrid) |
|--------|---------------------|------------------------|-------------------|
| Implementation effort | Medium | High | High |
| Memory efficiency | Excellent | Good | Excellent |
| Convergence quality | Fair | Excellent | Excellent |
| Time to solution | Slower | Faster | Balanced |
| Code complexity | Low | Medium | High |
| Maintenance burden | Low | Medium | High |

**Recommendation**: Start with **Option 1** (streaming) as a quick fix, then implement **Option 2** (chunked Jacobian) for better convergence if needed.
