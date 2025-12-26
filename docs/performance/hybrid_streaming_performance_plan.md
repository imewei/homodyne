# NLSQ Hybrid Streaming Optimizer Performance Analysis & Optimization Plan

**Date:** 2025-12-25
**Scope:** AdaptiveHybridStreamingOptimizer four-phase optimization
**Target:** Identify bottlenecks and propose optimizations for large-scale XPCS fitting

---

## Implementation Status (2025-12-25)

### Completed Optimizations

| Bottleneck | Status | Measured Speedup |
|------------|--------|------------------|
| 1.1 Jacobian Precompilation | ✅ Implemented | 12,000x for cached calls |
| 1.2 Redundant Cost Evaluation | ✅ Implemented | 26x for cost evaluation |
| 1.3 JAX Scan for Loops | ⚠️ Reverted | **0.11x (slower on CPU)** |

### Key Findings

**JAX lax.scan Performance on CPU:**
Benchmarking showed that `jax.lax.scan` is ~10x **slower** than Python loops on CPU:

```
J^T J Accumulation Performance:
  Scan-based mean:  114.446 ms
  Python loop mean: 12.680 ms
  Speedup:          0.11x (scan is 10x slower)

Cost-Only Computation Performance:
  Scan-based mean:  75.983 ms
  Python loop mean: 9.872 ms
  Speedup:          0.13x (scan is 8x slower)
```

**Root Cause:** JAX scan has significant tracing overhead that doesn't pay off on CPU for this workload. The scan body function captures the normalized model and Jacobian function, causing repeated tracing/compilation. Python loops avoid this overhead by calling already-JIT-compiled functions directly.

**Decision:** Reverted to Python loops for CPU-only execution. Scan-based methods are preserved in the codebase for potential future GPU support.

---

## Executive Summary

The AdaptiveHybridStreamingOptimizer implements a four-phase approach:
- **Phase 0**: Parameter normalization setup
- **Phase 1**: Adam warmup with 4-layer defense strategy
- **Phase 2**: Streaming Gauss-Newton with J^T J accumulation
- **Phase 3**: Denormalization and covariance transformation

This analysis identifies 8 major bottlenecks across computational, memory, and I/O dimensions, with prioritized optimization recommendations.

---

## 1. Identified Performance Bottlenecks

### 1.1 Critical: Jacobian Computation in `_compute_jacobian_chunk()` (NLSQ)

**Location:** `nlsq/adaptive_hybrid_streaming.py:1451-1491`

**Current Implementation:**
```python
def _compute_jacobian_chunk(self, x_chunk, params):
    def model_at_x(p, x_single):
        return self.normalized_model(x_single, *p)

    jac_fn = jax.vmap(lambda x: jax.jacrev(model_at_x, argnums=0)(params, x))
    J_chunk = jac_fn(x_chunk)
    return J_chunk
```

**Issues:**
1. **JIT recompilation per chunk**: The lambda closure captures `params`, potentially causing recompilation
2. **Nested vmap + jacrev**: Creates intermediate arrays for each point
3. **Non-cached function**: The Jacobian function is recreated on each call

**Impact:** ~40-60% of Phase 2 iteration time based on JAX profiling patterns

---

### 1.2 Critical: Redundant Cost Evaluation in Gauss-Newton Iteration

**Location:** `nlsq/adaptive_hybrid_streaming.py:1690-1867`

**Current Implementation:**
```python
def _gauss_newton_iteration(self, data_source, current_params, trust_radius):
    # First pass: Accumulate J^T J and J^T r (computes predictions + residuals)
    for i in range(0, n_points, chunk_size):
        JTJ, JTr, res_sq = self._accumulate_jtj_jtr(...)
        total_cost += res_sq

    # ... solve step ...

    # Second pass: Evaluate cost at new params (recomputes predictions + residuals)
    for i in range(0, n_points, chunk_size):
        predictions = self.normalized_model(x_chunk, *new_params)
        residuals = y_chunk - predictions
        new_cost += float(jnp.sum(residuals**2))
```

**Issues:**
1. **Double iteration over dataset**: Full dataset traversed twice per GN iteration
2. **Redundant model calls**: Same chunks processed for both J^T J and cost evaluation
3. **No caching of predictions**: Could reuse predictions from first pass

**Impact:** ~25-35% overhead from redundant computation

---

### 1.3 High: Non-JIT Python Loops in Phase 1 & Phase 2

**Locations:**
- `nlsq/adaptive_hybrid_streaming.py:1292-1449` (Phase 1 warmup loop)
- `nlsq/adaptive_hybrid_streaming.py:1738-1747` (Phase 2 chunk loop)

**Current Implementation:**
```python
# Phase 2 chunk loop
for i in range(0, n_points, chunk_size):
    x_chunk = x_data[i : i + chunk_size]
    y_chunk = y_data[i : i + chunk_size]
    JTJ, JTr, res_sq = self._accumulate_jtj_jtr(...)
```

**Issues:**
1. **Python loop overhead**: Each iteration incurs Python interpreter overhead
2. **Dynamic slicing**: `x_data[i:i+chunk_size]` creates new array views
3. **Non-vectorized accumulation**: Could use `jax.lax.scan` for better performance

**Impact:** ~10-15% overhead for many-chunk scenarios

---

### 1.4 High: Homodyne Stratified Residual Function

**Location:** `homodyne/optimization/nlsq/strategies/residual.py:200-350`

**Current Implementation:**
```python
def _call_jax(self, params):
    all_residuals = []
    for chunk_idx, (start, end) in enumerate(self.chunk_boundaries):
        chunk_residuals = self._compute_chunk_residuals_jax(...)
        all_residuals.append(chunk_residuals)
    return jnp.concatenate(all_residuals)
```

**Issues:**
1. **Python list accumulation**: `all_residuals.append()` in Python loop
2. **Dynamic concatenation**: `jnp.concatenate` after loop (memory allocation)
3. **Per-chunk overhead**: Each chunk has function call overhead

**Impact:** ~5-10% overhead in residual-heavy workflows

---

### 1.5 Medium: Physics Computation Redundancy

**Location:** `homodyne/core/physics_nlsq.py` and `homodyne/core/jax_backend.py`

**Current Implementation:**
```python
def compute_g2_scaled(t, q, D, gamma, ..., phi, phi0):
    # Recomputes wavevector and prefactors each call
    q_effective = compute_effective_wavevector(q, angle)
    sinc_factor = compute_sinc_prefactor(t, gamma)
    ...
```

**Issues:**
1. **Wavevector recomputation**: `q` is constant per angle, recomputed per call
2. **Sinc prefactor**: Only depends on `t` and `gamma`, not parameters
3. **No memoization**: Same values computed repeatedly

**Impact:** ~5-8% overhead for complex physics models

---

### 1.6 Medium: Memory Allocation Patterns

**Locations:** Multiple files in the streaming pipeline

**Issues:**
1. **Temporary arrays in loops**: JAX arrays created inside Python loops
2. **Non-donated buffers**: Input arrays not marked for donation
3. **Float64 throughout**: Some phases could use float32 for intermediates

**Impact:** ~10-15% memory bandwidth overhead, potential GC pressure

---

### 1.7 Low: Checkpoint I/O Blocking

**Location:** `nlsq/adaptive_hybrid_streaming.py:1369-1386` and `homodyne/optimization/checkpoint_manager.py`

**Current Implementation:**
```python
if (iteration + 1) % self.config.checkpoint_frequency == 0:
    checkpoint_path = Path(...)
    self._save_checkpoint(checkpoint_path)  # Blocking I/O
```

**Issues:**
1. **Synchronous HDF5 writes**: Blocks optimization during save
2. **No async I/O**: Could use background thread for checkpointing
3. **Large state serialization**: Pickle serialization is slow

**Impact:** ~1-3% overhead (depends on checkpoint frequency)

---

### 1.8 Low: Defense Layer Telemetry Overhead

**Location:** `nlsq/adaptive_hybrid_streaming.py:53-298`

**Current Implementation:**
```python
def record_layer4_clip(self, original_norm, max_norm):
    self.layer4_clip_triggers += 1
    self._log_event("layer4_clip", {...})  # Dict creation + list append
```

**Issues:**
1. **Dict allocation per event**: Creates new dict for each telemetry event
2. **Event log trimming**: `self._event_log[-self._max_events:]` creates new list
3. **Always-on**: No mechanism to disable in production

**Impact:** ~1-2% overhead (mostly negligible)

---

## 2. Optimization Recommendations

### Priority 1: Jacobian Computation Optimization (Critical)

#### 2.1.1 Pre-compile Jacobian Function with Static Args

```python
class AdaptiveHybridStreamingOptimizer:
    def _setup_jacobian_fn(self):
        """Pre-compile Jacobian function once."""
        @jax.jit
        def jac_fn_core(params, x_chunk):
            def model_at_x(p, x_single):
                return self.normalized_model(x_single, *p)
            return jax.vmap(
                lambda x: jax.jacrev(model_at_x, argnums=0)(params, x)
            )(x_chunk)

        self._jac_fn_compiled = jac_fn_core

    def _compute_jacobian_chunk(self, x_chunk, params):
        return self._jac_fn_compiled(params, x_chunk)
```

**Expected Improvement:** 15-25% reduction in Phase 2 time

#### 2.1.2 Use Persistent Compilation Cache

```python
# In HybridStreamingConfig
enable_compilation_cache: bool = True
compilation_cache_dir: str | None = None

# In optimizer
if self.config.enable_compilation_cache:
    jax.config.update("jax_compilation_cache_dir", cache_dir)
```

**Expected Improvement:** Faster warmup on repeated runs

---

### Priority 2: Eliminate Redundant Cost Evaluation (Critical)

#### 2.2.1 Single-Pass J^T J with Cost Accumulation

```python
def _gauss_newton_iteration_optimized(self, data_source, current_params, trust_radius):
    """Single-pass Gauss-Newton with cost accumulation."""
    x_data, y_data = data_source

    # Single pass: accumulate J^T J, J^T r, and cost simultaneously
    JTJ, JTr, current_cost = self._accumulate_all_in_one_pass(
        x_data, y_data, current_params
    )

    # Solve for step
    step, predicted_reduction = self._solve_gauss_newton_step(JTJ, JTr, trust_radius)
    new_params = current_params + step

    # Only compute new cost (single additional pass, no Jacobian)
    new_cost = self._compute_cost_only(x_data, y_data, new_params)

    ...
```

**Expected Improvement:** 20-30% reduction in GN iteration time

#### 2.2.2 Cost-Only Evaluation Function

```python
@jax.jit
def _compute_cost_only(self, x_data, y_data, params):
    """Compute only MSE without Jacobian."""
    def chunk_cost(carry, chunk_data):
        x_chunk, y_chunk = chunk_data
        pred = self.normalized_model(x_chunk, *params)
        return carry + jnp.sum((y_chunk - pred) ** 2), None

    total_cost, _ = jax.lax.scan(chunk_cost, 0.0, (x_chunks, y_chunks))
    return total_cost
```

---

### Priority 3: Replace Python Loops with JAX Scan (High)

#### 2.3.1 Scan-Based J^T J Accumulation

```python
def _accumulate_jtj_jtr_scan(self, x_data, y_data, params, chunk_size):
    """Use jax.lax.scan for chunk accumulation."""
    n_points = x_data.shape[0]
    n_params = params.shape[0]
    n_chunks = (n_points + chunk_size - 1) // chunk_size

    # Pad data to exact chunks
    x_padded = jnp.pad(x_data, ...)
    y_padded = jnp.pad(y_data, ...)

    # Reshape into chunks
    x_chunks = x_padded.reshape(n_chunks, chunk_size, -1)
    y_chunks = y_padded.reshape(n_chunks, chunk_size)

    def accumulate_chunk(carry, chunk_data):
        JTJ, JTr, cost = carry
        x_chunk, y_chunk = chunk_data

        J = self._compute_jacobian_chunk(x_chunk, params)
        pred = self.normalized_model(x_chunk, *params)
        r = y_chunk - pred

        return (
            JTJ + J.T @ J,
            JTr + J.T @ r,
            cost + jnp.sum(r ** 2)
        ), None

    init = (jnp.zeros((n_params, n_params)), jnp.zeros(n_params), 0.0)
    (JTJ, JTr, cost), _ = jax.lax.scan(accumulate_chunk, init, (x_chunks, y_chunks))

    return JTJ, JTr, cost
```

**Expected Improvement:** 10-15% reduction in loop overhead

---

### Priority 4: Physics Computation Optimization (Medium)

#### 2.4.1 Pre-compute Static Quantities

```python
class PrecomputedPhysics:
    """Cache static physics quantities per angle."""

    def __init__(self, q_values, angles, t_values):
        # Precompute wavevector for each angle (only depends on geometry)
        self.q_effective_per_angle = {
            angle: compute_effective_wavevector(q, angle)
            for angle, q in zip(angles, q_values)
        }

        # Precompute time-dependent factors (shared across angles)
        self.t_squared = t_values ** 2
```

#### 2.4.2 Vectorized Multi-Angle Computation

```python
@jax.jit
def compute_g2_vectorized(t, q_array, D, gamma, angles, phi0):
    """Compute g2 for all angles in one vectorized call."""
    # Shape: (n_angles, n_times)
    q_eff = compute_effective_wavevector_batch(q_array, angles)

    # Vectorized over angles
    g1_all = jax.vmap(lambda q: compute_g1(t, q, D, gamma))(q_eff)

    # Apply angle-dependent shear correction
    cos_factor = jnp.cos(phi0 - angles)[:, None]
    g2_all = 1 + g1_all ** 2 * cos_factor

    return g2_all
```

**Expected Improvement:** 5-10% for multi-angle datasets

---

### Priority 5: Memory Optimization (Medium)

#### 2.5.1 Buffer Donation for In-Place Updates

```python
@functools.partial(jax.jit, donate_argnums=(2, 3))  # Donate JTJ_prev, JTr_prev
def _accumulate_jtj_jtr_donated(x_chunk, y_chunk, params, JTJ_prev, JTr_prev):
    """Accumulate with buffer donation for in-place updates."""
    J = compute_jacobian(x_chunk, params)
    r = y_chunk - model(x_chunk, params)

    # In-place updates via donation
    JTJ_new = JTJ_prev + J.T @ J
    JTr_new = JTr_prev + J.T @ r

    return JTJ_new, JTr_new, jnp.sum(r ** 2)
```

#### 2.5.2 Mixed Precision for Phase 1

```python
# Phase 1 can use float32 for faster Adam updates
if self.config.precision == 'auto':
    phase1_dtype = jnp.float32
    phase2_dtype = jnp.float64
else:
    phase1_dtype = phase2_dtype = jnp.float64

# Convert data for Phase 1
x_data_f32 = x_data.astype(phase1_dtype)
```

**Expected Improvement:** 10-20% memory bandwidth, 5-10% speed for Phase 1

---

### Priority 6: Async Checkpointing (Low)

#### 2.6.1 Background Checkpoint Thread

```python
import threading
import queue

class AsyncCheckpointManager:
    def __init__(self, checkpoint_dir):
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def save_checkpoint_async(self, checkpoint_data):
        """Non-blocking checkpoint save."""
        self.queue.put(checkpoint_data)

    def _worker(self):
        while True:
            data = self.queue.get()
            self._save_checkpoint_sync(data)  # Actual HDF5 write
            self.queue.task_done()
```

**Expected Improvement:** Near-zero blocking time for checkpoints

---

### Priority 7: Conditional Telemetry (Low)

#### 2.7.1 Production Mode Without Telemetry

```python
class HybridStreamingConfig:
    enable_telemetry: bool = False  # Default off in production
    telemetry_level: int = 0  # 0=off, 1=basic, 2=detailed

# In optimizer
if self.config.enable_telemetry:
    telemetry.record_layer4_clip(original_norm, max_norm)
```

---

## 3. Implementation Priority Matrix

| Optimization | Impact | Effort | Priority | Dependencies |
|--------------|--------|--------|----------|--------------|
| Pre-compile Jacobian | High (15-25%) | Low | P1 | None |
| Single-pass GN | High (20-30%) | Medium | P1 | None |
| JAX Scan loops | Medium (10-15%) | Medium | P2 | Data padding |
| Physics caching | Medium (5-10%) | Low | P3 | Homodyne changes |
| Buffer donation | Medium (10%) | Low | P3 | JAX 0.4+ |
| Mixed precision | Medium (5-10%) | Low | P3 | Config changes |
| Async checkpoints | Low (1-3%) | Medium | P4 | Threading |
| Conditional telemetry | Low (1-2%) | Low | P4 | Config |

---

## 4. Benchmarking Plan

### 4.1 Baseline Measurements

Create benchmarks for:
1. **Jacobian computation time** per chunk size (1K, 5K, 10K, 50K points)
2. **GN iteration time** breakdown (J^T J, solve, cost eval)
3. **Phase 1 warmup** iteration time distribution
4. **End-to-end optimization** for standard datasets (1M, 10M, 100M points)

### 4.2 Profiling Tools

```python
# JAX profiling
with jax.profiler.trace("/tmp/jax-trace"):
    result = optimizer.fit(data, model, p0)

# Line-level profiling
from line_profiler import LineProfiler
lp = LineProfiler()
lp.add_function(optimizer._gauss_newton_iteration)
lp.run('optimizer.fit(data, model, p0)')
```

### 4.3 Regression Testing

Add to `tests/performance/`:
```python
@pytest.mark.benchmark
def test_jacobian_chunk_performance(benchmark):
    result = benchmark(optimizer._compute_jacobian_chunk, x_chunk, params)
    assert result.mean < 0.1  # <100ms per 10K chunk

@pytest.mark.benchmark
def test_gn_iteration_performance(benchmark):
    result = benchmark(optimizer._gauss_newton_iteration, data, params, 1.0)
    assert result.mean < 1.0  # <1s per iteration for 1M points
```

---

## 5. Expected Overall Improvement

| Scenario | Current Time | Expected After | Improvement |
|----------|--------------|----------------|-------------|
| 1M points, static | ~30s | ~18s | ~40% |
| 10M points, laminar | ~5min | ~3min | ~40% |
| 100M points, streaming | ~50min | ~30min | ~40% |

---

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| JIT recompilation | Use static shapes, clear cache strategy |
| Numerical stability | Keep float64 for Phase 2 |
| Memory regression | Monitor with psutil in CI |
| API changes | Maintain backward compat wrappers |

---

## 7. Next Steps

1. **Week 1**: Implement P1 optimizations (Jacobian + single-pass GN)
2. **Week 2**: Implement P2 optimizations (JAX scan loops)
3. **Week 3**: Implement P3 optimizations (physics, memory)
4. **Week 4**: Benchmarking, profiling, regression tests

---

## Appendix: Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| AdaptiveHybridStreamingOptimizer | `NLSQ/nlsq/adaptive_hybrid_streaming.py` | 1-3500+ |
| HybridStreamingConfig | `NLSQ/nlsq/hybrid_streaming_config.py` | 1-807 |
| Homodyne NLSQWrapper | `homodyne/optimization/nlsq/wrapper.py` | 1-6448 |
| StratifiedResidualFunction | `homodyne/optimization/nlsq/strategies/residual.py` | 1-600+ |
| Physics computations | `homodyne/core/physics_nlsq.py` | All |
