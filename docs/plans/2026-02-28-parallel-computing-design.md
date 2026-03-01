# Parallel Computing Improvement Design

**Date:** 2026-02-28 **Status:** Proposed **Approach:** Layered Parallelism Architecture
(Approach C)

## Executive Summary

Homodyne's parallel computing infrastructure is mature for CMC shard dispatch
(SharedMemory + LPT scheduling) but leaves significant performance on the table in four
areas: (1) per-shard NUTS chain parallelism, (2) NLSQ streaming chunk accumulation, (3)
CMC worker lifecycle overhead, and (4) pipeline-level I/O overlap. This design proposes
a phased improvement plan that scales from workstation (8 cores) to multi-node HPC
(1000+ cores), with each phase independently shippable.

## Current State

| Component | Mechanism | Bottleneck | |-----------|-----------|-----------| | CMC shard
dispatch | multiprocessing (spawn) + SharedMemory + LPT | Worker lifecycle overhead
(190ms/shard) | | CMC per-shard NUTS | Sequential 4-chain execution in NumPyro | Chains
run serially despite 4 XLA devices | | NLSQ streaming | Sequential chunk accumulation
(JIT per-chunk) | 1000 chunks processed one at a time | | NLSQ multi-start |
ProcessPoolExecutor (already parallel) | Data serialization at 500K+ points | | Data I/O
| Synchronous HDF5 reads, blocking result writes | No overlap with computation | | pjit
backend | Sequential (NOT parallel), misleading name | Dead code |

## Architecture: Four Layers

```
Layer 4: Pipeline Orchestrator (async I/O, stage overlap)
Layer 3: Multi-Node Distribution (optional Ray/Slurm backends)
Layer 2: Worker-Level Parallelism (persistent pools, work dispatch)
Layer 1: Intra-Computation (JAX vmap/pmap, NumPyro chain parallelism)
```

Each layer is independently improvable. Phase 1 addresses Layers 1-2 with zero new
dependencies.

______________________________________________________________________

## Phase 1: Immediate Wins (zero new dependencies)

### 1.1 NumPyro Chain Parallelism

**Problem:** Each CMC worker runs 4 NUTS chains sequentially within `MCMC.run()`,
despite `--xla_force_host_platform_device_count=4` already creating 4 XLA devices per
worker.

**Solution:** Pass `chain_method="parallel"` to the NumPyro MCMC constructor.

**Files changed:**

- `optimization/cmc/config.py`: Add `chain_method: str = "parallel"` field with
  from_dict/validate/to_dict support
- `optimization/cmc/sampler.py:854`: Pass `chain_method=config.chain_method` to `MCMC()`
- `optimization/cmc/backends/multiprocessing.py:548`: Dynamically set
  `xla_force_host_platform_device_count` to `config.num_chains` (currently hardcoded 4)

**Safety analysis:**

- All CMC models are pure JAX functions with no mutable state (verified: 5 model
  variants in model.py)
- Sample extraction at sampler.py:927 uses `group_by_chain=True` which returns identical
  `(n_chains, n_samples, ...)` shape regardless of chain method
- Downstream diagnostics (ESS, R-hat, divergence rate) are chain-method agnostic

**Risks:**

- pmap compilation overhead adds ~5-15% on small shards (\<500 points). Mitigate by
  auto-fallback: if shard_size < 500, log WARNING and fall back to sequential chains
- Device count must match `num_chains`. The dynamic linking in multiprocessing.py:548
  ensures this

**Expected speedup:** 1.5-3.5x per shard (4 chains on 4 devices). Impact is largest on
compute-bound shards (5K+ points).

**Config example:**

```yaml
optimization:
  cmc:
    per_shard_mcmc:
      chain_method: "parallel"  # "parallel" (default) or "sequential"
      num_chains: 4
```

### 1.2 Persistent CMC Worker Pool

**Problem:** Each shard spawns a new process (50ms process + 50ms JAX import + 130ms JIT
= 230ms overhead). For 1000 shards on 14 cores, that's 71 batches of process creation =
~16s pure overhead.

**Solution:** Pre-spawn N persistent workers that process multiple shards via a task
queue.

**Architecture:**

```
Parent Process
  |
  +-- WorkerPool (N persistent workers)
  |     |
  |     +-- Worker 0: event loop (task_queue -> run_nuts -> result_queue)
  |     +-- Worker 1: event loop
  |     +-- ...
  |     +-- Worker N-1: event loop
  |
  +-- Dispatch Loop (LPT-ordered shards -> round-robin to workers)
  +-- Result Collector (drain result_queue, quality filter, combine)
```

**Files changed:**

- New: `optimization/cmc/backends/worker_pool.py` (~300 lines)
  - `WorkerPool` class: spawn/shutdown lifecycle, task dispatch, heartbeat monitoring
  - Worker event loop: persistent process with JAX init done once
- Modified: `optimization/cmc/backends/multiprocessing.py`
  - Replace per-shard `ctx.Process()` dispatch (lines 1298-1340) with
    `WorkerPool.submit()`
  - Keep all existing shared memory, LPT scheduling, quality filtering logic

**Worker lifecycle:**

```python
def _worker_event_loop(task_queue, result_queue, worker_id, config_ref):
    """Persistent worker: init once, process many shards."""
    # One-time JAX initialization
    os.environ["JAX_ENABLE_X64"] = "true"
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    while True:
        task = task_queue.get()  # Block until work arrives
        if task is None:  # Shutdown sentinel
            break
        shard_idx, shard_ref, model, rng_key = task
        try:
            result = run_nuts_sampling(model, shard_ref, config, rng_key)
            result_queue.put({"type": "result", "shard_idx": shard_idx, ...})
        except Exception as e:
            result_queue.put({"type": "error", "shard_idx": shard_idx, ...})
```

**Task queue protocol:**

- Task: `(shard_idx, shared_shard_ref, model_fn, rng_key)` — lightweight (refs only, no
  data copy)
- Result: Same format as current `{"type": "result", "shard_idx": ..., "samples": ...}`
- Shutdown: `None` sentinel per worker

**Default:** ON. Falls back to per-shard process spawn when `n_shards < 3` or pool
initialization fails.

**Expected speedup:** 20-50% wall-time reduction for CMC runs with 50+ shards. Marginal
benefit for \<10 shards.

### 1.3 Parallel NLSQ Streaming Chunk Accumulation

**Problem:** The streaming optimizer (wrapper.py:5814-5835) processes 200-1000 chunks
sequentially. Each chunk computes `(J^T J, J^T r, chi2)` via a JIT kernel taking
~5-15ms. Total: ~3-15s per L-M iteration, 5-20 iterations = 15-300s.

**Solution:** Dispatch chunks to a process pool, accumulate results as they arrive.

**Key insight:** Matrix addition is associative and commutative.
`total_JtJ = sum(chunk_JtJ_i)` is order-independent, so parallel accumulation produces
bitwise-identical results.

**Architecture:**

```
Streaming Optimizer (per L-M iteration)
  |
  +-- SharedMemory: phi_flat, t1_flat, t2_flat, g2_flat, sigma_flat
  |
  +-- ChunkPool (M persistent workers, reused across iterations)
  |     |
  |     +-- Worker 0: compute_chunk_accumulators(chunk_indices, params)
  |     +-- Worker 1: ...
  |     +-- ...
  |
  +-- Reduce: total_JtJ += chunk_JtJ, total_Jtr += chunk_Jtr, total_chi2 += chunk_chi2
```

**Files changed:**

- New: `optimization/nlsq/parallel_accumulator.py` (~200 lines)
  - `ParallelChunkAccumulator` context manager
  - Shared memory setup for flat data arrays (allocated once, used across iterations)
  - Worker function: attach to shared memory, slice by indices, compute JIT kernel
- Modified: `optimization/nlsq/wrapper.py:5814-5835`
  - Replace sequential loop with
    `ParallelChunkAccumulator.map_reduce(params, chunk_iterator)`

**Data flow per chunk:**

- Parent sends: `(chunk_indices, params_current)` — small (~100KB indices + 9-53 float
  params)
- Worker reads: data from shared memory using indices (zero-copy slice)
- Worker returns: `(JtJ, Jtr, chi2)` — tiny (~9x9 matrix + 9-vector + scalar = \<1KB)
- Communication overhead is negligible vs compute

**Chunk statistics (100M points, laminar_flow, 23 angles):**

- Chunk size: ~460K points (adaptive, angle-scaled)
- Chunk count: ~217
- Per-chunk compute: ~10-15ms
- Sequential: 217 x 12ms = 2.6s per iteration
- Parallel (6 workers): 217/6 x 12ms + overhead = ~450ms per iteration
- Speedup: ~5-6x per iteration

**Default:** ON. Falls back to sequential loop when `n_chunks < 10` or shared memory
allocation fails.

**Expected speedup:** 4-8x for streaming optimizer inner loop on 10M+ point datasets.

### 1.4 Background I/O

**Problem:** HDF5 data loading (1-10s for large files) and result serialization (JSON +
NPZ, 0.5-2s) block the computation pipeline.

**Solution:** Use `concurrent.futures.ThreadPoolExecutor` for overlapped I/O (GIL-safe
since HDF5 and numpy release the GIL during I/O).

**Components:**

1. **HDF5 Prefetch:** While optimizing dataset i, load dataset i+1 in a background
   thread
1. **Async Result Write:** Write JSON/NPZ results in a background thread while next
   stage starts
1. **Shard Data Preparation:** Build CMC shared memory arrays while NLSQ computes final
   covariance

**Files changed:**

- New: `utils/async_io.py` (~100 lines)
  - `PrefetchLoader`: Thread-based dataset prefetch with 1-element buffer
  - `AsyncWriter`: Background thread for JSON/NPZ serialization
- Modified: `cli/commands.py` — wrap dataset iteration with PrefetchLoader
- Modified: `optimization/cmc/core.py` — overlap shard data prep with NLSQ finalization

**Expected speedup:** 10-20% pipeline-level improvement by hiding I/O latency.

______________________________________________________________________

## Phase 2: Execution Backend Abstraction

### 2.1 `ExecutionBackend` Protocol

**Problem:** CMC multiprocessing, NLSQ streaming, and NLSQ multi-start each have
independent process pool implementations. Adding Ray/Slurm requires duplicating dispatch
logic.

**Solution:** Define a common `ExecutionBackend` protocol that all parallel dispatch
paths use.

```python
# optimization/parallel/backend.py

from typing import Protocol, TypeVar, Callable, Iterator

T = TypeVar("T")
R = TypeVar("R")

class ExecutionBackend(Protocol):
    """Pluggable parallel execution backend."""

    def submit(self, fn: Callable[..., R], *args, **kwargs) -> Future[R]:
        """Submit a task for execution."""
        ...

    def map(self, fn: Callable[[T], R], items: Iterator[T],
            chunk_size: int = 1) -> Iterator[R]:
        """Map function over items in parallel."""
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the backend, optionally waiting for completion."""
        ...

    @property
    def n_workers(self) -> int:
        """Number of available workers."""
        ...
```

### 2.2 Backend Implementations

**Phase 2 delivers:**

- `LocalPoolBackend`: Wraps persistent multiprocessing pool (from Phase 1)
- `SequentialBackend`: Single-process execution for debugging/testing

**Phase 3 adds:**

- `RayBackend`: Ray actors for cross-node execution
- `SlurmBackend`: Slurm job array submission

### 2.3 Backend Selection

```python
# optimization/parallel/selector.py

def select_backend(config: dict, hardware: HardwareConfig) -> ExecutionBackend:
    """Auto-select optimal backend based on hardware and config."""
    if hardware.cluster_type == "slurm":
        return SlurmBackend(config)
    elif hardware.cluster_type == "pbs":
        return PBSBackend(config)
    elif config.get("backend") == "ray":
        return RayBackend(config)
    else:
        return LocalPoolBackend(config)
```

### 2.4 Retire pjit Backend

The current `backends/pjit.py` is sequential (processes shards in a for loop) and
misleadingly named. Phase 2 deprecates it in favor of `SequentialBackend` for the same
use case.

**Files changed:**

- New: `optimization/parallel/` package
  - `backend.py`: Protocol definition
  - `local_pool.py`: Persistent multiprocessing pool (consolidates Phase 1 pools)
  - `sequential.py`: Single-process fallback
  - `selector.py`: Auto-selection logic
- Modified: `optimization/cmc/backends/multiprocessing.py` — use `LocalPoolBackend`
- Modified: `optimization/nlsq/wrapper.py` — use `LocalPoolBackend` for streaming
- Deprecated: `optimization/cmc/backends/pjit.py`

______________________________________________________________________

## Phase 3: Multi-Node Distribution (optional Ray)

### 3.1 Ray Backend

**Dependency:** `ray >= 2.9` (optional, not required for single-node)

**Architecture:**

```
Ray Head Node
  |
  +-- Ray Actor Pool (N actors across M nodes)
  |     |
  |     +-- Actor 0 (Node A): persistent JAX worker
  |     +-- Actor 1 (Node A): persistent JAX worker
  |     +-- Actor 2 (Node B): persistent JAX worker
  |     +-- ...
  |
  +-- Ray Object Store (distributed SharedMemory replacement)
  |     |
  |     +-- phi_flat, t1_flat, t2_flat, g2_flat (plasma store)
  |
  +-- Ray Dashboard (real-time monitoring)
```

**Benefits over multiprocessing:**

- Cross-node execution (no shared filesystem required for data)
- Object store eliminates data serialization (zero-copy across actors on same node)
- Actor persistence is built-in (no custom WorkerPool needed)
- Dashboard provides real-time shard progress monitoring

**Files changed:**

- New: `optimization/parallel/ray_backend.py` (~300 lines)
- New: `optimization/parallel/ray_actor.py` (~150 lines) — JAX-initialized Ray actor

### 3.2 Slurm Backend Improvement

The existing PBS backend stub (`backends/pbs.py`) is extended with:

- Native Slurm array job support (`sbatch --array=0-N`)
- Checkpoint/resume for long-running jobs
- Shared filesystem result collection

### 3.3 Distributed NLSQ Streaming

With the `ExecutionBackend` abstraction, the parallel chunk accumulator from Phase 1
works across nodes via Ray:

- Data arrays placed in Ray object store (one copy per node, shared across actors)
- Chunk tasks dispatched to actors across nodes
- Reduction happens on the head node

______________________________________________________________________

## Phase 4: Pipeline Orchestrator

### 4.1 Stage-Level Parallelism

**Problem:** For batch processing (multiple datasets), the pipeline runs
`load -> NLSQ -> CMC -> write` sequentially per dataset.

**Solution:** Overlap stages across datasets:

```
Dataset 1:  [load] [NLSQ] [CMC] [write]
Dataset 2:        [load] [NLSQ] [CMC] [write]
Dataset 3:              [load] [NLSQ] [CMC] [write]
```

**Implementation:** Async pipeline with bounded buffer (1-2 datasets ahead).

### 4.2 NLSQ-to-CMC Overlap

Start building CMC shard data structures while NLSQ computes final covariance matrix:

- NLSQ converges -> trigger shard data preparation in background
- NLSQ covariance done -> CMC shard dispatch begins immediately (data already prepared)
- Saves 1-5s transition time

______________________________________________________________________

## Impact Summary

| Phase | Component | Speedup | New Dependencies | Timeline |
|-------|-----------|---------|-----------------|----------| | 1.1 | CMC chain
parallelism | 1.5-3.5x per shard | None | 1 week | | 1.2 | CMC persistent workers |
20-50% overhead reduction | None | 2 weeks | | 1.3 | NLSQ parallel chunks | 4-8x
streaming inner loop | None | 2 weeks | | 1.4 | Background I/O | 10-20% pipeline overlap
| None | 1 week | | 2 | Backend abstraction | Architectural (enables Phase 3) | None | 3
weeks | | 3 | Ray multi-node | Linear node scaling | ray (optional) | 4 weeks | | 4 |
Pipeline orchestrator | 10-30% batch throughput | None | 2 weeks |

**Phase 1 total estimated speedup:** 3-10x end-to-end depending on dataset size and
hardware.

## Testing Strategy

Each phase includes:

1. **Correctness tests:** Bit-exact comparison of parallel vs sequential results (using
   `np.allclose` with `rtol=1e-14`)
1. **Scaling tests:** Measure speedup vs core count on representative datasets (1M, 10M,
   100M points)
1. **Memory tests:** Verify peak RSS stays within bounds (persistent workers add ~150MB
   each)
1. **Regression tests:** Full test suite (2718 tests) must pass after each phase
1. **Integration tests:** End-to-end pipeline on real XPCS datasets (static +
   laminar_flow)

## Default Behavior

- All improvements default to ON — parallel execution is the standard path
- Sequential is the automatic fallback when parallel fails (e.g., insufficient cores,
  shared memory allocation failure, import error for optional deps)
- Config-driven override: `chain_method: "sequential"`, `parallel_chunks: false`,
  `backend: "sequential"` to force serial execution
- No API changes to `fit_nlsq_jax()` or `fit_mcmc_jax()` public interfaces
- pjit backend deprecated with warning, not removed until Phase 3

**Fallback triggers (automatic, logged at WARNING level):**

- `chain_method="parallel"` -> sequential if `num_chains > available_xla_devices` or
  pmap compilation fails
- Persistent worker pool -> per-shard process spawn if pool initialization fails or
  `n_shards < 3`
- Parallel chunk accumulation -> sequential loop if `n_chunks < 10` or shared memory
  allocation fails
- Ray backend -> LocalPoolBackend if `import ray` fails
- Background I/O -> synchronous I/O if thread pool creation fails
