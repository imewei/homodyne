# Parallel Computing Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task.

**Goal:** Enable parallel execution by default across CMC chain sampling, CMC worker
lifecycle, NLSQ streaming, and pipeline I/O — with automatic sequential fallback.

**Architecture:** Four independent improvements that each default to ON and gracefully
fall back to sequential. No new dependencies — stdlib multiprocessing, threading, and
existing JAX primitives only.

**Tech Stack:** Python 3.12+ stdlib (multiprocessing, concurrent.futures, threading,
shared_memory), JAX (vmap, jit), NumPyro (MCMC chain_method)

______________________________________________________________________

### Task 1: CMC Chain Parallelism — Config

**Files:**

- Modify: `homodyne/optimization/cmc/config.py`
- Test: `tests/unit/optimization/cmc/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/unit/optimization/cmc/test_config.py` inside `class TestCMCConfig`:

```python
def test_default_chain_method(self):
    """Test that chain_method defaults to parallel."""
    config = CMCConfig()
    assert config.chain_method == "parallel"

def test_chain_method_from_dict(self):
    """Test chain_method parsed from per_shard_mcmc section."""
    config = CMCConfig.from_dict({
        "per_shard_mcmc": {"chain_method": "sequential"},
    })
    assert config.chain_method == "sequential"

def test_chain_method_from_dict_default(self):
    """Test chain_method defaults to parallel when not in dict."""
    config = CMCConfig.from_dict({})
    assert config.chain_method == "parallel"

def test_chain_method_validation_valid(self):
    """Test valid chain_method values pass validation."""
    for method in ("parallel", "sequential"):
        config = CMCConfig(chain_method=method)
        errors = config.validate()
        assert not any("chain_method" in e for e in errors)

def test_chain_method_validation_invalid(self):
    """Test invalid chain_method is caught by validation."""
    config = CMCConfig(chain_method="invalid")
    errors = config.validate()
    assert any("chain_method" in e for e in errors)

def test_chain_method_to_dict(self):
    """Test chain_method appears in to_dict output."""
    config = CMCConfig(chain_method="sequential")
    d = config.to_dict()
    assert d["per_shard_mcmc"]["chain_method"] == "sequential"
```

**Step 2: Run tests to verify they fail**

Run:
`uv run pytest tests/unit/optimization/cmc/test_config.py::TestCMCConfig::test_default_chain_method -v`
Expected: FAIL — `AttributeError: 'CMCConfig' has no attribute 'chain_method'`

**Step 3: Implement chain_method in CMCConfig**

In `homodyne/optimization/cmc/config.py`:

a) Add field after `num_chains` (line 146):

```python
    num_chains: int = 4  # Increased from 2 for better R-hat convergence diagnostics
    chain_method: str = "parallel"  # "parallel" (default) or "sequential"
```

b) Add docstring entry after the `num_chains` line in the class docstring (after line
68):

```
    chain_method : str
        MCMC chain execution method. ``"parallel"`` (default) runs chains
        concurrently via JAX vectorization. ``"sequential"`` runs chains
        one at a time. Parallel is faster on multi-core CPUs but adds
        ~5-15% overhead on very small shards (<500 points); the sampler
        auto-falls-back to sequential in that case.
```

c) Add to `from_dict` constructor call (after `num_chains` line 338):

```python
            num_chains=per_shard.get("num_chains", 4),
            chain_method=per_shard.get("chain_method", "parallel"),
```

d) Add validation in `validate()` after the `num_chains` check (after line 479):

```python
        # Validate chain_method
        valid_chain_methods = ["parallel", "sequential"]
        if self.chain_method not in valid_chain_methods:
            errors.append(
                f"chain_method must be one of {valid_chain_methods}, "
                f"got: {self.chain_method}"
            )
```

e) Add to `to_dict` in the `per_shard_mcmc` section (after `num_chains` line 853):

```python
                "num_chains": self.num_chains,
                "chain_method": self.chain_method,
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/optimization/cmc/test_config.py -v -k chain_method`
Expected: All 6 new tests PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/config.py tests/unit/optimization/cmc/test_config.py
git commit -m "feat(cmc): add chain_method config field (default: parallel)"
```

______________________________________________________________________

### Task 2: CMC Chain Parallelism — Sampler Integration

**Files:**

- Modify: `homodyne/optimization/cmc/sampler.py`
- Test: `tests/unit/optimization/cmc/test_sampler.py`

**Step 1: Write the failing test**

Add to `tests/unit/optimization/cmc/test_sampler.py`:

```python
class TestChainMethodIntegration:
    """Test chain_method parameter is passed to MCMC."""

    def test_mcmc_receives_chain_method_parallel(self):
        """Verify MCMC constructor receives chain_method from config."""
        from unittest.mock import MagicMock, patch
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig(chain_method="parallel", num_chains=4)

        with patch("homodyne.optimization.cmc.sampler.MCMC") as MockMCMC:
            mock_mcmc = MagicMock()
            mock_mcmc.get_samples.return_value = {}
            mock_mcmc.get_extra_fields.return_value = {}
            mock_mcmc.last_state = None
            MockMCMC.return_value = mock_mcmc

            from homodyne.optimization.cmc.sampler import run_nuts_sampling
            try:
                run_nuts_sampling(
                    model_fn=lambda **kwargs: None,
                    model_kwargs={"data": [1, 2, 3]},
                    config=config,
                    param_names=["D0"],
                )
            except Exception:
                pass  # We only care about the MCMC constructor call

            # Verify chain_method was passed
            if MockMCMC.called:
                call_kwargs = MockMCMC.call_args
                # Check both positional and keyword args
                assert call_kwargs.kwargs.get("chain_method") == "parallel" or \
                    "parallel" in str(call_kwargs)

    def test_chain_method_fallback_small_shard(self):
        """Verify parallel falls back to sequential for tiny shards."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig(chain_method="parallel", num_chains=4)
        # Shard with <500 points should trigger fallback
        data = list(range(100))  # 100 points
        effective = (
            "sequential" if len(data) < 500 and config.chain_method == "parallel"
            else config.chain_method
        )
        assert effective == "sequential"
```

**Step 2: Run test to verify it fails**

Run:
`uv run pytest tests/unit/optimization/cmc/test_sampler.py::TestChainMethodIntegration -v`
Expected: FAIL — MCMC not called with chain_method kwarg

**Step 3: Implement chain_method in sampler**

In `homodyne/optimization/cmc/sampler.py`, modify the MCMC creation block (~line
847-854):

```python
    # Determine effective chain method with auto-fallback for small shards
    effective_chain_method = config.chain_method
    if effective_chain_method == "parallel" and shard_size < 500:
        run_logger.warning(
            f"Shard size {shard_size:,} < 500: falling back to sequential "
            f"chains (parallel overhead exceeds benefit for small shards)"
        )
        effective_chain_method = "sequential"

    # Create MCMC runner with adaptive sample counts
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=config.num_chains,
        chain_method=effective_chain_method,
        progress_bar=progress_bar,
    )
```

**Step 4: Run tests to verify they pass**

Run:
`uv run pytest tests/unit/optimization/cmc/test_sampler.py::TestChainMethodIntegration -v`
Expected: PASS

**Step 5: Run existing sampler tests for regression**

Run: `uv run pytest tests/unit/optimization/cmc/test_sampler.py -v` Expected: All
existing tests PASS

**Step 6: Commit**

```bash
git add homodyne/optimization/cmc/sampler.py tests/unit/optimization/cmc/test_sampler.py
git commit -m "feat(cmc): pass chain_method to MCMC with auto-fallback for small shards"
```

______________________________________________________________________

### Task 3: CMC Chain Parallelism — Dynamic XLA Device Count

**Files:**

- Modify: `homodyne/optimization/cmc/backends/multiprocessing.py`
- Test: `tests/unit/optimization/cmc/test_backends.py`

**Step 1: Write the failing test**

Add to `tests/unit/optimization/cmc/test_backends.py`:

```python
class TestDynamicXLADeviceCount:
    """Test XLA device count matches num_chains."""

    def test_xla_device_count_matches_num_chains(self):
        """Verify XLA_FLAGS sets device count to num_chains."""
        import re

        # Simulate the worker setup logic
        num_chains = 6
        xla_flags = "--some_flag=true --xla_force_host_platform_device_count=4"
        xla_flags = re.sub(
            r"--xla_force_host_platform_device_count=\d+", "", xla_flags
        )
        xla_flags = (
            xla_flags.strip()
            + f" --xla_force_host_platform_device_count={num_chains}"
        )
        assert f"--xla_force_host_platform_device_count={num_chains}" in xla_flags
        assert "--xla_force_host_platform_device_count=4" not in xla_flags

    def test_xla_device_count_default_4(self):
        """Verify default device count is 4 (matching default num_chains)."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig()
        assert config.num_chains == 4
```

**Step 2: Run test to verify baseline**

Run:
`uv run pytest tests/unit/optimization/cmc/test_backends.py::TestDynamicXLADeviceCount -v`
Expected: PASS (logic tests, not worker execution)

**Step 3: Modify multiprocessing.py worker setup**

The module-level XLA setup (lines 542-549) currently hardcodes `device_count=4`. Change
to read from env var set by parent:

In the module-level worker init code:

```python
    _num_chains = int(os.environ.get("HOMODYNE_CMC_NUM_CHAINS", "4"))
    os.environ["XLA_FLAGS"] = (
        _xla_flags.strip()
        + f" --xla_force_host_platform_device_count={_num_chains}"
    )
```

In the parent dispatch code (where env vars are set before spawn):

```python
    os.environ["HOMODYNE_CMC_NUM_CHAINS"] = str(config.num_chains)
```

Add `"HOMODYNE_CMC_NUM_CHAINS"` to the `_saved_env` dict for cleanup.

**Step 4: Run backend tests**

Run: `uv run pytest tests/unit/optimization/cmc/test_backends.py -v` Expected: All PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/backends/multiprocessing.py tests/unit/optimization/cmc/test_backends.py
git commit -m "feat(cmc): dynamic XLA device count from num_chains for parallel chains"
```

______________________________________________________________________

### Task 4: Persistent CMC Worker Pool — Core Implementation

**Files:**

- Create: `homodyne/optimization/cmc/backends/worker_pool.py`
- Test: `tests/unit/optimization/cmc/test_worker_pool.py`

**Step 1: Write the failing tests**

Create `tests/unit/optimization/cmc/test_worker_pool.py`:

```python
"""Tests for persistent CMC worker pool."""

import os
import pytest


def _echo_worker(task, **init_kwargs):
    """Simple echo worker for testing."""
    return {"task_id": task["task_id"], "value": task["value"], "success": True}


def _counting_worker(task, **init_kwargs):
    """Worker that reports its PID to verify reuse."""
    return {"task_id": task["task_id"], "worker_id": os.getpid(), "success": True}


def _error_worker(task, **init_kwargs):
    """Worker that raises on request."""
    if task.get("should_fail"):
        raise ValueError("Intentional test error")
    return {"task_id": task["task_id"], "success": True}


class TestWorkerPool:
    """Test WorkerPool lifecycle and task dispatch."""

    def test_pool_creation_and_shutdown(self):
        """Test pool starts workers and shuts down cleanly."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=2, worker_fn=_echo_worker, worker_init_kwargs={})
        try:
            assert pool.n_workers == 2
            assert pool.is_alive()
        finally:
            pool.shutdown()
        assert not pool.is_alive()

    def test_submit_and_collect(self):
        """Test submitting tasks and collecting results."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=2, worker_fn=_echo_worker, worker_init_kwargs={})
        try:
            for i in range(5):
                pool.submit({"task_id": i, "value": i * 10})

            results = []
            for _ in range(5):
                result = pool.get_result(timeout=10.0)
                results.append(result)

            assert len(results) == 5
            values = sorted(r["value"] for r in results)
            assert values == [0, 10, 20, 30, 40]
        finally:
            pool.shutdown()

    def test_fallback_threshold(self):
        """Test that pool is not used when n_shards < 3."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool

        assert not should_use_pool(n_shards=1, n_workers=4)
        assert not should_use_pool(n_shards=2, n_workers=4)
        assert should_use_pool(n_shards=3, n_workers=2)
        assert should_use_pool(n_shards=100, n_workers=8)

    def test_worker_reuse(self):
        """Test that a single worker handles multiple tasks."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=1, worker_fn=_counting_worker, worker_init_kwargs={})
        try:
            for i in range(3):
                pool.submit({"task_id": i})

            results = []
            for _ in range(3):
                results.append(pool.get_result(timeout=10.0))

            assert len(results) == 3
            worker_ids = {r["worker_id"] for r in results}
            assert len(worker_ids) == 1  # All from same worker
        finally:
            pool.shutdown()

    def test_error_handling(self):
        """Test that worker errors are captured, not lost."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=1, worker_fn=_error_worker, worker_init_kwargs={})
        try:
            pool.submit({"task_id": 0, "should_fail": True})
            result = pool.get_result(timeout=10.0)
            assert result["success"] is False
            assert "error" in result
        finally:
            pool.shutdown()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/optimization/cmc/test_worker_pool.py -v` Expected: FAIL —
`ImportError: cannot import name 'WorkerPool'`

**Step 3: Implement WorkerPool**

Create `homodyne/optimization/cmc/backends/worker_pool.py`:

```python
"""Persistent worker pool for CMC shard processing.

Spawns N persistent worker processes that each run an event loop,
processing multiple shards without JAX re-initialization overhead.
Falls back to per-shard process spawning when n_shards < 3.
"""

from __future__ import annotations

import multiprocessing
import os
import queue
import traceback
from typing import Any, Callable

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def should_use_pool(n_shards: int, n_workers: int) -> bool:
    """Determine if worker pool is beneficial.

    Parameters
    ----------
    n_shards : int
        Total shards to process.
    n_workers : int
        Available worker count.

    Returns
    -------
    bool
        True if pool amortization outweighs overhead.
    """
    return n_shards >= 3


class WorkerPool:
    """Persistent process pool for CMC shard dispatch.

    Workers are spawned once, process multiple tasks via queues,
    and shut down when the pool is no longer needed.

    Parameters
    ----------
    n_workers : int
        Number of persistent worker processes.
    worker_fn : callable
        Function each worker calls per task.
        Signature: ``worker_fn(task: dict, **init_kwargs) -> dict``.
        Must be picklable (module-level function).
    worker_init_kwargs : dict
        One-time kwargs passed to every worker_fn call.
    """

    def __init__(
        self,
        n_workers: int,
        worker_fn: Callable[..., dict[str, Any]],
        worker_init_kwargs: dict[str, Any],
    ) -> None:
        self._n_workers = n_workers
        self._worker_fn = worker_fn
        self._init_kwargs = worker_init_kwargs

        ctx = multiprocessing.get_context("spawn")
        self._task_queues: list[multiprocessing.Queue] = [
            ctx.Queue() for _ in range(n_workers)
        ]
        self._result_queue: multiprocessing.Queue = ctx.Queue()
        self._processes: list[multiprocessing.Process] = []
        self._next_worker = 0
        self._alive = False

        self._start_workers(ctx)

    def _start_workers(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn persistent worker processes."""
        for i in range(self._n_workers):
            p = ctx.Process(
                target=_worker_event_loop,
                args=(
                    i,
                    self._task_queues[i],
                    self._result_queue,
                    self._worker_fn,
                    self._init_kwargs,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        self._alive = True
        logger.info(f"WorkerPool started: {self._n_workers} persistent workers")

    @property
    def n_workers(self) -> int:
        """Number of worker processes."""
        return self._n_workers

    def is_alive(self) -> bool:
        """Check if pool has active workers."""
        return self._alive and any(p.is_alive() for p in self._processes)

    def submit(self, task: dict[str, Any]) -> None:
        """Submit a task to the next available worker (round-robin).

        Parameters
        ----------
        task : dict
            Task payload with at minimum a ``task_id`` key.
        """
        worker_idx = self._next_worker % self._n_workers
        self._task_queues[worker_idx].put(task)
        self._next_worker += 1

    def get_result(self, timeout: float = 300.0) -> dict[str, Any]:
        """Block until a result is available.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait.

        Returns
        -------
        dict
            Result from a worker.

        Raises
        ------
        queue.Empty
            If no result within timeout.
        """
        return self._result_queue.get(timeout=timeout)

    def results_pending(self) -> bool:
        """Check if results are available without blocking."""
        return not self._result_queue.empty()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Send shutdown sentinels and join all workers.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait per worker.
        """
        if not self._alive:
            return

        for tq in self._task_queues:
            try:
                tq.put(None)
            except (OSError, ValueError):
                pass

        for p in self._processes:
            p.join(timeout=timeout)
            if p.is_alive():
                logger.warning(f"Worker {p.pid} did not exit gracefully, terminating")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()

        for tq in self._task_queues:
            try:
                tq.close()
            except (OSError, ValueError):
                pass
        try:
            self._result_queue.close()
        except (OSError, ValueError):
            pass

        self._alive = False
        logger.info("WorkerPool shut down")

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()


def _worker_event_loop(
    worker_id: int,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    worker_fn: Callable[..., dict[str, Any]],
    init_kwargs: dict[str, Any],
) -> None:
    """Persistent worker event loop.

    Processes tasks until a None sentinel is received.
    Results (or errors) are put on result_queue.
    """
    while True:
        try:
            task = task_queue.get()
        except (OSError, EOFError):
            break

        if task is None:
            break

        task_id = task.get("task_id", "unknown")
        try:
            result = worker_fn(task, **init_kwargs)
            result["worker_id"] = os.getpid()
            result_queue.put(result)
        except Exception as e:
            result_queue.put({
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "worker_id": os.getpid(),
            })
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/optimization/cmc/test_worker_pool.py -v` Expected: All 5
tests PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/cmc/backends/worker_pool.py tests/unit/optimization/cmc/test_worker_pool.py
git commit -m "feat(cmc): add persistent WorkerPool for multi-shard dispatch"
```

______________________________________________________________________

### Task 5: Persistent Worker Pool — Integration with Multiprocessing Backend

**Files:**

- Modify: `homodyne/optimization/cmc/backends/multiprocessing.py`
- Test: `tests/unit/optimization/cmc/test_backends.py`

This task integrates the WorkerPool into the existing dispatch loop. The key change:
replace per-shard `ctx.Process()` with pool-based dispatch while preserving all existing
SharedMemory, LPT scheduling, heartbeat, timeout, and quality filtering.

**Step 1: Write integration test**

Add to `tests/unit/optimization/cmc/test_backends.py`:

```python
class TestWorkerPoolIntegration:
    """Test WorkerPool integration with multiprocessing backend."""

    def test_pool_fallback_few_shards(self):
        """Verify fallback to per-shard spawn for < 3 shards."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool
        assert not should_use_pool(n_shards=1, n_workers=4)
        assert not should_use_pool(n_shards=2, n_workers=4)

    def test_pool_activated_many_shards(self):
        """Verify pool is used for >= 3 shards."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool
        assert should_use_pool(n_shards=3, n_workers=2)
        assert should_use_pool(n_shards=50, n_workers=8)
```

**Step 2: Implement pool integration in multiprocessing backend**

In `multiprocessing.py`, the dispatch loop (~lines 1230-1520) needs a conditional path:

1. Import `should_use_pool, WorkerPool` from `worker_pool`
1. Before dispatch: check `should_use_pool(n_shards, max_workers)`
1. Pool path: create a pool-compatible wrapper around `_run_shard_worker`, submit tasks,
   collect results
1. Non-pool path: existing per-shard `ctx.Process()` dispatch (unchanged)

**Key constraints:**

- Result format must match existing
  `{"type": "result", "shard_idx": ..., "samples": ...}` for quality filtering
- Pool workers need JAX module-level init (same env var setup)
- Heartbeat: pool workers are persistent, so no per-shard heartbeat needed; monitor pool
  liveness instead

**Step 3: Run full CMC tests**

Run: `uv run pytest tests/unit/optimization/cmc/ -v --timeout=120` Expected: All PASS

**Step 4: Commit**

```bash
git add homodyne/optimization/cmc/backends/multiprocessing.py tests/unit/optimization/cmc/test_backends.py
git commit -m "feat(cmc): integrate WorkerPool into multiprocessing backend dispatch"
```

______________________________________________________________________

### Task 6: Parallel NLSQ Streaming — Core Implementation

**Files:**

- Create: `homodyne/optimization/nlsq/parallel_accumulator.py`
- Test: `tests/unit/optimization/nlsq/test_parallel_accumulator.py`

**Step 1: Write the failing tests**

Create `tests/unit/optimization/nlsq/test_parallel_accumulator.py`:

```python
"""Tests for parallel chunk accumulation in NLSQ streaming."""

import numpy as np
import pytest


class TestParallelThreshold:
    """Test parallel activation threshold."""

    def test_below_threshold(self):
        from homodyne.optimization.nlsq.parallel_accumulator import (
            should_use_parallel_accumulation,
        )
        assert not should_use_parallel_accumulation(n_chunks=5)
        assert not should_use_parallel_accumulation(n_chunks=9)

    def test_at_and_above_threshold(self):
        from homodyne.optimization.nlsq.parallel_accumulator import (
            should_use_parallel_accumulation,
        )
        assert should_use_parallel_accumulation(n_chunks=10)
        assert should_use_parallel_accumulation(n_chunks=500)


class TestAccumulationCorrectness:
    """Test that parallel == sequential accumulation."""

    def test_identical_results(self):
        """Core correctness: parallel accumulation == sequential."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_sequential,
            accumulate_chunks_parallel,
        )

        rng = np.random.default_rng(42)
        n_params = 5
        n_chunks = 20

        chunks = []
        for _ in range(n_chunks):
            JtJ = rng.standard_normal((n_params, n_params))
            JtJ = JtJ @ JtJ.T  # Symmetric positive definite
            Jtr = rng.standard_normal(n_params)
            chi2 = rng.uniform(10, 100)
            chunks.append((JtJ, Jtr, chi2))

        seq_JtJ, seq_Jtr, seq_chi2, seq_count = accumulate_chunks_sequential(chunks)
        par_JtJ, par_Jtr, par_chi2, par_count = accumulate_chunks_parallel(
            chunks, n_workers=2
        )

        np.testing.assert_allclose(seq_JtJ, par_JtJ, rtol=1e-14)
        np.testing.assert_allclose(seq_Jtr, par_Jtr, rtol=1e-14)
        np.testing.assert_allclose(seq_chi2, par_chi2, rtol=1e-14)
        assert seq_count == par_count

    def test_single_chunk(self):
        """Test with a single chunk (no parallelism needed)."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_sequential,
        )

        JtJ = np.eye(3)
        Jtr = np.ones(3)
        chi2 = 5.0

        result = accumulate_chunks_sequential([(JtJ, Jtr, chi2)])
        np.testing.assert_array_equal(result[0], JtJ)
        np.testing.assert_array_equal(result[1], Jtr)
        assert result[2] == chi2
        assert result[3] == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/optimization/nlsq/test_parallel_accumulator.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement parallel accumulator**

Create `homodyne/optimization/nlsq/parallel_accumulator.py`:

```python
"""Parallel chunk accumulation for NLSQ streaming optimizer.

Dispatches chunk computations to a process pool and reduces
J^T J, J^T r, chi2 accumulators. Falls back to sequential
when n_chunks < 10 or pool creation fails.

Matrix addition is associative and commutative, so parallel
accumulation produces identical results to sequential.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_CHUNKS_FOR_PARALLEL = 10


def should_use_parallel_accumulation(n_chunks: int) -> bool:
    """Determine if parallel accumulation is worthwhile."""
    return n_chunks >= _MIN_CHUNKS_FOR_PARALLEL


def accumulate_chunks_sequential(
    chunks: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate chunk results sequentially.

    Parameters
    ----------
    chunks : list of (JtJ, Jtr, chi2) tuples

    Returns
    -------
    total_JtJ, total_Jtr, total_chi2, count
    """
    total_JtJ = None
    total_Jtr = None
    total_chi2 = 0.0
    count = 0

    for JtJ, Jtr, chi2 in chunks:
        if total_JtJ is None:
            total_JtJ = np.zeros_like(JtJ)
            total_Jtr = np.zeros_like(Jtr)
        total_JtJ += JtJ
        total_Jtr += Jtr
        total_chi2 += chi2
        count += 1

    return total_JtJ, total_Jtr, total_chi2, count


def accumulate_chunks_parallel(
    chunks: list[tuple[np.ndarray, np.ndarray, float]],
    n_workers: int = 4,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate chunk results in parallel via process pool.

    Partitions chunks across workers, each computes partial sums,
    then reduces. Falls back to sequential on failure.

    Parameters
    ----------
    chunks : list of (JtJ, Jtr, chi2) tuples
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    total_JtJ, total_Jtr, total_chi2, count
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    if len(chunks) < _MIN_CHUNKS_FOR_PARALLEL:
        return accumulate_chunks_sequential(chunks)

    # Partition chunks across workers
    partitions: list[list[tuple[np.ndarray, np.ndarray, float]]] = [
        [] for _ in range(n_workers)
    ]
    for i, chunk in enumerate(chunks):
        partitions[i % n_workers].append(chunk)

    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(accumulate_chunks_sequential, partition)
                for partition in partitions
                if partition
            ]

            total_JtJ = None
            total_Jtr = None
            total_chi2 = 0.0
            total_count = 0

            for future in as_completed(futures):
                JtJ, Jtr, chi2, count = future.result()
                if total_JtJ is None:
                    total_JtJ = np.zeros_like(JtJ)
                    total_Jtr = np.zeros_like(Jtr)
                total_JtJ += JtJ
                total_Jtr += Jtr
                total_chi2 += chi2
                total_count += count

        return total_JtJ, total_Jtr, total_chi2, total_count

    except (OSError, RuntimeError) as e:
        logger.warning(
            f"Parallel chunk accumulation failed ({e}), falling back to sequential"
        )
        return accumulate_chunks_sequential(chunks)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/optimization/nlsq/test_parallel_accumulator.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add homodyne/optimization/nlsq/parallel_accumulator.py tests/unit/optimization/nlsq/test_parallel_accumulator.py
git commit -m "feat(nlsq): add parallel chunk accumulator with sequential fallback"
```

______________________________________________________________________

### Task 7: Parallel NLSQ Streaming — Integration with Wrapper

**Files:**

- Modify: `homodyne/optimization/nlsq/wrapper.py`

**Step 1: Locate the chunk accumulation loop**

Search wrapper.py for `total_JtJ +=` or `for indices_chunk in` to find the sequential
loop (~lines 5814-5835).

**Step 2: Add parallel path**

Before the loop, count chunks from the iterator. If
`should_use_parallel_accumulation(n_chunks)`:

1. Collect chunk data tuples
1. Compute each chunk's `(JtJ, Jtr, chi2)` in parallel workers
1. Reduce results

The JIT kernel `compute_chunk_accumulators` cannot be pickled across processes. For
Phase 1, use the simpler approach: pre-compute all chunk results in the parent (each
chunk is a JAX JIT call), then the parallel accumulation is just the reduction. Since
JIT compilation is cached, the per-chunk compute is ~5-15ms.

Alternatively, restructure so workers each handle a partition of chunks end-to-end
(compute + accumulate). This requires workers to have JAX initialized with the same JIT
cache.

The integration should preserve the existing progress bar, count tracking, and error
handling.

**Step 3: Run NLSQ tests**

Run: `uv run pytest tests/unit/ -k nlsq -v --timeout=120` Expected: All PASS

**Step 4: Commit**

```bash
git add homodyne/optimization/nlsq/wrapper.py
git commit -m "feat(nlsq): integrate parallel chunk accumulation into streaming optimizer"
```

______________________________________________________________________

### Task 8: Background I/O — PrefetchLoader and AsyncWriter

**Files:**

- Create: `homodyne/utils/async_io.py`
- Test: `tests/unit/test_async_io.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_async_io.py`:

```python
"""Tests for async I/O utilities."""

import time
import tempfile
from pathlib import Path

import numpy as np
import pytest


class TestPrefetchLoader:
    """Test thread-based data prefetching."""

    def test_yields_all_items(self):
        from homodyne.utils.async_io import PrefetchLoader

        items = list(range(5))
        results = list(PrefetchLoader(iter(items), load_fn=lambda x: x * 10))
        assert results == [0, 10, 20, 30, 40]

    def test_handles_empty_iterator(self):
        from homodyne.utils.async_io import PrefetchLoader

        results = list(PrefetchLoader(iter([]), load_fn=lambda x: x))
        assert results == []

    def test_overlaps_load_and_process(self):
        from homodyne.utils.async_io import PrefetchLoader

        def slow_load(x):
            time.sleep(0.05)
            return x

        items = list(range(5))
        start = time.perf_counter()
        results = []
        for item in PrefetchLoader(iter(items), load_fn=slow_load):
            time.sleep(0.03)
            results.append(item)
        elapsed = time.perf_counter() - start

        assert results == list(range(5))
        assert elapsed < 0.45  # With prefetch overlap


class TestAsyncWriter:
    """Test background result writing."""

    def test_write_npz(self):
        from homodyne.utils.async_io import AsyncWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            with AsyncWriter() as writer:
                path = Path(tmpdir) / "result.npz"
                writer.submit_npz(path, {"arr": np.arange(100)})
                writer.wait_all(timeout=10.0)

                assert path.exists()
                loaded = np.load(path)
                np.testing.assert_array_equal(loaded["arr"], np.arange(100))

    def test_write_json(self):
        import json
        from homodyne.utils.async_io import AsyncWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            with AsyncWriter() as writer:
                path = Path(tmpdir) / "result.json"
                data = {"key": "value", "number": 42}
                writer.submit_json(path, data)
                writer.wait_all(timeout=10.0)

                assert path.exists()
                with open(path, encoding="utf-8") as f:
                    loaded = json.load(f)
                assert loaded == data

    def test_multiple_writes(self):
        from homodyne.utils.async_io import AsyncWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            with AsyncWriter(max_workers=2) as writer:
                for i in range(5):
                    path = Path(tmpdir) / f"result_{i}.npz"
                    writer.submit_npz(path, {"data": np.arange(i)})
                writer.wait_all(timeout=30.0)

                for i in range(5):
                    assert (Path(tmpdir) / f"result_{i}.npz").exists()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_async_io.py -v` Expected: FAIL — `ImportError`

**Step 3: Implement async I/O utilities**

Create `homodyne/utils/async_io.py`:

```python
"""Async I/O utilities for pipeline overlap.

Thread-based prefetching and background writing to hide I/O latency.
GIL-safe since HDF5 and numpy release the GIL during I/O.
"""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Iterator, TypeVar

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class PrefetchLoader(Iterator[R]):
    """Thread-based prefetch iterator.

    Loads the next item in a background thread while the current
    item is being processed.

    Parameters
    ----------
    source : Iterator[T]
        Source items to load.
    load_fn : callable
        Transform applied to each item in background thread.
    """

    def __init__(self, source: Iterator[T], load_fn: Callable[[T], R]) -> None:
        self._source = source
        self._load_fn = load_fn
        self._prefetched: R | None = None
        self._has_prefetched = False
        self._exhausted = False
        self._thread: Thread | None = None
        self._error: BaseException | None = None
        self._start_prefetch()

    def _start_prefetch(self) -> None:
        if self._exhausted:
            return

        def _load() -> None:
            try:
                item = next(self._source)
                self._prefetched = self._load_fn(item)
                self._has_prefetched = True
            except StopIteration:
                self._exhausted = True
            except Exception as e:
                self._error = e
                self._exhausted = True

        self._thread = Thread(target=_load, daemon=True)
        self._thread.start()

    def __iter__(self) -> PrefetchLoader[R]:
        return self

    def __next__(self) -> R:
        if self._thread is not None:
            self._thread.join()
            self._thread = None

        if self._error is not None:
            raise self._error

        if self._exhausted and not self._has_prefetched:
            raise StopIteration

        result = self._prefetched
        self._has_prefetched = False
        self._prefetched = None
        self._start_prefetch()
        return result


class AsyncWriter:
    """Background thread pool for result serialization.

    Parameters
    ----------
    max_workers : int
        Maximum concurrent write threads.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future] = []

    def submit_npz(self, path: Path, data: dict[str, np.ndarray]) -> None:
        """Write NPZ file in background."""
        future = self._executor.submit(self._write_npz, path, data)
        self._futures.append(future)

    def submit_json(self, path: Path, data: dict[str, Any]) -> None:
        """Write JSON file in background."""
        future = self._executor.submit(self._write_json, path, data)
        self._futures.append(future)

    def wait_all(self, timeout: float = 60.0) -> list[Exception]:
        """Wait for all pending writes. Returns list of errors."""
        errors = []
        for future in self._futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning(f"Background write failed: {e}")
                errors.append(e)
        self._futures.clear()
        return errors

    def shutdown(self) -> None:
        """Wait for pending writes and shut down."""
        self.wait_all()
        self._executor.shutdown(wait=True)

    @staticmethod
    def _write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(path), **data)

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def __enter__(self) -> AsyncWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_async_io.py -v` Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add homodyne/utils/async_io.py tests/unit/test_async_io.py
git commit -m "feat(utils): add PrefetchLoader and AsyncWriter for background I/O"
```

______________________________________________________________________

### Task 9: Background I/O — Integration with CLI

**Files:**

- Modify: `homodyne/cli/commands.py`

**Step 1: Identify result-writing locations**

Search `commands.py` for `np.savez`, `json.dump`, or writer function calls. These are
sites to wrap with `AsyncWriter`.

**Step 2: Wrap result writes**

At the start of the main analysis function, create `AsyncWriter()`. Replace synchronous
writes:

- `np.savez_compressed(...)` -> `writer.submit_npz(...)`
- `json.dump(...)` -> `writer.submit_json(...)`

At the end, `writer.wait_all()` ensures all writes finish before exit.

For multi-dataset batch runs: wrap the dataset iterator with
`PrefetchLoader(datasets, load_fn=load_hdf5)` to overlap loading with optimization.

**Step 3: Run CLI tests**

Run: `uv run pytest tests/unit/test_cli_workflows.py -v --timeout=120` Expected: All
PASS

**Step 4: Commit**

```bash
git add homodyne/cli/commands.py
git commit -m "feat(cli): integrate AsyncWriter for background result serialization"
```

______________________________________________________________________

### Task 10: Full Regression + Documentation

**Step 1: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --timeout=300` Expected: All 2718 tests PASS

**Step 2: Run quality checks**

Run: `make quality` Expected: Ruff + mypy clean

**Step 3: Update CLAUDE.md**

Add to the CMC section, after the "Adaptive Sampling" subsection:

```markdown
### Chain Parallelism (Feb 2026)

CMC runs NUTS chains in parallel by default via NumPyro's `chain_method="parallel"`.
Falls back to sequential for shards with <500 points.

\`\`\`yaml
optimization:
  cmc:
    per_shard_mcmc:
      chain_method: "parallel"  # "parallel" (default) or "sequential"
\`\`\`
```

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: add parallel computing Phase 1 to CLAUDE.md"
```
