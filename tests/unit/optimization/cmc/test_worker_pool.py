"""Tests for persistent CMC worker pool."""

import multiprocessing
import os


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


def _raw_queue_worker(task, **init_kwargs):
    """Worker that puts results on an external queue and returns None."""
    ext_queue = init_kwargs["result_queue"]
    ext_queue.put(
        {
            "task_id": task["task_id"],
            "value": task["value"] * 2,
            "success": True,
        }
    )
    return None


def _init_fn_that_sets_flag(worker_id, **init_kwargs):
    """Init function that writes worker_id to a shared value."""
    shared_val = init_kwargs["shared_init_flag"]
    shared_val.value = worker_id + 100


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

        pool = WorkerPool(
            n_workers=1, worker_fn=_counting_worker, worker_init_kwargs={}
        )
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

    def test_context_manager(self):
        """Test WorkerPool as context manager."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        with WorkerPool(
            n_workers=1, worker_fn=_echo_worker, worker_init_kwargs={}
        ) as pool:
            assert pool.is_alive()
            pool.submit({"task_id": 0, "value": 42})
            result = pool.get_result(timeout=10.0)
            assert result["value"] == 42
        # After context exit, pool should be shut down
        assert not pool.is_alive()

    def test_shutdown_idempotent(self):
        """Test that calling shutdown() twice is safe."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=1, worker_fn=_echo_worker, worker_init_kwargs={})
        pool.shutdown()
        assert not pool.is_alive()
        pool.shutdown()  # Should not raise
        assert not pool.is_alive()

    def test_worker_init_fn(self):
        """Test that worker_init_fn is called once per worker before tasks."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        ctx = multiprocessing.get_context("spawn")
        shared_val = ctx.Value("i", 0)

        pool = WorkerPool(
            n_workers=1,
            worker_fn=_echo_worker,
            worker_init_kwargs={"shared_init_flag": shared_val},
            worker_init_fn=_init_fn_that_sets_flag,
        )
        try:
            # Submit a task to ensure the worker has started and init_fn ran
            pool.submit({"task_id": 0, "value": 1})
            pool.get_result(timeout=10.0)
            # init_fn sets value to worker_id + 100; worker_id=0 -> 100
            assert shared_val.value == 100
        finally:
            pool.shutdown()

    def test_none_result_not_queued(self):
        """Test that worker_fn returning None does not put on result queue."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        ctx = multiprocessing.get_context("spawn")
        ext_queue = ctx.Queue()

        pool = WorkerPool(
            n_workers=1,
            worker_fn=_raw_queue_worker,
            worker_init_kwargs={"result_queue": ext_queue},
        )
        try:
            pool.submit({"task_id": 0, "value": 21})
            # Result should be on the external queue, not the pool's queue
            result = ext_queue.get(timeout=10.0)
            assert result["value"] == 42
            assert result["success"] is True
            # Pool's internal queue should be empty (None was not put there)
            assert not pool.results_pending()
        finally:
            pool.shutdown()
            ext_queue.close()

    def test_result_queue_property(self):
        """Test that result_queue property returns the internal queue."""
        from homodyne.optimization.cmc.backends.worker_pool import WorkerPool

        pool = WorkerPool(n_workers=1, worker_fn=_echo_worker, worker_init_kwargs={})
        try:
            assert pool.result_queue is not None
            assert isinstance(pool.result_queue, multiprocessing.queues.Queue)
        finally:
            pool.shutdown()
