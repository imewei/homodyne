"""Tests for persistent CMC worker pool."""

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
