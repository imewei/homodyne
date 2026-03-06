"""Persistent worker pool for CMC shard processing.

Spawns N persistent worker processes that each run an event loop,
processing multiple shards without JAX re-initialization overhead.
Falls back to per-shard process spawning when n_shards < 3.
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.context
import multiprocessing.process
import os
import queue
import threading
from collections.abc import Callable
from typing import Any

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def _estimate_physical_workers() -> int:
    """Estimate the number of worker processes from physical core count.

    Uses :func:`homodyne.device.detect_cpu_info` for accurate physical-core
    detection (via psutil), reserves one core for the main process, and
    guarantees at least 1 worker.

    Falls back to ``os.cpu_count() // 2 - 1`` when ``detect_cpu_info``
    is unavailable (e.g. psutil not installed).

    Returns
    -------
    int
        Recommended number of CMC worker processes (>= 1).
    """
    try:
        from homodyne.device import detect_cpu_info

        info = detect_cpu_info()
        physical: int | None = info.get("physical_cores")
        if physical is not None and physical >= 1:
            return max(1, physical - 1)
    except (ImportError, OSError, RuntimeError):
        logger.debug("detect_cpu_info unavailable, falling back to os.cpu_count")

    # Fallback: assume hyper-threading (logical = 2 * physical)
    logical = os.cpu_count() or 2
    return max(1, logical // 2 - 1)


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
        Signature: ``worker_fn(task: dict, **init_kwargs) -> dict | None``.
        Must be picklable (module-level function).
        If it returns ``None``, the pool does not put a result on the
        result queue (useful when the worker manages its own queue).
    worker_init_kwargs : dict
        One-time kwargs passed to every worker_fn call.
    worker_init_fn : callable or None
        Optional one-time initialization function called once per worker
        before the event loop starts.
        Signature: ``worker_init_fn(worker_id: int, **init_kwargs) -> None``.
        Use for expensive setup like JAX/OMP initialization.
    """

    def __init__(
        self,
        n_workers: int,
        worker_fn: Callable[..., dict[str, Any] | None],
        worker_init_kwargs: dict[str, Any],
        worker_init_fn: Callable[..., None] | None = None,
        startup_timeout: float = 120.0,
    ) -> None:
        self._n_workers = n_workers
        self._worker_fn = worker_fn
        self._init_kwargs = worker_init_kwargs
        self._init_fn = worker_init_fn
        self._startup_timeout = startup_timeout

        ctx = multiprocessing.get_context("spawn")
        self._task_queues: list[multiprocessing.Queue] = [
            ctx.Queue() for _ in range(n_workers)
        ]
        self._result_queue: multiprocessing.Queue = ctx.Queue(
            maxsize=self._n_workers * 4
        )
        self._processes: list[multiprocessing.process.BaseProcess] = []
        self._next_worker = 0
        self._alive = False
        self._lock = threading.Lock()

        self._start_workers(ctx)

    def _start_workers(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn persistent worker processes and wait for readiness.

        Each worker sends a ready signal after completing initialization
        (module imports + init_fn). This ensures tasks are not submitted
        to workers that are still starting up.
        """
        ready_queue: multiprocessing.Queue = ctx.Queue()
        for i in range(self._n_workers):
            p = ctx.Process(
                target=_worker_event_loop,
                args=(
                    i,
                    self._task_queues[i],
                    self._result_queue,
                    self._worker_fn,
                    self._init_kwargs,
                    self._init_fn,
                    ready_queue,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        # Wait for all workers to signal readiness
        ready_count = 0
        for _ in range(self._n_workers):
            try:
                ready_queue.get(timeout=self._startup_timeout)
                ready_count += 1
            except queue.Empty:
                logger.warning(
                    "Worker startup timed out after %.0fs (%d/%d ready)",
                    self._startup_timeout,
                    ready_count,
                    self._n_workers,
                )
                break

        try:
            ready_queue.close()
        except (OSError, ValueError):
            pass

        self._alive = True
        logger.info(
            "WorkerPool started: %d/%d workers ready", ready_count, self._n_workers
        )

    @property
    def n_workers(self) -> int:
        """Number of worker processes."""
        return self._n_workers

    @property
    def result_queue(self) -> multiprocessing.Queue:
        """The shared result queue drained by the parent."""
        return self._result_queue

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
        with self._lock:
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
        result: dict[str, Any] = self._result_queue.get(timeout=timeout)
        return result

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
                logger.warning("Worker %d did not exit gracefully, terminating", p.pid)
                p.terminate()
                p.join(timeout=15)
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
    worker_fn: Callable[..., dict[str, Any] | None],
    init_kwargs: dict[str, Any],
    init_fn: Callable[..., None] | None = None,
    ready_queue: multiprocessing.Queue | None = None,
) -> None:
    """Persistent worker event loop.

    Processes tasks until a None sentinel is received.
    Results (or errors) are put on result_queue.

    Parameters
    ----------
    init_fn : callable or None
        If provided, called once with ``(worker_id, **init_kwargs)``
        before the event loop starts.
    ready_queue : multiprocessing.Queue or None
        If provided, a ready signal is sent after initialization completes.
    """
    if init_fn is not None:
        init_fn(worker_id, **init_kwargs)

    if ready_queue is not None:
        try:
            ready_queue.put(worker_id)
        except (OSError, ValueError):
            pass

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
            if result is not None:
                result["worker_id"] = os.getpid()
                result_queue.put(result)
        except MemoryError:
            result_queue.put(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": "MemoryError: worker ran out of memory",
                    "worker_id": os.getpid(),
                }
            )
            # Drain remaining queued tasks so parent's get_result() won't hang
            while True:
                try:
                    remaining = task_queue.get(block=False)
                    if remaining is None:
                        break
                    result_queue.put(
                        {
                            "task_id": remaining.get("task_id", "unknown"),
                            "success": False,
                            "error": "Worker terminated due to MemoryError",
                            "worker_id": os.getpid(),
                        }
                    )
                except (queue.Empty, queue.Full, EOFError):
                    break
            break
        except Exception as e:
            result_queue.put(
                {
                    "task_id": task_id,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "worker_id": os.getpid(),
                }
            )
