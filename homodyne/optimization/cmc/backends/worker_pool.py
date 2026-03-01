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
    ) -> None:
        self._n_workers = n_workers
        self._worker_fn = worker_fn
        self._init_kwargs = worker_init_kwargs
        self._init_fn = worker_init_fn

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
                    self._init_fn,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
        self._alive = True
        logger.info("WorkerPool started: %d persistent workers", self._n_workers)

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
                logger.warning(
                    "Worker %d did not exit gracefully, terminating", p.pid
                )
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
) -> None:
    """Persistent worker event loop.

    Processes tasks until a None sentinel is received.
    Results (or errors) are put on result_queue.

    Parameters
    ----------
    init_fn : callable or None
        If provided, called once with ``(worker_id, **init_kwargs)``
        before the event loop starts.
    """
    if init_fn is not None:
        init_fn(worker_id, **init_kwargs)

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
