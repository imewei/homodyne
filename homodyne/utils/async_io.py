"""Async I/O utilities for pipeline overlap.

Thread-based prefetching and background writing to hide I/O latency.
GIL-safe since HDF5 and numpy release the GIL during I/O.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Thread
from typing import Any, TypeVar

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
        return result  # type: ignore[return-value]


class AsyncWriter:
    """Background thread pool for result serialization.

    Parameters
    ----------
    max_workers : int
        Maximum concurrent write threads.
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: list[Future[None]] = []

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
        errors: list[Exception] = []
        for future in self._futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.warning("Background write failed: %s", e)
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
