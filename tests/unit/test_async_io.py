"""Tests for async I/O utilities."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np


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
