import numpy as np

from homodyne.optimization.cmc.backends.multiprocessing import (
    SharedDataManager,
    _compute_lpt_schedule,
    _compute_threads_per_worker,
    _get_physical_cores,
    _load_shared_shard_data,
)


def test_compute_threads_per_worker_caps_by_physical_cores():
    """Test that thread allocation respects physical core count.

    With psutil optimization (#7), _get_physical_cores() returns actual
    physical cores, not just logical/2. The result depends on the
    machine's physical cores but must always be >= 1.
    """
    physical = _get_physical_cores()
    # With 4 workers, we should get physical_cores // 4 or at least 1
    result = _compute_threads_per_worker(total_threads=16, workers=4)
    expected_max = max(1, min(16, physical) // 4)
    assert result >= 1
    assert result <= expected_max or result == max(1, min(16, physical) // 4)


def test_compute_threads_per_worker_never_zero():
    # More workers than safe_pool should still return 1
    assert _compute_threads_per_worker(total_threads=14, workers=18) == 1


def test_shared_shard_data_roundtrip():
    """Test that per-shard arrays survive shared memory roundtrip."""
    rng = np.random.default_rng(42)
    shard_data_list = [
        {
            "data": rng.standard_normal(500).astype(np.float64),
            "t1": np.arange(500, dtype=np.float64),
            "t2": np.arange(500, dtype=np.float64) + 1,
            "phi_unique": np.array([0.0, 1.0, 2.0]),
            "phi_indices": rng.integers(0, 3, size=500).astype(np.int64),
            "noise_scale": 0.05,
        }
        for _ in range(3)
    ]

    mgr = SharedDataManager()
    try:
        refs = mgr.create_shared_shard_arrays(shard_data_list)
        assert len(refs) == 3

        for i, ref in enumerate(refs):
            loaded = _load_shared_shard_data(ref)
            np.testing.assert_array_equal(loaded["data"], shard_data_list[i]["data"])
            np.testing.assert_array_equal(loaded["t1"], shard_data_list[i]["t1"])
            np.testing.assert_array_equal(loaded["t2"], shard_data_list[i]["t2"])
            np.testing.assert_array_equal(
                loaded["phi_unique"], shard_data_list[i]["phi_unique"]
            )
            np.testing.assert_array_equal(
                loaded["phi_indices"], shard_data_list[i]["phi_indices"]
            )
            assert loaded["noise_scale"] == shard_data_list[i]["noise_scale"]
    finally:
        mgr.cleanup()


def _make_shard(size: int, noise: float) -> dict:
    """Helper: minimal shard dict for LPT scheduling tests."""
    return {"data": np.zeros(size), "noise_scale": noise}


def test_lpt_scheduling_largest_first():
    """Test that LPT scheduling sorts shards by size (largest first)."""
    shards = [
        _make_shard(100, 0.05),
        _make_shard(5000, 0.05),
        _make_shard(200, 0.05),
        _make_shard(3000, 0.05),
        _make_shard(1500, 0.05),
    ]
    pending = _compute_lpt_schedule(shards)

    # First dispatched should be the largest shard (index 1, size 5000)
    assert pending.popleft() == 1
    # Second should be index 3 (size 3000)
    assert pending.popleft() == 3
    # Third should be index 4 (size 1500)
    assert pending.popleft() == 4


def test_lpt_scheduling_noise_weighted():
    """Test that noise-weighted LPT prioritizes high-cost shards."""
    # Same size but different noise: shard 2 has highest noise
    shards = [
        _make_shard(5000, 0.01),
        _make_shard(5000, 0.05),
        _make_shard(5000, 0.10),
    ]
    pending = _compute_lpt_schedule(shards)

    # Highest noise shard (index 2) should be dispatched first
    assert pending.popleft() == 2
    # Then index 1 (medium noise)
    assert pending.popleft() == 1
    # Then index 0 (lowest noise)
    assert pending.popleft() == 0
