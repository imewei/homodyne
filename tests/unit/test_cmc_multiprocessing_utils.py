from homodyne.optimization.cmc.backends.multiprocessing import (
    _compute_threads_per_worker,
    _get_physical_cores,
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
