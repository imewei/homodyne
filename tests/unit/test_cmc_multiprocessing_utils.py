from homodyne.optimization.cmc.backends.multiprocessing import (
    _compute_threads_per_worker,
)


def test_compute_threads_per_worker_caps_by_physical_guess():
    # logical=16, workers=4 → safe_pool=8 → 2 threads/worker
    assert _compute_threads_per_worker(total_threads=16, workers=4) == 2


def test_compute_threads_per_worker_never_zero():
    # More workers than safe_pool should still return 1
    assert _compute_threads_per_worker(total_threads=14, workers=18) == 1
