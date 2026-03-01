"""Parallel chunk accumulation for NLSQ streaming optimizer.

Dispatches chunk computations to a process pool and reduces
J^T J, J^T r, chi2 accumulators. Falls back to sequential
when n_chunks < 10 or pool creation fails.

Matrix addition is associative and commutative, so parallel
accumulation produces identical results to sequential.
"""

from __future__ import annotations

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

_MIN_CHUNKS_FOR_PARALLEL = 10


def should_use_parallel_accumulation(n_chunks: int) -> bool:
    """Determine if parallel accumulation is worthwhile.

    Parameters
    ----------
    n_chunks : int
        Number of chunks to accumulate.

    Returns
    -------
    bool
        True if n_chunks >= threshold for parallel accumulation.
    """
    return n_chunks >= _MIN_CHUNKS_FOR_PARALLEL


def accumulate_chunks_sequential(
    chunks: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Accumulate chunk results sequentially.

    Parameters
    ----------
    chunks : list of (JtJ, Jtr, chi2) tuples
        Each tuple contains:
        - JtJ: (n_params, n_params) symmetric matrix
        - Jtr: (n_params,) vector
        - chi2: scalar cost contribution

    Returns
    -------
    total_JtJ : np.ndarray
        Sum of all JtJ matrices.
    total_Jtr : np.ndarray
        Sum of all Jtr vectors.
    total_chi2 : float
        Sum of all chi2 values.
    count : int
        Number of chunks accumulated.
    """
    total_JtJ: np.ndarray | None = None
    total_Jtr: np.ndarray | None = None
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

    assert total_JtJ is not None and total_Jtr is not None, "Empty chunks list"
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
        Each tuple contains:
        - JtJ: (n_params, n_params) symmetric matrix
        - Jtr: (n_params,) vector
        - chi2: scalar cost contribution
    n_workers : int
        Number of parallel workers.

    Returns
    -------
    total_JtJ : np.ndarray
        Sum of all JtJ matrices.
    total_Jtr : np.ndarray
        Sum of all Jtr vectors.
    total_chi2 : float
        Sum of all chi2 values.
    count : int
        Number of chunks accumulated.
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

            total_JtJ: np.ndarray | None = None
            total_Jtr: np.ndarray | None = None
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

        assert total_JtJ is not None and total_Jtr is not None, "No partitions"
        return total_JtJ, total_Jtr, total_chi2, total_count

    except (OSError, RuntimeError) as e:
        logger.warning(
            "Parallel chunk accumulation failed (%s), falling back to sequential", e
        )
        return accumulate_chunks_sequential(chunks)
