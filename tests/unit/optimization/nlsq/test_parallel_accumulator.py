"""Tests for parallel chunk accumulation in NLSQ streaming."""

from unittest.mock import patch

import numpy as np
import pytest


class TestParallelThreshold:
    """Test parallel activation threshold."""

    def test_below_threshold(self):
        from homodyne.optimization.nlsq.parallel_accumulator import (
            should_use_parallel_accumulation,
        )

        assert not should_use_parallel_accumulation(n_chunks=5)
        assert not should_use_parallel_accumulation(n_chunks=9)

    def test_at_and_above_threshold(self):
        from homodyne.optimization.nlsq.parallel_accumulator import (
            should_use_parallel_accumulation,
        )

        assert should_use_parallel_accumulation(n_chunks=10)
        assert should_use_parallel_accumulation(n_chunks=500)


class TestAccumulationCorrectness:
    """Test that parallel == sequential accumulation."""

    def test_identical_results(self):
        """Core correctness: parallel accumulation == sequential."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_parallel,
            accumulate_chunks_sequential,
        )

        rng = np.random.default_rng(42)
        n_params = 5
        n_chunks = 20

        chunks = []
        for _ in range(n_chunks):
            JtJ = rng.standard_normal((n_params, n_params))
            JtJ = JtJ @ JtJ.T  # Symmetric positive definite
            Jtr = rng.standard_normal(n_params)
            chi2 = rng.uniform(10, 100)
            chunks.append((JtJ, Jtr, chi2))

        seq_JtJ, seq_Jtr, seq_chi2, seq_count = accumulate_chunks_sequential(chunks)
        par_JtJ, par_Jtr, par_chi2, par_count = accumulate_chunks_parallel(
            chunks, n_workers=2
        )

        np.testing.assert_allclose(seq_JtJ, par_JtJ, rtol=1e-14)
        np.testing.assert_allclose(seq_Jtr, par_Jtr, rtol=1e-14)
        np.testing.assert_allclose(seq_chi2, par_chi2, rtol=1e-14)
        assert seq_count == par_count

    def test_single_chunk(self):
        """Test with a single chunk (no parallelism needed)."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_sequential,
        )

        JtJ = np.eye(3)
        Jtr = np.ones(3)
        chi2 = 5.0

        result = accumulate_chunks_sequential([(JtJ, Jtr, chi2)])
        np.testing.assert_array_equal(result[0], JtJ)
        np.testing.assert_array_equal(result[1], Jtr)
        assert result[2] == chi2
        assert result[3] == 1

    def test_empty_chunks_raises_valueerror(self):
        """Test that empty chunks list raises ValueError (not assert)."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_sequential,
        )

        with pytest.raises(ValueError, match="Cannot accumulate empty chunks list"):
            accumulate_chunks_sequential([])

    def test_parallel_fallback_on_error(self):
        """Test parallel fallback to sequential on ProcessPoolExecutor failure."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_parallel,
            accumulate_chunks_sequential,
        )

        rng = np.random.default_rng(99)
        n_params = 3
        chunks = [
            (
                rng.standard_normal((n_params, n_params)),
                rng.standard_normal(n_params),
                float(i),
            )
            for i in range(12)
        ]

        # Mock ProcessPoolExecutor to raise RuntimeError (simulating pool failure)
        with patch(
            "concurrent.futures.ProcessPoolExecutor",
            side_effect=RuntimeError("Pool creation failed"),
        ):
            # Should fall back to sequential and still return correct results
            par_JtJ, par_Jtr, par_chi2, par_count = accumulate_chunks_parallel(
                chunks, n_workers=2
            )

        seq_JtJ, seq_Jtr, seq_chi2, seq_count = accumulate_chunks_sequential(chunks)
        np.testing.assert_allclose(par_JtJ, seq_JtJ, rtol=1e-14)
        np.testing.assert_allclose(par_Jtr, seq_Jtr, rtol=1e-14)
        assert par_chi2 == pytest.approx(seq_chi2)
        assert par_count == seq_count

    def test_parallel_more_workers_than_chunks(self):
        """Test parallel with n_workers > n_chunks (sparse partitioning)."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_parallel,
            accumulate_chunks_sequential,
        )

        rng = np.random.default_rng(77)
        n_params = 3
        chunks = [
            (
                rng.standard_normal((n_params, n_params)),
                rng.standard_normal(n_params),
                float(i),
            )
            for i in range(12)
        ]

        # 8 workers for 12 chunks: some partitions will have 1 chunk, some 2
        par_JtJ, par_Jtr, par_chi2, par_count = accumulate_chunks_parallel(
            chunks, n_workers=8
        )
        seq_JtJ, seq_Jtr, seq_chi2, seq_count = accumulate_chunks_sequential(chunks)

        np.testing.assert_allclose(par_JtJ, seq_JtJ, rtol=1e-14)
        np.testing.assert_allclose(par_Jtr, seq_Jtr, rtol=1e-14)
        assert par_count == seq_count

    def test_parallel_zero_workers_fallback(self):
        """Test n_workers=0 falls back to sequential."""
        from homodyne.optimization.nlsq.parallel_accumulator import (
            accumulate_chunks_parallel,
            accumulate_chunks_sequential,
        )

        rng = np.random.default_rng(55)
        n_params = 3
        chunks = [
            (
                rng.standard_normal((n_params, n_params)),
                rng.standard_normal(n_params),
                float(i),
            )
            for i in range(12)
        ]

        par_JtJ, par_Jtr, par_chi2, par_count = accumulate_chunks_parallel(
            chunks, n_workers=0
        )
        seq_JtJ, seq_Jtr, seq_chi2, seq_count = accumulate_chunks_sequential(chunks)

        np.testing.assert_allclose(par_JtJ, seq_JtJ, rtol=1e-14)
        assert par_count == seq_count
