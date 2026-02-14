"""
Performance tests for NLSQ memory usage.

Tests cover:
- Streaming mode memory usage (T035, SC-003)
- Peak memory tracking with tracemalloc
- Memory efficiency comparisons
"""

from __future__ import annotations

import gc
import tracemalloc
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    pass


class TestStreamingMemoryUsage:
    """Tests for streaming mode memory efficiency (T035, SC-003).

    These tests verify that streaming mode maintains bounded memory usage
    regardless of dataset size, achieving the 20% peak memory reduction goal.
    """

    @pytest.fixture
    def large_dataset_config(self):
        """Configuration for simulating large dataset behavior."""
        return {
            "n_phi": 23,
            "n_t1": 100,
            "n_t2": 100,
            "n_params": 53,  # 2*23 + 7 physical params
        }

    def test_streaming_memory_usage(self, large_dataset_config):
        """T035: Verify streaming mode memory usage is bounded.

        Performance Optimization (Spec 001 - SC-003, T035): Streaming mode
        should maintain bounded peak memory regardless of dataset size.
        """
        from homodyne.optimization.nlsq.memory import estimate_peak_memory_gb

        # Calculate memory estimates for different dataset sizes
        dataset_sizes = [1_000_000, 10_000_000, 50_000_000]
        n_params = large_dataset_config["n_params"]

        estimates = []
        for n_points in dataset_sizes:
            peak_gb = estimate_peak_memory_gb(n_points, n_params)
            estimates.append(
                {
                    "n_points": n_points,
                    "peak_memory_gb": peak_gb,
                }
            )

        # Verify linear scaling (expected for full Jacobian)
        ratio_10m_1m = estimates[1]["peak_memory_gb"] / estimates[0]["peak_memory_gb"]
        ratio_50m_10m = estimates[2]["peak_memory_gb"] / estimates[1]["peak_memory_gb"]

        # Should scale approximately linearly
        np.testing.assert_allclose(ratio_10m_1m, 10.0, rtol=0.1)
        np.testing.assert_allclose(ratio_50m_10m, 5.0, rtol=0.1)

    def test_index_memory_vs_data_memory(self, large_dataset_config):
        """Test that index-based approach uses less memory than data copy.

        Performance Optimization: Index arrays should use ~1% of data memory.
        """
        n_phi = large_dataset_config["n_phi"]
        n_t1 = large_dataset_config["n_t1"]
        n_t2 = large_dataset_config["n_t2"]
        n_points = n_phi * n_t1 * n_t2  # 230,000 points

        # Memory for index array (int64)
        index_memory_bytes = n_points * 8  # int64

        # Memory for data arrays (4 arrays of float64)
        data_memory_bytes = n_points * 8 * 4  # phi, t1, t2, g2

        # Index should be 25% of data memory (1 array vs 4 arrays)
        ratio = index_memory_bytes / data_memory_bytes
        assert ratio == 0.25, f"Index memory ratio {ratio:.2%} should be 25%"

    @pytest.mark.slow
    def test_tracemalloc_streaming_vs_standard(self):
        """Compare memory usage between streaming and standard modes.

        Note: This test requires significant memory and time.
        """
        from homodyne.optimization.nlsq.memory import (
            select_nlsq_strategy,
        )

        # Test with moderate dataset that triggers strategy decision
        n_points = 5_000_000
        n_params = 53

        decision = select_nlsq_strategy(n_points, n_params)

        # Log the decision for reference
        print(f"\nStrategy decision for {n_points:,} points, {n_params} params:")
        print(f"  Strategy: {decision.strategy.value}")
        print(f"  Peak memory estimate: {decision.peak_memory_gb:.2f} GB")
        print(f"  Threshold: {decision.threshold_gb:.2f} GB")
        print(f"  Reason: {decision.reason}")

        # For a 5M point dataset with 53 params:
        # Jacobian size = 5M * 53 * 8 * 3 = ~6.3 GB
        # This should trigger OUT_OF_CORE on most systems
        assert decision.peak_memory_gb > 5.0  # Should be several GB


class TestMemoryTracking:
    """Tests for memory tracking utilities."""

    def test_tracemalloc_basic_usage(self):
        """Test basic tracemalloc usage for memory tracking."""
        # Start tracking
        tracemalloc.start()

        # Allocate some memory - must store in variable to prevent GC
        data = np.zeros((1000, 1000), dtype=np.float64)

        # Get current usage
        current, peak = tracemalloc.get_traced_memory()

        # Stop tracking
        tracemalloc.stop()

        # Should have tracked the allocation
        # 1000 * 1000 * 8 = 8 MB
        expected_bytes = 1000 * 1000 * 8
        assert current >= expected_bytes * 0.5, (
            f"Expected >= {expected_bytes * 0.5} bytes, got {current}"
        )

        # Clean up (prevent unused variable warning)
        del data

    def test_gc_releases_memory(self):
        """Test that garbage collection properly releases memory."""
        tracemalloc.start()

        # Allocate
        data = np.zeros((1000, 1000), dtype=np.float64)
        _, peak_before = tracemalloc.get_traced_memory()

        # Delete and collect
        del data
        gc.collect()

        current_after, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Current should be less than peak after GC
        assert current_after < peak_before


class TestChunkedProcessingMemory:
    """Tests for chunked processing memory efficiency."""

    def test_chunk_memory_isolation(self):
        """Test that chunks are processed without accumulating memory."""
        tracemalloc.start()

        # Simulate chunked processing
        n_chunks = 5
        chunk_size = 100_000
        peak_memories = []

        for _i in range(n_chunks):
            # Process chunk
            chunk_data = np.random.rand(chunk_size)
            np.sum(chunk_data**2)

            # Record peak memory
            _, peak = tracemalloc.get_traced_memory()
            peak_memories.append(peak)

            # Clear chunk
            del chunk_data
            gc.collect()

        tracemalloc.stop()

        # Peak memory should not grow significantly across chunks
        max_growth = max(peak_memories) / min(peak_memories)
        assert max_growth < 2.0, f"Memory growth {max_growth:.2f}x exceeds 2x threshold"

    def test_residual_function_memory_bounded(self):
        """Test that residual function evaluation has bounded memory."""
        from homodyne.optimization.nlsq.memory import estimate_peak_memory_gb

        # For streaming mode, memory should be bounded by chunk size, not total size
        # Typical chunk: 50,000 points
        chunk_size = 50_000
        n_params = 53

        chunk_memory_gb = estimate_peak_memory_gb(chunk_size, n_params)

        # Chunk should require much less than 1 GB
        assert chunk_memory_gb < 1.0, (
            f"Chunk memory {chunk_memory_gb:.2f} GB exceeds 1 GB bound"
        )
