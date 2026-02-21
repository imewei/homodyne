"""
Unit tests for NLSQ chunking and stratification strategies.

Tests cover:
- Index-based stratification for zero-copy access (T034, FR-007)
- Angle distribution analysis
- Chunk size optimization
- Memory-efficient data access patterns
"""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.nlsq.strategies.chunking import (
    analyze_angle_distribution,
    create_angle_stratified_data,
    create_angle_stratified_indices,
)


class TestIndexBasedStratification:
    """Tests for index-based stratification (T034, FR-007).

    Index-based stratification reduces memory overhead from 2x (full copy)
    to ~1% (index array only) by returning indices instead of copied data.
    """

    @pytest.fixture
    def multi_angle_data(self):
        """Create multi-angle test data."""
        n_phi = 5
        n_points_per_angle = 1000
        phi_vals = np.linspace(-60, 60, n_phi)

        # Create data with varying points per angle
        phi_all = []
        for i, phi_val in enumerate(phi_vals):
            # Slightly different counts per angle to test edge cases
            n_points = n_points_per_angle + i * 10
            phi_all.extend([phi_val] * n_points)

        return np.array(phi_all, dtype=np.float64)

    def test_index_stratification_returns_indices(self, multi_angle_data):
        """T034: Verify create_angle_stratified_indices returns index array.

        Performance Optimization (Spec 001 - FR-007, T034): Index-based
        stratification should return indices for zero-copy data access.
        """
        indices, chunk_sizes = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=500,
        )

        # Should return indices, not data
        assert isinstance(indices, np.ndarray)
        assert indices.dtype in (np.int32, np.int64)

        # Indices should cover all original data points
        assert len(indices) == len(multi_angle_data)

        # Indices should be a permutation (no duplicates, all valid)
        assert len(np.unique(indices)) == len(indices)
        assert np.all(indices >= 0)
        assert np.all(indices < len(multi_angle_data))

    def test_index_stratification_chunk_sizes(self, multi_angle_data):
        """Test that chunk sizes are returned correctly."""
        indices, chunk_sizes = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=500,
        )

        # Chunk sizes should sum to total points
        assert sum(chunk_sizes) == len(multi_angle_data)

        # Each chunk size should be positive
        assert all(size > 0 for size in chunk_sizes)

    def test_index_stratification_angle_balance(self, multi_angle_data):
        """Test that index stratification maintains angle balance per chunk."""
        indices, chunk_sizes = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=500,
        )

        # Reorder phi values using indices
        phi_stratified = multi_angle_data[indices]

        # Check each chunk has all angles
        unique_angles = np.unique(multi_angle_data)
        start = 0
        for chunk_size in chunk_sizes:
            chunk_phi = phi_stratified[start : start + chunk_size]
            chunk_angles = np.unique(chunk_phi)

            # Each chunk should contain all angles (for balanced data)
            assert len(chunk_angles) == len(unique_angles), (
                f"Chunk missing angles: {set(unique_angles) - set(chunk_angles)}"
            )
            start += chunk_size

    def test_index_stratification_zero_copy_usage(self, multi_angle_data):
        """Test zero-copy data access pattern using indices."""
        # Create additional data arrays
        n_points = len(multi_angle_data)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        g2 = np.random.rand(n_points)

        indices, chunk_sizes = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=500,
        )

        # Use indices for zero-copy access
        phi_stratified = multi_angle_data[indices]
        t1_stratified = t1[indices]
        t2_stratified = t2[indices]
        g2_stratified = g2[indices]

        # Verify data integrity - should be able to reconstruct original
        reverse_indices = np.argsort(indices)
        np.testing.assert_array_equal(multi_angle_data, phi_stratified[reverse_indices])
        np.testing.assert_array_equal(t1, t1_stratified[reverse_indices])
        np.testing.assert_array_equal(t2, t2_stratified[reverse_indices])
        np.testing.assert_array_equal(g2, g2_stratified[reverse_indices])

    def test_index_vs_data_stratification_equivalence(self, multi_angle_data):
        """Test that index stratification produces same ordering as data stratification."""
        import jax.numpy as jnp

        n_points = len(multi_angle_data)
        t1 = np.random.rand(n_points)
        t2 = np.random.rand(n_points)
        g2 = np.random.rand(n_points)

        target_chunk_size = 500

        # Get indices from index-based stratification
        indices, chunk_sizes_idx = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=target_chunk_size,
        )

        # Get data from data-based stratification
        phi_data, t1_data, t2_data, g2_data, chunk_sizes_data = (
            create_angle_stratified_data(
                phi=jnp.array(multi_angle_data),
                t1=jnp.array(t1),
                t2=jnp.array(t2),
                g2_exp=jnp.array(g2),
                target_chunk_size=target_chunk_size,
            )
        )

        # Apply indices to original data
        phi_indexed = multi_angle_data[indices]
        t1_indexed = t1[indices]
        t2_indexed = t2[indices]
        g2_indexed = g2[indices]

        # Both methods should produce same ordering
        np.testing.assert_array_almost_equal(phi_indexed, np.asarray(phi_data))
        np.testing.assert_array_almost_equal(t1_indexed, np.asarray(t1_data))
        np.testing.assert_array_almost_equal(t2_indexed, np.asarray(t2_data))
        np.testing.assert_array_almost_equal(g2_indexed, np.asarray(g2_data))

        # Chunk sizes should match
        assert chunk_sizes_idx == chunk_sizes_data

    def test_memory_overhead_is_minimal(self, multi_angle_data):
        """Test that index array memory overhead is minimal compared to data copy."""
        len(multi_angle_data)

        # Memory for index array (int64)
        indices, _ = create_angle_stratified_indices(
            phi=multi_angle_data,
            target_chunk_size=500,
        )
        index_memory_bytes = indices.nbytes

        # Memory for data copy (4 arrays of float64)
        data_memory_bytes = multi_angle_data.nbytes * 4  # phi, t1, t2, g2

        # Index memory should be much smaller than data copy
        # Index: n_points * 8 bytes (int64)
        # Data: n_points * 8 bytes * 4 arrays = n_points * 32 bytes
        overhead_ratio = index_memory_bytes / data_memory_bytes
        assert overhead_ratio < 0.30, (
            f"Index overhead {overhead_ratio:.2%} exceeds 30% of data memory"
        )


class TestAngleDistributionAnalysis:
    """Tests for angle distribution analysis."""

    def test_single_angle_detection(self):
        """Test detection of single-angle data."""
        phi = np.zeros(1000)  # All same angle
        stats = analyze_angle_distribution(phi)

        assert stats.n_angles == 1
        assert stats.imbalance_ratio == 1.0

    def test_multi_angle_statistics(self):
        """Test statistics for multi-angle data."""
        # Create imbalanced distribution
        phi = np.concatenate(
            [
                np.zeros(1000),  # 1000 points at angle 0
                np.ones(500),  # 500 points at angle 1
            ]
        )
        stats = analyze_angle_distribution(phi)

        assert stats.n_angles == 2
        assert stats.imbalance_ratio == 1000 / 500  # max/min = 2.0

    def test_balanced_distribution(self):
        """Test statistics for perfectly balanced distribution."""
        n_angles = 5
        n_points_per_angle = 100
        phi = np.repeat(np.arange(n_angles), n_points_per_angle)

        stats = analyze_angle_distribution(phi)

        assert stats.n_angles == n_angles
        assert stats.imbalance_ratio == 1.0


class TestStratificationEdgeCases:
    """Tests for edge cases in stratification."""

    def test_single_angle_no_stratification(self):
        """Test that single-angle data skips stratification."""
        import jax.numpy as jnp

        phi = np.zeros(1000)
        t1 = np.random.rand(1000)
        t2 = np.random.rand(1000)
        g2 = np.random.rand(1000)

        phi_out, t1_out, t2_out, g2_out, chunk_sizes = create_angle_stratified_data(
            phi=jnp.array(phi),
            t1=jnp.array(t1),
            t2=jnp.array(t2),
            g2_exp=jnp.array(g2),
            target_chunk_size=500,
        )

        # Should return data unchanged
        np.testing.assert_array_equal(np.asarray(phi_out), phi)
        np.testing.assert_array_equal(np.asarray(t1_out), t1)
        np.testing.assert_array_equal(np.asarray(t2_out), t2)
        np.testing.assert_array_equal(np.asarray(g2_out), g2)

        # Single angle: one chunk containing all points
        assert chunk_sizes == [1000]

    def test_small_target_chunk_size(self):
        """Test stratification with very small target chunk size."""
        phi = np.repeat(np.arange(3), 100)  # 3 angles, 100 points each

        indices, chunk_sizes = create_angle_stratified_indices(
            phi=phi,
            target_chunk_size=10,  # Very small
        )

        # Should create many small chunks
        assert len(chunk_sizes) > 10
        # But still cover all data
        assert sum(chunk_sizes) == len(phi)

    def test_large_target_chunk_size(self):
        """Test stratification with target larger than data."""
        phi = np.repeat(np.arange(5), 100)  # 500 points total

        indices, chunk_sizes = create_angle_stratified_indices(
            phi=phi,
            target_chunk_size=10000,  # Larger than data
        )

        # Should create single chunk
        assert len(chunk_sizes) == 1
        assert chunk_sizes[0] == len(phi)
