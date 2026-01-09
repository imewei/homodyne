"""Tests for CMC data preparation module.

Updated: 2026-01-08 (v2.14.2 - Diagonal filtering support)
"""

import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.data_prep import (  # noqa: E402
    PreparedData,
    estimate_noise_scale,
    extract_phi_info,
    prepare_mcmc_data,
    validate_pooled_data,
)


class TestValidatePooledData:
    """Tests for validate_pooled_data function."""

    def test_valid_data(self):
        """Test validation passes for valid data."""
        n = 1000
        data = np.random.randn(n) + 1.0
        t1 = np.random.rand(n)
        t2 = np.random.rand(n)
        phi = np.random.choice([0.0, 30.0, 60.0], size=n)

        # Should not raise
        validate_pooled_data(data, t1, t2, phi)

    def test_mismatched_lengths(self):
        """Test validation fails for mismatched array lengths."""
        data = np.random.randn(100)
        t1 = np.random.rand(100)
        t2 = np.random.rand(99)  # Different length
        phi = np.random.rand(100)

        with pytest.raises(ValueError, match="does not match"):
            validate_pooled_data(data, t1, t2, phi)

    def test_nan_in_data(self):
        """Test validation fails for NaN values in data."""
        n = 100
        data = np.random.randn(n)
        data[50] = np.nan
        t1 = np.random.rand(n)
        t2 = np.random.rand(n)
        phi = np.random.rand(n)

        with pytest.raises(ValueError, match="NaN"):
            validate_pooled_data(data, t1, t2, phi)

    def test_inf_in_data(self):
        """Test validation fails for inf values in data."""
        n = 100
        data = np.random.randn(n)
        data[50] = np.inf
        t1 = np.random.rand(n)
        t2 = np.random.rand(n)
        phi = np.random.rand(n)

        with pytest.raises(ValueError, match="inf"):
            validate_pooled_data(data, t1, t2, phi)

    def test_empty_arrays(self):
        """Test validation with empty arrays.

        Note: Empty arrays pass basic validation but will fail when used
        in prepare_mcmc_data due to compute_data_statistics.
        """
        data = np.array([])
        t1 = np.array([])
        t2 = np.array([])
        phi = np.array([])

        # Empty arrays pass validate_pooled_data (no explicit empty check)
        # but would fail downstream in prepare_mcmc_data
        # This validates the function handles edge case gracefully
        validate_pooled_data(data, t1, t2, phi)  # Should not raise


class TestExtractPhiInfo:
    """Tests for extract_phi_info function."""

    def test_single_phi(self):
        """Test extraction with single phi angle."""
        phi = np.array([0.0, 0.0, 0.0, 0.0])

        phi_unique, phi_indices = extract_phi_info(phi)

        assert len(phi_unique) == 1
        assert phi_unique[0] == 0.0
        np.testing.assert_array_equal(phi_indices, [0, 0, 0, 0])

    def test_multiple_phi(self):
        """Test extraction with multiple phi angles."""
        phi = np.array([0.0, 30.0, 0.0, 60.0, 30.0, 60.0])

        phi_unique, phi_indices = extract_phi_info(phi)

        assert len(phi_unique) == 3
        # Indices should map each phi to its position in unique array
        for i, p in enumerate(phi):
            assert phi_unique[phi_indices[i]] == p

    def test_phi_order_preserved(self):
        """Test unique phi values are sorted."""
        phi = np.array([60.0, 30.0, 0.0, 30.0, 60.0])

        phi_unique, _ = extract_phi_info(phi)

        # Should be sorted
        assert np.all(phi_unique[:-1] <= phi_unique[1:])


class TestEstimateNoiseScale:
    """Tests for estimate_noise_scale function."""

    def test_low_noise_data(self):
        """Test noise estimation on low-noise data."""
        # Simulate C2 data around 1.0 with low noise
        data = np.ones(1000) + np.random.randn(1000) * 0.01

        noise = estimate_noise_scale(data)

        assert 0.005 < noise < 0.05  # Should detect low noise

    def test_high_noise_data(self):
        """Test noise estimation on high-noise data."""
        # Simulate C2 data with high noise
        data = np.ones(1000) + np.random.randn(1000) * 0.2

        noise = estimate_noise_scale(data)

        assert 0.1 < noise < 0.5  # Should detect higher noise

    def test_minimum_noise_floor(self):
        """Test noise has minimum floor."""
        # Constant data (zero variance)
        data = np.ones(1000)

        noise = estimate_noise_scale(data)

        assert noise > 0  # Should have minimum floor


class TestPreparedData:
    """Tests for PreparedData dataclass."""

    def test_creation(self):
        """Test PreparedData creation."""
        n = 100
        prepared = PreparedData(
            data=np.random.randn(n),
            t1=np.random.rand(n),
            t2=np.random.rand(n),
            phi=np.zeros(n),
            phi_unique=np.array([0.0]),
            phi_indices=np.zeros(n, dtype=int),
            n_total=n,
            n_phi=1,
            noise_scale=0.1,
        )

        assert len(prepared.data) == n
        assert prepared.n_phi == 1
        assert prepared.noise_scale == 0.1
        assert prepared.n_total == n


class TestPrepareMCMCData:
    """Tests for prepare_mcmc_data function."""

    def test_full_preparation(self):
        """Test complete data preparation pipeline."""
        n = 1000
        data = np.random.randn(n) * 0.1 + 1.0
        t1 = np.random.rand(n) * 10
        t2 = np.random.rand(n) * 10
        phi = np.random.choice([0.0, 30.0, 60.0], size=n)

        prepared = prepare_mcmc_data(data, t1, t2, phi)

        assert isinstance(prepared, PreparedData)
        assert len(prepared.data) == n
        assert prepared.n_phi == 3
        assert prepared.noise_scale > 0
        assert len(prepared.phi_unique) == 3

    def test_phi_indices_consistency(self):
        """Test phi indices correctly map to unique values."""
        n = 100
        data = np.random.randn(n) + 1.0
        t1 = np.random.rand(n)
        t2 = np.random.rand(n)
        phi = np.random.choice([0.0, 45.0, 90.0], size=n)

        prepared = prepare_mcmc_data(data, t1, t2, phi)

        # Each phi should map to correct unique value
        for i in range(n):
            assert prepared.phi_unique[prepared.phi_indices[i]] == phi[i]


class TestDiagonalFiltering:
    """Tests for diagonal filtering in prepare_mcmc_data (v2.14.2+).

    The unified diagonal handling ensures CMC excludes diagonal points (t1==t2)
    from the MCMC likelihood, consistent with NLSQ diagonal masking.
    """

    def test_filter_diagonal_default_true(self):
        """Test that diagonal filtering is enabled by default."""
        n = 100
        # Create mixed data: 20 diagonal, 80 off-diagonal
        t1 = np.concatenate([np.arange(20), np.arange(80)])
        t2 = np.concatenate([np.arange(20), np.arange(80) + 1])  # +1 for off-diagonal
        data = np.random.randn(n) + 1.0
        phi = np.zeros(n)

        # With filter_diagonal=True (default), diagonal points should be removed
        prepared = prepare_mcmc_data(data, t1, t2, phi)

        # 20 diagonal points should be removed, 80 remain
        assert prepared.n_total == 80

    def test_filter_diagonal_all_diagonal_raises_error(self):
        """Test that filtering all diagonal points raises appropriate error."""
        n = 100
        # Create data with ALL diagonal points
        t1 = np.arange(n, dtype=float)
        t2 = np.arange(n, dtype=float)  # All diagonal
        data = np.random.randn(n) + 1.0
        phi = np.zeros(n)

        # Filtering all points should raise a clear error
        with pytest.raises(ValueError, match="All.*diagonal|No off-diagonal"):
            prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=True)

    def test_filter_diagonal_removes_diagonal_points(self):
        """Test that diagonal points are removed."""
        n = 100
        # Create mixed data: 20 diagonal, 80 off-diagonal
        t1 = np.concatenate([np.arange(20), np.arange(80)])
        t2 = np.concatenate([np.arange(20), np.arange(80) + 1])  # +1 for off-diagonal
        data = np.random.randn(n) + 1.0
        phi = np.zeros(n)

        prepared = prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=True)

        # 20 diagonal points should be removed, 80 remain
        assert prepared.n_total == 80

    def test_filter_diagonal_false_preserves_all(self):
        """Test that filter_diagonal=False preserves diagonal points."""
        n = 100
        # Create mixed data: 20 diagonal, 80 off-diagonal
        t1 = np.concatenate([np.arange(20), np.arange(80)])
        t2 = np.concatenate([np.arange(20), np.arange(80) + 1])
        data = np.random.randn(n) + 1.0
        phi = np.zeros(n)

        prepared = prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=False)

        # All points should be preserved
        assert prepared.n_total == n

    def test_filter_diagonal_updates_phi_indices(self):
        """Test that phi indices are correctly recomputed after filtering."""
        n = 100
        # Create data with multiple phi angles
        # First 20 points: diagonal (will be removed)
        # Next 80 points: off-diagonal with varied phi
        t1 = np.concatenate([np.arange(20), np.arange(80)])
        t2 = np.concatenate([np.arange(20), np.arange(80) + 1])
        phi = np.concatenate([np.zeros(20), np.repeat([0.0, 30.0, 60.0, 90.0], 20)])
        data = np.random.randn(n) + 1.0

        prepared = prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=True)

        # Should have 80 points remaining
        assert prepared.n_total == 80
        # Should still have 4 unique phi angles
        assert prepared.n_phi == 4
        # Each remaining phi should map correctly
        for i in range(prepared.n_total):
            assert prepared.phi_unique[prepared.phi_indices[i]] == prepared.phi[i]

    def test_filter_diagonal_no_diagonal_points(self):
        """Test filtering when there are no diagonal points."""
        n = 100
        # Create data with no diagonal points (t1 != t2)
        t1 = np.arange(n, dtype=float)
        t2 = np.arange(n, dtype=float) + 1  # All off-diagonal
        data = np.random.randn(n) + 1.0
        phi = np.zeros(n)

        prepared = prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=True)

        # No points should be removed
        assert prepared.n_total == n

    def test_filter_diagonal_realistic_xpcs_data(self):
        """Test diagonal filtering with realistic XPCS-like data structure."""
        # Simulate typical XPCS data: n_phi angles, n_t time points
        n_phi = 5
        n_t = 20
        n_total = n_phi * n_t * n_t  # Full grid

        # Create full grid data
        phi_vals = np.array([0.0, 30.0, 60.0, 90.0, 120.0])
        t_vals = np.arange(n_t, dtype=float)

        # Create pooled data
        phi_list = []
        t1_list = []
        t2_list = []
        for phi_idx in range(n_phi):
            for i in range(n_t):
                for j in range(n_t):
                    phi_list.append(phi_vals[phi_idx])
                    t1_list.append(t_vals[i])
                    t2_list.append(t_vals[j])

        phi = np.array(phi_list)
        t1 = np.array(t1_list)
        t2 = np.array(t2_list)
        data = np.random.randn(n_total) * 0.1 + 1.0

        # Count expected diagonal points
        n_diagonal_per_angle = n_t  # One diagonal point per time index
        n_diagonal_total = n_phi * n_diagonal_per_angle

        prepared = prepare_mcmc_data(data, t1, t2, phi, filter_diagonal=True)

        # Expected points remaining
        expected_remaining = n_total - n_diagonal_total
        assert prepared.n_total == expected_remaining
        assert prepared.n_phi == n_phi

        # Verify no diagonal points remain
        assert np.all(prepared.t1 != prepared.t2)
