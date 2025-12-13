"""Tests for data-driven quantile estimation of contrast/offset.

These tests verify the physics-informed quantile estimation that computes
initial values for contrast and offset from C2 data, using the correlation
decay structure: C2 = contrast * g1^2 + offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.cmc.priors import (
    build_init_values_dict,
    estimate_contrast_offset_from_data,
    estimate_per_angle_scaling,
)


class TestEstimateContrastOffsetFromData:
    """Tests for estimate_contrast_offset_from_data function."""

    def test_basic_estimation_accuracy(self):
        """Verify estimation recovers known contrast/offset within tolerance."""
        np.random.seed(42)
        n_points = 10000

        # Generate synthetic C2 data with known parameters
        true_contrast = 0.35
        true_offset = 0.85
        decay_rate = 0.1

        t1 = np.random.uniform(0.1, 100, n_points)
        t2 = np.random.uniform(0.1, 100, n_points)
        delta_t = np.abs(t1 - t2)

        # C2 = contrast * g1^2 + offset, where g1^2 = exp(-decay_rate * delta_t)
        g1_squared = np.exp(-decay_rate * delta_t)
        c2_data = (
            true_contrast * g1_squared
            + true_offset
            + np.random.normal(0, 0.02, n_points)
        )

        contrast_est, offset_est = estimate_contrast_offset_from_data(
            c2_data, t1, t2, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5)
        )

        # Should be within 15% of true values
        assert abs(contrast_est - true_contrast) < 0.15 * true_contrast
        assert abs(offset_est - true_offset) < 0.15 * true_offset

    def test_bounds_clipping(self):
        """Verify estimates are clipped to valid bounds."""
        n_points = 1000
        t1 = np.linspace(0, 100, n_points)
        t2 = np.zeros(n_points)

        # Data that would estimate outside bounds
        c2_high = np.ones(n_points) * 3.0  # Would give contrast > 1, offset > 1.5
        c2_low = np.ones(n_points) * 0.3  # Would give offset < 0.5

        contrast_bounds = (0.0, 1.0)
        offset_bounds = (0.5, 1.5)

        # High data should clip to upper bounds
        contrast_h, offset_h = estimate_contrast_offset_from_data(
            c2_high, t1, t2, contrast_bounds, offset_bounds
        )
        assert contrast_h <= contrast_bounds[1]
        assert offset_h <= offset_bounds[1]

        # Low data should clip to lower bounds
        contrast_l, offset_l = estimate_contrast_offset_from_data(
            c2_low, t1, t2, contrast_bounds, offset_bounds
        )
        assert contrast_l >= contrast_bounds[0]
        assert offset_l >= offset_bounds[0]

    def test_insufficient_data_fallback(self):
        """Verify fallback to midpoint with insufficient data."""
        # Less than 100 points
        n_points = 50
        c2_data = np.random.uniform(0.8, 1.2, n_points)
        t1 = np.random.uniform(0, 10, n_points)
        t2 = np.random.uniform(0, 10, n_points)

        contrast_bounds = (0.0, 1.0)
        offset_bounds = (0.5, 1.5)

        contrast_est, offset_est = estimate_contrast_offset_from_data(
            c2_data, t1, t2, contrast_bounds, offset_bounds
        )

        # Should return midpoints
        expected_contrast = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        expected_offset = (offset_bounds[0] + offset_bounds[1]) / 2.0

        assert contrast_est == expected_contrast
        assert offset_est == expected_offset

    def test_large_lag_offset_estimation(self):
        """Verify offset is estimated from large-lag region."""
        np.random.seed(123)
        n_points = 5000

        true_offset = 0.9
        true_contrast = 0.4

        # Create data where offset dominates at large lags
        t1 = np.concatenate(
            [
                np.random.uniform(0, 1, n_points // 2),  # Small lags
                np.random.uniform(50, 100, n_points // 2),  # Large lags
            ]
        )
        t2 = np.zeros(n_points)
        delta_t = np.abs(t1 - t2)

        # g1^2 -> 0 at large lags, so C2 -> offset
        g1_squared = np.exp(-0.2 * delta_t)
        c2_data = (
            true_contrast * g1_squared
            + true_offset
            + np.random.normal(0, 0.01, n_points)
        )

        _, offset_est = estimate_contrast_offset_from_data(
            c2_data, t1, t2, contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5)
        )

        # Offset estimate should be close to true value
        assert abs(offset_est - true_offset) < 0.1


class TestEstimatePerAngleScaling:
    """Tests for estimate_per_angle_scaling function."""

    def test_multi_angle_estimation(self):
        """Verify per-angle estimation with different parameters per angle."""
        np.random.seed(456)
        n_points = 6000
        n_phi = 3

        # True parameters for each angle
        true_contrasts = [0.2, 0.4, 0.6]
        true_offsets = [0.7, 0.9, 1.1]

        # Create pooled data
        phi_indices = np.repeat(np.arange(n_phi), n_points // n_phi)
        t1 = np.random.uniform(0.1, 100, len(phi_indices))
        t2 = np.random.uniform(0.1, 100, len(phi_indices))

        c2_data = np.zeros(len(phi_indices))
        for i in range(n_phi):
            mask = phi_indices == i
            delta_t = np.abs(t1[mask] - t2[mask])
            g1_sq = np.exp(-0.1 * delta_t)
            c2_data[mask] = (
                true_contrasts[i] * g1_sq
                + true_offsets[i]
                + np.random.normal(0, 0.02, np.sum(mask))
            )

        estimates = estimate_per_angle_scaling(
            c2_data,
            t1,
            t2,
            phi_indices,
            n_phi,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Verify all angles have estimates
        for i in range(n_phi):
            assert f"contrast_{i}" in estimates
            assert f"offset_{i}" in estimates

            # Should be within 20% of true values
            assert (
                abs(estimates[f"contrast_{i}"] - true_contrasts[i])
                < 0.2 * true_contrasts[i] + 0.05
            )
            assert (
                abs(estimates[f"offset_{i}"] - true_offsets[i])
                < 0.2 * true_offsets[i] + 0.05
            )

    def test_insufficient_angle_data(self):
        """Verify fallback when angle has too few data points."""
        n_phi = 2
        phi_indices = np.array([0, 0, 0, 1] * 25)  # Only 25 points for angle 1
        t1 = np.random.uniform(0, 10, len(phi_indices))
        t2 = np.random.uniform(0, 10, len(phi_indices))
        c2_data = np.random.uniform(0.8, 1.2, len(phi_indices))

        contrast_bounds = (0.0, 1.0)
        offset_bounds = (0.5, 1.5)

        estimates = estimate_per_angle_scaling(
            c2_data, t1, t2, phi_indices, n_phi, contrast_bounds, offset_bounds
        )

        # Angle 1 should use midpoint fallback (only 25 points)
        expected_contrast = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        expected_offset = (offset_bounds[0] + offset_bounds[1]) / 2.0

        assert estimates["contrast_1"] == expected_contrast
        assert estimates["offset_1"] == expected_offset


class TestBuildInitValuesDictWithData:
    """Tests for build_init_values_dict with data-driven estimation."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a minimal parameter space for testing."""

        class MockParameterSpace:
            def get_bounds(self, name):
                bounds_map = {
                    "contrast": (0.0, 1.0),
                    "offset": (0.5, 1.5),
                    "D0": (1e3, 1e6),
                    "alpha": (-2.0, 0.0),
                    "D_offset": (0.0, 1e4),
                }
                return bounds_map.get(name, (0.0, 1.0))

        return MockParameterSpace()

    def test_uses_data_estimation_when_contrast_offset_missing(
        self, mock_parameter_space
    ):
        """Verify data-driven estimation is used when contrast/offset not in initial_values."""
        np.random.seed(789)
        n_points = 2000

        # Create synthetic data
        t1 = np.random.uniform(0.1, 100, n_points)
        t2 = np.random.uniform(0.1, 100, n_points)
        phi_indices = np.zeros(n_points, dtype=int)
        c2_data = (
            0.3 * np.exp(-0.1 * np.abs(t1 - t2))
            + 0.85
            + np.random.normal(0, 0.02, n_points)
        )

        # initial_values with only physical params (no contrast/offset)
        initial_values = {"D0": 1e4, "alpha": -0.5, "D_offset": 500.0}

        result = build_init_values_dict(
            n_phi=1,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
            c2_data=c2_data,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
        )

        # contrast_0 should NOT be midpoint (0.5) - should be estimated from data
        assert "contrast_0" in result
        # Midpoint would be 0.5, data-driven should be ~0.3
        assert result["contrast_0"] != 0.5  # Not the midpoint fallback

    def test_explicit_values_override_data_estimation(self, mock_parameter_space):
        """Verify explicit initial_values take precedence over data estimation."""
        np.random.seed(111)
        n_points = 2000

        t1 = np.random.uniform(0.1, 100, n_points)
        t2 = np.random.uniform(0.1, 100, n_points)
        phi_indices = np.zeros(n_points, dtype=int)
        c2_data = (
            0.3 * np.exp(-0.1 * np.abs(t1 - t2))
            + 0.85
            + np.random.normal(0, 0.02, n_points)
        )

        # Explicit contrast/offset in initial_values
        explicit_contrast = 0.42
        explicit_offset = 1.05
        initial_values = {
            "contrast": explicit_contrast,
            "offset": explicit_offset,
            "D0": 1e4,
            "alpha": -0.5,
            "D_offset": 500.0,
        }

        result = build_init_values_dict(
            n_phi=1,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
            c2_data=c2_data,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
        )

        # Should use explicit values, not data-driven
        assert result["contrast_0"] == pytest.approx(explicit_contrast, abs=0.01)
        assert result["offset_0"] == pytest.approx(explicit_offset, abs=0.01)

    def test_backward_compatible_without_data(self, mock_parameter_space):
        """Verify function works without data arrays (backward compatibility)."""
        initial_values = {"D0": 1e4, "alpha": -0.5, "D_offset": 500.0}

        result = build_init_values_dict(
            n_phi=2,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
            # No data arrays provided
        )

        # Should use midpoint fallback for contrast/offset
        assert result["contrast_0"] == 0.5
        assert result["contrast_1"] == 0.5
        assert result["offset_0"] == 1.0
        assert result["offset_1"] == 1.0
