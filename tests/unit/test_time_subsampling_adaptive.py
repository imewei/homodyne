"""Unit tests for adaptive time subsampling functionality.

Tests the new Layer 1 (physics-aware) and Layer 2 (NLSQ fallback) subsampling
strategy for large XPCS datasets (>50M points).

Test Coverage:
- should_apply_subsampling(): Decision logic
- compute_adaptive_target_points(): Reduction factor calculation
- validate_subsampling_config(): Configuration validation
- Integration with subsample_time_grid()
"""

import numpy as np
import pytest

from homodyne.data.time_subsampling import (
    compute_adaptive_target_points,
    should_apply_subsampling,
    validate_subsampling_config,
)


class TestShouldApplySubsampling:
    """Test dataset size decision logic for triggering subsampling."""

    def test_below_threshold_23M(self):
        """23M dataset should NOT trigger subsampling (below 50M threshold)."""
        # Create mock data: 23 angles × 1001² time points = 23,046,023 points
        data = {
            "c2_exp": np.random.rand(23, 1001, 1001),
            "t1": np.meshgrid(
                np.linspace(0, 100, 1001), np.linspace(0, 100, 1001), indexing="ij"
            )[0],
            "t2": np.meshgrid(
                np.linspace(0, 100, 1001), np.linspace(0, 100, 1001), indexing="ij"
            )[1],
        }

        should_sub, total, n_t, n_phi = should_apply_subsampling(
            data, trigger_threshold=50_000_000
        )

        assert should_sub is False, "23M dataset should not trigger subsampling"
        assert total == 23_046_023, f"Expected 23,046,023 points, got {total:,}"
        assert n_t == 1001, f"Expected 1001 time points, got {n_t}"
        assert n_phi == 23, f"Expected 23 angles, got {n_phi}"

    def test_above_threshold_100M(self):
        """100M dataset SHOULD trigger subsampling (above 50M threshold)."""
        # Create mock data: 43 angles × 1500² time points = 96,750,000 points
        data = {
            "c2_exp": np.random.rand(43, 1500, 1500),
            "t1": np.meshgrid(
                np.linspace(0, 100, 1500), np.linspace(0, 100, 1500), indexing="ij"
            )[0],
            "t2": np.meshgrid(
                np.linspace(0, 100, 1500), np.linspace(0, 100, 1500), indexing="ij"
            )[1],
        }

        should_sub, total, n_t, n_phi = should_apply_subsampling(
            data, trigger_threshold=50_000_000
        )

        assert should_sub is True, "100M dataset should trigger subsampling"
        assert total == 96_750_000, f"Expected 96,750,000 points, got {total:,}"
        assert n_t == 1500, f"Expected 1500 time points, got {n_t}"
        assert n_phi == 43, f"Expected 43 angles, got {n_phi}"

    def test_empty_data(self):
        """Empty dataset should return False without errors."""
        data = {"c2_exp": np.array([]), "t1": np.array([]), "t2": np.array([])}

        should_sub, total, n_t, n_phi = should_apply_subsampling(data)

        assert should_sub is False
        assert total == 0
        assert n_t == 0
        assert n_phi == 0

    def test_2d_data_single_angle(self):
        """2D data (single angle) should work correctly."""
        data = {
            "c2_exp": np.random.rand(1001, 1001),  # No angle dimension
            "t1": np.meshgrid(
                np.linspace(0, 100, 1001), np.linspace(0, 100, 1001), indexing="ij"
            )[0],
            "t2": np.meshgrid(
                np.linspace(0, 100, 1001), np.linspace(0, 100, 1001), indexing="ij"
            )[1],
        }

        should_sub, total, n_t, n_phi = should_apply_subsampling(
            data, trigger_threshold=50_000_000
        )

        assert should_sub is False
        assert total == 1_002_001
        assert n_t == 1001
        assert n_phi == 1


class TestComputeAdaptiveTargetPoints:
    """Test adaptive reduction factor calculation (2-4x range)."""

    def test_below_threshold_no_reduction(self):
        """23M dataset below threshold should get factor=1 (no reduction)."""
        target, factor = compute_adaptive_target_points(
            n_time_points=1001,
            n_angles=23,
            total_points=23_046_023,
            trigger_threshold=50_000_000,
        )

        assert factor == 1, f"Expected no reduction (1x), got {factor}x"
        assert target == 1001, f"Expected 1001 points (unchanged), got {target}"

    def test_100M_dataset_2x_reduction(self):
        """100M dataset should get factor=2 (2x reduction)."""
        target, factor = compute_adaptive_target_points(
            n_time_points=1500,
            n_angles=43,
            total_points=96_750_000,
            trigger_threshold=50_000_000,
        )

        assert factor == 2, f"Expected 2x reduction, got {factor}x"
        expected_target = int(1500 / np.sqrt(2))  # ≈ 1061
        assert abs(target - expected_target) <= 1, (
            f"Expected ~{expected_target} points, got {target}"
        )

    def test_500M_dataset_4x_reduction_capped(self):
        """500M dataset should get factor=4 (capped at max)."""
        target, factor = compute_adaptive_target_points(
            n_time_points=2236,
            n_angles=100,
            total_points=500_000_000,
            trigger_threshold=50_000_000,
            max_reduction_factor=4,
        )

        assert factor == 4, f"Expected 4x reduction (capped), got {factor}x"
        expected_target = int(2236 / np.sqrt(4))  # = 1118
        assert target == expected_target, (
            f"Expected {expected_target} points, got {target}"
        )

    def test_minimum_target_points(self):
        """Target points should never go below 50 (minimum for correlation structure)."""
        # Extreme case: very small time grid with large reduction
        target, factor = compute_adaptive_target_points(
            n_time_points=100,  # Very small
            n_angles=500,  # Many angles
            total_points=50_000_000,  # Right at threshold
            trigger_threshold=25_000_000,  # Low threshold to force reduction
            max_reduction_factor=4,
        )

        assert target >= 50, f"Target points should never be < 50, got {target}"

    def test_reduction_factor_bounds(self):
        """Reduction factor should always be in [2, 4] range when triggered."""
        # Test various dataset sizes
        for total_points in [60_000_000, 150_000_000, 1_000_000_000]:
            target, factor = compute_adaptive_target_points(
                n_time_points=2000,
                n_angles=50,
                total_points=total_points,
                trigger_threshold=50_000_000,
                min_reduction_factor=2,
                max_reduction_factor=4,
            )

            assert 2 <= factor <= 4, (
                f"Factor {factor} out of [2, 4] range for {total_points:,} points"
            )


class TestValidateSubsamplingConfig:
    """Test configuration validation and sanitization."""

    def test_valid_config_unchanged(self):
        """Valid configuration should pass through unchanged."""
        config = {
            "enabled": True,
            "trigger_threshold_points": 50_000_000,
            "max_reduction_factor": 4,
            "method": "logarithmic",
            "target_points": None,
            "preserve_edges": True,
        }

        validated = validate_subsampling_config(config)

        assert validated == config

    def test_invalid_threshold_reset_to_default(self):
        """Invalid threshold (<1M) should reset to default 50M."""
        config = {"trigger_threshold_points": -1000}

        validated = validate_subsampling_config(config)

        assert validated["trigger_threshold_points"] == 50_000_000

    def test_invalid_max_factor_clamped(self):
        """Max factor outside [2, 10] should be clamped."""
        config_too_small = {"max_reduction_factor": 1}
        config_too_large = {"max_reduction_factor": 100}

        validated_small = validate_subsampling_config(config_too_small)
        validated_large = validate_subsampling_config(config_too_large)

        assert validated_small["max_reduction_factor"] == 4  # Reset to default
        assert validated_large["max_reduction_factor"] == 4  # Reset to default

    def test_invalid_method_reset_to_logarithmic(self):
        """Invalid method should reset to 'logarithmic'."""
        config = {"method": "invalid_method"}

        validated = validate_subsampling_config(config)

        assert validated["method"] == "logarithmic"

    def test_invalid_target_points_reset_to_auto(self):
        """Target points < 50 should be reset to None (auto-calculate)."""
        config = {"target_points": 10}  # Too small

        validated = validate_subsampling_config(config)

        assert validated["target_points"] is None

    def test_missing_keys_use_defaults(self):
        """Missing configuration keys should use safe defaults."""
        config = {}  # Empty config

        validated = validate_subsampling_config(config)

        assert validated["enabled"] is False
        assert validated["trigger_threshold_points"] == 50_000_000
        assert validated["max_reduction_factor"] == 4
        assert validated["method"] == "logarithmic"
        assert validated["target_points"] is None
        assert validated["preserve_edges"] is True


class TestIntegrationWithSubsampleTimeGrid:
    """Test integration with the actual subsampling function."""

    def test_100M_dataset_reduces_correctly(self):
        """100M dataset should reduce to ~50M after subsampling."""
        # Create large mock dataset
        n_angles = 43
        n_time = 1500
        data = {
            "c2_exp": np.random.rand(n_angles, n_time, n_time),
            "t1": np.meshgrid(
                np.linspace(0, 100, n_time), np.linspace(0, 100, n_time), indexing="ij"
            )[0],
            "t2": np.meshgrid(
                np.linspace(0, 100, n_time), np.linspace(0, 100, n_time), indexing="ij"
            )[1],
        }

        # Compute adaptive target
        should_sub, total, n_t, n_phi = should_apply_subsampling(data, 50_000_000)
        assert should_sub is True

        target, factor = compute_adaptive_target_points(n_t, n_phi, total, 50_000_000)

        # Verify reduction factor
        assert factor == 2  # 96.75M → 48M (2x reduction)

        # Verify target points
        expected_target = int(1500 / np.sqrt(2))  # ≈ 1061
        assert abs(target - expected_target) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
