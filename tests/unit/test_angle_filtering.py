"""
Unit tests for angle filtering functionality.

Tests the _apply_angle_filtering_for_optimization() function that filters
phi angles before optimization, as well as the shared _apply_angle_filtering()
core logic.
"""

import numpy as np
import pytest

from homodyne.cli.commands import (
    _apply_angle_filtering,
    _apply_angle_filtering_for_optimization,
)
from tests.factories.config_factory import (
    create_anisotropic_filtering_config,
    create_disabled_filtering_config,
    create_empty_ranges_config,
    create_laminar_flow_filtering_config,
    create_non_matching_ranges_config,
    create_overlapping_ranges_config,
    create_phi_filtering_config,
    create_single_angle_range_config,
)
from tests.factories.data_factory import create_specific_angles_test_data


class TestApplyAngleFiltering:
    """Test the core _apply_angle_filtering() function."""

    def test_filtering_with_two_ranges(self):
        """Test filtering with two angle ranges (anisotropic case)."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 120.0, 180.0]
        data = create_specific_angles_test_data(angles, n_t1=20, n_t2=20)
        config = create_anisotropic_filtering_config()
        # Config has ranges: [-10, 10] and [80, 100]

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # Should match: 0.0, 10.0 (in [-10, 10]) and 85.0, 90.0 (in [80, 100])
        expected_indices = [0, 1, 5, 6]
        expected_angles = [0.0, 10.0, 85.0, 90.0]
        assert filtered_indices == expected_indices
        assert list(filtered_phi) == pytest.approx(expected_angles)
        assert filtered_c2.shape == (4, 20, 20)

    def test_filtering_with_eight_ranges(self):
        """Test filtering with eight angle ranges (laminar flow case)."""
        # Arrange
        angles = np.linspace(0, 360, 36, endpoint=False)  # 36 angles
        data = create_specific_angles_test_data(angles.tolist(), n_t1=15, n_t2=15)
        config = create_laminar_flow_filtering_config()
        # 8 ranges covering flow directions

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # Should select angles in the 8 ranges
        assert len(filtered_indices) > 0
        assert len(filtered_indices) < 36  # Not all angles
        assert filtered_c2.shape[0] == len(filtered_indices)

    def test_filtering_disabled_returns_all_angles(self):
        """Test that disabled filtering returns all angles."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_disabled_filtering_config()

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        assert filtered_indices == list(range(4))
        assert list(filtered_phi) == pytest.approx(angles)
        assert filtered_c2.shape == (4, 10, 10)

    def test_empty_target_ranges_returns_all_angles(self):
        """Test that empty target_ranges returns all angles."""
        # Arrange
        angles = [0.0, 45.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_empty_ranges_config()

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        assert filtered_indices == [0, 1, 2]
        assert list(filtered_phi) == pytest.approx(angles)

    def test_no_matching_angles_returns_all_angles(self):
        """Test that no matching angles returns all angles (fallback)."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        # Config with ranges that don't match [0, 30, 60, 90]
        config = create_non_matching_ranges_config(angles)

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # Should return all angles as fallback
        assert filtered_indices == [0, 1, 2, 3]
        assert list(filtered_phi) == pytest.approx(angles)

    def test_overlapping_ranges_no_duplicates(self):
        """Test that overlapping ranges don't create duplicate indices."""
        # Arrange
        angles = [5.0, 15.0, 25.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_overlapping_ranges_config()
        # Ranges: [0, 20] and [10, 30] - both match 15.0

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # All three angles match, but each only once
        assert filtered_indices == [0, 1, 2]
        assert len(set(filtered_indices)) == len(filtered_indices)  # No duplicates

    def test_single_angle_range(self):
        """Test filtering with very narrow range (single angle)."""
        # Arrange
        angles = [0.0, 45.0, 90.0, 135.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_single_angle_range_config(angle=90.0)

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # Should match only 90.0 (within [89.5, 90.5])
        assert filtered_indices == [2]
        assert list(filtered_phi) == pytest.approx([90.0])

    def test_inclusive_bounds_lower(self):
        """Test that lower bound is inclusive."""
        # Arrange
        angles = [10.0, 20.0, 30.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": 10.0, "max_angle": 20.0}]
        )

        # Act
        filtered_indices, _, _ = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # 10.0 should be included (min_angle <= 10.0)
        assert 0 in filtered_indices

    def test_inclusive_bounds_upper(self):
        """Test that upper bound is inclusive."""
        # Arrange
        angles = [10.0, 20.0, 30.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": 10.0, "max_angle": 20.0}]
        )

        # Act
        filtered_indices, _, _ = _apply_angle_filtering(
            data["phi_angles_list"], data["c2_exp"], config
        )

        # Assert
        # 20.0 should be included (20.0 <= max_angle)
        assert 1 in filtered_indices


class TestApplyAngleFilteringForOptimization:
    """Test the _apply_angle_filtering_for_optimization() wrapper function."""

    def test_enabled_filtering_with_matching_angles(self):
        """Test filtering enabled with angles that match ranges."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=20, n_t2=20)
        config = create_anisotropic_filtering_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Should filter to angles near 0 and 90
        assert len(filtered_data["phi_angles_list"]) == 3  # [0, 10, 90]
        assert filtered_data["c2_exp"].shape == (3, 20, 20)
        # Other keys preserved
        assert "wavevector_q_list" in filtered_data
        assert "t1" in filtered_data
        assert "t2" in filtered_data

    def test_disabled_filtering_returns_original(self):
        """Test that disabled filtering returns original data unchanged."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=15, n_t2=15)
        config = create_disabled_filtering_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        assert len(filtered_data["phi_angles_list"]) == 4
        assert filtered_data["c2_exp"].shape == (4, 15, 15)
        np.testing.assert_array_equal(
            filtered_data["phi_angles_list"], data["phi_angles_list"]
        )

    def test_empty_target_ranges_returns_original(self):
        """Test that empty target_ranges returns original data with warning."""
        # Arrange
        angles = [0.0, 45.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_empty_ranges_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        assert len(filtered_data["phi_angles_list"]) == 3
        np.testing.assert_array_equal(
            filtered_data["phi_angles_list"], data["phi_angles_list"]
        )

    def test_no_matching_angles_returns_original(self):
        """Test that no matching angles returns original data with warning."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_non_matching_ranges_config(angles)

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Fallback: returns all angles
        assert len(filtered_data["phi_angles_list"]) == 4

    def test_data_dict_structure_preserved(self):
        """Test that filtered data dict preserves all required keys."""
        # Arrange
        angles = [0.0, 10.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        # Add extra key to verify it's preserved
        data["extra_key"] = "test_value"
        config = create_anisotropic_filtering_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Required keys present
        assert "phi_angles_list" in filtered_data
        assert "c2_exp" in filtered_data
        assert "wavevector_q_list" in filtered_data
        assert "t1" in filtered_data
        assert "t2" in filtered_data
        # Extra key preserved
        assert "extra_key" in filtered_data
        assert filtered_data["extra_key"] == "test_value"

    def test_c2_exp_first_dimension_correctly_sliced(self):
        """Test that C2 data first dimension is sliced correctly."""
        # Arrange
        angles = np.linspace(0, 360, 10, endpoint=False)
        data = create_specific_angles_test_data(angles.tolist(), n_t1=25, n_t2=25)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": 0.0, "max_angle": 50.0}]
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        n_filtered = len(filtered_data["phi_angles_list"])
        assert filtered_data["c2_exp"].shape[0] == n_filtered
        assert filtered_data["c2_exp"].shape[1] == 25  # t1 unchanged
        assert filtered_data["c2_exp"].shape[2] == 25  # t2 unchanged

    def test_wavevector_t1_t2_unchanged(self):
        """Test that wavevector_q_list, t1, t2 remain unchanged."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=20, n_t2=20)
        original_q = data["wavevector_q_list"].copy()
        original_t1 = data["t1"].copy()
        original_t2 = data["t2"].copy()
        config = create_anisotropic_filtering_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        np.testing.assert_array_equal(filtered_data["wavevector_q_list"], original_q)
        np.testing.assert_array_equal(filtered_data["t1"], original_t1)
        np.testing.assert_array_equal(filtered_data["t2"], original_t2)

    def test_config_manager_compatibility(self):
        """Test that function works with ConfigManager objects."""
        # Arrange

        angles = [0.0, 45.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)

        # Create a mock ConfigManager-like object
        class MockConfigManager:
            def __init__(self, config_dict):
                self._config = config_dict

            def get_config(self):
                return self._config

        config_dict = create_anisotropic_filtering_config()
        config_manager = MockConfigManager(config_dict)

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config_manager)

        # Assert
        # Should work with ConfigManager objects
        assert len(filtered_data["phi_angles_list"]) <= 3

    def test_all_angles_matched_returns_original(self):
        """Test that when all angles match, original data is returned."""
        # Arrange
        angles = [0.0, 30.0, 60.0, 90.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": -10.0, "max_angle": 100.0}]
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # All 4 angles match the range, so original data returned
        assert len(filtered_data["phi_angles_list"]) == 4
        np.testing.assert_array_equal(
            filtered_data["phi_angles_list"], data["phi_angles_list"]
        )


class TestAngleFilteringEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_phi_angles_array(self):
        """Test handling of empty phi angles array."""
        # Arrange
        data = {
            "phi_angles_list": np.array([]),
            "c2_exp": np.array([]),
            "wavevector_q_list": np.array([0.01]),
            "t1": np.linspace(0, 1, 10),
            "t2": np.linspace(0, 1, 10),
        }
        config = create_anisotropic_filtering_config()

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Should return original data with warning
        assert len(filtered_data["phi_angles_list"]) == 0

    def test_negative_angles(self):
        """Test filtering with negative angle values."""
        # Arrange
        angles = [-10.0, -5.0, 0.0, 5.0, 10.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": -10.0, "max_angle": 0.0}]
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Should match -10.0, -5.0, 0.0
        assert len(filtered_data["phi_angles_list"]) == 3
        expected = [-10.0, -5.0, 0.0]
        assert list(filtered_data["phi_angles_list"]) == pytest.approx(expected)

    def test_angles_exceeding_360_degrees(self):
        """Test filtering with angles > 360 degrees."""
        # Arrange
        # Angles: [0, 180, 360, 400] → normalized to [0, 180, 0, 40]
        angles = [0.0, 180.0, 360.0, 400.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)

        # Range [350, 410] → [-10, 50] after normalization
        # This spans across 0°, so it matches: 0° (from both 0 and 360), and 40° (from 400)
        config = create_phi_filtering_config(
            enabled=True, target_ranges=[{"min_angle": 350.0, "max_angle": 410.0}]
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert
        # Should match angles at indices 0 (0°), 2 (360°→0°), and 3 (400°→40°)
        assert len(filtered_data["phi_angles_list"]) == 3
        # Check that the matched angles are correct
        assert 0.0 in filtered_data["phi_angles_list"]  # Both 0° and 360° normalize to 0°
        assert 40.0 in filtered_data["phi_angles_list"]  # 400° normalizes to 40°


class TestAngleNormalization:
    """Tests for angle normalization to [-180°, 180°] range."""

    def test_normalize_basic_positive_angles(self):
        """Test normalization of angles in (180°, 360°) range."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange & Act
        result_210 = normalize_angle_to_symmetric_range(210.0)
        result_270 = normalize_angle_to_symmetric_range(270.0)
        result_350 = normalize_angle_to_symmetric_range(350.0)

        # Assert
        assert result_210 == pytest.approx(-150.0)
        assert result_270 == pytest.approx(-90.0)
        assert result_350 == pytest.approx(-10.0)

    def test_normalize_basic_negative_angles(self):
        """Test normalization of angles in (-360°, -180°) range."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange & Act
        result_minus_210 = normalize_angle_to_symmetric_range(-210.0)
        result_minus_270 = normalize_angle_to_symmetric_range(-270.0)
        result_minus_350 = normalize_angle_to_symmetric_range(-350.0)

        # Assert
        assert result_minus_210 == pytest.approx(150.0)
        assert result_minus_270 == pytest.approx(90.0)
        assert result_minus_350 == pytest.approx(10.0)

    def test_normalize_already_in_range(self):
        """Test that angles already in [-180°, 180°] remain unchanged."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange & Act & Assert
        # Note: -180° and 180° are equivalent and both normalize to 180°
        assert normalize_angle_to_symmetric_range(0.0) == pytest.approx(0.0)
        assert normalize_angle_to_symmetric_range(45.0) == pytest.approx(45.0)
        assert normalize_angle_to_symmetric_range(90.0) == pytest.approx(90.0)
        assert normalize_angle_to_symmetric_range(-45.0) == pytest.approx(-45.0)
        assert normalize_angle_to_symmetric_range(-90.0) == pytest.approx(-90.0)
        assert normalize_angle_to_symmetric_range(180.0) == pytest.approx(180.0)
        assert normalize_angle_to_symmetric_range(-180.0) == pytest.approx(180.0)  # -180° → 180°

    def test_normalize_boundary_cases(self):
        """Test normalization at boundaries (±180°, 0°, 360°)."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange & Act
        result_0 = normalize_angle_to_symmetric_range(0.0)
        result_180 = normalize_angle_to_symmetric_range(180.0)
        result_minus_180 = normalize_angle_to_symmetric_range(-180.0)
        result_360 = normalize_angle_to_symmetric_range(360.0)
        result_minus_360 = normalize_angle_to_symmetric_range(-360.0)

        # Assert
        # Note: -180° and 180° are mathematically equivalent angles
        assert result_0 == pytest.approx(0.0)
        assert result_180 == pytest.approx(180.0)
        assert result_minus_180 == pytest.approx(180.0)  # -180° → 180° (equivalent)
        assert result_360 == pytest.approx(0.0)  # 360° wraps to 0°
        assert result_minus_360 == pytest.approx(0.0)  # -360° wraps to 0°

    def test_normalize_multiple_wraps(self):
        """Test normalization with multiple 360° wraps."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange & Act
        result_540 = normalize_angle_to_symmetric_range(540.0)  # 360° + 180°
        result_minus_540 = normalize_angle_to_symmetric_range(-540.0)  # -360° - 180°
        result_900 = normalize_angle_to_symmetric_range(900.0)  # 2.5 × 360°

        # Assert
        # Note: -180° and 180° are equivalent and both normalize to 180°
        assert result_540 == pytest.approx(180.0)
        assert result_minus_540 == pytest.approx(180.0)  # -540° → 180° (equivalent to -180°)
        assert result_900 == pytest.approx(180.0)

    def test_normalize_array_input(self):
        """Test normalization with NumPy array input."""
        import numpy as np
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange
        angles = np.array([0, 90, 180, 210, -210, 270, 360])

        # Act
        result = normalize_angle_to_symmetric_range(angles)

        # Assert
        expected = np.array([0, 90, 180, -150, 150, -90, 0])
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_normalize_scalar_returns_scalar(self):
        """Test that scalar input returns scalar (not 0-d array)."""
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange
        angle = 210.0

        # Act
        result = normalize_angle_to_symmetric_range(angle)

        # Assert
        assert isinstance(result, float)
        assert not isinstance(result, np.ndarray)
        assert result == pytest.approx(-150.0)

    def test_normalize_array_returns_array(self):
        """Test that array input returns array."""
        import numpy as np
        from homodyne.cli.commands import normalize_angle_to_symmetric_range

        # Arrange
        angles = np.array([210.0, -210.0])

        # Act
        result = normalize_angle_to_symmetric_range(angles)

        # Assert
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestAngleInRange:
    """Tests for wrap-aware angle range checking."""

    def test_normal_range_inside(self):
        """Test angle inside normal range (min <= max)."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 90.0
        min_angle = 85.0
        max_angle = 95.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_normal_range_outside(self):
        """Test angle outside normal range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 100.0
        min_angle = 85.0
        max_angle = 95.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is False

    def test_normal_range_boundary_min(self):
        """Test angle at minimum boundary of normal range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 85.0
        min_angle = 85.0
        max_angle = 95.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_normal_range_boundary_max(self):
        """Test angle at maximum boundary of normal range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 95.0
        min_angle = 85.0
        max_angle = 95.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_wrapped_range_upper_section(self):
        """Test angle in upper section of wrapped range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        # Range [170°, -170°] represents angles >= 170° OR <= -170°
        angle = 175.0
        min_angle = 170.0
        max_angle = -170.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_wrapped_range_lower_section(self):
        """Test angle in lower section of wrapped range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = -175.0
        min_angle = 170.0
        max_angle = -170.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_wrapped_range_outside(self):
        """Test angle outside wrapped range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 0.0
        min_angle = 170.0
        max_angle = -170.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is False

    def test_wrapped_range_boundary_upper(self):
        """Test angle at upper boundary of wrapped range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = 170.0
        min_angle = 170.0
        max_angle = -170.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True

    def test_wrapped_range_boundary_lower(self):
        """Test angle at lower boundary of wrapped range."""
        from homodyne.cli.commands import _angle_in_range

        # Arrange
        angle = -170.0
        min_angle = 170.0
        max_angle = -170.0

        # Act
        result = _angle_in_range(angle, min_angle, max_angle)

        # Assert
        assert result is True


class TestNormalizationIntegration:
    """Integration tests for angle normalization in filtering workflow."""

    def test_filtering_with_angles_above_180(self):
        """Test that angles > 180° are normalized and filtered correctly."""
        # Arrange
        # Angles: [0, 90, 210, 270] → normalized to [0, 90, -150, -90]
        angles = [0.0, 90.0, 210.0, 270.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)

        # Filter for angles near -90° (i.e., 270° before normalization)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -100.0, "max_angle": -80.0}],
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Should match 270° → -90°
        assert len(filtered_data["phi_angles_list"]) == 1
        assert filtered_data["phi_angles_list"][0] == pytest.approx(-90.0)

    def test_filtering_with_wrapped_range(self):
        """Test filtering with range that spans ±180° boundary."""
        # Arrange
        # Angles: [0, 90, 175, 185] → normalized to [0, 90, 175, -175]
        angles = [0.0, 90.0, 175.0, 185.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)

        # Range [170, 190] → [170, -170] after normalization (wrapped range)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": 170.0, "max_angle": 190.0}],
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Should match both 175° and 185°→-175°
        assert len(filtered_data["phi_angles_list"]) == 2
        assert 175.0 in filtered_data["phi_angles_list"]
        assert -175.0 in filtered_data["phi_angles_list"]

    def test_normalization_preserves_data_structure(self):
        """Test that normalization preserves c2_exp and other data."""
        # Arrange
        angles = [0.0, 210.0, 270.0]  # 210→-150, 270→-90
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -160.0, "max_angle": -140.0}],
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Should match 210° → -150°
        assert len(filtered_data["phi_angles_list"]) == 1
        assert filtered_data["phi_angles_list"][0] == pytest.approx(-150.0)
        assert filtered_data["c2_exp"].shape[0] == 1
        assert "t1" in filtered_data
        assert "t2" in filtered_data

    def test_all_angles_normalized_to_symmetric_range(self):
        """Test that all output angles are in [-180°, 180°] range."""
        # Arrange
        angles = [0.0, 90.0, 180.0, 210.0, 270.0, 350.0]
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -180.0, "max_angle": 180.0}],
        )

        # Act
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - All angles should be in [-180, 180]
        for angle in filtered_data["phi_angles_list"]:
            assert -180.0 <= angle <= 180.0


class TestAngleValidation:
    """Tests for angle validation warnings (|φ| > 360°)."""

    def test_warning_for_angles_exceeding_360(self, caplog):
        """Test that angles > 360° trigger warning."""
        # Arrange
        angles = [0.0, 90.0, 400.0, 500.0]  # Two angles > 360°
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -180.0, "max_angle": 180.0}],
        )

        # Act
        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Warning logged
        log_messages = [rec.message for rec in caplog.records]
        warning_found = any(
            "|φ| > 360°" in msg and "400" in msg and "500" in msg
            for msg in log_messages
        )
        assert warning_found, "Should log warning for angles > 360°"

        # Assert - Angles are normalized correctly
        assert 40.0 in filtered_data["phi_angles_list"]  # 400° → 40°
        assert 140.0 in filtered_data["phi_angles_list"]  # 500° → 140°

    def test_warning_for_angles_below_minus_360(self, caplog):
        """Test that angles < -360° trigger warning."""
        # Arrange
        angles = [0.0, 90.0, -400.0, -500.0]  # Two angles < -360°
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -180.0, "max_angle": 180.0}],
        )

        # Act
        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Warning logged
        log_messages = [rec.message for rec in caplog.records]
        warning_found = any(
            "|φ| > 360°" in msg and "-400" in msg and "-500" in msg
            for msg in log_messages
        )
        assert warning_found, "Should log warning for angles < -360°"

        # Assert - Angles are normalized correctly
        assert -40.0 in filtered_data["phi_angles_list"]  # -400° → -40°
        assert -140.0 in filtered_data["phi_angles_list"]  # -500° → -140°

    def test_no_warning_for_normal_angles(self, caplog):
        """Test that normal angles don't trigger warning."""
        # Arrange
        angles = [0.0, 90.0, 180.0, 270.0, -90.0, -180.0]  # All |φ| <= 360°
        data = create_specific_angles_test_data(angles, n_t1=10, n_t2=10)
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -180.0, "max_angle": 180.0}],
        )

        # Act
        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - No warning logged
        log_messages = [rec.message for rec in caplog.records]
        warning_found = any("|φ| > 360°" in msg for msg in log_messages)
        assert not warning_found, "Should not log warning for normal angles"
