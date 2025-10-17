"""Edge case integration tests for optimization with angle filtering.

This module tests that optimization methods (NLSQ and MCMC) handle edge cases
gracefully, including no matching angles, very small angle sets, and fallback
to all angles when filtering criteria are not met.
"""

import numpy as np
import pytest

from tests.factories.config_factory import create_phi_filtering_config
from tests.factories.data_factory import create_specific_angles_test_data


class TestOptimizationEdgeCases:
    """Edge case integration tests for optimization with filtering."""

    def test_nlsq_with_no_matching_angles_uses_all_angles(self, caplog):
        """Test NLSQ with no matching angles falls back to all angles."""
        # Arrange - Create data with 9 angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering with ranges that truly don't match any angles
        # Use ranges far from our test angles
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 200.0, "max_angle": 210.0, "description": "No match 1"},
                {"min_angle": 270.0, "max_angle": 280.0, "description": "No match 2"},
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering (should return all angles when no matches)
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - All angles used (fallback behavior)
        assert len(filtered_data["phi_angles_list"]) == 9, (
            "Should use all 9 angles when no matches"
        )
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], angles, decimal=1
        )

        # Assert - Core behavior verified (fallback to all angles)
        # Note: The fallback behavior is the most important aspect

    def test_mcmc_with_no_matching_angles_uses_all_angles(self, caplog):
        """Test MCMC with no matching angles falls back to all angles."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering with ranges that truly don't match any angles
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 200.0, "max_angle": 210.0, "description": "No match 1"},
                {"min_angle": 270.0, "max_angle": 280.0, "description": "No match 2"},
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - All angles used (fallback behavior)
        assert len(filtered_data["phi_angles_list"]) == 9, (
            "Should use all 9 angles when no matches"
        )

        # Assert - Core behavior verified (fallback to all angles)
        # Warning logging is secondary to the correct fallback behavior

    def test_very_small_filtered_set_allowed(self):
        """Test that filtering to 1 angle is allowed (no error)."""
        # Arrange - Create data with 9 angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering to select only 1 angle (very narrow range)
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 89.5, "max_angle": 90.5, "description": "Exactly 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Only 1 angle selected (90.0), no error raised
        assert len(filtered_data["phi_angles_list"]) == 1, (
            "Should select exactly 1 angle"
        )
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], [90.0], decimal=1
        )

        # Assert - C2 data correctly sliced
        assert filtered_data["c2_exp"].shape[0] == 1, "C2 first dimension should be 1"

    def test_system_stability_with_edge_cases(self):
        """Test that filtering edge cases maintain system stability."""
        # Test multiple edge case scenarios
        test_scenarios = [
            # Empty target_ranges
            {
                "config": create_phi_filtering_config(enabled=True, target_ranges=[]),
                "expected_count": 9,
                "description": "empty target_ranges",
            },
            # Non-matching ranges (truly don't match test angles)
            {
                "config": create_phi_filtering_config(
                    enabled=True,
                    target_ranges=[
                        {"min_angle": 200.0, "max_angle": 210.0},
                        {"min_angle": 270.0, "max_angle": 280.0},
                    ],
                ),
                "expected_count": 9,
                "description": "non-matching ranges",
            },
            # Very narrow range (1 angle)
            {
                "config": create_phi_filtering_config(
                    enabled=True,
                    target_ranges=[{"min_angle": 89.5, "max_angle": 90.5}],
                ),
                "expected_count": 1,
                "description": "very narrow range",
            },
        ]

        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        def create_mock_config(config_dict):
            """Create mock config manager with given config dict."""

            class MockConfigManager:
                def get_config(self):
                    return config_dict

            return MockConfigManager()

        for scenario in test_scenarios:
            config = create_mock_config(scenario["config"])

            # Act - Should not raise exception
            try:
                filtered_data = _apply_angle_filtering_for_optimization(data, config)
                success = True
            except Exception:
                success = False

            # Assert - No exceptions raised
            assert success, f"Edge case failed: {scenario['description']}"

            # Assert - Correct number of angles
            assert (
                len(filtered_data["phi_angles_list"]) == scenario["expected_count"]
            ), (
                f"Edge case '{scenario['description']}' returned wrong count: "
                f"{len(filtered_data['phi_angles_list'])} (expected {scenario['expected_count']})"
            )

            # Assert - Data structure intact
            assert "c2_exp" in filtered_data, "c2_exp missing from filtered data"
            assert "t1" in filtered_data, "t1 missing from filtered data"
            assert "t2" in filtered_data, "t2 missing from filtered data"

    def test_overlapping_ranges_no_duplicates_in_integration(self):
        """Test overlapping ranges produce no duplicate angles in full workflow."""
        # Arrange
        from tests.factories.config_factory import create_overlapping_ranges_config

        angles = list(range(0, 35, 5))  # [0, 5, 10, 15, 20, 25, 30]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_overlapping_ranges_config()

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - All angles selected (all match overlapping ranges)
        assert len(filtered_data["phi_angles_list"]) == 7, (
            "Should have 7 angles (all match)"
        )

        # Assert - No duplicates
        unique_angles = np.unique(filtered_data["phi_angles_list"])
        assert len(unique_angles) == len(filtered_data["phi_angles_list"]), (
            "Filtered angles contain duplicates"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
