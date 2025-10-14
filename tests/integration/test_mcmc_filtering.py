"""Integration tests for MCMC optimization with angle filtering.

This module validates that angle filtering works correctly in the MCMC
optimization workflow. Tests focus on data preparation and filtering,
not on MCMC convergence (which requires realistic data and is slow).
"""

import numpy as np
import pytest

from tests.factories.config_factory import (
    create_disabled_filtering_config,
    create_phi_filtering_config,
)
from tests.factories.data_factory import create_specific_angles_test_data


class TestMCMCWithAngleFiltering:
    """Integration tests for MCMC optimization with angle filtering."""

    def test_mcmc_receives_filtered_angles(self, caplog):
        """Test that MCMC receives correctly filtered angle data."""
        # Arrange - Create data with 9 specific angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering to select only [85, 90, 95] (3 angles)
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering before MCMC (simulating _run_optimization behavior)
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Dataset size reduction (filtering worked)
        assert len(filtered_data["phi_angles_list"]) == 3, "Should have 3 filtered angles"
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], [85.0, 90.0, 95.0], decimal=1
        )

        # Assert - C2 data first dimension reduced
        assert filtered_data["c2_exp"].shape[0] == 3, "C2 first dimension should be 3"

        # Assert - Other dimensions preserved (required for MCMC)
        assert "t1" in filtered_data, "t1 should be preserved for MCMC"
        assert "t2" in filtered_data, "t2 should be preserved for MCMC"

        # Assert - Log messages confirm filtering
        log_messages = [rec.message for rec in caplog.records]

        found_filtering_msg = any(
            "3 angles selected from 9" in msg for msg in log_messages
        )
        assert found_filtering_msg, "Should log '3 angles selected from 9 total angles'"

        # Verify MCMC would receive the correct filtered arrays
        mcmc_data = filtered_data["c2_exp"]
        mcmc_phi = filtered_data.get("phi_angles_list")
        mcmc_t1 = filtered_data.get("t1")
        mcmc_t2 = filtered_data.get("t2")

        assert mcmc_data.shape[0] == 3, "MCMC should receive 3 angles in c2_exp"
        assert len(mcmc_phi) == 3, "MCMC should receive 3 phi angles"
        assert mcmc_t1 is not None, "MCMC should receive t1 array"
        assert mcmc_t2 is not None, "MCMC should receive t2 array"

    def test_mcmc_with_disabled_filtering_uses_all_angles(self):
        """Test that MCMC uses all 9 angles when filtering is disabled."""
        # Arrange - Create data with 9 angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure with filtering disabled
        config_dict = create_disabled_filtering_config()

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Apply filtering (should return all angles when disabled)
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - All 9 angles used (no filtering)
        assert (
            len(filtered_data["phi_angles_list"]) == 9
        ), "Should use all 9 angles when disabled"
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], angles, decimal=1
        )

        # Verify MCMC would receive all angles
        mcmc_data = filtered_data["c2_exp"]
        mcmc_phi = filtered_data.get("phi_angles_list")

        assert mcmc_data.shape[0] == 9, "MCMC should receive all 9 angles"
        assert len(mcmc_phi) == 9, "MCMC should receive all 9 phi angles"

    def test_mcmc_dataset_size_reduction_verified(self):
        """Test that dataset size reduction is measurable (9 → 3 angles)."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        original_size = len(data["phi_angles_list"])
        original_c2_size = data["c2_exp"].shape[0]

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        filtered_size = len(filtered_data["phi_angles_list"])
        filtered_c2_size = filtered_data["c2_exp"].shape[0]

        # Assert - Size reduction
        assert original_size == 9, "Original should have 9 angles"
        assert filtered_size == 3, "Filtered should have 3 angles"
        assert original_c2_size == 9, "Original C2 should have 9 in first dimension"
        assert filtered_c2_size == 3, "Filtered C2 should have 3 in first dimension"

        # Calculate reduction
        reduction_factor = original_size / filtered_size
        assert reduction_factor == 3.0, "Should have 3x reduction (9 → 3)"

    def test_mcmc_log_messages_confirm_angle_selection(self, caplog):
        """Test that log messages confirm correct angle selection for MCMC."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        caplog.clear()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Check log messages
        log_messages = [rec.message for rec in caplog.records]

        # Should log "3 angles selected from 9 total angles"
        found_count_msg = any("3 angles selected from 9" in msg for msg in log_messages)
        assert found_count_msg, "Should log angle count: '3 angles selected from 9'"

        # Should log the selected angles: [85.0, 90.0, 95.0]
        found_angles_msg = any(
            "85" in msg and "90" in msg and "95" in msg for msg in log_messages
        )
        assert (
            found_angles_msg
        ), "Should log selected angles containing 85, 90, and 95"

    def test_mcmc_data_arrays_correctly_formatted(self):
        """Test that MCMC receives data arrays in correct format."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()

        # Act
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Extract data as MCMC would receive it (from CLI code)
        mcmc_data = filtered_data["c2_exp"]
        mcmc_t1 = filtered_data.get("t1")
        mcmc_t2 = filtered_data.get("t2")
        mcmc_phi = filtered_data.get("phi_angles_list")

        # Assert - Data types are NumPy arrays
        assert isinstance(mcmc_data, np.ndarray), "c2_exp should be NumPy array"
        assert isinstance(mcmc_t1, np.ndarray), "t1 should be NumPy array"
        assert isinstance(mcmc_t2, np.ndarray), "t2 should be NumPy array"
        assert isinstance(mcmc_phi, np.ndarray), "phi should be NumPy array"

        # Assert - Data shapes are correct
        assert mcmc_data.ndim == 3, "c2_exp should be 3D array (n_phi, n_t1, n_t2)"
        assert mcmc_data.shape[0] == 3, "c2_exp first dimension should be 3 angles"
        # t1 and t2 can be 1D or 2D (meshgrid) depending on data factory
        assert mcmc_t1.ndim in [1, 2], "t1 should be 1D or 2D array"
        assert mcmc_t2.ndim in [1, 2], "t2 should be 1D or 2D array"
        assert mcmc_phi.ndim == 1, "phi should be 1D array"
        assert len(mcmc_phi) == 3, "phi should have 3 angles"

        # Assert - Data values are reasonable
        assert np.all(np.isfinite(mcmc_data)), "c2_exp should have finite values"
        assert np.all(np.isfinite(mcmc_t1)), "t1 should have finite values"
        assert np.all(np.isfinite(mcmc_t2)), "t2 should have finite values"
        assert np.all(np.isfinite(mcmc_phi)), "phi should have finite values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
