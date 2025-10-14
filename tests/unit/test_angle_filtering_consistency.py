"""Consistency validation tests for angle filtering.

This module ensures that optimization and plotting use identical filtering logic
by testing that they return the same filtered results for identical inputs.
"""

import numpy as np
import pytest

from tests.factories.config_factory import (
    create_anisotropic_filtering_config,
    create_disabled_filtering_config,
    create_laminar_flow_filtering_config,
    create_single_angle_range_config,
)
from tests.factories.data_factory import (
    create_angle_filtering_test_data,
    create_specific_angles_test_data,
)


class TestAngleFilteringConsistency:
    """Test consistency between optimization and plotting filtering."""

    def test_optimization_and_plotting_identical_results_anisotropic(self):
        """Verify optimization and plotting return identical results (anisotropic)."""
        # Arrange
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 120.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=25, n_t2=25)
        config = create_anisotropic_filtering_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act - Core function
        core_indices, core_phi, core_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Act - Optimization wrapper
        # Mock ConfigManager
        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()
        opt_data = _apply_angle_filtering_for_optimization(data, mock_config)
        opt_phi = opt_data["phi_angles_list"]
        opt_c2 = opt_data["c2_exp"]

        # Act - Plotting wrapper
        data_with_config = data.copy()
        data_with_config["config"] = config
        plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
            phi_angles, c2_exp, data_with_config
        )

        # Assert - All three should return identical results
        np.testing.assert_array_equal(
            core_phi, opt_phi, err_msg="Core and optimization phi angles differ"
        )
        np.testing.assert_array_equal(
            core_phi, plot_phi, err_msg="Core and plotting phi angles differ"
        )
        np.testing.assert_array_equal(
            core_c2, opt_c2, err_msg="Core and optimization C2 data differ"
        )
        np.testing.assert_array_equal(
            core_c2, plot_c2, err_msg="Core and plotting C2 data differ"
        )
        assert core_indices == plot_indices, "Core and plotting indices differ"

    def test_optimization_and_plotting_identical_results_laminar_flow(self):
        """Verify optimization and plotting return identical results (laminar flow)."""
        # Arrange
        data = create_angle_filtering_test_data(n_phi=72, n_t1=50, n_t2=50)
        config = create_laminar_flow_filtering_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        core_indices, core_phi, core_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()
        opt_data = _apply_angle_filtering_for_optimization(data, mock_config)

        data_with_config = data.copy()
        data_with_config["config"] = config
        plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
            phi_angles, c2_exp, data_with_config
        )

        # Assert
        np.testing.assert_array_equal(core_phi, opt_data["phi_angles_list"])
        np.testing.assert_array_equal(core_phi, plot_phi)
        np.testing.assert_array_equal(core_c2, opt_data["c2_exp"])
        np.testing.assert_array_equal(core_c2, plot_c2)
        assert core_indices == plot_indices

    def test_consistency_with_disabled_filtering(self):
        """Verify all methods return original data when filtering disabled."""
        # Arrange
        data = create_angle_filtering_test_data(n_phi=20, n_t1=25, n_t2=25)
        config = create_disabled_filtering_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        core_indices, core_phi, core_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()
        opt_data = _apply_angle_filtering_for_optimization(data, mock_config)

        data_with_config = data.copy()
        data_with_config["config"] = config
        plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
            phi_angles, c2_exp, data_with_config
        )

        # Assert - All should return original unfiltered data
        np.testing.assert_array_equal(phi_angles, core_phi)
        np.testing.assert_array_equal(phi_angles, opt_data["phi_angles_list"])
        np.testing.assert_array_equal(phi_angles, plot_phi)

        np.testing.assert_array_equal(c2_exp, core_c2)
        np.testing.assert_array_equal(c2_exp, opt_data["c2_exp"])
        np.testing.assert_array_equal(c2_exp, plot_c2)

        # Indices should be full range
        assert core_indices == list(range(len(phi_angles)))
        assert plot_indices == list(range(len(phi_angles)))

    def test_consistency_with_single_angle_range(self):
        """Verify consistency with very narrow range (single angle)."""
        # Arrange
        angles = [0.0, 45.0, 89.5, 90.0, 90.5, 135.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=25, n_t2=25)
        config = create_single_angle_range_config(angle=90.0)

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        core_indices, core_phi, core_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()
        opt_data = _apply_angle_filtering_for_optimization(data, mock_config)

        data_with_config = data.copy()
        data_with_config["config"] = config
        plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
            phi_angles, c2_exp, data_with_config
        )

        # Assert - Should select [89.5, 90.0, 90.5] (range 89.5-90.5)
        expected_angles = np.array([89.5, 90.0, 90.5])
        np.testing.assert_array_equal(core_phi, expected_angles)
        np.testing.assert_array_equal(opt_data["phi_angles_list"], expected_angles)
        np.testing.assert_array_equal(plot_phi, expected_angles)

    def test_consistency_across_multiple_datasets(self):
        """Verify consistency holds across different dataset sizes."""
        # Arrange
        config = create_anisotropic_filtering_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        # Test with multiple dataset sizes
        for n_phi in [10, 50, 100, 200]:
            data = create_angle_filtering_test_data(
                n_phi=n_phi, n_t1=25, n_t2=25, phi_min=-10.0, phi_max=370.0
            )

            phi_angles = data["phi_angles_list"]
            c2_exp = data["c2_exp"]

            # Act
            core_indices, core_phi, core_c2 = _apply_angle_filtering(
                phi_angles, c2_exp, config
            )

            class MockConfigManager:
                def get_config(self):
                    return config

            mock_config = MockConfigManager()
            opt_data = _apply_angle_filtering_for_optimization(data, mock_config)

            data_with_config = data.copy()
            data_with_config["config"] = config
            plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
                phi_angles, c2_exp, data_with_config
            )

            # Assert - All should be identical for this dataset size
            np.testing.assert_array_equal(
                core_phi,
                opt_data["phi_angles_list"],
                err_msg=f"Mismatch at n_phi={n_phi}",
            )
            np.testing.assert_array_equal(
                core_phi, plot_phi, err_msg=f"Mismatch at n_phi={n_phi}"
            )
            np.testing.assert_array_equal(
                core_c2, opt_data["c2_exp"], err_msg=f"Mismatch at n_phi={n_phi}"
            )
            np.testing.assert_array_equal(
                core_c2, plot_c2, err_msg=f"Mismatch at n_phi={n_phi}"
            )
            assert (
                core_indices == plot_indices
            ), f"Index mismatch at n_phi={n_phi}"

    def test_consistency_with_overlapping_ranges(self):
        """Verify consistency with overlapping angle ranges."""
        # Arrange
        from tests.factories.config_factory import create_overlapping_ranges_config

        angles = list(range(0, 35, 5))  # [0, 5, 10, 15, 20, 25, 30]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=25, n_t2=25)
        config = create_overlapping_ranges_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
            _apply_angle_filtering_for_plot,
        )

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        core_indices, core_phi, core_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()
        opt_data = _apply_angle_filtering_for_optimization(data, mock_config)

        data_with_config = data.copy()
        data_with_config["config"] = config
        plot_indices, plot_phi, plot_c2 = _apply_angle_filtering_for_plot(
            phi_angles, c2_exp, data_with_config
        )

        # Assert - Should select [0, 5, 10, 15, 20, 25, 30] (all match)
        # Ranges: [0, 20] and [10, 30] overlap, but no duplicates
        np.testing.assert_array_equal(core_phi, opt_data["phi_angles_list"])
        np.testing.assert_array_equal(core_phi, plot_phi)
        np.testing.assert_array_equal(core_c2, opt_data["c2_exp"])
        np.testing.assert_array_equal(core_c2, plot_c2)

        # Verify no duplicates in indices
        assert len(core_indices) == len(
            set(core_indices)
        ), "Core indices contain duplicates"
        assert len(plot_indices) == len(
            set(plot_indices)
        ), "Plot indices contain duplicates"
        assert core_indices == plot_indices


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
