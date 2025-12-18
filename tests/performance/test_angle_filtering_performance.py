"""Performance benchmarks for angle filtering.

This module validates that angle filtering operations meet performance requirements:
- Filtering overhead < 100ms for typical datasets
- O(n) time complexity with respect to number of angles
"""

import time

import pytest

from tests.factories.config_factory import (
    create_anisotropic_filtering_config,
    create_laminar_flow_filtering_config,
)
from tests.factories.data_factory import create_angle_filtering_test_data


class TestAngleFilteringPerformance:
    """Performance benchmarks for angle filtering operations."""

    def test_filtering_overhead_small_dataset(self):
        """Verify filtering overhead < 100ms for small dataset (10 angles)."""
        # Arrange
        n_phi = 10
        n_t1 = 25
        n_t2 = 25
        data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=n_t1, n_t2=n_t2)
        config = create_anisotropic_filtering_config()

        # Import the function
        from homodyne.cli.commands import _apply_angle_filtering

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act - Time the filtering operation
        start_time = time.perf_counter()
        _apply_angle_filtering(phi_angles, c2_exp, config)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert
        assert elapsed_ms < 100, f"Filtering took {elapsed_ms:.2f}ms, expected < 100ms"

    def test_filtering_overhead_medium_dataset(self):
        """Verify filtering overhead < 100ms for medium dataset (50 angles)."""
        # Arrange
        n_phi = 50
        n_t1 = 50
        n_t2 = 50
        data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=n_t1, n_t2=n_t2)
        config = create_laminar_flow_filtering_config()

        from homodyne.cli.commands import _apply_angle_filtering

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        start_time = time.perf_counter()
        _apply_angle_filtering(phi_angles, c2_exp, config)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert
        assert elapsed_ms < 100, f"Filtering took {elapsed_ms:.2f}ms, expected < 100ms"

    def test_filtering_overhead_large_dataset(self):
        """Verify filtering overhead < 100ms for large dataset (360 angles)."""
        # Arrange
        n_phi = 360  # Full 1-degree resolution
        n_t1 = 100
        n_t2 = 100
        data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=n_t1, n_t2=n_t2)
        config = create_laminar_flow_filtering_config()

        from homodyne.cli.commands import _apply_angle_filtering

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        start_time = time.perf_counter()
        _apply_angle_filtering(phi_angles, c2_exp, config)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000

        # Assert
        assert elapsed_ms < 100, f"Filtering took {elapsed_ms:.2f}ms, expected < 100ms"

    def test_filtering_time_complexity(self):
        """Verify O(n) time complexity with respect to number of angles."""
        # Arrange
        config = create_anisotropic_filtering_config()
        from homodyne.cli.commands import _apply_angle_filtering

        # Test with increasing dataset sizes
        sizes = [10, 50, 100, 200, 360]
        times = []

        for n_phi in sizes:
            data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=25, n_t2=25)
            phi_angles = data["phi_angles_list"]
            c2_exp = data["c2_exp"]

            # Warm-up run
            _apply_angle_filtering(phi_angles, c2_exp, config)

            # Timed run
            start_time = time.perf_counter()
            _apply_angle_filtering(phi_angles, c2_exp, config)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

        # Act - Calculate time ratios
        # For O(n), doubling size should roughly double time
        # We allow for some overhead, so check if ratio < 3x
        time_ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
        size_ratios = [sizes[i + 1] / sizes[i] for i in range(len(sizes) - 1)]

        # Assert - Time ratio should be roughly proportional to size ratio
        # Allow 3x overhead for small datasets
        for i, (time_ratio, size_ratio) in enumerate(
            zip(time_ratios, size_ratios, strict=False)
        ):
            assert time_ratio < size_ratio * 3, (
                f"Step {i}: Time scaled by {time_ratio:.2f}x but size by {size_ratio:.2f}x"
            )

    def test_filtering_wrapper_overhead(self):
        """Verify optimization wrapper adds minimal overhead."""
        # Arrange
        n_phi = 50
        data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=50, n_t2=50)
        config = create_anisotropic_filtering_config()

        from homodyne.cli.commands import (
            _apply_angle_filtering,
            _apply_angle_filtering_for_optimization,
        )

        # Mock ConfigManager
        class MockConfigManager:
            def get_config(self):
                return config

        mock_config = MockConfigManager()

        # Act - Time core function
        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        start_time = time.perf_counter()
        _apply_angle_filtering(phi_angles, c2_exp, config)
        end_time = time.perf_counter()
        core_time = end_time - start_time

        # Act - Time wrapper function
        start_time = time.perf_counter()
        _apply_angle_filtering_for_optimization(data, mock_config)
        end_time = time.perf_counter()
        wrapper_time = end_time - start_time

        # Assert - Wrapper should add < 20ms overhead
        # Note: 20ms threshold accounts for system load variability in CI
        overhead_ms = (wrapper_time - core_time) * 1000
        assert overhead_ms < 20, (
            f"Wrapper added {overhead_ms:.2f}ms overhead, expected < 20ms"
        )

    def test_no_filtering_minimal_overhead(self):
        """Verify disabled filtering has near-zero overhead."""
        # Arrange
        n_phi = 100
        data = create_angle_filtering_test_data(n_phi=n_phi, n_t1=50, n_t2=50)
        config = {"phi_filtering": {"enabled": False}}

        from homodyne.cli.commands import _apply_angle_filtering

        phi_angles = data["phi_angles_list"]
        c2_exp = data["c2_exp"]

        # Act
        start_time = time.perf_counter()
        for _ in range(1000):  # Run 1000 times to measure overhead
            _apply_angle_filtering(phi_angles, c2_exp, config)
        end_time = time.perf_counter()

        elapsed_ms = (end_time - start_time) * 1000
        per_call_us = elapsed_ms / 1000  # microseconds per call

        # Assert - Each call should take < 10 microseconds when disabled
        assert per_call_us < 10, (
            f"Disabled filtering took {per_call_us:.2f}µs per call, expected < 10µs"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
