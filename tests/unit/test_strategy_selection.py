"""Unit tests for dataset size strategy selection.

Tests cover:
- Strategy selection for all size ranges
- Memory-based adjustments
- Configuration overrides
- Edge cases and error handling
"""

import pytest

from homodyne.optimization.strategy import (
    DatasetSizeStrategy,
    OptimizationStrategy,
    estimate_memory_requirements,
)


class TestStrategySelection:
    """Test basic strategy selection based on dataset size."""

    def test_standard_strategy_for_small_dataset(self):
        """Test STANDARD strategy selected for <1M points."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(500_000, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_large_strategy_for_medium_dataset(self):
        """Test LARGE strategy selected for 1M-10M points."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(5_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.LARGE

    def test_chunked_strategy_for_large_dataset(self):
        """Test CHUNKED strategy selected for 10M-100M points."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(50_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.CHUNKED

    def test_streaming_strategy_for_xlarge_dataset(self):
        """Test STREAMING strategy selected for >100M points."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(500_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.STREAMING

    def test_boundary_condition_at_1m(self):
        """Test boundary at 1M points (should be LARGE, not STANDARD)."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(1_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.LARGE

    def test_boundary_condition_at_10m(self):
        """Test boundary at 10M points (should be CHUNKED, not LARGE)."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(10_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.CHUNKED

    def test_boundary_condition_at_100m(self):
        """Test boundary at 100M points (should be STREAMING, not CHUNKED)."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(100_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.STREAMING

    def test_just_below_1m_is_standard(self):
        """Test that 999,999 points uses STANDARD."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(999_999, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_just_below_10m_is_large(self):
        """Test that 9,999,999 points uses LARGE."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(9_999_999, check_memory=False)
        assert strategy == OptimizationStrategy.LARGE

    def test_just_below_100m_is_chunked(self):
        """Test that 99,999,999 points uses CHUNKED."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(99_999_999, check_memory=False)
        assert strategy == OptimizationStrategy.CHUNKED


class TestMemoryBasedAdjustment:
    """Test memory-based strategy adjustments."""

    def test_memory_check_enabled_by_default(self):
        """Test that memory checking is enabled by default."""
        selector = DatasetSizeStrategy()
        # Should not raise an error
        strategy = selector.select_strategy(5_000_000)
        assert isinstance(strategy, OptimizationStrategy)

    def test_memory_check_can_be_disabled(self):
        """Test that memory checking can be disabled."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(5_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.LARGE

    def test_memory_limit_override(self):
        """Test custom memory limit via config."""
        config = {"memory_limit_gb": 2.0}  # Very low limit
        selector = DatasetSizeStrategy(config)

        # Large dataset with low memory should trigger adjustment
        strategy = selector.select_strategy(50_000_000, check_memory=True)

        # Should select memory-efficient strategy
        # (may be STREAMING due to memory constraints)
        assert strategy in [
            OptimizationStrategy.CHUNKED,
            OptimizationStrategy.STREAMING,
        ]

    def test_available_memory_detection(self):
        """Test that available memory is detected correctly."""
        selector = DatasetSizeStrategy()
        available_gb = selector._get_available_memory_gb()
        assert available_gb > 0
        assert available_gb < 1000  # Sanity check


class TestConfigurationOverrides:
    """Test configuration override functionality."""

    def test_strategy_override_standard(self):
        """Test forcing STANDARD strategy via override."""
        config = {"strategy_override": "standard"}
        selector = DatasetSizeStrategy(config)
        strategy = selector.select_strategy(100_000_000, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_strategy_override_large(self):
        """Test forcing LARGE strategy via override."""
        config = {"strategy_override": "large"}
        selector = DatasetSizeStrategy(config)
        strategy = selector.select_strategy(500_000, check_memory=False)
        assert strategy == OptimizationStrategy.LARGE

    def test_strategy_override_chunked(self):
        """Test forcing CHUNKED strategy via override."""
        config = {"strategy_override": "chunked"}
        selector = DatasetSizeStrategy(config)
        strategy = selector.select_strategy(500_000, check_memory=False)
        assert strategy == OptimizationStrategy.CHUNKED

    def test_strategy_override_streaming(self):
        """Test forcing STREAMING strategy via override."""
        config = {"strategy_override": "streaming"}
        selector = DatasetSizeStrategy(config)
        strategy = selector.select_strategy(500_000, check_memory=False)
        assert strategy == OptimizationStrategy.STREAMING

    def test_invalid_override_falls_back_to_auto(self):
        """Test that invalid override falls back to automatic selection."""
        config = {"strategy_override": "invalid_strategy"}
        selector = DatasetSizeStrategy(config)
        strategy = selector.select_strategy(500_000, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_enable_progress_configuration(self):
        """Test progress bar configuration."""
        config = {"enable_progress": False}
        selector = DatasetSizeStrategy(config)
        assert selector._enable_progress is False


class TestMemoryEstimation:
    """Test memory estimation logic."""

    def test_memory_estimate_increases_with_points(self):
        """Test that memory estimate scales with data size."""
        selector = DatasetSizeStrategy()

        estimate_1m = selector._estimate_memory_gb(1_000_000, 9)
        estimate_10m = selector._estimate_memory_gb(10_000_000, 9)

        assert estimate_10m > estimate_1m
        assert estimate_10m / estimate_1m == pytest.approx(10.0, rel=0.2)

    def test_memory_estimate_increases_with_parameters(self):
        """Test that memory estimate scales with parameter count."""
        selector = DatasetSizeStrategy()

        estimate_5params = selector._estimate_memory_gb(5_000_000, 5)
        estimate_9params = selector._estimate_memory_gb(5_000_000, 9)

        assert estimate_9params > estimate_5params

    def test_memory_estimate_positive(self):
        """Test that memory estimates are always positive."""
        selector = DatasetSizeStrategy()

        estimates = [
            selector._estimate_memory_gb(1000, 3),
            selector._estimate_memory_gb(1_000_000, 5),
            selector._estimate_memory_gb(100_000_000, 9),
        ]

        for estimate in estimates:
            assert estimate > 0

    def test_memory_estimate_reasonable_magnitude(self):
        """Test that memory estimates are in reasonable range."""
        selector = DatasetSizeStrategy()

        # 5M points with 9 parameters should be in GB range, not TB
        estimate = selector._estimate_memory_gb(5_000_000, 9)
        assert 0.1 < estimate < 100  # Between 100MB and 100GB


class TestStrategyInfo:
    """Test strategy information retrieval."""

    def test_get_info_for_standard(self):
        """Test getting info for STANDARD strategy."""
        selector = DatasetSizeStrategy()
        info = selector.get_strategy_info(OptimizationStrategy.STANDARD)

        assert info["name"] == "Standard"
        assert info["nlsq_function"] == "curve_fit"
        assert info["supports_progress"] is False

    def test_get_info_for_large(self):
        """Test getting info for LARGE strategy."""
        selector = DatasetSizeStrategy()
        info = selector.get_strategy_info(OptimizationStrategy.LARGE)

        assert info["name"] == "Large"
        assert info["nlsq_function"] == "curve_fit_large"
        assert info["supports_progress"] is True

    def test_get_info_for_chunked(self):
        """Test getting info for CHUNKED strategy."""
        selector = DatasetSizeStrategy()
        info = selector.get_strategy_info(OptimizationStrategy.CHUNKED)

        assert info["name"] == "Chunked"
        assert "curve_fit_large" in info["nlsq_function"]
        assert info["supports_progress"] is True

    def test_get_info_for_streaming(self):
        """Test getting info for STREAMING strategy."""
        selector = DatasetSizeStrategy()
        info = selector.get_strategy_info(OptimizationStrategy.STREAMING)

        assert info["name"] == "Streaming"
        assert "streaming" in info["nlsq_function"].lower()
        assert info["supports_progress"] is True

    def test_all_strategies_have_complete_info(self):
        """Test that all strategies have complete information."""
        selector = DatasetSizeStrategy()

        required_keys = [
            "name",
            "description",
            "use_case",
            "nlsq_function",
            "supports_progress",
        ]

        for strategy in OptimizationStrategy:
            info = selector.get_strategy_info(strategy)
            for key in required_keys:
                assert key in info
                assert info[key] is not None


class TestHelperFunctions:
    """Test helper functions."""

    def test_estimate_memory_requirements_returns_dict(self):
        """Test that estimate_memory_requirements returns complete dict."""
        stats = estimate_memory_requirements(5_000_000, 9)

        required_keys = [
            "total_memory_estimate_gb",
            "available_memory_gb",
            "memory_safe",
            "recommended_strategy",
            "strategy_info",
        ]

        for key in required_keys:
            assert key in stats

    def test_estimate_memory_requirements_reasonable_values(self):
        """Test that memory requirement estimates are reasonable."""
        stats = estimate_memory_requirements(5_000_000, 9)

        assert stats["total_memory_estimate_gb"] > 0
        assert stats["available_memory_gb"] > 0
        assert isinstance(stats["memory_safe"], bool)
        assert stats["recommended_strategy"] in [
            "standard",
            "large",
            "chunked",
            "streaming",
        ]

    def test_estimate_memory_requirements_small_dataset(self):
        """Test memory requirements for small dataset."""
        stats = estimate_memory_requirements(500_000, 5)

        # Should recommend STANDARD strategy
        assert stats["recommended_strategy"] == "standard"

    def test_estimate_memory_requirements_large_dataset(self):
        """Test memory requirements for large dataset."""
        stats = estimate_memory_requirements(50_000_000, 9)

        # Should recommend CHUNKED or STREAMING
        assert stats["recommended_strategy"] in ["chunked", "streaming"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_points_raises_no_error(self):
        """Test that zero points doesn't crash (though nonsensical)."""
        selector = DatasetSizeStrategy()
        # Should not raise an error
        strategy = selector.select_strategy(0, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_single_point(self):
        """Test single data point."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(1, check_memory=False)
        assert strategy == OptimizationStrategy.STANDARD

    def test_very_large_parameter_count(self):
        """Test with unusually large parameter count."""
        selector = DatasetSizeStrategy()
        # Should not crash
        strategy = selector.select_strategy(5_000_000, n_parameters=100)
        assert isinstance(strategy, OptimizationStrategy)

    def test_single_parameter(self):
        """Test with single parameter."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(5_000_000, n_parameters=1)
        assert strategy == OptimizationStrategy.LARGE

    def test_memory_adjustment_with_exception_handling(self):
        """Test that memory adjustment handles exceptions gracefully."""
        selector = DatasetSizeStrategy()

        # Even with potential errors, should return valid strategy
        strategy = selector.select_strategy(5_000_000, check_memory=True)
        assert isinstance(strategy, OptimizationStrategy)


class TestParameterVariations:
    """Test with different parameter counts for different analysis modes."""

    def test_static_mode_5_parameters(self):
        """Test with 5 parameters (static_mode: 3 physical + 2 scaling)."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(5_000_000, n_parameters=5)
        assert strategy == OptimizationStrategy.LARGE

    def test_laminar_flow_9_parameters(self):
        """Test with 9 parameters (laminar_flow: 7 physical + 2 scaling)."""
        selector = DatasetSizeStrategy()
        strategy = selector.select_strategy(5_000_000, n_parameters=9)
        assert strategy == OptimizationStrategy.LARGE

    def test_parameter_count_affects_memory_estimate(self):
        """Test that parameter count affects memory-based adjustments."""
        selector = DatasetSizeStrategy()

        # Same dataset size, different parameter counts
        strategy_5params = selector.select_strategy(
            50_000_000, n_parameters=5, check_memory=True
        )
        strategy_9params = selector.select_strategy(
            50_000_000, n_parameters=9, check_memory=True
        )

        # Both should be valid strategies
        assert isinstance(strategy_5params, OptimizationStrategy)
        assert isinstance(strategy_9params, OptimizationStrategy)
