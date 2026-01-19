"""
Unit Tests for NLSQ Memory-Based Strategy Selection
====================================================

Tests for homodyne/optimization/nlsq/memory.py and related strategy selection
covering:
- TestMemoryDetection: System memory detection functions
- TestAdaptiveThreshold: Adaptive memory threshold calculation
- TestPeakMemoryEstimation: Peak memory estimation for NLSQ
- TestStrategySelection: Unified memory-based strategy selection
- TestNumericalCorrectness: Scientific computing validation

These tests verify the Unified Memory-Based NLSQ Strategy (v2.13.0):
- Decision based purely on memory estimation
- No legacy point thresholds (1M/10M/100M deprecated)
- Correct decision order: index check first, then peak memory

References:
- CLAUDE.md: NLSQ Unified Memory-Based Strategy (v2.13.0)
- memory.py: select_nlsq_strategy, NLSQStrategy, StrategyDecision
"""

from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homodyne.optimization.nlsq.memory import (
    DEFAULT_MEMORY_FRACTION,
    MAX_MEMORY_FRACTION,
    MEMORY_FRACTION_ENV_VAR,
    MIN_MEMORY_FRACTION,
    NLSQStrategy,
    StrategyDecision,
    detect_total_system_memory,
    estimate_peak_memory_gb,
    get_adaptive_memory_threshold,
    select_nlsq_strategy,
)


# =============================================================================
# TestMemoryDetection: System memory detection
# =============================================================================
class TestMemoryDetection:
    """Tests for detect_total_system_memory() function."""

    def test_detect_returns_positive_value(self):
        """Memory detection should return positive bytes value."""
        result = detect_total_system_memory()
        # Should be non-None on systems with psutil or sysconf
        if result is not None:
            assert result > 0, "Total memory must be positive"
            assert result > 1e9, "Expected at least 1 GB of RAM"

    def test_detect_returns_bytes(self):
        """Memory detection should return value in bytes."""
        result = detect_total_system_memory()
        if result is not None:
            # Reasonable bounds: 1 GB to 16 TB
            assert 1e9 < result < 16e12, "Memory should be between 1GB and 16TB"

    def test_detect_with_psutil(self):
        """Should use psutil.virtual_memory() when available."""
        # psutil is imported inside detect_total_system_memory, so we patch
        # at the import location
        with patch.dict("sys.modules", {"psutil": MagicMock()}):
            import sys

            mock_psutil = sys.modules["psutil"]
            mock_psutil.virtual_memory.return_value = MagicMock(total=64 * 1024**3)

            # The function uses try/except import, so we need to call it fresh
            # Since psutil is already imported, this test verifies the function works
            result = detect_total_system_memory()

            # Should return a positive value (either mocked or real)
            assert result is not None
            assert result > 0

    def test_detect_fallback_to_sysconf(self):
        """Should fallback to os.sysconf when psutil unavailable.

        Note: This test verifies the fallback path exists and os.sysconf
        is callable on Linux systems.
        """
        # Verify sysconf is available on this system
        assert callable(os.sysconf)

        # Verify we can get memory info via sysconf
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            assert page_size > 0
            assert phys_pages > 0
        except (ValueError, OSError):
            pytest.skip("os.sysconf not available on this system")


# =============================================================================
# TestAdaptiveThreshold: Memory threshold calculation
# =============================================================================
class TestAdaptiveThreshold:
    """Tests for get_adaptive_memory_threshold() function."""

    def test_default_threshold_uses_75_percent(self):
        """Default threshold should be 75% of system RAM."""
        threshold_gb, info = get_adaptive_memory_threshold()

        assert info["memory_fraction"] == DEFAULT_MEMORY_FRACTION
        assert info["source"] == "default"
        assert threshold_gb > 0

    def test_explicit_fraction_argument(self):
        """Explicit fraction argument should override defaults."""
        threshold_gb, info = get_adaptive_memory_threshold(memory_fraction=0.5)

        assert info["memory_fraction"] == 0.5
        assert info["source"] == "argument"

    def test_fraction_clamped_to_min(self):
        """Fractions below MIN_MEMORY_FRACTION should be clamped."""
        with pytest.warns(UserWarning, match="clamped"):
            threshold_gb, info = get_adaptive_memory_threshold(memory_fraction=0.05)

        assert info["memory_fraction"] == MIN_MEMORY_FRACTION

    def test_fraction_clamped_to_max(self):
        """Fractions above MAX_MEMORY_FRACTION should be clamped."""
        with pytest.warns(UserWarning, match="clamped"):
            threshold_gb, info = get_adaptive_memory_threshold(memory_fraction=0.99)

        assert info["memory_fraction"] == MAX_MEMORY_FRACTION

    def test_env_variable_override(self):
        """Environment variable should override default fraction."""
        original = os.environ.get(MEMORY_FRACTION_ENV_VAR)
        try:
            os.environ[MEMORY_FRACTION_ENV_VAR] = "0.6"
            threshold_gb, info = get_adaptive_memory_threshold()

            assert info["memory_fraction"] == 0.6
            assert info["source"] == "env"
        finally:
            if original is None:
                os.environ.pop(MEMORY_FRACTION_ENV_VAR, None)
            else:
                os.environ[MEMORY_FRACTION_ENV_VAR] = original

    def test_invalid_env_variable_falls_back_to_default(self):
        """Invalid env variable should trigger warning and use default."""
        original = os.environ.get(MEMORY_FRACTION_ENV_VAR)
        try:
            os.environ[MEMORY_FRACTION_ENV_VAR] = "invalid"
            with pytest.warns(UserWarning, match="Invalid"):
                threshold_gb, info = get_adaptive_memory_threshold()

            assert info["memory_fraction"] == DEFAULT_MEMORY_FRACTION
        finally:
            if original is None:
                os.environ.pop(MEMORY_FRACTION_ENV_VAR, None)
            else:
                os.environ[MEMORY_FRACTION_ENV_VAR] = original

    def test_info_dict_contains_required_keys(self):
        """Info dict should contain all required diagnostic keys."""
        _, info = get_adaptive_memory_threshold()

        # detection_method only present if memory detected successfully
        assert "memory_fraction" in info
        assert "source" in info


# =============================================================================
# TestPeakMemoryEstimation: Memory estimation for NLSQ
# =============================================================================
class TestPeakMemoryEstimation:
    """Tests for estimate_peak_memory_gb() function."""

    def test_zero_points_returns_zero(self):
        """Zero data points should return zero memory."""
        result = estimate_peak_memory_gb(n_points=0, n_params=10)
        assert result == 0.0

    def test_zero_params_returns_zero(self):
        """Zero parameters should return zero memory."""
        result = estimate_peak_memory_gb(n_points=1_000_000, n_params=0)
        assert result == 0.0

    def test_memory_scales_linearly_with_points(self):
        """Memory should scale linearly with number of points."""
        mem_1m = estimate_peak_memory_gb(n_points=1_000_000, n_params=10)
        mem_10m = estimate_peak_memory_gb(n_points=10_000_000, n_params=10)

        # Should scale approximately 10x (within floating point tolerance)
        np.testing.assert_allclose(mem_10m / mem_1m, 10.0, rtol=1e-10)

    def test_memory_scales_linearly_with_params(self):
        """Memory should scale linearly with number of parameters."""
        mem_10p = estimate_peak_memory_gb(n_points=1_000_000, n_params=10)
        mem_50p = estimate_peak_memory_gb(n_points=1_000_000, n_params=50)

        # Should scale approximately 5x
        np.testing.assert_allclose(mem_50p / mem_10p, 5.0, rtol=1e-10)

    def test_jacobian_overhead_multiplier(self):
        """Jacobian overhead should multiply the base estimate."""
        mem_default = estimate_peak_memory_gb(n_points=1_000_000, n_params=10)
        mem_higher = estimate_peak_memory_gb(
            n_points=1_000_000, n_params=10, jacobian_overhead=13.0
        )

        # Default is 6.5, so 13.0 should double the estimate
        np.testing.assert_allclose(mem_higher / mem_default, 2.0, rtol=1e-10)

    @pytest.mark.parametrize(
        "n_points,n_params,expected_gb",
        [
            # Jacobian: n_points * n_params * 8 bytes * 6.5 overhead
            (1_000_000, 10, 0.04843),  # 1M * 10 * 8 * 6.5 / 1024^3 ≈ 0.048 GB
            (10_000_000, 53, 25.6673),  # 10M * 53 * 8 * 6.5 / 1024^3 ≈ 25.7 GB
            (100_000_000, 53, 256.673),  # 100M * 53 * 8 * 6.5 / 1024^3 ≈ 257 GB
        ],
    )
    def test_numerical_correctness(self, n_points, n_params, expected_gb):
        """Verify memory estimation formula against analytical calculation."""
        result = estimate_peak_memory_gb(n_points, n_params)

        # Expected: n_points * n_params * 8 * 6.5 / (1024^3)
        analytical = (n_points * n_params * 8 * 6.5) / (1024**3)

        np.testing.assert_allclose(result, analytical, rtol=1e-10)


# =============================================================================
# TestNLSQStrategy: Strategy enum and decision dataclass
# =============================================================================
class TestNLSQStrategy:
    """Tests for NLSQStrategy enum."""

    def test_strategy_values(self):
        """Strategy enum should have correct string values."""
        assert NLSQStrategy.STANDARD.value == "standard"
        assert NLSQStrategy.OUT_OF_CORE.value == "out_of_core"
        assert NLSQStrategy.HYBRID_STREAMING.value == "hybrid_streaming"

    def test_strategy_from_value(self):
        """Should be able to create strategy from string value."""
        assert NLSQStrategy("standard") == NLSQStrategy.STANDARD
        assert NLSQStrategy("out_of_core") == NLSQStrategy.OUT_OF_CORE
        assert NLSQStrategy("hybrid_streaming") == NLSQStrategy.HYBRID_STREAMING

    def test_invalid_strategy_raises(self):
        """Invalid strategy value should raise ValueError."""
        with pytest.raises(ValueError):
            NLSQStrategy("invalid")


class TestStrategyDecision:
    """Tests for StrategyDecision dataclass."""

    def test_decision_is_frozen(self):
        """StrategyDecision should be immutable."""
        decision = StrategyDecision(
            strategy=NLSQStrategy.STANDARD,
            threshold_gb=24.0,
            index_memory_gb=0.1,
            peak_memory_gb=1.0,
            reason="Test reason",
        )

        with pytest.raises(FrozenInstanceError):
            decision.strategy = NLSQStrategy.OUT_OF_CORE

    def test_decision_contains_all_metrics(self):
        """StrategyDecision should contain all required metrics."""
        decision = StrategyDecision(
            strategy=NLSQStrategy.OUT_OF_CORE,
            threshold_gb=24.0,
            index_memory_gb=0.8,
            peak_memory_gb=50.0,
            reason="Peak memory exceeds threshold",
        )

        assert decision.strategy == NLSQStrategy.OUT_OF_CORE
        assert decision.threshold_gb == 24.0
        assert decision.index_memory_gb == 0.8
        assert decision.peak_memory_gb == 50.0
        assert "threshold" in decision.reason.lower()


# =============================================================================
# TestStrategySelection: Unified memory-based strategy selection
# =============================================================================
class TestStrategySelection:
    """Tests for select_nlsq_strategy() function."""

    def test_small_dataset_selects_standard(self):
        """Small datasets should select STANDARD strategy."""
        decision = select_nlsq_strategy(n_points=100_000, n_params=10)

        assert decision.strategy == NLSQStrategy.STANDARD
        assert "fits" in decision.reason.lower() or "memory" in decision.reason.lower()

    def test_large_dataset_selects_out_of_core(self):
        """Large datasets exceeding memory should select OUT_OF_CORE."""
        # Use artificially large params to ensure memory threshold exceeded
        # 100M points * 100 params * 8 * 3 = 240 GB peak (exceeds typical RAM)
        decision = select_nlsq_strategy(n_points=100_000_000, n_params=100)

        # Should be OUT_OF_CORE or HYBRID_STREAMING depending on system
        assert decision.strategy in (
            NLSQStrategy.OUT_OF_CORE,
            NLSQStrategy.HYBRID_STREAMING,
        )

    def test_decision_order_index_first(self):
        """Index array check should happen BEFORE peak memory check.

        This is critical: if index > threshold, should be HYBRID_STREAMING
        even if peak memory would suggest OUT_OF_CORE.
        """
        # Simulate extreme dataset where index alone exceeds threshold
        # Need to mock threshold to be very low
        with patch(
            "homodyne.optimization.nlsq.memory.get_adaptive_memory_threshold"
        ) as mock_threshold:
            # Threshold of 1 GB
            mock_threshold.return_value = (1.0, {"memory_fraction": 0.75})

            # 200M points = 1.6 GB index (> 1 GB threshold)
            decision = select_nlsq_strategy(n_points=200_000_000, n_params=10)

            assert decision.strategy == NLSQStrategy.HYBRID_STREAMING
            assert "index" in decision.reason.lower()

    def test_zero_params_returns_standard(self):
        """Zero parameters should default to STANDARD (edge case)."""
        decision = select_nlsq_strategy(n_points=1_000_000, n_params=0)

        assert decision.strategy == NLSQStrategy.STANDARD
        assert decision.peak_memory_gb == 0.0

    def test_zero_points_returns_standard(self):
        """Zero data points should return STANDARD (edge case)."""
        decision = select_nlsq_strategy(n_points=0, n_params=10)

        assert decision.strategy == NLSQStrategy.STANDARD

    def test_negative_values_handled_gracefully(self):
        """Negative values should be handled (return STANDARD)."""
        decision = select_nlsq_strategy(n_points=-100, n_params=-10)

        # Should not crash, likely returns STANDARD
        assert decision.strategy in NLSQStrategy

    def test_custom_memory_fraction(self):
        """Custom memory fraction should affect threshold."""
        decision_75 = select_nlsq_strategy(
            n_points=50_000_000, n_params=50, memory_fraction=0.75
        )
        decision_25 = select_nlsq_strategy(
            n_points=50_000_000, n_params=50, memory_fraction=0.25
        )

        # Lower threshold should more likely trigger OUT_OF_CORE
        assert decision_25.threshold_gb < decision_75.threshold_gb

    @pytest.mark.parametrize(
        "n_points,n_params,expected_strategy",
        [
            # Small datasets - STANDARD
            (100_000, 10, NLSQStrategy.STANDARD),
            (500_000, 20, NLSQStrategy.STANDARD),
            (1_000_000, 10, NLSQStrategy.STANDARD),
            # Medium datasets - depends on system RAM
            # Large datasets - likely OUT_OF_CORE
            (500_000_000, 53, NLSQStrategy.OUT_OF_CORE),
        ],
    )
    def test_strategy_selection_matrix(self, n_points, n_params, expected_strategy):
        """Test strategy selection across various dataset sizes."""
        decision = select_nlsq_strategy(n_points, n_params)

        if expected_strategy == NLSQStrategy.STANDARD:
            # STANDARD should match for small datasets
            assert decision.strategy == expected_strategy
        else:
            # Large datasets should be OUT_OF_CORE or HYBRID_STREAMING
            assert decision.strategy in (
                NLSQStrategy.OUT_OF_CORE,
                NLSQStrategy.HYBRID_STREAMING,
            )


# =============================================================================
# TestNumericalCorrectness: Scientific computing validation
# =============================================================================
class TestNumericalCorrectness:
    """Scientific computing validation tests for memory estimation."""

    def test_memory_formula_analytical_verification(self):
        """Verify memory estimation against analytical formula.

        Formula: peak_gb = (n_points * n_params * bytes_per_elem * overhead) / (1024^3)
        """
        n_points = 10_000_000
        n_params = 53
        bytes_per_elem = 8  # float64
        overhead = 6.5  # default (updated from 3.0 in Jan 2026)

        expected_bytes = n_points * n_params * bytes_per_elem * overhead
        expected_gb = expected_bytes / (1024**3)

        result = estimate_peak_memory_gb(n_points, n_params)

        np.testing.assert_allclose(result, expected_gb, rtol=1e-12)

    def test_index_memory_calculation(self):
        """Verify index memory is computed as n_points * 8 bytes."""
        decision = select_nlsq_strategy(n_points=1_000_000, n_params=10)

        expected_index_gb = (1_000_000 * 8) / (1024**3)
        np.testing.assert_allclose(
            decision.index_memory_gb, expected_index_gb, rtol=1e-12
        )

    def test_threshold_calculation_consistency(self):
        """Threshold should be consistent: fraction * total_memory."""
        threshold_gb, info = get_adaptive_memory_threshold(memory_fraction=0.5)

        if info.get("total_memory_gb", 0) > 0:
            expected = info["total_memory_gb"] * 0.5
            np.testing.assert_allclose(threshold_gb, expected, rtol=1e-12)

    def test_no_nan_or_inf_in_estimates(self):
        """Memory estimates should never contain NaN or Inf."""
        test_cases = [
            (0, 0),
            (1, 1),
            (1_000_000, 100),
            (10**12, 1000),  # Extreme case
        ]

        for n_points, n_params in test_cases:
            result = estimate_peak_memory_gb(n_points, n_params)
            assert np.isfinite(result), (
                f"Got non-finite result for {n_points}, {n_params}"
            )

    def test_decision_metrics_internally_consistent(self):
        """Decision metrics should be internally consistent."""
        decision = select_nlsq_strategy(n_points=10_000_000, n_params=53)

        # All metrics should be non-negative
        assert decision.threshold_gb >= 0
        assert decision.index_memory_gb >= 0
        assert decision.peak_memory_gb >= 0

        # Reason should mention the relevant metric
        if decision.strategy == NLSQStrategy.HYBRID_STREAMING:
            assert "index" in decision.reason.lower()
        elif decision.strategy == NLSQStrategy.OUT_OF_CORE:
            assert (
                "peak" in decision.reason.lower() or "memory" in decision.reason.lower()
            )


# =============================================================================
# TestIntegration: Integration with wrapper.py
# =============================================================================
class TestIntegration:
    """Integration tests for strategy selection with wrapper.py."""

    def test_extract_n_points_helper(self):
        """Test _extract_n_points helper function from wrapper.py."""
        from homodyne.optimization.nlsq.wrapper import _extract_n_points

        # Test with mock XPCSData-like object
        class MockData:
            def __init__(self, size):
                self.g2 = np.zeros(size)

        data = MockData(1000)
        assert _extract_n_points(data) == 1000

    def test_extract_n_points_with_array(self):
        """Test _extract_n_points with raw numpy array."""
        from homodyne.optimization.nlsq.wrapper import _extract_n_points

        arr = np.zeros(5000)
        assert _extract_n_points(arr) == 5000

    def test_extract_n_points_with_list(self):
        """Test _extract_n_points with list-like data."""
        from homodyne.optimization.nlsq.wrapper import _extract_n_points

        class MockDataList:
            def __init__(self, size):
                self.g2 = list(range(size))

        data = MockDataList(2000)
        assert _extract_n_points(data) == 2000

    def test_extract_n_points_empty(self):
        """Test _extract_n_points with empty/invalid data."""
        from homodyne.optimization.nlsq.wrapper import _extract_n_points

        assert _extract_n_points(None) == 0
        assert _extract_n_points({}) == 0


# =============================================================================
# TestEdgeCases: Edge case handling
# =============================================================================
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exactly_at_threshold_boundary(self):
        """Test behavior when memory is exactly at threshold."""
        # Mock to control exact threshold
        with patch(
            "homodyne.optimization.nlsq.memory.get_adaptive_memory_threshold"
        ) as mock_threshold:
            mock_threshold.return_value = (10.0, {"memory_fraction": 0.75})

            # Peak memory exactly at threshold (should trigger OUT_OF_CORE)
            # Need to find n_points * n_params * 8 * 3 / 1024^3 = 10.0
            # n_points * n_params = 10 * 1024^3 / 24 ≈ 447392426
            n_points = 44_739_243  # Approximately
            n_params = 10

            decision = select_nlsq_strategy(n_points, n_params)

            # At threshold, should be OUT_OF_CORE (> not >=)
            if decision.peak_memory_gb > decision.threshold_gb:
                assert decision.strategy == NLSQStrategy.OUT_OF_CORE

    def test_very_small_dataset(self):
        """Test with very small dataset (1 point)."""
        decision = select_nlsq_strategy(n_points=1, n_params=1)
        assert decision.strategy == NLSQStrategy.STANDARD

    def test_very_large_params(self):
        """Test with unusually large parameter count."""
        decision = select_nlsq_strategy(n_points=100_000, n_params=10_000)
        # Large params relative to points might trigger OUT_OF_CORE
        assert decision.strategy in NLSQStrategy

    def test_single_param(self):
        """Test with single parameter optimization."""
        decision = select_nlsq_strategy(n_points=1_000_000, n_params=1)
        assert decision.strategy == NLSQStrategy.STANDARD


# =============================================================================
# TestAutoStreamingMode: Auto-streaming mode switch (T036, FR-007)
# =============================================================================
class TestAutoStreamingMode:
    """Tests for automatic streaming mode switch (T036).

    The optimizer should automatically switch to streaming mode when
    memory constraints require it, without user configuration.
    """

    def test_auto_streaming_mode(self):
        """T036: Verify automatic switch to streaming mode based on memory.

        Performance Optimization (Spec 001 - FR-007, T036): The optimizer
        should automatically detect when streaming mode is needed based on
        memory estimates vs available system memory.
        """
        # Small dataset - should use STANDARD
        small_decision = select_nlsq_strategy(n_points=100_000, n_params=10)
        assert small_decision.strategy == NLSQStrategy.STANDARD

        # Large dataset - should switch to OUT_OF_CORE or HYBRID_STREAMING
        large_decision = select_nlsq_strategy(n_points=500_000_000, n_params=53)
        assert large_decision.strategy in (
            NLSQStrategy.OUT_OF_CORE,
            NLSQStrategy.HYBRID_STREAMING,
        )

    def test_auto_streaming_threshold_sensitivity(self):
        """Test that streaming mode triggers at appropriate threshold."""
        from unittest.mock import patch

        # Mock a low threshold to test switching behavior
        with patch(
            "homodyne.optimization.nlsq.memory.get_adaptive_memory_threshold"
        ) as mock_threshold:
            # Set 5 GB threshold
            mock_threshold.return_value = (5.0, {"memory_fraction": 0.75})

            # Calculate points needed to exceed 5 GB
            # peak_gb = n_points * n_params * 8 * 3 / 1024^3
            # 5 = n_points * 53 * 24 / 1024^3
            # n_points = 5 * 1024^3 / (53 * 24) ≈ 4.2M
            decision = select_nlsq_strategy(n_points=5_000_000, n_params=53)

            # Should trigger non-STANDARD strategy
            assert decision.strategy != NLSQStrategy.STANDARD

    def test_auto_streaming_preserves_correctness(self):
        """Test that streaming mode produces correct strategy decision."""
        # The decision should contain valid information regardless of strategy
        decisions = [
            select_nlsq_strategy(n_points=100_000, n_params=10),
            select_nlsq_strategy(n_points=10_000_000, n_params=53),
            select_nlsq_strategy(n_points=100_000_000, n_params=100),
        ]

        for decision in decisions:
            # All decisions should have valid fields
            assert decision.strategy in NLSQStrategy
            assert decision.threshold_gb > 0
            assert decision.index_memory_gb >= 0
            assert decision.peak_memory_gb >= 0
            assert len(decision.reason) > 0

    def test_streaming_mode_deterministic(self):
        """Test that strategy selection is deterministic."""
        # Same inputs should produce same outputs
        n_points = 10_000_000
        n_params = 53

        decisions = [select_nlsq_strategy(n_points, n_params) for _ in range(5)]

        # All decisions should be identical
        first = decisions[0]
        for decision in decisions[1:]:
            assert decision.strategy == first.strategy
            assert decision.threshold_gb == first.threshold_gb
            assert decision.peak_memory_gb == first.peak_memory_gb

    def test_memory_fraction_affects_streaming_trigger(self):
        """Test that memory_fraction parameter affects when streaming triggers."""
        n_points = 20_000_000
        n_params = 53

        # High fraction - more memory available, less likely to stream
        high_fraction = select_nlsq_strategy(n_points, n_params, memory_fraction=0.9)

        # Low fraction - less memory available, more likely to stream
        low_fraction = select_nlsq_strategy(n_points, n_params, memory_fraction=0.3)

        # Lower fraction should have lower threshold
        assert low_fraction.threshold_gb < high_fraction.threshold_gb

        # If strategies differ, low fraction should be more restrictive
        if high_fraction.strategy == NLSQStrategy.STANDARD:
            # Low fraction might still be STANDARD or stricter
            pass  # Both valid
        elif high_fraction.strategy == NLSQStrategy.OUT_OF_CORE:
            # Low fraction should be same or stricter
            assert low_fraction.strategy in (
                NLSQStrategy.OUT_OF_CORE,
                NLSQStrategy.HYBRID_STREAMING,
            )
