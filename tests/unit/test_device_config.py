"""
Unit tests for hardware detection and CMC selection logic.

Tests cover:
- Dual-criteria OR logic: (num_samples >= 15) OR (memory > 30%)
- Parallelism trigger (many samples)
- Memory trigger (large datasets)
- NUTS selection (small datasets)
- Threshold configurability from YAML
"""

import pytest
from homodyne.device.config import HardwareConfig, should_use_cmc


class TestCMCSelectionLogic:
    """Test CMC selection logic with dual-criteria OR conditions."""

    @pytest.fixture
    def mock_hardware_config(self):
        """Create a mock hardware configuration for testing."""
        return HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=32.0,
            num_nodes=1,
            cores_per_node=14,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=14,
        )

    def test_dual_criteria_or_logic_parallelism_mode(self, mock_hardware_config):
        """
        Test dual-criteria OR logic: num_samples >= 15 triggers CMC (parallelism mode).

        Spec requirement: (num_samples >= 15) OR (memory > 30%) → CMC
        This test verifies the parallelism trigger.
        """
        # 20 samples with small dataset → CMC via parallelism trigger
        num_samples = 20
        dataset_size = 1_000_000  # Small enough to not trigger memory threshold

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            f"CMC should be selected for {num_samples} samples "
            f"(>= 15 threshold) even with small dataset"
        )

    def test_dual_criteria_or_logic_memory_mode(self, mock_hardware_config):
        """
        Test dual-criteria OR logic: memory > 30% triggers CMC (memory mode).

        Spec requirement: (num_samples >= 15) OR (memory > 30%) → CMC
        This test verifies the memory trigger.
        """
        # 5 samples (below threshold) but HUGE dataset → CMC via memory trigger
        num_samples = 5
        dataset_size = 50_000_000  # Large enough to trigger memory threshold

        # Calculate expected memory: 50M * 8 bytes * 30 / 1e9 ≈ 12 GB
        # 12 GB / 32 GB = 37.5% > 30% threshold → should trigger CMC

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            f"CMC should be selected for large dataset ({dataset_size:,} points) "
            f"even with only {num_samples} samples (< 15 threshold)"
        )

    def test_parallelism_trigger_exactly_at_threshold(self, mock_hardware_config):
        """
        Test parallelism trigger: exactly 15 samples should trigger CMC.

        Spec requirement: min_samples_for_cmc=15 (default)
        Boundary condition: exactly at threshold.
        """
        num_samples = 15  # Exactly at threshold
        dataset_size = 1_000_000

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            "CMC should be selected at exactly 15 samples (threshold boundary)"
        )

    def test_nuts_selection_below_all_thresholds(self, mock_hardware_config):
        """
        Test NUTS selection: small samples AND small dataset → NUTS.

        Spec requirement: If both conditions fail → NUTS
        """
        num_samples = 10  # Below 15 threshold
        dataset_size = 5_000_000  # Small dataset

        # Calculate expected memory: 5M * 8 bytes * 30 / 1e9 ≈ 1.2 GB
        # 1.2 GB / 32 GB = 3.75% < 30% threshold → both conditions fail

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, (
            f"NUTS should be selected for {num_samples} samples (< 15) "
            f"and small dataset (< 30% memory)"
        )

    def test_threshold_configurability_custom_sample_threshold(self, mock_hardware_config):
        """
        Test threshold configurability: custom min_samples_for_cmc.

        Spec requirement: Thresholds should be configurable from YAML.
        """
        num_samples = 25
        dataset_size = 1_000_000

        # Test with custom threshold of 30 samples
        result_high_threshold = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=30,  # Custom: higher than default 15
        )

        assert result_high_threshold is False, (
            f"NUTS should be selected when {num_samples} < custom threshold (30)"
        )

        # Test with custom threshold of 20 samples
        result_low_threshold = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=20,  # Custom: 25 >= 20
        )

        assert result_low_threshold is True, (
            f"CMC should be selected when {num_samples} >= custom threshold (20)"
        )

    def test_threshold_configurability_custom_memory_threshold(self, mock_hardware_config):
        """
        Test threshold configurability: custom memory_threshold_pct.

        Spec requirement: Thresholds should be configurable from YAML.
        """
        num_samples = 5  # Below sample threshold
        dataset_size = 30_000_000

        # Memory: 30M * 8 * 30 / 1e9 = 7.2 GB → 7.2/32 = 22.5% memory usage

        # Test with strict 20% threshold
        result_strict = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.20,  # 22.5% > 20% → trigger CMC
            min_samples_for_cmc=15,
        )

        assert result_strict is True, (
            "CMC should be selected when memory (22.5%) > strict threshold (20%)"
        )

        # Test with relaxed 25% threshold
        result_relaxed = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.25,  # 22.5% < 25% → both conditions fail
            min_samples_for_cmc=15,
        )

        assert result_relaxed is False, (
            "NUTS should be selected when memory (22.5%) < relaxed threshold (25%)"
        )

    def test_memory_estimation_without_dataset_size(self, mock_hardware_config):
        """
        Test behavior when dataset_size is None (only sample-based decision).

        Spec requirement: Memory trigger only applies when dataset_size is provided.
        """
        num_samples = 10  # Below threshold

        # No dataset_size provided → only sample-based decision
        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=None,  # No memory estimation
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, (
            "NUTS should be selected when num_samples < threshold and "
            "dataset_size is None (no memory trigger)"
        )

    def test_realistic_xpcs_scenario_small_experiment(self, mock_hardware_config):
        """
        Test realistic XPCS scenario: small experiment (10 phi angles, 5M points).

        Real-world use case: Small dataset that should use NUTS.
        """
        num_samples = 10  # 10 phi angles
        dataset_size = 5_000_000  # 5M points total

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, "Small XPCS experiment should use NUTS"

    def test_realistic_xpcs_scenario_medium_experiment(self, mock_hardware_config):
        """
        Test realistic XPCS scenario: medium experiment (20 phi angles, 10M points).

        Real-world use case: Parallelism-triggered CMC.
        """
        num_samples = 20  # 20 phi angles
        dataset_size = 10_000_000  # 10M points total

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            "Medium XPCS experiment with 20 phi angles should use CMC "
            "(parallelism mode)"
        )

    def test_realistic_xpcs_scenario_large_memory_experiment(self, mock_hardware_config):
        """
        Test realistic XPCS scenario: few angles but huge data (5 phi, 50M points).

        Real-world use case: Memory-triggered CMC.
        """
        num_samples = 5  # Only 5 phi angles
        dataset_size = 50_000_000  # 50M points total

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is True, (
            "Large memory XPCS experiment should use CMC (memory mode) "
            "even with only 5 phi angles"
        )
