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

        assert (
            result is True
        ), "CMC should be selected at exactly 15 samples (threshold boundary)"

    def test_nuts_selection_below_all_thresholds(self, mock_hardware_config):
        """
        Test NUTS selection: small samples AND small dataset → NUTS.

        Spec requirement: If all quad-criteria fail → NUTS

        NOTE: Must use dataset_size < 1M to avoid triggering Criterion 3
        (JAX Broadcasting Protection at 1M threshold)
        """
        num_samples = 10  # Below 15 threshold
        dataset_size = 500_000  # Small dataset (< 1M to avoid Criterion 3)

        # Calculate expected memory: 500K * 8 bytes * 30 / 1e9 ≈ 0.12 GB
        # 0.12 GB / 32 GB = 0.375% < 30% threshold → all quad-criteria fail

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=15,
        )

        assert result is False, (
            f"NUTS should be selected for {num_samples} samples (< 15), "
            f"small dataset (< 30% memory), and dataset_size < 1M (no broadcasting protection)"
        )

    def test_threshold_configurability_custom_sample_threshold(
        self, mock_hardware_config
    ):
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

        assert (
            result_high_threshold is False
        ), f"NUTS should be selected when {num_samples} < custom threshold (30)"

        # Test with custom threshold of 20 samples
        result_low_threshold = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.30,
            min_samples_for_cmc=20,  # Custom: 25 >= 20
        )

        assert (
            result_low_threshold is True
        ), f"CMC should be selected when {num_samples} >= custom threshold (20)"

    def test_threshold_configurability_custom_memory_threshold(
        self, mock_hardware_config
    ):
        """
        Test threshold configurability: custom memory_threshold_pct.

        Spec requirement: Thresholds should be configurable from YAML.

        NOTE (Quad-Criteria): With large dataset (30M > 1M), Criterion 3
        (JAX Broadcasting Protection) and Criterion 4 (Large Dataset, Few Samples)
        will also trigger. This test verifies memory threshold configurability
        while acknowledging other criteria may override.
        """
        num_samples = 5  # Below sample threshold
        dataset_size = 30_000_000

        # Memory: 30M * 8 * 5 / 1e9 = 1.2 GB → 1.2/32 = 3.75% memory usage
        # Note: Adjusted num_samples from 30 to 5 to match test intent

        # Test with strict 1% threshold
        result_strict = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.01,  # 3.75% > 1% → trigger CMC (Criterion 2)
            min_samples_for_cmc=15,
        )

        assert (
            result_strict is True
        ), "CMC should be selected when memory (3.75%) > strict threshold (1%)"

        # Test with relaxed 10% threshold
        # NOTE: Criterion 3 and 4 will still trigger due to large dataset
        result_relaxed = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.10,  # 3.75% < 10% → Criterion 2 doesn't trigger
            min_samples_for_cmc=15,
        )

        # With quad-criteria, CMC will still be selected due to Criteria 3 and 4
        assert (
            result_relaxed is True
        ), "CMC selected due to Criterion 3 (JAX protection) and Criterion 4 (large dataset, few samples)"

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
        Test realistic XPCS scenario: small experiment (10 phi angles, 500K points).

        Real-world use case: Small dataset that should use NUTS.

        NOTE: Using 500K points (< 1M) to avoid Criterion 3 (JAX Broadcasting Protection).
        Real XPCS experiments with < 1M points are common for quick scans.
        """
        num_samples = 10  # 10 phi angles
        dataset_size = 500_000  # 500K points total (< 1M to avoid Criterion 3)

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

    def test_realistic_xpcs_scenario_large_memory_experiment(
        self, mock_hardware_config
    ):
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
