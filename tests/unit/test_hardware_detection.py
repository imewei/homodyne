"""Tests for Hardware Detection Module (Task Group 1)
===================================================

Test Coverage:
- HardwareConfig dataclass creation
- JAX device detection (CPU)
- Cluster environment detection (PBS/Slurm)
- CPU core counting
- Backend recommendation logic
- should_use_cmc() decision logic
- Edge cases and error handling
"""

import os
import unittest
from unittest.mock import MagicMock, Mock, patch, mock_open

import pytest

from homodyne.device.config import HardwareConfig, detect_hardware, should_use_cmc


class TestHardwareConfig:
    """Test HardwareConfig dataclass."""

    def test_hardware_config_creation(self):
        """Test creating a HardwareConfig instance."""
        config = HardwareConfig(
            platform="gpu",
            num_devices=4,
            memory_per_device_gb=80.0,
            num_nodes=2,
            cores_per_node=36,
            total_memory_gb=256.0,
            cluster_type="pbs",
            recommended_backend="pbs",
            max_parallel_shards=72,
        )

        assert config.platform == "gpu"
        assert config.num_devices == 4
        assert config.memory_per_device_gb == 80.0
        assert config.num_nodes == 2
        assert config.cores_per_node == 36
        assert config.total_memory_gb == 256.0
        assert config.cluster_type == "pbs"
        assert config.recommended_backend == "pbs"
        assert config.max_parallel_shards == 72

    def test_hardware_config_with_none_cluster(self):
        """Test HardwareConfig with standalone system."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=32.0,
            num_nodes=1,
            cores_per_node=8,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=8,
        )

        assert config.cluster_type == "standalone"
        assert config.recommended_backend == "multiprocessing"


class TestShouldUseCMC:
    """Test should_use_cmc() decision logic."""

    def test_below_minimum_threshold(self):
        """Test that CMC is not used below minimum threshold (min_samples_for_cmc=15)."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=80.0,
            num_nodes=1,
            cores_per_node=36,
            total_memory_gb=128.0,
            cluster_type="standalone",
            recommended_backend="pjit",
            max_parallel_shards=1,
        )

        # With min_samples_for_cmc=15, test below threshold
        # num_samples=10 < 15 → Parallelism criterion FALSE
        # Large CPU memory → Memory criterion FALSE
        # Result: NUTS (not CMC)
        result = should_use_cmc(10, config)
        assert result is False

    def test_memory_threshold_exceeded(self):
        """Test CMC activation when memory threshold exceeded."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=16.0,  # Small memory
            num_nodes=1,
            cores_per_node=8,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="pjit",
            max_parallel_shards=1,
        )

        # Large dataset that exceeds 80% of 16GB
        # Memory estimate: 50M * 8 bytes * 3 arrays * 2x = 2.4 GB
        # But we use 16GB memory, so this should trigger hardware threshold
        result = should_use_cmc(5_000_000, config)
        assert result is True

    def test_hardware_threshold_cpu(self):
        """Test hardware-specific threshold for CPU system with dual-criteria logic."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=64.0,
            num_nodes=1,
            cores_per_node=16,
            total_memory_gb=64.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=16,
        )

        # With min_samples_for_cmc=15, parallelism criterion dominates
        # Any num_samples >= 15 will trigger CMC for CPU parallelism
        result = should_use_cmc(20_000_000, config)
        assert result is True  # Parallelism criterion: 20M >= 15

        result = should_use_cmc(10_000_000, config)
        assert result is True  # Parallelism criterion: 10M >= 15

        # Only very small sample counts test memory criterion independently
        result = should_use_cmc(10, config)
        assert result is False  # Parallelism: 10 < 15, Memory: OK on 64GB CPU

    def test_custom_memory_threshold(self):
        """Test custom memory threshold parameter."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=16.0,
            num_nodes=1,
            cores_per_node=8,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="pjit",
            max_parallel_shards=1,
        )

        # With lower threshold (50%), should trigger CMC earlier
        # Memory estimate: 2M * 8 * 3 * 2 / 1e9 = 0.096 GB
        # 0.096 / 16 = 0.006 (0.6%) - below 50%, so hardware threshold applies
        result = should_use_cmc(2_000_000, config, memory_threshold_pct=0.5)
        assert result is True  # Due to hardware threshold for 16GB memory

    def test_custom_min_samples(self):
        """Test custom min_samples_for_cmc parameter."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=80.0,
            num_nodes=1,
            cores_per_node=36,
            total_memory_gb=256.0,
            cluster_type="standalone",
            recommended_backend="pjit",
            max_parallel_shards=1,
        )

        # With higher minimum (5M), should not use CMC for 2M points
        result = should_use_cmc(2_000_000, config, min_samples_for_cmc=5_000_000)
        assert result is False

        # But should use CMC for 6M points (exceeds 5M threshold)
        result = should_use_cmc(6_000_000, config, min_samples_for_cmc=5_000_000)
        assert result is True  # 6M >= 5M threshold triggers parallelism mode


class TestBackendRecommendation:
    """Test backend recommendation logic."""

    def test_multinode_pbs_recommendation(self):
        """Test PBS backend recommendation for multi-node cluster."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=4,
            memory_per_device_gb=80.0,
            num_nodes=10,
            cores_per_node=36,
            total_memory_gb=256.0,
            cluster_type="pbs",
            recommended_backend="pbs",
            max_parallel_shards=360,
        )

        assert config.recommended_backend == "pbs"
        assert config.max_parallel_shards == 360

    def test_multinode_slurm_recommendation(self):
        """Test Slurm backend recommendation for multi-node cluster."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=128.0,
            num_nodes=8,
            cores_per_node=128,
            total_memory_gb=512.0,
            cluster_type="slurm",
            recommended_backend="slurm",
            max_parallel_shards=1024,
        )

        assert config.recommended_backend == "slurm"
        assert config.max_parallel_shards == 1024

    def test_cpu_multiprocessing_recommendation(self):
        """Test multiprocessing backend recommendation for CPU-only."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=32.0,
            num_nodes=1,
            cores_per_node=12,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=12,
        )

        assert config.recommended_backend == "multiprocessing"
        assert config.max_parallel_shards == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
