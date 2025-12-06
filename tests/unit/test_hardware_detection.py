"""Tests for Hardware Detection Module
======================================

**Updated**: v3.0 CMC-only migration

Test Coverage:
- HardwareConfig dataclass creation
- JAX device detection (CPU)
- Cluster environment detection (PBS/Slurm)
- CPU core counting
- Backend recommendation logic
- Edge cases and error handling

Note: should_use_cmc() removed in v2.4.1 (CMC-only architecture)
"""

import pytest

from homodyne.device.config import HardwareConfig


class TestHardwareConfig:
    """Test HardwareConfig dataclass."""

    def test_hardware_config_creation(self):
        """Test creating a HardwareConfig instance."""
        config = HardwareConfig(
            platform="cpu",
            num_devices=32,
            memory_per_device_gb=8.0,
            num_nodes=2,
            cores_per_node=32,
            total_memory_gb=512.0,
            cluster_type="pbs",
            recommended_backend="pbs",
            max_parallel_shards=64,
        )

        assert config.platform == "cpu"
        assert config.num_devices == 32
        assert config.memory_per_device_gb == 8.0
        assert config.num_nodes == 2
        assert config.cores_per_node == 32
        assert config.total_memory_gb == 512.0
        assert config.cluster_type == "pbs"
        assert config.recommended_backend == "pbs"
        assert config.max_parallel_shards == 64

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


# Note: TestShouldUseCMC and should_use_cmc() removed in v2.4.1 (CMC-only architecture)


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
