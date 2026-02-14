"""Unit tests for device detection and configuration.

Tests for CPU detection, device configuration, and performance benchmarking.
"""

import pytest

from homodyne.device import (
    HAS_CPU_MODULE,
    configure_optimal_device,
    get_device_status,
)

# Import CPU-specific functions if available
if HAS_CPU_MODULE:
    from homodyne.device import (
        benchmark_cpu_performance,
        configure_cpu_hpc,
        detect_cpu_info,
    )


class TestDeviceConfiguration:
    """Tests for device configuration functions."""

    def test_configure_optimal_device_returns_dict(self):
        """Test that configure_optimal_device returns a configuration dict."""
        config = configure_optimal_device()
        assert isinstance(config, dict)
        assert "device_type" in config
        assert config["device_type"] == "cpu"

    def test_configure_optimal_device_has_required_keys(self):
        """Test that configuration dict has required keys."""
        config = configure_optimal_device()
        assert "configuration_successful" in config
        assert "performance_ready" in config
        assert "recommendations" in config
        assert isinstance(config["recommendations"], list)

    def test_configure_optimal_device_with_threads(self):
        """Test configuration with explicit thread count."""
        config = configure_optimal_device(cpu_threads=4)
        assert config["device_type"] == "cpu"
        assert isinstance(config, dict)

    def test_get_device_status_returns_dict(self):
        """Test that get_device_status returns status dict."""
        status = get_device_status()
        assert isinstance(status, dict)
        assert "cpu_info" in status
        assert isinstance(status["cpu_info"], dict)

    def test_get_device_status_has_timestamp(self):
        """Test that status includes timestamp."""
        status = get_device_status()
        assert "timestamp" in status
        assert status["timestamp"] is not None

    def test_get_device_status_has_performance_estimate(self):
        """Test that status includes performance estimate."""
        status = get_device_status()
        assert "performance_estimate" in status
        assert status["performance_estimate"] in [
            "high",
            "medium-high",
            "medium",
            "unknown",
        ]


@pytest.mark.skipif(not HAS_CPU_MODULE, reason="CPU module not available")
class TestCPUDetection:
    """Tests for CPU detection functions."""

    def test_detect_cpu_info_returns_dict(self):
        """Test that detect_cpu_info returns CPU information dict."""
        info = detect_cpu_info()
        assert isinstance(info, dict)

    def test_detect_cpu_info_has_core_counts(self):
        """Test that CPU info includes core counts."""
        info = detect_cpu_info()
        assert "physical_cores" in info
        assert "logical_cores" in info
        assert isinstance(info["physical_cores"], int)
        assert isinstance(info["logical_cores"], int)
        assert info["physical_cores"] > 0
        assert info["logical_cores"] > 0
        assert info["logical_cores"] >= info["physical_cores"]

    def test_detect_cpu_info_has_architecture(self):
        """Test that CPU info includes architecture details."""
        info = detect_cpu_info()
        assert "architecture" in info
        assert "processor" in info
        assert isinstance(info["architecture"], str)

    def test_detect_cpu_info_has_numa_nodes(self):
        """Test that CPU info includes NUMA node count."""
        info = detect_cpu_info()
        assert "numa_nodes" in info
        assert isinstance(info["numa_nodes"], int)
        assert info["numa_nodes"] >= 1

    def test_configure_cpu_hpc_returns_dict(self):
        """Test that configure_cpu_hpc returns configuration dict."""
        config = configure_cpu_hpc()
        assert isinstance(config, dict)

    def test_configure_cpu_hpc_with_explicit_threads(self):
        """Test CPU configuration with explicit thread count."""
        config = configure_cpu_hpc(num_threads=4)
        assert isinstance(config, dict)

    def test_configure_cpu_hpc_with_hyperthreading(self):
        """Test CPU configuration with hyperthreading enabled."""
        config = configure_cpu_hpc(enable_hyperthreading=True)
        assert isinstance(config, dict)

    def test_configure_cpu_hpc_numa_policies(self):
        """Test CPU configuration with different NUMA policies."""
        for policy in ["auto", "local", "interleave"]:
            config = configure_cpu_hpc(numa_policy=policy)
            assert isinstance(config, dict)

    def test_configure_cpu_hpc_memory_optimization(self):
        """Test CPU configuration with different memory optimization levels."""
        for level in ["minimal", "standard", "aggressive"]:
            config = configure_cpu_hpc(memory_optimization=level)
            assert isinstance(config, dict)

    def test_benchmark_cpu_performance_returns_dict(self):
        """Test that benchmark_cpu_performance returns benchmark results."""
        results = benchmark_cpu_performance(test_size=100)
        assert isinstance(results, dict)

    def test_benchmark_cpu_performance_small_test(self):
        """Test CPU benchmark with small test size."""
        results = benchmark_cpu_performance(test_size=50)
        assert isinstance(results, dict)
        assert "cpu_info" in results


class TestDeviceBenchmarking:
    """Tests for device benchmarking functions."""

    def test_benchmark_device_performance_returns_dict(self):
        """Test that device benchmarking returns results dict."""
        from homodyne.device import benchmark_device_performance

        results = benchmark_device_performance(test_size=100)
        assert isinstance(results, dict)

    def test_benchmark_device_performance_has_device_type(self):
        """Test that benchmark results include device type."""
        from homodyne.device import benchmark_device_performance

        results = benchmark_device_performance(test_size=100)
        assert "device_type" in results
        assert results["device_type"] == "cpu"

    def test_benchmark_device_performance_has_test_size(self):
        """Test that benchmark results include test size."""
        from homodyne.device import benchmark_device_performance

        test_size = 200
        results = benchmark_device_performance(test_size=test_size)
        assert "test_size" in results
        assert results["test_size"] == test_size


class TestModuleExports:
    """Tests for module exports and API."""

    def test_has_cpu_module_flag(self):
        """Test that HAS_CPU_MODULE flag is defined."""
        assert isinstance(HAS_CPU_MODULE, bool)

    def test_configure_optimal_device_exported(self):
        """Test that configure_optimal_device is exported."""
        assert callable(configure_optimal_device)

    def test_get_device_status_exported(self):
        """Test that get_device_status is exported."""
        assert callable(get_device_status)

    @pytest.mark.skipif(not HAS_CPU_MODULE, reason="CPU module not available")
    def test_cpu_functions_exported_when_available(self):
        """Test that CPU functions are exported when module is available."""
        assert callable(detect_cpu_info)
        assert callable(configure_cpu_hpc)
        assert callable(benchmark_cpu_performance)
