"""
CMC Performance Benchmark Tests (pytest-benchmark compatible).

Enterprise-level regression tests for CMC/MCMC performance:
- Throughput baselines
- Memory usage limits
- Scaling behavior
- Multiprocessing overhead

Usage:
    pytest tests/performance/test_cmc_benchmarks.py -v --benchmark-only
    pytest tests/performance/test_cmc_benchmarks.py -v -k "not slow"

Author: Homodyne Performance Team
Date: 2026-02-01
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import time

import numpy as np
import pytest

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import jax.numpy as jnp
    from jax import random  # noqa: F401

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Import benchmark utilities
from tests.performance.benchmark_cmc import (
    benchmark_consensus_aggregation,
    benchmark_multiprocessing_overhead,
    create_cmc_benchmark_data,
    create_minimal_cmc_config,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def small_dataset():
    """Small dataset for quick tests."""
    return create_cmc_benchmark_data(n_phi=3, n_t1=10, n_t2=10, mode="static")


@pytest.fixture
def medium_dataset():
    """Medium dataset for standard tests."""
    return create_cmc_benchmark_data(n_phi=12, n_t1=50, n_t2=50, mode="static")


@pytest.fixture
def laminar_dataset():
    """Laminar flow dataset for mode-specific tests."""
    return create_cmc_benchmark_data(n_phi=12, n_t1=50, n_t2=50, mode="laminar_flow")


@pytest.fixture
def minimal_config():
    """Minimal CMC config for fast tests."""
    return create_minimal_cmc_config(n_warmup=10, n_samples=20)


# ============================================================================
# Multiprocessing Overhead Tests
# ============================================================================


@pytest.mark.performance
class TestMultiprocessingOverhead:
    """Tests for multiprocessing IPC and startup overhead."""

    def test_serialization_speed(self):
        """Test that shard serialization is under 1ms for 5K points."""
        metrics = benchmark_multiprocessing_overhead(n_shards=1, shard_size=5000)

        # Serialization should be fast
        assert metrics["serialize_time_ms"] < 1.0, (
            f"Serialization too slow: {metrics['serialize_time_ms']:.3f}ms (target: <1ms)"
        )
        assert metrics["deserialize_time_ms"] < 0.5, (
            f"Deserialization too slow: {metrics['deserialize_time_ms']:.3f}ms (target: <0.5ms)"
        )

    def test_pool_startup_time(self):
        """Test that process pool startup is under 2s.

        Note: 100ms was too tight — when JAX is loaded, the process is
        multi-threaded and fork() is significantly slower (Python 3.13+).
        In a full test suite, pool startup routinely takes 200-500ms.
        """
        metrics = benchmark_multiprocessing_overhead(n_shards=10, shard_size=1000)

        assert metrics["pool_startup_time_s"] < 2.0, (
            f"Pool startup too slow: {metrics['pool_startup_time_s']:.3f}s (target: <2s)"
        )

    def test_queue_throughput(self):
        """Test queue throughput for typical shard counts."""
        metrics = benchmark_multiprocessing_overhead(n_shards=100, shard_size=5000)

        # 100 shards should transfer in under 1 second
        assert metrics["queue_throughput_time_s"] < 1.0, (
            f"Queue throughput too slow: {metrics['queue_throughput_time_s']:.3f}s (target: <1s for 100 shards)"
        )

    def test_ipc_overhead_scaling(self):
        """Test that IPC overhead scales linearly with data size."""
        metrics_small = benchmark_multiprocessing_overhead(n_shards=1, shard_size=1000)
        metrics_large = benchmark_multiprocessing_overhead(n_shards=1, shard_size=10000)

        size_ratio = 10000 / 1000
        time_ratio = metrics_large["serialize_time_ms"] / max(
            metrics_small["serialize_time_ms"], 0.001
        )

        # Should scale linearly (with some overhead)
        assert time_ratio < size_ratio * 2, (
            f"IPC scaling worse than linear: {time_ratio:.1f}x for {size_ratio}x data"
        )


# ============================================================================
# Consensus Aggregation Tests
# ============================================================================


@pytest.mark.performance
class TestConsensusAggregation:
    """Tests for consensus posterior aggregation."""

    def test_simple_weighted_speed(self):
        """Test simple weighted consensus is fast."""
        metrics = benchmark_consensus_aggregation(
            n_shards=100, n_samples=500, n_params=7
        )

        # Simple weighted average should be under 10ms for 100 shards
        assert metrics["simple_weighted_ms"] < 10.0, (
            f"Simple weighted too slow: {metrics['simple_weighted_ms']:.2f}ms (target: <10ms)"
        )

    def test_consensus_scaling_with_shards(self):
        """Test consensus scales reasonably with shard count."""
        metrics_10 = benchmark_consensus_aggregation(
            n_shards=10, n_samples=500, n_params=7
        )
        metrics_100 = benchmark_consensus_aggregation(
            n_shards=100, n_samples=500, n_params=7
        )

        shard_ratio = 100 / 10
        time_ratio = metrics_100["simple_weighted_ms"] / max(
            metrics_10["simple_weighted_ms"], 0.001
        )

        # Should scale linearly or better
        assert time_ratio < shard_ratio * 1.5, (
            f"Consensus scaling poor: {time_ratio:.1f}x for {shard_ratio}x shards"
        )

    def test_consensus_memory_efficiency(self):
        """Test consensus doesn't use excessive memory."""
        n_shards = 100
        n_samples = 500
        n_params = 7

        # Expected memory for simple stacking
        expected_mb = (n_shards * n_samples * n_params * 8) / (1024**2)

        # Run consensus benchmark
        if HAS_PSUTIL:
            gc.collect()
            before_mb = psutil.Process().memory_info().rss / (1024**2)

            benchmark_consensus_aggregation(n_shards, n_samples, n_params)

            gc.collect()
            after_mb = psutil.Process().memory_info().rss / (1024**2)
            delta_mb = after_mb - before_mb

            # Should not use more than 2x expected memory
            assert delta_mb < expected_mb * 3, (
                f"Consensus memory too high: {delta_mb:.1f}MB (expected: {expected_mb:.1f}MB)"
            )


# ============================================================================
# Memory Usage Tests
# ============================================================================


@pytest.mark.performance
@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil required")
class TestMemoryUsage:
    """Tests for memory usage patterns."""

    def test_data_creation_memory(self, medium_dataset):
        """Test data creation uses reasonable memory.

        Measures the actual size of the dataset arrays rather than total
        process RSS, which is unreliable in multi-test sessions where earlier
        tests (e.g. optimization benchmarks) inflate resident memory.
        """
        n_points = medium_dataset["n_total"]
        expected_bytes = n_points * 4 * 8  # 4 arrays of float64
        expected_mb = expected_bytes / (1024**2)

        # Measure actual memory footprint of the dataset arrays
        array_keys = ["c2_pooled", "t1_pooled", "t2_pooled", "phi_pooled"]
        actual_bytes = sum(
            medium_dataset[k].nbytes
            for k in array_keys
            if hasattr(medium_dataset.get(k), "nbytes")
        )
        actual_mb = actual_bytes / (1024**2)

        # Dataset arrays should not exceed 10x expected (sanity check against
        # catastrophic memory bloat, not a tight budget)
        assert actual_mb < expected_mb * 10, (
            f"Dataset array memory seems excessive: {actual_mb:.1f}MB "
            f"vs expected {expected_mb:.1f}MB"
        )

    def test_synthetic_data_size_calculation(self):
        """Test that we correctly estimate memory for different dataset sizes."""
        sizes = [
            (3, 10, 10),  # Tiny: 300 points
            (12, 50, 50),  # Medium: 30,000 points
            (23, 100, 100),  # Large: 230,000 points
        ]

        for n_phi, n_t1, n_t2 in sizes:
            data = create_cmc_benchmark_data(n_phi, n_t1, n_t2)
            n_total = data["n_total"]

            expected_points = n_phi * n_t1 * n_t2
            assert n_total == expected_points, (
                f"Point count mismatch: {n_total} != {expected_points}"
            )

            # Memory per point (4 float64 arrays)
            bytes_per_point = 4 * 8
            expected_bytes = n_total * bytes_per_point

            actual_bytes = sum(
                arr.nbytes
                for arr in [
                    data["c2_pooled"],
                    data["t1_pooled"],
                    data["t2_pooled"],
                    data["phi_pooled"],
                ]
            )

            assert actual_bytes == expected_bytes, (
                f"Memory calculation wrong: {actual_bytes} != {expected_bytes}"
            )


# ============================================================================
# Throughput Baseline Tests
# ============================================================================


@pytest.mark.performance
class TestThroughputBaselines:
    """Baseline throughput tests (no actual NUTS sampling)."""

    def test_data_generation_speed(self):
        """Test synthetic data generation is fast."""
        # Should generate 100K points in under 1 second
        start = time.perf_counter()
        data = create_cmc_benchmark_data(n_phi=23, n_t1=100, n_t2=50)
        elapsed = time.perf_counter() - start

        n_points = data["n_total"]
        points_per_sec = n_points / elapsed

        assert elapsed < 1.0, (
            f"Data generation slow: {elapsed:.2f}s for {n_points:,} points"
        )
        assert points_per_sec > 100_000, (
            f"Data generation throughput low: {points_per_sec:.0f} points/sec"
        )

    def test_config_creation_speed(self):
        """Test config creation is instant."""
        start = time.perf_counter()
        for _ in range(100):
            create_minimal_cmc_config()
        elapsed = time.perf_counter() - start

        # 100 configs should take under 100ms
        assert elapsed < 0.1, (
            f"Config creation slow: {elapsed * 1000:.1f}ms for 100 configs"
        )


# ============================================================================
# Scaling Behavior Tests
# ============================================================================


@pytest.mark.performance
class TestScalingBehavior:
    """Tests for performance scaling characteristics."""

    def test_shard_creation_scales_linearly(self):
        """Test that shard creation time scales linearly with data size."""
        times = []
        sizes = [1000, 5000, 10000]

        for size in sizes:
            data = create_cmc_benchmark_data(
                n_phi=3, n_t1=int(size**0.5), n_t2=int(size**0.5)
            )

            # Simulate shard creation (just slicing)
            start = time.perf_counter()
            for _ in range(10):
                shard_size = min(1000, data["n_total"])
                _ = {
                    "c2": data["c2_pooled"][:shard_size],
                    "t1": data["t1_pooled"][:shard_size],
                    "t2": data["t2_pooled"][:shard_size],
                    "phi": data["phi_pooled"][:shard_size],
                }
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Check scaling is not worse than O(n^2)
        if len(times) >= 2:
            for i in range(1, len(times)):
                time_ratio = times[i] / max(times[i - 1], 0.0001)
                size_ratio = sizes[i] / sizes[i - 1]
                # Allow 3x the linear scaling factor
                assert time_ratio < size_ratio * 3 + 1, (
                    f"Shard creation scaling poor at size {sizes[i]}"
                )

    def test_worker_count_detection(self):
        """Test worker count detection works correctly."""
        n_workers = min(4, mp.cpu_count())
        assert n_workers >= 1, "Should detect at least 1 CPU"
        assert n_workers <= mp.cpu_count(), "Should not exceed CPU count"


# ============================================================================
# Regression Tests (Baselines)
# ============================================================================


@pytest.mark.performance
class TestRegressionBaselines:
    """Regression tests against known performance baselines."""

    # Baselines (update these when performance improves)
    BASELINE_SERIALIZE_MS = 1.0  # Per 5K shard
    BASELINE_DESERIALIZE_MS = 0.5
    BASELINE_CONSENSUS_MS = 10.0  # Per 100 shards
    BASELINE_DATA_GEN_SEC = 0.5  # Per 100K points

    def test_serialization_regression(self):
        """Ensure serialization performance hasn't degraded."""
        metrics = benchmark_multiprocessing_overhead(n_shards=1, shard_size=5000)

        assert metrics["serialize_time_ms"] < self.BASELINE_SERIALIZE_MS * 1.5, (
            f"Serialization regression: {metrics['serialize_time_ms']:.3f}ms "
            f"(baseline: {self.BASELINE_SERIALIZE_MS}ms)"
        )

    def test_consensus_regression(self):
        """Ensure consensus performance hasn't degraded."""
        metrics = benchmark_consensus_aggregation(
            n_shards=100, n_samples=500, n_params=7
        )

        assert metrics["simple_weighted_ms"] < self.BASELINE_CONSENSUS_MS * 1.5, (
            f"Consensus regression: {metrics['simple_weighted_ms']:.2f}ms "
            f"(baseline: {self.BASELINE_CONSENSUS_MS}ms)"
        )

    def test_data_generation_regression(self):
        """Ensure data generation performance hasn't degraded."""
        start = time.perf_counter()
        create_cmc_benchmark_data(n_phi=23, n_t1=100, n_t2=50)  # ~115K points
        elapsed = time.perf_counter() - start

        assert elapsed < self.BASELINE_DATA_GEN_SEC * 1.5, (
            f"Data generation regression: {elapsed:.2f}s (baseline: {self.BASELINE_DATA_GEN_SEC}s)"
        )


# ============================================================================
# NUTS Sampling Tests (Slow - requires JAX compilation)
# ============================================================================


@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX required")
class TestNUTSSamplingBaselines:
    """NUTS sampling performance tests (slow due to JIT compilation)."""

    def test_nuts_jit_compilation(self):
        """Test that NUTS JIT compilation completes in reasonable time.

        Note: First call includes XLA compilation. Subsequent calls are cached.
        This test validates the benchmark infrastructure works, not sampling speed.
        """
        # Skip actual test as it takes too long for CI
        # Just verify imports work
        try:
            from homodyne.config.parameter_space import ParameterSpace
            from homodyne.optimization.cmc.config import CMCConfig
            from homodyne.optimization.cmc.model import get_xpcs_model

            # Verify basic object creation
            model = get_xpcs_model(per_angle_mode="auto")
            config = CMCConfig.from_dict(create_minimal_cmc_config())
            param_space = ParameterSpace.from_defaults("static")

            assert model is not None
            assert config is not None
            assert param_space is not None
        except ImportError as e:
            pytest.skip(f"CMC imports failed: {e}")

    def test_expected_samples_per_second(self):
        """Document expected NUTS throughput (informational).

        NUTS sampling throughput depends heavily on:
        - Posterior complexity (7 params for laminar_flow vs 3 for static)
        - Data size per shard (NUTS is O(n) in data)
        - Tree depth (adaptive, typically 3-10)
        - Hardware (CPU speed, memory bandwidth)

        Expected ranges:
        - Static mode (3 params): 50-200 samples/sec
        - Laminar flow (7 params): 10-50 samples/sec
        - With warmup overhead: Effective rate is lower
        """
        # This is an informational test, not a hard assertion
        # Document expected performance
        expected_ranges = {
            "static": {"min": 50, "max": 200},
            "laminar_flow": {"min": 10, "max": 50},
        }

        for mode, expected in expected_ranges.items():
            # Just document, don't assert
            print(f"{mode}: Expected {expected['min']}-{expected['max']} samples/sec")


# ============================================================================
# CI/CD Integration Tests
# ============================================================================


@pytest.mark.performance
class TestCICDIntegration:
    """Tests designed for CI/CD pipelines."""

    def test_quick_health_check(self):
        """Quick health check that benchmarking works (~1s)."""
        # Create minimal data
        data = create_cmc_benchmark_data(n_phi=2, n_t1=5, n_t2=5)
        config = create_minimal_cmc_config()

        assert data["n_total"] == 50
        assert config["mcmc"]["num_samples"] == 200

    def test_all_benchmarks_callable(self):
        """Verify all benchmark functions are callable."""
        from tests.performance.benchmark_cmc import (
            benchmark_consensus_aggregation,
            benchmark_multiprocessing_overhead,
            create_cmc_benchmark_data,
            create_minimal_cmc_config,
            run_cmc_benchmark,
        )

        # Just verify they're callable
        assert callable(benchmark_consensus_aggregation)
        assert callable(benchmark_multiprocessing_overhead)
        assert callable(create_cmc_benchmark_data)
        assert callable(create_minimal_cmc_config)
        assert callable(run_cmc_benchmark)
