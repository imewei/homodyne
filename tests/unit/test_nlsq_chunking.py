"""
Unit Tests for NLSQ Angle-Stratified Chunking
==============================================

Tests for homodyne/optimization/nlsq/strategies/chunking.py covering:
- TestAngleDistributionAnalysis: Phi angle distribution statistics
- TestStratificationMemory: Memory estimation for stratification
- TestNLSQOptimizationMemory: Complete NLSQ memory estimation
- TestAdaptiveChunkSize: Adaptive chunk size calculation
- TestAngleStratifiedData: Data stratification with full copy
- TestAngleStratifiedIndices: Zero-copy index-based stratification
- TestNumericalCorrectness: Scientific computing validation

These tests verify angle-stratified chunking which fixes per-angle parameter
incompatibility with NLSQ chunking (ultra-think-20251106-012247).

Key Properties Tested:
- All phi angles present in every chunk (prevents zero gradients)
- No data loss or duplication during stratification
- Memory estimation accuracy
- Chunk size bounds enforcement
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from homodyne.optimization.nlsq.strategies.chunking import (
    AngleDistributionStats,
    StratificationDiagnostics,
    StratifiedIndexIterator,
    analyze_angle_distribution,
    calculate_adaptive_chunk_size,
    compute_stratification_diagnostics,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_nlsq_optimization_memory,
    estimate_stratification_memory,
    format_diagnostics_report,
    get_stratified_chunk_iterator,
    should_use_stratification,
)


# =============================================================================
# Test Fixtures
# =============================================================================
@pytest.fixture
def balanced_phi_data():
    """Create balanced phi angle data (equal points per angle)."""
    n_per_angle = 10_000
    angles = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    phi = np.repeat(angles, n_per_angle)
    return phi, angles


@pytest.fixture
def imbalanced_phi_data():
    """Create imbalanced phi angle data (varying points per angle)."""
    # 10:5:1 imbalance ratio
    phi = np.concatenate(
        [
            np.full(10_000, 0.0),
            np.full(5_000, np.pi / 2),
            np.full(1_000, np.pi),
        ]
    )
    return phi


@pytest.fixture
def single_angle_data():
    """Create data with single phi angle."""
    return np.full(10_000, np.pi / 4)


@pytest.fixture
def synthetic_xpcs_data():
    """Create synthetic XPCS-like data for stratification testing."""
    n_per_angle = 5_000
    n_angles = 3
    angles = np.linspace(0, np.pi, n_angles, endpoint=False)

    phi = np.repeat(angles, n_per_angle)
    t1 = np.tile(np.linspace(0, 10, n_per_angle), n_angles)
    t2 = np.tile(np.linspace(0, 10, n_per_angle), n_angles)
    g2 = 1.0 + 0.5 * np.exp(-t1 * 0.1) * np.cos(phi)

    return phi, t1, t2, g2


# =============================================================================
# TestAngleDistributionAnalysis
# =============================================================================
class TestAngleDistributionAnalysis:
    """Tests for analyze_angle_distribution() function."""

    def test_balanced_distribution_is_balanced(self, balanced_phi_data):
        """Balanced data should have is_balanced=True."""
        phi, _ = balanced_phi_data
        stats = analyze_angle_distribution(phi)

        assert stats.is_balanced is True
        assert stats.imbalance_ratio < 5.0

    def test_imbalanced_distribution_detected(self, imbalanced_phi_data):
        """Imbalanced data should have is_balanced=False."""
        stats = analyze_angle_distribution(imbalanced_phi_data)

        assert stats.is_balanced is False
        assert stats.imbalance_ratio >= 5.0  # 10:1 ratio

    def test_unique_angles_count(self, balanced_phi_data):
        """Should correctly count unique angles."""
        phi, angles = balanced_phi_data
        stats = analyze_angle_distribution(phi)

        assert stats.n_angles == len(angles)
        np.testing.assert_array_equal(np.sort(stats.unique_angles), np.sort(angles))

    def test_counts_dict_correct(self, balanced_phi_data):
        """Counts dict should have correct values."""
        phi, angles = balanced_phi_data
        stats = analyze_angle_distribution(phi)

        for angle in angles:
            assert stats.counts[float(angle)] == 10_000

    def test_fractions_sum_to_one(self, balanced_phi_data):
        """Fractions should sum to 1.0."""
        phi, _ = balanced_phi_data
        stats = analyze_angle_distribution(phi)

        total_fraction = sum(stats.fractions.values())
        np.testing.assert_allclose(total_fraction, 1.0, rtol=1e-10)

    def test_min_max_angle_identification(self, imbalanced_phi_data):
        """Should correctly identify min/max count angles."""
        stats = analyze_angle_distribution(imbalanced_phi_data)

        # 0.0 has most points (10k), np.pi has fewest (1k)
        assert stats.max_angle == 0.0
        assert stats.min_angle == np.pi

    def test_single_angle_stats(self, single_angle_data):
        """Single angle should have n_angles=1 and imbalance_ratio=1."""
        stats = analyze_angle_distribution(single_angle_data)

        assert stats.n_angles == 1
        assert stats.imbalance_ratio == 1.0
        assert stats.is_balanced is True

    def test_jax_array_input(self, balanced_phi_data):
        """Should handle JAX arrays as input."""
        import jax.numpy as jnp

        phi, _ = balanced_phi_data
        phi_jax = jnp.array(phi)
        stats = analyze_angle_distribution(phi_jax)

        assert stats.n_angles == 5


# =============================================================================
# TestStratificationMemory
# =============================================================================
class TestStratificationMemory:
    """Tests for estimate_stratification_memory() function."""

    def test_full_copy_peak_memory(self):
        """Full copy should have 2x original memory as peak."""
        result = estimate_stratification_memory(
            n_points=1_000_000, use_index_based=False
        )

        # Peak should be approximately 2x original
        assert result["peak_memory_mb"] >= result["original_memory_mb"] * 1.5

    def test_index_based_lower_memory(self):
        """Index-based stratification should use less memory."""
        full_copy = estimate_stratification_memory(
            n_points=1_000_000, use_index_based=False
        )
        index_based = estimate_stratification_memory(
            n_points=1_000_000, use_index_based=True
        )

        assert index_based["peak_memory_mb"] < full_copy["peak_memory_mb"]

    def test_memory_scales_with_points(self):
        """Memory should scale linearly with points."""
        mem_1m = estimate_stratification_memory(n_points=1_000_000)
        mem_10m = estimate_stratification_memory(n_points=10_000_000)

        ratio = mem_10m["original_memory_mb"] / mem_1m["original_memory_mb"]
        np.testing.assert_allclose(ratio, 10.0, rtol=0.01)

    def test_expansion_factor_affects_memory(self):
        """Expansion factor should increase stratified memory."""
        base = estimate_stratification_memory(
            n_points=1_000_000, estimated_expansion=1.0
        )
        expanded = estimate_stratification_memory(
            n_points=1_000_000, estimated_expansion=2.0
        )

        # With expansion, peak should be higher
        assert expanded["peak_memory_mb"] > base["peak_memory_mb"]


# =============================================================================
# TestNLSQOptimizationMemory
# =============================================================================
class TestNLSQOptimizationMemory:
    """Tests for estimate_nlsq_optimization_memory() function."""

    def test_jacobian_dominates(self):
        """Jacobian should be the dominant memory component."""
        result = estimate_nlsq_optimization_memory(
            n_points=10_000_000, n_params=53
        )

        assert result["jacobian_mb"] > result["data_mb"]
        assert result["jacobian_mb"] > result["jax_overhead_mb"]

    def test_memory_components_present(self):
        """All memory components should be present."""
        result = estimate_nlsq_optimization_memory(
            n_points=1_000_000, n_params=10
        )

        required_keys = [
            "data_mb",
            "jacobian_mb",
            "jax_overhead_mb",
            "optimizer_mb",
            "total_mb",
            "peak_gb",
        ]
        for key in required_keys:
            assert key in result

    def test_safety_margin_applied(self):
        """20% safety margin should be applied to total."""
        result = estimate_nlsq_optimization_memory(
            n_points=1_000_000, n_params=10
        )

        # Total should be ~20% higher than sum of components
        components_sum = (
            result["data_mb"]
            + result["jacobian_mb"]
            + result["jax_overhead_mb"]
            + result["optimizer_mb"]
        )
        expected_with_margin = components_sum * 1.20

        np.testing.assert_allclose(
            result["total_mb"], expected_with_margin, rtol=0.01
        )

    @pytest.mark.parametrize(
        "n_points,n_params",
        [
            (1_000_000, 10),
            (10_000_000, 53),
            (100_000_000, 100),
        ],
    )
    def test_numerical_correctness_jacobian(self, n_points, n_params):
        """Verify Jacobian memory calculation."""
        result = estimate_nlsq_optimization_memory(n_points, n_params)

        # Jacobian: n_points * n_params * 8 bytes / 1024^2
        expected_jacobian_mb = (n_points * n_params * 8) / (1024**2)
        np.testing.assert_allclose(
            result["jacobian_mb"], expected_jacobian_mb, rtol=1e-10
        )


# =============================================================================
# TestAdaptiveChunkSize
# =============================================================================
class TestAdaptiveChunkSize:
    """Tests for calculate_adaptive_chunk_size() function."""

    def test_chunk_size_bounded_by_min(self):
        """Chunk size should not go below minimum."""
        # Very limited memory should still respect minimum
        chunk_size = calculate_adaptive_chunk_size(
            total_points=1_000_000,
            n_params=100,
            n_angles=10,
            available_memory_gb=0.1,  # Very limited
            min_chunk_size=10_000,
        )

        assert chunk_size >= 10_000

    def test_chunk_size_bounded_by_max(self):
        """Chunk size should not exceed maximum."""
        # Large memory should still respect maximum
        chunk_size = calculate_adaptive_chunk_size(
            total_points=1_000_000,
            n_params=5,
            n_angles=3,
            available_memory_gb=1000.0,  # Very large
            max_chunk_size=500_000,
        )

        assert chunk_size <= 500_000

    def test_chunk_size_scales_with_memory(self):
        """More memory should allow larger chunks."""
        chunk_small = calculate_adaptive_chunk_size(
            total_points=10_000_000,
            n_params=50,
            n_angles=10,
            available_memory_gb=8.0,
        )
        chunk_large = calculate_adaptive_chunk_size(
            total_points=10_000_000,
            n_params=50,
            n_angles=10,
            available_memory_gb=64.0,
        )

        assert chunk_large >= chunk_small

    def test_chunk_size_inversely_scales_with_params(self):
        """More parameters should result in smaller chunks."""
        chunk_few_params = calculate_adaptive_chunk_size(
            total_points=10_000_000,
            n_params=10,
            n_angles=5,
            available_memory_gb=32.0,
        )
        chunk_many_params = calculate_adaptive_chunk_size(
            total_points=10_000_000,
            n_params=100,
            n_angles=5,
            available_memory_gb=32.0,
        )

        assert chunk_many_params <= chunk_few_params

    def test_no_angles_handled(self):
        """Zero angles should still return valid chunk size."""
        chunk_size = calculate_adaptive_chunk_size(
            total_points=1_000_000,
            n_params=10,
            n_angles=0,  # Edge case
            available_memory_gb=32.0,
        )

        assert chunk_size >= 10_000


# =============================================================================
# TestAngleStratifiedData
# =============================================================================
class TestAngleStratifiedData:
    """Tests for create_angle_stratified_data() function."""

    def test_no_data_loss(self, synthetic_xpcs_data):
        """Stratification should not lose any data points."""
        phi, t1, t2, g2 = synthetic_xpcs_data
        n_original = len(phi)

        phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=5000
        )

        if chunk_sizes is not None:
            assert sum(chunk_sizes) == n_original
        assert len(phi_s) == n_original

    def test_no_duplicate_data(self, synthetic_xpcs_data):
        """Stratification should not create duplicate indices."""
        phi, t1, t2, g2 = synthetic_xpcs_data

        # Create indices to track
        indices = np.arange(len(phi))
        phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=5000
        )

        # Verify all original values are present (sum should match)
        np.testing.assert_allclose(np.sum(g2), np.sum(g2_s), rtol=1e-10)

    def test_all_angles_in_chunks(self, synthetic_xpcs_data):
        """Each chunk should contain all phi angles."""
        phi, t1, t2, g2 = synthetic_xpcs_data
        n_angles = len(np.unique(phi))

        phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        if chunk_sizes is not None:
            start = 0
            for chunk_size in chunk_sizes:
                chunk_phi = phi_s[start : start + chunk_size]
                chunk_angles = len(np.unique(np.asarray(chunk_phi)))
                # Each chunk should have all angles
                assert chunk_angles == n_angles, (
                    f"Chunk has {chunk_angles} angles, expected {n_angles}"
                )
                start += chunk_size

    def test_single_angle_passthrough(self, single_angle_data):
        """Single angle data should pass through unchanged."""
        phi = single_angle_data
        t1 = np.random.rand(len(phi))
        t2 = np.random.rand(len(phi))
        g2 = np.random.rand(len(phi))

        import jax.numpy as jnp

        phi_jax = jnp.array(phi)
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        g2_jax = jnp.array(g2)

        phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
            phi_jax, t1_jax, t2_jax, g2_jax
        )

        # Should return original (no stratification needed)
        assert chunk_sizes is None
        np.testing.assert_array_equal(np.asarray(phi_s), np.asarray(phi_jax))


# =============================================================================
# TestAngleStratifiedIndices
# =============================================================================
class TestAngleStratifiedIndices:
    """Tests for create_angle_stratified_indices() function."""

    def test_indices_cover_all_points(self, balanced_phi_data):
        """Index array should cover all data points exactly once."""
        phi, _ = balanced_phi_data

        indices, chunk_sizes = create_angle_stratified_indices(
            phi, target_chunk_size=10000
        )

        assert len(indices) == len(phi)
        assert len(np.unique(indices)) == len(phi)  # No duplicates

    def test_indices_are_valid(self, balanced_phi_data):
        """All indices should be valid array indices."""
        phi, _ = balanced_phi_data

        indices, _ = create_angle_stratified_indices(phi, target_chunk_size=10000)

        assert np.all(indices >= 0)
        assert np.all(indices < len(phi))

    def test_chunk_sizes_sum_correctly(self, balanced_phi_data):
        """Chunk sizes should sum to total points."""
        phi, _ = balanced_phi_data

        indices, chunk_sizes = create_angle_stratified_indices(
            phi, target_chunk_size=10000
        )

        assert sum(chunk_sizes) == len(phi)

    def test_stratified_access_all_angles(self, balanced_phi_data):
        """Accessing phi via stratified indices should yield balanced chunks."""
        phi, angles = balanced_phi_data
        n_angles = len(angles)

        indices, chunk_sizes = create_angle_stratified_indices(
            phi, target_chunk_size=5000
        )

        # Check each chunk has all angles
        start = 0
        for chunk_size in chunk_sizes:
            chunk_indices = indices[start : start + chunk_size]
            chunk_phi = phi[chunk_indices]
            unique_angles = len(np.unique(chunk_phi))
            assert unique_angles == n_angles
            start += chunk_size


# =============================================================================
# TestStratifiedIndexIterator
# =============================================================================
class TestStratifiedIndexIterator:
    """Tests for StratifiedIndexIterator class."""

    def test_iterator_length(self, balanced_phi_data):
        """Iterator length should match number of chunks."""
        phi, _ = balanced_phi_data
        indices, chunk_sizes = create_angle_stratified_indices(phi, 10000)

        iterator = StratifiedIndexIterator(indices, chunk_sizes)

        assert len(iterator) == len(chunk_sizes)

    def test_iterator_yields_correct_indices(self, balanced_phi_data):
        """Iterator should yield correct index chunks."""
        phi, _ = balanced_phi_data
        indices, chunk_sizes = create_angle_stratified_indices(phi, 10000)

        iterator = StratifiedIndexIterator(indices, chunk_sizes)

        # Collect all indices
        all_indices = []
        for chunk in iterator:
            all_indices.extend(chunk.tolist())

        np.testing.assert_array_equal(all_indices, indices)

    def test_get_stratified_chunk_iterator(self, balanced_phi_data):
        """get_stratified_chunk_iterator should return valid iterator."""
        phi, _ = balanced_phi_data

        iterator = get_stratified_chunk_iterator(phi, target_chunk_size=10000)

        assert isinstance(iterator, StratifiedIndexIterator)
        assert len(iterator) > 0


# =============================================================================
# TestShouldUseStratification
# =============================================================================
class TestShouldUseStratification:
    """Tests for should_use_stratification() function."""

    def test_small_dataset_no_stratification(self):
        """Small datasets should not need stratification."""
        should, reason = should_use_stratification(
            n_points=50_000,
            n_angles=3,
            per_angle_scaling=True,
            imbalance_ratio=1.5,
        )

        assert should is False
        assert "< 100k" in reason.lower() or "standard" in reason.lower()

    def test_no_per_angle_scaling_no_stratification(self):
        """Without per-angle scaling, no stratification needed."""
        should, reason = should_use_stratification(
            n_points=10_000_000,
            n_angles=5,
            per_angle_scaling=False,
            imbalance_ratio=1.5,
        )

        assert should is False
        assert "per-angle" in reason.lower()

    def test_single_angle_no_stratification(self):
        """Single angle datasets don't need stratification."""
        should, reason = should_use_stratification(
            n_points=10_000_000,
            n_angles=1,
            per_angle_scaling=True,
            imbalance_ratio=1.0,
        )

        assert should is False
        assert "single" in reason.lower()

    def test_high_imbalance_no_stratification(self):
        """Highly imbalanced data should use sequential instead."""
        should, reason = should_use_stratification(
            n_points=10_000_000,
            n_angles=5,
            per_angle_scaling=True,
            imbalance_ratio=10.0,  # > 5.0 threshold
        )

        assert should is False
        assert "imbalance" in reason.lower()

    def test_large_balanced_dataset_uses_stratification(self):
        """Large balanced dataset with per-angle scaling should use stratification."""
        should, reason = should_use_stratification(
            n_points=10_000_000,
            n_angles=5,
            per_angle_scaling=True,
            imbalance_ratio=1.5,  # < 5.0
        )

        assert should is True
        assert "large" in reason.lower() or "balanced" in reason.lower()


# =============================================================================
# TestStratificationDiagnostics
# =============================================================================
class TestStratificationDiagnostics:
    """Tests for compute_stratification_diagnostics() function."""

    def test_diagnostics_chunk_count(self, synthetic_xpcs_data):
        """Diagnostics should report correct chunk count."""
        phi, t1, t2, g2 = synthetic_xpcs_data
        phi_np = np.asarray(phi)

        phi_s, t1_s, t2_s, g2_s, chunk_sizes = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        diagnostics = compute_stratification_diagnostics(
            phi_original=phi_np,
            phi_stratified=np.asarray(phi_s),
            execution_time_ms=100.0,
            target_chunk_size=1000,
            chunk_sizes=chunk_sizes,
        )

        if chunk_sizes is not None:
            assert diagnostics.n_chunks == len(chunk_sizes)

    def test_diagnostics_throughput(self, synthetic_xpcs_data):
        """Diagnostics should compute throughput correctly."""
        phi, t1, t2, g2 = synthetic_xpcs_data
        phi_np = np.asarray(phi)
        n_points = len(phi)
        execution_time_ms = 100.0

        phi_s, _, _, _, chunk_sizes = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        diagnostics = compute_stratification_diagnostics(
            phi_original=phi_np,
            phi_stratified=np.asarray(phi_s),
            execution_time_ms=execution_time_ms,
            chunk_sizes=chunk_sizes,
        )

        expected_throughput = (n_points / execution_time_ms) * 1000
        np.testing.assert_allclose(
            diagnostics.throughput_points_per_sec, expected_throughput, rtol=0.01
        )


# =============================================================================
# TestFormatDiagnosticsReport
# =============================================================================
class TestFormatDiagnosticsReport:
    """Tests for format_diagnostics_report() function."""

    def test_report_contains_sections(self):
        """Report should contain all major sections."""
        diagnostics = StratificationDiagnostics(
            n_chunks=10,
            chunk_sizes=[1000] * 10,
            chunk_balance={"mean": 1000, "std": 0, "min": 1000, "max": 1000, "cv": 0},
            angles_per_chunk=[5] * 10,
            angle_coverage={
                "mean_angles": 5.0,
                "std_angles": 0.0,
                "min_coverage_ratio": 1.0,
                "perfect_coverage_chunks": 10,
            },
            execution_time_ms=100.0,
            memory_overhead_mb=10.0,
            memory_efficiency=0.9,
            throughput_points_per_sec=100_000,
            use_index_based=False,
        )

        report = format_diagnostics_report(diagnostics)

        assert "STRATIFICATION" in report.upper()
        assert "Chunking" in report or "chunks" in report.lower()
        assert "Memory" in report or "memory" in report.lower()
        assert "Performance" in report or "Throughput" in report


# =============================================================================
# TestNumericalCorrectness
# =============================================================================
class TestNumericalCorrectness:
    """Scientific computing validation tests."""

    def test_data_integrity_after_stratification(self, synthetic_xpcs_data):
        """Data values should be preserved exactly after stratification."""
        phi, t1, t2, g2 = synthetic_xpcs_data

        phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        # Sorted values should be identical
        np.testing.assert_allclose(
            np.sort(np.asarray(g2)), np.sort(np.asarray(g2_s)), rtol=1e-15
        )
        np.testing.assert_allclose(
            np.sort(np.asarray(t1)), np.sort(np.asarray(t1_s)), rtol=1e-15
        )

    def test_statistical_properties_preserved(self, synthetic_xpcs_data):
        """Statistical properties should be preserved."""
        phi, t1, t2, g2 = synthetic_xpcs_data

        phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        # Mean should be preserved
        np.testing.assert_allclose(
            np.mean(np.asarray(g2)), np.mean(np.asarray(g2_s)), rtol=1e-12
        )
        # Std should be preserved
        np.testing.assert_allclose(
            np.std(np.asarray(g2)), np.std(np.asarray(g2_s)), rtol=1e-12
        )

    def test_no_nan_or_inf(self, synthetic_xpcs_data):
        """Stratification should not introduce NaN or Inf."""
        phi, t1, t2, g2 = synthetic_xpcs_data

        phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
            phi, t1, t2, g2, target_chunk_size=1000
        )

        assert np.all(np.isfinite(np.asarray(phi_s)))
        assert np.all(np.isfinite(np.asarray(t1_s)))
        assert np.all(np.isfinite(np.asarray(t2_s)))
        assert np.all(np.isfinite(np.asarray(g2_s)))

    def test_memory_formula_analytical(self):
        """Verify memory estimation formulas."""
        n_points = 10_000_000
        n_params = 53
        n_features = 4
        dtype_bytes = 8

        result = estimate_nlsq_optimization_memory(
            n_points, n_params, n_features, dtype_bytes
        )

        # Verify Jacobian: n_points * n_params * dtype_bytes
        expected_jacobian_bytes = n_points * n_params * dtype_bytes
        expected_jacobian_mb = expected_jacobian_bytes / (1024**2)
        np.testing.assert_allclose(
            result["jacobian_mb"], expected_jacobian_mb, rtol=1e-12
        )

        # Verify data: n_points * n_features * dtype_bytes
        expected_data_bytes = n_points * n_features * dtype_bytes
        expected_data_mb = expected_data_bytes / (1024**2)
        np.testing.assert_allclose(result["data_mb"], expected_data_mb, rtol=1e-12)
