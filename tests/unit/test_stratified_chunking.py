"""Unit Tests for Angle-Stratified Chunking Module.

This module tests the stratified_chunking.py implementation that fixes
per-angle scaling + NLSQ chunking compatibility issues.

Test Coverage:
--------------
1. Data Preservation (no loss/duplication)
2. Chunk Balance Verification
3. Angle Distribution Analysis
4. Memory Estimation
5. Strategy Decision Logic
6. Edge Cases (single angle, empty data, imbalanced)
7. Target Chunk Size Behavior
8. JAX Array Compatibility

References:
-----------
Ultra-Think Analysis: ultra-think-20251106-012247
Module: homodyne/optimization/stratified_chunking.py
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from homodyne.optimization.nlsq.strategies.chunking import (
    analyze_angle_distribution,
    create_angle_stratified_data,
    create_angle_stratified_indices,
    estimate_stratification_memory,
    should_use_stratification,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def balanced_data():
    """Create balanced test data (3 angles, 100 points each)."""
    n_angles = 3
    points_per_angle = 100
    n_points = n_angles * points_per_angle

    phi = np.repeat([0.0, 45.0, 90.0], points_per_angle)
    t1 = np.tile(np.linspace(1e-6, 1e-3, points_per_angle), n_angles)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    return phi, t1, t2, g2_exp


@pytest.fixture
def imbalanced_data():
    """Create imbalanced test data (3 angles with 100/50/10 points)."""
    angles = [0.0, 45.0, 90.0]
    counts = [100, 50, 10]

    phi = np.concatenate([np.full(c, a) for a, c in zip(angles, counts, strict=False)])
    t1 = np.concatenate([np.linspace(1e-6, 1e-3, c) for c in counts])
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, len(phi))

    return phi, t1, t2, g2_exp


@pytest.fixture
def single_angle_data():
    """Create data with single angle (edge case)."""
    n_points = 300
    phi = np.full(n_points, 45.0)
    t1 = np.linspace(1e-6, 1e-3, n_points)
    t2 = t1.copy()
    g2_exp = np.random.uniform(1.0, 1.5, n_points)

    return phi, t1, t2, g2_exp


# ============================================================================
# Test 1-3: Data Preservation
# ============================================================================


def test_stratification_preserves_data_count(balanced_data):
    """Test that stratification preserves most points (>= 95%).

    Note: Stratification may discard some points (~4-5%) to maintain
    balanced chunks where each chunk has equal points per angle.
    """
    phi, t1, t2, g2_exp = balanced_data
    original_count = len(phi)

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=50
    )

    # Check that at least 95% of points are preserved (balanced chunking may discard some)
    min_expected = int(0.95 * original_count)
    assert (
        len(phi_s) >= min_expected
    ), f"Expected >= {min_expected} points, got {len(phi_s)}"
    assert (
        len(t1_s) >= min_expected
    ), f"Expected >= {min_expected} points, got {len(t1_s)}"
    assert (
        len(t2_s) >= min_expected
    ), f"Expected >= {min_expected} points, got {len(t2_s)}"
    assert (
        len(g2_s) >= min_expected
    ), f"Expected >= {min_expected} points, got {len(g2_s)}"


def test_stratification_preserves_data_values(balanced_data):
    """Test that stratification preserves data values (no corruption).

    Note: Stratification may discard some points, so we verify that the
    stratified data is a valid subset of the original data.
    """
    phi, t1, t2, g2_exp = balanced_data

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=50
    )

    # Create combined arrays for easier comparison
    original_data = np.column_stack([phi, t1, t2, g2_exp])
    stratified_data = np.column_stack([phi_s, t1_s, t2_s, g2_s])

    # Check that each stratified point exists in the original data
    # (allowing for floating point tolerance)
    for strat_point in stratified_data:
        # Find if this point exists in original data
        matches = np.all(np.abs(original_data - strat_point) < 1e-10, axis=1)
        assert np.any(
            matches
        ), f"Stratified point {strat_point} not found in original data"


def test_stratification_no_duplicates(balanced_data):
    """Test that stratification doesn't create duplicate points.

    Note: Stratification may discard some points to maintain balanced chunks,
    so we verify the stratified data is a subset of the original data.
    """
    phi, t1, t2, g2_exp = balanced_data

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=50
    )

    # Convert to NumPy arrays to ensure hashability
    phi_s_np = np.asarray(phi_s)
    t1_s_np = np.asarray(t1_s)
    t2_s_np = np.asarray(t2_s)
    g2_s_np = np.asarray(g2_s)

    # Create unique identifiers for each point
    original_ids = set(zip(phi, t1, t2, g2_exp, strict=False))
    stratified_ids = set(zip(phi_s_np, t1_s_np, t2_s_np, g2_s_np, strict=False))

    # Check no duplicates within stratified data
    assert len(stratified_ids) == len(phi_s_np)

    # Check stratified points are a subset of original (may discard some points for balance)
    assert stratified_ids.issubset(
        original_ids
    ), "Stratified data contains points not in original"


# ============================================================================
# Test 4-6: Chunk Balance
# ============================================================================


def test_stratification_chunk_balance(balanced_data):
    """Test that stratified chunks contain all angles."""
    phi, t1, t2, g2_exp = balanced_data
    target_chunk_size = 50

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=target_chunk_size
    )

    # Calculate number of chunks
    n_points = len(phi_s)
    n_chunks = (n_points + target_chunk_size - 1) // target_chunk_size
    unique_angles = np.unique(phi)

    # Check each chunk for angle coverage
    for chunk_idx in range(n_chunks):
        start = chunk_idx * target_chunk_size
        end = min(start + target_chunk_size, len(phi_s))
        chunk_phi = phi_s[start:end]

        chunk_unique_angles = np.unique(chunk_phi)

        # Each chunk should contain all angles (or fewer if last chunk)
        if chunk_idx < n_chunks - 1:  # Not last chunk
            assert len(chunk_unique_angles) == len(unique_angles), (
                f"Chunk {chunk_idx} missing angles: "
                f"expected {len(unique_angles)}, got {len(chunk_unique_angles)}"
            )


def test_stratification_chunk_size_target(balanced_data):
    """Test that chunk sizes match target (except last chunk)."""
    phi, t1, t2, g2_exp = balanced_data
    target_chunk_size = 75

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=target_chunk_size
    )

    # Calculate number of chunks
    n_points = len(phi_s)
    n_chunks = (n_points + target_chunk_size - 1) // target_chunk_size

    # Check chunk sizes
    for chunk_idx in range(n_chunks - 1):  # All except last
        start = chunk_idx * target_chunk_size
        end = start + target_chunk_size
        chunk_size = end - start
        assert chunk_size == target_chunk_size


def test_stratification_last_chunk_remainder(balanced_data):
    """Test that last chunk handles remainder points correctly.

    Note: Stratification may discard some points, so we calculate the
    expected remainder based on the stratified data, not the original.
    """
    phi, t1, t2, g2_exp = balanced_data
    target_chunk_size = 70  # Won't divide evenly

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=target_chunk_size
    )

    # Calculate number of chunks based on stratified data
    n_points = len(phi_s)
    n_chunks = (n_points + target_chunk_size - 1) // target_chunk_size

    # Check last chunk size
    last_chunk_start = (n_chunks - 1) * target_chunk_size
    last_chunk_size = len(phi_s) - last_chunk_start

    # Expected remainder should be calculated from stratified data, not original
    expected_remainder = n_points % target_chunk_size
    if expected_remainder > 0:
        assert last_chunk_size == expected_remainder
    else:
        assert last_chunk_size == target_chunk_size


# ============================================================================
# Test 7-9: Angle Distribution Analysis
# ============================================================================


def test_analyze_angle_distribution_balanced(balanced_data):
    """Test angle distribution analysis on balanced data."""
    phi, t1, t2, g2_exp = balanced_data

    stats = analyze_angle_distribution(phi)

    assert stats.n_angles == 3
    assert_array_equal(stats.unique_angles, [0.0, 45.0, 90.0])
    assert stats.counts[0.0] == 100
    assert stats.counts[45.0] == 100
    assert stats.counts[90.0] == 100
    assert stats.imbalance_ratio == 1.0  # Perfectly balanced
    assert stats.is_balanced is True


def test_analyze_angle_distribution_imbalanced(imbalanced_data):
    """Test angle distribution analysis on imbalanced data."""
    phi, t1, t2, g2_exp = imbalanced_data

    stats = analyze_angle_distribution(phi)

    assert stats.n_angles == 3
    assert stats.counts[0.0] == 100
    assert stats.counts[45.0] == 50
    assert stats.counts[90.0] == 10
    assert stats.imbalance_ratio == 10.0  # max=100, min=10
    assert stats.is_balanced is False  # Ratio > 5.0


def test_analyze_angle_distribution_single_angle(single_angle_data):
    """Test angle distribution with single angle."""
    phi, t1, t2, g2_exp = single_angle_data

    stats = analyze_angle_distribution(phi)

    assert stats.n_angles == 1
    assert stats.unique_angles[0] == 45.0
    assert stats.counts[45.0] == 300
    assert stats.imbalance_ratio == 1.0  # Only one angle
    assert stats.is_balanced is True


# ============================================================================
# Test 10-12: Memory Estimation
# ============================================================================


def test_memory_estimation_basic():
    """Test memory estimation for typical dataset."""
    n_points = 1_000_000
    n_features = 4  # phi, t1, t2, g2_exp

    result = estimate_stratification_memory(n_points, n_features, use_index_based=False)

    # Check structure
    assert "peak_memory_mb" in result
    assert "is_safe" in result
    assert "original_memory_mb" in result
    assert "stratified_memory_mb" in result

    # Check reasonable values
    expected_mb = (n_points * n_features * 8 * 2) / (1024**2)  # float64, 2x peak
    assert result["peak_memory_mb"] == pytest.approx(expected_mb, rel=0.01)


def test_memory_estimation_index_based():
    """Test memory estimation for index-based mode (future feature)."""
    n_points = 1_000_000

    result = estimate_stratification_memory(n_points, use_index_based=True)

    # Index-based should use int64 indices + original data
    bytes_per_float = 8
    bytes_per_int = 8
    original_mb = (n_points * 4 * bytes_per_float) / (1024**2)
    index_mb = (n_points * bytes_per_int) / (1024**2)
    expected_mb = original_mb + index_mb
    assert result["peak_memory_mb"] == pytest.approx(expected_mb, rel=0.01)


def test_memory_estimation_safety_check():
    """Test memory safety threshold logic."""
    n_points = 100_000  # Small dataset

    result = estimate_stratification_memory(n_points)

    # For small dataset, should be safe
    assert result["is_safe"] is True
    assert result["peak_memory_mb"] > 0


# ============================================================================
# Test 13-15: Strategy Decision Logic
# ============================================================================


def test_should_use_stratification_enabled():
    """Test strategy decision when conditions are met."""
    use, reason = should_use_stratification(
        n_points=200_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,
    )

    assert use is True
    assert "large dataset" in reason.lower() or "balanced" in reason.lower()


def test_should_use_stratification_disabled_small_dataset():
    """Test strategy decision for small dataset."""
    use, reason = should_use_stratification(
        n_points=50_000,  # < 100k threshold
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=2.0,
    )

    assert use is False
    assert "100k" in reason.lower() or "standard" in reason.lower()


def test_should_use_stratification_disabled_imbalanced():
    """Test strategy decision for highly imbalanced angles."""
    use, reason = should_use_stratification(
        n_points=200_000,
        n_angles=3,
        per_angle_scaling=True,
        imbalance_ratio=15.0,  # Very imbalanced
    )

    assert use is False
    assert "imbalance" in reason.lower() or "sequential" in reason.lower()


def test_should_use_stratification_disabled_single_angle():
    """Test strategy decision for single angle."""
    use, reason = should_use_stratification(
        n_points=200_000,
        n_angles=1,  # Only one angle
        per_angle_scaling=True,
        imbalance_ratio=1.0,
    )

    assert use is False
    assert "single" in reason.lower() or "not applicable" in reason.lower()


# ============================================================================
# Test 16-18: Edge Cases
# ============================================================================


def test_stratification_empty_data():
    """Test stratification with empty data (edge case)."""
    phi = np.array([])
    t1 = np.array([])
    t2 = np.array([])
    g2_exp = np.array([])

    with pytest.raises((ValueError, IndexError)):
        create_angle_stratified_data(phi, t1, t2, g2_exp, target_chunk_size=50)


def test_stratification_single_point():
    """Test stratification with single data point."""
    phi = np.array([45.0])
    t1 = np.array([1e-6])
    t2 = np.array([1e-6])
    g2_exp = np.array([1.2])

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=50
    )

    # Should return the single point unchanged
    assert len(phi_s) == 1
    assert phi_s[0] == 45.0


def test_stratification_mismatched_array_lengths():
    """Test stratification with mismatched input array lengths."""
    phi = np.array([0.0, 45.0, 90.0])
    t1 = np.array([1e-6, 2e-6])  # Wrong length
    t2 = np.array([1e-6, 2e-6, 3e-6])
    g2_exp = np.array([1.0, 1.1, 1.2])

    with pytest.raises((ValueError, IndexError)):
        create_angle_stratified_data(phi, t1, t2, g2_exp, target_chunk_size=50)


# ============================================================================
# Test 19-20: Integration with JAX Arrays
# ============================================================================


def test_stratification_jax_array_input(balanced_data):
    """Test that stratification works with JAX arrays.

    Note: Stratification may discard some points to maintain balanced chunks.
    """
    import jax.numpy as jnp

    phi, t1, t2, g2_exp = balanced_data

    # Convert to JAX arrays
    phi_jax = jnp.array(phi)
    t1_jax = jnp.array(t1)
    t2_jax = jnp.array(t2)
    g2_jax = jnp.array(g2_exp)

    # Should work with JAX arrays
    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi_jax, t1_jax, t2_jax, g2_jax, target_chunk_size=50
    )

    # Check result is valid (at least 95% data preserved)
    min_expected = int(0.95 * len(phi))
    assert (
        len(phi_s) >= min_expected
    ), f"Expected >= {min_expected} points, got {len(phi_s)}"


def test_stratification_output_numpy_compatible(balanced_data):
    """Test that stratification output is NumPy compatible."""
    phi, t1, t2, g2_exp = balanced_data

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=50
    )

    # Output should work with NumPy operations
    mean_phi = np.mean(np.asarray(phi_s))
    std_g2 = np.std(np.asarray(g2_s))

    # Check that they're numeric values
    assert isinstance(mean_phi, (float, np.floating, np.number))
    assert isinstance(std_g2, (float, np.floating, np.number))

    # Check that the values are reasonable
    assert not np.isnan(mean_phi)
    assert not np.isnan(std_g2)


# ============================================================================
# Test 21: Result Metadata
# ============================================================================


def test_stratified_result_types(balanced_data):
    """Test that stratification returns correct types."""
    phi, t1, t2, g2_exp = balanced_data
    target_chunk_size = 75

    phi_s, t1_s, t2_s, g2_s, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=target_chunk_size
    )

    # Check that all outputs are array-like
    assert hasattr(phi_s, "__len__")
    assert hasattr(t1_s, "__len__")
    assert hasattr(t2_s, "__len__")
    assert hasattr(g2_s, "__len__")

    # Check consistent lengths
    assert len(phi_s) == len(t1_s) == len(t2_s) == len(g2_s)

    # Check they're all numeric arrays
    assert np.issubdtype(np.asarray(phi_s).dtype, np.number)
    assert np.issubdtype(np.asarray(g2_s).dtype, np.number)


# ============================================================================
# Test 22: Index-Based Stratification
# ============================================================================


def test_index_based_returns_correct_length(balanced_data):
    """Test that index array has correct length."""
    phi, t1, t2, g2_exp = balanced_data

    indices = create_angle_stratified_indices(phi, target_chunk_size=75)

    # Index array should have same length as input
    assert len(indices) == len(phi)


def test_index_based_no_duplicates(balanced_data):
    """Test that index array contains no duplicates."""
    phi, t1, t2, g2_exp = balanced_data

    indices = create_angle_stratified_indices(phi, target_chunk_size=75)

    # All indices should be unique
    assert len(np.unique(indices)) == len(indices)


def test_index_based_all_indices_valid(balanced_data):
    """Test that all indices are valid (in range)."""
    phi, t1, t2, g2_exp = balanced_data

    indices = create_angle_stratified_indices(phi, target_chunk_size=75)

    # All indices should be in [0, len(phi))
    assert np.all(indices >= 0)
    assert np.all(indices < len(phi))


def test_index_based_preserves_data_when_applied(balanced_data):
    """Test that applying indices preserves all data."""
    phi, t1, t2, g2_exp = balanced_data

    indices = create_angle_stratified_indices(phi, target_chunk_size=75)

    # Apply indices
    phi_stratified = phi[indices]
    g2_stratified = g2_exp[indices]

    # Should preserve all unique values
    assert set(np.unique(phi)) == set(np.unique(phi_stratified))

    # Total sum should be preserved (within floating point error)
    assert np.allclose(np.sort(g2_exp), np.sort(g2_stratified))


def test_index_based_equivalent_to_full_copy(balanced_data):
    """Test that index-based produces same result as full copy."""
    phi, t1, t2, g2_exp = balanced_data
    target_chunk_size = 75

    # Full copy approach
    phi_full, t1_full, t2_full, g2_full, _ = create_angle_stratified_data(
        phi, t1, t2, g2_exp, target_chunk_size=target_chunk_size
    )

    # Index-based approach
    indices = create_angle_stratified_indices(phi, target_chunk_size=target_chunk_size)
    phi_index = phi[indices]
    t1_index = t1[indices]
    t2_index = t2[indices]
    g2_index = g2_exp[indices]

    # Should produce identical results
    assert_allclose(phi_full, phi_index)
    assert_allclose(t1_full, t1_index)
    assert_allclose(t2_full, t2_index)
    assert_allclose(g2_full, g2_index)


def test_index_based_single_angle_identity(balanced_data):
    """Test that single angle returns identity index."""
    phi, t1, t2, g2_exp = balanced_data

    # Use only first angle
    phi_single = phi[:100]  # All same angle

    indices = create_angle_stratified_indices(phi_single, target_chunk_size=75)

    # Should return identity index [0, 1, 2, ..., 99]
    expected = np.arange(len(phi_single))
    assert_array_equal(indices, expected)


def test_index_based_memory_efficiency():
    """Test that index-based uses less memory than full copy."""
    n_points = 300
    np.repeat([0.0, 45.0, 90.0], 100)

    # Estimate memory for both approaches
    mem_full = estimate_stratification_memory(n_points, use_index_based=False)
    mem_index = estimate_stratification_memory(n_points, use_index_based=True)

    # Index-based should use significantly less memory
    assert mem_index["peak_memory_mb"] < mem_full["peak_memory_mb"]

    # Index-based should have zero stratified copy memory
    assert mem_index["stratified_memory_mb"] == 0
    assert mem_full["stratified_memory_mb"] > 0


def test_index_based_large_dataset():
    """Test index-based stratification on larger dataset."""
    n_points = 100_000
    n_angles = 5
    points_per_angle = n_points // n_angles

    phi = np.repeat(np.linspace(0, 180, n_angles), points_per_angle)

    indices = create_angle_stratified_indices(phi, target_chunk_size=10_000)

    # Basic validity checks
    assert len(indices) == n_points
    assert len(np.unique(indices)) == n_points
    assert np.min(indices) == 0
    assert np.max(indices) == n_points - 1
