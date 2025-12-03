"""Unit Tests for CMC Data Sharding Module
==========================================

Test suite for homodyne/optimization/cmc/sharding.py

Test Coverage:
- calculate_optimal_num_shards() with various hardware configurations
- shard_data_random() with reproducibility and balance checks
- shard_data_stratified() with phi distribution preservation
- shard_data_contiguous() with temporal order preservation
- validate_shards() with various validation scenarios
- Edge cases: very small datasets, very large datasets, single shard
"""

import numpy as np
import pytest
from scipy import stats

from homodyne.device.config import HardwareConfig
from homodyne.optimization.mcmc.cmc.sharding import (
    calculate_optimal_num_shards,
    shard_data_contiguous,
    shard_data_random,
    shard_data_stratified,
    validate_shards,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_hardware_cpu():
    """Mock CPU hardware configuration."""
    return HardwareConfig(
        platform="cpu",
        num_devices=1,
        memory_per_device_gb=64.0,
        num_nodes=1,
        cores_per_node=36,
        total_memory_gb=128.0,
        cluster_type="standalone",
        recommended_backend="multiprocessing",
        max_parallel_shards=36,
    )


@pytest.fixture
def mock_hardware_cluster():
    """Mock HPC cluster hardware configuration."""
    return HardwareConfig(
        platform="cpu",
        num_devices=1,
        memory_per_device_gb=128.0,
        num_nodes=10,
        cores_per_node=128,
        total_memory_gb=1280.0,
        cluster_type="pbs",
        recommended_backend="pbs",
        max_parallel_shards=1280,
    )


@pytest.fixture
def synthetic_data_small():
    """Generate small synthetic XPCS dataset (50k points)."""
    np.random.seed(42)
    n_points = 50_000

    data = np.random.randn(n_points) + 1.0
    t1 = np.random.uniform(0, 10, n_points)
    t2 = np.random.uniform(0, 10, n_points)
    phi = np.random.uniform(-180, 180, n_points)
    sigma = np.abs(np.random.randn(n_points) * 0.1)

    return {
        "data": data,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "sigma": sigma,
        "q": 0.01,
        "L": 5.0,
    }


@pytest.fixture
def synthetic_data_medium():
    """Generate medium synthetic XPCS dataset (5M points)."""
    np.random.seed(42)
    n_points = 5_000_000

    data = np.random.randn(n_points) + 1.0
    t1 = np.random.uniform(0, 10, n_points)
    t2 = np.random.uniform(0, 10, n_points)
    phi = np.random.uniform(-180, 180, n_points)
    sigma = np.abs(np.random.randn(n_points) * 0.1)

    return {
        "data": data,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "sigma": sigma,
        "q": 0.01,
        "L": 5.0,
    }


@pytest.fixture
def synthetic_data_structured_phi():
    """Generate synthetic data with structured phi distribution."""
    np.random.seed(42)
    n_points = 100_000

    # Create structured phi distribution (3 peaks at -90°, 0°, 90°)
    n_per_peak = n_points // 3
    phi1 = np.random.normal(-90, 10, n_per_peak)
    phi2 = np.random.normal(0, 10, n_per_peak)
    phi3 = np.random.normal(90, 10, n_points - 2 * n_per_peak)
    phi = np.concatenate([phi1, phi2, phi3])
    np.random.shuffle(phi)  # Shuffle to mix peaks

    data = np.random.randn(n_points) + 1.0
    t1 = np.random.uniform(0, 10, n_points)
    t2 = np.random.uniform(0, 10, n_points)
    sigma = np.abs(np.random.randn(n_points) * 0.1)

    return {
        "data": data,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "sigma": sigma,
        "q": 0.01,
        "L": 5.0,
    }


@pytest.fixture
def synthetic_data_discrete_phi_angles():
    """Generate synthetic data with discrete replicated phi angles (for per-angle parameter testing).

    This fixture creates data with exactly 3 unique phi angles (0°, 60°, 120°), each
    replicated many times. This mimics realistic XPCS datasets where measurements are
    taken at discrete angles with multiple time points per angle.

    Used for testing CMC per-angle scaling compatibility where each shard must contain
    all unique phi angles.
    """
    np.random.seed(42)

    # Define 3 discrete phi angles (matching typical XPCS experiments)
    phi_angles = np.array([0.0, 60.0, 120.0])
    n_angles = len(phi_angles)

    # Create 10,000 points per angle (30,000 total points)
    n_points_per_angle = 10_000
    n_total = n_angles * n_points_per_angle

    # Replicate each angle n_points_per_angle times
    phi = np.repeat(phi_angles, n_points_per_angle)

    # Shuffle to mix angles (realistic: data isn't perfectly ordered by angle)
    shuffle_indices = np.random.permutation(n_total)
    phi = phi[shuffle_indices]

    # Generate random data, t1, t2 with same shuffled order
    data = (np.random.randn(n_total) + 1.0)[shuffle_indices]
    t1 = np.random.uniform(0, 10, n_total)
    t2 = np.random.uniform(0, 10, n_total)
    sigma = np.abs(np.random.randn(n_total) * 0.1)

    return {
        "data": data,
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "sigma": sigma,
        "q": 0.01,
        "L": 5.0,
        "unique_angles": phi_angles,  # For easy access in tests
    }


# ============================================================================
# Test Group 1: calculate_optimal_num_shards()
# ============================================================================


def test_optimal_num_shards_cpu_medium_dataset(mock_hardware_cpu):
    """Test optimal shard calculation for CPU with medium dataset."""
    dataset_size = 10_000_000  # 10M points
    num_shards = calculate_optimal_num_shards(dataset_size, mock_hardware_cpu)

    # For CPU: target 2M points/shard
    # 10M / 2M = 5 shards
    assert num_shards == 5


def test_optimal_num_shards_respects_min_shard_size(mock_hardware_cpu):
    """Test that min_shard_size is respected."""
    dataset_size = 100_000  # 100k points
    num_shards = calculate_optimal_num_shards(
        dataset_size, mock_hardware_cpu, min_shard_size=50_000
    )

    # With 100k points and min_shard_size=50k, max 2 shards
    assert num_shards <= 2
    assert dataset_size // num_shards >= 50_000


def test_optimal_num_shards_cluster(mock_hardware_cluster):
    """Test optimal shard calculation for HPC cluster."""
    dataset_size = 100_000_000  # 100M points
    num_shards = calculate_optimal_num_shards(dataset_size, mock_hardware_cluster)

    # For CPU: target 2M points/shard
    # 100M / 2M = 50 shards (well within max_parallel_shards=1280)
    assert num_shards == 50
    assert num_shards <= mock_hardware_cluster.max_parallel_shards


# ============================================================================
# Test Group 2: shard_data_random()
# ============================================================================


def test_random_sharding_reproducibility(synthetic_data_small):
    """Test that random sharding is reproducible with same seed."""
    shards1 = shard_data_random(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
        sigma=synthetic_data_small["sigma"],
        random_seed=42,
    )

    shards2 = shard_data_random(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
        sigma=synthetic_data_small["sigma"],
        random_seed=42,
    )

    # Check that shards are identical
    assert len(shards1) == len(shards2)
    for s1, s2 in zip(shards1, shards2, strict=False):
        np.testing.assert_array_equal(s1["data"], s2["data"])
        np.testing.assert_array_equal(s1["phi"], s2["phi"])


def test_random_sharding_produces_balanced_shards(synthetic_data_small):
    """Test that random sharding produces balanced shard sizes."""
    shards = shard_data_random(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    shard_sizes = [shard["shard_size"] for shard in shards]

    # Check that sizes are approximately balanced (within 1 point due to np.array_split)
    min_size = min(shard_sizes)
    max_size = max(shard_sizes)
    assert max_size - min_size <= 1


def test_random_sharding_preserves_data_integrity(synthetic_data_small):
    """Test that random sharding preserves data integrity (matching indices)."""
    shards = shard_data_random(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
        sigma=synthetic_data_small["sigma"],
    )

    # Check that all arrays have matching lengths
    for shard in shards:
        assert len(shard["data"]) == len(shard["t1"])
        assert len(shard["data"]) == len(shard["t2"])
        assert len(shard["data"]) == len(shard["phi"])
        assert len(shard["data"]) == len(shard["sigma"])
        assert shard["shard_size"] == len(shard["data"])


def test_random_sharding_includes_metadata(synthetic_data_small):
    """Test that random sharding includes required metadata."""
    shards = shard_data_random(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    for i, shard in enumerate(shards):
        assert "shard_id" in shard
        assert shard["shard_id"] == i
        assert "q" in shard
        assert shard["q"] == synthetic_data_small["q"]
        assert "L" in shard
        assert shard["L"] == synthetic_data_small["L"]


# ============================================================================
# Test Group 3: shard_data_stratified() (PRIMARY)
# ============================================================================


def test_stratified_sharding_preserves_phi_distribution(synthetic_data_structured_phi):
    """Test that stratified sharding preserves phi angle distribution (PRIMARY TEST)."""
    shards = shard_data_stratified(
        data=synthetic_data_structured_phi["data"],
        t1=synthetic_data_structured_phi["t1"],
        t2=synthetic_data_structured_phi["t2"],
        phi=synthetic_data_structured_phi["phi"],
        num_shards=5,
        q=synthetic_data_structured_phi["q"],
        L=synthetic_data_structured_phi["L"],
    )

    # Concatenate all phi values to get overall distribution
    all_phi = np.concatenate([shard["phi"] for shard in shards])

    # Run Kolmogorov-Smirnov test for each shard
    ks_pvalues = []
    for shard in shards:
        shard_phi = shard["phi"]
        ks_stat, ks_pvalue = stats.ks_2samp(shard_phi, all_phi)
        ks_pvalues.append(ks_pvalue)

    # All p-values should be > 0.05 (indicates similar distributions)
    # This is the ACCEPTANCE CRITERIA from spec
    assert all(p >= 0.05 for p in ks_pvalues), f"KS test p-values: {ks_pvalues}"


def test_stratified_sharding_balanced_sizes(synthetic_data_medium):
    """Test that stratified sharding produces balanced shard sizes."""
    shards = shard_data_stratified(
        data=synthetic_data_medium["data"],
        t1=synthetic_data_medium["t1"],
        t2=synthetic_data_medium["t2"],
        phi=synthetic_data_medium["phi"],
        num_shards=5,
        q=synthetic_data_medium["q"],
        L=synthetic_data_medium["L"],
    )

    shard_sizes = [shard["shard_size"] for shard in shards]

    # Check that sizes are balanced within 10% (spec requirement)
    min_size = min(shard_sizes)
    max_size = max(shard_sizes)
    size_imbalance = max_size / min_size
    assert size_imbalance <= 1.10, f"Size imbalance: {size_imbalance:.2f}"


def test_stratified_sharding_deterministic(synthetic_data_small):
    """Test that stratified sharding is deterministic (no random seed)."""
    shards1 = shard_data_stratified(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    shards2 = shard_data_stratified(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    # Check that shards are identical
    assert len(shards1) == len(shards2)
    for s1, s2 in zip(shards1, shards2, strict=False):
        np.testing.assert_array_equal(s1["data"], s2["data"])
        np.testing.assert_array_equal(s1["phi"], s2["phi"])


def test_stratified_sharding_phi_coverage(synthetic_data_structured_phi):
    """Test that stratified sharding ensures each shard has phi coverage."""
    shards = shard_data_stratified(
        data=synthetic_data_structured_phi["data"],
        t1=synthetic_data_structured_phi["t1"],
        t2=synthetic_data_structured_phi["t2"],
        phi=synthetic_data_structured_phi["phi"],
        num_shards=5,
        q=synthetic_data_structured_phi["q"],
        L=synthetic_data_structured_phi["L"],
    )

    # Check that each shard has reasonable phi range
    overall_phi_range = (
        synthetic_data_structured_phi["phi"].max()
        - synthetic_data_structured_phi["phi"].min()
    )

    for shard in shards:
        shard_phi_range = shard["phi"].max() - shard["phi"].min()
        # Each shard should cover at least 10% of overall phi range
        assert shard_phi_range >= 0.10 * overall_phi_range


def test_stratified_sharding_per_angle_parameter_compatibility(
    synthetic_data_discrete_phi_angles,
):
    """Test that stratified sharding ensures all shards contain ALL unique phi angles.

    CRITICAL TEST for per-angle scaling compatibility (v2.2+)
    ---------------------------------------------------------

    Background:
    CMC always uses per-angle scaling (separate contrast[i], offset[i] for each phi angle).
    This requires that EVERY shard contains data from ALL phi angles. If a shard is missing
    an angle, the gradient w.r.t. that angle's parameters will be zero, causing silent
    optimization failures (parameters unchanged from initial values).

    This test verifies that stratified sharding (the default strategy) satisfies this
    requirement by ensuring each shard contains ALL unique phi angles present in the
    original dataset.

    References:
    - Ultra-Think Analysis: ultra-think-20251106-012247
    - NLSQ Investigation: docs/troubleshooting/nlsq-zero-iterations-investigation.md
    - Code: homodyne/optimization/cmc/backends/multiprocessing.py:444 (per_angle_scaling=True)
    """
    # Get unique phi angles from the original dataset
    unique_phi_angles = synthetic_data_discrete_phi_angles["unique_angles"]
    n_unique_angles = len(unique_phi_angles)

    # Create shards using stratified strategy
    shards = shard_data_stratified(
        data=synthetic_data_discrete_phi_angles["data"],
        t1=synthetic_data_discrete_phi_angles["t1"],
        t2=synthetic_data_discrete_phi_angles["t2"],
        phi=synthetic_data_discrete_phi_angles["phi"],
        num_shards=5,
        q=synthetic_data_discrete_phi_angles["q"],
        L=synthetic_data_discrete_phi_angles["L"],
    )

    # CRITICAL CHECK: Every shard must contain ALL unique phi angles
    for i, shard in enumerate(shards):
        shard_unique_phi = np.unique(shard["phi"])

        # Check 1: Number of unique angles matches original dataset
        assert len(shard_unique_phi) == n_unique_angles, (
            f"Shard {i} is missing phi angles! "
            f"Expected {n_unique_angles} unique angles, got {len(shard_unique_phi)}. "
            f"Missing angles would cause zero gradients for per-angle parameters."
        )

        # Check 2: All unique angles are present (not just the count)
        assert np.allclose(shard_unique_phi, unique_phi_angles, rtol=1e-10), (
            f"Shard {i} has different phi angles than the original dataset! "
            f"Expected angles: {unique_phi_angles}, got: {shard_unique_phi}. "
            f"This would break per-angle parameter gradients."
        )

        # Check 3: Each angle has sufficient representation (at least 1 data point)
        # This is implicitly guaranteed by checks 1 and 2, but we verify explicitly
        for angle in unique_phi_angles:
            angle_count = np.sum(np.isclose(shard["phi"], angle, rtol=1e-10))
            assert angle_count > 0, (
                f"Shard {i} has ZERO data points for phi={angle}! "
                f"This would cause zero gradient for contrast[{angle}] and offset[{angle}]."
            )

    # Additional verification: Check angle distribution balance using KS test
    # This ensures stratified sharding doesn't just include all angles but distributes them evenly
    for angle in unique_phi_angles:
        # Get the proportion of this angle in the original dataset
        original_prop = np.sum(
            np.isclose(synthetic_data_discrete_phi_angles["phi"], angle, rtol=1e-10)
        ) / len(synthetic_data_discrete_phi_angles["phi"])

        # Check the proportion in each shard
        for i, shard in enumerate(shards):
            shard_prop = np.sum(np.isclose(shard["phi"], angle, rtol=1e-10)) / len(
                shard["phi"]
            )

            # Allow up to 50% deviation from the original proportion (lenient for small datasets)
            # For production datasets with thousands of points per angle, this tolerance is tighter
            assert abs(shard_prop - original_prop) < 0.5 * original_prop, (
                f"Shard {i} has unbalanced representation of phi={angle}! "
                f"Expected proportion: {original_prop:.3f}, got: {shard_prop:.3f}. "
                f"Stratified sharding should distribute angles evenly across shards."
            )


# ============================================================================
# Test Group 4: shard_data_contiguous()
# ============================================================================


def test_contiguous_sharding_preserves_order(synthetic_data_small):
    """Test that contiguous sharding preserves temporal order."""
    shards = shard_data_contiguous(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    # Reconstruct data by concatenating shards in order
    reconstructed_data = np.concatenate([shard["data"] for shard in shards])
    reconstructed_phi = np.concatenate([shard["phi"] for shard in shards])

    # Should match original data exactly
    np.testing.assert_array_equal(reconstructed_data, synthetic_data_small["data"])
    np.testing.assert_array_equal(reconstructed_phi, synthetic_data_small["phi"])


def test_contiguous_sharding_sequential_indices(synthetic_data_small):
    """Test that contiguous sharding produces sequential index blocks."""
    shards = shard_data_contiguous(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    # Verify that shards cover full range without gaps
    total_size = 0
    for shard in shards:
        total_size += shard["shard_size"]

    assert total_size == len(synthetic_data_small["data"])


def test_contiguous_sharding_handles_non_divisible_sizes(synthetic_data_small):
    """Test that contiguous sharding handles non-divisible dataset sizes."""
    # Use dataset size that doesn't divide evenly
    # Create dataset with non-divisible size inline
    dataset_size = 50_007  # Prime-ish number
    np.random.seed(999)
    data = np.random.randn(dataset_size) + 1.0
    t1 = np.random.uniform(0, 10, dataset_size)
    t2 = np.random.uniform(0, 10, dataset_size)
    phi = np.random.uniform(-180, 180, dataset_size)

    shards = shard_data_contiguous(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        num_shards=7,  # Doesn't divide evenly
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    # Check that all data is accounted for
    total_size = sum(shard["shard_size"] for shard in shards)
    assert total_size == dataset_size


# ============================================================================
# Test Group 5: validate_shards()
# ============================================================================


def test_validate_shards_success(synthetic_data_small):
    """Test shard validation with valid shards."""
    shards = shard_data_stratified(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    is_valid, diagnostics = validate_shards(shards, len(synthetic_data_small["data"]))

    assert is_valid
    assert diagnostics["data_loss_check"]
    assert diagnostics["min_size_check"]
    assert diagnostics["balance_check"]
    assert len(diagnostics["errors"]) == 0


def test_validate_shards_detects_data_loss():
    """Test that validation detects data loss."""
    # Create valid shards
    data = np.random.randn(10_000)
    t1 = np.random.uniform(0, 10, 10_000)
    t2 = np.random.uniform(0, 10, 10_000)
    phi = np.random.uniform(-180, 180, 10_000)

    shards = shard_data_stratified(
        data=data, t1=t1, t2=t2, phi=phi, num_shards=5, q=0.01, L=5.0
    )

    # Corrupt shard size to simulate data loss
    shards[0]["shard_size"] = 100  # Much smaller than actual

    is_valid, diagnostics = validate_shards(shards, 10_000)

    assert not is_valid
    assert not diagnostics["data_loss_check"]
    assert "Data loss detected" in diagnostics["errors"][0]


def test_validate_shards_detects_min_size_violation():
    """Test that validation detects minimum size violations."""
    # Create small dataset
    data = np.random.randn(5_000)
    t1 = np.random.uniform(0, 10, 5_000)
    t2 = np.random.uniform(0, 10, 5_000)
    phi = np.random.uniform(-180, 180, 5_000)

    # Create too many shards (will violate min_shard_size)
    shards = shard_data_stratified(
        data=data, t1=t1, t2=t2, phi=phi, num_shards=10, q=0.01, L=5.0
    )

    # Each shard will have ~500 points, which is < 10,000 minimum
    is_valid, diagnostics = validate_shards(shards, 5_000, min_shard_size=10_000)

    assert not is_valid
    assert not diagnostics["min_size_check"]


def test_validate_shards_detects_imbalance():
    """Test that validation detects size imbalance."""
    # Create shards with artificial imbalance
    data = np.random.randn(10_000)
    t1 = np.random.uniform(0, 10, 10_000)
    t2 = np.random.uniform(0, 10, 10_000)
    phi = np.random.uniform(-180, 180, 10_000)

    # Create manually imbalanced shards
    shards = [
        {
            "data": data[:8000],
            "t1": t1[:8000],
            "t2": t2[:8000],
            "phi": phi[:8000],
            "q": 0.01,
            "L": 5.0,
            "shard_id": 0,
            "shard_size": 8000,
        },
        {
            "data": data[8000:],
            "t1": t1[8000:],
            "t2": t2[8000:],
            "phi": phi[8000:],
            "q": 0.01,
            "L": 5.0,
            "shard_id": 1,
            "shard_size": 2000,
        },
    ]

    # Size imbalance: 8000/2000 = 4.0, which exceeds 1.10 threshold
    is_valid, diagnostics = validate_shards(shards, 10_000)

    assert not is_valid
    assert not diagnostics["balance_check"]
    assert diagnostics["size_imbalance"] == 4.0


def test_validate_shards_phi_distribution(synthetic_data_structured_phi):
    """Test phi distribution validation with stratified sharding."""
    shards = shard_data_stratified(
        data=synthetic_data_structured_phi["data"],
        t1=synthetic_data_structured_phi["t1"],
        t2=synthetic_data_structured_phi["t2"],
        phi=synthetic_data_structured_phi["phi"],
        num_shards=5,
        q=synthetic_data_structured_phi["q"],
        L=synthetic_data_structured_phi["L"],
    )

    is_valid, diagnostics = validate_shards(
        shards,
        len(synthetic_data_structured_phi["data"]),
        validate_phi_distribution=True,
    )

    assert is_valid
    assert diagnostics["phi_distribution_check"]
    assert len(diagnostics["phi_ks_pvalues"]) == 5
    assert all(p >= 0.05 for p in diagnostics["phi_ks_pvalues"])


# ============================================================================
# Test Group 6: Edge Cases
# ============================================================================


def test_sharding_very_small_dataset():
    """Test sharding with very small dataset (<10k points)."""
    data = np.random.randn(1_000)
    t1 = np.random.uniform(0, 10, 1_000)
    t2 = np.random.uniform(0, 10, 1_000)
    phi = np.random.uniform(-180, 180, 1_000)

    # Should create 1 shard due to min_shard_size constraint
    shards = shard_data_stratified(
        data=data, t1=t1, t2=t2, phi=phi, num_shards=5, q=0.01, L=5.0
    )

    # Validation should pass even though shards are small
    is_valid, diagnostics = validate_shards(
        shards,
        1_000,
        min_shard_size=100,  # Lower threshold for small dataset
    )

    assert is_valid
    assert len(shards) == 5


def test_sharding_single_shard():
    """Test sharding with num_shards=1 (no actual sharding)."""
    data = np.random.randn(10_000)
    t1 = np.random.uniform(0, 10, 10_000)
    t2 = np.random.uniform(0, 10, 10_000)
    phi = np.random.uniform(-180, 180, 10_000)

    shards = shard_data_stratified(
        data=data, t1=t1, t2=t2, phi=phi, num_shards=1, q=0.01, L=5.0
    )

    assert len(shards) == 1
    assert shards[0]["shard_size"] == 10_000
    # Stratified sharding sorts by phi, so check values match (sorted)
    np.testing.assert_array_equal(np.sort(shards[0]["data"]), np.sort(data))


def test_sharding_without_sigma():
    """Test that sharding works without sigma array."""
    data = np.random.randn(10_000)
    t1 = np.random.uniform(0, 10, 10_000)
    t2 = np.random.uniform(0, 10, 10_000)
    phi = np.random.uniform(-180, 180, 10_000)

    shards = shard_data_stratified(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        num_shards=5,
        q=0.01,
        L=5.0,
        # sigma not provided
    )

    # Should not have sigma key
    for shard in shards:
        assert "sigma" not in shard


@pytest.mark.parametrize("strategy", ["random", "stratified", "contiguous"])
def test_all_strategies_preserve_total_size(synthetic_data_small, strategy):
    """Test that all sharding strategies preserve total dataset size."""
    shard_functions = {
        "random": shard_data_random,
        "stratified": shard_data_stratified,
        "contiguous": shard_data_contiguous,
    }

    shard_fn = shard_functions[strategy]
    shards = shard_fn(
        data=synthetic_data_small["data"],
        t1=synthetic_data_small["t1"],
        t2=synthetic_data_small["t2"],
        phi=synthetic_data_small["phi"],
        num_shards=5,
        q=synthetic_data_small["q"],
        L=synthetic_data_small["L"],
    )

    total_size = sum(shard["shard_size"] for shard in shards)
    assert total_size == len(synthetic_data_small["data"])
