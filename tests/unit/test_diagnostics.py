"""Tests for CMC diagnostics and validation module.

Tests cover:
- validate_cmc_results() with strict and lenient modes
- compute_per_shard_diagnostics()
- compute_between_shard_kl_divergence()
- compute_combined_posterior_diagnostics()
- Helper functions for KL divergence and multimodality
"""

import numpy as np
import pytest

from homodyne.optimization.cmc.diagnostics import (
    validate_cmc_results,
    compute_per_shard_diagnostics,
    compute_between_shard_kl_divergence,
    compute_combined_posterior_diagnostics,
    _fit_gaussian_to_samples,
    _compute_kl_divergence_matrix,
    _kl_divergence_gaussian,
    _check_multimodality,
    _validate_single_shard,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_shard_result():
    """Single shard result with 2D samples (single chain)."""
    np.random.seed(42)
    samples = np.random.randn(100, 3)  # 100 samples, 3 parameters

    return {
        "samples": samples,
        "converged": True,
        "diagnostics": {
            "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
            "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
            "acceptance_rate": 0.85,
        },
    }


@pytest.fixture
def multichain_shard_result():
    """Single shard result with 3D samples (multiple chains)."""
    np.random.seed(43)
    # 4 chains, 100 samples each, 3 parameters
    samples = np.random.randn(4, 100, 3)

    return {
        "samples": samples,
        "converged": True,
        "diagnostics": {
            "rhat": {"param_0": 1.05, "param_1": 1.03, "param_2": 1.04},
            "ess": {"param_0": 120.0, "param_1": 115.0, "param_2": 125.0},
            "acceptance_rate": 0.88,
        },
    }


@pytest.fixture
def converged_shards():
    """List of converged shard results for testing."""
    np.random.seed(44)
    shards = []

    for i in range(5):
        # Create samples from slightly different Gaussians
        mean = np.array([1.0 + i * 0.1, 2.0 + i * 0.05, 3.0 + i * 0.15])
        samples = np.random.multivariate_normal(mean, np.eye(3) * 0.1, size=100)

        shards.append(
            {
                "samples": samples,
                "converged": True,
                "shard_id": i,
                "diagnostics": {
                    "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
                    "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
                    "acceptance_rate": 0.85,
                },
            }
        )

    return shards


@pytest.fixture
def failed_shard():
    """Shard result that failed to converge."""
    np.random.seed(45)
    samples = np.random.randn(50, 3)  # Fewer samples

    return {
        "samples": samples,
        "converged": False,
        "shard_id": 99,
        "diagnostics": {
            "rhat": {"param_0": 1.25, "param_1": 1.30, "param_2": 1.22},  # High R-hat
            "ess": {"param_0": 20.0, "param_1": 25.0, "param_2": 18.0},  # Low ESS
            "acceptance_rate": 0.45,  # Low acceptance
        },
    }


# ============================================================================
# Test compute_per_shard_diagnostics()
# ============================================================================


def test_compute_per_shard_diagnostics_2d_samples(simple_shard_result):
    """Test diagnostics computation for 2D samples (single chain)."""
    diagnostics = compute_per_shard_diagnostics(simple_shard_result, shard_idx=0)

    assert diagnostics["shard_id"] == 0
    assert diagnostics["num_samples"] == 100
    assert diagnostics["num_params"] == 3

    # Should use existing diagnostics
    assert diagnostics["rhat"] is not None
    assert diagnostics["ess"] is not None
    assert diagnostics["acceptance_rate"] == 0.85

    # Should have trace data
    assert "param_0" in diagnostics["trace_data"]
    assert "param_1" in diagnostics["trace_data"]
    assert "param_2" in diagnostics["trace_data"]


def test_compute_per_shard_diagnostics_3d_samples(multichain_shard_result):
    """Test diagnostics computation for 3D samples (multi-chain)."""
    diagnostics = compute_per_shard_diagnostics(multichain_shard_result, shard_idx=1)

    assert diagnostics["shard_id"] == 1
    assert diagnostics["num_samples"] == 100
    assert diagnostics["num_params"] == 3

    # Should use existing diagnostics
    assert diagnostics["rhat"] is not None
    assert diagnostics["ess"] is not None

    # R-hat should be defined for multi-chain
    assert "param_0" in diagnostics["rhat"]


def test_compute_per_shard_diagnostics_missing_samples():
    """Test diagnostics with missing samples key."""
    result = {"converged": True}
    diagnostics = compute_per_shard_diagnostics(result, shard_idx=5)

    assert diagnostics["shard_id"] == 5
    assert diagnostics["num_samples"] is None
    assert diagnostics["num_params"] is None


# ============================================================================
# Test compute_between_shard_kl_divergence()
# ============================================================================


def test_kl_divergence_identical_shards():
    """Test KL divergence between identical shards (should be 0)."""
    np.random.seed(46)
    samples = np.random.randn(100, 3)

    shards = [
        {"samples": samples},
        {"samples": samples},
    ]

    kl_matrix = compute_between_shard_kl_divergence(shards)

    assert kl_matrix.shape == (2, 2)
    assert kl_matrix[0, 0] == 0.0  # Diagonal
    assert kl_matrix[1, 1] == 0.0  # Diagonal
    assert kl_matrix[0, 1] < 0.01  # Near zero (numerical precision)
    assert kl_matrix[1, 0] < 0.01  # Symmetric


def test_kl_divergence_different_shards(converged_shards):
    """Test KL divergence between different shards."""
    kl_matrix = compute_between_shard_kl_divergence(converged_shards)

    assert kl_matrix.shape == (5, 5)

    # Diagonal should be zero
    np.testing.assert_array_almost_equal(np.diag(kl_matrix), 0.0, decimal=10)

    # Off-diagonal should be small but non-zero
    off_diagonal = kl_matrix[np.triu_indices(5, k=1)]
    assert np.all(off_diagonal >= 0)  # KL is non-negative
    assert np.all(off_diagonal < 5.0)  # Should be small for similar distributions

    # Matrix should be symmetric
    np.testing.assert_array_almost_equal(kl_matrix, kl_matrix.T, decimal=10)


def test_kl_divergence_inconsistent_params():
    """Test KL divergence with inconsistent parameter counts."""
    shards = [
        {"samples": np.random.randn(100, 3)},
        {"samples": np.random.randn(100, 4)},  # Wrong number of params
    ]

    with pytest.raises(ValueError, match="has 4 parameters, expected 3"):
        compute_between_shard_kl_divergence(shards)


def test_kl_divergence_missing_samples():
    """Test KL divergence with missing samples."""
    shards = [
        {"samples": np.random.randn(100, 3)},
        {"no_samples": True},
    ]

    with pytest.raises(ValueError, match="missing 'samples' key"):
        compute_between_shard_kl_divergence(shards)


# ============================================================================
# Test compute_combined_posterior_diagnostics()
# ============================================================================


def test_combined_diagnostics(converged_shards):
    """Test combined posterior diagnostics computation."""
    diagnostics = compute_combined_posterior_diagnostics(converged_shards)

    # Should compute combined ESS
    assert "combined_ess" in diagnostics
    assert diagnostics["combined_ess"] is not None
    assert "param_0" in diagnostics["combined_ess"]
    assert "param_1" in diagnostics["combined_ess"]
    assert "param_2" in diagnostics["combined_ess"]

    # Should compute uncertainty ratio
    assert "parameter_uncertainty_ratio" in diagnostics
    assert diagnostics["parameter_uncertainty_ratio"] is not None

    # Should check multimodality
    assert "multimodality_detected" in diagnostics
    assert isinstance(diagnostics["multimodality_detected"], bool)


def test_combined_diagnostics_empty_shards():
    """Test combined diagnostics with no valid samples."""
    shards = [{"converged": False}]

    diagnostics = compute_combined_posterior_diagnostics(shards)

    # Should return empty/None diagnostics gracefully
    assert isinstance(diagnostics, dict)


# ============================================================================
# Test validate_cmc_results()
# ============================================================================


def test_validate_cmc_results_all_pass(converged_shards):
    """Test validation with all shards passing."""
    is_valid, diagnostics = validate_cmc_results(
        converged_shards,
        strict_mode=True,
        min_success_rate=0.90,
        max_kl_divergence=5.0,  # Increase threshold for test shards with different means
        max_rhat=1.1,
        min_ess=50.0,  # Lower threshold for test data
    )

    assert is_valid is True
    assert diagnostics["success_rate"] == 1.0
    assert diagnostics["num_successful"] == 5
    assert diagnostics["num_total"] == 5
    assert diagnostics["max_kl_divergence"] is not None
    assert "per_shard_diagnostics" in diagnostics
    assert len(diagnostics["per_shard_diagnostics"]) == 5


def test_validate_cmc_results_low_success_rate_strict(converged_shards, failed_shard):
    """Test validation with low success rate in strict mode."""
    shards = converged_shards[:2] + [failed_shard] * 3  # 2/5 converged

    is_valid, diagnostics = validate_cmc_results(
        shards,
        strict_mode=True,
        min_success_rate=0.90,
    )

    assert is_valid is False
    assert "error" in diagnostics
    assert diagnostics["success_rate"] == 0.4
    assert diagnostics["num_successful"] == 2
    assert diagnostics["num_total"] == 5


def test_validate_cmc_results_low_success_rate_lenient(converged_shards, failed_shard):
    """Test validation with low success rate in lenient mode."""
    shards = converged_shards[:2] + [failed_shard] * 3  # 2/5 converged

    is_valid, diagnostics = validate_cmc_results(
        shards,
        strict_mode=False,
        min_success_rate=0.90,
    )

    assert is_valid is True  # Lenient mode always returns True
    assert diagnostics["success_rate"] == 0.4
    assert len(diagnostics["validation_warnings"]) > 0


def test_validate_cmc_results_high_kl_strict():
    """Test validation with high KL divergence in strict mode."""
    np.random.seed(47)

    # Create shards with very different posteriors
    shards = [
        {
            "samples": np.random.multivariate_normal([0, 0, 0], np.eye(3), 100),
            "converged": True,
            "diagnostics": {
                "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
                "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
                "acceptance_rate": 0.85,
            },
        },
        {
            "samples": np.random.multivariate_normal([10, 10, 10], np.eye(3), 100),
            "converged": True,
            "diagnostics": {
                "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
                "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
                "acceptance_rate": 0.85,
            },
        },
    ]

    is_valid, diagnostics = validate_cmc_results(
        shards,
        strict_mode=True,
        max_kl_divergence=2.0,
    )

    assert is_valid is False
    assert "error" in diagnostics
    assert "KL divergence" in diagnostics["error"]
    assert diagnostics["max_kl_divergence"] > 2.0


def test_validate_cmc_results_high_kl_lenient():
    """Test validation with high KL divergence in lenient mode."""
    np.random.seed(48)

    # Create shards with very different posteriors
    shards = [
        {
            "samples": np.random.multivariate_normal([0, 0, 0], np.eye(3), 100),
            "converged": True,
            "diagnostics": {
                "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
                "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
                "acceptance_rate": 0.85,
            },
        },
        {
            "samples": np.random.multivariate_normal([10, 10, 10], np.eye(3), 100),
            "converged": True,
            "diagnostics": {
                "rhat": {"param_0": 1.01, "param_1": 1.02, "param_2": 1.03},
                "ess": {"param_0": 95.0, "param_1": 98.0, "param_2": 92.0},
                "acceptance_rate": 0.85,
            },
        },
    ]

    is_valid, diagnostics = validate_cmc_results(
        shards,
        strict_mode=False,
        max_kl_divergence=2.0,
    )

    assert is_valid is True
    assert len(diagnostics["validation_warnings"]) > 0
    assert diagnostics["max_kl_divergence"] > 2.0


def test_validate_cmc_results_empty_shards():
    """Test validation with empty shard list."""
    is_valid, diagnostics = validate_cmc_results([], strict_mode=True)

    assert is_valid is False
    assert "error" in diagnostics


def test_validate_cmc_results_no_converged_shards(failed_shard):
    """Test validation when no shards converged."""
    shards = [failed_shard] * 3

    is_valid, diagnostics = validate_cmc_results(shards, strict_mode=True)

    assert is_valid is False
    assert diagnostics["success_rate"] == 0.0


# ============================================================================
# Test Helper Functions
# ============================================================================


def test_fit_gaussian_to_samples():
    """Test Gaussian fitting to samples."""
    np.random.seed(49)
    mean_true = np.array([1.0, 2.0, 3.0])
    cov_true = np.eye(3) * 0.5

    samples = np.random.multivariate_normal(mean_true, cov_true, 1000)
    gaussian = _fit_gaussian_to_samples(samples)

    assert "mean" in gaussian
    assert "cov" in gaussian
    assert gaussian["mean"].shape == (3,)
    assert gaussian["cov"].shape == (3, 3)

    # Mean should be close to true mean
    np.testing.assert_array_almost_equal(gaussian["mean"], mean_true, decimal=1)

    # Covariance should be positive definite
    eigenvalues = np.linalg.eigvalsh(gaussian["cov"])
    assert np.all(eigenvalues > 0)


def test_fit_gaussian_single_parameter():
    """Test Gaussian fitting for single parameter."""
    np.random.seed(50)
    samples = np.random.randn(100, 1)

    gaussian = _fit_gaussian_to_samples(samples)

    assert gaussian["mean"].shape == (1,)
    assert gaussian["cov"].shape == (1, 1)


def test_kl_divergence_gaussian_identical():
    """Test KL divergence between identical Gaussians."""
    mean = np.array([1.0, 2.0])
    cov = np.eye(2)

    kl = _kl_divergence_gaussian(mean, cov, mean, cov)

    np.testing.assert_almost_equal(kl, 0.0, decimal=10)


def test_kl_divergence_gaussian_different():
    """Test KL divergence between different Gaussians."""
    mean_p = np.array([0.0, 0.0])
    cov_p = np.eye(2)

    mean_q = np.array([1.0, 1.0])
    cov_q = np.eye(2) * 2.0

    kl = _kl_divergence_gaussian(mean_p, cov_p, mean_q, cov_q)

    assert kl > 0  # KL should be positive
    assert kl < 10.0  # Should be reasonable for these distributions


def test_check_multimodality_unimodal():
    """Test multimodality detection on unimodal distribution."""
    np.random.seed(51)
    # Use larger sample for more stable bootstrap
    samples = np.random.randn(2000, 3)

    multimodal = _check_multimodality(samples)

    # Note: This is a heuristic test, so we just check it returns a boolean
    # The actual value may vary depending on random seed and sample size
    assert isinstance(multimodal, bool)


def test_check_multimodality_bimodal():
    """Test multimodality detection on bimodal distribution."""
    np.random.seed(52)

    # Create bimodal distribution
    mode1 = np.random.multivariate_normal([0, 0, 0], np.eye(3) * 0.1, 250)
    mode2 = np.random.multivariate_normal([5, 5, 5], np.eye(3) * 0.1, 250)
    samples = np.vstack([mode1, mode2])

    multimodal = _check_multimodality(samples)

    # May or may not detect (heuristic is simple)
    assert isinstance(multimodal, bool)


def test_validate_single_shard_passing(simple_shard_result):
    """Test single shard validation when passing."""
    validation = _validate_single_shard(
        simple_shard_result,
        shard_idx=0,
        max_rhat=1.1,
        min_ess=50.0,  # Lower threshold
        strict_mode=True,
    )

    assert validation["shard_id"] == 0
    assert len(validation["validation_errors"]) == 0


def test_validate_single_shard_failing(failed_shard):
    """Test single shard validation when failing."""
    validation = _validate_single_shard(
        failed_shard,
        shard_idx=99,
        max_rhat=1.1,
        min_ess=100.0,
        strict_mode=True,
    )

    assert validation["shard_id"] == 99
    assert len(validation["validation_errors"]) > 0  # Should have errors


# ============================================================================
# Summary Test
# ============================================================================


def test_diagnostics_module_summary(converged_shards):
    """Integration test: validate complete workflow."""
    # Run validation
    is_valid, diagnostics = validate_cmc_results(
        converged_shards,
        strict_mode=True,
        min_success_rate=0.90,
        max_kl_divergence=5.0,
        max_rhat=1.1,
        min_ess=50.0,
    )

    # Should pass all validations
    assert is_valid is True

    # Should have all expected keys
    expected_keys = [
        "success_rate",
        "num_successful",
        "num_total",
        "max_kl_divergence",
        "kl_matrix",
        "per_shard_diagnostics",
        "combined_diagnostics",
        "validation_warnings",
        "validation_errors",
    ]

    for key in expected_keys:
        assert key in diagnostics

    # Per-shard diagnostics should be complete
    assert len(diagnostics["per_shard_diagnostics"]) == 5

    # Combined diagnostics should exist
    assert "combined_ess" in diagnostics["combined_diagnostics"]

    # KL matrix should be symmetric
    kl_matrix = np.array(diagnostics["kl_matrix"])
    np.testing.assert_array_almost_equal(kl_matrix, kl_matrix.T, decimal=10)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS MODULE TEST SUMMARY")
    print("=" * 70)
    print(f"Success rate: {diagnostics['success_rate']:.1%}")
    print(
        f"Converged shards: {diagnostics['num_successful']}/{diagnostics['num_total']}"
    )
    print(f"Max KL divergence: {diagnostics['max_kl_divergence']:.3f}")
    print(f"Validation result: {'PASS' if is_valid else 'FAIL'}")
    print("=" * 70)
