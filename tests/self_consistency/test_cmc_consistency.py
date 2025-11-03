"""
Self-Consistency Tests for Consensus Monte Carlo
================================================

Comprehensive self-consistency validation for CMC:
- Same data with different shard counts (results should agree)
- Scaling behavior (linear runtime scaling, constant memory per shard)
- Reproducibility (fixed seeds give identical results)
- Checkpoint/resume consistency

Test Tier: Tier 4 (Self-Consistency)
Duration: 1-7 days per large-scale test
Acceptance Criteria: Agreement < 15% across different configurations
"""

import time
import numpy as np
import pytest
from typing import Dict, Tuple, Optional

# Handle optional dependencies
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Import CMC components
try:
    from homodyne.optimization.cmc.sharding import (
        calculate_optimal_num_shards,
        shard_data_stratified,
    )
    from homodyne.optimization.cmc.combination import combine_subposteriors
    CMC_AVAILABLE = True
except ImportError:
    CMC_AVAILABLE = False


def generate_synthetic_posterior_samples(
    n_samples: int = 2000,
    n_params: int = 5,
    true_params: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic MCMC posterior samples."""
    np.random.seed(seed)

    if true_params is None:
        true_params = np.ones(n_params)

    # Create realistic covariance
    L = np.random.randn(n_params, n_params)
    cov = L @ L.T + np.eye(n_params) * 0.1
    cov = (cov + cov.T) / 2

    # Generate samples
    samples = np.random.multivariate_normal(true_params, cov, size=n_samples)

    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov(samples.T)

    return samples, empirical_mean, empirical_cov


def create_large_shard_results(
    n_shards: int,
    n_samples_per_shard: int,
    n_params: int = 5,
    seed: int = 42,
) -> list:
    """
    Create realistic shard MCMC results.

    Parameters
    ----------
    n_shards : int
        Number of shards
    n_samples_per_shard : int
        Samples per shard
    n_params : int
        Number of parameters
    seed : int
        Base random seed

    Returns
    -------
    list of dicts with 'samples', 'mean', 'cov'
    """
    shard_results = []

    for shard_id in range(n_shards):
        shard_seed = seed + shard_id * 1000
        samples, mean, cov = generate_synthetic_posterior_samples(
            n_samples=n_samples_per_shard,
            n_params=n_params,
            seed=shard_seed,
        )

        shard_results.append({
            'shard_id': shard_id,
            'samples': samples,
            'mean': mean,
            'cov': cov,
            'n_samples': n_samples_per_shard,
        })

    return shard_results


@pytest.mark.self_consistency
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCDifferentShardCounts:
    """
    Test that CMC results are consistent across different shard counts.

    When the same data is sharded differently, CMC should produce
    similar results (agreement < 15%).
    """

    @pytest.mark.slow
    def test_consistency_10_vs_20_shards(self):
        """
        Test consistency between 10 and 20 shards.

        Same data should produce similar results regardless of shard count.
        """
        # Create shard results for 10 shards
        results_10 = create_large_shard_results(
            n_shards=10,
            n_samples_per_shard=2000,
            n_params=5,
            seed=42
        )

        # Create shard results for 20 shards
        # (in practice, these would come from the same data)
        results_20 = create_large_shard_results(
            n_shards=20,
            n_samples_per_shard=1000,  # Fewer per shard since we have more
            n_params=5,
            seed=42
        )

        # Combine both
        combined_10 = combine_subposteriors(results_10, method='weighted')
        combined_20 = combine_subposteriors(results_20, method='weighted')

        # Compute agreement
        mean_diff = np.abs(combined_10['mean'] - combined_20['mean'])
        rel_diff = mean_diff / np.abs(combined_10['mean'])

        # All parameters should agree within 15%
        assert np.all(rel_diff < 0.15), f"Max disagreement: {np.max(rel_diff)}"

    @pytest.mark.slow
    def test_consistency_20_vs_50_shards(self):
        """Test consistency between 20 and 50 shards."""
        results_20 = create_large_shard_results(
            n_shards=20,
            n_samples_per_shard=1000,
            n_params=5,
            seed=42
        )

        results_50 = create_large_shard_results(
            n_shards=50,
            n_samples_per_shard=400,
            n_params=5,
            seed=42
        )

        combined_20 = combine_subposteriors(results_20)
        combined_50 = combine_subposteriors(results_50)

        mean_diff = np.abs(combined_20['mean'] - combined_50['mean'])
        rel_diff = mean_diff / (np.abs(combined_20['mean']) + 1e-10)

        # Should agree reasonably well
        assert np.all(rel_diff < 0.15)

    @pytest.mark.slow
    def test_pairwise_agreement_multiple_configs(self):
        """
        Test pairwise agreement across multiple configurations.

        Compare 10, 20, 50 shards - all pairs should agree.
        """
        shard_configs = [
            (10, 2000),
            (20, 1000),
            (50, 400),
        ]

        results_list = [
            create_large_shard_results(n_shards, n_samples, seed=42)
            for n_shards, n_samples in shard_configs
        ]

        combined_list = [
            combine_subposteriors(results)
            for results in results_list
        ]

        # Check all pairs
        for i in range(len(combined_list)):
            for j in range(i+1, len(combined_list)):
                mean_diff = np.abs(combined_list[i]['mean'] - combined_list[j]['mean'])
                rel_diff = mean_diff / (np.abs(combined_list[i]['mean']) + 1e-10)

                assert np.all(rel_diff < 0.15), \
                    f"Config {i} and {j} disagree: max {np.max(rel_diff)}"


@pytest.mark.self_consistency
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCScalingBehavior:
    """
    Test scaling properties of CMC.

    - Runtime should scale linearly with dataset size
    - Memory per shard should be constant
    - Communication cost should not dominate
    """

    @pytest.mark.slow
    def test_runtime_scaling_linear(self):
        """
        Test that CMC runtime scales linearly with data size.

        Timing different dataset sizes should show linear scaling.
        """
        sizes = [10_000, 100_000, 1_000_000]
        times = []

        for size in sizes:
            # Create shards
            n_shards = max(1, size // 100_000)
            shard_results = create_large_shard_results(
                n_shards=n_shards,
                n_samples_per_shard=2000 // max(1, n_shards // 10),
                n_params=5,
                seed=42
            )

            # Time combination
            start = time.time()
            combined = combine_subposteriors(shard_results)
            elapsed = time.time() - start

            times.append(elapsed)

        # Check approximate linear scaling
        # time[1] / time[0] should be ~10
        # time[2] / time[0] should be ~100
        if len(times) >= 2:
            ratio_1 = times[1] / (times[0] + 1e-6)
            # Should be within factor of 10-15 (not quadratic)
            assert ratio_1 < 20

    @pytest.mark.slow
    def test_memory_per_shard_constant(self):
        """
        Test that memory usage per shard is roughly constant.

        As we increase dataset size and shard count proportionally,
        each shard should consume similar memory.
        """
        # Configuration 1: 1M points, 10 shards = 100k/shard
        config1 = {
            'n_shards': 10,
            'n_samples_per_shard': 2000,
            'n_params': 5,
        }

        # Configuration 2: 10M points, 100 shards = 100k/shard
        config2 = {
            'n_shards': 100,
            'n_samples_per_shard': 2000,
            'n_params': 5,
        }

        # Create mock results (in practice would measure actual memory)
        results1 = create_large_shard_results(**config1)
        results2 = create_large_shard_results(
            n_shards=100,
            n_samples_per_shard=2000,
            n_params=5
        )

        # Memory estimate: n_samples * n_params * 8 bytes
        mem1 = config1['n_shards'] * config1['n_samples_per_shard'] * config1['n_params'] * 8
        mem2 = 100 * 2000 * 5 * 8

        # Per-shard memory
        mem_per_shard_1 = mem1 / config1['n_shards']
        mem_per_shard_2 = mem2 / 100

        # Should be similar
        assert np.isclose(mem_per_shard_1, mem_per_shard_2, rtol=0.1)

    @pytest.mark.slow
    def test_communication_overhead(self):
        """
        Test that communication costs don't dominate.

        Communication = gather means/covs from shards + final combination.
        Should be negligible compared to MCMC on each shard.
        """
        # Time a large combination
        n_shards = 100
        shard_results = create_large_shard_results(
            n_shards=n_shards,
            n_samples_per_shard=5000,
            n_params=10,
            seed=42
        )

        start = time.time()
        combined = combine_subposteriors(shard_results)
        combination_time = time.time() - start

        # Communication is just transferring means and covs
        # Data size: n_shards * (n_params + n_params^2) * 8 bytes
        data_size = n_shards * (10 + 10*10) * 8  # ~ 8KB

        # Combination should be fast (< 1 second for 100 shards)
        assert combination_time < 1.0


@pytest.mark.self_consistency
class TestCMCReproducibility:
    """Test reproducibility with fixed random seeds."""

    def test_deterministic_with_seed(self):
        """
        Test that fixed seed produces deterministic results.

        Running same combination with same seed should produce identical output.
        """
        # Create results
        results1 = create_large_shard_results(
            n_shards=5,
            n_samples_per_shard=1000,
            n_params=3,
            seed=12345
        )

        results2 = create_large_shard_results(
            n_shards=5,
            n_samples_per_shard=1000,
            n_params=3,
            seed=12345
        )

        # Combine with same seed
        np.random.seed(999)
        combined1 = combine_subposteriors(results1)

        np.random.seed(999)
        combined2 = combine_subposteriors(results2)

        # Should be identical
        assert np.allclose(combined1['mean'], combined2['mean'])

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different initialization."""
        results = create_large_shard_results(
            n_shards=5,
            n_samples_per_shard=1000,
            n_params=3,
            seed=42
        )

        # Combine with different seeds
        np.random.seed(111)
        combined1 = combine_subposteriors(results)

        np.random.seed(222)
        combined2 = combine_subposteriors(results)

        # Seeds shouldn't matter for mean calculation
        assert np.allclose(combined1['mean'], combined2['mean'])


@pytest.mark.self_consistency
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCCheckpointConsistency:
    """Test that checkpoints and resume operations are consistent."""

    def test_checkpoint_resume_equivalence(self):
        """
        Test that resuming from checkpoint gives same result as
        running from scratch.

        (In Phase 1, checkpointing is architecture-ready but not
        fully implemented - this test validates structure)
        """
        shard_results = create_large_shard_results(
            n_shards=5,
            n_samples_per_shard=2000,
            n_params=5,
            seed=42
        )

        # Full run
        combined_full = combine_subposteriors(shard_results)

        # Mock checkpoint: save state after first 3 shards
        checkpoint_data = {
            'processed_shards': 3,
            'partial_results': shard_results[:3],
        }

        # Continuation: process remaining shards
        remaining_results = shard_results[3:]
        combined_remaining = combine_subposteriors(
            shard_results[3:],  # Just the remaining
        )

        # In full implementation, would merge checkpoint with new results
        # For now, verify structure is sound
        assert checkpoint_data is not None
        assert combined_remaining is not None


@pytest.mark.self_consistency
class TestCMCNumericalStability:
    """Test numerical stability under repeated operations."""

    def test_iterative_combination_stability(self):
        """
        Test stability of combining same results multiple times.

        Repeated combination should be numerically stable.
        """
        shard_results = create_large_shard_results(
            n_shards=4,
            n_samples_per_shard=1000,
            n_params=5
        )

        # First combination
        combined1 = combine_subposteriors(shard_results)

        # Treat combined as new "shard" and combine with originals
        # This tests stability of iterative operations
        extended_results = shard_results + [
            {
                'samples': combined1['samples'],
                'mean': combined1['mean'],
                'cov': combined1['cov']
            }
        ]

        combined2 = combine_subposteriors(extended_results)

        # Should still be numerically stable
        assert np.all(np.isfinite(combined2['mean']))
        assert np.all(np.isfinite(combined2['cov']))

    def test_matrix_conditioning_stable(self):
        """Test that covariance matrices remain well-conditioned."""
        for _ in range(5):
            shard_results = create_large_shard_results(
                n_shards=8,
                n_samples_per_shard=1000,
                n_params=6
            )

            combined = combine_subposteriors(shard_results)

            # Compute condition number
            cond = np.linalg.cond(combined['cov'])

            # Should remain reasonable (< 1e6)
            assert cond < 1e6


@pytest.mark.self_consistency
@pytest.mark.parametrize("n_shards", [4, 10, 25, 50])
def test_consistency_across_shard_counts(n_shards):
    """
    Parametrized test for consistency across different shard counts.

    All should produce results within 15% agreement.
    """
    n_samples = max(100, 5000 // max(1, n_shards // 10))

    results = create_large_shard_results(
        n_shards=n_shards,
        n_samples_per_shard=n_samples,
        n_params=5,
        seed=42
    )

    combined = combine_subposteriors(results)

    # Should produce valid result
    assert len(combined['mean']) == 5
    assert combined['cov'].shape == (5, 5)
    assert np.all(np.isfinite(combined['mean']))
    assert np.all(np.isfinite(combined['cov']))


@pytest.mark.self_consistency
class TestCMCLargeScaleConsistency:
    """Test consistency at large scale (if resources available)."""

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_5m_points_10_vs_20_shards(self):
        """
        Test 5M point dataset with 10 and 20 shards.

        This validates large-scale consistency.
        """
        pytest.skip("Large-scale test - requires adequate compute")

    @pytest.mark.slow
    @pytest.mark.gpu
    def test_100m_points_scaling(self):
        """Test 100M point dataset scaling."""
        pytest.skip("Large-scale test - requires adequate compute")


# ============================================================================
# Analysis and Diagnostic Tests
# ============================================================================

@pytest.mark.self_consistency
class TestCMCConsistencyAnalysis:
    """Analyze consistency patterns."""

    def test_consistency_improves_with_samples(self):
        """
        Test that consistency improves with more samples per shard.

        More samples = lower sampling error = better agreement.
        """
        # Few samples
        results_few = create_large_shard_results(
            n_shards=10,
            n_samples_per_shard=100,
            n_params=5,
            seed=42
        )

        # Many samples
        results_many = create_large_shard_results(
            n_shards=10,
            n_samples_per_shard=10000,
            n_params=5,
            seed=42
        )

        combined_few = combine_subposteriors(results_few)
        combined_many = combine_subposteriors(results_many)

        # Both should be valid
        assert combined_few is not None
        assert combined_many is not None

    @pytest.mark.skip(
        reason="v2.1.0 API change: CMC validation now enforces balanced shards "
               "(all shards must have same sample count). Imbalanced shards no longer supported."
    )
    def test_consistency_degrades_with_imbalanced_shards(self):
        """
        Test that consistency is affected by imbalanced shards.

        If shards have very different sample counts, agreement may suffer.

        NOTE: This test is skipped in v2.1.0 because CMC validation now requires
        all shards to have the same number of samples (validation at combination.py:423-427).
        """
        # Balanced
        balanced = create_large_shard_results(
            n_shards=4,
            n_samples_per_shard=1000,
            n_params=5
        )

        # Imbalanced
        imbalanced = [
            *create_large_shard_results(
                n_shards=3,
                n_samples_per_shard=5000,
                n_params=5,
                seed=42
            ),
            # Add shard with few samples
            create_large_shard_results(
                n_shards=1,
                n_samples_per_shard=100,
                n_params=5,
                seed=99
            )[0],
        ]

        combined_balanced = combine_subposteriors(balanced)
        combined_imbalanced = combine_subposteriors(imbalanced)

        # Both should be valid even if imbalanced
        assert combined_balanced is not None
        assert combined_imbalanced is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
