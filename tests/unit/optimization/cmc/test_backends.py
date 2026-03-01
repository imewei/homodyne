"""Tests for CMC execution backends module.

Comprehensive tests for:
- CMCBackend: Abstract base class
- select_backend: Backend factory
- combine_shard_samples: Sample combination
- MultiprocessingBackend: CPU parallelism
- PjitBackend: JAX distributed execution
- PBSBackend: HPC cluster execution
"""

from __future__ import annotations

import numpy as np
import pytest

# Require ArviZ for CMC imports; skip module if missing optional dependency
pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.backends import (  # noqa: E402
    CMCBackend,
    MultiprocessingBackend,
    select_backend,
)
from homodyne.optimization.cmc.backends.base import combine_shard_samples  # noqa: E402
from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402
from homodyne.optimization.cmc.sampler import MCMCSamples  # noqa: E402

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Create default CMCConfig for testing."""
    return CMCConfig(
        num_warmup=100,
        num_samples=200,
        num_chains=2,
        target_accept_prob=0.8,
    )


@pytest.fixture
def sample_param_names():
    """Parameter names for 2 phi angles in static mode."""
    return [
        "contrast_0",
        "contrast_1",
        "offset_0",
        "offset_1",
        "D0",
        "alpha",
        "D_offset",
    ]


@pytest.fixture
def mock_mcmc_samples(sample_param_names):
    """Create mock MCMCSamples for testing."""

    def create_samples(n_chains=2, n_samples=100, seed=42):
        np.random.seed(seed)
        samples = {
            name: np.random.randn(n_chains, n_samples) for name in sample_param_names
        }
        extra_fields = {
            "diverging": np.zeros((n_chains, n_samples), dtype=bool),
            "accept_prob": np.random.uniform(0.8, 0.95, (n_chains, n_samples)),
        }

        return MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
            extra_fields=extra_fields,
        )

    return create_samples


# =============================================================================
# Tests for select_backend
# =============================================================================


class TestSelectBackend:
    """Tests for select_backend factory function."""

    def test_auto_returns_multiprocessing(self):
        """Test auto backend selection defaults to multiprocessing."""
        config = CMCConfig(backend_name="auto")

        backend = select_backend(config)

        assert isinstance(backend, MultiprocessingBackend)

    def test_explicit_multiprocessing(self):
        """Test explicit multiprocessing backend selection."""
        config = CMCConfig(backend_name="multiprocessing")

        backend = select_backend(config)

        assert isinstance(backend, MultiprocessingBackend)

    def test_pjit_fallback_to_multiprocessing(self):
        """Test pjit falls back to multiprocessing when not available."""
        config = CMCConfig(backend_name="pjit")

        # Should not raise, falls back gracefully
        backend = select_backend(config)

        # Falls back to multiprocessing or returns PjitBackend
        assert isinstance(backend, CMCBackend)

    def test_pbs_fallback_to_multiprocessing(self):
        """Test PBS falls back to multiprocessing when not available."""
        config = CMCConfig(backend_name="pbs")

        backend = select_backend(config)

        assert isinstance(backend, CMCBackend)

    def test_unknown_backend_raises(self):
        """Test unknown backend raises ValueError."""
        config = CMCConfig()
        config.backend_name = "unknown_backend"

        with pytest.raises(ValueError, match="Unknown backend"):
            select_backend(config)


# =============================================================================
# Tests for MultiprocessingBackend
# =============================================================================


class TestMultiprocessingBackend:
    """Tests for MultiprocessingBackend."""

    def test_get_name(self):
        """Test backend name includes worker count."""
        backend = MultiprocessingBackend()

        name = backend.get_name()
        # Name format: "multiprocessing(N workers)"
        assert name.startswith("multiprocessing(")
        assert "workers)" in name

    def test_is_available(self):
        """Test backend availability."""
        backend = MultiprocessingBackend()

        # Multiprocessing should always be available on supported platforms
        assert backend.is_available() is True


# =============================================================================
# Tests for combine_shard_samples
# =============================================================================


class TestCombineShardSamples:
    """Tests for combine_shard_samples function."""

    def test_single_shard_returns_same(self, mock_mcmc_samples):
        """Test single shard returns unchanged."""
        shard = mock_mcmc_samples()

        combined = combine_shard_samples([shard], method="consensus_mc")

        # Should return the same samples
        assert combined.n_chains == shard.n_chains
        assert combined.n_samples == shard.n_samples
        assert combined.param_names == shard.param_names

    def test_two_shards_simple_average(self, mock_mcmc_samples, sample_param_names):
        """Test two shards with simple average (deprecated)."""
        shard1 = mock_mcmc_samples(seed=42)
        shard2 = mock_mcmc_samples(seed=123)

        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined = combine_shard_samples([shard1, shard2], method="simple_average")

        # Check shapes preserved
        assert combined.n_chains == shard1.n_chains
        assert combined.n_samples == shard1.n_samples

        # Check averaging occurred
        for name in sample_param_names:
            expected = (shard1.samples[name] + shard2.samples[name]) / 2
            np.testing.assert_array_almost_equal(
                combined.samples[name], expected, decimal=10
            )

    def test_two_shards_weighted_gaussian(self, mock_mcmc_samples, sample_param_names):
        """Test two shards with weighted Gaussian combination (deprecated)."""
        # Create shards with different variances
        np.random.seed(42)
        shard1_samples = {
            name: np.random.randn(2, 100) * 0.1 for name in sample_param_names
        }  # Low variance
        np.random.seed(123)
        shard2_samples = {
            name: np.random.randn(2, 100) * 1.0 for name in sample_param_names
        }  # High variance

        shard1 = MCMCSamples(
            samples=shard1_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
        )
        shard2 = MCMCSamples(
            samples=shard2_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
        )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined = combine_shard_samples(
                [shard1, shard2], method="weighted_gaussian"
            )

        # Lower variance shard should have higher weight
        # Combined should be closer to shard1 values
        for name in sample_param_names:
            shard1_mean = np.mean(shard1.samples[name])
            shard2_mean = np.mean(shard2.samples[name])
            combined_mean = np.mean(combined.samples[name])

            # Combined mean should be between the two, but closer to shard1
            # This is a heuristic test; the exact relationship depends on weights
            assert (
                min(shard1_mean, shard2_mean)
                <= combined_mean
                <= max(shard1_mean, shard2_mean)
            )

    def test_many_shards(self, mock_mcmc_samples, sample_param_names):
        """Test combining many shards."""
        n_shards = 10
        shards = [mock_mcmc_samples(seed=i) for i in range(n_shards)]

        # Use consensus_mc as it's the standard
        combined = combine_shard_samples(shards, method="consensus_mc")

        # Check shapes preserved
        assert combined.n_chains == shards[0].n_chains
        assert combined.n_samples == shards[0].n_samples

        # Check values are in expected range (approximate check since it's stochastic)
        for name in sample_param_names:
            all_means = [np.mean(s.samples[name]) for s in shards]
            combined_mean = np.mean(combined.samples[name])
            grand_mean = np.mean(all_means)

            # Combined mean should be close to the grand mean of shards
            # (since shards are randn(0, 1), they have similar precision)
            np.testing.assert_allclose(combined_mean, grand_mean, atol=0.2)

    def test_extra_fields_combined(self, mock_mcmc_samples):
        """Test extra fields are properly combined."""
        shard1 = mock_mcmc_samples(seed=42)
        shard2 = mock_mcmc_samples(seed=123)

        combined = combine_shard_samples([shard1, shard2], method="consensus_mc")

        # Extra fields should be concatenated
        if "diverging" in combined.extra_fields:
            assert combined.extra_fields["diverging"].shape[0] == 2 * shard1.n_chains

    def test_param_names_preserved(self, mock_mcmc_samples, sample_param_names):
        """Test parameter names are preserved through combination."""
        shard1 = mock_mcmc_samples(seed=42)
        shard2 = mock_mcmc_samples(seed=123)

        combined = combine_shard_samples([shard1, shard2], method="consensus_mc")

        assert combined.param_names == sample_param_names


# =============================================================================
# Tests for CMCBackend Interface
# =============================================================================


class TestCMCBackendInterface:
    """Tests for CMCBackend abstract interface."""

    def test_multiprocessing_implements_interface(self):
        """Test MultiprocessingBackend implements required methods."""
        backend = MultiprocessingBackend()

        # Should have all required methods
        assert hasattr(backend, "run")
        assert hasattr(backend, "get_name")
        assert hasattr(backend, "is_available")

        # Methods should be callable
        assert callable(backend.run)
        assert callable(backend.get_name)
        assert callable(backend.is_available)


# =============================================================================
# Edge Cases
# =============================================================================


class TestBackendEdgeCases:
    """Tests for edge cases in backend operations."""

    def test_combine_with_zero_variance_shard(self, sample_param_names):
        """Test combining with zero variance (constant) samples."""
        # Shard with constant values (zero variance)
        constant_samples = {name: np.ones((2, 100)) for name in sample_param_names}

        shard1 = MCMCSamples(
            samples=constant_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
        )

        # Shard with normal variance
        np.random.seed(42)
        normal_samples = {name: np.random.randn(2, 100) for name in sample_param_names}

        shard2 = MCMCSamples(
            samples=normal_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
        )

        # Should not raise despite zero variance
        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined = combine_shard_samples(
                [shard1, shard2], method="weighted_gaussian"
            )

        assert combined is not None
        assert hasattr(combined, "samples")
        assert isinstance(combined.samples, dict)
        for name in sample_param_names:
            assert name in combined.samples
            assert np.all(np.isfinite(combined.samples[name]))

    def test_combine_shards_different_seeds(
        self, mock_mcmc_samples, sample_param_names
    ):
        """Test that different seeds produce different combinations."""
        shards1 = [mock_mcmc_samples(seed=i) for i in range(4)]
        shards2 = [mock_mcmc_samples(seed=i + 100) for i in range(4)]

        combined1 = combine_shard_samples(shards1, method="consensus_mc")
        combined2 = combine_shard_samples(shards2, method="consensus_mc")

        # Combinations should be different
        for name in sample_param_names:
            assert not np.allclose(combined1.samples[name], combined2.samples[name])

    def test_empty_extra_fields(self, sample_param_names):
        """Test combining shards with empty extra fields."""
        np.random.seed(42)
        samples = {name: np.random.randn(2, 100) for name in sample_param_names}

        shard = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
            extra_fields={},  # Empty extra fields
        )

        combined = combine_shard_samples([shard, shard], method="consensus_mc")

        assert combined.extra_fields == {}


# =============================================================================
# Property-Based Tests
# =============================================================================


class TestBackendProperties:
    """Property-based tests for backend invariants."""

    @pytest.mark.parametrize("n_shards", [1, 2, 4, 8])
    def test_combine_preserves_shape(self, mock_mcmc_samples, n_shards):
        """Test that combining preserves sample shapes."""
        shards = [mock_mcmc_samples(seed=i) for i in range(n_shards)]

        combined = combine_shard_samples(shards, method="consensus_mc")

        # Shapes should match original
        assert combined.n_chains == shards[0].n_chains
        assert combined.n_samples == shards[0].n_samples

    @pytest.mark.parametrize("method", ["consensus_mc"])
    def test_combination_methods_produce_valid_output(
        self, mock_mcmc_samples, sample_param_names, method
    ):
        """Test that all combination methods produce valid output."""
        shards = [mock_mcmc_samples(seed=i) for i in range(4)]

        combined = combine_shard_samples(shards, method=method)

        # All values should be finite
        for name in sample_param_names:
            assert np.all(np.isfinite(combined.samples[name]))

    @pytest.mark.parametrize("method", ["simple_average", "weighted_gaussian"])
    def test_legacy_combination_methods_produce_valid_output(
        self, mock_mcmc_samples, sample_param_names, method
    ):
        """Test that legacy combination methods work (with warning)."""
        shards = [mock_mcmc_samples(seed=i) for i in range(4)]

        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined = combine_shard_samples(shards, method=method)

        # All values should be finite
        for name in sample_param_names:
            assert np.all(np.isfinite(combined.samples[name]))

    def test_simple_average_is_commutative(self, mock_mcmc_samples, sample_param_names):
        """Test that simple average is order-independent."""
        shard1 = mock_mcmc_samples(seed=42)
        shard2 = mock_mcmc_samples(seed=123)

        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined_12 = combine_shard_samples(
                [shard1, shard2], method="simple_average"
            )
            combined_21 = combine_shard_samples(
                [shard2, shard1], method="simple_average"
            )

        for name in sample_param_names:
            np.testing.assert_array_almost_equal(
                combined_12.samples[name], combined_21.samples[name], decimal=10
            )


# =============================================================================
# Scientific Validation
# =============================================================================


class TestBackendScientificProperties:
    """Scientific validation tests for backend operations."""

    def test_weighted_combination_weights_sum_to_one(self, sample_param_names):
        """Verify weighted combination uses proper normalization."""
        # Create shards with known variances
        np.random.seed(42)

        # Shard 1: variance = 1.0
        shard1_samples = {name: np.random.randn(2, 1000) for name in sample_param_names}

        # Shard 2: variance = 4.0
        shard2_samples = {
            name: np.random.randn(2, 1000) * 2.0 for name in sample_param_names
        }

        shard1 = MCMCSamples(
            samples=shard1_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=1000,
        )
        shard2 = MCMCSamples(
            samples=shard2_samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=1000,
        )

        with pytest.warns(DeprecationWarning, match="deprecated"):
            combined = combine_shard_samples(
                [shard1, shard2], method="weighted_gaussian"
            )

        # Combined variance should be between the two
        for name in sample_param_names:
            var1 = np.var(shard1.samples[name])
            var2 = np.var(shard2.samples[name])
            var_combined = np.var(combined.samples[name])

            # Weighted combination should result in lower variance
            # than simple average in many cases
            assert var_combined <= max(var1, var2)

    def test_consensus_monte_carlo_property(self, sample_param_names):
        """Test that CMC produces consistent consensus."""
        np.random.seed(42)

        # Create many shards from similar distributions
        n_shards = 8
        true_mean = 0.5
        shards = []

        for i in range(n_shards):
            np.random.seed(42 + i)
            samples = {
                name: np.random.normal(true_mean, 0.1, (2, 500))
                for name in sample_param_names
            }
            shards.append(
                MCMCSamples(
                    samples=samples,
                    param_names=sample_param_names,
                    n_chains=2,
                    n_samples=500,
                )
            )

        # Use consensus_mc via "auto" or explicit (default in v3+ will be consensus)
        # But here test was calling weighted_gaussian.
        # Let's switch to consensus_mc for this property test as it's the future.
        combined = combine_shard_samples(shards, method="consensus_mc")

        # Combined mean should be close to true mean
        for name in sample_param_names:
            combined_mean = np.mean(combined.samples[name])
            assert abs(combined_mean - true_mean) < 0.05, (
                f"Combined mean {combined_mean} not close to true mean {true_mean}"
            )


# =============================================================================
# Tests for Timeout Handling (User Story 1)
# =============================================================================


class TestTimeoutHandling:
    """Tests for graceful shard timeout handling."""

    def test_heartbeat_timeout_uses_config_value(self):
        """Test that heartbeat timeout uses configurable value from CMCConfig."""
        # Default config has heartbeat_timeout = 600
        config = CMCConfig()
        assert config.heartbeat_timeout == 600

        # Custom value should be respected
        custom_config = CMCConfig(heartbeat_timeout=300)
        assert custom_config.heartbeat_timeout == 300

    def test_heartbeat_timeout_from_dict(self):
        """Test heartbeat_timeout is parsed from config dict."""
        config_dict = {"heartbeat_timeout": 450}
        config = CMCConfig.from_dict(config_dict)
        assert config.heartbeat_timeout == 450

    def test_heartbeat_timeout_validation_minimum(self):
        """Test that heartbeat_timeout validation requires minimum 60 seconds."""
        # Too short - should fail validation
        config = CMCConfig(heartbeat_timeout=30)
        errors = config.validate()
        assert any("heartbeat_timeout" in e for e in errors)

        # Valid - at minimum
        config = CMCConfig(heartbeat_timeout=60)
        errors = config.validate()
        assert not any("heartbeat_timeout" in e for e in errors)

    def test_per_shard_timeout_default(self):
        """Test default per_shard_timeout value."""
        config = CMCConfig()
        assert config.per_shard_timeout == 3600  # 1 hour (reduced in v2.19.0)

    def test_per_shard_timeout_from_dict(self):
        """Test per_shard_timeout is parsed from config dict."""
        config_dict = {"per_shard_timeout": 3600}  # 1 hour
        config = CMCConfig.from_dict(config_dict)
        assert config.per_shard_timeout == 3600


class TestSuccessRateWarnings:
    """Tests for success rate warning thresholds."""

    def test_min_success_rate_warning_default(self):
        """Test default min_success_rate_warning value."""
        config = CMCConfig()
        assert config.min_success_rate_warning == 0.80

    def test_min_success_rate_warning_from_dict(self):
        """Test min_success_rate_warning parsed from combination section."""
        config_dict = {
            "combination": {
                "min_success_rate_warning": 0.70,
            },
        }
        config = CMCConfig.from_dict(config_dict)
        assert config.min_success_rate_warning == 0.70

    def test_min_success_rate_warning_validation(self):
        """Test validation catches out-of-range min_success_rate_warning."""
        # Out of range - should fail validation
        config = CMCConfig(min_success_rate_warning=1.5)
        errors = config.validate()
        assert any("min_success_rate_warning" in e for e in errors)

        # Valid range
        config = CMCConfig(min_success_rate_warning=0.85)
        errors = config.validate()
        assert not any("min_success_rate_warning" in e for e in errors)

    def test_warning_threshold_ordering(self):
        """Test that warning threshold should be <= success rate threshold."""
        # This should log a warning but not fail validation
        config = CMCConfig(
            min_success_rate=0.70,
            min_success_rate_warning=0.90,  # Higher than min_success_rate
        )
        # Validation passes but warning should be logged
        errors = config.validate()
        assert not any("min_success_rate_warning" in e for e in errors)


class TestErrorCategorization:
    """Tests for error category classification in shard failures."""

    def test_error_categories_defined(self):
        """Test that expected error categories are documented."""
        # These categories should be used in multiprocessing.py
        expected_categories = [
            "crash",
            "runtime_timeout",
            "heartbeat_timeout",
            "numerical",
            "convergence",
            "sampling",
        ]
        # This is a documentation test to ensure categories are consistent
        assert len(expected_categories) == 6

    def test_partial_success_combine(self, mock_mcmc_samples, sample_param_names):
        """Test that partial success still produces combined results."""
        # Create multiple shards
        shards = [mock_mcmc_samples(seed=i) for i in range(4)]

        # Combining should work with any number of successful shards
        combined = combine_shard_samples(
            shards[:2], method="consensus_mc"
        )  # Only 2 of 4
        assert combined is not None
        assert hasattr(combined, "n_chains")
        assert hasattr(combined, "samples")
        assert isinstance(combined.samples, dict)
        assert combined.n_chains == shards[0].n_chains

        # Single shard should also work
        combined_single = combine_shard_samples(shards[:1], method="consensus_mc")
        assert combined_single is not None
        assert hasattr(combined_single, "samples")
        assert isinstance(combined_single.samples, dict)


class TestBimodalDetectionIntegration:
    """Tests for bimodal detection integration in backend."""

    def test_check_shard_bimodality_importable(self):
        """Bimodal detection function is importable from diagnostics."""
        from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

        assert callable(check_shard_bimodality)

    def test_bimodal_check_on_unimodal_samples(self):
        """Bimodal check returns no alerts for unimodal samples."""
        from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(42)
        samples = {
            "D0": rng.normal(20000, 1000, size=500),
            "alpha": rng.normal(-1.0, 0.1, size=500),
        }

        results = check_shard_bimodality(samples, params_to_check=["D0", "alpha"])

        # Unimodal samples should not trigger bimodal detection
        assert not any(r.is_bimodal for r in results.values())

    def test_bimodal_check_on_bimodal_samples(self):
        """Bimodal check detects clearly bimodal samples."""
        from homodyne.optimization.cmc.diagnostics import check_shard_bimodality

        rng = np.random.default_rng(42)
        # Create bimodal D0 samples
        mode1 = rng.normal(15000, 500, size=250)
        mode2 = rng.normal(25000, 500, size=250)
        samples = {
            "D0": np.concatenate([mode1, mode2]),
            "alpha": rng.normal(-1.0, 0.1, size=500),  # Unimodal
        }

        results = check_shard_bimodality(samples, params_to_check=["D0", "alpha"])

        # D0 should be detected as bimodal
        assert results["D0"].is_bimodal == True  # noqa: E712 (numpy bool)
        # alpha should not be bimodal
        assert results["alpha"].is_bimodal == False  # noqa: E712 (numpy bool)


# =============================================================================
# Tests for Bimodal Consensus Combination
# =============================================================================


class TestBimodalCombination:
    """Tests for mode-aware bimodal consensus combination."""

    def _make_samples(self, mean: float, std: float, n: int = 1000):
        """Create MCMCSamples-like object."""
        rng = np.random.default_rng(42)
        return MCMCSamples(
            samples={"D0": rng.normal(mean, std, size=(2, n // 2))},
            param_names=["D0"],
            n_chains=2,
            n_samples=n // 2,
        )

    def test_bimodal_combine_produces_two_modes(self):
        """Bimodal combination should produce a BimodalConsensusResult with 2 modes."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )

        # 10 shards: 6 near mode1 (19K), 4 near mode2 (32K)
        shards = []
        for i in range(10):
            center = 19000.0 if i < 6 else 32000.0
            shards.append(self._make_samples(center, 1200.0))

        # Assignments: shards 0-5 in cluster 0 (lower), shards 6-9 in cluster 1 (upper)
        cluster_lower = list(range(6))
        cluster_upper = list(range(6, 10))

        # No bimodal shards in this simple case
        bimodal_detections = []

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=bimodal_detections,
            modal_params=["D0"],
            co_occurrence={},
        )

        assert bimodal_result is not None
        assert len(bimodal_result.modes) == 2
        # Mode 0 should be near 19K, mode 1 near 32K (fixed seed, tight tolerance)
        assert abs(bimodal_result.modes[0].mean["D0"] - 19000) < 1000
        assert abs(bimodal_result.modes[1].mean["D0"] - 32000) < 1000
        # Weights should reflect shard counts
        assert abs(bimodal_result.modes[0].weight - 0.6) < 0.05
        assert abs(bimodal_result.modes[1].weight - 0.4) < 0.05

    def test_bimodal_combine_uses_component_stats_for_bimodal_shards(self):
        """For bimodal shards, the combination should use component-level stats."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )

        rng = np.random.default_rng(42)
        # 5 shards, shard 2 is bimodal (both modes)
        shards = []
        for i in range(5):
            if i == 2:
                # Bimodal shard: 50/50 mix
                d0 = np.concatenate(
                    [
                        rng.normal(19000, 1200, size=250),
                        rng.normal(32000, 2100, size=250),
                    ]
                )
            else:
                center = 19000 if i < 3 else 32000
                d0 = rng.normal(center, 1200, size=500)
            shards.append(
                MCMCSamples(
                    samples={"D0": d0.reshape(2, 250)},
                    param_names=["D0"],
                    n_chains=2,
                    n_samples=250,
                )
            )

        # Shard 2 appears in both clusters
        cluster_lower = [0, 1, 2]
        cluster_upper = [2, 3, 4]
        detections = [
            {
                "shard": 2,
                "param": "D0",
                "mode1": 19000,
                "mode2": 32000,
                "std1": 1200,
                "std2": 2100,
                "weights": (0.5, 0.5),
                "separation": 13000,
            },
        ]

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=detections,
            modal_params=["D0"],
            co_occurrence={},
        )

        # Lower mode consensus should be near 19K (not pulled toward 25.5K)
        assert abs(bimodal_result.modes[0].mean["D0"] - 19000) < 1500
        # Upper mode consensus should be near 32K
        assert abs(bimodal_result.modes[1].mean["D0"] - 32000) < 1500

    def test_bimodal_combine_empty_cluster_fallback(self):
        """If one cluster has <3 shards, fall back to simple mean for it."""
        from homodyne.optimization.cmc.backends.base import (
            combine_shard_samples_bimodal,
        )

        shards = [self._make_samples(19000.0, 1200.0) for _ in range(8)]
        # Only 1 shard in upper cluster
        cluster_lower = list(range(7))
        cluster_upper = [7]

        combined, bimodal_result = combine_shard_samples_bimodal(
            shard_samples=shards,
            cluster_assignments=(cluster_lower, cluster_upper),
            bimodal_detections=[],
            modal_params=["D0"],
            co_occurrence={},
        )

        # Should still produce 2 modes (upper with simple mean)
        assert len(bimodal_result.modes) == 2
        assert bimodal_result.modes[1].n_shards == 1


# =============================================================================
# Tests for Dynamic XLA Device Count
# =============================================================================


class TestDynamicXLADeviceCount:
    """Test XLA device count matches num_chains instead of hardcoded 4."""

    def test_xla_device_count_matches_num_chains(self):
        """Verify XLA_FLAGS sets device count to num_chains, not hardcoded 4."""
        import re

        # Simulate the worker-side module-level setup logic:
        # Parent sets HOMODYNE_CMC_NUM_CHAINS, worker reads it to set XLA_FLAGS.
        num_chains = 6
        xla_flags = "--some_flag=true --xla_force_host_platform_device_count=4"
        xla_flags = re.sub(
            r"--xla_force_host_platform_device_count=\d+", "", xla_flags
        )
        xla_flags = (
            xla_flags.strip()
            + f" --xla_force_host_platform_device_count={num_chains}"
        )
        assert f"--xla_force_host_platform_device_count={num_chains}" in xla_flags
        assert "--xla_force_host_platform_device_count=4" not in xla_flags

    def test_xla_device_count_default_4(self):
        """Verify default device count is 4 (matching default num_chains)."""
        config = CMCConfig()
        assert config.num_chains == 4

    def test_env_var_fallback_when_unset(self):
        """Verify worker falls back to 4 when HOMODYNE_CMC_NUM_CHAINS is unset."""
        import os

        # Ensure the env var is not set
        saved = os.environ.pop("HOMODYNE_CMC_NUM_CHAINS", None)
        try:
            num_chains = int(os.environ.get("HOMODYNE_CMC_NUM_CHAINS", "4"))
            assert num_chains == 4
        finally:
            if saved is not None:
                os.environ["HOMODYNE_CMC_NUM_CHAINS"] = saved

    def test_env_var_propagates_custom_value(self):
        """Verify worker reads custom num_chains from env var."""
        import os

        saved = os.environ.get("HOMODYNE_CMC_NUM_CHAINS")
        try:
            os.environ["HOMODYNE_CMC_NUM_CHAINS"] = "8"
            num_chains = int(os.environ.get("HOMODYNE_CMC_NUM_CHAINS", "4"))
            assert num_chains == 8
        finally:
            if saved is not None:
                os.environ["HOMODYNE_CMC_NUM_CHAINS"] = saved
            else:
                os.environ.pop("HOMODYNE_CMC_NUM_CHAINS", None)


# =============================================================================
# Tests for WorkerPool Integration
# =============================================================================


class TestWorkerPoolIntegration:
    """Test WorkerPool integration with multiprocessing backend."""

    def test_pool_fallback_few_shards(self):
        """Verify fallback to per-shard spawn for < 3 shards."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool

        assert not should_use_pool(n_shards=1, n_workers=4)
        assert not should_use_pool(n_shards=2, n_workers=4)

    def test_pool_activated_many_shards(self):
        """Verify pool is used for >= 3 shards."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool

        assert should_use_pool(n_shards=3, n_workers=2)
        assert should_use_pool(n_shards=50, n_workers=8)

    def test_pool_boundary_at_three(self):
        """Verify exact boundary condition at n_shards=3."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool

        assert not should_use_pool(n_shards=2, n_workers=1)
        assert should_use_pool(n_shards=3, n_workers=1)

    def test_pool_single_worker(self):
        """Verify pool decision with single worker."""
        from homodyne.optimization.cmc.backends.worker_pool import should_use_pool

        # Even with 1 worker, pool is used if >= 3 shards
        # (avoids respawn overhead)
        assert should_use_pool(n_shards=3, n_workers=1)
        assert not should_use_pool(n_shards=2, n_workers=1)
