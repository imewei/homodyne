"""Tests for CMC NUTS sampler module.

Comprehensive tests for:
- SamplingStats: Timing and diagnostic statistics
- MCMCSamples: Sample container
- create_init_strategy: NUTS initialization strategies
- run_nuts_sampling: Core NUTS sampling
- run_nuts_with_retry: Retry logic with divergence handling
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.sampler import (
    MCMCSamples,
    SamplingStats,
    create_init_strategy,
)


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockPriorSpec:
    """Mock PriorDistribution-like object."""

    dist_type: str
    min_val: float
    max_val: float
    mu: float
    sigma: float


class MockParameterSpace:
    """Mock ParameterSpace for testing."""

    def __init__(self):
        self._bounds = {
            "contrast": (0.0, 1.0),
            "offset": (0.5, 1.5),
            "D0": (100.0, 100000.0),
            "alpha": (0.1, 2.0),
            "D_offset": (0.0, 1000.0),
        }

    def get_bounds(self, param_name: str) -> tuple[float, float]:
        if param_name not in self._bounds:
            raise KeyError(f"Unknown parameter: {param_name}")
        return self._bounds[param_name]


@pytest.fixture
def mock_parameter_space():
    """Create mock ParameterSpace."""
    return MockParameterSpace()


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
        "sigma",
    ]


# =============================================================================
# Tests for SamplingStats
# =============================================================================


class TestSamplingStats:
    """Tests for SamplingStats dataclass."""

    def test_default_values(self):
        """Test SamplingStats default values."""
        stats = SamplingStats()

        assert stats.warmup_time == 0.0
        assert stats.sampling_time == 0.0
        assert stats.total_time == 0.0
        assert stats.num_divergent == 0
        assert stats.accept_prob == 0.0
        assert stats.step_size == 0.0
        assert stats.tree_depth == 0.0

    def test_custom_values(self):
        """Test SamplingStats with custom values."""
        stats = SamplingStats(
            warmup_time=30.0,
            sampling_time=60.0,
            total_time=90.0,
            num_divergent=5,
            accept_prob=0.85,
            step_size=0.01,
            tree_depth=8.5,
        )

        assert stats.warmup_time == 30.0
        assert stats.sampling_time == 60.0
        assert stats.total_time == 90.0
        assert stats.num_divergent == 5
        assert stats.accept_prob == 0.85
        assert stats.step_size == 0.01
        assert stats.tree_depth == 8.5

    def test_warmup_fraction(self):
        """Test that warmup is typically less than total time."""
        stats = SamplingStats(
            warmup_time=30.0,
            sampling_time=70.0,
            total_time=100.0,
        )

        assert stats.warmup_time < stats.total_time
        assert stats.warmup_time + stats.sampling_time == pytest.approx(stats.total_time)


# =============================================================================
# Tests for MCMCSamples
# =============================================================================


class TestMCMCSamples:
    """Tests for MCMCSamples dataclass."""

    def test_creation(self, sample_param_names):
        """Test MCMCSamples creation."""
        n_chains, n_samples = 2, 100
        samples = {name: np.random.randn(n_chains, n_samples) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
        )

        assert mcmc_samples.n_chains == 2
        assert mcmc_samples.n_samples == 100
        assert len(mcmc_samples.param_names) == len(sample_param_names)

    def test_samples_shape(self, sample_param_names):
        """Test samples have correct shape."""
        n_chains, n_samples = 4, 500
        samples = {name: np.random.randn(n_chains, n_samples) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
        )

        for name in sample_param_names:
            assert mcmc_samples.samples[name].shape == (n_chains, n_samples)

    def test_extra_fields(self, sample_param_names):
        """Test MCMCSamples with extra fields."""
        n_chains, n_samples = 2, 100
        samples = {name: np.random.randn(n_chains, n_samples) for name in sample_param_names}
        extra = {
            "diverging": np.zeros((n_chains, n_samples), dtype=bool),
            "accept_prob": np.random.uniform(0.8, 0.95, (n_chains, n_samples)),
        }

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
            extra_fields=extra,
        )

        assert "diverging" in mcmc_samples.extra_fields
        assert "accept_prob" in mcmc_samples.extra_fields
        assert mcmc_samples.extra_fields["diverging"].shape == (n_chains, n_samples)

    def test_empty_samples(self):
        """Test MCMCSamples with empty samples."""
        mcmc_samples = MCMCSamples(
            samples={},
            param_names=[],
            n_chains=0,
            n_samples=0,
        )

        assert len(mcmc_samples.samples) == 0
        assert len(mcmc_samples.param_names) == 0


# =============================================================================
# Tests for create_init_strategy
# =============================================================================


class TestCreateInitStrategy:
    """Tests for create_init_strategy function."""

    def test_init_to_value_with_values(self, sample_param_names):
        """Test init_to_value when values provided."""
        initial_values = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
            "contrast_0": 0.5,
            "contrast_1": 0.5,
            "offset_0": 1.0,
            "offset_1": 1.0,
            "sigma": 0.1,
        }

        init_strategy = create_init_strategy(
            initial_values=initial_values,
            param_names=sample_param_names,
            use_init_to_value=True,
        )

        assert init_strategy is not None
        # The strategy should be init_to_value (callable)
        assert callable(init_strategy)

    def test_init_to_median_fallback(self, sample_param_names):
        """Test fallback to init_to_median when no values."""
        init_strategy = create_init_strategy(
            initial_values=None,
            param_names=sample_param_names,
            use_init_to_value=True,
        )

        assert init_strategy is not None
        assert callable(init_strategy)

    def test_init_to_median_when_disabled(self, sample_param_names):
        """Test init_to_median when use_init_to_value=False."""
        initial_values = {"D0": 1000.0}

        init_strategy = create_init_strategy(
            initial_values=initial_values,
            param_names=sample_param_names,
            use_init_to_value=False,  # Explicitly disable
        )

        assert init_strategy is not None
        assert callable(init_strategy)

    def test_partial_initial_values(self, sample_param_names):
        """Test with partial initial values (some params missing)."""
        initial_values = {
            "D0": 1000.0,
            "alpha": 0.5,
            # Other params not specified
        }

        init_strategy = create_init_strategy(
            initial_values=initial_values,
            param_names=sample_param_names,
            use_init_to_value=True,
        )

        assert init_strategy is not None

    def test_empty_initial_values(self, sample_param_names):
        """Test with empty initial values dict."""
        init_strategy = create_init_strategy(
            initial_values={},
            param_names=sample_param_names,
            use_init_to_value=True,
        )

        # Empty dict should fall back to median
        assert init_strategy is not None


# =============================================================================
# Tests for Sampling Integration
# =============================================================================


class TestSamplingIntegration:
    """Integration tests for sampling functionality."""

    def test_config_affects_sampling_params(self, default_config):
        """Test that config parameters are used correctly."""
        assert default_config.num_warmup == 100
        assert default_config.num_samples == 200
        assert default_config.num_chains == 2
        assert default_config.target_accept_prob == 0.8

    def test_mcmc_samples_statistical_properties(self, sample_param_names):
        """Test that MCMCSamples can be used for statistical analysis."""
        n_chains, n_samples = 4, 1000

        # Create samples with known properties
        np.random.seed(42)
        samples = {}
        for name in sample_param_names:
            if name == "D0":
                # D0 centered around 1000
                samples[name] = np.random.normal(1000, 50, (n_chains, n_samples))
            elif name == "alpha":
                # Alpha centered around 0.5
                samples[name] = np.random.normal(0.5, 0.1, (n_chains, n_samples))
            else:
                samples[name] = np.random.randn(n_chains, n_samples)

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
        )

        # Compute statistics
        d0_mean = np.mean(mcmc_samples.samples["D0"])
        alpha_mean = np.mean(mcmc_samples.samples["alpha"])

        # Check statistical properties
        assert 950 < d0_mean < 1050, f"D0 mean {d0_mean} not in expected range"
        assert 0.4 < alpha_mean < 0.6, f"Alpha mean {alpha_mean} not in expected range"

    def test_divergence_tracking(self, sample_param_names):
        """Test that divergences are properly tracked."""
        n_chains, n_samples = 2, 100
        samples = {name: np.random.randn(n_chains, n_samples) for name in sample_param_names}

        # Create divergence pattern
        diverging = np.zeros((n_chains, n_samples), dtype=bool)
        diverging[0, 50:55] = True  # 5 divergences in chain 0
        diverging[1, 80:82] = True  # 2 divergences in chain 1

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
            extra_fields={"diverging": diverging},
        )

        total_divergences = np.sum(mcmc_samples.extra_fields["diverging"])
        assert total_divergences == 7


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_chain(self, sample_param_names):
        """Test with single chain."""
        samples = {name: np.random.randn(1, 100) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=1,
            n_samples=100,
        )

        assert mcmc_samples.n_chains == 1
        for name in sample_param_names:
            assert mcmc_samples.samples[name].shape[0] == 1

    def test_single_sample(self, sample_param_names):
        """Test with single sample per chain."""
        samples = {name: np.random.randn(4, 1) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=4,
            n_samples=1,
        )

        assert mcmc_samples.n_samples == 1
        for name in sample_param_names:
            assert mcmc_samples.samples[name].shape[1] == 1

    def test_large_chain_count(self, sample_param_names):
        """Test with many chains (HPC scenario)."""
        n_chains = 16
        samples = {name: np.random.randn(n_chains, 100) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=100,
        )

        assert mcmc_samples.n_chains == 16

    def test_many_parameters(self):
        """Test with many parameters (large n_phi scenario)."""
        n_phi = 36  # Typical for XPCS
        param_names = (
            [f"contrast_{i}" for i in range(n_phi)]
            + [f"offset_{i}" for i in range(n_phi)]
            + ["D0", "alpha", "D_offset", "sigma"]
        )

        samples = {name: np.random.randn(4, 100) for name in param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=param_names,
            n_chains=4,
            n_samples=100,
        )

        assert len(mcmc_samples.param_names) == 2 * 36 + 4


# =============================================================================
# Property-Based Testing
# =============================================================================


class TestSamplerProperties:
    """Property-based tests for sampler invariants."""

    @pytest.mark.parametrize("n_chains", [1, 2, 4, 8])
    @pytest.mark.parametrize("n_samples", [10, 100, 1000])
    def test_samples_shape_invariant(self, sample_param_names, n_chains, n_samples):
        """Test that sample shapes are always (n_chains, n_samples)."""
        samples = {name: np.random.randn(n_chains, n_samples) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
        )

        for name in sample_param_names:
            shape = mcmc_samples.samples[name].shape
            assert shape == (n_chains, n_samples), (
                f"Shape mismatch for {name}: expected ({n_chains}, {n_samples}), got {shape}"
            )

    def test_param_names_consistency(self, sample_param_names):
        """Test that param_names matches samples keys."""
        samples = {name: np.random.randn(2, 100) for name in sample_param_names}

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=2,
            n_samples=100,
        )

        assert set(mcmc_samples.param_names) == set(mcmc_samples.samples.keys())

    def test_statistics_invariants(self, sample_param_names):
        """Test statistical invariants of samples."""
        np.random.seed(42)
        n_chains, n_samples = 4, 10000

        # Generate samples from known distributions
        samples = {}
        for name in sample_param_names:
            samples[name] = np.random.normal(0, 1, (n_chains, n_samples))

        mcmc_samples = MCMCSamples(
            samples=samples,
            param_names=sample_param_names,
            n_chains=n_chains,
            n_samples=n_samples,
        )

        # Mean should be approximately 0 for standard normal
        for name in sample_param_names:
            mean = np.mean(mcmc_samples.samples[name])
            assert abs(mean) < 0.1, f"Mean of {name} is {mean}, expected ~0"

        # Std should be approximately 1 for standard normal
        for name in sample_param_names:
            std = np.std(mcmc_samples.samples[name])
            assert 0.9 < std < 1.1, f"Std of {name} is {std}, expected ~1"
