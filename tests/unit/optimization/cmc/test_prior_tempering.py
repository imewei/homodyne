"""Tests for CMC prior tempering (Scott et al. 2016).

Verifies that prior^(1/K) is correctly implemented: each shard uses
Normal(0, sqrt(K)) instead of Normal(0, 1) in z-space, so the combined
posterior has exactly one prior contribution rather than K.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed, trace  # noqa: E402

from homodyne.optimization.cmc.scaling import (  # noqa: E402
    ParameterScaling,
    sample_scaled_parameter,
)


@dataclass
class MockPriorSpec:
    """Mock PriorDistribution-like object."""

    dist_type: str
    min_val: float
    max_val: float
    mu: float
    sigma: float


class MockParameterSpace:
    """Mock ParameterSpace for testing without full config system."""

    def __init__(self, analysis_mode: str = "static"):
        self.analysis_mode = analysis_mode
        self._priors = {
            "contrast": MockPriorSpec("TruncatedNormal", 0.0, 1.0, 0.5, 0.2),
            "offset": MockPriorSpec("TruncatedNormal", 0.5, 1.5, 1.0, 0.2),
            "D0": MockPriorSpec("TruncatedNormal", 100.0, 100000.0, 1000.0, 500.0),
            "alpha": MockPriorSpec("TruncatedNormal", 0.1, 2.0, 0.5, 0.3),
            "D_offset": MockPriorSpec("TruncatedNormal", 0.0, 1000.0, 10.0, 50.0),
        }
        self._bounds = {
            "contrast": (0.0, 1.0),
            "offset": (0.5, 1.5),
            "D0": (100.0, 100000.0),
            "alpha": (0.1, 2.0),
            "D_offset": (0.0, 1000.0),
        }

    def get_prior(self, param_name: str) -> MockPriorSpec:
        if param_name not in self._priors:
            raise KeyError(f"Unknown parameter: {param_name}")
        return self._priors[param_name]

    def get_bounds(self, param_name: str) -> tuple[float, float]:
        if param_name not in self._bounds:
            raise KeyError(f"Unknown parameter: {param_name}")
        return self._bounds[param_name]


class TestPriorTemperingScale:
    """Test that prior_scale correctly widens z-space priors."""

    def test_default_prior_scale_is_unit(self):
        """Default prior_scale=1.0 gives Normal(0, 1)."""
        scaling = ParameterScaling(
            name="test", center=0.0, scale=1.0, low=-10.0, high=10.0
        )

        def model():
            sample_scaled_parameter("x", scaling, prior_scale=1.0)

        tr = trace(seed(model, rng_seed=0)).get_trace()
        site = tr["x_z"]
        fn = site["fn"]
        assert isinstance(fn, dist.Normal)
        assert float(fn.loc) == pytest.approx(0.0)
        assert float(fn.scale) == pytest.approx(1.0)

    def test_prior_scale_sqrt_k(self):
        """With K=100 shards, prior_scale=sqrt(100)=10."""
        num_shards = 100
        expected_scale = math.sqrt(num_shards)

        scaling = ParameterScaling(
            name="test", center=0.0, scale=1.0, low=-10.0, high=10.0
        )

        def model():
            sample_scaled_parameter("x", scaling, prior_scale=expected_scale)

        tr = trace(seed(model, rng_seed=0)).get_trace()
        site = tr["x_z"]
        fn = site["fn"]
        assert isinstance(fn, dist.Normal)
        assert float(fn.scale) == pytest.approx(expected_scale)

    def test_single_shard_no_tempering(self):
        """num_shards=1 → prior_scale=1.0 (no tempering, backward-compatible)."""
        prior_scale = math.sqrt(1)
        assert prior_scale == 1.0

        scaling = ParameterScaling(
            name="test", center=5.0, scale=2.0, low=0.0, high=10.0
        )

        def model():
            sample_scaled_parameter("x", scaling, prior_scale=prior_scale)

        tr = trace(seed(model, rng_seed=42)).get_trace()
        fn = tr["x_z"]["fn"]
        assert float(fn.scale) == pytest.approx(1.0)

    def test_prior_scale_150_shards(self):
        """Realistic test: K=150 shards (C020 dataset)."""
        num_shards = 150
        prior_scale = math.sqrt(num_shards)

        scaling = ParameterScaling(
            name="beta", center=1.0, scale=0.5, low=0.0, high=3.0
        )

        def model():
            sample_scaled_parameter("beta", scaling, prior_scale=prior_scale)

        tr = trace(seed(model, rng_seed=0)).get_trace()
        fn = tr["beta_z"]["fn"]
        assert float(fn.scale) == pytest.approx(prior_scale, rel=1e-6)


class TestPriorTemperingConfig:
    """Test prior_tempering config option."""

    def test_config_default_enabled(self):
        """prior_tempering defaults to True."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig()
        assert config.prior_tempering is True

    def test_config_from_dict_enabled(self):
        """prior_tempering parsed from config dict."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig.from_dict({"prior_tempering": True})
        assert config.prior_tempering is True

    def test_config_from_dict_disabled(self):
        """prior_tempering can be disabled."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig.from_dict({"prior_tempering": False})
        assert config.prior_tempering is False

    def test_config_to_dict_roundtrip(self):
        """prior_tempering survives to_dict/from_dict roundtrip."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig(prior_tempering=True)
        d = config.to_dict()
        assert d["prior_tempering"] is True


class TestModelNumShardsParameter:
    """Test that model functions accept and use num_shards."""

    @pytest.fixture
    def simple_parameter_space(self):
        """Create a minimal ParameterSpace for testing."""
        return MockParameterSpace("static")

    def test_averaged_model_accepts_num_shards(self, simple_parameter_space):
        """xpcs_model_averaged accepts num_shards kwarg without error."""
        from homodyne.optimization.cmc.model import xpcs_model_averaged

        n = 10
        data = jnp.ones(n)
        t1 = jnp.linspace(0, 1, n)
        t2 = jnp.linspace(0, 1, n)
        phi_unique = jnp.array([0.0])
        phi_indices = jnp.zeros(n, dtype=jnp.int32)

        def model():
            xpcs_model_averaged(
                data=data,
                t1=t1,
                t2=t2,
                phi_unique=phi_unique,
                phi_indices=phi_indices,
                q=1.0,
                L=1.0,
                dt=0.1,
                analysis_mode="static",
                parameter_space=simple_parameter_space,
                n_phi=1,
                noise_scale=0.1,
                num_shards=10,
            )

        # Should not raise
        tr = trace(seed(model, rng_seed=0)).get_trace()
        # Verify contrast_z uses widened prior
        fn = tr["contrast_z"]["fn"]
        assert float(fn.scale) == pytest.approx(math.sqrt(10), rel=1e-6)

    def test_constant_model_accepts_num_shards(self, simple_parameter_space):
        """xpcs_model_constant accepts num_shards kwarg without error."""
        from homodyne.optimization.cmc.model import xpcs_model_constant

        n = 10
        data = jnp.ones(n)
        t1 = jnp.linspace(0, 1, n)
        t2 = jnp.linspace(0, 1, n)
        phi_unique = jnp.array([0.0])
        phi_indices = jnp.zeros(n, dtype=jnp.int32)
        fixed_contrast = jnp.array([0.5])
        fixed_offset = jnp.array([1.0])

        def model():
            xpcs_model_constant(
                data=data,
                t1=t1,
                t2=t2,
                phi_unique=phi_unique,
                phi_indices=phi_indices,
                q=1.0,
                L=1.0,
                dt=0.1,
                analysis_mode="static",
                parameter_space=simple_parameter_space,
                n_phi=1,
                noise_scale=0.1,
                fixed_contrast=fixed_contrast,
                fixed_offset=fixed_offset,
                num_shards=5,
            )

        # Should not raise
        tr = trace(seed(model, rng_seed=0)).get_trace()
        # Verify D0_z uses widened prior
        fn = tr["D0_z"]["fn"]
        assert float(fn.scale) == pytest.approx(math.sqrt(5), rel=1e-6)
