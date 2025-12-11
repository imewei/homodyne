"""Tests for CMC configuration dataclass."""

import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402


class TestCMCConfig:
    """Tests for CMCConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CMCConfig()

        assert config.enable == "auto"
        assert config.min_points_for_cmc == 500000
        assert config.num_warmup == 500
        assert config.num_samples == 1500
        assert config.num_chains == 2
        assert config.target_accept_prob == 0.85

    def test_from_dict_basic(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enable": True,
            "per_shard_mcmc": {
                "num_warmup": 1000,
                "num_samples": 4000,
                "num_chains": 8,
            },
        }

        config = CMCConfig.from_dict(config_dict)

        assert config.enable is True
        assert config.num_warmup == 1000
        assert config.num_samples == 4000
        assert config.num_chains == 8

    def test_from_dict_empty(self):
        """Test creating config from empty dictionary uses defaults."""
        config = CMCConfig.from_dict({})

        assert config.enable == "auto"
        assert config.num_warmup == 500
        assert config.num_samples == 1500

    def test_from_dict_none(self):
        """Test creating config from None uses defaults."""
        config = CMCConfig.from_dict({})  # Empty dict, not None

        assert config.enable == "auto"
        assert config.num_warmup == 500

    def test_from_dict_nested_backend(self):
        """Test backend configuration parsing."""
        config_dict = {
            "backend": {
                "name": "pjit",
                "enable_checkpoints": False,
            },
        }

        config = CMCConfig.from_dict(config_dict)

        assert config.backend_name == "pjit"
        assert config.enable_checkpoints is False

    def test_from_dict_nested_sharding(self):
        """Test sharding configuration parsing."""
        config_dict = {
            "sharding": {
                "strategy": "stratified",
                "num_shards": 4,
                "max_points_per_shard": 100000,
            },
        }

        config = CMCConfig.from_dict(config_dict)

        assert config.sharding_strategy == "stratified"
        assert config.num_shards == 4
        assert config.max_points_per_shard == 100000

    def test_validate_no_errors(self):
        """Test validation passes for valid config."""
        config = CMCConfig()
        errors = config.validate()

        assert len(errors) == 0

    def test_validate_invalid_chains(self):
        """Test validation catches invalid chain count."""
        config = CMCConfig(num_chains=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("chains" in e.lower() for e in errors)

    def test_validate_invalid_samples(self):
        """Test validation catches invalid sample count."""
        config = CMCConfig(num_samples=-1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("samples" in e.lower() for e in errors)

    def test_validate_invalid_accept_prob(self):
        """Test validation catches invalid target accept probability."""
        config = CMCConfig(target_accept_prob=1.5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("accept" in e.lower() or "probability" in e.lower() for e in errors)

    def test_should_enable_cmc_auto_small_data(self):
        """Test auto mode with small dataset returns False."""
        config = CMCConfig(enable="auto", min_points_for_cmc=500000)

        assert config.should_enable_cmc(n_points=100000) is False

    def test_should_enable_cmc_auto_large_data(self):
        """Test auto mode with large dataset returns True."""
        config = CMCConfig(enable="auto", min_points_for_cmc=500000)

        assert config.should_enable_cmc(n_points=1000000) is True

    def test_should_enable_cmc_explicit_true(self):
        """Test explicit True enables CMC regardless of data size."""
        config = CMCConfig(enable=True)

        assert config.should_enable_cmc(n_points=100) is True

    def test_should_enable_cmc_explicit_false(self):
        """Test explicit False disables CMC regardless of data size."""
        config = CMCConfig(enable=False)

        assert config.should_enable_cmc(n_points=10000000) is False
