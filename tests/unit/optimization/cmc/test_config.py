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
        assert config.num_chains == 4  # Increased from 2 for better R-hat diagnostics
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

    def test_heartbeat_timeout_default(self):
        """Test default heartbeat timeout value."""
        config = CMCConfig()
        assert config.heartbeat_timeout == 600  # 10 minutes

    def test_heartbeat_timeout_from_dict(self):
        """Test heartbeat_timeout parsed from config dict."""
        config_dict = {
            "heartbeat_timeout": 300,
        }
        config = CMCConfig.from_dict(config_dict)
        assert config.heartbeat_timeout == 300

    def test_heartbeat_timeout_validation(self):
        """Test validation catches invalid heartbeat timeout."""
        config = CMCConfig(heartbeat_timeout=30)  # Too short
        errors = config.validate()
        assert any("heartbeat_timeout" in e for e in errors)

    def test_min_success_rate_warning_default(self):
        """Test default min_success_rate_warning value."""
        config = CMCConfig()
        assert config.min_success_rate_warning == 0.80

    def test_min_success_rate_warning_from_dict(self):
        """Test min_success_rate_warning parsed from config dict."""
        config_dict = {
            "combination": {
                "min_success_rate_warning": 0.70,
            },
        }
        config = CMCConfig.from_dict(config_dict)
        assert config.min_success_rate_warning == 0.70

    def test_min_success_rate_warning_validation(self):
        """Test validation catches invalid min_success_rate_warning."""
        config = CMCConfig(min_success_rate_warning=1.5)  # Out of range
        errors = config.validate()
        assert any("min_success_rate_warning" in e for e in errors)

    def test_to_dict_includes_new_fields(self):
        """Test to_dict includes heartbeat_timeout and min_success_rate_warning."""
        config = CMCConfig(heartbeat_timeout=450, min_success_rate_warning=0.75)
        result = config.to_dict()

        assert result["heartbeat_timeout"] == 450
        assert result["combination"]["min_success_rate_warning"] == 0.75


class TestGetModelParamCount:
    """Tests for get_model_param_count function."""

    def test_static_auto_mode(self):
        """Static mode with auto per-angle has 6 params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        # Signature: get_model_param_count(n_phi, analysis_mode, per_angle_mode)
        count = get_model_param_count(n_phi=3, analysis_mode="static", per_angle_mode="auto")
        # 3 physical + 2 scaling + 1 sigma = 6
        assert count == 6

    def test_laminar_flow_auto_mode(self):
        """Laminar flow with auto per-angle has 10 params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count(n_phi=3, analysis_mode="laminar_flow", per_angle_mode="auto")
        # 7 physical + 2 scaling + 1 sigma = 10
        assert count == 10

    def test_laminar_flow_individual_mode(self):
        """Laminar flow with individual per-angle scales with n_phi."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count(n_phi=23, analysis_mode="laminar_flow", per_angle_mode="individual")
        # 7 physical + 46 scaling (2*23) + 1 sigma = 54
        assert count == 54

    def test_laminar_flow_constant_mode(self):
        """Constant mode has no sampled scaling params."""
        from homodyne.optimization.cmc.model import get_model_param_count

        count = get_model_param_count(n_phi=23, analysis_mode="laminar_flow", per_angle_mode="constant")
        # 7 physical + 0 scaling + 1 sigma = 8
        assert count == 8


class TestParamAwareShardSizing:
    """Tests for param-aware shard sizing in get_num_shards."""

    def test_no_adjustment_for_7_params(self):
        """No significant adjustment when n_params = 7 (param_factor = 1.0)."""
        # Use a larger base to avoid floor effects from min_points_per_param
        config = CMCConfig(max_points_per_shard=20000, min_points_per_param=1000)

        # 7 params: param_factor = 1.0, min_required = 7000
        # adjusted_max = max(20000, 7000) = 20000
        n_shards = config.get_num_shards(n_points=100000, n_phi=3, n_params=7)

        assert n_shards == 5  # 100000 / 20000

    def test_scales_up_for_10_params(self):
        """Shard size scales up for n_params > 7."""
        config = CMCConfig(max_points_per_shard=10000, min_points_per_param=1000)

        # 10 params: param_factor = 10/7 ≈ 1.43, min_required = 10000
        # adjusted_max = max(10000 * 1.43, 10000) = 14286
        n_shards = config.get_num_shards(n_points=100000, n_phi=3, n_params=10)

        # n_shards = 100000 / 14286 ≈ 7
        assert n_shards == 7

    def test_min_points_per_param_floor(self):
        """Shard size respects min_points_per_param floor."""
        # With 54 params and min_points_per_param=1500
        # min_required = 54 * 1500 = 81000
        config = CMCConfig(max_points_per_shard=10000, min_points_per_param=1500)

        n_shards = config.get_num_shards(n_points=500000, n_phi=23, n_params=54)

        # param_factor = 54/7 = 7.71, so 10000 * 7.71 = 77143
        # But min_required = 81000 takes precedence
        # n_shards = 500000 / 81000 ≈ 6
        assert n_shards == 6

    def test_backward_compatible_without_n_params(self):
        """get_num_shards works without n_params for backward compatibility."""
        config = CMCConfig(max_points_per_shard=20000, min_points_per_param=1000)

        # Default n_params=7: param_factor=1.0, min_required=7000
        # adjusted_max = max(20000, 7000) = 20000
        n_shards = config.get_num_shards(n_points=100000, n_phi=3)

        assert n_shards == 5  # 100000 / 20000

    def test_fewer_shards_with_more_params(self):
        """More parameters results in larger shards (fewer shards)."""
        config = CMCConfig(max_points_per_shard=20000, min_points_per_param=1000)

        n_shards_7 = config.get_num_shards(n_points=200000, n_phi=5, n_params=7)
        n_shards_14 = config.get_num_shards(n_points=200000, n_phi=5, n_params=14)

        # More params → larger adjusted_max → fewer shards
        assert n_shards_14 < n_shards_7
