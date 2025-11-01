"""Unit Tests for CMC Configuration System
==========================================

Tests for Consensus Monte Carlo configuration parsing, validation,
and TypedDict type safety.

Test Coverage:
- CMC config parsing from YAML
- Default values application
- Validation of invalid settings
- ConfigManager.get_cmc_config() method
- TypedDict type checking
- Backward compatibility
- Migration warnings for deprecated settings
- Nested configuration merging
"""

import pytest
import tempfile
from pathlib import Path
from typing import Any

from homodyne.config.manager import ConfigManager
from homodyne.config.types import (
    CMCConfig,
    CMCShardingConfig,
    CMCInitializationConfig,
    CMCBackendConfig,
    CMCCombinationConfig,
    CMCPerShardMCMCConfig,
    CMCValidationConfig,
)


class TestCMCConfigParsing:
    """Test CMC configuration parsing from YAML."""

    def test_parse_minimal_cmc_config(self):
        """Test parsing minimal CMC configuration with defaults."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  method: cmc
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Check that enable was parsed
            assert cmc_config["enable"] is True

            # Check that defaults were applied
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"
            # Note: initialization section removed in v2.1.0 (no more SVI)
            assert cmc_config["backend"]["name"] == "auto"
            assert cmc_config["combination"]["method"] == "weighted_gaussian"

        finally:
            Path(config_path).unlink()

    def test_parse_complete_cmc_config(self):
        """Test parsing complete CMC configuration with all fields."""
        config_yaml = """
analysis_mode: laminar_flow

optimization:
  method: cmc
  cmc:
    enable: auto
    min_points_for_cmc: 1000000
    sharding:
      strategy: random
      num_shards: 16
      max_points_per_shard: 500000
    backend:
      name: pjit
      enable_checkpoints: false
      checkpoint_frequency: 5
      checkpoint_dir: ./my_checkpoints
      keep_last_checkpoints: 5
      resume_from_checkpoint: false
    combination:
      method: simple_average
      validate_results: false
      min_success_rate: 0.80
    per_shard_mcmc:
      num_warmup: 1000
      num_samples: 3000
      num_chains: 2
      subsample_size: 100000
    validation:
      strict_mode: false
      min_per_shard_ess: 50.0
      max_per_shard_rhat: 1.2
      max_between_shard_kl: 3.0
      min_success_rate: 0.80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Validate all fields were parsed correctly
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 1000000

            # Sharding
            assert cmc_config["sharding"]["strategy"] == "random"
            assert cmc_config["sharding"]["num_shards"] == 16
            assert cmc_config["sharding"]["max_points_per_shard"] == 500000

            # Note: Initialization section removed in v2.1.0 (no more SVI)

            # Backend
            assert cmc_config["backend"]["name"] == "pjit"
            assert cmc_config["backend"]["enable_checkpoints"] is False
            assert cmc_config["backend"]["checkpoint_frequency"] == 5

            # Combination
            assert cmc_config["combination"]["method"] == "simple_average"
            assert cmc_config["combination"]["validate_results"] is False
            assert cmc_config["combination"]["min_success_rate"] == 0.80

            # Per-shard MCMC
            assert cmc_config["per_shard_mcmc"]["num_warmup"] == 1000
            assert cmc_config["per_shard_mcmc"]["num_samples"] == 3000
            assert cmc_config["per_shard_mcmc"]["num_chains"] == 2

            # Validation
            assert cmc_config["validation"]["strict_mode"] is False
            assert cmc_config["validation"]["min_per_shard_ess"] == 50.0
            assert cmc_config["validation"]["max_per_shard_rhat"] == 1.2

        finally:
            Path(config_path).unlink()

    def test_parse_partial_cmc_config_with_defaults(self):
        """Test that partial config merges correctly with defaults."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  method: cmc
  cmc:
    enable: true
    sharding:
      strategy: contiguous
    initialization:
      method: identity
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # User-specified values
            assert cmc_config["enable"] is True
            assert cmc_config["sharding"]["strategy"] == "contiguous"

            # Default values preserved
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["num_shards"] == "auto"
            assert cmc_config["backend"]["name"] == "auto"

        finally:
            Path(config_path).unlink()


class TestCMCConfigDefaults:
    """Test default CMC configuration values."""

    def test_defaults_without_cmc_section(self):
        """Test that defaults are returned when no CMC section exists."""
        config_yaml = """
analysis_mode: static_isotropic
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Should return complete default configuration
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"
            assert cmc_config["backend"]["name"] == "auto"
            assert cmc_config["combination"]["method"] == "weighted_gaussian"
            assert cmc_config["per_shard_mcmc"]["num_warmup"] == 500
            assert cmc_config["validation"]["strict_mode"] is True

        finally:
            Path(config_path).unlink()

    def test_defaults_with_empty_cmc_section(self):
        """Test defaults applied when CMC section is empty."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  method: cmc
  cmc: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # All defaults should be applied
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"

        finally:
            Path(config_path).unlink()


class TestCMCConfigValidation:
    """Test CMC configuration validation."""

    def test_invalid_enable_value(self):
        """Test validation catches invalid enable value."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    enable: "maybe"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="CMC enable must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_sharding_strategy(self):
        """Test validation catches invalid sharding strategy."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    sharding:
      strategy: invalid_strategy
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="Sharding strategy must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_num_shards(self):
        """Test validation catches invalid num_shards."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    sharding:
      num_shards: -5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="num_shards must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_initialization_method(self):
        """Test validation catches invalid initialization method."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    initialization:
      method: random_init
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="Initialization method must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_backend_name(self):
        """Test validation catches invalid backend name."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    backend:
      name: spark
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="Backend name must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_min_success_rate(self):
        """Test validation catches invalid min_success_rate."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    combination:
      min_success_rate: 1.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="min_success_rate must be between"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_per_shard_num_warmup(self):
        """Test validation catches invalid num_warmup."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    per_shard_mcmc:
      num_warmup: -100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="num_warmup must be a positive integer"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_rhat_threshold(self):
        """Test validation catches invalid max_per_shard_rhat."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    validation:
      max_per_shard_rhat: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="max_per_shard_rhat must be >= 1.0"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()


class TestCMCConfigManagerMethod:
    """Test ConfigManager.get_cmc_config() method."""

    def test_get_cmc_config_returns_dict(self):
        """Test that get_cmc_config() returns a dictionary."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            assert isinstance(cmc_config, dict)
            assert "enable" in cmc_config
            assert "sharding" in cmc_config
            assert "initialization" in cmc_config
            assert "backend" in cmc_config
            assert "combination" in cmc_config
            assert "per_shard_mcmc" in cmc_config
            assert "validation" in cmc_config

        finally:
            Path(config_path).unlink()

    def test_get_cmc_config_with_config_override(self):
        """Test get_cmc_config() with config_override parameter."""
        config_dict = {
            "analysis_mode": "static_isotropic",
            "optimization": {
                "cmc": {
                    "enable": True,
                    "sharding": {"strategy": "random"},
                }
            },
        }

        config_mgr = ConfigManager("dummy.yaml", config_override=config_dict)
        cmc_config = config_mgr.get_cmc_config()

        assert cmc_config["enable"] is True
        assert cmc_config["sharding"]["strategy"] == "random"


class TestCMCTypedDictCompatibility:
    """Test TypedDict type hints for CMC configuration."""

    def test_cmc_sharding_config_structure(self):
        """Test CMCShardingConfig TypedDict structure."""
        sharding_config: CMCShardingConfig = {
            "strategy": "stratified",
            "num_shards": "auto",
            "max_points_per_shard": "auto",
        }

        assert sharding_config["strategy"] == "stratified"
        assert sharding_config["num_shards"] == "auto"

    def test_cmc_initialization_config_structure(self):
        """Test CMCInitializationConfig TypedDict structure."""
        init_config: CMCInitializationConfig = {
            "method": "svi",
            "svi_steps": 5000,
            "svi_learning_rate": 0.001,
            "svi_rank": 5,
            "fallback_to_identity": True,
        }

        assert init_config["method"] == "svi"
        assert init_config["svi_steps"] == 5000

    def test_cmc_config_full_structure(self):
        """Test complete CMCConfig TypedDict structure."""
        cmc_config: CMCConfig = {
            "enable": "auto",
            "min_points_for_cmc": 500000,
            "sharding": {
                "strategy": "stratified",
                "num_shards": "auto",
                "max_points_per_shard": "auto",
            },
            "initialization": {
                "method": "svi",
                "svi_steps": 5000,
                "svi_learning_rate": 0.001,
                "svi_rank": 5,
                "fallback_to_identity": True,
            },
            "backend": {
                "name": "auto",
                "enable_checkpoints": True,
                "checkpoint_frequency": 10,
                "checkpoint_dir": "./checkpoints/cmc",
                "keep_last_checkpoints": 3,
                "resume_from_checkpoint": True,
            },
            "combination": {
                "method": "weighted_gaussian",
                "validate_results": True,
                "min_success_rate": 0.90,
            },
            "per_shard_mcmc": {
                "num_warmup": 500,
                "num_samples": 2000,
                "num_chains": 1,
                "subsample_size": "auto",
            },
            "validation": {
                "strict_mode": True,
                "min_per_shard_ess": 100.0,
                "max_per_shard_rhat": 1.1,
                "max_between_shard_kl": 2.0,
                "min_success_rate": 0.90,
            },
        }

        assert cmc_config["enable"] == "auto"
        assert cmc_config["sharding"]["strategy"] == "stratified"


class TestCMCDeprecationWarnings:
    """Test deprecation warnings for old CMC settings."""

    def test_deprecated_consensus_monte_carlo_key(self, caplog):
        """Test warning for deprecated 'consensus_monte_carlo' key."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  consensus_monte_carlo:
    enable: true
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            config_mgr.get_cmc_config()

            # Check for deprecation warning
            assert any("consensus_monte_carlo" in record.message for record in caplog.records)

        finally:
            Path(config_path).unlink()

    def test_deprecated_optimal_shard_size_key(self, caplog):
        """Test warning for deprecated 'optimal_shard_size' key."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  cmc:
    sharding:
      optimal_shard_size: 1000000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            config_mgr.get_cmc_config()

            # Check for deprecation warning
            assert any("optimal_shard_size" in record.message for record in caplog.records)

        finally:
            Path(config_path).unlink()


class TestCMCBackwardCompatibility:
    """Test backward compatibility with existing configurations."""

    def test_old_config_without_cmc_still_works(self):
        """Test that old configs without CMC section still work."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  method: nlsq
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            # Should not raise error
            cmc_config = config_mgr.get_cmc_config()

            # Should return defaults
            assert cmc_config["enable"] == "auto"

        finally:
            Path(config_path).unlink()

    def test_nlsq_method_with_cmc_config(self):
        """Test NLSQ method can coexist with CMC config."""
        config_yaml = """
analysis_mode: static_isotropic

optimization:
  method: nlsq
  cmc:
    enable: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            assert cmc_config["enable"] is False

        finally:
            Path(config_path).unlink()
