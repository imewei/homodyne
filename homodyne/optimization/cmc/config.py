"""CMC configuration dataclass and validation.

This module provides the CMCConfig dataclass for parsing and validating
CMC-specific configuration settings from the YAML config file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CMCConfig:
    """Configuration for Consensus Monte Carlo (CMC) analysis.

    Attributes
    ----------
    enable : bool | str
        Whether to enable CMC. "auto" enables based on data size.
    min_points_for_cmc : int
        Minimum data points to trigger CMC mode.
    sharding_strategy : str
        How to partition data: "stratified", "random", "contiguous".
    num_shards : int | str
        Number of data shards. "auto" calculates from data size.
    max_points_per_shard : int | str
        Maximum points per shard. "auto" calculates optimally.
    backend_name : str
        Execution backend: "auto", "multiprocessing", "pjit", "pbs".
    enable_checkpoints : bool
        Whether to save checkpoints during sampling.
    checkpoint_dir : str
        Directory for checkpoint files.
    num_warmup : int
        Number of warmup/burn-in samples per chain.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    target_accept_prob : float
        Target acceptance probability for NUTS.
    max_r_hat : float
        Maximum R-hat for convergence.
    min_ess : float
        Minimum effective sample size.
    combination_method : str
        How to combine shard posteriors: "weighted_gaussian", "simple_average".
    min_success_rate : float
        Minimum fraction of shards that must succeed.
    run_id : str | None
        Optional identifier used for structured logging across shards.
    """

    # Enable settings
    enable: bool | str = "auto"
    min_points_for_cmc: int = 500000

    # Sharding
    sharding_strategy: str = "stratified"
    num_shards: int | str = "auto"
    max_points_per_shard: int | str = "auto"

    # Backend
    backend_name: str = "auto"
    enable_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/cmc"

    # Sampling
    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 4
    target_accept_prob: float = 0.8

    # Validation thresholds
    max_r_hat: float = 1.1
    min_ess: float = 100.0

    # Combination
    combination_method: str = "weighted_gaussian"
    min_success_rate: float = 0.90
    run_id: str | None = None

    # Computed fields
    _validation_errors: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CMCConfig:
        """Create CMCConfig from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            CMC configuration dictionary from ConfigManager.get_cmc_config().

        Returns
        -------
        CMCConfig
            Validated configuration object.

        Raises
        ------
        ValueError
            If required fields are missing or invalid.
        """
        # Extract nested sections
        sharding = config_dict.get("sharding", {})
        backend = config_dict.get("backend", {})
        per_shard = config_dict.get("per_shard_mcmc", {})
        validation = config_dict.get("validation", {})
        combination = config_dict.get("combination", {})

        # Handle both old dict schema and new string schema for backend
        if isinstance(backend, str):
            # New schema: backend is computational backend string
            backend_name = config_dict.get("backend_config", {}).get("name", "auto")
            enable_checkpoints = config_dict.get("backend_config", {}).get(
                "enable_checkpoints", True
            )
            checkpoint_dir = config_dict.get("backend_config", {}).get(
                "checkpoint_dir", "./checkpoints/cmc"
            )
        else:
            # Old schema: backend is dict
            backend_name = backend.get("name", "auto")
            enable_checkpoints = backend.get("enable_checkpoints", True)
            checkpoint_dir = backend.get("checkpoint_dir", "./checkpoints/cmc")

        config = cls(
            # Enable settings
            enable=config_dict.get("enable", "auto"),
            min_points_for_cmc=config_dict.get("min_points_for_cmc", 500000),
            # Sharding
            sharding_strategy=sharding.get("strategy", "stratified"),
            num_shards=sharding.get("num_shards", "auto"),
            max_points_per_shard=sharding.get("max_points_per_shard", "auto"),
            # Backend
            backend_name=backend_name,
            enable_checkpoints=enable_checkpoints,
            checkpoint_dir=checkpoint_dir,
            # Sampling
            num_warmup=per_shard.get("num_warmup", 500),
            num_samples=per_shard.get("num_samples", 2000),
            num_chains=per_shard.get("num_chains", 4),
            target_accept_prob=per_shard.get("target_accept_prob", 0.8),
            # Validation
            max_r_hat=validation.get("max_per_shard_rhat", 1.1),
            min_ess=validation.get("min_per_shard_ess", 100.0),
            # Combination
            combination_method=combination.get("method", "weighted_gaussian"),
            min_success_rate=combination.get("min_success_rate", 0.90),
            run_id=config_dict.get("run_id"),
        )

        # Validate and log any issues
        errors = config.validate()
        if errors:
            for error in errors:
                logger.warning(f"CMC config validation: {error}")

        return config

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate enable
        if self.enable not in [True, False, "auto"]:
            errors.append(f"enable must be True, False, or 'auto', got: {self.enable}")

        # Validate min_points_for_cmc
        if not isinstance(self.min_points_for_cmc, int) or self.min_points_for_cmc < 0:
            errors.append(
                f"min_points_for_cmc must be non-negative int, got: {self.min_points_for_cmc}"
            )

        # Validate sharding_strategy
        valid_strategies = ["stratified", "random", "contiguous"]
        if self.sharding_strategy not in valid_strategies:
            errors.append(
                f"sharding_strategy must be one of {valid_strategies}, got: {self.sharding_strategy}"
            )

        # Validate num_shards
        if self.num_shards != "auto":
            if not isinstance(self.num_shards, int) or self.num_shards <= 0:
                errors.append(
                    f"num_shards must be 'auto' or positive int, got: {self.num_shards}"
                )

        # Validate backend_name
        valid_backends = ["auto", "multiprocessing", "pjit", "pbs", "slurm"]
        if self.backend_name not in valid_backends:
            errors.append(
                f"backend_name must be one of {valid_backends}, got: {self.backend_name}"
            )

        # Validate sampling parameters
        if not isinstance(self.num_warmup, int) or self.num_warmup <= 0:
            errors.append(f"num_warmup must be positive int, got: {self.num_warmup}")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            errors.append(f"num_samples must be positive int, got: {self.num_samples}")
        if not isinstance(self.num_chains, int) or self.num_chains <= 0:
            errors.append(f"num_chains must be positive int, got: {self.num_chains}")

        # Validate target_accept_prob
        if not 0.0 < self.target_accept_prob < 1.0:
            errors.append(
                f"target_accept_prob must be in (0, 1), got: {self.target_accept_prob}"
            )

        # Validate convergence thresholds
        if not isinstance(self.max_r_hat, (int, float)) or self.max_r_hat < 1.0:
            errors.append(f"max_r_hat must be >= 1.0, got: {self.max_r_hat}")
        if not isinstance(self.min_ess, (int, float)) or self.min_ess < 0:
            errors.append(f"min_ess must be non-negative, got: {self.min_ess}")

        # Validate combination settings
        valid_methods = ["weighted_gaussian", "simple_average", "auto"]
        if self.combination_method not in valid_methods:
            errors.append(
                f"combination_method must be one of {valid_methods}, got: {self.combination_method}"
            )
        if not 0.0 <= self.min_success_rate <= 1.0:
            errors.append(
                f"min_success_rate must be in [0, 1], got: {self.min_success_rate}"
            )

        self._validation_errors = errors
        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns
        -------
        bool
            True if configuration has no validation errors.
        """
        return len(self.validate()) == 0

    def should_enable_cmc(self, n_points: int) -> bool:
        """Determine if CMC should be enabled for given data size.

        Parameters
        ----------
        n_points : int
            Number of data points.

        Returns
        -------
        bool
            True if CMC should be enabled.
        """
        if self.enable is True:
            return True
        if self.enable is False:
            return False
        # "auto" mode
        return n_points >= self.min_points_for_cmc

    def get_num_shards(self, n_points: int, n_phi: int) -> int:
        """Calculate number of shards for given data.

        Parameters
        ----------
        n_points : int
            Total number of data points.
        n_phi : int
            Number of phi angles.

        Returns
        -------
        int
            Number of shards to use.
        """
        if isinstance(self.num_shards, int):
            return self.num_shards

        # Auto calculation: stratified by phi angle
        if self.sharding_strategy == "stratified":
            return n_phi

        # For other strategies, calculate based on max_points_per_shard
        if isinstance(self.max_points_per_shard, int):
            max_per_shard = self.max_points_per_shard
        else:
            # Default: ~100k points per shard
            max_per_shard = 100000

        return max(1, n_points // max_per_shard)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            "enable": self.enable,
            "min_points_for_cmc": self.min_points_for_cmc,
            "run_id": self.run_id,
            "sharding": {
                "strategy": self.sharding_strategy,
                "num_shards": self.num_shards,
                "max_points_per_shard": self.max_points_per_shard,
            },
            "backend": {
                "name": self.backend_name,
                "enable_checkpoints": self.enable_checkpoints,
                "checkpoint_dir": self.checkpoint_dir,
            },
            "per_shard_mcmc": {
                "num_warmup": self.num_warmup,
                "num_samples": self.num_samples,
                "num_chains": self.num_chains,
                "target_accept_prob": self.target_accept_prob,
            },
            "validation": {
                "max_per_shard_rhat": self.max_r_hat,
                "min_per_shard_ess": self.min_ess,
            },
            "combination": {
                "method": self.combination_method,
                "min_success_rate": self.min_success_rate,
            },
        }
