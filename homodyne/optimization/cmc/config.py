"""CMC configuration dataclass and validation.

This module provides the CMCConfig dataclass for parsing and validating
CMC-specific configuration settings from the YAML config file.

Config Precedence (Important)
-----------------------------
The CLI reads base `optimization.mcmc` settings and applies them to
`per_shard_mcmc`. This means if base mcmc differs from per_shard_mcmc
in your YAML config, the CLI will overwrite per_shard_mcmc with base
values. To avoid surprises, keep base mcmc and per_shard_mcmc aligned.

Example aligned config::

    optimization:
      mcmc:
        num_warmup: 500
        num_samples: 1500
        num_chains: 2
      cmc:
        per_shard_mcmc:
          num_warmup: 500
          num_samples: 1500
          num_chains: 2
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
        Default: 100K points/shard. For typical 3M point, 3-angle laminar
        flow datasets, this produces 18-30 shards (not 150+ as with 20K).
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
        How to combine shard posteriors. Options:

        - ``"consensus_mc"``: Correct Consensus Monte Carlo (precision-weighted means).
          Recommended. Combines per-shard posterior moments, then generates new
          samples from the combined Gaussian.
        - ``"weighted_gaussian"``: Legacy element-wise weighted averaging (deprecated).
        - ``"simple_average"``: Simple element-wise averaging (deprecated).

    min_success_rate : float
        Minimum fraction of shards that must succeed.
    run_id : str | None
        Optional identifier used for structured logging across shards.
    per_angle_mode : str
        Per-angle scaling mode for anti-degeneracy defense (v2.18.0+):

        - ``"auto"``: Auto-selects based on n_phi threshold (recommended).
          When n_phi >= threshold: Estimates per-angle values, AVERAGES them,
          broadcasts single value to all angles (matches NLSQ behavior).
          When n_phi < threshold: Uses individual mode.
        - ``"constant"``: Per-angle contrast/offset from quantile estimation,
          used DIRECTLY (different fixed value per angle, NOT averaged).
          Reduces to 8 params (7 physical + 1 sigma).
        - ``"individual"``: Independent contrast + offset per angle, all sampled.
          May suffer from parameter absorption degeneracy with many angles.

    constant_scaling_threshold : int
        n_phi threshold for auto mode to use constant scaling.
        When n_phi >= threshold, uses constant mode. Default: 3.
    """

    # Enable settings
    enable: bool | str = "auto"
    min_points_for_cmc: int = 500000

    # Anti-degeneracy: Per-angle scaling mode (v2.18.0+)
    per_angle_mode: str = "auto"
    constant_scaling_threshold: int = 3

    # Sharding
    sharding_strategy: str = "random"
    num_shards: int | str = "auto"
    max_points_per_shard: int | str = "auto"

    # Backend
    backend_name: str = "auto"
    enable_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints/cmc"

    # Sampling
    # NOTE: Defaults are intentionally conservative for laminar_flow
    # workloads to avoid per-shard timeouts on large pooled datasets
    # (millions of points across a handful of angles).
    #
    # Effective work per shard scales roughly as:
    #   O(num_chains * (num_warmup + num_samples) * max_points_per_shard)
    #
    # Values here are chosen to keep typical laminar_flow CMC shards
    # well under the 2 hour timeout on modest CPU nodes while still
    # providing usable posteriors and R-hat diagnostics.
    num_warmup: int = 500
    num_samples: int = 1500
    num_chains: int = 4  # Increased from 2 for better R-hat convergence diagnostics
    target_accept_prob: float = 0.85

    # Validation thresholds
    max_r_hat: float = 1.1
    min_ess: float = 100.0
    max_divergence_rate: float = 0.10  # Filter shards with >10% divergence rate

    # Combination
    combination_method: str = "consensus_mc"  # Correct CMC method (v2.4.3+)
    min_success_rate: float = 0.90
    run_id: str | None = None

    # Timeout (Jan 2026: reduced from 2h to 1h to fail faster on problematic shards)
    per_shard_timeout: int = 3600  # 1 hour per shard in seconds
    heartbeat_timeout: int = 600  # 10 minutes - terminate unresponsive workers

    # Warning thresholds
    min_success_rate_warning: float = 0.80  # Warn if success rate below this

    # Warm-start requirements (Jan 2026)
    require_nlsq_warmstart: bool = False  # Require NLSQ warm-start for laminar_flow

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
        backend_config = config_dict.get("backend_config", {})
        per_shard = config_dict.get("per_shard_mcmc", {})
        validation = config_dict.get("validation", {})
        combination = config_dict.get("combination", {})

        # Handle multiple schema variants for backend configuration:
        # 1. New schema: backend_config.name (preferred)
        # 2. Old schema: backend.name (dict)
        # 3. Legacy: backend as string (computational backend, separate from parallel backend)
        if backend_config and backend_config.get("name"):
            # New schema: use backend_config section
            backend_name = backend_config.get("name", "auto")
            enable_checkpoints = backend_config.get("enable_checkpoints", True)
            checkpoint_dir = backend_config.get("checkpoint_dir", "./checkpoints/cmc")
        elif isinstance(backend, str):
            # Legacy: backend is computational backend string, check backend_config
            backend_name = backend_config.get("name", "auto")
            enable_checkpoints = backend_config.get("enable_checkpoints", True)
            checkpoint_dir = backend_config.get("checkpoint_dir", "./checkpoints/cmc")
        elif isinstance(backend, dict) and backend.get("name"):
            # Old schema: backend is dict with name
            backend_name = backend.get("name", "auto")
            enable_checkpoints = backend.get("enable_checkpoints", True)
            checkpoint_dir = backend.get("checkpoint_dir", "./checkpoints/cmc")
        else:
            # Default
            backend_name = "auto"
            enable_checkpoints = True
            checkpoint_dir = "./checkpoints/cmc"

        # Backward compatibility: map legacy "jax" backend name to multiprocessing
        # NOTE: Map to multiprocessing, NOT pjit, because pjit backend is sequential
        # (it processes shards one at a time in a for loop, not in parallel)
        if backend_name == "jax":
            logger.warning(
                "CMC backend 'jax' is deprecated; mapping to 'multiprocessing' for parallel execution. "
                "Set backend_config.name to 'multiprocessing' or 'auto' instead."
            )
            backend_name = "multiprocessing"

        # Normalize possibly stringified ints
        num_shards_val = sharding.get("num_shards", "auto")
        if isinstance(num_shards_val, str) and num_shards_val.isdigit():
            num_shards_val = int(num_shards_val)

        max_points_val = sharding.get("max_points_per_shard", "auto")
        if isinstance(max_points_val, str) and max_points_val.isdigit():
            max_points_val = int(max_points_val)

        config = cls(
            # Enable settings
            enable=config_dict.get("enable", "auto"),
            min_points_for_cmc=config_dict.get("min_points_for_cmc", 500000),
            # Anti-degeneracy: Per-angle scaling mode (v2.18.0+)
            per_angle_mode=config_dict.get("per_angle_mode", "auto"),
            constant_scaling_threshold=config_dict.get("constant_scaling_threshold", 3),
            # Sharding
            sharding_strategy=sharding.get("strategy", "stratified"),
            num_shards=num_shards_val,
            max_points_per_shard=max_points_val,
            # Backend
            backend_name=backend_name,
            enable_checkpoints=enable_checkpoints,
            checkpoint_dir=checkpoint_dir,
            # Sampling
            num_warmup=per_shard.get("num_warmup", 500),
            num_samples=per_shard.get("num_samples", 1500),
            num_chains=per_shard.get("num_chains", 2),
            target_accept_prob=per_shard.get("target_accept_prob", 0.85),
            # Validation
            max_r_hat=validation.get("max_per_shard_rhat", 1.1),
            min_ess=validation.get("min_per_shard_ess", 100.0),
            max_divergence_rate=validation.get("max_divergence_rate", 0.10),
            # Combination
            combination_method=combination.get("method", "consensus_mc"),
            min_success_rate=combination.get("min_success_rate", 0.90),
            run_id=config_dict.get("run_id"),
            # Timeout
            per_shard_timeout=config_dict.get("per_shard_timeout", 3600),
            heartbeat_timeout=config_dict.get("heartbeat_timeout", 600),
            # Warning thresholds
            min_success_rate_warning=combination.get("min_success_rate_warning", 0.80),
            # Warm-start requirements
            require_nlsq_warmstart=validation.get("require_nlsq_warmstart", False),
        )

        # Validate and log any issues
        errors = config.validate()
        if errors:
            for error in errors:
                logger.warning(f"CMC config validation: {error}")

        # Warn about config precedence (CLI overwrites per_shard_mcmc with base mcmc)
        if per_shard:
            logger.debug(
                "Note: CLI applies base mcmc settings to per_shard_mcmc. "
                "If using CLI, ensure base mcmc and per_shard_mcmc are aligned."
            )

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

        # Validate per_angle_mode (v2.18.0+)
        valid_per_angle_modes = ["auto", "constant", "constant_averaged", "individual"]
        if self.per_angle_mode not in valid_per_angle_modes:
            errors.append(
                f"per_angle_mode must be one of {valid_per_angle_modes}, "
                f"got: {self.per_angle_mode}"
            )

        # Validate constant_scaling_threshold
        if (
            not isinstance(self.constant_scaling_threshold, int)
            or self.constant_scaling_threshold < 1
        ):
            errors.append(
                f"constant_scaling_threshold must be positive int, "
                f"got: {self.constant_scaling_threshold}"
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

        # Warn about small max_points_per_shard (creates excessive shards)
        if (
            isinstance(self.max_points_per_shard, int)
            and self.max_points_per_shard < 10000
        ):
            logger.warning(
                f"max_points_per_shard={self.max_points_per_shard:,} is very small. "
                "This will create many shards with high overhead. Recommended: 50000-100000."
            )

        # Validate backend_name (allow legacy 'jax' but normalize earlier)
        valid_backends = ["auto", "multiprocessing", "pjit", "pbs", "slurm", "jax"]
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
        if not 0.0 <= self.max_divergence_rate <= 1.0:
            errors.append(
                f"max_divergence_rate must be in [0, 1], got: {self.max_divergence_rate}"
            )

        # Validate combination settings
        valid_methods = ["consensus_mc", "weighted_gaussian", "simple_average", "auto"]
        if self.combination_method not in valid_methods:
            errors.append(
                f"combination_method must be one of {valid_methods}, got: {self.combination_method}"
            )
        if not 0.0 <= self.min_success_rate <= 1.0:
            errors.append(
                f"min_success_rate must be in [0, 1], got: {self.min_success_rate}"
            )

        # Validate timeout settings
        if not isinstance(self.heartbeat_timeout, int) or self.heartbeat_timeout < 60:
            errors.append(
                f"heartbeat_timeout must be int >= 60 seconds, got: {self.heartbeat_timeout}"
            )

        # Validate warning threshold
        if not 0.0 <= self.min_success_rate_warning <= 1.0:
            errors.append(
                f"min_success_rate_warning must be in [0, 1], got: {self.min_success_rate_warning}"
            )
        if self.min_success_rate_warning > self.min_success_rate:
            logger.warning(
                f"min_success_rate_warning ({self.min_success_rate_warning}) > "
                f"min_success_rate ({self.min_success_rate}); warning will never trigger"
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

    def should_enable_cmc(
        self, n_points: int, analysis_mode: str | None = None
    ) -> bool:
        """Determine if CMC should be enabled for given data size and mode.

        Parameters
        ----------
        n_points : int
            Number of data points.
        analysis_mode : str | None
            Analysis mode ("static" or "laminar_flow"). If provided, uses
            mode-specific thresholds when min_points_for_cmc is at default.

        Returns
        -------
        bool
            True if CMC should be enabled.

        Notes
        -----
        Mode-specific default thresholds (when min_points_for_cmc=500000):
        - laminar_flow: 100,000 (7-param model benefits from earlier sharding)
        - static: 500,000 (3-param model handles larger single-shard runs)

        If min_points_for_cmc is explicitly set to non-default value, that
        value is used regardless of analysis_mode.
        """
        if self.enable is True:
            return True
        if self.enable is False:
            return False

        # "auto" mode - apply mode-specific thresholds
        threshold = self.min_points_for_cmc

        # Apply mode-specific threshold if using default value
        if self.min_points_for_cmc == 500000 and analysis_mode == "laminar_flow":
            threshold = 100000  # Lower threshold for complex 7-param model

        return n_points >= threshold

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

    def get_effective_per_angle_mode(self, n_phi: int) -> str:
        """Determine effective per-angle mode based on configuration and data.

        Parameters
        ----------
        n_phi : int
            Number of phi angles in the dataset.

        Returns
        -------
        str
            Effective mode: "auto", "constant", or "individual".

        Notes
        -----
        Mode semantics (same as NLSQ):

        - auto: Sample single averaged contrast/offset (10 params for laminar_flow).
          Only activated when n_phi >= threshold (many angles).
        - constant: Use FIXED per-angle values from quantile estimation (8 params).
        - individual: Sample per-angle contrast/offset (n_phi*2 + 7 + 1 params).
        """
        if self.per_angle_mode == "auto":
            if n_phi >= self.constant_scaling_threshold:
                # Return "auto" - this uses the xpcs_model_averaged which samples
                # single averaged contrast/offset (10 params for laminar_flow)
                logger.info(
                    f"CMC anti-degeneracy: Using 'auto' mode (sampled averaged scaling) "
                    f"(n_phi={n_phi} >= threshold={self.constant_scaling_threshold})"
                )
                return "auto"
            else:
                # Few angles - use individual per-angle sampling
                logger.info(
                    f"CMC anti-degeneracy: Auto-selected 'individual' mode "
                    f"(n_phi={n_phi} < threshold={self.constant_scaling_threshold})"
                )
                return "individual"
        else:
            # Explicit mode (constant or individual)
            return self.per_angle_mode

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
            "per_angle_mode": self.per_angle_mode,
            "constant_scaling_threshold": self.constant_scaling_threshold,
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
                "max_divergence_rate": self.max_divergence_rate,
                "require_nlsq_warmstart": self.require_nlsq_warmstart,
            },
            "combination": {
                "method": self.combination_method,
                "min_success_rate": self.min_success_rate,
                "min_success_rate_warning": self.min_success_rate_warning,
            },
            "per_shard_timeout": self.per_shard_timeout,
            "heartbeat_timeout": self.heartbeat_timeout,
        }
