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
        num_chains: 4
      cmc:
        per_shard_mcmc:
          num_warmup: 500
          num_samples: 1500
          num_chains: 4
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
        Maximum points per shard. "auto" calculates optimally based on
        dataset size, analysis mode, and angle count (see
        ``_resolve_max_points_per_shard``).
        Default: "auto". Typical auto values: 5–20K for laminar_flow,
        10–20K for static (scales with dataset size).
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
        n_phi threshold for auto mode's per-angle strategy.
        When n_phi >= threshold, auto mode samples averaged contrast/offset
        (single value broadcast to all angles). When n_phi < threshold,
        auto mode falls back to individual per-angle sampling. Default: 3.
    """

    # Enable settings
    enable: bool | str = "auto"
    min_points_for_cmc: int = 100000

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
    #
    # DEPRECATION NOTE: Do not use config.num_warmup or config.num_samples
    # directly in sampling hot paths. Use SamplingPlan.from_config() instead,
    # which applies adaptive scaling. Direct access is only appropriate for:
    #   - Logging configured (pre-adaptation) values
    #   - Timeout estimation (safe upper bound)
    #   - Config serialization/validation
    num_warmup: int = 500
    num_samples: int = 1500
    num_chains: int = 4  # Increased from 2 for better R-hat convergence diagnostics
    target_accept_prob: float = 0.85

    # Adaptive sampling (Feb 2026): Scale warmup/samples based on shard size
    # Small datasets benefit from fewer samples to reduce NUTS overhead while
    # maintaining statistical validity. Profiling showed 1310s for 50 points
    # with default settings - adaptive scaling reduces this by 60-80%.
    adaptive_sampling: bool = True  # Enable adaptive sample count based on shard size
    max_tree_depth: int = 10  # NUTS tree depth (max 2^depth leapfrog steps per sample)
    min_warmup: int = 100  # Minimum warmup even for small datasets
    min_samples: int = 200  # Minimum samples even for small datasets

    # JAX profiling (Feb 2026): Capture XLA-level performance data
    # py-spy can only profile Python code; XLA runs native code invisible to py-spy.
    # Enable this to trace XLA operations and export to TensorBoard-compatible format.
    enable_jax_profiling: bool = False  # Enable jax.profiler tracing
    jax_profile_dir: str = "./profiles/jax"  # Directory for JAX profile output

    # Validation thresholds
    max_r_hat: float = 1.1
    min_ess: float = 400.0
    max_divergence_rate: float = 0.10  # Filter shards with >10% divergence rate

    # Combination
    combination_method: str = (
        "robust_consensus_mc"  # Robust CMC with MAD outlier filtering (v2.22.2)
    )
    min_success_rate: float = 0.90
    run_id: str | None = None

    # Timeout (Jan 2026: reduced from 2h to 1h to fail faster on problematic shards)
    per_shard_timeout: int = 3600  # 1 hour per shard in seconds
    heartbeat_timeout: int = 600  # 10 minutes - terminate unresponsive workers

    # Warning thresholds
    min_success_rate_warning: float = 0.80  # Warn if success rate below this

    # Warm-start requirements (Jan 2026)
    require_nlsq_warmstart: bool = False  # Require NLSQ warm-start for laminar_flow

    # NLSQ-informed priors (Feb 2026): Use NLSQ estimates to build tighter priors
    use_nlsq_informed_priors: bool = True  # Build TruncatedNormal priors from NLSQ
    nlsq_prior_width_factor: float = 2.0  # Width = NLSQ_std * factor (~95.4% coverage)

    # Prior tempering (Feb 2026): Scale priors by 1/K per shard (Scott et al. 2016)
    # Without tempering, K shards each apply the full prior → combined posterior = prior^K × likelihood.
    # With tempering, each shard uses prior^(1/K) → combined posterior = prior × likelihood (correct).
    # For Normal(μ,σ): prior^(1/K) ∝ Normal(μ, σ√K), i.e., widen std by √num_shards.
    prior_tempering: bool = True  # Enable prior tempering for multi-shard CMC

    # Heterogeneity detection (Jan 2026 v2)
    # Abort early if shard posteriors are too heterogeneous (high CV)
    max_parameter_cv: float = 1.0  # Abort if any parameter has CV > 1.0 across shards
    heterogeneity_abort: bool = True  # Enable heterogeneity abort (fail fast)
    min_points_per_shard: int = 10000  # Enforced minimum for laminar_flow
    min_points_per_param: int = 1500  # Minimum points per parameter per shard

    # Reparameterization (Jan 2026 v3)
    # Transform to break D0/D_offset degeneracy
    reparameterization_d_total: bool = True  # Sample D_total = D0 + D_offset
    reparameterization_log_gamma: bool = True  # Sample log(gamma_dot_t0)
    bimodal_min_weight: float = 0.2  # Minimum weight for GMM bimodal detection
    bimodal_min_separation: float = 0.5  # Minimum relative separation for bimodal

    # Reproducibility
    seed: int = 42  # Base seed for PRNG key generation

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
        # P2-4: Warn about unrecognized top-level keys that may be typos.
        _known_keys = {
            "sharding",
            "backend",
            "backend_config",
            "per_shard_mcmc",
            "validation",
            "combination",
            "reparameterization",
            "enable",
            "min_points_for_cmc",
            "per_angle_mode",
            "constant_scaling_threshold",
            "run_id",
            "per_shard_timeout",
            "heartbeat_timeout",
            "prior_tempering",
            "seed",
        }
        unknown_keys = set(config_dict.keys()) - _known_keys
        if unknown_keys:
            logger.warning(
                f"CMC config contains unrecognized keys (possible typos): "
                f"{sorted(unknown_keys)}"
            )

        # Extract nested sections
        sharding = config_dict.get("sharding", {})
        backend = config_dict.get("backend", {})
        backend_config = config_dict.get("backend_config", {})
        per_shard = config_dict.get("per_shard_mcmc", {})
        validation = config_dict.get("validation", {})
        combination = config_dict.get("combination", {})
        reparameterization = config_dict.get("reparameterization", {})

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

        # T3-6: Normalize possibly stringified numbers (handles "10" and "10.0")
        num_shards_val = sharding.get("num_shards", "auto")
        if isinstance(num_shards_val, str) and num_shards_val != "auto":
            try:
                num_shards_val = int(float(num_shards_val))
            except ValueError:
                pass  # Leave as string; validate() will catch it

        max_points_val = sharding.get("max_points_per_shard", "auto")
        if isinstance(max_points_val, str) and max_points_val != "auto":
            try:
                max_points_val = int(float(max_points_val))
            except ValueError:
                pass  # Leave as string; validate() will catch it

        config = cls(
            # Enable settings
            enable=config_dict.get("enable", "auto"),
            min_points_for_cmc=config_dict.get("min_points_for_cmc", 100000),
            # Anti-degeneracy: Per-angle scaling mode (v2.18.0+)
            per_angle_mode=config_dict.get("per_angle_mode", "auto"),
            constant_scaling_threshold=config_dict.get("constant_scaling_threshold", 3),
            # Sharding
            sharding_strategy=sharding.get("strategy", "random"),
            num_shards=num_shards_val,
            max_points_per_shard=max_points_val,
            # Backend
            backend_name=backend_name,
            enable_checkpoints=enable_checkpoints,
            checkpoint_dir=checkpoint_dir,
            # Sampling
            num_warmup=per_shard.get("num_warmup", 500),
            num_samples=per_shard.get("num_samples", 1500),
            num_chains=per_shard.get("num_chains", 4),
            target_accept_prob=per_shard.get("target_accept_prob", 0.85),
            # Adaptive sampling (Feb 2026)
            adaptive_sampling=per_shard.get("adaptive_sampling", True),
            max_tree_depth=per_shard.get("max_tree_depth", 10),
            min_warmup=per_shard.get("min_warmup", 100),
            min_samples=per_shard.get("min_samples", 200),
            # JAX profiling (Feb 2026)
            enable_jax_profiling=per_shard.get("enable_jax_profiling", False),
            jax_profile_dir=per_shard.get("jax_profile_dir", "./profiles/jax"),
            # Validation
            max_r_hat=validation.get("max_per_shard_rhat", 1.1),
            min_ess=validation.get(
                "min_ess", validation.get("min_per_shard_ess", 400.0)
            ),
            max_divergence_rate=validation.get("max_divergence_rate", 0.10),
            # Combination
            combination_method=combination.get("method", "robust_consensus_mc"),
            min_success_rate=combination.get("min_success_rate", 0.90),
            run_id=config_dict.get("run_id"),
            # Timeout
            per_shard_timeout=config_dict.get("per_shard_timeout", 3600),
            heartbeat_timeout=config_dict.get("heartbeat_timeout", 600),
            # Warning thresholds
            min_success_rate_warning=combination.get("min_success_rate_warning", 0.80),
            # Warm-start requirements
            require_nlsq_warmstart=validation.get("require_nlsq_warmstart", False),
            # NLSQ-informed priors (Feb 2026)
            use_nlsq_informed_priors=validation.get("use_nlsq_informed_priors", True),
            nlsq_prior_width_factor=validation.get("nlsq_prior_width_factor", 2.0),
            # Prior tempering (Feb 2026)
            prior_tempering=config_dict.get("prior_tempering", True),
            # Heterogeneity detection (Jan 2026 v2)
            max_parameter_cv=validation.get("max_parameter_cv", 1.0),
            heterogeneity_abort=validation.get("heterogeneity_abort", True),
            min_points_per_shard=sharding.get("min_points_per_shard", 10000),
            min_points_per_param=sharding.get("min_points_per_param", 1500),
            # Reparameterization (Jan 2026 v3)
            reparameterization_d_total=reparameterization.get("enable_d_total", True),
            reparameterization_log_gamma=reparameterization.get(
                "enable_log_gamma", True
            ),
            bimodal_min_weight=reparameterization.get("bimodal_min_weight", 0.2),
            bimodal_min_separation=reparameterization.get(
                "bimodal_min_separation", 0.5
            ),
            # Reproducibility
            seed=config_dict.get("seed", 42),
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
            and self.max_points_per_shard < 3000
        ):
            logger.warning(
                f"max_points_per_shard={self.max_points_per_shard:,} is very small. "
                "This will create many shards with high overhead. "
                "Recommended: 3000-10000 for laminar_flow, 50000-100000 for static."
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

        # Validate adaptive sampling settings (Feb 2026)
        if (
            not isinstance(self.max_tree_depth, int)
            or not 1 <= self.max_tree_depth <= 15
        ):
            errors.append(
                f"max_tree_depth must be int in [1, 15], got: {self.max_tree_depth}"
            )
        if not isinstance(self.min_warmup, int) or self.min_warmup < 10:
            errors.append(f"min_warmup must be int >= 10, got: {self.min_warmup}")
        if not isinstance(self.min_samples, int) or self.min_samples < 50:
            errors.append(f"min_samples must be int >= 50, got: {self.min_samples}")

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
        valid_methods = [
            "consensus_mc",
            "robust_consensus_mc",
            "weighted_gaussian",
            "simple_average",
            "auto",
        ]
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

        # Validate heterogeneity detection settings (Jan 2026 v2)
        if (
            not isinstance(self.max_parameter_cv, (int, float))
            or self.max_parameter_cv <= 0
        ):
            errors.append(
                f"max_parameter_cv must be positive number, got: {self.max_parameter_cv}"
            )
        if not isinstance(
            self.heterogeneity_abort, bool
        ):  # runtime check for YAML input
            errors.append(  # type: ignore[unreachable]
                f"heterogeneity_abort must be bool, got: {self.heterogeneity_abort}"
            )
        if (
            not isinstance(self.min_points_per_shard, int)
            or self.min_points_per_shard < 1000
        ):
            errors.append(
                f"min_points_per_shard must be int >= 1000, got: {self.min_points_per_shard}"
            )

        # Validate bimodal detection thresholds (Jan 2026 v3)
        if not (0.0 < self.bimodal_min_weight <= 0.5):
            errors.append(
                f"bimodal_min_weight must be in (0, 0.5], got: {self.bimodal_min_weight}"
            )
        if not (0.0 < self.bimodal_min_separation <= 2.0):
            errors.append(
                f"bimodal_min_separation must be in (0, 2.0], got: {self.bimodal_min_separation}"
            )

        # Validate seed (reproducibility)
        if not isinstance(self.seed, int) or self.seed < 0:
            errors.append(f"seed must be a non-negative integer, got: {self.seed}")

        # Validate NLSQ-informed priors (Feb 2026)
        if not (1.0 <= self.nlsq_prior_width_factor <= 10.0):
            errors.append(
                f"nlsq_prior_width_factor must be in [1.0, 10.0], got: {self.nlsq_prior_width_factor}"
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
        """Determine if CMC should be enabled for given data size.

        Parameters
        ----------
        n_points : int
            Number of data points.
        analysis_mode : str | None
            Deprecated — ignored. Kept for backward compatibility.

        Returns
        -------
        bool
            True if CMC should be enabled.

        Notes
        -----
        Threshold is ``min_points_for_cmc`` (default 100,000) for all modes.
        """
        if self.enable is True:
            return True
        if self.enable is False:
            return False

        # "auto" mode
        return n_points >= self.min_points_for_cmc

    def get_num_shards(self, n_points: int, n_phi: int, n_params: int = 7) -> int:
        """Calculate number of shards with param-aware sizing.

        Parameters
        ----------
        n_points : int
            Total number of data points.
        n_phi : int
            Number of phi angles.
        n_params : int
            Number of model parameters (default: 7 for static).

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
            base_max = self.max_points_per_shard
        else:
            # Default: ~100k points per shard
            base_max = 100000

        # Param-aware adjustment: scale up for n_params > 7
        param_factor = max(1.0, n_params / 7.0)
        min_required = int(self.min_points_per_param * n_params)

        adjusted_max = max(
            int(base_max * param_factor),
            min_required,
        )

        if param_factor > 1.0:
            logger.debug(
                f"Param-aware shard sizing: {n_params} params detected. "
                f"Adjusted max_points_per_shard: {base_max:,} -> {adjusted_max:,} "
                f"(factor={param_factor:.2f})"
            )

        return max(1, n_points // adjusted_max)

    def get_adaptive_sample_counts(
        self, shard_size: int, n_params: int = 7
    ) -> tuple[int, int]:
        """Calculate adaptive warmup/samples based on shard size.

        Small datasets benefit from fewer NUTS samples because:
        1. JIT compilation overhead is amortized over fewer samples
        2. Step size adaptation converges faster with simple likelihoods
        3. Mass matrix estimation requires fewer warmup iterations

        Profiling showed 1310s for 50 points with 500 warmup + 1500 samples.
        Adaptive scaling reduces this by 60-80% while maintaining statistical
        validity (ESS targets are reduced proportionally).

        Parameters
        ----------
        shard_size : int
            Number of data points in the shard.
        n_params : int
            Number of model parameters (affects minimum samples).

        Returns
        -------
        tuple[int, int]
            (num_warmup, num_samples) adjusted for shard size.
        """
        if not self.adaptive_sampling:
            return self.num_warmup, self.num_samples

        # Reference point: 10k points → full samples
        # Scale down for smaller datasets
        reference_size = 10000
        scale_factor = min(1.0, shard_size / reference_size)

        # Compute scaled counts
        scaled_warmup = int(self.num_warmup * scale_factor)
        scaled_samples = int(self.num_samples * scale_factor)

        # Ensure minimum viable sampling (ESS requires ~50 samples per param).
        # P2-B: Cap at configured defaults — adaptive scaling should only reduce,
        # never exceed, the user's configured num_warmup/num_samples.
        min_samples_for_params = min(
            max(self.min_samples, 50 * n_params), self.num_samples
        )
        min_warmup_for_params = min(
            max(self.min_warmup, 20 * n_params), self.num_warmup
        )

        # Apply bounds
        final_warmup = max(min_warmup_for_params, scaled_warmup)
        final_samples = max(min_samples_for_params, scaled_samples)

        # Log if different from defaults
        if final_warmup != self.num_warmup or final_samples != self.num_samples:
            logger.debug(
                f"Adaptive sampling: {shard_size:,} points, {n_params} params -> "
                f"warmup={final_warmup} (was {self.num_warmup}), "
                f"samples={final_samples} (was {self.num_samples})"
            )

        return final_warmup, final_samples

    def get_effective_per_angle_mode(
        self,
        n_phi: int,
        nlsq_per_angle_mode: str | None = None,
        has_nlsq_warmstart: bool = False,
    ) -> str:
        """Determine effective per-angle mode based on configuration and data.

        Parameters
        ----------
        n_phi : int
            Number of phi angles in the dataset.
        nlsq_per_angle_mode : str | None
            Optional per-angle mode from NLSQ result. When provided (from warm-start),
            CMC will use this mode to ensure parameterization parity with NLSQ.
            This prevents CMC vs NLSQ divergence from different model structures.
        has_nlsq_warmstart : bool
            Whether an NLSQ warm-start result is available. When True and both
            CMC and NLSQ use "auto" mode, upgrades to "constant_averaged" for
            fewer sampled parameters and better stability.

        Returns
        -------
        str
            Effective mode: "auto", "constant", "constant_averaged", or "individual".

        Notes
        -----
        Mode semantics (same as NLSQ):

        - auto: Sample single averaged contrast/offset (10 params for laminar_flow).
          Only activated when n_phi >= threshold (many angles).
        - constant: Use FIXED per-angle values from quantile estimation (8 params).
        - constant_averaged: Use FIXED averaged scaling for NLSQ parity.
        - individual: Sample per-angle contrast/offset (n_phi*2 + 7 + 1 params).

        Priority: nlsq_per_angle_mode > explicit config > auto-selection

        When NLSQ warm-start is present and both sides use "auto", upgrades to
        "constant_averaged" to fix scaling values and reduce parameter count.
        This prevents contrast/offset sampling from absorbing physical parameter
        signal, which was the root cause of heterogeneous shard posteriors.
        """
        # Jan 2026 v2: When NLSQ warm-start provides per-angle mode, match it
        # This ensures CMC and NLSQ use identical parameterizations
        if nlsq_per_angle_mode is not None:
            # Feb 2026: When NLSQ warm-start present and both sides use "auto",
            # upgrade to constant_averaged for fewer params and better stability
            if (
                has_nlsq_warmstart
                and nlsq_per_angle_mode == "auto"
                and self.per_angle_mode == "auto"
            ):
                logger.info(
                    "CMC per-angle mode: auto -> constant_averaged "
                    "(NLSQ warm-start present, fixing scaling for stability)"
                )
                return "constant_averaged"

            logger.info(
                f"CMC per-angle mode: Using NLSQ warm-start mode '{nlsq_per_angle_mode}' "
                f"for parameterization parity"
            )
            return nlsq_per_angle_mode

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
            # Explicit mode (constant, constant_averaged, or individual)
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
                "min_points_per_shard": self.min_points_per_shard,
                "min_points_per_param": self.min_points_per_param,
            },
            "backend_config": {
                "name": self.backend_name,
                "enable_checkpoints": self.enable_checkpoints,
                "checkpoint_dir": self.checkpoint_dir,
            },
            "per_shard_mcmc": {
                "num_warmup": self.num_warmup,
                "num_samples": self.num_samples,
                "num_chains": self.num_chains,
                "target_accept_prob": self.target_accept_prob,
                "adaptive_sampling": self.adaptive_sampling,
                "max_tree_depth": self.max_tree_depth,
                "min_warmup": self.min_warmup,
                "min_samples": self.min_samples,
                "enable_jax_profiling": self.enable_jax_profiling,
                "jax_profile_dir": self.jax_profile_dir,
            },
            "validation": {
                "max_per_shard_rhat": self.max_r_hat,
                "min_per_shard_ess": self.min_ess,
                "max_divergence_rate": self.max_divergence_rate,
                "require_nlsq_warmstart": self.require_nlsq_warmstart,
                "use_nlsq_informed_priors": self.use_nlsq_informed_priors,
                "nlsq_prior_width_factor": self.nlsq_prior_width_factor,
                "max_parameter_cv": self.max_parameter_cv,
                "heterogeneity_abort": self.heterogeneity_abort,
            },
            "combination": {
                "method": self.combination_method,
                "min_success_rate": self.min_success_rate,
                "min_success_rate_warning": self.min_success_rate_warning,
            },
            "prior_tempering": self.prior_tempering,
            "per_shard_timeout": self.per_shard_timeout,
            "heartbeat_timeout": self.heartbeat_timeout,
            "reparameterization": {
                "enable_d_total": self.reparameterization_d_total,
                "enable_log_gamma": self.reparameterization_log_gamma,
                "bimodal_min_weight": self.bimodal_min_weight,
                "bimodal_min_separation": self.bimodal_min_separation,
            },
            "seed": self.seed,
        }
