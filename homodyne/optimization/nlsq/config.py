"""NLSQ configuration dataclass and validation.

This module provides the NLSQConfig dataclass for parsing and validating
NLSQ-specific configuration settings from the YAML config file.

Part of Phase 3 architecture refactoring to reduce wrapper.py complexity.

Config Consolidation (v2.14.0, FR-014):
- Single entry point: NLSQConfig.from_yaml() or NLSQConfig.from_dict()
- Safe type conversion utilities: safe_float, safe_int, safe_bool
- Full validation via validate() method
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Safe Type Conversion Utilities (T094-T096)
# Consolidated from config_utils.py
# =============================================================================


def safe_float(value: Any, default: float) -> float:
    """Convert value to float safely, returning default on failure.

    Parameters
    ----------
    value : Any
        Value to convert to float.
    default : float
        Default value to return if conversion fails.

    Returns
    -------
    float
        Converted float value or default.

    Examples
    --------
    >>> safe_float("3.14", 0.0)
    3.14
    >>> safe_float(None, 1.0)
    1.0
    >>> safe_float("invalid", 2.5)
    2.5
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to float, using default {default}")
        return default


def safe_int(value: Any, default: int) -> int:
    """Convert value to int safely, returning default on failure.

    Parameters
    ----------
    value : Any
        Value to convert to int.
    default : int
        Default value to return if conversion fails.

    Returns
    -------
    int
        Converted int value or default.

    Examples
    --------
    >>> safe_int("42", 0)
    42
    >>> safe_int(None, 10)
    10
    >>> safe_int("invalid", 5)
    5
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to int, using default {default}")
        return default


def safe_bool(value: Any, default: bool) -> bool:
    """Convert value to bool safely, returning default on failure.

    Handles string values like "true", "false", "1", "0".

    Parameters
    ----------
    value : Any
        Value to convert to bool.
    default : bool
        Default value to return if conversion fails.

    Returns
    -------
    bool
        Converted bool value or default.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "1", "yes", "on"):
            return True
        if lower in ("false", "0", "no", "off"):
            return False
    try:
        return bool(value)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert {value!r} to bool, using default {default}")
        return default


# Valid loss functions for NLSQ
VALID_LOSS_FUNCTIONS = {"linear", "soft_l1", "huber", "cauchy", "arctan"}


@dataclass
class HybridRecoveryConfig:
    """Configuration for hybrid streaming optimizer recovery strategy.

    T029: Implements 3-attempt recovery with progressively conservative settings.

    When the hybrid streaming optimizer fails, it retries with:
    - Reduced learning rate (0.5× per retry)
    - Increased regularization (2× per retry)
    - Smaller trust region (0.5× per retry)

    Attributes
    ----------
    max_retries : int
        Maximum retry attempts. Default: 3.
    lr_decay : float
        Learning rate multiplier per retry. Default: 0.5.
    lambda_growth : float
        Regularization multiplier per retry. Default: 2.0.
    trust_decay : float
        Trust region multiplier per retry. Default: 0.5.
    log_retries : bool
        Whether to log retry attempts. Default: True.
    """

    max_retries: int = 3
    lr_decay: float = 0.5
    lambda_growth: float = 2.0
    trust_decay: float = 0.5
    log_retries: bool = True

    def get_retry_settings(self, attempt: int) -> dict:
        """Get settings for a specific retry attempt.

        Parameters
        ----------
        attempt : int
            Retry attempt number (1-based).

        Returns
        -------
        dict
            Settings for this retry attempt.
        """
        return {
            "lr_multiplier": self.lr_decay**attempt,
            "lambda_multiplier": self.lambda_growth**attempt,
            "trust_multiplier": self.trust_decay**attempt,
        }


@dataclass
class NLSQConfig:
    """Configuration for NLSQ (Nonlinear Least Squares) optimization.

    This dataclass consolidates NLSQ settings that were previously scattered
    across wrapper.py, improving maintainability and testability.

    Attributes
    ----------
    loss : str
        Loss function for robust fitting. Options: "linear", "soft_l1",
        "huber", "cauchy", "arctan". Default: "soft_l1".
    trust_region_scale : float
        Scale factor for trust region. Default: 1.0.
    max_iterations : int
        Maximum number of optimization iterations. Default: 1000.
    ftol : float
        Function tolerance for convergence. Default: 1e-8.
    xtol : float
        Parameter tolerance for convergence. Default: 1e-8.
    gtol : float
        Gradient tolerance for convergence. Default: 1e-8.
    x_scale : str | list[float] | None
        Parameter scaling. "jac" for Jacobian-based, list for manual. Default: "jac".
    x_scale_map : dict[str, float] | None
        Per-parameter scaling overrides. Default: None.
    enable_diagnostics : bool
        Whether to compute diagnostics (Jacobian stats, etc.). Default: True.
    enable_streaming : bool
        Whether to enable streaming optimizer for large datasets. Default: True.
    streaming_chunk_size : int
        Points per chunk for streaming optimizer. Default: 50000.
    enable_stratified : bool
        Whether to enable stratified least squares. Default: True.
    target_chunk_size : int
        Target points per chunk for stratified optimization. Default: 100000.
    enable_recovery : bool
        Whether to enable automatic error recovery. Default: True.
    max_recovery_attempts : int
        Maximum recovery attempts per strategy. Default: 3.
    """

    # NLSQ Workflow Settings (DEPRECATED in v0.6.0)
    # Use homodyne's select_nlsq_strategy() from memory.py instead
    workflow: str = "auto"  # "auto", "standard", "chunked", "streaming"
    goal: str = "quality"  # "fast", "robust", "quality", "memory_efficient"

    # Loss function settings
    loss: str = "soft_l1"
    trust_region_scale: float = 1.0

    # Convergence settings
    max_iterations: int = 1000
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

    # Scaling settings
    x_scale: str | list[float] | None = "jac"
    x_scale_map: dict[str, float] | None = None

    # Diagnostics
    enable_diagnostics: bool = True

    # Streaming optimizer settings
    enable_streaming: bool = True
    streaming_chunk_size: int = 50000

    # Stratified optimization settings
    enable_stratified: bool = True
    target_chunk_size: int = 100000

    # Recovery settings
    enable_recovery: bool = True
    max_recovery_attempts: int = 3

    # Progress and logging settings (v2.7.0)
    # Controls progress bar display and logging verbosity during optimization
    enable_progress_bar: bool = True  # Show tqdm progress bar during fitting
    verbose: int = 1  # Verbosity level: 0=quiet, 1=normal, 2=detailed
    log_iteration_interval: int = 10  # Log every N iterations (for verbose >= 2)

    # Hybrid streaming optimizer settings (v2.6.0)
    # Fixes: 1) Shear-term weak gradients, 2) Slow convergence, 3) Crude covariance
    enable_hybrid_streaming: bool = True
    hybrid_normalize: bool = True
    hybrid_normalization_strategy: str = "auto"  # 'auto', 'bounds', 'p0', 'none'
    hybrid_warmup_iterations: int = 200
    hybrid_max_warmup_iterations: int = 500
    hybrid_warmup_learning_rate: float = 0.001
    hybrid_gauss_newton_max_iterations: int = 100
    hybrid_gauss_newton_tol: float = 1e-8
    hybrid_chunk_size: int = 10000
    hybrid_trust_region_initial: float = 1.0
    hybrid_regularization_factor: float = 1e-10
    hybrid_enable_checkpoints: bool = True
    hybrid_checkpoint_frequency: int = 100
    hybrid_validate_numerics: bool = True

    # 4-Layer Defense Strategy for L-BFGS Warmup (v2.8.0 / NLSQ 0.3.6)
    # Prevents divergence when starting from good initial parameters
    #
    # Layer 1: Warm Start Detection - skip warmup if already at good solution
    hybrid_enable_warm_start_detection: bool = True
    hybrid_warm_start_threshold: float = 0.01  # Skip if loss/variance < this
    #
    # Layer 2: Adaptive Learning Rate - scale LR based on initial loss quality
    hybrid_enable_adaptive_warmup_lr: bool = True
    hybrid_warmup_lr_refinement: float = (
        1e-6  # LR for good starts (relative_loss < 0.1)
    )
    hybrid_warmup_lr_careful: float = (
        1e-5  # LR for moderate starts (relative_loss < 1.0)
    )
    #
    # Layer 3: Cost-Increase Guard - abort if loss increases during warmup
    hybrid_enable_cost_guard: bool = True
    hybrid_cost_increase_tolerance: float = 0.05  # Abort if loss increases >5%
    #
    # Layer 4: Step Clipping - limit max parameter change per L-BFGS iteration
    hybrid_enable_step_clipping: bool = True
    hybrid_max_warmup_step_size: float = 0.1  # Max step in normalized units

    # Multi-start optimization settings (v2.6.0)
    # Enables exploration of parameter space via Latin Hypercube Sampling
    # NOTE: Subsampling is explicitly NOT supported per project requirements.
    # Numerical precision and reproducibility take priority over computational speed.
    enable_multi_start: bool = False  # Default OFF - user opt-in
    multi_start_n_starts: int = 10
    multi_start_seed: int = 42
    multi_start_sampling_strategy: str = (
        "latin_hypercube"  # 'latin_hypercube' or 'random'
    )
    multi_start_n_workers: int = 0  # 0 = auto (min of n_starts, cpu_count)
    multi_start_use_screening: bool = True
    multi_start_screen_keep_fraction: float = 0.5
    multi_start_refine_top_k: int = 3
    multi_start_refinement_ftol: float = 1e-12
    multi_start_degeneracy_threshold: float = 0.1

    # === Anti-Degeneracy Defense System (v2.9.0) ===
    # See: docs/specs/anti-degeneracy-defense-v2.9.0.md
    #
    # Layer 1: Fourier Reparameterization / Constant Scaling
    # Reduces structural degeneracy by expressing per-angle params as Fourier series
    # or using a single constant value shared across all angles
    per_angle_mode: str = "auto"  # "individual", "constant", "fourier", "auto"
    fourier_order: int = 2  # Number of Fourier harmonics (order=2 -> 5 coeffs)
    fourier_auto_threshold: int = 6  # Use Fourier when n_phi > threshold
    constant_scaling_threshold: int = (
        3  # Use constant when n_phi >= threshold (auto mode)
    )
    #
    # Layer 2: Hierarchical Optimization
    # Alternates between physical and per-angle params to break gradient cancellation
    enable_hierarchical: bool = True
    hierarchical_max_outer_iterations: int = 5
    hierarchical_outer_tolerance: float = 1e-6
    hierarchical_physical_max_iterations: int = 100
    hierarchical_per_angle_max_iterations: int = 50
    #
    # Layer 3: Adaptive Relative Regularization
    # CV-based regularization that scales properly with data
    regularization_mode: str = "relative"  # "absolute", "relative", "auto"
    group_variance_lambda: float = 1.0  # 100x stronger than v2.8 default of 0.01
    regularization_target_cv: float = 0.10  # 10% variation target
    regularization_target_contribution: float = 0.10  # 10% of MSE contribution
    regularization_max_cv: float = 0.20  # 20% max variation
    regularization_auto_tune_lambda: bool = True
    #
    # Layer 4: Gradient Collapse Detection
    # Runtime detection and response to gradient collapse
    enable_gradient_monitoring: bool = True
    gradient_ratio_threshold: float = 0.01  # |∇_physical|/|∇_per_angle| threshold
    gradient_consecutive_triggers: int = 5  # Must trigger N times consecutively
    gradient_collapse_response: str = (
        "hierarchical"  # "warn", "hierarchical", "reset", "abort"
    )

    # Computed fields
    _validation_errors: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> NLSQConfig:
        """Create NLSQConfig from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            NLSQ configuration dictionary from ConfigManager.

        Returns
        -------
        NLSQConfig
            Validated configuration object.
        """
        # Extract nested sections
        diagnostics = config_dict.get("diagnostics", {})
        streaming = config_dict.get("streaming", {})
        stratified = config_dict.get("stratified", {})
        recovery = config_dict.get("recovery", {})
        hybrid_streaming = config_dict.get("hybrid_streaming", {})
        multi_start = config_dict.get("multi_start", {})

        # Extract progress/logging settings
        progress = config_dict.get("progress", {})

        # Extract anti-degeneracy settings (v2.9.0)
        anti_degeneracy = config_dict.get("anti_degeneracy", {})
        hierarchical = anti_degeneracy.get("hierarchical", {})
        regularization = anti_degeneracy.get("regularization", {})
        gradient_monitoring = anti_degeneracy.get("gradient_monitoring", {})

        config = cls(
            # NLSQ Workflow Settings (v2.11.0+)
            workflow=config_dict.get("workflow", "auto"),
            goal=config_dict.get("goal", "quality"),
            # Loss function
            loss=config_dict.get("loss", "soft_l1"),
            trust_region_scale=float(config_dict.get("trust_region_scale", 1.0)),
            # Convergence
            max_iterations=config_dict.get("max_iterations", 1000),
            ftol=float(config_dict.get("tolerance", 1e-8)),
            xtol=float(config_dict.get("xtol", 1e-8)),
            gtol=float(config_dict.get("gtol", 1e-8)),
            # Scaling
            x_scale=config_dict.get("x_scale", "jac"),
            x_scale_map=config_dict.get("x_scale_map"),
            # Diagnostics
            enable_diagnostics=diagnostics.get("enable", True),
            # Streaming
            enable_streaming=streaming.get("enable", True),
            streaming_chunk_size=streaming.get("chunk_size", 50000),
            # Stratified
            enable_stratified=stratified.get("enable", True),
            target_chunk_size=stratified.get("target_chunk_size", 100000),
            # Recovery
            enable_recovery=recovery.get("enable", True),
            max_recovery_attempts=recovery.get("max_attempts", 3),
            # Progress and logging (v2.7.0)
            enable_progress_bar=progress.get("enable", True),
            verbose=progress.get("verbose", 1),
            log_iteration_interval=progress.get("log_interval", 10),
            # Hybrid streaming (v2.6.0)
            enable_hybrid_streaming=hybrid_streaming.get("enable", True),
            hybrid_normalize=hybrid_streaming.get("normalize", True),
            hybrid_normalization_strategy=hybrid_streaming.get(
                "normalization_strategy", "bounds"
            ),
            hybrid_warmup_iterations=hybrid_streaming.get("warmup_iterations", 100),
            hybrid_max_warmup_iterations=hybrid_streaming.get(
                "max_warmup_iterations", 500
            ),
            hybrid_warmup_learning_rate=float(
                hybrid_streaming.get("warmup_learning_rate", 0.001)
            ),
            hybrid_gauss_newton_max_iterations=hybrid_streaming.get(
                "gauss_newton_max_iterations", 50
            ),
            hybrid_gauss_newton_tol=float(
                hybrid_streaming.get("gauss_newton_tol", 1e-8)
            ),
            hybrid_chunk_size=hybrid_streaming.get("chunk_size", 50000),
            hybrid_trust_region_initial=float(
                hybrid_streaming.get("trust_region_initial", 1.0)
            ),
            hybrid_regularization_factor=float(
                hybrid_streaming.get("regularization_factor", 1e-10)
            ),
            hybrid_enable_checkpoints=hybrid_streaming.get("enable_checkpoints", True),
            hybrid_checkpoint_frequency=hybrid_streaming.get(
                "checkpoint_frequency", 100
            ),
            hybrid_validate_numerics=hybrid_streaming.get("validate_numerics", True),
            # 4-Layer Defense Strategy (v2.8.0 / NLSQ 0.3.6)
            # Layer 1: Warm Start Detection
            hybrid_enable_warm_start_detection=hybrid_streaming.get(
                "enable_warm_start_detection", True
            ),
            hybrid_warm_start_threshold=float(
                hybrid_streaming.get("warm_start_threshold", 0.01)
            ),
            # Layer 2: Adaptive Learning Rate
            hybrid_enable_adaptive_warmup_lr=hybrid_streaming.get(
                "enable_adaptive_warmup_lr", True
            ),
            hybrid_warmup_lr_refinement=float(
                hybrid_streaming.get("warmup_lr_refinement", 1e-6)
            ),
            hybrid_warmup_lr_careful=float(
                hybrid_streaming.get("warmup_lr_careful", 1e-5)
            ),
            # Layer 3: Cost-Increase Guard
            hybrid_enable_cost_guard=hybrid_streaming.get("enable_cost_guard", True),
            hybrid_cost_increase_tolerance=float(
                hybrid_streaming.get("cost_increase_tolerance", 0.05)
            ),
            # Layer 4: Step Clipping
            hybrid_enable_step_clipping=hybrid_streaming.get(
                "enable_step_clipping", True
            ),
            hybrid_max_warmup_step_size=float(
                hybrid_streaming.get("max_warmup_step_size", 0.1)
            ),
            # Multi-start (v2.6.0)
            # NOTE: No subsampling - numerical precision takes priority
            enable_multi_start=multi_start.get("enable", False),
            multi_start_n_starts=multi_start.get("n_starts", 10),
            multi_start_seed=multi_start.get("seed", 42),
            multi_start_sampling_strategy=multi_start.get(
                "sampling_strategy", "latin_hypercube"
            ),
            multi_start_n_workers=multi_start.get("n_workers", 0),
            multi_start_use_screening=multi_start.get("use_screening", True),
            multi_start_screen_keep_fraction=float(
                multi_start.get("screen_keep_fraction", 0.5)
            ),
            multi_start_refine_top_k=multi_start.get("refine_top_k", 3),
            multi_start_refinement_ftol=float(
                multi_start.get("refinement_ftol", 1e-12)
            ),
            multi_start_degeneracy_threshold=float(
                multi_start.get("degeneracy_threshold", 0.1)
            ),
            # Anti-Degeneracy Defense System (v2.9.0)
            # Layer 1: Fourier Reparameterization / Constant Scaling
            per_angle_mode=anti_degeneracy.get("per_angle_mode", "auto"),
            fourier_order=anti_degeneracy.get("fourier_order", 2),
            fourier_auto_threshold=anti_degeneracy.get("fourier_auto_threshold", 6),
            constant_scaling_threshold=anti_degeneracy.get(
                "constant_scaling_threshold", 3
            ),
            # Layer 2: Hierarchical Optimization
            enable_hierarchical=hierarchical.get("enable", True),
            hierarchical_max_outer_iterations=hierarchical.get(
                "max_outer_iterations", 5
            ),
            hierarchical_outer_tolerance=float(
                hierarchical.get("outer_tolerance", 1e-6)
            ),
            hierarchical_physical_max_iterations=hierarchical.get(
                "physical_max_iterations", 100
            ),
            hierarchical_per_angle_max_iterations=hierarchical.get(
                "per_angle_max_iterations", 50
            ),
            # Layer 3: Adaptive Relative Regularization
            regularization_mode=regularization.get("mode", "relative"),
            group_variance_lambda=float(regularization.get("lambda", 1.0)),
            regularization_target_cv=float(regularization.get("target_cv", 0.10)),
            regularization_target_contribution=float(
                regularization.get("target_contribution", 0.10)
            ),
            regularization_max_cv=float(regularization.get("max_cv", 0.20)),
            regularization_auto_tune_lambda=regularization.get(
                "auto_tune_lambda", True
            ),
            # Layer 4: Gradient Collapse Detection
            enable_gradient_monitoring=gradient_monitoring.get("enable", True),
            gradient_ratio_threshold=float(
                gradient_monitoring.get("ratio_threshold", 0.01)
            ),
            gradient_consecutive_triggers=gradient_monitoring.get(
                "consecutive_triggers", 5
            ),
            gradient_collapse_response=gradient_monitoring.get(
                "response", "hierarchical"
            ),
        )

        # Validate and log any issues
        errors = config.validate()
        if errors:
            for error in errors:
                logger.warning(f"NLSQ config validation: {error}")

        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> NLSQConfig:
        """Create NLSQConfig from YAML configuration file (T099).

        This is the recommended single entry point for loading NLSQ configuration.
        It reads the YAML file, extracts the optimization.nlsq section, and
        creates a validated NLSQConfig object.

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file.

        Returns
        -------
        NLSQConfig
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        ValueError
            If the YAML file is invalid or missing required sections.

        Examples
        --------
        >>> config = NLSQConfig.from_yaml("homodyne_config.yaml")
        >>> print(config.loss)
        soft_l1
        """
        import yaml
        from pathlib import Path

        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(path) as f:
            full_config = yaml.safe_load(f)

        if full_config is None:
            full_config = {}

        # Extract optimization.nlsq section
        optimization = full_config.get("optimization", {})
        nlsq_config = optimization.get("nlsq", {})

        if not nlsq_config:
            logger.warning(
                f"No optimization.nlsq section found in {yaml_path}, using defaults"
            )

        return cls.from_dict(nlsq_config)

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns
        -------
        list[str]
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # Validate loss function
        valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
        if self.loss not in valid_losses:
            errors.append(f"loss must be one of {valid_losses}, got: {self.loss}")

        # Validate trust_region_scale
        if self.trust_region_scale <= 0:
            errors.append(
                f"trust_region_scale must be positive, got: {self.trust_region_scale}"
            )

        # Validate convergence tolerances
        if self.ftol <= 0:
            errors.append(f"ftol must be positive, got: {self.ftol}")
        if self.xtol <= 0:
            errors.append(f"xtol must be positive, got: {self.xtol}")
        if self.gtol <= 0:
            errors.append(f"gtol must be positive, got: {self.gtol}")

        # Validate max_iterations
        if self.max_iterations <= 0:
            errors.append(
                f"max_iterations must be positive, got: {self.max_iterations}"
            )

        # Validate chunk sizes
        if self.streaming_chunk_size <= 0:
            errors.append(
                f"streaming_chunk_size must be positive, got: {self.streaming_chunk_size}"
            )
        if self.target_chunk_size <= 0:
            errors.append(
                f"target_chunk_size must be positive, got: {self.target_chunk_size}"
            )

        # Validate recovery attempts
        if self.max_recovery_attempts < 0:
            errors.append(
                f"max_recovery_attempts must be non-negative, got: {self.max_recovery_attempts}"
            )

        # Validate hybrid streaming settings
        valid_norm_strategies = ["auto", "bounds", "p0", "none"]
        if self.hybrid_normalization_strategy not in valid_norm_strategies:
            errors.append(
                f"hybrid_normalization_strategy must be one of {valid_norm_strategies}, "
                f"got: {self.hybrid_normalization_strategy}"
            )
        if self.hybrid_warmup_iterations <= 0:
            errors.append(
                f"hybrid_warmup_iterations must be positive, got: {self.hybrid_warmup_iterations}"
            )
        if self.hybrid_max_warmup_iterations <= 0:
            errors.append(
                f"hybrid_max_warmup_iterations must be positive, "
                f"got: {self.hybrid_max_warmup_iterations}"
            )
        if self.hybrid_warmup_learning_rate <= 0:
            errors.append(
                f"hybrid_warmup_learning_rate must be positive, "
                f"got: {self.hybrid_warmup_learning_rate}"
            )
        if self.hybrid_gauss_newton_max_iterations <= 0:
            errors.append(
                f"hybrid_gauss_newton_max_iterations must be positive, "
                f"got: {self.hybrid_gauss_newton_max_iterations}"
            )
        if self.hybrid_gauss_newton_tol <= 0:
            errors.append(
                f"hybrid_gauss_newton_tol must be positive, got: {self.hybrid_gauss_newton_tol}"
            )
        if self.hybrid_chunk_size <= 0:
            errors.append(
                f"hybrid_chunk_size must be positive, got: {self.hybrid_chunk_size}"
            )

        # Validate 4-Layer Defense parameters
        # Layer 1: Warm Start Detection
        if self.hybrid_warm_start_threshold <= 0:
            errors.append(
                f"hybrid_warm_start_threshold must be positive, "
                f"got: {self.hybrid_warm_start_threshold}"
            )
        # Layer 2: Adaptive Learning Rate
        if self.hybrid_warmup_lr_refinement <= 0:
            errors.append(
                f"hybrid_warmup_lr_refinement must be positive, "
                f"got: {self.hybrid_warmup_lr_refinement}"
            )
        if self.hybrid_warmup_lr_careful <= 0:
            errors.append(
                f"hybrid_warmup_lr_careful must be positive, "
                f"got: {self.hybrid_warmup_lr_careful}"
            )
        # Layer 3: Cost-Increase Guard
        if not 0 < self.hybrid_cost_increase_tolerance < 1:
            errors.append(
                f"hybrid_cost_increase_tolerance must be in (0, 1), "
                f"got: {self.hybrid_cost_increase_tolerance}"
            )
        # Layer 4: Step Clipping
        if self.hybrid_max_warmup_step_size <= 0:
            errors.append(
                f"hybrid_max_warmup_step_size must be positive, "
                f"got: {self.hybrid_max_warmup_step_size}"
            )

        # Validate multi-start settings
        valid_sampling_strategies = ["latin_hypercube", "random"]
        if self.multi_start_sampling_strategy not in valid_sampling_strategies:
            errors.append(
                f"multi_start_sampling_strategy must be one of {valid_sampling_strategies}, "
                f"got: {self.multi_start_sampling_strategy}"
            )
        if self.multi_start_n_starts <= 0:
            errors.append(
                f"multi_start_n_starts must be positive, got: {self.multi_start_n_starts}"
            )
        if self.multi_start_n_workers < 0:
            errors.append(
                f"multi_start_n_workers must be non-negative, got: {self.multi_start_n_workers}"
            )
        if not 0 < self.multi_start_screen_keep_fraction <= 1:
            errors.append(
                f"multi_start_screen_keep_fraction must be in (0, 1], "
                f"got: {self.multi_start_screen_keep_fraction}"
            )
        if self.multi_start_refine_top_k < 0:
            errors.append(
                f"multi_start_refine_top_k must be non-negative, "
                f"got: {self.multi_start_refine_top_k}"
            )
        if self.multi_start_refinement_ftol <= 0:
            errors.append(
                f"multi_start_refinement_ftol must be positive, "
                f"got: {self.multi_start_refinement_ftol}"
            )
        if not 0 < self.multi_start_degeneracy_threshold < 1:
            errors.append(
                f"multi_start_degeneracy_threshold must be in (0, 1), "
                f"got: {self.multi_start_degeneracy_threshold}"
            )

        # Validate Anti-Degeneracy Defense System settings (v2.9.0)
        # Layer 1: Fourier Reparameterization / Constant Scaling
        valid_per_angle_modes = ["individual", "constant", "fourier", "auto"]
        if self.per_angle_mode not in valid_per_angle_modes:
            errors.append(
                f"per_angle_mode must be one of {valid_per_angle_modes}, "
                f"got: {self.per_angle_mode}"
            )
        if self.fourier_order < 1:
            errors.append(f"fourier_order must be >= 1, got: {self.fourier_order}")
        if self.fourier_auto_threshold < 1:
            errors.append(
                f"fourier_auto_threshold must be >= 1, got: {self.fourier_auto_threshold}"
            )
        if self.constant_scaling_threshold < 1:
            errors.append(
                f"constant_scaling_threshold must be >= 1, "
                f"got: {self.constant_scaling_threshold}"
            )

        # Layer 2: Hierarchical Optimization
        if self.hierarchical_max_outer_iterations <= 0:
            errors.append(
                f"hierarchical_max_outer_iterations must be positive, "
                f"got: {self.hierarchical_max_outer_iterations}"
            )
        if self.hierarchical_outer_tolerance <= 0:
            errors.append(
                f"hierarchical_outer_tolerance must be positive, "
                f"got: {self.hierarchical_outer_tolerance}"
            )
        if self.hierarchical_physical_max_iterations <= 0:
            errors.append(
                f"hierarchical_physical_max_iterations must be positive, "
                f"got: {self.hierarchical_physical_max_iterations}"
            )
        if self.hierarchical_per_angle_max_iterations <= 0:
            errors.append(
                f"hierarchical_per_angle_max_iterations must be positive, "
                f"got: {self.hierarchical_per_angle_max_iterations}"
            )

        # Layer 3: Adaptive Relative Regularization
        valid_regularization_modes = ["absolute", "relative", "auto"]
        if self.regularization_mode not in valid_regularization_modes:
            errors.append(
                f"regularization_mode must be one of {valid_regularization_modes}, "
                f"got: {self.regularization_mode}"
            )
        if self.group_variance_lambda <= 0:
            errors.append(
                f"group_variance_lambda must be positive, "
                f"got: {self.group_variance_lambda}"
            )
        if not 0 < self.regularization_target_cv < 1:
            errors.append(
                f"regularization_target_cv must be in (0, 1), "
                f"got: {self.regularization_target_cv}"
            )
        if not 0 < self.regularization_target_contribution < 1:
            errors.append(
                f"regularization_target_contribution must be in (0, 1), "
                f"got: {self.regularization_target_contribution}"
            )
        if not 0 < self.regularization_max_cv < 1:
            errors.append(
                f"regularization_max_cv must be in (0, 1), "
                f"got: {self.regularization_max_cv}"
            )

        # Layer 4: Gradient Collapse Detection
        if self.gradient_ratio_threshold <= 0:
            errors.append(
                f"gradient_ratio_threshold must be positive, "
                f"got: {self.gradient_ratio_threshold}"
            )
        if self.gradient_consecutive_triggers <= 0:
            errors.append(
                f"gradient_consecutive_triggers must be positive, "
                f"got: {self.gradient_consecutive_triggers}"
            )
        valid_collapse_responses = ["warn", "hierarchical", "reset", "abort"]
        if self.gradient_collapse_response not in valid_collapse_responses:
            errors.append(
                f"gradient_collapse_response must be one of {valid_collapse_responses}, "
                f"got: {self.gradient_collapse_response}"
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

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            # NLSQ Workflow Settings (v2.11.0+)
            "workflow": self.workflow,
            "goal": self.goal,
            "loss": self.loss,
            "trust_region_scale": self.trust_region_scale,
            "max_iterations": self.max_iterations,
            "ftol": self.ftol,
            "xtol": self.xtol,
            "gtol": self.gtol,
            "x_scale": self.x_scale,
            "x_scale_map": self.x_scale_map,
            "diagnostics": {
                "enable": self.enable_diagnostics,
            },
            "streaming": {
                "enable": self.enable_streaming,
                "chunk_size": self.streaming_chunk_size,
            },
            "stratified": {
                "enable": self.enable_stratified,
                "target_chunk_size": self.target_chunk_size,
            },
            "recovery": {
                "enable": self.enable_recovery,
                "max_attempts": self.max_recovery_attempts,
            },
            "progress": {
                "enable": self.enable_progress_bar,
                "verbose": self.verbose,
                "log_interval": self.log_iteration_interval,
            },
            "hybrid_streaming": {
                "enable": self.enable_hybrid_streaming,
                "normalize": self.hybrid_normalize,
                "normalization_strategy": self.hybrid_normalization_strategy,
                "warmup_iterations": self.hybrid_warmup_iterations,
                "max_warmup_iterations": self.hybrid_max_warmup_iterations,
                "warmup_learning_rate": self.hybrid_warmup_learning_rate,
                "gauss_newton_max_iterations": self.hybrid_gauss_newton_max_iterations,
                "gauss_newton_tol": self.hybrid_gauss_newton_tol,
                "chunk_size": self.hybrid_chunk_size,
                "trust_region_initial": self.hybrid_trust_region_initial,
                "regularization_factor": self.hybrid_regularization_factor,
                "enable_checkpoints": self.hybrid_enable_checkpoints,
                "checkpoint_frequency": self.hybrid_checkpoint_frequency,
                "validate_numerics": self.hybrid_validate_numerics,
                # 4-Layer Defense Strategy (v2.8.0 / NLSQ 0.3.6)
                "enable_warm_start_detection": self.hybrid_enable_warm_start_detection,
                "warm_start_threshold": self.hybrid_warm_start_threshold,
                "enable_adaptive_warmup_lr": self.hybrid_enable_adaptive_warmup_lr,
                "warmup_lr_refinement": self.hybrid_warmup_lr_refinement,
                "warmup_lr_careful": self.hybrid_warmup_lr_careful,
                "enable_cost_guard": self.hybrid_enable_cost_guard,
                "cost_increase_tolerance": self.hybrid_cost_increase_tolerance,
                "enable_step_clipping": self.hybrid_enable_step_clipping,
                "max_warmup_step_size": self.hybrid_max_warmup_step_size,
            },
            "multi_start": {
                "enable": self.enable_multi_start,
                "n_starts": self.multi_start_n_starts,
                "seed": self.multi_start_seed,
                "sampling_strategy": self.multi_start_sampling_strategy,
                "n_workers": self.multi_start_n_workers,
                "use_screening": self.multi_start_use_screening,
                "screen_keep_fraction": self.multi_start_screen_keep_fraction,
                "refine_top_k": self.multi_start_refine_top_k,
                "refinement_ftol": self.multi_start_refinement_ftol,
                "degeneracy_threshold": self.multi_start_degeneracy_threshold,
            },
            # Anti-Degeneracy Defense System (v2.9.0)
            "anti_degeneracy": {
                "per_angle_mode": self.per_angle_mode,
                "fourier_order": self.fourier_order,
                "fourier_auto_threshold": self.fourier_auto_threshold,
                "constant_scaling_threshold": self.constant_scaling_threshold,
                "hierarchical": {
                    "enable": self.enable_hierarchical,
                    "max_outer_iterations": self.hierarchical_max_outer_iterations,
                    "outer_tolerance": self.hierarchical_outer_tolerance,
                    "physical_max_iterations": self.hierarchical_physical_max_iterations,
                    "per_angle_max_iterations": self.hierarchical_per_angle_max_iterations,
                },
                "regularization": {
                    "mode": self.regularization_mode,
                    "lambda": self.group_variance_lambda,
                    "target_cv": self.regularization_target_cv,
                    "target_contribution": self.regularization_target_contribution,
                    "max_cv": self.regularization_max_cv,
                    "auto_tune_lambda": self.regularization_auto_tune_lambda,
                },
                "gradient_monitoring": {
                    "enable": self.enable_gradient_monitoring,
                    "ratio_threshold": self.gradient_ratio_threshold,
                    "consecutive_triggers": self.gradient_consecutive_triggers,
                    "response": self.gradient_collapse_response,
                },
            },
        }

    def to_workflow_kwargs(self) -> dict[str, Any]:
        """Convert workflow settings to kwargs for NLSQ's CurveFit.

        Maps NLSQConfig workflow/goal settings to NLSQ v0.4+ API parameters.

        Returns
        -------
        dict
            Kwargs for CurveFit or WorkflowSelector.

        Notes
        -----
        For NLSQ v0.4+, these kwargs can be passed to:
        - nlsq.CurveFit.curve_fit()
        - nlsq.fit()
        - homodyne's select_nlsq_strategy() (recommended replacement)

        Example
        -------
        >>> config = NLSQConfig.from_dict(yaml_config)
        >>> kwargs = config.to_workflow_kwargs()
        >>> result = fitter.curve_fit(f, xdata, ydata, **kwargs)
        """
        kwargs: dict[str, Any] = {}

        # Map workflow setting
        if self.workflow != "auto":
            # Map homodyne workflow names to NLSQ tier names
            workflow_map = {
                "standard": "STANDARD",
                "chunked": "CHUNKED",
                "streaming": "STREAMING",
            }
            kwargs["workflow_tier"] = workflow_map.get(self.workflow, self.workflow)

        # Map goal setting
        if self.goal != "quality":  # quality is default
            goal_map = {
                "fast": "FAST",
                "robust": "ROBUST",
                "quality": "QUALITY",
                "memory_efficient": "MEMORY_EFFICIENT",
            }
            kwargs["optimization_goal"] = goal_map.get(self.goal, self.goal)

        # Add convergence settings
        kwargs["ftol"] = self.ftol
        kwargs["gtol"] = self.gtol
        kwargs["xtol"] = self.xtol
        kwargs["max_nfev"] = self.max_iterations

        # Add loss setting
        kwargs["loss"] = self.loss

        return kwargs
