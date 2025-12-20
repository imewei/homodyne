"""NLSQ configuration dataclass and validation.

This module provides the NLSQConfig dataclass for parsing and validating
NLSQ-specific configuration settings from the YAML config file.

Part of Phase 3 architecture refactoring to reduce wrapper.py complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


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

    # Hybrid streaming optimizer settings (v2.6.0)
    # Fixes: 1) Shear-term weak gradients, 2) Slow convergence, 3) Crude covariance
    enable_hybrid_streaming: bool = True
    hybrid_normalize: bool = True
    hybrid_normalization_strategy: str = "bounds"  # 'auto', 'bounds', 'p0', 'none'
    hybrid_warmup_iterations: int = 100
    hybrid_max_warmup_iterations: int = 500
    hybrid_warmup_learning_rate: float = 0.001
    hybrid_gauss_newton_max_iterations: int = 50
    hybrid_gauss_newton_tol: float = 1e-8
    hybrid_chunk_size: int = 50000
    hybrid_trust_region_initial: float = 1.0
    hybrid_regularization_factor: float = 1e-10
    hybrid_enable_checkpoints: bool = True
    hybrid_checkpoint_frequency: int = 100
    hybrid_validate_numerics: bool = True

    # Multi-start optimization settings (v2.8.0)
    # Enables exploration of parameter space via Latin Hypercube Sampling
    enable_multi_start: bool = False  # Default OFF - user opt-in
    multi_start_n_starts: int = 10
    multi_start_seed: int = 42
    multi_start_sampling_strategy: str = (
        "latin_hypercube"  # 'latin_hypercube' or 'random'
    )
    multi_start_n_workers: int = 0  # 0 = auto (min of n_starts, cpu_count)
    multi_start_use_screening: bool = True
    multi_start_screen_keep_fraction: float = 0.5
    multi_start_subsample_size: int = 500_000  # For 1M-100M datasets
    multi_start_warmup_only_threshold: int = (
        100_000_000  # 100M: switch to phase1 strategy
    )
    multi_start_refine_top_k: int = 3
    multi_start_refinement_ftol: float = 1e-12
    multi_start_degeneracy_threshold: float = 0.1

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

        config = cls(
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
            # Multi-start (v2.8.0)
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
            multi_start_subsample_size=multi_start.get("subsample_size", 500_000),
            multi_start_warmup_only_threshold=multi_start.get(
                "warmup_only_threshold", 100_000_000
            ),
            multi_start_refine_top_k=multi_start.get("refine_top_k", 3),
            multi_start_refinement_ftol=float(
                multi_start.get("refinement_ftol", 1e-12)
            ),
            multi_start_degeneracy_threshold=float(
                multi_start.get("degeneracy_threshold", 0.1)
            ),
        )

        # Validate and log any issues
        errors = config.validate()
        if errors:
            for error in errors:
                logger.warning(f"NLSQ config validation: {error}")

        return config

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
        if self.multi_start_subsample_size <= 0:
            errors.append(
                f"multi_start_subsample_size must be positive, "
                f"got: {self.multi_start_subsample_size}"
            )
        if self.multi_start_warmup_only_threshold <= 0:
            errors.append(
                f"multi_start_warmup_only_threshold must be positive, "
                f"got: {self.multi_start_warmup_only_threshold}"
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
            },
            "multi_start": {
                "enable": self.enable_multi_start,
                "n_starts": self.multi_start_n_starts,
                "seed": self.multi_start_seed,
                "sampling_strategy": self.multi_start_sampling_strategy,
                "n_workers": self.multi_start_n_workers,
                "use_screening": self.multi_start_use_screening,
                "screen_keep_fraction": self.multi_start_screen_keep_fraction,
                "subsample_size": self.multi_start_subsample_size,
                "warmup_only_threshold": self.multi_start_warmup_only_threshold,
                "refine_top_k": self.multi_start_refine_top_k,
                "refinement_ftol": self.multi_start_refinement_ftol,
                "degeneracy_threshold": self.multi_start_degeneracy_threshold,
            },
        }
