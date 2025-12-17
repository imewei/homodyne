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
            errors.append(f"max_iterations must be positive, got: {self.max_iterations}")

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
        }
