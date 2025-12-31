"""Anti-Degeneracy Controller - Orchestrator for 4-Layer Defense System.

This module provides a clean interface for initializing and coordinating
the 4-layer anti-degeneracy defense system for NLSQ optimization.

The controller encapsulates:
- Layer 1: Fourier Reparameterization
- Layer 2: Hierarchical Optimization
- Layer 3: Adaptive CV-based Regularization
- Layer 4: Gradient Collapse Monitoring

Usage::

    controller = AntiDegeneracyController.from_config(
        config_dict, n_phi, phi_angles, n_physical
    )
    if controller.is_enabled:
        # Use controller.fourier, controller.hierarchical, etc.
        transformed_params = controller.transform_params_to_fourier(initial_params)
        model_fn = controller.wrap_model_fn(base_model_fn)

Version: 2.9.0
Author: Claude Code
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homodyne.optimization.nlsq.adaptive_regularization import (
    AdaptiveRegularizationConfig,
    AdaptiveRegularizer,
)
from homodyne.optimization.nlsq.fourier_reparam import (
    FourierReparamConfig,
    FourierReparameterizer,
)
from homodyne.optimization.nlsq.gradient_monitor import (
    GradientCollapseMonitor,
    GradientMonitorConfig,
)
from homodyne.optimization.nlsq.hierarchical import (
    HierarchicalConfig,
    HierarchicalOptimizer,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AntiDegeneracyConfig:
    """Configuration for the Anti-Degeneracy Defense System.

    Attributes
    ----------
    enable : bool
        Master switch for all anti-degeneracy defenses.
    per_angle_mode : str
        Mode for per-angle parameters: "independent", "fourier", or "auto".
    fourier_order : int
        Order of Fourier series (order=2 -> 5 coefficients per group).
    fourier_auto_threshold : int
        n_phi threshold for auto mode to switch to Fourier.
    hierarchical_enable : bool
        Enable hierarchical two-stage optimization.
    hierarchical_max_outer_iterations : int
        Maximum outer iterations for hierarchical optimization.
    hierarchical_outer_tolerance : float
        Convergence tolerance on physical parameter change.
    regularization_mode : str
        Regularization mode: "absolute", "relative", or "auto".
    regularization_lambda : float
        Base regularization strength.
    regularization_target_cv : float
        Target coefficient of variation (0-1).
    regularization_target_contribution : float
        Target regularization contribution to loss (0-1).
    gradient_monitoring_enable : bool
        Enable gradient collapse monitoring.
    gradient_ratio_threshold : float
        Collapse threshold for norm(grad_physical)/norm(grad_per_angle).
    gradient_consecutive_triggers : int
        Number of consecutive triggers to confirm collapse.
    gradient_response_mode : str
        Response action: "warn", "hierarchical", "reset", "abort".
    """

    enable: bool = True
    per_angle_mode: str = "auto"
    fourier_order: int = 2
    fourier_auto_threshold: int = 6
    hierarchical_enable: bool = True
    hierarchical_max_outer_iterations: int = 5
    hierarchical_outer_tolerance: float = 1e-6
    hierarchical_physical_max_iterations: int = 100
    hierarchical_per_angle_max_iterations: int = 50
    regularization_mode: str = "relative"
    regularization_lambda: float = 1.0
    regularization_target_cv: float = 0.10
    regularization_target_contribution: float = 0.10
    regularization_max_cv: float = 0.20
    gradient_monitoring_enable: bool = True
    gradient_ratio_threshold: float = 0.01
    gradient_consecutive_triggers: int = 5
    gradient_response_mode: str = "hierarchical"

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> AntiDegeneracyConfig:
        """Create config from nested dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary with structure::

                {
                    "enable": bool,
                    "per_angle_mode": str,
                    "fourier_order": int,
                    "fourier_auto_threshold": int,
                    "hierarchical": {...},
                    "regularization": {...},
                    "gradient_monitoring": {...}
                }

        Returns
        -------
        AntiDegeneracyConfig
            Validated configuration object.
        """
        hierarchical = config_dict.get("hierarchical", {})
        regularization = config_dict.get("regularization", {})
        gradient_monitoring = config_dict.get("gradient_monitoring", {})

        return cls(
            enable=config_dict.get("enable", True),
            per_angle_mode=config_dict.get("per_angle_mode", "auto"),
            fourier_order=config_dict.get("fourier_order", 2),
            fourier_auto_threshold=config_dict.get("fourier_auto_threshold", 6),
            # Hierarchical
            hierarchical_enable=hierarchical.get("enable", True),
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
            # Regularization
            regularization_mode=regularization.get("mode", "relative"),
            regularization_lambda=float(regularization.get("lambda", 1.0)),
            regularization_target_cv=float(regularization.get("target_cv", 0.10)),
            regularization_target_contribution=float(
                regularization.get("target_contribution", 0.10)
            ),
            regularization_max_cv=float(regularization.get("max_cv", 0.20)),
            # Gradient monitoring
            gradient_monitoring_enable=gradient_monitoring.get("enable", True),
            gradient_ratio_threshold=float(
                gradient_monitoring.get("ratio_threshold", 0.01)
            ),
            gradient_consecutive_triggers=gradient_monitoring.get(
                "consecutive_triggers", 5
            ),
            gradient_response_mode=gradient_monitoring.get("response", "hierarchical"),
        )


@dataclass
class AntiDegeneracyController:
    """Orchestrator for the 4-Layer Anti-Degeneracy Defense System.

    This controller provides a clean interface for initializing and
    coordinating all anti-degeneracy components.

    Attributes
    ----------
    config : AntiDegeneracyConfig
        Configuration for the defense system.
    n_phi : int
        Number of phi angles.
    n_physical : int
        Number of physical parameters.
    phi_angles : np.ndarray
        Array of phi angles in radians.
    fourier : FourierReparameterizer | None
        Layer 1: Fourier reparameterization component.
    hierarchical : HierarchicalOptimizer | None
        Layer 2: Hierarchical optimization component.
    regularizer : AdaptiveRegularizer | None
        Layer 3: Adaptive regularization component.
    monitor : GradientCollapseMonitor | None
        Layer 4: Gradient collapse monitoring component.
    per_angle_mode_actual : str
        Actual mode used ("fourier" or "independent").
    """

    config: AntiDegeneracyConfig
    n_phi: int
    n_physical: int
    phi_angles: np.ndarray
    fourier: FourierReparameterizer | None = None
    hierarchical: HierarchicalOptimizer | None = None
    regularizer: AdaptiveRegularizer | None = None
    monitor: GradientCollapseMonitor | None = None
    per_angle_mode_actual: str = "independent"
    _is_initialized: bool = field(default=False, repr=False)

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        n_phi: int,
        phi_angles: np.ndarray,
        n_physical: int,
        per_angle_scaling: bool = True,
        is_laminar_flow: bool = True,
    ) -> AntiDegeneracyController:
        """Create controller from configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Anti-degeneracy configuration dictionary.
        n_phi : int
            Number of phi angles.
        phi_angles : np.ndarray
            Array of phi angles in radians.
        n_physical : int
            Number of physical parameters (7 for laminar_flow).
        per_angle_scaling : bool
            Whether per-angle scaling is enabled.
        is_laminar_flow : bool
            Whether this is laminar_flow mode.

        Returns
        -------
        AntiDegeneracyController
            Initialized controller with all components.
        """
        config = AntiDegeneracyConfig.from_dict(config_dict)

        controller = cls(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=phi_angles,
        )

        # Only initialize if enabled and appropriate mode
        if config.enable and per_angle_scaling and is_laminar_flow:
            controller._initialize_components()

        return controller

    def _initialize_components(self) -> None:
        """Initialize all 4 layers of the defense system."""
        config = self.config

        # Determine actual per-angle mode
        if config.per_angle_mode == "auto":
            self.per_angle_mode_actual = (
                "fourier" if self.n_phi > config.fourier_auto_threshold else "independent"
            )
        else:
            self.per_angle_mode_actual = config.per_angle_mode

        # Layer 1: Fourier Reparameterization
        if self.per_angle_mode_actual == "fourier":
            fourier_config = FourierReparamConfig(
                mode="fourier",
                fourier_order=config.fourier_order,
                auto_threshold=config.fourier_auto_threshold,
            )
            self.fourier = FourierReparameterizer(self.phi_angles, fourier_config)
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 1 - Fourier Reparameterization")
            logger.info(f"  Mode: {self.per_angle_mode_actual}")
            logger.info(f"  n_phi: {self.n_phi}, Fourier order: {config.fourier_order}")
            logger.info(f"  Parameter reduction: {2*self.n_phi} -> {self.fourier.n_coeffs}")
            logger.info("=" * 60)

        # Layer 2: Hierarchical Optimization
        if config.hierarchical_enable:
            hier_config = HierarchicalConfig(
                enable=True,
                max_outer_iterations=config.hierarchical_max_outer_iterations,
                outer_tolerance=config.hierarchical_outer_tolerance,
                physical_max_iterations=config.hierarchical_physical_max_iterations,
                per_angle_max_iterations=config.hierarchical_per_angle_max_iterations,
            )
            self.hierarchical = HierarchicalOptimizer(
                config=hier_config,
                n_phi=self.n_phi,
                n_physical=self.n_physical,
                fourier_reparameterizer=self.fourier,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 2 - Hierarchical Optimization")
            logger.info("  Enabled: True")
            logger.info(f"  Max outer iterations: {config.hierarchical_max_outer_iterations}")
            logger.info(f"  Outer tolerance: {config.hierarchical_outer_tolerance}")
            logger.info("=" * 60)

        # Layer 3: Adaptive Regularization
        reg_config = AdaptiveRegularizationConfig(
            enable=True,
            mode=config.regularization_mode,
            lambda_base=config.regularization_lambda,
            target_cv=config.regularization_target_cv,
            target_contribution=config.regularization_target_contribution,
            max_cv=config.regularization_max_cv,
        )
        self.regularizer = AdaptiveRegularizer(reg_config, self.n_phi)
        logger.info("=" * 60)
        logger.info("ANTI-DEGENERACY: Layer 3 - Adaptive Regularization")
        logger.info(f"  Mode: {config.regularization_mode}")
        logger.info(f"  Auto-tuned lambda: {self.regularizer.lambda_value:.2f}")
        logger.info(f"  Target CV: {config.regularization_target_cv}")
        logger.info("=" * 60)

        # Layer 4: Gradient Collapse Monitor
        if config.gradient_monitoring_enable:
            n_per_angle = (
                self.fourier.n_coeffs if self.fourier else 2 * self.n_phi
            )
            per_angle_indices = list(range(n_per_angle))
            physical_indices = list(range(n_per_angle, n_per_angle + self.n_physical))

            monitor_config = GradientMonitorConfig(
                enable=True,
                ratio_threshold=config.gradient_ratio_threshold,
                consecutive_triggers=config.gradient_consecutive_triggers,
                response_mode=config.gradient_response_mode,
            )
            self.monitor = GradientCollapseMonitor(
                config=monitor_config,
                physical_indices=physical_indices,
                per_angle_indices=per_angle_indices,
            )
            logger.info("=" * 60)
            logger.info("ANTI-DEGENERACY: Layer 4 - Gradient Collapse Monitor")
            logger.info("  Enabled: True")
            logger.info(f"  Ratio threshold: {config.gradient_ratio_threshold}")
            logger.info(f"  Response mode: {config.gradient_response_mode}")
            logger.info("=" * 60)

        self._is_initialized = True

    @property
    def is_enabled(self) -> bool:
        """Check if the defense system is enabled and initialized."""
        return self._is_initialized and self.config.enable

    @property
    def use_fourier(self) -> bool:
        """Check if Fourier reparameterization is active."""
        return self.fourier is not None

    @property
    def use_hierarchical(self) -> bool:
        """Check if hierarchical optimization is active."""
        return self.hierarchical is not None

    @property
    def n_per_angle_params(self) -> int:
        """Get the number of per-angle parameters (Fourier or direct)."""
        if self.fourier:
            return self.fourier.n_coeffs
        return 2 * self.n_phi

    def transform_params_to_fourier(
        self, params: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Transform per-angle parameters to Fourier coefficients.

        Parameters
        ----------
        params : np.ndarray
            Full parameter array: [contrast(n_phi), offset(n_phi), physical].

        Returns
        -------
        tuple
            (fourier_params, original_bounds_if_transformed)
            fourier_params: [contrast_coeffs, offset_coeffs, physical]
            bounds: (lower, upper) in Fourier space if transformation applied
        """
        if not self.use_fourier:
            return params, None

        # Split parameters
        contrast = params[: self.n_phi]
        offset = params[self.n_phi : 2 * self.n_phi]
        physical = params[2 * self.n_phi :]

        # Transform to Fourier
        contrast_coeffs = self.fourier.to_fourier(contrast)
        offset_coeffs = self.fourier.to_fourier(offset)

        return np.concatenate([contrast_coeffs, offset_coeffs, physical]), None

    def transform_params_from_fourier(self, fourier_params: np.ndarray) -> np.ndarray:
        """Transform Fourier coefficients back to per-angle parameters.

        Parameters
        ----------
        fourier_params : np.ndarray
            Fourier parameter array: [contrast_coeffs, offset_coeffs, physical].

        Returns
        -------
        np.ndarray
            Per-angle parameter array: [contrast(n_phi), offset(n_phi), physical].
        """
        if not self.use_fourier:
            return fourier_params

        n_coeffs = self.fourier.n_coeffs_per_param

        # Extract Fourier coefficients
        contrast_coeffs = fourier_params[:n_coeffs]
        offset_coeffs = fourier_params[n_coeffs : 2 * n_coeffs]
        physical = fourier_params[2 * n_coeffs :]

        # Transform back to per-angle
        contrast = self.fourier.from_fourier(contrast_coeffs)
        offset = self.fourier.from_fourier(offset_coeffs)

        return np.concatenate([contrast, offset, physical])

    def get_group_variance_indices(self) -> list[tuple[int, int]] | None:
        """Get group variance indices for NLSQ regularization.

        Returns
        -------
        list[tuple[int, int]] | None
            List of (start, end) tuples for each parameter group.
        """
        if not self.is_enabled:
            return None

        n_per_group = (
            self.fourier.n_coeffs_per_param if self.use_fourier else self.n_phi
        )
        return [(0, n_per_group), (n_per_group, 2 * n_per_group)]

    def get_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostics from all components.

        Returns
        -------
        dict
            Nested diagnostics from all 4 layers.
        """
        diag: dict[str, Any] = {
            "version": "2.9.0",
            "enabled": self.is_enabled,
            "per_angle_mode": self.per_angle_mode_actual,
            "n_phi": self.n_phi,
            "n_physical": self.n_physical,
        }

        if self.fourier:
            diag["fourier"] = self.fourier.get_diagnostics()

        if self.hierarchical:
            diag["hierarchical"] = self.hierarchical.get_diagnostics()

        if self.regularizer:
            diag["regularization"] = self.regularizer.get_diagnostics()

        if self.monitor:
            diag["gradient_monitor"] = self.monitor.get_diagnostics()

        return diag

    def reset_monitor(self) -> None:
        """Reset the gradient collapse monitor state."""
        if self.monitor:
            self.monitor.reset()
