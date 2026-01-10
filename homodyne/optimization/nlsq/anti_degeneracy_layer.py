"""Anti-degeneracy layer interface and implementations (FR-013).

Provides a unified interface for anti-degeneracy defense layers:
- FourierReparamLayer: Fourier reparameterization (Layer 1)
- HierarchicalLayer: Hierarchical optimization (Layer 2)
- AdaptiveRegularizationLayer: CV-based regularization (Layer 3)
- GradientMonitorLayer: Gradient collapse detection (Layer 4)
- ShearWeightingLayer: Shear-sensitivity weighting (Layer 5)
- AntiDegeneracyChain: Executor for layer chain

Created as part of architecture refactoring (T064-T070).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationState:
    """State passed between anti-degeneracy layers.

    Represents the current state of optimization including parameters,
    residuals, gradients, and metadata.

    Attributes
    ----------
    params : np.ndarray
        Current parameter values
    residuals : np.ndarray
        Current residual values
    iteration : int
        Current iteration number
    chi_squared : float
        Current chi-squared value
    gradient : np.ndarray | None
        Current gradient vector
    jacobian : np.ndarray | None
        Current Jacobian matrix (optional, can be large)
    metadata : dict[str, Any]
        Additional metadata passed between layers
    """

    params: np.ndarray
    residuals: np.ndarray
    iteration: int
    chi_squared: float
    gradient: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AntiDegeneracyLayer(ABC):
    """Abstract base class for anti-degeneracy defense layers.

    Each layer implements a specific defense against parameter degeneracy:
    - Layer 1 (Fourier): Reduces per-angle parameters via Fourier basis
    - Layer 2 (Hierarchical): Alternates physical/per-angle optimization
    - Layer 3 (Regularization): Adaptive CV-based regularization
    - Layer 4 (Monitor): Gradient collapse detection
    - Layer 5 (Weighting): Shear-sensitivity residual weighting
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the layer name for logging."""
        ...

    @abstractmethod
    def apply(self, state: OptimizationState) -> OptimizationState:
        """Apply the layer transformation to optimization state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            Transformed optimization state
        """
        ...


class FourierReparamLayer(AntiDegeneracyLayer):
    """Layer 1: Fourier reparameterization wrapper.

    Wraps FourierReparameterization to reduce per-angle parameter count
    through Fourier basis expansion.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize FourierReparamLayer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration with keys:
            - 'fourier_order': Fourier expansion order
            - 'n_phi': Number of phi angles
        """
        self._config = config
        self._fourier_order = config.get("fourier_order", 2)
        self._n_phi = config.get("n_phi", 0)
        self._reparameterizer = None

    @property
    def name(self) -> str:
        return "FourierReparamLayer"

    def apply(self, state: OptimizationState) -> OptimizationState:
        """Apply Fourier reparameterization to state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            State with Fourier-reparameterized per-angle params
        """
        # Skip if n_phi is too small for Fourier expansion
        if self._n_phi < 3:
            logger.debug(f"{self.name}: Skipping, n_phi={self._n_phi} < 3")
            return state

        try:
            from homodyne.optimization.nlsq.fourier_reparam import (
                FourierReparameterization,
            )

            if self._reparameterizer is None:
                self._reparameterizer = FourierReparameterization(
                    n_phi=self._n_phi,
                    fourier_order=self._fourier_order,
                )

            # Apply transformation if applicable
            # Note: actual transformation happens in the residual function
            state.metadata["fourier_reparameterization"] = self._reparameterizer
            state.metadata["fourier_enabled"] = True

            logger.debug(
                f"{self.name}: Enabled Fourier reparameterization "
                f"(order={self._fourier_order}, n_phi={self._n_phi})"
            )

        except ImportError:
            logger.warning(f"{self.name}: FourierReparameterization not available")

        return state


class HierarchicalLayer(AntiDegeneracyLayer):
    """Layer 2: Hierarchical optimization wrapper.

    Wraps HierarchicalOptimizer to alternate between physical and
    per-angle parameter optimization.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize HierarchicalLayer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration with keys:
            - 'max_outer_iterations': Maximum outer iterations
            - 'outer_tolerance': Convergence tolerance
            - 'physical_max_iterations': Max iterations for physical params
            - 'per_angle_max_iterations': Max iterations for per-angle params
        """
        self._config = config
        self._max_outer = config.get("max_outer_iterations", 5)
        self._outer_tol = config.get("outer_tolerance", 1e-6)

    @property
    def name(self) -> str:
        return "HierarchicalLayer"

    def apply(self, state: OptimizationState) -> OptimizationState:
        """Apply hierarchical optimization setup to state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            State with hierarchical optimization metadata
        """
        state.metadata["hierarchical_enabled"] = True
        state.metadata["hierarchical_config"] = {
            "max_outer_iterations": self._max_outer,
            "outer_tolerance": self._outer_tol,
            "physical_max_iterations": self._config.get("physical_max_iterations", 100),
            "per_angle_max_iterations": self._config.get("per_angle_max_iterations", 50),
        }

        logger.debug(
            f"{self.name}: Enabled hierarchical optimization "
            f"(max_outer={self._max_outer}, tol={self._outer_tol})"
        )

        return state


class AdaptiveRegularizationLayer(AntiDegeneracyLayer):
    """Layer 3: Adaptive regularization wrapper.

    Wraps AdaptiveRegularization to auto-tune regularization strength
    based on coefficient of variation targets.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize AdaptiveRegularizationLayer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration with keys:
            - 'mode': 'absolute', 'relative', or 'auto'
            - 'lambda': Base regularization strength
            - 'target_cv': Target coefficient of variation
            - 'target_contribution': Target loss contribution fraction
        """
        self._config = config
        self._mode = config.get("mode", "relative")
        self._lambda = config.get("lambda", 1.0)
        self._target_cv = config.get("target_cv", 0.10)

    @property
    def name(self) -> str:
        return "AdaptiveRegularizationLayer"

    def apply(self, state: OptimizationState) -> OptimizationState:
        """Apply adaptive regularization setup to state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            State with regularization metadata
        """
        state.metadata["regularization_enabled"] = True
        state.metadata["regularization_config"] = {
            "mode": self._mode,
            "lambda": self._lambda,
            "target_cv": self._target_cv,
            "target_contribution": self._config.get("target_contribution", 0.10),
            "max_cv": self._config.get("max_cv", 0.20),
        }

        logger.debug(
            f"{self.name}: Enabled adaptive regularization "
            f"(mode={self._mode}, lambda={self._lambda}, target_cv={self._target_cv})"
        )

        return state


class GradientMonitorLayer(AntiDegeneracyLayer):
    """Layer 4: Gradient collapse monitoring wrapper.

    Wraps GradientCollapseMonitor to detect and respond to gradient
    collapse during optimization.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize GradientMonitorLayer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration with keys:
            - 'ratio_threshold': Trigger threshold for gradient ratio
            - 'consecutive_triggers': Required consecutive triggers
            - 'response': Response action ('warn', 'hierarchical', 'reset', 'abort')
        """
        self._config = config
        self._ratio_threshold = config.get("ratio_threshold", 0.01)
        self._consecutive_triggers = config.get("consecutive_triggers", 5)
        self._response = config.get("response", "hierarchical")
        self._trigger_count = 0

    @property
    def name(self) -> str:
        return "GradientMonitorLayer"

    def apply(self, state: OptimizationState) -> OptimizationState:
        """Check for gradient collapse and respond.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            State with gradient monitoring results
        """
        # Handle empty gradient case
        if state.gradient is None or len(state.gradient) == 0:
            return state

        # Initialize gradient history if needed
        if "gradient_history" not in state.metadata:
            state.metadata["gradient_history"] = []
            state.metadata["gradient_collapse_detected"] = False

        # Get gradient components (assume first half physical, second half per-angle)
        n_params = len(state.gradient)
        if n_params < 2:
            return state

        # Simple heuristic: physical params typically have smaller gradients
        # when absorption is occurring
        grad_mag = np.abs(state.gradient)
        median_grad = np.median(grad_mag)

        if median_grad > 0:
            min_ratio = np.min(grad_mag) / median_grad

            if min_ratio < self._ratio_threshold:
                self._trigger_count += 1
            else:
                self._trigger_count = 0

            if self._trigger_count >= self._consecutive_triggers:
                state.metadata["gradient_collapse_detected"] = True
                logger.warning(
                    f"{self.name}: Gradient collapse detected "
                    f"(ratio={min_ratio:.4f}, triggers={self._trigger_count})"
                )

        state.metadata["gradient_history"].append(np.copy(state.gradient))

        return state


class ShearWeightingLayer(AntiDegeneracyLayer):
    """Layer 5: Shear-sensitivity weighting wrapper.

    Wraps shear weighting to weight residuals by abs(cos(phi_0 - phi))
    to prevent gradient cancellation.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize ShearWeightingLayer.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration with keys:
            - 'enable': Whether to enable weighting
            - 'min_weight': Minimum weight for perpendicular angles
            - 'alpha': Shear sensitivity exponent
            - 'phi_0': Reference angle (radians)
        """
        self._config = config
        self._enabled = config.get("enable", True)
        self._min_weight = config.get("min_weight", 0.3)
        self._alpha = config.get("alpha", 1.0)
        self._phi_0 = config.get("phi_0", None)

    @property
    def name(self) -> str:
        return "ShearWeightingLayer"

    def apply(self, state: OptimizationState) -> OptimizationState:
        """Apply shear-sensitivity weighting setup to state.

        Parameters
        ----------
        state : OptimizationState
            Current optimization state

        Returns
        -------
        OptimizationState
            State with shear weighting metadata
        """
        if not self._enabled:
            return state

        state.metadata["shear_weighting_enabled"] = True
        state.metadata["shear_weighting_config"] = {
            "min_weight": self._min_weight,
            "alpha": self._alpha,
            "phi_0": self._phi_0,
            "normalize": self._config.get("normalize", True),
            "update_frequency": self._config.get("update_frequency", 1),
        }

        logger.debug(
            f"{self.name}: Enabled shear-sensitivity weighting "
            f"(min_weight={self._min_weight}, alpha={self._alpha})"
        )

        return state


class AntiDegeneracyChain:
    """Executor for sequential anti-degeneracy layer chain.

    Executes layers in order, passing state through each layer.
    """

    def __init__(self, layers: list[AntiDegeneracyLayer]):
        """Initialize AntiDegeneracyChain.

        Parameters
        ----------
        layers : list[AntiDegeneracyLayer]
            Ordered list of layers to execute
        """
        self._layers = layers

    def execute(self, state: OptimizationState) -> OptimizationState:
        """Execute all layers in sequence.

        Parameters
        ----------
        state : OptimizationState
            Initial optimization state

        Returns
        -------
        OptimizationState
            Final state after all layer transformations
        """
        current_state = state

        for layer in self._layers:
            try:
                current_state = layer.apply(current_state)
                logger.debug(f"AntiDegeneracyChain: Applied {layer.name}")
            except Exception as e:
                logger.warning(f"AntiDegeneracyChain: Layer {layer.name} failed: {e}")
                # Continue with current state on failure

        return current_state

    def add_layer(self, layer: AntiDegeneracyLayer) -> None:
        """Add a layer to the chain.

        Parameters
        ----------
        layer : AntiDegeneracyLayer
            Layer to add
        """
        self._layers.append(layer)

    @property
    def layers(self) -> list[AntiDegeneracyLayer]:
        """Get the list of layers."""
        return self._layers


__all__ = [
    "OptimizationState",
    "AntiDegeneracyLayer",
    "FourierReparamLayer",
    "HierarchicalLayer",
    "AdaptiveRegularizationLayer",
    "GradientMonitorLayer",
    "ShearWeightingLayer",
    "AntiDegeneracyChain",
]
