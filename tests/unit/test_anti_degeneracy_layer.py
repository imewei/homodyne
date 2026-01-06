"""Tests for AntiDegeneracyLayer interface (T056, T058).

Tests the abstract layer interface and independent layer testing capability.
"""

from __future__ import annotations

import numpy as np
import pytest
from dataclasses import dataclass


class TestAntiDegeneracyLayerInterface:
    """Tests for T056: AntiDegeneracyLayer interface."""

    def test_layer_interface_exists(self):
        """Test that AntiDegeneracyLayer ABC exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyLayer,
        )

        assert hasattr(AntiDegeneracyLayer, "apply")
        assert hasattr(AntiDegeneracyLayer, "name")

    def test_optimization_state_dataclass_exists(self):
        """Test that OptimizationState dataclass exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
        )

        # Should be a dataclass with expected fields
        state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1, 0.2]),
            iteration=0,
            chi_squared=0.05,
            gradient=np.array([0.01, 0.02]),
            jacobian=None,
            metadata={},
        )
        assert state.params is not None
        assert state.iteration == 0

    def test_fourier_reparam_layer_exists(self):
        """Test that FourierReparamLayer wrapper exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            FourierReparamLayer,
        )

        assert hasattr(FourierReparamLayer, "apply")

    def test_hierarchical_layer_exists(self):
        """Test that HierarchicalLayer wrapper exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            HierarchicalLayer,
        )

        assert hasattr(HierarchicalLayer, "apply")

    def test_adaptive_regularization_layer_exists(self):
        """Test that AdaptiveRegularizationLayer wrapper exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AdaptiveRegularizationLayer,
        )

        assert hasattr(AdaptiveRegularizationLayer, "apply")

    def test_gradient_monitor_layer_exists(self):
        """Test that GradientMonitorLayer wrapper exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            GradientMonitorLayer,
        )

        assert hasattr(GradientMonitorLayer, "apply")

    def test_shear_weighting_layer_exists(self):
        """Test that ShearWeightingLayer wrapper exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            ShearWeightingLayer,
        )

        assert hasattr(ShearWeightingLayer, "apply")

    def test_anti_degeneracy_chain_exists(self):
        """Test that AntiDegeneracyChain executor exists."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
        )

        assert hasattr(AntiDegeneracyChain, "execute")


class TestLayerIndependentTesting:
    """Tests for T058: Independent layer testing with mock OptimizationState."""

    def test_layer_independent_testing(self):
        """Test that layers can be tested independently with mock state."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            FourierReparamLayer,
        )

        # Create a mock state
        mock_state = OptimizationState(
            params=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            residuals=np.array([0.1, 0.2, 0.3]),
            iteration=5,
            chi_squared=0.1,
            gradient=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            jacobian=None,
            metadata={"n_phi": 3},
        )

        # Create a layer with mock config
        mock_config = {"fourier_order": 2, "n_phi": 3}
        layer = FourierReparamLayer(mock_config)

        # Layer should accept state and return modified state
        # (may return same state if not applicable)
        result_state = layer.apply(mock_state)

        assert isinstance(result_state, OptimizationState)
        assert result_state.iteration == mock_state.iteration

    def test_layer_applies_transformation(self):
        """Test that applying a layer transforms state correctly."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            GradientMonitorLayer,
        )

        # Create state with gradient information
        mock_state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
            gradient=np.array([0.5, 0.001]),  # Imbalanced gradients
            jacobian=None,
            metadata={"gradient_history": []},
        )

        layer = GradientMonitorLayer(
            {"ratio_threshold": 0.01, "consecutive_triggers": 3}
        )
        result_state = layer.apply(mock_state)

        # State should be returned (may have updated metadata)
        assert isinstance(result_state, OptimizationState)

    def test_chain_executes_layers_sequentially(self):
        """Test that AntiDegeneracyChain executes layers in order."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            AntiDegeneracyChain,
            GradientMonitorLayer,
        )

        mock_state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
            gradient=np.array([0.5, 0.5]),
            jacobian=None,
            metadata={},
        )

        # Create chain with single layer
        layer = GradientMonitorLayer({"ratio_threshold": 0.01, "consecutive_triggers": 3})
        chain = AntiDegeneracyChain([layer])

        result = chain.execute(mock_state)
        assert isinstance(result, OptimizationState)

    def test_layer_with_empty_state(self):
        """Test that layers handle edge case of minimal state."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            GradientMonitorLayer,
        )

        # Minimal state
        mock_state = OptimizationState(
            params=np.array([]),
            residuals=np.array([]),
            iteration=0,
            chi_squared=0.0,
            gradient=np.array([]),
            jacobian=None,
            metadata={},
        )

        layer = GradientMonitorLayer({"ratio_threshold": 0.01, "consecutive_triggers": 3})
        result_state = layer.apply(mock_state)

        # Should handle gracefully without error
        assert isinstance(result_state, OptimizationState)
