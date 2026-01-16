"""
Unit Tests for Anti-Degeneracy Layer Interface
===============================================

Tests for homodyne/optimization/nlsq/anti_degeneracy_layer.py covering:
- TestOptimizationState: State dataclass
- TestAntiDegeneracyLayer: ABC interface
- TestFourierReparamLayer: Layer 1 wrapper
- TestHierarchicalLayer: Layer 2 wrapper
- TestAdaptiveRegularizationLayer: Layer 3 wrapper
- TestGradientMonitorLayer: Layer 4 wrapper
- TestShearWeightingLayer: Layer 5 wrapper
- TestAntiDegeneracyChain: Layer chain executor

Part of v2.14.0 architecture refactoring tests.
"""

import numpy as np
import pytest


# =============================================================================
# TestOptimizationState
# =============================================================================
@pytest.mark.unit
class TestOptimizationState:
    """Tests for OptimizationState dataclass."""

    def test_state_creation_minimal(self):
        """State can be created with minimal required fields."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import OptimizationState

        state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1, 0.2, 0.3]),
            iteration=0,
            chi_squared=0.05,
        )

        assert len(state.params) == 2
        assert len(state.residuals) == 3
        assert state.iteration == 0
        assert state.chi_squared == 0.05

    def test_state_optional_fields_default_none(self):
        """Optional fields default to None or empty dict."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import OptimizationState

        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=1,
            chi_squared=0.01,
        )

        assert state.gradient is None
        assert state.jacobian is None
        assert state.metadata == {}

    def test_state_with_gradient_and_jacobian(self):
        """State accepts gradient and jacobian."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import OptimizationState

        gradient = np.array([0.01, 0.02])
        jacobian = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])

        state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1, 0.2, 0.3]),
            iteration=5,
            chi_squared=0.02,
            gradient=gradient,
            jacobian=jacobian,
        )

        np.testing.assert_array_equal(state.gradient, gradient)
        np.testing.assert_array_equal(state.jacobian, jacobian)

    def test_state_metadata_mutable(self):
        """Metadata can be modified after creation."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import OptimizationState

        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        state.metadata["layer_applied"] = "FourierReparamLayer"
        state.metadata["custom_value"] = 42

        assert state.metadata["layer_applied"] == "FourierReparamLayer"
        assert state.metadata["custom_value"] == 42


# =============================================================================
# TestAntiDegeneracyLayer
# =============================================================================
@pytest.mark.unit
class TestAntiDegeneracyLayer:
    """Tests for AntiDegeneracyLayer ABC."""

    def test_cannot_instantiate_abc(self):
        """AntiDegeneracyLayer cannot be instantiated directly."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyLayer,
        )

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            AntiDegeneracyLayer()

    def test_subclass_must_implement_name(self):
        """Subclass must implement name property."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyLayer,
            OptimizationState,
        )

        class IncompleteLayer(AntiDegeneracyLayer):
            def apply(self, state):
                return state

        with pytest.raises(TypeError):
            IncompleteLayer()

    def test_subclass_must_implement_apply(self):
        """Subclass must implement apply method."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyLayer,
        )

        class IncompleteLayer(AntiDegeneracyLayer):
            @property
            def name(self):
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteLayer()

    def test_complete_subclass_can_be_instantiated(self):
        """Complete subclass can be instantiated."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyLayer,
            OptimizationState,
        )

        class CompleteLayer(AntiDegeneracyLayer):
            @property
            def name(self):
                return "CompleteLayer"

            def apply(self, state):
                state.metadata["applied"] = True
                return state

        layer = CompleteLayer()
        assert layer.name == "CompleteLayer"


# =============================================================================
# TestFourierReparamLayer
# =============================================================================
@pytest.mark.unit
class TestFourierReparamLayer:
    """Tests for FourierReparamLayer (Layer 1)."""

    def test_layer_name(self):
        """Layer has correct name."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import FourierReparamLayer

        layer = FourierReparamLayer({"fourier_order": 2, "n_phi": 10})
        assert layer.name == "FourierReparamLayer"

    def test_skips_small_n_phi(self):
        """Layer skips when n_phi < 3."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            FourierReparamLayer,
            OptimizationState,
        )

        layer = FourierReparamLayer({"fourier_order": 2, "n_phi": 2})
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        # Fourier should not be enabled
        assert result.metadata.get("fourier_enabled", False) is False

    def test_enables_fourier_for_sufficient_n_phi(self):
        """Layer enables Fourier for n_phi >= 3 when FourierReparameterizer available."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            FourierReparamLayer,
            OptimizationState,
        )

        # Check if FourierReparameterizer is available
        try:
            from homodyne.optimization.nlsq.fourier_reparam import (
                FourierReparameterizer,
            )
            fourier_available = True
        except ImportError:
            fourier_available = False

        layer = FourierReparamLayer({"fourier_order": 2, "n_phi": 10})
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        if fourier_available:
            assert result.metadata.get("fourier_enabled") is True
            assert "fourier_reparameterization" in result.metadata
        else:
            # If import fails, fourier_enabled should not be set
            assert result.metadata.get("fourier_enabled", False) is False


# =============================================================================
# TestHierarchicalLayer
# =============================================================================
@pytest.mark.unit
class TestHierarchicalLayer:
    """Tests for HierarchicalLayer (Layer 2)."""

    def test_layer_name(self):
        """Layer has correct name."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import HierarchicalLayer

        layer = HierarchicalLayer({})
        assert layer.name == "HierarchicalLayer"

    def test_sets_hierarchical_metadata(self):
        """Layer sets hierarchical config in metadata."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            HierarchicalLayer,
            OptimizationState,
        )

        config = {
            "max_outer_iterations": 10,
            "outer_tolerance": 1e-8,
            "physical_max_iterations": 200,
            "per_angle_max_iterations": 100,
        }
        layer = HierarchicalLayer(config)
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        assert result.metadata["hierarchical_enabled"] is True
        assert result.metadata["hierarchical_config"]["max_outer_iterations"] == 10
        assert result.metadata["hierarchical_config"]["outer_tolerance"] == 1e-8

    def test_uses_default_config_values(self):
        """Layer uses defaults when config is empty."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            HierarchicalLayer,
            OptimizationState,
        )

        layer = HierarchicalLayer({})
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        # Check defaults
        assert result.metadata["hierarchical_config"]["max_outer_iterations"] == 5
        assert result.metadata["hierarchical_config"]["outer_tolerance"] == 1e-6


# =============================================================================
# TestAdaptiveRegularizationLayer
# =============================================================================
@pytest.mark.unit
class TestAdaptiveRegularizationLayer:
    """Tests for AdaptiveRegularizationLayer (Layer 3)."""

    def test_layer_name(self):
        """Layer has correct name."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AdaptiveRegularizationLayer,
        )

        layer = AdaptiveRegularizationLayer({})
        assert layer.name == "AdaptiveRegularizationLayer"

    def test_sets_regularization_metadata(self):
        """Layer sets regularization config in metadata."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AdaptiveRegularizationLayer,
            OptimizationState,
        )

        config = {
            "mode": "auto",
            "lambda": 0.5,
            "target_cv": 0.05,
            "target_contribution": 0.15,
            "max_cv": 0.25,
        }
        layer = AdaptiveRegularizationLayer(config)
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        assert result.metadata["regularization_enabled"] is True
        assert result.metadata["regularization_config"]["mode"] == "auto"
        assert result.metadata["regularization_config"]["lambda"] == 0.5
        assert result.metadata["regularization_config"]["target_cv"] == 0.05


# =============================================================================
# TestGradientMonitorLayer
# =============================================================================
@pytest.mark.unit
class TestGradientMonitorLayer:
    """Tests for GradientMonitorLayer (Layer 4)."""

    def test_layer_name(self):
        """Layer has correct name."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            GradientMonitorLayer,
        )

        layer = GradientMonitorLayer({})
        assert layer.name == "GradientMonitorLayer"

    def test_handles_none_gradient(self):
        """Layer handles None gradient gracefully."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            GradientMonitorLayer,
            OptimizationState,
        )

        layer = GradientMonitorLayer({})
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
            gradient=None,
        )

        result = layer.apply(state)

        # Should return state unchanged
        assert result.gradient is None

    def test_initializes_gradient_history(self):
        """Layer initializes gradient history in metadata."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            GradientMonitorLayer,
            OptimizationState,
        )

        layer = GradientMonitorLayer({})
        state = OptimizationState(
            params=np.array([1.0, 2.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
            gradient=np.array([0.1, 0.2]),
        )

        result = layer.apply(state)

        assert "gradient_history" in result.metadata
        assert "gradient_collapse_detected" in result.metadata
        assert len(result.metadata["gradient_history"]) == 1

    def test_detects_gradient_collapse(self):
        """Layer detects gradient collapse after consecutive triggers."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            GradientMonitorLayer,
            OptimizationState,
        )

        # Set up layer with low threshold and few required triggers
        layer = GradientMonitorLayer(
            {"ratio_threshold": 0.5, "consecutive_triggers": 2}
        )

        # Create state with very imbalanced gradients
        state = OptimizationState(
            params=np.array([1.0, 2.0, 3.0, 4.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
            gradient=np.array([0.001, 1.0, 1.0, 1.0]),  # First grad very small
        )

        # Apply twice to trigger detection
        result = layer.apply(state)
        result.gradient = np.array([0.001, 1.0, 1.0, 1.0])
        result = layer.apply(result)

        assert result.metadata["gradient_collapse_detected"] is True


# =============================================================================
# TestShearWeightingLayer
# =============================================================================
@pytest.mark.unit
class TestShearWeightingLayer:
    """Tests for ShearWeightingLayer (Layer 5)."""

    def test_layer_name(self):
        """Layer has correct name."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import ShearWeightingLayer

        layer = ShearWeightingLayer({})
        assert layer.name == "ShearWeightingLayer"

    def test_disabled_when_enable_false(self):
        """Layer does nothing when disabled."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            ShearWeightingLayer,
        )

        layer = ShearWeightingLayer({"enable": False})
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        assert result.metadata.get("shear_weighting_enabled", False) is False

    def test_sets_shear_weighting_metadata(self):
        """Layer sets shear weighting config in metadata."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            OptimizationState,
            ShearWeightingLayer,
        )

        config = {
            "enable": True,
            "min_weight": 0.2,
            "alpha": 2.0,
            "phi_0": 0.5,
            "normalize": True,
            "update_frequency": 5,
        }
        layer = ShearWeightingLayer(config)
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = layer.apply(state)

        assert result.metadata["shear_weighting_enabled"] is True
        assert result.metadata["shear_weighting_config"]["min_weight"] == 0.2
        assert result.metadata["shear_weighting_config"]["alpha"] == 2.0
        assert result.metadata["shear_weighting_config"]["phi_0"] == 0.5


# =============================================================================
# TestAntiDegeneracyChain
# =============================================================================
@pytest.mark.unit
class TestAntiDegeneracyChain:
    """Tests for AntiDegeneracyChain executor."""

    def test_empty_chain_returns_state_unchanged(self):
        """Empty chain returns state as-is."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
            OptimizationState,
        )

        chain = AntiDegeneracyChain([])
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = chain.execute(state)

        np.testing.assert_array_equal(result.params, state.params)

    def test_chain_executes_layers_in_order(self):
        """Chain executes layers sequentially."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
            HierarchicalLayer,
            OptimizationState,
            ShearWeightingLayer,
        )

        layer1 = HierarchicalLayer({})
        layer2 = ShearWeightingLayer({"enable": True})

        chain = AntiDegeneracyChain([layer1, layer2])
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        result = chain.execute(state)

        # Both layers should have applied
        assert result.metadata["hierarchical_enabled"] is True
        assert result.metadata["shear_weighting_enabled"] is True

    def test_chain_continues_on_layer_failure(self):
        """Chain continues if a layer raises an exception."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
            AntiDegeneracyLayer,
            HierarchicalLayer,
            OptimizationState,
        )

        class FailingLayer(AntiDegeneracyLayer):
            @property
            def name(self):
                return "FailingLayer"

            def apply(self, state):
                raise RuntimeError("Intentional failure")

        layer1 = FailingLayer()
        layer2 = HierarchicalLayer({})

        chain = AntiDegeneracyChain([layer1, layer2])
        state = OptimizationState(
            params=np.array([1.0]),
            residuals=np.array([0.1]),
            iteration=0,
            chi_squared=0.01,
        )

        # Should not raise, should continue to layer2
        result = chain.execute(state)

        # Layer2 should have been applied despite layer1 failure
        assert result.metadata["hierarchical_enabled"] is True

    def test_add_layer(self):
        """Layers can be added to chain."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
            HierarchicalLayer,
        )

        chain = AntiDegeneracyChain([])
        assert len(chain.layers) == 0

        chain.add_layer(HierarchicalLayer({}))
        assert len(chain.layers) == 1

    def test_layers_property(self):
        """Layers property returns layer list."""
        from homodyne.optimization.nlsq.anti_degeneracy_layer import (
            AntiDegeneracyChain,
            HierarchicalLayer,
            ShearWeightingLayer,
        )

        layers = [HierarchicalLayer({}), ShearWeightingLayer({"enable": True})]
        chain = AntiDegeneracyChain(layers)

        assert len(chain.layers) == 2
        assert chain.layers[0].name == "HierarchicalLayer"
        assert chain.layers[1].name == "ShearWeightingLayer"


# =============================================================================
# TestAllExports
# =============================================================================
@pytest.mark.unit
class TestAllExports:
    """Tests that __all__ exports are correct."""

    def test_all_exports(self):
        """All expected classes are exported."""
        from homodyne.optimization.nlsq import anti_degeneracy_layer

        expected_exports = [
            "OptimizationState",
            "AntiDegeneracyLayer",
            "FourierReparamLayer",
            "HierarchicalLayer",
            "AdaptiveRegularizationLayer",
            "GradientMonitorLayer",
            "ShearWeightingLayer",
            "AntiDegeneracyChain",
        ]

        for name in expected_exports:
            assert hasattr(anti_degeneracy_layer, name), f"Missing export: {name}"
            assert name in anti_degeneracy_layer.__all__, f"Not in __all__: {name}"
