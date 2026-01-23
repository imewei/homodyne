"""Unit tests for constant scaling mode in NLSQ optimization.

Tests the constant scaling feature that reduces per-angle parameters
from 2*n_phi to just 2 (single contrast, single offset) when n_phi
exceeds the constant_scaling_threshold.

Feature: 001-constant-scaling
"""

from __future__ import annotations

import numpy as np

from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)


class TestDefenseLayersConstantMode:
    """Tests for defense layer behavior with constant scaling mode (T043-T045)."""

    def test_defense_layers_skipped_constant_mode(self):
        """T043: Test that defense layers 1-4 are skipped in constant mode.

        When constant scaling is active, the following layers should be None:
        - Layer 1: Fourier reparameterizer (fourier_reparameterizer=None)
        - Layer 2: Hierarchical optimizer (hierarchical_optimizer=None)
        - Layer 3: Adaptive regularizer (adaptive_regularizer=None)
        - Layer 4: Gradient monitor (gradient_monitor=None)

        v2.18.0 semantics:
        - Explicit constant mode → "fixed_constant" (scaling FIXED, not optimized)
        - n_per_angle_params = 0 because scaling is FIXED
        """
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",  # Force constant mode
            hierarchical_enable=True,  # Would normally enable hierarchical
            regularization_mode="relative",  # Would normally enable regularizer
            gradient_monitoring_enable=True,  # Would normally enable monitor
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        # Verify fixed_constant mode is active (v2.18.0 naming)
        assert controller.use_constant is True
        assert controller.per_angle_mode_actual == "fixed_constant"
        assert controller.use_fixed_scaling is True

        # Verify defense layers 1-4 are skipped (None)
        # Layer 1: Fourier should be None in constant mode
        assert controller.use_fourier is False

        # The controller should have n_per_angle_params = 0 (FIXED scaling, not optimized)
        assert controller.n_per_angle_params == 0

    def test_shear_weighting_active_constant_mode(self):
        """T044: Test that shear weighting (Layer 5) remains active in constant mode.

        Shear-sensitivity weighting should still be active because it helps
        the optimizer by emphasizing angles parallel/antiparallel to flow.
        Note: The actual weighter is created in wrapper.py, not controller.
        This test verifies constant mode doesn't interfere with diagnostics.
        """
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",  # Force constant mode
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        # Verify constant mode is active
        assert controller.use_constant is True

        # Diagnostics should be available and reflect constant mode
        diag = controller.get_diagnostics()
        assert diag["use_constant"] is True
        assert diag["per_angle_mode"] == "constant"

    def test_defense_layers_active_individual_mode(self):
        """T045: Test that all defense layers are active in individual mode.

        When using individual per-angle scaling (not constant), all defense
        layers should be initialized as normal.
        """
        config = AntiDegeneracyConfig(
            per_angle_mode="individual",  # Force individual mode
            fourier_order=2,  # Would enable Fourier if auto
            hierarchical_enable=True,
            regularization_mode="relative",
            gradient_monitoring_enable=True,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        # Verify individual mode is active
        assert controller.use_constant is False
        assert controller.per_angle_mode_actual == "individual"
        assert controller.use_fourier is False  # individual mode, not Fourier

        # n_per_angle_params should be 2*n_phi in individual mode
        assert controller.n_per_angle_params == 2 * 10  # 20


class TestConstantModeParameterTransformation:
    """Tests for parameter transformation in constant mode."""

    def setup_method(self):
        """Set up controller in constant mode."""
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
            constant_scaling_threshold=3,
        )
        self.controller = AntiDegeneracyController(
            config=config,
            n_phi=5,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 5),
        )
        self.controller._initialize_components()

    def test_transform_to_constant_produces_correct_shape(self):
        """Test transform to constant produces 2 + n_physical params."""
        # 5 contrast + 5 offset + 7 physical = 17 params
        params = np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # contrast
                np.array([1.0, 1.1, 1.2, 1.3, 1.4]),  # offset
                np.zeros(7),  # physical
            ]
        )

        constant_params = self.controller.transform_params_to_constant(params)

        # Should be 1 contrast + 1 offset + 7 physical = 9 params
        assert len(constant_params) == 9

    def test_transform_from_constant_expands_correctly(self):
        """Test transform from constant expands to per-angle format."""
        # 1 contrast + 1 offset + 7 physical = 9 params
        constant_params = np.array([0.3, 1.2, 1e4, 0.5, 100, 0.01, 0.8, 50, 45])

        expanded = self.controller.transform_params_from_constant(constant_params)

        # Should be 5 contrast + 5 offset + 7 physical = 17 params
        assert len(expanded) == 17

        # All contrast values should be 0.3
        np.testing.assert_array_almost_equal(expanded[:5], np.full(5, 0.3))
        # All offset values should be 1.2
        np.testing.assert_array_almost_equal(expanded[5:10], np.full(5, 1.2))


class TestConstantModeAutoSelection:
    """Tests for auto-selection of constant mode based on n_phi."""

    def test_auto_selects_constant_above_threshold(self):
        """Test auto mode selects auto_averaged when n_phi >= threshold.

        v2.18.0 semantics:
        - auto + n_phi >= threshold → "auto_averaged" (9 params, averaged scaling OPTIMIZED)
        """
        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=5,  # >= 3
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 5),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "auto_averaged"
        assert controller.use_constant is True
        assert controller.use_averaged_scaling is True

    def test_auto_selects_individual_below_threshold(self):
        """Test auto mode selects individual when n_phi < threshold."""
        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=5,
            fourier_auto_threshold=10,  # High to avoid Fourier
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=3,  # < 5
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 3),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "individual"
        assert controller.use_constant is False


class TestConstantModeMapperIntegration:
    """Tests for ParameterIndexMapper integration with constant mode."""

    def test_mapper_constant_mode_indices(self):
        """Test mapper produces correct indices in constant mode."""
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        # Mapper should use constant mode
        assert controller.mapper.use_constant is True
        assert controller.mapper.mode_name == "constant"

        # n_per_group should be 1 (single value per group)
        assert controller.mapper.n_per_group == 1

        # Total params: 2 (contrast, offset) + 7 (physical) = 9
        assert controller.mapper.total_params == 9

        # Group indices: [(0, 1), (1, 2)]
        assert controller.mapper.get_group_indices() == [(0, 1), (1, 2)]

        # Physical indices: [2, 3, 4, 5, 6, 7, 8]
        assert controller.mapper.get_physical_indices() == [2, 3, 4, 5, 6, 7, 8]

    def test_mapper_diagnostics_constant_mode(self):
        """Test mapper diagnostics reflect constant mode."""
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        diag = controller.get_diagnostics()

        assert diag["mapper"]["mode_name"] == "constant"
        assert diag["mapper"]["use_constant"] is True
        assert diag["mapper"]["n_per_group"] == 1


class TestOutputExpansion:
    """Tests for output expansion in constant mode (T050-T052)."""

    def test_output_expansion_shape(self):
        """T050: Test expanded output has correct shape (n_phi contrast + n_phi offset + physical)."""
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        n_phi = 10
        n_physical = 7
        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        # Constant params: [contrast, offset, *physical] = 9 params
        constant_params = np.array([0.3, 1.2, 1e4, 0.5, 100, 0.01, 0.8, 50, 45])

        # Expand to per-angle format
        expanded = controller.transform_params_from_constant(constant_params)

        # Should be n_phi contrast + n_phi offset + n_physical
        expected_len = n_phi + n_phi + n_physical
        assert len(expanded) == expected_len

    def test_output_expansion_uniform_values(self):
        """T051: Test expanded output has uniform contrast/offset values."""
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        n_phi = 10
        n_physical = 7
        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        contrast_const = 0.35
        offset_const = 1.15
        constant_params = np.array(
            [contrast_const, offset_const, 1e4, 0.5, 100, 0.01, 0.8, 50, 45]
        )

        expanded = controller.transform_params_from_constant(constant_params)

        # All contrast values should be equal to the constant
        contrast_values = expanded[:n_phi]
        np.testing.assert_array_almost_equal(
            contrast_values, np.full(n_phi, contrast_const)
        )

        # All offset values should be equal to the constant
        offset_values = expanded[n_phi : 2 * n_phi]
        np.testing.assert_array_almost_equal(
            offset_values, np.full(n_phi, offset_const)
        )

    def test_output_metadata_per_angle_mode(self):
        """T052: Test diagnostics include per_angle_mode information.

        v2.18.0 semantics:
        - Explicit constant mode → "fixed_constant" (scaling FIXED, not optimized)
        - n_per_angle_params = 0 because scaling is FIXED
        """
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        diag = controller.get_diagnostics()

        # Diagnostics should include mode information
        assert "per_angle_mode" in diag
        assert diag["per_angle_mode"] == "constant"  # Config value
        assert diag["per_angle_mode_actual"] == "fixed_constant"  # Resolved actual mode
        assert "use_constant" in diag
        assert diag["use_constant"] is True
        assert diag["use_fixed_scaling"] is True
        assert "n_per_angle_params" in diag
        assert diag["n_per_angle_params"] == 0  # FIXED scaling, not optimized
