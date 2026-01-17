"""Unit tests for AntiDegeneracyController with constant scaling mode.

Tests the auto-selection logic, constant mode initialization, and
parameter transformations.

Feature: 001-constant-scaling
"""

from __future__ import annotations

import numpy as np

from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)


class TestAntiDegeneracyConfigConstantScaling:
    """Tests for AntiDegeneracyConfig constant scaling fields."""

    def test_default_constant_scaling_threshold(self):
        """Test default constant_scaling_threshold is 3."""
        config = AntiDegeneracyConfig()
        assert config.constant_scaling_threshold == 3

    def test_constant_scaling_threshold_from_dict(self):
        """Test constant_scaling_threshold is parsed from dict."""
        config = AntiDegeneracyConfig.from_dict({"constant_scaling_threshold": 5})
        assert config.constant_scaling_threshold == 5

    def test_per_angle_mode_options(self):
        """Test all per_angle_mode options are accepted."""
        for mode in ["individual", "constant", "fourier", "auto"]:
            config = AntiDegeneracyConfig(per_angle_mode=mode)
            assert config.per_angle_mode == mode


class TestAntiDegeneracyControllerAutoSelection:
    """Tests for auto-selection logic (T024-T025).

    v2.17.0: Updated tests to reflect new auto mode behavior.
    Auto mode now only selects between "fourier" and "individual".
    "constant" mode must be explicitly set and is the new default.
    """

    def test_auto_selects_fourier_when_n_phi_gt_threshold(self):
        """v2.17.0: Test auto mode selects fourier when n_phi > fourier threshold."""
        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            fourier_auto_threshold=6,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,  # > 6
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "fourier"
        assert controller.use_constant is False
        assert controller.use_fourier is True

    def test_auto_selects_individual_when_n_phi_lte_threshold(self):
        """v2.17.0: Test auto mode selects individual when n_phi <= fourier threshold."""
        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            fourier_auto_threshold=6,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=5,  # <= 6
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 5),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "individual"
        assert controller.use_constant is False
        assert controller.use_fourier is False

    def test_auto_selects_individual_at_threshold_boundary(self):
        """v2.17.0: Test auto mode selects individual when n_phi == threshold exactly."""
        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            fourier_auto_threshold=6,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=6,  # == 6
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 6),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "individual"
        assert controller.use_constant is False
        assert controller.use_fourier is False

    def test_n_per_angle_params_constant_mode(self):
        """T025: Test n_per_angle_params is 2 in constant mode (explicit)."""
        # v2.17.0: Constant mode must be explicitly set
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",  # Explicit, not auto
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        assert controller.n_per_angle_params == 2


class TestAntiDegeneracyControllerMapperIntegration:
    """Tests for ParameterIndexMapper integration in constant mode (T026).

    v2.17.0: Updated to use explicit per_angle_mode="constant".
    """

    def test_mapper_uses_constant_mode(self):
        """T026: Test mapper is initialized with use_constant=True."""
        # v2.17.0: Use explicit constant mode
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

        assert controller.mapper is not None
        assert controller.mapper.use_constant is True
        assert controller.mapper.mode_name == "constant"

    def test_mapper_indices_constant_mode(self):
        """Test mapper provides correct indices in constant mode."""
        # v2.17.0: Use explicit constant mode
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

        # In constant mode: 2 per-angle params + 7 physical = 9 total
        assert controller.mapper.n_per_angle_total == 2
        assert controller.mapper.total_params == 9
        assert controller.mapper.get_group_indices() == [(0, 1), (1, 2)]
        assert controller.mapper.get_physical_indices() == [2, 3, 4, 5, 6, 7, 8]


class TestAntiDegeneracyControllerTransformMethods:
    """Tests for parameter transformation methods (T027-T028).

    v2.17.0: Updated to use explicit per_angle_mode="constant".
    """

    def setup_method(self):
        """Set up controller in constant mode."""
        # v2.17.0: Use explicit constant mode
        config = AntiDegeneracyConfig(
            per_angle_mode="constant",
        )
        self.controller = AntiDegeneracyController(
            config=config,
            n_phi=5,
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 5),
        )
        self.controller._initialize_components()

    def test_transform_to_constant_reduces_params(self):
        """T027: Test transform_params_to_constant reduces parameter count."""
        # 5 contrast + 5 offset + 7 physical = 17 params
        params = np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # contrast
                np.array([1.0, 1.1, 1.2, 1.3, 1.4]),  # offset
                np.zeros(7),  # physical
            ]
        )

        constant_params = self.controller.transform_params_to_constant(params)

        # Should be 2 per-angle + 7 physical = 9 params
        assert len(constant_params) == 9

    def test_transform_to_constant_computes_mean(self):
        """Test contrast and offset means are computed correctly."""
        params = np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),  # contrast, mean = 0.3
                np.array([1.0, 1.1, 1.2, 1.3, 1.4]),  # offset, mean = 1.2
                np.zeros(7),  # physical
            ]
        )

        constant_params = self.controller.transform_params_to_constant(params)

        assert np.isclose(constant_params[0], 0.3)  # contrast mean
        assert np.isclose(constant_params[1], 1.2)  # offset mean

    def test_transform_to_constant_preserves_physical(self):
        """Test physical parameters are preserved during transformation."""
        physical = np.array([1e4, 0.5, 100, 0.01, 0.8, 50, 45])
        params = np.concatenate(
            [
                np.ones(5),  # contrast
                np.ones(5),  # offset
                physical,
            ]
        )

        constant_params = self.controller.transform_params_to_constant(params)

        np.testing.assert_array_equal(constant_params[2:], physical)

    def test_transform_from_constant_expands_params(self):
        """T028: Test transform_params_from_constant expands to per-angle."""
        constant_params = np.array([0.3, 1.2, 1e4, 0.5, 100, 0.01, 0.8, 50, 45])

        expanded = self.controller.transform_params_from_constant(constant_params)

        # Should be 5 contrast + 5 offset + 7 physical = 17 params
        assert len(expanded) == 17

    def test_transform_from_constant_broadcasts_values(self):
        """Test constant values are broadcast to all angles."""
        constant_params = np.array([0.3, 1.2] + [0.0] * 7)

        expanded = self.controller.transform_params_from_constant(constant_params)

        # All contrast values should be 0.3
        np.testing.assert_array_equal(expanded[:5], np.full(5, 0.3))
        # All offset values should be 1.2
        np.testing.assert_array_equal(expanded[5:10], np.full(5, 1.2))

    def test_transform_roundtrip_preserves_mean(self):
        """Test transform roundtrip preserves mean values."""
        params = np.concatenate(
            [
                np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
                np.zeros(7),
            ]
        )

        constant = self.controller.transform_params_to_constant(params)
        expanded = self.controller.transform_params_from_constant(constant)

        # Means should be preserved
        assert np.isclose(np.mean(expanded[:5]), np.mean(params[:5]))
        assert np.isclose(np.mean(expanded[5:10]), np.mean(params[5:10]))


class TestAntiDegeneracyControllerDiagnostics:
    """Tests for diagnostics output with constant mode.

    v2.17.0: Updated to use explicit per_angle_mode="constant".
    """

    def test_diagnostics_includes_constant_mode_info(self):
        """Test diagnostics include use_constant and mode info."""
        # v2.17.0: Use explicit constant mode
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

        assert diag["use_constant"] is True
        assert diag["use_fourier"] is False
        assert diag["per_angle_mode"] == "constant"
        assert diag["n_per_angle_params"] == 2

    def test_mapper_diagnostics_in_constant_mode(self):
        """Test mapper diagnostics reflect constant mode."""
        # v2.17.0: Use explicit constant mode
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

        assert "mapper" in diag
        assert diag["mapper"]["mode_name"] == "constant"
        assert diag["mapper"]["use_constant"] is True
        assert diag["mapper"]["n_per_group"] == 1


class TestAntiDegeneracyControllerExplicitMode:
    """Tests for explicit per_angle_mode='constant' (User Story 2 prep)."""

    def test_explicit_constant_mode_works(self):
        """Test explicit per_angle_mode='constant' activates constant mode."""
        config = AntiDegeneracyConfig(per_angle_mode="constant")
        controller = AntiDegeneracyController(
            config=config,
            n_phi=2,  # Below default threshold
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 2),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "constant"
        assert controller.use_constant is True

    def test_explicit_individual_mode_works(self):
        """Test explicit per_angle_mode='individual' avoids constant mode."""
        config = AntiDegeneracyConfig(
            per_angle_mode="individual",
            constant_scaling_threshold=3,
        )
        controller = AntiDegeneracyController(
            config=config,
            n_phi=10,  # Above threshold, but explicit mode overrides
            n_physical=7,
            phi_angles=np.linspace(0, np.pi, 10),
        )
        controller._initialize_components()

        assert controller.per_angle_mode_actual == "individual"
        assert controller.use_constant is False
