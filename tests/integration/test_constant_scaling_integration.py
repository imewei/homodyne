"""Integration tests for constant scaling mode in NLSQ optimization.

Tests the end-to-end functionality of constant scaling when optimizing
laminar flow data with many phi angles. Verifies that using a single
contrast/offset pair prevents parameter absorption and allows proper
recovery of physical parameters.

Feature: 001-constant-scaling
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from homodyne.optimization.nlsq.anti_degeneracy_controller import (
    AntiDegeneracyConfig,
    AntiDegeneracyController,
)


@dataclass
class MockOptimizationResult:
    """Mock optimization result for testing."""

    params: np.ndarray
    covariance: np.ndarray | None = None
    chi_squared: float = 1.0
    success: bool = True
    info: dict | None = None


class TestConstantScalingAutoSelection:
    """T053: Test constant scaling auto-selection with realistic setup."""

    @pytest.mark.integration
    def test_auto_selects_constant_for_23_phi(self):
        """Verify auto mode selects constant for 23 phi angles (C020 dataset).

        When n_phi=23 and constant_scaling_threshold=3 (default),
        auto mode should select constant scaling mode.
        """
        n_phi = 23  # C020 dataset has 23 angles
        n_physical = 7  # Laminar flow has 7 physical params

        config = AntiDegeneracyConfig(
            per_angle_mode="auto",
            constant_scaling_threshold=3,  # Default threshold
        )

        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        # Should auto-select constant mode
        assert controller.per_angle_mode_actual == "constant"
        assert controller.use_constant is True
        assert controller.use_fourier is False

        # Should have 2 per-angle params (1 contrast + 1 offset)
        assert controller.n_per_angle_params == 2

        # Total params: 2 + 7 = 9 (not 46 + 7 = 53)
        assert controller.mapper.total_params == 9


class TestConstantScalingParameterReduction:
    """T054-T055: Test parameter count reduction with constant scaling."""

    @pytest.mark.integration
    def test_params_reduced_from_53_to_9(self):
        """Verify 23-angle laminar flow uses 9 params (not 53).

        Without constant scaling: 23*2 per-angle + 7 physical = 53 params
        With constant scaling: 2 per-angle + 7 physical = 9 params
        """
        n_phi = 23
        n_physical = 7

        # Individual mode: 46 + 7 = 53 params
        config_individual = AntiDegeneracyConfig(per_angle_mode="individual")
        controller_individual = AntiDegeneracyController(
            config=config_individual,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller_individual._initialize_components()
        assert controller_individual.mapper.total_params == 2 * n_phi + n_physical  # 53

        # Constant mode: 2 + 7 = 9 params
        config_constant = AntiDegeneracyConfig(per_angle_mode="constant")
        controller_constant = AntiDegeneracyController(
            config=config_constant,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller_constant._initialize_components()
        assert controller_constant.mapper.total_params == 2 + n_physical  # 9

    @pytest.mark.integration
    def test_initial_params_transformation(self):
        """Test initial parameter transformation from per-angle to constant.

        Given per-angle initial guesses (uniform or varied),
        When transforming to constant mode,
        Then the mean is computed for contrast/offset groups.
        """
        n_phi = 23
        n_physical = 7

        config = AntiDegeneracyConfig(per_angle_mode="constant")
        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        # Create per-angle initial params with some variation
        contrast_init = 0.9 + 0.1 * np.random.randn(n_phi)
        offset_init = 1.0 + 0.05 * np.random.randn(n_phi)
        physical_init = np.array([1e4, 0.5, 100, 0.002, 0.8, 50, 45])

        per_angle_params = np.concatenate([contrast_init, offset_init, physical_init])

        # Transform to constant
        constant_params = controller.transform_params_to_constant(per_angle_params)

        # Should have 9 params
        assert len(constant_params) == 9

        # First two should be means of contrast and offset
        np.testing.assert_almost_equal(constant_params[0], np.mean(contrast_init))
        np.testing.assert_almost_equal(constant_params[1], np.mean(offset_init))

        # Physical params should be unchanged
        np.testing.assert_array_equal(constant_params[2:], physical_init)


class TestConstantScalingOutputCompatibility:
    """T056: Test output NPZ compatibility with constant scaling."""

    @pytest.mark.integration
    def test_output_expansion_to_per_angle_format(self):
        """Verify constant mode results expand to per-angle format.

        The output should have shape (n_phi, 2) for per_angle_scaling_solver,
        with uniform values across all angles.
        """
        n_phi = 23
        n_physical = 7

        config = AntiDegeneracyConfig(per_angle_mode="constant")
        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        # Simulated optimization result in constant mode
        contrast_opt = 0.92
        offset_opt = 1.03
        physical_opt = np.array([1.2e4, 0.48, 95, 0.0021, 0.79, 48, 44])
        constant_result = np.concatenate([[contrast_opt], [offset_opt], physical_opt])

        # Expand to per-angle format
        expanded = controller.transform_params_from_constant(constant_result)

        # Should have 2*n_phi + n_physical params
        assert len(expanded) == 2 * n_phi + n_physical

        # All contrast values should be uniform
        contrast_expanded = expanded[:n_phi]
        assert np.allclose(contrast_expanded, contrast_opt)

        # All offset values should be uniform
        offset_expanded = expanded[n_phi : 2 * n_phi]
        assert np.allclose(offset_expanded, offset_opt)

        # Physical params unchanged
        np.testing.assert_array_equal(expanded[2 * n_phi :], physical_opt)

        # Can reshape for per_angle_scaling_solver format
        per_angle_scaling = np.column_stack(
            [expanded[:n_phi], expanded[n_phi : 2 * n_phi]]
        )
        assert per_angle_scaling.shape == (n_phi, 2)

    @pytest.mark.integration
    def test_diagnostics_include_constant_mode_info(self):
        """Verify diagnostics include constant mode information."""
        n_phi = 23
        n_physical = 7

        config = AntiDegeneracyConfig(per_angle_mode="constant")
        controller = AntiDegeneracyController(
            config=config,
            n_phi=n_phi,
            n_physical=n_physical,
            phi_angles=np.linspace(0, np.pi, n_phi),
        )
        controller._initialize_components()

        diag = controller.get_diagnostics()

        # Should include constant mode info
        assert diag["use_constant"] is True
        assert diag["use_fourier"] is False
        assert diag["per_angle_mode"] == "constant"
        assert diag["n_per_angle_params"] == 2

        # Mapper diagnostics should reflect constant mode
        assert diag["mapper"]["mode_name"] == "constant"
        assert diag["mapper"]["use_constant"] is True
        assert diag["mapper"]["n_per_group"] == 1
