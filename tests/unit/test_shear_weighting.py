"""Tests for shear-sensitivity weighting module.

Anti-Degeneracy Defense System v2.9.1 - Layer 5.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from homodyne.optimization.nlsq.shear_weighting import (
    ShearSensitivityWeighting,
    ShearWeightingConfig,
    create_shear_weighting,
)


class TestShearWeightingConfig:
    """Tests for ShearWeightingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ShearWeightingConfig()
        assert config.enable is True
        assert config.min_weight == 0.3
        assert config.alpha == 1.0
        assert config.update_frequency == 1
        assert config.initial_phi0 is None
        assert config.normalize is True

    def test_from_config_empty(self):
        """Test creating config from empty dict."""
        config = ShearWeightingConfig.from_config({})
        assert config.enable is True
        assert config.min_weight == 0.3

    def test_from_config_with_values(self):
        """Test creating config from dict with values."""
        config_dict = {
            "shear_weighting": {
                "enable": False,
                "min_weight": 0.5,
                "alpha": 2.0,
                "update_frequency": 5,
                "initial_phi0": -10.0,
            }
        }
        config = ShearWeightingConfig.from_config(config_dict)
        assert config.enable is False
        assert config.min_weight == 0.5
        assert config.alpha == 2.0
        assert config.update_frequency == 5
        assert config.initial_phi0 == -10.0


class TestShearSensitivityWeighting:
    """Tests for ShearSensitivityWeighting class."""

    @pytest.fixture
    def phi_angles(self):
        """Standard phi angles for testing."""
        return np.linspace(-75, 75, 23)  # 23 angles like real data

    @pytest.fixture
    def weighter(self, phi_angles):
        """Create a weighter instance for testing."""
        config = ShearWeightingConfig(enable=True, min_weight=0.3, alpha=1.0)
        return ShearSensitivityWeighting(
            phi_angles=phi_angles,
            n_physical=7,
            phi0_index=6,
            config=config,
        )

    def test_initialization(self, weighter, phi_angles):
        """Test weighter initialization."""
        assert weighter.n_phi == len(phi_angles)
        assert weighter.n_physical == 7
        assert weighter.phi0_index == 6
        assert weighter._phi0_current == 0.0  # Default when initial_phi0 is None

    def test_weights_shape(self, weighter, phi_angles):
        """Test weights have correct shape."""
        weights = weighter.get_weights()
        assert weights.shape == (len(phi_angles),)

    def test_weights_normalized(self, weighter):
        """Test weights are normalized (mean = 1)."""
        weights = weighter.get_weights()
        np.testing.assert_allclose(np.mean(weights), 1.0, atol=1e-10)

    def test_weights_range(self, weighter):
        """Test weights are within expected range."""
        weights = weighter.get_weights()
        # After normalization, weights can be outside [min_weight, 1]
        # but should still be positive
        assert np.all(weights > 0)

    def test_weights_shear_sensitive(self, phi_angles):
        """Test that shear-sensitive angles get higher weights."""
        # phi0 = 0 means angles near 0 and 180 are shear-sensitive
        config = ShearWeightingConfig(
            enable=True, min_weight=0.3, alpha=1.0, normalize=False
        )
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles,
            n_physical=7,
            phi0_index=6,
            config=config,
        )
        weights = weighter.get_weights()

        # Find indices of angles near 0 (shear-sensitive) and 90 (not sensitive)
        idx_0 = np.argmin(np.abs(phi_angles - 0))
        idx_perpendicular = np.argmin(np.abs(np.abs(phi_angles) - 90))

        # Angle at 0 should have higher weight than angle at 90
        assert weights[idx_0] > weights[idx_perpendicular]

    def test_update_phi0(self, weighter):
        """Test phi0 update mechanism."""
        # Create parameter vector: [per_angle (46), physical (7)]
        n_per_angle = 2 * weighter.n_phi  # 46 for 23 angles
        params = np.zeros(n_per_angle + 7)
        # Set phi0 (last physical param) to 30 degrees
        params[-1] = 30.0  # phi0

        initial_weights = weighter.get_weights().copy()
        weighter.update_phi0(params, iteration=0)

        assert weighter._phi0_current == 30.0
        new_weights = weighter.get_weights()

        # Weights should have changed
        assert not np.allclose(initial_weights, new_weights)

    def test_update_phi0_threshold(self, weighter):
        """Test phi0 update respects threshold."""
        n_per_angle = 2 * weighter.n_phi
        params = np.zeros(n_per_angle + 7)
        params[-1] = 0.05  # Less than 0.1 degree threshold

        initial_phi0 = weighter._phi0_current
        weighter.update_phi0(params, iteration=0)

        # Should NOT update because change is below threshold
        assert weighter._phi0_current == initial_phi0

    def test_update_frequency(self, weighter):
        """Test update frequency control."""
        weighter.config.update_frequency = 2
        n_per_angle = 2 * weighter.n_phi
        params = np.zeros(n_per_angle + 7)
        params[-1] = 45.0  # phi0

        # Iteration 1 should NOT update (only updates on 0, 2, 4, ...)
        weighter.update_phi0(params, iteration=1)
        assert weighter._phi0_current == 0.0  # Unchanged

        # Iteration 2 should update
        weighter.update_phi0(params, iteration=2)
        assert weighter._phi0_current == 45.0

    def test_apply_weights_to_loss(self, weighter, phi_angles):
        """Test applying weights to loss computation."""
        n_data = len(phi_angles) * 100  # 100 tau points per angle
        residuals = jnp.ones(n_data) * 0.1
        phi_indices = jnp.array([i for i in range(len(phi_angles)) for _ in range(100)])

        weighted_loss = weighter.apply_weights_to_loss(residuals, phi_indices)

        # Loss should be positive
        assert float(weighted_loss) > 0

    def test_apply_weights_disabled(self, phi_angles):
        """Test weights are not applied when disabled."""
        config = ShearWeightingConfig(enable=False)
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles,
            n_physical=7,
            phi0_index=6,
            config=config,
        )

        n_data = 1000
        residuals = jnp.ones(n_data) * 0.1
        phi_indices = jnp.zeros(n_data, dtype=jnp.int32)

        weighted_loss = weighter.apply_weights_to_loss(residuals, phi_indices)

        # Without weighting, loss should be MSE * n_data
        expected = float(jnp.mean(residuals**2) * n_data)
        np.testing.assert_allclose(float(weighted_loss), expected, rtol=1e-6)

    def test_get_weights_jax(self, weighter):
        """Test getting weights as JAX array."""
        weights_jax = weighter.get_weights_jax()
        assert isinstance(weights_jax, jnp.ndarray)
        assert weights_jax.shape == (weighter.n_phi,)

    def test_diagnostics(self, weighter):
        """Test diagnostics output."""
        diag = weighter.get_diagnostics()
        assert diag["enabled"] is True
        assert diag["min_weight"] == 0.3
        assert diag["alpha"] == 1.0
        assert "current_phi0" in diag
        assert "update_count" in diag
        assert "weights_range" in diag
        assert "weights_mean" in diag
        assert "weights_std" in diag


class TestCreateShearWeighting:
    """Tests for create_shear_weighting factory function."""

    def test_create_when_disabled(self):
        """Test factory returns None when disabled."""
        config = {"shear_weighting": {"enable": False}}
        result = create_shear_weighting(
            phi_angles=np.linspace(-75, 75, 10),
            n_physical=7,
            config=config,
        )
        assert result is None

    def test_create_when_enabled(self):
        """Test factory creates weighter when enabled."""
        config = {"shear_weighting": {"enable": True}}
        result = create_shear_weighting(
            phi_angles=np.linspace(-75, 75, 10),
            n_physical=7,
            config=config,
        )
        assert isinstance(result, ShearSensitivityWeighting)

    def test_create_no_config(self):
        """Test factory returns None when no config provided."""
        result = create_shear_weighting(
            phi_angles=np.linspace(-75, 75, 10),
            n_physical=7,
            config=None,
        )
        assert result is None


class TestShearWeightingPhysics:
    """Tests verifying correct physics behavior of shear weighting."""

    def test_gradient_direction_enhancement(self):
        """Test that weighting enhances gradient in correct direction.

        The shear term gradient d(g1)/d(gamma_dot_t0) ~ cos(phi0 - phi).
        Weighting by |cos(phi0 - phi)| should emphasize angles where
        the gradient is largest (parallel/antiparallel to flow).
        """
        phi_angles = np.linspace(-90, 90, 37)  # Every 5 degrees
        phi0_true = 15.0  # True flow direction

        config = ShearWeightingConfig(
            enable=True, min_weight=0.1, alpha=1.0, initial_phi0=phi0_true
        )
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles, n_physical=7, phi0_index=6, config=config
        )

        weights = weighter.get_weights()

        # Calculate expected shear sensitivity: |cos(phi0 - phi)|
        phi0_rad = np.radians(phi0_true)
        phi_rad = np.radians(phi_angles)
        sensitivity = np.abs(np.cos(phi0_rad - phi_rad))

        # Weights should be positively correlated with sensitivity
        correlation = np.corrcoef(weights, sensitivity)[0, 1]
        assert correlation > 0.9, f"Correlation {correlation} too low"

    def test_symmetric_around_phi0(self):
        """Test weights are symmetric around phi0."""
        phi0 = 10.0
        # Create symmetric angles around phi0
        offsets = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60])
        phi_angles = phi0 + offsets

        config = ShearWeightingConfig(
            enable=True, min_weight=0.3, alpha=1.0, initial_phi0=phi0
        )
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles, n_physical=7, phi0_index=6, config=config
        )

        weights = weighter.get_weights()

        # Check symmetry: weight at phi0+offset should equal weight at phi0-offset
        n = len(offsets)
        for i in range(n // 2):
            np.testing.assert_allclose(
                weights[i], weights[n - 1 - i], rtol=1e-10,
                err_msg=f"Weights not symmetric for offset {offsets[i]}"
            )

    def test_extreme_angles_get_minimum_weight(self):
        """Test that angles perpendicular to flow get minimum weight."""
        phi0 = 0.0
        phi_angles = np.array([-90, -45, 0, 45, 90])

        config = ShearWeightingConfig(
            enable=True, min_weight=0.2, alpha=1.0, initial_phi0=phi0, normalize=False
        )
        weighter = ShearSensitivityWeighting(
            phi_angles=phi_angles, n_physical=7, phi0_index=6, config=config
        )

        weights = weighter.get_weights()

        # -90 and +90 should get minimum weight
        np.testing.assert_allclose(weights[0], 0.2, rtol=1e-10)  # -90
        np.testing.assert_allclose(weights[-1], 0.2, rtol=1e-10)  # +90

        # 0 should get maximum weight (1.0)
        np.testing.assert_allclose(weights[2], 1.0, rtol=1e-10)  # 0
