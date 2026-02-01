# tests/unit/optimization/cmc/test_reparameterization.py
"""Tests for CMC reparameterization module."""

import numpy as np
import pytest


class TestReparamConfig:
    """Tests for ReparamConfig dataclass."""

    def test_default_config_enables_all_transforms(self):
        """Default config enables D_total and log_gamma transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig()
        assert config.enable_d_total is True
        assert config.enable_log_gamma is True
        assert config.t_ref == 1.0

    def test_config_can_disable_transforms(self):
        """Config can selectively disable transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=False)
        assert config.enable_d_total is False
        assert config.enable_log_gamma is False


class TestTransformToSamplingSpace:
    """Tests for transform_to_sampling_space function."""

    def test_d_total_transform(self):
        """D0 + D_offset → D_total, D_offset_frac."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=False)
        params = {"D0": 20000.0, "D_offset": 1000.0, "alpha": -1.0}

        result = transform_to_sampling_space(params, config)

        assert "D_total" in result
        assert "D_offset_frac" in result
        assert "D0" not in result
        assert "D_offset" not in result
        assert result["D_total"] == pytest.approx(21000.0)
        assert result["D_offset_frac"] == pytest.approx(1000.0 / 21000.0)
        assert result["alpha"] == -1.0  # Unchanged

    def test_log_gamma_transform(self):
        """gamma_dot_t0 → log_gamma_dot_t0."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=True)
        params = {"gamma_dot_t0": 0.002, "beta": -0.3}

        result = transform_to_sampling_space(params, config)

        assert "log_gamma_dot_t0" in result
        assert "gamma_dot_t0" not in result
        assert result["log_gamma_dot_t0"] == pytest.approx(np.log(0.002))
        assert result["beta"] == -0.3  # Unchanged

    def test_both_transforms(self):
        """Both transforms applied together."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=True)
        params = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "beta": -0.3,
        }

        result = transform_to_sampling_space(params, config)

        assert "D_total" in result
        assert "log_gamma_dot_t0" in result


class TestTransformToPhysicsSpace:
    """Tests for transform_to_physics_space function."""

    def test_d_total_inverse(self):
        """D_total, D_offset_frac → D0, D_offset."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=False)
        samples = {
            "D_total": np.array([21000.0, 22000.0]),
            "D_offset_frac": np.array([0.05, 0.1]),
            "alpha": np.array([-1.0, -1.1]),
        }

        result = transform_to_physics_space(samples, config)

        assert "D0" in result
        assert "D_offset" in result
        assert "D_total" not in result
        np.testing.assert_allclose(result["D0"], [19950.0, 19800.0])
        np.testing.assert_allclose(result["D_offset"], [1050.0, 2200.0])

    def test_log_gamma_inverse(self):
        """log_gamma_dot_t0 → gamma_dot_t0."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_total=False, enable_log_gamma=True)
        samples = {
            "log_gamma_dot_t0": np.array([-6.0, -5.5]),
            "beta": np.array([-0.3, -0.25]),
        }

        result = transform_to_physics_space(samples, config)

        assert "gamma_dot_t0" in result
        assert "log_gamma_dot_t0" not in result
        np.testing.assert_allclose(result["gamma_dot_t0"], np.exp([-6.0, -5.5]))

    def test_roundtrip(self):
        """Transform to sampling and back preserves values."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_total=True, enable_log_gamma=True)
        original = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "alpha": -1.0,
            "beta": -0.3,
        }

        # Forward transform
        sampling = transform_to_sampling_space(original, config)

        # Convert to arrays for inverse
        sampling_arrays = {k: np.array([v]) for k, v in sampling.items()}

        # Inverse transform
        recovered = transform_to_physics_space(sampling_arrays, config)

        # Check roundtrip
        assert recovered["D0"][0] == pytest.approx(original["D0"], rel=1e-10)
        assert recovered["D_offset"][0] == pytest.approx(original["D_offset"], rel=1e-10)
        assert recovered["gamma_dot_t0"][0] == pytest.approx(
            original["gamma_dot_t0"], rel=1e-10
        )
