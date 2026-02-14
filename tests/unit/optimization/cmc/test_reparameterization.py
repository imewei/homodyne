# tests/unit/optimization/cmc/test_reparameterization.py
"""Tests for CMC reparameterization module."""

import math

import numpy as np
import pytest


class TestReparamConfig:
    """Tests for ReparamConfig dataclass."""

    def test_default_config_enables_all_transforms(self):
        """Default config enables D_ref and gamma_ref transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig()
        assert config.enable_d_ref is True
        assert config.enable_gamma_ref is True
        assert config.t_ref == 1.0

    def test_backward_compat_properties(self):
        """Backward-compatible properties map to new names."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=False)
        assert config.enable_d_total is True  # backward compat
        assert config.enable_log_gamma is False  # backward compat

    def test_config_can_disable_transforms(self):
        """Config can selectively disable transforms."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig(enable_d_ref=False, enable_gamma_ref=False)
        assert config.enable_d_ref is False
        assert config.enable_gamma_ref is False

    def test_config_with_t_ref(self):
        """Config stores custom t_ref."""
        from homodyne.optimization.cmc.reparameterization import ReparamConfig

        config = ReparamConfig(t_ref=3.16)
        assert config.t_ref == pytest.approx(3.16)


class TestComputeTRef:
    """Tests for compute_t_ref function."""

    def test_geometric_mean(self):
        """t_ref = sqrt(dt * t_max)."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        # C020 example: dt=0.1, t_max=100 → sqrt(10) ≈ 3.162
        t_ref = compute_t_ref(0.1, 100.0)
        assert t_ref == pytest.approx(math.sqrt(0.1 * 100.0))

    def test_dt_equals_t_max(self):
        """When dt == t_max, t_ref = dt."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        t_ref = compute_t_ref(5.0, 5.0)
        assert t_ref == pytest.approx(5.0)

    def test_invalid_dt_raises(self):
        """Invalid dt raises ValueError."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        with pytest.raises(ValueError, match="Invalid inputs"):
            compute_t_ref(0.0, 100.0)
        with pytest.raises(ValueError, match="Invalid inputs"):
            compute_t_ref(-1.0, 100.0)

    def test_invalid_t_max_raises(self):
        """Invalid t_max raises ValueError."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        with pytest.raises(ValueError, match="Invalid inputs"):
            compute_t_ref(0.1, 0.0)
        with pytest.raises(ValueError, match="Invalid inputs"):
            compute_t_ref(0.1, -50.0)

    def test_fallback_on_invalid_dt(self):
        """With fallback_value, invalid dt returns fallback instead of raising."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        result = compute_t_ref(0.0, 100.0, fallback_value=1.0)
        assert result == 1.0
        result = compute_t_ref(-1.0, 100.0, fallback_value=2.5)
        assert result == 2.5

    def test_fallback_on_invalid_t_max(self):
        """With fallback_value, invalid t_max returns fallback instead of raising."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        result = compute_t_ref(0.1, -50.0, fallback_value=1.0)
        assert result == 1.0

    def test_none_fallback_still_raises(self):
        """With fallback_value=None (default), invalid inputs still raise."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        with pytest.raises(ValueError, match="Invalid inputs"):
            compute_t_ref(0.0, 100.0, fallback_value=None)

    def test_valid_inputs_ignore_fallback(self):
        """With valid inputs, fallback_value is ignored."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        result = compute_t_ref(0.1, 100.0, fallback_value=999.0)
        assert result == pytest.approx(math.sqrt(0.1 * 100.0))
        assert result != 999.0

    def test_nan_fallback(self):
        """NaN inputs with fallback return the fallback value."""
        from homodyne.optimization.cmc.reparameterization import compute_t_ref

        result = compute_t_ref(float("nan"), 100.0, fallback_value=1.0)
        assert result == 1.0


class TestTransformToSamplingSpace:
    """Tests for transform_to_sampling_space function."""

    def test_d_ref_transform(self):
        """D0, D_offset → log_D_ref, D_offset_frac."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        t_ref = 3.16
        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=False, t_ref=t_ref)
        D0 = 20000.0
        alpha = -1.0
        D_offset = 1000.0
        params = {"D0": D0, "D_offset": D_offset, "alpha": alpha}

        result = transform_to_sampling_space(params, config)

        # D_ref = D0 * t_ref^alpha
        D_ref_expected = D0 * (t_ref**alpha)
        assert "log_D_ref" in result
        assert "D_offset_frac" in result
        assert "D0" not in result
        assert "D_offset" not in result
        assert result["log_D_ref"] == pytest.approx(np.log(D_ref_expected))
        assert result["D_offset_frac"] == pytest.approx(
            D_offset / (D_ref_expected + D_offset)
        )
        assert result["alpha"] == -1.0  # Unchanged

    def test_gamma_ref_transform(self):
        """gamma_dot_t0 → log_gamma_ref."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        t_ref = 3.16
        config = ReparamConfig(enable_d_ref=False, enable_gamma_ref=True, t_ref=t_ref)
        gamma_dot_t0 = 0.002
        beta = -0.3
        params = {"gamma_dot_t0": gamma_dot_t0, "beta": beta}

        result = transform_to_sampling_space(params, config)

        gamma_ref_expected = gamma_dot_t0 * (t_ref**beta)
        assert "log_gamma_ref" in result
        assert "gamma_dot_t0" not in result
        assert result["log_gamma_ref"] == pytest.approx(np.log(gamma_ref_expected))
        assert result["beta"] == -0.3  # Unchanged

    def test_both_transforms(self):
        """Both transforms applied together."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True, t_ref=3.16)
        params = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "alpha": -1.0,
            "gamma_dot_t0": 0.002,
            "beta": -0.3,
        }

        result = transform_to_sampling_space(params, config)

        assert "log_D_ref" in result
        assert "log_gamma_ref" in result


class TestTransformToPhysicsSpace:
    """Tests for transform_to_physics_space function."""

    def test_d_ref_inverse(self):
        """log_D_ref, D_offset_frac → D0, D_offset."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        t_ref = 3.16
        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=False, t_ref=t_ref)

        # Set up known values
        D0_true = 20000.0
        alpha_true = -1.0
        D_offset_true = 1000.0
        D_ref = D0_true * (t_ref**alpha_true)

        log_D_ref = np.log(D_ref)
        D_offset_frac = D_offset_true / (D_ref + D_offset_true)

        samples = {
            "log_D_ref": np.array([log_D_ref]),
            "D_offset_frac": np.array([D_offset_frac]),
            "alpha": np.array([alpha_true]),
        }

        result = transform_to_physics_space(samples, config)

        assert "D0" in result
        assert "D_offset" in result
        assert "log_D_ref" not in result
        np.testing.assert_allclose(result["D0"], [D0_true], rtol=1e-10)
        np.testing.assert_allclose(result["D_offset"], [D_offset_true], rtol=1e-6)

    def test_gamma_ref_inverse(self):
        """log_gamma_ref → gamma_dot_t0."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        t_ref = 3.16
        config = ReparamConfig(enable_d_ref=False, enable_gamma_ref=True, t_ref=t_ref)

        gamma_dot_t0_true = 0.002
        beta_true = -0.3
        gamma_ref = gamma_dot_t0_true * (t_ref**beta_true)

        samples = {
            "log_gamma_ref": np.array([np.log(gamma_ref)]),
            "beta": np.array([beta_true]),
        }

        result = transform_to_physics_space(samples, config)

        assert "gamma_dot_t0" in result
        assert "log_gamma_ref" not in result
        np.testing.assert_allclose(
            result["gamma_dot_t0"], [gamma_dot_t0_true], rtol=1e-10
        )

    def test_roundtrip_d_ref(self):
        """Transform to sampling and back preserves D0/D_offset."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
            transform_to_sampling_space,
        )

        t_ref = 3.16
        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True, t_ref=t_ref)
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
        assert recovered["D_offset"][0] == pytest.approx(original["D_offset"], rel=1e-6)
        assert recovered["gamma_dot_t0"][0] == pytest.approx(
            original["gamma_dot_t0"], rel=1e-10
        )

    def test_roundtrip_with_t_ref_1(self):
        """Roundtrip with t_ref=1.0 (backward compat)."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True, t_ref=1.0)
        original = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "alpha": -1.0,
            "beta": -0.3,
        }

        sampling = transform_to_sampling_space(original, config)
        sampling_arrays = {k: np.array([v]) for k, v in sampling.items()}
        recovered = transform_to_physics_space(sampling_arrays, config)

        assert recovered["D0"][0] == pytest.approx(original["D0"], rel=1e-10)
        assert recovered["D_offset"][0] == pytest.approx(original["D_offset"], rel=1e-6)
        assert recovered["gamma_dot_t0"][0] == pytest.approx(
            original["gamma_dot_t0"], rel=1e-10
        )

    def test_legacy_d_total_backward_compat(self):
        """Legacy D_total samples are still handled correctly."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=False)
        samples = {
            "D_total": np.array([21000.0, 22000.0]),
            "D_offset_frac": np.array([0.05, 0.1]),
            "alpha": np.array([-1.0, -1.1]),
        }

        result = transform_to_physics_space(samples, config)

        assert "D0" in result
        assert "D_offset" in result
        np.testing.assert_allclose(result["D0"], [19950.0, 19800.0])
        np.testing.assert_allclose(result["D_offset"], [1050.0, 2200.0])

    def test_legacy_log_gamma_dot_t0_backward_compat(self):
        """Legacy log_gamma_dot_t0 samples are still handled correctly."""
        from homodyne.optimization.cmc.reparameterization import (
            ReparamConfig,
            transform_to_physics_space,
        )

        config = ReparamConfig(enable_d_ref=False, enable_gamma_ref=True)
        samples = {
            "log_gamma_dot_t0": np.array([-6.0, -5.5]),
            "beta": np.array([-0.3, -0.25]),
        }

        result = transform_to_physics_space(samples, config)

        assert "gamma_dot_t0" in result
        assert "log_gamma_dot_t0" not in result
        np.testing.assert_allclose(result["gamma_dot_t0"], np.exp([-6.0, -5.5]))


class TestTransformNlsqToReparamSpace:
    """Tests for transform_nlsq_to_reparam_space function."""

    def test_basic_transform(self):
        """Transforms NLSQ values to reparameterized space."""
        from homodyne.optimization.cmc.reparameterization import (
            transform_nlsq_to_reparam_space,
        )

        t_ref = 3.16
        nlsq_values = {
            "D0": 20000.0,
            "alpha": -1.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "beta": -0.3,
        }

        reparam_vals, reparam_uncs = transform_nlsq_to_reparam_space(
            nlsq_values, None, t_ref
        )

        # D_ref = D0 * t_ref^alpha = 20000 * 3.16^(-1) ≈ 6329
        D_ref_expected = 20000.0 * (t_ref ** (-1.0))
        assert reparam_vals["log_D_ref"] == pytest.approx(math.log(D_ref_expected))
        assert "D_offset_frac" in reparam_vals
        assert "log_gamma_ref" in reparam_vals
        assert reparam_uncs == {}  # No uncertainties provided

    def test_uncertainty_propagation(self):
        """Delta-method uncertainty propagation works."""
        from homodyne.optimization.cmc.reparameterization import (
            transform_nlsq_to_reparam_space,
        )

        t_ref = 3.16
        nlsq_values = {
            "D0": 20000.0,
            "alpha": -1.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "beta": -0.3,
        }
        nlsq_uncertainties = {
            "D0": 2000.0,  # 10% relative
            "alpha": 0.1,
            "D_offset": 500.0,
            "gamma_dot_t0": 0.0005,  # 25% relative
            "beta": 0.05,
        }

        reparam_vals, reparam_uncs = transform_nlsq_to_reparam_space(
            nlsq_values, nlsq_uncertainties, t_ref
        )

        # log_D_ref uncertainty should be finite and positive
        assert reparam_uncs["log_D_ref"] > 0
        assert math.isfinite(reparam_uncs["log_D_ref"])

        # D_offset_frac uncertainty
        assert reparam_uncs["D_offset_frac"] > 0

        # log_gamma_ref uncertainty
        assert reparam_uncs["log_gamma_ref"] > 0

    def test_static_mode_no_shear(self):
        """Static mode produces no shear reparam values."""
        from homodyne.optimization.cmc.reparameterization import (
            transform_nlsq_to_reparam_space,
        )

        nlsq_values = {"D0": 10000.0, "alpha": -0.5, "D_offset": 500.0}

        reparam_vals, _ = transform_nlsq_to_reparam_space(nlsq_values, None, 3.16)

        assert "log_D_ref" in reparam_vals
        assert "D_offset_frac" in reparam_vals
        assert "log_gamma_ref" not in reparam_vals


class TestConfigWidthFactor:
    """Tests for nlsq_prior_width_factor default change."""

    def test_default_width_factor_is_2(self):
        """Default width factor should be 2.0 (not 3.0)."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig()
        assert config.nlsq_prior_width_factor == 2.0

    def test_from_dict_default_width_factor(self):
        """from_dict uses 2.0 default."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig.from_dict({})
        assert config.nlsq_prior_width_factor == 2.0

    def test_from_dict_explicit_width_factor(self):
        """from_dict respects explicit width factor."""
        from homodyne.optimization.cmc.config import CMCConfig

        config = CMCConfig.from_dict(
            {
                "validation": {"nlsq_prior_width_factor": 3.0},
            }
        )
        assert config.nlsq_prior_width_factor == 3.0
