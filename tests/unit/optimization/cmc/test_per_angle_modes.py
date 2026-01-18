"""Tests for CMC per-angle mode support (v2.18.0+).

This module tests the anti-degeneracy per-angle mode implementation in CMC,
which allows choosing between 'constant' and 'individual' modes to match
NLSQ's anti-degeneracy system.

Constant mode: Fixed per-angle contrast/offset from quantile estimation (NOT sampled)
Individual mode: Sampled per-angle contrast/offset (legacy)
"""

import numpy as np
import pytest

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.model import (
    get_model_param_count,
    get_xpcs_model,
    xpcs_model_constant,
    xpcs_model_scaled,
)
from homodyne.optimization.cmc.priors import (
    build_init_values_dict,
    get_param_names_in_order,
)


class TestCMCConfigPerAngleMode:
    """Tests for per_angle_mode in CMCConfig."""

    def test_default_mode_is_auto(self):
        """Verify default per_angle_mode is 'auto'."""
        config = CMCConfig()
        assert config.per_angle_mode == "auto"
        assert config.constant_scaling_threshold == 3

    def test_from_dict_with_per_angle_mode(self):
        """Test parsing per_angle_mode from config dict."""
        config_dict = {
            "per_angle_mode": "constant",
            "constant_scaling_threshold": 5,
        }
        config = CMCConfig.from_dict(config_dict)
        assert config.per_angle_mode == "constant"
        assert config.constant_scaling_threshold == 5

    def test_get_effective_mode_auto_constant(self):
        """Test auto mode selects constant when n_phi >= threshold."""
        config = CMCConfig(per_angle_mode="auto", constant_scaling_threshold=3)

        # n_phi >= 3 should use constant
        assert config.get_effective_per_angle_mode(3) == "constant"
        assert config.get_effective_per_angle_mode(5) == "constant"
        assert config.get_effective_per_angle_mode(23) == "constant"

    def test_get_effective_mode_auto_individual(self):
        """Test auto mode selects individual when n_phi < threshold."""
        config = CMCConfig(per_angle_mode="auto", constant_scaling_threshold=3)

        # n_phi < 3 should use individual
        assert config.get_effective_per_angle_mode(1) == "individual"
        assert config.get_effective_per_angle_mode(2) == "individual"

    def test_get_effective_mode_explicit(self):
        """Test explicit mode selection overrides auto logic."""
        config = CMCConfig(per_angle_mode="constant")
        assert config.get_effective_per_angle_mode(1) == "constant"

        config = CMCConfig(per_angle_mode="individual")
        assert config.get_effective_per_angle_mode(100) == "individual"

    def test_to_dict_includes_per_angle_mode(self):
        """Test to_dict includes per_angle_mode fields."""
        config = CMCConfig(per_angle_mode="constant", constant_scaling_threshold=5)
        d = config.to_dict()
        assert d["per_angle_mode"] == "constant"
        assert d["constant_scaling_threshold"] == 5

    def test_validation_rejects_invalid_mode(self):
        """Test validation rejects invalid per_angle_mode values."""
        config = CMCConfig(per_angle_mode="invalid_mode")
        errors = config.validate()
        assert any("per_angle_mode" in e for e in errors)

    def test_validation_rejects_fourier_mode(self):
        """Test validation rejects fourier mode (removed in v2.18.0)."""
        config = CMCConfig(per_angle_mode="fourier")
        errors = config.validate()
        assert any("per_angle_mode" in e for e in errors)


class TestModelSelection:
    """Tests for model selection based on per_angle_mode."""

    def test_get_xpcs_model_individual(self):
        """Test get_xpcs_model returns scaled model for individual mode."""
        model = get_xpcs_model("individual")
        assert model == xpcs_model_scaled

    def test_get_xpcs_model_constant(self):
        """Test get_xpcs_model returns constant model for constant mode."""
        model = get_xpcs_model("constant")
        assert model == xpcs_model_constant

    def test_get_xpcs_model_default_is_individual(self):
        """Test get_xpcs_model defaults to individual mode."""
        model = get_xpcs_model()
        assert model == xpcs_model_scaled


class TestParamCountByMode:
    """Tests for parameter count calculation with different modes."""

    def test_individual_mode_param_count_static(self):
        """Test individual mode parameter count for static analysis."""
        # static: 2*n_phi + 3 physical + 1 sigma
        assert get_model_param_count(1, "static", "individual") == 6  # 2 + 3 + 1
        assert get_model_param_count(3, "static", "individual") == 10  # 6 + 3 + 1
        assert get_model_param_count(23, "static", "individual") == 50  # 46 + 3 + 1

    def test_individual_mode_param_count_laminar(self):
        """Test individual mode parameter count for laminar_flow analysis."""
        # laminar: 2*n_phi + 7 physical + 1 sigma
        assert get_model_param_count(1, "laminar_flow", "individual") == 10  # 2 + 7 + 1
        assert get_model_param_count(3, "laminar_flow", "individual") == 14  # 6 + 7 + 1
        assert get_model_param_count(23, "laminar_flow", "individual") == 54  # 46 + 7 + 1

    def test_constant_mode_param_count_static(self):
        """Test constant mode parameter count for static analysis.

        Constant mode: 0 per-angle params (fixed from quantiles) + physical + sigma
        Static: 0 + 3 + 1 = 4
        """
        assert get_model_param_count(1, "static", "constant") == 4
        assert get_model_param_count(3, "static", "constant") == 4
        assert get_model_param_count(23, "static", "constant") == 4

    def test_constant_mode_param_count_laminar(self):
        """Test constant mode parameter count for laminar_flow analysis.

        Constant mode: 0 per-angle params (fixed from quantiles) + physical + sigma
        Laminar: 0 + 7 + 1 = 8
        """
        assert get_model_param_count(1, "laminar_flow", "constant") == 8
        assert get_model_param_count(3, "laminar_flow", "constant") == 8
        assert get_model_param_count(23, "laminar_flow", "constant") == 8

    def test_constant_mode_reduction(self):
        """Test constant mode reduces parameter count significantly."""
        # For n_phi=23 laminar_flow:
        individual_count = get_model_param_count(23, "laminar_flow", "individual")
        constant_count = get_model_param_count(23, "laminar_flow", "constant")

        # Should be 54 vs 8 = 85% reduction
        reduction = (individual_count - constant_count) / individual_count
        assert reduction > 0.85


class TestParamNamesInOrder:
    """Tests for parameter name ordering with different modes."""

    def test_individual_mode_names_static(self):
        """Test parameter names for individual mode static analysis."""
        names = get_param_names_in_order(3, "static", "individual")
        expected = [
            "contrast_0", "contrast_1", "contrast_2",
            "offset_0", "offset_1", "offset_2",
            "D0", "alpha", "D_offset",
        ]
        assert names == expected

    def test_individual_mode_names_laminar(self):
        """Test parameter names for individual mode laminar_flow analysis."""
        names = get_param_names_in_order(2, "laminar_flow", "individual")
        expected = [
            "contrast_0", "contrast_1",
            "offset_0", "offset_1",
            "D0", "alpha", "D_offset",
            "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0",
        ]
        assert names == expected

    def test_constant_mode_names_static(self):
        """Test parameter names for constant mode static analysis.

        Constant mode: NO contrast/offset params (fixed from quantiles).
        Only physical params are sampled.
        """
        names = get_param_names_in_order(10, "static", "constant")
        expected = [
            "D0", "alpha", "D_offset",
        ]
        assert names == expected

    def test_constant_mode_names_laminar(self):
        """Test parameter names for constant mode laminar_flow analysis.

        Constant mode: NO contrast/offset params (fixed from quantiles).
        Only physical params are sampled.
        """
        names = get_param_names_in_order(23, "laminar_flow", "constant")
        expected = [
            "D0", "alpha", "D_offset",
            "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0",
        ]
        assert names == expected


class TestBuildInitValuesDict:
    """Tests for build_init_values_dict with different modes."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a mock parameter space for testing."""
        from unittest.mock import MagicMock
        ps = MagicMock()
        ps.get_bounds.side_effect = lambda name: {
            "contrast": (0.0, 1.0),
            "offset": (0.5, 1.5),
            "D0": (1e8, 1e12),
            "alpha": (-2.0, 0.0),
            "D_offset": (0.0, 1e10),
            "gamma_dot_t0": (0.0, 1e5),
            "beta": (-2.0, 2.0),
            "gamma_dot_t_offset": (0.0, 1e4),
            "phi0": (-180.0, 180.0),
        }.get(name.split("_")[0] if name.startswith(("contrast_", "offset_")) else name, (0.0, 1.0))
        return ps

    def test_individual_mode_creates_per_angle_params(self, mock_parameter_space):
        """Test individual mode creates per-angle contrast/offset params."""
        init_values = build_init_values_dict(
            n_phi=3,
            analysis_mode="static",
            initial_values={"contrast": 0.5, "offset": 1.0},
            parameter_space=mock_parameter_space,
            per_angle_mode="individual",
        )

        # Should have contrast_0, contrast_1, contrast_2, offset_0, offset_1, offset_2
        assert "contrast_0" in init_values
        assert "contrast_1" in init_values
        assert "contrast_2" in init_values
        assert "offset_0" in init_values
        assert "offset_1" in init_values
        assert "offset_2" in init_values
        # Should also have physical params
        assert "D0" in init_values
        assert "alpha" in init_values
        assert "D_offset" in init_values

    def test_constant_mode_no_contrast_offset_params(self, mock_parameter_space):
        """Test constant mode does NOT create contrast/offset params.

        In constant mode, contrast/offset are FIXED from quantile estimation,
        not sampled. So init_values should only contain physical params.
        """
        init_values = build_init_values_dict(
            n_phi=3,
            analysis_mode="static",
            initial_values={"contrast": 0.5, "offset": 1.0},
            parameter_space=mock_parameter_space,
            per_angle_mode="constant",
        )

        # Should NOT have any contrast/offset params
        assert "contrast_avg" not in init_values
        assert "offset_avg" not in init_values
        assert "contrast_0" not in init_values
        assert "offset_0" not in init_values

        # Should ONLY have physical params
        assert "D0" in init_values
        assert "alpha" in init_values
        assert "D_offset" in init_values
        assert len(init_values) == 3  # D0, alpha, D_offset


class TestScalingUtilsShared:
    """Tests for shared scaling utilities."""

    def test_estimate_contrast_offset_from_quantiles(self):
        """Test the shared quantile estimation function."""
        from homodyne.core.scaling_utils import estimate_contrast_offset_from_quantiles

        # Create synthetic C2 data with known contrast/offset
        # C2 = contrast * g1^2 + offset
        # At small lags, g1 ≈ 1, so C2 ≈ contrast + offset
        # At large lags, g1 ≈ 0, so C2 ≈ offset
        n_points = 1000
        delta_t = np.linspace(0, 10, n_points)

        true_contrast = 0.4
        true_offset = 0.95

        # Simulate g1 decay and C2
        np.random.seed(42)
        g1_sq = np.exp(-delta_t)  # Exponential decay
        c2_data = true_contrast * g1_sq + true_offset + np.random.normal(0, 0.02, n_points)

        contrast_est, offset_est = estimate_contrast_offset_from_quantiles(
            c2_data, delta_t,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Should be reasonably close to true values
        assert contrast_est == pytest.approx(true_contrast, abs=0.1)
        assert offset_est == pytest.approx(true_offset, abs=0.1)

    def test_estimate_per_angle_scaling(self):
        """Test the per-angle scaling estimation function."""
        from homodyne.core.scaling_utils import estimate_per_angle_scaling

        # Create synthetic data for 3 angles
        n_points_per_angle = 500
        n_phi = 3
        np.random.seed(42)

        c2_list = []
        t1_list = []
        t2_list = []
        phi_indices_list = []

        for phi_idx in range(n_phi):
            # Different contrast/offset per angle
            true_contrast = 0.3 + phi_idx * 0.1  # 0.3, 0.4, 0.5
            true_offset = 0.9 + phi_idx * 0.05  # 0.9, 0.95, 1.0

            t1 = np.random.uniform(0, 100, n_points_per_angle)
            t2 = np.random.uniform(0, 100, n_points_per_angle)
            delta_t = np.abs(t1 - t2)
            g1_sq = np.exp(-delta_t / 10)
            c2 = true_contrast * g1_sq + true_offset + np.random.normal(0, 0.02, n_points_per_angle)

            c2_list.append(c2)
            t1_list.append(t1)
            t2_list.append(t2)
            phi_indices_list.append(np.full(n_points_per_angle, phi_idx, dtype=np.int32))

        c2_data = np.concatenate(c2_list)
        t1 = np.concatenate(t1_list)
        t2 = np.concatenate(t2_list)
        phi_indices = np.concatenate(phi_indices_list)

        estimates = estimate_per_angle_scaling(
            c2_data=c2_data,
            t1=t1,
            t2=t2,
            phi_indices=phi_indices,
            n_phi=n_phi,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Should have estimates for all angles
        for i in range(n_phi):
            assert f"contrast_{i}" in estimates
            assert f"offset_{i}" in estimates

        # Should be reasonably close to true values
        assert estimates["contrast_0"] == pytest.approx(0.3, abs=0.1)
        assert estimates["contrast_1"] == pytest.approx(0.4, abs=0.1)
        assert estimates["contrast_2"] == pytest.approx(0.5, abs=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
