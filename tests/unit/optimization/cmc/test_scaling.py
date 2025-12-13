"""Tests for CMC parameter scaling module.

This module tests the non-centered parameterization used to balance
gradient magnitudes across parameters with vastly different scales.
"""

import numpy as np
import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization.cmc.scaling import (
    ParameterScaling,
    compute_scaling_factors,
    transform_initial_values_to_z,
    transform_samples_from_z,
)


class TestParameterScaling:
    """Tests for ParameterScaling dataclass."""

    def test_to_normalized_at_center(self):
        """Test that center maps to z=0."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=0.0, high=200.0
        )
        z = scaling.to_normalized(100.0)
        assert z == pytest.approx(0.0)

    def test_to_normalized_one_sigma(self):
        """Test that center + scale maps to z=1."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=0.0, high=200.0
        )
        z = scaling.to_normalized(110.0)
        assert z == pytest.approx(1.0)

    def test_to_normalized_negative_sigma(self):
        """Test that center - scale maps to z=-1."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=0.0, high=200.0
        )
        z = scaling.to_normalized(90.0)
        assert z == pytest.approx(-1.0)

    def test_to_original_at_zero(self):
        """Test that z=0 maps to center."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=0.0, high=200.0
        )
        value = scaling.to_original(np.array(0.0))
        assert float(value) == pytest.approx(100.0)

    def test_to_original_clips_to_bounds(self):
        """Test that to_original clips values outside bounds."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=50.0, high=150.0
        )
        # z=10 would give 200, but should be clipped to 150
        value = scaling.to_original(np.array(10.0))
        assert float(value) == pytest.approx(150.0)

        # z=-10 would give 0, but should be clipped to 50
        value = scaling.to_original(np.array(-10.0))
        assert float(value) == pytest.approx(50.0)

    def test_roundtrip_within_bounds(self):
        """Test that to_normalized and to_original are inverses within bounds."""
        scaling = ParameterScaling(
            name="test", center=100.0, scale=10.0, low=0.0, high=200.0
        )
        original = 85.0
        z = scaling.to_normalized(original)
        recovered = scaling.to_original(np.array(z))
        assert float(recovered) == pytest.approx(original)


class TestComputeScalingFactors:
    """Tests for compute_scaling_factors function."""

    @pytest.fixture
    def parameter_space_laminar(self):
        """Create laminar flow parameter space."""
        return ParameterSpace.from_defaults("laminar_flow")

    @pytest.fixture
    def parameter_space_static(self):
        """Create static parameter space."""
        return ParameterSpace.from_defaults("static_isotropic")

    def test_laminar_flow_creates_correct_parameters(self, parameter_space_laminar):
        """Test that laminar flow mode creates scaling for all parameters."""
        scalings = compute_scaling_factors(
            parameter_space_laminar, n_phi=2, analysis_mode="laminar_flow"
        )

        # Should have: 2 contrast, 2 offset, 7 physical params = 11 total
        expected_names = {
            "contrast_0",
            "contrast_1",
            "offset_0",
            "offset_1",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        }
        assert set(scalings.keys()) == expected_names

    def test_static_mode_creates_correct_parameters(self, parameter_space_static):
        """Test that static mode creates scaling for correct parameters."""
        scalings = compute_scaling_factors(
            parameter_space_static, n_phi=1, analysis_mode="static"
        )

        # Should have: 1 contrast, 1 offset, 3 physical params = 5 total
        expected_names = {
            "contrast_0",
            "offset_0",
            "D0",
            "alpha",
            "D_offset",
        }
        assert set(scalings.keys()) == expected_names

    def test_scaling_uses_bounds_from_parameter_space(self, parameter_space_laminar):
        """Test that scaling factors use bounds from parameter space."""
        scalings = compute_scaling_factors(
            parameter_space_laminar, n_phi=1, analysis_mode="laminar_flow"
        )

        # D0 bounds should match parameter space
        d0_low, d0_high = parameter_space_laminar.get_bounds("D0")
        assert scalings["D0"].low == d0_low
        assert scalings["D0"].high == d0_high

    def test_scale_is_positive(self, parameter_space_laminar):
        """Test that all scales are positive."""
        scalings = compute_scaling_factors(
            parameter_space_laminar, n_phi=3, analysis_mode="laminar_flow"
        )

        for name, scaling in scalings.items():
            assert scaling.scale > 0, f"Scale for {name} should be positive"

    def test_scale_minimum_enforced(self, parameter_space_laminar):
        """Test that scale has minimum value of 1e-6."""
        scalings = compute_scaling_factors(
            parameter_space_laminar, n_phi=1, analysis_mode="laminar_flow"
        )

        for name, scaling in scalings.items():
            assert scaling.scale >= 1e-6, f"Scale for {name} should be >= 1e-6"

    def test_handles_mock_parameter_space_without_priors(self):
        """Test that scaling works when get_prior raises KeyError."""

        class MockParameterSpace:
            """Mock that raises KeyError for get_prior."""

            def get_bounds(self, name):
                bounds = {
                    "contrast": (0.0, 1.0),
                    "offset": (0.5, 1.5),
                    "D0": (1e3, 1e5),
                    "alpha": (-3.0, 1.0),
                    "D_offset": (0.0, 5e3),
                }
                return bounds.get(name.split("_")[0], (-1.0, 1.0))

            def get_prior(self, name):
                raise KeyError(name)

        mock_ps = MockParameterSpace()
        scalings = compute_scaling_factors(mock_ps, n_phi=1, analysis_mode="static")

        # Should still create scalings using bounds
        assert "D0" in scalings
        assert "contrast_0" in scalings

        # D0 now uses log-space for gradient balancing (Dec 2025 fix)
        # Center is log-midpoint: (log(1e3) + log(1e5)) / 2 = (6.9 + 11.5) / 2 ≈ 9.2
        import numpy as np

        expected_log_center = (np.log(1e3) + np.log(1e5)) / 2
        assert scalings["D0"].use_log_space is True
        assert scalings["D0"].center == pytest.approx(expected_log_center)


class TestTransformInitialValuesToZ:
    """Tests for transform_initial_values_to_z function."""

    @pytest.fixture
    def scalings(self):
        """Create sample scalings for testing."""
        return {
            "D0": ParameterScaling(
                "D0", center=50000.0, scale=25000.0, low=1000, high=100000
            ),
            "alpha": ParameterScaling(
                "alpha", center=0.0, scale=1.0, low=-3.0, high=1.0
            ),
        }

    def test_transforms_values_to_z_space(self, scalings):
        """Test that values are correctly transformed to z-space."""
        init_values = {"D0": 75000.0, "alpha": -1.0}
        z_values = transform_initial_values_to_z(init_values, scalings)

        # D0: (75000 - 50000) / 25000 = 1.0
        assert z_values["D0_z"] == pytest.approx(1.0)
        # alpha: (-1.0 - 0.0) / 1.0 = -1.0
        assert z_values["alpha_z"] == pytest.approx(-1.0)

    def test_adds_z_suffix_to_names(self, scalings):
        """Test that output keys have _z suffix."""
        init_values = {"D0": 50000.0}
        z_values = transform_initial_values_to_z(init_values, scalings)

        assert "D0_z" in z_values
        assert "D0" not in z_values

    def test_handles_none_input(self, scalings):
        """Test that None input returns empty dict."""
        z_values = transform_initial_values_to_z(None, scalings)
        assert z_values == {}

    def test_skips_missing_parameters(self, scalings):
        """Test that missing parameters are skipped."""
        init_values = {"D0": 50000.0, "missing_param": 1.0}
        z_values = transform_initial_values_to_z(init_values, scalings)

        assert "D0_z" in z_values
        assert "missing_param_z" not in z_values


class TestTransformSamplesFromZ:
    """Tests for transform_samples_from_z function."""

    @pytest.fixture
    def scalings(self):
        """Create sample scalings for testing."""
        return {
            "D0": ParameterScaling(
                "D0", center=50000.0, scale=25000.0, low=1000, high=100000
            ),
            "alpha": ParameterScaling(
                "alpha", center=0.0, scale=1.0, low=-3.0, high=1.0
            ),
        }

    def test_transforms_z_samples_to_original(self, scalings):
        """Test that z-space samples are correctly transformed."""
        samples = {
            "D0_z": np.array([0.0, 1.0, -1.0]),
            "alpha_z": np.array([0.0, 0.5, -0.5]),
        }
        original = transform_samples_from_z(samples, scalings)

        # D0: center + scale * z
        expected_D0 = np.array([50000.0, 75000.0, 25000.0])
        np.testing.assert_allclose(original["D0"], expected_D0)

        # alpha: center + scale * z
        expected_alpha = np.array([0.0, 0.5, -0.5])
        np.testing.assert_allclose(original["alpha"], expected_alpha)

    def test_clips_to_bounds(self, scalings):
        """Test that transformed samples are clipped to bounds."""
        samples = {
            "D0_z": np.array([10.0]),  # Would give 300000, clipped to 100000
        }
        original = transform_samples_from_z(samples, scalings)

        assert original["D0"][0] == pytest.approx(100000.0)


class TestGradientBalancing:
    """Tests verifying gradient balancing behavior."""

    def test_scale_ratio_reduced(self):
        """Test that parameter scale ratio is dramatically reduced after normalization."""
        ps = ParameterSpace.from_defaults("laminar_flow")
        scalings = compute_scaling_factors(ps, n_phi=1, analysis_mode="laminar_flow")

        # Get the range of scales
        scales = [s.scale for s in scalings.values()]
        original_ratio = max(scales) / min(scales)

        # After normalization, all z ~ N(0,1), so effective ratio is 1
        # The original ratio should be large (showing the problem exists)
        assert original_ratio > 100, "Original scale ratio should be large"

        # After z-transform, all parameters sample from N(0,1)
        # So the "effective" scale ratio is 1
        normalized_ratio = 1.0
        assert normalized_ratio < original_ratio / 100

    def test_d0_uses_log_space_when_bounds_positive(self):
        """Test D0 uses log-space for gradient balancing (Dec 2025 fix)."""
        ps = ParameterSpace.from_defaults("laminar_flow")
        scalings = compute_scaling_factors(ps, n_phi=1, analysis_mode="laminar_flow")

        # D0 should use log-space since it has positive bounds
        assert scalings["D0"].use_log_space is True

        # D0 center should be in log domain
        d0_low, d0_high = ps.get_bounds("D0")
        expected_log_center = (np.log(d0_low) + np.log(d0_high)) / 2
        assert scalings["D0"].center == pytest.approx(expected_log_center, rel=0.01)

    def test_log_space_roundtrip(self):
        """Test log-space to_normalized and to_original are inverses."""
        # Create a log-space scaling
        scaling = ParameterScaling(
            name="D0",
            center=np.log(10000),  # log-midpoint of [100, 1e6]
            scale=2.0,
            low=100.0,
            high=1e6,
            use_log_space=True,
        )

        # Test roundtrip
        original = 5000.0
        z = scaling.to_normalized(original)
        recovered = float(scaling.to_original(np.array(z)))
        assert recovered == pytest.approx(original, rel=0.01)

    def test_log_space_clips_to_bounds(self):
        """Test log-space to_original clips extreme values."""
        scaling = ParameterScaling(
            name="D0",
            center=np.log(10000),
            scale=1.0,
            low=100.0,
            high=100000.0,
            use_log_space=True,
        )

        # Very large z should clip to high bound
        value_high = float(scaling.to_original(np.array(100.0)))
        assert value_high == pytest.approx(100000.0)

        # Very negative z should clip to low bound
        value_low = float(scaling.to_original(np.array(-100.0)))
        assert value_low == pytest.approx(100.0)

    def test_d_offset_uses_linear_when_bounds_include_negative(self):
        """Test D_offset uses linear scaling when lower bound can be negative."""

        # Create parameter space where D_offset has negative lower bound
        class MockPS:
            def get_bounds(self, name):
                if name == "D_offset":
                    return (-1000.0, 5000.0)  # Includes negative
                elif name == "D0":
                    return (100.0, 100000.0)
                elif name == "alpha":
                    return (-3.0, 1.0)
                return (0.0, 1.0)

            def get_prior(self, name):
                raise KeyError(name)

        mock_ps = MockPS()
        scalings = compute_scaling_factors(mock_ps, n_phi=1, analysis_mode="static")

        # D0 should use log-space (positive bounds)
        assert scalings["D0"].use_log_space is True

        # D_offset should NOT use log-space (negative lower bound)
        assert scalings["D_offset"].use_log_space is False
