"""Tests for CMC prior builders module."""

import math

import pytest

pytest.importorskip("arviz", reason="ArviZ required for CMC unit tests")

from homodyne.optimization.cmc.priors import (  # noqa: E402
    build_init_values_dict,
    build_prior,
    get_init_value,
    get_param_names_in_order,
    validate_initial_value_bounds,
)


class TestGetParamNamesInOrder:
    """Tests for get_param_names_in_order function."""

    def test_static_mode_single_phi(self):
        """Test parameter names for static mode with 1 phi angle."""
        names = get_param_names_in_order(n_phi=1, analysis_mode="static")

        expected = [
            "contrast_0",
            "offset_0",
            "D0",
            "alpha",
            "D_offset",
        ]
        assert names == expected

    def test_static_mode_multi_phi(self):
        """Test parameter names for static mode with 3 phi angles."""
        names = get_param_names_in_order(n_phi=3, analysis_mode="static")

        expected = [
            "contrast_0",
            "contrast_1",
            "contrast_2",
            "offset_0",
            "offset_1",
            "offset_2",
            "D0",
            "alpha",
            "D_offset",
        ]
        assert names == expected

    def test_laminar_flow_mode(self):
        """Test parameter names for laminar_flow mode."""
        names = get_param_names_in_order(n_phi=2, analysis_mode="laminar_flow")

        expected = [
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
        ]
        assert names == expected

    def test_ordering_contrast_before_offset(self):
        """Test that contrast parameters come before offset parameters."""
        names = get_param_names_in_order(n_phi=3, analysis_mode="static")

        contrast_indices = [i for i, n in enumerate(names) if n.startswith("contrast_")]
        offset_indices = [i for i, n in enumerate(names) if n.startswith("offset_")]

        # All contrast indices should be less than all offset indices
        assert max(contrast_indices) < min(offset_indices)

    def test_ordering_scaling_before_physical(self):
        """Test that scaling parameters come before physical parameters."""
        names = get_param_names_in_order(n_phi=2, analysis_mode="static")

        scaling_indices = [
            i
            for i, n in enumerate(names)
            if n.startswith("contrast_") or n.startswith("offset_")
        ]
        physical_indices = [
            i for i, n in enumerate(names) if n in ["D0", "alpha", "D_offset"]
        ]

        # All scaling indices should be less than all physical indices
        assert max(scaling_indices) < min(physical_indices)


class TestBuildPrior:
    """Tests for build_prior function."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a mock ParameterSpace for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockPriorSpec:
            """Mock PriorDistribution-like object."""

            dist_type: str
            min_val: float
            max_val: float
            mu: float
            sigma: float

        class MockParameterSpace:
            def get_prior(self, param_name):
                # Use mock spec with correct dist_type
                priors = {
                    "contrast": MockPriorSpec(
                        dist_type="TruncatedNormal",
                        min_val=0.0,
                        max_val=1.0,
                        mu=0.5,
                        sigma=0.2,
                    ),
                    "offset": MockPriorSpec(
                        dist_type="TruncatedNormal",
                        min_val=0.5,
                        max_val=1.5,
                        mu=1.0,
                        sigma=0.2,
                    ),
                    "D0": MockPriorSpec(
                        dist_type="TruncatedNormal",
                        min_val=1.0,
                        max_val=1000000.0,
                        mu=1000.0,
                        sigma=500.0,
                    ),
                    "alpha": MockPriorSpec(
                        dist_type="TruncatedNormal",
                        min_val=-2.0,
                        max_val=2.0,
                        mu=0.0,
                        sigma=0.5,
                    ),
                }
                result = priors.get(param_name)
                if result is None:
                    raise KeyError(f"Unknown parameter: {param_name}")
                return result

            def get_bounds(self, param_name):
                bounds = {
                    "contrast": (0.0, 1.0),
                    "offset": (0.5, 1.5),
                    "D0": (1.0, 1000000.0),
                    "alpha": (-2.0, 2.0),
                }
                result = bounds.get(param_name)
                if result is None:
                    raise KeyError(f"Unknown parameter: {param_name}")
                return result

        return MockParameterSpace()

    def test_build_contrast_prior(self, mock_parameter_space):
        """Test building prior for contrast parameter."""
        prior = build_prior("contrast", mock_parameter_space)

        # Should return a NumPyro distribution
        assert prior is not None
        assert callable(prior) or hasattr(prior, 'sample')

    def test_build_d0_prior(self, mock_parameter_space):
        """Test building prior for D0 parameter."""
        prior = build_prior("D0", mock_parameter_space)

        assert prior is not None
        assert callable(prior) or hasattr(prior, 'sample')


class TestGetInitValue:
    """Tests for get_init_value function."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a mock ParameterSpace for testing."""

        class MockParameterSpace:
            def get_bounds(self, param_name):
                bounds = {
                    "contrast": (0.0, 1.0),
                    "D0": (1.0, 1000000.0),
                    "alpha": (-2.0, 2.0),
                }
                return bounds.get(param_name)

        return MockParameterSpace()

    def test_value_from_initial_values(self, mock_parameter_space):
        """Test getting value from initial_values dict."""
        initial_values = {"D0": 5000.0, "alpha": 0.5}

        value = get_init_value("D0", initial_values, mock_parameter_space)

        assert value == 5000.0

    def test_fallback_to_midpoint(self, mock_parameter_space):
        """Test falling back to midpoint when no initial value."""
        value = get_init_value("alpha", None, mock_parameter_space)

        # Midpoint of (-2, 2) is 0
        assert value == 0.0

    def test_fallback_when_param_not_in_dict(self, mock_parameter_space):
        """Test falling back when parameter not in initial_values."""
        initial_values = {"D0": 5000.0}  # alpha not included

        value = get_init_value("alpha", initial_values, mock_parameter_space)

        # Should use midpoint
        assert value == 0.0

    def test_contrast_midpoint(self, mock_parameter_space):
        """Test midpoint calculation for contrast."""
        value = get_init_value("contrast", None, mock_parameter_space)

        # Midpoint of (0, 1) is 0.5
        assert value == 0.5


class TestValidateInitialValueBounds:
    """Tests for validate_initial_value_bounds function."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a mock ParameterSpace for testing."""

        class MockParameterSpace:
            def get_bounds(self, param_name):
                bounds = {
                    "contrast": (0.0, 1.0),
                    "offset": (0.5, 1.5),
                    "D0": (1.0, 1000000.0),
                    "alpha": (-2.0, 2.0),
                }
                return bounds.get(param_name, (0.0, 1.0))

        return MockParameterSpace()

    def test_value_within_bounds_unchanged(self, mock_parameter_space):
        """Test that values within bounds are returned unchanged."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", 0.5, mock_parameter_space
        )
        assert value == 0.5
        assert was_clipped is False

    def test_value_below_lower_bound_clipped(self, mock_parameter_space):
        """Test that values below lower bound are clipped with margin."""
        # alpha bounds are (-2.0, 2.0), so -3.0 is below
        value, was_clipped = validate_initial_value_bounds(
            "alpha", -3.0, mock_parameter_space
        )
        # Should be clipped to lower + 1% margin = -2.0 + 0.04 = -1.96
        assert was_clipped is True
        assert value > -2.0  # Strictly inside bounds
        assert value == pytest.approx(-1.96, abs=0.001)

    def test_value_above_upper_bound_clipped(self, mock_parameter_space):
        """Test that values above upper bound are clipped with margin."""
        # alpha bounds are (-2.0, 2.0), so 3.0 is above
        value, was_clipped = validate_initial_value_bounds(
            "alpha", 3.0, mock_parameter_space
        )
        # Should be clipped to upper - 1% margin = 2.0 - 0.04 = 1.96
        assert was_clipped is True
        assert value < 2.0  # Strictly inside bounds
        assert value == pytest.approx(1.96, abs=0.001)

    def test_nan_value_reset_to_midpoint(self, mock_parameter_space):
        """Test that NaN values are reset to midpoint."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", float("nan"), mock_parameter_space
        )
        # alpha midpoint is (-2.0 + 2.0) / 2 = 0.0
        assert was_clipped is True
        assert value == 0.0
        assert math.isfinite(value)

    def test_inf_value_reset_to_midpoint(self, mock_parameter_space):
        """Test that inf values are reset to midpoint."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", float("inf"), mock_parameter_space
        )
        assert was_clipped is True
        assert value == 0.0
        assert math.isfinite(value)

    def test_negative_inf_value_reset_to_midpoint(self, mock_parameter_space):
        """Test that -inf values are reset to midpoint."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", float("-inf"), mock_parameter_space
        )
        assert was_clipped is True
        assert value == 0.0
        assert math.isfinite(value)

    def test_value_at_exact_lower_bound_unchanged(self, mock_parameter_space):
        """Test that value exactly at lower bound is not clipped."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", -2.0, mock_parameter_space
        )
        # Exactly at bound should be fine (not below)
        assert was_clipped is False
        assert value == -2.0

    def test_value_at_exact_upper_bound_unchanged(self, mock_parameter_space):
        """Test that value exactly at upper bound is not clipped."""
        value, was_clipped = validate_initial_value_bounds(
            "alpha", 2.0, mock_parameter_space
        )
        # Exactly at bound should be fine (not above)
        assert was_clipped is False
        assert value == 2.0

    def test_per_angle_param_uses_base_bounds(self, mock_parameter_space):
        """Test that per-angle params like contrast_0 use base bounds."""
        # contrast bounds are (0.0, 1.0)
        value, was_clipped = validate_initial_value_bounds(
            "contrast_0", 1.5, mock_parameter_space
        )
        # Should be clipped using contrast bounds
        assert was_clipped is True
        assert value < 1.0  # Strictly inside bounds


class TestBuildInitValuesDict:
    """Tests for build_init_values_dict function with out-of-bounds handling."""

    @pytest.fixture
    def mock_parameter_space(self):
        """Create a mock ParameterSpace for testing."""
        from dataclasses import dataclass

        @dataclass
        class MockPriorSpec:
            dist_type: str
            min_val: float
            max_val: float
            mu: float
            sigma: float

        class MockParameterSpace:
            def get_prior(self, param_name):
                priors = {
                    "contrast": MockPriorSpec("TruncatedNormal", 0.0, 1.0, 0.5, 0.2),
                    "offset": MockPriorSpec("TruncatedNormal", 0.5, 1.5, 1.0, 0.2),
                    "D0": MockPriorSpec("TruncatedNormal", 1.0, 1e6, 1e3, 500.0),
                    "alpha": MockPriorSpec("TruncatedNormal", -2.0, 2.0, 0.0, 0.5),
                    "D_offset": MockPriorSpec("TruncatedNormal", -1e5, 1e5, 0.0, 1e4),
                }
                return priors.get(param_name)

            def get_bounds(self, param_name):
                bounds = {
                    "contrast": (0.0, 1.0),
                    "offset": (0.5, 1.5),
                    "D0": (1.0, 1e6),
                    "alpha": (-2.0, 2.0),
                    "D_offset": (-1e5, 1e5),
                }
                return bounds.get(param_name, (0.0, 1.0))

        return MockParameterSpace()

    def test_out_of_bounds_values_clipped(self, mock_parameter_space):
        """Test that out-of-bounds initial values are clipped in build_init_values_dict."""
        # Provide out-of-bounds values
        initial_values = {
            "contrast": 1.5,  # Above upper bound (1.0)
            "offset": 0.2,  # Below lower bound (0.5)
            "D0": 5000.0,  # Within bounds
            "alpha": -3.0,  # Below lower bound (-2.0)
            "D_offset": 0.0,  # Within bounds
        }

        result = build_init_values_dict(
            n_phi=1,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
        )

        # contrast_0 should be clipped to below 1.0
        assert result["contrast_0"] < 1.0
        assert result["contrast_0"] > 0.0

        # offset_0 should be clipped to above 0.5
        assert result["offset_0"] > 0.5
        assert result["offset_0"] < 1.5

        # D0 should be unchanged (within bounds)
        assert result["D0"] == 5000.0

        # alpha should be clipped to above -2.0
        assert result["alpha"] > -2.0
        assert result["alpha"] < 2.0

    def test_scalar_broadcast_to_all_angles(self, mock_parameter_space):
        """Test that scalar values are broadcast to all phi angles."""
        initial_values = {
            "contrast": 0.6,  # Scalar for all angles
            "offset": 1.1,  # Scalar for all angles
            "D0": 5000.0,
            "alpha": -0.5,
            "D_offset": 100.0,
        }

        result = build_init_values_dict(
            n_phi=3,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
        )

        # All contrast_i should have same value
        assert result["contrast_0"] == 0.6
        assert result["contrast_1"] == 0.6
        assert result["contrast_2"] == 0.6

        # All offset_i should have same value
        assert result["offset_0"] == 1.1
        assert result["offset_1"] == 1.1
        assert result["offset_2"] == 1.1

    def test_explicit_per_angle_values_override_scalar(self, mock_parameter_space):
        """Test that explicit per-angle values take precedence over scalar."""
        initial_values = {
            "contrast": 0.5,  # Scalar fallback
            "contrast_0": 0.3,  # Explicit override
            "contrast_2": 0.7,  # Explicit override
            "offset": 1.0,
            "D0": 5000.0,
            "alpha": -0.5,
            "D_offset": 100.0,
        }

        result = build_init_values_dict(
            n_phi=3,
            analysis_mode="static",
            initial_values=initial_values,
            parameter_space=mock_parameter_space,
        )

        # Explicit values should be used
        assert result["contrast_0"] == 0.3
        assert result["contrast_2"] == 0.7
        # Scalar fallback for contrast_1
        assert result["contrast_1"] == 0.5
