"""Tests for CMC prior builders module."""

import numpy as np
import pytest

from homodyne.optimization.cmc.priors import (
    build_prior,
    get_init_value,
    get_param_names_in_order,
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
        physical_indices = [i for i, n in enumerate(names) if n in ["D0", "alpha", "D_offset"]]

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

    def test_build_d0_prior(self, mock_parameter_space):
        """Test building prior for D0 parameter."""
        prior = build_prior("D0", mock_parameter_space)

        assert prior is not None


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
