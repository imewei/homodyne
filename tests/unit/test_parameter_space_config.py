"""Unit Tests for ParameterSpace Configuration Loading
=====================================================

Tests the ParameterSpace class for loading parameter bounds and prior
distributions from YAML configuration files.

Part of Task Group 2.1: ParameterSpace Configuration Loading
See: /home/wei/Documents/GitHub/homodyne/agent-os/specs/2025-10-31-mcmc-simplification/tasks.md
"""

import numpy as np
import pytest

from homodyne.config.parameter_space import (
    ParameterSpace,
    PriorDistribution,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def static_config_valid():
    """Valid static configuration with complete parameter_space section."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                    "unit": "Å²/s",
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                    "unit": "dimensionless",
                },
                {
                    "name": "D_offset",
                    "min": -1000.0,
                    "max": 1000.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 100.0,
                    "unit": "Å²/s",
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def laminar_flow_config_valid():
    """Valid laminar flow configuration with complete parameter_space section."""
    return {
        "analysis_mode": "laminar_flow",
        "parameter_space": {
            "model": "laminar_flow",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                },
                {
                    "name": "D_offset",
                    "min": -1000.0,
                    "max": 1000.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 100.0,
                },
                {
                    "name": "gamma_dot_0",  # Config name (will be mapped)
                    "min": 1e-10,
                    "max": 1.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.01,
                    "prior_sigma": 0.01,
                },
                {
                    "name": "beta",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",
                    "prior_mu": 0.0,
                    "prior_sigma": 0.5,
                },
                {
                    "name": "gamma_dot_t_offset",
                    "min": 0.0,
                    "max": 1.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 0.0,
                    "prior_sigma": 0.1,
                },
                {
                    "name": "phi0",
                    "min": -np.pi,
                    "max": np.pi,
                    "type": "Uniform",
                    "prior_mu": 0.0,
                    "prior_sigma": 1.0,
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ],
        },
    }


@pytest.fixture
def config_missing_bounds():
    """Configuration with missing parameter_space.bounds section."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            # Missing 'bounds' section
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def config_partial_bounds():
    """Configuration with only partial bounds (some parameters missing)."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                },
                # Missing 'alpha' and 'D_offset'
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0", "alpha", "D_offset"],
        },
    }


@pytest.fixture
def config_invalid_prior():
    """Configuration with invalid prior distribution parameters."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "TruncatedNormal",
                    "prior_mu": 1000.0,
                    "prior_sigma": -500.0,  # Negative sigma (invalid)
                },
            ],
        },
        "initial_parameters": {
            "parameter_names": ["D0"],
        },
    }


# =============================================================================
# Test PriorDistribution Class
# =============================================================================


class TestPriorDistribution:
    """Tests for PriorDistribution dataclass."""

    def test_truncated_normal_creation(self):
        """Test creating TruncatedNormal prior."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=1000.0,
            sigma=500.0,
            min_val=100.0,
            max_val=1e5,
        )

        assert prior.dist_type == "TruncatedNormal"
        assert prior.mu == 1000.0
        assert prior.sigma == 500.0
        assert prior.min_val == 100.0
        assert prior.max_val == 1e5

    def test_normal_creation(self):
        """Test creating Normal prior."""
        prior = PriorDistribution(
            dist_type="Normal",
            mu=-1.2,
            sigma=0.3,
            min_val=-np.inf,
            max_val=np.inf,
        )

        assert prior.dist_type == "Normal"
        assert prior.mu == -1.2
        assert prior.sigma == 0.3

    def test_uniform_creation(self):
        """Test creating Uniform prior."""
        prior = PriorDistribution(
            dist_type="Uniform", mu=0.0, sigma=1.0, min_val=-np.pi, max_val=np.pi
        )

        assert prior.dist_type == "Uniform"
        assert prior.min_val == pytest.approx(-np.pi)
        assert prior.max_val == pytest.approx(np.pi)

    def test_invalid_bounds_raises_error(self):
        """Test that invalid bounds (min >= max) raise ValueError."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            PriorDistribution(
                dist_type="TruncatedNormal",
                mu=1000.0,
                sigma=500.0,
                min_val=1e5,  # min > max
                max_val=100.0,
            )

    def test_truncated_normal_requires_finite_bounds(self):
        """Test that TruncatedNormal requires finite bounds."""
        with pytest.raises(ValueError, match="finite bounds"):
            PriorDistribution(
                dist_type="TruncatedNormal",
                mu=0.0,
                sigma=1.0,
                min_val=-np.inf,  # Infinite bound (invalid for TruncatedNormal)
                max_val=np.inf,
            )

    def test_unknown_dist_type_warning(self):
        """Test that unknown distribution type defaults to TruncatedNormal."""
        prior = PriorDistribution(
            dist_type="UnknownDistribution",
            mu=0.0,
            sigma=1.0,
            min_val=0.0,
            max_val=1.0,
        )

        # Should default to TruncatedNormal
        assert prior.dist_type == "TruncatedNormal"

    def test_to_numpyro_kwargs_truncated_normal(self):
        """Test conversion to NumPyro kwargs for TruncatedNormal."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=1000.0,
            sigma=500.0,
            min_val=100.0,
            max_val=1e5,
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["loc"] == 1000.0
        assert kwargs["scale"] == 500.0
        assert kwargs["low"] == 100.0
        assert kwargs["high"] == 1e5

    def test_to_numpyro_kwargs_normal(self):
        """Test conversion to NumPyro kwargs for Normal."""
        prior = PriorDistribution(
            dist_type="Normal", mu=-1.2, sigma=0.3, min_val=-2.0, max_val=2.0
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["loc"] == -1.2
        assert kwargs["scale"] == 0.3
        assert "low" not in kwargs
        assert "high" not in kwargs

    def test_to_numpyro_kwargs_uniform(self):
        """Test conversion to NumPyro kwargs for Uniform."""
        prior = PriorDistribution(
            dist_type="Uniform", mu=0.0, sigma=1.0, min_val=-np.pi, max_val=np.pi
        )

        kwargs = prior.to_numpyro_kwargs()

        assert kwargs["low"] == pytest.approx(-np.pi)
        assert kwargs["high"] == pytest.approx(np.pi)


# =============================================================================
# Test ParameterSpace.from_config()
# =============================================================================


class TestParameterSpaceFromConfig:
    """Tests for ParameterSpace.from_config() class method."""

    def test_load_static_config_valid(self, static_config_valid):
        """Test loading valid static configuration."""
        param_space = ParameterSpace.from_config(static_config_valid)

        # Check model type
        assert param_space.model_type == "static"

        # Check parameter names
        assert len(param_space.parameter_names) == 3
        assert "D0" in param_space.parameter_names
        assert "alpha" in param_space.parameter_names
        assert "D_offset" in param_space.parameter_names

        # Check bounds
        assert param_space.get_bounds("D0") == (100.0, 1e5)
        assert param_space.get_bounds("alpha") == (-2.0, 2.0)
        assert param_space.get_bounds("D_offset") == (-1000.0, 1000.0)

        # Check priors
        prior_D0 = param_space.get_prior("D0")
        assert prior_D0.dist_type == "TruncatedNormal"
        assert prior_D0.mu == 1000.0
        assert prior_D0.sigma == 1000.0

        prior_alpha = param_space.get_prior("alpha")
        assert prior_alpha.dist_type == "Normal"
        assert prior_alpha.mu == -1.2
        assert prior_alpha.sigma == 0.3

        # Check units
        assert param_space.units["D0"] == "Å²/s"
        assert param_space.units["alpha"] == "dimensionless"

    def test_load_laminar_flow_config_valid(self, laminar_flow_config_valid):
        """Test loading valid laminar flow configuration."""
        param_space = ParameterSpace.from_config(laminar_flow_config_valid)

        # Check model type
        assert param_space.model_type == "laminar_flow"

        # Check parameter count (7 for laminar flow)
        assert len(param_space.parameter_names) == 7

        # Check name mapping (gamma_dot_0 → gamma_dot_t0)
        assert "gamma_dot_t0" in param_space.parameter_names

        # Check bounds for mapped parameter
        bounds = param_space.get_bounds("gamma_dot_t0")
        assert bounds == (1e-10, 1.0)

        # Check prior for mapped parameter
        prior = param_space.get_prior("gamma_dot_t0")
        assert prior.dist_type == "TruncatedNormal"
        assert prior.mu == 0.01

    def test_load_config_missing_bounds(self, config_missing_bounds):
        """Test loading config with missing bounds section (uses defaults)."""
        param_space = ParameterSpace.from_config(config_missing_bounds)

        # Should still create parameter space with defaults
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3

        # Should have bounds from ParameterManager defaults
        bounds_D0 = param_space.get_bounds("D0")
        assert bounds_D0[0] > 0  # Has some default min
        assert bounds_D0[1] > bounds_D0[0]  # max > min

    def test_load_config_partial_bounds(self, config_partial_bounds):
        """Test loading config with partial bounds (missing some parameters)."""
        param_space = ParameterSpace.from_config(config_partial_bounds)

        # Check that D0 has config bounds
        assert param_space.get_bounds("D0") == (100.0, 1e5)

        # Check that alpha and D_offset have default bounds
        bounds_alpha = param_space.get_bounds("alpha")
        assert bounds_alpha[0] < 0  # Should have some negative min
        assert bounds_alpha[1] > 0  # Should have some positive max

        bounds_offset = param_space.get_bounds("D_offset")
        assert bounds_offset is not None

    def test_load_config_no_parameter_space_section(self):
        """Test loading config without parameter_space section (uses defaults)."""
        config = {
            "analysis_mode": "static",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        param_space = ParameterSpace.from_config(config)

        # Should create parameter space with package defaults
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3
        assert len(param_space.bounds) == 3
        assert len(param_space.priors) == 3

    def test_auto_detect_analysis_mode(self):
        """Test automatic detection of analysis mode from config."""
        config = {
            "analysis_mode": "laminar_flow",  # Should auto-detect from here
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_t0",
                    "beta",
                    "gamma_dot_t_offset",
                    "phi0",
                ],
            },
        }

        # Don't specify analysis_mode parameter
        param_space = ParameterSpace.from_config(config)

        assert param_space.model_type == "laminar_flow"
        assert len(param_space.parameter_names) == 7

    def test_explicit_analysis_mode_override(self):
        """Test that explicit analysis_mode parameter overrides config."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
            },
        }

        # Explicitly override to static
        param_space = ParameterSpace.from_config(config, analysis_mode="static")

        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3


# =============================================================================
# Test ParameterSpace.from_defaults()
# =============================================================================


class TestParameterSpaceFromDefaults:
    """Tests for ParameterSpace.from_defaults() class method."""

    def test_from_defaults_static(self):
        """Test creating static parameter space from defaults."""
        param_space = ParameterSpace.from_defaults("static")

        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3
        assert "D0" in param_space.parameter_names
        assert "alpha" in param_space.parameter_names
        assert "D_offset" in param_space.parameter_names

        # Should have bounds
        assert len(param_space.bounds) == 3

        # Should have priors (with defaults)
        assert len(param_space.priors) == 3

    def test_from_defaults_laminar_flow(self):
        """Test creating laminar flow parameter space from defaults."""
        param_space = ParameterSpace.from_defaults("laminar_flow")

        assert param_space.model_type == "laminar_flow"
        assert len(param_space.parameter_names) == 7

        # Should have all laminar flow parameters
        expected_params = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for param in expected_params:
            assert param in param_space.parameter_names

    def test_from_defaults_has_wide_priors(self):
        """Test that default priors are reasonably wide."""
        param_space = ParameterSpace.from_defaults("static")

        # Check that priors are not too narrow
        for param_name in param_space.parameter_names:
            prior = param_space.get_prior(param_name)
            bounds = param_space.get_bounds(param_name)

            # Prior sigma should be reasonably large (at least 10% of range)
            bounds_range = bounds[1] - bounds[0]
            assert prior.sigma >= 0.1 * bounds_range


# =============================================================================
# Test ParameterSpace Methods
# =============================================================================


class TestParameterSpaceMethods:
    """Tests for ParameterSpace instance methods."""

    def test_get_bounds_valid_parameter(self, static_config_valid):
        """Test getting bounds for valid parameter."""
        param_space = ParameterSpace.from_config(static_config_valid)

        bounds = param_space.get_bounds("D0")

        assert bounds == (100.0, 1e5)

    def test_get_bounds_invalid_parameter(self, static_config_valid):
        """Test getting bounds for invalid parameter raises KeyError."""
        param_space = ParameterSpace.from_config(static_config_valid)

        with pytest.raises(KeyError, match="not in parameter space"):
            param_space.get_bounds("nonexistent_parameter")

    def test_get_prior_valid_parameter(self, static_config_valid):
        """Test getting prior for valid parameter."""
        param_space = ParameterSpace.from_config(static_config_valid)

        prior = param_space.get_prior("alpha")

        assert prior.dist_type == "Normal"
        assert prior.mu == -1.2
        assert prior.sigma == 0.3

    def test_get_prior_invalid_parameter(self, static_config_valid):
        """Test getting prior for invalid parameter raises KeyError."""
        param_space = ParameterSpace.from_config(static_config_valid)

        with pytest.raises(KeyError, match="not in parameter space"):
            param_space.get_prior("nonexistent_parameter")

    def test_get_bounds_array(self, static_config_valid):
        """Test getting bounds as numpy arrays."""
        param_space = ParameterSpace.from_config(static_config_valid)

        lower, upper = param_space.get_bounds_array()

        # Should have 3 elements (static mode)
        assert lower.shape == (3,)
        assert upper.shape == (3,)

        # Check values (order matches parameter_names)
        param_names = param_space.parameter_names
        for i, param_name in enumerate(param_names):
            expected_bounds = param_space.get_bounds(param_name)
            assert lower[i] == expected_bounds[0]
            assert upper[i] == expected_bounds[1]

        # All lower bounds should be less than upper bounds
        assert np.all(lower < upper)

    def test_get_prior_means(self, static_config_valid):
        """Test getting prior means as numpy array."""
        param_space = ParameterSpace.from_config(static_config_valid)

        means = param_space.get_prior_means()

        # Should have 3 elements
        assert means.shape == (3,)

        # Check values match priors
        param_names = param_space.parameter_names
        for i, param_name in enumerate(param_names):
            expected_mu = param_space.get_prior(param_name).mu
            assert means[i] == expected_mu

    def test_validate_values_all_valid(self, static_config_valid):
        """Test validating parameter values that are all within bounds."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {"D0": 1000.0, "alpha": -1.2, "D_offset": 0.0}

        is_valid, violations = param_space.validate_values(values)

        assert is_valid
        assert len(violations) == 0

    def test_validate_values_below_min(self, static_config_valid):
        """Test validating parameter value below minimum bound."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {"D0": 50.0, "alpha": -1.2, "D_offset": 0.0}  # D0 < min (100.0)

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "D0" in violations[0]
        assert "< min" in violations[0]

    def test_validate_values_above_max(self, static_config_valid):
        """Test validating parameter value above maximum bound."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 2e5,  # D0 > max (1e5)
            "alpha": -1.2,
            "D_offset": 0.0,
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "D0" in violations[0]
        assert "> max" in violations[0]

    def test_validate_values_multiple_violations(self, static_config_valid):
        """Test validating with multiple bound violations."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 50.0,  # Below min
            "alpha": 5.0,  # Above max
            "D_offset": -2000.0,  # Below min
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 3

    def test_validate_values_unknown_parameter(self, static_config_valid):
        """Test validating with unknown parameter name."""
        param_space = ParameterSpace.from_config(static_config_valid)

        values = {
            "D0": 1000.0,
            "unknown_param": 42.0,  # Not in parameter space
        }

        is_valid, violations = param_space.validate_values(values)

        assert not is_valid
        assert len(violations) == 1
        assert "unknown_param" in violations[0]
        assert "Unknown parameter" in violations[0]


# =============================================================================
# Test ParameterSpace String Representations
# =============================================================================


class TestParameterSpaceStringRepresentations:
    """Tests for __repr__ and __str__ methods."""

    def test_repr(self, static_config_valid):
        """Test __repr__ output."""
        param_space = ParameterSpace.from_config(static_config_valid)

        repr_str = repr(param_space)

        assert "ParameterSpace" in repr_str
        assert "static" in repr_str
        assert "n_params=3" in repr_str

    def test_str(self, static_config_valid):
        """Test __str__ output."""
        param_space = ParameterSpace.from_config(static_config_valid)

        str_repr = str(param_space)

        assert "ParameterSpace" in str_repr
        assert "static" in str_repr
        assert "D0" in str_repr
        assert "alpha" in str_repr
        assert "D_offset" in str_repr


# =============================================================================
# Integration Tests
# =============================================================================


class TestParameterSpaceIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow_static(self, static_config_valid):
        """Test complete workflow: load → validate → get arrays."""
        # Load from config
        param_space = ParameterSpace.from_config(static_config_valid)

        # Validate parameter values
        values = {"D0": 1000.0, "alpha": -1.2, "D_offset": 0.0}
        is_valid, _ = param_space.validate_values(values)
        assert is_valid

        # Get bounds as arrays
        lower, upper = param_space.get_bounds_array()
        assert lower.shape == (3,)

        # Get prior means
        means = param_space.get_prior_means()
        assert means.shape == (3,)

        # Check that means are within bounds
        assert np.all(means >= lower)
        assert np.all(means <= upper)

    def test_full_workflow_laminar_flow(self, laminar_flow_config_valid):
        """Test complete workflow for laminar flow mode."""
        # Load from config
        param_space = ParameterSpace.from_config(laminar_flow_config_valid)

        # Should have 7 parameters
        assert len(param_space.parameter_names) == 7

        # Get bounds array
        lower, upper = param_space.get_bounds_array()
        assert lower.shape == (7,)
        assert upper.shape == (7,)

        # Get prior means
        means = param_space.get_prior_means()
        assert means.shape == (7,)

        # Means should be within bounds
        assert np.all(means >= lower)
        assert np.all(means <= upper)

    def test_config_to_numpyro_workflow(self, static_config_valid):
        """Test workflow from config to NumPyro distribution kwargs."""
        param_space = ParameterSpace.from_config(static_config_valid)

        # For each parameter, get NumPyro kwargs
        for param_name in param_space.parameter_names:
            prior = param_space.get_prior(param_name)
            kwargs = prior.to_numpyro_kwargs()

            # Should have required keys for the distribution type
            if prior.dist_type == "TruncatedNormal":
                assert "loc" in kwargs
                assert "scale" in kwargs
                assert "low" in kwargs
                assert "high" in kwargs
            elif prior.dist_type == "Normal":
                assert "loc" in kwargs
                assert "scale" in kwargs
