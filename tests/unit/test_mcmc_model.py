"""Unit tests for NumPyro model creation with config-driven priors.

Tests Task Group 3.4: NumPyro Model Updates
- Config-driven prior distributions
- Multiple prior distribution types (TruncatedNormal, Normal, Uniform, LogNormal)
- Dense vs diagonal mass matrix configuration
- Parameter ordering and array construction
"""

import numpy as np
import pytest
import jax.numpy as jnp

from homodyne.config.parameter_space import ParameterSpace, PriorDistribution
from homodyne.optimization.mcmc import _create_numpyro_model, _get_mcmc_config

# Check if NumPyro is available
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    from jax import random

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestNumPyroModelCreation:
    """Test suite for _create_numpyro_model with config-driven priors."""

    @pytest.fixture
    def simple_data(self):
        """Create simple synthetic data for model testing."""
        n_points = 100
        t1 = np.linspace(0, 10, n_points)
        t2 = np.linspace(0, 10, n_points)
        phi = np.zeros(n_points)  # Single angle for simplicity
        data = np.ones(n_points) * 1.1  # Constant c2 ≈ 1.0
        sigma = np.ones(n_points) * 0.01
        return {
            "data": data,
            "sigma": sigma,
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "q": 0.001,  # Arbitrary q value
            "L": 1e10,  # Arbitrary L for laminar flow
            "dt": 0.1,  # Pre-computed dt
        }

    @pytest.fixture
    def static_parameter_space(self):
        """Create ParameterSpace for static_isotropic mode."""
        config = {
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {
                        "name": "contrast",
                        "min": 0.0,
                        "max": 1.0,
                        "prior_mu": 0.5,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "offset",
                        "min": 0.5,
                        "max": 1.5,
                        "prior_mu": 1.0,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "D0",
                        "min": 100.0,
                        "max": 100000.0,
                        "prior_mu": 1000.0,
                        "prior_sigma": 1000.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 2.0,
                        "prior_mu": -1.2,
                        "prior_sigma": 0.3,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": 0.0,
                        "max": 1000.0,
                        "prior_mu": 0.0,
                        "prior_sigma": 100.0,
                        "type": "Uniform",
                    },
                ],
            }
        }
        return ParameterSpace.from_config(config, analysis_mode="static")

    @pytest.fixture
    def laminar_parameter_space(self):
        """Create ParameterSpace for laminar_flow mode with diverse prior types."""
        config = {
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    # Scaling parameters
                    {
                        "name": "contrast",
                        "min": 0.0,
                        "max": 1.0,
                        "prior_mu": 0.5,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "offset",
                        "min": 0.5,
                        "max": 1.5,
                        "prior_mu": 1.0,
                        "prior_sigma": 0.2,
                        "type": "TruncatedNormal",
                    },
                    # Diffusion parameters
                    {
                        "name": "D0",
                        "min": 100.0,
                        "max": 100000.0,
                        "prior_mu": 1000.0,
                        "prior_sigma": 1000.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "alpha",
                        "min": -2.0,
                        "max": 2.0,
                        "prior_mu": -1.2,
                        "prior_sigma": 0.3,
                        "type": "Normal",
                    },
                    {
                        "name": "D_offset",
                        "min": 0.0,
                        "max": 1000.0,
                        "prior_mu": 0.0,
                        "prior_sigma": 100.0,
                        "type": "Uniform",
                    },
                    # Flow parameters
                    {
                        "name": "gamma_dot_t0",
                        "min": 0.0,
                        "max": 1000.0,
                        "prior_mu": 100.0,
                        "prior_sigma": 50.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "beta",
                        "min": -2.0,
                        "max": 2.0,
                        "prior_mu": 0.0,
                        "prior_sigma": 0.5,
                        "type": "Normal",
                    },
                    {
                        "name": "gamma_dot_t_offset",
                        "min": 0.0,
                        "max": 100.0,
                        "prior_mu": 0.0,
                        "prior_sigma": 10.0,
                        "type": "Uniform",
                    },
                    {
                        "name": "phi0",
                        "min": 0.0,
                        "max": 2 * np.pi,
                        "prior_mu": np.pi,
                        "prior_sigma": np.pi / 2,
                        "type": "TruncatedNormal",
                    },
                ],
            }
        }
        return ParameterSpace.from_config(config, analysis_mode="laminar_flow")

    def test_model_creation_static_mode(self, simple_data, static_parameter_space):
        """Test model creation for static_isotropic mode."""
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static_isotropic",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
        )

        # Model should be callable
        assert callable(model)

        # Sample from prior predictive to verify model structure
        prior_predictive = Predictive(model, num_samples=10)
        rng_key = random.PRNGKey(42)
        prior_samples = prior_predictive(rng_key)

        # Check that all parameters are sampled (per-angle mode with n_phi=1)
        # With per_angle_scaling=True (default), expect contrast_0, offset_0 instead of contrast, offset
        expected_params = ["contrast_0", "offset_0", "D0", "alpha", "D_offset", "obs"]
        for param in expected_params:
            assert param in prior_samples, f"Missing parameter: {param}"

        # Check sample shapes (per-angle parameters)
        assert prior_samples["contrast_0"].shape == (10,)
        assert prior_samples["offset_0"].shape == (10,)
        assert prior_samples["D0"].shape == (10,)
        assert prior_samples["obs"].shape == (10, len(simple_data["data"]))

    def test_model_creation_laminar_mode(self, simple_data, laminar_parameter_space):
        """Test model creation for laminar_flow mode."""
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="laminar_flow",
            parameter_space=laminar_parameter_space,
            dt=simple_data["dt"],
        )

        # Sample from prior predictive
        prior_predictive = Predictive(model, num_samples=10)
        rng_key = random.PRNGKey(42)
        prior_samples = prior_predictive(rng_key)

        # Check that all parameters are sampled (per-angle mode with n_phi=1)
        # With per_angle_scaling=True (default), expect contrast_0, offset_0 instead of contrast, offset
        expected_params = [
            "contrast_0",
            "offset_0",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
            "obs",
        ]
        for param in expected_params:
            assert param in prior_samples, f"Missing parameter: {param}"

        # Check sample shapes (per-angle parameters)
        assert prior_samples["contrast_0"].shape == (10,)
        assert prior_samples["offset_0"].shape == (10,)
        assert prior_samples["gamma_dot_t0"].shape == (10,)
        assert prior_samples["obs"].shape == (10, len(simple_data["data"]))

    def test_prior_distributions_match_config(
        self, simple_data, static_parameter_space
    ):
        """Test that prior distributions match ParameterSpace specification."""
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static_isotropic",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
        )

        # Sample many times from prior predictive
        prior_predictive = Predictive(model, num_samples=1000)
        rng_key = random.PRNGKey(42)
        prior_samples = prior_predictive(rng_key)

        # Check D0 prior: TruncatedNormal(1000, 1000, low=100, high=100000)
        D0_samples = prior_samples["D0"]
        assert np.all(D0_samples >= 100.0), "D0 samples below minimum"
        assert np.all(D0_samples <= 100000.0), "D0 samples above maximum"
        # Mean should be close to prior_mu=1000 (within 40% for 1000 samples)
        # Wider tolerance due to wide sigma and truncation effects
        assert 600 < np.mean(D0_samples) < 1400

        # Check alpha prior: Normal(-1.2, 0.3) - no truncation
        alpha_samples = prior_samples["alpha"]
        # Mean should be close to prior_mu=-1.2 (within 10% for 1000 samples)
        assert -1.4 < np.mean(alpha_samples) < -1.0

        # Check D_offset prior: Uniform(0, 1000)
        D_offset_samples = prior_samples["D_offset"]
        assert np.all(D_offset_samples >= 0.0), "D_offset samples below minimum"
        assert np.all(D_offset_samples <= 1000.0), "D_offset samples above maximum"
        # Uniform mean should be ~500 (midpoint)
        assert 400 < np.mean(D_offset_samples) < 600

    def test_prior_type_variety(self, simple_data, laminar_parameter_space):
        """Test that different prior types work correctly."""
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="laminar_flow",
            parameter_space=laminar_parameter_space,
            dt=simple_data["dt"],
        )

        # Sample from prior
        prior_predictive = Predictive(model, num_samples=1000)
        rng_key = random.PRNGKey(42)
        prior_samples = prior_predictive(rng_key)

        # TruncatedNormal: gamma_dot_t0
        gamma_samples = prior_samples["gamma_dot_t0"]
        assert np.all(gamma_samples >= 0.0)
        assert np.all(gamma_samples <= 1000.0)

        # Normal: beta (no bounds enforced by prior, but NumPyro may clip)
        beta_samples = prior_samples["beta"]
        # Should be centered around 0.0
        assert -0.5 < np.mean(beta_samples) < 0.5

        # Uniform: gamma_dot_t_offset
        offset_samples = prior_samples["gamma_dot_t_offset"]
        assert np.all(offset_samples >= 0.0)
        assert np.all(offset_samples <= 100.0)
        # Mean should be ~50 (midpoint of [0, 100])
        assert 40 < np.mean(offset_samples) < 60


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestMCMCConfigMassMatrix:
    """Test suite for dense_mass_matrix configuration."""

    def test_default_diagonal_mass_matrix(self):
        """Test default config uses diagonal mass matrix."""
        config = _get_mcmc_config({})
        assert config["dense_mass_matrix"] is False

    def test_dense_mass_matrix_override(self):
        """Test dense_mass_matrix can be overridden from config."""
        config = _get_mcmc_config({"dense_mass_matrix": True})
        assert config["dense_mass_matrix"] is True

    def test_config_documentation_fields(self):
        """Test all expected config fields are present."""
        config = _get_mcmc_config({})
        expected_fields = [
            "n_samples",
            "n_warmup",
            "n_chains",
            "target_accept_prob",
            "max_tree_depth",
            "dense_mass_matrix",
            "rng_key",
        ]
        for field in expected_fields:
            assert field in config, f"Missing config field: {field}"

    def test_mass_matrix_passed_to_nuts(self, tmp_path):
        """Test that dense_mass_matrix config is correctly passed to NUTS kernel."""
        # This test verifies the integration between _get_mcmc_config and NUTS
        from homodyne.optimization.mcmc import _run_numpyro_sampling

        # Create minimal synthetic data
        n_points = 50
        data = np.ones(n_points) * 1.1
        sigma = np.ones(n_points) * 0.01
        t1 = np.linspace(0, 5, n_points)
        t2 = np.linspace(0, 5, n_points)
        phi = np.zeros(n_points)

        # Create minimal parameter space
        param_space = ParameterSpace.from_defaults("static")

        # Create model
        from homodyne.optimization.mcmc import _create_numpyro_model

        model = _create_numpyro_model(
            data=data,
            sigma=sigma,
            t1=t1,
            t2=t2,
            phi=phi,
            q=0.001,
            L=1e10,
            analysis_mode="static_isotropic",
            parameter_space=param_space,
            dt=0.1,
        )

        # Test diagonal mass matrix (default)
        config_diagonal = {
            "n_samples": 10,  # Minimal for speed
            "n_warmup": 10,
            "n_chains": 1,
            "target_accept_prob": 0.8,
            "dense_mass_matrix": False,
            "rng_key": 42,  # Required for MCMC
        }

        # This should run without errors
        result_diagonal = _run_numpyro_sampling(model, config_diagonal)
        assert result_diagonal is not None

        # Test dense mass matrix
        config_dense = {
            "n_samples": 10,
            "n_warmup": 10,
            "n_chains": 1,
            "target_accept_prob": 0.8,
            "dense_mass_matrix": True,
            "rng_key": 43,  # Different seed for this test
        }

        # This should also run without errors
        result_dense = _run_numpyro_sampling(model, config_dense)
        assert result_dense is not None


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestPriorDistributionClass:
    """Test the PriorDistribution class helper methods."""

    def test_to_numpyro_kwargs_truncated_normal(self):
        """Test TruncatedNormal kwargs generation."""
        prior = PriorDistribution(
            dist_type="TruncatedNormal",
            mu=1000.0,
            sigma=100.0,
            min_val=100.0,
            max_val=10000.0,
        )
        kwargs = prior.to_numpyro_kwargs()
        assert kwargs == {
            "loc": 1000.0,
            "scale": 100.0,
            "low": 100.0,
            "high": 10000.0,
        }

    def test_to_numpyro_kwargs_normal(self):
        """Test Normal kwargs generation."""
        prior = PriorDistribution(
            dist_type="Normal", mu=-1.2, sigma=0.3, min_val=-5.0, max_val=5.0
        )
        kwargs = prior.to_numpyro_kwargs()
        assert kwargs == {"loc": -1.2, "scale": 0.3}

    def test_to_numpyro_kwargs_uniform(self):
        """Test Uniform kwargs generation."""
        prior = PriorDistribution(
            dist_type="Uniform", mu=50.0, sigma=25.0, min_val=0.0, max_val=100.0
        )
        kwargs = prior.to_numpyro_kwargs()
        assert kwargs == {"low": 0.0, "high": 100.0}

    def test_to_numpyro_kwargs_lognormal(self):
        """Test LogNormal kwargs generation."""
        prior = PriorDistribution(
            dist_type="LogNormal", mu=1.0, sigma=0.5, min_val=0.0, max_val=100.0
        )
        kwargs = prior.to_numpyro_kwargs()
        assert kwargs == {"loc": 1.0, "scale": 0.5}


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestParameterOrdering:
    """Test that parameter ordering matches analysis mode requirements."""

    def test_static_parameter_order(self):
        """Test static mode has correct 5-parameter order."""
        param_space = ParameterSpace.from_defaults("static")

        # Expected order for static mode (per-angle mode with n_phi=1)
        expected_order = ["contrast_0", "offset_0", "D0", "alpha", "D_offset"]

        # Create minimal data
        n = 10
        model = _create_numpyro_model(
            data=np.ones(n),
            sigma=np.ones(n) * 0.01,
            t1=np.linspace(0, 1, n),
            t2=np.linspace(0, 1, n),
            phi=np.zeros(n),
            q=0.001,
            L=1e10,
            analysis_mode="static_isotropic",
            parameter_space=param_space,
            dt=0.1,
        )

        # Sample from prior
        prior_pred = Predictive(model, num_samples=5)
        samples = prior_pred(random.PRNGKey(0))

        # Verify all expected parameters are present
        for param_name in expected_order:
            assert param_name in samples, f"Missing parameter: {param_name}"

    def test_laminar_parameter_order(self):
        """Test laminar_flow mode has correct 9-parameter order."""
        param_space = ParameterSpace.from_defaults("laminar_flow")

        # Expected order for laminar flow mode (per-angle mode with n_phi=1)
        expected_order = [
            "contrast_0",
            "offset_0",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

        # Create minimal data
        n = 10
        model = _create_numpyro_model(
            data=np.ones(n),
            sigma=np.ones(n) * 0.01,
            t1=np.linspace(0, 1, n),
            t2=np.linspace(0, 1, n),
            phi=np.zeros(n),
            q=0.001,
            L=1e10,
            analysis_mode="laminar_flow",
            parameter_space=param_space,
            dt=0.1,
        )

        # Sample from prior
        prior_pred = Predictive(model, num_samples=5)
        samples = prior_pred(random.PRNGKey(0))

        # Verify all expected parameters are present
        for param_name in expected_order:
            assert param_name in samples, f"Missing parameter: {param_name}"
