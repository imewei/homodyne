"""Integration tests for CMC reparameterization."""

import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro", reason="NumPyro required")

import jax.numpy as jnp

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.model import xpcs_model_reparameterized
from homodyne.optimization.cmc.reparameterization import (
    ReparamConfig,
    transform_to_physics_space,
)


@pytest.fixture
def mock_parameter_space():
    """Create parameter space for laminar_flow mode."""
    from homodyne.config.parameter_space import ParameterSpace

    return ParameterSpace.from_defaults("laminar_flow")


class TestReparameterizationIntegration:
    """Integration tests for reparameterization in CMC."""

    @pytest.mark.slow
    def test_reparameterized_model_produces_valid_samples(
        self, mock_parameter_space
    ):
        """Reparameterized model produces samples convertible to physics params."""
        import jax.random as random
        from numpyro.infer import MCMC, NUTS

        # Minimal synthetic data
        n_points = 200
        data = jnp.ones(n_points) * 1.2  # g2 ~ 1.2 (realistic)
        t1 = jnp.linspace(0.1, 1.0, n_points)
        t2 = jnp.linspace(0.1, 1.0, n_points)
        phi_unique = jnp.array([0.0, 0.5, 1.0])
        phi_indices = jnp.zeros(n_points, dtype=jnp.int32)

        reparam_config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True)

        # Run short MCMC
        kernel = NUTS(xpcs_model_reparameterized)
        mcmc = MCMC(kernel, num_warmup=50, num_samples=100, num_chains=1)

        mcmc.run(
            random.PRNGKey(0),
            data=data,
            t1=t1,
            t2=t2,
            phi_unique=phi_unique,
            phi_indices=phi_indices,
            q=0.005,
            L=2e6,
            dt=0.1,
            analysis_mode="laminar_flow",
            parameter_space=mock_parameter_space,
            n_phi=3,
            reparam_config=reparam_config,
        )

        samples = mcmc.get_samples()

        # Verify sampled parameters (reference-time reparameterization)
        assert "log_D_ref" in samples
        assert "D_offset_frac" in samples
        assert "log_gamma_ref" in samples

        # Verify deterministic physics params computed
        assert "D0" in samples
        assert "D_offset" in samples
        assert "gamma_dot_t0" in samples

        # Verify physical constraints
        assert np.all(samples["D0"] > 0), "D0 must be positive"
        assert np.all(samples["gamma_dot_t0"] > 0), "gamma_dot_t0 must be positive"
        assert np.all(samples["D_offset_frac"] >= 0), "D_offset_frac must be >= 0"
        assert np.all(samples["D_offset_frac"] <= 1), "D_offset_frac must be <= 1"

    def test_transform_roundtrip(self):
        """Samples converted to physics space then back are consistent."""
        from homodyne.optimization.cmc.reparameterization import (
            transform_to_sampling_space,
        )

        config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True)

        # Original physics parameters
        original = {
            "D0": 20000.0,
            "D_offset": 1000.0,
            "gamma_dot_t0": 0.002,
            "alpha": -1.0,
            "beta": -0.3,
        }

        # Transform to sampling space
        sampling_params = transform_to_sampling_space(original, config)

        # Convert to arrays for transform_to_physics_space
        samples = {k: np.array([v]) for k, v in sampling_params.items()}

        # Transform back to physics space
        recovered = transform_to_physics_space(samples, config)

        # Verify roundtrip
        np.testing.assert_allclose(recovered["D0"][0], original["D0"], rtol=1e-10)
        np.testing.assert_allclose(
            recovered["D_offset"][0], original["D_offset"], rtol=1e-10
        )
        np.testing.assert_allclose(
            recovered["gamma_dot_t0"][0], original["gamma_dot_t0"], rtol=1e-10
        )

    def test_param_aware_sizing_reduces_shards(self):
        """Higher param count results in fewer, larger shards."""
        # With 7 params (static)
        config_7 = CMCConfig(max_points_per_shard=50000, min_points_per_param=1500)
        n_shards_7 = config_7.get_num_shards(n_points=500000, n_phi=5, n_params=7)

        # With 10 params (laminar_flow with averaged scaling)
        config_10 = CMCConfig(max_points_per_shard=50000, min_points_per_param=1500)
        n_shards_10 = config_10.get_num_shards(n_points=500000, n_phi=5, n_params=10)

        # With 14 params (laminar_flow with individual scaling, many angles)
        config_14 = CMCConfig(max_points_per_shard=50000, min_points_per_param=1500)
        n_shards_14 = config_14.get_num_shards(n_points=500000, n_phi=5, n_params=14)

        # More params → larger shards → fewer shards
        assert n_shards_10 < n_shards_7, "10 params should produce fewer shards than 7"
        assert n_shards_14 < n_shards_10, "14 params should produce fewer shards than 10"

    def test_config_reparam_options_used_in_model(self, mock_parameter_space):
        """Config reparameterization options control model behavior."""
        import numpyro.handlers

        from homodyne.optimization.cmc.model import get_xpcs_model

        n_points = 50
        data = jnp.ones(n_points)
        t1 = jnp.linspace(0.1, 1.0, n_points)
        t2 = jnp.linspace(0.1, 1.0, n_points)
        phi_unique = jnp.array([0.0])
        phi_indices = jnp.zeros(n_points, dtype=jnp.int32)

        # Get reparameterized model
        model_fn = get_xpcs_model(per_angle_mode="auto", use_reparameterization=True)

        reparam_config = ReparamConfig(
            enable_d_ref=True, enable_gamma_ref=True
        )

        # Trace to see sampled params
        with numpyro.handlers.seed(rng_seed=42):
            with numpyro.handlers.trace() as trace:
                model_fn(
                    data=data,
                    t1=t1,
                    t2=t2,
                    phi_unique=phi_unique,
                    phi_indices=phi_indices,
                    q=0.005,
                    L=2e6,
                    dt=0.1,
                    analysis_mode="laminar_flow",
                    parameter_space=mock_parameter_space,
                    n_phi=1,
                    reparam_config=reparam_config,
                )

        sampled_params = [k for k, v in trace.items() if v.get("type") == "sample"]

        # Should sample log_D_ref and log_gamma_ref, not D0_z and gamma_dot_t0_z
        assert "log_D_ref" in sampled_params
        assert "log_gamma_ref" in sampled_params
        assert "D0_z" not in sampled_params
        assert "gamma_dot_t0_z" not in sampled_params
