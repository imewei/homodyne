"""Tests for single-angle parameter sampling (v2.4.1+).

Updated (v2.4.1): Tier system completely removed. This file now tests that
single-angle static mode correctly samples all 5 parameters without any
reparameterization (no log_D_center, delta_raw). Log-space D0 sampling
is handled via TransformedDistribution with ExpTransform.
"""

import numpy as np
import pytest

try:
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    from numpyro import handlers

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestSingleAngleParameterSampling:
    """Test suite for single-angle parameter sampling (v2.4.1+)."""

    def test_single_angle_samples_all_five_parameters(self):
        """Test that single-angle static samples ALL 5 parameters.

        v2.4.1: Tier system removed. Single-angle datasets now sample
        all 5 parameters: D0, alpha, D_offset, contrast, offset.
        """
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import (
            _create_numpyro_model,
            build_log_d0_prior_config,
        )

        # Single-angle data (n_phi=1)
        n_points = 20
        phi_vals = np.array([0.0])
        phi_flat = np.zeros(n_points)

        param_space = ParameterSpace.from_defaults("static")
        d0_bounds = param_space.get_bounds("D0")
        d0_prior = param_space.get_prior("D0")
        log_d0_prior_config = build_log_d0_prior_config(d0_bounds, d0_prior)

        model = _create_numpyro_model(
            data=np.ones(n_points) * 1.05,
            sigma=np.ones(n_points) * 0.01,
            t1=np.linspace(0, 1, n_points),
            t2=np.linspace(0, 2, n_points),
            phi=phi_vals,
            q=0.01,
            L=1.0,
            analysis_mode="static",
            parameter_space=param_space,
            dt=0.1,
            per_angle_scaling=True,
            log_d0_prior_config=log_d0_prior_config,
        )

        seeded = handlers.seed(model, random.PRNGKey(42))
        trace = handlers.trace(seeded).get_trace()

        # Verify all 5 parameters are sampled
        expected_params = ["D0", "alpha", "D_offset", "contrast_0", "offset_0"]
        for param in expected_params:
            assert param in trace, f"Missing parameter: {param}"
            assert "value" in trace[param], f"Parameter {param} has no sampled value"

        # Verify log_D0_latent exists (from log-space sampling)
        assert "log_D0_latent" in trace, "log_D0_latent should exist for diagnostics"

    def test_no_reparameterization_parameters(self):
        """Test that reparameterization params (log_D_center, delta_raw) are NOT used.

        v2.4.1: Tier system removed. The old reparameterization approach
        with log_D_center and delta_raw is no longer used. Log-space D0
        sampling is now done via TransformedDistribution with ExpTransform.
        """
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import (
            _create_numpyro_model,
            build_log_d0_prior_config,
        )

        # Single-angle data
        n_points = 20
        phi_vals = np.array([0.0])

        param_space = ParameterSpace.from_defaults("static")
        d0_bounds = param_space.get_bounds("D0")
        d0_prior = param_space.get_prior("D0")
        log_d0_prior_config = build_log_d0_prior_config(d0_bounds, d0_prior)

        model = _create_numpyro_model(
            data=np.ones(n_points) * 1.05,
            sigma=np.ones(n_points) * 0.01,
            t1=np.linspace(0, 1, n_points),
            t2=np.linspace(0, 2, n_points),
            phi=phi_vals,
            q=0.01,
            L=1.0,
            analysis_mode="static",
            parameter_space=param_space,
            dt=0.1,
            per_angle_scaling=True,
            log_d0_prior_config=log_d0_prior_config,
        )

        seeded = handlers.seed(model, random.PRNGKey(42))
        trace = handlers.trace(seeded).get_trace()

        # These old reparameterization params should NOT exist
        assert "log_D_center" not in trace, (
            "log_D_center (old reparam) should not exist"
        )
        assert "delta_raw" not in trace, "delta_raw (old reparam) should not exist"

    def test_log_d0_prior_config_structure(self):
        """Test that build_log_d0_prior_config returns correct structure."""
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import build_log_d0_prior_config

        param_space = ParameterSpace.from_defaults("static")
        d0_bounds = param_space.get_bounds("D0")
        d0_prior = param_space.get_prior("D0")

        config = build_log_d0_prior_config(d0_bounds, d0_prior)

        # Verify required keys
        required_keys = ["loc", "scale", "low", "high"]
        for key in required_keys:
            assert key in config, f"Missing key: {key}"

        # Verify values are valid
        assert config["scale"] > 0, "scale must be positive"
        assert config["high"] > config["low"], "high must be > low"
        assert np.isfinite(config["loc"]), "loc must be finite"

        # Verify log bounds correspond to D0 bounds
        assert np.exp(config["low"]) >= d0_bounds[0] * 0.99
        assert np.exp(config["high"]) <= d0_bounds[1] * 1.01

    def test_d0_value_is_positive(self):
        """Test that D0 values from log-space sampling are always positive."""
        from homodyne.optimization.mcmc import _sample_single_angle_log_d0

        prior_cfg = {
            "loc": np.log(1000.0),
            "scale": 0.5,
            "low": np.log(100.0),
            "high": np.log(10000.0),
        }

        def model():
            return _sample_single_angle_log_d0(prior_cfg, jnp.float64)

        # Sample multiple times to verify positivity
        for seed in range(5):
            seeded = handlers.seed(model, random.PRNGKey(seed))
            trace = handlers.trace(seeded).get_trace()

            d0_value = float(trace["D0"]["value"])
            assert d0_value > 0, f"D0 must be positive, got {d0_value}"
            assert d0_value >= 100.0, f"D0 below lower bound: {d0_value}"
            assert d0_value <= 10000.0, f"D0 above upper bound: {d0_value}"

    def test_multi_angle_does_not_use_log_d0_config(self):
        """Test that multi-angle paths don't use log-space D0 sampling."""
        from homodyne.config.parameter_space import ParameterSpace
        from homodyne.optimization.mcmc import _create_numpyro_model

        # Multi-angle data (n_phi=3)
        n_phi = 3
        n_per_phi = 10
        n_points = n_phi * n_per_phi

        phi_vals = np.linspace(0.0, np.pi / 4, n_phi)
        phi_flat = np.repeat(phi_vals, n_per_phi)

        param_space = ParameterSpace.from_defaults("static")

        # Don't pass log_d0_prior_config for multi-angle
        model = _create_numpyro_model(
            data=np.ones(n_points) * 1.05,
            sigma=np.ones(n_points) * 0.01,
            t1=np.linspace(0, 1, n_points),
            t2=np.linspace(0, 2, n_points),
            phi=phi_vals,
            q=0.01,
            L=1.0,
            analysis_mode="static",
            parameter_space=param_space,
            dt=0.1,
            per_angle_scaling=True,
            # No log_d0_prior_config for multi-angle
        )

        seeded = handlers.seed(model, random.PRNGKey(42))
        trace = handlers.trace(seeded).get_trace()

        # Multi-angle should NOT have log_D0_latent
        assert "log_D0_latent" not in trace, (
            "Multi-angle should use linear D0 sampling"
        )

        # Should have D0 sampled directly
        assert "D0" in trace, "D0 should be sampled for multi-angle"

        # Should have per-angle scaling (3 contrast, 3 offset)
        for i in range(n_phi):
            assert f"contrast_{i}" in trace, f"Missing contrast_{i}"
            assert f"offset_{i}" in trace, f"Missing offset_{i}"
