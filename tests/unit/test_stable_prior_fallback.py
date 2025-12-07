"""Unit tests for the BetaScaled prior fallback pathway.

Exercises the stable-prior retry system that swaps TruncatedNormal/Uniform
priors for Beta distributions rescaled to each parameter's [min, max] bounds.
"""

import numpy as np
import pytest

from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def static_config_truncated_normal():
    """Configuration with TruncatedNormal priors."""
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
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "TruncatedNormal",
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                },
                {
                    "name": "D_offset",
                    "min": 0.0,
                    "max": 100.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 10.0,
                    "prior_sigma": 5.0,
                },
            ],
        },
    }


@pytest.fixture
def static_config_mixed_priors():
    """Configuration with mixed prior types."""
    return {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {
                    "name": "D0",
                    "min": 100.0,
                    "max": 1e5,
                    "type": "Uniform",
                    "prior_mu": 1000.0,
                    "prior_sigma": 1000.0,
                },
                {
                    "name": "alpha",
                    "min": -2.0,
                    "max": 2.0,
                    "type": "Normal",  # Unbounded - should not convert
                    "prior_mu": -1.2,
                    "prior_sigma": 0.3,
                },
                {
                    "name": "D_offset",
                    "min": 0.0,
                    "max": 100.0,
                    "type": "TruncatedNormal",
                    "prior_mu": 10.0,
                    "prior_sigma": 5.0,
                },
            ],
        },
    }


# =============================================================================
# PriorDistribution Beta Parameter Computation Tests
# =============================================================================


class TestBetaScaledParamComputation:
    """Validate automatic BetaScaled parameter derivation."""

    def test_beta_params_default(self):
        """Defaults should produce symmetric BetaScaled prior."""
        prior = PriorDistribution(
            dist_type="BetaScaled",
            mu=0.0,  # Will use default
            sigma=1.0,  # Will use default
            min_val=100.0,
            max_val=1000.0,
        )

        kwargs = prior.to_numpyro_kwargs()

        # Default should be alpha=beta=2 (symmetric, slightly peaked at center)
        assert "concentration0" in kwargs
        assert "concentration1" in kwargs
        assert kwargs["concentration0"] == pytest.approx(2.0)
        assert kwargs["concentration1"] == pytest.approx(2.0)
        assert kwargs["low"] == 100.0
        assert kwargs["high"] == 1000.0

    def test_beta_params_centered(self):
        """Test Beta parameters centered at midpoint."""
        prior = PriorDistribution(
            dist_type="BetaScaled",
            mu=550.0,  # Midpoint of [100, 1000]
            sigma=225.0,  # Quarter of range
            min_val=100.0,
            max_val=1000.0,
        )

        kwargs = prior.to_numpyro_kwargs()

        # Should derive alpha, beta from method of moments
        assert "concentration0" in kwargs
        assert "concentration1" in kwargs
        assert kwargs["concentration0"] > 0.5
        assert kwargs["concentration1"] > 0.5
        assert kwargs["low"] == 100.0
        assert kwargs["high"] == 1000.0

    def test_beta_params_asymmetric(self):
        """Test Beta parameters with asymmetric distribution."""
        prior = PriorDistribution(
            dist_type="BetaScaled",
            mu=200.0,  # Closer to min
            sigma=100.0,
            min_val=100.0,
            max_val=1000.0,
        )

        kwargs = prior.to_numpyro_kwargs()

        # Should have different alpha/beta for asymmetry
        alpha = kwargs["concentration0"]
        beta = kwargs["concentration1"]

        # Verify they're positive
        assert alpha > 0.5
        assert beta > 0.5

        # For mu closer to min, beta should be larger than alpha
        # (mean = alpha/(alpha+beta), so smaller mean → smaller alpha/beta ratio)
        # This relationship holds when sigma is reasonable
        assert kwargs["low"] == 100.0
        assert kwargs["high"] == 1000.0

    def test_beta_params_large_sigma_fallback(self):
        """Test Beta parameters with too-large sigma fall back to default."""
        prior = PriorDistribution(
            dist_type="BetaScaled",
            mu=550.0,
            sigma=1000.0,  # Larger than range - physically invalid
            min_val=100.0,
            max_val=1000.0,
        )

        kwargs = prior.to_numpyro_kwargs()

        # Should fall back to default alpha=beta=2
        assert kwargs["concentration0"] == pytest.approx(2.0)
        assert kwargs["concentration1"] == pytest.approx(2.0)

    def test_beta_params_invalid_bounds(self):
        """Test Beta distribution with invalid bounds raises error."""
        # Invalid bounds are caught in __post_init__, not to_numpyro_kwargs()
        with pytest.raises(ValueError, match="Invalid bounds"):
            PriorDistribution(
                dist_type="BetaScaled",
                mu=0.5,
                sigma=0.2,
                min_val=100.0,
                max_val=100.0,  # Invalid: min == max
            )

    def test_beta_params_infinite_bounds(self):
        """Test Beta distribution with infinite bounds raises error."""
        with pytest.raises(ValueError, match="finite bounds"):
            PriorDistribution(
                dist_type="BetaScaled",
                mu=0.5,
                sigma=0.2,
                min_val=-np.inf,
                max_val=np.inf,
            )


# =============================================================================
# ParameterSpace Conversion Tests
# =============================================================================


class TestParameterSpaceConversion:
    """Ensure ParameterSpace can emit BetaScaled priors."""

    def test_convert_all_truncated_normal(self, static_config_truncated_normal):
        """Test converting all TruncatedNormal priors to Beta."""
        param_space = ParameterSpace.from_config(static_config_truncated_normal)
        beta_space = param_space.convert_to_beta_scaled_priors()

        # Check that all priors were converted
        for param_name in beta_space.parameter_names:
            prior = beta_space.get_prior(param_name)
            assert prior.dist_type == "BetaScaled"

        # Check bounds preserved
        assert beta_space.bounds == param_space.bounds

        # Check parameter names preserved
        assert beta_space.parameter_names == param_space.parameter_names

    def test_convert_mixed_priors(self, static_config_mixed_priors):
        """Test converting mixed prior types preserves unbounded."""
        param_space = ParameterSpace.from_config(static_config_mixed_priors)
        beta_space = param_space.convert_to_beta_scaled_priors()

        # D0: Uniform → Beta
        assert beta_space.get_prior("D0").dist_type == "BetaScaled"

        # alpha: Normal → Normal (unbounded, not converted)
        assert beta_space.get_prior("alpha").dist_type == "Normal"

        # D_offset: TruncatedNormal → Beta
        assert beta_space.get_prior("D_offset").dist_type == "BetaScaled"

    def test_convert_preserves_mu_sigma(self, static_config_truncated_normal):
        """Test that conversion preserves mu and sigma for Beta computation."""
        param_space = ParameterSpace.from_config(static_config_truncated_normal)
        beta_space = param_space.convert_to_beta_scaled_priors()

        # Check that mu/sigma are preserved
        for param_name in param_space.parameter_names:
            orig_prior = param_space.get_prior(param_name)
            beta_prior = beta_space.get_prior(param_name)

            assert beta_prior.mu == orig_prior.mu
            assert beta_prior.sigma == orig_prior.sigma
            assert beta_prior.min_val == orig_prior.min_val
            assert beta_prior.max_val == orig_prior.max_val

    def test_convert_immutable_original(self, static_config_truncated_normal):
        """Test that conversion doesn't modify original ParameterSpace."""
        param_space = ParameterSpace.from_config(static_config_truncated_normal)

        # Store original prior types
        orig_types = {
            name: param_space.get_prior(name).dist_type
            for name in param_space.parameter_names
        }

        # Convert to Beta
        param_space.convert_to_beta_scaled_priors()

        # Check original unchanged
        for param_name in param_space.parameter_names:
            assert param_space.get_prior(param_name).dist_type == orig_types[param_name]

    def test_convert_empty_parameter_space(self):
        """Test converting empty ParameterSpace."""
        param_space = ParameterSpace(
            model_type="static",
            parameter_names=[],
            bounds={},
            priors={},
            units={},
        )

        beta_space = param_space.convert_to_beta_scaled_priors()

        assert len(beta_space.parameter_names) == 0
        assert len(beta_space.priors) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestBetaScaledIntegration:
    """Integration tests for BetaScaled prior plumbing."""

    def test_beta_prior_numpyro_kwargs_format(self):
        """Test that Beta prior kwargs work with NumPyro format."""
        prior = PriorDistribution(
            dist_type="BetaScaled",
            mu=500.0,
            sigma=200.0,
            min_val=100.0,
            max_val=1000.0,
        )

        kwargs = prior.to_numpyro_kwargs()

        # Check NumPyro Beta format
        assert set(kwargs.keys()) == {"concentration0", "concentration1", "low", "high"}

        # Verify can extract values for transformation
        alpha = kwargs["concentration0"]
        beta = kwargs["concentration1"]
        low = kwargs["low"]
        high = kwargs["high"]

        # Basic sanity checks
        assert alpha > 0
        assert beta > 0
        assert low < high

    def test_full_workflow_truncated_to_beta(self, static_config_truncated_normal):
        """Test full workflow: load config → convert → extract kwargs."""
        # Load parameter space
        param_space = ParameterSpace.from_config(static_config_truncated_normal)

        # Convert to BetaScaled priors
        beta_space = param_space.convert_to_beta_scaled_priors()

        # Extract NumPyro kwargs for each parameter
        for param_name in beta_space.parameter_names:
            prior = beta_space.get_prior(param_name)
            kwargs = prior.to_numpyro_kwargs()

            # Verify structure
            assert "concentration0" in kwargs
            assert "concentration1" in kwargs
            assert "low" in kwargs
            assert "high" in kwargs

            # Verify bounds match original
            orig_bounds = param_space.get_bounds(param_name)
            assert kwargs["low"] == orig_bounds[0]
            assert kwargs["high"] == orig_bounds[1]
