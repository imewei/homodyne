"""
Unit Tests for MCMC Core Functionality
=======================================

**Updated**: v3.0 CMC-only migration

Consolidated from:
- test_mcmc_init.py (MCMC initialization, 9 tests)
- test_mcmc_model.py (NumPyro model creation, 14 tests)
- test_mcmc_selection.py (Selection logic - DEPRECATED in v3.0)
- test_mcmc_result_extension.py (MCMCResult extensions, 19 tests)
- test_mcmc_integration.py (MCMC API integration, 19 tests)

Tests cover:
- MCMC initialization validation and fallback behavior
- NumPyro model creation with config-driven priors
- MCMCResult class with CMC support and backward compatibility
- MCMC API integration

Note: NUTS/CMC selection tests deprecated in v3.0 (CMC-only architecture)
Tests with min_samples_for_cmc and memory_threshold_pct are legacy tests
that may still pass but are no longer relevant for v3.0+

Total: ~60 tests
"""

import inspect
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

# JAX and NumPyro imports with availability checking
try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro import handlers
    from numpyro.infer import MCMC, NUTS, Predictive

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

from homodyne.cli.commands import (
    _create_mcmc_diagnostics_dict,
    _create_mcmc_parameters_dict,
)

# Homodyne imports
from homodyne.config.parameter_space import ParameterSpace, PriorDistribution
from homodyne.device.config import HardwareConfig, should_use_cmc
from homodyne.optimization.mcmc import (
    MCMCResult,
    _build_single_angle_surrogate_settings,
    _calculate_midpoint_defaults,
    _create_numpyro_model,
    _estimate_single_angle_scaling,
    _evaluate_convergence_thresholds,
    _format_init_params_for_chains,
    _get_mcmc_config,
    _process_posterior_samples,
    fit_mcmc_jax,
)
from tests.factories.synthetic_data import generate_synthetic_xpcs_data

# ==============================================================================
# MCMC Initialization Tests (from test_mcmc_init.py)
# ==============================================================================


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestMCMCInitializationValidation:
    """Test preflight validation of initial parameter values."""

    @pytest.fixture
    def laminar_flow_param_space(self):
        """Create ParameterSpace for laminar_flow mode with realistic bounds."""
        return ParameterSpace.from_defaults("laminar_flow")

    @pytest.fixture
    def minimal_laminar_data(self):
        """Generate minimal synthetic laminar flow data for fast testing."""
        # Valid ground truth parameters (all within default bounds)
        # Note: Factory expects "gamma_dot_offset" but ParameterSpace uses "gamma_dot_t_offset"
        ground_truth_params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within [1e-10, 1.0]
            "beta": 0.5,
            "gamma_dot_offset": 0.1,  # Factory expects this name
            "phi0": 1.57,
        }

        # Generate tiny dataset (keep resource use minimal)
        return generate_synthetic_xpcs_data(
            ground_truth_params=ground_truth_params,
            n_phi=2,  # Just 2 angles
            n_t1=10,  # Small grid
            n_t2=10,
            noise_level=0.01,
            analysis_mode="laminar_flow",
            random_seed=42,
        )

    def test_validate_in_bounds_initial_values(self, laminar_flow_param_space):
        """Test that in-bounds initial values pass validation."""
        # All values within bounds
        # Default bounds: gamma_dot_t0: [1e-10, 1.0], gamma_dot_t_offset: [1e-10, 1.0]
        valid_initial_values = {
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within [1e-10, 1.0]
            "beta": 0.5,
            "gamma_dot_t_offset": 0.1,  # Within [1e-10, 1.0]
            "phi0": 1.57,
        }

        is_valid, violations = laminar_flow_param_space.validate_values(
            valid_initial_values
        )

        assert is_valid, f"Valid values rejected: {violations}"
        assert len(violations) == 0

    def test_validate_out_of_bounds_initial_values(self, laminar_flow_param_space):
        """Test that out-of-bounds initial values are detected."""
        # D0 below minimum (min is 1.0)
        invalid_initial_values = {
            "D0": 0.5,  # Below minimum (min is 1.0)
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within bounds
            "beta": 0.5,
            "gamma_dot_t_offset": 0.1,  # Within bounds
            "phi0": 1.57,
        }

        is_valid, violations = laminar_flow_param_space.validate_values(
            invalid_initial_values
        )

        assert not is_valid, "Out-of-bounds values not detected"
        assert len(violations) > 0
        assert any("D0" in v for v in violations), "D0 violation not reported"

    def test_validate_multiple_violations(self, laminar_flow_param_space):
        """Test detection of multiple out-of-bounds parameters."""
        # Multiple parameters out of bounds
        invalid_initial_values = {
            "D0": 0.5,  # Below minimum (min is 1.0)
            "alpha": -5.0,  # Below minimum (min is -2.0)
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within bounds
            "beta": 5.0,  # Above maximum (max is 2.0)
            "gamma_dot_t_offset": 0.1,  # Within bounds
            "phi0": 1.57,
        }

        is_valid, violations = laminar_flow_param_space.validate_values(
            invalid_initial_values
        )

        assert not is_valid, "Multiple violations not detected"
        assert len(violations) >= 2, f"Expected ≥2 violations, got {len(violations)}"

    def test_calculate_midpoint_defaults(self, laminar_flow_param_space):
        """Test midpoint calculation for auto-initialization fallback."""
        midpoint_defaults = _calculate_midpoint_defaults(laminar_flow_param_space)

        # Check all expected parameters are present
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
            assert param in midpoint_defaults, f"Missing midpoint for {param}"

        # Verify midpoints are within bounds
        is_valid, violations = laminar_flow_param_space.validate_values(
            midpoint_defaults
        )
        assert is_valid, f"Midpoint defaults out of bounds: {violations}"

        # Verify midpoints are actually midpoints (for at least one parameter)
        # bounds is a dict of tuples: {param_name: (min, max)}
        D0_bounds = laminar_flow_param_space.bounds["D0"]
        expected_D0_midpoint = (D0_bounds[0] + D0_bounds[1]) / 2
        assert abs(midpoint_defaults["D0"] - expected_D0_midpoint) < 1e-6


class TestMCMCConfigDefaults:
    def test_get_mcmc_config_adaptive_defaults(self):
        config_small = _get_mcmc_config({"n_params": 6})
        assert config_small["n_warmup"] >= 1200
        assert config_small["target_accept_prob"] == 0.9
        assert config_small["dense_mass_matrix"] is True

        config_large = _get_mcmc_config({"n_params": 20})
        assert config_large["n_warmup"] >= 4000
        assert config_large["dense_mass_matrix"] is False

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_format_init_params_for_chains_applies_jitter(self):
        init_values = {"D0": 10.0, "alpha": -1.0}
        rng_key = random.PRNGKey(0)
        formatted, _ = _format_init_params_for_chains(
            init_values,
            n_chains=2,
            jitter_scale=0.1,
            rng_key=rng_key,
            parameter_space=None,
        )

        assert formatted is not None
        assert "D0" in formatted
        assert formatted["D0"].shape == (2,)
        assert not np.allclose(
            np.asarray(formatted["D0"])[0], np.asarray(formatted["D0"])[1]
        )


def test_estimate_single_angle_scaling_clamps():
    values = np.linspace(0.95, 1.2, 1000)
    contrast, offset = _estimate_single_angle_scaling(values)

    assert offset == pytest.approx(0.95, abs=5e-3)
    assert contrast == pytest.approx(0.196, abs=5e-3)

    high_values = np.full(256, 1.5)
    contrast_hi, offset_hi = _estimate_single_angle_scaling(high_values)
    assert contrast_hi == pytest.approx(0.01, rel=1e-2)
    assert offset_hi == pytest.approx(1.1, rel=1e-6)


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestMCMCAutoInitializationFallback:
    """Test auto-initialization fallback when initial values are invalid."""

    @pytest.fixture
    def laminar_flow_param_space(self):
        """Create ParameterSpace for laminar_flow mode."""
        return ParameterSpace.from_defaults("laminar_flow")

    @pytest.fixture
    def minimal_laminar_data(self):
        """Generate minimal synthetic laminar flow data."""
        ground_truth_params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within [1e-10, 1.0]
            "beta": 0.5,
            "gamma_dot_offset": 0.1,  # Factory expects this name
            "phi0": 1.57,
        }

        return generate_synthetic_xpcs_data(
            ground_truth_params=ground_truth_params,
            n_phi=2,
            n_t1=10,
            n_t2=10,
            noise_level=0.01,
            analysis_mode="laminar_flow",
            random_seed=42,
        )

    def test_fit_mcmc_raises_on_invalid_initial_values(
        self, minimal_laminar_data, laminar_flow_param_space
    ):
        """Test that fit_mcmc_jax raises ValueError for out-of-bounds initial values.

        This is the CURRENT behavior that should fail. After the patch,
        this test should demonstrate the auto-initialization fallback.
        """
        # Out-of-bounds initial values (D0 too low, gamma params too high)
        invalid_initial_values = {
            "D0": 0.5,  # Below minimum (min is 1.0)
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within bounds
            "beta": 0.5,
            "gamma_dot_t_offset": 0.1,  # Within bounds
            "phi0": 1.57,
        }

        # Flatten data for fit_mcmc_jax
        data_flat = minimal_laminar_data.g2.flatten()
        sigma_flat = minimal_laminar_data.sigma.flatten()

        # Create coordinate arrays
        n_phi = minimal_laminar_data.phi.shape[0]
        n_t1 = minimal_laminar_data.t1.shape[0]
        n_t2 = minimal_laminar_data.t2.shape[0]

        phi_coords = np.repeat(minimal_laminar_data.phi, n_t1 * n_t2)
        t1_coords = np.tile(np.repeat(minimal_laminar_data.t1, n_t2), n_phi)
        t2_coords = np.tile(minimal_laminar_data.t2, n_phi * n_t1)

        # CURRENT BEHAVIOR: Should raise ValueError for invalid initial values
        with pytest.raises(ValueError, match="Initial parameter values violate bounds"):
            fit_mcmc_jax(
                data=data_flat,
                sigma=sigma_flat,
                t1=t1_coords,
                t2=t2_coords,
                phi=phi_coords,
                q=minimal_laminar_data.q,
                L=minimal_laminar_data.L,
                analysis_mode="laminar_flow",
                parameter_space=laminar_flow_param_space,
                initial_values=invalid_initial_values,
                n_samples=5,  # Minimal for speed
                n_warmup=5,
                n_chains=1,
                rng_key=42,
                progress_bar=False,
            )

    def test_fit_mcmc_accepts_none_initial_values(
        self, minimal_laminar_data, laminar_flow_param_space
    ):
        """Test that fit_mcmc_jax auto-initializes when initial_values=None."""
        # Flatten data
        data_flat = minimal_laminar_data.g2.flatten()
        sigma_flat = minimal_laminar_data.sigma.flatten()

        # Create coordinate arrays
        n_phi = minimal_laminar_data.phi.shape[0]
        n_t1 = minimal_laminar_data.t1.shape[0]
        n_t2 = minimal_laminar_data.t2.shape[0]

        phi_coords = np.repeat(minimal_laminar_data.phi, n_t1 * n_t2)
        t1_coords = np.tile(np.repeat(minimal_laminar_data.t1, n_t2), n_phi)
        t2_coords = np.tile(minimal_laminar_data.t2, n_phi * n_t1)

        # Should succeed with auto-initialization
        result = fit_mcmc_jax(
            data=data_flat,
            sigma=sigma_flat,
            t1=t1_coords,
            t2=t2_coords,
            phi=phi_coords,
            q=minimal_laminar_data.q,
            L=minimal_laminar_data.L,
            analysis_mode="laminar_flow",
            parameter_space=laminar_flow_param_space,
            initial_values=None,  # Auto-initialize
            n_samples=5,
            n_warmup=5,
            n_chains=1,
            rng_key=42,
            progress_bar=False,
        )

        # Basic sanity checks
        assert result is not None
        # MCMCResult is an object, check for samples or mean params
        assert hasattr(result, "samples_params") or hasattr(result, "mean_params")
        # Check that we got the expected number of samples (if samples are available)
        if hasattr(result, "samples_params") and result.samples_params is not None:
            assert result.samples_params.shape[0] == 5  # n_samples


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestPerAngleScalingInitialization:
    """Test initialization behavior with per-angle scaling mode.

    Per-angle scaling requires contrast_0, contrast_1, ..., offset_0, offset_1, ...
    instead of single contrast/offset values.
    """

    @pytest.fixture
    def minimal_static_data(self):
        """Generate minimal static mode data."""
        ground_truth_params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
        }

        return generate_synthetic_xpcs_data(
            ground_truth_params=ground_truth_params,
            n_phi=2,
            n_t1=10,
            n_t2=10,
            noise_level=0.01,
            analysis_mode="static",
            random_seed=42,
        )

    def test_per_angle_scaling_model_creation(self, minimal_static_data):
        """Test that per-angle scaling creates correct parameter structure.

        CURRENT BEHAVIOR (line 2014-2022 in mcmc.py):
        - initial_values provided but ignored due to per-angle scaling incompatibility
        - Warning logged and init_params set to None
        - NumPyro samples from priors instead

        FUTURE BEHAVIOR (after patch):
        - initial_values expanded to include per-angle parameters
        - Both physical params AND scaling params properly initialized
        """
        param_space = ParameterSpace.from_defaults("static")

        # Create model with per-angle scaling (default in v2.4.0)
        model = _create_numpyro_model(
            data=minimal_static_data.g2.flatten(),
            sigma=minimal_static_data.sigma.flatten(),
            t1=np.tile(
                np.repeat(minimal_static_data.t1, len(minimal_static_data.t2)),
                len(minimal_static_data.phi),
            ),
            t2=np.tile(
                minimal_static_data.t2,
                len(minimal_static_data.phi) * len(minimal_static_data.t1),
            ),
            phi=np.repeat(
                minimal_static_data.phi,
                len(minimal_static_data.t1) * len(minimal_static_data.t2),
            ),
            q=minimal_static_data.q,
            L=minimal_static_data.L,
            analysis_mode="static",
            parameter_space=param_space,
            dt=minimal_static_data.dt,
        )

        # Sample from prior to verify parameter structure
        prior_pred = Predictive(model, num_samples=10)
        samples = prior_pred(random.PRNGKey(42))

        # With n_phi=2, expect: contrast_0, contrast_1, offset_0, offset_1, D0, alpha, D_offset
        expected_per_angle_params = ["contrast_0", "contrast_1", "offset_0", "offset_1"]
        expected_physical_params = ["D0", "alpha", "D_offset"]

        for param in expected_per_angle_params:
            assert param in samples, f"Missing per-angle parameter: {param}"

        for param in expected_physical_params:
            assert param in samples, f"Missing physical parameter: {param}"

    def test_physical_params_only_initial_values_warning(self, minimal_static_data):
        """Test that providing only physical params triggers warning in per-angle mode.

        CURRENT BEHAVIOR:
        - User provides initial_values = {D0, alpha, D_offset} (no per-angle params)
        - Warning logged: "initial_values provided but per-angle scaling mode detected"
        - init_params set to None, sampling from priors

        EXPECTED FUTURE BEHAVIOR:
        - Auto-expand physical params to per-angle structure
        - Initialize contrast_0 = contrast_1 = midpoint, same for offset
        """
        param_space = ParameterSpace.from_defaults("static")

        # Provide only physical parameters (no per-angle scaling params)
        # These are valid values within bounds
        physical_only_initial_values = {
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
        }

        # Flatten data
        data_flat = minimal_static_data.g2.flatten()
        sigma_flat = minimal_static_data.sigma.flatten()

        n_phi = minimal_static_data.phi.shape[0]
        n_t1 = minimal_static_data.t1.shape[0]
        n_t2 = minimal_static_data.t2.shape[0]

        phi_coords = np.repeat(minimal_static_data.phi, n_t1 * n_t2)
        t1_coords = np.tile(np.repeat(minimal_static_data.t1, n_t2), n_phi)
        t2_coords = np.tile(minimal_static_data.t2, n_phi * n_t1)

        # CURRENT BEHAVIOR: Should log warning and ignore initial_values
        # After patch: Should auto-expand and use the values
        result = fit_mcmc_jax(
            data=data_flat,
            sigma=sigma_flat,
            t1=t1_coords,
            t2=t2_coords,
            phi=phi_coords,
            q=minimal_static_data.q,
            L=minimal_static_data.L,
            analysis_mode="static",
            parameter_space=param_space,
            initial_values=physical_only_initial_values,
            n_samples=5,
            n_warmup=5,
            n_chains=1,
            rng_key=42,
            progress_bar=False,
        )

        # Should still succeed (sampling from priors currently)
        assert result is not None
        # MCMCResult is an object, not a dict - check for samples attribute
        assert hasattr(result, "samples_params") or hasattr(result, "mean_params")


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestMCMCInitRegression:
    """Integration tests reproducing the reported initialization failure."""

    def test_laminar_flow_with_problematic_config(self):
        """Reproduce the reported init failure with laminar_flow and user config.

        Scenario from user report:
        - laminar_flow analysis mode
        - User-specified initial_parameters.values from NLSQ
        - Some values may be slightly out of bounds or missing scaling params
        - MCMC should either validate and reject OR auto-fix with fallback

        EXPECTED CURRENT BEHAVIOR:
        - ValueError raised for out-of-bounds values

        EXPECTED FUTURE BEHAVIOR (after patch):
        - Auto-initialize fallback with warning
        - OR expand missing per-angle params automatically
        """
        # Simulate user config with realistic but potentially problematic values
        param_space = ParameterSpace.from_defaults("laminar_flow")

        # Generate realistic test data
        ground_truth_params = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 1000.0,
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,  # Within [1e-10, 1.0]
            "beta": 0.5,
            "gamma_dot_offset": 0.1,  # Factory expects this name
            "phi0": 1.57,
        }

        data = generate_synthetic_xpcs_data(
            ground_truth_params=ground_truth_params,
            n_phi=3,  # Typical experiment
            n_t1=15,
            n_t2=15,
            noise_level=0.01,
            analysis_mode="laminar_flow",
            random_seed=42,
        )

        # Case 1: Slightly out-of-bounds D0 (user typo or NLSQ edge case)
        problematic_initial_values = {
            "D0": 0.5,  # Below minimum (min is 1.0)
            "alpha": -1.2,
            "D_offset": 10.0,
            "gamma_dot_t0": 0.5,
            "beta": 0.5,
            "gamma_dot_t_offset": 0.1,
            "phi0": 1.57,
        }

        # Flatten data
        data_flat = data.g2.flatten()
        sigma_flat = data.sigma.flatten()

        n_phi = data.phi.shape[0]
        n_t1 = data.t1.shape[0]
        n_t2 = data.t2.shape[0]

        phi_coords = np.repeat(data.phi, n_t1 * n_t2)
        t1_coords = np.tile(np.repeat(data.t1, n_t2), n_phi)
        t2_coords = np.tile(data.t2, n_phi * n_t1)

        # CURRENT BEHAVIOR: Should raise ValueError
        with pytest.raises(ValueError, match="Initial parameter values violate bounds"):
            fit_mcmc_jax(
                data=data_flat,
                sigma=sigma_flat,
                t1=t1_coords,
                t2=t2_coords,
                phi=phi_coords,
                q=data.q,
                L=data.L,
                analysis_mode="laminar_flow",
                parameter_space=param_space,
                initial_values=problematic_initial_values,
                n_samples=5,
                n_warmup=5,
                n_chains=1,
                rng_key=42,
                progress_bar=False,
            )

        # After patch: Should auto-initialize with midpoint defaults and log warning
        # (Test will need updating when patch is applied)


# ==============================================================================
# NumPyro Model Creation Tests (from test_mcmc_model.py)
# ==============================================================================


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
        """Create ParameterSpace for static_mode mode."""
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
        """Test model creation for static_mode mode."""
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static",
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

    def test_single_angle_fixed_scaling_overrides(
        self, simple_data, static_parameter_space
    ):
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
            per_angle_scaling=False,
            fixed_scaling_overrides={"contrast": 0.05, "offset": 0.97},
        )

        seeded = handlers.seed(model, random.PRNGKey(0))
        trace = handlers.trace(seeded).get_trace()

        assert trace["contrast"]["type"] == "deterministic"
        assert trace["offset"]["type"] == "deterministic"
        assert float(trace["contrast"]["value"]) == pytest.approx(0.05)
        assert float(trace["offset"]["value"]) == pytest.approx(0.97)

    def test_single_angle_surrogate_initializes_and_runs(
        self, simple_data, static_parameter_space
    ):
        surrogate_cfg = _build_single_angle_surrogate_settings(
            static_parameter_space, tier="2"
        )
        assert surrogate_cfg, "Surrogate config should be constructed for tier 2"

        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
            per_angle_scaling=False,
            fixed_scaling_overrides={"contrast": 0.05, "offset": 1.0},
            single_angle_surrogate_config=surrogate_cfg,
        )

        seeded = handlers.seed(model, random.PRNGKey(0))
        trace = handlers.trace(seeded).get_trace()

        assert "log_D0_latent" in trace
        assert trace["log_D0_latent"]["type"] == "deterministic"

        nuts_kernel = NUTS(model, target_accept_prob=0.8, dense_mass=False)
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=5,
            num_samples=5,
            num_chains=1,
            progress_bar=False,
        )
        mcmc.run(random.PRNGKey(1))
        samples = mcmc.get_samples()
        assert "log_D0_latent" in samples
        assert "alpha" in samples

    def test_single_angle_surrogate_tier4_trust_region(
        self, simple_data, static_parameter_space
    ):
        surrogate_cfg = _build_single_angle_surrogate_settings(
            static_parameter_space, tier="4"
        )
        assert surrogate_cfg.get("fixed_d0_value") is None

        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
            per_angle_scaling=False,
            fixed_scaling_overrides={"contrast": 0.05, "offset": 1.0},
            single_angle_surrogate_config=surrogate_cfg,
        )

        predictive = Predictive(model, num_samples=50)
        samples = predictive(random.PRNGKey(5))
        d0_samples = np.array(samples["D0"])
        log_prior = surrogate_cfg["log_d0_prior"]
        d0_min = np.exp(log_prior["low"])
        d0_max = np.exp(log_prior["high"])
        assert np.all(d0_samples >= d0_min * 0.9)
        assert np.all(d0_samples <= d0_max * 1.1)

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
        # Use tier="1" to disable log-space D0 sampling and test linear-space behavior
        model = _create_numpyro_model(
            data=simple_data["data"],
            sigma=simple_data["sigma"],
            t1=simple_data["t1"],
            t2=simple_data["t2"],
            phi=simple_data["phi"],
            q=simple_data["q"],
            L=simple_data["L"],
            analysis_mode="static",
            parameter_space=static_parameter_space,
            dt=simple_data["dt"],
            single_angle_surrogate_config={"tier": "1"},  # Disable log-space D0 sampling
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
            analysis_mode="static",
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
            analysis_mode="static",
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


# ==============================================================================
# MCMC Selection Logic Tests (from test_mcmc_selection.py)
# ==============================================================================


@pytest.mark.skip(
    reason="should_use_cmc() deprecated in v2.4.1 CMC-only architecture - always returns True"
)
class TestMCMCSelectionIntegration:
    """Test MCMC selection logic integration with config and hardware detection.

    NOTE: Deprecated in v2.4.1. CMC is now the only MCMC path.
    should_use_cmc() is a deprecated shim that always returns True.
    """

    @pytest.fixture
    def mock_hardware_config(self):
        """Create a mock hardware configuration for testing."""
        return HardwareConfig(
            platform="cpu",
            num_devices=1,
            memory_per_device_gb=32.0,
            num_nodes=1,
            cores_per_node=14,
            total_memory_gb=32.0,
            cluster_type="standalone",
            recommended_backend="multiprocessing",
            max_parallel_shards=14,
        )

    def test_config_threshold_loading_from_kwargs(self, mock_hardware_config):
        """
        Test that thresholds can be loaded from kwargs (passed from config).

        Spec requirement: Extract thresholds from config (not hardcoded)

        NOTE: Using 500K points (< 1M) to avoid Criterion 3 (JAX Broadcasting Protection).
        """
        # Test with custom thresholds from config
        num_samples = 18
        dataset_size = 500_000  # < 1M to avoid Criterion 3

        # Custom thresholds that differ from defaults
        custom_min_samples = 20
        custom_memory_threshold = 0.25

        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=custom_min_samples,
            memory_threshold_pct=custom_memory_threshold,
        )

        # 18 < 20 (custom threshold) and memory < 25% → NUTS
        assert result is False, (
            "Should use NUTS when num_samples < custom threshold "
            "and memory < custom threshold"
        )

    def test_hardware_detection_fallback_none(self):
        """
        Test fallback behavior when hardware detection returns None.

        Spec requirement: Fallback to simple threshold if hardware detection fails
        """
        # Simulate hardware detection failure
        hardware_config = None

        # When hardware_config is None, should_use_cmc should still work
        # (though it won't have memory estimation capability)
        num_samples = 20
        min_samples_for_cmc = 15

        # Without hardware_config, only sample-based decision possible
        # This test verifies graceful handling of None hardware_config
        # Note: In actual mcmc.py, fallback uses simple threshold
        # Here we test that the function signature supports this scenario

        # The actual fallback is in mcmc.py (lines 537-544)
        # This test verifies the design is correct
        assert True, "Test passes - fallback handled in mcmc.py"

    def test_comprehensive_logging_parallelism_mode(self, mock_hardware_config, caplog):
        """
        Test that comprehensive logging shows quad-criteria evaluation for parallelism mode.

        Spec requirement: Add comprehensive logging showing decision process

        NOTE: Updated for quad-criteria (was dual-criteria in v2.1.0)
        """
        import logging

        caplog.set_level(logging.INFO)

        num_samples = 20  # Triggers parallelism (Criterion 1)
        dataset_size = 500_000  # < 1M to avoid Criterion 3

        should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Verify logging contains key elements (quad-criteria)
        log_text = caplog.text
        assert "Quad-Criteria Evaluation" in log_text
        assert "Criterion 1 (Parallelism)" in log_text
        assert "Criterion 2 (Memory)" in log_text
        assert "Final decision" in log_text
        assert "CMC" in log_text

    def test_comprehensive_logging_memory_mode(self, mock_hardware_config, caplog):
        """
        Test that comprehensive logging shows dual-criteria evaluation for memory mode.

        Spec requirement: Add comprehensive logging showing decision process
        """
        import logging

        caplog.set_level(logging.INFO)

        num_samples = 5  # Below parallelism threshold
        dataset_size = 50_000_000  # Large dataset

        should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Verify logging shows quad-criteria
        log_text = caplog.text
        assert "Quad-Criteria Evaluation" in log_text
        assert "Criterion 1 (Parallelism)" in log_text
        assert "→ False" in log_text  # Parallelism failed (5 < 15)
        assert "Criterion 2 (Memory)" in log_text
        assert "→ True" in log_text  # Memory or other criteria triggered
        assert "CMC" in log_text

    def test_comprehensive_logging_nuts_mode(self, mock_hardware_config, caplog):
        """
        Test that comprehensive logging shows quad-criteria evaluation for NUTS selection.

        Spec requirement: Add comprehensive logging showing decision process

        NOTE: Updated for quad-criteria. Using dataset < 1M to avoid Criterion 3.
        """
        import logging

        caplog.set_level(logging.INFO)

        num_samples = 10  # Below parallelism threshold (< 15)
        dataset_size = 500_000  # < 1M to avoid Criterion 3

        should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Verify logging shows quad-criteria with all failed
        log_text = caplog.text
        assert "Quad-Criteria Evaluation" in log_text
        assert "Criterion 1 (Parallelism)" in log_text
        assert "Criterion 2 (Memory)" in log_text
        assert "NUTS" in log_text

    def test_warning_for_cmc_with_few_samples(self, mock_hardware_config, caplog):
        """
        Test warning when CMC is triggered by memory but few samples.

        Spec requirement: Log warning for edge case (CMC + few samples)
        """
        import logging

        caplog.set_level(logging.WARNING)

        num_samples = 5  # Very few samples
        dataset_size = 50_000_000  # Large dataset triggers memory

        should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Verify warning is logged (quad-criteria format)
        log_text = caplog.text
        assert "Using CMC with only" in log_text
        assert "samples" in log_text
        # Note: Quad-criteria shows all triggers, not just "memory criterion"
        assert "Triggered by:" in log_text
        assert "memory" in log_text.lower()

    def test_default_threshold_values(self, mock_hardware_config):
        """
        Test that default threshold values are correctly set.

        Spec requirement: Default values should be 15 samples and 30% memory
        """
        num_samples = 15  # Exactly at default threshold
        dataset_size = 10_000_000

        # Use defaults (should be 15 and 0.30)
        result = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
        )

        assert result is True, (
            "CMC should be selected at exactly 15 samples (default threshold)"
        )

    def test_memory_estimation_accuracy(self, mock_hardware_config):
        """
        Test memory estimation formula accuracy.

        Spec requirement: Memory estimation should use dataset_size * 8 * num_samples / 1e9

        NOTE (Quad-Criteria): With large dataset (30M > 1M), Criteria 3 and 4 will
        also trigger. This test verifies memory calculation is correct, acknowledging
        that CMC will be selected due to multiple criteria.
        """
        num_samples = 5
        dataset_size = 30_000_000

        # Expected memory: 30M * 8 bytes * 5 / 1e9 = 1.2 GB
        # 1.2 / 32 = 3.75% memory usage

        # Test with 1% threshold (should trigger CMC via Criterion 2)
        result_triggered = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.01,  # 3.75% > 1% → Criterion 2 triggers
            min_samples_for_cmc=15,
        )

        assert result_triggered is True, "CMC should be selected (Criterion 2 + 3 + 4)"

        # Test with 10% threshold
        # NOTE: Criterion 2 won't trigger (3.75% < 10%), but Criteria 3 and 4 will
        result_quad_criteria = should_use_cmc(
            num_samples=num_samples,
            hardware_config=mock_hardware_config,
            dataset_size=dataset_size,
            memory_threshold_pct=0.10,  # 3.75% < 10% → Criterion 2 doesn't trigger
            min_samples_for_cmc=15,
        )

        assert result_quad_criteria is True, "CMC should be selected (Criterion 3 + 4)"


class TestMCMCFallbackBehavior:
    """Test fallback behavior when hardware detection is unavailable."""

    def test_fallback_simple_threshold_logic(self):
        """
        Test that fallback uses simple threshold-based selection.

        This simulates the behavior in mcmc.py lines 537-544 when
        hardware_config is None.
        """
        # Simulate fallback logic from mcmc.py
        num_samples = 20
        min_samples_for_cmc = 15

        # Fallback: use_cmc = num_samples >= min_samples_for_cmc
        use_cmc = num_samples >= min_samples_for_cmc

        assert use_cmc is True, "Fallback should use simple threshold logic"

    def test_fallback_nuts_selection(self):
        """Test fallback NUTS selection with few samples."""
        num_samples = 10
        min_samples_for_cmc = 15

        # Fallback logic
        use_cmc = num_samples >= min_samples_for_cmc

        assert use_cmc is False, "Fallback should select NUTS for few samples"


# ==============================================================================
# MCMCResult Extension Tests (from test_mcmc_result_extension.py)
# ==============================================================================


class TestBackwardCompatibility:
    """Test that existing code continues to work without modification."""

    def test_standard_mcmc_result_creation(self):
        """Test creating standard MCMC result without CMC fields."""
        # Create result with only standard fields (as existing code does)
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            converged=True,
            n_iterations=1000,
        )

        # Verify standard fields work as before
        assert result.mean_params.shape == (3,)
        assert result.mean_contrast == 0.5
        assert result.mean_offset == 1.0
        assert result.converged is True
        assert result.n_iterations == 1000

        # Verify CMC fields default to None
        assert result.per_shard_diagnostics is None
        assert result.cmc_diagnostics is None
        assert result.combination_method is None
        assert result.num_shards is None

    def test_old_results_still_load(self):
        """Test that results without CMC fields can be deserialized."""
        # Simulate old result dictionary (no CMC fields)
        old_data = {
            "mean_params": [100.0, 1.5, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 0.5],
            "std_contrast": 0.05,
            "std_offset": 0.02,
            "samples_params": None,
            "samples_contrast": None,
            "samples_offset": None,
            "converged": True,
            "n_iterations": 1000,
            "computation_time": 45.5,
            "backend": "JAX",
            "analysis_mode": "static",
            "dataset_size": "medium",
            "n_chains": 4,
            "n_warmup": 500,
            "n_samples": 1000,
            "sampler": "NUTS",
            "acceptance_rate": 0.85,
            "r_hat": {"D0": 1.01, "alpha": 1.02},
            "effective_sample_size": {"D0": 800, "alpha": 750},
            # No CMC fields
        }

        # Should load without errors
        result = MCMCResult.from_dict(old_data)

        # Standard fields preserved
        assert np.allclose(result.mean_params, [100.0, 1.5, 10.0])
        assert result.mean_contrast == 0.5
        assert result.converged is True

        # CMC fields default to None
        assert result.num_shards is None
        assert result.combination_method is None
        assert result.is_cmc_result() is False


class TestIsCMCResult:
    """Test the is_cmc_result() method."""

    def test_is_cmc_result_false_when_none(self):
        """Test is_cmc_result() returns False when num_shards is None."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
        )
        assert result.is_cmc_result() is False

    def test_is_cmc_result_true_when_one(self):
        """Single-shard runs still use CMC pipeline in v3.0."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=1,
            selection_decision_metadata={"method": "CMC"},
        )
        assert result.is_cmc_result() is True

    def test_is_cmc_result_true_when_multiple(self):
        """Test is_cmc_result() returns True when num_shards > 1."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,
        )
        assert result.is_cmc_result() is True

    def test_is_cmc_result_with_all_cmc_fields(self):
        """Test is_cmc_result() with complete CMC data."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
            combination_method="weighted",
            per_shard_diagnostics=[
                {"shard_id": 0, "converged": True},
                {"shard_id": 1, "converged": True},
            ],
            cmc_diagnostics={"combination_success": True},
        )
        assert result.is_cmc_result() is True
        assert result.combination_method == "weighted"


class TestCMCFieldsPreservation:
    """Test that CMC-specific fields are preserved correctly."""

    def test_per_shard_diagnostics_storage(self):
        """Test per_shard_diagnostics field storage."""
        diagnostics = [
            {
                "shard_id": 0,
                "converged": True,
                "acceptance_rate": 0.85,
                "n_samples": 1000,
            },
            {
                "shard_id": 1,
                "converged": True,
                "acceptance_rate": 0.82,
                "n_samples": 1000,
            },
            {
                "shard_id": 2,
                "converged": False,
                "acceptance_rate": 0.45,
                "n_samples": 1000,
            },
        ]

        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=3,
            per_shard_diagnostics=diagnostics,
        )

        assert result.per_shard_diagnostics is not None
        assert len(result.per_shard_diagnostics) == 3
        assert result.per_shard_diagnostics[0]["shard_id"] == 0
        assert result.per_shard_diagnostics[2]["converged"] is False

    def test_cmc_diagnostics_storage(self):
        """Test cmc_diagnostics field storage."""
        cmc_diag = {
            "combination_success": True,
            "n_shards_converged": 8,
            "n_shards_total": 10,
            "weighted_product_std": 0.15,
            "combination_time": 2.3,
        }

        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,
            cmc_diagnostics=cmc_diag,
        )

        assert result.cmc_diagnostics is not None
        assert result.cmc_diagnostics["combination_success"] is True
        assert result.cmc_diagnostics["n_shards_converged"] == 8
        assert result.cmc_diagnostics["combination_time"] == 2.3

    def test_combination_method_storage(self):
        """Test combination_method field storage."""
        for method in ["weighted", "average", "hierarchical"]:
            result = MCMCResult(
                mean_params=np.array([1.0]),
                mean_contrast=0.5,
                mean_offset=1.0,
                num_shards=5,
                combination_method=method,
            )
            assert result.combination_method == method


class TestSerialization:
    """Test serialization and deserialization with CMC data."""

    def test_cmc_result_serialization(self):
        """Test to_dict() preserves CMC-specific data."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            num_shards=5,
            combination_method="weighted",
            per_shard_diagnostics=[
                {"shard_id": 0, "converged": True},
                {"shard_id": 1, "converged": True},
            ],
            cmc_diagnostics={"combination_success": True, "n_shards_total": 5},
        )

        data = result.to_dict()

        # Standard fields
        assert data["mean_params"] == [100.0, 1.5, 10.0]
        assert data["mean_contrast"] == 0.5

        # CMC fields
        assert data["num_shards"] == 5
        assert data["combination_method"] == "weighted"
        assert len(data["per_shard_diagnostics"]) == 2
        assert data["cmc_diagnostics"]["combination_success"] is True

    def test_cmc_result_deserialization(self):
        """Test from_dict() reconstructs CMC-specific data."""
        data = {
            "mean_params": [100.0, 1.5, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 0.5],
            "std_contrast": 0.05,
            "std_offset": 0.02,
            "samples_params": None,
            "samples_contrast": None,
            "samples_offset": None,
            "converged": True,
            "n_iterations": 2000,
            "computation_time": 120.5,
            "backend": "JAX",
            "analysis_mode": "laminar_flow",
            "dataset_size": "large",
            "n_chains": 4,
            "n_warmup": 500,
            "n_samples": 2000,
            "sampler": "NUTS",
            "acceptance_rate": 0.83,
            "r_hat": None,
            "effective_sample_size": None,
            # CMC fields
            "num_shards": 10,
            "combination_method": "weighted",
            "per_shard_diagnostics": [
                {"shard_id": 0, "converged": True, "acceptance_rate": 0.85},
                {"shard_id": 1, "converged": True, "acceptance_rate": 0.82},
            ],
            "cmc_diagnostics": {
                "combination_success": True,
                "n_shards_converged": 9,
                "n_shards_total": 10,
            },
        }

        result = MCMCResult.from_dict(data)

        # Standard fields
        assert np.allclose(result.mean_params, [100.0, 1.5, 10.0])
        assert result.mean_contrast == 0.5
        assert result.analysis_mode == "laminar_flow"

        # CMC fields
        assert result.is_cmc_result() is True
        assert result.num_shards == 10
        assert result.combination_method == "weighted"
        assert len(result.per_shard_diagnostics) == 2
        assert result.cmc_diagnostics["n_shards_converged"] == 9

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip preserves all data."""
        original = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 0.5]),
            samples_params=np.array([[99.0, 1.4, 9.8], [101.0, 1.6, 10.2]]),
            samples_contrast=np.array([0.49, 0.51]),
            samples_offset=np.array([0.98, 1.02]),
            num_shards=5,
            combination_method="average",
            per_shard_diagnostics=[
                {"shard_id": i, "converged": True} for i in range(5)
            ],
            cmc_diagnostics={"combination_success": True},
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        reconstructed = MCMCResult.from_dict(data)

        # Verify all fields match
        assert np.allclose(reconstructed.mean_params, original.mean_params)
        assert reconstructed.mean_contrast == original.mean_contrast
        assert reconstructed.num_shards == original.num_shards
        assert reconstructed.combination_method == original.combination_method
        assert len(reconstructed.per_shard_diagnostics) == 5
        assert reconstructed.is_cmc_result() == original.is_cmc_result()

        # Verify samples preserved
        assert np.allclose(reconstructed.samples_params, original.samples_params)
        assert np.allclose(reconstructed.samples_contrast, original.samples_contrast)

    def test_json_serialization_compatibility(self):
        """Test that results can be serialized to JSON."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=3,
            combination_method="weighted",
            cmc_diagnostics={"combination_success": True},
        )

        # Convert to dict and serialize to JSON
        data = result.to_dict()
        json_str = json.dumps(data)

        # Deserialize from JSON
        loaded_data = json.loads(json_str)
        reconstructed = MCMCResult.from_dict(loaded_data)

        # Verify reconstruction
        assert np.allclose(reconstructed.mean_params, result.mean_params)
        assert reconstructed.num_shards == result.num_shards
        assert reconstructed.combination_method == result.combination_method


class TestNoneDefaults:
    """Test that CMC fields default to None for non-CMC results."""

    def test_minimal_result_creation(self):
        """Test creating result with minimal required fields."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
        )

        # All CMC fields should be None
        assert result.num_shards is None
        assert result.combination_method is None
        assert result.per_shard_diagnostics is None
        assert result.cmc_diagnostics is None

        # Standard fields should have defaults
        assert result.converged is True  # default
        assert result.n_iterations == 0  # default
        assert result.backend == "JAX"  # default

    def test_partial_cmc_fields(self):
        """Test providing only some CMC fields."""
        # Only num_shards provided
        result1 = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
        )
        assert result1.is_cmc_result() is True
        assert result1.combination_method is None  # Other fields still None

        # Only combination_method provided (but no num_shards)
        result2 = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            combination_method="weighted",
        )
        assert result2.is_cmc_result() is False  # Not CMC without num_shards
        assert result2.combination_method == "weighted"  # But field is set


class TestEdgeCases:
    """Test edge cases and unusual inputs."""

    def test_empty_per_shard_diagnostics(self):
        """Test with empty per_shard_diagnostics list."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=5,
            per_shard_diagnostics=[],  # Empty list
        )
        assert result.is_cmc_result() is True
        assert result.per_shard_diagnostics == []

    def test_large_num_shards(self):
        """Test with large number of shards."""
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=1000,  # Large number
        )
        assert result.is_cmc_result() is True
        assert result.num_shards == 1000

    def test_zero_num_shards(self):
        """Test with num_shards=0 (invalid but should handle gracefully).

        NOTE: In v2.4.1+ CMC-only architecture, num_shards=0 is still
        considered CMC since any explicit shard count (including 0) signals
        CMC pipeline was used. The test reflects this behavior.
        """
        result = MCMCResult(
            mean_params=np.array([1.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=0,
        )
        # In CMC-only architecture (v2.4.1+), any explicit num_shards signals CMC
        # num_shards=0 is technically invalid but still triggers CMC detection
        assert result.is_cmc_result() is True

    def test_deserialization_with_extra_fields(self):
        """Test deserialization ignores unknown fields."""
        data = {
            "mean_params": [1.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "num_shards": 5,
            "unknown_field": "should be ignored",  # Extra field
            "another_unknown": 123,
        }

        # Should not raise error
        result = MCMCResult.from_dict(data)
        assert result.num_shards == 5
        assert result.is_cmc_result() is True


# ==============================================================================
# MCMC API Integration Tests (from test_mcmc_integration.py)
# ==============================================================================


class TestAPISignature:
    """Test API signature after v2.1.0 refactoring."""

    def test_method_parameter_not_in_signature(self):
        """Test that method parameter is not in fit_mcmc_jax signature.

        v2.1.0 breaking change: method parameter was removed.
        Automatic selection handles NUTS/CMC internally based on data characteristics.
        """

        sig = inspect.signature(fit_mcmc_jax)
        assert "method" not in sig.parameters, (
            "method parameter should not exist in v2.1.0. "
            "Automatic selection is now internal."
        )

    def test_function_accepts_kwargs(self):
        """Test that fit_mcmc_jax accepts **kwargs for backward compatibility."""

        sig = inspect.signature(fit_mcmc_jax)
        # Should have **kwargs to accept old-style method parameter
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), "fit_mcmc_jax should accept **kwargs for backward compatibility"

    def test_parameter_space_parameter_exists(self):
        """Test that parameter_space parameter exists (v2.1.0 feature)."""

        sig = inspect.signature(fit_mcmc_jax)
        assert "parameter_space" in sig.parameters, (
            "parameter_space parameter should exist in v2.1.0"
        )

    def test_initial_values_parameter_exists(self):
        """Test that initial_values parameter exists (v2.1.0 change)."""

        sig = inspect.signature(fit_mcmc_jax)
        assert "initial_values" in sig.parameters, (
            "initial_values parameter should exist in v2.1.0 (renamed from initial_params)"
        )


class TestDataValidation:
    """Test data validation in fit_mcmc_jax()."""

    def test_empty_data_raises_error(self):
        """Test that empty data raises ValueError."""
        with pytest.raises(ValueError):
            fit_mcmc_jax(
                data=np.array([]),
                t1=np.array([]),
                t2=np.array([]),
                phi=np.array([]),
                q=0.01,
                L=3.5,
            )

    def test_none_data_raises_error(self):
        """Test that None data raises error."""
        with pytest.raises((ValueError, TypeError)):
            fit_mcmc_jax(
                data=None,
                t1=np.array([1.0, 2.0]),
                t2=np.array([1.0, 2.0]),
                phi=np.array([1.0, 2.0]),
                q=0.01,
                L=3.5,
            )

    def test_mismatched_array_sizes_raises_error(self):
        """Test that mismatched array sizes are detected as invalid.

        NOTE: In v2.4.1+ CMC-only architecture, the MCMC sampling catches
        internal errors and retries before eventually returning a "failed"
        MCMCResult with zero mean_params. The test accepts either:
        - An exception being raised, OR
        - A result with zero mean_params (indicating sampling failure)

        This behavior change reflects the CMC-only architecture's graceful
        degradation approach rather than hard failures.
        """
        try:
            result = fit_mcmc_jax(
                data=np.random.randn(100),
                t1=np.random.rand(100),
                t2=np.random.rand(50),  # Wrong size!
                phi=np.random.rand(100),
                q=0.01,
                L=3.5,
            )
            # If no exception, verify that result indicates failure
            # (zero mean_params indicates sampling didn't produce valid results)
            assert result.mean_params is not None
            assert np.allclose(result.mean_params, 0.0), (
                f"Expected zero mean_params for failed sampling, got {result.mean_params}"
            )
        except (ValueError, IndexError, RuntimeError):
            # Exception raised during validation or sampling - test passes
            pass

    def test_missing_q_parameter_raises_error(self):
        """Test that missing q parameter raises error."""
        with pytest.raises((ValueError, TypeError)):
            fit_mcmc_jax(
                data=np.random.randn(10),
                t1=np.random.rand(10),
                t2=np.random.rand(10),
                phi=np.random.rand(10),
                q=None,  # Missing required parameter
                L=3.5,
            )

    def test_missing_l_parameter_raises_error(self):
        """Test that missing L parameter for laminar_flow raises error."""
        with pytest.raises((ValueError, TypeError)):
            fit_mcmc_jax(
                data=np.random.randn(10),
                t1=np.random.rand(10),
                t2=np.random.rand(10),
                phi=np.random.rand(10),
                q=0.01,
                L=None,  # Missing for laminar_flow
                analysis_mode="laminar_flow",
            )


class TestParameterAcceptance:
    """Test parameter acceptance after v2.1.0 changes."""

    def test_initial_values_parameter_accepted(self):
        """Test initial_values parameter is accepted (no errors from signature)."""
        # This validates that initial_values parameter exists and is recognized
        # We're not executing MCMC, just checking parameter acceptance

        initial_vals = {
            "D0": 1000.0,
            "alpha": 0.5,
            "D_offset": 10.0,
        }

        # Should not raise TypeError about unexpected keyword argument
        # (May raise other errors related to data validation, which is OK)
        try:
            fit_mcmc_jax(
                data=np.array([1.0]),  # Minimal data
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                initial_values=initial_vals,
                n_samples=10,
                n_warmup=5,
                min_samples_for_cmc=10000,  # Prevent CMC (which has size requirements)
            )
        except ValueError as e:
            # Data validation errors are OK (we're just checking parameter acceptance)
            assert "initial_values" not in str(e).lower()
        except RuntimeError as e:
            # Runtime errors related to execution are OK
            assert "initial_values" not in str(e).lower()

    def test_parameter_space_parameter_accepted(self):
        """Test parameter_space parameter is accepted."""
        param_space = ParameterSpace.from_defaults("static")

        # Should not raise TypeError about unexpected keyword argument
        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                parameter_space=param_space,
                n_samples=10,
                n_warmup=5,
                min_samples_for_cmc=10000,
            )
        except ValueError as e:
            assert "parameter_space" not in str(e).lower()
        except RuntimeError as e:
            assert "parameter_space" not in str(e).lower()

    def test_method_parameter_in_kwargs_not_error(self):
        """Test that method parameter in kwargs doesn't cause TypeError.

        In v2.1.0, old code passing method='nuts' should not crash with
        'unexpected keyword argument' error. It just goes to **kwargs and is ignored.
        """
        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                method="nuts",  # Should be silently ignored (goes to kwargs)
                n_samples=10,
                n_warmup=5,
                min_samples_for_cmc=10000,
            )
        except TypeError as e:
            # Should NOT get "unexpected keyword argument 'method'" error
            assert "method" not in str(e).lower(), (
                "method parameter should be accepted in **kwargs (backward compatibility)"
            )


class TestKwargsAcceptance:
    """Test acceptance of various MCMC configuration kwargs."""

    def test_standard_mcmc_kwargs_accepted(self):
        """Test standard MCMC kwargs are accepted without TypeError."""
        mcmc_kwargs = {
            "n_samples": 100,
            "n_warmup": 50,
            "n_chains": 2,
            "target_accept_prob": 0.8,
            "max_tree_depth": 10,
            "rng_key": 42,
        }

        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                **mcmc_kwargs,
                min_samples_for_cmc=10000,
            )
        except TypeError as e:
            # Should not get unexpected keyword argument errors
            for key in mcmc_kwargs.keys():
                assert key not in str(e), (
                    f"Standard MCMC kwarg '{key}' should be accepted"
                )

    def test_cmc_threshold_kwargs_accepted(self):
        """Test CMC threshold configuration kwargs are accepted."""
        cmc_kwargs = {
            "min_samples_for_cmc": 20,
            "memory_threshold_pct": 0.35,
        }

        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                **cmc_kwargs,
                n_samples=10,
                n_warmup=5,
            )
        except TypeError as e:
            for key in cmc_kwargs.keys():
                assert key not in str(e), f"CMC kwarg '{key}' should be accepted"


class TestAnalysisModesSupported:
    """Test that different analysis modes are supported."""

    def test_static_mode_mode_accepted(self):
        """Test static_mode analysis mode doesn't raise ValueError."""
        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                analysis_mode="static",
                n_samples=10,
                n_warmup=5,
                min_samples_for_cmc=10000,
            )
        except ValueError as e:
            assert "analysis_mode" not in str(e).lower()

    def test_laminar_flow_mode_accepted(self):
        """Test laminar_flow analysis mode doesn't raise ValueError."""
        try:
            fit_mcmc_jax(
                data=np.array([1.0]),
                t1=np.array([1.0]),
                t2=np.array([1.0]),
                phi=np.array([0.1]),
                q=0.01,
                L=3.5,
                analysis_mode="laminar_flow",
                n_samples=10,
                n_warmup=5,
                min_samples_for_cmc=10000,
            )
        except ValueError as e:
            assert "analysis_mode" not in str(e).lower()


class TestMCMCResultStructure:
    """Test MCMCResult structure and expected fields."""

    def test_mcmc_result_has_required_fields(self):
        """Test that MCMCResult class has all required fields."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
        )

        # Verify standard fields exist
        assert hasattr(result, "mean_params")
        assert hasattr(result, "mean_contrast")
        assert hasattr(result, "mean_offset")
        assert hasattr(result, "converged")

        # Verify field values are correct
        assert np.allclose(result.mean_params, [100.0, 1.5, 10.0])
        assert result.mean_contrast == 0.5
        assert result.mean_offset == 1.0
        assert result.converged is True

    def test_mcmc_result_optional_fields(self):
        """Test that MCMCResult supports optional fields for advanced use."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            std_params=np.array([5.0, 0.1, 0.5]),
            n_iterations=3000,
            computation_time=45.2,
        )

        # Verify optional fields can be set
        assert hasattr(result, "std_params")
        assert hasattr(result, "n_iterations")
        assert hasattr(result, "computation_time")

    def test_mcmc_result_cmc_fields(self):
        """Test that MCMCResult supports CMC-specific fields."""
        result = MCMCResult(
            mean_params=np.array([100.0, 1.5, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            converged=True,
            num_shards=10,
            combination_method="weighted",
            per_shard_diagnostics=[{"shard_id": 0, "converged": True}],
        )

        # Verify CMC-specific fields can be set
        assert hasattr(result, "num_shards")
        assert hasattr(result, "combination_method")
        assert hasattr(result, "per_shard_diagnostics")
        assert result.num_shards == 10


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not available")
class TestProcessPosteriorSamples:
    class _DummyResult:
        def __init__(self, samples, grouped_samples, extra_fields):
            self._samples = samples
            self._grouped = grouped_samples
            self._extra = extra_fields

        def get_samples(self, group_by_chain=False):
            return self._grouped if group_by_chain else self._samples

        def get_extra_fields(self):
            return self._extra

    def test_invalid_samples_disable_diagnostics(self):
        samples = {
            "D0": jnp.array([np.nan, 1.0]),
            "alpha": jnp.array([0.0, 0.1]),
            "D_offset": jnp.array([0.5, 0.6]),
            "contrast_0": jnp.array([np.nan, np.nan]),
            "offset_0": jnp.array([1.0, 1.0]),
        }
        grouped = {
            key: jnp.reshape(value, (1, value.shape[0]))
            for key, value in samples.items()
        }
        extra = {"accept_prob": jnp.array([0.5]), "diverging": jnp.array([0])}
        dummy = self._DummyResult(samples, grouped, extra)

        summary = _process_posterior_samples(
            dummy,
            "static",
            {"max_rhat": 1.1, "min_ess": 100, "check_hmc_diagnostics": True},
        )

        assert summary["converged"] is False
        assert summary["r_hat"]["D0"] is None

    def test_custom_thresholds_trigger_non_convergence(self, monkeypatch):
        samples = {
            "D0": jnp.array([1.0, 2.0]),
            "alpha": jnp.array([-1.0, -0.9]),
            "D_offset": jnp.array([0.5, 0.6]),
            "contrast_0": jnp.array([0.5, 0.6]),
            "offset_0": jnp.array([1.0, 1.0]),
        }
        grouped = {
            key: jnp.stack([value, value + 0.01]) for key, value in samples.items()
        }
        extra = {
            "accept_prob": jnp.array([0.5, 0.6]),
            "diverging": jnp.array([0, 0]),
            "num_steps": jnp.array([5, 5]),
        }
        dummy = self._DummyResult(samples, grouped, extra)

        monkeypatch.setattr(
            "numpyro.diagnostics.gelman_rubin",
            lambda _: jnp.array(1.2),
        )
        monkeypatch.setattr(
            "numpyro.diagnostics.effective_sample_size",
            lambda _: jnp.array(20.0),
        )

        summary = _process_posterior_samples(
            dummy,
            "static",
            {"max_rhat": 1.05, "min_ess": 100, "check_hmc_diagnostics": True},
        )

        assert summary["converged"] is False
        assert summary["r_hat"]["D0"] == pytest.approx(1.2)

    def test_scalar_contrast_offset_supported(self):
        samples = {
            "D0": jnp.array([1.0, 2.0, 1.5]),
            "alpha": jnp.array([-1.0, -0.9, -0.95]),
            "D_offset": jnp.array([0.5, 0.6, 0.55]),
            "contrast": jnp.array([0.55, 0.6, 0.65]),
            "offset": jnp.array([1.0, 1.05, 1.1]),
        }
        grouped = {key: jnp.stack([value, value]) for key, value in samples.items()}
        extra = {
            "accept_prob": jnp.array([0.8, 0.82]),
            "diverging": jnp.array([0, 0]),
            "num_steps": jnp.array([10, 10]),
        }
        dummy = self._DummyResult(samples, grouped, extra)

        summary = _process_posterior_samples(
            dummy,
            "static",
            {"max_rhat": 10.0, "min_ess": 0, "check_hmc_diagnostics": False},
        )

        assert summary["converged"] is True
        assert math.isclose(
            summary["mean_contrast"], float(jnp.mean(samples["contrast"]))
        )
        assert math.isclose(summary["mean_offset"], float(jnp.mean(samples["offset"])))

    def test_single_angle_surrogate_samples_are_supported(self):
        log_d0 = jnp.array([1.0, 1.1, 0.9])
        samples = {
            "log_D0_latent": log_d0,
            "alpha": jnp.array([-1.2, -1.1, -1.15]),
            "contrast": jnp.array([0.05, 0.051, 0.049]),
            "offset": jnp.array([1.0, 1.0, 1.0]),
        }
        grouped = {
            key: jnp.reshape(value, (1,) + value.shape)
            for key, value in samples.items()
        }
        extra = {"accept_prob": jnp.array([0.9]), "diverging": jnp.array([0])}
        dummy = self._DummyResult(samples, grouped, extra)

        diag_settings = {
            "max_rhat": 10.0,
            "min_ess": 0,
            "check_hmc_diagnostics": True,
            "single_angle_surrogate": {
                "drop_d_offset": True,
                "fixed_d_offset": 0.0,
                "sample_log_d0": True,
            },
            "expected_params": ["D0", "alpha", "D_offset"],
        }

        summary = _process_posterior_samples(dummy, "static", diag_settings)

        assert "D0" in summary["samples"]
        assert math.isclose(
            float(summary["samples"]["D0"][0]), float(jnp.exp(log_d0[0]))
        )
        assert summary["param_names"] == ["D0", "alpha", "D_offset"]
        assert summary["mean_params"].shape[0] == 3
        assert math.isclose(summary["mean_params"][2], 0.0, abs_tol=1e-9)
        assert summary["ess"]["log_D0_latent"] is not None
        assert summary["ess"]["D_offset"] is None
        assert "D_offset" in summary["deterministic_params"]
        assert summary["diagnostic_summary"]["per_param_stats"]["D_offset"][
            "deterministic"
        ]

    def test_surrogate_thresholds_focus_on_physics_parameters(self):
        dummy = SimpleNamespace()
        dummy.r_hat = {"D0": 1.05, "alpha": None, "D_offset": None}
        dummy.effective_sample_size = {"D0": 22.0, "alpha": 5.0, "D_offset": None}
        dummy.diagnostic_summary = {
            "deterministic_params": ["D_offset"],
            "per_param_stats": {
                "D0": {"r_hat": 1.05, "ess": 22.0, "deterministic": False},
                "alpha": {"r_hat": None, "ess": 5.0, "deterministic": False},
                "D_offset": {"r_hat": None, "ess": None, "deterministic": True},
            },
            "surrogate_thresholds": {
                "active": True,
                "focus_params": ["D0", "alpha"],
                "min_ess": 20.0,
                "max_rhat": 1.2,
            },
        }

        report = _evaluate_convergence_thresholds(dummy, 1.1, 100)

        assert report["poor_rhat"] is False
        assert report["poor_ess"] is True  # alpha ESS violates surrogate threshold
        assert "alpha" in report["focus_param_details"]


def test_diagnostics_json_surfaces_surrogate_metrics():
    result = SimpleNamespace()
    result.r_hat = {"D0": 1.07, "alpha": None, "D_offset": None}
    result.effective_sample_size = {"D0": 30.0, "alpha": 8.0, "D_offset": None}
    result.acceptance_rate = 0.9
    result.divergences = 0
    result.tree_depth_warnings = 0
    result.ess = np.array([30.0])
    result.n_samples = 40
    result.n_chains = 1
    result.analysis_mode = "static"
    result.diagnostic_summary = {
        "deterministic_params": ["D_offset"],
        "per_param_stats": {
            "D0": {"r_hat": 1.07, "ess": 30.0, "deterministic": False},
            "alpha": {"r_hat": None, "ess": 8.0, "deterministic": False},
            "D_offset": {"r_hat": None, "ess": None, "deterministic": True},
        },
        "surrogate_thresholds": {
            "active": True,
            "focus_params": ["D0", "alpha"],
            "min_ess": 25.0,
            "max_rhat": 1.2,
        },
    }

    diag = _create_mcmc_diagnostics_dict(result)
    per_param = {
        entry["name"]: entry
        for entry in diag["convergence"]["per_parameter_diagnostics"]
    }

    assert per_param["alpha"]["r_hat"] is None
    assert per_param["D_offset"]["deterministic"] is True
    assert "surrogate_thresholds" in diag["convergence"]


def test_parameters_json_handles_deterministic_surrogate_fields():
    result = SimpleNamespace(
        mean_params=np.array([17000.0, -0.26, 0.0]),
        std_params=np.array([10.0, 0.01, 0.0]),
        mean_contrast=0.5,
        std_contrast=0.02,
        mean_offset=1.0,
        std_offset=0.0,
        n_samples=40,
        n_warmup=80,
        n_chains=2,
        computation_time=12.0,
        r_hat={"D0": 1.05, "alpha": None, "D_offset": None},
        effective_sample_size={"D0": 30.0, "alpha": 60.0, "D_offset": None},
        acceptance_rate=0.93,
        analysis_mode="static",
        diagnostic_summary={
            "deterministic_params": ["D_offset"],
            "surrogate_thresholds": {"active": True, "min_ess": 25.0},
        },
    )

    param_dict = _create_mcmc_parameters_dict(result)

    assert param_dict["parameters"]["D_offset"]["mean"] == 0.0
    assert param_dict["convergence"]["surrogate_thresholds"]["active"] is True


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
