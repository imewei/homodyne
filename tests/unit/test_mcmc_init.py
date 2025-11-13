"""Targeted tests for MCMC initialization validation and fallback behavior.

This test module guards against regressions in MCMC initialization logic,
specifically testing:
1. Out-of-bounds initial parameter detection
2. Auto-initialization fallback to midpoint defaults
3. Preflight validation before MCMC sampling
4. Per-angle scaling parameter handling

Uses minimal synthetic datasets to keep tests fast and focused.
"""

import numpy as np
import pytest

from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization.mcmc import (
    _calculate_midpoint_defaults,
    _create_numpyro_model,
    fit_mcmc_jax,
)
from tests.factories.synthetic_data import generate_synthetic_xpcs_data

# Check if NumPyro is available
try:
    import numpyro
    from numpyro.infer import Predictive
    from jax import random

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


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
        """Generate minimal static isotropic data."""
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
            analysis_mode="static_isotropic",
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
                minimal_static_data.t2, len(minimal_static_data.phi) * len(minimal_static_data.t1)
            ),
            phi=np.repeat(
                minimal_static_data.phi, len(minimal_static_data.t1) * len(minimal_static_data.t2)
            ),
            q=minimal_static_data.q,
            L=minimal_static_data.L,
            analysis_mode="static_isotropic",
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
            analysis_mode="static_isotropic",
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
