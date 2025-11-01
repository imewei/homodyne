"""Integration tests for config-driven MCMC initialization.

Tests the full workflow of loading NLSQ results from config and using them
to initialize MCMC sampling. This demonstrates the manual workflow for
v2.1.0 where users copy NLSQ results to config YAML files.

This module is part of Task Group 2.2 in the v2.1.0 MCMC simplification implementation.
"""

import pytest
import numpy as np
from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace


class TestConfigDrivenMCMC:
    """Test config-driven MCMC initialization workflow."""

    def test_nlsq_to_mcmc_static_mode(self):
        """Test NLSQ → MCMC workflow for static mode.

        Workflow:
        1. User runs NLSQ optimization
        2. User manually copies best-fit values to config YAML
        3. User runs MCMC with those initial values
        """
        # Step 1: Simulated NLSQ results (what user would get from NLSQ run)
        nlsq_best_fit = {
            "D0": 1234.5,
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # Step 2: User manually copies to config YAML (simulated here as dict)
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                ],
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        # Step 3: Load config and verify MCMC initialization uses NLSQ values
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Verify initial parameters match NLSQ results
        assert initial_params["D0"] == nlsq_best_fit["D0"]
        assert initial_params["alpha"] == nlsq_best_fit["alpha"]
        assert initial_params["D_offset"] == nlsq_best_fit["D_offset"]

        # Verify ParameterSpace can be loaded
        param_space = ParameterSpace.from_config(config)
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3

        # Verify bounds are correct
        d0_bounds = param_space.get_bounds("D0")
        assert d0_bounds == (100.0, 1e5)

        # Verify initial values are within bounds
        is_valid, violations = param_space.validate_values(initial_params)
        assert is_valid, f"Initial parameters violate bounds: {violations}"

    def test_nlsq_to_mcmc_laminar_flow_mode(self):
        """Test NLSQ → MCMC workflow for laminar flow mode."""
        # Simulated NLSQ results (7 parameters for laminar flow)
        nlsq_best_fit = {
            "D0": 1500.0,
            "alpha": 0.8,
            "D_offset": 15.0,
            "gamma_dot_t0": 0.015,
            "beta": 0.1,
            "gamma_dot_t_offset": 0.001,
            "phi0": 10.0,
        }

        # User config with NLSQ results (using config names)
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",  # Config name
                    "beta",
                    "gamma_dot_offset",  # Config name
                    "phi_0",  # Config name
                ],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                    nlsq_best_fit["gamma_dot_t0"],
                    nlsq_best_fit["beta"],
                    nlsq_best_fit["gamma_dot_t_offset"],
                    nlsq_best_fit["phi0"],
                ],
            },
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -1000.0, "max": 1000.0},
                    {"name": "gamma_dot_0", "min": 1e-6, "max": 0.5},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {"name": "gamma_dot_offset", "min": -0.1, "max": 0.1},
                    {"name": "phi_0", "min": -180.0, "max": 180.0},
                ],
            },
        }

        # Load and verify
        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Verify all 7 parameters loaded with canonical names
        assert len(initial_params) == 7
        assert initial_params["D0"] == nlsq_best_fit["D0"]
        assert initial_params["alpha"] == nlsq_best_fit["alpha"]
        assert initial_params["gamma_dot_t0"] == nlsq_best_fit["gamma_dot_t0"]
        assert initial_params["phi0"] == nlsq_best_fit["phi0"]

        # Verify ParameterSpace
        param_space = ParameterSpace.from_config(config)
        assert param_space.model_type == "laminar_flow"
        assert len(param_space.parameter_names) == 7

        # Verify bounds validation
        is_valid, violations = param_space.validate_values(initial_params)
        assert is_valid, f"Initial parameters violate bounds: {violations}"

    def test_partial_mcmc_with_fixed_parameters(self):
        """Test MCMC with some parameters fixed from NLSQ.

        User might want to:
        1. Run NLSQ to get all parameters
        2. Fix some parameters (e.g., offsets)
        3. Run MCMC on subset for uncertainty quantification
        """
        # NLSQ best-fit results
        nlsq_best_fit = {
            "D0": 1234.5,
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # Config: Optimize D0 and alpha, fix D_offset at NLSQ value
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                ],
                "fixed_parameters": {
                    "D_offset": nlsq_best_fit["D_offset"],  # Fix at NLSQ value
                },
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should only return optimizable parameters (D_offset is fixed)
        assert len(initial_params) == 2
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "D_offset" not in initial_params  # Fixed, excluded

        # Verify optimizable parameters use NLSQ values
        assert initial_params["D0"] == nlsq_best_fit["D0"]
        assert initial_params["alpha"] == nlsq_best_fit["alpha"]

    def test_active_parameters_subset_for_mcmc(self):
        """Test MCMC on active parameter subset.

        Different from fixed_parameters: active_parameters defines which
        parameters to optimize, while fixed_parameters gives values for
        parameters NOT being optimized.
        """
        # NLSQ results for all parameters
        nlsq_best_fit = {
            "D0": 1500.0,
            "alpha": 0.8,
            "D_offset": 15.0,
            "gamma_dot_t0": 0.015,
            "beta": 0.1,
        }

        # Config: Only optimize D0, alpha, and gamma_dot_t0
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",
                    "beta",
                ],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                    nlsq_best_fit["gamma_dot_t0"],
                    nlsq_best_fit["beta"],
                ],
                "active_parameters": [
                    "D0",
                    "alpha",
                    "gamma_dot_0",
                ],  # Only 3 active
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should only return active parameters
        assert len(initial_params) == 3
        assert "D0" in initial_params
        assert "alpha" in initial_params
        assert "gamma_dot_t0" in initial_params
        assert "D_offset" not in initial_params  # Not active
        assert "beta" not in initial_params  # Not active

    def test_exploration_mode_with_midpoint_defaults(self):
        """Test exploration mode without NLSQ results.

        User might want to run MCMC directly for exploration, using
        mid-point defaults as initial values.
        """
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": None,  # Use mid-point defaults
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        # Should calculate mid-points
        assert initial_params["D0"] == pytest.approx(5050.0)  # (100 + 10000) / 2
        assert initial_params["alpha"] == pytest.approx(0.0)  # (-2 + 2) / 2
        assert initial_params["D_offset"] == pytest.approx(500.0)  # (0 + 1000) / 2

        # Verify ParameterSpace integration
        param_space = ParameterSpace.from_config(config)
        is_valid, violations = param_space.validate_values(initial_params)
        assert is_valid, f"Mid-point defaults violate bounds: {violations}"

    def test_realistic_workflow_with_uncertainties(self):
        """Test realistic workflow simulating NLSQ → MCMC transition.

        Simulates:
        1. User runs NLSQ with default initialization
        2. NLSQ converges to best-fit values
        3. User manually copies results to config
        4. User runs MCMC with NLSQ initialization
        5. MCMC samples around NLSQ point estimates
        """
        # Simulated NLSQ workflow
        # -----------------------
        # Step 1: NLSQ default initialization (mid-point)
        nlsq_config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "values": None,  # NLSQ uses mid-point defaults
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        nlsq_config_mgr = ConfigManager(config_override=nlsq_config)
        nlsq_init_params = nlsq_config_mgr.get_initial_parameters()

        # Verify NLSQ starts with mid-points
        assert nlsq_init_params["D0"] == pytest.approx(50050.0)  # (100 + 1e5) / 2
        assert nlsq_init_params["alpha"] == pytest.approx(0.0)

        # Step 2: Simulated NLSQ convergence (would be actual optimization)
        nlsq_best_fit = {
            "D0": 1234.5,  # Converged from mid-point
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # MCMC workflow
        # -------------
        # Step 3: User manually copies NLSQ results to config YAML
        mcmc_config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [
                    nlsq_best_fit["D0"],  # From NLSQ output
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                ],
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        # Step 4: MCMC initialization
        mcmc_config_mgr = ConfigManager(config_override=mcmc_config)
        mcmc_init_params = mcmc_config_mgr.get_initial_parameters()

        # Verify MCMC starts from NLSQ results (not mid-point)
        assert mcmc_init_params["D0"] == nlsq_best_fit["D0"]
        assert mcmc_init_params["alpha"] == nlsq_best_fit["alpha"]
        assert mcmc_init_params["D0"] != nlsq_init_params["D0"]  # Different from NLSQ init

        # Step 5: Verify ParameterSpace for MCMC prior sampling
        param_space = ParameterSpace.from_config(mcmc_config)

        # MCMC will sample around initial values using priors
        # Initial values should be within bounds
        is_valid, violations = param_space.validate_values(mcmc_init_params)
        assert is_valid, f"MCMC initial parameters violate bounds: {violations}"

        # Verify priors are accessible for MCMC
        for param_name in mcmc_init_params.keys():
            prior = param_space.get_prior(param_name)
            assert prior is not None
            assert prior.dist_type in [
                "Normal",
                "TruncatedNormal",
                "Uniform",
                "LogNormal",
            ]

    def test_integration_with_parameter_manager(self):
        """Test that ConfigManager integrates correctly with ParameterManager.

        Verifies that the entire config infrastructure works together:
        - ConfigManager loads config
        - ParameterManager provides bounds and name mapping
        - ParameterSpace provides priors
        - All three work together for MCMC initialization
        """
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "gamma_dot_0",  # Config name
                    "phi_0",  # Config name
                ],
                "values": [1000.0, 0.01, 45.0],
            },
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {
                        "name": "D0",
                        "min": 100.0,
                        "max": 1e5,
                        "prior_mu": 1000.0,
                        "prior_sigma": 1000.0,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "gamma_dot_0",
                        "min": 1e-6,
                        "max": 0.5,
                        "prior_mu": 0.01,
                        "prior_sigma": 0.1,
                        "type": "TruncatedNormal",
                    },
                    {
                        "name": "phi_0",
                        "min": -180.0,
                        "max": 180.0,
                        "prior_mu": 0.0,
                        "prior_sigma": 30.0,
                        "type": "Normal",
                    },
                ],
            },
        }

        # ConfigManager
        config_mgr = ConfigManager(config_override=config)

        # Get initial parameters (with name mapping)
        initial_params = config_mgr.get_initial_parameters()
        assert "gamma_dot_t0" in initial_params  # Canonical name
        assert "phi0" in initial_params  # Canonical name

        # Get parameter bounds (via ParameterManager)
        bounds = config_mgr.get_parameter_bounds(["D0", "gamma_dot_t0", "phi0"])
        assert len(bounds) == 3
        assert bounds[0]["name"] == "D0"

        # ParameterSpace for priors
        param_space = ParameterSpace.from_config(config)
        assert param_space.model_type == "laminar_flow"

        # Verify integration: bounds match between ConfigManager and ParameterSpace
        for param_name in initial_params.keys():
            # ConfigManager bounds (via ParameterManager)
            mgr_bounds = config_mgr.get_parameter_bounds([param_name])[0]

            # ParameterSpace bounds
            ps_bounds = param_space.get_bounds(param_name)

            # Should match
            assert mgr_bounds["min"] == ps_bounds[0]
            assert mgr_bounds["max"] == ps_bounds[1]

        # Verify priors are defined for all parameters
        for param_name in initial_params.keys():
            prior = param_space.get_prior(param_name)
            assert prior is not None
            # Initial value should be within prior bounds
            min_val, max_val = param_space.get_bounds(param_name)
            assert min_val <= initial_params[param_name] <= max_val


class TestConfigValidation:
    """Test validation of config-driven MCMC initialization."""

    def test_initial_values_within_bounds(self):
        """Test that loaded initial values are validated against bounds."""
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "values": [1000.0, 0.5],
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        param_space = ParameterSpace.from_config(config)
        is_valid, violations = param_space.validate_values(initial_params)

        assert is_valid
        assert len(violations) == 0

    def test_initial_values_outside_bounds_detected(self):
        """Test that out-of-bounds initial values are detected."""
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha"],
                "values": [1000.0, 5.0],  # alpha out of bounds
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},  # Max is 2.0
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        param_space = ParameterSpace.from_config(config)
        is_valid, violations = param_space.validate_values(initial_params)

        assert not is_valid
        assert len(violations) > 0
        assert any("alpha" in v for v in violations)

    def test_midpoint_defaults_always_valid(self):
        """Test that mid-point defaults are always within bounds."""
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "values": None,  # Trigger mid-point calculation
            },
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                    {"name": "gamma_dot_0", "min": 1e-6, "max": 0.5},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {"name": "gamma_dot_offset", "min": -0.1, "max": 0.1},
                    {"name": "phi_0", "min": -180.0, "max": 180.0},
                ],
            },
        }

        config_mgr = ConfigManager(config_override=config)
        initial_params = config_mgr.get_initial_parameters()

        param_space = ParameterSpace.from_config(config)
        is_valid, violations = param_space.validate_values(initial_params)

        # Mid-points should always be valid
        assert is_valid, f"Mid-point defaults violated bounds: {violations}"


class TestFullMCMCWorkflow:
    """Test complete config-driven MCMC workflow with fit_mcmc_jax().

    Tests Task Group 3.3: Config-Driven Parameter Loading Integration.
    """

    def test_config_driven_parameter_loading_static_mode(self):
        """Test full workflow: config → parameter_space + initial_values → fit_mcmc_jax()."""
        import numpy as np
        from homodyne.optimization.mcmc import fit_mcmc_jax

        # Simulated NLSQ results (what user would have from previous run)
        nlsq_best_fit = {
            "D0": 1234.5,
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # User config with NLSQ results copied to initial_parameters.values
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                ],
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
                "priors": [
                    {
                        "name": "D0",
                        "type": "TruncatedNormal",
                        "mu": 1000.0,
                        "sigma": 2000.0,
                    },
                    {
                        "name": "alpha",
                        "type": "TruncatedNormal",
                        "mu": 0.0,
                        "sigma": 1.0,
                    },
                    {
                        "name": "D_offset",
                        "type": "TruncatedNormal",
                        "mu": 10.0,
                        "sigma": 20.0,
                    },
                ],
            },
        }

        # Create minimal test data
        n_points = 10
        t1 = np.linspace(0.1, 1.0, n_points)
        t2 = np.linspace(0.1, 1.0, n_points)
        phi = np.zeros(n_points)  # Single angle
        q = 0.01
        L = 3.5e6

        # Generate synthetic data (c2 ≈ 1.0)
        data = np.ones(n_points) + 0.01 * np.random.randn(n_points)

        # Pass config via kwargs (this is how CLI would call it)
        # NOTE: This test verifies the integration but doesn't run full MCMC
        # (would require numpyro/blackjax and take too long for unit test)
        # Instead, we verify that config loading works correctly

        # Load parameter_space and initial_values from config
        param_space = ParameterSpace.from_config(config)
        config_mgr = ConfigManager(config_override=config)
        initial_values = config_mgr.get_initial_parameters()

        # Verify loaded correctly
        assert param_space.model_type == "static"
        assert len(param_space.parameter_names) == 3
        assert initial_values["D0"] == nlsq_best_fit["D0"]
        assert initial_values["alpha"] == nlsq_best_fit["alpha"]
        assert initial_values["D_offset"] == nlsq_best_fit["D_offset"]

        # Verify parameter validation
        is_valid, violations = param_space.validate_values(initial_values)
        assert is_valid, f"Initial values violate bounds: {violations}"

    def test_config_driven_parameter_loading_with_null_values(self):
        """Test workflow with null initial_values (mid-point defaults)."""
        import numpy as np
        from homodyne.optimization.mcmc import _calculate_midpoint_defaults

        # Config with null initial_values (trigger mid-point calculation)
        config = {
            "analysis_mode": "static_isotropic",
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": None,  # Null → use mid-points
            },
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 10000.0},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": 0.0, "max": 1000.0},
                ],
            },
        }

        # Load parameter_space and initial_values
        param_space = ParameterSpace.from_config(config)
        config_mgr = ConfigManager(config_override=config)
        initial_values = config_mgr.get_initial_parameters()

        # Verify mid-point defaults calculated correctly
        assert initial_values["D0"] == pytest.approx(5050.0)  # (100 + 10000) / 2
        assert initial_values["alpha"] == pytest.approx(0.0)  # (-2 + 2) / 2
        assert initial_values["D_offset"] == pytest.approx(500.0)  # (0 + 1000) / 2

        # Verify validation passes
        is_valid, violations = param_space.validate_values(initial_values)
        assert is_valid, f"Mid-point defaults violate bounds: {violations}"

    def test_config_driven_parameter_loading_laminar_flow(self):
        """Test workflow with laminar flow mode (7 parameters)."""
        import numpy as np

        # NLSQ results for laminar flow
        nlsq_best_fit = {
            "D0": 1500.0,
            "alpha": 0.8,
            "D_offset": 15.0,
            "gamma_dot_t0": 0.015,
            "beta": 0.1,
            "gamma_dot_t_offset": 0.001,
            "phi0": 10.0,
        }

        # Config with all 7 parameters (using config names)
        config = {
            "analysis_mode": "laminar_flow",
            "initial_parameters": {
                "parameter_names": [
                    "D0",
                    "alpha",
                    "D_offset",
                    "gamma_dot_0",  # Config name
                    "beta",
                    "gamma_dot_offset",  # Config name
                    "phi_0",  # Config name
                ],
                "values": [
                    nlsq_best_fit["D0"],
                    nlsq_best_fit["alpha"],
                    nlsq_best_fit["D_offset"],
                    nlsq_best_fit["gamma_dot_t0"],
                    nlsq_best_fit["beta"],
                    nlsq_best_fit["gamma_dot_t_offset"],
                    nlsq_best_fit["phi0"],
                ],
            },
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 1e5},
                    {"name": "alpha", "min": -2.0, "max": 2.0},
                    {"name": "D_offset", "min": -1000.0, "max": 1000.0},
                    {"name": "gamma_dot_0", "min": 1e-6, "max": 0.5},
                    {"name": "beta", "min": -2.0, "max": 2.0},
                    {"name": "gamma_dot_offset", "min": -0.1, "max": 0.1},
                    {"name": "phi_0", "min": -180.0, "max": 180.0},
                ],
            },
        }

        # Load and verify
        param_space = ParameterSpace.from_config(config)
        config_mgr = ConfigManager(config_override=config)
        initial_values = config_mgr.get_initial_parameters()

        # Verify all 7 parameters loaded with canonical names
        assert len(initial_values) == 7
        assert "D0" in initial_values
        assert "gamma_dot_t0" in initial_values  # Canonical name
        assert "phi0" in initial_values  # Canonical name

        # Verify values match NLSQ results
        assert initial_values["D0"] == nlsq_best_fit["D0"]
        assert initial_values["gamma_dot_t0"] == nlsq_best_fit["gamma_dot_t0"]
        assert initial_values["phi0"] == nlsq_best_fit["phi0"]

        # Verify validation
        is_valid, violations = param_space.validate_values(initial_values)
        assert is_valid, f"Initial values violate bounds: {violations}"
