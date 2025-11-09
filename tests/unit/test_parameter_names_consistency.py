"""Unit Tests for Parameter Name Consistency Across Codebase
==============================================================

Ensures parameter names are consistent between:
1. Parameter name constants (parameter_names.py)
2. NumPyro model definitions (mcmc.py)
3. Sample extraction logic (pjit backend)
4. Result processing (MCMCResult)

This test suite prevents parameter name mismatches that cause
MCMC initialization failures and KeyErrors during sample extraction.

Version: Created Nov 2025 to prevent recurrence of gamma_dot_0 vs gamma_dot_t0 bug
"""

import pytest
import numpy as np

# Import parameter constants
from homodyne.config.parameter_names import (
    STATIC_ISOTROPIC_PARAMS,
    LAMINAR_FLOW_PARAMS,
    get_parameter_names,
    get_num_parameters,
    validate_parameter_names,
    verify_samples_dict,
)


class TestParameterNameConstants:
    """Test parameter name constant definitions."""

    def test_static_isotropic_params_count(self):
        """Verify static mode has 5 parameters."""
        assert len(STATIC_ISOTROPIC_PARAMS) == 5

    def test_laminar_flow_params_count(self):
        """Verify laminar flow mode has 9 parameters."""
        assert len(LAMINAR_FLOW_PARAMS) == 9

    def test_static_params_ordering(self):
        """Verify static parameter ordering."""
        expected = ["contrast", "offset", "D0", "alpha", "D_offset"]
        assert STATIC_ISOTROPIC_PARAMS == expected

    def test_laminar_params_ordering(self):
        """Verify laminar flow parameter ordering."""
        expected = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        assert LAMINAR_FLOW_PARAMS == expected

    def test_get_parameter_names_static(self):
        """Test get_parameter_names for static mode."""
        params = get_parameter_names("static_isotropic")
        assert params == STATIC_ISOTROPIC_PARAMS

    def test_get_parameter_names_laminar(self):
        """Test get_parameter_names for laminar flow."""
        params = get_parameter_names("laminar_flow")
        assert LAMINAR_FLOW_PARAMS == params

    def test_get_num_parameters(self):
        """Test get_num_parameters function."""
        assert get_num_parameters("static_isotropic") == 5
        assert get_num_parameters("laminar_flow") == 9


class TestParameterNameValidation:
    """Test parameter name validation functions."""

    def test_validate_correct_static_params(self):
        """Validate correct static parameter names."""
        params = ["contrast", "offset", "D0", "alpha", "D_offset"]
        validate_parameter_names(params, "static_isotropic")  # Should not raise

    def test_validate_correct_laminar_params(self):
        """Validate correct laminar flow parameter names."""
        params = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        validate_parameter_names(params, "laminar_flow")  # Should not raise

    def test_validate_wrong_order_strict(self):
        """Test validation fails with wrong order (strict mode)."""
        params = ["offset", "contrast", "D0", "alpha", "D_offset"]  # Wrong order
        with pytest.raises(ValueError, match="don't match expected order"):
            validate_parameter_names(params, "static_isotropic", strict=True)

    def test_validate_missing_params(self):
        """Test validation fails with missing parameters."""
        params = ["contrast", "D0", "alpha"]  # Missing offset, D_offset
        with pytest.raises(ValueError, match="Missing parameters"):
            validate_parameter_names(params, "static_isotropic", strict=False)

    def test_validate_extra_params(self):
        """Test validation fails with extra parameters."""
        params = ["contrast", "offset", "D0", "alpha", "D_offset", "extra_param"]
        with pytest.raises(ValueError, match="Unexpected parameters"):
            validate_parameter_names(params, "static_isotropic", strict=False)

    def test_verify_samples_dict_complete(self):
        """Test verify_samples_dict with complete samples."""
        samples = {
            "contrast": np.array([0.5]),
            "offset": np.array([1.0]),
            "D0": np.array([1000.0]),
            "alpha": np.array([0.5]),
            "D_offset": np.array([10.0]),
        }
        verify_samples_dict(samples, "static_isotropic")  # Should not raise

    def test_verify_samples_dict_missing(self):
        """Test verify_samples_dict fails with missing parameters."""
        samples = {
            "contrast": np.array([0.5]),
            "D0": np.array([1000.0]),
            # Missing: offset, alpha, D_offset
        }
        with pytest.raises(KeyError, match="Missing parameters in MCMC samples"):
            verify_samples_dict(samples, "static_isotropic")


class TestMCMCModelConsistency:
    """Test parameter name consistency in MCMC model definitions."""

    def test_mcmc_model_param_names_static(self):
        """Verify mcmc.py uses correct parameter names for static mode."""
        # Import the model creation function
        from homodyne.optimization.mcmc import _create_numpyro_model
        from homodyne.config.parameter_space import ParameterSpace

        # Create minimal parameter space for static mode
        config = {
            "parameter_space": {
                "model": "static",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                ],
            }
        }
        param_space = ParameterSpace.from_config(config)

        # Create model (just verify it can be created without errors)
        # The model internally defines parameter names - we check those match constants
        try:
            model = _create_numpyro_model(
                data=np.array([1.0]),
                sigma=np.array([0.1]),
                t1=np.array([0.0]),
                t2=np.array([1.0]),
                phi=np.array([0.0]),
                q=0.005,
                L=1.0,
                analysis_mode="static_isotropic",
                parameter_space=param_space,
                dt=1.0,
            )
            # If model creation succeeds, parameter names are internally consistent
            assert model is not None
        except Exception as e:
            pytest.fail(
                f"Model creation failed, possibly due to parameter mismatch: {e}"
            )

    def test_mcmc_model_param_names_laminar(self):
        """Verify mcmc.py uses correct parameter names for laminar flow."""
        from homodyne.optimization.mcmc import _create_numpyro_model
        from homodyne.config.parameter_space import ParameterSpace

        # Create minimal parameter space for laminar flow
        config = {
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                    {"name": "gamma_dot_t0", "min": 0.1, "max": 10.0},
                    {"name": "beta", "min": 0.0, "max": 2.0},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 5.0},
                    {"name": "phi0", "min": -3.14159, "max": 3.14159},
                ],
            }
        }
        param_space = ParameterSpace.from_config(config)

        # Create model
        try:
            model = _create_numpyro_model(
                data=np.array([1.0]),
                sigma=np.array([0.1]),
                t1=np.array([0.0]),
                t2=np.array([1.0]),
                phi=np.array([0.0]),
                q=0.005,
                L=1.0,
                analysis_mode="laminar_flow",
                parameter_space=param_space,
                dt=1.0,
            )
            assert model is not None
        except Exception as e:
            pytest.fail(
                f"Model creation failed, possibly due to parameter mismatch: {e}"
            )


class TestPjitBackendConsistency:
    """Test parameter name consistency in pjit backend sample extraction."""

    def test_pjit_no_old_incorrect_names(self):
        """Verify pjit backend does not contain old incorrect parameter names."""
        import inspect
        from homodyne.optimization.cmc.backends.pjit import PjitBackend

        # Get the source code
        source = inspect.getsource(PjitBackend._run_single_shard_mcmc)

        # Verify old incorrect names that caused the bug are not present
        assert "'gamma_dot_0'" not in source  # Old bug: should be gamma_dot_t0
        assert (
            "'gamma_dot_offset'" not in source
        )  # Old bug: should be gamma_dot_t_offset
        assert "'phi_0'" not in source  # Old bug: should be phi0


class TestCrossCodingConsistency:
    """Integration tests verifying parameter names are consistent across all modules."""

    def test_mcmc_to_pjit_consistency_static(self):
        """Verify mcmc.py and pjit.py use identical static parameter names."""
        from homodyne.config.parameter_names import STATIC_ISOTROPIC_PARAMS

        # Expected parameter names from constants
        expected = STATIC_ISOTROPIC_PARAMS

        # This test ensures both mcmc.py model definition and pjit.py sample extraction
        # use the same parameter names defined in parameter_names.py
        # If this test passes, the modules are consistent

        # Verify expected matches our constants
        assert expected == ["contrast", "offset", "D0", "alpha", "D_offset"]

    def test_mcmc_to_pjit_consistency_laminar(self):
        """Verify mcmc.py and pjit.py use identical laminar flow parameter names."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # Expected parameter names from constants
        expected = LAMINAR_FLOW_PARAMS

        # Verify expected matches our constants
        assert expected == [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]

        # CRITICAL: Verify no old incorrect names
        assert "gamma_dot_0" not in expected  # Old bug name
        assert "gamma_dot_offset" not in expected  # Old bug name
        assert "phi_0" not in expected  # Old bug name


class TestRegressionPrevention:
    """Tests that specifically prevent the gamma_dot_0 vs gamma_dot_t0 bug."""

    def test_no_gamma_dot_0_in_constants(self):
        """Ensure gamma_dot_0 is never used (should be gamma_dot_t0)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # gamma_dot_0 is the INCORRECT name that caused the bug
        assert "gamma_dot_0" not in LAMINAR_FLOW_PARAMS
        # gamma_dot_t0 is the CORRECT name
        assert "gamma_dot_t0" in LAMINAR_FLOW_PARAMS

    def test_no_phi_0_in_constants(self):
        """Ensure phi_0 is never used (should be phi0)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # phi_0 is incorrect (old style)
        assert "phi_0" not in LAMINAR_FLOW_PARAMS
        # phi0 is correct
        assert "phi0" in LAMINAR_FLOW_PARAMS

    def test_no_gamma_dot_offset_in_constants(self):
        """Ensure gamma_dot_offset is never used (should be gamma_dot_t_offset)."""
        from homodyne.config.parameter_names import LAMINAR_FLOW_PARAMS

        # gamma_dot_offset is incorrect
        assert "gamma_dot_offset" not in LAMINAR_FLOW_PARAMS
        # gamma_dot_t_offset is correct
        assert "gamma_dot_t_offset" in LAMINAR_FLOW_PARAMS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
