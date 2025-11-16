"""Tests for MCMC integration with v2.1.0 automatic NUTS/CMC selection.

This test suite validates the MCMC API after v2.1.0 refactoring which:
- REMOVED: method parameter (no more method='nuts', method='cmc', method='auto')
- ADDED: Automatic NUTS/CMC selection based on dual-criteria OR logic
- CHANGED: fit_mcmc_jax() now uses parameter_space and initial_values (config-driven)

Test Coverage
-------------
1. API signature validation (no method parameter)
2. Parameter acceptance (initial_values, parameter_space)
3. Backward compatibility with kwargs
4. Data validation
5. Result structure validation

Note: These tests validate API contract and data validation.
Full MCMC execution tests are in tests/integration/test_mcmc_*.py

Author: Claude Code (v2.1.0 refactoring)
Date: 2025-11-01
"""

import numpy as np
import pytest

from homodyne.optimization.mcmc import fit_mcmc_jax, MCMCResult
from homodyne.config.parameter_space import ParameterSpace


class TestAPISignature:
    """Test API signature after v2.1.0 refactoring."""

    def test_method_parameter_not_in_signature(self):
        """Test that method parameter is not in fit_mcmc_jax signature.

        v2.1.0 breaking change: method parameter was removed.
        Automatic selection handles NUTS/CMC internally based on data characteristics.
        """
        import inspect

        sig = inspect.signature(fit_mcmc_jax)
        assert "method" not in sig.parameters, (
            "method parameter should not exist in v2.1.0. "
            "Automatic selection is now internal."
        )

    def test_function_accepts_kwargs(self):
        """Test that fit_mcmc_jax accepts **kwargs for backward compatibility."""
        import inspect

        sig = inspect.signature(fit_mcmc_jax)
        # Should have **kwargs to accept old-style method parameter
        assert any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        ), "fit_mcmc_jax should accept **kwargs for backward compatibility"

    def test_parameter_space_parameter_exists(self):
        """Test that parameter_space parameter exists (v2.1.0 feature)."""
        import inspect

        sig = inspect.signature(fit_mcmc_jax)
        assert (
            "parameter_space" in sig.parameters
        ), "parameter_space parameter should exist in v2.1.0"

    def test_initial_values_parameter_exists(self):
        """Test that initial_values parameter exists (v2.1.0 change)."""
        import inspect

        sig = inspect.signature(fit_mcmc_jax)
        assert (
            "initial_values" in sig.parameters
        ), "initial_values parameter should exist in v2.1.0 (renamed from initial_params)"


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
        """Test that mismatched array sizes raise ValueError or IndexError."""
        # Could raise ValueError during validation or IndexError during execution
        with pytest.raises((ValueError, IndexError)):
            fit_mcmc_jax(
                data=np.random.randn(100),
                t1=np.random.rand(100),
                t2=np.random.rand(50),  # Wrong size!
                phi=np.random.rand(100),
                q=0.01,
                L=3.5,
            )

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
            assert (
                "method" not in str(e).lower()
            ), "method parameter should be accepted in **kwargs (backward compatibility)"


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
                assert key not in str(
                    e
                ), f"Standard MCMC kwarg '{key}' should be accepted"

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


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
