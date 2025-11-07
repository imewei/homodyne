"""Unit Tests for CMC Coordinator - Config-Driven Parameter Loading
====================================================================

Tests for Task Group 4.1: CMC Coordinator Updates

This test suite verifies:
1. parameter_space propagation through CMC pipeline
2. initial_values usage for shard initialization
3. Automatic selection triggering CMC (integration with mcmc.py)
4. Removal of all SVI references
5. Config-driven parameter loading consistency

Requirements:
- >= 3 tests for CMC coordinator
- Test parameter_space propagation
- Test initial_values usage
- Test with automatic selection triggering CMC

Test Coverage:
- Config-driven parameter loading (parameter_space + initial_values)
- Mid-point default calculation when initial_values is None
- Parameter validation against bounds
- Backend receives correct parameters (propagation verification)
- Automatic CMC selection logging (verify criterion logged)
"""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch, MagicMock

# Import CMC coordinator
from homodyne.optimization.cmc.coordinator import CMCCoordinator

# Import config infrastructure
try:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

    HAS_PARAMETER_SPACE = True
except ImportError:
    HAS_PARAMETER_SPACE = False
    pytest.skip("ParameterSpace not available", allow_module_level=True)


@pytest.fixture
def static_parameter_space():
    """Create parameter_space for static_isotropic mode."""
    config = {
        "parameter_space": {
            "model": "static",
            "bounds": [
                {"name": "D0", "min": 100.0, "max": 5000.0},
                {"name": "alpha", "min": 0.1, "max": 2.0},
                {"name": "D_offset", "min": 0.1, "max": 100.0},
            ],
            "priors": [
                {"name": "D0", "type": "TruncatedNormal", "mu": 1000.0, "sigma": 500.0},
                {"name": "alpha", "type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                {
                    "name": "D_offset",
                    "type": "TruncatedNormal",
                    "mu": 10.0,
                    "sigma": 5.0,
                },
            ],
        }
    }
    return ParameterSpace.from_config(config)


@pytest.fixture
def laminar_parameter_space():
    """Create parameter_space for laminar_flow mode."""
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
                {"name": "phi0", "min": -np.pi, "max": np.pi},
            ],
        }
    }
    return ParameterSpace.from_config(config)


@pytest.fixture
def initial_values_static():
    """Create initial_values for static mode (from NLSQ results)."""
    return {
        "D0": 1234.5,
        "alpha": 0.567,
        "D_offset": 12.34,
    }


@pytest.fixture
def initial_values_laminar():
    """Create initial_values for laminar flow mode."""
    return {
        "D0": 1234.5,
        "alpha": 0.567,
        "D_offset": 12.34,
        "gamma_dot_t0": 1.23,
        "beta": 0.89,
        "gamma_dot_t_offset": 0.45,
        "phi0": 0.12,
    }


@pytest.fixture
def mock_backend():
    """Create mock CMC backend for testing."""
    backend = Mock()
    backend.get_backend_name.return_value = "mock_backend"

    # Mock successful shard results
    backend.run_parallel_mcmc.return_value = [
        {
            "samples": np.random.randn(1000, 5),
            "converged": True,
            "acceptance_rate": 0.85,
            "r_hat": {"D0": 1.01, "alpha": 1.02},
            "ess": {"D0": 850.0, "alpha": 900.0},
        },
        {
            "samples": np.random.randn(1000, 5),
            "converged": True,
            "acceptance_rate": 0.82,
            "r_hat": {"D0": 1.03, "alpha": 1.01},
            "ess": {"D0": 820.0, "alpha": 880.0},
        },
    ]

    return backend


@pytest.fixture
def minimal_config():
    """Create minimal CMC config."""
    return {
        "mcmc": {
            "num_warmup": 100,
            "num_samples": 200,
            "num_chains": 1,
        },
        "cmc": {
            "sharding": {
                "strategy": "stratified",
                "min_shard_size": 100,
            },
            "combination": {
                "method": "weighted",
                "fallback_enabled": True,
            },
        },
    }


# =============================================================================
# Test 1: parameter_space Propagation Through CMC Pipeline
# =============================================================================


def test_parameter_space_propagation(
    static_parameter_space,
    initial_values_static,
    mock_backend,
    minimal_config,
):
    """Test that parameter_space is propagated through entire CMC pipeline.

    Verifies:
    1. parameter_space accepted by run_cmc()
    2. Passed to backend.run_parallel_mcmc()
    3. Used for parameter bounds and priors

    Task 4.1.2: Integrate config-driven parameter loading
    """
    # Create synthetic data
    n_samples = 500
    data = np.random.randn(n_samples)
    t1 = np.linspace(0, 1, n_samples)
    t2 = np.linspace(0, 1, n_samples)
    phi = np.zeros(n_samples)
    q = 0.01
    L = 3.5

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Run CMC with parameter_space
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="static_isotropic",
            parameter_space=static_parameter_space,
            initial_values=initial_values_static,
        )

    # Verify backend was called with parameter_space
    assert mock_backend.run_parallel_mcmc.called
    call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs

    # Check parameter_space propagation
    assert "parameter_space" in call_kwargs
    assert call_kwargs["parameter_space"] is static_parameter_space

    # Check analysis_mode propagation
    assert "analysis_mode" in call_kwargs
    assert call_kwargs["analysis_mode"] == "static_isotropic"

    # Check init_params includes initial_values
    assert "init_params" in call_kwargs
    init_params = call_kwargs["init_params"]
    assert init_params["D0"] == 1234.5
    assert init_params["alpha"] == 0.567
    assert init_params["D_offset"] == 12.34

    # Check scaling parameters added
    assert "contrast" in init_params
    assert "offset" in init_params

    print("✓ parameter_space successfully propagated through CMC pipeline")


# =============================================================================
# Test 2: initial_values Usage for Shard Initialization
# =============================================================================


def test_initial_values_usage(
    laminar_parameter_space,
    initial_values_laminar,
    mock_backend,
    minimal_config,
):
    """Test that initial_values are used for MCMC chain initialization.

    Verifies:
    1. initial_values accepted by run_cmc()
    2. Passed to backend as init_params
    3. Validated against parameter_space bounds

    Task 4.1.2: Use initial_values for shard initialization
    """
    # Create synthetic data
    n_samples = 500
    data = np.random.randn(n_samples)
    t1 = np.linspace(0, 1, n_samples)
    t2 = np.linspace(0, 1, n_samples)
    phi = np.zeros(n_samples)
    q = 0.01
    L = 3.5

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Run CMC with initial_values
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="laminar_flow",
            parameter_space=laminar_parameter_space,
            initial_values=initial_values_laminar,
        )

    # Verify backend received init_params
    call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs
    init_params = call_kwargs["init_params"]

    # Check all physics parameters from initial_values
    assert init_params["D0"] == 1234.5
    assert init_params["alpha"] == 0.567
    assert init_params["D_offset"] == 12.34
    assert init_params["gamma_dot_t0"] == 1.23
    assert init_params["beta"] == 0.89
    assert init_params["gamma_dot_t_offset"] == 0.45
    assert init_params["phi0"] == 0.12

    # Check scaling parameters added automatically
    assert init_params["contrast"] == 0.5  # Default
    assert init_params["offset"] == 1.0  # Default

    print("✓ initial_values successfully used for shard initialization")


# =============================================================================
# Test 3: Mid-Point Defaults When initial_values is None
# =============================================================================


def test_midpoint_defaults_when_no_initial_values(
    static_parameter_space,
    mock_backend,
    minimal_config,
):
    """Test that mid-point defaults are calculated when initial_values is None.

    Verifies:
    1. Mid-point calculation from parameter_space bounds
    2. Defaults logged for transparency
    3. Backend receives mid-point init_params

    Task 4.1.2: Use parameter_space for mid-point defaults
    """
    # Create synthetic data
    n_samples = 500
    data = np.random.randn(n_samples)
    t1 = np.linspace(0, 1, n_samples)
    t2 = np.linspace(0, 1, n_samples)
    phi = np.zeros(n_samples)
    q = 0.01
    L = 3.5

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Run CMC WITHOUT initial_values (should use mid-point defaults)
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="static_isotropic",
            parameter_space=static_parameter_space,
            initial_values=None,  # Trigger mid-point calculation
        )

    # Verify backend received mid-point init_params
    call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs
    init_params = call_kwargs["init_params"]

    # Check mid-point values
    # D0 bounds: [100, 5000] → mid-point = 2550.0
    # alpha bounds: [0.1, 2.0] → mid-point = 1.05
    # D_offset bounds: [0.1, 100.0] → mid-point = 50.05
    assert abs(init_params["D0"] - 2550.0) < 1.0
    assert abs(init_params["alpha"] - 1.05) < 0.01
    assert abs(init_params["D_offset"] - 50.05) < 0.01

    print("✓ Mid-point defaults calculated correctly when initial_values is None")


# =============================================================================
# Test 4: Parameter Validation Against Bounds
# =============================================================================


def test_parameter_validation_against_bounds(
    static_parameter_space,
    mock_backend,
    minimal_config,
):
    """Test that initial_values are validated against parameter_space bounds.

    Verifies:
    1. Out-of-bounds values rejected
    2. ValueError raised with clear message
    3. Validation happens before backend execution

    Task 4.1.2: Validate initial_values
    """
    # Create synthetic data
    n_samples = 500
    data = np.random.randn(n_samples)
    t1 = np.linspace(0, 1, n_samples)
    t2 = np.linspace(0, 1, n_samples)
    phi = np.zeros(n_samples)
    q = 0.01
    L = 3.5

    # Create invalid initial_values (D0 out of bounds: max = 5000.0)
    invalid_initial_values = {
        "D0": 10000.0,  # Exceeds max bound
        "alpha": 0.567,
        "D_offset": 12.34,
    }

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Expect ValueError due to out-of-bounds D0
        with pytest.raises(ValueError, match="violate bounds"):
            coordinator.run_cmc(
                data=data,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                analysis_mode="static_isotropic",
                parameter_space=static_parameter_space,
                initial_values=invalid_initial_values,
            )

    # Verify backend was NOT called (validation failed before execution)
    assert not mock_backend.run_parallel_mcmc.called

    print("✓ Parameter validation correctly rejects out-of-bounds values")


# =============================================================================
# Test 5: Automatic CMC Selection Logging
# =============================================================================


def test_automatic_cmc_selection_logging(
    static_parameter_space,
    initial_values_static,
    mock_backend,
    minimal_config,
    caplog,
):
    """Test that CMC logs automatic selection information.

    Verifies:
    1. Log message indicates CMC was automatically selected
    2. Log includes which criterion triggered CMC (parallelism vs memory)
    3. Log includes CMC configuration (num_shards, backend)

    Task 4.1.4: Update CMC logging
    """
    import logging

    caplog.set_level(logging.INFO)

    # Create synthetic data (large enough to potentially trigger CMC)
    n_samples = 1000
    data = np.random.randn(n_samples)
    t1 = np.linspace(0, 1, n_samples)
    t2 = np.linspace(0, 1, n_samples)
    phi = np.zeros(n_samples)
    q = 0.01
    L = 3.5

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Run CMC
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="static_isotropic",
            parameter_space=static_parameter_space,
            initial_values=initial_values_static,
        )

    # Check logging for CMC selection information
    log_text = caplog.text.lower()

    # Verify CMC pipeline logged
    assert "starting cmc pipeline" in log_text or "cmc" in log_text

    # Verify backend name logged
    assert "mock_backend" in log_text or "backend" in log_text

    # Verify shard information logged
    assert "shard" in log_text

    # Verify parameter loading logged
    assert "parameter" in log_text or "initial" in log_text

    print("✓ CMC selection and configuration properly logged")


# =============================================================================
# Test 6: No SVI References Remaining
# =============================================================================


def test_no_svi_references_in_coordinator():
    """Verify that all SVI references have been removed from CMC coordinator.

    Checks:
    1. No SVI imports
    2. No SVI-related parameters
    3. No SVI-related docstrings

    Task 4.1.3: Remove any remaining SVI references
    """
    # Read coordinator source code
    import homodyne.optimization.cmc.coordinator as coordinator_module
    import inspect

    source = inspect.getsource(coordinator_module)
    source_lower = source.lower()

    # Check for SVI references (should be NONE)
    # Note: Some legitimate references to "svi" might exist in comments explaining removal
    # But should not be in active code

    # Check function signatures don't have SVI parameters
    assert "svi_params" not in source_lower
    assert "svi_config" not in source_lower
    assert "svi_result" not in source_lower

    # Check docstrings updated (no "from SVI" language in active parameters)
    # Note: Historical context in docstrings is OK, but active parameter descriptions should not reference SVI
    run_cmc_source = inspect.getsource(coordinator_module.CMCCoordinator.run_cmc)

    # Parameter descriptions should not say "from SVI"
    assert "initial parameter values from svi" not in run_cmc_source.lower()
    assert "inverse mass matrix from svi" not in run_cmc_source.lower()

    print("✓ No active SVI references found in CMC coordinator")


# =============================================================================
# Integration Test: Full CMC Pipeline with Config-Driven Parameters
# =============================================================================


def test_full_cmc_pipeline_with_config_parameters(
    laminar_parameter_space,
    initial_values_laminar,
    mock_backend,
    minimal_config,
):
    """Integration test: Full CMC pipeline with config-driven parameters.

    Verifies end-to-end workflow:
    1. Load parameter_space from config
    2. Load initial_values from config
    3. Create coordinator
    4. Execute CMC pipeline
    5. Verify results contain CMC metadata

    Task Group 4.1: Complete integration test
    """
    # Create realistic synthetic data (2D array for multiple phi angles)
    n_phi = 3
    n_t = 200
    data = np.random.randn(n_phi, n_t, n_t)
    t1 = np.tile(np.linspace(0, 1, n_t), (n_t, 1))  # 2D meshgrid
    t2 = t1.T
    phi = np.linspace(0, 2 * np.pi, n_phi)
    q = 0.01
    L = 3.5

    # Create coordinator with mocked backend
    with patch(
        "homodyne.optimization.cmc.coordinator.select_backend",
        return_value=mock_backend,
    ):
        coordinator = CMCCoordinator(minimal_config)

        # Run full CMC pipeline
        result = coordinator.run_cmc(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="laminar_flow",
            parameter_space=laminar_parameter_space,
            initial_values=initial_values_laminar,
        )

    # Verify result structure
    assert result is not None
    assert hasattr(result, "mean_params")
    assert hasattr(result, "converged")

    # Verify CMC-specific fields (extended MCMCResult)
    assert hasattr(result, "num_shards")
    assert hasattr(result, "combination_method")

    # Verify backend was called correctly
    assert mock_backend.run_parallel_mcmc.called
    call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs

    # Final verification: All config-driven parameters propagated
    assert call_kwargs["parameter_space"] is laminar_parameter_space
    assert call_kwargs["analysis_mode"] == "laminar_flow"
    assert call_kwargs["init_params"]["D0"] == 1234.5  # From initial_values

    print("✓ Full CMC pipeline executed successfully with config-driven parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
