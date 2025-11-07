"""Unit Tests for CMC Multiprocessing Backend with Laminar Flow
===================================================================

This test suite verifies:
1. Multiprocessing backend works with laminar_flow analysis_mode
2. Model creation uses correct parameter count (9 for laminar flow)
3. Validation of analysis_mode consistency passes/fails appropriately
4. Worker function properly handles laminar flow physics

Test Coverage:
- Laminar flow parameter space propagation
- Analysis mode validation (should pass for correct config)
- Analysis mode validation (should fail for mismatched config)
- Backend execution with laminar flow mode
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import multiprocessing backend
from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend

# Import config infrastructure
try:
    from homodyne.config.parameter_space import ParameterSpace

    HAS_PARAMETER_SPACE = True
except ImportError:
    HAS_PARAMETER_SPACE = False
    pytest.skip("ParameterSpace not available", allow_module_level=True)


@pytest.fixture
def laminar_parameter_space():
    """Create parameter_space for laminar_flow mode (7 physical params).

    NOTE: contrast and offset are NOT included in ParameterSpace.
    They are scaling parameters added automatically by the MCMC model.
    """
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
            "priors": [
                {"name": "D0", "type": "TruncatedNormal", "mu": 1000.0, "sigma": 500.0},
                {"name": "alpha", "type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                {
                    "name": "D_offset",
                    "type": "TruncatedNormal",
                    "mu": 10.0,
                    "sigma": 5.0,
                },
                {
                    "name": "gamma_dot_t0",
                    "type": "TruncatedNormal",
                    "mu": 1.0,
                    "sigma": 1.0,
                },
                {"name": "beta", "type": "TruncatedNormal", "mu": 1.0, "sigma": 0.5},
                {
                    "name": "gamma_dot_t_offset",
                    "type": "Uniform",
                    "low": 0.0,
                    "high": 5.0,
                },
                {"name": "phi0", "type": "Uniform", "low": -np.pi, "high": np.pi},
            ],
        }
    }
    return ParameterSpace.from_config(config)


@pytest.fixture
def static_parameter_space():
    """Create parameter_space for static_isotropic mode (3 physical params).

    NOTE: contrast and offset are NOT included in ParameterSpace.
    They are scaling parameters added automatically by the MCMC model.
    """
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
    return ParameterSpace.from_config(config)


def test_backend_validation_passes_for_laminar_flow(laminar_parameter_space):
    """Test that validation passes when analysis_mode matches parameter_space.

    Verifies:
    - analysis_mode='laminar_flow' with 9-parameter parameter_space passes
    - No ValueError is raised
    """
    backend = MultiprocessingBackend(num_workers=2)

    # Should not raise ValueError
    try:
        backend._validate_analysis_mode_consistency(
            analysis_mode="laminar_flow", parameter_space=laminar_parameter_space
        )
    except ValueError as e:
        pytest.fail(f"Validation should pass but raised ValueError: {e}")


def test_backend_validation_fails_for_mode_mismatch(
    laminar_parameter_space, static_parameter_space
):
    """Test that validation fails when analysis_mode doesn't match parameter_space.

    Verifies:
    - analysis_mode='static_isotropic' with 9-parameter space raises ValueError
    - analysis_mode='laminar_flow' with 5-parameter space raises ValueError
    - Error message is descriptive
    """
    backend = MultiprocessingBackend(num_workers=2)

    # Test 1: Static mode with laminar parameter space (9 params but expects 5)
    with pytest.raises(ValueError, match="Analysis mode mismatch"):
        backend._validate_analysis_mode_consistency(
            analysis_mode="static_isotropic", parameter_space=laminar_parameter_space
        )

    # Test 2: Laminar mode with static parameter space (5 params but expects 9)
    with pytest.raises(ValueError, match="Analysis mode mismatch"):
        backend._validate_analysis_mode_consistency(
            analysis_mode="laminar_flow", parameter_space=static_parameter_space
        )


def test_backend_validation_passes_for_static_flow(static_parameter_space):
    """Test that validation passes for static mode with correct parameter count.

    Verifies:
    - analysis_mode='static_isotropic' with 5-parameter parameter_space passes
    """
    backend = MultiprocessingBackend(num_workers=2)

    # Should not raise ValueError
    try:
        backend._validate_analysis_mode_consistency(
            analysis_mode="static_isotropic", parameter_space=static_parameter_space
        )
    except ValueError as e:
        pytest.fail(f"Validation should pass but raised ValueError: {e}")


@pytest.mark.slow
def test_multiprocessing_backend_with_laminar_flow_minimal(laminar_parameter_space):
    """Test multiprocessing backend executes with laminar flow mode.

    This is a minimal integration test that verifies:
    - Backend accepts laminar flow parameters
    - Worker function can be called without crashing
    - Validation is performed before execution

    NOTE: This test uses very small MCMC parameters for speed.
    It's not testing convergence, just that the pipeline works.
    """
    backend = MultiprocessingBackend(num_workers=1, timeout_minutes=2)

    # Create minimal synthetic shard data
    np.random.seed(42)
    n_points = 50  # Very small for speed
    shard = {
        "data": np.random.randn(n_points),
        "sigma": np.ones(n_points) * 0.1,
        "t1": np.linspace(0, 1, n_points),
        "t2": np.linspace(0, 1, n_points),
        "phi": np.zeros(n_points),  # Single angle
        "q": 0.01,
        "L": 1.0,
    }

    # Minimal MCMC config (just for smoke test, not convergence)
    mcmc_config = {
        "num_warmup": 10,
        "num_samples": 20,
        "num_chains": 1,
        "target_accept_prob": 0.8,
    }

    # Initial parameters (mid-point of bounds)
    init_params = {
        "contrast": 0.5,
        "offset": 1.0,
        "D0": 1000.0,
        "alpha": 1.0,
        "D_offset": 10.0,
        "gamma_dot_t0": 1.0,
        "beta": 1.0,
        "gamma_dot_t_offset": 0.5,
        "phi0": 0.0,
    }

    # Identity mass matrix
    inv_mass_matrix = np.eye(9)

    # Run backend (this will call the validation)
    results = backend.run_parallel_mcmc(
        shards=[shard],
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="laminar_flow",
        parameter_space=laminar_parameter_space,
    )

    # Verify we got results back
    assert len(results) == 1, "Should get 1 result for 1 shard"

    result = results[0]

    # Check result structure (may or may not converge, but should have structure)
    assert "converged" in result, "Result should have 'converged' field"
    assert "elapsed_time" in result, "Result should have 'elapsed_time' field"

    # If it converged, check samples shape
    if result["converged"]:
        assert "samples" in result, "Converged result should have 'samples'"
        samples = result["samples"]
        assert (
            samples.shape[1] == 9
        ), f"Samples should have 9 parameters, got {samples.shape[1]}"
    else:
        # If it didn't converge (which is OK for this minimal test),
        # just verify error is logged
        assert "error" in result, "Failed result should have 'error' field"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
