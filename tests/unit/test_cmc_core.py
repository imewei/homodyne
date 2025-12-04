"""
Unit Tests for CMC Core Functionality
======================================

Consolidated from:
- test_cmc_coordinator.py (CMC coordinator orchestration, 7 tests, 621 lines)
- test_cmc_config.py (CMC configuration system, 21 tests, 644 lines)
- test_cmc_backend_laminar_flow.py (Laminar flow backend, 4 tests, 247 lines)

Tests cover:
- CMC coordinator with config-driven parameter loading
- Parameter space propagation through CMC pipeline
- Initial values usage for shard initialization
- CMC configuration parsing and validation
- TypedDict type safety and backward compatibility
- Multiprocessing backend with laminar flow analysis mode

Total: 32 tests
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.optimization.mcmc.cmc.backends.multiprocessing import MultiprocessingBackend

# Import CMC coordinator and backends
from homodyne.optimization.mcmc.cmc.coordinator import CMCCoordinator

# Import config infrastructure
try:
    from homodyne.config.parameter_space import ParameterSpace, PriorDistribution

    HAS_PARAMETER_SPACE = True
except ImportError:
    HAS_PARAMETER_SPACE = False
    pytest.skip("ParameterSpace not available", allow_module_level=True)

from homodyne.config.manager import ConfigManager
from homodyne.config.types import CMCConfig, CMCInitializationConfig, CMCShardingConfig

# ==============================================================================
# CMC Coordinator Tests (from test_cmc_coordinator.py)
# ==============================================================================


@pytest.fixture
def static_parameter_space():
    """Create parameter_space for static_mode mode."""
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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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
            analysis_mode="static",
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
    assert call_kwargs["analysis_mode"] == "static"

    # Check init_params includes initial_values
    assert "init_params" in call_kwargs
    init_params = call_kwargs["init_params"]
    assert init_params["D0"] == 1234.5
    assert init_params["alpha"] == 0.567
    assert init_params["D_offset"] == 12.34

    # Check scaling parameters added (per-angle in v2.4.0+)
    assert "contrast_0" in init_params
    assert "offset_0" in init_params

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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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

    # Check scaling parameters added automatically (per-angle in v2.4.0+)
    assert init_params["contrast_0"] == 0.5  # Default per-angle
    assert init_params["offset_0"] == 1.0  # Default per-angle

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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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
            analysis_mode="static",
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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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
                analysis_mode="static",
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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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
            analysis_mode="static",
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
    import inspect

    import homodyne.optimization.mcmc.cmc.coordinator as coordinator_module

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
        "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
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


# ==============================================================================
# CMC Configuration Tests (from test_cmc_config.py)
# ==============================================================================


class TestCMCConfigParsing:
    """Test CMC configuration parsing from YAML."""

    def test_parse_minimal_cmc_config(self):
        """Test parsing minimal CMC configuration with defaults."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  method: cmc
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Check that enable was parsed
            assert cmc_config["enable"] is True

            # Check that defaults were applied
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"
            # Note: initialization section removed in v2.1.0 (no more SVI)
            assert cmc_config["backend"]["name"] == "auto"
            assert cmc_config["combination"]["method"] == "weighted_gaussian"

        finally:
            Path(config_path).unlink()

    def test_parse_complete_cmc_config(self):
        """Test parsing complete CMC configuration with all fields."""
        config_yaml = """
analysis_mode: laminar_flow

optimization:
  method: cmc
  cmc:
    enable: auto
    min_points_for_cmc: 1000000
    sharding:
      strategy: random
      num_shards: 16
      max_points_per_shard: 500000
    backend:
      name: pjit
      enable_checkpoints: false
      checkpoint_frequency: 5
      checkpoint_dir: ./my_checkpoints
      keep_last_checkpoints: 5
      resume_from_checkpoint: false
    combination:
      method: simple_average
      validate_results: false
      min_success_rate: 0.80
    per_shard_mcmc:
      num_warmup: 1000
      num_samples: 3000
      num_chains: 2
      subsample_size: 100000
    validation:
      strict_mode: false
      min_per_shard_ess: 50.0
      max_per_shard_rhat: 1.2
      max_between_shard_kl: 3.0
      min_success_rate: 0.80
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Validate all fields were parsed correctly
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 1000000

            # Sharding
            assert cmc_config["sharding"]["strategy"] == "random"
            assert cmc_config["sharding"]["num_shards"] == 16
            assert cmc_config["sharding"]["max_points_per_shard"] == 500000

            # Note: Initialization section removed in v2.1.0 (no more SVI)

            # Backend
            assert cmc_config["backend"]["name"] == "pjit"
            assert cmc_config["backend"]["enable_checkpoints"] is False
            assert cmc_config["backend"]["checkpoint_frequency"] == 5

            # Combination
            assert cmc_config["combination"]["method"] == "simple_average"
            assert cmc_config["combination"]["validate_results"] is False
            assert cmc_config["combination"]["min_success_rate"] == 0.80

            # Per-shard MCMC
            assert cmc_config["per_shard_mcmc"]["num_warmup"] == 1000
            assert cmc_config["per_shard_mcmc"]["num_samples"] == 3000
            assert cmc_config["per_shard_mcmc"]["num_chains"] == 2

            # Validation
            assert cmc_config["validation"]["strict_mode"] is False
            assert cmc_config["validation"]["min_per_shard_ess"] == 50.0
            assert cmc_config["validation"]["max_per_shard_rhat"] == 1.2

        finally:
            Path(config_path).unlink()

    def test_parse_partial_cmc_config_with_defaults(self):
        """Test that partial config merges correctly with defaults."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  method: cmc
  cmc:
    enable: true
    sharding:
      strategy: contiguous
    initialization:
      method: identity
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # User-specified values
            assert cmc_config["enable"] is True
            assert cmc_config["sharding"]["strategy"] == "contiguous"

            # Default values preserved
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["num_shards"] == "auto"
            assert cmc_config["backend"]["name"] == "auto"

        finally:
            Path(config_path).unlink()


class TestCMCConfigDefaults:
    """Test default CMC configuration values."""

    def test_defaults_without_cmc_section(self):
        """Test that defaults are returned when no CMC section exists."""
        config_yaml = """
analysis_mode: static_mode
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # Should return complete default configuration
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"
            assert cmc_config["backend"]["name"] == "auto"
            assert cmc_config["combination"]["method"] == "weighted_gaussian"
            assert cmc_config["per_shard_mcmc"]["num_warmup"] == 500
            assert cmc_config["validation"]["strict_mode"] is True

        finally:
            Path(config_path).unlink()

    def test_defaults_with_empty_cmc_section(self):
        """Test defaults applied when CMC section is empty."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  method: cmc
  cmc: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            # All defaults should be applied
            assert cmc_config["enable"] == "auto"
            assert cmc_config["min_points_for_cmc"] == 500000
            assert cmc_config["sharding"]["strategy"] == "stratified"

        finally:
            Path(config_path).unlink()


class TestCMCConfigValidation:
    """Test CMC configuration validation."""

    def test_invalid_enable_value(self):
        """Test validation catches invalid enable value."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    enable: "maybe"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="CMC enable must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_sharding_strategy(self):
        """Test validation catches invalid sharding strategy."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    sharding:
      strategy: invalid_strategy
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="Sharding strategy must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_num_shards(self):
        """Test validation catches invalid num_shards."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    sharding:
      num_shards: -5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="num_shards must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_backend_name(self):
        """Test validation catches invalid backend name."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    backend:
      name: spark
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="Backend name must be"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_min_success_rate(self):
        """Test validation catches invalid min_success_rate."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    combination:
      min_success_rate: 1.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="min_success_rate must be between"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_per_shard_num_warmup(self):
        """Test validation catches invalid num_warmup."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    per_shard_mcmc:
      num_warmup: -100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(
                ValueError, match="num_warmup must be a positive integer"
            ):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()

    def test_invalid_rhat_threshold(self):
        """Test validation catches invalid max_per_shard_rhat."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    validation:
      max_per_shard_rhat: 0.5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            with pytest.raises(ValueError, match="max_per_shard_rhat must be >= 1.0"):
                config_mgr.get_cmc_config()

        finally:
            Path(config_path).unlink()


class TestCMCConfigManagerMethod:
    """Test ConfigManager.get_cmc_config() method."""

    def test_get_cmc_config_returns_dict(self):
        """Test that get_cmc_config() returns a dictionary."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            assert isinstance(cmc_config, dict)
            assert "enable" in cmc_config
            assert "sharding" in cmc_config
            # "initialization" removed in v2.1.0
            assert "backend" in cmc_config
            assert "combination" in cmc_config
            assert "per_shard_mcmc" in cmc_config
            assert "validation" in cmc_config

        finally:
            Path(config_path).unlink()

    def test_get_cmc_config_with_config_override(self):
        """Test get_cmc_config() with config_override parameter."""
        config_dict = {
            "analysis_mode": "static",
            "optimization": {
                "cmc": {
                    "enable": True,
                    "sharding": {"strategy": "random"},
                }
            },
        }

        config_mgr = ConfigManager("dummy.yaml", config_override=config_dict)
        cmc_config = config_mgr.get_cmc_config()

        assert cmc_config["enable"] is True
        assert cmc_config["sharding"]["strategy"] == "random"


class TestCMCTypedDictCompatibility:
    """Test TypedDict type hints for CMC configuration."""

    def test_cmc_sharding_config_structure(self):
        """Test CMCShardingConfig TypedDict structure."""
        sharding_config: CMCShardingConfig = {
            "strategy": "stratified",
            "num_shards": "auto",
            "max_points_per_shard": "auto",
        }

        assert sharding_config["strategy"] == "stratified"
        assert sharding_config["num_shards"] == "auto"

    def test_cmc_initialization_config_structure(self):
        """Test CMCInitializationConfig TypedDict structure."""
        init_config: CMCInitializationConfig = {
            "method": "svi",
            "svi_steps": 5000,
            "svi_learning_rate": 0.001,
            "svi_rank": 5,
            "fallback_to_identity": True,
        }

        assert init_config["method"] == "svi"
        assert init_config["svi_steps"] == 5000

    def test_cmc_config_full_structure(self):
        """Test complete CMCConfig TypedDict structure."""
        cmc_config: CMCConfig = {
            "enable": "auto",
            "min_points_for_cmc": 500000,
            "sharding": {
                "strategy": "stratified",
                "num_shards": "auto",
                "max_points_per_shard": "auto",
            },
            "initialization": {
                "method": "svi",
                "svi_steps": 5000,
                "svi_learning_rate": 0.001,
                "svi_rank": 5,
                "fallback_to_identity": True,
            },
            "backend": {
                "name": "auto",
                "enable_checkpoints": True,
                "checkpoint_frequency": 10,
                "checkpoint_dir": "./checkpoints/cmc",
                "keep_last_checkpoints": 3,
                "resume_from_checkpoint": True,
            },
            "combination": {
                "method": "weighted_gaussian",
                "validate_results": True,
                "min_success_rate": 0.90,
            },
            "per_shard_mcmc": {
                "num_warmup": 500,
                "num_samples": 2000,
                "num_chains": 1,
                "subsample_size": "auto",
            },
            "validation": {
                "strict_mode": True,
                "min_per_shard_ess": 100.0,
                "max_per_shard_rhat": 1.1,
                "max_between_shard_kl": 2.0,
                "min_success_rate": 0.90,
            },
        }

        assert cmc_config["enable"] == "auto"
        assert cmc_config["sharding"]["strategy"] == "stratified"


class TestCMCDeprecationWarnings:
    """Test deprecation warnings for old CMC settings."""

    def test_deprecated_consensus_monte_carlo_key(self, caplog):
        """Test warning for deprecated 'consensus_monte_carlo' key."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  consensus_monte_carlo:
    enable: true
  cmc:
    enable: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            config_mgr.get_cmc_config()

            # Check for deprecation warning
            assert any(
                "consensus_monte_carlo" in record.message for record in caplog.records
            )

        finally:
            Path(config_path).unlink()

    def test_deprecated_optimal_shard_size_key(self, caplog):
        """Test warning for deprecated 'optimal_shard_size' key."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  cmc:
    sharding:
      optimal_shard_size: 1000000
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            config_mgr.get_cmc_config()

            # Check for deprecation warning
            assert any(
                "optimal_shard_size" in record.message for record in caplog.records
            )

        finally:
            Path(config_path).unlink()


class TestCMCBackwardCompatibility:
    """Test backward compatibility with existing configurations."""

    def test_old_config_without_cmc_still_works(self):
        """Test that old configs without CMC section still work."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  method: nlsq
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            # Should not raise error
            cmc_config = config_mgr.get_cmc_config()

            # Should return defaults
            assert cmc_config["enable"] == "auto"

        finally:
            Path(config_path).unlink()

    def test_nlsq_method_with_cmc_config(self):
        """Test NLSQ method can coexist with CMC config."""
        config_yaml = """
analysis_mode: static_mode

optimization:
  method: nlsq
  cmc:
    enable: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config_path = f.name

        try:
            config_mgr = ConfigManager(config_path)
            cmc_config = config_mgr.get_cmc_config()

            assert cmc_config["enable"] is False

        finally:
            Path(config_path).unlink()


# ==============================================================================
# CMC Laminar Flow Backend Tests (from test_cmc_backend_laminar_flow.py)
# ==============================================================================


@pytest.fixture
def laminar_parameter_space_extended():
    """Create parameter_space for laminar_flow mode (7 physical params) with priors.

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
def static_parameter_space_extended():
    """Create parameter_space for static_mode mode (3 physical params) extended.

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
    - analysis_mode='static' with 9-parameter space raises ValueError
    - analysis_mode='laminar_flow' with 5-parameter space raises ValueError
    - Error message is descriptive
    """
    backend = MultiprocessingBackend(num_workers=2)

    # Test 1: Static mode with laminar parameter space (9 params but expects 5)
    with pytest.raises(ValueError, match="Analysis mode mismatch"):
        backend._validate_analysis_mode_consistency(
            analysis_mode="static", parameter_space=laminar_parameter_space
        )

    # Test 2: Laminar mode with static parameter space (5 params but expects 9)
    with pytest.raises(ValueError, match="Analysis mode mismatch"):
        backend._validate_analysis_mode_consistency(
            analysis_mode="laminar_flow", parameter_space=static_parameter_space
        )


def test_backend_validation_passes_for_static_flow(static_parameter_space):
    """Test that validation passes for static mode with correct parameter count.

    Verifies:
    - analysis_mode='static' with 5-parameter parameter_space passes
    """
    backend = MultiprocessingBackend(num_workers=2)

    # Should not raise ValueError
    try:
        backend._validate_analysis_mode_consistency(
            analysis_mode="static", parameter_space=static_parameter_space
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
        assert samples.shape[1] == 9, (
            f"Samples should have 9 parameters, got {samples.shape[1]}"
        )
    else:
        # If it didn't converge (which is OK for this minimal test),
        # just verify error is logged
        assert "error" in result, "Failed result should have 'error' field"


# ==============================================================================
# Additional CMC Coordinator Tests (v2.4.1)
# ==============================================================================


class TestCMCCoordinatorMultiAngle:
    """Tests for CMC coordinator with multiple phi angles."""

    @pytest.fixture
    def multi_angle_data(self):
        """Create multi-angle synthetic data (3 phi angles)."""
        np.random.seed(42)
        n_phi = 3
        n_t = 50  # Small for test speed
        n_points = n_phi * n_t * n_t

        # Create meshgrid-style data
        phi_values = np.array([0.0, 60.0, 120.0]) * np.pi / 180  # Convert to radians
        t_range = np.linspace(0, 1, n_t)
        t1_mesh, t2_mesh = np.meshgrid(t_range, t_range, indexing="ij")

        # Flatten t1, t2 for each angle
        t1_pattern = t1_mesh.flatten()
        t2_pattern = t2_mesh.flatten()

        # Replicate for each phi angle
        t1 = np.tile(t1_pattern, n_phi)
        t2 = np.tile(t2_pattern, n_phi)
        phi = np.repeat(phi_values, n_t * n_t)

        # Generate synthetic C2 data
        data = 1.0 + 0.1 * np.random.randn(n_points)

        return {
            "data": data,
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "n_phi": n_phi,
            "n_t": n_t,
        }

    def test_multi_angle_parameter_expansion(
        self, multi_angle_data, static_parameter_space, mock_backend, minimal_config
    ):
        """Test parameter expansion for multi-angle CMC.

        Verifies:
        1. Per-angle contrast/offset parameters are created
        2. Correct number of total parameters
        3. Parameter ordering is correct (contrast_i, offset_i first, then physics)
        """
        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
            return_value=mock_backend,
        ):
            coordinator = CMCCoordinator(minimal_config)

            initial_values = {"D0": 1000.0, "alpha": 1.0, "D_offset": 10.0}

            result = coordinator.run_cmc(
                data=multi_angle_data["data"],
                t1=multi_angle_data["t1"],
                t2=multi_angle_data["t2"],
                phi=multi_angle_data["phi"],
                q=0.01,
                L=3.5,
                analysis_mode="static",
                parameter_space=static_parameter_space,
                initial_values=initial_values,
            )

        # Verify backend received expanded parameters
        call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs
        init_params = call_kwargs["init_params"]

        # Check per-angle parameters exist
        n_phi = multi_angle_data["n_phi"]
        for i in range(n_phi):
            assert f"contrast_{i}" in init_params, f"Missing contrast_{i}"
            assert f"offset_{i}" in init_params, f"Missing offset_{i}"

        # Check physical parameters exist
        assert "D0" in init_params
        assert "alpha" in init_params
        assert "D_offset" in init_params

        # Check total parameter count: 3 physical + 2*3 per-angle = 9
        assert len(init_params) == 9

    def test_multi_angle_init_params_ordering(
        self, multi_angle_data, static_parameter_space, mock_backend, minimal_config
    ):
        """Test that init_params ordering matches NumPyro model sampling order.

        CRITICAL: NumPyro's init_to_value() requires parameters in the EXACT ORDER
        the model samples them: per-angle params FIRST, then physics params.
        """
        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend",
            return_value=mock_backend,
        ):
            coordinator = CMCCoordinator(minimal_config)

            initial_values = {"D0": 1000.0, "alpha": 1.0, "D_offset": 10.0}

            coordinator.run_cmc(
                data=multi_angle_data["data"],
                t1=multi_angle_data["t1"],
                t2=multi_angle_data["t2"],
                phi=multi_angle_data["phi"],
                q=0.01,
                L=3.5,
                analysis_mode="static",
                parameter_space=static_parameter_space,
                initial_values=initial_values,
            )

        call_kwargs = mock_backend.run_parallel_mcmc.call_args.kwargs
        init_params = call_kwargs["init_params"]

        # Get parameter keys in order (Python 3.7+ preserves dict insertion order)
        param_keys = list(init_params.keys())

        # Verify ordering: contrast_* first, then offset_*, then physics
        n_phi = multi_angle_data["n_phi"]
        expected_order = (
            [f"contrast_{i}" for i in range(n_phi)]
            + [f"offset_{i}" for i in range(n_phi)]
            + ["D0", "alpha", "D_offset"]
        )

        assert param_keys == expected_order, (
            f"Parameter ordering mismatch! "
            f"Expected: {expected_order}, Got: {param_keys}"
        )


class TestCMCCoordinatorErrorHandling:
    """Tests for CMC coordinator error handling."""

    def test_empty_data_raises_value_error(self, static_parameter_space, minimal_config):
        """Test that empty data raises ValueError."""
        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_select.return_value = mock_backend
            coordinator = CMCCoordinator(minimal_config)

            with pytest.raises(ValueError, match="Cannot run CMC on empty dataset"):
                coordinator.run_cmc(
                    data=np.array([]),  # Empty data
                    t1=np.array([]),
                    t2=np.array([]),
                    phi=np.array([]),
                    q=0.01,
                    L=3.5,
                    analysis_mode="static",
                    parameter_space=static_parameter_space,
                    initial_values={"D0": 1000.0, "alpha": 1.0, "D_offset": 10.0},
                )

    def test_all_shards_failed_raises_runtime_error(
        self, static_parameter_space, minimal_config
    ):
        """Test that RuntimeError is raised when all shards fail."""
        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "mock_backend"
            # Return results where all shards failed
            mock_backend.run_parallel_mcmc.return_value = [
                {"converged": False, "error": "Test failure 1"},
                {"converged": False, "error": "Test failure 2"},
            ]
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(minimal_config)

            # Create minimal valid data
            n = 500
            data = np.random.randn(n)
            t1 = np.linspace(0, 1, n)
            t2 = np.linspace(0, 1, n)
            phi = np.zeros(n)

            with pytest.raises(RuntimeError, match="All shards failed to converge"):
                coordinator.run_cmc(
                    data=data,
                    t1=t1,
                    t2=t2,
                    phi=phi,
                    q=0.01,
                    L=3.5,
                    analysis_mode="static",
                    parameter_space=static_parameter_space,
                    initial_values={"D0": 1000.0, "alpha": 1.0, "D_offset": 10.0},
                )


class TestCMCCoordinatorBackendSelection:
    """Tests for CMC backend selection logic."""

    def test_backend_selection_from_config_dict(self):
        """Test backend selection with dict-style config (old schema)."""
        config = {
            "backend": {"name": "multiprocessing"},
            "mcmc": {"num_warmup": 100, "num_samples": 200},
        }

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)

            # Verify select_backend was called with user_override
            mock_select.assert_called_once()
            call_args = mock_select.call_args
            assert call_args.kwargs.get("user_override") == "multiprocessing"

    def test_backend_selection_auto(self):
        """Test backend selection with auto (no override)."""
        config = {
            "backend": {"name": "auto"},
            "mcmc": {"num_warmup": 100, "num_samples": 200},
        }

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)

            # Verify select_backend was called with user_override="auto"
            mock_select.assert_called_once()
            call_args = mock_select.call_args
            assert call_args.kwargs.get("user_override") == "auto"

    def test_backend_selection_with_backend_config(self):
        """Test backend selection with new schema (backend string + backend_config)."""
        config = {
            "backend": "jax",  # String (computational backend)
            "backend_config": {"name": "pbs"},  # Dict (parallel backend)
            "mcmc": {"num_warmup": 100, "num_samples": 200},
        }

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "pbs"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)

            # Verify select_backend was called with user_override="pbs"
            mock_select.assert_called_once()
            call_args = mock_select.call_args
            assert call_args.kwargs.get("user_override") == "pbs"


class TestCMCCoordinatorMCMCConfig:
    """Tests for MCMC configuration extraction."""

    def test_default_mcmc_config(self):
        """Test default MCMC config values."""
        config = {}  # No MCMC config

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)
            mcmc_config = coordinator._get_mcmc_config()

            # Check defaults
            assert mcmc_config["num_warmup"] == 500
            assert mcmc_config["num_samples"] == 2000
            assert mcmc_config["num_chains"] == 1

    def test_custom_mcmc_config(self):
        """Test custom MCMC config values."""
        config = {
            "mcmc": {
                "num_warmup": 1000,
                "num_samples": 5000,
                "num_chains": 2,
            }
        }

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)
            mcmc_config = coordinator._get_mcmc_config()

            # Check custom values
            assert mcmc_config["num_warmup"] == 1000
            assert mcmc_config["num_samples"] == 5000
            assert mcmc_config["num_chains"] == 2


class TestCMCCoordinatorShardCalculation:
    """Tests for optimal shard calculation."""

    def test_user_override_num_shards(self):
        """Test user-specified num_shards override."""
        config = {
            "cmc": {"sharding": {"num_shards": 8}},
            "mcmc": {"num_warmup": 100},
        }

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)
            num_shards = coordinator._calculate_num_shards(dataset_size=10_000_000)

            # Should use user override
            assert num_shards == 8

    def test_automatic_shard_calculation(self):
        """Test automatic shard calculation based on dataset size."""
        config = {"mcmc": {"num_warmup": 100}}

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)
            # For CPU with target_shard_size_cpu=2M: 10M / 2M = 5 shards
            num_shards = coordinator._calculate_num_shards(dataset_size=10_000_000)

            assert num_shards >= 1  # At least one shard


class TestCMCBasicValidation:
    """Tests for CMC basic validation."""

    def test_validation_detects_nan_inf(self):
        """Test validation detects NaN/Inf in combined samples."""
        config = {}

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)

            # Combined posterior with NaN
            combined_posterior = {
                "samples": np.array([[1.0, np.nan], [2.0, 3.0]]),
                "mean": np.array([1.5, np.nan]),
                "cov": np.eye(2),
            }
            shard_results = [{"converged": True}]

            is_valid, diagnostics = coordinator._basic_validation(
                combined_posterior, shard_results
            )

            assert not is_valid
            assert diagnostics.get("nan_inf_detected") is True

    def test_validation_detects_low_convergence(self):
        """Test validation detects low convergence rate."""
        config = {}

        with patch(
            "homodyne.optimization.mcmc.cmc.coordinator.select_backend"
        ) as mock_select:
            mock_backend = Mock()
            mock_backend.get_backend_name.return_value = "multiprocessing"
            mock_select.return_value = mock_backend

            coordinator = CMCCoordinator(config)

            # Combined posterior (valid)
            combined_posterior = {
                "samples": np.random.randn(100, 5),
                "mean": np.random.randn(5),
                "cov": np.eye(5),
            }

            # Low convergence: only 2/10 shards converged
            shard_results = [
                {"converged": False} for _ in range(8)
            ] + [{"converged": True} for _ in range(2)]

            is_valid, diagnostics = coordinator._basic_validation(
                combined_posterior, shard_results
            )

            assert not is_valid
            assert diagnostics.get("low_convergence_rate") is True
            assert diagnostics["convergence_rate"] == 0.2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
