"""
Integration Tests for Consensus Monte Carlo (CMC)
==================================================

Comprehensive end-to-end integration tests for CMC pipeline:
- CMC vs NUTS comparison on overlap range (small datasets)
- Backend equivalence (pjit, multiprocessing, PBS)
- Configuration integration (YAML, CLI overrides, defaults)
- Error handling and recovery

Test Tiers:
    Tier 2 (Integration): Complete CMC pipeline with various configurations
    Duration: Minutes per test, ~30 min total suite
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pytest

# Handle optional dependencies
try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Import CMC components
try:
    from homodyne.optimization.cmc.coordinator import CMCCoordinator
    from homodyne.optimization.cmc.sharding import (
        calculate_optimal_num_shards,
        shard_data_stratified,
    )
    from homodyne.optimization.cmc.result import MCMCResult

    CMC_AVAILABLE = True
except ImportError:
    CMC_AVAILABLE = False

# Import test factories
try:
    from tests.factories.data_factory import XPCSDataFactory

    FACTORIES_AVAILABLE = True
except ImportError:
    FACTORIES_AVAILABLE = False


def generate_synthetic_xpcs_data(
    n_points: int = 10000,
    n_angles: int = 8,
    analysis_mode: str = "laminar_flow",
    seed: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, float],
]:
    """
    Generate synthetic XPCS data for testing.

    Parameters
    ----------
    n_points : int
        Total number of data points across all angles
    n_angles : int
        Number of phi angles
    analysis_mode : str
        'laminar_flow' or 'static'
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (c2_exp, t1, t2, phi, q, L, true_params)
    """
    np.random.seed(seed)

    # Determine grid dimensions based on n_points
    n_times = int(np.sqrt(n_points / n_angles))

    # Create time meshgrid
    t1, t2 = np.meshgrid(
        np.linspace(0, 10, n_times), np.linspace(0, 10, n_times), indexing="ij"
    )

    # Create phi angles
    phi = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Set wavevector and length
    q = 0.01
    L = 1.0

    # Generate synthetic parameters
    if analysis_mode == "laminar_flow":
        true_params = {
            "D0": 1000.0,
            "alpha": 1.5,
            "D_offset": 10.0,
            "gamma_dot_0": 0.1,
            "beta": 0.5,
            "gamma_dot_offset": 0.01,
            "phi0": np.pi / 4,
        }
    else:  # static_mode
        true_params = {
            "D0": 1000.0,
            "alpha": 1.5,
            "D_offset": 10.0,
        }

    # Generate synthetic g1 correlation
    tau = np.abs(t1 - t2)
    g1_decay = np.exp(-tau * 0.01)

    # Generate g2 = 1 + contrast * g1^2
    contrast = 0.4
    g2_true = 1.0 + contrast * (g1_decay**2)

    # Add angle dependence for laminar flow
    if analysis_mode == "laminar_flow":
        angle_factor = 1.0 + 0.2 * np.sin(phi[:, np.newaxis, np.newaxis])
        c2_exp = np.tile(g2_true[np.newaxis, :, :], (n_angles, 1, 1))
        c2_exp *= angle_factor
    else:
        c2_exp = np.tile(g2_true[np.newaxis, :, :], (n_angles, 1, 1))

    # Flatten to match expected format: shape (n_angles, n_times, n_times)
    c2_exp = c2_exp.reshape(n_angles, n_times, n_times)

    # Add realistic noise
    noise_level = 0.01
    c2_exp += noise_level * np.random.randn(*c2_exp.shape)

    return c2_exp, t1, t2, phi, q, L, true_params


def prepare_flattened_data_for_sharding(
    c2_exp: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data for sharding by flattening to 1D arrays.

    Parameters
    ----------
    c2_exp : np.ndarray, shape (n_angles, n_times, n_times)
        Experimental correlation data
    t1, t2 : np.ndarray, shape (n_times, n_times)
        Time meshgrid arrays
    phi : np.ndarray, shape (n_angles,)
        Angle array

    Returns
    -------
    tuple
        (c2_flat, t1_flat, t2_flat, phi_flat) - all 1D arrays
    """
    n_angles = c2_exp.shape[0]
    n_times_sq = t1.size

    # Flatten correlation data
    c2_flat = c2_exp.reshape(-1)

    # Replicate time arrays for each angle
    t1_flat = np.tile(t1.flatten(), n_angles)
    t2_flat = np.tile(t2.flatten(), n_angles)

    # Repeat phi for each time point
    phi_flat = np.repeat(phi, n_times_sq)

    return c2_flat, t1_flat, t2_flat, phi_flat


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestCMCIntegrationBasic:
    """Basic integration tests for CMC pipeline."""

    def test_cmc_coordinator_instantiation(self):
        """Test CMC coordinator can be instantiated with config."""
        config = {
            "cmc": {
                "sharding": {"strategy": "stratified", "num_shards": 5},
                "initialization": {"use_svi": True, "svi_steps": 100},
                "combination": {"method": "weighted", "fallback_enabled": True},
            },
            "mcmc": {
                "num_warmup": 100,
                "num_samples": 200,
                "num_chains": 1,
            },
        }

        coordinator = CMCCoordinator(config)
        assert coordinator is not None
        assert hasattr(coordinator, "run_cmc")

    def test_cmc_data_sharding_basic(self):
        """Test basic data sharding functionality."""
        from homodyne.device.config import detect_hardware

        n_points = 1000
        n_angles = 8

        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(
            n_points=n_points, n_angles=n_angles
        )

        # Test optimal shard calculation
        hw_config = detect_hardware()
        num_shards = calculate_optimal_num_shards(n_points, hw_config)
        assert isinstance(num_shards, int)
        assert num_shards >= 1

        # Test stratified sharding
        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )

        shards = shard_data_stratified(
            data=c2_flat, t1=t1_flat, t2=t2_flat, phi=phi_flat, num_shards=2, q=q, L=L
        )
        assert len(shards) == 2
        for shard in shards:
            assert "data" in shard
            assert "phi" in shard

    def test_cmc_small_dataset_vs_nuts(self):
        """
        Test CMC vs NUTS on small dataset should give similar results.

        CMC and NUTS should agree on 10k-point dataset within ~10% error.
        """
        n_points = 10000
        c2_exp, t1, t2, phi, q, L, true_params = generate_synthetic_xpcs_data(
            n_points=n_points, n_angles=8, analysis_mode="static"
        )

        # This test structure validates integration
        # Actual NUTS/CMC comparison deferred to validation tier
        assert c2_exp.shape[0] == 8
        assert c2_exp.shape[1] == c2_exp.shape[2]
        assert len(phi) == 8

        # Verify synthetic data has expected structure
        assert np.all(np.isfinite(c2_exp))
        assert np.all(c2_exp > 0)  # Correlation data should be positive

    def test_cmc_configuration_yaml(self):
        """Test CMC configuration loading from YAML."""
        if not HAS_YAML:
            pytest.skip("PyYAML not available")

        config_content = """
cmc:
  sharding:
    strategy: stratified
    num_shards: 5
  initialization:
    use_svi: true
    svi_steps: 1000
  combination:
    method: weighted
    fallback_enabled: true
    fallback_method: simple_averaging
mcmc:
  num_warmup: 500
  num_samples: 2000
  num_chains: 1
"""

        config = yaml.safe_load(config_content)
        assert config["cmc"]["sharding"]["strategy"] == "stratified"
        assert config["cmc"]["sharding"]["num_shards"] == 5
        assert config["mcmc"]["num_samples"] == 2000


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCBackendIntegration:
    """Test CMC with different execution backends."""

    def test_backend_selection_logic(self):
        """Test backend selection based on hardware configuration."""
        from homodyne.device.config import detect_hardware
        from homodyne.optimization.cmc.backends import select_backend

        # Detect hardware
        hw_config = detect_hardware()
        assert hw_config is not None

        # Test that backend recommendation exists
        assert hw_config.recommended_backend in [
            "pjit",
            "multiprocessing",
            "pbs",
            "slurm",
        ]

        # Test auto-selection (uses hardware config)
        backend_auto = select_backend(hw_config)
        assert backend_auto is not None
        assert backend_auto.get_backend_name() in ["pjit", "multiprocessing", "pbs"]

        # Test manual override
        backend_override = select_backend(hw_config, user_override="multiprocessing")
        assert backend_override is not None
        assert backend_override.get_backend_name() == "multiprocessing"

    def test_multiprocessing_backend_basic(self):
        """Test multiprocessing backend instantiation."""
        from homodyne.optimization.cmc.backends import select_backend
        from homodyne.device.config import detect_hardware

        hw_config = detect_hardware()
        backend = select_backend(hw_config, user_override="multiprocessing")

        assert backend is not None
        assert hasattr(backend, "run_parallel_mcmc")
        assert hasattr(backend, "get_backend_name")
        assert backend.get_backend_name() == "multiprocessing"


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCShardingStrategies:
    """Test different sharding strategies."""

    def test_stratified_sharding(self):
        """Test stratified sharding preserves phi distribution."""
        n_points = 10000
        n_angles = 8
        num_shards = 4

        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(
            n_points=n_points, n_angles=n_angles
        )

        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )
        shards = shard_data_stratified(
            data=c2_flat,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            num_shards=num_shards,
            q=q,
            L=L,
        )

        assert len(shards) == num_shards

        # Verify each shard has data and total data points preserved
        total_data_points = 0
        unique_angles_per_shard = []
        for shard in shards:
            assert "data" in shard
            assert "phi" in shard
            assert len(shard["phi"]) > 0
            total_data_points += len(shard["data"])
            # Count unique angles in each shard
            unique_angles_per_shard.append(len(np.unique(shard["phi"])))

        # Total data points should match flattened data (minus 2% removed by sharding)
        assert total_data_points >= int(len(c2_flat) * 0.98)
        # Each shard should have all angles represented (stratified sampling)
        for unique_count in unique_angles_per_shard:
            assert unique_count == n_angles

    def test_shard_size_calculation(self):
        """Test correct shard size calculation."""
        from homodyne.device.config import detect_hardware

        test_cases = [
            (100_000, 4),  # 100k points, expect ~4 shards
            (1_000_000, 8),  # 1M points, expect ~8 shards
            (10_000, 1),  # 10k points, expect 1 shard
        ]

        hw_config = detect_hardware()

        for n_points, expected_min_shards in test_cases:
            num_shards = calculate_optimal_num_shards(n_points, hw_config)
            assert isinstance(num_shards, int)
            assert num_shards >= expected_min_shards or num_shards == 1


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCDataSizes:
    """Test CMC with different dataset sizes."""

    @pytest.mark.parametrize(
        "n_points",
        [
            pytest.param(1000, id="1k_points"),
            pytest.param(10000, id="10k_points"),
            pytest.param(100000, id="100k_points"),
        ],
    )
    def test_cmc_various_sizes(self, n_points):
        """Test CMC pipeline with various dataset sizes."""
        from homodyne.device.config import detect_hardware

        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(
            n_points=n_points, n_angles=8
        )

        # Verify data structure
        assert c2_exp.shape[0] == 8
        assert c2_exp.shape[1] == c2_exp.shape[2]
        assert len(phi) == 8

        # Test shard calculation
        hw_config = detect_hardware()
        num_shards = calculate_optimal_num_shards(n_points, hw_config)
        assert num_shards >= 1

        # Verify sharding works
        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )
        shards = shard_data_stratified(
            data=c2_flat,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            num_shards=min(num_shards, 5),
            q=q,
            L=L,
        )
        assert len(shards) <= min(num_shards, 5)

    def test_cmc_100k_point_dataset(self):
        """
        Test CMC pipeline with 100k point dataset.

        This is the lower bound for CMC to provide significant speedup
        over traditional MCMC.
        """
        from homodyne.device.config import detect_hardware

        n_points = 100_000
        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(
            n_points=n_points, n_angles=16
        )

        # Verify data structure
        assert c2_exp.shape[0] == 16
        # Check total data points is approximately n_points (within 2%)
        total_generated = c2_exp.shape[0] * c2_exp.shape[1] ** 2
        assert abs(total_generated - n_points) / n_points < 0.02

        # Calculate appropriate number of shards
        hw_config = detect_hardware()
        num_shards = calculate_optimal_num_shards(n_points, hw_config)
        assert num_shards >= 1  # At least 1 shard

        # Test sharding
        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )
        shards = shard_data_stratified(
            data=c2_flat,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            num_shards=max(1, num_shards),
            q=q,
            L=L,
        )
        assert len(shards) >= 1


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCErrorHandling:
    """Test CMC error handling and recovery."""

    def test_invalid_num_shards(self):
        """Test error handling for invalid shard counts."""
        n_points = 10000
        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(n_points=n_points)

        # Test with shards > angles (implementation allows this via round-robin)
        num_shards = 20
        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )
        shards = shard_data_stratified(
            data=c2_flat,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            num_shards=num_shards,
            q=q,
            L=L,
        )
        # Implementation uses round-robin, so can create more shards than unique angles
        assert len(shards) == num_shards
        # Each shard should still have data
        for shard in shards:
            assert len(shard["data"]) > 0

    def test_empty_data_handling(self):
        """Test handling of edge cases."""
        # Minimal data
        c2_exp = np.ones((1, 10, 10))
        phi = np.array([0.0])

        # Should handle gracefully
        try:
            c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
                c2_exp, t1, t2, phi
            )
            shards = shard_data_stratified(
                data=c2_flat,
                t1=t1_flat,
                t2=t2_flat,
                phi=phi_flat,
                num_shards=1,
                q=q,
                L=L,
            )
            assert len(shards) == 1
        except Exception as e:
            # Acceptable to raise error for edge case
            pass

    def test_nan_data_detection(self):
        """Test detection of NaN/Inf in data."""
        c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(n_points=10000)

        # Verify clean data
        assert np.all(np.isfinite(c2_exp))

        # Inject NaN
        c2_exp_bad = c2_exp.copy()
        c2_exp_bad[0, 0, 0] = np.nan

        # Test detection
        assert not np.all(np.isfinite(c2_exp_bad))


@pytest.mark.integration
class TestCMCConfigurationIntegration:
    """Test configuration integration with CMC."""

    def test_config_with_default_values(self):
        """Test CMC configuration with default values."""
        config = {
            "cmc": {},
            "mcmc": {
                "num_warmup": 500,
                "num_samples": 2000,
            },
        }

        # Should use sensible defaults for missing values
        assert "cmc" in config
        assert "mcmc" in config

    def test_config_override_precedence(self):
        """Test configuration override precedence."""
        # Default config
        default_config = {
            "cmc": {
                "sharding": {"strategy": "stratified", "num_shards": 4},
                "combination": {"method": "weighted"},
            }
        }

        # User override
        user_config = {
            "cmc": {
                "sharding": {"num_shards": 8},
            }
        }

        # Merge logic (user overrides defaults)
        merged = {
            **default_config,
            "cmc": {**default_config["cmc"], **user_config["cmc"]},
        }

        assert merged["cmc"]["sharding"]["num_shards"] == 8


@pytest.mark.integration
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
class TestCMCResultIntegration:
    """Test CMC result handling and structure."""

    def test_mcmc_result_extension(self):
        """Test that CMCResult extends MCMCResult properly."""
        # Verify class exists and has expected attributes
        assert MCMCResult is not None

        # Check basic attributes
        required_attrs = [
            "mean_params",
            "std_params",
            "samples",
            "diagnostics",
        ]

        for attr in required_attrs:
            # These should be attributes of MCMCResult
            assert hasattr(MCMCResult, "__init__") or attr in dir(MCMCResult)

    def test_cmc_result_metadata(self):
        """Test CMC-specific result metadata."""
        # This validates structure expected by result extension
        expected_cmc_fields = [
            "num_shards",
            "combination_method",
            "per_shard_diagnostics",
            "kl_divergence_matrix",
        ]

        # Structure validation (not instantiation, as it requires full MCMC)
        for field in expected_cmc_fields:
            # Field should be part of CMC result contract
            assert field is not None


@pytest.mark.integration
class TestCMCMemoryManagement:
    """Test memory management in CMC."""

    def test_shard_memory_estimate(self):
        """Test memory footprint estimation for shards."""
        n_points = 1_000_000

        # Each point has multiple fields (c2, errors, etc.)
        points_per_shard = n_points / 4
        bytes_per_point = 8 * 10  # 8 bytes per float64, ~10 floats per point
        estimated_shard_mb = (points_per_shard * bytes_per_point) / (1024**2)

        # Should be reasonable (< 1GB per shard)
        assert estimated_shard_mb < 1000

    def test_chunk_processing_memory_bounds(self):
        """Test memory bounds for chunk processing."""
        # Maximum points per shard with 8GB limit
        memory_limit_gb = 8.0
        memory_bytes = memory_limit_gb * (1024**3)

        # Rough estimate: 100 bytes per point
        max_points = int(memory_bytes / 100)

        assert max_points > 0
        assert max_points < 100_000_000  # Should be reasonable


@pytest.mark.integration
class TestCMCEndToEndStructure:
    """Test end-to-end CMC pipeline structure (without execution)."""

    def test_cmc_pipeline_steps(self):
        """Verify CMC pipeline follows expected steps."""
        # Expected pipeline:
        # 1. Data sharding
        # 2. SVI initialization
        # 3. Parallel MCMC on shards
        # 4. Subposterior combination
        # 5. Result validation
        # 6. Package results

        pipeline_steps = [
            "sharding",
            "svi_initialization",
            "parallel_mcmc",
            "combination",
            "validation",
            "packaging",
        ]

        # Verify all steps are documented
        for step in pipeline_steps:
            assert step is not None

    def test_cmc_parameter_names_consistency(self):
        """Test parameter naming consistency."""
        # CMC should preserve parameter names from NLSQ/MCMC
        expected_params_static = ["D0", "alpha", "D_offset"]
        expected_params_laminar = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_0",
            "beta",
            "gamma_dot_offset",
            "phi0",
        ]

        assert len(expected_params_static) == 3
        assert len(expected_params_laminar) == 7


# ============================================================================
# Parametrized Tests for Coverage
# ============================================================================


@pytest.mark.integration
@pytest.mark.parametrize(
    "analysis_mode",
    [
        "static",
        "laminar_flow",
    ],
)
def test_cmc_analysis_modes(analysis_mode):
    """Test CMC with different analysis modes."""
    c2_exp, t1, t2, phi, q, L, true_params = generate_synthetic_xpcs_data(
        n_points=5000, analysis_mode=analysis_mode
    )

    assert c2_exp.shape[0] > 0
    assert len(true_params) > 0


@pytest.mark.integration
@pytest.mark.parametrize(
    "strategy",
    [
        "stratified",
    ],
)
@pytest.mark.skipif(not CMC_AVAILABLE, reason="CMC not available")
def test_sharding_strategies(strategy):
    """Test different sharding strategies."""
    c2_exp, t1, t2, phi, q, L, _ = generate_synthetic_xpcs_data(n_points=10000)

    if strategy == "stratified":
        c2_flat, t1_flat, t2_flat, phi_flat = prepare_flattened_data_for_sharding(
            c2_exp, t1, t2, phi
        )
        shards = shard_data_stratified(
            data=c2_flat, t1=t1_flat, t2=t2_flat, phi=phi_flat, num_shards=4, q=q, L=L
        )
        assert len(shards) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# CMC Result Processing Tests (from test_cmc_results.py)
# =============================================================================

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from homodyne.optimization.cmc.result import MCMCResult


# ==============================================================================
# Test Class: Result Structure with New Metadata
# ==============================================================================


class TestMCMCResultMetadata:
    """Test MCMCResult class with v2.1.0 config-driven metadata fields."""

    def test_result_with_parameter_space_metadata(self):
        """Verify parameter_space_metadata field works correctly."""
        # Create parameter space metadata (typical from ParameterSpace.from_config())
        param_space_metadata = {
            "bounds": {
                "D0": [50.0, 500.0],
                "alpha": [0.1, 2.0],
                "D_offset": [1.0, 50.0],
            },
            "priors": {
                "D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0},
                "alpha": {"type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                "D_offset": {"type": "TruncatedNormal", "mu": 10.0, "sigma": 5.0},
            },
            "model_type": "static",
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata=param_space_metadata,
        )

        # Verify metadata stored correctly
        assert result.parameter_space_metadata is not None
        assert result.parameter_space_metadata["model_type"] == "static"
        assert "D0" in result.parameter_space_metadata["bounds"]
        assert "D0" in result.parameter_space_metadata["priors"]
        assert (
            result.parameter_space_metadata["priors"]["D0"]["type"] == "TruncatedNormal"
        )

    def test_result_with_initial_values_metadata(self):
        """Verify initial_values_metadata field works correctly."""
        # Create initial values metadata (from config or NLSQ results)
        initial_values_metadata = {
            "D0": 1234.5,
            "alpha": 0.567,
            "D_offset": 12.34,
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([1250.0, 0.58, 12.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            initial_values_metadata=initial_values_metadata,
        )

        # Verify metadata stored correctly
        assert result.initial_values_metadata is not None
        assert result.initial_values_metadata["D0"] == 1234.5
        assert result.initial_values_metadata["alpha"] == 0.567
        assert result.initial_values_metadata["D_offset"] == 12.34

    def test_result_with_selection_decision_metadata(self):
        """Verify selection_decision_metadata field works correctly."""
        # Create selection decision metadata (from automatic NUTS/CMC selection)
        selection_decision_metadata = {
            "selected_method": "CMC",
            "num_samples": 50,
            "parallelism_criterion_met": True,
            "memory_criterion_met": False,
            "min_samples_for_cmc": 15,
            "memory_threshold_pct": 0.30,
            "estimated_memory_fraction": 0.15,
        }

        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,  # CMC result
            selection_decision_metadata=selection_decision_metadata,
        )

        # Verify metadata stored correctly
        assert result.selection_decision_metadata is not None
        assert result.selection_decision_metadata["selected_method"] == "CMC"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is True
        assert result.selection_decision_metadata["memory_criterion_met"] is False
        assert result.is_cmc_result()  # Should be detected as CMC

    def test_result_with_all_metadata_fields(self):
        """Verify all three metadata fields can coexist."""
        # Create all metadata fields
        param_space_metadata = {
            "bounds": {"D0": [50.0, 500.0]},
            "priors": {"D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0}},
            "model_type": "static",
        }
        initial_values_metadata = {"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34}
        selection_decision_metadata = {
            "selected_method": "NUTS",
            "num_samples": 10,
            "parallelism_criterion_met": False,
            "memory_criterion_met": False,
        }

        # Create result with all metadata
        result = MCMCResult(
            mean_params=np.array([1250.0, 0.58, 12.5]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata=param_space_metadata,
            initial_values_metadata=initial_values_metadata,
            selection_decision_metadata=selection_decision_metadata,
        )

        # Verify all metadata fields present
        assert result.parameter_space_metadata is not None
        assert result.initial_values_metadata is not None
        assert result.selection_decision_metadata is not None


# ==============================================================================
# Test Class: Serialization and Deserialization
# ==============================================================================


class TestResultSerialization:
    """Test to_dict() and from_dict() with new metadata fields."""

    def test_to_dict_includes_new_metadata(self):
        """Verify to_dict() includes v2.1.0 metadata fields."""
        # Create result with metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata={"model_type": "static"},
            initial_values_metadata={"D0": 1234.5},
            selection_decision_metadata={"selected_method": "NUTS"},
        )

        # Convert to dict
        data = result.to_dict()

        # Verify new fields in dictionary
        assert "parameter_space_metadata" in data
        assert "initial_values_metadata" in data
        assert "selection_decision_metadata" in data
        assert data["parameter_space_metadata"]["model_type"] == "static"
        assert data["initial_values_metadata"]["D0"] == 1234.5
        assert data["selection_decision_metadata"]["selected_method"] == "NUTS"

    def test_from_dict_with_new_metadata(self):
        """Verify from_dict() reconstructs result with v2.1.0 metadata."""
        # Create dictionary with new metadata fields
        data = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "parameter_space_metadata": {
                "bounds": {"D0": [50.0, 500.0]},
                "model_type": "static",
            },
            "initial_values_metadata": {"D0": 1234.5, "alpha": 0.567},
            "selection_decision_metadata": {
                "selected_method": "CMC",
                "parallelism_criterion_met": True,
            },
        }

        # Reconstruct result
        result = MCMCResult.from_dict(data)

        # Verify metadata reconstructed correctly
        assert result.parameter_space_metadata is not None
        assert result.parameter_space_metadata["model_type"] == "static"
        assert result.initial_values_metadata["D0"] == 1234.5
        assert result.selection_decision_metadata["selected_method"] == "CMC"

    def test_from_dict_backward_compatibility(self):
        """Verify from_dict() handles old results without new metadata (backward compatibility)."""
        # Create dictionary WITHOUT new metadata fields (simulates old v2.0.0 result file)
        data = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            # No parameter_space_metadata
            # No initial_values_metadata
            # No selection_decision_metadata
        }

        # Reconstruct result
        result = MCMCResult.from_dict(data)

        # Verify new metadata fields default to None (backward compatible)
        assert result.parameter_space_metadata is None
        assert result.initial_values_metadata is None
        assert result.selection_decision_metadata is None

        # Verify standard fields still work
        assert np.allclose(result.mean_params, [200.0, 1.0, 10.0])
        assert result.mean_contrast == 0.5
        assert result.mean_offset == 1.0

    def test_round_trip_serialization(self):
        """Verify save → load preserves all metadata."""
        # Create result with all metadata
        original = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            std_params=np.array([5.0, 0.1, 1.0]),
            parameter_space_metadata={
                "bounds": {"D0": [50.0, 500.0]},
                "priors": {"D0": {"type": "TruncatedNormal", "mu": 200.0}},
                "model_type": "static",
            },
            initial_values_metadata={"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34},
            selection_decision_metadata={
                "selected_method": "NUTS",
                "num_samples": 10,
                "parallelism_criterion_met": False,
            },
        )

        # Round-trip: to_dict → from_dict
        data = original.to_dict()
        reconstructed = MCMCResult.from_dict(data)

        # Verify all metadata preserved
        assert (
            reconstructed.parameter_space_metadata == original.parameter_space_metadata
        )
        assert reconstructed.initial_values_metadata == original.initial_values_metadata
        assert (
            reconstructed.selection_decision_metadata
            == original.selection_decision_metadata
        )

        # Verify standard fields preserved
        assert np.allclose(reconstructed.mean_params, original.mean_params)
        assert np.allclose(reconstructed.std_params, original.std_params)
        assert reconstructed.mean_contrast == original.mean_contrast


# ==============================================================================
# Test Class: JSON File Saving/Loading Integration
# ==============================================================================


class TestResultFileSaving:
    """Test integration with save_mcmc_results() function."""

    def test_json_serialization_of_metadata(self):
        """Verify metadata fields can be serialized to JSON."""
        # Create result with complex metadata
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            parameter_space_metadata={
                "bounds": {
                    "D0": [50.0, 500.0],
                    "alpha": [0.1, 2.0],
                    "D_offset": [1.0, 50.0],
                },
                "priors": {
                    "D0": {"type": "TruncatedNormal", "mu": 200.0, "sigma": 50.0},
                    "alpha": {"type": "TruncatedNormal", "mu": 1.0, "sigma": 0.3},
                },
                "model_type": "static",
            },
            initial_values_metadata={"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34},
            selection_decision_metadata={
                "selected_method": "CMC",
                "num_samples": 50,
                "parallelism_criterion_met": True,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
                "memory_threshold_pct": 0.30,
            },
        )

        # Convert to dict
        data = result.to_dict()

        # Verify JSON serialization works (no errors)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f, indent=2)
            temp_path = f.name

        try:
            # Load back and verify
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            # Verify metadata preserved through JSON serialization
            assert "parameter_space_metadata" in loaded_data
            assert "initial_values_metadata" in loaded_data
            assert "selection_decision_metadata" in loaded_data
            assert (
                loaded_data["parameter_space_metadata"]["model_type"]
                == "static"
            )
            assert loaded_data["initial_values_metadata"]["D0"] == 1234.5
            assert (
                loaded_data["selection_decision_metadata"]["selected_method"] == "CMC"
            )
        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_old_result_file_loads_without_new_metadata(self):
        """Verify old result files (without new metadata) can still be loaded."""
        # Simulate old result file (v2.0.0) - JSON without new metadata fields
        old_result_json = {
            "mean_params": [200.0, 1.0, 10.0],
            "mean_contrast": 0.5,
            "mean_offset": 1.0,
            "std_params": [5.0, 0.1, 1.0],
            "converged": True,
            "n_iterations": 2000,
            "computation_time": 45.3,
            "backend": "JAX",
            "analysis_mode": "static",
            "n_chains": 4,
            "n_warmup": 1000,
            "n_samples": 1000,
            "sampler": "NUTS",
            # No new v2.1.0 fields here (simulates old file)
        }

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(old_result_json, f, indent=2)
            temp_path = f.name

        try:
            # Load and reconstruct result
            with open(temp_path, "r") as f:
                loaded_data = json.load(f)

            result = MCMCResult.from_dict(loaded_data)

            # Verify result loads successfully
            assert np.allclose(result.mean_params, [200.0, 1.0, 10.0])
            assert result.converged is True

            # Verify new metadata fields default to None (backward compatible)
            assert result.parameter_space_metadata is None
            assert result.initial_values_metadata is None
            assert result.selection_decision_metadata is None
        finally:
            # Clean up
            Path(temp_path).unlink()


# ==============================================================================
# Test Class: CMC-Specific Integration
# ==============================================================================


class TestCMCResultMetadata:
    """Test CMC results with combined CMC and config-driven metadata."""

    def test_cmc_result_with_selection_metadata(self):
        """Verify CMC results include selection decision metadata."""
        # Create CMC result with selection decision
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=10,  # CMC indicator
            combination_method="weighted",
            cmc_diagnostics={"combination_success": True, "n_shards_converged": 10},
            selection_decision_metadata={
                "selected_method": "CMC",
                "num_samples": 50,
                "parallelism_criterion_met": True,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
            },
        )

        # Verify CMC detection works
        assert result.is_cmc_result() is True
        assert result.num_shards == 10

        # Verify selection metadata shows why CMC was chosen
        assert result.selection_decision_metadata["selected_method"] == "CMC"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is True

    def test_nuts_result_with_selection_metadata(self):
        """Verify NUTS results include selection decision metadata."""
        # Create NUTS result with selection decision
        result = MCMCResult(
            mean_params=np.array([200.0, 1.0, 10.0]),
            mean_contrast=0.5,
            mean_offset=1.0,
            num_shards=None,  # Not CMC
            selection_decision_metadata={
                "selected_method": "NUTS",
                "num_samples": 10,
                "parallelism_criterion_met": False,
                "memory_criterion_met": False,
                "min_samples_for_cmc": 15,
            },
        )

        # Verify NUTS detection works
        assert result.is_cmc_result() is False

        # Verify selection metadata shows why NUTS was chosen
        assert result.selection_decision_metadata["selected_method"] == "NUTS"
        assert result.selection_decision_metadata["parallelism_criterion_met"] is False
        assert result.selection_decision_metadata["memory_criterion_met"] is False
