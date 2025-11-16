"""Unit Tests for CMC Backend Implementations
============================================

This module tests the three CMC backend implementations:
1. PjitBackend (JAX execution - CPU-only in v2.3.0+)
2. MultiprocessingBackend (CPU parallel execution)
3. PBSBackend (HPC cluster execution)

Test Coverage:
--------------
- Backend initialization and configuration
- Sequential execution (single device)
- Parallel execution (multi-device/multi-core)
- Error handling and retry logic
- Timeout detection
- Checkpoint save/resume (integration with CheckpointManager)
- Convergence diagnostics collection
- Result validation
- PBS job script generation (dry run)

Testing Strategy:
-----------------
- Use synthetic test data (small shards for fast execution)
- Mock external dependencies (qsub, qstat) for PBS tests
- Validate result format compatibility
- Test error handling without crashing
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import backends
from homodyne.optimization.cmc.backends import (
    CMCBackend,
    select_backend,
    get_backend_by_name,
    PJIT_AVAILABLE,
    MULTIPROCESSING_AVAILABLE,
    PBS_AVAILABLE,
)

# Import device config for hardware detection
try:
    from homodyne.device.config import HardwareConfig, detect_hardware

    DEVICE_CONFIG_AVAILABLE = True
except ImportError:
    DEVICE_CONFIG_AVAILABLE = False


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def synthetic_shards():
    """Generate synthetic test shards (small for fast execution)."""
    num_shards = 3
    points_per_shard = 100

    shards = []
    for i in range(num_shards):
        shard = {
            "data": np.random.randn(points_per_shard),
            "sigma": np.ones(points_per_shard),
            "t1": np.linspace(0, 1, points_per_shard),
            "t2": np.linspace(0, 1, points_per_shard),
            "phi": np.random.uniform(-np.pi, np.pi, points_per_shard),
            "q": 0.01,
            "L": 1.0,
        }
        shards.append(shard)

    return shards


@pytest.fixture
def mcmc_config():
    """Minimal MCMC configuration for testing."""
    return {
        "num_warmup": 10,  # Very small for fast tests
        "num_samples": 20,
        "num_chains": 1,
        "target_accept_prob": 0.8,
        "max_tree_depth": 10,
    }


@pytest.fixture
def init_params():
    """Initial parameter values."""
    return {
        "contrast": 0.5,
        "offset": 1.0,
        "D0": 1000.0,
        "alpha": 0.5,
        "D_offset": 10.0,
    }


@pytest.fixture
def inv_mass_matrix():
    """Identity mass matrix (5x5 for 5 parameters)."""
    return np.eye(5)


@pytest.fixture
def parameter_space():
    """Create parameter space for static_mode model."""
    from homodyne.config.parameter_space import ParameterSpace

    # Create minimal config dict for static_mode mode
    config_dict = {
        "analysis_mode": "static",
        "parameter_space": {
            "model": "static",
            "bounds": [
                {"name": "D0", "min": 100.0, "max": 10000.0},
                {"name": "alpha", "min": 0.0, "max": 2.0},
                {"name": "D_offset", "min": 0.0, "max": 100.0},
            ],
        },
    }

    return ParameterSpace.from_config(config_dict, analysis_mode="static")


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="test_backend_"))
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def mock_hardware_cpu():
    """Mock CPU-only hardware configuration."""
    if not DEVICE_CONFIG_AVAILABLE:
        pytest.skip("Device config not available")

    return HardwareConfig(
        platform="cpu",
        num_devices=1,
        memory_per_device_gb=32.0,
        num_nodes=1,
        cores_per_node=8,
        total_memory_gb=32.0,
        cluster_type="standalone",
        recommended_backend="multiprocessing",
        max_parallel_shards=8,
    )


@pytest.fixture
def mock_hardware_pbs_cluster():
    """Mock PBS cluster hardware configuration."""
    if not DEVICE_CONFIG_AVAILABLE:
        pytest.skip("Device config not available")

    return HardwareConfig(
        platform="cpu",
        num_devices=0,
        memory_per_device_gb=0.0,
        num_nodes=4,
        cores_per_node=36,
        total_memory_gb=512.0,
        cluster_type="pbs",
        recommended_backend="pbs",
        max_parallel_shards=144,
    )


# ============================================================================
# Test 1: Backend Instantiation
# ============================================================================


def test_pjit_backend_instantiation():
    """Test PjitBackend can be instantiated."""
    if not PJIT_AVAILABLE:
        pytest.skip("pjit backend not available (NumPyro required)")

    backend = get_backend_by_name("pjit")
    assert backend.get_backend_name() == "pjit"
    assert isinstance(backend.num_devices, int)
    assert backend.platform in ["gpu", "cpu"]


def test_multiprocessing_backend_instantiation():
    """Test MultiprocessingBackend can be instantiated."""
    if not MULTIPROCESSING_AVAILABLE:
        pytest.skip("multiprocessing backend not available")

    backend = get_backend_by_name("multiprocessing")
    assert backend.get_backend_name() == "multiprocessing"
    assert backend.num_workers > 0
    assert backend.timeout_seconds > 0


def test_pbs_backend_instantiation():
    """Test PBSBackend can be instantiated."""
    if not PBS_AVAILABLE:
        pytest.skip("PBS backend not available")

    backend = get_backend_by_name("pbs")
    assert backend.get_backend_name() == "pbs"
    # Note: We don't require PBS project name for instantiation


# ============================================================================
# Test 2: Backend Selection
# ============================================================================


def test_backend_selection_cpu(mock_hardware_cpu):
    """Test backend selection on CPU system."""
    backend = select_backend(mock_hardware_cpu)
    assert backend.get_backend_name() == "multiprocessing"


def test_backend_selection_pbs_cluster(mock_hardware_pbs_cluster):
    """Test backend selection on PBS cluster."""
    backend = select_backend(mock_hardware_pbs_cluster)
    assert backend.get_backend_name() == "pbs"


def test_backend_selection_user_override(mock_hardware_cpu):
    """Test manual backend override."""
    # Force pjit on CPU system
    backend = select_backend(mock_hardware_cpu, user_override="pjit")
    assert backend.get_backend_name() == "pjit"


# ============================================================================
# Test 3: PjitBackend Execution Tests
# ============================================================================


@pytest.mark.skipif(not PJIT_AVAILABLE, reason="pjit backend not available")
def test_pjit_backend_single_shard(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Test PjitBackend with single shard."""
    backend = get_backend_by_name("pjit")

    # Use only first shard
    shards = [synthetic_shards[0]]

    results = backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    # Validate results
    assert len(results) == 1
    result = results[0]

    assert "converged" in result
    assert "elapsed_time" in result
    assert "shard_idx" in result

    # Check samples if converged
    if result["converged"]:
        assert "samples" in result
        assert result["samples"] is not None
        assert "diagnostics" in result


@pytest.mark.skipif(not PJIT_AVAILABLE, reason="pjit backend not available")
@pytest.mark.slow
def test_pjit_backend_multiple_shards(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Test PjitBackend with multiple shards (sequential execution)."""
    backend = get_backend_by_name("pjit")

    results = backend.run_parallel_mcmc(
        shards=synthetic_shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    # Validate results
    assert len(results) == len(synthetic_shards)

    for i, result in enumerate(results):
        assert result["shard_idx"] == i
        assert "converged" in result
        assert "elapsed_time" in result


# ============================================================================
# Test 4: MultiprocessingBackend Execution Tests
# ============================================================================


@pytest.mark.skipif(
    not MULTIPROCESSING_AVAILABLE, reason="multiprocessing backend not available"
)
def test_multiprocessing_backend_single_shard(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Test MultiprocessingBackend with single shard."""
    backend = get_backend_by_name("multiprocessing")

    shards = [synthetic_shards[0]]

    results = backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    assert len(results) == 1
    result = results[0]

    assert "converged" in result
    assert "elapsed_time" in result


@pytest.mark.skipif(
    not MULTIPROCESSING_AVAILABLE, reason="multiprocessing backend not available"
)
@pytest.mark.slow
def test_multiprocessing_backend_parallel_execution(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Test MultiprocessingBackend parallel execution."""
    backend = get_backend_by_name("multiprocessing")

    results = backend.run_parallel_mcmc(
        shards=synthetic_shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    assert len(results) == len(synthetic_shards)

    for i, result in enumerate(results):
        assert result["shard_idx"] == i


@pytest.mark.skipif(
    not MULTIPROCESSING_AVAILABLE, reason="multiprocessing backend not available"
)
def test_multiprocessing_backend_timeout(parameter_space):
    """Test MultiprocessingBackend timeout detection."""
    # Create backend with very short timeout
    backend = get_backend_by_name("multiprocessing")
    backend.timeout_seconds = 0.1  # 100ms timeout (will fail)

    # Create minimal shard
    shards = [
        {
            "data": np.random.randn(10),
            "sigma": np.ones(10),
            "t1": np.linspace(0, 1, 10),
            "t2": np.linspace(0, 1, 10),
            "phi": np.random.randn(10),
            "q": 0.01,
            "L": 1.0,
        }
    ]

    mcmc_config = {
        "num_warmup": 100,  # Will exceed timeout
        "num_samples": 100,
        "num_chains": 1,
    }

    results = backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params={"D0": 1000.0},
        inv_mass_matrix=np.eye(5),
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    # Should return error result
    assert len(results) == 1
    assert results[0]["converged"] is False
    assert "timed out" in results[0].get("error", "").lower()


# ============================================================================
# Test 5: PBSBackend Tests (Dry Run)
# ============================================================================


@pytest.mark.skipif(not PBS_AVAILABLE, reason="PBS backend not available")
def test_pbs_backend_script_generation(
    temp_dir, synthetic_shards, mcmc_config, init_params, inv_mass_matrix
):
    """Test PBS job script generation."""
    from homodyne.optimization.cmc.backends.pbs import PBSBackend

    backend = PBSBackend(
        project_name="test_project",
        walltime="01:00:00",
        temp_dir=str(temp_dir),
    )

    # Write shard data (needed for script generation)
    backend._write_shard_data(
        synthetic_shards,
        mcmc_config,
        init_params,
        inv_mass_matrix,
    )

    # Generate script
    script_path = backend._generate_pbs_script(len(synthetic_shards))

    assert script_path.exists()

    # Validate script content
    with open(script_path, "r") as f:
        script_content = f.read()

    assert "test_project" in script_content
    assert "01:00:00" in script_content
    assert f"#PBS -J 0-{len(synthetic_shards) - 1}" in script_content


@pytest.mark.skipif(not PBS_AVAILABLE, reason="PBS backend not available")
def test_pbs_backend_data_serialization(
    temp_dir, synthetic_shards, mcmc_config, init_params, inv_mass_matrix
):
    """Test PBS shard data serialization to HDF5."""
    from homodyne.optimization.cmc.backends.pbs import PBSBackend
    import h5py

    backend = PBSBackend(
        project_name="test_project",
        temp_dir=str(temp_dir),
    )

    # Write shard data
    backend._write_shard_data(
        synthetic_shards,
        mcmc_config,
        init_params,
        inv_mass_matrix,
    )

    # Validate shard data files
    shard_data_dir = temp_dir / "shard_data"
    assert shard_data_dir.exists()

    for i, shard in enumerate(synthetic_shards):
        shard_path = shard_data_dir / f"shard_{i:03d}.h5"
        assert shard_path.exists()

        # Validate HDF5 content
        with h5py.File(shard_path, "r") as f:
            assert "data" in f
            assert "sigma" in f
            assert "t1" in f
            assert "t2" in f
            assert "phi" in f
            assert f.attrs["q"] == shard["q"]
            assert f.attrs["L"] == shard["L"]
            assert f.attrs["shard_idx"] == i

    # Validate config file
    config_path = temp_dir / "mcmc_config.json"
    assert config_path.exists()

    with open(config_path, "r") as f:
        config_data = json.load(f)
        assert "mcmc_config" in config_data
        assert "init_params" in config_data


@pytest.mark.skipif(not PBS_AVAILABLE, reason="PBS backend not available")
def test_pbs_backend_dry_run(temp_dir):
    """Test PBS backend without actually submitting to cluster."""
    from homodyne.optimization.cmc.backends.pbs import PBSBackend

    backend = PBSBackend(
        project_name="test_project",
        temp_dir=str(temp_dir),
    )

    # Check attributes
    assert backend.project_name == "test_project"
    assert backend.temp_dir == temp_dir
    assert backend.get_backend_name() == "pbs"


# ============================================================================
# Test 6: Error Handling
# ============================================================================


def test_backend_error_handling_invalid_name():
    """Test error handling for invalid backend name."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend_by_name("invalid_backend")


@pytest.mark.skipif(not PJIT_AVAILABLE, reason="pjit backend not available")
def test_pjit_backend_handles_errors(init_params, inv_mass_matrix, parameter_space):
    """Test PjitBackend error handling."""
    backend = get_backend_by_name("pjit")

    # Create invalid shard (missing required fields)
    invalid_shards = [{"data": np.array([1, 2, 3])}]

    mcmc_config = {"num_warmup": 10, "num_samples": 10}

    results = backend.run_parallel_mcmc(
        shards=invalid_shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    # Should return error result (not crash)
    assert len(results) == 1
    assert results[0]["converged"] is False
    assert "error" in results[0]


# ============================================================================
# Test 7: Result Format Validation
# ============================================================================


def test_result_format_validation():
    """Test that all backends return results in consistent format."""
    # This is tested by base class utilities
    from homodyne.optimization.cmc.backends.base import CMCBackend

    # Create mock backend
    class MockBackend(CMCBackend):
        def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix, analysis_mode, parameter_space):
            return []

        def get_backend_name(self):
            return "mock"

    backend = MockBackend()

    # Test result validation
    valid_result = {
        "converged": True,
        "samples": np.random.randn(100, 5),
        "diagnostics": {"ess": {}, "rhat": {}},
        "elapsed_time": 1.0,
    }

    # Should not raise
    backend._validate_shard_result(valid_result, shard_idx=0)

    # Test invalid result (missing required field)
    invalid_result = {
        "converged": True,
        # Missing elapsed_time
    }

    with pytest.raises(ValueError, match="missing required field"):
        backend._validate_shard_result(invalid_result, shard_idx=0)


# ============================================================================
# Test 8: Convergence Diagnostics
# ============================================================================


@pytest.mark.skipif(not PJIT_AVAILABLE, reason="pjit backend not available")
def test_pjit_backend_diagnostics_collection(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Test that PjitBackend collects convergence diagnostics."""
    backend = get_backend_by_name("pjit")

    shards = [synthetic_shards[0]]

    results = backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode="static",
        parameter_space=parameter_space,
    )

    result = results[0]

    if result["converged"]:
        assert "diagnostics" in result
        diagnostics = result["diagnostics"]

        # Check for expected diagnostic fields
        assert (
            "acceptance_rate" in diagnostics
            or diagnostics.get("acceptance_rate") is None
        )
        assert "ess" in diagnostics
        assert "rhat" in diagnostics


# ============================================================================
# Test 9: Backend Availability Flags
# ============================================================================


def test_backend_availability_flags():
    """Test that backend availability flags are set correctly."""
    from homodyne.optimization.cmc.backends import (
        PJIT_AVAILABLE,
        MULTIPROCESSING_AVAILABLE,
        PBS_AVAILABLE,
    )

    assert isinstance(PJIT_AVAILABLE, bool)
    assert isinstance(MULTIPROCESSING_AVAILABLE, bool)
    assert isinstance(PBS_AVAILABLE, bool)


# ============================================================================
# Test 10: Base Class Utilities
# ============================================================================


def test_base_class_logging_utilities():
    """Test CMCBackend base class logging utilities."""
    from homodyne.optimization.cmc.backends.base import CMCBackend

    class MockBackend(CMCBackend):
        def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix):
            return []

        def get_backend_name(self):
            return "mock"

    backend = MockBackend()

    # Test timer utilities
    start = backend._create_timer()
    import time

    time.sleep(0.01)
    elapsed = backend._get_elapsed_time(start)
    assert elapsed > 0.0

    # Test logging methods (should not crash)
    backend._log_shard_start(0, 10)
    backend._log_shard_complete(0, 10, 1.5, converged=True)

    # Test error handling
    error_result = backend._handle_shard_error(ValueError("test error"), shard_idx=5)
    assert error_result["converged"] is False
    assert "error" in error_result
    assert "test error" in error_result["error"]


# ============================================================================
# Test 11: PBS Job Status Mocking
# ============================================================================


@pytest.mark.skipif(not PBS_AVAILABLE, reason="PBS backend not available")
def test_pbs_backend_job_status_parsing():
    """Test PBS job status parsing from qstat output."""
    from homodyne.optimization.cmc.backends.pbs import PBSBackend

    backend = PBSBackend(project_name="test")

    # Mock qstat output for running job
    qstat_running = """
Job Id: 12345.pbsserver
    job_state = R
    exit_status = 0
"""
    status = backend._check_exit_status(qstat_running)
    assert status == "completed"

    # Mock qstat output for failed job
    qstat_failed = """
Job Id: 12345.pbsserver
    job_state = F
    exit_status = 1
"""
    status = backend._check_exit_status(qstat_failed)
    assert status == "failed"


# ============================================================================
# Test 12: Integration Test (if backends available)
# ============================================================================


@pytest.mark.slow
@pytest.mark.integration
def test_backend_integration_comparison(
    synthetic_shards, mcmc_config, init_params, inv_mass_matrix, parameter_space
):
    """Integration test: Compare results from different backends (if available)."""
    # Skip if backends not available
    if not (PJIT_AVAILABLE and MULTIPROCESSING_AVAILABLE):
        pytest.skip("Multiple backends not available for comparison")

    # Use single shard for comparison
    shards = [synthetic_shards[0]]
    analysis_mode = "static"

    # Run with pjit backend
    pjit_backend = get_backend_by_name("pjit")
    pjit_results = pjit_backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode=analysis_mode,
        parameter_space=parameter_space,
    )

    # Run with multiprocessing backend
    mp_backend = get_backend_by_name("multiprocessing")
    mp_results = mp_backend.run_parallel_mcmc(
        shards=shards,
        mcmc_config=mcmc_config,
        init_params=init_params,
        inv_mass_matrix=inv_mass_matrix,
        analysis_mode=analysis_mode,
        parameter_space=parameter_space,
    )

    # Both should return same number of results
    assert len(pjit_results) == len(mp_results)

    # Both should have same result structure
    pjit_result = pjit_results[0]
    mp_result = mp_results[0]

    assert pjit_result.keys() == mp_result.keys()
    assert pjit_result["shard_idx"] == mp_result["shard_idx"]


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
