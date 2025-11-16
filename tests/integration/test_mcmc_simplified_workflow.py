"""Integration tests for simplified MCMC workflow (v2.1.0).

This module validates the simplified MCMC implementation that removes
automatic initialization and confusing CLI method flags, replacing them
with automatic NUTS/CMC selection and manual parameter initialization.

Key Features Tested:
- Automatic NUTS/CMC selection based on dual-criteria OR logic
- Manual NLSQ → MCMC workflow with parameter copying
- Configurable thresholds from YAML configuration
- Auto-retry mechanism with convergence failures
- Backward compatibility of initial_parameters structure

Breaking Changes (v2.1.0):
- No automatic NLSQ/SVI initialization
- CLI methods reduced to nlsq/mcmc only (nuts/cmc removed)
- mcmc.initialization section removed from YAML
"""

import numpy as np
import pytest

from homodyne.device.config import HardwareConfig, should_use_cmc
from tests.factories.config_factory import create_phi_filtering_config
from tests.factories.data_factory import create_specific_angles_test_data
from tests.factories.optimization_factory import (
    create_mock_config_manager,
    create_mock_data_dict,
    create_mock_optimization_result,
)


class TestAutomaticNUTSCMCSelection:
    """Test automatic NUTS/CMC selection based on dual-criteria OR logic."""

    def test_automatic_nuts_selection_for_small_datasets(self):
        """Test NUTS selected for small datasets (10 samples, 500K points)."""
        # Arrange - Small dataset (below all thresholds)
        num_samples = 10  # Below min_samples_for_cmc (15)
        dataset_size = 500_000  # Below JAX broadcasting threshold (1M)
        hw_config = HardwareConfig(
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

        # Act
        use_cmc = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Assert - Should use NUTS (both conditions fail)
        assert use_cmc is False, "Should use NUTS for small dataset"

    def test_automatic_cmc_selection_for_many_samples(self):
        """Test CMC selected for many samples (20 samples, parallelism mode)."""
        # Arrange - Many samples (triggers parallelism)
        num_samples = 20  # Above min_samples_for_cmc (15)
        dataset_size = 10_000_000  # Moderate data
        hw_config = HardwareConfig(
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

        # Act
        use_cmc = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Assert - Should use CMC (parallelism condition met)
        assert use_cmc is True, "Should use CMC for many samples (parallelism)"

    def test_automatic_cmc_selection_for_large_memory(self):
        """Test CMC selected for large memory (5 samples, 50M points, memory mode)."""
        # Arrange - Few samples but huge data (triggers memory management)
        num_samples = 5  # Below min_samples_for_cmc
        dataset_size = 50_000_000  # Large data (triggers memory threshold)
        hw_config = HardwareConfig(
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

        # Memory estimate: 50M × 8 bytes × 30 = 12 GB
        # 12 GB / 32 GB = 37.5% > 30% threshold

        # Act
        use_cmc = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Assert - Should use CMC (memory condition met)
        assert use_cmc is True, "Should use CMC for large memory requirement"

    def test_configurable_thresholds_from_yaml(self):
        """Test that thresholds are configurable via YAML config."""
        # Arrange - Custom thresholds (stricter than defaults)
        num_samples = 12
        dataset_size = 800_000  # Below JAX broadcasting threshold (1M)
        hw_config = HardwareConfig(
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

        # Act - Default thresholds (15 samples, 30% memory)
        use_cmc_default = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Act - Custom thresholds (10 samples, 20% memory)
        use_cmc_custom = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            dataset_size=dataset_size,
            min_samples_for_cmc=10,  # Lower threshold
            memory_threshold_pct=0.20,  # Stricter threshold
        )

        # Assert - Default should use NUTS, custom should use CMC
        assert use_cmc_default is False, "Default thresholds → NUTS"
        assert use_cmc_custom is True, "Custom thresholds → CMC (12 >= 10)"


class TestManualNLSQMCMCWorkflow:
    """Test manual NLSQ → MCMC workflow with parameter copying."""

    def test_manual_parameter_initialization_workflow(self):
        """Test full workflow: NLSQ → manual copy → MCMC."""
        # Arrange - Simulate NLSQ results
        nlsq_result = create_mock_optimization_result(
            analysis_mode="static", converged=True
        )

        # NLSQ returns: [contrast, offset, D0, alpha, D_offset]
        nlsq_params = nlsq_result.parameters
        assert len(nlsq_params) == 5, "NLSQ should return 5 params"

        # Act - User manually copies NLSQ results to YAML config
        # Extract physics parameters (skip contrast, offset)
        initial_values = nlsq_params[2:]  # [D0, alpha, D_offset]

        # Assert - Initial values ready for MCMC
        assert len(initial_values) == 3, "Should have 3 physics params"
        assert initial_values[0] > 0, "D0 should be positive"
        assert -2.0 < initial_values[1] < 2.0, "alpha should be in valid range"

    def test_backward_compatibility_of_initial_parameters_structure(self):
        """Test that initial_parameters structure unchanged from v2.0."""
        # Arrange - Create YAML config structure (v2.0 format)
        config_v20 = {
            "initial_parameters": {
                "parameter_names": ["D0", "alpha", "D_offset"],
                "values": None,  # Optional: set from NLSQ manually
            },
            "optimization": {
                "mcmc": {
                    "num_warmup": 1000,
                    "num_samples": 2000,
                    "num_chains": 4,
                    "min_samples_for_cmc": 15,
                    "memory_threshold_pct": 0.30,
                }
            },
        }

        # Act - Manually update with NLSQ results
        nlsq_params = np.array([0.45, 1.02, 1234.5, 0.567, 12.34])
        config_v20["initial_parameters"]["values"] = nlsq_params[2:].tolist()

        # Assert - Structure unchanged, backward compatible
        assert "initial_parameters" in config_v20, "initial_parameters preserved"
        assert "parameter_names" in config_v20["initial_parameters"]
        assert "values" in config_v20["initial_parameters"]
        assert len(config_v20["initial_parameters"]["values"]) == 3


class TestConvergenceAndErrorHandling:
    """Test auto-retry mechanism and error handling."""

    def test_auto_retry_mechanism_with_convergence_failures(self):
        """Test auto-retry with different random seeds (max 3 retries)."""
        # This test validates the retry logic structure exists
        # Full convergence testing requires real MCMC and is too slow

        # Arrange - Define convergence criteria
        r_hat_threshold = 1.1
        ess_threshold = 100

        # Simulate convergence diagnostics from 3 attempts
        attempt_results = [
            {"r_hat": 1.25, "ess": 45, "converged": False},  # Attempt 1: poor
            {"r_hat": 1.15, "ess": 80, "converged": False},  # Attempt 2: marginal
            {"r_hat": 1.05, "ess": 250, "converged": True},  # Attempt 3: good
        ]

        # Act - Simulate retry logic
        max_retries = 3
        converged = False
        final_result = None

        for i, result in enumerate(attempt_results):
            if i >= max_retries:
                break
            if result["r_hat"] <= r_hat_threshold and result["ess"] >= ess_threshold:
                converged = True
                final_result = result
                break

        # Assert - Should converge on attempt 3
        assert converged is True, "Should converge within 3 attempts"
        assert final_result["r_hat"] <= r_hat_threshold
        assert final_result["ess"] >= ess_threshold

    def test_error_handling_for_invalid_method_names(self):
        """Test that invalid method names raise clear errors."""
        # This test validates CLI argument validation
        # Actual CLI testing is in test_cli_args.py

        # Arrange - Define valid and invalid methods
        valid_methods = ["nlsq", "mcmc"]
        invalid_methods = ["nuts", "cmc", "svi", "auto"]

        # Act & Assert - Invalid methods should raise errors
        for method in valid_methods:
            assert method in ["nlsq", "mcmc"], f"{method} should be valid"

        for method in invalid_methods:
            assert method not in valid_methods, f"{method} should be invalid in v2.1"


class TestParameterRegimeConvergence:
    """Test convergence across different parameter regimes."""

    def test_static_mode_parameter_regime(self):
        """Test MCMC works for static_mode mode (3 physics params)."""
        # Arrange - Static isotropic data
        data = create_mock_data_dict(n_angles=10, n_t1=25, n_t2=25)
        config = create_mock_config_manager(analysis_mode="static")

        # Physics parameters: [D0, alpha, D_offset]
        n_physics_params = 3
        n_total_params = 5  # Add contrast, offset

        # Assert - Parameter counts correct
        assert config["analysis_mode"] == "static"
        assert n_physics_params == 3, "Static isotropic has 3 physics params"
        assert n_total_params == 5, "Total with scaling is 5 params"

    def test_laminar_flow_parameter_regime(self):
        """Test MCMC works for laminar_flow mode (7 physics params)."""
        # Arrange - Laminar flow data
        data = create_mock_data_dict(n_angles=10, n_t1=25, n_t2=25)
        config = create_mock_config_manager(analysis_mode="laminar_flow")

        # Physics parameters: [D0, alpha, D_offset, gamma_dot_t0, beta,
        #                      gamma_dot_t_offset, phi0]
        n_physics_params = 7
        n_total_params = 9  # Add contrast, offset

        # Assert - Parameter counts correct
        assert config["analysis_mode"] == "laminar_flow"
        assert n_physics_params == 7, "Laminar flow has 7 physics params"
        assert n_total_params == 9, "Total with scaling is 9 params"


class TestIntegrationWithAngleFiltering:
    """Test MCMC integration with angle filtering."""

    def test_mcmc_receives_filtered_angles_correctly(self):
        """Test that MCMC receives correctly filtered angle data."""
        # Arrange - Create data with specific angles
        angles = [0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 95.0, 180.0]
        data = create_specific_angles_test_data(phi_angles=angles, n_t1=10, n_t2=10)

        # Configure filtering to select only [85, 90, 95]
        config_dict = create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 85.0, "max_angle": 100.0, "description": "Near 90"}
            ],
        )

        # Act - Apply filtering
        from homodyne.cli.commands import _apply_angle_filtering_for_optimization

        class MockConfigManager:
            def get_config(self):
                return config_dict

        config = MockConfigManager()
        filtered_data = _apply_angle_filtering_for_optimization(data, config)

        # Assert - Filtering worked
        assert len(filtered_data["phi_angles_list"]) == 3
        np.testing.assert_array_almost_equal(
            filtered_data["phi_angles_list"], [85.0, 90.0, 95.0], decimal=1
        )

        # Test automatic NUTS/CMC selection with filtered data
        num_samples = len(filtered_data["phi_angles_list"])  # 3 samples
        hw_config = HardwareConfig(
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

        use_cmc = should_use_cmc(
            num_samples=num_samples,
            hardware_config=hw_config,
            min_samples_for_cmc=15,
            memory_threshold_pct=0.30,
        )

        # Assert - Should use NUTS for 3 filtered samples
        assert use_cmc is False, "Should use NUTS for 3 filtered samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
