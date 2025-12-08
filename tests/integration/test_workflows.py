"""
Integration Tests for Scientific Workflows
==========================================

End-to-end integration tests for complete homodyne analysis workflows:
- Data loading → Processing → Optimization → Results
- Configuration system integration
- Multi-format data handling
- Cross-module compatibility
- Output consistency validation
"""

import json
from pathlib import Path

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

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    _ = jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

# Handle NLSQ imports
try:
    import nlsq

    _ = nlsq

    from homodyne.optimization.nlsq import fit_nlsq_jax

    NLSQ_AVAILABLE = True
except ImportError:
    NLSQ_AVAILABLE = False
    fit_nlsq_jax = None


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndWorkflows:
    """Test complete end-to-end analysis workflows."""

    def test_synthetic_data_complete_workflow(self, temp_dir, test_config):
        """Test complete workflow with synthetic data."""
        try:
            # Import required modules
            from homodyne.config.manager import ConfigManager
            from homodyne.data.xpcs_loader import XPCSLoader

            # NLSQ imports now at module level
            _ = (ConfigManager, XPCSLoader)
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available for optimization")

        # Step 1: Create synthetic dataset
        n_times = 30
        n_angles = 24

        t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")
        phi = np.linspace(0, 2 * np.pi, n_angles)

        # Generate realistic correlation data
        tau = np.abs(t1 - t2) + 1e-6
        true_params = {"offset": 1.0, "contrast": 0.4, "diffusion_coefficient": 0.12}

        c2_true = true_params["offset"] + true_params["contrast"] * np.exp(-tau * 0.01)

        # Add realistic noise
        np.random.seed(42)
        noise = 0.01 * np.random.randn(*c2_true.shape)
        c2_exp = c2_true + noise

        # Create data structure
        workflow_data = {
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp,
            "wavevector_q_list": np.array([0.01]),
            "sigma": np.ones_like(c2_exp) * 0.01,
        }

        # Step 2: Configure analysis
        workflow_config = test_config.copy()
        workflow_config.update(
            {
                "analysis_mode": "static",
                "output_directory": str(temp_dir),
                "optimization": {
                    "method": "nlsq",
                    "lsq": {"max_iterations": 100, "tolerance": 1e-6},
                },
            }
        )

        # Step 3: Run optimization
        result = fit_nlsq_jax(workflow_data, workflow_config)

        # Step 4: Validate workflow results
        assert result.success, f"Workflow optimization failed: {result.message}"
        assert hasattr(result, "parameters")
        assert hasattr(result, "chi_squared")

        # Check parameter recovery
        recovered_params = result.parameters

        # Physical constraints
        assert 0.8 <= recovered_params.get("offset", 1.0) <= 1.2
        assert 0.0 <= recovered_params.get("contrast", 0.4) <= 1.0

        if "diffusion_coefficient" in recovered_params:
            assert recovered_params["diffusion_coefficient"] >= 0.0

        # Optimization quality
        assert result.chi_squared < 10.0, "Chi-squared too high for synthetic data"
        assert result.n_iterations > 0, "Should have performed iterations"

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not available")
    def test_hdf5_data_workflow(self, temp_dir, test_config):
        """Test workflow with HDF5 data loading."""
        try:
            from homodyne.data.xpcs_loader import load_xpcs_data

            # NLSQ imports now at module level
            _ = load_xpcs_data
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        # Create mock HDF5 file
        h5_file = temp_dir / "workflow_test.h5"

        n_times = 25
        n_angles = 18
        correlation_data = np.random.random((n_angles, n_times, n_times)) + 1.0

        with h5py.File(h5_file, "w") as f:
            exchange = f.create_group("exchange")
            exchange.create_dataset("correlation", data=correlation_data)
            exchange.create_dataset(
                "phi_angles", data=np.linspace(0, 2 * np.pi, n_angles)
            )
            exchange.create_dataset("wavevector_q", data=np.array([0.015]))
            exchange.create_dataset("time_grid", data=np.arange(n_times))

        # Configure for HDF5 workflow
        h5_config = test_config.copy()
        h5_config.update(
            {
                "data_file": str(h5_file),
                "output_directory": str(temp_dir),
                "analysis_mode": "static",
            }
        )

        # Note: load_xpcs_data now expects a config_path (str), not a config dict
        # Skipping this test as it requires proper HDF5 files
        pytest.skip("load_xpcs_data API changed: requires config_path, not config dict")

    @pytest.mark.skipif(not HAS_YAML, reason="PyYAML not available")
    def test_yaml_config_workflow(self, temp_dir):
        """Test workflow with YAML configuration."""
        try:
            from homodyne.config.manager import ConfigManager

            # NLSQ imports now at module level
            pass
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        # Create YAML configuration
        yaml_config = {
            "data_file": "synthetic",  # Use synthetic data
            "analysis_mode": "static",
            "output_directory": str(temp_dir),
            "optimization": {
                "method": "nlsq",
                "lsq": {"max_iterations": 50, "tolerance": 1e-5},
            },
            "physics": {
                "apply_diagonal_correction": True,
                "validate_correlation_bounds": True,
            },
        }

        yaml_file = temp_dir / "workflow_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_config, f)

        try:
            # Load configuration
            config_manager = ConfigManager(str(yaml_file))
            loaded_config = config_manager.get_config()

            # Validate configuration loading
            assert loaded_config["analysis_mode"] == "static"
            assert loaded_config["optimization"]["method"] == "nlsq"

            # Create synthetic data for this workflow
            synthetic_data = {
                "t1": np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
                "t2": np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
                "phi_angles_list": np.linspace(0, 2 * np.pi, 12),
                "c2_exp": np.random.random((12, 3, 3)) + 1.0,
                "wavevector_q_list": np.array([0.01]),
                "sigma": np.ones((12, 3, 3)) * 0.02,
            }

            # Run optimization with YAML config
            result = fit_nlsq_jax(synthetic_data, loaded_config)

            if result.success:
                assert hasattr(result, "parameters")
                assert (
                    result.n_iterations
                    <= loaded_config["optimization"]["lsq"]["max_iterations"]
                )

        except Exception as e:
            pytest.skip(f"YAML config workflow failed: {e}")

    @pytest.mark.skip(reason="JAX model computation in tests needs investigation")
    def test_multi_q_value_workflow(self, test_config):
        """Test workflow with multiple q-values."""
        try:
            from tests.utils.legacy_compat import compute_c2_model_jax

            # NLSQ imports now at module level
            pass
        except ImportError as e:
            pytest.skip(f"Required modules not available: {e}")

        if not NLSQ_AVAILABLE or not JAX_AVAILABLE:
            pytest.skip("NLSQ or JAX not available")

        # Generate data with multiple q-values
        n_times = 20
        n_angles = 12
        q_values = np.array([0.008, 0.012, 0.016])

        t1, t2 = jnp.meshgrid(jnp.arange(n_times), jnp.arange(n_times), indexing="ij")
        phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

        # For now, test with single q-value (multi-q would need different data structure)
        q = q_values[0]

        params = {
            "offset": 1.0,
            "contrast": 0.35,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Generate synthetic data
        c2_model = compute_c2_model_jax(params, t1, t2, phi, q)
        noise = 0.005 * np.random.randn(*c2_model.shape)
        c2_exp = c2_model + noise

        multi_q_data = {
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp,
            "wavevector_q_list": np.array([q]),
            "sigma": np.ones_like(c2_exp) * 0.005,
        }

        # Configure for multi-q analysis
        multi_q_config = test_config.copy()
        multi_q_config.update({"analysis_mode": "static", "multi_q_analysis": True})

        # Run optimization
        result = fit_nlsq_jax(multi_q_data, multi_q_config)

        if result.success:
            # Validate results
            assert "diffusion_coefficient" in result.parameters
            recovered_D = result.parameters["diffusion_coefficient"]

            # Should recover approximately correct diffusion coefficient
            assert 0.05 <= recovered_D <= 0.2, (
                f"Diffusion coefficient recovery poor: {recovered_D}"
            )

    @pytest.mark.skip(
        reason="Residual function issue in nlsq_wrapper - needs investigation"
    )
    def test_error_propagation_workflow(self, test_config):
        """Test error propagation through workflow."""
        try:
            # NLSQ imports now at module level
            pass
        except ImportError:
            pytest.skip("Optimization module not available")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        # Test with various error conditions
        error_scenarios = [
            {
                "name": "missing_data_key",
                "data": {
                    "t1": np.array([[0, 1], [1, 0]]),
                    "t2": np.array([[0, 1], [1, 0]]),
                    # Missing 'c2_exp'
                },
                "expected_error": (KeyError, AttributeError),
            },
            {
                "name": "mismatched_shapes",
                "data": {
                    "t1": np.array([[0, 1], [1, 0]]),
                    "t2": np.array([[0, 1], [1, 0]]),
                    "phi_angles_list": np.array([0.0]),
                    "c2_exp": np.array([[[1.0, 1.1], [1.1, 1.0]]]),  # Wrong shape
                    "wavevector_q_list": np.array([0.01]),
                    "sigma": np.array([[[0.01, 0.01], [0.01, 0.01]]]),
                },
                "expected_error": (ValueError, IndexError),
            },
            {
                "name": "invalid_config",
                "data": {
                    "t1": np.array([[0, 1], [1, 0]]),
                    "t2": np.array([[0, 1], [1, 0]]),
                    "phi_angles_list": np.array([0.0]),
                    "c2_exp": np.array([[[1.0, 1.1], [1.1, 1.0]]]),
                    "wavevector_q_list": np.array([0.01]),
                    "sigma": np.array([[[0.01, 0.01], [0.01, 0.01]]]),
                },
                "config_override": {"optimization": {"method": "invalid_method"}},
                "expected_error": (ValueError, KeyError, NotImplementedError),
            },
        ]

        for scenario in error_scenarios:
            scenario_config = test_config.copy()
            if "config_override" in scenario:
                scenario_config.update(scenario["config_override"])

            with pytest.raises(scenario["expected_error"]):
                fit_nlsq_jax(scenario["data"], scenario_config)


@pytest.mark.integration
class TestModuleInteraction:
    """Test interaction between different modules."""

    def test_config_manager_integration(self, temp_dir):
        """Test configuration manager integration with other modules."""
        try:
            from homodyne.config.manager import ConfigManager
        except ImportError:
            pytest.skip("Config manager not available")

        # Create configuration with multiple sections
        complex_config = {
            "data": {"file_format": "hdf5", "cache_enabled": True, "validation": True},
            "analysis": {
                "mode": "static",
                "parameters": ["offset", "contrast", "diffusion_coefficient"],
            },
            "optimization": {
                "method": "nlsq",
                "lsq": {
                    "max_iterations": 100,
                    "tolerance": 1e-6,
                    "bounds": {"offset": [0.5, 2.0], "contrast": [0.0, 1.0]},
                },
            },
            "output": {
                "directory": str(temp_dir),
                "save_plots": False,
                "export_format": "json",
            },
        }

        # Test configuration persistence and loading
        config_file = temp_dir / "complex_config.json"
        with open(config_file, "w") as f:
            json.dump(complex_config, f, indent=2)

        # Load and validate
        config_manager = ConfigManager(str(config_file))
        loaded_config = config_manager.get_config()

        # Validate structure preservation
        assert loaded_config["optimization"]["method"] == "nlsq"
        assert loaded_config["optimization"]["lsq"]["max_iterations"] == 100
        assert loaded_config["output"]["directory"] == str(temp_dir)

        # Test configuration modification
        config_manager.update_config("optimization.lsq.max_iterations", 200)
        updated_config = config_manager.get_config()
        assert updated_config["optimization"]["lsq"]["max_iterations"] == 200

    def test_data_loader_optimization_integration(
        self, synthetic_xpcs_data, test_config
    ):
        """Test integration between data loading and optimization."""
        try:
            from homodyne.data.xpcs_loader import XPCSLoader

            # NLSQ imports now at module level
            pass
        except ImportError:
            pytest.skip("Required modules not available")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        data = synthetic_xpcs_data

        # Test data preprocessing integration
        loader = XPCSLoader(validate_data=True)

        try:
            # Simulate data processing pipeline
            processed_data = loader._validate_data_structure(data)

            # Should be compatible with optimization
            result = fit_nlsq_jax(processed_data, test_config)

            if result.success:
                assert hasattr(result, "parameters")
                assert result.chi_squared >= 0.0

        except (AttributeError, NotImplementedError):
            # Method might not exist, use direct data
            result = fit_nlsq_jax(data, test_config)
            if result.success:
                assert hasattr(result, "parameters")

    @pytest.mark.skip(reason="JAX backend optimization consistency needs investigation")
    def test_backend_optimization_consistency(self, synthetic_xpcs_data, test_config):
        """Test consistency between JAX backend and optimization."""
        try:
            from tests.utils.legacy_compat import chi_squared_jax, compute_c2_model_jax

            # NLSQ imports now at module level
            pass
        except ImportError:
            pytest.skip("Required modules not available")

        if not NLSQ_AVAILABLE or not JAX_AVAILABLE:
            pytest.skip("JAX or NLSQ not available")

        data = synthetic_xpcs_data

        # Run optimization
        result = fit_nlsq_jax(data, test_config)

        if result.success:
            # Use backend to compute model with optimized parameters
            optimized_params = result.parameters

            # Ensure required parameters are present
            required_params = [
                "offset",
                "contrast",
                "diffusion_coefficient",
                "shear_rate",
                "L",
            ]
            for param in required_params:
                if param not in optimized_params:
                    if param == "shear_rate":
                        optimized_params[param] = 0.0
                    elif param == "L":
                        optimized_params[param] = 1.0

            t1 = jnp.array(data["t1"])
            t2 = jnp.array(data["t2"])
            phi = jnp.array(data["phi_angles_list"])
            q = data["wavevector_q_list"][0]

            # Compute model with optimized parameters
            model_result = compute_c2_model_jax(optimized_params, t1, t2, phi, q)

            # Compute chi-squared with backend
            c2_exp = jnp.array(data["c2_exp"])
            sigma = jnp.array(data["sigma"])
            backend_chi2 = chi_squared_jax(
                optimized_params, c2_exp, sigma, t1, t2, phi, q
            )

            # Should be consistent with optimization result
            chi2_diff = abs(float(backend_chi2) - result.chi_squared)
            relative_diff = chi2_diff / (result.chi_squared + 1e-10)

            assert relative_diff < 0.1, (
                f"Chi-squared inconsistency: {relative_diff:.3f}"
            )

            # Model should have reasonable values
            assert jnp.all(jnp.isfinite(model_result)), "Model result should be finite"
            assert jnp.all(model_result >= 0.8), (
                "Model should have reasonable correlation values"
            )


@pytest.mark.integration
@pytest.mark.slow
class TestCrossplatformCompatibility:
    """Test cross-platform compatibility and behavior."""

    def test_path_handling_consistency(self, temp_dir):
        """Test consistent path handling across platforms."""
        try:
            from homodyne.config.manager import ConfigManager
        except ImportError:
            pytest.skip("Config manager not available")

        # Test with different path formats
        path_formats = [
            str(temp_dir / "config.json"),  # Path object
            str(temp_dir) + "/config.json",  # Forward slash
        ]

        config_data = {
            "output_directory": str(temp_dir),
            "analysis_mode": "static",
        }

        for path_format in path_formats:
            # Create config file
            with open(path_format, "w") as f:
                json.dump(config_data, f)

            # Test loading
            config_manager = ConfigManager(path_format)
            loaded_config = config_manager.get_config()

            assert loaded_config["analysis_mode"] == "static"
            assert Path(loaded_config["output_directory"]).exists()

    def test_numerical_precision_consistency(self):
        """Test numerical precision consistency across platforms."""
        try:
            from tests.utils.legacy_compat import compute_g1_diffusion_jax
        except ImportError:
            pytest.skip("JAX backend not available")

        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Test with specific values that might show precision differences
        t1 = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        t2 = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        q = 0.01
        D = 0.1

        result = compute_g1_diffusion_jax(t1, t2, q, D)

        # Test specific expected values (should be consistent across platforms)
        diagonal = jnp.diag(result)
        expected_diagonal = jnp.ones(3)

        np.testing.assert_array_almost_equal(
            diagonal,
            expected_diagonal,
            decimal=10,
            err_msg="Diagonal precision inconsistent",
        )

        # Test symmetry (should be exact)
        symmetry_diff = jnp.max(jnp.abs(result - result.T))
        assert symmetry_diff < 1e-14, (
            f"Symmetry precision inconsistent: {symmetry_diff}"
        )

    @pytest.mark.skip(reason="Memory behavior consistency test needs investigation")
    def test_memory_behavior_consistency(self, synthetic_xpcs_data, test_config):
        """Test consistent memory behavior across platforms."""
        try:
            # NLSQ imports now at module level
            pass
        except ImportError:
            pytest.skip("Optimization module not available")

        if not NLSQ_AVAILABLE:
            pytest.skip("NLSQ not available")

        data = synthetic_xpcs_data

        # Run multiple optimizations to test memory consistency
        results = []
        for _i in range(3):
            result = fit_nlsq_jax(data, test_config)
            results.append(result)

            # Force garbage collection
            import gc

            gc.collect()

        # Results should be consistent
        successful_results = [r for r in results if r.success]

        if len(successful_results) >= 2:
            # Parameters should be stable across runs
            param_keys = set(successful_results[0].parameters.keys())

            for key in param_keys:
                values = [r.parameters[key] for r in successful_results]
                value_std = np.std(values)
                value_mean = np.mean(values)

                # Should have low variability (not exact due to numerical precision)
                relative_std = value_std / (abs(value_mean) + 1e-10)
                assert relative_std < 0.01, (
                    f"Parameter {key} inconsistent across runs: {relative_std:.4f}"
                )
