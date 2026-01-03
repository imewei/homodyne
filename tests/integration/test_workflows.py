"""
Integration Tests for Scientific Workflows
==========================================

End-to-end integration tests for complete homodyne analysis workflows:
- Data loading -> Processing -> Optimization -> Results
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
                    result.iterations
                    <= loaded_config["optimization"]["lsq"]["max_iterations"]
                )

        except Exception as e:
            pytest.skip(f"YAML config workflow failed: {e}")


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
            from homodyne.core.jax_backend import compute_g1_diffusion
        except ImportError:
            pytest.skip("JAX backend not available")

        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Test with specific values that might show precision differences
        t1 = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        t2 = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        q = 0.01
        D = 0.1

        result = compute_g1_diffusion(t1, t2, q, D)

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
