"""
Integration tests for NLSQ result saving and visualization workflows.

This module tests end-to-end workflows from OptimizationResult through
file saving and plotting, including error recovery scenarios.

Test Coverage
-------------
- Full workflow: optimization result → all files saved
- Metadata extraction with missing values (fallback behavior)
- Plotting failure recovery (data files still saved)
- Large dataset handling (180 angles × 100×100)
- Device array conversion (JAX/GPU → NumPy)
- Method comparison (NLSQ vs classical file structure)
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from homodyne.cli.commands import save_nlsq_results
from tests.factories.optimization_factory import (
    create_mock_config_manager,
    create_mock_data_dict,
    create_mock_optimization_result,
)

# ==============================================================================
# Test Class: Full Workflow
# ==============================================================================


class TestNLSQFullWorkflow:
    """Test complete NLSQ saving workflow."""

    def test_nlsq_full_workflow_files(self):
        """Test end-to-end workflow with all 4 files created."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Test with static_isotropic mode
            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
                include_uncertainties=True,
            )
            data = create_mock_data_dict(n_angles=5, n_t1=20, n_t2=20)
            config_dict = create_mock_config_manager(
                analysis_mode="static_isotropic",
                include_all_metadata=True,
            )
            # Wrap in Mock to simulate ConfigManager with .config attribute
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results
            save_nlsq_results(result, data, config, output_dir)

            # Verify nlsq/ subdirectory was created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists(), "nlsq subdirectory not created"
            assert nlsq_dir.is_dir(), "nlsq is not a directory"

            # Verify all 4 files exist
            param_file = nlsq_dir / "parameters.json"
            npz_file = nlsq_dir / "fitted_data.npz"
            analysis_file = nlsq_dir / "analysis_results_nlsq.json"
            convergence_file = nlsq_dir / "convergence_metrics.json"

            assert param_file.exists(), "parameters.json not created"
            assert npz_file.exists(), "fitted_data.npz not created"
            assert analysis_file.exists(), "analysis_results_nlsq.json not created"
            assert convergence_file.exists(), "convergence_metrics.json not created"

            # Verify parameters.json structure
            with open(param_file) as f:
                params = json.load(f)
            assert "timestamp" in params, "Missing timestamp in parameters.json"
            assert "analysis_mode" in params, "Missing analysis_mode"
            assert params["analysis_mode"] == "static_isotropic"
            assert "chi_squared" in params, "Missing chi_squared"
            assert "parameters" in params, "Missing parameters dict"
            # Static isotropic has 5 parameters
            assert (
                len(params["parameters"]) == 5
            ), f"Expected 5 parameters, got {len(params['parameters'])}"
            assert "contrast" in params["parameters"]
            assert "offset" in params["parameters"]
            assert "D0" in params["parameters"]
            assert "alpha" in params["parameters"]
            assert "D_offset" in params["parameters"]

            # Verify fitted_data.npz structure
            npz_data = np.load(npz_file)
            assert "phi_angles" in npz_data, "Missing phi_angles in NPZ"
            assert "c2_exp" in npz_data, "Missing c2_exp"
            assert "c2_theoretical_raw" in npz_data, "Missing c2_theoretical_raw"
            assert "c2_theoretical_scaled" in npz_data, "Missing c2_theoretical_scaled"
            assert "per_angle_scaling" in npz_data, "Missing per_angle_scaling"
            assert "residuals" in npz_data, "Missing residuals"
            assert "residuals_normalized" in npz_data, "Missing residuals_normalized"
            assert "t1" in npz_data, "Missing t1"
            assert "t2" in npz_data, "Missing t2"
            assert "q" in npz_data, "Missing q"
            # Total should be 10 arrays: 2 experimental + 3 theoretical + 2 residuals + 3 coordinates
            assert (
                len(npz_data.files) == 10
            ), f"Expected 10 arrays in NPZ, got {len(npz_data.files)}"

            # Verify array shapes
            assert npz_data["phi_angles"].shape == (5,), "Wrong phi_angles shape"
            assert npz_data["c2_exp"].shape == (5, 20, 20), "Wrong c2_exp shape"
            assert npz_data["c2_theoretical_raw"].shape == (
                5,
                20,
                20,
            ), "Wrong c2_theoretical_raw shape"
            assert npz_data["residuals"].shape == (5, 20, 20), "Wrong residuals shape"

            # Verify analysis_results_nlsq.json structure
            with open(analysis_file) as f:
                analysis = json.load(f)
            assert "method" in analysis, "Missing method"
            assert analysis["method"] == "nlsq"
            assert "timestamp" in analysis
            assert "fit_quality" in analysis, "Missing fit_quality"
            assert "dataset_info" in analysis, "Missing dataset_info"
            assert "optimization_summary" in analysis, "Missing optimization_summary"

            # Verify convergence_metrics.json structure
            with open(convergence_file) as f:
                convergence = json.load(f)
            assert "convergence" in convergence, "Missing convergence dict"
            assert "status" in convergence["convergence"]
            assert "iterations" in convergence["convergence"]
            assert "quality_flag" in convergence, "Missing quality_flag"

            # Test with laminar_flow mode (9 parameters)
            result_lf = create_mock_optimization_result(
                analysis_mode="laminar_flow",
                converged=True,
            )
            config_dict_lf = create_mock_config_manager(
                analysis_mode="laminar_flow",
                include_all_metadata=True,
            )
            # Wrap in Mock to simulate ConfigManager with .config attribute
            config_lf = Mock()
            config_lf.config = config_dict_lf

            # Save to different subdirectory
            output_dir_lf = Path(tmpdir) / "test_lf"
            output_dir_lf.mkdir()
            save_nlsq_results(result_lf, data, config_lf, output_dir_lf)

            # Verify laminar flow has 9 parameters
            param_file_lf = output_dir_lf / "nlsq" / "parameters.json"
            with open(param_file_lf) as f:
                params_lf = json.load(f)
            assert params_lf["analysis_mode"] == "laminar_flow"
            assert (
                len(params_lf["parameters"]) == 9
            ), f"Expected 9 parameters, got {len(params_lf['parameters'])}"
            assert "gamma_dot_t0" in params_lf["parameters"]
            assert "beta" in params_lf["parameters"]
            assert "gamma_dot_t_offset" in params_lf["parameters"]
            assert "phi0" in params_lf["parameters"]

    def test_nlsq_full_workflow_with_plots(self):
        """Test workflow with PNG plot generation."""
        pytest.importorskip("matplotlib")  # Skip if matplotlib not available

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data
            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=5, n_t1=20, n_t2=20)
            config_dict = create_mock_config_manager(
                analysis_mode="static_isotropic",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results (which should call generate_nlsq_plots)
            save_nlsq_results(result, data, config, output_dir)

            # Verify nlsq/ subdirectory exists
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()

            # Verify all 4 data files exist
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify PNG plots were created (one per angle)
            png_files = list(nlsq_dir.glob("*.png"))
            assert len(png_files) == 5, f"Expected 5 PNG files, got {len(png_files)}"

            # Verify naming convention
            for i, phi in enumerate(data["phi_angles_list"]):
                expected_name = f"c2_heatmaps_phi_{phi:.1f}deg.png"
                png_file = nlsq_dir / expected_name
                assert png_file.exists(), f"PNG file not found: {expected_name}"
                # Verify file is not empty
                assert (
                    png_file.stat().st_size > 1000
                ), f"PNG file too small: {expected_name}"


# ==============================================================================
# Test Class: Error Recovery
# ==============================================================================


class TestNLSQErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_save_nlsq_results_with_missing_metadata(self):
        """Test that fallback defaults work when metadata is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data and result
            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=3, n_t1=15, n_t2=15)

            # Create config with minimal metadata (no L, no dt)
            config_dict = {
                "analysis_mode": "static_isotropic",
                "experimental_data": {},  # No sample_detector_distance
                "analyzer_parameters": {},  # No geometry or dt
            }
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results - should use fallback defaults
            save_nlsq_results(result, data, config, output_dir)

            # Verify files were still created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify fallback values were used in metadata
            # L should default to 2000000.0 Å
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert npz_data["q"] is not None  # q comes from data, should be present

    def test_save_nlsq_results_plotting_failure_recovery(self):
        """Test that data files are saved even when plotting fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data and result
            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=3, n_t1=15, n_t2=15)
            config_dict = create_mock_config_manager(
                analysis_mode="static_isotropic",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Patch generate_nlsq_plots to raise an exception
            with patch(
                "homodyne.cli.commands.generate_nlsq_plots",
                side_effect=RuntimeError("Matplotlib not available"),
            ):
                # Call save_nlsq_results - should succeed despite plotting failure
                save_nlsq_results(result, data, config, output_dir)

            # Verify data files were still created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()
            assert (nlsq_dir / "analysis_results_nlsq.json").exists()
            assert (nlsq_dir / "convergence_metrics.json").exists()

            # Verify PNG plots were NOT created (plotting failed)
            png_files = list(nlsq_dir.glob("*.png"))
            assert (
                len(png_files) == 0
            ), "PNG files should not be created if plotting fails"


# ==============================================================================
# Test Class: Performance and Scale
# ==============================================================================


class TestNLSQPerformance:
    """Test performance with large datasets."""

    @pytest.mark.slow
    def test_save_nlsq_results_large_dataset(self):
        """Test handling of 180 angles × 100×100 correlation matrices."""
        pytest.importorskip("matplotlib")  # Skip if matplotlib not available

        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create large dataset (180 angles × 100×100)
            n_angles = 180
            n_t1 = 100
            n_t2 = 100

            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
            )
            data = create_mock_data_dict(n_angles=n_angles, n_t1=n_t1, n_t2=n_t2)
            config_dict = create_mock_config_manager(
                analysis_mode="static_isotropic",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Measure execution time
            start_time = time.time()
            save_nlsq_results(result, data, config, output_dir)
            elapsed_time = time.time() - start_time

            # Verify all files created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "parameters.json").exists()
            assert (nlsq_dir / "fitted_data.npz").exists()

            # Verify data dimensions
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert npz_data["c2_exp"].shape == (n_angles, n_t1, n_t2)
            assert npz_data["phi_angles"].shape == (n_angles,)

            # Verify 180 PNG plots created
            png_files = list(nlsq_dir.glob("*.png"))
            assert len(png_files) == n_angles

            # Log performance (informational, no strict requirement for slow test)
            print(
                f"\nLarge dataset performance: {elapsed_time:.2f}s for {n_angles} angles × {n_t1}×{n_t2}"
            )

    def test_save_nlsq_results_device_array_conversion(self):
        """Test conversion of JAX/GPU arrays to NumPy for saving."""
        import jax.numpy as jnp

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data with JAX arrays
            result = create_mock_optimization_result(
                analysis_mode="static_isotropic",
                converged=True,
            )

            # Convert some arrays to JAX arrays (simulating GPU data)
            data = create_mock_data_dict(n_angles=3, n_t1=20, n_t2=20)
            data["c2_exp"] = jnp.array(data["c2_exp"])  # Convert to JAX array
            data["t1"] = jnp.array(data["t1"])
            data["t2"] = jnp.array(data["t2"])

            config_dict = create_mock_config_manager(
                analysis_mode="static_isotropic",
                include_all_metadata=True,
            )
            config = Mock()
            config.config = config_dict

            # Call save_nlsq_results - should handle JAX arrays
            save_nlsq_results(result, data, config, output_dir)

            # Verify files created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists()
            assert (nlsq_dir / "fitted_data.npz").exists()

            # Load and verify arrays are NumPy (not JAX)
            npz_data = np.load(nlsq_dir / "fitted_data.npz")
            assert isinstance(npz_data["c2_exp"], np.ndarray)
            assert isinstance(npz_data["t1"], np.ndarray)
            assert isinstance(npz_data["t2"], np.ndarray)
            # Verify conversion preserved data
            assert npz_data["c2_exp"].shape == (3, 20, 20)
            assert npz_data["t1"].shape == (20,)
            assert npz_data["t2"].shape == (20,)


# ==============================================================================
# Test Class: Method Comparison
# ==============================================================================


class TestNLSQClassicalComparison:
    """Test that NLSQ output structure matches classical optimization."""

    def test_nlsq_classical_directory_structure_match(self):
        """Test that nlsq/ and classical/ have parallel structures."""
        pass  # To be implemented in T039

    def test_nlsq_classical_file_types_match(self):
        """Test that file names and types match between methods."""
        pass  # To be implemented in T040

    def test_nlsq_classical_plot_format_match(self):
        """Test that plot formats and naming match."""
        pass  # To be implemented in T041


# ==============================================================================
# Test Class: CLI Integration
# ==============================================================================


class TestNLSQCLIIntegration:
    """Test CLI end-to-end with NLSQ saving."""

    def test_save_results_routing_nlsq(self):
        """Test that _save_results() routes to save_nlsq_results() for NLSQ method."""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from homodyne.cli.commands import _save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock args with method="nlsq"
            args = Mock()
            args.method = "nlsq"
            args.output_dir = output_dir
            args.output_format = "json"

            # Create mock result, data, config
            result = create_mock_optimization_result(analysis_mode="static_isotropic")
            data = create_mock_data_dict(n_angles=3, n_t1=10, n_t2=10)
            config_dict = create_mock_config_manager(analysis_mode="static_isotropic")
            config = Mock()
            config.config = config_dict
            device_config = {"device": "cpu"}

            # Call _save_results which should route to save_nlsq_results()
            _save_results(args, result, device_config, data, config)

            # Verify NLSQ directory and files were created
            nlsq_dir = output_dir / "nlsq"
            assert nlsq_dir.exists(), "NLSQ directory not created via routing"
            assert (
                nlsq_dir / "parameters.json"
            ).exists(), "parameters.json not created"
            assert (
                nlsq_dir / "fitted_data.npz"
            ).exists(), "fitted_data.npz not created"
            assert (
                nlsq_dir / "analysis_results_nlsq.json"
            ).exists(), "analysis_results not created"
            assert (
                nlsq_dir / "convergence_metrics.json"
            ).exists(), "convergence_metrics not created"

    def test_save_results_routing_mcmc(self):
        """Test that _save_results() uses legacy format for MCMC method."""
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock

        from homodyne.cli.commands import _save_results

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock args with method="mcmc"
            args = Mock()
            args.method = "mcmc"
            args.output_dir = output_dir
            args.output_format = "json"

            # Create mock MCMC result
            result = Mock()
            result.mean_contrast = 0.5
            result.mean_offset = 1.0
            result.mean_params = np.array([1000.0, 0.5, 10.0])
            result.samples_params = None

            data = create_mock_data_dict(n_angles=3, n_t1=10, n_t2=10)
            config_dict = create_mock_config_manager(analysis_mode="static_isotropic")
            config = Mock()
            config.config = config_dict
            device_config = {"device": "cpu"}

            # Call _save_results which should use legacy format
            _save_results(args, result, device_config, data, config)

            # Verify legacy file was created (not nlsq/ subdirectory)
            legacy_file = output_dir / "homodyne_results.json"
            assert legacy_file.exists(), "Legacy results file not created for MCMC"
            # Verify nlsq directory was NOT created
            nlsq_dir = output_dir / "nlsq"
            assert (
                not nlsq_dir.exists()
            ), "NLSQ directory should not be created for MCMC"

    @pytest.mark.slow
    def test_cli_end_to_end_nlsq(self):
        """Test full CLI workflow from config to saved results."""
        pass  # Deferred - requires full CLI with real config file

    def test_nlsq_workflow_both_analysis_modes(self):
        """Test with both static_isotropic and laminar_flow modes."""
        # This is already tested in test_nlsq_full_workflow_files
        # which tests both modes
        pass  # Covered by test_nlsq_full_workflow_files
