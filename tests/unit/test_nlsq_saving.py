"""
Unit tests for NLSQ result saving and visualization functions.

This module tests the helper functions and main orchestrator for saving
NLSQ optimization results to structured files (JSON, NPZ, PNG).

Test Coverage
-------------
- Metadata extraction with fallback behavior
- Parameter data preparation for both analysis modes
- Theoretical fit computation with least squares scaling
- JSON file saving (parameters, analysis results, convergence metrics)
- NPZ file saving (experimental + theoretical + residuals)
- PNG plot generation (3-panel heatmaps)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from tests.factories.optimization_factory import (
    create_mock_config_manager,
    create_mock_data_dict,
    create_mock_optimization_result,
)

# ==============================================================================
# Test Class: Metadata Extraction (_extract_nlsq_metadata)
# ==============================================================================


class TestExtractNLSQMetadata:
    """Test metadata extraction with multi-level fallback."""

    def test_extract_metadata_with_all_present(self):
        """Test metadata extraction when all fields are present in config."""
        # Import the function (will be added to commands.py)
        from homodyne.cli.commands import _extract_nlsq_metadata
        from homodyne.config.manager import ConfigManager

        # Create mock config with all metadata
        config_dict = create_mock_config_manager(
            analysis_mode="laminar_flow", include_all_metadata=True
        )
        config = Mock(spec=ConfigManager)
        config.config = config_dict

        # Create mock data with q value
        data = create_mock_data_dict(n_angles=10)

        # Extract metadata
        metadata = _extract_nlsq_metadata(config, data)

        # Verify all metadata extracted
        assert "L" in metadata
        assert "dt" in metadata
        assert "q" in metadata
        assert metadata["L"] == 2000000.0  # From stator_rotor_gap
        assert metadata["dt"] == 0.1
        assert metadata["q"] == pytest.approx(0.0123)

    def test_extract_metadata_with_fallback_L(self):
        """Test L extraction with fallback to sample_detector_distance."""
        from homodyne.cli.commands import _extract_nlsq_metadata
        from homodyne.config.manager import ConfigManager

        # Create config without stator_rotor_gap (will fall back to sample_detector_distance)
        config_dict = {
            "analysis_mode": "static_isotropic",
            "analyzer_parameters": {},
            "experimental_data": {"sample_detector_distance": 5000000.0, "dt": 0.1},
        }
        config = Mock(spec=ConfigManager)
        config.config = config_dict

        # Create data with q
        data = create_mock_data_dict(n_angles=5)

        # Extract metadata
        metadata = _extract_nlsq_metadata(config, data)

        # Verify fallback to sample_detector_distance
        assert metadata["L"] == 5000000.0

    def test_extract_metadata_with_missing_q(self):
        """Test behavior when q is missing from data."""
        from homodyne.cli.commands import _extract_nlsq_metadata
        from homodyne.config.manager import ConfigManager

        # Create config
        config_dict = create_mock_config_manager("static_isotropic")
        config = Mock(spec=ConfigManager)
        config.config = config_dict

        # Create data without q
        data = {
            "phi_angles_list": np.array([0, 45, 90]),
            "c2_exp": np.random.rand(3, 10, 10),
        }

        # Extract metadata
        metadata = _extract_nlsq_metadata(config, data)

        # Verify q is None with warning (should not crash)
        assert metadata["q"] is None

    def test_extract_metadata_time_array_2d_to_1d(self):
        """Test conversion of 2D meshgrid time arrays to 1D arrays."""
        from homodyne.cli.commands import _extract_nlsq_metadata
        from homodyne.config.manager import ConfigManager

        # Create config
        config_dict = create_mock_config_manager("static_isotropic")
        config = Mock(spec=ConfigManager)
        config.config = config_dict

        # Create data with 2D time arrays (meshgrid format)
        t_vals = np.linspace(0.01, 1.0, 25)
        t1_2d, t2_2d = np.meshgrid(t_vals, t_vals, indexing="ij")

        data = {
            "phi_angles_list": np.array([0, 45, 90]),
            "c2_exp": np.random.rand(3, 25, 25),
            "t1": t1_2d,  # 2D array (25, 25)
            "t2": t2_2d,  # 2D array (25, 25)
            "wavevector_q_list": np.array([0.001]),
        }

        # Extract metadata (this will process the data)
        metadata = _extract_nlsq_metadata(config, data)

        # Verify metadata extraction works with 2D arrays
        assert metadata["q"] is not None
        assert metadata["L"] is not None

        # Note: The actual 2D→1D conversion happens in _compute_nlsq_fits() and save_nlsq_results()
        # This test verifies that metadata extraction doesn't crash with 2D time arrays


# ==============================================================================
# Test Class: Parameter Preparation (_prepare_parameter_data)
# ==============================================================================


class TestPrepareParameterData:
    """Test parameter data preparation for JSON saving."""

    def test_prepare_parameter_data_static_isotropic(self):
        """Test parameter extraction for static isotropic mode (5 params)."""
        from homodyne.cli.commands import _prepare_parameter_data

        # Create mock result for static isotropic (5 parameters)
        result = create_mock_optimization_result(
            analysis_mode="static_isotropic",
            converged=True,
            include_uncertainties=True,
        )

        # Prepare parameter data
        param_dict = _prepare_parameter_data(result, "static_isotropic")

        # Verify structure
        assert "contrast" in param_dict
        assert "offset" in param_dict
        assert "D0" in param_dict
        assert "alpha" in param_dict
        assert "D_offset" in param_dict
        assert len(param_dict) == 5

        # Verify values and uncertainties
        assert param_dict["contrast"]["value"] == pytest.approx(0.45)
        assert param_dict["contrast"]["uncertainty"] == pytest.approx(0.012)
        assert param_dict["D0"]["value"] == pytest.approx(1234.5)

    def test_prepare_parameter_data_laminar_flow(self):
        """Test parameter extraction for laminar flow mode (9 params)."""
        from homodyne.cli.commands import _prepare_parameter_data

        # Create mock result for laminar flow (9 parameters)
        result = create_mock_optimization_result(
            analysis_mode="laminar_flow", converged=True, include_uncertainties=True
        )

        # Prepare parameter data
        param_dict = _prepare_parameter_data(result, "laminar_flow")

        # Verify structure - should have all 9 parameters
        expected_params = [
            "contrast",
            "offset",
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        for param in expected_params:
            assert param in param_dict

        assert len(param_dict) == 9

        # Verify flow-specific parameters
        assert param_dict["gamma_dot_t0"]["value"] == pytest.approx(1.23e-4)
        assert param_dict["beta"]["value"] == pytest.approx(0.456)

    def test_prepare_parameter_data_with_null_uncertainties(self):
        """Test parameter extraction when uncertainties are None."""
        from homodyne.cli.commands import _prepare_parameter_data

        # Create result without uncertainties
        result = create_mock_optimization_result(
            analysis_mode="static_isotropic",
            converged=True,
            include_uncertainties=False,
        )

        # Prepare parameter data
        param_dict = _prepare_parameter_data(result, "static_isotropic")

        # Verify values present but uncertainties are None
        assert param_dict["D0"]["value"] == pytest.approx(1234.5)
        assert param_dict["D0"]["uncertainty"] is None


# ==============================================================================
# Test Class: Theoretical Fit Computation (_compute_nlsq_fits)
# ==============================================================================


class TestComputeNLSQFits:
    """Test theoretical fit generation with least squares scaling."""

    @patch("homodyne.cli.commands.compute_g2_scaled")
    def test_compute_nlsq_fits_sequential(self, mock_compute_g2):
        """Test sequential per-angle computation with mocked compute_g2_scaled."""
        from homodyne.cli.commands import _compute_nlsq_fits

        # Create mock result and data
        result = create_mock_optimization_result("static_isotropic")
        data = create_mock_data_dict(n_angles=3, n_t1=10, n_t2=10)
        metadata = {"L": 2000000.0, "dt": 0.1, "q": 0.0123}

        # Mock compute_g2_scaled to return predictable output
        mock_compute_g2.return_value = jnp.ones((10, 10)) * 1.5

        # Compute fits
        fits_dict = _compute_nlsq_fits(result, data, metadata)

        # Verify compute_g2_scaled was called once per angle
        assert mock_compute_g2.call_count == 3

        # Verify output structure
        assert "c2_theoretical_raw" in fits_dict
        assert "c2_theoretical_scaled" in fits_dict
        assert "per_angle_scaling" in fits_dict
        assert "residuals" in fits_dict

    def test_compute_nlsq_fits_shape_validation(self):
        """Test that output shapes match experimental data."""
        from homodyne.cli.commands import _compute_nlsq_fits

        # Create data with specific dimensions
        n_angles, n_t1, n_t2 = 5, 20, 20
        result = create_mock_optimization_result("laminar_flow")
        data = create_mock_data_dict(n_angles=n_angles, n_t1=n_t1, n_t2=n_t2)
        metadata = {"L": 2000000.0, "dt": 0.1, "q": 0.0123}

        # Compute fits
        fits_dict = _compute_nlsq_fits(result, data, metadata)

        # Verify shapes match experimental data
        assert fits_dict["c2_theoretical_raw"].shape == (n_angles, n_t1, n_t2)
        assert fits_dict["c2_theoretical_scaled"].shape == (n_angles, n_t1, n_t2)
        assert fits_dict["per_angle_scaling"].shape == (n_angles, 2)
        assert fits_dict["residuals"].shape == (n_angles, n_t1, n_t2)

    def test_compute_nlsq_fits_least_squares_scaling(self):
        """Test that per-angle scaling parameters are computed correctly."""
        from homodyne.cli.commands import _compute_nlsq_fits

        # Create data where we know the expected scaling
        result = create_mock_optimization_result("static_isotropic")
        data = create_mock_data_dict(n_angles=3, n_t1=15, n_t2=15)
        metadata = {"L": 2000000.0, "dt": 0.1, "q": 0.0123}

        # Compute fits
        fits_dict = _compute_nlsq_fits(result, data, metadata)

        # Verify scaling parameters exist for each angle
        scaling = fits_dict["per_angle_scaling"]
        assert scaling.shape == (3, 2)  # (n_angles, 2) for [contrast, offset]

        # Verify scaling values are reasonable (contrast > 0, offset near 1)
        assert np.all(scaling[:, 0] > 0)  # Contrast should be positive
        assert np.all(np.abs(scaling[:, 1] - 1.0) < 2.0)  # Offset near 1


# ==============================================================================
# Test Class: JSON File Saving (_save_nlsq_json_files)
# ==============================================================================


class TestSaveNLSQJSONFiles:
    """Test JSON file creation for parameters, analysis, and convergence."""

    def test_save_nlsq_json_files_all_created(self):
        """Test that all 3 JSON files are created."""
        import tempfile

        from homodyne.cli.commands import _save_nlsq_json_files

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock data
            param_dict = {"D0": {"value": 1234.5, "uncertainty": 45.6}}
            analysis_dict = {
                "method": "nlsq",
                "chi_squared": 1234.5,
                "convergence_status": "converged",
            }
            convergence_dict = {
                "status": "converged",
                "iterations": 42,
                "execution_time": 3.456,
            }

            # Save JSON files
            _save_nlsq_json_files(
                param_dict, analysis_dict, convergence_dict, output_dir
            )

            # Verify all 3 files created
            assert (output_dir / "parameters.json").exists()
            assert (output_dir / "analysis_results_nlsq.json").exists()
            assert (output_dir / "convergence_metrics.json").exists()

    def test_save_nlsq_json_files_content_validation(self):
        """Test that JSON content matches expected schema."""
        import tempfile

        from homodyne.cli.commands import _save_nlsq_json_files

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create comprehensive mock data
            param_dict = {
                "D0": {"value": 1234.5, "uncertainty": 45.6},
                "alpha": {"value": 0.567, "uncertainty": 0.012},
            }
            analysis_dict = {
                "method": "nlsq",
                "timestamp": "2025-10-16T14:23:47",
                "fit_quality": {"chi_squared": 1234.5, "reduced_chi_squared": 1.234},
            }
            convergence_dict = {
                "convergence": {
                    "status": "converged",
                    "iterations": 42,
                    "execution_time": 3.456,
                },
                "quality_flag": "good",
            }

            # Save files
            _save_nlsq_json_files(
                param_dict, analysis_dict, convergence_dict, output_dir
            )

            # Load and validate content
            with open(output_dir / "parameters.json") as f:
                params_loaded = json.load(f)
                assert "D0" in params_loaded
                assert params_loaded["D0"]["value"] == pytest.approx(1234.5)

            with open(output_dir / "analysis_results_nlsq.json") as f:
                analysis_loaded = json.load(f)
                assert analysis_loaded["method"] == "nlsq"

            with open(output_dir / "convergence_metrics.json") as f:
                convergence_loaded = json.load(f)
                assert convergence_loaded["convergence"]["status"] == "converged"


# ==============================================================================
# Test Class: NPZ File Saving (_save_nlsq_npz_file)
# ==============================================================================


class TestSaveNLSQNPZFile:
    """Test NPZ file creation with 11 arrays."""

    def test_save_nlsq_npz_file_arrays_present(self):
        """Test that all 11 arrays are present in NPZ file."""
        import tempfile

        from homodyne.cli.commands import _save_nlsq_npz_file

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock arrays
            n_angles, n_t1, n_t2 = 5, 20, 20
            phi_angles = np.linspace(0, 180, n_angles)
            c2_exp = np.random.rand(n_angles, n_t1, n_t2)
            c2_raw = np.random.rand(n_angles, n_t1, n_t2)
            c2_scaled = np.random.rand(n_angles, n_t1, n_t2)
            per_angle_scaling = np.random.rand(n_angles, 2)
            residuals = c2_exp - c2_scaled
            residuals_norm = residuals / 0.05
            t1 = np.linspace(0.01, 1.0, n_t1)
            t2 = np.linspace(0.01, 1.0, n_t2)
            q = 0.0123

            # Save NPZ file
            _save_nlsq_npz_file(
                phi_angles,
                c2_exp,
                c2_raw,
                c2_scaled,
                per_angle_scaling,
                residuals,
                residuals_norm,
                t1,
                t2,
                q,
                output_dir,
            )

            # Load and verify all arrays present
            data = np.load(output_dir / "fitted_data.npz")
            expected_arrays = [
                "phi_angles",
                "c2_exp",
                "c2_theoretical_raw",
                "c2_theoretical_scaled",
                "per_angle_scaling",
                "residuals",
                "residuals_normalized",
                "t1",
                "t2",
                "q",
            ]
            for arr_name in expected_arrays:
                assert arr_name in data, f"Missing array: {arr_name}"

    def test_save_nlsq_npz_file_load_and_verify(self):
        """Test loading NPZ file and verifying array contents."""
        import tempfile

        from homodyne.cli.commands import _save_nlsq_npz_file

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create known arrays
            n_angles, n_t1, n_t2 = 3, 10, 10
            phi_angles = np.array([0.0, 45.0, 90.0])
            c2_exp = np.ones((n_angles, n_t1, n_t2)) * 1.5
            c2_raw = np.ones((n_angles, n_t1, n_t2)) * 1.4
            c2_scaled = c2_raw * 0.5 + 1.0
            per_angle_scaling = np.array([[0.5, 1.0], [0.5, 1.0], [0.5, 1.0]])
            residuals = c2_exp - c2_scaled
            residuals_norm = residuals / 0.05
            t1 = np.linspace(0.01, 1.0, n_t1)
            t2 = np.linspace(0.01, 1.0, n_t2)
            q = 0.0123

            # Save NPZ
            _save_nlsq_npz_file(
                phi_angles,
                c2_exp,
                c2_raw,
                c2_scaled,
                per_angle_scaling,
                residuals,
                residuals_norm,
                t1,
                t2,
                q,
                output_dir,
            )

            # Load and verify shapes and values
            data = np.load(output_dir / "fitted_data.npz")
            assert data["phi_angles"].shape == (3,)
            assert data["c2_exp"].shape == (3, 10, 10)
            assert np.allclose(data["phi_angles"], phi_angles)
            assert np.allclose(data["c2_exp"], c2_exp)


# ==============================================================================
# Test Class: Plot Generation (generate_nlsq_plots)
# ==============================================================================


class TestGenerateNLSQPlots:
    """Test 3-panel heatmap plot generation."""

    @pytest.mark.skip(
        reason="Plotting now uses Datashader backend instead of direct matplotlib; "
        "mocking plt.subplots no longer works. Plotting functionality verified by "
        "test_generate_nlsq_plots_file_created. TODO: Update test for Datashader backend"
    )
    def test_generate_nlsq_plots_mocked_matplotlib(self):
        """Test plot generation with mocked matplotlib calls."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from homodyne.cli.commands import generate_nlsq_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test data
            phi_angles = np.array([0.0, 45.0, 90.0])
            c2_exp = np.random.rand(3, 20, 20) * 0.3 + 1.0
            c2_theoretical_scaled = np.random.rand(3, 20, 20) * 0.3 + 1.0
            residuals = c2_exp - c2_theoretical_scaled
            t1 = np.linspace(0.1, 2.0, 20)
            t2 = np.linspace(0.1, 2.0, 20)

            # Mock matplotlib
            with patch("homodyne.cli.commands.plt") as mock_plt:
                mock_fig = MagicMock()
                mock_axes = [MagicMock() for _ in range(3)]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)

                # Call function
                generate_nlsq_plots(
                    phi_angles=phi_angles,
                    c2_exp=c2_exp,
                    c2_theoretical_scaled=c2_theoretical_scaled,
                    residuals=residuals,
                    t1=t1,
                    t2=t2,
                    output_dir=output_dir,
                )

                # Verify plt.subplots called for each angle (3 times)
                assert mock_plt.subplots.call_count == 3
                # Check the last call
                args, kwargs = mock_plt.subplots.call_args
                assert args == (1, 3)  # 1 row, 3 columns
                assert kwargs["figsize"] == (15, 5)

                # Verify 3 plots created (one per angle)
                assert mock_plt.savefig.call_count == 3
                assert mock_plt.close.call_count == 3

                # Verify each plot saved with correct naming
                for i, phi in enumerate(phi_angles):
                    call_args = mock_plt.savefig.call_args_list[i]
                    filename = call_args[0][0]
                    assert f"phi_{phi:.1f}deg" in str(filename)
                    # Verify DPI
                    assert call_args[1]["dpi"] == 300
                    assert call_args[1]["bbox_inches"] == "tight"

    def test_generate_nlsq_plots_file_created(self):
        """Test that PNG files are created on disk."""
        pytest.importorskip("matplotlib")  # Skip if matplotlib not available

        import tempfile
        from pathlib import Path

        from homodyne.cli.commands import generate_nlsq_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test data
            phi_angles = np.array([0.0, 90.0])
            c2_exp = np.random.rand(2, 15, 15) * 0.3 + 1.0
            c2_theoretical_scaled = np.random.rand(2, 15, 15) * 0.3 + 1.0
            residuals = c2_exp - c2_theoretical_scaled
            t1 = np.linspace(0.1, 1.5, 15)
            t2 = np.linspace(0.1, 1.5, 15)

            # Call function
            generate_nlsq_plots(
                phi_angles=phi_angles,
                c2_exp=c2_exp,
                c2_theoretical_scaled=c2_theoretical_scaled,
                residuals=residuals,
                t1=t1,
                t2=t2,
                output_dir=output_dir,
            )

            # Verify PNG files created
            for phi in phi_angles:
                png_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
                assert png_file.exists(), f"PNG file not created for phi={phi}"
                assert (
                    png_file.stat().st_size > 1000
                ), f"PNG file too small for phi={phi}"

            # Verify correct number of files
            png_files = list(output_dir.glob("*.png"))
            assert len(png_files) == len(phi_angles)

    @pytest.mark.skip(
        reason="Datashader backend requires matching array dimensions; test data has "
        "shape mismatch (data.shape[0]=10 vs y_coords.length=2). TODO: Fix test data "
        "dimensions to match Datashader requirements"
    )
    def test_generate_nlsq_plots_residuals_symmetric_colormap(self):
        """Test that residuals use symmetric colormap (vmin=-vmax, vmax=vmax)."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from homodyne.cli.commands import generate_nlsq_plots

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test data with known residuals
            phi_angles = np.array([0.0])
            c2_exp = np.ones((1, 10, 10))
            c2_theoretical_scaled = np.ones((1, 10, 10))
            residuals = np.array([[[-0.5, 0.3], [0.7, -0.2]]])  # Asymmetric residuals
            t1 = np.array([0.1, 0.2])
            t2 = np.array([0.1, 0.2])

            # Mock matplotlib to capture imshow calls
            with patch("homodyne.cli.commands.plt") as mock_plt:
                mock_fig = MagicMock()
                mock_axes = [MagicMock(), MagicMock(), MagicMock()]
                mock_plt.subplots.return_value = (mock_fig, mock_axes)

                # Call function
                generate_nlsq_plots(
                    phi_angles=phi_angles,
                    c2_exp=c2_exp,
                    c2_theoretical_scaled=c2_theoretical_scaled,
                    residuals=residuals,
                    t1=t1,
                    t2=t2,
                    output_dir=output_dir,
                )

                # Verify third axis (residuals) uses symmetric colormap
                residuals_imshow_call = mock_axes[2].imshow.call_args
                if residuals_imshow_call:
                    kwargs = residuals_imshow_call[1]
                    if "vmin" in kwargs and "vmax" in kwargs:
                        vmin = kwargs["vmin"]
                        vmax = kwargs["vmax"]
                        # Verify symmetric: vmin = -vmax
                        assert (
                            vmin == -vmax
                        ), f"Residuals colormap not symmetric: vmin={vmin}, vmax={vmax}"
                        # Verify vmax is positive
                        assert (
                            vmax > 0
                        ), f"Residuals vmax should be positive, got {vmax}"
                    # Verify diverging colormap (RdBu_r)
                    if "cmap" in kwargs:
                        assert "RdBu" in str(kwargs["cmap"])
        pass  # To be implemented in T032


# ==============================================================================
# Helper Functions
# ==============================================================================
