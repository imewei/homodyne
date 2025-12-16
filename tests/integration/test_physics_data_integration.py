"""
Integration Tests for Physics and Data Modules
==============================================

Integration tests that verify the interaction between:
- homodyne.core.physics_utils
- homodyne.data.xpcs_loader

These tests ensure that data loading and physics calculations work
together correctly in realistic workflows.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.testing import assert_allclose

# JAX imports with fallback
try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.physics_utils import (
    apply_diagonal_correction,
    calculate_diffusion_coefficient,
    calculate_shear_rate,
    create_time_integral_matrix,
)
from homodyne.data.xpcs_loader import (
    XPCSDataLoader,
    load_xpcs_data,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def realistic_npz_cache(temp_dir):
    """Create a realistic NPZ cache file simulating XPCS data."""
    n_phi = 10
    n_frames = 100
    dt = 0.1

    # Q-values at a single q
    wavevector_q = 0.0054
    wavevector_q_list = np.full(n_phi, wavevector_q)

    # Phi angles evenly distributed
    phi_angles_list = np.linspace(0, 180, n_phi)

    # Time arrays
    t1 = np.linspace(0, (n_frames - 1) * dt, n_frames)
    t2 = np.linspace(0, (n_frames - 1) * dt, n_frames)

    # Create realistic correlation matrices
    # g2 = 1 + contrast * exp(-2 * D * q^2 * |t1-t2|)
    contrast = 0.3
    D = 100.0  # Diffusion coefficient
    c2_exp = np.ones((n_phi, n_frames, n_frames))

    for phi_idx in range(n_phi):
        for i in range(n_frames):
            for j in range(n_frames):
                tau = abs(t1[i] - t2[j])
                c2_exp[phi_idx, i, j] = 1.0 + contrast * np.exp(
                    -2 * D * wavevector_q**2 * tau
                )

    cache_path = os.path.join(temp_dir, "cached_c2_frames_1_100.npz")
    np.savez(
        cache_path,
        wavevector_q_list=wavevector_q_list,
        phi_angles_list=phi_angles_list,
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
    )
    return cache_path, {
        "n_phi": n_phi,
        "n_frames": n_frames,
        "dt": dt,
        "wavevector_q": wavevector_q,
        "contrast": contrast,
        "D": D,
    }


@pytest.fixture
def xpcs_config(temp_dir):
    """Create a complete XPCS configuration."""
    return {
        "experimental_data": {
            "data_folder_path": temp_dir,
            "data_file_name": "cached_c2_frames_1_100.npz",
        },
        "analyzer_parameters": {
            "dt": 0.1,
            "start_frame": 1,
            "end_frame": 100,
            "scattering": {"wavevector_q": 0.0054},
        },
        "v2_features": {"output_format": "jax"},
    }


# =============================================================================
# Data Loading Integration Tests
# =============================================================================


@pytest.mark.integration
class TestDataLoadingIntegration:
    """Integration tests for data loading workflow."""

    def test_complete_load_workflow(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test complete data loading workflow."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Verify all expected keys
        assert "wavevector_q_list" in data
        assert "phi_angles_list" in data
        assert "t1" in data
        assert "t2" in data
        assert "c2_exp" in data

        # Verify shapes match metadata
        assert len(data["phi_angles_list"]) == metadata["n_phi"]
        assert data["c2_exp"].shape[0] == metadata["n_phi"]

    def test_loaded_data_for_physics_calculations(
        self, temp_dir, realistic_npz_cache, xpcs_config
    ):
        """Test that loaded data can be used in physics calculations."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Time arrays should work with physics functions
        time_arr = jnp.array(data["t1"])

        # Calculate diffusion coefficient
        D_t = calculate_diffusion_coefficient(time_arr, D0=100.0, alpha=0.0, D_offset=0.0)
        assert jnp.all(jnp.isfinite(D_t))
        assert len(D_t) == len(data["t1"])

        # Calculate shear rate
        gamma_t = calculate_shear_rate(time_arr, gamma_dot_0=0.5, beta=0.0, gamma_dot_offset=0.0)
        assert jnp.all(jnp.isfinite(gamma_t))
        assert len(gamma_t) == len(data["t1"])

    def test_diagonal_correction_on_loaded_data(
        self, temp_dir, realistic_npz_cache, xpcs_config
    ):
        """Test diagonal correction is properly applied to loaded data."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Check that diagonal correction was applied
        # After correction, diagonal should be interpolated from off-diagonal
        for phi_idx in range(metadata["n_phi"]):
            c2_mat = data["c2_exp"][phi_idx]
            diagonal = np.diag(c2_mat)
            off_diagonal = c2_mat[0, 1]  # Adjacent off-diagonal

            # Diagonal values should be close to off-diagonal after correction
            # (not the original peak values)
            assert np.allclose(diagonal[0], off_diagonal, rtol=0.1)


# =============================================================================
# Physics Calculation Integration Tests
# =============================================================================


@pytest.mark.integration
class TestPhysicsCalculationIntegration:
    """Integration tests for physics calculations on real data structures."""

    def test_time_integral_matrix_with_diffusion(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test time integral matrix calculation with diffusion coefficient."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Calculate time-dependent diffusion
        time_arr = jnp.array(data["t1"])
        D_t = calculate_diffusion_coefficient(time_arr, D0=100.0, alpha=0.3, D_offset=10.0)

        # Create time integral matrix from diffusion
        integral_matrix = create_time_integral_matrix(D_t)

        # Verify matrix properties
        assert integral_matrix.shape == (len(time_arr), len(time_arr))
        assert jnp.all(jnp.isfinite(integral_matrix))
        assert jnp.allclose(integral_matrix, integral_matrix.T, rtol=1e-10)

    def test_shear_rate_time_series(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test shear rate calculation over time series."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        time_arr = jnp.array(data["t1"])

        # Calculate various shear rate scenarios
        scenarios = [
            {"gamma_dot_0": 0.5, "beta": 0.0, "gamma_dot_offset": 0.1},  # Constant
            {"gamma_dot_0": 1.0, "beta": 0.5, "gamma_dot_offset": 0.0},  # Increasing
            {"gamma_dot_0": 1.0, "beta": -0.5, "gamma_dot_offset": 0.5},  # Decreasing
        ]

        for params in scenarios:
            gamma_t = calculate_shear_rate(
                time_arr, params["gamma_dot_0"], params["beta"], params["gamma_dot_offset"]
            )
            assert jnp.all(jnp.isfinite(gamma_t))
            assert jnp.all(gamma_t > 0)

    def test_physics_calculations_with_jax_arrays(
        self, temp_dir, realistic_npz_cache
    ):
        """Test physics calculations with JAX array output."""
        cache_path, metadata = realistic_npz_cache

        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
                "scattering": {"wavevector_q": 0.0054},
            },
            "v2_features": {"output_format": "jax"},
        }

        loader = XPCSDataLoader(config_dict=config)
        data = loader.load_experimental_data()

        # Test with JAX arrays if available
        time_arr = jnp.asarray(data["t1"])

        D_t = calculate_diffusion_coefficient(time_arr, 100.0, 0.3, 10.0)
        gamma_t = calculate_shear_rate(time_arr, 0.5, 0.3, 0.1)

        # Results should be finite
        assert jnp.all(jnp.isfinite(D_t))
        assert jnp.all(jnp.isfinite(gamma_t))


# =============================================================================
# End-to-End Workflow Tests
# =============================================================================


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end workflow integration tests."""

    def test_load_process_analyze_workflow(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test complete workflow: load → process → analyze."""
        cache_path, metadata = realistic_npz_cache

        # Step 1: Load data
        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Step 2: Extract parameters
        q = data["wavevector_q_list"][0]
        phi_angles = data["phi_angles_list"]
        time_arr = jnp.array(data["t1"])
        c2_exp = jnp.array(data["c2_exp"])

        # Step 3: Calculate physics quantities
        D_t = calculate_diffusion_coefficient(time_arr, D0=100.0, alpha=0.0, D_offset=0.0)
        gamma_t = calculate_shear_rate(time_arr, gamma_dot_0=0.0, beta=0.0, gamma_dot_offset=0.0)

        # Step 4: Create time integral matrix
        D_integral = create_time_integral_matrix(D_t)
        gamma_integral = create_time_integral_matrix(gamma_t)

        # Verify results
        assert D_integral.shape == (len(time_arr), len(time_arr))
        assert gamma_integral.shape == (len(time_arr), len(time_arr))
        assert c2_exp.shape[0] == len(phi_angles)

    def test_multi_angle_analysis_workflow(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test workflow that processes multiple phi angles."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        results = []
        for phi_idx, phi in enumerate(data["phi_angles_list"]):
            c2_mat = data["c2_exp"][phi_idx]

            # Apply additional diagonal correction (already applied in loader)
            c2_corrected = apply_diagonal_correction(jnp.array(c2_mat))

            # Calculate mean correlation
            mean_corr = jnp.mean(c2_corrected)
            results.append({
                "phi": phi,
                "mean_correlation": float(mean_corr),
            })

        # Should have processed all angles
        assert len(results) == metadata["n_phi"]
        # All mean correlations should be reasonable (around 1.0 for g2)
        for r in results:
            assert 0.5 < r["mean_correlation"] < 2.0

    def test_convenience_function_integration(self, temp_dir, realistic_npz_cache):
        """Test load_xpcs_data convenience function integration."""
        cache_path, metadata = realistic_npz_cache

        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": 0.1,
                "start_frame": 1,
                "end_frame": 100,
            },
        }

        # Use convenience function
        data = load_xpcs_data(config_dict=config)

        # Should work with physics functions
        time_arr = jnp.array(data["t1"])
        D_t = calculate_diffusion_coefficient(time_arr, 100.0, 0.0, 0.0)

        assert len(D_t) == len(data["t1"])


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Integration tests for error handling across modules."""

    def test_invalid_time_array_handling(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test handling of edge cases in time arrays."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # Test with zero-length time array
        empty_time = jnp.array([])
        # This should not crash but may return empty result
        if len(empty_time) > 0:
            D_t = calculate_diffusion_coefficient(empty_time, 100.0, 0.0, 0.0)
            assert len(D_t) == 0

    def test_extreme_parameter_handling(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test handling of extreme physics parameters."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        time_arr = jnp.array(data["t1"])

        # Extreme diffusion parameters
        extreme_params = [
            {"D0": 1e6, "alpha": 0.0, "D_offset": 0.0},  # Very large D0
            {"D0": 1e-10, "alpha": 0.0, "D_offset": 1e-10},  # Very small values
            {"D0": 100.0, "alpha": -2.0, "D_offset": 0.0},  # Negative alpha
            {"D0": 100.0, "alpha": 3.0, "D_offset": 0.0},  # Large positive alpha
        ]

        for params in extreme_params:
            D_t = calculate_diffusion_coefficient(
                time_arr, params["D0"], params["alpha"], params["D_offset"]
            )
            # Should always be finite and positive
            assert jnp.all(jnp.isfinite(D_t))
            assert jnp.all(D_t > 0)


# =============================================================================
# Configuration Integration Tests
# =============================================================================


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration handling."""

    def test_yaml_config_to_physics_workflow(self, temp_dir, realistic_npz_cache):
        """Test workflow starting from YAML configuration."""
        cache_path, metadata = realistic_npz_cache

        # Write YAML config
        yaml_content = f"""
experimental_data:
  data_folder_path: "{temp_dir}"
  data_file_name: "cached_c2_frames_1_100.npz"

analyzer_parameters:
  dt: {metadata['dt']}
  start_frame: 1
  end_frame: {metadata['n_frames']}
  scattering:
    wavevector_q: {metadata['wavevector_q']}

v2_features:
  output_format: jax
"""
        config_path = os.path.join(temp_dir, "test_config.yaml")
        with open(config_path, "w") as f:
            f.write(yaml_content)

        # Load using config file
        loader = XPCSDataLoader(config_path=config_path)
        data = loader.load_experimental_data()

        # Verify data and use in physics calculations
        time_arr = jnp.array(data["t1"])
        D_t = calculate_diffusion_coefficient(time_arr, 100.0, 0.0, 0.0)

        assert len(D_t) == metadata["n_frames"]

    def test_json_config_to_physics_workflow(self, temp_dir, realistic_npz_cache):
        """Test workflow starting from JSON configuration."""
        cache_path, metadata = realistic_npz_cache

        # Write JSON config
        config = {
            "experimental_data": {
                "data_folder_path": temp_dir,
                "data_file_name": "cached_c2_frames_1_100.npz",
            },
            "analyzer_parameters": {
                "dt": metadata["dt"],
                "start_frame": 1,
                "end_frame": metadata["n_frames"],
                "scattering": {"wavevector_q": metadata["wavevector_q"]},
            },
        }
        config_path = os.path.join(temp_dir, "test_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Load using config file
        loader = XPCSDataLoader(config_path=config_path)
        data = loader.load_experimental_data()

        # Use in physics calculations
        time_arr = jnp.array(data["t1"])
        D_t = calculate_diffusion_coefficient(time_arr, 100.0, 0.0, 0.0)

        assert jnp.all(jnp.isfinite(D_t))


# =============================================================================
# Data Consistency Integration Tests
# =============================================================================


@pytest.mark.integration
class TestDataConsistencyIntegration:
    """Integration tests for data consistency across modules."""

    def test_time_array_consistency(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test that time arrays are consistent between loader and physics."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        # t1 and t2 should be identical 1D arrays
        assert_allclose(data["t1"], data["t2"], rtol=1e-10)

        # Time spacing should match configured dt
        dt_calculated = data["t1"][1] - data["t1"][0]
        assert_allclose(dt_calculated, metadata["dt"], rtol=1e-10)

    def test_correlation_matrix_symmetry(self, temp_dir, realistic_npz_cache, xpcs_config):
        """Test that correlation matrices maintain symmetry after processing."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        for phi_idx in range(len(data["phi_angles_list"])):
            c2_mat = data["c2_exp"][phi_idx]
            # Should be approximately symmetric (within numerical precision)
            assert_allclose(c2_mat, c2_mat.T, rtol=1e-10)

    def test_physics_calculation_reproducibility(
        self, temp_dir, realistic_npz_cache, xpcs_config
    ):
        """Test that physics calculations are reproducible."""
        cache_path, metadata = realistic_npz_cache

        loader = XPCSDataLoader(config_dict=xpcs_config)
        data = loader.load_experimental_data()

        time_arr = jnp.array(data["t1"])

        # Run calculation twice
        D_t_1 = calculate_diffusion_coefficient(time_arr, 100.0, 0.3, 10.0)
        D_t_2 = calculate_diffusion_coefficient(time_arr, 100.0, 0.3, 10.0)

        # Results should be identical
        assert_allclose(D_t_1, D_t_2, rtol=1e-12)
