"""
Unit Tests for HomodyneModel Hybrid Architecture
=================================================

Tests for the HomodyneModel class which implements the hybrid architecture
combining stateful robustness with functional JIT performance.
"""

import numpy as np
import pytest

from homodyne.core.homodyne_model import HomodyneModel
from homodyne.core.physics_factors import PhysicsFactors


class TestHomodyneModelInitialization:
    """Test HomodyneModel initialization and configuration extraction."""

    def test_initialization_laminar_flow(self):
        """Test initialization with laminar flow configuration."""
        config = {
            "analyzer_parameters": {
                "temporal": {
                    "dt": 0.1,
                    "start_frame": 1,
                    "end_frame": 100,
                },
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            },
            "analysis_settings": {
                "static_mode": False,
            },
        }

        model = HomodyneModel(config)

        assert model.dt == 0.1
        assert model.wavevector_q == 0.01
        assert model.stator_rotor_gap == 2e6
        assert model.analysis_mode == "laminar_flow"
        assert len(model.time_array) == 100
        assert model.t1_grid.shape == (100, 100)
        assert model.t2_grid.shape == (100, 100)

    def test_initialization_static_mode(self):
        """Test initialization with static mode configuration."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.05, "start_frame": 0, "end_frame": 49},
                "scattering": {"wavevector_q": 0.02},
                "geometry": {"stator_rotor_gap": 1e6},
            },
            "analysis_settings": {
                "static_mode": True,
                "isotropic_mode": True,
            },
        }

        model = HomodyneModel(config)

        assert model.analysis_mode == "static_isotropic"
        assert len(model.time_array) == 50

    def test_physics_factors_precomputed(self):
        """Test that physics factors are pre-computed at initialization."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 10},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }

        model = HomodyneModel(config)

        assert isinstance(model.physics_factors, PhysicsFactors)
        assert model.physics_factors.wavevector_q == 0.01
        assert model.physics_factors.dt == 0.1

        # Check pre-computed values
        expected_q_factor = 0.5 * (0.01**2) * 0.1
        expected_sinc_factor = 0.5 / np.pi * 0.01 * 2e6 * 0.1

        assert model.physics_factors.wavevector_q_squared_half_dt == pytest.approx(
            expected_q_factor
        )
        assert model.physics_factors.sinc_prefactor == pytest.approx(
            expected_sinc_factor
        )

    def test_time_array_correctness(self):
        """Test that time array follows correct formula: t[i] = dt * i."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.2, "start_frame": 1, "end_frame": 50},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }

        model = HomodyneModel(config)

        # Check time array formula
        n = len(model.time_array)
        assert n == 50

        for i in range(n):
            expected_t = 0.2 * i
            # Use float32-appropriate tolerance (JAX defaults to float32)
            assert model.time_array[i] == pytest.approx(expected_t, abs=1e-6)

        # Check time_max
        assert model.time_array[-1] == pytest.approx(0.2 * (n - 1), abs=1e-6)

    def test_missing_config_keys(self):
        """Test that missing configuration keys raise appropriate errors."""
        invalid_config = {"some_key": "some_value"}

        with pytest.raises(KeyError) as exc_info:
            HomodyneModel(invalid_config)

        assert "analyzer_parameters" in str(exc_info.value)


class TestHomodyneModelComputation:
    """Test HomodyneModel C2 computation methods."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 20},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }
        return HomodyneModel(config)

    def test_compute_c2_shape(self, model):
        """Test that compute_c2 returns correct shape."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([0, 30, 45, 60, 90])

        c2 = model.compute_c2(params, phi_angles)

        assert c2.shape == (5, 20, 20)  # 5 angles, 20 time points

    def test_compute_c2_values_finite(self, model):
        """Test that computed C2 values are finite."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([0, 45, 90])

        c2 = model.compute_c2(params, phi_angles)

        assert np.all(np.isfinite(c2))

    def test_compute_c2_positive(self, model):
        """Test that C2 values are positive (with default offset=1.0)."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([0, 45, 90])

        c2 = model.compute_c2(params, phi_angles, contrast=0.5, offset=1.0)

        assert np.all(c2 > 0)

    def test_compute_c2_single_angle(self, model):
        """Test single angle computation."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi = 45.0

        c2 = model.compute_c2_single_angle(params, phi)

        assert c2.shape == (20, 20)
        assert np.all(np.isfinite(c2))

    def test_compute_c2_contrast_effect(self, model):
        """Test that contrast parameter affects results."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([45])

        c2_low = model.compute_c2(params, phi_angles, contrast=0.3, offset=1.0)
        c2_high = model.compute_c2(params, phi_angles, contrast=0.7, offset=1.0)

        # Different contrast should give different results
        assert not np.allclose(c2_low, c2_high)

    def test_compute_c2_deterministic(self, model):
        """Test that repeated calls give identical results."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([0, 45, 90])

        c2_1 = model.compute_c2(params, phi_angles)
        c2_2 = model.compute_c2(params, phi_angles)

        assert np.allclose(c2_1, c2_2, atol=1e-10)


class TestHomodyneModelConvenienceMethods:
    """Test HomodyneModel convenience methods."""

    @pytest.fixture
    def model(self):
        """Create a test model."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 10},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }
        return HomodyneModel(config)

    def test_config_summary(self, model):
        """Test config_summary property."""
        summary = model.config_summary

        assert "dt" in summary
        assert "time_length" in summary
        assert "wavevector_q" in summary
        assert "analysis_mode" in summary
        assert "physics_factors" in summary

        assert summary["dt"] == 0.1
        assert summary["time_length"] == 10
        assert summary["wavevector_q"] == 0.01

    def test_repr(self, model):
        """Test string representation."""
        repr_str = repr(model)

        assert "HomodyneModel" in repr_str
        assert "dt=" in repr_str
        assert "time_points=" in repr_str

    def test_plot_simulated_data_no_plots(self, model, tmp_path):
        """Test plot_simulated_data without generating plots."""
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi_angles = np.array([0, 45])

        c2_data, output_path = model.plot_simulated_data(
            params,
            phi_angles,
            output_dir=str(tmp_path),
            generate_plots=False,  # Don't generate plots
        )

        # Check data
        assert c2_data.shape == (2, 10, 10)

        # Check file was created
        assert output_path.exists()

        # Load and verify
        loaded = np.load(output_path)
        assert "c2_data" in loaded
        assert np.allclose(loaded["c2_data"], c2_data)


class TestHomodyneModelVsLegacy:
    """Test HomodyneModel compatibility with legacy CombinedModel."""

    def test_same_results_as_combined_model(self):
        """Test that HomodyneModel gives same results as CombinedModel."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 20},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }

        # Create HomodyneModel
        hybrid_model = HomodyneModel(config)

        # Create legacy CombinedModel
        from homodyne.core.models import CombinedModel

        legacy_model = CombinedModel("laminar_flow")

        # Same parameters
        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        phi = 45.0

        # Compute with HomodyneModel
        c2_hybrid = hybrid_model.compute_c2_single_angle(params, phi)

        # Compute with legacy model
        import jax.numpy as jnp

        phi_array = jnp.array([phi])
        c2_legacy = legacy_model.compute_g2(
            jnp.array(params),
            hybrid_model.t1_grid,
            hybrid_model.t2_grid,
            phi_array,
            hybrid_model.wavevector_q,
            hybrid_model.stator_rotor_gap,
            0.5,  # contrast
            1.0,  # offset
            hybrid_model.dt,  # dt
        )[0]

        # Should give identical results
        assert np.allclose(c2_hybrid, np.array(c2_legacy), atol=1e-8)


class TestHomodyneModelEdgeCases:
    """Test edge cases and error handling."""

    def test_single_time_point(self):
        """Test with single time point."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 1},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }

        model = HomodyneModel(config)
        assert len(model.time_array) == 1

        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        c2 = model.compute_c2_single_angle(params, 0)
        assert c2.shape == (1, 1)

    def test_very_small_dt(self):
        """Test with very small dt value."""
        config = {
            "analyzer_parameters": {
                "temporal": {"dt": 1e-6, "start_frame": 1, "end_frame": 10},
                "scattering": {"wavevector_q": 0.01},
                "geometry": {"stator_rotor_gap": 2e6},
            }
        }

        model = HomodyneModel(config)
        assert model.dt == 1e-6

        params = np.array([100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0])
        c2 = model.compute_c2_single_angle(params, 45)
        assert np.all(np.isfinite(c2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
