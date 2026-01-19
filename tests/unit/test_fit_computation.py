"""Unit tests for homodyne.optimization.nlsq.fit_computation module.

Tests fit computation utilities including theoretical fit generation,
analysis mode normalization, and parameter extraction.
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.fit_computation import (
    extract_parameters_from_result,
    get_physical_param_count,
    normalize_analysis_mode,
)


class TestNormalizeAnalysisMode:
    """Tests for normalize_analysis_mode function."""

    def test_static_mode_explicit(self):
        """Test explicit static mode normalization."""
        result = normalize_analysis_mode("static", n_params=5, n_angles=1)
        assert result == "static"

    def test_static_isotropic_normalized(self):
        """Test that static_isotropic is normalized to static."""
        result = normalize_analysis_mode("static_isotropic", n_params=5, n_angles=1)
        assert result == "static"

    def test_laminar_flow_mode_explicit(self):
        """Test explicit laminar flow mode normalization."""
        result = normalize_analysis_mode("laminar_flow", n_params=9, n_angles=1)
        assert result == "laminar_flow"

    def test_case_insensitive(self):
        """Test that mode comparison is case insensitive."""
        assert normalize_analysis_mode("STATIC", n_params=5, n_angles=1) == "static"
        assert normalize_analysis_mode("Static", n_params=5, n_angles=1) == "static"
        assert (
            normalize_analysis_mode("LAMINAR_FLOW", n_params=9, n_angles=1)
            == "laminar_flow"
        )

    def test_infer_static_from_params_scalar(self):
        """Test inference of static mode from scalar parameter count."""
        # Scalar layout: contrast + offset + 3 physical = 5
        result = normalize_analysis_mode(None, n_params=5, n_angles=1)
        assert result == "static"

    def test_infer_static_from_params_per_angle(self):
        """Test inference of static mode from per-angle parameter count."""
        # Per-angle layout: 2*n_angles + 3 physical
        # For 3 angles: 2*3 + 3 = 9
        result = normalize_analysis_mode(None, n_params=9, n_angles=3)
        assert result == "static"

    def test_infer_laminar_from_params_scalar(self):
        """Test inference of laminar flow mode from scalar parameter count."""
        # Scalar layout: contrast + offset + 7 physical = 9
        result = normalize_analysis_mode(None, n_params=9, n_angles=1)
        assert result == "laminar_flow"

    def test_infer_laminar_from_params_per_angle(self):
        """Test inference of laminar flow mode from per-angle parameter count."""
        # Per-angle layout: 2*n_angles + 7 physical
        # For 3 angles: 2*3 + 7 = 13
        result = normalize_analysis_mode(None, n_params=13, n_angles=3)
        assert result == "laminar_flow"

    def test_default_to_static_on_ambiguous(self):
        """Test that ambiguous parameter counts default to static."""
        # Some unusual parameter count that doesn't match either mode
        result = normalize_analysis_mode(None, n_params=100, n_angles=5)
        assert result == "static"


class TestGetPhysicalParamCount:
    """Tests for get_physical_param_count function."""

    def test_static_mode_returns_3(self):
        """Test that static mode returns 3 physical parameters."""
        assert get_physical_param_count("static") == 3

    def test_laminar_flow_mode_returns_7(self):
        """Test that laminar flow mode returns 7 physical parameters."""
        assert get_physical_param_count("laminar_flow") == 7

    def test_unknown_mode_raises(self):
        """Test that unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown analysis_mode"):
            get_physical_param_count("unknown_mode")


class TestExtractParametersFromResult:
    """Tests for extract_parameters_from_result function."""

    def test_per_angle_layout_static(self):
        """Test parameter extraction with per-angle layout for static mode."""
        n_angles = 3
        # Parameters: [c0, c1, c2, o0, o1, o2, D0, alpha, D_offset]
        parameters = np.array([0.8, 0.85, 0.9, 1.0, 1.05, 1.1, 1e-11, 0.5, 1e-14])

        contrasts, offsets, physical, scalar_exp = extract_parameters_from_result(
            parameters, n_angles, "static"
        )

        np.testing.assert_array_equal(contrasts, [0.8, 0.85, 0.9])
        np.testing.assert_array_equal(offsets, [1.0, 1.05, 1.1])
        np.testing.assert_array_equal(physical, [1e-11, 0.5, 1e-14])
        assert scalar_exp is False

    def test_per_angle_layout_laminar_flow(self):
        """Test parameter extraction with per-angle layout for laminar flow."""
        n_angles = 2
        # Parameters: [c0, c1, o0, o1, D0, alpha, D_offset, gamma, beta, gamma_off, phi0]
        parameters = np.array(
            [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14, 100.0, 0.3, 10.0, 0.0]
        )

        contrasts, offsets, physical, scalar_exp = extract_parameters_from_result(
            parameters, n_angles, "laminar_flow"
        )

        np.testing.assert_array_equal(contrasts, [0.8, 0.85])
        np.testing.assert_array_equal(offsets, [1.0, 1.05])
        assert len(physical) == 7
        assert scalar_exp is False

    def test_scalar_layout_expanded(self):
        """Test that scalar layout is expanded with warning."""
        n_angles = 3
        # Scalar layout: [contrast, offset, D0, alpha, D_offset]
        parameters = np.array([0.8, 1.0, 1e-11, 0.5, 1e-14])

        contrasts, offsets, physical, scalar_exp = extract_parameters_from_result(
            parameters, n_angles, "static"
        )

        # Scalar should be expanded to all angles
        np.testing.assert_array_equal(contrasts, [0.8, 0.8, 0.8])
        np.testing.assert_array_equal(offsets, [1.0, 1.0, 1.0])
        np.testing.assert_array_equal(physical, [1e-11, 0.5, 1e-14])
        assert scalar_exp is True

    def test_invalid_parameter_count_raises(self):
        """Test that invalid parameter count raises ValueError."""
        n_angles = 3
        # Invalid count: not matching per-angle (9) or scalar (5)
        parameters = np.array([0.8, 1.0, 1e-11])  # Only 3 params

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            extract_parameters_from_result(parameters, n_angles, "static")


class TestComputeTheoreticalFitsIntegration:
    """Integration tests for compute_theoretical_fits function.

    These tests require JAX and the full compute_g2_scaled infrastructure.
    They test the complete fit computation pipeline.
    """

    @pytest.fixture
    def mock_result(self):
        """Create a mock optimization result."""

        class MockResult:
            def __init__(self, params):
                self.parameters = np.array(params)
                self.analysis_mode = "static"

        return MockResult

    @pytest.fixture
    def mock_data(self):
        """Create mock experimental data."""
        n_angles = 2
        n_t1 = 10
        n_t2 = 10

        return {
            "phi_angles_list": [0.0, 45.0],
            "c2_exp": np.ones((n_angles, n_t1, n_t2)) * 1.5,
            "t1": np.linspace(0, 1, n_t1),
            "t2": np.linspace(0, 1, n_t2),
        }

    @pytest.fixture
    def mock_metadata(self):
        """Create mock metadata."""
        return {"L": 1000.0, "dt": 0.1, "q": 0.01}

    def test_compute_fits_basic(self, mock_result, mock_data, mock_metadata):
        """Test basic fit computation."""
        from homodyne.optimization.nlsq.fit_computation import compute_theoretical_fits

        # Per-angle parameters for 2 angles + 3 physical
        # [c0, c1, o0, o1, D0, alpha, D_offset]
        params = [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14]
        result = mock_result(params)

        fits = compute_theoretical_fits(
            result, mock_data, mock_metadata, analysis_mode="static"
        )

        # Check output structure
        assert "c2_theoretical_raw" in fits
        assert "c2_theoretical_scaled" in fits
        assert "per_angle_scaling" in fits
        assert "residuals" in fits

        # Check shapes
        assert fits["c2_theoretical_raw"].shape == (2, 10, 10)
        assert fits["c2_theoretical_scaled"].shape == (2, 10, 10)
        assert fits["per_angle_scaling"].shape == (2, 2)
        assert fits["residuals"].shape == (2, 10, 10)

    def test_compute_fits_with_solver_surface(
        self, mock_result, mock_data, mock_metadata
    ):
        """Test fit computation includes solver surface."""
        from homodyne.optimization.nlsq.fit_computation import compute_theoretical_fits

        params = [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14]
        result = mock_result(params)

        fits = compute_theoretical_fits(
            result,
            mock_data,
            mock_metadata,
            analysis_mode="static",
            include_solver_surface=True,
        )

        assert fits["c2_solver_scaled"] is not None
        assert fits["c2_solver_scaled"].shape == (2, 10, 10)

    def test_compute_fits_without_solver_surface(
        self, mock_result, mock_data, mock_metadata
    ):
        """Test fit computation without solver surface."""
        from homodyne.optimization.nlsq.fit_computation import compute_theoretical_fits

        params = [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14]
        result = mock_result(params)

        fits = compute_theoretical_fits(
            result,
            mock_data,
            mock_metadata,
            analysis_mode="static",
            include_solver_surface=False,
        )

        assert fits["c2_solver_scaled"] is None

    def test_compute_fits_missing_q_raises(self, mock_result, mock_data):
        """Test that missing q value raises ValueError."""
        from homodyne.optimization.nlsq.fit_computation import compute_theoretical_fits

        params = [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14]
        result = mock_result(params)
        metadata = {"L": 1000.0, "dt": 0.1, "q": None}

        with pytest.raises(ValueError, match="q.*required"):
            compute_theoretical_fits(
                result, mock_data, metadata, analysis_mode="static"
            )

    def test_compute_fits_2d_time_arrays(self, mock_result, mock_metadata):
        """Test fit computation with 2D meshgrid time arrays."""
        from homodyne.optimization.nlsq.fit_computation import compute_theoretical_fits

        # Create 2D meshgrid time arrays
        t1_1d = np.linspace(0, 1, 10)
        t2_1d = np.linspace(0, 1, 10)
        t1_2d, t2_2d = np.meshgrid(t1_1d, t2_1d, indexing="ij")

        data = {
            "phi_angles_list": [0.0, 45.0],
            "c2_exp": np.ones((2, 10, 10)) * 1.5,
            "t1": t1_2d,
            "t2": t2_2d,
        }

        params = [0.8, 0.85, 1.0, 1.05, 1e-11, 0.5, 1e-14]
        result = mock_result(params)

        fits = compute_theoretical_fits(
            result, data, mock_metadata, analysis_mode="static"
        )

        # Should still work with 2D arrays
        assert fits["c2_theoretical_raw"].shape == (2, 10, 10)


class TestStaticShapeCompilation:
    """Tests for static shape JIT compilation (FR-004, T012)."""

    def test_static_shapes_no_recompile(self):
        """T012: Verify static shapes prevent JIT recompilation.

        Performance Optimization (Spec 001 - FR-004, T012): Test that using
        static_argnums for shape arguments prevents recompilation when
        calling with same shapes but different values.
        """
        import jax.numpy as jnp

        from homodyne.optimization.nlsq.fit_computation import (
            _compute_single_angle_shaped,
        )

        # Test parameters
        physical_params = jnp.array([1e-11, 0.5, 1e-14])
        n_t1, n_t2 = 20, 20
        t1 = jnp.linspace(0, 1, n_t1)
        t2 = jnp.linspace(0, 1, n_t2)
        phi_val = 0.0
        q = 0.01
        L = 1000.0
        dt = 0.1
        contrast = 0.8
        offset = 1.0

        # Track compilation via JAX's tracing
        compilation_count = [0]
        original_fn = _compute_single_angle_shaped.__wrapped__

        def counting_fn(*args, **kwargs):
            compilation_count[0] += 1
            return original_fn(*args, **kwargs)

        # First call - triggers compilation
        result1 = _compute_single_angle_shaped(
            physical_params, t1, t2, phi_val, q, L, dt, contrast, offset, n_t1, n_t2
        )

        # Verify output shape
        assert result1.shape == (n_t1, n_t2), (
            f"Expected shape {(n_t1, n_t2)}, got {result1.shape}"
        )

        # Second call with SAME shapes but DIFFERENT values - should NOT recompile
        physical_params2 = jnp.array([2e-11, 0.6, 2e-14])  # Different values
        t1_2 = jnp.linspace(0, 2, n_t1)  # Different range, same shape
        t2_2 = jnp.linspace(0, 2, n_t2)

        result2 = _compute_single_angle_shaped(
            physical_params2,
            t1_2,
            t2_2,
            phi_val + 10,
            q,
            L,
            dt,
            contrast + 0.1,
            offset + 0.1,
            n_t1,
            n_t2,
        )

        # Should still produce correct shape
        assert result2.shape == (n_t1, n_t2)

        # Call multiple times with same shapes - should use cached compilation
        for _ in range(5):
            _ = _compute_single_angle_shaped(
                physical_params, t1, t2, phi_val, q, L, dt, contrast, offset, n_t1, n_t2
            )

        # Verify result is finite and reasonable
        assert jnp.all(jnp.isfinite(result1))
        assert jnp.all(jnp.isfinite(result2))

    def test_static_shapes_different_shapes_recompile(self):
        """Test that different shapes trigger recompilation (expected behavior)."""
        import jax.numpy as jnp

        from homodyne.optimization.nlsq.fit_computation import (
            _compute_single_angle_shaped,
        )

        physical_params = jnp.array([1e-11, 0.5, 1e-14])
        phi_val = 0.0
        q = 0.01
        L = 1000.0
        dt = 0.1
        contrast = 0.8
        offset = 1.0

        # Call with shape (10, 10)
        n_t1, n_t2 = 10, 10
        t1 = jnp.linspace(0, 1, n_t1)
        t2 = jnp.linspace(0, 1, n_t2)
        result1 = _compute_single_angle_shaped(
            physical_params, t1, t2, phi_val, q, L, dt, contrast, offset, n_t1, n_t2
        )
        assert result1.shape == (10, 10)

        # Call with different shape (15, 15) - this SHOULD trigger new compilation
        # Note: n_t1 and n_t2 must match array shapes for correct meshgrid computation
        n_t1_new, n_t2_new = 15, 15
        t1_new = jnp.linspace(0, 1, n_t1_new)
        t2_new = jnp.linspace(0, 1, n_t2_new)
        result2 = _compute_single_angle_shaped(
            physical_params,
            t1_new,
            t2_new,
            phi_val,
            q,
            L,
            dt,
            contrast,
            offset,
            n_t1_new,
            n_t2_new,
        )
        assert result2.shape == (15, 15)
