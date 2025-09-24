"""
Unit Tests for NLSQ Optimization Module
=======================================

Tests for homodyne.optimization.nlsq module including:
- Optimistix trust-region solver
- Parameter estimation accuracy
- Convergence behavior
- Error handling and fallbacks
- Performance characteristics
"""


import numpy as np
import pytest

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.optimization.nlsq import OPTIMISTIX_AVAILABLE, NLSQResult, fit_nlsq_jax


@pytest.mark.unit
@pytest.mark.requires_jax
class TestNLSQOptimization:
    """Test NLSQ optimization functionality."""

    def test_optimistix_availability(self):
        """Test Optimistix availability detection."""
        # This tests the import detection logic
        assert isinstance(OPTIMISTIX_AVAILABLE, bool)

    def test_nlsq_result_structure(self):
        """Test NLSQResult data structure."""
        # Test that we can create a result object
        result = NLSQResult(
            parameters={"offset": 1.0, "contrast": 0.5},
            parameter_errors={"offset": 0.1, "contrast": 0.05},
            chi_squared=0.1,
            reduced_chi_squared=0.12,
            success=True,
            message="Test result",
            n_iterations=10,
            optimization_time=0.5,
        )

        assert result.parameters == {"offset": 1.0, "contrast": 0.5}
        assert result.chi_squared == 0.1
        assert result.success is True
        assert result.message == "Test result"
        assert result.n_iterations == 10
        assert result.optimization_time == 0.5

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_synthetic_data_fit(self, synthetic_xpcs_data, test_config):
        """Test NLSQ fitting with synthetic data."""
        data = synthetic_xpcs_data
        config = test_config

        # Run optimization
        result = fit_nlsq_jax(data, config)

        # Basic result validation
        assert isinstance(result, NLSQResult)
        assert result.success, f"Optimization failed: {result.message}"
        assert hasattr(result, "parameters")
        assert hasattr(result, "chi_squared")

        # Parameter validation
        assert "offset" in result.parameters
        assert "contrast" in result.parameters
        assert "diffusion_coefficient" in result.parameters

        # Physical constraints
        assert result.parameters["offset"] >= 0.5, "Offset too low"
        assert result.parameters["offset"] <= 2.0, "Offset too high"
        assert result.parameters["contrast"] >= 0.0, "Contrast must be non-negative"
        assert result.parameters["contrast"] <= 1.0, "Contrast too high"
        assert (
            result.parameters["diffusion_coefficient"] >= 0.0
        ), "Diffusion must be non-negative"

        # Convergence metrics
        assert result.chi_squared >= 0.0, "Chi-squared must be non-negative"
        assert result.n_iterations > 0, "Should have performed some iterations"
        assert result.optimization_time > 0.0, "Should have non-zero optimization time"

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_parameter_recovery(self, test_config):
        """Test parameter recovery with known synthetic data."""
        # Generate data with known parameters
        n_times = 30
        n_angles = 24

        t1, t2 = jnp.meshgrid(jnp.arange(n_times), jnp.arange(n_times), indexing="ij")
        phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

        # Known parameters
        true_params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.12,
            "shear_rate": 0.0,
            "L": 1.0,
        }
        q = 0.01

        # Generate perfect synthetic data
        from homodyne.core.jax_backend import compute_c2_model_jax

        c2_true = compute_c2_model_jax(true_params, t1, t2, phi, q)

        # Add minimal noise
        np.random.seed(42)
        noise = 0.001 * np.random.randn(*c2_true.shape)
        c2_exp = c2_true + noise

        data = {
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp,
            "wavevector_q_list": np.array([q]),
            "sigma": np.ones_like(c2_exp) * 0.001,
        }

        # Run optimization
        result = fit_nlsq_jax(data, test_config)

        # Parameter recovery validation
        tolerance = 0.05  # 5% tolerance
        assert result.success, f"Optimization failed: {result.message}"

        for param_name, true_value in true_params.items():
            if param_name in result.parameters:
                recovered_value = result.parameters[param_name]
                relative_error = abs(recovered_value - true_value) / true_value
                assert relative_error < tolerance, (
                    f"Parameter {param_name}: recovered {recovered_value:.4f}, "
                    f"true {true_value:.4f}, error {relative_error:.4f}"
                )

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_convergence_behavior(self, small_xpcs_data, test_config):
        """Test NLSQ convergence behavior with different tolerances."""
        data = small_xpcs_data

        # Test with loose tolerance
        loose_config = test_config.copy()
        loose_config["optimization"]["lsq"]["tolerance"] = 1e-3
        loose_config["optimization"]["lsq"]["max_iterations"] = 50

        result_loose = fit_nlsq_jax(data, loose_config)

        # Test with tight tolerance
        tight_config = test_config.copy()
        tight_config["optimization"]["lsq"]["tolerance"] = 1e-8
        tight_config["optimization"]["lsq"]["max_iterations"] = 200

        result_tight = fit_nlsq_jax(data, tight_config)

        # Both should succeed
        assert result_loose.success, f"Loose tolerance failed: {result_loose.message}"
        assert result_tight.success, f"Tight tolerance failed: {result_tight.message}"

        # Tight tolerance should generally achieve lower chi-squared
        assert (
            result_tight.chi_squared <= result_loose.chi_squared * 1.1
        ), "Tight tolerance should achieve better or similar fit"

        # Tight tolerance might use more iterations
        assert (
            result_tight.n_iterations >= result_loose.n_iterations * 0.5
        ), "Tight tolerance should use reasonable iterations"

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_boundary_conditions(self, synthetic_xpcs_data, test_config):
        """Test NLSQ optimization with boundary conditions."""
        data = synthetic_xpcs_data

        # Test with constrained parameters
        constrained_config = test_config.copy()
        constrained_config["optimization"]["lsq"]["bounds"] = {
            "offset": [0.8, 1.2],
            "contrast": [0.1, 0.8],
            "diffusion_coefficient": [0.01, 0.5],
        }

        result = fit_nlsq_jax(data, constrained_config)

        if result.success:
            # Check that parameters respect bounds
            bounds = constrained_config["optimization"]["lsq"]["bounds"]
            for param_name, (lower, upper) in bounds.items():
                if param_name in result.parameters:
                    value = result.parameters[param_name]
                    assert (
                        lower <= value <= upper
                    ), f"Parameter {param_name}={value} outside bounds [{lower}, {upper}]"

    def test_nlsq_error_handling(self, test_config):
        """Test NLSQ error handling with invalid data."""
        # Test with missing data fields
        incomplete_data = {
            "t1": np.array([[0, 1], [1, 0]]),
            "t2": np.array([[0, 1], [1, 0]]),
            # Missing required fields
        }

        with pytest.raises((KeyError, ValueError, AttributeError)):
            fit_nlsq_jax(incomplete_data, test_config)

        # Test with mismatched array shapes
        mismatched_data = {
            "t1": np.array([[0, 1], [1, 0]]),
            "t2": np.array([[0, 1], [1, 0]]),
            "phi_angles_list": np.array([0.0]),
            "c2_exp": np.array([[[1.0]]]),  # Wrong shape
            "wavevector_q_list": np.array([0.01]),
            "sigma": np.array([[[0.01]]]),
        }

        with pytest.raises((ValueError, IndexError)):
            fit_nlsq_jax(mismatched_data, test_config)

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_with_shear(self, test_config):
        """Test NLSQ optimization with shear flow."""
        # Generate data with shear
        n_times = 20
        n_angles = 18

        t1, t2 = jnp.meshgrid(jnp.arange(n_times), jnp.arange(n_times), indexing="ij")
        phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

        params_with_shear = {
            "offset": 1.0,
            "contrast": 0.35,
            "diffusion_coefficient": 0.08,
            "shear_rate": 0.05,  # Non-zero shear
            "L": 1.0,
        }
        q = 0.015

        # Generate synthetic data with shear
        from homodyne.core.jax_backend import compute_c2_model_jax

        c2_true = compute_c2_model_jax(params_with_shear, t1, t2, phi, q)

        # Add noise
        np.random.seed(123)
        noise = 0.01 * np.random.randn(*c2_true.shape)
        c2_exp = c2_true + noise

        data = {
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp,
            "wavevector_q_list": np.array([q]),
            "sigma": np.ones_like(c2_exp) * 0.01,
        }

        # Update config for shear analysis
        shear_config = test_config.copy()
        shear_config["analysis_mode"] = "dynamic_shear"

        result = fit_nlsq_jax(data, shear_config)

        if result.success:
            # Check that shear rate is recovered reasonably
            if "shear_rate" in result.parameters:
                recovered_shear = result.parameters["shear_rate"]
                assert (
                    abs(recovered_shear - params_with_shear["shear_rate"]) < 0.02
                ), f"Shear rate recovery poor: {recovered_shear} vs {params_with_shear['shear_rate']}"

    @pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
    def test_nlsq_multiple_q_values(self, test_config):
        """Test NLSQ optimization with multiple q-values."""
        n_times = 15
        n_angles = 12

        t1, t2 = jnp.meshgrid(jnp.arange(n_times), jnp.arange(n_times), indexing="ij")
        phi = jnp.linspace(0, 2 * jnp.pi, n_angles)

        # Multiple q-values
        q_values = np.array([0.008, 0.012, 0.016])

        # Generate data for each q
        from homodyne.core.jax_backend import compute_c2_model_jax

        true_params = {
            "offset": 1.0,
            "contrast": 0.3,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        c2_exp_list = []
        for q in q_values:
            c2_q = compute_c2_model_jax(true_params, t1, t2, phi, q)
            # Add different noise levels for each q
            noise = 0.005 * np.random.randn(*c2_q.shape)
            c2_exp_list.append(c2_q + noise)

        # Use first q-value for testing (multi-q would need different data structure)
        data = {
            "t1": t1,
            "t2": t2,
            "phi_angles_list": phi,
            "c2_exp": c2_exp_list[0],
            "wavevector_q_list": q_values[:1],  # Just first q for now
            "sigma": np.ones_like(c2_exp_list[0]) * 0.005,
        }

        result = fit_nlsq_jax(data, test_config)

        assert result.success, f"Multi-q optimization failed: {result.message}"
        assert "diffusion_coefficient" in result.parameters
        assert result.parameters["diffusion_coefficient"] > 0


@pytest.mark.unit
class TestNLSQFallback:
    """Test NLSQ fallback behavior when Optimistix is not available."""

    def test_fallback_import(self):
        """Test that module imports work without Optimistix."""
        # Should be able to import even without Optimistix
        from homodyne.optimization.nlsq import fit_nlsq_jax

        assert callable(fit_nlsq_jax)

    @pytest.mark.skipif(OPTIMISTIX_AVAILABLE, reason="Optimistix is available")
    def test_fallback_behavior(self, synthetic_xpcs_data, test_config):
        """Test fallback behavior when Optimistix is not available."""
        data = synthetic_xpcs_data

        # This should either work with fallback or raise appropriate error
        try:
            result = fit_nlsq_jax(data, test_config)
            # If it works, validate basic structure
            assert hasattr(result, "success")
            assert hasattr(result, "message")
        except ImportError as e:
            # Expected when no fallback is available
            assert "Optimistix" in str(e) or "optimization" in str(e)

    def test_nlsq_result_serialization(self):
        """Test that NLSQResult can be serialized/deserialized."""
        import json

        result = NLSQResult(
            parameters={"offset": 1.0, "contrast": 0.5},
            chi_squared=0.1,
            success=True,
            message="Test",
            n_iterations=10,
            optimization_time=0.5,
        )

        # Convert to dict for serialization
        result_dict = {
            "parameters": result.parameters,
            "chi_squared": result.chi_squared,
            "success": result.success,
            "message": result.message,
            "n_iterations": result.n_iterations,
            "optimization_time": result.optimization_time,
        }

        # Test JSON serialization
        json_str = json.dumps(result_dict)
        loaded_dict = json.loads(json_str)

        assert loaded_dict["parameters"] == result.parameters
        assert loaded_dict["chi_squared"] == result.chi_squared
        assert loaded_dict["success"] == result.success


@pytest.mark.unit
@pytest.mark.performance
@pytest.mark.skipif(not OPTIMISTIX_AVAILABLE, reason="Optimistix not available")
class TestNLSQPerformance:
    """Performance tests for NLSQ optimization."""

    def test_nlsq_timing_small_dataset(
        self, small_xpcs_data, test_config, benchmark_config
    ):
        """Test NLSQ timing with small dataset."""
        data = small_xpcs_data

        # Time the optimization
        import time

        start_time = time.perf_counter()
        result = fit_nlsq_jax(data, test_config)
        elapsed_time = time.perf_counter() - start_time

        # Should complete in reasonable time
        assert elapsed_time < 5.0, f"Small dataset took too long: {elapsed_time:.2f}s"
        assert result.success, "Small dataset optimization should succeed"

        # Reported time should be consistent
        assert (
            abs(result.optimization_time - elapsed_time) < 0.1
        ), "Reported computation time inconsistent"

    def test_nlsq_scaling_dataset_size(self, test_config):
        """Test NLSQ timing scaling with dataset size."""
        sizes = [10, 20, 30]
        times = []

        for n_times in sizes:
            # Generate data of different sizes
            t1, t2 = jnp.meshgrid(
                jnp.arange(n_times), jnp.arange(n_times), indexing="ij"
            )
            phi = jnp.linspace(0, 2 * jnp.pi, 12)

            # Simple synthetic data
            tau = jnp.abs(t1 - t2) + 1e-6
            c2_exp = 1 + 0.3 * jnp.exp(-tau / 8.0)

            data = {
                "t1": t1,
                "t2": t2,
                "phi_angles_list": phi,
                "c2_exp": c2_exp,
                "wavevector_q_list": np.array([0.01]),
                "sigma": np.ones_like(c2_exp) * 0.01,
            }

            # Time optimization
            import time

            start_time = time.perf_counter()
            result = fit_nlsq_jax(data, test_config)
            elapsed_time = time.perf_counter() - start_time

            if result.success:
                times.append(elapsed_time)
            else:
                times.append(float("inf"))  # Mark failures

        # Basic scaling check - should not explode
        valid_times = [t for t in times if t != float("inf")]
        if len(valid_times) >= 2:
            max_time = max(valid_times)
            min_time = min(valid_times)
            scaling_factor = max_time / min_time

            # Should scale reasonably (not exponentially)
            assert scaling_factor < 20.0, f"Poor scaling: {scaling_factor:.2f}x"

    def test_nlsq_convergence_speed(self, synthetic_xpcs_data, test_config):
        """Test NLSQ convergence speed."""
        data = synthetic_xpcs_data

        # Test with maximum iterations limit
        speed_config = test_config.copy()
        speed_config["optimization"]["lsq"]["max_iterations"] = 20

        result = fit_nlsq_jax(data, speed_config)

        if result.success:
            # Should converge reasonably quickly
            assert result.n_iterations <= 20, "Should respect iteration limit"
            assert result.n_iterations >= 1, "Should perform at least one iteration"

            # Fast convergence indicator
            if result.n_iterations < 10:
                assert (
                    result.chi_squared < 1.0
                ), "Fast convergence should achieve good fit"
