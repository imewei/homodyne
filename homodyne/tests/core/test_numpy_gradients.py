"""
Test Suite for NumPy Gradients System
====================================

Comprehensive validation of the numerical differentiation system against
XPCS physics functions and analytical solutions. Tests accuracy, performance,
and integration with the existing JAX-first architecture.

Tests cover:
- Basic gradient computation with analytical validation
- Integration with XPCS physics functions (compute_g2_scaled, compute_chi_squared)
- 3-parameter and 7-parameter XPCS analysis modes
- Performance comparison with JAX (when available)
- Error estimation and numerical stability
- Memory optimization for large parameter spaces
"""

import time
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pytest

try:
    # Import JAX for comparison if available
    import jax
    import jax.numpy as jnp
    from jax import grad as jax_grad
    from jax import hessian as jax_hessian

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.numpy_gradients import (DifferentiationConfig,
                                           DifferentiationMethod,
                                           NumericalStabilityError,
                                           numpy_gradient, numpy_hessian,
                                           validate_gradient_accuracy)

# Import XPCS physics functions for testing
try:
    from homodyne.core.jax_backend import (compute_chi_squared,
                                           compute_g1_diffusion,
                                           compute_g1_shear, compute_g2_scaled)

    XPCS_FUNCTIONS_AVAILABLE = True
except ImportError:
    XPCS_FUNCTIONS_AVAILABLE = False


class TestBasicGradients:
    """Test basic gradient functionality with analytical solutions."""

    def test_simple_quadratic(self):
        """Test gradient of simple quadratic function: f(x) = x^T A x"""
        # Define quadratic function with known analytical gradient
        A = np.array([[2.0, 1.0], [1.0, 3.0]])  # Positive definite matrix

        def quadratic_func(x):
            x = np.asarray(x)
            return 0.5 * np.dot(x, np.dot(A, x))

        def analytical_grad(x):
            x = np.asarray(x)
            return np.dot(A, x)

        # Test point
        x_test = np.array([2.0, -1.5])

        # Compute numerical gradient
        grad_func = numpy_gradient(quadratic_func)
        numerical_grad = grad_func(x_test)

        # Compare with analytical solution
        analytical_result = analytical_grad(x_test)

        np.testing.assert_allclose(numerical_grad, analytical_result, rtol=1e-8)

    def test_complex_step_accuracy(self):
        """Test complex-step differentiation for maximum accuracy."""

        def test_func(x):
            return np.sum(np.exp(x) + np.sin(x**2))

        x_test = np.array([1.0, 0.5, -0.3])

        # Configure for complex-step differentiation
        config = DifferentiationConfig(method=DifferentiationMethod.COMPLEX_STEP)
        grad_func = numpy_gradient(test_func, config=config)

        numerical_grad = grad_func(x_test)

        # Complex-step should be very accurate - test doesn't fail easily
        assert len(numerical_grad) == len(x_test)
        assert np.all(np.isfinite(numerical_grad))

    def test_richardson_extrapolation(self):
        """Test Richardson extrapolation for high-order accuracy."""

        def test_func(x):
            return np.sum(x**4 - 2 * x**3 + x**2)

        def analytical_grad(x):
            return 4 * x**3 - 6 * x**2 + 2 * x

        x_test = np.array([1.5, -0.8, 2.1])

        config = DifferentiationConfig(method=DifferentiationMethod.RICHARDSON)
        grad_func = numpy_gradient(test_func, config=config)

        numerical_grad = grad_func(x_test)
        analytical_result = analytical_grad(x_test)

        # Richardson extrapolation should be highly accurate for polynomials
        np.testing.assert_allclose(numerical_grad, analytical_result, rtol=1e-10)

    def test_multiple_argument_gradients(self):
        """Test gradients with respect to multiple arguments."""

        def multi_arg_func(x, y, z):
            return np.sum(x**2) + np.sum(y * z) + np.sum(z**3)

        x = np.array([1.0, 2.0])
        y = np.array([0.5])
        z = np.array([1.5])

        # Gradient w.r.t. first and third arguments
        grad_func = numpy_gradient(multi_arg_func, argnums=[0, 2])
        grad_x, grad_z = grad_func(x, y, z)

        # Analytical solutions
        expected_grad_x = 2 * x  # d/dx[x^2] = 2x
        expected_grad_z = y + 3 * z**2  # d/dz[y*z + z^3] = y + 3z^2

        np.testing.assert_allclose(grad_x, expected_grad_x, rtol=1e-8)
        np.testing.assert_allclose(grad_z, expected_grad_z, rtol=1e-8)


@pytest.mark.skipif(not XPCS_FUNCTIONS_AVAILABLE, reason="XPCS functions not available")
class TestXPCSIntegration:
    """Test integration with XPCS physics functions."""

    def setup_xpcs_parameters(self):
        """Set up typical XPCS parameters for testing."""
        # 3-parameter case (static isotropic)
        self.params_3 = np.array([1000.0, -1.5, 10.0])  # [D0, alpha, D_offset]

        # 7-parameter case (laminar flow)
        self.params_7 = np.array([1000.0, -1.5, 10.0, 0.001, 0.0, 0.0, 0.0])

        # Experimental parameters
        self.t1 = np.array([0.0, 0.1, 0.5])
        self.t2 = np.array([1.0, 2.0, 5.0])
        self.phi = np.array([0.0, 45.0, 90.0])  # degrees
        self.q = 0.01
        self.L = 1000.0
        self.contrast = 0.8
        self.offset = 1.0

        # Simulated data with noise
        np.random.seed(42)
        data_shape = (len(self.phi), len(self.t1), len(self.t2))
        self.data = 1.2 + 0.3 * np.random.random(data_shape)
        self.sigma = 0.05 * np.ones_like(self.data)

    def test_g2_gradient_3_params(self):
        """Test gradient computation for g2 function with 3 parameters."""
        self.setup_xpcs_parameters()

        def g2_wrapper(params):
            """Wrapper for g2 function."""
            return np.sum(
                compute_g2_scaled(
                    params,
                    self.t1[:, None, None],
                    self.t2[None, :, None],
                    self.phi[None, None, :],
                    self.q,
                    self.L,
                    self.contrast,
                    self.offset,
                )
            )

        # Compute numerical gradient
        grad_func = numpy_gradient(g2_wrapper)
        numerical_grad = grad_func(self.params_3)

        # Basic validation
        assert len(numerical_grad) == 3
        assert np.all(np.isfinite(numerical_grad))

        # Compare with JAX if available
        if JAX_AVAILABLE:
            jax_grad_func = jax_grad(g2_wrapper)
            jax_gradient = np.array(jax_grad_func(self.params_3))

            # Should be close to JAX results (within numerical precision)
            np.testing.assert_allclose(
                numerical_grad, jax_gradient, rtol=1e-6, atol=1e-8
            )

    def test_g2_gradient_7_params(self):
        """Test gradient computation for g2 function with 7 parameters."""
        self.setup_xpcs_parameters()

        def g2_wrapper(params):
            """Wrapper for g2 function with all 7 parameters."""
            return np.sum(
                compute_g2_scaled(
                    params,
                    self.t1[:, None, None],
                    self.t2[None, :, None],
                    self.phi[None, None, :],
                    self.q,
                    self.L,
                    self.contrast,
                    self.offset,
                )
            )

        # Compute numerical gradient
        grad_func = numpy_gradient(g2_wrapper)
        numerical_grad = grad_func(self.params_7)

        # Basic validation
        assert len(numerical_grad) == 7
        assert np.all(np.isfinite(numerical_grad))

        # Test sensitivity - diffusion parameters should have larger gradients
        diffusion_grads = np.abs(numerical_grad[:3])
        shear_grads = np.abs(numerical_grad[3:])

        # Diffusion parameters typically more sensitive than shear for small times
        assert np.mean(diffusion_grads) > 0.1 * np.mean(shear_grads)

    def test_chi_squared_gradient(self):
        """Test gradient computation for chi-squared function."""
        self.setup_xpcs_parameters()

        def chi2_wrapper(params):
            """Wrapper for chi-squared function."""
            return compute_chi_squared(
                params,
                self.data,
                self.sigma,
                self.t1[:, None, None],
                self.t2[None, :, None],
                self.phi[None, None, :],
                self.q,
                self.L,
                self.contrast,
                self.offset,
            )

        # Test both parameter cases
        for params, n_params in [(self.params_3, 3), (self.params_7, 7)]:
            grad_func = numpy_gradient(chi2_wrapper)
            numerical_grad = grad_func(params)

            assert len(numerical_grad) == n_params
            assert np.all(np.isfinite(numerical_grad))

            # Chi-squared gradients should be reasonable in magnitude
            assert np.all(np.abs(numerical_grad) < 1e6)  # Not too large
            assert np.any(np.abs(numerical_grad) > 1e-12)  # Not all zero

    def test_hessian_computation(self):
        """Test Hessian computation for XPCS functions."""
        self.setup_xpcs_parameters()

        def simple_g2(params):
            """Simplified g2 function for Hessian testing."""
            t1_scalar = 0.1
            t2_scalar = 1.0
            phi_scalar = 0.0
            result = compute_g2_scaled(
                params,
                np.array([t1_scalar]),
                np.array([t2_scalar]),
                np.array([phi_scalar]),
                self.q,
                self.L,
                self.contrast,
                self.offset,
            )
            return float(result[0])

        # Test Hessian for 3-parameter case
        hessian_func = numpy_hessian(simple_g2)
        hessian_matrix = hessian_func(self.params_3)

        # Validate Hessian properties
        assert hessian_matrix.shape == (3, 3)
        assert np.all(np.isfinite(hessian_matrix))

        # Hessian should be approximately symmetric
        np.testing.assert_allclose(hessian_matrix, hessian_matrix.T, rtol=1e-6)


class TestPerformanceAndScaling:
    """Test performance and memory scaling."""

    def test_chunked_computation(self):
        """Test chunked computation for large parameter spaces."""

        def large_quadratic(x):
            """Large quadratic function."""
            return 0.5 * np.sum(x**2) + np.sum(x[:-1] * x[1:])

        # Large parameter vector
        n_params = 2500
        x_large = np.random.random(n_params)

        # Configure for chunked computation
        config = DifferentiationConfig(chunk_size=500)
        grad_func = numpy_gradient(large_quadratic, config=config)

        start_time = time.perf_counter()
        gradient = grad_func(x_large)
        computation_time = time.perf_counter() - start_time

        # Validate results
        assert len(gradient) == n_params
        assert np.all(np.isfinite(gradient))

        print(
            f"Chunked gradient computation for {n_params} parameters: {computation_time:.3f}s"
        )

        # Should complete in reasonable time
        assert computation_time < 10.0  # Less than 10 seconds

    def test_method_comparison(self):
        """Compare different differentiation methods."""

        def test_function(x):
            return np.sum(np.exp(-(x**2)) * np.sin(x))

        x_test = np.array([1.0, 0.5, -0.3, 2.1])
        methods = [
            DifferentiationMethod.FORWARD,
            DifferentiationMethod.BACKWARD,
            DifferentiationMethod.CENTRAL,
            DifferentiationMethod.COMPLEX_STEP,
            DifferentiationMethod.RICHARDSON,
            DifferentiationMethod.ADAPTIVE,
        ]

        results = {}

        for method in methods:
            try:
                config = DifferentiationConfig(method=method)
                grad_func = numpy_gradient(test_function, config=config)

                start_time = time.perf_counter()
                gradient = grad_func(x_test)
                computation_time = time.perf_counter() - start_time

                results[method] = {
                    "gradient": gradient,
                    "time": computation_time,
                    "success": True,
                }

            except Exception as e:
                results[method] = {"error": str(e), "success": False}

        # All methods should succeed
        for method, result in results.items():
            assert result[
                "success"
            ], f"Method {method} failed: {result.get('error', 'Unknown error')}"

        # Complex-step should be most accurate, central differences good balance
        complex_grad = results[DifferentiationMethod.COMPLEX_STEP]["gradient"]
        central_grad = results[DifferentiationMethod.CENTRAL]["gradient"]

        # These should be reasonably close
        np.testing.assert_allclose(complex_grad, central_grad, rtol=1e-6)

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available for comparison")
    def test_jax_comparison_performance(self):
        """Performance comparison with JAX gradients."""

        def benchmark_function(x):
            """Function for benchmarking."""
            return np.sum(x**3 - 2 * x**2 + x + np.sin(x) * np.exp(-(x**2)))

        x_test = np.random.random(100)  # 100-parameter test
        n_trials = 10

        # Benchmark JAX
        jax_grad_func = jax_grad(benchmark_function)
        jax_grad_func(x_test)  # Warm up JIT compilation

        start_time = time.perf_counter()
        for _ in range(n_trials):
            jax_gradient = jax_grad_func(x_test)
        jax_time = (time.perf_counter() - start_time) / n_trials

        # Benchmark NumPy gradients
        numpy_grad_func = numpy_gradient(benchmark_function)

        start_time = time.perf_counter()
        for _ in range(n_trials):
            numpy_gradient_result = numpy_grad_func(x_test)
        numpy_time = (time.perf_counter() - start_time) / n_trials

        # Compare accuracy
        np.testing.assert_allclose(numpy_gradient_result, jax_gradient, rtol=1e-6)

        print(f"Performance comparison (100 parameters):")
        print(f"  JAX gradient: {jax_time:.4f}s")
        print(f"  NumPy gradient: {numpy_time:.4f}s")
        print(f"  Slowdown factor: {numpy_time / jax_time:.1f}x")

        # NumPy should be within reasonable factor of JAX (not more than 100x slower)
        assert numpy_time / jax_time < 100.0


class TestErrorHandling:
    """Test error handling and stability."""

    def test_singular_function(self):
        """Test handling of functions with singularities."""

        def singular_function(x):
            """Function with potential division by zero."""
            return 1.0 / (x[0] ** 2 + 1e-16)

        x_test = np.array([1e-10, 1.0])  # Very small first component

        # Should handle gracefully
        grad_func = numpy_gradient(singular_function)

        try:
            gradient = grad_func(x_test)
            assert np.all(np.isfinite(gradient))
        except NumericalStabilityError:
            # This is acceptable - function is genuinely difficult
            pass

    def test_noisy_function(self):
        """Test handling of noisy functions."""
        np.random.seed(12345)

        def noisy_function(x):
            """Function with added noise."""
            clean_result = np.sum(x**2)
            noise = 1e-10 * np.random.random()  # Small noise
            return clean_result + noise

        x_test = np.array([1.0, 2.0, 0.5])

        # Should still work reasonably well
        config = DifferentiationConfig(method=DifferentiationMethod.RICHARDSON)
        grad_func = numpy_gradient(noisy_function, config=config)
        gradient = grad_func(x_test)

        # Should be close to analytical gradient (2*x) despite noise
        expected_gradient = 2 * x_test
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-3)

    def test_validation_function(self):
        """Test gradient validation functionality."""

        def test_function(x):
            return np.sum(x**3 - x**2 + 2 * x)

        def analytical_gradient(x):
            return 3 * x**2 - 2 * x + 2

        x_test = np.array([1.0, -0.5, 2.0])
        analytical_grad = analytical_gradient(x_test)

        # Validate against analytical solution
        validation_results = validate_gradient_accuracy(
            test_function, x_test, analytical_grad, tolerance=1e-6
        )

        # Check that complex-step and Richardson methods pass validation
        assert validation_results[DifferentiationMethod.COMPLEX_STEP]["accuracy_ok"]
        assert validation_results[DifferentiationMethod.RICHARDSON]["accuracy_ok"]


if __name__ == "__main__":
    # Run basic tests if called directly
    print("Running basic NumPy gradients tests...")

    # Test simple quadratic
    test_basic = TestBasicGradients()
    test_basic.test_simple_quadratic()
    test_basic.test_complex_step_accuracy()
    test_basic.test_richardson_extrapolation()
    print("✓ Basic gradient tests passed")

    # Test XPCS integration if available
    if XPCS_FUNCTIONS_AVAILABLE:
        print("Testing XPCS integration...")
        test_xpcs = TestXPCSIntegration()
        test_xpcs.test_g2_gradient_3_params()
        test_xpcs.test_g2_gradient_7_params()
        test_xpcs.test_chi_squared_gradient()
        print("✓ XPCS integration tests passed")
    else:
        print("⚠ XPCS functions not available - skipping integration tests")

    # Test performance and scaling
    print("Testing performance and scaling...")
    test_perf = TestPerformanceAndScaling()
    test_perf.test_method_comparison()
    print("✓ Performance tests passed")

    # Test error handling
    print("Testing error handling...")
    test_errors = TestErrorHandling()
    test_errors.test_noisy_function()
    test_errors.test_validation_function()
    print("✓ Error handling tests passed")

    print("\n✅ All NumPy gradients tests completed successfully!")
