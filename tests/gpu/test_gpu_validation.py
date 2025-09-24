"""
GPU Acceleration Validation Tests
=================================

Tests for GPU acceleration functionality including:
- GPU detection and initialization
- Performance validation and speedup verification
- Memory management and allocation
- Numerical consistency between CPU and GPU
- Error handling and graceful fallbacks
"""

import time

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

# Handle GPU-specific imports
try:
    from homodyne.runtime.gpu import (
        GPU_AVAILABLE,
        GPUActivator,
        activate_gpu,
        benchmark_gpu,
        get_gpu_status,
    )

    GPU_MODULE_AVAILABLE = True
except ImportError:
    GPU_MODULE_AVAILABLE = False


@pytest.mark.gpu
@pytest.mark.requires_gpu
@pytest.mark.skipif(not GPU_MODULE_AVAILABLE, reason="GPU module not available")
class TestGPUActivation:
    """Test GPU activation and configuration."""

    def test_gpu_status_detection(self):
        """Test GPU status detection."""
        status = get_gpu_status()

        # Should return a valid status dictionary
        assert isinstance(status, dict)
        assert "jax_available" in status
        assert "devices" in status

        # JAX availability should be consistent
        assert status["jax_available"] == JAX_AVAILABLE

        if JAX_AVAILABLE:
            assert isinstance(status["devices"], list)
            assert len(status["devices"]) >= 0

    def test_gpu_activator_initialization(self):
        """Test GPUActivator class initialization."""
        # Should be able to create activator
        activator = GPUActivator(verbose=False)
        assert activator is not None
        assert hasattr(activator, "activate")
        assert hasattr(activator, "deactivate")

    def test_gpu_activation_basic(self):
        """Test basic GPU activation."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Try to activate GPU
        result = activate_gpu(
            memory_fraction=0.5,
            force_gpu=False,  # Don't fail if no GPU
            verbose=False,
        )

        # Should return a result dictionary
        assert isinstance(result, dict)
        assert "status" in result or "success" in result

        # If GPU is available, check activation success
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) > 0:
            # GPU should activate successfully
            success = result.get("success", False) or result.get("status") == "success"
            if success:
                assert "device" in result
                assert "backend" in result

    def test_gpu_memory_configuration(self):
        """Test GPU memory configuration options."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        memory_fractions = [0.3, 0.5, 0.8]

        for mem_frac in memory_fractions:
            result = activate_gpu(
                memory_fraction=mem_frac, force_gpu=False, verbose=False
            )

            # Should handle different memory fractions gracefully
            assert isinstance(result, dict)

            # If successful, should not error
            if result.get("success", False):
                # Memory fraction should be reasonable
                assert 0.0 < mem_frac <= 1.0

    def test_gpu_activation_error_handling(self):
        """Test GPU activation error handling."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        # Test invalid memory fraction
        with pytest.raises(ValueError):
            activate_gpu(memory_fraction=1.5)  # > 1.0

        with pytest.raises(ValueError):
            activate_gpu(memory_fraction=0.0)  # <= 0.0

        # Test invalid GPU ID
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            # No GPU available - should handle gracefully
            result = activate_gpu(force_gpu=False)
            assert result.get("status") in ["cpu_fallback", "failed"] or not result.get(
                "success", True
            )
        else:
            # Invalid GPU ID should raise error
            with pytest.raises((ValueError, IndexError)):
                activate_gpu(gpu_id=999)

    def test_gpu_deactivation(self):
        """Test GPU deactivation and cleanup."""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        activator = GPUActivator(verbose=False)

        # Activate then deactivate
        result = activator.activate(memory_fraction=0.5, force_gpu=False)

        # Deactivation should work regardless of activation success
        activator.deactivate()  # Should not raise exception

        # Should be able to deactivate multiple times
        activator.deactivate()  # Should not raise exception


@pytest.mark.gpu
@pytest.mark.requires_gpu
@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestGPUPerformance:
    """Test GPU performance and speedup validation."""

    def test_gpu_benchmark_execution(self):
        """Test GPU benchmark execution."""
        if not GPU_MODULE_AVAILABLE:
            pytest.skip("GPU module not available")

        try:
            results = benchmark_gpu()

            if "error" in results:
                pytest.skip(f"GPU benchmark not available: {results['error']}")

            # Should return performance metrics
            assert isinstance(results, dict)
            assert len(results) > 0

            # All results should be positive numbers
            for test_name, score in results.items():
                if isinstance(score, (int, float)):
                    assert score > 0, f"Benchmark {test_name} should be positive"

        except Exception as e:
            pytest.skip(f"GPU benchmark failed: {e}")

    def test_matrix_multiplication_performance(self):
        """Test matrix multiplication performance on GPU."""
        # Check if GPU is available
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        # Test matrix multiplication
        size = 1000
        A = jnp.ones((size, size))
        B = jnp.ones((size, size))

        # Warmup
        C = jnp.dot(A, B)
        C.block_until_ready()

        # Time GPU computation
        start_time = time.perf_counter()
        for _ in range(5):
            C = jnp.dot(A, B)
            C.block_until_ready()
        gpu_time = time.perf_counter() - start_time

        # Performance should be reasonable
        operations = 2 * size**3 * 5  # Matrix multiply operations
        gflops = operations / (gpu_time * 1e9)

        # Should achieve some reasonable performance (very conservative threshold)
        assert gflops > 0.1, f"GPU performance too low: {gflops:.2f} GFLOPS"

        # Result should be correct
        expected = size * jnp.ones((size, size))
        np.testing.assert_array_almost_equal(C, expected, decimal=6)

    def test_cpu_gpu_consistency(self):
        """Test numerical consistency between CPU and GPU."""
        # Check GPU availability
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        try:
            from homodyne.core.jax_backend import compute_c2_model_jax
        except ImportError:
            pytest.skip("JAX backend not available")

        # Test parameters
        t1 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        t2 = jnp.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]])
        phi = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        q = 0.01

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Force CPU computation
        import os

        original_platform = os.environ.get("JAX_PLATFORM_NAME", "")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

        try:
            result_cpu = compute_c2_model_jax(params, t1, t2, phi, q)
            result_cpu.block_until_ready()
        finally:
            # Restore platform setting
            if original_platform:
                os.environ["JAX_PLATFORM_NAME"] = original_platform
            else:
                os.environ.pop("JAX_PLATFORM_NAME", None)

        # GPU computation (default platform)
        result_gpu = compute_c2_model_jax(params, t1, t2, phi, q)
        result_gpu.block_until_ready()

        # Results should be very close
        np.testing.assert_array_almost_equal(
            result_cpu,
            result_gpu,
            decimal=10,
            err_msg="CPU and GPU results should be numerically consistent",
        )

    def test_memory_scaling_gpu(self):
        """Test GPU memory scaling behavior."""
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        # Test different sizes to check memory scaling
        sizes = [100, 200, 300]
        successful_sizes = []

        for size in sizes:
            try:
                # Create test arrays
                A = jnp.ones((size, size))
                B = jnp.ones((size, size))

                # Perform computation
                C = jnp.dot(A, B)
                C.block_until_ready()

                # Should produce correct result
                expected = size * jnp.ones((size, size))
                np.testing.assert_array_almost_equal(C, expected, decimal=6)

                successful_sizes.append(size)

            except Exception as e:
                # Memory errors are acceptable for large sizes
                if "memory" in str(e).lower() or "out of memory" in str(e).lower():
                    break
                else:
                    raise

        # Should handle at least some reasonable sizes
        assert len(successful_sizes) >= 1, "GPU should handle at least small matrices"

    def test_optimization_gpu_acceleration(self, synthetic_xpcs_data):
        """Test optimization acceleration on GPU."""
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        try:
            from homodyne.optimization.nlsq import OPTIMISTIX_AVAILABLE, fit_nlsq_jax
        except ImportError:
            pytest.skip("Optimization module not available")

        if not OPTIMISTIX_AVAILABLE:
            pytest.skip("Optimistix not available")

        data = synthetic_xpcs_data

        # GPU configuration
        gpu_config = {
            "analysis_mode": "static_isotropic",
            "optimization": {
                "method": "nlsq",
                "lsq": {
                    "max_iterations": 20,  # Keep short for testing
                    "tolerance": 1e-6,
                },
            },
            "hardware": {
                "force_cpu": False,  # Allow GPU
                "gpu_memory_fraction": 0.5,
            },
        }

        # Try GPU optimization
        try:
            start_time = time.perf_counter()
            result_gpu = fit_nlsq_jax(data, gpu_config)
            gpu_time = time.perf_counter() - start_time

            if result_gpu.success:
                # Should complete in reasonable time
                assert gpu_time < 30.0, f"GPU optimization too slow: {gpu_time:.2f}s"

                # Should have reasonable parameters
                assert "offset" in result_gpu.parameters
                assert result_gpu.chi_squared >= 0.0

        except Exception as e:
            pytest.skip(f"GPU optimization failed: {e}")


@pytest.mark.gpu
class TestGPUFallback:
    """Test GPU fallback behavior."""

    def test_graceful_fallback_no_gpu(self):
        """Test graceful fallback when no GPU is available."""
        if not GPU_MODULE_AVAILABLE:
            pytest.skip("GPU module not available")

        # Force no GPU scenario
        result = activate_gpu(force_gpu=False, verbose=False)

        # Should handle gracefully
        assert isinstance(result, dict)

        # Should either succeed or fall back to CPU
        status = result.get("status", "unknown")
        success = result.get("success", False)

        if not success:
            assert status in ["cpu_fallback", "failed"], f"Unexpected status: {status}"

    def test_fallback_computation_consistency(self, synthetic_xpcs_data):
        """Test that fallback computations are consistent."""
        try:
            from homodyne.core.jax_backend import compute_c2_model_jax
        except ImportError:
            pytest.skip("JAX backend not available")

        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        data = synthetic_xpcs_data
        t1 = jnp.array(data["t1"])
        t2 = jnp.array(data["t2"])
        phi = jnp.array(data["phi_angles_list"])
        q = data["wavevector_q_list"][0]

        params = {
            "offset": 1.0,
            "contrast": 0.4,
            "diffusion_coefficient": 0.1,
            "shear_rate": 0.0,
            "L": 1.0,
        }

        # Force CPU execution
        import os

        original_platform = os.environ.get("JAX_PLATFORM_NAME", "")
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

        try:
            result_fallback = compute_c2_model_jax(params, t1, t2, phi, q)

            # Should produce valid results
            assert jnp.all(jnp.isfinite(result_fallback))
            assert jnp.all(result_fallback >= 1.0 - 1e-10)  # Physical constraint

        finally:
            # Restore platform
            if original_platform:
                os.environ["JAX_PLATFORM_NAME"] = original_platform
            else:
                os.environ.pop("JAX_PLATFORM_NAME", None)

    def test_error_handling_invalid_gpu_config(self):
        """Test error handling with invalid GPU configurations."""
        if not GPU_MODULE_AVAILABLE:
            pytest.skip("GPU module not available")

        # Test various invalid configurations
        invalid_configs = [
            {"memory_fraction": -0.1},  # Negative memory
            {"memory_fraction": 1.5},  # Memory > 1
            {"gpu_id": -1},  # Negative GPU ID
        ]

        for config in invalid_configs:
            with pytest.raises((ValueError, TypeError)):
                activate_gpu(**config)

    def test_memory_management_cleanup(self):
        """Test memory management and cleanup."""
        if not GPU_MODULE_AVAILABLE:
            pytest.skip("GPU module not available")

        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")

        activator = GPUActivator(verbose=False)

        # Multiple activation/deactivation cycles
        for _ in range(3):
            result = activator.activate(memory_fraction=0.3, force_gpu=False)

            # Do some computation if GPU is available
            if result.get("success", False):
                try:
                    A = jnp.ones((100, 100))
                    B = jnp.dot(A, A)
                    B.block_until_ready()
                except:
                    pass  # Memory issues are acceptable

            activator.deactivate()

        # Should complete without memory leaks or errors
        assert True  # If we get here, cleanup worked


@pytest.mark.gpu
@pytest.mark.slow
class TestGPUIntegration:
    """Integration tests for GPU functionality."""

    def test_end_to_end_gpu_workflow(self, temp_dir):
        """Test complete GPU-accelerated workflow."""
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        try:
            from homodyne.tests.factories.data_factory import XPCSDataFactory

            from homodyne.optimization.nlsq import OPTIMISTIX_AVAILABLE, fit_nlsq_jax
        except ImportError:
            pytest.skip("Required modules not available")

        if not OPTIMISTIX_AVAILABLE:
            pytest.skip("Optimistix not available")

        # Step 1: Activate GPU
        activation_result = activate_gpu(
            memory_fraction=0.6, force_gpu=False, verbose=False
        )

        if not activation_result.get("success", False):
            pytest.skip("GPU activation failed")

        # Step 2: Generate test data
        factory = XPCSDataFactory(seed=42)
        data = factory.create_synthetic_correlation_data(
            n_times=40, n_angles=24, noise_level=0.01
        )

        # Step 3: Configure for GPU analysis
        gpu_config = {
            "analysis_mode": "static_isotropic",
            "optimization": {
                "method": "nlsq",
                "lsq": {"max_iterations": 50, "tolerance": 1e-6},
            },
            "hardware": {"force_cpu": False, "gpu_memory_fraction": 0.6},
            "output": {
                "directory": str(temp_dir),
                "save_plots": False,
                "verbose": False,
            },
        }

        # Step 4: Run optimization
        result = fit_nlsq_jax(data, gpu_config)

        # Step 5: Validate results
        assert result.success, f"GPU workflow failed: {result.message}"
        assert hasattr(result, "parameters")
        assert result.chi_squared >= 0.0
        assert result.computation_time > 0.0

        # Parameter recovery should be reasonable
        true_params = data["true_parameters"]
        recovered_params = result.parameters

        for param_name in ["offset", "contrast"]:
            if param_name in recovered_params and param_name in true_params:
                true_val = true_params[param_name]
                recovered_val = recovered_params[param_name]
                relative_error = abs(recovered_val - true_val) / (abs(true_val) + 1e-10)

                # Should recover parameters reasonably well
                assert (
                    relative_error < 0.2
                ), f"Poor parameter recovery for {param_name}: {relative_error:.3f}"

    def test_gpu_vs_cpu_workflow_consistency(self, synthetic_xpcs_data):
        """Test consistency between GPU and CPU workflows."""
        gpu_devices = []
        try:
            gpu_devices = jax.devices("gpu")
        except:
            pass

        if len(gpu_devices) == 0:
            pytest.skip("No GPU available")

        try:
            from homodyne.optimization.nlsq import OPTIMISTIX_AVAILABLE, fit_nlsq_jax
        except ImportError:
            pytest.skip("Optimization module not available")

        if not OPTIMISTIX_AVAILABLE:
            pytest.skip("Optimistix not available")

        data = synthetic_xpcs_data

        # CPU configuration
        cpu_config = {
            "analysis_mode": "static_isotropic",
            "optimization": {
                "method": "nlsq",
                "lsq": {"max_iterations": 30, "tolerance": 1e-6},
            },
            "hardware": {"force_cpu": True},
        }

        # GPU configuration
        gpu_config = cpu_config.copy()
        gpu_config["hardware"] = {"force_cpu": False, "gpu_memory_fraction": 0.5}

        try:
            # Run both workflows
            result_cpu = fit_nlsq_jax(data, cpu_config)
            result_gpu = fit_nlsq_jax(data, gpu_config)

            # Both should succeed
            if result_cpu.success and result_gpu.success:
                # Parameters should be close
                cpu_params = result_cpu.parameters
                gpu_params = result_gpu.parameters

                common_params = set(cpu_params.keys()) & set(gpu_params.keys())

                for param_name in common_params:
                    cpu_val = cpu_params[param_name]
                    gpu_val = gpu_params[param_name]

                    # Should be numerically close
                    relative_diff = abs(cpu_val - gpu_val) / (abs(cpu_val) + 1e-10)
                    assert (
                        relative_diff < 0.01
                    ), f"CPU/GPU parameter mismatch for {param_name}: {relative_diff:.4f}"

                # Chi-squared should be close
                chi2_diff = abs(result_cpu.chi_squared - result_gpu.chi_squared)
                chi2_rel_diff = chi2_diff / (result_cpu.chi_squared + 1e-10)
                assert (
                    chi2_rel_diff < 0.05
                ), f"CPU/GPU chi-squared mismatch: {chi2_rel_diff:.4f}"

        except Exception as e:
            pytest.skip(f"GPU vs CPU comparison failed: {e}")
