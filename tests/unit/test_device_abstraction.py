"""
Unit tests for device abstraction transparency (FR-005).

Tests cover:
- JAX device-agnostic code execution
- Automatic CPU/GPU detection and fallback
- Same results across devices (within numerical tolerance)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from homodyne.optimization.nlsq_wrapper import NLSQWrapper


class TestDeviceAbstraction:
    """Test device abstraction transparency (FR-005, T013b)."""

    def test_residual_function_device_agnostic(self):
        """
        Verify residual function works on both CPU and GPU without code changes.

        FR-005: Users should not need to write device-specific code.
        """
        # Create mock XPCS data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, np.pi/2])
                self.t1 = np.linspace(0, 1, 5)
                self.t2 = np.linspace(0, 1, 5)
                self.g2 = np.random.rand(2, 5, 5)
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Create residual function (should be device-agnostic)
        residual_fn = wrapper._create_residual_function(
            mock_data,
            analysis_mode="static_isotropic"
        )

        # Test parameters
        xdata = np.arange(2 * 5 * 5, dtype=np.float64)
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])

        # Get available devices
        devices = jax.devices()
        cpu_device = jax.devices('cpu')[0]

        # Test on CPU
        with jax.default_device(cpu_device):
            cpu_result = residual_fn(xdata, *params)

        assert cpu_result.shape == (50,), \
            f"CPU result should have shape (50,), got {cpu_result.shape}"
        assert not np.any(np.isnan(cpu_result)), \
            "CPU result should not contain NaN"

        # If GPU available, test on GPU
        try:
            gpu_device = jax.devices('gpu')[0]

            with jax.default_device(gpu_device):
                gpu_result = residual_fn(xdata, *params)

            assert gpu_result.shape == (50,), \
                f"GPU result should have shape (50,), got {gpu_result.shape}"
            assert not np.any(np.isnan(gpu_result)), \
                "GPU result should not contain NaN"

            # Verify CPU and GPU results match (within numerical tolerance)
            np.testing.assert_allclose(
                cpu_result,
                gpu_result,
                rtol=1e-5,
                atol=1e-6,
                err_msg="CPU and GPU results should match within tolerance"
            )

        except RuntimeError:
            # GPU not available - test passes with CPU only
            pytest.skip("GPU not available for device abstraction test")

    def test_automatic_device_detection(self):
        """
        Verify JAX automatically detects and uses available devices.

        FR-005: No explicit device placement needed by user.
        """
        # JAX should have at least CPU available
        devices = jax.devices()
        assert len(devices) > 0, "JAX should detect at least one device (CPU)"

        # First device should be usable
        default_device = jax.devices()[0]
        assert default_device is not None

        # Simple computation should work on default device
        x = jnp.array([1.0, 2.0, 3.0])
        y = x * 2.0

        assert y.shape == (3,)
        np.testing.assert_array_equal(y, jnp.array([2.0, 4.0, 6.0]))

    def test_wrapper_methods_device_transparent(self):
        """
        Verify all NLSQWrapper methods work without device-specific code.

        FR-005: Complete device abstraction across all operations.
        """
        # Create mock data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0])
                self.t1 = np.array([0.0, 1.0])
                self.t2 = np.array([0.0, 1.0])
                self.g2 = np.ones((1, 2, 2))
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Test data preparation (device-agnostic)
        xdata, ydata = wrapper._prepare_data(mock_data)
        assert xdata.shape == (4,)
        assert ydata.shape == (4,)

        # Test bounds conversion (device-agnostic)
        bounds = (np.array([0.0, 0.0, 100.0, 0.3, 1.0]),
                  np.array([1.0, 2.0, 1e5, 1.5, 1000.0]))
        nlsq_bounds = wrapper._convert_bounds(bounds)
        assert nlsq_bounds is not None
        assert len(nlsq_bounds) == 2

        # Test parameter validation (device-agnostic)
        params = np.array([0.5, 1.0, 1000.0, 0.5, 10.0])
        validated_params = wrapper._validate_initial_params(params, bounds)
        assert validated_params.shape == (5,)

        # Test residual function creation (device-agnostic)
        residual_fn = wrapper._create_residual_function(
            mock_data,
            analysis_mode="static_isotropic"
        )
        assert callable(residual_fn)

        # All operations succeeded without explicit device management
        # This validates FR-005: Device abstraction transparency

    def test_jax_jit_compilation_device_independent(self):
        """
        Verify JIT-compiled functions work on available devices.

        FR-005: JIT compilation should be device-transparent.
        """
        # Create simple mock data
        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0])
                self.t1 = np.array([0.0, 0.5])
                self.t2 = np.array([0.0, 0.5])
                self.g2 = np.ones((1, 2, 2))
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        residual_fn = wrapper._create_residual_function(
            mock_data,
            analysis_mode="static_isotropic"
        )

        # JIT compile (should work on any device)
        jitted_fn = jax.jit(residual_fn)

        xdata = jnp.arange(4, dtype=jnp.float32)
        params = jnp.array([0.5, 1.0, 1000.0, 0.5, 10.0])

        # First call: compilation
        result1 = jitted_fn(xdata, *params)
        assert result1.shape == (4,)

        # Second call: use cached compilation
        result2 = jitted_fn(xdata, *params)
        assert result2.shape == (4,)

        # Results should be identical (deterministic)
        np.testing.assert_array_equal(result1, result2)

    def test_graceful_gpu_fallback(self):
        """
        Verify code works when GPU is unavailable (CPU fallback).

        FR-005: Graceful degradation when GPU not available.
        """
        # Force CPU execution
        cpu_device = jax.devices('cpu')[0]

        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0])
                self.t1 = np.array([0.0, 1.0])
                self.t2 = np.array([0.0, 1.0])
                self.g2 = np.ones((1, 2, 2))
                self.sigma = np.ones_like(self.g2) * 0.1
                self.q = 0.01
                self.L = 1.0

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        with jax.default_device(cpu_device):
            # All operations should work on CPU
            residual_fn = wrapper._create_residual_function(
                mock_data,
                analysis_mode="static_isotropic"
            )

            xdata = jnp.arange(4, dtype=jnp.float32)
            params = jnp.array([0.5, 1.0, 1000.0, 0.5, 10.0])

            result = residual_fn(xdata, *params)

            assert result.shape == (4,)
            assert not np.any(np.isnan(result))

        # Test passes: CPU fallback works correctly
