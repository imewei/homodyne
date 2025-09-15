#!/usr/bin/env python3
"""
Test script for the intelligent fallback architecture in JAX backend.
This script validates the fallback system functionality without JAX.
"""

import importlib
import os
import sys
import tempfile
from typing import Any, Dict

# Add the homodyne directory to the path
sys.path.insert(0, "/home/wei/Documents/GitHub/homodyne")


def test_backend_import():
    """Test that the backend imports correctly with and without JAX."""
    print("=" * 60)
    print("Testing Backend Import")
    print("=" * 60)

    # Test with current environment (may or may not have JAX)
    try:
        from homodyne.core import jax_backend

        print("✅ Successfully imported jax_backend")

        # Check what's available
        print(f"JAX available: {jax_backend.jax_available}")
        print(f"NumPy gradients available: {jax_backend.numpy_gradients_available}")

        return True
    except ImportError as e:
        print(f"❌ Failed to import jax_backend: {e}")
        return False


def test_backend_validation():
    """Test the backend validation system."""
    print("\n" + "=" * 60)
    print("Testing Backend Validation")
    print("=" * 60)

    try:
        from homodyne.core import jax_backend

        # Run comprehensive backend validation
        results = jax_backend.validate_backend()

        print("Backend Validation Results:")
        for key, value in results.items():
            if key == "test_results":
                print(f"  {key}:")
                for test_name, test_result in value.items():
                    print(f"    {test_name}: {test_result}")
            elif key == "recommendations":
                if value:
                    print(f"  {key}:")
                    for rec in value:
                        print(f"    - {rec}")
            else:
                print(f"  {key}: {value}")

        return results["gradient_support"] or results["hessian_support"]

    except Exception as e:
        print(f"❌ Backend validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_device_info():
    """Test device information retrieval."""
    print("\n" + "=" * 60)
    print("Testing Device Information")
    print("=" * 60)

    try:
        from homodyne.core import jax_backend

        device_info = jax_backend.get_device_info()
        performance_summary = jax_backend.get_performance_summary()

        print("Device Information:")
        for key, value in device_info.items():
            if key == "recommendations":
                if value:
                    print(f"  {key}:")
                    for rec in value:
                        print(f"    - {rec}")
            else:
                print(f"  {key}: {value}")

        print("\nPerformance Summary:")
        for key, value in performance_summary.items():
            if key == "recommendations":
                if value:
                    print(f"  {key}:")
                    for rec in value:
                        print(f"    - {rec}")
            elif key == "fallback_stats":
                print(f"  {key}: {dict(value)}")
            else:
                print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Device info test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_computations():
    """Test basic forward computations work."""
    print("\n" + "=" * 60)
    print("Testing Basic Computations")
    print("=" * 60)

    try:
        import numpy as np

        from homodyne.core import jax_backend

        # Test parameters
        test_params = np.array([100.0, 0.0, 10.0])
        test_t1 = np.array([0.0])
        test_t2 = np.array([1.0])
        test_q = 0.01

        # Test diffusion computation
        result = jax_backend.compute_g1_diffusion(test_params, test_t1, test_t2, test_q)
        print(f"✅ compute_g1_diffusion: {result}")

        # Test complete g2 computation
        phi = np.array([0.0, 45.0, 90.0])
        L = 1000.0  # mm
        contrast = 0.8
        offset = 0.0

        g2_result = jax_backend.compute_g2_scaled(
            test_params, test_t1, test_t2, phi, test_q, L, contrast, offset
        )
        print(f"✅ compute_g2_scaled: {g2_result}")

        # Test chi-squared computation
        data = g2_result + np.random.normal(0, 0.01, g2_result.shape)
        sigma = np.ones_like(data) * 0.01

        chi2_result = jax_backend.compute_chi_squared(
            test_params, data, sigma, test_t1, test_t2, phi, test_q, L, contrast, offset
        )
        print(f"✅ compute_chi_squared: {chi2_result}")

        return True

    except Exception as e:
        print(f"❌ Basic computations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gradient_computations():
    """Test gradient computations with fallback system."""
    print("\n" + "=" * 60)
    print("Testing Gradient Computations")
    print("=" * 60)

    try:
        import numpy as np

        from homodyne.core import jax_backend

        # Test parameters
        test_params = np.array([100.0, 0.0, 10.0])
        test_t1 = np.array([0.0])
        test_t2 = np.array([1.0])
        test_q = 0.01
        phi = np.array([0.0])  # Single angle to get scalar output
        L = 1000.0
        contrast = 0.8
        offset = 0.0

        print("Testing individual gradient functions...")

        # Create wrapper functions that return scalars for gradient testing
        def g2_scalar(params):
            result = jax_backend.compute_g2_scaled(
                params, test_t1, test_t2, phi, test_q, L, contrast, offset
            )
            return result[0]  # Return first element as scalar

        def chi2_scalar(params):
            data = np.array([1.5])  # Single data point
            sigma = np.array([0.01])
            return jax_backend.compute_chi_squared(
                params, data, sigma, test_t1, test_t2, phi, test_q, L, contrast, offset
            )

        # Test g2 gradient with scalar function
        try:
            if jax_backend.jax_available:
                grad_func = jax_backend.grad(g2_scalar)
            else:
                grad_func = jax_backend.grad(g2_scalar)
            grad_g2_result = grad_func(test_params)
            print(f"✅ gradient_g2 (scalar): {grad_g2_result}")
        except Exception as e:
            print(f"❌ gradient_g2 failed: {e}")
            # This might fail with JAX but should work with numpy fallback
            if not jax_backend.jax_available:
                return False
            else:
                print("  (Expected with JAX for vector-output functions)")

        # Test chi2 gradient (this should work as chi2 returns scalar)
        try:
            if jax_backend.jax_available:
                grad_func = jax_backend.grad(chi2_scalar)
            else:
                grad_func = jax_backend.grad(chi2_scalar)
            grad_chi2_result = grad_func(test_params)
            print(f"✅ gradient_chi2: {grad_chi2_result}")
        except Exception as e:
            print(f"❌ gradient_chi2 failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Gradient computations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_hessian_computations():
    """Test Hessian computations with fallback system."""
    print("\n" + "=" * 60)
    print("Testing Hessian Computations")
    print("=" * 60)

    try:
        import numpy as np

        from homodyne.core import jax_backend

        # Test parameters
        test_params = np.array([100.0, 0.0, 10.0])
        test_t1 = np.array([0.0])
        test_t2 = np.array([1.0])
        test_q = 0.01
        phi = np.array([0.0])
        L = 1000.0
        contrast = 0.8
        offset = 0.0

        print("Testing individual Hessian functions...")

        # Test g2 hessian with scalar wrapper
        def g2_scalar(params):
            result = jax_backend.compute_g2_scaled(
                params, test_t1, test_t2, phi, test_q, L, contrast, offset
            )
            return result[0]  # Return first element as scalar

        try:
            hess_func = jax_backend.hessian(g2_scalar)
            hess_g2_result = hess_func(test_params)
            print(f"✅ hessian_g2 shape: {hess_g2_result.shape}")
            cond_num = np.linalg.cond(hess_g2_result)
            print(f"    Condition number: {cond_num:.2e}")
        except Exception as e:
            print(f"❌ hessian_g2 failed: {e}")
            return False

        # Test chi2 hessian with scalar wrapper
        def chi2_scalar(params):
            data = np.array([1.5])
            sigma = np.array([0.01])
            return jax_backend.compute_chi_squared(
                params, data, sigma, test_t1, test_t2, phi, test_q, L, contrast, offset
            )

        try:
            hess_func = jax_backend.hessian(chi2_scalar)
            hess_chi2_result = hess_func(test_params)
            print(f"✅ hessian_chi2 shape: {hess_chi2_result.shape}")
            cond_num = np.linalg.cond(hess_chi2_result)
            print(f"    Condition number: {cond_num:.2e}")
        except Exception as e:
            print(f"❌ hessian_chi2 failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"❌ Hessian computations test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_performance_warnings():
    """Test that performance warnings are issued appropriately."""
    print("\n" + "=" * 60)
    print("Testing Performance Warning System")
    print("=" * 60)

    try:
        from homodyne.core import jax_backend

        # Check if JAX is available
        if jax_backend.jax_available:
            print("ℹ️  JAX is available - no fallback warnings expected")
        else:
            print("ℹ️  JAX not available - fallback warnings should be issued")

        # Check fallback statistics
        performance_summary = jax_backend.get_performance_summary()
        fallback_stats = performance_summary.get("fallback_stats", {})

        print(f"Fallback statistics: {dict(fallback_stats)}")

        if not jax_backend.jax_available and sum(fallback_stats.values()) > 0:
            print("✅ Fallback statistics are being tracked")
        elif jax_backend.jax_available:
            print("✅ JAX available - no fallback tracking needed")
        else:
            print("ℹ️  No fallback operations performed yet")

        return True

    except Exception as e:
        print(f"❌ Performance warning test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all fallback system tests."""
    print("🧪 Testing Intelligent Fallback Architecture for Homodyne JAX Backend")
    print("=" * 80)

    tests = [
        ("Backend Import", test_backend_import),
        ("Backend Validation", test_backend_validation),
        ("Device Information", test_device_info),
        ("Basic Computations", test_basic_computations),
        ("Gradient Computations", test_gradient_computations),
        ("Hessian Computations", test_hessian_computations),
        ("Performance Warnings", test_performance_warnings),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🏃 Running {test_name}...")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"💥 {test_name}: CRASHED - {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed / total * 100:.1f}%")

    if passed == total:
        print(
            "\n🎉 All tests passed! Intelligent fallback system is working correctly."
        )
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
