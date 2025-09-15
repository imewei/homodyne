#!/usr/bin/env python3
"""
Test script for enhanced model classes with intelligent gradient handling.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

import numpy as np

from homodyne.core.jax_backend import jax_available, numpy_gradients_available
from homodyne.core.models import CombinedModel, create_model


def test_basic_model_creation():
    """Test basic model creation and information retrieval."""
    print("=" * 80)
    print("Testing Enhanced Model Classes")
    print("=" * 80)

    # Test different analysis modes
    analysis_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]

    for mode in analysis_modes:
        print(f"\n--- Testing {mode} mode ---")
        model = create_model(mode)

        # Test basic properties
        print(f"Model name: {model.name}")
        print(f"Parameters: {model.n_params} ({model.parameter_names})")
        print(f"Supports gradients: {model.supports_gradients()}")
        print(f"Best gradient method: {model.get_best_gradient_method()}")


def test_gradient_capabilities():
    """Test gradient capability detection and reporting."""
    print("\n" + "=" * 80)
    print("Testing Gradient Capabilities")
    print("=" * 80)

    model = create_model("laminar_flow")

    # Get gradient capabilities
    capabilities = model.get_gradient_capabilities()

    print(f"Backend type: {capabilities['backend_type']}")
    print(f"Best method: {capabilities['best_method']}")
    print(f"Performance estimate: {capabilities['performance_estimate']}")
    print(f"JAX available: {capabilities['jax_available']}")
    print(f"NumPy gradients available: {capabilities['numpy_gradients_available']}")

    if capabilities["performance_warning"]:
        print(f"Performance warning: {capabilities['performance_warning']}")

    print(f"Backend summary: {capabilities['backend_summary']}")

    print("\nRecommendations:")
    for rec in capabilities["recommendations"]:
        print(f"  • {rec}")


def test_gradient_functions():
    """Test gradient function retrieval with enhanced error handling."""
    print("\n" + "=" * 80)
    print("Testing Gradient Function Retrieval")
    print("=" * 80)

    model = create_model("static_isotropic")

    try:
        print("Attempting to get gradient function...")
        grad_func = model.get_gradient_function()
        print("✅ Gradient function retrieved successfully")
        print(f"   Function: {grad_func}")

        print("\nAttempting to get Hessian function...")
        hess_func = model.get_hessian_function()
        print("✅ Hessian function retrieved successfully")
        print(f"   Function: {hess_func}")

    except ImportError as e:
        print("❌ Gradient/Hessian functions not available:")
        print(f"   Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def test_model_info():
    """Test enhanced model information."""
    print("\n" + "=" * 80)
    print("Testing Enhanced Model Information")
    print("=" * 80)

    model = create_model("laminar_flow")
    info = model.get_model_info()

    print("Model Information:")
    print(f"  Name: {info['name']}")
    print(f"  Analysis mode: {info['analysis_mode']}")
    print(f"  Parameters: {info['n_parameters']}")
    print(f"  Parameter names: {info['parameter_names']}")
    print(f"  Supports gradients: {info['supports_gradients']}")
    print(f"  Gradient method: {info['gradient_method']}")
    print(f"  Backend summary: {info['backend_summary']}")
    print(f"  Performance estimate: {info['performance_estimate']}")

    print("\nOptimization Recommendations:")
    for rec in info["optimization_recommendations"]:
        print(f"  {rec}")


def test_optimization_recommendations():
    """Test optimization recommendations for different modes."""
    print("\n" + "=" * 80)
    print("Testing Optimization Recommendations")
    print("=" * 80)

    modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]

    for mode in modes:
        print(f"\n--- {mode.upper()} MODE ---")
        model = create_model(mode)
        recommendations = model.get_optimization_recommendations()

        for rec in recommendations:
            print(f"  {rec}")


def test_performance_benchmark():
    """Test performance benchmarking if gradients are available."""
    print("\n" + "=" * 80)
    print("Testing Performance Benchmarking")
    print("=" * 80)

    model = create_model("static_isotropic")

    if model.supports_gradients():
        print("Running performance benchmark...")
        try:
            benchmark = model.benchmark_gradient_performance()

            print(f"Test conditions:")
            print(f"  Parameters: {benchmark['test_conditions']['n_parameters']}")
            print(f"  Time points: {benchmark['test_conditions']['n_time_points']}")
            print(f"  Analysis mode: {benchmark['test_conditions']['analysis_mode']}")

            print(f"\nMethods tested:")
            for method_key, method_info in benchmark["methods"].items():
                if method_info["success"]:
                    time_ms = method_info["computation_time"] * 1000
                    ratio = method_info.get("performance_ratio", "N/A")
                    print(f"  {method_info['name']}: {time_ms:.2f}ms (ratio: {ratio})")
                else:
                    print(f"  {method_info['name']}: Failed - {method_info['error']}")

            if benchmark.get("best_method"):
                best = benchmark["best_method"]
                print(f"\nBest method: {best['name']} ({best['time'] * 1000:.2f}ms)")

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    else:
        print("⚠️  No gradient support available for benchmarking")


def test_gradient_validation():
    """Test gradient accuracy validation."""
    print("\n" + "=" * 80)
    print("Testing Gradient Validation")
    print("=" * 80)

    model = create_model("static_isotropic")

    try:
        validation = model.validate_gradient_accuracy()

        print("Validation results:")
        assessment = validation["accuracy_assessment"]

        if assessment.get("gradient_computed", False):
            print(f"  ✅ Gradient computed successfully")
            print(f"  Shape: {assessment['gradient_shape']}")
            print(f"  All finite: {assessment['gradient_finite']}")
            print(f"  Magnitude: {assessment['gradient_magnitude']:.2e}")
            print(f"  Max component: {assessment['max_gradient_component']:.2e}")
            print(f"  Method used: {assessment['method_used']}")
        else:
            print(f"  ❌ Gradient computation failed")
            if "error" in assessment:
                print(f"     Error: {assessment['error']}")

        print("\nRecommendations:")
        for rec in validation["recommendations"]:
            print(f"  • {rec}")

    except Exception as e:
        print(f"❌ Validation failed: {e}")


def main():
    """Run all tests."""
    print(f"JAX available: {jax_available}")
    print(f"NumPy gradients available: {numpy_gradients_available}")

    test_basic_model_creation()
    test_gradient_capabilities()
    test_gradient_functions()
    test_model_info()
    test_optimization_recommendations()
    test_performance_benchmark()
    test_gradient_validation()

    print("\n" + "=" * 80)
    print("Enhanced Model Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
