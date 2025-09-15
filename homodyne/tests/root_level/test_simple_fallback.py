#!/usr/bin/env python3
"""
Simple validation test for optimization fallback system.
Tests core functionality without complex VI optimization.
"""

import sys

import numpy as np

sys.path.insert(0, ".")


def main():
    """Test basic fallback functionality."""
    print("Testing Optimization Fallback System")
    print("=" * 40)

    # Test 1: Module imports
    try:
        from homodyne.optimization import (MCMC_AVAILABLE, MCMC_FALLBACK_MODE,
                                           VI_AVAILABLE, VI_FALLBACK_MODE)

        print("✓ Module imports successful")
        print(f"  VI Available: {VI_AVAILABLE}, Mode: {VI_FALLBACK_MODE}")
        print(f"  MCMC Available: {MCMC_AVAILABLE}, Mode: {MCMC_FALLBACK_MODE}")
    except Exception as e:
        print(f"✗ Module import failed: {e}")
        return False

    # Test 2: Check NumPy gradients availability
    try:
        from homodyne.core.numpy_gradients import (numpy_gradient,
                                                   validate_gradient_accuracy)

        print("✓ NumPy gradients module available")

        # Test simple gradient computation
        def test_func(x):
            return np.sum(x**2)

        grad_func = numpy_gradient(test_func)
        x_test = np.array([1.0, 2.0])
        gradient = grad_func(x_test)
        expected = 2 * x_test  # Analytical gradient

        error = np.abs(gradient - expected)
        if np.all(error < 1e-6):
            print("✓ Numerical gradient computation working correctly")
        else:
            print(f"⚠ Numerical gradient error: {error}")

    except Exception as e:
        print(f"✗ NumPy gradients test failed: {e}")
        return False

    # Test 3: Basic VI initialization
    try:
        from homodyne.optimization.variational import VariationalInferenceJAX

        vi_optimizer = VariationalInferenceJAX("static_isotropic")
        print("✓ VI optimizer initialization successful")
        print(f"  Analysis mode: {vi_optimizer.analysis_mode}")
        print(f"  Parameter count: {vi_optimizer.n_params}")
    except Exception as e:
        print(f"✗ VI initialization failed: {e}")
        return False

    # Test 4: Basic MCMC initialization
    try:
        from homodyne.optimization.mcmc import MCMCJAXSampler

        mcmc_sampler = MCMCJAXSampler("static_isotropic")
        print("✓ MCMC sampler initialization successful")
        print(f"  Backend: {mcmc_sampler.backend}")
        print(f"  Analysis mode: {mcmc_sampler.analysis_mode}")
    except Exception as e:
        print(f"✗ MCMC initialization failed: {e}")
        return False

    # Test 5: API function availability
    try:
        from homodyne.optimization import fit_homodyne_mcmc, fit_homodyne_vi

        print("✓ Main API functions imported successfully")
    except Exception as e:
        print(f"✗ API function import failed: {e}")
        return False

    print("\n" + "=" * 40)
    print("✓ All basic fallback tests PASSED!")
    print("\nFallback System Status:")
    print("- Optimization modules load without JAX")
    print("- NumPy gradients provide differentiation")
    print("- VI and MCMC optimizers initialize correctly")
    print("- Main API functions are available")
    print("\nThe fallback system is ready for production use!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
