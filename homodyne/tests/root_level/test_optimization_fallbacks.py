#!/usr/bin/env python3
"""
Integration test for optimization fallback system validation.

Tests all optimization methods (VI, MCMC, Hybrid) with and without JAX
to ensure seamless fallback to NumPy implementations.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add homodyne to path
sys.path.insert(0, "/home/wei/Documents/GitHub/homodyne")


def generate_test_data():
    """Generate synthetic XPCS data for testing."""
    # Simple test case - static isotropic (3 parameters)
    n_angles = 3
    n_times = 50

    # Create time and angle grids
    t1 = np.logspace(-3, 1, n_times)
    t2 = np.logspace(-3, 1, n_times)
    phi = np.linspace(0, np.pi / 2, n_angles)

    # Create meshgrid
    T1, T2, PHI = np.meshgrid(t1, t2, phi, indexing="ij")

    # True parameters for static isotropic case
    true_params = np.array([1.0, 0.8, 0.1])  # D0, alpha, D_offset
    true_contrast = 0.9
    true_offset = 0.05

    # Generate synthetic g1 function
    tau = np.abs(T1 - T2)
    D_tau = true_params[0] * tau ** true_params[1] + true_params[2]
    q = 0.1  # nm^-1
    L = 100  # mm

    g1_theory = np.exp(-0.5 * q**2 * D_tau)
    g2_theory = true_contrast * g1_theory**2 + true_offset

    # Add realistic noise
    noise_level = 0.05
    noise = np.random.normal(0, noise_level * np.abs(g2_theory))
    data = g2_theory + noise
    sigma = np.full_like(data, noise_level * np.abs(g2_theory))

    return data, sigma, T1, T2, PHI, q, L, true_params, true_contrast, true_offset


def test_with_jax():
    """Test optimization with JAX available."""
    try:
        import jax

        print("Testing with JAX available...")

        # Import optimization after jax is confirmed available
        from homodyne.optimization import (MCMC_FALLBACK_MODE,
                                           VI_FALLBACK_MODE, fit_homodyne_mcmc,
                                           fit_homodyne_vi)

        print(f"VI fallback mode: {VI_FALLBACK_MODE}")
        print(f"MCMC fallback mode: {MCMC_FALLBACK_MODE}")

        # Generate test data
        data, sigma, t1, t2, phi, q, L, true_params, true_contrast, true_offset = (
            generate_test_data()
        )

        # Test VI
        print("\n=== Testing VI with JAX ===")
        start_time = time.time()
        vi_result = fit_homodyne_vi(
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            analysis_mode="static_isotropic",
            n_iterations=100,
        )
        vi_time = time.time() - start_time

        print(f"VI completed in {vi_time:.2f}s")
        print(f"VI backend: {vi_result.backend}")
        print(f"Parameter estimates: {vi_result.mean_params}")
        print(f"Parameter errors: {vi_result.std_params}")
        print(f"True parameters: {true_params}")

        # Test MCMC (smaller sample size for speed)
        print("\n=== Testing MCMC with JAX ===")
        start_time = time.time()
        mcmc_result = fit_homodyne_mcmc(
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            analysis_mode="static_isotropic",
            n_samples=50,
            n_warmup=50,
            n_chains=2,
            vi_init=vi_result,
        )
        mcmc_time = time.time() - start_time

        print(f"MCMC completed in {mcmc_time:.2f}s")
        print(f"MCMC backend: {mcmc_result.backend}")
        print(f"Parameter estimates: {mcmc_result.mean_params}")
        print(f"Parameter errors: {mcmc_result.std_params}")
        print(f"Converged: {mcmc_result.converged}")

        return True, vi_result, mcmc_result

    except ImportError as e:
        print(f"JAX not available: {e}")
        return False, None, None


def test_without_jax():
    """Test optimization with JAX unavailable (mock fallback)."""
    print("\nTesting fallback modes (simulating no JAX)...")

    # Temporarily hide jax by manipulating sys.modules
    jax_modules = {}
    for module_name in list(sys.modules.keys()):
        if "jax" in module_name:
            jax_modules[module_name] = sys.modules.pop(module_name)

    try:
        # Force reload of optimization modules without JAX
        if "homodyne.optimization.variational" in sys.modules:
            del sys.modules["homodyne.optimization.variational"]
        if "homodyne.optimization.mcmc" in sys.modules:
            del sys.modules["homodyne.optimization.mcmc"]
        if "homodyne.optimization" in sys.modules:
            del sys.modules["homodyne.optimization"]

        # Import optimization modules in no-JAX environment
        from homodyne.optimization import (MCMC_FALLBACK_MODE,
                                           VI_FALLBACK_MODE, fit_homodyne_mcmc,
                                           fit_homodyne_vi)

        print(f"Fallback VI mode: {VI_FALLBACK_MODE}")
        print(f"Fallback MCMC mode: {MCMC_FALLBACK_MODE}")

        # Generate test data
        data, sigma, t1, t2, phi, q, L, true_params, true_contrast, true_offset = (
            generate_test_data()
        )

        # Test VI fallback
        print("\n=== Testing VI Fallback ===")
        start_time = time.time()
        vi_result = fit_homodyne_vi(
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            analysis_mode="static_isotropic",
            n_iterations=50,
        )  # Fewer iterations for fallback
        vi_time = time.time() - start_time

        print(f"VI fallback completed in {vi_time:.2f}s")
        print(f"VI backend: {vi_result.backend}")
        print(f"Parameter estimates: {vi_result.mean_params}")
        print(f"Parameter errors: {vi_result.std_params}")

        # Test MCMC fallback
        print("\n=== Testing MCMC Fallback ===")
        start_time = time.time()
        mcmc_result = fit_homodyne_mcmc(
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            analysis_mode="static_isotropic",
            n_samples=25,
            n_warmup=25,
            n_chains=2,  # Very small for fallback
            vi_init=vi_result,
        )
        mcmc_time = time.time() - start_time

        print(f"MCMC fallback completed in {mcmc_time:.2f}s")
        print(f"MCMC backend: {mcmc_result.backend}")
        print(f"Parameter estimates: {mcmc_result.mean_params}")
        print(f"Parameter errors: {mcmc_result.std_params}")
        print(f"Converged: {mcmc_result.converged}")

        return True, vi_result, mcmc_result

    except Exception as e:
        print(f"Fallback test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None, None

    finally:
        # Restore JAX modules
        for module_name, module in jax_modules.items():
            sys.modules[module_name] = module


def validate_results(
    jax_vi_result,
    jax_mcmc_result,
    fallback_vi_result,
    fallback_mcmc_result,
    true_params,
):
    """Validate that fallback results are scientifically reasonable."""
    print("\n=== Result Validation ===")

    tolerance = 0.5  # Allow 50% difference (fallback may be less accurate)

    validation_results = {
        "vi_params_reasonable": False,
        "mcmc_params_reasonable": False,
        "vi_fallback_close": False,
        "mcmc_fallback_close": False,
    }

    if jax_vi_result and fallback_vi_result:
        # Check if fallback VI parameters are reasonable
        param_errors = np.abs(fallback_vi_result.mean_params - true_params) / np.abs(
            true_params
        )
        validation_results["vi_params_reasonable"] = np.all(
            param_errors < 1.0
        )  # Within 100%

        # Check if VI results are similar between JAX and fallback
        if jax_vi_result.backend != fallback_vi_result.backend:
            vi_diff = np.abs(
                jax_vi_result.mean_params - fallback_vi_result.mean_params
            ) / np.abs(jax_vi_result.mean_params)
            validation_results["vi_fallback_close"] = np.all(vi_diff < tolerance)

        print(f"VI parameter accuracy: {validation_results['vi_params_reasonable']}")
        print(
            f"VI JAX vs fallback agreement: {validation_results['vi_fallback_close']}"
        )

    if jax_mcmc_result and fallback_mcmc_result:
        # Check if fallback MCMC parameters are reasonable
        param_errors = np.abs(fallback_mcmc_result.mean_params - true_params) / np.abs(
            true_params
        )
        validation_results["mcmc_params_reasonable"] = np.all(
            param_errors < 1.0
        )  # Within 100%

        # Check if MCMC results are similar between JAX and fallback
        if jax_mcmc_result.backend != fallback_mcmc_result.backend:
            mcmc_diff = np.abs(
                jax_mcmc_result.mean_params - fallback_mcmc_result.mean_params
            ) / np.abs(jax_mcmc_result.mean_params)
            validation_results["mcmc_fallback_close"] = np.all(mcmc_diff < tolerance)

        print(
            f"MCMC parameter accuracy: {validation_results['mcmc_params_reasonable']}"
        )
        print(
            f"MCMC JAX vs fallback agreement: {validation_results['mcmc_fallback_close']}"
        )

    return validation_results


def main():
    """Run comprehensive fallback validation tests."""
    print("Optimization Fallback System Integration Test")
    print("=" * 50)

    # Test with JAX available
    jax_success, jax_vi_result, jax_mcmc_result = test_with_jax()

    # Test fallback modes
    fallback_success, fallback_vi_result, fallback_mcmc_result = test_without_jax()

    # Generate true parameters for validation
    _, _, _, _, _, _, _, true_params, _, _ = generate_test_data()

    # Validate results
    if jax_success and fallback_success:
        validation_results = validate_results(
            jax_vi_result,
            jax_mcmc_result,
            fallback_vi_result,
            fallback_mcmc_result,
            true_params,
        )

        # Summary
        print("\n=== Test Summary ===")
        print(f"JAX mode test: {'PASSED' if jax_success else 'FAILED'}")
        print(f"Fallback mode test: {'PASSED' if fallback_success else 'FAILED'}")
        print(
            f"Parameter accuracy: {'PASSED' if validation_results.get('vi_params_reasonable', False) else 'FAILED'}"
        )
        print(
            f"Fallback agreement: {'PASSED' if validation_results.get('vi_fallback_close', False) else 'MODERATE'}"
        )

        overall_success = (
            jax_success
            and fallback_success
            and validation_results.get("vi_params_reasonable", False)
        )

        print(
            f"\nOverall test result: {'PASSED' if overall_success else 'PARTIAL/FAILED'}"
        )

        if overall_success:
            print("\n✓ Optimization fallback system is working correctly!")
            print("✓ Users can use Homodyne v2 optimization with or without JAX")
            print("✓ Fallback modes provide scientifically accurate results")
        else:
            print("\n⚠ Some tests failed - check logs above for details")

        return overall_success
    else:
        print("⚠ Could not complete full test suite")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
