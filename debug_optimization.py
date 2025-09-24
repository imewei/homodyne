#!/usr/bin/env python
"""
Debug script for NLSQ and MCMC NUTS implementations.
Tests both optimization methods with synthetic data.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.config import ConfigManager
from homodyne.core.jax_backend import compute_g2_scaled

# Enable JAX debugging
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)  # Keep JIT enabled for performance

def generate_synthetic_data(n_angles=3, n_frames=50, noise_level=0.01):
    """Generate synthetic XPCS data for testing."""
    print("Generating synthetic data...")

    # True parameters
    true_params = {
        'D0': 10000.0,
        'alpha': -1.5,
        'D_offset': 100.0,
        'gamma_dot_0': 0.001,
        'beta': 0.0,
        'gamma_dot_offset': 0.0,
        'phi_0': 0.0,
    }

    # Create time arrays
    dt = 0.1
    times = np.linspace(0, dt * n_frames, n_frames)
    t1, t2 = np.meshgrid(times, times, indexing='ij')

    # Create angle array
    phi_angles = np.linspace(0, 90, n_angles)

    # Physical parameters
    q = 0.0054  # wavevector
    L = 100.0   # sample-detector distance
    contrast = 0.3
    offset = 1.0

    # Generate theoretical correlation
    params_array = jnp.array([
        true_params['D0'], true_params['alpha'], true_params['D_offset'],
        true_params['gamma_dot_0'], true_params['beta'],
        true_params['gamma_dot_offset'], true_params['phi_0']
    ])

    # Generate data for each angle
    c2_exp = []
    sigma = []

    for phi in phi_angles:
        # Use simplified model for testing
        # c2 = contrast * g2 + offset where g2 decays with time
        dt_matrix = np.abs(t2 - t1)
        g2 = np.exp(-dt_matrix / 10.0)  # Simple exponential decay
        c2 = contrast * g2 + offset

        # Add noise
        noise = np.random.normal(0, noise_level, c2.shape)
        c2_noisy = c2 + noise

        c2_exp.append(c2_noisy)
        sigma.append(np.ones_like(c2) * noise_level)

    c2_exp = np.array(c2_exp)
    sigma = np.array(sigma)

    data = {
        'c2_exp': c2_exp,
        'sigma': sigma,
        't1': t1,
        't2': t2,
        'phi_angles_list': phi_angles,
        'wavevector_q_list': np.array([q]),
        'q': q,
        'L': L,
        'contrast': contrast,
        'offset': offset,
        'true_params': true_params,
    }

    print(f"Generated data shapes:")
    print(f"  c2_exp: {c2_exp.shape}")
    print(f"  sigma: {sigma.shape}")
    print(f"  t1, t2: {t1.shape}")
    print(f"  phi_angles: {phi_angles.shape}")

    return data

def test_nlsq_optimization(data):
    """Test NLSQ optimization."""
    print("\n" + "="*60)
    print("Testing NLSQ (Optimistix) Optimization")
    print("="*60)

    # Create minimal config
    config = {
        'analysis_mode': 'laminar_flow',
        'optimization': {
            'nlsq': {
                'max_iterations': 100,
                'tolerance': 1e-6,
                'verbose': True,
            }
        }
    }

    try:
        start_time = time.time()
        result = fit_nlsq_jax(
            data=data,
            config_dict=config,
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\nNLSQ Results:")
        print(f"  Success: {result.success}")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Iterations: {result.iterations}")
        print(f"  Chi-squared: {result.chi_squared:.6f}")
        print(f"\nFitted Parameters:")
        param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
        for i, name in enumerate(param_names):
            if i < len(result.parameters):
                true_val = data['true_params'][name]
                fitted_val = result.parameters[i]
                error = abs(fitted_val - true_val) / abs(true_val) * 100
                print(f"  {name}: {fitted_val:.6f} (true: {true_val:.6f}, error: {error:.1f}%)")

        print(f"\nScaling Parameters:")
        print(f"  Contrast: {result.contrast:.6f} (true: {data['contrast']:.6f})")
        print(f"  Offset: {result.offset:.6f} (true: {data['offset']:.6f})")

        return result

    except Exception as e:
        print(f"NLSQ optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mcmc_optimization(data):
    """Test MCMC NUTS optimization."""
    print("\n" + "="*60)
    print("Testing MCMC NUTS (NumPyro) Optimization")
    print("="*60)

    # Create minimal config
    config = {
        'analysis_mode': 'laminar_flow',
        'optimization': {
            'mcmc': {
                'n_warmup': 100,
                'n_samples': 200,
                'n_chains': 2,
                'backend': 'numpyro',  # Use NumPyro NUTS
                'verbose': True,
            }
        }
    }

    try:
        start_time = time.time()
        result = fit_mcmc_jax(
            data=data,
            config_dict=config,
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\nMCMC Results:")
        print(f"  Success: {result.success}")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Chains: {result.n_chains}")
        print(f"  Samples per chain: {result.n_samples}")

        if result.samples is not None:
            print(f"\nPosterior Means:")
            param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
            for i, name in enumerate(param_names):
                if i < len(result.mean_params):
                    true_val = data['true_params'][name]
                    mean_val = result.mean_params[i]
                    std_val = result.std_params[i] if result.std_params is not None else 0
                    error = abs(mean_val - true_val) / abs(true_val) * 100
                    print(f"  {name}: {mean_val:.6f} ± {std_val:.6f} (true: {true_val:.6f}, error: {error:.1f}%)")

        print(f"\nDiagnostics:")
        if hasattr(result, 'diagnostics') and result.diagnostics:
            for key, value in result.diagnostics.items():
                print(f"  {key}: {value}")

        return result

    except Exception as e:
        print(f"MCMC optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run debug tests."""
    print("Starting Optimization Debug Tests")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    data = generate_synthetic_data(
        n_angles=2,      # Reduced for faster testing
        n_frames=30,     # Reduced for faster testing
        noise_level=0.01
    )

    # Test NLSQ
    nlsq_result = test_nlsq_optimization(data)

    # Test MCMC
    mcmc_result = test_mcmc_optimization(data)

    # Compare results
    if nlsq_result and mcmc_result:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)
        print(f"NLSQ Chi-squared: {nlsq_result.chi_squared:.6f}")
        if hasattr(mcmc_result, 'chi_squared'):
            print(f"MCMC Chi-squared: {mcmc_result.chi_squared:.6f}")

        # Parameter comparison
        print("\nParameter Agreement:")
        param_names = ['D0', 'alpha', 'D_offset']
        for i, name in enumerate(param_names[:3]):  # Just check first 3 params
            if i < min(len(nlsq_result.parameters), len(mcmc_result.mean_params)):
                nlsq_val = nlsq_result.parameters[i]
                mcmc_val = mcmc_result.mean_params[i]
                diff = abs(nlsq_val - mcmc_val) / abs(nlsq_val) * 100
                print(f"  {name}: NLSQ={nlsq_val:.6f}, MCMC={mcmc_val:.6f}, diff={diff:.1f}%")

    print("\n" + "="*60)
    print("Debug tests completed!")

if __name__ == "__main__":
    main()