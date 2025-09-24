#!/usr/bin/env python
"""
Fixed debug script for NLSQ and MCMC NUTS implementations.
Tests both optimization methods with synthetic data using correct API.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.config import ConfigManager

# Enable JAX debugging
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", False)  # Keep JIT enabled for performance

def generate_synthetic_data(n_angles=3, n_frames=50, noise_level=0.01):
    """Generate synthetic XPCS data for testing."""
    print("Generating synthetic data...")

    # True parameters (laminar flow mode - 7 parameters)
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

    # Create data dictionary for NLSQ (uses full data dict)
    data_nlsq = {
        'c2_exp': c2_exp,
        'sigma': sigma,
        't1': t1,
        't2': t2,
        'phi_angles_list': phi_angles,
        'wavevector_q_list': np.array([q]),
    }

    # Additional info for comparison
    data_info = {
        'q': q,
        'L': L,
        'contrast': contrast,
        'offset': offset,
        'true_params': true_params,
        'phi_angles': phi_angles,
        't1': t1,
        't2': t2,
        'c2_exp': c2_exp,
        'sigma': sigma,
    }

    print(f"Generated data shapes:")
    print(f"  c2_exp: {c2_exp.shape}")
    print(f"  sigma: {sigma.shape}")
    print(f"  t1, t2: {t1.shape}")
    print(f"  phi_angles: {phi_angles.shape}")

    return data_nlsq, data_info

def test_nlsq_optimization(data_dict, data_info):
    """Test NLSQ optimization with correct API."""
    print("\n" + "="*60)
    print("Testing NLSQ (Optimistix) Optimization")
    print("="*60)

    # Create config manager with appropriate settings
    config_dict = {
        'analysis_mode': 'laminar_flow',
        'experiment': {
            'data_file_path': 'synthetic.h5',
            'wavevector_q': data_info['q'],
            'L': data_info['L'],
        },
        'optimization': {
            'analysis_mode': 'laminar_flow',
            'nlsq': {
                'max_iterations': 100,
                'tolerance': 1e-6,
            }
        }
    }

    config = ConfigManager(config_override=config_dict)

    try:
        start_time = time.time()
        result = fit_nlsq_jax(
            data=data_dict,
            config=config,
            initial_params=None  # Use default initial parameters
        )
        elapsed = time.time() - start_time

        print(f"\nNLSQ Results:")
        print(f"  Success: {result.success}")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Iterations: {result.n_iterations}")  # Fixed attribute name
        print(f"  Chi-squared: {result.chi_squared:.6f}")

        print(f"\nFitted Parameters:")
        param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
        for i, name in enumerate(param_names):
            if i < len(result.parameters):
                true_val = data_info['true_params'][name]
                fitted_val = result.parameters[i]
                error = abs(fitted_val - true_val) / abs(true_val + 1e-10) * 100
                print(f"  {name}: {fitted_val:.6f} (true: {true_val:.6f}, error: {error:.1f}%)")

        print(f"\nScaling Parameters:")
        print(f"  Contrast: {result.contrast:.6f} (true: {data_info['contrast']:.6f})")
        print(f"  Offset: {result.offset:.6f} (true: {data_info['offset']:.6f})")

        return result

    except Exception as e:
        print(f"NLSQ optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mcmc_optimization(data_info):
    """Test MCMC NUTS optimization with correct API."""
    print("\n" + "="*60)
    print("Testing MCMC NUTS (NumPyro) Optimization")
    print("="*60)

    try:
        start_time = time.time()

        # MCMC uses separate arrays, not a dictionary
        result = fit_mcmc_jax(
            data=data_info['c2_exp'],
            sigma=data_info['sigma'],
            t1=data_info['t1'],
            t2=data_info['t2'],
            phi=data_info['phi_angles'],
            q=data_info['q'],
            L=data_info['L'],
            analysis_mode='laminar_flow',
            n_warmup=100,
            n_samples=200,
            n_chains=2,
            backend='numpyro',  # Use NumPyro NUTS
            verbose=True
        )
        elapsed = time.time() - start_time

        print(f"\nMCMC Results:")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Chains: {result.n_chains if hasattr(result, 'n_chains') else 'N/A'}")
        print(f"  Samples per chain: {result.n_samples if hasattr(result, 'n_samples') else 'N/A'}")

        if hasattr(result, 'mean_params') and result.mean_params is not None:
            print(f"\nPosterior Means:")
            param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
            for i, name in enumerate(param_names):
                if i < len(result.mean_params):
                    true_val = data_info['true_params'][name]
                    mean_val = result.mean_params[i]
                    std_val = result.std_params[i] if hasattr(result, 'std_params') and result.std_params is not None else 0
                    error = abs(mean_val - true_val) / abs(true_val + 1e-10) * 100
                    print(f"  {name}: {mean_val:.6f} ± {std_val:.6f} (true: {true_val:.6f}, error: {error:.1f}%)")

        # Print available attributes for debugging
        print(f"\nAvailable result attributes:")
        for attr in dir(result):
            if not attr.startswith('_'):
                print(f"  - {attr}")

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
    data_nlsq, data_info = generate_synthetic_data(
        n_angles=2,      # Reduced for faster testing
        n_frames=30,     # Reduced for faster testing
        noise_level=0.01
    )

    # Test NLSQ
    nlsq_result = test_nlsq_optimization(data_nlsq, data_info)

    # Test MCMC
    mcmc_result = test_mcmc_optimization(data_info)

    # Compare results if both succeeded
    if nlsq_result and mcmc_result:
        print("\n" + "="*60)
        print("Comparison Summary")
        print("="*60)

        if hasattr(nlsq_result, 'chi_squared'):
            print(f"NLSQ Chi-squared: {nlsq_result.chi_squared:.6f}")

        if hasattr(mcmc_result, 'chi_squared'):
            print(f"MCMC Chi-squared: {mcmc_result.chi_squared:.6f}")
        elif hasattr(mcmc_result, 'log_likelihood'):
            print(f"MCMC Log-likelihood: {mcmc_result.log_likelihood:.6f}")

        # Parameter comparison if both have parameters
        if hasattr(nlsq_result, 'parameters') and hasattr(mcmc_result, 'mean_params'):
            print("\nParameter Agreement:")
            param_names = ['D0', 'alpha', 'D_offset']
            for i, name in enumerate(param_names[:3]):  # Just check first 3 params
                if i < min(len(nlsq_result.parameters), len(mcmc_result.mean_params)):
                    nlsq_val = nlsq_result.parameters[i]
                    mcmc_val = mcmc_result.mean_params[i]
                    diff = abs(nlsq_val - mcmc_val) / abs(nlsq_val + 1e-10) * 100
                    print(f"  {name}: NLSQ={nlsq_val:.6f}, MCMC={mcmc_val:.6f}, diff={diff:.1f}%")

    print("\n" + "="*60)
    print("Debug tests completed!")

if __name__ == "__main__":
    main()