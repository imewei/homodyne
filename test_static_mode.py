#!/usr/bin/env python
"""
Simplified test using static mode (3 parameters) instead of laminar flow (7 parameters).
This should be easier to debug and converge.
"""

import numpy as np
import jax
import jax.numpy as jnp
import time
import numpyro
from homodyne.optimization import fit_nlsq_jax, fit_mcmc_jax
from homodyne.config import ConfigManager
from optimize_mcmc import setup_mcmc_optimization, get_optimized_mcmc_config

# Enable optimizations
numpyro.set_host_device_count(4)

def generate_simple_synthetic_data(n_angles=2, n_frames=20, noise_level=0.01):
    """Generate simple synthetic data for static mode testing."""
    print("Generating simple synthetic data for static mode...")

    # True parameters for static mode (only 3 parameters!)
    true_params = {
        'D0': 5000.0,      # Diffusion coefficient
        'alpha': -1.0,     # Anomalous exponent
        'D_offset': 50.0,  # Offset
    }

    # Simple time arrays
    dt = 0.1
    times = np.linspace(0, dt * n_frames, n_frames)
    t1, t2 = np.meshgrid(times, times, indexing='ij')

    # Simple angle array
    phi_angles = np.array([0.0, 45.0])  # Just 2 angles

    # Physical parameters
    q = 0.005  # wavevector
    L = 100.0  # sample-detector distance
    contrast = 0.25  # Lower contrast
    offset = 1.0

    # Generate simple exponential decay
    c2_exp = []
    sigma = []

    for phi in phi_angles:
        # Simple model: exponential decay
        dt_matrix = np.abs(t2 - t1)
        D_eff = true_params['D0'] * dt_matrix**true_params['alpha'] + true_params['D_offset']
        g1 = np.exp(-q**2 * D_eff)
        c2 = offset + contrast * g1**2

        # Add small noise
        noise = np.random.normal(0, noise_level, c2.shape)
        c2_noisy = c2 + noise

        c2_exp.append(c2_noisy)
        sigma.append(np.ones_like(c2) * noise_level)

    c2_exp = np.array(c2_exp)
    sigma = np.array(sigma)

    data_dict = {
        'c2_exp': c2_exp,
        'sigma': sigma,
        't1': t1,
        't2': t2,
        'phi_angles_list': phi_angles,
        'wavevector_q_list': np.array([q]),
    }

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

    print(f"Generated static mode data:")
    print(f"  c2_exp shape: {c2_exp.shape}")
    print(f"  Parameters: {len(true_params)} (static mode)")
    print(f"  True values: D0={true_params['D0']:.1f}, α={true_params['alpha']:.2f}, D_offset={true_params['D_offset']:.1f}")

    return data_dict, data_info

def test_nlsq_static(data_dict, data_info):
    """Test NLSQ with static mode."""
    print("\n" + "="*60)
    print("Testing NLSQ with Static Mode (3 parameters)")
    print("="*60)

    # Simple static mode configuration
    config_dict = {
        'analysis_mode': 'static_isotropic',  # STATIC MODE
        'experiment': {
            'data_file_path': 'synthetic.h5',
            'wavevector_q': data_info['q'],
            'L': data_info['L'],
        },
        'optimization': {
            'analysis_mode': 'static_isotropic',
            'nlsq': {
                'max_iterations': 50,  # Fewer iterations needed
                'tolerance': 1e-5,
            },
            'parameter_bounds': {
                'D0': [100.0, 10000.0],      # Tighter bounds
                'alpha': [-2.0, 0.0],        # Negative only
                'D_offset': [0.0, 200.0],    # Positive only
            }
        }
    }

    config = ConfigManager(config_override=config_dict)

    try:
        start_time = time.time()
        result = fit_nlsq_jax(
            data=data_dict,
            config=config,
            initial_params={
                'D0': 4000.0,      # Close to true value
                'alpha': -1.2,     # Close to true value
                'D_offset': 40.0,  # Close to true value
            }
        )
        elapsed = time.time() - start_time

        print(f"\nNLSQ Results:")
        print(f"  Success: {result.success}")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Iterations: {result.n_iterations}")
        print(f"  Chi-squared: {result.chi_squared:.6f}")

        if result.parameters is not None and len(result.parameters) > 0:
            print(f"\nFitted Parameters (Static Mode):")
            param_names = ['D0', 'alpha', 'D_offset']
            for i, name in enumerate(param_names):
                if i < len(result.parameters):
                    true_val = data_info['true_params'][name]
                    fitted_val = result.parameters[i]
                    error = abs(fitted_val - true_val) / abs(true_val + 1e-10) * 100
                    print(f"  {name}: {fitted_val:.4f} (true: {true_val:.4f}, error: {error:.1f}%)")

            print(f"\nScaling Parameters:")
            print(f"  Contrast: {result.contrast:.4f} (true: {data_info['contrast']:.4f})")
            print(f"  Offset: {result.offset:.4f} (true: {data_info['offset']:.4f})")

        return result

    except Exception as e:
        print(f"NLSQ optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mcmc_static(data_info, nlsq_result=None):
    """Test MCMC with static mode."""
    print("\n" + "="*60)
    print("Testing MCMC NUTS with Static Mode (3 parameters)")
    print("="*60)

    # Get optimized configuration
    n_devices = setup_mcmc_optimization()
    mcmc_config = get_optimized_mcmc_config(n_devices)

    # Initialize from NLSQ if available
    init_params = None
    if nlsq_result and hasattr(nlsq_result, 'parameters'):
        init_params = {
            'D0': float(nlsq_result.parameters[0]) if len(nlsq_result.parameters) > 0 else 5000.0,
            'alpha': float(nlsq_result.parameters[1]) if len(nlsq_result.parameters) > 1 else -1.0,
            'D_offset': float(nlsq_result.parameters[2]) if len(nlsq_result.parameters) > 2 else 50.0,
        }
        print(f"Initializing MCMC from NLSQ results: {init_params}")

    try:
        start_time = time.time()

        # Run MCMC with static mode
        result = fit_mcmc_jax(
            data=data_info['c2_exp'],
            sigma=data_info['sigma'],
            t1=data_info['t1'],
            t2=data_info['t2'],
            phi=data_info['phi_angles'],
            q=data_info['q'],
            L=data_info['L'],
            analysis_mode='static_isotropic',  # STATIC MODE
            n_warmup=mcmc_config['n_warmup'],
            n_samples=mcmc_config['n_samples'],
            n_chains=mcmc_config['n_chains'],
            backend='numpyro',
            verbose=True,
            initial_params=init_params,  # Use NLSQ results
        )
        elapsed = time.time() - start_time

        print(f"\nMCMC Results:")
        print(f"  Time: {elapsed:.3f} seconds")
        print(f"  Chains: {result.n_chains if hasattr(result, 'n_chains') else 'N/A'}")
        print(f"  Samples: {result.n_samples if hasattr(result, 'n_samples') else 'N/A'}")

        if hasattr(result, 'mean_params') and result.mean_params is not None:
            print(f"\nPosterior Means (Static Mode):")
            param_names = ['D0', 'alpha', 'D_offset']
            for i, name in enumerate(param_names):
                if i < len(result.mean_params):
                    true_val = data_info['true_params'][name]
                    mean_val = result.mean_params[i]
                    std_val = result.std_params[i] if hasattr(result, 'std_params') and result.std_params is not None else 0
                    error = abs(mean_val - true_val) / abs(true_val + 1e-10) * 100
                    print(f"  {name}: {mean_val:.4f} ± {std_val:.4f} (true: {true_val:.4f}, error: {error:.1f}%)")

            if hasattr(result, 'mean_contrast'):
                print(f"\nScaling Parameters:")
                print(f"  Contrast: {result.mean_contrast:.4f} ± {result.std_contrast:.4f} (true: {data_info['contrast']:.4f})")
                print(f"  Offset: {result.mean_offset:.4f} ± {result.std_offset:.4f} (true: {data_info['offset']:.4f})")

        return result

    except Exception as e:
        print(f"MCMC optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run static mode tests."""
    print("Starting Static Mode Tests (Simpler 3-parameter model)")
    print("="*60)

    # Set random seed
    np.random.seed(42)

    # Generate simple data
    data_dict, data_info = generate_simple_synthetic_data(
        n_angles=2,
        n_frames=20,  # Even smaller for faster testing
        noise_level=0.01
    )

    # Test NLSQ with static mode
    nlsq_result = test_nlsq_static(data_dict, data_info)

    # Test MCMC with static mode, initialized from NLSQ
    mcmc_result = test_mcmc_static(data_info, nlsq_result)

    # Compare results
    if nlsq_result and mcmc_result:
        print("\n" + "="*60)
        print("Static Mode Comparison Summary")
        print("="*60)

        if hasattr(nlsq_result, 'chi_squared'):
            print(f"NLSQ Chi-squared: {nlsq_result.chi_squared:.6f}")

        if hasattr(mcmc_result, 'acceptance_rate'):
            print(f"MCMC Acceptance rate: {mcmc_result.acceptance_rate:.3f}")

        # Compare parameters
        if hasattr(nlsq_result, 'parameters') and hasattr(mcmc_result, 'mean_params'):
            print("\nParameter Agreement (Static Mode):")
            param_names = ['D0', 'alpha', 'D_offset']
            for i, name in enumerate(param_names):
                if i < min(len(nlsq_result.parameters), len(mcmc_result.mean_params)):
                    nlsq_val = nlsq_result.parameters[i]
                    mcmc_val = mcmc_result.mean_params[i]
                    true_val = data_info['true_params'][name]
                    print(f"  {name}:")
                    print(f"    True:  {true_val:.4f}")
                    print(f"    NLSQ:  {nlsq_val:.4f}")
                    print(f"    MCMC:  {mcmc_val:.4f}")

    print("\n" + "="*60)
    print("Static mode tests completed!")

if __name__ == "__main__":
    main()