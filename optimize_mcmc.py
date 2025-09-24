#!/usr/bin/env python
"""
Optimized MCMC configuration for better performance.
"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro

# Enable parallel chains on CPU
numpyro.set_host_device_count(4)  # Use 4 CPU devices for parallel chains

def setup_mcmc_optimization():
    """Setup optimizations for MCMC sampling."""

    # 1. Configure JAX for better performance
    jax.config.update("jax_enable_x64", False)  # Use float32 for speed
    jax.config.update("jax_platform_name", "gpu" if jax.devices("gpu") else "cpu")

    # 2. Set NumPyro optimizations
    numpyro.set_platform("gpu" if jax.devices("gpu") else "cpu")

    # 3. Get device info
    devices = jax.devices()
    print(f"Available devices: {devices}")
    print(f"Number of devices: {len(devices)}")

    return len(devices)

def get_optimized_mcmc_config(n_devices=1):
    """Get optimized MCMC configuration based on available devices."""

    # Adjust chains based on devices
    n_chains = min(4, n_devices) if n_devices > 1 else 2

    config = {
        # Reduced iterations for faster testing
        'n_warmup': 200,      # Reduced from 500
        'n_samples': 500,     # Reduced from 1000
        'n_chains': n_chains,

        # NUTS specific settings
        'target_accept_prob': 0.8,  # Standard value
        'max_tree_depth': 8,        # Reduced from 10

        # Better initialization
        'init_strategy': 'median',  # Use median of prior

        # Thinning to reduce memory
        'thinning': 2,  # Keep every 2nd sample

        # Progress bar
        'progress_bar': True,

        # Backend
        'backend': 'numpyro',
    }

    return config

def create_informed_priors(true_params=None):
    """Create more informed priors based on problem knowledge."""

    if true_params is None:
        # Default reasonable priors for XPCS
        priors = {
            'contrast': (0.3, 0.05),     # Mean=0.3, std=0.05
            'offset': (1.0, 0.1),         # Mean=1.0, std=0.1
            'D0': (10000.0, 1000.0),      # Mean=10000, std=1000
            'alpha': (-1.5, 0.2),         # Mean=-1.5, std=0.2
            'D_offset': (100.0, 50.0),    # Mean=100, std=50
            'gamma_dot_0': (0.001, 0.0005),
            'beta': (0.0, 0.1),
            'gamma_dot_offset': (0.0, 0.01),
            'phi_0': (0.0, 5.0),
        }
    else:
        # Create tight priors around true values
        priors = {}
        for param, value in true_params.items():
            # Use 10% relative uncertainty
            std = abs(value * 0.1) if value != 0 else 0.1
            priors[param] = (value, std)

    return priors

def get_initial_values_from_nlsq(nlsq_result):
    """Extract initial values from NLSQ result for MCMC."""

    if nlsq_result is None or not hasattr(nlsq_result, 'parameters'):
        return None

    # Convert NLSQ parameters to MCMC initial values
    init_values = {
        'contrast': nlsq_result.contrast if hasattr(nlsq_result, 'contrast') else 0.3,
        'offset': nlsq_result.offset if hasattr(nlsq_result, 'offset') else 1.0,
    }

    # Add physical parameters
    param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0']
    for i, name in enumerate(param_names):
        if i < len(nlsq_result.parameters):
            init_values[name] = float(nlsq_result.parameters[i])

    return init_values

def test_optimized_mcmc():
    """Test optimized MCMC configuration."""

    print("Setting up MCMC optimizations...")
    n_devices = setup_mcmc_optimization()

    print("\nOptimized MCMC configuration:")
    config = get_optimized_mcmc_config(n_devices)
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\nInformed priors:")
    priors = create_informed_priors()
    for param, (mean, std) in priors.items():
        print(f"  {param}: N({mean:.4f}, {std:.4f})")

    print("\nOptimization complete!")
    return config, priors

if __name__ == "__main__":
    test_optimized_mcmc()