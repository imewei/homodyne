"""Reproduction of ConcretizationTypeError under pmap.

This script simulates the multiprocessing worker environment:
- Sets XLA_FLAGS for 4 virtual devices
- Uses the EXACT production model (xpcs_model_scaled) and ParameterSpace
- Runs MCMC with chain_method="parallel" (pmap) via run_nuts_sampling
"""

import os

# Set XLA device count BEFORE importing JAX
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=4 --xla_disable_hlo_passes=constant_folding"
)
os.environ["OMP_NUM_THREADS"] = "2"

import traceback

import jax
import jax.numpy as jnp
import numpy as np

# Ensure 4 devices
print(f"JAX devices: {jax.device_count()}")

from homodyne.config.parameter_space import ParameterSpace  # noqa: E402
from homodyne.optimization.cmc.config import CMCConfig  # noqa: E402
from homodyne.optimization.cmc.model import get_xpcs_model  # noqa: E402
from homodyne.optimization.cmc.sampler import run_nuts_sampling  # noqa: E402


def main():
    # Create minimal test data
    n_points = 500
    rng = np.random.default_rng(42)

    dt = 0.5
    t_max = 300.0
    t1 = rng.uniform(dt, t_max, n_points)
    t2 = rng.uniform(dt, t_max, n_points)

    time_grid = np.linspace(0, t_max, int(t_max / dt) + 1)
    phi_unique = np.array([0.0])
    phi_indices = np.zeros(n_points, dtype=np.int32)
    data = rng.normal(1.0, 0.02, n_points)

    # Build ParameterSpace from config
    config_dict = {
        "bounds": {
            "D0": [1.0, 50000.0],
            "alpha": [-3.0, 0.0],
            "D_offset": [0.0, 100.0],
            "contrast": [0.0, 1.0],
            "offset": [0.0, 2.0],
        },
        "priors": {
            "D0": {"type": "normal", "mu": 16400.0, "sigma": 171.9},
            "alpha": {"type": "normal", "mu": -1.566, "sigma": 0.002665},
            "D_offset": {"type": "normal", "mu": 2.994, "sigma": 0.02204},
        },
    }
    parameter_space = ParameterSpace.from_config(
        config_dict=config_dict,
        analysis_mode="static",
    )

    # Get the EXACT model function used in production
    model_fn = get_xpcs_model(per_angle_mode="individual", use_reparameterization=False)
    print(f"Model function: {model_fn.__name__}")

    model_kwargs = {
        "data": jnp.array(data),
        "t1": jnp.array(t1),
        "t2": jnp.array(t2),
        "phi_unique": jnp.array(phi_unique),
        "phi_indices": jnp.array(phi_indices),
        "q": 0.0237,
        "L": 2e6,
        "dt": dt,
        "time_grid": jnp.array(time_grid),
        "analysis_mode": "static",
        "parameter_space": parameter_space,
        "n_phi": 1,
        "noise_scale": 0.02,
        "num_shards": 6,
    }

    # Initial values from NLSQ
    initial_values = {
        "D0": 16400.0,
        "alpha": -1.566,
        "D_offset": 2.994,
        "contrast_0": 0.05,
        "offset_0": 1.001,
    }

    # Create CMC config
    cmc_config = CMCConfig(
        num_chains=4,
        num_warmup=10,
        num_samples=10,
        target_accept_prob=0.8,
        max_tree_depth=10,
        adaptive_sampling=False,
    )

    rng_key = jax.random.PRNGKey(0)

    try:
        samples, stats = run_nuts_sampling(
            model=model_fn,
            model_kwargs=model_kwargs,
            config=cmc_config,
            initial_values=initial_values,
            parameter_space=parameter_space,
            n_phi=1,
            analysis_mode="static",
            rng_key=rng_key,
            progress_bar=True,
            per_angle_mode="individual",
        )
        print("SUCCESS: MCMC completed without error")
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("\n=== FULL TRACEBACK ===")
        traceback.print_exc()


if __name__ == "__main__":
    main()
