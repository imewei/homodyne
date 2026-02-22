"""Regression test for CMC potential energy availability.

Migrated from tests/verify_cmc_fix.py.
Ensures that 'potential_energy' is available in the inference data sample stats
after a CMC run.
"""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Skip if arviz is not installed (CMC requires it)
az = pytest.importorskip("arviz")

from homodyne.config.parameter_space import ParameterSpace  # noqa: E402
from homodyne.optimization.cmc.core import CMCConfig, fit_mcmc_jax  # noqa: E402
from homodyne.optimization.cmc.data_prep import (  # noqa: E402
    PreparedData,
    estimate_noise_scale,
    extract_phi_info,
)

# Configure logging to capture our interest
logger = logging.getLogger("test_cmc_potential_energy")


def generate_synthetic_data(n_points=1000, n_phi=2):
    """Generate minimal synthetic data for testing.

    Scaled down from original 20k points for faster regression testing.
    """
    rng = np.random.default_rng(42)

    # 2 angles
    phi = np.repeat(np.linspace(0, 1, n_phi), n_points // n_phi)

    # Random time points
    t1 = rng.uniform(0, 1, n_points)
    t2 = rng.uniform(0, 1, n_points)

    # Fake correlations
    g2 = np.exp(-10 * (t1 - t2) ** 2) + rng.normal(0, 0.01, n_points)

    phi_unique, phi_indices = extract_phi_info(phi)
    noise_scale = estimate_noise_scale(g2)

    return PreparedData(
        data=g2,
        t1=t1,
        t2=t2,
        phi=phi,
        phi_unique=phi_unique,
        phi_indices=phi_indices,
        n_total=n_points,
        n_phi=n_phi,
        noise_scale=noise_scale,
    )


@pytest.mark.regression
@pytest.mark.slow
def test_cmc_potential_energy_availability():
    """Verify that energy (potential_energy) is present in CMC results."""
    # Use fewer points for regression test speed
    p_data = generate_synthetic_data(n_points=2000, n_phi=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoints"

        # Configure for a very fast run
        config = CMCConfig(
            enable=True,
            sharding_strategy="random",  # Explicitly test random sharding
            num_shards=2,  # Force 2 shards
            num_warmup=10,
            num_samples=10,
            num_chains=1,
            backend_name="multiprocessing",  # Use multiprocessing backend
            checkpoint_dir=str(checkpoint_dir),
            per_angle_mode="constant",  # Simplify model for speed
        )

        # Create default parameter space
        parameter_space = ParameterSpace.from_defaults("static")

        result = fit_mcmc_jax(
            data=p_data.data,
            t1=p_data.t1,
            t2=p_data.t2,
            phi=p_data.phi,
            q=0.01,  # Dummy q
            L=2e6,  # Dummy L
            config={
                "enable": True,
                "sharding": {"strategy": "random", "num_shards": 2},
                "backend_config": {"name": "multiprocessing"},
                "per_shard_mcmc": {
                    "num_warmup": 10,
                    "num_samples": 10,
                    "num_chains": 1,
                },
                "combination": {"method": "consensus_mc"},
            },
            cmc_config=config.to_dict(),
            dt=0.1,  # Dummy dt
            output_dir=tmp_path,
            run_id="test_run",
            analysis_mode="static",
            parameter_space=parameter_space,  # Pass valid parameter space
        )

        assert result is not None, "CMC result should not be None"
        assert result.convergence_status, "Result convergence status missing"

        # Verify inference_data structure
        assert hasattr(result, "inference_data"), "Result missing inference_data"
        assert hasattr(result.inference_data, "sample_stats"), (
            "inference_data missing sample_stats"
        )

        sample_stats = result.inference_data.sample_stats

        # KEY CHECK: energy must be present (ArviZ convention; NumPyro's
        # "potential_energy" is mapped to "energy" during InferenceData creation)
        assert "energy" in sample_stats, (
            f"'energy' missing. Available vars: {list(sample_stats.data_vars)}"
        )
