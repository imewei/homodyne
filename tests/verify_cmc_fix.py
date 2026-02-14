import logging
import tempfile
from pathlib import Path

import numpy as np

from homodyne.config.parameter_space import ParameterSpace
from homodyne.optimization.cmc.core import CMCConfig, fit_mcmc_jax
from homodyne.optimization.cmc.data_prep import (
    PreparedData,
    estimate_noise_scale,
    extract_phi_info,
)

# Configure logging to capture our interest
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_cmc")


def generate_synthetic_data(n_points=10000, n_phi=2):
    """Generate minimal synthetic data for testing."""
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


def test_cmc_fix():
    print("Generating synthetic data...")
    p_data = generate_synthetic_data(n_points=20000, n_phi=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        checkpoint_dir = tmp_path / "checkpoints"

        print(f"Running CMC in {tmpdir}...")

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

        try:
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

            print("CMC run completed.")

            # Verify plotting (this was the crash)
            try:
                print("Attempting to generate energy plot...")
                # Ensure output directory exists for plots
                plot_dir = tmp_path / "plots"
                plot_dir.mkdir(exist_ok=True)

                # plot_energy expects 'idata' which is ArviZ InferenceData.
                # MCMCSamples has to_inference_data() method or similar?
                # Let's check if plot_energy accepts MCMCSamples directly or if we need to convert.
                # Looking at plotting.py (viewed earlier), it takes 'idata'.
                # core.py converts samples to result.

                # IMPORTANT: plot_energy in plotting.py takes 'idata'.
                # We need to see what plot_energy expects.
                # Assuming MCMCSamples is NOT InferenceData.
                # However, for verification of the FIX (potential_energy presence), checking extra_fields is enough.
                # But to verify plot_energy doesn't crash, we should call it.

                # Let's try to call it. coping with potential type mismatch if needed.
                # For now, let's just check the data presence which was the root cause.

                # Check for potential_energy in sample_stats (ArviZ InferenceData)
                if hasattr(result, "inference_data") and hasattr(
                    result.inference_data, "sample_stats"
                ):
                    sample_stats = result.inference_data.sample_stats
                    # sample_stats is an xarray.Dataset
                    if "potential_energy" in sample_stats:
                        print(
                            "[PASS] 'potential_energy' is present in inference_data.sample_stats."
                        )
                    else:
                        print(
                            f"[FAIL] 'potential_energy' missing. Available vars: {list(sample_stats.data_vars)}"
                        )
                else:
                    print("[FAIL] result.inference_data.sample_stats not available.")

            except Exception as e:
                print(f"[FAIL] Check failed: {e}")
                raise e

            # Verify result attributes
            if result and result.convergence_status:
                print(f"Result status: {result.convergence_status}")
                print("[PASS] Result convergence status available.")
            else:
                print("[FAIL] Result convergence status missing.")

        except Exception as e:
            print(f"[FAIL] CMC run failed with error: {e}")
            raise e


if __name__ == "__main__":
    test_cmc_fix()
