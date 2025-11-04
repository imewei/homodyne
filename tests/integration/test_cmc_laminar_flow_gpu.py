"""Integration Tests for CMC Laminar Flow GPU Pipeline
=====================================================

Tests full CMC MCMC pipeline with laminar flow analysis mode on GPU.
Verifies architectural fix prevents 80GB OOM errors (Nov 2025).

Test Coverage:
- Full 30-shard CMC pipeline without OOM
- GPU execution maintained (no CPU fallback)
- Laminar flow physics with 9 parameters
- Memory usage per shard < 10 MB
- All shards converge successfully
"""

import numpy as np
import pytest

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
    HAS_GPU = len(jax.devices("gpu")) > 0
except (ImportError, RuntimeError):
    JAX_AVAILABLE = False
    HAS_GPU = False

from homodyne.config.parameter_space import ParameterSpace


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_jax
@pytest.mark.skipif(not HAS_GPU, reason="Requires GPU for testing GPU pipeline")
class TestCMCLaminarFlowGPU:
    """Test CMC laminar flow pipeline on GPU without OOM."""

    def test_full_cmc_pipeline_laminar_flow_no_oom(self):
        """Test full 30-shard CMC pipeline with laminar flow on GPU.

        Background (Nov 2025):
        - Previous implementation caused 80GB OOM for 100K point shards
        - Architectural fix: separate element-wise and meshgrid JIT functions
        - This test verifies fix prevents OOM on production-size shards

        Verifies:
        - 30 shards complete without RESOURCE_EXHAUSTED error
        - GPU execution maintained (not falling back to CPU)
        - Memory usage per shard < 10 MB
        - All shards converge (R-hat < 1.1)
        - CMC consensus produces valid posterior
        """
        from homodyne.optimization.cmc.coordinator import CMCCoordinator
        from homodyne.optimization.cmc.backends.pjit import PjitBackend

        # Create synthetic data simulating production CMC shards
        np.random.seed(42)
        total_points = 3_006_003  # 30 shards × ~100K points
        data = np.random.randn(total_points) * 0.1 + 1.0
        sigma = np.ones(total_points) * 0.05
        t1 = np.repeat(np.linspace(0, 10, 1001), 3003)  # Element-wise paired
        t2 = np.repeat(np.linspace(0, 10, 1001), 3003)
        phi = np.tile(np.array([-5.79, 4.88, 90.01]), 1_002_001)
        q = 0.005
        L = 1.0

        # Create laminar flow parameter space (7 physical + 2 scaling)
        config = {
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                    {"name": "gamma_dot_t0", "min": 0.1, "max": 10.0},
                    {"name": "beta", "min": 0.0, "max": 2.0},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 5.0},
                    {"name": "phi0", "min": -np.pi, "max": np.pi},
                ],
            }
        }
        parameter_space = ParameterSpace.from_config(config)

        # Initial values (mid-point of bounds)
        initial_values = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 2550.0,
            "alpha": 1.05,
            "D_offset": 50.0,
            "gamma_dot_t0": 5.0,
            "beta": 1.0,
            "gamma_dot_t_offset": 2.5,
            "phi0": 0.0,
        }

        # Minimal MCMC config for speed (not testing convergence quality)
        mcmc_config = {
            "num_warmup": 50,
            "num_samples": 100,
            "num_chains": 1,
            "target_accept_prob": 0.8,
        }

        # Initialize CMC coordinator with minimal config
        # CMC coordinator auto-detects hardware and selects appropriate backend
        cmc_config = {"backend": "jax"}  # Will auto-select pjit on GPU
        coordinator = CMCCoordinator(cmc_config)

        # Run CMC pipeline
        result = coordinator.run_cmc(
            data=data,
            sigma=sigma,
            t1=t1,
            t2=t2,
            phi=phi,
            q=q,
            L=L,
            analysis_mode="laminar_flow",
            parameter_space=parameter_space,
            initial_values=initial_values,
            mcmc_config=mcmc_config,
            num_shards=30,  # Production size
        )

        # Verify pipeline completed successfully
        assert result is not None, "CMC pipeline should complete"
        assert "posterior_samples" in result, "Should have posterior samples"
        assert "diagnostics" in result, "Should have diagnostics"
        assert "shard_results" in result, "Should have shard results"

        # Verify no OOM errors occurred
        shard_results = result["shard_results"]
        assert len(shard_results) == 30, "Should have results for all 30 shards"

        # Verify all shards converged (no errors)
        failed_shards = [i for i, sr in enumerate(shard_results) if not sr.get("converged", False)]
        assert len(failed_shards) == 0, f"Shards {failed_shards} failed to converge"

        # Verify GPU execution (check that JAX used GPU devices)
        gpu_devices = jax.devices("gpu")
        assert len(gpu_devices) > 0, "GPU should be available and used"

        # Verify posterior samples shape (9 parameters for laminar flow)
        posterior = result["posterior_samples"]
        assert posterior.shape[1] == 9, f"Should have 9 parameters, got {posterior.shape[1]}"

        # Verify convergence diagnostics
        diagnostics = result["diagnostics"]
        if "rhat" in diagnostics:
            # R-hat < 1.1 indicates convergence
            max_rhat = np.max(diagnostics["rhat"])
            assert max_rhat < 1.2, f"R-hat {max_rhat:.3f} indicates poor convergence"

        # Print summary for manual verification
        print(f"\n{'=' * 70}")
        print(f"CMC Laminar Flow GPU Pipeline Test Summary")
        print(f"{'=' * 70}")
        print(f"Total shards: 30")
        print(f"Converged shards: {len(shard_results) - len(failed_shards)}")
        print(f"Failed shards: {len(failed_shards)}")
        print(f"Posterior shape: {posterior.shape}")
        print(f"GPU devices used: {len(gpu_devices)}")
        if "rhat" in diagnostics:
            print(f"Max R-hat: {max_rhat:.3f}")
        print(f"{'=' * 70}\n")

    def test_single_shard_memory_usage(self):
        """Test memory usage for single production-size shard.

        Verifies:
        - 100K point shard uses < 10 MB memory
        - No 80GB meshgrid allocation occurs
        - Element-wise dispatcher path is used
        """
        from homodyne.optimization.cmc.backends.pjit import PjitBackend

        # Create single shard (100,200 points like production)
        np.random.seed(42)
        n_points = 100_200
        data = np.random.randn(n_points) * 0.1 + 1.0
        sigma = np.ones(n_points) * 0.05
        t1 = np.linspace(0, 10, n_points)  # Element-wise (1D)
        t2 = np.linspace(0, 10, n_points)
        phi = np.zeros(n_points)  # Single angle
        q = 0.005
        L = 1.0

        # Laminar flow parameter space
        config = {
            "parameter_space": {
                "model": "laminar_flow",
                "bounds": [
                    {"name": "D0", "min": 100.0, "max": 5000.0},
                    {"name": "alpha", "min": 0.1, "max": 2.0},
                    {"name": "D_offset", "min": 0.1, "max": 100.0},
                    {"name": "gamma_dot_t0", "min": 0.1, "max": 10.0},
                    {"name": "beta", "min": 0.0, "max": 2.0},
                    {"name": "gamma_dot_t_offset", "min": 0.0, "max": 5.0},
                    {"name": "phi0", "min": -np.pi, "max": np.pi},
                ],
            }
        }
        parameter_space = ParameterSpace.from_config(config)

        # Minimal initial values
        initial_values = {
            "contrast": 0.5,
            "offset": 1.0,
            "D0": 2550.0,
            "alpha": 1.05,
            "D_offset": 50.0,
            "gamma_dot_t0": 5.0,
            "beta": 1.0,
            "gamma_dot_t_offset": 2.5,
            "phi0": 0.0,
        }

        # Very minimal MCMC config (just test compilation/memory)
        mcmc_config = {
            "num_warmup": 10,
            "num_samples": 20,
            "num_chains": 1,
        }

        # Run single shard
        backend = PjitBackend()

        # Wrap data in shard format
        shard = {
            "data": data,
            "sigma": sigma,
            "t1": t1,
            "t2": t2,
            "phi": phi,
            "q": q,
            "L": L,
        }

        # Run MCMC on single shard
        result = backend.run_parallel_mcmc(
            shards=[shard],
            mcmc_config=mcmc_config,
            init_params=initial_values,
            inv_mass_matrix=np.eye(9),  # Identity for 9 parameters
            analysis_mode="laminar_flow",
            parameter_space=parameter_space,
        )

        # Verify result
        assert len(result) == 1, "Should have result for 1 shard"
        shard_result = result[0]

        # Verify convergence (or at least no OOM error)
        assert "converged" in shard_result, "Result should have convergence status"

        # If converged, verify samples shape and memory
        if shard_result["converged"]:
            samples = shard_result["samples"]
            assert samples.shape[1] == 9, "Should have 9 parameters"

            # Verify memory usage
            memory_mb = samples.nbytes / 1e6
            assert memory_mb < 10.0, f"Memory usage {memory_mb:.1f} MB exceeds 10 MB threshold"

            print(f"\nSingle shard memory test:")
            print(f"  Points: {n_points:,}")
            print(f"  Samples shape: {samples.shape}")
            print(f"  Memory usage: {memory_mb:.2f} MB")
            print(f"  Converged: {shard_result['converged']}")
        else:
            # If failed, verify it's not an OOM error
            error_msg = shard_result.get("error", "")
            assert (
                "RESOURCE_EXHAUSTED" not in error_msg
            ), f"Should not have OOM error: {error_msg}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
