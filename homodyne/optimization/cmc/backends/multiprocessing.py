"""Multiprocessing Backend for Consensus Monte Carlo
==================================================

This backend implements parallel MCMC execution using Python's multiprocessing.Pool
for CPU-based parallelization. It's optimized for multi-core workstations and
single-node CPU clusters.

Execution Strategy
------------------
- Uses multiprocessing.Pool to create worker processes
- Each worker executes MCMC on one shard independently
- Parallelism limited by number of CPU cores
- Handles serialization of JAX functions using cloudpickle

Key Features
------------
- Full CPU core utilization (up to os.cpu_count())
- Timeout detection (30 min per shard by default)
- Graceful shutdown on errors
- Worker process isolation for fault tolerance
- Progress tracking via logging

Implementation Details
----------------------
- Worker function runs in separate process
- Each worker imports necessary modules independently
- Results collected via multiprocessing.Pool.map
- Failed shards return error dict (don't crash entire pipeline)

Serialization Challenges
------------------------
JAX functions and NumPyro models are difficult to serialize with standard pickle.
We use cloudpickle for better serialization support, but some objects (e.g.,
JIT-compiled functions) may still fail. Workers import and recreate these objects
locally to avoid serialization issues.

Usage Example
-------------
    from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend

    backend = MultiprocessingBackend(num_workers=8, timeout_minutes=30)
    results = backend.run_parallel_mcmc(
        shards=data_shards,
        mcmc_config={'num_warmup': 500, 'num_samples': 2000},
        init_params={'D0': 1000.0, 'alpha': 0.5},
        inv_mass_matrix=mass_matrix,
    )

Integration Points
------------------
- Called by CMC coordinator via select_backend()
- Uses same MCMC implementation as pjit backend
- Integrates with CheckpointManager for fault tolerance
- Returns results in standard format for combination.py

Performance Considerations
--------------------------
- Startup overhead: ~1-2 seconds per worker process
- Memory usage: ~2-4GB per worker (depends on shard size)
- CPU overhead: ~10% for multiprocessing coordination
- Best for: 8+ core systems with large memory

Error Handling
--------------
- Worker crashes: Logged and returned as failed shard
- Timeouts: Detected and terminated gracefully
- Serialization errors: Caught and reported with helpful message
- Pool shutdown: Always ensures clean cleanup
"""

from typing import List, Dict, Any, Optional
import multiprocessing
import time
import os

import numpy as np

from homodyne.optimization.cmc.backends.base import CMCBackend
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Try to import cloudpickle for better serialization
try:
    import cloudpickle

    CLOUDPICKLE_AVAILABLE = True
except ImportError:
    CLOUDPICKLE_AVAILABLE = False
    logger.warning(
        "cloudpickle not available. Multiprocessing backend may have "
        "serialization issues with complex objects. "
        "Install with: pip install cloudpickle"
    )


class MultiprocessingBackend(CMCBackend):
    """Python multiprocessing backend for parallel MCMC execution on CPU.

    This backend uses multiprocessing.Pool to execute MCMC sampling on
    multiple CPU cores in parallel. It's optimized for multi-core workstations
    and provides good CPU utilization.

    Attributes
    ----------
    num_workers : int
        Number of worker processes (default: cpu_count)
    timeout_minutes : float
        Timeout per shard in minutes (default: 30)
    use_cloudpickle : bool
        Whether cloudpickle is available for serialization

    Methods
    -------
    run_parallel_mcmc(shards, mcmc_config, init_params, inv_mass_matrix)
        Execute MCMC on all shards using multiprocessing.Pool
    get_backend_name()
        Return 'multiprocessing'

    Notes
    -----
    - Requires Python 3.8+ for proper multiprocessing support
    - Each worker runs in separate process (memory isolation)
    - Workers are created at pool initialization (startup overhead)
    - Pool is reused across calls (if instance is kept alive)
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        timeout_minutes: float = 30.0,
        max_memory_per_worker_gb: float = 8.0,  # Increased to match actual usage (7.96 GB observed)
    ):
        """Initialize multiprocessing backend.

        Parameters
        ----------
        num_workers : int, optional
            Number of worker processes. If None, uses memory-aware calculation
        timeout_minutes : float, optional
            Timeout per shard in minutes, by default 30.0
        max_memory_per_worker_gb : float, optional
            Maximum memory per worker in GB, by default 4.0
        """
        # Calculate memory-aware worker count if not specified
        if num_workers is None:
            num_workers = self._calculate_safe_worker_count(max_memory_per_worker_gb)

        self.num_workers = num_workers
        self.max_concurrent_shards = num_workers  # Maximum shards to run at once
        self.timeout_seconds = timeout_minutes * 60
        self.use_cloudpickle = CLOUDPICKLE_AVAILABLE

        logger.info(
            f"Multiprocessing backend initialized: {self.num_workers} workers, "
            f"{timeout_minutes:.1f} min timeout per shard, "
            f"max {self.max_concurrent_shards} concurrent shards"
        )

        if not self.use_cloudpickle:
            logger.warning(
                "Running without cloudpickle. Some complex objects may not serialize. "
                "Install cloudpickle for better compatibility."
            )

    def _calculate_safe_worker_count(self, max_memory_per_worker_gb: float) -> int:
        """Calculate safe number of workers based on available memory.

        Parameters
        ----------
        max_memory_per_worker_gb : float
            Maximum memory per worker in GB

        Returns
        -------
        int
            Safe number of workers
        """
        try:
            import psutil

            # Get available memory (in GB)
            available_memory_gb = psutil.virtual_memory().available / (1024**3)

            # Calculate workers based on memory
            memory_based_workers = max(1, int(available_memory_gb / max_memory_per_worker_gb))

            # Get CPU count (physical cores only, not logical/hyperthreading)
            cpu_count = psutil.cpu_count(logical=False) or 1

            # Use minimum of memory-based and CPU-based limits
            # Reserve more workers for OS and be conservative (use 60% of physical CPUs)
            cpu_workers = max(1, int(cpu_count * 0.6))

            safe_workers = min(memory_based_workers, cpu_workers)

            logger.info(
                f"Memory-aware worker calculation: {available_memory_gb:.1f}GB available, "
                f"{memory_based_workers} memory-based workers, "
                f"{cpu_workers} CPU-based workers (physical cores × 0.6), "
                f"selected {safe_workers} workers"
            )

            return safe_workers

        except ImportError:
            # Fallback if psutil not available
            cpu_workers = max(1, int((os.cpu_count() or 1) * 0.8))
            logger.warning(
                f"psutil not available, using CPU-based worker count: {cpu_workers}"
            )
            return cpu_workers

    def get_backend_name(self) -> str:
        """Return backend name 'multiprocessing'."""
        return "multiprocessing"

    def run_parallel_mcmc(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space,
    ) -> List[Dict[str, Any]]:
        """Run MCMC on all shards using multiprocessing.Pool.

        Creates a pool of worker processes and distributes shards across them.
        Each worker executes MCMC independently and returns results.

        Parameters
        ----------
        shards : list of dict
            Data shards to process
        mcmc_config : dict
            MCMC configuration (num_warmup, num_samples, etc.)
        init_params : dict
            Initial parameter values for MCMC chain initialization
            (loaded from config: `initial_parameters.values`)
        inv_mass_matrix : np.ndarray
            Inverse mass matrix for NUTS initialization
            (typically identity matrix, adapted during warmup)
        analysis_mode : str
            Analysis mode ("static_isotropic" or "laminar_flow")
        parameter_space : ParameterSpace
            Parameter space with bounds and constraints

        Returns
        -------
        list of dict
            Per-shard MCMC results
        """
        logger.info(
            f"Starting multiprocessing backend execution for {len(shards)} shards "
            f"using {self.num_workers} workers"
        )

        # Validate analysis_mode consistency before execution
        self._validate_analysis_mode_consistency(analysis_mode, parameter_space)

        # Prepare arguments for workers
        worker_args = [
            (
                i,
                shard,
                mcmc_config,
                init_params,
                inv_mass_matrix,
                analysis_mode,
                parameter_space,
            )
            for i, shard in enumerate(shards)
        ]

        # Create multiprocessing pool
        pool = None
        try:
            # Use 'spawn' method for better cross-platform compatibility
            # and to avoid issues with JAX/NumPyro in forked processes
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=self.num_workers)

            logger.info(f"Created pool with {self.num_workers} worker processes")

            # PARALLEL EXECUTION FIX (Nov 2025): Submit all jobs first, then collect results
            # Previous code used apply_async() + immediate get() which blocked (sequential execution)
            # New approach: Submit all → wait for all → true parallelization

            # MEMORY-SAFE BATCHED EXECUTION (Nov 2025)
            # Process shards in batches to limit concurrent memory usage
            # Each batch contains at most max_concurrent_shards shards

            batch_size = self.max_concurrent_shards
            num_batches = (len(worker_args) + batch_size - 1) // batch_size

            logger.info(
                f"Processing {len(worker_args)} shards in {num_batches} batches "
                f"(max {batch_size} concurrent shards per batch)"
            )

            results = []

            # Process shards in batches
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(worker_args))
                batch_args = worker_args[batch_start:batch_end]

                logger.info(
                    f"Starting batch {batch_idx + 1}/{num_batches}: "
                    f"shards {batch_start} to {batch_end - 1} "
                    f"({len(batch_args)} shards)"
                )

                # Phase 1: Submit batch jobs asynchronously
                async_results = []
                start_times = []
                for i, args in enumerate(batch_args):
                    shard_idx = batch_start + i
                    self._log_shard_start(shard_idx, len(shards))
                    start_time = self._create_timer()
                    start_times.append(start_time)

                    # Submit job without blocking
                    async_result = pool.apply_async(_worker_function, (args,))
                    async_results.append(async_result)

                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches}: "
                    f"{len(async_results)} shards submitted, waiting for completion..."
                )

                # Phase 2: Collect batch results as they complete
                for i, (async_result, start_time) in enumerate(zip(async_results, start_times)):
                    shard_idx = batch_start + i
                    try:
                        # Wait for this specific result with timeout
                        result = async_result.get(timeout=self.timeout_seconds)

                        # Add timing
                        result["elapsed_time"] = self._get_elapsed_time(start_time)

                        # Validate result
                        self._validate_shard_result(result, shard_idx)

                        # Log error details if shard failed
                        if not result["converged"] and result.get("error"):
                            logger.error(
                                f"[{self.get_backend_name()}] Shard {shard_idx+1}/{len(shards)} error details:\n"
                                f"{result['error']}"
                            )

                        # Log completion
                        self._log_shard_complete(
                            shard_idx, len(shards), result["elapsed_time"], result["converged"]
                        )

                        results.append(result)

                    except multiprocessing.TimeoutError:
                        # Handle timeout
                        elapsed = self._get_elapsed_time(start_time)
                        error_msg = (
                            f"Shard {shard_idx} timed out after {self.timeout_seconds:.0f}s "
                            f"({self.timeout_seconds / 60:.1f} min)"
                        )
                        logger.error(f"[{self.get_backend_name()}] {error_msg}")

                        error_result = {
                            "converged": False,
                            "error": error_msg,
                            "elapsed_time": elapsed,
                            "samples": None,
                            "diagnostics": {},
                            "shard_idx": shard_idx,
                        }
                        results.append(error_result)

                    except Exception as e:
                        # Handle other errors
                        error_result = self._handle_shard_error(e, shard_idx)
                        error_result["elapsed_time"] = self._get_elapsed_time(start_time)
                        results.append(error_result)

                logger.info(
                    f"Batch {batch_idx + 1}/{num_batches} complete: "
                    f"{len(results)}/{len(shards)} total shards processed"
                )

            logger.info(
                f"All batches complete: {len(results)}/{len(shards)} shards processed"
            )
            return results

        finally:
            # Clean up pool
            if pool is not None:
                logger.info("Shutting down worker pool...")
                pool.close()
                pool.join()
                logger.info("Worker pool shutdown complete")

    def run_parallel_mcmc_batch(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space,
    ) -> List[Dict[str, Any]]:
        """Alternative implementation using Pool.map for batch execution.

        This method processes all shards in a single batch using Pool.map,
        which may be more efficient for large numbers of shards.

        Note: This is an alternative to run_parallel_mcmc. Not currently used.

        Parameters
        ----------
        shards : list of dict
            Data shards
        mcmc_config : dict
            MCMC configuration
        init_params : dict
            Initial parameters
        inv_mass_matrix : np.ndarray
            Mass matrix
        analysis_mode : str
            Analysis mode ("static_isotropic" or "laminar_flow")
        parameter_space : ParameterSpace
            Parameter space with bounds and constraints

        Returns
        -------
        list of dict
            Per-shard results
        """
        logger.info(
            f"Starting batch execution for {len(shards)} shards "
            f"using {self.num_workers} workers"
        )

        # Prepare arguments
        worker_args = [
            (
                i,
                shard,
                mcmc_config,
                init_params,
                inv_mass_matrix,
                analysis_mode,
                parameter_space,
            )
            for i, shard in enumerate(shards)
        ]

        # Create pool and execute
        pool = None
        try:
            ctx = multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=self.num_workers)

            # Execute all shards
            results = pool.map(_worker_function, worker_args)

            # Validate results
            for i, result in enumerate(results):
                self._validate_shard_result(result, i)

            return results

        finally:
            if pool is not None:
                pool.close()
                pool.join()


# -----------------------------------------------------------------------------
# Worker Function (executed in separate process)
# -----------------------------------------------------------------------------


def _worker_function(args: tuple) -> Dict[str, Any]:
    """Worker function executed in separate process.

    This function runs in a separate process and executes MCMC on a single
    shard. It imports necessary modules locally to avoid serialization issues.

    Parameters
    ----------
    args : tuple
        (shard_idx, shard_data, mcmc_config, init_params, inv_mass_matrix, analysis_mode, parameter_space)

    Returns
    -------
    dict
        Shard result with samples and diagnostics
    """
    (
        shard_idx,
        shard,
        mcmc_config,
        init_params,
        inv_mass_matrix,
        analysis_mode,
        parameter_space,
    ) = args

    # Import modules locally (avoids serialization issues)
    import numpy as np

    # Configure XLA for memory-constrained environments (before JAX import)
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

    import jax
    import jax.numpy as jnp

    try:
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from numpyro import sample
    except ImportError as e:
        return {
            "converged": False,
            "error": f"NumPyro not available in worker: {str(e)}",
            "samples": None,
            "diagnostics": {},
            "shard_idx": shard_idx,
        }

    try:
        # Import the correct model creation function from mcmc.py
        from homodyne.optimization.mcmc import _create_numpyro_model

        # Extract MCMC configuration
        num_warmup = mcmc_config.get("num_warmup", 500)
        num_samples = mcmc_config.get("num_samples", 2000)
        num_chains = mcmc_config.get("num_chains", 1)
        target_accept_prob = mcmc_config.get("target_accept_prob", 0.8)
        max_tree_depth = mcmc_config.get("max_tree_depth", 10)

        # Convert data to JAX arrays
        data_jax = jnp.array(shard["data"])
        sigma_jax = jnp.array(shard.get("sigma", np.ones_like(shard["data"])))
        t1_jax = jnp.array(shard["t1"])
        t2_jax = jnp.array(shard["t2"])
        phi_jax = jnp.array(shard["phi"])
        q = float(shard["q"])
        L = float(shard["L"])

        # Compute dt from t1 array (required for physics computation)
        if len(t1_jax) > 1:
            dt = float(np.median(np.diff(np.unique(t1_jax))))
        else:
            dt = 1.0  # Fallback

        # CRITICAL FIX (Nov 2025): Pre-compute unique phi values to prevent 80GB OOM error
        # CMC shards contain replicated phi arrays (e.g., 100K elements for 3 unique angles)
        # Extract unique values before JIT compilation (NumPyro model creation)
        phi_unique = np.unique(np.asarray(phi_jax))
        logger.debug(
            f"Multiprocessing shard {shard_idx}: Extracted {len(phi_unique)} unique phi values from "
            f"{len(phi_jax)} replicated elements (memory reduction: "
            f"{len(phi_jax)}→{len(phi_unique)}, {len(phi_jax) / len(phi_unique):.1f}x)"
        )

        # Create proper NumPyro model with actual XPCS physics
        model = _create_numpyro_model(
            data=data_jax,
            sigma=sigma_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi=phi_unique,  # Use unique phi values for theory computation
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            parameter_space=parameter_space,
            use_simplified=True,
            dt=dt,
            phi_full=phi_jax,  # Full replicated phi array for per-angle scaling mapping
            per_angle_scaling=True,  # Enable per-angle contrast/offset
        )

        # Create initial values dict from init_params
        # Note: init_params keys should match parameter names from parameter_space
        init_param_values = {k: float(v) for k, v in init_params.items()}

        # CRITICAL FIX (Nov 2025): Add per-angle scaling initial values
        # When per_angle_scaling=True, NumPyro model expects separate parameters
        # for each phi angle: contrast_0, contrast_1, ..., offset_0, offset_1, ...
        # Without these, NUTS initialization fails with missing parameter errors
        # (per_angle_scaling is always True in multiprocessing backend as of line 491)

        # IMPROVED FIX (Nov 10, 2025): Compute data-driven initial values
        # Previous hardcoded values (offset=1.0, contrast=0.5) were too far from experimental data
        # causing NumPyro initialization to fail with "Cannot find valid initial parameters"
        # Solution: Estimate offset and contrast from data statistics
        data_min = float(np.min(shard["data"]))
        data_max = float(np.max(shard["data"]))
        data_mean = float(np.mean(shard["data"]))

        # c2_theory typically ranges from ~1.0 (decorrelated) to ~2.0 (fully correlated)
        # With scaling: c2_fitted = offset + contrast × c2_theory
        # At c2_theory=1.0: c2_fitted = data_min (approximately)
        # At c2_theory=2.0: c2_fitted = data_max (approximately)
        # Solving: contrast = (data_max - data_min), offset = data_min - contrast
        estimated_contrast = max(0.01, data_max - data_min)  # At least 0.01 for numerical stability
        estimated_offset = max(0.5, data_mean - estimated_contrast)  # At least 0.5 (physical minimum)

        # Use logger from worker process
        from homodyne.utils.logging import get_logger
        worker_logger = get_logger(__name__)
        worker_logger.info(
            f"Multiprocessing shard {shard_idx}: Data statistics: "
            f"min={data_min:.4f}, max={data_max:.4f}, mean={data_mean:.4f}"
        )
        worker_logger.info(
            f"Multiprocessing shard {shard_idx}: Estimated per-angle scaling: "
            f"contrast={estimated_contrast:.4f}, offset={estimated_offset:.4f}"
        )

        for phi_idx in range(len(phi_unique)):
            # Data-driven initial values (closer to experimental observations)
            init_param_values[f"contrast_{phi_idx}"] = estimated_contrast
            init_param_values[f"offset_{phi_idx}"] = estimated_offset

        worker_logger.info(
            f"Multiprocessing shard {shard_idx}: Added {len(phi_unique)} per-angle scaling parameters "
            f"(contrast and offset) to init_param_values for NUTS initialization"
        )

        # Log first few parameter values to verify
        sample_params = {k: v for i, (k, v) in enumerate(init_param_values.items()) if i < 10}
        worker_logger.info(
            f"Multiprocessing shard {shard_idx}: init_param_values (first 10): {sample_params}"
        )

        # Create NUTS sampler
        nuts_kernel = NUTS(
            model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            init_strategy=numpyro.infer.init_to_value(values=init_param_values),
        )

        # Create MCMC object
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )

        # Run MCMC
        rng_key = jax.random.PRNGKey(shard_idx)
        mcmc.run(rng_key)

        # Extract samples
        samples_dict = mcmc.get_samples()

        # Use centralized parameter names from homodyne.config.parameter_names
        from homodyne.config.parameter_names import get_parameter_names

        param_names_base = get_parameter_names(analysis_mode)

        # PER-ANGLE PARAMETERS: Expand contrast and offset into per-angle names
        # With per-angle scaling enabled, contrast and offset are sampled as:
        # contrast_0, contrast_1, ..., contrast_{n_phi-1}
        # offset_0, offset_1, ..., offset_{n_phi-1}
        n_phi_samples = len(phi_unique)
        param_names_expanded = []

        for param_name in param_names_base:
            if param_name in ["contrast", "offset"]:
                # Expand into per-angle parameters
                for phi_idx in range(n_phi_samples):
                    param_names_expanded.append(f"{param_name}_{phi_idx}")
            else:
                # Regular physics parameter (not per-angle)
                param_names_expanded.append(param_name)

        # Stack samples in correct order
        samples_array = np.stack(
            [np.array(samples_dict[name]) for name in param_names_expanded], axis=1
        )

        # Compute diagnostics
        diagnostics = _compute_diagnostics_worker(samples_dict, mcmc)

        # Check convergence
        converged = _check_convergence_worker(diagnostics)

        return {
            "converged": converged,
            "error": None,  # No error on success (for backend consistency with pjit)
            "samples": samples_array,
            "diagnostics": diagnostics,
            "shard_idx": shard_idx,
        }

    except Exception as e:
        import traceback

        return {
            "converged": False,
            "error": f"Worker MCMC failed: {str(e)}\n{traceback.format_exc()}",
            "samples": None,
            "diagnostics": {},
            "shard_idx": shard_idx,
        }


def _compute_diagnostics_worker(
    samples_dict: Dict[str, np.ndarray],
    mcmc,
) -> Dict[str, Any]:
    """Compute MCMC diagnostics in worker process.

    Parameters
    ----------
    samples_dict : dict
        Parameter samples
    mcmc : MCMC
        NumPyro MCMC object

    Returns
    -------
    dict
        Diagnostics
    """
    import numpy as np

    try:
        diagnostics = {}

        # Acceptance rate
        extra_fields = mcmc.get_extra_fields()
        if "accept_prob" in extra_fields:
            diagnostics["acceptance_rate"] = float(np.mean(extra_fields["accept_prob"]))
        else:
            diagnostics["acceptance_rate"] = None

        # ESS and R-hat
        if mcmc.num_chains > 1:
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            ess_dict = {}
            for param_name, samples in samples_dict.items():
                ess = effective_sample_size(samples)
                ess_dict[param_name] = (
                    float(ess) if ess.size == 1 else float(np.mean(ess))
                )

            diagnostics["ess"] = ess_dict

            rhat_dict = {}
            for param_name, samples in samples_dict.items():
                rhat = gelman_rubin(samples)
                rhat_dict[param_name] = (
                    float(rhat) if rhat.size == 1 else float(np.mean(rhat))
                )

            diagnostics["rhat"] = rhat_dict
        else:
            ess_dict = {}
            for param_name in samples_dict.keys():
                ess_dict[param_name] = len(samples_dict[param_name])

            diagnostics["ess"] = ess_dict
            diagnostics["rhat"] = {k: 1.0 for k in samples_dict.keys()}

        return diagnostics

    except Exception:
        return {
            "acceptance_rate": None,
            "ess": {},
            "rhat": {},
        }


def _check_convergence_worker(diagnostics: Dict[str, Any]) -> bool:
    """Check convergence in worker process.

    Parameters
    ----------
    diagnostics : dict
        Diagnostic metrics

    Returns
    -------
    bool
        True if converged
    """
    # Check R-hat
    if "rhat" in diagnostics and diagnostics["rhat"]:
        max_rhat = max(diagnostics["rhat"].values())
        if max_rhat > 1.1:
            return False

    # Check ESS
    if "ess" in diagnostics and diagnostics["ess"]:
        min_ess = min(diagnostics["ess"].values())
        if min_ess < 100:
            return False

    # Check acceptance rate
    if diagnostics.get("acceptance_rate") is not None:
        if diagnostics["acceptance_rate"] < 0.5:
            return False

    return True


# Export backend class
__all__ = ["MultiprocessingBackend"]
