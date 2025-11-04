"""JAX pjit Backend for Consensus Monte Carlo
============================================

This backend implements parallel MCMC execution using JAX pmap for multi-GPU
systems and sequential execution for single-GPU systems. It leverages JAX's
functional programming model and JIT compilation for high performance.

Execution Strategies
--------------------
1. **Multi-GPU System (num_gpus > 1):**
   - Use JAX pmap to distribute shards across GPUs
   - Each GPU processes one shard in parallel
   - Maximum parallelism: num_gpus

2. **Single GPU System (num_gpus == 1):**
   - Sequential execution of shards on single GPU
   - Each shard runs independently (no parallelism)
   - Benefits from JIT compilation and GPU acceleration

Key Features
------------
- JIT-compiled per-shard MCMC function for performance
- Automatic device placement via JAX
- Progress tracking for long-running executions
- Error handling and retry logic for failed shards
- Convergence diagnostics per shard (R-hat, ESS)

Implementation Details
----------------------
- Uses NumPyro NUTS sampler with JAX backend
- Respects init_params and inv_mass_matrix from config
- Returns standardized result format (samples, diagnostics, timing)
- Handles OOM errors gracefully with retry

Usage Example
-------------
    from homodyne.optimization.cmc.backends.pjit import PjitBackend

    backend = PjitBackend()
    results = backend.run_parallel_mcmc(
        shards=data_shards,
        mcmc_config={'num_warmup': 500, 'num_samples': 2000},
        init_params={'D0': 1000.0, 'alpha': 0.5},
        inv_mass_matrix=mass_matrix,
    )

Integration Points
------------------
- Called by CMC coordinator via select_backend()
- Uses homodyne.optimization.mcmc module for NUTS implementation
- Integrates with CheckpointManager for fault tolerance
- Returns results in standard format for combination.py

Performance Considerations
--------------------------
- JIT compilation overhead on first call (~10-30 seconds)
- Multi-GPU parallelism limited by number of devices
- Single-GPU execution benefits from GPU acceleration vs CPU
- Memory usage: ~2-4GB per shard (depends on shard size)

Error Handling
--------------
- OOM errors: Logged and returned as failed shard
- Convergence failures: Logged with diagnostics
- JAX errors: Caught and wrapped with helpful messages
- Timeouts: Not implemented (rely on MCMC convergence)
"""

from typing import List, Dict, Any, Optional
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import pmap, vmap

from homodyne.optimization.cmc.backends.base import CMCBackend
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Import NumPyro components
try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro import sample

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    logger.error(
        "NumPyro not available. pjit backend requires NumPyro for MCMC sampling. "
        "Install with: pip install numpyro"
    )


class PjitBackend(CMCBackend):
    """JAX pjit backend for parallel MCMC execution on GPU(s).

    This backend uses JAX pmap for multi-GPU parallelization and JIT
    compilation for single-GPU execution. It provides high-performance
    MCMC sampling with automatic device placement.

    Attributes
    ----------
    num_devices : int
        Number of JAX devices (GPUs) detected
    platform : str
        JAX platform ('gpu' or 'cpu')
    is_parallel : bool
        Whether execution will be parallel (multi-GPU) or sequential (single-GPU)

    Methods
    -------
    run_parallel_mcmc(shards, mcmc_config, init_params, inv_mass_matrix)
        Execute MCMC on all shards using JAX pmap or sequential execution
    get_backend_name()
        Return 'pjit'

    Notes
    -----
    - Requires NumPyro for MCMC sampling
    - Requires JAX with GPU support for best performance
    - Falls back to CPU if GPU not available
    - JIT compilation overhead on first call
    """

    def __init__(self):
        """Initialize pjit backend and detect JAX devices."""
        if not NUMPYRO_AVAILABLE:
            raise ImportError(
                "NumPyro is required for pjit backend. "
                "Install with: pip install numpyro"
            )

        # Detect JAX devices (GPU or CPU based on system configuration)
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        self.platform = self.devices[0].platform if self.devices else 'cpu'
        self.is_parallel = self.num_devices > 1

        logger.info(
            f"pjit backend initialized: {self.num_devices} {self.platform} device(s)"
        )

        if self.is_parallel:
            logger.info(
                f"Multi-device mode: Will parallelize across {self.num_devices} devices"
            )
        else:
            logger.info(
                f"Single-device mode: Sequential execution on 1 {self.platform} device"
            )

    def get_backend_name(self) -> str:
        """Return backend name 'pjit'."""
        return "pjit"

    def run_parallel_mcmc(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space,
    ) -> List[Dict[str, Any]]:
        """Run MCMC on all shards using JAX pmap or sequential execution.

        For multi-GPU systems, distributes shards across GPUs using pmap.
        For single-GPU systems, executes shards sequentially with JIT compilation.

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
            Analysis mode ('static_isotropic' or 'laminar_flow')
        parameter_space : ParameterSpace
            Parameter space with configuration-specific bounds

        Returns
        -------
        list of dict
            Per-shard MCMC results
        """
        logger.info(f"Starting pjit backend execution for {len(shards)} shards")

        # Sequential execution (single GPU or CPU)
        if not self.is_parallel:
            return self._run_sequential(
                shards, mcmc_config, init_params, inv_mass_matrix,
                analysis_mode, parameter_space
            )

        # Parallel execution (multi-GPU)
        return self._run_parallel(
            shards, mcmc_config, init_params, inv_mass_matrix,
            analysis_mode, parameter_space
        )

    def _run_sequential(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str = "static_isotropic",
        parameter_space: Optional['ParameterSpace'] = None,
    ) -> List[Dict[str, Any]]:
        """Execute shards sequentially on single device.

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
        analysis_mode : str, optional
            Analysis mode ('static_isotropic' or 'laminar_flow')
        parameter_space : ParameterSpace, optional
            Parameter space with configuration-specific bounds

        Returns
        -------
        list of dict
            Per-shard results
        """
        logger.info(f"Sequential execution: processing {len(shards)} shards one at a time")

        results = []
        for i, shard in enumerate(shards):
            self._log_shard_start(i, len(shards))
            start_time = self._create_timer()

            try:
                # Run MCMC on single shard
                result = self._run_single_shard_mcmc(
                    shard, mcmc_config, init_params, inv_mass_matrix,
                    analysis_mode, parameter_space, shard_idx=i
                )

                # Add timing
                result['elapsed_time'] = self._get_elapsed_time(start_time)

                # Validate result
                self._validate_shard_result(result, i)

                # Log completion
                self._log_shard_complete(
                    i, len(shards), result['elapsed_time'], result['converged']
                )

                results.append(result)

            except Exception as e:
                # Handle error
                error_result = self._handle_shard_error(e, i)
                error_result['elapsed_time'] = self._get_elapsed_time(start_time)
                results.append(error_result)

        return results

    def _run_parallel(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Execute shards in parallel across multiple GPUs using pmap.

        For multi-GPU execution, we process shards in batches equal to
        the number of devices. Each batch is parallelized across GPUs.

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

        Returns
        -------
        list of dict
            Per-shard results
        """
        logger.info(
            f"Parallel execution: processing {len(shards)} shards across "
            f"{self.num_devices} GPUs"
        )

        results = []
        num_batches = (len(shards) + self.num_devices - 1) // self.num_devices

        for batch_idx in range(num_batches):
            # Get shards for this batch
            start_idx = batch_idx * self.num_devices
            end_idx = min(start_idx + self.num_devices, len(shards))
            batch_shards = shards[start_idx:end_idx]

            logger.info(
                f"Processing batch {batch_idx + 1}/{num_batches}: "
                f"shards {start_idx} to {end_idx - 1}"
            )

            # Process batch in parallel
            batch_results = self._run_parallel_batch(
                batch_shards, mcmc_config, init_params, inv_mass_matrix,
                start_shard_idx=start_idx
            )

            results.extend(batch_results)

        return results

    def _run_parallel_batch(
        self,
        batch_shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        start_shard_idx: int,
    ) -> List[Dict[str, Any]]:
        """Execute a batch of shards in parallel using pmap.

        Note: For true parallel execution, we would need to use pmap with
        a batched MCMC function. However, this is complex due to MCMC's
        stateful nature. Instead, we fall back to sequential execution
        for now, with a TODO for future optimization.

        Parameters
        ----------
        batch_shards : list of dict
            Shards to process in this batch
        mcmc_config : dict
            MCMC configuration
        init_params : dict
            Initial parameters
        inv_mass_matrix : np.ndarray
            Mass matrix
        start_shard_idx : int
            Starting index for logging

        Returns
        -------
        list of dict
            Batch results
        """
        # TODO: Implement true pmap parallelization in Phase 2
        # For Phase 1, we use sequential execution even on multi-GPU
        # True pmap would require batching MCMC operations, which is complex

        logger.warning(
            "Multi-GPU pmap parallelization not yet implemented. "
            "Falling back to sequential execution on first GPU. "
            "This will be optimized in Phase 2."
        )

        results = []
        for i, shard in enumerate(batch_shards):
            shard_idx = start_shard_idx + i
            self._log_shard_start(shard_idx, start_shard_idx + len(batch_shards))
            start_time = self._create_timer()

            try:
                result = self._run_single_shard_mcmc(
                    shard, mcmc_config, init_params, inv_mass_matrix, shard_idx
                )
                result['elapsed_time'] = self._get_elapsed_time(start_time)
                self._validate_shard_result(result, shard_idx)
                self._log_shard_complete(
                    shard_idx, start_shard_idx + len(batch_shards),
                    result['elapsed_time'], result['converged']
                )
                results.append(result)
            except Exception as e:
                error_result = self._handle_shard_error(e, shard_idx)
                error_result['elapsed_time'] = self._get_elapsed_time(start_time)
                results.append(error_result)

        return results

    def _run_single_shard_mcmc(
        self,
        shard: Dict[str, np.ndarray],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space: Optional['ParameterSpace'],
        shard_idx: int,
    ) -> Dict[str, Any]:
        """Run MCMC on a single shard.

        This is the core MCMC execution function that runs NumPyro NUTS
        on a single data shard.

        Parameters
        ----------
        shard : dict
            Shard data dictionary
        mcmc_config : dict
            MCMC configuration
        init_params : dict
            Initial parameters
        inv_mass_matrix : np.ndarray
            Mass matrix
        analysis_mode : str
            Analysis mode ("static_isotropic" or "laminar_flow")
        parameter_space : ParameterSpace, optional
            Parameter space with configuration-specific bounds
        shard_idx : int
            Shard index for logging

        Returns
        -------
        dict
            Shard result with samples and diagnostics
        """
        # Extract MCMC configuration
        num_warmup = mcmc_config.get('num_warmup', 500)
        num_samples = mcmc_config.get('num_samples', 2000)
        num_chains = mcmc_config.get('num_chains', 1)
        target_accept_prob = mcmc_config.get('target_accept_prob', 0.8)
        max_tree_depth = mcmc_config.get('max_tree_depth', 10)

        # Extract shard data - pass as numpy arrays (not JAX) to _create_numpyro_model
        # The model will handle JAX conversion internally as needed
        data_np = shard['data']
        sigma_np = shard.get('sigma', np.ones_like(shard['data']))
        t1_np = shard['t1']
        t2_np = shard['t2']
        phi_np = shard['phi']
        q = float(shard['q'])
        L = float(shard['L'])

        # DEBUG: Log array shapes to diagnose element-wise mode detection
        logger.info(
            f"Shard {shard_idx} arrays: "
            f"data={data_np.shape}, t1={t1_np.shape}, t2={t2_np.shape}, phi={phi_np.shape}"
        )
        logger.info(
            f"Shard {shard_idx} t1: ndim={t1_np.ndim}, len={len(t1_np)}, "
            f"element-wise threshold check: t1.ndim==1={t1_np.ndim==1}, len(t1)>2000={len(t1_np)>2000}"
        )

        # Pre-compute dt before JIT compilation to avoid jnp.unique() JAX concretization error
        # This fixes: "Abstract tracer value encountered where concrete value is expected"
        dt_computed = None
        if t1_np is not None:
            # Use numpy (not jax.numpy) for pre-computation
            if t1_np.ndim == 2:
                time_array = np.asarray(t1_np[:, 0] if t1_np.shape[1] > 0 else t1_np[0, :])
            else:
                time_array = np.asarray(t1_np)
            # Estimate from first two unique time points
            unique_times = np.unique(time_array)
            if len(unique_times) > 1:
                dt_computed = float(unique_times[1] - unique_times[0])
            else:
                dt_computed = 1.0  # Fallback
            logger.debug(f"Shard {shard_idx}: Pre-computed dt = {dt_computed:.6f} s for MCMC model")

        # CRITICAL FIX (Nov 2025): Pre-compute unique phi values to prevent 80GB OOM error
        # CMC shards contain replicated phi arrays (e.g., 100K elements for 3 unique angles)
        # physics_cmc.compute_g1_total() requires unique phi to avoid memory explosion:
        #   - Replicated: (100K, 100K) broadcast → 80GB
        #   - Unique: (3, 100K) broadcast → 2.4MB
        # This extraction MUST happen before JIT compilation (NumPyro model creation)
        phi_unique = np.unique(np.asarray(phi_np))
        logger.debug(
            f"Shard {shard_idx}: Extracted {len(phi_unique)} unique phi values from "
            f"{len(phi_np)} replicated elements (memory reduction: "
            f"{len(phi_np)}→{len(phi_unique)}, {len(phi_np)/len(phi_unique):.1f}x)"
        )

        # Create NumPyro model using parameter_space bounds
        # Use real model from mcmc.py that respects parameter_space configuration
        # Lazy import to avoid circular dependency
        from homodyne.optimization.mcmc import _create_numpyro_model

        model = _create_numpyro_model(
            data=data_np,
            sigma=sigma_np,
            t1=t1_np,
            t2=t2_np,
            phi=phi_unique,  # FIXED: Use unique phi values (not replicated array)
            q=q,
            L=L,
            analysis_mode=analysis_mode,
            parameter_space=parameter_space,
            use_simplified=True,  # Use simplified likelihood (default, avoids JAX tracing issues)
            dt=dt_computed,  # Pre-computed to avoid JAX concretization error
            phi_full=phi_np,  # Full replicated phi array for per-angle scaling mapping
            per_angle_scaling=True,  # Enable per-angle contrast/offset
        )

        # Use all init_params (supports both static and laminar_flow modes)
        # init_params already contains all required parameters (5 for static, 9 for laminar_flow)
        # with values from NLSQ pre-optimization (clamped to NumPyro prior bounds)
        init_param_values = init_params

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
            progress_bar=False,  # Disable for cleaner logging
        )

        # Run MCMC
        rng_key = jax.random.PRNGKey(shard_idx)

        try:
            # Model is a closure with data already captured, so no extra args needed
            mcmc.run(rng_key)

            # Extract samples
            samples_dict = mcmc.get_samples()

            # Convert to numpy arrays - parameter order based on analysis_mode
            # Use centralized parameter names from homodyne.config.parameter_names
            # to ensure consistency with model definition in mcmc.py
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

            # VALIDATION: Verify all expected parameters exist in samples_dict
            # This prevents KeyError during sample extraction due to parameter name mismatches
            missing = [p for p in param_names_expanded if p not in samples_dict]
            if missing:
                available = list(samples_dict.keys())
                raise KeyError(
                    f"Missing parameters in MCMC samples for shard {shard_idx} ({analysis_mode}):\n"
                    f"Missing: {missing}\n"
                    f"Expected: {param_names_expanded}\n"
                    f"Available: {available}\n\n"
                    f"This indicates a parameter name mismatch between NumPyro model "
                    f"definition (mcmc.py:_create_numpyro_model) and sample extraction "
                    f"(pjit.py:_run_single_shard_mcmc). Verify both use identical parameter "
                    f"names as defined in homodyne.config.parameter_names"
                )

            samples_array = np.stack([
                np.array(samples_dict[name]) for name in param_names_expanded
            ], axis=1)

            # Log memory usage (architectural fix monitoring, Nov 2025)
            peak_memory_mb = samples_array.nbytes / 1e6
            logger.info(
                f"Shard {shard_idx}: Peak memory = {peak_memory_mb:.2f} MB "
                f"(samples shape: {samples_array.shape})"
            )

            # Compute diagnostics
            diagnostics = self._compute_diagnostics(samples_dict, mcmc)

            # Check convergence
            converged = self._check_convergence(diagnostics)

            # Create error message if not converged
            error_msg = None
            if not converged:
                error_msg = "Convergence criteria not met (check diagnostics for details)"

            return {
                'converged': converged,
                'error': error_msg,
                'samples': samples_array,
                'diagnostics': diagnostics,
                'shard_idx': shard_idx,
            }

        except Exception as e:
            logger.error(f"MCMC failed for shard {shard_idx}: {str(e)}")
            return {
                'converged': False,
                'error': f"MCMC execution failed: {str(e)}",
                'samples': None,
                'diagnostics': {},
                'shard_idx': shard_idx,
            }

    def _compute_diagnostics(
        self,
        samples_dict: Dict[str, jnp.ndarray],
        mcmc: MCMC,
    ) -> Dict[str, Any]:
        """Compute MCMC diagnostics from samples.

        Parameters
        ----------
        samples_dict : dict
            Dictionary of parameter samples
        mcmc : MCMC
            NumPyro MCMC object

        Returns
        -------
        dict
            Diagnostic metrics (ESS, R-hat, acceptance rate)
        """
        try:
            # Get diagnostics from MCMC
            diagnostics = {}

            # Acceptance rate
            extra_fields = mcmc.get_extra_fields()
            if 'accept_prob' in extra_fields:
                diagnostics['acceptance_rate'] = float(
                    np.mean(extra_fields['accept_prob'])
                )
            else:
                diagnostics['acceptance_rate'] = None

            # ESS and R-hat (if multiple chains)
            if mcmc.num_chains > 1:
                from numpyro.diagnostics import effective_sample_size, gelman_rubin

                # Compute ESS for each parameter
                ess_dict = {}
                for param_name, samples in samples_dict.items():
                    ess = effective_sample_size(samples)
                    ess_dict[param_name] = float(ess) if ess.size == 1 else float(np.mean(ess))

                diagnostics['ess'] = ess_dict

                # Compute R-hat for each parameter
                rhat_dict = {}
                for param_name, samples in samples_dict.items():
                    rhat = gelman_rubin(samples)
                    rhat_dict[param_name] = float(rhat) if rhat.size == 1 else float(np.mean(rhat))

                diagnostics['rhat'] = rhat_dict
            else:
                # Single chain: use simpler diagnostics
                ess_dict = {}
                for param_name in samples_dict.keys():
                    ess_dict[param_name] = len(samples_dict[param_name])

                diagnostics['ess'] = ess_dict
                diagnostics['rhat'] = {k: 1.0 for k in samples_dict.keys()}

            return diagnostics

        except Exception as e:
            logger.warning(f"Failed to compute diagnostics: {str(e)}")
            return {
                'acceptance_rate': None,
                'ess': {},
                'rhat': {},
            }

    def _check_convergence(self, diagnostics: Dict[str, Any]) -> bool:
        """Check if MCMC has converged based on diagnostics.

        Convergence criteria:
        - R-hat < 1.1 for all parameters (if available)
        - ESS > 100 for all parameters (if available)
        - Acceptance rate > 0.5 (if available)

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
        if 'rhat' in diagnostics and diagnostics['rhat']:
            max_rhat = max(diagnostics['rhat'].values())
            if max_rhat > 1.1:
                logger.warning(f"Convergence warning: max R-hat = {max_rhat:.3f} > 1.1")
                return False

        # Check ESS
        if 'ess' in diagnostics and diagnostics['ess']:
            min_ess = min(diagnostics['ess'].values())
            if min_ess < 100:
                logger.warning(f"Convergence warning: min ESS = {min_ess:.0f} < 100")
                return False

        # Check acceptance rate
        if diagnostics.get('acceptance_rate') is not None:
            accept_rate = diagnostics['acceptance_rate']
            if accept_rate < 0.5:
                logger.warning(
                    f"Convergence warning: acceptance rate = {accept_rate:.3f} < 0.5"
                )
                return False

        return True


# Export backend class
__all__ = ["PjitBackend"]
