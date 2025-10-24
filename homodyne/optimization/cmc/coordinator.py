"""CMC Coordinator - Main Orchestrator for Consensus Monte Carlo
=================================================================

This module implements the main coordinator for Consensus Monte Carlo (CMC),
orchestrating all CMC components to perform scalable Bayesian uncertainty
quantification on large XPCS datasets (4M-200M+ points).

The CMCCoordinator executes a 6-step pipeline:
    1. Create data shards (stratified by phi angle)
    2. Run SVI initialization (pooled samples → mass matrix)
    3. Execute parallel MCMC on shards (via selected backend)
    4. Combine subposteriors (weighted Gaussian product or averaging)
    5. Validate results (convergence checks)
    6. Package results (extended MCMCResult with CMC diagnostics)

Key Features
------------
- Automatic hardware detection and backend selection
- Robust error handling with fallback mechanisms
- Progress tracking across all pipeline steps
- Checkpoint integration architecture (full implementation Phase 2)
- Comprehensive diagnostics and logging

Usage
-----
    from homodyne.optimization.cmc.coordinator import CMCCoordinator

    # Create coordinator with configuration
    config = {
        'cmc': {
            'sharding': {'strategy': 'stratified'},
            'initialization': {'use_svi': True, 'svi_steps': 5000},
            'combination': {'method': 'weighted', 'fallback_enabled': True},
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 2000,
            'num_chains': 1,
        },
    }
    coordinator = CMCCoordinator(config)

    # Run complete CMC pipeline
    result = coordinator.run_cmc(
        data=c2_exp,
        t1=t1,
        t2=t2,
        phi=phi,
        q=q,
        L=L,
        analysis_mode='laminar_flow',
        nlsq_params={'D0': 1000.0, 'alpha': 1.5, ...},
    )

    # Check if CMC was used
    if result.is_cmc_result():
        print(f"Used {result.num_shards} shards")
        print(f"Combination method: {result.combination_method}")

Integration
-----------
This coordinator is called by:
- fit_mcmc_jax() in homodyne/optimization/mcmc.py (Task Group 9)
- High-level analysis workflows (Task Group 11)
- CLI commands for large dataset processing

References
----------
Scott et al. (2016): "Bayes and big data: the consensus Monte Carlo algorithm"
https://arxiv.org/abs/1411.7435
"""

from typing import Dict, Any, Optional, Callable
import time

import numpy as np
import jax.numpy as jnp

from homodyne.device.config import detect_hardware, HardwareConfig
from homodyne.optimization.cmc.backends import select_backend
from homodyne.optimization.cmc.sharding import (
    calculate_optimal_num_shards,
    shard_data_stratified,
    validate_shards,
)
from homodyne.optimization.cmc.svi_init import (
    pool_samples_from_shards,
    run_svi_initialization,
)
from homodyne.optimization.cmc.combination import combine_subposteriors
from homodyne.optimization.cmc.result import MCMCResult
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class CMCCoordinator:
    """Main coordinator for Consensus Monte Carlo.

    This class orchestrates all CMC components to perform scalable Bayesian
    uncertainty quantification on large XPCS datasets.

    Attributes
    ----------
    config : dict
        Full configuration dictionary
    hardware_config : HardwareConfig
        Detected hardware capabilities
    backend : CMCBackend
        Selected execution backend (pjit/multiprocessing/pbs)
    checkpoint_manager : CheckpointManager, optional
        Checkpoint manager (Phase 2 integration)

    Parameters
    ----------
    config : dict
        Configuration dictionary containing:
        - cmc: CMC-specific settings
        - mcmc: MCMC sampling parameters
        - backend: Optional backend override
        - checkpoint_dir: Optional checkpoint directory (Phase 2)

    Examples
    --------
    >>> config = {'mcmc': {'num_warmup': 500, 'num_samples': 2000}}
    >>> coordinator = CMCCoordinator(config)
    >>> result = coordinator.run_cmc(data, t1, t2, phi, q, L, 'laminar_flow', nlsq_params)
    >>> print(f"Used {result.num_shards} shards")
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize CMC coordinator with configuration.

        Parameters
        ----------
        config : dict
            Configuration dictionary with CMC and MCMC settings
        """
        self.config = config

        # Step 1: Detect hardware
        logger.info("Detecting hardware configuration...")
        self.hardware_config = detect_hardware()
        logger.info(
            f"Hardware detected: platform={self.hardware_config.platform}, "
            f"num_devices={self.hardware_config.num_devices}, "
            f"recommended_backend={self.hardware_config.recommended_backend}"
        )

        # Step 2: Select backend
        user_override = config.get('backend', {}).get('type')
        self.backend = select_backend(self.hardware_config, user_override=user_override)
        logger.info(f"Selected backend: {self.backend.get_backend_name()}")

        # Step 3: Initialize checkpoint manager (Phase 2)
        # For Phase 1, we set up the architecture but don't use it
        self.checkpoint_manager = None
        checkpoint_dir = config.get('checkpoint_dir')
        if checkpoint_dir:
            logger.info(
                f"Checkpoint architecture initialized (dir={checkpoint_dir}). "
                "Full checkpoint integration will be completed in Phase 2."
            )
            # TODO (Phase 2): Initialize CheckpointManager
            # self.checkpoint_manager = CheckpointManager(
            #     checkpoint_dir=checkpoint_dir,
            #     checkpoint_frequency=config.get('checkpoint_frequency', 10),
            # )

    def run_cmc(
        self,
        data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        analysis_mode: str,
        nlsq_params: Dict[str, float],
        model_fn: Optional[Callable] = None,
    ) -> MCMCResult:
        """Run complete CMC pipeline.

        Executes the 6-step CMC workflow:
        1. Create shards
        2. Run SVI initialization
        3. Execute parallel MCMC
        4. Combine subposteriors
        5. Validate results
        6. Return MCMCResult

        Parameters
        ----------
        data : np.ndarray
            Experimental c2 values (flattened)
        t1 : np.ndarray
            First time delay array
        t2 : np.ndarray
            Second time delay array
        phi : np.ndarray
            Azimuthal angle array
        q : float
            Wavevector magnitude
        L : float
            Sample-detector distance
        analysis_mode : str
            Analysis mode: 'static_isotropic' or 'laminar_flow'
        nlsq_params : dict
            NLSQ parameter estimates (initial values for SVI)
        model_fn : callable, optional
            NumPyro model function (if None, will be imported)

        Returns
        -------
        MCMCResult
            Extended MCMC result with CMC-specific fields:
            - per_shard_diagnostics: List of per-shard convergence info
            - cmc_diagnostics: Overall CMC diagnostics
            - combination_method: Method used to combine posteriors
            - num_shards: Number of shards used

        Raises
        ------
        ValueError
            If data is empty or invalid
        RuntimeError
            If all shards fail or combination fails

        Examples
        --------
        >>> coordinator = CMCCoordinator(config)
        >>> result = coordinator.run_cmc(
        ...     data=c2_exp,
        ...     t1=t1, t2=t2, phi=phi,
        ...     q=0.01, L=3.5,
        ...     analysis_mode='laminar_flow',
        ...     nlsq_params={'D0': 1000.0, 'alpha': 1.5, 'D_offset': 10.0},
        ... )
        >>> print(f"CMC used {result.num_shards} shards")
        >>> print(f"Combination method: {result.combination_method}")
        """
        # Validate inputs
        if len(data) == 0:
            raise ValueError("Cannot run CMC on empty dataset")

        logger.info(f"Starting CMC pipeline for {len(data)} data points")
        logger.info(f"Analysis mode: {analysis_mode}")
        pipeline_start_time = time.time()

        # =====================================================================
        # Step 1: Create shards
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 1: Creating data shards")
        logger.info("=" * 70)

        num_shards = self._calculate_num_shards(len(data))
        logger.info(
            f"Calculated optimal number of shards: {num_shards} "
            f"(~{len(data) // num_shards:,} points/shard)"
        )

        strategy = self.config.get('cmc', {}).get('sharding', {}).get('strategy', 'stratified')
        min_shard_size = self.config.get('cmc', {}).get('sharding', {}).get('min_shard_size', 10_000)

        shards = shard_data_stratified(
            data=data,
            t1=t1,
            t2=t2,
            phi=phi,
            num_shards=num_shards,
            q=q,
            L=L,
        )

        # Validate shards (pass min_shard_size from config)
        is_valid, shard_diagnostics = validate_shards(shards, len(data), min_shard_size=min_shard_size)
        if not is_valid:
            raise RuntimeError(f"Shard validation failed: {shard_diagnostics}")

        logger.info(f"✓ Created {len(shards)} shards using {strategy} strategy")
        logger.info(f"  Shard sizes: {[len(s['data']) for s in shards]}")

        # =====================================================================
        # Step 2: SVI initialization
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 2: Running SVI initialization")
        logger.info("=" * 70)

        use_svi = self.config.get('cmc', {}).get('initialization', {}).get('use_svi', True)

        if use_svi:
            # Pool samples from shards
            samples_per_shard = self.config.get('cmc', {}).get('initialization', {}).get('samples_per_shard', 200)
            pooled_data = pool_samples_from_shards(shards, samples_per_shard=samples_per_shard)
            logger.info(f"Pooled {len(pooled_data['data'])} samples from {len(shards)} shards")

            # Import model function if not provided
            if model_fn is None:
                from homodyne.optimization.mcmc import create_numpyro_model
                model_fn = create_numpyro_model(analysis_mode)

            # Run SVI initialization
            svi_steps = self.config.get('cmc', {}).get('initialization', {}).get('svi_steps', 5000)
            timeout = self.config.get('cmc', {}).get('initialization', {}).get('svi_timeout', 900)  # 15 min

            init_params, inv_mass_matrix = run_svi_initialization(
                model_fn=model_fn,
                pooled_data=pooled_data,
                init_params=nlsq_params,
                num_steps=svi_steps,
                timeout=timeout,
            )

            # Check if SVI succeeded
            if inv_mass_matrix is None:
                logger.warning("SVI initialization failed, using identity mass matrix")
                num_params = len(nlsq_params)
                inv_mass_matrix = jnp.eye(num_params)
            else:
                logger.info("✓ SVI initialization completed successfully")
        else:
            logger.info("SVI disabled, using identity mass matrix")
            init_params = nlsq_params
            num_params = len(nlsq_params)
            inv_mass_matrix = jnp.eye(num_params)

        # =====================================================================
        # Step 3: Parallel MCMC execution
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 3: Executing parallel MCMC on shards")
        logger.info("=" * 70)

        mcmc_config = self._get_mcmc_config()
        logger.info(
            f"MCMC config: warmup={mcmc_config['num_warmup']}, "
            f"samples={mcmc_config['num_samples']}, "
            f"chains={mcmc_config['num_chains']}"
        )

        shard_results = self.backend.run_parallel_mcmc(
            shards=shards,
            mcmc_config=mcmc_config,
            init_params=init_params,
            inv_mass_matrix=inv_mass_matrix,
        )

        logger.info(f"✓ Completed MCMC on {len(shard_results)} shards")

        # Check how many shards converged
        n_converged = sum(1 for r in shard_results if r.get('converged', True))
        logger.info(f"  Converged shards: {n_converged}/{len(shard_results)}")

        if n_converged == 0:
            raise RuntimeError("All shards failed to converge. Cannot combine posteriors.")

        # =====================================================================
        # Step 4: Combine subposteriors
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 4: Combining subposteriors")
        logger.info("=" * 70)

        combination_method = self.config.get('cmc', {}).get('combination', {}).get('method', 'weighted')
        fallback_enabled = self.config.get('cmc', {}).get('combination', {}).get('fallback_enabled', True)

        combination_start = time.time()
        combined_posterior = combine_subposteriors(
            shard_results,
            method=combination_method,
            fallback_enabled=fallback_enabled,
        )
        combination_time = time.time() - combination_start

        logger.info(
            f"✓ Combined posteriors using method: {combined_posterior['method']} "
            f"(requested: {combination_method})"
        )
        logger.info(f"  Combination time: {combination_time:.2f}s")

        # =====================================================================
        # Step 5: Validate results
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 5: Validating CMC results")
        logger.info("=" * 70)

        # TODO (Task Group 10): Integrate full validation module
        # For now, perform basic validation
        is_valid, validation_diagnostics = self._basic_validation(
            combined_posterior, shard_results
        )

        if not is_valid:
            logger.warning(f"Validation warnings: {validation_diagnostics}")
        else:
            logger.info("✓ Basic validation passed")

        # =====================================================================
        # Step 6: Package results
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 6: Packaging results")
        logger.info("=" * 70)

        result = self._create_mcmc_result(
            combined_posterior=combined_posterior,
            shard_results=shard_results,
            num_shards=num_shards,
            combination_method=combined_posterior['method'],
            combination_time=combination_time,
            validation_diagnostics=validation_diagnostics,
            analysis_mode=analysis_mode,
            mcmc_config=mcmc_config,
        )

        pipeline_time = time.time() - pipeline_start_time
        logger.info(f"✓ CMC pipeline completed in {pipeline_time:.2f}s")
        logger.info(f"  Total shards: {result.num_shards}")
        logger.info(f"  Combination method: {result.combination_method}")
        logger.info("=" * 70)

        return result

    def _calculate_num_shards(self, dataset_size: int) -> int:
        """Calculate optimal number of shards.

        Parameters
        ----------
        dataset_size : int
            Total number of data points

        Returns
        -------
        int
            Optimal number of shards
        """
        # Allow user override
        user_num_shards = self.config.get('cmc', {}).get('sharding', {}).get('num_shards')
        if user_num_shards is not None:
            logger.info(f"Using user-specified num_shards: {user_num_shards}")
            return user_num_shards

        # Calculate automatically
        target_shard_size_gpu = self.config.get('cmc', {}).get('sharding', {}).get('target_shard_size_gpu', 1_000_000)
        target_shard_size_cpu = self.config.get('cmc', {}).get('sharding', {}).get('target_shard_size_cpu', 2_000_000)
        min_shard_size = self.config.get('cmc', {}).get('sharding', {}).get('min_shard_size', 10_000)

        return calculate_optimal_num_shards(
            dataset_size=dataset_size,
            hardware_config=self.hardware_config,
            target_shard_size_gpu=target_shard_size_gpu,
            target_shard_size_cpu=target_shard_size_cpu,
            min_shard_size=min_shard_size,
        )

    def _get_mcmc_config(self) -> Dict[str, Any]:
        """Extract MCMC configuration parameters.

        Returns
        -------
        dict
            MCMC configuration with keys:
            - num_warmup: int
            - num_samples: int
            - num_chains: int
        """
        mcmc = self.config.get('mcmc', {})

        # Default values for CMC (reduced warmup since we have SVI init)
        return {
            'num_warmup': mcmc.get('num_warmup', 500),
            'num_samples': mcmc.get('num_samples', 2000),
            'num_chains': mcmc.get('num_chains', 1),  # 1 chain per shard
        }

    def _basic_validation(
        self,
        combined_posterior: Dict[str, Any],
        shard_results: list,
    ) -> tuple[bool, Dict[str, Any]]:
        """Perform basic validation of CMC results.

        This is a placeholder for full validation (Task Group 10).

        Parameters
        ----------
        combined_posterior : dict
            Combined posterior samples and statistics
        shard_results : list
            Per-shard MCMC results

        Returns
        -------
        is_valid : bool
            True if validation passed
        diagnostics : dict
            Validation diagnostics
        """
        diagnostics = {}
        is_valid = True

        # Check for NaN/Inf in combined samples
        samples = combined_posterior['samples']
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            is_valid = False
            diagnostics['nan_inf_detected'] = True

        # Check convergence rate
        n_converged = sum(1 for r in shard_results if r.get('converged', True))
        convergence_rate = n_converged / len(shard_results)
        diagnostics['convergence_rate'] = convergence_rate

        if convergence_rate < 0.5:
            is_valid = False
            diagnostics['low_convergence_rate'] = True

        return is_valid, diagnostics

    def _create_mcmc_result(
        self,
        combined_posterior: Dict[str, Any],
        shard_results: list,
        num_shards: int,
        combination_method: str,
        combination_time: float,
        validation_diagnostics: Dict[str, Any],
        analysis_mode: str,
        mcmc_config: Dict[str, Any],
    ) -> MCMCResult:
        """Package results into extended MCMCResult.

        Parameters
        ----------
        combined_posterior : dict
            Combined posterior from combine_subposteriors()
        shard_results : list
            Per-shard MCMC results
        num_shards : int
            Number of shards used
        combination_method : str
            Method used to combine posteriors
        combination_time : float
            Time spent combining posteriors
        validation_diagnostics : dict
            Validation diagnostics
        analysis_mode : str
            Analysis mode used
        mcmc_config : dict
            MCMC configuration

        Returns
        -------
        MCMCResult
            Extended MCMC result with CMC fields
        """
        samples = combined_posterior['samples']
        mean = combined_posterior['mean']
        cov = combined_posterior['cov']

        # Compute standard deviations
        std = np.sqrt(np.diag(cov))

        # For XPCS, first two params are contrast and offset
        # Remaining params are physics parameters
        mean_contrast = float(mean[0])
        mean_offset = float(mean[1])
        mean_params = mean[2:]

        std_contrast = float(std[0])
        std_offset = float(std[1])
        std_params = std[2:]

        samples_contrast = samples[:, 0]
        samples_offset = samples[:, 1]
        samples_params = samples[:, 2:]

        # Extract per-shard diagnostics
        per_shard_diagnostics = []
        for i, shard_result in enumerate(shard_results):
            shard_diag = {
                'shard_id': i,
                'n_samples': len(shard_result.get('samples', [])),
                'converged': shard_result.get('converged', True),
            }

            # Add convergence diagnostics if available
            if 'acceptance_rate' in shard_result:
                shard_diag['acceptance_rate'] = shard_result['acceptance_rate']
            if 'r_hat' in shard_result:
                shard_diag['r_hat'] = shard_result['r_hat']
            if 'ess' in shard_result:
                shard_diag['ess'] = shard_result['ess']

            per_shard_diagnostics.append(shard_diag)

        # Create CMC diagnostics
        n_shards_converged = sum(1 for r in shard_results if r.get('converged', True))
        cmc_diagnostics = {
            'combination_success': True,
            'n_shards_converged': n_shards_converged,
            'n_shards_total': len(shard_results),
            'combination_time': combination_time,
            'convergence_rate': n_shards_converged / len(shard_results),
        }
        cmc_diagnostics.update(validation_diagnostics)

        # Compute overall acceptance rate (if available)
        acceptance_rates = [r.get('acceptance_rate') for r in shard_results if 'acceptance_rate' in r]
        if acceptance_rates:
            overall_acceptance_rate = float(np.mean(acceptance_rates))
        else:
            overall_acceptance_rate = None

        # Create MCMCResult
        result = MCMCResult(
            # Standard MCMC fields
            mean_params=mean_params,
            mean_contrast=mean_contrast,
            mean_offset=mean_offset,
            std_params=std_params,
            std_contrast=std_contrast,
            std_offset=std_offset,
            samples_params=samples_params,
            samples_contrast=samples_contrast,
            samples_offset=samples_offset,
            converged=cmc_diagnostics['convergence_rate'] >= 0.5,
            n_iterations=mcmc_config['num_warmup'] + mcmc_config['num_samples'],
            computation_time=combination_time,  # This is just combination time; full time tracked elsewhere
            backend=self.backend.get_backend_name(),
            analysis_mode=analysis_mode,
            dataset_size=f"{num_shards}_shards",
            n_chains=mcmc_config['num_chains'],
            n_warmup=mcmc_config['num_warmup'],
            n_samples=mcmc_config['num_samples'],
            sampler="NUTS",
            acceptance_rate=overall_acceptance_rate,
            # CMC-specific fields
            per_shard_diagnostics=per_shard_diagnostics,
            cmc_diagnostics=cmc_diagnostics,
            combination_method=combination_method,
            num_shards=num_shards,
        )

        return result
