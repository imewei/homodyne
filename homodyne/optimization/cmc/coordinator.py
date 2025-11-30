"""CMC Coordinator - Main Orchestrator for Consensus Monte Carlo
=================================================================

This module implements the main coordinator for Consensus Monte Carlo (CMC),
orchestrating all CMC components to perform scalable Bayesian uncertainty
quantification on large XPCS datasets (4M-200M+ points).

The CMCCoordinator executes a 5-step pipeline:
    1. Create data shards (stratified by phi angle)
    2. Execute parallel MCMC on shards (via selected backend)
    3. Combine subposteriors (weighted Gaussian product or averaging)
    4. Validate results (convergence checks)
    5. Package results (extended MCMCResult with CMC diagnostics)

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
            'combination': {'method': 'weighted', 'fallback_enabled': True},
        },
        'mcmc': {
            'num_warmup': 500,
            'num_samples': 2000,
            'num_chains': 1,
        },
    }
    coordinator = CMCCoordinator(config)

    # Load configuration-driven parameters
    from homodyne.config.parameter_space import ParameterSpace
    param_space = ParameterSpace.from_config(config)
    initial_vals = {'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}

    # Run complete CMC pipeline
    result = coordinator.run_cmc(
        data=c2_exp,
        t1=t1,
        t2=t2,
        phi=phi,
        q=q,
        L=L,
        analysis_mode='laminar_flow',
        parameter_space=param_space,
        initial_values=initial_vals,
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
    calculate_adaptive_min_shard_size,
    calculate_optimal_num_shards,
    shard_data_stratified,
    validate_shards,
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
    >>> from homodyne.config.parameter_space import ParameterSpace
    >>> param_space = ParameterSpace.from_config(config)
    >>> initial_vals = {'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}
    >>> result = coordinator.run_cmc(
    ...     data, t1, t2, phi, q, L, 'laminar_flow',
    ...     parameter_space=param_space,
    ...     initial_values=initial_vals
    ... )
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
        # Handle both config schemas:
        # - New schema: backend="jax" (string) + backend_config={name: "auto"} (dict)
        # - Old schema: backend={name: "auto"} or backend={type: "auto"} (dict only)
        backend_value = config.get("backend", {})
        if isinstance(backend_value, str):
            # New schema: computational backend is string, parallel backend in backend_config
            backend_config = config.get("backend_config", {})
            user_override = backend_config.get("name") if backend_config else None
        elif isinstance(backend_value, dict):
            # Old schema: backend is dict with 'name' or 'type' field for parallel execution
            # Support both 'name' and 'type' for backward compatibility
            user_override = backend_value.get("name") or backend_value.get("type")
        else:
            user_override = None

        self.backend = select_backend(self.hardware_config, user_override=user_override)
        logger.info(f"Selected backend: {self.backend.get_backend_name()}")

        # Step 3: Initialize checkpoint manager (Phase 2)
        # For Phase 1, we set up the architecture but don't use it
        self.checkpoint_manager = None
        checkpoint_dir = config.get("checkpoint_dir")
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
        parameter_space: Optional["ParameterSpace"] = None,
        initial_values: Optional[Dict[str, float]] = None,
        model_fn: Optional[Callable] = None,
    ) -> MCMCResult:
        """Run complete CMC pipeline.

        Executes the 5-step CMC workflow:
        1. Create shards
        2. Execute parallel MCMC
        3. Combine subposteriors
        4. Validate results
        5. Return MCMCResult

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
        parameter_space : ParameterSpace, optional
            Parameter space with configuration-specific bounds and prior distributions.
            Loaded from YAML config file. If None, defaults are used.
        initial_values : dict[str, float], optional
            Initial parameter values for MCMC chain initialization (e.g., from NLSQ results).
            Loaded from YAML config: `initial_parameters.values` section.
            If None, defaults to mid-point of parameter bounds.
            Example: {'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}
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
        >>> # Config-driven parameter loading (recommended)
        >>> from homodyne.config.parameter_space import ParameterSpace
        >>> param_space = ParameterSpace.from_config(config)
        >>> initial_vals = config.get('initial_parameters', {}).get('values', {})
        >>> result = coordinator.run_cmc(
        ...     data=c2_exp,
        ...     t1=t1, t2=t2, phi=phi,
        ...     q=0.01, L=3.5,
        ...     analysis_mode='laminar_flow',
        ...     parameter_space=param_space,
        ...     initial_values=initial_vals,
        ... )
        >>> print(f"CMC used {result.num_shards} shards")
        >>> print(f"Combination method: {result.combination_method}")
        """
        # Validate inputs
        if len(data) == 0:
            raise ValueError("Cannot run CMC on empty dataset")

        # Get total dataset size (handles multi-dimensional arrays correctly)
        dataset_size = data.size if hasattr(data, "size") else len(data)
        num_samples = (
            data.shape[0] if hasattr(data, "shape") and data.ndim > 1 else dataset_size
        )

        logger.info(
            f"Starting CMC pipeline for {num_samples} samples ({dataset_size:,} total data points)"
        )
        logger.info(f"Analysis mode: {analysis_mode}")
        pipeline_start_time = time.time()

        # =====================================================================
        # Step 1: Create shards
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 1: Creating data shards")
        logger.info("=" * 70)

        num_shards = self._calculate_num_shards(dataset_size)
        logger.info(
            f"Calculated optimal number of shards: {num_shards} "
            f"(~{dataset_size // num_shards:,} points/shard)"
        )

        strategy = (
            self.config.get("cmc", {}).get("sharding", {}).get("strategy", "stratified")
        )

        # CRITICAL VALIDATION: CMC always uses per-angle scaling (per_angle_scaling=True)
        # which requires stratified sharding to ensure all shards contain all phi angles.
        # Random/contiguous sharding may create angle imbalance → zero gradients → silent failures.
        # See: docs/troubleshooting/nlsq-zero-iterations-investigation.md (NLSQ chunking issue)
        if strategy != "stratified":
            logger.warning(
                f"CMC sharding strategy is '{strategy}', but CMC always uses per-angle scaling "
                f"which requires stratified sharding for correct results. "
                f"Non-stratified strategies (random, contiguous) may cause angle imbalance "
                f"in individual shards, leading to zero gradients and parameter estimation failures. "
                f"RECOMMENDATION: Set cmc.sharding.strategy='stratified' in configuration. "
                f"See ultra-think-20251106-012247 for technical details."
            )

        # Use adaptive min_shard_size based on dataset size
        # This allows small test datasets to pass validation while maintaining
        # statistical rigor for production datasets
        default_min_shard_size = (
            self.config.get("cmc", {}).get("sharding", {}).get("min_shard_size", 10_000)
        )
        min_shard_size = calculate_adaptive_min_shard_size(
            dataset_size=dataset_size,
            num_shards=num_shards,
            default_min=default_min_shard_size,
        )

        # Flatten multi-dimensional data for sharding
        # Sharding functions expect 1D arrays: data (N,), t1 (N,), t2 (N,), phi (N,)
        # Input data shape: (n_phi, n_t1, n_t2) → flatten to (n_phi * n_t1 * n_t2,)
        # Note: t1 and t2 are already 2D meshgrids from data loader
        if hasattr(data, "ndim") and data.ndim > 1:
            logger.info(
                f"Flattening multi-dimensional data from shape {data.shape} for sharding"
            )

            # Flatten data (preserves phi-major ordering: all points from phi[0], then phi[1], etc.)
            data_flat = data.flatten()

            # Calculate points per phi angle
            points_per_phi = (
                data.shape[1] * data.shape[2] if data.ndim == 3 else data.shape[1]
            )

            # t1 and t2 are already 2D meshgrids (n_t1, n_t2) from data loader
            # Flatten each meshgrid once, then tile for each phi angle
            t1_pattern = t1.flatten()  # (n_t1 * n_t2,) = (1,002,001,)
            t2_pattern = t2.flatten()  # (n_t1 * n_t2,) = (1,002,001,)
            t1_flat = np.tile(
                t1_pattern, num_samples
            )  # (n_phi * n_t1 * n_t2,) = (23,046,023,)
            t2_flat = np.tile(
                t2_pattern, num_samples
            )  # (n_phi * n_t1 * n_t2,) = (23,046,023,)

            # Replicate phi values for each (t1, t2) pair
            phi_flat = np.repeat(
                phi, points_per_phi
            )  # (n_phi * n_t1 * n_t2,) = (23,046,023,)

            logger.info(
                f"Flattened arrays: data={data_flat.shape}, t1={t1_flat.shape}, t2={t2_flat.shape}, phi={phi_flat.shape}"
            )
        else:
            # Data already 1D, use as-is
            data_flat = data
            t1_flat = t1
            t2_flat = t2
            phi_flat = phi

        shards = shard_data_stratified(
            data=data_flat,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            num_shards=num_shards,
            q=q,
            L=L,
        )

        # Validate shards (pass min_shard_size from config)
        is_valid, shard_diagnostics = validate_shards(
            shards, dataset_size, min_shard_size=min_shard_size
        )
        if not is_valid:
            raise RuntimeError(f"Shard validation failed: {shard_diagnostics}")

        logger.info(f"✓ Created {len(shards)} shards using {strategy} strategy")
        logger.info(
            f"  Shard sizes: {[s['shard_size'] for s in shards]} data points per shard"
        )

        # =====================================================================
        # Step 2: Config-Driven Parameter Loading
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 2: Loading parameters from configuration")
        logger.info("=" * 70)

        # Load parameter_space and initial_values using same logic as mcmc.py
        # This ensures consistency between NUTS and CMC parameter handling
        if parameter_space is None:
            logger.info("No parameter_space provided, using package defaults")
            try:
                from homodyne.config.parameter_space import ParameterSpace

                parameter_space = ParameterSpace.from_defaults(analysis_mode)
                logger.info(
                    f"Loaded default parameter_space: model={parameter_space.model_type}, "
                    f"num_params={len(parameter_space.parameter_names)}"
                )
            except Exception as e:
                logger.error(f"Failed to load default parameter_space: {e}")
                raise RuntimeError(
                    "Cannot initialize CMC without parameter_space. "
                    "Provide parameter_space argument or ensure config is valid."
                ) from e
        else:
            logger.info(
                f"Using provided parameter_space: model={parameter_space.model_type}, "
                f"num_params={len(parameter_space.parameter_names)}"
            )

        # Load initial_values (used for MCMC chain initialization)
        if initial_values is None:
            logger.info(
                "No initial_values provided, calculating mid-point defaults from bounds"
            )
            # Calculate mid-point of parameter bounds as default
            initial_values = {}
            for param_name in parameter_space.parameter_names:
                # Skip scaling parameters (contrast, offset)
                if param_name in ["contrast", "offset"]:
                    continue
                bounds = parameter_space.get_bounds(param_name)
                if bounds is not None:
                    min_val, max_val = bounds
                    midpoint = (min_val + max_val) / 2.0
                    initial_values[param_name] = midpoint
            logger.info(
                f"Using mid-point defaults: {list(initial_values.keys())} = "
                f"{[f'{v:.4g}' for v in initial_values.values()]}"
            )
        else:
            logger.info(
                f"Using provided initial_values: {list(initial_values.keys())} = "
                f"{[f'{v:.4g}' for v in initial_values.values()]}"
            )

        # Validate initial_values against parameter_space bounds
        # (exclude scaling parameters which are handled separately)
        if initial_values:
            physical_params = {
                k: v
                for k, v in initial_values.items()
                if k not in ["contrast", "offset"]
            }
            if physical_params:
                is_valid, violations = parameter_space.validate_values(physical_params)
                if not is_valid:
                    raise ValueError(
                        f"Initial parameter values violate bounds:\n"
                        + "\n".join(violations)
                    )
                logger.info(
                    "Initial parameter values validated successfully (all within bounds)"
                )

        # Determine number of parameters from parameter_space
        # Note: parameter_space.parameter_names contains only physical parameters
        # Per-angle scaling adds 2*n_phi additional parameters (contrast_i, offset_i)
        # Total = len(physical_params) + 2*n_phi

        phi_unique_temp = np.unique(np.asarray(phi))
        n_phi_temp = len(phi_unique_temp)
        num_params = len(parameter_space.parameter_names) + (2 * n_phi_temp)

        # CRITICAL FIX (Nov 14, 2025): Parameter ordering for NumPyro initialization
        # NumPyro's init_to_value() strategy requires parameters in the EXACT ORDER
        # that the model samples them. For per-angle scaling, the model samples:
        #   1. contrast_0, contrast_1, ..., contrast_{n_phi-1}
        #   2. offset_0, offset_1, ..., offset_{n_phi-1}
        #   3. Physical parameters (D0, alpha, D_offset, ...)
        #
        # This ordering is determined by parameter_space.parameter_names iteration
        # in mcmc.py:1706-1720. If we send parameters in wrong order, NumPyro assigns
        # wrong values (e.g., D0 → contrast_0), causing initialization failure.

        # Determine number of unique phi angles
        phi_unique = np.unique(np.asarray(phi))
        n_phi = len(phi_unique)

        logger.info(f"Expanding initial_values for {n_phi} unique phi angles (per-angle scaling)")

        # Get default scaling values from config (will be removed from physical params)
        default_contrast = initial_values.get("contrast", 0.5)
        default_offset = initial_values.get("offset", 1.0)

        # Create NEW dict with CORRECT ORDER: per-angle params FIRST, then physical params
        init_params = {}

        # STEP 1: Add per-angle contrast parameters FIRST (contrast_0, contrast_1, ...)
        for phi_idx in range(n_phi):
            contrast_key = f"contrast_{phi_idx}"

            # Use midpoint from parameter_space if available, otherwise use defaults
            try:
                contrast_midpoint = sum(parameter_space.get_bounds("contrast")) / 2.0
                init_params[contrast_key] = parameter_space.clamp_to_open_interval(
                    "contrast", contrast_midpoint, epsilon=1e-6
                )
            except (KeyError, AttributeError):
                init_params[contrast_key] = default_contrast

        # STEP 2: Add per-angle offset parameters SECOND (offset_0, offset_1, ...)
        for phi_idx in range(n_phi):
            offset_key = f"offset_{phi_idx}"

            try:
                offset_midpoint = sum(parameter_space.get_bounds("offset")) / 2.0
                init_params[offset_key] = parameter_space.clamp_to_open_interval(
                    "offset", offset_midpoint, epsilon=1e-6
                )
            except (KeyError, AttributeError):
                init_params[offset_key] = default_offset

        # STEP 3: Add physical parameters LAST (D0, alpha, D_offset, ...)
        # Exclude base 'contrast' and 'offset' as they're replaced by per-angle versions
        for param_name, param_value in initial_values.items():
            if param_name not in ["contrast", "offset"]:
                init_params[param_name] = param_value

        logger.info(
            f"✓ Expanded initial_values: {len(initial_values)} physical params + "
            f"{n_phi * 2} per-angle params = {len(init_params)} total parameters"
        )

        # DEBUG: Log actual parameter ordering being sent to worker
        logger.debug(f"Coordinator sending init_params to worker (ORDERED): {list(init_params.keys())}")
        per_angle_params = {k: v for k, v in init_params.items() if 'contrast_' in k or 'offset_' in k}
        if per_angle_params:
            logger.debug(f"Per-angle parameters: {per_angle_params}")

        # Use identity mass matrix (diagonal covariance)
        # CMC uses simple initialization - mass matrix adapted during warmup
        inv_mass_matrix = jnp.eye(num_params)
        logger.info(f"✓ Initialized {num_params} parameters with identity mass matrix")

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
            analysis_mode=analysis_mode,
            parameter_space=parameter_space,
        )

        logger.info(f"✓ Completed MCMC on {len(shard_results)} shards")

        # Check how many shards converged
        n_converged = sum(1 for r in shard_results if r.get("converged", True))
        logger.info(f"  Converged shards: {n_converged}/{len(shard_results)}")

        if n_converged == 0:
            raise RuntimeError(
                "All shards failed to converge. Cannot combine posteriors."
            )

        # =====================================================================
        # Step 3: Combine subposteriors
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 3: Combining subposteriors")
        logger.info("=" * 70)

        combination_section = self.config.get("combination")
        if not isinstance(combination_section, dict) or not combination_section:
            combination_section = self.config.get("cmc", {}).get("combination", {})

        combination_method = combination_section.get("method", "weighted")
        fallback_enabled = combination_section.get("fallback_enabled", True)

        validation_section = self.config.get("validation")
        if not isinstance(validation_section, dict) or not validation_section:
            validation_section = self.config.get("cmc", {}).get("validation", {})

        combination_start = time.time()
        combined_posterior = combine_subposteriors(
            shard_results,
            method=combination_method,
            fallback_enabled=fallback_enabled,
            diagnostics_config=validation_section,
        )
        combination_time = time.time() - combination_start

        logger.info(
            f"✓ Combined posteriors using method: {combined_posterior['method']} "
            f"(requested: {combination_method})"
        )
        logger.info(f"  Combination time: {combination_time:.2f}s")

        # =====================================================================
        # Step 4: Validate results
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 4: Validating CMC results")
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
        # Step 5: Package results
        # =====================================================================
        logger.info("=" * 70)
        logger.info("STEP 5: Packaging results")
        logger.info("=" * 70)

        result = self._create_mcmc_result(
            combined_posterior=combined_posterior,
            shard_results=shard_results,
            num_shards=num_shards,
            combination_method=combined_posterior["method"],
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
        user_num_shards = (
            self.config.get("cmc", {}).get("sharding", {}).get("num_shards")
        )
        if user_num_shards is not None:
            logger.info(f"Using user-specified num_shards: {user_num_shards}")
            return user_num_shards

        # Calculate automatically
        target_shard_size_gpu = (
            self.config.get("cmc", {})
            .get("sharding", {})
            .get("target_shard_size_gpu", 100_000)
        )
        target_shard_size_cpu = (
            self.config.get("cmc", {})
            .get("sharding", {})
            .get("target_shard_size_cpu", 2_000_000)
        )
        min_shard_size = (
            self.config.get("cmc", {}).get("sharding", {}).get("min_shard_size", 10_000)
        )

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
        mcmc = self.config.get("mcmc", {})

        # Default values for CMC
        return {
            "num_warmup": mcmc.get("num_warmup", 500),
            "num_samples": mcmc.get("num_samples", 2000),
            "num_chains": mcmc.get("num_chains", 1),  # 1 chain per shard
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
        samples = combined_posterior["samples"]
        if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
            is_valid = False
            diagnostics["nan_inf_detected"] = True

        # Check convergence rate
        n_converged = sum(1 for r in shard_results if r.get("converged", True))
        convergence_rate = n_converged / len(shard_results)
        diagnostics["convergence_rate"] = convergence_rate

        if convergence_rate < 0.5:
            is_valid = False
            diagnostics["low_convergence_rate"] = True

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
        samples = combined_posterior["samples"]
        mean = combined_posterior["mean"]
        cov = combined_posterior["cov"]

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
                "shard_id": i,
                "n_samples": len(shard_result.get("samples", [])),
                "converged": shard_result.get("converged", True),
            }

            # Add convergence diagnostics if available
            if "acceptance_rate" in shard_result:
                shard_diag["acceptance_rate"] = shard_result["acceptance_rate"]
            if "r_hat" in shard_result:
                shard_diag["r_hat"] = shard_result["r_hat"]
            if "ess" in shard_result:
                shard_diag["ess"] = shard_result["ess"]

            per_shard_diagnostics.append(shard_diag)

        # Create CMC diagnostics
        n_shards_converged = sum(1 for r in shard_results if r.get("converged", True))
        cmc_diagnostics = {
            "combination_success": True,
            "n_shards_converged": n_shards_converged,
            "n_shards_total": len(shard_results),
            "combination_time": combination_time,
            "convergence_rate": n_shards_converged / len(shard_results),
        }
        cmc_diagnostics.update(validation_diagnostics)

        # Compute overall acceptance rate (if available)
        acceptance_rates = [
            r.get("acceptance_rate") for r in shard_results if "acceptance_rate" in r
        ]
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
            converged=cmc_diagnostics["convergence_rate"] >= 0.5,
            n_iterations=mcmc_config["num_warmup"] + mcmc_config["num_samples"],
            computation_time=combination_time,  # This is just combination time; full time tracked elsewhere
            backend=self.backend.get_backend_name(),
            analysis_mode=analysis_mode,
            dataset_size=f"{num_shards}_shards",
            n_chains=mcmc_config["num_chains"],
            n_warmup=mcmc_config["num_warmup"],
            n_samples=mcmc_config["num_samples"],
            sampler="NUTS",
            acceptance_rate=overall_acceptance_rate,
            # CMC-specific fields
            per_shard_diagnostics=per_shard_diagnostics,
            cmc_diagnostics=cmc_diagnostics,
            combination_method=combination_method,
            num_shards=num_shards,
        )

        return result
