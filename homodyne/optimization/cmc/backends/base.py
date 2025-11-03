"""Abstract Backend Interface for Consensus Monte Carlo
========================================================

This module defines the abstract CMCBackend base class that all CMC execution
backends must implement. It provides a consistent interface for parallel MCMC
execution across different hardware platforms (GPU, CPU, HPC clusters).

Backend Interface
-----------------
All backends must implement:
1. run_parallel_mcmc() - Execute MCMC on all shards (sequential or parallel)
2. get_backend_name() - Return backend name for logging and diagnostics

The interface is intentionally minimal to support diverse execution strategies:
- Sequential execution on single device (pjit on single GPU)
- Parallel execution with multiprocessing (CPU)
- Distributed execution on HPC clusters (PBS/Slurm)

Design Philosophy
-----------------
- **Stateless**: Backends do not maintain state between calls
- **Self-contained**: Each shard execution is independent
- **Error-tolerant**: Backends should handle failures gracefully
- **Observable**: Backends log execution progress and errors

Common Utilities
----------------
The base class provides shared utilities for:
- Logging shard execution progress
- Error handling and recovery
- Validation of shard results
- Timing and performance metrics

Usage Example
-------------
    class MyBackend(CMCBackend):
        def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix):
            results = []
            for i, shard in enumerate(shards):
                self._log_shard_start(i, len(shards))
                result = self._run_single_shard(shard, mcmc_config, init_params, inv_mass_matrix)
                self._validate_shard_result(result, i)
                results.append(result)
            return results

        def get_backend_name(self):
            return "my_backend"

Integration
-----------
- Backends are instantiated by select_backend() in selection.py
- Used by CMC coordinator to execute parallel MCMC
- Results are passed to combination.py for subposterior merging
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class CMCBackend(ABC):
    """Abstract base class for CMC execution backends.

    All CMC backends must inherit from this class and implement the required
    abstract methods. The base class provides common utilities for logging,
    error handling, and result validation.

    Attributes
    ----------
    None (backends are stateless)

    Methods
    -------
    run_parallel_mcmc(shards, mcmc_config, init_params, inv_mass_matrix)
        Execute MCMC on all shards (must be implemented by subclass)
    get_backend_name()
        Return backend name (must be implemented by subclass)

    Common Utilities (provided by base class)
    ------------------------------------------
    _log_shard_start(shard_idx, total_shards)
        Log the start of shard execution
    _log_shard_complete(shard_idx, total_shards, elapsed_time)
        Log the completion of shard execution
    _validate_shard_result(result, shard_idx)
        Validate that a shard result contains required fields
    _handle_shard_error(error, shard_idx)
        Log and wrap shard execution errors

    Notes
    -----
    - Backends are expected to be thread-safe for parallel execution
    - Each shard execution should be independent (no shared state)
    - Backends should log progress for long-running executions
    - Failed shards should be reported but not crash the entire pipeline
    """

    @abstractmethod
    def run_parallel_mcmc(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space,
    ) -> List[Dict[str, Any]]:
        """Run MCMC on all shards in parallel or sequentially.

        This is the main entry point for backend execution. Implementations
        should execute MCMC sampling on each shard using the provided
        initialization parameters and configuration.

        Execution Strategy
        ------------------
        - **Parallel backends** (pjit multi-GPU, multiprocessing):
          Execute multiple shards simultaneously
        - **Sequential backends** (pjit single-GPU):
          Execute shards one at a time
        - **Distributed backends** (PBS, Slurm):
          Submit jobs to cluster scheduler

        Parameters
        ----------
        shards : list of dict
            Data shards to process. Each shard is a dictionary containing:
            - 'data': Experimental c2 values (np.ndarray)
            - 'sigma': Noise estimates (np.ndarray)
            - 't1', 't2': Time arrays (np.ndarray)
            - 'phi': Angle array (np.ndarray)
            - 'q': Wavevector (float)
            - 'L': Sample-detector distance (float)
        mcmc_config : dict
            MCMC configuration dictionary containing:
            - 'num_warmup': Number of warmup iterations (int)
            - 'num_samples': Number of posterior samples (int)
            - 'num_chains': Number of MCMC chains per shard (int, typically 1)
            - 'target_accept_prob': Target acceptance probability (float)
            - 'max_tree_depth': Maximum NUTS tree depth (int)
        init_params : dict
            Initial parameter values for MCMC chain initialization.
            Loaded from config: `initial_parameters.values` section.
            Keys are parameter names (e.g., 'D0', 'alpha', 'contrast')
        inv_mass_matrix : np.ndarray
            Inverse mass matrix (precision matrix) for NUTS initialization.
            Typically identity matrix (diagonal); adapted during warmup.
            Shape: (num_params, num_params)
        analysis_mode : str
            Analysis mode specifying the physics model to use.
            Either "static_isotropic" or "laminar_flow"
        parameter_space : ParameterSpace
            Parameter space object containing parameter bounds and physics constraints.
            Used for prior distributions and parameter validation

        Returns
        -------
        shard_results : list of dict
            List of per-shard MCMC results. Each result dictionary contains:
            - 'samples': Posterior samples (np.ndarray, shape: [num_samples, num_params])
            - 'converged': Whether MCMC converged (bool)
            - 'diagnostics': Diagnostic information (dict)
              - 'ess': Effective sample size per parameter (np.ndarray)
              - 'rhat': R-hat convergence statistic (np.ndarray)
              - 'acceptance_rate': NUTS acceptance rate (float)
            - 'elapsed_time': Execution time in seconds (float)
            - 'error': Error message if failed (str, optional)

        Raises
        ------
        NotImplementedError
            If the backend has not implemented this method

        Notes
        -----
        - Implementations should use _log_shard_start() and _log_shard_complete()
        - Failed shards should return {'converged': False, 'error': <message>}
        - Implementations should validate results with _validate_shard_result()
        - Long-running backends should periodically log progress
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run_parallel_mcmc()"
        )

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend for logging and diagnostics.

        Returns
        -------
        str
            Backend name (e.g., 'pjit', 'multiprocessing', 'pbs')

        Examples
        --------
        >>> backend = PjitBackend()
        >>> backend.get_backend_name()
        'pjit'

        Notes
        -----
        - Should be a short, lowercase identifier
        - Used in log messages and diagnostic output
        - Must be unique among implemented backends
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_backend_name()"
        )

    # -------------------------------------------------------------------------
    # Common Utilities (provided by base class)
    # -------------------------------------------------------------------------

    def _log_shard_start(self, shard_idx: int, total_shards: int) -> None:
        """Log the start of shard execution.

        Parameters
        ----------
        shard_idx : int
            Index of the shard being executed (0-based)
        total_shards : int
            Total number of shards
        """
        logger.info(
            f"[{self.get_backend_name()}] Starting shard {shard_idx + 1}/{total_shards}"
        )

    def _log_shard_complete(
        self,
        shard_idx: int,
        total_shards: int,
        elapsed_time: float,
        converged: bool = True,
    ) -> None:
        """Log the completion of shard execution.

        Parameters
        ----------
        shard_idx : int
            Index of the shard that completed (0-based)
        total_shards : int
            Total number of shards
        elapsed_time : float
            Execution time in seconds
        converged : bool, default True
            Whether the shard converged successfully
        """
        status = "converged" if converged else "FAILED"
        logger.info(
            f"[{self.get_backend_name()}] Shard {shard_idx + 1}/{total_shards} "
            f"{status} in {elapsed_time:.2f}s"
        )

    def _validate_shard_result(
        self, result: Dict[str, Any], shard_idx: int
    ) -> None:
        """Validate that a shard result contains required fields.

        Parameters
        ----------
        result : dict
            Shard result dictionary to validate
        shard_idx : int
            Index of the shard (for error messages)

        Raises
        ------
        ValueError
            If required fields are missing

        Notes
        -----
        Required fields:
        - 'converged': bool
        - 'samples': np.ndarray (if converged=True)
        - 'diagnostics': dict (if converged=True)
        - 'elapsed_time': float
        """
        required_fields = ["converged", "elapsed_time"]

        # Check required fields
        for field in required_fields:
            if field not in result:
                raise ValueError(
                    f"Shard {shard_idx} result missing required field '{field}'"
                )

        # If converged, check for samples and diagnostics
        if result["converged"]:
            if "samples" not in result:
                raise ValueError(
                    f"Shard {shard_idx} marked as converged but missing 'samples'"
                )
            if "diagnostics" not in result:
                logger.warning(
                    f"Shard {shard_idx} missing diagnostics (non-fatal)"
                )
        else:
            # If failed, check for error message
            if "error" not in result:
                logger.warning(
                    f"Shard {shard_idx} marked as failed but missing 'error' message"
                )

    def _handle_shard_error(
        self, error: Exception, shard_idx: int
    ) -> Dict[str, Any]:
        """Log and wrap a shard execution error.

        Parameters
        ----------
        error : Exception
            The exception that occurred
        shard_idx : int
            Index of the shard that failed

        Returns
        -------
        dict
            Error result dictionary with 'converged': False and error message
        """
        error_msg = f"Shard {shard_idx} failed: {str(error)}"
        logger.error(f"[{self.get_backend_name()}] {error_msg}")

        return {
            "converged": False,
            "error": error_msg,
            "elapsed_time": 0.0,
            "samples": None,
            "diagnostics": {},
        }

    def _create_timer(self) -> float:
        """Create a timer for measuring execution time.

        Returns
        -------
        float
            Start time (timestamp from time.time())

        Examples
        --------
        >>> start = self._create_timer()
        >>> # ... do work ...
        >>> elapsed = self._get_elapsed_time(start)
        """
        return time.time()

    def _get_elapsed_time(self, start_time: float) -> float:
        """Get elapsed time since start_time.

        Parameters
        ----------
        start_time : float
            Start time from _create_timer()

        Returns
        -------
        float
            Elapsed time in seconds
        """
        return time.time() - start_time

    def _validate_analysis_mode_consistency(
        self,
        analysis_mode: str,
        parameter_space,
    ) -> None:
        """Validate that analysis_mode is consistent with parameter_space.

        This prevents model/mode mismatches that cause convergence failures.
        For example, using a static_isotropic model when analysis_mode is
        laminar_flow would result in wrong parameter count and physics.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode ('static_isotropic' or 'laminar_flow')
        parameter_space : ParameterSpace
            Parameter space object containing parameter definitions

        Raises
        ------
        ValueError
            If analysis_mode is inconsistent with parameter_space configuration

        Notes
        -----
        Expected parameter counts:
        - static_isotropic: 5 parameters (contrast, offset, D0, alpha, D_offset)
        - laminar_flow: 9 parameters (+ gamma_dot_t0, beta, gamma_dot_t_offset, phi0)

        Examples
        --------
        >>> self._validate_analysis_mode_consistency("laminar_flow", param_space)
        # Passes if param_space has 9 parameters
        >>> self._validate_analysis_mode_consistency("static_isotropic", param_space)
        # Passes if param_space has 5 parameters
        """
        # Get expected parameter count based on analysis_mode
        # NOTE: ParameterSpace.bounds only contains PHYSICAL parameters
        # (contrast and offset are scaling parameters added separately in MCMC)
        if "static" in analysis_mode.lower():
            expected_param_count = 3  # D0, alpha, D_offset
            expected_mode_name = "static_isotropic"
        elif "laminar" in analysis_mode.lower():
            expected_param_count = 7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
            expected_mode_name = "laminar_flow"
        else:
            # Unknown mode, log warning but don't fail
            logger.warning(
                f"Unknown analysis_mode '{analysis_mode}'. "
                f"Expected 'static_isotropic' or 'laminar_flow'. "
                f"Skipping validation."
            )
            return

        # Get actual parameter count from parameter_space
        # Check if parameter_space has the expected attributes
        if not hasattr(parameter_space, "bounds"):
            logger.warning(
                f"ParameterSpace missing 'bounds' attribute. "
                f"Cannot validate analysis_mode consistency."
            )
            return

        # Count parameters from bounds dictionary
        actual_param_count = len(parameter_space.bounds)

        # Validate consistency
        if actual_param_count != expected_param_count:
            error_msg = (
                f"Analysis mode mismatch detected!\n"
                f"  analysis_mode: '{analysis_mode}' (expects {expected_mode_name})\n"
                f"  expected parameters: {expected_param_count}\n"
                f"  actual parameters: {actual_param_count}\n"
                f"This mismatch will cause MCMC convergence failures.\n"
                f"Ensure config analysis_mode matches parameter_space configuration."
            )
            logger.error(f"[{self.get_backend_name()}] {error_msg}")
            raise ValueError(error_msg)

        # Log successful validation
        logger.info(
            f"[{self.get_backend_name()}] Validation passed: "
            f"analysis_mode='{analysis_mode}' matches {actual_param_count} parameters"
        )


# Export abstract base class
__all__ = ["CMCBackend"]
