"""CMA-ES global optimization wrapper for homodyne.

Provides CMA-ES integration using NLSQ's CMAESOptimizer with:
- Automatic memory configuration for large datasets
- BIPOP restart strategy for robust convergence
- Scale-ratio based method selection
- Integration with homodyne's model caching

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is particularly
beneficial for XPCS laminar_flow mode where parameters have vastly different
scales (e.g., D₀ ~ 1e4 vs γ̇₀ ~ 1e-3, scale ratio > 1e7).

NLSQ v0.6.4+ Features:
- evosax backend for JAX-accelerated evolution
- BIPOP restart strategy (alternating large/small populations)
- Memory batching: population_batch_size, data_chunk_size
- MethodSelector for auto-selection based on scale ratio

Usage
-----
>>> from homodyne.optimization.nlsq.cmaes_wrapper import CMAESWrapper
>>> wrapper = CMAESWrapper()
>>> if wrapper.should_use_cmaes(bounds):
...     result = wrapper.fit(model_func, xdata, ydata, p0, bounds)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from homodyne.utils.logging import get_logger, log_exception, log_phase

if TYPE_CHECKING:
    from homodyne.optimization.nlsq.config import NLSQConfig

logger = get_logger(__name__)


def _format_bounds_summary(bounds: tuple[np.ndarray, np.ndarray]) -> str:
    """Format bounds summary for logging.

    Parameters
    ----------
    bounds : tuple[np.ndarray, np.ndarray]
        Lower and upper bounds as (lower, upper) arrays.

    Returns
    -------
    str
        Human-readable bounds summary.
    """
    lower, upper = bounds
    ranges = upper - lower
    n_params = len(lower)

    # Find min/max ranges for summary
    valid_ranges = ranges[ranges > 0]
    if len(valid_ranges) == 0:
        return f"{n_params} params (no valid ranges)"

    min_range = np.min(valid_ranges)
    max_range = np.max(valid_ranges)

    return f"{n_params} params, range=[{min_range:.2e}, {max_range:.2e}]"


def _is_cmaes_available() -> bool:
    """Check if CMA-ES is available via NLSQ."""
    try:
        from nlsq.global_optimization import is_evosax_available

        return is_evosax_available()
    except ImportError:
        return False


CMAES_AVAILABLE = _is_cmaes_available()


@dataclass
class CMAESWrapperConfig:
    """Configuration for CMA-ES wrapper.

    Attributes
    ----------
    preset : str
        CMA-ES preset: "cmaes-fast" (50 gen), "cmaes" (100 gen), "cmaes-global" (200 gen).
    max_generations : int
        Maximum number of CMA-ES generations.
    sigma : float
        Initial step size as fraction of search range (0, 1].
    tol_fun : float
        Function value tolerance for convergence.
    tol_x : float
        Parameter tolerance for convergence.
    restart_strategy : str
        Restart strategy: "none" or "bipop".
    max_restarts : int
        Maximum number of BIPOP restarts.
    population_batch_size : int | None
        Batch size for population evaluation (None = auto).
    data_chunk_size : int | None
        Chunk size for data streaming (None = auto).
    refine_with_nlsq : bool
        Whether to refine CMA-ES solution with NLSQ TRF.
    auto_memory : bool
        Whether to auto-configure memory parameters.
    memory_limit_gb : float
        Memory limit for auto-configuration in GB.
    refinement_workflow : str
        NLSQ workflow for refinement: "auto" (recommended), "standard", "streaming".
    refinement_ftol : float
        Function tolerance for NLSQ refinement.
    refinement_xtol : float
        Parameter tolerance for NLSQ refinement.
    refinement_gtol : float
        Gradient tolerance for NLSQ refinement.
    refinement_max_nfev : int
        Maximum function evaluations for NLSQ refinement.
    refinement_loss : str
        Loss function for NLSQ refinement: "linear", "soft_l1", "huber", etc.
    """

    # CMA-ES global search settings
    preset: str = "cmaes"
    max_generations: int = 100
    sigma: float = 0.5
    tol_fun: float = 1e-8
    tol_x: float = 1e-8
    restart_strategy: str = "bipop"
    max_restarts: int = 9
    population_batch_size: int | None = None
    data_chunk_size: int | None = None
    auto_memory: bool = True
    memory_limit_gb: float = 8.0

    # NLSQ TRF refinement settings (post-CMA-ES)
    refine_with_nlsq: bool = True
    refinement_workflow: str = "auto"  # "auto" auto-selects memory strategy
    refinement_ftol: float = 1e-10  # Tighter than CMA-ES for local refinement
    refinement_xtol: float = 1e-10
    refinement_gtol: float = 1e-10
    refinement_max_nfev: int = 500  # Refinement shouldn't need many iterations
    refinement_loss: str = "linear"  # Linear loss for final refinement

    @classmethod
    def from_nlsq_config(cls, config: NLSQConfig) -> CMAESWrapperConfig:
        """Create CMAESWrapperConfig from NLSQConfig.

        Parameters
        ----------
        config : NLSQConfig
            NLSQ configuration object.

        Returns
        -------
        CMAESWrapperConfig
            CMA-ES wrapper configuration.
        """
        return cls(
            # CMA-ES global search settings
            preset=config.cmaes_preset,
            max_generations=config.cmaes_max_generations,
            sigma=config.cmaes_sigma,
            tol_fun=config.cmaes_tol_fun,
            tol_x=config.cmaes_tol_x,
            restart_strategy=config.cmaes_restart_strategy,
            max_restarts=config.cmaes_max_restarts,
            population_batch_size=config.cmaes_population_batch_size,
            data_chunk_size=config.cmaes_data_chunk_size,
            auto_memory=config.cmaes_population_batch_size is None,
            memory_limit_gb=config.cmaes_memory_limit_gb,
            # NLSQ TRF refinement settings
            refine_with_nlsq=config.cmaes_refine_with_nlsq,
            refinement_workflow=config.cmaes_refinement_workflow,
            refinement_ftol=config.cmaes_refinement_ftol,
            refinement_xtol=config.cmaes_refinement_xtol,
            refinement_gtol=config.cmaes_refinement_gtol,
            refinement_max_nfev=config.cmaes_refinement_max_nfev,
            refinement_loss=config.cmaes_refinement_loss,
        )

    def to_cmaes_config(self, n_params: int) -> Any:
        """Convert to NLSQ CMAESConfig.

        Parameters
        ----------
        n_params : int
            Number of parameters for popsize calculation.

        Returns
        -------
        CMAESConfig
            NLSQ CMAESConfig object.

        Raises
        ------
        ImportError
            If NLSQ CMA-ES is not available.
        """
        if not CMAES_AVAILABLE:
            raise ImportError(
                "CMA-ES requires NLSQ 0.6.4+ with evosax backend. "
                "Install with: pip install nlsq[evosax]"
            )

        from nlsq.global_optimization import CMAESConfig, compute_default_popsize

        # Compute population size if not specified
        popsize = compute_default_popsize(n_params)

        # Map preset to max_generations if using preset
        preset_generations = {
            "cmaes-fast": 50,
            "cmaes": 100,
            "cmaes-global": 200,
        }
        max_gen = preset_generations.get(self.preset, self.max_generations)

        return CMAESConfig(
            popsize=popsize,
            max_generations=max_gen,
            sigma=self.sigma,
            tol_fun=self.tol_fun,
            tol_x=self.tol_x,
            restart_strategy=self.restart_strategy,
            max_restarts=self.max_restarts,
            population_batch_size=self.population_batch_size,
            data_chunk_size=self.data_chunk_size,
            # Disable NLSQ's internal refinement - we do it explicitly in homodyne
            refine_with_nlsq=False,
        )


@dataclass
class CMAESResult:
    """Result from CMA-ES optimization.

    Attributes
    ----------
    parameters : np.ndarray
        Optimized parameter values.
    covariance : np.ndarray | None
        Parameter covariance matrix (if computed).
    chi_squared : float
        Final chi-squared value.
    success : bool
        Whether optimization converged successfully.
    diagnostics : dict
        CMA-ES diagnostics (generations, evaluations, etc.).
    method_used : str
        Method used: "cmaes" or "multi-start".
    nlsq_refined : bool
        Whether result was refined with NLSQ L-M.
    message : str
        Convergence message.
    """

    parameters: np.ndarray
    covariance: np.ndarray | None
    chi_squared: float
    success: bool
    diagnostics: dict = field(default_factory=dict)
    method_used: str = "cmaes"
    nlsq_refined: bool = False
    message: str = ""


class CMAESWrapper:
    """Wrapper around NLSQ's CMAESOptimizer for homodyne integration.

    This wrapper provides:
    - Scale-ratio based method selection (CMA-ES vs multi-start)
    - Automatic memory configuration for large datasets
    - BIPOP restart strategy for robust global optimization
    - Optional L-M refinement of CMA-ES solutions

    Parameters
    ----------
    config : CMAESWrapperConfig | None
        Configuration for CMA-ES wrapper. If None, uses defaults.

    Examples
    --------
    >>> wrapper = CMAESWrapper()
    >>> if wrapper.should_use_cmaes(bounds, scale_threshold=1000):
    ...     result = wrapper.fit(model_func, xdata, ydata, p0, bounds)
    """

    def __init__(self, config: CMAESWrapperConfig | None = None) -> None:
        """Initialize CMA-ES wrapper.

        Parameters
        ----------
        config : CMAESWrapperConfig | None
            Configuration for wrapper. Uses defaults if None.
        """
        self.config = config or CMAESWrapperConfig()
        self._optimizer = None
        self._restarter = None

    @property
    def is_available(self) -> bool:
        """Check if CMA-ES is available."""
        return CMAES_AVAILABLE

    def compute_scale_ratio(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Compute scale ratio from parameter bounds.

        The scale ratio is the ratio of the largest to smallest parameter
        range. High scale ratios (> 1000) indicate multi-scale problems
        where CMA-ES excels.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Lower and upper bounds as (lower, upper) arrays.

        Returns
        -------
        float
            Scale ratio (max_range / min_range).

        Examples
        --------
        >>> lower = np.array([0, 0.001, 100])
        >>> upper = np.array([1, 0.01, 10000])
        >>> wrapper.compute_scale_ratio((lower, upper))
        11000.0  # (10000-100) / (0.01-0.001)
        """
        lower, upper = bounds
        ranges = upper - lower

        # Avoid division by zero
        valid_ranges = ranges[ranges > 0]
        if len(valid_ranges) < 2:
            return 1.0

        min_range = np.min(valid_ranges)
        max_range = np.max(valid_ranges)

        return max_range / min_range if min_range > 0 else 1.0

    def should_use_cmaes(
        self,
        bounds: tuple[np.ndarray, np.ndarray],
        scale_threshold: float = 1000.0,
    ) -> bool:
        """Determine if CMA-ES should be used based on scale ratio.

        CMA-ES adapts its covariance matrix to different parameter scales,
        making it ideal for multi-scale optimization problems. This method
        checks if the scale ratio exceeds the threshold.

        Parameters
        ----------
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper) arrays.
        scale_threshold : float
            Scale ratio threshold for CMA-ES selection. Default: 1000.

        Returns
        -------
        bool
            True if CMA-ES should be used.

        Notes
        -----
        XPCS laminar_flow mode typically has scale ratios > 1e7:
        - D₀ ~ 1e4 (diffusion coefficient)
        - γ̇₀ ~ 1e-3 (shear rate)
        """
        if not self.is_available:
            logger.info(
                "[CMA-ES] Method unavailable: evosax not installed. "
                "Install with: pip install nlsq[evosax]"
            )
            return False

        scale_ratio = self.compute_scale_ratio(bounds)
        should_use = scale_ratio >= scale_threshold
        bounds_summary = _format_bounds_summary(bounds)

        if should_use:
            # Log at INFO when CMA-ES is selected - this is an important decision
            logger.info(
                f"[CMA-ES] Auto-selected: scale_ratio={scale_ratio:.2e} "
                f"(threshold={scale_threshold:.2e}), {bounds_summary}"
            )
        else:
            # Log at DEBUG when not selected - less important
            logger.debug(
                f"[CMA-ES] Not selected: scale_ratio={scale_ratio:.2e} "
                f"< threshold={scale_threshold:.2e}"
            )

        return should_use

    def _run_nlsq_refinement(
        self,
        model_func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run NLSQ TRF refinement on CMA-ES solution.

        Uses NLSQ's curve_fit with workflow="auto" for memory-aware strategy
        selection, similar to NLSQ's "auto_global" workflow.

        Parameters
        ----------
        model_func : Callable
            Model function: ``y = f(x, *params)``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data.
        p0 : np.ndarray
            Initial parameters (CMA-ES solution).
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper).
        sigma : np.ndarray | None
            Data uncertainties (optional).

        Returns
        -------
        dict[str, Any]
            Refinement result with keys: popt, pcov, infodict, mesg, ier.
        """
        from nlsq import curve_fit

        n_data = len(ydata)
        logger.info(
            f"[CMA-ES] Refinement starting: workflow={self.config.refinement_workflow}, "
            f"n_data={n_data:,}, ftol={self.config.refinement_ftol:.1e}, "
            f"max_nfev={self.config.refinement_max_nfev}"
        )

        try:
            with log_phase("CMA-ES refinement", logger, track_memory=True) as phase:
                # Build refinement kwargs for NLSQ curve_fit
                # Note: NLSQ curve_fit uses 'workflow' instead of full scipy interface
                refinement_kwargs: dict[str, Any] = {
                    "ftol": self.config.refinement_ftol,
                    "xtol": self.config.refinement_xtol,
                    "gtol": self.config.refinement_gtol,
                    "max_nfev": self.config.refinement_max_nfev,
                    "loss": self.config.refinement_loss,
                }

                # Run NLSQ curve_fit with workflow="auto" for memory-aware selection
                # NLSQ returns (popt, pcov), not scipy's 5-tuple with full_output
                popt, pcov = curve_fit(
                    f=model_func,
                    xdata=xdata,
                    ydata=ydata,
                    p0=p0,
                    sigma=sigma,
                    bounds=bounds,
                    workflow=self.config.refinement_workflow,
                    **refinement_kwargs,
                )

            # Compute refined chi-squared
            pred = model_func(xdata, *popt)
            residuals = ydata - pred
            if sigma is not None:
                residuals = residuals / sigma
            chi_squared = float(np.sum(residuals**2))

            logger.info(
                f"[CMA-ES] Refinement completed: chi²={chi_squared:.4e}, "
                f"time={phase.duration:.1f}s"
            )

            return {
                "popt": np.asarray(popt),
                "pcov": np.asarray(pcov) if pcov is not None else None,
                "chi_squared": chi_squared,
                "infodict": {},  # NLSQ doesn't return infodict
                "mesg": "NLSQ TRF refinement converged",
                "ier": 1,  # Success
                "success": True,
                "duration_s": phase.duration,
            }

        except Exception as e:
            log_exception(
                logger,
                e,
                context={
                    "phase": "NLSQ refinement",
                    "workflow": self.config.refinement_workflow,
                    "n_data": n_data,
                },
                level=30,  # WARNING
                include_traceback=False,  # Keep it concise
            )
            # Return original parameters on refinement failure
            return {
                "popt": p0,
                "pcov": None,
                "chi_squared": None,
                "infodict": {},
                "mesg": f"Refinement failed: {e}",
                "ier": -1,
                "success": False,
            }

    def _configure_memory(
        self,
        n_data: int,
        n_params: int,
    ) -> tuple[int | None, int | None]:
        """Auto-configure memory parameters for large datasets.

        Parameters
        ----------
        n_data : int
            Number of data points.
        n_params : int
            Number of parameters.

        Returns
        -------
        tuple[int | None, int | None]
            (population_batch_size, data_chunk_size) or (None, None) if
            auto-configuration is disabled.
        """
        if not self.config.auto_memory:
            return self.config.population_batch_size, self.config.data_chunk_size

        if not CMAES_AVAILABLE:
            return None, None

        try:
            from nlsq.global_optimization import (
                auto_configure_cmaes_memory,
                compute_default_popsize,
            )

            popsize = compute_default_popsize(n_params)
            pop_batch, data_chunk = auto_configure_cmaes_memory(
                n_data=n_data,
                popsize=popsize,
                available_memory_gb=self.config.memory_limit_gb,
            )

            # Calculate estimated memory usage for logging
            # Each individual evaluation: n_data * 8 bytes (float64)
            est_memory_mb = (pop_batch * n_data * 8) / (1024 * 1024) if pop_batch else 0

            logger.info(
                f"[CMA-ES] Memory configured: population_batch={pop_batch}, "
                f"data_chunk={data_chunk:,}, popsize={popsize}, "
                f"limit={self.config.memory_limit_gb:.1f}GB, "
                f"est_batch_memory={est_memory_mb:.0f}MB"
            )

            return pop_batch, data_chunk

        except Exception as e:
            log_exception(
                logger,
                e,
                context={
                    "phase": "memory auto-configuration",
                    "n_data": n_data,
                    "n_params": n_params,
                    "memory_limit_gb": self.config.memory_limit_gb,
                },
                level=30,  # WARNING
                include_traceback=False,
            )
            return None, None

    def fit(
        self,
        model_func: Callable,
        xdata: np.ndarray,
        ydata: np.ndarray,
        p0: np.ndarray,
        bounds: tuple[np.ndarray, np.ndarray],
        sigma: np.ndarray | None = None,
    ) -> CMAESResult:
        """Run CMA-ES global optimization.

        Parameters
        ----------
        model_func : Callable
            Model function: ``y = f(x, *params)``.
        xdata : np.ndarray
            Independent variable data.
        ydata : np.ndarray
            Dependent variable data to fit.
        p0 : np.ndarray
            Initial parameter guess.
        bounds : tuple[np.ndarray, np.ndarray]
            Parameter bounds as (lower, upper).
        sigma : np.ndarray | None
            Data uncertainties (optional).

        Returns
        -------
        CMAESResult
            Optimization result with parameters, covariance, diagnostics.

        Raises
        ------
        ImportError
            If CMA-ES is not available.
        RuntimeError
            If optimization fails.
        """
        if not CMAES_AVAILABLE:
            raise ImportError(
                "CMA-ES requires NLSQ 0.6.4+ with evosax backend. "
                "Install with: pip install nlsq[evosax]"
            )

        from nlsq.global_optimization import CMAESOptimizer

        n_params = len(p0)
        n_data = len(ydata)
        scale_ratio = self.compute_scale_ratio(bounds)
        bounds_summary = _format_bounds_summary(bounds)

        # Log comprehensive configuration summary
        logger.info(
            f"[CMA-ES] Optimization starting: n_params={n_params}, "
            f"n_data={n_data:,}, preset={self.config.preset}"
        )
        logger.info(
            f"[CMA-ES] Problem characteristics: scale_ratio={scale_ratio:.2e}, "
            f"{bounds_summary}"
        )

        # Configure memory batching
        pop_batch, data_chunk = self._configure_memory(n_data, n_params)

        # Build CMAESConfig with memory settings
        cmaes_config = self.config.to_cmaes_config(n_params)

        # Override with auto-configured memory settings
        if pop_batch is not None:
            cmaes_config.population_batch_size = pop_batch
        if data_chunk is not None:
            cmaes_config.data_chunk_size = data_chunk

        # Log algorithm configuration
        logger.info(
            f"[CMA-ES] Algorithm settings: max_generations={cmaes_config.max_generations}, "
            f"popsize={cmaes_config.popsize}, sigma={self.config.sigma:.2f}, "
            f"restart={self.config.restart_strategy}"
        )
        if self.config.refine_with_nlsq:
            logger.info(
                f"[CMA-ES] Post-refinement enabled: workflow={self.config.refinement_workflow}, "
                f"ftol={self.config.refinement_ftol:.1e}"
            )

        # Create optimizer with config
        optimizer = CMAESOptimizer(config=cmaes_config)

        # Run CMA-ES global search (NLSQ internal refinement is disabled)
        # - CMA-ES global search with covariance adaptation
        # - Optional BIPOP restarts (configured via cmaes_config.restart_strategy)
        logger.info("[CMA-ES] Global search phase starting...")

        with log_phase(
            "CMA-ES global search", logger, track_memory=True
        ) as search_phase:
            result = optimizer.fit(
                f=model_func,
                xdata=xdata,
                ydata=ydata,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
            )

        # Extract CMA-ES results
        cmaes_params = np.asarray(result.get("popt", p0))
        cmaes_covariance = result.get("pcov", None)
        if cmaes_covariance is not None:
            cmaes_covariance = np.asarray(cmaes_covariance)

        # Build CMA-ES diagnostics
        generations = result.get("cmaes_generations", 0)
        evaluations = result.get("cmaes_evaluations", 0)
        restarts = result.get("cmaes_restarts", 0)
        convergence_reason = result.get("message", "unknown")

        diagnostics = {
            "generations": generations,
            "evaluations": evaluations,
            "restarts": restarts,
            "convergence_reason": convergence_reason,
            "global_search_time_s": search_phase.duration,
        }
        if search_phase.memory_peak_gb is not None:
            diagnostics["global_search_memory_gb"] = search_phase.memory_peak_gb

        # Compute CMA-ES chi-squared
        pred = model_func(xdata, *cmaes_params)
        residuals = ydata - pred
        if sigma is not None:
            residuals = residuals / sigma
        cmaes_chi_squared = float(np.sum(residuals**2))

        # Calculate evaluations per second for performance insight
        evals_per_sec = (
            evaluations / search_phase.duration if search_phase.duration > 0 else 0
        )

        logger.info(
            f"[CMA-ES] Global search completed: chi²={cmaes_chi_squared:.4e}, "
            f"generations={generations}, restarts={restarts}"
        )
        logger.info(
            f"[CMA-ES] Performance: {evaluations:,} evals in {search_phase.duration:.1f}s "
            f"({evals_per_sec:.0f} evals/s), reason={convergence_reason}"
        )

        # Run explicit NLSQ TRF refinement if enabled
        if self.config.refine_with_nlsq:
            refinement_result = self._run_nlsq_refinement(
                model_func=model_func,
                xdata=xdata,
                ydata=ydata,
                p0=cmaes_params,  # Start from CMA-ES solution
                bounds=bounds,
                sigma=sigma,
            )

            if refinement_result["success"]:
                # Use refined parameters
                best_params = refinement_result["popt"]
                best_covariance = refinement_result["pcov"]
                best_chi_squared = refinement_result["chi_squared"]
                nlsq_refined = True

                # Update diagnostics with refinement info
                diagnostics["refinement_nfev"] = refinement_result["infodict"].get(
                    "nfev", 0
                )
                diagnostics["refinement_message"] = refinement_result["mesg"]
                diagnostics["cmaes_chi_squared"] = cmaes_chi_squared
                diagnostics["refined_chi_squared"] = best_chi_squared

                # Calculate improvement percentage
                improvement = (cmaes_chi_squared - best_chi_squared) / cmaes_chi_squared
                diagnostics["chi_squared_improvement"] = improvement

                # Add refinement timing if available
                if "duration_s" in refinement_result:
                    diagnostics["refinement_time_s"] = refinement_result["duration_s"]

                logger.info(
                    f"[CMA-ES] Refinement improved chi²: "
                    f"{cmaes_chi_squared:.4e} → {best_chi_squared:.4e} "
                    f"({improvement:.2%} improvement)"
                )
            else:
                # Refinement failed, use CMA-ES result
                logger.warning(
                    f"[CMA-ES] Refinement failed, using CMA-ES result. "
                    f"Reason: {refinement_result['mesg']}"
                )
                best_params = cmaes_params
                best_covariance = cmaes_covariance
                best_chi_squared = cmaes_chi_squared
                nlsq_refined = False
                diagnostics["refinement_failed"] = True
                diagnostics["refinement_message"] = refinement_result["mesg"]
        else:
            # No refinement requested
            best_params = cmaes_params
            best_covariance = cmaes_covariance
            best_chi_squared = cmaes_chi_squared
            nlsq_refined = False
            logger.debug(
                "[CMA-ES] Post-refinement disabled, using global search result"
            )

        # Calculate total time
        total_time = search_phase.duration
        if nlsq_refined and "refinement_time_s" in diagnostics:
            total_time += diagnostics["refinement_time_s"]
        diagnostics["total_time_s"] = total_time

        # Log final summary
        refined_str = " (refined)" if nlsq_refined else ""
        logger.info(
            f"[CMA-ES] Optimization completed{refined_str}: "
            f"chi²={best_chi_squared:.4e}, total_time={total_time:.1f}s"
        )

        return CMAESResult(
            parameters=best_params,
            covariance=best_covariance,
            chi_squared=best_chi_squared,
            success=True,
            diagnostics=diagnostics,
            method_used="cmaes",
            nlsq_refined=nlsq_refined,
            message=f"CMA-ES converged: {convergence_reason}",
        )


def fit_with_cmaes(
    model_func: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    sigma: np.ndarray | None = None,
    config: CMAESWrapperConfig | None = None,
) -> CMAESResult:
    """Convenience function for CMA-ES optimization.

    Parameters
    ----------
    model_func : Callable
        Model function: ``y = f(x, *params)``.
    xdata : np.ndarray
        Independent variable data.
    ydata : np.ndarray
        Dependent variable data to fit.
    p0 : np.ndarray
        Initial parameter guess.
    bounds : tuple[np.ndarray, np.ndarray]
        Parameter bounds as (lower, upper).
    sigma : np.ndarray | None
        Data uncertainties (optional).
    config : CMAESWrapperConfig | None
        Configuration. Uses defaults if None.

    Returns
    -------
    CMAESResult
        Optimization result.

    Examples
    --------
    >>> result = fit_with_cmaes(model, x, y, p0, bounds)
    >>> print(f"Best params: {result.parameters}")
    """
    wrapper = CMAESWrapper(config)
    return wrapper.fit(model_func, xdata, ydata, p0, bounds, sigma)
