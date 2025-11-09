"""
Sequential Per-Angle Optimization Module

Provides fallback optimization strategy when angle-stratified chunking cannot be used.
Optimizes each phi angle independently and combines results.

Use Cases:
- Extreme angle imbalance (ratio > 5.0)
- Stratification explicitly disabled
- Debugging and validation
- Memory-constrained environments

Author: Homodyne Development Team
Version: 2.2.0
Date: 2025-11-06
"""

import logging
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AngleSubset:
    """Data subset for a single phi angle.

    Attributes
    ----------
    phi_angle : float
        The phi angle value for this subset
    phi_indices : np.ndarray
        Indices where phi == phi_angle
    n_points : int
        Number of data points for this angle
    phi : np.ndarray
        Phi values (all equal to phi_angle)
    t1 : np.ndarray
        Time 1 values
    t2 : np.ndarray
        Time 2 values
    g2_exp : np.ndarray
        Experimental g2 values
    """

    phi_angle: float
    phi_indices: np.ndarray
    n_points: int
    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2_exp: np.ndarray


@dataclass
class SequentialResult:
    """Result from sequential per-angle optimization.

    Attributes
    ----------
    combined_parameters : np.ndarray
        Combined optimized parameters (weighted average)
    combined_covariance : np.ndarray
        Combined covariance matrix
    per_angle_results : list[dict]
        Individual results for each angle
    n_angles_optimized : int
        Number of angles successfully optimized
    n_angles_failed : int
        Number of angles that failed optimization
    total_cost : float
        Combined optimization cost
    success_rate : float
        Fraction of angles that converged (0.0-1.0)
    """

    combined_parameters: np.ndarray
    combined_covariance: np.ndarray
    per_angle_results: list[dict[str, Any]]
    n_angles_optimized: int
    n_angles_failed: int
    total_cost: float
    success_rate: float


def split_data_by_angle(
    phi: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    g2_exp: np.ndarray,
    min_points_per_angle: int = 10,
) -> list[AngleSubset]:
    """Split dataset into per-angle subsets.

    Parameters
    ----------
    phi : np.ndarray
        Phi angle values (flattened)
    t1 : np.ndarray
        Time 1 values (flattened)
    t2 : np.ndarray
        Time 2 values (flattened)
    g2_exp : np.ndarray
        Experimental g2 values (flattened)
    min_points_per_angle : int, optional
        Minimum points required per angle, default: 10

    Returns
    -------
    list[AngleSubset]
        List of angle subsets, one per unique phi value

    Raises
    ------
    ValueError
        If any angle has fewer than min_points_per_angle points

    Examples
    --------
    >>> phi = np.array([0, 0, 90, 90, 180, 180])
    >>> t1 = np.linspace(0, 1, 6)
    >>> t2 = np.linspace(0, 1, 6)
    >>> g2 = np.ones(6)
    >>> subsets = split_data_by_angle(phi, t1, t2, g2)
    >>> len(subsets)
    3
    >>> subsets[0].phi_angle
    0.0
    >>> subsets[0].n_points
    2
    """
    # Convert to numpy for indexing
    phi_np = np.asarray(phi)
    t1_np = np.asarray(t1)
    t2_np = np.asarray(t2)
    g2_np = np.asarray(g2_exp)

    # Get unique angles
    unique_angles = np.unique(phi_np)
    logger.info(f"Splitting data into {len(unique_angles)} angle subsets")

    subsets = []
    for angle in unique_angles:
        # Find indices for this angle
        indices = np.where(np.isclose(phi_np, angle, atol=1e-6))[0]
        n_points = len(indices)

        if n_points < min_points_per_angle:
            raise ValueError(
                f"Angle {angle:.2f}° has only {n_points} points, "
                f"minimum required: {min_points_per_angle}"
            )

        # Extract subset
        subset = AngleSubset(
            phi_angle=float(angle),
            phi_indices=indices,
            n_points=n_points,
            phi=phi_np[indices],
            t1=t1_np[indices],
            t2=t2_np[indices],
            g2_exp=g2_np[indices],
        )

        subsets.append(subset)
        logger.debug(
            f"  Angle {angle:6.2f}°: {n_points:,} points "
            f"({n_points / len(phi_np) * 100:.1f}% of total)"
        )

    return subsets


def optimize_single_angle(
    subset: AngleSubset,
    residual_func: callable,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    **optimizer_kwargs,
) -> dict[str, Any]:
    """Optimize parameters for a single phi angle.

    Parameters
    ----------
    subset : AngleSubset
        Data for this angle
    residual_func : callable
        Residual function: residual_func(params, phi, t1, t2) -> residuals
    initial_params : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray
        (lower_bounds, upper_bounds) for parameters
    **optimizer_kwargs
        Additional arguments passed to optimizer

    Returns
    -------
    dict
        Result dictionary with keys:
        - 'parameters': Optimized parameters
        - 'covariance': Covariance matrix
        - 'cost': Final cost
        - 'success': Whether optimization converged
        - 'n_iterations': Number of iterations
        - 'message': Status message
        - 'n_points': Number of points used
        - 'phi_angle': Angle value

    Notes
    -----
    Uses scipy.optimize.least_squares as the optimizer since we're
    optimizing per-angle (no chunking issues).
    """
    from scipy.optimize import least_squares

    logger.debug(
        f"Optimizing angle {subset.phi_angle:.2f}° ({subset.n_points:,} points)"
    )

    try:
        # Define residual function for this angle
        def residuals(params):
            return residual_func(
                params, subset.phi, subset.t1, subset.t2, subset.g2_exp
            )

        # Run optimization
        result = least_squares(
            residuals,
            initial_params,
            bounds=bounds,
            **optimizer_kwargs,
        )

        # Compute covariance if possible
        try:
            # Covariance from Jacobian
            J = result.jac
            cov = (
                np.linalg.inv(J.T @ J)
                * (result.fun @ result.fun)
                / (len(result.fun) - len(initial_params))
            )
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to identity if singular
            logger.warning(
                f"Could not compute covariance for angle {subset.phi_angle:.2f}°"
            )
            cov = np.eye(len(initial_params))

        return {
            "parameters": result.x,
            "covariance": cov,
            "cost": result.cost,
            "success": result.success,
            "n_iterations": result.nfev,
            "message": result.message,
            "n_points": subset.n_points,
            "phi_angle": subset.phi_angle,
        }

    except Exception as e:
        logger.error(f"Optimization failed for angle {subset.phi_angle:.2f}°: {e}")
        return {
            "parameters": initial_params,
            "covariance": np.eye(len(initial_params)),
            "cost": np.inf,
            "success": False,
            "n_iterations": 0,
            "message": f"Failed: {str(e)}",
            "n_points": subset.n_points,
            "phi_angle": subset.phi_angle,
        }


def combine_angle_results(
    per_angle_results: list[dict[str, Any]],
    weighting: str = "inverse_variance",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Combine per-angle optimization results.

    Parameters
    ----------
    per_angle_results : list of dict
        Results from optimize_single_angle for each angle
    weighting : str, optional
        Weighting scheme: 'inverse_variance' | 'uniform' | 'n_points'
        Default: 'inverse_variance' (optimal statistical weighting)

    Returns
    -------
    combined_params : np.ndarray
        Weighted average of parameters
    combined_cov : np.ndarray
        Combined covariance matrix
    total_cost : float
        Sum of individual costs

    Notes
    -----
    Inverse variance weighting:
        w_i = 1 / σ²_i
        μ = Σ(w_i × x_i) / Σ(w_i)
        σ² = 1 / Σ(w_i)

    This provides optimal statistical combination when errors are independent.
    """
    # Filter to successful optimizations
    successful = [r for r in per_angle_results if r["success"]]

    if not successful:
        raise ValueError("No angles converged - cannot combine results")

    logger.info(
        f"Combining results from {len(successful)}/{len(per_angle_results)} "
        f"successful angle optimizations"
    )

    # Extract parameters and covariances
    params_list = np.array([r["parameters"] for r in successful])
    cov_list = np.array([r["covariance"] for r in successful])
    n_params = params_list.shape[1]

    # Compute weights
    if weighting == "inverse_variance":
        # Weight by 1/σ² (diagonal of inverse covariance)
        # Add small epsilon to prevent division by zero
        weights = np.array([1.0 / (np.diag(cov).mean() + 1e-10) for cov in cov_list])
    elif weighting == "n_points":
        # Weight by number of data points
        weights = np.array([r["n_points"] for r in successful], dtype=float)
    elif weighting == "uniform":
        # Equal weights
        weights = np.ones(len(successful))
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Normalize weights
    # Add small epsilon to prevent division by zero
    weights = weights / (weights.sum() + 1e-10)

    # Weighted average of parameters
    combined_params = np.sum(params_list * weights[:, np.newaxis], axis=0)

    # Combined covariance (inverse variance weighting formula)
    if weighting == "inverse_variance":
        # σ² = 1 / Σ(1/σ²_i)
        # Add small epsilon to prevent division by zero
        inv_vars = np.array([1.0 / (np.diag(cov) + 1e-10) for cov in cov_list])
        combined_var = 1.0 / inv_vars.sum(axis=0)
        combined_cov = np.diag(combined_var)
    else:
        # Weighted average of covariances
        combined_cov = np.sum(cov_list * weights[:, np.newaxis, np.newaxis], axis=0)

    # Total cost
    total_cost = sum(r["cost"] for r in successful)

    logger.info(f"Combined parameters using {weighting} weighting")
    logger.debug(f"  Weights: {weights}")
    logger.debug(f"  Total cost: {total_cost:.4f}")

    return combined_params, combined_cov, total_cost


def optimize_per_angle_sequential(
    phi: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    g2_exp: np.ndarray,
    residual_func: callable,
    initial_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    weighting: str = "inverse_variance",
    min_success_rate: float = 0.5,
    **optimizer_kwargs,
) -> SequentialResult:
    """Optimize parameters sequentially for each phi angle.

    Main entry point for sequential per-angle optimization.

    Parameters
    ----------
    phi : np.ndarray
        Phi angle values (flattened)
    t1 : np.ndarray
        Time 1 values (flattened)
    t2 : np.ndarray
        Time 2 values (flattened)
    g2_exp : np.ndarray
        Experimental g2 values (flattened)
    residual_func : callable
        Residual function: residual_func(params, phi, t1, t2, g2) -> residuals
    initial_params : np.ndarray
        Initial parameter guess
    bounds : tuple of np.ndarray
        (lower_bounds, upper_bounds)
    weighting : str, optional
        Result combination weighting: 'inverse_variance' | 'uniform' | 'n_points'
    min_success_rate : float, optional
        Minimum fraction of angles that must converge (0.0-1.0), default: 0.5
    **optimizer_kwargs
        Additional arguments passed to scipy.optimize.least_squares

    Returns
    -------
    SequentialResult
        Combined optimization results

    Raises
    ------
    RuntimeError
        If success rate < min_success_rate

    Examples
    --------
    >>> # Simple example with 3 angles
    >>> phi = np.array([0]*100 + [90]*100 + [180]*100)
    >>> t1 = np.tile(np.linspace(0, 1, 100), 3)
    >>> t2 = np.tile(np.linspace(0, 1, 100), 3)
    >>> g2 = np.ones(300)
    >>>
    >>> def residuals(params, phi, t1, t2, g2):
    ...     # Simple model
    ...     return g2 - (1.0 + params[0] * np.exp(-params[1] * t1))
    >>>
    >>> result = optimize_per_angle_sequential(
    ...     phi, t1, t2, g2,
    ...     residuals,
    ...     initial_params=np.array([0.5, 1.0]),
    ...     bounds=(np.array([0.0, 0.0]), np.array([1.0, 10.0]))
    ... )
    >>> result.success_rate
    1.0
    >>> len(result.per_angle_results)
    3
    """
    logger.info(
        f"Starting sequential per-angle optimization\n"
        f"  Total points: {len(phi):,}\n"
        f"  Unique angles: {len(np.unique(phi))}\n"
        f"  Parameters: {len(initial_params)}\n"
        f"  Weighting: {weighting}"
    )

    # Split data by angle
    subsets = split_data_by_angle(phi, t1, t2, g2_exp)

    # Optimize each angle
    per_angle_results = []
    for subset in subsets:
        result = optimize_single_angle(
            subset,
            residual_func,
            initial_params,
            bounds,
            **optimizer_kwargs,
        )
        per_angle_results.append(result)

        status = "✓" if result["success"] else "✗"
        logger.info(
            f"  {status} Angle {result['phi_angle']:6.2f}°: "
            f"cost={result['cost']:.4f}, "
            f"iterations={result['n_iterations']}"
        )

    # Check success rate
    n_success = sum(1 for r in per_angle_results if r["success"])
    n_total = len(per_angle_results)
    success_rate = n_success / n_total

    if success_rate < min_success_rate:
        raise RuntimeError(
            f"Insufficient convergence: {n_success}/{n_total} angles converged "
            f"({success_rate:.1%}), minimum required: {min_success_rate:.1%}"
        )

    # Combine results
    combined_params, combined_cov, total_cost = combine_angle_results(
        per_angle_results, weighting=weighting
    )

    logger.info(
        f"Sequential optimization complete:\n"
        f"  Success rate: {success_rate:.1%} ({n_success}/{n_total})\n"
        f"  Combined cost: {total_cost:.4f}\n"
        f"  Combined parameters: {combined_params}"
    )

    return SequentialResult(
        combined_parameters=combined_params,
        combined_covariance=combined_cov,
        per_angle_results=per_angle_results,
        n_angles_optimized=n_success,
        n_angles_failed=n_total - n_success,
        total_cost=total_cost,
        success_rate=success_rate,
    )
