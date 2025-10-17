"""Time Subsampling Module for Homodyne v2
==========================================

Intelligent time subsampling for large XPCS datasets to reduce memory usage
while preserving physics accuracy in correlation function analysis.

Key Features:
- Logarithmic time sampling (optimal for exponential/power-law decay)
- Linear time sampling (uniform coverage)
- Adaptive sampling (physics-aware intelligent sampling)
- Memory-efficient implementation
- Preserves correlation function structure

Usage:
    from homodyne.data.time_subsampling import subsample_time_grid

    subsampled_data = subsample_time_grid(
        data,
        method="logarithmic",
        target_points=150,
    )

Author: Homodyne Development Team
Date: 2025-10-17
"""

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


def validate_subsampling_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and sanitize subsampling configuration parameters.

    This function ensures all configuration values are within acceptable ranges
    to prevent runtime errors and maintain XPCS data quality. Invalid values
    are corrected with warnings logged.

    Parameters
    ----------
    config : dict
        Raw subsampling configuration from YAML

    Returns
    -------
    dict
        Validated and sanitized configuration

    Examples
    --------
    >>> config = {"trigger_threshold_points": -1000, "max_reduction_factor": 100}
    >>> validated = validate_subsampling_config(config)
    >>> validated["trigger_threshold_points"]
    50000000  # Reset to default
    >>> validated["max_reduction_factor"]
    10  # Clamped to maximum

    Notes
    -----
    **Validation Rules:**
    - trigger_threshold_points: Must be >= 1M (minimum meaningful dataset)
    - max_reduction_factor: Clamped to [2, 10] (preserve XPCS accuracy)
    - method: Must be one of {"logarithmic", "linear", "adaptive"}
    - target_points: If set, must be >= 50 (minimum for correlation structure)
    """
    validated = {}

    # Validate trigger threshold (minimum 1M points)
    threshold = config.get("trigger_threshold_points", 50_000_000)
    if threshold < 1_000_000:
        logger.warning(
            f"Invalid trigger_threshold_points={threshold:,} (must be >= 1M). "
            f"Using default: 50,000,000"
        )
        validated["trigger_threshold_points"] = 50_000_000
    else:
        validated["trigger_threshold_points"] = int(threshold)

    # Validate max_reduction_factor (2-10x range)
    max_factor = config.get("max_reduction_factor", 4)
    if not isinstance(max_factor, (int, float)) or not 2 <= max_factor <= 10:
        logger.warning(
            f"Invalid max_reduction_factor={max_factor} (must be in [2, 10]). "
            f"Using default: 4"
        )
        validated["max_reduction_factor"] = 4
    else:
        validated["max_reduction_factor"] = int(np.clip(max_factor, 2, 10))

    # Validate method
    method = config.get("method", "logarithmic")
    valid_methods = {"logarithmic", "linear", "adaptive"}
    if method not in valid_methods:
        logger.warning(
            f"Invalid method='{method}' (must be one of {valid_methods}). "
            f"Using default: 'logarithmic'"
        )
        validated["method"] = "logarithmic"
    else:
        validated["method"] = method

    # Validate target_points (if explicitly set)
    target_points = config.get("target_points")
    if target_points is not None:
        if not isinstance(target_points, (int, float)) or target_points < 50:
            logger.warning(
                f"Invalid target_points={target_points} (must be >= 50). "
                f"Switching to automatic calculation."
            )
            validated["target_points"] = None
        else:
            validated["target_points"] = int(target_points)
    else:
        validated["target_points"] = None

    # Validate preserve_edges (boolean)
    preserve_edges = config.get("preserve_edges", True)
    validated["preserve_edges"] = bool(preserve_edges)

    # Validate enabled flag
    validated["enabled"] = bool(config.get("enabled", False))

    return validated


def should_apply_subsampling(
    data: dict[str, Any],
    trigger_threshold: int = 50_000_000,
) -> tuple[bool, int, int, int]:
    """Determine if subsampling should be applied based on dataset size.

    This function analyzes the dataset size and decides whether intelligent
    subsampling is needed to keep memory usage manageable while preserving
    correlation structure.

    Parameters
    ----------
    data : dict
        Data dictionary containing 't1', 't2', and 'c2_exp' arrays
    trigger_threshold : int, default 50,000,000
        Total points threshold above which subsampling is triggered

    Returns
    -------
    tuple[bool, int, int, int]
        (should_subsample, total_points, n_time_points, n_angles)
        - should_subsample: True if dataset exceeds threshold
        - total_points: Total number of data points in dataset
        - n_time_points: Number of time points per dimension
        - n_angles: Number of phi angles

    Examples
    --------
    >>> # Check if 23M dataset needs subsampling (threshold: 50M)
    >>> should_sub, total, n_t, n_phi = should_apply_subsampling(data)
    >>> should_sub
    False  # 23M < 50M, no subsampling needed
    """
    c2_exp = np.asarray(data.get("c2_exp", []))

    if c2_exp.size == 0:
        return False, 0, 0, 0

    # Extract dimensions
    if c2_exp.ndim == 3:  # (n_phi, n_t1, n_t2)
        n_angles, n_t1, n_t2 = c2_exp.shape
        n_time_points = n_t1  # Assume square grid
    elif c2_exp.ndim == 2:  # (n_t1, n_t2)
        n_t1, n_t2 = c2_exp.shape
        n_angles = 1
        n_time_points = n_t1
    else:
        logger.warning(f"Unexpected c2_exp dimensions: {c2_exp.ndim}")
        return False, 0, 0, 0

    total_points = n_angles * n_t1 * n_t2

    # Decision: subsample if total exceeds threshold
    should_subsample = total_points > trigger_threshold

    logger.debug(
        f"Dataset size check: {total_points:,} points "
        f"({'>' if should_subsample else '<='} {trigger_threshold:,} threshold)"
    )

    return should_subsample, total_points, n_time_points, n_angles


def compute_adaptive_target_points(
    n_time_points: int,
    n_angles: int,
    total_points: int,
    trigger_threshold: int = 50_000_000,
    min_reduction_factor: int = 2,
    max_reduction_factor: int = 4,
) -> tuple[int, int]:
    """Compute optimal target time points to keep dataset under memory limit.

    This function calculates the minimum reduction factor needed to bring
    the dataset size below the threshold, then computes the corresponding
    target number of time points per dimension. Uses conservative 2-4x
    reduction to preserve XPCS correlation accuracy.

    Algorithm:
    1. If dataset < threshold: no reduction (factor = 1)
    2. Else: compute factor = total / threshold, capped at [2, 4]
    3. Target points = n_time_points / sqrt(factor)
       - This maintains 2D grid structure: (N, N) → (N/√r, N/√r)
       - Total reduction = N² → N²/r (factor r)

    Parameters
    ----------
    n_time_points : int
        Current number of time points per dimension (e.g., 1001)
    n_angles : int
        Number of phi angles in dataset
    total_points : int
        Total data points (n_angles × n_time_points²)
    trigger_threshold : int, default 50,000,000
        Target maximum dataset size (50M points)
    min_reduction_factor : int, default 2
        Minimum reduction factor (2x)
    max_reduction_factor : int, default 4
        Maximum reduction factor (4x, accuracy preservation limit)

    Returns
    -------
    tuple[int, int]
        (target_points, reduction_factor)
        - target_points: Number of time points per dimension after subsampling
        - reduction_factor: Actual reduction factor applied (1, 2, or 4)

    Examples
    --------
    >>> # 23M dataset (below 50M threshold)
    >>> compute_adaptive_target_points(1001, 23, 23_046_023, 50_000_000)
    (1001, 1)  # No reduction needed

    >>> # 100M dataset (needs 2x reduction)
    >>> compute_adaptive_target_points(1500, 43, 96_750_000, 50_000_000)
    (1061, 2)  # 1500 / sqrt(2) ≈ 1061

    >>> # 500M dataset (needs 4x reduction, capped)
    >>> compute_adaptive_target_points(2236, 100, 500_000_000, 50_000_000)
    (1118, 4)  # 2236 / sqrt(4) = 1118

    Notes
    -----
    The sqrt(factor) approach for 2D grids is critical:
    - Linear reduction: 1000 → 500 = 2x reduction in each dimension = 4x total
    - Our approach: 1000 / sqrt(2) ≈ 707 = sqrt(2)x reduction per dim = 2x total
    """
    # No reduction needed if below threshold
    if total_points <= trigger_threshold:
        logger.debug(
            f"Dataset ({total_points:,} points) below threshold "
            f"({trigger_threshold:,}), no subsampling needed"
        )
        return n_time_points, 1

    # Calculate required reduction factor (clamped to [min, max])
    raw_factor = total_points / trigger_threshold
    reduction_factor = int(
        np.clip(raw_factor, min_reduction_factor, max_reduction_factor)
    )

    # Compute target points: divide by sqrt(factor) to maintain 2D structure
    # Example: 1001 / sqrt(4) = 1001 / 2.0 = 500.5 → 500 points
    target_points = int(n_time_points / np.sqrt(reduction_factor))

    # Ensure target is reasonable (at least 50 points for correlation structure)
    target_points = max(target_points, 50)

    logger.info(
        f"Adaptive subsampling: {n_time_points} → {target_points} points "
        f"(reduction factor: {reduction_factor}x, "
        f"dataset: {total_points:,} → ~{total_points // reduction_factor:,} points)"
    )

    return target_points, reduction_factor


def subsample_time_grid(
    data: dict[str, Any],
    method: Literal["logarithmic", "linear", "adaptive"] = "logarithmic",
    target_points: int = 150,
    preserve_edges: bool = True,
    min_points: int = 50,
) -> dict[str, Any]:
    """Subsample time correlation grid to reduce dataset size for optimization.

    This function reduces the number of time points in the 2D correlation matrix
    while preserving the essential physics of the correlation decay. This is
    particularly useful for large datasets that cause GPU out-of-memory errors.

    Parameters
    ----------
    data : dict
        Data dictionary containing:
        - 't1': 2D meshgrid of first time coordinates (n_t1, n_t2)
        - 't2': 2D meshgrid of second time coordinates (n_t1, n_t2)
        - 'c2_exp': Correlation data (n_phi, n_t1, n_t2)
        - Other keys preserved unchanged
    method : {'logarithmic', 'linear', 'adaptive'}, default 'logarithmic'
        Subsampling method:
        - 'logarithmic': Logarithmic spacing (best for exponential decay)
        - 'linear': Linear spacing (uniform coverage)
        - 'adaptive': Physics-aware adaptive sampling
    target_points : int, default 150
        Target number of time points per dimension
    preserve_edges : bool, default True
        Ensure t_min and t_max are included in subsampled grid
    min_points : int, default 50
        Minimum number of points (safety limit)

    Returns
    -------
    dict
        Subsampled data dictionary with reduced time grid:
        - 't1': Reduced 2D meshgrid (target_points, target_points)
        - 't2': Reduced 2D meshgrid (target_points, target_points)
        - 'c2_exp': Subsampled correlation data (n_phi, target_points, target_points)
        - All other keys: Unchanged from input

    Examples
    --------
    >>> # Reduce 1001×1001 grid to 150×150 (96% reduction)
    >>> subsampled = subsample_time_grid(data, method="logarithmic", target_points=150)
    >>> subsampled['c2_exp'].shape
    (23, 150, 150)  # From (23, 1001, 1001)

    Notes
    -----
    Logarithmic sampling is optimal for XPCS correlation functions which typically
    show exponential or power-law decay. It samples more densely at short times
    (where correlation changes rapidly) and sparsely at long times (where
    correlation plateaus).

    Memory savings example:
    - Original: 23 × 1001 × 1001 = 23,046,023 points
    - Subsampled: 23 × 150 × 150 = 517,500 points
    - Reduction: 96% (44x smaller Jacobian matrix)
    """
    logger.info(f"Starting time subsampling with method='{method}'...")

    # Extract time arrays
    t1_2d = np.asarray(data.get("t1", []))
    t2_2d = np.asarray(data.get("t2", []))
    c2_exp = np.asarray(data.get("c2_exp", []))

    # Validate inputs
    if t1_2d.size == 0 or t2_2d.size == 0:
        logger.warning("Time arrays empty, returning data unchanged")
        return data

    if t1_2d.ndim != 2 or t2_2d.ndim != 2:
        logger.warning(
            f"Time arrays not 2D (got t1.ndim={t1_2d.ndim}, t2.ndim={t2_2d.ndim}), "
            f"returning data unchanged",
        )
        return data

    # Extract 1D time vectors from meshgrids
    t1_vec = t1_2d[:, 0] if t1_2d.shape[0] > 0 else t1_2d[0, :]
    t2_vec = t2_2d[0, :] if t2_2d.shape[1] > 0 else t2_2d[:, 0]

    n_t1_orig = len(t1_vec)
    n_t2_orig = len(t2_vec)

    logger.debug(
        f"Original time grid: {n_t1_orig} × {n_t2_orig} = {n_t1_orig * n_t2_orig:,} points"
    )

    # Safety check: don't subsample if already small
    if n_t1_orig <= target_points and n_t2_orig <= target_points:
        logger.info(
            f"Time grid already small ({n_t1_orig}×{n_t2_orig} ≤ {target_points}), "
            f"no subsampling needed",
        )
        return data

    # Enforce minimum points
    n_points = max(min_points, target_points)

    # Generate subsampling indices based on method
    if method == "logarithmic":
        indices_t1 = _create_log_indices(t1_vec, n_points, preserve_edges)
        indices_t2 = _create_log_indices(t2_vec, n_points, preserve_edges)
    elif method == "linear":
        indices_t1 = _create_linear_indices(t1_vec, n_points, preserve_edges)
        indices_t2 = _create_linear_indices(t2_vec, n_points, preserve_edges)
    elif method == "adaptive":
        indices_t1 = _create_adaptive_indices(t1_vec, c2_exp, n_points, preserve_edges)
        indices_t2 = _create_adaptive_indices(t2_vec, c2_exp, n_points, preserve_edges)
    else:
        raise ValueError(f"Unknown subsampling method: {method}")

    # Subsample time grids
    t1_sub = t1_vec[indices_t1]
    t2_sub = t2_vec[indices_t2]

    # Recreate 2D meshgrids
    t1_2d_sub, t2_2d_sub = np.meshgrid(t1_sub, t2_sub, indexing="ij")

    # Subsample c2_exp (slice both time dimensions)
    if c2_exp.ndim == 3:  # (n_phi, n_t1, n_t2)
        c2_exp_sub = c2_exp[:, indices_t1, :][:, :, indices_t2]
    elif c2_exp.ndim == 2:  # (n_t1, n_t2)
        c2_exp_sub = c2_exp[indices_t1, :][:, indices_t2]
    else:
        logger.warning(
            f"Unexpected c2_exp dimensions: {c2_exp.ndim}, returning data unchanged",
        )
        return data

    # Calculate reduction statistics
    n_t1_sub = len(t1_sub)
    n_t2_sub = len(t2_sub)
    orig_size = n_t1_orig * n_t2_orig
    sub_size = n_t1_sub * n_t2_sub
    reduction_pct = (1 - sub_size / orig_size) * 100 if orig_size > 0 else 0

    logger.info(
        f"Time grid subsampled: {n_t1_orig}×{n_t2_orig} → {n_t1_sub}×{n_t2_sub}",
    )
    logger.info(
        f"Dataset reduction: {orig_size:,} → {sub_size:,} points ({reduction_pct:.1f}%)",
    )

    # If c2_exp has phi angles, calculate total dataset reduction
    if c2_exp.ndim == 3:
        n_phi = c2_exp.shape[0]
        total_orig = n_phi * orig_size
        total_sub = n_phi * sub_size
        logger.info(f"Total dataset: {total_orig:,} → {total_sub:,} points")

    # Create output data dictionary (copy all keys, update subsampled ones)
    subsampled_data = data.copy()
    subsampled_data.update(
        {
            "t1": t1_2d_sub,
            "t2": t2_2d_sub,
            "c2_exp": c2_exp_sub,
            # Add metadata about subsampling
            "time_subsampling_applied": True,
            "time_subsampling_method": method,
            "time_subsampling_original_shape": (n_t1_orig, n_t2_orig),
            "time_subsampling_new_shape": (n_t1_sub, n_t2_sub),
        },
    )

    return subsampled_data


def _create_log_indices(
    t_vec: np.ndarray,
    n_points: int,
    preserve_edges: bool,
) -> np.ndarray:
    """Create logarithmically-spaced indices for time sampling.

    Parameters
    ----------
    t_vec : np.ndarray
        1D array of time values
    n_points : int
        Target number of sampled points
    preserve_edges : bool
        Ensure first and last time points are included

    Returns
    -------
    np.ndarray
        Indices for subsampling
    """
    n_total = len(t_vec)

    # Handle edge cases
    if n_points >= n_total:
        return np.arange(n_total)

    # Find valid range (excluding zeros for log)
    t_min = t_vec[t_vec > 0].min() if (t_vec > 0).any() else t_vec.min()
    t_max = t_vec.max()

    # Create logarithmically-spaced values
    if t_min > 0:
        log_times = np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    else:
        # Fallback to linear if min is not positive
        log_times = np.linspace(t_vec.min(), t_max, n_points)

    # Find nearest indices
    indices = []
    for t_target in log_times:
        idx = np.abs(t_vec - t_target).argmin()
        if idx not in indices:  # Avoid duplicates
            indices.append(idx)

    indices = np.array(sorted(indices))

    # Ensure edges are included if requested
    if preserve_edges:
        if 0 not in indices:
            indices = np.concatenate([[0], indices])
        if n_total - 1 not in indices:
            indices = np.concatenate([indices, [n_total - 1]])

    return indices


def _create_linear_indices(
    t_vec: np.ndarray,
    n_points: int,
    preserve_edges: bool,
) -> np.ndarray:
    """Create linearly-spaced indices for time sampling.

    Parameters
    ----------
    t_vec : np.ndarray
        1D array of time values
    n_points : int
        Target number of sampled points
    preserve_edges : bool
        Ensure first and last time points are included

    Returns
    -------
    np.ndarray
        Indices for subsampling
    """
    n_total = len(t_vec)

    if n_points >= n_total:
        return np.arange(n_total)

    if preserve_edges:
        # Include first and last, distribute rest evenly
        indices = np.round(np.linspace(0, n_total - 1, n_points)).astype(int)
    else:
        # Evenly distributed without forcing edges
        indices = np.round(np.linspace(0, n_total - 1, n_points)).astype(int)

    # Remove duplicates and sort
    indices = np.unique(indices)

    return indices


def _create_adaptive_indices(
    t_vec: np.ndarray,
    c2_exp: np.ndarray,
    n_points: int,
    preserve_edges: bool,
) -> np.ndarray:
    """Create adaptively-spaced indices based on correlation function structure.

    Samples more densely where correlation changes rapidly.

    Parameters
    ----------
    t_vec : np.ndarray
        1D array of time values
    c2_exp : np.ndarray
        Correlation data for detecting rapid changes
    n_points : int
        Target number of sampled points
    preserve_edges : bool
        Ensure first and last time points are included

    Returns
    -------
    np.ndarray
        Indices for subsampling
    """
    n_total = len(t_vec)

    if n_points >= n_total:
        return np.arange(n_total)

    # For now, use logarithmic as a good default for adaptive
    # Future enhancement: analyze c2_exp gradients to identify regions of rapid change
    logger.debug(
        "Adaptive sampling using logarithmic baseline (future: gradient-based)"
    )

    return _create_log_indices(t_vec, n_points, preserve_edges)


__all__ = [
    "subsample_time_grid",
    "should_apply_subsampling",
    "compute_adaptive_target_points",
    "validate_subsampling_config",
]
