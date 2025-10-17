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
]
