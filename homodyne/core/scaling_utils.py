"""Shared scaling utilities for per-angle contrast/offset estimation.

This module provides unified quantile-based estimation of contrast and offset
parameters that can be used by both NLSQ and CMC optimization backends.

The physics basis:
    C2 = contrast × g1² + offset

    - At large time lags, g1² → 0, so C2 → offset (the "floor")
    - At small time lags, g1² ≈ 1, so C2 ≈ contrast + offset (the "ceiling")

Version: 2.18.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from logging import Logger, LoggerAdapter

logger = get_logger(__name__)


def estimate_contrast_offset_from_quantiles(
    c2_data: np.ndarray,
    delta_t: np.ndarray,
    contrast_bounds: tuple[float, float] = (0.0, 1.0),
    offset_bounds: tuple[float, float] = (0.5, 1.5),
    lag_floor_quantile: float = 0.80,
    lag_ceiling_quantile: float = 0.20,
    value_quantile_low: float = 0.10,
    value_quantile_high: float = 0.90,
) -> tuple[float, float]:
    """Estimate contrast and offset from C2 data using physics-informed quantile analysis.

    Uses the correlation decay structure: C2 = contrast × g1² + offset
    - At large time lags, g1² → 0, so C2 → offset (the "floor")
    - At small time lags, g1² ≈ 1, so C2 ≈ contrast + offset (the "ceiling")

    Parameters
    ----------
    c2_data : np.ndarray
        C2 correlation values (1D array).
    delta_t : np.ndarray
        Time lag values ``abs(t1 - t2)`` (same shape as c2_data).
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast parameter.
    offset_bounds : tuple[float, float]
        Valid bounds for offset parameter.
    lag_floor_quantile : float
        Quantile threshold for "large lag" region (default: 0.80 = top 20% of lags).
    lag_ceiling_quantile : float
        Quantile threshold for "small lag" region (default: 0.20 = bottom 20% of lags).
    value_quantile_low : float
        Quantile for robust floor estimation (default: 0.10).
    value_quantile_high : float
        Quantile for robust ceiling estimation (default: 0.90).

    Returns
    -------
    tuple[float, float]
        (contrast_est, offset_est) - Estimated values clipped to bounds.

    Notes
    -----
    The estimation is robust to outliers by using quantiles instead of min/max.
    The lag-based segmentation ensures we're sampling from the appropriate
    regions of the correlation decay curve.
    """
    c2 = np.asarray(c2_data)
    dt = np.asarray(delta_t)

    # Filter non-finite values before estimation
    finite_mask = np.isfinite(c2)
    if not np.all(finite_mask):
        c2 = c2[finite_mask]
        dt = dt[finite_mask]

    # Sanity checks
    if len(c2) < 100:
        # Not enough data for robust estimation - return midpoints
        contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
        offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0
        return contrast_mid, offset_mid

    # Find lag thresholds
    lag_threshold_high = np.percentile(dt, lag_floor_quantile * 100)
    lag_threshold_low = np.percentile(dt, lag_ceiling_quantile * 100)

    # OFFSET estimation: From large-lag region where g1² ≈ 0
    # C2 → offset at large lags
    large_lag_mask = dt >= lag_threshold_high
    if np.sum(large_lag_mask) >= 10:
        c2_floor_region = c2[large_lag_mask]
        # Use low quantile for robustness (in case of noise spikes)
        offset_est = np.percentile(c2_floor_region, value_quantile_low * 100)
    else:
        # Fallback: use overall low quantile
        offset_est = np.percentile(c2, value_quantile_low * 100)

    # Clip offset to bounds
    offset_est = float(np.clip(offset_est, offset_bounds[0], offset_bounds[1]))

    # CONTRAST estimation: From small-lag region where g1² ≈ 1
    # C2 ≈ contrast + offset at small lags
    small_lag_mask = dt <= lag_threshold_low
    if np.sum(small_lag_mask) >= 10:
        c2_ceiling_region = c2[small_lag_mask]
        # Use high quantile for robustness
        c2_ceiling = np.percentile(c2_ceiling_region, value_quantile_high * 100)
    else:
        # Fallback: use overall high quantile
        c2_ceiling = np.percentile(c2, value_quantile_high * 100)

    # contrast ≈ c2_ceiling - offset
    contrast_est = c2_ceiling - offset_est

    # Clip contrast to bounds
    contrast_est = float(np.clip(contrast_est, contrast_bounds[0], contrast_bounds[1]))

    return contrast_est, offset_est


def estimate_per_angle_scaling(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
    log: Logger | LoggerAdapter[Logger] | None = None,
) -> dict[str, float]:
    """Estimate contrast and offset initial values for each phi angle.

    This is the unified implementation used by both NLSQ and CMC backends.

    Optimization (v2.9.1): Uses vectorized grouped operations instead of
    sequential loop over angles. Provides 3-5x speedup for typical datasets
    with 20+ phi angles.

    Parameters
    ----------
    c2_data : np.ndarray
        Pooled C2 correlation values.
    t1 : np.ndarray
        Pooled first time coordinates.
    t2 : np.ndarray
        Pooled second time coordinates.
    phi_indices : np.ndarray
        Index mapping each data point to its phi angle (0 to n_phi-1).
    n_phi : int
        Number of unique phi angles.
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast.
    offset_bounds : tuple[float, float]
        Valid bounds for offset.
    log : logging.Logger | None
        Logger for diagnostic messages.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'contrast_0', 'offset_0', 'contrast_1', 'offset_1', etc.
    """
    if log is None:
        log = logger

    c2 = np.asarray(c2_data)
    t1_arr = np.asarray(t1)
    t2_arr = np.asarray(t2)
    phi_idx = np.asarray(phi_indices)

    # Pre-compute time lags once (vectorized)
    delta_t = np.abs(t1_arr - t2_arr)

    # Pre-compute midpoint defaults
    contrast_mid = (contrast_bounds[0] + contrast_bounds[1]) / 2.0
    offset_mid = (offset_bounds[0] + offset_bounds[1]) / 2.0

    # Pre-allocate result arrays
    contrast_results = np.full(n_phi, contrast_mid)
    offset_results = np.full(n_phi, offset_mid)

    # Count points per angle using bincount (vectorized)
    points_per_angle = np.bincount(phi_idx, minlength=n_phi)

    # Identify angles with sufficient data
    sufficient_mask = points_per_angle >= 100
    n_sufficient = np.sum(sufficient_mask)

    if n_sufficient == 0:
        # No angles have enough data - return defaults
        log.info(f"All {n_phi} angles have insufficient data, using midpoint defaults")
        return {
            **{f"contrast_{i}": contrast_mid for i in range(n_phi)},
            **{f"offset_{i}": offset_mid for i in range(n_phi)},
        }

    # Sort data by phi index for efficient grouped operations
    sort_idx = np.argsort(phi_idx)
    c2_sorted = c2[sort_idx]
    delta_t_sorted = delta_t[sort_idx]
    phi_sorted = phi_idx[sort_idx]

    # Find group boundaries
    group_starts = np.searchsorted(phi_sorted, np.arange(n_phi))
    group_ends = np.searchsorted(phi_sorted, np.arange(n_phi), side="right")

    # Process each angle with sufficient data
    for i in range(n_phi):
        if not sufficient_mask[i]:
            log.debug(
                f"Angle {i}: insufficient data ({points_per_angle[i]} points), "
                f"using midpoint init contrast={contrast_mid:.4f}, offset={offset_mid:.4f}"
            )
            continue

        # Extract data for this angle using pre-sorted arrays
        start, end = group_starts[i], group_ends[i]
        c2_angle = c2_sorted[start:end]
        delta_t_angle = delta_t_sorted[start:end]

        # Use shared estimation function
        contrast_est, offset_est = estimate_contrast_offset_from_quantiles(
            c2_angle,
            delta_t_angle,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        contrast_results[i] = contrast_est
        offset_results[i] = offset_est

        log.debug(
            f"Angle {i}: estimated contrast={contrast_est:.4f}, offset={offset_est:.4f} "
            f"from {end - start:,} data points"
        )

    # Build result dictionary
    estimates: dict[str, float] = {}
    for i in range(n_phi):
        estimates[f"contrast_{i}"] = float(contrast_results[i])
        estimates[f"offset_{i}"] = float(offset_results[i])

    return estimates


def compute_averaged_scaling(
    c2_data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_indices: np.ndarray,
    n_phi: int,
    contrast_bounds: tuple[float, float],
    offset_bounds: tuple[float, float],
    log: Logger | LoggerAdapter[Logger] | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute averaged contrast and offset for constant mode.

    This function estimates per-angle contrast/offset using quantile analysis,
    then averages them to produce single values for constant mode optimization.

    Parameters
    ----------
    c2_data : np.ndarray
        Pooled C2 correlation values.
    t1 : np.ndarray
        Pooled first time coordinates.
    t2 : np.ndarray
        Pooled second time coordinates.
    phi_indices : np.ndarray
        Index mapping each data point to its phi angle (0 to n_phi-1).
    n_phi : int
        Number of unique phi angles.
    contrast_bounds : tuple[float, float]
        Valid bounds for contrast.
    offset_bounds : tuple[float, float]
        Valid bounds for offset.
    log : logging.Logger | None
        Logger for diagnostic messages.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray]
        (contrast_avg, offset_avg, contrast_per_angle, offset_per_angle)
        - contrast_avg: Averaged contrast for constant mode
        - offset_avg: Averaged offset for constant mode
        - contrast_per_angle: Per-angle estimates (for diagnostics)
        - offset_per_angle: Per-angle estimates (for diagnostics)
    """
    if log is None:
        log = logger

    # Get per-angle estimates
    estimates = estimate_per_angle_scaling(
        c2_data=c2_data,
        t1=t1,
        t2=t2,
        phi_indices=phi_indices,
        n_phi=n_phi,
        contrast_bounds=contrast_bounds,
        offset_bounds=offset_bounds,
        log=log,
    )

    # Extract arrays
    contrast_per_angle = np.array([estimates[f"contrast_{i}"] for i in range(n_phi)])
    offset_per_angle = np.array([estimates[f"offset_{i}"] for i in range(n_phi)])

    # Compute averaged values
    contrast_avg = float(np.nanmean(contrast_per_angle))
    offset_avg = float(np.nanmean(offset_per_angle))

    log.info(
        f"Computed averaged scaling for constant mode:\n"
        f"  contrast_avg: {contrast_avg:.4f} (per-angle range: "
        f"[{float(np.nanmin(contrast_per_angle)):.4f}, {float(np.nanmax(contrast_per_angle)):.4f}])\n"
        f"  offset_avg: {offset_avg:.4f} (per-angle range: "
        f"[{float(np.nanmin(offset_per_angle)):.4f}, {float(np.nanmax(offset_per_angle)):.4f}])"
    )

    return contrast_avg, offset_avg, contrast_per_angle, offset_per_angle


__all__ = [
    "estimate_contrast_offset_from_quantiles",
    "estimate_per_angle_scaling",
    "compute_averaged_scaling",
]
