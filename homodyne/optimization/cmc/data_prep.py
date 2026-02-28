"""Data preparation and validation for CMC analysis.

This module provides utilities for validating and preparing pooled XPCS data
for MCMC sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreparedData:
    """Validated and prepared data for MCMC sampling.

    Attributes
    ----------
    data : np.ndarray
        Pooled C2 correlation data, shape (n_total,).
    t1 : np.ndarray
        Pooled time coordinates t1, shape (n_total,).
    t2 : np.ndarray
        Pooled time coordinates t2, shape (n_total,).
    phi : np.ndarray
        Pooled phi angles, shape (n_total,).
    phi_unique : np.ndarray
        Unique phi angles, shape (n_phi,).
    phi_indices : np.ndarray
        Index of phi_unique for each data point, shape (n_total,).
    n_total : int
        Total number of data points.
    n_phi : int
        Number of unique phi angles.
    noise_scale : float
        Estimated observation noise scale.
    """

    data: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    phi: np.ndarray
    phi_unique: np.ndarray
    phi_indices: np.ndarray
    n_total: int
    n_phi: int
    noise_scale: float


def validate_pooled_data(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
) -> None:
    """Validate that pooled data arrays are consistent.

    Parameters
    ----------
    data : np.ndarray
        Pooled C2 correlation data.
    t1 : np.ndarray
        Pooled time coordinates t1.
    t2 : np.ndarray
        Pooled time coordinates t2.
    phi : np.ndarray
        Pooled phi angles.

    Raises
    ------
    ValueError
        If arrays have inconsistent shapes or contain invalid values.
    """
    # Check 1D arrays
    if data.ndim != 1:
        raise ValueError(f"data must be 1D, got shape {data.shape}")
    if t1.ndim != 1:
        raise ValueError(f"t1 must be 1D, got shape {t1.shape}")
    if t2.ndim != 1:
        raise ValueError(f"t2 must be 1D, got shape {t2.shape}")
    if phi.ndim != 1:
        raise ValueError(f"phi must be 1D, got shape {phi.shape}")

    # Check matching lengths
    n = len(data)
    if len(t1) != n:
        raise ValueError(f"t1 length {len(t1)} does not match data length {n}")
    if len(t2) != n:
        raise ValueError(f"t2 length {len(t2)} does not match data length {n}")
    if len(phi) != n:
        raise ValueError(f"phi length {len(phi)} does not match data length {n}")

    # Check for NaN/inf in data
    if np.any(np.isnan(data)):
        nan_count = np.sum(np.isnan(data))
        raise ValueError(f"data contains {nan_count} NaN values")
    if np.any(np.isinf(data)):
        inf_count = np.sum(np.isinf(data))
        raise ValueError(f"data contains {inf_count} infinite values")

    # Check for NaN/inf in coordinates
    for name, arr in [("t1", t1), ("t2", t2), ("phi", phi)]:
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN values")
        if np.any(np.isinf(arr)):
            raise ValueError(f"{name} contains infinite values")

    # Check data range (C2 should be physically reasonable)
    if np.any(data < 0):
        neg_count = np.sum(data < 0)
        logger.warning(
            f"data contains {neg_count} negative values (may indicate noise)"
        )
    if np.any(data > 10):
        high_count = np.sum(data > 10)
        logger.warning(f"data contains {high_count} values > 10 (unusual for C2)")

    # Check time coordinates are non-negative
    if np.any(t1 < 0) or np.any(t2 < 0):
        raise ValueError("Time coordinates must be non-negative")

    logger.debug(f"Data validation passed: {n:,} points")


def extract_phi_info(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique phi angles and compute index mapping.

    Parameters
    ----------
    phi : np.ndarray
        Pooled phi angles, shape (n_total,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - phi_unique: Unique phi values sorted, shape (n_phi,)
        - phi_indices: Index into phi_unique for each point, shape (n_total,)
    """
    phi_unique = np.unique(phi)
    n_phi = len(phi_unique)

    # P2-2: Use tolerance-aware nearest-neighbor matching instead of searchsorted.
    # searchsorted + clip silently assigns points to the wrong angle when phi
    # values have float precision differences (e.g., after dtype conversion).
    # np.argmin(|phi - phi_unique|) always finds the closest angle.
    if n_phi <= 256:
        # Nearest-neighbor: O(n * n_phi), feasible for typical angle counts
        phi_indices = np.argmin(
            np.abs(phi[:, None] - phi_unique[None, :]), axis=1
        ).astype(np.int32)
    else:
        # Fallback for extremely many angles: searchsorted + nearest-neighbor
        # searchsorted returns insertion point; check both neighbors to find
        # the closest match (handles float precision differences).
        idx = np.searchsorted(phi_unique, phi)
        idx = np.clip(idx, 0, n_phi - 1)
        # Check if the left neighbor (idx-1) is closer
        left = np.clip(idx - 1, 0, n_phi - 1)
        use_left = np.abs(phi - phi_unique[left]) < np.abs(phi - phi_unique[idx])
        phi_indices = np.where(use_left, left, idx).astype(np.int32)

    return phi_unique, phi_indices


def estimate_noise_scale(data: np.ndarray) -> float:
    """Estimate observation noise scale from data.

    Uses robust MAD (Median Absolute Deviation) estimator scaled
    to approximate standard deviation for Gaussian noise.

    Parameters
    ----------
    data : np.ndarray
        Observed data values.

    Returns
    -------
    float
        Estimated noise scale (standard deviation).
    """
    # Use MAD for robust estimation
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Scale MAD to approximate std for Gaussian
    # sigma = MAD * 1.4826
    noise_scale = float(mad * 1.4826)

    # Ensure reasonable minimum
    noise_scale = max(noise_scale, 0.001)

    return noise_scale


def compute_data_statistics(data: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for data.

    Parameters
    ----------
    data : np.ndarray
        Data array.

    Returns
    -------
    dict[str, float]
        Statistics including min, max, mean, std, median.
    """
    return {
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
        "median": float(np.nanmedian(data)),
    }


def prepare_mcmc_data(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    filter_diagonal: bool = True,
) -> PreparedData:
    """Prepare and validate data for MCMC sampling.

    Parameters
    ----------
    data : np.ndarray
        Pooled C2 correlation data, shape (n_total,).
    t1 : np.ndarray
        Pooled time coordinates t1, shape (n_total,).
    t2 : np.ndarray
        Pooled time coordinates t2, shape (n_total,).
    phi : np.ndarray
        Pooled phi angles, shape (n_total,).
    filter_diagonal : bool, default=True
        If True, exclude diagonal points (t1 == t2) from the dataset.
        Diagonal points represent autocorrelation peaks that are corrected
        at load time but should not contribute to the likelihood.
        Added in v2.14.2 for consistency with NLSQ diagonal handling.

    Returns
    -------
    PreparedData
        Validated and prepared data object.

    Raises
    ------
    ValueError
        If data validation fails.
    """
    # Ensure numpy arrays
    data = np.asarray(data)
    t1 = np.asarray(t1)
    t2 = np.asarray(t2)
    phi = np.asarray(phi)

    # Validate
    validate_pooled_data(data, t1, t2, phi)

    # v2.14.2+: Filter out diagonal points (t1 == t2)
    # Diagonal points have autocorrelation peaks that are corrected at load time,
    # but the corrected values are interpolated estimates. Excluding them from
    # the likelihood avoids biasing the fit with synthetic data.
    # This is consistent with NLSQ strategies that mask/filter diagonal residuals.
    if filter_diagonal:
        n_before = len(data)
        # Use epsilon-based comparison instead of exact equality to handle float
        # rounding errors in time arrays derived from integer frame indices.
        # Epsilon = 1 ppm of the smallest time step (or 1e-12 fallback).
        _t_unique = np.unique(np.concatenate([t1, t2]))
        _diffs = np.diff(_t_unique)
        _dt_min = (
            float(_diffs[_diffs > 0].min())
            if _diffs.size > 0 and np.any(_diffs > 0)
            else 1.0
        )
        _diag_eps = max(_dt_min * 1e-6, 1e-12)
        non_diagonal_mask = np.abs(t1 - t2) > _diag_eps
        data = data[non_diagonal_mask]
        t1 = t1[non_diagonal_mask]
        t2 = t2[non_diagonal_mask]
        phi = phi[non_diagonal_mask]
        n_filtered = n_before - len(data)
        if n_filtered > 0:
            logger.info(
                f"Filtered {n_filtered:,} diagonal points (t1==t2), "
                f"{len(data):,} points remaining"
            )
        if len(data) == 0:
            raise ValueError(
                f"All {n_before:,} data points were diagonal (t1==t2). "
                "No off-diagonal points remain after filtering. "
                "Check data preparation or use filter_diagonal=False."
            )

    # Extract phi info
    phi_unique, phi_indices = extract_phi_info(phi)

    # Estimate noise
    noise_scale = estimate_noise_scale(data)

    # Log statistics
    stats = compute_data_statistics(data)
    logger.info(
        f"Prepared {len(data):,} data points from {len(phi_unique)} angles, "
        f"range=[{stats['min']:.3f}, {stats['max']:.3f}], "
        f"noise_scale={noise_scale:.4f}"
    )

    return PreparedData(
        data=data,
        t1=t1,
        t2=t2,
        phi=phi,
        phi_unique=phi_unique,
        phi_indices=phi_indices,
        n_total=len(data),
        n_phi=len(phi_unique),
        noise_scale=noise_scale,
    )


def shard_data_stratified(
    prepared: PreparedData,
    num_shards: int | None = None,
    max_points_per_shard: int | None = None,
    max_shards_per_angle: int = 100,
    seed: int = 42,
) -> list[PreparedData]:
    """Shard data by phi angle (stratified sharding).

    Each shard contains data for one phi angle. If a single angle has more
    data points than max_points_per_shard, multiple shards are created for
    that angle by splitting the data randomly.

    When the number of required shards exceeds max_shards_per_angle, shard
    size increases to fit all data (no subsampling).

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    num_shards : int | None
        Desired total shard count. When provided, it forces a target shard
        size; max_points_per_shard is derived if not set.
    max_points_per_shard : int | None
        Maximum points per shard. If an angle exceeds this, multiple shards
        are created for that angle. If None, uses one shard per angle unless
        num_shards is provided (then it is derived).
        Recommended: 25000-100000 for NUTS.
    max_shards_per_angle : int
        Maximum shards to create per angle. If more would be needed, shard
        size increases to fit all data. Default: 100.
    seed : int
        Random seed for reproducible splitting.

    Returns
    -------
    list[PreparedData]
        List of shard data objects.
    """
    shards: list[PreparedData] = []
    rng = np.random.default_rng(seed)
    shard_idx = 0

    if num_shards is not None and num_shards > 0 and max_points_per_shard is None:
        # Derive an approximate per-shard target to honor explicit shard counts
        max_points_per_shard = (prepared.n_total + num_shards - 1) // num_shards

    # Process each phi angle separately
    for angle_idx in range(prepared.n_phi):
        # Get data for this angle
        mask = prepared.phi_indices == angle_idx
        angle_data = prepared.data[mask]
        angle_t1 = prepared.t1[mask]
        angle_t2 = prepared.t2[mask]
        angle_phi = prepared.phi[mask]
        angle_phi_value = prepared.phi_unique[angle_idx]

        n_points = len(angle_data)

        # Determine how many shards needed for this angle
        # effective_max_points tracks actual points per shard (may exceed max_points_per_shard if capped)
        effective_max_points = max_points_per_shard

        if max_points_per_shard is not None and n_points > max_points_per_shard:
            n_angle_shards = (
                n_points + max_points_per_shard - 1
            ) // max_points_per_shard

            # Cap shards per angle - increase points per shard to use ALL data
            if n_angle_shards > max_shards_per_angle:
                n_angle_shards = max_shards_per_angle
                # Recalculate points per shard to fit all data into capped shards
                effective_max_points = (
                    n_points + max_shards_per_angle - 1
                ) // max_shards_per_angle
                logger.debug(
                    f"Angle {angle_idx} (phi={angle_phi_value:.4f}): {n_points:,} points -> "
                    f"{max_shards_per_angle} shards (~{effective_max_points:,} points each, "
                    f"increased from {max_points_per_shard:,} to fit all data)"
                )
            else:
                logger.debug(
                    f"Angle {angle_idx} (phi={angle_phi_value:.4f}): {n_points:,} points -> "
                    f"{n_angle_shards} shards (~{max_points_per_shard:,} points each)"
                )
        else:
            n_angle_shards = 1

        if n_angle_shards == 1:
            # Single shard for this angle - no splitting needed
            shard_phi_unique, shard_phi_indices = extract_phi_info(angle_phi)
            shard_noise = estimate_noise_scale(angle_data)

            shards.append(
                PreparedData(
                    data=angle_data,
                    t1=angle_t1,
                    t2=angle_t2,
                    phi=angle_phi,
                    phi_unique=shard_phi_unique,
                    phi_indices=shard_phi_indices,
                    n_total=len(angle_data),
                    n_phi=len(shard_phi_unique),
                    noise_scale=shard_noise,
                )
            )
            logger.debug(
                f"Shard {shard_idx}: {len(angle_data):,} points, "
                f"angle {angle_idx} (phi={angle_phi_value:.4f})"
            )
            shard_idx += 1
        else:
            # Split this angle's data into multiple shards
            indices = np.arange(n_points)
            rng.shuffle(indices)

            # Calculate points per shard - use all data
            points_per_shard = n_points // n_angle_shards

            for j in range(n_angle_shards):
                start_idx = j * points_per_shard
                if j == n_angle_shards - 1:
                    # Last shard gets remaining points
                    end_idx = n_points
                else:
                    end_idx = (j + 1) * points_per_shard

                shard_indices = indices[start_idx:end_idx]

                # Sort to preserve temporal structure within shard
                shard_indices = np.sort(shard_indices)

                shard_data = angle_data[shard_indices]
                shard_t1 = angle_t1[shard_indices]
                shard_t2 = angle_t2[shard_indices]
                shard_phi = angle_phi[shard_indices]

                shard_phi_unique, shard_phi_indices = extract_phi_info(shard_phi)
                shard_noise = estimate_noise_scale(shard_data)

                shards.append(
                    PreparedData(
                        data=shard_data,
                        t1=shard_t1,
                        t2=shard_t2,
                        phi=shard_phi,
                        phi_unique=shard_phi_unique,
                        phi_indices=shard_phi_indices,
                        n_total=len(shard_data),
                        n_phi=len(shard_phi_unique),
                        noise_scale=shard_noise,
                    )
                )
                logger.debug(
                    f"Shard {shard_idx}: {len(shard_data):,} points, "
                    f"angle {angle_idx} part {j + 1}/{n_angle_shards} (phi={angle_phi_value:.4f})"
                )
                shard_idx += 1

    logger.info(f"Created {len(shards)} total shards from {prepared.n_phi} angles")
    return shards


def shard_data_random(
    prepared: PreparedData,
    num_shards: int | None = None,
    max_points_per_shard: int | None = None,
    max_shards: int = 100,
    seed: int = 42,
) -> list[PreparedData]:
    """Shard data randomly into approximately equal parts.

    This is used when there's only one phi angle but the dataset is too
    large for efficient NUTS sampling. Each shard gets a random subset
    of the data. ALL data is used (no subsampling).

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    num_shards : int | None
        Number of shards to create. If None, calculated from data size
        and max_points_per_shard.
    max_points_per_shard : int | None
        Target points per shard. Used to calculate num_shards if not provided.
        If num_shards would exceed max_shards, shard size increases to fit all data.
        Recommended: 25000-100000 for NUTS.
    max_shards : int
        Maximum number of shards. Default: 100.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    list[PreparedData]
        List of shard data objects.
    """
    rng = np.random.default_rng(seed)

    # Calculate number of shards if not provided
    if num_shards is None:
        if max_points_per_shard is not None:
            num_shards = (
                prepared.n_total + max_points_per_shard - 1
            ) // max_points_per_shard
        else:
            num_shards = 1

    # Cap shards and increase shard size to use ALL data
    if num_shards > max_shards:
        effective_points_per_shard = (prepared.n_total + max_shards - 1) // max_shards
        logger.info(
            f"Random sharding: {prepared.n_total:,} points -> {max_shards} shards "
            f"(~{effective_points_per_shard:,} points each"
            f"{f', increased from {max_points_per_shard:,}' if max_points_per_shard is not None else ''}"
            f" to fit all data)"
        )
        num_shards = max_shards
    else:
        points_per_shard = prepared.n_total // num_shards
        logger.info(
            f"Random sharding: {prepared.n_total:,} points -> {num_shards} shards "
            f"(~{points_per_shard:,} points each)"
        )

    # Shuffle indices
    indices = np.arange(prepared.n_total)
    rng.shuffle(indices)

    # Split into shards - use ALL data
    points_per_shard = prepared.n_total // num_shards
    shards: list[PreparedData] = []

    for i in range(num_shards):
        start_idx = i * points_per_shard
        if i == num_shards - 1:
            # Last shard gets remaining points
            end_idx = prepared.n_total
        else:
            end_idx = (i + 1) * points_per_shard

        shard_indices = indices[start_idx:end_idx]

        # Sort to preserve some temporal structure
        shard_indices = np.sort(shard_indices)

        # Extract shard data
        shard_data = prepared.data[shard_indices]
        shard_t1 = prepared.t1[shard_indices]
        shard_t2 = prepared.t2[shard_indices]
        shard_phi = prepared.phi[shard_indices]

        # Create shard PreparedData
        shard_phi_unique, shard_phi_indices = extract_phi_info(shard_phi)
        shard_noise = estimate_noise_scale(shard_data)

        shards.append(
            PreparedData(
                data=shard_data,
                t1=shard_t1,
                t2=shard_t2,
                phi=shard_phi,
                phi_unique=shard_phi_unique,
                phi_indices=shard_phi_indices,
                n_total=len(shard_data),
                n_phi=len(shard_phi_unique),
                noise_scale=shard_noise,
            )
        )

    logger.debug(
        f"Random sharding complete: {num_shards} shards, "
        f"~{points_per_shard:,} points each"
    )

    return shards


def shard_data_angle_balanced(
    prepared: PreparedData,
    num_shards: int | None = None,
    max_points_per_shard: int | None = None,
    max_shards: int = 500,
    min_angle_coverage: float = 0.8,
    seed: int = 42,
) -> list[PreparedData]:
    """Shard data with balanced angle coverage per shard.

    This is the preferred sharding method for multi-angle datasets (n_phi > 1)
    when using random/mixed sharding. Unlike pure random sharding, this method
    ensures each shard has representative coverage from each phi angle.

    CRITICAL (Jan 2026): Prevents heterogeneous posteriors that cause high CV
    across shards. The D_offset CV=1.58 failure case was caused by pure random
    sharding creating shards with uneven angle coverage.

    Algorithm:
    1. Shuffle data within each angle independently
    2. For each shard, sample proportionally from each angle
    3. Verify angle coverage meets minimum threshold
    4. Log coverage statistics for diagnostics

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object with multi-angle data.
    num_shards : int | None
        Number of shards to create. If None, calculated from data size
        and max_points_per_shard.
    max_points_per_shard : int | None
        Target points per shard. Used to calculate num_shards if not provided.
        Recommended: 3000-10000 for laminar_flow with few angles.
    max_shards : int
        Maximum number of shards. Default: 500 (higher than random to allow
        smaller shards for multi-angle data).
    min_angle_coverage : float
        Minimum fraction of angles that must be present in each shard.
        Default: 0.8 (80% of angles). Shards below this threshold are logged.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    list[PreparedData]
        List of shard data objects, each with balanced angle coverage.

    Notes
    -----
    - ALL data is used (no subsampling)
    - Each shard aims to have the same proportion of each angle as the full dataset
    - The last shard may have slightly different proportions to include all data
    """
    rng = np.random.default_rng(seed)
    n_phi = prepared.n_phi

    # If only one angle, fall back to random sharding
    if n_phi == 1:
        logger.info("Single angle detected - falling back to random sharding")
        return shard_data_random(
            prepared, num_shards, max_points_per_shard, max_shards, seed
        )

    # Calculate number of shards if not provided
    if num_shards is None:
        if max_points_per_shard is not None:
            num_shards = (
                prepared.n_total + max_points_per_shard - 1
            ) // max_points_per_shard
        else:
            num_shards = max(1, n_phi)  # At least one shard per angle

    # Cap shards and adjust points per shard
    if num_shards > max_shards:
        num_shards = max_shards

    # Ensure at least 1 shard
    num_shards = max(1, num_shards)

    # Group data indices by angle
    angle_indices: list[np.ndarray] = []
    angle_counts: list[int] = []
    for angle_idx in range(n_phi):
        mask = prepared.phi_indices == angle_idx
        indices = np.where(mask)[0]
        rng.shuffle(indices)  # Shuffle within each angle
        angle_indices.append(indices)
        angle_counts.append(len(indices))

    # Calculate target points per shard per angle (proportional allocation)

    # Track how many points we've used from each angle
    angle_positions = [0] * n_phi

    shards: list[PreparedData] = []
    coverage_stats: list[float] = []

    for shard_num in range(num_shards):
        shard_indices_list: list[np.ndarray] = []

        # Calculate how many points to take from each angle for this shard
        is_last_shard = shard_num == num_shards - 1

        for angle_idx in range(n_phi):
            angle_total = angle_counts[angle_idx]
            already_used = angle_positions[angle_idx]
            remaining_in_angle = angle_total - already_used

            if is_last_shard:
                # Last shard takes all remaining points
                n_take = remaining_in_angle
            else:
                # Proportional allocation: same fraction from each angle
                target = int(angle_total / num_shards)
                n_take = min(target, remaining_in_angle)
                # Ensure we don't leave too few points for remaining shards
                remaining_shards = num_shards - shard_num
                min_take = remaining_in_angle // remaining_shards
                n_take = max(n_take, min_take)

            if n_take > 0:
                start = angle_positions[angle_idx]
                end = start + n_take
                shard_indices_list.append(angle_indices[angle_idx][start:end])
                angle_positions[angle_idx] = end

        # Combine indices from all angles
        if shard_indices_list:
            shard_all_indices = np.concatenate(shard_indices_list)
        else:
            # Edge case: empty shard (shouldn't happen but handle gracefully)
            continue

        # Sort to preserve temporal structure
        shard_all_indices = np.sort(shard_all_indices)

        # Extract shard data
        shard_data = prepared.data[shard_all_indices]
        shard_t1 = prepared.t1[shard_all_indices]
        shard_t2 = prepared.t2[shard_all_indices]
        shard_phi = prepared.phi[shard_all_indices]

        # Create shard PreparedData
        shard_phi_unique, shard_phi_indices = extract_phi_info(shard_phi)
        shard_noise = estimate_noise_scale(shard_data)

        # Calculate angle coverage for this shard
        shard_n_phi = len(shard_phi_unique)
        coverage = shard_n_phi / n_phi
        coverage_stats.append(coverage)

        shards.append(
            PreparedData(
                data=shard_data,
                t1=shard_t1,
                t2=shard_t2,
                phi=shard_phi,
                phi_unique=shard_phi_unique,
                phi_indices=shard_phi_indices,
                n_total=len(shard_data),
                n_phi=shard_n_phi,
                noise_scale=shard_noise,
            )
        )

        if coverage < min_angle_coverage:
            logger.warning(
                f"Shard {shard_num}: {len(shard_data):,} points, "
                f"angle coverage {coverage:.1%} < {min_angle_coverage:.1%} threshold "
                f"({shard_n_phi}/{n_phi} angles)"
            )

    # Log summary statistics
    if coverage_stats:
        min_cov = min(coverage_stats)
        max_cov = max(coverage_stats)
        mean_cov = sum(coverage_stats) / len(coverage_stats)
        below_threshold = sum(1 for c in coverage_stats if c < min_angle_coverage)
        total_shard_points = sum(s.n_total for s in shards)

        logger.info(
            f"Angle-balanced sharding: {prepared.n_total:,} points -> {len(shards)} shards "
            f"(~{total_shard_points // len(shards):,} points each)"
        )
        logger.info(
            f"Angle coverage stats: min={min_cov:.1%}, max={max_cov:.1%}, mean={mean_cov:.1%}, "
            f"below threshold: {below_threshold}/{len(shards)}"
        )

        if below_threshold > len(shards) * 0.1:  # More than 10% below threshold
            logger.warning(
                f"Many shards ({below_threshold}) have low angle coverage. "
                f"Consider using fewer shards or larger max_points_per_shard."
            )

    return shards


def create_xdata_dict(
    prepared: PreparedData,
    q: float,
    L: float,
    dt: float,
    analysis_mode: str,
) -> dict[str, Any]:
    """Create xdata dictionary for physics model.

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    q : float
        Wavevector magnitude.
    L : float
        Stator-rotor gap length.
    dt : float
        Time step.
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow").

    Returns
    -------
    dict[str, Any]
        Dictionary of model inputs.
    """
    return {
        "data": prepared.data,
        "t1": prepared.t1,
        "t2": prepared.t2,
        "phi": prepared.phi,
        "phi_unique": prepared.phi_unique,
        "phi_indices": prepared.phi_indices,
        "q": q,
        "L": L,
        "dt": dt,
        "analysis_mode": analysis_mode,
        "n_phi": prepared.n_phi,
        "n_total": prepared.n_total,
        "noise_scale": prepared.noise_scale,
    }
