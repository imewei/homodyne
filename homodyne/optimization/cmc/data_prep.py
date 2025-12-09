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

    # Create index mapping
    phi_indices = np.searchsorted(phi_unique, phi)

    logger.debug(f"Extracted {n_phi} unique phi angles: {phi_unique}")

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

    logger.debug(f"Estimated noise scale: {noise_scale:.4f} (median={median:.4f})")

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
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "median": float(np.median(data)),
    }


def prepare_mcmc_data(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
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
    max_shards_per_angle: int = 50,
    seed: int = 42,
) -> list[PreparedData]:
    """Shard data by phi angle (stratified sharding).

    Each shard contains data for one phi angle. If a single angle has more
    data points than max_points_per_shard, multiple shards are created for
    that angle by splitting the data randomly.

    When the number of required shards exceeds max_shards_per_angle, data is
    subsampled within each shard to maintain tractable shard counts.

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    num_shards : int | None
        Ignored. Number of shards is determined automatically based on
        data size and max_points_per_shard.
    max_points_per_shard : int | None
        Maximum points per shard. If an angle exceeds this, multiple shards
        are created for that angle. If None, uses one shard per angle.
        Recommended: 25000-100000 for NUTS.
    max_shards_per_angle : int
        Maximum shards to create per angle. If more would be needed, data is
        subsampled within each shard. Default: 50 (balances coverage vs overhead).
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
        if max_points_per_shard is not None and n_points > max_points_per_shard:
            n_angle_shards = (n_points + max_points_per_shard - 1) // max_points_per_shard

            # Cap shards per angle and subsample if needed
            if n_angle_shards > max_shards_per_angle:
                # Too many shards - cap and subsample within each shard
                total_sampled = max_shards_per_angle * max_points_per_shard
                pct_used = 100 * total_sampled / n_points
                logger.info(
                    f"Angle {angle_idx} (phi={angle_phi_value:.4f}): {n_points:,} points → "
                    f"{max_shards_per_angle} shards (capped from {n_angle_shards}, "
                    f"subsampling to {pct_used:.1f}% of data)"
                )
                n_angle_shards = max_shards_per_angle
            else:
                logger.info(
                    f"Angle {angle_idx} (phi={angle_phi_value:.4f}): {n_points:,} points → "
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

            # Calculate points per shard - cap at max_points_per_shard
            raw_points_per_shard = n_points // n_angle_shards
            if max_points_per_shard is not None and raw_points_per_shard > max_points_per_shard:
                # Subsample within each shard
                target_points = max_points_per_shard
            else:
                target_points = raw_points_per_shard

            for j in range(n_angle_shards):
                start_idx = j * raw_points_per_shard
                if j == n_angle_shards - 1:
                    # Last shard gets remaining points
                    end_idx = n_points
                else:
                    end_idx = (j + 1) * raw_points_per_shard

                shard_indices = indices[start_idx:end_idx]

                # Subsample if this shard exceeds target
                if len(shard_indices) > target_points:
                    shard_indices = rng.choice(shard_indices, target_points, replace=False)

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
                    f"angle {angle_idx} part {j+1}/{n_angle_shards} (phi={angle_phi_value:.4f})"
                )
                shard_idx += 1

    logger.info(f"Created {len(shards)} total shards from {prepared.n_phi} angles")
    return shards


def shard_data_random(
    prepared: PreparedData,
    num_shards: int,
    max_points_per_shard: int | None = None,
    seed: int = 42,
) -> list[PreparedData]:
    """Shard data randomly into approximately equal parts.

    This is used when there's only one phi angle but the dataset is too
    large for efficient NUTS sampling. Each shard gets a random subset
    of the data.

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    num_shards : int
        Number of shards to create.
    max_points_per_shard : int | None
        Maximum points per shard. If total/num_shards exceeds this,
        data is subsampled. Recommended: 50000-100000 for NUTS.
    seed : int
        Random seed for reproducible shuffling.

    Returns
    -------
    list[PreparedData]
        List of shard data objects.
    """
    rng = np.random.default_rng(seed)

    # Shuffle indices
    indices = np.arange(prepared.n_total)
    rng.shuffle(indices)

    # Split into shards
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

        # Apply max_points_per_shard limit
        if max_points_per_shard is not None and len(shard_indices) > max_points_per_shard:
            original_size = len(shard_indices)
            shard_indices = rng.choice(shard_indices, max_points_per_shard, replace=False)
            logger.warning(
                f"Shard {i}: Subsampled from {original_size:,} to {max_points_per_shard:,} points "
                f"for MCMC tractability"
            )

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
            f"Shard {i}: {len(shard_data):,} points (random split)"
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
