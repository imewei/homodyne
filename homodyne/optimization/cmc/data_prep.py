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
        logger.warning(
            f"data contains {high_count} values > 10 (unusual for C2)"
        )

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
) -> list[PreparedData]:
    """Shard data by phi angle (stratified sharding).

    Each shard contains all data for one or more phi angles.

    Parameters
    ----------
    prepared : PreparedData
        Prepared data object.
    num_shards : int | None
        Number of shards. If None, uses one shard per angle.

    Returns
    -------
    list[PreparedData]
        List of shard data objects.
    """
    if num_shards is None:
        num_shards = prepared.n_phi

    shards: list[PreparedData] = []

    # Group phi angles into shards
    angles_per_shard = max(1, prepared.n_phi // num_shards)

    for i in range(num_shards):
        # Determine which angles go in this shard
        start_angle = i * angles_per_shard
        if i == num_shards - 1:
            # Last shard gets remaining angles
            end_angle = prepared.n_phi
        else:
            end_angle = (i + 1) * angles_per_shard

        # Get indices for these angles
        mask = np.zeros(prepared.n_total, dtype=bool)
        for angle_idx in range(start_angle, end_angle):
            mask |= prepared.phi_indices == angle_idx

        # Extract shard data
        shard_data = prepared.data[mask]
        shard_t1 = prepared.t1[mask]
        shard_t2 = prepared.t2[mask]
        shard_phi = prepared.phi[mask]

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
            f"Shard {i}: {len(shard_data):,} points, "
            f"{len(shard_phi_unique)} angles: {shard_phi_unique}"
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
