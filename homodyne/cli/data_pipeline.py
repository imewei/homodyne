"""Data loading and preparation pipeline for Homodyne CLI.

Handles experimental data loading, t=0 exclusion, angle filtering,
CMC config preparation, and MCMC data pooling.
"""

from __future__ import annotations

import argparse
import time
from typing import Any, cast

import numpy as np

from homodyne.data.angle_filtering import (
    apply_angle_filtering as _data_apply_angle_filtering,
)
from homodyne.data.angle_filtering import (
    normalize_angle_to_symmetric_range,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Common XPCS experimental angles (in degrees) for validation
COMMON_XPCS_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180]

# Import core modules with fallback
try:
    from homodyne.config.manager import ConfigManager

    _HAS_CONFIG_MANAGER = True
except ImportError:
    _HAS_CONFIG_MANAGER = False


def _exclude_t0_from_analysis(data: dict[str, Any]) -> dict[str, Any]:
    """Exclude t=0 (index 0) from time arrays and C2 data for analysis.

    With anomalous diffusion D(t) = D0 * t^alpha where alpha < 0,
    D(t=0) approaches infinity. This singularity corrupts the cumulative
    integral in g1 computation, causing:
    - CMC: 0% acceptance rate due to constant likelihood
    - NLSQ: Potential numerical instability

    The fix: exclude index 0 from analysis arrays. The physics at t=0 is
    known analytically (g1(t,t) = 1 by definition), not computed numerically.

    Parameters
    ----------
    data : dict[str, Any]
        Data dictionary containing 't1', 't2', and 'c2_exp' arrays.

    Returns
    -------
    dict[str, Any]
        Modified data dictionary with t=0 excluded from arrays.
        Returns input unchanged if arrays are missing or too small.
    """
    # Extract arrays
    t1_raw = data.get("t1")
    t2_raw = data.get("t2")
    c2_raw = data.get("c2_exp")

    # Validate arrays exist
    if t1_raw is None or t2_raw is None or c2_raw is None:
        logger.debug("Skipping t=0 exclusion: missing t1, t2, or c2_exp arrays")
        return data

    # Convert to numpy arrays if needed
    t1_raw = np.asarray(t1_raw)
    t2_raw = np.asarray(t2_raw)
    c2_raw = np.asarray(c2_raw)

    # Validate minimum size (need at least 2 points to slice)
    min_size = min(
        t1_raw.shape[0] if t1_raw.ndim >= 1 else 1,
        t2_raw.shape[0] if t2_raw.ndim >= 1 else 1,
        c2_raw.shape[-1] if c2_raw.ndim >= 2 else 1,
    )
    if min_size < 2:
        logger.debug("Skipping t=0 exclusion: arrays too small to slice")
        return data

    # Create modified copy of data
    result = data.copy()

    # Slice based on array dimensions
    if t1_raw.ndim == 2 and t2_raw.ndim == 2:
        # 2D meshgrids: slice both axes [1:, 1:]
        result["t1"] = t1_raw[1:, 1:]
        result["t2"] = t2_raw[1:, 1:]
        old_shape = c2_raw.shape
        result["c2_exp"] = c2_raw[:, 1:, 1:]  # All phi angles, slice time axes
        logger.info(
            f"Excluded t=0 for analysis (2D meshgrid): "
            f"c2_exp {old_shape} -> {result['c2_exp'].shape}, "
            f"t1 {t1_raw.shape} -> {result['t1'].shape}"
        )
    elif t1_raw.ndim == 1 and t2_raw.ndim == 1:
        # 1D arrays: slice directly [1:]
        result["t1"] = t1_raw[1:]
        result["t2"] = t2_raw[1:]
        old_shape = c2_raw.shape
        result["c2_exp"] = c2_raw[:, 1:, 1:]  # All phi angles, slice time axes
        logger.info(
            f"Excluded t=0 for analysis (1D arrays): "
            f"c2_exp {old_shape} -> {result['c2_exp'].shape}, "
            f"t1 {t1_raw.shape} -> {result['t1'].shape}"
        )
    else:
        logger.warning(
            f"Unexpected array dimensions: t1.ndim={t1_raw.ndim}, t2.ndim={t2_raw.ndim}. "
            f"Skipping t=0 exclusion."
        )
        return data

    return result


def _apply_angle_filtering_for_optimization(
    data: dict[str, Any],
    config: ConfigManager,
) -> dict[str, Any]:
    """Apply angle filtering to data before optimization.

    This function filters phi angles and corresponding C2 data based on the
    phi_filtering configuration before passing data to optimization methods
    (NLSQ or MCMC). It creates a filtered copy of the data dictionary while
    preserving all other keys unchanged.

    Parameters
    ----------
    data : dict
        Full data dictionary with all angles, containing keys:
        - phi_angles_list: np.ndarray of phi angles (n_phi,)
        - c2_exp: np.ndarray of correlation data (n_phi, n_t1, n_t2)
        - wavevector_q_list: np.ndarray (preserved unchanged)
        - t1: np.ndarray (preserved unchanged)
        - t2: np.ndarray (preserved unchanged)
    config : ConfigManager
        Configuration manager with phi_filtering settings

    Returns
    -------
    dict
        Filtered data dictionary with same structure as input but with:
        - phi_angles_list: Filtered to selected angles only
        - c2_exp: First dimension sliced to match selected angles
        - All other keys: Unchanged from input

    Notes
    -----
    Edge Case Handling:
    - If phi_filtering.enabled is False: Returns unfiltered data (DEBUG log)
    - If target_ranges is empty: Returns unfiltered data (WARNING log)
    - If no angles match: Returns unfiltered data (WARNING log: "No angles matched phi_filtering criteria, using all angles")

    Logging:
    - DEBUG: "Phi filtering not enabled, using all angles for optimization"
    - DEBUG: "Angle filtering completed in X.XXXms" (performance monitoring)
    - INFO: "Angle filtering for optimization: X angles selected from Y total angles"
    - INFO: "Selected angles: [angle_list]"
    - WARNING: "Phi filtering enabled but no target_ranges specified, using all angles"
    - WARNING: "No angles matched phi_filtering criteria, using all angles"
    - WARNING: "Configured angle ranges do not overlap with common XPCS angles" (config validation)

    Examples
    --------
    >>> # With filtering enabled
    >>> filtered_data = _apply_angle_filtering_for_optimization(data, config)
    >>> len(filtered_data["phi_angles_list"])  # e.g., 3 angles selected
    3
    >>> filtered_data["c2_exp"].shape[0]  # First dimension matches
    3
    """
    import numpy as np

    # Extract required arrays
    phi_angles = np.asarray(data.get("phi_angles_list", []))
    c2_exp = np.asarray(data.get("c2_exp", []))

    if len(phi_angles) == 0 or len(c2_exp) == 0:
        logger.warning("No phi angles or C2 data available, cannot apply filtering")
        return data

    # Validate angles are in reasonable range (data quality check)
    angles_too_large = phi_angles[np.abs(phi_angles) > 360]
    if len(angles_too_large) > 0:
        logger.warning(
            f"Found {len(angles_too_large)} angle(s) with |phi| > 360 deg: {angles_too_large}. "
            f"This may indicate data loading issues, unit confusion (radians vs degrees), "
            f"or instrument malfunction. Angles will be normalized to [-180 deg, 180 deg] range.",
        )

    # Normalize phi angles to [-180, 180] range (flow direction at 0)
    original_phi_angles = np.asarray(phi_angles).copy()
    phi_angles_normalized = normalize_angle_to_symmetric_range(np.asarray(phi_angles))
    phi_angles = np.asarray(phi_angles_normalized)
    logger.info(
        "Normalized phi angles to [-180, 180] deg range (flow direction at 0 deg)"
    )
    logger.debug(f"Original angles: {original_phi_angles}")
    logger.debug(f"Normalized angles: {phi_angles}")

    # Get config dict (handle both ConfigManager and dict types)
    config_dict: dict[str, Any] = (
        config.get_config()
        if hasattr(config, "get_config")
        else cast(dict[str, Any], config)
    )

    # Check if filtering is enabled
    phi_filtering_config = config_dict.get("phi_filtering", {})
    if not phi_filtering_config.get("enabled", False):
        logger.debug("Phi filtering not enabled, using all angles for optimization")
        # Return data with normalized angles even when filtering disabled
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Check for target_ranges
    target_ranges = phi_filtering_config.get("target_ranges", [])
    if not target_ranges:
        logger.warning(
            "Phi filtering enabled but no target_ranges specified, using all angles",
        )
        # Return data with normalized angles
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Normalize target_ranges to [-180, 180] for consistency
    normalized_ranges = []
    for range_spec in target_ranges:
        min_angle = range_spec.get("min_angle", -180)
        max_angle = range_spec.get("max_angle", 180)
        normalized_min = normalize_angle_to_symmetric_range(min_angle)
        normalized_max = normalize_angle_to_symmetric_range(max_angle)
        normalized_ranges.append(
            {
                "min_angle": normalized_min,
                "max_angle": normalized_max,
                "description": range_spec.get("description", ""),
            },
        )
        logger.debug(
            f"Normalized range [{min_angle:.1f} deg, {max_angle:.1f} deg] -> "
            f"[{normalized_min:.1f} deg, {normalized_max:.1f} deg]",
        )
    target_ranges = normalized_ranges

    # Validate that target_ranges overlap with common XPCS angles
    # This helps catch configuration errors (e.g., typos in angle values)
    common_angles_matched = False
    for common_angle in COMMON_XPCS_ANGLES:
        for range_spec in target_ranges:
            min_angle = range_spec.get("min_angle", -np.inf)
            max_angle = range_spec.get("max_angle", np.inf)
            if min_angle <= common_angle <= max_angle:
                common_angles_matched = True
                break
        if common_angles_matched:
            break

    if not common_angles_matched:
        logger.warning(
            f"Configured angle ranges {target_ranges} do not overlap with "
            f"common XPCS angles {COMMON_XPCS_ANGLES}. Verify your configuration "
            f"is correct (check for typos in min_angle/max_angle values).",
        )

    # Create modified config with normalized target_ranges
    modified_config = config_dict.copy()
    modified_phi_filtering = phi_filtering_config.copy()
    modified_phi_filtering["target_ranges"] = target_ranges
    modified_config["phi_filtering"] = modified_phi_filtering

    # Call shared filtering function with performance timing
    start_time = time.perf_counter()
    filtered_indices, filtered_phi_angles, filtered_c2_exp = _apply_angle_filtering(
        phi_angles,
        c2_exp,
        modified_config,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    logger.debug(f"Angle filtering completed in {elapsed_ms:.3f}ms")

    # Check if any angles were filtered
    if not filtered_indices:
        logger.warning("No angles matched phi_filtering criteria, using all angles")
        # Return data with normalized angles
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Check if all angles matched (no actual filtering)
    if len(filtered_indices) == len(phi_angles):
        logger.debug(
            f"All {len(phi_angles)} angles matched filter criteria, no reduction",
        )
        # Return data with normalized angles
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Create filtered data dictionary
    filtered_data = {
        "phi_angles_list": filtered_phi_angles,
        "c2_exp": filtered_c2_exp,
        # Preserve other keys unchanged
        "wavevector_q_list": data.get("wavevector_q_list"),
        "t1": data.get("t1"),
        "t2": data.get("t2"),
    }

    # Copy any additional keys that might be present
    for key in data:
        if key not in filtered_data:
            filtered_data[key] = data[key]

    # Log filtering results
    logger.info(
        f"Angle filtering for optimization: {len(filtered_indices)} angles selected "
        f"from {len(phi_angles)} total angles",
    )
    logger.info(f"Selected angles: {filtered_phi_angles}")

    return filtered_data


def _apply_angle_filtering(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Core angle filtering logic shared by optimization and plotting.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_apply_angle_filtering(phi_angles, c2_exp, config)


def _prepare_cmc_config(
    args: argparse.Namespace, config: ConfigManager
) -> dict[str, Any]:
    """Prepare CMC configuration with CLI overrides applied.

    Extracts CMC config from ConfigManager and applies CLI argument overrides
    for num_shards and backend.

    Args:
        args: Parsed CLI arguments
        config: Configuration manager

    Returns:
        CMC configuration dictionary with CLI overrides applied
    """
    cmc_config = config.get_cmc_config()

    # Normalize backend config to dict form for mutation
    backend_cfg = cmc_config.get("backend", {})
    if isinstance(backend_cfg, str):
        backend_cfg = {"name": backend_cfg}
    elif backend_cfg is None:
        backend_cfg = {}
    cmc_config["backend"] = backend_cfg

    # Apply CLI overrides
    if args.cmc_num_shards is not None:
        logger.info(f"Overriding CMC num_shards from CLI: {args.cmc_num_shards}")
        cmc_config.setdefault("sharding", {})["num_shards"] = args.cmc_num_shards

    if args.cmc_backend is not None:
        logger.info(f"Overriding CMC backend from CLI: {args.cmc_backend}")
        backend_cfg["name"] = args.cmc_backend

    # Auto-select multiprocessing for multiple shards with auto/jax backend
    num_shards_override = cmc_config.get("sharding", {}).get("num_shards", "auto")
    try:
        num_shards_int = int(num_shards_override)
    except (TypeError, ValueError):
        num_shards_int = None

    backend_name_current = backend_cfg.get("name", "auto")
    if (
        num_shards_int
        and num_shards_int > 1
        and backend_name_current in ("auto", "jax")
    ):
        logger.warning(
            "Multiple shards requested but backend is 'auto/jax'; defaulting to multiprocessing. "
            "Use --cmc-backend to override explicitly."
        )
        backend_cfg["name"] = "multiprocessing"

    return cmc_config


def _pool_mcmc_data(filtered_data: dict[str, Any]) -> dict[str, Any]:
    """Pool and flatten 3D correlation data for MCMC.

    MCMC expects 1D pooled/flattened data with matching array lengths.
    This function transforms 3D c2_exp (n_phi, n_t, n_t) into 1D arrays
    and creates corresponding t1, t2, phi arrays of the same length.

    Args:
        filtered_data: Dictionary with c2_exp (3D), t1, t2, and phi_angles_list

    Returns:
        Dictionary with pooled data:
        - mcmc_data: flattened c2 data (n_phi * n_t * n_t,)
        - t1_pooled: tiled t1 values (n_phi * n_t * n_t,)
        - t2_pooled: tiled t2 values (n_phi * n_t * n_t,)
        - phi_pooled: repeated phi values (n_phi * n_t * n_t,)
        - n_phi, n_t, n_total: dimension info
    """
    c2_3d = filtered_data["c2_exp"]  # Shape: (n_phi, n_t, n_t)
    t1_raw = filtered_data.get("t1")
    t2_raw = filtered_data.get("t2")
    phi_angles = filtered_data.get("phi_angles_list")

    n_phi = c2_3d.shape[0]
    n_t = c2_3d.shape[1]
    n_total = n_phi * n_t * n_t

    # Flatten correlation data
    mcmc_data = c2_3d.ravel()

    # Handle both 1D and 2D time arrays
    if t1_raw is None or t2_raw is None:
        raise ValueError("Missing t1 or t2 arrays in filtered_data")

    t1_arr = np.asarray(t1_raw)
    t2_arr = np.asarray(t2_raw)

    if t1_arr.ndim == 1 and t2_arr.ndim == 1:
        t1_2d, t2_2d = np.meshgrid(t1_arr, t2_arr, indexing="ij")
        logger.debug(
            f"Created 2D meshgrids from 1D arrays: t1={t1_arr.shape} -> {t1_2d.shape}"
        )
    elif t1_arr.ndim == 2 and t2_arr.ndim == 2:
        t1_2d = t1_arr
        t2_2d = t2_arr
        logger.debug(f"Using existing 2D meshgrids: t1={t1_2d.shape}, t2={t2_2d.shape}")
    else:
        raise ValueError(
            f"Inconsistent t1/t2 dimensions: t1.ndim={t1_arr.ndim}, t2.ndim={t2_arr.ndim}. "
            f"Expected both 1D or both 2D."
        )

    # Tile time arrays for each phi angle
    t1_pooled = np.tile(t1_2d.ravel(), n_phi)
    t2_pooled = np.tile(t2_2d.ravel(), n_phi)

    if phi_angles is not None:
        phi_pooled = np.repeat(np.asarray(phi_angles), n_t * n_t)
    else:
        raise ValueError("Missing phi_angles_list in filtered_data")

    # Verify all arrays have matching lengths
    for name, arr in [
        ("mcmc_data", mcmc_data),
        ("t1", t1_pooled),
        ("t2", t2_pooled),
        ("phi", phi_pooled),
    ]:
        if arr.shape[0] != n_total:
            raise ValueError(
                f"Data pooling failed: {name}={arr.shape[0]}, expected={n_total}"
            )

    logger.debug(
        f"Pooled MCMC data: {n_phi} angles x {n_t}x{n_t} = {n_total:,} data points"
    )

    return {
        "mcmc_data": mcmc_data,
        "t1_pooled": t1_pooled,
        "t2_pooled": t2_pooled,
        "phi_pooled": phi_pooled,
        "n_phi": n_phi,
        "n_t": n_t,
        "n_total": n_total,
    }
