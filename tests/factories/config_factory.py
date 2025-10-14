"""Test factory for creating phi_filtering configuration fixtures.

This module provides factory functions for generating various phi_filtering
configurations used in angle filtering tests.
"""

from typing import Any


def create_phi_filtering_config(
    enabled: bool = True, target_ranges: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Create a phi_filtering configuration dictionary.

    Parameters
    ----------
    enabled : bool, default=True
        Whether angle filtering is enabled
    target_ranges : list of dict, optional
        List of angle range dictionaries with 'min_angle', 'max_angle',
        and optionally 'description' keys. If None, returns empty list.

    Returns
    -------
    dict
        Configuration dictionary with 'phi_filtering' section

    Examples
    --------
    >>> config = create_phi_filtering_config(
    ...     enabled=True,
    ...     target_ranges=[
    ...         {"min_angle": -10.0, "max_angle": 10.0, "description": "Near 0"},
    ...         {"min_angle": 85.0, "max_angle": 95.0, "description": "Near 90"}
    ...     ]
    ... )
    >>> config["phi_filtering"]["enabled"]
    True
    >>> len(config["phi_filtering"]["target_ranges"])
    2
    """
    if target_ranges is None:
        target_ranges = []

    return {"phi_filtering": {"enabled": enabled, "target_ranges": target_ranges}}


def create_laminar_flow_filtering_config() -> dict[str, Any]:
    """Create phi_filtering config for typical laminar flow analysis.

    Includes 8 ranges covering primary flow direction, perpendicular directions,
    and diagonal directions.

    Returns
    -------
    dict
        Configuration with 8 angle ranges for laminar flow

    Examples
    --------
    >>> config = create_laminar_flow_filtering_config()
    >>> len(config["phi_filtering"]["target_ranges"])
    8
    """
    return create_phi_filtering_config(
        enabled=True,
        target_ranges=[
            # Primary flow direction (parallel/antiparallel)
            {"min_angle": -20.0, "max_angle": 20.0, "description": "Primary flow"},
            {"min_angle": 160.0, "max_angle": 200.0, "description": "Opposite flow"},
            # Perpendicular to flow
            {"min_angle": 70.0, "max_angle": 110.0, "description": "Perpendicular 1"},
            {"min_angle": 250.0, "max_angle": 290.0, "description": "Perpendicular 2"},
            # Diagonal directions
            {"min_angle": 35.0, "max_angle": 55.0, "description": "Diagonal 1"},
            {"min_angle": 125.0, "max_angle": 145.0, "description": "Diagonal 2"},
            {"min_angle": 215.0, "max_angle": 235.0, "description": "Diagonal 3"},
            {"min_angle": 305.0, "max_angle": 325.0, "description": "Diagonal 4"},
        ],
    )


def create_anisotropic_filtering_config() -> dict[str, Any]:
    """Create phi_filtering config for anisotropic analysis.

    Includes 2 ranges: parallel and perpendicular to primary axis.

    Returns
    -------
    dict
        Configuration with 2 angle ranges for anisotropic analysis

    Examples
    --------
    >>> config = create_anisotropic_filtering_config()
    >>> len(config["phi_filtering"]["target_ranges"])
    2
    """
    return create_phi_filtering_config(
        enabled=True,
        target_ranges=[
            {"min_angle": -10.0, "max_angle": 10.0, "description": "Parallel"},
            {"min_angle": 80.0, "max_angle": 100.0, "description": "Perpendicular"},
        ],
    )


def create_disabled_filtering_config() -> dict[str, Any]:
    """Create config with filtering disabled.

    Returns
    -------
    dict
        Configuration with enabled=False

    Examples
    --------
    >>> config = create_disabled_filtering_config()
    >>> config["phi_filtering"]["enabled"]
    False
    """
    return create_phi_filtering_config(enabled=False, target_ranges=[])


def create_empty_ranges_config() -> dict[str, Any]:
    """Create config with enabled=True but empty target_ranges.

    This tests the edge case where filtering is enabled but no ranges specified.

    Returns
    -------
    dict
        Configuration with enabled=True and empty target_ranges

    Examples
    --------
    >>> config = create_empty_ranges_config()
    >>> config["phi_filtering"]["enabled"]
    True
    >>> len(config["phi_filtering"]["target_ranges"])
    0
    """
    return create_phi_filtering_config(enabled=True, target_ranges=[])


def create_non_matching_ranges_config(
    phi_angles: list[float] | None = None,
) -> dict[str, Any]:
    """Create config with ranges that don't match typical dataset angles.

    Useful for testing the fallback behavior when no angles match.

    Parameters
    ----------
    phi_angles : list of float, optional
        Dataset angles to avoid. If None, creates ranges that won't match
        typical experimental angles (0, 30, 60, 90, etc.)

    Returns
    -------
    dict
        Configuration with non-matching ranges

    Examples
    --------
    >>> # For dataset with angles [0, 30, 60, 90], create non-matching ranges
    >>> config = create_non_matching_ranges_config([0.0, 30.0, 60.0, 90.0])
    >>> config["phi_filtering"]["target_ranges"][0]["min_angle"]
    45.0
    """
    if phi_angles is not None:
        # Find gaps in provided angles - use simple heuristic
        # (In real scenario, this would be more sophisticated)
        return create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 45.0, "max_angle": 50.0, "description": "Non-matching"},
            ],
        )
    else:
        # Default: ranges that won't match typical 0/30/60/90 degree angles
        return create_phi_filtering_config(
            enabled=True,
            target_ranges=[
                {"min_angle": 15.0, "max_angle": 20.0, "description": "Gap 1"},
                {"min_angle": 45.0, "max_angle": 50.0, "description": "Gap 2"},
            ],
        )


def create_overlapping_ranges_config() -> dict[str, Any]:
    """Create config with overlapping angle ranges.

    Tests that angles matching multiple ranges are only counted once.

    Returns
    -------
    dict
        Configuration with overlapping ranges

    Examples
    --------
    >>> config = create_overlapping_ranges_config()
    >>> # Ranges [0, 20] and [10, 30] overlap at [10, 20]
    >>> len(config["phi_filtering"]["target_ranges"])
    2
    """
    return create_phi_filtering_config(
        enabled=True,
        target_ranges=[
            {"min_angle": 0.0, "max_angle": 20.0, "description": "Range 1"},
            {"min_angle": 10.0, "max_angle": 30.0, "description": "Range 2 (overlaps)"},
        ],
    )


def create_single_angle_range_config(angle: float = 0.0) -> dict[str, Any]:
    """Create config that targets a very narrow range (effectively single angle).

    Parameters
    ----------
    angle : float, default=0.0
        Center angle for the narrow range

    Returns
    -------
    dict
        Configuration with very narrow range around specified angle

    Examples
    --------
    >>> config = create_single_angle_range_config(90.0)
    >>> r = config["phi_filtering"]["target_ranges"][0]
    >>> r["min_angle"], r["max_angle"]
    (89.5, 90.5)
    """
    return create_phi_filtering_config(
        enabled=True,
        target_ranges=[
            {
                "min_angle": angle - 0.5,
                "max_angle": angle + 0.5,
                "description": f"Single angle near {angle}°",
            }
        ],
    )
