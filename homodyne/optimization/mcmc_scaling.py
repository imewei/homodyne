"""Per-angle scaling utilities for NumPyro MCMC models.

This module extracts per-angle scaling logic from _create_numpyro_model
to reduce cyclomatic complexity and improve maintainability.

Extracted from mcmc.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def prepare_phi_mapping(
    phi_array: np.ndarray | None,
    *,
    data_size: int,
    n_phi: int,
    phi_unique_np: np.ndarray,
    target_dtype: Any,
) -> jnp.ndarray | None:
    """Ensure phi mapping array matches flattened data size.

    The NumPyro model needs a 1D array whose length matches the pooled
    data array so that each data point can be mapped back to its source
    phi angle when applying per-angle contrast/offset scaling. Some
    callers accidentally pass only the list of unique phi angles rather
    than the full replicated mapping. This helper auto-expands that list
    to the expected length when the relationship is unambiguous.

    Parameters
    ----------
    phi_array : np.ndarray | None
        Raw phi array provided by the caller (may already be replicated).
    data_size : int
        Flattened data length used by the likelihood.
    n_phi : int
        Number of unique phi angles present in the dataset.
    phi_unique_np : np.ndarray
        Array of unique phi values computed outside of JIT tracing.
    target_dtype : dtype
        Target dtype for JAX computations.

    Returns
    -------
    jnp.ndarray | None
        Phi mapping array with ``data_size`` elements when inference is
        possible, otherwise the original array (mismatch will be
        validated downstream).
    """
    if phi_array is None:
        return None

    phi_mapping = jnp.ravel(jnp.asarray(phi_array, dtype=target_dtype))
    phi_length = int(phi_mapping.size)

    # Nothing to do if either length is zero or already matches data size
    if data_size <= 0 or phi_length == data_size:
        return phi_mapping

    # Auto-expand when we only received the list of unique angles but can
    # infer how many data points belong to each angle deterministically.
    if (
        n_phi > 0
        and phi_length == n_phi
        and data_size % n_phi == 0
        and phi_unique_np.size == n_phi
    ):
        points_per_angle = data_size // n_phi
        expanded = np.repeat(phi_unique_np, points_per_angle)
        logger.warning(
            "phi array length (%d) does not match data size (%d). "
            "Auto-expanding by repeating each of the %d unique angles for "
            "%d points. Update caller to pass the flattened phi array.",
            phi_length,
            data_size,
            n_phi,
            points_per_angle,
        )
        return jnp.asarray(expanded, dtype=target_dtype)

    return phi_mapping


def compute_phi_indices(
    phi_array_for_mapping: jnp.ndarray,
    phi_unique_for_sampling: jnp.ndarray,
) -> jnp.ndarray:
    """Compute phi angle indices using nearest-neighbor matching.

    Creates a mapping from data point indices to phi angle indices.
    Uses argmin-based nearest-neighbor matching instead of searchsorted
    to handle floating-point mismatches correctly.

    Parameters
    ----------
    phi_array_for_mapping : jnp.ndarray
        Full phi array with replicated values matching data length.
    phi_unique_for_sampling : jnp.ndarray
        Unique phi values.

    Returns
    -------
    jnp.ndarray
        Array of phi indices for each data point (shape: n_data).

    Notes
    -----
    CRITICAL FIX (Nov 2025): Uses nearest-neighbor matching instead of
    searchsorted.

    PROBLEM with searchsorted:
    - searchsorted finds INSERTION POINTS to maintain sorted order, NOT
      nearest matches
    - Example: phi_unique=[-174.20, -163.56, -154.49], value=-163.57
      searchsorted returns 2 (insert after -163.56), but nearest match
      is index 1
    - This maps data points to WRONG phi angle rows in c2_theory

    SOLUTION: argmin-based nearest-neighbor matching
    - Handles floating-point mismatches (e.g., -154.48506165 vs -154.485)
    - Always finds the CLOSEST phi value, not insertion point
    - Memory: (n_data x n_phi x 8 bytes) approx 368 MB for 2M points x 23 angles
    """
    phi_array_jax = jnp.atleast_1d(phi_array_for_mapping)

    # Compute pairwise distances: shape (n_data, n_phi)
    # Broadcasting: phi_array[:,None] - phi_unique[None,:] creates distance matrix
    distances = jnp.abs(
        phi_array_jax[:, None] - phi_unique_for_sampling[None, :]
    )

    # Find index of nearest phi for each data point: shape (n_data,)
    phi_indices = jnp.argmin(distances, axis=1)

    return phi_indices


def extract_per_point_theory(
    c2_theory: jnp.ndarray,
    phi_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Extract theory values for each data point based on phi indices.

    Handles both 1D and 2D c2_theory arrays:
    - laminar_flow mode: c2_theory is 2D (n_phi, n_data) - angle-dependent
    - static mode: c2_theory is 1D (n_data,) - angle-independent

    Parameters
    ----------
    c2_theory : jnp.ndarray
        Theoretical c2 values. Shape is either (n_data,) for static mode
        or (n_phi, n_data) for laminar_flow mode.
    phi_indices : jnp.ndarray
        Array of phi indices for each data point (shape: n_data).

    Returns
    -------
    jnp.ndarray
        Theory values per data point (shape: n_data).

    Notes
    -----
    CRITICAL FIX (Nov 10, 2025): Handle both 1D and 2D c2_theory.
    For static mode, c2_theory is the same for all angles (diffusion only),
    so we don't need phi indexing - just use it directly.
    Only the contrast/offset scaling is per-angle.
    """
    n_data_points = phi_indices.shape[0]

    if c2_theory.ndim == 2:
        # laminar_flow mode: 2D theory (n_phi, n_data)
        # Extract the appropriate angle's row for each data point
        # Advanced indexing: c2_theory[phi_indices, range(len)] -> shape (n_data,)
        c2_theory_per_point = c2_theory[phi_indices, jnp.arange(n_data_points)]
    else:
        # static mode: 1D theory (n_data,)
        # Theory is angle-independent (same for all phi)
        # No indexing needed - use directly
        c2_theory_per_point = c2_theory

    return c2_theory_per_point


def apply_per_angle_scaling(
    c2_theory_per_point: jnp.ndarray,
    contrast_per_point: jnp.ndarray,
    offset_per_point: jnp.ndarray,
) -> jnp.ndarray:
    """Apply per-angle contrast and offset scaling to theory values.

    Computes: c2_fitted = offset + contrast * g1^2
    where g1^2 = c2_theory - 1 (since c2_theory = 1 + g1^2).

    Parameters
    ----------
    c2_theory_per_point : jnp.ndarray
        Theory values per data point (shape: n_data).
    contrast_per_point : jnp.ndarray
        Contrast values per data point (shape: n_data).
    offset_per_point : jnp.ndarray
        Offset values per data point (shape: n_data).

    Returns
    -------
    jnp.ndarray
        Scaled c2 fitted values (shape: n_data).

    Notes
    -----
    CRITICAL FIX: c2_theory = 1 + g1^2, so g1^2 = c2_theory - 1
    Correct physics: c2_fitted = offset + contrast * g1^2
                               = offset + contrast * (c2_theory - 1)

    WRONG (previous): c2_fitted = contrast * c2_theory + offset
                                = contrast * (1 + g1^2) + offset
                                = contrast + contrast*g1^2 + offset  <- Extra "contrast" term!
    """
    g1_squared = c2_theory_per_point - 1.0
    c2_fitted = offset_per_point + contrast_per_point * g1_squared
    return c2_fitted


def apply_global_scaling(
    c2_theory: jnp.ndarray,
    contrast: jnp.ndarray,
    offset: jnp.ndarray,
) -> jnp.ndarray:
    """Apply global (scalar) contrast and offset scaling.

    This is the legacy behavior where contrast and offset are shared
    across all angles.

    Parameters
    ----------
    c2_theory : jnp.ndarray
        Theory values (any shape).
    contrast : jnp.ndarray
        Scalar contrast value.
    offset : jnp.ndarray
        Scalar offset value.

    Returns
    -------
    jnp.ndarray
        Scaled c2 fitted values (same shape as c2_theory).
    """
    g1_squared = c2_theory - 1.0
    c2_fitted = offset + contrast * g1_squared
    return c2_fitted


def select_scaling_per_point(
    contrast: jnp.ndarray,
    offset: jnp.ndarray,
    phi_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Select per-point contrast and offset values from per-angle arrays.

    Parameters
    ----------
    contrast : jnp.ndarray
        Per-angle contrast values (shape: n_phi).
    offset : jnp.ndarray
        Per-angle offset values (shape: n_phi).
    phi_indices : jnp.ndarray
        Phi indices for each data point (shape: n_data).

    Returns
    -------
    tuple[jnp.ndarray, jnp.ndarray]
        (contrast_per_point, offset_per_point) each with shape (n_data,).
    """
    contrast_per_point = contrast[phi_indices]
    offset_per_point = offset[phi_indices]
    return contrast_per_point, offset_per_point


def validate_c2_fitted_shape(
    c2_fitted: jnp.ndarray,
    data: jnp.ndarray,
) -> None:
    """Validate that c2_fitted matches data shape.

    Parameters
    ----------
    c2_fitted : jnp.ndarray
        Fitted c2 values.
    data : jnp.ndarray
        Experimental data.

    Raises
    ------
    ValueError
        If shapes don't match.

    Notes
    -----
    CRITICAL VALIDATION: Ensures c2_fitted matches data shape before sampling.
    If shapes mismatch, NumPyro will fail with "invalid loc parameter" error.
    This catches bugs in phi_indices indexing or c2_theory shape issues.
    """
    expected_shape = data.shape
    if c2_fitted.shape != expected_shape:
        raise ValueError(
            f"Shape mismatch in MCMC model: c2_fitted.shape={c2_fitted.shape} "
            f"but data.shape={expected_shape}. This indicates a bug in "
            f"per-angle scaling indexing or c2_theory computation."
        )


def apply_scaling_to_theory(
    c2_theory: jnp.ndarray,
    contrast: jnp.ndarray,
    offset: jnp.ndarray,
    per_angle_scaling: bool,
    phi_mapping_for_scaling: jnp.ndarray | None,
    phi_unique_for_sampling: jnp.ndarray | None,
    data: jnp.ndarray,
) -> jnp.ndarray:
    """Apply appropriate scaling (per-angle or global) to theory values.

    This is the main entry point that orchestrates the scaling process.

    Parameters
    ----------
    c2_theory : jnp.ndarray
        Theoretical c2 values.
    contrast : jnp.ndarray
        Contrast value(s) - scalar or (n_phi,) array.
    offset : jnp.ndarray
        Offset value(s) - scalar or (n_phi,) array.
    per_angle_scaling : bool
        Whether to use per-angle scaling.
    phi_mapping_for_scaling : jnp.ndarray | None
        Phi values for each data point (shape: n_data).
    phi_unique_for_sampling : jnp.ndarray | None
        Unique phi values (shape: n_phi).
    data : jnp.ndarray
        Experimental data for shape validation.

    Returns
    -------
    jnp.ndarray
        Scaled c2 fitted values matching data shape.

    Raises
    ------
    ValueError
        If per_angle_scaling is True but phi arrays are None,
        or if final shape doesn't match data.
    """
    if per_angle_scaling:
        if phi_mapping_for_scaling is None:
            raise ValueError(
                "Per-angle scaling requires phi_mapping_for_scaling, but it is None"
            )
        if phi_unique_for_sampling is None:
            raise ValueError(
                "Per-angle scaling requires phi_unique_for_sampling, but it is None"
            )

        # Compute phi indices for each data point
        phi_indices = compute_phi_indices(
            phi_mapping_for_scaling,
            phi_unique_for_sampling,
        )

        # Select per-point contrast and offset
        contrast_per_point, offset_per_point = select_scaling_per_point(
            contrast, offset, phi_indices
        )

        # Extract theory values per point (handles 1D and 2D c2_theory)
        c2_theory_per_point = extract_per_point_theory(c2_theory, phi_indices)

        # Apply per-angle scaling
        c2_fitted = apply_per_angle_scaling(
            c2_theory_per_point,
            contrast_per_point,
            offset_per_point,
        )
    else:
        # Global scaling (legacy behavior)
        c2_fitted = apply_global_scaling(c2_theory, contrast, offset)

    # Validate shape
    validate_c2_fitted_shape(c2_fitted, data)

    return c2_fitted
