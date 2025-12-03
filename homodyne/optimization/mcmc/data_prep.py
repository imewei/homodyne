"""Data preparation utilities for NumPyro MCMC models.

This module extracts data preparation logic from _create_numpyro_model
to reduce cyclomatic complexity and improve maintainability.

Extracted from mcmc.py as part of technical debt remediation (Dec 2025).
"""

from __future__ import annotations

import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def get_target_dtype():
    """Get target dtype from environment variable.

    Returns
    -------
    dtype
        JAX dtype based on HOMODYNE_MCMC_DTYPE environment variable.
        Defaults to float64.

    Examples
    --------
    >>> import os
    >>> os.environ["HOMODYNE_MCMC_DTYPE"] = "float32"
    >>> dtype = get_target_dtype()
    >>> dtype == jnp.float32
    True
    """
    dtype_flag = os.environ.get("HOMODYNE_MCMC_DTYPE", "float64").lower()
    if dtype_flag in {"float32", "fp32", "32", "single"}:
        return jnp.float32
    elif dtype_flag in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    else:
        return jnp.float64


def normalize_array(
    value: Any,
    name: str,
    target_dtype: Any,
    allow_none: bool = False,
) -> jnp.ndarray | None:
    """Convert input to JAX array with proper dtype.

    Parameters
    ----------
    value : array-like or None
        Input value to normalize
    name : str
        Parameter name for error messages
    target_dtype : dtype
        Target JAX dtype
    allow_none : bool, default=False
        If True, allow None values to pass through

    Returns
    -------
    jnp.ndarray or None
        Normalized JAX array

    Raises
    ------
    ValueError
        If value is None and allow_none is False
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{name} cannot be None for MCMC model")
    return jnp.asarray(value, dtype=target_dtype)


def normalize_scalar(
    value: Any,
    target_dtype: Any,
    allow_none: bool = False,
) -> float | None:
    """Convert input to scalar float with proper dtype.

    Parameters
    ----------
    value : scalar or None
        Input value to normalize
    target_dtype : dtype
        Target JAX dtype for conversion
    allow_none : bool, default=False
        If True, allow None values to pass through

    Returns
    -------
    float or None
        Normalized scalar value

    Raises
    ------
    ValueError
        If value is None and allow_none is False
    """
    if value is None:
        if allow_none:
            return None
        raise ValueError("Scalar value required for MCMC model")
    return float(jnp.asarray(value, dtype=target_dtype))


def prepare_mcmc_arrays(
    data: Any,
    sigma: Any,
    t1: Any,
    t2: Any,
    phi: Any,
    q: Any,
    L: Any,
    dt: Any | None = None,
    phi_full: Any | None = None,
    target_dtype: Any | None = None,
) -> dict[str, Any]:
    """Prepare input arrays for MCMC model.

    Converts all input arrays to JAX arrays with consistent dtype.

    Parameters
    ----------
    data : array-like
        Experimental correlation data
    sigma : array-like
        Noise standard deviations
    t1, t2 : array-like
        Time delay arrays
    phi : array-like
        Azimuthal angle array
    q : float
        Wavevector magnitude
    L : float
        Sample-detector distance
    dt : float, optional
        Time step
    phi_full : array-like, optional
        Full replicated phi array
    target_dtype : dtype, optional
        Target dtype. If None, determined from environment.

    Returns
    -------
    dict
        Dictionary with normalized arrays:
        - data, sigma, t1, t2, phi, phi_full (JAX arrays)
        - q, L, dt (floats)
        - phi_unique_np (numpy array of unique phi values)
        - n_phi (int)
        - phi_array_for_mapping (JAX array for scaling)
        - phi_unique_for_sampling (JAX array)
        - target_dtype
    """
    if target_dtype is None:
        target_dtype = get_target_dtype()

    # Capture phi uniqueness using numpy before converting to JAX
    phi_unique_np = np.unique(np.asarray(phi))
    n_phi = len(phi_unique_np)

    # Normalize arrays
    data_arr = normalize_array(data, "data", target_dtype)
    sigma_arr = normalize_array(sigma, "sigma", target_dtype)
    t1_arr = normalize_array(t1, "t1", target_dtype)
    t2_arr = normalize_array(t2, "t2", target_dtype)
    phi_arr = normalize_array(phi, "phi", target_dtype)
    phi_full_arr = normalize_array(phi_full, "phi_full", target_dtype, allow_none=True)

    # Determine phi array for mapping
    phi_array_for_mapping = phi_full_arr if phi_full_arr is not None else phi_arr
    phi_unique_for_sampling = jnp.asarray(phi_unique_np, dtype=target_dtype)

    # Normalize scalars
    q_val = normalize_scalar(q, target_dtype)
    L_val = normalize_scalar(L, target_dtype)
    dt_val = normalize_scalar(dt, target_dtype, allow_none=True)

    return {
        "data": data_arr,
        "sigma": sigma_arr,
        "t1": t1_arr,
        "t2": t2_arr,
        "phi": phi_arr,
        "phi_full": phi_full_arr,
        "q": q_val,
        "L": L_val,
        "dt": dt_val,
        "phi_unique_np": phi_unique_np,
        "n_phi": n_phi,
        "phi_array_for_mapping": phi_array_for_mapping,
        "phi_unique_for_sampling": phi_unique_for_sampling,
        "target_dtype": target_dtype,
    }


def validate_array_shapes(
    data: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    sigma: jnp.ndarray,
    phi_array_for_mapping: jnp.ndarray | None = None,
    per_angle_scaling: bool = False,
) -> list[str]:
    """Validate that all arrays have compatible shapes.

    Parameters
    ----------
    data : jnp.ndarray
        Experimental data array
    t1, t2 : jnp.ndarray
        Time delay arrays
    sigma : jnp.ndarray
        Noise standard deviations
    phi_array_for_mapping : jnp.ndarray, optional
        Phi array for per-angle scaling
    per_angle_scaling : bool, default=False
        Whether per-angle scaling is enabled

    Returns
    -------
    list[str]
        List of warning messages for any mismatches
    """
    warnings = []

    data_size = data.shape[0] if hasattr(data, "shape") else len(data)
    t1_size = t1.shape[0] if hasattr(t1, "shape") else len(t1)
    t2_size = t2.shape[0] if hasattr(t2, "shape") else len(t2)
    sigma_size = sigma.shape[0] if hasattr(sigma, "shape") else len(sigma)

    if not (t1_size == t2_size == data_size == sigma_size):
        warnings.append(
            f"Array size mismatch: data={data_size}, t1={t1_size}, "
            f"t2={t2_size}, sigma={sigma_size}"
        )

    if per_angle_scaling and phi_array_for_mapping is not None:
        phi_mapping_size = len(phi_array_for_mapping)
        if phi_mapping_size != data_size:
            warnings.append(
                f"Per-angle scaling size mismatch: phi_array={phi_mapping_size}, "
                f"data={data_size}"
            )

    return warnings


def compute_phi_mapping(
    phi_array_for_mapping: jnp.ndarray,
    phi_unique_for_sampling: jnp.ndarray,
) -> jnp.ndarray:
    """Create mapping from data indices to phi angle indices.

    Uses nearest-neighbor matching to handle floating-point mismatches.

    Parameters
    ----------
    phi_array_for_mapping : jnp.ndarray
        Full phi array with replicated values matching data length
    phi_unique_for_sampling : jnp.ndarray
        Unique phi values

    Returns
    -------
    jnp.ndarray
        Array of phi indices for each data point

    Notes
    -----
    Uses argmin-based nearest-neighbor matching instead of searchsorted
    to handle floating-point mismatches correctly.
    """
    phi_array_jax = jnp.atleast_1d(phi_array_for_mapping)

    # Compute pairwise distances: shape (n_data, n_phi)
    distances = jnp.abs(
        phi_array_jax[:, None] - phi_unique_for_sampling[None, :]
    )

    # Find index of nearest phi for each data point
    phi_indices = jnp.argmin(distances, axis=1)

    return phi_indices


def log_dtype_info(
    target_dtype: Any,
    data: jnp.ndarray,
    sigma: jnp.ndarray,
    t1: jnp.ndarray,
    t2: jnp.ndarray,
    phi: jnp.ndarray,
) -> None:
    """Log dtype information for debugging.

    Parameters
    ----------
    target_dtype : dtype
        Target dtype
    data, sigma, t1, t2, phi : jnp.ndarray
        Input arrays
    """
    logger.info(
        "MCMC dtype normalization: target=%s data=%s sigma=%s t1=%s t2=%s phi=%s",
        target_dtype,
        data.dtype,
        sigma.dtype,
        t1.dtype,
        t2.dtype,
        phi.dtype,
    )
