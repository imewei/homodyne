"""
Synthetic XPCS Data Generator for Scientific Validation (T034).

Generates synthetic XPCS correlation data with known ground-truth parameters
for testing parameter recovery accuracy.

Uses homodyne physics models to generate realistic data with controllable noise.
"""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticXPCSData:
    """
    Container for synthetic XPCS data with known ground truth.

    Attributes:
        phi: Angle array
        t1: First delay time array
        t2: Second delay time array
        g2: Synthetic correlation data (n_phi, n_t1, n_t2)
        sigma: Uncertainty array (same shape as g2)
        q: Wave vector magnitude
        L: Sample-detector distance
        dt: Time step
        ground_truth_params: True parameter values used to generate data
        per_angle_scaling: Whether to use per-angle scaling parameters (default: True as of v2.4.0)
    """

    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2: np.ndarray
    sigma: np.ndarray
    q: float
    L: float
    dt: float
    ground_truth_params: dict[str, float]
    per_angle_scaling: bool = True  # v2.4.0: per-angle scaling is mandatory


def get_per_angle_param_names(n_angles: int, mode: str = "static") -> list[str]:
    """
    Get parameter names in correct NumPyro order for per-angle scaling.

    NumPyro requires parameters to be returned in exact order:
    1. Per-angle contrast parameters (contrast_0, contrast_1, ...)
    2. Per-angle offset parameters (offset_0, offset_1, ...)
    3. Physical parameters (D0, alpha, D_offset, ...)

    Parameters
    ----------
    n_angles : int
        Number of phi angles
    mode : str
        Analysis mode: 'static' or 'laminar_flow'

    Returns
    -------
    list[str]
        Parameter names in NumPyro-compatible order

    Examples
    --------
    >>> params = get_per_angle_param_names(3, mode='static')
    >>> assert params == ['contrast_0', 'contrast_1', 'contrast_2',
    ...                    'offset_0', 'offset_1', 'offset_2',
    ...                    'D0', 'alpha', 'D_offset']

    >>> params = get_per_angle_param_names(2, mode='laminar_flow')
    >>> assert params[-1] == 'phi0'
    """
    param_names = []

    # Per-angle contrast parameters first
    param_names.extend([f"contrast_{i}" for i in range(n_angles)])

    # Per-angle offset parameters second
    param_names.extend([f"offset_{i}" for i in range(n_angles)])

    # Physical parameters last
    param_names.extend(["D0", "alpha", "D_offset"])

    # Add laminar flow parameters if needed
    if "laminar" in mode.lower():
        param_names.extend(["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"])

    return param_names


def generate_synthetic_xpcs_data(
    ground_truth_params: dict[str, float],
    n_phi: int = 10,
    n_t1: int = 20,
    n_t2: int = 20,
    noise_level: float = 0.05,
    q: float = 0.01,
    L: float = 1.0,
    dt: float = 0.1,
    analysis_mode: str = "static",
    per_angle_scaling: bool = True,
    random_seed: int | None = 42,
) -> SyntheticXPCSData:
    """
    Generate synthetic XPCS data with known ground-truth parameters.

    Uses homodyne physics models to compute theoretical g2, then adds noise.

    Parameters
    ----------
    ground_truth_params : dict
        Ground truth parameter values. Should include:
        - For static mode: [contrast_0...N, offset_0...N, D0, alpha, D_offset] (per-angle)
          OR [contrast, offset, D0, alpha, D_offset] (legacy, deprecated)
        - For laminar flow: add gamma_dot_t0, beta, gamma_dot_offset, phi0
    n_phi : int
        Number of phi angles
    n_t1 : int
        Number of t1 time points
    n_t2 : int
        Number of t2 time points
    noise_level : float
        Relative noise level (sigma as fraction of signal)
    q : float
        Wave vector magnitude
    L : float
        Sample-detector distance
    dt : float
        Time step
    analysis_mode : str
        'static' or 'laminar_flow'
    per_angle_scaling : bool
        Use per-angle scaling for contrast and offset (default: True as of v2.4.0)
        If False, raises DeprecationWarning (will be removed in v2.5.0)
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    SyntheticXPCSData
        Synthetic data with known ground truth

    Raises
    ------
    DeprecationWarning
        If per_angle_scaling=False is explicitly passed

    Examples
    --------
    >>> params = {
    ...     'contrast_0': 0.5, 'contrast_1': 0.5, 'contrast_2': 0.5,
    ...     'offset_0': 1.0, 'offset_1': 1.0, 'offset_2': 1.0,
    ...     'D0': 1000.0,
    ...     'alpha': 0.5,
    ...     'D_offset': 10.0
    ... }
    >>> data = generate_synthetic_xpcs_data(params, n_phi=3, n_t1=10, n_t2=10, per_angle_scaling=True)
    >>> assert data.g2.shape == (3, 10, 10)
    >>> assert 'D0' in data.ground_truth_params
    """
    # Handle deprecation of per_angle_scaling=False
    if per_angle_scaling is False:
        warnings.warn(
            "per_angle_scaling=False is deprecated as of v2.4.0 and will be removed in v2.5.0. "
            "Per-angle scaling is now mandatory. Please update your code to use per_angle_scaling=True "
            "or use per-angle parameter format (contrast_0, contrast_1, ..., offset_0, offset_1, ...)",
            DeprecationWarning,
            stacklevel=2,
        )

    if random_seed is not None:
        np.random.seed(random_seed)

    # Import physics computation here to avoid circular imports
    import jax.numpy as jnp

    from homodyne.core.jax_backend import compute_g2_scaled

    # Create coordinate arrays
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    t1 = np.linspace(0, 1, n_t1)
    t2 = np.linspace(0, 1, n_t2)

    # Extract per-angle parameters (with fallback for scalar values)
    contrasts = []
    offsets = []
    for i in range(n_phi):
        # Try per-angle format first, fall back to scalar
        if f"contrast_{i}" in ground_truth_params:
            contrasts.append(ground_truth_params[f"contrast_{i}"])
        elif "contrast" in ground_truth_params:
            contrasts.append(ground_truth_params["contrast"])
        else:
            contrasts.append(0.5)  # default value

        if f"offset_{i}" in ground_truth_params:
            offsets.append(ground_truth_params[f"offset_{i}"])
        elif "offset" in ground_truth_params:
            offsets.append(ground_truth_params["offset"])
        else:
            offsets.append(1.0)  # default value

    # Physical parameters (excluding scaling params)
    param_order = ["D0", "alpha", "D_offset"]
    if "laminar" in analysis_mode.lower():
        param_order.extend(["gamma_dot_t0", "beta", "gamma_dot_offset", "phi0"])

    physical_params = jnp.array([ground_truth_params[name] for name in param_order])

    # Generate theoretical g2 for each phi angle
    g2_theoretical = []
    for i, phi_val in enumerate(phi):
        g2_phi = compute_g2_scaled(
            params=physical_params,
            t1=jnp.array(t1),
            t2=jnp.array(t2),
            phi=float(phi_val),
            q=q,
            L=L,
            contrast=contrasts[i],
            offset=offsets[i],
            dt=dt,
        )
        g2_theoretical.append(np.array(g2_phi))

    g2_theoretical = np.stack(g2_theoretical, axis=0)
    # Squeeze out singleton dimensions from compute_g2_scaled (e.g., (5, 1, 15, 15) -> (5, 15, 15))
    g2_theoretical = np.squeeze(g2_theoretical)

    # Add noise
    sigma = np.abs(g2_theoretical) * noise_level + 1e-6  # Avoid zero sigma
    noise = np.random.normal(0, 1, g2_theoretical.shape) * sigma
    g2_noisy = g2_theoretical + noise

    # Create data object
    data = SyntheticXPCSData(
        phi=phi,
        t1=t1,
        t2=t2,
        g2=g2_noisy,
        sigma=sigma,
        q=q,
        L=L,
        dt=dt,
        ground_truth_params=ground_truth_params.copy(),
        per_angle_scaling=per_angle_scaling,
    )

    return data


def generate_static_mode_dataset(
    D0: float = 1000.0,
    alpha: float = 0.5,
    D_offset: float = 10.0,
    contrast: float = 0.5,
    offset: float = 1.0,
    noise_level: float = 0.05,
    n_phi: int = 10,
    n_t1: int = 20,
    n_t2: int = 20,
    per_angle_scaling: bool = True,
    **kwargs,
) -> SyntheticXPCSData:
    """
    Generate static mode synthetic dataset with default parameters.

    Generates per-angle scaling parameters by default (v2.4.0+).

    Parameters
    ----------
    D0 : float
        Diffusion coefficient prefactor
    alpha : float
        Anomalous diffusion exponent
    D_offset : float
        Diffusion offset
    contrast : float
        Baseline contrast parameter (will be replicated per angle if per_angle_scaling=True)
    offset : float
        Baseline offset (will be replicated per angle if per_angle_scaling=True)
    noise_level : float
        Relative noise level
    n_phi, n_t1, n_t2 : int
        Grid dimensions
    per_angle_scaling : bool
        Use per-angle scaling for contrast and offset (default: True as of v2.4.0)
    **kwargs
        Additional arguments passed to generate_synthetic_xpcs_data

    Returns
    -------
    SyntheticXPCSData
        Static isotropic synthetic data with per-angle scaling parameters
    """
    # Build per-angle parameters
    params = {
        "D0": D0,
        "alpha": alpha,
        "D_offset": D_offset,
    }

    # Add per-angle contrast and offset parameters
    for i in range(n_phi):
        params[f"contrast_{i}"] = contrast
        params[f"offset_{i}"] = offset

    return generate_synthetic_xpcs_data(
        ground_truth_params=params,
        n_phi=n_phi,
        n_t1=n_t1,
        n_t2=n_t2,
        noise_level=noise_level,
        analysis_mode="static",
        per_angle_scaling=per_angle_scaling,
        **kwargs,
    )


def generate_laminar_flow_dataset(
    D0: float = 1000.0,
    alpha: float = 0.5,
    D_offset: float = 10.0,
    gamma_dot_t0: float = 1e-4,
    beta: float = 0.5,
    gamma_dot_offset: float = 1e-5,
    phi0: float = 0.0,
    contrast: float = 0.5,
    offset: float = 1.0,
    noise_level: float = 0.05,
    n_phi: int = 15,
    n_t1: int = 20,
    n_t2: int = 20,
    per_angle_scaling: bool = True,
    **kwargs,
) -> SyntheticXPCSData:
    """
    Generate laminar flow synthetic dataset with default parameters.

    Generates per-angle scaling parameters by default (v2.4.0+).

    Parameters
    ----------
    D0 : float
        Diffusion coefficient prefactor
    alpha : float
        Anomalous diffusion exponent
    D_offset : float
        Diffusion offset
    gamma_dot_t0 : float
        Shear rate at t=0
    beta : float
        Shear rate time exponent
    gamma_dot_offset : float
        Shear rate offset
    phi0 : float
        Reference angle for shear
    contrast : float
        Baseline contrast parameter (will be replicated per angle if per_angle_scaling=True)
    offset : float
        Baseline offset (will be replicated per angle if per_angle_scaling=True)
    noise_level : float
        Relative noise level
    n_phi, n_t1, n_t2 : int
        Grid dimensions
    per_angle_scaling : bool
        Use per-angle scaling for contrast and offset (default: True as of v2.4.0)
    **kwargs
        Additional arguments passed to generate_synthetic_xpcs_data

    Returns
    -------
    SyntheticXPCSData
        Laminar flow synthetic data with per-angle scaling parameters
    """
    # Build per-angle parameters
    params = {
        "D0": D0,
        "alpha": alpha,
        "D_offset": D_offset,
        "gamma_dot_t0": gamma_dot_t0,
        "beta": beta,
        "gamma_dot_offset": gamma_dot_offset,
        "phi0": phi0,
    }

    # Add per-angle contrast and offset parameters
    for i in range(n_phi):
        params[f"contrast_{i}"] = contrast
        params[f"offset_{i}"] = offset

    return generate_synthetic_xpcs_data(
        ground_truth_params=params,
        n_phi=n_phi,
        n_t1=n_t1,
        n_t2=n_t2,
        noise_level=noise_level,
        analysis_mode="laminar_flow",
        per_angle_scaling=per_angle_scaling,
        **kwargs,
    )
