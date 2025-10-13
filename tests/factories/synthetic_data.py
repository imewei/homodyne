"""
Synthetic XPCS Data Generator for Scientific Validation (T034).

Generates synthetic XPCS correlation data with known ground-truth parameters
for testing parameter recovery accuracy.

Uses homodyne physics models to generate realistic data with controllable noise.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


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
    """
    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2: np.ndarray
    sigma: np.ndarray
    q: float
    L: float
    dt: float
    ground_truth_params: Dict[str, float]


def generate_synthetic_xpcs_data(
    ground_truth_params: Dict[str, float],
    n_phi: int = 10,
    n_t1: int = 20,
    n_t2: int = 20,
    noise_level: float = 0.05,
    q: float = 0.01,
    L: float = 1.0,
    dt: float = 0.1,
    analysis_mode: str = "static_isotropic",
    random_seed: Optional[int] = 42
) -> SyntheticXPCSData:
    """
    Generate synthetic XPCS data with known ground-truth parameters.

    Uses homodyne physics models to compute theoretical g2, then adds noise.

    Parameters
    ----------
    ground_truth_params : dict
        Ground truth parameter values. Should include:
        - For static isotropic: contrast, offset, D0, alpha, D_offset
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
        'static_isotropic' or 'laminar_flow'
    random_seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    SyntheticXPCSData
        Synthetic data with known ground truth

    Examples
    --------
    >>> params = {
    ...     'contrast': 0.5,
    ...     'offset': 1.0,
    ...     'D0': 1000.0,
    ...     'alpha': 0.5,
    ...     'D_offset': 10.0
    ... }
    >>> data = generate_synthetic_xpcs_data(params, n_phi=5, n_t1=10, n_t2=10)
    >>> assert data.g2.shape == (5, 10, 10)
    >>> assert 'D0' in data.ground_truth_params
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Import physics computation here to avoid circular imports
    from homodyne.core.jax_backend import compute_g2_scaled
    import jax.numpy as jnp

    # Create coordinate arrays
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    t1 = np.linspace(0, 1, n_t1)
    t2 = np.linspace(0, 1, n_t2)

    # Extract parameters based on analysis mode
    contrast = ground_truth_params['contrast']
    offset = ground_truth_params['offset']

    # Physical parameters (excluding scaling params)
    param_order = ['D0', 'alpha', 'D_offset']
    if 'laminar' in analysis_mode.lower():
        param_order.extend(['gamma_dot_t0', 'beta', 'gamma_dot_offset', 'phi0'])

    physical_params = jnp.array([ground_truth_params[name] for name in param_order])

    # Generate theoretical g2 for each phi angle
    g2_theoretical = []
    for phi_val in phi:
        g2_phi = compute_g2_scaled(
            params=physical_params,
            t1=jnp.array(t1),
            t2=jnp.array(t2),
            phi=float(phi_val),
            q=q,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt
        )
        g2_theoretical.append(np.array(g2_phi))

    g2_theoretical = np.stack(g2_theoretical, axis=0)

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
        ground_truth_params=ground_truth_params.copy()
    )

    return data


def generate_static_isotropic_dataset(
    D0: float = 1000.0,
    alpha: float = 0.5,
    D_offset: float = 10.0,
    contrast: float = 0.5,
    offset: float = 1.0,
    noise_level: float = 0.05,
    n_phi: int = 10,
    n_t1: int = 20,
    n_t2: int = 20,
    **kwargs
) -> SyntheticXPCSData:
    """
    Generate static isotropic synthetic dataset with default parameters.

    Parameters
    ----------
    D0 : float
        Diffusion coefficient prefactor
    alpha : float
        Anomalous diffusion exponent
    D_offset : float
        Diffusion offset
    contrast : float
        Contrast parameter
    offset : float
        Baseline offset
    noise_level : float
        Relative noise level
    n_phi, n_t1, n_t2 : int
        Grid dimensions
    **kwargs
        Additional arguments passed to generate_synthetic_xpcs_data

    Returns
    -------
    SyntheticXPCSData
        Static isotropic synthetic data
    """
    params = {
        'contrast': contrast,
        'offset': offset,
        'D0': D0,
        'alpha': alpha,
        'D_offset': D_offset
    }

    return generate_synthetic_xpcs_data(
        ground_truth_params=params,
        n_phi=n_phi,
        n_t1=n_t1,
        n_t2=n_t2,
        noise_level=noise_level,
        analysis_mode="static_isotropic",
        **kwargs
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
    **kwargs
) -> SyntheticXPCSData:
    """
    Generate laminar flow synthetic dataset with default parameters.

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
        Contrast parameter
    offset : float
        Baseline offset
    noise_level : float
        Relative noise level
    n_phi, n_t1, n_t2 : int
        Grid dimensions
    **kwargs
        Additional arguments passed to generate_synthetic_xpcs_data

    Returns
    -------
    SyntheticXPCSData
        Laminar flow synthetic data
    """
    params = {
        'contrast': contrast,
        'offset': offset,
        'D0': D0,
        'alpha': alpha,
        'D_offset': D_offset,
        'gamma_dot_t0': gamma_dot_t0,
        'beta': beta,
        'gamma_dot_offset': gamma_dot_offset,
        'phi0': phi0
    }

    return generate_synthetic_xpcs_data(
        ground_truth_params=params,
        n_phi=n_phi,
        n_t1=n_t1,
        n_t2=n_t2,
        noise_level=noise_level,
        analysis_mode="laminar_flow",
        **kwargs
    )
