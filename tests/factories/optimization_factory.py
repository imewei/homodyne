"""
Factory for creating mock OptimizationResult objects for testing.

This module provides utilities for creating mock NLSQ optimization results
without running full optimizations, enabling efficient unit testing of
result saving and visualization functions.
"""

import numpy as np

from homodyne.optimization.nlsq_wrapper import OptimizationResult


def create_mock_optimization_result(
    analysis_mode: str = "static",
    converged: bool = True,
    include_uncertainties: bool = True,
    include_covariance: bool = True,
    quality_flag: str = "good",
    n_angles: int = 5,
) -> OptimizationResult:
    """
    Create mock OptimizationResult for testing.

    Parameters
    ----------
    analysis_mode : str
        "static" (per-angle + 3 physical params) or "laminar_flow" (per-angle + 7 physical params)
    converged : bool
        Whether optimization converged
    include_uncertainties : bool
        Whether to include uncertainty estimates
    include_covariance : bool
        Whether to include covariance matrix
    quality_flag : str
        "good", "marginal", or "poor"
    n_angles : int
        Number of angles (for per-angle scaling parameters).
        v2.4.0+ requires per-angle scaling, so parameter count = 2*n_angles + n_physical

    Returns
    -------
    OptimizationResult
        Mock result object with realistic parameter values

    Examples
    --------
    >>> result = create_mock_optimization_result("static", n_angles=5)
    >>> len(result.parameters)  # 5 contrasts + 5 offsets + 3 physical = 13
    13
    >>> result.convergence_status
    'converged'

    Notes
    -----
    v2.4.0 mandates per-angle scaling. Parameter structure:
    [contrast_0, ..., contrast_N, offset_0, ..., offset_N, physical_params...]
    """
    # Per-angle scaling parameters (contrast, offset for each angle)
    # Realistic values with slight variation per angle
    contrast_per_angle = 0.45 + 0.01 * np.random.randn(n_angles)
    offset_per_angle = 1.02 + 0.005 * np.random.randn(n_angles)

    if analysis_mode == "static":
        # 3 physical parameters: D0, alpha, D_offset
        physical_params = np.array([1234.5, 0.567, 12.34])
        physical_uncertainties = np.array([45.6, 0.012, 1.23])
        n_physical = 3
    elif analysis_mode == "laminar_flow":
        # 7 physical parameters: D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
        physical_params = np.array([1234.5, 0.567, 12.34, 1.23e-4, 0.456, 5.6e-6, 0.123])
        physical_uncertainties = np.array([45.6, 0.012, 1.23, 1.2e-5, 0.023, 1.2e-6, 0.023])
        n_physical = 7
    else:
        raise ValueError(
            f"Unknown analysis_mode: {analysis_mode}. Expected 'static' or 'laminar_flow'"
        )

    # Build per-angle scaling parameter array: [contrasts..., offsets..., physical...]
    parameters = np.concatenate([contrast_per_angle, offset_per_angle, physical_params])
    n_params = len(parameters)  # 2*n_angles + n_physical

    if include_uncertainties:
        contrast_uncertainties = np.full(n_angles, 0.012)
        offset_uncertainties = np.full(n_angles, 0.008)
        uncertainties = np.concatenate([
            contrast_uncertainties, offset_uncertainties, physical_uncertainties
        ])
    else:
        uncertainties = None

    # Covariance matrix (identity scaled by variance for simplicity)
    if include_covariance:
        covariance = np.eye(n_params) * 0.01
    else:
        covariance = None

    # Convergence status
    convergence_status = "converged" if converged else "max_iter"
    iterations = 42 if converged else 100

    # Recovery actions
    recovery_actions = (
        []
        if converged
        else [
            "Attempt 1: Original parameters - Failed (bounds violation)",
            "Attempt 2: Perturbed parameters 10% - Converged",
        ]
    )

    return OptimizationResult(
        parameters=parameters,
        uncertainties=uncertainties,
        covariance=covariance,
        chi_squared=1234.5678,
        reduced_chi_squared=1.234,
        convergence_status=convergence_status,
        iterations=iterations,
        execution_time=3.456,
        device_info={"device_type": "cpu", "device_name": "Intel Xeon E5-2680 v4"},
        recovery_actions=recovery_actions,
        quality_flag=quality_flag,
    )


def create_mock_config_manager(
    analysis_mode: str = "static", include_all_metadata: bool = True
) -> dict:
    """
    Create mock ConfigManager dict for testing metadata extraction.

    Parameters
    ----------
    analysis_mode : str
        "static" or "laminar_flow"
    include_all_metadata : bool
        If True, include all metadata fields. If False, omit some for
        testing fallback behavior.

    Returns
    -------
    dict
        Mock configuration dictionary

    Examples
    --------
    >>> config = create_mock_config_manager("laminar_flow")
    >>> config["analysis_mode"]
    'laminar_flow'
    >>> config["analyzer_parameters"]["geometry"]["stator_rotor_gap"]
    2000000.0
    """
    config = {
        "analysis_mode": analysis_mode,
        "analyzer_parameters": {},
        "experimental_data": {},
        # Disable phi_filtering by default to avoid Mock iteration errors
        "phi_filtering": {"enabled": False, "target_ranges": []},
    }

    if include_all_metadata:
        config["analyzer_parameters"]["geometry"] = {"stator_rotor_gap": 2000000.0}
        config["analyzer_parameters"]["dt"] = 0.1
        config["experimental_data"]["sample_detector_distance"] = 5000000.0
        config["experimental_data"]["dt"] = 0.1
    else:
        # Minimal config for testing fallback behavior
        config["experimental_data"]["sample_detector_distance"] = 5000000.0

    return config


def create_mock_data_dict(
    n_angles: int = 10, n_t1: int = 25, n_t2: int = 25, include_sigma: bool = False
) -> dict:
    """
    Create mock experimental data dictionary for testing.

    Parameters
    ----------
    n_angles : int
        Number of scattering angles
    n_t1 : int
        Number of t1 time points
    n_t2 : int
        Number of t2 time points
    include_sigma : bool
        Whether to include uncertainty (sigma) data

    Returns
    -------
    dict
        Mock data dictionary with all required arrays

    Examples
    --------
    >>> data = create_mock_data_dict(n_angles=5, n_t1=10, n_t2=10)
    >>> data["c2_exp"].shape
    (5, 10, 10)
    >>> len(data["phi_angles_list"])
    5
    """
    # Generate synthetic data
    phi_angles = np.linspace(0, 180, n_angles)
    t1 = np.linspace(0.01, 1.0, n_t1)
    t2 = np.linspace(0.01, 1.0, n_t2)

    # Generate realistic correlation function (exponential decay)
    t1_grid, t2_grid = np.meshgrid(t1, t2, indexing="ij")
    c2_base = 1.0 + 0.5 * np.exp(-0.5 * (t1_grid + t2_grid))

    # Replicate for all angles with slight variation
    c2_exp = np.array(
        [c2_base * (1.0 + 0.1 * np.sin(phi * np.pi / 180)) for phi in phi_angles]
    )

    data = {
        "phi_angles_list": phi_angles,
        "c2_exp": c2_exp,
        "t1": t1,
        "t2": t2,
        "wavevector_q_list": np.array([0.0123]),  # Single q-value
    }

    if include_sigma:
        # Generate uncertainty data (5% of signal)
        data["sigma"] = 0.05 * c2_exp

    return data
