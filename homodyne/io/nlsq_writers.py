"""NLSQ result saving functions for homodyne XPCS analysis.

This module provides functions for saving NLSQ optimization results to disk,
including JSON parameter files and NPZ data files.
Extracted from cli/commands.py for better modularity.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def save_nlsq_json_files(
    param_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
    convergence_dict: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save 3 JSON files: parameters, analysis results, convergence metrics.

    Parameters
    ----------
    param_dict : dict[str, Any]
        Parameter dictionary with {name: {value, uncertainty}}
    analysis_dict : dict[str, Any]
        Analysis results with method, fit_quality, dataset_info, etc.
    convergence_dict : dict[str, Any]
        Convergence diagnostics with status, iterations, recovery_actions
    output_dir : Path
        Output directory for JSON files

    Returns
    -------
    None
        Files saved to disk

    Notes
    -----
    Creates 3 JSON files:
    - parameters.json: Complete parameter values and uncertainties
    - analysis_results_nlsq.json: Analysis summary and fit quality
    - convergence_metrics.json: Convergence diagnostics and device info
    """
    # Save parameters.json
    param_file = output_dir / "parameters.json"
    with open(param_file, "w") as f:
        json.dump(param_dict, f, indent=2)
    # T056: Log file path and write completion
    logger.debug(f"Saved parameters to {param_file}")

    # Save analysis_results_nlsq.json
    analysis_file = output_dir / "analysis_results_nlsq.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis_dict, f, indent=2)
    logger.debug(f"Saved analysis results to {analysis_file}")

    # Save convergence_metrics.json
    convergence_file = output_dir / "convergence_metrics.json"
    with open(convergence_file, "w") as f:
        json.dump(convergence_dict, f, indent=2)
    logger.debug(f"Saved convergence metrics to {convergence_file}")

    # T058a: Log file sizes after write completion
    total_size_kb = (
        param_file.stat().st_size
        + analysis_file.stat().st_size
        + convergence_file.stat().st_size
    ) / 1024
    logger.info(
        f"Saved 3 JSON files to {output_dir} (total: {total_size_kb:.1f} KB)"
    )


def save_nlsq_npz_file(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_raw: np.ndarray,
    c2_scaled: np.ndarray,
    c2_solver: np.ndarray | None,
    per_angle_scaling: np.ndarray,
    per_angle_scaling_solver: np.ndarray,
    residuals: np.ndarray,
    residuals_norm: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    q: float,
    output_dir: Path,
) -> None:
    """Save NPZ file with experimental/theoretical data and metadata.

    Parameters
    ----------
    phi_angles : np.ndarray
        Scattering angles (n_angles,)
    c2_exp : np.ndarray
        Experimental correlation data (n_angles, n_t1, n_t2)
    c2_raw : np.ndarray
        Raw theoretical fits before scaling (n_angles, n_t1, n_t2)
    c2_scaled : np.ndarray
        Scaled theoretical fits (n_angles, n_t1, n_t2)
    c2_solver : np.ndarray | None
        Solver-evaluated theoretical fits (optional, n_angles, n_t1, n_t2)
    per_angle_scaling : np.ndarray
        Per-angle scaling parameters (n_angles, 2) [contrast, offset]
    per_angle_scaling_solver : np.ndarray
        Original per-angle scaling parameters from the solver (n_angles, 2)
    residuals : np.ndarray
        Residuals: exp - scaled (n_angles, n_t1, n_t2)
    residuals_norm : np.ndarray
        Normalized residuals (n_angles, n_t1, n_t2)
    t1 : np.ndarray
        Time array 1 (n_t1,)
    t2 : np.ndarray
        Time array 2 (n_t2,)
    q : float
        Wavevector magnitude [1/Å]
    output_dir : Path
        Output directory

    Returns
    -------
    None
        NPZ file saved to disk
    """
    npz_file = output_dir / "fitted_data.npz"

    np.savez_compressed(
        npz_file,
        # Experimental data (2 arrays)
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        # Theoretical fits (4 arrays)
        c2_theoretical_raw=c2_raw,
        c2_theoretical_scaled=c2_scaled,
        c2_solver_scaled=c2_solver,
        per_angle_scaling=per_angle_scaling,
        per_angle_scaling_solver=per_angle_scaling_solver,
        # Residuals (2 arrays)
        residuals=residuals,
        residuals_normalized=residuals_norm,
        # Coordinate arrays (3 arrays)
        t1=t1,
        t2=t2,
        q=np.array([q]),  # Wrap scalar in array
    )

    # T058a: Log file path and file size after write completion
    file_size_mb = npz_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved NPZ file with 10 arrays to {npz_file} ({file_size_mb:.2f} MB)")
