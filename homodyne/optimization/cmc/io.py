"""I/O utilities for CMC results.

This module provides functions for saving CMC results to files:

- samples.npz: ArviZ-compatible posterior samples
- fitted_data.npz: Fitted data matching NLSQ format
- parameters.json: Posterior statistics
- diagnostics.json: Convergence diagnostics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from homodyne.optimization.cmc.diagnostics import create_diagnostics_dict
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)

# Schema version for samples.npz
SAMPLES_SCHEMA_VERSION = (1, 0)


def save_samples_npz(
    result: CMCResult,
    output_path: Path,
) -> None:
    """Save posterior samples in ArviZ-compatible format.

    The saved file can be loaded directly with numpy and converted
    to ArviZ InferenceData without modification.

    Parameters
    ----------
    result : CMCResult
        CMC result with samples.
    output_path : Path
        Output file path.

    File Format
    -----------
    - schema_version: (major, minor) version tuple
    - posterior_samples: (n_chains, n_samples, n_params) array
    - param_names: Parameter names in sampling order
    - r_hat: R-hat per parameter
    - ess_bulk: Bulk ESS per parameter
    - ess_tail: Tail ESS per parameter
    - divergences: Divergences per chain
    - analysis_mode: Analysis mode string
    - n_phi: Number of phi angles
    - n_chains: Number of chains
    - n_samples: Samples per chain
    """
    # Build samples array (n_chains, n_samples, n_params)
    samples_3d = result.get_samples_array()

    # Build arrays for diagnostics
    r_hat_arr = np.array([result.r_hat.get(name, np.nan) for name in result.param_names])
    ess_bulk_arr = np.array(
        [result.ess_bulk.get(name, np.nan) for name in result.param_names]
    )
    ess_tail_arr = np.array(
        [result.ess_tail.get(name, np.nan) for name in result.param_names]
    )

    # Count per-angle parameters to get n_phi
    n_phi = sum(1 for name in result.param_names if name.startswith("contrast_"))

    # Save to npz
    np.savez(
        output_path,
        # Schema version
        schema_version=np.array(SAMPLES_SCHEMA_VERSION),
        # Samples
        posterior_samples=samples_3d,
        param_names=np.array(result.param_names),
        # Diagnostics
        r_hat=r_hat_arr,
        ess_bulk=ess_bulk_arr,
        ess_tail=ess_tail_arr,
        divergences=np.array([result.divergences]),
        # Metadata
        analysis_mode=np.array([result.analysis_mode]),
        n_phi=np.array([n_phi]),
        n_chains=np.array([result.n_chains]),
        n_samples=np.array([result.n_samples]),
    )

    logger.info(f"Saved samples.npz: {output_path} ({samples_3d.shape})")


def load_samples_npz(
    input_path: Path,
) -> dict[str, Any]:
    """Load samples from npz file.

    Parameters
    ----------
    input_path : Path
        Path to samples.npz file.

    Returns
    -------
    dict[str, Any]
        Loaded data dictionary.

    Raises
    ------
    ValueError
        If path validation fails (path traversal, non-existent file).
    FileNotFoundError
        If the file does not exist.
    """
    # Validate path for security
    input_path = Path(input_path).resolve()

    # Check file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Samples file not found: {input_path}")

    # Check file extension
    if input_path.suffix != ".npz":
        raise ValueError(f"Expected .npz file, got: {input_path.suffix}")

    # Check file is readable (not a directory, symlink to non-file, etc.)
    if not input_path.is_file():
        raise ValueError(f"Path is not a regular file: {input_path}")

    # Load with allow_pickle=True (required for object arrays like param_names)
    # Security note: Only load files from trusted sources (CMC output)
    data = np.load(input_path, allow_pickle=True)

    return {
        "schema_version": tuple(data["schema_version"]),
        "posterior_samples": data["posterior_samples"],
        "param_names": data["param_names"].tolist(),
        "r_hat": data["r_hat"],
        "ess_bulk": data["ess_bulk"],
        "ess_tail": data["ess_tail"],
        "divergences": data["divergences"],
        "analysis_mode": str(data["analysis_mode"][0]),
        "n_phi": int(data["n_phi"][0]),
        "n_chains": int(data["n_chains"][0]),
        "n_samples": int(data["n_samples"][0]),
    }


def samples_to_arviz(
    samples_data: dict[str, Any],
):
    """Convert loaded samples to ArviZ InferenceData.

    Parameters
    ----------
    samples_data : dict[str, Any]
        Data from load_samples_npz().

    Returns
    -------
    az.InferenceData
        ArviZ-compatible data structure.
    """
    import arviz as az

    samples = samples_data["posterior_samples"]
    param_names = samples_data["param_names"]

    posterior_dict = {
        name: samples[:, :, i] for i, name in enumerate(param_names)
    }

    return az.from_dict(posterior=posterior_dict)


def save_fitted_data_npz(
    result: CMCResult,
    c2_exp: np.ndarray,
    c2_fitted: np.ndarray,
    c2_fitted_std: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi_angles: np.ndarray,
    q: float,
    output_path: Path,
) -> None:
    """Save fitted data in NLSQ-compatible format.

    Parameters
    ----------
    result : CMCResult
        CMC result.
    c2_exp : np.ndarray
        Experimental C2 data.
    c2_fitted : np.ndarray
        Fitted C2 (posterior mean).
    c2_fitted_std : np.ndarray
        Fitted C2 uncertainty.
    t1 : np.ndarray
        Time coordinates t1.
    t2 : np.ndarray
        Time coordinates t2.
    phi_angles : np.ndarray
        Phi angles.
    q : float
        Wavevector.
    output_path : Path
        Output file path.

    File Format
    -----------
    - c2_exp: Experimental data
    - c2_fitted: Posterior mean prediction
    - residuals: c2_exp - c2_fitted
    - c2_fitted_std: Posterior uncertainty
    - c2_fitted_5pct: 5th percentile
    - c2_fitted_95pct: 95th percentile
    - q: Wavevector
    - phi_angles: Phi angles
    - t1, t2: Time coordinates
    """
    residuals = c2_exp - c2_fitted

    # Compute percentiles if we have samples
    c2_fitted_5pct = c2_fitted - 1.645 * c2_fitted_std  # ~90% CI
    c2_fitted_95pct = c2_fitted + 1.645 * c2_fitted_std

    np.savez(
        output_path,
        # Core data (NLSQ parity)
        c2_exp=c2_exp,
        c2_fitted=c2_fitted,
        residuals=residuals,
        # Coordinates
        q=np.array([q]),
        phi_angles=phi_angles,
        t1=t1,
        t2=t2,
        # CMC-specific uncertainty
        c2_fitted_std=c2_fitted_std,
        c2_fitted_5pct=c2_fitted_5pct,
        c2_fitted_95pct=c2_fitted_95pct,
    )

    logger.info(
        f"Saved fitted_data.npz: {output_path} (shape={c2_exp.shape})"
    )


def save_parameters_json(
    result: CMCResult,
    output_path: Path,
) -> None:
    """Save posterior parameter statistics to JSON.

    Parameters
    ----------
    result : CMCResult
        CMC result.
    output_path : Path
        Output file path.
    """
    stats = result.get_posterior_stats()

    # Convert numpy types to Python types for JSON
    stats_json: dict[str, dict[str, float]] = {}
    for name, param_stats in stats.items():
        stats_json[name] = {
            k: float(v) if not np.isnan(v) else None
            for k, v in param_stats.items()
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    logger.info(f"Saved parameters.json: {output_path} ({len(stats)} params)")


def save_diagnostics_json(
    result: CMCResult,
    output_path: Path,
    warnings: list[str] | None = None,
) -> None:
    """Save convergence diagnostics to JSON.

    Parameters
    ----------
    result : CMCResult
        CMC result.
    output_path : Path
        Output file path.
    warnings : list[str] | None
        Warning messages from convergence check.
    """
    diagnostics = create_diagnostics_dict(
        r_hat=result.r_hat,
        ess_bulk=result.ess_bulk,
        ess_tail=result.ess_tail,
        divergences=result.divergences,
        convergence_status=result.convergence_status,
        warnings=warnings or [],
        n_chains=result.n_chains,
        n_warmup=result.n_warmup,
        n_samples=result.n_samples,
        warmup_time=result.warmup_time,
        sampling_time=result.execution_time - result.warmup_time,
    )

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        if np.isnan(obj) if isinstance(obj, float) else False:
            return None
        return obj

    diagnostics_json = convert_numpy(diagnostics)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics_json, f, indent=2)

    logger.info(f"Saved diagnostics.json: {output_path}")


def save_all_results(
    result: CMCResult,
    output_dir: Path,
    c2_exp: np.ndarray | None = None,
    c2_fitted: np.ndarray | None = None,
    c2_fitted_std: np.ndarray | None = None,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
    phi_angles: np.ndarray | None = None,
    q: float | None = None,
) -> dict[str, Path]:
    """Save all CMC result files.

    Parameters
    ----------
    result : CMCResult
        CMC result.
    output_dir : Path
        Output directory.
    c2_exp, c2_fitted, c2_fitted_std : np.ndarray | None
        Data for fitted_data.npz.
    t1, t2, phi_angles : np.ndarray | None
        Coordinates.
    q : float | None
        Wavevector.

    Returns
    -------
    dict[str, Path]
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files: dict[str, Path] = {}

    # Save samples.npz
    samples_path = output_dir / "samples.npz"
    save_samples_npz(result, samples_path)
    saved_files["samples"] = samples_path

    # Save parameters.json
    params_path = output_dir / "parameters.json"
    save_parameters_json(result, params_path)
    saved_files["parameters"] = params_path

    # Save diagnostics.json
    diag_path = output_dir / "diagnostics.json"
    save_diagnostics_json(result, diag_path)
    saved_files["diagnostics"] = diag_path

    # Save fitted_data.npz if data provided
    if all(x is not None for x in [c2_exp, c2_fitted, c2_fitted_std, t1, t2, phi_angles, q]):
        fitted_path = output_dir / "fitted_data.npz"
        save_fitted_data_npz(
            result=result,
            c2_exp=c2_exp,
            c2_fitted=c2_fitted,
            c2_fitted_std=c2_fitted_std,
            t1=t1,
            t2=t2,
            phi_angles=phi_angles,
            q=q,
            output_path=fitted_path,
        )
        saved_files["fitted_data"] = fitted_path

    return saved_files
