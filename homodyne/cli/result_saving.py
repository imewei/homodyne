"""Result saving for Homodyne CLI.

Handles saving NLSQ and MCMC/CMC optimization results including
JSON files, NPZ data, and plot generation.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from homodyne.config.parameter_names import get_physical_param_names
from homodyne.config.types import (
    LAMINAR_FLOW_PARAM_NAMES,
    SCALING_PARAM_NAMES,
    STATIC_PARAM_NAMES,
)
from homodyne.io.json_utils import json_safe as _io_json_safe
from homodyne.io.mcmc_writers import (
    create_mcmc_analysis_dict as _io_create_mcmc_analysis_dict,
)
from homodyne.io.mcmc_writers import (
    create_mcmc_diagnostics_dict as _io_create_mcmc_diagnostics_dict,
)
from homodyne.io.mcmc_writers import (
    create_mcmc_parameters_dict as _io_create_mcmc_parameters_dict,
)
from homodyne.io.nlsq_writers import (
    save_nlsq_json_files as _io_save_nlsq_json_files,
)
from homodyne.io.nlsq_writers import (
    save_nlsq_npz_file as _io_save_nlsq_npz_file,
)
from homodyne.optimization.nlsq.fit_computation import (
    compute_theoretical_fits,
)
from homodyne.optimization.nlsq.validation.fit_quality import (
    FitQualityConfig,
    validate_fit_quality,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def _json_safe(value: Any) -> Any:
    """Convert nested objects to JSON-serializable primitives.

    This is a wrapper that delegates to homodyne.io.json_utils.
    """
    return _io_json_safe(value)


def _json_serializer(obj: Any) -> Any:
    """JSON serializer for numpy arrays and other objects.

    Delegates to homodyne.io.json_utils.json_serializer for NaN/Inf sanitization.
    """
    from homodyne.io.json_utils import json_serializer

    return json_serializer(obj)


def _save_nlsq_json_files(
    param_dict: dict[str, Any],
    analysis_dict: dict[str, Any],
    convergence_dict: dict[str, Any],
    output_dir: Path,
) -> None:
    """Save 3 JSON files: parameters, analysis results, convergence metrics.

    This is a wrapper that delegates to homodyne.io.nlsq_writers.
    """
    _io_save_nlsq_json_files(param_dict, analysis_dict, convergence_dict, output_dir)


def _save_nlsq_npz_file(
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

    This is a wrapper that delegates to homodyne.io.nlsq_writers.
    """
    _io_save_nlsq_npz_file(
        phi_angles,
        c2_exp,
        c2_raw,
        c2_scaled,
        c2_solver,
        per_angle_scaling,
        per_angle_scaling_solver,
        residuals,
        residuals_norm,
        t1,
        t2,
        q,
        output_dir,
    )


def _extract_nlsq_metadata(config: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Extract required metadata for NLSQ theoretical fit computation.

    Implements multi-level fallback hierarchy for robust metadata extraction:
    - L (characteristic length): stator_rotor_gap -> sample_detector_distance -> default
    - dt (time step): analyzer_parameters.dt -> experimental_data.dt -> None
    - q (wavevector): from data['wavevector_q_list'][0]

    Parameters
    ----------
    config : ConfigManager
        Configuration manager with analyzer_parameters and experimental_data
    data : dict[str, Any]
        Experimental data dictionary with wavevector_q_list

    Returns
    -------
    dict[str, Any]
        Dictionary with keys 'L', 'dt', 'q' (may be None if not found)

    Notes
    -----
    Default L = 2000000.0 A (200 um, typical rheology-XPCS stator-rotor gap).
    Missing dt or q will log warnings but not crash - downstream functions
    must handle None values appropriately.
    """
    metadata: dict[str, Any] = {}

    # Normalize config access: support dict, ConfigManager, or object with .config
    if isinstance(config, dict):
        config_dict = config
    elif hasattr(config, "config") and isinstance(config.config, dict):
        config_dict = config.config
    elif hasattr(config, "get_config"):
        config_dict = config.get_config()
    else:
        config_dict = getattr(config, "config", {})

    # L (characteristic length) extraction with fallback hierarchy
    try:
        analyzer_params = config_dict.get("analyzer_parameters", {})
        geometry = analyzer_params.get("geometry", {})

        if "stator_rotor_gap" in geometry:
            metadata["L"] = float(geometry["stator_rotor_gap"])
            logger.debug(f"Using stator_rotor_gap L = {metadata['L']:.1f} A")
        else:
            exp_config = config_dict.get("experimental_data", {})
            exp_geometry = exp_config.get("geometry", {})

            if "stator_rotor_gap" in exp_geometry:
                metadata["L"] = float(exp_geometry["stator_rotor_gap"])
                logger.debug("Using L from experimental_data.geometry")
            elif "sample_detector_distance" in exp_config:
                metadata["L"] = float(exp_config["sample_detector_distance"])
                logger.debug(
                    f"Using sample_detector_distance L = {metadata['L']:.1f} A",
                )
            else:
                metadata["L"] = 2000000.0  # Default: 200 um
                logger.warning(
                    f"No L parameter found, using default L = {metadata['L']:.1f} A",
                )
    except (AttributeError, TypeError, ValueError) as e:
        metadata["L"] = 2000000.0
        logger.warning(f"Error reading L: {e}, using default L = 2000000.0 A")

    # dt (time step) extraction (optional)
    try:
        analyzer_params = config_dict.get("analyzer_parameters", {})
        dt_value = analyzer_params.get("dt")

        if dt_value is None:
            exp_config = config_dict.get("experimental_data", {})
            dt_value = exp_config.get("dt")

        if dt_value is not None:
            metadata["dt"] = float(dt_value)
            logger.debug(f"Using dt = {metadata['dt']:.6f} s")
        else:
            metadata["dt"] = None
            logger.warning("dt not found in config - may need manual specification")
    except (AttributeError, TypeError, ValueError) as e:
        metadata["dt"] = None
        logger.warning(f"Error reading dt: {e}")

    # q (wavevector magnitude) extraction from data
    try:
        q_list = np.asarray(data["wavevector_q_list"])
        if q_list.size > 0:
            metadata["q"] = float(q_list[0])
            logger.debug(f"Using q = {metadata['q']:.6f} A^-1")
        else:
            metadata["q"] = None
            logger.warning("Empty wavevector_q_list")
    except (KeyError, IndexError, TypeError) as e:
        metadata["q"] = None
        logger.error(f"Error extracting q: {e}")

    return metadata


def _prepare_parameter_data(
    result: Any,
    analysis_mode: str,
    n_angles: int | None = None,
) -> dict[str, Any]:
    """Prepare parameter data dictionary for JSON saving.

    Extracts parameter values and uncertainties from OptimizationResult and
    organizes them by name according to the analysis mode.

    Handles both legacy scalar scaling (9 params) and per-angle scaling (13+ params).

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result
    analysis_mode : str
        Analysis mode ("static_isotropic"/"static" or "laminar_flow")
    n_angles : int, optional
        Number of angles in the data (used to detect per-angle scaling).
        If omitted, it is inferred from the parameter vector assuming the
        canonical 2*n_angles + n_physical layout.

    Returns
    -------
    dict[str, Any]
        Dictionary mapping parameter names to {value, uncertainty} dicts
    """
    # Get parameter names for analysis mode
    normalized_mode = analysis_mode.lower()

    if normalized_mode in {"static", "static_isotropic"}:
        param_names = SCALING_PARAM_NAMES + STATIC_PARAM_NAMES
        n_physical = len(STATIC_PARAM_NAMES)
        mode_key = "static"
    elif normalized_mode == "laminar_flow":
        param_names = SCALING_PARAM_NAMES + LAMINAR_FLOW_PARAM_NAMES
        n_physical = len(LAMINAR_FLOW_PARAM_NAMES)
        mode_key = "laminar_flow"
    else:
        raise ValueError(f"Unknown analysis_mode: {analysis_mode}")

    # Detect if per-angle scaling was used
    n_params_expected_legacy = len(param_names)  # 9 for laminar_flow, 5 for static

    if n_angles is None:
        remainder = max(0, len(result.parameters) - n_physical)
        if remainder % 2 != 0 and remainder > 0:
            logger.warning(
                "Cannot cleanly infer n_angles: parameter count %d - n_physical %d = %d (odd). "
                "Falling back to n_angles=1.",
                len(result.parameters),
                n_physical,
                remainder,
            )
        inferred = remainder // 2 if remainder % 2 == 0 and remainder else 1
        n_angles = max(1, inferred)
        logger.debug(
            "Inferred n_angles=%s for _prepare_parameter_data (mode=%s, params=%s)",
            n_angles,
            mode_key,
            len(result.parameters),
        )

    n_params_expected_per_angle = 2 * n_angles + n_physical  # Per-angle scaling format
    n_params_actual = len(result.parameters)

    # Check if per-angle scaling was used
    if n_params_actual == n_params_expected_per_angle:
        # Per-angle scaling detected
        logger.info(
            f"Detected per-angle scaling: {n_params_actual} parameters for {n_angles} angles"
        )
        logger.debug(
            f"Parameter structure: [{n_angles} contrast] + [{n_angles} offset] + [{n_physical} physical]"
        )

        # Extract per-angle contrast and offset
        contrast_per_angle = result.parameters[:n_angles]
        offset_per_angle = result.parameters[n_angles : 2 * n_angles]

        # Extract physical parameters (start after 2*n_angles)
        physical_params = result.parameters[2 * n_angles :]
        physical_uncertainties = (
            result.uncertainties[2 * n_angles :]
            if result.uncertainties is not None
            else None
        )

        logger.debug(
            f"Physical params array (indices {2 * n_angles}-{len(result.parameters) - 1}): {physical_params[:7]}"
        )

        # Use mean contrast/offset for JSON (representative value; NaN-safe for failed angles)
        contrast_mean = float(np.nanmean(contrast_per_angle))
        offset_mean = float(np.nanmean(offset_per_angle))

        # Compute uncertainties for contrast/offset (RMS of per-angle uncertainties)
        if result.uncertainties is not None:
            contrast_unc_per_angle = result.uncertainties[:n_angles]
            offset_unc_per_angle = result.uncertainties[n_angles : 2 * n_angles]
            contrast_unc = float(np.sqrt(np.nanmean(contrast_unc_per_angle**2)))
            offset_unc = float(np.sqrt(np.nanmean(offset_unc_per_angle**2)))
        else:
            contrast_unc = None
            offset_unc = None

        # Build parameter dictionary
        param_dict = {
            "contrast": {"value": contrast_mean, "uncertainty": contrast_unc},
            "offset": {"value": offset_mean, "uncertainty": offset_unc},
        }

        # Add physical parameters
        physical_param_names = (
            STATIC_PARAM_NAMES if mode_key == "static" else LAMINAR_FLOW_PARAM_NAMES
        )
        for i, name in enumerate(physical_param_names):
            param_dict[name] = {
                "value": float(physical_params[i]),
                "uncertainty": (
                    float(physical_uncertainties[i])
                    if physical_uncertainties is not None
                    else None
                ),
            }

        logger.debug(
            f"Extracted parameters - contrast_mean={contrast_mean:.4f}, "
            f"offset_mean={offset_mean:.4f}, "
            f"D0={param_dict.get('D0', {}).get('value', 'N/A')}, "
            f"alpha={param_dict.get('alpha', {}).get('value', 'N/A')}, "
            f"D_offset={param_dict.get('D_offset', {}).get('value', 'N/A')}, "
            f"gamma_dot_t0={param_dict.get('gamma_dot_t0', {}).get('value', 'N/A')}, "
            f"beta={param_dict.get('beta', {}).get('value', 'N/A')}, "
            f"gamma_dot_t_offset={param_dict.get('gamma_dot_t_offset', {}).get('value', 'N/A')}, "
            f"phi0={param_dict.get('phi0', {}).get('value', 'N/A')}"
        )

    elif n_params_actual == n_params_expected_legacy:
        # Legacy scalar scaling format
        logger.warning(
            f"Legacy format detected: {n_params_actual} parameters (expected {n_params_expected_per_angle} for per-angle). "
            f"This typically indicates a failed optimization."
        )

        param_dict = {}
        for i, name in enumerate(param_names):
            param_dict[name] = {
                "value": float(result.parameters[i]),
                "uncertainty": (
                    float(result.uncertainties[i])
                    if result.uncertainties is not None
                    and i < len(result.uncertainties)
                    else None
                ),
            }

        logger.debug(f"Extracted legacy parameters: {list(param_dict.keys())}")

    else:
        # Unexpected parameter count
        raise ValueError(
            f"Unexpected parameter count: got {n_params_actual}, expected {n_params_expected_per_angle} "
            f"for per-angle scaling with {n_angles} angles, or {n_params_expected_legacy} for legacy format."
        )

    return param_dict


def _save_results(
    args: Any,
    result: Any,
    device_config: dict[str, Any],
    data: dict[str, Any],
    config: Any,
) -> None:
    """Save optimization results to output directory.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    result : Any
        Optimization result (OptimizationResult or MCMC result)
    device_config : dict
        Device configuration
    data : dict
        Experimental data dictionary
    config : ConfigManager
        Configuration manager
    """
    import yaml

    from homodyne.utils.async_io import AsyncWriter

    logger.info(f"Saving results to: {args.output_dir}")

    writer = AsyncWriter(max_workers=2)

    # Route to appropriate saving method based on optimization method.
    if args.method == "nlsq":
        logger.info("Using comprehensive NLSQ result saving (async)")
        writer.submit_task(save_nlsq_results, result, data, config, args.output_dir)
        if args.output_format != "json":
            logger.info("Saving legacy results summary for backward compatibility")
    elif args.method == "cmc":
        logger.info("Using comprehensive CMC result saving (async)")
        writer.submit_task(save_mcmc_results, result, data, config, args.output_dir)
        if args.output_format != "json":
            logger.info("Saving legacy results summary for backward compatibility")

    # Create results summary
    results_summary = {
        "method": args.method,
        "analysis_mode": getattr(result, "analysis_mode", "unknown"),
        "success": getattr(result, "success", True),
        "optimization_time": getattr(result, "optimization_time", 0.0),
        "device_config": device_config,
        "parameters": {},
        "diagnostics": {},
    }

    # Extract parameters based on result type
    if hasattr(result, "parameters"):
        results_summary["parameters"] = (
            result.parameters.tolist()
            if hasattr(result.parameters, "tolist")
            else result.parameters
        )
    elif hasattr(result, "mean_params"):
        # MCMC result format
        results_summary["parameters"] = {
            "contrast": result.mean_contrast,
            "offset": result.mean_offset,
            "physical_params": (
                result.mean_params.tolist()
                if hasattr(result.mean_params, "tolist")
                else result.mean_params
            ),
        }

    # Extract diagnostics
    if hasattr(result, "chi_squared"):
        results_summary["diagnostics"]["chi_squared"] = result.chi_squared
    if hasattr(result, "converged"):
        results_summary["diagnostics"]["converged"] = result.converged

    # Save in requested format
    output_file = args.output_dir / f"homodyne_results.{args.output_format}"

    try:
        if args.output_format == "yaml":
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(_json_safe(results_summary), f, default_flow_style=False)
        elif args.output_format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_summary, f, indent=2, default=_json_serializer)
        elif args.output_format == "npz":
            # Save numpy arrays
            arrays_to_save: dict[str, Any] = {
                "results_summary": np.array([results_summary], dtype=object),
            }
            if hasattr(result, "samples_params") and result.samples_params is not None:
                arrays_to_save["samples_params"] = result.samples_params
            np.savez_compressed(output_file, **arrays_to_save)

        logger.info(f"OK: Results saved: {output_file}")

    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to save results: {e}")

    # Wait for background writes to complete
    logger.debug("Waiting for background result writes to complete...")
    errors = writer.wait_all(timeout=60.0)
    if errors:
        logger.warning(
            "Background save errors (%d): %s",
            len(errors),
            [f"{type(e).__name__}: {e}" for e in errors],
        )
    writer.shutdown()


def save_nlsq_results(
    result: Any,
    data: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> None:
    """Save complete NLSQ optimization results to structured directory.

    Main orchestrator function that coordinates all helper functions to save:
    - parameters.json: Parameter values and uncertainties
    - fitted_data.npz: Experimental + theoretical + residuals
    - analysis_results_nlsq.json: Analysis summary
    - convergence_metrics.json: Convergence diagnostics

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result with parameters, uncertainties, chi-squared, etc.
    data : dict[str, Any]
        Experimental data with phi_angles_list, c2_exp, t1, t2, wavevector_q_list
    config : ConfigManager
        Configuration with analysis_mode and metadata
    output_dir : Path
        Output directory (nlsq/ subdirectory will be created)
    """
    from homodyne.cli.data_pipeline import _apply_angle_filtering_for_optimization
    from homodyne.cli.plot_dispatch import generate_nlsq_plots

    # Create nlsq subdirectory
    nlsq_dir = output_dir / "nlsq"
    nlsq_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving NLSQ results to {nlsq_dir}")

    # Get analysis mode
    analysis_mode = config.config.get("analysis_mode", "static_isotropic")

    # Apply phi filtering to data (if enabled in config)
    filtered_data = _apply_angle_filtering_for_optimization(data, config)
    logger.debug(
        f"Applied phi filtering for NLSQ saving: "
        f"{len(filtered_data['phi_angles_list'])} angles selected"
    )

    # Step 1: Extract metadata
    logger.debug("Extracting metadata (L, dt, q)")
    metadata = _extract_nlsq_metadata(config, filtered_data)

    # Step 2: Prepare parameter data
    n_angles = len(filtered_data["phi_angles_list"])
    logger.debug(
        f"Preparing parameter data for {analysis_mode} mode with {n_angles} angles"
    )
    param_dict = _prepare_parameter_data(result, analysis_mode, n_angles)

    # Add timestamp and convergence info to parameters
    param_dict_complete = {
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": analysis_mode,
        "chi_squared": float(result.chi_squared),
        "reduced_chi_squared": float(result.reduced_chi_squared),
        "convergence_status": result.convergence_status,
        "parameters": param_dict,
    }

    # Step 3: Compute theoretical fits with per-angle scaling
    logger.info("Computing theoretical fits with per-angle scaling")
    try:
        fits_dict = compute_theoretical_fits(
            result,
            filtered_data,
            metadata,
            analysis_mode=analysis_mode,
            include_solver_surface=True,
        )
    except ValueError as exc:
        logger.warning(
            f"Could not compute theoretical fits (skipping NPZ and plots): {exc}"
        )
        fits_dict = None

    if fits_dict is not None:
        scalar_expanded = fits_dict.pop("scalar_per_angle_expansion", False)
        if scalar_expanded:
            logger.warning(
                "Recorded scalar_per_angle_expansion=true in diagnostics (scalar contrast/offset replicated per angle)."
            )
            if getattr(result, "nlsq_diagnostics", None) is None:
                result.nlsq_diagnostics = {"scalar_per_angle_expansion": True}
            elif isinstance(result.nlsq_diagnostics, dict):
                result.nlsq_diagnostics["scalar_per_angle_expansion"] = True

    # Step 4: Prepare analysis results dictionary
    phi_angles = np.asarray(filtered_data["phi_angles_list"])
    c2_exp = np.asarray(filtered_data["c2_exp"])
    n_angles = len(phi_angles)
    n_data_points = c2_exp.size
    n_params = len(result.parameters)
    degrees_of_freedom = n_data_points - n_params

    analysis_dict = {
        "method": "nlsq",
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": analysis_mode,
        "fit_quality": {
            "chi_squared": float(result.chi_squared),
            "reduced_chi_squared": float(result.reduced_chi_squared),
            "degrees_of_freedom": degrees_of_freedom,
            "quality_flag": result.quality_flag,
        },
        "dataset_info": {
            "n_angles": n_angles,
            "n_time_points": c2_exp.shape[1] * c2_exp.shape[2],
            "total_data_points": n_data_points,
            "q_value": float(metadata["q"]) if metadata.get("q") is not None else None,
        },
        "optimization_summary": {
            "convergence_status": result.convergence_status,
            "iterations": result.iterations,
            "execution_time": float(result.execution_time),
        },
    }

    # Step 5: Prepare convergence metrics dictionary
    convergence_dict = {
        "convergence": {
            "status": result.convergence_status,
            "iterations": result.iterations,
            "execution_time": float(result.execution_time),
            "final_chi_squared": float(result.chi_squared),
            "chi_squared_reduction": (
                1.0 - result.reduced_chi_squared
                if result.reduced_chi_squared < 1.0
                else 0.0
            ),
        },
        "recovery_actions": result.recovery_actions,
        "quality_flag": result.quality_flag,
        "device_info": result.device_info,
    }

    # Step 5.5: Validate fit quality (T056)
    param_labels = getattr(result, "param_labels", None)
    if param_labels is None:
        if analysis_mode == "laminar_flow":
            physical_labels = list(LAMINAR_FLOW_PARAM_NAMES)
        else:
            physical_labels = list(STATIC_PARAM_NAMES)
        scaling_labels = []
        for i in range(n_angles):
            scaling_labels.append(f"contrast[{i}]")
        for i in range(n_angles):
            scaling_labels.append(f"offset[{i}]")
        param_labels = scaling_labels + physical_labels

    # Get bounds from config if available
    bounds = None
    try:
        if hasattr(config, "get_parameter_bounds"):
            bounds_list = config.get_parameter_bounds(param_labels)
            if bounds_list:
                lower = np.array([b["min"] for b in bounds_list])
                upper = np.array([b["max"] for b in bounds_list])
                bounds = (lower, upper)
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug(f"Could not get bounds for quality validation: {e}")

    # Create quality config from nlsq settings
    nlsq_config = config.config.get("optimization", {}).get("nlsq", {})
    quality_config = FitQualityConfig(
        enable=nlsq_config.get("enable_quality_validation", True),
        reduced_chi_squared_threshold=nlsq_config.get(
            "quality_reduced_chi_squared_threshold", 10.0
        ),
        warn_on_max_restarts=nlsq_config.get("quality_warn_on_max_restarts", True),
        warn_on_bounds_hit=nlsq_config.get("quality_warn_on_bounds_hit", True),
        warn_on_convergence_failure=nlsq_config.get(
            "quality_warn_on_convergence_failure", True
        ),
        bounds_tolerance=nlsq_config.get("quality_bounds_tolerance", 1e-9),
    )

    # Run validation
    quality_report = validate_fit_quality(
        result, bounds=bounds, config=quality_config, param_labels=param_labels
    )

    # Add quality report to convergence dict
    convergence_dict.update(quality_report.to_dict())

    # Step 6: Save JSON files
    logger.info("Saving JSON files (parameters, analysis, convergence)")
    _save_nlsq_json_files(
        param_dict_complete,
        analysis_dict,
        convergence_dict,
        nlsq_dir,
    )

    # Step 8b: Persist diagnostics payload if available
    diagnostics_payload = getattr(result, "nlsq_diagnostics", None)
    if diagnostics_payload:
        diagnostics_file = nlsq_dir / "diagnostics.json"
        try:
            with open(diagnostics_file, "w", encoding="utf-8") as f:
                json.dump(_json_safe(diagnostics_payload), f, indent=2)
            logger.info(f"Saved diagnostics to {diagnostics_file}")
        except (TypeError, OSError) as exc:
            logger.warning(
                "Failed to save diagnostics.json (%s). Payload keys: %s",
                exc,
                list(diagnostics_payload.keys()),
            )

    if fits_dict is None:
        logger.info(f"OK: NLSQ results saved successfully to {nlsq_dir}")
        logger.info(
            "  - 3 JSON files (parameters, analysis results, convergence metrics)"
        )
        logger.info("  - NPZ and plots skipped (theoretical fits unavailable)")
        return

    # Step 7: Compute normalized residuals
    residuals_norm = np.divide(
        fits_dict["residuals"],
        0.05 * c2_exp,
        out=np.zeros_like(fits_dict["residuals"]),
        where=(c2_exp != 0),
    )

    # Convert time arrays to 1D
    t1 = np.asarray(filtered_data["t1"])
    t2 = np.asarray(filtered_data["t2"])
    if t1.ndim == 2:
        t1 = t1[:, 0]
    if t2.ndim == 2:
        t2 = t2[0, :]

    logger.debug("Using time arrays in seconds (already converted by data loader)")

    # Step 8: Save NPZ file
    logger.info("Saving NPZ file with all arrays")
    _save_nlsq_npz_file(
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        c2_raw=fits_dict["c2_theoretical_raw"],
        c2_scaled=fits_dict["c2_theoretical_scaled"],
        c2_solver=fits_dict["c2_solver_scaled"],
        per_angle_scaling=fits_dict["per_angle_scaling"],
        per_angle_scaling_solver=fits_dict["per_angle_scaling_solver"],
        residuals=fits_dict["residuals"],
        residuals_norm=residuals_norm,
        t1=t1,
        t2=t2,
        q=metadata["q"],
        output_dir=nlsq_dir,
    )

    logger.info(f"OK: NLSQ results saved successfully to {nlsq_dir}")
    logger.info("  - 3 JSON files (parameters, analysis results, convergence metrics)")
    n_arrays = 10 + (1 if fits_dict.get("c2_solver_scaled") is not None else 0)
    logger.info(
        f"  - 1 NPZ file ({n_arrays} arrays: experimental + theoretical + residuals)"
    )

    # Step 9: Generate plots with graceful degradation
    try:
        logger.info("Generating heatmap plots")
        generate_nlsq_plots(
            phi_angles=phi_angles,
            c2_exp=c2_exp,
            c2_theoretical_scaled=fits_dict["c2_theoretical_scaled"],
            residuals=fits_dict["residuals"],
            t1=t1,
            t2=t2,
            output_dir=nlsq_dir,
            config=config,
            c2_solver_scaled=fits_dict["c2_solver_scaled"],
        )
        logger.info(f"  - {len(phi_angles)} PNG plots")
    except Exception as e:
        logger.warning(f"Plot generation failed (data files still saved): {e}")
        logger.debug("Plot error details:", exc_info=True)


def save_mcmc_results(
    result: Any,
    data: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> None:
    """Save CMC results with comprehensive diagnostics.

    Creates cmc/ directory and saves:
    1. parameters.json: Posterior mean +/- std for each parameter
    2. analysis_results_cmc.json: Sampling summary and diagnostics
    3. samples.npz: Full posterior samples, r_hat, ess
    4. diagnostics.json: Convergence metrics
    5. fitted_data.npz: Experimental + theoretical data (optional)
    6. c2_heatmaps_phi_*.png: Comparison plots using posterior mean
    7. ArviZ diagnostic plots: pair, forest, energy, autocorr, rank, ess

    Parameters
    ----------
    result : CMCResult
        CMC optimization result with posterior samples and diagnostics
    data : dict
        Experimental data dictionary containing c2_exp, phi_angles_list, t1, t2, q
    config : ConfigManager
        Configuration manager with analysis settings
    output_dir : Path
        Base output directory (cmc/ subdirectory will be created)
    """
    from homodyne.cli.data_pipeline import _apply_angle_filtering_for_optimization
    from homodyne.cli.plot_dispatch import generate_nlsq_plots

    # CMC-only - always use "cmc" directory
    method_name = "cmc"

    # Create method-specific directory
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {method_name.upper()} results to: {method_dir}")

    # Step 1: Save parameters.json with posterior statistics
    try:
        param_dict = _create_mcmc_parameters_dict(result)
        param_file = method_dir / "parameters.json"
        with open(param_file, "w", encoding="utf-8") as f:
            json.dump(param_dict, f, indent=2, default=_json_serializer)
        logger.debug(f"Saved parameters to {param_file}")
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to save parameters.json: {e}")
        logger.debug("Parameter saving error:", exc_info=True)

    # Step 2: Save samples.npz with full posterior
    try:
        samples_file = method_dir / "samples.npz"
        save_dict: dict[str, Any] = {}

        # Combine samples from separate attributes
        if hasattr(result, "samples_params") and result.samples_params is not None:
            samples_list = [result.samples_params]
            contrast_inserted = False

            if (
                hasattr(result, "samples_contrast")
                and result.samples_contrast is not None
            ):
                contrast_samples = result.samples_contrast
                if contrast_samples.ndim == 1:
                    contrast_samples = contrast_samples[:, np.newaxis]
                samples_list.insert(0, contrast_samples)
                contrast_inserted = True

            if hasattr(result, "samples_offset") and result.samples_offset is not None:
                offset_samples = result.samples_offset
                if offset_samples.ndim == 1:
                    offset_samples = offset_samples[:, np.newaxis]
                samples_list.insert(1 if contrast_inserted else 0, offset_samples)

            save_dict["samples"] = np.concatenate(samples_list, axis=1)

        elif (
            hasattr(result, "samples")
            and isinstance(result.samples, dict)
            and result.samples
        ):
            param_names = getattr(result, "param_names", list(result.samples.keys()))
            arrays = []
            for name in param_names:
                if name in result.samples:
                    arr = result.samples[name].flatten()
                    arrays.append(arr[:, np.newaxis])
            if arrays:
                save_dict["samples"] = np.concatenate(arrays, axis=1)
                save_dict["param_names"] = np.array(param_names)

        # Add optional diagnostics if available
        if hasattr(result, "log_prob") and result.log_prob is not None:
            save_dict["log_prob"] = result.log_prob
        if hasattr(result, "r_hat") and result.r_hat is not None:
            if isinstance(result.r_hat, dict):
                save_dict["r_hat"] = np.array(list(result.r_hat.values()))
            else:
                save_dict["r_hat"] = result.r_hat

        # ESS: try ess_bulk (CMCResult) then effective_sample_size (legacy)
        ess_source = None
        if hasattr(result, "ess_bulk") and result.ess_bulk is not None:
            ess_source = result.ess_bulk
        elif (
            hasattr(result, "effective_sample_size")
            and result.effective_sample_size is not None
        ):
            ess_source = result.effective_sample_size
        if ess_source is not None:
            if isinstance(ess_source, dict):
                save_dict["ess"] = np.array(list(ess_source.values()))
            else:
                save_dict["ess"] = ess_source

        if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
            save_dict["acceptance_rate"] = np.array([result.acceptance_rate])

        if save_dict:
            np.savez_compressed(str(samples_file), **save_dict)
            try:
                samples_size_mb = samples_file.stat().st_size / (1024 * 1024)
            except OSError:
                samples_size_mb = 0
                logger.debug("Could not stat samples file")
            logger.debug(
                f"Saved posterior samples to {samples_file} ({samples_size_mb:.2f} MB)"
            )
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to save samples.npz: {e}")
        logger.debug("Samples saving error:", exc_info=True)

    # Step 3: Save analysis_results_mcmc.json
    try:
        analysis_dict = _create_mcmc_analysis_dict(result, data, method_name)
        analysis_file = method_dir / f"analysis_results_{method_name}.json"
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2, default=_json_serializer)
        logger.debug(f"Saved analysis results to {analysis_file}")
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to save analysis_results_{method_name}.json: {e}")
        logger.debug("Analysis results saving error:", exc_info=True)

    # Step 4: Save diagnostics.json
    try:
        diagnostics_dict = _create_mcmc_diagnostics_dict(result)
        diagnostics_file = method_dir / "diagnostics.json"
        with open(diagnostics_file, "w", encoding="utf-8") as f:
            json.dump(diagnostics_dict, f, indent=2, default=_json_serializer)
        logger.debug(f"Saved diagnostics to {diagnostics_file}")
    except (OSError, ValueError, TypeError) as e:
        logger.warning(f"Failed to save diagnostics.json: {e}")
        logger.debug("Diagnostics saving error:", exc_info=True)

    # Step 4b: Save shard_diagnostics.json for CMC results
    if (
        method_name == "cmc"
        and hasattr(result, "per_shard_diagnostics")
        and result.per_shard_diagnostics
    ):
        try:
            shard_diag_file = method_dir / "shard_diagnostics.json"
            with open(shard_diag_file, "w", encoding="utf-8") as f:
                json.dump(
                    result.per_shard_diagnostics, f, indent=2, default=_json_serializer
                )
            logger.debug(f"Saved per-shard diagnostics to {shard_diag_file}")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Failed to save shard_diagnostics.json: {e}")
            logger.debug("Shard diagnostics saving error:", exc_info=True)

    # Step 5: Generate heatmap plots (reuse NLSQ plotting)
    try:
        logger.info("Generating comparison heatmap plots")

        # Apply phi filtering to data (if enabled in config)
        filtered_data = _apply_angle_filtering_for_optimization(data, config)
        logger.debug(
            f"Applied phi filtering for MCMC plotting: "
            f"{len(filtered_data['phi_angles_list'])} angles selected"
        )

        # Compute theoretical C2 using posterior mean parameters
        c2_result = _compute_theoretical_c2_from_mcmc(result, filtered_data, config)
        c2_theoretical_scaled = c2_result["c2_theoretical_scaled"]
        c2_theoretical_raw = c2_result["c2_theoretical_raw"]
        per_angle_scaling = c2_result["per_angle_scaling"]

        # Calculate residuals
        c2_exp = filtered_data["c2_exp"]
        residuals = c2_exp - c2_theoretical_scaled

        # Convert time arrays
        t1 = np.asarray(filtered_data["t1"])
        t2 = np.asarray(filtered_data["t2"])
        if t1.ndim == 2:
            t1 = t1[:, 0]
        if t2.ndim == 2:
            t2 = t2[0, :]

        # Save fitted_data.npz (before plotting so data is saved even if plots fail)
        phi_angles = np.asarray(filtered_data["phi_angles_list"])
        q_val = filtered_data.get("wavevector_q_list", [1.0])[0]
        residuals_normalized = residuals / (0.05 * np.where(c2_exp != 0, c2_exp, 1.0))
        npz_file = method_dir / "fitted_data.npz"
        np.savez_compressed(
            npz_file,
            phi_angles=phi_angles,
            c2_exp=np.asarray(c2_exp),
            c2_theoretical_raw=c2_theoretical_raw,
            c2_theoretical_scaled=c2_theoretical_scaled,
            per_angle_scaling=per_angle_scaling,
            residuals=residuals,
            residuals_normalized=residuals_normalized,
            t1=t1,
            t2=t2,
            q=np.array([q_val]),
        )
        try:
            fitted_size_mb = npz_file.stat().st_size / (1024 * 1024)
        except OSError:
            fitted_size_mb = 0
            logger.debug("Could not stat fitted data file")
        logger.info(f"Saved fitted_data.npz ({fitted_size_mb:.2f} MB)")

        # Generate plots using NLSQ plotting function
        generate_nlsq_plots(
            phi_angles=filtered_data["phi_angles_list"],
            c2_exp=c2_exp,
            c2_theoretical_scaled=c2_theoretical_scaled,
            residuals=residuals,
            t1=t1,
            t2=t2,
            output_dir=method_dir,
            config=config,
            c2_solver_scaled=None,
        )
        logger.info(f"  - {len(filtered_data['phi_angles_list'])} PNG heatmap plots")
    except Exception as e:
        logger.warning(f"Heatmap plot generation failed (data files still saved): {e}")
        logger.debug("Plot error details:", exc_info=True)

    # T057: Calculate and log total file sizes
    json_files = list(method_dir.glob("*.json"))
    npz_files = list(method_dir.glob("*.npz"))
    try:
        total_json_kb = sum(f.stat().st_size for f in json_files) / 1024
    except OSError:
        total_json_kb = 0
    try:
        total_npz_mb = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)
    except OSError:
        total_npz_mb = 0

    logger.info(f"OK: {method_name.upper()} results saved successfully to {method_dir}")
    if (
        method_name == "cmc"
        and hasattr(result, "per_shard_diagnostics")
        and result.per_shard_diagnostics
    ):
        logger.info(
            f"  - 4 JSON files (parameters, analysis results, diagnostics, shard diagnostics) "
            f"({total_json_kb:.1f} KB total)"
        )
    else:
        logger.info(
            f"  - 3 JSON files (parameters, analysis results, diagnostics) "
            f"({total_json_kb:.1f} KB total)"
        )
    logger.info(
        f"  - {len(npz_files)} NPZ file(s) (posterior samples) ({total_npz_mb:.2f} MB)"
    )


def _create_mcmc_parameters_dict(result: Any) -> dict:
    """Create parameters dictionary with posterior statistics.

    This is a wrapper that delegates to homodyne.io.mcmc_writers.
    """
    return _io_create_mcmc_parameters_dict(result)


def _create_mcmc_analysis_dict(
    result: Any,
    data: dict[str, Any],
    method_name: str,
) -> dict:
    """Create analysis results dictionary for MCMC/CMC.

    This is a wrapper that delegates to homodyne.io.mcmc_writers.
    """
    return _io_create_mcmc_analysis_dict(result, data, method_name)


def _create_mcmc_diagnostics_dict(result: Any) -> dict:
    """Create diagnostics dictionary for MCMC/CMC.

    This is a wrapper that delegates to homodyne.io.mcmc_writers.
    """
    return _io_create_mcmc_diagnostics_dict(result)


def _get_parameter_names(analysis_mode: str) -> list[str]:
    """Get physical parameter names for given analysis mode.

    This is a thin wrapper around get_physical_param_names() that handles
    unknown modes gracefully with a warning instead of raising an exception.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode ("static", "static_isotropic", or "laminar_flow")

    Returns
    -------
    list[str]
        List of physical parameter names (without scaling params)
    """
    # Normalize "static" to "static_isotropic" for compatibility
    normalized_mode = "static_isotropic" if analysis_mode == "static" else analysis_mode

    try:
        # Type cast to handle literal type requirement
        if normalized_mode in ("static_isotropic", "laminar_flow"):
            return get_physical_param_names(cast(Any, normalized_mode))
        else:
            logger.warning(
                f"Unknown analysis mode: {analysis_mode}, assuming static_isotropic"
            )
            return get_physical_param_names("static_isotropic")
    except ValueError:
        logger.warning(
            f"Unknown analysis mode: {analysis_mode}, assuming static_isotropic"
        )
        return get_physical_param_names("static_isotropic")


def _compute_theoretical_c2_from_mcmc(
    result: Any,
    data: dict[str, Any],
    config: Any,
) -> dict[str, np.ndarray]:
    """Compute theoretical C2 using MCMC posterior mean parameters.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with posterior mean parameters
    data : dict
        Experimental data dictionary
    config : ConfigManager
        Configuration manager

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys:
        - "c2_theoretical_scaled": shape (n_angles, n_t1, n_t2), contrast * g1^2 + offset
        - "c2_theoretical_raw": shape (n_angles, n_t1, n_t2), unscaled g1^2
        - "per_angle_scaling": shape (n_angles, 2), [contrast, offset] per angle
    """
    # Extract parameters from MCMC result
    contrast = getattr(result, "mean_contrast", 0.5)
    offset = getattr(result, "mean_offset", 1.0)

    mean_params_obj = getattr(result, "mean_params", None)
    analysis_mode = getattr(result, "analysis_mode", "static")
    ordered_names = _get_parameter_names(analysis_mode)

    # mean_params may be a ParameterStats (hybrid), dict, or array-like
    if mean_params_obj is not None and hasattr(mean_params_obj, "get"):
        mean_params = np.array(
            [mean_params_obj.get(name, np.nan) for name in ordered_names]
        )
    elif mean_params_obj is not None and hasattr(mean_params_obj, "as_array"):
        mean_params_array = mean_params_obj.as_array
        mean_params = (
            np.asarray(mean_params_array)
            if mean_params_array is not None
            else np.array([])
        )
    elif mean_params_obj is not None:
        mean_params = np.asarray(mean_params_obj)
    else:
        mean_params = np.array([])

    # Log parameter values for debugging
    logger.info("Computing theoretical C2 with posterior means:")
    logger.info(f"  Contrast: {contrast:.6f}")
    logger.info(f"  Offset: {offset:.6f}")
    if mean_params_obj is not None and hasattr(mean_params_obj, "get"):
        D0_val = mean_params_obj.get("D0", np.nan)
        alpha_val = mean_params_obj.get("alpha", np.nan)
        D_offset_val = mean_params_obj.get("D_offset", np.nan)
        logger.info(
            f"  Physical params: D0={D0_val:.2f}, alpha={alpha_val:.4f}, D_offset={D_offset_val:.4f}"
        )
    else:
        logger.info(
            f"  Physical params (positional): [{', '.join(f'{v:.4f}' for v in mean_params[:3])}]"
        )

    # Validate parameters for reasonable theoretical prediction
    if contrast < 0.05:
        logger.warning(
            f"Very small contrast ({contrast:.4f} < 0.05) may produce nearly constant c2_theory. "
            "This suggests poor MCMC convergence or inappropriate initial values."
        )
    D0_for_validation = (
        mean_params_obj.get("D0", 0.0)
        if mean_params_obj is not None and hasattr(mean_params_obj, "get")
        else mean_params[0]
    )
    if D0_for_validation >= 99990:
        logger.warning(
            f"D0 ({D0_for_validation:.1f}) near upper bound (100000). "
            "Consider increasing max D0 bound or improving initial values."
        )

    # Get data arrays
    phi_angles = np.asarray(data["phi_angles_list"])
    t1 = np.asarray(data["t1"])
    t2 = np.asarray(data["t2"])
    q_val = data.get("wavevector_q_list", [1.0])[0]

    # Convert to 1D if needed
    if t1.ndim == 2:
        t1 = t1[:, 0]
    if t2.ndim == 2:
        t2 = t2[0, :]

    # Get analysis mode
    config_dict_mcmc: dict[str, Any] = (
        config.get_config()
        if hasattr(config, "get_config")
        else cast(dict[str, Any], config)
    )
    _analysis_mode = config_dict_mcmc.get("analysis_mode", "static_isotropic")  # noqa: F841

    # Get L parameter (stator-rotor gap) from correct config path
    analyzer_params = config_dict_mcmc.get("analyzer_parameters", {})
    L = analyzer_params.get("geometry", {}).get("stator_rotor_gap", 2000000.0)

    # Get dt parameter
    dt = analyzer_params.get("dt", 0.001)

    # Compute theoretical C2 for all angles with per-angle scaling estimation
    c2_theoretical_list: list[np.ndarray] = []
    c2_raw_list: list[np.ndarray] = []
    scaling_list: list[list[float]] = []

    # Get experimental data for per-angle lstsq fitting
    c2_exp = data.get("c2_exp", None)
    use_per_angle_lstsq = c2_exp is not None and len(c2_exp) == len(phi_angles)

    if use_per_angle_lstsq:
        logger.info(
            "Using per-angle least squares estimation for contrast/offset "
            "(fixes CMC sharding aggregation issue)"
        )

    # Pre-compute angle-independent factors outside the loop
    import jax.numpy as jnp

    from homodyne.core.jax_backend import _compute_g1_total_core

    params_jax = jnp.array(mean_params)
    wavevector_q_squared_half_dt = 0.5 * (q_val**2) * dt
    sinc_prefactor = 0.5 / jnp.pi * q_val * L * dt
    t1_jax = jnp.array(t1)
    t2_jax = jnp.array(t2)

    for i, phi in enumerate(phi_angles):
        t1_grid, t2_grid = jnp.meshgrid(t1_jax, t2_jax, indexing="ij")
        t1_flat = t1_grid.ravel()
        t2_flat = t2_grid.ravel()
        phi_flat = jnp.full_like(t1_flat, phi)

        g1_flat = _compute_g1_total_core(
            params=params_jax,
            t1=t1_flat,
            t2=t2_flat,
            phi=phi_flat,
            wavevector_q_squared_half_dt=wavevector_q_squared_half_dt,
            sinc_prefactor=sinc_prefactor,
            dt=dt,
        )
        g2_theory_np = np.array(g1_flat**2).reshape(len(t1), len(t2))

        if use_per_angle_lstsq and c2_exp is not None:
            c2_exp_angle = np.array(c2_exp[i])

            g2_flat = g2_theory_np.flatten()
            c2_flat = c2_exp_angle.flatten()

            A = np.vstack([g2_flat, np.ones_like(g2_flat)]).T

            try:
                lstsq_result = np.linalg.lstsq(A, c2_flat, rcond=None)
                contrast_raw, offset_raw = lstsq_result[0]

                if contrast_raw > 1.0:
                    logger.warning(
                        f"Angle {i} (phi={phi:.2f} deg): lstsq contrast={contrast_raw:.4f} > 1.0 "
                        "(unphysical). This indicates the physics model underestimates "
                        "the C2 variation. Clipping to 1.0."
                    )
                elif contrast_raw <= 0:
                    logger.warning(
                        f"Angle {i} (phi={phi:.2f} deg): lstsq contrast={contrast_raw:.4f} <= 0 "
                        "(unphysical). Clipping to 0.01."
                    )

                contrast_i = float(np.clip(np.asarray(contrast_raw), 0.01, 1.0))
                offset_i = float(np.clip(np.asarray(offset_raw), 0.5, 1.5))

                if i == 0:
                    logger.debug(
                        f"  Angle {i} (phi={phi:.2f} deg): fitted contrast={contrast_i:.4f}, "
                        f"offset={offset_i:.4f}"
                    )
            except (np.linalg.LinAlgError, ValueError) as e:
                logger.warning(f"lstsq failed for angle {i}, using global values: {e}")
                contrast_i = contrast
                offset_i = offset
        else:
            contrast_i = contrast
            offset_i = offset

        c2_raw_list.append(g2_theory_np)
        scaling_list.append([contrast_i, offset_i])

        c2_theoretical_np = contrast_i * g2_theory_np + offset_i
        c2_theoretical_list.append(c2_theoretical_np)

    # Stack all angles
    c2_theoretical_scaled = np.array(c2_theoretical_list)

    # Validate theoretical prediction quality (NaN-safe)
    c2_min = float(np.nanmin(c2_theoretical_scaled))
    c2_max = float(np.nanmax(c2_theoretical_scaled))
    c2_range = c2_max - c2_min
    logger.info(
        f"Theoretical C2 range: [{c2_min:.6f}, {c2_max:.6f}], variation: {c2_range:.6f}"
    )

    # Check if per-angle scaling produced reasonable variation
    if use_per_angle_lstsq and c2_exp is not None:
        c2_exp_arr = np.asarray(c2_exp)
        exp_min = float(np.nanmin(c2_exp_arr))
        exp_max = float(np.nanmax(c2_exp_arr))
        exp_range = exp_max - exp_min
        coverage = c2_range / exp_range if exp_range > 0.01 else 0
        logger.info(
            f"Per-angle lstsq scaling: fitted range covers {coverage:.1%} of "
            f"experimental range [{exp_min:.4f}, {exp_max:.4f}]"
        )

    if c2_range < 0.01:
        logger.warning(
            f"Theoretical C2 has very low variation ({c2_range:.6f} < 0.01). "
            f"The model prediction is nearly constant (c2 ~ {c2_min:.4f}). "
            "This indicates:\n"
            "  1. Poor MCMC convergence to local minimum\n"
            "  2. Inappropriate initial parameter values\n"
            "  3. Physical parameters may have hit bounds\n"
            "Recommendations:\n"
            "  - Run NLSQ first to get better initial values\n"
            "  - Check parameter bounds (especially D0 upper limit)\n"
            "  - Verify initial_parameters.values in config are reasonable"
        )

    return {
        "c2_theoretical_scaled": c2_theoretical_scaled,
        "c2_theoretical_raw": np.array(c2_raw_list),
        "per_angle_scaling": np.array(scaling_list),
    }
