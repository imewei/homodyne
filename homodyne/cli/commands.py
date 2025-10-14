"""
Command Dispatcher for Homodyne v2 CLI
======================================

Handles command execution and coordination between CLI arguments,
configuration, and optimization methods.
"""

import time
from pathlib import Path
from typing import Any

import numpy as np

from homodyne.cli.args_parser import validate_args
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Common XPCS experimental angles (in degrees) for validation
COMMON_XPCS_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180]


def normalize_angle_to_symmetric_range(angle):
    """Normalize angle(s) to [-180°, 180°] range.

    The horizontal flow direction is defined as 0°. Angles are normalized
    to be symmetric around 0° in the range [-180°, 180°].

    Physical Interpretation
    -----------------------
    In XPCS experiments with flow, the flow direction is typically set as 0°
    (horizontal reference). Angles are measured relative to this reference.
    Normalizing to [-180°, 180°] provides a natural symmetric representation
    where positive angles are counterclockwise and negative angles are
    clockwise from the flow direction.

    Normalization Rules
    -------------------
    - If 180° < φ < 360°: φ_norm = φ - 360°
      (e.g., 210° → -150°)
    - If -360° < φ < -180°: φ_norm = φ + 360°
      (e.g., -210° → 150°)
    - If -180° ≤ φ ≤ 180°: φ_norm = φ (no change)

    Parameters
    ----------
    angle : float or np.ndarray
        Angle(s) in degrees. Can be scalar (float) or array (np.ndarray).

    Returns
    -------
    float or np.ndarray
        Normalized angle(s) in range [-180°, 180°]. Returns scalar if input
        is scalar, array if input is array.

    Examples
    --------
    >>> normalize_angle_to_symmetric_range(210.0)
    -150.0
    >>> normalize_angle_to_symmetric_range(-210.0)
    150.0
    >>> normalize_angle_to_symmetric_range(np.array([0, 90, 210, -210]))
    array([  0.,  90., -150.,  150.])
    >>> normalize_angle_to_symmetric_range(np.array([180, -180, 360]))
    array([180., -180.,   0.])
    """
    angle_array = np.asarray(angle)
    normalized = angle_array % 360
    normalized = np.where(normalized > 180, normalized - 360, normalized)

    # Return scalar if input was scalar
    if np.isscalar(angle):
        return float(normalized)
    return normalized


def _angle_in_range(angle, min_angle, max_angle):
    """Check if angle is in range, accounting for wrap-around at ±180°.

    This function handles both normal ranges (where min_angle ≤ max_angle)
    and wrapped ranges that span the ±180° boundary (where min_angle > max_angle).

    Wrapped Range Logic
    -------------------
    When a range spans the ±180° boundary after normalization, the comparison
    logic changes:
    - Normal range [85°, 95°]: 85° ≤ angle ≤ 95°
    - Wrapped range [170°, -170°]: angle ≥ 170° OR angle ≤ -170°

    Example: User specifies range [170°, 190°]
    - After normalization: [170°, -170°] (min > max, wrapped)
    - Angle 175° matches: 175° ≥ 170° ✓
    - Angle -175° matches: -175° ≤ -170° ✓
    - Angle 0° does not match: 0° < 170° and 0° > -170° ✗

    Parameters
    ----------
    angle : float
        Angle to check (should be normalized to [-180°, 180°])
    min_angle : float
        Minimum angle of range (normalized to [-180°, 180°])
    max_angle : float
        Maximum angle of range (normalized to [-180°, 180°])

    Returns
    -------
    bool
        True if angle is in range, False otherwise

    Examples
    --------
    >>> _angle_in_range(90.0, 85.0, 95.0)  # Normal range
    True
    >>> _angle_in_range(175.0, 170.0, -170.0)  # Wrapped range
    True
    >>> _angle_in_range(-175.0, 170.0, -170.0)  # Wrapped range
    True
    >>> _angle_in_range(0.0, 170.0, -170.0)  # Outside wrapped range
    False
    """
    if min_angle <= max_angle:
        # Normal range (doesn't span ±180° boundary)
        return min_angle <= angle <= max_angle
    else:
        # Wrapped range (spans ±180° boundary)
        # Angle matches if it's >= min_angle OR <= max_angle
        return angle >= min_angle or angle <= max_angle


# Import core modules with fallback
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.data.xpcs_loader import XPCSDataLoader
    from homodyne.device import configure_optimal_device
    from homodyne.optimization import fit_mcmc_jax, fit_nlsq_jax

    HAS_CORE_MODULES = True
    HAS_XPCS_LOADER = True
except ImportError as e:
    HAS_CORE_MODULES = False
    HAS_XPCS_LOADER = False
    logger.error(f"Core modules not available: {e}")

    # Fallback for missing XPCSDataLoader
    class XPCSDataLoader:
        """Placeholder when XPCSDataLoader is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("XPCSDataLoader not available")


def dispatch_command(args) -> dict[str, Any]:
    """
    Dispatch command based on parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments

    Returns
    -------
    dict
        Command execution result with success status and details
    """
    logger.info("Dispatching homodyne analysis command")

    # Validate arguments
    if not validate_args(args):
        return {"success": False, "error": "Invalid command-line arguments"}

    if not HAS_CORE_MODULES:
        return {
            "success": False,
            "error": "Core modules not available. Please check installation.",
        }

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup file logging
        _setup_file_logging(args)

        # Configure logging based on args
        _configure_logging(args)

        # Configure device (CPU/GPU)
        device_config = _configure_device(args)

        # Load configuration
        config = _load_configuration(args)

        # Check if only simulated data plotting is requested (no experimental data needed)
        plot_exp = getattr(args, "plot_experimental_data", False)
        plot_sim = getattr(args, "plot_simulated_data", False)
        save_plots = getattr(args, "save_plots", False)

        # Simulated data plotting doesn't need experimental data or optimization
        if plot_sim and not plot_exp and not save_plots:
            logger.info(
                "Plotting simulated data only (skipping data loading and optimization)"
            )
            # Skip data loading for pure simulated data mode
            # Extract config dictionary from ConfigManager
            config_dict = (
                config.get_config() if hasattr(config, "get_config") else config
            )
            _handle_plotting(args, None, {}, config_dict)
            logger.info("Analysis completed successfully")
            return {
                "success": True,
                "result": None,
                "device_config": device_config,
                "output_dir": str(args.output_dir),
            }

        # Load data (needed for experimental plots and optimization)
        data = _load_data(args, config)

        # Plot experimental data only (no optimization needed)
        plot_only = plot_exp and not save_plots and not plot_sim

        if plot_only:
            # Skip optimization, just plot experimental data
            logger.info("Plotting experimental data only (skipping optimization)")
            result = None
        else:
            # Run optimization
            result = _run_optimization(args, config, data)

            # Save results
            _save_results(args, result, device_config)

        # Handle plotting options
        # Extract config dictionary from ConfigManager
        config_dict = config.get_config() if hasattr(config, "get_config") else config
        _handle_plotting(args, result, data, config_dict)

        logger.info("Analysis completed successfully")

        # Summary message
        log_dir = args.output_dir / "logs"
        log_files = list(log_dir.glob("homodyne_analysis_*.log"))
        if log_files:
            logger.info(f"Analysis log saved to: {log_files[-1]}")

        return {
            "success": True,
            "result": result,
            "device_config": device_config,
            "output_dir": str(args.output_dir),
        }

    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return {"success": False, "error": str(e)}


def _setup_file_logging(args) -> None:
    """
    Setup file logging to save analysis logs.

    Creates a log file in the output directory to capture the full analysis log.
    """
    import logging
    from datetime import datetime

    # Create logs subdirectory
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"homodyne_analysis_{timestamp}.log"

    # Add file handler to root homodyne logger
    root_logger = logging.getLogger("homodyne")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file

    # Add handler
    root_logger.addHandler(file_handler)

    logger.info(f"Log file created: {log_file}")
    return log_file


def _configure_logging(args) -> None:
    """Configure logging based on CLI arguments."""
    import logging

    # Set root logger to DEBUG so file handler captures everything
    root_logger = logging.getLogger("homodyne")
    root_logger.setLevel(logging.DEBUG)

    # Configure console handler level based on args
    for handler in root_logger.handlers:
        # Only adjust console (StreamHandler), not file handlers
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            if args.quiet:
                handler.setLevel(logging.ERROR)
            elif args.verbose:
                handler.setLevel(logging.DEBUG)
                logger.debug("Verbose logging enabled")
            else:
                handler.setLevel(logging.INFO)


def _configure_device(args) -> dict[str, Any]:
    """Configure optimal device based on CLI arguments."""
    import os

    logger.info("Configuring computational device...")

    # Disable JAX GPU autotuning to prevent hanging
    # This is a known issue with JAX gemm_fusion_autotuner
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
        " --xla_gpu_deterministic_ops=true"
    ).strip()
    logger.debug("Disabled JAX GPU autotuning to prevent hangs")

    device_config = configure_optimal_device(
        prefer_gpu=not args.force_cpu,
        gpu_memory_fraction=args.gpu_memory_fraction,
        force_cpu=args.force_cpu,
    )

    if device_config["configuration_successful"]:
        device_type = device_config["device_type"]
        logger.info(f"✓ Device configured: {device_type.upper()}")

        if device_type == "gpu":
            logger.info(f"GPU memory fraction: {args.gpu_memory_fraction:.0%}")
    else:
        logger.warning("Device configuration failed, using defaults")

    return device_config


def _load_configuration(args) -> ConfigManager:
    """Load configuration from file or create default."""
    logger.info(f"Loading configuration from: {args.config}")

    try:
        # Try to load from file
        if args.config.exists():
            config = ConfigManager(str(args.config))
            logger.info(f"✓ Configuration loaded: {args.config}")
        else:
            # Create default configuration
            logger.info("Configuration file not found, using defaults")
            config = ConfigManager(config_override=_get_default_config(args))

        # Apply CLI overrides
        _apply_cli_overrides(config, args)

        return config

    except Exception as e:
        logger.warning(f"Configuration loading failed: {e}, using defaults")
        return ConfigManager(config_override=_get_default_config(args))


def _get_default_config(args) -> dict[str, Any]:
    """Create default configuration from CLI arguments."""
    # Determine analysis mode
    if args.static_mode:
        analysis_mode = "static_isotropic"
    elif args.laminar_flow:
        analysis_mode = "laminar_flow"
    else:
        analysis_mode = "auto_detect"

    config = {
        "metadata": {
            "config_version": "2.1",
            "description": "CLI-generated configuration",
        },
        "analysis_mode": analysis_mode,
        "experimental_data": {
            "file_path": str(args.data_file) if args.data_file else None
        },
        "optimization": {
            "method": args.method,
            "lsq": {
                "max_iterations": args.max_iterations,
                "tolerance": args.tolerance,
            },
            "mcmc": {
                "n_samples": args.n_samples,
                "n_warmup": args.n_warmup,
                "n_chains": args.n_chains,
            },
        },
        "hardware": {
            "force_cpu": args.force_cpu,
            "gpu_memory_fraction": args.gpu_memory_fraction,
        },
        "output": {
            "formats": [args.output_format],
            "save_plots": args.save_plots,
            "output_dir": str(args.output_dir),
        },
    }

    return config


def _apply_cli_overrides(config: ConfigManager, args) -> None:
    """Apply CLI argument overrides to configuration."""
    if not hasattr(config, "config") or not config.config:
        return

    # Override data file if provided
    if args.data_file:
        config.config.setdefault("experimental_data", {})
        config.config["experimental_data"]["file_path"] = str(args.data_file)

    # Override analysis mode if specified
    if args.static_mode:
        config.config["analysis_mode"] = "static_isotropic"
    elif args.laminar_flow:
        config.config["analysis_mode"] = "laminar_flow"

    # Override optimization parameters
    if "optimization" not in config.config:
        config.config["optimization"] = {}

    config.config["optimization"]["method"] = args.method

    # Override hardware settings
    if "hardware" not in config.config:
        config.config["hardware"] = {}

    config.config["hardware"]["force_cpu"] = args.force_cpu
    config.config["hardware"]["gpu_memory_fraction"] = args.gpu_memory_fraction


def _load_data(args, config: ConfigManager) -> dict[str, Any]:
    """
    Load experimental data using XPCSDataLoader.

    Uses XPCSDataLoader which properly handles the config format
    (data_folder_path + data_file_name) internally.
    """
    logger.info("Loading experimental data...")

    # Check if XPCSDataLoader is available
    if not HAS_XPCS_LOADER:
        raise RuntimeError(
            "XPCSDataLoader not available. "
            "Please ensure homodyne.data module is properly installed"
        )

    try:
        # Determine loading strategy
        if args.data_file:
            # CLI override: Create minimal config for XPCSDataLoader
            # Convert to absolute path to handle relative paths properly
            data_file_path = Path(args.data_file).resolve()

            # Handle edge case: if only filename provided (no directory)
            parent_dir = data_file_path.parent
            if parent_dir == Path.cwd():
                # File in current directory - be explicit
                logger.debug(
                    f"Using current directory for data file: {data_file_path.name}"
                )

            temp_config = {
                "experimental_data": {
                    "data_folder_path": str(parent_dir),
                    "data_file_name": data_file_path.name,
                },
                "analyzer_parameters": config.config.get(
                    "analyzer_parameters",
                    {"dt": 0.1, "start_frame": 1, "end_frame": -1},
                ),
            }
            logger.info(f"Loading data from CLI override: {data_file_path}")
            loader = XPCSDataLoader(config_dict=temp_config)
        else:
            # Use full config - XPCSDataLoader handles data_folder_path + data_file_name
            if not hasattr(config, "config") or not config.config:
                raise ValueError("No configuration loaded")

            exp_data = config.config.get("experimental_data", {})
            if not exp_data.get("data_folder_path") and not exp_data.get("file_path"):
                raise ValueError(
                    "No data file specified in configuration.\n"
                    "Config must have either:\n"
                    "  experimental_data:\n"
                    "    data_folder_path: ./path/to/data/\n"
                    "    data_file_name: experiment.hdf\n"
                    "Or:\n"
                    "  experimental_data:\n"
                    "    file_path: ./path/to/data/experiment.hdf\n"
                    "Or use: --data-file path/to/data.hdf"
                )

            logger.info("Loading data from configuration")
            loader = XPCSDataLoader(config_dict=config.config)

        # Load data using the configured loader
        data = loader.load_experimental_data()

        # Get data size for reporting
        data_size = 0
        if "c2_exp" in data:
            c2_exp = data["c2_exp"]
            data_size = c2_exp.size if hasattr(c2_exp, "size") else len(c2_exp)

        logger.info(f"✓ Data loaded successfully: {data_size:,} data points")
        return data

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise RuntimeError(f"Data file not found: {e}") from e
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise RuntimeError(f"Failed to load experimental data: {e}") from e


def _apply_angle_filtering_for_optimization(
    data: dict[str, Any], config: ConfigManager
) -> dict[str, Any]:
    """
    Apply angle filtering to data before optimization.

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
            f"Found {len(angles_too_large)} angle(s) with |φ| > 360°: {angles_too_large}. "
            f"This may indicate data loading issues, unit confusion (radians vs degrees), "
            f"or instrument malfunction. Angles will be normalized to [-180°, 180°] range."
        )

    # Normalize phi angles to [-180°, 180°] range (flow direction at 0°)
    original_phi_angles = phi_angles.copy()
    phi_angles = normalize_angle_to_symmetric_range(phi_angles)
    logger.info("Normalized phi angles to [-180°, 180°] range (flow direction at 0°)")
    logger.debug(f"Original angles: {original_phi_angles}")
    logger.debug(f"Normalized angles: {phi_angles}")

    # Get config dict (handle both ConfigManager and dict types)
    config_dict = config.get_config() if hasattr(config, "get_config") else config

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
            "Phi filtering enabled but no target_ranges specified, using all angles"
        )
        # Return data with normalized angles
        normalized_data = data.copy()
        normalized_data["phi_angles_list"] = phi_angles
        return normalized_data

    # Normalize target_ranges to [-180°, 180°] for consistency
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
            }
        )
        logger.debug(
            f"Normalized range [{min_angle:.1f}°, {max_angle:.1f}°] → "
            f"[{normalized_min:.1f}°, {normalized_max:.1f}°]"
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
            f"is correct (check for typos in min_angle/max_angle values)."
        )

    # Create modified config with normalized target_ranges
    modified_config = config_dict.copy()
    modified_phi_filtering = phi_filtering_config.copy()
    modified_phi_filtering["target_ranges"] = target_ranges
    modified_config["phi_filtering"] = modified_phi_filtering

    # Call shared filtering function with performance timing
    start_time = time.perf_counter()
    filtered_indices, filtered_phi_angles, filtered_c2_exp = _apply_angle_filtering(
        phi_angles, c2_exp, modified_config
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
            f"All {len(phi_angles)} angles matched filter criteria, no reduction"
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
        f"from {len(phi_angles)} total angles"
    )
    logger.info(f"Selected angles: {filtered_phi_angles}")

    return filtered_data


def _run_optimization(args, config: ConfigManager, data: dict[str, Any]) -> Any:
    """Run the specified optimization method."""
    method = args.method
    logger.info(f"Running {method.upper()} optimization...")

    start_time = time.perf_counter()

    # Apply angle filtering before optimization (if configured)
    filtered_data = _apply_angle_filtering_for_optimization(data, config)

    try:
        if method == "nlsq":
            result = fit_nlsq_jax(filtered_data, config)
        elif method == "mcmc":
            # Convert data format for MCMC if needed
            mcmc_data = filtered_data["c2_exp"]
            result = fit_mcmc_jax(
                mcmc_data,
                t1=filtered_data.get("t1"),
                t2=filtered_data.get("t2"),
                phi=filtered_data.get("phi_angles_list"),
                q=(
                    filtered_data.get("wavevector_q_list", [1.0])[0]
                    if filtered_data.get("wavevector_q_list")
                    else 1.0
                ),
                L=2000000.0,  # Default: 200 µm stator-rotor gap (typical rheology-XPCS)
                analysis_mode=(
                    config.config.get("analysis_mode", "static_isotropic")
                    if hasattr(config, "config")
                    else "static_isotropic"
                ),
                n_samples=args.n_samples,
                n_warmup=args.n_warmup,
                n_chains=args.n_chains,
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        optimization_time = time.perf_counter() - start_time
        logger.info(
            f"✓ {method.upper()} optimization completed in {optimization_time:.3f}s"
        )

        return result

    except Exception as e:
        optimization_time = time.perf_counter() - start_time
        logger.error(
            f"{method.upper()} optimization failed after {optimization_time:.3f}s: {e}"
        )
        raise


def _save_results(args, result: Any, device_config: dict[str, Any]) -> None:
    """Save optimization results to output directory."""
    logger.info(f"Saving results to: {args.output_dir}")

    import json

    import numpy as np
    import yaml

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
        results_summary["parameters"] = result.parameters
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
            with open(output_file, "w") as f:
                yaml.dump(results_summary, f, default_flow_style=False)
        elif args.output_format == "json":
            with open(output_file, "w") as f:
                json.dump(results_summary, f, indent=2, default=_json_serializer)
        elif args.output_format == "npz":
            # Save numpy arrays
            arrays_to_save = {
                "results_summary": np.array([results_summary], dtype=object)
            }
            if hasattr(result, "samples_params") and result.samples_params is not None:
                arrays_to_save["samples_params"] = result.samples_params
            np.savez(output_file, **arrays_to_save)

        logger.info(f"✓ Results saved: {output_file}")

    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


def _handle_plotting(
    args, result: Any, data: dict[str, Any], config: dict[str, Any] = None
) -> None:
    """
    Handle plotting options for experimental and simulated data.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    result : Any
        Optimization result
    data : dict
        Experimental data dictionary
    config : dict, optional
        Configuration dictionary (required for simulated data plotting)
    """
    # Check if any plotting was requested
    plot_exp = getattr(args, "plot_experimental_data", False)
    plot_sim = getattr(args, "plot_simulated_data", False)
    save_plots = getattr(args, "save_plots", False)

    if not (save_plots or plot_exp or plot_sim):
        return

    # Check for plotting dependencies
    try:
        import matplotlib.pyplot  # noqa: F401 - Import check only
    except ImportError:
        logger.warning(
            "Plotting requested but matplotlib not installed. "
            "Install with: pip install matplotlib"
        )
        return

    logger.info("Generating plots...")

    # Create plots directory
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot experimental data if requested
    if plot_exp:
        try:
            # Add config to data dict for angle filtering (if available)
            data_with_config = data.copy()
            if config:
                data_with_config["config"] = config
            _plot_experimental_data(data_with_config, plots_dir)
            logger.info(f"✓ Experimental data plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate experimental data plots: {e}")

    # Plot simulated data if requested
    if plot_sim:
        try:
            if config is None:
                logger.error("Configuration required for simulated data plotting")
                return

            # Get contrast, offset, and phi_angles from args
            contrast = getattr(args, "contrast", 0.5)  # Match working version default
            offset = getattr(args, "offset", 1.0)
            phi_angles_str = getattr(args, "phi_angles", None)

            _plot_simulated_data(
                config, contrast, offset, phi_angles_str, plots_dir, data
            )
            logger.info(f"✓ Simulated data plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate simulated data plots: {e}")
            logger.debug("Simulated data plotting error:", exc_info=True)

    # Plot fit comparison if save_plots is enabled
    if save_plots:
        try:
            _plot_fit_comparison(result, data, plots_dir)
            logger.info(f"✓ Fit comparison plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate fit comparison plots: {e}")


def _apply_angle_filtering(
    phi_angles: np.ndarray, c2_exp: np.ndarray, config: dict[str, Any]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Core angle filtering logic shared by optimization and plotting.

    Filters phi angles and corresponding C2 data based on target_ranges
    specified in configuration. Uses OR logic across ranges: an angle is
    selected if it falls within ANY of the specified ranges.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles in degrees, shape (n_phi,)
    c2_exp : np.ndarray
        Experimental correlation data, shape (n_phi, n_t1, n_t2)
    config : dict
        Configuration dictionary with phi_filtering section

    Returns
    -------
    filtered_indices : list of int
        Indices of angles that matched target ranges
    filtered_phi_angles : np.ndarray
        Filtered phi angles array, shape (n_matched,)
    filtered_c2_exp : np.ndarray
        Filtered C2 data array, shape (n_matched, n_t1, n_t2)

    Notes
    -----
    - Returns all angles (unfiltered) if phi_filtering.enabled is False
    - Returns all angles with warning if no target_ranges specified
    - Returns all angles with warning if no angles match target ranges
    - Angle matching uses wrap-aware range checking (handles ±180° boundary)
    - Normal range [85°, 95°]: 85° ≤ angle ≤ 95°
    - Wrapped range [170°, -170°]: angle ≥ 170° OR angle ≤ -170°
    - Angles matching multiple ranges are only included once
    """
    # Get phi_filtering configuration
    phi_filtering_config = config.get("phi_filtering", {})

    if not phi_filtering_config.get("enabled", False):
        # Filtering disabled - return all angles
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Get target ranges
    target_ranges = phi_filtering_config.get("target_ranges", [])
    if not target_ranges:
        # No ranges specified - return all angles with warning
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Filter angles based on target ranges (OR logic)
    # Uses wrap-aware range checking to handle ranges spanning ±180° boundary
    filtered_indices = []
    for i, angle in enumerate(phi_angles):
        for range_spec in target_ranges:
            min_angle = range_spec.get("min_angle", -180.0)
            max_angle = range_spec.get("max_angle", 180.0)
            if _angle_in_range(angle, min_angle, max_angle):
                filtered_indices.append(i)
                break  # Angle matches this range, no need to check other ranges

    if not filtered_indices:
        # No matches - return all angles with warning
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Apply filtering
    # Convert list to numpy array for JAX compatibility
    # (JAX arrays don't accept Python list indexing, see https://github.com/jax-ml/jax/issues/4564)
    filtered_indices_array = np.array(filtered_indices)
    filtered_phi_angles = phi_angles[filtered_indices_array]
    filtered_c2_exp = c2_exp[filtered_indices_array]

    return filtered_indices, filtered_phi_angles, filtered_c2_exp


def _apply_angle_filtering_for_plot(
    phi_angles: np.ndarray, c2_exp: np.ndarray, data: dict[str, Any]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    Apply angle filtering to select specific angles for plotting.

    This is a wrapper around _apply_angle_filtering() that extracts the
    configuration from the data dictionary and adds plot-specific logging.

    This filters the loaded data (which contains ALL angles) to show only
    the angles specified in phi_filtering configuration.

    Parameters
    ----------
    phi_angles : np.ndarray
        Array of phi angles in degrees
    c2_exp : np.ndarray
        Experimental correlation data
    data : dict
        Data dictionary containing 'config' key with phi_filtering settings

    Returns
    -------
    tuple
        (filtered_indices, filtered_phi_angles, filtered_c2_exp)

    Notes
    -----
    Uses shared _apply_angle_filtering() function for consistent filtering
    logic with optimization workflow.
    """
    # Check if filtering config is available in data dict
    config = data.get("config", None)
    if config is None:
        # No config available - plot all angles
        logger.debug("No config available for angle filtering, plotting all angles")
        return list(range(len(phi_angles))), phi_angles, c2_exp

    # Call shared filtering function
    filtered_indices, filtered_phi_angles, filtered_c2_exp = _apply_angle_filtering(
        phi_angles, c2_exp, config
    )

    # Add plot-specific logging
    phi_filtering_config = config.get("phi_filtering", {})

    if not phi_filtering_config.get("enabled", False):
        logger.debug("Phi filtering not enabled, plotting all angles")
    elif not phi_filtering_config.get("target_ranges", []):
        logger.warning(
            "Phi filtering enabled but no target_ranges specified, plotting all angles"
        )
    elif not filtered_indices or len(filtered_indices) == len(phi_angles):
        if len(filtered_indices) == 0:
            logger.warning("No angles matched target ranges, plotting all angles")
        # else: all angles matched, no special logging needed
    else:
        logger.info(
            f"Angle filtering applied: {len(filtered_indices)} angles selected "
            f"from {len(phi_angles)} total angles"
        )

    return filtered_indices, filtered_phi_angles, filtered_c2_exp


def _plot_experimental_data(data: dict[str, Any], plots_dir) -> None:
    """Generate validation plots of experimental data."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Basic experimental data visualization
    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        logger.warning("No experimental data to plot")
        return

    # Get time arrays if available for proper axis labels
    t1 = data.get("t1", None)
    t2 = data.get("t2", None)

    # Extract time extent for imshow if time arrays are available
    if t1 is not None and t2 is not None:
        t_min = float(np.min(t1))
        t_max = float(np.max(t1))
        extent = [t_min, t_max, t_min, t_max]
        xlabel = "t₂ (s)"
        ylabel = "t₁ (s)"
        logger.debug(f"Using time extent: [{t_min:.3f}, {t_max:.3f}] seconds")
    else:
        extent = None
        xlabel = "t₂ Index"
        ylabel = "t₁ Index"
        logger.debug("Time arrays not available, using frame indices")

    # Get phi angles array from data
    phi_angles_list = data.get("phi_angles_list", None)
    if phi_angles_list is None:
        logger.warning("phi_angles_list not found in data, using indices")
        phi_angles_list = np.arange(c2_exp.shape[0])

    # Apply angle filtering for plotting if configured
    # This filters the loaded data to show only selected angles
    filtered_indices, filtered_phi_angles, filtered_c2_exp = (
        _apply_angle_filtering_for_plot(phi_angles_list, c2_exp, data)
    )

    # Use filtered data for plotting
    phi_angles_list = filtered_phi_angles
    c2_exp = filtered_c2_exp

    logger.info(
        f"Plotting {len(filtered_indices)} angles after filtering: {filtered_phi_angles}"
    )

    # Handle different data shapes
    if c2_exp.ndim == 3:
        # Data shape: (n_phi, n_t1, n_t2)
        # Plot up to 3 phi angles (matching working version)
        from matplotlib import gridspec

        n_angles = c2_exp.shape[0]
        n_plot_angles = min(3, n_angles)  # Show up to 3 angles

        # Create figure with 2-column GridSpec layout (heatmap + statistics)
        fig = plt.figure(figsize=(10, 4 * n_plot_angles))
        gs = gridspec.GridSpec(n_plot_angles, 2, hspace=0.3, wspace=0.3)

        for i in range(n_plot_angles):
            # Plot the first n_plot_angles angles directly
            # (phi_angles_list is already filtered, so just use first N angles)
            angle_idx = i

            # Get actual phi angle value
            phi_deg = (
                phi_angles_list[angle_idx] if len(phi_angles_list) > angle_idx else 0.0
            )
            angle_data = c2_exp[angle_idx]

            # Calculate statistics
            mean_val = np.mean(angle_data)
            std_val = np.std(angle_data)
            min_val = np.min(angle_data)
            max_val = np.max(angle_data)
            diagonal = np.diag(angle_data)
            diag_mean = np.mean(diagonal)

            # Calculate contrast with proper handling of zero/near-zero min_val
            if abs(min_val) < 1e-10:  # Near zero
                if abs(max_val) < 1e-10:  # Both near zero
                    contrast = 0.0
                else:
                    contrast = float("inf")  # Infinite contrast
            else:
                contrast = (max_val - min_val) / min_val

            # Format contrast value appropriately
            if contrast == float("inf"):
                contrast_str = "∞"
            elif contrast == 0.0:
                contrast_str = "0.000"
            else:
                contrast_str = f"{contrast:.3f}"

            # 1. C2 heatmap (left panel)
            ax1 = fig.add_subplot(gs[i, 0])
            im = ax1.imshow(
                angle_data,
                aspect="equal",
                cmap="viridis",
                origin="lower",
                extent=extent,
            )
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(ylabel)
            ax1.set_title(f"$g_2(t_1, t_2)$ at φ={phi_deg:.1f}°")
            plt.colorbar(im, ax=ax1, label="C₂", shrink=0.8)

            # 2. Statistics panel (right panel)
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.axis("off")

            stats_text = f"""Data Statistics (φ={phi_deg:.1f}°):

Shape: {angle_data.shape[0]} × {angle_data.shape[1]}

g₂ Values:
Mean: {mean_val:.4f}
Std:  {std_val:.4f}
Min:  {min_val:.4f}
Max:  {max_val:.4f}

Diagonal mean: {diag_mean:.4f}
Contrast: {contrast_str}

Validation:
{"✓" if 1 < mean_val < 2 else "✗"} Mean around 1.0
{"✓" if diag_mean > mean_val else "✗"} Diagonal enhanced
{"✓" if contrast > 0.001 else "✗"} Sufficient contrast"""

            ax2.text(
                0.05,
                0.95,
                stats_text,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
            )

        # Overall title
        plt.suptitle(
            "Experimental Data Validation: Unknown Sample",
            fontsize=16,
            fontweight="bold",
        )

        # Save with bbox_inches="tight" to handle layout automatically
        # (no plt.tight_layout() needed - causes warning with suptitle)
        plt.savefig(
            plots_dir / "experimental_data_phi_slices.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

        # Plot diagonal (t1=t2) for all phi angles
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get time values for x-axis if available
        if t1 is not None:
            time_diagonal = np.diag(
                t1
            )  # Extract diagonal of t1 (which equals t2 on diagonal)
        else:
            time_diagonal = np.arange(c2_exp.shape[-1])

        for idx in range(min(10, c2_exp.shape[0])):
            diagonal = np.diag(c2_exp[idx])
            phi_deg = phi_angles_list[idx] if len(phi_angles_list) > idx else idx
            ax.plot(time_diagonal, diagonal, label=f"φ={phi_deg:.1f}°", alpha=0.7)
        ax.set_xlabel("Time (s)" if t1 is not None else "Time Index")
        ax.set_ylabel("C₂(t, t)")
        ax.set_title("C₂ Diagonal (t₁=t₂) for Different φ Angles")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            plots_dir / "experimental_data_diagonal.png", dpi=150, bbox_inches="tight"
        )
        plt.close()

    elif c2_exp.ndim == 2:
        # 2D data: single correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            c2_exp, aspect="equal", cmap="viridis", origin="lower", extent=extent
        )
        plt.colorbar(im, ax=ax, label="C₂(t₁,t₂)", shrink=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Experimental C₂ Data")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
        plt.close()

    elif c2_exp.ndim == 1:
        # 1D data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(c2_exp, marker="o", linestyle="-", alpha=0.7)
        ax.set_xlabel("Data Point Index")
        ax.set_ylabel("C₂")
        ax.set_title("Experimental C₂ Data")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
        plt.close()

    else:
        logger.warning(f"Unsupported data dimensionality: {c2_exp.ndim}D")
        return

    logger.debug(f"Plotted experimental data with shape {c2_exp.shape}")


def _plot_simulated_data(
    config: dict[str, Any],
    contrast: float,
    offset: float,
    phi_angles_str: str | None,
    plots_dir,
    data: dict[str, Any] | None = None,
) -> None:
    """Generate plots of simulated/theoretical data."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from homodyne.core.models import CombinedModel

    # BUGFIX: Force contrast to 0.5 to match working version
    # The default was incorrectly set to 0.3 somewhere upstream
    if contrast < 0.4:  # If it's the old default 0.3, override it
        logger.debug(f"Overriding contrast={contrast} → 0.5 (matching working version)")
        contrast = 0.5

    logger.info(
        f"Generating simulated data plots (contrast={contrast:.3f}, offset={offset:.3f})"
    )

    # Determine analysis mode
    analysis_mode = config.get("analysis_mode", "static_isotropic")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Create model
    model = CombinedModel(analysis_mode)

    # Get parameters from configuration
    # Read from top-level initial_parameters (not nested in optimization)
    initial_params_config = config.get("initial_parameters", {})
    param_names = initial_params_config.get("parameter_names", [])
    param_values = initial_params_config.get("values", [])

    # Create dict mapping parameter names to values
    params_dict = dict(zip(param_names, param_values)) if param_names and param_values else {}

    if analysis_mode.startswith("static"):
        # Static mode: 3 parameters
        params = jnp.array(
            [
                params_dict.get("D0", 100.0),
                params_dict.get("alpha", -0.5),
                params_dict.get("D_offset", 0.0),
            ]
        )
    else:
        # Laminar flow: 7 parameters
        # Use correct parameter names matching config (gamma_dot_0, phi_0 with underscores)
        params = jnp.array(
            [
                params_dict.get("D0", 100.0),
                params_dict.get("alpha", -0.5),
                params_dict.get("D_offset", 0.0),
                params_dict.get("gamma_dot_0", 0.01),
                params_dict.get("beta", 0.5),
                params_dict.get("gamma_dot_offset", 0.0),
                params_dict.get("phi_0", 0.0),
            ]
        )

    logger.debug(
        f"Using parameters: {dict(zip(model.parameter_names, params, strict=False))}"
    )

    # Parse phi angles
    if phi_angles_str:
        phi_degrees = np.array([float(x.strip()) for x in phi_angles_str.split(",")])
        phi = phi_degrees  # Keep in degrees (physics code expects degrees)
    else:
        # Default: 8 evenly spaced angles from 0 to 180 degrees
        phi_degrees = np.linspace(0, 180, 8)
        phi = phi_degrees  # Keep in degrees (physics code expects degrees)

    logger.debug(f"Using {len(phi)} phi angles: {phi_degrees}")

    # Generate time arrays matching configuration specification
    # CRITICAL: Simulated data must be independent of experimental data
    analyzer_params = config.get("analyzer_parameters", {})
    dt = analyzer_params.get("dt", 0.1)
    start_frame = analyzer_params.get("start_frame", 1)
    end_frame = analyzer_params.get("end_frame", 8000)

    # Calculate number of time points (inclusive frame counting)
    # This matches the data loader convention: n = end - start + 1
    n_time_points = end_frame - start_frame + 1

    # Generate time array: t[i] = dt * i for i = 0, 1, ..., n-1
    # For linspace(0, T, N): T = dt * (N - 1) to ensure t[i] = dt * i
    time_max = dt * (n_time_points - 1)
    t_vals = jnp.linspace(0, time_max, n_time_points)
    t1_grid, t2_grid = jnp.meshgrid(t_vals, t_vals, indexing="ij")

    logger.debug(
        f"Simulated data time grid: dt={dt}, start_frame={start_frame}, end_frame={end_frame}"
    )
    logger.debug(f"Time range: [0, {time_max:.2f}] seconds with {n_time_points} points")
    logger.debug(f"Time spacing verification: t[1]-t[0]={float(t_vals[1] - t_vals[0]):.6f} (should equal dt={dt})")

    # Get wavevector_q and stator_rotor_gap from correct config sections
    scattering_config = analyzer_params.get("scattering", {})
    geometry_config = analyzer_params.get("geometry", {})

    q = scattering_config.get("wavevector_q", 0.0054)  # Wave vector in Å⁻¹
    L_angstroms = geometry_config.get("stator_rotor_gap", 2000000)  # Gap in Angstroms

    # Convert to microns for display (1 μm = 10,000 Å)
    L_microns = L_angstroms / 10000.0

    logger.info(
        f"Generating theoretical C₂ with q={q:.6f} Å⁻¹, L={L_microns:.1f} μm ({L_angstroms:.0f} Å)"
    )
    logger.debug(
        f"Physics parameters: q={q}, L={L_angstroms} (Angstroms - used by physics code)"
    )

    # Generate simulated C₂ for each phi angle
    c2_simulated = []

    for _i, phi_val in enumerate(phi):
        phi_array = jnp.array([phi_val])

        logger.debug(f"Computing C₂ for φ={phi_val}° (phi_array={phi_array})")

        # Compute g2 for this phi angle (L_angstroms: physics code expects Angstroms)
        # CRITICAL: Pass dt explicitly to ensure correct physics calculations
        c2_phi = model.compute_g2(
            params,
            t1_grid,
            t2_grid,
            phi_array,
            q,
            L_angstroms,
            contrast,
            offset,
            dt,  # Pass dt from config for accurate physics
        )

        # Extract the 2D array (remove phi dimension)
        c2_result = np.array(c2_phi[0])
        logger.debug(f"  C₂ shape: {c2_result.shape}, range: [{c2_result.min():.4f}, {c2_result.max():.4f}]")
        c2_simulated.append(c2_result)

    c2_simulated = np.array(c2_simulated)  # Shape: (n_phi, n_t, n_t)

    logger.info(f"Generated simulated C₂ with shape: {c2_simulated.shape}")

    # Compute global color scale across all angles for consistent visualization
    # CRITICAL: Use same vmin/vmax for all subplots to make them comparable
    vmin = c2_simulated.min()
    vmax = c2_simulated.max()
    logger.debug(f"Global color scale: vmin={vmin:.6f}, vmax={vmax:.6f}")

    # Plot 1: C₂ heatmaps for first 4 phi angles (2x2 grid)
    n_phi_to_plot = min(4, len(phi))
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx in range(n_phi_to_plot):
        im = axes[idx].imshow(
            c2_simulated[idx],
            extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]],
            aspect="auto",
            cmap="viridis",
            origin="lower",
            vmin=vmin,  # Use global scale
            vmax=vmax,  # Use global scale
        )
        axes[idx].set_xlabel("t₂ (s)")
        axes[idx].set_ylabel("t₁ (s)")
        axes[idx].set_title(f"Simulated C₂ at φ={phi_degrees[idx]:.1f}°")
        plt.colorbar(im, ax=axes[idx], label="C₂")

    # Hide unused subplots
    for idx in range(n_phi_to_plot, 4):
        axes[idx].axis("off")

    plt.suptitle(
        f"Theoretical C₂ Heatmaps\n(contrast={contrast:.3f}, offset={offset:.3f}, mode={analysis_mode})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(plots_dir / "simulated_data_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("✓ Generated C₂ heatmap plot: simulated_data_heatmap.png")

    # Plot 2: Diagonal (t1=t2) for all phi angles
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in range(min(10, len(phi))):
        diagonal = np.diag(c2_simulated[idx])
        ax.plot(
            t_vals, diagonal, label=f"φ={phi_degrees[idx]:.1f}°", alpha=0.7, linewidth=2
        )

    ax.set_xlabel("Time t (s)", fontsize=12)
    ax.set_ylabel("C₂(t, t)", fontsize=12)
    ax.set_title(
        f"Simulated C₂ Along Diagonal (t₁=t₂)\n(contrast={contrast:.3f}, offset={offset:.3f}, mode={analysis_mode})",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "simulated_data_diagonal.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("✓ Generated diagonal plot: simulated_data_diagonal.png")

    logger.info("✓ Simulated data plots generated successfully")


def _plot_fit_comparison(result: Any, data: dict[str, Any], plots_dir) -> None:
    """Generate comparison plots between fit and experimental data."""
    import matplotlib.pyplot as plt

    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot experimental data
    if c2_exp.ndim == 1:
        axes[0].plot(c2_exp, marker="o", linestyle="-", alpha=0.7, label="Experimental")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("C₂")
    else:
        im0 = axes[0].imshow(c2_exp, aspect="auto", cmap="viridis")
        plt.colorbar(im0, ax=axes[0], label="C₂")
        axes[0].set_xlabel("t₂ Index")
        axes[0].set_ylabel("φ Index")
    axes[0].set_title("Experimental Data")
    axes[0].grid(True, alpha=0.3)

    # Plot fit results
    axes[1].text(
        0.5,
        0.5,
        "Fit visualization\nrequires full\nplotting backend",
        ha="center",
        va="center",
        fontsize=14,
    )
    axes[1].set_title("Fit Results")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "fit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Generated basic fit comparison plot")


def _json_serializer(obj):
    """JSON serializer for numpy arrays and other objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return str(obj)
