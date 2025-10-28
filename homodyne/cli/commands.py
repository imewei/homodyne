"""Command Dispatcher for Homodyne v2 CLI
======================================

Handles command execution and coordination between CLI arguments,
configuration, and optimization methods.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp

# Set matplotlib backend for HPC headless support (must be before pyplot import)
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Check if Datashader backend is available (actual import done lazily)
try:
    import homodyne.viz.datashader_backend  # noqa: F401

    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False
    # Will log warning later when plotting is attempted

from homodyne.cli.args_parser import validate_args  # noqa: E402
from homodyne.config.types import (  # noqa: E402
    LAMINAR_FLOW_PARAM_NAMES,
    SCALING_PARAM_NAMES,
    STATIC_PARAM_NAMES,
)
from homodyne.core.jax_backend import compute_g2_scaled  # noqa: E402
from homodyne.utils.logging import get_logger  # noqa: E402

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
    """Dispatch command based on parsed CLI arguments.

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
                "Plotting simulated data only (skipping data loading and optimization)",
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

            # Save results (with data and config for NLSQ comprehensive saving)
            _save_results(args, result, device_config, data, config)

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
    """Setup file logging to save analysis logs.

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
            handler,
            logging.FileHandler,
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


def _check_deprecated_config(config: "ConfigManager") -> None:
    """Check for deprecated configuration sections and warn user.

    Warns about:
    - performance.subsampling (removed in v3.0)
    - optimization_performance.time_subsampling (deprecated in v2.1)

    Parameters
    ----------
    config : ConfigManager
        Configuration object to check
    """
    warnings_issued = []

    # Check for deprecated performance.subsampling
    if "performance" in config.config:
        perf = config.config["performance"]
        if "subsampling" in perf:
            warnings_issued.append(
                "⚠️  DEPRECATED CONFIG: 'performance.subsampling'\n"
                "   This section is no longer used (removed in v3.0).\n"
                "   NLSQ now handles large datasets automatically.\n"
                "   Please remove this section from your configuration file."
            )

    # Check for old optimization_performance path
    if "optimization_performance" in config.config:
        old_perf = config.config["optimization_performance"]
        if "time_subsampling" in old_perf:
            warnings_issued.append(
                "⚠️  DEPRECATED CONFIG: 'optimization_performance.time_subsampling'\n"
                "   This path was deprecated in v2.1 and removed in v3.0.\n"
                "   Please remove this section from your configuration file."
            )

    # Issue all warnings at once for visibility
    if warnings_issued:
        logger.warning(
            "\n" + "="*70 + "\n"
            "DEPRECATED CONFIGURATION DETECTED\n"
            + "="*70 + "\n"
            + "\n\n".join(warnings_issued) + "\n"
            + "="*70 + "\n"
            "Migration: NLSQ v3.0+ uses native large dataset handling.\n"
            "Simply remove the deprecated sections - no replacement needed.\n"
            "See: https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html\n"
            + "="*70
        )


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

        # Check for deprecated configuration sections
        _check_deprecated_config(config)

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
            "file_path": str(args.data_file) if args.data_file else None,
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
    """Apply CLI argument overrides to configuration.

    Implements precedence: CLI args > Config file > Code defaults
    For MCMC parameters, config uses 'num_*' prefix, args use 'n_*' prefix
    """
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

    # Load MCMC parameters from config if not provided via CLI
    # Config uses 'num_samples' etc., args use 'n_samples' etc.
    mcmc_config = config.config.get("optimization", {}).get("mcmc", {})

    # Code defaults from homodyne/optimization/mcmc.py:_get_mcmc_config
    if args.n_samples is None:
        args.n_samples = mcmc_config.get("num_samples", 1000)
    if args.n_warmup is None:
        args.n_warmup = mcmc_config.get("num_warmup", 500)
    if args.n_chains is None:
        args.n_chains = mcmc_config.get("num_chains", 4)

    # Override hardware settings
    if "hardware" not in config.config:
        config.config["hardware"] = {}

    config.config["hardware"]["force_cpu"] = args.force_cpu
    config.config["hardware"]["gpu_memory_fraction"] = args.gpu_memory_fraction


def _load_data(args, config: ConfigManager) -> dict[str, Any]:
    """Load experimental data using XPCSDataLoader.

    Uses XPCSDataLoader which properly handles the config format
    (data_folder_path + data_file_name) internally.
    """
    logger.info("Loading experimental data...")

    # Check if XPCSDataLoader is available
    if not HAS_XPCS_LOADER:
        raise RuntimeError(
            "XPCSDataLoader not available. "
            "Please ensure homodyne.data module is properly installed",
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
                    f"Using current directory for data file: {data_file_path.name}",
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
                    "Or use: --data-file path/to/data.hdf",
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
    data: dict[str, Any],
    config: ConfigManager,
) -> dict[str, Any]:
    """Apply angle filtering to data before optimization.

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
            f"or instrument malfunction. Angles will be normalized to [-180°, 180°] range.",
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
            "Phi filtering enabled but no target_ranges specified, using all angles",
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
            },
        )
        logger.debug(
            f"Normalized range [{min_angle:.1f}°, {max_angle:.1f}°] → "
            f"[{normalized_min:.1f}°, {normalized_max:.1f}°]",
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
            f"is correct (check for typos in min_angle/max_angle values).",
        )

    # Create modified config with normalized target_ranges
    modified_config = config_dict.copy()
    modified_phi_filtering = phi_filtering_config.copy()
    modified_phi_filtering["target_ranges"] = target_ranges
    modified_config["phi_filtering"] = modified_phi_filtering

    # Call shared filtering function with performance timing
    start_time = time.perf_counter()
    filtered_indices, filtered_phi_angles, filtered_c2_exp = _apply_angle_filtering(
        phi_angles,
        c2_exp,
        modified_config,
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
            f"All {len(phi_angles)} angles matched filter criteria, no reduction",
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
        f"from {len(phi_angles)} total angles",
    )
    logger.info(f"Selected angles: {filtered_phi_angles}")

    return filtered_data


def _run_optimization(args, config: ConfigManager, data: dict[str, Any]) -> Any:
    """Run the specified optimization method."""
    method = args.method

    # Normalize method aliases: mcmc → auto
    if method == "mcmc":
        method = "auto"
        logger.debug("Method 'mcmc' is an alias for 'auto' (automatic NUTS/CMC selection)")

    logger.info(f"Running {method.upper()} optimization...")

    start_time = time.perf_counter()

    # Apply angle filtering before optimization (if configured)
    filtered_data = _apply_angle_filtering_for_optimization(data, config)

    # NLSQ will handle large datasets natively via streaming optimization
    logger.debug("Using NLSQ native large dataset handling")

    try:
        if method == "nlsq":
            # Try NLSQ with automatic CPU fallback on GPU OOM
            try:
                result = fit_nlsq_jax(filtered_data, config)
            except RuntimeError as nlsq_error:
                # Check if this is a GPU OOM error
                error_msg = str(nlsq_error).lower()
                if "out_of_memory" in error_msg or "resource_exhausted" in error_msg:
                    from homodyne.device import is_gpu_active, switch_to_cpu

                    if is_gpu_active():
                        logger.warning(
                            "GPU out of memory detected during NLSQ optimization.",
                        )
                        logger.warning(
                            "Automatically falling back to CPU (slower but will complete)...",
                        )

                        # Switch to CPU
                        cpu_result = switch_to_cpu()
                        if cpu_result.get("success"):
                            logger.info(
                                f"Successfully switched to CPU with {cpu_result['num_threads']} threads",
                            )
                            logger.info(
                                "Retrying optimization on CPU (this may take 5-10x longer)...",
                            )

                            # Retry on CPU
                            result = fit_nlsq_jax(filtered_data, config)
                            logger.info(
                                "✓ CPU fallback successful! Optimization completed.",
                            )
                        else:
                            logger.error(
                                f"Failed to switch to CPU: {cpu_result.get('error')}",
                            )
                            raise nlsq_error
                    else:
                        # Already on CPU, can't fallback further
                        raise nlsq_error
                else:
                    # Not an OOM error, re-raise
                    raise nlsq_error
        elif method in ["auto", "nuts", "cmc"]:
            # MCMC/CMC methods: auto, nuts, cmc
            # Get CMC configuration from config file
            cmc_config = config.get_cmc_config()

            # Apply CLI overrides to CMC configuration
            if args.cmc_num_shards is not None:
                logger.info(
                    f"Overriding CMC num_shards from CLI: {args.cmc_num_shards}",
                )
                cmc_config.setdefault("sharding", {})["num_shards"] = (
                    args.cmc_num_shards
                )

            if args.cmc_backend is not None:
                logger.info(f"Overriding CMC backend from CLI: {args.cmc_backend}")
                cmc_config.setdefault("backend", {})["name"] = args.cmc_backend

            # Log CMC configuration being used
            logger.info(f"MCMC method: {method}")
            if method == "auto":
                logger.info(
                    "Automatic method selection: CMC if (samples >= 20) OR (memory > 40%), else NUTS",
                )
            elif method == "nuts":
                logger.info("Forcing standard NUTS MCMC (single-device execution)")
            elif method == "cmc":
                logger.info(
                    "Forcing Consensus Monte Carlo (distributed Bayesian inference)"
                )

            # Log key CMC parameters
            sharding = cmc_config.get("sharding", {})
            backend = cmc_config.get("backend", {})

            # Handle both old dict schema (backend={name: ...}) and new string schema (backend="jax")
            if isinstance(backend, str):
                # New schema: backend is computational backend string
                backend_str = backend
                backend_config = cmc_config.get("backend_config", {})
                parallel_backend = backend_config.get("name", "auto") if backend_config else "auto"
                backend_display = f"{backend_str}/{parallel_backend}"
            else:
                # Old schema: backend is dict with name for parallel execution
                backend_display = backend.get("name", "auto")

            logger.debug(
                f"CMC sharding: strategy={sharding.get('strategy', 'auto')}, "
                f"num_shards={sharding.get('num_shards', 'auto')}, "
                f"backend={backend_display}",
            )

            # Convert data format for MCMC if needed
            mcmc_data = filtered_data["c2_exp"]
            result = fit_mcmc_jax(
                mcmc_data,
                t1=filtered_data.get("t1"),
                t2=filtered_data.get("t2"),
                phi=filtered_data.get("phi_angles_list"),
                q=(
                    filtered_data.get("wavevector_q_list", [1.0])[0]
                    if filtered_data.get("wavevector_q_list") is not None
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
                method=method,  # Pass method to fit_mcmc_jax for auto/nuts/cmc selection
                cmc_config=cmc_config,  # Pass CMC configuration
            )

            # Generate CMC diagnostic plots if requested
            if args.cmc_plot_diagnostics:
                if hasattr(result, "is_cmc_result") and result.is_cmc_result():
                    logger.info("Generating CMC diagnostic plots...")
                    _generate_cmc_diagnostic_plots(
                        result, args.output_dir, config.config.get("analysis_mode")
                    )
                else:
                    logger.warning(
                        "CMC diagnostic plots requested but result is not a CMC result "
                        "(use --method cmc or ensure dataset is large enough for auto-selection)"
                    )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        optimization_time = time.perf_counter() - start_time
        logger.info(
            f"✓ {method.upper()} optimization completed in {optimization_time:.3f}s",
        )

        return result

    except Exception as e:
        optimization_time = time.perf_counter() - start_time
        logger.error(
            f"{method.upper()} optimization failed after {optimization_time:.3f}s: {e}",
        )
        raise


def _generate_cmc_diagnostic_plots(
    result: Any, output_dir: Path, analysis_mode: str
) -> None:
    """Generate CMC diagnostic plots.

    This function generates diagnostic visualizations for Consensus Monte Carlo results:
    - Per-shard convergence diagnostics (R-hat, ESS, acceptance rate)
    - Between-shard KL divergence heatmap
    - Combined posterior parameter distributions

    Note: This is a placeholder implementation. Full visualization functionality
    will be added when Task Group 11 (Visualization) is complete.

    Parameters
    ----------
    result : Any
        MCMC result object with CMC diagnostics (must be a CMC result)
    output_dir : Path
        Output directory for saving plots
    analysis_mode : str
        Analysis mode (static_isotropic or laminar_flow)
    """
    # Check if result is a CMC result
    if not (hasattr(result, "is_cmc_result") and result.is_cmc_result()):
        logger.warning(
            "Result is not a CMC result - skipping diagnostic plots"
        )
        return

    # Check if result has CMC diagnostics
    if not hasattr(result, "cmc_diagnostics") or result.cmc_diagnostics is None:
        logger.warning(
            "CMC diagnostics not available in result - skipping diagnostic plots"
        )
        return

    try:
        # Create diagnostics subdirectory
        diag_dir = output_dir / "cmc_diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        # Save diagnostic data as JSON for now (visualization to be implemented)
        diag_data = {
            "per_shard_diagnostics": result.cmc_diagnostics.get(
                "per_shard_diagnostics", []
            ),
            "between_shard_kl": result.cmc_diagnostics.get("kl_matrix", []),
            "success_rate": result.cmc_diagnostics.get("success_rate", 0.0),
            "combined_diagnostics": result.cmc_diagnostics.get(
                "combined_diagnostics", {}
            ),
        }

        import json

        diag_file = diag_dir / "cmc_diagnostics.json"
        with open(diag_file, "w") as f:
            json.dump(diag_data, f, indent=2, default=str)

        logger.info(f"CMC diagnostic data saved to: {diag_file}")
        logger.info(
            "Note: Graphical diagnostic plots will be available after Task Group 11 (Visualization) is complete"
        )

    except Exception as e:
        logger.warning(f"Failed to generate CMC diagnostic plots: {e}")


def _save_results(
    args,
    result: Any,
    device_config: dict[str, Any],
    data: dict[str, Any],
    config: Any,
) -> None:
    """Save optimization results to output directory.

    Breaking Change (October 2025)
    -------------------------------
    Added `data` and `config` parameters to support NLSQ comprehensive saving.

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
    logger.info(f"Saving results to: {args.output_dir}")

    import json

    import numpy as np
    import yaml

    # Route to appropriate saving method based on optimization method
    if args.method == "nlsq":
        # Use comprehensive NLSQ saving (4 files: 3 JSON + 1 NPZ)
        logger.info("Using comprehensive NLSQ result saving")
        save_nlsq_results(result, data, config, args.output_dir)
        # Also save legacy format for backward compatibility if requested
        if args.output_format != "json":
            logger.info("Saving legacy results summary for backward compatibility")
    elif args.method in ["auto", "nuts", "cmc", "mcmc"]:
        # Use comprehensive MCMC/CMC saving (4 files: 3 JSON + 1 NPZ + plots)
        logger.info("Using comprehensive MCMC result saving")
        save_mcmc_results(result, data, config, args.output_dir)
        # Also save legacy format for backward compatibility if requested
        if args.output_format != "json":
            logger.info("Saving legacy results summary for backward compatibility")
    # For other methods, continue with legacy saving format below

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
                "results_summary": np.array([results_summary], dtype=object),
            }
            if hasattr(result, "samples_params") and result.samples_params is not None:
                arrays_to_save["samples_params"] = result.samples_params
            np.savez(output_file, **arrays_to_save)

        logger.info(f"✓ Results saved: {output_file}")

    except Exception as e:
        logger.warning(f"Failed to save results: {e}")


def _handle_plotting(
    args,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any] = None,
) -> None:
    """Handle plotting options for experimental and simulated data.

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
            "Install with: pip install matplotlib",
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
                config,
                contrast,
                offset,
                phi_angles_str,
                plots_dir,
                data,
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

        # Generate and plot fitted simulations
        if result is not None and config is not None:
            try:
                _generate_and_plot_fitted_simulations(
                    result,
                    data,
                    config,
                    args.output_dir,
                )
            except Exception as e:
                logger.warning(f"Failed to generate fitted simulations: {e}")
                logger.debug("Fitted simulation error:", exc_info=True)


def _apply_angle_filtering(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Core angle filtering logic shared by optimization and plotting.

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
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering to select specific angles for plotting.

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
        phi_angles,
        c2_exp,
        config,
    )

    # Add plot-specific logging
    phi_filtering_config = config.get("phi_filtering", {})

    if not phi_filtering_config.get("enabled", False):
        logger.debug("Phi filtering not enabled, plotting all angles")
    elif not phi_filtering_config.get("target_ranges", []):
        logger.warning(
            "Phi filtering enabled but no target_ranges specified, plotting all angles",
        )
    elif not filtered_indices or len(filtered_indices) == len(phi_angles):
        if len(filtered_indices) == 0:
            logger.warning("No angles matched target ranges, plotting all angles")
        # else: all angles matched, no special logging needed
    else:
        logger.info(
            f"Angle filtering applied: {len(filtered_indices)} angles selected "
            f"from {len(phi_angles)} total angles",
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
        f"Plotting {len(filtered_indices)} angles after filtering: {filtered_phi_angles}",
    )

    # Handle different data shapes
    if c2_exp.ndim == 3:
        # Data shape: (n_phi, n_t1, n_t2)
        # Save individual heatmap for EACH phi angle
        n_angles = c2_exp.shape[0]

        logger.info(f"Generating individual C₂ heatmaps for {n_angles} phi angles...")

        for angle_idx in range(n_angles):
            # Get actual phi angle value
            phi_deg = (
                phi_angles_list[angle_idx] if len(phi_angles_list) > angle_idx else 0.0
            )
            angle_data = c2_exp[angle_idx]

            # Create individual figure for this phi angle
            fig, ax = plt.subplots(figsize=(8, 7))

            # Create C2 heatmap
            # Transpose to show diagonal from bottom-left to top-right
            # Data structure: c2[t1_idx, t2_idx] → c2.T for correct imshow display
            im = ax.imshow(
                angle_data.T,
                aspect="equal",
                cmap="viridis",
                origin="lower",
                extent=extent,
            )
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(
                f"Experimental C₂(t₁, t₂) at φ={phi_deg:.1f}°",
                fontsize=13,
                fontweight="bold",
            )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
            cbar.ax.tick_params(labelsize=9)

            # Calculate and display key statistics on the plot
            mean_val = np.mean(angle_data)
            max_val = np.max(angle_data)
            min_val = np.min(angle_data)

            # Add text box with statistics
            stats_text = f"Mean: {mean_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

            plt.tight_layout()

            # Save individual file with phi angle in filename
            filename = f"experimental_data_phi_{phi_deg:.1f}.png"
            plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
            plt.close()

            logger.debug(f"  ✓ Saved: {filename}")

        logger.info(f"✓ Generated {n_angles} individual C₂ heatmaps")

        # Plot diagonal (t1=t2) for all phi angles
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get time values for x-axis if available
        if t1 is not None:
            time_diagonal = np.diag(
                t1,
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
            plots_dir / "experimental_data_diagonal.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    elif c2_exp.ndim == 2:
        # 2D data: single correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        # Transpose to show diagonal from bottom-left to top-right
        im = ax.imshow(
            c2_exp.T,
            aspect="equal",
            cmap="viridis",
            origin="lower",
            extent=extent,
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
        f"Generating simulated data plots (contrast={contrast:.3f}, offset={offset:.3f})",
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
    params_dict = (
        dict(zip(param_names, param_values, strict=False))
        if param_names and param_values
        else {}
    )

    if analysis_mode.startswith("static"):
        # Static mode: 3 parameters
        params = jnp.array(
            [
                params_dict.get("D0", 100.0),
                params_dict.get("alpha", -0.5),
                params_dict.get("D_offset", 0.0),
            ],
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
            ],
        )

    logger.debug(
        f"Using parameters: {dict(zip(model.parameter_names, params, strict=False))}",
    )

    # Determine phi angles for theoretical simulation plots
    # Note: These plots show theoretical behavior with initial parameters,
    # independent of angle filtering used for optimization
    #
    # Priority:
    # 1. Use CLI-provided phi_angles_str if specified
    # 2. Use ALL experimental data phi angles (unfiltered) if available
    # 3. Fall back to default range
    if phi_angles_str:
        # Use CLI-provided angles (highest priority for explicit control)
        phi_degrees = np.array([float(x.strip()) for x in phi_angles_str.split(",")])
        phi = phi_degrees
        logger.info(
            f"Using CLI-provided phi angles for theoretical plots: {phi_degrees}",
        )
    elif data is not None and "phi_angles_list" in data:
        # Use experimental data's phi angles
        # Note: May be filtered if angle filtering is enabled in config
        phi_degrees = np.array(data["phi_angles_list"])
        phi = phi_degrees
        logger.info(
            f"Using experimental data phi angles for theoretical plots: {phi_degrees}",
        )
        logger.warning(
            "Theoretical plots using potentially filtered phi angles from experimental data. "
            "To use all angles, disable phi_filtering in config or provide --phi-angles explicitly.",
        )
    else:
        # Default: 8 evenly spaced angles from 0 to 180 degrees
        phi_degrees = np.linspace(0, 180, 8)
        phi = phi_degrees
        logger.info(f"Using default phi angles for theoretical plots: {phi_degrees}")

    logger.debug(f"Generating simulated data for {len(phi)} phi angles")

    # Generate time arrays matching configuration specification
    # CRITICAL: Simulated data must be independent of experimental data
    analyzer_params = config.get("analyzer_parameters", {})
    dt = analyzer_params.get("dt", 0.1)
    start_frame = analyzer_params.get("start_frame", 1)
    end_frame = analyzer_params.get("end_frame", 8000)

    # Calculate number of time points (inclusive frame counting)
    # This matches the data loader convention: n = end - start + 1
    n_time_points = end_frame - start_frame + 1

    # Generate time array starting at t=0 (matches experimental data convention)
    # The JAX backend handles t=0 singularity internally via epsilon protection
    # in _calculate_shear_rate_impl_jax() when beta < 0
    time_max = dt * (end_frame - start_frame)
    t_vals = np.linspace(0, time_max, n_time_points)
    t1_grid, t2_grid = np.meshgrid(t_vals, t_vals, indexing="ij")

    logger.debug(
        f"Simulated data time grid: dt={dt}, start_frame={start_frame}, end_frame={end_frame}",
    )
    logger.debug(
        f"Time range: [{float(t_vals[0]):.4f}, {float(t_vals[-1]):.2f}] seconds with {n_time_points} points",
    )
    logger.debug(
        f"Time spacing verification: t[1]-t[0]={float(t_vals[1] - t_vals[0]):.6f} (should equal dt={dt})",
    )

    # Get wavevector_q and stator_rotor_gap from correct config sections
    scattering_config = analyzer_params.get("scattering", {})
    geometry_config = analyzer_params.get("geometry", {})

    q = scattering_config.get("wavevector_q", 0.0054)  # Wave vector in Å⁻¹
    L_angstroms = geometry_config.get("stator_rotor_gap", 2000000)  # Gap in Angstroms

    # Convert to microns for display (1 μm = 10,000 Å)
    L_microns = L_angstroms / 10000.0

    logger.info(
        f"Generating theoretical C₂ with q={q:.6f} Å⁻¹, L={L_microns:.1f} μm ({L_angstroms:.0f} Å)",
    )
    logger.debug(
        f"Physics parameters: q={q}, L={L_angstroms} (Angstroms - used by physics code)",
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
        logger.debug(
            f"  C₂ shape: {c2_result.shape}, range: [{c2_result.min():.4f}, {c2_result.max():.4f}]",
        )
        c2_simulated.append(c2_result)

    c2_simulated = np.array(c2_simulated)  # Shape: (n_phi, n_t, n_t)

    logger.info(f"Generated simulated C₂ with shape: {c2_simulated.shape}")

    # Compute global color scale across all angles for consistent visualization
    # CRITICAL: Use same vmin/vmax for all subplots to make them comparable
    vmin = c2_simulated.min()
    vmax = c2_simulated.max()
    logger.debug(f"Global color scale: vmin={vmin:.6f}, vmax={vmax:.6f}")

    # Save individual C₂ heatmap for EACH phi angle
    n_phi = len(phi)
    logger.info(
        f"Generating individual simulated C₂ heatmaps for {n_phi} phi angles...",
    )

    for idx in range(n_phi):
        # Create individual figure for this phi angle
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create C2 heatmap
        # Note: No vmin/vmax for individual plots - auto-scale each plot
        # for optimal visualization (like experimental plots)
        # Transpose to show diagonal from bottom-left to top-right
        im = ax.imshow(
            c2_simulated[idx].T,
            extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]],
            aspect="equal",
            cmap="viridis",
            origin="lower",
        )
        ax.set_xlabel("t₁ (s)", fontsize=11)
        ax.set_ylabel("t₂ (s)", fontsize=11)
        ax.set_title(
            f"Simulated C₂(t₁, t₂) at φ={phi_degrees[idx]:.1f}°",
            fontsize=13,
            fontweight="bold",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        # Calculate and display key statistics
        mean_val = np.mean(c2_simulated[idx])
        max_val = np.max(c2_simulated[idx])
        min_val = np.min(c2_simulated[idx])

        # Add text box with statistics
        stats_text = f"Mean: {mean_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # Add analysis mode info
        mode_text = (
            f"Mode: {analysis_mode}\nContrast: {contrast:.3f}\nOffset: {offset:.3f}"
        )
        ax.text(
            0.02,
            0.02,
            mode_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.7},
        )

        plt.tight_layout()

        # Save individual file with phi angle in filename
        filename = f"simulated_data_phi_{phi_degrees[idx]:.1f}.png"
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  ✓ Saved: {filename}")

    logger.info(f"✓ Generated {n_phi} individual simulated C₂ heatmaps")

    # Plot 2: Diagonal (t1=t2) for all phi angles
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx in range(min(10, len(phi))):
        diagonal = np.diag(c2_simulated[idx])
        ax.plot(
            t_vals,
            diagonal,
            label=f"φ={phi_degrees[idx]:.1f}°",
            alpha=0.7,
            linewidth=2,
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


def _generate_and_plot_fitted_simulations(
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    output_dir,
) -> None:
    """Generate and plot C2 simulations using fitted parameters from optimization.

    This function generates simulated C2 data using the optimized parameters and
    saves individual plots for each phi angle in the simulated_data/ subdirectory.

    Parameters
    ----------
    result : OptimizationResult
        Optimization result containing fitted parameters
    data : dict
        Experimental data dictionary containing phi_angles_list, t1, t2, c2_exp
    config : dict
        Configuration dictionary with analysis_mode and physics parameters
    output_dir : Path
        Output directory path (simulated_data/ subdirectory will be created here)
    """
    import json

    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np

    from homodyne.core.models import CombinedModel

    logger.info("Generating fitted C₂ simulations...")

    # Create simulated_data subdirectory
    simulated_data_dir = output_dir / "simulated_data"
    simulated_data_dir.mkdir(parents=True, exist_ok=True)

    # Extract fitted parameters from result
    if hasattr(result, "parameters"):
        # NLSQ result format
        fitted_params_dict = result.parameters
        contrast = fitted_params_dict.get("contrast", 0.5)
        offset = fitted_params_dict.get("offset", 1.0)
        physical_params = fitted_params_dict.get("physical_params", [])
    elif hasattr(result, "mean_params"):
        # MCMC result format
        contrast = result.mean_contrast
        offset = result.mean_offset
        physical_params = result.mean_params
    else:
        logger.warning("Cannot extract fitted parameters from result")
        return

    # Convert to JAX array
    if isinstance(physical_params, list):
        params = jnp.array(physical_params)
    elif hasattr(physical_params, "tolist"):
        params = jnp.array(physical_params.tolist())
    else:
        params = jnp.array(physical_params)

    logger.info(
        f"Using fitted parameters: contrast={contrast:.4f}, offset={offset:.4f}",
    )
    logger.debug(f"Physical parameters: {params}")

    # Get analysis mode
    analysis_mode = config.get("analysis_mode", "static_isotropic")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Create model
    model = CombinedModel(analysis_mode)

    # Get experimental data structure
    phi_angles_list = data.get("phi_angles_list", None)
    t1 = data.get("t1", None)
    t2 = data.get("t2", None)

    if phi_angles_list is None or t1 is None or t2 is None:
        logger.warning("Missing experimental data structure (phi_angles_list, t1, t2)")
        return

    # Convert to JAX arrays
    t1_grid = jnp.array(t1)
    t2_grid = jnp.array(t2)

    # Get physics parameters from config
    analyzer_params = config.get("analyzer_parameters", {})
    scattering_config = analyzer_params.get("scattering", {})
    geometry_config = analyzer_params.get("geometry", {})
    dt = analyzer_params.get("dt", 0.1)

    q = scattering_config.get("wavevector_q", 0.0054)
    L_angstroms = geometry_config.get("stator_rotor_gap", 2000000)

    logger.debug(f"Physics: q={q:.6f} Å⁻¹, L={L_angstroms:.0f} Å, dt={dt}")

    # Generate fitted C2 for each phi angle
    c2_fitted_list = []

    for _i, phi_deg in enumerate(phi_angles_list):
        phi_array = jnp.array([phi_deg])

        logger.debug(f"Generating fitted C₂ for φ={phi_deg:.1f}°")

        # Compute g2 with fitted parameters
        c2_phi = model.compute_g2(
            params,
            t1_grid,
            t2_grid,
            phi_array,
            q,
            L_angstroms,
            contrast,
            offset,
            dt,
        )

        # Extract 2D array (remove phi dimension)
        c2_result = np.array(c2_phi[0])
        c2_fitted_list.append(c2_result)

        logger.debug(f"  C₂ range: [{c2_result.min():.4f}, {c2_result.max():.4f}]")

    c2_fitted = np.array(c2_fitted_list)  # Shape: (n_phi, n_t1, n_t2)

    logger.info(f"Generated fitted C₂ with shape: {c2_fitted.shape}")

    # Save fitted C2 data as NPZ
    npz_file = simulated_data_dir / "c2_fitted_data.npz"
    np.savez(
        npz_file,
        c2_data=c2_fitted,
        phi_angles=phi_angles_list,
        t1=t1,
        t2=t2,
        initial_params=params,
        contrast=contrast,
        offset=offset,
    )
    logger.info(f"✓ Saved fitted C₂ data: {npz_file}")

    # Save configuration for fitted simulation
    config_file = simulated_data_dir / "simulation_config_fitted.json"
    sim_config = {
        "command_line_args": {
            "contrast": float(contrast),
            "offset": float(offset),
            "phi_angles": ",".join(f"{x:.1f}" for x in phi_angles_list),
        },
        "parameters": {
            "values": params.tolist() if hasattr(params, "tolist") else list(params),
            "names": model.parameter_names,
        },
        "data_type": "fitted",
        "analysis_mode": analysis_mode,
    }
    with open(config_file, "w") as f:
        json.dump(sim_config, f, indent=2)
    logger.info(f"✓ Saved simulation config: {config_file}")

    # Generate individual plots for each phi angle
    # Note: Using auto-scaling per plot for optimal visualization
    logger.info(
        f"Generating individual fitted C₂ plots for {len(phi_angles_list)} angles...",
    )

    # Get time extent for plotting
    if t1 is not None and t2 is not None:
        t_min = float(np.min(t1))
        t_max = float(np.max(t1))
        extent = [t_min, t_max, t_min, t_max]
        xlabel = "t₂ (s)"
        ylabel = "t₁ (s)"
    else:
        extent = None
        xlabel = "t₂ Index"
        ylabel = "t₁ Index"

    for i, phi_deg in enumerate(phi_angles_list):
        # Create individual figure
        fig, ax = plt.subplots(figsize=(8, 7))

        # Create heatmap
        # Note: No vmin/vmax for individual plots - auto-scale each plot
        # for optimal visualization (like experimental plots)
        # Transpose to show diagonal from bottom-left to top-right
        im = ax.imshow(
            c2_fitted[i].T,
            aspect="equal",
            cmap="viridis",
            origin="lower",
            extent=extent,
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"Fitted C₂(t₁, t₂) at φ={phi_deg:.1f}°",
            fontsize=13,
            fontweight="bold",
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        # Calculate statistics
        mean_val = np.mean(c2_fitted[i])
        max_val = np.max(c2_fitted[i])
        min_val = np.min(c2_fitted[i])

        # Add statistics box
        stats_text = f"Mean: {mean_val:.4f}\nRange: [{min_val:.4f}, {max_val:.4f}]"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )

        # Add fitting info
        fit_text = f"Fitted Parameters\nContrast: {contrast:.3f}\nOffset: {offset:.3f}"
        ax.text(
            0.02,
            0.02,
            fit_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            bbox={"boxstyle": "round", "facecolor": "lightgreen", "alpha": 0.7},
        )

        plt.tight_layout()

        # Save with correct filename pattern: simulated_c2_fitted_phi_{angle}deg.png
        filename = f"simulated_c2_fitted_phi_{phi_deg:.1f}deg.png"
        plt.savefig(simulated_data_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  ✓ Saved: {filename}")

    logger.info(f"✓ Generated {len(phi_angles_list)} individual fitted C₂ plots")
    logger.info(f"✓ Fitted simulation data saved to: {simulated_data_dir}")


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


# ==============================================================================
# NLSQ Result Saving Helper Functions
# ==============================================================================


def _extract_nlsq_metadata(config: Any, data: dict[str, Any]) -> dict[str, Any]:
    """Extract required metadata for NLSQ theoretical fit computation.

    Implements multi-level fallback hierarchy for robust metadata extraction:
    - L (characteristic length): stator_rotor_gap → sample_detector_distance → default
    - dt (time step): analyzer_parameters.dt → experimental_data.dt → None
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
    Default L = 2000000.0 Å (200 µm, typical rheology-XPCS stator-rotor gap).
    Missing dt or q will log warnings but not crash - downstream functions
    must handle None values appropriately.

    Examples
    --------
    >>> metadata = _extract_nlsq_metadata(config, data)
    >>> metadata['L']  # Should be float in Angstroms
    2000000.0
    >>> metadata['q']  # Should be float in Å⁻¹
    0.0123
    """
    metadata = {}

    # L (characteristic length) extraction with fallback hierarchy
    try:
        analyzer_params = config.config.get("analyzer_parameters", {})
        geometry = analyzer_params.get("geometry", {})

        if "stator_rotor_gap" in geometry:
            metadata["L"] = float(geometry["stator_rotor_gap"])
            logger.debug(f"Using stator_rotor_gap L = {metadata['L']:.1f} Å")
        else:
            exp_config = config.config.get("experimental_data", {})
            exp_geometry = exp_config.get("geometry", {})

            if "stator_rotor_gap" in exp_geometry:
                metadata["L"] = float(exp_geometry["stator_rotor_gap"])
                logger.debug("Using L from experimental_data.geometry")
            elif "sample_detector_distance" in exp_config:
                metadata["L"] = float(exp_config["sample_detector_distance"])
                logger.debug(
                    f"Using sample_detector_distance L = {metadata['L']:.1f} Å",
                )
            else:
                metadata["L"] = 2000000.0  # Default: 200 µm
                logger.warning(
                    f"No L parameter found, using default L = {metadata['L']:.1f} Å",
                )
    except (AttributeError, TypeError, ValueError) as e:
        metadata["L"] = 2000000.0
        logger.warning(f"Error reading L: {e}, using default L = 2000000.0 Å")

    # dt (time step) extraction (optional)
    try:
        analyzer_params = config.config.get("analyzer_parameters", {})
        dt_value = analyzer_params.get("dt")

        if dt_value is None:
            exp_config = config.config.get("experimental_data", {})
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
            logger.debug(f"Using q = {metadata['q']:.6f} Å⁻¹")
        else:
            metadata["q"] = None
            logger.warning("Empty wavevector_q_list")
    except (KeyError, IndexError, TypeError) as e:
        metadata["q"] = None
        logger.error(f"Error extracting q: {e}")

    return metadata


def _prepare_parameter_data(result: Any, analysis_mode: str) -> dict[str, Any]:
    """Prepare parameter data dictionary for JSON saving.

    Extracts parameter values and uncertainties from OptimizationResult and
    organizes them by name according to the analysis mode.

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result with parameters and uncertainties
    analysis_mode : str
        "static_isotropic" (5 params) or "laminar_flow" (9 params)

    Returns
    -------
    dict[str, Any]
        Dictionary mapping parameter names to {value, uncertainty} dicts

    Notes
    -----
    Parameter order in result.parameters:
    - Static isotropic: [contrast, offset, D0, alpha, D_offset]
    - Laminar flow: [contrast, offset, D0, alpha, D_offset,
                     gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    Examples
    --------
    >>> param_dict = _prepare_parameter_data(result, "laminar_flow")
    >>> param_dict["D0"]
    {'value': 1234.5, 'uncertainty': 45.6}
    >>> param_dict["gamma_dot_t0"]
    {'value': 0.000123, 'uncertainty': 0.000012}
    """
    # Get parameter names for analysis mode
    if analysis_mode == "static_isotropic":
        param_names = SCALING_PARAM_NAMES + STATIC_PARAM_NAMES
    elif analysis_mode == "laminar_flow":
        param_names = SCALING_PARAM_NAMES + LAMINAR_FLOW_PARAM_NAMES
    else:
        raise ValueError(f"Unknown analysis_mode: {analysis_mode}")

    # Extract values and uncertainties
    param_dict = {}
    for i, name in enumerate(param_names):
        param_dict[name] = {
            "value": float(result.parameters[i]),
            "uncertainty": (
                float(result.uncertainties[i])
                if result.uncertainties is not None
                else None
            ),
        }

    return param_dict


def _apply_diagonal_correction_to_c2(c2_mat: np.ndarray) -> np.ndarray:
    """Apply diagonal correction to correlation matrix.

    Matches the diagonal correction applied to experimental data in xpcs_loader.py.
    Replaces diagonal values with average of adjacent off-diagonal values.

    Based on pyXPCSViewer's correct_diagonal_c2 function.

    Parameters
    ----------
    c2_mat : np.ndarray
        Correlation matrix with shape (n_times, n_times)

    Returns
    -------
    np.ndarray
        Correlation matrix with corrected diagonal

    Notes
    -----
    This correction:
    - Removes systematic artifacts in diagonal C2(t,t) values
    - Ensures fitted data matches experimental data processing
    - Fixes constant diagonal issue in theoretical model

    Reference: homodyne/data/xpcs_loader.py lines 925-953
    """
    size = c2_mat.shape[0]

    # Extract side band (one off-diagonal: i,i+1 elements)
    side_band = c2_mat[(np.arange(size - 1), np.arange(1, size))]

    # Create diagonal values as average of adjacent off-diagonal elements
    diag_val = np.zeros(size)
    diag_val[:-1] += side_band  # Upper side
    diag_val[1:] += side_band   # Lower side (same as upper for symmetric matrix)

    # Normalization: edge elements averaged once, interior elements averaged twice
    norm = np.ones(size)
    norm[1:-1] = 2

    # Create corrected matrix (copy to avoid modifying input)
    c2_corrected = c2_mat.copy()
    c2_corrected[np.diag_indices(size)] = diag_val / norm

    return c2_corrected


def _compute_nlsq_fits(
    result: Any,
    data: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Compute theoretical fits with per-angle least squares scaling.

    Generates theoretical correlation functions using optimized parameters,
    then applies per-angle scaling (contrast, offset) via least squares fitting
    to match experimental intensities.

    Parameters
    ----------
    result : OptimizationResult
        NLSQ optimization result with physical parameters
    data : dict[str, Any]
        Experimental data with phi_angles_list, c2_exp, t1, t2
    metadata : dict[str, Any]
        Metadata with L, dt, q for theoretical computation

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'c2_theoretical_raw': Raw theoretical fits (n_angles, n_t1, n_t2)
        - 'c2_theoretical_scaled': Scaled fits (n_angles, n_t1, n_t2)
        - 'per_angle_scaling': Scaling params (n_angles, 2) [contrast, offset]
        - 'residuals': Exp - scaled fit (n_angles, n_t1, n_t2)

    Notes
    -----
    Uses sequential per-angle computation (not vectorized). Each angle calls
    compute_g2_scaled() independently. Per-angle scaling via np.linalg.lstsq
    solves: c2_exp = contrast * c2_theory + offset.
    """
    phi_angles = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])
    t1 = np.asarray(data["t1"])
    t2 = np.asarray(data["t2"])

    # Convert 2D meshgrids to 1D if needed
    if t1.ndim == 2:
        t1 = t1[:, 0]  # Extract first column
    if t2.ndim == 2:
        t2 = t2[0, :]  # Extract first row

    # Extract physical parameters (skip first 2 which are scaling params)
    physical_params = result.parameters[2:]

    # Extract metadata with defaults
    L = metadata["L"]
    dt = metadata.get("dt")
    if dt is None:
        dt = 0.1  # Default time resolution in seconds
        logger.debug(f"Using default dt = {dt}s (not found in config)")
    q = metadata["q"]

    if q is None:
        raise ValueError("q (wavevector) is required but was not found")

    logger.debug(
        f"Computing theoretical fits for {len(phi_angles)} angles using L={L:.1f} Å, q={q:.6f} Å⁻¹",
    )
    logger.info("Diagonal correction will be applied to theoretical fits to match experimental data processing")

    # Sequential per-angle computation
    c2_theoretical_raw = []
    per_angle_scaling = []

    for i, phi_angle in enumerate(phi_angles):
        # Convert to JAX arrays
        phi_jax = jnp.array([float(phi_angle)])
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        params_jax = jnp.array(physical_params)

        # Compute theoretical fit (raw, before scaling)
        g2_theory = compute_g2_scaled(
            params=params_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi=phi_jax,
            q=float(q),
            L=float(L),
            contrast=1.0,  # Will scale later
            offset=0.0,
            dt=float(dt),
        )

        # Convert to NumPy and squeeze out extra dimension (phi axis)
        g2_theory_np = np.asarray(g2_theory)
        if g2_theory_np.ndim == 3:
            g2_theory_np = g2_theory_np[0]  # Remove phi dimension (size 1)

        # Apply diagonal correction to match experimental data processing
        # This fixes the constant diagonal issue in theoretical model (g1(t,t) = 1 always)
        diag_before = np.diag(g2_theory_np).copy()
        g2_theory_np = _apply_diagonal_correction_to_c2(g2_theory_np)
        diag_after = np.diag(g2_theory_np).copy()

        logger.debug(
            f"Angle {phi_angle:.1f}°: Diagonal correction - before: [{diag_before[0]:.3f}, {diag_before[1]:.3f}, ..., {diag_before[-1]:.3f}], "
            f"after: [{diag_after[0]:.3f}, {diag_after[1]:.3f}, ..., {diag_after[-1]:.3f}]"
        )

        c2_theoretical_raw.append(g2_theory_np)

        # Least squares scaling: c2_exp = contrast * c2_theory + offset
        # Solve: [c2_theory, ones] @ [contrast, offset] = c2_exp
        c2_exp_angle = c2_exp[i].flatten()
        c2_theory_flat = g2_theory_np.flatten()

        # Design matrix: [c2_theory, ones]
        A = np.column_stack([c2_theory_flat, np.ones_like(c2_theory_flat)])
        b = c2_exp_angle

        # Least squares solution
        scaling, residuals_lstsq, rank, s = np.linalg.lstsq(A, b, rcond=None)
        contrast_fit, offset_fit = scaling
        per_angle_scaling.append([contrast_fit, offset_fit])

        logger.debug(
            f"Angle {phi_angle:.1f}°: contrast={contrast_fit:.3f}, offset={offset_fit:.3f}",
        )

    # Stack arrays
    c2_theoretical_raw = np.array(c2_theoretical_raw)
    per_angle_scaling = np.array(per_angle_scaling)

    # Apply per-angle scaling
    c2_theoretical_scaled = np.zeros_like(c2_theoretical_raw)
    for i in range(len(phi_angles)):
        contrast, offset = per_angle_scaling[i]
        c2_theoretical_scaled[i] = contrast * c2_theoretical_raw[i] + offset

    # Compute residuals
    residuals = c2_exp - c2_theoretical_scaled

    logger.info(
        f"Computed theoretical fits for {len(phi_angles)} angles (sequential computation, diagonal corrected)",
    )

    return {
        "c2_theoretical_raw": c2_theoretical_raw,
        "c2_theoretical_scaled": c2_theoretical_scaled,
        "per_angle_scaling": per_angle_scaling,
        "residuals": residuals,
    }


def _save_nlsq_json_files(
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

    logger.info("Saved 3 JSON files (parameters, analysis results, convergence)")


def _save_nlsq_npz_file(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_raw: np.ndarray,
    c2_scaled: np.ndarray,
    per_angle_scaling: np.ndarray,
    residuals: np.ndarray,
    residuals_norm: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    q: float,
    output_dir: Path,
) -> None:
    """Save NPZ file with 10 arrays: experimental + theoretical + residuals + coordinates.

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
    per_angle_scaling : np.ndarray
        Per-angle scaling parameters (n_angles, 2) [contrast, offset]
    residuals : np.ndarray
        Residuals: exp - scaled (n_angles, n_t1, n_t2)
    residuals_norm : np.ndarray
        Normalized residuals (n_angles, n_t1, n_t2)
    t1 : np.ndarray
        Time array 1 (n_t1,)
    t2 : np.ndarray
        Time array 2 (n_t2,)
    q : float
        Wavevector magnitude [Å⁻¹]
    output_dir : Path
        Output directory

    Returns
    -------
    None
        NPZ file saved to disk

    Notes
    -----
    Creates fitted_data.npz with 10 arrays matching classical implementation format.
    """
    npz_file = output_dir / "fitted_data.npz"

    np.savez_compressed(
        npz_file,
        # Experimental data (2 arrays)
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        # Note: sigma (uncertainties) not included - would need to pass from data if available
        # Theoretical fits (3 arrays)
        c2_theoretical_raw=c2_raw,
        c2_theoretical_scaled=c2_scaled,
        per_angle_scaling=per_angle_scaling,
        # Residuals (2 arrays)
        residuals=residuals,
        residuals_normalized=residuals_norm,
        # Coordinate arrays (3 arrays)
        t1=t1,
        t2=t2,
        q=np.array([q]),  # Wrap scalar in array
    )

    logger.info(f"Saved NPZ file with 10 arrays to {npz_file}")


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

    Returns
    -------
    None
        All files saved to output_dir/nlsq/

    Raises
    ------
    ValueError
        If required metadata (q) cannot be extracted

    Notes
    -----
    Creates nlsq/ subdirectory matching classical/ structure for method comparison.
    Performs sequential per-angle theoretical fit computation.
    """
    # Create nlsq subdirectory
    nlsq_dir = output_dir / "nlsq"
    nlsq_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving NLSQ results to {nlsq_dir}")

    # Get analysis mode
    analysis_mode = config.config.get("analysis_mode", "static_isotropic")

    # Step 1: Extract metadata
    logger.debug("Extracting metadata (L, dt, q)")
    metadata = _extract_nlsq_metadata(config, data)

    # Step 2: Prepare parameter data
    logger.debug(f"Preparing parameter data for {analysis_mode} mode")
    param_dict = _prepare_parameter_data(result, analysis_mode)

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
    fits_dict = _compute_nlsq_fits(result, data, metadata)

    # Step 4: Prepare analysis results dictionary
    phi_angles = np.asarray(data["phi_angles_list"])
    c2_exp = np.asarray(data["c2_exp"])
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
            "q_value": float(metadata["q"]),
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

    # Step 6: Save JSON files
    logger.info("Saving JSON files (parameters, analysis, convergence)")
    _save_nlsq_json_files(
        param_dict_complete,
        analysis_dict,
        convergence_dict,
        nlsq_dir,
    )

    # Step 7: Compute normalized residuals
    # For now, assume uniform uncertainty of 5% (would need sigma from data for real normalization)
    # Use safe division to avoid divide-by-zero warnings where c2_exp == 0
    residuals_norm = np.divide(
        fits_dict["residuals"],
        0.05 * c2_exp,
        out=np.zeros_like(fits_dict["residuals"]),
        where=(c2_exp != 0),
    )

    # Convert time arrays to 1D
    # Note: t1 and t2 are already in SECONDS from the data loader (_calculate_time_arrays)
    # Do NOT multiply by dt again - that would give seconds²
    t1 = np.asarray(data["t1"])
    t2 = np.asarray(data["t2"])
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
        per_angle_scaling=fits_dict["per_angle_scaling"],
        residuals=fits_dict["residuals"],
        residuals_norm=residuals_norm,
        t1=t1,
        t2=t2,
        q=metadata["q"],
        output_dir=nlsq_dir,
    )

    logger.info(f"✓ NLSQ results saved successfully to {nlsq_dir}")
    logger.info("  - 3 JSON files (parameters, analysis results, convergence metrics)")
    logger.info("  - 1 NPZ file (10 arrays: experimental + theoretical + residuals)")

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
            config=config,  # Pass config for preview_mode setting
        )
        logger.info(f"  - {len(phi_angles)} PNG plots")
    except Exception as e:
        logger.warning(f"Plot generation failed (data files still saved): {e}")
        logger.debug("Plot error details:", exc_info=True)


def generate_nlsq_plots(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_theoretical_scaled: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    config: Any = None,
    use_datashader: bool = True,
    parallel: bool = True,
) -> None:
    """Generate 3-panel heatmap plots for NLSQ fit visualization.

    **Hybrid Rendering Approach:**
    - **Preview mode (preview_mode: true)**: Datashader backend, 5-10x faster
    - **Publication mode (preview_mode: false)**: Matplotlib backend, high quality

    Mode selection priority:
    1. Config file: output.plots.preview_mode
    2. Legacy parameter: use_datashader (backward compatible)
    3. Default: Publication mode (matplotlib)

    Performance:
        - Publication (matplotlib): ~150-300ms per plot
        - Preview (Datashader): ~30-60ms per plot (5-10x speedup)
        - Preview + parallel (8 cores): ~4-8ms per plot (20-40x speedup)

    Parameters
    ----------
    phi_angles : np.ndarray
        Scattering angles in degrees (n_angles,)
    c2_exp : np.ndarray
        Experimental correlation data (n_angles, n_t1, n_t2)
    c2_theoretical_scaled : np.ndarray
        Scaled theoretical fits (n_angles, n_t1, n_t2)
    residuals : np.ndarray
        Residuals: exp - scaled (n_angles, n_t1, n_t2)
    t1 : np.ndarray
        Time array 1 in seconds (n_t1,)
    t2 : np.ndarray
        Time array 2 in seconds (n_t2,)
    output_dir : Path
        Output directory for PNG files
    config : ConfigManager or dict, optional
        Configuration object/dict containing output.plots.preview_mode setting
    use_datashader : bool, default=True
        Legacy parameter for backward compatibility. Overridden by config.
    parallel : bool, default=True
        Generate plots in parallel using multiprocessing (Nx speedup, N=cores).
        Recommended for multiple angles.

    Returns
    -------
    None
        PNG files saved to disk

    Notes
    -----
    - Creates one PNG per angle: c2_heatmaps_phi_{angle:.1f}deg.png
    - Layout: 3 panels (experimental, fitted, residuals)
    - Colormaps: viridis (exp, fit), RdBu_r (residuals, symmetric)
    - Resolution: 300 DPI for publication quality
    - Datashader canvas: 1200×1200 pixels (configurable via output.plots.datashader.canvas_width)
    - Matplotlib interpolation: bilinear (configurable via output.plots.matplotlib.interpolation)

    Examples
    --------
    >>> # Publication mode (default, matplotlib)
    >>> generate_nlsq_plots(
    ...     phi_angles, c2_exp, c2_fit, residuals, t1, t2,
    ...     output_dir=Path("./results/nlsq"),
    ...     config=config_manager,  # preview_mode: false
    ... )

    >>> # Preview mode (fast, Datashader)
    >>> generate_nlsq_plots(
    ...     phi_angles, c2_exp, c2_fit, residuals, t1, t2,
    ...     output_dir=Path("./results/nlsq"),
    ...     config=config_manager,  # preview_mode: true
    ...     parallel=True,
    ... )
    """
    logger.info(f"Generating heatmap plots for {len(phi_angles)} angles")

    # Determine rendering mode from config (priority: config > use_datashader legacy param)
    preview_mode = use_datashader  # Default to legacy parameter
    width = 1200
    height = 1200

    if config is not None:
        # Extract config dict if ConfigManager object
        config_dict = config.config if hasattr(config, "config") else config

        # Read preview_mode from config (output.plots.preview_mode)
        output_config = config_dict.get("output", {})
        plots_config = output_config.get("plots", {})
        preview_mode = plots_config.get("preview_mode", preview_mode)

        # Read Datashader canvas resolution
        datashader_config = plots_config.get("datashader", {})
        width = datashader_config.get("canvas_width", width)
        height = datashader_config.get("canvas_height", height)

        logger.debug(
            f"Plot config: preview_mode={preview_mode}, "
            f"canvas={width}×{height}, parallel={parallel}",
        )

    # Select backend based on mode
    if preview_mode and DATASHADER_AVAILABLE:
        logger.info("Using Datashader backend (preview mode, fast rendering)")
        _generate_plots_datashader(
            phi_angles,
            c2_exp,
            c2_theoretical_scaled,
            residuals,
            t1,
            t2,
            output_dir,
            parallel=parallel,
            width=1200,
            height=1200,
        )
    else:
        if preview_mode and not DATASHADER_AVAILABLE:
            logger.warning(
                "Preview mode (Datashader) requested but Datashader not available. "
                "Install with: pip install datashader xarray colorcet"
            )
            logger.info("Falling back to matplotlib backend (publication quality)")
        else:
            logger.info("Using matplotlib backend (publication quality)")

        _generate_plots_matplotlib(
            phi_angles,
            c2_exp,
            c2_theoretical_scaled,
            residuals,
            t1,
            t2,
            output_dir,
        )


# ============================================================================
# MCMC/CMC Result Saving Functions
# ============================================================================


def save_mcmc_results(
    result: Any,
    data: dict[str, Any],
    config: Any,
    output_dir: Path,
) -> None:
    """Save MCMC/CMC results with comprehensive diagnostics.

    Creates method-specific directory (mcmc/ or cmc/) and saves:
    1. parameters.json: Posterior mean ± std for each parameter
    2. analysis_results_mcmc.json: Sampling summary and diagnostics
    3. samples.npz: Full posterior samples, r_hat, ess
    4. diagnostics.json: Convergence metrics
    5. fitted_data.npz: Experimental + theoretical data (optional)
    6. c2_heatmaps_phi_*.png: Comparison plots using posterior mean
    7. trace_plots.png: MCMC diagnostic plots (future)

    Parameters
    ----------
    result : MCMCResult
        MCMC/CMC optimization result with posterior samples and diagnostics
    data : dict
        Experimental data dictionary containing c2_exp, phi_angles_list, t1, t2, q
    config : ConfigManager
        Configuration manager with analysis settings
    output_dir : Path
        Base output directory (method-specific subdirectory will be created)

    Notes
    -----
    This function follows the same structure as save_nlsq_results() for consistency.
    It reuses plotting functions from NLSQ for heatmap generation.

    Examples
    --------
    >>> from pathlib import Path
    >>> result = fit_mcmc_jax(data, ...)
    >>> save_mcmc_results(result, data, config, Path("homodyne_results"))
    # Creates: homodyne_results/mcmc/parameters.json, samples.npz, etc.
    """
    import json

    import numpy as np

    # Determine method name (mcmc or cmc)
    method_name = (
        "cmc"
        if (hasattr(result, "is_cmc_result") and result.is_cmc_result())
        else "mcmc"
    )

    # Create method-specific directory
    method_dir = output_dir / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {method_name.upper()} results to: {method_dir}")

    # Step 1: Save parameters.json with posterior statistics
    try:
        param_dict = _create_mcmc_parameters_dict(result)
        param_file = method_dir / "parameters.json"
        with open(param_file, "w") as f:
            json.dump(param_dict, f, indent=2)
        logger.debug(f"Saved parameters to {param_file}")
    except Exception as e:
        logger.warning(f"Failed to save parameters.json: {e}")
        logger.debug("Parameter saving error:", exc_info=True)

    # Step 2: Save samples.npz with full posterior
    try:
        samples_file = method_dir / "samples.npz"
        save_dict = {}

        # Combine samples from separate attributes (samples_params, samples_contrast, samples_offset)
        if hasattr(result, "samples_params") and result.samples_params is not None:
            samples_list = [result.samples_params]

            if hasattr(result, "samples_contrast") and result.samples_contrast is not None:
                # Reshape to (n_samples, 1) if needed
                contrast_samples = result.samples_contrast
                if contrast_samples.ndim == 1:
                    contrast_samples = contrast_samples[:, np.newaxis]
                samples_list.insert(0, contrast_samples)

            if hasattr(result, "samples_offset") and result.samples_offset is not None:
                # Reshape to (n_samples, 1) if needed
                offset_samples = result.samples_offset
                if offset_samples.ndim == 1:
                    offset_samples = offset_samples[:, np.newaxis]
                samples_list.insert(1 if hasattr(result, "samples_contrast") else 0, offset_samples)

            # Concatenate all samples
            save_dict["samples"] = np.concatenate(samples_list, axis=1)

        # Add optional diagnostics if available
        if hasattr(result, "log_prob") and result.log_prob is not None:
            save_dict["log_prob"] = result.log_prob
        if hasattr(result, "r_hat") and result.r_hat is not None:
            # Convert dict to array if needed
            if isinstance(result.r_hat, dict):
                save_dict["r_hat"] = np.array(list(result.r_hat.values()))
            else:
                save_dict["r_hat"] = result.r_hat
        if hasattr(result, "effective_sample_size") and result.effective_sample_size is not None:
            # Convert dict to array if needed
            if isinstance(result.effective_sample_size, dict):
                save_dict["ess"] = np.array(list(result.effective_sample_size.values()))
            else:
                save_dict["ess"] = result.effective_sample_size
        if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
            save_dict["acceptance_rate"] = np.array([result.acceptance_rate])

        if save_dict:  # Only save if we have data
            np.savez_compressed(samples_file, **save_dict)
            logger.debug(f"Saved posterior samples to {samples_file}")
    except Exception as e:
        logger.warning(f"Failed to save samples.npz: {e}")
        logger.debug("Samples saving error:", exc_info=True)

    # Step 3: Save analysis_results_mcmc.json
    try:
        analysis_dict = _create_mcmc_analysis_dict(result, data, method_name)
        analysis_file = method_dir / f"analysis_results_{method_name}.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis_dict, f, indent=2)
        logger.debug(f"Saved analysis results to {analysis_file}")
    except Exception as e:
        logger.warning(f"Failed to save analysis_results_{method_name}.json: {e}")
        logger.debug("Analysis results saving error:", exc_info=True)

    # Step 4: Save diagnostics.json
    try:
        diagnostics_dict = _create_mcmc_diagnostics_dict(result)
        diagnostics_file = method_dir / "diagnostics.json"
        with open(diagnostics_file, "w") as f:
            json.dump(diagnostics_dict, f, indent=2)
        logger.debug(f"Saved diagnostics to {diagnostics_file}")
    except Exception as e:
        logger.warning(f"Failed to save diagnostics.json: {e}")
        logger.debug("Diagnostics saving error:", exc_info=True)

    # Step 4b: Save shard_diagnostics.json for CMC results
    if method_name == "cmc" and hasattr(result, "per_shard_diagnostics") and result.per_shard_diagnostics:
        try:
            shard_diag_file = method_dir / "shard_diagnostics.json"
            with open(shard_diag_file, "w") as f:
                json.dump(result.per_shard_diagnostics, f, indent=2, default=str)
            logger.debug(f"Saved per-shard diagnostics to {shard_diag_file}")
        except Exception as e:
            logger.warning(f"Failed to save shard_diagnostics.json: {e}")
            logger.debug("Shard diagnostics saving error:", exc_info=True)

    # Step 5: Generate heatmap plots (reuse NLSQ plotting)
    try:
        logger.info("Generating comparison heatmap plots")

        # Compute theoretical C2 using posterior mean parameters
        c2_theoretical_scaled = _compute_theoretical_c2_from_mcmc(result, data, config)

        # Calculate residuals
        c2_exp = data["c2_exp"]
        residuals = c2_exp - c2_theoretical_scaled

        # Convert time arrays
        t1 = np.asarray(data["t1"])
        t2 = np.asarray(data["t2"])
        if t1.ndim == 2:
            t1 = t1[:, 0]
        if t2.ndim == 2:
            t2 = t2[0, :]

        # Generate plots using NLSQ plotting function
        generate_nlsq_plots(
            phi_angles=data["phi_angles_list"],
            c2_exp=c2_exp,
            c2_theoretical_scaled=c2_theoretical_scaled,
            residuals=residuals,
            t1=t1,
            t2=t2,
            output_dir=method_dir,
            config=config,
        )
        logger.info(f"  - {len(data['phi_angles_list'])} PNG heatmap plots")
    except Exception as e:
        logger.warning(f"Heatmap plot generation failed (data files still saved): {e}")
        logger.debug("Plot error details:", exc_info=True)

    logger.info(f"✓ {method_name.upper()} results saved successfully to {method_dir}")
    if method_name == "cmc" and hasattr(result, "per_shard_diagnostics") and result.per_shard_diagnostics:
        logger.info("  - 4 JSON files (parameters, analysis results, diagnostics, shard diagnostics)")
    else:
        logger.info("  - 3 JSON files (parameters, analysis results, diagnostics)")
    logger.info("  - 1 NPZ file (posterior samples)")


def _create_mcmc_parameters_dict(result: Any) -> dict:
    """Create parameters dictionary with posterior statistics.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with posterior samples and statistics

    Returns
    -------
    dict
        Structured parameter dictionary with posterior mean ± std
    """
    import numpy as np
    from datetime import datetime

    param_dict = {
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": getattr(result, "analysis_mode", "unknown"),
        "method": "cmc" if (hasattr(result, "is_cmc_result") and result.is_cmc_result()) else "mcmc",
        "sampling_summary": {
            "n_samples": getattr(result, "n_samples", 0),
            "n_warmup": getattr(result, "n_warmup", 0),
            "n_chains": getattr(result, "n_chains", 1),
            "total_samples": getattr(result, "n_samples", 0) * getattr(result, "n_chains", 1),
            "computation_time": getattr(result, "computation_time", 0.0),
        },
        "convergence": {},
        "parameters": {},
    }

    # Add convergence diagnostics if available
    if hasattr(result, "r_hat") and result.r_hat is not None:
        # r_hat can be either dict or array
        if isinstance(result.r_hat, dict):
            # Filter out None values
            r_hat_values = [v for v in result.r_hat.values() if v is not None]
            if r_hat_values:
                param_dict["convergence"]["all_chains_converged"] = bool(all(v < 1.1 for v in r_hat_values))
                param_dict["convergence"]["min_r_hat"] = float(min(r_hat_values))
                param_dict["convergence"]["max_r_hat"] = float(max(r_hat_values))
        else:
            r_hat = np.asarray(result.r_hat)
            param_dict["convergence"]["all_chains_converged"] = bool(np.all(r_hat < 1.1))
            param_dict["convergence"]["min_r_hat"] = float(np.min(r_hat))
            param_dict["convergence"]["max_r_hat"] = float(np.max(r_hat))

    if hasattr(result, "effective_sample_size") and result.effective_sample_size is not None:
        # ESS can be either dict or array (attribute name is effective_sample_size, not ess)
        if isinstance(result.effective_sample_size, dict):
            # Filter out None values
            ess_values = [v for v in result.effective_sample_size.values() if v is not None]
            if ess_values:
                param_dict["convergence"]["min_ess"] = float(min(ess_values))
        else:
            ess = np.asarray(result.effective_sample_size)
            param_dict["convergence"]["min_ess"] = float(np.min(ess))

    if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
        param_dict["convergence"]["acceptance_rate"] = float(result.acceptance_rate)

    # Add scaling parameters (contrast, offset)
    if hasattr(result, "mean_contrast"):
        param_dict["parameters"]["contrast"] = {
            "mean": float(result.mean_contrast),
            "std": float(getattr(result, "std_contrast", 0.0)),
        }

    if hasattr(result, "mean_offset"):
        param_dict["parameters"]["offset"] = {
            "mean": float(result.mean_offset),
            "std": float(getattr(result, "std_offset", 0.0)),
        }

    # Add physical parameters
    if hasattr(result, "mean_params") and result.mean_params is not None:
        analysis_mode = getattr(result, "analysis_mode", "static_isotropic")
        param_names = _get_parameter_names(analysis_mode)

        mean_params = np.asarray(result.mean_params)
        std_params = np.asarray(getattr(result, "std_params", np.zeros_like(mean_params)))

        for i, name in enumerate(param_names):
            if i < len(mean_params):
                param_dict["parameters"][name] = {
                    "mean": float(mean_params[i]),
                    "std": float(std_params[i]) if i < len(std_params) else 0.0,
                }

    return param_dict


def _create_mcmc_analysis_dict(
    result: Any,
    data: dict[str, Any],
    method_name: str,
) -> dict:
    """Create analysis results dictionary for MCMC/CMC.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with diagnostics
    data : dict
        Experimental data dictionary
    method_name : str
        "mcmc" or "cmc"

    Returns
    -------
    dict
        Analysis summary dictionary
    """
    import numpy as np
    from datetime import datetime

    # Get dataset dimensions
    c2_exp = data.get("c2_exp", [])
    n_angles = len(data.get("phi_angles_list", []))
    n_time_points = c2_exp.shape[1] * c2_exp.shape[2] if hasattr(c2_exp, "shape") and len(c2_exp.shape) >= 3 else 0
    total_data_points = c2_exp.size if hasattr(c2_exp, "size") else 0

    # Determine sampling quality
    quality_flag = "unknown"
    warnings = []
    recommendations = []

    if hasattr(result, "r_hat") and result.r_hat is not None:
        # r_hat can be either dict or array
        if isinstance(result.r_hat, dict):
            # Filter out None values
            r_hat_values = [v for v in result.r_hat.values() if v is not None]
            max_r_hat = max(r_hat_values) if r_hat_values else None
        else:
            r_hat = np.asarray(result.r_hat)
            max_r_hat = np.max(r_hat)

        if max_r_hat is not None:
            if max_r_hat < 1.05:
                quality_flag = "good"
            elif max_r_hat < 1.1:
                quality_flag = "acceptable"
                warnings.append(f"Some parameters have R-hat between 1.05-1.1 (max={max_r_hat:.3f})")
            else:
                quality_flag = "poor"
                warnings.append(f"Convergence issues detected (max R-hat={max_r_hat:.3f})")
                recommendations.append("Consider increasing n_warmup or n_samples")

    if hasattr(result, "effective_sample_size") and result.effective_sample_size is not None:
        # ESS can be either dict or array
        if isinstance(result.effective_sample_size, dict):
            # Filter out None values
            ess_values = [v for v in result.effective_sample_size.values() if v is not None]
            min_ess = min(ess_values) if ess_values else None
        else:
            ess = np.asarray(result.effective_sample_size)
            min_ess = np.min(ess)

        if min_ess is not None and min_ess < 400:
            warnings.append(f"Low effective sample size (min ESS={min_ess:.0f})")
            recommendations.append("Consider increasing n_samples for better posterior estimates")

    analysis_dict = {
        "method": method_name,
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": getattr(result, "analysis_mode", "unknown"),
        "sampling_quality": {
            "convergence_status": "converged" if quality_flag in ["good", "acceptable"] else "not_converged",
            "quality_flag": quality_flag,
            "warnings": warnings,
            "recommendations": recommendations,
        },
        "dataset_info": {
            "n_angles": n_angles,
            "n_time_points": n_time_points,
            "total_data_points": total_data_points,
            "q_value": float(data.get("wavevector_q_list", [0.0])[0]) if data.get("wavevector_q_list") is not None else 0.0,
        },
        "sampling_summary": {
            "n_samples": getattr(result, "n_samples", 0),
            "n_warmup": getattr(result, "n_warmup", 0),
            "n_chains": getattr(result, "n_chains", 1),
            "execution_time": float(getattr(result, "computation_time", 0.0)),
        },
    }

    return analysis_dict


def _create_mcmc_diagnostics_dict(result: Any) -> dict:
    """Create diagnostics dictionary for MCMC/CMC.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with convergence diagnostics

    Returns
    -------
    dict
        Diagnostics dictionary with convergence metrics
    """
    import numpy as np

    diagnostics_dict = {
        "convergence": {},
        "sampling_efficiency": {},
        "posterior_checks": {},
    }

    # Convergence diagnostics
    if hasattr(result, "r_hat") and result.r_hat is not None:
        # r_hat can be either dict or array
        if isinstance(result.r_hat, dict):
            # Filter out None values
            r_hat_values = [v for v in result.r_hat.values() if v is not None]
            if r_hat_values:
                diagnostics_dict["convergence"]["all_chains_converged"] = bool(all(v < 1.1 for v in r_hat_values))
                diagnostics_dict["convergence"]["r_hat_threshold"] = 1.1

            # Add per-parameter diagnostics using dict keys (only for non-None values)
            per_param = []
            for param_name, r_hat_val in result.r_hat.items():
                # Skip None values
                if r_hat_val is None:
                    continue

                ess_val = None
                if hasattr(result, "effective_sample_size") and isinstance(result.effective_sample_size, dict):
                    ess_val = result.effective_sample_size.get(param_name, None)

                per_param.append({
                    "name": param_name,
                    "r_hat": float(r_hat_val),
                    "ess": float(ess_val) if ess_val is not None else 0.0,
                    "converged": bool(r_hat_val < 1.1),
                })
            if per_param:
                diagnostics_dict["convergence"]["per_parameter_diagnostics"] = per_param
        else:
            r_hat = np.asarray(result.r_hat)
            diagnostics_dict["convergence"]["all_chains_converged"] = bool(np.all(r_hat < 1.1))
            diagnostics_dict["convergence"]["r_hat_threshold"] = 1.1

            # Get parameter names if available
            analysis_mode = getattr(result, "analysis_mode", "static_isotropic")
            param_names = _get_parameter_names(analysis_mode)

            # Add per-parameter diagnostics
            per_param = []
            ess_array = np.asarray(result.effective_sample_size) if (
                hasattr(result, "effective_sample_size") and
                result.effective_sample_size is not None and
                not isinstance(result.effective_sample_size, dict)
            ) else None

            for i, name in enumerate(param_names):
                if i < len(r_hat):
                    ess_val = ess_array[i] if (ess_array is not None and i < len(ess_array)) else 0.0
                    per_param.append({
                        "name": name,
                        "r_hat": float(r_hat[i]),
                        "ess": float(ess_val),
                        "converged": bool(r_hat[i] < 1.1),
                    })

            diagnostics_dict["convergence"]["per_parameter_diagnostics"] = per_param

    if hasattr(result, "effective_sample_size") and result.effective_sample_size is not None:
        diagnostics_dict["convergence"]["ess_threshold"] = 400

    # Sampling efficiency
    if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
        diagnostics_dict["sampling_efficiency"]["acceptance_rate"] = float(result.acceptance_rate)
        diagnostics_dict["sampling_efficiency"]["target_acceptance"] = 0.80

    if hasattr(result, "divergences"):
        diagnostics_dict["sampling_efficiency"]["divergences"] = int(result.divergences)

    if hasattr(result, "tree_depth_warnings"):
        diagnostics_dict["sampling_efficiency"]["tree_depth_warnings"] = int(result.tree_depth_warnings)

    # Posterior checks
    if hasattr(result, "ess") and hasattr(result, "n_samples"):
        ess = np.asarray(result.ess)
        total_samples = result.n_samples * getattr(result, "n_chains", 1)
        if total_samples > 0:
            ess_ratio = float(np.mean(ess) / total_samples)
            diagnostics_dict["posterior_checks"]["effective_sample_size_ratio"] = ess_ratio

    # CMC-specific diagnostics
    if hasattr(result, "is_cmc_result") and result.is_cmc_result():
        diagnostics_dict["cmc_specific"] = {}

        # Per-shard diagnostics summary
        if hasattr(result, "per_shard_diagnostics") and result.per_shard_diagnostics:
            per_shard = result.per_shard_diagnostics

            # Extract acceptance rates
            acceptance_rates = []
            converged_shards = 0

            for shard in per_shard:
                if isinstance(shard, dict):
                    if shard.get("acceptance_rate") is not None:
                        acceptance_rates.append(float(shard["acceptance_rate"]))
                    if shard.get("converged", False):
                        converged_shards += 1

            shard_summary = {
                "num_shards": len(per_shard),
                "shards_converged": converged_shards,
                "convergence_rate": float(converged_shards / len(per_shard)) if len(per_shard) > 0 else 0.0,
            }

            # Add acceptance rate statistics if available
            if acceptance_rates:
                shard_summary["acceptance_rate_stats"] = {
                    "mean": float(np.mean(acceptance_rates)),
                    "min": float(np.min(acceptance_rates)),
                    "max": float(np.max(acceptance_rates)),
                    "std": float(np.std(acceptance_rates)),
                }

            diagnostics_dict["cmc_specific"]["shard_summary"] = shard_summary

        # Overall CMC diagnostics
        if hasattr(result, "cmc_diagnostics") and result.cmc_diagnostics:
            cmc_diag = result.cmc_diagnostics

            # Extract key metrics safely
            overall_metrics = {}

            if isinstance(cmc_diag, dict):
                if "combination_success" in cmc_diag:
                    overall_metrics["combination_success"] = bool(cmc_diag["combination_success"])
                if "n_shards_converged" in cmc_diag:
                    overall_metrics["n_shards_converged"] = int(cmc_diag["n_shards_converged"])
                if "n_shards_total" in cmc_diag:
                    overall_metrics["n_shards_total"] = int(cmc_diag["n_shards_total"])
                if "weighted_product_std" in cmc_diag:
                    overall_metrics["weighted_product_std"] = float(cmc_diag["weighted_product_std"])
                if "combination_time" in cmc_diag:
                    overall_metrics["combination_time"] = float(cmc_diag["combination_time"])
                if "success_rate" in cmc_diag:
                    overall_metrics["success_rate"] = float(cmc_diag["success_rate"])

                # Include full diagnostics if available
                diagnostics_dict["cmc_specific"]["overall_diagnostics"] = overall_metrics

        # Combination method
        if hasattr(result, "combination_method") and result.combination_method:
            diagnostics_dict["cmc_specific"]["combination_method"] = str(result.combination_method)

        # Number of shards
        if hasattr(result, "num_shards") and result.num_shards:
            diagnostics_dict["cmc_specific"]["num_shards"] = int(result.num_shards)

    return diagnostics_dict


def _get_parameter_names(analysis_mode: str) -> list[str]:
    """Get parameter names for given analysis mode.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode ("static_isotropic" or "laminar_flow")

    Returns
    -------
    list[str]
        List of parameter names

    Raises
    ------
    ValueError
        If analysis mode is unknown
    """
    if analysis_mode == "static_isotropic":
        return ["D0", "alpha", "D_offset"]
    elif analysis_mode == "laminar_flow":
        return ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
    else:
        logger.warning(f"Unknown analysis mode: {analysis_mode}, assuming static_isotropic")
        return ["D0", "alpha", "D_offset"]


def _compute_theoretical_c2_from_mcmc(
    result: Any,
    data: dict[str, Any],
    config: Any,
) -> np.ndarray:
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
    np.ndarray
        Theoretical C2 with shape (n_angles, n_t1, n_t2)
    """
    import jax.numpy as jnp
    import numpy as np

    from homodyne.core.jax_backend import compute_g2_scaled

    # Extract parameters from MCMC result
    contrast = getattr(result, "mean_contrast", 0.5)
    offset = getattr(result, "mean_offset", 1.0)
    mean_params = np.asarray(result.mean_params)

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
    config_dict = config.get_config() if hasattr(config, "get_config") else config
    analysis_mode = config_dict.get("analysis_mode", "static_isotropic")

    # Get L parameter (stator-rotor gap)
    L = config_dict.get("model_params", {}).get("L", 2000000.0)

    # Get dt parameter (required for correct physics)
    dt = config_dict.get("acquisition", {}).get("dt", 1e-8)

    # Compute theoretical C2 for all angles
    c2_theoretical_list = []

    for phi in phi_angles:
        # Convert parameters to JAX arrays
        params_jax = jnp.array(mean_params)

        # Compute G2 for this angle
        c2_theoretical = compute_g2_scaled(
            params=params_jax,
            t1=jnp.array(t1),
            t2=jnp.array(t2),
            phi=jnp.array([phi]),  # Single angle as array
            q=q_val,
            L=L,
            contrast=contrast,
            offset=offset,
            dt=dt,
        )

        # Extract result for this single angle
        c2_theoretical_np = np.array(c2_theoretical[0])  # First angle
        c2_theoretical_list.append(c2_theoretical_np)

    # Stack all angles
    c2_theoretical_scaled = np.array(c2_theoretical_list)

    return c2_theoretical_scaled


def _worker_init_cpu_only():
    """Initialize worker process with CPU-only mode to prevent CUDA OOM.

    When spawning multiple workers for parallel plotting, each worker would
    try to initialize its own CUDA context, leading to GPU memory exhaustion.
    Since plotting is CPU-bound (Datashader/matplotlib), we force CPU mode.

    Sets multiple environment variables to aggressively disable CUDA/GPU:
    - JAX_PLATFORMS: Tells JAX to only use CPU platform
    - CUDA_VISIBLE_DEVICES: Hides all CUDA devices from process
    - JAX_ENABLE_X64: Prevents any GPU-specific 64-bit operations
    """
    import os

    # Primary: Tell JAX to only use CPU platform
    os.environ["JAX_PLATFORMS"] = "cpu"

    # Secondary: Hide all CUDA devices from this process
    # Use "-1" instead of "" to explicitly disable (empty string might be ignored)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Tertiary: Disable XLA GPU compilation and memory preallocation
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    # Suppress TensorFlow/XLA warnings
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _plot_single_angle_datashader(args):
    """Plot single angle for parallel processing (picklable module-level function).

    Args:
        args: Tuple of (i, phi_angles, c2_exp, c2_fit, residuals, t1, t2, output_dir, width, height)

    Returns:
        Path to generated plot file
    """
    # CRITICAL: Set CPU-only mode BEFORE any imports that might trigger CUDA
    # This must be first to prevent CUDA OOM in parallel workers
    # Belt-and-suspenders approach: set here even though initializer also sets them
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Explicitly disable (not empty string)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    # Lazy import to ensure environment variables take effect
    from homodyne.viz.datashader_backend import plot_c2_comparison_fast

    i, phi_angles, c2_exp, c2_fit, residuals, t1, t2, output_dir, width, height = args
    phi = phi_angles[i]
    output_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"

    # Use Datashader fast plotting with higher resolution
    plot_c2_comparison_fast(
        c2_exp,
        c2_fit,
        residuals,
        t1,
        t2,
        output_file,
        phi_angle=phi,
        width=width,
        height=height,
    )

    return output_file


def _generate_plots_datashader(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_theoretical_scaled: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    parallel: bool = True,
    width: int = 1200,
    height: int = 1200,
) -> None:
    """Generate plots using Datashader backend with optional parallelization.

    IMPORTANT: Uses 'spawn' multiprocessing method to avoid JAX deadlock.
    JAX is multithreaded, and fork() + threading = deadlock on Linux.

    Parameters
    ----------
    width : int, default=1200
        Datashader canvas width in pixels. Higher values preserve more detail
        but increase file size. Recommended: 1200-1500 for publication quality.
    height : int, default=1200
        Datashader canvas height in pixels.
    """
    import multiprocessing

    if parallel and len(phi_angles) > 1:
        # Use 'spawn' method to avoid JAX threading deadlock
        # fork() + JAX multithreading = deadlock on Linux
        ctx = multiprocessing.get_context("spawn")

        # Parallel processing for maximum speed
        n_workers = min(multiprocessing.cpu_count(), len(phi_angles))
        logger.info(f"Using {n_workers} parallel workers for plotting (spawn method)")

        # Prepare arguments for parallel processing
        args_list = [
            (
                i,
                phi_angles,
                c2_exp[i],
                c2_theoretical_scaled[i],
                residuals[i],
                t1,
                t2,
                output_dir,
                width,
                height,
            )
            for i in range(len(phi_angles))
        ]

        try:
            # Use map_async with timeout to prevent indefinite hangs
            # Initialize workers with CPU-only mode to prevent CUDA OOM
            with ctx.Pool(
                processes=n_workers, initializer=_worker_init_cpu_only
            ) as pool:
                # Timeout: 30 seconds per plot * number of angles / workers + 60s buffer
                timeout_seconds = (30 * len(phi_angles) / n_workers) + 60
                logger.debug(f"Parallel plotting timeout: {timeout_seconds:.0f}s")

                result = pool.map_async(_plot_single_angle_datashader, args_list)
                result.get(timeout=timeout_seconds)

            logger.info(f"✓ Generated {len(phi_angles)} heatmap plots (parallel)")

        except Exception as e:
            logger.warning(f"Parallel plotting failed: {e.__class__.__name__}: {e}")
            logger.info("Falling back to sequential plotting...")

            # Fallback to sequential processing
            for i in range(len(phi_angles)):
                args = (
                    i,
                    phi_angles,
                    c2_exp[i],
                    c2_theoretical_scaled[i],
                    residuals[i],
                    t1,
                    t2,
                    output_dir,
                    width,
                    height,
                )
                _plot_single_angle_datashader(args)

            logger.info(
                f"✓ Generated {len(phi_angles)} heatmap plots (sequential fallback)"
            )
    else:
        # Sequential processing
        for i in range(len(phi_angles)):
            args = (
                i,
                phi_angles,
                c2_exp[i],
                c2_theoretical_scaled[i],
                residuals[i],
                t1,
                t2,
                output_dir,
                width,
                height,
            )
            _plot_single_angle_datashader(args)

        logger.info(f"✓ Generated {len(phi_angles)} heatmap plots (sequential)")


def _generate_plots_matplotlib(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_theoretical_scaled: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate plots using matplotlib backend (original implementation)."""
    logger.info(f"Generating heatmap plots for {len(phi_angles)} angles")

    for i, phi in enumerate(phi_angles):
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Experimental data
        # Transpose because data is structured as c2[t1_index, t2_index] with indexing="ij"
        # but we want x-axis=t1, y-axis=t2 for display
        im0 = axes[0].imshow(
            c2_exp[i].T,
            origin="lower",
            aspect="equal",
            cmap="viridis",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
        )
        axes[0].set_title(f"Experimental C₂ (φ={phi:.1f}°)", fontsize=12)
        axes[0].set_xlabel("t₁ (s)", fontsize=10)
        axes[0].set_ylabel("t₂ (s)", fontsize=10)
        cbar0 = plt.colorbar(im0, ax=axes[0], label="C₂(t₁,t₂)")
        cbar0.ax.tick_params(labelsize=8)

        # Panel 2: Theoretical fit
        im1 = axes[1].imshow(
            c2_theoretical_scaled[i].T,
            origin="lower",
            aspect="equal",
            cmap="viridis",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
        )
        axes[1].set_title(f"Classical Fit (φ={phi:.1f}°)", fontsize=12)
        axes[1].set_xlabel("t₁ (s)", fontsize=10)
        axes[1].set_ylabel("t₂ (s)", fontsize=10)
        cbar1 = plt.colorbar(im1, ax=axes[1], label="C₂(t₁,t₂)")
        cbar1.ax.tick_params(labelsize=8)

        # Panel 3: Residuals (symmetric colormap)
        residual_max = np.max(np.abs(residuals[i]))
        im2 = axes[2].imshow(
            residuals[i].T,
            origin="lower",
            aspect="equal",
            cmap="RdBu_r",
            vmin=-residual_max,
            vmax=residual_max,
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
        )
        axes[2].set_title(f"Residuals (φ={phi:.1f}°)", fontsize=12)
        axes[2].set_xlabel("t₁ (s)", fontsize=10)
        axes[2].set_ylabel("t₂ (s)", fontsize=10)
        cbar2 = plt.colorbar(im2, ax=axes[2], label="ΔC₂")
        cbar2.ax.tick_params(labelsize=8)

        # Adjust layout and save
        plt.tight_layout()
        plot_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved plot: {plot_file}")

    logger.info(f"✓ Generated {len(phi_angles)} heatmap plots (matplotlib)")


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
