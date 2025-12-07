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

from homodyne.cli.args_parser import validate_args  # noqa: E402
from homodyne.config.parameter_space import ParameterSpace  # noqa: E402
from homodyne.config.types import (  # noqa: E402
    LAMINAR_FLOW_PARAM_NAMES,
    SCALING_PARAM_NAMES,
    STATIC_PARAM_NAMES,
)
from homodyne.core.jax_backend import compute_g2_scaled  # noqa: E402
from homodyne.data.angle_filtering import (  # noqa: E402
    angle_in_range as _data_angle_in_range,
)
from homodyne.data.angle_filtering import (
    apply_angle_filtering as _data_apply_angle_filtering,
)
from homodyne.data.angle_filtering import (
    apply_angle_filtering_for_plot as _data_apply_angle_filtering_for_plot,
)
from homodyne.data.angle_filtering import (
    normalize_angle_to_symmetric_range as _data_normalize_angle_to_symmetric_range,
)
from homodyne.io.json_utils import json_safe as _io_json_safe  # noqa: E402
from homodyne.io.mcmc_writers import (  # noqa: E402
    create_mcmc_analysis_dict as _io_create_mcmc_analysis_dict,
)
from homodyne.io.mcmc_writers import (
    create_mcmc_diagnostics_dict as _io_create_mcmc_diagnostics_dict,
)
from homodyne.io.mcmc_writers import (
    create_mcmc_parameters_dict as _io_create_mcmc_parameters_dict,
)
from homodyne.io.nlsq_writers import (  # noqa: E402
    save_nlsq_json_files as _io_save_nlsq_json_files,
)
from homodyne.io.nlsq_writers import save_nlsq_npz_file as _io_save_nlsq_npz_file
from homodyne.utils.logging import configure_logging, get_logger  # noqa: E402
from homodyne.viz.experimental_plots import (  # noqa: E402
    plot_experimental_data as _viz_plot_experimental_data,
)
from homodyne.viz.experimental_plots import (
    plot_fit_comparison as _viz_plot_fit_comparison,
)
from homodyne.viz.nlsq_plots import (  # noqa: E402
    generate_and_plot_fitted_simulations as _viz_generate_and_plot_fitted_simulations,
)
from homodyne.viz.nlsq_plots import generate_nlsq_plots as _viz_generate_nlsq_plots
from homodyne.viz.nlsq_plots import plot_simulated_data as _viz_plot_simulated_data

logger = get_logger(__name__)

# Common XPCS experimental angles (in degrees) for validation
COMMON_XPCS_ANGLES = [0, 30, 45, 60, 90, 120, 135, 150, 180]


def clamp_parameters_to_bounds(
    params: dict[str, float],
    config: Any,
    analysis_mode: str,
) -> dict[str, float]:
    """Clamp NLSQ parameters to valid NumPyro prior bounds.

    NumPyro uses TruncatedNormal priors with strict bounds. NLSQ can sometimes
    return values outside these bounds (e.g., negative values for positive parameters),
    which causes MCMC initialization to fail with "Cannot find valid initial parameters".

    This function ensures all parameters are within their valid NumPyro prior bounds
    before being passed to MCMC/CMC initialization.

    NOTE: Uses hardcoded NumPyro prior bounds, not NLSQ optimization bounds.
    NLSQ bounds allow negative values for exploration, but NumPyro priors enforce
    strict physical constraints (e.g., diffusion coefficients must be positive).

    IMPORTANT: Parameters are clamped to slightly INSIDE the bounds (with epsilon offset)
    to avoid numerical issues with TruncatedNormal PDFs at exact boundary values.
    Epsilon is chosen as max(1e-6, 0.001 * range) to handle both small and large ranges.

    Parameters
    ----------
    params : dict[str, float]
        Parameter dictionary from NLSQ optimization
    config : Any
        Configuration object (unused, kept for API compatibility)
    analysis_mode : str
        Analysis mode ("static_isotropic" or "laminar_flow")

    Returns
    -------
    dict[str, float]
        Clamped parameters with all values within valid NumPyro prior bounds (with epsilon offset)

    Examples
    --------
    >>> params = {"alpha": -3.38, "D_offset": -2386.63}
    >>> clamped = clamp_parameters_to_bounds(params, config, "laminar_flow")
    >>> clamped["alpha"]  # Clamped to [-2.0 + epsilon, 2.0 - epsilon], epsilon = 0.004
    -1.996
    >>> clamped["D_offset"]  # Clamped to [0.0 + epsilon, 1e6 - epsilon], epsilon = 1000.0
    1000.0
    """
    # NumPyro prior bounds (strict physics constraints)
    # These MUST match ParameterSpace bounds from homodyne/core/fitting.py
    # which are used by TruncatedNormal priors in homodyne/optimization/mcmc.py
    NUMPYRO_PRIOR_BOUNDS = {
        # Scaling parameters (always present)
        "contrast": (0.0, 1.0),  # fitting.py: implied from prior
        "offset": (0.8, 1.2),  # fitting.py: implied from prior
        # Core diffusion parameters
        "D0": (1.0, 1.0e6),  # fitting.py:73
        "alpha": (-10.0, 10.0),  # fitting.py:74
        "D_offset": (-100000.0, 100000.0),  # fitting.py:75-78
        # Laminar flow parameters
        "gamma_dot_t0": (1.0e-5, 1.0),  # fitting.py:81
        "beta": (-10.0, 10.0),  # fitting.py:82
        "gamma_dot_t_offset": (-1.0, 1.0),  # fitting.py:83-86 (CAN BE NEGATIVE!)
        "phi0": (-30.0, 30.0),  # fitting.py:87
    }

    # Clamp each parameter
    clamped_params = {}
    clamped_count = 0

    for param_name, value in params.items():
        if param_name in NUMPYRO_PRIOR_BOUNDS:
            min_val, max_val = NUMPYRO_PRIOR_BOUNDS[param_name]

            # Check if value is already within bounds (no clamping needed)
            if min_val <= value <= max_val:
                # Value is valid, no need to clamp
                clamped_params[param_name] = float(value)
            else:
                # Value is outside bounds, clamp to epsilon-adjusted bounds
                # NumPyro's TruncatedNormal PDF approaches zero at boundaries,
                # so we use epsilon offset to avoid exact boundary values
                param_range = max_val - min_val
                epsilon = max(
                    1e-6, 0.001 * param_range
                )  # 0.1% of range or 1e-6, whichever is larger

                # Adjust bounds to be slightly inside the valid region
                min_val_adjusted = min_val + epsilon
                max_val_adjusted = max_val - epsilon

                # Clamp to epsilon-adjusted bounds
                clamped_value = np.clip(value, min_val_adjusted, max_val_adjusted)

                logger.warning(
                    f"Clamping {param_name}: {value:.6f} → {clamped_value:.6f} "
                    f"(NumPyro prior bounds: [{min_val:.3e}, {max_val:.3e}], epsilon={epsilon:.3e})"
                )
                clamped_count += 1

                clamped_params[param_name] = float(clamped_value)
        else:
            # Unknown parameter, keep as-is (shouldn't happen)
            logger.debug(f"Unknown parameter {param_name}, keeping value {value}")
            clamped_params[param_name] = value

    if clamped_count > 0:
        logger.info(
            f"Clamped {clamped_count}/{len(params)} parameters to valid NumPyro prior bounds"
        )
    else:
        logger.debug("All NLSQ parameters within NumPyro prior bounds")

    return clamped_params


def normalize_angle_to_symmetric_range(angle):
    """Normalize angle(s) to [-180°, 180°] range.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_normalize_angle_to_symmetric_range(angle)


def _angle_in_range(angle, min_angle, max_angle):
    """Check if angle is in range, accounting for wrap-around at ±180°.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_angle_in_range(angle, min_angle, max_angle)


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
    run_id = getattr(args, "run_id", None) or datetime.now().strftime("%Y%m%d_%H%M%S")
    args.run_id = run_id
    logger.info(f"Dispatching homodyne analysis command (run_id={run_id})")

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

        # Load configuration (needed for logging settings)
        config = _load_configuration(args)

        # Configure logging using config + CLI verbosity flags
        config_dict = config.get_config() if hasattr(config, "get_config") else config
        logging_cfg = (
            config_dict.get("logging", {}) if isinstance(config_dict, dict) else {}
        )
        log_file = configure_logging(
            logging_cfg,
            verbose=getattr(args, "verbose", False),
            quiet=getattr(args, "quiet", False),
            output_dir=args.output_dir,
            run_id=run_id,
        )
        if log_file:
            logger.info(f"Log file created: {log_file}")

        # Configure device (CPU/GPU)
        device_config = _configure_device(args)

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
        if log_file:
            logger.info(f"Analysis log saved to: {log_file}")
        else:
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


def _configure_device(args) -> dict[str, Any]:
    """Configure optimal device based on CLI arguments."""

    logger.info("Configuring computational device...")

    # Configure CPU-only device (GPU support removed in v2.3.0)
    device_config = configure_optimal_device()

    if device_config["configuration_successful"]:
        device_type = device_config["device_type"]
        logger.info(f"✓ Device configured: {device_type.upper()}")
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
            "\n" + "=" * 70 + "\n"
            "DEPRECATED CONFIGURATION DETECTED\n"
            + "=" * 70
            + "\n"
            + "\n\n".join(warnings_issued)
            + "\n"
            + "=" * 70
            + "\n"
            "Migration: NLSQ v3.0+ uses native large dataset handling.\n"
            "Simply remove the deprecated sections - no replacement needed.\n"
            "See: https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html\n"
            + "="
            * 70
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
        "hardware": {},
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

    Supports overriding:
    - Data file path
    - Analysis mode
    - MCMC sampling parameters (n_samples, n_warmup, n_chains)
    - CMC sharding/backend parameters
    - Initial parameter values (D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0)
    - Mass matrix type (dense_mass_matrix flag)
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

    # CMC-only: selection thresholds removed; ensure mcmc section exists
    if "mcmc" not in config.config["optimization"]:
        config.config["optimization"]["mcmc"] = {}

    # Override dense mass matrix flag
    if args.dense_mass_matrix:
        old_value = config.config["optimization"]["mcmc"].get("dense_mass", False)
        config.config["optimization"]["mcmc"]["dense_mass"] = True
        logger.info(
            f"Overriding config dense_mass={old_value} with CLI flag dense_mass=True"
        )

    # Override initial parameter values
    # Map CLI argument names to canonical parameter names
    param_overrides = {
        "initial_d0": "D0",
        "initial_alpha": "alpha",
        "initial_d_offset": "D_offset",
        "initial_gamma_dot_t0": "gamma_dot_t0",
        "initial_beta": "beta",
        "initial_gamma_dot_offset": "gamma_dot_t_offset",
        "initial_phi0": "phi0",
    }

    # Collect CLI overrides
    cli_param_values = {}
    for arg_name, param_name in param_overrides.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            cli_param_values[param_name] = arg_value

    # Apply parameter overrides to config if any provided
    if cli_param_values:
        # Ensure initial_parameters section exists
        if "initial_parameters" not in config.config:
            config.config["initial_parameters"] = {}

        # Get or create parameter_names list and values list
        param_names = config.config["initial_parameters"].get("parameter_names", [])
        param_values = config.config["initial_parameters"].get("values", [])

        # Handle case where values is None (null in YAML)
        if param_values is None:
            param_values = []

        # Convert to dict for easier manipulation
        if param_names and param_values and len(param_names) == len(param_values):
            # Use ParameterManager for name mapping (config → canonical)
            from homodyne.config.parameter_manager import ParameterManager

            pm = ParameterManager(
                config.config, config.config.get("analysis_mode", "laminar_flow")
            )

            # Build current param dict with name mapping
            current_params = {}
            for pname, pval in zip(param_names, param_values, strict=False):
                # Map config name to canonical name
                canonical_name = pm._param_name_mapping.get(pname, pname)
                current_params[canonical_name] = pval
        else:
            # No existing values or mismatch - start fresh
            current_params = {}

        # Apply CLI overrides and log
        for param_name, new_value in cli_param_values.items():
            old_value = current_params.get(param_name, None)
            current_params[param_name] = new_value

            if old_value is not None:
                logger.info(
                    f"Overriding config {param_name}={old_value:.6g} with CLI value {param_name}={new_value:.6g}"
                )
            else:
                logger.info(
                    f"Setting {param_name}={new_value:.6g} from CLI (not in config)"
                )

        # Update config with new parameter values
        # Build parameter_names and values lists in canonical order
        analysis_mode = config.config.get("analysis_mode", "laminar_flow")
        if "static" in analysis_mode.lower():
            expected_params = ["D0", "alpha", "D_offset"]
        else:
            expected_params = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

        # Only include parameters that have values (either from config or CLI)
        final_param_names = []
        final_param_values = []
        for param in expected_params:
            if param in current_params:
                final_param_names.append(param)
                final_param_values.append(current_params[param])

        config.config["initial_parameters"]["parameter_names"] = final_param_names
        config.config["initial_parameters"]["values"] = final_param_values

    # Override hardware settings
    if "hardware" not in config.config:
        config.config["hardware"] = {}

    # Hardware configuration (CPU-only in v2.3.0+)
    # No hardware overrides needed


def _build_mcmc_runtime_kwargs(args, config: ConfigManager) -> dict[str, Any]:
    """Collect runtime kwargs for fit_mcmc_jax from CLI args and YAML config."""

    cfg_dict = config.config if hasattr(config, "config") else {}
    optimization_cfg = cfg_dict.get("optimization", {}) if cfg_dict else {}
    mcmc_cfg = optimization_cfg.get("mcmc", {}) if optimization_cfg else {}

    runtime_kwargs: dict[str, Any] = {
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "n_chains": args.n_chains,
        "run_id": getattr(args, "run_id", None),
    }

    if cfg_dict:
        runtime_kwargs["config"] = cfg_dict

    def _set_runtime_value(dest: str, *aliases: str) -> None:
        for key in (dest, *aliases):
            if key in mcmc_cfg and mcmc_cfg[key] is not None:
                runtime_kwargs[dest] = mcmc_cfg[key]
                return

    _set_runtime_value("target_accept_prob")
    _set_runtime_value("max_tree_depth")
    _set_runtime_value("dense_mass_matrix", "dense_mass")
    _set_runtime_value("rng_key")
    _set_runtime_value("stable_prior_fallback")
    _set_runtime_value("min_ess")
    _set_runtime_value("max_rhat")
    _set_runtime_value("n_retries")
    _set_runtime_value("check_hmc_diagnostics")

    if args.dense_mass_matrix:
        runtime_kwargs["dense_mass_matrix"] = True

    return runtime_kwargs


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

    logger.info(f"Running {method.upper()} optimization...")

    start_time = time.perf_counter()

    # Apply angle filtering before optimization (if configured)
    filtered_data = _apply_angle_filtering_for_optimization(data, config)

    # NLSQ will handle large datasets natively via streaming optimization
    logger.debug("Using NLSQ native large dataset handling")

    try:
        if method == "nlsq":
            # Run NLSQ optimization (CPU-only in v2.3.0+)
            result = fit_nlsq_jax(filtered_data, config)
        elif method == "mcmc":
            # CMC-only MCMC
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
            logger.info(f"MCMC method: {method} (CMC-only)")

            # Log key CMC parameters
            sharding = cmc_config.get("sharding", {})
            backend = cmc_config.get("backend", {})

            # Handle both old dict schema (backend={name: ...}) and new string schema (backend="jax")
            if isinstance(backend, str):
                # New schema: backend is computational backend string
                backend_str = backend
                backend_config = cmc_config.get("backend_config", {})
                parallel_backend = (
                    backend_config.get("name", "auto") if backend_config else "auto"
                )
                backend_display = f"{backend_str}/{parallel_backend}"
            else:
                # Old schema: backend is dict with name for parallel execution
                backend_display = backend.get("name", "auto")

            logger.debug(
                f"CMC sharding: strategy={sharding.get('strategy', 'auto')}, "
                f"num_shards={sharding.get('num_shards', 'auto')}, "
                f"backend={backend_display}",
            )

            # =========================================================================
            # MCMC DATA POOLING (CRITICAL FIX)
            # =========================================================================
            # MCMC expects 1D pooled/flattened data with matching array lengths
            # Problem: c2_exp is 3D (n_phi, n_t, n_t), but t1/t2 are 2D meshgrids
            # Solution: Flatten all arrays and tile/repeat to match pooled data length

            # Extract raw data from filtered_data
            c2_3d = filtered_data["c2_exp"]  # Shape: (n_phi, n_t, n_t)
            t1_raw = filtered_data.get("t1")  # Could be 1D (n_t,) or 2D (n_t, n_t)
            t2_raw = filtered_data.get("t2")  # Could be 1D (n_t,) or 2D (n_t, n_t)
            phi_angles = filtered_data.get("phi_angles_list")  # Shape: (n_phi,)

            # Get dimensions
            n_phi = c2_3d.shape[0]
            n_t = c2_3d.shape[1]
            n_total = n_phi * n_t * n_t

            # Pool/flatten correlation data: (n_phi, n_t, n_t) → (n_phi * n_t * n_t,)
            mcmc_data = c2_3d.ravel()  # Flattens to 1D

            # Handle both 1D and 2D time arrays
            # If t1/t2 are 1D, create 2D meshgrids; if already 2D, use as-is
            if t1_raw.ndim == 1 and t2_raw.ndim == 1:
                # Create 2D meshgrids from 1D arrays
                t2_2d, t1_2d = np.meshgrid(t1_raw, t2_raw, indexing="ij")
                logger.debug(
                    f"Created 2D meshgrids from 1D arrays: t1={t1_raw.shape} → {t1_2d.shape}"
                )
            elif t1_raw.ndim == 2 and t2_raw.ndim == 2:
                # Already 2D meshgrids
                t1_2d = t1_raw
                t2_2d = t2_raw
                logger.debug(
                    f"Using existing 2D meshgrids: t1={t1_2d.shape}, t2={t2_2d.shape}"
                )
            else:
                raise ValueError(
                    f"Inconsistent t1/t2 dimensions: t1.ndim={t1_raw.ndim}, t2.ndim={t2_raw.ndim}. "
                    f"Expected both 1D or both 2D."
                )

            # Create corresponding t1, t2, phi arrays of same length
            # Tile the 2D meshgrid for each phi angle
            t1_pooled = np.tile(t1_2d.ravel(), n_phi)  # (n_t*n_t,) repeated n_phi times
            t2_pooled = np.tile(t2_2d.ravel(), n_phi)
            phi_pooled = np.repeat(
                phi_angles, n_t * n_t
            )  # Each phi repeated n_t*n_t times

            # Verify all arrays have matching lengths (CRITICAL for MCMC)
            if mcmc_data.shape[0] != n_total:
                raise ValueError(
                    f"Data pooling failed: mcmc_data={mcmc_data.shape[0]}, expected={n_total}"
                )
            if t1_pooled.shape[0] != n_total:
                raise ValueError(
                    f"Data pooling failed: t1={t1_pooled.shape[0]}, expected={n_total}"
                )
            if t2_pooled.shape[0] != n_total:
                raise ValueError(
                    f"Data pooling failed: t2={t2_pooled.shape[0]}, expected={n_total}"
                )
            if phi_pooled.shape[0] != n_total:
                raise ValueError(
                    f"Data pooling failed: phi={phi_pooled.shape[0]}, expected={n_total}"
                )

            logger.debug(
                f"Pooled MCMC data: {n_phi} angles × {n_t}×{n_t} = {n_total:,} data points"
            )
            logger.debug(
                f"Array shapes: data={mcmc_data.shape}, t1={t1_pooled.shape}, "
                f"t2={t2_pooled.shape}, phi={phi_pooled.shape}"
            )

            # ✅ v2.1.0 BREAKING CHANGE: Removed automatic NLSQ/SVI initialization
            # Manual workflow required: Run NLSQ separately, copy results to YAML, then run MCMC
            # Load initial values from config YAML (initial_parameters.values section)
            initial_values = (
                config.get_initial_parameters()
                if hasattr(config, "get_initial_parameters")
                else None
            )
            if initial_values:
                logger.debug(
                    f"MCMC initial values from config: {list(initial_values.keys())} = "
                    f"{[f'{v:.4g}' for v in initial_values.values()]}"
                )
            else:
                logger.debug(
                    "MCMC will use mid-point defaults (no initial_parameters.values in config)"
                )

            # Determine analysis mode (needed for ParameterSpace creation)
            analysis_mode_str = (
                config.config.get("analysis_mode", "static_isotropic")
                if hasattr(config, "config")
                else "static_isotropic"
            )

            # Create ParameterSpace from config to ensure NumPyro uses config bounds
            # This is CRITICAL: ParameterSpace loads bounds/priors from configuration
            # instead of hardcoded default bounds from fitting.py
            parameter_space = ParameterSpace.from_config(
                config_dict=config.config if hasattr(config, "config") else config,
                analysis_mode=analysis_mode_str,
            )
            logger.debug(
                f"Created ParameterSpace with config for {analysis_mode_str} mode"
            )

            # Run MCMC with pooled 1D arrays (all same length)
            # Pass the flattened phi array (same length as data). The MCMC runner
            # extracts the unique angles internally but also needs the per-point
            # mapping to apply contrast/offset scaling correctly.
            mcmc_runtime_kwargs = _build_mcmc_runtime_kwargs(args, config)

            result = fit_mcmc_jax(
                mcmc_data,
                t1=t1_pooled,
                t2=t2_pooled,
                phi=phi_pooled,
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
                method=method,  # Pass "mcmc" for CMC-only path
                cmc_config=cmc_config,  # Pass CMC configuration
                initial_values=initial_values,  # ✅ FIXED: Load from config initial_parameters.values
                parameter_space=parameter_space,  # ✅ Pass config-aware ParameterSpace
                **mcmc_runtime_kwargs,
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
                        "(ensure dataset is large enough for CMC auto-selection: samples >= 15 OR memory > 30%)"
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
        logger.warning("Result is not a CMC result - skipping diagnostic plots")
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
    elif args.method == "mcmc":
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

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_apply_angle_filtering(phi_angles, c2_exp, config)


def _apply_angle_filtering_for_plot(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering to select specific angles for plotting.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_apply_angle_filtering_for_plot(phi_angles, c2_exp, data)


def _plot_experimental_data(data: dict[str, Any], plots_dir) -> None:
    """Generate validation plots of experimental data.

    This is a wrapper that delegates to homodyne.viz.experimental_plots.
    """
    _viz_plot_experimental_data(
        data,
        plots_dir,
        angle_filter_func=_apply_angle_filtering_for_plot,
    )


def _plot_simulated_data(
    config: dict[str, Any],
    contrast: float,
    offset: float,
    phi_angles_str: str | None,
    plots_dir,
    data: dict[str, Any] | None = None,
) -> None:
    """Generate plots of simulated/theoretical data.

    This is a wrapper that delegates to homodyne.viz.nlsq_plots.
    """
    _viz_plot_simulated_data(config, contrast, offset, phi_angles_str, plots_dir, data)


def _generate_and_plot_fitted_simulations(
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    output_dir,
) -> None:
    """Generate and plot C2 simulations using fitted parameters from optimization.

    This is a wrapper that delegates to homodyne.viz.nlsq_plots.
    """
    _viz_generate_and_plot_fitted_simulations(
        result,
        data,
        config,
        output_dir,
        angle_filter_func=_apply_angle_filtering_for_optimization,
    )


def _plot_fit_comparison(result: Any, data: dict[str, Any], plots_dir) -> None:
    """Generate comparison plots between fit and experimental data.

    This is a wrapper that delegates to homodyne.viz.experimental_plots.
    """
    _viz_plot_fit_comparison(result, data, plots_dir)


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

    Notes
    -----
    Parameter order in result.parameters:

    **Legacy scalar scaling (deprecated):**
    - Static isotropic: [contrast, offset, D0, alpha, D_offset]
    - Laminar flow: [contrast, offset, D0, alpha, D_offset,
                     gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    **Per-angle scaling (current default, v2.4.0+):**
    - Static isotropic: [c0, c1, ..., cN, o0, o1, ..., oN, D0, alpha, D_offset]
    - Laminar flow: [c0, c1, ..., cN, o0, o1, ..., oN, D0, alpha, D_offset,
                     gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
    where N = number of phi angles

    For per-angle scaling, contrast/offset in JSON are set to mean of per-angle values.

    Examples
    --------
    >>> param_dict = _prepare_parameter_data(result, "laminar_flow", n_angles=3)
    >>> param_dict["D0"]
    {'value': 1234.5, 'uncertainty': 45.6}
    >>> param_dict["gamma_dot_t0"]
    {'value': 0.000123, 'uncertainty': 0.000012}
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
    # Note: When n_angles=1, per-angle (5 params) equals legacy (5 params),
    # so we explicitly check for the per-angle count first (v2.4.0 mandates per-angle)
    if n_params_actual == n_params_expected_per_angle:
        # Per-angle scaling detected
        # Structure: [c0, c1, ..., cN, o0, o1, ..., oN, physical_params...]
        # where N = n_angles

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

        # Use mean contrast/offset for JSON (representative value)
        contrast_mean = float(np.mean(contrast_per_angle))
        offset_mean = float(np.mean(offset_per_angle))

        # Compute uncertainties for contrast/offset (RMS of per-angle uncertainties)
        if result.uncertainties is not None:
            contrast_unc_per_angle = result.uncertainties[:n_angles]
            offset_unc_per_angle = result.uncertainties[n_angles : 2 * n_angles]
            contrast_unc = float(np.sqrt(np.mean(contrast_unc_per_angle**2)))
            offset_unc = float(np.sqrt(np.mean(offset_unc_per_angle**2)))
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

    else:
        # Unexpected parameter count - should not happen in v2.4.0+
        raise ValueError(
            f"Unexpected parameter count: got {n_params_actual}, expected {n_params_expected_per_angle} "
            f"for per-angle scaling with {n_angles} angles. "
            f"(Legacy scalar scaling with {n_params_expected_legacy} params is no longer supported in v2.4.0+)"
        )

    return param_dict


def _compute_nlsq_fits(
    result: Any,
    data: dict[str, Any],
    metadata: dict[str, Any],
    *,
    analysis_mode: str | None = None,
    include_solver_surface: bool = True,
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

    # Extract fitted parameters - prefer per-angle scaling but tolerate scalar fallback
    n_params = len(result.parameters)
    n_angles = len(phi_angles)

    def _normalize_mode(mode: str | None) -> str:
        """Resolve analysis mode, inferring from parameter counts if needed."""

        if mode:
            mode_lower = mode.lower()
            if mode_lower in {"static", "static_isotropic"}:
                return "static"
            if mode_lower == "laminar_flow":
                return "laminar_flow"

        # Infer from parameter counts (legacy scalar vs per-angle layout)
        candidates = {
            "static": 3,
            "laminar_flow": 7,
        }
        for candidate_mode, n_phys in candidates.items():
            if n_params in {n_phys + 2, 2 * n_angles + n_phys}:
                return candidate_mode

        # Default to static for backward compatibility
        logger.debug(
            "Unable to infer analysis_mode from params=%s angles=%s; defaulting to static",
            n_params,
            n_angles,
        )
        return "static"

    normalized_mode = _normalize_mode(
        analysis_mode or getattr(result, "analysis_mode", None)
    )

    # Determine number of physical parameters based on analysis mode
    if normalized_mode == "static":
        n_physical = 3  # D0, alpha, D_offset
    elif normalized_mode == "laminar_flow":
        n_physical = (
            7  # D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0
        )
    else:
        raise ValueError(
            f"Unknown analysis_mode: '{analysis_mode}'. Expected 'static' or 'laminar_flow'"
        )

    expected_params_per_angle = 2 * n_angles + n_physical

    scalar_per_angle_expansion = False
    if n_params == expected_params_per_angle:
        fitted_contrasts = result.parameters[0:n_angles]
        fitted_offsets = result.parameters[n_angles : 2 * n_angles]
        physical_params = result.parameters[2 * n_angles :]
    elif n_params == (n_physical + 2):
        logger.warning(
            "Solver returned scalar contrast/offset (parameter count %d). Expanding "
            "scalars across %d filtered angles for result saving.",
            n_params,
            n_angles,
        )
        scalar_per_angle_expansion = True
        scalar_contrast = float(result.parameters[0])
        scalar_offset = float(result.parameters[1])
        fitted_contrasts = np.full(n_angles, scalar_contrast, dtype=float)
        fitted_offsets = np.full(n_angles, scalar_offset, dtype=float)
        physical_params = result.parameters[2:]
    else:
        raise ValueError(
            f"Parameter count mismatch! Expected {expected_params_per_angle} "
            f"(2×{n_angles} scaling + {n_physical} physical), got {n_params}. "
            f"Per-angle scaling is REQUIRED in v2.4.0+"
        )

    logger.info(
        f"Per-angle scaling: {n_angles} angles, using FITTED scaling parameters from NLSQ optimization"
    )
    logger.debug(
        f"Extracted fitted parameters - "
        f"contrasts: mean={np.mean(fitted_contrasts):.4f}, "
        f"offsets: mean={np.mean(fitted_offsets):.4f}"
    )
    logger.debug(
        f"DEBUG: result.parameters shape: {result.parameters.shape}, "
        f"n_angles={n_angles}, 2*n_angles={2 * n_angles}"
    )
    logger.debug(f"DEBUG: physical_params extracted = {physical_params}")

    # Extract metadata with defaults
    L = metadata["L"]
    dt = metadata.get("dt")
    if dt is None:
        dt = 0.1  # Default time resolution in seconds
        logger.debug(f"Using default dt = {dt}s (not found in config)")
    q = metadata["q"]

    if q is None:
        raise ValueError("q (wavevector) is required but was not found")

    logger.info(
        f"Computing theoretical fits for {len(phi_angles)} angles using L={L:.1f} Å, q={q:.6f} Å⁻¹"
    )
    logger.info(
        "Using FITTED per-angle scaling parameters (contrast, offset) from NLSQ optimization"
    )
    logger.info(
        "Diagonal correction will be applied to theoretical fits to match experimental data processing"
    )

    # Sequential per-angle computation
    c2_theoretical_raw_list = []
    c2_theoretical_fitted = []
    solver_surface = []
    per_angle_scaling_posthoc = []  # Preserve legacy behavior
    solver_scaling = np.column_stack((fitted_contrasts, fitted_offsets))

    for i, phi_angle in enumerate(phi_angles):
        # Convert to JAX arrays
        phi_jax = jnp.array([float(phi_angle)])
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        params_jax = jnp.array(physical_params)

        # ✅ FIXED (Nov 11, 2025): Compute RAW theory WITHOUT scaling
        # NLSQ optimization minimizes weighted residuals (c2_exp - c2_theory)/sigma,
        # so fitted contrast/offset are optimized for residuals, NOT for absolute scale matching.
        # For visualization, we compute raw theory and use lstsq to find scaling that maps
        # theory → experiment (following old working version approach).
        # Formula: c₂_raw = 1.0 + 1.0 × c₁²  (normalized baseline without experimental scaling)
        c2_theory_raw = compute_g2_scaled(
            params=params_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi=phi_jax,
            q=float(q),
            L=float(L),
            contrast=1.0,  # ✅ No contrast scaling for raw theory
            offset=1.0,  # ✅ Normalized baseline (homodyne c2 baseline)
            dt=float(dt),
        )

        # Convert to NumPy and squeeze out extra dimension (phi axis)
        c2_theory_raw_np = np.asarray(c2_theory_raw)
        if c2_theory_raw_np.ndim == 3:
            c2_theory_raw_np = c2_theory_raw_np[0]  # Remove phi dimension (size 1)

        # Apply diagonal correction to match experimental data processing
        # This fixes the constant diagonal issue in theoretical model (c1(t,t) = 1 always)

        # Store raw theory
        c2_theoretical_raw_list.append(c2_theory_raw_np)

        if include_solver_surface:
            # Evaluate solver surface using original per-angle contrast/offset
            c2_solver = compute_g2_scaled(
                params=params_jax,
                t1=t1_jax,
                t2=t2_jax,
                phi=phi_jax,
                q=float(q),
                L=float(L),
                contrast=float(fitted_contrasts[i]),
                offset=float(fitted_offsets[i]),
                dt=float(dt),
            )
            c2_solver_np = np.asarray(c2_solver)
            if c2_solver_np.ndim == 3:
                c2_solver_np = c2_solver_np[0]
            solver_surface.append(c2_solver_np)

        # ✅ POST-HOC LEAST-SQUARES SCALING (Nov 11, 2025)
        # Find optimal contrast/offset that maps theory → experiment for visualization
        # Solve: c2_exp = contrast * c2_theory_raw + offset (per-angle)
        # This matches the old working version approach (homodyne-analysis)
        # JAX-first implementation for consistency with codebase
        theory_flat_jax = jnp.array(c2_theory_raw_np.flatten())
        exp_flat_jax = jnp.array(c2_exp[i].flatten())

        # Build design matrix A = [theory, ones] using JAX
        A_jax = jnp.column_stack([theory_flat_jax, jnp.ones_like(theory_flat_jax)])
        # Solve: A @ [contrast, offset] = exp using JAX lstsq
        solution_jax, _, _, _ = jnp.linalg.lstsq(A_jax, exp_flat_jax, rcond=None)
        contrast_lstsq = float(solution_jax[0])
        offset_lstsq = float(solution_jax[1])

        # Apply lstsq scaling (keep as NumPy for storage)
        c2_theoretical_scaled_angle = contrast_lstsq * c2_theory_raw_np + offset_lstsq
        c2_theoretical_fitted.append(c2_theoretical_scaled_angle)

        # Store lstsq scaling parameters (for visualization, not from optimization)
        per_angle_scaling_posthoc.append([contrast_lstsq, offset_lstsq])

        # Log scaling parameters
        logger.debug(
            f"Angle {phi_angle:.1f}°: lstsq contrast={contrast_lstsq:.4f}, offset={offset_lstsq:.4f} "
            f"(post-hoc for visualization)"
        )

    # Stack arrays
    c2_theoretical_raw = np.array(
        c2_theoretical_raw_list
    )  # Raw theory (contrast=1.0, offset=1.0)
    c2_theoretical_fitted = np.array(c2_theoretical_fitted)  # Scaled via lstsq
    c2_solver_surface = (
        np.array(solver_surface) if include_solver_surface and solver_surface else None
    )
    per_angle_scaling = np.array(per_angle_scaling_posthoc)

    # Scaled version is the lstsq-fitted result for visualization
    c2_theoretical_scaled = c2_theoretical_fitted

    # Compute residuals
    residuals = c2_exp - c2_theoretical_scaled

    logger.info(
        f"Computed theoretical fits for {len(phi_angles)} angles (sequential computation, diagonal corrected, lstsq scaled)",
    )

    return {
        "c2_theoretical_raw": c2_theoretical_raw,
        "c2_theoretical_scaled": c2_theoretical_scaled,
        "c2_solver_scaled": c2_solver_surface,
        "per_angle_scaling": per_angle_scaling,
        "per_angle_scaling_solver": solver_scaling,
        "residuals": residuals,
        "scalar_per_angle_expansion": scalar_per_angle_expansion,
    }


def _json_safe(value: Any) -> Any:
    """Convert nested objects to JSON-serializable primitives.

    This is a wrapper that delegates to homodyne.io.json_utils.
    """
    return _io_json_safe(value)


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

    # Apply phi filtering to data (if enabled in config)
    # This ensures saved data and plots respect phi_filtering configuration
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
    fits_dict = _compute_nlsq_fits(
        result,
        filtered_data,
        metadata,
        analysis_mode=analysis_mode,  # Pass analysis_mode to fix parameter count detection
        include_solver_surface=True,
    )
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

    logger.info(f"✓ NLSQ results saved successfully to {nlsq_dir}")
    logger.info("  - 3 JSON files (parameters, analysis results, convergence metrics)")
    logger.info("  - 1 NPZ file (10 arrays: experimental + theoretical + residuals)")

    # Step 8b: Persist diagnostics payload if available
    diagnostics_payload = getattr(result, "nlsq_diagnostics", None)
    if diagnostics_payload:
        diagnostics_file = nlsq_dir / "diagnostics.json"
        try:
            with open(diagnostics_file, "w") as f:
                json.dump(_json_safe(diagnostics_payload), f, indent=2)
            logger.info(f"Saved diagnostics to {diagnostics_file}")
        except TypeError as exc:
            logger.warning(
                "Failed to save diagnostics.json (%s). Payload keys: %s",
                exc,
                list(diagnostics_payload.keys()),
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
            config=config,  # Pass config for preview_mode setting
            c2_solver_scaled=fits_dict["c2_solver_scaled"],  # Use FITTED solver surface
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
    *,
    c2_solver_scaled: np.ndarray | None = None,
) -> None:
    """Generate 3-panel heatmap plots for NLSQ fit visualization.

    This is a wrapper that delegates to homodyne.viz.nlsq_plots.
    """
    _viz_generate_nlsq_plots(
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        c2_theoretical_scaled=c2_theoretical_scaled,
        residuals=residuals,
        t1=t1,
        t2=t2,
        output_dir=output_dir,
        config=config,
        use_datashader=use_datashader,
        parallel=parallel,
        c2_solver_scaled=c2_solver_scaled,
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

            if (
                hasattr(result, "samples_contrast")
                and result.samples_contrast is not None
            ):
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
                samples_list.insert(
                    1 if hasattr(result, "samples_contrast") else 0, offset_samples
                )

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
        if (
            hasattr(result, "effective_sample_size")
            and result.effective_sample_size is not None
        ):
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
    if (
        method_name == "cmc"
        and hasattr(result, "per_shard_diagnostics")
        and result.per_shard_diagnostics
    ):
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

        # Apply phi filtering to data (if enabled in config)
        # This ensures plots respect phi_filtering configuration
        filtered_data = _apply_angle_filtering_for_optimization(data, config)
        logger.debug(
            f"Applied phi filtering for MCMC plotting: "
            f"{len(filtered_data['phi_angles_list'])} angles selected"
        )

        # Compute theoretical C2 using posterior mean parameters
        c2_theoretical_scaled = _compute_theoretical_c2_from_mcmc(
            result, filtered_data, config
        )

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

    logger.info(f"✓ {method_name.upper()} results saved successfully to {method_dir}")
    if (
        method_name == "cmc"
        and hasattr(result, "per_shard_diagnostics")
        and result.per_shard_diagnostics
    ):
        logger.info(
            "  - 4 JSON files (parameters, analysis results, diagnostics, shard diagnostics)"
        )
    else:
        logger.info("  - 3 JSON files (parameters, analysis results, diagnostics)")
    logger.info("  - 1 NPZ file (posterior samples)")


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
    if analysis_mode == "static":
        return ["D0", "alpha", "D_offset"]
    elif analysis_mode == "laminar_flow":
        return [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
    else:
        logger.warning(
            f"Unknown analysis mode: {analysis_mode}, assuming static_isotropic"
        )
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

    # Log parameter values for debugging
    logger.info("Computing theoretical C2 with posterior means:")
    logger.info(f"  Contrast: {contrast:.6f}")
    logger.info(f"  Offset: {offset:.6f}")
    logger.info(
        f"  Physical params: D0={mean_params[0]:.2f}, alpha={mean_params[1]:.4f}, D_offset={mean_params[2]:.4f}"
    )

    # Validate parameters for reasonable theoretical prediction
    if contrast < 0.05:
        logger.warning(
            f"Very small contrast ({contrast:.4f} < 0.05) may produce nearly constant c2_theory. "
            "This suggests poor MCMC convergence or inappropriate initial values."
        )
    if mean_params[0] >= 99990:  # Near D0 upper bound
        logger.warning(
            f"D0 ({mean_params[0]:.1f}) near upper bound (100000). "
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
    config_dict = config.get_config() if hasattr(config, "get_config") else config
    _analysis_mode = config_dict.get("analysis_mode", "static_isotropic")  # noqa: F841

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

    # Validate theoretical prediction quality
    c2_min = float(np.min(c2_theoretical_scaled))
    c2_max = float(np.max(c2_theoretical_scaled))
    c2_range = c2_max - c2_min
    logger.info(
        f"Theoretical C2 range: [{c2_min:.6f}, {c2_max:.6f}], variation: {c2_range:.6f}"
    )

    if c2_range < 0.01:
        logger.warning(
            f"Theoretical C2 has very low variation ({c2_range:.6f} < 0.01). "
            f"The model prediction is nearly constant (c2 ≈ {c2_min:.4f}). "
            "This indicates:\n"
            "  1. Poor MCMC convergence to local minimum\n"
            "  2. Inappropriate initial parameter values\n"
            "  3. Physical parameters may have hit bounds\n"
            "Recommendations:\n"
            "  - Run NLSQ first to get better initial values\n"
            "  - Check parameter bounds (especially D0 upper limit)\n"
            "  - Verify initial_parameters.values in config are reasonable"
        )

    return c2_theoretical_scaled


def _worker_init_cpu_only():
    """Initialize worker process with CPU-only mode.

    When spawning multiple workers for parallel plotting, we ensure each worker
    uses CPU-only execution. Plotting is CPU-bound (Datashader/matplotlib).

    Sets environment variables for CPU-only JAX execution:
    - JAX_PLATFORMS: Tells JAX to only use CPU platform
    - CUDA_VISIBLE_DEVICES: Disables any CUDA device access
    - XLA_PYTHON_CLIENT_PREALLOCATE: Prevents memory preallocation
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
    import numpy as np

    from homodyne.viz.datashader_backend import plot_c2_comparison_fast

    (
        i,
        phi_angles,
        c2_exp,
        c2_fit,
        residuals,
        t1,
        t2,
        output_dir,
        width,
        height,
        color_options,
    ) = args
    phi = phi_angles[i]
    output_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"

    # Convert JAX GPU arrays to numpy CPU arrays before plotting
    # This fixes: "BufferError: INVALID_ARGUMENT: Python buffer protocol is only defined for CPU buffers"
    # Numba (used by Datashader) requires CPU arrays for buffer protocol access
    c2_exp_cpu = np.asarray(c2_exp)
    c2_fit_cpu = np.asarray(c2_fit)
    residuals_cpu = np.asarray(residuals)
    t1_cpu = np.asarray(t1)
    t2_cpu = np.asarray(t2)

    # Use Datashader fast plotting with higher resolution
    plot_c2_comparison_fast(
        c2_exp_cpu,
        c2_fit_cpu,
        residuals_cpu,
        t1_cpu,
        t2_cpu,
        output_file,
        phi_angle=phi,
        width=width,
        height=height,
        **color_options,
    )

    return output_file


def _generate_plots_datashader(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_fit_display: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    parallel: bool = True,
    width: int = 1200,
    height: int = 1200,
    color_options: dict[str, Any] | None = None,
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
                c2_fit_display[i],
                residuals[i],
                t1,
                t2,
                output_dir,
                width,
                height,
                color_options or {},
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
                    c2_fit_display[i],
                    residuals[i],
                    t1,
                    t2,
                    output_dir,
                    width,
                    height,
                    color_options or {},
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
                c2_fit_display[i],
                residuals[i],
                t1,
                t2,
                output_dir,
                width,
                height,
                color_options or {},
            )
            _plot_single_angle_datashader(args)

        logger.info(f"✓ Generated {len(phi_angles)} heatmap plots (sequential)")


def _generate_plots_matplotlib(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    c2_fit_display: np.ndarray,
    residuals: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    output_dir: Path,
    color_options: dict[str, Any] | None = None,
) -> None:
    """Generate plots using matplotlib backend (original implementation)."""
    logger.info(f"Generating heatmap plots for {len(phi_angles)} angles")

    for i, phi in enumerate(phi_angles):
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Experimental data with fixed color scale [1.0, 1.5]
        # Transpose because data is structured as c2[t1_index, t2_index] with indexing="ij"
        # but we want x-axis=t1, y-axis=t2 for display
        vmin_use, vmax_use = _resolve_color_limits(
            c2_exp[i],
            color_options,
        )

        im0 = axes[0].imshow(
            c2_exp[i].T,
            origin="lower",
            aspect="equal",
            cmap="jet",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
            vmin=vmin_use,
            vmax=vmax_use,
        )
        axes[0].set_title(f"Experimental C₂ (φ={phi:.1f}°)", fontsize=12)
        axes[0].set_xlabel("t₁ (s)", fontsize=10)
        axes[0].set_ylabel("t₂ (s)", fontsize=10)
        cbar0 = plt.colorbar(im0, ax=axes[0], label="C₂(t₁,t₂)")
        cbar0.ax.tick_params(labelsize=8)

        # Panel 2: Theoretical fit with fixed color scale [1.0, 1.5]
        im1 = axes[1].imshow(
            c2_fit_display[i].T,
            origin="lower",
            aspect="equal",
            cmap="jet",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
            vmin=vmin_use,
            vmax=vmax_use,
        )
        axes[1].set_title(f"Classical Fit (φ={phi:.1f}°)", fontsize=12)
        axes[1].set_xlabel("t₁ (s)", fontsize=10)
        axes[1].set_ylabel("t₂ (s)", fontsize=10)
        cbar1 = plt.colorbar(im1, ax=axes[1], label="C₂(t₁,t₂)")
        cbar1.ax.tick_params(labelsize=8)

        # Panel 3: Residuals using actual min/max
        residual_min = float(np.min(residuals[i]))
        residual_max = float(np.max(residuals[i]))
        im2 = axes[2].imshow(
            residuals[i].T,
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=residual_min,
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


def _resolve_color_limits(
    matrix: np.ndarray,
    color_options: dict[str, Any] | None,
) -> tuple[float, float]:
    opts = color_options or {}
    adaptive = opts.get("adaptive", False)
    vmin = opts.get("vmin")
    vmax = opts.get("vmax")
    percentile_min = opts.get("percentile_min", 1.0)
    percentile_max = opts.get("percentile_max", 99.0)

    if adaptive and matrix.size > 0:
        if vmin is None:
            vmin = float(np.percentile(matrix, percentile_min))
        if vmax is None:
            vmax = float(np.percentile(matrix, percentile_max))

    if vmin is None:
        vmin = 1.0
    if vmax is None:
        vmax = 1.5

    return vmin, vmax


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
