"""
Command Dispatcher for Homodyne v2 CLI
======================================

Handles command execution and coordination between CLI arguments,
configuration, and optimization methods.
"""

import time
from typing import Any

from homodyne.cli.args_parser import validate_args
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Import core modules with fallback
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.data import load_xpcs_data
    from homodyne.device import configure_optimal_device
    from homodyne.optimization import fit_mcmc_jax, fit_nlsq_jax

    HAS_CORE_MODULES = True
except ImportError as e:
    HAS_CORE_MODULES = False
    logger.error(f"Core modules not available: {e}")


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

        # Configure logging based on args
        _configure_logging(args)

        # Configure device (CPU/GPU)
        device_config = _configure_device(args)

        # Load configuration
        config = _load_configuration(args)

        # Load data
        data = _load_data(args, config)

        # Run optimization
        result = _run_optimization(args, config, data)

        # Save results
        _save_results(args, result, device_config)

        logger.info("Analysis completed successfully")
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


def _configure_logging(args) -> None:
    """Configure logging based on CLI arguments."""
    import logging

    if args.quiet:
        # Only show errors
        logging.getLogger("homodyne").setLevel(logging.ERROR)
    elif args.verbose:
        # Show debug information
        logging.getLogger("homodyne").setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        # Default: show info and above
        logging.getLogger("homodyne").setLevel(logging.INFO)


def _configure_device(args) -> dict[str, Any]:
    """Configure optimal device based on CLI arguments."""
    logger.info("Configuring computational device...")

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
    """Load experimental data."""
    logger.info("Loading experimental data...")

    # Get data file path from args or config
    data_file = None
    if args.data_file:
        data_file = str(args.data_file)
    elif hasattr(config, "config") and config.config:
        data_file = config.config.get("experimental_data", {}).get("file_path")

    if not data_file:
        raise ValueError(
            "No data file specified. Use --data-file or configure in YAML file."
        )

    # Load data using the data module
    try:
        data = load_xpcs_data(data_file)
        logger.info(f"✓ Data loaded: {len(data.get('c2_exp', []))} data points")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_file}: {e}")


def _run_optimization(args, config: ConfigManager, data: dict[str, Any]) -> Any:
    """Run the specified optimization method."""
    method = args.method
    logger.info(f"Running {method.upper()} optimization...")

    start_time = time.perf_counter()

    try:
        if method == "nlsq":
            result = fit_nlsq_jax(data, config)
        elif method == "mcmc":
            # Convert data format for MCMC if needed
            mcmc_data = data["c2_exp"]
            result = fit_mcmc_jax(
                mcmc_data,
                t1=data.get("t1"),
                t2=data.get("t2"),
                phi=data.get("phi_angles_list"),
                q=(
                    data.get("wavevector_q_list", [1.0])[0]
                    if data.get("wavevector_q_list")
                    else 1.0
                ),
                L=100.0,  # Default sample-detector distance
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
