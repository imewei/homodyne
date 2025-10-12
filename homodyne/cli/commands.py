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
        plot_exp = getattr(args, 'plot_experimental_data', False)
        plot_sim = getattr(args, 'plot_simulated_data', False)
        save_plots = getattr(args, 'save_plots', False)

        # Simulated data plotting doesn't need experimental data or optimization
        if plot_sim and not plot_exp and not save_plots:
            logger.info("Plotting simulated data only (skipping data loading and optimization)")
            # Skip data loading for pure simulated data mode
            # Extract config dictionary from ConfigManager
            config_dict = config.get_config() if hasattr(config, 'get_config') else config
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
        config_dict = config.get_config() if hasattr(config, 'get_config') else config
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
    file_handler = logging.FileHandler(log_file, mode='w')
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
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
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
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
    os.environ['XLA_FLAGS'] = (
        os.environ.get('XLA_FLAGS', '') +
        ' --xla_gpu_autotune_level=0'
        ' --xla_gpu_deterministic_ops=true'
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
                logger.debug(f"Using current directory for data file: {data_file_path.name}")

            temp_config = {
                "experimental_data": {
                    "data_folder_path": str(parent_dir),
                    "data_file_name": data_file_path.name,
                },
                "analyzer_parameters": config.config.get("analyzer_parameters", {
                    "dt": 0.1,
                    "start_frame": 1,
                    "end_frame": -1
                })
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


def _handle_plotting(args, result: Any, data: dict[str, Any], config: dict[str, Any] = None) -> None:
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
    plot_exp = getattr(args, 'plot_experimental_data', False)
    plot_sim = getattr(args, 'plot_simulated_data', False)
    save_plots = getattr(args, 'save_plots', False)

    if not (save_plots or plot_exp or plot_sim):
        return

    # Check for plotting dependencies
    try:
        import matplotlib.pyplot as plt
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
            _plot_experimental_data(data, plots_dir)
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
            contrast = getattr(args, 'contrast', 0.3)
            offset = getattr(args, 'offset', 1.0)
            phi_angles_str = getattr(args, 'phi_angles', None)

            _plot_simulated_data(config, contrast, offset, phi_angles_str, plots_dir)
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


def _plot_experimental_data(data: dict[str, Any], plots_dir) -> None:
    """Generate validation plots of experimental data."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Basic experimental data visualization
    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        logger.warning("No experimental data to plot")
        return

    # Handle different data shapes
    if c2_exp.ndim == 3:
        # Data shape: (n_phi, n_t1, n_t2)
        # Plot first few phi angles
        n_phi = min(4, c2_exp.shape[0])
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        for idx in range(n_phi):
            im = axes[idx].imshow(c2_exp[idx], aspect='auto', cmap='viridis', origin='lower')
            axes[idx].set_xlabel("t₂ Index")
            axes[idx].set_ylabel("t₁ Index")
            axes[idx].set_title(f"C₂ at φ index {idx}")
            plt.colorbar(im, ax=axes[idx], label='C₂')

        plt.tight_layout()
        plt.savefig(plots_dir / "experimental_data_phi_slices.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Plot diagonal (t1=t2) for all phi angles
        fig, ax = plt.subplots(figsize=(10, 6))
        for idx in range(min(10, c2_exp.shape[0])):
            diagonal = np.diag(c2_exp[idx])
            ax.plot(diagonal, label=f'φ{idx}', alpha=0.7)
        ax.set_xlabel("Time Index")
        ax.set_ylabel("C₂(t, t)")
        ax.set_title("C₂ Diagonal (t₁=t₂) for Different φ Angles")
        ax.legend(ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "experimental_data_diagonal.png", dpi=150, bbox_inches="tight")
        plt.close()

    elif c2_exp.ndim == 2:
        # 2D data: single correlation matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(c2_exp, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='C₂(t₁,t₂)')
        ax.set_xlabel("t₂ Index")
        ax.set_ylabel("t₁ Index")
        ax.set_title("Experimental C₂ Data")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
        plt.close()

    elif c2_exp.ndim == 1:
        # 1D data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(c2_exp, marker='o', linestyle='-', alpha=0.7)
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
) -> None:
    """Generate plots of simulated/theoretical data."""
    import matplotlib.pyplot as plt
    from homodyne.core.models import CombinedModel
    import jax.numpy as jnp

    logger.info(f"Generating simulated data plots (contrast={contrast:.3f}, offset={offset:.3f})")

    # Determine analysis mode
    analysis_mode = config.get("analysis_mode", "static_isotropic")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Create model
    model = CombinedModel(analysis_mode)

    # Get parameters from configuration
    initial_params = config.get("optimization", {}).get("initial_parameters", {})

    if analysis_mode.startswith("static"):
        # Static mode: 3 parameters
        params = jnp.array([
            initial_params.get("D0", 100.0),
            initial_params.get("alpha", -0.5),
            initial_params.get("D_offset", 0.0),
        ])
    else:
        # Laminar flow: 7 parameters
        params = jnp.array([
            initial_params.get("D0", 100.0),
            initial_params.get("alpha", -0.5),
            initial_params.get("D_offset", 0.0),
            initial_params.get("gamma_dot_t0", 0.01),
            initial_params.get("beta", 0.5),
            initial_params.get("gamma_dot_t_offset", 0.0),
            initial_params.get("phi0", 0.0),
        ])

    logger.debug(f"Using parameters: {dict(zip(model.parameter_names, params))}")

    # Parse phi angles
    if phi_angles_str:
        phi_degrees = np.array([float(x.strip()) for x in phi_angles_str.split(",")])
        phi = jnp.radians(phi_degrees)
    else:
        # Default: 8 evenly spaced angles from 0 to 180 degrees
        phi_degrees = np.linspace(0, 180, 8)
        phi = jnp.radians(phi_degrees)

    logger.debug(f"Using {len(phi)} phi angles: {phi_degrees}")

    # Generate time arrays (100 points from 0.1 to 100)
    analyzer_params = config.get("analyzer_parameters", {})
    dt = analyzer_params.get("dt", 0.1)
    n_time_points = 100

    t_vals = jnp.linspace(1, n_time_points, n_time_points) * dt
    t1_grid, t2_grid = jnp.meshgrid(t_vals, t_vals, indexing="ij")

    # Get wavevector and gap size from config
    experimental_data = config.get("experimental_data", {})
    q = experimental_data.get("wavevector_q", 0.005)
    L = experimental_data.get("gap_size", 80.0)

    logger.info(f"Generating theoretical C₂ with q={q:.4f} Å⁻¹, L={L:.1f} μm")

    # Generate simulated C₂ for each phi angle
    c2_simulated = []

    for i, phi_val in enumerate(phi):
        phi_array = jnp.array([phi_val])

        # Compute g2 for this phi angle
        c2_phi = model.compute_g2(
            params,
            t1_grid,
            t2_grid,
            phi_array,
            q,
            L,
            contrast,
            offset,
        )

        # Extract the 2D array (remove phi dimension)
        c2_simulated.append(np.array(c2_phi[0]))

    c2_simulated = np.array(c2_simulated)  # Shape: (n_phi, n_t, n_t)

    logger.info(f"Generated simulated C₂ with shape: {c2_simulated.shape}")

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
        ax.plot(t_vals, diagonal, label=f"φ={phi_degrees[idx]:.1f}°", alpha=0.7, linewidth=2)

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
        axes[0].plot(c2_exp, marker='o', linestyle='-', alpha=0.7, label="Experimental")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("C₂")
    else:
        im0 = axes[0].imshow(c2_exp, aspect='auto', cmap='viridis')
        plt.colorbar(im0, ax=axes[0], label='C₂')
        axes[0].set_xlabel("t₂ Index")
        axes[0].set_ylabel("φ Index")
    axes[0].set_title("Experimental Data")
    axes[0].grid(True, alpha=0.3)

    # Plot fit results
    axes[1].text(
        0.5, 0.5,
        "Fit visualization\nrequires full\nplotting backend",
        ha='center', va='center', fontsize=14
    )
    axes[1].set_title("Fit Results")
    axes[1].axis('off')

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
