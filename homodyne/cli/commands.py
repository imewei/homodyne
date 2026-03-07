"""Command Dispatcher for Homodyne CLI
======================================

Handles command execution and coordination between CLI arguments,
configuration, and optimization methods.

This module serves as the main orchestrator and re-export hub.
Implementation details are in focused submodules:
- config_handling: Device config, config loading, CLI overrides
- data_pipeline: Data loading, t=0 exclusion, angle filtering, MCMC pooling
- optimization_runner: NLSQ/CMC optimization, warm-start
- result_saving: JSON/NPZ saving for NLSQ and MCMC results
- plot_dispatch: Plotting dispatch for experimental/simulated data

Functions that are mock-patched by tests via @patch("homodyne.cli.commands.X")
remain in this module to preserve test compatibility.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from homodyne.cli.args_parser import validate_args

# Re-export from config_handling
from homodyne.cli.config_handling import (  # noqa: F401
    _apply_cli_overrides,
    _build_mcmc_runtime_kwargs,
    _configure_device,
    _get_default_config,
    _load_configuration,
)

# Re-export from data_pipeline
from homodyne.cli.data_pipeline import (  # noqa: F401
    COMMON_XPCS_ANGLES,
    _apply_angle_filtering,
    _apply_angle_filtering_for_optimization,
    _exclude_t0_from_analysis,
    _pool_mcmc_data,
    _prepare_cmc_config,
)

# Re-export from optimization_runner
from homodyne.cli.optimization_runner import (  # noqa: F401
    _resolve_nlsq_warmstart,
    _run_nlsq_optimization,
    load_nlsq_result_from_file,
)

# Re-export from plot_dispatch
from homodyne.cli.plot_dispatch import (  # noqa: F401
    _apply_angle_filtering_for_plot,
    _handle_plotting,
    generate_nlsq_plots,
)

# Re-export from result_saving
from homodyne.cli.result_saving import (  # noqa: F401
    _compute_theoretical_c2_from_mcmc,
    _extract_nlsq_metadata,
    _json_safe,
    _json_serializer,
    _prepare_parameter_data,
    _save_nlsq_json_files,
    _save_nlsq_npz_file,
    _save_results,
    save_mcmc_results,
    save_nlsq_results,
)
from homodyne.config.parameter_space import ParameterSpace
from homodyne.data.angle_filtering import (
    angle_in_range as _data_angle_in_range,
)
from homodyne.data.angle_filtering import (
    normalize_angle_to_symmetric_range as _data_normalize_angle_to_symmetric_range,
)
from homodyne.utils.logging import (
    AnalysisSummaryLogger,
    configure_logging,
    get_logger,
    log_exception,
    log_phase,
)

logger = get_logger(__name__)

# Import core modules with fallback.
# These module-level names are also used as mock patch targets by tests
# (e.g. @patch("homodyne.cli.commands.XPCSDataLoader")), so they must
# remain importable from this module.
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.data.xpcs_loader import XPCSDataLoader
    from homodyne.optimization import fit_mcmc_jax

    HAS_CORE_MODULES = True
    HAS_XPCS_LOADER = True
except ImportError as e:
    HAS_CORE_MODULES = False
    HAS_XPCS_LOADER = False
    logger.error(f"Core modules not available: {e}")

    # Fallback for missing XPCSDataLoader
    class XPCSDataLoader:  # type: ignore[no-redef]
        """Placeholder when XPCSDataLoader is not available."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("XPCSDataLoader not available")


def normalize_angle_to_symmetric_range(
    angle: float | NDArray[np.floating[Any]],
) -> float | NDArray[np.floating[Any]]:
    """Normalize angle(s) to [-180, 180] range.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_normalize_angle_to_symmetric_range(angle)


def _angle_in_range(angle: float, min_angle: float, max_angle: float) -> bool:
    """Check if angle is in range, accounting for wrap-around at +/-180.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_angle_in_range(angle, min_angle, max_angle)


def dispatch_command(args: argparse.Namespace) -> dict[str, Any]:
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
    logger.info(f"[CLI] Dispatching homodyne analysis command (run_id={run_id})")

    # Log resolved command-line arguments at DEBUG level (T023)
    logger.debug(f"[CLI] Resolved arguments: {vars(args)}")

    # Validate arguments
    if not validate_args(args):
        return {"success": False, "error": "Invalid command-line arguments"}

    if not HAS_CORE_MODULES:
        return {
            "success": False,
            "error": "Core modules not available. Please check installation.",
        }

    # Initialize analysis summary logger (T024)
    cli_mode = "laminar_flow" if getattr(args, "laminar_flow", False) else "static"
    if getattr(args, "static_mode", False):
        cli_mode = "static_isotropic"
    summary = AnalysisSummaryLogger(run_id=run_id, analysis_mode=cli_mode)

    try:
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Load configuration (T019)
        summary.start_phase("config_loading")
        with log_phase("config_loading"):
            config = _load_configuration(args)

            # Configure logging using config + CLI verbosity flags
            config_dict = (
                config.get_config() if hasattr(config, "get_config") else config
            )
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
                logger.info(f"[CLI] Log file created: {log_file}")
                summary.add_output_file(log_file)
        summary.end_phase("config_loading")

        # Update analysis mode from loaded config (T055)
        config_dict = config.get_config() if hasattr(config, "get_config") else config
        if isinstance(config_dict, dict):
            config_analysis_mode = config_dict.get("analysis_mode", cli_mode)
            if config_analysis_mode != summary.analysis_mode:
                logger.debug(
                    f"[CLI] Updated analysis_mode from '{summary.analysis_mode}' "
                    f"to '{config_analysis_mode}' (from config)"
                )
                summary.analysis_mode = config_analysis_mode

        # Configure device (CPU/GPU)
        device_config = _configure_device(args)

        # Check if only simulated data plotting is requested
        plot_exp = getattr(args, "plot_experimental_data", False)
        plot_sim = getattr(args, "plot_simulated_data", False)
        save_plots = getattr(args, "save_plots", False)

        # Simulated data plotting doesn't need experimental data or optimization
        if plot_sim and not plot_exp and not save_plots:
            logger.info(
                "[CLI] Plotting simulated data only (skipping data loading and optimization)",
            )
            config_dict_plot: dict[str, Any] = (
                config.get_config()
                if hasattr(config, "get_config")
                else cast(dict[str, Any], config)
            )
            _handle_plotting(args, None, {}, config_dict_plot)
            summary.set_convergence_status("skipped_simulated_only")
            summary.log_summary(logger)
            return {
                "success": True,
                "result": None,
                "device_config": device_config,
                "output_dir": str(args.output_dir),
            }

        # Phase 2: Load data (T020)
        summary.start_phase("data_loading")
        with log_phase("data_loading", track_memory=True) as data_phase:
            data = _load_data(args, config)
        summary.end_phase("data_loading", memory_peak_gb=data_phase.memory_peak_gb)

        # Plot experimental data only (no optimization needed)
        plot_only = plot_exp and not save_plots and not plot_sim

        if plot_only:
            logger.info("[CLI] Plotting experimental data only (skipping optimization)")
            result = None
            summary.set_convergence_status("skipped_plot_only")
        else:
            # Phase 3: Run optimization (T021)
            summary.start_phase("optimization")
            with log_phase("optimization", track_memory=True) as opt_phase:
                result = _run_optimization(args, config, data)
            summary.end_phase("optimization", memory_peak_gb=opt_phase.memory_peak_gb)

            # Record optimization metrics
            if result is not None:
                is_cmc = (
                    callable(getattr(result, "is_cmc_result", None))
                    and result.is_cmc_result()
                )
                if hasattr(result, "chi_squared") and not is_cmc:
                    summary.record_metric("chi_squared", float(result.chi_squared))
                if hasattr(result, "n_iterations"):
                    summary.record_metric("n_iterations", float(result.n_iterations))
                if hasattr(result, "convergence_status"):
                    summary.set_convergence_status(result.convergence_status)
                elif hasattr(result, "converged"):
                    summary.set_convergence_status(
                        "converged" if result.converged else "not_converged"
                    )
                elif hasattr(result, "success"):
                    summary.set_convergence_status(
                        "converged" if result.success else "failed"
                    )

            # Phase 4: Save results (T022)
            summary.start_phase("result_saving")
            with log_phase("result_saving"):
                _save_results(args, result, device_config, data, config)
            summary.end_phase("result_saving")

        # Handle plotting options
        config_dict2: dict[str, Any] = (
            config.get_config()
            if hasattr(config, "get_config")
            else cast(dict[str, Any], config)
        )
        _handle_plotting(args, result, data, config_dict2)

        logger.info("[CLI] Analysis completed successfully")

        # Log analysis summary (T024)
        summary.log_summary(logger)

        # Summary message
        if log_file:
            logger.info(f"[CLI] Analysis log saved to: {log_file}")
        else:
            log_dir = args.output_dir / "logs"
            log_files = list(log_dir.glob("homodyne_analysis_*.log"))
            if log_files:
                logger.info(f"[CLI] Analysis log saved to: {log_files[-1]}")

        return {
            "success": True,
            "result": result,
            "device_config": device_config,
            "output_dir": str(args.output_dir),
            "summary": summary.as_dict(),
        }

    except Exception as e:
        log_exception(logger, e, context={"run_id": run_id, "phase": "dispatch"})
        summary.set_convergence_status("failed")
        summary.increment_error_count()
        return {"success": False, "error": str(e)}


def _load_data(args: argparse.Namespace, config: ConfigManager) -> dict[str, Any]:
    """Load experimental data using XPCSDataLoader.

    Uses XPCSDataLoader which properly handles the config format
    (data_folder_path + data_file_name) internally.
    """
    logger.info("Loading experimental data...")

    if not HAS_XPCS_LOADER:
        raise RuntimeError(
            "XPCSDataLoader not available. "
            "Please ensure homodyne.data module is properly installed",
        )

    try:
        if args.data_file:
            data_file_path = Path(args.data_file).resolve()

            parent_dir = data_file_path.parent
            if parent_dir == Path.cwd():
                logger.debug(
                    f"Using current directory for data file: {data_file_path.name}",
                )

            temp_config = {
                "experimental_data": {
                    "data_folder_path": str(parent_dir),
                    "data_file_name": data_file_path.name,
                },
                "analyzer_parameters": (
                    config.config.get("analyzer_parameters", {})
                    if hasattr(config, "config") and config.config is not None
                    else {"dt": 0.1, "start_frame": 1, "end_frame": -1}
                ),
            }
            logger.info(f"Loading data from CLI override: {data_file_path}")
            loader = XPCSDataLoader(config_dict=temp_config)
        else:
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

        data = loader.load_experimental_data()

        data_size = 0
        if "c2_exp" in data:
            c2_exp = data["c2_exp"]
            data_size = c2_exp.size if hasattr(c2_exp, "size") else len(c2_exp)

        logger.info(f"OK: Data loaded successfully: {data_size:,} data points")
        return data

    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise RuntimeError(f"Data file not found: {e}") from e
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise RuntimeError(f"Failed to load experimental data: {e}") from e


def _run_optimization(
    args: argparse.Namespace, config: ConfigManager, data: dict[str, Any]
) -> Any:
    """Run the specified optimization method."""
    method = args.method

    logger.info(f"Running {method.upper()} optimization...")

    start_time = time.perf_counter()

    # Apply angle filtering before optimization (if configured)
    filtered_data = _apply_angle_filtering_for_optimization(data, config)

    # CRITICAL FIX: Exclude t=0 from analysis to prevent D(t) singularity
    filtered_data = _exclude_t0_from_analysis(filtered_data)

    logger.debug("Using NLSQ native large dataset handling")

    try:
        if method == "nlsq":
            result = _run_nlsq_optimization(filtered_data, config, args)
        elif method == "cmc":
            cmc_config = _prepare_cmc_config(args, config)
            _backend_cfg = cmc_config["backend"]  # noqa: F841

            nlsq_result = _resolve_nlsq_warmstart(args, filtered_data, config)

            config_config_early = (
                config.config
                if hasattr(config, "config") and config.config is not None
                else {}
            )
            analysis_mode_early = cast(
                str, config_config_early.get("analysis_mode", "static_isotropic")
            )
            require_warmstart = cmc_config.get("validation", {}).get(
                "require_nlsq_warmstart", False
            )

            if (
                require_warmstart
                and nlsq_result is None
                and "laminar" in analysis_mode_early.lower()
            ):
                raise ValueError(
                    "CMC WARM-START REQUIRED: laminar_flow mode requires NLSQ warm-start "
                    "when require_nlsq_warmstart=True. Remove --no-nlsq-warmstart flag or set "
                    "validation.require_nlsq_warmstart=false in CMC config."
                )

            logger.info(f"Method: {method.upper()} (Consensus Monte Carlo)")

            sharding = cmc_config.get("sharding", {})
            backend = cmc_config.get("backend", {})

            if isinstance(backend, str):
                backend_str = backend
                backend_config = cmc_config.get("backend_config", {})
                parallel_backend = (
                    backend_config.get("name", "auto") if backend_config else "auto"
                )
                backend_display = f"{backend_str}/{parallel_backend}"
            else:
                backend_display = backend.get("name", "auto")

            logger.debug(
                f"CMC sharding: strategy={sharding.get('strategy', 'auto')}, "
                f"num_shards={sharding.get('num_shards', 'auto')}, "
                f"backend={backend_display}",
            )

            pooled = _pool_mcmc_data(filtered_data)
            mcmc_data = pooled["mcmc_data"]
            t1_pooled = pooled["t1_pooled"]
            t2_pooled = pooled["t2_pooled"]
            phi_pooled = pooled["phi_pooled"]
            _n_phi = pooled["n_phi"]  # noqa: F841
            _n_t = pooled["n_t"]  # noqa: F841

            initial_values = (
                config.get_initial_parameters()
                if hasattr(config, "get_initial_parameters")
                else {}
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

            config_config = (
                config.config
                if hasattr(config, "config") and config.config is not None
                else {}
            )
            analysis_mode_str = cast(
                str, config_config.get("analysis_mode", "static_isotropic")
            )

            parameter_space = ParameterSpace.from_config(
                config_dict=config_config,
                analysis_mode=analysis_mode_str,
            )
            logger.debug(
                f"Created ParameterSpace with config for {analysis_mode_str} mode"
            )

            mcmc_runtime_kwargs = _build_mcmc_runtime_kwargs(args, config)

            result = fit_mcmc_jax(
                mcmc_data,
                t1=t1_pooled,
                t2=t2_pooled,
                phi=phi_pooled,
                q=(
                    filtered_data.get("wavevector_q_list", [1.0])[0]
                    if (
                        filtered_data.get("wavevector_q_list") is not None
                        and len(filtered_data.get("wavevector_q_list", [])) > 0
                    )
                    else 1.0
                ),
                L=float(
                    config_config.get("analyzer_parameters", {})
                    .get("geometry", {})
                    .get("stator_rotor_gap", 2000000.0)
                ),
                analysis_mode=cast(
                    str, config_config.get("analysis_mode", "static_isotropic")
                ),
                method=method,
                cmc_config=cmc_config,
                initial_values=initial_values,
                parameter_space=parameter_space,
                dt=config_config.get("analyzer_parameters", {}).get("dt"),
                nlsq_result=nlsq_result,
                **mcmc_runtime_kwargs,
            )

            if hasattr(result, "inference_data") and result.inference_data is not None:
                analysis_mode_for_plot = cast(
                    str, config_config.get("analysis_mode", "static_isotropic")
                )
                _generate_cmc_diagnostic_plots(
                    result, args.output_dir, analysis_mode_for_plot
                )
            else:
                logger.warning(
                    "Cannot generate ArviZ diagnostic plots: inference_data not available"
                )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        optimization_time = time.perf_counter() - start_time
        logger.info(
            f"OK: {method.upper()} optimization completed in {optimization_time:.3f}s",
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
    """Generate CMC/MCMC diagnostic plots using ArviZ.

    This function generates 6 standard ArviZ diagnostic plots:
    1. Pair plot (corner plot) - pairwise parameter correlations
    2. Forest plot - posterior distributions with HDI
    3. Energy plot - HMC energy diagnostics
    4. Autocorrelation plot - sample independence
    5. Rank plot - chain mixing diagnostics
    6. ESS plot - effective sample size evolution

    These plots are generated REGARDLESS of convergence status to help
    diagnose sampling problems.

    Parameters
    ----------
    result : Any
        MCMC result object with inference_data (CMCResult or similar)
    output_dir : Path
        Output directory for saving plots
    analysis_mode : str
        Analysis mode (static_isotropic or laminar_flow)
    """
    if not hasattr(result, "inference_data") or result.inference_data is None:
        logger.warning(
            "No inference_data available in result - skipping ArviZ diagnostic plots"
        )
        return

    try:
        from homodyne.optimization.cmc.plotting import generate_diagnostic_plots

        diag_dir = output_dir / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Generating ArviZ diagnostic plots...")
        saved_plots = generate_diagnostic_plots(
            result=result,
            output_dir=diag_dir,
        )

        if saved_plots:
            logger.info(f"Generated {len(saved_plots)} ArviZ diagnostic plots:")
            for plot_path in saved_plots:
                logger.info(f"  - {plot_path.name}")
        else:
            logger.warning("No diagnostic plots were generated")

        if hasattr(result, "cmc_diagnostics") and result.cmc_diagnostics is not None:
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

            diag_file = diag_dir / "cmc_diagnostics.json"
            with open(diag_file, "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=2, default=_json_serializer)
            logger.debug(f"CMC diagnostic data saved to: {diag_file}")

    except ImportError as e:
        logger.warning(f"ArviZ plotting not available: {e}")
    except Exception as e:
        logger.warning(f"Failed to generate diagnostic plots: {e}")
        logger.debug(f"Diagnostic plot error details: {e}", exc_info=True)
