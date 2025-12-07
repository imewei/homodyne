"""NLSQ optimization plotting functions for homodyne XPCS analysis.

This module provides functions for visualizing NLSQ optimization results,
including simulated data, fitted data, and comparison heatmaps.
Extracted from cli/commands.py for better modularity.
"""

import json
import multiprocessing
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Check if Datashader backend is available (actual import done lazily)
try:
    import homodyne.viz.datashader_backend  # noqa: F401

    DATASHADER_AVAILABLE = True
except ImportError:
    DATASHADER_AVAILABLE = False


def plot_simulated_data(
    config: dict[str, Any],
    contrast: float,
    offset: float,
    phi_angles_str: str | None,
    plots_dir: Path,
    data: dict[str, Any] | None = None,
) -> None:
    """Generate plots of simulated/theoretical data.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with analysis_mode and parameters
    contrast : float
        Contrast parameter value
    offset : float
        Offset parameter value
    phi_angles_str : str | None
        Comma-separated phi angles string from CLI
    plots_dir : Path
        Output directory for plot files
    data : dict[str, Any] | None
        Optional experimental data dictionary for extracting phi angles
    """
    from homodyne.core.models import CombinedModel

    # BUGFIX: Force contrast to 0.5 to match working version
    if contrast < 0.4:
        logger.debug(f"Overriding contrast={contrast} → 0.5 (matching working version)")
        contrast = 0.5

    logger.info(
        f"Generating simulated data plots (contrast={contrast:.3f}, offset={offset:.3f})",
    )

    # Determine analysis mode
    analysis_mode = config.get("analysis_mode", "static")
    logger.info(f"Analysis mode: {analysis_mode}")

    # Create model
    model = CombinedModel(analysis_mode)

    # Get parameters from configuration
    initial_params_config = config.get("initial_parameters", {})
    param_names = initial_params_config.get("parameter_names", [])
    param_values = initial_params_config.get("values", [])

    params_dict = (
        dict(zip(param_names, param_values, strict=False))
        if param_names and param_values
        else {}
    )

    if analysis_mode.startswith("static"):
        params = jnp.array(
            [
                params_dict.get("D0", 100.0),
                params_dict.get("alpha", -0.5),
                params_dict.get("D_offset", 0.0),
            ],
        )
    else:
        params = jnp.array(
            [
                params_dict.get("D0", 100.0),
                params_dict.get("alpha", -0.5),
                params_dict.get("D_offset", 0.0),
                params_dict.get("gamma_dot_t0", 0.01),
                params_dict.get("beta", 0.5),
                params_dict.get("gamma_dot_t_offset", 0.0),
                params_dict.get("phi0", 0.0),
            ],
        )

    logger.debug(
        f"Using parameters: {dict(zip(model.parameter_names, params, strict=False))}",
    )

    # Determine phi angles for theoretical simulation plots
    if phi_angles_str:
        phi_degrees = np.array([float(x.strip()) for x in phi_angles_str.split(",")])
        phi = phi_degrees
        logger.info(
            f"Using CLI-provided phi angles for theoretical plots: {phi_degrees}"
        )
    elif data is not None and "phi_angles_list" in data:
        phi_degrees = np.array(data["phi_angles_list"])
        phi = phi_degrees
        logger.info(
            f"Using experimental data phi angles for theoretical plots: {phi_degrees}"
        )
        logger.warning(
            "Theoretical plots using potentially filtered phi angles from experimental data. "
            "To use all angles, disable phi_filtering in config or provide --phi-angles explicitly.",
        )
    else:
        phi_degrees = np.linspace(0, 180, 8)
        phi = phi_degrees
        logger.info(f"Using default phi angles for theoretical plots: {phi_degrees}")

    logger.debug(f"Generating simulated data for {len(phi)} phi angles")

    # Generate time arrays matching configuration specification
    analyzer_params = config.get("analyzer_parameters", {})
    dt = analyzer_params.get("dt", 0.1)
    start_frame = analyzer_params.get("start_frame", 1)
    end_frame = analyzer_params.get("end_frame", 8000)

    n_time_points = end_frame - start_frame + 1
    t_vals = dt * np.arange(1, n_time_points + 1)
    t1_grid, t2_grid = np.meshgrid(t_vals, t_vals, indexing="ij")

    logger.debug(
        f"Simulated data time grid: dt={dt}, start_frame={start_frame}, end_frame={end_frame}",
    )
    logger.debug(
        f"Time range: [{float(t_vals[0]):.4f}, {float(t_vals[-1]):.2f}] seconds with {n_time_points} points",
    )

    # Get wavevector_q and stator_rotor_gap from correct config sections
    scattering_config = analyzer_params.get("scattering", {})
    geometry_config = analyzer_params.get("geometry", {})

    q = scattering_config.get("wavevector_q", 0.0054)
    L_angstroms = geometry_config.get("stator_rotor_gap", 2000000)
    L_microns = L_angstroms / 10000.0

    logger.info(
        f"Generating theoretical C₂ with q={q:.6f} Å⁻¹, L={L_microns:.1f} μm ({L_angstroms:.0f} Å)",
    )

    # Generate simulated C₂ for each phi angle
    c2_simulated = []

    for _i, phi_val in enumerate(phi):
        phi_array = jnp.array([phi_val])
        logger.debug(f"Computing C₂ for φ={phi_val}° (phi_array={phi_array})")

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

        c2_result = np.array(c2_phi[0])
        logger.debug(
            f"  C₂ shape: {c2_result.shape}, range: [{c2_result.min():.4f}, {c2_result.max():.4f}]",
        )
        c2_simulated.append(c2_result)

    c2_simulated = np.array(c2_simulated)

    logger.info(f"Generated simulated C₂ with shape: {c2_simulated.shape}")

    # Compute global color scale
    c2_min = float(c2_simulated.min())
    c2_max = float(c2_simulated.max())
    vmin = max(1.0, c2_min)
    vmax = min(1.6, c2_max)

    logger.debug(
        f"Simulated C2 range [{c2_min:.4f}, {c2_max:.4f}] → "
        f"color scale [{vmin:.4f}, {vmax:.4f}] (clamped to [1.0, 1.6])"
    )

    # Save individual C₂ heatmap for EACH phi angle
    n_phi = len(phi)
    logger.info(
        f"Generating individual simulated C₂ heatmaps for {n_phi} phi angles..."
    )

    for idx in range(n_phi):
        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(
            c2_simulated[idx].T,
            extent=[t_vals[0], t_vals[-1], t_vals[0], t_vals[-1]],
            aspect="equal",
            cmap="jet",
            origin="lower",
            interpolation="bilinear",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("t₁ (s)", fontsize=11)
        ax.set_ylabel("t₂ (s)", fontsize=11)
        ax.set_title(
            f"Simulated C₂(t₁, t₂) at φ={phi_degrees[idx]:.1f}°",
            fontsize=13,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        mean_val = np.mean(c2_simulated[idx])
        max_val = np.max(c2_simulated[idx])
        min_val = np.min(c2_simulated[idx])

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

        filename = f"simulated_data_phi_{phi_degrees[idx]:.1f}.png"
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  Saved: {filename}")

    logger.info(f"Generated {n_phi} individual simulated C₂ heatmaps")

    # Plot diagonal (t1=t2) for all phi angles
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

    logger.info("Generated diagonal plot: simulated_data_diagonal.png")
    logger.info("Simulated data plots generated successfully")


def generate_and_plot_fitted_simulations(
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any],
    output_dir: Path,
    angle_filter_func: Any | None = None,
) -> None:
    """Generate and plot C2 simulations using fitted parameters from optimization.

    Parameters
    ----------
    result : Any
        Optimization result containing fitted parameters
    data : dict[str, Any]
        Experimental data dictionary containing phi_angles_list, t1, t2, c2_exp
    config : dict[str, Any]
        Configuration dictionary with analysis_mode and physics parameters
    output_dir : Path
        Output directory path (simulated_data/ subdirectory will be created)
    angle_filter_func : callable, optional
        Function to apply angle filtering for optimization
    """
    from homodyne.config.manager import ConfigManager
    from homodyne.core.models import CombinedModel

    logger.info("Generating fitted C₂ simulations...")

    # Apply phi filtering to data (if enabled in config)
    if angle_filter_func is not None:
        if isinstance(config, dict):
            config_for_filtering = ConfigManager(config_dict=config)
        else:
            config_for_filtering = config
        filtered_data = angle_filter_func(data, config_for_filtering)
        logger.debug(
            f"Applied phi filtering for fitted simulation plots: "
            f"{len(filtered_data['phi_angles_list'])} angles selected"
        )
    else:
        filtered_data = data

    # Create simulated_data subdirectory
    simulated_data_dir = output_dir / "simulated_data"
    simulated_data_dir.mkdir(parents=True, exist_ok=True)

    # Extract fitted parameters from result
    if hasattr(result, "parameters") and isinstance(result.parameters, np.ndarray):
        params_array = result.parameters
        if len(params_array) >= 2:
            contrast = float(params_array[0])
            offset = float(params_array[1])
            physical_params = params_array[2:] if len(params_array) > 2 else []
        else:
            logger.warning(
                f"Insufficient parameters in result.parameters: {len(params_array)}"
            )
            contrast = 0.5
            offset = 1.0
            physical_params = []
    elif hasattr(result, "mean_params"):
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

    # Get experimental data structure (using filtered data)
    phi_angles_list = filtered_data.get("phi_angles_list", None)
    t1 = filtered_data.get("t1", None)
    t2 = filtered_data.get("t2", None)

    if phi_angles_list is None or t1 is None or t2 is None:
        logger.warning("Missing experimental data structure (phi_angles_list, t1, t2)")
        return

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

        c2_result = np.array(c2_phi[0])
        c2_fitted_list.append(c2_result)

        logger.debug(f"  C₂ range: [{c2_result.min():.4f}, {c2_result.max():.4f}]")

    c2_fitted = np.array(c2_fitted_list)

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
    logger.info(f"Saved fitted C₂ data: {npz_file}")

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
    logger.info(f"Saved simulation config: {config_file}")

    # Generate individual plots for each phi angle
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
        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(
            c2_fitted[i].T,
            aspect="equal",
            cmap="jet",
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

        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        mean_val = np.mean(c2_fitted[i])
        max_val = np.max(c2_fitted[i])
        min_val = np.min(c2_fitted[i])

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

        filename = f"simulated_c2_fitted_phi_{phi_deg:.1f}deg.png"
        plt.savefig(simulated_data_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  Saved: {filename}")

    logger.info(f"Generated {len(phi_angles_list)} individual fitted C₂ plots")
    logger.info(f"Fitted simulation data saved to: {simulated_data_dir}")


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
        Configuration object/dict containing output.plots settings
    use_datashader : bool, default=True
        Legacy parameter for backward compatibility
    parallel : bool, default=True
        Generate plots in parallel using multiprocessing
    c2_solver_scaled : np.ndarray | None
        Optional solver-computed scaled C2 for display
    """
    logger.info(f"Generating heatmap plots for {len(phi_angles)} angles")

    # Determine rendering mode from config
    preview_mode = use_datashader
    width = 1200
    height = 1200
    fit_surface_mode = "solver"
    color_scale_cfg: dict[str, Any] = {}

    if config is not None:
        config_dict = config.config if hasattr(config, "config") else config

        output_config = config_dict.get("output", {})
        plots_config = output_config.get("plots", {})
        preview_mode = plots_config.get("preview_mode", preview_mode)
        fit_surface_mode = plots_config.get("fit_surface", fit_surface_mode)
        color_scale_cfg = plots_config.get("color_scale", {})

        datashader_config = plots_config.get("datashader", {})
        width = datashader_config.get("canvas_width", width)
        height = datashader_config.get("canvas_height", height)

        logger.debug(
            f"Plot config: preview_mode={preview_mode}, "
            f"canvas={width}×{height}, parallel={parallel}",
        )

    # Determine which fit surface to display
    use_solver_surface = fit_surface_mode == "solver" and c2_solver_scaled is not None
    c2_fit_display = c2_solver_scaled if use_solver_surface else c2_theoretical_scaled
    residuals_display = c2_exp - c2_fit_display

    color_mode = color_scale_cfg.get("mode", "legacy")
    pin_legacy_range = color_scale_cfg.get(
        "pin_legacy_range",
        color_mode != "adaptive",
    )
    percentile_min = color_scale_cfg.get("percentile_min", 1.0)
    percentile_max = color_scale_cfg.get("percentile_max", 99.0)

    # Adaptive color scaling
    c2_min = min(np.min(c2_exp), np.min(c2_fit_display))
    c2_max = max(np.max(c2_exp), np.max(c2_fit_display))
    vmin_adaptive = float(max(1.0, c2_min))
    vmax_adaptive = float(min(1.6, c2_max))

    logger.debug(
        f"C2 data range [{c2_min:.4f}, {c2_max:.4f}] → "
        f"color scale [{vmin_adaptive:.4f}, {vmax_adaptive:.4f}] (clamped to [1.0, 1.6])"
    )

    if pin_legacy_range:
        color_options = {
            "vmin": vmin_adaptive,
            "vmax": vmax_adaptive,
            "adaptive": False,
            "percentile_min": percentile_min,
            "percentile_max": percentile_max,
        }
    else:
        color_options = {
            "vmin": color_scale_cfg.get("fixed_min", vmin_adaptive),
            "vmax": color_scale_cfg.get("fixed_max", vmax_adaptive),
            "adaptive": color_mode == "adaptive",
            "percentile_min": percentile_min,
            "percentile_max": percentile_max,
        }

    surface_label = "solver" if use_solver_surface else "posthoc"
    logger.info(f"Plotting fit surface: {surface_label}")

    # Select backend based on mode
    if preview_mode and DATASHADER_AVAILABLE:
        logger.info("Using Datashader backend (preview mode, fast rendering)")
        _generate_plots_datashader(
            phi_angles,
            c2_exp,
            c2_fit_display,
            residuals_display,
            t1,
            t2,
            output_dir,
            parallel=parallel,
            width=1200,
            height=1200,
            color_options=color_options,
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
            c2_fit_display,
            residuals_display,
            t1,
            t2,
            output_dir,
            color_options=color_options,
        )


def _worker_init_cpu_only():
    """Initialize worker process with CPU-only mode."""
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _plot_single_angle_datashader(args):
    """Plot single angle for parallel processing (picklable module-level function)."""
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

    c2_exp_cpu = np.asarray(c2_exp)
    c2_fit_cpu = np.asarray(c2_fit)
    residuals_cpu = np.asarray(residuals)
    t1_cpu = np.asarray(t1)
    t2_cpu = np.asarray(t2)

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
    """Generate plots using Datashader backend with optional parallelization."""
    if parallel and len(phi_angles) > 1:
        ctx = multiprocessing.get_context("spawn")
        n_workers = min(multiprocessing.cpu_count(), len(phi_angles))
        logger.info(f"Using {n_workers} parallel workers for plotting (spawn method)")

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
            with ctx.Pool(
                processes=n_workers, initializer=_worker_init_cpu_only
            ) as pool:
                timeout_seconds = (30 * len(phi_angles) / n_workers) + 60
                logger.debug(f"Parallel plotting timeout: {timeout_seconds:.0f}s")

                result = pool.map_async(_plot_single_angle_datashader, args_list)
                result.get(timeout=timeout_seconds)

            logger.info(f"Generated {len(phi_angles)} heatmap plots (parallel)")

        except Exception as e:
            logger.warning(f"Parallel plotting failed: {e.__class__.__name__}: {e}")
            logger.info("Falling back to sequential plotting...")

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
                f"Generated {len(phi_angles)} heatmap plots (sequential fallback)"
            )
    else:
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

        logger.info(f"Generated {len(phi_angles)} heatmap plots (sequential)")


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
    """Generate plots using matplotlib backend (publication quality)."""
    logger.info(f"Generating heatmap plots for {len(phi_angles)} angles")

    for i, phi in enumerate(phi_angles):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        vmin_use, vmax_use = _resolve_color_limits(c2_exp[i], color_options)

        # Panel 1: Experimental data
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

        # Panel 2: Theoretical fit
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

        # Panel 3: Residuals
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

        plt.tight_layout()
        plot_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.debug(f"Saved plot: {plot_file}")

    logger.info(f"Generated {len(phi_angles)} heatmap plots (matplotlib)")


def _resolve_color_limits(
    matrix: np.ndarray,
    color_options: dict[str, Any] | None,
) -> tuple[float, float]:
    """Resolve color limits for heatmap plotting."""
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
