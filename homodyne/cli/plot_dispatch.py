"""Plot dispatch for Homodyne CLI.

Handles plotting options for experimental data, simulated data,
and fit comparison visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from homodyne.data.angle_filtering import (
    apply_angle_filtering_for_plot as _data_apply_angle_filtering_for_plot,
)
from homodyne.utils.logging import get_logger
from homodyne.viz.experimental_plots import (
    plot_experimental_data as _viz_plot_experimental_data,
)
from homodyne.viz.experimental_plots import (
    plot_fit_comparison as _viz_plot_fit_comparison,
)
from homodyne.viz.nlsq_plots import (
    generate_and_plot_fitted_simulations as _viz_generate_and_plot_fitted_simulations,
)
from homodyne.viz.nlsq_plots import (
    generate_nlsq_plots as _viz_generate_nlsq_plots,
)
from homodyne.viz.nlsq_plots import (
    plot_simulated_data as _viz_plot_simulated_data,
)

logger = get_logger(__name__)


def _handle_plotting(
    args: Any,
    result: Any,
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
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
            logger.info(f"OK: Experimental data plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate experimental data plots: {e}")

    # Plot simulated data if requested
    if plot_sim:
        try:
            if config is None:
                logger.error("Configuration required for simulated data plotting")
                return

            # Get contrast, offset, and phi_angles from args
            contrast = getattr(args, "contrast", 0.5)
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
            logger.info(f"OK: Simulated data plots saved to: {plots_dir}")
        except Exception as e:
            logger.warning(f"Failed to generate simulated data plots: {e}")

    # Plot fit comparison if save_plots is enabled
    if save_plots:
        try:
            _plot_fit_comparison(result, data, plots_dir)
            logger.info(f"OK: Fit comparison plots saved to: {plots_dir}")
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


def _apply_angle_filtering_for_plot(
    phi_angles: np.ndarray,
    c2_exp: np.ndarray,
    data: dict[str, Any],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Apply angle filtering to select specific angles for plotting.

    This is a wrapper that delegates to homodyne.data.angle_filtering.
    """
    return _data_apply_angle_filtering_for_plot(phi_angles, c2_exp, data)


def _plot_experimental_data(data: dict[str, Any], plots_dir: Path) -> None:
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
    plots_dir: Path,
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
    output_dir: Path,
) -> None:
    """Generate and plot C2 simulations using fitted parameters from optimization.

    This is a wrapper that delegates to homodyne.viz.nlsq_plots.
    """
    from homodyne.cli.data_pipeline import _apply_angle_filtering_for_optimization

    _viz_generate_and_plot_fitted_simulations(
        result,
        data,
        config,
        output_dir,
        angle_filter_func=_apply_angle_filtering_for_optimization,
    )


def _plot_fit_comparison(result: Any, data: dict[str, Any], plots_dir: Path) -> None:
    """Generate comparison plots between fit and experimental data.

    This is a wrapper that delegates to homodyne.viz.experimental_plots.
    """
    _viz_plot_fit_comparison(result, data, plots_dir)


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
