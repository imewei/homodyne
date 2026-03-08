"""ArviZ-Based MCMC Plotting Functions.

Provides ArviZ-powered trace, posterior, and pair plots for MCMC/CMC results.
Falls back gracefully when ArviZ is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from homodyne.utils.logging import get_logger
from homodyne.utils.path_validation import PathValidationError, validate_plot_save_path

logger = get_logger(__name__)


def _create_empty_figure(message: str) -> Figure:
    """Create an empty figure with a message."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(
        0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=14
    )
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def plot_arviz_trace(
    result: Any,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    **kwargs: Any,
) -> Figure:
    """Plot MCMC trace plots using ArviZ.

    Creates trace plots showing parameter evolution and posterior distributions
    side by side for each parameter.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object containing samples.
    var_names : list of str, optional
        Parameter names to plot. If None, plots all parameters.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    show : bool, default=False
        If True, display the figure interactively.
    save_path : str or Path, optional
        If provided, save figure to this path.
    dpi : int, default=150
        DPI for saved figure.
    **kwargs
        Additional arguments passed to az.plot_trace().

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> plot_arviz_trace(result, var_names=["D0", "alpha", "D_offset"])
    >>> plot_arviz_trace(result, save_path="traces.png")
    """
    try:
        import arviz as az
    except ImportError:
        logger.warning("ArviZ not available. Falling back to custom trace plots.")
        from homodyne.viz.mcmc_diagnostics import plot_trace_plots

        return plot_trace_plots(result, show=show, save_path=save_path, dpi=dpi)

    if result.samples_params is None:
        logger.warning("No parameter samples available for trace plots")
        fig = _create_empty_figure("No samples available")
        if not show:
            plt.close(fig)
        return fig

    # Convert to ArviZ InferenceData
    idata = result.to_arviz()

    # Plot traces
    axes = az.plot_trace(idata, var_names=var_names, figsize=figsize, **kwargs)
    fig = axes.ravel()[0].figure

    # Add title
    title = "MCMC Trace Plots (ArviZ)"
    if result.is_cmc_result():
        title += f" - CMC ({result.num_shards} shards)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"ArviZ trace plots saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save ArviZ trace plots: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig  # type: ignore[no-any-return]


def plot_arviz_posterior(
    result: Any,
    var_names: list[str] | None = None,
    hdi_prob: float = 0.95,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    **kwargs: Any,
) -> Figure:
    """Plot posterior distributions with 95% credible intervals using ArviZ.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object containing samples.
    var_names : list of str, optional
        Parameter names to plot. If None, plots all parameters.
    hdi_prob : float, default=0.95
        Highest density interval probability (e.g., 0.95 for 95% HDI).
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    show : bool, default=False
        If True, display the figure interactively.
    save_path : str or Path, optional
        If provided, save figure to this path.
    dpi : int, default=150
        DPI for saved figure.
    **kwargs
        Additional arguments passed to az.plot_posterior().

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> plot_arviz_posterior(result, var_names=["D0", "alpha", "D_offset"])
    >>> plot_arviz_posterior(result, hdi_prob=0.90)  # 90% CI
    """
    try:
        import arviz as az
    except ImportError:
        logger.warning("ArviZ not available. Cannot create posterior plots.")
        fig = _create_empty_figure("ArviZ not installed")
        if not show:
            plt.close(fig)
        return fig

    if result.samples_params is None:
        logger.warning("No parameter samples available for posterior plots")
        fig = _create_empty_figure("No samples available")
        if not show:
            plt.close(fig)
        return fig

    # Convert to ArviZ InferenceData
    idata = result.to_arviz()

    # Plot posteriors with HDI
    axes = az.plot_posterior(
        idata, var_names=var_names, hdi_prob=hdi_prob, figsize=figsize, **kwargs
    )

    # Handle both single and multi-panel cases
    if hasattr(axes, "ravel"):
        fig = axes.ravel()[0].figure
    else:
        fig = axes.figure

    # Add title
    title = f"Posterior Distributions ({int(hdi_prob * 100)}% HDI)"
    if result.is_cmc_result():
        title += f" - CMC ({result.num_shards} shards)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"ArviZ posterior plots saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save ArviZ posterior plots: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig  # type: ignore[no-any-return]


def plot_arviz_pair(
    result: Any,
    var_names: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
    **kwargs: Any,
) -> Figure:
    """Plot pair plots showing parameter correlations using ArviZ.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object containing samples.
    var_names : list of str, optional
        Parameter names to plot. If None, plots physical parameters only.
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated.
    show : bool, default=False
        If True, display the figure interactively.
    save_path : str or Path, optional
        If provided, save figure to this path.
    dpi : int, default=150
        DPI for saved figure.
    **kwargs
        Additional arguments passed to az.plot_pair().

    Returns
    -------
    Figure
        Matplotlib figure object.

    Examples
    --------
    >>> plot_arviz_pair(result, var_names=["D0", "alpha", "D_offset"])
    >>> plot_arviz_pair(result)  # Auto-selects physical parameters
    """
    try:
        import arviz as az
    except ImportError:
        logger.warning("ArviZ not available. Cannot create pair plots.")
        fig = _create_empty_figure("ArviZ not installed")
        if not show:
            plt.close(fig)
        return fig

    if result.samples_params is None:
        logger.warning("No parameter samples available for pair plots")
        fig = _create_empty_figure("No samples available")
        if not show:
            plt.close(fig)
        return fig

    # Convert to ArviZ InferenceData
    idata = result.to_arviz()

    # Default to physical parameters only (exclude per-angle scaling)
    if var_names is None:
        if result.analysis_mode == "static":
            var_names = ["D0", "alpha", "D_offset"]
        elif result.analysis_mode == "laminar_flow":
            var_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

    # Plot pair plots
    axes = az.plot_pair(
        idata,
        var_names=var_names,
        figsize=figsize,
        kind="kde",
        marginals=True,
        **kwargs,
    )

    # Get figure
    if hasattr(axes, "ravel"):
        fig = axes.ravel()[0].figure
    elif hasattr(axes, "figure"):
        fig = axes.figure
    else:
        fig = plt.gcf()

    # Add title
    title = "Parameter Correlations"
    if result.is_cmc_result():
        title += f" - CMC ({result.num_shards} shards)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"ArviZ pair plots saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save ArviZ pair plots: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig  # type: ignore[no-any-return]
