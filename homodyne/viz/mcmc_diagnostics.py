"""MCMC Convergence Diagnostics Visualization.

Provides trace plots, KL divergence matrix heatmaps, and convergence
diagnostics (R-hat, ESS) for NUTS and CMC results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from homodyne.utils.logging import get_logger
from homodyne.utils.path_validation import PathValidationError, validate_plot_save_path
from homodyne.viz.mcmc_arviz import _create_empty_figure

logger = get_logger(__name__)


def plot_trace_plots(
    result: Any,  # MCMCResult type
    param_names: list[str] | None = None,
    max_params: int = 9,
    figsize: tuple[float, float] | None = None,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Plot MCMC trace plots for convergence visualization.

    Creates a grid of trace plots showing parameter evolution across MCMC samples.
    For CMC results, overlays traces from multiple shards with different colors.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object containing samples
    param_names : list of str, optional
        Parameter names to plot. If None, uses default names (param_0, param_1, ...)
    max_params : int, default=9
        Maximum number of parameters to plot (to avoid cluttered figures)
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated based on param count
    show : bool, default=False
        If True, display the figure interactively
    save_path : str or Path, optional
        If provided, save figure to this path
    dpi : int, default=150
        DPI for saved figure

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> # Standard NUTS result
    >>> plot_trace_plots(nuts_result, param_names=['D0', 'alpha', 'D_offset'])

    >>> # CMC result with multiple shards
    >>> plot_trace_plots(cmc_result, save_path='traces_cmc.png')

    Notes
    -----
    - For NUTS: Single trace line per parameter
    - For CMC: Multiple colored lines (one per shard)
    - X-axis: Sample index (after warmup)
    - Y-axis: Parameter value
    - Good mixing: trace should look like "hairy caterpillar"
    - Poor mixing: trace shows trends or gets stuck
    """
    # Extract samples
    if not hasattr(result, "samples_params") or result.samples_params is None:
        logger.warning("No parameter samples available for trace plots")
        fig = _create_empty_figure("No samples available")
        if not show:
            plt.close(fig)
        return fig

    samples = result.samples_params
    num_params = samples.shape[-1] if samples.ndim >= 2 else 1

    # Limit number of parameters to plot
    num_params_to_plot = min(num_params, max_params)

    # Generate parameter names if not provided
    if param_names is None:
        if getattr(result, "analysis_mode", None) == "static":
            param_names = ["D0", "alpha", "D_offset"]
        elif getattr(result, "analysis_mode", None) == "laminar_flow":
            param_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
        else:
            param_names = [f"param_{i}" for i in range(num_params)]

    # Ensure we have enough parameter names
    if len(param_names) < num_params_to_plot:
        param_names.extend(
            [f"param_{i}" for i in range(len(param_names), num_params_to_plot)]
        )

    # Calculate figure layout
    ncols = min(3, num_params_to_plot)
    nrows = (num_params_to_plot + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Check if this is a CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, "is_cmc_result") else False

    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot multiple shard traces
        num_shards = len(result.per_shard_diagnostics)
        colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, num_shards))

        for param_idx in range(num_params_to_plot):
            ax = axes[param_idx]

            # Plot trace for each shard
            for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
                if "trace_data" in shard_diag:
                    trace_key = f"param_{param_idx}"
                    if trace_key in shard_diag["trace_data"]:
                        trace = np.array(shard_diag["trace_data"][trace_key])

                        # Handle multi-chain traces (flatten if needed)
                        if trace.ndim == 2:
                            # Multi-chain: plot all chains
                            for chain_idx in range(trace.shape[0]):
                                ax.plot(
                                    trace[chain_idx, :],
                                    color=colors[shard_idx],
                                    alpha=0.7,
                                    linewidth=0.5,
                                    label=(
                                        f"Shard {shard_idx}" if chain_idx == 0 else ""
                                    ),
                                )
                        else:
                            # Single chain
                            ax.plot(
                                trace,
                                color=colors[shard_idx],
                                alpha=0.7,
                                linewidth=0.8,
                                label=f"Shard {shard_idx}",
                            )

            ax.set_xlabel("Sample Index")
            ax.set_ylabel(param_names[param_idx])
            ax.set_title(f"{param_names[param_idx]} Trace (CMC)")
            ax.grid(True, alpha=0.3)

            # Add legend for first subplot only (to avoid clutter)
            if param_idx == 0 and num_shards <= 10:
                ax.legend(loc="upper right", fontsize=8, ncol=2)

    else:
        # Standard NUTS: Plot single trace (possibly multi-chain)
        if samples.ndim == 1:
            # Single parameter, single chain
            samples = samples.reshape(-1, 1)
        elif samples.ndim == 2:
            # (num_samples, num_params) - already correct shape
            pass
        elif samples.ndim == 3:
            # (num_chains, num_samples, num_params) - flatten chains
            num_chains, num_samples, num_params_actual = samples.shape
            samples = samples.reshape(num_chains * num_samples, num_params_actual)

        for param_idx in range(num_params_to_plot):
            ax = axes[param_idx]
            trace = samples[:, param_idx]

            ax.plot(trace, linewidth=0.5, alpha=0.8, color="steelblue")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel(param_names[param_idx])
            ax.set_title(f"{param_names[param_idx]} Trace")
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_params_to_plot, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    title = "MCMC Trace Plots"
    if is_cmc:
        title += f" (CMC: {result.num_shards} shards)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"Trace plots saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save trace plots: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)
    # Note: when show=False and save_path=None, caller owns the figure and must close it.

    return fig


def plot_kl_divergence_matrix(
    result: Any,  # MCMCResult type
    figsize: tuple[float, float] = (8, 7),
    cmap: str = "coolwarm",
    threshold: float = 2.0,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Plot KL divergence matrix heatmap for CMC results.

    Visualizes pairwise KL divergence between shards to assess posterior agreement.
    High KL divergence (>2.0) indicates shards found different posteriors.

    Parameters
    ----------
    result : MCMCResult
        CMC result object with cmc_diagnostics containing KL matrix
    figsize : tuple, default=(8, 7)
        Figure size (width, height)
    cmap : str, default='coolwarm'
        Matplotlib colormap name (coolwarm shows cool=low, warm=high KL)
    threshold : float, default=2.0
        KL divergence threshold to highlight (standard: 2.0)
    show : bool, default=False
        If True, display the figure interactively
    save_path : str or Path, optional
        If provided, save figure to this path
    dpi : int, default=150
        DPI for saved figure

    Returns
    -------
    Figure
        Matplotlib figure object

    Raises
    ------
    ValueError
        If result is not a CMC result or KL matrix not available

    Examples
    --------
    >>> plot_kl_divergence_matrix(cmc_result, threshold=2.0, save_path='kl_matrix.png')

    Notes
    -----
    - Diagonal elements are 0.0 (self-divergence)
    - Matrix is symmetric by construction (averaged KL)
    - Values < 0.5: Excellent agreement
    - Values 0.5-2.0: Acceptable agreement
    - Values > 2.0: Poor agreement (possible multimodality)
    """
    # Check if CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, "is_cmc_result") else False

    if not is_cmc:
        raise ValueError("KL divergence matrix is only available for CMC results")

    # Extract KL matrix from diagnostics
    if result.cmc_diagnostics is None or "kl_matrix" not in result.cmc_diagnostics:
        raise ValueError("KL divergence matrix not found in CMC diagnostics")

    kl_matrix = np.array(result.cmc_diagnostics["kl_matrix"])
    num_shards = kl_matrix.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    kl_max = float(np.nanmax(kl_matrix)) if np.any(np.isfinite(kl_matrix)) else 0.0
    vmax = max(threshold * 1.5, kl_max)
    im = ax.imshow(
        kl_matrix,
        cmap=cmap,
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="KL Divergence")

    # Add threshold line on colorbar
    cbar.ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({threshold})",
    )

    # Annotate cells with KL values.  For large matrices (>20 shards) text
    # annotations would be illegible and O(n^2) ax.text() calls degrade render
    # performance significantly; skip text but keep the red-border highlights
    # for problematic off-diagonal cells, which remain readable at any scale.
    _ANNOTATION_THRESHOLD = 20
    _annotate = num_shards <= _ANNOTATION_THRESHOLD
    for i in range(num_shards):
        for j in range(num_shards):
            kl_val = kl_matrix[i, j]

            if _annotate:
                # Choose text color based on background
                text_color = "white" if kl_val > threshold else "black"
                _text = ax.text(  # noqa: F841 - Text object kept for reference
                    j,
                    i,
                    f"{kl_val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                )

            # Highlight problematic off-diagonal shards (KL > threshold)
            if kl_val > threshold and i != j:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )

    # Set labels and title
    ax.set_xticks(np.arange(num_shards))
    ax.set_yticks(np.arange(num_shards))
    ax.set_xticklabels([f"S{i}" for i in range(num_shards)])
    ax.set_yticklabels([f"S{i}" for i in range(num_shards)])
    ax.set_xlabel("Shard Index")
    ax.set_ylabel("Shard Index")
    ax.set_title(
        f"Between-Shard KL Divergence Matrix\n({num_shards} shards, threshold={threshold})",
        fontweight="bold",
    )

    # Add grid
    ax.set_xticks(np.arange(num_shards) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_shards) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"KL divergence matrix saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save KL divergence matrix: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig


def plot_convergence_diagnostics(
    result: Any,  # MCMCResult type
    metrics: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    rhat_threshold: float = 1.1,
    ess_threshold: float = 400.0,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Plot convergence diagnostics (R-hat and ESS) for MCMC results.

    Visualizes convergence metrics to assess MCMC sampling quality.
    For CMC results, shows per-shard and combined diagnostics.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object with convergence diagnostics
    metrics : list of str, optional
        Metrics to plot. Options: 'rhat', 'ess'. Defaults to ['rhat', 'ess']
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    rhat_threshold : float, default=1.1
        R-hat threshold for convergence (standard: 1.1)
    ess_threshold : float, default=400.0
        ESS threshold for adequate sampling (standard: 400)
    show : bool, default=False
        If True, display the figure interactively
    save_path : str or Path, optional
        If provided, save figure to this path
    dpi : int, default=150
        DPI for saved figure

    Returns
    -------
    Figure
        Matplotlib figure object

    Examples
    --------
    >>> plot_convergence_diagnostics(result, metrics=['rhat', 'ess'])

    >>> # CMC result with per-shard diagnostics
    >>> plot_convergence_diagnostics(cmc_result, save_path='convergence.png')

    Notes
    -----
    - R-hat < 1.1: Converged (good)
    - R-hat > 1.1: Not converged (bad)
    - ESS > 100: Adequate sampling (good)
    - ESS < 100: Poor sampling efficiency (bad)
    """
    # Set default metrics if not provided
    if metrics is None:
        metrics = ["rhat", "ess"]

    # Check if CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, "is_cmc_result") else False

    # Determine number of subplots
    num_metrics = len(metrics)

    if figsize is None:
        figsize = (10, 4 * num_metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Extract parameter names (guard against None mean_params from failed CMC runs)
    num_params = len(result.mean_params) if result.mean_params is not None else 0
    if result.analysis_mode == "static":
        param_names = ["D0", "alpha", "D_offset"][:num_params]
    elif result.analysis_mode == "laminar_flow":
        param_names = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ][:num_params]
    else:
        param_names = [f"param_{i}" for i in range(num_params)]

    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        if metric == "rhat":
            _plot_rhat(ax, result, param_names, rhat_threshold, is_cmc)
        elif metric == "ess":
            _plot_ess(ax, result, param_names, ess_threshold, is_cmc)
        else:
            logger.warning(f"Unknown metric: {metric}")
            ax.text(
                0.5,
                0.5,
                f"Unknown metric: {metric}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # Add overall title
    title = "MCMC Convergence Diagnostics"
    if is_cmc:
        title += f" (CMC: {result.num_shards} shards)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"Convergence diagnostics saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save convergence diagnostics: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig


def _plot_rhat(
    ax: Any, result: Any, param_names: list[str], threshold: float, is_cmc: bool
) -> None:
    """Helper function to plot R-hat diagnostics."""
    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot per-shard R-hat values
        num_shards = len(result.per_shard_diagnostics)
        num_params = len(param_names)

        # Collect R-hat values for each parameter across shards
        rhat_matrix = np.full((num_shards, num_params), np.nan)

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if "rhat" in shard_diag and shard_diag["rhat"] is not None:
                for param_idx in range(num_params):
                    param_key = f"param_{param_idx}"
                    if param_key in shard_diag["rhat"]:
                        rhat_matrix[shard_idx, param_idx] = shard_diag["rhat"][
                            param_key
                        ]

        # Plot as grouped bar chart
        x = np.arange(num_params)
        width = 0.8 / num_shards if num_shards < 10 else 0.8 / 10

        for shard_idx in range(
            min(num_shards, 10)
        ):  # Limit to first 10 shards for clarity
            offset = (shard_idx - num_shards / 2) * width
            values = rhat_matrix[shard_idx, :]

            # Color based on convergence
            colors = ["green" if v < threshold else "red" for v in values]
            ax.bar(
                x + offset,
                values,
                width,
                label=f"Shard {shard_idx}",
                color=colors,
                alpha=0.7,
            )

        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )
        ax.set_xlabel("Parameter")
        ax.set_ylabel("R-hat")
        ax.set_title("R-hat Convergence Diagnostic (per shard)")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # Standard NUTS: Plot combined R-hat
        if result.r_hat is None:
            ax.text(
                0.5,
                0.5,
                "R-hat not available\n(requires multiple chains)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Extract R-hat values
        rhat_values = []
        for param_name in param_names:
            # Try different key formats
            for key in [param_name, param_name.lower(), param_name.replace("_", "")]:
                if key in result.r_hat:
                    rhat_values.append(result.r_hat[key])
                    break
            else:
                rhat_values.append(np.nan)

        # Plot as bar chart
        x = np.arange(len(param_names))
        colors = ["green" if v < threshold else "red" for v in rhat_values]

        ax.bar(x, rhat_values, color=colors, alpha=0.7)
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )
        ax.set_xlabel("Parameter")
        ax.set_ylabel("R-hat")
        ax.set_title("R-hat Convergence Diagnostic")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")


def _plot_ess(
    ax: Any, result: Any, param_names: list[str], threshold: float, is_cmc: bool
) -> None:
    """Helper function to plot ESS diagnostics."""
    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot per-shard ESS values
        num_shards = len(result.per_shard_diagnostics)
        num_params = len(param_names)

        # Collect ESS values for each parameter across shards
        ess_matrix = np.full((num_shards, num_params), np.nan)

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if "ess" in shard_diag and shard_diag["ess"] is not None:
                for param_idx in range(num_params):
                    param_key = f"param_{param_idx}"
                    if param_key in shard_diag["ess"]:
                        ess_matrix[shard_idx, param_idx] = shard_diag["ess"][param_key]

        # Plot as grouped bar chart
        x = np.arange(num_params)
        width = 0.8 / num_shards if num_shards < 10 else 0.8 / 10

        for shard_idx in range(min(num_shards, 10)):  # Limit to first 10 shards
            offset = (shard_idx - num_shards / 2) * width
            values = ess_matrix[shard_idx, :]

            # Color based on adequacy
            colors = ["green" if v > threshold else "orange" for v in values]
            ax.bar(
                x + offset,
                values,
                width,
                label=f"Shard {shard_idx}",
                color=colors,
                alpha=0.7,
            )

        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Effective Sample Size (ESS)")
        ax.set_title("ESS Diagnostic (per shard)")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    else:
        # Standard NUTS: Plot combined ESS
        if result.effective_sample_size is None:
            ax.text(
                0.5,
                0.5,
                "ESS not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Extract ESS values
        ess_values = []
        for param_name in param_names:
            # Try different key formats
            for key in [param_name, param_name.lower(), param_name.replace("_", "")]:
                if key in result.effective_sample_size:
                    ess_values.append(result.effective_sample_size[key])
                    break
            else:
                ess_values.append(np.nan)

        # Plot as bar chart
        x = np.arange(len(param_names))
        colors = ["green" if v > threshold else "orange" for v in ess_values]

        ax.bar(x, ess_values, color=colors, alpha=0.7)
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold ({threshold})",
        )
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Effective Sample Size (ESS)")
        ax.set_title("ESS Diagnostic")
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
