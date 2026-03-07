"""MCMC Posterior Comparison Visualization.

Provides per-shard vs combined posterior distribution comparisons for CMC results.
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

logger = get_logger(__name__)


def plot_posterior_comparison(
    result: Any,  # MCMCResult type
    param_indices: list[int] | None = None,
    figsize: tuple[float, float] | None = None,
    bins: int = 30,
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Compare per-shard posteriors with combined posterior (CMC only).

    Visualizes posterior distributions for each parameter, showing both
    per-shard distributions and the combined posterior.

    Parameters
    ----------
    result : MCMCResult
        CMC result object with per-shard diagnostics
    param_indices : list of int, optional
        Parameter indices to plot. If None, plots first 6 parameters
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    bins : int, default=30
        Number of bins for histograms
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
        If result is not a CMC result

    Examples
    --------
    >>> plot_posterior_comparison(cmc_result, param_indices=[0, 1, 2])

    Notes
    -----
    - Light colored lines: Per-shard posteriors
    - Bold colored line: Combined posterior
    - Good agreement: All distributions overlap
    - Poor agreement: Shards show different modes
    """
    # Check if CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, "is_cmc_result") else False

    if not is_cmc:
        raise ValueError("Posterior comparison is only available for CMC results")

    if result.per_shard_diagnostics is None:
        raise ValueError("Per-shard diagnostics not available")

    # Extract samples
    num_params = (
        result.samples_params.shape[-1] if result.samples_params is not None else 0
    )

    # Select parameters to plot
    if param_indices is None:
        param_indices = list(range(min(6, num_params)))

    num_plots = len(param_indices)

    # Calculate figure layout
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Get parameter names
    if result.analysis_mode == "static":
        param_names = ["D0", "alpha", "D_offset"]
    elif result.analysis_mode == "laminar_flow":
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

    # Plot each parameter
    for plot_idx, param_idx in enumerate(param_indices):
        ax = axes[plot_idx]

        # Extract per-shard samples
        num_shards = len(result.per_shard_diagnostics)
        colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, num_shards))

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if "trace_data" in shard_diag:
                trace_key = f"param_{param_idx}"
                if trace_key in shard_diag["trace_data"]:
                    trace = np.array(shard_diag["trace_data"][trace_key])

                    # Flatten if multi-chain
                    if trace.ndim == 2:
                        trace = trace.flatten()

                    # Plot histogram
                    ax.hist(
                        trace,
                        bins=bins,
                        alpha=0.3,
                        color=colors[shard_idx],
                        density=True,
                        label=f"Shard {shard_idx}" if num_shards <= 10 else "",
                    )

        # Plot combined posterior
        if result.samples_params is not None:
            combined_samples = result.samples_params[:, param_idx]
            ax.hist(
                combined_samples,
                bins=bins,
                alpha=0.5,
                color="black",
                density=True,
                histtype="step",
                linewidth=2,
                label="Combined",
            )

        ax.set_xlabel(param_names[param_idx])
        ax.set_ylabel("Density")
        ax.set_title(f"{param_names[param_idx]} Posterior")

        # Add legend for first subplot only
        if plot_idx == 0 and num_shards <= 10:
            ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle(
        f"Posterior Comparison (CMC: {result.num_shards} shards)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"Posterior comparison saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save posterior comparison: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig
