"""MCMC Summary Dashboard Visualization.

Provides a comprehensive multi-panel CMC summary dashboard combining
KL divergence, convergence diagnostics, trace plots, and posterior histograms.
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


def plot_cmc_summary_dashboard(
    result: Any,  # MCMCResult type
    figsize: tuple[float, float] = (16, 12),
    show: bool = False,
    save_path: str | Path | None = None,
    dpi: int = 150,
) -> Figure:
    """Create comprehensive multi-panel CMC summary dashboard.

    Combines all diagnostic plots into a single comprehensive figure:
    - Panel 1: KL divergence matrix
    - Panel 2: Convergence diagnostics (R-hat, ESS)
    - Panel 3: Trace plots (selected parameters)
    - Panel 4: Posterior comparison

    Parameters
    ----------
    result : MCMCResult
        CMC result object
    figsize : tuple, default=(16, 12)
        Figure size (width, height)
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
    >>> plot_cmc_summary_dashboard(cmc_result, save_path='cmc_summary.png')

    Notes
    -----
    This is the primary diagnostic tool for CMC results. It provides a
    comprehensive overview of convergence, agreement between shards,
    and posterior quality in a single figure.
    """
    # Check if CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, "is_cmc_result") else False

    if not is_cmc:
        raise ValueError("Summary dashboard is only available for CMC results")

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: KL divergence matrix (top left)
    ax_kl = fig.add_subplot(gs[0, 0])
    try:
        if result.cmc_diagnostics is not None and "kl_matrix" in result.cmc_diagnostics:
            kl_matrix = np.array(result.cmc_diagnostics["kl_matrix"])
            num_shards = kl_matrix.shape[0]
            threshold = 2.0

            _kl_max = (
                float(np.nanmax(kl_matrix)) if np.any(np.isfinite(kl_matrix)) else 0.0
            )
            im = ax_kl.imshow(
                kl_matrix,
                cmap="coolwarm",
                aspect="auto",
                vmin=0,
                vmax=max(threshold * 1.5, _kl_max),
            )
            plt.colorbar(im, ax=ax_kl, label="KL Divergence")

            # Annotate cells (skip text for >20 shards -- O(n^2) and illegible)
            _annotate = num_shards <= 20
            for i in range(num_shards):
                for j in range(num_shards):
                    kl_val = kl_matrix[i, j]
                    if _annotate:
                        text_color = "white" if kl_val > threshold else "black"
                        ax_kl.text(
                            j,
                            i,
                            f"{kl_val:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=7,
                        )

            ax_kl.set_xticks(np.arange(num_shards))
            ax_kl.set_yticks(np.arange(num_shards))
            ax_kl.set_xticklabels([f"S{i}" for i in range(num_shards)], fontsize=8)
            ax_kl.set_yticklabels([f"S{i}" for i in range(num_shards)], fontsize=8)
            ax_kl.set_xlabel("Shard Index", fontsize=9)
            ax_kl.set_ylabel("Shard Index", fontsize=9)
            ax_kl.set_title("KL Divergence Matrix", fontsize=10, fontweight="bold")
        else:
            ax_kl.text(
                0.5,
                0.5,
                "KL matrix not available",
                ha="center",
                va="center",
                transform=ax_kl.transAxes,
            )
    except (ValueError, TypeError, KeyError, IndexError) as e:
        ax_kl.text(
            0.5,
            0.5,
            f"Error plotting KL matrix:\n{str(e)}",
            ha="center",
            va="center",
            transform=ax_kl.transAxes,
        )

    # Panel 2: Convergence diagnostics (top right)
    ax_conv = fig.add_subplot(gs[0, 1])
    try:
        # Plot ESS for all parameters
        if result.per_shard_diagnostics is not None:
            num_shards = len(result.per_shard_diagnostics)

            # Define canonical parameter names first, then derive count
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
                _count = (
                    len(result.mean_params) if result.mean_params is not None else 0
                )
                param_names = [f"P{i}" for i in range(_count)]
            num_params = len(param_names)

            # Collect ESS values
            ess_list: list[list[float]] = []
            for shard_diag in result.per_shard_diagnostics:
                if "ess" in shard_diag and shard_diag["ess"] is not None:
                    ess_vals = [
                        shard_diag["ess"].get(f"param_{i}", np.nan)
                        for i in range(num_params)
                    ]
                    ess_list.append(ess_vals)

            if ess_list:
                ess_matrix = np.array(ess_list)

                # Plot as box plot
                positions = np.arange(num_params)
                import matplotlib as _mpl

                _mpl_ver = tuple(int(x) for x in _mpl.__version__.split(".")[:2])
                _bp_kwargs: dict = {
                    "positions": positions,
                    "patch_artist": True,
                }
                if _mpl_ver >= (3, 9):
                    _bp_kwargs["tick_labels"] = param_names
                else:
                    _bp_kwargs["labels"] = param_names
                bp = ax_conv.boxplot(
                    [ess_matrix[:, i] for i in range(num_params)],
                    **_bp_kwargs,
                )

                # Color boxes
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")

                ax_conv.axhline(
                    y=400,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label="ESS threshold (400)",
                )
                ax_conv.set_ylabel("Effective Sample Size", fontsize=9)
                ax_conv.set_title(
                    "ESS Distribution Across Shards", fontsize=10, fontweight="bold"
                )
                ax_conv.legend(fontsize=8)
                ax_conv.grid(True, alpha=0.3, axis="y")
        else:
            ax_conv.text(
                0.5,
                0.5,
                "Convergence diagnostics not available",
                ha="center",
                va="center",
                transform=ax_conv.transAxes,
            )
    except (ValueError, TypeError, KeyError, IndexError) as e:
        ax_conv.text(
            0.5,
            0.5,
            f"Error plotting convergence:\n{str(e)}",
            ha="center",
            va="center",
            transform=ax_conv.transAxes,
        )

    # Panel 3: Trace plots for up to 2 parameters (middle row, 2 columns)
    _n_params_total = len(result.mean_params) if result.mean_params is not None else 0
    num_trace_params = min(2, _n_params_total)
    for i in range(num_trace_params):
        ax_trace = fig.add_subplot(gs[1, i])

        try:
            # Plot traces for this parameter
            if result.per_shard_diagnostics is not None:
                num_shards = len(result.per_shard_diagnostics)
                colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, num_shards))

                for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
                    if "trace_data" in shard_diag:
                        trace_key = f"param_{i}"
                        if trace_key in shard_diag["trace_data"]:
                            trace = np.array(shard_diag["trace_data"][trace_key])

                            if trace.ndim == 2:
                                trace = trace[0, :]  # Use first chain only

                            ax_trace.plot(
                                trace, color=colors[shard_idx], alpha=0.6, linewidth=0.5
                            )

                if result.analysis_mode == "static":
                    param_names = ["D0", "alpha", "D_offset"]
                else:
                    param_names = [f"param_{i}" for i in range(len(result.mean_params))]

                ax_trace.set_xlabel("Sample Index", fontsize=9)
                ax_trace.set_ylabel(param_names[i], fontsize=9)
                ax_trace.set_title(
                    f"{param_names[i]} Trace", fontsize=10, fontweight="bold"
                )
                ax_trace.grid(True, alpha=0.3)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            ax_trace.text(
                0.5,
                0.5,
                f"Error:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax_trace.transAxes,
            )

    # Panel 4: Posterior histograms (bottom row)
    _n_hist_params_total = (
        len(result.mean_params) if result.mean_params is not None else 0
    )
    num_hist_params = min(2, _n_hist_params_total)
    for i in range(num_hist_params):
        ax_hist = fig.add_subplot(gs[2, i])

        try:
            # Plot posterior distribution
            if result.samples_params is not None:
                combined_samples = result.samples_params[:, i]

                ax_hist.hist(
                    combined_samples,
                    bins=30,
                    alpha=0.7,
                    color="steelblue",
                    density=True,
                )

                # Add vertical line for mean
                mean_val = result.mean_params[i]
                ax_hist.axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Mean: {mean_val:.2f}",
                )

                if result.analysis_mode == "static":
                    param_names = ["D0", "alpha", "D_offset"]
                else:
                    param_names = [f"param_{i}" for i in range(len(result.mean_params))]

                ax_hist.set_xlabel(param_names[i], fontsize=9)
                ax_hist.set_ylabel("Density", fontsize=9)
                ax_hist.set_title(
                    f"{param_names[i]} Posterior", fontsize=10, fontweight="bold"
                )
                ax_hist.legend(fontsize=8)
        except (ValueError, TypeError, KeyError, IndexError) as e:
            ax_hist.text(
                0.5,
                0.5,
                f"Error:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax_hist.transAxes,
            )

    # Add overall title
    fig.suptitle(
        f"CMC Summary Dashboard ({result.num_shards} shards, {result.analysis_mode})",
        fontsize=14,
        fontweight="bold",
    )

    # Save or show
    if save_path is not None:
        try:
            validated_path = validate_plot_save_path(save_path)
            if validated_path is not None:
                fig.savefig(validated_path, dpi=dpi, bbox_inches="tight")
                logger.info(f"CMC summary dashboard saved to {validated_path.name}")
        except (PathValidationError, ValueError) as e:
            logger.warning(f"Could not save CMC summary dashboard: {e}")

    if show:
        plt.show()
    elif save_path is not None:
        plt.close(fig)

    return fig
