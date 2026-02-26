"""Experimental data plotting functions for homodyne XPCS analysis.

This module provides functions for visualizing raw experimental C2 correlation data.
Extracted from cli/commands.py for better modularity.
"""

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Set Agg backend only if no interactive backend is already active.
# Checking is_interactive() alone can be True for non-GUI backends in some
# environments; compare against the known interactive backend families instead.
_current_backend = matplotlib.get_backend().lower()
_interactive_backends = ("qt", "gtk", "wx", "tk", "macosx", "nbagg", "webagg")
if not any(_current_backend.startswith(b) for b in _interactive_backends):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from homodyne.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def plot_experimental_data(
    data: dict[str, Any],
    plots_dir: Path,
    angle_filter_func: Any | None = None,
) -> None:
    """Generate validation plots of experimental data.

    Parameters
    ----------
    data : dict[str, Any]
        Data dictionary containing:
        - c2_exp: Experimental correlation data (n_phi, n_t1, n_t2) or (n_t1, n_t2)
        - t1: Time array 1 (optional)
        - t2: Time array 2 (optional)
        - phi_angles_list: Phi angles in degrees (optional)
        - config: Configuration dict for angle filtering (optional)
    plots_dir : Path
        Output directory for plot files
    angle_filter_func : callable, optional
        Function to apply angle filtering. Signature:
        (phi_angles, c2_exp, data) -> (filtered_indices, filtered_phi, filtered_c2)
    """
    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        logger.warning("No experimental data to plot")
        return

    # Get time arrays if available for proper axis labels
    t1 = data.get("t1", None)
    t2 = data.get("t2", None)

    # Extract time extent for imshow if time arrays are available
    if t1 is not None and t2 is not None:
        t1_min, t1_max = float(np.min(t1)), float(np.max(t1))
        t2_min, t2_max = float(np.min(t2)), float(np.max(t2))
        extent = [t1_min, t1_max, t2_min, t2_max]  # [xmin, xmax, ymin, ymax] = [t1, t2]
        xlabel = "t₁ (s)"
        ylabel = "t₂ (s)"
        logger.debug(f"Using time extent: t1=[{t1_min:.3f}, {t1_max:.3f}], t2=[{t2_min:.3f}, {t2_max:.3f}] seconds")
    else:
        extent = None
        xlabel = "t₁ Index"
        ylabel = "t₂ Index"
        logger.debug("Time arrays not available, using frame indices")

    # Get phi angles array from data
    phi_angles_list = data.get("phi_angles_list", None)
    if phi_angles_list is None:
        logger.warning("phi_angles_list not found in data, using indices")
        phi_angles_list = np.arange(c2_exp.shape[0])

    # Apply angle filtering for plotting if configured and filter function provided
    if angle_filter_func is not None:
        filtered_indices, filtered_phi_angles, filtered_c2_exp = angle_filter_func(
            phi_angles_list, c2_exp, data
        )
    else:
        filtered_indices = list(range(len(phi_angles_list)))
        filtered_phi_angles = phi_angles_list
        filtered_c2_exp = c2_exp

    # Use filtered data for plotting
    phi_angles_list = filtered_phi_angles
    c2_exp = filtered_c2_exp

    logger.info(
        f"Plotting {len(filtered_indices)} angles after filtering: {filtered_phi_angles}",
    )

    # Handle different data shapes
    if c2_exp.ndim == 3:
        _plot_3d_experimental_data(
            c2_exp, phi_angles_list, t1, extent, xlabel, ylabel, plots_dir
        )
    elif c2_exp.ndim == 2:
        _plot_2d_experimental_data(c2_exp, extent, xlabel, ylabel, plots_dir)
    elif c2_exp.ndim == 1:
        _plot_1d_experimental_data(c2_exp, plots_dir)
    else:
        logger.warning(f"Unsupported data dimensionality: {c2_exp.ndim}D")
        return

    logger.debug(f"Plotted experimental data with shape {c2_exp.shape}")


def _plot_3d_experimental_data(
    c2_exp: np.ndarray,
    phi_angles_list: np.ndarray,
    t1: np.ndarray | None,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot 3D experimental data (n_phi, n_t1, n_t2)."""
    n_angles = c2_exp.shape[0]

    logger.info(f"Generating individual C₂ heatmaps for {n_angles} phi angles...")

    for angle_idx in range(n_angles):
        phi_deg = (
            phi_angles_list[angle_idx] if len(phi_angles_list) > angle_idx else 0.0
        )
        angle_data = c2_exp[angle_idx]

        fig, ax = plt.subplots(figsize=(8, 7))

        im = ax.imshow(
            angle_data.T,
            aspect="equal",
            cmap="jet",
            origin="lower",
            extent=extent,
        )
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"Experimental C₂(t₁, t₂) at φ={phi_deg:.1f}°",
            fontsize=13,
            fontweight="bold",
        )

        cbar = plt.colorbar(im, ax=ax, label="C₂", shrink=0.9)
        cbar.ax.tick_params(labelsize=9)

        # Calculate and display key statistics
        mean_val = np.mean(angle_data)
        max_val = np.max(angle_data)
        min_val = np.min(angle_data)

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

        plt.tight_layout()

        filename = f"experimental_data_phi_{phi_deg:.1f}.png"
        plt.savefig(plots_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  Saved: {filename}")

    logger.info(f"Generated {n_angles} individual C₂ heatmaps")

    # Plot diagonal (t1=t2) for all phi angles
    fig, ax = plt.subplots(figsize=(10, 6))

    if t1 is not None:
        time_diagonal = t1
    else:
        time_diagonal = np.arange(c2_exp.shape[-1])

    for idx in range(min(10, c2_exp.shape[0])):
        min_dim = min(c2_exp[idx].shape)
        diagonal = np.diag(c2_exp[idx][:min_dim, :min_dim])
        phi_deg = phi_angles_list[idx] if len(phi_angles_list) > idx else idx
        ax.plot(time_diagonal[:min_dim], diagonal, label=f"φ={phi_deg:.1f}°", alpha=0.7)

    ax.set_xlabel("Time (s)" if t1 is not None else "Time Index")
    ax.set_ylabel("C₂(t, t)")
    ax.set_title("C₂ Diagonal (t₁=t₂) for Different φ Angles")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.savefig(
        plots_dir / "experimental_data_diagonal.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def _plot_2d_experimental_data(
    c2_exp: np.ndarray,
    extent: list[float] | None,
    xlabel: str,
    ylabel: str,
    plots_dir: Path,
) -> None:
    """Plot 2D experimental data (single correlation matrix)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        c2_exp.T,
        aspect="equal",
        cmap="jet",
        origin="lower",
        extent=extent,
    )
    plt.colorbar(im, ax=ax, label="C₂(t₁,t₂)", shrink=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Experimental C₂ Data")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_1d_experimental_data(c2_exp: np.ndarray, plots_dir: Path) -> None:
    """Plot 1D experimental data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(c2_exp, marker="o", linestyle="-", alpha=0.7)
    ax.set_xlabel("Data Point Index")
    ax.set_ylabel("C₂")
    ax.set_title("Experimental C₂ Data")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "experimental_data.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_fit_comparison(
    result: Any,
    data: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate comparison plots between fit and experimental data.

    Parameters
    ----------
    result : Any
        Optimization result object
    data : dict[str, Any]
        Experimental data dictionary
    plots_dir : Path
        Output directory for plot files
    """
    c2_exp = data.get("c2_exp", None)
    if c2_exp is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot experimental data
    if c2_exp.ndim == 1:
        axes[0].plot(c2_exp, marker="o", linestyle="-", alpha=0.7, label="Experimental")
        axes[0].set_xlabel("Data Point Index")
        axes[0].set_ylabel("C₂")
    else:
        im0 = axes[0].imshow(c2_exp, aspect="auto", cmap="jet", vmin=1.0, vmax=1.5)
        plt.colorbar(im0, ax=axes[0], label="C₂")
        axes[0].set_xlabel("t₂ Index")
        axes[0].set_ylabel("φ Index")
    axes[0].set_title("Experimental Data")
    axes[0].grid(True, alpha=0.3)

    # Plot fit results placeholder
    axes[1].text(
        0.5,
        0.5,
        "Fit visualization\nrequires full\nplotting backend",
        ha="center",
        va="center",
        fontsize=14,
    )
    axes[1].set_title("Fit Results")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "fit_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Generated basic fit comparison plot")
