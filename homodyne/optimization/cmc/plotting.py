"""ArviZ diagnostic plots for CMC results.

This module provides the 6 standard ArviZ diagnostic plots:
1. Pair plot (corner plot)
2. Forest plot
3. Energy plot
4. Autocorrelation plot
5. Rank plot
6. ESS plot
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az
import matplotlib.pyplot as plt

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.optimization.cmc.results import CMCResult

logger = get_logger(__name__)

# Default figure settings
DEFAULT_FIGSIZE = (12, 10)
DEFAULT_DPI = 150


def generate_diagnostic_plots(
    result: CMCResult,
    output_dir: Path,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
    param_subset: list[str] | None = None,
) -> list[Path]:
    """Generate all 6 ArviZ diagnostic plots.

    Parameters
    ----------
    result : CMCResult
        CMC result with inference_data.
    output_dir : Path
        Directory to save plots.
    figsize : tuple[int, int]
        Figure size in inches.
    dpi : int
        Figure resolution.
    param_subset : list[str] | None
        Subset of parameters to plot. If None, plots all.

    Returns
    -------
    list[Path]
        Paths to saved plot files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    idata = result.inference_data
    saved_plots: list[Path] = []

    # Filter parameters if subset specified
    if param_subset is not None:
        var_names = param_subset
    else:
        var_names = None

    # 1. Pair plot
    try:
        path = plot_pair(idata, output_dir, var_names, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate pair plot: {e}")

    # 2. Forest plot
    try:
        path = plot_forest(idata, output_dir, var_names, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate forest plot: {e}")

    # 3. Energy plot
    try:
        path = plot_energy(idata, output_dir, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate energy plot: {e}")

    # 4. Autocorrelation plot
    try:
        path = plot_autocorr(idata, output_dir, var_names, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate autocorr plot: {e}")

    # 5. Rank plot
    try:
        path = plot_rank(idata, output_dir, var_names, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate rank plot: {e}")

    # 6. ESS plot
    try:
        path = plot_ess(idata, output_dir, var_names, figsize, dpi)
        saved_plots.append(path)
    except Exception as e:
        logger.warning(f"Failed to generate ESS plot: {e}")

    logger.info(f"Generated {len(saved_plots)} diagnostic plots in {output_dir}")

    return saved_plots


def plot_pair(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate pair (corner) plot.

    Shows pairwise parameter correlations and marginal distributions.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Limit to physical parameters for readability
    if var_names is None:
        # Get parameter names, limit to avoid huge plots
        all_vars = list(idata.posterior.data_vars)
        # Prioritize physical parameters over per-angle scaling
        physical = [v for v in all_vars if not v.startswith(("contrast_", "offset_"))]
        scaling = [v for v in all_vars if v.startswith(("contrast_", "offset_"))]

        # Limit per-angle to first 3
        scaling_limited = scaling[:6]  # contrast_0,1,2 + offset_0,1,2
        var_names = physical + scaling_limited

    az.plot_pair(
        idata,
        var_names=var_names,
        kind="kde",
        figsize=figsize,
        marginals=True,
    )

    output_path = output_dir / "pair_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved pair plot: {output_path}")
    return output_path


def plot_forest(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate forest plot.

    Shows posterior distributions with HDI intervals.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=0.94,
        figsize=figsize,
    )

    output_path = output_dir / "forest_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved forest plot: {output_path}")
    return output_path


def plot_energy(
    idata: az.InferenceData,
    output_dir: Path,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate energy plot.

    Compares marginal energy distribution to energy transition distribution.
    Large differences indicate sampling problems.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Energy plot requires sample_stats with energy info
    has_energy = False
    if hasattr(idata, "sample_stats") and idata.sample_stats is not None:
        if hasattr(idata.sample_stats, "energy"):
            has_energy = True
        elif hasattr(idata.sample_stats, "potential_energy"):
            has_energy = True

    if not has_energy:
        # Create minimal figure with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Energy plot not available\n(energy/potential_energy not in sample_stats)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        output_path = output_dir / "energy_plot.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return output_path

    ax = az.plot_energy(idata, figsize=figsize)

    output_path = output_dir / "energy_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved energy plot: {output_path}")
    return output_path


def plot_autocorr(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate autocorrelation plot.

    Shows how quickly samples become independent.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Limit parameters for readability
    if var_names is None:
        all_vars = list(idata.posterior.data_vars)
        # Focus on physical parameters
        var_names = [v for v in all_vars if not v.startswith(("contrast_", "offset_"))]
        if len(var_names) == 0:
            var_names = all_vars[:6]

    az.plot_autocorr(
        idata,
        var_names=var_names,
        combined=True,
        figsize=figsize,
    )

    output_path = output_dir / "autocorr_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved autocorr plot: {output_path}")
    return output_path


def plot_rank(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate rank plot.

    Rank plots help identify chain mixing problems.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Limit parameters
    if var_names is None:
        all_vars = list(idata.posterior.data_vars)
        var_names = [v for v in all_vars if not v.startswith(("contrast_", "offset_"))]
        if len(var_names) == 0:
            var_names = all_vars[:6]

    az.plot_rank(
        idata,
        var_names=var_names,
        figsize=figsize,
    )

    output_path = output_dir / "rank_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved rank plot: {output_path}")
    return output_path


def plot_ess(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate ESS evolution plot.

    Shows how effective sample size grows with more samples.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Limit parameters
    if var_names is None:
        all_vars = list(idata.posterior.data_vars)
        var_names = [v for v in all_vars if not v.startswith(("contrast_", "offset_"))]
        if len(var_names) == 0:
            var_names = all_vars[:6]

    az.plot_ess(
        idata,
        var_names=var_names,
        kind="evolution",
        figsize=figsize,
    )

    output_path = output_dir / "ess_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved ESS plot: {output_path}")
    return output_path


def plot_trace(
    idata: az.InferenceData,
    output_dir: Path,
    var_names: list[str] | None = None,
    figsize: tuple[int, int] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """Generate trace plot (bonus diagnostic).

    Shows parameter values over sampling iterations.

    Parameters
    ----------
    idata : az.InferenceData
        ArviZ inference data.
    output_dir : Path
        Output directory.
    var_names : list[str] | None
        Parameters to include.
    figsize : tuple[int, int]
        Figure size.
    dpi : int
        Resolution.

    Returns
    -------
    Path
        Path to saved plot.
    """
    # Limit parameters
    if var_names is None:
        all_vars = list(idata.posterior.data_vars)
        var_names = [v for v in all_vars if not v.startswith(("contrast_", "offset_"))]
        if len(var_names) == 0:
            var_names = all_vars[:6]

    az.plot_trace(
        idata,
        var_names=var_names,
        figsize=figsize,
    )

    output_path = output_dir / "trace_plot.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved trace plot: {output_path}")
    return output_path
