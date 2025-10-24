"""MCMC Diagnostic Visualization Module

This module provides comprehensive diagnostic plots for MCMC results including
both standard NUTS MCMC and Consensus Monte Carlo (CMC) results.

Features:
- Trace plots for convergence visualization
- KL divergence matrix heatmaps for CMC shard agreement
- Convergence diagnostics (R-hat, ESS) visualization
- Posterior distribution comparison plots
- Comprehensive multi-panel summary dashboard

Supported Result Types:
- Standard NUTS MCMC results (single posterior)
- CMC results with per-shard diagnostics (multiple subposteriors)

References:
    Scott et al. (2016): "Bayes and big data: the consensus Monte Carlo algorithm"
    Gelman et al. (2013): "Bayesian Data Analysis"

Examples:
    # Plot trace plots for standard NUTS result
    >>> from homodyne.optimization.mcmc import fit_mcmc_jax
    >>> result = fit_mcmc_jax(data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5)
    >>> plot_trace_plots(result, save_path='traces.png')

    # Create CMC summary dashboard
    >>> result_cmc = fit_mcmc_jax(data=large_data, method='cmc', ...)
    >>> plot_cmc_summary_dashboard(result_cmc, save_path='cmc_summary.png')
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Type alias for MCMCResult
try:
    from homodyne.optimization.cmc.result import MCMCResult
except ImportError:
    # Fallback for non-CMC installations
    from homodyne.optimization.mcmc import MCMCResult


def plot_trace_plots(
    result: MCMCResult,
    param_names: Optional[List[str]] = None,
    max_params: int = 9,
    figsize: Optional[tuple] = None,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
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
    if result.samples_params is None:
        logger.warning("No parameter samples available for trace plots")
        return _create_empty_figure("No samples available")

    samples = result.samples_params
    num_params = samples.shape[-1] if samples.ndim >= 2 else 1

    # Limit number of parameters to plot
    num_params_to_plot = min(num_params, max_params)

    # Generate parameter names if not provided
    if param_names is None:
        if result.analysis_mode == "static_isotropic":
            param_names = ['D0', 'alpha', 'D_offset']
        elif result.analysis_mode == "laminar_flow":
            param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
                          'gamma_dot_t_offset', 'phi0']
        else:
            param_names = [f'param_{i}' for i in range(num_params)]

    # Ensure we have enough parameter names
    if len(param_names) < num_params_to_plot:
        param_names.extend([f'param_{i}' for i in range(len(param_names), num_params_to_plot)])

    # Calculate figure layout
    ncols = min(3, num_params_to_plot)
    nrows = (num_params_to_plot + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Check if this is a CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, 'is_cmc_result') else False

    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot multiple shard traces
        num_shards = len(result.per_shard_diagnostics)
        colors = plt.cm.tab10(np.linspace(0, 1, num_shards))

        for param_idx in range(num_params_to_plot):
            ax = axes[param_idx]

            # Plot trace for each shard
            for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
                if 'trace_data' in shard_diag:
                    trace_key = f'param_{param_idx}'
                    if trace_key in shard_diag['trace_data']:
                        trace = np.array(shard_diag['trace_data'][trace_key])

                        # Handle multi-chain traces (flatten if needed)
                        if trace.ndim == 2:
                            # Multi-chain: plot all chains
                            for chain_idx in range(trace.shape[0]):
                                ax.plot(trace[chain_idx, :],
                                       color=colors[shard_idx],
                                       alpha=0.7,
                                       linewidth=0.5,
                                       label=f'Shard {shard_idx}' if chain_idx == 0 else '')
                        else:
                            # Single chain
                            ax.plot(trace,
                                   color=colors[shard_idx],
                                   alpha=0.7,
                                   linewidth=0.8,
                                   label=f'Shard {shard_idx}')

            ax.set_xlabel('Sample Index')
            ax.set_ylabel(param_names[param_idx])
            ax.set_title(f'{param_names[param_idx]} Trace (CMC)')
            ax.grid(True, alpha=0.3)

            # Add legend for first subplot only (to avoid clutter)
            if param_idx == 0 and num_shards <= 10:
                ax.legend(loc='upper right', fontsize=8, ncol=2)

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

            ax.plot(trace, linewidth=0.5, alpha=0.8, color='steelblue')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel(param_names[param_idx])
            ax.set_title(f'{param_names[param_idx]} Trace')
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_params_to_plot, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    title = 'MCMC Trace Plots'
    if is_cmc:
        title += f' (CMC: {result.num_shards} shards)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Trace plots saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_kl_divergence_matrix(
    result: MCMCResult,
    figsize: tuple = (8, 7),
    cmap: str = 'coolwarm',
    threshold: float = 2.0,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
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
    is_cmc = result.is_cmc_result() if hasattr(result, 'is_cmc_result') else False

    if not is_cmc:
        raise ValueError("KL divergence matrix is only available for CMC results")

    # Extract KL matrix from diagnostics
    if result.cmc_diagnostics is None or 'kl_matrix' not in result.cmc_diagnostics:
        raise ValueError("KL divergence matrix not found in CMC diagnostics")

    kl_matrix = np.array(result.cmc_diagnostics['kl_matrix'])
    num_shards = kl_matrix.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(kl_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=max(threshold * 1.5, kl_matrix.max()))

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='KL Divergence')

    # Add threshold line on colorbar
    cbar.ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')

    # Annotate cells with KL values
    for i in range(num_shards):
        for j in range(num_shards):
            kl_val = kl_matrix[i, j]

            # Choose text color based on background
            text_color = 'white' if kl_val > threshold else 'black'

            # Add text annotation
            text = ax.text(j, i, f'{kl_val:.2f}',
                          ha="center", va="center",
                          color=text_color, fontsize=9)

            # Highlight problematic shards (KL > threshold)
            if kl_val > threshold and i != j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                          fill=False, edgecolor='red',
                                          linewidth=2))

    # Set labels and title
    ax.set_xticks(np.arange(num_shards))
    ax.set_yticks(np.arange(num_shards))
    ax.set_xticklabels([f'S{i}' for i in range(num_shards)])
    ax.set_yticklabels([f'S{i}' for i in range(num_shards)])
    ax.set_xlabel('Shard Index')
    ax.set_ylabel('Shard Index')
    ax.set_title(f'Between-Shard KL Divergence Matrix\n({num_shards} shards, threshold={threshold})',
                fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(num_shards) - 0.5, minor=True)
    ax.set_yticks(np.arange(num_shards) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"KL divergence matrix saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_convergence_diagnostics(
    result: MCMCResult,
    metrics: List[str] = ['rhat', 'ess'],
    figsize: Optional[tuple] = None,
    rhat_threshold: float = 1.1,
    ess_threshold: float = 100.0,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Plot convergence diagnostics (R-hat and ESS) for MCMC results.

    Visualizes convergence metrics to assess MCMC sampling quality.
    For CMC results, shows per-shard and combined diagnostics.

    Parameters
    ----------
    result : MCMCResult
        MCMC or CMC result object with convergence diagnostics
    metrics : list of str, default=['rhat', 'ess']
        Metrics to plot. Options: 'rhat', 'ess'
    figsize : tuple, optional
        Figure size (width, height). If None, auto-calculated
    rhat_threshold : float, default=1.1
        R-hat threshold for convergence (standard: 1.1)
    ess_threshold : float, default=100.0
        ESS threshold for adequate sampling (standard: 100)
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
    # Check if CMC result
    is_cmc = result.is_cmc_result() if hasattr(result, 'is_cmc_result') else False

    # Determine number of subplots
    num_metrics = len(metrics)

    if figsize is None:
        figsize = (10, 4 * num_metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Extract parameter names
    num_params = len(result.mean_params)
    if result.analysis_mode == "static_isotropic":
        param_names = ['D0', 'alpha', 'D_offset'][:num_params]
    elif result.analysis_mode == "laminar_flow":
        param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
                      'gamma_dot_t_offset', 'phi0'][:num_params]
    else:
        param_names = [f'param_{i}' for i in range(num_params)]

    # Plot each metric
    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        if metric == 'rhat':
            _plot_rhat(ax, result, param_names, rhat_threshold, is_cmc)
        elif metric == 'ess':
            _plot_ess(ax, result, param_names, ess_threshold, is_cmc)
        else:
            logger.warning(f"Unknown metric: {metric}")
            ax.text(0.5, 0.5, f'Unknown metric: {metric}',
                   ha='center', va='center', transform=ax.transAxes)

    # Add overall title
    title = 'MCMC Convergence Diagnostics'
    if is_cmc:
        title += f' (CMC: {result.num_shards} shards)'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Convergence diagnostics saved to {save_path}")

    if show:
        plt.show()

    return fig


def _plot_rhat(ax, result, param_names, threshold, is_cmc):
    """Helper function to plot R-hat diagnostics."""
    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot per-shard R-hat values
        num_shards = len(result.per_shard_diagnostics)
        num_params = len(param_names)

        # Collect R-hat values for each parameter across shards
        rhat_matrix = np.full((num_shards, num_params), np.nan)

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if 'rhat' in shard_diag and shard_diag['rhat'] is not None:
                for param_idx in range(num_params):
                    param_key = f'param_{param_idx}'
                    if param_key in shard_diag['rhat']:
                        rhat_matrix[shard_idx, param_idx] = shard_diag['rhat'][param_key]

        # Plot as grouped bar chart
        x = np.arange(num_params)
        width = 0.8 / num_shards if num_shards < 10 else 0.8 / 10

        for shard_idx in range(min(num_shards, 10)):  # Limit to first 10 shards for clarity
            offset = (shard_idx - num_shards / 2) * width
            values = rhat_matrix[shard_idx, :]

            # Color based on convergence
            colors = ['green' if v < threshold else 'red' for v in values]
            ax.bar(x + offset, values, width, label=f'Shard {shard_idx}', color=colors, alpha=0.7)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('R-hat')
        ax.set_title('R-hat Convergence Diagnostic (per shard)')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

    else:
        # Standard NUTS: Plot combined R-hat
        if result.r_hat is None:
            ax.text(0.5, 0.5, 'R-hat not available\n(requires multiple chains)',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Extract R-hat values
        rhat_values = []
        for param_name in param_names:
            # Try different key formats
            for key in [param_name, param_name.lower(), param_name.replace('_', '')]:
                if key in result.r_hat:
                    rhat_values.append(result.r_hat[key])
                    break
            else:
                rhat_values.append(np.nan)

        # Plot as bar chart
        x = np.arange(len(param_names))
        colors = ['green' if v < threshold else 'red' for v in rhat_values]

        ax.bar(x, rhat_values, color=colors, alpha=0.7)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('R-hat')
        ax.set_title('R-hat Convergence Diagnostic')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')


def _plot_ess(ax, result, param_names, threshold, is_cmc):
    """Helper function to plot ESS diagnostics."""
    if is_cmc and result.per_shard_diagnostics is not None:
        # CMC: Plot per-shard ESS values
        num_shards = len(result.per_shard_diagnostics)
        num_params = len(param_names)

        # Collect ESS values for each parameter across shards
        ess_matrix = np.full((num_shards, num_params), np.nan)

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if 'ess' in shard_diag and shard_diag['ess'] is not None:
                for param_idx in range(num_params):
                    param_key = f'param_{param_idx}'
                    if param_key in shard_diag['ess']:
                        ess_matrix[shard_idx, param_idx] = shard_diag['ess'][param_key]

        # Plot as grouped bar chart
        x = np.arange(num_params)
        width = 0.8 / num_shards if num_shards < 10 else 0.8 / 10

        for shard_idx in range(min(num_shards, 10)):  # Limit to first 10 shards
            offset = (shard_idx - num_shards / 2) * width
            values = ess_matrix[shard_idx, :]

            # Color based on adequacy
            colors = ['green' if v > threshold else 'orange' for v in values]
            ax.bar(x + offset, values, width, label=f'Shard {shard_idx}', color=colors, alpha=0.7)

        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Effective Sample Size (ESS)')
        ax.set_title('ESS Diagnostic (per shard)')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')

    else:
        # Standard NUTS: Plot combined ESS
        if result.effective_sample_size is None:
            ax.text(0.5, 0.5, 'ESS not available',
                   ha='center', va='center', transform=ax.transAxes)
            return

        # Extract ESS values
        ess_values = []
        for param_name in param_names:
            # Try different key formats
            for key in [param_name, param_name.lower(), param_name.replace('_', '')]:
                if key in result.effective_sample_size:
                    ess_values.append(result.effective_sample_size[key])
                    break
            else:
                ess_values.append(np.nan)

        # Plot as bar chart
        x = np.arange(len(param_names))
        colors = ['green' if v > threshold else 'orange' for v in ess_values]

        ax.bar(x, ess_values, color=colors, alpha=0.7)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Effective Sample Size (ESS)')
        ax.set_title('ESS Diagnostic')
        ax.set_xticks(x)
        ax.set_xticklabels(param_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')


def plot_posterior_comparison(
    result: MCMCResult,
    param_indices: Optional[List[int]] = None,
    figsize: Optional[tuple] = None,
    bins: int = 30,
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
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
    is_cmc = result.is_cmc_result() if hasattr(result, 'is_cmc_result') else False

    if not is_cmc:
        raise ValueError("Posterior comparison is only available for CMC results")

    if result.per_shard_diagnostics is None:
        raise ValueError("Per-shard diagnostics not available")

    # Extract samples
    num_params = result.samples_params.shape[-1] if result.samples_params is not None else 0

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
    if result.analysis_mode == "static_isotropic":
        param_names = ['D0', 'alpha', 'D_offset']
    elif result.analysis_mode == "laminar_flow":
        param_names = ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
                      'gamma_dot_t_offset', 'phi0']
    else:
        param_names = [f'param_{i}' for i in range(num_params)]

    # Plot each parameter
    for plot_idx, param_idx in enumerate(param_indices):
        ax = axes[plot_idx]

        # Extract per-shard samples
        num_shards = len(result.per_shard_diagnostics)
        colors = plt.cm.tab10(np.linspace(0, 1, num_shards))

        for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
            if 'trace_data' in shard_diag:
                trace_key = f'param_{param_idx}'
                if trace_key in shard_diag['trace_data']:
                    trace = np.array(shard_diag['trace_data'][trace_key])

                    # Flatten if multi-chain
                    if trace.ndim == 2:
                        trace = trace.flatten()

                    # Plot histogram
                    ax.hist(trace, bins=bins, alpha=0.3,
                           color=colors[shard_idx],
                           density=True,
                           label=f'Shard {shard_idx}' if num_shards <= 10 else '')

        # Plot combined posterior
        if result.samples_params is not None:
            combined_samples = result.samples_params[:, param_idx]
            ax.hist(combined_samples, bins=bins, alpha=0.5,
                   color='black', density=True,
                   histtype='step', linewidth=2,
                   label='Combined')

        ax.set_xlabel(param_names[param_idx])
        ax.set_ylabel('Density')
        ax.set_title(f'{param_names[param_idx]} Posterior')

        # Add legend for first subplot only
        if plot_idx == 0 and num_shards <= 10:
            ax.legend(loc='upper right', fontsize=8)

    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle(f'Posterior Comparison (CMC: {result.num_shards} shards)',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Posterior comparison saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_cmc_summary_dashboard(
    result: MCMCResult,
    figsize: tuple = (16, 12),
    show: bool = False,
    save_path: Optional[Union[str, Path]] = None,
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
    is_cmc = result.is_cmc_result() if hasattr(result, 'is_cmc_result') else False

    if not is_cmc:
        raise ValueError("Summary dashboard is only available for CMC results")

    # Create figure with custom layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: KL divergence matrix (top left)
    ax_kl = fig.add_subplot(gs[0, 0])
    try:
        if result.cmc_diagnostics is not None and 'kl_matrix' in result.cmc_diagnostics:
            kl_matrix = np.array(result.cmc_diagnostics['kl_matrix'])
            num_shards = kl_matrix.shape[0]
            threshold = 2.0

            im = ax_kl.imshow(kl_matrix, cmap='coolwarm', aspect='auto', vmin=0, vmax=max(threshold * 1.5, kl_matrix.max()))
            plt.colorbar(im, ax=ax_kl, label='KL Divergence')

            # Annotate cells
            for i in range(num_shards):
                for j in range(num_shards):
                    kl_val = kl_matrix[i, j]
                    text_color = 'white' if kl_val > threshold else 'black'
                    ax_kl.text(j, i, f'{kl_val:.2f}',
                             ha="center", va="center",
                             color=text_color, fontsize=7)

            ax_kl.set_xticks(np.arange(num_shards))
            ax_kl.set_yticks(np.arange(num_shards))
            ax_kl.set_xticklabels([f'S{i}' for i in range(num_shards)], fontsize=8)
            ax_kl.set_yticklabels([f'S{i}' for i in range(num_shards)], fontsize=8)
            ax_kl.set_xlabel('Shard Index', fontsize=9)
            ax_kl.set_ylabel('Shard Index', fontsize=9)
            ax_kl.set_title('KL Divergence Matrix', fontsize=10, fontweight='bold')
        else:
            ax_kl.text(0.5, 0.5, 'KL matrix not available',
                      ha='center', va='center', transform=ax_kl.transAxes)
    except Exception as e:
        ax_kl.text(0.5, 0.5, f'Error plotting KL matrix:\n{str(e)}',
                  ha='center', va='center', transform=ax_kl.transAxes)

    # Panel 2: Convergence diagnostics (top right)
    ax_conv = fig.add_subplot(gs[0, 1])
    try:
        # Plot ESS for all parameters
        if result.per_shard_diagnostics is not None:
            num_shards = len(result.per_shard_diagnostics)
            num_params = len(result.mean_params)

            # Get parameter names
            if result.analysis_mode == "static_isotropic":
                param_names = ['D0', 'alpha', 'D_offset'][:num_params]
            else:
                param_names = [f'P{i}' for i in range(num_params)]

            # Collect ESS values
            ess_matrix = []
            for shard_diag in result.per_shard_diagnostics:
                if 'ess' in shard_diag and shard_diag['ess'] is not None:
                    ess_vals = [shard_diag['ess'].get(f'param_{i}', np.nan)
                               for i in range(num_params)]
                    ess_matrix.append(ess_vals)

            if ess_matrix:
                ess_matrix = np.array(ess_matrix)

                # Plot as box plot
                positions = np.arange(num_params)
                bp = ax_conv.boxplot([ess_matrix[:, i] for i in range(num_params)],
                                    positions=positions,
                                    labels=param_names,
                                    patch_artist=True)

                # Color boxes
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')

                ax_conv.axhline(y=100, color='red', linestyle='--', linewidth=2, label='ESS threshold (100)')
                ax_conv.set_ylabel('Effective Sample Size', fontsize=9)
                ax_conv.set_title('ESS Distribution Across Shards', fontsize=10, fontweight='bold')
                ax_conv.legend(fontsize=8)
                ax_conv.grid(True, alpha=0.3, axis='y')
        else:
            ax_conv.text(0.5, 0.5, 'Convergence diagnostics not available',
                        ha='center', va='center', transform=ax_conv.transAxes)
    except Exception as e:
        ax_conv.text(0.5, 0.5, f'Error plotting convergence:\n{str(e)}',
                    ha='center', va='center', transform=ax_conv.transAxes)

    # Panel 3: Trace plots for first 3 parameters (middle row)
    num_trace_params = min(3, len(result.mean_params))
    for i in range(num_trace_params):
        ax_trace = fig.add_subplot(gs[1, i % 2 if num_trace_params <= 2 else 0])

        try:
            # Plot traces for this parameter
            if result.per_shard_diagnostics is not None:
                num_shards = len(result.per_shard_diagnostics)
                colors = plt.cm.tab10(np.linspace(0, 1, num_shards))

                for shard_idx, shard_diag in enumerate(result.per_shard_diagnostics):
                    if 'trace_data' in shard_diag:
                        trace_key = f'param_{i}'
                        if trace_key in shard_diag['trace_data']:
                            trace = np.array(shard_diag['trace_data'][trace_key])

                            if trace.ndim == 2:
                                trace = trace[0, :]  # Use first chain only

                            ax_trace.plot(trace, color=colors[shard_idx],
                                        alpha=0.6, linewidth=0.5)

                if result.analysis_mode == "static_isotropic":
                    param_names = ['D0', 'alpha', 'D_offset']
                else:
                    param_names = [f'param_{i}' for i in range(len(result.mean_params))]

                ax_trace.set_xlabel('Sample Index', fontsize=9)
                ax_trace.set_ylabel(param_names[i], fontsize=9)
                ax_trace.set_title(f'{param_names[i]} Trace', fontsize=10, fontweight='bold')
                ax_trace.grid(True, alpha=0.3)
        except Exception as e:
            ax_trace.text(0.5, 0.5, f'Error:\n{str(e)}',
                         ha='center', va='center', transform=ax_trace.transAxes)

    # Panel 4: Posterior histograms (bottom row)
    num_hist_params = min(2, len(result.mean_params))
    for i in range(num_hist_params):
        ax_hist = fig.add_subplot(gs[2, i])

        try:
            # Plot posterior distribution
            if result.samples_params is not None:
                combined_samples = result.samples_params[:, i]

                ax_hist.hist(combined_samples, bins=30, alpha=0.7,
                           color='steelblue', density=True)

                # Add vertical line for mean
                mean_val = result.mean_params[i]
                ax_hist.axvline(mean_val, color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {mean_val:.2f}')

                if result.analysis_mode == "static_isotropic":
                    param_names = ['D0', 'alpha', 'D_offset']
                else:
                    param_names = [f'param_{i}' for i in range(len(result.mean_params))]

                ax_hist.set_xlabel(param_names[i], fontsize=9)
                ax_hist.set_ylabel('Density', fontsize=9)
                ax_hist.set_title(f'{param_names[i]} Posterior', fontsize=10, fontweight='bold')
                ax_hist.legend(fontsize=8)
        except Exception as e:
            ax_hist.text(0.5, 0.5, f'Error:\n{str(e)}',
                        ha='center', va='center', transform=ax_hist.transAxes)

    # Add overall title
    fig.suptitle(f'CMC Summary Dashboard ({result.num_shards} shards, {result.analysis_mode})',
                fontsize=14, fontweight='bold')

    # Save or show
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"CMC summary dashboard saved to {save_path}")

    if show:
        plt.show()

    return fig


def _create_empty_figure(message: str) -> Figure:
    """Create an empty figure with a message."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center',
           transform=ax.transAxes, fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
