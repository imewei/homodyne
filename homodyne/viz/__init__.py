"""Visualization backends for homodyne XPCS analysis.

This module provides high-performance visualization backends for C2 heatmap plotting
and comprehensive MCMC diagnostic plots.
"""

from homodyne.viz.datashader_backend import (
    DatashaderRenderer,
    plot_c2_comparison_fast,
    plot_c2_heatmap_fast,
)
from homodyne.viz.mcmc_plots import (
    plot_cmc_summary_dashboard,
    plot_convergence_diagnostics,
    plot_kl_divergence_matrix,
    plot_posterior_comparison,
    plot_trace_plots,
)

__all__ = [
    # C2 heatmap visualization
    "DatashaderRenderer",
    "plot_c2_heatmap_fast",
    "plot_c2_comparison_fast",
    # MCMC diagnostic plots
    "plot_trace_plots",
    "plot_kl_divergence_matrix",
    "plot_convergence_diagnostics",
    "plot_posterior_comparison",
    "plot_cmc_summary_dashboard",
]
