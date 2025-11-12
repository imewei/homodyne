"""Visualization backends for homodyne XPCS analysis.

This module provides high-performance visualization backends for C2 heatmap plotting
and comprehensive MCMC diagnostic plots.
"""

try:
    from homodyne.viz.datashader_backend import (
        DatashaderRenderer,
        plot_c2_comparison_fast,
        plot_c2_heatmap_fast,
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    DatashaderRenderer = None

    def _missing_datashader(*_args, **_kwargs):
        raise ImportError(
            "Datashader-based plotting requires the 'datashader' dependency. "
            "Install with `pip install datashader xarray colorcet`."
        ) from None

    plot_c2_comparison_fast = _missing_datashader
    plot_c2_heatmap_fast = _missing_datashader
from homodyne.viz.diagnostics import (
    DiagonalOverlayResult,
    compute_diagonal_overlay_stats,
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
    "compute_diagonal_overlay_stats",
    "DiagonalOverlayResult",
    # MCMC diagnostic plots
    "plot_trace_plots",
    "plot_kl_divergence_matrix",
    "plot_convergence_diagnostics",
    "plot_posterior_comparison",
    "plot_cmc_summary_dashboard",
]
