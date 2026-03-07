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
    >>> from homodyne.optimization.cmc import fit_mcmc_jax
    >>> result = fit_mcmc_jax(data=c2_exp, t1=t1, t2=t2, phi=phi, q=0.01, L=3.5)
    >>> plot_trace_plots(result, save_path='traces.png')

    # Create CMC summary dashboard
    >>> result_cmc = fit_mcmc_jax(data=large_data, method='cmc', ...)
    >>> plot_cmc_summary_dashboard(result_cmc, save_path='cmc_summary.png')
"""

from __future__ import annotations

# Re-export all public symbols from decomposed modules
from homodyne.viz.mcmc_arviz import (  # noqa: F401
    _create_empty_figure,
    plot_arviz_pair,
    plot_arviz_posterior,
    plot_arviz_trace,
)
from homodyne.viz.mcmc_comparison import plot_posterior_comparison  # noqa: F401
from homodyne.viz.mcmc_dashboard import plot_cmc_summary_dashboard  # noqa: F401
from homodyne.viz.mcmc_diagnostics import (  # noqa: F401
    plot_convergence_diagnostics,
    plot_kl_divergence_matrix,
    plot_trace_plots,
)
from homodyne.viz.mcmc_report import (  # noqa: F401
    generate_mcmc_diagnostic_report,
    print_mcmc_summary,
)

__all__ = [
    # Diagnostics (trace, KL, convergence)
    "plot_trace_plots",
    "plot_kl_divergence_matrix",
    "plot_convergence_diagnostics",
    # Comparison
    "plot_posterior_comparison",
    # Dashboard
    "plot_cmc_summary_dashboard",
    # ArviZ
    "_create_empty_figure",
    "plot_arviz_trace",
    "plot_arviz_posterior",
    "plot_arviz_pair",
    # Report
    "generate_mcmc_diagnostic_report",
    "print_mcmc_summary",
]
