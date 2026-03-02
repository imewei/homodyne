Visualization Module
====================

The :mod:`homodyne.viz` module provides comprehensive visualization utilities for homodyne scattering analysis, including experimental data plots, optimization results, and MCMC diagnostics.


Overview
--------

**Visualization Categories**:

- **Experimental Data**: C2 heatmaps, time series, angular dependence
- **NLSQ Results**: Fit comparisons, residuals, parameter correlations
- **MCMC Diagnostics**: Trace plots, posterior distributions, convergence metrics
- **Performance**: Datashader-accelerated plotting for large datasets

**Key Features**:

- Adaptive color scaling for C2 heatmaps
- High-performance rendering with Datashader (optional)
- ArviZ integration for MCMC diagnostics
- Publication-quality figure generation

Module Contents
---------------

.. automodule:: homodyne.viz
   :members:
   :undoc-members:
   :show-inheritance:

Experimental Data Plots
-----------------------

Visualization of experimental XPCS data.

.. automodule:: homodyne.viz.experimental_plots
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.viz.experimental_plots.plot_experimental_data
   homodyne.viz.experimental_plots.plot_fit_comparison

C2 Heatmap Plotting
~~~~~~~~~~~~~~~~~~~

**Adaptive Color Scaling**:

All C2 heatmap plots use conditional color scaling logic:

.. code-block:: python

    vmin = max(1.0, c2_min)  # Use data_min if >= 1.0, else clamp to 1.0
    vmax = min(1.6, c2_max)  # Use data_max if <= 1.6, else clamp to 1.6

This ensures physically meaningful visualization while preventing extreme outliers from dominating the color scale.

Usage Example
~~~~~~~~~~~~~

::

    from pathlib import Path
    from homodyne.viz import plot_experimental_data, plot_fit_comparison

    # Plot experimental C2 heatmaps (saves to plots_dir)
    plot_experimental_data(
        data=data,          # dict from XPCSDataLoader
        plots_dir=Path("plots/"),
    )

    # Compare fit to data (saves to plots_dir)
    plot_fit_comparison(
        result=nlsq_result,  # optimization result object
        data=data,
        plots_dir=Path("plots/"),
    )

NLSQ Optimization Plots
------------------------

Visualization of NLSQ optimization results.

.. automodule:: homodyne.viz.nlsq_plots
   :members:
   :undoc-members:
   :show-inheritance:

Plot Generation
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.viz.nlsq_plots.generate_nlsq_plots
   homodyne.viz.nlsq_plots.plot_simulated_data
   homodyne.viz.nlsq_plots.generate_and_plot_fitted_simulations

Usage Example
~~~~~~~~~~~~~

::

    from pathlib import Path
    from homodyne.viz import generate_nlsq_plots

    # Generate comprehensive NLSQ plots (saves to output_dir)
    generate_nlsq_plots(
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        c2_theoretical_scaled=c2_fitted,
        residuals=residuals,
        t1=t1,
        t2=t2,
        output_dir=Path("plots/"),
    )

    # Output includes:
    # - Fit comparison heatmaps
    # - Residual plots

MCMC Diagnostic Plots
----------------------

Comprehensive MCMC convergence and posterior diagnostics.

.. automodule:: homodyne.viz.mcmc_plots
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostic Functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.viz.mcmc_plots.plot_trace_plots
   homodyne.viz.mcmc_plots.plot_convergence_diagnostics
   homodyne.viz.mcmc_plots.plot_posterior_comparison
   homodyne.viz.mcmc_plots.plot_kl_divergence_matrix
   homodyne.viz.mcmc_plots.plot_cmc_summary_dashboard

Trace Plots
~~~~~~~~~~~

::

    from homodyne.viz import plot_trace_plots

    # Plot MCMC traces for all parameters (saves to save_path)
    plot_trace_plots(
        result=cmc_result,
        save_path="trace_plots.png",
    )

**Trace plots show**:

- Time series of sampled parameter values
- Multiple chains (for convergence assessment)
- Burn-in period visualization
- Mixing quality

Convergence Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_convergence_diagnostics

    # Plot R-hat and ESS diagnostics (saves to save_path)
    plot_convergence_diagnostics(
        result=cmc_result,
        save_path="convergence.png",
    )

**Diagnostics include**:

- R-hat (Gelman-Rubin statistic): Should be < 1.01
- ESS (Effective Sample Size): Higher is better
- Autocorrelation plots
- Chain comparison

Posterior Distributions
~~~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_posterior_comparison

    # Plot posterior distributions (saves to save_path)
    plot_posterior_comparison(
        result=cmc_result,
        save_path="posteriors.png",
    )

**Posterior plots show**:

- Marginal distributions for each parameter
- Credible intervals (94% HDI by default)
- Prior vs. posterior comparison
- Kernel density estimates

CMC Summary Dashboard
~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_cmc_summary_dashboard

    # Generate comprehensive CMC dashboard (saves to save_path)
    plot_cmc_summary_dashboard(
        result=cmc_result,
        save_path="cmc_dashboard.png",
    )

**Dashboard includes**:

- Trace plots
- Posterior distributions
- Fit comparison
- Convergence metrics
- Parameter correlations

KL Divergence Matrix
~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_kl_divergence_matrix

    # Plot KL divergence between chains (saves to save_path)
    plot_kl_divergence_matrix(
        result=cmc_result,
        save_path="kl_divergence.png",
    )

**KL divergence**:

- Measures agreement between chains
- Low KL = good convergence
- High KL = chains exploring different modes

Datashader Backend (Optional)
------------------------------

High-performance visualization for large datasets using Datashader.

.. automodule:: homodyne.viz.datashader_backend
   :members:
   :undoc-members:
   :show-inheritance:

**Installation**::

    pip install datashader xarray colorcet

Fast Plotting
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.viz.datashader_backend.plot_c2_heatmap_fast
   homodyne.viz.datashader_backend.plot_c2_comparison_fast
   homodyne.viz.datashader_backend.DatashaderRenderer

Usage Example
~~~~~~~~~~~~~

::

    from pathlib import Path
    from homodyne.viz import plot_c2_heatmap_fast

    # Fast rendering for large datasets (saves directly to output_path)
    plot_c2_heatmap_fast(
        c2_data=c2_exp,
        t1=t1,
        t2=t2,
        output_path=Path("c2_fast.png"),
        title="Experimental C2",
    )

**Performance**:

- 10-100x faster than matplotlib for large arrays
- Automatic downsampling for display
- Preserves visual fidelity
- Ideal for > 1000x1000 arrays

Visualization Diagnostics
--------------------------

Quantitative diagnostics for visualizations.

.. automodule:: homodyne.viz.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Diagonal Overlay Stats
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.viz.diagnostics.compute_diagonal_overlay_stats
   homodyne.viz.diagnostics.DiagonalOverlayResult

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.viz import compute_diagonal_overlay_stats

    # Compute statistics for diagonal overlay
    stats = compute_diagonal_overlay_stats(
        c2_exp=c2_exp,
        c2_solver=c2_solver,
        c2_posthoc=c2_posthoc,
        phi_index=0,
    )

    # stats is a DiagonalOverlayResult with trace arrays and statistics

Publication-Quality Figures
----------------------------

**Recommended Settings**::

    import matplotlib.pyplot as plt

    # Set publication style
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

    # Generate plot
    from homodyne.viz import plot_experimental_data
    fig = plot_experimental_data(...)

    # Save with tight layout
    fig.tight_layout()
    fig.savefig("figure.pdf", bbox_inches='tight')

**Figure Formats**:

- PNG: Web display, presentations
- PDF: Publications, vector graphics
- SVG: Editable vector graphics
- EPS: Legacy publications

Color Maps
~~~~~~~~~~

**Default C2 Heatmap Colormap**: ``viridis``

- Perceptually uniform
- Colorblind-friendly
- Good for grayscale printing

**Alternative Colormaps**:

- ``plasma``: Higher contrast
- ``inferno``: Warmer tones
- ``cividis``: Colorblind-optimized

**Custom Colormap**::

    fig = plot_c2_heatmap_fast(
        ...,
        cmap='plasma'
    )

Best Practices
--------------

**Heatmap Visualization**:

1. Use adaptive color scaling (automatic)
2. Include colorbars with physical units
3. Label axes with time in seconds
4. Add title with phi angle for multi-angle plots

**MCMC Diagnostics**:

1. Always check trace plots for mixing
2. Verify R-hat < 1.01 for all parameters
3. Check ESS > 400 per chain (minimum)
4. Plot posteriors vs. priors to assess learning

**Performance**:

1. Use Datashader for datasets > 1000x1000
2. Reduce DPI for exploratory plots (150)
3. Save high-res only for final figures (300+)
4. Close figures after saving to free memory

See Also
--------

- :mod:`homodyne.optimization` - Optimization results to visualize
- :mod:`homodyne.data` - Data loading for visualization
- External: `ArviZ Documentation <https://arviz-devs.github.io/arviz/>`_
- External: `Datashader Documentation <https://datashader.org/>`_
