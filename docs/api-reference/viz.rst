Visualization Module
====================

The :mod:`homodyne.viz` module provides comprehensive visualization utilities for homodyne scattering analysis, including experimental data plots, optimization results, and MCMC diagnostics.

.. contents:: Contents
   :local:
   :depth: 2

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

    from homodyne.viz import plot_experimental_data, plot_fit_comparison

    # Plot experimental C2 heatmaps
    fig = plot_experimental_data(
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
        phi_angles=phi_angles,
        title="Experimental Data"
    )
    fig.savefig("experimental_data.png", dpi=300)

    # Compare fit to data
    fig = plot_fit_comparison(
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
        c2_fit=c2_fit,
        phi_angles=phi_angles
    )
    fig.savefig("fit_comparison.png", dpi=300)

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

    from homodyne.viz import generate_nlsq_plots

    # Generate comprehensive NLSQ plots
    figures = generate_nlsq_plots(
        result=nlsq_result,
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
        phi_angles=phi_angles,
        output_dir="plots/"
    )

    # Figures include:
    # - Fit comparison heatmaps
    # - Residual plots
    # - Parameter correlation matrix
    # - Chi-squared evolution

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

    # Plot MCMC traces for all parameters
    fig = plot_trace_plots(
        arviz_data=cmc_result.arviz_data,
        title="MCMC Trace Plots"
    )
    fig.savefig("trace_plots.png", dpi=300)

**Trace plots show**:

- Time series of sampled parameter values
- Multiple chains (for convergence assessment)
- Burn-in period visualization
- Mixing quality

Convergence Diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_convergence_diagnostics

    # Plot R-hat and ESS diagnostics
    fig = plot_convergence_diagnostics(
        arviz_data=cmc_result.arviz_data
    )
    fig.savefig("convergence.png", dpi=300)

**Diagnostics include**:

- R-hat (Gelman-Rubin statistic): Should be < 1.01
- ESS (Effective Sample Size): Higher is better
- Autocorrelation plots
- Chain comparison

Posterior Distributions
~~~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_posterior_comparison

    # Plot posterior distributions
    fig = plot_posterior_comparison(
        arviz_data=cmc_result.arviz_data,
        prior_samples=prior_samples  # optional
    )
    fig.savefig("posteriors.png", dpi=300)

**Posterior plots show**:

- Marginal distributions for each parameter
- Credible intervals (94% HDI by default)
- Prior vs. posterior comparison
- Kernel density estimates

CMC Summary Dashboard
~~~~~~~~~~~~~~~~~~~~~

::

    from homodyne.viz import plot_cmc_summary_dashboard

    # Generate comprehensive CMC dashboard
    fig = plot_cmc_summary_dashboard(
        cmc_result=cmc_result,
        t1=t1,
        t2=t2,
        c2_exp=c2_exp,
        phi_angles=phi_angles
    )
    fig.savefig("cmc_dashboard.png", dpi=300)

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

    # Plot KL divergence between chains
    fig = plot_kl_divergence_matrix(
        arviz_data=cmc_result.arviz_data
    )
    fig.savefig("kl_divergence.png", dpi=300)

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

    from homodyne.viz import plot_c2_heatmap_fast

    # Fast rendering for large datasets
    fig = plot_c2_heatmap_fast(
        t1=t1,
        t2=t2,
        c2=c2_exp,
        title="Experimental C2"
    )
    fig.savefig("c2_fast.png", dpi=300)

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
        c2_fit=c2_fit,
        t1=t1,
        t2=t2
    )

    print(f"Mean deviation: {stats.mean_deviation:.4f}")
    print(f"RMS error: {stats.rms_error:.4f}")

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
