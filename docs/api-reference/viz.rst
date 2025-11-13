homodyne.viz - Visualization
============================

.. automodule:: homodyne.viz
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.viz`` module provides visualization utilities for XPCS analysis results, including MCMC posterior diagnostics, correlation function plots, and result comparisons.

**Key Features:**

* **MCMC Diagnostics**: Posterior distributions, trace plots, corner plots
* **Correlation Plots**: Experimental and fitted g₂ correlation functions
* **Comparison Plots**: Experimental vs. theoretical comparisons
* **Residual Plots**: Fit residual visualization

Module Structure
----------------

The viz module is organized into several submodules:

* :mod:`homodyne.viz.mcmc_plots` - MCMC posterior visualization

Submodules
----------

homodyne.viz.mcmc_plots
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.viz.mcmc_plots
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

MCMC posterior visualization and diagnostics.

**Key Functions:**

* ``plot_posterior()`` - Plot posterior distributions
* ``plot_trace()`` - Plot MCMC trace plots
* ``plot_corner()`` - Plot corner (pair) plots
* ``plot_diagnostics()`` - Plot convergence diagnostics

**Usage Example:**

.. code-block:: python

   from homodyne.viz.mcmc_plots import plot_posterior, plot_trace, plot_corner
   import matplotlib.pyplot as plt

   # Assuming mcmc_result from fit_mcmc_jax()

   # Plot posterior distributions
   fig_posterior = plot_posterior(
       samples=mcmc_result.samples,
       param_names=['D0', 'alpha', 'D_offset']
   )
   plt.savefig('posterior_distributions.png')

   # Plot trace plots
   fig_trace = plot_trace(
       samples=mcmc_result.samples,
       param_names=['D0', 'alpha', 'D_offset']
   )
   plt.savefig('trace_plots.png')

   # Plot corner plot
   fig_corner = plot_corner(
       samples=mcmc_result.samples,
       param_names=['D0', 'alpha', 'D_offset']
   )
   plt.savefig('corner_plot.png')

Plotting Utilities
------------------

**Experimental Data Plots:**

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Plot g2 correlation heatmap
   fig, ax = plt.subplots(figsize=(8, 6))

   # IMPORTANT: Transpose for correct diagonal orientation
   im = ax.imshow(
       c2_exp[0].T,  # Note .T transpose
       origin='lower',
       extent=[t1[0], t1[-1], t2[0], t2[-1]],
       aspect='auto',
       cmap='jet'
   )

   ax.set_xlabel('t₁ (s)')
   ax.set_ylabel('t₂ (s)')
   ax.set_title('Experimental g₂ Correlation')
   plt.colorbar(im, ax=ax, label='g₂')

   plt.savefig('experimental_g2.png', dpi=300, bbox_inches='tight')

**Fitted Comparison Plots:**

.. code-block:: python

   # Plot experimental vs. fitted
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Experimental
   axes[0].imshow(c2_exp[0].T, origin='lower', aspect='auto')
   axes[0].set_title('Experimental')

   # Fitted
   axes[1].imshow(c2_fitted[0].T, origin='lower', aspect='auto')
   axes[1].set_title('Fitted')

   plt.tight_layout()
   plt.savefig('comparison.png', dpi=300, bbox_inches='tight')

**Residual Plots:**

.. code-block:: python

   # Compute residuals
   residuals = c2_exp - c2_fitted

   # Plot residuals
   fig, ax = plt.subplots(figsize=(8, 6))
   im = ax.imshow(
       residuals[0].T,
       origin='lower',
       cmap='jet',
       vmin=residuals[0].min(),
       vmax=residuals[0].max()
   )
   ax.set_title('Fit Residuals')
   plt.colorbar(im, ax=ax, label='Residual')
   plt.savefig('residuals.png', dpi=300, bbox_inches='tight')

Plot Customization
------------------

**Figure Size and DPI:**

.. code-block:: python

   # High-resolution plot for publication
   fig, ax = plt.subplots(figsize=(10, 8))
   # ... plotting code ...
   plt.savefig('figure.png', dpi=600, bbox_inches='tight')

**Color Maps:**

.. code-block:: python

   # Use perceptually uniform colormaps
   colormaps = ['jet', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']

   # For diverging data (residuals) - alternative to jet
   diverging = ['RdBu_r', 'seismic', 'coolwarm']

**LaTeX Rendering:**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Enable LaTeX rendering
   plt.rcParams.update({
       'text.usetex': True,
       'font.family': 'serif',
       'font.size': 12
   })

   # Use LaTeX in labels
   ax.set_xlabel(r'$t_1$ (s)')
   ax.set_ylabel(r'$t_2$ (s)')
   ax.set_title(r'$g_2(t_1, t_2)$ Correlation')

See Also
--------

* :doc:`../user-guide/examples` - Visualization guide
* :doc:`../advanced-topics/mcmc-uncertainty` - MCMC diagnostic interpretation
* :doc:`optimization` - Optimization module that produces results

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.viz import (
       plot_posterior,
       plot_trace,
       plot_corner,
       plot_diagnostics,
   )

**Related Functions:**

* :func:`homodyne.optimization.fit_mcmc_jax` - Produces MCMC results to visualize
* :func:`homodyne.cli.commands.plot_results` - CLI plotting integration
