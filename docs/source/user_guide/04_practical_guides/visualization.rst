.. _visualization:

Plotting Results
================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- How to plot two-time correlation heatmaps
- How to plot fitted vs experimental C2
- How to visualize parameter distributions from CMC
- How to create publication-quality figures

---

Two-Time Correlation Heatmaps
-------------------------------

The most important diagnostic visualization is the :math:`C_2(t_1, t_2)`
heatmap for each azimuthal angle:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import matplotlib.colors as mcolors
   from homodyne.data import load_xpcs_data

   data = load_xpcs_data("config.yaml")
   c2 = data['c2_exp']       # (n_phi, n_t1, n_t2)
   t1 = data['t1']
   t2 = data['t2']
   phi = data['phi_angles_list']

   n_phi = len(phi)
   ncols = min(n_phi, 4)
   nrows = (n_phi + ncols - 1) // ncols
   fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)

   for i_phi, phi_val in enumerate(phi):
       row, col = divmod(i_phi, ncols)
       ax = axes[row][col]

       # Plot C2 with log-scale time axes
       im = ax.pcolormesh(
           t1, t2, c2[i_phi].T,
           cmap='hot', vmin=0.95, vmax=1.5,
           shading='auto'
       )
       ax.set_xscale('log')
       ax.set_yscale('log')
       ax.set_xlabel("t1 (s)")
       ax.set_ylabel("t2 (s)")
       ax.set_title(f"φ = {phi_val:.0f}°")
       plt.colorbar(im, ax=ax, label="C2")

   # Hide unused axes
   for idx in range(n_phi, nrows * ncols):
       row, col = divmod(idx, ncols)
       axes[row][col].set_visible(False)

   plt.suptitle("Two-Time Correlation Matrix", fontsize=14)
   plt.tight_layout()
   plt.savefig("c2_heatmaps.pdf", dpi=150, bbox_inches='tight')
   plt.show()

---

Fitted vs Experimental Cuts
------------------------------

To inspect fit quality, compare one-dimensional cuts through :math:`C_2`:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Extract the anti-diagonal (fixed lag time index = lag)
   def extract_diagonal_cut(c2, lag=1):
       """Extract C2 values at fixed lag index (t2 - t1 = lag steps)."""
       n = min(c2.shape[0], c2.shape[1]) - lag
       return np.array([c2[k, k + lag] for k in range(n)])

   c2_exp = data['c2_exp']   # (n_phi, n_t1, n_t2)
   phi = data['phi_angles_list']
   t1 = data['t1']

   fig, ax = plt.subplots(figsize=(8, 5))
   lags_to_plot = [1, 5, 20, 50]
   colors = plt.cm.viridis(np.linspace(0, 1, len(lags_to_plot)))

   for lag, color in zip(lags_to_plot, colors):
       for i_phi in range(len(phi)):
           cut = extract_diagonal_cut(c2_exp[i_phi], lag=lag)
           ax.plot(
               t1[:len(cut)], cut,
               color=color, alpha=0.3, linewidth=0.5
           )
   ax.set_xscale('log')
   ax.set_xlabel("t1 (s)")
   ax.set_ylabel("C2")
   ax.set_title("C2 diagonal cuts (all angles overlaid)")
   plt.tight_layout()
   plt.show()

---

NLSQ Fit Quality Plot
----------------------

Visualize the NLSQ result alongside experimental data:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # After fitting:
   result = fit_nlsq_jax(data, config)

   # Extract relevant data for one angle
   i_phi = 0  # First angle
   c2_exp_slice = data['c2_exp'][i_phi]   # (n_t1, n_t2)
   t1 = data['t1']
   t2 = data['t2']

   # Compute lag-time dependence (anti-diagonal)
   lag_times = []
   c2_exp_lag = []
   max_lag = min(50, c2_exp_slice.shape[0] - 1)

   for lag in range(1, max_lag):
       n = c2_exp_slice.shape[0] - lag
       c2_at_lag = np.array([c2_exp_slice[k, k+lag] for k in range(n)])
       lag_times.append(t1[lag] - t1[0])
       c2_exp_lag.append(np.mean(c2_at_lag))

   fig, ax = plt.subplots(figsize=(7, 4))
   ax.plot(lag_times, c2_exp_lag, 'ko', markersize=3, label='Experiment')
   # Model curve would be added here after computing with fitted parameters
   ax.set_xscale('log')
   ax.set_xlabel("Lag time (s)")
   ax.set_ylabel("C2")
   ax.set_title(f"Fit quality: chi^2_nu = {result.reduced_chi_squared:.3f}")
   ax.legend()
   plt.tight_layout()
   plt.show()

---

CMC Posterior Plots
---------------------

Use ArviZ for posterior visualization:

.. code-block:: python

   import arviz as az

   idata = cmc_result.inference_data

   # 1. Trace plots (time series of sampled values)
   az.plot_trace(idata, var_names=["D0", "alpha"])
   plt.suptitle("MCMC Trace Plots", y=1.02)
   plt.tight_layout()
   plt.savefig("trace_plots.pdf", bbox_inches='tight')

   # 2. Posterior distributions
   az.plot_posterior(
       idata,
       var_names=["D0", "alpha", "D_offset"],
       hdi_prob=0.95,
   )
   plt.suptitle("Posterior Distributions", y=1.02)
   plt.tight_layout()
   plt.savefig("posterior_distributions.pdf", bbox_inches='tight')

   # 3. Pair plot (parameter correlations)
   az.plot_pair(
       idata,
       var_names=["D0", "alpha", "gamma_dot_0"],
       divergences=True,    # Mark divergent transitions
       figsize=(8, 8),
   )
   plt.suptitle("Parameter Pair Plot", y=1.02)
   plt.tight_layout()
   plt.savefig("pair_plot.pdf", bbox_inches='tight')

---

Parameter Evolution Across Samples
-------------------------------------

Plot how a parameter (e.g., D₀) changes across a series of experiments:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Collected results from batch processing
   labels = ["Sample A", "Sample B", "Sample C", "Sample D"]
   D0_vals = np.array([1245.3, 1389.2, 1102.8, 1560.4])
   D0_errs = np.array([45.2, 67.8, 38.1, 89.5])

   fig, ax = plt.subplots(figsize=(7, 4))
   x = np.arange(len(labels))
   ax.errorbar(x, D0_vals, yerr=D0_errs, fmt='o-', capsize=5,
               color='steelblue', linewidth=1.5, markersize=6)
   ax.axhline(np.mean(D0_vals), color='gray', linestyle='--', label='Mean')
   ax.fill_between(
       [-0.5, len(labels) - 0.5],
       [np.mean(D0_vals) - np.std(D0_vals)] * 2,
       [np.mean(D0_vals) + np.std(D0_vals)] * 2,
       alpha=0.2, color='gray', label='±1 std'
   )
   ax.set_xticks(x)
   ax.set_xticklabels(labels)
   ax.set_ylabel("D₀ (Å²/s)")
   ax.set_title("Diffusion Coefficient Across Samples")
   ax.legend()
   plt.tight_layout()
   plt.savefig("D0_comparison.pdf", bbox_inches='tight')

---

Publication-Quality Figures
-----------------------------

For publication figures, set matplotlib style and use tight layouts:

.. code-block:: python

   import matplotlib as mpl
   import matplotlib.pyplot as plt

   # Use a publication-friendly style
   plt.rcParams.update({
       'font.family': 'serif',
       'font.size': 10,
       'axes.labelsize': 11,
       'axes.titlesize': 11,
       'xtick.labelsize': 9,
       'ytick.labelsize': 9,
       'legend.fontsize': 9,
       'figure.dpi': 300,
       'axes.spines.top': False,
       'axes.spines.right': False,
   })

   # Single column width: 3.5 inches; double: 7 inches (typical journal)
   fig, ax = plt.subplots(figsize=(3.5, 3.0))

   # ... your plot ...

   plt.tight_layout(pad=0.4)
   plt.savefig("figure_1.pdf", bbox_inches='tight', dpi=300)

---

See Also
---------

- :doc:`../02_data_and_fitting/result_interpretation` — Interpreting results
- :doc:`../03_advanced_topics/bayesian_inference` — CMC posterior analysis
- :doc:`../05_appendices/troubleshooting` — Plot rendering issues
