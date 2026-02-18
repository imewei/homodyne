.. _result_interpretation:

Interpreting Results
====================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The structure of ``OptimizationResult`` and ``CMCResult`` objects
- How to interpret parameter values and their uncertainties
- What reduced chi-squared tells you
- How to perform residual analysis
- How to save and load results (JSON + NPZ format)

---

NLSQ Result Structure
----------------------

After calling ``fit_nlsq_jax``, you receive an ``OptimizationResult``:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data

   config = ConfigManager.from_yaml("config.yaml")
   data = load_xpcs_data("config.yaml")
   result = fit_nlsq_jax(data, config)

   # Convergence status
   print(result.convergence_status)   # "converged", "max_iter", or "failed"
   print(result.success)              # True if converged
   print(result.quality_flag)         # "good", "marginal", or "poor"

   # Parameter estimates
   print(result.parameters)           # np.ndarray of fitted values
   print(result.uncertainties)        # np.ndarray of 1-sigma errors
   print(result.covariance)           # Full covariance matrix

   # Fit quality
   print(result.chi_squared)          # Total chi^2
   print(result.reduced_chi_squared)  # chi^2 per degree of freedom

   # Performance
   print(result.iterations)           # Optimizer iterations
   print(result.execution_time)       # Seconds

**Getting parameter values by name:**

The ``result.parameters`` array follows the order defined by the parameter
space. To map names to values:

.. code-block:: python

   from homodyne.optimization.nlsq import _get_param_names

   # Get parameter names in the same order as result.parameters
   param_names = _get_param_names(config)

   params_dict = dict(zip(param_names, result.parameters))
   errors_dict = dict(zip(param_names, result.uncertainties))

   for name in param_names:
       print(f"{name:20s}: {params_dict[name]:.4g} ± {errors_dict[name]:.4g}")

---

Reduced Chi-Squared Interpretation
------------------------------------

The reduced chi-squared :math:`\chi^2_\nu = \chi^2 / (n - p)` is the most
important fit quality metric, where :math:`n` is the number of data points
and :math:`p` is the number of free parameters.

**Interpreting the value:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - :math:`\chi^2_\nu`
     - Interpretation and action
   * - 0.5–2.0
     - Acceptable fit. The model describes the data well.
   * - 2–5
     - Marginal fit. The model may be incomplete or outliers are present.
       Inspect residuals.
   * - > 5
     - Poor fit. Check: correct mode? correct q? correct gap distance?
   * - > 10
     - Very poor fit. Likely a systematic error in data or configuration.
   * - < 0.5
     - Unusually good. Possible over-estimation of uncertainties (sigma),
       or the model has too many parameters.

**Typical values for good XPCS fits:** 1.0–3.0

.. note::

   If your sigma array is set to the default ``0.01 × ones_like(c2_exp)``,
   the chi-squared value is on an arbitrary scale and only useful for
   **comparing** different fits, not for absolute goodness-of-fit assessment.
   Use experimental uncertainties if available.

---

Parameter Uncertainties
------------------------

NLSQ returns uncertainties estimated from the covariance matrix of the
least-squares problem:

.. math::

   \text{Cov}(\theta) \approx \left(J^T W J\right)^{-1}

where :math:`J` is the Jacobian and :math:`W = \text{diag}(1/\sigma_i^2)`.
The uncertainty for parameter :math:`k` is:

.. math::

   \delta\theta_k = \sqrt{\text{Cov}_{kk}}

**Limitations of NLSQ uncertainties:**

- They are only valid near the solution minimum (linear error propagation)
- They can be unreliable if the problem is near-degenerate (large condition number)
- They underestimate uncertainty in multi-modal posteriors

For rigorous uncertainty quantification, use CMC (see
:doc:`../03_advanced_topics/bayesian_inference`).

**Checking uncertainty reliability:**

.. code-block:: python

   import numpy as np

   # Condition number of covariance matrix
   eigvals = np.linalg.eigvalsh(result.covariance)
   if eigvals.min() <= 0:
       print("WARNING: Covariance matrix is not positive-definite")
       print("Uncertainties may be unreliable")
   else:
       cond = eigvals.max() / eigvals.min()
       print(f"Covariance condition number: {cond:.2e}")
       if cond > 1e10:
           print("WARNING: Near-singular covariance (degenerate problem)")

---

Residual Analysis
------------------

Residuals reveal systematic discrepancies between model and data:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from homodyne.core.jax_backend import compute_c2_theory

   # Compute model prediction at fitted parameters
   # (Implementation depends on your analysis mode)
   # Here we show the general approach:

   c2_exp = data['c2_exp']        # (n_phi, n_t1, n_t2)
   sigma = data.get('sigma', 0.01 * np.ones_like(c2_exp))

   # Residuals (normalized)
   # c2_model would be computed from fitted parameters
   # residuals = (c2_exp - c2_model) / sigma

   # Plot 2D heatmap of residuals
   # fig, axes = plt.subplots(1, n_phi, figsize=(4*n_phi, 4))
   # for i_phi in range(n_phi):
   #     im = axes[i_phi].imshow(residuals[i_phi], cmap='RdBu_r', vmin=-3, vmax=3)
   #     axes[i_phi].set_title(f"phi = {phi[i_phi]:.0f}°")
   # plt.colorbar(im, label="Residual (sigma units)")

**What to look for in residuals:**

- **Random residuals**: good fit
- **Systematic patterns** (stripes, quadrants): model is wrong or missing a component
- **Large residuals near the diagonal** (short lags): short-time dynamics not captured
- **Large residuals at specific angles**: angular dependence not fully modeled

---

CMC Result Structure
---------------------

After calling ``fit_mcmc_jax``, you receive a ``CMCResult``:

.. code-block:: python

   from homodyne.optimization.cmc import fit_mcmc_jax, CMCResult

   cmc_result: CMCResult = fit_mcmc_jax(
       data=c2_pooled, t1=t1, t2=t2, phi=phi,
       q=0.054, L=5.0e6,
       analysis_mode="laminar_flow",
       cmc_config=config.get_cmc_config(),
       initial_values=config.get_initial_parameters(),
       parameter_space=parameter_space,
       nlsq_result=nlsq_result,
   )

   # Posterior summaries
   print(cmc_result.parameters)      # Posterior mean values
   print(cmc_result.uncertainties)   # Posterior standard deviations
   print(cmc_result.param_names)     # Parameter names

   # MCMC diagnostics
   print(cmc_result.r_hat)           # dict: R-hat per parameter
   print(cmc_result.ess_bulk)        # dict: bulk ESS per parameter
   print(cmc_result.ess_tail)        # dict: tail ESS per parameter
   print(cmc_result.divergences)     # int: total divergent transitions
   print(cmc_result.convergence_status)  # "converged", "divergences", etc.

   # Raw posterior samples
   samples = cmc_result.samples      # dict: {name: (n_chains, n_samples)}

   # ArviZ InferenceData for analysis
   idata = cmc_result.inference_data

**Posterior analysis with ArviZ:**

.. code-block:: python

   import arviz as az

   idata = cmc_result.inference_data

   # Summary table
   summary = az.summary(idata, var_names=cmc_result.param_names)
   print(summary)

   # Trace plots
   az.plot_trace(idata, var_names=["D0", "alpha"])

   # Posterior distributions
   az.plot_posterior(idata, var_names=["D0", "alpha", "gamma_dot_0"])

   # Corner plot (pair plot)
   az.plot_pair(idata, var_names=["D0", "alpha", "gamma_dot_0"])

---

Convergence Diagnostics
------------------------

For CMC, check these diagnostics before trusting the results:

**R-hat (Gelman-Rubin statistic):**

.. math::

   \hat{R} = \sqrt{\frac{\hat{V}}{W}}

where :math:`\hat{V}` is the posterior variance estimate and :math:`W` is
the within-chain variance. Values near 1.0 indicate convergence.

.. code-block:: python

   for param, rhat in cmc_result.r_hat.items():
       if rhat > 1.05:
           print(f"WARNING: R-hat = {rhat:.3f} for {param} (should be < 1.05)")

**Effective Sample Size (ESS):**

ESS measures how many independent samples the chains are equivalent to:

.. code-block:: python

   for param, ess in cmc_result.ess_bulk.items():
       if ess < 400:
           print(f"WARNING: Bulk ESS = {ess:.0f} for {param} (should be >= 400)")

**Divergent transitions:**

.. code-block:: python

   n_total = cmc_result.n_chains * cmc_result.n_samples
   div_rate = cmc_result.divergences / n_total
   if div_rate > 0.10:
       print(f"HIGH divergence rate: {100*div_rate:.1f}% ({cmc_result.divergences} transitions)")
       print("Consider: increase max_tree_depth, check model specification")

---

Saving and Loading Results
---------------------------

Results are saved in JSON (parameters) and NPZ (arrays) format:

.. code-block:: python

   import json
   import numpy as np
   from pathlib import Path

   output_dir = Path("results/")
   output_dir.mkdir(exist_ok=True)

   # Save NLSQ result
   nlsq_dict = {
       "parameters": result.parameters.tolist(),
       "uncertainties": result.uncertainties.tolist(),
       "chi_squared": result.chi_squared,
       "reduced_chi_squared": result.reduced_chi_squared,
       "convergence_status": result.convergence_status,
       "execution_time": result.execution_time,
   }
   with open(output_dir / "nlsq_result.json", "w") as f:
       json.dump(nlsq_dict, f, indent=2)

   # Save arrays
   np.savez(
       output_dir / "nlsq_arrays.npz",
       parameters=result.parameters,
       uncertainties=result.uncertainties,
       covariance=result.covariance,
   )

   # Save CMC result (via ArviZ NetCDF format)
   if cmc_result is not None:
       cmc_result.inference_data.to_netcdf(str(output_dir / "cmc_posterior.nc"))
       print("Saved CMC posterior to cmc_posterior.nc")

**Loading saved results:**

.. code-block:: python

   import arviz as az

   # Load CMC posterior
   idata = az.from_netcdf("results/cmc_posterior.nc")
   summary = az.summary(idata)

   # Load NLSQ result
   with open("results/nlsq_result.json") as f:
       nlsq_dict = json.load(f)

---

See Also
---------

- :doc:`nlsq_fitting` — Running NLSQ fits
- :doc:`../03_advanced_topics/bayesian_inference` — CMC and posterior analysis
- :doc:`../03_advanced_topics/diagnostics` — Convergence diagnostics in depth
- :doc:`../04_practical_guides/visualization` — Plotting results
