.. _diagnostics:

Convergence Diagnostics
========================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- NLSQ convergence metrics (chi-squared, residuals, Jacobian condition)
- CMC convergence metrics (divergences, R-hat, ESS)
- Quality filtering for CMC shards
- How to systematically diagnose and fix convergence failures

---

NLSQ Diagnostics
-----------------

Reduced Chi-Squared
~~~~~~~~~~~~~~~~~~~~

The primary NLSQ fit quality metric:

.. code-block:: python

   result = fit_nlsq_jax(data, config)

   print(f"chi^2:         {result.chi_squared:.4g}")
   print(f"chi^2 / dof:   {result.reduced_chi_squared:.4f}")
   print(f"Quality flag:  {result.quality_flag}")    # "good", "marginal", "poor"

**Thresholds:**

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - chi^2_nu
     - quality_flag
     - Action
   * - < 2.0
     - ``good``
     - Proceed with results
   * - 2.0–5.0
     - ``marginal``
     - Inspect residuals; consider different starting values
   * - > 5.0
     - ``poor``
     - Check mode, q-value, gap; inspect data quality
   * - > 10.0
     - ``poor``
     - Likely systematic error in data or configuration

Convergence Status
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   print(result.convergence_status)
   # "converged":  optimizer reached the solution
   # "max_iter":   stopped by iteration limit (may still be a good fit)
   # "failed":     optimization failed (bad start or ill-conditioned problem)

``max_iter`` is not necessarily bad: check ``reduced_chi_squared``. If
chi^2_nu is acceptable, the result is usable even if the iteration limit
was reached.

Error Recovery Actions
~~~~~~~~~~~~~~~~~~~~~~~

The NLSQ optimizer attempts up to 3 retries with modified initial conditions.
Inspect the actions taken:

.. code-block:: python

   for action in result.recovery_actions:
       print(f"Recovery: {action}")
   # Example: "Retry 2/3: perturbed initial parameters by 10%"

Jacobian Condition Number
~~~~~~~~~~~~~~~~~~~~~~~~~~

A high condition number indicates near-degeneracy:

.. code-block:: python

   if result.nlsq_diagnostics:
       cond = result.nlsq_diagnostics.get('jacobian_condition', None)
       if cond is not None:
           print(f"Jacobian condition: {cond:.2e}")
           if cond > 1e10:
               print("Near-singular Jacobian: consider per_angle_mode='auto'")

Residual Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Compute residuals from the fit
   # (requires computing model at fitted parameters)
   # result.streaming_diagnostics may contain residual statistics

   if result.streaming_diagnostics:
       print(result.streaming_diagnostics.get('residual_rms', 'N/A'))

**Visual inspection:**

Plot the two-time correlation matrix for each angle:

- Smooth, symmetric off-diagonal features → good fit
- Striped patterns (horizontal or vertical) → systematic model error
- Isolated bright spots → outliers in data

---

CMC Diagnostics
----------------

R-hat (Gelman-Rubin Statistic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

R-hat measures convergence of multiple chains. Values near 1.0 are good:

.. code-block:: python

   for param, rhat in cmc_result.r_hat.items():
       status = "OK" if rhat < 1.05 else "WARNING"
       print(f"  R-hat[{param:20s}]: {rhat:.4f} [{status}]")

**Guideline:** R-hat < 1.05 for all parameters before trusting the posterior.

If R-hat is large (> 1.1) for some parameters:

1. Increase ``num_warmup`` (currently too short for the sampler to mix)
2. Increase ``num_samples`` (more samples to average away transient behavior)
3. Decrease ``max_tree_depth`` to prevent stuck chains

Effective Sample Size (ESS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ESS measures the number of independent samples:

.. code-block:: python

   for param, ess in cmc_result.ess_bulk.items():
       status = "OK" if ess >= 400 else "LOW"
       print(f"  ESS_bulk[{param:20s}]: {ess:.0f} [{status}]")

   for param, ess in cmc_result.ess_tail.items():
       status = "OK" if ess >= 400 else "LOW"
       print(f"  ESS_tail[{param:20s}]: {ess:.0f} [{status}]")

If ESS is low:

1. Increase ``num_samples``
2. Reduce ``max_tree_depth`` (long trajectory may cause correlation)
3. Check that chain method is ``"parallel"`` not ``"vectorized"``

Divergent Transitions
~~~~~~~~~~~~~~~~~~~~~~

Divergences indicate the sampler is leaving the posterior's support:

.. code-block:: python

   n_total = cmc_result.n_chains * cmc_result.n_samples
   div_rate = cmc_result.divergences / n_total
   print(f"Divergence rate: {100*div_rate:.1f}%  ({cmc_result.divergences}/{n_total})")

**Guidelines:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Divergence Rate
     - Action
   * - < 1%
     - Excellent; proceed
   * - 1–5%
     - Acceptable; note in analysis
   * - 5–10%
     - Marginal; check priors and bounds
   * - > 10%
     - Poor; shard is rejected by quality filter; investigate
   * - > 25%
     - CMC cold start signature; always use NLSQ warm-start

If divergences are high despite NLSQ warm-start:

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         max_tree_depth: 12   # Increase from 10 (more leapfrog steps)

BFMI (Bayesian Fraction of Missing Information)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BFMI measures how well NUTS explores the posterior energy landscape:

.. code-block:: python

   import arviz as az
   bfmi = az.bfmi(cmc_result.inference_data)
   print(f"BFMI: {bfmi}")

   if any(bfmi < 0.3):
       print("Low BFMI: possible funnel geometry or bad parameterization")

Low BFMI suggests the posterior has a challenging geometry. Consider:

1. Reparameterizing parameters (CMC does this automatically for :math:`D_0`)
2. Using informative priors to tighten the posterior
3. Increasing the number of warmup steps

ArviZ Comprehensive Check
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import arviz as az

   idata = cmc_result.inference_data

   # Check all diagnostics at once
   summary = az.summary(idata)
   print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk', 'ess_tail']])

   # Posterior pair plot (check for correlations and multi-modality)
   az.plot_pair(idata, var_names=cmc_result.param_names, divergences=True)

   # Energy plot
   az.plot_energy(idata)

---

Quality Filtering in CMC
--------------------------

CMC automatically filters shards that exceed the divergence threshold:

.. code-block:: yaml

   optimization:
     cmc:
       validation:
         max_divergence_rate: 0.10   # Reject shards with > 10% divergences

Filtered shards are **excluded from the final consensus**. A warning is
logged with the shard index and divergence rate.

Check the number of accepted vs filtered shards:

.. code-block:: python

   print(f"Total shards: {cmc_result.n_shards_total}")
   print(f"Accepted:     {cmc_result.n_shards_accepted}")
   print(f"Rejected:     {cmc_result.n_shards_rejected}")

   if cmc_result.n_shards_accepted < 0.5 * cmc_result.n_shards_total:
       print("WARNING: More than 50% of shards rejected. CMC result may be unreliable.")
       print("         Consider: increase shard size, use NLSQ warm-start,")
       print("         check data quality, or relax max_divergence_rate.")

---

Diagnosing Common Failures
---------------------------

**Systematic table of failure modes:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Symptom
     - Likely Cause and Fix
   * - NLSQ: ``convergence_status = "failed"``
     - Bad initial values → provide better initial params; check bounds
   * - NLSQ: ``reduced_chi_squared > 10``
     - Wrong mode or q-value; data quality issue; outliers
   * - NLSQ: Parameters at bounds
     - Degeneracy; enable ``per_angle_mode: "auto"``
   * - CMC: R-hat > 1.1
     - Insufficient warmup; increase ``num_warmup``
   * - CMC: ESS < 100
     - Too few samples; increase ``num_samples``
   * - CMC: Divergences > 20%
     - Missing NLSQ warm-start; bad priors; ``max_tree_depth`` too low
   * - CMC: BFMI < 0.3
     - Poor energy geometry; consider reparameterization
   * - CMC: > 50% shards rejected
     - Data quality; ``max_divergence_rate`` too strict; increase shard size

---

See Also
---------

- :doc:`bayesian_inference` — CMC setup and workflow
- :doc:`../02_data_and_fitting/result_interpretation` — Reading results
- :doc:`../05_appendices/troubleshooting` — Comprehensive troubleshooting guide
