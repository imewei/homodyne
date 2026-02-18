.. _bayesian_inference:

Bayesian Inference with CMC
============================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- When Bayesian inference adds value beyond NLSQ
- The NLSQ warm-start → CMC pipeline
- How to configure and run CMC analysis
- The ``SamplingPlan`` and adaptive sampling
- How to use ArviZ for posterior analysis
- Key convergence diagnostics (R-hat, ESS, BFMI)

---

When to Use Bayesian Inference
-------------------------------

CMC (Consensus Monte Carlo) provides **full posterior distributions** over
parameters. Use it when:

- You need **publication-quality uncertainty estimates** with proper error bars
- The problem may have **multi-modal posteriors** (NLSQ only finds one mode)
- You want to **propagate uncertainties** into derived quantities
- You are near a **phase boundary** or regime change
- NLSQ gives suspiciously wide or irregular parameter uncertainties

NLSQ is sufficient when:

- Rapid characterization is needed (screening many samples)
- The posterior is known to be approximately Gaussian (linear regime)
- Computational resources are limited

---

The NLSQ Warm-Start Pipeline
------------------------------

The recommended workflow always uses NLSQ as a warm start for CMC.
This is not just a performance optimization: it also dramatically reduces
the rate of divergent NUTS transitions:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Workflow
     - Divergence Rate
     - Posterior Quality
   * - CMC cold start
     - ~28%
     - Often poor (biased)
   * - CMC with NLSQ warm-start
     - < 5%
     - Reliable

The warm-start also dramatically tightens posterior distributions:

.. list-table:: Impact of NLSQ Warm-Start on CMC
   :header-rows: 1
   :widths: 35 25 25 15

   * - Metric
     - Without Warm-Start
     - With Warm-Start
     - Improvement
   * - Uncertainty ratio (CMC/NLSQ)
     - 33--43x
     - 2--5x
     - 7--15x
   * - Convergence speed
     - Slow
     - Fast
     - 2--3x
   * - Divergence rate
     - High (>10%)
     - Low (<5%)
     - Stable
   * - Posterior agreement with NLSQ
     - May disagree
     - Good overlap
     - Consistent

**How warm-start works:**

When you pass ``nlsq_result`` to ``fit_mcmc_jax()``, informative priors are
constructed from NLSQ estimates: ``TruncatedNormal`` centered on the NLSQ value
with width ``3 * NLSQ_uncertainty`` (3-sigma). This focuses the sampler on the
physically relevant region while still allowing the posterior to differ from
the NLSQ point estimate.

**Full pipeline:**

.. code-block:: python

   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.optimization.cmc import fit_mcmc_jax
   from homodyne.utils.logging import get_logger, log_phase

   logger = get_logger(__name__)
   config = ConfigManager.from_yaml("config.yaml")
   data = load_xpcs_data("config.yaml")

   # Step 1: NLSQ warm-start (fast, point estimate)
   with log_phase("NLSQ"):
       nlsq_result = fit_nlsq_jax(data, config)
       logger.info(f"NLSQ: chi2_nu = {nlsq_result.reduced_chi_squared:.3f}")

   # Step 2: CMC with warm-start (Bayesian posterior)
   # Prepare pooled data for CMC (CMC requires flat arrays)
   import numpy as np
   c2_exp = data['c2_exp']         # (n_phi, n_t1, n_t2)
   phi = data['phi_angles_list']
   t1 = data['t1']
   t2 = data['t2']
   q = float(data['wavevector_q_list'][0])
   L = float(data.get('L', 5.0e6))

   # Pool all angles into flat arrays
   n_phi, n_t1, n_t2 = c2_exp.shape
   PHI, T1, T2 = np.meshgrid(phi, t1, t2, indexing='ij')
   c2_pooled = c2_exp.ravel()
   phi_pooled = PHI.ravel()
   t1_pooled = T1.ravel()
   t2_pooled = T2.ravel()

   # Keep only upper-triangle (t2 >= t1) to avoid duplicates
   mask = t2_pooled >= t1_pooled
   c2_pooled = c2_pooled[mask]
   phi_pooled = phi_pooled[mask]
   t1_pooled = t1_pooled[mask]
   t2_pooled = t2_pooled[mask]

   with log_phase("CMC"):
       cmc_result = fit_mcmc_jax(
           data=c2_pooled,
           t1=t1_pooled,
           t2=t2_pooled,
           phi=phi_pooled,
           q=q,
           L=L,
           analysis_mode=config.analysis_mode,
           cmc_config=config.get_cmc_config(),
           initial_values=config.get_initial_parameters(),
           parameter_space=config.get_parameter_space(),
           nlsq_result=nlsq_result,   # Key: warm-start priors
       )
       logger.info(f"CMC: divergences = {cmc_result.divergences}")

---

CMC Configuration
------------------

Key CMC settings in your YAML file:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: "auto"    # ALWAYS use auto
         sharding_strategy: "stratified" # angle-balanced shards
       per_shard_mcmc:
         num_warmup: 500        # Warmup samples per chain
         num_samples: 1500      # Posterior samples per chain
         num_chains: 4          # Number of NUTS chains
         max_tree_depth: 10     # NUTS depth (max 2^10 steps)
         adaptive_sampling: true # Reduce for small shards
       per_angle_mode: "auto"   # Match NLSQ per_angle_mode
       validation:
         max_divergence_rate: 0.10  # Reject shards > 10% divergences

**Shard size selection** (see :doc:`../04_practical_guides/configuration`
for the full table). Always use ``"auto"`` — the auto-selector accounts
for dataset size, angle count, and iteration count.

---

SamplingPlan and Adaptive Sampling
------------------------------------

The ``SamplingPlan`` class captures the actual warmup/samples used per shard
after adaptive scaling. Always use ``SamplingPlan`` in code that needs the
actual values:

.. code-block:: python

   from homodyne.optimization.cmc.sampler import SamplingPlan

   # Create plan from config with adaptive scaling
   plan = SamplingPlan.from_config(
       config=cmc_config,
       shard_size=5000,
       n_params=9,
   )

   print(f"Warmup: {plan.n_warmup}")      # May differ from config.num_warmup
   print(f"Samples: {plan.n_samples}")    # May differ from config.num_samples
   print(f"Adapted: {plan.was_adapted}")  # True if adaptive scaling applied

Adaptive scaling reduces warmup/samples for small shards to avoid NUTS
overhead dominating the computation:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Shard Size
     - Warmup
     - Samples
     - Reduction
   * - 50 pts
     - 140
     - 350
     - 75%
   * - 5K pts
     - 250
     - 750
     - 50%
   * - 50K+ pts
     - 500
     - 1500
     - None (default)

.. warning::

   Never read ``config.num_warmup`` or ``config.num_samples`` directly in
   sampling code paths. Use ``SamplingPlan.from_config()`` instead, which
   applies the correct adaptive scaling.

---

Chain Execution Methods
------------------------

For CPU multiprocessing, use ``parallel`` chains (the default):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Best For
     - Performance
   * - ``parallel``
     - **CPU multiprocessing** (default)
     - Uses pmap across 4 virtual JAX devices per worker. Full CPU utilization.
   * - ``vectorized``
     - Single-process debugging
     - Slower in multiprocessing (drops to 1–2 CPUs per worker)
   * - ``sequential``
     - Debugging / small tests
     - Runs chains one at a time

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         chain_method: "parallel"   # Default; best for multiprocessing

Empirical comparison (same workload, 4 chains):

- ``parallel``: 4.9 s wall time
- ``vectorized``: 101 s wall time

---

ArviZ Diagnostics
------------------

After CMC completes, use ArviZ for comprehensive posterior analysis:

.. code-block:: python

   import arviz as az

   idata = cmc_result.inference_data

   # 1. Summary table (R-hat, ESS, parameter estimates)
   summary = az.summary(idata)
   print(summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat', 'ess_bulk']])

   # 2. Check convergence criteria
   assert (summary['r_hat'] < 1.05).all(), "Some parameters have R-hat > 1.05"
   assert (summary['ess_bulk'] > 400).all(), "Some parameters have low ESS"

   # 3. Trace plots (visual convergence check)
   az.plot_trace(idata, var_names=["D0", "alpha", "D_offset"])

   # 4. Posterior distributions
   az.plot_posterior(idata)

   # 5. Pair plot (correlations between parameters)
   az.plot_pair(idata, var_names=["D0", "alpha"])

   # 6. BFMI (energy diagnostic)
   bfmi = az.bfmi(idata)
   if any(bfmi < 0.3):
       print("WARNING: Low BFMI — NUTS may not be exploring well")

**R-hat (Gelman-Rubin statistic):**

Values close to 1.0 indicate chains have converged to the same distribution.
Guideline: R-hat < 1.05 for all parameters.

**Effective Sample Size (ESS):**

The number of effectively independent samples. A rule of thumb is ESS > 400
for reliable posterior summaries.

**BFMI (Bayesian Fraction of Missing Information):**

Measures how well NUTS is exploring the posterior. Values < 0.3 suggest the
sampler is struggling and the results may be unreliable.

---

Posterior Comparison: NLSQ vs CMC
-----------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   param_name = "D0"

   # NLSQ: Gaussian approximation
   nlsq_mean = nlsq_params_dict[param_name]
   nlsq_std = nlsq_errors_dict[param_name]
   x = np.linspace(nlsq_mean - 4*nlsq_std, nlsq_mean + 4*nlsq_std, 200)
   from scipy.stats import norm
   nlsq_pdf = norm.pdf(x, nlsq_mean, nlsq_std)

   # CMC: actual posterior samples
   cmc_samples = cmc_result.samples[param_name].ravel()

   fig, ax = plt.subplots(figsize=(6, 4))
   ax.plot(x, nlsq_pdf, 'b-', label='NLSQ (Gaussian approx)')
   ax.hist(cmc_samples, bins=50, density=True, alpha=0.5, color='orange', label='CMC posterior')
   ax.axvline(np.mean(cmc_samples), color='red', linestyle='--', label='CMC mean')
   ax.set_xlabel(f"{param_name}")
   ax.set_ylabel("Probability density")
   ax.legend()
   plt.tight_layout()
   plt.show()

---

Quality Filtering
------------------

CMC automatically filters shards with too many divergences:

.. code-block:: yaml

   optimization:
     cmc:
       validation:
         max_divergence_rate: 0.10       # Reject shards with > 10% divergences
         require_nlsq_warmstart: false   # True = require warm-start (strict)

Shards that fail the quality filter are excluded from the final consensus.
A warning is logged for each filtered shard.

---

Checkpointing for Long Runs
-----------------------------

For large datasets where CMC may run for hours, enable checkpointing to allow
resume after interruption:

.. code-block:: yaml

   optimization:
     cmc:
       enable_checkpoints: true
       checkpoint_dir: "./cmc_checkpoints"

When ``enable_checkpoints`` is true, the backend saves completed shard
results after each shard finishes. If the process is interrupted and
restarted with the same configuration and checkpoint directory, it will
automatically resume from the last completed shard.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Setting
     - Purpose
   * - ``enable_checkpoints``
     - Enable checkpoint saving (default: ``false``)
   * - ``checkpoint_dir``
     - Directory for checkpoint files (default: ``"./checkpoints"``)

---

See Also
---------

- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ as warm-start
- :doc:`diagnostics` — Full diagnostics guide
- :doc:`../02_data_and_fitting/result_interpretation` — Reading CMCResult
- :doc:`../04_practical_guides/configuration` — CMC YAML configuration
