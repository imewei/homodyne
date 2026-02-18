.. _cmc-quickstart-snippet:

CMC Bayesian Quick Start
~~~~~~~~~~~~~~~~~~~~~~~~

Run Consensus Monte Carlo Bayesian inference after an NLSQ warm-start:

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager
   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.optimization.cmc import fit_mcmc_jax

   config = ConfigManager.from_yaml("my_config.yaml")
   data   = XPCSDataLoader(config).load()

   # Step 1: NLSQ warm-start (recommended — reduces CMC divergences)
   nlsq_result = fit_nlsq_jax(data, config)

   # Step 2: CMC Bayesian inference
   cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)

   # cmc_result.posterior   — ArviZ InferenceData object
   # cmc_result.summary     — DataFrame with mean, sd, HDI, R-hat, ESS
   # cmc_result.diagnostics — divergences, BFMI, acceptance rates

.. tip::

   Always pass ``nlsq_result`` to ``fit_mcmc_jax()``. Empirical testing shows
   the warm-start reduces NUTS divergence rates from ~28% to <5% for
   laminar flow problems.

.. warning::

   CMC shard size must be chosen carefully. Use
   ``max_points_per_shard: "auto"`` (default) — never set it above 100,000
   for any analysis mode. See :doc:`/user_guide/03_advanced_topics/bayesian_inference`
   for a shard-size selection guide.

.. _end-cmc-quickstart-snippet:
