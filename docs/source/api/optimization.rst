.. _api-optimization:

=======================
homodyne.optimization
=======================

The ``homodyne.optimization`` package provides two independent optimization
backends for XPCS parameter estimation:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Backend
     - Module
     - Best For
   * - **NLSQ**
     - :doc:`nlsq_adapter` / :doc:`nlsq_wrapper`
     - Fast point estimates; primary method
   * - **CMC**
     - :doc:`cmc`
     - Bayesian uncertainty quantification

----

Choosing Between NLSQ and CMC
------------------------------

Use **NLSQ** when you need:

- Fast parameter estimates (seconds to minutes per dataset)
- Initial values for a subsequent CMC run (warm-start)
- Exploratory analysis or parameter sensitivity studies
- Datasets up to ~1 billion points (streaming mode)

Use **CMC** when you need:

- Posterior distributions for uncertainty quantification
- Publication-quality error bars on physical parameters
- Evidence of multi-modality in the posterior
- Convergence diagnostics (R-hat, ESS, divergence rate)

.. tip::

   The recommended production workflow is always NLSQ first, then CMC with
   the NLSQ result as warm-start. This reduces CMC divergences from ~28 %
   to under 5 % and cuts warmup time significantly.

----

Common Workflow Patterns
------------------------

Pattern 1: NLSQ only (fast, no uncertainties)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.config.manager import ConfigManager
   from homodyne.data.xpcs_loader import XPCSDataLoader

   config_manager = ConfigManager("my_config.yaml")
   loader = XPCSDataLoader(config_manager)
   data = loader.load()

   result = fit_nlsq_jax(data, config_manager.config)
   print(result.params)        # best-fit physical parameters
   print(result.chi_squared)   # goodness-of-fit

Pattern 2: NLSQ warm-start → CMC (recommended for publications)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.optimization.cmc import fit_mcmc_jax
   from homodyne.config.manager import ConfigManager
   from homodyne.data.xpcs_loader import XPCSDataLoader

   config_manager = ConfigManager("my_config.yaml")
   loader = XPCSDataLoader(config_manager)
   data = loader.load()

   # Step 1 — NLSQ warm-start
   nlsq_result = fit_nlsq_jax(data, config_manager.config)

   # Step 2 — CMC with warm-start (reduces divergences significantly)
   cmc_result = fit_mcmc_jax(
       data,
       config_manager.config,
       nlsq_result=nlsq_result,
   )

   print(cmc_result.posterior_mean)     # posterior means
   print(cmc_result.posterior_std)      # posterior standard deviations
   print(cmc_result.diagnostics)        # R-hat, ESS, divergence rate

Pattern 3: CLI workflow (two-step)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Step 1: NLSQ
   homodyne --method nlsq --config config.yaml --output-dir results/

   # Step 2: CMC using pre-computed NLSQ warm-start
   homodyne --method cmc --config config.yaml \
            --nlsq-result results/ \
            --output-dir results/

----

Package Structure
-----------------

::

   homodyne/optimization/
   ├── nlsq/
   │   ├── core.py              # fit_nlsq_jax() — primary entry point
   │   ├── adapter.py           # NLSQAdapter (recommended, JIT caching)
   │   ├── wrapper.py           # NLSQWrapper (streaming, full anti-degeneracy)
   │   ├── config.py            # NLSQConfig dataclass
   │   ├── cmaes_wrapper.py     # CMA-ES global optimizer
   │   ├── anti_degeneracy_layer.py  # Per-angle scaling defense
   │   ├── strategies/          # Fitting strategy implementations
   │   └── validation/          # Input and result validation
   └── cmc/
       ├── core.py              # fit_mcmc_jax() — primary entry point
       ├── config.py            # CMCConfig dataclass
       ├── model.py             # XPCS model variants (5 models)
       ├── sampler.py           # SamplingPlan, NUTS execution
       ├── reparameterization.py  # Log-space priors, t_ref
       ├── priors.py            # Prior distribution builders
       └── backends/
           └── multiprocessing.py  # Multi-process execution
