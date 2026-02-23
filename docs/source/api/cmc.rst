.. _api-cmc:

==============================
homodyne.optimization.cmc
==============================

The ``cmc`` sub-package implements **Consensus Monte Carlo (CMC)** — a
divide-and-conquer Bayesian inference strategy for large-scale XPCS datasets.
The dataset is partitioned into independent shards, each processed with
per-shard NUTS/MCMC (via NumPyro), and the posterior distributions are
combined using precision-weighted Gaussian moments.

.. note::

   CMC is a secondary method. Always run NLSQ first and pass the result as a
   warm-start. This reduces divergence rates from ~28 % to under 5 %.
   See :doc:`optimization` for the recommended two-step workflow.

----

.. _api-fit-mcmc:

fit\_mcmc\_jax
--------------

The primary entry point for CMC analysis.

.. autofunction:: homodyne.optimization.cmc.core.fit_mcmc_jax

----

Shard Size Selection
---------------------

NUTS is :math:`O(n)` per leapfrog step — it evaluates all points in a shard.
Never use shard sizes above 100 K for any mode. The ``"auto"`` setting is
strongly recommended.

.. warning::

   Do not set ``max_points_per_shard`` above 100 K. Extremely large shards will
   cause NUTS to time out or exhaust memory.

.. note::

   **Single-shard hard limit**: If a dataset exceeds 100 K points and would run
   as a single shard (non-CMC path), homodyne automatically falls back to random
   CMC sharding to prevent NUTS from running :math:`O(n)` leapfrog on the full
   dataset. This also applies when ``num_shards=1`` is forced.

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Dataset size
     - laminar_flow auto
     - static auto
     - Notes
   * - < 2 M pts
     - 3–5 K
     - 5–6 K
     -
   * - 2–5 M pts
     - 3–5 K
     - 5–6 K
     -
   * - 5–50 M pts
     - 5 K
     - 6–10 K
     -
   * - 50–100 M pts
     - 5–8 K
     - 9–15 K
     -
   * - 100 M+ pts
     - 8–10 K
     - 12–20 K
     - HPC only

----

.. _api-cmc-config:

CMCConfig
---------

.. autoclass:: homodyne.optimization.cmc.config.CMCConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

CMCResult
---------

.. autoclass:: homodyne.optimization.cmc.results.CMCResult
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   **ArviZ field mapping**: NumPyro stores NUTS energy as ``potential_energy``
   but ArviZ ``plot_energy()`` expects ``energy``. The ``CMCResult.inference_data``
   attribute automatically maps ``potential_energy`` → ``energy`` and replaces dots
   in ``extra_fields`` keys (e.g., ``adapt_state.step_size`` →
   ``adapt_state_step_size``) for xarray compatibility.

----

Per-Angle Mode
--------------

The ``per_angle_mode`` parameter controls how contrast and offset are handled
across azimuthal angles. This must match the setting used in NLSQ to ensure
the warm-start is compatible.

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Mode
     - Behaviour
     - Parameters (laminar_flow, 23 angles)
   * - ``"auto"``
     - n_phi ≥ threshold → averaged scaling (9), n_phi < threshold → individual
     - 9 (7 physical + 2 optimised avg scaling)
   * - ``"constant"``
     - Fixed per-angle scaling from quantile estimation, not optimised
     - 7 (physical only)
   * - ``"individual"``
     - Independent contrast + offset per angle, all sampled
     - 53 (7 + 46)

.. tip::

   Always use ``per_angle_mode: "auto"`` (the default). It prevents parameter
   absorption degeneracy while keeping the parameter count manageable.

----

YAML Configuration Reference
-----------------------------

.. code-block:: yaml

   optimization:
     cmc:
       enable: true
       sharding:
         strategy: "random"               # random | stratified | contiguous
         max_points_per_shard: "auto"     # ALWAYS use auto
         min_points_per_param: 1500       # Minimum data points per parameter
       backend_name: "auto"               # auto | multiprocessing | pjit | pbs
       per_angle_mode: "auto"             # Match NLSQ per_angle_mode
       combination_method: "robust_consensus_mc"  # Default: MAD-based outlier detection
       min_success_rate: 0.80
       per_shard_mcmc:
         num_warmup: 500
         num_samples: 1500
         num_chains: 4
         chain_method: "parallel"         # parallel | vectorized | sequential
         target_accept_prob: 0.8
         max_tree_depth: 10
         adaptive_sampling: true
       validation:
         max_divergence_rate: 0.10
         max_r_hat: 1.1
         min_ess: 100

----

Quality Filtering
-----------------

CMC includes automatic quality filtering to prevent corrupted posteriors from
being included in the consensus combination:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Setting
     - Default
     - Purpose
   * - ``max_divergence_rate``
     - 0.10
     - Exclude shards with >10 % divergences
   * - ``require_nlsq_warmstart``
     - ``false``
     - Require NLSQ warm-start for ``laminar_flow``

----

Usage Examples
--------------

Standard NLSQ → CMC workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.optimization.cmc import fit_mcmc_jax
   from homodyne.config.manager import ConfigManager
   from homodyne.data.xpcs_loader import XPCSDataLoader

   config_manager = ConfigManager("config.yaml")
   data = XPCSDataLoader(config_manager).load()

   # Step 1 — NLSQ warm-start
   nlsq_result = fit_nlsq_jax(data, config_manager.config)

   # Step 2 — CMC with warm-start
   cmc_result = fit_mcmc_jax(
       data,
       config_manager.config,
       nlsq_result=nlsq_result,
   )

   print("Posterior means:", cmc_result.posterior_mean)
   print("Posterior stds:", cmc_result.posterior_std)

CMC without warm-start (exploratory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc import fit_mcmc_jax

   cmc_result = fit_mcmc_jax(data, config)   # uses prior-median initialisation

Accessing diagnostics
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   diagnostics = cmc_result.diagnostics

   for param, rhat in diagnostics["r_hat"].items():
       print(f"{param}: R-hat = {rhat:.3f}")

   print(f"Divergence rate: {diagnostics['divergence_rate']:.2%}")
   print(f"Effective sample size (min): {min(diagnostics['ess'].values()):.0f}")

----

.. automodule:: homodyne.optimization.cmc.core
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: fit_mcmc_jax
