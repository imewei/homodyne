.. _api-cmc-sampler:

=================================
homodyne.optimization.cmc.sampler
=================================

The ``sampler`` module manages NUTS (No-U-Turn Sampler) execution for each
CMC shard. It provides ``SamplingPlan`` — the single source of truth for
per-shard warmup and sample counts — and ``run_nuts_sampling()``, which
configures NumPyro's ``MCMC`` object and executes sampling.

.. note::

   Always use ``SamplingPlan.from_config()`` instead of accessing
   ``config.num_warmup`` / ``config.num_samples`` directly in sampling
   code paths. ``SamplingPlan`` applies adaptive scaling that can
   significantly reduce overhead for small shards.

----

Divergence Rate Constants
--------------------------

Centralized thresholds for NUTS convergence diagnostics:

.. list-table::
   :widths: 35 15 50
   :header-rows: 1

   * - Constant
     - Value
     - Meaning
   * - ``DIVERGENCE_RATE_TARGET``
     - 0.05
     - Below this: acceptable sampling quality
   * - ``DIVERGENCE_RATE_HIGH``
     - 0.10
     - Above this: posterior may be biased
   * - ``DIVERGENCE_RATE_CRITICAL``
     - 0.30
     - Above this: posterior likely unreliable; shard filtered

----

.. _api-sampling-plan:

SamplingPlan
------------

Captures the actual warmup and sample counts used per shard after adaptive
scaling. Instantiate via ``SamplingPlan.from_config()`` — never construct
directly in hot paths.

.. autoclass:: homodyne.optimization.cmc.sampler.SamplingPlan
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

SamplingPlan.from\_config
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: homodyne.optimization.cmc.sampler.SamplingPlan.from_config

----

Adaptive Sampling Behaviour
----------------------------

When ``adaptive_sampling: true`` (default), warmup and sample counts are
scaled down automatically for small shards to reduce NUTS overhead:

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Shard Size
     - Warmup
     - Samples
     - Total
     - Reduction
   * - 50 pts
     - 140
     - 350
     - 490
     - ~75 %
   * - 5 K pts
     - 250
     - 750
     - 1 000
     - ~50 %
   * - 50 K+ pts
     - 500
     - 1 500
     - 2 000
     - None

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         adaptive_sampling: true
         min_warmup: 100
         min_samples: 200
         max_tree_depth: 10

----

run\_nuts\_sampling
-------------------

.. autofunction:: homodyne.optimization.cmc.sampler.run_nuts_sampling

----

MCMCSamples
-----------

Return type for per-shard sampling results.

.. autoclass:: homodyne.optimization.cmc.sampler.MCMCSamples
   :members:
   :undoc-members:
   :show-inheritance:

----

NUTS Configuration
------------------

Key NumPyro NUTS parameters accessible via ``CMCConfig.per_shard_mcmc``:

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``target_accept_prob``
     - 0.8
     - Target HMC acceptance probability
   * - ``max_tree_depth``
     - 10
     - Max NUTS tree depth (2\ :sup:`10` = 1024 leapfrog steps)
   * - ``chain_method``
     - ``"parallel"``
     - ``parallel`` / ``vectorized`` / ``sequential``
   * - ``num_chains``
     - 4
     - Number of parallel MCMC chains

.. warning::

   For the multiprocessing backend, ``chain_method: "parallel"`` is the only
   recommended setting. ``vectorized`` causes workers to drop to 1–2 CPUs,
   resulting in an empirically observed 20× slowdown (4.9 s vs 101 s wall time
   for identical workloads).

----

JAX Profiling
-------------

Enable XLA-level profiling to diagnose JIT compilation and execution bottlenecks.
Note that ``py-spy`` only profiles Python code; XLA runs native code invisible
to it. JAX profiling provides XLA-level insights.

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         enable_jax_profiling: true
         jax_profile_dir: ./profiles/jax

View results with TensorBoard:

.. code-block:: bash

   tensorboard --logdir=./profiles/jax

----

Usage Examples
--------------

Checking adaptive scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc.sampler import SamplingPlan
   from homodyne.optimization.cmc.config import CMCConfig

   config = CMCConfig()   # default settings

   # Small shard (1000 pts, 9 params)
   plan = SamplingPlan.from_config(config, shard_size=1000, n_params=9)
   print(f"Warmup:  {plan.n_warmup}")
   print(f"Samples: {plan.n_samples}")
   print(f"Adapted: {plan.was_adapted}")

   # Full-size shard (50K pts)
   plan_full = SamplingPlan.from_config(config, shard_size=50_000, n_params=9)
   print(f"Adapted: {plan_full.was_adapted}")   # False

Inspecting divergence constants
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc.sampler import (
       DIVERGENCE_RATE_TARGET,
       DIVERGENCE_RATE_HIGH,
       DIVERGENCE_RATE_CRITICAL,
   )

   div_rate = 0.08   # example shard divergence rate

   if div_rate < DIVERGENCE_RATE_TARGET:
       print("Excellent sampling quality")
   elif div_rate < DIVERGENCE_RATE_HIGH:
       print("Acceptable — monitor carefully")
   else:
       print("High divergence — shard may be filtered")

----

.. automodule:: homodyne.optimization.cmc.sampler
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SamplingPlan, MCMCSamples, run_nuts_sampling
