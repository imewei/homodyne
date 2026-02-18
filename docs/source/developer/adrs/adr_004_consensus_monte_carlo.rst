.. _adr_cmc_consensus:

ADR-004: Consensus Monte Carlo for Bayesian Inference
======================================================

:Status: Accepted
:Date: 2024–2025
:Deciders: Core team

Context
-------

XPCS datasets from modern synchrotron sources can exceed :math:`10^8` data points
(the full :math:`c_2(t_1, t_2)` matrix at high temporal resolution). Standard MCMC methods
— including NUTS — scale as :math:`O(n)` per leapfrog step, where :math:`n` is the number
of data points. For :math:`10^8` points, a single NUTS step with 100 leapfrog substeps
would require evaluating :math:`10^{10}` model instances, which is computationally
intractable even with JIT compilation.

The question is: how to make full Bayesian inference tractable on large XPCS datasets while
preserving statistical validity?


Decision
--------

Homodyne implements **Consensus Monte Carlo** (CMC, Scott et al. 2016) as its Bayesian
backend. The algorithm is:

1. **Shard**: Partition the :math:`n` data points into :math:`M` shards of size
   :math:`n_s \approx n/M`.
2. **Parallel NUTS**: Run independent NUTS chains on each shard (in separate worker
   processes), obtaining :math:`K` posterior samples :math:`\{\theta^{(k)}_s\}` from the
   shard-specific posterior :math:`p_s(\theta | \mathcal{D}_s)`.
3. **Consensus**: Combine the :math:`M` sets of shard samples into a single approximation
   of the global posterior :math:`p(\theta | \mathcal{D})`.

The multiprocessing backend spawns :math:`N_\mathrm{workers} = \lfloor N_\mathrm{cores}/2\rfloor - 1`
worker processes, each with 4 virtual JAX devices (via
``--xla_force_host_platform_device_count=4``). Each worker runs NUTS in ``parallel`` mode
(pmap over 4 devices), achieving near-full CPU utilization.


Shard Size Selection
--------------------

The shard size :math:`n_s` controls the trade-off between:

- **Statistical accuracy**: Larger shards → shard posterior closer to global posterior.
- **Computational cost**: NUTS is :math:`O(n_s)` per step.

Homodyne uses ``max_points_per_shard: "auto"`` by default, which selects :math:`n_s` based
on dataset size:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dataset size
     - laminar_flow auto
     - static auto
   * - < 2M points
     - 3–5K per shard
     - 5–6K per shard
   * - 2–5M points
     - 3–5K per shard
     - 5–6K per shard
   * - 5–50M points
     - 5K per shard
     - 6–10K per shard
   * - 50–100M points
     - 5–8K per shard
     - 9–15K per shard
   * - 100M+ points
     - 8–10K per shard
     - 12–20K per shard

The upper bound is set by NUTS scalability: shard sizes above ~100K points per shard
incur unacceptably long NUTS steps and are never used.


Chain Execution Method
----------------------

Each worker process runs 4 NUTS chains using NumPyro's ``parallel`` execution method
(``pmap`` over 4 virtual JAX devices):

.. code-block:: python

   kernel = NUTS(model, max_tree_depth=10)
   mcmc = MCMC(kernel, num_warmup=plan.n_warmup, num_samples=plan.n_samples)
   mcmc.run(rng_key, ..., extra_fields=("energy",))

The ``parallel`` method is empirically 20x faster than ``vectorized`` (vmap) for the
multiprocessing backend (4.9 s vs 101 s wall time for identical workloads). The reason:
``pmap`` distributes chains across 4 virtual CPU "devices" (NUMA-aware XLA partitions),
while ``vmap`` batches chains on a single device sequentially.


Worker Environment
------------------

Before spawning workers, the backend:

1. Saves the current environment (``OMP_PROC_BIND``, ``OMP_PLACES``).
2. Clears ``OMP_PROC_BIND`` and ``OMP_PLACES`` to prevent OpenMP thread binding
   conflicts between workers.
3. Sets ``OMP_NUM_THREADS=1`` or 2 per worker to prevent thread oversubscription
   (each worker manages its own JAX device count via XLA_FLAGS).
4. Restores the parent environment after all workers are spawned.


Rationale
---------

**1. CMC is asymptotically exact**

The consensus combination is exact when:

(a) Measurements are independent (true for XPCS: each :math:`(t_1, t_2, \phi)` triplet
    is an independent measurement conditioned on the parameters).
(b) The shards are drawn i.i.d. from the full dataset.
(c) The prior is the same in all shard models.

Under these conditions, the consensus product of shard posteriors equals the global
posterior (up to normalization). See Scott et al. 2016.

**2. CMC enables linear scalability**

With :math:`M` shards and :math:`P` parallel workers:

- Total NUTS cost: :math:`O(n_s \cdot L)` per worker per chain, not :math:`O(n \cdot L)`.
- Consensus step: :math:`O(M \cdot K)` — negligible.
- Wall time scales as :math:`O(n / (P \cdot n_s))` — linear in :math:`1/P`.

In practice, wall time is dominated by the NUTS warmup, which is :math:`O(n_s)` per shard.

**3. NLSQ warm-start dramatically improves CMC quality**

Without NLSQ initialization, shard-level NUTS chains have high divergence rates (~28%)
because the default broad priors place chains far from the posterior mode. The NLSQ
covariance matrix provides tight, data-informed priors that:

- Reduce divergences to <5%.
- Allow NUTS warmup to complete in fewer steps.
- Prevent chains from exploring unphysical parameter regions.

**4. Quality filtering prevents posterior corruption**

Shards with divergence rate > 10% are excluded from the consensus. This is a conservative
threshold that discards shards where NUTS clearly failed (bad geometry, wrong step size)
while retaining the majority of shards with acceptable mixing.


Consequences
------------

**Positive**:

- Scales to arbitrarily large datasets by increasing the number of shards.
- Full posterior uncertainty quantification, not just linearized NLSQ errors.
- ArviZ diagnostics (:math:`\hat{R}`, ESS, BFMI, divergence fraction) provide
  quantitative quality assessment.

**Negative / Accepted trade-offs**:

- CMC produces an approximation of the global posterior, not the exact posterior.
  The approximation quality depends on shard size (larger shards → better approximation).
- CMC requires the multiprocessing spawn of worker processes; startup overhead is
  ~1–2 seconds per worker.
- Auto shard-size selection may be suboptimal for unusual dataset characteristics;
  users can override with ``max_points_per_shard: <integer>``.


Alternatives Considered
-----------------------

**A. Standard NUTS on full dataset**

Exact posterior. Rejected because: :math:`O(n)` per step is intractable for
:math:`n > 10^6`.

**B. Minibatch MCMC (stochastic gradient MCMC)**

Scales better than standard NUTS. Rejected because: stochastic gradient MCMC has
known bias and is difficult to diagnose; the convergence guarantees are weaker.

**C. Variational inference (ADVI)**

Fast (:math:`O(\text{epochs} \cdot n_\mathrm{minibatch})`). Rejected because:
mean-field ADVI systematically underestimates posterior variance for correlated
parameters — a known failure mode for the :math:`(D_0, \dot{\gamma}_0)` correlation
in the laminar-flow model.

**D. Sequential Monte Carlo (SMC)**

SMC is asymptotically exact and handles multimodal posteriors well. Rejected for the
initial version because: SMC requires many sequential passes over the data, which is
harder to parallelize across shards than independent NUTS chains. Reconsidered for
future work.


.. seealso::

   - Scott et al. 2016 — original CMC paper (:ref:`theory_citations`)
   - :ref:`theory_computational_methods` — NUTS and CMC algorithm details
   - :mod:`homodyne.optimization.cmc.backends.multiprocessing` — implementation
   - :mod:`homodyne.optimization.cmc.sampler` — SamplingPlan with adaptive scaling
