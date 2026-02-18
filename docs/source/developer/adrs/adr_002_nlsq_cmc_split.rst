.. _adr_nlsq_primary:

ADR-002: NLSQ / CMC Architectural Split
=========================================

:Status: Accepted
:Date: 2024
:Deciders: Core team

Context
-------

XPCS data analysis requires both **point estimates** (fast, for online feedback at the
beamline) and **uncertainty quantification** (slower, for publication-quality results).
Two fundamentally different algorithmic approaches are needed:

1. **Non-linear least squares (NLSQ)**: Produces a point estimate
   :math:`\hat{\theta}_\mathrm{MLE}` with a linearized covariance matrix in seconds.

2. **Markov Chain Monte Carlo (MCMC)**: Produces full posterior samples
   :math:`\{\theta^{(k)}\}` with honest uncertainty estimates, but at much higher
   computational cost.

The question is: how should these two methods be integrated architecturally?


Decision
--------

Homodyne implements a **two-stage pipeline** where NLSQ is the **primary** optimizer
and CMC is the **secondary** Bayesian sampler:

.. code-block:: python

   # Stage 1: Fast point estimate (seconds to minutes)
   nlsq_result = fit_nlsq_jax(data, config)

   # Stage 2: Full posterior (minutes to hours), initialized from Stage 1
   cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)

The two stages share the same physics model (via ``HomodyneModel``) but use
**separate physics backends**:

- NLSQ uses ``jax_backend.py`` (**meshgrid mode**): evaluates :math:`c_2(t_i, t_j)` for
  all pairs simultaneously using 2D JAX broadcasting.
- CMC uses ``physics_cmc.py`` (**element-wise mode**): evaluates :math:`c_2` at specific
  :math:`(t_1, t_2)` pairs given as flat shard vectors.

This split is intentional and maintained as a first-class architectural boundary.


Rationale
---------

**1. NLSQ as warm-start for CMC is essential**

Without NLSQ initialization, CMC divergence rates during warmup are ~28% for the
``laminar_flow`` mode with 7+ parameters. With NLSQ warm-start, divergences drop
to <5%. This is because:

- The posterior of the laminar-flow model has a narrow ridge in the
  :math:`(D_0, \dot{\gamma}_0)` plane (near-degeneracy).
- NUTS warmup requires many leapfrog steps to find this ridge from a generic prior.
- Starting near the ridge (from NLSQ) allows NUTS warmup to tune the mass matrix
  efficiently.

**2. Different computational patterns require different backends**

The NLSQ Jacobian computation requires the full :math:`(t_1, t_2)` grid because the
optimizer needs :math:`\partial r_{ij} / \partial\theta` for all residuals simultaneously
to form the normal equations.

NUTS requires evaluating the log-likelihood gradient at a single point :math:`\theta`
over the shard data. The "shard" consists of randomly selected :math:`(t_1, t_2)` pairs,
not a structured grid. Using the meshgrid backend for this would waste memory on
unneeded grid entries.

Attempting to use a single backend for both cases would either waste memory (meshgrid for
NUTS) or lose vectorization efficiency (element-wise for NLSQ Jacobian).

**3. NLSQ provides fast feedback**

At synchrotron beamlines, experimenters need to know within seconds whether the data
quality is sufficient and whether the model fits. NLSQ convergence in 1–30 seconds
meets this requirement. CMC is run later (during data reduction), not in real time.

**4. Covariance matrix from NLSQ initializes CMC priors**

The NLSQ covariance matrix :math:`\Sigma_\mathrm{NLSQ} = (J^\top W J)^{-1}` is used to
construct tight Gaussian priors for the CMC model. This prior is much more informative
than the default broad priors, further improving NUTS efficiency.


Consequences
------------

**Positive**:

- Fast feedback (NLSQ alone) is always available, even if CMC is skipped.
- CMC has dramatically lower divergence rates due to NLSQ warm-start.
- Clean separation: NLSQ code does not depend on CMC internals, and vice versa.
- Each backend is independently testable.

**Negative / Accepted trade-offs**:

- Two physics backends (``jax_backend.py`` and ``physics_cmc.py``) must be kept
  synchronized when the physics model changes.
- Adding a new analysis mode requires updating both backends.
- The split is non-obvious to new contributors; requires documentation (this ADR).

**Mitigation**: Shared utilities in ``physics_utils.py`` (``safe_exp``, ``safe_sinc``,
``calculate_diffusion_coefficient``) prevent divergence of the core physics logic between
the two backends.


Alternatives Considered
-----------------------

**A. MCMC only (no NLSQ)**

Simpler API. Rejected because: no fast feedback path; CMC without warm-start has 28%+
divergence rates for laminar_flow; users cannot run analysis at the beamline in real time.

**B. Single physics backend for both**

Would simplify maintenance. Rejected because: the meshgrid and element-wise calling
conventions cannot be efficiently unified without JAX control flow that is not JIT-safe
for one of the two use cases.

**C. Variational inference (VI) instead of MCMC**

VI (e.g., ADVI in NumPyro) produces approximate posteriors much faster than NUTS.
Rejected because: VI produces overconfident posteriors for the correlated parameters
in the laminar-flow model (known pathology of mean-field VI); full NUTS is required
for accurate uncertainty quantification.

**D. Ensemble sampler (emcee-style)**

Affine-invariant ensemble samplers handle correlated posteriors without tuning.
Rejected because: ensemble samplers are not trivially parallelizable across data shards,
and the CMC consensus framework is not directly applicable to ensemble states.


.. seealso::

   - :ref:`developer_architecture` — full data flow diagram
   - :ref:`adr_cmc_consensus` — why Consensus Monte Carlo
   - :ref:`theory_computational_methods` — mathematical details of both methods
