.. _adrs:

Architecture Decision Records
==============================

Architecture Decision Records (ADRs) document the significant technical
decisions made during Homodyne's development — what was decided, why, and
what alternatives were considered.

.. toctree::
   :maxdepth: 1

   adr_001_jax_cpu_only
   adr_002_nlsq_cmc_split
   adr_003_anti_degeneracy
   adr_004_consensus_monte_carlo
   adr_005_per_angle_scaling

----

**ADR Summary:**

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - ADR
     - Title
     - Status
   * - 001
     - JAX CPU-only backend
     - Accepted
   * - 002
     - NLSQ / CMC architectural split
     - Accepted
   * - 003
     - Anti-degeneracy layer for laminar flow
     - Accepted
   * - 004
     - Consensus Monte Carlo over standard MCMC
     - Accepted
   * - 005
     - Per-angle scaling modes (auto/constant/individual/fourier)
     - Accepted

.. note::

   New ADRs should follow the template in ``docs/source/developer/adrs/``.
   Each ADR covers: Context, Decision, Consequences, and Alternatives Considered.
