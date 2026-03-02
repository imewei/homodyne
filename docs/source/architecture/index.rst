.. _architecture-deep-dives:

Architecture Deep Dives
=======================

Detailed internal architecture documentation for each major subsystem.
These documents describe the design, data flow, and implementation details
at a level useful for contributors, maintainers, and advanced users.

For a concise module-level overview, see :doc:`../developer/architecture`.
For formal design rationale, see :doc:`../developer/adrs/index`.

.. toctree::
   :maxdepth: 2

   homodyne-architecture-overview
   physical-model-architecture
   nlsq-fitting-architecture
   cmc-fitting-architecture
   data-handler-architecture

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - :doc:`homodyne-architecture-overview`
     - System-level architecture: package layout, data flow, module dependencies, and cross-cutting concerns.
   * - :doc:`physical-model-architecture`
     - Physics kernel: g1/g2 theory, JIT-compiled backends, model variants, and gradient-safe numerics.
   * - :doc:`nlsq-fitting-architecture`
     - NLSQ optimization: trust-region LM, anti-degeneracy layers, CMA-ES, memory routing, and streaming.
   * - :doc:`cmc-fitting-architecture`
     - Consensus Monte Carlo: sharding, multiprocessing workers, NUTS execution, and posterior combination.
   * - :doc:`data-handler-architecture`
     - Data pipeline: HDF5 loading, configuration management, validation, and result serialization.
