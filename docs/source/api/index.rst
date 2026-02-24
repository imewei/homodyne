.. _api-reference:

=============
API Reference
=============

This section provides the complete API reference for the Homodyne package.
All documentation is auto-generated from docstrings in the source code using
Sphinx autodoc. Module and class signatures reflect the installed package version.

.. note::

   The recommended entry points for most users are :func:`homodyne.optimization.nlsq.core.fit_nlsq_jax`
   for non-linear least squares fitting and :func:`homodyne.optimization.cmc.core.fit_mcmc_jax`
   for Bayesian uncertainty quantification. Start with the :doc:`nlsq_adapter` page
   for the high-level adapter interface.

----

Module Overview
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`core`
     - Physical constants, parameter bounds, validation, and shared utilities
   * - :doc:`homodyne_model`
     - ``HomodyneModel`` — unified model facade for NLSQ and CMC
   * - :doc:`jax_backend`
     - JIT-compiled g₁/g₂ physics, autodiff, meshgrid cache
   * - :doc:`models`
     - ``DiffusionModel``, ``ShearModel``, ``CombinedModel`` — OO model interface
   * - :doc:`theory_engine`
     - ``TheoryEngine`` — high-level g1/g2 computation interface
   * - :doc:`optimization`
     - Optimization overview: NLSQ vs CMC, common workflow patterns
   * - :doc:`nlsq_adapter`
     - ``NLSQAdapter`` — recommended adapter for standard optimizations
   * - :doc:`nlsq_wrapper`
     - ``NLSQWrapper`` — full-featured wrapper for advanced use cases
   * - :doc:`cmc`
     - CMC overview, ``fit_mcmc_jax()``, ``CMCConfig``, shard size guide
   * - :doc:`cmc_sampler`
     - ``SamplingPlan``, NUTS execution, adaptive sampling
   * - :doc:`cmc_reparameterization`
     - Parameter reparameterization for MCMC sampling efficiency
   * - :doc:`cmc_backends`
     - Multiprocessing backend, worker configuration, chain execution methods
   * - :doc:`data`
     - ``XPCSDataLoader`` — HDF5 data ingestion and validation
   * - :doc:`config`
     - ``ConfigManager``, ``ParameterSpace``, YAML schema
   * - :doc:`cli`
     - All CLI entry points and shell completion system
   * - :doc:`device`
     - CPU device management, NUMA configuration, and status utilities
   * - :doc:`io`
     - Result saving (JSON, NPZ) and serialization utilities
   * - :doc:`viz`
     - Plotting, visualization, and diagnostic figure generation
   * - :doc:`utils`
     - Logging utilities, CPU/NUMA detection

----

Quick Navigation
----------------

**Core Physics**

- :doc:`core` — Physics constants, validation, ``safe_exp()``, ``safe_sinc()``, scaling utilities
- :doc:`homodyne_model` — ``HomodyneModel`` unified model facade
- :doc:`jax_backend` — JIT-compiled g₁/g₂ functions, autodiff, cache management

**Models**

- :ref:`api-models-base` — ``PhysicsModelBase`` abstract interface
- :ref:`api-diffusion-model` — Static-mode diffusion model
- :ref:`api-shear-model` — Shear-flow contribution
- :ref:`api-combined-model` — Full laminar-flow model

**Optimization**

- :ref:`api-fit-nlsq` — ``fit_nlsq_jax()`` primary entry point
- :ref:`api-nlsq-adapter` — ``NLSQAdapter`` (JIT caching, multi-start)
- :ref:`api-nlsq-wrapper` — ``NLSQWrapper`` (streaming, anti-degeneracy)
- :ref:`api-fit-mcmc` — ``fit_mcmc_jax()`` Bayesian pipeline
- :ref:`api-cmc-config` — ``CMCConfig`` dataclass
- :ref:`api-sampling-plan` — ``SamplingPlan`` adaptive scheduling
- :ref:`api-reparam-config` — ``ReparamConfig`` and parameter transforms

**Infrastructure**

- :ref:`api-xpcs-loader` — ``XPCSDataLoader``
- :ref:`api-config-manager` — ``ConfigManager``
- :doc:`cli` — ``homodyne`` main command, all CLI entry points
- :doc:`utils` — ``get_logger()``, ``log_phase()``, path validation

----

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   core
   homodyne_model
   jax_backend
   models
   theory_engine
   optimization
   optimization_guide
   nlsq_adapter
   nlsq_wrapper
   cmc
   cmc_sampler
   cmc_reparameterization
   cmc_backends
   data
   config
   cli
   device
   io
   viz
   utils
