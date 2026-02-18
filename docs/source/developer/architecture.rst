.. _developer_architecture:

System Architecture
===================

This page describes the module structure of homodyne, the data flow from user input to
analysis result, and the key design decisions that shape the codebase. Understanding the
architecture is essential for contributing to any layer of the stack.

.. contents:: Contents
   :local:
   :depth: 2


Component Overview
------------------

Homodyne is organized into seven top-level packages, each with a single well-defined
responsibility:

.. code-block:: text

   homodyne/
   ├── core/           # Physics kernel — JIT-compiled JAX functions
   ├── optimization/
   │   ├── nlsq/       # Primary optimizer: Trust-region Levenberg-Marquardt
   │   └── cmc/        # Secondary: Consensus Monte Carlo (NumPyro NUTS)
   ├── data/           # HDF5 loading and I/O boundary validation
   ├── config/         # YAML configuration: parsing, validation, defaults
   ├── cli/            # Entry points and shell completion
   ├── device/         # CPU/NUMA topology detection
   └── utils/          # Logging, exceptions, shared utilities

The key architectural constraint is that **numerical logic never depends on Python-level
conditionals over runtime array values inside JIT-compiled functions**. All branching that
depends on data must happen before the JIT boundary.


Data Flow
----------

The canonical data flow for a full NLSQ + CMC analysis is:

.. code-block:: text

   YAML config file
        │
        ▼
   ConfigManager (config/)
   ├── Validates parameter bounds and types
   ├── Populates NLSQConfig, CMCConfig
   └── Returns typed config object
        │
        ▼
   XPCSDataLoader (data/)
   ├── Opens HDF5 file
   ├── Validates shape, dtype, monotonicity
   ├── Returns (t1_grid, t2_grid, c2_matrix, q_values, phi_angles, h_gap)
   └── All arrays at JAX dtype (float64)
        │
        ▼
   HomodyneModel (core/)
   ├── Selects analysis mode: "static" or "laminar_flow"
   ├── Wraps JAX backend (jax_backend.py for NLSQ, physics_cmc.py for CMC)
   └── Exposes model.residuals(theta) for optimizer
        │
        ├──► NLSQAdapter (optimization/nlsq/)
        │    ├── Anti-degeneracy initialization (per-angle mode)
        │    ├── Trust-region LM iterations
        │    ├── Optional CMA-ES global pre-optimizer
        │    └── Returns NLSQResult (params, covariance, chi2)
        │
        └──► CMCBackend (optimization/cmc/)
             ├── Optional NLSQ warm-start priors
             ├── Shard the data (max_points_per_shard)
             ├── Spawn worker processes (multiprocessing)
             ├── Each worker runs NumPyro NUTS on one shard
             ├── Consensus combination of shard posteriors
             └── Returns CMCResult (posterior samples, diagnostics)
                  │
                  ▼
             Result (JSON + NPZ)
             ├── Parameter estimates + uncertainties (NLSQ)
             ├── Posterior samples (CMC)
             ├── ArviZ diagnostics (R-hat, ESS, BFMI)
             └── Convergence metadata


Core Module Map
---------------

.. code-block:: text

   homodyne/core/
   ├── jax_backend.py        # NLSQ path: meshgrid-mode c2 computation
   │                         #   compute_g2_static(), compute_g2_laminar_flow()
   │                         #   JIT-compiled, vmap over angles
   ├── physics_cmc.py        # CMC path: element-wise (shard-vector) c2
   │                         #   Different calling convention from jax_backend
   ├── physics_utils.py      # Shared utilities (both paths use these)
   │                         #   safe_exp(), safe_sinc(), calculate_diffusion_coefficient()
   │                         #   EPS constant, PI constant
   ├── physics.py            # Physical constants, parameter bounds, ValidationResult
   │                         #   PhysicsConstants, ParameterBounds, validate_params()
   ├── models.py             # Object-oriented model interface
   │                         #   DiffusionModel, ShearModel, CombinedModel
   │                         #   create_model() factory function
   ├── theory.py             # High-level TheoryEngine class
   │                         #   compute_g1(), compute_g2() user-facing wrappers
   ├── homodyne_model.py     # Unified model facade for both NLSQ and CMC
   ├── fitting.py            # Fitting utilities and weight computation
   ├── scaling_utils.py      # Per-angle contrast/offset estimation
   │                         #   quantile_scaling_estimate()
   ├── physics_factors.py    # Factored physics computations
   ├── physics_nlsq.py       # NLSQ-specific physics helpers
   ├── backend_api.py        # Abstract backend API (NLSQ/CMC interface)
   ├── diagonal_correction.py # Diagonal correction for c2 matrices
   └── model_mixins.py       # Mixin classes for model composition

**Critical distinction**: NLSQ and CMC use different physics implementations:

- ``jax_backend.py`` uses **meshgrid mode**: evaluates :math:`c_2(t_i, t_j)` for all
  :math:`(i,j)` pairs simultaneously using JAX broadcasting over 2D arrays.
- ``physics_cmc.py`` uses **element-wise mode**: evaluates :math:`c_2` at specific
  :math:`(t_1, t_2)` pairs given as flat vectors (one entry per shard data point).

The element-wise mode is required by NUTS because the log-likelihood gradient must be
computed with respect to parameters at specific data points, not the full grid.


NLSQ Module Map
---------------

.. code-block:: text

   homodyne/optimization/nlsq/
   ├── adapter.py            # NLSQAdapter: recommended entry point
   │                         #   fit_nlsq_jax() delegates here
   │                         #   Auto-selects strategy, wraps NLSQWrapper
   ├── wrapper.py            # NLSQWrapper: full-featured interface
   │                         #   Direct access to all LM options
   ├── anti_degeneracy_controller.py  # 5-layer anti-degeneracy orchestrator
   ├── fourier_reparam.py    # Layer 1: Fourier/constant reparameterization
   ├── hierarchical.py       # Layer 2: Hierarchical optimization stages
   ├── adaptive_regularization.py  # Layer 3: CV-based regularization
   ├── gradient_monitor.py   # Layer 4: Gradient collapse detection
   ├── shear_weighting.py    # Layer 5: Shear-sensitivity data weighting
   ├── cmaes_wrapper.py      # CMA-ES global optimizer (optional)
   ├── config.py             # NLSQConfig dataclass
   ├── core.py               # Core NLSQ computation loop
   ├── data_prep.py          # Data preprocessing for NLSQ
   ├── memory.py             # Memory threshold / streaming selection
   ├── results.py            # NLSQResult dataclass
   ├── strategies/           # Memory strategy implementations
   └── validation/           # Input validation for NLSQ


CMC Module Map
--------------

.. code-block:: text

   homodyne/optimization/cmc/
   ├── core.py               # Main CMC entry point
   │                         #   fit_mcmc_jax() delegates here
   │                         #   Orchestrates sharding, workers, consensus
   ├── sampler.py            # SamplingPlan, NUTS execution per shard
   │                         #   SamplingPlan.from_config() with adaptive scaling
   ├── model.py              # NumPyro model definition
   │                         #   5 model variants, get_xpcs_model() factory
   ├── backends/
   │   └── multiprocessing.py  # Multiprocessing backend
   │                            #   Spawns N workers (physical_cores/2 - 1)
   │                            #   Each worker has 4 virtual JAX devices
   ├── config.py             # CMCConfig dataclass, CMCConfig.from_dict()
   ├── data_prep.py          # Shard construction and data preparation
   ├── diagnostics.py        # ArviZ diagnostics: R-hat, ESS, BFMI
   ├── io.py                 # Result serialization
   ├── priors.py             # Prior distribution constructors
   ├── reparameterization.py # Log-space parameter reparameterization
   │                         #   compute_t_ref(), reparameterize_nlsq()
   ├── results.py            # CMCResult dataclass
   └── scaling.py            # Per-angle scaling for CMC


Key Design Decisions
--------------------

**1. JAX-first numerical core**

All numerical computations use JAX arrays and JIT-compiled functions. NumPy appears
only at I/O boundaries (loading HDF5 data, writing JSON results). This ensures:

- Reproducible computation via explicit PRNG keys.
- Automatic differentiation for Jacobians (NLSQ) and gradients (NUTS).
- Platform-independent performance via XLA compilation.

See :ref:`adr_jax_cpu_only`.

**2. NLSQ as the primary optimizer**

NLSQ runs first and provides warm-start parameters for CMC. This is not just a
performance optimization: NLSQ initialization dramatically reduces CMC divergences (from
~28% to <5%) by placing chains near the posterior mode. See :ref:`adr_nlsq_primary`.

**3. Two physics backends**

The meshgrid (NLSQ) and element-wise (CMC) backends compute the same physics but in
structurally different ways. Maintaining two implementations avoids JAX control flow issues
that would arise from trying to use a single implementation for both callers. Shared
utilities in ``physics_utils.py`` prevent code duplication for the common sub-computations.

**4. No global mutable state**

All configuration flows through typed dataclass objects. No module-level globals are
modified at runtime. This makes tests deterministic and enables safe multiprocessing (each
worker spawns with a clean module state).

**5. Narrow exception handling**

Specific exception types (``OSError``, ``ValueError``, ``KeyError``) are caught at function
boundaries. Broad ``except Exception`` is used only at top-level dispatchers in the CLI,
where it is always accompanied by ``log_exception()`` for traceability.


Module Dependency Map
---------------------

The dependency graph (edges represent "imports from"):

.. code-block:: text

   cli/ ──────────────────────────────► config/
     │                                   │
     ▼                                   ▼
   optimization/nlsq/ ◄────────────── core/
   optimization/cmc/  ◄────────────── core/
     │                                   ▲
     ▼                                   │
   data/ ──────────────────────────────► utils/
   device/ ─────────────────────────────► utils/

Rules enforced:

- ``core/`` has **no imports from** ``optimization/``, ``data/``, ``config/``, or ``cli/``.
- ``optimization/`` imports from ``core/`` and ``utils/``, but not from ``data/`` or ``cli/``.
- ``data/`` imports from ``utils/`` only.
- ``cli/`` imports from all other layers.

This guarantees that the physics kernel can be tested and used without any dependency on
the I/O or configuration machinery.


.. seealso::

   - :ref:`developer_contributing` — setting up a dev environment
   - :ref:`developer_testing` — test strategy
   - :ref:`adr_jax_cpu_only` — CPU-only decision
   - :ref:`adr_nlsq_primary` — NLSQ-first strategy
   - :ref:`adr_cmc_consensus` — Consensus Monte Carlo design
   - :ref:`adr_anti_degeneracy` — anti-degeneracy system
