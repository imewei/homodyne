.. _developer_architecture:

System Architecture
===================

This page describes the module structure of homodyne, the data flow from user input to
analysis result, and the key design decisions that shape the codebase. Understanding the
architecture is essential for contributing to any layer of the stack.



Component Overview
------------------

Homodyne is organized into ten top-level packages, each with a single well-defined
responsibility:

.. code-block:: text

   homodyne/
   ├── core/           # Physics kernel -- JIT-compiled JAX functions
   ├── optimization/
   │   ├── nlsq/       # Primary optimizer: Trust-region Levenberg-Marquardt
   │   │               #   wrapper.py orchestrator, fallback_chain.py, recovery.py
   │   │               #   strategies/: stratified_ls, hybrid_streaming, out_of_core
   │   └── cmc/        # Secondary: Consensus Monte Carlo (NumPyro NUTS)
   ├── data/           # HDF5 loading and I/O boundary validation
   ├── config/         # YAML configuration: parsing, validation, defaults
   ├── cli/            # Entry points: commands.py orchestrator delegates to
   │                   #   config_handling, data_pipeline, optimization_runner,
   │                   #   result_saving, plot_dispatch
   ├── device/         # CPU/NUMA topology detection
   ├── io/             # Result serialization (JSON + NPZ writers)
   ├── viz/            # Visualization: mcmc_plots.py re-exports from
   │                   #   mcmc_diagnostics, mcmc_comparison, mcmc_dashboard,
   │                   #   mcmc_arviz, mcmc_report
   ├── runtime/        # Shell completion, system validation, post-install
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
   ├── numpy_gradients.py    # NumPy-based gradient/Hessian computation
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
   ├── wrapper.py            # NLSQWrapper: thin orchestrator
   │                         #   Delegates to fallback_chain, recovery, strategies/
   ├── fallback_chain.py     # OptimizationStrategy enum + fallback logic
   │                         #   execute_optimization_with_fallback()
   │                         #   handle_nlsq_result(), get_fallback_strategy()
   ├── recovery.py           # 3-attempt error recovery
   │                         #   execute_with_recovery(), diagnose_error()
   │                         #   safe_uncertainties_from_pcov()
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
   ├── results.py            # OptimizationResult dataclass
   ├── result_builder.py     # Result construction and quality flags
   ├── multistart.py         # Multi-start optimization
   ├── jacobian.py           # Jacobian computation utilities
   ├── parallel_accumulator.py  # Parallel residual accumulation
   ├── parameter_utils.py    # Parameter transformation utilities
   ├── transforms.py         # Parameter bound transforms
   ├── progress.py           # Progress reporting
   ├── strategies/           # Fitting strategy implementations
   │   ├── stratified_ls.py  #   Primary: stratified least-squares (JTJ via NLSQ)
   │   ├── hybrid_streaming.py  # Hybrid streaming for large datasets
   │   ├── out_of_core.py    #   Out-of-core JTJ accumulation
   │   ├── executors.py      #   Strategy selection and dispatch
   │   ├── residual.py       #   Residual function (NumPy path)
   │   ├── residual_jit.py   #   Residual function (JIT path)
   │   ├── sequential.py     #   Sequential per-angle optimization
   │   └── chunking.py       #   Out-of-core data chunking
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
   │   ├── base.py             # Backend abstract base class
   │   ├── multiprocessing.py  # Primary: multiprocessing backend
   │   │                       #   Spawns N workers (physical_cores/2 - 1)
   │   │                       #   Each worker has 4 virtual JAX devices
   │   ├── worker_pool.py      # Persistent WorkerPool + SharedDataManager
   │   │                       #   Reuses workers across shards
   │   │                       #   Shared memory for shard data arrays
   │   ├── pbs.py              # PBS cluster backend
   │   └── pjit.py             # pjit-based single-process backend
   ├── config.py             # CMCConfig dataclass, CMCConfig.from_dict()
   ├── data_prep.py          # Shard construction and data preparation
   ├── diagnostics.py        # ArviZ diagnostics: R-hat, ESS, BFMI
   ├── io.py                 # Result serialization
   ├── priors.py             # Prior distribution constructors
   ├── reparameterization.py # Log-space parameter reparameterization
   │                         #   compute_t_ref(), transform_nlsq_to_reparam_space()
   ├── results.py            # CMCResult dataclass
   └── scaling.py            # Per-angle scaling for CMC


CLI Module Map
--------------

.. code-block:: text

   homodyne/cli/
   ├── main.py               # Entry point: main()
   ├── args_parser.py        # Argument parser and validation
   ├── commands.py           # dispatch_command() orchestrator + re-exports
   ├── config_handling.py    # Device config, YAML loading, CLI overrides
   │                         #   _configure_device(), _load_configuration()
   │                         #   _apply_cli_overrides(), _build_mcmc_runtime_kwargs()
   ├── data_pipeline.py      # Data loading and preprocessing
   │                         #   _load_data(), _exclude_t0_from_analysis()
   │                         #   _apply_angle_filtering_for_optimization()
   │                         #   _prepare_cmc_config(), _pool_mcmc_data()
   ├── optimization_runner.py  # NLSQ/CMC execution and warm-start
   │                         #   _run_nlsq_optimization(), _run_optimization()
   │                         #   _resolve_nlsq_warmstart()
   ├── result_saving.py      # JSON/NPZ result serialization
   │                         #   save_nlsq_results(), save_mcmc_results()
   │                         #   _extract_nlsq_metadata(), _prepare_parameter_data()
   ├── plot_dispatch.py      # Plotting dispatch
   │                         #   _handle_plotting(), generate_nlsq_plots()
   ├── config_generator.py   # homodyne-config entry point
   └── xla_config.py         # homodyne-config-xla entry point

``commands.py`` acts as a re-export hub: all internal functions are importable
from ``homodyne.cli.commands`` for backward compatibility, even though the
implementations live in the submodules above.


Visualization Module Map
------------------------

.. code-block:: text

   homodyne/viz/
   ├── mcmc_plots.py         # Re-export hub for all MCMC visualization
   ├── mcmc_diagnostics.py   # plot_trace_plots(), plot_kl_divergence_matrix()
   │                         #   plot_convergence_diagnostics()
   ├── mcmc_comparison.py    # plot_posterior_comparison()
   ├── mcmc_dashboard.py     # plot_cmc_summary_dashboard()
   ├── mcmc_arviz.py         # ArviZ wrappers: plot_arviz_trace/posterior/pair
   ├── mcmc_report.py        # generate_mcmc_diagnostic_report(), print_mcmc_summary()
   ├── nlsq_plots.py         # NLSQ result visualization
   ├── experimental_plots.py # Experimental data C2 heatmaps
   ├── datashader_backend.py # High-performance rendering (optional)
   └── diagnostics.py        # Quantitative viz diagnostics


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
