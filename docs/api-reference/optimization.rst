Optimization Module
===================

The :mod:`homodyne.optimization` module provides two complementary optimization methods for parameter estimation in homodyne scattering analysis:

1. **NLSQ** (Primary): Fast, reliable trust-region optimization using Levenberg-Marquardt
2. **CMC** (Secondary): Bayesian uncertainty quantification using Consensus Monte Carlo

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Optimization Philosophy** (v2.4.1):

- NLSQ as primary method for fast parameter estimation
- CMC (NumPyro/NUTS) for publication-quality uncertainty quantification
- CPU-optimized architecture (v2.3.0+)
- Dataset size-aware strategy selection

**Method Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Use Case
     - Performance
   * - NLSQ
     - Fast parameter estimation, exploratory analysis
     - ~seconds for 1M points
   * - CMC
     - Uncertainty quantification, publication figures
     - ~minutes (parallelized)

Module Contents
---------------

.. automodule:: homodyne.optimization
   :noindex:

Primary Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.optimization.fit_nlsq_jax
   homodyne.optimization.fit_mcmc_jax
   homodyne.optimization.get_optimization_info

NLSQ: Non-Linear Least Squares
-------------------------------

Trust-region optimization using the Levenberg-Marquardt algorithm, implemented via the ``nlsq`` package.

Core Module
~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.core
   :members:
   :undoc-members:
   :show-inheritance:

Wrapper
~~~~~~~

High-level interface with automatic strategy selection.

.. automodule:: homodyne.optimization.nlsq.wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
^^^^^^^^^^^^

- Automatic strategy selection based on dataset size
- Memory-aware chunking for large datasets
- JIT-compiled residual functions
- Stratified sampling for per-angle scaling

Results
~~~~~~~

.. automodule:: homodyne.optimization.nlsq.results
   :members:
   :undoc-members:
   :show-inheritance:

Data Preparation
~~~~~~~~~~~~~~~~

Data preparation utilities for NLSQ optimization.

.. automodule:: homodyne.optimization.nlsq.data_prep
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.data_prep.PreparedData
   homodyne.optimization.nlsq.data_prep.ExpandedParameters

Fit Computation
~~~~~~~~~~~~~~~

Utilities for computing theoretical fits from NLSQ results.

.. automodule:: homodyne.optimization.nlsq.fit_computation
   :members:
   :undoc-members:
   :show-inheritance:

Result Builder
~~~~~~~~~~~~~~

Result building and quality metrics for NLSQ optimization.

.. automodule:: homodyne.optimization.nlsq.result_builder
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.result_builder.QualityMetrics

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

The NLSQ module implements multiple optimization strategies for different dataset sizes:

.. automodule:: homodyne.optimization.nlsq.strategies
   :members:
   :undoc-members:
   :show-inheritance:

Strategy Selection
^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.selection
   :members:
   :undoc-members:
   :show-inheritance:

Chunking Strategy
^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.chunking
   :members:
   :undoc-members:
   :show-inheritance:

Residual Functions
^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.residual
   :members:
   :undoc-members:
   :show-inheritance:

Sequential Optimization
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.sequential
   :members:
   :undoc-members:
   :show-inheritance:

Strategy Executors
^^^^^^^^^^^^^^^^^^

Implementation of the Strategy pattern for optimization execution.

.. automodule:: homodyne.optimization.nlsq.strategies.executors
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
"""""""""""

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.strategies.executors.ExecutionResult
   homodyne.optimization.nlsq.strategies.executors.OptimizationExecutor
   homodyne.optimization.nlsq.strategies.executors.StandardExecutor
   homodyne.optimization.nlsq.strategies.executors.LargeDatasetExecutor
   homodyne.optimization.nlsq.strategies.executors.StreamingExecutor

CMC: Consensus Monte Carlo
---------------------------

CMC provides Bayesian parameter estimation with full posterior sampling using NumPyro/NUTS.

**Key Features**:

- Physics-informed priors
- Automatic retry mechanism (max 3 attempts)
- Single-angle log-space D0 sampling for stability
- ArviZ-native output format

Core Module
~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.core
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.config
   :members:
   :undoc-members:
   :show-inheritance:

.. _cmc-decision-logic:

CMC vs Single-Shard MCMC Decision Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMC uses a **unified sampler architecture**: both single-shard (standard) MCMC and
per-shard CMC sampling use the identical ``run_nuts_sampling()`` function from
``homodyne/optimization/cmc/sampler.py``. The only difference is data volume and
orchestration:

**Decision Flow** (see ``homodyne/optimization/cmc/core.py:620-664``):

.. code-block:: text

   fit_mcmc_jax() called
         │
         ▼
   ┌─────────────────────────────────────────┐
   │ n_points >= min_points_for_cmc (500K)?  │
   │         OR explicit shards requested?   │
   └─────────────────────────────────────────┘
         │                    │
        YES                   NO
         │                    │
         ▼                    ▼
   ┌─────────────┐     ┌─────────────────────┐
   │ CMC Path    │     │ Single-Shard Path   │
   │             │     │                     │
   │ 1. Shard    │     │ run_nuts_sampling() │
   │    data     │     │ with ALL data       │
   │             │     │                     │
   │ 2. Backend  │     │ Returns MCMCSamples │
   │    runs     │     │ directly            │
   │    run_nuts │     └─────────────────────┘
   │    _sampling│
   │    per shard│
   │             │
   │ 3. Combine  │
   │    posteriors│
   └─────────────┘

**Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Single-Shard (Standard)
     - CMC (Sharded)
   * - Data handling
     - All points in one call
     - Subsets per shard (e.g., 10K each)
   * - Execution
     - Single ``run_nuts_sampling()``
     - Backend orchestrates parallel ``run_nuts_sampling()`` per shard
   * - Results
     - Direct posterior samples
     - Combined via precision-weighted Gaussian consensus
   * - Parallelization
     - Within-chain only
     - Across shards + within-chain
   * - Memory
     - Must fit entire dataset
     - Each shard fits independently
   * - Typical use
     - < 500K points
     - > 500K points

**Key Configuration Parameter**:

The ``min_points_for_cmc`` threshold (default: 500,000) controls automatic switching:

.. code-block:: yaml

   optimization:
     cmc:
       enable: "auto"              # "auto" | true | false
       min_points_for_cmc: 500000  # Threshold for auto-enable

- ``enable: "auto"``: Uses CMC when ``n_points >= min_points_for_cmc``
- ``enable: true``: Always uses CMC sharding (even for small datasets)
- ``enable: false``: Always uses single-shard MCMC

**Code Reference**:

The decision is made in ``fit_mcmc_jax()`` (``core.py:425-508``):

.. code-block:: python

   # Determine if CMC sharding is needed
   use_cmc = config.should_enable_cmc(prepared.n_total) or forced_shards

   if shards is not None and len(shards) > 1:
       # CMC path: parallel backend
       backend = select_backend(config)
       mcmc_samples = backend.run(
           model=xpcs_model_scaled,
           ...
       )
   else:
       # Single-shard path: direct sampling
       mcmc_samples, stats = run_nuts_sampling(
           model=xpcs_model_scaled,
           model_kwargs=model_kwargs,
           config=config,
           ...
       )

Both paths use identical:

- Model: ``xpcs_model_scaled`` (scaled/z-space parameterization)
- Sampler: ``run_nuts_sampling()`` with NumPyro NUTS
- Configuration: ``num_warmup``, ``num_samples``, ``num_chains``, ``target_accept_prob``
- Gradient balancing: Dense mass matrix (``dense_mass=True``)

.. _cmc-sharding-strategy:

Sharding Strategy (Detailed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMC (Consensus Monte Carlo) partitions large datasets into smaller **shards** that can be
processed in parallel. Each shard runs independent NUTS sampling, and posteriors are
combined using weighted Gaussian consensus.

**Why Sharding?**

NUTS MCMC is O(n) per iteration—it evaluates ALL data points in a shard for each
gradient computation. For XPCS datasets with millions of points, a single NUTS run
would take days. Sharding enables:

1. **Parallelization**: Run multiple shards simultaneously across CPU cores
2. **Memory efficiency**: Each shard fits in available RAM
3. **Timeout management**: Per-shard timeouts prevent runaway computations

Shard Size Selection Algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``_resolve_max_points_per_shard()`` function automatically selects optimal shard
sizes based on **analysis mode** and **dataset size**:

**Laminar Flow Mode** (7 parameters, complex gradients):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Dataset Size
     - Shard Size
     - Est. Shards
     - Per-Shard Runtime
   * - < 2M points
     - 20,000
     - ~100
     - ~15-25 min
   * - 2M - 20M points
     - 10,000
     - 200-2,000
     - ~5-8 min
   * - 20M - 50M points
     - 8,000
     - 2.5K-6K
     - ~5 min
   * - 50M - 100M points
     - 6,000
     - 8K-17K
     - ~4 min
   * - 100M+ points
     - 5,000
     - 20K+
     - ~3 min

**Static Mode** (3 parameters, simpler gradients):

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Dataset Size
     - Shard Size
     - Est. Shards
     - Per-Shard Runtime
   * - < 50M points
     - 100,000
     - ~500
     - ~5-10 min
   * - 50M - 100M points
     - 80,000
     - ~1K
     - ~5 min
   * - 100M+ points
     - 50,000
     - ~2K+
     - ~3-5 min

**Key insight**: Laminar flow requires ~10x smaller shards than static mode due to
the computational complexity of trigonometric functions and cumulative integrals in
the shear contribution to G1.

Sharding Strategies
^^^^^^^^^^^^^^^^^^^

CMC supports two sharding strategies:

**Stratified Sharding** (default, recommended)

Partitions data by phi angle. Each shard contains data for one angle:

- Preserves physical grouping of measurements
- Enables per-angle posterior estimates
- If an angle exceeds ``max_points_per_shard``, it's split into multiple shards
- Cap: ``max_shards_per_angle=100`` (increases shard size if exceeded)

.. code-block:: python

   shards = shard_data_stratified(
       prepared,
       num_shards=None,  # Auto-calculate
       max_points_per_shard=10000,  # For laminar_flow
       max_shards_per_angle=100,
   )

**Random Sharding**

Used when there's only one phi angle but the dataset is large:

- Shuffles data indices randomly
- Splits into approximately equal parts
- Sorts within each shard to preserve temporal structure
- ALL data is used (no subsampling)

.. code-block:: python

   shards = shard_data_random(
       prepared,
       num_shards=None,
       max_points_per_shard=10000,
       max_shards=100,  # Cap to prevent memory issues
   )

Memory Scalability
^^^^^^^^^^^^^^^^^^

Each shard result contains posterior samples that must be held in memory during
combination. Memory requirements scale with shard count:

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Platform
     - Available RAM
     - Max Shards
     - Max Dataset (laminar)
   * - Personal workstation
     - ~20 GB
     - ~500
     - ~5M points
   * - Bebop (36 cores)
     - ~100 GB
     - ~2,500
     - ~25M points
   * - Improv (128 cores)
     - ~200 GB
     - ~5,000
     - ~50M points

**Memory formula**: Each shard result ≈ 100 KB (13 params × 2 chains × 1500 samples × 8 bytes).
Peak memory ≈ 6 × K MB where K = number of shards.

The algorithm automatically caps shard count (default: 2000) and increases shard size
to prevent memory exhaustion. For very large datasets exceeding limits, a warning
is logged.

Runtime Estimation
^^^^^^^^^^^^^^^^^^

CMC provides runtime estimates before sampling begins:

.. code-block:: text

   Runtime estimate: 2.5h total (100 shards / 18 workers, ~15min/shard with 4000 iterations)

The estimate accounts for:

- JIT compilation overhead (~30-60s per worker)
- MCMC iterations: ``num_chains × (num_warmup + num_samples)``
- Points per shard and analysis mode complexity
- Parallel execution across available workers

After completion, actual vs. estimated runtime is logged with recommendations:

.. code-block:: text

   Runtime: 2.1h actual vs 2.5h estimated (84% - close to estimate)

Configuration Reference
^^^^^^^^^^^^^^^^^^^^^^^

Full YAML configuration for sharding:

.. code-block:: yaml

   optimization:
     cmc:
       enable: auto  # true, false, or "auto" (based on data size)
       min_points_for_cmc: 500000  # Threshold for auto-enable

       sharding:
         strategy: stratified  # "stratified" or "random"
         num_shards: auto  # "auto" or explicit integer
         max_points_per_shard: auto  # "auto" or explicit integer

       backend_config:
         name: multiprocessing  # "auto", "multiprocessing", "pjit", "pbs"
         enable_checkpoints: true
         checkpoint_dir: ./checkpoints/cmc

       per_shard_mcmc:
         num_warmup: 500
         num_samples: 1500
         num_chains: 2
         target_accept_prob: 0.85

       validation:
         max_per_shard_rhat: 1.1
         min_per_shard_ess: 100

       combination:
         method: weighted_gaussian  # or "simple_average"
         min_success_rate: 0.90

       per_shard_timeout: 7200  # 2 hours max per shard
       heartbeat_timeout: 600   # 10 min worker heartbeat

**Critical settings for laminar_flow**:

- Use ``max_points_per_shard: auto`` (resolves to 5K-20K based on size)
- Do NOT set ``max_points_per_shard: 100000`` — this causes 1-2+ hour per-shard runtimes
- Keep ``num_warmup`` and ``num_samples`` aligned between ``mcmc`` and ``per_shard_mcmc``

Practical Guidelines
^^^^^^^^^^^^^^^^^^^^

**For typical 3-angle, 3M point laminar_flow datasets**:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: auto  # Will select ~10K-20K
       per_shard_mcmc:
         num_warmup: 300
         num_samples: 700
         num_chains: 2

Expected: ~150-300 shards, ~5-8 min/shard, ~2-4 hours total on 18-core workstation.

**For 50M+ point production datasets on HPC**:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: auto  # Will select ~6K-8K
       per_shard_mcmc:
         num_warmup: 500
         num_samples: 1500
         num_chains: 2
       per_shard_timeout: 7200  # 2 hours

Expected: ~6K-8K shards, parallel execution across cluster nodes.

Model Definition
~~~~~~~~~~~~~~~~

NumPyro model definition for MCMC sampling.

.. automodule:: homodyne.optimization.cmc.model
   :members:
   :undoc-members:
   :show-inheritance:

Priors
~~~~~~

Physics-informed prior distributions.

.. automodule:: homodyne.optimization.cmc.priors
   :members:
   :undoc-members:
   :show-inheritance:

Prior Specifications
^^^^^^^^^^^^^^^^^^^^

**Static Mode** (3 physical parameters):

- D0: LogNormal(log(1000), 1.5)
- alpha: Uniform(0.0, 2.0)
- D_offset: TruncatedNormal(0, 100, low=0)

**Laminar Flow Mode** (+4 shear parameters):

- gamma_dot_t0: LogNormal(log(100), 1.5)
- beta: Uniform(-2.0, 2.0)
- gamma_dot_t_offset: TruncatedNormal(0, 100, low=0)
- phi0: Uniform(0, 2π)

**Per-Angle Scaling** (mandatory in v2.4.0+):

- contrast_i: TruncatedNormal(0.5, 0.3, low=0.1, high=2.0) for each angle i
- offset_i: TruncatedNormal(1.0, 0.2, low=0.5, high=1.5) for each angle i

Data Preparation
~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.data_prep
   :members:
   :undoc-members:
   :show-inheritance:

Sampler
~~~~~~~

NUTS sampler interface with warmup and sampling phases.

.. automodule:: homodyne.optimization.cmc.sampler
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Scaling (Gradient Balancing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.scaling
   :members:
   :undoc-members:
   :show-inheritance:

**Understanding Z-Space Parameters**

CMC uses non-centered parameterization to balance gradient magnitudes across
parameters with vastly different scales (e.g., D0 ~ 10^4 vs gamma_dot_t0 ~ 10^-3).

When sampling, parameters are transformed to normalized z-space:

- Each parameter is sampled as ``z ~ Normal(0, 1)``
- Transformed to original space: ``param = center + scale * z``
- Clipped to physical bounds

**MCMC Sample Names**:

The MCMC output includes both z-space and original-space parameter names:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Z-Space Name
     - Original Name
     - Description
   * - ``D0_z``
     - ``D0``
     - Diffusion coefficient (normalized / original)
   * - ``alpha_z``
     - ``alpha``
     - Diffusion exponent
   * - ``contrast_0_z``
     - ``contrast_0``
     - Per-angle contrast (phi index 0)
   * - ``offset_0_z``
     - ``offset_0``
     - Per-angle offset (phi index 0)

**Filtering Samples**:

When working with MCMC samples, you may want to filter out z-space parameters::

    # Get only original-space parameters
    original_params = {k: v for k, v in samples.items() if not k.endswith('_z')}

    # Get only physical parameters (exclude sigma, n_numerical_issues)
    physical_params = ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
                       'gamma_dot_t_offset', 'phi0']
    physical_samples = {k: v for k, v in samples.items() if k in physical_params}

Results
~~~~~~~

.. automodule:: homodyne.optimization.cmc.results
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
~~~~~~~~~~~

MCMC convergence diagnostics including R-hat, effective sample size (ESS), and divergence analysis.

.. automodule:: homodyne.optimization.cmc.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.cmc.diagnostics.compute_r_hat
   homodyne.optimization.cmc.diagnostics.compute_ess
   homodyne.optimization.cmc.diagnostics.count_divergences
   homodyne.optimization.cmc.diagnostics.check_convergence
   homodyne.optimization.cmc.diagnostics.create_diagnostics_dict
   homodyne.optimization.cmc.diagnostics.summarize_diagnostics
   homodyne.optimization.cmc.diagnostics.log_analysis_summary
   homodyne.optimization.cmc.diagnostics.get_convergence_recommendations

Convergence Thresholds
^^^^^^^^^^^^^^^^^^^^^^

**Default thresholds**:

- ``MAX_RHAT``: 1.05 (chains should have R-hat < 1.05 for convergence)
- ``MIN_ESS``: 400 (effective sample size should exceed 400)
- ``MAX_DIVERGENCE_RATE``: 5% (divergence rate should be < 5%)

**Diagnostics Output**:

The ``check_convergence`` function returns one of three statuses:

- ``converged``: All chains mixed well, ESS adequate, no excessive divergences
- ``divergences``: High divergence rate indicates model geometry issues
- ``not_converged``: R-hat or ESS thresholds not met

I/O Operations
~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.io
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

.. automodule:: homodyne.optimization.cmc.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Backends
~~~~~~~~

CMC supports multiple parallelization backends for distributed MCMC execution.

.. automodule:: homodyne.optimization.cmc.backends
   :members:
   :undoc-members:
   :show-inheritance:

Backend Selection
^^^^^^^^^^^^^^^^^

**Available backends**:

- ``multiprocessing``: Python multiprocessing for multi-core workstations (default)
- ``pjit``: JAX pjit for single-node multi-device parallelism
- ``pbs``: PBS job scheduler for HPC clusters

**Backend Configuration**:

The backend is auto-selected based on environment, but can be overridden via configuration:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         backend: multiprocessing  # or pjit, pbs

Per-Shard Sampling Behavior
^^^^^^^^^^^^^^^^^^^^^^^^^^^

All backends follow the same per-shard sampling pattern:

1. **Shard preparation**: Extract data subset with associated metadata (phi indices, time arrays)
2. **Model kwargs construction**: Build model arguments for the shard's data
3. **Sampler invocation**: Call ``run_nuts_sampling()`` with shard-specific data
4. **Result collection**: Gather ``MCMCSamples`` and ``SamplingStats``

**Per-shard execution** (simplified from ``backends/multiprocessing.py``):

.. code-block:: python

   def _run_single_shard(shard_data, config, model, ...):
       # Build model kwargs for this shard
       model_kwargs = {
           "data": shard_data.data,
           "t1": shard_data.t1,
           "t2": shard_data.t2,
           "phi_indices": shard_data.phi_indices,
           ...
       }

       # Same sampler as single-shard path
       samples, stats = run_nuts_sampling(
           model=model,
           model_kwargs=model_kwargs,
           config=config,
           initial_values=initial_values,
           ...
       )
       return samples, stats

**What each shard receives**:

- Subset of data points (respecting ``max_points_per_shard``)
- Full ``phi_unique`` array (all angles, for proper indexing)
- Shard-specific ``phi_indices`` (mapping points to angles)
- Same physics parameters (``q``, ``L``, ``dt``, ``time_grid``)
- Same MCMC configuration (``num_warmup``, ``num_samples``, etc.)

**What each shard produces**:

- ``MCMCSamples``: Posterior samples for all parameters
- ``SamplingStats``: Timing, divergences, acceptance rate
- Per-shard diagnostics: R-hat, ESS (within-shard convergence)

The combination phase (see :ref:`cmc-sharding-strategy`) then merges these
independent subposteriors using precision-weighted Gaussian consensus.

Base Backend
^^^^^^^^^^^^

.. automodule:: homodyne.optimization.cmc.backends.base
   :members:
   :undoc-members:
   :show-inheritance:

Multiprocessing Backend
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.cmc.backends.multiprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Key features:

- Automatic worker allocation based on CPU cores
- Configurable timeout handling
- Progress tracking with shard completion estimates
- Memory-efficient worker pool management

PJIT Backend
^^^^^^^^^^^^

.. automodule:: homodyne.optimization.cmc.backends.pjit
   :members:
   :undoc-members:
   :show-inheritance:

PBS Backend
^^^^^^^^^^^

.. automodule:: homodyne.optimization.cmc.backends.pbs
   :members:
   :undoc-members:
   :show-inheritance:

Supporting Modules
------------------

The optimization module includes several supporting utilities:

Checkpoint Manager
~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.checkpoint_manager
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Diagnostics
~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.gradient_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
~~~~~~~~~~

.. automodule:: homodyne.optimization.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Recovery Strategies
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.recovery_strategies
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**NLSQ Optimization**::

    from homodyne.optimization import fit_nlsq_jax

    result = fit_nlsq_jax(
        t1=t1,
        t2=t2,
        c2=c2,
        q2_phi_t1_t2=q2_phi_t1_t2,
        phi_rad=phi_rad,
        initial_params=initial_params,
        mode="static"
    )

    print(f"Best-fit D0: {result.params['D0']:.2f}")

**CMC Optimization**::

    from homodyne.optimization import fit_mcmc_jax, CMCConfig

    config = CMCConfig(
        num_warmup=1000,
        num_samples=2000,
        num_chains=4
    )

    result = fit_mcmc_jax(
        t1=t1,
        t2=t2,
        c2=c2,
        q2_phi_t1_t2=q2_phi_t1_t2,
        phi_rad=phi_rad,
        initial_params=initial_params,
        mode="static",
        config=config
    )

    # Access posterior samples
    print(result.summary())

See Also
--------

- :mod:`homodyne.core` - Core physics and computation
- :mod:`homodyne.config` - Parameter management
- External: `NLSQ Package Documentation <https://nlsq.readthedocs.io/>`_
- External: `NumPyro Documentation <https://num.pyro.ai/>`_
