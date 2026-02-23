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

**Optimization Philosophy**:

- NLSQ as primary method for fast parameter estimation
- CMC (NumPyro/NUTS) for publication-quality uncertainty quantification
- CPU-optimized architecture
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

Wrapper (Legacy)
~~~~~~~~~~~~~~~~

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

.. _nlsq-adapter:

NLSQAdapter
~~~~~~~~~~~~~~~~~~~~~~

Modern adapter for NLSQ v0.4+ CurveFit class with model caching and JIT support.
This is the **recommended** path for new optimizations.

.. automodule:: homodyne.optimization.nlsq.adapter
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.adapter.NLSQAdapter
   homodyne.optimization.nlsq.adapter.AdapterConfig
   homodyne.optimization.nlsq.adapter.ModelCacheKey
   homodyne.optimization.nlsq.adapter.CachedModel

Key Features
^^^^^^^^^^^^^^^^^^^^^^

**Model Caching (3-5× Multi-Start Speedup)**:

- Cached model instances avoid redundant model creation
- LRU eviction with 64-entry cache limit
- Cache hit/miss statistics for monitoring

**JIT Compilation Flag**:

- Signals intent for JIT optimization
- Underlying CombinedModel uses JAX internally
- Graceful fallback if JAX unavailable

**Automatic Fallback**:

- NLSQAdapter failures automatically retry with NLSQWrapper
- Logged warnings include original error for debugging
- Fallback metadata in ``device_info``

Configuration
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.optimization.nlsq import AdapterConfig, NLSQAdapter

   config = AdapterConfig(
       enable_cache=True,      # Model caching (default: True)
       enable_jit=True,        # JIT compilation (default: True)
       enable_recovery=True,   # NLSQ recovery system
       goal="quality",         # Optimization goal
   )

   adapter = NLSQAdapter(config)
   result = adapter.fit(data, config, initial_params, bounds, analysis_mode)

Cache Management
^^^^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.optimization.nlsq import get_cache_stats, clear_model_cache

   # View cache statistics
   stats = get_cache_stats()
   print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

   # Clear cache (useful for testing)
   n_cleared = clear_model_cache()

When to Use Which Adapter
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Adapter
     - Use Case
     - Advantages
   * - **NLSQAdapter**
     - Multi-start optimization, repeated fits
     - Model caching, modern API
   * - **NLSQWrapper**
     - Complex workflows, anti-degeneracy
     - Full feature set, streaming support

Memory Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Utilities for adaptive memory thresholds and streaming decisions.

.. automodule:: homodyne.optimization.nlsq.memory
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Helper functions for parameter handling and per-angle initialization.

.. automodule:: homodyne.optimization.nlsq.parameter_utils
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Index Mapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single source of truth for parameter indices across all anti-degeneracy modes.

.. automodule:: homodyne.optimization.nlsq.parameter_index_mapper
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.parameter_index_mapper.ParameterIndexMapper

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

   # Constant mode (23 phi angles, 7 physical params)
   mapper = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=True)
   print(mapper.mode_name)           # "constant"
   print(mapper.n_per_angle_total)   # 2 (single contrast + offset, shared)
   print(mapper.total_params)        # 9 (2 + 7)

   # Fourier mode (order=2)
   mapper = ParameterIndexMapper(n_phi=23, n_physical=7, use_fourier=True, fourier_order=2)
   print(mapper.mode_name)           # "fourier"
   print(mapper.n_per_angle_total)   # 10 (5 contrast + 5 offset coefficients)
   print(mapper.total_params)        # 17 (10 + 7)

Jacobian Utilities
~~~~~~~~~~~~~~~~~~

Jacobian computation utilities for convergence diagnostics.

.. automodule:: homodyne.optimization.nlsq.jacobian
   :members:
   :undoc-members:
   :show-inheritance:

Progress Tracking
~~~~~~~~~~~~~~~~~

Progress bar and logging callbacks for NLSQ optimization.

.. automodule:: homodyne.optimization.nlsq.progress
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.progress.ProgressConfig

Parameter Transforms
~~~~~~~~~~~~~~~~~~~~

Parameter transformation utilities and name normalization.

.. automodule:: homodyne.optimization.nlsq.transforms
   :members:
   :undoc-members:
   :show-inheritance:

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

JIT-Compiled Residual Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JIT-compatible stratified residual function using padded vmap for full JIT compilation.

.. automodule:: homodyne.optimization.nlsq.strategies.residual_jit
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
"""""""""""

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.strategies.residual_jit.StratifiedResidualFunctionJIT

Key Features
"""""""""""""""""""""

- **Static shapes**: Pads chunks to uniform size for JIT compatibility
- **vmap vectorization**: Parallel chunk processing without Python loops
- **Angle stratification**: Maintains all angles in each chunk

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

.. _nlsq-multistart-optimizer:

Multi-Start Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-start optimization explores parameter space from multiple starting points
using Latin Hypercube Sampling to find the global optimum and detect parameter
degeneracy.

.. automodule:: homodyne.optimization.nlsq.multistart
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.multistart.MultiStartConfig
   homodyne.optimization.nlsq.multistart.MultiStartResult
   homodyne.optimization.nlsq.multistart.SingleStartResult

Configuration
^^^^^^^^^^^^^

Multi-start can be enabled via YAML configuration:

.. code-block:: yaml

   optimization:
     nlsq:
       multi_start:
         enable: true
         n_starts: 10
         seed: 42
         sampling_strategy: latin_hypercube
         use_screening: true
         screen_keep_fraction: 0.5
         refine_top_k: 3
         degeneracy_threshold: 0.1

Key Features
^^^^^^^^^^^^

- **Latin Hypercube Sampling**: Better space-filling than random sampling
- **Screening Phase**: Filters poor starting points before expensive optimization
- **Parallel Execution**: Uses ProcessPoolExecutor for multi-core parallelism
- **Basin Clustering**: Identifies unique local minima in parameter space
- **Degeneracy Detection**: Warns when multiple solutions have similar chi-squared
- **FULL Strategy Only**: No subsampling per project requirements (numerical precision priority)

Determining n_starts
^^^^^^^^^^^^^^^^^^^^

The number of starting points (``n_starts``) significantly impacts both optimization
quality and computational cost. This section provides guidance for selecting appropriate
values.

**Minimum Requirements**

For Latin Hypercube Sampling to provide adequate parameter space coverage,
``n_starts`` should be at least equal to the number of parameters:

.. list-table:: Minimum n_starts by Analysis Mode
   :header-rows: 1
   :widths: 30 40 30

   * - Analysis Mode
     - Parameters
     - Minimum n_starts
   * - static_isotropic
     - 5 (contrast, offset, D₀, α, D_offset)
     - 5
   * - laminar_flow
     - 9 (+ γ̇₀, β, γ̇_offset, φ₀)
     - 9
   * - laminar_flow + per-angle (individual)
     - 2×n_phi + 7
     - 2×n_phi + 7
   * - laminar_flow + per-angle (constant)
     - 2 + 7 = 9
     - 9

**Impact of Anti-Degeneracy per_angle_mode**

The ``per_angle_mode`` setting dramatically affects parameter count and thus n_starts:

.. list-table:: Parameter Count by per_angle_mode (23-angle laminar_flow)
   :header-rows: 1
   :widths: 20 25 25 30

   * - Mode
     - Per-Angle Params
     - Total Params
     - Recommended n_starts
   * - individual
     - 2 × 23 = 46
     - 53
     - 100-150
   * - fourier (order=2)
     - 2 × 5 = 10
     - 17
     - 20-40
   * - **constant**
     - 2
     - **9**
     - **10-20**

**Constant mode** (``per_angle_mode: "constant"``) assumes all angles share the same
contrast and offset, reducing parameter count from 53 to 9. This makes multi-start
optimization tractable for many-angle datasets.

**Recommended Settings by Use Case**

.. list-table:: n_starts Recommendations
   :header-rows: 1
   :widths: 25 20 55

   * - Use Case
     - n_starts Formula
     - Description
   * - Quick exploration
     - 10
     - Default, fast baseline
   * - Standard analysis
     - 2 × n_params
     - Good coverage of parameter space
   * - Degeneracy detection
     - 3 × n_params
     - Better basin discovery
   * - Publication quality
     - 5 × n_params
     - Thorough exploration

**Screening Considerations**

When ``use_screening: true`` (default), only a fraction of starting points proceed
to full optimization:

- With ``screen_keep_fraction: 0.5`` (default):
  - 20 starts → 10 full optimizations
  - 100 starts → 50 full optimizations

Increase ``n_starts`` accordingly to achieve desired effective sample size.

**Computational Cost**

- Execution time scales linearly with effective n_starts
- For datasets ≥ 500K points: sequential execution (no parallelism benefit)
- Each fit runs the full optimization pipeline

**Example Configuration**

.. code-block:: yaml

   optimization:
     nlsq:
       # Use constant mode to reduce parameters (53 → 9)
       anti_degeneracy:
         enable: true
         per_angle_mode: "constant"
         constant_scaling_threshold: 3

       multi_start:
         enable: true
         n_starts: 20              # ~2× for 9 params
         use_screening: true
         screen_keep_fraction: 0.5 # 10 full fits
         seed: 42

**Validation Warning**

The code validates ``n_starts`` and warns if inadequate:

.. code-block:: text

   WARNING: n_starts (5) < n_params (9): LHS coverage may be inadequate.
   Consider n_starts >= 9.

CLI Integration
^^^^^^^^^^^^^^^

.. code-block:: bash

   homodyne --config config.yaml --method nlsq --multi-start

.. _nlsq-cmaes-optimizer:

CMA-ES Global Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) provides robust global
optimization for multi-scale parameter estimation problems. It excels when
parameter scales differ by several orders of magnitude, such as in laminar_flow
mode (D₀ ~ 10⁴ vs γ̇₀ ~ 10⁻³).

.. automodule:: homodyne.optimization.nlsq.cmaes_wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.cmaes_wrapper.CMAESWrapper
   homodyne.optimization.nlsq.cmaes_wrapper.CMAESWrapperConfig
   homodyne.optimization.nlsq.cmaes_wrapper.CMAESResult

When to Use CMA-ES
^^^^^^^^^^^^^^^^^^

CMA-ES is recommended when:

- **Multi-scale parameters**: Scale ratio > 1000 (e.g., D₀/γ̇₀ > 10⁶)
- **Complex loss landscapes**: Multiple local minima, saddle points
- **Poor initial guess**: CMA-ES explores globally, not just locally
- **laminar_flow mode**: 7 physical parameters with vastly different scales

The ``CMAESWrapper.should_use_cmaes()`` method automatically detects multi-scale
problems by computing the scale ratio from parameter bounds.

Two-Phase Architecture
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: text

   Phase 1: CMA-ES Global Search
   ├─ Population-based evolutionary optimization
   ├─ Covariance matrix adapts to parameter scales
   ├─ BIPOP restart strategy (alternating large/small populations)
   └─ Memory batching: population_batch_size, data_chunk_size

   Phase 2: NLSQ TRF Refinement (if refine_with_nlsq=True)
   ├─ Uses NLSQ curve_fit with workflow="auto"
   ├─ Memory-aware: auto-selects standard/chunked/streaming
   ├─ Tighter tolerances (ftol=1e-10 vs CMA-ES 1e-8)
   └─ Provides proper covariance matrix via Jacobian

Configuration
^^^^^^^^^^^^^

CMA-ES can be configured via YAML:

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true                      # Enable CMA-ES global optimization
         preset: "cmaes"                   # "cmaes-fast" (50), "cmaes" (100), "cmaes-global" (200)
         max_generations: 100              # Maximum CMA-ES generations
         sigma: 0.5                        # Initial step size (fraction of bounds)
         tol_fun: 1.0e-8                   # Function tolerance
         tol_x: 1.0e-8                     # Parameter tolerance
         restart_strategy: "bipop"         # "none" or "bipop"
         max_restarts: 9                   # Maximum BIPOP restarts
         population_batch_size: null       # null = auto, or explicit batch size
         data_chunk_size: null             # null = auto, or explicit chunk size
         refine_with_nlsq: true            # Refine with NLSQ TRF after CMA-ES
         auto_select: true                 # Auto-select CMA-ES vs multi-start
         scale_threshold: 1000.0           # Scale ratio threshold

         # NLSQ TRF Refinement Settings
         refinement_workflow: "auto"       # "auto", "standard", "streaming"
         refinement_ftol: 1.0e-10          # Tighter for local refinement
         refinement_xtol: 1.0e-10
         refinement_gtol: 1.0e-10
         refinement_max_nfev: 500          # Bounded iterations
         refinement_loss: "linear"         # "linear", "soft_l1", "huber"

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.optimization.nlsq.cmaes_wrapper import CMAESWrapper, CMAESWrapperConfig

   # Create wrapper with custom config
   wrapper = CMAESWrapper(CMAESWrapperConfig(
       preset="cmaes",
       refine_with_nlsq=True,
       refinement_ftol=1e-10,
   ))

   # Check if CMA-ES is appropriate for this problem
   if wrapper.should_use_cmaes(bounds, scale_threshold=1000):
       result = wrapper.fit(model_func, xdata, ydata, p0, bounds)
       print(f"Chi²: {result.chi_squared:.4e}")
       print(f"Refined: {result.nlsq_refined}")
       print(f"Improvement: {result.diagnostics.get('chi_squared_improvement', 0):.2%}")

CMA-ES vs Multi-Start vs Local
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Method
     - Best For
     - Convergence
     - Memory
   * - **CMA-ES**
     - Multi-scale (ratio > 1000)
     - Global (covariance)
     - Bounded
   * - **Multi-start**
     - Multiple local minima
     - Local from N starts
     - N × single fit
   * - **Local (TRF)**
     - Good initial guess
     - Local (quadratic)
     - Jacobian-based

.. _nlsq-streaming-optimizer:

Streaming Optimizer for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets exceeding available memory (>10M points on typical systems), the NLSQ
wrapper automatically switches to **streaming optimization** using mini-batch gradient
descent. This eliminates OOM errors by processing data in small batches.

**Why Streaming?**

Standard Levenberg-Marquardt optimization requires computing a dense Jacobian matrix
(n_points × n_params × 8 bytes) plus JAX autodiff intermediates (~3× Jacobian size).
For 23M points with 53 parameters, this exceeds 30 GB. Streaming mode processes data
in 10K-point batches, keeping memory usage below 2 GB.

Memory-Based Auto-Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``NLSQWrapper._should_use_streaming()`` method estimates peak memory usage and
automatically selects streaming mode when:

1. Estimated memory > ``memory_threshold_gb`` (default: 16 GB), OR
2. Estimated memory > 70% of available system RAM

**Decision Logic**:

.. code-block:: text

   fit() called
         │
         ▼
   ┌─────────────────────────────────────────┐
   │ Estimate memory for Jacobian + autodiff │
   │ = n_points × n_params × 8 × 4           │
   └─────────────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────────────┐
   │ Estimated > threshold OR > 70% RAM?     │
   └─────────────────────────────────────────┘
         │                    │
        YES                   NO
         │                    │
         ▼                    ▼
   ┌─────────────┐     ┌─────────────────────┐
   │ Streaming   │     │ Stratified L-M      │
   │ Optimizer   │     │ (Full Jacobian)     │
   │             │     │                     │
   │ Mini-batch  │     │ Trust-region        │
   │ L-BFGS      │     │ Levenberg-Marquardt │
   └─────────────┘     └─────────────────────┘

Configuration
^^^^^^^^^^^^^

Streaming mode can be configured via YAML:

.. code-block:: yaml

   optimization:
     nlsq:
       # Memory threshold for automatic streaming mode (GB)
       memory_threshold_gb: 16.0

       # Force streaming mode regardless of memory (default: false)
       use_streaming: false

       # Streaming optimizer settings
       streaming:
         batch_size: 10000       # Points per mini-batch
         max_epochs: 50          # Maximum training epochs
         learning_rate: 0.001    # L-BFGS line search scale
         convergence_tol: 1e-6   # Convergence tolerance

Performance Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 20 25 35

   * - Mode
     - Memory Usage
     - Convergence
     - Time (23M points)
   * - Stratified L-M
     - ~30+ GB
     - Exact (Newton)
     - 10-15 min
   * - Streaming
     - ~2 GB
     - Approximate (L-BFGS)
     - 15-30 min

**When to Use**:

- **Stratified L-M (default)**: Datasets < 10M points, sufficient RAM (64GB+)
- **Streaming**: Datasets > 10M points, memory-constrained systems (32GB RAM)

Implementation Details
^^^^^^^^^^^^^^^^^^^^^^

The streaming optimizer uses NLSQ's ``AdaptiveHybridStreamingOptimizer`` class:

.. code-block:: python

   from nlsq import AdaptiveHybridStreamingOptimizer, HybridStreamingConfig

   config = HybridStreamingConfig(
       chunk_size=50000,
       warmup_iterations=100,
       max_warmup_iterations=500,
       gauss_newton_max_iterations=50,
       gauss_newton_tol=1e-8,
       normalize=True,
       normalization_strategy="bounds",
   )

   optimizer = AdaptiveHybridStreamingOptimizer(config)
   result = optimizer.fit(
       data_source=(x_data, y_data),
       func=model_fn,
       p0=initial_params,
       bounds=bounds,
   )

Key features:

- **4-phase hybrid optimization**: L-BFGS warmup + Gauss-Newton refinement
- **Parameter normalization**: Equalizes gradient magnitudes across multi-scale parameters
- **Exact J^T J accumulation**: Proper covariance estimation in streaming mode
- **Chunk-based processing**: Memory-efficient for unlimited dataset sizes
- **Progress tracking**: Logs phase progress and convergence metrics

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
     - 4,800
     - ~400
     - ~1-2 min
   * - 2M - 50M points
     - 3,000
     - 600-16K
     - ~1 min
   * - 50M - 100M points
     - 4,800
     - 10K-20K
     - ~1 min
   * - 100M - 1B points
     - 4,800
     - 20K-50K
     - <1 min
   * - 1B+ points
     - 6,000-10,000
     - 100K+
     - <1 min

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

**Key insight**: Laminar flow uses ~20x smaller shards than static mode. The
reparameterization to (D_ref, gamma_ref) produces unimodal posteriors, enabling
3-5K shards with adaptive sampling and prior tempering.

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
       max_points_per_shard=5000,  # For laminar_flow
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
         # Adaptive Sampling
         adaptive_sampling: true           # Scale by shard size
         max_tree_depth: 10                # NUTS tree depth limit
         min_warmup: 100                   # Minimum warmup floor
         min_samples: 200                  # Minimum samples floor
         # JAX Profiling
         enable_jax_profiling: false       # XLA-level profiling
         jax_profile_dir: "./profiles/jax"

       validation:
         max_per_shard_rhat: 1.1
         min_per_shard_ess: 100

       combination:
         method: robust_consensus_mc  # MAD-based outlier detection (default)
         min_success_rate: 0.90

       validation:
         max_per_shard_rhat: 1.1
         min_per_shard_ess: 100
         max_divergence_rate: 0.10       # Quality filter: exclude shards >10%
         require_nlsq_warmstart: false   # Require NLSQ warm-start

       per_shard_timeout: 3600  # 1 hour max per shard (reduced)
       heartbeat_timeout: 600   # 10 min worker heartbeat

**Critical settings for laminar_flow**:

- Use ``max_points_per_shard: auto`` (resolves to 3K-5K based on size)
- Do NOT set ``max_points_per_shard: 100000`` — this causes 1-2+ hour per-shard runtimes
- Keep ``num_warmup`` and ``num_samples`` aligned between ``mcmc`` and ``per_shard_mcmc``
- Consider ``require_nlsq_warmstart: true`` for production runs (reduces divergences from ~28% to <5%)

**Quality Filtering**:

The ``max_divergence_rate`` setting automatically filters out shards with excessive
divergent transitions before consensus combination:

.. code-block:: yaml

   optimization:
     cmc:
       validation:
         max_divergence_rate: 0.10  # Exclude shards with >10% divergence

Shards with divergence rate exceeding this threshold are excluded from the final
posterior combination, preventing corrupted posteriors from biasing estimates.

**NLSQ Warm-Start Requirement**:

For laminar_flow mode with 7 parameters spanning 6+ orders of magnitude, cold-start
CMC runs often show high divergence rates (28%+) and inflated uncertainty. Enable
warm-start requirement for production:

.. code-block:: yaml

   optimization:
     cmc:
       validation:
         require_nlsq_warmstart: true

When enabled, ``fit_mcmc_jax()`` will raise ``ValueError`` if called without
``nlsq_result`` or ``initial_values`` for laminar_flow mode

**Adaptive Sampling**:

Adaptive sampling automatically scales warmup and sample counts based on shard size,
reducing NUTS overhead by 60-80% for small datasets while maintaining statistical validity.

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         adaptive_sampling: true     # Enable adaptive scaling
         max_tree_depth: 10          # Limit NUTS tree depth (2^10 max leapfrog)
         min_warmup: 100             # Floor for warmup scaling
         min_samples: 200            # Floor for samples scaling

The scaling formula uses a 10K point reference:

- ``scale_factor = min(1.0, shard_size / 10000)``
- Small shards (< 10K points) get proportionally fewer warmup/samples
- Minimum samples scale with parameter count: ``max(min_samples, 50 × n_params)``

This optimization was informed by profiling showing that XLA JIT compilation and
NUTS leapfrog integration dominate runtime (not Python overhead), making sample
count reduction the most effective optimization.

**JAX Profiling**:

XLA-level profiling for diagnosing NUTS performance bottlenecks. Standard Python
profilers (py-spy, cProfile) cannot see inside JIT-compiled code.

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         enable_jax_profiling: true
         jax_profile_dir: "./profiles/jax"

View profiles with TensorBoard: ``tensorboard --logdir=./profiles/jax``

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

.. _cmc-per-angle-modes:

CMC Per-Angle Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMC supports three per-angle modes that control how contrast/offset parameters are handled
during MCMC sampling. This matches the NLSQ anti-degeneracy system for consistent behavior
across optimization backends.

.. list-table:: CMC Per-Angle Mode Comparison
   :header-rows: 1
   :widths: 15 25 20 40

   * - Mode
     - Sampled Params
     - Per-Angle Handling
     - Use Case
   * - ``auto`` (default)
     - 8 (7 physical + σ)
     - Quantile → average → broadcast
     - Default for n_phi ≥ 3 (NLSQ parity)
   * - ``constant``
     - 8 (7 physical + σ)
     - Quantile → use directly (fixed)
     - Different fixed value per angle
   * - ``individual``
     - 8 + 2×n_phi
     - All sampled independently
     - Full flexibility (n_phi < 3)

**Auto Mode (Default)**:

When ``per_angle_mode: "auto"`` and n_phi ≥ 3 (configurable via ``constant_scaling_threshold``):

1. Estimates per-angle contrast/offset from data using quantile analysis
2. **Averages** the per-angle estimates to single values
3. **Broadcasts** the averaged values to all angles (same fixed value for all)
4. Only samples 8 parameters: 7 physical + 1 sigma

This provides NLSQ parity—CMC ``auto`` mode matches NLSQ ``constant`` mode behavior.

**Constant Mode**:

When ``per_angle_mode: "constant"``:

1. Estimates per-angle contrast/offset from data using quantile analysis
2. Uses the per-angle estimates **directly** (different fixed value per angle)
3. Only samples 8 parameters: 7 physical + 1 sigma

Both ``auto`` (n_phi ≥ 3) and ``constant`` modes use fixed scaling arrays passed to the
model function, reducing degeneracy risk by not sampling per-angle parameters.

**Individual Mode**:

When ``per_angle_mode: "individual"`` or ``auto`` with n_phi < 3:

1. Samples contrast and offset for each phi angle independently
2. Total sampled parameters: 8 + 2×n_phi
3. Full flexibility but higher degeneracy risk for large n_phi

**Model Selection**:

.. code-block:: python

   from homodyne.optimization.cmc.model import get_xpcs_model

   # Get appropriate model function
   model = get_xpcs_model("constant")  # Returns xpcs_model_constant
   model = get_xpcs_model("individual")  # Returns xpcs_model_scaled
   model = get_xpcs_model()  # Default: xpcs_model_scaled

**Configuration**:

.. code-block:: yaml

   optimization:
     cmc:
       per_angle_mode: "auto"           # "auto", "constant", "individual"
       constant_scaling_threshold: 3    # Threshold for auto mode

Key Functions
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.cmc.model.xpcs_model_constant
   homodyne.optimization.cmc.model.xpcs_model_scaled
   homodyne.optimization.cmc.model.get_xpcs_model
   homodyne.optimization.cmc.model.get_model_param_count
   homodyne.optimization.cmc.priors.get_param_names_in_order
   homodyne.optimization.cmc.priors.build_init_values_dict
   homodyne.optimization.cmc.config.CMCConfig.get_effective_per_angle_mode

.. _cmc-convergence-fixes:

CMC Convergence and Precision Fixes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section documents comprehensive fixes for CMC failures on multi-angle datasets, addressing
94% shard timeout rates, 28.4% divergence rates, and 33-43x uncertainty inflation observed
in 3-angle laminar_flow analysis.

Angle-Aware Shard Sizing
^^^^^^^^^^^^^^^^^^^^^^^^

The ``_resolve_max_points_per_shard()`` function now accepts an ``n_phi`` parameter that
scales shard sizes inversely with angle count:

.. list-table:: Angle-Aware Shard Scaling
   :header-rows: 1
   :widths: 20 25 55

   * - n_phi
     - Scale Factor
     - Rationale
   * - ≤ 3
     - 30%
     - Few-angle data has more complex per-shard posteriors
   * - 4-5
     - 50%
     - Moderate angle count
   * - 6-10
     - 70%
     - Good angle coverage per shard
   * - > 10
     - 100%
     - Full capacity

**Example**: For 3-angle laminar_flow with base 20K shard size, effective size = 6K points.

Angle-Balanced Sharding
^^^^^^^^^^^^^^^^^^^^^^^

New ``shard_data_angle_balanced()`` function ensures proportional angle coverage per shard:

.. code-block:: python

   from homodyne.optimization.cmc.data_prep import shard_data_angle_balanced

   shards = shard_data_angle_balanced(
       prepared,
       num_shards=None,           # Auto-calculate
       max_points_per_shard=6000, # Angle-aware size
       min_angle_coverage=0.8,    # 80% minimum coverage
       seed=42,
   )

Key features:

- Samples proportionally from each angle group
- Logs coverage statistics per shard
- Falls back to random sharding if angle-balanced impossible

NLSQ Warm-Start Priors
^^^^^^^^^^^^^^^^^^^^^^

New functions in ``homodyne.optimization.cmc.priors`` for NLSQ-informed prior construction:

.. code-block:: python

   from homodyne.optimization.cmc.priors import (
       build_nlsq_informed_prior,
       build_nlsq_informed_priors,
       extract_nlsq_values_for_cmc,
   )

   # Extract NLSQ values from various result formats
   nlsq_values = extract_nlsq_values_for_cmc(nlsq_result)

   # Build informative prior for single parameter
   prior = build_nlsq_informed_prior(
       param_name="D0",
       nlsq_value=1234.5,
       nlsq_std=45.6,
       bounds=(100, 10000),
       width_factor=3.0,  # 3σ width
   )

   # Build priors for all physical parameters
   priors = build_nlsq_informed_priors(nlsq_values, nlsq_stds, bounds, analysis_mode)

**Usage in fit_mcmc_jax**:

.. code-block:: python

   from homodyne.optimization.cmc import fit_mcmc_jax
   from homodyne.optimization.nlsq import fit_nlsq_jax

   # Step 1: Run NLSQ
   nlsq_result = fit_nlsq_jax(data, config)

   # Step 2: Run CMC with NLSQ warm-start
   cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)

Constant-Averaged Per-Angle Mode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

New ``xpcs_model_constant_averaged()`` model for exact NLSQ "auto" mode parity:

- Uses FIXED averaged contrast/offset (not sampled)
- 8 parameters (7 physical + sigma) instead of 10
- Matches NLSQ constant mode averaging behavior

.. code-block:: yaml

   optimization:
     cmc:
       per_angle_mode: "constant_averaged"  # Match NLSQ "auto"

Early Abort Mechanism
^^^^^^^^^^^^^^^^^^^^^

The multiprocessing backend now tracks failure categories and aborts early:

.. list-table:: Failure Categories
   :header-rows: 1
   :widths: 25 75

   * - Category
     - Description
   * - ``timeout``
     - Shard exceeded ``per_shard_timeout``
   * - ``heartbeat_timeout``
     - Worker stopped responding
   * - ``crash``
     - Worker process crashed
   * - ``numerical``
     - NaN/Inf in posterior samples
   * - ``convergence``
     - High R-hat or low ESS

**Abort condition**: If >50% of first 10 shards fail, the run aborts immediately.

NUTS Convergence Improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For laminar_flow mode:

- ``target_accept_prob`` automatically elevated to 0.9 (from default 0.85)
- Divergence rate checking with severity levels:

  - >30%: CRITICAL (logged as error, run continues)
  - >10%: WARNING
  - >5%: ELEVATED (info)

Precision Diagnostics
^^^^^^^^^^^^^^^^^^^^^

New functions in ``homodyne.optimization.cmc.diagnostics``:

.. code-block:: python

   from homodyne.optimization.cmc.diagnostics import (
       compute_posterior_contraction,
       compute_nlsq_comparison_metrics,
       compute_precision_analysis,
       log_precision_analysis,
   )

   # Posterior Contraction Ratio: PCR = 1 - (posterior_std / prior_std)
   pcr = compute_posterior_contraction(posterior_std=10.0, prior_std=100.0)
   # pcr = 0.9 (90% contraction = informative data)

   # Compare CMC to NLSQ
   metrics = compute_nlsq_comparison_metrics(
       cmc_mean=1234.5,
       cmc_std=45.6,
       nlsq_value=1250.0,
       nlsq_std=50.0,
   )
   # Returns: z_score, uncertainty_ratio, overlap

Configuration Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: "auto"  # Angle-aware scaling
         strategy: "angle_balanced"    # Ensure coverage per shard
         min_angle_coverage: 0.8       # 80% of angles per shard
       sampler:
         target_accept_prob: 0.9       # Higher for laminar_flow
       execution:
         per_shard_timeout: 3600       # 1 hour (down from 2)
         early_abort_threshold: 0.5    # Abort if >50% of first 10 fail
       per_angle_mode: "constant_averaged"  # Match NLSQ "auto"

Shared Scaling Utilities
^^^^^^^^^^^^^^^^^^^^^^^^

CMC uses shared utilities from ``homodyne.core.scaling_utils`` for quantile-based
contrast/offset estimation:

.. automodule:: homodyne.core.scaling_utils
   :members: estimate_contrast_offset_from_quantiles, estimate_per_angle_scaling, compute_averaged_scaling
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

**Per-Angle Scaling** (mandatory):

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

Anti-Degeneracy Defense System
----------------------------------------

The NLSQ module includes a comprehensive anti-degeneracy defense system for laminar flow
analysis with many phi angles. See :doc:`/theory/anti_degeneracy_defense` for theoretical
background and usage tutorials.

Fourier Reparameterization (Layer 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reduces per-angle parameter count by expressing contrast/offset as Fourier series.

.. automodule:: homodyne.optimization.nlsq.fourier_reparam
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.fourier_reparam.FourierReparamConfig
   homodyne.optimization.nlsq.fourier_reparam.FourierReparameterizer

Hierarchical Optimization (Layer 2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternates between physical and per-angle parameter optimization to break gradient cancellation.

.. automodule:: homodyne.optimization.nlsq.hierarchical
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.hierarchical.HierarchicalConfig
   homodyne.optimization.nlsq.hierarchical.HierarchicalResult
   homodyne.optimization.nlsq.hierarchical.HierarchicalOptimizer

Adaptive Regularization (Layer 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CV-based regularization with automatic lambda tuning.

.. automodule:: homodyne.optimization.nlsq.adaptive_regularization
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.adaptive_regularization.AdaptiveRegularizationConfig
   homodyne.optimization.nlsq.adaptive_regularization.AdaptiveRegularizer

Gradient Collapse Monitor (Layer 4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runtime detection of gradient collapse with automatic response actions.

.. automodule:: homodyne.optimization.nlsq.gradient_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.gradient_monitor.GradientMonitorConfig
   homodyne.optimization.nlsq.gradient_monitor.CollapseEvent
   homodyne.optimization.nlsq.gradient_monitor.GradientCollapseMonitor

Shear-Sensitivity Weighting (Layer 5)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weights residuals by \|cos(φ₀-φ)\| to prevent gradient cancellation. Computed in
Homodyne and passed to NLSQ as generic residual weights.

.. automodule:: homodyne.optimization.nlsq.shear_weighting
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.shear_weighting.ShearWeightingConfig
   homodyne.optimization.nlsq.shear_weighting.ShearSensitivityWeighting

Anti-Degeneracy Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified controller that orchestrates all defense layers.

.. automodule:: homodyne.optimization.nlsq.anti_degeneracy_controller
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyConfig
   homodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController

NLSQ Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuration dataclasses and utilities for NLSQ optimization.

.. automodule:: homodyne.optimization.nlsq.config
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.config.NLSQConfig
   homodyne.optimization.nlsq.config.safe_float
   homodyne.optimization.nlsq.config.safe_int
   homodyne.optimization.nlsq.config.safe_bool

Configuration Entry Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``NLSQConfig.from_yaml()`` as the single entry point for loading NLSQ configuration:

.. code-block:: python

   from homodyne.optimization.nlsq.config import NLSQConfig

   # Load configuration from YAML file
   config = NLSQConfig.from_yaml("config.yaml")

   # Access configuration values
   print(f"Tolerance: {config.tolerance}")
   print(f"Max iterations: {config.max_iterations}")

Configuration Utilities (Deprecated)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. deprecated:: 2.14.0
   Use ``homodyne.optimization.nlsq.config`` instead. The ``safe_float``, ``safe_int``,
   and ``safe_bool`` utilities have been moved to ``config.py``.

.. automodule:: homodyne.optimization.nlsq.config_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. _nlsq-adapter-base:

NLSQAdapterBase
~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract base class providing shared functionality for NLSQAdapter and NLSQWrapper.
This enables code reuse and consistent interfaces across both adapter implementations.

.. automodule:: homodyne.optimization.nlsq.adapter_base
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.adapter_base.NLSQAdapterBase

Shared Methods
^^^^^^^^^^^^^^

The ``NLSQAdapterBase`` provides these common methods:

- ``_prepare_data()``: Flatten and validate input data
- ``_validate_input()``: Input validation with shape and type checking
- ``_build_result()``: Construct optimization result objects
- ``_handle_error()``: Error handling with recovery actions
- ``_setup_bounds()``: Bounds configuration and validation
- ``_compute_covariance()``: Covariance matrix computation from Jacobian

.. _nlsq-anti-degeneracy-layer:

.. _nlsq-validation:

Input and Result Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validation utilities extracted from wrapper.py for independent testing and reuse.

Input Validator
^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.validation.input_validator
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
"""""""""""""

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.validation.input_validator.InputValidator
   homodyne.optimization.nlsq.validation.input_validator.validate_array_dimensions
   homodyne.optimization.nlsq.validation.input_validator.validate_no_nan_inf
   homodyne.optimization.nlsq.validation.input_validator.validate_bounds_consistency
   homodyne.optimization.nlsq.validation.input_validator.validate_initial_params

Result Validator
^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.validation.result_validator
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
"""""""""""""

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.validation.result_validator.ResultValidator
   homodyne.optimization.nlsq.validation.result_validator.validate_optimized_params
   homodyne.optimization.nlsq.validation.result_validator.validate_covariance
   homodyne.optimization.nlsq.validation.result_validator.validate_result_consistency

Fit Quality Validator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Post-optimization quality checks with configurable thresholds.

.. automodule:: homodyne.optimization.nlsq.validation.fit_quality
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
"""""""""""

.. autosummary::
   :nosignatures:

   homodyne.optimization.nlsq.validation.fit_quality.FitQualityConfig
   homodyne.optimization.nlsq.validation.fit_quality.FitQualityReport
   homodyne.optimization.nlsq.validation.fit_quality.validate_fit_quality

Quality Checks
""""""""""""""

1. **Reduced χ² threshold**: Warns if χ²_reduced > threshold (default 10.0)
2. **CMA-ES convergence**: Warns if CMA-ES reached max_restarts without converging
3. **Physical parameters at bounds**: Warns if D₀, α, γ̇₀, etc. hit their bounds
4. **Convergence status**: Warns if optimization failed or hit max iterations

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.optimization.nlsq.validation import InputValidator, ResultValidator
   import numpy as np

   # Input validation
   validator = InputValidator(strict_mode=True)
   xdata = np.random.rand(1000, 3)
   ydata = np.random.rand(1000)
   initial = np.array([1000.0, 0.8, 100.0])
   bounds = (np.array([100, 0, 0]), np.array([10000, 2, 1000]))

   is_valid = validator.validate_all(xdata, ydata, initial, bounds)

   # Result validation
   result_validator = ResultValidator(strict_mode=False)
   optimized = np.array([1234.5, 0.85, 150.0])
   covariance = np.eye(3) * 0.01

   is_valid = result_validator.validate_all(optimized, covariance, bounds)

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

Batch Statistics
~~~~~~~~~~~~~~~~

Batch-level statistics tracking for streaming optimization.

.. automodule:: homodyne.optimization.batch_statistics
   :members:
   :undoc-members:
   :show-inheritance:

Numerical Validation
~~~~~~~~~~~~~~~~~~~~

Validation functions to detect numerical issues (NaN, Inf, bounds violations)
at critical points during optimization.

.. automodule:: homodyne.optimization.numerical_validation
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
