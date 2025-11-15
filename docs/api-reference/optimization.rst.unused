homodyne.optimization - Optimization Methods
============================================

.. automodule:: homodyne.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.optimization`` module provides state-of-the-art optimization methods for fitting XPCS correlation functions to experimental data. It implements both deterministic (NLSQ trust-region) and probabilistic (MCMC/NUTS) approaches with comprehensive error handling, checkpointing, and recovery strategies.

**Primary Methods:**

* **NLSQ** (Nonlinear Least Squares) - Fast deterministic optimization with automatic strategy selection
* **MCMC** (Markov Chain Monte Carlo) - Bayesian inference for uncertainty quantification via NumPyro/BlackJAX

Key Features
------------

* **Automatic Strategy Selection**: STANDARD → LARGE → CHUNKED → STREAMING based on dataset size
* **StreamingOptimizer**: Handle unlimited dataset sizes (>100M points) with constant memory
* **Checkpoint/Resume**: HDF5-based fault tolerance for long-running optimizations
* **Error Recovery**: 5 error-specific recovery strategies with adaptive retry
* **GPU Acceleration**: Transparent CUDA support via JAX
* **Progress Tracking**: Real-time optimization progress with batch statistics

Module Structure
----------------

The optimization module is organized into several submodules:

* :mod:`homodyne.optimization.nlsq_wrapper` - NLSQ optimization interface
* :mod:`homodyne.optimization.mcmc` - MCMC/NUTS Bayesian sampling
* :mod:`homodyne.optimization.strategy` - Intelligent strategy selection
* :mod:`homodyne.optimization.checkpoint_manager` - Checkpoint/resume system
* :mod:`homodyne.optimization.recovery_strategies` - Error recovery logic
* :mod:`homodyne.optimization.numerical_validation` - NaN/Inf validation
* :mod:`homodyne.optimization.batch_statistics` - Batch-level monitoring
* :mod:`homodyne.optimization.exceptions` - Custom exception hierarchy
* :mod:`homodyne.optimization.cmc` - Covariance Matrix Combination (advanced)
* :mod:`homodyne.optimization.stratified_chunking` - Angle-stratified data reorganization (v2.2+)
* :mod:`homodyne.optimization.sequential_angle` - Sequential per-angle optimization fallback (v2.2+)

Submodules
----------

homodyne.optimization.nlsq_wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq_wrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

NLSQ trust-region optimization with automatic strategy selection and streaming support.

**Key Functions:**

* ``fit_nlsq_jax()`` - Main NLSQ optimization entry point
* ``NLSQWrapper`` - Class-based wrapper for advanced usage

**Strategy Selection (Automatic):**

* **< 1M points** → STANDARD strategy (``curve_fit``)
* **1M-10M points** → LARGE strategy (``curve_fit_large``)
* **10M-100M points** → CHUNKED strategy (``curve_fit_large`` with progress)
* **> 100M points** → STREAMING strategy (unlimited data)

**Usage Example:**

.. code-block:: python

   from homodyne.optimization import fit_nlsq_jax
   import jax.numpy as jnp

   # Prepare data
   t1 = jnp.linspace(0, 1, 50)
   t2 = jnp.linspace(0, 1, 50)
   phi = jnp.array([0.0, 45.0, 90.0])
   c2_exp = jnp.ones((len(phi), len(t1), len(t2)))  # Experimental data

   # Initial parameters (static isotropic)
   initial_params = {
       'D0': 1000.0,
       'alpha': 0.8,
       'D_offset': 10.0
   }

   # Run NLSQ optimization
   result = fit_nlsq_jax(
       t1, t2, phi,
       c2_exp,
       q=0.01,
       analysis_type='static_isotropic',
       initial_params=initial_params
   )

   print(f"Optimal parameters: {result['parameters']}")
   print(f"Uncertainties: {result['uncertainties']}")
   print(f"Chi-squared: {result['chi_squared']}")

**Streaming Large Datasets:**

.. code-block:: python

   # For datasets > 100M points, enable streaming
   config = {
       'performance': {
           'strategy_override': 'streaming',  # Force streaming mode
           'memory_limit_gb': 16.0
       },
       'optimization': {
           'streaming': {
               'enable_checkpoints': True,
               'checkpoint_dir': './checkpoints',
               'checkpoint_frequency': 10,
               'resume_from_checkpoint': True,
               'keep_last_checkpoints': 3,
               'enable_fault_tolerance': True,
               'max_retries_per_batch': 2
           }
       }
   }

   result = fit_nlsq_jax(
       t1, t2, phi, c2_exp, q=0.01,
       analysis_type='laminar_flow',
       initial_params=initial_params,
       config=config
   )

homodyne.optimization.mcmc
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.mcmc
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

MCMC Bayesian inference with automatic NUTS/CMC selection for uncertainty quantification.

.. versionchanged:: 2.1.0
   Automatic NUTS/CMC selection replaces manual method parameter. Configuration-driven
   parameter management via ``parameter_space`` and ``initial_values``.

**Key Functions:**

* ``fit_mcmc_jax()`` - Main MCMC entry point with automatic method selection
* ``MCMCResult`` - Result container with posterior samples and diagnostics

**Automatic Selection (v2.1.0):**

MCMC automatically selects between NUTS (single-device) and CMC (multi-shard) based on:

* **Parallelism criterion**: ``num_samples >= 15`` → CMC for CPU parallelization
* **Memory criterion**: ``estimated_memory > 30%`` → CMC for OOM prevention
* **Decision logic**: CMC if **(Criterion 1 OR Criterion 2)**, otherwise NUTS

**Configuration-Driven Parameters (v2.1.0):**

.. code-block:: yaml

   # YAML configuration structure
   parameter_space:
     bounds:
       - name: D0
         min: 100.0
         max: 10000.0
       - name: alpha
         min: 0.1
         max: 2.0
       - name: D_offset
         min: 0.1
         max: 100.0
     priors:
       D0:
         type: TruncatedNormal
         mu: 1000.0
         sigma: 500.0
       alpha:
         type: TruncatedNormal
         mu: 1.0
         sigma: 0.3

   initial_parameters:
     parameter_names: [D0, alpha, D_offset]
     values: [1234.5, 0.567, 12.34]  # From NLSQ results

   optimization:
     mcmc:
       min_samples_for_cmc: 15       # Parallelism threshold
       memory_threshold_pct: 0.30    # Memory threshold (30%)
       dense_mass_matrix: false      # Diagonal vs full covariance

**Usage Example (v2.1.0):**

.. code-block:: python

   from homodyne.optimization import fit_mcmc_jax
   from homodyne.config.parameter_space import ParameterSpace
   from homodyne.config.manager import ConfigManager
   import numpy as np

   # Load configuration (recommended approach)
   config_manager = ConfigManager("config.yaml")
   parameter_space = ParameterSpace.from_config(config_manager.config)
   initial_values = config_manager.get_initial_parameters()

   # Run MCMC with automatic NUTS/CMC selection
   mcmc_result = fit_mcmc_jax(
       data=c2_exp,
       t1=t1,
       t2=t2,
       phi=phi,
       q=0.01,
       analysis_mode='static_isotropic',
       parameter_space=parameter_space,    # NEW in v2.1.0
       initial_values=initial_values,      # NEW in v2.1.0 (renamed from initial_params)
       n_warmup=500,
       n_samples=1000,
       n_chains=4
   )

   # Extract posterior statistics
   print(f"Method used: {mcmc_result.metadata.get('method_used')}")  # 'NUTS' or 'CMC'
   print(f"Mean parameters: {mcmc_result.mean_params}")
   print(f"Std parameters: {mcmc_result.std_params}")
   print(f"Credible intervals (95%): {mcmc_result.credible_intervals_95}")

   # Access full posterior samples
   samples = mcmc_result.samples  # Shape: (num_chains, num_samples, num_params)

**Manual NLSQ → MCMC Workflow (v2.1.0):**

.. code-block:: bash

   # Step 1: Run NLSQ optimization
   homodyne --config config.yaml --method nlsq

   # Step 2: Manually copy best-fit results to config.yaml
   # Edit initial_parameters.values: [D0_result, alpha_result, D_offset_result]

   # Step 3: Run MCMC with initialized parameters
   homodyne --config config.yaml --method mcmc

.. warning::
   **Breaking Changes in v2.1.0**

   * Removed ``method`` parameter (use automatic selection)
   * Renamed ``initial_params`` → ``initial_values``
   * Added required ``parameter_space`` parameter
   * No automatic NLSQ/SVI initialization (manual workflow required)

   See :doc:`../migration/v2.0-to-v2.1` for migration guide.

**Diagnostic Checks:**

.. code-block:: python

   # Check convergence diagnostics
   diagnostics = mcmc_result.diagnostics

   # R-hat (should be < 1.01 for convergence)
   print(f"R-hat: {diagnostics['r_hat']}")

   # Effective sample size (higher is better)
   print(f"ESS: {diagnostics['ess']}")

   # Divergences (should be 0)
   print(f"Divergences: {diagnostics['num_divergences']}")

homodyne.optimization.strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.strategy
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Intelligent strategy selection and memory estimation for NLSQ optimization.

**Key Functions:**

* ``select_optimization_strategy()`` - Automatic strategy selection
* ``estimate_memory_usage()`` - Memory footprint estimation
* ``build_streaming_config()`` - StreamingOptimizer configuration

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.strategy import select_optimization_strategy

   # Automatic strategy selection
   strategy = select_optimization_strategy(
       num_data_points=50_000_000,  # 50M points
       num_parameters=5,
       available_memory_gb=16.0
   )

   print(f"Selected strategy: {strategy}")  # Output: "chunked"

homodyne.optimization.checkpoint_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.checkpoint_manager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

HDF5-based checkpoint management for fault-tolerant long-running optimizations.

**Key Classes:**

* ``CheckpointManager`` - Checkpoint save/load/management

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.checkpoint_manager import CheckpointManager

   # Create checkpoint manager
   manager = CheckpointManager(
       checkpoint_dir='./checkpoints',
       keep_last_n=3
   )

   # Save checkpoint
   checkpoint_data = {
       'batch_idx': 42,
       'current_params': params,
       'batch_results': results
   }
   manager.save_checkpoint(checkpoint_data, batch_idx=42)

   # Resume from checkpoint
   if manager.has_checkpoints():
       checkpoint = manager.load_latest_checkpoint()
       batch_idx = checkpoint['batch_idx']
       params = checkpoint['current_params']

homodyne.optimization.recovery_strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.recovery_strategies
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Error-specific recovery strategies for robust optimization.

**Recovery Strategies:**

1. **OOMRecovery** - Out-of-memory errors (reduce batch size)
2. **ConvergenceRecovery** - Convergence failures (perturb parameters)
3. **BoundsRecovery** - Parameter bound violations (enforce bounds)
4. **NumericalRecovery** - NaN/Inf errors (validate inputs)
5. **UnknownRecovery** - Catch-all fallback

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.recovery_strategies import RecoveryStrategyFactory

   # Create recovery strategy for specific error
   factory = RecoveryStrategyFactory()
   strategy = factory.create_strategy('convergence')

   # Apply recovery
   recovered_params = strategy.recover(
       initial_params=params,
       error_info={'message': 'Failed to converge'},
       context={'bounds': bounds}
   )

homodyne.optimization.numerical_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.numerical_validation
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

NaN/Inf validation and numerical stability checks.

**Key Functions:**

* ``validate_numerical_stability()`` - Check for NaN/Inf in arrays
* ``sanitize_parameters()`` - Ensure parameter validity

homodyne.optimization.batch_statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.batch_statistics
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Batch-level monitoring and statistics for streaming optimization.

**Key Classes:**

* ``BatchStatistics`` - Track batch-level metrics

homodyne.optimization.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Custom exception hierarchy for optimization errors.

**Exception Hierarchy:**

* ``OptimizationError`` - Base exception

  * ``ConvergenceError`` - Convergence failures
  * ``NumericalError`` - NaN/Inf issues
  * ``MemoryError`` - Out-of-memory
  * ``ParameterError`` - Invalid parameters
  * ``CheckpointError`` - Checkpoint issues

homodyne.optimization.cmc
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Covariance Matrix Combination (CMC) for advanced multi-chain MCMC analysis with **dual-criteria automatic selection**.

**Automatic CMC Selection (October 2025 Optimization):**

CMC is automatically enabled when EITHER of these conditions is met:

1. **Sample-based parallelism**: ``num_samples >= 20`` (e.g., 20+ phi angles)

   * Optimized for multi-core CPU workloads (14+ cores)
   * Example: 23 samples on 14-core CPU → ~1.4x speedup via CMC
   * Sharding: Split samples (phi angles) across parallel shards

2. **Memory management**: ``estimated_memory > 40%`` of available memory

   * Conservative OOM prevention for large datasets
   * Example: 100M+ points → automatic data sharding
   * Sharding: Split data points across shards, keep all samples in each shard

**Hardware-Adaptive Backend Selection:**

* **Multi-node cluster (PBS/Slurm)**: Use cluster scheduler backend
* **Multi-GPU**: Use ``pjit`` backend for parallel GPU execution
* **Single GPU**: Use ``pjit`` backend with sequential shard execution
* **CPU-only**: Use ``multiprocessing`` backend

**Key Components:**

* :mod:`homodyne.optimization.cmc.coordinator` - CMC orchestration
* :mod:`homodyne.optimization.cmc.combination` - Covariance combination
* :mod:`homodyne.optimization.cmc.svi_init` - SVI initialization
* :mod:`homodyne.optimization.cmc.diagnostics` - CMC diagnostics
* :mod:`homodyne.optimization.cmc.sharding` - Chain sharding
* :mod:`homodyne.optimization.cmc.backends` - Parallel execution backends

**Usage Examples:**

.. code-block:: python

   from homodyne.optimization.mcmc import fit_mcmc_jax

   # Example 1: Many samples → CMC for parallelism (automatic)
   result = fit_mcmc_jax(
       t1, t2, phi,  # 23 angles → triggers CMC (>= 20)
       c2_exp,
       q=0.01,
       analysis_type='static_isotropic',
       initial_params={'D0': 1000.0, 'alpha': 0.8, 'D_offset': 10.0},
       method='auto'  # Automatic NUTS/CMC selection
   )

   # Example 2: Few samples, huge dataset → CMC for memory (automatic)
   result = fit_mcmc_jax(
       t1_large, t2_large, phi_small,  # 2 angles, 100M points → triggers CMC (memory > 40%)
       c2_exp_large,
       q=0.01,
       analysis_type='static_isotropic',
       initial_params=params,
       method='auto'
   )

   # Example 3: Manual CMC control (advanced)
   from homodyne.optimization.cmc import CMCCoordinator

   coordinator = CMCCoordinator(
       num_chains=8,
       backend='multiprocessing'  # or 'pjit', 'pbs', 'slurm'
   )

   result = coordinator.run_cmc(
       t1, t2, phi, c2_exp,
       q=0.01,
       analysis_type='laminar_flow',
       initial_params=nlsq_params
   )

**For More Details:**

See :doc:`../architecture/cmc-dual-mode-strategy` for comprehensive explanation of CMC design and decision logic.

homodyne.optimization.stratified_chunking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.stratified_chunking
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

.. versionadded:: 2.2.0
   Angle-stratified chunking to fix per-angle scaling compatibility with NLSQ chunking on large datasets.

**Critical Fix for Large Datasets (>1M points)**

This module solves silent NLSQ optimization failures caused by incompatibility between per-angle scaling parameters and arbitrary data chunking. When datasets exceed 1M points, NLSQ uses chunked processing where chunks may not contain all phi angles, resulting in zero gradients for per-angle parameters and silent optimization failures.

**Solution:** Reorganize data BEFORE optimization to ensure every chunk contains all phi angles, making gradients always well-defined.

**Key Functions:**

* ``reorganize_data_stratified()`` - Angle-stratified data reorganization (primary method)
* ``sequential_per_angle_optimization()`` - Sequential fallback for extreme angle imbalance
* ``StratificationDiagnostics`` - Performance monitoring and validation

**Automatic Activation (Recommended):**

Stratification activates automatically when:

* ``per_angle_scaling=True`` (default in v2.2.0)
* AND ``n_points >= 100,000``

**Performance:**

* Overhead: <1% (0.15s for 3M points)
* Scaling: O(n^1.01) sub-linear
* Memory: 2x peak (temporary during reorganization)

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.nlsq_wrapper import NLSQWrapper
   from homodyne.config.manager import ConfigManager

   # Load configuration
   config = ConfigManager("config.yaml")

   # Run NLSQ with automatic stratification
   wrapper = NLSQWrapper(config=config.to_dict())
   result = wrapper.optimize(
       data={'c2': c2_data, 't1': t1, 't2': t2, 'phi': phi},
       per_angle_scaling=True  # Triggers stratification if n >= 100k
   )

   # Check if stratification was used
   if result.diagnostics.get('stratification_applied'):
       print("✓ Stratification enabled")
       print(f"  Overhead: {result.diagnostics['stratification_time']:.2f}s")

**Manual Configuration:**

.. code-block:: yaml

   optimization:
     stratification:
       enabled: "auto"  # Options: true, false, "auto"
       target_chunk_size: 100000
       max_imbalance_ratio: 5.0  # Use sequential if max/min count > 5.0
       force_sequential_fallback: false
       check_memory_safety: true
       use_index_based: false  # Future: zero-copy optimization
       collect_diagnostics: false
       log_diagnostics: false

**Sequential Fallback (Extreme Imbalance):**

When angle imbalance ratio exceeds ``max_imbalance_ratio`` (default: 5.0), automatic fallback to sequential per-angle optimization:

.. code-block:: python

   # Automatic fallback for extreme imbalance
   # Dataset: 10k points for phi=0°, 1k for phi=60°, 500 for phi=120°
   # Imbalance ratio: 10000/500 = 20.0 > 5.0 → Sequential optimization

   result = wrapper.optimize(data, per_angle_scaling=True)
   # Uses sequential_per_angle_optimization() internally

**References:**

* Release Notes: :doc:`../releases/v2.2-stratification-release-notes`
* Investigation: :doc:`../troubleshooting/nlsq-zero-iterations-investigation`

homodyne.optimization.sequential_angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.sequential_angle
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

.. versionadded:: 2.2.0
   Sequential per-angle optimization fallback for extreme angle imbalance scenarios.

**Fallback Strategy for Extreme Angle Imbalance**

This module provides sequential per-angle optimization when stratification cannot be applied (e.g., angle count imbalance ratio > 5.0). Instead of chunking, it splits data by phi angle and optimizes each angle independently using ``scipy.optimize.least_squares``.

**Strategy:**

1. Split data by phi angle
2. Optimize each angle independently (per-angle contrast/offset + shared physical params)
3. Combine results using weighted averaging (inverse variance weighting)

**Use Cases:**

* Extreme angle imbalance (ratio > 5.0)
* Stratification explicitly disabled
* Memory-constrained environments
* Debugging and validation

**Key Functions:**

* ``sequential_per_angle_optimization()`` - Main sequential optimization
* ``combine_per_angle_results()`` - Weighted result combination

**Configuration:**

.. code-block:: yaml

   optimization:
     sequential:
       min_success_rate: 0.5  # Minimum fraction of angles that must converge
       weighting: "inverse_variance"  # Options: inverse_variance, uniform, n_points

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.sequential_angle import (
       sequential_per_angle_optimization,
       combine_per_angle_results
   )

   # Explicitly use sequential optimization
   angle_results = sequential_per_angle_optimization(
       data_dict=data,
       analysis_mode='static_isotropic',
       initial_params=initial_params,
       bounds=bounds,
       min_success_rate=0.5
   )

   # Combine results with inverse variance weighting
   combined = combine_per_angle_results(
       angle_results,
       weighting='inverse_variance'
   )

   print(f"Combined parameters: {combined['parameters']}")
   print(f"Combined uncertainties: {combined['uncertainties']}")
   print(f"Success rate: {combined['success_rate']:.1%}")

**Weighting Methods:**

* ``inverse_variance`` (recommended): Optimal statistical weighting (w_i = 1/σ²_i)
* ``uniform``: Equal weights for all angles
* ``n_points``: Weight by number of data points per angle

**Performance:**

* Execution: Sequential (no parallelization)
* Memory: Minimal (one angle at a time)
* Convergence: Dependent on per-angle data quality

Optimization Workflow
---------------------

**Recommended Workflow:**

1. **NLSQ Optimization** - Fast deterministic point estimate

   .. code-block:: python

      nlsq_result = fit_nlsq_jax(t1, t2, phi, c2_exp, q, 'static_isotropic', initial_params)

2. **MCMC Uncertainty Quantification** - Bayesian posterior sampling

   .. code-block:: python

      mcmc_result = fit_mcmc_jax(t1, t2, phi, c2_exp, q, 'static_isotropic', nlsq_result['parameters'])

3. **Result Analysis** - Combine deterministic and probabilistic insights

   .. code-block:: python

      # Use NLSQ for point estimate
      best_params = nlsq_result['parameters']

      # Use MCMC for uncertainties
      uncertainties = mcmc_result.std_params
      credible_intervals = mcmc_result.credible_intervals_95

Configuration
-------------

**YAML Configuration:**

.. code-block:: yaml

   performance:
     # Optional advanced settings
     strategy_override: null    # Force specific strategy: standard | large | chunked | streaming
     memory_limit_gb: null      # Custom memory limit (GB)
     enable_progress: true      # Show progress bars

   optimization:
     method: 'nlsq'              # Method: nlsq | mcmc
     streaming:
       enable_checkpoints: true
       checkpoint_dir: "./checkpoints"
       checkpoint_frequency: 10
       resume_from_checkpoint: true
       keep_last_checkpoints: 3
       enable_fault_tolerance: true
       max_retries_per_batch: 2
       min_success_rate: 0.5

   mcmc:
     num_warmup: 500
     num_samples: 1000
     num_chains: 4
     target_accept_prob: 0.8
     max_tree_depth: 10
     min_samples_for_cmc: 15        # NEW in v2.1.0
     memory_threshold_pct: 0.30     # NEW in v2.1.0
     dense_mass_matrix: false       # NEW in v2.1.0

Performance Considerations
--------------------------

**Memory Management**
   * STANDARD: < 1M points, ~100MB memory
   * LARGE: 1M-10M points, ~1-10GB memory
   * CHUNKED: 10M-100M points, constant memory (chunked processing)
   * STREAMING: > 100M points, constant memory + checkpoints

**GPU Acceleration**
   Both NLSQ and MCMC automatically use GPU when available. Check with:

   .. code-block:: python

      from homodyne.device import get_device_status
      print(get_device_status())

**Checkpoint Overhead**
   * Fast mode: < 1% overhead
   * Full fault tolerance: < 5% overhead
   * Checkpoint save: < 2 seconds

**MCMC Performance**
   * Warmup: ~1-2 min for 500 samples
   * Sampling: ~2-5 min for 1000 samples × 4 chains
   * GPU speedup: 5-10× vs CPU

Troubleshooting
---------------

**NLSQ Convergence Issues:**

* **Zero iterations** - Check model function and initial parameters
* **Parameter bounds** - Ensure bounds are physically reasonable
* **Numerical instability** - Use ``validate_numerical_stability()``

**MCMC Diagnostics:**

* **High R-hat (> 1.01)** - Increase warmup samples or num_chains
* **Low ESS (< 100)** - Increase num_samples or adjust step size
* **Divergences** - Lower target_accept_prob (e.g., 0.9) or increase max_tree_depth

**Memory Errors:**

* Use automatic strategy selection (don't override)
* Enable streaming for large datasets
* Reduce memory_limit_gb if needed

See Also
--------

* :doc:`../advanced-topics/nlsq-optimization` - Usage guide
* :doc:`../advanced-topics/streaming-optimization` - Large dataset handling
* :doc:`../advanced-topics/mcmc-uncertainty` - MCMC theoretical background
* :doc:`core` - Core physics engine used by optimization
* :doc:`../developer-guide/architecture` - Implementation details

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.optimization import (
       fit_nlsq_jax,
       fit_mcmc_jax,
       select_optimization_strategy,
       CheckpointManager,
   )

**Related Functions:**

* :func:`homodyne.core.jax_backend.compute_g2_scaled` - Used internally for residuals
* :func:`homodyne.data.xpcs_loader.XPCSDataLoader` - Load experimental data
* :func:`homodyne.viz.mcmc_plots.plot_posterior` - Visualize MCMC results
