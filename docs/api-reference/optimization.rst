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

MCMC Bayesian inference using NumPyro's No-U-Turn Sampler (NUTS) for uncertainty quantification.

**Key Functions:**

* ``fit_mcmc_jax()`` - Main MCMC entry point
* ``MCMCResult`` - Result container with posterior samples

**Usage Example:**

.. code-block:: python

   from homodyne.optimization import fit_mcmc_jax
   import jax.numpy as jnp

   # MCMC typically initialized from NLSQ result
   nlsq_params = {
       'D0': 1000.0,
       'alpha': 0.8,
       'D_offset': 10.0
   }

   # Run MCMC with NUTS sampler
   mcmc_result = fit_mcmc_jax(
       t1, t2, phi, c2_exp,
       q=0.01,
       analysis_type='static_isotropic',
       initial_params=nlsq_params,
       num_warmup=500,
       num_samples=1000,
       num_chains=4
   )

   # Extract posterior statistics
   print(f"Mean parameters: {mcmc_result.mean_params}")
   print(f"Std parameters: {mcmc_result.std_params}")
   print(f"Credible intervals (95%): {mcmc_result.credible_intervals_95}")

   # Access full posterior samples
   samples = mcmc_result.samples  # Shape: (num_chains, num_samples, num_params)

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

Covariance Matrix Combination (CMC) for advanced multi-chain MCMC analysis.

**Key Components:**

* :mod:`homodyne.optimization.cmc.coordinator` - CMC orchestration
* :mod:`homodyne.optimization.cmc.combination` - Covariance combination
* :mod:`homodyne.optimization.cmc.svi_init` - SVI initialization
* :mod:`homodyne.optimization.cmc.diagnostics` - CMC diagnostics
* :mod:`homodyne.optimization.cmc.sharding` - Chain sharding
* :mod:`homodyne.optimization.cmc.backends` - Parallel execution backends

**Usage Example:**

.. code-block:: python

   from homodyne.optimization.cmc import CMCCoordinator

   # Create CMC coordinator
   coordinator = CMCCoordinator(
       num_chains=8,
       backend='multiprocessing'
   )

   # Run CMC analysis
   result = coordinator.run_cmc(
       t1, t2, phi, c2_exp,
       q=0.01,
       analysis_type='laminar_flow',
       initial_params=nlsq_params
   )

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
