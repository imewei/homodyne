Computational Methods
=====================

This section details the computational algorithms and implementation strategies
used in the Homodyne package for high-performance XPCS analysis.

Architecture Overview
---------------------

Homodyne employs a JAX-first architecture optimized for CPU-based high-performance
computing. The computational stack consists of:

1. **Primary Optimization**: NLSQ trust-region (Levenberg-Marquardt)
2. **Bayesian Inference**: NumPyro/BlackJAX MCMC (CMC-only in v2.4.1+)
3. **Core Kernels**: JAX JIT-compiled functions

JAX JIT Compilation
-------------------

Core Computational Kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~

The package leverages JAX JIT compilation for critical computational kernels.
All core functions are decorated with ``@jax.jit`` for optimal performance:

.. code-block:: python

   import jax
   import jax.numpy as jnp

   @jax.jit
   def compute_g2_scaled(
       params: jnp.ndarray,
       phi_angles: jnp.ndarray,
       time_matrix: jnp.ndarray,
       q_value: float,
       h_value: float,
   ) -> jnp.ndarray:
       """
       Compute g2 correlation with per-angle scaling.

       JAX JIT compilation provides:
       - XLA compilation to optimized machine code
       - Automatic vectorization
       - Efficient memory access patterns
       """
       # Extract per-angle scaling parameters
       n_angles = len(phi_angles)
       contrasts = params[:n_angles]
       offsets = params[n_angles:2*n_angles]
       physical_params = params[2*n_angles:]

       # Compute field correlation (vectorized)
       g1 = compute_g1_correlation(physical_params, phi_angles, time_matrix, q_value, h_value)

       # Apply Siegert relation with per-angle scaling
       g2 = offsets[:, None, None] + contrasts[:, None, None] * (g1 ** 2)

       return g2

Key JIT-Compiled Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The following core functions are JIT-compiled:

.. list-table:: JIT-Compiled Kernels
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Purpose
   * - ``compute_g2_scaled``
     - Full correlation computation with per-angle scaling
   * - ``compute_residuals``
     - Residual calculation for NLSQ optimization
   * - ``compute_chi_squared``
     - Fast chi-squared evaluation for objective function
   * - ``compute_diffusion_integral``
     - Analytical diffusion integral :math:`J(t_1, t_2)`
   * - ``compute_shear_integral``
     - Analytical shear integral :math:`\Gamma(t_1, t_2)`

Performance Benefits
~~~~~~~~~~~~~~~~~~~~

JAX JIT compilation provides significant performance advantages:

* **XLA Compilation**: Functions are compiled to optimized XLA (Accelerated Linear Algebra) code
* **Automatic Differentiation**: Gradients computed via ``jax.grad`` for optimization
* **Vectorization**: ``jax.vmap`` enables efficient batch processing
* **Memory Efficiency**: Optimized memory layouts and operation fusion

NLSQ Trust-Region Optimization
------------------------------

The primary optimization method uses non-linear least squares with trust-region
algorithms, specifically Levenberg-Marquardt.

Algorithm Description
~~~~~~~~~~~~~~~~~~~~~

The trust-region method iteratively solves:

.. math::

   \min_{\delta} \| \mathbf{r}(\boldsymbol{\theta}) + \mathbf{J} \delta \|^2
   \quad \text{subject to} \quad \|\delta\| \leq \Delta

where:

* :math:`\mathbf{r}(\boldsymbol{\theta})`: Residual vector
* :math:`\mathbf{J}`: Jacobian matrix
* :math:`\Delta`: Trust-region radius

Implementation
~~~~~~~~~~~~~~

Homodyne uses the ``nlsq`` library for trust-region optimization:

.. code-block:: python

   from nlsq import curve_fit_large

   def nlsq_optimization(
       model_func,
       experimental_data,
       initial_params,
       bounds,
   ):
       """
       NLSQ trust-region optimization with automatic strategy selection.

       Strategy selection:
       - STANDARD: Default for moderate datasets
       - LARGE: Memory-efficient for large datasets
       - CHUNKED: Block processing for very large data
       - STREAMING: Minimal memory footprint
       """
       popt, pcov = curve_fit_large(
           model_func,
           xdata=time_data,
           ydata=experimental_data,
           p0=initial_params,
           bounds=bounds,
           method='trf',  # Trust Region Reflective
       )
       return popt, pcov

Jacobian Computation
~~~~~~~~~~~~~~~~~~~~

The Jacobian matrix is computed using JAX automatic differentiation:

.. code-block:: python

   @jax.jit
   def compute_jacobian(params, data):
       """Compute Jacobian via JAX autodiff."""
       return jax.jacobian(compute_residuals)(params, data)

This provides exact gradients rather than finite-difference approximations,
improving convergence and numerical stability.

MCMC Inference
--------------

For Bayesian uncertainty quantification, Homodyne uses NumPyro with BlackJAX
samplers. Since v2.4.1, only Consensus Monte Carlo (CMC) is supported for
parallelized MCMC.

Consensus Monte Carlo (CMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMC partitions data across workers and combines posterior samples:

1. **Sharding**: Data partitioned across :math:`K` workers
2. **Parallel MCMC**: Each worker runs independent MCMC on its data partition
3. **Combination**: Posterior samples combined via weighted averaging

.. math::

   \hat{\boldsymbol{\theta}} = \sum_{k=1}^{K} \mathbf{W}_k \boldsymbol{\theta}_k

where :math:`\mathbf{W}_k` are precision-weighted combination matrices.

NumPyro Model Definition
~~~~~~~~~~~~~~~~~~~~~~~~

The probabilistic model is defined using NumPyro:

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist

   def xpcs_model(data, n_angles, mode="laminar_flow"):
       """NumPyro probabilistic model for XPCS analysis."""

       # Per-angle scaling priors
       for i in range(n_angles):
           numpyro.sample(f"contrast_{i}",
               dist.TruncatedNormal(0.5, 0.2, low=0.0, high=1.0))
           numpyro.sample(f"offset_{i}",
               dist.TruncatedNormal(1.0, 0.1, low=0.5, high=1.5))

       # Physical parameter priors
       D0 = numpyro.sample("D0",
           dist.TruncatedNormal(1000, 500, low=100, high=100000))
       alpha = numpyro.sample("alpha",
           dist.TruncatedNormal(0.0, 0.5, low=-2.0, high=2.0))
       # ... additional parameters

       # Likelihood
       predicted = compute_g2_model(params)
       numpyro.sample("obs", dist.Normal(predicted, sigma), obs=data)

BlackJAX Samplers
~~~~~~~~~~~~~~~~~

MCMC sampling uses BlackJAX with NUTS (No-U-Turn Sampler):

.. code-block:: python

   import blackjax

   def run_mcmc_sampling(
       model,
       data,
       num_samples=1000,
       num_warmup=500,
   ):
       """Run MCMC with NUTS sampler."""
       kernel = blackjax.nuts(model.potential_fn, step_size=0.01)
       samples = run_sampling(kernel, num_samples, num_warmup)
       return samples

Chi-Squared Objective Function
------------------------------

The objective function for optimization:

.. math::

   \chi^2(\boldsymbol{\theta}) = \sum_{i=1}^{N_\phi} \sum_{j,k} \frac{\left[c_2^{\text{exp}}(\phi_i, t_j, t_k) - c_2^{\text{model}}(\phi_i, t_j, t_k; \boldsymbol{\theta})\right]^2}{\sigma_{ijk}^2}

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   @jax.jit
   def compute_chi_squared(
       params: jnp.ndarray,
       experimental_data: jnp.ndarray,
       uncertainties: jnp.ndarray,
       model_args: tuple,
   ) -> float:
       """
       Compute chi-squared objective.

       Vectorized computation across all data points.
       """
       model_prediction = compute_g2_scaled(params, *model_args)
       residuals = (experimental_data - model_prediction) / uncertainties
       chi_squared = jnp.sum(residuals ** 2)
       return chi_squared

Numerical Stability
-------------------

Sinc Function Handling
~~~~~~~~~~~~~~~~~~~~~~

The sinc function requires special handling near zero:

.. code-block:: python

   @jax.jit
   def safe_sinc_squared(x: jnp.ndarray) -> jnp.ndarray:
       """
       Numerically stable sinc^2(x) computation.

       Uses Taylor expansion for |x| < threshold.
       """
       threshold = 1e-10

       # Taylor expansion: sinc(x) = 1 - (pi*x)^2/6 + O(x^4)
       small_x = jnp.abs(x) < threshold
       pi_x = jnp.pi * x
       pi_x_sq = pi_x ** 2

       taylor_result = (1.0 - pi_x_sq / 6.0) ** 2
       direct_result = (jnp.sin(pi_x) / (pi_x + 1e-100)) ** 2

       return jnp.where(small_x, taylor_result, direct_result)

Exponential Overflow Prevention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Large arguments to exponential functions are clamped:

.. code-block:: python

   MAX_EXP_ARG = 700.0  # Prevent overflow

   @jax.jit
   def safe_exp_decay(x: jnp.ndarray) -> jnp.ndarray:
       """Numerically stable exponential decay."""
       clamped_x = jnp.clip(x, -MAX_EXP_ARG, MAX_EXP_ARG)
       return jnp.exp(-clamped_x)

Memory Management
-----------------

Chunked Processing
~~~~~~~~~~~~~~~~~~

For large datasets, correlation matrices are processed in chunks:

.. code-block:: python

   def process_large_dataset(data, chunk_size=10000):
       """Process large dataset in memory-efficient chunks."""
       n_total = data.shape[0]
       results = []

       for i in range(0, n_total, chunk_size):
           chunk = data[i:i+chunk_size]
           result = process_chunk(chunk)
           results.append(result)

       return concatenate_results(results)

Memory-Efficient Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NLSQ optimizer automatically selects memory-efficient strategies:

.. list-table:: Optimization Strategies
   :header-rows: 1
   :widths: 20 20 60

   * - Strategy
     - Data Size
     - Description
   * - STANDARD
     - < 100K points
     - Full Jacobian in memory
   * - LARGE
     - 100K - 1M points
     - Sparse Jacobian representation
   * - CHUNKED
     - 1M - 10M points
     - Block-wise Jacobian computation
   * - STREAMING
     - > 10M points
     - Minimal memory footprint

Performance Benchmarks
----------------------

Typical performance on CPU (Intel Xeon, 8 cores):

.. list-table:: Performance Comparison
   :header-rows: 1
   :widths: 20 20 20 20

   * - Data Points
     - NLSQ Time
     - MCMC Time (1000 samples)
     - Speedup vs Pure Python
   * - 10,000
     - 0.5 s
     - 30 s
     - 10-20x
   * - 100,000
     - 2.5 s
     - 120 s
     - 15-25x
   * - 1,000,000
     - 15 s
     - 600 s
     - 20-30x

Performance scales well with data size due to vectorized JAX operations.

CPU Optimization
----------------

Multi-Core Utilization
~~~~~~~~~~~~~~~~~~~~~~

JAX automatically parallelizes operations across CPU cores:

.. code-block:: python

   # Set number of CPU threads
   import os
   os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"

   # JAX uses all available cores for matrix operations
   import jax
   jax.config.update("jax_platform_name", "cpu")

HPC Considerations
~~~~~~~~~~~~~~~~~~

For HPC environments:

* **Physical cores only**: Use physical cores, not hyperthreads, for MCMC
* **Memory allocation**: 5.5 GB per MCMC worker recommended
* **Affinity**: Pin processes to specific cores for cache efficiency

See the User Guide for HPC deployment patterns.

Debugging and Profiling
-----------------------

JAX Compilation Logging
~~~~~~~~~~~~~~~~~~~~~~~

Enable compilation logging for debugging:

.. code-block:: bash

   JAX_LOG_COMPILES=1 python script.py

This shows when functions are recompiled, helping identify performance issues.

Profiling
~~~~~~~~~

Profile JAX operations:

.. code-block:: python

   with jax.profiler.trace("/tmp/jax-trace"):
       result = compute_g2_scaled(params, *args)

   # View with: tensorboard --logdir=/tmp/jax-trace
