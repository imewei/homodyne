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

Backend Architecture: NLSQ vs CMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homodyne uses two separate physics backends with intentionally different computation modes.
This separation (introduced Nov 2025) prevents NLSQ from being affected by CMC-specific
memory management complexity.

.. list-table:: Backend Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - NLSQ Backend (``jax_backend.py``)
     - CMC Backend (``physics_cmc.py``)
   * - Purpose
     - Deterministic point estimates
     - Bayesian uncertainty quantification
   * - Computation Mode
     - Meshgrid-only
     - Dual-mode (element-wise + meshgrid)
   * - Data Format
     - 3D matrices: (n_phi, n_t1, n_t2)
     - Flattened arrays OR 3D matrices
   * - Memory Pattern
     - O(N²) for N time points
     - O(N) for N measurements (element-wise)
   * - Typical Data Size
     - 100-1000 time points (~1M elements)
     - 4600+ measurements per shard (~23M pooled)

**NLSQ Backend (Meshgrid-Only)**

The NLSQ backend creates full 2D correlation matrices for all time pairs:

.. code-block:: python

   # NLSQ creates meshgrid for all (t1[i], t2[j]) combinations
   t1_grid, t2_grid = jnp.meshgrid(t1, t2, indexing="ij")
   # Result: g2 shape = (n_phi, n_t1, n_t2)

This is optimal for NLSQ because:

- Trust-region requires full Jacobian computation across all data points
- Diagonal correction removes autocorrelation peak from the NxN matrix
- Memory usage is predictable and manageable for typical XPCS data

**CMC Backend (Dual-Mode)**

The CMC backend detects data format and switches computation mode:

.. code-block:: python

   # Element-wise detection (threshold: 2000 elements)
   is_elementwise = t1.ndim == 1 and safe_len(t1) > 2000

   if is_elementwise:
       # CMC mode: compute for each (t1[i], t2[i], phi[i]) pair independently
       # Result: 1D array of shape (n_measurements,)
   else:
       # Meshgrid mode: same as NLSQ
       # Result: 3D array of shape (n_phi, n_t1, n_t2)

This dual-mode design prevents memory catastrophe:

- **Without element-wise mode**: CMC would create meshgrid from ~4600 elements per shard
- **Pooled for final inference**: ~23M elements
- **Full meshgrid**: 23M × 23M = 530 quadrillion elements = **35TB memory allocation!**
- **With element-wise mode**: Only 23M elements = ~180MB

**Shared Components (``physics_utils.py``)**

Common utilities consolidated to eliminate duplication:

- ``apply_diagonal_correction``: Removes autocorrelation peak from C₂ matrices
- ``safe_exp``, ``safe_sinc``: Numerically stable operations
- ``calculate_diffusion_coefficient``: Time-dependent D(t) = D₀ t^α + D_offset
- ``calculate_shear_rate``: Time-dependent γ̇(t) = γ̇₀ t^β + γ̇_offset
- ``create_time_integral_matrix``: Trapezoidal numerical integration

**When to Use Which Backend**

.. list-table:: Backend Selection Guide
   :header-rows: 1
   :widths: 30 70

   * - Use Case
     - Recommended Backend
   * - Quick parameter estimation
     - NLSQ (``homodyne fit --method nlsq``)
   * - Uncertainty quantification
     - CMC (``homodyne fit --method cmc``)
   * - Large datasets (>1M points)
     - CMC with sharding
   * - Publication-quality error bars
     - CMC with convergence diagnostics

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
   * - ``compute_chi_squared``
     - Fast chi-squared evaluation (residuals computed internally)
   * - ``compute_g1_diffusion``
     - Diffusion contribution to g1 correlation
   * - ``compute_g1_shear``
     - Shear contribution to g1 correlation

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
   def compute_jacobian(params, fixed_args):
       """Compute Jacobian via JAX autodiff."""
       return jax.jacobian(compute_chi_squared)(params, *fixed_args)

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

CMC Data Sharding Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The CMC module uses intelligent data sharding to enable parallel MCMC on large datasets.
A key design principle is that **100% of data is always used**—sharding partitions data
without any subsampling.

**Sharding Methods**

Two sharding strategies are available based on the dataset structure:

.. list-table:: Sharding Strategy Selection
   :header-rows: 1
   :widths: 25 35 40

   * - Strategy
     - When Used
     - Behavior
   * - Stratified
     - Multiple phi angles
     - Shards by angle, preserves angle structure
   * - Random
     - Single phi angle
     - Random partitioning of data points

**Stratified Sharding (Multi-Angle)**

For datasets with multiple phi angles, stratified sharding processes each angle separately:

.. code-block:: text

   Input: Dataset with 3 angles × 1M points/angle = 3M total
   Config: max_points_per_shard = 25,000
           max_shards_per_angle = 100

   ┌─────────────────────────────────────────────────────────────────┐
   │                    STRATIFIED SHARDING                         │
   │         (Multi-angle: processes each φ angle separately)       │
   └─────────────────────────────────────────────────────────────────┘

   For EACH angle:
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 1: Calculate required shards                            │
   │                                                               │
   │   required_shards = ⌈n_points / max_points_per_shard⌉        │
   │                   = ⌈1,000,000 / 25,000⌉ = 40 shards         │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 2: Check against max_shards_per_angle (100)             │
   │                                                               │
   │   40 ≤ 100? ✓ Yes → Use 40 shards as planned                 │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 3: Split data                                           │
   │                                                               │
   │   points_per_shard = 1,000,000 / 40 = 25,000                 │
   │   (Last shard gets any remainder)                            │
   └───────────────────────────────────────────────────────────────┘

   Result: 3 angles × 40 shards = 120 total shards
           Each shard: ~25,000 points
           Data used: 100%

**Handling Very Large Angles (Cap Behavior)**

When an angle has more data than ``max_shards_per_angle × max_points_per_shard``,
the shard size increases to fit all data:

.. code-block:: text

   Input: 100M points per angle
   Config: max_points_per_shard = 25,000
           max_shards_per_angle = 100

   ┌───────────────────────────────────────────────────────────────┐
   │ Step 1: Calculate required shards                            │
   │                                                               │
   │   required = ⌈100,000,000 / 25,000⌉ = 4,000 shards          │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 2: Check cap                                            │
   │                                                               │
   │   4,000 > 100? ✓ Yes → Cap at 100 shards                     │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 3: INCREASE shard size (never subsample!)               │
   │                                                               │
   │   effective_points = ⌈100,000,000 / 100⌉ = 1,000,000        │
   │   → Each shard holds 1M points instead of 25K                │
   └───────────────────────────────────────────────────────────────┘

   Result: 100 shards × 1M points = 100% data used
           No subsampling, just larger shards

**Random Sharding (Single-Angle)**

For single-phi-angle datasets, random sharding distributes data evenly:

.. code-block:: text

   Input: 50M total points (single φ angle)
   Config: max_points_per_shard = 100,000
           max_shards = 100

   ┌─────────────────────────────────────────────────────────────────┐
   │                      RANDOM SHARDING                           │
   │              (Single-angle: random partitioning)               │
   └─────────────────────────────────────────────────────────────────┘

   ┌───────────────────────────────────────────────────────────────┐
   │ Step 1: Calculate required shards                            │
   │                                                               │
   │   required = ⌈50,000,000 / 100,000⌉ = 500 shards            │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 2: Check cap                                            │
   │                                                               │
   │   500 > 100? ✓ Yes → Cap at 100 shards                       │
   └───────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌───────────────────────────────────────────────────────────────┐
   │ Step 3: INCREASE shard size                                  │
   │                                                               │
   │   effective_points = ⌈50,000,000 / 100⌉ = 500,000           │
   │   → Each shard holds 500K points                             │
   └───────────────────────────────────────────────────────────────┘

   Result: 100 shards × 500K points = 100% data used

**Default Configuration Parameters**

.. list-table:: Sharding Configuration Defaults
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``max_points_per_shard``
     - 25,000 (laminar_flow) |br| 100,000 (static)
     - Target points per shard for NUTS tractability
   * - ``max_shards_per_angle``
     - 100
     - Cap on shards per phi angle (stratified)
   * - ``max_shards``
     - 100
     - Cap on total shards (random)
   * - ``seed``
     - 42
     - Random seed for reproducible splitting

.. |br| raw:: html

   <br/>

**Optimal Points Per Shard**

The ``max_points_per_shard`` values are tuned for NUTS sampling efficiency:

.. list-table:: NUTS Complexity Scaling
   :header-rows: 1
   :widths: 20 20 30 30

   * - Analysis Mode
     - Parameters
     - Recommended max_points
     - Expected Time/Shard
   * - Static
     - 3 physical + 2×n_angles scaling
     - 100,000
     - ~20-40 min
   * - Laminar Flow
     - 7 physical + 2×n_angles scaling
     - 25,000
     - ~20-40 min

**Example Scenarios**

.. list-table:: Sharding Examples
   :header-rows: 1
   :widths: 30 15 25 15 15

   * - Dataset
     - Config
     - Shards
     - Points/Shard
     - Data Used
   * - 3 angles × 1M pts
     - 25K/shard
     - 3 × 40 = 120
     - ~25K
     - 100%
   * - 2 angles × 100M pts
     - 25K/shard, 100 cap
     - 2 × 100 = 200
     - ~1M
     - 100%
   * - 1 angle × 50M pts
     - 100K/shard, 100 cap
     - 100
     - ~500K
     - 100%

**Implementation Details**

The sharding algorithm is implemented in ``homodyne.optimization.cmc.data_prep``:

.. code-block:: python

   from homodyne.optimization.cmc.data_prep import (
       shard_data_stratified,  # Multi-angle datasets
       shard_data_random,      # Single-angle datasets
   )

   # Stratified sharding (automatic for multi-angle)
   shards = shard_data_stratified(
       prepared_data,
       max_points_per_shard=25000,
       max_shards_per_angle=100,
       seed=42,
   )

   # Random sharding (for single phi angle)
   shards = shard_data_random(
       prepared_data,
       max_points_per_shard=100000,
       max_shards=100,
       seed=42,
   )

The CMC entry point (``fit_mcmc_jax``) automatically selects the appropriate
sharding strategy based on the number of phi angles in the dataset.

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

Parameter Bound Enforcement
---------------------------

Both NLSQ and CMC enforce parameter bounds to ensure physically meaningful results
and numerical stability. This section documents the canonical bounds and enforcement
mechanisms across the codebase.

Canonical Physical Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

The primary source of truth for parameter bounds is ``homodyne.core.fitting.ParameterSpace``.
All optimization methods (NLSQ and CMC) read bounds from this class.

.. list-table:: Canonical Parameter Bounds
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Min
     - Max
     - Notes
   * - contrast
     - 0.0
     - 1.0
     - Physical range for visibility/coherence. **Cannot exceed 1.0.**
   * - offset
     - 0.5
     - 1.5
     - Baseline around 1.0 ± 50%
   * - D0
     - 1.0
     - 1,000,000
     - Diffusion coefficient [length²/time^(1+α)]
   * - alpha
     - -2.0
     - 2.0
     - Diffusion exponent (tighter for numerical stability)
   * - D_offset
     - -100,000
     - 100,000
     - Diffusion baseline correction
   * - gamma_dot_t0
     - 1e-5
     - 1.0
     - Shear rate [1/time^(1+β)]
   * - beta
     - -2.0
     - 2.0
     - Shear exponent (tighter for numerical stability)
   * - gamma_dot_t_offset
     - -1.0
     - 1.0
     - Shear rate baseline correction [s⁻¹]
   * - phi0
     - -30.0
     - 30.0
     - Flow direction angle offset [degrees]

**Important**: The alpha and beta bounds were tightened from (±10) to (±2) for
numerical stability. Extreme values like alpha=-10 cause numerical underflow in
``exp(-q² × D₀ × t^α × dt/2) → 0``.

Bound Enforcement Locations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bounds are enforced at multiple points in the codebase to ensure robustness:

.. list-table:: Bound Enforcement Mechanisms
   :header-rows: 1
   :widths: 30 25 45

   * - Location
     - Parameters
     - Enforcement Method
   * - ``core/fitting.py:484-485``
     - contrast, offset
     - ``np.clip()`` via ``parameter_space.contrast_bounds``
   * - ``cmc/priors.py:107,124``
     - contrast, offset
     - ``np.clip()`` for data-based initial estimates
   * - ``cmc/scaling.py``
     - ALL parameters
     - ``smooth_bound()`` via differentiable tanh transform
   * - ``cli/commands.py:2684-2685``
     - contrast, offset
     - ``np.clip(0.01, 1.0)``, ``np.clip(0.5, 1.5)``

CMC Shard Bound Enforcement
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each CMC shard uses the same model and parameter space, ensuring consistent bound
enforcement across all parallel workers:

1. **Initial Value Validation** (``cmc/priors.py:validate_initial_value_bounds``)

   - Clips out-of-bounds initial values to bounds ± 1% margin
   - Logs warning when clipping occurs
   - Handles NaN/Inf by resetting to bound midpoint

2. **Smooth Tanh Bounds** (``cmc/scaling.py:_smooth_bound``)

   Maps ℝ → (low, high) smoothly during MCMC sampling:

   .. math::

      \text{smooth\_bound}(x; \text{low}, \text{high}) = \text{mid} + \frac{\text{high} - \text{low}}{2} \times \tanh\left(\frac{x - \text{mid}}{\frac{\text{high} - \text{low}}{2}}\right)

   This ensures:

   - All gradients remain finite (differentiable everywhere)
   - Parameters stay within bounds during NUTS exploration
   - Smooth behavior near bound edges (no hard clipping)

3. **Per-Angle Contrast/Offset Estimation**

   When estimating per-angle scaling from data via least squares:

   .. code-block:: python

      # Enforce physical bounds after lstsq estimation
      contrast_i = np.clip(contrast_raw, 0.01, 1.0)  # Physical: (0, 1]
      offset_i = np.clip(offset_raw, 0.5, 1.5)      # Physical: around 1.0

   **Warning**: If lstsq produces contrast > 1.0, this indicates the physics model
   underestimates the C₂ variation—a symptom of model-data mismatch, not an error
   in bound enforcement.

NLSQ Bound Enforcement
~~~~~~~~~~~~~~~~~~~~~~

NLSQ uses the trust-region reflective algorithm which natively supports box constraints:

.. code-block:: python

   from nlsq import curve_fit_large

   # Bounds passed directly to optimizer
   popt, pcov = curve_fit_large(
       model_func,
       xdata=time_data,
       ydata=experimental_data,
       p0=initial_params,
       bounds=(lower_bounds, upper_bounds),  # Enforced by scipy
       method='trf',  # Trust Region Reflective
   )

The bounds are constructed from ``ParameterSpace``:

.. code-block:: python

   lower_bounds, upper_bounds = parameter_space.get_bounds_array()
   # Returns numpy arrays for all parameters in model order

Contrast Physical Constraint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contrast parameter has special physical significance:

- **Definition**: Represents visibility/coherence of the scattering pattern
- **Physical Formula**: ``β = (I_max - I_min) / (I_max + I_min)`` which is always in [0, 1]
- **Hard Upper Bound**: contrast > 1.0 is physically impossible

When per-angle least squares fitting produces contrast > 1.0, the system logs a warning:

.. code-block:: text

   WARNING: Angle 0 (φ=4.90°): lstsq contrast=1.2345 > 1.0 (unphysical).
   This indicates the physics model underestimates the C2 variation.
   Clipping to 1.0.

This warning indicates the fitted physics parameters (D0, alpha, etc.) don't fully
capture the dynamics in the data—a model quality issue separate from bound enforcement.

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

Data Flow: HDF5 to Cache to Analysis
------------------------------------

Understanding the data flow from raw HDF5 files through the caching system to
analysis is critical for debugging and extending the package.

Complete Data Pipeline
~~~~~~~~~~~~~~~~~~~~~~

The data flows through these stages:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                           HDF5 FILE                                     │
   │  C2 matrices from synchrotron (APS old or APS-U new format)            │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                   XPCSDataLoader._load_from_hdf()                       │
   │  - Detects format (APS old vs APS-U)                                    │
   │  - Loads C2 matrices and metadata                                       │
   │  - Applies frame slicing [start_frame:end_frame]                        │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │              XPCSDataLoader._calculate_time_arrays()                    │
   │                                                                         │
   │  time_1d = np.linspace(0, time_max, matrix_size)  ◀── STARTS FROM 0    │
   │                                                                         │
   │  Returns 1D array: time_1d = [0, dt, 2*dt, 3*dt, ..., (N-1)*dt]        │
   │  CMC uses 1D directly; NLSQ regenerates meshgrids as needed            │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                   XPCSDataLoader._save_to_cache()                       │
   │                                                                         │
   │  np.savez_compressed(cache_path,                                        │
   │      t1=time_1d,         # 1D array, starts from t=0                    │
   │      t2=time_1d,         # Same 1D array (backward compatibility)       │
   │      c2_exp=c2_exp,      # Shape: (n_phi, n_t1, n_t2)                   │
   │      phi_angles_list=...,                                               │
   │      wavevector_q_list=...,                                             │
   │      cache_metadata=...                                                 │
   │  )                                                                      │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                         CACHE FILE (.npz)                               │
   │                                                                         │
   │  Contains: t1, t2 (1D arrays), c2_exp (3D), metadata                    │
   │  Note: t1[0] = t2[0] = 0 (includes t=0 point)                           │
   │  Old 2D caches are auto-converted to 1D during loading                  │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                   XPCSDataLoader._load_from_cache()                     │
   │                                                                         │
   │  Only supports 1D array cache format (rejects old 2D meshgrid caches) │
   │  Returns 1D arrays: t1, t2 = [0, dt, 2*dt, ...]                        │
   │  CMC uses 1D directly; NLSQ regenerates meshgrids as needed            │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │              commands.py: _exclude_t0_from_analysis()                   │
   │                                                                         │
   │  CRITICAL: Removes t=0 to prevent D(t) singularity                      │
   │                                                                         │
   │  if t1.ndim == 2:                                                       │
   │      t1_sliced = t1[1:, 1:]    # Remove first row & column              │
   │      t2_sliced = t2[1:, 1:]                                             │
   │      c2_sliced = c2[:, 1:, 1:]                                          │
   │  else:                                                                  │
   │      t1_sliced = t1[1:]        # Remove first element                   │
   │      t2_sliced = t2[1:]                                                 │
   │      c2_sliced = c2[:, 1:, 1:]                                          │
   │                                                                         │
   │  Result: Analysis uses t ∈ {dt, 2*dt, 3*dt, ...}                        │
   └──────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      NLSQ / CMC Analysis                                │
   │                                                                         │
   │  Optimization runs on data excluding t=0                                │
   │  D(t) = D₀ × t^α + D_offset is finite for all t > 0                     │
   └─────────────────────────────────────────────────────────────────────────┘

Time Array Generation
~~~~~~~~~~~~~~~~~~~~~

The ``_calculate_time_arrays()`` method creates time arrays:

.. code-block:: python

   def _calculate_time_arrays(self, matrix_size: int) -> tuple[NDArray, NDArray]:
       """Calculate t1 and t2 time arrays as 2D meshgrids."""
       dt = self.analyzer_config.get("dt", 1.0)
       start_frame = self.analyzer_config.get("start_frame", 1)
       end_frame = self.analyzer_config.get("end_frame", matrix_size + start_frame - 1)

       # Create 1D time array from configuration
       time_max = dt * (end_frame - start_frame)
       time_1d = np.linspace(0, time_max, matrix_size)  # STARTS FROM 0

       # Create 2D meshgrids for correlation analysis
       t1_2d, t2_2d = np.meshgrid(time_1d, time_1d, indexing="ij")

       return t1_2d, t2_2d

**Example with concrete values**:

- ``matrix_size = 100`` (100×100 C₂ matrix)
- ``dt = 0.001`` seconds
- ``start_frame = 1``, ``end_frame = 100`` (default)

Result:

- ``time_max = 0.001 × 99 = 0.099`` seconds
- ``time_1d = np.linspace(0, 0.099, 100)``
- Step size = ``0.099 / 99 = 0.001 = dt``
- ``time_1d = [0, 0.001, 0.002, ..., 0.099]``

The t=0 Exclusion Fix
~~~~~~~~~~~~~~~~~~~~~

**Problem**: When the diffusion exponent α < 0 (common in subdiffusive systems):

.. math::

   D(t) = D_0 \times t^\alpha + D_{\text{offset}}

At t=0 with α < 0:

.. math::

   D(0) = D_0 \times 0^{-1.5} = D_0 \times \infty \rightarrow \text{numerical overflow}

**Example**: With α = -1.571 and D₀ = 16830:

- ``D(1e-10) ≈ 1.3 × 10²⁴`` (dominates cumulative integrals)
- This causes ``g₁ → 1`` (constant), making C₂ output constant
- MCMC shows 0% acceptance rate due to numerical instability

**Solution**: Exclude t=0 from analysis while preserving it for plotting:

.. code-block:: python

   def _exclude_t0_from_analysis(data: dict[str, Any]) -> dict[str, Any]:
       """Exclude t=0 (index 0) from time arrays and C2 data for analysis.

       This prevents the D(t=0) singularity when α < 0 (anomalous diffusion).
       The physics model D(t) = D₀ × t^α diverges at t=0 for negative α.

       Slicing strategy:
       - 2D meshgrid: t1[1:, 1:], t2[1:, 1:], c2[:, 1:, 1:]
       - 1D arrays: t1[1:], t2[1:], c2[:, 1:, 1:]

       The cache file preserves t=0 for plotting purposes.
       """
       # ... implementation

This design ensures:

1. **Cache preserves full data**: Including t=0 for complete visualization
2. **Analysis excludes t=0**: Preventing numerical singularities
3. **Both NLSQ and CMC benefit**: Applied before optimization branch

**Verification**:

After slicing with ``[1:, 1:]``:

- Original: ``t1_2d.shape = (100, 100)``, min value = 0
- Sliced: ``t1_sliced.shape = (99, 99)``, min value = dt

The dimensions correctly align:

.. code-block:: python

   t1_sliced = t1_2d[1:, 1:]     # Shape: (N-1, N-1)
   t2_sliced = t2_2d[1:, 1:]     # Shape: (N-1, N-1)
   c2_sliced = c2_exp[:, 1:, 1:] # Shape: (n_phi, N-1, N-1)

   # All dimensions match: ✓

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
