NLSQ Optimization
=================

Overview
--------

NLSQ (Nonlinear Least Squares) is the primary optimization method in Homodyne v2+. It uses a trust-region Levenberg-Marquardt algorithm for robust and efficient fitting of correlation functions to experimental data.

**Key Features:**

- **Trust-Region Algorithm**: Adaptive trust region radius for stable convergence
- **Levenberg-Marquardt Method**: Hybrid of Steepest Descent and Gauss-Newton
- **Automatic Strategy Selection**: Chooses optimal algorithm based on dataset size
- **Error Recovery**: Automatic retry with parameter perturbation
- **Device-Agnostic**: Transparent CPU/GPU execution via JAX
- **Verified Accuracy**: 1.88-14.23% ground truth recovery error

When to Use NLSQ
----------------

NLSQ is the default optimization method for all analyses:

- **Primary method** for obtaining parameter point estimates
- **Fast**: Converges in 10-50 iterations for most datasets
- **Robust**: Automatic error recovery handles difficult fits
- **Scalable**: Handles datasets from 1K to 1B points

For uncertainty quantification, pair NLSQ with MCMC (see :doc:`mcmc-uncertainty`).

Trust-Region Algorithm
----------------------

Overview
~~~~~~~~

The trust-region Levenberg-Marquardt algorithm balances between:

1. **Steepest Descent**: Slow but stable when far from optimum
2. **Gauss-Newton**: Fast but unstable near optimum

**Algorithm Flow:**

.. code-block:: text

    Initialize parameters
    while not converged:
        1. Compute Jacobian at current parameters
        2. Solve normal equations with regularization
        3. Compute proposed step and trust region radius
        4. Evaluate improvement at proposed step
        5. Update trust region radius based on improvement ratio
        6. Accept or reject step
        7. Check convergence criteria


**Key Parameters:**

- **Trust Region Radius (μ)**: Constrains step size, adapts each iteration
- **Levenberg-Marquardt Parameter (λ)**: Regularization strength
- **Convergence Tolerance**: Gradient norm threshold for stopping

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

Homodyne uses the NLSQ package (`github.com/imewei/NLSQ`) which implements:

- **Automatic Jacobian Computation**: Via JAX autodiff
- **JIT Compilation**: Gradient computations compiled to machine code
- **Numerical Stability**: Cholesky decomposition with fallback
- **Bounds Enforcement**: Constrained least squares via active set method

Configuration Options
---------------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~~

The default NLSQ configuration works for most analyses:

.. code-block:: yaml

    optimization:
      method: "nlsq"
      nlsq:
        max_iterations: 100         # Maximum iterations
        tolerance: 1e-8             # Gradient norm convergence threshold
        trust_region_scale: 1.0     # Initial trust region radius scaling

**Recommendations:**

- **max_iterations**: 50-200 (default 100 is good for most cases)
- **tolerance**: 1e-6 to 1e-8 (smaller = better accuracy but slower)
- **trust_region_scale**: 0.5-2.0 (1.0 is standard, increase for aggressive steps)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For difficult fits or specific requirements:

.. code-block:: yaml

    optimization:
      nlsq:
        # Convergence control
        max_iterations: 150
        tolerance: 1e-10

        # Trust region adaptation
        trust_region_scale: 0.5     # Conservative (smaller steps)

        # Numerical settings (usually not needed)
        ftol: 1e-12                 # Function convergence
        xtol: 1e-12                 # Parameter convergence

**When to Adjust:**

- **Increase tolerance** if fit looks good but iterations exhaust
- **Decrease tolerance** if you need high-precision estimates
- **Adjust trust_region_scale** if convergence is too slow (increase) or unstable (decrease)

Strategy Selection
------------------

Automatic Strategy Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homodyne automatically selects the optimal NLSQ variant based on dataset size:

.. code-block:: text

    < 1M points         →  STANDARD (curve_fit)
    1M-10M points       →  LARGE (curve_fit_large)
    10M-100M points     →  CHUNKED (curve_fit_large with progress)
    > 100M points       →  STREAMING (StreamingOptimizer)

**Automatic Detection:**

.. code-block:: python

    from homodyne.optimization.strategy import estimate_memory_requirements

    # Homodyne automatically estimates dataset size and selects strategy
    strategy = estimate_memory_requirements(n_points, available_memory)
    # Returns: "standard" | "large" | "chunked" | "streaming"

Manual Override
~~~~~~~~~~~~~~~

For advanced use cases, force a specific strategy:

.. code-block:: yaml

    performance:
      strategy_override: "large"    # Force LARGE strategy
      memory_limit_gb: 8.0          # Custom memory limit

**Strategy Characteristics:**

.. list-table::
    :header-rows: 1
    :widths: 20 20 20 40

    * - Strategy
      - Dataset Size
      - Function Calls
      - Use Case
    * - STANDARD
      - < 1M
      - Full dataset in memory
      - Small analyses, quick testing
    * - LARGE
      - 1M-10M
      - Chunked with 1M batch size
      - Typical production analyses
    * - CHUNKED
      - 10M-100M
      - Large strategy with progress bars
      - Long-running fits
    * - STREAMING
      - > 100M
      - Batch optimization with checkpoint/resume
      - Very large datasets, fault tolerance

**Performance Implications:**

.. code-block:: text

    STANDARD:     1.0x baseline (reference)
    LARGE:        0.95-1.05x overhead for chunking
    CHUNKED:      1.0-1.1x overhead (progress tracking)
    STREAMING:    1.02-1.05x overhead (checkpoints)

See Also
~~~~~~~~

For large datasets with streaming, see :doc:`streaming-optimization`.

Angle-Stratified Chunking (v2.2.0+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Overview:**

For large datasets (>100k points) with per-angle scaling enabled, Homodyne automatically reorganizes data using angle-stratified chunking to ensure each optimization chunk contains all phi angles.

**Why Stratification is Needed:**

When using per-angle scaling (separate contrast/offset parameters for each angle), NLSQ's chunking must see all angles in each chunk to compute gradients correctly. Without stratification:

.. code-block:: text

    Problem: 3M points split into 11 chunks of 300k points each
    Issue: Each chunk may contain only 1-2 angles
    Result: Gradient w.r.t. angle-specific parameters = 0 for missing angles
    Outcome: Optimization returns unchanged parameters (0 iterations)

**Stratification Solution:**

.. code-block:: text

    Solution: Reorganize data so each chunk contains all 3 angles
    Example: 3M points, 3 angles, 100k chunk size
    → 30 chunks, each with 33.3k points per angle
    Result: Every chunk has complete angle coverage
    Outcome: NLSQ gradients computed correctly for all parameters

**Automatic Activation:**

Stratification activates automatically when **all** conditions are met:

1. ``per_angle_scaling=True`` (default in v2.2.0+)
2. ``n_points >= 100,000``
3. Angle distribution is balanced (imbalance ratio ≤ 5.0)

**Configuration:**

.. code-block:: yaml

    optimization:
      stratification:
        enabled: "auto"              # "auto" | true | false
        target_chunk_size: 100000    # Points per chunk (default 100k)
        max_imbalance_ratio: 5.0     # Balance threshold
        collect_diagnostics: false   # Enable diagnostics collection
        use_index_based: false       # Zero-copy mode (advanced)

**Configuration Options:**

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Parameter
      - Default
      - Description
    * - ``enabled``
      - ``"auto"``
      - "auto" activates when needed, ``true`` forces on, ``false`` disables
    * - ``target_chunk_size``
      - 100000
      - Target points per chunk (NLSQ processes chunks sequentially)
    * - ``max_imbalance_ratio``
      - 5.0
      - If imbalance > threshold, falls back to sequential per-angle optimization
    * - ``collect_diagnostics``
      - false
      - Collect detailed stratification metrics (adds ~1% overhead)
    * - ``use_index_based``
      - false
      - Zero-copy mode via index reordering (advanced, ~99% memory savings)

**Diagnostics Collection:**

When ``collect_diagnostics=true``, stratification metrics are saved in ``OptimizationResult``:

.. code-block:: python

    from homodyne.optimization.nlsq import fit_nlsq_jax

    result = fit_nlsq_jax(data, config)

    # Access stratification diagnostics
    if result.stratification_diagnostics is not None:
        diag = result.stratification_diagnostics
        print(f"Chunks created: {diag.n_chunks}")
        print(f"Chunk sizes: {diag.chunk_sizes}")
        print(f"Angles per chunk: {diag.angles_per_chunk}")
        print(f"Execution time: {diag.execution_time_ms:.2f} ms")
        print(f"Memory overhead: {diag.memory_overhead_mb:.1f} MB")
        print(f"Throughput: {diag.throughput_points_per_sec:.0f} pts/s")

**Diagnostics Attributes:**

.. list-table::
    :header-rows: 1
    :widths: 30 70

    * - Attribute
      - Description
    * - ``n_chunks``
      - Number of chunks created
    * - ``chunk_sizes``
      - List of chunk sizes (points per chunk)
    * - ``chunk_balance``
      - Statistics: {mean, std, min, max, cv}
    * - ``angles_per_chunk``
      - Number of unique angles in each chunk
    * - ``angle_coverage``
      - Coverage statistics: {mean, std, min_coverage_ratio}
    * - ``execution_time_ms``
      - Stratification execution time (milliseconds)
    * - ``memory_overhead_mb``
      - Peak memory overhead during stratification
    * - ``memory_efficiency``
      - Ratio of data size to peak memory (1.0 = perfect)
    * - ``throughput_points_per_sec``
      - Processing throughput (points/sec)
    * - ``use_index_based``
      - Whether index-based (zero-copy) mode was used

**Performance Characteristics:**

.. code-block:: text

    Overhead: < 1% (typically 50-200 ms for 1-10M points)
    Memory: 2x peak (full-copy) or 1.01x (index-based)
    Scaling: O(n^1.01) sub-linear (cache-friendly reorganization)

**When Stratification is Skipped:**

Stratification automatically falls back to sequential per-angle optimization when:

1. Dataset is small (< 100k points) → use STANDARD strategy
2. Per-angle scaling disabled (``per_angle_scaling=False``)
3. Extreme angle imbalance (ratio > ``max_imbalance_ratio``)
4. Single angle only (stratification unnecessary)
5. User disabled (``enabled: false``)

**Troubleshooting:**

If you see "Parameters unchanged after optimization" with large datasets:

1. **Check stratification status** in logs:

   .. code-block:: text

       INFO | Applying angle-stratified chunking: ...
       INFO | Stratification complete: X points reorganized

2. **Verify per-angle scaling** is needed:

   .. code-block:: yaml

       # Try disabling if not physically required
       per_angle_scaling: false

3. **Check angle balance**:

   .. code-block:: python

       from homodyne.optimization.stratified_chunking import analyze_angle_distribution
       stats = analyze_angle_distribution(data.phi)
       print(f"Imbalance ratio: {stats.imbalance_ratio:.2f}")
       print(f"Balanced: {stats.is_balanced}")

**See Also:**

- Log analysis guide: ``docs/troubleshooting/nlsq-zero-iterations-investigation.md``
- Known issues: ``CLAUDE.md`` Section "Known Issues → NLSQ Optimization"
- Stratification algorithm: ``homodyne/optimization/stratified_chunking.py``

Error Recovery
---------------

Automatic Retry Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

NLSQ includes automatic error recovery for common failure modes:

.. code-block:: yaml

    optimization:
      nlsq:
        max_retries: 3              # Retry attempts
        retry_perturbation: 0.1     # Perturbation magnitude

**Error Categories and Recovery:**

.. list-table::
    :header-rows: 1
    :widths: 25 35 40

    * - Error Type
      - Cause
      - Recovery Strategy
    * - Out of Memory (OOM)
      - Dataset too large for strategy
      - Fall back to simpler strategy
    * - Non-convergence
      - Poor initial parameters
      - Perturb and retry
    * - Bounds Violation
      - Parameters outside allowed range
      - Enforce bounds and retry
    * - Numerical (NaN/Inf)
      - Ill-conditioned Jacobian
      - Reduce step size, retry
    * - Unknown
      - Unexpected failure
      - Log and skip batch

Recovery in Action
~~~~~~~~~~~~~~~~~~~

Example: Out of Memory Error

.. code-block:: text

    Attempt 1: LARGE strategy → OOM error
    Attempt 2: Fall back to STANDARD strategy → Success!

    Result: Analysis completes using less aggressive chunking
    Log: Warning about strategy downgrade, final results valid

Best Parameter Tracking
~~~~~~~~~~~~~~~~~~~~~~~

Homodyne tracks the best parameters across all recovery attempts:

.. code-block:: python

    # Even if later batches fail, keep the best result
    best_result = {
        'parameters': [...],      # Best parameters found
        'chi_squared': 0.95,      # Chi-squared value
        'converged': True,        # Convergence status
        'iterations': 42,         # Iterations to convergence
        'recovery_actions': [...]  # List of recovery steps taken
    }

Output Interpretation
---------------------

Result Files
~~~~~~~~~~~~

After NLSQ optimization, Homodyne saves four main files:

.. code-block:: text

    output_dir/nlsq/
    ├── parameters.json           # Parameter estimates + uncertainties
    ├── fitted_data.npz           # Experimental vs. theoretical data
    ├── analysis_results_nlsq.json # Fit quality metrics
    └── convergence_metrics.json   # Convergence diagnostics

parameters.json
^^^^^^^^^^^^^^^

Contains estimated parameters and their covariance-derived uncertainties:

.. code-block:: json

    {
      "D0": {
        "value": 1234.5,
        "uncertainty": 45.2,
        "unit": "Å²/s"
      },
      "alpha": {
        "value": 0.85,
        "uncertainty": 0.05,
        "unit": "dimensionless"
      },
      "gamma_dot_0": {
        "value": 0.0125,
        "uncertainty": 0.0008,
        "unit": "s⁻¹"
      },
      "phi0": {
        "value": 15.2,
        "uncertainty": 2.3,
        "unit": "degrees"
      }
    }

**Interpreting Uncertainties:**

- Uncertainties are estimated from the covariance matrix at convergence
- Represent ~68% confidence interval (1-sigma)
- Smaller uncertainties indicate more constrained parameters
- Large uncertainties suggest the parameter is weakly determined by the data

fitted_data.npz
^^^^^^^^^^^^^^^

NumPy archive containing experimental and theoretical data:

.. code-block:: python

    import numpy as np

    data = np.load('fitted_data.npz')

    # Available arrays
    print(data.files)
    # Output: ['c2_exp', 'c2_theory', 'residuals', 'chi_squared_per_point', ...]

    # Access data
    c2_exp = data['c2_exp']           # Experimental c2
    c2_theory = data['c2_theory']     # Fitted theoretical c2
    residuals = data['residuals']     # Deviations

    # Plotting example
    import matplotlib.pyplot as plt

    for phi_idx in range(len(c2_exp)):
        plt.figure(figsize=(10, 6))
        tau = data['tau']
        plt.loglog(tau, c2_exp[phi_idx], 'o-', label='Experimental', alpha=0.7)
        plt.loglog(tau, c2_theory[phi_idx], 's-', label='Fitted', alpha=0.7)
        plt.xlabel('Delay time τ (s)')
        plt.ylabel('g2(τ)')
        plt.legend()
        plt.title(f'Phi angle {phi_idx}')
        plt.show()

analysis_results_nlsq.json
^^^^^^^^^^^^^^^^^^^^^^^^^^

Summary of fit quality and dataset information:

.. code-block:: json

    {
      "fit_quality": {
        "chi_squared": 0.98,
        "chi_squared_per_point": 0.00234,
        "r_squared": 0.9985,
        "residual_std": 0.0045
      },
      "convergence": {
        "iterations": 42,
        "converged": true,
        "final_gradient_norm": 1.2e-9,
        "strategy_used": "large"
      },
      "dataset_info": {
        "total_points": 4850000,
        "num_angles": 3,
        "points_per_angle": 1616667,
        "q_ranges": [0.001, 0.1],
        "time_ranges": [0.001, 100.0]
      },
      "parameter_space": {
        "model": "laminar_flow",
        "num_parameters": 13,
        "active_parameters": 7,
        "fixed_parameters": {}
      }
    }

**Key Metrics:**

- **χ² ≈ 1.0**: Good fit (theoretical and experimental variance match)
- **R² > 0.99**: Excellent fit
- **Converged = true**: Algorithm reached convergence criteria
- **gradient_norm < 1e-8**: Parameters at (numerical) optimum

convergence_metrics.json
^^^^^^^^^^^^^^^^^^^^^^^^

Detailed convergence diagnostics:

.. code-block:: json

    {
      "iteration_history": [
        {"iteration": 0, "chi_squared": 2.45, "gradient_norm": 0.85},
        {"iteration": 1, "chi_squared": 1.23, "gradient_norm": 0.42},
        {"iteration": 2, "chi_squared": 0.98, "gradient_norm": 0.01},
        {"iteration": 3, "chi_squared": 0.98, "gradient_norm": 0.0000012}
      ],
      "trust_region_history": [
        {"iteration": 0, "radius": 1.0},
        {"iteration": 1, "radius": 1.5},
        {"iteration": 2, "radius": 2.0},
        {"iteration": 3, "radius": 2.0}
      ],
      "recovery_actions": [],
      "final_status": "converged"
    }

Diagnostic Plots
~~~~~~~~~~~~~~~~

Visualizing convergence:

.. code-block:: python

    import json
    import matplotlib.pyplot as plt

    # Load convergence history
    with open('convergence_metrics.json') as f:
        metrics = json.load(f)

    history = metrics['iteration_history']
    iterations = [h['iteration'] for h in history]
    chi_squared = [h['chi_squared'] for h in history]
    gradient_norm = [h['gradient_norm'] for h in history]

    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(iterations, chi_squared, 'o-')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('χ² (log scale)')
    ax1.set_title('Objective Function Convergence')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(iterations, gradient_norm, 's-', color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm (log scale)')
    ax2.set_title('Gradient Norm Convergence')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Workflow Examples
-----------------

Basic Static Isotropic Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal example for static isotropic diffusion:

.. code-block:: yaml

    # config_static.yaml
    experimental_data:
      file_path: "./data/experiment.hdf"

    parameter_space:
      model: "static_isotropic"
      bounds:
        - name: D0
          min: 100.0
          max: 100000.0
        - name: alpha
          min: 0.0
          max: 2.0
        - name: D_offset
          min: -100.0
          max: 100.0

    initial_parameters:
      parameter_names: ["D0", "alpha", "D_offset"]

    optimization:
      method: "nlsq"
      nlsq:
        max_iterations: 100
        tolerance: 1e-8

Run the analysis:

.. code-block:: bash

    homodyne --config config_static.yaml --output-dir results_static

Expected runtime: 1-5 minutes (depending on dataset size)

Laminar Flow with Angle Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

More complex example with flow parameters and angle filtering:

.. code-block:: yaml

    # config_laminar.yaml
    experimental_data:
      file_path: "./data/experiment.hdf"

    parameter_space:
      model: "laminar_flow"
      bounds:
        - name: D0
          min: 100.0
          max: 100000.0
        - name: alpha
          min: 0.0
          max: 2.0
        - name: D_offset
          min: -100.0
          max: 100.0
        - name: gamma_dot_0
          min: 1e-6
          max: 0.5
        - name: beta
          min: 0.0
          max: 2.0
        - name: gamma_dot_offset
          min: -0.1
          max: 0.1
        - name: phi_0
          min: -180.0
          max: 180.0

    initial_parameters:
      parameter_names: ["D0", "alpha", "D_offset", "gamma_dot_0", "beta", "gamma_dot_offset", "phi_0"]
      active_parameters: ["D0", "alpha", "gamma_dot_0"]

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -10.0
          max_angle: 10.0
          description: "Parallel to flow"
        - min_angle: 85.0
          max_angle: 95.0
          description: "Perpendicular to flow"

    optimization:
      method: "nlsq"
      nlsq:
        max_iterations: 150
        tolerance: 1e-8

Run with angle filtering:

.. code-block:: bash

    homodyne --config config_laminar.yaml --output-dir results_laminar

Expected runtime: 3-10 minutes (depends on angle filtering impact)

Large Dataset with Strategy Override
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explicitly control strategy selection for large datasets:

.. code-block:: yaml

    # config_large.yaml
    experimental_data:
      file_path: "./data/large_dataset.hdf"  # 50M+ points

    # ... parameter_space and optimization settings ...

    performance:
      strategy_override: "chunked"   # Force CHUNKED strategy
      memory_limit_gb: 16.0          # 16 GB available
      enable_progress: true          # Show progress bars

Run large dataset:

.. code-block:: bash

    homodyne --config config_large.yaml --output-dir results_large

Homodyne automatically processes in batches with progress updates.

Troubleshooting
---------------

Non-Convergence
~~~~~~~~~~~~~~~

**Symptom**: Iterations exhaust without converging (gradient norm > 1e-6)

**Causes and Solutions:**

1. **Poor Initial Parameters**
   - Solution: Adjust initial guesses closer to expected values
   - Example: If D0 should be ~1000, don't start at 1

2. **Parameter Bounds Too Tight**
   - Solution: Widen bounds to allow more exploration
   - Check: Can the true value fit within bounds?

3. **Tolerance Too Strict**
   - Solution: Relax tolerance (e.g., 1e-6 instead of 1e-10)
   - Try: Gradually tighten tolerance after first successful fit

4. **Too Few Iterations**
   - Solution: Increase max_iterations
   - Try: 200-500 for difficult fits

Solution Process:

.. code-block:: yaml

    # Step 1: Relax requirements
    optimization:
      nlsq:
        max_iterations: 200
        tolerance: 1e-6              # Relax tolerance
        trust_region_scale: 0.5      # Conservative stepping

    # Step 2: After successful fit, refine
    # Adjust initial_parameters based on Step 1 results
    # Then re-run with tighter tolerance

Unstable Convergence
~~~~~~~~~~~~~~~~~~~~

**Symptom**: χ² oscillates, parameters jump between iterations

**Solutions:**

1. **Reduce Trust Region Scale**

   .. code-block:: yaml

       optimization:
         nlsq:
           trust_region_scale: 0.5  # Smaller steps

2. **Improve Parameter Bounds**
   - Bounds should reflect physical constraints
   - Avoid bounds that are too wide

3. **Check Data Quality**
   - Ensure experimental data is clean
   - Look for outliers or noise spikes

Poor Fit Quality (χ² >> 1)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Fitted curve doesn't match experimental data

**Diagnosis:**

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Load results
    data = np.load('fitted_data.npz')

    # Check individual angles
    for i in range(len(data['c2_exp'])):
        plt.figure(figsize=(10, 6))
        plt.loglog(data['c2_exp'][i], 'o', label='Experimental', alpha=0.5)
        plt.loglog(data['c2_theory'][i], '-', label='Fitted')
        plt.xlabel('Tau')
        plt.ylabel('g2')
        plt.legend()
        plt.title(f'Angle {i}')
        plt.show()

        # Check residuals
        residuals = data['residuals'][i]
        print(f"Angle {i}: Mean residual = {np.mean(residuals):.6f}, "
              f"Std = {np.std(residuals):.6f}")

**Common Causes:**

- **Wrong Physical Model**: Check if static vs. laminar is correct
- **Angle Filtering**: Filtered angles may not be sufficient
- **Data Quality**: Experimental data may have artifacts
- **Parameter Space**: Bounds may exclude true values

See Also
--------

- :doc:`mcmc-uncertainty` - Quantify uncertainties with MCMC
- :doc:`streaming-optimization` - Large datasets (> 100M points)
- :doc:`../api-reference/optimization` - Full optimization API
- :doc:`../user-guide/configuration` - Configuration system
- :doc:`../theoretical-framework/core-equations` - Theory behind g2 model

References
----------

**NLSQ Package:**
- GitHub: https://github.com/imewei/NLSQ
- Documentation: https://nlsq.readthedocs.io/en/latest/

**Levenberg-Marquardt Algorithm:**
- Levenberg, K. (1944). "A Method for the Solution of Certain Non-linear Problems in Least Squares"
- Marquardt, D.W. (1963). "An Algorithm for Least-Squares Estimation of Nonlinear Parameters"

**Trust-Region Methods:**
- Nocedal, J. & Wright, S.J. (2006). "Numerical Optimization" (2nd ed.)
- Dennis, J.E. & Schnabel, R.B. (1983). "Numerical Methods for Unconstrained Optimization and Nonlinear Equations"

**XPCS Theory:**

- He et al. (2024). "Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter", *PNAS* **121**(31), e2401162121. DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_
