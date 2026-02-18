.. _nlsq_fitting:

NLSQ Fitting Guide
==================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The complete NLSQ workflow: config → load → fit → results
- How to call ``fit_nlsq_jax`` and interpret the output
- The difference between ``NLSQAdapter`` and ``NLSQWrapper``
- How to diagnose convergence and identify fitting problems
- Common failure modes and their fixes

---

Overview
---------

NLSQ (Non-Linear Least Squares) is the **primary optimization method** in
homodyne. It uses a trust-region Levenberg-Marquardt algorithm to minimize the
weighted sum of squared residuals:

.. math::

   \chi^2 = \sum_{i} \frac{\left(C_2^\text{exp}(\phi_i, t_1^i, t_2^i) - C_2^\text{model}(\phi_i, t_1^i, t_2^i; \theta)\right)^2}{\sigma_i^2}

Key properties:

- **Fast**: seconds to minutes for typical datasets (10³–10⁷ data points)
- **JAX JIT-compiled**: model and Jacobian evaluated on CPU with XLA
- **Memory-adaptive**: automatically switches to streaming for large datasets
- **Robust**: 3-attempt retry with error recovery

---

Complete Workflow
-----------------

.. code-block:: python

   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.utils.logging import get_logger, log_phase

   logger = get_logger(__name__)

   # Step 1: Load configuration
   config = ConfigManager.from_yaml("config.yaml")

   # Step 2: Load experimental data
   data = load_xpcs_data("config.yaml")

   # Step 3: Run NLSQ optimization
   with log_phase("NLSQ Optimization"):
       result = fit_nlsq_jax(data, config)

   # Step 4: Inspect results
   if result.success:
       logger.info(f"Converged: chi^2 = {result.reduced_chi_squared:.3f}")
       for name, val in zip(result.param_names, result.parameters):
           err = result.uncertainties[result.param_names.index(name)]
           logger.info(f"  {name}: {val:.4g} ± {err:.4g}")
   else:
       logger.error(f"Failed: {result.message}")

---

The fit_nlsq_jax Function
---------------------------

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax

   result = fit_nlsq_jax(
       data,           # dict: XPCS data (see data_loading.rst)
       config,         # ConfigManager: loaded from YAML
       initial_params=None,   # dict | None: override initial parameters
       per_angle_scaling=True,  # bool: enable per-angle corrections (recommended)
       use_adapter=False,       # bool: use NLSQAdapter (experimental)
   )

**Parameters:**

- ``data``: Data dictionary from ``load_xpcs_data`` or ``XPCSDataLoader``.
  Accepts both CLI format (``phi_angles_list``, ``c2_exp``) and direct format
  (``phi``, ``g2``).

- ``config``: A ``ConfigManager`` instance loaded from a YAML configuration
  file. Contains all fitting settings, parameter bounds, and initial values.

- ``initial_params``: Optional override for initial parameter values.
  If ``None``, uses values from the config's ``parameter_space`` section.

- ``per_angle_scaling``: Should always be ``True`` (default). Per-angle
  contrast and offset corrections prevent parameter absorption degeneracy.

- ``use_adapter``: Set to ``True`` to use the ``NLSQAdapter`` (CurveFit
  class-based implementation). Default is ``False`` (uses ``NLSQWrapper``).

**Returns:** An ``OptimizationResult`` object.

---

Understanding the Result Object
---------------------------------

The ``OptimizationResult`` dataclass contains:

.. code-block:: python

   from homodyne.optimization.nlsq.results import OptimizationResult

   result: OptimizationResult = fit_nlsq_jax(data, config)

   # Core results
   result.parameters          # np.ndarray: fitted parameter values
   result.uncertainties       # np.ndarray: parameter standard deviations
   result.covariance          # np.ndarray: (n_params, n_params) covariance matrix

   # Fit quality
   result.chi_squared         # float: total chi^2
   result.reduced_chi_squared # float: chi^2 per degree of freedom
   result.convergence_status  # str: "converged", "max_iter", or "failed"
   result.quality_flag        # str: "good", "marginal", or "poor"

   # Diagnostics
   result.iterations          # int: number of iterations taken
   result.execution_time      # float: wall-clock seconds
   result.recovery_actions    # list[str]: error recovery steps taken

   # Convenience properties
   result.success             # bool: True if convergence_status == "converged"
   result.message             # str: human-readable outcome description

**Example: printing all fitted parameters:**

.. code-block:: python

   import numpy as np

   # Get parameter names from config
   param_names = result.param_names if hasattr(result, 'param_names') else []

   print(f"Convergence: {result.convergence_status}")
   print(f"Reduced chi-squared: {result.reduced_chi_squared:.4f}")
   print(f"Iterations: {result.iterations}")
   print(f"Time: {result.execution_time:.2f} s")
   print()
   print("Parameters:")
   for i, (val, err) in enumerate(zip(result.parameters, result.uncertainties)):
       print(f"  param[{i}]: {val:.4g} ± {err:.4g}")

---

NLSQAdapter vs NLSQWrapper
----------------------------

Homodyne provides two NLSQ implementations:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Implementation
     - ``NLSQWrapper`` (default)
     - ``NLSQAdapter`` (experimental)
   * - JIT caching
     - Per-call recompilation possible
     - Uses NLSQ ``CurveFit`` class for persistent JIT cache
   * - Error recovery
     - 3-attempt retry strategy
     - Delegates to NLSQ's internal recovery
   * - Streaming
     - Full support
     - Full support
   * - Use when
     - Production runs (default)
     - Repeated fits with same model structure

To use ``NLSQAdapter``:

.. code-block:: python

   result = fit_nlsq_jax(data, config, use_adapter=True)

.. note::

   ``use_adapter=True`` is experimental. Use the default (``False``) for
   production work until the adapter is fully validated.

---

Memory-Adaptive Strategy Selection
------------------------------------

Homodyne automatically selects the optimization strategy based on dataset size
and available system memory:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dataset Size
     - Strategy
     - Memory Usage
   * - < 10M points
     - Stratified LS (full-batch)
     - ~2 GB per million points
   * - > 10M points (or >75% RAM)
     - Hybrid Streaming
     - ~2 GB fixed overhead
   * - Scale ratio > 1000
     - CMA-ES + NLSQ refinement
     - Bounded

The strategy is selected automatically. You can tune the memory threshold:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.75      # Trigger streaming at 75% RAM usage
       # memory_threshold_gb: 48  # Or set explicit GB limit

---

Setting Initial Parameters
----------------------------

Provide custom initial parameters to guide the optimizer:

.. code-block:: python

   initial_params = {
       "D0": 5000.0,          # Å²/s
       "alpha": -0.3,
       "D_offset": 0.1,
       "gamma_dot_0": 0.5,    # s⁻¹ (laminar_flow only)
       "beta": 0.0,
       "gamma_dot_offset": 0.001,
       "phi_0": 0.0,
   }

   result = fit_nlsq_jax(data, config, initial_params=initial_params)

Or configure initial values in YAML:

.. code-block:: yaml

   parameter_space:
     D0:
       initial: 5000.0
       bounds: [1.0, 1.0e6]
     alpha:
       initial: -0.3
       bounds: [-2.0, 0.5]

---

Multi-Start Optimization
-------------------------

For difficult problems with multiple local minima, use multi-start NLSQ:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_multistart, MultiStartConfig

   ms_config = MultiStartConfig(
       n_starts=20,           # Number of random starting points
       use_lhs=True,          # Latin Hypercube Sampling for start coverage
       max_parallel=4,        # Parallel starts (CPU cores)
   )

   result = fit_nlsq_multistart(data, config, ms_config=ms_config)

   # The best result across all starts
   print(f"Best chi^2: {result.best_result.reduced_chi_squared:.4f}")
   print(f"Starts converged: {result.n_converged}/{result.n_starts}")

---

Convergence Diagnostics
-------------------------

**Reduced chi-squared:**

The reduced chi-squared :math:`\chi^2_\nu = \chi^2 / \nu` (where
:math:`\nu = n_\text{data} - n_\text{params}`) is the primary fit quality metric:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - :math:`\chi^2_\nu` value
     - Interpretation
   * - ~1.0
     - Good fit; uncertainties are well-estimated
   * - > 3
     - Poor fit; model may be wrong or outliers present
   * - > 10
     - Very poor fit; investigate data quality
   * - < 0.5
     - Over-fitting or uncertainties overestimated

**Jacobian condition number** (from ``nlsq_diagnostics``):

.. code-block:: python

   if result.nlsq_diagnostics:
       cond = result.nlsq_diagnostics.get('jacobian_condition', None)
       if cond and cond > 1e12:
           print("WARNING: ill-conditioned Jacobian (near-degenerate problem)")

**Parameter at bounds:**

If a fitted parameter is at its bound (lower or upper), the fit may be
constrained and the uncertainty estimate is unreliable:

.. code-block:: python

   for i, (val, lo, hi) in enumerate(zip(
       result.parameters, bounds_lower, bounds_upper
   )):
       if abs(val - lo) < 1e-6 * abs(hi - lo) or abs(val - hi) < 1e-6 * abs(hi - lo):
           print(f"Parameter {i} is at its bound: {val:.4g}")

---

Common Fitting Issues
----------------------

**Issue: ``convergence_status == "failed"``**

Possible causes and fixes:

1. Bad initial parameters → provide better initial values closer to the true solution
2. Bounds too tight → check that the true parameter values are within bounds
3. Wrong analysis mode → verify static vs laminar_flow is correct
4. Corrupted data → run ``validate_xpcs_data`` to check input

**Issue: ``reduced_chi_squared >> 1``**

The model does not fit the data:

1. Wrong mode selected (static vs laminar_flow)
2. Incorrect q-value or gap distance
3. Data contains outliers → inspect C2 heatmap
4. Wrong per-angle mode → try ``per_angle_mode: "auto"``

**Issue: Very long execution time**

1. Large dataset → check memory and streaming mode activation
2. JIT recompilation → use ``NLSQAdapter`` (``use_adapter=True``) for repeated fits
3. Many angles with ``individual`` per-angle mode → switch to ``auto``

**Issue: Parameters at bounds (physically unreasonable)**

1. Competing degeneracies → enable anti-degeneracy (``per_angle_mode: "auto"``)
2. Wrong mode for the data
3. Multi-modal likelihood → use multi-start or CMA-ES

---

See Also
---------

- :doc:`data_loading` — Loading input data
- :doc:`result_interpretation` — Reading and saving results
- :doc:`model_selection` — Choosing the right model
- :doc:`../03_advanced_topics/per_angle_modes` — Per-angle scaling in depth
- :doc:`../03_advanced_topics/cmaes_optimization` — CMA-ES for multi-scale problems
- :doc:`../03_advanced_topics/bayesian_inference` — CMC for uncertainty quantification
