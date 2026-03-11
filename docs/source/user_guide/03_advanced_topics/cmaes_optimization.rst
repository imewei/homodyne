.. _cmaes_optimization:

CMA-ES for Multi-Scale Problems
=================================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- When CMA-ES global optimization is needed (scale ratio > 1000)
- How to configure and run CMA-ES
- The CMA-ES → NLSQ refinement pipeline
- Available presets

---

When to Use CMA-ES
-------------------

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) is a global
optimization algorithm that avoids getting trapped in local minima.
Use it when:

- Parameters span very different scales (e.g., :math:`D_0 \sim 10^4` and
  :math:`\dot\gamma_0 \sim 10^{-3}`, scale ratio > 10⁶)
- Standard NLSQ converges to clearly unphysical solutions
- Multi-start NLSQ gives inconsistent results across starts

CMA-ES is **slower** than NLSQ (many function evaluations) but explores
the parameter space globally before passing a good starting point to NLSQ
for local refinement.

.. note::

   CMA-ES requires the ``evosax`` package. Install it with:

   .. code-block:: bash

      uv add evosax

---

Configuration
--------------

Enable CMA-ES in your YAML:

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true
         preset: "cmaes-global"    # 200 generations (thorough)
         refine_with_nlsq: true    # Run NLSQ after CMA-ES

Available presets:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Preset
     - Generations
     - Use case
   * - ``"cmaes-fast"``
     - 50
     - Quick scan; rough starting point
   * - ``"cmaes-global"``
     - 200
     - Thorough global search (recommended)
   * - ``"cmaes-hpc"``
     - 500
     - HPC cluster runs; maximum coverage

---

Python API
-----------

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_cmaes, CMAESWrapperConfig

   # Configure CMA-ES
   cmaes_config = CMAESWrapperConfig(
       preset="cmaes-global",
       refine_with_nlsq=True,
       population_size=None,   # None = auto-select from preset
   )

   # Run CMA-ES + NLSQ pipeline
   result = fit_nlsq_cmaes(data, config, cmaes_config=cmaes_config)

   print(f"CMA-ES result: {result.cmaes_result.best_fitness:.4g}")
   print(f"NLSQ refined:  {result.reduced_chi_squared:.4f}")

Or use the lower-level API:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_with_cmaes, CMAESWrapper

   wrapper = CMAESWrapper(config=cmaes_config)
   cmaes_result = wrapper.optimize(data, config)

   print(f"Best parameters: {cmaes_result.parameters}")
   print(f"Best fitness:    {cmaes_result.best_fitness:.4g}")

---

CMA-ES → NLSQ Refinement Pipeline
------------------------------------

The recommended pipeline:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_cmaes

   # 1. Global search with CMA-ES (slow, finds global region)
   # 2. Local refinement with NLSQ (fast, finds precise minimum)
   # Both happen automatically when refine_with_nlsq=True

   result = fit_nlsq_cmaes(data, config)

   print(f"Convergence: {result.convergence_status}")
   print(f"Chi^2_nu:    {result.reduced_chi_squared:.4f}")

**When CMA-ES + NLSQ outperforms standard NLSQ:**

- Multi-scale laminar flow data (:math:`D_0/\dot\gamma_0 > 1000`)
- When NLSQ alone returns ``reduced_chi_squared > 5``
- When ``phi_0`` is unknown and the landscape is multi-modal

---

Warm-Start Optimization
-------------------------

By default, CMA-ES uses **NLSQ warm-start** (``nlsq_warmstart: true``).
This runs a quick NLSQ fit first, then uses the result as CMA-ES's starting
point with a reduced step size:

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true
         nlsq_warmstart: true           # Phase 1: run NLSQ first (default)
         sigma_warmstart: 0.05          # Reduced step size for local refinement
         warmstart_auto_skip: true      # Skip CMA-ES if NLSQ is good enough
         warmstart_skip_threshold: 5.0  # Skip if reduced chi2 < 5.0

**How it works:**

1. **Phase 1 (NLSQ)**: A quick NLSQ fit runs with the standard configuration.
2. **Auto-skip check**: If the NLSQ reduced chi-squared is below
   ``warmstart_skip_threshold`` (default 5.0), CMA-ES is skipped entirely.
   This avoids wasted computation when NLSQ already found a good solution.
3. **Phase 2 (CMA-ES)**: If not skipped, CMA-ES runs with ``sigma_warmstart``
   (0.05) instead of ``sigma`` (0.5), restricting the search to a local
   neighborhood around the NLSQ solution.
4. **Comparison**: The best result by chi-squared (NLSQ or CMA-ES) is kept.

**When to tune these settings:**

- **Increase** ``warmstart_skip_threshold`` (e.g., 10.0) if CMA-ES is
  rarely improving on NLSQ — saves computation time.
- **Decrease** ``warmstart_skip_threshold`` (e.g., 2.0) if you want CMA-ES
  to run more aggressively even when NLSQ is decent.
- **Increase** ``sigma_warmstart`` (e.g., 0.1) if CMA-ES needs a wider
  search around the NLSQ solution.
- Set ``nlsq_warmstart: false`` to skip Phase 1 and run CMA-ES as a
  pure global search (uses ``sigma: 0.5``).

---

Performance Considerations
----------------------------

CMA-ES is much slower than NLSQ due to many function evaluations:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Function Evaluations
     - Approximate Time (100K pts)
   * - NLSQ alone
     - ~50–200
     - 5–30 s
   * - CMA-ES (fast preset)
     - ~5,000
     - 5–15 min
   * - CMA-ES (global preset)
     - ~20,000
     - 20–60 min
   * - CMA-ES + NLSQ refinement
     - ~20,000 + 200
     - ~same as CMA-ES

Use CMA-ES only when standard NLSQ fails to converge adequately.

---

See Also
---------

- :doc:`../02_data_and_fitting/nlsq_fitting` — Standard NLSQ workflow
- :doc:`laminar_flow` — When multi-scale issues arise
- :doc:`../01_fundamentals/parameter_guide` — Parameter scale considerations
