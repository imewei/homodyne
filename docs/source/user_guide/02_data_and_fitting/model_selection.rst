.. _model_selection:

Choosing Your Model
===================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- How to decide between static and laminar_flow mode
- Memory and performance implications of each mode
- How to use quantitative criteria (angular dependence, chi-squared) for model selection
- Performance benchmarks for each mode

---

Overview
---------

Model selection in homodyne is primarily driven by **physics**: the model you
choose should reflect the physical dynamics present in your experiment. Using
an overly complex model (e.g., laminar_flow on equilibrium data) risks
overfitting and spurious parameter values.

This section provides a systematic decision process.

---

Decision Guide
--------------

**Step 1: Know your experimental setup**

Answer these questions:

- Is the sample in a shear cell (Couette, cone-plate)?
- Is there applied flow or pressure-driven flow?
- Does the scattering show an angular dependence?

If the answers are all "no" → use ``static`` mode.
If any answer is "yes" → consider ``laminar_flow`` mode.

**Step 2: Inspect the raw C2 data**

Before fitting, visually inspect the correlation data:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from homodyne.data import load_xpcs_data

   data = load_xpcs_data("config.yaml")
   c2 = data['c2_exp']          # (n_phi, n_t1, n_t2)
   phi = data['phi_angles_list']

   # Plot diagonal (lag-time dependence) for each angle
   fig, ax = plt.subplots()
   n_t = c2.shape[1]
   for i_phi, phi_val in enumerate(phi):
       # Extract anti-diagonal (fixed lag time)
       lag_idx = np.arange(n_t - 1)
       g2_lag = [c2[i_phi, k, k+1] for k in lag_idx]
       ax.plot(lag_idx, g2_lag, label=f"phi={phi_val:.0f}°")
   ax.set_xlabel("Lag index")
   ax.set_ylabel("C2")
   ax.legend()
   plt.show()

- If all curves overlap → **use static mode**
- If curves spread with :math:`\phi` → **use laminar_flow mode**

**Step 3: Quantitative criterion — angular variance test**

.. code-block:: python

   import numpy as np

   def angular_variance_ratio(c2, lag_index=1):
       """Ratio of angular variance to mean at given lag.
       Values > 0.05 suggest significant angular dependence."""
       c2_at_lag = c2[:, :-lag_index, lag_index:]  # shape (n_phi, ...)
       # Extract diagonal element at lag_index for each phi
       vals = np.array([c2[i, 0, lag_index] for i in range(c2.shape[0])])
       if vals.mean() == 0:
           return 0.0
       return vals.std() / vals.mean()

   ratio = angular_variance_ratio(data['c2_exp'], lag_index=5)
   print(f"Angular variance ratio: {ratio:.3f}")

   if ratio > 0.05:
       print("Significant angular dependence detected → consider laminar_flow mode")
   else:
       print("No angular dependence → use static mode")

**Step 4: Compare model fits**

If still uncertain, fit both modes and compare:

.. code-block:: python

   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax

   data = load_xpcs_data("config.yaml")

   # Fit static mode
   config_static = ConfigManager.from_yaml("config_static.yaml")
   result_static = fit_nlsq_jax(data, config_static)
   print(f"Static  chi2_nu: {result_static.reduced_chi_squared:.3f}")

   # Fit laminar flow mode
   config_flow = ConfigManager.from_yaml("config_flow.yaml")
   result_flow = fit_nlsq_jax(data, config_flow)
   print(f"Flow    chi2_nu: {result_flow.reduced_chi_squared:.3f}")

Use the **Akaike Information Criterion (AIC)** for model comparison:

.. code-block:: python

   def aic(chi_squared, n_params, n_data):
       """Akaike Information Criterion (lower is better)."""
       return chi_squared + 2 * n_params

   n_data = data['c2_exp'].size

   # 3 params for static (+ 2 if per-angle auto), 7 for flow (+ 2 if auto)
   aic_static = aic(result_static.chi_squared, n_params=5, n_data=n_data)
   aic_flow   = aic(result_flow.chi_squared,   n_params=9, n_data=n_data)

   print(f"AIC static: {aic_static:.1f}")
   print(f"AIC flow:   {aic_flow:.1f}")
   print(f"Preferred model: {'flow' if aic_flow < aic_static else 'static'}")

---

Memory Considerations
----------------------

The memory footprint scales with the number of data points and the model
complexity:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Mode
     - Parameters
     - Jacobian columns
     - Memory per million points
   * - ``static`` (constant)
     - 3
     - 3
     - ~0.3 GB
   * - ``static`` (auto)
     - 5
     - 5
     - ~0.5 GB
   * - ``laminar_flow`` (auto)
     - 9
     - 9
     - ~1.5 GB
   * - ``laminar_flow`` (individual, 23 angles)
     - 53
     - 53
     - ~9 GB

.. warning::

   ``individual`` per-angle mode with many angles creates very large Jacobians.
   For datasets > 1M points with > 10 angles, prefer ``auto`` mode or use
   streaming (enabled automatically when memory exceeds the threshold).

---

Performance Benchmarks
-----------------------

Approximate execution times on a 16-core CPU (JAX JIT-compiled):

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Dataset
     - Mode
     - Points
     - Time
   * - Small
     - static (auto)
     - 100K
     - ~1 s
   * - Medium
     - static (auto)
     - 1M
     - ~5 s
   * - Large
     - static (auto)
     - 10M
     - ~30 s (streaming)
   * - Small
     - laminar_flow (auto)
     - 100K
     - ~3 s
   * - Medium
     - laminar_flow (auto)
     - 1M
     - ~15 s
   * - Large
     - laminar_flow (auto)
     - 10M
     - ~90 s (streaming)

.. note::

   First-time execution includes JAX JIT compilation overhead (~10–60 s
   depending on model complexity). Subsequent calls with the same model
   structure are faster.

---

Anti-Degeneracy System
-----------------------

For ``laminar_flow`` mode, enabling the anti-degeneracy system is **critical**:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"

Without per-angle correction, the shear parameters (:math:`\dot\gamma_0`,
:math:`\beta`) can absorb the contrast and offset, producing physically
meaningless but numerically low :math:`\chi^2` fits.

The ``auto`` mode prevents this by:

1. Estimating per-angle contrast and offset from quantile analysis
2. Averaging them across angles to get single representative values
3. Optimizing these 2 averaged values alongside the 7 physical parameters

This adds only 2 parameters (total 9) while fully protecting against degeneracy.

---

See Also
---------

- :doc:`../01_fundamentals/analysis_modes` — Mode equations and parameters
- :doc:`nlsq_fitting` — Running NLSQ fits
- :doc:`result_interpretation` — Comparing fit quality
- :doc:`../03_advanced_topics/per_angle_modes` — Per-angle scaling options
- :doc:`../03_advanced_topics/laminar_flow` — Complete laminar flow guide
