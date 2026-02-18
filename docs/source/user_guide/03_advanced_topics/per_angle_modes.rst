.. _per_angle_modes:

Per-Angle Scaling Modes
========================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- Why per-angle corrections are needed in XPCS
- The four per-angle modes and their parameter counts
- When to use each mode
- How to configure per-angle modes for both NLSQ and CMC
- How to diagnose and fix degeneracy issues

---

Why Per-Angle Scaling Matters
-------------------------------

In a real XPCS experiment, the speckle contrast :math:`\beta(\phi)` and
incoherent background offset can vary with azimuthal angle :math:`\phi`. This
happens because:

- Different detector segments have different quantum efficiencies
- The coherent flux illuminates pixels differently across azimuthal positions
- Multiple scattering contributions can vary with scattering direction
- Beamstop shadows affect some angular ranges

Without correcting for these variations, the optimizer will absorb angular
anisotropies into the **physical parameters** (:math:`D_0`, :math:`\dot\gamma_0`),
producing biased and physically meaningless results.

The anti-degeneracy system in homodyne addresses this by introducing per-angle
contrast and offset corrections.

---

The Four Modes
--------------

Per-angle scaling is controlled by ``per_angle_mode`` in both the NLSQ and
CMC configuration sections.

auto (Recommended)
~~~~~~~~~~~~~~~~~~~

**What it does:**

When ``n_phi >= threshold`` (default: 3):

1. Estimates per-angle contrast and offset using quantile analysis
2. **Averages** the estimates across all angles
3. Optimizes the **2 averaged values** alongside physical parameters

When ``n_phi < threshold``:

- Falls back to ``individual`` mode

**Parameter count:** +2 (total: 5 for static, 9 for laminar_flow)

**When to use:** Always, unless you have a specific reason otherwise.

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"
     cmc:
       per_angle_mode: "auto"

The ``auto`` mode balances physical accuracy (+2 parameters) against
complexity (avoids the degeneracy of ``individual`` mode).

constant
~~~~~~~~~

**What it does:**

1. Estimates per-angle contrast and offset using quantile analysis
2. **Fixes** these values — they are not optimized

**Parameter count:** +0 (total: 3 for static, 7 for laminar_flow)

**When to use:**

- When you want the fastest possible fit
- When contrast/offset variations are known from calibration
- For initial exploration before enabling ``auto``

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "constant"

.. note::

   ``constant`` mode produces a **different fixed value per angle** based
   on quantile estimation. Unlike ``auto``, it does not average across angles.

individual
~~~~~~~~~~~

**What it does:**

Treats contrast and offset as **independent free parameters for each angle**.
Each angle gets its own :math:`\beta_i` and :math:`\text{offset}_i`, all
optimized simultaneously.

**Parameter count:** +2×n_phi (total: 3+2n_phi for static)

Example (static, 23 angles): 3 + 2×23 = 49 parameters

**When to use:**

- When angles are physically isolated (e.g., different sample regions)
- With very few angles (n_phi ≤ 2)
- For detailed angular analysis of optical effects

.. warning::

   ``individual`` mode with many angles creates a large parameter space.
   For n_phi ≥ 5, the optimizer may struggle with degeneracy between
   per-angle contrast/offset and physical parameters. Use ``auto`` instead.

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "individual"

fourier
~~~~~~~~

**What it does:**

Models the angular variation of contrast and offset as a **truncated Fourier
series** with K=2:

.. math::

   \beta(\phi) = a_0 + \sum_{k=1}^{K} \left[a_k \cos(k\phi) + b_k \sin(k\phi)\right]

**Parameter count:** +4K+2 = +10 for K=2 (total: 13 for static)

**When to use:**

- When angular variation is smooth and physically expected (e.g., beam path)
- When you want to regularize the angular dependence
- For experiments with strong azimuthal anisotropy

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "fourier"

---

Configuration Examples
-----------------------

**NLSQ + CMC with matching per_angle_mode:**

Always set the same ``per_angle_mode`` in both NLSQ and CMC to ensure
that the NLSQ warm-start priors match the CMC parameterization:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # NLSQ setting
     cmc:
       per_angle_mode: "auto"     # CMC setting (must match NLSQ)
       constant_scaling_threshold: 3   # Use averaged mode when n_phi >= 3

**Advanced: custom threshold for auto mode:**

.. code-block:: yaml

   optimization:
     cmc:
       per_angle_mode: "auto"
       constant_scaling_threshold: 5  # Switch to averaged mode only for n_phi >= 5

---

Parameter Count Summary
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Mode
     - Extra Params
     - static (23 angles)
     - laminar_flow (23 angles)
     - laminar_flow (1 angle)
   * - ``constant``
     - 0
     - 3
     - 7
     - 7
   * - ``auto``
     - 2
     - 5
     - 9
     - 7 (falls back to individual for n_phi < 3)
   * - ``individual``
     - 2×n_phi = 46
     - 49
     - 53
     - 9
   * - ``fourier`` (K=2)
     - 10
     - 13
     - 17
     - 17

---

Diagnosing Degeneracy
-----------------------

Signs that degeneracy is present (per-angle mode not working):

**1. Parameters at bounds:**

.. code-block:: python

   import numpy as np

   bounds_lower = config.get_parameter_bounds()[0]
   bounds_upper = config.get_parameter_bounds()[1]

   for i, (val, lo, hi) in enumerate(zip(
       result.parameters, bounds_lower, bounds_upper
   )):
       tol = 1e-4 * (hi - lo)
       if abs(val - lo) < tol or abs(val - hi) < tol:
           print(f"WARNING: param[{i}] = {val:.4g} at bound [{lo}, {hi}]")

**2. Suspiciously large D₀ with small contrast:**

If :math:`D_0` is much larger than physically expected AND the contrast is
near zero, this suggests degeneracy: the optimizer is absorbing the signal
into :math:`D_0` and zeroing out the contrast.

**3. Angular dependence of residuals:**

If residuals (model - data) show systematic angular patterns (all angles in
one quadrant have positive residuals), per-angle corrections are insufficient.
Switch from ``constant`` to ``auto`` mode.

**4. NLSQ vs CMC disagreement > 20%:**

Large discrepancies between NLSQ and CMC estimates often indicate degeneracy
that NLSQ resolved by finding a local minimum.

---

Fixing Degeneracy Issues
--------------------------

**Approach 1: Switch to ``auto`` mode (most effective)**

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # Was "constant" or not set

**Approach 2: Provide tight initial estimates**

If you know the contrast from calibration measurements:

.. code-block:: yaml

   parameter_space:
     contrast:
       initial: 0.15    # From beamline calibration
       bounds: [0.05, 0.4]

**Approach 3: Multi-start optimization**

Use Latin Hypercube Sampling to explore parameter space:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_multistart, MultiStartConfig

   result = fit_nlsq_multistart(
       data, config,
       ms_config=MultiStartConfig(n_starts=30, use_lhs=True),
   )

**Approach 4: CMC with informative priors**

CMC with informative priors from calibration prevents the optimizer from
exploring degenerate regions:

.. code-block:: yaml

   parameter_space:
     contrast:
       prior: "normal"
       prior_mean: 0.15
       prior_std: 0.05

---

See Also
---------

- :doc:`../01_fundamentals/analysis_modes` — Mode selection overview
- :doc:`laminar_flow` — Complete laminar flow guide
- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ configuration
- :doc:`bayesian_inference` — CMC with anti-degeneracy
