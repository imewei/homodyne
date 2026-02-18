.. _laminar_flow_guide:

Laminar Flow Analysis Guide
============================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The physical setup for laminar flow XPCS (Taylor-Couette geometry)
- How to configure homodyne for laminar flow experiments
- The anti-degeneracy system and why it matters
- Angular dependence diagnostics
- Common laminar flow fitting problems and solutions

---

Physical Setup: Taylor-Couette Geometry
-----------------------------------------

Laminar flow XPCS is most commonly performed in a **Taylor-Couette** (concentric
cylinder) shear cell. The sample fills the gap between a stationary outer
cylinder (stator) and a rotating inner cylinder (rotor).

The key geometric parameters:

- **Gap distance** :math:`h`: stator-rotor separation (µm). Configured as
  ``gap_distance`` in YAML (µm, converted to Å internally: 1 µm = 1×10⁴ Å).
- **Flow direction** :math:`\phi_0`: azimuthal angle of the flow velocity
  relative to the detector coordinate system.
- **Wavevector** :math:`q`: in the horizontal plane, magnitude in Å⁻¹.

The shear-induced dynamics appear in the XPCS signal because particles are
displaced by the shear flow during the correlation time. The displacement at
time lag :math:`\Delta t = t_2 - t_1` is:

.. math::

   \Delta x(\phi) = h \, \Gamma(t_1, t_2) \cos(\phi - \phi_0)

where :math:`\Gamma` is the accumulated shear strain. This displacement
introduces a sinc² factor in :math:`C_2`:

.. math::

   C_2 \propto \mathrm{sinc}^2\!\left(\frac{q h \, \Gamma \cos(\phi - \phi_0)}{2}\right)

The sinc² function creates zeros (nulls) at specific combinations of
:math:`q`, :math:`h`, :math:`\Gamma`, and :math:`\phi`, giving XPCS its
sensitivity to shear rate.

---

Configuring for Laminar Flow
------------------------------

Minimum required configuration:

.. code-block:: yaml

   data:
     file_path: "shear_data.h5"
     gap_distance: 500.0     # µm (e.g., 500 µm = 0.5 mm gap)
     q_value: 0.054          # Å⁻¹
     dt: 0.1                 # s (frame interval)

   analysis:
     mode: "laminar_flow"

   optimization:
     method: "nlsq"
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # Critical!

   parameter_space:
     D0:
       initial: 1.0
       bounds: [0.001, 1.0e5]
     alpha:
       initial: -0.5
       bounds: [-2.0, 1.0]
     D_offset:
       initial: 0.01
       bounds: [0.0, 1.0e3]
     gamma_dot_0:
       initial: 0.5          # Start near expected shear rate
       bounds: [1.0e-6, 1.0e4]
     beta:
       initial: 0.0
       bounds: [-2.0, 2.0]
     gamma_dot_offset:
       initial: 0.001
       bounds: [0.0, 1.0e3]
     phi_0:
       initial: 0.0
       bounds: [-180.0, 180.0]

See :doc:`../04_practical_guides/configuration` for the complete YAML schema.

**Generate a laminar flow template:**

.. code-block:: bash

   homodyne-config --mode laminar_flow --output config_flow.yaml
   # Edit the output file to set your q_value, gap_distance, and dt

---

Anti-Degeneracy System
-----------------------

The ``per_angle_mode: "auto"`` setting is the **most important** configuration
choice for laminar flow analysis. Without it, the optimizer can find degenerate
solutions where:

- :math:`\dot\gamma_0` becomes very large (unphysical)
- Contrast drops to near-zero to compensate
- :math:`\chi^2` appears low but the result is physically wrong

**How ``auto`` mode prevents degeneracy:**

1. Before optimization, quantile-based estimates of contrast and offset are
   computed for each angle separately
2. These estimates are averaged across all angles
3. The averaged contrast and averaged offset are added as 2 free parameters
   (total: 9 parameters for laminar_flow)
4. The optimizer adjusts these 2 averaged values, preventing the physical
   parameters from absorbing them

**Empirical validation:**

The ``auto`` mode has been validated on synthetic data with known ground truth.
It recovers:

- :math:`D_0` to within 2–5%
- :math:`\dot\gamma_0` to within 3–8%
- :math:`\phi_0` to within 1–3°

Without anti-degeneracy, :math:`D_0` errors can exceed 50%.

---

Angular Dependence Visualization
----------------------------------

Visualizing the angular dependence helps verify that the laminar flow model
is appropriate and that the fit is capturing the physics:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from homodyne.data import load_xpcs_data

   data = load_xpcs_data("config.yaml")
   c2 = data['c2_exp']       # (n_phi, n_t1, n_t2)
   phi = data['phi_angles_list']
   t1 = data['t1']
   t2 = data['t2']

   # Choose a fixed lag time
   lag = 5   # index into time axis
   fig, ax = plt.subplots(figsize=(8, 5))

   for i_phi, phi_val in enumerate(phi):
       # Extract one-lag diagonal of C2 for this angle
       n = min(c2.shape[1], c2.shape[2]) - lag
       c2_lag = np.array([c2[i_phi, k, k + lag] for k in range(n)])
       ax.plot(np.arange(n), c2_lag, label=f"{phi_val:.0f}°", alpha=0.7)

   ax.set_xlabel("Frame index")
   ax.set_ylabel(f"C2 at lag={lag}")
   ax.set_title("Angular dependence of C2 (expect spread for laminar flow)")
   ax.legend(ncol=3, fontsize=8)
   plt.tight_layout()
   plt.show()

**Expected patterns:**

- **No angular dependence** (curves overlap) → use static mode instead
- **Systematic spread** with :math:`\phi` → laminar flow present; check :math:`\phi_0`
- **Oscillatory pattern** (sinc nulls) → strong shear, clear signal

---

Interpreting Shear Parameters
-------------------------------

**gamma_dot_0 (reference shear rate):**

Compare to the **applied shear rate** from your rheometer:

.. code-block:: python

   applied_shear_rate = 10.0   # s⁻¹ (from rheometer setting)
   fitted_shear_rate = params_dict['gamma_dot_0']
   print(f"Applied:  {applied_shear_rate:.3f} s⁻¹")
   print(f"Fitted:   {fitted_shear_rate:.3f} s⁻¹")
   print(f"Ratio:    {fitted_shear_rate / applied_shear_rate:.2f}")

A ratio near 1.0 validates the measurement. Significant deviations may
indicate slip at the walls or non-Newtonian flow profiles.

**phi_0 (angular offset):**

Should be reproducible across measurements with the same shear cell
orientation. If it varies randomly, check:

1. That the flow direction is consistent between experiments
2. That the shear cell is properly aligned to the beam

**beta (shear rate time dependence):**

For steady Couette flow: :math:`\beta \approx 0`.
Significant :math:`|\beta| > 0.2` may indicate:

- Transient startup behavior in the measured time window
- Shear banding or wall slip at some timescales
- Data collected during a flow ramp

---

Troubleshooting Laminar Flow Fits
-----------------------------------

**Problem: gamma_dot_0 converges to near zero**

Possible causes:

1. The shear rate is too small to produce visible sinc oscillations
   (check: :math:`q h \dot\gamma \Delta t_\text{max} \ll 1`)
2. phi_0 is far from correct — the sinc cos(phi - phi_0) factor cancels
   the angular dependence at all angles

Fix: provide a tighter initial estimate for ``phi_0`` and ``gamma_dot_0``.

.. code-block:: python

   # Estimate expected sinc argument at maximum lag time
   q = 0.054       # Å⁻¹
   h = 5000.0      # Å (0.5 µm gap)
   gamma_dot = 1.0  # s⁻¹ (expected)
   t_max = 10.0    # s (max lag)

   sinc_arg = 0.5 * q * h * gamma_dot * t_max
   print(f"Max sinc argument: {sinc_arg:.2f}")
   # Values < 0.5 → weak shear signal; values 1-3 → clear oscillations

**Problem: phi_0 oscillates between +90° and -90°**

The cos factor in the sinc argument is periodic, so :math:`\phi_0` and
:math:`\phi_0 + 180°` can give similar :math:`\chi^2`. This is a true
degeneracy. Fix: constrain phi_0 to a smaller range based on known geometry:

.. code-block:: yaml

   parameter_space:
     phi_0:
       initial: 0.0
       bounds: [-90.0, 90.0]   # Constrained by geometry

**Problem: Very slow convergence**

Laminar flow has a more complex loss landscape than static mode. Options:

1. Enable multi-start NLSQ with LHS sampling
2. Enable CMA-ES for the first pass (see :doc:`cmaes_optimization`)
3. Increase NLSQ max iterations:

.. code-block:: yaml

   optimization:
     nlsq:
       max_nfev: 5000   # Increase from default 1000

**Problem: Physical parameters unphysical (D0 >> expected)**

Enable anti-degeneracy:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # Was "constant" or missing

---

See Also
---------

- :doc:`../01_fundamentals/analysis_modes` — Laminar flow mode parameters
- :doc:`../01_fundamentals/parameter_guide` — Shear parameter interpretation
- :doc:`per_angle_modes` — Anti-degeneracy in depth
- :doc:`cmaes_optimization` — CMA-ES for multi-scale problems
- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ fitting workflow
