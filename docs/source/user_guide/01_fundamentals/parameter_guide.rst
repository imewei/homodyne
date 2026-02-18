.. _parameter_guide:

Parameter Interpretation Guide
===============================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The physical meaning of every parameter in both analysis modes
- Typical ranges and units for real XPCS samples
- How to interpret anomalous exponents (:math:`\alpha`, :math:`\beta`)
- What the optical parameters (contrast, offset) tell you about your detector setup
- A reference table of parameter bounds and defaults

---

Overview
---------

Homodyne fits a total of 3–9 parameters depending on the analysis mode and
per-angle configuration. Parameters fall into three groups:

1. **Diffusion parameters**: describe translational dynamics (:math:`D_0, \alpha, D_\text{offset}`)
2. **Shear parameters**: describe flow-induced advection (laminar_flow only: :math:`\dot\gamma_0, \beta, \dot\gamma_\text{offset}, \phi_0`)
3. **Optical parameters**: describe detector/beam properties (contrast :math:`\beta`, offset)

---

Diffusion Parameters
---------------------

D₀ — Reference Diffusion Coefficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`D_0`
**Units:** Å²/s

:math:`D_0` is the amplitude of the diffusion coefficient at a reference time
:math:`t_\text{ref} = \sqrt{dt \cdot t_\text{max}}`. For normal diffusion
(:math:`\alpha = 0`), :math:`D(t) = D_0 + D_\text{offset}` is a constant.

.. warning::

   Homodyne's :math:`D_0` absorbs a factor of 2 from the transport coefficient
   convention: :math:`D_0 = 2 D_\text{SE}`, where :math:`D_\text{SE}` is the
   Stokes-Einstein diffusion coefficient:

   .. math::

      D_\text{SE} = \frac{k_B T}{6 \pi \eta R_h}, \qquad D_0 = 2 D_\text{SE}

   See :ref:`theory_transport_coefficient` for the derivation of this convention.

**Physical interpretation:**

- Larger :math:`D_0` → faster decay of :math:`C_2` at a given :math:`q`
- At a fixed :math:`q`, the characteristic relaxation time is
  :math:`\tau_q \sim (q^2 D_0)^{-1}`
- For a 10 nm sphere in water at 25 °C:
  :math:`D_\text{SE} \approx 2.4 \times 10^4` Å²/s, so
  :math:`D_0 \approx 4.8 \times 10^4` Å²/s

**Typical ranges:**

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - System
     - :math:`D_0` (Å²/s)
     - Notes
   * - 10 nm nanoparticles in water
     - ~5×10⁴
     - Stokes-Einstein (:math:`2 D_\text{SE}`)
   * - 100 nm particles in water
     - ~5×10³
     - —
   * - Concentrated dispersions
     - 10²–10³
     - Crowding reduces mobility
   * - Gels / soft glasses
     - 1–100
     - Caged motion
   * - Large aggregates / very slow
     - 0.01–10
     - Sub-diffusive

**Default bounds:** ``[0.0, 1.0e6]`` Å²/s

.. note::

   :math:`D_0` is internally reparameterized in CMC as
   :math:`\log(D_0 / t_\text{ref})` to improve sampling efficiency across
   multiple orders of magnitude. The reported value is always in Å²/s.

alpha — Diffusion Time-Dependence Exponent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\alpha`
**Units:** dimensionless

:math:`\alpha` controls how the effective diffusion coefficient scales with time.
Homodyne models:

.. math::

   D(t) = D_0 \, t^\alpha + D_\text{offset}

so :math:`\alpha` is the power-law exponent of the time-dependent part. The
diffusion integral :math:`\int_{t_1}^{t_2} J(t')\,dt'` that enters the
correlation function is evaluated numerically from :math:`D(t)` (see
:ref:`theory_transport_coefficient`).

**Physical interpretation:**

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Value
     - Classification
     - Physical picture
   * - :math:`\alpha = 0`
     - Normal diffusion
     - Brownian motion; :math:`D(t) = D_0` constant; MSD ∝ t
   * - :math:`\alpha < 0`
     - Sub-diffusion
     - Caged motion, viscoelastic medium, aging glass; :math:`D(t)` decreases
   * - :math:`\alpha > 0`
     - Super-diffusion
     - Active particles, driven systems; :math:`D(t)` increases

.. warning::

   :math:`\alpha < -1` is unphysical (the diffusion integral can become
   negative). The default lower bound is ``-2.0`` to allow exploration of
   highly sub-diffusive regimes; physical systems should not reach this limit.

**Default bounds:** ``[-2.0, 2.0]``

D_offset — Baseline Diffusion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`D_\text{offset}`
**Units:** Å²/s

:math:`D_\text{offset}` adds a constant, time-independent component to the
diffusion coefficient:

.. math::

   D(t) = D_0\,t^\alpha + D_\text{offset}

The :math:`D_\text{offset}` term contributes a linear-in-time growth to the
diffusion integral, independent of :math:`\alpha`.

**Physical interpretation:**

- Represents a fast-diffusing background component (solvent molecules, dissolved
  polymer that doesn't contribute to the main scatterer signal)
- Can also absorb contributions from laser speckle in mixed detection setups
- Often near zero for pure samples; relevant for heterogeneous systems

**Default bounds:** ``[0.0, 1.0e4]`` Å²/s

---

Shear Parameters (laminar_flow only)
--------------------------------------

gamma_dot_0 — Reference Shear Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\dot\gamma_0`
**Units:** s⁻¹

:math:`\dot\gamma_0` is the amplitude of the shear rate. For time-independent
shear (:math:`\beta = 0`), it equals the applied shear rate :math:`\dot\gamma`.
For time-dependent shear, it is the coefficient in:

.. math::

   \dot\gamma(t) = \dot\gamma_0 \, t^\beta + \dot\gamma_\text{offset}

The accumulated shear strain is the numerical integral:

.. math::

   \Gamma(t_1, t_2) = \int_{t_1}^{t_2} \dot\gamma(t)\,dt

**Physical interpretation:**

- In Couette flow: typically equal to the applied angular velocity × geometric factor
- Typical Couette shear rates: 0.01–1000 s⁻¹
- :math:`\dot\gamma_0 \approx 0` for a sample at rest in the shear cell

**Default bounds:** ``[0.0, 1.0e4]`` s⁻¹

.. note::

   For multi-scale problems where :math:`D_0 \sim 10^4` and
   :math:`\dot\gamma_0 \sim 10^{-3}` (scale ratio > 10⁶), consider enabling
   CMA-ES global optimization. See
   :doc:`../03_advanced_topics/cmaes_optimization`.

beta — Shear Rate Time-Dependence Exponent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\beta`
**Units:** dimensionless

Analogous to :math:`\alpha` for the shear rate:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Value
     - Classification
     - Physical picture
   * - :math:`\beta = 0`
     - Steady shear
     - Constant applied shear rate
   * - :math:`\beta < 0`
     - Shear rate decreasing with time
     - Shear thinning; time-dependent flow
   * - :math:`\beta > 0`
     - Shear rate increasing with time
     - Shear thickening; time-dependent startup flow

**Default bounds:** ``[-2.0, 2.0]``

gamma_dot_offset — Baseline Shear Rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\dot\gamma_\text{offset}`
**Units:** s⁻¹

Constant shear rate contribution, independent of time. Analogous to
:math:`D_\text{offset}` for shear. Represents steady background flow
superimposed on the time-dependent component.

**Default bounds:** ``[0.0, 1.0e3]`` s⁻¹

phi_0 — Angular Offset
~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\phi_0`
**Units:** degrees

:math:`\phi_0` is the angular offset between the shear flow direction and the
detector coordinate system. The shear-induced decorrelation is maximal when the
scattering vector is **parallel** to the flow direction (:math:`\phi = \phi_0`)
and zero when perpendicular.

**Physical interpretation:**

- If the scattering geometry is perfectly aligned with the flow direction,
  :math:`\phi_0 = 0`
- Misalignment between shear cell and detector introduces a non-zero :math:`\phi_0`
- The value should be consistent across measurements with the same cell alignment

**Default bounds:** ``[-180.0, 180.0]`` degrees

---

Optical Parameters
-------------------

These parameters describe properties of the beam and detector, not the sample.
They appear as per-angle corrections when ``per_angle_mode`` is not ``constant``.

Contrast (beta_coherence)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symbol:** :math:`\beta_\phi`
**Units:** dimensionless, range [0, 1]

The speckle contrast is the amplitude of the correlation function at zero lag:

.. math::

   C_2(\phi, 0) = \text{offset}(\phi) + \beta(\phi)

It is determined by the **coherence** of the X-ray beam and the detector
pixel size relative to the speckle size:

- Perfect coherence, single speckle per pixel: :math:`\beta = 1`
- Partial coherence or multiple speckles per pixel: :math:`\beta < 1`

Typical values: 0.01–0.5

Offset
~~~~~~~

**Units:** dimensionless

The incoherent background level in the correlation function at large lag times:

.. math::

   \lim_{|t_2 - t_1| \to \infty} C_2(\phi, t_1, t_2) = \text{offset}(\phi)

For an ideal experiment with no background: offset = 1.0. In practice:

- Values near 1.0 indicate clean data
- Values significantly above or below 1.0 suggest static or rapidly-fluctuating
  background contributions

---

Parameter Reference Table
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 12 12 14 14 28

   * - Parameter
     - Symbol
     - Units
     - Default Init
     - Bounds
     - Mode
   * - ``D0``
     - :math:`D_0`
     - Å²/s
     - 1.0
     - [0, 1e6]
     - static, laminar_flow
   * - ``alpha``
     - :math:`\alpha`
     - —
     - -0.5
     - [-2, 2]
     - static, laminar_flow
   * - ``D_offset``
     - :math:`D_\text{offset}`
     - Å²/s
     - 0.01
     - [0, 1e4]
     - static, laminar_flow
   * - ``gamma_dot_0``
     - :math:`\dot\gamma_0`
     - s⁻¹
     - 0.01
     - [0, 1e4]
     - laminar_flow
   * - ``beta``
     - :math:`\beta`
     - —
     - -0.5
     - [-2, 2]
     - laminar_flow
   * - ``gamma_dot_offset``
     - :math:`\dot\gamma_\text{offset}`
     - s⁻¹
     - 0.001
     - [0, 1e3]
     - laminar_flow
   * - ``phi_0``
     - :math:`\phi_0`
     - degrees
     - 0.0
     - [-180, 180]
     - laminar_flow
   * - ``contrast``
     - :math:`\beta_\phi`
     - —
     - 0.1
     - [0, 1]
     - per-angle
   * - ``offset``
     - —
     - —
     - 1.0
     - [0.5, 2.0]
     - per-angle

---

Setting Initial Values
-----------------------

Poor initial values are a common cause of convergence failures. Here are
guidelines:

**For D₀:**

.. code-block:: python

   # Estimate from Stokes-Einstein (remember: D0_homodyne = 2 * D_SE)
   # For water at 25°C: kT = 4.11e-21 J
   # eta_water = 8.9e-4 Pa·s
   # Rh = particle radius in meters

   import numpy as np
   kT = 4.11e-21  # J at 25°C
   eta = 8.9e-4   # Pa·s (water)
   Rh_m = 10e-9   # 10 nm in meters
   D_SE = kT / (6 * np.pi * eta * Rh_m)  # Stokes-Einstein D
   D0_homodyne = 2 * D_SE * 1e20         # Å²/s (factor of 2!)
   print(f"D0 estimate: {D0_homodyne:.1e} Å²/s")
   # D0 estimate: 4.8e+04 Å²/s

**For alpha:** start at -0.5 (mild sub-diffusion) as a neutral starting point.

**For gamma_dot_0:** use the nominal applied shear rate from your rheometer setting.

**For phi_0:** use 0 if the geometry is known to be aligned; scan if uncertain.

---

See Also
---------

- :doc:`analysis_modes` — Which parameters apply to each mode
- :doc:`../02_data_and_fitting/result_interpretation` — Reading parameter values from results
- :doc:`../04_practical_guides/configuration` — Setting bounds in YAML
- :ref:`theory` — Derivation of the model equations
