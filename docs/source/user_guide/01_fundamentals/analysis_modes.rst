.. _analysis_modes:

Analysis Modes
==============

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The two analysis modes (static and laminar_flow) and when to use each
- The model equations and free parameters for each mode
- How to choose the right mode for your experiment
- What per-angle scaling is and when it matters
- Example YAML configurations for each mode

---

Overview
---------

Homodyne provides two analysis modes, corresponding to two different physical
models for the dynamics:

.. list-table::
   :header-rows: 1
   :widths: 20 15 45 20

   * - Mode
     - Parameters
     - Physical System
     - Key Equation Feature
   * - ``static``
     - 3
     - Brownian/anomalous diffusion; no flow
     - Purely diffusive decay
   * - ``laminar_flow``
     - 7
     - Diffusion + laminar shear (Couette)
     - Sinc factor from shear advection

---

Static Mode
-----------

**When to use static mode:**

- Equilibrium samples (no applied shear or flow)
- Gels, glasses, concentrated suspensions aging under gravity
- Any experiment where the dominant dynamics are diffusive
- When you want to characterize :math:`D_0` and check for anomalous diffusion
- Quick initial characterization of an unknown sample

**Model equation:**

.. math::

   C_2(\phi, t_1, t_2) = \text{offset}(\phi) + \beta(\phi) \cdot
   \exp\!\left(-q^2 \int_{t_1}^{t_2} J(t')\,dt'\right)

where :math:`J(t)` is the **transport coefficient** — the instantaneous rate of
growth of mean-squared displacement — and the integral
:math:`\int J\,dt'` equals the variance of particle displacement over the
interval :math:`[t_1, t_2]`. Homodyne evaluates this integral **numerically**
using cumulative trapezoidal integration (see :ref:`theory_transport_coefficient`).

**Free parameters (3):**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``D0``
     - :math:`D_0`
     - Å²/s
     - Reference diffusion coefficient
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - Diffusion time-dependence exponent
   * - ``D_offset``
     - :math:`D_\text{offset}`
     - Å²/s
     - Baseline (constant) diffusion rate

**Example YAML configuration for static mode:**

.. code-block:: yaml

   analysis:
     mode: "static"

   parameter_space:
     D0:
       initial: 1.0
       bounds: [0.01, 100.0]
     alpha:
       initial: -0.5
       bounds: [-2.0, 0.0]
     D_offset:
       initial: 0.01
       bounds: [0.0, 10.0]

**Generating a static template:**

.. code-block:: bash

   homodyne-config --mode static --output config_static.yaml

---

Laminar Flow Mode
-----------------

**When to use laminar flow mode:**

- Samples in Couette (concentric cylinder) or cone-plate shear cells
- Any experiment where velocity gradients create an angular dependence in :math:`C_2`
- When you observe characteristic oscillations in the azimuthal :math:`q`-dependence
- Flow characterization: measuring shear rate profiles, velocity gradients

**Physical setup:**

In a Taylor-Couette geometry, the sample is sheared between concentric
cylinders. The velocity field creates a **displacement** of scatterers that
depends on both the lag time :math:`(t_2 - t_1)` and the azimuthal angle
:math:`\phi` between the scattering vector and the flow direction.

The **accumulated shear strain** over the interval :math:`[t_1, t_2]` is:

.. math::

   \Gamma(t_1, t_2) = \int_{t_1}^{t_2} \dot\gamma(t)\,dt

where :math:`\dot\gamma(t)` is the time-dependent shear rate. Like the
diffusion integral, this is evaluated **numerically** by homodyne.

**Model equation:**

.. math::

   C_2(\phi, t_1, t_2) = \text{offset}(\phi) + \beta(\phi) \cdot
   \exp\!\left(-q^2 \int_{t_1}^{t_2} J(t')\,dt'\right) \cdot
   \mathrm{sinc}^2\!\left(\frac{q h\,\cos(\phi - \phi_0)\;\Gamma(t_1, t_2)}{2\pi}\right)

where :math:`h` is the gap distance (stator-rotor separation in Å),
:math:`J(t)` is the transport coefficient (same as in static mode),
:math:`\Gamma(t_1, t_2)` is the accumulated strain integral, and
:math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)` (normalized sinc; NumPy/JAX
convention). See :ref:`theory_homodyne_scattering` for the full derivation.

**Free parameters (7):**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``D0``
     - :math:`D_0`
     - Å²/s
     - Reference diffusion coefficient
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - Diffusion time-dependence exponent
   * - ``D_offset``
     - :math:`D_\text{offset}`
     - Å²/s
     - Baseline diffusion
   * - ``gamma_dot_0``
     - :math:`\dot\gamma_0`
     - s⁻¹
     - Reference shear rate
   * - ``beta``
     - :math:`\beta`
     - dimensionless
     - Shear rate time-dependence exponent
   * - ``gamma_dot_offset``
     - :math:`\dot\gamma_\text{offset}`
     - s⁻¹
     - Baseline shear rate
   * - ``phi_0``
     - :math:`\phi_0`
     - degrees
     - Angular offset (flow direction)

**Example YAML configuration for laminar flow mode:**

.. code-block:: yaml

   data:
     file_path: "shear_data.h5"
     q_value: 0.054       # Å⁻¹
     gap_distance: 500.0  # µm → stored as Å internally (x10⁴)
     dt: 0.1              # s

   analysis:
     mode: "laminar_flow"

   optimization:
     method: "nlsq"
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"  # CRITICAL: prevents parameter absorption

   parameter_space:
     D0:
       initial: 1.0
       bounds: [0.01, 100.0]
     alpha:
       initial: -0.5
       bounds: [-2.0, 0.0]
     D_offset:
       initial: 0.01
       bounds: [0.0, 10.0]
     gamma_dot_0:
       initial: 0.01
       bounds: [1.0e-6, 10.0]
     beta:
       initial: -0.5
       bounds: [-2.0, 0.0]
     gamma_dot_offset:
       initial: 0.001
       bounds: [0.0, 1.0]
     phi_0:
       initial: 0.0
       bounds: [-180.0, 180.0]

**Generating a laminar flow template:**

.. code-block:: bash

   homodyne-config --mode laminar_flow --output config_flow.yaml

---

Choosing Between Modes
-----------------------

The choice of mode should be guided by the **physics of your experiment**, not
by which mode gives a better fit. Here is a decision guide:

**Use static mode if:**

- No shear cell or flow cell is present
- The :math:`C_2` data shows no angular dependence (all :math:`\phi` give
  identical correlation functions)
- You are characterizing equilibrium dynamics (aging, gelation, etc.)
- You want a 3-parameter baseline before adding complexity

**Use laminar flow mode if:**

- The sample is in a Couette or cone-plate shear cell
- You observe a clear angular dependence: :math:`C_2` varies with :math:`\phi`
- The decay timescale is short compared to :math:`1/\dot\gamma` (shear advection
  is visible in the correlation)
- You want to separate diffusion from advection contributions

.. warning::

   Do not use ``laminar_flow`` mode on equilibrium data "to be safe."
   The extra parameters (:math:`\dot\gamma_0`, :math:`\beta`, etc.) can absorb
   fitting degrees of freedom and produce spurious results. Use the simplest
   model consistent with your physical setup.

.. note::

   A good diagnostic: plot :math:`C_2` for several azimuthal angles on the
   same axes. If the curves overlap, use static mode. If they diverge
   systematically with :math:`\phi`, use laminar_flow mode.

---

Per-Angle Scaling
------------------

For experiments with **multiple azimuthal angles** (:math:`n_\phi \geq 3`),
homodyne supports per-angle corrections to the speckle contrast and incoherent
background. This is controlled by ``per_angle_mode`` in the configuration.

**Why per-angle scaling matters:**

In practice, the speckle contrast :math:`\beta(\phi)` and background offset
can vary with azimuthal angle due to:

- Spatial variations in detector sensitivity
- Angle-dependent multiple scattering
- Variations in the coherent flux footprint across detector segments

Without per-angle correction, these effects are absorbed into the physical
parameters, biasing :math:`D_0` and :math:`\dot\gamma_0`.

**Available modes:**

.. list-table::
   :header-rows: 1
   :widths: 15 50 15 20

   * - Mode
     - Behavior
     - Extra Params
     - When to Use
   * - ``auto``
     - When n_phi ≥ 3: estimates per-angle values, averages them, optimizes 2 averaged values alongside physical params
     - +2
     - **Default; always recommended**
   * - ``constant``
     - Per-angle contrast/offset from quantile estimation; fixed (not optimized)
     - 0
     - Very fast runs; when scaling is well-known a priori
   * - ``individual``
     - Independent contrast + offset per angle; all optimized
     - +2×n_phi
     - When angles are truly independent (rarely needed)
   * - ``fourier``
     - Angular variation modeled as truncated Fourier series (K=2)
     - +10
     - Smooth angular dependence; physical smoothness constraints

.. tip::

   Always use ``per_angle_mode: "auto"`` unless you have a specific reason to
   deviate. The ``auto`` mode prevents the most common degeneracy: contrast
   or offset absorbing :math:`D_0` or :math:`\dot\gamma_0`.

**Configuration example:**

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # Recommended for laminar_flow with n_phi >= 3
     cmc:
       per_angle_mode: "auto"     # Should match NLSQ setting

---

Mode-Specific Parameter Counts
--------------------------------

The total number of free parameters depends on mode and per-angle setting:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Mode
     - per_angle_mode
     - n_phi = 1
     - n_phi = 23
   * - ``static``
     - ``constant``
     - 3
     - 3
   * - ``static``
     - ``auto``
     - 3
     - 5 (3 + 2 avg)
   * - ``laminar_flow``
     - ``constant``
     - 7
     - 7
   * - ``laminar_flow``
     - ``auto``
     - 7
     - 9 (7 + 2 avg)
   * - ``laminar_flow``
     - ``individual``
     - 7
     - 53 (7 + 46)

---

See Also
---------

- :doc:`parameter_guide` — Detailed interpretation of each parameter
- :doc:`../02_data_and_fitting/nlsq_fitting` — Running NLSQ fits
- :doc:`../03_advanced_topics/per_angle_modes` — Deep dive into per-angle scaling
- :doc:`../03_advanced_topics/laminar_flow` — Complete laminar flow guide
- :doc:`../04_practical_guides/configuration` — Full YAML configuration reference
