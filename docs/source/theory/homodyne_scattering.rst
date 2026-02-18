.. _theory_homodyne_scattering:

Homodyne Scattering: Laminar Flow Model
========================================

In a Taylor-Couette or shear-cell geometry, the colloidal suspension undergoes laminar
flow that imposes a systematic drift on every particle. XPCS in this configuration measures
both diffusive and advective contributions to particle dynamics. This page derives the full
:math:`c_2` equation for homodyne detection under laminar shear flow, as implemented in the
``laminar_flow`` analysis mode.

.. contents:: Contents
   :local:
   :depth: 2


Taylor-Couette Geometry
-----------------------

The standard experimental geometry is a **concentric-cylinder** (Taylor-Couette) cell:

- **Inner cylinder (rotor)**: rotates at angular velocity :math:`\Omega`
- **Outer cylinder (stator)**: fixed
- **Gap** :math:`h`: the distance between rotor and stator where the sample sits
- The local shear rate :math:`\dot{\gamma}(t)` may vary with time (e.g., during a
  start-up flow or yielding ramp)

The X-ray beam traverses the sample at an azimuthal angle :math:`\phi` relative to the
flow direction (the tangential direction of the inner cylinder). Because the beam samples
a distribution of velocities across the gap, the correlation function receives a
characteristic **sinc-squared** modulation from the velocity dispersion.

.. note::

   The gap parameter :math:`h` is fixed by the instrument geometry (typically 0.5–2 mm)
   and is not a fitted parameter in homodyne. It is loaded from the experimental
   configuration.


Gap Integration and the sinc² Modulation
-----------------------------------------

Consider a scattering element at position :math:`x` within the gap (:math:`0 \leq x \leq h`).
In simple shear, the velocity along the flow direction is:

.. math::

   v_x(x, t) \;=\; \dot{\gamma}(t)\,x

The projection onto the scattering vector :math:`\mathbf{q}` (at azimuthal angle :math:`\phi`
to the flow) is:

.. math::

   v_\parallel(x, t) \;=\; v_x(x, t)\cos\phi \;=\; \dot{\gamma}(t)\,x\cos\phi

The contribution to :math:`c_1` from advection is a phase factor
:math:`e^{iq\int_{t_1}^{t_2} v_\parallel(x,t)dt}`. Integrating uniformly across the gap:

.. math::
   :label: gap_integration

   c_1^\mathrm{(shear)}(\mathbf{q}, t_1, t_2)
   \;=\; \frac{1}{h}\int_0^h
         \exp\!\left(iq\cos\phi \int_{t_1}^{t_2}\dot{\gamma}(t)x\,dt\right)dx

Let :math:`\Gamma(t_1, t_2) = \int_{t_1}^{t_2}\dot{\gamma}(t)\,dt` be the accumulated strain.
The integral over :math:`x` is:

.. math::

   \frac{1}{h}\int_0^h e^{iq\cos\phi\,\Gamma\,x}\,dx
   \;=\; e^{i\frac{qh\cos\phi\,\Gamma}{2}}
     \cdot \mathrm{sinc}\!\left(\frac{qh\cos\phi\,\Gamma}{2\pi}\right)

In the homodyne (intensity) correlation :math:`|c_1^\mathrm{(shear)}|^2`, the phase factor
cancels and we obtain:

.. math::
   :label: sinc2_term

   \left|c_1^\mathrm{(shear)}\right|^2
   \;=\; \mathrm{sinc}^2\!\left(\frac{qh\cos\phi\,\Gamma(t_1,t_2)}{2\pi}\right)

.. note::

   The homodyne package uses the **normalized sinc** convention
   :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)`, so that :math:`\mathrm{sinc}(0)=1`.
   This is consistent with NumPy/JAX conventions.


Full Laminar Flow Equation
---------------------------

Combining the diffusive decay (Debye-Waller factor) from :eq:`c2_general_homodyne` with
the shear modulation :eq:`sinc2_term`, the full homodyne correlation function for laminar
flow is:

.. math::
   :label: c2_laminar_flow

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1
     + \beta(t_1,t_2)
       \exp\!\left(-q^2\int_{t_1}^{t_2}J(t')\,dt'\right)
       \times
       \mathrm{sinc}^2\!\left(\frac{qh\cos\phi\,\Gamma(t_1,t_2)}{2\pi}\right)

where:

.. math::

   \Gamma(t_1, t_2) \;=\; \int_{t_1}^{t_2}\dot{\gamma}(t)\,dt

is the **accumulated strain** and :math:`\phi` is the azimuthal angle of the scattering
vector relative to the flow direction.

.. note::

   In this derivation, :math:`\phi` is the **physical** angle between the scattering
   vector and the flow direction. In homodyne's implementation, the detector reports
   azimuthal angles :math:`\phi_\mathrm{det}` in the laboratory frame. The angular
   offset parameter :math:`\phi_0` maps between them:
   :math:`\phi = \phi_\mathrm{det} - \phi_0`. The code therefore computes
   :math:`\cos(\phi_0 - \phi_\mathrm{det})`, which equals :math:`\cos(\phi_\mathrm{det} - \phi_0)`
   since cosine is even.

.. warning::

   Equation :eq:`c2_laminar_flow` is valid only for **homodyne** detection (measuring
   scattered intensity only). If a reference beam is mixed with the scattered beam
   (heterodyne detection), or if multiple scattering components exist, see
   :ref:`theory_heterodyne_scattering`.


Angular Dependence
------------------

The :math:`\cos\phi` factor in the sinc argument makes the correlation function strongly
**azimuthal-angle dependent**:

- At :math:`\phi = 90°` (scattering vector perpendicular to flow): :math:`\cos\phi = 0`,
  the sinc term equals 1 and only diffusion contributes.
- At :math:`\phi = 0°` (scattering vector parallel to flow): :math:`\cos\phi = 1`,
  the sinc modulation is maximal.
- For intermediate angles: mixed contribution.

In practice, XPCS data is collected simultaneously at multiple azimuthal angles (typically
:math:`N_\phi = 23` sectors), enabling simultaneous fitting across all angles. The per-angle
contrast :math:`\beta(\phi)` accounts for additional angle-dependent coherence variations.

See :ref:`theory_anti_degeneracy` for the per-angle scaling strategy that prevents
parameter absorption degeneracy.


Homodyne Implementation Parameters
------------------------------------

The homodyne package parameterizes the shear rate as a power law with offset:

.. math::
   :label: gamma_dot_model

   \dot{\gamma}(t) \;=\; \dot{\gamma}_0 \cdot t^\beta + \dot{\gamma}_\mathrm{offset}

where :math:`\dot{\gamma}_0` is the shear rate prefactor, :math:`\beta` is the time
exponent (note: separate from :math:`\beta` speckle contrast; context distinguishes them),
and :math:`\dot{\gamma}_\mathrm{offset}` is a constant background shear rate.

The accumulated strain is computed by **cumulative trapezoidal integration** of
:math:`\dot{\gamma}(t)` on the experimental time grid:

.. math::
   :label: gamma_integral_homodyne

   \Gamma(t_1, t_2) \;=\; \int_{t_1}^{t_2} \dot{\gamma}(t')\,dt'

For reference, the analytical result is:

.. math::

   \Gamma(t_1, t_2)
   \;=\; \frac{\dot{\gamma}_0}{1+\beta_\gamma}\!\left(t_2^{1+\beta_\gamma} - t_1^{1+\beta_\gamma}\right)
         + \dot{\gamma}_\mathrm{offset}(t_2 - t_1)

where :math:`\beta_\gamma` is the shear exponent parameter (named ``beta`` in config,
not to be confused with optical contrast).

**Full parameter set for** ``laminar_flow`` **mode**:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Config key
     - Symbol
     - Role in Equation :eq:`c2_laminar_flow`
   * - ``D0``
     - :math:`D_0`
     - Diffusion integral prefactor
   * - ``alpha``
     - :math:`\alpha`
     - Diffusion power-law exponent
   * - ``D_offset``
     - :math:`D_\mathrm{offset}`
     - Constant diffusion background
   * - ``gamma_dot_0``
     - :math:`\dot{\gamma}_0`
     - Shear rate prefactor
   * - ``beta``
     - :math:`\beta_\gamma`
     - Shear rate power-law exponent
   * - ``gamma_dot_offset``
     - :math:`\dot{\gamma}_\mathrm{offset}`
     - Constant shear rate background
   * - ``phi_0``
     - :math:`\phi_0`
     - Reference azimuthal angle offset


Code Example
------------

.. code-block:: python

   from homodyne.core.jax_backend import compute_g2_laminar_flow
   import jax.numpy as jnp

   # Physical parameters
   params = {
       "D0": 50.0,               # Å²/s
       "alpha": 0.33,            # Andrade creep exponent
       "D_offset": 0.1,
       "gamma_dot_0": 1e-3,      # s^{-1}
       "beta": -0.67,            # shear rate exponent
       "gamma_dot_offset": 0.0,
       "phi_0": 0.0,
   }

   # Time grid (two-time matrix)
   t = jnp.linspace(0.01, 100.0, 200)
   t1, t2 = jnp.meshgrid(t, t, indexing="ij")

   # Experimental parameters
   q = 0.01       # Å^{-1}
   phi = 0.0      # flow direction
   h = 1.0        # mm gap
   beta_contrast = 0.3

   # Result: c2 matrix
   c2 = compute_g2_laminar_flow(t1, t2, q, phi, h, beta_contrast, **params)


Connection to Rheology
----------------------

The Andrade creep law (:math:`\gamma \sim t^{1/3}`) observed in colloidal suspensions
near the glass transition (He et al. PNAS 2025) corresponds to:

.. math::

   \dot{\gamma}(t) \;\propto\; t^{-2/3}
   \quad \Rightarrow \quad \beta_\gamma = -2/3, \quad \dot{\gamma}_0 > 0

This is the canonical signature of **plastic creep** driven by stress-activated cage
rearrangements. The homodyne model captures this through the time-dependent shear rate
:math:`\dot{\gamma}(t)` parameterization, making it directly comparable to bulk rheological
measurements.


.. seealso::

   - :ref:`theory_transport_coefficient` — definition of :math:`J(t)`
   - :ref:`theory_correlation_functions` — Siegert relation and :math:`c_2` derivation
   - :ref:`theory_heterodyne_scattering` — multi-component extension
   - :ref:`theory_anti_degeneracy` — per-angle scaling modes
   - :ref:`theory_yielding_dynamics` — Andrade creep physics
   - :mod:`homodyne.core.jax_backend` — JIT-compiled implementation
