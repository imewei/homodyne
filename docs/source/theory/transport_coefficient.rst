.. _theory_transport_coefficient:

Transport Coefficient J(t)
==========================

The **transport coefficient** :math:`J(t)` is the central quantity in the homodyne framework.
It encodes how the variance of particle displacements grows in time and provides a direct
bridge between microscopic stochastic dynamics and macroscopic rheological observables through
a generalized Green-Kubo relation.

.. contents:: Contents
   :local:
   :depth: 2


Definition
----------

The transport coefficient admits three equivalent representations.

**Variance form** (definition):

.. math::
   :label: J_variance

   J(t) \;=\; \frac{d}{dt}\,\mathrm{Var}\!\left[x(t)\right]

This is the instantaneous rate of growth of the mean-squared displacement (MSD) of a
particle at time :math:`t` (He et al. PNAS 2025, Eq. S-38).

**Covariance form**:

.. math::
   :label: J_covariance

   J(t) \;=\; 2\,\mathrm{Cov}\!\left[x(t),\, v(t)\right]

where :math:`v(t) = \dot{x}(t)` is the instantaneous particle velocity. The factor of 2 arises
from the symmetry :math:`\mathrm{Cov}[x, v] = \frac{1}{2} \frac{d}{dt}\mathrm{Var}[x]`.

**Green-Kubo form** (microscopic origin):

.. math::
   :label: J_greenkubo

   J(t) \;=\; 2 \int_0^{t} \mathrm{Cov}\!\left[v(t),\, v(t')\right] dt'

Equation :eq:`J_greenkubo` is the **generalized Green-Kubo relation**: the transport
coefficient equals twice the integral of the velocity autocorrelation function (VACF) from
:math:`0` to :math:`t`. For a stationary process this reduces to the classical
:math:`J = 2\int_0^\infty C_v(\tau)\,d\tau = 2D`.


Physical Interpretation
-----------------------

The transport coefficient :math:`J(t)` has units of :math:`\text{length}^2 / \text{time}`,
identical to a diffusion coefficient. It measures how rapidly positional uncertainty accumulates
at time :math:`t`.

.. note::

   In equilibrium, :math:`J` is constant: :math:`J = 2D`, where :math:`D` is the Stokes-Einstein
   diffusion coefficient. The homodyne framework generalizes this to **non-stationary** processes
   where :math:`J(t)` can vary with time (e.g., aging, yielding).

The **diffusion integral** that enters the correlation function is:

.. math::
   :label: diffusion_integral

   \mathcal{D}(t_1, t_2) \;=\; \int_{t_1}^{t_2} J(t') \, dt'
   \;=\; \mathrm{Var}\!\left[x(t_2) - x(t_1)\right]

This equals the variance of the net displacement, confirming that :math:`J` directly controls
how the Debye-Waller factor decays in the correlation function.


Homodyne Implementation
-----------------------

The homodyne package parameterizes a time-dependent diffusion rate as a power law with offset:

.. math::
   :label: D_model

   D(t) \;=\; D_0 \cdot t^\alpha + D_\mathrm{offset}

where :math:`D_0 > 0` is the diffusion prefactor, :math:`\alpha` is the anomalous exponent
(:math:`\alpha = 0` for normal diffusion, :math:`\alpha < 0` for sub-diffusion,
:math:`\alpha > 0` for super-diffusion), and :math:`D_\mathrm{offset} \geq 0` is a constant
background diffusion.

The **diffusion integral** entering the correlation function :eq:`c1_general` is computed
by **cumulative trapezoidal integration** of :math:`D(t)` on the experimental time grid:

.. math::
   :label: J_integral_homodyne

   \mathcal{D}(t_1, t_2) \;=\; \int_{t_1}^{t_2} D(t') \, dt'

For reference, the analytical result (used in tests and for intuition, but not in the
implementation) is:

.. math::

   \int_{t_1}^{t_2} D(t') \, dt'
   \;=\; \frac{D_0}{1+\alpha}\!\left(t_2^{1+\alpha} - t_1^{1+\alpha}\right)
         + D_\mathrm{offset}(t_2 - t_1)

The numerical integration is implemented in :mod:`homodyne.core.physics_utils` (shared
trapezoidal kernels) and used by both :mod:`homodyne.core.jax_backend` (NLSQ meshgrid mode)
and :mod:`homodyne.core.physics_cmc` (CMC element-wise mode).

.. warning::

   **Convention**: The :math:`D_0` parameter in homodyne absorbs a factor of 2 from the
   formal transport coefficient. For standard Brownian motion, the physical
   Stokes-Einstein diffusion coefficient is :math:`D_\mathrm{SE} = k_B T / (6\pi\eta R_h)`,
   while homodyne's :math:`D_0 = 2 D_\mathrm{SE}`. This is because the Siegert relation
   :eq:`siegert` squares :math:`c_1`, and the correlation function depends on
   :math:`|c_1|^2 = \exp(-q^2 \mathcal{D})` where :math:`\mathcal{D} = \int D(t')\,dt'`.
   For the standard result :math:`|c_1|^2 = e^{-2 q^2 D_\mathrm{SE}\,\tau}` to hold,
   we need :math:`D_0 = 2 D_\mathrm{SE}`.


Connection to Physical Diffusion Coefficient
---------------------------------------------

In homodyne's parameterization (where :math:`D_0` absorbs the factor of 2; see warning above),
the mean-squared displacement is:

.. math::
   :label: msd_anomalous

   \mathrm{MSD}(t) \;\equiv\; \mathrm{Var}[x(t) - x(0)]
   \;=\; \int_0^t D(t')\,dt'
   \;=\; \frac{D_0}{1+\alpha}\,t^{1+\alpha} + D_\mathrm{offset}\,t

The effective **physical** (Stokes-Einstein) diffusion coefficient at time :math:`t` is
half the homodyne value:

.. math::

   D_\mathrm{SE}(t) \;=\; \frac{\mathrm{MSD}(t)}{2t}
   \;=\; \frac{D_0}{2(1+\alpha)}\,t^\alpha + \frac{D_\mathrm{offset}}{2}

For :math:`\alpha = 0` (standard Brownian motion): :math:`D_\mathrm{SE} = D_0/2 + D_\mathrm{offset}/2`,
a constant. For :math:`\alpha < 0` (sub-diffusion with aging): particles slow down over time.


Table of J(t) for Classical Processes
--------------------------------------

The following table summarizes :math:`J(t)` for Langevin processes that arise naturally
in soft matter. See :ref:`theory_classical_processes` for full derivations.

.. list-table:: Transport coefficients for classical stochastic processes
   :header-rows: 1
   :widths: 30 40 30

   * - Process
     - :math:`J(t)`
     - Regime
   * - Wiener (free diffusion)
     - :math:`2D`
     - :math:`D` = constant
   * - Anomalous diffusion
     - :math:`2D_0 t^{\alpha}`
     - :math:`\alpha \in (-1, 1]`
   * - Ornstein-Uhlenbeck
     - :math:`2D\!\left(1 - e^{-\gamma t}\right)^2`
     - Confinement radius :math:`\sqrt{D/\gamma}`
   * - Brownian oscillator (underdamped)
     - :math:`\displaystyle 2D\frac{\gamma^2}{\omega_s^2} e^{-\gamma t}\sin^2(\omega_s t)`
     - :math:`\omega_s^2 = \omega_0^2 - \gamma^2/4 > 0`
   * - Brownian oscillator (overdamped)
     - :math:`\displaystyle 8D\frac{\gamma^2}{\gamma_s^2} e^{-\gamma t}\sinh^2\!\left(\tfrac{1}{2}\gamma_s t\right)`
     - :math:`\gamma_s^2 = \gamma^2 - 4\omega_0^2 > 0`

where :math:`\gamma` is the friction coefficient, :math:`\omega_0` is the trap frequency,
:math:`\omega_s = \sqrt{\omega_0^2 - \gamma^2/4}` and :math:`\gamma_s = \sqrt{\gamma^2 - 4\omega_0^2}`.
Here :math:`D` is the **physical** Stokes-Einstein diffusion coefficient; in homodyne's
parameterization :math:`D_0 = 2D` (see :eq:`D_model`).


Relationship to Rheology
------------------------

The Green-Kubo form :eq:`J_greenkubo` links :math:`J(t)` to the **complex shear modulus**
:math:`G^*(\omega)` of the suspending medium through the generalized Stokes-Einstein relation
(GSER):

.. math::

   D(\omega) \;=\; \frac{k_B T}{6\pi R \eta(\omega)}
   \;=\; \frac{k_B T}{6\pi R}\,\frac{1}{G^*(\omega)}

where :math:`R` is the particle radius, :math:`k_B T` is the thermal energy, and
:math:`\eta(\omega)` is the frequency-dependent viscosity. Measuring :math:`J(t)` from
XPCS data therefore yields a non-invasive probe of local viscoelastic properties.

For the **yielding transition** studied in He et al. PNAS 2025, the time-evolution of
:math:`J(t)` during the rheological loading protocol distinguishes:

- **Repulsive suspensions**: Andrade creep (:math:`\gamma \sim t^{1/3}`) maps to
  :math:`J(t) \propto t^{-2/3}` — a sub-diffusive, aging transport coefficient.
- **Attractive suspensions**: heterogeneous shear banding produces non-Gaussian
  displacement distributions not captured by a single :math:`J(t)`.

See :ref:`theory_yielding_dynamics` for details.


.. seealso::

   - :ref:`theory_correlation_functions` — how :math:`J(t)` enters :math:`c_1` and :math:`c_2`
   - :ref:`theory_classical_processes` — derivation of :math:`J(t)` for Langevin models
   - :mod:`homodyne.core.jax_backend` — JIT-compiled implementation
   - :mod:`homodyne.core.physics_cmc` — element-wise CMC implementation
