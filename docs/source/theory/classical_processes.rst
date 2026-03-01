.. _theory_classical_processes:

Classical Langevin Processes
=============================

The homodyne framework is built on a Gaussian model for particle dynamics, motivated by the
classical Langevin equation. This page derives the transport coefficient :math:`J(t)` for the
four canonical stochastic processes relevant to soft-matter XPCS experiments, and compares
their physical behaviour.



The Langevin Equation
---------------------

The overdamped Langevin equation for a colloidal particle in one dimension is:

.. math::
   :label: langevin_full

   m\ddot{x}(t) \;=\; -m\gamma\,\dot{x}(t) + \eta(t) + F_\mathrm{ex}(t)

where:

- :math:`m` is the particle mass
- :math:`\gamma` is the friction coefficient (:math:`\text{s}^{-1}`)
- :math:`\eta(t)` is Gaussian white noise with :math:`\langle\eta(t)\rangle = 0` and
  :math:`\langle\eta(t)\eta(t')\rangle = 2m^2\gamma k_B T\,\delta(t-t')` (fluctuation-dissipation theorem)
- :math:`F_\mathrm{ex}(t)` is an external deterministic force (shear, applied field, etc.)

For colloidal particles (large :math:`m\gamma \tau_B \gg 1`, where :math:`\tau_B = m/\gamma`
is the Brownian relaxation time), the inertial term :math:`m\ddot{x}` is negligible on
experimental timescales, giving the **overdamped** equation:

.. math::
   :label: langevin_overdamped

   \dot{x}(t) \;=\; \mu\,F_\mathrm{ex}(t) + \sqrt{2D}\,\xi(t)

where :math:`\mu = 1/(m\gamma)` is the mobility, :math:`D = k_B T\mu` (Einstein relation),
and :math:`\xi(t)` is unit-variance white noise.


Wiener Process (Standard Diffusion)
-------------------------------------

Setting :math:`F_\mathrm{ex} = 0` in :eq:`langevin_overdamped`, the position evolves as a
**Wiener process** (Brownian motion):

.. math::

   x(t) \;=\; x(0) + \sqrt{2D}\int_0^t \xi(t')\,dt'

**Variance** (MSD):

.. math::

   \mathrm{Var}[x(t) - x(0)] \;=\; 2Dt

**Transport coefficient**:

.. math::
   :label: J_wiener

   J(t) \;=\; \frac{d}{dt}\,\mathrm{Var}[x(t)] \;=\; 2D \;=\; \text{const.}

**Equilibrium** :math:`g_2`: simple exponential decay with
:math:`\Gamma = Dq^2` and :math:`g_2(q,\tau) = 1 + \beta\,e^{-2Dq^2\tau}`.

**XPCS application**: This is the baseline model for Brownian diffusion in dilute suspension.
The Stokes-Einstein relation gives :math:`D = k_BT/(6\pi\eta R)` for a sphere of radius
:math:`R` in a medium of viscosity :math:`\eta`.


Ornstein-Uhlenbeck Process (Inertial Brownian Motion)
------------------------------------------------------

When inertial effects are retained (finite particle mass :math:`m`), the full Langevin
equation :eq:`langevin_full` with :math:`F_\mathrm{ex} = 0` reads:

.. math::
   :label: OU_langevin

   m\ddot{x}(t) \;=\; -m\gamma\,\dot{x}(t) + \eta(t)

The **velocity** :math:`v(t) = \dot{x}(t)` follows an Ornstein-Uhlenbeck process:

.. math::

   \dot{v}(t) \;=\; -\gamma\,v(t) + \tfrac{1}{m}\,\eta(t)

with relaxation rate :math:`\gamma` (the friction coefficient from :eq:`langevin_full`).
The momentum relaxation time is :math:`\tau_B = 1/\gamma`.

.. note::

   The **velocity** (not the position) is the OU variable here. The position
   :math:`x(t) = \int_0^t v(t')\,dt'` is the integral of the OU velocity, and its
   variance grows without bound (free diffusion at long times). A position-OU process
   — where a harmonic potential confines the particle — gives qualitatively different
   behaviour; see the Brownian oscillator sections below.

**Velocity covariance** (starting from :math:`v(0) = 0`):

.. math::

   \mathrm{Cov}\!\left[v(t),\, v(t')\right]
   \;=\; D\gamma\!\left(e^{-\gamma|t-t'|} - e^{-\gamma(t+t')}\right)

**Transport coefficient** (from the Green-Kubo relation :eq:`J_greenkubo`):

.. math::
   :label: J_OU

   J(t) \;=\; 2D\!\left(1 - e^{-\gamma t}\right)^2

**Physical behaviour**:

- At short times (:math:`t \ll 1/\gamma`): :math:`J \approx 2D\gamma^2 t^2`
  (ballistic regime — inertia dominates and displacement variance grows as :math:`t^3`).
- At long times (:math:`t \gg 1/\gamma`): :math:`J \to 2D` (diffusive regime —
  velocity correlations have fully decorrelated, recovering free Brownian motion).
- The crossover occurs at the momentum relaxation time :math:`\tau_B = 1/\gamma`.

**XPCS application**: For colloidal particles in water, :math:`\tau_B \sim 10^{-9}` s
(far below XPCS time resolution), so the OU correction is negligible. It becomes
relevant for nanoparticles, proteins, or ultrafast XPCS at next-generation sources.
In the overdamped limit (:math:`\gamma \to \infty` with :math:`D` fixed),
:math:`J \to 2D` instantly, recovering the Wiener process. Conversely, the overdamped
Brownian oscillator (below) reduces to this OU result in the limit
:math:`\omega_0 \to 0`.


Brownian Oscillator (Underdamped)
----------------------------------

A particle in a **harmonic trap with weak damping** (:math:`\gamma < 2\omega_0`) exhibits
oscillatory dynamics before relaxing:

.. math::
   :label: oscillator_underdamped

   m\ddot{x} + m\gamma\dot{x} + m\omega_0^2 x \;=\; \eta(t)

Define the **shifted frequency** :math:`\omega_s = \sqrt{\omega_0^2 - \gamma^2/4}`.

**Transport coefficient** (underdamped):

.. math::
   :label: J_underdamped

   J(t) \;=\; \frac{2D\gamma^2}{\omega_s^2}\,
     e^{-\gamma t}\sin^2\!\left(\omega_s t\right)

**Physical behaviour**:

- Oscillates at frequency :math:`\omega_s` (ringing).
- Envelope decays at rate :math:`\gamma` (damping).
- Maximum :math:`J_\mathrm{max} = 2D\gamma^2/\omega_s^2` at :math:`\tau = \pi/(2\omega_s)`.

**XPCS application**: Models colloidal particles in strongly viscoelastic media (gels, glasses
near the Lindemann criterion) where particles oscillate within cages before escaping.


Brownian Oscillator (Overdamped)
---------------------------------

When damping is strong (:math:`\gamma > 2\omega_0`), define the **shifted decay rate**
:math:`\gamma_s = \sqrt{\gamma^2 - 4\omega_0^2}`.

**Transport coefficient** (overdamped):

.. math::
   :label: J_overdamped

   J(t) \;=\; \frac{8D\gamma^2}{\gamma_s^2}\,
     e^{-\gamma t}\sinh^2\!\left(\tfrac{1}{2}\gamma_s t\right)

**Physical behaviour**:

- Monotone growth then decay (no oscillations).
- At very long times: :math:`J \to 0` (particle becomes truly localized if the trap dominates).
- Equivalent to superposition of two exponential relaxation modes.

**XPCS application**: Models particles in dense suspensions or gels where the cage effect
completely suppresses oscillation but limits long-time diffusion.


Comparison Table
----------------

The following table summarizes the four processes:

.. list-table::
   :header-rows: 1
   :widths: 22 28 25 25

   * - Process
     - :math:`J(t)`
     - Long-time limit
     - Physical regime
   * - Wiener
     - :math:`2D`
     - :math:`2D` (free diffusion)
     - Dilute suspension
   * - Ornstein-Uhlenbeck
     - :math:`2D\left(1-e^{-\gamma\tau}\right)^2`
     - :math:`2D` (free diffusion)
     - Inertial effects (:math:`t \sim \tau_B`)
   * - Brown. Osc. (underdamped)
     - :math:`\frac{2D\gamma^2}{\omega_s^2}e^{-\gamma\tau}\sin^2(\omega_s\tau)`
     - 0 (localized)
     - Colloidal crystal, gel
   * - Brown. Osc. (overdamped)
     - :math:`\frac{8D\gamma^2}{\gamma_s^2}e^{-\gamma\tau}\sinh^2(\frac{\gamma_s\tau}{2})`
     - 0 (localized)
     - Dense glass, arrested gel

**Variance of displacement** :math:`\mathrm{Var}[x(t_2)-x(t_1)] = \int_{t_1}^{t_2} J(t')dt'`:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Process
     - :math:`\mathrm{Var}[x(t_2)-x(t_1)]`
   * - Wiener
     - :math:`2D\tau`
   * - OU
     - :math:`2D\!\left[\tau - \frac{2}{\gamma}(1-e^{-\gamma\tau}) + \frac{1}{2\gamma}(1-e^{-2\gamma\tau})\right]`
   * - Brown. Osc. (underdamped)
     - :math:`\frac{D\gamma^2}{\omega_s^2\omega_0^2}\left[2\omega_0^2\tau - 2\gamma(1-e^{-\gamma\tau}) + \frac{\gamma^2+\omega_s^2}{\omega_s}\sin(2\omega_s\tau)e^{-\gamma\tau} + \cdots\right]`
   * - Brown. Osc. (overdamped)
     - Analogous expression with :math:`\sinh` terms


Non-Gaussian Corrections
-------------------------

The Langevin framework predicts **Gaussian** displacement statistics. Deviations — measured
by the **non-Gaussian parameter**:

.. math::

   \alpha_2(t) \;=\; \frac{\langle\Delta x^4\rangle}{3\langle\Delta x^2\rangle^2} - 1

— signal heterogeneous dynamics, multi-population behaviour, or strongly anharmonic
potentials. For colloidal suspensions near the yielding transition (He et al. PNAS 2025):

- **Repulsive suspensions**: :math:`\alpha_2 \approx 0` (Gaussian, homodyne model valid).
- **Attractive suspensions**: :math:`\alpha_2 \gg 0` (power-law tails, heterodyne model needed).

This non-Gaussian criterion determines whether the homodyne or heterodyne formalism
should be applied to a given experiment.


.. seealso::

   - :ref:`theory_transport_coefficient` — general theory of :math:`J(t)`
   - :ref:`theory_heterodyne_scattering` — multi-component extension
   - :ref:`theory_yielding_dynamics` — experimental context
