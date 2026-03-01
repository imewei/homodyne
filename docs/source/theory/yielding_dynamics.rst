.. _theory_yielding_dynamics:

Yielding Dynamics of Colloidal Suspensions
===========================================

This page summarizes the physics of yielding transitions in charged colloidal suspensions as
studied by He et al. (PNAS 2025) using the homodyne XPCS framework. Two fundamentally
distinct yielding mechanisms are identified — **ductile Andrade creep** in repulsive systems
and **brittle shear banding** in attractive systems — and connected to measurable signatures
in the two-time correlation function :math:`c_2`.



Overview: Repulsive vs Attractive Suspensions
----------------------------------------------

Charged colloidal suspensions near the glass or gel transition exhibit qualitatively different
rheological responses depending on the inter-particle potential:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Property
     - Repulsive (electrostatic)
     - Attractive (depletion/van der Waals)
   * - Microstructure
     - Ordered, charge-stabilized glass
     - Fractal gel network
   * - Yielding type
     - Ductile (gradual, homogeneous)
     - Brittle (abrupt, heterogeneous)
   * - Flow field
     - Homogeneous laminar flow
     - Shear banding
   * - :math:`c_2` signature
     - Homodyne + Andrade creep
     - Heterodyne + oscillations
   * - Non-affine motion
     - Gaussian (small)
     - Non-Gaussian (power-law tails)
   * - PNAS 2025 section
     - Section II
     - Section III–IV


Repulsive Suspensions: Andrade Creep
--------------------------------------

Under a step-stress protocol, repulsive colloidal glasses exhibit **Andrade creep**, first
identified in polycrystalline metals and now established in soft matter:

.. math::
   :label: andrade_creep

   \gamma(t) \;\propto\; t^{1/3}
   \quad \Rightarrow \quad
   \dot{\gamma}(t) \;\propto\; t^{-2/3}

The :math:`t^{1/3}` scaling arises from the cooperative, thermally-activated rearrangement
of particle cages under a sub-yield stress. Each rearrangement event relieves a small amount
of stored elastic energy, producing a decreasing strain rate.

**Mapping to homodyne parameters**:

.. math::

   \dot{\gamma}_0 > 0, \quad \beta_\gamma = -2/3, \quad \dot{\gamma}_\mathrm{offset} = 0

The corresponding shear integral is:

.. math::

   \Gamma(t_1, t_2)
   \;=\; \frac{\dot{\gamma}_0}{1/3}\!\left(t_2^{1/3} - t_1^{1/3}\right)
   \;=\; 3\dot{\gamma}_0\!\left(t_2^{1/3} - t_1^{1/3}\right)

**Diffusion during creep**: Simultaneously, the mean-squared displacement of particles
grows anomalously with an exponent :math:`\alpha < 1`:

.. math::

   \mathrm{MSD}(t) \;\propto\; t^\alpha, \quad \alpha \approx 0.3 \text{–} 0.5

This is sub-diffusion driven by caged motion — particles rattle within increasingly
disrupted cages but do not undergo long-range diffusion until the yield stress is exceeded.

**XPCS signature**: The two-time :math:`c_2` matrix shows a **diagonal ridge** whose width
grows as :math:`t^{1/3}` — the fingerprint of Andrade creep directly observable without
bulk rheometry.


Attractive Suspensions: Shear Banding and Delayed Yielding
------------------------------------------------------------

In attractive gels (formed by depletion or van der Waals forces), yielding is fundamentally
different:

**Delayed yielding**: The gel remains apparently solid until a critical accumulated strain
:math:`\gamma_c \approx 0.3` is reached, at which point it **abruptly fluidizes** ("brittle failure").

.. math::

   \gamma(t) \;\approx\; 0 \quad (t < t_\mathrm{yield}),
   \qquad
   \gamma(t) \;\gg\; 0 \quad (t > t_\mathrm{yield})

**Shear banding**: During and after yielding, the flow field becomes **heterogeneous**:
a fast-flowing band (near the rotor) coexists with a slow or static band (near the stator).
This breaks the laminar flow assumption.

**Resolidification**: After yielding, the sample may partially re-arrest in the static band
due to network reformation. This manifests as a non-monotone evolution of the correlation
function.


Bond Dynamics and Interfacial Layers
--------------------------------------

The microscopic origin of the two yielding mechanisms is revealed by tracking the **bond
lifetime distribution**:

- **Repulsive glass**: Cage lifetime :math:`\tau_\alpha` decreases monotonically under
  stress as bonds are thermally activated. No preferential bond breaking.

- **Attractive gel**: Network strands break **irreversibly** at the applied stress
  ("bond rupture" mechanism). The failed bonds concentrate at the interface between the
  two shear bands.

The **interfacial layer** (few-particle-diameter thickness) between bands is the locus of
all plastic deformation. XPCS probes this layer directly because the beam coherence length
(:math:`\sim`nm) is much smaller than the layer thickness (:math:`\sim \mu`m).


Non-Affine Displacements
-------------------------

Affine displacements follow the macroscopic strain field exactly. Non-affine displacements
are the residuals:

.. math::

   \delta\mathbf{u}_i(t)
   \;=\; \mathbf{u}_i(t) - \mathbf{E}(t)\cdot\mathbf{r}_i(0)

where :math:`\mathbf{E}(t)` is the macroscopic strain tensor. The distribution
:math:`P(\delta u)` characterizes the heterogeneity of the flow:

- **Gaussian** :math:`P(\delta u)`: homogeneous flow, homodyne XPCS model valid.
- **Power-law tails** in :math:`P(\delta u)`: shear banding, heterodyne model required.

For repulsive suspensions, He et al. PNAS 2025 find:

.. math::

   P(\delta u) \;\sim\; \exp(-\delta u^2 / 2\sigma^2)

confirming the Gaussian ansatz underlying the homodyne single-component model.

For attractive suspensions:

.. math::

   P(\delta u) \;\sim\; |\delta u|^{-\mu}, \quad \mu < 3

indicating a population of rare but large displacements ("particle avalanches") not
described by a single effective :math:`D`.


Connection to XPCS Measurements
---------------------------------

The mapping from microscopic dynamics to XPCS observables is:

**Repulsive (Andrade creep)**:

.. math::

   c_2^\mathrm{exp}(q, t_1, t_2)
   \;=\;
   1 + \beta\,
     e^{-q^2\mathcal{D}(t_1,t_2)}
     \mathrm{sinc}^2\!\left(\tfrac{qh\cos\phi}{2\pi}\Gamma(t_1,t_2)\right)

with :math:`\mathcal{D}` and :math:`\Gamma` reflecting the Andrade creep laws. This is
exactly the homodyne ``laminar_flow`` model implemented in this package.

**Attractive (shear banding)**: The measured :math:`c_2` matrix shows oscillatory patterns
described by the two-component heterodyne formula. The oscillation frequency increases
abruptly at :math:`t = t_\mathrm{yield}` as the fast band velocity jumps.

**Practical XPCS diagnostic criteria**:

1. If :math:`c_2(t_1, t_2)` is smooth and decays monotonically away from the diagonal:
   use homodyne ``laminar_flow`` mode.

2. If :math:`c_2(t_1, t_2)` shows oscillatory fringes: use heterodyne multi-component
   analysis (currently outside the scope of the homodyne package; see He et al. PNAS 2025).

3. If :math:`c_2` is featureless (no angle dependence): use ``static`` mode (no flow).


Summary of Physical Signatures
--------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Observable
     - Repulsive (Andrade)
     - Attractive (banding)
   * - Strain :math:`\gamma(t)`
     - :math:`\propto t^{1/3}` (smooth)
     - Step-like at :math:`t_\mathrm{yield}`
   * - :math:`c_2` matrix shape
     - Ridge, width :math:`\propto t^{1/3}`
     - Oscillatory fringes
   * - :math:`\alpha` (diffusion exp.)
     - :math:`0.3 < \alpha < 1` (sub-diffusive)
     - Multi-modal MSD
   * - Non-Gaussian param. :math:`\alpha_2`
     - :math:`\approx 0`
     - :math:`\gg 0`
   * - Analysis mode
     - ``laminar_flow``
     - Heterodyne (future work)


.. seealso::

   - He et al. PNAS 2024 — transport coefficient framework
   - He et al. PNAS 2025 — yielding dynamics paper
   - :ref:`theory_homodyne_scattering` — laminar flow model
   - :ref:`theory_heterodyne_scattering` — shear banding model
   - :ref:`theory_classical_processes` — sub-diffusion (Ornstein-Uhlenbeck)
   - :ref:`theory_citations` — full bibliography
