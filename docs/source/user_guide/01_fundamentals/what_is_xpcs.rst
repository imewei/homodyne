.. _what_is_xpcs:

What is X-ray Photon Correlation Spectroscopy?
==============================================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- What XPCS measures and why coherent X-rays are essential
- How speckle patterns encode dynamics
- What two-time correlation functions reveal about particle motion
- Why XPCS is particularly powerful for soft matter and complex fluids

---

Introduction
------------

X-ray Photon Correlation Spectroscopy (XPCS) is a synchrotron-based technique
that probes the dynamics of materials at nanometer length scales and
microsecond-to-hour timescales. It is the X-ray analogue of Dynamic Light
Scattering (DLS), but with access to much shorter wavelengths and consequently
much smaller spatial scales.

XPCS is particularly suited for studying:

- Colloidal suspensions and nanoparticle dynamics
- Polymer melts and solutions
- Gels and soft glasses undergoing aging
- Complex fluids under shear (rheology-XPCS)
- Structural dynamics near phase transitions

The Coherent X-ray Requirement
--------------------------------

Conventional X-ray scattering (SAXS, WAXS) uses partially coherent beams and
measures ensemble-averaged scattering intensities. The result is a smooth,
time-averaged diffraction pattern that carries structural but not dynamical
information.

XPCS uses a **fully coherent X-ray beam**, produced by undulator sources at
third- and fourth-generation synchrotrons. When a coherent beam illuminates a
disordered sample, the scattered intensity forms a **speckle pattern**: a
seemingly random but reproducible arrangement of bright and dark spots.

.. note::

   A speckle pattern is not noise. It is a deterministic fingerprint of the
   instantaneous microscopic configuration of scatterers in the beam. As
   particles move, the speckle pattern evolves in a measurable way.

The key insight is that the **time evolution of the speckle pattern encodes the
dynamics** of the underlying particles. Fast-moving particles cause the speckle
pattern to fluctuate rapidly; slow particles produce slow fluctuations; frozen
structures produce a static speckle pattern.

From Speckles to Correlation Functions
-----------------------------------------

XPCS quantifies speckle dynamics through **intensity correlation functions**.
A pixel-area detector records a time series of speckle images. For each pixel
at scattering vector **q**, the time-autocorrelation of the intensity is computed:

.. math::

   g_2(\mathbf{q}, t) = \frac{\langle I(\mathbf{q}, t_0) \, I(\mathbf{q}, t_0 + t) \rangle}{\langle I(\mathbf{q}, t_0) \rangle^2}

where :math:`t` is the delay time and the angle brackets denote time averages.
For an ergodic system in equilibrium, :math:`g_2` depends only on the delay
:math:`t`, not on the reference time :math:`t_0`.

The Siegert relation connects :math:`g_2` to the normalized intermediate
scattering function :math:`g_1`:

.. math::

   g_2(\mathbf{q}, t) = 1 + \beta \, |g_1(\mathbf{q}, t)|^2

where :math:`\beta` is the **speckle contrast** (0 to 1), determined by the
coherence properties of the beam and the detector pixel size relative to the
speckle size.

Two-Time Correlation Functions
---------------------------------

The standard :math:`g_2(t)` analysis assumes the system is **stationary**:
dynamics do not change with time. Many real systems violate this assumption:

- Gels and glasses undergo slow aging
- Systems under shear may show transient rheological responses
- Driven systems (e.g., cyclic loading) show periodic dynamics

The **two-time correlation function** :math:`C_2(t_1, t_2)` resolves this
limitation by correlating intensities at two absolute times :math:`t_1` and
:math:`t_2` rather than a single lag:

.. math::

   C_2(\mathbf{q}, t_1, t_2) = \frac{\langle I(\mathbf{q}, t_1) \, I(\mathbf{q}, t_2) \rangle_\text{speckle}}{\langle I(\mathbf{q}, t_1) \rangle \, \langle I(\mathbf{q}, t_2) \rangle}

The averaging is performed over pixels at equivalent :math:`|\mathbf{q}|` and
azimuthal angle :math:`\phi`, not over time. This preserves the full
non-stationary dynamics.

For a stationary system, :math:`C_2` depends only on :math:`|t_2 - t_1|` and
reduces to the standard :math:`g_2`. For a non-stationary system, the
:math:`C_2` matrix has structure away from the main diagonal that reveals
how dynamics change with time.

.. note::

   Homodyne works with **two-time correlation functions** :math:`C_2(t_1, t_2)`
   as its primary input. The raw data from APS and APS-U beamlines is stored
   in HDF5 files with the full :math:`(n_\phi, n_{t_1}, n_{t_2})` array.

Connection to Particle Dynamics
-----------------------------------

The intermediate scattering function :math:`c_1` is determined by the
**transport coefficient** :math:`J(t)`, which measures the instantaneous rate of
growth of the mean-squared displacement at time :math:`t`
(see :ref:`theory_transport_coefficient`):

.. math::

   c_1(\mathbf{q}, t_1, t_2)
   = \exp\!\left(-\frac{q^2}{2}\int_{t_1}^{t_2} J(t')\,dt'\right)

The integral :math:`\int J\,dt'` equals the variance of the net particle
displacement and must in general be evaluated **numerically**.

For **normal (Brownian) diffusion** with constant diffusion coefficient
:math:`D`, the transport coefficient is :math:`J = 2D` and the equilibrium
single-time result simplifies to:

.. math::

   g_1(q, \tau) = \exp\!\left(-D q^2 \tau\right)

For **anomalous diffusion**, homodyne models the time-dependent diffusion
coefficient as :math:`D(t) = D_0\,t^\alpha + D_\text{offset}`, which modifies
the integral and produces slower (:math:`\alpha < 0`, sub-diffusion) or faster
(:math:`\alpha > 0`, super-diffusion) decay of correlations. The case
:math:`\alpha = 0` recovers normal diffusion.

For particles under **laminar shear flow** (Taylor-Couette geometry), an
additional **sinc-squared** modulation arises from integrating the velocity
field across the gap (see :ref:`theory_homodyne_scattering`):

.. math::

   |c_1|^2 =
   \exp\!\left(-q^2\!\int_{t_1}^{t_2} J(t')\,dt'\right)
   \;\times\;
   \mathrm{sinc}^2\!\left(\frac{q h\,\cos(\phi - \phi_0)\;\Gamma(t_1, t_2)}{2\pi}\right)

where :math:`\phi` is the azimuthal angle, :math:`h` is the gap distance,
:math:`\Gamma(t_1, t_2) = \int_{t_1}^{t_2}\dot\gamma(t)\,dt` is the
accumulated shear strain, and :math:`\mathrm{sinc}(x) = \sin(\pi x)/(\pi x)`
is the normalized sinc function (NumPy/JAX convention). The sinc-squared factor
produces a characteristic angular dependence that is the signature of shear.

Homodyne Detection
----------------------

Homodyne XPCS uses only the **scattered beam** from the sample, without mixing
in a reference beam. The measured correlation function is therefore the full
:math:`C_2` described above.

The name "homodyne" distinguishes it from "heterodyne" detection, where the
scattered beam is mixed with a reference beam (local oscillator). In the
homodyne case, the Siegert relation applies directly, and:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \beta(\phi) \cdot |c_1(\phi, t_1, t_2)|^2

where :math:`\beta(\phi)` is the per-angle speckle contrast. In practice,
homodyne fits this as:

.. math::

   C_2(\phi, t_1, t_2) = \text{offset}(\phi) + \beta(\phi) \cdot |c_1(\phi, t_1, t_2)|^2

where :math:`\text{offset}(\phi)` accounts for incoherent background
(ideally 1.0). This is the **model equation** implemented in homodyne.

Why XPCS for Soft Matter?
-----------------------------

XPCS occupies a unique niche in the experimental toolkit:

.. list-table:: XPCS vs Complementary Techniques
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Technique
     - Length Scale
     - Timescale
     - Sample Requirements
     - Dynamics Accessible
   * - DLS
     - 10 nm – 10 µm
     - µs – hours
     - Dilute, transparent
     - Diffusion, flow
   * - NMR relaxometry
     - 0.1 – 10 nm
     - ns – ms
     - Soluble compounds
     - Local motion
   * - Neutron spin echo
     - 1 – 100 nm
     - ns – µs
     - Deuterated samples
     - Collective modes
   * - **XPCS**
     - **1 – 1000 nm**
     - **µs – hours**
     - **Any (bulk capable)**
     - **Diffusion, flow, aging**

Key advantages of XPCS:

- **No dilution required**: can study concentrated dispersions, gels, glasses
- **Spatial selectivity**: probes a specific length scale via :math:`q` selection
- **Azimuthal resolution**: resolves anisotropic dynamics (e.g., shear flow direction)
- **Non-invasive**: no fluorescent labels or contrast agents needed
- **In situ compatible**: works with pressure cells, shear cells, furnaces

Experimental Setup
---------------------

A schematic XPCS beamline consists of:

1. **Undulator source**: produces coherent X-rays (typically 8–25 keV)
2. **Monochromator**: selects a narrow energy bandwidth (ΔE/E ~ 10⁻⁴)
3. **Coherence aperture (pinhole)**: defines the transversely coherent beam area
4. **Sample stage**: with optional environments (Couette cell, pressure, cryostat)
5. **Area detector**: fast pixel detector (EIGER, JUNGFRAU, etc.) with frame rates
   from Hz to kHz

The coherence length of the beam and the detector pixel solid angle together
determine the speckle contrast :math:`\beta`.

.. note::

   At APS and APS-U (Advanced Photon Source Upgrade), XPCS data is stored in
   HDF5 files following the ``NeXus`` or APS-specific formats. Homodyne
   supports both the APS legacy format and the APS-U new format.

---

See Also
---------

- :doc:`homodyne_overview` — What the homodyne package does
- :doc:`analysis_modes` — Static vs laminar flow mode choice
- :doc:`parameter_guide` — Physical meaning of each fitted parameter
- :ref:`theory` — Mathematical derivation of the model equations
