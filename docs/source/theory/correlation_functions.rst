.. _theory_correlation_functions:

Correlation Functions in XPCS
==============================

X-ray Photon Correlation Spectroscopy (XPCS) measures intensity fluctuations in the speckle
pattern formed at the detector. These fluctuations encode the collective dynamics of the
scattering sample. This page derives the two-time correlation function :math:`c_2` from first
principles and explains why the equilibrium approximation :math:`g_2(q,\tau)` is insufficient
for non-stationary systems.



Position Density and Scattered Field
-------------------------------------

For a sample containing :math:`N` scattering centres at positions :math:`\mathbf{r}_j(t)`,
the **position density** in Fourier space is:

.. math::
   :label: rho_q

   \rho(\mathbf{q}, t) \;=\; \sum_{j=1}^N f_j \exp\!\left(i\mathbf{q}\cdot\mathbf{r}_j(t)\right)

where :math:`f_j` is the form factor of particle :math:`j` and :math:`\mathbf{q}` is the
momentum transfer vector with :math:`|\mathbf{q}| = q = 4\pi\sin(\theta)/\lambda`.

The **scattered electric field** at the detector is proportional to :math:`\rho`:

.. math::
   :label: E_scattered

   E(\mathbf{q}, t) \;\propto\; \rho(\mathbf{q}, t)

and the measured **intensity** is:

.. math::
   :label: I_intensity

   I(\mathbf{q}, t) \;=\; |E(\mathbf{q}, t)|^2
   \;=\; \left|\sum_j f_j e^{i\mathbf{q}\cdot\mathbf{r}_j(t)}\right|^2


First-Order Correlation Function
---------------------------------

The **normalized first-order correlation function** is:

.. math::
   :label: c1_definition

   c_1(\mathbf{q}, t_1, t_2)
   \;=\; \frac{\langle E^*(\mathbf{q}, t_1)\,E(\mathbf{q}, t_2)\rangle}
              {\sqrt{\langle I(\mathbf{q}, t_1)\rangle\,\langle I(\mathbf{q}, t_2)\rangle}}

For a Gaussian process (valid for large :math:`N` by the central limit theorem), :math:`c_1`
depends only on the **transport coefficient** :math:`J(t)` and the mean drift:

.. math::
   :label: c1_general

   c_1(\mathbf{q}, t_1, t_2)
   \;=\; \exp\!\left(-\frac{q^2}{2}\int_{t_1}^{t_2} J(t')\,dt'\right)
   \;\times\;
   \exp\!\left(iq\int_{t_1}^{t_2} \langle v(t')\rangle\,dt'\right)

The first factor is the **Debye-Waller-like** diffusive decay; the second is a phase shift
from the mean flow. This factorization is exact for Gaussian displacement statistics.

**Factorization into internal and external contributions**:

.. math::
   :label: c1_factorization

   c_1 \;=\; c_1^\mathrm{(in)} \;\times\; c_1^\mathrm{(ex)}

where the internal (diffusive) part is:

.. math::

   c_1^\mathrm{(in)}(\mathbf{q}, t_1, t_2)
   \;=\; \exp\!\left(-\frac{q^2}{2}\mathcal{D}(t_1, t_2)\right),
   \quad \mathcal{D}(t_1,t_2) = \int_{t_1}^{t_2} J(t')\,dt'

and the external (drift) part is:

.. math::

   c_1^\mathrm{(ex)}(\mathbf{q}, t_1, t_2)
   \;=\; \exp\!\left(iq\int_{t_1}^{t_2}\langle v(t')\rangle\,dt'\right)


Second-Order Correlation Function
----------------------------------

The **normalized second-order (intensity) correlation function** is:

.. math::
   :label: c2_definition

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; \frac{\langle I(\mathbf{q}, t_1)\,I(\mathbf{q}, t_2)\rangle}
              {\langle I(\mathbf{q}, t_1)\rangle\,\langle I(\mathbf{q}, t_2)\rangle}

This is the quantity measured directly in XPCS experiments. Each pixel in the 2D detector
contributes one time series :math:`I(\mathbf{q}, t)`, and the correlation is accumulated
as a function of the two absolute times :math:`t_1` and :math:`t_2`.

.. note::

   :math:`c_2` is dimensionless and satisfies :math:`c_2 \geq 1` for classical fluctuations
   (Cauchy-Schwarz inequality). The baseline :math:`c_2 \to 1` as :math:`|t_2 - t_1| \to \infty`
   when fluctuations decorrelate.


Siegert Relation
----------------

For a **Gaussian field** (single-mode, thermal light statistics), Wick's theorem gives the
**Siegert relation**:

.. math::
   :label: siegert

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \beta(t_1, t_2)\,|c_1(\mathbf{q}, t_1, t_2)|^2

where :math:`\beta(t_1, t_2) \in (0, 1]` is the **speckle contrast** (also called the
optical coherence parameter). For a perfectly coherent X-ray beam and ideal single-mode
detection :math:`\beta = 1`; realistic experiments yield :math:`\beta \approx 0.05–0.8`.

The Siegert relation is the foundation of all homodyne models implemented in this package.
Combining :eq:`c1_general` and :eq:`siegert` gives the general homodyne result:

.. math::
   :label: c2_general_homodyne

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \beta(t_1, t_2)
         \exp\!\left(-q^2 \int_{t_1}^{t_2} J(t')\,dt'\right)

The phase (drift) term drops out of :math:`|c_1|^2` for homodyne detection. Heterodyne
detection retains it — see :ref:`theory_heterodyne_scattering`.

.. note::

   In homodyne's implementation, the integral :math:`\int J(t')\,dt'` is computed
   as :math:`\int D(t')\,dt'` using cumulative trapezoidal integration, where
   :math:`D(t) = D_0 t^\alpha + D_\mathrm{offset}` is homodyne's parameterization
   with :math:`D_0 = 2 D_\mathrm{SE}`. See :ref:`theory_transport_coefficient`
   for the convention details.


Equilibrium Approximation: g₂(q, τ)
-------------------------------------

When the sample is in **thermal equilibrium**, the dynamics are stationary and :math:`c_2`
depends only on the **lag time** :math:`\tau = t_2 - t_1`, not on the absolute time :math:`t_1`.
The standard equilibrium correlation function is:

.. math::
   :label: g2_equilibrium

   g_2(q, \tau) \;=\; \frac{\langle I(q, t)\,I(q, t+\tau)\rangle}{\langle I(q,t)\rangle^2}
   \;=\; 1 + \beta\,e^{-2\Gamma \tau}

For Brownian diffusion, :math:`\Gamma = Dq^2` and the decay is a simple exponential.

**Why equilibrium fails for yielding systems**: During a yielding transition, the sample
properties change on timescales comparable to (or shorter than) the experiment duration.
The diagonal average :math:`g_2(q, \tau) = \langle c_2(q, t_1, t_1+\tau)\rangle_{t_1}`
washes out genuine transient behaviour. The full :math:`c_2(q, t_1, t_2)` matrix is required.

.. warning::

   Using :math:`g_2(q, \tau)` analysis on a non-stationary sample produces
   **artifact-corrupted parameters**: :math:`D` absorbs time-averaged heterogeneity,
   :math:`\beta` appears artificially depressed, and the functional form may no longer
   be a single exponential even when the underlying physics is simple.


Two-Time Correlation Matrix
----------------------------

In practice, the two-time correlation function is represented as a **matrix** indexed
by discrete time points :math:`(t_i, t_j)`:

.. math::

   c_2^{ij} \;=\; c_2(q, t_i, t_j), \qquad i, j \in \{1, \ldots, N_t\}

The matrix is **symmetric** (:math:`c_2^{ij} = c_2^{ji}`) and has :math:`c_2^{ii} = 1 + \beta`
on the diagonal (zero lag). The standard :math:`g_2(\tau)` corresponds to the average along
each off-diagonal at lag :math:`\tau = (j-i)\Delta t`.

Homodyne accesses this full matrix from the HDF5 data file loaded by
:class:`homodyne.data.XPCSDataLoader`.


Model Fitting
-------------

Given measured :math:`\{c_2^{ij}\}`, homodyne fits the theoretical model by minimizing:

.. math::

   \chi^2 = \sum_{i,j} w_{ij}\!\left[c_2^{ij,\mathrm{meas}} - c_2^{ij,\mathrm{model}}(\theta)\right]^2

where :math:`\theta = (D_0, \alpha, D_\mathrm{offset})` for static mode and
:math:`\theta = (D_0, \alpha, D_\mathrm{offset}, \dot{\gamma}_0, \beta_\gamma, \dot{\gamma}_\mathrm{offset}, \phi_0)`
for laminar flow mode.

The weights :math:`w_{ij}` are determined by the measurement variance (Poisson statistics
for photon counting). See :ref:`theory_computational_methods` for solver details.


.. seealso::

   - :ref:`theory_transport_coefficient` — definition and physical interpretation of :math:`J(t)`
   - :ref:`theory_homodyne_scattering` — full laminar-flow equation
   - :ref:`theory_heterodyne_scattering` — multi-component extension
   - :class:`homodyne.data.XPCSDataLoader` — HDF5 data loading
   - :mod:`homodyne.core.jax_backend` — JIT-compiled :math:`g_2` computation
