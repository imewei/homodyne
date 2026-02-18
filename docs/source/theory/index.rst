.. _theory_index:
.. _theory:

Theory & Physics
================

Homodyne implements the **transport coefficient framework** developed by He et al. (PNAS 2024, 2025)
for X-ray Photon Correlation Spectroscopy (XPCS) analysis of soft matter under nonequilibrium
conditions. This section provides the complete theoretical foundation underlying every computation
in the package.

The central quantity is the **transport coefficient** :math:`J(t)`, which connects microscopic
particle dynamics to macroscopic rheological observables via a generalized Green-Kubo relation.
From :math:`J(t)`, the package constructs the full two-time intensity correlation function
:math:`c_2(\vec{q}, t_1, t_2)` — avoiding the equilibrium assumption embedded in the standard
:math:`g_2(q, \tau)` representation.

Overview of Sections
--------------------

.. toctree::
   :maxdepth: 2
   :caption: Theory

   transport_coefficient
   correlation_functions
   homodyne_scattering
   heterodyne_scattering
   classical_processes
   yielding_dynamics
   anti_degeneracy
   computational_methods
   citations

Quick Physics Reference
-----------------------

**Static mode** (:math:`n_\mathrm{params} = 3`):

.. math::

   c_2(q, t_1, t_2) = 1 + \beta(t_1, t_2) \exp\!\left(-q^2 \int_{t_1}^{t_2} J(t)\,dt\right)

**Laminar flow mode** (:math:`n_\mathrm{params} = 7`):

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \beta(t_1, t_2)
     \exp\!\left(-q^2 \int_{t_1}^{t_2} J(t)\,dt\right)
     \operatorname{sinc}^2\!\left(\tfrac{1}{2} q h
       \int_{t_1}^{t_2} \dot{\gamma}(t) \cos\phi\,dt\right)

**Parameter table**:

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Symbol
     - Parameter name
     - Physical meaning
   * - :math:`D_0`
     - ``D0``
     - Diffusion prefactor (:math:`\text{Å}^2/\text{s}`)
   * - :math:`\alpha`
     - ``alpha``
     - Diffusion anomalous exponent (0 < α ≤ 1)
   * - :math:`D_\mathrm{offset}`
     - ``D_offset``
     - Constant diffusion background
   * - :math:`\dot{\gamma}_0`
     - ``gamma_dot_0``
     - Shear rate prefactor (:math:`\text{s}^{-1}`)
   * - :math:`\beta`
     - ``beta``
     - Shear rate power-law exponent
   * - :math:`\dot{\gamma}_\mathrm{offset}`
     - ``gamma_dot_offset``
     - Constant shear rate background
   * - :math:`\phi_0`
     - ``phi_0``
     - Azimuthal reference angle (rad)

See :ref:`theory_transport_coefficient` for derivation of :math:`J(t)`, and
:ref:`theory_homodyne_scattering` for the full laminar-flow equation.
