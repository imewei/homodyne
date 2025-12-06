Analysis Modes
==============

Homodyne provides two analysis modes for XPCS data: **Static** mode for equilibrium
systems and **Laminar Flow** mode for systems under shear. This section documents
the physical parameters, bounds, and usage for each mode.

Mode Overview
-------------

.. list-table:: Analysis Mode Comparison
   :header-rows: 1
   :widths: 20 15 40 25

   * - Mode
     - Physical Parameters
     - Description
     - Typical Applications
   * - Static
     - 3
     - Anomalous diffusion without flow
     - Colloidal suspensions, gels, glasses
   * - Laminar Flow
     - 7
     - Full nonequilibrium with shear
     - Rheology, microfluidics, active matter

Static Mode
-----------

Static mode analyzes systems at equilibrium without macroscopic flow. The correlation
function simplifies to:

.. math::

   c_2(t_1, t_2) = \text{offset} + \text{contrast} \times \exp\left[-q^2 J(t_1, t_2)\right]

Physical Parameters (3)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Static Mode Parameters
   :header-rows: 1
   :widths: 15 15 15 55

   * - Parameter
     - Symbol
     - Units
     - Description
   * - D0
     - :math:`D_0`
     - Varies with :math:`\alpha`
     - Baseline diffusion coefficient. For :math:`\alpha = 0`, units are [length :sup:`2` /time].
   * - alpha
     - :math:`\alpha`
     - dimensionless
     - Diffusion scaling exponent. :math:`\alpha = 0` for normal diffusion, :math:`\alpha < 0` for subdiffusion, :math:`\alpha > 0` for superdiffusion.
   * - D_offset
     - :math:`D_{\text{offset}}`
     - [length :sup:`2` /time]
     - Additive offset for baseline correction. Can be negative for jammed/arrested systems.

Parameter Bounds (Static Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Static Mode Default Bounds
   :header-rows: 1
   :widths: 20 20 20 40

   * - Parameter
     - Min
     - Max
     - Notes
   * - D0
     - 1e2
     - 1e5
     - Realistic range for colloidal systems
   * - alpha
     - -2.0
     - 2.0
     - Covers subdiffusion through superdiffusion
   * - D_offset
     - -1e5
     - 1e5
     - Negative values allowed for arrested systems

Diffusion Integral
~~~~~~~~~~~~~~~~~~

The diffusion integral is computed as:

.. math::

   J(t_1, t_2) = \frac{D_0}{1+\alpha}\left(t_2^{1+\alpha} - t_1^{1+\alpha}\right) + D_{\text{offset}}(t_2 - t_1)

.. note::
   In code this integral is evaluated numerically with a cumulative trapezoid on the
   discrete time grid (see :mod:`homodyne.core.physics_nlsq` and :mod:`homodyne.core.physics_cmc`).
   The integral between two frames uses the absolute difference of cumulative trapezoid
   sums; ``dt`` scaling is applied via the precomputed physics factors.

Laminar Flow Mode
-----------------

Laminar flow mode analyzes systems under shear with the full nonequilibrium correlation
function:

.. math::

   c_2(\phi, t_1, t_2) = \text{offset} + \text{contrast} \times c_1^{\text{diff}} \times \left[c_1^{\text{shear}}\right]^2

where the shear contribution introduces angular anisotropy.

Physical Parameters (7)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Laminar Flow Mode Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Symbol
     - Units
     - Description
   * - D0
     - :math:`D_0`
     - Varies with :math:`\alpha`
     - Baseline diffusion coefficient
   * - alpha
     - :math:`\alpha`
     - dimensionless
     - Diffusion scaling exponent
   * - D_offset
     - :math:`D_{\text{offset}}`
     - [length :sup:`2` /time]
     - Diffusion additive offset
   * - gamma_dot_t0
     - :math:`\dot{\gamma}_0`
     - Varies with :math:`\beta`
     - Baseline shear rate
   * - beta
     - :math:`\beta`
     - dimensionless
     - Shear rate scaling exponent
   * - gamma_dot_t_offset
     - :math:`\dot{\gamma}_{\text{offset}}`
     - [s :sup:`-1`]
     - Shear rate additive offset
   * - phi0
     - :math:`\phi_0`
     - [degrees]
     - Flow direction angle offset

Parameter Bounds (Laminar Flow Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Laminar Flow Mode Default Bounds
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Min
     - Max
     - Notes
   * - D0
     - 1e2
     - 1e5
     - Realistic diffusion range
   * - alpha
     - -2.0
     - 2.0
     - Anomalous diffusion exponent
   * - D_offset
     - -1e5
     - 1e5
     - Allows negative for arrested systems
   * - gamma_dot_t0
     - 1e-6
     - 0.5
     - Realistic shear rates for XPCS
   * - beta
     - -2.0
     - 2.0
     - Shear rate scaling exponent
   * - gamma_dot_t_offset
     - -0.1
     - 0.1
     - Allows small negative offsets
   * - phi0
     - -10.0
     - 10.0
     - Tight bounds for MCMC convergence (degrees)

Shear Integral
~~~~~~~~~~~~~~

The shear integral is computed as:

.. math::

   \Gamma(t_1, t_2) = \frac{\dot{\gamma}_0}{1+\beta}\left(t_2^{1+\beta} - t_1^{1+\beta}\right) + \dot{\gamma}_{\text{offset}}(t_2 - t_1)

.. note::
   As with diffusion, the code evaluates this with a cumulative trapezoid and smooth
   absolute differences between cumulative sums to cover multi-step intervals robustly
   in both NLSQ (meshgrid) and CMC (element-wise) paths.


Parameter Bounds and Priors (NLSQ & CMC)
----------------------------------------

- **Where bounds come from:** Both NLSQ and CMC read bounds from the ``parameter_space``
  section of the YAML. If a parameter is omitted, defaults from ``ParameterManager`` are
  used; if those are missing, the fallbacks are:
  - ``contrast``: [0.0, 1.0]
  - ``offset``: [0.5, 1.5]
  - other parameters: [0.0, 1.0]
- **NLSQ usage:** Bounds are used for deterministic optimization only; NLSQ does not
  sample priors.
- **CMC usage:** Bounds are used to build priors via ``ParameterSpace`` (see
  :mod:`homodyne.config.parameter_space` and :mod:`homodyne.optimization.cmc.priors`).
  Per-angle parameters ``contrast_i`` / ``offset_i`` are always included; if you only
  supply a base ``contrast``/``offset`` bound or prior, it applies to all angles.
- **Default priors when not specified:**
  - Type: ``TruncatedNormal``
  - Location: midpoint of the bound interval
  - Scale: one quarter of the interval width
  If no prior spec exists at all for a parameter, the runtime fallback is
  ``Uniform(min, max)``.
- **Supported prior types in CMC:** ``TruncatedNormal``, ``Uniform``, ``LogNormal``,
  ``HalfNormal``, ``Normal``, ``BetaScaled`` (Beta on [min, max]). All require finite
  bounds; ``BetaScaled`` computes the Beta concentrations from ``mu``/``sigma`` on the
  scaled interval.

Example YAML snippet
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: laminar_flow
     bounds:
       - name: D0
         min: 1e2
         max: 1e5
         prior_mu: 1e3
         prior_sigma: 1e3
         type: TruncatedNormal
       - name: alpha
         min: -2.0
         max: 2.0
         prior_mu: 0.0
         prior_sigma: 0.5
         type: Normal
       - name: contrast        # applies to all contrast_i
         min: 0.0
         max: 1.0
         prior_mu: 0.5
         prior_sigma: 0.2
         type: BetaScaled

Per-Angle Scaling
-----------------

Since v2.4.0, per-angle scaling is mandatory for both modes. This accounts for
instrumental variations across different detector positions.

Scaling Parameters (Per Angle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each azimuthal angle :math:`\phi_i`, two scaling parameters are fitted:

.. list-table:: Per-Angle Scaling Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Symbol
     - Range
     - Description
   * - contrast_i
     - :math:`\beta_i`
     - [0.0, 1.0]
     - Per-angle contrast (instrumental coherence)
   * - offset_i
     - :math:`c_{0,i}`
     - [0.5, 1.5]
     - Per-angle baseline offset

Total Parameter Count
~~~~~~~~~~~~~~~~~~~~~

The total number of parameters depends on the number of azimuthal angles :math:`n_\phi`:

.. math::

   N_{\text{total}} = N_{\text{physical}} + 2 \times n_\phi

**Examples:**

* Static mode with 3 angles: :math:`3 + 2 \times 3 = 9` parameters
* Laminar flow mode with 3 angles: :math:`7 + 2 \times 3 = 13` parameters
* Laminar flow mode with 5 angles: :math:`7 + 2 \times 5 = 17` parameters

Parameter Ordering
~~~~~~~~~~~~~~~~~~

For MCMC initialization, parameters must follow a specific ordering:

1. Per-angle contrast: ``contrast_0, contrast_1, ..., contrast_{n_phi-1}``
2. Per-angle offset: ``offset_0, offset_1, ..., offset_{n_phi-1}``
3. Physical parameters: ``D0, alpha, D_offset, [gamma_dot_t0, beta, gamma_dot_t_offset, phi0]``

This ordering is critical for proper MCMC initialization with NumPyro.

Mode Selection Guidelines
-------------------------

When to Use Static Mode
~~~~~~~~~~~~~~~~~~~~~~~

* Equilibrium colloidal suspensions
* Gels and glasses near arrest transition
* Systems without macroscopic flow
* Initial exploratory analysis

When to Use Laminar Flow Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Rheology experiments with shear cells
* Microfluidic flow measurements
* Active matter under flow
* Systems with time-dependent shear rates

Physical Constraints
--------------------

The optimization respects physical constraints:

**Positivity Constraints:**

* :math:`D_0 > 0`: Baseline diffusion must be positive
* :math:`\dot{\gamma}_0 \geq 0`: Non-negative baseline shear rate
* :math:`\text{contrast}_i \in [0, 1]`: Physical contrast bounds

**Scaling Exponent Bounds:**

* :math:`|\alpha| \leq 2`: Physically reasonable diffusion scaling
* :math:`|\beta| \leq 2`: Physically reasonable shear scaling

**Angular Constraints:**

* :math:`\phi_0 \in [-10^\circ, 10^\circ]`: Tight bounds for MCMC convergence

Configuration Example
---------------------

YAML configuration for laminar flow mode with 3 angles:

.. code-block:: yaml

   physics:
     mode: laminar_flow  # or "static"

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset
       - gamma_dot_t0
       - beta
       - gamma_dot_t_offset
       - phi0
     values:
       - 1000.0    # D0
       - 0.0       # alpha (normal diffusion)
       - 0.0       # D_offset
       - 0.01      # gamma_dot_t0
       - 0.0       # beta
       - 0.0       # gamma_dot_t_offset
       - 0.0       # phi0 (degrees)

   # Per-angle scaling is automatically enabled (v2.4.0+)

See :doc:`../configuration/templates` for complete configuration templates.
