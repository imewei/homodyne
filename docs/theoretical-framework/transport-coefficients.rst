Transport Coefficients
======================

Overview
--------

Homodyne v2 parameterizes time-dependent transport properties using power-law functions to capture evolving dynamics in non-equilibrium soft matter systems. The three key time-dependent quantities are:

1. **Diffusion Coefficient** :math:`D(t)` - characterizes random Brownian motion
2. **Shear Rate** :math:`\dot{\gamma}(t)` - describes externally driven flow
3. **Angle Evolution** :math:`\phi(t)` - tracks orientation between flow and scattering vector

These parameterizations enable direct fitting of experimental two-time correlation functions to extract transport properties without assuming equilibrium or time-translation invariance.

**Physical Context:**

In non-equilibrium systems (e.g., relaxation after stress removal, shear banding, yielding), transport properties evolve in time as the system responds to external perturbations or internal rearrangements. Power-law parameterizations provide flexible functional forms that capture a wide range of dynamical behaviors while maintaining analytical tractability for optimization.


Time-Dependent Diffusion D(t)
------------------------------

Power-Law Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The time-dependent diffusion coefficient is parameterized as:

.. math::
   :label: diffusion_powerlaw

   D(t) = D_0 + D_{\text{offset}} \cdot t^\alpha

where:

- :math:`D_0` is the initial diffusion coefficient [:math:`\text{Å}^2/\text{s}`]
- :math:`\alpha` is the power-law exponent [dimensionless]
- :math:`D_{\text{offset}}` is the offset (can be positive, negative, or zero) [:math:`\text{Å}^2/\text{s}`]
- :math:`t` is time [s]

**Physical Interpretation:**

- :math:`D_0 > 0`: Intrinsic diffusivity at :math:`t = 0`, reflecting thermal fluctuations and particle mobility
- :math:`\alpha`: Governs temporal evolution
   - :math:`\alpha = 0`: Constant diffusion (equilibrium)
   - :math:`0 < \alpha < 1`: Sub-diffusive behavior (e.g., caging, crowding)
   - :math:`\alpha = 1`: Linear time-dependence (e.g., accelerating/decelerating systems)
   - :math:`\alpha > 1`: Super-diffusive or rapidly evolving dynamics
- :math:`D_{\text{offset}}`: Controls direction and magnitude of evolution
   - :math:`D_{\text{offset}} > 0`: Increasing diffusivity (e.g., fluidization, relaxation from jammed state)
   - :math:`D_{\text{offset}} < 0`: Decreasing diffusivity (e.g., aging, gelation, approaching arrest)
   - :math:`D_{\text{offset}} = 0`: Constant diffusion

Relation to Transport Coefficient J(t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamical transport coefficient :math:`J(t)` used in :doc:`core-equations` is related to diffusion by:

.. math::
   :label: transport_diffusion

   J(t) = \frac{d}{dt}\left[\text{Var}[x(t)]\right] = 2 \text{Cov}[x(t), v(t)]

For **standard 3D isotropic diffusion** in equilibrium:

.. math::

   J(t) = 6D

This factor of 6 arises from :math:`J_x = J_y = J_z = \frac{1}{3}J` with :math:`J = 2D` (1D transport coefficient).

For **non-equilibrium systems** with time-dependent diffusion:

.. math::

   J(t) \approx 6D(t) = 6\left(D_0 + D_{\text{offset}} \cdot t^\alpha\right)

This approximation holds when external forces and systematic interactions can be absorbed into an effective time-dependent diffusion coefficient.

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     bounds:
       - name: D0
         min: 100.0
         max: 1e5
       - name: alpha
         min: 0.0
         max: 2.0
       - name: D_offset
         min: -100.0
         max: 100.0

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset

**Typical Values:**

- **D0:** 100 - 100,000 Å²/s (system-dependent, higher for smaller particles)
- **alpha:** 0.0 - 2.0 (covers sub-diffusive to super-diffusive regimes)
- **D_offset:** -100 to +100 Å²/s (allows negative values for aging/arrest dynamics)

.. seealso::

   - :doc:`parameter-models` - Full parameter counting (3+2n static, 7+2n laminar)
   - :doc:`../user-guide/configuration` - Complete configuration examples


Time-Dependent Shear Rate γ̇(t)
-------------------------------

Power-Law Parameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~

The time-dependent shear rate is parameterized as:

.. math::
   :label: shear_powerlaw

   \dot{\gamma}(t) = \dot{\gamma}_0 + \dot{\gamma}_{\text{offset}} \cdot t^\beta

where:

- :math:`\dot{\gamma}_0` is the initial shear rate [:math:`\text{s}^{-1}`]
- :math:`\beta` is the power-law exponent [dimensionless]
- :math:`\dot{\gamma}_{\text{offset}}` is the offset [:math:`\text{s}^{-1}`]
- :math:`t` is time [s]

**Physical Interpretation:**

- :math:`\dot{\gamma}_0`: Shear rate at :math:`t = 0`, set by experimental conditions (rheometer applied shear)
- :math:`\beta`: Describes temporal evolution of shear
   - :math:`\beta = 0`: Constant shear (steady-state flow)
   - :math:`\beta < 0`: Decaying shear (relaxation after stress removal)
   - :math:`\beta > 0`: Increasing shear (e.g., creep deformation, accelerating flow)
- :math:`\dot{\gamma}_{\text{offset}}`: Magnitude and direction of shear evolution
   - Positive: Accelerating flow
   - Negative: Decelerating flow or relaxation
   - Zero: Constant shear rate

Application in Laminar Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the laminar flow model (Equation 13 from :doc:`core-equations`), the shear rate appears in the sinc² term:

.. math::

   \text{sinc}^2\left[\frac{1}{2}qh \int_{t_1}^{t_2} \dot{\gamma}(t) \cos\left(\phi(t)\right) \,\mathrm{d}t\right]

Substituting the power-law parameterization:

.. math::

   \int_{t_1}^{t_2} \dot{\gamma}(t) \,\mathrm{d}t = \dot{\gamma}_0 (t_2 - t_1) + \dot{\gamma}_{\text{offset}} \frac{t_2^{\beta+1} - t_1^{\beta+1}}{\beta + 1}

This integral captures the accumulated shear strain over the time interval :math:`[t_1, t_2]`.

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: "laminar_flow"
     bounds:
       # Shear parameters (laminar flow only)
       - name: gamma_dot_0
         min: 1e-6
         max: 0.5
       - name: beta
         min: 0.0
         max: 2.0
       - name: gamma_dot_offset
         min: -0.1
         max: 0.1

   initial_parameters:
     parameter_names:
       - gamma_dot_0
       - beta
       - gamma_dot_offset

**Typical Values:**

- **gamma_dot_0:** 1e-6 - 0.5 s⁻¹ (set by rheometer, system-dependent)
- **beta:** 0.0 - 2.0 (covers constant to rapidly evolving shear)
- **gamma_dot_offset:** -0.1 to +0.1 s⁻¹ (smaller magnitude than gamma_dot_0)

.. note::

   **Parameter Name Mapping (Config → Code):**

   - ``gamma_dot_0`` (config) → ``gamma_dot_t0`` (code)

.. seealso::

   - :doc:`core-equations` - Full laminar flow equation implementation
   - :doc:`parameter-models` - Laminar flow (7+2n) model details


Angle Evolution φ(t)
--------------------

Angle Parameterization
~~~~~~~~~~~~~~~~~~~~~~

The angle :math:`\phi(t)` between the shear/flow direction and the scattering wavevector :math:`\vec{q}` is parameterized through:

.. math::
   :label: angle_param

   \phi(t) = \phi_0

where:

- :math:`\phi_0` is the initial angle [degrees], typically constant during an experiment
- Range: :math:`-180° \leq \phi_0 \leq 180°` (automatically normalized to symmetric range)

**Physical Interpretation:**

- :math:`\phi_0 = 0°`: Scattering vector aligned with flow direction (maximum advective decorrelation)
- :math:`\phi_0 = 90°`: Scattering vector perpendicular to flow (no advective decorrelation, only diffusion)
- :math:`\phi_0` intermediate: Partial contribution from both diffusion and flow

**Anisotropic Behavior:**

The :math:`\cos(\phi(t))` factor in the laminar flow equation creates directional dependence:

- **Parallel to flow** (:math:`\phi = 0°`): :math:`\cos(0°) = 1` → maximum shear effect
- **Perpendicular to flow** (:math:`\phi = 90°`): :math:`\cos(90°) = 0` → no shear effect, diffusion only
- **Intermediate angles**: Partial shear contribution scaled by :math:`\cos(\phi)`

Angle Filtering
~~~~~~~~~~~~~~~

Homodyne v2 supports **phi angle filtering** to select specific angular regimes before optimization, reducing parameter count and improving convergence:

.. code-block:: yaml

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Near 0 degrees (parallel to flow)"
       - min_angle: 85.0
         max_angle: 95.0
         description: "Near 90 degrees (perpendicular to flow)"

**Benefits:**

- Reduced parameter count (fewer phi angles → fewer per-angle scaling parameters)
- Focus on specific flow regimes (parallel vs. perpendicular dynamics)
- Improved numerical stability (fewer parameters to optimize)

Configuration Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: "laminar_flow"
     bounds:
       - name: phi_0
         min: -180.0
         max: 180.0

   initial_parameters:
     parameter_names:
       - phi_0

**Typical Values:**

- **phi_0:** Determined by experimental geometry (detector position relative to flow cell)
- Common values: 0° (parallel), 90° (perpendicular), or intermediate angles
- Angles automatically normalized to [-180°, 180°] range

.. note::

   **Parameter Name Mapping (Config → Code):**

   - ``phi_0`` (config) → ``phi0`` (code)

.. seealso::

   - :doc:`../advanced-topics/angle-filtering` - Comprehensive angle filtering guide
   - :doc:`../user-guide/configuration` - Angle filtering configuration examples


Integration in Optimization
----------------------------

Parameter Extraction Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Define Parameter Bounds:**

   Specify physically reasonable ranges for :math:`D_0`, :math:`\alpha`, :math:`D_{\text{offset}}`, :math:`\dot{\gamma}_0`, :math:`\beta`, :math:`\dot{\gamma}_{\text{offset}}`, :math:`\phi_0` in configuration file.

2. **Load Experimental Data:**

   Import two-time correlation function :math:`c_2(\vec{q}, t_1, t_2)` from HDF5 XPCS data.

3. **Run Optimization:**

   Use NLSQ trust-region optimization or MCMC sampling to fit parameterized model to experimental data.

4. **Extract Transport Properties:**

   Retrieve optimized parameters and construct :math:`D(t)`, :math:`\dot{\gamma}(t)` over experimental time range.

**Example:**

.. code-block:: python

   from homodyne.cli.commands import run_analysis

   # Run NLSQ optimization
   results = run_analysis(config_path="laminar_flow_config.yaml", method="nlsq")

   # Extract optimized parameters
   D0 = results['parameters']['D0']['value']
   alpha = results['parameters']['alpha']['value']
   D_offset = results['parameters']['D_offset']['value']

   # Construct D(t) over experimental time range
   import numpy as np
   t = np.linspace(0, 100, 1000)  # 0-100 seconds
   D_t = D0 + D_offset * t**alpha

   # Plot time evolution
   import matplotlib.pyplot as plt
   plt.plot(t, D_t)
   plt.xlabel('Time (s)')
   plt.ylabel('D(t) (Å²/s)')
   plt.title('Time-Dependent Diffusion Coefficient')
   plt.show()

Physical Constraints
~~~~~~~~~~~~~~~~~~~~

Homodyne v2 validates parameters against physics-based constraints:

- :math:`D_0 > 0`: Diffusion coefficient must be positive at :math:`t = 0`
- :math:`\alpha` reasonable: Typically :math:`0 \leq \alpha \leq 2` for most soft matter systems
- :math:`D_{\text{offset}}` allows negative values: Captures aging/arrest dynamics

See :class:`homodyne.config.parameter_manager.ParameterManager` for validation implementation.

.. seealso::

   - :doc:`parameter-models` - Complete parameter model documentation
   - :doc:`../advanced-topics/nlsq-optimization` - Optimization workflows
   - :doc:`../advanced-topics/mcmc-uncertainty` - Bayesian parameter inference


Summary
-------

The power-law parameterizations provide flexible yet tractable functional forms for capturing time-dependent transport properties:

.. list-table:: Transport Coefficient Summary
   :widths: 25 35 40
   :header-rows: 1

   * - Quantity
     - Parameterization
     - Physical Significance
   * - **Diffusion** :math:`D(t)`
     - :math:`D_0 + D_{\text{offset}} \cdot t^\alpha`
     - Random Brownian motion, particle mobility
   * - **Shear Rate** :math:`\dot{\gamma}(t)`
     - :math:`\dot{\gamma}_0 + \dot{\gamma}_{\text{offset}} \cdot t^\beta`
     - Externally driven flow, advective transport
   * - **Angle** :math:`\phi(t)`
     - :math:`\phi_0` (constant)
     - Flow direction relative to scattering vector

These parameterizations enable **direct extraction** of transport properties from experimental :math:`c_2(t_1, t_2)` data without temporal averaging or equilibrium assumptions, unlocking detailed insights into non-equilibrium soft matter dynamics.

.. seealso::

   - :doc:`core-equations` - Full laminar flow equation implementation
   - :doc:`parameter-models` - Parameter counting (3+2n static, 7+2n laminar)
   - :doc:`../user-guide/configuration` - Complete configuration examples
   - :doc:`../api-reference/core` - Core physics API documentation
