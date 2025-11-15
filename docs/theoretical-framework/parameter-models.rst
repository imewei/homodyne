Parameter Models
================

Overview
--------

Homodyne v2 implements two physical models with fundamentally different parameter counting schemes:

1. **Static Isotropic Model:** 3 physical parameters + 2 per phi angle = **3 + 2n** total parameters
2. **Laminar Flow Model:** 7 physical parameters + 2 per phi angle = **7 + 2n** total parameters

where :math:`n` is the number of phi angles (scattering directions) included in the analysis after optional angle filtering.

**Key Concept:**

Physical parameters (diffusion, shear) characterize the underlying transport properties of the system, while per-angle scaling parameters (contrast, offset) account for experimental variations in signal intensity and baseline across different detector positions.

.. important::

   **Critical Parameter Counting:**

   - Physical parameters are **shared across all phi angles** (single set of D₀, α, γ̇₀, etc.)
   - Scaling parameters are **unique for each phi angle** (separate contrast and offset for each angle)
   - This structure reflects the physical reality: transport properties are isotropic (or have angular dependence through cos(φ)), but experimental signal strength varies by detector position

.. seealso::

   - :doc:`core-equations` - Mathematical formulation with full laminar flow equation
   - :doc:`transport-coefficients` - Time-dependent transport coefficient framework
   - :doc:`../user-guide/configuration` - Configuration examples
   - :doc:`../configuration-templates/index` - Template selection guide


Static Isotropic Model (3+2n Parameters)
-----------------------------------------

Physical Model
~~~~~~~~~~~~~~

The static isotropic model describes equilibrium or quasi-equilibrium systems with isotropic Brownian diffusion and **no external flow or shear**. The second-order correlation function simplifies to:

.. math::
   :label: static_isotropic

   c_2(\vec{q}, t_1, t_2) = 1 + \beta \exp\left\{-q^2 \int_{t_1}^{t_2} J(t) \, \mathrm{d}t\right\}

with time-dependent diffusion parameterized as :math:`D(t) = D_0 + D_{\text{offset}} \cdot t^\alpha` (see :doc:`transport-coefficients`).

Physical Parameters (3)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Static Isotropic Physical Parameters
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - **D0**
     - :math:`D_0`
     - :math:`\text{Å}^2/\text{s}`
     - Initial diffusion coefficient at :math:`t = 0`, reflecting particle mobility and thermal fluctuations
   * - **alpha**
     - :math:`\alpha`
     - dimensionless
     - Power-law exponent for diffusion evolution (0 = constant, 0-1 = sub-diffusive, 1 = linear, >1 = super-diffusive)
   * - **D_offset**
     - :math:`D_{\text{offset}}`
     - :math:`\text{Å}^2/\text{s}`
     - Offset controlling magnitude and direction of diffusion evolution (positive = increasing, negative = aging/arrest)

**Shared Across All Phi Angles:** These 3 parameters describe the intrinsic transport properties of the system and are **the same** regardless of scattering direction (phi angle).

Per-Angle Scaling Parameters (2n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **each phi angle** :math:`\phi_i` (where :math:`i = 1, 2, \dots, n`), two scaling parameters are required:

.. list-table:: Per-Angle Scaling Parameters
   :widths: 20 25 55
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
   * - **contrast**
     - :math:`\beta_i`
     - Signal contrast for angle :math:`\phi_i`, accounting for variations in coherent scattering intensity across detector positions (0 ≤ β ≤ 1)
   * - **offset**
     - :math:`C_i`
     - Baseline offset for angle :math:`\phi_i`, correcting for incoherent background and detector dark current

**Why Per-Angle Scaling?**

Experimental XPCS data from different detector positions (phi angles) exhibit variations in:

- **Signal strength** (detector efficiency, solid angle, sample alignment)
- **Baseline** (dark current, background scattering, incoherent contributions)

These are **experimental artifacts**, not physical properties of the system. Per-angle scaling parameters absorb these variations, allowing the physical parameters (D₀, α, D_offset) to be extracted cleanly.

Total Parameter Count
~~~~~~~~~~~~~~~~~~~~~~

For a dataset with :math:`n` phi angles:

.. math::

   \text{Total Parameters} = 3 + 2n

**Examples:**

- **1 angle:** :math:`3 + 2(1) = 5` parameters
- **3 angles:** :math:`3 + 2(3) = 9` parameters
- **5 angles:** :math:`3 + 2(5) = 13` parameters
- **10 angles:** :math:`3 + 2(10) = 23` parameters

Configuration Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: "static_isotropic"

     bounds:
       # Physical parameters (3)
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

   # Per-angle scaling (contrast, offset) handled automatically by optimizer
   # Number of scaling parameters = 2 × (number of phi angles after filtering)

.. note::

   **Automatic Scaling Management:**

   Per-angle scaling parameters (contrast, offset) are **automatically added** by the optimization engine based on the number of phi angles present in the experimental data after optional angle filtering. Users only specify the 3 physical parameters in the configuration file.

Use Cases
~~~~~~~~~

The static isotropic model is appropriate for:

- **Equilibrium systems:** No external drive, thermalized dynamics
- **Isotropic materials:** Glasses, colloids, polymers in quiescent state
- **Baseline characterization:** Establishing intrinsic diffusive properties before applying shear
- **Aging dynamics:** Tracking diffusion evolution in gels, glasses, jammed systems (negative D_offset)
- **Relaxation after stress removal:** Monitoring diffusion recovery (positive D_offset)

.. seealso::

   - :doc:`../configuration-templates/static-isotropic` - Complete static isotropic configuration template
   - :doc:`../user-guide/examples` - Worked example: static isotropic analysis


Laminar Flow Model (7+2n Parameters)
-------------------------------------

Physical Model
~~~~~~~~~~~~~~

The laminar flow model describes **non-equilibrium systems** under external shear or flow, capturing both Brownian diffusion and advective transport. The full second-order correlation function is:

.. math::
   :label: laminar_flow

   c_2(\vec{q}, t_1, t_2) = 1 + \beta \left[ \exp\left\{-q^2 \int_{t_1}^{t_2} J(t) \, \mathrm{d}t\right\}\right] \times \text{sinc}^2\left[\frac{1}{2}qh \int_{t_1}^{t_2} \dot{\gamma}(t) \cos\left(\phi(t)\right) \,\mathrm{d}t\right]

This extends the static isotropic model by including:

- Time-dependent shear rate :math:`\dot{\gamma}(t) = \dot{\gamma}_0 + \dot{\gamma}_{\text{offset}} \cdot t^\beta`
- Angle evolution :math:`\phi(t)` (typically constant, parameterized by :math:`\phi_0`)
- Gap height :math:`h` between shearing surfaces

Physical Parameters (7)
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Laminar Flow Physical Parameters
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - **D0**
     - :math:`D_0`
     - :math:`\text{Å}^2/\text{s}`
     - Initial diffusion coefficient (thermal fluctuations + particle interactions)
   * - **alpha**
     - :math:`\alpha`
     - dimensionless
     - Diffusion power-law exponent (0 = constant, 0-1 = sub-diffusive, >1 = super-diffusive)
   * - **D_offset**
     - :math:`D_{\text{offset}}`
     - :math:`\text{Å}^2/\text{s}`
     - Diffusion offset (positive = fluidization, negative = jamming/arrest)
   * - **gamma_dot_0**
     - :math:`\dot{\gamma}_0`
     - :math:`\text{s}^{-1}`
     - Initial shear rate at :math:`t = 0`, set by rheometer or flow cell
   * - **beta**
     - :math:`\beta`
     - dimensionless
     - Shear rate power-law exponent (0 = constant, <0 = relaxation, >0 = creep)
   * - **gamma_dot_offset**
     - :math:`\dot{\gamma}_{\text{offset}}`
     - :math:`\text{s}^{-1}`
     - Shear rate offset (magnitude and direction of shear evolution)
   * - **phi_0**
     - :math:`\phi_0`
     - degrees
     - Initial angle between flow direction and scattering vector (experimental geometry)

**Shared Across All Phi Angles:** These 7 parameters describe the intrinsic transport properties (diffusion + shear) and are **the same** for all scattering directions.

.. note::

   **Parameter Name Mapping (Config → Code):**

   - ``gamma_dot_0`` (config) → ``gamma_dot_t0`` (code)
   - ``phi_0`` (config) → ``phi0`` (code)

Per-Angle Scaling Parameters (2n)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Identical to the static isotropic model, **each phi angle** requires 2 scaling parameters:

.. list-table:: Per-Angle Scaling Parameters
   :widths: 20 25 55
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
   * - **contrast**
     - :math:`\beta_i`
     - Signal contrast for angle :math:`\phi_i` (0 ≤ β ≤ 1)
   * - **offset**
     - :math:`C_i`
     - Baseline offset for angle :math:`\phi_i`

Total Parameter Count
~~~~~~~~~~~~~~~~~~~~~~

For a dataset with :math:`n` phi angles:

.. math::

   \text{Total Parameters} = 7 + 2n

**Examples:**

- **1 angle:** :math:`7 + 2(1) = 9` parameters
- **3 angles:** :math:`7 + 2(3) = 13` parameters
- **5 angles:** :math:`7 + 2(5) = 17` parameters
- **10 angles:** :math:`7 + 2(10) = 27` parameters

.. warning::

   **Parameter Count Growth:**

   With many phi angles, the total parameter count can become large (e.g., 10 angles → 27 parameters). Use **angle filtering** (:doc:`../advanced-topics/angle-filtering`) to reduce :math:`n` by selecting specific angular regimes (e.g., near 0° and 90°), improving convergence and reducing computation time.

Configuration Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: "laminar_flow"

     bounds:
       # Diffusion parameters (3)
       - name: D0
         min: 100.0
         max: 1e5
       - name: alpha
         min: 0.0
         max: 2.0
       - name: D_offset
         min: -100.0
         max: 100.0

       # Flow parameters (4)
       - name: gamma_dot_0
         min: 1e-6
         max: 0.5
       - name: beta
         min: 0.0
         max: 2.0
       - name: gamma_dot_offset
         min: -0.1
         max: 0.1
       - name: phi_0
         min: -180.0
         max: 180.0

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset
       - gamma_dot_0
       - beta
       - gamma_dot_offset
       - phi_0

   # Optional: Reduce parameter count with angle filtering
   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Near 0 degrees (parallel to flow)"
       - min_angle: 85.0
         max_angle: 95.0
         description: "Near 90 degrees (perpendicular to flow)"

   # Per-angle scaling (contrast, offset) handled automatically
   # With angle filtering, n = 2 regions → 7 + 2(2) = 11 parameters

Use Cases
~~~~~~~~~

The laminar flow model is appropriate for:

- **Flowing systems:** Shear flow, pressure-driven flow, Couette/Poiseuille geometry
- **Shear experiments:** Rheometer with parallel-plate or cone-plate geometry
- **Non-equilibrium dynamics:** Time-dependent shear, relaxation after shear removal
- **Shear banding:** Coexistence of high-shear and low-shear regions (requires spatial resolution)
- **Creep and recovery:** Applying constant stress and monitoring strain evolution
- **Yielding transitions:** Tracking diffusion and flow as system transitions from solid-like to liquid-like

.. seealso::

   - :doc:`../configuration-templates/laminar-flow` - Complete laminar flow configuration template
   - :doc:`../advanced-topics/angle-filtering` - Reducing parameter count with angle filtering
   - :doc:`../user-guide/examples` - Worked example: laminar flow analysis


Parameter Bounds and Constraints
---------------------------------

Physics-Based Bounds
~~~~~~~~~~~~~~~~~~~~

Homodyne v2 validates parameters against physically meaningful constraints:

**Diffusion Parameters:**

- :math:`D_0 > 0` (ERROR): Diffusion coefficient must be positive at :math:`t = 0`
- :math:`0 \leq \alpha \leq 2` (WARNING): Exponents outside this range are unusual for soft matter
- :math:`D_{\text{offset}}` allows negative values: Required for aging/arrest dynamics

**Shear Parameters:**

- :math:`\dot{\gamma}_0 \geq 0` (WARNING): Negative shear rates are non-physical
- :math:`-180° \leq \phi_0 \leq 180°` (automatic normalization): Angles wrapped to symmetric range

**Scaling Parameters:**

- :math:`0 \leq \beta \leq 1` (WARNING): Contrast outside [0, 1] suggests calibration issues
- Offset :math:`C` typically near 1.0: Extreme values suggest baseline problems

Default Parameter Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

**Updated November 15, 2025** - Default bounds for NLSQ optimization and MCMC priors:

.. list-table:: Physical Parameter Bounds
   :widths: 18 12 12 15 30 13
   :header-rows: 1

   * - Parameter
     - Min
     - Max
     - Units
     - Physical Meaning
     - Notes
   * - **D0**
     - 1×10²
     - 1×10⁵
     - Å²/s
     - Diffusion coefficient prefactor
     - Typical colloidal range
   * - **alpha**
     - -2.0
     - 2.0
     - —
     - Diffusion time exponent
     - Anomalous diffusion
   * - **D_offset**
     - -1×10⁵
     - 1×10⁵
     - Å²/s
     - Diffusion baseline correction
     - **Negative for jammed systems**
   * - **gamma_dot_t0**
     - 1×10⁻⁶
     - 0.5
     - s⁻¹
     - Initial shear rate
     - Laminar flow only
   * - **beta**
     - -2.0
     - 2.0
     - —
     - Shear rate time exponent
     - Laminar flow only
   * - **gamma_dot_t_offset**
     - -0.1
     - 0.1
     - s⁻¹
     - Shear rate baseline correction
     - Laminar flow only
   * - **phi0**
     - -10
     - 10
     - degrees
     - Initial flow angle
     - **Uses degrees, not radians**

.. list-table:: Scaling Parameter Bounds
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Min
     - Max
     - Physical Meaning
   * - **contrast**
     - 0.0
     - 1.0
     - Visibility parameter (homodyne detection efficiency)
   * - **offset**
     - 0.5
     - 1.5
     - Baseline level (±50% from theoretical g2=1.0)

.. list-table:: Correlation Function Constraints
   :widths: 20 15 15 50
   :header-rows: 1

   * - Function
     - Min
     - Max
     - Notes
   * - **g1 (c1)**
     - 0.0
     - 1.0
     - Normalized correlation function; Log-space clipping: :math:`\log(g_1) \in [-700, 0]`
   * - **g2 (c2)**
     - 0.5
     - 2.5
     - Experimental range with headroom; Theoretical: :math:`g_2 = 1 + \beta \cdot g_1^2`

.. important::

   **Critical Constraint Updates:**

   - **D_offset** can be **negative** for arrested/jammed systems (caging, jamming transitions)
   - **phi0** uses **degrees** throughout the codebase (templates, physics modules)
   - **gamma_dot_t_offset** allows **negative values** (baseline correction)
   - All bounds align with template files: ``homodyne_static.yaml``, ``homodyne_laminar_flow.yaml``
   - User configs **override** these default bounds (no breaking changes)

Validation System
~~~~~~~~~~~~~~~~~

The :class:`homodyne.config.parameter_manager.ParameterManager` class implements three severity levels:

1. **ERROR:** Unphysical values that prevent optimization (e.g., :math:`D_0 \leq 0`)
2. **WARNING:** Unusual but potentially valid values (e.g., :math:`\alpha > 2`)
3. **INFO:** Observations about parameter ranges (e.g., large D_offset relative to D0)

Configuration Override
~~~~~~~~~~~~~~~~~~~~~~

Users can override default bounds in the configuration file:

.. code-block:: yaml

   parameter_space:
     bounds:
       - name: D0
         min: 500.0      # Tighter bound for specific system
         max: 5000.0
       - name: alpha
         min: 0.5        # Restrict to sub-diffusive regime
         max: 1.0

The :class:`~homodyne.config.parameter_manager.ParameterManager` merges user-specified bounds with defaults, prioritizing user values while maintaining physics validation.

.. seealso::

   - :doc:`../api-reference/config` - ParameterManager API documentation
   - :doc:`../user-guide/configuration` - Configuration system overview


Initial Parameter Selection
----------------------------

Starting Values
~~~~~~~~~~~~~~~

Choosing reasonable initial parameter values improves convergence:

**Diffusion Parameters:**

- **D0:** Estimate from particle size using Stokes-Einstein (:math:`D = k_B T / 6\pi\eta a`)
- **alpha:** Start with 0.5 (sub-diffusive) for crowded/confined systems, 0.0 (constant) for equilibrium
- **D_offset:** Start with 0.0, allow optimizer to determine sign and magnitude

**Shear Parameters:**

- **gamma_dot_0:** Use rheometer setpoint or calculate from flow rate and gap height
- **beta:** Start with 0.0 (constant shear), allow optimizer to capture relaxation/creep
- **gamma_dot_offset:** Start with 0.0
- **phi_0:** Determined by experimental geometry (detector position)

Active vs. Fixed Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can selectively optimize a subset of parameters:

.. code-block:: yaml

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset
       - gamma_dot_0
       - beta
       - gamma_dot_offset
       - phi_0

     # Optional: Optimize only critical parameters
     active_parameters:
       - D0
       - alpha
       - gamma_dot_0

     # Optional: Fix known parameters
     fixed_parameters:
       D_offset: 10.0
       phi_0: 0.0

This reduces the effective parameter count, improving convergence speed and numerical stability.

**Strategy:**

1. **First pass:** Optimize only D0 and gamma_dot_0 (if known) with all other parameters fixed
2. **Second pass:** Add alpha and beta to capture time-dependence
3. **Third pass:** Refine with all parameters active, using first-pass results as initial values

.. seealso::

   - :doc:`../advanced-topics/nlsq-optimization` - Optimization workflows and strategies


Optimization Considerations
----------------------------

Parameter Count vs. Convergence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Challenge:** As the number of phi angles increases, total parameter count grows linearly:

- **3 angles:** 9 (static) or 13 (laminar) parameters
- **10 angles:** 23 (static) or 27 (laminar) parameters
- **20 angles:** 43 (static) or 47 (laminar) parameters

**Impact:**

- **Computational cost:** More parameters → longer optimization time
- **Numerical stability:** High-dimensional parameter space increases risk of local minima
- **Data requirements:** More parameters require more data points for reliable fitting

**Mitigation Strategies:**

1. **Angle filtering:** Reduce :math:`n` by selecting specific angular regimes (:doc:`../advanced-topics/angle-filtering`)
2. **Sequential optimization:** Optimize subsets of parameters iteratively
3. **CMC (Covariance Matrix Combination):** Parallelize optimization across angles (:doc:`../advanced-topics/cmc-large-datasets`)
4. **MCMC uncertainty quantification:** Sample posterior distributions instead of point optimization (:doc:`../advanced-topics/mcmc-uncertainty`)

Angle Filtering to Reduce Parameter Count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By selecting specific phi angle ranges, the effective :math:`n` can be dramatically reduced:

**Example:**

.. code-block:: yaml

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
       - min_angle: 85.0
         max_angle: 95.0

   # Before filtering: 20 angles → 7 + 2(20) = 47 parameters
   # After filtering: 2 angle regions → 7 + 2(2) = 11 parameters

**Benefits:**

- **Faster convergence:** Fewer parameters → simpler optimization landscape
- **Physical insight:** Focus on critical angular regimes (parallel vs. perpendicular to flow)
- **Numerical stability:** Reduced dimensionality → more robust optimization

.. seealso::

   - :doc:`../advanced-topics/angle-filtering` - Comprehensive angle filtering guide


Summary
-------

Homodyne v2 parameter models capture the essential physics of soft matter systems with minimal parameter sets:

.. list-table:: Parameter Model Summary
   :widths: 25 20 25 30
   :header-rows: 1

   * - Model
     - Physical Params
     - Scaling Params
     - Total Count
   * - **Static Isotropic**
     - 3 (D₀, α, D_offset)
     - 2n (contrast, offset per angle)
     - **3 + 2n**
   * - **Laminar Flow**
     - 7 (D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀)
     - 2n (contrast, offset per angle)
     - **7 + 2n**

**Key Principles:**

1. **Physical parameters are shared:** Single set of transport coefficients describes intrinsic system properties
2. **Scaling parameters are per-angle:** Experimental variations absorbed by contrast and offset for each phi angle
3. **Parameter count grows linearly:** Total parameters = physical + 2 × (number of phi angles)
4. **Angle filtering reduces n:** Focus on specific angular regimes to improve convergence

**Example Parameter Counts:**

.. list-table:: Parameter Count Examples
   :widths: 20 20 30 30
   :header-rows: 1

   * - Phi Angles (n)
     - Static (3+2n)
     - Laminar (7+2n)
     - Notes
   * - 1
     - 5
     - 9
     - Minimal case
   * - 3
     - 9
     - 13
     - Typical filtered dataset
   * - 5
     - 13
     - 17
     - Good angular coverage
   * - 10
     - 23
     - 27
     - Full detector array
   * - 20
     - 43
     - 47
     - Unfiltered, high-resolution

**Recommendations:**

- **Start simple:** Use static isotropic model first to establish baseline diffusion properties
- **Apply angle filtering:** Reduce :math:`n` to 2-5 critical angular regimes for laminar flow analysis
- **Sequential refinement:** Optimize subsets of parameters, then refine with full parameter set
- **Validate results:** Check parameter uncertainties and convergence diagnostics

.. seealso::

   - :doc:`core-equations` - Full mathematical formulation
   - :doc:`transport-coefficients` - Time-dependent transport coefficient framework
   - :doc:`../configuration-templates/index` - Template selection guide
   - :doc:`../user-guide/configuration` - Configuration system documentation
   - :doc:`../advanced-topics/nlsq-optimization` - Optimization workflows
   - :doc:`../advanced-topics/angle-filtering` - Angle filtering strategies
