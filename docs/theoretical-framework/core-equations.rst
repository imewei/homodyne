Core Equations
==============

Introduction
------------

Homodyne v2 implements the theoretical framework from He et al., *PNAS* **121**(31), e2401162121 (2024) for characterizing transport properties in flowing soft matter systems using X-ray Photon Correlation Spectroscopy (XPCS). The framework enables direct extraction of time-dependent transport coefficients from two-time correlation functions without requiring temporal averaging or assumptions of pseudo-equilibrium.

**Key Innovation:** By formulating Langevin dynamics with Markovian (memoryless) non-equilibrium processes, the theory derives analytical expressions for the second-order intensity autocorrelation function :math:`c_2(\vec{q}, t_1, t_2)` that capture both intrinsic random motion (diffusion) and externally driven motion (flow/shear).

**Physical Context:**

The two-time correlation function :math:`c_2(\vec{q}, t_1, t_2)` measures temporal correlations in X-ray scattering intensity at two different times (:math:`t_1`, :math:`t_2`) for a given scattering wavevector :math:`\vec{q}`. In equilibrium systems, this reduces to a one-time correlation function :math:`g_2(\vec{q}, \tau)` dependent only on the delay time :math:`\tau = t_2 - t_1`. However, in non-equilibrium systems (e.g., flowing, relaxing, or yielding soft matter), the full two-time dependence is required to capture time-varying dynamics.

.. seealso::

   **Reference:**
   He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J. J., Tirrell, M. V., Narayanan, S., & Chen, W. (2024).
   "Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter."
   *Proceedings of the National Academy of Sciences*, **121**(31), e2401162121.
   DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_


Full Nonequilibrium Laminar Flow (Manuscript Equation 13)
----------------------------------------------------------

The full time-dependent correlation function under nonequilibrium laminar flow between two parallel plates with time-dependent shear rate :math:`\dot{\gamma}(t)` is given by:

.. math::
   :label: eq13_laminar_flow

   c_2(\vec{q}, t_1, t_2) = 1 + \beta \left[ \exp\left\{-q^2 \int_{t_1}^{t_2} J(t) \, \mathrm{d}t\right\}\right] \times \text{sinc}^2\left[\frac{1}{2}qh \int_{t_1}^{t_2} \dot{\gamma}(t) \cos\left(\phi(t)\right) \,\mathrm{d}t\right]

where:

- :math:`\vec{q}` is the scattering wavevector [:math:`\text{Å}^{-1}`]
- :math:`q = |\vec{q}|` is the magnitude of the scattering wavevector [:math:`\text{Å}^{-1}`]
- :math:`h` is the gap between stator and rotor [:math:`\text{Å}`]
- :math:`\phi(t)` is the angle between shear/flow direction and :math:`\vec{q}` [degrees]
- :math:`\dot{\gamma}(t)` is the time-dependent shear rate [:math:`\text{s}^{-1}`]
- :math:`J(t)` is the dynamical transport coefficient [:math:`\text{Å}^2/\text{s}`], related to diffusion
- :math:`\beta` is the contrast parameter (coherent contrast) [dimensionless], where :math:`0 \leq \beta \leq 1`
- :math:`t_1, t_2` are two absolute time points in the experiment [s]

**Physical Interpretation:**

This equation captures the interplay between two fundamental decorrelation mechanisms:

1. **Diffusive Decorrelation (Exponential Term):**

   The term :math:`\exp\left\{-q^2 \int_{t_1}^{t_2} J(t) \, \mathrm{d}t\right\}` describes decorrelation due to Brownian diffusion and thermal fluctuations. The transport coefficient :math:`J(t)` characterizes the rate of diffusive spreading, encompassing systematic dynamics (particle interactions, hydrodynamic effects) and thermal fluctuations. This term exhibits isotropic behavior proportional to :math:`q^2`.

2. **Advective Decorrelation (Sinc² Term):**

   The term :math:`\text{sinc}^2\left[\frac{1}{2}qh \int_{t_1}^{t_2} \dot{\gamma}(t) \cos\left(\phi(t)\right) \,\mathrm{d}t\right]` accounts for decorrelation from shear-induced flow across the gap :math:`h`. The :math:`\cos(\phi(t))` factor reflects anisotropic behavior: maximum decorrelation when :math:`\vec{q}` aligns with the flow direction (:math:`\phi = 0^\circ`), and minimum when perpendicular (:math:`\phi = 90^\circ`).

**Implementation:**

This equation is implemented in :func:`homodyne.core.jax_backend.compute_g2_scaled` and :func:`homodyne.core.jax_backend.compute_g2` with time-dependent transport coefficients parameterized as power-law functions (see :doc:`transport-coefficients`).

.. code-block:: python

   from homodyne.core.jax_backend import compute_g2_scaled
   import jax.numpy as jnp

   # Physical parameters (laminar flow model)
   params = {
       'D0': 1000.0,           # Initial diffusion coefficient [Å²/s]
       'alpha': 0.5,           # Diffusion power-law exponent [dimensionless]
       'D_offset': 10.0,       # Diffusion offset [Å²/s]
       'gamma_dot_t0': 0.01,   # Initial shear rate [s⁻¹]
       'beta': 0.8,            # Shear power-law exponent [dimensionless]
       'gamma_dot_offset': 0.0,# Shear rate offset [s⁻¹]
       'phi0': 0.0             # Initial angle [degrees]
   }

   # Scaling parameters (per phi angle)
   contrast = 0.5
   offset = 1.0

   # Experimental parameters
   q = 0.01  # Scattering wavevector [Å⁻¹]
   L = 5000.0  # Gap h [Å]
   dt = 0.1  # Time step [s]
   phi = 0.0  # Angle [degrees]

   # Time arrays
   t1 = jnp.linspace(0, 10, 100)
   t2 = jnp.linspace(0, 10, 100)

   # Compute g2
   g2 = compute_g2_scaled(t1, t2, q, L, dt, phi, params, contrast, offset)

.. note::

   **Parameter Name Mapping (Config → Code):**

   - ``gamma_dot_0`` (config) → ``gamma_dot_t0`` (code)
   - ``phi_0`` (config) → ``phi0`` (code)

.. seealso::

   - :doc:`transport-coefficients` - Time-dependent diffusion :math:`D(t)` and shear rate :math:`\dot{\gamma}(t)` parameterizations
   - :doc:`parameter-models` - Static (3+2n) vs. Laminar (7+2n) parameter counting
   - :doc:`../api-reference/core` - Core physics API documentation
   - :doc:`../user-guide/configuration` - Configuration system


Equilibrium Under Constant Shear (SI Equation S-75)
----------------------------------------------------

For equilibrium systems under constant shear, where the shear rate is time-independent (:math:`\dot{\gamma}(t) = \dot{\gamma}`, :math:`\phi(t) = \phi`) and diffusion follows standard Brownian motion (:math:`J(t) = 6D`), the correlation function simplifies to:

.. math::
   :label: eq_s75_constant_shear

   c_2(\vec{q}, t_1, t_2) = 1 + \beta \left[ \exp\left\{-6q^2 D (t_2 - t_1)\right\}\right] \text{sinc}^2\left[\frac{1}{2}qh \cos(\phi) \dot{\gamma}(t_2 - t_1) \right]

where:

- :math:`D` is the constant diffusion coefficient [:math:`\text{Å}^2/\text{s}`]
- :math:`\dot{\gamma}` is the constant shear rate [:math:`\text{s}^{-1}`]
- :math:`\phi` is the constant angle between flow and :math:`\vec{q}` [degrees]
- All other parameters defined as in Equation :eq:`eq13_laminar_flow`

**Relationship to Full Nonequilibrium Model:**

This is a special case of Equation :eq:`eq13_laminar_flow` under equilibrium approximations:

1. **Time-independent shear:** :math:`\dot{\gamma}(t) \cos(\phi(t)) = \dot{\gamma} \cos(\phi)` (constant)
2. **Standard diffusion:** :math:`J(t) = 6D` (constant transport coefficient)

**When This Approximation Applies:**

- **Steady-state flow:** Shear rate has reached a constant plateau
- **Equilibrium dynamics:** System thermalized under constant external drive
- **No transient effects:** No relaxation, yielding, or time-dependent structural changes
- **Homogeneous shear:** Single shear band with uniform velocity profile

**Physical Significance:**

The factor :math:`6D` arises from isotropic diffusion in 3D, where :math:`J_x = J_y = J_z = \frac{1}{3}J` and :math:`J = 2D` (1D transport coefficient), giving :math:`J = 6D` for 3D.

.. seealso::

   - :doc:`transport-coefficients` - Derivation of :math:`J(t) = 6D` for standard diffusion


One-Time Correlation / Siegert Relation (SI Equation S-76)
-----------------------------------------------------------

By substituting absolute time coordinates :math:`(t_1, t_2)` with the delay time :math:`\tau = t_2 - t_1`, the two-time correlation function reduces to the classical one-time correlation function:

.. math::
   :label: eq_s76_one_time

   g_2(\vec{q}, \tau) = 1 + \beta \left[ \exp\left\{-6q^2 D \tau\right\}\right] \text{sinc}^2\left[\frac{1}{2}qh \cos(\phi) \dot{\gamma} \tau \right]

where:

- :math:`\tau = t_2 - t_1` is the delay time [s]
- :math:`g_2(\vec{q}, \tau)` is the one-time intensity autocorrelation function
- All other parameters defined as in Equation :eq:`eq_s75_constant_shear`

**Siegert Relation:**

This equation is derived from the Siegert relation, which connects the second-order intensity correlation :math:`c_2` to the first-order field correlation :math:`c_1` under Gaussian statistics of the scattered field:

.. math::

   c_2(\vec{q}, t_1, t_2) = 1 + \beta \left| c_1(\vec{q}, t_1, t_2) \right|^2

where :math:`c_1(\vec{q}, t_1, t_2)` is the normalized first-order field correlation function. The Siegert relation applies when the scattered fields are generated by Gaussian processes, which holds for systems with many independent scatterers and randomized phases.

**When This Form Applies:**

- **Time-translation invariance:** Dynamics depend only on :math:`\tau`, not absolute time
- **Equilibrium or steady-state:** No temporal evolution of system properties
- **Classical XPCS analysis:** Traditional diagonal averaging of :math:`c_2(t_1, t_2)` along constant :math:`\tau`

**Limitations for Homodyne v2:**

Homodyne v2 uses the **full two-time correlation function** (Equation :eq:`eq13_laminar_flow`) to capture non-equilibrium dynamics. The one-time form (Equation :eq:`eq_s76_one_time`) is **only valid** for equilibrium/steady-state systems and would lose critical information about time-dependent transport properties in flowing, relaxing, or yielding systems.

.. warning::

   **Important:**

   Classical XPCS analysis relies on reducing :math:`c_2(t_1, t_2)` to :math:`g_2(\tau)` by averaging within "time-translation invariant zones" (TIZs). This approach can distort or overlook intricate time-dependent variations, leading to loss of information about non-equilibrium dynamics. Homodyne v2 directly fits the full :math:`c_2(t_1, t_2)` without temporal averaging.

.. seealso::

   - **Manuscript Section 2.3:** Derivation of two-time correlation function from Langevin dynamics
   - **SI Section 2.E:** Laminar flow derivation with time-dependent shear
   - :doc:`../user-guide/quickstart` - Quick start guide for non-equilibrium analysis


Summary of Core Equations
--------------------------

The three equations represent a hierarchy of increasing complexity:

.. list-table:: Equation Hierarchy
   :widths: 20 40 40
   :header-rows: 1

   * - Equation
     - Applicability
     - Key Assumptions
   * - **S-76 (One-Time)**
     - Equilibrium, steady-state, time-translation invariance
     - :math:`\tau`-dependent only, constant :math:`D` and :math:`\dot{\gamma}`
   * - **S-75 (Constant Shear)**
     - Equilibrium under constant external drive
     - Time-independent :math:`\dot{\gamma}` and :math:`\phi`, standard diffusion (:math:`J = 6D`)
   * - **13 (Full Nonequilibrium)**
     - General non-equilibrium systems
     - Time-dependent :math:`J(t)`, :math:`\dot{\gamma}(t)`, :math:`\phi(t)`

**Homodyne v2 Implementation:**

Homodyne v2 implements the **most general form (Equation 13)** with parameterized time-dependent transport coefficients:

- :math:`J(t)` parameterized through :math:`D(t) = D_0 + D_{\text{offset}} \cdot t^\alpha`
- :math:`\dot{\gamma}(t)` parameterized as :math:`\dot{\gamma}(t) = \dot{\gamma}_0 + \dot{\gamma}_{\text{offset}} \cdot t^\beta`
- :math:`\phi(t)` evolution tracked through :math:`\phi_0` parameter

This enables direct extraction of transport properties from experimental two-time correlation functions without averaging or equilibrium assumptions.

.. seealso::

   - :doc:`transport-coefficients` - Detailed parameterizations of :math:`D(t)`, :math:`\dot{\gamma}(t)`, :math:`\phi(t)`
   - :doc:`parameter-models` - Static isotropic (3+2n) vs. Laminar flow (7+2n) models
   - :doc:`../advanced-topics/nlsq-optimization` - Optimization workflows for parameter extraction
   - :doc:`../user-guide/configuration` - Configuration examples


Cross-References
----------------

**Theory:**

- :doc:`transport-coefficients` - Time-dependent transport coefficient framework
- :doc:`parameter-models` - Parameter counting and physical models

**Implementation:**

- :doc:`../api-reference/core` - Core physics API (``jax_backend.compute_g2``, ``jax_backend.compute_g2_scaled``)
- :doc:`../user-guide/configuration` - YAML configuration examples

**Advanced Topics:**

- :doc:`../advanced-topics/nlsq-optimization` - Trust-region optimization for parameter extraction
- :doc:`../advanced-topics/mcmc-uncertainty` - Bayesian uncertainty quantification
- :doc:`../user-guide/examples` - Worked examples (static isotropic, laminar flow)
