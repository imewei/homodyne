Theoretical Framework
=====================

This section provides the mathematical foundation for the homodyne scattering analysis
implemented in the package, based on the theoretical framework developed by He et al. (2024).

Mathematical Foundation
-----------------------

Core Correlation Equation
~~~~~~~~~~~~~~~~~~~~~~~~~

The fundamental quantity analyzed is the two-time intensity correlation function.
The relationship between the intensity correlation :math:`c_2` and the field correlation
:math:`c_1` follows the Siegert relation:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \beta \left[ c_1(\phi, t_1, t_2) \right]^2

where:

* :math:`c_2(\phi, t_1, t_2)`: Two-time intensity correlation function
* :math:`c_1(\phi, t_1, t_2)`: Two-time field correlation function
* :math:`\beta`: Instrumental contrast parameter (typically 0 < :math:`\beta` < 1)
* :math:`\phi`: Azimuthal angle between flow direction and scattering wavevector
* :math:`t_1, t_2`: Two time points defining the correlation window

In the Homodyne implementation, this is parameterized with per-angle scaling as:

.. math::

   c_2(\phi_i, t_1, t_2) = \text{offset}_i + \text{contrast}_i \times \left[ c_1(\phi_i, t_1, t_2) \right]^2

where :math:`\text{contrast}_i` and :math:`\text{offset}_i` are per-angle parameters
that account for instrumental variations across different detector positions.

Field Correlation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~

The field correlation function captures both diffusive and advective contributions:

.. math::

   c_1(\phi, t_1, t_2) = c_1^{\text{diff}}(t_1, t_2) \times c_1^{\text{shear}}(\phi, t_1, t_2)

**Diffusion Contribution:**

.. math::

   c_1^{\text{diff}}(t_1, t_2) = \exp\left[-\frac{q^2}{2} J(t_1, t_2)\right]

where :math:`q` is the scattering wavevector magnitude and :math:`J(t_1, t_2)` is the
diffusion integral.

**Shear Contribution:**

.. math::

   c_1^{\text{shear}}(\phi, t_1, t_2) = \text{sinc}^2\left[\frac{qh}{2\pi} \Gamma(t_1, t_2) \cos(\phi - \phi_0)\right]

where :math:`h` is the gap between stator and rotor, :math:`\Gamma(t_1, t_2)` is the
shear integral, and :math:`\phi_0` is the flow direction angle.

Time-Dependent Transport Coefficients
-------------------------------------

Diffusion Coefficient
~~~~~~~~~~~~~~~~~~~~~

The time-dependent diffusion coefficient follows a power-law parameterization for
anomalous diffusion:

.. math::

   D(t) = D_0 \cdot t^{\alpha} + D_{\text{offset}}

where:

* :math:`D_0`: Baseline diffusion coefficient [units depend on :math:`\alpha`]
* :math:`\alpha`: Diffusion scaling exponent (dimensionless)
    - :math:`\alpha = 0`: Normal diffusion
    - :math:`\alpha < 0`: Subdiffusion (constrained motion)
    - :math:`\alpha > 0`: Superdiffusion (enhanced motion)
* :math:`D_{\text{offset}}`: Additive offset for baseline correction

Shear Rate
~~~~~~~~~~

The time-dependent shear rate is parameterized as:

.. math::

   \dot{\gamma}(t) = \dot{\gamma}_0 \cdot t^{\beta} + \dot{\gamma}_{\text{offset}}

where:

* :math:`\dot{\gamma}_0`: Baseline shear rate [units depend on :math:`\beta`]
* :math:`\beta`: Shear rate scaling exponent (dimensionless)
* :math:`\dot{\gamma}_{\text{offset}}`: Additive offset [s\ :sup:`-1`]

Integral Functions
------------------

Diffusion Integral
~~~~~~~~~~~~~~~~~~

The diffusion integral over the time window :math:`[t_1, t_2]`:

.. math::

   J(t_1, t_2) = \int_{t_1}^{t_2} D(t) \, dt

For the power-law parameterization, this evaluates analytically to:

.. math::

   J(t_1, t_2) = \frac{D_0}{1+\alpha}\left(t_2^{1+\alpha} - t_1^{1+\alpha}\right) + D_{\text{offset}}(t_2 - t_1)

**Special case** (:math:`\alpha = 0`, normal diffusion):

.. math::

   J(t_1, t_2) = D_0(t_2 - t_1) + D_{\text{offset}}(t_2 - t_1) = (D_0 + D_{\text{offset}})(t_2 - t_1)

Shear Integral
~~~~~~~~~~~~~~

The shear integral over the time window :math:`[t_1, t_2]`:

.. math::

   \Gamma(t_1, t_2) = \int_{t_1}^{t_2} \dot{\gamma}(t) \, dt

For the power-law parameterization:

.. math::

   \Gamma(t_1, t_2) = \frac{\dot{\gamma}_0}{1+\beta}\left(t_2^{1+\beta} - t_1^{1+\beta}\right) + \dot{\gamma}_{\text{offset}}(t_2 - t_1)

Scattering Geometry
-------------------

Wavevector Definition
~~~~~~~~~~~~~~~~~~~~~

The scattering wavevector magnitude is related to the scattering angle:

.. math::

   q = \frac{4\pi}{\lambda} \sin\left(\frac{\theta}{2}\right)

where :math:`\lambda` is the X-ray wavelength and :math:`\theta` is the scattering angle.

Angular Dependence
~~~~~~~~~~~~~~~~~~

The azimuthal angle :math:`\phi` represents the orientation between the flow direction
and the scattering wavevector in the detector plane:

.. math::

   \phi = \arctan\left(\frac{q_y}{q_x}\right)

The flow direction offset :math:`\phi_0` accounts for any misalignment between the
nominal flow axis and the actual shear direction.

Physical Interpretation
-----------------------

Static Systems
~~~~~~~~~~~~~~

For systems at equilibrium without flow (:math:`\dot{\gamma} = 0`):

.. math::

   c_2(t_1, t_2) = 1 + \beta \exp\left[-q^2 J(t_1, t_2)\right]

This simplified form captures pure diffusive dynamics without advection.

Flowing Systems
~~~~~~~~~~~~~~~

For systems under laminar shear flow:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \beta \exp\left[-q^2 J(t_1, t_2)\right] \times
   \text{sinc}^2\left[\frac{qh}{2\pi} \Gamma(t_1, t_2) \cos(\phi - \phi_0)\right]

The sinc function introduces angular anisotropy aligned with the flow direction,
with the decorrelation rate depending on the shear rate amplitude.

Chi-Squared Objective Function
------------------------------

Parameter estimation proceeds by minimizing the chi-squared objective:

.. math::

   \chi^2(\boldsymbol{\theta}) = \sum_{i,j} \frac{\left[c_2^{\text{exp}}(\phi_i, t_j) - c_2^{\text{model}}(\phi_i, t_j; \boldsymbol{\theta})\right]^2}{\sigma_{ij}^2}

where:

* :math:`\boldsymbol{\theta}`: Parameter vector
* :math:`c_2^{\text{exp}}`: Experimental correlation function
* :math:`c_2^{\text{model}}`: Theoretical model prediction
* :math:`\sigma_{ij}`: Measurement uncertainty at each data point

Uncertainty Quantification
--------------------------

Parameter Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~

Confidence intervals are computed from the covariance matrix:

.. math::

   \boldsymbol{\theta}_{\text{CI}} = \boldsymbol{\theta}_{\text{opt}} \pm t_{\alpha/2} \sqrt{\text{diag}(\mathbf{C})}

where :math:`\mathbf{C}` is the parameter covariance matrix estimated from the
inverse Hessian at the optimum.

Goodness of Fit
~~~~~~~~~~~~~~~

The reduced chi-squared statistic assesses fit quality:

.. math::

   \chi^2_{\text{red}} = \frac{\chi^2}{N - p}

where :math:`N` is the number of data points and :math:`p` is the number of parameters.
Values near 1.0 indicate good fit quality; values >> 1 suggest underestimated
uncertainties or model inadequacy.

Numerical Implementation
------------------------

The package implements these equations using JAX for high-performance computation:

**JIT Compilation:**
All core computational kernels are JIT-compiled for optimal performance:

- ``compute_g2_scaled``: Computes the full :math:`c_2` correlation with per-angle scaling
- ``compute_residuals``: Calculates residuals for optimization
- ``compute_chi_squared``: Fast chi-squared evaluation

**Vectorized Operations:**
Time integrals and correlation matrices are computed using vectorized JAX operations
for efficient memory access and parallelization.

**Numerical Stability:**
Special handling for edge cases:

- Sinc function: Taylor expansion near zero argument
- Exponential overflow: Argument clamping for numerical stability
- Division by zero: Epsilon regularization

See :doc:`computational_methods` for implementation details.

References
----------

See :doc:`citations` for complete citation information.
