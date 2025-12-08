Theoretical Framework
=====================

This section provides the mathematical foundation for the homodyne scattering analysis
implemented in the package, based on the theoretical framework developed by He et al. (2024).

Mathematical Foundation
-----------------------

Core Correlation Equation
~~~~~~~~~~~~~~~~~~~~~~~~~

The homodyne intensity correlation uses per-angle scaling:

.. math::

   c_2(\phi, t_1, t_2) = \text{offset} + \text{contrast} \times \left[c_1(\phi, t_1, t_2)\right]^2

with a separable field correlation:

.. math::

   c_1(\phi, t_1, t_2) = c_1^{\text{diff}}(t_1, t_2) \times c_1^{\text{shear}}(\phi, t_1, t_2)

Diffusion Contribution
~~~~~~~~~~~~~~~~~~~~~~

.. math::

   c_1^{\text{diff}}(t_1, t_2) = \exp\!\left[-\frac{q^2}{2} \int\limits_{|t_2 - t_1|} D(t') \, dt'\right]

Shear Contribution
~~~~~~~~~~~~~~~~~~

.. math::

   c_1^{\text{shear}}(\phi, t_1, t_2) = \left[\operatorname{sinc}\!\big(\Phi(\phi, t_1, t_2)\big)\right]^2

with

.. math::

   \Phi(\phi, t_1, t_2) = \frac{1}{2\pi} \, q \, L \, \cos(\phi_0 - \phi) \, \int\limits_{|t_2 - t_1|} \dot{\gamma}(t') \, dt'

Time-Dependent Transport Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   D(t) = D_0 \, t^{\alpha} + D_{\text{offset}}

.. math::

   \dot{\gamma}(t) = \dot{\gamma}_0 \, t^{\beta} + \dot{\gamma}_{\text{offset}}

Parameter Sets
--------------

**Static Mode (3 parameters)**

- :math:`D_0`: Reference diffusion coefficient [Å²/s]
- :math:`\alpha`: Diffusion time-dependence exponent [-]
- :math:`D_{\text{offset}}`: Baseline diffusion [Å²/s]
- Shear parameters set to zero, :math:`\phi_0 = 0` (irrelevant)

**Laminar Flow Mode (7 parameters)**

- :math:`D_0`, :math:`\alpha`, :math:`D_{\text{offset}}`
- :math:`\dot{\gamma}_0`: Reference shear rate [s⁻¹]
- :math:`\beta`: Shear rate time-dependence exponent [-]
- :math:`\dot{\gamma}_{\text{offset}}`: Baseline shear rate [s⁻¹]
- :math:`\phi_0`: Angular offset [degrees]

Experimental Parameters
-----------------------

- :math:`q`: Scattering wavevector magnitude [Å⁻¹]
- :math:`L`: Characteristic length scale / gap size [Å]
- :math:`\phi`: Scattering angle [degrees]
- :math:`dt`: Time step between frames [s/frame]
- :math:`\alpha`: Diffusion scaling exponent (dimensionless)

  - :math:`\alpha = 0`: Normal diffusion
  - :math:`\alpha < 0`: Subdiffusion (constrained motion)
  - :math:`\alpha > 0`: Superdiffusion (enhanced motion)

- :math:`D_{\text{offset}}`: Additive offset for baseline correction

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

Numerical Evaluation (NLSQ vs CMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While the analytic expressions above are useful for intuition, the code evaluates
the time integrals numerically using a cumulative trapezoid built on the discrete
time grid:

* Build trapezoid averages between adjacent samples: ``trap_avg[k] = 0.5*(f[k] + f[k+1])``.
* Cumulative sum of those averages (no ``dt`` applied here): ``cumsum[0]=0`` and
  ``cumsum[i] = Σ_{k=0}^{i-1} trap_avg[k]``.
* The integral between any two frames is the absolute difference of cumulative sums:
  ``|cumsum[i2] - cumsum[i1]|`` (a smooth abs is used for JAX gradients). The actual
  ``dt`` scaling is applied outside via the precomputed physics factors
  (``wavevector_q_squared_half_dt`` and ``sinc_prefactor``).

Implementation differences:

* NLSQ (meshgrid): builds a full cumulative-trapezoid matrix once on the 1D grid,
  then uses ``|cumsum[i] - cumsum[j]|`` for every meshgrid pair (see
  ``homodyne/core/physics_nlsq.py::_create_time_integral_matrix_impl_jax``).
* CMC (element-wise): builds the same cumulative trapezoid on the 1D grid, maps
  each pooled ``(t1, t2)`` pair to grid indices with ``searchsorted``, and uses the
  same absolute cumulative difference per pair (see
  ``homodyne/core/physics_cmc.py``). This replaces the old single-endpoint trapezoid,
  ensuring multi-step intervals sum all intermediate trapezoids just like NLSQ.

Concrete example (uniform grid, no ``dt`` scaling shown):

* Samples: ``f = [f0, f1, f2]``.
* Trapezoid averages: ``trap_avg = [0.5(f0+f1), 0.5(f1+f2)]``.
* Cumulative sums: ``cumsum = [0, 0.5(f0+f1), 0.5(f0+f1)+0.5(f1+f2)]``.
* Interval ``[t0, t2]`` uses all interior trapezoids:
  ``|cumsum[2] - cumsum[0]| = 0.5(f0+f1) + 0.5(f1+f2)``.
* Interval ``[t1, t2]`` uses only the last trapezoid:
  ``|cumsum[2] - cumsum[1]| = 0.5(f1+f2)``.

CMC applies a smooth absolute (``sqrt(diff**2 + eps)``) so gradients stay finite
for zero-length intervals; NLSQ uses the same smooth-abs on the meshgrid matrix.

Parameter Space and Priors (CMC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CMC builds priors from the ``parameter_space`` section of the YAML using
``homodyne.config.parameter_space`` and ``homodyne.optimization.cmc.priors``:

* Bounds are required and shared with NLSQ. Missing bounds fall back to
  package defaults (``contrast`` [0.0, 1.0], ``offset`` [0.5, 1.5], others [0.0, 1.0]).
  Those bounds are enforced twice: NLSQ clamps the optimizer to them, and CMC
  samples only from distributions truncated to that interval (NumPyro uses the
  bounds directly in ``Uniform``/``TruncatedNormal`` draws).
* If a prior is not specified for a parameter, a ``TruncatedNormal`` is built with
  ``mu`` at the interval midpoint and ``sigma`` at one-quarter of the width.
  If no prior spec is found at runtime, the fallback is ``Uniform(min, max)``.
* Supported prior types: ``TruncatedNormal``, ``Uniform``, ``LogNormal``,
  ``HalfNormal``, ``Normal``, ``BetaScaled`` (scaled Beta on [min, max], with
  concentrations inferred from ``mu``/``sigma``).
* Per-angle parameters ``contrast_i`` / ``offset_i`` inherit the base bounds/priors
  from a single ``contrast``/``offset`` entry unless per-angle overrides are provided.

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
- ``compute_chi_squared``: Fast chi-squared evaluation (includes residual calculation)

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
