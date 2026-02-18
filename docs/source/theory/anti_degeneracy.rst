.. _theory_anti_degeneracy:

Per-Angle Scaling and Anti-Degeneracy
=======================================

A subtle but critical issue arises when fitting the laminar-flow model simultaneously to
data from multiple azimuthal angles :math:`\phi`: the optical contrast :math:`\beta(\phi)`
and the constant offset in :math:`c_2` are mathematically degenerate with the physical
parameters :math:`D_0` and :math:`\dot{\gamma}_0`. This page explains the degeneracy
mechanism and the per-angle scaling strategy that prevents it.

.. contents:: Contents
   :local:
   :depth: 2


The Parameter Absorption Degeneracy
-------------------------------------

The laminar-flow model at a single angle :math:`\phi_k` is:

.. math::
   :label: c2_single_angle

   c_2(\phi_k, t_1, t_2; \theta)
   \;=\; \underbrace{c_\mathrm{offset}(\phi_k)}_{\text{angle-dep. offset}}
   + \underbrace{\beta(\phi_k)}_{\text{angle-dep. contrast}}
   \exp\!\left(-q^2 \mathcal{D}(t_1,t_2)\right)
   \operatorname{sinc}^2\!\left(\tfrac{qh\cos\phi_k}{2\pi}\Gamma(t_1,t_2)\right)

The angle-dependent contrast :math:`\beta(\phi_k)` arises from beam partial coherence,
pixel geometry, and sample inhomogeneities. If :math:`\beta(\phi_k)` is treated as
**free parameters** to be optimized independently at each angle, the optimization landscape
has a flat direction:

**Degeneracy direction**: Increasing :math:`D_0 \to D_0 + \delta` and simultaneously
increasing :math:`\beta(\phi_k) \to \beta(\phi_k) \cdot e^{q^2\delta t_\mathrm{ref}}` for all
:math:`k` produces identical :math:`c_2` values. The physical parameters are **not
identifiable** from the per-angle contrasts without a constraint.

This degeneracy is generic whenever:

1. Per-angle :math:`\beta(\phi_k)` are freely optimized.
2. The number of angle-specific parameters exceeds the information content per angle.
3. The diffusion contribution and the contrast contribution have the same functional form.

.. warning::

   Using ``per_angle_mode: "individual"`` (free :math:`\beta(\phi_k)` for all :math:`k`)
   without strong priors will produce **unphysical** solutions where :math:`D_0 \approx 0`
   with inflated per-angle contrasts, or vice versa. Always verify that fitted :math:`D_0`
   is physically reasonable.


Per-Angle Modes
---------------

The anti-degeneracy system provides four modes, configured in YAML:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # recommended

**Mode: ``constant``**

The simplest fix: estimate :math:`\beta(\phi_k)` and :math:`c_\mathrm{offset}(\phi_k)` from
the data using a quantile estimator *before* optimization, then hold them **fixed** during
the NLSQ run. Only the 7 physical parameters are optimized.

.. math::

   \hat{\beta}(\phi_k) \;=\; \hat{q}_{0.95}\!\left[c_2(\phi_k, t_i, t_i)\right] - 1
   \qquad \text{(diagonal quantile)}

This prevents degeneracy by construction, but may introduce systematic bias if
:math:`\hat{\beta}` is inaccurate (e.g., when the zero-lag estimator is contaminated by
noise).

**Mode: ``individual``**

Each angle has independently optimized :math:`\beta(\phi_k)` and offset. This adds
:math:`2 N_\phi` free parameters (where :math:`N_\phi` is the number of angle sectors).
For :math:`N_\phi = 23`, this gives 53 total parameters.

Use **only** when:

- Physical contrast variation across angles is expected and significant.
- Excellent initial estimates for :math:`\beta(\phi_k)` are available (e.g., from a prior
  ``constant`` run).
- The dataset is large enough to constrain 53 parameters.

**Mode: ``fourier``**

Models the angular variation of contrast as a **truncated Fourier series** (default
:math:`K = 2`):

.. math::
   :label: fourier_contrast

   \beta(\phi_k) \;=\; \bar{\beta}
   + \sum_{\ell=1}^K \left[a_\ell \cos(\ell\phi_k) + b_\ell \sin(\ell\phi_k)\right]

This adds :math:`2K + 1 = 5` contrast parameters (for :math:`K=2`) instead of :math:`2N_\phi`.
The physical constraint is that contrast must vary smoothly with angle, as expected from beam
coherence geometry.

Total parameters: :math:`7 + 2K + 2 = 7 + 2K + \text{offsets}` (approximately 17 for
:math:`K=2`).

**Mode: ``auto`` (recommended)**

Examines the number of angle sectors :math:`N_\phi`:

- If :math:`N_\phi \geq 3`: uses **averaged scaling** — computes per-angle
  :math:`\beta(\phi_k)` and offset by quantile estimation, averages them to get a single
  :math:`\bar{\beta}` and :math:`\bar{c}_\mathrm{offset}`, and optimizes these 2 averaged
  values alongside the 7 physical parameters (total 9 parameters).
- If :math:`N_\phi < 3`: falls back to ``individual``.

**Auto mode breaks the degeneracy** because:

1. The averaged scaling is initialized from data (not zero), providing a good starting point.
2. Only 2 scaling parameters are free (not :math:`2N_\phi`), so the problem is well-posed.
3. The quantile-based initialization ensures the optimizer starts near the true solution.


Mathematical Formulation
-------------------------

Let :math:`\hat{\mu}_k` and :math:`\hat{\sigma}_k` be the estimated mean contrast and offset
for angle :math:`\phi_k`. The auto-mode averaging is:

.. math::

   \bar{\beta} \;=\; \frac{1}{N_\phi}\sum_{k=1}^{N_\phi}\hat{\beta}_k,
   \qquad
   \bar{c}_\mathrm{offset} \;=\; \frac{1}{N_\phi}\sum_{k=1}^{N_\phi}\hat{c}_{\mathrm{offset},k}

The model then becomes:

.. math::
   :label: c2_auto_mode

   c_2(\phi_k, t_1, t_2; \theta, \bar{\beta}, \bar{c}_\mathrm{offset})
   \;=\; \bar{c}_\mathrm{offset}
   + \bar{\beta}
     \exp\!\left(-q^2\mathcal{D}(t_1,t_2)\right)
     \operatorname{sinc}^2\!\left(\tfrac{qh\cos\phi_k}{2\pi}\Gamma(t_1,t_2)\right)

where :math:`\theta = (D_0, \alpha, D_\mathrm{offset}, \dot{\gamma}_0, \beta_\gamma, \dot{\gamma}_\mathrm{offset}, \phi_0)`
are the 7 physical parameters and :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})` are the 2
optimized scaling parameters.


Parameter Count Summary
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Mode
     - Parameters (23 angles)
     - Notes
   * - ``constant``
     - 7
     - Scaling fixed; may bias if quantile estimate poor
   * - ``auto``
     - 9
     - **Recommended**: 7 physical + 2 averaged scaling
   * - ``fourier`` (:math:`K=2`)
     - 17
     - 7 physical + 10 Fourier coefficients (contrast + offset)
   * - ``individual``
     - 53
     - 7 physical + 46 per-angle parameters; risk of degeneracy

When to Use Each Mode
----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Recommended when
   * - ``auto``
     - Default for all production runs. :math:`N_\phi \geq 3` (virtually always true).
   * - ``constant``
     - Debugging. When physical contrast variation is known to be negligible and
       quantile estimate is reliable. Fastest convergence.
   * - ``fourier``
     - Strong azimuthal contrast variation expected (anisotropic beam or sample).
       More parameters than ``auto``; verify with AIC/BIC.
   * - ``individual``
     - Post-hoc analysis only. Use ``auto`` result as initialization.
       Never as a first attempt.

Implementation Reference
-------------------------

The anti-degeneracy system is implemented in:

- :mod:`homodyne.optimization.nlsq.anti_degeneracy_controller` — orchestrator
- :mod:`homodyne.optimization.nlsq.fourier_reparam` — Fourier mode
- :mod:`homodyne.optimization.nlsq.hierarchical` — hierarchical optimizer
- :mod:`homodyne.optimization.nlsq.adaptive_regularization` — CV-based regularization
- :mod:`homodyne.optimization.nlsq.gradient_monitor` — gradient collapse detection
- :mod:`homodyne.optimization.nlsq.shear_weighting` — shear-sensitivity weighting

The five-layer defense system (beyond per-angle mode selection) includes:

1. **Fourier/Constant Reparameterization** — transforms the parameter space to remove
   the flat direction algebraically.
2. **Hierarchical Optimization** — first fits physical parameters with fixed scaling,
   then jointly refines all.
3. **Adaptive CV-based Regularization** — penalizes solutions where the coefficient of
   variation of residuals per-angle is anomalously large.
4. **Gradient Collapse Monitoring** — detects when gradient norms collapse (flat landscape)
   and restarts with perturbed initialization.
5. **Shear-Sensitivity Weighting** — upweights data points near :math:`\phi = 0` where
   the shear term is most informative.

See :ref:`adr_anti_degeneracy` for the architectural rationale.


.. seealso::

   - :ref:`theory_homodyne_scattering` — the full :math:`c_2` equation
   - :ref:`adr_anti_degeneracy` — architectural decision record
   - :mod:`homodyne.optimization.nlsq.anti_degeneracy_controller` — implementation
