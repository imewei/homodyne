.. _theory_heterodyne_scattering:

Heterodyne Scattering: Multi-Component Models
=============================================

When a sample contains multiple scattering populations with different dynamics — for example,
a flowing region and a static (arrested) region — the intensity correlation function receives
contributions from cross-correlations between all component pairs. This page develops the
general :math:`N`-component heterodyne scattering formula from He et al. PNAS 2025 and
details the special cases most relevant to colloidal suspensions undergoing yielding.



Physical Motivation
-------------------

In **shear banding**, the sample separates into a flowing band and a slow (or arrested) band
coexisting in the scattering volume. Pure homodyne analysis (single component) cannot
distinguish:

1. A single component with intermediate effective :math:`\dot{\gamma}` and :math:`D`.
2. Two coexisting populations with different :math:`\dot{\gamma}_1`, :math:`\dot{\gamma}_2`, and
   scattering weights.

The heterodyne formula resolves this ambiguity by modelling the **mixture scattering** from
:math:`N` distinguishable populations explicitly.

.. note::

   "Heterodyne" here refers to the presence of multiple scattering components with different
   mean velocities, not to optical heterodyne detection with a local oscillator. The scattered
   field from different components interferes, producing oscillatory patterns in :math:`c_2`.


N-Component General Formula
----------------------------

Consider :math:`N` scattering populations. At time :math:`t`, component :math:`n` carries a
fraction :math:`x_n(t)` of the total scattering weight, has mean velocity
:math:`\langle v_n(t)\rangle`, and transport coefficient :math:`J_n(t)`.

The **N-component second-order correlation function** is:

.. math::
   :label: c2_N_component

   c_2(\mathbf{q}, t_1, t_2)
   \;=\; 1 + \frac{\beta}{f^2(t_1, t_2)}
     \sum_{n=1}^N \sum_{m=1}^N
       x_n(t_1)\,x_n(t_2)\,x_m(t_1)\,x_m(t_2)
       \cdot A_{nm}(t_1, t_2)

where the cross-correlation amplitude :math:`A_{nm}` is:

.. math::
   :label: A_nm

   A_{nm}(t_1, t_2)
   \;=\; \exp\!\left(-\frac{q^2}{2}\int_{t_1}^{t_2}\left[J_n(t_1,t')+J_m(t_1,t')\right]dt'\right)
     \cos\!\left(q\cos\phi\int_{t_1}^{t_2}\left[\langle v_n(t')\rangle - \langle v_m(t')\rangle\right]dt'\right)

and the normalization factor is:

.. math::
   :label: f2_normalization

   f^2(t_1, t_2) \;=\;
     \left[\sum_{n=1}^N x_n(t_1)^2\right]
     \left[\sum_{n=1}^N x_n(t_2)^2\right]

**Physical interpretation**: Each term :math:`A_{nm}` represents the interference between
components :math:`n` and :math:`m`. The cosine factor oscillates when the two components
drift apart (relative velocity :math:`\Delta v_{nm}` non-zero), producing characteristic
fringes in the :math:`c_2` matrix. The :math:`f^2` factor normalizes for the fact that
mixing reduces the maximum achievable contrast.


Two-Component Case: Static + Flowing
--------------------------------------

The simplest heterodyne case has **two components**:

- Component 1 (index ``s``): **static** (arrested), :math:`\langle v_s\rangle = 0`, transport :math:`J_s(t)`.
- Component 2 (index ``f``): **flowing**, :math:`\langle v_f(t)\rangle = \dot{\gamma}(t)\,r\cos\phi`,
  transport :math:`J_f(t)`.

With scattering weights :math:`x_s + x_f = 1`, the correlation function becomes:

.. math::
   :label: c2_2component

   c_2 \;=\; 1 + \frac{\beta}{\left(x_s^2 + x_f^2\right)^2}
   \Bigl[
     x_s^4\,A_{ss} + x_f^4\,A_{ff} + 2x_s^2 x_f^2\,A_{sf}
   \Bigr]

with:

.. math::

   A_{ss} &= \exp\!\left(-q^2 \mathcal{D}_s\right), \\
   A_{ff} &= \exp\!\left(-q^2 \mathcal{D}_f\right), \\
   A_{sf} &= \exp\!\left(-\frac{q^2(\mathcal{D}_s + \mathcal{D}_f)}{2}\right)
             \cos\!\left(q\cos\phi\int_{t_1}^{t_2}\langle v_f(t)\rangle\,dt\right)

The **cross term** :math:`A_{sf}` generates oscillations in :math:`c_2` as a function of lag
time :math:`\tau = t_2 - t_1`, with a period set by the inverse of the characteristic velocity
:math:`\langle v_f\rangle`.


Three-Component Case: Two Flowing + One Static
------------------------------------------------

The three-component case models a **shear-banded** state with two flowing bands and one
static band:

- Component ``s``: static (:math:`v = 0`)
- Component ``f1``: slow-flow band (:math:`\dot{\gamma}_1`)
- Component ``f2``: fast-flow band (:math:`\dot{\gamma}_2 > \dot{\gamma}_1`)

The :math:`c_2` matrix for three components is:

.. math::
   :label: c2_3component

   c_2 = 1 + \frac{\beta}{f^2}
   \Bigl[
     x_s^4\,A_{ss} + x_{f1}^4\,A_{f1f1} + x_{f2}^4\,A_{f2f2}
     + 2x_s^2 x_{f1}^2\,A_{sf1} + 2x_s^2 x_{f2}^2\,A_{sf2}
     + 2x_{f1}^2 x_{f2}^2\,A_{f1f2}
   \Bigr]

The three cross-terms :math:`A_{sf1}`, :math:`A_{sf2}`, :math:`A_{f1f2}` each oscillate at
a different frequency determined by the pairwise velocity differences
:math:`\Delta v_{ij} = \langle v_i\rangle - \langle v_j\rangle`.

This multi-frequency beating pattern is a direct **fingerprint of shear banding** and can
be distinguished from single-component laminar flow by the appearance of additional fringes
in the two-time :math:`c_2` matrix.


Normalization Factor f²
------------------------

The normalization factor :math:`f^2` defined in :eq:`f2_normalization` ensures that
:math:`c_2 \leq 1 + \beta` always holds. Its geometric interpretation is:

.. math::

   f^2 = \left\langle x^2(t_1)\right\rangle \left\langle x^2(t_2)\right\rangle
   \;\equiv\; P_2(t_1)\,P_2(t_2)

where :math:`P_2(t) = \sum_n x_n(t)^2` is the **participation ratio** (inverse Herfindahl
index) of the weight distribution. For a single component :math:`x_1 = 1`,
:math:`f^2 = 1` and Equation :eq:`c2_N_component` reduces to the homodyne formula. As the
weight disperses over many components, :math:`f^2 \to 1/N^2` and the effective contrast
:math:`\beta/f^2` increases accordingly.


Oscillatory Patterns as Diagnostic
------------------------------------

The presence of oscillatory fringes in measured :math:`c_2(t_1, t_2)` at fixed
:math:`t_1 - t_2 = \tau` is a direct diagnostic of **multi-component scattering**:

- **No oscillations**: single component, homodyne model applies.
- **One oscillation frequency**: two-component (static + flowing) system.
- **Multiple frequencies**: three or more components, or shear banding.

The frequency of oscillations gives the velocity difference between components:

.. math::

   \nu_{nm} \;=\; \frac{q\cos\phi\,|\langle v_n\rangle - \langle v_m\rangle|}{2\pi}

from which the individual component velocities can be extracted.


Comparison to Homodyne
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - Homodyne (single component)
     - Heterodyne (multi-component)
   * - Number of parameters
     - 7 (laminar_flow mode)
     - 7 + 2(N-1) per additional component
   * - :math:`c_2` shape
     - Monotone decay from diagonal
     - Oscillatory patterns
   * - Applicable to
     - Homogeneous laminar flow
     - Shear banding, mixed phases
   * - Analysis mode in package
     - ``laminar_flow``
     - Not yet in current version
   * - Paper reference
     - He et al. PNAS 2024
     - He et al. PNAS 2025

.. note::

   The multi-component heterodyne model is described theoretically in He et al. PNAS 2025
   for **attractive suspensions** exhibiting shear banding. The current homodyne package
   implements the single-component laminar flow model. Multi-component support is planned.


.. seealso::

   - :ref:`theory_homodyne_scattering` — single-component laminar flow model
   - :ref:`theory_correlation_functions` — Siegert relation and :math:`c_2` definition
   - :ref:`theory_yielding_dynamics` — physical context (attractive vs repulsive suspensions)
