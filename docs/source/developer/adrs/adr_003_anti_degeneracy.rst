.. _adr_anti_degeneracy:

ADR-003: Anti-Degeneracy Layer for Laminar Flow
================================================

:Status: Accepted
:Date: 2024–2025
:Deciders: Core team

Context
-------

The laminar-flow model fits the two-time correlation function at multiple azimuthal angles
simultaneously:

.. math::

   c_2(\phi_k, t_1, t_2; \theta)
   = c_\mathrm{offset}(\phi_k)
   + \beta(\phi_k) \exp(-q^2\mathcal{D}) \operatorname{sinc}^2(\ldots)

Each angle :math:`\phi_k` has its own contrast :math:`\beta(\phi_k)` and offset
:math:`c_\mathrm{offset}(\phi_k)` arising from the detector geometry and beam coherence.

If these angle-specific parameters are treated as free variables in the optimization (one pair
per angle, :math:`2 \times 23 = 46` extra free parameters for 23 angles), the NLSQ landscape
has a **degenerate flat direction**: any increase in :math:`D_0` can be compensated by
increasing all :math:`\beta(\phi_k)` by a corresponding factor, without changing the
residual.

This degeneracy causes optimizers to return unphysical solutions with :math:`D_0 \approx 0`
and inflated contrasts, or vice versa. It is not a numerical artifact — it is a genuine
identifiability issue rooted in the functional form of the model.


Decision
--------

Homodyne implements a **five-layer anti-degeneracy defense system** coordinated by
:class:`~homodyne.optimization.nlsq.anti_degeneracy_controller.AntiDegeneracyController`:

**Layer 1: Fourier/Constant Reparameterization**
   Transforms the parameter space to remove the flat direction. In ``auto`` mode: the
   per-angle contrasts and offsets are summarized by a single pair
   :math:`(\bar{\beta}, \bar{c}_\mathrm{offset})`, initialized from quantile estimates and
   jointly optimized with the 7 physical parameters.

**Layer 2: Hierarchical Optimization**
   First stage: fix scaling parameters at quantile estimates, optimize only the 7 physical
   parameters. Second stage: jointly optimize all 9 parameters from the Layer-1 solution.
   This avoids the flat landscape during the physically important first phase.

**Layer 3: Adaptive CV-based Regularization**
   Penalizes solutions where the per-angle residual coefficient of variation (CV) is
   anomalously large, which signals that the optimizer is "fitting noise" in the
   degeneracy direction rather than the physics signal.

**Layer 4: Gradient Collapse Monitoring**
   Detects when gradient norms approach zero (flat landscape symptom). On detection:
   perturbs the current iterate and restarts from the new point.

**Layer 5: Shear-Sensitivity Weighting**
   Upweights data points from angles near :math:`\phi = 0` (parallel to flow), where the
   sinc term is most sensitive to :math:`\dot{\gamma}_0`. This provides additional
   information to break the :math:`(D_0, \dot{\gamma}_0)` degeneracy.

The five layers are controlled by a single YAML setting:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # "auto" activates all 5 layers

See :ref:`theory_anti_degeneracy` for the mathematical details of each mode.


Rationale
---------

**1. The degeneracy is generic and unavoidable without intervention**

The degeneracy arises from the Siegert relation: :math:`c_2 = 1 + \beta |c_1|^2`. If
:math:`|c_1|^2 \to |c_1|^2 / k` and :math:`\beta \to k\beta`, the model is unchanged.
This scale invariance is broken only when:

(a) The functional form of :math:`c_1` depends on :math:`\beta` (it does not in the
    standard Siegert model), or
(b) The per-angle contrasts are constrained to share structure (as in ``auto``/``fourier``
    mode).

Any NLSQ implementation without constraint (b) will find the degenerate manifold.

**2. Five layers are needed because each addresses a different failure mode**

- Layer 1 is the primary fix (algebraically reduces degrees of freedom).
- Layer 2 ensures the optimizer reaches the physical minimum before the contrast
  parameters are released.
- Layer 3 catches cases where Layer 1 is bypassed by large initial steps.
- Layer 4 handles restarts when the optimizer stalls in the degenerate region.
- Layer 5 improves conditioning so that the physical minimum is more distinct.

Testing showed that any single layer alone was insufficient for all dataset types.

**3. ``auto`` mode is the safe default**

Auto mode uses quantile-based initialization and only 2 extra parameters (not 46). It
is robust to noisy data, handles well-conditioned datasets efficiently (converges in 1
stage), and prevents the degeneracy in all tested configurations.

**4. Extensibility**

The ``AntiDegeneracyController`` is a thin orchestrator: each layer is an independently
testable module. Adding a new defense strategy requires only a new layer class and a
small update to the controller.


Consequences
------------

**Positive**:

- No degenerate solutions in production use of ``auto`` mode.
- Clear error messages and diagnostics when degeneracy is detected.
- All five layers are independently tested.
- ``constant`` mode allows ablation studies (disabling the optimization of scaling).

**Negative / Accepted trade-offs**:

- ``auto`` mode adds 2 parameters (9 vs 7) and 2 optimization stages.
- The 5-layer system adds code complexity: 5 modules, each with its own config class.
- New contributors must read this ADR and :ref:`theory_anti_degeneracy` to understand
  why the optimizer is structured this way.

**Operational advice**:

- Always use ``auto`` mode unless there is a specific reason to use another.
- If fitted :math:`D_0 \approx 0` and contrasts are large: degeneracy was not fully
  resolved; switch to ``auto`` if not already, or verify initial parameter guesses.


Alternatives Considered
-----------------------

**A. Constrain beta to a fixed value**

Fix :math:`\beta(\phi_k) = \beta_\mathrm{fixed}` for all angles. Simple but introduces
systematic bias when true contrast varies across angles (which it always does at some level).

**B. Regularize beta toward equality**

Add an L2 penalty :math:`\lambda \sum_k (\beta_k - \bar{\beta})^2` to the objective.
Requires tuning :math:`\lambda`; hard to set automatically without domain knowledge.

**C. Sequential fitting (per-angle then joint)**

Fit each angle independently to get :math:`\beta(\phi_k)`, then fix contrasts and fit
physical parameters jointly. Fails when per-angle data is insufficient for independent
fits (small datasets, few time points).

**D. Bayesian priors on beta**

In the CMC model, use a hyperprior :math:`\beta_k \sim \mathcal{N}(\mu_\beta, \sigma_\beta)`.
This is the correct Bayesian approach but does not help the NLSQ point estimate, which is
needed first. CMC with NLSQ warm-start already benefits from having resolved the degeneracy.


.. seealso::

   - :ref:`theory_anti_degeneracy` — mathematical formulation of all modes
   - :mod:`homodyne.optimization.nlsq.anti_degeneracy_controller` — implementation
   - :ref:`adr_nlsq_primary` — why NLSQ must succeed before CMC
