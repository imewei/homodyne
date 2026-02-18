.. _adr_per_angle_scaling:

ADR-005: Per-Angle Scaling Modes
==================================

:Status: Accepted
:Date: 2025
:Deciders: Core team

Context
-------

See :ref:`adr_anti_degeneracy` for the primary ADR on the anti-degeneracy system.

This ADR documents specifically the decision to expose **four distinct per-angle scaling
modes** (``auto``, ``constant``, ``individual``, ``fourier``) rather than a single strategy.

The per-angle scaling modes control how the angle-dependent speckle contrast :math:`\beta(\phi_k)`
and offset :math:`c_\mathrm{offset}(\phi_k)` are handled in NLSQ optimization. Each mode
makes a different assumption about the structure of angular contrast variation and provides a
different trade-off between flexibility and identifiability.


Decision
--------

Expose all four modes through a single YAML configuration key:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # "auto" | "constant" | "individual" | "fourier"

The modes differ in the number of free parameters and the constraints imposed:

.. list-table::
   :header-rows: 1
   :widths: 18 18 64

   * - Mode
     - Free params (23 angles)
     - Constraint / assumption
   * - ``constant``
     - 7
     - Scaling fixed from quantile estimate; no optimization
   * - ``auto``
     - 9
     - Quantile-initialized averaging; 2 shared scaling params optimized
   * - ``fourier``
     - 17
     - Smooth angular variation; K=2 Fourier series (5 contrast + 5 offset coefficients)
   * - ``individual``
     - 53
     - No constraint; 46 per-angle params freely optimized (high degeneracy risk)

The default is ``auto``. All modes are documented, tested, and supported.


Rationale
---------

**1. No single mode is optimal for all experiments**

The physical contrast variation across azimuthal angles depends on:

- Beam shape (round vs. elliptical focus).
- Detector geometry (pixel size, distance).
- Sample anisotropy (crystalline vs. amorphous).
- Alignment quality (angular resolution of the detector sectors).

For a well-aligned, isotropic sample with a round beam, contrast variation across angles
is small and ``constant`` or ``auto`` mode is appropriate. For a highly anisotropic sample
or a detector with known angle-dependent efficiency, ``fourier`` or ``individual`` may be
better.

Exposing all modes allows users to perform model selection (e.g., compare AIC/BIC) and
choose the most appropriate constraint for their experiment.

**2. ``auto`` as a safe default covers the majority of cases**

The ``auto`` mode:

- Uses only 2 extra parameters beyond the 7 physical parameters.
- Initializes from data (quantile-based estimates), not from zero.
- Is robust to noisy data and small datasets.
- Never produces degenerate solutions in testing.

Setting ``auto`` as the default means that users who do not read the documentation get
a correct result.

**3. ``fourier`` mode provides interpretable intermediate complexity**

If contrast varies smoothly with angle (as expected from beam geometry), the Fourier
expansion is a natural basis. The constraint that contrast is a smooth function of
:math:`\phi` is physically motivated and reduces parameters from 46 (``individual``) to
10 (``fourier`` K=2) while fitting the variation.

**4. ``individual`` mode serves as an upper bound for comparison**

Running ``individual`` mode after ``auto`` mode allows users to verify that the 2-parameter
approximation is adequate. If ``individual`` mode produces significantly different results,
it signals that the contrast variation is not well-captured by the averaged model.

**5. ``constant`` mode enables ablation studies**

By fixing all scaling parameters, ``constant`` mode isolates the 7 physical parameters for
sensitivity analysis and cross-validation. It is also the fastest mode and useful for
real-time beamline analysis when speed is more important than full accuracy.


Consequences
------------

**Positive**:

- Users can match the constraint level to their experimental conditions.
- All modes are validated against synthetic data with known ground-truth parameters.
- Mode selection is transparent (single YAML key, documented trade-offs).

**Negative / Accepted trade-offs**:

- Four modes increases the documentation burden and test matrix.
- ``individual`` mode can produce degenerate results if used without understanding the
  identifiability issue. The documentation warns against it explicitly.
- Fourier mode requires choosing :math:`K` (number of harmonics); the current default
  :math:`K=2` is reasonable but not universally optimal.

**Guidance for mode selection**:

1. Start with ``auto`` (default).
2. If residuals are structured (systematic angle-dependent pattern): try ``fourier``.
3. If an independent calibration of contrast is available: use ``constant`` with the
   calibrated values as fixed inputs.
4. Never use ``individual`` as a first attempt; only as post-hoc diagnostic.


Alternatives Considered
-----------------------

**A. Single mode (auto only)**

Simpler user interface. Rejected because: limits the ability to perform model selection
and may not be optimal for all experimental configurations.

**B. Hierarchical Bayesian model for contrast**

Model :math:`\beta(\phi_k) \sim \mathcal{N}(\mu_\beta, \sigma_\beta^2)` with hyperpriors.
Correct Bayesian treatment. Rejected because: this is only available in the CMC path;
for the NLSQ point estimate (which runs first), a simple parameterization is needed.
The CMC model could implement this hierarchy in a future version.

**C. Automatic model selection (AIC/BIC)**

Run all modes and select based on information criterion. Rejected because: would quadruple
the optimization time; users who want model selection can run the modes manually.


.. seealso::

   - :ref:`theory_anti_degeneracy` — mathematical details of all modes
   - :ref:`adr_anti_degeneracy` — primary ADR on the 5-layer system
   - :mod:`homodyne.optimization.nlsq.anti_degeneracy_controller` — implementation
   - :mod:`homodyne.optimization.nlsq.fourier_reparam` — Fourier mode
