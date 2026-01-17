Anti-Degeneracy Defense System (v2.17.0)
========================================

This section provides comprehensive documentation for the Anti-Degeneracy Defense System
introduced in Homodyne v2.9.0 and enhanced in v2.17.0 with quantile-based per-angle scaling.
The system addresses fundamental optimization challenges in laminar flow XPCS analysis
where structural degeneracy and gradient cancellation prevent accurate parameter estimation.

.. contents:: Contents
   :local:
   :depth: 3

Introduction
------------

When fitting XPCS data under laminar flow conditions with many azimuthal angles,
the optimizer often fails to correctly estimate shear parameters (particularly
``gamma_dot_t0``). This "parameter collapse" occurs because the optimization
landscape contains degenerate solutions where per-angle scaling parameters
absorb physical signals.

**Example Problem Scenario**:

- Dataset: 23 phi angles, 1M+ points per angle
- Expected: ``gamma_dot_t0 ≈ 10^-3`` (true physical value)
- Observed: ``gamma_dot_t0 → 10^-6`` (collapsed to lower bound)
- Root cause: Per-angle contrast/offset parameters absorb angle-dependent shear signal

The Anti-Degeneracy Defense System provides a five-layer solution:

.. list-table:: Defense Layers Summary
   :header-rows: 1
   :widths: 10 30 30 15 15

   * - Layer
     - Solution
     - Root Cause Addressed
     - Status
     - Effectiveness
   * - 1
     - Fourier Reparameterization
     - Structural Degeneracy
     - NEW
     - **High**
   * - 2
     - Hierarchical Optimization
     - Gradient Cancellation
     - NEW
     - **High**
   * - 3
     - Adaptive CV Regularization
     - Ineffective λ
     - ENHANCED
     - Medium
   * - 4
     - Gradient Collapse Detection
     - Runtime Monitoring
     - NEW
     - Medium
   * - 5
     - Shear-Sensitivity Weighting
     - Gradient Cancellation
     - NEW (v2.9.1)
     - **High**
   * - —
     - Data Shuffling
     - Sequential Bias
     - EXISTING
     - Medium

Theoretical Background
----------------------

Understanding the Root Causes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To understand why the defense system is necessary, we must examine the three
fundamental root causes of parameter collapse in laminar flow fitting.

Root Cause 1: Gradient Cancellation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shear term in the XPCS model introduces angle-dependent dynamics:

.. math::

   c_1(\phi, t_1, t_2) = \exp\left(-q^2 J(t_1, t_2)\right) \times
   \text{sinc}\left[\frac{qh}{2\pi} \Gamma(t_1, t_2) \cos(\phi - \phi_0)\right]

where :math:`\Gamma(t_1, t_2)` is the cumulative shear integral proportional to
``gamma_dot_t0``.

The gradient of the loss with respect to ``gamma_dot_t0`` is:

.. math::

   \frac{\partial L}{\partial \dot{\gamma}_0} \propto \sum_{\phi} \cos(\phi_0 - \phi) \times (\text{data terms})

**The Problem**: The term :math:`\cos(\phi_0 - \phi)` changes sign across different
angles. When summed globally (as in standard gradient descent), contributions from
different angles partially cancel, producing a weak net gradient.

.. code-block:: text

   Example with 8 equally-spaced angles (0°, 45°, 90°, ...):

   φ = 0°:   cos(0°)   = +1.0    ─┐
   φ = 45°:  cos(45°)  = +0.71   ─┼─ These partially cancel when summed
   φ = 90°:  cos(90°)  = 0.0     ─┤
   φ = 135°: cos(135°) = -0.71   ─┤
   φ = 180°: cos(180°) = -1.0    ─┘

   Net effect: Weak gradient signal for gamma_dot_t0

Root Cause 2: Structural Degeneracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With ``n_phi`` azimuthal angles, the per-angle scaling introduces ``2 × n_phi``
free parameters (contrast and offset for each angle). This creates an over-determined
fitting problem:

.. math::

   c_2^{\text{fitted}}(\phi_i) = \text{offset}_i + \text{contrast}_i \times [c_1(\phi_i)]^2

**The Problem**: Per-angle parameters can absorb nearly any angle-dependent signal.
When ``gamma_dot_t0`` produces angle-dependent :math:`c_1` values, the optimizer
finds it easier to adjust ``2 × 23 = 46`` per-angle parameters than to correctly
estimate 7 physical parameters.

.. code-block:: text

   Gradient magnitude comparison (typical 23-angle fit):

   |∇_{contrast}| ≈ 10⁻²  ─┐
   |∇_{offset}|   ≈ 10⁻²  ─┼─ ~92% of gradient magnitude
   |∇_{physical}| ≈ 10⁻⁴  ─┘   ~8% of gradient magnitude

   Result: Optimizer preferentially adjusts per-angle params

Root Cause 3: Ineffective Regularization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previous versions used group variance regularization with ``λ = 0.01``:

.. math::

   L_{\text{reg}} = \lambda \times \left[\text{Var}(\text{contrast}) + \text{Var}(\text{offset})\right] \times N

With typical MSE values and data sizes, this contributed only ~0.05% to the
total loss—providing no effective constraint on per-angle parameter variation.

**The Math**:

.. code-block:: text

   Typical values:
   - MSE ≈ 10⁻⁴
   - N_points = 23M
   - Var(contrast) ≈ 0.01 (10% std on mean of 0.5)

   L_data = MSE × N = 10⁻⁴ × 23×10⁶ = 2.3×10³
   L_reg  = 0.01 × 0.01 × 23×10⁶ = 2.3×10³

   Contribution = L_reg / L_total ≈ 0.0001 (0.01%)

   The regularization is effectively invisible to the optimizer!

Layer 1: Fourier Reparameterization
-----------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Instead of treating per-angle contrast and offset as independent parameters,
Layer 1 expresses them as truncated Fourier series:

.. math::

   \text{contrast}(\phi) = c_0 + \sum_{k=1}^{K} \left[c_k \cos(k\phi) + s_k \sin(k\phi)\right]

.. math::

   \text{offset}(\phi) = o_0 + \sum_{k=1}^{K} \left[o_k \cos(k\phi) + t_k \sin(k\phi)\right]

For Fourier order ``K=2``:

- Contrast: 5 coefficients [c₀, c₁, s₁, c₂, s₂]
- Offset: 5 coefficients [o₀, o₁, t₁, o₂, t₂]
- **Total: 10 Fourier coefficients vs 2×n_phi independent parameters**

Per-Angle Mode Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``per_angle_mode`` setting controls how per-angle parameters (contrast/offset) are handled.
This is the **most important configuration choice** for laminar flow fitting with many angles.

.. list-table:: per_angle_mode Options
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - Description
   * - ``constant``
     - **DEFAULT (v2.17.0+)** Per-angle scaling computed from quantile estimation and **FIXED** during optimization. Only 7 physical parameters are optimized. Most robust against degeneracy.
   * - ``individual``
     - Each angle has independent contrast and offset (2 × n_phi optimized params). Full flexibility but high degeneracy risk for n_phi > 6.
   * - ``fourier``
     - Contrast/offset expressed as truncated Fourier series (2 × (2K+1) optimized params). Enforces smooth angular variation, reduces degeneracy risk.
   * - ``auto``
     - Auto-selects between ``fourier`` (n_phi > threshold) and ``individual``. Does **NOT** select ``constant`` mode.

.. note::

   **v2.17.0 Change**: The default ``per_angle_mode`` is now ``"constant"``, which provides the
   most robust parameter estimation. Previous versions defaulted to ``"auto"``.

Mode Comparison
^^^^^^^^^^^^^^^

.. list-table:: per_angle_mode Comparison
   :header-rows: 1
   :widths: 16 14 14 14 14 14 14

   * - Aspect
     - constant
     - individual
     - fourier
     - auto
     - Recommendation
     - Notes
   * - **Optimized Params**
     - 7
     - 2N + 7
     - 10-17 + 7
     - Varies
     - constant
     - Lower = faster, more robust
   * - **Degeneracy Risk**
     - None
     - High (N > 6)
     - Medium
     - Medium
     - constant
     - None = cannot absorb shear signal
   * - **Flexibility**
     - Low
     - Maximum
     - Medium
     - Varies
     - depends
     - High flex = can fit noise
   * - **Speed**
     - Fastest
     - Slowest
     - Medium
     - Varies
     - constant
     - Fewer params = faster convergence
   * - **Angular Variation**
     - Fixed from data
     - Fully flexible
     - Smooth only
     - Varies
     - depends
     - Physical basis matters

Decision Flowchart
^^^^^^^^^^^^^^^^^^

Use this flowchart to select the appropriate ``per_angle_mode``:

.. code-block:: text

                        ┌─────────────────────────┐
                        │ Start: laminar_flow     │
                        │ mode with N phi angles  │
                        └───────────┬─────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │ Is experimental setup │
                        │ reasonably uniform    │
                        │ across angles?        │
                        └───────────┬───────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │ YES                                       │ NO
              ▼                                           ▼
    ┌─────────────────────┐               ┌───────────────────────────┐
    │ Use "constant"      │               │ Do angles have known      │
    │ (DEFAULT, v2.17.0+) │               │ smooth optical variation? │
    │                     │               │ (sample asymmetry, etc.)  │
    │ • Most robust       │               └─────────────┬─────────────┘
    │ • 7 params only     │                             │
    │ • Fastest           │               ┌─────────────┴─────────────┐
    └─────────────────────┘               │ YES (smooth)              │ NO (random)
                                          ▼                           ▼
                              ┌─────────────────────┐   ┌─────────────────────┐
                              │ Use "fourier"       │   │ Use "individual"    │
                              │                     │   │                     │
                              │ • Smooth variation  │   │ • Full flexibility  │
                              │ • N > 6 recommended │   │ • N ≤ 3 only        │
                              │ • 17 params (K=2)   │   │ • Needs Layer 2-5   │
                              └─────────────────────┘   └─────────────────────┘

Parameter Count Reduction
^^^^^^^^^^^^^^^^^^^^^^^^^

The different modes provide varying degrees of parameter reduction:

.. list-table:: Parameter Count Comparison
   :header-rows: 1
   :widths: 12 18 18 18 18 16

   * - n_phi
     - Individual
     - Fourier (K=2)
     - Constant
     - Reduction
     - Default Mode
   * - 2
     - 4
     - 4*
     - 0†
     - 100%
     - Constant
   * - 3
     - 6
     - 6*
     - 0†
     - 100%
     - Constant
   * - 6
     - 12
     - 10
     - 0†
     - 100%
     - Constant
   * - 10
     - 20
     - 10
     - 0†
     - 100%
     - Constant
   * - 23
     - 46
     - 10
     - 0†
     - **100%**
     - Constant
   * - 100
     - 200
     - 10
     - 0†
     - **100%**
     - Constant

| \* For n_phi ≤ 2×(order+1), Fourier mode provides no reduction
| † Constant mode uses quantile-based fixed scaling: per-angle params are computed once and not optimized

Constant Mode with Quantile-Based Estimation (v2.17.0+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Constant mode** is now the **default** (v2.17.0+) and provides the most aggressive
parameter reduction. Instead of optimizing per-angle contrast/offset, they are
**estimated once from data quantiles** and treated as fixed during optimization.

**Physics Foundation**:

The Siegert relation connects intensity autocorrelation (C2) to field autocorrelation (g1):

.. math::

   C_2 = \text{contrast} \times g_1^2 + \text{offset}

The key insight is that g1 decays with time lag:

- At **small time lags** (Δt → 0): g1² ≈ 1, so C2 ≈ contrast + offset (the "ceiling")
- At **large time lags** (Δt → ∞): g1² → 0, so C2 → offset (the "floor")

**Quantile-Based Estimation Algorithm**:

For each phi angle independently:

.. code-block:: text

   Step 1: Partition data by time lag
   ┌──────────────────────────────────────────────────────────────┐
   │ Time Lag Distribution for Angle φᵢ                          │
   │                                                              │
   │    ▼ Small lags (bottom 20%)    ▼ Large lags (top 20%)      │
   │ ├─────────────────────────────────────────────────────────┤ │
   │ └──── Used for ceiling ────┘       └──── Used for floor ───┘│
   └──────────────────────────────────────────────────────────────┘

   Step 2: Estimate OFFSET (from large-lag region where g1² → 0)
   offset = 10th percentile of C2 values at large lags

   Step 3: Estimate CONTRAST (from small-lag region where g1² ≈ 1)
   ceiling = 90th percentile of C2 values at small lags
   contrast = ceiling - offset

   Step 4: Apply bounds clipping
   offset = clip(offset, offset_bounds)
   contrast = clip(contrast, contrast_bounds)

**Why Quantiles Instead of Min/Max**:

1. **Outlier Robustness**: Quantiles ignore extreme values from noise spikes
2. **Noise Tolerance**: Less sensitive to measurement noise
3. **Systematic Error Handling**: Avoids contamination from partially decayed points

**Default Quantile Parameters**:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``lag_floor_quantile``
     - 0.80
     - Top 20% of lags for floor estimation
   * - ``lag_ceiling_quantile``
     - 0.20
     - Bottom 20% of lags for ceiling estimation
   * - ``value_quantile_low``
     - 0.10
     - 10th percentile for robust floor
   * - ``value_quantile_high``
     - 0.90
     - 90th percentile for robust ceiling

**Parameter Reduction**:

.. code-block:: text

   CONSTANT MODE FIXED SCALING (v2.17.0)
   =====================================

   Before optimization:
       Compute per-angle scaling from data quantiles
       fixed_contrast[n_phi] ← quantile estimation
       fixed_offset[n_phi]   ← quantile estimation

   During optimization:
       Parameter vector: [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
                         (7 params only for laminar_flow)

   After optimization:
       Expand for output: [fixed_contrast..., fixed_offset..., physical_opt]
                          (53 values for backward compatibility)

**When to Use Constant Mode** (Default):

- Most XPCS datasets where detector response is reasonably uniform
- Large datasets (>1M points) where parameter reduction improves convergence
- Multi-start optimization with many angles (tractable parameter count)
- When per-angle variation is primarily noise, not physics

**When to Switch to Other Modes**:

- Use ``fourier`` if smooth angular variation is physically expected (sample asymmetry)
- Use ``individual`` for very small n_phi (≤ 3) or when full flexibility is needed

**Configuration**:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         enable: true
         per_angle_mode: "constant"      # Default in v2.17.0+
         # per_angle_mode: "auto"        # Use "auto" to select fourier/individual
         # per_angle_mode: "fourier"     # Use "fourier" for smooth angular variation
         # per_angle_mode: "individual"  # Use "individual" for full flexibility

**Impact on Multi-Start Optimization**:

Constant mode makes multi-start optimization tractable for many-angle datasets:

.. list-table:: Multi-Start Tractability (23-angle laminar_flow)
   :header-rows: 1
   :widths: 25 20 25 30

   * - Mode
     - n_params
     - Min n_starts
     - Recommended n_starts
   * - individual
     - 53
     - 53
     - 100-150 (expensive)
   * - fourier
     - 17
     - 17
     - 20-40 (moderate)
   * - **constant (default)**
     - **7**
     - **7**
     - **10-15 (tractable)**

Why Constant Mode Works Best (v2.17.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Maximum Parameter Reduction**: Only 7 physical parameters are optimized.
   Per-angle scaling is fixed from data, eliminating all 46 per-angle parameters.

2. **No Degeneracy Risk**: Since per-angle parameters are not optimized,
   they cannot absorb physical signals that belong to shear parameters.

3. **Data-Driven Initialization**: Quantile-based estimation uses actual
   experimental data, not arbitrary defaults, for contrast/offset values.

4. **Robust to Noise**: Quantile-based estimation is more robust to outliers
   than least-squares or min/max approaches.

5. **Consistent Across Paths**: Both stratified LS and hybrid streaming
   use the same quantile-based estimation (v2.17.0+).

Why Fourier Mode is Still Useful
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduced Degrees of Freedom**: 10 Fourier coefficients cannot absorb
   arbitrary angle-dependent signals that would require 46 free parameters.

2. **Smooth Variation**: Fourier series produces physically reasonable smooth
   variation in contrast/offset across angles, matching experimental reality.

3. **Preserved Information**: The dominant modes (DC + first two harmonics)
   capture physically meaningful variations (e.g., sample asymmetry, beam
   position offsets).

4. **Breaks Degeneracy**: With only 10 coefficients for per-angle scaling,
   the optimizer must rely on physical parameters to explain angle-dependent
   shear dynamics.

Implementation
~~~~~~~~~~~~~~

The ``FourierReparameterizer`` class handles all conversions:

.. code-block:: python

   from homodyne.optimization.nlsq.fourier_reparam import (
       FourierReparameterizer,
       FourierReparamConfig,
   )

   # Configure Fourier reparameterization
   config = FourierReparamConfig(
       mode="auto",         # "independent", "fourier", or "auto"
       fourier_order=2,     # Number of Fourier harmonics
       auto_threshold=6,    # Use Fourier when n_phi > threshold
   )

   # Create reparameterizer for your phi angles
   phi_angles = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # radians
   reparameterizer = FourierReparameterizer(phi_angles, config)

   # Convert Fourier coefficients to per-angle values
   fourier_coeffs = np.array([0.5, 0.01, -0.02, 0.005, 0.003,  # contrast
                              1.0, 0.02, 0.01, -0.01, 0.005])  # offset
   contrast, offset = reparameterizer.fourier_to_per_angle(fourier_coeffs)

   # Convert back (useful for initialization)
   recovered_coeffs = reparameterizer.per_angle_to_fourier(contrast, offset)

Basis Matrix Computation
^^^^^^^^^^^^^^^^^^^^^^^^

The Fourier basis matrix ``B`` satisfies ``values = B @ coefficients``:

.. math::

   B = \begin{bmatrix}
   1 & \cos(\phi_1) & \sin(\phi_1) & \cos(2\phi_1) & \sin(2\phi_1) \\
   1 & \cos(\phi_2) & \sin(\phi_2) & \cos(2\phi_2) & \sin(2\phi_2) \\
   \vdots & \vdots & \vdots & \vdots & \vdots \\
   1 & \cos(\phi_n) & \sin(\phi_n) & \cos(2\phi_n) & \sin(2\phi_n)
   \end{bmatrix}

Covariance Transformation
^^^^^^^^^^^^^^^^^^^^^^^^^

After optimization, uncertainties must be transformed from Fourier space
to per-angle space. The Jacobian of the transformation is used:

.. math::

   \Sigma_{\text{per-angle}} = J \times \Sigma_{\text{Fourier}} \times J^T

where :math:`J = \frac{\partial (\text{per-angle})}{\partial (\text{Fourier})}`.

ParameterIndexMapper Reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``ParameterIndexMapper`` class provides consistent parameter index calculations
across all modes. This is the **single source of truth** for parameter group boundaries.

.. code-block:: python

   from homodyne.optimization.nlsq.parameter_index_mapper import ParameterIndexMapper

   # Constant mode (23 phi angles, 7 physical params)
   mapper = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=True)
   print(mapper.mode_name)           # "constant"
   print(mapper.n_per_angle_total)   # 2 (single contrast + offset, shared)
   print(mapper.total_params)        # 9 (2 + 7)
   print(mapper.get_group_indices()) # [(0, 1), (1, 2)]

   # Individual mode (23 phi angles, 7 physical params)
   mapper = ParameterIndexMapper(n_phi=23, n_physical=7, use_constant=False)
   print(mapper.mode_name)           # "individual"
   print(mapper.n_per_angle_total)   # 46 (23 contrasts + 23 offsets)
   print(mapper.total_params)        # 53 (46 + 7)
   print(mapper.get_group_indices()) # [(0, 23), (23, 46)]

   # Fourier mode (23 phi angles, order=2, 7 physical params)
   mapper = ParameterIndexMapper(n_phi=23, n_physical=7, fourier=fourier_obj)
   print(mapper.mode_name)           # "fourier"
   print(mapper.n_per_angle_total)   # 10 (5 contrast + 5 offset coeffs)
   print(mapper.total_params)        # 17 (10 + 7)
   print(mapper.get_group_indices()) # [(0, 5), (5, 10)]

.. list-table:: ParameterIndexMapper by Mode (n_phi=23, n_physical=7)
   :header-rows: 1
   :widths: 18 14 18 30 20

   * - Mode
     - n_per_group
     - n_per_angle_total
     - get_group_indices()
     - total_params
   * - ``constant``
     - 1
     - 2
     - [(0, 1), (1, 2)]
     - 9
   * - ``individual``
     - 23
     - 46
     - [(0, 23), (23, 46)]
     - 53
   * - ``fourier`` (K=2)
     - 5
     - 10
     - [(0, 5), (5, 10)]
     - 17

Layer 2: Hierarchical Optimization
----------------------------------

Algorithmic Approach
~~~~~~~~~~~~~~~~~~~~

Hierarchical optimization breaks the gradient cancellation by alternating
between two optimization stages:

**Stage 1: Physical Parameters Only**
   - Freeze per-angle parameters at current values
   - Optimize only physical parameters (D₀, α, γ̇₀, β, etc.)
   - Physical gradients are not diluted by per-angle updates

**Stage 2: Per-Angle Parameters Only**
   - Freeze physical parameters at current values
   - Optimize only per-angle scaling (contrast, offset)
   - Per-angle params adjust to match the fixed physics model

This alternation continues until convergence or maximum iterations.

.. code-block:: text

   HIERARCHICAL OPTIMIZATION ALGORITHM
   ====================================

   Initialize: params = [per_angle_params, physical_params]

   for outer_iter in range(max_outer_iterations):

       ┌──────────────────────────────────────────────────┐
       │ STAGE 1: Fit PHYSICAL params (per-angle frozen) │
       │                                                  │
       │ • Per-angle gradients = 0 (frozen)              │
       │ • Physical gradients get 100% of update         │
       │ • Uses L-BFGS with physical-only bounds         │
       └──────────────────────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────┐
       │ STAGE 2: Fit PER-ANGLE params (physical frozen) │
       │                                                  │
       │ • Physical gradients = 0 (frozen)               │
       │ • Per-angle params adjust to fixed physics      │
       │ • Uses L-BFGS with per-angle-only bounds        │
       └──────────────────────────────────────────────────┘
                              │
                              ▼
       ┌──────────────────────────────────────────────────┐
       │ CHECK CONVERGENCE                               │
       │                                                  │
       │ if ||physical_new - physical_old|| < tolerance: │
       │     break                                        │
       └──────────────────────────────────────────────────┘

   Return: optimized [per_angle_params, physical_params]

Why This Works
~~~~~~~~~~~~~~

1. **No Gradient Competition**: In Stage 1, physical parameters receive 100%
   of the gradient update without competition from per-angle parameters.

2. **Proper Attribution**: In Stage 2, per-angle parameters must adjust to
   match a fixed physics model, correctly attributing angle-dependent
   variations.

3. **Iterative Refinement**: Multiple alternations allow both parameter
   groups to converge to a consistent solution.

4. **Natural Regularization**: The alternating freeze prevents either
   parameter group from dominating the optimization.

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.hierarchical import (
       HierarchicalOptimizer,
       HierarchicalConfig,
       HierarchicalResult,
   )

   # Configure hierarchical optimization
   config = HierarchicalConfig(
       enable=True,
       max_outer_iterations=5,
       outer_tolerance=1e-6,
       physical_max_iterations=100,
       physical_ftol=1e-8,
       per_angle_max_iterations=50,
       per_angle_ftol=1e-6,
       log_stage_transitions=True,
   )

   # Create optimizer
   optimizer = HierarchicalOptimizer(
       config=config,
       n_phi=23,
       n_physical=7,
       fourier_reparameterizer=fourier,  # Optional, for Layer 1 integration
   )

   # Run optimization
   result = optimizer.fit(
       loss_fn=loss_function,
       grad_fn=gradient_function,
       p0=initial_params,
       bounds=(lower_bounds, upper_bounds),
   )

   # Access results
   print(f"Converged in {result.n_outer_iterations} outer iterations")
   print(f"Final loss: {result.fun:.6g}")

   # Examine optimization history
   for entry in result.history:
       print(f"Iter {entry['outer_iter']}: Stage1 loss={entry['stage1_loss']:.4g}, "
             f"Stage2 loss={entry['stage2_loss']:.4g}")

Layer 3: Adaptive CV Regularization
-----------------------------------

From Absolute to Relative Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous regularization used absolute variance:

.. math::

   L_{\text{reg,old}} = \lambda \times \text{Var}(\text{params}) \times N

The new approach uses coefficient of variation (CV) with relative scaling:

.. math::

   CV = \frac{\text{std}(\text{params})}{|\text{mean}(\text{params})|}

.. math::

   L_{\text{reg,new}} = \lambda \times CV^2 \times \text{MSE} \times N

Auto-Tuned Lambda
~~~~~~~~~~~~~~~~~

The regularization strength is automatically computed to achieve target
contribution to the loss:

.. math::

   \lambda = \frac{\text{target\_contribution}}{\text{target\_CV}^2}

**Example**:

- Target: Allow 10% variation (CV = 0.1)
- Target contribution: 10% of MSE
- Auto-computed: λ = 0.1 / 0.01 = 10

This is **100× stronger** than the previous λ = 0.01, providing effective
constraint on per-angle parameter variation.

Why CV-Based Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Scale-Invariant**: CV measures relative variation, not absolute.
   A 5% variation is penalized the same whether the mean is 0.5 or 5.0.

2. **Physically Meaningful**: CV = 0.1 means 10% standard deviation
   relative to mean—easy to interpret and tune.

3. **Properly Scaled**: By scaling with MSE, the regularization maintains
   consistent importance regardless of data noise level.

4. **Adaptive**: Auto-tuning ensures regularization contributes meaningfully
   to the optimization objective.

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.adaptive_regularization import (
       AdaptiveRegularizer,
       AdaptiveRegularizationConfig,
   )

   # Configure adaptive regularization
   config = AdaptiveRegularizationConfig(
       enable=True,
       mode="relative",            # "absolute", "relative", or "auto"
       lambda_base=1.0,            # Base lambda (used if auto_tune=False)
       target_cv=0.10,             # Target coefficient of variation
       target_contribution=0.10,   # Target fraction of MSE
       auto_tune_lambda=True,      # Compute lambda from targets
       max_cv=0.20,                # Maximum allowed CV
   )

   # Create regularizer
   regularizer = AdaptiveRegularizer(config, n_phi=23)

   # Compute regularization in loss function
   def regularized_loss(params, data):
       mse = compute_mse(params, data)
       n_points = len(data)
       reg_term = regularizer.compute_regularization(params, mse, n_points)
       return mse * n_points + reg_term

   # Check for constraint violations
   violations = regularizer.check_constraint_violation(params)
   if violations:
       print(f"Warning: CV exceeds max_cv threshold: {violations}")

Layer 4: Gradient Collapse Detection
------------------------------------

Runtime Monitoring
~~~~~~~~~~~~~~~~~~

Layer 4 monitors the optimization in real-time to detect gradient collapse—
when physical parameter gradients become negligible compared to per-angle
gradients.

**Detection Criterion**:

.. math::

   \text{ratio} = \frac{|\nabla_{\text{physical}}|}{|\nabla_{\text{per-angle}}|}

When ``ratio < threshold`` (default: 0.01) for ``N`` consecutive iterations
(default: 5), gradient collapse is declared.

Response Actions
~~~~~~~~~~~~~~~~

Upon detecting gradient collapse, the system can:

1. **warn**: Log warning, continue optimization
2. **hierarchical**: Switch to hierarchical optimization (Layer 2)
3. **reset**: Reset per-angle parameters to their mean values
4. **abort**: Terminate optimization with warning

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.gradient_monitor import (
       GradientCollapseMonitor,
       GradientMonitorConfig,
   )

   # Configure gradient monitoring
   config = GradientMonitorConfig(
       enable=True,
       ratio_threshold=0.01,       # Collapse threshold
       consecutive_triggers=5,     # Must trigger N times
       response_mode="hierarchical",  # Action on collapse
       reset_per_angle_to_mean=True,
       lambda_multiplier_on_collapse=10.0,
   )

   # Create monitor
   monitor = GradientCollapseMonitor(
       config=config,
       physical_indices=[10, 11, 12, 13, 14, 15, 16],  # Physical param indices
       per_angle_indices=list(range(10)),  # Per-angle param indices
   )

   # Check gradients during optimization
   for iteration in range(max_iterations):
       gradients = compute_gradients(params)

       status = monitor.check(gradients, iteration, params)

       if status == "COLLAPSE_DETECTED":
           response = monitor.get_response()
           if response["mode"] == "hierarchical":
               # Switch to hierarchical optimization
               result = run_hierarchical_optimization(params)
               break

       # Normal optimization step
       params = update_params(params, gradients)

   # Get final diagnostics
   diagnostics = monitor.get_diagnostics()
   print(f"Gradient monitoring: {diagnostics}")

Layer 5: Shear-Sensitivity Weighting
------------------------------------

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~

Layer 5 addresses gradient cancellation by weighting residuals according to their
sensitivity to the shear parameter ``gamma_dot_t0``. The shear term in the XPCS model
produces gradients that scale with :math:`|\cos(\phi_0 - \phi)|`:

- Angles where :math:`\phi \approx \phi_0` or :math:`\phi \approx \phi_0 + \pi` are
  **highly sensitive** to shear (parallel/anti-parallel flow)
- Angles where :math:`\phi \approx \phi_0 \pm \pi/2` are **insensitive** to shear
  (perpendicular flow)

**Weight Formula**:

.. math::

   w(\phi) = w_{\min} + (1 - w_{\min}) \times |\cos(\phi_0 - \phi)|^\alpha

where:

- :math:`w_{\min}` is the minimum weight (default: 0.3) ensuring perpendicular angles
  still contribute
- :math:`\alpha` is the sensitivity exponent (default: 1.0 for linear weighting)

The weights are normalized so their mean equals 1.0, preserving the overall loss scale.

Why This Works
~~~~~~~~~~~~~~

Standard unweighted least squares gives equal importance to all angles. When angles
span 360°, the positive and negative contributions to :math:`\partial L / \partial \dot{\gamma}_0`
partially cancel (94.6% cancellation for 23 uniformly-spaced angles).

Shear-sensitivity weighting addresses this by:

1. **Amplifying informative residuals**: Errors at shear-sensitive angles (parallel flow)
   contribute more to the loss gradient
2. **Attenuating uninformative residuals**: Errors at insensitive angles (perpendicular flow)
   contribute less
3. **Breaking symmetry**: The asymmetric weighting prevents gradient cancellation

.. code-block:: text

   Weight distribution for 8 angles with φ₀ = 0°:

   φ = 0°:   |cos(0°)|   = 1.0  → w = 1.0   (HIGH weight)
   φ = 45°:  |cos(45°)|  = 0.71 → w = 0.80
   φ = 90°:  |cos(90°)|  = 0.0  → w = 0.30  (LOW weight)
   φ = 135°: |cos(135°)| = 0.71 → w = 0.80
   φ = 180°: |cos(180°)| = 1.0  → w = 1.0   (HIGH weight)
   ...

   Net effect: Gradient from parallel angles dominates,
               preventing cancellation

Data Flow
~~~~~~~~~

Shear-sensitivity weighting is computed in Homodyne and passed to NLSQ as generic
residual weights, maintaining separation of concerns:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Homodyne wrapper.py                                                     │
   │  ┌──────────────────────┐    ┌─────────────────────────────────────┐   │
   │  │ ShearSensitivity     │───▶│ get_weights() → numpy array (n_phi) │   │
   │  │ Weighting            │    └─────────────────────────────────────┘   │
   │  └──────────────────────┘                      │                        │
   │                                                ▼                        │
   │                                    .tolist() → Python list              │
   │                                                │                        │
   │  ┌─────────────────────────────────────────────▼────────────────────┐  │
   │  │ HybridStreamingConfig(                                           │  │
   │  │   enable_residual_weighting=True,                                │  │
   │  │   residual_weights=[w0, w1, ..., w_{n_phi-1}]                    │  │
   │  │ )                                                                │  │
   │  └──────────────────────────────────────────────────────────────────┘  │
   └─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ NLSQ adaptive_hybrid.py                                                 │
   │  ┌─────────────────────┐                                               │
   │  │ _setup_residual_    │ Converts list → jnp.array(float64)            │
   │  │ weights()           │                                               │
   │  └─────────────────────┘                                               │
   │           │                                                             │
   │           ▼                                                             │
   │  ┌─────────────────────────────────────────────────────────────────┐   │
   │  │ loss_fn(params, x_batch, y_batch):                              │   │
   │  │   group_idx = x_batch[:, 0].astype(int32)  # φ index per point  │   │
   │  │   weights = residual_weights[group_idx]    # lookup per-point   │   │
   │  │   wmse = sum(w * r²) / sum(w)              # weighted MSE       │   │
   │  └─────────────────────────────────────────────────────────────────┘   │
   └─────────────────────────────────────────────────────────────────────────┘

Dynamic Weight Updates
~~~~~~~~~~~~~~~~~~~~~~

As the optimizer converges, :math:`\phi_0` may change from its initial guess. The
``ShearSensitivityWeighter`` class supports dynamic weight updates:

.. code-block:: python

   from homodyne.optimization.nlsq.shear_weighting import (
       ShearSensitivityWeighter,
       ShearWeightingConfig,
   )

   # Create weighter with initial phi0 estimate
   config = ShearWeightingConfig(
       enable=True,
       min_weight=0.3,
       alpha=1.0,
       update_frequency=1,  # Update every outer iteration
       normalize=True,
   )
   weighter = ShearSensitivityWeighter(phi_angles, config, phi0_initial=0.0)

   # Get initial weights
   weights = weighter.get_weights()
   print(f"Initial weights: {weights}")

   # During optimization, update phi0 as it converges
   for outer_iter in range(max_iterations):
       # ... optimization step ...

       if outer_iter % config.update_frequency == 0:
           weighter.update_phi0(current_phi0_estimate)
           weights = weighter.get_weights()  # Weights recomputed

   # Get diagnostics
   diagnostics = weighter.get_diagnostics()
   print(f"Weight range: [{diagnostics['weight_min']:.3f}, {diagnostics['weight_max']:.3f}]")

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.shear_weighting import (
       ShearSensitivityWeighter,
       ShearWeightingConfig,
       create_shear_weighting,
   )

   # High-level factory function (recommended)
   phi_angles = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])  # radians
   weighter = create_shear_weighting(
       phi_angles=phi_angles,
       phi0=0.0,
       config={
           'enable': True,
           'min_weight': 0.3,
           'alpha': 1.0,
       }
   )

   if weighter is not None:
       weights = weighter.get_weights()
       print(f"n_weights: {len(weights)}")
       print(f"Weight range: [{min(weights):.3f}, {max(weights):.3f}]")

Configuration Reference
-----------------------

YAML Configuration
~~~~~~~~~~~~~~~~~~

Complete YAML configuration for the Anti-Degeneracy Defense System:

.. code-block:: yaml

   optimization:
     nlsq:
       # ... existing settings ...

       # Anti-Degeneracy Defense System (v2.17.0+)
       anti_degeneracy:

         # Layer 1: Per-Angle Mode Selection (v2.17.0 defaults)
         per_angle_mode: "constant"        # DEFAULT: fixed per-angle from quantiles
         # per_angle_mode: "auto"          # Auto-selects fourier/individual
         # per_angle_mode: "fourier"       # Fourier reparameterization
         # per_angle_mode: "individual"    # Independent per-angle params
         fourier_order: 2                  # Number of Fourier harmonics (if fourier mode)
         fourier_auto_threshold: 6         # Use Fourier when n_phi > threshold (auto mode)

         # Layer 2: Hierarchical Optimization
         hierarchical:
           enable: true
           max_outer_iterations: 5
           outer_tolerance: 1.0e-6
           physical_max_iterations: 100
           per_angle_max_iterations: 50

         # Layer 3: Adaptive Relative Regularization
         regularization:
           mode: "relative"            # "absolute", "relative", "auto"
           lambda: 1.0                 # Base lambda (100x higher than v2.8)
           target_cv: 0.10             # 10% variation target
           target_contribution: 0.10   # 10% of MSE contribution
           max_cv: 0.20                # Hard limit: 20% max variation
           auto_tune_lambda: true      # Compute lambda automatically

         # Layer 4: Gradient Collapse Detection
         gradient_monitoring:
           enable: true
           ratio_threshold: 0.01       # |∇_physical|/|∇_per_angle| threshold
           consecutive_triggers: 5     # Must trigger N times
           response: "hierarchical"    # "warn", "hierarchical", "reset", "abort"

         # Layer 5: Shear-Sensitivity Weighting (v2.9.1+)
         shear_weighting:
           enable: true              # Enable angle-dependent loss weighting
           min_weight: 0.3           # Minimum weight for perpendicular angles
           alpha: 1.0                # Shear sensitivity exponent (1 = linear)
           update_frequency: 1       # Update weights every N outer iterations
           normalize: true           # Normalize weights so mean = 1.0

Configuration Options Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Layer 1 Configuration (v2.17.0)
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``per_angle_mode``
     - "constant"
     - **DEFAULT CHANGED v2.17.0**: "constant", "fourier", "individual", or "auto"
   * - ``fourier_order``
     - 2
     - Number of Fourier harmonics (K) for fourier mode
   * - ``fourier_auto_threshold``
     - 6
     - Use Fourier when n_phi > threshold (auto mode only)

.. list-table:: Layer 2 Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``hierarchical.enable``
     - true
     - Enable hierarchical optimization
   * - ``hierarchical.max_outer_iterations``
     - 5
     - Maximum alternating iterations
   * - ``hierarchical.outer_tolerance``
     - 1e-6
     - Convergence tolerance for physical params

.. list-table:: Layer 3 Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``regularization.mode``
     - "relative"
     - Regularization type: "absolute", "relative", "auto"
   * - ``regularization.lambda``
     - 1.0
     - Base regularization strength (100× v2.8)
   * - ``regularization.target_cv``
     - 0.10
     - Target coefficient of variation (10%)
   * - ``regularization.target_contribution``
     - 0.10
     - Target contribution to loss (10%)
   * - ``regularization.max_cv``
     - 0.20
     - Maximum allowed CV before violation

.. list-table:: Layer 4 Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``gradient_monitoring.enable``
     - true
     - Enable gradient collapse detection
   * - ``gradient_monitoring.ratio_threshold``
     - 0.01
     - Collapse detection threshold
   * - ``gradient_monitoring.consecutive_triggers``
     - 5
     - Required consecutive triggers
   * - ``gradient_monitoring.response``
     - "hierarchical"
     - Response action on collapse

.. list-table:: Layer 5 Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``shear_weighting.enable``
     - true
     - Enable shear-sensitivity weighting
   * - ``shear_weighting.min_weight``
     - 0.3
     - Minimum weight for perpendicular angles
   * - ``shear_weighting.alpha``
     - 1.0
     - Sensitivity exponent (1.0 = linear)
   * - ``shear_weighting.update_frequency``
     - 1
     - Update weights every N outer iterations
   * - ``shear_weighting.normalize``
     - true
     - Normalize weights so mean = 1.0

Usage Tutorial
--------------

Basic Usage
~~~~~~~~~~~

For most users, the default configuration provides robust fitting:

.. code-block:: python

   from homodyne.optimization import fit_nlsq_jax

   # Load your data
   data = load_xpcs_data("experiment.hdf5")

   # Fit with default anti-degeneracy settings
   result = fit_nlsq_jax(
       t1=data.t1,
       t2=data.t2,
       c2=data.c2,
       phi_rad=data.phi_angles,
       mode="laminar_flow",
       # Anti-degeneracy defense is enabled by default for n_phi > 6
   )

   # gamma_dot_t0 should now be correctly estimated
   print(f"gamma_dot_t0 = {result.params['gamma_dot_t0']:.4e}")

Advanced Usage: Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For challenging datasets, you may need to adjust settings:

.. code-block:: python

   from homodyne.config import ConfigManager

   # Load configuration from YAML
   config = ConfigManager.from_yaml("my_config.yaml")

   # Or modify programmatically
   config.optimization.nlsq.anti_degeneracy.hierarchical.max_outer_iterations = 10
   config.optimization.nlsq.anti_degeneracy.regularization.target_cv = 0.05  # Tighter

   # Run fit with custom config
   result = fit_nlsq_jax(
       data,
       config=config,
       mode="laminar_flow",
   )

Diagnosing Problems
~~~~~~~~~~~~~~~~~~~

If fitting still produces collapsed parameters:

**Step 1: Check diagnostics**

.. code-block:: python

   # Access optimization diagnostics
   diagnostics = result.diagnostics

   # Check if gradient collapse was detected
   if diagnostics.get("gradient_monitor", {}).get("collapse_detected"):
       print("Gradient collapse was detected!")
       print(f"Response taken: {diagnostics['gradient_monitor']['response_actions']}")

   # Check per-angle parameter CV
   contrast_cv = np.std(result.params['contrast']) / np.mean(result.params['contrast'])
   if contrast_cv > 0.20:
       print(f"Warning: High contrast variation (CV={contrast_cv:.2%})")

**Step 2: Increase regularization**

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         regularization:
           lambda: 10.0          # Increase from 1.0
           target_cv: 0.05       # Tighter constraint

**Step 3: Force Fourier mode**

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "fourier"  # Force Fourier, even for small n_phi
         fourier_order: 1           # Reduce degrees of freedom further

**Step 4: Increase hierarchical iterations**

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         hierarchical:
           max_outer_iterations: 20  # More alternations
           outer_tolerance: 1.0e-8   # Tighter convergence

When to Disable
~~~~~~~~~~~~~~~

The anti-degeneracy system can be disabled for:

- **Small n_phi (≤ 3)**: Structural degeneracy is less problematic
- **Static mode**: No shear parameters to collapse
- **Quick exploratory fits**: When speed matters more than accuracy

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "independent"
         hierarchical:
           enable: false
         regularization:
           enable: false
         gradient_monitoring:
           enable: false

Migration Guide
---------------

From v2.16.x to v2.17.0
~~~~~~~~~~~~~~~~~~~~~~~

**Default Behavior Changes**:

1. ``per_angle_mode`` default: "auto" → "constant" (quantile-based fixed scaling)
2. "auto" mode logic changed: now only selects between "fourier" and "individual"
3. Quantile-based estimation used in both stratified LS and hybrid streaming paths

**What This Means for Your Fits**:

- Per-angle contrast/offset are now computed from data quantiles and **not optimized**
- Only 7 physical parameters are optimized (for laminar_flow)
- Most fits should converge faster with more robust shear parameter estimation

**To Preserve v2.16.x Behavior** (if needed):

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"       # v2.16.x: auto selected constant >= threshold
         # OR
         per_angle_mode: "individual" # Full per-angle optimization

**To Switch to Fourier Mode** (smooth angular variation):

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "fourier"
         fourier_order: 2

From v2.8.x to v2.9.0
~~~~~~~~~~~~~~~~~~~~~

**Default Behavior Changes**:

1. ``group_variance_lambda`` default: 0.01 → 1.0 (100× stronger)
2. ``per_angle_mode`` default: "independent" → "auto"
3. ``hierarchical.enable`` default: false → true
4. New gradient monitoring (enabled by default)

**To Preserve Old Behavior**:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "independent"
         hierarchical:
           enable: false
         regularization:
           lambda: 0.01            # Old default
         gradient_monitoring:
           enable: false

**Recommended Migration Path**:

1. Start with v2.17.0 defaults (most robust)
2. If per-angle variation is physically important, try ``per_angle_mode: "fourier"``
3. Only use ``per_angle_mode: "individual"`` for n_phi ≤ 3 or special cases
4. Monitor diagnostics for convergence and parameter recovery

API Reference
-------------

Parameter Utilities (v2.17.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.parameter_utils
   :members: compute_quantile_per_angle_scaling, compute_consistent_per_angle_init
   :undoc-members:
   :show-inheritance:

Fourier Reparameterization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.fourier_reparam
   :members:
   :undoc-members:
   :show-inheritance:

Hierarchical Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.hierarchical
   :members:
   :undoc-members:
   :show-inheritance:

Adaptive Regularization
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.adaptive_regularization
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Monitoring
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.gradient_monitor
   :members:
   :undoc-members:
   :show-inheritance:

Shear-Sensitivity Weighting
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.shear_weighting
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Index Mapper
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.parameter_index_mapper
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

- :doc:`computational_methods` - General optimization algorithms
- :doc:`theoretical_framework` - XPCS physics foundations
- :doc:`/api-reference/optimization` - Complete optimization API
- :doc:`/configuration/options` - Full configuration reference
