Anti-Degeneracy Defense System (v2.9.0)
========================================

This section provides comprehensive documentation for the Anti-Degeneracy Defense System
introduced in Homodyne v2.9.0. The system addresses fundamental optimization challenges
in laminar flow XPCS analysis where structural degeneracy and gradient cancellation
prevent accurate parameter estimation.

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

Parameter Count Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~

The ``per_angle_mode`` setting controls how per-angle parameters are handled:

.. list-table:: per_angle_mode Options
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``individual``
     - Each angle has independent contrast and offset (2 × n_phi params)
   * - ``fourier``
     - Contrast/offset expressed as Fourier series (2 × (2K+1) params)
   * - ``constant``
     - All angles share one contrast and one offset (2 params)
   * - ``auto``
     - Auto-selects based on n_phi and thresholds

.. list-table:: Parameter Count Comparison
   :header-rows: 1
   :widths: 12 18 18 18 18 16

   * - n_phi
     - Individual
     - Fourier (K=2)
     - Constant
     - Reduction
     - Auto Mode
   * - 2
     - 4
     - 4*
     - 2
     - 50%
     - Individual
   * - 3
     - 6
     - 6*
     - 2
     - 67%
     - Constant†
   * - 6
     - 12
     - 10
     - 2
     - 83%
     - Constant†
   * - 10
     - 20
     - 10
     - 2
     - 90%
     - Constant†
   * - 23
     - 46
     - 10
     - 2
     - **96%**
     - Constant†
   * - 100
     - 200
     - 10
     - 2
     - **99%**
     - Constant†

| \* For n_phi ≤ 2×(order+1), Fourier mode provides no reduction
| † With ``constant_scaling_threshold: 3`` (default), constant mode is auto-selected when n_phi ≥ 3

Constant Mode (v2.14.0+)
~~~~~~~~~~~~~~~~~~~~~~~~

**Constant mode** provides the most aggressive parameter reduction by assuming all
angles share identical contrast and offset values:

.. math::

   \text{contrast}(\phi) = c_{\text{const}} \quad \forall \phi

   \text{offset}(\phi) = o_{\text{const}} \quad \forall \phi

**Parameter Transformation**:

.. code-block:: text

   CONSTANT MODE TRANSFORMATION
   ============================

   Input:  per-angle params [c₀, c₁, ..., c₂₂, o₀, o₁, ..., o₂₂]
           (46 values for 23 angles)
                      │
                      ▼
   Contract: [mean(contrast), mean(offset)]
             (2 values)
                      │
                      ▼
   Optimize: [contrast, offset, D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
             (9 params total for laminar_flow)
                      │
                      ▼
   Expand:   [c, c, ..., c, o, o, ..., o] + physical
             (53 values for backward compatibility)

**When to Use Constant Mode**:

- Detector response is uniform across angles
- Per-angle variation is primarily noise, not physics
- Multi-start optimization with many angles (parameter count reduction critical)
- Quick exploratory fits where per-angle detail is not important

**When NOT to Use Constant Mode**:

- Significant physical per-angle variation exists (sample asymmetry)
- High-precision per-angle contrast/offset needed for downstream analysis
- Fourier mode's smooth angular variation better matches experimental reality

**Configuration**:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         enable: true
         per_angle_mode: "auto"          # or "constant" to force
         constant_scaling_threshold: 3   # Use constant when n_phi >= 3

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
   * - **constant**
     - **9**
     - **9**
     - **10-20 (tractable)**

Why This Works
~~~~~~~~~~~~~~

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

       # Anti-Degeneracy Defense System (v2.9.0+)
       anti_degeneracy:

         # Layer 1: Per-Angle Mode Selection
         per_angle_mode: "auto"            # "individual", "fourier", "constant", "auto"
         constant_scaling_threshold: 3     # Use constant when n_phi >= threshold
         fourier_order: 2                  # Number of Fourier harmonics (if fourier mode)
         fourier_auto_threshold: 6         # Use Fourier when n_phi > threshold (if not constant)

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

.. list-table:: Layer 1 Configuration
   :header-rows: 1
   :widths: 25 15 60

   * - Option
     - Default
     - Description
   * - ``per_angle_mode``
     - "auto"
     - Mode selection: "individual", "fourier", "constant", or "auto"
   * - ``constant_scaling_threshold``
     - 3
     - Use constant mode when n_phi >= threshold (auto mode only)
   * - ``fourier_order``
     - 2
     - Number of Fourier harmonics (K)
   * - ``fourier_auto_threshold``
     - 6
     - Use Fourier when n_phi > threshold (if constant not selected)

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

1. Start with v2.9.0 defaults (most robust)
2. If fitting is slow, try ``per_angle_mode: "independent"`` with
   ``hierarchical: true``
3. Monitor diagnostics for gradient collapse warnings
4. Adjust regularization strength if per-angle CV exceeds 20%

API Reference
-------------

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

See Also
--------

- :doc:`computational_methods` - General optimization algorithms
- :doc:`theoretical_framework` - XPCS physics foundations
- :doc:`/api-reference/optimization` - Complete optimization API
- :doc:`/configuration/options` - Full configuration reference
