# NLSQ Divergence Root Cause Analysis

**Date:** 2026-01-02 **Issue:** gamma_dot_t0 parameter collapse in 23-phi adaptive
hybrid streaming optimization **Resolution:** Auto-enable hierarchical optimization when
shear weighting is configured

## Problem Summary

When running NLSQ optimization with:

- 23 phi angles (spanning 360 degrees)
- Adaptive hybrid streaming mode (for >10M data points)
- `hierarchical.enable: false` in config
- `shear_weighting.enable: true` in config

The shear rate parameter `gamma_dot_t0` collapsed to its lower bound (1e-6) with zero
uncertainty, producing a poor fit (chi-squared 10x worse than baseline).

## Comparison

| Metric | 3-phi Baseline (Working) | 23-phi Streaming (Failing) |
|--------|-------------------------|---------------------------| | Method | Stratified
LS (full Jacobian) | Adaptive Hybrid Streaming | | Data Points | 3,000,000 | 23,000,000
| | Angles | 3 | 23 | | chi_squared | 1.0064e+08 | 1.0201e+09 | | reduced_chi² | 33.55 |
44.35 | | D0 | 19,254 | 92,021 | | **gamma_dot_t0** | **0.00194 ± 1.6e-5** | **1e-6 ±
0** |

## Root Cause

### 1. Gradient Cancellation

The shear term gradient suffers from severe cancellation when summing over many angles:

```
∂L/∂γ̇₀ ∝ Σ_angles cos(φ₀ - φ_i)
```

For 23 angles spanning 360°:

- 11 angles with positive cos contribution
- 12 angles with negative cos contribution
- Sum of cos ≈ -0.89
- Sum of |cos| ≈ 16.5
- **Cancellation factor: |sum|/Σ|cos| ≈ 0.054 (94.6% cancellation!)**

This makes the effective gradient for gamma_dot_t0 nearly zero, causing the optimizer to
treat it as irrelevant.

### 2. Shear Weighting Not Applied

The configuration had:

```yaml
anti_degeneracy:
  hierarchical:
    enable: false    # Hierarchical disabled
  shear_weighting:
    enable: true     # Shear weighting enabled
```

**Critical Bug:** Shear weighting is ONLY applied inside the hierarchical optimizer's
loss function:

```python
# wrapper.py lines 5489-5500 (inside hierarchical path)
if shear_weighter_local is not None:
    weighted_loss = shear_weighter_local.apply_weights_to_loss(
        residuals, phi_indices_jax
    )
```

When `hierarchical.enable: false`, the code uses the standard hybrid streaming path
(line 5606), which calls `optimizer.fit()` directly WITHOUT applying shear weights.

**Result:** Shear weighting was configured but never applied, allowing gradient
cancellation to collapse gamma_dot_t0.

### 3. Constant Scaling Mode Insufficient

The config used `per_angle_mode: "auto"` which selected "constant" scaling (reducing 46
per-angle parameters to just 2). While this eliminates structural degeneracy (per-angle
params can't absorb shear signal), it does NOT fix gradient cancellation.

Gradient cancellation is a property of the physical model computation, not the per-angle
scaling.

## Fix Applied

**File:** `homodyne/optimization/nlsq/wrapper.py` (lines 4703-4739)

Auto-enable hierarchical optimization when shear weighting is configured for
laminar_flow mode:

```python
# Check if shear weighting will be enabled
shear_weighting_will_be_enabled = (
    shear_weighting_config_early.get("enable", True)
    and is_laminar_flow
    and n_phi > 3
)

# Override: shear weighting requires hierarchical optimization
if shear_weighting_will_be_enabled and not enable_hierarchical:
    logger.warning("Auto-enabling hierarchical optimization to apply shear weights.")
    enable_hierarchical = True
```

This ensures that when shear_weighting is enabled, hierarchical optimization is also
enabled, guaranteeing the shear weights are applied during L-BFGS optimization.

## How Shear Weighting Prevents Cancellation

Shear weighting applies angle-dependent weights to residuals:

```python
# Weight by shear sensitivity: |cos(φ₀ - φ)|^α
weights[i] = min_weight + (1 - min_weight) * |cos(φ₀ - φ_i)|^α
```

This emphasizes angles parallel/antiparallel to the flow direction (where shear signal
is strongest) and de-emphasizes perpendicular angles. When the weighted loss is
differentiated:

```
∂L_weighted/∂γ̇₀ = Σ_angles w_i * |cos(φ₀ - φ_i)| * (residual term)
```

The `w_i * |cos(φ₀ - φ_i)|` factor is always positive, preventing sign-based
cancellation.

## Lessons Learned

1. **Config interdependencies must be enforced in code.** Shear weighting requires
   hierarchical optimization; this dependency should be automatic, not rely on user
   configuration.

1. **Defense layers must be fail-safe.** If a defense mechanism is enabled but a
   prerequisite is disabled, the system should warn and auto-enable the prerequisite.

1. **Constant scaling mode addresses structural degeneracy but not gradient
   cancellation.** These are different problems requiring different solutions.

1. **Test with production-scale data.** The 3-phi test passed because with fewer angles,
   gradient cancellation is less severe. The 23-phi case exposed the bug.

## Recommended Configuration

For laminar_flow mode with many angles:

```yaml
anti_degeneracy:
  enable: true
  per_angle_mode: "auto"
  hierarchical:
    enable: true          # REQUIRED for shear weighting to work
    max_outer_iterations: 5
  shear_weighting:
    enable: true          # Prevents gradient cancellation
    alpha: 1.0
    min_weight: 0.3
```

## Files Modified

- `homodyne/optimization/nlsq/wrapper.py`: Added auto-enable logic for hierarchical when
  shear weighting is configured

## Related Documentation

- `docs/specs/anti-degeneracy-defense-v2.9.0.md`
- `CLAUDE.md` - NLSQ Gradient Cancellation Fix (v2.10.0+)
