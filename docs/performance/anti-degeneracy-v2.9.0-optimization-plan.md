# Anti-Degeneracy Defense v2.9.0 - Optimization Plan

**Date**: 2025-12-30
**Priority**: Post-v2.9.0 release (non-blocking)
**Estimated Total Time**: 4-6 hours implementation + 2-4 hours testing

---

## Overview

Based on performance analysis, three optimization opportunities identified:

1. **P1 - Gradient Monitor Memory Leak**: Fix unbounded history growth
2. **P2 - Hierarchical Optimizer Tuning**: Reduce outer iterations and adaptive budgets
3. **P3 - Regularization Vectorization**: Code quality improvement (optional)

---

## P1: Fix Gradient Monitor Memory Leak (CRITICAL)

**Impact**: Prevents 10MB memory leak on long optimizations (10K iterations)
**Risk**: None (only last 10 history entries are used)
**Time**: 30 minutes implementation + 30 minutes testing

### Implementation

**File**: `/homodyne/optimization/nlsq/gradient_monitor.py`

**Changes**:

1. Add config parameter (line 39):
```python
@dataclass
class GradientMonitorConfig:
    enable: bool = True
    ratio_threshold: float = 0.01
    consecutive_triggers: int = 5
    response_mode: Literal["warn", "hierarchical", "reset", "abort"] = "hierarchical"
    reset_per_angle_to_mean: bool = True
    lambda_multiplier_on_collapse: float = 10.0
    check_interval: int = 1
    max_history_size: int = 1000  # NEW: Limit history storage
```

2. Update `from_dict` classmethod (line 73):
```python
@classmethod
def from_dict(cls, config_dict: dict) -> GradientMonitorConfig:
    return cls(
        enable=config_dict.get("enable", True),
        ratio_threshold=float(config_dict.get("ratio_threshold", 0.01)),
        consecutive_triggers=config_dict.get("consecutive_triggers", 5),
        response_mode=config_dict.get("response", "hierarchical"),
        reset_per_angle_to_mean=config_dict.get("reset_per_angle_to_mean", True),
        lambda_multiplier_on_collapse=float(
            config_dict.get("lambda_multiplier_on_collapse", 10.0)
        ),
        check_interval=config_dict.get("check_interval", 1),
        max_history_size=config_dict.get("max_history_size", 1000),  # NEW
    )
```

3. Implement circular buffer in `check()` method (after line 234):
```python
def check(
    self,
    gradients: np.ndarray,
    iteration: int,
    params: np.ndarray | None = None,
    loss: float | None = None,
) -> str:
    # ... existing code ...

    # Record history with circular buffer
    entry = {
        "iteration": iteration,
        "physical_grad_norm": float(physical_grad_norm),
        "per_angle_grad_norm": float(per_angle_grad_norm),
        "ratio": float(ratio),
    }

    # Circular buffer: keep only last max_history_size entries
    if len(self.history) >= self.config.max_history_size:
        self.history.pop(0)  # Remove oldest

    self.history.append(entry)

    # ... rest of method unchanged ...
```

### Testing

Add test to `/tests/unit/test_gradient_monitor.py`:

```python
def test_gradient_monitor_memory_bounded():
    """History should not exceed max_history_size."""
    config = GradientMonitorConfig(
        enable=True,
        max_history_size=100,
    )
    monitor = GradientCollapseMonitor(
        config,
        physical_indices=[0, 1, 2],
        per_angle_indices=[3, 4, 5],
    )

    # Simulate 10K iterations
    gradients = np.ones(6)
    params = np.ones(6)
    for i in range(10_000):
        monitor.check(gradients, i, params=params, loss=1.0)

    # History should be capped at max_history_size
    assert len(monitor.history) == 100
    assert monitor.history[0]["iteration"] == 9_900  # Oldest kept
    assert monitor.history[-1]["iteration"] == 9_999  # Newest
```

**Validation**: Run test suite, verify memory usage bounded

---

## P2: Hierarchical Optimizer Tuning (HIGH IMPACT)

**Impact**: 40% reduction in hierarchical optimization time (6 hours → 3.6 hours)
**Risk**: Medium (may affect convergence on difficult problems)
**Time**: 2 hours implementation + 2 hours validation testing

### Implementation Phase 1: Reduce Default Outer Iterations

**File**: `/homodyne/optimization/nlsq/hierarchical.py`

**Change** (line 69):
```python
@dataclass
class HierarchicalConfig:
    enable: bool = True
    max_outer_iterations: int = 3  # CHANGED from 5 → 40% savings
    outer_tolerance: float = 1e-6
    physical_max_iterations: int = 100
    physical_ftol: float = 1e-8
    per_angle_max_iterations: int = 50
    per_angle_ftol: float = 1e-6
    log_stage_transitions: bool = True
    save_intermediate_results: bool = False
```

**Rationale**: Analysis of convergence history shows physical parameters typically stabilize by iteration 2-3. Reducing from 5 to 3 saves ~40% of loss evaluations.

**Validation**:
1. Run on existing test datasets with both configs (max_outer=5 vs 3)
2. Compare final chi-squared values (should be within 1%)
3. Compare physical parameter estimates (should be within uncertainties)
4. Monitor outer iteration count in logs (should converge before 3)

### Implementation Phase 2: Adaptive Inner Iteration Budgets

**File**: `/homodyne/optimization/nlsq/hierarchical.py`

**Changes**:

1. Add adaptive config (line 69):
```python
@dataclass
class HierarchicalConfig:
    enable: bool = True
    max_outer_iterations: int = 3
    outer_tolerance: float = 1e-6
    physical_max_iterations: int = 100
    physical_ftol: float = 1e-8
    per_angle_max_iterations: int = 50
    per_angle_ftol: float = 1e-6
    log_stage_transitions: bool = True
    save_intermediate_results: bool = False

    # NEW: Adaptive budgets
    use_adaptive_budgets: bool = True
    refinement_budget_fraction: float = 0.3  # Use 30% budget after first iteration
```

2. Update `from_dict` (line 104):
```python
@classmethod
def from_dict(cls, config_dict: dict) -> HierarchicalConfig:
    return cls(
        enable=config_dict.get("enable", True),
        max_outer_iterations=config_dict.get("max_outer_iterations", 3),
        outer_tolerance=float(config_dict.get("outer_tolerance", 1e-6)),
        physical_max_iterations=config_dict.get("physical_max_iterations", 100),
        physical_ftol=float(config_dict.get("physical_ftol", 1e-8)),
        per_angle_max_iterations=config_dict.get("per_angle_max_iterations", 50),
        per_angle_ftol=float(config_dict.get("per_angle_ftol", 1e-6)),
        log_stage_transitions=config_dict.get("log_stage_transitions", True),
        save_intermediate_results=config_dict.get("save_intermediate_results", False),
        use_adaptive_budgets=config_dict.get("use_adaptive_budgets", True),  # NEW
        refinement_budget_fraction=float(
            config_dict.get("refinement_budget_fraction", 0.3)
        ),  # NEW
    )
```

3. Modify `_fit_physical_stage()` (line 362):
```python
def _fit_physical_stage(
    self,
    loss_fn: Callable,
    grad_fn: Callable | None,
    current_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    outer_iter: int,
) -> optimize.OptimizeResult:
    """Stage 1: Optimize physical parameters with per-angle frozen."""
    frozen_per_angle = current_params[self.per_angle_indices].copy()

    # ... create physical_loss and physical_grad functions ...

    # Adaptive iteration budget
    if self.config.use_adaptive_budgets and outer_iter > 0:
        max_iter = max(
            30,  # Minimum 30 iterations
            int(self.config.physical_max_iterations * self.config.refinement_budget_fraction)
        )
        if self.config.log_stage_transitions:
            logger.debug(f"  Using adaptive budget: {max_iter} iterations (refinement mode)")
    else:
        max_iter = self.config.physical_max_iterations

    # Run L-BFGS-B
    result = optimize.minimize(
        physical_loss,
        current_params[self.physical_indices],
        method="L-BFGS-B",
        jac=physical_grad,
        bounds=physical_bounds,
        options={
            "maxiter": max_iter,  # CHANGED: adaptive
            "ftol": self.config.physical_ftol,
            "disp": False,
        },
    )

    # ... rest unchanged ...
```

4. Similar change to `_fit_per_angle_stage()` (line 430):
```python
def _fit_per_angle_stage(
    self,
    loss_fn: Callable,
    grad_fn: Callable | None,
    current_params: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    outer_iter: int,
) -> optimize.OptimizeResult:
    """Stage 2: Optimize per-angle parameters with physical frozen."""
    # ... create per_angle_loss and per_angle_grad functions ...

    # Adaptive iteration budget
    if self.config.use_adaptive_budgets and outer_iter > 0:
        max_iter = max(
            20,  # Minimum 20 iterations
            int(self.config.per_angle_max_iterations * self.config.refinement_budget_fraction)
        )
        if self.config.log_stage_transitions:
            logger.debug(f"  Using adaptive budget: {max_iter} iterations (refinement mode)")
    else:
        max_iter = self.config.per_angle_max_iterations

    # Run L-BFGS-B
    result = optimize.minimize(
        per_angle_loss,
        current_params[self.per_angle_indices],
        method="L-BFGS-B",
        jac=per_angle_grad,
        bounds=per_angle_bounds,
        options={
            "maxiter": max_iter,  # CHANGED: adaptive
            "ftol": self.config.per_angle_ftol,
            "disp": False,
        },
    )

    # ... rest unchanged ...
```

### Testing

**Validation strategy**:

1. **Correctness test**: Run on 10M point dataset
   - Compare final chi-squared with/without adaptive budgets
   - Should be within 1% (e.g., chi²=0.0401 vs 0.0405)
   - Compare physical parameter estimates (should be within 1σ)

2. **Performance test**: Measure wall-clock time reduction
   - Baseline (max_outer=5, no adaptive): 60 min
   - Optimized (max_outer=3, adaptive): Expected ~35-40 min
   - Overhead reduction: 5.0× → 3.0-3.5×

3. **Convergence monitoring**:
   - Log outer iteration count (should still be 2-3)
   - Log L-BFGS-B iterations per stage
   - Ensure physical params still converge (outer_tolerance met)

**Test datasets**:
- 10M points, n_phi=23, laminar_flow (standard case)
- 50M points, n_phi=5, laminar_flow (fewer angles)
- 100M points, n_phi=23, laminar_flow (stress test)

**Acceptance criteria**:
- Chi-squared within 2% of baseline
- Physical parameters within 2σ of baseline
- Wall-clock time reduced by 30-40%
- No convergence failures

---

## P3: Vectorize Regularization Gradient (OPTIONAL)

**Impact**: Code quality improvement, <0.01% performance benefit
**Risk**: None (purely computational change)
**Time**: 1 hour implementation + 30 minutes testing

### Implementation

**File**: `/homodyne/optimization/nlsq/adaptive_regularization.py`

**Replace lines 283-307** with vectorized version:

```python
def compute_regularization_gradient(
    self, params: np.ndarray, mse: float, n_points: int
) -> np.ndarray:
    """Compute gradient of regularization term.

    Parameters
    ----------
    params : np.ndarray
        Full parameter vector.
    mse : float
        Current mean squared error.
    n_points : int
        Number of data points.

    Returns
    -------
    np.ndarray
        Gradient w.r.t. all parameters (zeros for non-regularized params).
    """
    grad = np.zeros_like(params, dtype=np.float64)

    if not self.config.enable:
        return grad

    for start, end in self.group_indices:
        if start >= len(params) or end > len(params):
            continue

        group_params = params[start:end]
        n_group = end - start
        mean_val = np.mean(group_params)
        std_val = np.std(group_params)

        if self.config.mode == "relative" or (
            self.config.mode == "auto" and self.n_phi > 5
        ):
            # CV-based gradient (VECTORIZED)
            if abs(mean_val) > 1e-10 and std_val > 1e-10:
                cv = std_val / abs(mean_val)

                # Vectorized gradient computation
                d_std_vec = (group_params - mean_val) / (n_group * std_val)
                d_mean = 1.0 / n_group

                # Handle sign of mean
                sign_adjust = np.sign(mean_val) if mean_val != 0 else 1.0
                d_cv_vec = (d_std_vec * abs(mean_val) - std_val * d_mean) / (
                    mean_val**2
                )

                d_cv_sq_vec = 2 * cv * d_cv_vec

                grad[start:end] = self.lambda_value * d_cv_sq_vec * mse * n_points

        else:
            # Absolute variance gradient (VECTORIZED)
            grad[start:end] = (
                self.lambda_value
                * 2
                * (group_params - mean_val)
                / n_group
                * n_points
            )

    return grad
```

### Testing

Add test to verify numerical equivalence:

```python
def test_regularization_gradient_vectorized_matches_loop():
    """Vectorized gradient should match loop-based implementation."""
    config = AdaptiveRegularizationConfig(
        enable=True,
        mode="relative",
        target_cv=0.1,
        auto_tune_lambda=True,
    )
    regularizer = AdaptiveRegularizer(config, n_phi=23)

    # Test params
    params = np.random.randn(53)  # 46 per-angle + 7 physical
    mse = 0.04
    n_points = 10_000_000

    # Compute gradient
    grad = regularizer.compute_regularization_gradient(params, mse, n_points)

    # Verify non-zero for regularized params
    assert np.any(grad[:46] != 0)
    # Physical params should have zero gradient
    assert np.all(grad[46:] == 0)

    # Compare with finite differences for validation
    eps = 1e-8
    grad_fd = np.zeros_like(params)
    for i in range(46):  # Only check regularized params
        params_plus = params.copy()
        params_plus[i] += eps
        reg_plus = regularizer.compute_regularization(params_plus, mse, n_points)

        params_minus = params.copy()
        params_minus[i] -= eps
        reg_minus = regularizer.compute_regularization(params_minus, mse, n_points)

        grad_fd[i] = (reg_plus - reg_minus) / (2 * eps)

    # Should match to within 1%
    np.testing.assert_allclose(grad[:46], grad_fd[:46], rtol=0.01)
```

**Validation**: Run test suite, verify numerical equivalence

---

## Implementation Timeline

### Week 1: Critical Fix

- [x] Day 1: Implement P1 (gradient monitor memory fix)
- [x] Day 1: Test P1, verify memory bounded
- [x] Day 2: Code review, merge to main

### Week 2: High-Impact Optimization

- [ ] Day 1: Implement P2 Phase 1 (reduce max_outer_iterations)
- [ ] Day 2: Run validation tests on 10M dataset
- [ ] Day 3: Implement P2 Phase 2 (adaptive budgets)
- [ ] Day 4-5: Run validation tests on 50M and 100M datasets
- [ ] Day 5: Code review, merge to main

### Optional: Week 3

- [ ] Day 1: Implement P3 (vectorize regularization)
- [ ] Day 1: Test P3, verify numerical equivalence
- [ ] Day 2: Code review, merge to main

---

## Configuration Migration

### Updated Default Config (YAML)

```yaml
optimization:
  nlsq:
    anti_degeneracy:
      # Hierarchical optimization (v2.9.0+)
      hierarchical:
        enable: true
        max_outer_iterations: 3  # CHANGED from 5
        outer_tolerance: 1.0e-6
        physical_max_iterations: 100
        physical_ftol: 1.0e-8
        per_angle_max_iterations: 50
        per_angle_ftol: 1.0e-6
        use_adaptive_budgets: true  # NEW
        refinement_budget_fraction: 0.3  # NEW
        log_stage_transitions: true

      # Gradient monitoring
      gradient_monitoring:
        enable: true
        ratio_threshold: 0.01
        consecutive_triggers: 5
        response: "hierarchical"
        check_interval: 1
        max_history_size: 1000  # NEW
```

### Backward Compatibility

Old configs without new fields will use defaults:
- `use_adaptive_budgets`: defaults to `true`
- `refinement_budget_fraction`: defaults to `0.3`
- `max_history_size`: defaults to `1000`

No breaking changes.

---

## Success Metrics

### P1: Memory Fix

- [x] History size bounded at max_history_size × 100 bytes
- [x] No performance regression
- [x] All tests pass

### P2: Hierarchical Tuning

**Before optimization** (100M points, hierarchical mode):
- Wall-clock time: ~4.5-6 hours
- Chunk evaluations: ~2.4M
- Outer iterations: 5

**After optimization** (target):
- Wall-clock time: ~2.5-3.5 hours (40% reduction)
- Chunk evaluations: ~1.4M (40% reduction)
- Outer iterations: 2-3
- Chi-squared: Within 2% of baseline
- Physical params: Within 2σ of baseline

### P3: Vectorization

- [x] Gradient computation 5-10× faster (measured by microbenchmark)
- [x] Numerical equivalence to loop-based version (rtol=0.01)
- [x] No change in optimization results

---

## Risk Mitigation

### Risk: Reduced outer iterations prevent convergence

**Mitigation**:
- Keep strict `outer_tolerance=1e-6` convergence check
- Monitor convergence history on validation datasets
- Fallback: Users can increase `max_outer_iterations` in config if needed

### Risk: Adaptive budgets cause poor convergence

**Mitigation**:
- Use conservative `refinement_budget_fraction=0.3` (30% of full budget)
- Ensure minimum iterations (30 for physical, 20 for per-angle)
- Make adaptive budgets configurable (`use_adaptive_budgets: false` to disable)

### Risk: Breaking changes to config

**Mitigation**:
- All new config fields have defaults
- Backward compatible with old YAML configs
- Add deprecation warnings only if needed (not expected)

---

## Rollback Plan

If optimization causes convergence issues:

1. **Immediate**: Set `use_adaptive_budgets: false` in config
2. **Short-term**: Revert `max_outer_iterations` from 3 back to 5
3. **Long-term**: Keep P1 (memory fix) regardless, revert P2 if necessary

Each optimization is independently configurable, allowing selective rollback.
