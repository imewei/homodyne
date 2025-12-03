# Technical Debt Remediation Plan

**Generated:** December 2025
**Target Version:** v2.5.0
**Baseline:** v2.4.1 (CMC-only architecture)

---

## Executive Summary

This document provides a detailed implementation plan for addressing technical debt identified in the Homodyne codebase. The plan is structured into 5 priority tiers with actionable tasks, file locations, and verification criteria.

---

## Priority 1: Refactor `_create_numpyro_model` (CC=102)

**Current State:** 1,117 lines (lines 1969-3086 in `mcmc.py`)
**Target:** CC < 15 per extracted function
**Effort:** 3 days
**Impact:** High - Maintainability, Testability

### Analysis

The function handles multiple concerns:
1. **Dtype configuration** (lines 2157-2163)
2. **Reparameterization setup** (lines 2165-2250)
3. **Data array conversion** (lines 2170-2200)
4. **Prior sampling** (lines 2250-2500)
5. **Per-angle scaling logic** (lines 2500-2700)
6. **Single-angle surrogate mode** (lines 2550-2650)
7. **Physics model computation** (lines 2700-2950)
8. **Likelihood computation** (lines 2950-3086)

### Implementation Plan

#### Step 1: Extract Data Preparation Module

Create `homodyne/optimization/mcmc_data_prep.py`:

```python
# homodyne/optimization/mcmc_data_prep.py
"""Data preparation utilities for NumPyro MCMC models."""

from typing import Any
import jax.numpy as jnp

def prepare_mcmc_arrays(
    data, sigma, t1, t2, phi, q, L, dt, phi_full,
    target_dtype=jnp.float64
) -> dict[str, Any]:
    """Convert input arrays to JAX arrays with proper dtype.

    Returns dict with keys: data, sigma, t1, t2, phi, phi_full, q, L, dt
    """
    ...

def validate_array_shapes(arrays: dict[str, Any]) -> None:
    """Validate that all arrays have compatible shapes."""
    ...

def compute_phi_mapping(phi_full, phi_unique) -> jnp.ndarray:
    """Create mapping from data indices to phi angles."""
    ...
```

**Lines to extract:** 2170-2200 from `mcmc.py`

#### Step 2: Extract Prior Sampling Module

Create `homodyne/optimization/mcmc_priors.py`:

```python
# homodyne/optimization/mcmc_priors.py
"""Prior distribution sampling for NumPyro MCMC models."""

import numpyro
import numpyro.distributions as dist

def sample_physical_parameters(
    parameter_space,
    analysis_mode: str,
    single_angle_config: dict | None = None
) -> dict[str, Any]:
    """Sample physical parameters (D0, alpha, etc.) from priors."""
    ...

def sample_scaling_parameters(
    n_phi: int,
    per_angle_scaling: bool,
    fixed_overrides: dict | None = None,
    parameter_space=None
) -> tuple[Any, Any]:
    """Sample contrast and offset parameters.

    Returns (contrast, offset) - either scalars or arrays.
    """
    ...

def create_prior_distribution(
    param_name: str,
    bounds: tuple[float, float],
    prior_config: dict
) -> dist.Distribution:
    """Create NumPyro distribution from config."""
    ...
```

**Lines to extract:** 2250-2500 from `mcmc.py`

#### Step 3: Extract Per-Angle Scaling Module

Create `homodyne/optimization/mcmc_scaling.py`:

```python
# homodyne/optimization/mcmc_scaling.py
"""Per-angle scaling logic for MCMC models."""

import jax.numpy as jnp

def apply_per_angle_scaling(
    c1_theory: jnp.ndarray,
    contrast: jnp.ndarray,
    offset: jnp.ndarray,
    phi_mapping: jnp.ndarray | None
) -> jnp.ndarray:
    """Apply per-angle contrast and offset to C1 theory.

    c2 = offset[phi_idx] + contrast[phi_idx] * c1^2
    """
    ...

def validate_scaling_parameters(
    contrast, offset, n_phi: int, per_angle_scaling: bool
) -> None:
    """Validate scaling parameter shapes match expectations."""
    ...
```

**Lines to extract:** 2500-2700 from `mcmc.py`

#### Step 4: Extract Single-Angle Surrogate Module

Create `homodyne/optimization/mcmc_single_angle.py`:

```python
# homodyne/optimization/mcmc_single_angle.py
"""Single-angle surrogate model configuration."""

from typing import Any

def configure_single_angle_reparam(
    reparam_config: dict | None,
    surrogate_config: dict | None
) -> tuple[bool, dict, dict]:
    """Configure single-angle reparameterization.

    Returns (use_reparam, reparam_cfg, surrogate_cfg)
    """
    ...

def apply_d0_log_transform(
    log_d0_centered: float,
    center_loc: float,
    center_scale: float
) -> float:
    """Transform log-space D0 to linear space."""
    ...
```

**Lines to extract:** 2550-2650 from `mcmc.py`

#### Step 5: Refactor Main Function

After extraction, `_create_numpyro_model` becomes:

```python
def _create_numpyro_model(
    data, sigma, t1, t2, phi, q, L, analysis_mode, parameter_space,
    use_simplified=True, dt=None, phi_full=None, per_angle_scaling=True,
    single_angle_reparam_config=None, fixed_scaling_overrides=None,
    single_angle_surrogate_config=None
):
    """Create NumPyro probabilistic model using config-driven priors."""
    # 1. Prepare arrays
    arrays = prepare_mcmc_arrays(data, sigma, t1, t2, phi, q, L, dt, phi_full)

    # 2. Configure single-angle mode
    use_reparam, reparam_cfg, surrogate_cfg = configure_single_angle_reparam(
        single_angle_reparam_config, single_angle_surrogate_config
    )

    def model():
        # 3. Sample scaling parameters
        contrast, offset = sample_scaling_parameters(
            n_phi=len(phi), per_angle_scaling=per_angle_scaling,
            fixed_overrides=fixed_scaling_overrides
        )

        # 4. Sample physical parameters
        params = sample_physical_parameters(
            parameter_space, analysis_mode, surrogate_cfg
        )

        # 5. Compute physics
        c1 = compute_c1_theory(params, arrays, analysis_mode)

        # 6. Apply scaling
        c2 = apply_per_angle_scaling(c1, contrast, offset, arrays['phi_mapping'])

        # 7. Likelihood
        numpyro.sample('obs', dist.Normal(c2, arrays['sigma']), obs=arrays['data'])

    return model
```

**Target CC:** ~12 (from 102)

### Verification Criteria

1. All existing MCMC tests pass
2. Each extracted module has unit tests
3. CC < 15 for all new functions
4. No behavioral changes (same MCMC results)

### Files to Create

| File | Purpose | Lines |
|------|---------|-------|
| `homodyne/optimization/mcmc_data_prep.py` | Array preparation | ~80 |
| `homodyne/optimization/mcmc_priors.py` | Prior sampling | ~150 |
| `homodyne/optimization/mcmc_scaling.py` | Per-angle scaling | ~100 |
| `homodyne/optimization/mcmc_single_angle.py` | Single-angle mode | ~80 |

---

## Priority 2: Split `nlsq_wrapper.py` (4,358 lines)

**Current State:** Single 4,358-line file
**Target:** 4-5 modules, each < 800 lines
**Effort:** 2 days
**Impact:** High - Readability, Maintainability

### Current Structure Analysis

```
nlsq_wrapper.py (4,358 lines)
├── Helper functions (lines 110-424)
│   ├── FunctionEvaluationCounter class
│   ├── _build_parameter_labels
│   ├── _classify_parameter_status
│   ├── _sample_xdata
│   ├── _normalize_param_key
│   ├── _normalize_x_scale_map
│   ├── _build_per_parameter_x_scale
│   ├── _format_x_scale_for_log
│   ├── _parse_shear_transform_config
│   ├── _build_physical_index_map
│   ├── _apply_forward_shear_transforms_to_vector
│   ├── _apply_forward_shear_transforms_to_bounds
│   ├── _apply_inverse_shear_transforms_to_vector
│   ├── _adjust_covariance_for_transforms
│   ├── _wrap_model_function_with_transforms
│   ├── _wrap_stratified_function_with_transforms
│   └── _compute_jacobian_stats
├── OptimizationResult class (lines 425-482)
├── UseSequentialOptimization exception (lines 483-497)
└── NLSQWrapper class (lines 498-4358)
    ├── __init__
    ├── fit (main entry point)
    ├── _standard_fit
    ├── _large_fit
    ├── _chunked_fit
    ├── _streaming_fit
    └── ... (many more methods)
```

### Implementation Plan

#### Step 1: Extract Transform Utilities

Create `homodyne/optimization/nlsq_transforms.py`:

```python
# homodyne/optimization/nlsq_transforms.py
"""Shear transform utilities for NLSQ optimization."""

def _parse_shear_transform_config(config: Any | None) -> dict[str, Any]:
    ...

def _build_physical_index_map(labels: list[str], transform_map: dict) -> dict:
    ...

def _apply_forward_shear_transforms_to_vector(params, index_map, config) -> np.ndarray:
    ...

def _apply_forward_shear_transforms_to_bounds(bounds, index_map, config) -> tuple:
    ...

def _apply_inverse_shear_transforms_to_vector(params, index_map, config) -> np.ndarray:
    ...

def _adjust_covariance_for_transforms(pcov, index_map, config) -> np.ndarray:
    ...

def _wrap_model_function_with_transforms(func, index_map, config) -> callable:
    ...

def _wrap_stratified_function_with_transforms(func, index_map, config) -> callable:
    ...
```

**Lines to extract:** 233-395

#### Step 2: Extract Parameter Utilities

Create `homodyne/optimization/nlsq_parameters.py`:

```python
# homodyne/optimization/nlsq_parameters.py
"""Parameter handling utilities for NLSQ optimization."""

def _build_parameter_labels(param_names: list[str], n_phi: int) -> list[str]:
    ...

def _classify_parameter_status(
    result: OptimizationResult,
    config: dict
) -> dict[str, list[str]]:
    ...

def _normalize_param_key(name: str | None) -> str:
    ...

def _normalize_x_scale_map(raw_map: Any) -> dict[str, float]:
    ...

def _build_per_parameter_x_scale(
    param_labels: list[str],
    user_map: dict[str, float],
    default_scale: float
) -> np.ndarray:
    ...

def _format_x_scale_for_log(value: Any) -> str:
    ...
```

**Lines to extract:** 121-227

#### Step 3: Extract Result Classes

Create `homodyne/optimization/nlsq_result.py`:

```python
# homodyne/optimization/nlsq_result.py
"""Result classes for NLSQ optimization."""

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class OptimizationResult:
    """Container for NLSQ optimization results."""
    popt: np.ndarray
    pcov: np.ndarray | None
    chi_squared: float
    n_iterations: int
    n_func_evals: int
    strategy_used: str
    success: bool
    message: str
    parameter_labels: list[str]
    # ... additional fields

    def to_dict(self) -> dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationResult":
        ...

class UseSequentialOptimization(Exception):
    """Signal to switch to sequential angle optimization."""
    pass

class FunctionEvaluationCounter:
    """Thread-safe function evaluation counter."""
    ...
```

**Lines to extract:** 110-120, 425-497

#### Step 4: Extract Strategy Methods

Create `homodyne/optimization/nlsq_strategies.py`:

```python
# homodyne/optimization/nlsq_strategies.py
"""Optimization strategy implementations for NLSQ."""

def standard_fit(wrapper, xdata, ydata, sigma, p0, bounds, **kwargs):
    """Standard curve_fit strategy for small datasets."""
    ...

def large_fit(wrapper, xdata, ydata, sigma, p0, bounds, **kwargs):
    """Large dataset strategy with streaming."""
    ...

def chunked_fit(wrapper, xdata, ydata, sigma, p0, bounds, **kwargs):
    """Chunked processing for memory efficiency."""
    ...

def streaming_fit(wrapper, xdata, ydata, sigma, p0, bounds, **kwargs):
    """Streaming strategy for very large datasets."""
    ...
```

**Lines to extract:** Large portions from NLSQWrapper methods

#### Step 5: Refactor Main Class

After extraction, `nlsq_wrapper.py` becomes:

```python
# homodyne/optimization/nlsq_wrapper.py
"""Main NLSQ optimization wrapper - orchestration only."""

from .nlsq_result import OptimizationResult, UseSequentialOptimization
from .nlsq_parameters import (
    _build_parameter_labels, _build_per_parameter_x_scale
)
from .nlsq_transforms import (
    _wrap_model_function_with_transforms,
    _apply_forward_shear_transforms_to_vector
)
from .nlsq_strategies import standard_fit, large_fit, chunked_fit, streaming_fit

class NLSQWrapper:
    """Orchestrates NLSQ optimization with automatic strategy selection."""

    def fit(self, xdata, ydata, sigma=None, **kwargs) -> OptimizationResult:
        """Main entry point - selects and executes appropriate strategy."""
        strategy = self._select_strategy(len(ydata))
        return strategy(self, xdata, ydata, sigma, **kwargs)

    def _select_strategy(self, data_size: int) -> callable:
        """Select optimization strategy based on data size."""
        if data_size < 10_000:
            return standard_fit
        elif data_size < 100_000:
            return large_fit
        elif data_size < 1_000_000:
            return chunked_fit
        else:
            return streaming_fit
```

### Resulting File Structure

| File | Lines | Purpose |
|------|-------|---------|
| `nlsq_wrapper.py` | ~800 | Main orchestration |
| `nlsq_result.py` | ~200 | Result classes |
| `nlsq_parameters.py` | ~150 | Parameter utilities |
| `nlsq_transforms.py` | ~250 | Transform utilities |
| `nlsq_strategies.py` | ~600 | Strategy implementations |

### Verification Criteria

1. All existing NLSQ tests pass
2. No import cycles
3. Public API unchanged
4. Each file < 800 lines

---

## Priority 3: Address TODO/FIXME Items (12 items)

**Effort:** 1 day
**Impact:** Medium - Code quality, Correctness

### Actionable TODOs

| # | Location | TODO | Action |
|---|----------|------|--------|
| 1 | `nlsq.py:944` | Implement proper error estimation | Implement finite-difference or bootstrap error estimation |
| 2 | `nlsq.py:964` | Re-evaluate residuals at final point | Add residual re-evaluation for accurate chi-squared |
| 3 | `pjit.py:399` | Implement true pmap parallelization | Evaluate if pmap is needed or mark as WONTFIX |
| 4 | `coordinator.py:190` | Initialize CheckpointManager | Implement checkpoint integration or remove placeholder |
| 5 | `coordinator.py:641` | Integrate full validation module | Connect to existing validation infrastructure |
| 6 | `post_install.py:478` | Create macOS aliases script | Implement or document as unsupported |

### Implementation Details

#### TODO #1: Error Estimation (nlsq.py:944)

```python
# Current (placeholder):
# TODO: Implement proper error estimation using finite differences or bootstrap

# Proposed implementation:
def estimate_parameter_errors(
    popt: np.ndarray,
    residual_func: callable,
    method: str = "finite_diff"
) -> np.ndarray:
    """Estimate parameter standard errors.

    Methods:
    - finite_diff: Use numerical Jacobian at solution
    - bootstrap: Bootstrap resampling (slower but robust)
    """
    if method == "finite_diff":
        # Compute Jacobian numerically
        J = approx_fprime(popt, residual_func, epsilon=1e-8)
        # Covariance from J^T J inverse
        JTJ = J.T @ J
        pcov = np.linalg.pinv(JTJ)
        return np.sqrt(np.diag(pcov))
    elif method == "bootstrap":
        # Implement bootstrap resampling
        ...
```

#### TODO #4: CheckpointManager (coordinator.py:190)

```python
# Current:
# TODO (Phase 2): Initialize CheckpointManager

# Decision: Either implement or remove
# Option A - Implement:
from homodyne.optimization.checkpoint_manager import CheckpointManager

self.checkpoint_mgr = CheckpointManager(
    checkpoint_dir=config.get("checkpoint_dir"),
    checkpoint_interval=config.get("checkpoint_interval", 100)
)

# Option B - Remove placeholder:
# Delete the TODO comment and document that checkpointing is not yet supported
```

### Non-Actionable TODOs (Document or Remove)

| Location | Status | Action |
|----------|--------|--------|
| `pjit.py:379` | Documentation note | Convert to docstring |

---

## Priority 4: Remove Deprecated References (19 items)

**Effort:** 0.5 days
**Impact:** Medium - Clean codebase

### Deprecated Items by Category

#### Category 1: CMC Selection Logic (should_use_cmc)

**Files affected:**
- `homodyne/device/config.py:254-261`

**Action:** Keep as deprecated shim for backward compatibility, add deprecation warning.

```python
# Current implementation is correct - already a deprecated shim
def should_use_cmc(...) -> bool:
    """Deprecated shim; CMC is always selected."""
    warnings.warn(
        "should_use_cmc is deprecated; CMC is the only MCMC path now.",
        DeprecationWarning,
        stacklevel=2
    )
    return True
```

**Decision:** Keep for v2.x, remove in v3.0

#### Category 2: Config Migration Warnings

**Files affected:**
- `homodyne/config/manager.py:807-832`
- `homodyne/cli/commands.py:417-464`

**Action:** These are intentional deprecation checks - keep them. They help users migrate.

#### Category 3: Truly Unused Deprecated Code

**Files to audit:**
- `homodyne/config/types.py:290-298` - CMCInitializationConfig class

**Action:** Mark for removal in v3.0, keep for backward compatibility in v2.x

### Implementation

1. **Add `@deprecated` decorator** for formal deprecation:

```python
# homodyne/utils/deprecation.py
import warnings
from functools import wraps

def deprecated(reason: str, removal_version: str = "3.0"):
    """Mark function/class as deprecated."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}. "
                f"Will be removed in v{removal_version}.",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

2. **Document deprecations in CHANGELOG.md**

3. **Create migration guide** for v2.x → v3.0

---

## Priority 5: Reduce Duplicate Patterns (180 items)

**Effort:** 2 days
**Impact:** Low - DRY compliance

### Common Duplication Patterns

Based on codebase analysis, key duplication areas:

#### Pattern 1: Array Validation

```python
# Duplicated in multiple files:
if data is None or data.size == 0:
    raise ValueError("Data cannot be None or empty")
```

**Solution:** Create `homodyne/utils/validation.py`:

```python
def validate_array(arr, name: str, allow_empty: bool = False) -> np.ndarray:
    """Validate array is not None and optionally not empty."""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    arr = np.asarray(arr)
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return arr
```

#### Pattern 2: Parameter Bounds Checking

```python
# Duplicated pattern:
if value < bounds[0] or value > bounds[1]:
    raise ValueError(f"Parameter {name} out of bounds")
```

**Solution:** Use existing ParameterSpace validation.

#### Pattern 3: Logging Setup

Multiple files have similar logging initialization.

**Solution:** Centralize in `homodyne/utils/logging.py` (already exists, ensure consistent usage).

### Deduplication Priorities

| Pattern | Occurrences | Priority | Action |
|---------|-------------|----------|--------|
| Array validation | ~25 | High | Create utility function |
| Bounds checking | ~15 | Medium | Use ParameterSpace |
| Logging init | ~10 | Low | Document best practice |
| Error messages | ~20 | Low | Create message constants |

---

## Implementation Schedule

### Phase 1: High Priority (Week 1-2)

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Extract `mcmc_data_prep.py` | New module + tests |
| 3 | Extract `mcmc_priors.py` | New module + tests |
| 4 | Extract `mcmc_scaling.py`, `mcmc_single_angle.py` | New modules + tests |
| 5 | Refactor `_create_numpyro_model` | Reduced CC |
| 6-7 | Extract NLSQ modules | 4 new modules |
| 8 | Refactor `NLSQWrapper` | Clean orchestration |

### Phase 2: Medium Priority (Week 3)

| Day | Task | Deliverable |
|-----|------|-------------|
| 9 | Address TODOs #1, #2 (error estimation) | Improved accuracy |
| 10 | Address TODOs #4, #5 (checkpointing) | Decision + implementation |
| 11 | Audit deprecated references | Migration guide |

### Phase 3: Low Priority (Week 4)

| Day | Task | Deliverable |
|-----|------|-------------|
| 12-13 | Create validation utilities | `homodyne/utils/validation.py` |
| 14 | Update all callers | No duplication |

---

## Verification Checklist

### After Each Change

- [ ] All existing tests pass (`make test-all`)
- [ ] No new linting errors (`make lint`)
- [ ] Type checking passes (`make typecheck`)
- [ ] Coverage >= 80%

### After Phase Completion

- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No import cycles
- [ ] No behavioral changes (same outputs for same inputs)

---

## Risk Mitigation

1. **Regression Risk:** Run full test suite after each extraction
2. **Import Cycles:** Use lazy imports if needed
3. **API Breaks:** Maintain public API, only change internals
4. **Performance:** Benchmark before/after major changes

---

## Success Metrics

| Metric | Before | Target | Current (Dec 2025) |
|--------|--------|--------|-------------------|
| `_create_numpyro_model` CC | 102 | < 15 | Extracted to 4 modules |
| `nlsq_wrapper.py` lines | 4,358 | < 800 | Extracted 3 modules |
| TODO/FIXME items | 12 | 0 actionable | 0 (all addressed) |
| Deprecated references | 19 | 0 (or documented) | Documented |
| Duplicate patterns | 180 | < 50 | Validation utilities created |

---

## Implementation Status (Dec 2025)

### Completed

**Priority 1: Refactor `_create_numpyro_model` (CC=102)**
- Created `mcmc_data_prep.py` - Array preparation utilities
- Created `mcmc_priors.py` - Prior distribution sampling
- Created `mcmc_scaling.py` - Per-angle scaling logic
- Created `mcmc_single_angle.py` - Single-angle surrogate mode

**Priority 2: Split `nlsq_wrapper.py` (4,358 lines)**
- Created `nlsq_transforms.py` - Shear transform utilities
- Created `nlsq_results.py` - Result dataclasses
- Created `nlsq_jacobian.py` - Jacobian computation utilities

**Priority 3: Address TODO/FIXME Items (12 items)**
- Removed dead code functions in `nlsq.py`
- Updated pjit.py TODO (obsolete in CPU-only architecture)
- Documented Phase 2 TODOs as deferred in `coordinator.py`
- Updated post_install.py macOS aliases as not implemented
- All 0 actionable TODOs remaining

**Priority 4: Remove Deprecated References (19 items)**
- Updated runtime/README.md (removed obsolete GPU references)
- Updated __init__.py (clarified CPU-only architecture)
- Deprecated references documented for v2.x compatibility

**Priority 5: Reduce Duplicate Patterns (180 items)**
- Created `homodyne/utils/validation.py` with centralized validation functions
- Functions: `validate_array_not_none`, `validate_array_not_empty`, etc.
- Available for gradual adoption across codebase
