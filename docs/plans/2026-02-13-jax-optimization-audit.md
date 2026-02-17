# JAX Optimization Audit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task.

**Goal:** Add `static_argnums` to 15 JIT-compiled physics functions to enable XLA
constant folding for dataset-invariant parameters, and audit XLA CPU flags for
completeness.

**Architecture:** All changes are decorator-level (`@jit` →
`@partial(jit, static_argnums=(...))`) or configuration audit. No function signatures,
logic, or APIs change. The `functools.partial` pattern already exists in
`fit_computation.py:27`.

**Tech Stack:** JAX (`jax.jit`, `functools.partial`), existing test suite via pytest

______________________________________________________________________

## Context: What Deep Analysis Revealed

The initial audit identified 4 optimization areas. Deep analysis eliminated 2:

| Section | Status | Reason | |---------|--------|--------| | static_argnums expansion |
**PROCEED** | 15 functions have truly-static float params (dt, q, L, prefactors) | |
vmap per-angle loops | **DROPPED** | Dynamic shapes (variable points per angle) prevent
vmap; already optimized with pre-sorted grouped ops | | JIT caching in residual
strategies | **DROPPED** | Already properly cached as instance attributes in `__init__`
| | XLA CPU flags audit | **PROCEED** | Comprehensive config exists; verify completeness
|

**CRITICAL safety rules for static_argnums:**

- `contrast` and `offset`: **NEVER static** — they vary per optimization call
- Physics params (D0, alpha, etc.): **NEVER static** — they're optimized each iteration
- Dataset constants (dt, q, L, sinc_prefactor, wavevector_q_squared_half_dt): **SAFE** —
  constant per dataset run

______________________________________________________________________

### Task 1: Add static_argnums to `core/jax_backend.py` (7 functions)

**Files:**

- Modify: `homodyne/core/jax_backend.py`
- Test: `tests/unit/test_fit_computation.py`, `tests/unit/test_nlsq_core.py`

**Step 1: Add `from functools import partial` import**

Check if already imported. If not, add at top of file:

```python
from functools import partial
```

**Step 2: Update `safe_divide` (line 411)**

```python
# Before:
@jit
def safe_divide(a: jnp.ndarray, b: jnp.ndarray, default: float = 0.0) -> jnp.ndarray:

# After:
@partial(jit, static_argnums=(2,))
def safe_divide(a: jnp.ndarray, b: jnp.ndarray, default: float = 0.0) -> jnp.ndarray:
```

**Step 3: Update `_compute_g1_diffusion_core` (line 420)**

```python
# Before:
@jit
def _compute_g1_diffusion_core(
    params, t1, t2, wavevector_q_squared_half_dt, dt,
):

# After:
@partial(jit, static_argnums=(3, 4))
def _compute_g1_diffusion_core(
    params, t1, t2, wavevector_q_squared_half_dt, dt,
):
```

Static args: `wavevector_q_squared_half_dt` (3), `dt` (4)

**Step 4: Update `_compute_g1_shear_core` (line 538)**

```python
# Before:
@jit
# After:
@partial(jit, static_argnums=(4, 5))
```

Static args: `sinc_prefactor` (4), `dt` (5)

**Step 5: Update `_compute_g1_total_core` (line 737)**

```python
# Before:
@jit
# After:
@partial(jit, static_argnums=(4, 5, 6))
```

Static args: `wavevector_q_squared_half_dt` (4), `sinc_prefactor` (5), `dt` (6)

**Step 6: Update `_compute_g2_scaled_core` (line 822)**

```python
# Before:
@jit
# After:
@partial(jit, static_argnums=(4, 5, 8))
```

Static args: `wavevector_q_squared_half_dt` (4), `sinc_prefactor` (5), `dt` (8) **NOT
static**: `contrast` (6), `offset` (7) — these vary during optimization

**Step 7: Update `compute_g2_scaled_with_factors` (line 1101)**

```python
# Before:
@jit
# After:
@partial(jit, static_argnums=(4, 5, 8))
```

Same rationale as Step 6.

**Step 8: Update `compute_chi_squared` (line 1169)**

```python
# Before:
@jit
# After:
@partial(jit, static_argnums=(6, 7, 10))
```

Static args: `q` (6), `L` (7), `dt` (10) **NOT static**: `contrast` (8), `offset` (9) —
vary during optimization

**Step 9: Run tests to verify**

Run: `uv run pytest tests/unit/test_fit_computation.py tests/unit/test_nlsq_core.py -v`
Expected: All tests PASS

**Step 10: Commit**

```bash
git add homodyne/core/jax_backend.py
git commit -m "perf(jax): add static_argnums to jax_backend.py JIT functions

Enable XLA constant folding for dataset-invariant parameters (dt, q, L,
wavevector_q_squared_half_dt, sinc_prefactor). Contrast and offset remain
traced as they vary during optimization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 2: Add static_argnums to `core/physics_nlsq.py` (5 functions)

**Files:**

- Modify: `homodyne/core/physics_nlsq.py`
- Test: `tests/unit/test_fit_computation.py`, `tests/unit/test_nlsq_core.py`

**Step 1: Add `from functools import partial` import if missing**

**Step 2: Update `_compute_g1_diffusion_meshgrid` (line 58)**

```python
@partial(jit, static_argnums=(3, 4))
def _compute_g1_diffusion_meshgrid(
    params, t1, t2, wavevector_q_squared_half_dt, dt,
):
```

**Step 3: Update `_compute_g1_shear_meshgrid` (line 138)**

```python
@partial(jit, static_argnums=(4, 5))
def _compute_g1_shear_meshgrid(
    params, t1, t2, phi, sinc_prefactor, dt,
):
```

**Step 4: Update `_compute_g1_total_meshgrid` (line 274)**

```python
@partial(jit, static_argnums=(4, 5, 6))
def _compute_g1_total_meshgrid(
    params, t1, t2, phi, wavevector_q_squared_half_dt, sinc_prefactor, dt,
):
```

**Step 5: Update `_compute_g2_scaled_meshgrid` (line 337)**

```python
@partial(jit, static_argnums=(4, 5, 8))
def _compute_g2_scaled_meshgrid(
    params, t1, t2, phi, wavevector_q_squared_half_dt, sinc_prefactor, contrast, offset, dt,
):
```

**NOT static**: `contrast` (6), `offset` (7)

**Step 6: Update `compute_g2_scaled_with_factors` (line 467)**

```python
@partial(jit, static_argnums=(4, 5, 8))
def compute_g2_scaled_with_factors(
    params, t1, t2, phi, wavevector_q_squared_half_dt, sinc_prefactor, contrast, offset, dt,
):
```

**Step 7: Run tests**

Run: `uv run pytest tests/unit/test_fit_computation.py tests/unit/test_nlsq_core.py -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add homodyne/core/physics_nlsq.py
git commit -m "perf(jax): add static_argnums to physics_nlsq.py meshgrid functions

Same constant-folding optimization as jax_backend.py. Dataset constants
(dt, q, sinc_prefactor, wavevector_q_squared_half_dt) marked static.
Contrast/offset remain traced.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 3: Add static_argnums to `core/physics_cmc.py` (3 functions)

**Files:**

- Modify: `homodyne/core/physics_cmc.py`
- Test: `tests/unit/optimization/cmc/test_core.py`

**Step 1: Add `from functools import partial` import if missing**

**Step 2: Update `_compute_g1_diffusion_elementwise` (line 49)**

```python
@partial(jit, static_argnums=(4,))
def _compute_g1_diffusion_elementwise(
    params, t1, t2, time_grid, wavevector_q_squared_half_dt,
):
```

Static: `wavevector_q_squared_half_dt` (4)

**Step 3: Update `_compute_g1_shear_elementwise` (line 98)**

```python
@partial(jit, static_argnums=(4,))
def _compute_g1_shear_elementwise(
    params, t1, t2, phi_unique, sinc_prefactor, time_grid,
):
```

Static: `sinc_prefactor` (4)

**Step 4: Update `_compute_g1_total_elementwise` (line 177)**

```python
@partial(jit, static_argnums=(5, 6))
def _compute_g1_total_elementwise(
    params, t1, t2, phi_unique, time_grid, wavevector_q_squared_half_dt, sinc_prefactor,
):
```

Static: `wavevector_q_squared_half_dt` (5), `sinc_prefactor` (6)

**Step 5: Run tests**

Run: `uv run pytest tests/unit/optimization/cmc/test_core.py -v` Expected: All tests
PASS

**Step 6: Commit**

```bash
git add homodyne/core/physics_cmc.py
git commit -m "perf(jax): add static_argnums to physics_cmc.py elementwise functions

Dataset-constant prefactors (wavevector_q_squared_half_dt, sinc_prefactor)
marked static for XLA constant folding. Safe because these are Python
floats computed in the wrapper, not NumPyro traced values.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 4: Add static_argnums to `core/fitting.py` (1 function)

**Files:**

- Modify: `homodyne/core/fitting.py`
- Test: `tests/unit/test_fit_computation.py`

**Step 1: Update `solve_least_squares_general_jax` (line 650)**

```python
@partial(jit, static_argnums=(2,))
def solve_least_squares_general_jax(
    design_matrix, target_vector, regularization=1e-10,
):
```

Static: `regularization` (2) — always 1e-10 default

**Step 2: Run tests**

Run: `uv run pytest tests/unit/test_fit_computation.py -v` Expected: All tests PASS

**Step 3: Commit**

```bash
git add homodyne/core/fitting.py
git commit -m "perf(jax): add static_argnums to fitting.py regularization param

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

______________________________________________________________________

### Task 5: XLA CPU Flags Audit

**Files:**

- Read: `homodyne/__init__.py`, `homodyne/device/cpu.py`, `homodyne/device/__init__.py`
- Read: `homodyne/runtime/shell/activation/xla_config.bash`

**Step 1: Verify current flags are comprehensive**

Current XLA flags audit:

- `--xla_force_host_platform_device_count=4` ✓ (virtual devices for CMC)
- `--xla_disable_hlo_passes=constant_folding` ✓ (prevents slow compilation for large
  datasets)
- `--xla_cpu_multi_thread_eigen=true` ✓ (in cpu.py)
- `--xla_cpu_enable_fast_math=true` ✓ (AVX-512 detected)

**Step 2: Check for missing beneficial flags**

Potentially beneficial flags to evaluate:

- `--xla_cpu_enable_fast_min_max=true` — Faster min/max for clipping operations
- `--xla_cpu_fast_math_honor_nans=true` — Maintain NaN correctness with fast math
  (CRITICAL for XPCS)

**Step 3: Document findings**

Document the complete flag inventory and any recommendations in the audit report. No
code changes unless a clearly beneficial missing flag is identified.

**Step 4: Commit documentation if any changes made**

______________________________________________________________________

### Task 6: Run Full Test Suite

**Step 1: Run full unit test suite**

Run: `uv run pytest tests/unit/ -v --tb=short` Expected: All tests PASS (no regressions
from decorator changes)

**Step 2: Run integration tests if available**

Run: `uv run pytest tests/integration/ -v --tb=short` Expected: All tests PASS

**Step 3: Final verification commit if needed**

______________________________________________________________________

## Summary of Changes

| File | Functions Modified | Change | |------|-------------------|--------| |
`core/jax_backend.py` | 7 | `@jit` → `@partial(jit, static_argnums=(...))` | |
`core/physics_nlsq.py` | 5 | Same pattern | | `core/physics_cmc.py` | 3 | Same pattern |
| `core/fitting.py` | 1 | Same pattern | | **Total** | **16** | Decorator-only changes |

## What Was Ruled Out (and Why)

1. **vmap in scaling_utils.py**: Dynamic shapes (variable points per angle) prevent
   vmap. Boolean indexing in quantile estimation creates ragged arrays. Already
   optimized with pre-sorted grouped operations.

1. **JIT caching in residual strategies**: Both `residual.py` and `residual_jit.py`
   already cache JIT results as instance attributes during `__init__`. No work needed.

1. **static_argnums for physics params (D0, alpha, etc.)**: These are OPTIMIZED
   parameters that change every iteration. Making them static would cause catastrophic
   recompilation.

1. **static_argnums for contrast/offset**: These vary per optimization call in per-angle
   modes. Must remain traced.
