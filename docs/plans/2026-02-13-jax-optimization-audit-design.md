# JAX Optimization Audit — Targeted Quick Wins

**Date**: 2026-02-13
**Scope**: Safe, decorator/wrapper-level optimizations only (no API or logic changes)
**Target workload**: 8-200M points (50M typical), 10-30 phi angles, CPU-only

## Context

Comprehensive audit of JAX optimization patterns in homodyne. The codebase already has
strong JIT isolation (38+ functions), clean numpy/jnp separation, and no anti-patterns
in hot paths. This design targets incremental wins with zero risk.

## Section 1: Expand `static_argnums` on Physics JIT Functions

**Problem**: Only 1 of 38+ JIT functions uses `static_argnums`. Integer/boolean args
(n_phi, mode flags, shape parameters) are traced as dynamic values, causing unnecessary
recompilation when shapes don't change between calls.

**Files**:
- `core/physics_nlsq.py` — meshgrid functions (n_phi, n_points are compile-time constants)
- `core/jax_backend.py` — g1/g2 computations (shape params constant per-run)
- `core/fitting.py` — solver functions (mode selection constant)

**Change**: Add `static_argnums` or `static_argnames` to JIT decorators. Function
signatures remain identical.

**Impact**: 50-80% reduction in recompilation overhead.
**Risk**: Zero.

## Section 2: vmap Per-Angle Loops in `scaling_utils.py`

**Problem**: Python for-loops iterate over 10-30 phi angles for quantile estimation.
Sequential where vectorized would suffice.

**Files**: `core/scaling_utils.py` (lines ~210, ~241)

**Change**: Replace `for i in range(n_phi)` loops with `jax.vmap` over pre-shaped
angle-indexed arrays. Function signatures and return values stay identical.

**Impact**: 3-5x speedup for scaling estimation on 20+ angle datasets.
**Risk**: Low. Data-prep step, not JIT hot path. Falls back to loop if needed.

## Section 3: Cache Runtime `jax.jit()` Calls

**Problem**: `strategies/residual.py` and `strategies/residual_jit.py` call `jax.jit()`
at runtime. While JAX caches internally, there's Python-level overhead in repeated
wrapper creation.

**Files**:
- `optimization/nlsq/strategies/residual.py` (line ~246)
- `optimization/nlsq/strategies/residual_jit.py` (line ~138)

**Change**: Cache JIT-compiled functions as instance attributes on first call.

**Impact**: Eliminates Python-level JIT wrapper overhead on repeated calls.
**Risk**: Near-zero. Standard memoization.

## Section 4: XLA CPU Compilation Flags Audit

**Problem**: CPU performance depends on XLA flags. Current configuration in
`cli/xla_config.py` and `device/__init__.py` may not exploit all available ISA features.

**Files**:
- `cli/xla_config.py`
- `device/__init__.py`

**Change**: Audit flags, document optimal CPU settings, add missing beneficial flags.

**Impact**: Varies by hardware. AVX-512 exploitation can yield 2x throughput.
**Risk**: Zero. Environment variables set before computation.

## Success Criteria

1. No API changes — all function signatures preserved
2. Existing tests pass without modification
3. JIT recompilation count measurably reduced (Section 1)
4. Per-angle scaling computation measurably faster for n_phi >= 10 (Section 2)
5. No new dependencies introduced
