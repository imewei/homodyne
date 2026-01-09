# Homodyne NLSQ Fitting Architecture

Complete documentation of the NLSQ (Nonlinear Least Squares) fitting system in homodyne.

**Version:** 2.14.0
**Last Updated:** January 2026

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Setup Phase](#1-setup-phase)
3. [Adapter Selection](#2-adapter-selection)
4. [Memory & Strategy Selection](#3-memory--strategy-selection)
5. [Stratification Decision](#4-stratification-decision)
6. [Residual Function Setup](#5-residual-function-setup)
7. [Anti-Degeneracy Defense System](#6-anti-degeneracy-defense-system)
8. [Strategy Execution](#7-strategy-execution)
9. [Error Recovery](#8-error-recovery)
10. [Result Building](#9-result-building)
11. [Quick Reference Tables](#quick-reference-tables)
12. [Key Files Reference](#key-files-reference)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER ENTRY POINTS                                   │
│                                                                                  │
│   fit_nlsq_jax(data, config)                fit_nlsq_multistart(data, config)   │
│         (core.py)                                 (multistart.py)                │
│              │                                          │                        │
│              │                               Latin Hypercube Sampling            │
│              │                               N starting points (default 10)      │
│              │                                          │                        │
│              └────────────────────┬─────────────────────┘                        │
│                                   ▼                                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            1. SETUP PHASE                                        │
│                               (core.py)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      2. ADAPTER SELECTION (T021-T024)                            │
│                                                                                  │
│                    NLSQAdapter ◄──────────► NLSQWrapper                          │
│                   (with fallback)         (stable fallback)                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     3. MEMORY & STRATEGY SELECTION                               │
│                             (memory.py)                                          │
│                                                                                  │
│              STANDARD ◄────► OUT_OF_CORE ◄────► HYBRID_STREAMING                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      4. STRATIFICATION DECISION                                  │
│                       (strategies/chunking.py)                                   │
│                                                                                  │
│                    Angle-Stratified Chunking (if needed)                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       5. RESIDUAL FUNCTION SETUP                                 │
│                                                                                  │
│         StratifiedResidualFunction ◄────► StratifiedResidualFunctionJIT          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│               6. ANTI-DEGENERACY DEFENSE SYSTEM (laminar_flow only)              │
│                                                                                  │
│     Layer 1: Fourier Reparameterization                                          │
│     Layer 2: Hierarchical Optimization                                           │
│     Layer 3: Adaptive CV Regularization                                          │
│     Layer 4: Gradient Collapse Monitor                                           │
│     Layer 5: Shear-Sensitivity Weighting                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         7. STRATEGY EXECUTION                                    │
│                        (strategies/executors.py)                                 │
│                                                                                  │
│     StandardExecutor │ LargeDatasetExecutor │ StreamingExecutor                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          8. ERROR RECOVERY                                       │
│                            (wrapper.py)                                          │
│                                                                                  │
│                    3-Attempt Recovery with Perturbation                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          9. RESULT BUILDING                                      │
│                          (result_builder.py)                                     │
│                                                                                  │
│                         OptimizationResult                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Setup Phase

**File:** `core.py`

### Input Validation & Conversion

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Input Validation & Conversion                                              │
│                                                                            │
│ • Validate analysis_mode: static_isotropic (3 phys) | laminar_flow (7)    │
│ • Data format conversion:                                                  │
│     phi_angles_list → phi                                                  │
│     c2_exp → g2                                                            │
│     wavevector_q_list[0] → q (scalar)                                      │
│ • Extract 1D time vectors from 2D meshgrids:                               │
│     t1_2d[:, 0] → t1_1d                                                    │
│     t2_2d[0, :] → t2_1d                                                    │
│ • Generate default sigma (1%) if missing                                   │
│ • Load L (stator_rotor_gap or sample_detector_distance) from config        │
└───────────────────────────────────────────────────────────────────────────┘
```

### Parameter Initialization

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Parameter Initialization                                                   │
│                                                                            │
│ • Load initial params from config OR estimate from data                    │
│ • Setup parameter bounds via ParameterManager                              │
│ • Parameter structure:                                                     │
│                                                                            │
│   static_isotropic:  [contrast, offset, D0, alpha, D_offset]              │
│   laminar_flow:      [contrast, offset, D0, alpha, D_offset,              │
│                       gamma_dot_t0, beta, gamma_dot_t_offset, phi0]       │
│                                                                            │
│   With per-angle scaling (n_phi angles):                                   │
│     [c0, c1, ..., c_{n-1}, o0, o1, ..., o_{n-1}, *physical_params]        │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Adapter Selection

**Decision Point:** T021-T024

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ADAPTER SELECTION (T021-T024)                               │
│                                                                                  │
│                           use_adapter=True?                                      │
│                                  │                                               │
│                    ┌─────────────┴─────────────┐                                 │
│                   YES                         NO                                 │
│                    │                           │                                 │
│                    ▼                           │                                 │
│   ┌────────────────────────────────┐           │                                 │
│   │   Try NLSQAdapter (adapter.py) │           │                                 │
│   │                                │           │                                 │
│   │ • Model caching via            │           │                                 │
│   │   WeakValueDictionary          │           │                                 │
│   │   (3-5× speedup for multistart)│           │                                 │
│   │ • JIT compilation via          │           │                                 │
│   │   NLSQ CurveFit class          │           │                                 │
│   │   (2-3× per-iteration speedup) │           │                                 │
│   │ • Native NLSQ stability        │           │                                 │
│   └────────────────────────────────┘           │                                 │
│                    │                           │                                 │
│               Success?                         │                                 │
│                    │                           │                                 │
│          ┌────────┴────────┐                   │                                 │
│         YES               NO                   │                                 │
│          │                 │                   │                                 │
│          ▼                 ▼                   ▼                                 │
│   ┌────────────┐    ┌─────────────────────────────────────────────────────────┐ │
│   │  Return    │    │            NLSQWrapper (wrapper.py)                      │ │
│   │  Result    │    │                                                          │ │
│   │            │    │  • Full anti-degeneracy defense system (5 layers)       │ │
│   │ fallback   │    │  • 3-attempt error recovery with parameter perturbation │ │
│   │ =False     │    │  • Streaming mode support (AdaptiveHybridStreaming)     │ │
│   └────────────┘    │  • Per-angle consistent initialization (v2.7.1 fix)     │ │
│                     │  • Angle-stratified chunking for per-angle scaling      │ │
│                     │                                                          │ │
│                     │  Sets: fallback_occurred=True (if came from adapter)    │ │
│                     └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Adapter Comparison

| Feature | NLSQAdapter | NLSQWrapper |
|---------|-------------|-------------|
| Model caching | Yes (3-5× speedup) | No |
| JIT compilation | Yes (2-3× speedup) | No |
| Anti-degeneracy | No | Yes (5 layers) |
| Error recovery | Basic | 3-attempt with perturbation |
| Streaming mode | No | Yes |
| Best for | Standard fits, multistart | Complex laminar_flow, large datasets |

---

## 3. Memory & Strategy Selection

**File:** `memory.py`

### Memory Estimation

```
peak_memory_gb = n_points × n_params × 8 bytes × 6.5× / 1e9

6.5× Jacobian Overhead Factor (v2.14.0):
├─ 1.0× Base Jacobian matrix
├─ 2.0× Autodiff intermediates (JAX gradient computation)
├─ 1.5× Stratified array padding (copy + padding overhead)
├─ 1.5× JIT compilation buffers (XLA, remat traces)
└─ 0.5× Optimizer working memory (QR, trust-region storage)

Example: 23M points × 53 params × 8 × 6.5 = 63.6 GB
```

### Adaptive Threshold

```
threshold = total_system_RAM × memory_fraction

memory_fraction = 0.75 (default, override via NLSQ_MEMORY_FRACTION env var)
Detection priority: psutil → sysconf → fallback 16 GB

Example: 64 GB system × 0.75 = 48 GB threshold
```

### Strategy Decision Tree

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     STRATEGY DECISION TREE                                 │
│                    select_nlsq_strategy()                                  │
│                                                                            │
│                      peak_memory_gb                                        │
│                           │                                                │
│              ┌────────────┴────────────┐                                   │
│         ≤ threshold              > threshold                               │
│              │                         │                                   │
│              ▼                         ▼                                   │
│        n_points?              index_memory > threshold?                    │
│         │     │                    │           │                           │
│       <1M    ≥1M                  YES         NO                           │
│         │     │                    │           │                           │
│         ▼     ▼                    ▼           ▼                           │
│    ┌────────┐ ┌────────────┐  ┌─────────────┐ ┌────────────┐               │
│    │STANDARD│ │OUT_OF_CORE │  │   HYBRID    │ │OUT_OF_CORE │               │
│    └────────┘ └────────────┘  │  STREAMING  │ └────────────┘               │
│                               └─────────────┘                              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Strategy to Executor Mapping

```
NLSQStrategy Enum → Executor Mapping:
├─ STANDARD         → StandardExecutor      → curve_fit()
├─ OUT_OF_CORE      → LargeDatasetExecutor  → curve_fit_large()
└─ HYBRID_STREAMING → StreamingExecutor     → AdaptiveHybridStreamingOptimizer
```

---

## 4. Stratification Decision

**File:** `strategies/chunking.py`

### Decision Logic

```
┌───────────────────────────────────────────────────────────────────────────┐
│ should_use_stratification() Decision Tree                                  │
│                                                                            │
│   Dataset < 100k points?     ──YES──▶ NO  (STANDARD, no chunking)         │
│          │                                                                 │
│         NO                                                                 │
│          ▼                                                                 │
│   No per-angle scaling?      ──YES──▶ NO  (regular chunking works)        │
│          │                                                                 │
│         NO                                                                 │
│          ▼                                                                 │
│   Single phi angle?          ──YES──▶ NO  (no stratification needed)      │
│          │                                                                 │
│         NO                                                                 │
│          ▼                                                                 │
│   Imbalance ratio > 5.0?     ──YES──▶ NO  (use sequential per-angle)      │
│          │                                                                 │
│         NO                                                                 │
│          ▼                                                                 │
│        YES ──▶ Use angle-stratified chunking                              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Angle-Stratified Chunking

**Problem Solved:**

NLSQ's default chunking splits data ARBITRARILY without angle awareness:
- Some chunks may have NO points for angle[i]
- Gradient w.r.t. contrast[i] = ZERO
- Optimization fails silently (no convergence)

**Solution: Round-Robin Distribution**

```
1. Group data by phi angle
2. Distribute points from EACH angle across ALL chunks
3. Cyclic stratification ensures ALL data used (Jan 2026 fix)

Example (3 angles, 1.2M points, target 100k chunks):
  Angle A: 600k pts → distributed as 50k to each of 12 chunks
  Angle B: 400k pts → distributed as 33k to each of 12 chunks
  Angle C: 200k pts → distributed as 17k to each of 12 chunks
  ──────────────────────────────────────────────────────────
  Result: 12 chunks of ~100k each, ALL angles in EACH chunk
```

**Key Functions:**
- `create_angle_stratified_data()` - Full copy approach
- `create_angle_stratified_indices()` - Zero-copy approach (~1% memory overhead)

---

## 5. Residual Function Setup

### Comparison

| Feature | StratifiedResidualFunction | StratifiedResidualFunctionJIT |
|---------|---------------------------|------------------------------|
| File | `strategies/residual.py` | `strategies/residual_jit.py` |
| Callable | Python | JIT-compiled |
| Shapes | Dynamic | Static (padded) |
| Parallelization | Vectorized indexing | vmap over chunks |
| Memory | Lower | Slightly higher (padding) |
| Speed | 15-20% speedup | 2-3× after JIT compile |

### Residual Computation Flow

```
1. Compute g2_theory on global (phi, t1, t2) grid ONCE (not per-chunk)
2. Extract values via pre-computed flat indices (vectorized)
3. residual = (g2_data - g2_theory) / sigma
4. Mask diagonals: jnp.where(t1_idx != t2_idx, residual, 0.0)
5. Return flattened residuals for least-squares optimizer

Per-angle parameter handling:
  params = [c0, c1, ..., c_n, o0, o1, ..., o_n, *physical]
  Each angle uses its corresponding contrast[i], offset[i]
```

### Optimizations

**StratifiedResidualFunction:**
- Pre-compute global unique (phi, t1, t2) at init
- Pre-compute flat indices (avoid searchsorted in loop)
- Vectorized indexing (15-20% per-iter speedup)

**StratifiedResidualFunctionJIT:**
- Padded arrays to max_chunk_size (static shapes for JIT)
- vmap parallelization over chunks
- Buffer donation (`donate_argnums`) reduces peak memory
- Boolean mask for real vs padded data

---

## 6. Anti-Degeneracy Defense System

**Activation Conditions (ALL must be true):**
- `analysis_mode = laminar_flow`
- `n_phi > 3` (many angles where absorption is problematic)
- `per_angle_scaling = True`

### Root Problem: Structural Degeneracy

```
With 23 angles: 46 per-angle params vs 7 physical params
Per-angle params have larger, more consistent gradients
→ Optimizer "absorbs" angle-dependent shear signal into per-angle params
→ Physical params (especially γ̇₀) collapse to bounds

Additional: Shear gradient ∂L/∂γ̇₀ ∝ Σcos(φ₀-φ) suffers 94.6% cancellation
(11 positive + 12 negative cos contributions with 23 angles)
```

### Layer 1: Fourier Reparameterization

**File:** `fourier_reparam.py`

```
Problem: Too many per-angle degrees of freedom

Solution: Replace n_phi independent params with truncated Fourier series
  contrast(φ) = c₀ + Σₖ[cₖ×cos(kφ) + sₖ×sin(kφ)]  for k=1..order
  offset(φ)   = o₀ + Σₖ[oₖ×cos(kφ) + tₖ×sin(kφ)]

Parameter Reduction (order=2, 5 coeffs per group):
  n_phi=10:   20 → 10 params (50% reduction)
  n_phi=23:   46 → 10 params (78% reduction)
  n_phi=100: 200 → 10 params (95% reduction)

Config: per_angle_mode="auto", fourier_order=2, fourier_auto_threshold=6
```

### Layer 2: Hierarchical Optimization

**File:** `hierarchical.py`

```
Problem: Physical and per-angle params compete; gradients cancel

Solution: Alternating two-stage optimization

  for outer_iteration in range(max_outer):     # default 5
    Stage 1: Freeze per-angle → optimize physical (L-BFGS, 100 iters)
    Stage 2: Freeze physical → optimize per-angle (L-BFGS, 50 iters)
    Check convergence on physical params (tol=1e-6)

Why it works: Physical params can't be absorbed when per-angle frozen

Config: hierarchical.enable=true, max_outer_iterations=5
```

### Layer 3: Adaptive CV Regularization

**File:** `adaptive_regularization.py`

```
Problem: Absolute variance regularization (λ=0.01) was too weak
         Contributed only ~0.05% to total loss

Solution: CV-based (Coefficient of Variation) relative regularization

  CV = std(params) / |mean(params)|
  L_reg = λ × CV² × MSE × n_points

  Auto-tunes λ to contribute ~10% to loss (100× stronger than v2.8)
  λ = target_contribution / target_cv²

Config: mode="relative", lambda=1.0, target_cv=0.10, target_contrib=0.10
```

### Layer 4: Gradient Collapse Monitor

**File:** `gradient_monitor.py`

```
Problem: Need runtime detection of physical param gradient loss

Solution: Monitor gradient ratio with automatic response

  ratio = ‖∇_physical‖ / ‖∇_per_angle‖

  if ratio < threshold for N consecutive iterations:
    → trigger response action

Response Actions:
  "warn"        → Log warning only
  "hierarchical"→ Switch to hierarchical mode (recommended)
  "reset"       → Reset per-angle params to mean
  "abort"       → Abort optimization

Config: enable=true, ratio_threshold=0.01, consecutive_triggers=5
```

### Layer 5: Shear-Sensitivity Weighting

**File:** `shear_weighting.py`

```
Problem: Shear gradient ∂L/∂γ̇₀ ∝ Σcos(φ₀-φ) suffers 94.6% cancellation
         With 23 angles spanning 360°: sum≈0, gradient≈0

Solution: Angle-dependent residual weighting

  L = Σ_φ w(φ) × Σ_τ (g2_model - g2_exp)²
  w(φ) = w_min + (1 - w_min) × |cos(φ₀ - φ)|^α

Effect: Emphasizes shear-sensitive angles (where cos is large)
        Prevents gradient cancellation across angle sum

Config: enable=true, min_weight=0.3, alpha=1.0, normalize=true
```

---

## 7. Strategy Execution

**File:** `strategies/executors.py`

### STANDARD Strategy (StandardExecutor)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        STANDARD STRATEGY                                   ║
║                       (StandardExecutor)                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Uses: NLSQ curve_fit()                                                   ║
║   Best for: < 1M points with sufficient RAM                                ║
║                                                                            ║
║   Memory Pattern: Full Jacobian J (n_points × n_params) in RAM             ║
║                                                                            ║
║   Algorithm:                                                               ║
║     1. Compute residuals r(p) for current params p                         ║
║     2. Compute full Jacobian J = ∂r/∂p via JAX autodiff                    ║
║     3. Solve (J^T J + λI) Δp = -J^T r  (trust-region)                      ║
║     4. Update p ← p + Δp                                                   ║
║     5. Repeat until convergence                                            ║
║                                                                            ║
║   No chunking, no progress bar, fastest for small datasets                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### OUT_OF_CORE Strategy (LargeDatasetExecutor)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                       OUT_OF_CORE STRATEGY                                 ║
║                     (LargeDatasetExecutor)                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Uses: NLSQ curve_fit_large()                                             ║
║   Best for: 1M - 100M points                                               ║
║                                                                            ║
║   Memory Pattern: Chunked J^T J accumulation (never full J in RAM)         ║
║                   Uses angle-stratified chunking if per_angle_scaling      ║
║                                                                            ║
║   Algorithm:                                                               ║
║     for each iteration:                                                    ║
║       J^T_J = 0, J^T_r = 0                                                 ║
║       for each chunk (with progress bar):                                  ║
║         J_chunk = autodiff(residual_fn, chunk_data)  # small J             ║
║         r_chunk = residual_fn(chunk_data)                                  ║
║         J^T_J += J_chunk^T @ J_chunk                 # accumulate          ║
║         J^T_r += J_chunk^T @ r_chunk                 # accumulate          ║
║       solve: (J^T_J + λI) Δp = -J^T_r                                      ║
║       trust-region update                                                  ║
║                                                                            ║
║   Key insight: J^T J is only (n_params × n_params) ≈ small                 ║
║                Never need full J (n_points × n_params) in RAM              ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### HYBRID_STREAMING Strategy (StreamingExecutor)

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     HYBRID_STREAMING STRATEGY                              ║
║                      (StreamingExecutor)                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Uses: NLSQ AdaptiveHybridStreamingOptimizer                              ║
║   Best for: 100M+ points, memory-constrained, many phi angles              ║
║                                                                            ║
║   Memory Pattern: Bounded ~2 GB regardless of dataset size                 ║
║                                                                            ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ PHASE 0: PARAMETER NORMALIZATION                                    │ ║
║   │                                                                     │ ║
║   │   Problem: Gradient magnitudes differ wildly (D₀~10⁴ vs γ̇₀~10⁻³)   │ ║
║   │   Solution: Bounds-based normalization to [0, 1]                    │ ║
║   │     p_norm = (p - lower) / (upper - lower)                          │ ║
║   │                                                                     │ ║
║   │   Equalizes gradient magnitudes for balanced optimization           │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ PHASE 1: L-BFGS WARMUP (100-500 iterations)                         │ ║
║   │                                                                     │ ║
║   │   • Fast initial exploration of parameter space                     │ ║
║   │   • Gradient-only (no Jacobian storage required)                    │ ║
║   │   • Memory: O(n_params²) for inverse Hessian approx                 │ ║
║   │   • Adaptive switching to Phase 2 when progress stalls              │ ║
║   │                                                                     │ ║
║   │   If shear_weighting enabled:                                       │ ║
║   │     Creates HierarchicalOptimizer for weighted L-BFGS               │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ PHASE 2: STREAMING GAUSS-NEWTON (50 iterations)                     │ ║
║   │                                                                     │ ║
║   │   • Quadratic convergence near minimum                              │ ║
║   │   • Exact J^T J accumulation via streaming (50k points/chunk)       │ ║
║   │   • Trust-region refinement for robustness                          │ ║
║   │                                                                     │ ║
║   │   for each chunk:                                                   │ ║
║   │     stream chunk to device                                          │ ║
║   │     compute J_chunk, r_chunk                                        │ ║
║   │     accumulate J^T J, J^T r                                         │ ║
║   │     free chunk memory                                               │ ║
║   │   solve normal equations                                            │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                       ║
║                                    ▼                                       ║
║   ┌─────────────────────────────────────────────────────────────────────┐ ║
║   │ PHASE 3: DENORMALIZATION + COVARIANCE                               │ ║
║   │                                                                     │ ║
║   │   • Transform params back to physical scales:                       │ ║
║   │       p = p_norm × (upper - lower) + lower                          │ ║
║   │                                                                     │ ║
║   │   • Transform covariance to original parameter space:               │ ║
║   │       Cov_orig = S @ Cov_norm @ S^T                                 │ ║
║   │       where S = diag(upper - lower)                                 │ ║
║   │                                                                     │ ║
║   │   • Proper uncertainties in physical units                          │ ║
║   └─────────────────────────────────────────────────────────────────────┘ ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 8. Error Recovery

**File:** `wrapper.py`

### 3-Attempt Recovery System

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 3-Attempt Recovery System                                                  │
│                                                                            │
│   Attempt 1: Primary optimization                                          │
│       │                                                                    │
│       └─ Failure? (bounds violation, NaN, convergence failure)             │
│              │                                                             │
│              ▼                                                             │
│   Attempt 2: Reset params to bounds center                                 │
│       │      p_reset = (lower + upper) / 2                                 │
│       │      Log recovery action                                           │
│       │                                                                    │
│       └─ Failure? (numerical instability)                                  │
│              │                                                             │
│              ▼                                                             │
│   Attempt 3: Reset to geometric mean                                       │
│       │      p_reset = sqrt(lower × upper)                                 │
│       │      Comprehensive logging                                         │
│       │                                                                    │
│       └─ Return result with recovery_actions in device_info                │
│                                                                            │
│   Recovery actions logged: ["bounds_reset", "geometric_mean_reset", ...]  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Result Building

**File:** `result_builder.py`

### OptimizationResult Structure

```python
@dataclass
class OptimizationResult:
    # Core results
    parameters: np.ndarray        # Optimized parameter values
    uncertainties: np.ndarray     # sqrt(diag(covariance))
    covariance: np.ndarray        # Full covariance matrix
    chi_squared: float            # Sum of squared residuals
    reduced_chi_squared: float    # χ² / (n_points - n_params)

    # Status
    success: bool                 # Optimization succeeded
    message: str                  # Detailed status message
    iterations: int               # Number of iterations
    execution_time: float         # Wall clock time (seconds)
    convergence_status: str       # "converged"|"partial"|"failed"
    quality_flag: str             # "good"|"marginal"|"poor"

    # Metadata
    device_info: dict
        adapter: str              # "NLSQAdapter"|"NLSQWrapper"
        fallback_occurred: bool   # True if adapter→wrapper fallback
        strategy: str             # "standard"|"out_of_core"|"hybrid"
        recovery_actions: list    # List of recovery attempts
        peak_memory_gb: float     # Estimated peak memory
        n_chunks: int             # Number of chunks (if chunked)
```

### Quality Flag Determination

```
├─ "good":     reduced_χ² < 2.0 AND converged
├─ "marginal": reduced_χ² < 5.0 OR partial convergence
└─ "poor":     reduced_χ² ≥ 5.0 OR failed
```

---

## Quick Reference Tables

### Strategy Selection

| Dataset Size | Peak Memory | Strategy | Executor | NLSQ Function | Memory Pattern |
|--------------|-------------|----------|----------|---------------|----------------|
| < 1M pts | < threshold | **STANDARD** | StandardExecutor | `curve_fit()` | Full J in RAM |
| 1M - 100M | < threshold | **OUT_OF_CORE** | LargeDatasetExecutor | `curve_fit_large()` | Chunked J^T J |
| Any | > threshold | **HYBRID_STREAMING** | StreamingExecutor | `AdaptiveHybridStreaming` | Bounded ~2 GB |

### Mode-Specific Parameters

| Mode | Physical Params | With 23-angle scaling | Anti-degeneracy |
|------|----------------|----------------------|-----------------|
| static_isotropic | 3: D₀, α, D_offset | 3 + 46 = 49 total | No |
| laminar_flow | 7: + γ̇₀, β, γ̇_offset, φ₀ | 7 + 46 = 53 total | **Yes (5 layers)** |

### Jacobian Overhead Factor Evolution

| Version | Factor | Justification |
|---------|--------|---------------|
| v2.7.0 | 3.0× | Initial estimate |
| v2.11.0 | 4.0× | Added JAX overhead |
| v2.13.0 | 5.0× | Padding + stratification |
| v2.14.0 | **6.5×** | Empirical validation on 23M dataset |

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `core.py` | Entry points: `fit_nlsq_jax()`, `fit_nlsq_multistart()` |
| `memory.py` | `select_nlsq_strategy()`, memory estimation (6.5× factor) |
| `adapter.py` | NLSQAdapter with model caching |
| `wrapper.py` | NLSQWrapper with full feature set + 3-attempt recovery |
| `strategies/chunking.py` | `create_angle_stratified_data()`, `should_use_stratification()` |
| `strategies/residual.py` | `StratifiedResidualFunction` (Python-callable) |
| `strategies/residual_jit.py` | `StratifiedResidualFunctionJIT` (JIT-compiled) |
| `strategies/executors.py` | StandardExecutor, LargeDatasetExecutor, StreamingExecutor |
| `anti_degeneracy_controller.py` | 5-layer defense orchestration |
| `anti_degeneracy_layer.py` | Abstract layer interface |
| `fourier_reparam.py` | Layer 1: Fourier parameter compression |
| `hierarchical.py` | Layer 2: Alternating physical/per-angle stages |
| `adaptive_regularization.py` | Layer 3: CV-based regularization |
| `gradient_monitor.py` | Layer 4: Runtime collapse detection |
| `shear_weighting.py` | Layer 5: Angle-dependent loss weighting |
| `multistart.py` | Multi-start optimization with LHS |
| `result_builder.py` | Result construction and quality assessment |
