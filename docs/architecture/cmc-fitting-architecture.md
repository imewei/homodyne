# Homodyne CMC (Consensus Monte Carlo) Fitting Architecture

Complete documentation of the CMC (Consensus Monte Carlo) fitting system in homodyne.

**Version:** 2.20.0
**Last Updated:** January 2026

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Entry Point & Orchestration](#1-entry-point--orchestration)
3. [Data Preparation & Sharding](#2-data-preparation--sharding)
4. [Auto Shard Size Selection](#3-auto-shard-size-selection)
5. [Time Grid Construction](#4-time-grid-construction)
6. [Physics Model](#5-physics-model)
7. [Gradient Balancing (Z-Space)](#6-gradient-balancing-z-space)
8. [NUTS Sampling](#7-nuts-sampling)
9. [Backend Execution](#8-backend-execution)
10. [Sample Combination](#9-sample-combination)
11. [Result Creation & Diagnostics](#10-result-creation--diagnostics)
12. [Complete Data Flow](#complete-data-flow)
13. [Quick Reference Tables](#quick-reference-tables)
14. [Key Files Reference](#key-files-reference)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER ENTRY POINTS                                   │
│                                                                                  │
│    CLI: homodyne --method cmc              API: fit_mcmc_jax(data, config)      │
│              │                                         │                         │
│              ▼                                         │                         │
│    ┌────────────────────────┐                          │                         │
│    │ AUTOMATIC NLSQ WARMUP  │ (v2.20.0)                │                         │
│    │ fit_nlsq_jax() first   │◄────────────────────────┤ (optional nlsq_result)  │
│    │ unless --no-nlsq-...   │                          │                         │
│    └────────────────────────┘                          │                         │
│              │                                         │                         │
│              └──────────────────┬──────────────────────┘                         │
│                                 │                                                │
│                                 ▼                                                │
│                     fit_mcmc_jax(data, config, nlsq_result=...)                  │
│                                  (core.py)                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        1. DATA PREPARATION                                       │
│                          (data_prep.py)                                          │
│                                                                                  │
│              Validation → Diagonal Filtering → Noise Estimation                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     2. SHARDING DECISION                                         │
│                                                                                  │
│           Auto shard size → Stratified or Random sharding                        │
│                                                                                  │
│              Single Shard ◄──────────────► Multiple Shards                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      3. BACKEND SELECTION                                        │
│                                                                                  │
│          MultiprocessingBackend │ PjitBackend │ PBSBackend                       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    4. PER-SHARD NUTS SAMPLING                                    │
│                         (sampler.py)                                             │
│                                                                                  │
│     Z-Space Transform → Preflight → NUTS (dense_mass) → Sample Extraction       │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     5. SAMPLE COMBINATION                                        │
│                       (backends/base.py)                                         │
│                                                                                  │
│        Hierarchical Combination → Consensus MC → Combined Posterior              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     6. RESULT & DIAGNOSTICS                                      │
│                         (results.py)                                             │
│                                                                                  │
│              R-hat │ ESS │ Divergences → CMCResult + ArviZ                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Entry Point & Orchestration

**File:** `core.py`

### fit_mcmc_jax() Signature

```python
def fit_mcmc_jax(
    data: np.ndarray,           # Pooled C2 correlation (n_total,)
    t1, t2: np.ndarray,         # Time coordinates
    phi: np.ndarray,            # Phi angles
    q, L: float,                # Physics parameters
    analysis_mode: str,         # "static" or "laminar_flow"
    method: str = "mcmc",
    cmc_config: dict | None = None,
    initial_values: dict | None = None,
    parameter_space: ParameterSpace = None,
    dt: float | None = None,
    output_dir: Path | str | None = None,
    progress_bar: bool = True,
    run_id: str | None = None,
) -> CMCResult
```

### Orchestration Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│ fit_mcmc_jax() Orchestration                                               │
│                                                                            │
│  1. Normalize analysis mode, generate run_id                               │
│  2. Validate and prepare pooled data                                       │
│     └─ prepare_mcmc_data() → filter diagonals, estimate noise             │
│  3. Determine shard size via _resolve_max_points_per_shard()               │
│  4. Construct time_grid with proper dt spacing                             │
│  5. Shard data if needed:                                                  │
│     ├─ shard_data_stratified() for multiple phi angles                    │
│     └─ shard_data_random() for single phi angle                           │
│  6. Select backend: multiprocessing (default), pjit, or pbs               │
│  7. Execute:                                                               │
│     ├─ If shards: backend.run() → parallel NUTS → combine                 │
│     └─ If single: run_nuts_sampling() directly                            │
│  8. Create CMCResult with diagnostics                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Preparation & Sharding

**File:** `data_prep.py`

### Data Validation (prepare_mcmc_data)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ prepare_mcmc_data()                                                        │
│                                                                            │
│  • Ensure pooled arrays are 1D with matching lengths                       │
│  • Check for NaN/Inf values                                                │
│  • Filter diagonal points (t1 == t2) - autocorrelation artifacts (v2.14.2)│
│  • Extract unique phi angles and create index mapping                      │
│  • Estimate noise using robust MAD (Median Absolute Deviation)             │
│  • Return PreparedData dataclass                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Sharding Strategies

```
┌───────────────────────────────────────────────────────────────────────────┐
│ STRATIFIED SHARDING (shard_data_stratified)                                │
│ Used when: Multiple phi angles exist (n_phi > 1)                           │
│                                                                            │
│ Algorithm:                                                                 │
│   1. Group data by unique phi angle (stratification)                       │
│   2. For each angle with n_points > max_points_per_shard:                  │
│      • Split into multiple shards                                          │
│      • Randomly shuffle but preserve temporal order within shard           │
│      • Cap at max_shards_per_angle=100                                     │
│   3. Each shard is complete PreparedData with its own phi info             │
│                                                                            │
│ Key property: Preserves stratification for balanced angle coverage         │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ RANDOM SHARDING (shard_data_random)                                        │
│ Used when: Single phi angle but dataset > threshold                        │
│                                                                            │
│ Algorithm:                                                                 │
│   1. Shuffle all point indices randomly                                    │
│   2. Split into num_shards equal parts (ALL data used, none dropped)       │
│   3. Sort within shard to preserve some temporal structure                 │
│   4. Cap at max_shards=100 (increases shard size if needed)                │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Auto Shard Size Selection

**Function:** `_resolve_max_points_per_shard()` in core.py

### Critical Design Principles (v2.20.0)

**Minimum Shard Size Enforcement:**
- **laminar_flow**: 10,000 points minimum (prevents data-starved shards)
- **static**: 5,000 points minimum

**Dynamic max_shards Scaling:**
```
┌───────────────────────────────────────────────────────────────────────────┐
│ max_shards by Dataset Size (v2.20.0)                                       │
│                                                                            │
│   Dataset Size    │ max_shards │ Rationale                                 │
│   ────────────────┼────────────┼──────────────────────────────────────────  │
│   < 10M points    │ 2,000      │ Standard parallel workload                │
│   10M - 100M      │ 10,000     │ Balanced shard count                      │
│   100M - 1B       │ 50,000     │ High parallelism for large datasets       │
│   1B+             │ 100,000    │ Extreme scale support                     │
└───────────────────────────────────────────────────────────────────────────┘
```

### Angle-Aware Scaling

The shard size is adjusted based on the number of phi angles to ensure each shard has sufficient data per angle:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Angle Factor by n_phi                                                      │
│                                                                            │
│   n_phi  │ angle_factor │ Effect on Shard Size                            │
│   ───────┼──────────────┼─────────────────────────────────────────────────  │
│   ≤ 3    │ 0.6          │ 40% reduction (ensures coverage per angle)      │
│   4-5    │ 0.7          │ 30% reduction                                   │
│   6-10   │ 0.85         │ 15% reduction                                   │
│   > 10   │ 1.0          │ No reduction (many angles spread data)          │
└───────────────────────────────────────────────────────────────────────────┘
```

### Decision Logic by Mode

```
┌───────────────────────────────────────────────────────────────────────────┐
│ LAMINAR FLOW MODE (7 parameters, complex gradients)                        │
│                                                                            │
│   Dataset Size    │ Base Size │ After n_phi≤3 │ Est. Shards │ Per-Shard   │
│   ────────────────┼───────────┼───────────────┼─────────────┼─────────────  │
│   < 2M points     │ 20K       │ 12K           │ ~150        │ ~10-15 min  │
│   2M - 10M        │ 17K       │ 10K           │ 200-1000    │ ~5-8 min    │
│   10M - 100M      │ 25K       │ 15K           │ 700-6500    │ ~5 min      │
│   100M - 1B       │ 30K       │ 18K           │ 5K-55K      │ ~4 min      │
│   1B+             │ 35K       │ 21K           │ 47K+        │ ~3 min      │
│                                                                            │
│   MINIMUM ENFORCED: 10,000 points per shard (prevents data starvation)    │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ STATIC MODE (3 parameters, simple gradients - 10× larger shards OK)       │
│                                                                            │
│   Dataset Size    │ max_points_per_shard │ Est. Shards                     │
│   ────────────────┼──────────────────────┼──────────────                   │
│   < 50M points    │ 100K                 │ ~500                            │
│   50M - 100M      │ 80K                  │ ~1K                             │
│   100M+           │ 50K                  │ ~2K+                            │
│                                                                            │
│   MINIMUM ENFORCED: 5,000 points per shard                                │
└───────────────────────────────────────────────────────────────────────────┘
```

### Memory Capping

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Memory Capping Logic                                                       │
│                                                                            │
│   Per-shard result size: ~100KB                                            │
│     (13 params × 2 chains × 1500 samples × 8 bytes)                        │
│                                                                            │
│   Dynamic max_shards by dataset size (see table above)                     │
│                                                                            │
│   If shard count would exceed max_shards:                                  │
│     • Increases shard size (no subsampling, all data used)                 │
│     • For laminar_flow: caps adjusted shard size at 50K max                │
│     • Enforces minimum shard size (10K laminar_flow, 5K static)            │
│                                                                            │
│   Platform Scaling (based on dynamic max_shards):                          │
│   ├─ 3M dataset:    ~150-300 shards (manageable on personal systems)      │
│   ├─ 100M dataset:  ~5K-10K shards (requires cluster)                     │
│   └─ 1B dataset:    ~50K shards (extreme scale, HPC required)             │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Time Grid Construction

**Critical Fix (December 2025)**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Time Grid Construction                                                     │
│                                                                            │
│ Problem: Previously used np.unique(t1, t2) which gave incorrect grid      │
│          density when data is subsampled                                   │
│                                                                            │
│ Solution: Construct time_grid explicitly with config dt spacing:           │
│                                                                            │
│   dt_used = dt if dt is not None else inferred_dt                          │
│   t_max = max(t1_pooled.max(), t2_pooled.max())                            │
│   n_time_points = int(round(t_max / dt_used)) + 1                          │
│   time_grid = np.linspace(0.0, t_max, n_time_points)                       │
│                                                                            │
│ This ensures physics integration (trapezoidal cumsum) uses correct         │
│ grid density matching NLSQ                                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Physics Model

**File:** `model.py`

### Three Model Variants (v2.18.0)

| Model | Purpose | Per-Angle Mode |
|-------|---------|----------------|
| `xpcs_model()` | Original model with standard parameterization | Legacy |
| `xpcs_model_scaled()` | Gradient-balanced model with z-space sampling | individual |
| `xpcs_model_constant()` | Fixed per-angle scaling (not sampled) | auto/constant |

### Per-Angle Mode Selection (v2.18.0)

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Per-Angle Mode Decision (get_effective_per_angle_mode)                    │
│                                                                           │
│   per_angle_mode = "auto" (default)?                                      │
│        │                                                                  │
│        ├─ YES: Check n_phi >= constant_scaling_threshold (default: 3)    │
│        │       ├─ n_phi >= 3: Use CONSTANT → xpcs_model_constant()       │
│        │       │   • Quantile estimation → AVERAGE → broadcast           │
│        │       │   • 8 sampled params (7 physical + 1 sigma)             │
│        │       │                                                         │
│        │       └─ n_phi < 3: Use INDIVIDUAL → xpcs_model_scaled()        │
│        │           • Sample per-angle contrast/offset                    │
│        │           • 8 + 2×n_phi sampled params                          │
│        │                                                                  │
│        └─ NO: Check explicit mode                                        │
│             ├─ "constant": Use xpcs_model_constant()                     │
│             │   • Quantile estimation → use directly (not averaged)      │
│             │   • 8 sampled params (7 physical + 1 sigma)                │
│             │                                                            │
│             └─ "individual": Use xpcs_model_scaled()                     │
│                 • Sample per-angle contrast/offset                       │
│                 • 8 + 2×n_phi sampled params                             │
└───────────────────────────────────────────────────────────────────────────┘
```

**Key Distinction: Auto vs Constant Mode**

| Mode | Quantile Estimation | Fixed Values Used |
|------|---------------------|-------------------|
| `auto` (n_phi ≥ 3) | Estimate 23 per-angle → **AVERAGE** → single value | Same value for all angles (NLSQ parity) |
| `constant` | Estimate 23 per-angle → use **DIRECTLY** | Different value per angle |

### xpcs_model_scaled() Structure

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Sampling Order (CRITICAL for init_to_value to work)                        │
│                                                                            │
│  1. PER-ANGLE CONTRAST PARAMETERS (FIRST)                                  │
│     for i in range(n_phi):                                                 │
│         contrast_i = sample_scaled_parameter(f"contrast_{i}", scaling)     │
│                                                                            │
│  2. PER-ANGLE OFFSET PARAMETERS (SECOND)                                   │
│     for i in range(n_phi):                                                 │
│         offset_i = sample_scaled_parameter(f"offset_{i}", scaling)         │
│                                                                            │
│  3. PHYSICAL PARAMETERS (THIRD)                                            │
│     Static:       D0, alpha, D_offset                                      │
│     Laminar flow: + gamma_dot_t0, beta, gamma_dot_t_offset, phi0           │
│                                                                            │
│  4. NOISE PARAMETER (FOURTH)                                               │
│     sigma ~ HalfNormal(scale=noise_scale × 3.0)                            │
└───────────────────────────────────────────────────────────────────────────┘
```

### xpcs_model_constant() Structure (v2.18.0)

The constant model takes **fixed** contrast/offset arrays instead of sampling them:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ xpcs_model_constant() - Fixed Per-Angle Scaling                           │
│                                                                           │
│  REQUIRED INPUTS (not sampled):                                          │
│    fixed_contrast: jnp.ndarray (n_phi,) - Pre-computed contrast values   │
│    fixed_offset:   jnp.ndarray (n_phi,) - Pre-computed offset values     │
│                                                                           │
│  SAMPLING ORDER (8 parameters total):                                    │
│                                                                           │
│  1. PHYSICAL PARAMETERS (FIRST)                                          │
│     Static:       D0, alpha, D_offset                                    │
│     Laminar flow: + gamma_dot_t0, beta, gamma_dot_t_offset, phi0         │
│                                                                           │
│  2. NOISE PARAMETER (SECOND)                                             │
│     sigma ~ HalfNormal(scale=noise_scale × 3.0)                          │
│                                                                           │
│  NO per-angle contrast/offset sampling - they are FIXED from quantiles   │
└───────────────────────────────────────────────────────────────────────────┘
```

**Quantile-Based Scaling Estimation (core.py)**:

```python
from homodyne.core.scaling_utils import estimate_per_angle_scaling

# Estimate per-angle contrast/offset from raw C2 data
estimates = estimate_per_angle_scaling(
    c2_data, t1, t2, phi_indices, n_phi,
    contrast_bounds, offset_bounds
)
# Returns: {"contrast_0": 0.4, "offset_0": 0.95, ...}

# For AUTO mode: average the estimates
if config.per_angle_mode == "auto":
    contrast_avg = np.mean([estimates[f"contrast_{i}"] for i in range(n_phi)])
    offset_avg = np.mean([estimates[f"offset_{i}"] for i in range(n_phi)])
    fixed_contrast = np.full(n_phi, contrast_avg)  # Same for all angles
    fixed_offset = np.full(n_phi, offset_avg)
else:
    # CONSTANT mode: use per-angle estimates directly
    fixed_contrast = np.array([estimates[f"contrast_{i}"] for i in range(n_phi)])
    fixed_offset = np.array([estimates[f"offset_{i}"] for i in range(n_phi)])
```

### Physics Computation

```python
# Compute g1 using exact same physics as NLSQ
g1_all_phi = compute_g1_total(params, t1, t2, phi_unique, q, L, dt, time_grid)
# g1_all_phi shape: (n_phi, n_points)

# Map per-point g1 using phi indices
g1_per_point = g1_all_phi[phi_indices, point_idx]

# Apply per-angle scaling
c2_theory = contrast[phi_idx] × g1² + offset[phi_idx]
```

### Likelihood

```python
sigma = numpyro.sample("sigma", dist.HalfNormal(scale=sigma_scale))
numpyro.sample("obs", dist.Normal(c2_theory, sigma), obs=data)
```

---

## 6. Gradient Balancing (Z-Space)

**File:** `scaling.py`

### The Problem

```
Parameters span vastly different magnitudes:
  D0:           ~10⁴  (diffusion)
  alpha:        ~10⁰  (exponent)
  gamma_dot_t0: ~10⁻³ (shear)
  contrast:     ~10⁻¹ (optical)

This causes 10⁶:1 gradient imbalance → 0% NUTS acceptance rate
```

### The Solution: Non-Centered Reparameterization

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Z-Space Transformation                                                     │
│                                                                            │
│   Sample in z-space:     z ~ Normal(0, 1)                                  │
│   Transform to original: P = center + scale × z                            │
│   Apply smooth bounds:   P = smooth_bound(P, low, high) using tanh         │
│                                                                            │
│ Smooth bounding (avoids hard clipping artifacts):                          │
│   smooth_bound(x) = mid + (half × tanh((x - mid) / half))                  │
│                                                                            │
│ ParameterScaling dataclass:                                                │
│   name: str           # Parameter name                                     │
│   center: float       # Midpoint of bounds or prior mean                   │
│   scale: float        # (high - low) / 4 or prior std                      │
│   low: float          # Lower bound                                        │
│   high: float         # Upper bound                                        │
│                                                                            │
│ Key methods:                                                               │
│   to_normalized(value): Original space → z-space (for initialization)     │
│   to_original(z_value): z-space → original space (in model)               │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 7. NUTS Sampling

**File:** `sampler.py`

### run_nuts_sampling() Workflow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         NUTS SAMPLING WORKFLOW                                   │
│                                                                                  │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 1. PARAMETER ORDER RESOLUTION                                              │ │
│   │    get_param_names_in_order(n_phi, analysis_mode)                          │ │
│   │    → [contrast_0, ..., offset_0, ..., D0, alpha, ...]                      │ │
│   │    + "sigma" as final parameter                                            │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 2. MCMC-SAFE D0 CHECK (_compute_mcmc_safe_d0)                              │ │
│   │    • Detects if initial D0 causes g1 → 0 everywhere (vanishing gradients) │ │
│   │    • Computes scaled D0 that produces g1 ≈ 0.5 at typical time lag        │ │
│   │    • Ensures gradients are alive for NUTS exploration                      │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 3. BUILD FULL INITIAL VALUES                                               │ │
│   │    build_init_values_dict() from priors.py                                 │ │
│   │    • Data-driven contrast/offset estimation                                │ │
│   │    • Combine with config initial_values                                    │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 4. GRADIENT BALANCING TRANSFORMATION                                       │ │
│   │    scalings = compute_scaling_factors(parameter_space, n_phi, mode)        │ │
│   │    z_space_init = transform_initial_values_to_z(full_init, scalings)       │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 5. PREFLIGHT VALIDATION                                                    │ │
│   │    _preflight_log_density(model, params, ...)                              │ │
│   │    • Catches non-finite log density before expensive sampling              │ │
│   │    • Validates model can compute gradients at init point                   │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 6. NUTS KERNEL CREATION                                                    │ │
│   │                                                                            │ │
│   │    kernel = NUTS(                                                          │ │
│   │        model,                                                              │ │
│   │        init_strategy=init_to_value(values=z_space_init),                   │ │
│   │        target_accept_prob=0.85,                                            │ │
│   │        dense_mass=True  # CRITICAL: Learn cross-correlations               │ │
│   │    )                                                                       │ │
│   │                                                                            │ │
│   │    Why dense_mass=True:                                                    │ │
│   │    Diagonal mass matrix can't adapt per-dimension to handle the           │ │
│   │    10⁶:1 gradient imbalance. Dense matrix learns covariance structure     │ │
│   │    during warmup.                                                          │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 7. MCMC EXECUTION                                                          │ │
│   │                                                                            │ │
│   │    mcmc = MCMC(                                                            │ │
│   │        kernel,                                                             │ │
│   │        num_warmup=500,                                                     │ │
│   │        num_samples=1500,                                                   │ │
│   │        num_chains=4                                                        │ │
│   │    )                                                                       │ │
│   │    mcmc.run(                                                               │ │
│   │        rng_key,                                                            │ │
│   │        extra_fields=("accept_prob", "diverging", "num_steps"),             │ │
│   │        **model_kwargs                                                      │ │
│   │    )                                                                       │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 8. COMPUTATION BLOCKING (Critical for proper timing)                       │ │
│   │                                                                            │ │
│   │    jax.block_until_ready(last_state)                                       │ │
│   │                                                                            │ │
│   │    Without this, lazy evaluation delays computation to device_get(),       │ │
│   │    causing misleading timing measurements                                  │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                             │
│                                    ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────────┐ │
│   │ 9. SAMPLE EXTRACTION                                                       │ │
│   │                                                                            │ │
│   │    • Extract in group_by_chain format: (n_chains, n_samples)               │ │
│   │    • Compute per-shard diagnostics (accept_prob, divergences, step_size)   │ │
│   │    • Return (MCMCSamples, SamplingStats)                                   │ │
│   └───────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Key Dataclasses

```python
@dataclass
class SamplingStats:
    warmup_time: float
    sampling_time: float
    total_time: float
    num_divergent: int
    accept_prob: float
    step_size: float
    step_size_min: float | None
    step_size_max: float | None
    inverse_mass_matrix_summary: str | None
    tree_depth: float

@dataclass
class MCMCSamples:
    samples: dict[str, np.ndarray]    # {name: (n_chains, n_samples)}
    param_names: list[str]
    n_chains: int
    n_samples: int
    extra_fields: dict[str, Any]      # diverging, accept_prob, etc.
    num_shards: int = 1               # For correct divergence rate in CMC
```

---

## 8. Backend Execution

**File:** `backends/`

### Backend Selection

```python
def select_backend(config: CMCConfig) -> CMCBackend:
    backend_name = config.backend_name

    if backend_name == "auto":
        backend_name = "multiprocessing"  # Default for CPU

    if backend_name == "multiprocessing":
        return MultiprocessingBackend()
    elif backend_name == "pjit":
        return PjitBackend()       # Sequential JAX execution
    elif backend_name == "pbs":
        return PBSBackend()        # HPC cluster execution
```

### MultiprocessingBackend (Primary)

**File:** `backends/multiprocessing.py`

```
┌───────────────────────────────────────────────────────────────────────────┐
│ MultiprocessingBackend Architecture                                        │
│                                                                            │
│  1. Pre-generate all shard PRNG keys in single JAX call                    │
│     (_generate_shard_keys - amortizes JAX compilation)                     │
│                                                                            │
│  2. Create worker pool with adaptive thread limiting                       │
│     physical_cores = psutil.cpu_count(logical=False)                       │
│     threads_per_worker = physical_cores // n_workers                       │
│                                                                            │
│  3. Submit shards to queue-based workers                                   │
│                                                                            │
│  4. Monitor with progress bar and timeout handling                         │
│     • Adaptive polling interval (reduces CPU spinning)                     │
│     • Event.wait() with timeout (efficient heartbeat)                      │
│                                                                            │
│  5. Combine results via combine_shard_samples()                            │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ Worker Function (_run_shard_worker)                                        │
│                                                                            │
│  • Set XLA environment variables for thread safety                         │
│  • Import JAX, deserialize parameter_space                                 │
│  • Call run_nuts_sampling() on shard                                       │
│  • Return serialized MCMCSamples                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Sample Combination

**File:** `backends/base.py`

### Hierarchical Combination

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Hierarchical Combination for Large Shard Counts                            │
│                                                                            │
│   If K > 500 shards:                                                       │
│     1. Combine in chunks of 500                                            │
│     2. Recursively combine chunks                                          │
│     3. Reduces peak memory from O(K) to O(500) × K/500 pattern             │
│     4. Garbage collection between chunks                                   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Combination Methods

```
╔═══════════════════════════════════════════════════════════════════════════╗
║ CONSENSUS_MC (Default, v2.12.0+)                                           ║
║ Implements Scott et al. (2016) correctly                                   ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   For each parameter:                                                      ║
║     1. Compute per-shard mean μ_s and variance σ²_s                        ║
║     2. Combined precision: 1/σ²_combined = Σ_s (1/σ²_s)                    ║
║     3. Combined mean: μ = σ² × Σ_s (μ_s / σ²_s)                            ║
║     4. Generate new samples: N(μ, σ²_combined)                             ║
║                                                                            ║
║   Precision-weighted combination of posterior moments                      ║
║                                                                            ║
║   LIMITATION: Biases toward low-variance shards when heterogeneity exists ║
╚═══════════════════════════════════════════════════════════════════════════╝

╔═══════════════════════════════════════════════════════════════════════════╗
║ ROBUST_CONSENSUS_MC (v2.20.0+)                                             ║
║ Outlier-resistant combination for heterogeneous shards                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║   Algorithm:                                                               ║
║     1. Compute per-shard means and variances                               ║
║     2. Detect outlier shards using MAD (Median Absolute Deviation):        ║
║        median_mean = median(shard_means)                                   ║
║        mad = median(|shard_means - median_mean|)                           ║
║        outlier if |mean - median_mean| > 3 × 1.4826 × mad                  ║
║     3. Exclude outlier shards from combination                             ║
║     4. Apply standard consensus_mc to retained shards                      ║
║                                                                            ║
║   Use when: High per-shard heterogeneity detected (CV > 0.5)              ║
║   Auto-enabled: When heterogeneity_abort=False but CV > threshold         ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌───────────────────────────────────────────────────────────────────────────┐
│ WEIGHTED_GAUSSIAN (Deprecated)                                             │
│   Element-wise weighted averaging (mathematically incorrect)               │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ SIMPLE_AVERAGE (Deprecated)                                                │
│   Element-wise mean across shards                                          │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Result Creation & Diagnostics

**File:** `results.py`

### CMCResult Structure

```python
@dataclass
class CMCResult:
    # Core results
    parameters: np.ndarray                   # Posterior means
    uncertainties: np.ndarray                # Posterior stds
    param_names: list[str]

    # MCMC-specific
    samples: dict[str, np.ndarray]           # {name: (n_chains, n_samples)}
    convergence_status: str                  # "converged", "divergences", "not_converged"
    r_hat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergences: int

    # ArviZ
    inference_data: az.InferenceData

    # Timing
    execution_time: float
    warmup_time: float

    # Config
    n_chains: int = 4
    n_samples: int = 2000
    n_warmup: int = 500
    analysis_mode: str = "static"
    num_shards: int = 1                      # For correct divergence rate
```

### CMCResult.from_mcmc_samples() Workflow

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Result Creation                                                            │
│                                                                            │
│  1. Compute R-hat from samples: per-parameter convergence statistic        │
│  2. Compute ESS (bulk & tail): effective sample size                       │
│  3. Check convergence thresholds:                                          │
│     ├─ R-hat < 1.1 ✓                                                       │
│     ├─ ESS > 100 ✓                                                         │
│     └─ Divergence rate < 5% ✓                                              │
│  4. Aggregate legacy stats (contrast, offset means/stds)                   │
│  5. Create ArviZ InferenceData for plotting                                │
│  6. Return CMCResult                                                       │
│                                                                            │
│  CRITICAL FIX (Dec 2025): Excludes _z (z-space) parameters from            │
│  legacy stats. Scaled model samples contrast_0_z but registers             │
│  contrast_0 as deterministic.                                              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Convergence Status Determination

```
├─ "converged":     All R-hat < 1.1 AND All ESS > 100 AND divergence rate < 5%
├─ "divergences":   Divergence rate ≥ 5%
└─ "not_converged": R-hat ≥ 1.1 OR ESS < 100
```

---

## Complete Data Flow

```
fit_mcmc_jax() [core.py]
│
├─> prepare_mcmc_data() [data_prep.py]
│   ├─ Validate pooled arrays
│   ├─ Filter diagonal points (v2.14.2+)
│   └─ Extract phi info & estimate noise
│
├─> _resolve_max_points_per_shard()
│   ├─ Auto-size based on analysis_mode & dataset size
│   └─ Cap shard count to limit memory
│
├─> Construct time_grid (with proper dt spacing)
│
├─> Sharding decision (CMC needed?)
│   │
│   ├─ YES (large dataset, multiple angles):
│   │   ├─> shard_data_stratified() or shard_data_random()
│   │   └─> Create list[PreparedData] for each shard
│   │
│   └─ NO (small dataset):
│       └─> shards = None (single-shard mode)
│
├─> select_backend(config)
│   └─ MultiprocessingBackend() [or pjit/pbs]
│
├─> If shards:
│   │
│   └─> backend.run()
│       ├─ For each shard (parallel via multiprocessing):
│       │   ├─ Worker: _run_shard_worker()
│       │   │   ├─ Set thread limits (XLA_FLAGS, OMP_NUM_THREADS)
│       │   │   ├─ Reconstruct JAX key from tuple
│       │   │   ├─ Call run_nuts_sampling() → MCMCSamples
│       │   │   └─ Queue result
│       │   └─ Progress bar & timeout monitoring
│       │
│       └─ Combine results:
│           ├─ Hierarchical combination for K > 500 shards
│           └─ combine_shard_samples(shards, method="consensus_mc")
│               ├─ Per-param: combine means & variances
│               ├─ Generate new samples from combined Gaussian
│               └─ Return combined MCMCSamples
│
└─> Single-shard mode:
    └─> run_nuts_sampling(xpcs_model_scaled, ...)
        ├─ MCMC-safe D0 check
        ├─ Build init values (data-driven contrast/offset)
        ├─ Transform to z-space
        ├─ Preflight validation
        ├─ NUTS sampling (dense_mass=True)
        ├─ Extract samples & stats
        └─ Return (MCMCSamples, SamplingStats)
                │
                ▼
CMCResult.from_mcmc_samples()
├─ Compute R-hat, ESS
├─ Check convergence thresholds
├─ Create ArviZ InferenceData
└─ Return CMCResult
```

---

## Quick Reference Tables

### Auto Shard Size Selection (v2.20.0)

#### Laminar Flow Mode (with n_phi ≤ 3, angle_factor = 0.6)

| Dataset Size | Base Size | After Scaling | max_shards | Est. Shards |
|--------------|-----------|---------------|------------|-------------|
| < 2M | 20K | 12K | 2,000 | ~150 |
| 2M - 10M | 17K | 10K | 2,000 | 200-1000 |
| 10M - 100M | 25K | 15K | 10,000 | 700-6500 |
| 100M - 1B | 30K | 18K | 50,000 | 5K-55K |
| 1B+ | 35K | 21K | 100,000 | 47K+ |

**Minimum shard size: 10,000 points** (prevents data-starved shards)

#### Static Mode (10× larger shards)

| Dataset Size | max_points_per_shard | Est. Shards |
|--------------|---------------------|-------------|
| < 50M | 100K | ~500 |
| 50M - 100M | 80K | ~1K |
| 100M+ | 50K | ~2K+ |

**Minimum shard size: 5,000 points**

### Dynamic max_shards by Dataset Size

| Dataset Size | max_shards | Rationale |
|--------------|------------|-----------|
| < 10M points | 2,000 | Standard parallel workload |
| 10M - 100M | 10,000 | Balanced shard count |
| 100M - 1B | 50,000 | High parallelism for large datasets |
| 1B+ | 100,000 | Extreme scale support |

### Mode-Specific Parameters

| Mode | Physical Params | Per-Angle Params (23 angles) | Total |
|------|----------------|------------------------------|-------|
| static | 3: D₀, α, D_offset | 46: contrast + offset | 49 + σ |
| laminar_flow | 7: + γ̇₀, β, γ̇_offset, φ₀ | 46: contrast + offset | 53 + σ |

### CMC Configuration Defaults (v2.20.0)

| Parameter | Default | Description |
|-----------|---------|-------------|
| min_points_for_cmc | 500,000 | Auto-enable threshold |
| sharding_strategy | "stratified" | "stratified" or "random" |
| backend_name | "auto" | → "multiprocessing" |
| num_warmup | 500 | NUTS warmup iterations |
| num_samples | 1500 | NUTS sampling iterations |
| num_chains | 4 | Parallel chains |
| target_accept_prob | 0.85 | NUTS target acceptance |
| max_r_hat | 1.1 | Convergence threshold |
| min_ess | 100.0 | Minimum effective sample size |
| max_divergence_rate | 0.10 | Quality filter threshold |
| require_nlsq_warmstart | False | Enforce NLSQ warm-start |
| combination_method | "consensus_mc" | Shard combination algorithm |
| per_shard_timeout | 3600 | 1 hour max per shard |
| **min_points_per_shard** | 10,000 (laminar) / 5,000 (static) | **NEW**: Minimum shard size |
| **max_parameter_cv** | 1.0 | **NEW**: Heterogeneity abort threshold |
| **heterogeneity_abort** | True | **NEW**: Abort on high heterogeneity |

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| **core.py** | 807 | Main orchestration, shard size selection, runtime estimation |
| **data_prep.py** | 622 | Validation, sharding (stratified & random), noise estimation |
| **sampler.py** | 1084 | NUTS sampling, preflight checks, MCMC-safe adjustments |
| **model.py** | 373 | xpcs_model & xpcs_model_scaled (z-space) |
| **scaling.py** | 342 | Gradient balancing via ParameterScaling & z-space |
| **priors.py** | 791 | Prior distributions, data-driven initial value estimation |
| **results.py** | 598 | CMCResult dataclass, convergence diagnostics |
| **config.py** | 477 | CMCConfig parsing, validation, defaults |
| **diagnostics.py** | 522 | R-hat, ESS computation, convergence checks |
| **backends/base.py** | 200+ | Abstract backend, combine_shard_samples() |
| **backends/multiprocessing.py** | 400+ | Parallel execution, worker pool, thread management |
| **io.py** | 403 | Result serialization (JSON/NPZ) |
| **plotting.py** | 478 | Visualization utilities |

---

## Critical Features & Fixes

### v2.14.2: Diagonal Point Filtering
- Excludes t1 == t2 points (autocorrelation peaks)
- Prevents biasing fit with synthetic/interpolated data

### v2.12.0: Correct Consensus MC
- Implements Scott et al. (2016) properly
- Precision-weighted combination of posterior moments

### December 2025: Smooth Bounding
- Replaced hard clipping with tanh-based smooth bounds
- Prevents non-smooth behavior at parameter boundaries
- Enables better HMC/NUTS adaptation during warmup

### December 2025: Proper Time Grid Construction
- Fixed incorrect grid density from `np.unique(t1, t2)`
- Constructs with explicit dt spacing matching NLSQ

### Dense Mass Matrix
- `dense_mass=True` is CRITICAL for NUTS
- Learns parameter covariance during warmup
- Handles 10⁶:1 gradient imbalance from different parameter scales

### January 2026: Quality Filtering & Warm-Start (v2.19.0)

**Divergence-Based Shard Quality Filter**

After root cause analysis of CMC runs with 28% divergence rates, automatic quality filtering was added:

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Divergence Filtering (backends/multiprocessing.py)                        │
│                                                                           │
│   max_divergence_rate = config.max_divergence_rate (default: 0.10)       │
│                                                                           │
│   After all shards complete:                                             │
│   1. Calculate divergence_rate = num_divergent / total_samples per shard │
│   2. Filter out shards where divergence_rate > max_divergence_rate       │
│   3. Log excluded shards with divergence details                         │
│   4. Re-check post-filter success rate against min_success_rate          │
│                                                                           │
│   Purpose: Prevent corrupted posteriors from biasing consensus combination│
│   Shards with >10% divergences have unreliable posterior estimates       │
└───────────────────────────────────────────────────────────────────────────┘
```

### January 2026: CMC Divergence & Precision Loss Fix (v2.20.0)

**Root Cause Analysis**

CMC was producing parameter estimates diverging significantly from NLSQ:
- D0: -37% difference (12,444 vs 19,665)
- D_offset: -92% difference (71 vs 844)
- CMC uncertainties artificially small (precision-weighted bias)

Root causes identified:
1. **Excessive sharding**: 999 shards with only 3000 points each (data-starved)
2. **No NLSQ warm-start**: Cold start from config values → 28% divergence rate
3. **Consensus MC bias**: Precision-weighted combination biased toward low-variance shards

**Automatic NLSQ→CMC Warm-Start (CLI)**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ AUTOMATIC NLSQ Warm-Start (cli/commands.py, v2.20.0)                      │
│                                                                           │
│   When user runs: homodyne --method cmc --config my_config.yaml          │
│                                                                           │
│   The CLI AUTOMATICALLY:                                                  │
│   1. Runs NLSQ optimization first (unless --no-nlsq-warmstart)           │
│   2. Uses NLSQ results as initial values for CMC                         │
│   3. Reduces divergence rate from ~28% to <5%                            │
│                                                                           │
│   To disable (NOT recommended):                                          │
│     homodyne --method cmc --no-nlsq-warmstart --config my_config.yaml   │
└───────────────────────────────────────────────────────────────────────────┘
```

**Heterogeneity Detection & Abort**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Heterogeneity Detection (backends/multiprocessing.py, v2.20.0)            │
│                                                                           │
│   After collecting all shard results:                                    │
│                                                                           │
│   1. Compute coefficient of variation (CV) for each parameter:           │
│      CV = std(shard_means) / |mean(shard_means)|                         │
│                                                                           │
│   2. Check against threshold (max_parameter_cv, default 1.0):            │
│      • CV > threshold for critical params → high heterogeneity detected  │
│      • Critical params: D0, D_offset (static); + gamma_dot_t0 (laminar)  │
│                                                                           │
│   3. If heterogeneity_abort=True (default):                              │
│      → Raises RuntimeError with actionable guidance:                     │
│        "High heterogeneity detected (D0 CV=1.80). Consider:              │
│         1. Run NLSQ first for warm-start                                 │
│         2. Increase min_points_per_shard                                 │
│         3. Reduce n_shards"                                              │
│                                                                           │
│   4. If heterogeneity_abort=False:                                       │
│      → Falls back to robust_consensus_mc combination                     │
└───────────────────────────────────────────────────────────────────────────┘
```

**Robust Consensus MC Combination**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Robust Combination (backends/base.py, v2.20.0)                            │
│                                                                           │
│   When standard consensus_mc would produce biased results:               │
│                                                                           │
│   1. Compute per-shard means for each parameter                          │
│   2. Detect outlier shards using MAD (Median Absolute Deviation):        │
│      • median_mean = median(shard_means)                                 │
│      • mad = median(|shard_means - median_mean|)                         │
│      • threshold = 3 × 1.4826 × mad                                      │
│      • outlier if |mean - median_mean| > threshold                       │
│   3. Exclude outlier shards from combination                             │
│   4. Apply standard consensus_mc to retained shards                      │
│                                                                           │
│   Auto-enabled when: combination_method="robust_consensus_mc"            │
│   Or when heterogeneity detected but heterogeneity_abort=False          │
└───────────────────────────────────────────────────────────────────────────┘
```

**Dynamic Shard Sizing for Large Datasets**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Dynamic Shard Sizing (core.py, v2.20.0)                                   │
│                                                                           │
│   Problem: Fixed max_shards=500/1000 doesn't scale to 100M+ datasets     │
│                                                                           │
│   Solution: Dynamic max_shards based on dataset size                     │
│                                                                           │
│   Dataset Size    │ max_shards │ Rationale                               │
│   ────────────────┼────────────┼─────────────────────────────────────────  │
│   < 10M          │ 2,000      │ Standard parallelism                     │
│   10M - 100M     │ 10,000     │ Balanced shard count                     │
│   100M - 1B      │ 50,000     │ High parallelism                         │
│   1B+            │ 100,000    │ Extreme scale                            │
│                                                                           │
│   Additionally enforces MINIMUM shard size:                              │
│   • laminar_flow: 10,000 points (prevents data starvation)               │
│   • static: 5,000 points                                                 │
│                                                                           │
│   Angle-aware scaling factor (less aggressive, 0.3→0.6 for n_phi≤3)     │
└───────────────────────────────────────────────────────────────────────────┘
```

**Enhanced CMC vs NLSQ Diagnostics**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Enhanced Precision Analysis (diagnostics.py, v2.20.0)                     │
│                                                                           │
│   When comparing CMC to NLSQ results:                                    │
│                                                                           │
│   1. Logs detailed comparison table:                                     │
│      Parameter │ NLSQ Value │ CMC Value │ Diff% │ Z-Score │ Status      │
│      ──────────┼────────────┼───────────┼───────┼─────────┼─────────────  │
│      D0        │ 19665 ±68  │ 12444 ±14 │ -37%  │ 106.2   │ ⚠ EXCEEDS   │
│      D_offset  │ 844 ±0.9   │ 71 ±0.4   │ -92%  │ 858.9   │ ⚠ EXCEEDS   │
│                                                                           │
│   2. Flags parameters exceeding tolerance (default 3σ)                   │
│   3. Provides actionable guidance if discrepancies detected              │
└───────────────────────────────────────────────────────────────────────────┘
```

### New CMCConfig Fields (v2.20.0)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_divergence_rate` | float | 0.10 | Filter shards exceeding this divergence rate |
| `require_nlsq_warmstart` | bool | False | Require NLSQ warm-start (API-level) |
| `min_points_per_shard` | int | 10,000 (laminar) / 5,000 (static) | **NEW**: Enforced minimum shard size |
| `max_parameter_cv` | float | 1.0 | **NEW**: Heterogeneity abort threshold |
| `heterogeneity_abort` | bool | True | **NEW**: Abort on high heterogeneity |
| `min_points_per_param` | int | 1,500 | **NEW**: Param-aware shard sizing floor |

### January 2026: Heterogeneity Prevention (v2.21.0)

**Parameter Degeneracy in Laminar Flow Mode**

The `laminar_flow` model has two known parameter degeneracies that can cause
high heterogeneity across CMC shards:

**1. D₀/D_offset Linear Degeneracy**

The diffusion contribution depends on `D₀ + D_offset`, creating a linear manifold
in parameter space where different (D₀, D_offset) pairs produce equivalent fits.

| Symptom | Cause |
|---------|-------|
| `D_offset` CV > 1.0 | Shards find different points along the D₀ + D_offset = const ridge |
| `D_offset` spans positive and negative | Ridge crosses zero for D_offset |
| High `D₀` range despite good NLSQ fit | Compensating D_offset values |

**Mitigation (automatic in v2.21.0+):**
CMC internally reparameterizes to `D_total = D₀ + D_offset` and
`D_offset_frac = D_offset / D_total`, which are orthogonal and well-constrained.
Results are automatically converted back to D₀/D_offset for output.

**2. γ̇₀/β Multiplicative Correlation**

The shear contribution scales as `γ̇₀ · t^(1+β)`. Higher γ̇₀ with more negative β
can produce similar effects to lower γ̇₀ with less negative β.

| Symptom | Cause |
|---------|-------|
| `gamma_dot_t0` CV > 1.0 | Shards explore the γ̇₀-β correlation ridge |
| `gamma_dot_t0` spans 10-100× range | Compensating β values |
| `beta` moderate heterogeneity (CV ~0.5-0.8) | Correlated with γ̇₀ |

**Mitigation (automatic in v2.21.0+):**
CMC samples `log(γ̇₀)` instead of γ̇₀ directly, which improves conditioning
and reduces posterior ridge exploration.

**Bimodal Detection**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Bimodal Posterior Detection (diagnostics.py, v2.21.0)                     │
│                                                                           │
│   After MCMC sampling, each shard is checked for bimodal posteriors:     │
│                                                                           │
│   1. Fit 2-component GMM to each parameter's samples                     │
│   2. Flag as bimodal if:                                                 │
│      • min(weights) > 0.2 (both modes significant)                       │
│      • relative_separation > 0.5 (modes well-separated)                  │
│                                                                           │
│   3. Log warnings for bimodal posteriors:                                │
│      "BIMODAL POSTERIOR: Shard 3, D_offset: modes at 500 and 1500        │
│       (weights: 0.45/0.55)"                                              │
│                                                                           │
│   Purpose: Early warning of model misspecification or local minima       │
└───────────────────────────────────────────────────────────────────────────┘
```

**Param-Aware Shard Sizing**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Param-Aware Shard Sizing (config.py, v2.21.0)                             │
│                                                                           │
│   Problem: High-dimensional models need more points per shard             │
│   Solution: Scale shard size with parameter count                         │
│                                                                           │
│   adjusted_max = max(base_max × param_factor, min_points_per_param × n)  │
│   where param_factor = max(1.0, n_params / 7.0)                           │
│                                                                           │
│   Example (laminar_flow + individual scaling, 23 angles):                │
│   • n_params = 7 + 46 + 1 = 54                                           │
│   • param_factor = 54/7 = 7.71                                           │
│   • min_required = 1500 × 54 = 81,000 points                             │
│   • For 500K points → ~6 shards (vs 50+ with default sizing)             │
│                                                                           │
│   Prevents data starvation in high-dimensional per-angle modes           │
└───────────────────────────────────────────────────────────────────────────┘
```

**Diagnostic Indicators**

When heterogeneity abort triggers, check these indicators:

| Indicator | Healthy | Problematic |
|-----------|---------|-------------|
| D_offset CV | < 0.5 | > 1.0 |
| D_offset range | Within ±20% of D₀ | Spans ±D₀ or sign changes |
| gamma_dot_t0 CV | < 0.5 | > 1.0 |
| Bimodal warnings | 0 | Multiple shards |

**Configuration Options**

If heterogeneity persists after v2.21.0+ mitigations:

```yaml
optimization:
  cmc:
    reparameterization:
      d_total: true           # Default: true for laminar_flow
      log_gamma_dot: true     # Default: true for laminar_flow
    sharding:
      max_points_per_shard: 50000  # Increase for more statistical power
    validation:
      max_parameter_cv: 1.5   # Relax threshold if physical heterogeneity expected
```
