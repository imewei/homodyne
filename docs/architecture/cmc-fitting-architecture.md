# Homodyne CMC (Consensus Monte Carlo) Fitting Architecture

Complete documentation of the CMC (Consensus Monte Carlo) fitting system in homodyne.

**Version:** 2.19.0
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
│                              USER ENTRY POINT                                    │
│                                                                                  │
│                         fit_mcmc_jax(data, config)                               │
│                               (core.py)                                          │
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

### Decision Logic by Mode

```
┌───────────────────────────────────────────────────────────────────────────┐
│ LAMINAR FLOW MODE (7 parameters, complex gradients)                        │
│                                                                            │
│   Dataset Size    │ max_points_per_shard │ Est. Shards │ Per-Shard Time   │
│   ────────────────┼──────────────────────┼─────────────┼─────────────────  │
│   < 2M points     │ 20K                  │ ~100        │ ~10-15 min       │
│   2M - 20M        │ 10K (sweet spot)     │ 200-2000    │ ~5-8 min         │
│   20M - 50M       │ 8K                   │ 2.5K-6K     │ ~5 min           │
│   50M - 100M      │ 6K                   │ 8K-17K      │ ~4 min           │
│   100M+           │ 5K                   │ 20K+        │ ~3 min           │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ STATIC MODE (3 parameters, simple gradients - 10× larger shards OK)       │
│                                                                            │
│   Dataset Size    │ max_points_per_shard │ Est. Shards                     │
│   ────────────────┼──────────────────────┼──────────────                   │
│   < 50M points    │ 100K                 │ ~500                            │
│   50M - 100M      │ 80K                  │ ~1K                             │
│   100M+           │ 50K                  │ ~2K+                            │
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
│   Default max_shards = 2000 → ~12GB peak memory during combination         │
│                                                                            │
│   If exceeded:                                                             │
│     • Increases shard size (no subsampling, all data used)                 │
│     • For laminar_flow: caps adjusted shard size at 50K max                │
│                                                                            │
│   Platform Limits:                                                         │
│   ├─ Personal (32GB):  ~500 shards  → ~5M points practical max            │
│   ├─ Bebop (128GB):    ~2500 shards → ~25M points practical max           │
│   └─ Improv (256GB):   ~5000 shards → ~50M points practical max           │
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
║ CONSENSUS_MC (Recommended, v2.12.0+)                                       ║
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

### Auto Shard Size Selection

#### Laminar Flow Mode

| Dataset Size | max_points_per_shard | Est. Shards | Per-Shard Time |
|--------------|---------------------|-------------|----------------|
| < 2M | 20K | ~100 | ~10-15 min |
| 2M - 20M | 10K | 200-2000 | ~5-8 min |
| 20M - 50M | 8K | 2.5K-6K | ~5 min |
| 50M - 100M | 6K | 8K-17K | ~4 min |
| 100M+ | 5K | 20K+ | ~3 min |

#### Static Mode (10× larger shards)

| Dataset Size | max_points_per_shard | Est. Shards |
|--------------|---------------------|-------------|
| < 50M | 100K | ~500 |
| 50M - 100M | 80K | ~1K |
| 100M+ | 50K | ~2K+ |

### Platform Shard Limits

| Platform | Memory | Max Shards | Max Dataset (laminar) |
|----------|--------|------------|----------------------|
| Personal (32GB) | ~20GB | ~500 | ~5M points |
| Bebop (128GB) | ~100GB | ~2500 | ~25M points |
| Improv (256GB) | ~200GB | ~5000 | ~50M points |

### Mode-Specific Parameters

| Mode | Physical Params | Per-Angle Params (23 angles) | Total |
|------|----------------|------------------------------|-------|
| static | 3: D₀, α, D_offset | 46: contrast + offset | 49 + σ |
| laminar_flow | 7: + γ̇₀, β, γ̇_offset, φ₀ | 46: contrast + offset | 53 + σ |

### CMC Configuration Defaults

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
| max_divergence_rate | 0.10 | Quality filter threshold (v2.19.0) |
| require_nlsq_warmstart | False | Enforce NLSQ warm-start (v2.19.0) |
| combination_method | "consensus_mc" | Shard combination algorithm |
| per_shard_timeout | 3600 | 1 hour max per shard (reduced from 2h) |

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

### January 2026: Quality Filtering & Warm-Start

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

**NLSQ Warm-Start Validation**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ NLSQ Warm-Start Advisory (core.py)                                        │
│                                                                           │
│   When running laminar_flow without nlsq_result or initial_values:       │
│                                                                           │
│   If require_nlsq_warmstart=True (from CMCConfig):                       │
│     → Raises ValueError (hard failure)                                   │
│                                                                           │
│   If require_nlsq_warmstart=False (default):                             │
│     → Logs warning explaining:                                           │
│       • 7 parameters span 6+ orders of magnitude                         │
│       • NUTS adaptation wastes warmup without good initial values        │
│       • Higher divergence rates expected (~28% vs <5% with warm-start)   │
│                                                                           │
│   Best practice: Always run NLSQ first, pass nlsq_result to fit_mcmc_jax │
└───────────────────────────────────────────────────────────────────────────┘
```

**Dynamic Shard Size Caps for laminar_flow**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Dynamic max_shards by Analysis Mode (core.py)                             │
│                                                                           │
│   laminar_flow mode:                                                     │
│     • max_shards = 1000 (up from 500)                                    │
│     • Allows 3K-5K points per shard for datasets up to 5M points         │
│     • NUTS is O(n) per step; smaller shards prevent timeout              │
│                                                                           │
│   static mode:                                                           │
│     • max_shards = 500 (unchanged)                                       │
│     • 3 parameters handle larger shards efficiently                      │
└───────────────────────────────────────────────────────────────────────────┘
```

### New CMCConfig Fields (v2.19.0)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_divergence_rate` | float | 0.10 | Filter shards exceeding this divergence rate |
| `require_nlsq_warmstart` | bool | False | Require NLSQ warm-start for laminar_flow |
