# Homodyne Physical Model Architecture

Complete documentation of the physical model implementation in homodyne.

**Version:** 2.22.2 **Last Updated:** February 2026

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
1. [Mathematical Foundation](#1-mathematical-foundation)
1. [Physical Constants & Validation](#2-physical-constants--validation)
1. [Model Hierarchy](#3-model-hierarchy)
1. [Physics Factors Pre-Computation](#4-physics-factors-pre-computation)
1. [JIT-Compiled Computation Kernels](#5-jit-compiled-computation-kernels)
1. [Shadow-Copy Architecture](#6-shadow-copy-architecture)
1. [Per-Angle Scaling System](#7-per-angle-scaling-system)
1. [Numerical Stability Techniques](#8-numerical-stability-techniques)
1. [Automatic Differentiation & Fallback](#9-automatic-differentiation--fallback)
1. [HomodyneModel Unified Interface](#10-homodynemodel-unified-interface)
1. [TheoryEngine High-Level API](#11-theoryengine-high-level-api)
1. [Fitting Infrastructure](#12-fitting-infrastructure)
1. [Complete Computation Flow](#complete-computation-flow)
1. [Quick Reference Tables](#quick-reference-tables)
1. [Key Files Reference](#key-files-reference)

______________________________________________________________________

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            USER ENTRY POINTS                                    │
│                                                                                 │
│   HomodyneModel(config)    TheoryEngine(mode)    CombinedModel(mode)           │
│     (homodyne_model.py)      (theory.py)           (models.py)                 │
│            │                      │                      │                      │
│            │   Stateful wrapper   │   Validated API      │   Pure model         │
│            │   Pre-computed grids  │   Error handling     │   JAX arrays         │
│            │                      │                      │                      │
│            └──────────────────────┼──────────────────────┘                      │
│                                   │                                             │
│                                   ▼                                             │
│                 JIT-compiled computation kernels                                │
│                     (jax_backend.py)                                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     PHYSICS FACTOR PRE-COMPUTATION                              │
│                        (physics_factors.py)                                     │
│                                                                                 │
│      PhysicsFactors.from_config(q, L, dt) → (wq_dt, sinc_prefactor)           │
│                                                                                 │
│      wavevector_q_squared_half_dt = 0.5 * q^2 * dt                             │
│      sinc_prefactor = (q * L / 2*pi) * dt                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      COMPUTATION DISPATCH                                       │
│                                                                                 │
│   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────────┐  │
│   │ NLSQ Hot Path     │  │ CMC Hot Path      │  │ Shared Utilities          │  │
│   │ (physics_nlsq.py) │  │ (physics_cmc.py)  │  │ (physics_utils.py)        │  │
│   │                   │  │                   │  │                           │  │
│   │ Meshgrid mode     │  │ Element-wise mode │  │ D(t), gamma_dot(t)       │  │
│   │ (n_t, n_t) matrix │  │ (N,) paired pts   │  │ safe_sinc, safe_exp      │  │
│   │ Full NLSQ residual│  │ ShardGrid precomp │  │ Integral matrices         │  │
│   └───────────────────┘  └───────────────────┘  └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SCALING & FITTING INFRASTRUCTURE                              │
│                                                                                 │
│   ┌─────────────────────┐  ┌──────────────────┐  ┌────────────────────────┐    │
│   │ scaling_utils.py    │  │ fitting.py        │  │ diagonal_correction.py │    │
│   │                     │  │                   │  │                        │    │
│   │ Quantile estimation │  │ ParameterSpace    │  │ C2 matrix diagonal     │    │
│   │ Per-angle contrast  │  │ FitResult         │  │ artifact removal       │    │
│   │ Auto/constant modes │  │ LS solvers (JIT)  │  │ Basic/stat/interp      │    │
│   └─────────────────────┘  └──────────────────┘  └────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 1. Mathematical Foundation

### The Homodyne XPCS Model

The homodyne model computes two-time intensity correlation functions g2(t1, t2) for
X-ray Photon Correlation Spectroscopy experiments under Couette flow geometry.

### Time-Dependent Coefficients

**Diffusion coefficient (anomalous diffusion):**

```
D(t) = D0 * t^alpha + D_offset

  D0      : Reference diffusion coefficient [A^2/s]
  alpha   : Anomalous exponent (0 = normal, >0 = super, <0 = sub)
  D_offset: Baseline diffusion at t=0 [A^2/s]
```

**Shear rate (time-dependent laminar flow):**

```
gamma_dot(t) = gamma_dot_t0 * t^beta + gamma_dot_t_offset

  gamma_dot_t0    : Reference shear rate [s^-1]
  beta            : Shear exponent (0 = constant, >0 = accelerating)
  gamma_dot_offset: Baseline shear rate at t=0 [s^-1]
  phi0            : Angular offset of flow direction [degrees]
```

### Correlation Functions

**g1 diffusion contribution:**

```
g1_diff(t1, t2) = exp[ -0.5 * q^2 * dt * |integral from t1 to t2 of D(t') dt'| ]
                = exp[ -wq_dt * D_integral ]

  wq_dt = 0.5 * q^2 * dt     (pre-computed PhysicsFactors)
```

**g1 shear contribution (per scattering angle phi):**

```
Phase(phi, t1, t2) = sinc_prefactor * cos(phi0 - phi) * |integral of gamma_dot(t') dt'|

g1_shear(phi, t1, t2) = [ sinc(Phase) ]^2

  sinc_prefactor = (q * L / 2*pi) * dt     (pre-computed PhysicsFactors)
  sinc(x) = sin(x) / x                     (UNNORMALIZED sinc)
```

**Total g1 (multiplicative):**

```
g1(phi, t1, t2) = g1_diff(t1, t2) * g1_shear(phi, t1, t2)
```

**g2 (homodyne detection equation):**

```
g2(phi, t1, t2) = offset + contrast * [g1(phi, t1, t2)]^2

  offset  ~ 1.0  (includes baseline "1" from Siegert relation)
  contrast ~ 0.5  (optical coupling/coherence factor)
```

### Integration Method

All integrals are computed via **numerical cumulative trapezoid** on the discrete time
grid. No closed-form integral formulas are used in the implementation (see MEMORY.md
critical rule).

______________________________________________________________________

## 2. Physical Constants & Validation

**File:** `homodyne/core/physics.py` (553 lines)

### PhysicsConstants

Static class with reference values for XPCS experiments:

| Category | Constant | Value | Unit | |----------|----------|-------|------| | X-ray
wavelengths | `WAVELENGTH_CU_KA` | 1.54 | A | | | `WAVELENGTH_8KEV` | 1.55 | A | | |
`WAVELENGTH_12KEV` | 1.03 | A | | | `WAVELENGTH_15KEV` | 0.83 | A | | Q-range |
`Q_MIN_TYPICAL` | 1e-4 | A^-1 | | | `Q_MAX_TYPICAL` | 1.0 | A^-1 | | Time scales |
`TIME_MIN_XPCS` | 1e-6 | s | | | `TIME_MAX_XPCS` | 1e3 | s | | Diffusion |
`DIFFUSION_MIN` / `MAX` | 1.0 / 1e6 | A^2/s | | Shear rates | `SHEAR_RATE_MIN` / `MAX` |
1e-5 / 1.0 | s^-1 | | Exponent bounds | `ALPHA_MIN` / `MAX` | -2.0 / 2.0 | dimensionless
| | | `BETA_MIN` / `MAX` | -2.0 / 2.0 | dimensionless | | Numerical | `EPS` | 1e-12 | |
| | `MAX_EXP_ARG` | 700.0 | | | | `MIN_POSITIVE` | 1e-100 | |

### Parameter Bounds

```python
def parameter_bounds() -> dict[str, list[tuple[float, float]]]:
    """Returns bounds for three model types."""
    return {
        "diffusion": [
            (1.0, 1e6),        # D0
            (-2.0, 2.0),       # alpha
            (-1e5, 1e5),       # D_offset
        ],
        "shear": [
            (1e-5, 1.0),       # gamma_dot_t0
            (-2.0, 2.0),       # beta
            (-1.0, 1.0),       # gamma_dot_t_offset
            (-30.0, 30.0),     # phi0
        ],
        "combined": [...]       # diffusion + shear (7 bounds)
    }
```

### Validation Functions

| Function | Purpose | |----------|---------| |
`validate_parameters_detailed(params, bounds, names)` | Per-parameter violation report;
skips JAX tracers | | `validate_parameters(params, bounds)` | Legacy boolean wrapper | |
`clip_parameters(params, bounds)` | Clips to bounds, logs clipped values | |
`get_default_parameters(model_type)` | Sensible defaults per model type | |
`validate_experimental_setup(q, L, wavelength)` | Range checks for experimental geometry
| | `estimate_correlation_time(D0, alpha, q)` | Estimates tau ~ 1/(q^2 * D0) |

### ValidationResult Dataclass

```python
@dataclass
class ValidationResult:
    valid: bool
    violations: list[str]      # Per-parameter violation messages
    parameters_checked: int
    message: str               # Summary: "OK ..." or "FAIL ..."
```

**JAX tracer safety:** `validate_parameters_detailed()` detects JAX tracers by checking
if the type string contains `"Tracer"` and skips validation during JIT compilation.

______________________________________________________________________

## 3. Model Hierarchy

**File:** `homodyne/core/models.py` (614 lines)

### Class Hierarchy

```
PhysicsModelBase (ABC)
├── DiffusionModel (3 parameters)
├── ShearModel (4 parameters)
└── CombinedModel (3 or 7 parameters)       ← Primary model used by NLSQ/CMC
    ├── GradientCapabilityMixin              ← JAX/NumPy gradient dispatch
    ├── BenchmarkingMixin                    ← Performance testing
    └── OptimizationRecommendationMixin      ← Method recommendations
```

### PhysicsModelBase (ABC)

```python
class PhysicsModelBase(ABC):
    def __init__(self, name: str, parameter_names: list[str])

    # Abstract
    def compute_g1(params, t1, t2, phi, q, L, dt) -> jnp.ndarray  # Core computation
    def get_parameter_bounds() -> list[tuple[float, float]]
    def get_default_parameters() -> jnp.ndarray

    # Concrete
    def validate_parameters(params) -> bool        # Delegates to physics.py
    def get_parameter_dict(params) -> dict          # Array -> named dict
```

### DiffusionModel

| Property | Value | |----------|-------| | Name | `"anomalous_diffusion"` | |
Parameters | `["D0", "alpha", "D_offset"]` (3) | | Defaults | `[100.0, 0.0, 10.0]` |

```python
def compute_g1(params, t1, t2, phi, q, L, dt):
    return compute_g1_diffusion(params, t1, t2, q, dt)  # jax_backend.py
```

### ShearModel

| Property | Value | |----------|-------| | Name | `"time_dependent_shear"` | |
Parameters | `["gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]` (4) | | Defaults |
`[1.0, 0.0, 0.0, 0.0]` |

```python
def compute_g1(params, t1, t2, phi, q, L, dt):
    # Prepend dummy diffusion params [100.0, 0.0, 10.0] for full g1_shear call
    full_params = jnp.concatenate([jnp.array([100.0, 0.0, 10.0]), params])
    return compute_g1_shear(full_params, t1, t2, phi, q, L, dt)
```

### CombinedModel (Primary Model)

```python
class CombinedModel(
    PhysicsModelBase,
    GradientCapabilityMixin,
    BenchmarkingMixin,
    OptimizationRecommendationMixin,
):
    def __init__(self, analysis_mode: str = "laminar_flow")
```

**Analysis mode mapping:**

| Input Mode | Internal Name | Parameters | Behavior |
|------------|---------------|------------|----------| | `"static"` |
`"static_diffusion"` | 3 | Diffusion only | | `"static_isotropic"` |
`"static_diffusion"` | 3 | Explicit isotropic | | `"static_anisotropic"` |
`"static_diffusion"` | 3 | Still isotropic (diffusion has no anisotropy) | |
`"laminar_flow"` | `"laminar_flow_complete"` | 7 | Full diffusion + shear |

**Key methods:**

```python
def compute_g1(params, t1, t2, phi, q, L, dt):
    if self.analysis_mode.startswith("static"):
        return compute_g1_diffusion(params, t1, t2, q, dt)
    return compute_g1_total(params, t1, t2, phi, q, L, dt)

def compute_g1_batch(params, t1_batch, t2_batch, phi_batch, q, L, dt):
    # jax.vmap over (t1, t2, phi) — cached on first call
    return self._cached_g1_vmap(params, t1_batch, t2_batch, phi_batch, q, L, dt)

def compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt):
    return compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)

def get_parameter_bounds():
    bounds = self.diffusion_model.get_parameter_bounds()  # Always 3
    if not self.analysis_mode.startswith("static"):
        bounds.extend(self.shear_model.get_parameter_bounds())  # +4
    return bounds
```

### Factory Functions

```python
def create_model(analysis_mode: str) -> CombinedModel
def get_available_models() -> list[str]
```

______________________________________________________________________

## 4. Physics Factors Pre-Computation

**File:** `homodyne/core/physics_factors.py` (369 lines)

### PhysicsFactors Dataclass

```python
@dataclass(frozen=True)
class PhysicsFactors:
    wavevector_q: float                    # q [A^-1]
    stator_rotor_gap: float                # L [A]
    dt: float                              # Time step [s]
    wavevector_q_squared_half_dt: float    # 0.5 * q^2 * dt (pre-computed)
    sinc_prefactor: float                  # (q * L / 2*pi) * dt (pre-computed)
```

**Construction:**

```python
@classmethod
def from_config(cls, q: float, L: float, dt: float, validate: bool = True):
    wavevector_q_squared_half_dt = 0.5 * (q ** 2) * dt
    sinc_prefactor = 0.5 / np.pi * q * L * dt
    return cls(q, L, dt, wavevector_q_squared_half_dt, sinc_prefactor)
```

**Validation (in `_validate()`):**

| Check | Range | Severity | |-------|-------|----------| | Positivity | q > 0, L > 0,
dt > 0 | Error | | Finiteness | All values finite | Error | | q range | [1e-4, 1.0] A^-1
| Warning | | L range | [1e5, 1e8] A (10 um to 10 mm) | Warning | | dt range | \[1e-6,
1e3\] s | Warning |

**JIT-safe unpacking:**

```python
def to_tuple(self) -> tuple[float, float]:
    return (self.wavevector_q_squared_half_dt, self.sinc_prefactor)
```

All JIT-compiled kernels accept pre-computed factors as scalar arguments rather than
dataclass references, avoiding tracer issues in the NUTS hot path.

______________________________________________________________________

## 5. JIT-Compiled Computation Kernels

**File:** `homodyne/core/jax_backend.py` (1,556 lines)

### Meshgrid Caching

```python
_meshgrid_cache: OrderedDict[tuple, tuple]  # Max 64 entries, LRU eviction
_MESHGRID_CACHE_MAX_SIZE = 64

def get_cached_meshgrid(t1, t2) -> tuple[jnp.ndarray, jnp.ndarray]
```

**Cache key:** `(len(t1), t1[0], t1[-1], dtype, len(t2), t2[0], t2[-1], dtype)`

**Note:** The actual key structure is nested: `(t1_key, t2_key)` where each component
key is `(n, float(arr[0]), float(arr[-1]), str(dtype))`. The flat representation above
is a conceptual summary.

**Skip conditions:** Large arrays (n > 2000), JAX traced values (inside JIT).

### Core Kernel: g1 Diffusion

```python
@jit
def _compute_g1_diffusion_core(params, t1, t2, wavevector_q_squared_half_dt, dt,
                                 time_grid=None):
```

**Dual-mode dispatch (determined by `t1.ndim`):**

| Mode | t1.ndim | Shape In | Shape Out | Used By |
|------|---------|----------|-----------|---------| | Element-wise | 1 | (N,) | (N,) |
CMC shards | | Matrix | 2 | (n_t, n_t) | (n_t, n_t) | NLSQ |

**Algorithm (both modes):**

```
1. Compute D(t) = D0 * time_safe^alpha + D_offset
                  (time_safe uses jnp.where floor to prevent t=0 singularity)

2. Build cumulative integral via trapezoid cumsum

3. Map t1, t2 to integral values:
   - Element-wise: searchsorted → lookup → smooth_abs(diff)
   - Matrix: index arithmetic → integral matrix

4. Log-space computation:
   log_g1 = -wq_dt * D_integral
   log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)
   g1 = jnp.exp(log_g1_bounded)
```

### Core Kernel: g1 Shear

```python
@jit
def _compute_g1_shear_core(params, t1, t2, phi, sinc_prefactor, dt, time_grid=None):
```

**Static mode short-circuit:** Returns `jnp.ones_like(t1)` when `params.shape[0] < 7`.

**Laminar flow algorithm:**

```
1. Compute gamma_dot(t) integral (same cumsum pattern as diffusion)

2. Element-wise (1D):
   angle_diff = deg2rad(phi0 - phi)
   phase = sinc_prefactor * cos(angle_diff) * gamma_integral
   g1_shear = safe_sinc(phase) ** 2      → shape (N,)

3. Matrix (2D) — vmap over angles for memory efficiency:
   def _sinc2_for_one_phi(phi_scalar):
       phase = sinc_prefactor * cos(deg2rad(phi0 - phi_scalar)) * gamma_integral
       return safe_sinc(phase) ** 2

   g1_shear = vmap(_sinc2_for_one_phi)(phi_array)  → shape (n_phi, n_t, n_t)
```

**Memory design:** `vmap` over angles keeps peak memory at O(n_t^2) instead of O(n_phi
\* n_t^2).

### Core Kernel: g1 Total

```python
@jit
def _compute_g1_total_core(params, t1, t2, phi, wq_dt, sinc_prefactor, dt,
                             time_grid=None):
```

```
g1_diff  = _compute_g1_diffusion_core(...)     # (n_t, n_t) or (N,)
g1_shear = _compute_g1_shear_core(...)          # (n_phi, n_t, n_t) or (N,)

# Broadcast and multiply
if matrix mode:
    g1_total = g1_diff[None, :, :] * g1_shear   # (n_phi, n_t, n_t)
else:
    g1_total = g1_diff * g1_shear                # (N,)

# Gradient-safe lower floor
g1_bounded = jnp.where(g1_total > 1e-10, g1_total, 1e-10)
```

### Core Kernel: g2 Scaled

```python
@jit
def _compute_g2_scaled_core(params, t1, t2, phi, wq_dt, sinc_prefactor,
                              contrast, offset, dt):
```

```
g1 = _compute_g1_total_core(...)
g2 = offset + contrast * g1 ** 2
return g2                                  # No output clipping
```

### Public Wrapper Functions

| Function | Pre-computes | Delegates To | |----------|-------------|-------------| |
`compute_g1_diffusion(params, t1, t2, q, dt)` | meshgrid, wq_dt |
`_compute_g1_diffusion_core` | | `compute_g1_shear(params, t1, t2, phi, q, L, dt)` |
meshgrid, sinc_pre | `_compute_g1_shear_core` | |
`compute_g1_total(params, t1, t2, phi, q, L, dt)` | meshgrid, wq_dt, sinc_pre |
`_compute_g1_total_core` | |
`compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)` | wq_dt, sinc_pre |
`_compute_g2_scaled_core` | |
`compute_g2_scaled_with_factors(params, ..., wq_dt, sinc_pre, ...)` | None
(pre-computed) | `_compute_g2_scaled_core` | |
`compute_chi_squared(params, data, sigma, ...)` | wq_dt, sinc_pre |
`_compute_g2_scaled_core` |

### Automatic Differentiation (Module-Level)

```python
gradient_g2  = grad(compute_g2_scaled, argnums=0)     # d(g2)/d(params)
hessian_g2   = hessian(compute_g2_scaled, argnums=0)   # d^2(g2)/d(params)^2
gradient_chi2 = grad(compute_chi_squared, argnums=0)
hessian_chi2  = hessian(compute_chi_squared, argnums=0)
```

### Vectorization (Module-Level)

```python
_vmap_g2_scaled    = vmap(compute_g2_scaled, in_axes=(0, None, ...))
_vmap_chi_squared  = vmap(compute_chi_squared, in_axes=(0, None, ...))

def vectorized_g2_computation(params_batch, ...) -> jnp.ndarray   # (n_batch, ...)
def batch_chi_squared(params_batch, ...) -> jnp.ndarray            # (n_batch,)
```

______________________________________________________________________

## 6. Shadow-Copy Architecture

The physics computation is implemented in **5 parallel paths** optimized for different
execution contexts. All paths must produce identical results for the same inputs.

### Shadow-Copy Registry

| Computation | jax_backend.py | physics_nlsq.py | physics_cmc.py (precomp) |
physics_cmc.py (legacy) | physics_utils.py |
|-------------|---------------|-----------------|------------------------|------------------------|-----------------|
| **D(t)** | (inline) | (inline) | (inline) | (inline) |
`calculate_diffusion_coefficient` | | **gamma_dot(t)** | (inline) | (inline) | (inline)
| (inline) | `calculate_shear_rate` | | **g1_diff** | `_compute_g1_diffusion_core` |
`_compute_g1_diffusion_meshgrid` | `_compute_g1_diffusion_from_idx` |
`_compute_g1_diffusion_elementwise` | - | | **g1_shear** | `_compute_g1_shear_core` |
`_compute_g1_shear_meshgrid` | `_compute_g1_shear_from_idx` |
`_compute_g1_shear_elementwise` | - | | **g1_total** | `_compute_g1_total_core` |
`_compute_g1_total_meshgrid` | `_compute_g1_total_with_precomputed` |
`_compute_g1_total_elementwise` | - | | **g2_scaled** | `_compute_g2_scaled_core` |
`_compute_g2_scaled_meshgrid` | (in model.py) | (in model.py) | - |

### Why Shadow Copies Exist

| Path | Optimized For | Data Layout | Memory Profile |
|------|---------------|-------------|----------------| | **jax_backend** |
General/dispatcher | Both modes | Depends on caller | | **physics_nlsq** | NLSQ
optimizer | Meshgrid (n_t, n_t) | O(n_phi * n_t^2) — quadratic | | **physics_cmc
(precomp)** | NUTS leapfrog (Feb 2026) | Pre-indexed (N,) | O(N) — linear | |
**physics_cmc (legacy)** | NUTS leapfrog (pre-Feb 2026) | Element-wise (N,) | O(N +
G\*log(G)) per step | | **physics_utils** | Base D(t)/gamma(t) only | 1D array | O(G) |

### NLSQ-Specific Path

**File:** `homodyne/core/physics_nlsq.py` (480 lines)

Meshgrid-only implementation. Time arrays are 2D: `t1[:, 0]` extracts unique times
(indexing="ij").

```python
@jit
def _compute_g1_diffusion_meshgrid(params, t1, t2, wavevector_q_squared_half_dt, dt):
    # CRITICAL: time_array is ALREADY physical time (seconds), NOT frame indices
    # DO NOT multiply by dt
    time_array = t1[:, 0]
    ...

@jit
def _compute_g1_shear_meshgrid(params, t1, t2, phi, sinc_prefactor, dt):
    # Output: (n_phi, n_t, n_t) — broadcasts (n_phi, 1, 1) * (n_t, n_t)

# Public API
def compute_g2_scaled_with_factors(params, t1, t2, phi, wq_dt, sinc_pre,
                                     contrast, offset, dt) -> jnp.ndarray
```

### CMC-Specific Path

**File:** `homodyne/core/physics_cmc.py` (807 lines)

Element-wise implementation with **ShardGrid pre-computation** (Feb 2026 optimization).

#### ShardGrid (NamedTuple)

```python
class ShardGrid(NamedTuple):
    time_safe: jnp.ndarray    # (G,) — epsilon-floored time grid
    idx1: jnp.ndarray         # (N,) int — searchsorted(time_grid, t1)
    idx2: jnp.ndarray         # (N,) int — searchsorted(time_grid, t2)
    dt_safe: float             # Time step for epsilon computation
```

**Pre-computation (called ONCE per shard, outside NUTS loop):**

```python
def precompute_shard_grid(time_grid, t1, t2, dt) -> ShardGrid:
    epsilon = max(dt/2, 1e-8)
    time_safe = jnp.where(time_grid > epsilon, time_grid, epsilon)
    idx1 = jnp.searchsorted(time_grid, t1).clip(0, len(time_grid)-1)
    idx2 = jnp.searchsorted(time_grid, t2).clip(0, len(time_grid)-1)
    return ShardGrid(time_safe, idx1, idx2, dt_safe=float(dt))
```

**NUTS hot-path kernel (called every leapfrog step):**

```python
@jit
def _compute_g1_total_with_precomputed(params, phi_unique, time_safe, idx1, idx2,
                                         wq_dt, sinc_pre):
    # Uses pre-computed indices — no searchsorted per step
    # 2-5x wall time speedup for laminar_flow
```

**Performance impact:** Eliminates O(N * log(G)) searchsorted per NUTS leapfrog step.
Only pays O(N * log(G)) once during shard initialization.

### Shared Utilities

**File:** `homodyne/core/physics_utils.py` (369 lines)

| Function | Signature | JIT | Purpose | |----------|-----------|-----|---------| |
`safe_exp(x, max_val=700.0)` | `jnp.ndarray -> jnp.ndarray` | Yes | Overflow-protected
exponential | | `safe_sinc(x)` | `jnp.ndarray -> jnp.ndarray` | Yes | UNNORMALIZED sinc
with Taylor near zero | | `calculate_diffusion_coefficient(t, D0, alpha, D_offset)` |
`-> jnp.ndarray` | Yes | D(t) with singularity floor | |
`calculate_shear_rate(t, gamma0, beta, offset)` | `-> jnp.ndarray` | Yes | gamma_dot(t)
| | `calculate_shear_rate_cmc(t, gamma0, beta, offset)` | `-> jnp.ndarray` | Yes | CMC
variant with dt=0 guard | | `create_time_integral_matrix(f)` | `-> jnp.ndarray` | Yes |
2D integral matrix via cumsum | | `trapezoid_cumsum(values)` | `-> jnp.ndarray` | No |
1D cumulative trapezoidal sum |

### Consistency Invariants (Verified Across All 5 Paths)

- `epsilon_abs = 1e-12` for division-by-zero guards
- `jnp.clip(log_g1, -700, 0)` for log-space computation
- `jnp.where(g1 > eps, g1, eps)` gradient-safe floor (NOT `jnp.maximum`)
- No `jnp.clip(g2)` — output bounds enforced by parameter constraints

______________________________________________________________________

## 7. Per-Angle Scaling System

**File:** `homodyne/core/scaling_utils.py` (335 lines)

### Overview

Per-angle scaling prevents **parameter absorption degeneracy** where D0 and contrast
trade off against each other. Each scattering angle phi has independent contrast and
offset values for the g2 equation.

### Quantile-Based Estimation Algorithm

```
Input: c2_data, delta_t = |t1 - t2| (pooled across all angles)

Step 1: Find lag thresholds
    large_lag = percentile(delta_t, 80%)     # Top 20% of time lags
    small_lag = percentile(delta_t, 20%)     # Bottom 20% of time lags

Step 2: OFFSET estimation (large-lag region where g1^2 -> 0)
    c2_floor = nanpercentile(c2[large_lag_mask], 10%)
    offset_est = clip(c2_floor, [0.5, 1.5])

Step 3: CONTRAST estimation (small-lag region where g1^2 -> 1)
    c2_ceiling = nanpercentile(c2[small_lag_mask], 90%)
    contrast_est = clip(c2_ceiling - offset_est, [0.0, 1.0])

Output: (contrast, offset) per angle
```

### Functions

```python
def estimate_contrast_offset_from_quantiles(
    c2_data, delta_t,
    contrast_bounds=(0.0, 1.0), offset_bounds=(0.5, 1.5),
    lag_floor_quantile=0.80, lag_ceiling_quantile=0.20,
    value_quantile_low=0.10, value_quantile_high=0.90,
) -> tuple[float, float]

def estimate_per_angle_scaling(
    c2_data, t1, t2, phi_indices, n_phi,
    contrast_bounds, offset_bounds, log=None,
) -> dict[str, float]
    # Returns {"contrast_0": ..., "offset_0": ..., "contrast_1": ..., ...}

def compute_averaged_scaling(
    c2_data, t1, t2, phi_indices, n_phi,
    contrast_bounds, offset_bounds, log=None,
) -> tuple[float, float, ndarray, ndarray]
    # Returns (avg_contrast, avg_offset, per_angle_contrasts, per_angle_offsets)
```

### The 4 Per-Angle Modes

| Mode | Scaling Params | How Scaling is Handled | Total Params (laminar_flow, 23
angles) |
|------|---------------|------------------------|----------------------------------------|
| **auto** | 2 (averaged) | Quantile estimate per angle, average, optimize 2 averaged
values | **9** (7 physical + 2 optimized) | | **constant** | 0 (fixed) | Quantile
estimate per angle, average, fix at averaged values | **7** (physical only) | |
**individual** | 2 * n_phi | Optimize all per-angle contrast/offset independently |
**53** (7 + 46) | | **fourier** | 2\*K + 2 | Truncated Fourier series (K=2) for angular
variation | **17** (7 + 10) |

**`auto` mode (default for n_phi >= 3):**

```
1. estimate_per_angle_scaling()  →  {contrast_i, offset_i} for each angle
2. Average:  contrast_avg = mean(contrast_0, ..., contrast_{n_phi-1})
             offset_avg   = mean(offset_0, ..., offset_{n_phi-1})
3. Optimize: 9 parameters = 7 physical + contrast_avg + offset_avg
4. Apply:    g2_i = offset_avg + contrast_avg * g1_i^2  (same for all angles)
```

**Performance note:** `estimate_per_angle_scaling()` uses vectorized `np.bincount()` and
`np.searchsorted()` for grouped operations. Pre-sorts data by phi_indices once and
reuses for all angles (3-5x speedup on 20+ angles). Falls back to midpoint bounds when
insufficient data (\<100 points per angle).

______________________________________________________________________

## 8. Numerical Stability Techniques

### Log-Space Computation

```python
# Instead of: g1 = exp(-wq_dt * D_integral)   → underflow for large D_integral
# Use:
log_g1 = -wavevector_q_squared_half_dt * D_integral
log_g1_bounded = jnp.clip(log_g1, -700.0, 0.0)     # -700 → exp(-700) ~ 1e-304
g1 = jnp.exp(log_g1_bounded)                         # 0 → exp(0) = 1.0
```

Previous code clipped g1 directly, creating a ~16% artificial plateau. Log-space
clipping preserves the natural decay shape.

### Gradient-Safe Floors (CLAUDE.md Rule 7)

```python
# CORRECT: preserves gradients (d/dx = 1 for x < eps)
g1_bounded = jnp.where(g1_total > epsilon, g1_total, epsilon)

# INCORRECT: zeros gradients (d/dx = 0 for x < eps)
g1_bounded = jnp.maximum(g1_total, epsilon)
```

Applied to: g1_total, D(t), gamma_dot(t), time_safe, contrast in LS solver. Critical for
NLSQ Jacobian computation and NUTS leapfrog integration.

### Smooth Absolute Value

```python
# For integral matrices: need |cumsum[i] - cumsum[j]|
smooth_abs(x) = sqrt(x^2 + 1e-12)    # Differentiable at x=0
```

Maintains gradient continuity at zero crossing. Epsilon 1e-12 is chosen to be above
float32 machine epsilon (~1.2e-7) when squared.

### Taylor Expansion for sinc

```python
@jit
def safe_sinc(x):
    """UNNORMALIZED sinc: sin(x)/x with smooth Taylor near zero."""
    x2 = x * x
    near_zero = 1.0 - x2 / 6.0 + x2 * x2 / 120.0     # |x| < 1e-4
    far = jnp.sin(x) / jnp.where(jnp.abs(x) > EPS, x, 1.0)
    return jnp.where(jnp.abs(x) < 1e-4, near_zero, far)
```

The hard switch from sin(x)/x to 1.0 at |x| = EPS created gradient discontinuities that
caused spurious NUTS rejections near gamma_dot_t0 ~ 0.

### Singularity Floor for Power Laws

```python
# D(t) = D0 * t^alpha → undefined at t=0 when alpha < 0
dt_inferred = jnp.abs(time_array[min(1, n-1)] - time_array[0])
epsilon = jnp.where(dt_inferred * 0.5 > 1e-8, dt_inferred * 0.5, 1e-8)
time_safe = jnp.where(time_array > epsilon, time_array, epsilon)
```

Uses dt/2 to preserve monotonicity: D(dt/2) < D(dt) for alpha > 0. The computation is
done without Python `if` to avoid JIT recompilation per unique array length.

### Float64 Requirement

`JAX_ENABLE_X64=1` must be set before the first JAX import. Set in:

- `homodyne/__init__.py` (package import)
- `cli/main.py` (CLI entry)
- `optimization/cmc/backends/multiprocessing.py` (spawn-mode workers)

______________________________________________________________________

## 9. Automatic Differentiation & Fallback

### JAX Automatic Differentiation (Primary)

**File:** `homodyne/core/jax_backend.py`

Module-level AD primitives, all JIT-compiled:

```python
gradient_g2  = grad(compute_g2_scaled, argnums=0)
hessian_g2   = hessian(compute_g2_scaled, argnums=0)
gradient_chi2 = grad(compute_chi_squared, argnums=0)
hessian_chi2  = hessian(compute_chi_squared, argnums=0)
```

### NumPy Numerical Differentiation (Fallback)

**File:** `homodyne/core/numpy_gradients.py` (1,049 lines)

Production-grade numerical differentiation with 6 methods:

| Method | Formula | Error Order | Function Evals/Param |
|--------|---------|-------------|---------------------| | `forward` |
`(f(x+h) - f(x)) / h` | O(h) | 1 | | `backward` | `(f(x) - f(x-h)) / h` | O(h) | 1 | |
`central` | `(f(x+h) - f(x-h)) / 2h` | O(h^2) | 2 | | `complex_step` |
`Im(f(x + ih)) / h` | Machine precision | 1 | | `richardson` | Central + Neville
extrapolation | O(h^8) | 8 (4 terms) | | `adaptive` (default) | Auto-selects method per
parameter | Best available | 2-8 |

**Adaptive step size estimation:**

```
h_init = max(|x| * sqrt(eps), 1e-15)
f'' estimate = (f(x+h) - 2f(x) + f(x-h)) / h^2
h_optimal = (2 * eps * |f| / |f''|) ^ (1/3)
h_bounded = clip(h_optimal, [1e-15, 1e-3])
```

**Parallel execution:** `ThreadPoolExecutor(max_workers=min(8, n_params))` for
independent Richardson extrapolation per parameter (2-4x speedup on multi-core).

**Public API:**

```python
def numpy_gradient(func, argnums=0, config=None) -> Callable
    # Returns: gradient_func(*args) -> np.ndarray

def numpy_hessian(func, argnums=0, config=None) -> Callable
    # Returns: hessian_func(*args) -> np.ndarray (n_params, n_params)
```

### Model Mixins (Gradient Dispatch)

**File:** `homodyne/core/model_mixins.py` (520 lines)

**GradientCapabilityMixin:**

- `get_gradient_function()` — Returns JAX `gradient_g2` or NumPy fallback
- `get_hessian_function()` — Returns JAX `hessian_g2` or NumPy fallback
- `supports_gradients()` — `jax_available or numpy_gradients_available`
- `get_best_gradient_method()` — `"jax_native"` / `"numpy_fallback"` /
  `"none_available"`

**BenchmarkingMixin:**

- `benchmark_gradient_performance()` — Timing comparison of available methods
- `validate_gradient_accuracy()` — Checks gradients are finite and reasonable magnitude

**OptimizationRecommendationMixin:**

- `get_optimization_recommendations()` — Method suggestions based on backend + mode
- `get_model_info()` — Comprehensive model capabilities dict

______________________________________________________________________

## 10. HomodyneModel Unified Interface

**File:** `homodyne/core/homodyne_model.py` (505 lines)

### Design Pattern

Stateful wrapper combining robustness (pre-computed state) with performance
(JIT-compiled functional cores).

```python
class HomodyneModel:
    physics_factors: PhysicsFactors      # Pre-computed at __init__
    time_array: jnp.ndarray              # [0, dt*(n-1)]
    t1_grid: jnp.ndarray                 # (n_time, n_time) meshgrid
    t2_grid: jnp.ndarray                 # (n_time, n_time) meshgrid
    model: CombinedModel                 # Underlying physics model
    dt: float
    wavevector_q: float
    stator_rotor_gap: float
    analysis_mode: str
    start_frame: int
    end_frame: int
```

### Construction

```python
def __init__(self, config: dict):
    # 1. Extract config: dt, q, L, start_frame, end_frame, analysis_mode
    self._extract_config(config)

    # 2. Pre-compute physics factors (ONCE)
    self.physics_factors = create_physics_factors_from_config_dict(config)

    # 3. Validate end_frame sentinel resolved
    if self.end_frame < 0:
        raise ValueError("end_frame must be resolved before HomodyneModel construction")

    # 4. Create time array and meshgrids
    n_time = self.end_frame - self.start_frame + 1
    self.time_array = jnp.linspace(0, self.dt * (n_time - 1), n_time)
    self.t1_grid, self.t2_grid = jnp.meshgrid(
        self.time_array, self.time_array, indexing="ij"
    )

    # 5. Create underlying model
    self.model = CombinedModel(analysis_mode=self.analysis_mode)
```

**Config structure expected:**

```python
config = {
    "analyzer_parameters": {
        "temporal": {"dt": float, "start_frame": int, "end_frame": int},
        "scattering": {"wavevector_q": float},
        "geometry": {"stator_rotor_gap": float}
    },
    "analysis_settings": {     # Optional
        "static_mode": bool,
        "isotropic_mode": bool
    }
}
```

### Key Methods

```python
def compute_c2(self, params, phi_angles, contrast=0.5, offset=1.0) -> np.ndarray:
    """Compute C2 correlation matrices.

    Uses pre-computed physics factors and meshgrids.
    Returns shape (n_phi, n_time, n_time).
    """
    params_jax = jnp.array(params)
    phi_jax = jnp.array(phi_angles)
    q_factor, sinc_factor = self.physics_factors.to_tuple()
    result = compute_g2_scaled_with_factors(
        params_jax, self.t1_grid, self.t2_grid, phi_jax,
        q_factor, sinc_factor, contrast, offset, self.dt
    )
    return np.array(result)

def compute_c2_single_angle(self, params, phi, contrast=0.5, offset=1.0) -> np.ndarray:
    """Single angle convenience. Returns shape (n_time, n_time)."""
    return self.compute_c2(params, np.array([phi]), contrast, offset)[0]

def plot_simulated_data(self, params, phi_angles, output_dir="./simulated_data",
                         contrast=0.5, offset=1.0, generate_plots=True):
    """Compute C2, save to NPZ, generate heatmaps. Returns (c2_data, output_path)."""
```

### Analysis Mode Detection

```python
def _determine_analysis_mode(self, config):
    # Priority:
    # 1. analysis_settings.static_mode + isotropic_mode
    # 2. analysis_mode key (with fuzzy matching)
    # 3. Default: "laminar_flow"
    if static_mode:
        return "static_isotropic" if isotropic else "static_anisotropic"
    return "laminar_flow"
```

______________________________________________________________________

## 11. TheoryEngine High-Level API

**File:** `homodyne/core/theory.py` (567 lines)

### Purpose

User-friendly API with validation, error handling, and performance monitoring. Wraps
`CombinedModel` with input checking and backend fallback.

```python
class TheoryEngine:
    def __init__(self, analysis_mode: str = "laminar_flow"):
        self.model = create_model(analysis_mode)
```

### Methods

```python
def compute_g1(self, params, t1, t2, phi, q, L, dt=None):
    """Validated g1 computation."""
    # Validates: params bounds, q > 0, L > 0, typical ranges (warning)
    # Converts NumPy → JAX arrays

def compute_g2(self, params, t1, t2, phi, q, L, contrast, offset, dt=None):
    """Validated g2 computation. dt is REQUIRED (no fallback)."""
    if dt is None:
        raise ValueError("TheoryEngine.compute_g2 requires explicit dt.")

def compute_chi_squared(self, params, data, sigma, t1, t2, phi, q, L,
                          contrast, offset):
    """Chi-squared with shape/dtype/NaN validation on data and sigma."""

def batch_computation(self, params_batch, data, sigma, t1, t2, phi, q, L,
                        contrast, offset):
    """Vectorized chi-squared via vmap (JAX) or Python loop (fallback)."""

def estimate_computation_cost(self, t1, t2, phi) -> dict:
    """Returns n_total_points, estimated_operations, estimated_memory_mb,
    performance_tier (light/medium/heavy)."""
```

**Operations per point:** Static = 10, Laminar flow = 50.

### Convenience Functions

```python
def compute_g2_theory(params, t1, t2, phi, q, L, contrast, offset, dt,
                       analysis_mode="laminar_flow"):
    """One-shot g2 (creates engine on each call)."""

def compute_chi2_theory(params, data, sigma, t1, t2, phi, q, L, contrast, offset,
                          analysis_mode="laminar_flow"):
    """One-shot chi-squared."""
```

______________________________________________________________________

## 12. Fitting Infrastructure

### ParameterSpace

**File:** `homodyne/core/fitting.py` (881 lines)

```python
@dataclass
class ParameterSpace:
    # Scaling
    contrast_bounds: tuple = (0.0, 1.0)
    offset_bounds: tuple = (0.5, 1.5)
    contrast_prior: tuple = (0.5, 0.2)    # (mu, sigma) for CMC
    offset_prior: tuple = (1.0, 0.3)

    # Physical (mode-dependent)
    D0_bounds: tuple = (1.0, 1e6)
    alpha_bounds: tuple = (-2.0, 2.0)
    D_offset_bounds: tuple = (-1e5, 1e5)
    gamma_dot_t0_bounds: tuple = (1e-5, 1.0)
    beta_bounds: tuple = (-2.0, 2.0)
    gamma_dot_t_offset_bounds: tuple = (-1.0, 1.0)
    phi0_bounds: tuple = (-30.0, 30.0)

    def get_param_bounds(analysis_mode) -> list[tuple]
    def get_param_priors(analysis_mode) -> list[tuple]
```

### FitResult

```python
@dataclass
class FitResult:
    params: np.ndarray
    contrast: float
    offset: float
    chi_squared: float
    reduced_chi_squared: float
    degrees_of_freedom: int
    p_value: float
    param_errors: np.ndarray | None
    converged: bool
    computation_time: float
    backend: str                     # "JAX" or "NumPy"
    dataset_size: str                # "small" / "medium" / "large"
    analysis_mode: str
```

**Note:** Additional fields exist that are not shown above: `contrast_error`,
`offset_error`, `residual_std`, `max_residual`, `fit_iterations`.

### JIT-Compiled Least Squares Solvers

| Solver | Input | Output | Method | Use Case |
|--------|-------|--------|--------|----------| | `solve_least_squares_jax` |
theory_batch, exp_batch | (contrast[], offset[]) | 2x2 normal equations | Per-angle
contrast/offset | | `solve_least_squares_general_jax` | design_matrix, target, lambda |
coefficients | Cholesky or SVD | N-parameter LS | | `solve_least_squares_chunked_jax` |
theory_chunks, exp_chunks | (contrast[], offset[]) | lax.scan accumulation |
Memory-efficient batching |

**Singular matrix handling:** Falls back to `(1.0, 1.0)` when `|det| <= 1e-12`.

**SVD threshold:** Condition number kappa >= 1e10 triggers SVD instead of Cholesky.

### Diagonal Correction

**File:** `homodyne/core/diagonal_correction.py` (522 lines)

```python
def apply_diagonal_correction(c2_mat, method="basic", backend="auto", **config):
    """Single matrix correction. Methods: basic, statistical, interpolation."""

def apply_diagonal_correction_batch(c2_matrices, method="basic", backend="auto", **config):
    """Batch (n_phi, N, N). Uses vmap for JAX backend."""
```

| Method | Algorithm | Speed | |--------|-----------|-------| | **basic** | Average
adjacent off-diagonal neighbors | Fastest | | **statistical** | Median/trimmed-mean in
window | Robust to outliers | | **interpolation** | Linear interpolation between
neighbors | Smoothest |

### UnifiedHomodyneEngine

```python
class UnifiedHomodyneEngine:
    def __init__(self, analysis_mode="laminar_flow", parameter_space=None)

    def estimate_scaling_parameters(data, theory, validate_bounds=True):
        """JAX least-squares for contrast/offset."""

    def compute_likelihood(params, contrast, offset, data, sigma, t1, t2, phi, q, L, dt):
        """Negative log-likelihood: 0.5*chi2 + 0.5*sum(log(2*pi*sigma^2))."""

    def detect_dataset_size(data) -> str:      # "small" / "medium" / "large"
    def validate_inputs(data, sigma, t1, t2, phi, q, L) -> None
```

______________________________________________________________________

## Complete Computation Flow

### Static Mode (3 Parameters)

```
Config {dt: 0.1, q: 0.01, L: 2e6}
  │
  ▼
HomodyneModel.__init__(config)
  ├── PhysicsFactors.from_config(q=0.01, L=2e6, dt=0.1)
  │     wq_dt = 0.5 * 0.01^2 * 0.1 = 5e-6
  │     sinc_pre = (0.01 * 2e6 / 2*pi) * 0.1 = 318.3
  ├── time_array = [0, 0.1, 0.2, ..., dt*(n-1)]
  └── t1_grid, t2_grid = meshgrid(time_array)     (n_t, n_t)
  │
  ▼
params = [100.0, 0.0, 10.0]      # [D0, alpha, D_offset]
phi = [0, 45, 90]                 # degrees (ignored in static)
  │
  ▼
compute_c2(params, phi)
  └── compute_g2_scaled_with_factors(params, t1_grid, t2_grid, phi,
                                       wq_dt=5e-6, sinc_pre=318.3,
                                       contrast=0.5, offset=1.0, dt=0.1)
        │
        ├── _compute_g1_diffusion_core(params, t1, t2, wq_dt, dt)
        │     ├── D(t) = 100.0 * time_safe^0.0 + 10.0 = 110.0  (constant)
        │     ├── D_integral = cumtrapez(D(t))     (n_t, n_t) matrix
        │     ├── log_g1 = -5e-6 * D_integral
        │     └── g1_diff = exp(clip(log_g1, -700, 0))
        │
        ├── _compute_g1_shear_core(params, ...)
        │     └── params.shape[0] == 3 < 7 → return ones(n_t, n_t)
        │
        ├── g1_total = g1_diff * ones = g1_diff
        │     └── jnp.where(g1_total > 1e-10, g1_total, 1e-10)
        │
        └── g2 = 1.0 + 0.5 * g1_total^2
  │
  ▼
result: np.ndarray shape (3, n_t, n_t)     # 3 angles, same C2 per angle
```

### Laminar Flow Mode (7 Parameters)

```
params = [100.0, 0.0, 10.0, 1e-4, 0.0, 0.0, 0.0]
         │       │     │     │     │    │     │
         D0    alpha D_off  gamma beta g_off  phi0
  │
  ▼
compute_g2_scaled_with_factors(...)
  │
  ├── _compute_g1_diffusion_core(...)  →  g1_diff (n_t, n_t)
  │
  ├── _compute_g1_shear_core(...)
  │     ├── gamma_dot(t) = 1e-4 * time_safe^0.0 + 0.0 = 1e-4  (constant)
  │     ├── gamma_integral = cumtrapez(gamma_dot(t))
  │     └── vmap over phi:
  │           phi=0:   phase = 318.3 * cos(0) * gamma_integral
  │           phi=45:  phase = 318.3 * cos(-45deg) * gamma_integral
  │           phi=90:  phase = 318.3 * cos(-90deg) * gamma_integral
  │           g1_shear = safe_sinc(phase)^2           (n_phi, n_t, n_t)
  │
  ├── g1_total = g1_diff[None,:,:] * g1_shear         (n_phi, n_t, n_t)
  │
  └── g2 = 1.0 + 0.5 * g1_total^2                    (n_phi, n_t, n_t)
```

### CMC Shard Computation

```
shard_data: t1(N,), t2(N,), phi(N,), c2(N,)
  │
  ▼
precompute_shard_grid(time_grid, t1, t2, dt)     ← Called ONCE per shard
  ├── time_safe = jnp.where(time_grid > eps, time_grid, eps)    (G,)
  ├── idx1 = searchsorted(time_grid, t1).clip(0, G-1)           (N,)
  └── idx2 = searchsorted(time_grid, t2).clip(0, G-1)           (N,)
  │
  ▼
For each NUTS leapfrog step:                      ← Called 2^max_tree_depth times
  │
  ▼
_compute_g1_total_with_precomputed(              ← HOT PATH (JIT-compiled)
    params, phi_unique, time_safe, idx1, idx2, wq_dt, sinc_pre)
  ├── D(t) on time_safe → cumsum → lookup idx1, idx2
  ├── g1_diff(N,)
  ├── gamma_dot(t) on time_safe → cumsum → lookup idx1, idx2
  ├── g1_shear(P, N)     # P = n_phi_unique
  ├── g1_total = g1_diff * g1_shear[phi_indices]
  └── g2 = offset + contrast * g1_total^2
  │
  ▼
likelihood = Normal(g2, sigma).log_prob(c2_observed)
```

______________________________________________________________________

## Quick Reference Tables

### Parameter Summary

| Parameter | Symbol | Bounds | Default | Unit | Mode |
|-----------|--------|--------|---------|------|------| | D0 | D0 | [1.0, 1e6] | 100.0 |
A^2/s | Both | | alpha | alpha | [-2.0, 2.0] | 0.0 | - | Both | | D_offset | D_offset |
[-1e5, 1e5] | 10.0 | A^2/s | Both | | gamma_dot_t0 | gamma0 | [1e-5, 1.0] | 1.0 | s^-1 |
laminar_flow | | beta | beta | [-2.0, 2.0] | 0.0 | - | laminar_flow | |
gamma_dot_t_offset | gamma_off | [-1.0, 1.0] | 0.0 | s^-1 | laminar_flow | | phi0 | phi0
| [-30.0, 30.0] | 0.0 | deg | laminar_flow | | contrast | C | [0.0, 1.0] | 0.5 | - |
Both | | offset | O | [0.5, 1.5] | 1.0 | - | Both |

### Memory Layout

| Context | Shape Convention | Memory Profile |
|---------|-----------------|----------------| | NLSQ meshgrid | (n_phi, n_t, n_t) |
O(n_phi * n_t^2) — 80 GB for 100K pts | | CMC element-wise | (N,) paired points | O(N) —
2.4 MB for 3 angles x 100K pts | | CMC precomputed grid | (G,) time grid | O(G) ~ 80 KB
for 10K grid points | | Diagonal correction | (n_phi, N, N) | O(n_phi * N^2) |

### JIT Compilation Points

| Function | File | Decorator | Purpose | |----------|------|-----------|---------| |
`_compute_g1_diffusion_core` | jax_backend.py | `@jit` | g1 diffusion kernel | |
`_compute_g1_shear_core` | jax_backend.py | `@jit` | g1 shear kernel | |
`_compute_g1_total_core` | jax_backend.py | `@jit` | Combined g1 kernel | |
`_compute_g2_scaled_core` | jax_backend.py | `@jit` | Homodyne equation | |
`_compute_g1_diffusion_meshgrid` | physics_nlsq.py | `@jit` | NLSQ diffusion | |
`_compute_g1_shear_meshgrid` | physics_nlsq.py | `@jit` | NLSQ shear | |
`_compute_g1_total_meshgrid` | physics_nlsq.py | `@jit` | NLSQ combined | |
`_compute_g2_scaled_meshgrid` | physics_nlsq.py | `@jit` | NLSQ g2 | |
`_compute_g1_diffusion_from_idx` | physics_cmc.py | `@jit` | CMC precomp diffusion | |
`_compute_g1_shear_from_idx` | physics_cmc.py | `@jit` | CMC precomp shear | |
`_compute_g1_total_with_precomputed` | physics_cmc.py | `@jit` | CMC hot path | |
`safe_sinc` | physics_utils.py | `@jit` | Numerical sinc | | `safe_exp` |
physics_utils.py | `@jit` | Overflow-safe exp | | `calculate_diffusion_coefficient` |
physics_utils.py | `@jit` | D(t) evaluation | | `calculate_shear_rate` |
physics_utils.py | `@jit` | gamma_dot(t) | | `solve_least_squares_jax` | fitting.py |
`@jit` | 2x2 LS batch solver | | `solve_least_squares_general_jax` | fitting.py | `@jit`
| N-param LS solver | | `solve_least_squares_chunked_jax` | fitting.py | `@jit` |
lax.scan LS solver | | `_diagonal_correction_jax_core` | diagonal_correction.py | `@jit`
| Diagonal fix |

### Numerical Constants

| Constant | Value | Where Used | Purpose | |----------|-------|------------|---------|
| `EPS` | 1e-12 | physics_utils, jax_backend | Division-by-zero guard | | `MAX_EXP_ARG`
| 700.0 | safe_exp, log_g1 clip | Overflow prevention | | g1 floor | 1e-10 |
g1_total_core | Gradient-safe positivity | | contrast floor | 1e-6 | LS solver |
Gradient-safe positivity | | sinc Taylor threshold | 1e-4 | safe_sinc | Smooth gradient
transition | | Singularity floor | max(dt/2, 1e-8) | time_safe | Power law t=0
protection | | Log-space clip | [-700, 0] | all g1 kernels | Underflow/overflow |

______________________________________________________________________

## Key Files Reference

### Core Physics Model (~9,100 lines)

| File | Lines | Purpose | |------|-------|---------| | `core/jax_backend.py` | 1,556 |
JIT kernels, AD, caching, vectorization | | `core/numpy_gradients.py` | 1,049 | NumPy
gradient fallback (6 methods) | | `core/fitting.py` | 881 | ParameterSpace, FitResult,
LS solvers | | `core/physics_cmc.py` | 807 | CMC element-wise + ShardGrid precomp | |
`core/models.py` | 614 | DiffusionModel, ShearModel, CombinedModel | | `core/theory.py`
| 567 | TheoryEngine validated API | | `core/physics.py` | 553 | PhysicsConstants,
bounds, validation | | `core/diagonal_correction.py` | 522 | C2 diagonal artifact
correction | | `core/model_mixins.py` | 520 | Gradient, benchmark, recommendation mixins
| | `core/homodyne_model.py` | 505 | HomodyneModel unified interface | |
`core/physics_nlsq.py` | 480 | NLSQ meshgrid physics | | `core/physics_factors.py` | 369
| PhysicsFactors pre-computation | | `core/physics_utils.py` | 369 | Shared safe_sinc,
D(t), gamma(t), integrals | | `core/scaling_utils.py` | 335 | Per-angle quantile
estimation | | `core/backend_api.py` | 192 | Backend selection API | |
`core/__init__.py` | 80 | Package init and public exports | | **Total** | **~9,399** | |

### Integration Points

| Consumer | Physics File Used | Entry Function |
|----------|------------------|----------------| | NLSQ optimizer | `physics_nlsq.py` |
`compute_g2_scaled_with_factors()` | | CMC NUTS sampler | `physics_cmc.py` |
`compute_g1_total_with_precomputed()` | | HomodyneModel | `jax_backend.py` |
`compute_g2_scaled_with_factors()` | | TheoryEngine | `models.py` → `jax_backend.py` |
`CombinedModel.compute_g1()` | | CLI simulated data | `homodyne_model.py` |
`HomodyneModel.compute_c2()` | | Diagonal correction | `diagonal_correction.py` |
`apply_diagonal_correction_batch()` | | Scaling estimation | `scaling_utils.py` |
`estimate_per_angle_scaling()` |
