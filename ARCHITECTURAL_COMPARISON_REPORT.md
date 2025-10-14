# Comprehensive Architectural Comparison: Simulated Data Plotting
## Current Codebase vs. Working Version (homodyne-analysis)

**Analysis Date**: October 14, 2025
**Branch**: `fix/simulated-data-plotting-errors`
**Status**: ✅ **BUGS FIXED** - Architectural differences documented

---

## Executive Summary

This ultra-think analysis compares the simulated data plotting implementation between the current JAX-based homodyne v2 codebase and the working NumPy-based homodyne-analysis version. While **all critical bugs have been fixed**, fundamental architectural differences remain that explain why the bugs occurred and provide insights for future development.

### Key Findings:
1. ✅ **Bugs Fixed**: Time grid formula, dt propagation, data independence
2. 📊 **Architecture**: Stateful OOP (working) vs. Stateless Functional (current)
3. 🎯 **Root Cause**: Architectural transition without disciplined parameter management
4. 💡 **Recommendation**: Hybrid approach combining robustness with flexibility

---

## Part 1: Architectural Comparison

### 1.1 Overall Design Philosophy

#### **Working Version (homodyne-analysis)** - Object-Oriented, Stateful
```python
class HomodyneAnalysisCore:
    def __init__(self, config_file):
        # State initialization
        self.dt = params["temporal"]["dt"]
        self.time_length = end_frame - start_frame + 1
        self.time_array = np.linspace(0, self.dt * (self.time_length - 1),
                                     self.time_length)

        # Pre-compute physics factors ONCE
        self.wavevector_q_squared_half_dt = 0.5 * q² * self.dt
        self.sinc_prefactor = 0.5/π * q * L * self.dt

    def calculate_c2_nonequilibrium_laminar_parallel(self, params, phi_angles):
        # Methods access pre-computed state
        g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)
        # ... uses self.dt, self.time_array internally
```

**Characteristics**:
- ✅ **Single source of truth**: dt stored once, validated once
- ✅ **Eager pre-computation**: Physics factors computed at initialization
- ✅ **Memory efficiency**: Pre-allocated result pools
- ✅ **Robust**: Configuration errors caught early
- ❌ **Less flexible**: State tied to instance
- ❌ **Not JIT-friendly**: State access prevents some optimizations

---

#### **Current Version (homodyne v2)** - Functional, Stateless

```python
def compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt=None):
    # NO state - everything passed as parameters

    # Compute physics factors AT RUNTIME
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g2_scaled_core(params, t1, t2, phi,
                                   wavevector_q_squared_half_dt,
                                   sinc_prefactor, contrast, offset)

# Usage in plotting:
c2_phi = model.compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt)
```

**Characteristics**:
- ✅ **JIT-compilable**: Pure functions suitable for JAX JIT
- ✅ **Flexible**: No state dependencies
- ✅ **Testable**: Easy to test individual functions
- ✅ **Composable**: Functions can be combined easily
- ❌ **Parameter discipline required**: dt must be passed correctly everywhere
- ❌ **Repeated computation**: Physics factors computed on every call
- ❌ **Error-prone**: Fallback estimation can mask configuration errors

---

### 1.2 Time Array Generation

#### **Working Version** ✅ CORRECT
```python
# homodyne-analysis/homodyne/analysis/core.py:264-269
self.time_array = np.linspace(
    0,
    self.dt * (self.time_length - 1),  # ✅ Correct formula
    self.time_length,
    dtype=np.float64,
)
```

**Formula**: `t[i] = dt * i` for `i = 0, 1, ..., n-1`
**Result**: Time array perfectly aligned with configuration

---

#### **Current Version** ✅ NOW CORRECT (after fix)

**BEFORE Fix** ❌:
```python
# WRONG - Missing +1 in formula
time_max = dt * (end_frame - start_frame)  # Should be dt * (n - 1)
t_vals = jnp.linspace(0, time_max, n_time_points)
```

**AFTER Fix** ✅:
```python
# homodyne/cli/commands.py:1387-1388 (FIXED)
n_time_points = end_frame - start_frame + 1  # Inclusive counting
time_max = dt * (n_time_points - 1)  # ✅ Correct formula
t_vals = jnp.linspace(0, time_max, n_time_points)
```

**Key Difference**:
- Working version: Formula was always correct
- Current version: Formula was FIXED on October 14, 2025

---

### 1.3 Physics Factors Pre-computation

#### **Working Version**: Eager Pre-computation ✅

```python
# homodyne-analysis/homodyne/analysis/core.py:252-257
def _initialize_parameters(self):
    # Computed ONCE during initialization
    self.wavevector_q_squared = self.wavevector_q**2
    self.wavevector_q_squared_half_dt = 0.5 * self.wavevector_q_squared * self.dt
    self.sinc_prefactor = (
        0.5 / np.pi * self.wavevector_q * self.stator_rotor_gap * self.dt
    )
```

**Benefits**:
- Computed once, validated once
- Reused across all calculations
- More efficient for repeated calls
- Configuration errors caught immediately

---

#### **Current Version**: Lazy Computation

```python
# homodyne/homodyne/core/jax_backend.py:936-937
def compute_g2_scaled(..., dt=None):
    # Computed EVERY TIME the function is called
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g2_scaled_core(...)
```

**Trade-offs**:
- More flexible (can use different dt values)
- Suitable for JIT compilation
- Less efficient (repeated computation)
- Configuration errors may appear later in workflow

---

### 1.4 dt Parameter Handling

#### **Working Version**: Centralized ✅

```python
# Stored once in instance
self.dt = params["temporal"]["dt"]

# Used throughout methods
def _calculate_c2_single_angle_fast(self, ...):
    g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)
    # self.dt is implicit in self.wavevector_q_squared_half_dt
```

**Pattern**: `__init__ → validate → store → use`
**Safety**: Single source of truth, validated once

---

#### **Current Version**: Distributed ✅ (after fix)

**BEFORE Fix** ❌:
```python
def compute_g2_scaled(..., dt=None):
    if dt is None:
        # FALLBACK: Estimate from spacing - ERROR PRONE!
        dt = time_array[1] - time_array[0]
```

**AFTER Fix** ✅:
```python
# homodyne/core/models.py:357 - dt parameter added
def compute_g2(self, ..., dt: float = None) -> jnp.ndarray:
    return compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)

# homodyne/cli/commands.py:1433 - dt passed explicitly
c2_phi = model.compute_g2(params, t1, t2, phi, q, L, contrast, offset, dt)
```

**Pattern**: `load config → pass as parameter → use`
**Safety**: Requires discipline at every call site, but now enforced

---

## Part 2: Line-by-Line Comparison

### 2.1 Simulated Data Generation Workflow

#### **Working Version** (simulation.py):

```python
# Step 1: Initialize core with config (dt stored internally)
core = HomodyneAnalysisCore(config_file)

# Step 2: Create time arrays (uses core.dt, core.time_length)
t1, t2, n_time = create_time_arrays_for_simulation(config)
# Internally: np.linspace(0, dt * (n - 1), n)

# Step 3: Generate C2 (NO dt parameter needed)
c2_theoretical = core.calculate_c2_nonequilibrium_laminar_parallel(
    initial_params, phi_angles
)
# Uses core.dt internally for all physics calculations
```

**Flow**: Config → Core (stores dt) → Calculate (uses stored dt)

---

#### **Current Version** (commands.py) - AFTER FIX:

```python
# Step 1: Load config (dt extracted but not stored in model)
config = ConfigManager(config_file).config
dt = config["analyzer_parameters"]["temporal"]["dt"]

# Step 2: Create time arrays (uses dt from config)
n_time_points = end_frame - start_frame + 1
time_max = dt * (n_time_points - 1)  # ✅ FIXED
t_vals = jnp.linspace(0, time_max, n_time_points)

# Step 3: Generate C2 (dt passed explicitly)
c2_phi = model.compute_g2(
    params, t1_grid, t2_grid, phi_array, q, L,
    contrast, offset,
    dt  # ✅ FIXED - dt passed explicitly
)
```

**Flow**: Config → Extract dt → Pass to each function

---

### 2.2 C2 Calculation Implementation

#### **Working Version** (core.py:1143-1182):

```python
def _calculate_c2_single_angle_fast(
    self, parameters, phi_angle, D_integral, is_static,
    shear_params, gamma_integral=None
):
    # Diffusion contribution - uses PRE-COMPUTED factor
    g1 = np.exp(-self.wavevector_q_squared_half_dt * D_integral)

    if is_static:
        return g1**2

    # Shear contribution - uses PRE-COMPUTED factor
    phi_offset = parameters[-1]
    angle_rad = np.deg2rad(phi_offset - phi_angle)
    cos_phi = np.cos(angle_rad)
    prefactor = self.sinc_prefactor * cos_phi  # self.sinc_prefactor pre-computed

    if NUMBA_AVAILABLE:
        sinc2 = compute_sinc_squared_numba(gamma_integral, prefactor)
    else:
        arg = prefactor * gamma_integral
        sinc_values = np.sin(arg) / arg
        sinc_values = np.where(np.abs(arg) < 1e-10, 1.0, sinc_values)
        sinc2 = sinc_values**2

    # Combine: c2 = (g1 × sinc²)²
    return (sinc2 * g1) ** 2
```

**Key Points**:
- Uses `self.wavevector_q_squared_half_dt` (pre-computed at init)
- Uses `self.sinc_prefactor` (pre-computed at init)
- Factors include dt, so dt never needs to be passed

---

#### **Current Version** (jax_backend.py:891-948):

```python
def compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt=None):
    # Create meshgrids if needed
    if t1.ndim == 1 and t2.ndim == 1:
        t2_grid, t1_grid = jnp.meshgrid(t2, t1, indexing="ij")
        t1 = t1_grid
        t2 = t2_grid

    # FALLBACK if dt not provided (after fix, dt IS provided)
    if dt is None:
        if t1.ndim == 2:
            time_array = t1[:, 0]
        else:
            time_array = t1
        dt = time_array[1] - time_array[0] if safe_len(time_array) > 1 else 1.0

    # Compute physics factors AT RUNTIME
    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    # Call core computation
    return _compute_g2_scaled_core(
        params, t1, t2, phi,
        wavevector_q_squared_half_dt,
        sinc_prefactor,
        contrast, offset
    )
```

**Key Points**:
- Computes `wavevector_q_squared_half_dt` and `sinc_prefactor` at RUNTIME
- Requires dt as parameter (after fix, this is passed explicitly)
- Fallback estimation still exists but rarely used after fix
- Functional design suitable for JAX JIT compilation

---

## Part 3: Root Cause Analysis

### 3.1 Primary Root Cause: Architectural Transition

**The Deep Issue**: Homodyne v2 transitioned from **stateful OOP** to **stateless functional** design for JAX compatibility, but:

1. ❌ **Incomplete transition**: Not all code adapted to functional paradigm
2. ❌ **Unsafe fallbacks**: dt estimation provided false safety net
3. ❌ **Mixed concerns**: Time grid generation used experimental data instead of config alone
4. ❌ **Lack of discipline**: dt parameter passing not consistently enforced

---

### 3.2 Specific Bug Root Causes

#### **Bug #1**: Time Grid Formula Error

**Cause**: Confusion between two interpretations:
- Inclusive counting: `n = end - start + 1` frames
- Time duration: `T = dt * n` seconds

**Error**: Used `time_max = dt * (end - start)` instead of `dt * (n - 1)`

**Working version avoided this** by using `self.time_length` consistently:
```python
self.time_length = self.end_frame - self.start_frame + 1
self.time_array = np.linspace(0, self.dt * (self.time_length - 1), self.time_length)
```

---

#### **Bug #2**: Missing dt Parameter

**Cause**: Functional design requires explicit parameter passing, but:
- dt was optional (`dt=None`)
- Fallback estimation masked the problem
- No compile-time enforcement

**Working version avoided this** by storing dt in instance:
```python
self.dt = params["temporal"]["dt"]  # Stored once, used everywhere
```

---

#### **Bug #3**: Data-Dependent Time Grid

**Cause**: Simulated data generation checked experimental data:
```python
if data is not None and "c2_exp" in data:
    n_time_points = c2_exp.shape[-1]  # ❌ WRONG LOGIC
```

**Philosophy violation**: Simulated data should be config-determined, NOT data-dependent

**Working version avoided this** by always using config:
```python
# simulation.py never checks experimental data size
n_time = end_frame - start_frame + 1  # Always from config
```

---

### 3.3 Why These Bugs Occurred Together

**Systemic Issue**: The architectural transition created vulnerability:

1. **Stateful → Stateless**: Lost centralized validation
2. **Eager → Lazy**: Errors discovered later in workflow
3. **Implicit → Explicit**: Requires discipline at every call site
4. **Fallbacks added**: Masked configuration errors

**Result**: Small oversights compounded into systematic failures

---

## Part 4: Current Status After Fixes

### 4.1 What Was Fixed ✅

1. ✅ **Time grid formula**: Now uses `dt * (n - 1)`
2. ✅ **dt parameter**: Added to `CombinedModel.compute_g2()` signature
3. ✅ **dt propagation**: Passed explicitly in `_plot_simulated_data()`
4. ✅ **Data independence**: Removed experimental data dependency
5. ✅ **Test coverage**: 10 new tests validating all fixes

---

### 4.2 What Remains Different (By Design)

| Aspect | Working Version | Current Version (After Fix) |
|--------|----------------|----------------------------|
| **Architecture** | Stateful OOP | Stateless Functional |
| **dt Storage** | Instance variable | Parameter passing |
| **Physics Factors** | Pre-computed (init) | Runtime computation |
| **Robustness** | High (early validation) | Medium (requires discipline) |
| **Flexibility** | Low (state-bound) | High (pure functions) |
| **JIT Compatibility** | Limited | Full (JAX optimized) |
| **Performance** | Fast (pre-computation) | Fast (JIT compensation) |

---

## Part 5: Recommendations

### 5.1 Short-term (Current Fixes) ✅

- [x] Fix time grid formula
- [x] Add dt parameter
- [x] Remove data dependency
- [x] Add comprehensive tests
- [x] Document architectural differences

---

### 5.2 Medium-term Improvements

#### **Option A**: Enforce dt Parameter (Recommended)

```python
def compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt):
    # Remove dt=None default - make it REQUIRED
    # Remove fallback estimation entirely

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
    sinc_prefactor = 0.5 / PI * q * L * dt

    return _compute_g2_scaled_core(...)
```

**Benefits**: Compile-time enforcement, no fallback risks

---

#### **Option B**: Pre-compute Physics Factors

```python
class PhysicsFactors:
    """Pre-computed physics factors for efficient computation."""

    def __init__(self, q: float, L: float, dt: float):
        self.wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
        self.sinc_prefactor = 0.5 / PI * q * L * dt

    def to_tuple(self):
        """Convert to tuple for JAX JIT compatibility."""
        return (self.wavevector_q_squared_half_dt, self.sinc_prefactor)

# Usage:
factors = PhysicsFactors(q, L, dt)
c2 = compute_g2_with_factors(params, t1, t2, phi, *factors.to_tuple(),
                             contrast, offset)
```

**Benefits**: Pre-computation efficiency + JAX compatibility

---

### 5.3 Long-term Architecture Evolution

#### **Hybrid Approach**: Stateful Wrapper + Functional Core

```python
class HomodyneModel:
    """Stateful wrapper for robustness."""

    def __init__(self, config):
        # Validate and store configuration
        self.dt = config["analyzer_parameters"]["temporal"]["dt"]
        self.q = config["analyzer_parameters"]["scattering"]["wavevector_q"]
        self.L = config["analyzer_parameters"]["geometry"]["stator_rotor_gap"]

        # Pre-compute physics factors
        self.physics_factors = PhysicsFactors(self.q, self.L, self.dt)

        # Create time array
        n = config["end_frame"] - config["start_frame"] + 1
        self.time_array = jnp.linspace(0, self.dt * (n - 1), n)

    def compute_c2(self, params, phi_angles):
        """High-level method using stored configuration."""
        t1, t2 = jnp.meshgrid(self.time_array, self.time_array, indexing="ij")

        # Call functional core with pre-computed factors
        return compute_g2_scaled_functional(
            params, t1, t2, phi_angles,
            *self.physics_factors.to_tuple(),
            contrast=0.5, offset=1.0
        )

# Functional core (JIT-compilable)
@jit
def compute_g2_scaled_functional(params, t1, t2, phi,
                                wavevector_q_squared_half_dt,
                                sinc_prefactor, contrast, offset):
    """Pure function suitable for JAX JIT."""
    return _compute_g2_scaled_core(params, t1, t2, phi,
                                   wavevector_q_squared_half_dt,
                                   sinc_prefactor, contrast, offset)
```

**Benefits**:
- ✅ Robustness of stateful design
- ✅ Performance of pre-computation
- ✅ JIT compatibility of functional core
- ✅ Best of both architectures

---

## Part 6: Lessons Learned

### 6.1 Architecture Transitions

**Lesson**: When transitioning between architectural paradigms:
1. **Complete the transition**: Don't mix paradigms inconsistently
2. **Remove safety nets**: Fallbacks can mask deeper issues
3. **Enforce discipline**: Use type systems, validators, tests
4. **Document differences**: Make design decisions explicit

---

### 6.2 Configuration Management

**Lesson**: Configuration should be:
1. **Single source of truth**: Never mix config with data
2. **Validated early**: Catch errors at load time
3. **Immutable after load**: Prevent accidental modifications
4. **Explicitly propagated**: No implicit assumptions

---

### 6.3 Testing Strategy

**Lesson**: Test architectural boundaries:
1. **Unit tests**: Individual function correctness
2. **Integration tests**: Cross-boundary interactions
3. **Regression tests**: Prevent known bugs
4. **Property tests**: Mathematical invariants (e.g., `t[i] = dt * i`)

---

## Conclusion

### Current Status: ✅ **FUNCTIONALLY CORRECT**

After the October 14, 2025 fixes:
- ✅ All critical bugs fixed
- ✅ Time grid generation correct
- ✅ dt parameter propagation working
- ✅ Simulated data independent of experimental data
- ✅ Comprehensive test coverage (10 new tests)

---

### Architectural Status: ⚠️ **DIFFERENT BY DESIGN**

The current codebase and working version have fundamentally different architectures:
- **Working Version**: Stateful, robust, eager pre-computation
- **Current Version**: Stateless, flexible, lazy computation

**Both are valid designs** with different trade-offs:
- Working version: Better for **robustness**
- Current version: Better for **JAX optimization**

---

### Recommendation: ✅ **MERGE FIXES + CONSIDER HYBRID**

1. **Immediate**: Merge the bug fixes (production-ready)
2. **Short-term**: Consider removing dt fallback for safety
3. **Long-term**: Evaluate hybrid architecture for best of both worlds

---

## Appendix: File References

### Working Version (homodyne-analysis)
- `homodyne/analysis/core.py:230-270` - Parameter initialization
- `homodyne/analysis/core.py:1110-1182` - C2 calculation
- `homodyne/cli/simulation.py:271-305` - Time array generation
- `homodyne/cli/simulation.py:308-353` - Theoretical C2 generation

### Current Version (homodyne v2)
- `homodyne/cli/commands.py:1374-1395` - Time array generation (FIXED)
- `homodyne/cli/commands.py:1424-1434` - C2 calculation call (FIXED)
- `homodyne/core/models.py:347-389` - compute_g2 signature (FIXED)
- `homodyne/core/jax_backend.py:891-948` - C2 computation core
- `tests/unit/test_simulated_data_fixes.py` - New validation tests

---

**Report Generated**: October 14, 2025
**Author**: Claude (via ultrathink analysis)
**Status**: Complete and validated ✅
