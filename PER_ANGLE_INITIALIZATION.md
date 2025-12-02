# Per-Angle Parameter Initialization in NUTS and CMC

**Date**: 2025-12-02
**Issue**: How are contrast/offset initialized for each phi angle in per-angle scaling mode?
**Applies to**: Both NUTS and CMC

---

## Overview

When `per_angle_scaling=True` (mandatory in v2.4.0+), each unique phi angle gets its own contrast and offset parameters. For example, with 3 unique angles:

**Scalar mode (legacy)**:
```python
parameters = ['contrast', 'offset', 'D0', 'alpha', 'D_offset']
# 5 parameters total
```

**Per-angle mode (v2.4.0+)**:
```python
parameters = [
    'contrast_0', 'contrast_1', 'contrast_2',  # One per angle
    'offset_0', 'offset_1', 'offset_2',        # One per angle
    'D0', 'alpha', 'D_offset'                  # Physical params (shared)
]
# 9 parameters total (6 scaling + 3 physical)
```

---

## User's Question

**Question**: "How are the contrasts and offsets of each phi angle initiated for per_angle_scaling in both NUTS and CMC?"

**Answer**: Both NUTS and CMC use the same fallback mechanism:

1. **If user provides `initial_values`** (e.g., `{'contrast': 0.05, 'offset': 1.0}`):
   - Expand to all angles: `{'contrast_0': 0.05, 'contrast_1': 0.05, 'contrast_2': 0.05, ...}`
   - Use config value for all angles

2. **If user does NOT provide `initial_values`**:
   - Use parameter bounds midpoint as fallback
   - For `contrast`: `(bounds[0] + bounds[1]) / 2.0` → typically `(0 + 1) / 2 = 0.5`
   - For `offset`: `(bounds[0] + bounds[1]) / 2.0` → typically `(0.5 + 1.5) / 2 = 1.0`

---

## NUTS Per-Angle Expansion

**File**: `homodyne/optimization/mcmc.py:3365-3414`

### Code Implementation

```python
if per_angle_scaling and initial_values is not None and parameter_space is not None:
    logger.info(
        f"Expanding initial_values for per-angle scaling: {list(initial_values.keys())}"
    )

    # Determine number of unique phi angles
    if phi_unique is not None:
        n_phi = len(phi_unique)
    else:
        # Infer from existing per-angle keys
        inferred = [
            name
            for name in initial_values
            if name.startswith("contrast_") or name.startswith("offset_")
        ]
        n_phi = len({name.split("_")[-1] for name in inferred if name.count("_") == 1})

    if n_phi > 0:
        # Expand for each phi angle
        for phi_idx in range(n_phi):
            contrast_key = f"contrast_{phi_idx}"
            offset_key = f"offset_{phi_idx}"

            # Expand contrast
            if contrast_key not in initial_values:
                try:
                    # Use parameter bounds midpoint as fallback
                    contrast_midpoint = sum(parameter_space.get_bounds("contrast")) / 2.0
                    initial_values[contrast_key] = parameter_space.clamp_to_open_interval(
                        "contrast", contrast_midpoint, epsilon=1e-6
                    )
                except KeyError:
                    # Last resort default
                    initial_values[contrast_key] = 0.5

            # Expand offset
            if offset_key not in initial_values:
                try:
                    # Use parameter bounds midpoint as fallback
                    offset_midpoint = sum(parameter_space.get_bounds("offset")) / 2.0
                    initial_values[offset_key] = parameter_space.clamp_to_open_interval(
                        "offset", offset_midpoint, epsilon=1e-6
                    )
                except KeyError:
                    # Last resort default
                    initial_values[offset_key] = 1.0

        # Remove base contrast/offset entries after expansion
        removed = False
        if initial_values.pop("contrast", None) is not None:
            removed = True
        if initial_values.pop("offset", None) is not None:
            removed = True

        if removed:
            logger.info("Removed base contrast/offset entries after per-angle expansion")
```

### Behavior Example

**Input** (from config):
```python
initial_values = {
    'contrast': 0.05015,  # Scalar value from config
    'offset': 1.001,      # Scalar value from config
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026
}
```

**After NUTS expansion** (3 angles):
```python
initial_values = {
    'contrast_0': 0.05015,  # Expanded from 'contrast'
    'contrast_1': 0.05015,  # Expanded from 'contrast'
    'contrast_2': 0.05015,  # Expanded from 'contrast'
    'offset_0': 1.001,      # Expanded from 'offset'
    'offset_1': 1.001,      # Expanded from 'offset'
    'offset_2': 1.001,      # Expanded from 'offset'
    'D0': 16830.0,          # Physical param (unchanged)
    'alpha': -1.571,        # Physical param (unchanged)
    'D_offset': 3.026       # Physical param (unchanged)
}
# Note: 'contrast' and 'offset' keys removed after expansion
```

**Fallback behavior** (no contrast/offset in config):
```python
# If initial_values = {'D0': 16830.0, 'alpha': -1.571, 'D_offset': 3.026}
# (no contrast/offset provided)

# NUTS uses parameter_space.get_bounds("contrast") → [0.0, 1.0]
contrast_midpoint = (0.0 + 1.0) / 2.0 = 0.5

# NUTS uses parameter_space.get_bounds("offset") → [0.5, 1.5]
offset_midpoint = (0.5 + 1.5) / 2.0 = 1.0

# After expansion:
initial_values = {
    'contrast_0': 0.5,    # Midpoint fallback
    'contrast_1': 0.5,    # Midpoint fallback
    'contrast_2': 0.5,    # Midpoint fallback
    'offset_0': 1.0,      # Midpoint fallback
    'offset_1': 1.0,      # Midpoint fallback
    'offset_2': 1.0,      # Midpoint fallback
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026
}
```

---

## CMC Per-Angle Expansion

**File**: `homodyne/optimization/cmc/coordinator.py:485-549`

### Code Implementation

```python
# NumPyro's init_to_value() strategy requires parameters in the EXACT ORDER
# that the model samples them. For per-angle scaling, the model samples:
#   1. contrast_0, contrast_1, ..., contrast_{n_phi-1}
#   2. offset_0, offset_1, ..., offset_{n_phi-1}
#   3. Physical parameters (D0, alpha, D_offset, ...)

# Determine number of unique phi angles
phi_unique = np.unique(np.asarray(phi))
n_phi = len(phi_unique)

logger.info(f"Expanding initial_values for {n_phi} unique phi angles (per-angle scaling)")

# Get default scaling values from config (will be removed from physical params)
default_contrast = initial_values.get("contrast", 0.5)
default_offset = initial_values.get("offset", 1.0)

# Create NEW dict with CORRECT ORDER: per-angle params FIRST, then physical params
init_params = {}

# STEP 1: Add per-angle contrast parameters FIRST (contrast_0, contrast_1, ...)
for phi_idx in range(n_phi):
    contrast_key = f"contrast_{phi_idx}"
    try:
        # Use parameter bounds midpoint as fallback
        contrast_midpoint = sum(parameter_space.get_bounds("contrast")) / 2.0
        init_params[contrast_key] = parameter_space.clamp_to_open_interval(
            "contrast", contrast_midpoint, epsilon=1e-6
        )
    except (KeyError, AttributeError):
        # Use config value or last resort default
        init_params[contrast_key] = default_contrast

# STEP 2: Add per-angle offset parameters SECOND (offset_0, offset_1, ...)
for phi_idx in range(n_phi):
    offset_key = f"offset_{phi_idx}"
    try:
        # Use parameter bounds midpoint as fallback
        offset_midpoint = sum(parameter_space.get_bounds("offset")) / 2.0
        init_params[offset_key] = parameter_space.clamp_to_open_interval(
            "offset", offset_midpoint, epsilon=1e-6
        )
    except (KeyError, AttributeError):
        # Use config value or last resort default
        init_params[offset_key] = default_offset

# STEP 3: Add physical parameters LAST (D0, alpha, D_offset, ...)
for param_name, param_value in initial_values.items():
    if param_name not in ["contrast", "offset"]:
        init_params[param_name] = param_value

logger.debug(
    f"Expanded init_params for CMC: {list(init_params.keys())}"
)
```

### Behavior Example

**Input** (same as NUTS):
```python
initial_values = {
    'contrast': 0.05015,
    'offset': 1.001,
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026
}
```

**After CMC expansion** (3 angles):
```python
init_params = {
    # STEP 1: Per-angle contrast (FIRST)
    'contrast_0': 0.05015,  # From default_contrast (extracted from initial_values)
    'contrast_1': 0.05015,
    'contrast_2': 0.05015,

    # STEP 2: Per-angle offset (SECOND)
    'offset_0': 1.001,      # From default_offset (extracted from initial_values)
    'offset_1': 1.001,
    'offset_2': 1.001,

    # STEP 3: Physical params (LAST)
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026
}
```

**CRITICAL DIFFERENCE**: CMC creates a NEW dict (`init_params`) with strict ordering, while NUTS modifies the existing dict in place.

**Fallback behavior** (no contrast/offset in config):
```python
# If initial_values = {'D0': 16830.0, 'alpha': -1.571, 'D_offset': 3.026}

# default_contrast = initial_values.get("contrast", 0.5) → 0.5 (fallback)
# default_offset = initial_values.get("offset", 1.0) → 1.0 (fallback)

# CMC tries parameter_space.get_bounds() first:
contrast_midpoint = (0.0 + 1.0) / 2.0 = 0.5
offset_midpoint = (0.5 + 1.5) / 2.0 = 1.0

# After expansion (same as NUTS):
init_params = {
    'contrast_0': 0.5,
    'contrast_1': 0.5,
    'contrast_2': 0.5,
    'offset_0': 1.0,
    'offset_1': 1.0,
    'offset_2': 1.0,
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026
}
```

---

## Key Differences: NUTS vs CMC

| Aspect | NUTS | CMC |
|--------|------|-----|
| **File** | `mcmc.py:3365-3414` | `coordinator.py:485-549` |
| **Modification** | In-place (modifies `initial_values`) | New dict (`init_params`) |
| **Ordering** | No strict requirement | **STRICT** ordering (per-angle first) |
| **Fallback priority** | 1. Bounds midpoint<br>2. Default (0.5, 1.0) | 1. Bounds midpoint<br>2. Config value<br>3. Default (0.5, 1.0) |
| **Scalar removal** | Removes `contrast`, `offset` after expansion | Never adds them (creates new dict) |
| **Logging** | "Expanding initial_values for per-angle scaling" | "Expanding initial_values for N unique phi angles" |

---

## Parameter Ordering Requirement (CMC Critical)

**Why CMC requires strict ordering**:

NumPyro's `init_to_value()` initialization strategy uses **positional matching** based on the order that the model samples parameters. If the order is wrong, NumPyro assigns wrong values (e.g., `D0` → `contrast_0`).

**NumPyro model sampling order** (`mcmc.py:2581-2618`):
```python
# STEP 1: Sample per-angle contrast (if per_angle_scaling=True)
for phi_idx in range(n_phi):
    contrast_val = sample(f"contrast_{phi_idx}", TruncatedNormal(...))

# STEP 2: Sample per-angle offset (if per_angle_scaling=True)
for phi_idx in range(n_phi):
    offset_val = sample(f"offset_{phi_idx}", TruncatedNormal(...))

# STEP 3: Sample physical parameters
D0 = sample("D0", TruncatedNormal(...))
alpha = sample("alpha", TruncatedNormal(...))
D_offset = sample("D_offset", TruncatedNormal(...))
```

**CMC coordinator ensures this order** (lines 485-549):
1. `contrast_0, contrast_1, contrast_2` (first)
2. `offset_0, offset_1, offset_2` (second)
3. `D0, alpha, D_offset` (last)

**Validation in worker** (`backends/multiprocessing.py:875-913`):
```python
# Worker validates parameter ordering at runtime
expected_order = [f"contrast_{i}" for i in range(n_phi)] + \
                 [f"offset_{i}" for i in range(n_phi)] + \
                 [param for param in init_params if param not in scaling_params]

if list(init_params.keys()) != expected_order:
    raise ValueError(
        f"Parameter ordering mismatch! Expected: {expected_order}, "
        f"Got: {list(init_params.keys())}"
    )
```

---

## Fallback Mechanism: Parameter Bounds Midpoint

**Function**: `parameter_space.get_bounds(param_name)`

**Returns**: Tuple `(lower_bound, upper_bound)` for the parameter

**Example bounds** (from `config/parameter_space.py`):
```python
PARAMETER_BOUNDS = {
    'contrast': (0.0, 1.0),              # Midpoint: 0.5
    'offset': (0.5, 1.5),                # Midpoint: 1.0
    'D0': (1000.0, 50000.0),             # Midpoint: 25500.0
    'alpha': (-3.14159, 3.14159),        # Midpoint: 0.0
    'D_offset': (-10.0, 10.0),           # Midpoint: 0.0
}
```

**Clamping to open interval** (`parameter_space.clamp_to_open_interval`):
```python
# Ensures value is STRICTLY within bounds (not on boundary)
# For TruncatedNormal support (doesn't like boundary values)

contrast_midpoint = 0.5  # Exactly on midpoint
clamped = clamp_to_open_interval("contrast", 0.5, epsilon=1e-6)
# Returns: 0.500001 (slightly inside lower bound)
```

**Why use midpoint**:
- Neutral starting point (not biased towards either bound)
- Reasonable default when user doesn't provide initial guess
- Compatible with TruncatedNormal priors (inside support)

---

## Integration with Single-Angle Initialization

**Relationship**: Per-angle expansion happens AFTER single-angle initialization logic.

**Flow** (for single-angle data with per-angle scaling):

```
1. Single-angle detection (mcmc.py:1459-1523)
   ↓
   Check if contrast/offset in initial_values
   ↓
   If yes: Use config values (0.05015, 1.001)
   If no: Use data-derived (0.0406, 1.0002)
   ↓
   initial_values = {'contrast': 0.05015, 'offset': 1.001, 'D0': ..., 'alpha': ...}

2. Per-angle expansion (mcmc.py:3365-3414 or coordinator.py:485-549)
   ↓
   Detect n_phi = 1 (single angle)
   ↓
   Expand: {'contrast': 0.05015} → {'contrast_0': 0.05015}
   Expand: {'offset': 1.001} → {'offset_0': 1.001}
   ↓
   Remove base keys: 'contrast', 'offset'
   ↓
   initial_values = {'contrast_0': 0.05015, 'offset_0': 1.001, 'D0': ..., 'alpha': ...}

3. Pass to NumPyro sampler
   ↓
   NUTS initialization: init_params={'contrast_0': [0.05015, 0.05015, 0.05015, 0.05015]}
   (4 chains, each starting from 0.05015)
```

**Key insight**: Even for single-angle data, per-angle expansion creates `contrast_0`, `offset_0` (not scalar `contrast`, `offset`). This ensures consistent parameter naming across all phi angle counts.

---

## Example Walkthrough

### Scenario: 3-Angle Laminar Flow Analysis

**Config YAML**:
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset', 'gamma_dot_t0', 'beta',
                    'gamma_dot_t_offset', 'phi0', 'contrast', 'offset']
  values: [16830.0, -1.571, 3.026, 100.0, 0.5, 10.0, 1.57, 0.05015, 1.001]
```

**Step-by-step**:

**1. Load from config**:
```python
initial_values = config.get_initial_parameters()
# {
#   'D0': 16830.0,
#   'alpha': -1.571,
#   'D_offset': 3.026,
#   'gamma_dot_t0': 100.0,
#   'beta': 0.5,
#   'gamma_dot_t_offset': 10.0,
#   'phi0': 1.57,
#   'contrast': 0.05015,
#   'offset': 1.001
# }
```

**2. Detect 3 unique phi angles**:
```python
phi_unique = np.unique(phi)  # [0.0, 1.047, 2.094]
n_phi = 3
```

**3. Per-angle expansion (NUTS)**:
```python
# For each phi_idx in [0, 1, 2]:
for phi_idx in range(3):
    # contrast_0, contrast_1, contrast_2
    initial_values[f"contrast_{phi_idx}"] = 0.05015  # From config

    # offset_0, offset_1, offset_2
    initial_values[f"offset_{phi_idx}"] = 1.001      # From config

# Remove base keys
initial_values.pop("contrast")
initial_values.pop("offset")
```

**4. Final initial_values (NUTS)**:
```python
{
    'contrast_0': 0.05015,
    'contrast_1': 0.05015,
    'contrast_2': 0.05015,
    'offset_0': 1.001,
    'offset_1': 1.001,
    'offset_2': 1.001,
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026,
    'gamma_dot_t0': 100.0,
    'beta': 0.5,
    'gamma_dot_t_offset': 10.0,
    'phi0': 1.57
}
# 13 parameters total (6 scaling + 7 physical)
```

**5. Final init_params (CMC - ORDERED)**:
```python
{
    # Per-angle contrast (FIRST)
    'contrast_0': 0.05015,
    'contrast_1': 0.05015,
    'contrast_2': 0.05015,

    # Per-angle offset (SECOND)
    'offset_0': 1.001,
    'offset_1': 1.001,
    'offset_2': 1.001,

    # Physical params (LAST)
    'D0': 16830.0,
    'alpha': -1.571,
    'D_offset': 3.026,
    'gamma_dot_t0': 100.0,
    'beta': 0.5,
    'gamma_dot_t_offset': 10.0,
    'phi0': 1.57
}
```

**6. NUTS initialization** (4 chains):
```python
formatted_initial_values = {
    'contrast_0': jnp.array([0.05015, 0.05015, 0.05015, 0.05015]),
    'contrast_1': jnp.array([0.05015, 0.05015, 0.05015, 0.05015]),
    'contrast_2': jnp.array([0.05015, 0.05015, 0.05015, 0.05015]),
    'offset_0': jnp.array([1.001, 1.001, 1.001, 1.001]),
    'offset_1': jnp.array([1.001, 1.001, 1.001, 1.001]),
    'offset_2': jnp.array([1.001, 1.001, 1.001, 1.001]),
    'D0': jnp.array([16830.0, 16830.0, 16830.0, 16830.0]),
    # ... other params
}

# Pass to NUTS
mcmc.run(rng_key, init_params=formatted_initial_values)
```

---

## Summary

### How Per-Angle Initialization Works

**Both NUTS and CMC**:
1. Detect number of unique phi angles (`n_phi`)
2. For each `phi_idx` in `[0, 1, ..., n_phi-1]`:
   - Create `contrast_{phi_idx}` parameter
   - Create `offset_{phi_idx}` parameter
3. Use config value if provided, otherwise use parameter bounds midpoint
4. Remove scalar `contrast`, `offset` keys (NUTS) or never add them (CMC)

**Key differences**:
- **NUTS**: Modifies dict in place, no strict ordering
- **CMC**: Creates new ordered dict (per-angle params first, then physical)

**Fallback priority**:
1. Config `initial_values` (e.g., `{'contrast': 0.05015}`)
2. Parameter bounds midpoint (e.g., `(0 + 1) / 2 = 0.5`)
3. Last resort defaults (0.5 for contrast, 1.0 for offset)

**Integration with single-angle logic**:
- Single-angle initialization sets `initial_values['contrast']`, `initial_values['offset']`
- Per-angle expansion then creates `contrast_0`, `offset_0` from these values
- Works seamlessly for both config-provided and data-derived values

**Critical for CMC**: Parameter ordering MUST match NumPyro model sampling order, otherwise initialization fails with "Cannot find valid initial parameters" error.

---

## Code References

### NUTS Per-Angle Expansion
- **File**: `homodyne/optimization/mcmc.py`
- **Lines**: 3365-3414
- **Function**: Part of `fit_mcmc_jax()` preprocessing

### CMC Per-Angle Expansion
- **File**: `homodyne/optimization/cmc/coordinator.py`
- **Lines**: 485-549
- **Function**: `CMCCoordinator._run_inference()`

### Parameter Bounds
- **File**: `homodyne/config/parameter_space.py`
- **Method**: `ParameterSpace.get_bounds(param_name)`
- **Method**: `ParameterSpace.clamp_to_open_interval(param_name, value, epsilon)`

### NumPyro Model Sampling Order
- **File**: `homodyne/optimization/mcmc.py`
- **Lines**: 2581-2618 (per-angle contrast/offset iteration)
- **Function**: `_create_numpyro_model()` inner function

### CMC Worker Validation
- **File**: `homodyne/optimization/cmc/backends/multiprocessing.py`
- **Lines**: 875-913
- **Function**: `CMCWorker._validate_init_params()`
