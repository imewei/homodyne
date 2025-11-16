# Per-Angle Contrast and Offset Scaling Guide

## Overview

Starting from v2.2.0 (unreleased), Homodyne implements **per-angle scaling** where each scattering angle (φ) has independent contrast and offset parameters. This is the **physically correct behavior** and is now the default mode for both MCMC and NLSQ analyses.

### Why Per-Angle Scaling?

Different scattering angles can have different optical properties:

1. **Detector Response Variation**: Detector sensitivity varies across the detector surface
2. **Optical Path Differences**: Different angles have different beam paths through optics
3. **Sample Heterogeneity**: Different q-vectors probe different length scales
4. **Instrumental Artifacts**: Angle-dependent systematic effects

## Quick Start

### Default Behavior (Recommended)

Both MCMC and NLSQ now use per-angle scaling by default:

```python
from homodyne.optimization import fit_mcmc_jax, fit_nlsq_jax
from homodyne.config.manager import ConfigManager

# Load configuration
config = ConfigManager("config.yaml")

# Run NLSQ optimization (per-angle by default)
nlsq_result = fit_nlsq_jax(data, config)

# Run MCMC analysis (per-angle by default)
mcmc_result = fit_mcmc_jax(data, config)
```

### Understanding the Results

With per-angle scaling, you get separate contrast and offset for each φ angle:

```python
# NLSQ results
n_phi = len(data["phi_angles"])
for i in range(n_phi):
    contrast_i = nlsq_result.parameters[i]  # contrast_0, contrast_1, ...
    offset_i = nlsq_result.parameters[n_phi + i]  # offset_0, offset_1, ...
    print(f"Angle {i}: contrast={contrast_i:.3f}, offset={offset_i:.3f}")

# MCMC results
for i in range(n_phi):
    contrast_samples = mcmc_result.samples[f"contrast_{i}"]
    offset_samples = mcmc_result.samples[f"offset_{i}"]

    # Posterior statistics
    contrast_mean = contrast_samples.mean()
    contrast_std = contrast_samples.std()
    print(f"Angle {i}: contrast = {contrast_mean:.3f} ± {contrast_std:.3f}")
```

## Parameter Structure

### Static Isotropic Mode

**Total Parameters:** `2 × n_phi + 3`

For `n_phi = 3` angles:

```
Parameters (9 total):
├── Scaling (6):
│   ├── contrast_0  # Contrast for φ = 0°
│   ├── contrast_1  # Contrast for φ = 60°
│   ├── contrast_2  # Contrast for φ = 120°
│   ├── offset_0    # Offset for φ = 0°
│   ├── offset_1    # Offset for φ = 60°
│   └── offset_2    # Offset for φ = 120°
└── Physical (3):
    ├── D0          # Diffusion coefficient
    ├── alpha       # Anomalous exponent
    └── D_offset    # Baseline diffusion
```

### Laminar Flow Mode

**Total Parameters:** `2 × n_phi + 7`

For `n_phi = 3` angles:

```
Parameters (15 total):
├── Scaling (6):
│   ├── contrast_0, contrast_1, contrast_2
│   └── offset_0, offset_1, offset_2
└── Physical (7):
    ├── D0, alpha, D_offset         # Diffusion
    └── gamma_dot_t0, beta,         # Flow
        gamma_dot_t_offset, phi0
```

## Practical Examples

### Example 1: Basic MCMC with Per-Angle

```python
import numpy as np
from homodyne.optimization import fit_mcmc_jax
from homodyne.config.manager import ConfigManager
from homodyne.config.parameter_space import ParameterSpace

# Load data
config = ConfigManager("config.yaml")
data = load_xpcs_data("experiment.hdf")

# Run MCMC with per-angle scaling (default)
result = fit_mcmc_jax(
    data=data,
    config=config,
    num_warmup=500,
    num_samples=1000,
)

# Extract per-angle posteriors
n_phi = len(np.unique(data["phi"]))
for i in range(n_phi):
    contrast = result.samples[f"contrast_{i}"]
    offset = result.samples[f"offset_{i}"]

    # Plot posterior distributions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(contrast, bins=50, density=True, alpha=0.7)
    plt.xlabel(f"contrast_{i}")
    plt.ylabel("Density")
    plt.title(f"Phi Angle {i}")

    plt.subplot(1, 2, 2)
    plt.hist(offset, bins=50, density=True, alpha=0.7)
    plt.xlabel(f"offset_{i}")
    plt.ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"posterior_angle_{i}.png")
```

### Example 2: NLSQ with Per-Angle

```python
from homodyne.optimization import fit_nlsq_jax

# Run NLSQ optimization
result = fit_nlsq_jax(
    data=data,
    config=config,
    per_angle_scaling=True,  # Explicit (but this is the default)
)

# Extract results
n_phi = len(np.unique(data["phi"]))

# Scaling parameters
contrasts = result.parameters[:n_phi]
offsets = result.parameters[n_phi:2*n_phi]

# Physical parameters (after scaling params)
physical_params = result.parameters[2*n_phi:]

print("Per-Angle Contrasts:", contrasts)
print("Per-Angle Offsets:", offsets)
print("Physical Parameters:", physical_params)

# Compare scaling across angles
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(range(n_phi), contrasts)
plt.xlabel("Phi Angle Index")
plt.ylabel("Contrast")
plt.title("Contrast vs Angle")

plt.subplot(1, 3, 2)
plt.bar(range(n_phi), offsets)
plt.xlabel("Phi Angle Index")
plt.ylabel("Offset")
plt.title("Offset vs Angle")

plt.subplot(1, 3, 3)
plt.scatter(contrasts, offsets)
plt.xlabel("Contrast")
plt.ylabel("Offset")
plt.title("Contrast vs Offset")

plt.tight_layout()
plt.savefig("per_angle_scaling.png")
```

### Example 3: Detecting Angle-Dependent Artifacts

```python
# Use per-angle scaling to identify problematic angles
result = fit_nlsq_jax(data, config)

n_phi = len(np.unique(data["phi"]))
contrasts = result.parameters[:n_phi]
offsets = result.parameters[n_phi:2*n_phi]

# Identify outliers (angles with unusual scaling)
contrast_mean = np.mean(contrasts)
contrast_std = np.std(contrasts)

for i in range(n_phi):
    z_score = abs(contrasts[i] - contrast_mean) / contrast_std
    if z_score > 2.0:
        print(f"⚠️  Angle {i}: Unusual contrast (z={z_score:.2f})")
        print(f"   Consider excluding this angle or investigating detector issues")
```

## Legacy Mode (Backward Compatibility)

If you need the old behavior (single contrast/offset for all angles):

```python
# Explicitly disable per-angle scaling
result = fit_mcmc_jax(
    data=data,
    config=config,
    per_angle_scaling=False,  # Legacy mode
)

# Access global parameters
contrast = result.samples["contrast"]  # Single value
offset = result.samples["offset"]      # Single value
```

**Note:** Legacy mode is provided only for backward compatibility testing and is **not recommended for production analysis**.

## Migration from v2.1.0

### Step 1: Update Test Expectations

```python
# OLD (v2.1.0)
assert "contrast" in samples
assert "offset" in samples

# NEW (per-angle with n_phi=3)
assert "contrast_0" in samples
assert "contrast_1" in samples
assert "contrast_2" in samples
assert "offset_0" in samples
assert "offset_1" in samples
assert "offset_2" in samples
```

### Step 2: Update Result Processing

```python
# OLD (v2.1.0)
contrast = result.parameters[0]
offset = result.parameters[1]
physical = result.parameters[2:]

# NEW (per-angle with n_phi=3)
n_phi = 3
contrasts = result.parameters[:n_phi]        # [contrast_0, contrast_1, contrast_2]
offsets = result.parameters[n_phi:2*n_phi]   # [offset_0, offset_1, offset_2]
physical = result.parameters[2*n_phi:]       # [D0, alpha, D_offset, ...]
```

### Step 3: Update Plotting Code

```python
# OLD (v2.1.0): Plot single contrast
plt.hist(samples["contrast"], bins=50)

# NEW (per-angle): Plot each angle separately
for i in range(n_phi):
    plt.hist(samples[f"contrast_{i}"], bins=50, alpha=0.5, label=f"Angle {i}")
plt.legend()
```

## Frequently Asked Questions

### Q: Why did the default change?

**A:** Per-angle scaling is the **physically correct behavior**. Different scattering angles genuinely can have different optical properties and detector responses. Using a single global contrast/offset was an oversimplification that could lead to systematic errors.

### Q: Will this increase computation time?

**A:** Slightly, due to the increased number of parameters:
- **MCMC**: ~10-20% slower due to higher-dimensional sampling
- **NLSQ**: Minimal impact (trust-region handles extra parameters efficiently)

The improved physical accuracy is worth the small performance cost.

### Q: How do I know if my angles have different scaling?

**A:** Run the analysis and check:
1. Compare contrast/offset values across angles
2. If they're all similar (within uncertainties), angles are consistent
3. If they differ significantly, per-angle mode is capturing real physics

### Q: Can I constrain angles to have similar scaling?

**A:** Yes, use hierarchical priors in MCMC (advanced topic, see documentation).

### Q: What if I only have one phi angle?

**A:** Per-angle mode still works! You'll get `contrast_0` and `offset_0` instead of `contrast` and `offset`. The behavior is equivalent to legacy mode but with consistent naming.

## Best Practices

1. **Always use per-angle mode** (default) for production analyses
2. **Inspect per-angle results** to identify problematic angles
3. **Compare angles** to detect systematic effects
4. **Use legacy mode** only for backward compatibility testing
5. **Update tests and plotting code** to handle per-angle parameters

## Technical Details

### Implementation

- **MCMC**: Each angle samples contrast/offset independently via NumPyro
- **NLSQ**: JAX `vmap` applies correct scaling to each angle during optimization
- **Data structure**: `phi_full` array maps each data point to its angle index

### JAX Concretization Fix

The implementation pre-computes `phi_unique` before JIT tracing:

```python
# Pre-compute (concrete values)
phi_unique = np.unique(np.asarray(phi))
n_phi = len(phi_unique)

def model_function():
    # Use pre-computed values from closure
    # Avoid jnp.unique() inside JIT-traced function
    pass
```

This avoids `ConcretizationTypeError` during NumPyro model compilation.

## References

- **CHANGELOG.md**: Full breaking changes documentation
- **tests/unit/test_per_angle_scaling.py**: Comprehensive test suite
- **homodyne/optimization/mcmc.py:1347-1360**: JAX concretization fix
- **homodyne/optimization/nlsq_wrapper.py:1127**: NLSQ per-angle implementation

## Support

For questions or issues:
1. Check the CHANGELOG for migration examples
2. Review test files for usage patterns
3. Open an issue on GitHub with per-angle tag
