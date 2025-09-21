# Per-Angle Noise Model - Detailed Implementation in Homodyne v2

## Overview

The **per_angle noise model** is designed for anisotropic XPCS data where noise characteristics vary across different scattering angles (φ). Unlike the hierarchical model that assumes uniform noise across all angles, the per_angle model estimates independent noise parameters for each angle, capturing angular-dependent variations in data quality.

## Mathematical Formulation

The per_angle model assigns independent noise to each φ angle:

```
σ_i ~ Gamma(α=2.0, β=20.0)  for i = 1, ..., n_phi

Likelihood:
For each angle i:
    data[i] ~ Normal(theory[i], σ_i)
```

Where:
- `n_phi`: Number of phi angles in the dataset
- `σ_i`: Independent noise parameter for angle i
- Each σ_i has the same prior but is estimated independently

## Implementation Details

### 1. Core Model Definition
**File**: `homodyne/optimization/hybrid_noise_models.py`

#### VI (Variational Inference) Implementation
```python
elif noise_type == "per_angle":
    # Different noise for each phi angle
    n_phi = len(phi)
    sigma_angles = sample("sigma_angles",
                         dist.Gamma(concentration=2.0, rate=20.0),
                         sample_shape=(n_phi,))

    # Per-angle likelihood
    for i in range(n_phi):
        if data.ndim == 3:
            angle_data = data[i].flatten()
        else:
            # Handle different data shapes
            n_points_per_angle = data.size // n_phi
            start_idx = i * n_points_per_angle
            end_idx = (i + 1) * n_points_per_angle
            angle_data = data.flatten()[start_idx:end_idx]

        angle_centered = angle_data - jnp.mean(angle_data)
        sample(f"obs_angle_{i}",
              dist.Normal(0.0, sigma_angles[i]),
              obs=angle_centered)
```

Key features:
- Creates array of `n_phi` noise parameters
- Each angle gets its own likelihood term
- Handles both 3D data `[n_phi, n_t1, n_t2]` and flattened formats

#### MCMC Implementation
```python
elif noise_type == "per_angle":
    n_phi = len(phi)
    sigma = sample("sigma_angles",
                  dist.Gamma(concentration=2.0, rate=20.0),
                  sample_shape=(n_phi,))

    # In likelihood section:
    elif noise_type == "per_angle":
        # Per-angle noise
        n_phi = len(phi)
        if data.ndim == 3:
            # Standard [n_phi, n_t1, n_t2] format
            for i in range(n_phi):
                sample(f"obs_angle_{i}",
                      dist.Normal(theory_fitted[i], sigma[i]),
                      obs=data[i])
        else:
            # Flattened data - need to reshape appropriately
            data_reshaped = data.reshape((n_phi, -1))
            theory_reshaped = theory_fitted.reshape((n_phi, -1))
            for i in range(n_phi):
                sample(f"obs_angle_{i}",
                      dist.Normal(theory_reshaped[i], sigma[i]),
                      obs=data_reshaped[i])
```

### 2. Parameter Structure

The per_angle model creates:
- **Number of parameters**: `n_phi` (one per angle)
- **Parameter names**: `sigma_angles[0]`, `sigma_angles[1]`, ..., `sigma_angles[n_phi-1]`
- **Prior**: Each σ_i ~ Gamma(2.0, 20.0)
  - Prior mean ≈ 0.1
  - Prior variance ≈ 0.005

### 3. Data Handling
**File**: `homodyne/optimization/hybrid_noise_estimation.py`

The model intelligently handles different data formats:

```python
# For 3D data [n_phi, n_t1, n_t2]:
for i in range(n_phi):
    angle_data = data[i]  # Extract 2D matrix for angle i

# For flattened data:
n_points_per_angle = total_points // n_phi
for i in range(n_phi):
    start = i * n_points_per_angle
    end = (i + 1) * n_points_per_angle
    angle_data = data_flat[start:end]
```

### 4. Parameter Extraction and Results

```python
elif noise_type == "per_angle":
    sigma_samples = posterior_samples["sigma_angles"]
    sigma_mean = np.array(jnp.mean(sigma_samples, axis=0))  # Array of n_phi values
    sigma_std = np.array(jnp.std(sigma_samples, axis=0))    # Array of n_phi uncertainties
```

Results structure:
```python
{
    "noise_type": "per_angle",
    "sigma_mean": [0.05, 0.07, 0.04, ...],  # n_phi values
    "sigma_std": [0.005, 0.008, 0.003, ...], # n_phi uncertainties
    "n_angles": 23,
    "angle_values": [0, 8, 16, ...],  # Actual phi angles in degrees
}
```

### 5. Configuration

**File**: `homodyne/config/templates/homodyne_static_anisotropic.yaml`

```yaml
noise_estimation:
  model: per_angle  # Can be "hierarchical", "per_angle", or "adaptive"

  per_angle:
    min_angles_required: 2    # Minimum angles for meaningful estimation
    validate_coverage: true   # Check angle coverage quality
    adam_steps: 500          # More steps than hierarchical (250)

  adam_config:
    learning_rate: 0.01
    convergence_threshold: 1e-6
    max_epochs: 1000
    early_stopping: true
```

### 6. Usage via CLI

```bash
# VI with per-angle noise estimation
homodyne --method vi --estimate-noise --noise-model per_angle config.yaml

# MCMC with per-angle noise
homodyne --method mcmc --estimate-noise --noise-model per_angle config.yaml

# Hybrid method with per-angle noise
homodyne --method hybrid --estimate-noise --noise-model per_angle config.yaml
```

## Advantages of Per-Angle Model

1. **Anisotropic Noise Capture**: Accounts for angle-dependent data quality variations
2. **Detector Artifacts**: Handles detector regions with different noise characteristics
3. **Flow Direction Effects**: Captures noise variations due to flow direction in laminar experiments
4. **Better Uncertainty**: More accurate uncertainty quantification for anisotropic systems

## When to Use Per-Angle Model

### Recommended For:
- **Anisotropic scattering**: When signal varies significantly with angle
- **Multi-detector regions**: Different detector areas have different noise
- **Flow experiments**: Noise varies with flow direction
- **High-quality analysis**: When accurate per-angle uncertainty is critical

### Not Recommended For:
- **Limited angles**: Fewer than 4-5 angles (insufficient statistics)
- **Isotropic systems**: When noise is truly uniform
- **Quick analysis**: Adds computational overhead (n_phi parameters vs 1)
- **Low SNR data**: May overfit to noise with limited signal

## Computational Considerations

### Performance Impact:
- **Memory**: Scales with n_phi (typically 10-50 angles)
- **VI**: ~50-100% slower than hierarchical (500 vs 250 Adam steps)
- **MCMC**: ~30-50% slower due to more parameters
- **Storage**: n_phi times more noise parameters to store

### Optimization Settings:
- **Adam steps**: 500 (optimal for multi-parameter estimation)
- **Learning rate**: 0.01 (same as other models)
- **Early stopping**: Enabled with patience=50 steps
- **Convergence threshold**: 1e-6

## Model Comparison

| Aspect | Hierarchical | Per-Angle | Adaptive |
|--------|-------------|-----------|----------|
| Parameters | 1 (σ) | n_phi (σ_i) | 2 (σ_base, σ_scale) |
| Complexity | Low | Medium | High |
| Angular variation | No | Yes | No |
| Signal-dependent | No | No | Yes |
| Best for | Isotropic | Anisotropic | Heteroscedastic |
| Adam steps | 250 | 500 | 750 |

## Validation and Quality Control

The per_angle model includes specific validation:

1. **Angle Coverage Check**:
   ```python
   if n_phi < min_angles_required:
       raise ValueError(f"Need at least {min_angles_required} angles")
   ```

2. **Noise Consistency Check**:
   - Compares noise levels across angles
   - Warns if variation is extreme (>10x difference)
   - Suggests hierarchical model if variation is minimal

3. **Convergence Monitoring**:
   - Tracks individual σ_i convergence
   - Ensures all angles converge properly

## Output Interpretation

Example output for per_angle noise estimation:

```python
{
    "noise_type": "per_angle",
    "n_angles": 23,
    "sigma_mean": {
        "angle_0": 0.045,    # φ = 0°
        "angle_1": 0.052,    # φ = 8°
        "angle_2": 0.048,    # φ = 16°
        ...
        "angle_22": 0.051    # φ = 352°
    },
    "sigma_std": {
        "angle_0": 0.004,
        "angle_1": 0.005,
        ...
    },
    "statistics": {
        "mean_noise": 0.049,
        "std_noise": 0.003,
        "min_noise": 0.045,
        "max_noise": 0.052,
        "anisotropy_ratio": 1.16  # max/min ratio
    }
}
```

## Integration with Physics Models

The per_angle model seamlessly integrates with all analysis modes:

1. **Static Isotropic**: Usually not needed (isotropic by definition)
2. **Static Anisotropic**: Natural fit for angle-dependent analysis
3. **Laminar Flow**: Captures flow-induced noise anisotropy

## Advanced Features

### 1. Automatic Angle Grouping
For datasets with many angles (>50), the system can group nearby angles:
```python
# Groups angles within 10° into bins
grouped_sigma = group_angles(sigma_angles, phi, bin_size=10.0)
```

### 2. Angular Interpolation
For visualization, noise can be interpolated between measured angles:
```python
# Smooth interpolation for plotting
sigma_interpolated = interpolate_angular_noise(sigma_angles, phi, n_points=360)
```

### 3. Correlation Analysis
The model can detect correlations between noise and physics parameters:
```python
# Check if noise correlates with flow direction
correlation = compute_noise_flow_correlation(sigma_angles, phi, flow_params)
```

## Physical Motivation

The per_angle model reflects real experimental conditions:

1. **Detector Inhomogeneity**: Different pixel regions have different characteristics
2. **Beam Profile**: X-ray beam intensity varies across detector
3. **Sample Anisotropy**: Oriented samples scatter differently in different directions
4. **Flow Effects**: Shear flow creates direction-dependent dynamics

## Summary

The per_angle noise model provides:
- Independent noise estimation for each scattering angle
- Optimal for anisotropic XPCS experiments
- Medium complexity between hierarchical and adaptive models
- Essential for accurate uncertainty in angle-resolved analysis

It represents the standard choice for experiments where angular resolution is important and noise characteristics vary with scattering direction.