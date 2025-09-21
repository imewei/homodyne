# Adaptive Noise Model - Detailed Implementation in Homodyne v2

## Overview

The **adaptive noise model** (also called **heteroscedastic noise model**) is the most sophisticated noise estimation method implemented in the Homodyne v2 codebase. Unlike simpler models that assume constant noise across all data points, the adaptive model allows noise to vary with signal strength, providing more realistic uncertainty quantification for XPCS data.

## Mathematical Formulation

The adaptive noise model is defined as:

```
σ(x) = σ_base × (1 + σ_scale × |signal(x)|)
```

Where:
- `σ_base`: Base noise level (minimum noise floor)
- `σ_scale`: Scaling factor for signal-dependent noise
- `signal(x)`: Signal strength at position x (typically |data - mean(data)|)

## Implementation Details

### 1. Core Model Definition
**File**: `homodyne/optimization/hybrid_noise_models.py`

#### VI (Variational Inference) Implementation
```python
elif noise_type == "adaptive":
    # Heteroscedastic noise model
    sigma_base = sample("sigma_base",
                       dist.Gamma(concentration=2.0, rate=20.0))
    sigma_scale = sample("sigma_scale",
                        dist.Beta(concentration1=2.0, concentration0=5.0))

    # Adaptive noise depends on signal strength
    # Use absolute deviation as signal strength proxy
    signal_strength = jnp.abs(data - jnp.mean(data))
    sigma_adaptive = sigma_base * (1.0 + sigma_scale * signal_strength)

    # Flatten for likelihood
    data_flat = data.flatten()
    sigma_flat = sigma_adaptive.flatten()
    data_centered = data_flat - jnp.mean(data_flat)

    sample("obs",
          dist.Normal(0.0, sigma_flat),
          obs=data_centered)
```

#### MCMC Implementation
```python
elif noise_type == "adaptive":
    sigma_base = sample("sigma_base",
                       dist.Gamma(concentration=2.0, rate=20.0))
    sigma_scale = sample("sigma_scale",
                        dist.Beta(concentration1=2.0, concentration0=5.0))
    # ...
    # In likelihood:
    sigma_full = sigma_base * (1.0 + sigma_scale * jnp.abs(theory_fitted))
    sample("obs", dist.Normal(theory_fitted, sigma_full), obs=data)
```

### 2. Parameter Priors

The adaptive model uses carefully chosen priors:

- **σ_base**: `Gamma(concentration=2.0, rate=20.0)`
  - Ensures positive values
  - Prior mean ≈ 0.1
  - Represents the baseline noise floor

- **σ_scale**: `Beta(concentration1=2.0, concentration0=5.0)`
  - Bounded between 0 and 1
  - Prior mean ≈ 0.286
  - Controls how much noise scales with signal

### 3. Noise Estimation Process

**File**: `homodyne/optimization/hybrid_noise_estimation.py`

The estimation follows these steps:

1. **Signal Strength Calculation**:
   - Computes `|data - mean(data)|` as proxy for signal strength
   - Higher deviations from mean indicate stronger signal

2. **Adaptive Scaling**:
   - Base noise `σ_base` provides minimum uncertainty
   - Signal-dependent term `σ_scale × signal_strength` adds heteroscedasticity
   - Total noise varies spatially with data

3. **Parameter Extraction**:
```python
elif noise_type == "adaptive":
    sigma_base_samples = posterior_samples["sigma_base"]
    sigma_scale_samples = posterior_samples["sigma_scale"]

    sigma_mean = {
        "sigma_base": float(jnp.mean(sigma_base_samples)),
        "sigma_scale": float(jnp.mean(sigma_scale_samples))
    }
    sigma_std = {
        "sigma_base": float(jnp.std(sigma_base_samples)),
        "sigma_scale": float(jnp.std(sigma_scale_samples))
    }
```

### 4. Configuration

**File**: `homodyne/config/templates/homodyne_laminar_flow.yaml`

```yaml
noise_estimation:
  model: adaptive  # Can be "hierarchical", "per_angle", or "adaptive"

  adaptive:
    base_noise_range: [0.001, 0.1]      # Range for σ_base
    scaling_range: [0.0, 0.5]           # Range for σ_scale
    validate_heteroscedasticity: true   # Check if adaptive model is needed
```

### 5. Usage via CLI

```bash
# VI with adaptive noise estimation
homodyne --method vi --estimate-noise --noise-model adaptive config.yaml

# MCMC with joint physics+noise estimation
homodyne --method mcmc --estimate-noise --noise-model adaptive config.yaml

# Hybrid method (VI → MCMC) with adaptive noise
homodyne --method hybrid --estimate-noise --noise-model adaptive config.yaml
```

## Advantages of Adaptive Model

1. **Realistic Uncertainty**: Captures varying noise levels across different signal strengths
2. **Better Fits**: Accounts for heteroscedasticity common in XPCS data
3. **Outlier Robustness**: Naturally down-weights high-noise regions
4. **Physical Motivation**: Reflects real experimental conditions where noise varies

## When to Use Adaptive Model

### Recommended For:
- **Varying signal quality**: When different regions have different SNR
- **Laminar flow experiments**: Where flow creates spatially varying signals
- **Large dynamic range data**: When signal strength varies significantly
- **Advanced analysis**: When accurate uncertainty quantification is critical

### Not Recommended For:
- **Quick analysis**: Adds computational overhead
- **Uniform noise data**: When noise is truly homoscedastic
- **Small datasets**: May overfit with limited data

## Computational Considerations

### Performance Impact:
- **VI**: ~20-30% slower than hierarchical model
- **MCMC**: ~15-25% slower due to additional parameters
- **Memory**: Minimal additional memory (2 extra parameters)

### Optimization Settings:
- **Adam steps**: 750 (vs 250 for hierarchical)
- **Learning rate**: 0.01 (same as other models)
- **Early stopping**: Enabled with patience=50 steps

## Model Comparison

| Model | Parameters | Use Case | Speed | Accuracy |
|-------|------------|----------|-------|----------|
| Hierarchical | 1 (σ) | Simple, global noise | Fast | Good for uniform noise |
| Per-angle | n_phi (σ_i) | Anisotropic noise | Medium | Good for angle-dependent noise |
| Adaptive | 2 (σ_base, σ_scale) | Heteroscedastic noise | Slower | Best for varying signal quality |

## Validation and Diagnostics

The adaptive model includes built-in validation:

1. **Heteroscedasticity Test**: Checks if adaptive model is justified
2. **Convergence Monitoring**: Tracks ELBO/loss during optimization
3. **Parameter Bounds Checking**: Ensures physically reasonable values
4. **Residual Analysis**: Validates noise model assumptions post-fit

## Output Interpretation

When using adaptive noise estimation, the results include:

```python
{
    "noise_type": "adaptive",
    "sigma_mean": {
        "sigma_base": 0.05,    # Base noise level
        "sigma_scale": 0.15    # Signal-dependent scaling
    },
    "sigma_std": {
        "sigma_base": 0.005,   # Uncertainty in base noise
        "sigma_scale": 0.02    # Uncertainty in scaling
    },
    "effective_noise_range": [0.05, 0.20],  # Min/max noise across data
    "heteroscedasticity_score": 0.75        # Degree of noise variation
}
```

## Advanced Features

### 1. Adaptive Model Selection
The system can automatically determine if adaptive model is needed:
- Analyzes residuals from simple model
- Tests for heteroscedasticity
- Suggests appropriate noise model

### 2. Integration with Physics Models
Seamlessly works with all analysis modes:
- Static isotropic (3 parameters)
- Static anisotropic (3 parameters + filtering)
- Laminar flow (7 parameters)

### 3. Uncertainty Propagation
Full Bayesian treatment ensures proper uncertainty propagation:
- Noise uncertainty affects parameter uncertainties
- Correlation between noise and physics parameters captured
- Credible intervals reflect total uncertainty

## References and Theory

The adaptive noise model is based on:
1. **Heteroscedastic regression**: Statistical theory for varying variance
2. **XPCS noise characteristics**: Empirical observations from experiments
3. **Bayesian hierarchical models**: Principled uncertainty quantification

## Summary

The adaptive noise model represents the state-of-the-art in noise estimation for XPCS analysis, providing:
- Spatially varying noise estimation
- Improved fit quality for real experimental data
- Robust uncertainty quantification
- Seamless integration with all optimization methods

It's particularly valuable for complex experiments like laminar flow where signal quality varies significantly across the detector and over time.