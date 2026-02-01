# CMC Heterogeneity Prevention Design

**Date:** 2026-01-31  
**Status:** Approved  
**Author:** Claude Code

## Problem Statement

CMC (Consensus Monte Carlo) runs on `laminar_flow` mode can fail with `HETEROGENEITY ABORT` due to:
1. **D₀/D_offset linear degeneracy**: D₀ + D_offset is well-constrained but individual values are not
2. **γ̇₀/β multiplicative correlation**: Shear effect depends on both parameters jointly
3. **Insufficient statistical power**: Small shards lack sensitivity to constrain all parameters

This results in high coefficient of variation (CV > 1.0) across shard posterior means, causing the heterogeneity check to abort after hours of computation.

## Solution Overview

Implement four prevention measures:

| Component | Approach |
|-----------|----------|
| **Model Reparameterization** | Transform to orthogonal sampling space internally |
| **Bimodal Detection** | GMM-based per-shard posterior mode detection |
| **Param-aware Shard Sizing** | Auto-scale shard size based on parameter count |
| **Documentation** | Document degeneracy symptoms and mitigations |

## Detailed Design

### 1. Reparameterization Module

**New file:** `homodyne/optimization/cmc/reparameterization.py`

#### Transformations

| Original Space | Sampling Space | Inverse |
|----------------|----------------|---------|
| D₀, D_offset | D_total = D₀ + D_offset | D₀ = D_total × (1 - D_offset_frac) |
| | D_offset_frac = D_offset / D_total | D_offset = D_total × D_offset_frac |
| γ̇₀ | log_γ̇₀ = log(γ̇₀) | γ̇₀ = exp(log_γ̇₀) |

#### Implementation

```python
@dataclass
class ReparamConfig:
    """Configuration for internal reparameterization."""
    enable_d_total: bool = True      # D0 + D_offset → D_total
    enable_shear_log: bool = True    # gamma_dot_t0 → log space
    t_ref: float = 1.0               # Reference time for shear scaling

def transform_to_sampling_space(
    params: dict[str, float],
    config: ReparamConfig,
) -> dict[str, float]:
    """Transform physics params → sampling params."""
    result = dict(params)
    
    if config.enable_d_total:
        D0, D_offset = params["D0"], params["D_offset"]
        D_total = D0 + D_offset
        D_offset_frac = D_offset / D_total if D_total != 0 else 0.0
        result["D_total"] = D_total
        result["D_offset_frac"] = D_offset_frac
        del result["D0"], result["D_offset"]
    
    if config.enable_shear_log and "gamma_dot_t0" in params:
        result["log_gamma_dot_t0"] = np.log(params["gamma_dot_t0"])
        del result["gamma_dot_t0"]
    
    return result

def transform_to_physics_space(
    samples: dict[str, np.ndarray],
    config: ReparamConfig,
) -> dict[str, np.ndarray]:
    """Transform sampling params → physics params (vectorized)."""
    result = dict(samples)
    
    if config.enable_d_total:
        D_total = samples["D_total"]
        D_offset_frac = samples["D_offset_frac"]
        result["D0"] = D_total * (1 - D_offset_frac)
        result["D_offset"] = D_total * D_offset_frac
        del result["D_total"], result["D_offset_frac"]
    
    if config.enable_shear_log and "log_gamma_dot_t0" in samples:
        result["gamma_dot_t0"] = np.exp(samples["log_gamma_dot_t0"])
        del result["log_gamma_dot_t0"]
    
    return result
```

#### Model Integration

In `model.py`, create `xpcs_model_reparameterized`:

```python
def xpcs_model_reparameterized(
    t1, t2, c2_obs, phi, ...,
    reparam_config: ReparamConfig = None,
):
    """CMC model with internal reparameterization."""
    
    # Sample in transformed space
    D_total = numpyro.sample("D_total", dist.LogNormal(log_D_total_prior, 0.5))
    D_offset_frac = numpyro.sample("D_offset_frac", dist.Normal(0.0, 0.3))
    
    log_gamma_dot_t0 = numpyro.sample("log_gamma_dot_t0", dist.Normal(-6, 2))
    beta = numpyro.sample("beta", dist.Normal(-0.3, 0.3))
    
    # Convert to physics space for model evaluation
    D0 = D_total * (1 - D_offset_frac)
    D_offset = D_total * D_offset_frac
    gamma_dot_t0 = jnp.exp(log_gamma_dot_t0)
    
    # Register as deterministic for output
    numpyro.deterministic("D0", D0)
    numpyro.deterministic("D_offset", D_offset)
    numpyro.deterministic("gamma_dot_t0", gamma_dot_t0)
    
    # ... rest of model unchanged
```

#### Priors

| Parameter | Prior | Rationale |
|-----------|-------|-----------|
| D_total | LogNormal(log(20000), 0.5) | Positive, spans typical range |
| D_offset_frac | Normal(0, 0.3) | Allows ±30% offset, centered at zero |
| log_gamma_dot_t0 | Normal(-6, 2) | Spans ~0.0001 to ~0.1 (4 orders of magnitude) |
| beta | Normal(-0.3, 0.3) | Bounded exponent around typical value |

### 2. Bimodal Detection

**File:** `homodyne/optimization/cmc/diagnostics.py`

#### Implementation

```python
@dataclass
class BimodalResult:
    """Result of bimodal detection for a single parameter."""
    is_bimodal: bool
    weights: tuple[float, float]
    means: tuple[float, float]
    separation: float
    relative_separation: float

def detect_bimodal(
    samples: np.ndarray,
    min_weight: float = 0.2,
    min_relative_separation: float = 0.5,
) -> BimodalResult:
    """Detect bimodality using 2-component GMM."""
    samples_2d = samples.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, random_state=42, n_init=3)
    gmm.fit(samples_2d)
    
    weights = tuple(gmm.weights_.tolist())
    means = tuple(gmm.means_.flatten().tolist())
    separation = abs(means[0] - means[1])
    scale = max(abs(np.mean(means)), 1e-10)
    relative_separation = separation / scale
    
    is_bimodal = (
        min(weights) > min_weight and 
        relative_separation > min_relative_separation
    )
    
    return BimodalResult(
        is_bimodal=is_bimodal,
        weights=weights,
        means=means,
        separation=separation,
        relative_separation=relative_separation,
    )
```

#### Integration

In `backends/multiprocessing.py`, after collecting shard samples:

```python
# Check for bimodal posteriors (per-shard)
bimodal_alerts = []
for i, shard_result in enumerate(successful_samples):
    bimodal_results = check_shard_bimodality(shard_result.samples)
    for param, result in bimodal_results.items():
        if result.is_bimodal:
            bimodal_alerts.append((i, param, result))
            run_logger.warning(
                f"BIMODAL POSTERIOR: Shard {i}, {param}: "
                f"modes at {result.means[0]:.4g} and {result.means[1]:.4g} "
                f"(weights: {result.weights[0]:.2f}/{result.weights[1]:.2f})"
            )
```

### 3. Param-aware Shard Sizing

**File:** `homodyne/optimization/cmc/config.py`

#### Modified `get_num_shards`

```python
def get_num_shards(self, n_points: int, n_phi: int, n_params: int = 7) -> int:
    """Calculate number of shards with param-aware sizing."""
    if isinstance(self.num_shards, int):
        return self.num_shards

    # Base max_points_per_shard
    if isinstance(self.max_points_per_shard, int):
        base_max = self.max_points_per_shard
    else:
        base_max = 100000

    # Param-aware adjustment: scale up for n_params > 7
    param_factor = max(1.0, n_params / 7.0)
    min_required = int(self.min_points_per_param * n_params)
    
    adjusted_max = max(
        int(base_max * param_factor),
        min_required,
    )
    
    if param_factor > 1.0:
        logger.warning(
            f"Param-aware shard sizing: {n_params} params detected. "
            f"Adjusted max_points_per_shard: {base_max:,} → {adjusted_max:,} "
            f"(factor={param_factor:.2f})"
        )
    
    return max(1, n_points // adjusted_max)
```

#### Helper function

```python
def get_model_param_count(
    analysis_mode: str,
    per_angle_mode: str,
    n_phi: int,
) -> int:
    """Get total parameter count for given configuration."""
    if analysis_mode == "static":
        base = 3
    else:  # laminar_flow
        base = 7
    
    if per_angle_mode == "constant":
        scaling = 0
    elif per_angle_mode == "auto":
        scaling = 2
    elif per_angle_mode == "individual":
        scaling = 2 * n_phi
    elif per_angle_mode == "fourier":
        scaling = 10
    else:
        scaling = 0
    
    return base + scaling + 1  # +1 for sigma
```

### 4. Documentation

**File:** `docs/architecture/cmc-fitting-architecture.md`

Add section "Parameter Degeneracy in Laminar Flow Mode" documenting:
- D₀/D_offset linear degeneracy (symptoms, cause, mitigation)
- γ̇₀/β multiplicative correlation (symptoms, cause, mitigation)
- Diagnostic indicators table
- Configuration options for edge cases

## New Configuration Options

```yaml
optimization:
  cmc:
    reparameterization:
      d_total: true           # Enable D_total reparameterization
      log_gamma_dot: true     # Sample log(gamma_dot_t0)
    sharding:
      min_points_per_param: 1500  # Minimum points per param per shard
    validation:
      bimodal_min_weight: 0.2     # GMM component weight threshold
      bimodal_min_separation: 0.5 # Relative separation threshold
```

## Files to Modify/Create

| File | Action |
|------|--------|
| `homodyne/optimization/cmc/reparameterization.py` | **CREATE** |
| `homodyne/optimization/cmc/model.py` | MODIFY - add reparameterized model |
| `homodyne/optimization/cmc/diagnostics.py` | MODIFY - add bimodal detection |
| `homodyne/optimization/cmc/config.py` | MODIFY - add param-aware sizing |
| `homodyne/optimization/cmc/core.py` | MODIFY - pass n_params to get_num_shards |
| `homodyne/optimization/cmc/backends/multiprocessing.py` | MODIFY - integrate bimodal alerts |
| `docs/architecture/cmc-fitting-architecture.md` | MODIFY - add degeneracy docs |

## Backward Compatibility

- All features **enabled by default** for `laminar_flow` mode
- Output format unchanged (D0, D_offset, gamma_dot_t0 in results)
- Existing configs work without modification
- Can disable with `reparameterization.d_total: false`

## Testing Strategy

1. Unit tests for `reparameterization.py` transforms (roundtrip)
2. Unit tests for `detect_bimodal` with synthetic bimodal/unimodal samples
3. Integration test: laminar_flow CMC run with reparameterization enabled
4. Regression test: verify output format unchanged
