# MCMC NUTS: Complete Parameter Flow from Config → Fitting → Plotting

This document traces the complete flow of parameters through the MCMC NUTS analysis pipeline.

---

## Overview: 5-Step Flow

```
CONFIG YAML → Load Initial Values → NUTS Sampling → Extract Fitted Parameters → Plot Heatmaps
```

---

## Step 1: Loading Initial Parameters from Config

### Config File Structure

**File**: `config.yaml`
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset', 'contrast', 'offset']
  values: [16830.0, -1.571, 3.026, 0.05015, 1.001]
```

### Loading Code

**File**: `homodyne/config/manager.py:328-447`

```python
def get_initial_parameters(self, use_midpoint_defaults: bool = True) -> dict[str, float]:
    """Get initial parameter values from configuration."""

    # 1. Get initial_parameters section from config
    initial_params = self.config.get("initial_parameters", {})

    # 2. Extract parameter names and values
    param_names_config = initial_params.get("parameter_names")  # ['D0', 'alpha', ...]
    param_values = initial_params.get("values")                 # [16830.0, -1.571, ...]

    # 3. Build dictionary with name mapping
    initial_params_dict: dict[str, float] = {}
    for param_name, value in zip(param_names_config, param_values):
        # Apply canonical name mapping (e.g., gamma_dot_0 → gamma_dot_t0)
        canonical_name = PARAMETER_NAME_MAPPING.get(param_name, param_name)
        initial_params_dict[canonical_name] = float(value)

    # Result: {'D0': 16830.0, 'alpha': -1.571, 'D_offset': 3.026,
    #          'contrast': 0.05015, 'offset': 1.001}
    return initial_params_dict
```

### CLI Entry Point

**File**: `homodyne/cli/commands.py:1247`

```python
# Load initial values from config YAML (initial_parameters.values section)
initial_values = config.get_initial_parameters()

logger.debug(
    f"MCMC initial values from config: {list(initial_values.keys())} = "
    f"{[f'{v:.4g}' for v in initial_values.values()]}"
)
# Output: MCMC initial values from config: ['D0', 'alpha', 'D_offset', 'contrast', 'offset'] =
#         ['1.683e+04', '-1.571', '3.026', '0.05015', '1.001']
```

**Pass to MCMC**:
```python
result = fit_mcmc_jax(
    data,
    t1=t1_pooled,
    t2=t2_pooled,
    phi=phi_pooled,
    q=0.0237,
    L=2000000.0,
    analysis_mode='static',
    method='mcmc',
    initial_values=initial_values,  # ← Passed here!
    parameter_space=parameter_space,
    **mcmc_runtime_kwargs,
)
```

**Key Location**: `homodyne/cli/commands.py:1282-1303`

---

## Step 2: NUTS Initialization with Initial Values

### Formatting for NumPyro Chains

**File**: `homodyne/optimization/mcmc.py:3147-3184`

```python
def _format_init_params_for_chains(
    initial_values: dict[str, float],
    n_chains: int,
    jitter_scale: float,
    rng_key,
    parameter_space: ParameterSpace | None = None,
):
    """Broadcast initial parameters across chains with optional jitter.

    NumPyro requires init_params as arrays of shape (num_chains,) for each parameter.
    """
    formatted: dict[str, Any] = {}

    for param, value in initial_values.items():
        # Broadcast scalar to all chains
        base = jnp.full((n_chains,), float(value), dtype=jnp.float64)
        # base = [16830.0, 16830.0, 16830.0, 16830.0] for D0 with 4 chains

        # Optional jitter (default: 0.0, disabled)
        if jitter_scale > 0.0:
            perturb = jitter_scale * random.normal(subkey, shape=(n_chains,))
            base = base * (1.0 + perturb)

        # Clip to parameter bounds
        if parameter_space is not None:
            lower, upper = parameter_space.get_bounds(param)
            epsilon = 1e-9 * max(1.0, abs(upper - lower))
            base = jnp.clip(base, lower + epsilon, upper - epsilon)

        formatted[param] = base

    # Result: {
    #   'D0': array([16830., 16830., 16830., 16830.]),
    #   'alpha': array([-1.571, -1.571, -1.571, -1.571]),
    #   'D_offset': array([3.026, 3.026, 3.026, 3.026]),
    #   'contrast': array([0.05015, 0.05015, 0.05015, 0.05015]),
    #   'offset': array([1.001, 1.001, 1.001, 1.001])
    # }
    return formatted, current_key
```

### Running NUTS Sampler

**File**: `homodyne/optimization/mcmc.py:3404-3416`

```python
# Format initial values for all chains
init_params_formatted, rng_key = _format_init_params_for_chains(
    initial_values,      # {'D0': 16830.0, 'alpha': -1.571, ...}
    config["n_chains"],  # 4
    config.get("init_jitter_scale", 0.0),
    rng_key,
    parameter_space,
)

# Run MCMC with NUTS kernel
mcmc.run(
    rng_key,
    init_params=init_params_formatted,  # ← Starting points for all chains!
    extra_fields=("potential_energy", "accept_prob", "diverging", "num_steps"),
)
```

**What happens**:
- Each of 4 chains starts at these values
- NUTS explores the posterior around these starting points
- Chains adapt step size and mass matrix during warmup
- After warmup, chains sample from posterior distribution

---

## Step 3: NUTS Sampling Process

### NumPyro Model Definition

**File**: `homodyne/optimization/mcmc.py:2252-2600`

The model samples parameters using priors defined in the config:

```python
def homodyne_model():
    """Inner NumPyro model function with config-driven priors."""

    # For each parameter (D0, alpha, D_offset, contrast, offset):
    for param_name in param_names_ordered:

        # Get prior from ParameterSpace (loaded from config)
        prior_spec = parameter_space.get_prior(param_name)
        # Example: PriorDistribution(dist_type="TruncatedNormal", mu=0.5, sigma=0.2,
        #                             min_val=0.0, max_val=1.0) for contrast

        # Create NumPyro distribution
        dist_class = DIST_TYPE_MAP[prior_spec.dist_type]  # dist.TruncatedNormal

        # Sample parameter value
        if per_angle_scaling and param_name in ["contrast", "offset"]:
            # Per-angle scaling: sample separate value for each phi angle
            param_values = []
            for phi_idx in range(n_phi):
                param_name_phi = f"{param_name}_{phi_idx}"
                param_value_phi = sample(param_name_phi, dist_class(**dist_kwargs))
                param_values.append(param_value_phi)
            sampled_values[param_name] = jnp.array(param_values)
        else:
            # Scalar sampling: single value for all angles
            param_value = sample(param_name, dist_class(**dist_kwargs))
            sampled_values[param_name] = param_value

    # Compute theoretical C2 with sampled parameters
    c2_fitted = _compute_c2_for_model(sampled_values, t1, t2, phi, q, L, ...)

    # Likelihood: compare to observed data
    sample("obs", dist.Normal(c2_fitted, sigma), obs=data)
```

**Key Locations**:
- Prior extraction: `mcmc.py:2352`
- Parameter sampling: `mcmc.py:2586-2599`
- Likelihood: `mcmc.py:3034`

### NUTS Algorithm

NUTS (No-U-Turn Sampler) works as follows:

1. **Start**: Begin at initial values (e.g., D0=16830.0, contrast=0.05015)
2. **Propose**: Use Hamiltonian dynamics to propose new parameter values
3. **Accept/Reject**: Accept based on Metropolis-Hastings criterion
4. **Adapt**: During warmup, adapt step size and mass matrix
5. **Sample**: After warmup, collect samples from posterior

**Example trajectory for contrast**:
```
Initial:    0.05015
Sample 1:   0.04892  (accepted)
Sample 2:   0.05123  (accepted)
Sample 3:   0.04756  (accepted)
...
Sample 3000: 0.04985 (accepted)

Final posterior: mean=0.0495, std=0.0034 (hypothetical)
```

---

## Step 4: Extracting Fitted Parameters

### After MCMC Completes

**File**: `homodyne/optimization/mcmc.py:1689-1695`

```python
# Process posterior samples to get summary statistics
posterior_summary = _process_posterior_samples(
    result,           # MCMC result with samples
    analysis_mode,    # 'static'
    diag_settings     # Diagnostic settings
)
```

### Sample Processing

**File**: `homodyne/optimization/mcmc.py:3542-3845`

```python
def _process_posterior_samples(mcmc_result, analysis_mode, diagnostic_settings):
    """Process posterior samples to extract summary statistics and diagnostics."""

    # 1. Get raw samples from MCMC
    raw_samples = mcmc_result.get_samples()
    # raw_samples = {
    #   'D0': array([68791.4, 70563.3, 61967.3, ...]),      # 12000 samples (4 chains × 3000)
    #   'alpha': array([1.418, 0.799, 0.860, ...]),
    #   'D_offset': array([8781.6, -10276.9, 23140.2, ...]),
    #   'contrast': array([0.0489, 0.0512, 0.0476, ...]),   # IF sampled
    #   'offset': array([1.002, 0.998, 1.005, ...])         # IF sampled
    # }

    # 2. Separate physical params from scaling params
    param_names = ['D0', 'alpha', 'D_offset']  # Physical parameters
    param_samples = jnp.column_stack([
        samples['D0'],
        samples['alpha'],
        samples['D_offset']
    ])  # Shape: (12000, 3)

    contrast_samples = samples.get('contrast')  # Shape: (12000,)
    offset_samples = samples.get('offset')      # Shape: (12000,)

    # 3. Compute summary statistics
    mean_params = jnp.mean(param_samples, axis=0)  # [mean_D0, mean_alpha, mean_D_offset]
    std_params = jnp.std(param_samples, axis=0)    # [std_D0, std_alpha, std_D_offset]

    mean_contrast = float(jnp.mean(contrast_samples))  # Average across all samples
    std_contrast = float(jnp.std(contrast_samples))
    mean_offset = float(jnp.mean(offset_samples))
    std_offset = float(jnp.std(offset_samples))

    # 4. Return summary
    return {
        "mean_params": np.array(mean_params),      # [55491.7, 0.497, 34017.9]
        "std_params": np.array(std_params),        # [21583.9, 0.724, 33281.5]
        "param_names": param_names,
        "mean_contrast": mean_contrast,            # 0.0495 (hypothetical)
        "std_contrast": std_contrast,              # 0.0034
        "mean_offset": mean_offset,                # 1.001
        "std_offset": std_offset,                  # 0.0012
        "samples_params": np.array(param_samples),
        "samples_contrast": np.array(contrast_samples),
        "samples_offset": np.array(offset_samples),
        ...
    }
```

### Creating Result Object

**File**: `homodyne/optimization/mcmc.py:1697-1748`

```python
# Create MCMCResult object with fitted parameters
mcmc_result_obj = MCMCResult(
    method="mcmc",
    parameters=posterior_summary["mean_params"],      # [D0, alpha, D_offset] means
    param_uncertainties=posterior_summary["std_params"],
    mean_contrast=posterior_summary["mean_contrast"],  # Fitted contrast mean
    std_contrast=posterior_summary["std_contrast"],
    mean_offset=posterior_summary["mean_offset"],      # Fitted offset mean
    std_offset=posterior_summary["std_offset"],
    samples=posterior_summary["samples"],
    convergence_info={
        "converged": posterior_summary["converged"],
        "r_hat": posterior_summary["r_hat"],
        "ess": posterior_summary["ess"],
        "acceptance_rate": posterior_summary["acceptance_rate"],
    },
    ...
)

# Write to JSON file
with open(output_dir / "parameters.json", "w") as f:
    json.dump({
        "parameters": {
            "contrast": {
                "mean": mcmc_result_obj.mean_contrast,    # e.g., 0.0495
                "std": mcmc_result_obj.std_contrast        # e.g., 0.0034
            },
            "offset": {
                "mean": mcmc_result_obj.mean_offset,       # e.g., 1.001
                "std": mcmc_result_obj.std_offset          # e.g., 0.0012
            },
            "D0": {
                "mean": mcmc_result_obj.parameters[0],     # e.g., 55491.7
                "std": mcmc_result_obj.param_uncertainties[0]
            },
            ...
        }
    }, f)
```

---

## Step 5: Plotting Heatmaps with Fitted Parameters

### Entry Point

**File**: `homodyne/cli/commands.py:1599-1607`

```python
# After MCMC completes, generate fitted simulation plots
_generate_and_plot_fitted_simulations(
    result,           # MCMCResult with fitted parameters
    filtered_data,    # Experimental data
    config.config,    # Configuration
    output_dir,       # Where to save plots
)
```

### Extracting Parameters for Plotting

**File**: `homodyne/cli/commands.py:2329-2349`

```python
def _generate_and_plot_fitted_simulations(result, data, config, output_dir):
    """Generate and plot C2 simulations using fitted parameters."""

    # Extract fitted parameters from MCMCResult
    if hasattr(result, "mean_params"):
        # MCMC result format
        contrast = result.mean_contrast    # 0.0495 (fitted value!)
        offset = result.mean_offset        # 1.001 (fitted value!)
        physical_params = result.mean_params  # [D0, alpha, D_offset] means

    # Convert to JAX array
    params = jnp.array(physical_params)  # [55491.7, 0.497, 34017.9]

    logger.info(
        f"Using fitted parameters: contrast={contrast:.4f}, offset={offset:.4f}"
    )
    # Output: "Using fitted parameters: contrast=0.0495, offset=1.0010"
```

### Computing Theoretical C2 with Fitted Parameters

**File**: `homodyne/cli/commands.py:2382-2409`

```python
# Create physics model
model = CombinedModel(analysis_mode='static')

# For each phi angle, compute theoretical C2
for phi_deg in phi_angles_list:
    phi_array = jnp.array([phi_deg])

    # Compute g2 with FITTED parameters
    c2_phi = model.compute_g2(
        params,        # [D0=55491.7, alpha=0.497, D_offset=34017.9] ← FITTED!
        t1_grid,       # Time delays
        t2_grid,
        phi_array,     # [0.0] for single angle
        q,             # 0.0237 Å⁻¹
        L_angstroms,   # 2000000 Å
        contrast,      # 0.0495 ← FITTED contrast!
        offset,        # 1.001 ← FITTED offset!
        dt,            # 0.1 s
    )

    # Extract 2D array
    c2_result = np.array(c2_phi[0])  # Shape: (n_t1, n_t2)
    c2_fitted_list.append(c2_result)
```

**Physics Equation Used**:
```
C₂(t₁,t₂) = offset + contrast × [C₁(t₁,t₂)]²

where C₁(t₁,t₂) = D₀ × |t₂ - t₁|^α + D_offset
```

With fitted values:
```
C₂(t₁,t₂) = 1.001 + 0.0495 × [55491.7 × |t₂ - t₁|^0.497 + 34017.9]²
```

### Plotting Heatmap

**File**: `homodyne/cli/commands.py:4858-4915`

```python
def _generate_plots_matplotlib(phi_angles, c2_exp, c2_fit_display, residuals, t1, t2, output_dir):
    """Generate plots using matplotlib backend."""

    for i, phi in enumerate(phi_angles):
        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Experimental data
        im0 = axes[0].imshow(
            c2_exp[i].T,              # Experimental C2
            origin="lower",
            cmap="jet",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
            vmin=1.0, vmax=1.5        # Fixed color scale
        )
        axes[0].set_title(f"Experimental C₂ (φ={phi:.1f}°)")

        # Panel 2: Theoretical fit WITH FITTED PARAMETERS
        im1 = axes[1].imshow(
            c2_fit_display[i].T,      # C2 computed with fitted params!
            origin="lower",
            cmap="jet",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
            vmin=1.0, vmax=1.5
        )
        axes[1].set_title(f"Classical Fit (φ={phi:.1f}°)")
        # This shows: C₂ = 1.001 + 0.0495 × [fitted_model(t₁,t₂)]²

        # Panel 3: Residuals (data - fit)
        im2 = axes[2].imshow(
            residuals[i].T,           # c2_exp - c2_fit
            origin="lower",
            cmap="jet",
            extent=[t1[0], t1[-1], t2[0], t2[-1]],
        )
        axes[2].set_title(f"Residuals (φ={phi:.1f}°)")

        # Save figure
        plot_file = output_dir / f"c2_heatmaps_phi_{phi:.1f}deg.png"
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)
```

**Output**: `c2_heatmaps_phi_0.0deg.png` showing:
- Left: Experimental data
- Middle: Fit using contrast=0.0495, offset=1.001 (FITTED!)
- Right: Residuals

---

## Complete Flow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: CONFIG YAML                                             │
├─────────────────────────────────────────────────────────────────┤
│ initial_parameters:                                             │
│   values: [16830.0, -1.571, 3.026, 0.05015, 1.001]             │
│            ↓↓↓↓↓↓↓  ↓↓↓↓↓  ↓↓↓↓   ↓↓↓↓↓↓↓  ↓↓↓↓↓                │
│            D0      alpha  D_off  contrast  offset               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: config.get_initial_parameters()                         │
├─────────────────────────────────────────────────────────────────┤
│ Returns: {'D0': 16830.0, 'alpha': -1.571, 'D_offset': 3.026,   │
│           'contrast': 0.05015, 'offset': 1.001}                 │
│                                                                 │
│ Location: homodyne/config/manager.py:328-447                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: _format_init_params_for_chains()                        │
├─────────────────────────────────────────────────────────────────┤
│ Broadcasts to 4 chains:                                         │
│   'D0': [16830., 16830., 16830., 16830.]                        │
│   'contrast': [0.05015, 0.05015, 0.05015, 0.05015]              │
│   ...                                                           │
│                                                                 │
│ Location: homodyne/optimization/mcmc.py:3147-3184               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: mcmc.run(init_params=...)                               │
├─────────────────────────────────────────────────────────────────┤
│ NUTS sampling:                                                  │
│   - Start all chains at initial values                          │
│   - Sample parameters using priors                              │
│   - Contrast: TruncatedNormal(0.5, 0.2, [0, 1])                 │
│   - Offset: TruncatedNormal(1.0, 0.2, [0.5, 1.5])               │
│   - Collect 12,000 samples (4 chains × 3000)                    │
│                                                                 │
│ Location: homodyne/optimization/mcmc.py:3404-3416               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: _process_posterior_samples()                            │
├─────────────────────────────────────────────────────────────────┤
│ Extract fitted parameters:                                      │
│   mean_contrast = mean(contrast_samples) = 0.0495 (example)     │
│   std_contrast  = std(contrast_samples)  = 0.0034               │
│   mean_offset   = mean(offset_samples)   = 1.001                │
│   std_offset    = std(offset_samples)    = 0.0012               │
│   mean_params   = [mean(D0), mean(alpha), mean(D_offset)]       │
│                 = [55491.7, 0.497, 34017.9] (example)           │
│                                                                 │
│ Location: homodyne/optimization/mcmc.py:3686-3701               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Save to parameters.json                                 │
├─────────────────────────────────────────────────────────────────┤
│ {                                                               │
│   "contrast": {"mean": 0.0495, "std": 0.0034},                  │
│   "offset": {"mean": 1.001, "std": 0.0012},                     │
│   "D0": {"mean": 55491.7, "std": 21583.9},                      │
│   ...                                                           │
│ }                                                               │
│                                                                 │
│ Location: homodyne/optimization/mcmc.py:1697-1748               │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 7: _generate_and_plot_fitted_simulations()                 │
├─────────────────────────────────────────────────────────────────┤
│ Extract fitted parameters:                                      │
│   contrast = result.mean_contrast  # 0.0495                     │
│   offset   = result.mean_offset    # 1.001                      │
│   params   = result.mean_params    # [55491.7, 0.497, 34017.9] │
│                                                                 │
│ Location: homodyne/cli/commands.py:2329-2349                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 8: model.compute_g2() WITH FITTED PARAMETERS               │
├─────────────────────────────────────────────────────────────────┤
│ C₂(t₁,t₂) = offset + contrast × [C₁(t₁,t₂)]²                    │
│           = 1.001 + 0.0495 × [55491.7×|t₂-t₁|^0.497 + 34017.9]²│
│                                                                 │
│ Location: homodyne/cli/commands.py:2391-2401                    │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 9: Plot heatmap                                            │
├─────────────────────────────────────────────────────────────────┤
│ axes[1].imshow(c2_fit_display[i].T, ...)                        │
│   → Shows C₂ computed with FITTED contrast & offset!            │
│                                                                 │
│ Output: c2_heatmaps_phi_0.0deg.png                              │
│   - Left panel: Experimental data                               │
│   - Middle panel: Fit with contrast=0.0495, offset=1.001        │
│   - Right panel: Residuals                                      │
│                                                                 │
│ Location: homodyne/cli/commands.py:4874-4882                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Files Reference

| Step | File | Function | Lines |
|------|------|----------|-------|
| 1. Load config | `config/manager.py` | `get_initial_parameters()` | 328-447 |
| 2. Pass to MCMC | `cli/commands.py` | (main MCMC command) | 1247, 1300 |
| 3. Format for chains | `optimization/mcmc.py` | `_format_init_params_for_chains()` | 3147-3184 |
| 4. Run NUTS | `optimization/mcmc.py` | `_run_numpyro_sampling()` | 3187-3469 |
| 5. NumPyro model | `optimization/mcmc.py` | `homodyne_model()` | 2252-2600 |
| 6. Extract results | `optimization/mcmc.py` | `_process_posterior_samples()` | 3542-3845 |
| 7. Generate plots | `cli/commands.py` | `_generate_and_plot_fitted_simulations()` | 2261-2410 |
| 8. Compute C2 | `cli/commands.py` | (inside plot function) | 2391-2401 |
| 9. Plot heatmap | `cli/commands.py` | `_generate_plots_matplotlib()` | 4833-4915 |

---

## Important Notes

### Initial Values vs. Fixed Parameters

**CRITICAL**: After the bug fix (2025-12-01), initial values are used as **MCMC starting points**, NOT as fixed parameters:

- ✅ **Correct**: `initial_values` → Starting points for NUTS sampling
- ❌ **Before bug fix**: `initial_values` → Incorrectly treated as fixed overrides

### The Bug That Was Fixed

**Lines 1420-1433** (now removed) incorrectly extracted contrast/offset from `initial_values` and treated them as fixed overrides. This violated MCMC convention and caused garbage fitted values.

**After fix**: Contrast and offset are now properly sampled using TruncatedNormal priors, starting from the initial values provided in config.

---

## Example: Single-Angle Static Analysis

**Config**:
```yaml
initial_parameters:
  parameter_names: ['D0', 'alpha', 'D_offset', 'contrast', 'offset']
  values: [16830.0, -1.571, 3.026, 0.05015, 1.001]
```

**NUTS sampling**:
- Starts at: D0=16830, contrast=0.05015, offset=1.001
- Samples for 3000 iterations × 4 chains = 12000 samples
- Priors guide sampling (TruncatedNormal for contrast/offset)

**Fitted results** (hypothetical):
```json
{
  "D0": {"mean": 55491.7, "std": 21583.9},
  "alpha": {"mean": 0.497, "std": 0.724},
  "D_offset": {"mean": 34017.9, "std": 33281.5},
  "contrast": {"mean": 0.0495, "std": 0.0034},
  "offset": {"mean": 1.001, "std": 0.0012}
}
```

**Heatmap**:
- Middle panel shows C₂ = 1.001 + 0.0495 × [fitted_model]²
- Uses mean values from posterior
