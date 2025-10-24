# Consensus Monte Carlo (CMC) User Guide

**Version:** 3.0+
**Last Updated:** 2025-10-24
**Status:** Production Ready

---

## Table of Contents

1. [Introduction](#introduction)
2. [When to Use CMC](#when-to-use-cmc)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Configuration Guide](#configuration-guide)
6. [CLI Usage](#cli-usage)
7. [Understanding Diagnostic Output](#understanding-diagnostic-output)
8. [Performance Tuning](#performance-tuning)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Introduction

### What is Consensus Monte Carlo (CMC)?

**Consensus Monte Carlo** is a divide-and-conquer strategy for scalable Bayesian inference that enables full uncertainty quantification on datasets with millions to billions of data points.

**The Core Idea:**
```
Large Dataset (50M points)
    ↓
Split into Shards (50 shards × 1M points each)
    ↓
Run Independent MCMC on Each Shard (parallel execution)
    ↓
Combine Subposteriors (weighted Gaussian product)
    ↓
Final Posterior Distribution (same accuracy as full MCMC)
```

**Key Benefits:**
- **Scalability**: Handle datasets that don't fit in memory (unlimited size)
- **Speed**: Linear speedup with number of shards (50 shards = 50x faster)
- **Memory Efficiency**: Each shard fits in single GPU memory (~16GB)
- **Fault Tolerance**: Failed shards can be retried independently
- **Full Bayesian Uncertainty**: Complete posterior distributions, not just point estimates

### Why Use CMC for XPCS Analysis?

Modern XPCS detectors generate **10M-100M point datasets routinely**. Traditional MCMC NUTS sampling becomes impractical for these large datasets:

**Standard NUTS Limitations:**
- Memory bottleneck at ~1M points (16GB GPU)
- Runtime scales poorly (hours to days for 10M+ points)
- No parallelization across data

**CMC Solution:**
- Constant memory footprint regardless of dataset size
- Linear speedup with hardware parallelization
- Production workflows complete in ~1-2 hours for 50M points

### Scientific Foundation

CMC is based on peer-reviewed research:

- **Scott et al. (2016)**: "Bayes and big data: the consensus Monte Carlo algorithm"
  *International Journal of Management Science and Engineering Management*
  https://arxiv.org/abs/1411.7435

- **Neiswanger et al. (2014)**: "Asymptotically exact, embarrassingly parallel MCMC"
  *ICML 2014*

The method is **asymptotically exact**: as shard size increases, the combined posterior converges to the true full-data posterior.

---

## When to Use CMC

### Decision Matrix

| Dataset Size | Hardware | Recommended Method | Rationale |
|-------------|----------|-------------------|-----------|
| < 500k points | Any | **Standard NUTS** | No overhead, standard MCMC sufficient |
| 500k - 5M points | Single GPU | **Standard NUTS** or **CMC** | Either works; CMC adds ~10% overhead |
| 5M - 50M points | Single GPU | **CMC** (auto) | Memory-limited NUTS, CMC enables completion |
| 5M - 50M points | Multi-GPU | **CMC** (parallel) | 4-8x speedup with parallel execution |
| 50M+ points | Any | **CMC** (required) | Standard NUTS not feasible |
| Any size | HPC Cluster | **CMC** (PBS/Slurm) | Leverage cluster parallelization |

### Use CMC When:

✅ **Dataset > 500k points** and you need Bayesian uncertainty quantification
✅ **Production pipelines** requiring reproducible uncertainty estimates
✅ **Multi-GPU systems** or HPC clusters available for parallelization
✅ **Memory-limited** environments (e.g., 16GB GPU with 10M point dataset)
✅ **Publication-quality** analysis requiring full posterior distributions

### Use Standard NUTS When:

✅ **Dataset < 500k points** (CMC overhead not worth it)
✅ **Exploratory analysis** where point estimates (NLSQ) are sufficient
✅ **Single-GPU** system with small datasets (< 1M points)
✅ **Quick iteration** during development (faster for small data)

### Comparison with Other Methods

| Feature | NLSQ | Standard NUTS | CMC |
|---------|------|---------------|-----|
| **Dataset Size** | Unlimited | < 1M points | Unlimited |
| **Uncertainty Quantification** | No (point estimate only) | Yes (full posterior) | Yes (full posterior) |
| **Speed** | Fast (minutes) | Slow (hours) | Fast (minutes-hours) |
| **Memory Usage** | Low | High (O(n)) | Constant (O(n/m)) |
| **Parallelization** | No | No | Yes (linear speedup) |
| **Use Case** | Point estimates | Small-scale Bayesian | Large-scale Bayesian |

**Recommended Workflow:**
1. **Exploratory Analysis**: Use NLSQ for initial parameter estimates
2. **Uncertainty Quantification**: Use CMC (auto-detection) for final Bayesian analysis
3. **Small Datasets**: Standard NUTS is simpler if dataset < 500k points

---

## Installation

### Prerequisites

**Software Requirements:**
- Python 3.12+
- JAX 0.8.0
- NumPyro 0.18-0.19
- homodyne 3.0+

**Hardware Requirements:**
- **Minimum**: 8-core CPU, 32GB RAM
- **Recommended**: 16GB+ GPU (NVIDIA A100, H100) or HPC cluster

### Platform Support

| Platform | CPU Support | GPU Support | Status |
|----------|------------|-------------|--------|
| Linux | ✅ Full | ✅ Full (CUDA 12+) | Production Ready |
| macOS | ✅ Full | ❌ No GPU | CPU-only |
| Windows | ✅ Full | ❌ No GPU | CPU-only |

### Installation Steps

#### 1. CPU-Only Installation (All Platforms)

```bash
pip install homodyne
```

This works on all platforms (Linux, macOS, Windows) and is suitable for:
- Development and testing
- Small datasets (< 500k points)
- HPC CPU clusters

#### 2. GPU Installation (Linux Only)

For 20-100x speedup on large datasets:

```bash
# Uninstall CPU-only JAX
pip uninstall -y jax jaxlib

# Install GPU-enabled JAX (CUDA 12.1-12.9 required)
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Install homodyne
pip install homodyne

# Verify GPU detection
python -c "import jax; print('Devices:', jax.devices())"
# Expected output: [cuda(id=0)]
```

**GPU Requirements:**
- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA 12.1-12.9 pre-installed on system
- NVIDIA driver >= 525

#### 3. HPC Cluster Installation

On HPC systems with PBS/Slurm schedulers:

```bash
# Load system CUDA modules
module load cuda/12.2 cudnn/9.8

# Install JAX with CUDA support
pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

# Install homodyne
pip install homodyne[dev]
```

**Cluster-Specific Configuration:**
- CMC automatically detects PBS/Slurm schedulers
- Set PBS project name in configuration
- Configure walltime and resource allocation

### Verification

After installation, verify CMC is available:

```python
from homodyne.optimization.mcmc import fit_mcmc_jax

# Check for CMC method support
help(fit_mcmc_jax)
# Should show method='auto', 'nuts', or 'cmc'
```

---

## Quick Start

### Example 1: Automatic Method Selection (Recommended)

The simplest way to use CMC is **automatic method selection**:

```python
from homodyne.optimization.mcmc import fit_mcmc_jax

# Load your XPCS data
data = load_xpcs_data("experiment.hdf")  # 10M points
t1, t2, phi = data['t1'], data['t2'], data['phi']
c2_exp = data['c2']

# Run MCMC with automatic method selection
# CMC will be used automatically for large datasets (> 500k points)
result = fit_mcmc_jax(
    data=c2_exp,
    t1=t1,
    t2=t2,
    phi=phi,
    q=0.0054,
    L=2000000,
    analysis_mode='static_isotropic',
    initial_params={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
)

# Check which method was used
if result.is_cmc_result():
    print(f"✓ CMC used with {result.num_shards} shards")
    print(f"  Combination method: {result.combination_method}")
    print(f"  Converged shards: {result.cmc_diagnostics['n_shards_converged']}/{result.cmc_diagnostics['n_shards_total']}")
else:
    print("Standard NUTS used (dataset < 500k points)")

# Access posterior samples
mean_params = result.mean_params
std_params = result.std_params
samples = result.samples_params
```

**What Happened:**
1. Dataset size detected (10M points)
2. Hardware detected (GPU available)
3. CMC automatically selected (dataset > 500k threshold)
4. Data split into ~10 shards (1M points each)
5. Parallel MCMC executed on all shards
6. Posteriors combined using weighted Gaussian product
7. Results validated and returned as extended `MCMCResult`

### Example 2: Force CMC with Custom Configuration

For production workflows, explicitly configure CMC:

```python
# Define CMC configuration
cmc_config = {
    'sharding': {
        'strategy': 'stratified',  # Stratified sampling across (t1, t2, phi)
        'num_shards': 16,           # Manual override (auto-detection: ~10)
    },
    'initialization': {
        'method': 'svi',            # SVI for better convergence
        'svi_steps': 10000,         # 10k optimization steps
    },
    'backend': {
        'name': 'pjit',             # Force pjit backend (GPU)
    },
    'combination': {
        'method': 'weighted',       # Weighted Gaussian product
        'min_success_rate': 0.90,   # 90% of shards must converge
    },
    'validation': {
        'strict_mode': True,        # Fail on validation errors
        'max_kl_divergence': 2.0,   # Between-shard agreement threshold
    },
}

# Run CMC with custom configuration
result = fit_mcmc_jax(
    data=c2_exp,
    t1=t1, t2=t2, phi=phi,
    q=0.0054, L=2000000,
    analysis_mode='laminar_flow',  # 9-parameter flow analysis
    initial_params={
        'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0,
        'gamma_dot_0': 0.01, 'beta': 1.0, 'gamma_dot_offset': 0.0, 'phi_0': 0.0,
    },
    method='cmc',              # Force CMC
    cmc_config=cmc_config,     # Custom configuration
)
```

### Example 3: CLI Usage

For production pipelines, use the command-line interface:

```bash
# Create configuration file
cat > cmc_config.yaml <<EOF
analysis_mode: static_isotropic

experimental_data:
  file_path: "./data/large_dataset.hdf"

optimization:
  method: cmc
  cmc:
    enable: true
    sharding:
      strategy: stratified
      num_shards: auto
    initialization:
      method: svi
      svi_steps: 5000
EOF

# Run analysis
homodyne --config cmc_config.yaml --output-dir ./results/cmc_analysis

# Output will include:
# - results/cmc_analysis/cmc/parameters.json
# - results/cmc_analysis/cmc/fitted_data.npz
# - results/cmc_analysis/cmc/convergence_metrics.json
# - results/cmc_analysis/cmc/per_shard_diagnostics.json
```

---

## Configuration Guide

### Full Configuration Template

CMC configuration is specified in YAML format. See the full template at:
```
homodyne/config/templates/homodyne_cmc_config.yaml
```

### Key Configuration Sections

#### 1. Method Selection

```yaml
optimization:
  method: cmc  # Options: auto, nuts, cmc
```

**Options:**
- `auto`: Automatic selection based on dataset size and hardware (recommended)
- `nuts`: Force standard NUTS MCMC
- `cmc`: Force Consensus Monte Carlo

#### 2. CMC Enable/Disable

```yaml
optimization:
  cmc:
    enable: auto  # Options: true, false, auto
    min_points_for_cmc: 500000  # Minimum dataset size for CMC
```

**Enable Options:**
- `true`: Always use CMC (regardless of dataset size)
- `false`: Never use CMC (use standard NUTS)
- `auto`: Use CMC if dataset > `min_points_for_cmc` (recommended)

#### 3. Data Sharding Strategy

```yaml
optimization:
  cmc:
    sharding:
      strategy: stratified  # Options: stratified, random, contiguous
      num_shards: auto      # Options: auto, <integer>
      max_points_per_shard: auto  # Options: auto, <integer>
```

**Sharding Strategies:**

- **`stratified`** (RECOMMENDED):
  - Ensures each shard is representative of full dataset
  - Stratified sampling across (t1, t2, phi) dimensions
  - Best for heterogeneous data distributions
  - Example: Each shard has proportional samples from all phi angles

- **`random`**:
  - Random permutation before sharding
  - Simpler and faster than stratified
  - Good for homogeneous data
  - Example: Randomly shuffle data, then split

- **`contiguous`**:
  - Split data into contiguous blocks
  - Fastest (no reordering)
  - Assumes data is already shuffled
  - Example: First N points → shard 1, next N → shard 2, etc.

**Num Shards Selection:**

- **`auto`** (RECOMMENDED): Automatically calculate based on hardware
  - GPU: 1M points per shard (fits in 16GB GPU memory)
  - CPU: 2M points per shard (uses available system memory)
  - Cluster: Scales to available nodes × cores_per_node

- **`<integer>`**: Manual specification (advanced users only)
  - Example: `num_shards: 16` for 16 shards
  - Useful for reproducibility or hardware-specific tuning

#### 4. Initialization Strategy

```yaml
optimization:
  cmc:
    initialization:
      method: svi  # Options: svi, nlsq, identity
      svi_steps: 5000
      svi_learning_rate: 0.001
      svi_rank: 5
      fallback_to_identity: true
```

**Initialization Methods:**

- **`svi`** (RECOMMENDED):
  - Stochastic Variational Inference on full dataset
  - Estimates inverse mass matrix for NUTS
  - Improves per-shard MCMC convergence
  - Overhead: ~30-60 seconds for 1M-10M points
  - Best for: Production workflows requiring optimal convergence

- **`nlsq`**:
  - Use NLSQ optimization results for initialization
  - Fast (no additional computation if NLSQ already run)
  - Requires NLSQ to converge successfully
  - Best for: Workflows where NLSQ has already been run

- **`identity`**:
  - Use identity matrix for inverse mass matrix
  - Fallback option when SVI/NLSQ fail
  - Slowest convergence (more warmup needed)
  - Best for: Debugging or when other methods fail

#### 5. Backend Selection

```yaml
optimization:
  cmc:
    backend:
      name: auto  # Options: auto, pjit, multiprocessing, pbs, slurm
      enable_checkpoints: true
      checkpoint_frequency: 10
      checkpoint_dir: "./checkpoints/cmc"
```

**Backend Options:**

- **`auto`** (RECOMMENDED):
  - Automatic selection based on detected hardware
  - Multi-node cluster → pbs or slurm
  - Multi-GPU → pjit (parallel)
  - Single GPU → pjit (sequential)
  - CPU-only → multiprocessing

- **`pjit`**:
  - JAX pjit backend for GPU/CPU execution
  - Parallel execution on multi-GPU systems
  - Sequential execution on single GPU
  - Best for: GPU workstations and servers

- **`multiprocessing`**:
  - Python multiprocessing.Pool for CPU parallelism
  - Best for: CPU-only systems, high-core-count workstations

- **`pbs`** / **`slurm`**:
  - HPC cluster job array submission
  - Best for: PBS Pro or Slurm clusters with 32+ nodes

#### 6. Subposterior Combination

```yaml
optimization:
  cmc:
    combination:
      method: weighted_gaussian  # Options: weighted_gaussian, simple_average, auto
      validate_results: true
      min_success_rate: 0.90
```

**Combination Methods:**

- **`weighted_gaussian`** (RECOMMENDED):
  - Weighted Gaussian product (Scott et al. 2016)
  - Optimal for Gaussian-like posteriors
  - Uses precision weighting for better accuracy
  - Example: Combines N(μ₁, Σ₁) and N(μ₂, Σ₂) → N(μ, Σ)

- **`simple_average`**:
  - Simple concatenation and resampling
  - More robust to non-Gaussian posteriors
  - Handles multi-modal distributions
  - Less statistically efficient

- **`auto`**:
  - Try weighted Gaussian first, fallback to simple average
  - Best for: Unknown posterior shapes or debugging

#### 7. Per-Shard MCMC Configuration

```yaml
optimization:
  cmc:
    per_shard_mcmc:
      num_warmup: 500     # Reduced from standard 1000 (SVI init helps)
      num_samples: 2000   # Same as standard MCMC
      num_chains: 1       # 1 chain per shard (parallelism across shards)
      subsample_size: auto
```

**Configuration Guidelines:**

- **`num_warmup`**: 500-1000 iterations
  - 500: If using SVI initialization (recommended)
  - 1000: If using identity initialization

- **`num_samples`**: 2000-5000 samples
  - 2000: Standard (sufficient for most analyses)
  - 5000: High-precision uncertainty estimates

- **`num_chains`**: 1-4 chains per shard
  - 1: Maximum parallelism across shards (recommended)
  - 4: Better per-shard diagnostics, slower overall

#### 8. Convergence Validation

```yaml
optimization:
  cmc:
    validation:
      strict_mode: true       # Fail on validation errors
      min_per_shard_ess: 100  # Minimum effective sample size
      max_per_shard_rhat: 1.1 # Maximum R-hat (< 1.1 = converged)
      max_between_shard_kl: 2.0  # Maximum KL divergence between shards
      min_success_rate: 0.90  # Minimum fraction of shards that must converge
```

**Validation Criteria:**

- **`strict_mode`**:
  - `true`: Fail optimization if validation criteria not met (production)
  - `false`: Log warnings but continue (exploratory analysis)

- **`min_per_shard_ess`**:
  - 100: Standard (sufficient for reliable inference)
  - 200: High-quality (better uncertainty estimates)

- **`max_per_shard_rhat`**:
  - 1.1: Standard convergence criterion (Gelman et al. 2013)
  - 1.05: Stricter (more conservative)

- **`max_between_shard_kl`**:
  - 2.0: Standard (shards agree reasonably well)
  - 1.0: Stricter (shards must agree very well)

- **`min_success_rate`**:
  - 0.90: At least 90% of shards must converge
  - 0.80: More lenient (allows 20% failure)

### Hardware-Specific Recommendations

#### Single 16GB GPU

```yaml
optimization:
  cmc:
    sharding:
      num_shards: auto      # Auto: ~8 shards for 8M points
      max_points_per_shard: 1000000  # 1M points per shard
    backend:
      name: pjit            # Sequential execution on single GPU
```

#### 8x A100 (80GB) GPUs

```yaml
optimization:
  cmc:
    sharding:
      num_shards: 32        # 4 shards per GPU
    backend:
      name: pjit            # Parallel pjit execution
```

#### HPC Cluster (36-core nodes)

```yaml
optimization:
  cmc:
    sharding:
      num_shards: 64        # 2-4 shards per node
    backend:
      name: pbs
pbs:
  project_name: "your_project"
  walltime: "02:00:00"      # 2 hours per shard
  cores_per_node: 36
```

---

## CLI Usage

### Basic Command

```bash
homodyne --config config.yaml --method cmc
```

### Common CLI Options

```bash
# Automatic method selection
homodyne --config config.yaml

# Force CMC method
homodyne --config config.yaml --method cmc

# Force standard NUTS
homodyne --config config.yaml --method nuts

# Override data file
homodyne --config config.yaml --data-file /path/to/experiment.hdf

# Specify output directory
homodyne --config config.yaml --output-dir ./results/cmc_run_001

# Enable verbose logging
homodyne --config config.yaml --verbose

# Quiet mode (errors only)
homodyne --config config.yaml --quiet
```

### Output Files

CMC generates the following output files:

```
output_dir/
├── cmc/
│   ├── parameters.json           # Parameter estimates + uncertainties
│   ├── fitted_data.npz            # Experimental + theoretical data
│   ├── convergence_metrics.json   # Per-shard convergence diagnostics
│   └── per_shard_diagnostics.json # Detailed shard-level diagnostics
├── plots/
│   ├── posterior_distributions.png
│   ├── trace_plots.png
│   └── kl_divergence_matrix.png
└── logs/
    └── homodyne_cmc_YYYYMMDD_HHMMSS.log
```

### Production Workflow Example

```bash
#!/bin/bash
# production_cmc_workflow.sh

# 1. Run CMC analysis
homodyne --config cmc_production_config.yaml \
         --output-dir ./results/experiment_001 \
         --verbose \
         2>&1 | tee cmc_run.log

# 2. Check exit code
if [ $? -eq 0 ]; then
    echo "✓ CMC analysis completed successfully"

    # 3. Validate results
    python validate_cmc_results.py ./results/experiment_001

    # 4. Generate report
    python generate_report.py ./results/experiment_001
else
    echo "✗ CMC analysis failed"
    exit 1
fi
```

---

## Understanding Diagnostic Output

### Per-Shard Diagnostics

Each shard produces convergence diagnostics:

```json
{
  "shard_id": 0,
  "n_samples": 2000,
  "converged": true,
  "acceptance_rate": 0.85,
  "r_hat": 1.02,
  "ess": 1543.2
}
```

**Interpretation:**

- **`converged`**: `true` if shard passed all convergence criteria
- **`acceptance_rate`**: 0.65-0.85 is ideal (NUTS target: 0.8)
- **`r_hat`**: < 1.1 indicates convergence (Gelman-Rubin statistic)
- **`ess`**: Effective sample size (> 100 is sufficient, > 200 is good)

### CMC-Level Diagnostics

Overall CMC pipeline diagnostics:

```json
{
  "combination_success": true,
  "n_shards_converged": 14,
  "n_shards_total": 16,
  "convergence_rate": 0.875,
  "combination_time": 12.3,
  "max_kl_divergence": 1.42
}
```

**Interpretation:**

- **`combination_success`**: Whether posterior combination succeeded
- **`convergence_rate`**: Fraction of shards that converged (> 0.9 is good)
- **`max_kl_divergence`**: Maximum KL divergence between any two shards
  - < 0.5: Excellent agreement
  - < 2.0: Good agreement
  - > 5.0: Poor agreement (potential issues)

### Warning Signs

⚠️ **Low Success Rate** (`< 0.8`):
- Check data quality (outliers, missing values)
- Increase `num_warmup` or `num_samples`
- Try `stratified` sharding strategy

⚠️ **High KL Divergence** (`> 5.0`):
- Shards converged to different posteriors
- Check for multi-modal distributions
- Verify data is homogeneous
- Try `simple_average` combination method

⚠️ **Low ESS** (`< 100`):
- Increase `num_samples`
- Check SVI initialization quality
- Verify priors are reasonable

---

## Performance Tuning

### Optimizing Shard Size

**Rule of Thumb:**
- GPU: 1M points per shard (fits in 16GB memory)
- CPU: 2M points per shard (uses available RAM)

**Example Calculations:**

```python
# 50M point dataset on 8x A100 GPUs
num_shards = 50  # 1M points per shard
shards_per_gpu = 50 / 8 = 6.25  # ~6 shards per GPU

# 100M point dataset on 36-core CPU
num_shards = 50  # 2M points per shard
shards_per_core = 50 / 36 = 1.4  # ~1-2 shards per core
```

### Optimizing SVI Initialization

**Trade-off:** SVI overhead vs. convergence speed

```yaml
# Fast SVI (30-60 seconds overhead)
initialization:
  method: svi
  svi_steps: 5000
  svi_learning_rate: 0.001

# High-quality SVI (2-5 minutes overhead)
initialization:
  method: svi
  svi_steps: 20000
  svi_learning_rate: 0.0005
```

**Recommendation:**
- Development: 5000 steps (fast iteration)
- Production: 10000-20000 steps (better convergence)

### Reducing Walltime

**Strategies:**

1. **Reduce per-shard MCMC iterations:**
```yaml
per_shard_mcmc:
  num_warmup: 500   # Down from 1000 (with SVI init)
  num_samples: 2000 # Keep at 2000 for quality
```

2. **Increase parallelism:**
```yaml
sharding:
  num_shards: 32  # More shards = more parallelism
```

3. **Use GPU backend:**
```yaml
backend:
  name: pjit  # GPU: 10-20x faster than CPU
```

### Memory Optimization

**Reduce memory usage per shard:**

```yaml
sharding:
  max_points_per_shard: 500000  # Down from 1M (for 8GB GPUs)

per_shard_mcmc:
  num_chains: 1  # Use 1 chain instead of 4
```

### Benchmarking Your System

```python
from homodyne.device import benchmark_device_performance

# Run performance benchmark
results = benchmark_device_performance()
print(results)
# Example output:
# {
#   'platform': 'gpu',
#   'num_devices': 8,
#   'compute_time_ms': 123.4,
#   'memory_bandwidth_gb_s': 890.2,
#   'recommended_shard_size': 1000000,
# }
```

---

## Troubleshooting

### Common Issues

#### Issue 1: "All shards failed to converge"

**Symptoms:**
```
RuntimeError: All shards failed to converge. Cannot combine posteriors.
```

**Causes:**
- Poor initialization (bad initial parameters)
- Data quality issues (NaN, Inf, outliers)
- Too few warmup iterations

**Solutions:**

1. Check initial parameters:
```python
# Run NLSQ first to get good initial values
from homodyne.optimization.nlsq import fit_nlsq_jax
nlsq_result = fit_nlsq_jax(data, t1, t2, phi, q, L, analysis_mode)
initial_params = nlsq_result.best_fit_parameters
```

2. Increase warmup:
```yaml
per_shard_mcmc:
  num_warmup: 1000  # Up from 500
```

3. Check data quality:
```python
import numpy as np
assert not np.any(np.isnan(data))
assert not np.any(np.isinf(data))
```

#### Issue 2: "High KL divergence between shards"

**Symptoms:**
```
WARNING: Max KL divergence (7.3) exceeds threshold (2.0)
```

**Causes:**
- Multi-modal posterior
- Non-representative shards
- Data heterogeneity

**Solutions:**

1. Use stratified sharding:
```yaml
sharding:
  strategy: stratified  # Ensures representative shards
```

2. Switch to simple averaging:
```yaml
combination:
  method: simple_average  # More robust to multi-modality
```

3. Increase shard size:
```yaml
sharding:
  num_shards: 10  # Fewer, larger shards (more representative)
```

#### Issue 3: "Memory errors during execution"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Causes:**
- Shard size too large for GPU memory
- Multiple chains per shard
- Memory leak

**Solutions:**

1. Reduce shard size:
```yaml
sharding:
  max_points_per_shard: 500000  # Down from 1M
```

2. Use 1 chain per shard:
```yaml
per_shard_mcmc:
  num_chains: 1  # Down from 4
```

3. Switch to CPU backend:
```yaml
backend:
  name: multiprocessing  # Use CPU instead of GPU
```

#### Issue 4: "Slow SVI initialization"

**Symptoms:**
- SVI takes > 5 minutes
- No progress after many iterations

**Causes:**
- Too many SVI steps
- Poor initial parameters
- Large dataset

**Solutions:**

1. Reduce SVI steps:
```yaml
initialization:
  svi_steps: 5000  # Down from 20000
```

2. Use NLSQ initialization instead:
```yaml
initialization:
  method: nlsq  # Skip SVI entirely
```

3. Set SVI timeout:
```yaml
initialization:
  svi_timeout: 300  # 5 minutes max
```

### Debugging Workflow

```python
# 1. Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. Run with lenient validation
result = fit_mcmc_jax(
    data=data, ...,
    method='cmc',
    cmc_config={
        'validation': {'strict_mode': False},  # Don't fail, just warn
    },
)

# 3. Inspect diagnostics
print("Per-shard diagnostics:", result.per_shard_diagnostics)
print("CMC diagnostics:", result.cmc_diagnostics)

# 4. Check for failed shards
failed_shards = [
    diag for diag in result.per_shard_diagnostics
    if not diag['converged']
]
print(f"Failed shards: {len(failed_shards)}/{result.num_shards}")
```

### Getting Help

If issues persist:

1. **Check documentation**: `docs/troubleshooting/cmc_troubleshooting.md`
2. **Search GitHub issues**: https://github.com/your-org/homodyne/issues
3. **Open new issue**: Include:
   - Configuration file
   - Error messages
   - System information (`python -m homodyne.device`)
   - CMC diagnostics output

---

## Advanced Usage

### Custom Sharding Strategies

For special data distributions:

```python
from homodyne.optimization.cmc.sharding import shard_data_stratified

# Custom stratified sharding by phi angle
def custom_phi_stratified_sharding(data, phi, num_shards):
    """Shard data to ensure equal phi coverage per shard."""
    unique_phi = np.unique(phi)
    shards = []

    for phi_angle in unique_phi:
        # Get all data for this phi
        phi_mask = phi == phi_angle
        phi_data = data[phi_mask]

        # Split into num_shards parts
        phi_shard_size = len(phi_data) // num_shards
        for i in range(num_shards):
            start = i * phi_shard_size
            end = (i + 1) * phi_shard_size if i < num_shards - 1 else len(phi_data)
            shard_data = phi_data[start:end]

            if i < len(shards):
                # Append to existing shard
                shards[i]['data'] = np.concatenate([shards[i]['data'], shard_data])
            else:
                # Create new shard
                shards.append({'data': shard_data, 'shard_id': i})

    return shards

# Use custom sharding
shards = custom_phi_stratified_sharding(c2_exp, phi, num_shards=16)
```

### Checkpoint/Resume Workflows

For long-running CMC jobs:

```yaml
backend:
  enable_checkpoints: true
  checkpoint_frequency: 5      # Save every 5 shards
  checkpoint_dir: "./checkpoints/experiment_001"
  resume_from_checkpoint: true # Auto-resume if checkpoint exists
```

**Manual checkpoint management:**

```python
from homodyne.optimization.cmc.coordinator import CMCCoordinator

# Create coordinator
coordinator = CMCCoordinator(config)

# Run with checkpointing
result = coordinator.run_cmc(
    data=data, t1=t1, t2=t2, phi=phi,
    q=q, L=L,
    analysis_mode='laminar_flow',
    nlsq_params=initial_params,
)

# If interrupted, resume from checkpoint:
# coordinator.resume_from_checkpoint()
```

### Multi-Modal Posteriors

For distributions with multiple modes:

```yaml
combination:
  method: simple_average  # More robust to multi-modality

validation:
  strict_mode: false  # Allow high KL divergence
  max_between_shard_kl: 10.0  # Relax threshold
```

**Post-processing:**

```python
# Check for multimodality
from homodyne.optimization.cmc.diagnostics import compute_combined_posterior_diagnostics

diagnostics = compute_combined_posterior_diagnostics(combined_posterior)
if diagnostics['is_multimodal']:
    print("⚠️ Posterior is multimodal")
    print("Consider using simple_average combination")
```

### Hierarchical Combination (Phase 2)

For very large numbers of shards (> 100):

```yaml
combination:
  method: hierarchical  # Tree-based combination
  tree_depth: 2         # 2-level hierarchy
```

**How it works:**
```
128 shards
    ↓
Combine in groups of 16 → 8 intermediate posteriors
    ↓
Combine 8 intermediates → Final posterior
```

---

## Summary

**Quick Reference:**

| Task | Command/Configuration |
|------|----------------------|
| **Auto CMC** | `fit_mcmc_jax(..., method='auto')` |
| **Force CMC** | `fit_mcmc_jax(..., method='cmc')` |
| **Check if CMC** | `result.is_cmc_result()` |
| **GPU backend** | `backend: {name: pjit}` |
| **CPU backend** | `backend: {name: multiprocessing}` |
| **Cluster backend** | `backend: {name: pbs}` |
| **Fast mode** | `svi_steps: 5000, num_warmup: 500` |
| **High quality** | `svi_steps: 20000, num_warmup: 1000` |

**Default Thresholds:**
- Minimum dataset size for CMC: 500k points
- GPU: 1M points per shard
- CPU: 2M points per shard
- Minimum success rate: 90%
- Maximum KL divergence: 2.0

For more information:
- **API Reference**: `docs/api/cmc_api.md`
- **Developer Guide**: `docs/developer_guide/cmc_architecture.md`
- **Troubleshooting**: `docs/troubleshooting/cmc_troubleshooting.md`
- **Migration Guide**: `docs/migration/v3_cmc_migration.md`
