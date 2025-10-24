# Consensus Monte Carlo API Reference

**Version:** 3.0+
**Last Updated:** 2025-10-24

---

## Table of Contents

1. [Overview](#overview)
2. [Main Entry Point](#main-entry-point)
3. [CMCCoordinator Class](#cmccoordinator-class)
4. [Sharding Module](#sharding-module)
5. [SVI Initialization Module](#svi-initialization-module)
6. [Backend Interface](#backend-interface)
7. [Combination Module](#combination-module)
8. [Diagnostics Module](#diagnostics-module)
9. [Configuration Schema](#configuration-schema)
10. [Extended MCMCResult](#extended-mcmcresult)

---

## Overview

The Consensus Monte Carlo (CMC) API provides a complete framework for scalable Bayesian inference on large XPCS datasets. The API is organized into several modules:

```
homodyne.optimization.cmc/
├── coordinator.py      # Main orchestrator (CMCCoordinator)
├── sharding.py         # Data partitioning
├── svi_init.py         # SVI initialization
├── backends/           # Execution backends (pjit, multiprocessing, PBS)
├── combination.py      # Subposterior combination
├── diagnostics.py      # Validation and diagnostics
└── result.py           # Extended MCMCResult class
```

---

## Main Entry Point

### `fit_mcmc_jax()`

**Module:** `homodyne.optimization.mcmc`

The primary user-facing function for MCMC sampling with automatic CMC support.

#### Signature

```python
def fit_mcmc_jax(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    analysis_mode: str,
    initial_params: Dict[str, float],
    method: str = 'auto',
    cmc_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MCMCResult
```

#### Parameters

- **`data`** (*np.ndarray*): Experimental c2 values (flattened)
  - Shape: `(n_points,)`
  - Must not contain NaN or Inf

- **`t1`** (*np.ndarray*): First time delay array
  - Shape: `(n_t1,)`
  - Units: seconds

- **`t2`** (*np.ndarray*): Second time delay array
  - Shape: `(n_t2,)`
  - Units: seconds

- **`phi`** (*np.ndarray*): Azimuthal angle array
  - Shape: `(n_phi,)`
  - Units: degrees, normalized to [-180°, 180°]

- **`q`** (*float*): Wavevector magnitude
  - Units: Å⁻¹
  - Typical range: 0.001 - 0.1

- **`L`** (*float*): Sample-detector distance (stator-rotor gap)
  - Units: Angstroms
  - Typical value: 2,000,000 Å

- **`analysis_mode`** (*str*): Analysis mode
  - Options: `'static_isotropic'` (5 parameters), `'laminar_flow'` (9 parameters)

- **`initial_params`** (*Dict[str, float]*): Initial parameter values
  - Keys: Parameter names (e.g., `'D0'`, `'alpha'`, `'D_offset'`)
  - Values: Initial guesses

- **`method`** (*str*, optional): MCMC method selection
  - Options: `'auto'` (default), `'nuts'`, `'cmc'`
  - Default: `'auto'` (automatic hardware-adaptive selection)

- **`cmc_config`** (*Dict[str, Any]*, optional): CMC configuration
  - If None, uses defaults from configuration system
  - See [Configuration Schema](#configuration-schema)

- **`**kwargs`**: Additional MCMC parameters
  - `num_warmup`, `num_samples`, `num_chains`, etc.

#### Returns

- **`MCMCResult`**: Extended MCMC result object
  - Standard fields: `mean_params`, `std_params`, `samples_params`, etc.
  - CMC fields (if CMC used): `num_shards`, `combination_method`, `per_shard_diagnostics`, `cmc_diagnostics`

#### Raises

- **`ValueError`**: Invalid method, data shape, or configuration
- **`RuntimeError`**: All shards failed to converge
- **`ImportError`**: CMC module not available

#### Examples

**Example 1: Automatic method selection**

```python
from homodyne.optimization.mcmc import fit_mcmc_jax

result = fit_mcmc_jax(
    data=c2_exp,
    t1=t1, t2=t2, phi=phi,
    q=0.0054, L=2000000,
    analysis_mode='static_isotropic',
    initial_params={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
)

# Check which method was used
if result.is_cmc_result():
    print(f"CMC used with {result.num_shards} shards")
else:
    print("Standard NUTS used")
```

**Example 2: Force CMC with configuration**

```python
cmc_config = {
    'sharding': {'strategy': 'stratified', 'num_shards': 16},
    'initialization': {'method': 'svi', 'svi_steps': 10000},
    'backend': {'name': 'pjit'},
    'combination': {'method': 'weighted', 'min_success_rate': 0.90},
}

result = fit_mcmc_jax(
    data=c2_exp,
    t1=t1, t2=t2, phi=phi,
    q=0.0054, L=2000000,
    analysis_mode='laminar_flow',
    initial_params={
        'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0,
        'gamma_dot_0': 0.01, 'beta': 1.0,
        'gamma_dot_offset': 0.0, 'phi_0': 0.0,
    },
    method='cmc',
    cmc_config=cmc_config,
)
```

---

## CMCCoordinator Class

**Module:** `homodyne.optimization.cmc.coordinator`

Main orchestrator for the CMC pipeline.

### Class Definition

```python
class CMCCoordinator:
    """Main coordinator for Consensus Monte Carlo."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize CMC coordinator.

        Parameters
        ----------
        config : dict
            Configuration dictionary with CMC and MCMC settings
        """

    def run_cmc(
        self,
        data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        analysis_mode: str,
        nlsq_params: Dict[str, float],
        model_fn: Optional[Callable] = None,
    ) -> MCMCResult:
        """Run complete CMC pipeline.

        Parameters
        ----------
        data : np.ndarray
            Experimental c2 values
        t1, t2, phi : np.ndarray
            Time and angle arrays
        q, L : float
            Physics parameters
        analysis_mode : str
            'static_isotropic' or 'laminar_flow'
        nlsq_params : dict
            Initial parameter values from NLSQ
        model_fn : callable, optional
            NumPyro model function

        Returns
        -------
        MCMCResult
            Extended MCMC result with CMC diagnostics
        """
```

### Methods

#### `run_cmc()`

Executes the 6-step CMC workflow:

1. **Create shards** - Partition data using specified strategy
2. **Run SVI initialization** - Estimate inverse mass matrix
3. **Execute parallel MCMC** - Run NUTS on each shard
4. **Combine subposteriors** - Weighted Gaussian product or averaging
5. **Validate results** - Check convergence criteria
6. **Return MCMCResult** - Package results with diagnostics

**Pipeline Flow:**

```
Input Data (50M points)
    ↓
[Step 1] Sharding → 50 shards × 1M points
    ↓
[Step 2] SVI Init → inv_mass_matrix
    ↓
[Step 3] Parallel MCMC → 50 subposteriors
    ↓
[Step 4] Combination → combined posterior
    ↓
[Step 5] Validation → diagnostics
    ↓
[Step 6] MCMCResult (extended)
```

### Attributes

- **`config`** (*dict*): Full configuration dictionary
- **`hardware_config`** (*HardwareConfig*): Detected hardware capabilities
- **`backend`** (*CMCBackend*): Selected execution backend
- **`checkpoint_manager`** (*CheckpointManager*, optional): Checkpoint manager (Phase 2)

### Example Usage

```python
from homodyne.optimization.cmc.coordinator import CMCCoordinator

# Create configuration
config = {
    'mcmc': {'num_warmup': 500, 'num_samples': 2000, 'num_chains': 1},
    'cmc': {
        'sharding': {'strategy': 'stratified', 'num_shards': 'auto'},
        'initialization': {'method': 'svi', 'svi_steps': 5000},
        'combination': {'method': 'weighted', 'fallback_enabled': True},
    },
}

# Initialize coordinator
coordinator = CMCCoordinator(config)

# Run CMC
result = coordinator.run_cmc(
    data=c2_exp,
    t1=t1, t2=t2, phi=phi,
    q=0.0054, L=2000000,
    analysis_mode='static_isotropic',
    nlsq_params={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
)

# Access results
print(f"Num shards: {result.num_shards}")
print(f"Combination method: {result.combination_method}")
print(f"Converged shards: {result.cmc_diagnostics['n_shards_converged']}")
```

---

## Sharding Module

**Module:** `homodyne.optimization.cmc.sharding`

Data partitioning functions for creating representative shards.

### Functions

#### `shard_data_stratified()`

Create stratified shards ensuring representative sampling across all dimensions.

```python
def shard_data_stratified(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    num_shards: int,
    q: float,
    L: float,
    strategy: str = 'stratified',
) -> List[Dict[str, Any]]:
    """Create stratified data shards.

    Parameters
    ----------
    data : np.ndarray
        Experimental c2 values
    t1, t2, phi : np.ndarray
        Time and angle arrays
    num_shards : int
        Number of shards to create
    q, L : float
        Physics parameters
    strategy : str
        Sharding strategy: 'stratified', 'random', 'contiguous'

    Returns
    -------
    List[Dict]
        List of shard dictionaries, each containing:
        - 'data': np.ndarray - Shard data
        - 't1', 't2', 'phi': np.ndarray - Coordinate arrays
        - 'q', 'L': float - Physics parameters
        - 'shard_id': int - Shard identifier
    """
```

**Sharding Strategies:**

- **`stratified`**: Stratified sampling across (t1, t2, phi) bins
  - Ensures each shard is representative of full dataset
  - Best for heterogeneous data

- **`random`**: Random permutation before splitting
  - Simpler and faster
  - Good for homogeneous data

- **`contiguous`**: Split into contiguous blocks
  - Fastest (no reordering)
  - Assumes data is pre-shuffled

#### `calculate_optimal_num_shards()`

Calculate optimal number of shards based on hardware and dataset size.

```python
def calculate_optimal_num_shards(
    dataset_size: int,
    hardware_config: HardwareConfig,
    target_shard_size_gpu: int = 1_000_000,
    target_shard_size_cpu: int = 2_000_000,
    min_shard_size: int = 10_000,
) -> int:
    """Calculate optimal number of shards.

    Parameters
    ----------
    dataset_size : int
        Total number of data points
    hardware_config : HardwareConfig
        Detected hardware configuration
    target_shard_size_gpu : int
        Target shard size for GPU (default: 1M)
    target_shard_size_cpu : int
        Target shard size for CPU (default: 2M)
    min_shard_size : int
        Minimum allowable shard size

    Returns
    -------
    int
        Optimal number of shards
    """
```

**Auto-Detection Logic:**

```python
# GPU: 1M points per shard
num_shards = dataset_size / 1_000_000

# CPU: 2M points per shard
num_shards = dataset_size / 2_000_000

# Cluster: Scale to available nodes
num_shards = num_nodes * cores_per_node / shards_per_core
```

#### `validate_shards()`

Validate shard quality and consistency.

```python
def validate_shards(
    shards: List[Dict[str, Any]],
    original_size: int,
    min_shard_size: int = 10_000,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate shards.

    Parameters
    ----------
    shards : List[Dict]
        List of shard dictionaries
    original_size : int
        Original dataset size
    min_shard_size : int
        Minimum allowable shard size

    Returns
    -------
    is_valid : bool
        True if all validation checks passed
    diagnostics : dict
        Validation diagnostics
    """
```

**Validation Checks:**

- All data points accounted for (sum of shard sizes = original size)
- No duplicate data points across shards
- All shards above minimum size threshold
- Consistent phi angle coverage across shards

### Example

```python
from homodyne.optimization.cmc.sharding import (
    shard_data_stratified,
    calculate_optimal_num_shards,
    validate_shards,
)
from homodyne.device.config import detect_hardware

# Calculate optimal shards
hw = detect_hardware()
num_shards = calculate_optimal_num_shards(
    dataset_size=len(data),
    hardware_config=hw,
)

# Create shards
shards = shard_data_stratified(
    data=c2_exp,
    t1=t1, t2=t2, phi=phi,
    num_shards=num_shards,
    q=0.0054, L=2000000,
    strategy='stratified',
)

# Validate
is_valid, diagnostics = validate_shards(shards, len(data))
assert is_valid, f"Shard validation failed: {diagnostics}"
```

---

## SVI Initialization Module

**Module:** `homodyne.optimization.cmc.svi_init`

Stochastic Variational Inference for NUTS initialization.

### Functions

#### `run_svi_initialization()`

Run SVI to estimate inverse mass matrix for NUTS.

```python
def run_svi_initialization(
    model_fn: Callable,
    pooled_data: Dict[str, np.ndarray],
    init_params: Dict[str, float],
    num_steps: int = 5000,
    learning_rate: float = 0.001,
    rank: int = 5,
    timeout: Optional[float] = None,
) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Run SVI initialization.

    Parameters
    ----------
    model_fn : callable
        NumPyro model function
    pooled_data : dict
        Pooled data from shards
    init_params : dict
        Initial parameter values
    num_steps : int
        Number of SVI optimization steps
    learning_rate : float
        Adam learning rate
    rank : int
        Low-rank approximation rank (1-10)
    timeout : float, optional
        Maximum SVI runtime in seconds

    Returns
    -------
    init_params : dict
        Optimized initial parameters
    inv_mass_matrix : np.ndarray or None
        Inverse mass matrix (None if SVI failed)
    """
```

**SVI Configuration:**

- **`num_steps`**: 5000-20000 (more steps = better convergence)
- **`learning_rate`**: 0.0001-0.01 (default: 0.001)
- **`rank`**: 1-10 (low-rank approximation, default: 5)
- **`timeout`**: 300-900 seconds (prevent infinite loops)

#### `pool_samples_from_shards()`

Pool representative samples from shards for SVI.

```python
def pool_samples_from_shards(
    shards: List[Dict[str, Any]],
    samples_per_shard: int = 200,
    strategy: str = 'uniform',
) -> Dict[str, np.ndarray]:
    """Pool samples from shards.

    Parameters
    ----------
    shards : List[Dict]
        List of shard dictionaries
    samples_per_shard : int
        Number of samples to draw per shard
    strategy : str
        Pooling strategy: 'uniform', 'weighted'

    Returns
    -------
    pooled_data : dict
        Pooled data dictionary with keys:
        - 'data': Pooled c2 values
        - 't1', 't2', 'phi': Pooled coordinates
        - 'q', 'L': Physics parameters
    """
```

**Pooling Strategies:**

- **`uniform`**: Sample uniformly from each shard
- **`weighted`**: Sample proportionally to shard size

### Example

```python
from homodyne.optimization.cmc.svi_init import (
    run_svi_initialization,
    pool_samples_from_shards,
)

# Pool samples
pooled_data = pool_samples_from_shards(
    shards=shards,
    samples_per_shard=200,
    strategy='uniform',
)

# Run SVI
init_params, inv_mass_matrix = run_svi_initialization(
    model_fn=create_numpyro_model('static_isotropic'),
    pooled_data=pooled_data,
    init_params={'D0': 10000.0, 'alpha': 0.8, 'D_offset': 100.0},
    num_steps=5000,
    learning_rate=0.001,
    rank=5,
    timeout=600,  # 10 minutes max
)

if inv_mass_matrix is not None:
    print("SVI succeeded, using estimated mass matrix")
else:
    print("SVI failed, using identity matrix")
```

---

## Backend Interface

**Module:** `homodyne.optimization.cmc.backends`

Abstract backend interface and selection logic.

### Backend Selection

#### `select_backend()`

Select optimal backend based on hardware configuration.

```python
def select_backend(
    hardware_config: HardwareConfig,
    user_override: Optional[str] = None,
) -> CMCBackend:
    """Select execution backend.

    Parameters
    ----------
    hardware_config : HardwareConfig
        Detected hardware capabilities
    user_override : str, optional
        Manual backend selection ('pjit', 'multiprocessing', 'pbs')

    Returns
    -------
    CMCBackend
        Selected backend instance

    Raises
    ------
    ValueError
        If user_override is invalid
    ImportError
        If requested backend is not available
    """
```

**Auto-Selection Logic:**

```python
if num_nodes > 1:
    backend = 'pbs' or 'slurm'  # Multi-node cluster
elif num_gpus > 1:
    backend = 'pjit'  # Multi-GPU
elif num_gpus == 1:
    backend = 'pjit'  # Single GPU (sequential)
else:
    backend = 'multiprocessing'  # CPU-only
```

### CMCBackend Base Class

Abstract base class for all backends.

```python
class CMCBackend(ABC):
    """Abstract base class for CMC backends."""

    @abstractmethod
    def run_parallel_mcmc(
        self,
        shards: List[Dict[str, Any]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Execute parallel MCMC on shards.

        Parameters
        ----------
        shards : List[Dict]
            List of data shards
        mcmc_config : dict
            MCMC configuration (num_warmup, num_samples, num_chains)
        init_params : dict
            Initial parameter values
        inv_mass_matrix : np.ndarray
            Inverse mass matrix for NUTS

        Returns
        -------
        List[Dict]
            Per-shard MCMC results
        """

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return backend name."""
```

### Available Backends

#### PjitBackend

**JAX pjit backend for GPU/CPU execution.**

```python
from homodyne.optimization.cmc.backends.pjit import PjitBackend

backend = PjitBackend()
shard_results = backend.run_parallel_mcmc(
    shards=shards,
    mcmc_config={'num_warmup': 500, 'num_samples': 2000, 'num_chains': 1},
    init_params=init_params,
    inv_mass_matrix=inv_mass_matrix,
)
```

**Features:**
- Parallel execution on multi-GPU systems
- Sequential execution on single GPU
- JAX-native implementation

#### MultiprocessingBackend

**Python multiprocessing for CPU parallelism.**

```python
from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend

backend = MultiprocessingBackend()
shard_results = backend.run_parallel_mcmc(...)
```

**Features:**
- CPU-only execution
- Uses `multiprocessing.Pool`
- Good for high-core-count workstations

#### PBSBackend

**PBS job array backend for HPC clusters.**

```python
from homodyne.optimization.cmc.backends.pbs import PBSBackend

backend = PBSBackend(
    project_name='my_project',
    walltime='02:00:00',
    cores_per_node=36,
)
shard_results = backend.run_parallel_mcmc(...)
```

**Features:**
- PBS Pro job submission
- Job array for parallel execution
- Checkpoint/resume capability

---

## Combination Module

**Module:** `homodyne.optimization.cmc.combination`

Subposterior combination methods.

### Functions

#### `combine_subposteriors()`

Combine shard posteriors into final posterior.

```python
def combine_subposteriors(
    shard_results: List[Dict[str, Any]],
    method: str = 'weighted',
    fallback_enabled: bool = True,
) -> Dict[str, Any]:
    """Combine subposteriors.

    Parameters
    ----------
    shard_results : List[Dict]
        Per-shard MCMC results with 'samples' key
    method : str
        Combination method: 'weighted', 'average', 'auto'
    fallback_enabled : bool
        Enable fallback to simple averaging if weighted fails

    Returns
    -------
    combined_posterior : dict
        Combined posterior with keys:
        - 'samples': Combined samples
        - 'mean': Posterior mean
        - 'cov': Posterior covariance
        - 'method': Method used ('weighted', 'average', 'single_shard')
    """
```

**Combination Methods:**

- **`weighted`**: Weighted Gaussian product (Scott et al. 2016)
  ```
  Given M shards with Gaussians N(μᵢ, Σᵢ):
  1. Compute precisions: Λᵢ = Σᵢ⁻¹
  2. Combined precision: Λ = ∑ᵢ Λᵢ
  3. Combined covariance: Σ = Λ⁻¹
  4. Combined mean: μ = Σ · (∑ᵢ Λᵢ μᵢ)
  ```

- **`average`**: Simple concatenation and resampling
  ```
  1. Concatenate all samples: S = [S₁, S₂, ..., Sₘ]
  2. Resample uniformly: sample N points from S
  3. Compute mean and covariance
  ```

- **`auto`**: Try weighted, fallback to average on failure

### Example

```python
from homodyne.optimization.cmc.combination import combine_subposteriors

# Combine posteriors
combined = combine_subposteriors(
    shard_results=shard_results,
    method='weighted',
    fallback_enabled=True,
)

# Access combined posterior
samples = combined['samples']  # Shape: (num_samples, num_params)
mean = combined['mean']        # Posterior mean
cov = combined['cov']          # Posterior covariance
method_used = combined['method']  # 'weighted' or 'average'
```

---

## Diagnostics Module

**Module:** `homodyne.optimization.cmc.diagnostics`

Validation and diagnostic functions.

### Functions

#### `validate_cmc_results()`

Validate CMC results against convergence criteria.

```python
def validate_cmc_results(
    shard_results: List[Dict[str, Any]],
    strict_mode: bool = True,
    min_success_rate: float = 0.90,
    max_kl_divergence: float = 2.0,
    max_rhat: float = 1.1,
    min_ess: float = 100.0,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate CMC results.

    Parameters
    ----------
    shard_results : List[Dict]
        Per-shard MCMC results
    strict_mode : bool
        If True, fail on validation errors; if False, log warnings only
    min_success_rate : float
        Minimum fraction of shards that must converge (0.0-1.0)
    max_kl_divergence : float
        Maximum KL divergence between shards
    max_rhat : float
        Maximum R-hat for convergence
    min_ess : float
        Minimum effective sample size

    Returns
    -------
    is_valid : bool
        True if validation passed (always True in lenient mode)
    diagnostics : dict
        Validation diagnostics with keys:
        - 'success_rate': Fraction of converged shards
        - 'max_kl_divergence': Maximum between-shard KL
        - 'validation_warnings': List of warning messages
        - 'validation_errors': List of error messages (strict mode only)
    """
```

#### `compute_per_shard_diagnostics()`

Compute convergence diagnostics for a single shard.

```python
def compute_per_shard_diagnostics(
    shard_result: Dict[str, Any],
    shard_idx: int,
) -> Dict[str, Any]:
    """Compute per-shard diagnostics.

    Parameters
    ----------
    shard_result : dict
        Shard MCMC result
    shard_idx : int
        Shard index

    Returns
    -------
    diagnostics : dict
        Per-shard diagnostics with keys:
        - 'shard_id': Shard index
        - 'n_samples': Number of samples
        - 'rhat': R-hat statistic
        - 'ess': Effective sample size
        - 'acceptance_rate': NUTS acceptance rate
    """
```

#### `compute_between_shard_kl_divergence()`

Compute KL divergence matrix between all shard pairs.

```python
def compute_between_shard_kl_divergence(
    shard_results: List[Dict[str, Any]]
) -> np.ndarray:
    """Compute KL divergence matrix.

    Parameters
    ----------
    shard_results : List[Dict]
        Per-shard MCMC results

    Returns
    -------
    kl_matrix : np.ndarray
        Symmetric matrix of pairwise KL divergences
        Shape: (num_shards, num_shards)
    """
```

**KL Divergence Formula (Gaussian approximation):**

```
For two Gaussians N(μₚ, Σₚ) and N(μᵧ, Σᵧ):

KL(p||q) = 0.5 * [
    trace(Σᵧ⁻¹ Σₚ) +
    (μᵧ - μₚ)ᵀ Σᵧ⁻¹ (μᵧ - μₚ) -
    k +
    log(det(Σᵧ) / det(Σₚ))
]

Symmetric KL = 0.5 * (KL(p||q) + KL(q||p))
```

### Example

```python
from homodyne.optimization.cmc.diagnostics import (
    validate_cmc_results,
    compute_per_shard_diagnostics,
    compute_between_shard_kl_divergence,
)

# Compute per-shard diagnostics
for i, shard_result in enumerate(shard_results):
    diag = compute_per_shard_diagnostics(shard_result, i)
    print(f"Shard {diag['shard_id']}: R-hat={diag['rhat']:.3f}, ESS={diag['ess']:.1f}")

# Compute KL divergence matrix
kl_matrix = compute_between_shard_kl_divergence(shard_results)
print(f"Max KL divergence: {np.max(kl_matrix):.3f}")

# Validate results
is_valid, diagnostics = validate_cmc_results(
    shard_results,
    strict_mode=True,
    min_success_rate=0.90,
    max_kl_divergence=2.0,
)

if not is_valid:
    print(f"Validation failed: {diagnostics['validation_errors']}")
```

---

## Configuration Schema

### CMCConfig TypedDict

**Module:** `homodyne.config.types`

Type-safe configuration schema.

```python
from typing import TypedDict, Literal, Optional

class CMCShardingConfig(TypedDict, total=False):
    strategy: Literal['stratified', 'random', 'contiguous']
    num_shards: int | Literal['auto']
    max_points_per_shard: int | Literal['auto']
    min_shard_size: int
    target_shard_size_gpu: int
    target_shard_size_cpu: int

class CMCInitializationConfig(TypedDict, total=False):
    method: Literal['svi', 'nlsq', 'identity']
    use_svi: bool
    svi_steps: int
    svi_learning_rate: float
    svi_rank: int
    svi_timeout: float
    fallback_to_identity: bool
    samples_per_shard: int

class CMCBackendConfig(TypedDict, total=False):
    name: Literal['auto', 'pjit', 'multiprocessing', 'pbs', 'slurm']
    type: str  # Alias for 'name'
    enable_checkpoints: bool
    checkpoint_frequency: int
    checkpoint_dir: str
    keep_last_checkpoints: int
    resume_from_checkpoint: bool

class CMCCombinationConfig(TypedDict, total=False):
    method: Literal['weighted_gaussian', 'simple_average', 'auto']
    validate_results: bool
    min_success_rate: float
    fallback_enabled: bool

class CMCPerShardMCMCConfig(TypedDict, total=False):
    num_warmup: int
    num_samples: int
    num_chains: int
    subsample_size: int | Literal['auto'] | None

class CMCValidationConfig(TypedDict, total=False):
    strict_mode: bool
    min_per_shard_ess: float
    max_per_shard_rhat: float
    max_between_shard_kl: float
    min_success_rate: float

class CMCConfig(TypedDict, total=False):
    enable: bool | Literal['auto']
    min_points_for_cmc: int
    sharding: CMCShardingConfig
    initialization: CMCInitializationConfig
    backend: CMCBackendConfig
    combination: CMCCombinationConfig
    per_shard_mcmc: CMCPerShardMCMCConfig
    validation: CMCValidationConfig
```

### Default Configuration

```python
DEFAULT_CMC_CONFIG = {
    'enable': 'auto',
    'min_points_for_cmc': 500000,
    'sharding': {
        'strategy': 'stratified',
        'num_shards': 'auto',
        'max_points_per_shard': 'auto',
        'min_shard_size': 10000,
        'target_shard_size_gpu': 1000000,
        'target_shard_size_cpu': 2000000,
    },
    'initialization': {
        'method': 'svi',
        'use_svi': True,
        'svi_steps': 5000,
        'svi_learning_rate': 0.001,
        'svi_rank': 5,
        'svi_timeout': 900,
        'fallback_to_identity': True,
        'samples_per_shard': 200,
    },
    'backend': {
        'name': 'auto',
        'enable_checkpoints': True,
        'checkpoint_frequency': 10,
        'keep_last_checkpoints': 3,
        'resume_from_checkpoint': True,
    },
    'combination': {
        'method': 'weighted_gaussian',
        'validate_results': True,
        'min_success_rate': 0.90,
        'fallback_enabled': True,
    },
    'per_shard_mcmc': {
        'num_warmup': 500,
        'num_samples': 2000,
        'num_chains': 1,
        'subsample_size': 'auto',
    },
    'validation': {
        'strict_mode': True,
        'min_per_shard_ess': 100.0,
        'max_per_shard_rhat': 1.1,
        'max_between_shard_kl': 2.0,
        'min_success_rate': 0.90,
    },
}
```

---

## Extended MCMCResult

**Module:** `homodyne.optimization.cmc.result`

Extended MCMC result class with CMC-specific fields.

### Class Definition

```python
@dataclass
class MCMCResult:
    """Extended MCMC result with CMC support.

    Standard MCMC Fields
    --------------------
    mean_params : np.ndarray
        Mean parameter values
    mean_contrast : float
        Mean contrast
    mean_offset : float
        Mean offset
    std_params : np.ndarray
        Parameter standard deviations
    std_contrast : float
        Contrast standard deviation
    std_offset : float
        Offset standard deviation
    samples_params : np.ndarray
        Parameter posterior samples
    samples_contrast : np.ndarray
        Contrast posterior samples
    samples_offset : np.ndarray
        Offset posterior samples
    converged : bool
        Overall convergence flag
    n_iterations : int
        Total iterations (warmup + samples)
    computation_time : float
        Total computation time (seconds)
    backend : str
        Backend used
    analysis_mode : str
        Analysis mode
    dataset_size : str
        Dataset size description
    n_chains : int
        Number of chains
    n_warmup : int
        Number of warmup iterations
    n_samples : int
        Number of samples
    sampler : str
        Sampler name
    acceptance_rate : float or None
        Acceptance rate
    r_hat : np.ndarray or None
        R-hat statistics
    effective_sample_size : np.ndarray or None
        Effective sample sizes

    CMC-Specific Fields (all optional, default=None)
    -------------------------------------------------
    per_shard_diagnostics : List[Dict] or None
        Per-shard convergence diagnostics
    cmc_diagnostics : Dict or None
        Overall CMC diagnostics
    combination_method : str or None
        Method used to combine posteriors
    num_shards : int or None
        Number of shards used
    """

    # Standard MCMC fields
    mean_params: np.ndarray
    mean_contrast: float
    mean_offset: float
    std_params: np.ndarray
    std_contrast: float
    std_offset: float
    samples_params: np.ndarray
    samples_contrast: np.ndarray
    samples_offset: np.ndarray
    converged: bool
    n_iterations: int
    computation_time: float
    backend: str
    analysis_mode: str
    dataset_size: str
    n_chains: int
    n_warmup: int
    n_samples: int
    sampler: str
    acceptance_rate: Optional[float] = None
    r_hat: Optional[np.ndarray] = None
    effective_sample_size: Optional[np.ndarray] = None

    # CMC-specific fields (optional)
    per_shard_diagnostics: Optional[List[Dict[str, Any]]] = None
    cmc_diagnostics: Optional[Dict[str, Any]] = None
    combination_method: Optional[str] = None
    num_shards: Optional[int] = None

    def is_cmc_result(self) -> bool:
        """Check if this is a CMC result.

        Returns
        -------
        bool
            True if num_shards > 1
        """
        return self.num_shards is not None and self.num_shards > 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-compatible).

        Returns
        -------
        dict
            Dictionary representation
        """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCMCResult':
        """Create from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary representation

        Returns
        -------
        MCMCResult
            Reconstructed result
        """
```

### Methods

- **`is_cmc_result()`**: Returns `True` if `num_shards > 1`
- **`to_dict()`**: Serialize to JSON-compatible dictionary
- **`from_dict()`**: Deserialize from dictionary

### Example

```python
from homodyne.optimization.cmc.result import MCMCResult

# Check if CMC result
if result.is_cmc_result():
    print(f"CMC result with {result.num_shards} shards")
    print(f"Combination method: {result.combination_method}")

    # Access per-shard diagnostics
    for diag in result.per_shard_diagnostics:
        print(f"Shard {diag['shard_id']}: converged={diag['converged']}")

    # Access CMC diagnostics
    print(f"Convergence rate: {result.cmc_diagnostics['convergence_rate']:.1%}")
    print(f"Max KL: {result.cmc_diagnostics.get('max_kl_divergence', 'N/A')}")

# Serialize
data = result.to_dict()
with open('result.json', 'w') as f:
    json.dump(data, f)

# Deserialize
with open('result.json', 'r') as f:
    data = json.load(f)
result_loaded = MCMCResult.from_dict(data)
```

---

## Summary

**Quick API Reference:**

| Task | API Call |
|------|----------|
| **Run CMC** | `fit_mcmc_jax(..., method='cmc')` |
| **Check if CMC** | `result.is_cmc_result()` |
| **Create shards** | `shard_data_stratified(...)` |
| **Run SVI** | `run_svi_initialization(...)` |
| **Combine posteriors** | `combine_subposteriors(...)` |
| **Validate results** | `validate_cmc_results(...)` |
| **Select backend** | `select_backend(hardware_config)` |

**Module Organization:**

```
homodyne.optimization.cmc/
├── coordinator.py           # CMCCoordinator (main orchestrator)
├── sharding.py              # Data partitioning
├── svi_init.py              # SVI initialization
├── backends/                # Execution backends
│   ├── base.py              # CMCBackend base class
│   ├── selection.py         # Backend selection logic
│   ├── pjit.py              # JAX pjit backend
│   ├── multiprocessing.py  # CPU multiprocessing backend
│   └── pbs.py               # PBS cluster backend
├── combination.py           # Subposterior combination
├── diagnostics.py           # Validation and diagnostics
└── result.py                # Extended MCMCResult class
```

For more information:
- **User Guide**: `docs/user-guide/cmc_guide.md`
- **Developer Guide**: `docs/developer-guide/cmc_architecture.md`
- **Troubleshooting**: `docs/troubleshooting/cmc_troubleshooting.md`
