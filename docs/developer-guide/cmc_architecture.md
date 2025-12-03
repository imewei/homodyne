# Consensus Monte Carlo Architecture - Developer Guide

**Version:** 3.0+ **Last Updated:** 2025-10-24 **Target Audience:** Contributors,
Maintainers, Advanced Users

______________________________________________________________________

## Table of Contents

1. [Architecture Overview](#architecture-overview)
1. [Module Structure](#module-structure)
1. [Design Principles](#design-principles)
1. [Core Components](#core-components)
1. [Backend Implementation](#backend-implementation)
1. [Adding New Features](#adding-new-features)
1. [Testing Guidelines](#testing-guidelines)
1. [Code Style](#code-style)
1. [Contributing](#contributing)
1. [Performance Optimization](#performance-optimization)

______________________________________________________________________

## Architecture Overview

### System Design Philosophy

Consensus Monte Carlo (CMC) is built on three foundational principles:

1. **Modularity**: Each component has a single, well-defined responsibility
1. **Extensibility**: Easy to add new backends, combination methods, or sharding
   strategies
1. **Reliability**: Comprehensive validation, error handling, and fallback mechanisms

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Entry Point                             │
│             fit_mcmc_jax(method='auto')                         │
│                 (homodyne/optimization/mcmc.py)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────┐
         │  Dataset Size + Hardware   │
         │   should_use_cmc()         │
         │  (homodyne/device/config.py)│
         └────────┬──────────────────┘
                  │
       ┌──────────┴──────────┐
       │                     │
       ▼                     ▼
   Standard NUTS      CMC Pipeline
   (existing)         (CMCCoordinator)
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
    ┌──────────────────┐        ┌─────────────────────┐
    │  Task Group 1-3  │        │   Task Group 4-6    │
    │  Infrastructure  │        │   Execution         │
    └──────────────────┘        └─────────────────────┘
              │                            │
              │        CMCCoordinator      │
              │                            │
    ┌─────────▼────────────────────────────▼─────────┐
    │  STEP 1: Hardware Detection                    │
    │  STEP 2: Data Sharding                         │
    │  STEP 3: SVI Initialization                    │
    │  STEP 4: Parallel MCMC Execution               │
    │  STEP 5: Subposterior Combination              │
    │  STEP 6: Validation & Results Packaging        │
    └────────────────────┬───────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Extended        │
              │  MCMCResult      │
              └──────────────────┘
```

### Data Flow

```
Input
  ↓
c2_exp (50M points), t1, t2, phi, q, L
  ↓
Hardware Detection → HardwareConfig
  ↓
Data Sharding → List[Shard] (50 shards × 1M points)
  ↓
SVI Init → (init_params, inv_mass_matrix)
  ↓
Parallel MCMC → List[ShardResult] (50 subposteriors)
  ↓
Combination → CombinedPosterior (samples, mean, cov)
  ↓
Validation → Diagnostics (convergence, KL divergence)
  ↓
Output → MCMCResult (extended with CMC fields)
```

______________________________________________________________________

## Module Structure

### Directory Layout

```
homodyne/optimization/cmc/
├── __init__.py              # Public API exports
├── coordinator.py           # Main CMC orchestrator (648 lines)
├── sharding.py              # Data partitioning (450 lines)
├── svi_init.py              # SVI initialization (380 lines)
├── combination.py           # Subposterior combination (509 lines)
├── diagnostics.py           # Validation and diagnostics (820 lines)
├── result.py                # Extended MCMCResult (449 lines)
└── backends/
    ├── __init__.py          # Backend registry (84 lines)
    ├── base.py              # CMCBackend base class (361 lines)
    ├── selection.py         # Backend selection logic (333 lines)
    ├── pjit.py              # JAX pjit backend (527 lines)
    ├── multiprocessing.py   # CPU multiprocessing backend (418 lines)
    └── pbs.py               # PBS cluster backend (631 lines)
```

**Total Lines of Code:** ~5,600 lines (production code only)

### Module Responsibilities

| Module | Responsibility | Lines | Key Functions |
|--------|---------------|-------|---------------| | `coordinator.py` | Orchestrate CMC
pipeline | 648 | `CMCCoordinator.run_cmc()` | | `sharding.py` | Data partitioning | 450
| `shard_data_stratified()` | | `svi_init.py` | NUTS initialization | 380 |
`run_svi_initialization()` | | `combination.py` | Posterior combination | 509 |
`combine_subposteriors()` | | `diagnostics.py` | Validation | 820 |
`validate_cmc_results()` | | `result.py` | Result data class | 449 | `MCMCResult` | |
`backends/base.py` | Abstract backend | 361 | `CMCBackend` | | `backends/selection.py` |
Backend selection | 333 | `select_backend()` | | `backends/pjit.py` | GPU execution |
527 | `PjitBackend` | | `backends/multiprocessing.py` | CPU execution | 418 |
`MultiprocessingBackend` | | `backends/pbs.py` | Cluster execution | 631 | `PBSBackend`
|

### Dependency Graph

```
coordinator.py
    ├─→ homodyne.device.config (hardware detection)
    ├─→ backends.selection (backend selection)
    ├─→ sharding (data partitioning)
    ├─→ svi_init (initialization)
    ├─→ combination (posterior combination)
    ├─→ diagnostics (validation) [Phase 1: basic, Phase 2: full]
    └─→ result (MCMCResult)

backends/selection.py
    ├─→ homodyne.device.config (HardwareConfig)
    └─→ backends/{pjit, multiprocessing, pbs} (lazy import)

backends/pjit.py
    ├─→ jax.experimental.pjit (JAX parallelism)
    ├─→ numpyro (MCMC sampling)
    └─→ backends.base (CMCBackend)

sharding.py
    ├─→ numpy (array operations)
    └─→ homodyne.device.config (optimal shard calculation)

svi_init.py
    ├─→ numpyro.infer.autoguide (SVI)
    ├─→ jax.random (RNG)
    └─→ numpy (array operations)

combination.py
    ├─→ numpy (linear algebra)
    └─→ scipy.stats (Gaussian fitting)

diagnostics.py
    ├─→ numpyro.diagnostics (R-hat, ESS)
    └─→ numpy (KL divergence calculations)
```

______________________________________________________________________

## Design Principles

### 1. Separation of Concerns

Each module has a single, well-defined responsibility:

- **Coordinator**: Pipeline orchestration (no direct computation)
- **Sharding**: Data partitioning only (no MCMC execution)
- **SVI Init**: Initialization only (no sharding or combination)
- **Backends**: Parallel execution only (no data validation)
- **Combination**: Posterior combination only (no validation)
- **Diagnostics**: Validation only (no execution)

**Anti-Pattern Example (violates SoC):**

```python
# ❌ BAD: Sharding module running MCMC
def shard_and_run_mcmc(data, ...):
    shards = create_shards(data)
    results = run_mcmc_on_shards(shards)  # Wrong! This belongs in backend
    return results
```

**Good Example:**

```python
# ✅ GOOD: Sharding module only shards
def shard_data_stratified(data, ...):
    shards = create_shards(data)
    return shards  # Backend will run MCMC separately
```

### 2. Extensibility Through Abstraction

**Backend Pattern:**

```python
# Abstract base class defines interface
class CMCBackend(ABC):
    @abstractmethod
    def run_parallel_mcmc(self, shards, ...) -> List[Dict]:
        """Execute MCMC on shards."""

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return backend identifier."""

# Concrete implementations extend base
class PjitBackend(CMCBackend):
    def run_parallel_mcmc(self, shards, ...):
        # GPU-specific implementation
        ...

class MultiprocessingBackend(CMCBackend):
    def run_parallel_mcmc(self, shards, ...):
        # CPU-specific implementation
        ...
```

**Benefits:**

- New backends require only 2 methods
- No changes to coordinator needed
- Compile-time interface validation

### 3. Fail-Fast with Graceful Degradation

**Validation Strategy:**

```python
# Step 1: Validate inputs early (fail-fast)
if len(data) == 0:
    raise ValueError("Cannot run CMC on empty dataset")

# Step 2: Try optimal method
try:
    combined = weighted_gaussian_product(shard_results)
except np.linalg.LinAlgError:
    # Step 3: Fallback gracefully
    logger.warning("Weighted method failed, falling back to simple averaging")
    combined = simple_averaging(shard_results)
```

**Fallback Chain:**

```
Weighted Gaussian Product (optimal)
    ↓ (if singular matrices)
Simple Averaging (robust)
    ↓ (if all shards fail)
RuntimeError with clear message
```

### 4. Observable Execution

**Logging Philosophy:**

- **INFO**: High-level progress (shard creation, combination complete)
- **DEBUG**: Detailed execution (SVI convergence, backend selection)
- **WARNING**: Fallbacks and suboptimal choices
- **ERROR**: Validation failures and exceptions

**Progress Tracking:**

```python
# Coordinator logs each step
logger.info("=" * 70)
logger.info("STEP 1: Creating data shards")
logger.info("=" * 70)
shards = shard_data_stratified(...)
logger.info(f"✓ Created {len(shards)} shards using {strategy} strategy")
```

### 5. Stateless Components

**Design Choice:**

All modules are **stateless** - they don't maintain internal state between calls.

**Benefits:**

- Thread-safe by default
- Easy to test (no setup/teardown)
- Supports retry logic naturally
- No hidden dependencies

**Example:**

```python
# ✅ GOOD: Stateless backend
class PjitBackend(CMCBackend):
    def run_parallel_mcmc(self, shards, mcmc_config, ...):
        # No self.state, self.cache, etc.
        # All state passed as arguments
        results = []
        for shard in shards:
            result = self._run_single_shard(shard, mcmc_config, ...)
            results.append(result)
        return results

# ❌ BAD: Stateful backend
class StatefulBackend(CMCBackend):
    def __init__(self):
        self.cache = {}  # Shared state (problematic)

    def run_parallel_mcmc(self, shards, ...):
        # Uses self.cache (not thread-safe)
        ...
```

______________________________________________________________________

## Core Components

### 1. CMCCoordinator

**File:** `homodyne/optimization/cmc/coordinator.py`

**Architecture:**

```python
class CMCCoordinator:
    def __init__(self, config):
        # 1. Detect hardware
        self.hardware_config = detect_hardware()

        # 2. Select backend
        self.backend = select_backend(self.hardware_config, ...)

        # 3. Initialize checkpoint manager (Phase 2)
        self.checkpoint_manager = None

    def run_cmc(self, data, t1, t2, phi, q, L, analysis_mode, nlsq_params):
        # STEP 1: Create shards
        shards = self._create_shards(data, ...)

        # STEP 2: SVI initialization
        init_params, inv_mass_matrix = self._run_svi(shards, ...)

        # STEP 3: Parallel MCMC
        shard_results = self.backend.run_parallel_mcmc(shards, ...)

        # STEP 4: Combine posteriors
        combined_posterior = self._combine_posteriors(shard_results, ...)

        # STEP 5: Validate results
        is_valid, diagnostics = self._validate_results(combined_posterior, ...)

        # STEP 6: Package results
        result = self._create_mcmc_result(combined_posterior, ...)

        return result
```

**Design Decisions:**

1. **Single entry point**: `run_cmc()` orchestrates entire pipeline
1. **Private helper methods**: `_create_shards()`, `_run_svi()`, etc. (not public API)
1. **Hardware-adaptive**: Auto-detects hardware in `__init__()`
1. **Error handling**: Each step has try/except with fallbacks

### 2. Data Sharding

**File:** `homodyne/optimization/cmc/sharding.py`

**Stratified Sharding Algorithm:**

```python
def shard_data_stratified(data, t1, t2, phi, num_shards, ...):
    """
    Goal: Ensure each shard is representative of full dataset.

    Algorithm:
    1. Create bins for each dimension (t1, t2, phi)
    2. For each bin:
        a. Get all data points in this bin
        b. Split into num_shards parts (round-robin)
        c. Assign each part to a different shard
    3. Validate shard sizes and coverage

    Example (simplified 1D):
        data = [0,1,2,3,4,5,6,7,8,9]  (10 points)
        num_shards = 2
        bins = [[0,1,2], [3,4,5], [6,7,8,9]]  (3 bins)

        Shard 0: bin[0][0::2] + bin[1][0::2] + bin[2][0::2]
               = [0, 2] + [3, 5] + [6, 8]
               = [0,2,3,5,6,8]

        Shard 1: bin[0][1::2] + bin[1][1::2] + bin[2][1::2]
               = [1] + [4] + [7, 9]
               = [1,4,7,9]

    Both shards have data from all bins (representative).
    """
```

**Validation Checks:**

```python
def validate_shards(shards, original_size):
    # Check 1: All data points accounted for
    total_size = sum(len(s['data']) for s in shards)
    assert total_size == original_size

    # Check 2: No duplicates
    all_indices = np.concatenate([s['indices'] for s in shards])
    assert len(np.unique(all_indices)) == original_size

    # Check 3: Minimum shard size
    min_size = min(len(s['data']) for s in shards)
    assert min_size >= MIN_SHARD_SIZE

    # Check 4: Phi coverage
    for shard in shards:
        unique_phi = np.unique(shard['phi'])
        assert len(unique_phi) >= MIN_PHI_ANGLES  # At least 3 angles
```

### 3. SVI Initialization

**File:** `homodyne/optimization/cmc/svi_init.py`

**SVI Algorithm:**

```python
def run_svi_initialization(model_fn, pooled_data, init_params, num_steps, ...):
    """
    Stochastic Variational Inference for inverse mass matrix estimation.

    Algorithm (NumPyro AutoGuide):
    1. Create variational guide (low-rank normal)
        guide = AutoLowRankMultivariateNormal(model_fn, rank=5)

    2. Setup SVI optimizer
        optimizer = Adam(learning_rate=0.001)
        svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())

    3. Run optimization
        for step in range(num_steps):
            loss = svi.step(rng_key, pooled_data)

    4. Extract inverse mass matrix
        guide_params = guide.get_params()
        cov_factor = guide_params['auto_loc']
        cov_diag = guide_params['auto_scale']
        inv_mass_matrix = compute_inverse_mass(cov_factor, cov_diag)

    5. Return init_params (guide median) and inv_mass_matrix

    Low-Rank Approximation:
        Σ ≈ LL^T + D
        where L is (n x rank) and D is diagonal

        This reduces parameters from O(n²) to O(n*rank).
        For rank=5, n=9: 45 + 9 = 54 params instead of 81.
    """
```

**Pooling Strategy:**

```python
def pool_samples_from_shards(shards, samples_per_shard=200):
    """
    Pool representative samples for SVI.

    Goal: Get ~10k total samples (50 shards × 200 samples each).

    Sampling Strategy:
    1. Uniform random sampling from each shard
    2. Equal samples per shard (unbiased)
    3. Stratified by phi angle within each shard

    Example:
        Shard 0 (1M points):
            Sample 200 points uniformly
            Ensure 3 phi angles × 67 points each ≈ 200 total

        Shard 1 (1M points):
            Sample 200 points uniformly
            ...

        Pooled dataset: 10,000 points (representative of 50M)
    """
```

### 4. Subposterior Combination

**File:** `homodyne/optimization/cmc/combination.py`

**Weighted Gaussian Product (Scott et al. 2016):**

```python
def _weighted_gaussian_product(shard_results):
    """
    Combine M Gaussian posteriors N(μᵢ, Σᵢ) into single posterior N(μ, Σ).

    Algorithm:
    1. Fit Gaussian to each shard's samples:
        for each shard i:
            μᵢ = np.mean(samples_i, axis=0)
            Σᵢ = np.cov(samples_i.T)

    2. Add regularization for numerical stability:
        Σᵢ_reg = Σᵢ + 1e-6 * I

    3. Compute precision matrices:
        Λᵢ = inv(Σᵢ_reg)

    4. Combine precisions (weighted sum):
        Λ = Σᵢ Λᵢ

    5. Compute combined covariance:
        Σ = inv(Λ)

    6. Compute combined mean:
        μ = Σ · (Σᵢ Λᵢ μᵢ)

    7. Sample from N(μ, Σ):
        samples = np.random.multivariate_normal(μ, Σ, size=num_samples)

    Numerical Stability:
    - Use slogdet() for log determinant (avoid overflow)
    - Regularize small eigenvalues (Σ += 1e-6 * I)
    - Check positive definiteness (all eigenvalues > 0)
    - Fallback to simple averaging if inversion fails

    Complexity:
    - O(M * K³) for M shards, K parameters
    - Example: 50 shards, 9 params → ~366k flops
    """
```

**Simple Averaging (Fallback):**

```python
def _simple_averaging(shard_results):
    """
    Robust fallback for non-Gaussian posteriors.

    Algorithm:
    1. Concatenate all samples:
        all_samples = np.vstack([s['samples'] for s in shard_results])

    2. Resample uniformly:
        indices = np.random.choice(len(all_samples), size=num_samples)
        samples = all_samples[indices]

    3. Compute mean and covariance:
        μ = np.mean(samples, axis=0)
        Σ = np.cov(samples.T)

    Benefits:
    - No assumptions about posterior shape
    - Handles multi-modal distributions
    - Always succeeds (no matrix inversions)

    Trade-offs:
    - Less statistically efficient (wider uncertainties)
    - Doesn't weight by shard precision
    """
```

### 5. Diagnostics and Validation

**File:** `homodyne/optimization/cmc/diagnostics.py`

**KL Divergence Calculation:**

```python
def _kl_divergence_gaussian(mu_p, Sigma_p, mu_q, Sigma_q):
    """
    KL divergence between two multivariate Gaussians.

    Formula:
        KL(p||q) = 0.5 * [
            trace(Σ_q^{-1} Σ_p) +
            (μ_q - μ_p)^T Σ_q^{-1} (μ_q - μ_p) -
            k +
            log(det(Σ_q) / det(Σ_p))
        ]

    Symmetric KL (used for comparison):
        KL_sym(p, q) = 0.5 * (KL(p||q) + KL(q||p))

    Numerical Stability:
    1. Regularization:
        Σ_p += 1e-6 * I
        Σ_q += 1e-6 * I

    2. Log determinant:
        sign_p, logdet_p = np.linalg.slogdet(Sigma_p)
        sign_q, logdet_q = np.linalg.slogdet(Sigma_q)
        log_det_ratio = logdet_q - logdet_p

    3. Pseudoinverse fallback:
        try:
            Sigma_q_inv = np.linalg.inv(Sigma_q)
        except np.linalg.LinAlgError:
            Sigma_q_inv = np.linalg.pinv(Sigma_q)

    Interpretation:
    - KL < 0.5: Shards agree very well (excellent)
    - KL < 2.0: Shards agree reasonably (good)
    - KL < 5.0: Some disagreement (warning)
    - KL > 5.0: Shards diverged (error)
    """
```

______________________________________________________________________

## Backend Implementation

### Abstract Backend Interface

**File:** `homodyne/optimization/cmc/backends/base.py`

**Design:**

```python
class CMCBackend(ABC):
    """Abstract base class for all CMC backends.

    Design Principles:
    1. Stateless: No shared state between calls
    2. Minimal interface: Only 2 required methods
    3. Observable: Logging hooks for progress tracking
    4. Error-tolerant: Failed shards don't crash pipeline
    """

    @abstractmethod
    def run_parallel_mcmc(
        self,
        shards: List[Dict],
        mcmc_config: Dict,
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
    ) -> List[Dict]:
        """Execute parallel MCMC on shards.

        Implementation Requirements:
        1. Process all shards (sequential or parallel)
        2. Use provided init_params and inv_mass_matrix
        3. Return results in same order as input shards
        4. Include 'converged' flag in each result
        5. Log progress using _log_shard_start/_log_shard_complete

        Error Handling:
        - Don't raise on single shard failure
        - Mark shard as converged=False
        - Log error but continue with remaining shards
        """

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return backend identifier.

        Must be one of: 'pjit', 'multiprocessing', 'pbs', 'slurm'
        """

    # Common utilities (provided by base class)
    def _log_shard_start(self, shard_idx, total_shards):
        """Log shard execution start."""

    def _log_shard_complete(self, shard_idx, total_shards, elapsed_time):
        """Log shard execution completion."""

    def _validate_shard_result(self, result, shard_idx):
        """Validate shard result format."""

    def _handle_shard_error(self, error, shard_idx):
        """Handle shard execution errors."""
```

### PjitBackend Implementation

**File:** `homodyne/optimization/cmc/backends/pjit.py`

**Key Design:**

```python
class PjitBackend(CMCBackend):
    """JAX pjit backend for GPU/CPU execution.

    Execution Modes:
    1. Multi-GPU: Parallel execution using pjit
    2. Single GPU: Sequential execution
    3. CPU: Sequential execution (no pjit overhead)

    Implementation:
    - Uses JAX pjit for device parallelism
    - Automatically detects available devices
    - Falls back to sequential if pjit unavailable
    """

    def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix):
        devices = jax.devices()

        if len(devices) > 1:
            # Multi-device parallel execution
            return self._run_parallel_pjit(shards, ...)
        else:
            # Single-device sequential execution
            return self._run_sequential(shards, ...)

    def _run_parallel_pjit(self, shards, ...):
        """Parallel execution using JAX pjit.

        Strategy:
        1. Shard data across devices
        2. Replicate MCMC function on all devices
        3. Execute in parallel using pjit
        4. Gather results from all devices

        JAX pjit API:
            @jax.pjit(
                in_axis_resources=(P('data'),),
                out_axis_resources=P('data'),
            )
            def run_mcmc_sharded(shards_sharded):
                # Runs on all devices in parallel
                return [run_mcmc(shard) for shard in shards_sharded]
        """

    def _run_sequential(self, shards, ...):
        """Sequential execution on single device.

        Simpler fallback for single GPU or CPU.
        """
```

### MultiprocessingBackend Implementation

**File:** `homodyne/optimization/cmc/backends/multiprocessing.py`

**Key Design:**

```python
class MultiprocessingBackend(CMCBackend):
    """Python multiprocessing backend for CPU parallelism.

    Uses multiprocessing.Pool for parallel execution across CPU cores.

    Design:
    - One process per shard (up to cpu_count)
    - Each process runs NUTS independently
    - Results collected via Pool.map()
    """

    def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix):
        # Determine number of workers
        num_workers = min(len(shards), multiprocessing.cpu_count())

        # Create worker function
        def worker_fn(shard_with_config):
            shard, config, params, mass = shard_with_config
            return self._run_single_shard(shard, config, params, mass)

        # Prepare arguments for each shard
        args = [
            (shard, mcmc_config, init_params, inv_mass_matrix)
            for shard in shards
        ]

        # Execute in parallel
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(worker_fn, args)

        return results

    def _run_single_shard(self, shard, mcmc_config, init_params, inv_mass_matrix):
        """Run NUTS on single shard.

        Called in separate process.
        Must be pickleable (no closures or JAX JIT).
        """
```

### PBSBackend Implementation

**File:** `homodyne/optimization/cmc/backends/pbs.py`

**Key Design:**

```python
class PBSBackend(CMCBackend):
    """PBS job array backend for HPC clusters.

    Workflow:
    1. Generate PBS job script with array directive
    2. Submit job array to cluster queue
    3. Each array element runs one shard
    4. Poll for job completion
    5. Load results from shared filesystem

    PBS Job Script Template:
        #!/bin/bash
        #PBS -N cmc_job
        #PBS -t 0-49           # 50 array elements (one per shard)
        #PBS -l walltime=02:00:00
        #PBS -l nodes=1:ppn=36
        #PBS -A project_name

        # Run shard $PBS_ARRAYID
        python run_shard.py $PBS_ARRAYID
    """

    def run_parallel_mcmc(self, shards, ...):
        # Step 1: Serialize shards to disk
        self._save_shards_to_disk(shards, work_dir)

        # Step 2: Generate PBS job script
        job_script = self._generate_pbs_script(num_shards=len(shards))

        # Step 3: Submit job array
        job_id = self._submit_pbs_job(job_script)

        # Step 4: Poll for completion
        self._wait_for_job_completion(job_id, timeout=7200)  # 2 hours

        # Step 5: Load results from disk
        results = self._load_results_from_disk(work_dir, num_shards=len(shards))

        return results

    def _generate_pbs_script(self, num_shards):
        """Generate PBS job script for array job."""

    def _submit_pbs_job(self, job_script):
        """Submit job using qsub command."""

    def _wait_for_job_completion(self, job_id, timeout):
        """Poll qstat until job completes."""
```

______________________________________________________________________

## Adding New Features

### Adding a New Backend

**Steps:**

1. **Create backend file**:

```bash
touch homodyne/optimization/cmc/backends/ray.py
```

2. **Implement CMCBackend interface**:

```python
from homodyne.optimization.cmc.backends.base import CMCBackend

class RayBackend(CMCBackend):
    def run_parallel_mcmc(self, shards, mcmc_config, init_params, inv_mass_matrix):
        import ray

        # Initialize Ray cluster
        ray.init()

        # Define remote function
        @ray.remote
        def run_shard_remote(shard, config, params, mass):
            return self._run_single_shard(shard, config, params, mass)

        # Submit tasks
        futures = [
            run_shard_remote.remote(shard, mcmc_config, init_params, inv_mass_matrix)
            for shard in shards
        ]

        # Gather results
        results = ray.get(futures)

        ray.shutdown()
        return results

    def get_backend_name(self):
        return "ray"
```

3. **Register in backend registry**:

```python
# homodyne/optimization/cmc/backends/__init__.py
_BACKEND_REGISTRY = {
    "pjit": "homodyne.optimization.cmc.backends.pjit.PjitBackend",
    "multiprocessing": "homodyne.optimization.cmc.backends.multiprocessing.MultiprocessingBackend",
    "pbs": "homodyne.optimization.cmc.backends.pbs.PBSBackend",
    "ray": "homodyne.optimization.cmc.backends.ray.RayBackend",  # NEW
}
```

4. **Add tests**:

```python
# tests/unit/test_ray_backend.py
def test_ray_backend_execution():
    backend = RayBackend()
    results = backend.run_parallel_mcmc(shards, ...)
    assert len(results) == len(shards)
```

5. **Update documentation**:

- Add to API reference
- Add to user guide
- Update configuration template

### Adding a New Combination Method

**Steps:**

1. **Implement combination function**:

```python
# homodyne/optimization/cmc/combination.py

def _hierarchical_combination(shard_results, tree_depth=2):
    """Hierarchical combination for large number of shards.

    Algorithm:
    1. Group shards into clusters of size ~sqrt(M)
    2. Combine each cluster using weighted Gaussian product
    3. Combine cluster results recursively until single posterior

    Example (128 shards, depth=2):
        Level 0: 128 shards
        Level 1: 128 → 16 clusters × 8 shards each → 16 intermediate posteriors
        Level 2: 16 → 1 cluster × 16 posteriors → 1 final posterior
    """
    if len(shard_results) <= 10:
        # Base case: use weighted method directly
        return _weighted_gaussian_product(shard_results)

    # Recursive case: cluster and combine
    cluster_size = int(np.sqrt(len(shard_results)))
    clusters = [
        shard_results[i:i+cluster_size]
        for i in range(0, len(shard_results), cluster_size)
    ]

    # Combine each cluster
    cluster_results = [
        _weighted_gaussian_product(cluster)
        for cluster in clusters
    ]

    # Recursively combine cluster results
    return _hierarchical_combination(cluster_results, tree_depth - 1)
```

2. **Update combine_subposteriors()**:

```python
def combine_subposteriors(shard_results, method='weighted', ...):
    if method == 'weighted':
        return _weighted_gaussian_product(shard_results)
    elif method == 'average':
        return _simple_averaging(shard_results)
    elif method == 'hierarchical':  # NEW
        return _hierarchical_combination(shard_results)
    else:
        raise ValueError(f"Invalid method: {method}")
```

3. **Add configuration support**:

```yaml
# homodyne/config/templates/homodyne_cmc_config.yaml
combination:
  method: hierarchical  # NEW option
  tree_depth: 2         # NEW parameter
```

4. **Add tests**:

```python
def test_hierarchical_combination():
    # Create 128 synthetic shard results
    shard_results = [create_synthetic_shard() for _ in range(128)]

    # Test hierarchical combination
    combined = _hierarchical_combination(shard_results, tree_depth=2)

    # Verify result
    assert 'samples' in combined
    assert combined['method'] == 'hierarchical'
```

### Adding a New Sharding Strategy

**Steps:**

1. **Implement sharding function**:

```python
# homodyne/optimization/cmc/sharding.py

def shard_data_time_stratified(data, t1, t2, phi, num_shards, ...):
    """Time-stratified sharding (stratify by t1, t2 only, ignore phi).

    Use case: When phi coverage is already uniform.

    Algorithm:
    1. Create 2D bins for (t1, t2)
    2. Stratify samples across time bins only
    3. Distribute to shards round-robin
    """
    # Create time bins
    t1_bins = np.linspace(t1.min(), t1.max(), num=10)
    t2_bins = np.linspace(t2.min(), t2.max(), num=10)

    # Bin data
    t1_indices = np.digitize(t1, t1_bins)
    t2_indices = np.digitize(t2, t2_bins)

    # Stratified sampling
    shards = [{'data': [], 'shard_id': i} for i in range(num_shards)]
    for t1_idx in range(len(t1_bins)):
        for t2_idx in range(len(t2_bins)):
            # Get points in this bin
            mask = (t1_indices == t1_idx) & (t2_indices == t2_idx)
            bin_data = data[mask]

            # Distribute to shards
            for i, point in enumerate(bin_data):
                shard_idx = i % num_shards
                shards[shard_idx]['data'].append(point)

    return shards
```

2. **Update shard_data_stratified() dispatcher**:

```python
def shard_data_stratified(data, t1, t2, phi, num_shards, q, L, strategy='stratified'):
    if strategy == 'stratified':
        return _shard_stratified_3d(data, t1, t2, phi, num_shards, q, L)
    elif strategy == 'random':
        return _shard_random(data, t1, t2, phi, num_shards, q, L)
    elif strategy == 'contiguous':
        return _shard_contiguous(data, t1, t2, phi, num_shards, q, L)
    elif strategy == 'time_stratified':  # NEW
        return shard_data_time_stratified(data, t1, t2, phi, num_shards, q, L)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
```

______________________________________________________________________

## Testing Guidelines

### Test Organization

```
tests/unit/
├── test_backend_infrastructure.py  # Backend selection, registry (21 tests)
├── test_backend_implementations.py # Pjit, multiprocessing, PBS (62 tests)
├── test_sharding.py                # Data partitioning (28 tests)
├── test_svi_initialization.py      # SVI init (18 tests)
├── test_combination.py             # Posterior combination (20 tests)
├── test_diagnostics.py             # Validation (25 tests)
├── test_mcmc_result_extension.py   # Extended MCMCResult (19 tests)
├── test_coordinator.py             # CMCCoordinator (24 tests)
└── test_mcmc_integration.py        # fit_mcmc_jax integration (15 tests)

Total: 232 tests, 100% pass rate
```

### Test Categories

**1. Unit Tests (Isolated Components)**

```python
# Test single function in isolation
def test_weighted_gaussian_product_two_shards():
    # Arrange: Create synthetic data
    shard1 = {'samples': np.random.randn(1000, 3)}
    shard2 = {'samples': np.random.randn(1000, 3)}

    # Act: Run combination
    combined = _weighted_gaussian_product([shard1, shard2])

    # Assert: Verify output
    assert 'samples' in combined
    assert combined['samples'].shape == (1000, 3)
    assert combined['method'] == 'weighted'
```

**2. Integration Tests (Multi-Component)**

```python
# Test full pipeline
def test_cmc_coordinator_end_to_end():
    # Create synthetic dataset
    data, t1, t2, phi = create_synthetic_xpcs_data(size=100_000)

    # Configure CMC
    config = {
        'mcmc': {'num_warmup': 100, 'num_samples': 200},
        'cmc': {
            'sharding': {'num_shards': 4},
            'initialization': {'method': 'identity'},
        },
    }

    # Run CMC
    coordinator = CMCCoordinator(config)
    result = coordinator.run_cmc(data, t1, t2, phi, q=0.01, L=2e6, ...)

    # Verify
    assert result.is_cmc_result()
    assert result.num_shards == 4
    assert len(result.per_shard_diagnostics) == 4
```

**3. Property-Based Tests (Mathematical Invariants)**

```python
import hypothesis
from hypothesis import given, strategies as st

@given(
    num_shards=st.integers(min_value=2, max_value=10),
    num_samples=st.integers(min_value=100, max_value=1000),
)
def test_weighted_combination_preserves_sample_count(num_shards, num_samples):
    """Property: Combined posterior should have same sample count as inputs."""
    # Create shards
    shards = [
        {'samples': np.random.randn(num_samples, 3)}
        for _ in range(num_shards)
    ]

    # Combine
    combined = _weighted_gaussian_product(shards)

    # Verify property
    assert combined['samples'].shape[0] == num_samples
```

**4. Performance Benchmarks**

```python
import pytest

@pytest.mark.benchmark
def test_sharding_performance():
    data = np.random.randn(10_000_000)  # 10M points
    t1 = np.linspace(0, 10, 1000)
    t2 = np.linspace(0, 10, 1000)
    phi = np.linspace(0, 360, 10)

    import time
    start = time.time()
    shards = shard_data_stratified(data, t1, t2, phi, num_shards=16, ...)
    elapsed = time.time() - start

    # Assert performance target
    assert elapsed < 5.0  # Must complete in < 5 seconds
```

### Testing Best Practices

**1. Use Fixtures for Reusable Test Data**

```python
@pytest.fixture
def synthetic_shards():
    """Create synthetic shard results."""
    return [
        {
            'samples': np.random.randn(1000, 3),
            'converged': True,
            'shard_id': i,
        }
        for i in range(5)
    ]

def test_combination_with_fixture(synthetic_shards):
    combined = combine_subposteriors(synthetic_shards)
    assert len(combined['samples']) == 1000
```

**2. Mock External Dependencies**

```python
from unittest.mock import patch, MagicMock

@patch('homodyne.optimization.cmc.backends.pjit.jax.devices')
def test_pjit_backend_device_detection(mock_devices):
    # Mock JAX device detection
    mock_devices.return_value = [MagicMock(), MagicMock()]  # 2 GPUs

    backend = PjitBackend()
    # Backend should detect 2 devices
    ...
```

**3. Test Error Handling**

```python
def test_combination_handles_singular_matrices():
    # Create shards with singular covariance
    shards = [
        {'samples': np.zeros((1000, 3))},  # All zeros → singular
        {'samples': np.random.randn(1000, 3)},
    ]

    # Should fallback to simple averaging
    combined = combine_subposteriors(shards, method='weighted', fallback_enabled=True)
    assert combined['method'] == 'average'  # Fallback occurred
```

**4. Use Parameterized Tests**

```python
@pytest.mark.parametrize("strategy", ['stratified', 'random', 'contiguous'])
def test_all_sharding_strategies(strategy):
    shards = shard_data_stratified(..., strategy=strategy)
    assert len(shards) == 10
    assert validate_shards(shards, original_size)[0]
```

______________________________________________________________________

## Code Style

### General Guidelines

**Follow homodyne conventions:**

- **No emojis** in code (except user-facing messages)
- **Explicit imports** (no `from module import *`)
- **Type hints** for all public functions
- **Docstrings** (Google style) for all modules, classes, and functions
- **Black formatting** (line length: 100)
- **Ruff linting** (enforced via pre-commit)

### Docstring Template

```python
def function_name(
    param1: np.ndarray,
    param2: Dict[str, float],
    optional_param: Optional[int] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Short one-line summary.

    Longer description explaining the function's purpose, algorithm,
    and any important implementation details.

    Parameters
    ----------
    param1 : np.ndarray
        Description of param1
        Shape: (n_points,)
    param2 : dict
        Description of param2
        Keys: parameter names, Values: parameter values
    optional_param : int, optional
        Description of optional parameter
        Default: None (uses automatic value)

    Returns
    -------
    is_valid : bool
        True if validation passed
    diagnostics : dict
        Diagnostic information with keys:
        - 'key1': Description
        - 'key2': Description

    Raises
    ------
    ValueError
        If param1 is empty
    RuntimeError
        If computation fails

    Examples
    --------
    >>> result = function_name(data, params)
    >>> print(result)
    (True, {'key1': 123})

    Notes
    -----
    Additional information about algorithm, complexity, or references.

    References
    ----------
    .. [1] Author et al. (2016). "Paper Title". Journal.
    """
```

### Logging Conventions

```python
import logging
logger = logging.getLogger(__name__)

# INFO: High-level progress
logger.info("Starting CMC pipeline for 50M points")
logger.info(f"✓ Created {num_shards} shards")

# DEBUG: Detailed execution
logger.debug(f"Shard {i} shape: {shard['data'].shape}")
logger.debug(f"SVI loss at step {step}: {loss:.6f}")

# WARNING: Suboptimal choices or fallbacks
logger.warning("Weighted combination failed, falling back to simple averaging")
logger.warning(f"Only {n_converged}/{n_total} shards converged")

# ERROR: Validation failures or exceptions
logger.error(f"Shard {i} failed: {error}")
logger.error("All shards failed to converge")
```

### Error Handling Pattern

```python
# Standard error handling pattern
def process_shard(shard, config):
    try:
        # Main logic
        result = run_mcmc(shard, config)

    except SpecificError as e:
        # Handle expected errors
        logger.warning(f"Expected error: {e}, applying fallback")
        result = apply_fallback(shard, config)

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error: {e}")
        raise  # Re-raise after logging

    else:
        # Success path
        logger.debug("Processing completed successfully")

    finally:
        # Cleanup
        cleanup_resources()

    return result
```

______________________________________________________________________

## Contributing

### Development Workflow

**1. Setup Development Environment**

```bash
# Clone repository
git clone https://github.com/your-org/homodyne.git
cd homodyne

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**2. Create Feature Branch**

```bash
git checkout -b feature/cmc-ray-backend
```

**3. Implement Feature**

- Write code following style guidelines
- Add comprehensive tests
- Update documentation
- Run local tests: `pytest tests/unit/test_*.py`

**4. Run Quality Checks**

```bash
# Format code
make format

# Run linters
make lint

# Run type checker
make type-check

# Run all quality checks
make quality
```

**5. Submit Pull Request**

- Push branch to GitHub
- Open PR with clear description
- Address review feedback
- Ensure CI passes

### Pre-Commit Checks

**`.pre-commit-config.yaml`:**

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.0.0
    hooks:
      - id: black
        args: ['--line-length=100']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.13.0
    hooks:
      - id: ruff
        args: ['--fix']

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.18.0
    hooks:
      - id: mypy
        additional_dependencies: [types-PyYAML]
```

### Code Review Guidelines

**Reviewer Checklist:**

- [ ] Code follows style guidelines
- [ ] All tests pass (100% pass rate required)
- [ ] Documentation updated
- [ ] Type hints present
- [ ] Error handling appropriate
- [ ] Performance acceptable (no regressions)
- [ ] Backward compatibility maintained
- [ ] Security considerations addressed

______________________________________________________________________

## Performance Optimization

### Profiling CMC Execution

**1. Python Profiling**

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run CMC
result = coordinator.run_cmc(data, ...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

**2. JAX Profiling**

```python
import jax
import jax.profiler

# Start profiling
jax.profiler.start_trace("/tmp/tensorboard")

# Run CMC with JAX operations
result = coordinator.run_cmc(data, ...)

# Stop profiling
jax.profiler.stop_trace()

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

### Common Bottlenecks and Solutions

**1. SVI Initialization Too Slow**

**Problem:** SVI takes > 5 minutes for 10M points

**Solution:**

```python
# Reduce SVI steps
config['cmc']['initialization']['svi_steps'] = 5000  # Down from 20000

# Or use NLSQ initialization instead
config['cmc']['initialization']['method'] = 'nlsq'
```

**2. Stratified Sharding Overhead**

**Problem:** Sharding takes > 30 seconds for 50M points

**Solution:**

```python
# Use random sharding (faster)
config['cmc']['sharding']['strategy'] = 'random'

# Or increase minimum shard size (fewer shards)
config['cmc']['sharding']['num_shards'] = 16  # Down from 50
```

**3. Weighted Combination Fails**

**Problem:** Matrix inversion fails for large parameter spaces

**Solution:**

```python
# Enable fallback
config['cmc']['combination']['fallback_enabled'] = True

# Or use simple averaging directly
config['cmc']['combination']['method'] = 'simple_average'
```

### Memory Optimization

**1. Reduce Per-Shard Memory**

```yaml
sharding:
  max_points_per_shard: 500000  # Down from 1M

per_shard_mcmc:
  num_samples: 1000  # Down from 2000
  num_chains: 1      # Down from 4
```

**2. Enable Checkpointing**

```yaml
backend:
  enable_checkpoints: true
  checkpoint_frequency: 5  # Save every 5 shards
```

**3. Use 32-bit Precision**

```yaml
advanced:
  use_64bit: false  # Use float32 instead of float64
  jax_enable_x64: false
```

______________________________________________________________________

## Summary

**Key Takeaways:**

1. **Modular Architecture**: Each component has a single responsibility
1. **Extensible Design**: Easy to add new backends, combination methods, or sharding
   strategies
1. **Comprehensive Testing**: 232 tests with 100% pass rate
1. **Production-Ready**: Fault-tolerant, observable, and well-documented
1. **Performance-Optimized**: Linear speedup with hardware parallelization

**For Contributors:**

- Follow code style guidelines (Black, Ruff, type hints)
- Add comprehensive tests for all new features
- Update documentation (user guide, API reference, this developer guide)
- Run pre-commit hooks before submitting PR
- Ensure 100% test pass rate

**For Maintainers:**

- Review PRs against acceptance criteria
- Ensure backward compatibility
- Monitor performance regressions
- Keep documentation up-to-date
- Respond to issues promptly

______________________________________________________________________

**Further Reading:**

- **User Guide**: `docs/user-guide/cmc_guide.md`
- **API Reference**: `docs/api-reference/cmc_api.md`
- **Troubleshooting**: `docs/troubleshooting/cmc_troubleshooting.md`
- **Migration Guide**: `docs/migration/v3_cmc_migration.md`
