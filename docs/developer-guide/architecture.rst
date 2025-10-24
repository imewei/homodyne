Architecture Guide
==================

This guide provides a comprehensive overview of Homodyne's architecture, design philosophy, and key design patterns.

Overview
--------

Homodyne is a JAX-first, high-performance package for X-ray Photon Correlation Spectroscopy (XPCS) analysis. The architecture is designed around three core principles:

1. **JAX-First Design**: All computational code uses JAX for automatic differentiation, JIT compilation, and device-agnostic execution
2. **Modular Structure**: Clear separation of concerns across 8 major modules
3. **Performance-Critical Paths**: Optimized hot paths for residual calculation, G2 computation, and memory management

Core Equation
~~~~~~~~~~~~~

Homodyne implements the theoretical framework from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing transport properties in flowing soft matter systems:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \text{contrast} \times [c_1(\phi, t_1, t_2)]^2

JAX-First Design Philosophy
----------------------------

The entire computational engine is built on JAX for automatic differentiation, JIT compilation, and hardware acceleration.

Key Components
~~~~~~~~~~~~~~

**1. Primary Optimization: NLSQ Trust-Region Solver**

Located in: :mod:`homodyne.optimization` (:class:`~homodyne.optimization.nlsq_wrapper.NLSQWrapper`)

* Levenberg-Marquardt algorithm via NLSQ package
* JIT-compiled for performance
* Transparent CPU/GPU execution
* Automatic strategy selection (STANDARD → LARGE → CHUNKED → STREAMING)

See also: :doc:`../advanced-topics/nlsq-optimization`

**2. Secondary Optimization: NumPyro/BlackJAX MCMC**

Located in: :mod:`homodyne.optimization` (``mcmc.py``)

* NUTS (No-U-Turn Sampler) for uncertainty quantification
* Built-in progress tracking
* Complements NLSQ for Bayesian parameter estimation

See also: :doc:`../advanced-topics/mcmc-uncertainty`

**3. Device Management**

Located in: :mod:`homodyne.device`

* :func:`~homodyne.device.configure_optimal_device`: Auto-detects optimal device (CPU/GPU)
* System CUDA integration via ``jax[cuda12-local]``
* Graceful GPU→CPU fallback
* HPC CPU optimization (36/128-core nodes)
* Performance benchmarking: :func:`~homodyne.device.benchmark_device_performance`

See also: :doc:`../advanced-topics/gpu-acceleration`

Why JAX?
~~~~~~~~

**Automatic Differentiation**:
   JAX provides forward and reverse-mode autodiff, essential for computing gradients in nonlinear optimization.

**JIT Compilation**:
   Just-in-time compilation via XLA reduces Python overhead and enables aggressive optimizations.

**Device-Agnostic Execution**:
   Same code runs on CPU, GPU, or TPU without modification. Device selection happens at runtime.

**Vectorization**:
   ``jax.vmap`` enables efficient vectorization over phi angles and time points.

Example: JAX-Optimized Residual Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit, grad, vmap

   @jit
   def compute_residuals(params, phi_angles, t1_grid, t2_grid, c2_exp):
       """
       Compute residuals between experimental and theoretical G2.

       This function is JIT-compiled and called repeatedly during optimization.
       It represents a critical performance path.
       """
       # Compute theoretical G2 for all angles (see homodyne.core.jax_backend)
       c2_theory = compute_g2_scaled(params, phi_angles, t1_grid, t2_grid)

       # Flatten and compute residuals
       residuals = (c2_theory - c2_exp).ravel()

       return residuals

See :func:`homodyne.core.jax_backend.compute_g2_scaled` and :func:`homodyne.core.jax_backend.compute_g1` for implementation details.

Module Structure
----------------

Homodyne is organized into 8 major modules, each with a clear responsibility:

.. code-block:: text

   homodyne/
   ├── cli/                    # Command-line interface
   ├── config/                 # Configuration + templates (4 YAML files)
   ├── core/                   # Physics engine (JAX backend)
   ├── data/                   # Data pipeline (HDF5 loading, preprocessing)
   ├── device/                 # Device management (CPU/GPU)
   ├── optimization/           # NLSQ and MCMC methods
   ├── runtime/                # GPU activation, shell completion
   ├── utils/                  # Logging, progress tracking
   └── viz/                    # Visualization and plotting

1. Core Physics (:mod:`homodyne.core`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Physics calculations and model definitions

**Key Files**:

* ``jax_backend.py``: JAX physics functions (:func:`~homodyne.core.jax_backend.compute_g1`, :func:`~homodyne.core.jax_backend.compute_g2`, :func:`~homodyne.core.jax_backend.chi_squared`)
* ``physics.py``: Physical model definitions
* ``models.py``: Object-oriented wrappers
* ``fitting.py``: ScaledFittingEngine with ParameterSpace
* ``theory.py``: TheoryEngine for g2 calculations

**Critical Performance Path**: :func:`~homodyne.core.jax_backend.compute_g2_scaled` is JIT-compiled and GPU-accelerated, vectorized over phi angles.

**See also**: :doc:`../theoretical-framework/core-equations`, :doc:`../api-reference/core`

**Example**:

.. code-block:: python

   from homodyne.core.jax_backend import compute_g2_scaled

   # Compute G2 for all phi angles (vectorized)
   c2_theory = compute_g2_scaled(
       params=params,
       phi_angles=phi_angles,  # [n_angles]
       t1_grid=t1_grid,        # [n_t1, n_t2]
       t2_grid=t2_grid,
   )  # Returns: [n_angles, n_t1, n_t2]

2. Data Pipeline (:mod:`homodyne.data`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: HDF5 data loading, preprocessing, and quality control

**Key Files**:

* ``xpcs_loader.py``: HDF5 data loading (:class:`~homodyne.data.xpcs_loader.XPCSDataLoader` class)
* ``preprocessing.py``: Data preparation and normalization
* ``phi_filtering.py``: Angular filtering algorithms
* ``memory_manager.py``: Memory-efficient data handling
* ``quality_controller.py``: Data quality assessment

**Three-Tier Data Loading Architecture**:

1. :class:`~homodyne.config.manager.ConfigManager` (``config/manager.py``): YAML/JSON config loading

   * Auto-converts ``data_folder_path + data_file_name`` → ``file_path``
   * Handles template and modern formats

2. :class:`~homodyne.data.xpcs_loader.XPCSDataLoader` (``data/xpcs_loader.py``): HDF5 experimental data

   * Supports APS old format and APS-U new format
   * Intelligent caching and validation

3. CLI Commands (:mod:`homodyne.cli`): Workflow orchestration

   * Config + data loading integration
   * CLI overrides (``--data-file``)

**See also**: :doc:`../api-reference/data`, :doc:`../user-guide/configuration`

**Example**:

.. code-block:: python

   from homodyne.data.xpcs_loader import XPCSDataLoader

   # Load experimental data
   loader = XPCSDataLoader(file_path="experiment.hdf")
   data = loader.load_data()

   # Access arrays
   phi_angles = data['phi_angles']
   c2_exp = data['c2_exp']
   t1_grid = data['t1_grid']
   t2_grid = data['t2_grid']

3. Configuration (:mod:`homodyne.config`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Configuration management with type safety and parameter validation

**Key Files**:

* ``manager.py``: YAML configuration system (:class:`~homodyne.config.manager.ConfigManager`)
* ``parameter_manager.py``: Centralized parameter management (:class:`~homodyne.config.parameter_manager.ParameterManager`)
* ``types.py``: TypedDict definitions for type safety
* ``templates/``: 4 YAML configuration templates

**Configuration Templates**:

1. ``homodyne_static_isotropic.yaml``: Static isotropic diffusion (3+2n parameters)
2. ``homodyne_laminar_flow.yaml``: Laminar flow with shear (7+2n parameters)

**See also**: :doc:`../user-guide/configuration`, :doc:`../configuration-templates/index`, :doc:`../api-reference/config`
3. ``homodyne_master_template.yaml``: Comprehensive template with all options
4. ``homodyne_streaming_config.yaml``: Streaming optimization for > 100M points

**Example**:

.. code-block:: python

   from homodyne.config.manager import ConfigManager

   # Load configuration
   config_mgr = ConfigManager("config.yaml")
   config = config_mgr.config

   # Get parameter bounds
   bounds = config_mgr.get_parameter_bounds()

   # Validate parameters
   is_valid = config_mgr.validate_config()

4. Device Management (``homodyne/device/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Hardware detection and optimization (CPU/GPU)

**Key Files**:

* ``__init__.py``: Main API with ``configure_optimal_device()``
* ``gpu.py``: GPU/CUDA configuration
* ``cpu.py``: HPC CPU threading optimization

**Platform Support**:

* **Linux**: CPU + optional GPU (CUDA 12.1-12.9)
* **macOS**: CPU-only
* **Windows**: CPU-only

**Example**:

.. code-block:: python

   from homodyne.device import (
       configure_optimal_device,
       get_device_status,
       benchmark_device_performance
   )

   # Configure device
   device = configure_optimal_device()
   print(f"Using device: {device}")

   # Check status
   status = get_device_status()
   print(f"GPU available: {status['gpu_available']}")

   # Benchmark
   results = benchmark_device_performance()
   print(f"Throughput: {results['throughput']:.2f} GFLOPS")

5. Optimization (``homodyne/optimization/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Nonlinear least squares (NLSQ) and MCMC optimization

**Key Files**:

* ``nlsq_wrapper.py``: Enhanced NLSQWrapper with streaming support
* ``mcmc.py``: NumPyro MCMC with NUTS sampler
* ``strategy.py``: Strategy selection with ``build_streaming_config()``
* ``checkpoint_manager.py``: HDF5 checkpoint management
* ``batch_statistics.py``: Batch-level monitoring
* ``numerical_validation.py``: NaN/Inf validation
* ``recovery_strategies.py``: Error recovery logic
* ``exceptions.py``: Custom exception hierarchy

**Strategy Selection (Automatic)**:

* **< 1M points** → STANDARD strategy (``curve_fit``)
* **1M-10M points** → LARGE strategy (``curve_fit_large``)
* **10M-100M points** → CHUNKED strategy (``curve_fit_large`` with progress)
* **> 100M points** → STREAMING strategy (unlimited data)

**Fallback Chain**: STREAMING → CHUNKED → LARGE → STANDARD

**Example**:

.. code-block:: python

   from homodyne.optimization.nlsq_wrapper import NLSQWrapper

   # Create wrapper with automatic strategy selection
   wrapper = NLSQWrapper(
       enable_large_dataset=True,
       enable_recovery=True,
       enable_numerical_validation=True,
   )

   # Fit (strategy selected automatically based on n_points)
   result = wrapper.fit(
       model_func=model_func,
       xdata=xdata,
       ydata=ydata,
       p0=p0,
       bounds=bounds,
   )

6. Command-Line Interface (``homodyne/cli/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: User-facing CLI with workflow orchestration

**Key Files**:

* ``main.py``: CLI entry point
* ``args_parser.py``: Argument parsing (argparse)
* ``commands.py``: Command implementations (run analysis, save results, plot)

**CLI Design Patterns**:

* Configuration-driven workflows
* CLI overrides for key parameters (``--data-file``, ``--output-dir``)
* Dual logging streams (file + console)
* Progress bars for long-running operations

**Example**:

.. code-block:: bash

   # Basic usage
   homodyne --config config.yaml

   # Override data file
   homodyne --config config.yaml --data-file /path/to/data.hdf

   # Select optimization method
   homodyne --config config.yaml --method nlsq

7. Utilities (``homodyne/utils/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Logging, progress tracking, and helper functions

**Key Files**:

* ``logging.py``: Dual-stream logging (file + console)
* ``progress.py``: Progress bar integration
* ``validation.py``: Input validation helpers

8. Visualization (``homodyne/viz/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Plotting and visualization

**Key Files**:

* ``mcmc_plots.py``: MCMC trace plots, corner plots, autocorrelation

**Example**:

.. code-block:: python

   from homodyne.viz.mcmc_plots import plot_mcmc_traces

   # Plot MCMC traces
   fig = plot_mcmc_traces(samples, parameter_names)
   fig.savefig("mcmc_traces.png")

Analysis Modes
--------------

Homodyne supports two primary analysis modes, corresponding to different physical scenarios:

1. Static Isotropic Mode
~~~~~~~~~~~~~~~~~~~~~~~~~

**Physical Scenario**: Isotropic diffusion with no flow

**Parameters** (5 total):

* **Scaling parameters** (2): ``contrast``, ``offset``
* **Physical parameters** (3): ``D0`` (diffusion coefficient), ``alpha`` (power-law exponent), ``D_offset``

**Configuration**:

.. code-block:: yaml

   analysis:
     mode: "static_isotropic"

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset

2. Laminar Flow Mode
~~~~~~~~~~~~~~~~~~~~~

**Physical Scenario**: Laminar flow with shear-induced transport

**Parameters** (9 total):

* **Scaling parameters** (2): ``contrast``, ``offset``
* **Physical parameters** (7):

  * ``D0``, ``alpha``, ``D_offset`` (diffusion)
  * ``gamma_dot_0`` (shear rate), ``beta`` (flow exponent), ``gamma_dot_offset``, ``phi0`` (flow angle)

**Configuration**:

.. code-block:: yaml

   analysis:
     mode: "laminar_flow"

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset
       - gamma_dot_0
       - beta
       - gamma_dot_offset
       - phi_0

**Note**: Config files typically specify only physical parameters. Optimization code automatically adds scaling parameters (``contrast=0.5``, ``offset=1.0``).

**Parameter Name Mapping**: ``gamma_dot_0`` → ``gamma_dot_t0``, ``phi_0`` → ``phi0``

Parameter Management System
---------------------------

Architecture
~~~~~~~~~~~~

**Centralized via ParameterManager class** (``homodyne/config/parameter_manager.py``)

**Key Features**:

* Centralized parameter bounds, active parameters, and validation
* Automatic parameter name mapping (config → code names)
* Config bounds override support
* Physics-based constraint validation (3 severity levels: ERROR, WARNING, INFO)
* Performance caching (~10-100x speedup for repeated queries)

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     bounds:
       - name: D0
         min: 100.0
         max: 1e5
       - name: gamma_dot_0  # Auto-mapped to gamma_dot_t0
         min: 1e-6
         max: 0.5

   initial_parameters:
     parameter_names:  # Parameters to optimize
       - D0
       - alpha
       - D_offset
     active_parameters:  # Optional: subset to optimize
       - D0
       - alpha
     fixed_parameters:  # Optional: hold fixed
       D_offset: 10.0

Usage
~~~~~

.. code-block:: python

   # Direct ParameterManager use
   from homodyne.config.parameter_manager import ParameterManager

   pm = ParameterManager(config_dict, "laminar_flow")
   bounds = pm.get_parameter_bounds(["D0", "alpha"])
   active_params = pm.get_active_parameters()

   # Through ConfigManager
   from homodyne.config.manager import ConfigManager

   config_mgr = ConfigManager("config.yaml")
   bounds = config_mgr.get_parameter_bounds()

Physics Validation
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validate physics constraints
   result = pm.validate_physical_constraints(params, severity_level="warning")
   # Severity levels: "error" (critical), "warning" (unusual), "info" (observations)

**Type Safety**: TypedDict definitions in ``homodyne/config/types.py`` (``BoundDict``, ``InitialParametersConfig``, ``ParameterSpaceConfig``, ``HomodyneConfig``)

Critical Performance Paths
---------------------------

These are the hot paths in the codebase that dominate execution time:

1. Residual Calculation
~~~~~~~~~~~~~~~~~~~~~~~~

**Location**: ``core/jax_backend.py:compute_residuals()``

**Why Critical**: Called repeatedly during optimization (100-1000+ iterations)

**Optimizations**:

* JIT-compiled via ``@jit`` decorator
* Vectorized over phi angles via ``vmap``
* Minimized Python overhead

**Code Pattern**:

.. code-block:: python

   @jit
   def compute_residuals(params, phi_angles, t1_grid, t2_grid, c2_exp):
       c2_theory = compute_g2_scaled(params, phi_angles, t1_grid, t2_grid)
       residuals = (c2_theory - c2_exp).ravel()
       return residuals

2. G2 Computation
~~~~~~~~~~~~~~~~~~

**Location**: ``core/jax_backend.py:compute_g2_scaled()``

**Why Critical**: Core physics calculation, dominates memory and compute

**Optimizations**:

* Vectorized over phi angles (``vmap``)
* GPU-accelerated via JAX
* Efficient array broadcasting

**Performance Characteristics**:

* **Memory**: O(n_angles × n_t1 × n_t2 × 8 bytes)
* **Compute**: O(n_angles × n_t1 × n_t2 × n_ops)
* **Speedup**: 10-100x on GPU vs CPU

3. Memory Management
~~~~~~~~~~~~~~~~~~~~~

**Location**: ``data/memory_manager.py``

**Why Critical**: Determines maximum dataset size and memory footprint

**Strategies**:

* **STANDARD**: Full data in memory
* **LARGE**: Memory-optimized NLSQ
* **CHUNKED**: Progress tracking
* **STREAMING**: Constant memory (batch-based)

**Memory Formula**:

.. code-block:: python

   memory_gb = n_points × (1 + n_parameters) × 16 / (1024**3)
   # Factor of 16 = 8 bytes (float64) × 2 (overhead)

Design Patterns
---------------

1. Strategy Pattern: Optimization Strategy Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Select optimal NLSQ strategy based on dataset size and memory

**Location**: ``homodyne/optimization/strategy.py``

**Implementation**:

.. code-block:: python

   class DatasetSizeStrategy:
       def select_strategy(self, n_points):
           if n_points < 1_000_000:
               return 'standard'
           elif n_points < 10_000_000:
               return 'large'
           elif n_points < 100_000_000:
               return 'chunked'
           else:
               return 'streaming'

**Benefits**:

* Automatic selection based on data characteristics
* Fallback chain for robustness
* Optional manual override for advanced users

2. Template Method: Error Recovery
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Standardized error recovery across different failure modes

**Location**: ``homodyne/optimization/recovery_strategies.py``

**Implementation**:

.. code-block:: python

   class RecoveryStrategy(ABC):
       @abstractmethod
       def recover(self, error, params, bounds):
           """Attempt recovery from specific error type."""
           pass

   class OOMRecoveryStrategy(RecoveryStrategy):
       def recover(self, error, params, bounds):
           # Reduce batch size or switch to STANDARD strategy
           pass

   class ConvergenceRecoveryStrategy(RecoveryStrategy):
       def recover(self, error, params, bounds):
           # Perturb parameters and retry
           pass

**5 Error Categories**:

1. **OOM (Out of Memory)**: Reduce batch size, switch strategy
2. **Convergence**: Perturb parameters, retry with different initialization
3. **Bounds**: Project back into bounds, adjust step size
4. **Numerical**: Skip NaN/Inf, increase regularization
5. **Unknown**: Fallback to simpler strategy

3. Checkpoint Pattern: Fault Tolerance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Save/resume capability for long-running optimizations

**Location**: ``homodyne/optimization/checkpoint_manager.py``

**Implementation**:

.. code-block:: python

   class CheckpointManager:
       def save_checkpoint(self, batch_num, best_params, state):
           """Save checkpoint to HDF5 with compression."""
           checkpoint = {
               'batch_num': batch_num,
               'best_params': best_params,
               'state': state,
               'timestamp': time.time(),
           }
           # Save to HDF5 with compression
           self._save_hdf5(checkpoint)

       def load_checkpoint(self):
           """Load most recent valid checkpoint."""
           # Detect corruption, return None if invalid
           return self._load_hdf5()

**Features**:

* HDF5-based with compression (< 2 second saves)
* Automatic corruption detection
* Keep last N checkpoints (default: 3)
* Resume from any valid checkpoint

4. Caching Pattern: Parameter Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Avoid repeated parameter validation and bounds computation

**Location**: ``homodyne/config/parameter_manager.py``

**Implementation**:

.. code-block:: python

   class ParameterManager:
       def __init__(self):
           self._bounds_cache = {}
           self._validation_cache = {}

       def get_parameter_bounds(self, param_names):
           cache_key = tuple(param_names)
           if cache_key not in self._bounds_cache:
               self._bounds_cache[cache_key] = self._compute_bounds(param_names)
           return self._bounds_cache[cache_key]

**Benefits**:

* 10-100x speedup for repeated queries
* Minimal memory overhead
* Thread-safe (immutable cache keys)

Architecture Diagrams
---------------------

High-Level Architecture
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────┐
   │                        CLI Entry Point                           │
   │                    (homodyne/cli/main.py)                        │
   └────────────────────────────┬────────────────────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    Configuration Manager                         │
   │              (homodyne/config/manager.py)                        │
   │  - Load YAML config                                              │
   │  - Parameter validation                                          │
   │  - Bounds management                                             │
   └────────────┬───────────────────────────────────┬─────────────────┘
                │                                   │
                ▼                                   ▼
   ┌────────────────────────┐         ┌────────────────────────────┐
   │   Data Pipeline        │         │   Device Management        │
   │  (homodyne/data/)      │         │  (homodyne/device/)        │
   │  - HDF5 loading        │         │  - GPU detection           │
   │  - Preprocessing       │         │  - JAX configuration       │
   │  - Phi filtering       │         │  - CPU optimization        │
   └──────────┬─────────────┘         └─────────────┬──────────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                   Optimization Engine                            │
   │             (homodyne/optimization/)                             │
   │  - Strategy selection (STANDARD/LARGE/CHUNKED/STREAMING)        │
   │  - NLSQ wrapper                                                  │
   │  - Error recovery                                                │
   │  - Checkpoint management                                         │
   └────────────┬─────────────────────────────────────────────────────┘
                │
                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                   Core Physics Engine                            │
   │                 (homodyne/core/)                                 │
   │  - JAX backend (JIT-compiled)                                    │
   │  - G1/G2 computation                                             │
   │  - Residual calculation                                          │
   │  - GPU-accelerated                                               │
   └─────────────────────────────────────────────────────────────────┘

Optimization Workflow
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   ┌─────────────────────┐
   │   Load Config       │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │   Load Data         │
   │   (HDF5)            │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐
   │  Apply Phi Filter   │
   │  (if enabled)       │
   └──────────┬──────────┘
              │
              ▼
   ┌─────────────────────┐      ┌────────────────────────┐
   │  Strategy Selection │─────▶│  Memory Estimation     │
   │  (Auto)             │      │  Device Detection      │
   └──────────┬──────────┘      └────────────────────────┘
              │
              ├─────────────────┬─────────────────┬─────────────────┐
              ▼                 ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  STANDARD    │  │  LARGE       │  │  CHUNKED     │  │  STREAMING   │
   │  (< 1M pts)  │  │  (1M-10M)    │  │  (10M-100M)  │  │  (> 100M)    │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          │                 │                 │                 │
          └─────────────────┴─────────────────┴─────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │               NLSQ Optimization Loop                             │
   │  ┌────────────────────────────────────────────────────┐         │
   │  │  1. Compute residuals (JAX JIT-compiled)           │         │
   │  │  2. Compute Jacobian (automatic differentiation)   │         │
   │  │  3. Trust-region step                              │         │
   │  │  4. Update parameters                              │         │
   │  │  5. Check convergence                              │         │
   │  └────────────────────────────────────────────────────┘         │
   │                                                                  │
   │  If error → Recovery Strategy → Retry (max 3 attempts)          │
   │  If batch complete (STREAMING) → Checkpoint → Next batch        │
   └─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                    Save Results                                  │
   │  - parameters.json (values + uncertainties)                      │
   │  - fitted_data.npz (10 arrays)                                   │
   │  - analysis_results_nlsq.json (fit quality)                      │
   │  - convergence_metrics.json (diagnostics)                        │
   └─────────────────────────────────────────────────────────────────┘

Data Flow
~~~~~~~~~

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │                     HDF5 Data File                            │
   │  - /exchange/phi_angles                                       │
   │  - /exchange/c2_exp                                           │
   │  - /exchange/t1_grid, t2_grid                                 │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              XPCSDataLoader                                   │
   │  - Detect format (APS old / APS-U new)                        │
   │  - Load arrays                                                │
   │  - Cache for repeated access                                  │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              Phi Filtering (Optional)                         │
   │  - Normalize angles to [-180°, 180°]                          │
   │  - Apply target_ranges                                        │
   │  - Handle wrap-around at ±180°                                │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              Preprocessing                                    │
   │  - Normalize c2_exp                                           │
   │  - Flatten arrays for optimization                            │
   │  - Create xdata indices (for curve_fit_large chunking)        │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              Model Function                                   │
   │  def model_func(xdata, *params):                              │
   │      full_output = compute_g2_scaled(params, ...)  # JAX      │
   │      indices = xdata.astype(int)                              │
   │      return full_output[indices]  # Chunking-compatible       │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │         NLSQ Optimization (curve_fit / curve_fit_large)       │
   │  - Iteratively refine parameters                              │
   │  - Return popt, pcov, info                                    │
   └──────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
   ┌──────────────────────────────────────────────────────────────┐
   │              Result Saving                                    │
   │  - Extract parameter values + uncertainties                   │
   │  - Compute theoretical fits per angle                         │
   │  - Save JSON (3 files) + NPZ (10 arrays)                      │
   └──────────────────────────────────────────────────────────────┘

Best Practices
--------------

When Working with JAX
~~~~~~~~~~~~~~~~~~~~~~

**DO**:

* Use ``jax.numpy`` instead of ``numpy`` in computational code
* Apply ``@jit`` to functions called repeatedly
* Use ``vmap`` for vectorization instead of loops
* Keep arrays on GPU throughout computation
* Profile with ``JAX_LOG_COMPILES=1`` to detect unnecessary recompilation

**DON'T**:

* Use Python control flow (if/for) inside JIT-compiled functions
* Mutate arrays in-place (JAX arrays are immutable)
* Mix NumPy and JAX arrays without conversion
* Forget ``jax.device_put()`` to move data to GPU

When Adding New Optimization Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pattern to Follow** (see ``homodyne/optimization/nlsq_wrapper.py``):

1. Implement interface matching ``fit()`` signature
2. Use residual functions from ``core/jax_backend.py``
3. Add error recovery for common failure modes
4. Include progress tracking for long optimizations
5. Write unit tests in ``tests/unit/test_optimization_*.py``
6. Add integration tests in ``tests/integration/test_workflows.py``
7. Document in Sphinx API reference

When Modifying Physics Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Steps**:

1. Update JAX functions in ``core/jax_backend.py``
2. Ensure JIT compatibility (no Python control flow)
3. Update wrappers in ``core/models.py``
4. Test gradient/Hessian computations
5. Run scientific validation tests (``tests/self_consistency/``)
6. Update parameter bounds in ``config/parameter_manager.py``

When Debugging Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Diagnostic Steps**:

1. Check JAX compilation: ``JAX_LOG_COMPILES=1 python script.py``
2. Profile: ``make profile-nlsq`` or ``make profile-mcmc``
3. Monitor GPU: ``nvidia-smi -l 1``
4. Check memory: ``JAX_TRACEBACK_FILTERING=off`` for full traces
5. Review ``DEBUG_REPORT.md`` for diagnostics template

HPC/GPU Considerations
----------------------

GPU Support Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

* **OS**: Linux x86_64 or aarch64 **only**
* **CUDA**: 12.1-12.9 (pre-installed on system)
* **NVIDIA driver**: >= 525
* **Not supported**: Windows, macOS (CPU-only on these platforms)

HPC Installation
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Load system CUDA modules
   module load cuda/12.2 cudnn/9.8

   # Install JAX with CUDA support
   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

   # Install homodyne
   pip install homodyne[dev]

CPU-Only Fallback
~~~~~~~~~~~~~~~~~

The package automatically falls back to CPU computation if:

* Running on Windows or macOS
* CUDA not available on Linux
* GPU initialization fails

Device selection is handled by ``homodyne/device/configure_optimal_device()``.

References
----------

* **NLSQ Package**: https://github.com/imewei/NLSQ
* **JAX Documentation**: https://jax.readthedocs.io/
* **NumPyro Documentation**: https://num.pyro.ai/
* **He et al. PNAS 2024**: https://doi.org/10.1073/pnas.2401162121

Next Steps
----------

* **Testing Guide**: Learn about test categories, running tests, and writing new tests
* **Contributing Guide**: Development workflow, Git conventions, pull request guidelines
* **Performance Guide**: Profiling, GPU optimization, memory management
