# Homodyne Architecture Overview

Comprehensive system-level architecture documentation for the homodyne XPCS analysis package.

**Version:** 2.22.2 **Last Updated:** February 2026 **Codebase:** ~87,700 lines Python + ~680 lines shell

## Table of Contents

1. [System Overview](#system-overview)
1. [Package Structure](#1-package-structure)
1. [Initialization & Environment](#2-initialization--environment)
1. [CLI Layer](#3-cli-layer)
1. [Configuration System](#4-configuration-system)
1. [Data Loading Pipeline](#5-data-loading-pipeline)
1. [Physical Model Layer](#6-physical-model-layer)
1. [NLSQ Optimization](#7-nlsq-optimization)
1. [CMC Bayesian Inference](#8-cmc-bayesian-inference)
1. [IO & Result Serialization](#9-io--result-serialization)
1. [Visualization](#10-visualization)
1. [Device & HPC Configuration](#11-device--hpc-configuration)
1. [Utilities](#12-utilities)
1. [Runtime & Deployment](#13-runtime--deployment)
1. [Fault Tolerance & Recovery](#14-fault-tolerance--recovery)
1. [Cross-Cutting Concerns](#15-cross-cutting-concerns)
1. [Complete End-to-End Flow](#complete-end-to-end-flow)
1. [Module Size & Dependency Matrix](#module-size--dependency-matrix)
1. [Architecture Decision Records](#architecture-decision-records)
1. [Companion Documents](#companion-documents)

______________________________________________________________________

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          HOMODYNE v2.22.2                                       │
│            CPU-Optimized JAX Package for XPCS Analysis                         │
│                                                                                 │
│   Core Equation:  c2(phi, t1, t2) = 1 + contrast * [g1(phi, t1, t2)]^2        │
│                                                                                 │
│   Modes:  static (3 params)  |  laminar_flow (7 params)                        │
│           + per-angle scaling: auto(+2), constant(+0), individual(+2*n_phi),   │
│                                fourier(+2*(1+2K) coefficients)                   │
│                                                                                 │
│   Stack:  Python 3.12+ | JAX >= 0.8.2 (CPU-only) | NumPy >= 2.3 | NLSQ >= 0.6.4│
│           NumPyro (MCMC) | h5py (HDF5) | ArviZ (diagnostics)                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             USER ENTRY POINTS                                   │
│                                                                                 │
│   CLI                    Python API                  Config Gen                 │
│   $ homodyne             from homodyne import        $ homodyne-config          │
│     --config x.yaml        fit_nlsq_jax,             --mode laminar_flow       │
│     --method nlsq          fit_mcmc_jax,             --output config.yaml      │
│     --output ./out         ConfigManager                                       │
│                                                                                 │
│   Shell Aliases           XLA Config                 System Check              │
│   $ hm, hm-nlsq,         $ homodyne-config-xla      $ homodyne-post-install   │
│     hm-cmc, hconfig        --mode parallel                                     │
└────────┬──────────────────────┬──────────────────────────┬──────────────────────┘
         │                      │                          │
═════════╪══════════════════════╪══════════════════════════╪══════════════════════
         ▼                      ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATION LAYER                                     │
│                                                                                 │
│   cli/main.py ──► cli/commands.py ──► 4-Phase Pipeline:                        │
│                                                                                 │
│     Phase 1: Config      Phase 2: Data       Phase 3: Optimize    Phase 4: Save│
│     ConfigManager ──►    XPCSDataLoader ──►   NLSQ / CMC   ──►    JSON + NPZ  │
│     ParameterManager     HDF5 → validate     (or both)            ArviZ NetCDF │
│     ParameterSpace       filter → preprocess                      Plots        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
         │                      │                          │
═════════╪══════════════════════╪══════════════════════════╪══════════════════════
         ▼                      ▼                          ▼
┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────────────────┐
│  CONFIG SYSTEM   │  │   DATA LOADING   │  │      PHYSICAL MODEL LAYER          │
│  config/         │  │   data/          │  │      core/                         │
│                  │  │                  │  │                                    │
│  ConfigManager   │  │  XPCSDataLoader  │  │  HomodyneModel  TheoryEngine      │
│  ParameterMgr    │  │  HDF5 old/new    │  │  CombinedModel  DiffusionModel    │
│  ParameterSpace  │  │  Filtering       │  │  ShearModel     PhysicsConstants  │
│  ParameterReg    │  │  Preprocessing   │  │  JIT kernels    Shadow copies     │
│  TypedDicts      │  │  QC + Caching    │  │  Per-angle scaling  Grad-safe     │
└──────────────────┘  └──────────────────┘  └──────────┬─────────────────────────┘
                                                        │
════════════════════════════════════════════════════════╪═════════════════════════
                                                        ▼
┌──────────────────────────────────┐  ┌──────────────────────────────────────────┐
│  NLSQ OPTIMIZATION               │  │  CMC BAYESIAN INFERENCE                  │
│  optimization/nlsq/              │  │  optimization/cmc/                       │
│                                  │  │                                          │
│  Trust-Region L-M (NLSQ 0.6.4)  │  │  Consensus Monte Carlo (NumPyro NUTS)   │
│  NLSQAdapter / NLSQWrapper       │  │  Sharding + Multiprocessing Backend     │
│  Anti-Degeneracy Controller      │  │  Adaptive Sampling + Reparameterization │
│  CMA-ES Global Optimization      │  │  Quality Filtering + Diagnostics        │
│  Gradient Monitoring              │  │  Shared Memory + JIT Cache              │
│  Memory-Aware Routing             │  │  NLSQ Warm-Start Integration           │
└──────────────────────────────────┘  └──────────────────────────────────────────┘
         │                                       │
═════════╪═══════════════════════════════════════╪════════════════════════════════
         ▼                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT & VISUALIZATION                                   │
│                                                                                  │
│  io/                          viz/                                               │
│  nlsq_writers.py (JSON+NPZ)  nlsq_plots.py (heatmaps, residuals, contours)     │
│  mcmc_writers.py (params,     mcmc_plots.py (traces, posteriors, forest, ESS)   │
│    analysis, diagnostics)     datashader_backend.py (fast rendering >100K pts)  │
│  json_utils.py (NaN-safe)    experimental_plots.py (raw data inspection)        │
│                               diagnostics.py (diagonal overlay)                  │
└──────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│                       CROSS-CUTTING INFRASTRUCTURE                               │
│                                                                                  │
│  device/             utils/               runtime/          optimization/ (top)  │
│  CPU topology        Structured logging   Shell completion  Checkpoint manager   │
│  NUMA detection      Phase timing         Post-install      Gradient diagnostics │
│  HPC optimization    Memory tracking      System validator  Numerical validation │
│  XLA configuration   Path validation      Venv activation   Recovery strategies  │
│                                                             Exception hierarchy  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

| Principle | Implementation |
|-----------|----------------|
| **JAX-first** | All numerical computation in JAX; NumPy only at I/O boundaries |
| **CPU-only** | No GPU/TPU paths; 4 virtual XLA devices for parallel MCMC |
| **Float64 always** | `JAX_ENABLE_X64=1` set before first JAX import |
| **Full precision** | Never subsample or downsample data |
| **Gradient-safe** | `jnp.where(x > eps, x, eps)` not `jnp.maximum` for floors |
| **Lazy loading** | `__getattr__` hook defers heavy imports until needed |
| **Narrow exceptions** | Specific types at boundaries; broad only at top-level dispatchers |
| **Reproducible** | Explicit seeds, version-locked deps via `uv.lock` |

______________________________________________________________________

## 1. Package Structure

```
homodyne/                              # Root package (358 lines)
├── __init__.py                        # Lazy imports, XLA/Float64 env setup
├── _version.py                        # Auto-generated: "2.22.2"
│
├── core/                   9,399 L    # Physics models & JAX primitives
│   ├── jax_backend.py      1,556 L    #   JIT dispatcher (meshgrid/element-wise)
│   ├── numpy_gradients.py  1,049 L    #   Finite-difference fallback
│   ├── fitting.py            881 L    #   ScaledFittingEngine (Cholesky/SVD)
│   ├── physics_cmc.py        807 L    #   CMC-optimized kernels + ShardGrid
│   ├── models.py             614 L    #   DiffusionModel → ShearModel → CombinedModel
│   ├── theory.py             567 L    #   TheoryEngine high-level API
│   ├── physics.py            553 L    #   PhysicsConstants, bounds, validation
│   ├── diagonal_correction.py 522 L   #   C2 diagonal artifact removal
│   ├── model_mixins.py       520 L    #   Reusable model behaviors
│   ├── homodyne_model.py     505 L    #   HomodyneModel unified interface
│   ├── physics_nlsq.py       480 L    #   NLSQ-optimized g1/g2 kernels
│   ├── physics_factors.py    369 L    #   Pre-computed factor matrices
│   ├── physics_utils.py      369 L    #   Shared: safe_sinc, safe_exp, D(q)
│   ├── scaling_utils.py      335 L    #   Per-angle scaling transforms
│   ├── backend_api.py        192 L    #   Backend selection API
│   └── __init__.py            80 L
│
├── data/                  12,302 L    # HDF5 loading & preprocessing
│   ├── xpcs_loader.py     2,107 L    #   XPCSDataLoader (main entry)
│   ├── quality_controller.py 1,646 L  #   Data quality assessment
│   ├── performance_engine.py 1,502 L  #   Multi-level cache (memory + disk)
│   ├── preprocessing.py    1,153 L    #   Pipeline: standardize → diagonal → trim
│   ├── validation.py       1,115 L    #   Shape/dtype/NaN validators
│   ├── memory_manager.py   1,030 L    #   Memory tracking & pressure detection
│   ├── optimization.py       971 L    #   Dataset optimization strategies
│   ├── config.py             752 L    #   DataConfig from YAML
│   ├── filtering_utils.py    613 L    #   Angle/Q-range filtering
│   ├── angle_filtering.py    413 L    #   Angle-specific filtering
│   ├── phi_filtering.py      385 L    #   Phi-angle filtering
│   ├── validators.py         296 L    #   Input range checks
│   ├── __init__.py           275 L
│   └── types.py               44 L    #   Type definitions
│
├── config/                 4,832 L    # YAML/JSON config management
│   ├── manager.py          1,296 L    #   ConfigManager (load, merge, validate)
│   ├── parameter_space.py    895 L    #   Bounds + priors for MCMC
│   ├── parameter_manager.py  809 L    #   Active params, initial guesses
│   ├── parameter_registry.py 632 L    #   Singleton metadata store
│   ├── types.py              522 L    #   TypedDict definitions
│   ├── parameter_names.py    315 L    #   String constants
│   ├── physics_validators.py  286 L    #   Physics parameter validation
│   └── __init__.py            77 L
│
├── optimization/          45,726 L    # NLSQ + CMC fitting engines
│   ├── __init__.py           198 L    #   fit_nlsq_jax, fit_mcmc_jax exports
│   ├── checkpoint_manager.py 507 L    #   HDF5 fault-tolerant checkpoints
│   ├── gradient_diagnostics.py 437 L  #   Gradient imbalance detection
│   ├── exceptions.py         352 L    #   OptimizationError hierarchy
│   ├── numerical_validation.py 254 L  #   NaN/Inf/condition checks
│   ├── recovery_strategies.py  209 L  #   Automatic error recovery
│   ├── batch_statistics.py    186 L   #   Circular buffer convergence stats
│   │
│   ├── nlsq/              29,356 L    #   Trust-region L-M optimizer
│   │   ├── wrapper.py      8,248 L    #     NLSQWrapper (full feature set)
│   │   ├── core.py         2,327 L    #     fit_nlsq_jax dispatcher
│   │   ├── multistart.py   1,485 L    #     Multi-start with basin hopping
│   │   ├── config.py       1,260 L    #     NLSQConfig from YAML
│   │   ├── adapter.py      1,143 L    #     NLSQAdapter (recommended entry)
│   │   ├── cmaes_wrapper.py 1,097 L   #     CMA-ES global optimization
│   │   ├── anti_degeneracy_controller.py 1,021 L
│   │   ├── hierarchical.py   736 L    #     Hierarchical fitting
│   │   ├── fourier_reparam.py 659 L   #     Fourier angular parameterization
│   │   ├── gradient_monitor.py 548 L  #     Gradient collapse detection
│   │   ├── parameter_utils.py  540 L  #     Parameter transforms
│   │   ├── fit_computation.py  522 L  #     Core fit logic
│   │   ├── progress.py        531 L   #     Progress reporting
│   │   ├── adaptive_regularization.py 507 L
│   │   ├── shear_weighting.py 453 L   #     Angular weight computation
│   │   ├── transforms.py     447 L    #     Param expansion/compression
│   │   ├── memory.py         420 L    #     Memory estimation & routing
│   │   ├── adapter_base.py   366 L    #     Adapter ABC
│   │   ├── data_prep.py      319 L    #     Data preprocessing
│   │   ├── result_builder.py 479 L    #     NLSQResult construction
│   │   ├── results.py        245 L    #     Result dataclass
│   │   ├── parameter_index_mapper.py 255 L
│   │   ├── jacobian.py       250 L    #     Jacobian inspection
│   │   └── strategies/      4,054 L   #     Execution strategies
│   │       ├── residual.py     786 L  #       Standard residual function
│   │       ├── residual_jit.py 545 L  #       JIT-compiled variant
│   │       ├── chunking.py   1,205 L  #       Out-of-core streaming
│   │       ├── sequential.py 1,020 L  #       Per-angle sequential fitting
│   │       └── executors.py    428 L  #       Strategy executor dispatch
│   │   ├── config_utils.py    132 L    #     Config utility helpers
│   │   ├── validation/        842 L   #     Parameter validation subsystem
│   │   └── __init__.py        470 L
│   │
│   └── cmc/               14,227 L    #   Consensus Monte Carlo
│       ├── core.py         1,566 L    #     fit_mcmc_jax dispatcher
│       ├── sampler.py      1,326 L    #     NUTS sampling + SamplingPlan
│       ├── diagnostics.py  1,269 L    #     R-hat, ESS, divergences
│       ├── model.py        1,168 L    #     NumPyro model specification
│       ├── priors.py       1,100 L    #     Prior construction from NLSQ
│       ├── config.py         887 L    #     CMCConfig from YAML
│       ├── data_prep.py      848 L    #     Shard construction
│       ├── results.py        789 L    #     CMCResult + convergence
│       ├── plotting.py       502 L    #     ArviZ plot generation
│       ├── io.py             430 L    #     Shard I/O + shared memory
│       ├── scaling.py        344 L    #     Per-angle scaling for CMC
│       ├── reparameterization.py 336 L #    Log-space transforms
│       ├── backends/       3,625 L    #     Execution backends
│       │   ├── multiprocessing.py 1,953 L  # Primary: process pool
│       │   ├── base.py       887 L    #       ABC + scheduling
│       │   ├── pbs.py        494 L    #       PBS/Torque cluster
│       │   └── pjit.py       269 L    #       pjit single-process
│       └── __init__.py        37 L
│
├── io/                       953 L    # Result serialization
│   ├── mcmc_writers.py       639 L    #   MCMC params + analysis + diagnostics
│   ├── nlsq_writers.py       171 L    #   NLSQ JSON + NPZ
│   └── json_utils.py         114 L    #   NaN/Inf-safe JSON encoder
│
├── viz/                    3,761 L    # Visualization
│   ├── mcmc_plots.py       1,812 L    #   Traces, posteriors, forest, ESS
│   ├── nlsq_plots.py         961 L    #   Heatmaps, residuals, contours
│   ├── datashader_backend.py  515 L   #   Fast rendering for >100K points
│   ├── experimental_plots.py  298 L   #   Raw data inspection
│   ├── diagnostics.py          59 L   #   Diagonal overlay
│   └── validation.py           49 L   #   Plot input validation
│
├── cli/                    4,635 L    # Command-line interface
│   ├── commands.py         3,361 L    #   4-phase pipeline orchestration
│   ├── config_generator.py   537 L    #   Template generation + builder
│   ├── args_parser.py        449 L    #   Argument definition + parsing
│   ├── xla_config.py         156 L    #   XLA device parallelism
│   └── main.py               111 L    #   Entry point + env setup
│
├── device/                 1,055 L    # CPU/NUMA configuration
│   ├── cpu.py                514 L    #   HPC CPU optimization
│   ├── __init__.py           278 L    #   Device API
│   └── config.py             263 L    #   Hardware detection
│
├── utils/                  1,478 L    # Logging & path security
│   ├── logging.py          1,145 L    #   Structured logging framework
│   └── path_validation.py    299 L    #   Path traversal prevention
│
├── runtime/                  ~700 L   # Deployment & shell integration
│   ├── utils/
│   │   └── system_validator.py 1,472 L #  Comprehensive health checks
│   └── shell/
│       └── completion.sh      680 L   #   Tab completion + aliases
│
├── post_install.py           981 L    # Shell completion installer
└── uninstall_scripts.py      750 L    # Cleanup utility
```

### Size Distribution

| Module | Lines | % | Primary Responsibility |
|--------|------:|--:|------------------------|
| optimization/ | 45,726 | 52.1% | NLSQ (29,356) + CMC (14,227) + shared (2,143) |
| data/ | 12,302 | 14.0% | HDF5 loading, preprocessing, QC, caching |
| core/ | 9,399 | 10.7% | Physics models, JIT kernels, fitting |
| config/ | 4,832 | 5.5% | YAML/JSON config, parameter management |
| cli/ | 4,635 | 5.3% | CLI entry points, pipeline orchestration |
| viz/ | 3,761 | 4.3% | NLSQ/MCMC plots, datashader backend |
| runtime/ | 3,219 | 3.7% | System validator, shell completion, installer |
| utils/ | 1,478 | 1.7% | Structured logging, path security |
| device/ | 1,055 | 1.2% | CPU topology, NUMA, HPC optimization |
| io/ | 953 | 1.1% | JSON/NPZ/NetCDF result writers |
| root | 358 | 0.4% | Package init, version |
| **Total** | **~87,700** | | |

______________________________________________________________________

## 2. Initialization & Environment

### Package Import Sequence

```
import homodyne
    │
    ├── os.environ.setdefault("JAX_ENABLE_X64", "1")          # Float64 before JAX import
    ├── os.environ["XLA_FLAGS"] = flags              # 4 virtual devices + disable constant_folding
    ├── os.environ["NLSQ_SKIP_GPU_CHECK"] = "1"     # CPU-only package
    ├── warnings.filterwarnings(NumPyro pxla)        # Suppress JAX 0.8.2 deprecation
    ├── logging.getLogger("jax._src.xla_bridge").setLevel(ERROR)
    ├── logging.getLogger("jax._src.compiler").setLevel(ERROR)
    │
    ├── __version__ = "2.22.2"                       # From _version.py (setuptools_scm)
    │
    └── __getattr__(name) ──► Lazy import on first access
         homodyne.fit_nlsq_jax  → from homodyne.optimization import fit_nlsq_jax
         homodyne.ConfigManager → from homodyne.config import ConfigManager
         homodyne.HAS_DATA      → bool (True if homodyne.data importable)
```

### Environment Variables

| Variable | Value | Set By | Purpose |
|----------|-------|--------|---------|
| `JAX_ENABLE_X64` | `"1"` | `__init__.py`, `cli/main.py`, workers | Float64 precision |
| `XLA_FLAGS` | `--xla_force_host_platform_device_count=4 --xla_disable_hlo_passes=constant_folding` | `__init__.py` | Parallel MCMC + large-constant compilation |
| `NLSQ_SKIP_GPU_CHECK` | `"1"` | `__init__.py` | CPU-only |
| `OMP_NUM_THREADS` | `1-2` | CMC multiprocessing backend | Prevent thread oversubscription |

### Lazy Import System

The `__init__.py` uses `__getattr__` to defer all submodule imports until first attribute access.
This avoids the 3-6 second JAX/NumPyro/h5py startup penalty for CLI invocations that only need
argument parsing.

**Lazy symbols:** `XPCSDataLoader`, `fit_nlsq_jax`, `fit_mcmc_jax`, `ConfigManager`,
`configure_optimal_device`, `ParameterSpace`, `ScaledFittingEngine`, `TheoryEngine`,
`compute_g2_scaled`, `cli_main`, `load_xpcs_data`, `get_optimization_info`, `get_device_status`

**HAS_* flags:** `HAS_DATA`, `HAS_CORE`, `HAS_OPTIMIZATION`, `HAS_CONFIG`, `HAS_DEVICE`, `HAS_CLI`
-- resolve to `bool` on first access by attempting the corresponding submodule import.

______________________________________________________________________

## 3. CLI Layer

**Location:** `homodyne/cli/` (4,635 lines)

### Console Scripts

| Script | Entry Point | Purpose |
|--------|-------------|---------|
| `homodyne` | `cli.main:main` | Main analysis (NLSQ/CMC) |
| `homodyne-config` | `cli.config_generator:main` | Config generation + validation |
| `homodyne-config-xla` | `cli.xla_config:main` | XLA device parallelism config |
| `homodyne-post-install` | `post_install:main` | Shell completion setup |
| `homodyne-cleanup` | `uninstall_scripts:main` | Remove completion files |

### CLI Dispatch Flow

```
main.py:main()
    │
    ├── Set JAX_ENABLE_X64=1 (redundant with __init__.py, needed for direct CLI entry)
    ├── configure_xla_devices() from xla_config.py
    ├── parse_arguments() from args_parser.py
    │
    └── commands.py:run_analysis(args)
         │
         ├── Phase 1: _load_and_validate_config(args)
         │   └── ConfigManager(yaml_path) → validate → ParameterManager
         │
         ├── Phase 2: _load_and_prepare_data(config, args)
         │   └── XPCSDataLoader(config) → load_data() → filtered arrays
         │
         ├── Phase 3: _run_optimization(data, config, args)
         │   ├── method == "nlsq":
         │   │   └── fit_nlsq_jax(data, config) → NLSQResult
         │   ├── method == "cmc":
         │   │   └── fit_mcmc_jax(data, config, nlsq_result=...) → CMCResult
         │   └── method == "both":
         │       └── NLSQ first → CMC with warm-start
         │
         └── Phase 4: _save_results(result, config, args)
             ├── save_nlsq_results() → JSON + NPZ
             ├── save_mcmc_results() → JSON + NPZ + NetCDF
             ├── _generate_nlsq_plots() or _generate_cmc_diagnostic_plots()
             └── _generate_analysis_summary()
```

### Argument Groups (`args_parser.py`)

| Group | Key Arguments |
|-------|---------------|
| Method | `--method {nlsq,cmc,both}`, `--use-adapter` |
| Config/IO | `--config`, `--output`, `--format {json,npz,both}` |
| NLSQ | `--max-iter`, `--tolerance`, `--cmaes`, `--multi-start` |
| CMC | `--num-chains`, `--num-warmup`, `--num-samples`, `--backend` |
| Plotting | `--plot-experimental-data`, `--plot-simulated-data`, `--no-plots` |
| Debug | `--verbose`, `--profile`, `--dry-run` |

______________________________________________________________________

## 4. Configuration System

**Location:** `homodyne/config/` (4,832 lines)

### Configuration Pipeline

```
YAML file
    │
    ▼
ConfigManager(yaml_path)                    # config/manager.py (1,296 L)
    ├── _load_yaml() → raw dict
    ├── _merge_defaults() → complete dict
    ├── _validate_config() → type + range checks
    │
    ├── .analysis_mode → "static" | "laminar_flow"
    ├── .data → DataConfig section
    ├── .optimization → {nlsq: NLSQConfig, cmc: CMCConfig}
    │
    └── get_parameter_manager() → ParameterManager
         │
         ├── .active_params → ["D0", "alpha", "D_offset", ...]
         ├── .initial_guesses → {D0: 1e4, alpha: 1.0, ...}
         ├── .bounds → {D0: (1e-2, 1e8), ...}
         │
         └── get_parameter_space() → ParameterSpace
              │
              ├── .bounds → {name: (lo, hi)} for all active params
              ├── .priors → {name: Prior} for MCMC sampling
              ├── .transforms → log-space / identity
              └── Backed by ParameterRegistry (singleton)
```

### Key Classes

| Class | File | Lines | Purpose |
|-------|------|------:|---------|
| `ConfigManager` | `manager.py` | 1,296 | YAML loading, validation, section access |
| `ParameterManager` | `parameter_manager.py` | 809 | Active params, bounds, initial guesses |
| `ParameterSpace` | `parameter_space.py` | 895 | Bounds + MCMC priors, log transforms |
| `ParameterRegistry` | `parameter_registry.py` | 632 | Singleton metadata: names, units, ranges |
| `parameter_names.py` | `parameter_names.py` | 315 | Module-level string constants (avoid magic strings, not a class) |

### Parameter Flow by Analysis Mode

| Mode | Physical Params | Per-Angle (auto) | Total |
|------|----------------:|------------------:|------:|
| `static` | 3 (D0, alpha, D_offset) | +2 (contrast, offset) | 5 |
| `laminar_flow` | 7 (+ gamma_dot0, beta, gamma_dot_offset, phi0) | +2 (contrast, offset) | 9 |

**Per-angle modes** (anti-degeneracy system):
- `auto`: averaged scaling (+2 params, quantile-estimated, then optimized)
- `constant`: fixed scaling from quantile estimation (+0 optimized params)
- `individual`: independent per angle (+2*n_phi params)
- `fourier`: truncated Fourier series (+2*(1+2K) coefficient params)

> See [physical-model-architecture.md](physical-model-architecture.md) Section 7 for full per-angle scaling details.

______________________________________________________________________

## 5. Data Loading Pipeline

**Location:** `homodyne/data/` (12,302 lines)

### Pipeline Stages

```
XPCSDataLoader(config)
    │
    ├── 1. load_data(hdf5_path)
    │   ├── _detect_format() → "old_aps" | "new_aps"
    │   ├── _load_old_format() or _load_new_format()
    │   └── _reconstruct_half_triangle() → full symmetric C2 matrix
    │
    ├── 2. _validate(data)
    │   ├── Shape consistency: C2 matches (n_q, n_phi, n_frames, n_frames)
    │   ├── Dtype: float64
    │   ├── NaN: < threshold (configurable)
    │   └── Monotonicity: time arrays strictly increasing
    │
    ├── 3. _filter(data, config)
    │   ├── Q-range filter: config.q_min .. config.q_max
    │   ├── Phi-range filter: config.phi_min .. config.phi_max (OR logic for wrapped)
    │   └── Frame-range filter: config.start_frame .. config.end_frame
    │
    ├── 4. _preprocess(data)
    │   ├── _standardize_format() → canonical shape (deep copy)
    │   ├── _correct_diagonal_enhanced() → remove C2 diagonal artifacts
    │   └── _trim_to_valid_range() → remove trailing NaN frames
    │
    ├── 5. DataQualityController.assess(data) → QualityControlResult
    │   ├── NaN fraction
    │   ├── Signal-to-noise ratio
    │   └── Angular coverage assessment
    │
    └── 6. Cache (PerformanceEngine)
        ├── Memory cache (LRU, eviction by access time)
        └── Disk cache (XDG_CACHE_HOME/homodyne/)

Output: {wavevector_q_list, phi_angles_list, t1, t2, c2_exp, dt, metadata}
```

### Key Output Arrays

| Array | Shape | Description |
|-------|-------|-------------|
| `wavevector_q_list` | `(n_q,)` | Scattering vectors |
| `phi_angles_list` | `(n_phi,)` | Azimuthal angles (radians) |
| `t1` | `(n_points,)` | First time coordinate (flattened upper triangle) |
| `t2` | `(n_points,)` | Second time coordinate |
| `c2_exp` | `(n_phi, n_points)` | Experimental C2 correlation data |
| `dt` | `float` | Time step between frames |

> See [data-handler-architecture.md](data-handler-architecture.md) for complete details.

______________________________________________________________________

## 6. Physical Model Layer

**Location:** `homodyne/core/` (9,399 lines)

### Model Hierarchy

```
CombinedModel(mode)                          # models.py
    ├── DiffusionModel                       #   g1_diffusion = exp(-D(q) * |t1 - t2|^alpha)
    ├── ShearModel (laminar_flow only)       #   g1_shear = sinc(Gamma * (t1 - t2))
    └── g1 = g1_diffusion * g1_shear        #   Combined: g1 = product
                                              #   g2 = 1 + contrast * g1^2

HomodyneModel(config)                        # homodyne_model.py (stateful wrapper)
    └── Holds pre-computed grids, calls CombinedModel

TheoryEngine(mode)                           # theory.py (validated API)
    └── High-level g1/g2 with error handling
```

### Shadow-Copy Architecture (5 Parallel Implementations)

The core physics computation (`g1 → g2`) is duplicated across 5 files for context-specific
optimization. All 5 must remain numerically consistent:

| Path | File | Optimization | Used By |
|------|------|-------------|---------|
| 1 | `physics_nlsq.py` | Meshgrid broadcasting, `@jit` | NLSQ residual functions |
| 2 | `physics_cmc.py` (precomputed) | ShardGrid pre-computation | CMC NUTS hot-path |
| 3 | `physics_cmc.py` (element-wise) | Point-by-point | CMC legacy/fallback |
| 4 | `jax_backend.py` | Dispatcher (meshgrid/element) | General API |
| 5 | `physics_utils.py` | Base shared utilities | Test reference |

**Consistency invariants** (verified across all 5):
- `epsilon_abs = 1e-12`
- `jnp.clip(log_g1, -700, 0)` for underflow protection
- `jnp.where(g1 > eps, g1, eps)` gradient-safe floor (not `jnp.maximum`)
- No `jnp.clip(g2)` on output

### Numerical Stability

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| Log-space g1 | `log_g1 = -D(q) * abs(t1-t2)^alpha + log(sinc(...))` | Prevent underflow |
| Gradient-safe floor | `jnp.where(x > eps, x, eps)` | Preserve JAX gradients |
| Safe sinc | Taylor expansion for `|x| < 1e-4` (UNNORMALIZED) | Smooth NUTS gradients |
| Safe exp | `jnp.clip(arg, -700, 700)` before `jnp.exp` | Prevent overflow |
| D(q) singularity | `dt_eff = jnp.where(dt > 1e-6, dt, 1e-6)` | Prevent div-by-zero |

> See [physical-model-architecture.md](physical-model-architecture.md) for complete model details.

______________________________________________________________________

## 7. NLSQ Optimization

**Location:** `homodyne/optimization/nlsq/` (29,356 lines)

### Entry Points

```python
# Recommended: NLSQAdapter with auto-fallback
from homodyne.optimization.nlsq import fit_nlsq_jax
result = fit_nlsq_jax(data, config, use_adapter=True)

# Full features: NLSQWrapper
from homodyne.optimization.nlsq import NLSQWrapper
wrapper = NLSQWrapper(data, config)
result = wrapper.fit()
```

### Optimization Pipeline

```
fit_nlsq_jax(data, config)                    # core.py (2,327 L)
    │
    ├── 1. CMA-ES Pre-Optimization (if scale_ratio > 1e6)
    │   └── cmaes_wrapper.py → CMAESResult → warm-start p0
    │
    ├── 2. Adapter Selection
    │   ├── use_adapter=True  → NLSQAdapter  (adapter.py, 1,143 L)
    │   └── use_adapter=False → NLSQWrapper  (wrapper.py, 8,248 L)
    │
    ├── 3. Memory Estimation & Strategy Selection (memory.py, 420 L)
    │   ├── < memory_threshold → Stratified LS (in-memory)
    │   └── > memory_threshold → Hybrid Streaming (out-of-core chunking)
    │   NOTE: Uses effective param count (9 for auto_averaged, not 53)
    │
    ├── 4. Anti-Degeneracy Defense System
    │   ├── N1: AdaptiveRegularizer (adaptive_regularization.py, 507 L)
    │   ├── N2: AntiDegeneracyController (anti_degeneracy_controller.py, 1,021 L)
    │   ├── N3: GradientMonitor (gradient_monitor.py, 548 L)
    │   └── N4: ShearWeighting (shear_weighting.py, 453 L)
    │
    ├── 5. Residual Function Construction
    │   ├── Standard: residual.py (786 L) → JIT-compiled: residual_jit.py (545 L)
    │   └── Out-of-core: chunking.py (1,205 L) → sequential.py (1,020 L)
    │
    ├── 6. NLSQ 0.6.4 curve_fit() Execution
    │   ├── Trust-region Levenberg-Marquardt
    │   ├── Jacobian via JAX autodiff (or finite-difference fallback)
    │   └── executors.py dispatches strategy
    │
    ├── 7. Multi-Start (if enabled)
    │   └── multistart.py (1,485 L) → basin hopping with best-of-N
    │
    └── 8. Result Construction
        └── result_builder.py (479 L) → NLSQResult
            │   NOTE: actual class is OptimizationResult (aliased as NLSQResult in some contexts)
            ├── .parameters: dict[str, float]
            ├── .covariance: ndarray
            ├── .reduced_chi_squared: float
            ├── .success: bool
            └── .diagnostics: dict
```

### Memory-Aware Routing

| Dataset Size | Memory | Strategy | Anti-Degeneracy |
|-------------|--------|----------|-----------------|
| < 10M points | ~2 GB/M | Stratified LS (in-memory) | Full defense layers |
| > 10M points | ~2 GB fixed | Hybrid Streaming (out-of-core) | Limited (no per-iteration) |
| Scale ratio > 1000 | bounded | CMA-ES + NLSQ refinement | CMA-ES handles global |

> See [nlsq-fitting-architecture.md](nlsq-fitting-architecture.md) for complete NLSQ details.

______________________________________________________________________

## 8. CMC Bayesian Inference

**Location:** `homodyne/optimization/cmc/` (14,227 lines)

### Entry Point

```python
from homodyne.optimization.cmc import fit_mcmc_jax

# Always NLSQ first for warm-start
nlsq_result = fit_nlsq_jax(data, config)
cmc_result = fit_mcmc_jax(data, config, nlsq_result=nlsq_result)
```

### CMC Pipeline

```
fit_mcmc_jax(data, config, nlsq_result)        # core.py (1,566 L)
    │
    ├── 1. Data Preparation & Sharding
    │   ├── data_prep.py (848 L) → split data into shards
    │   ├── Auto shard size: 3-20K points (mode + dataset dependent)
    │   └── Noise-weighted LPT scheduling (largest/noisiest first)
    │
    ├── 2. Prior Construction
    │   ├── priors.py (1,100 L) → from NLSQ posteriors
    │   ├── reparameterization.py (336 L) → log-space D0, etc.
    │   └── Reparameterized after t_ref = sqrt(dt * t_max) computed
    │
    ├── 3. NumPyro Model Specification
    │   └── model.py (1,168 L) → numpyro.sample() with priors
    │       ├── 5 model variants by analysis_mode + per_angle_mode
    │       └── Per-angle scaling integrated into model
    │
    ├── 4. NUTS Sampling per Shard
    │   ├── sampler.py (1,326 L) → SamplingPlan + NUTS execution
    │   ├── Adaptive sampling: 100-500 warmup, 200-1500 samples (by shard size)
    │   ├── 4 chains per shard (parallel via pmap on 4 virtual XLA devices)
    │   └── max_tree_depth=10 (max 2^10 leapfrog steps per NUTS step)
    │
    ├── 5. Backend Execution
    │   ├── multiprocessing.py (1,953 L) → Primary: process pool
    │   │   ├── Workers = physical_cores/2 - 1
    │   │   ├── 4 virtual JAX devices per worker
    │   │   ├── Shared memory for shard data arrays
    │   │   ├── JIT cache: jax.config.update() (not env var)
    │   │   └── OMP_NUM_THREADS=1-2 per worker
    │   ├── pbs.py (494 L) → PBS/Torque cluster
    │   └── pjit.py (269 L) → Single-process (debug)
    │
    ├── 6. Quality Filtering
    │   ├── Max divergence rate: 10% (filter corrupted shards)
    │   └── diagnostics.py (1,269 L) → R-hat, ESS, divergence analysis
    │
    ├── 7. Consensus Combination
    │   ├── Weighted average of shard posteriors
    │   ├── Bounds-aware heterogeneity detection (CV vs param range)
    │   └── scaling.py (344 L) → per-angle consensus
    │
    └── 8. Result Construction
        └── results.py (789 L) → CMCResult
            ├── .parameters: np.ndarray (posterior means)
            ├── .uncertainties: np.ndarray (posterior std devs)
            ├── .inference_data: arviz.InferenceData
            ├── .r_hat: dict[str, float]
            ├── .ess_bulk: dict[str, float]
            ├── .convergence_status: str
            └── .divergences: int
```

### Backend Comparison

| Backend | Workers | Chains | Best For |
|---------|--------:|-------:|----------|
| `multiprocessing` | cores/2 - 1 | 4/worker (pmap) | Production CPU runs |
| `pbs` | PBS nodes | 4/node | HPC clusters |
| `pjit` | 1 | 4 (pmap) | Debugging |

> See [cmc-fitting-architecture.md](cmc-fitting-architecture.md) for complete CMC details.

______________________________________________________________________

## 9. IO & Result Serialization

**Location:** `homodyne/io/` (953 lines)

### NLSQ Output

```
nlsq_writers.py:save_nlsq_results(result, output_dir)
    │
    ├── {output_dir}/nlsq_results.json
    │   ├── parameters: {D0: ..., alpha: ..., ...}
    │   ├── uncertainties: {D0: ..., ...}
    │   ├── reduced_chi_squared: float
    │   ├── success: bool
    │   └── metadata: {mode, n_points, timestamp, version}
    │
    └── {output_dir}/nlsq_results.npz
        ├── parameters (array)
        ├── covariance (matrix)
        ├── residuals (array)
        └── fitted_values (array)
```

### CMC Output

```
mcmc_writers.py:save_mcmc_results(result, output_dir)
    │
    ├── {output_dir}/mcmc_parameters.json     # Posterior means + uncertainties
    ├── {output_dir}/mcmc_analysis.json        # Convergence summary
    ├── {output_dir}/mcmc_diagnostics.json     # R-hat, ESS, divergences
    ├── {output_dir}/mcmc_results.npz          # Full posterior arrays
    └── {output_dir}/mcmc_inference_data.nc    # ArviZ NetCDF (full traces)
```

### NaN-Safe JSON Serialization

`json_utils.py` provides a custom JSON encoder that handles:
- `float('nan')` → `null`
- `float('inf')` / `float('-inf')` → `null`
- NumPy scalars → Python floats
- NumPy arrays → Python lists

______________________________________________________________________

## 10. Visualization

**Location:** `homodyne/viz/` (3,761 lines)

### Plot Types

| Module | Lines | Plots Generated |
|--------|------:|-----------------|
| `nlsq_plots.py` | 961 | C2 heatmaps (exp vs fit), residual maps, parameter contours, angular profiles |
| `mcmc_plots.py` | 1,812 | Trace plots, posterior distributions, forest plots, ESS bar charts, R-hat diagnostics, pair plots |
| `experimental_plots.py` | 298 | Raw C2 heatmaps, time series, angular coverage |
| `datashader_backend.py` | 515 | Fast rendering for datasets with >100K points |
| `diagnostics.py` | 59 | Diagonal overlay on C2 heatmaps |

### Rendering Strategy

```
Dataset size check
    │
    ├── > 100K points → datashader_backend.py
    │   └── Rasterize to pixel grid, then overlay on matplotlib
    │
    └── <= 100K points → direct matplotlib
        └── Standard scatter/imshow/contour
```

**Libraries used:** matplotlib (publication-quality), PyQtGraph (interactive), ArviZ (MCMC diagnostics), datashader (fast rendering)

______________________________________________________________________

## 11. Device & HPC Configuration

**Location:** `homodyne/device/` (1,055 lines)

### CPU Topology Detection

```
configure_optimal_device()                  # device/__init__.py
    │
    ├── cpu.py:detect_cpu_topology()
    │   ├── Physical cores (vs. hyperthreads)
    │   ├── NUMA node count
    │   ├── L3 cache size
    │   ├── AVX-512 / AVX2 support
    │   └── Memory per NUMA node
    │
    ├── config.py:recommend_workers()
    │   ├── CMC workers = physical_cores / 2 - 1
    │   ├── OMP_NUM_THREADS = 1-2 per worker
    │   └── NUMA-aware process pinning (if available)
    │
    └── Return: DeviceConfig
        ├── .n_workers: int
        ├── .threads_per_worker: int
        ├── .total_memory_gb: float
        └── .has_avx512: bool
```

### XLA Configuration

`cli/xla_config.py` and `__init__.py` configure XLA for CPU-only operation:

| XLA Flag | Value | Purpose |
|----------|-------|---------|
| `xla_force_host_platform_device_count` | 4 | Virtual devices for parallel MCMC chains |
| `xla_disable_hlo_passes` | `constant_folding` | Prevent slow compilation for large datasets |

______________________________________________________________________

## 12. Utilities

### Structured Logging (`utils/logging.py`, 1,145 lines)

```python
from homodyne.utils.logging import get_logger, log_phase

logger = get_logger(__name__)

with log_phase("NLSQ Optimization"):       # Automatic timing + memory tracking
    result = fit_nlsq_jax(data, config)
    # Logs: [NLSQ Optimization] Started...
    # Logs: [NLSQ Optimization] Completed in 12.3s (RSS: 1.2 GB)
```

Key features:
- **Phase timing:** `log_phase()` context manager with automatic duration + peak RSS
- **Memory tracking:** RSS monitoring at phase boundaries
- **AnalysisSummaryLogger:** Aggregates all phases into final summary
- **Convergence logging:** Per-iteration NLSQ cost, per-shard CMC divergences
- **NaN/Inf filtering:** Lambda fallback filters invalid values from log output

### Path Security (`utils/path_validation.py`, 299 lines)

- Path traversal prevention (no `../` escape from output directory)
- Symlink resolution validation
- Safe file creation with atomic writes where possible

______________________________________________________________________

## 13. Runtime & Deployment

### System Validator (`runtime/utils/system_validator.py`, 1,472 lines)

Comprehensive pre-flight checks:

| Check | Purpose |
|-------|---------|
| Python version | >= 3.12 required |
| JAX availability | Import + Float64 mode |
| NLSQ version | >= 0.6.4 |
| NumPyro availability | For CMC method |
| h5py availability | For HDF5 loading |
| Memory | Sufficient RAM for dataset |
| CPU topology | Core count, AVX support |
| XLA configuration | Device count, flags |

### Shell Completion System

```
homodyne-post-install
    │
    ├── Copies completion.sh (680 L) into $VENV/etc/homodyne/shell/
    ├── Configures venv activation script to source it
    │
    └── Provides:
        ├── Tab completion for all 5 console scripts
        ├── Shell aliases: hm, hconfig, hm-nlsq, hm-cmc, hc-stat, hc-flow, ...
        └── Interactive config builder (bash/zsh)
```

| Alias | Expansion |
|-------|-----------|
| `hm` | `homodyne` |
| `hm-nlsq` | `homodyne --method nlsq` |
| `hm-cmc` | `homodyne --method cmc` |
| `hconfig` | `homodyne-config` |
| `hc-stat` | `homodyne-config --mode static` |
| `hc-flow` | `homodyne-config --mode laminar_flow` |

______________________________________________________________________

## 14. Fault Tolerance & Recovery

**Location:** `homodyne/optimization/` (top-level, 2,143 lines)

### Checkpoint Manager (`checkpoint_manager.py`, 507 lines)

```
HDF5-based fault-tolerant checkpointing
    │
    ├── save_checkpoint(state, path)
    │   ├── Atomic write (temp file → rename)
    │   ├── Version metadata for forward compatibility
    │   └── Compression for large arrays
    │
    └── load_checkpoint(path)
        ├── Version check (warn if mismatch)
        └── Graceful degradation for missing fields
```

### Exception Hierarchy (`exceptions.py`, 352 lines)

```
NLSQOptimizationError(Exception)
    ├── NLSQConvergenceError
    ├── NLSQNumericalError
    └── NLSQCheckpointError
```

### Recovery Strategies (`recovery_strategies.py`, 209 lines)

| Strategy | Trigger | Action |
|----------|---------|--------|
| Retry with relaxed tolerance | ConvergenceError | Increase `xtol` by 10x |
| Fallback to finite-difference | Jacobian NaN | Switch from autodiff to FD |
| Reduce trust region | NumericalError | Halve initial trust radius |
| Skip corrupted shard | DivergenceError | Remove shard from consensus |

### Numerical Validation (`numerical_validation.py`, 254 lines)

Pre- and post-optimization checks:
- Parameter NaN/Inf detection
- Covariance matrix positive-definiteness
- Condition number monitoring
- Gradient norm sanity checks

### Gradient Diagnostics (`gradient_diagnostics.py`, 437 lines)

Detects parameter-gradient imbalance that causes NLSQ to stall:
- Per-parameter gradient magnitude analysis
- Identifies when shear parameters dominate diffusion (or vice versa)
- Recommends rescaling or regularization adjustments

______________________________________________________________________

## 15. Cross-Cutting Concerns

### NLSQ → CMC Integration

The primary analysis workflow runs NLSQ first, then uses results to warm-start CMC:

```
NLSQ Result
    │
    ├── .parameters → CMC prior centers (Normal distributions)
    ├── .covariance → CMC prior widths (scaled by 3-5x)
    ├── .reduced_chi_squared → Quality gate (skip CMC if poor fit)
    │
    └── priors.py:build_priors_from_nlsq(nlsq_result)
        ├── Transform to reparameterized space (log D0, etc.)
        ├── Clip widths to physical bounds
        └── Return: {name: numpyro.distributions.Normal(loc, scale)}
```

**Impact:** Reduces CMC divergences from ~28% (flat priors) to <5% (NLSQ warm-start).

### Per-Angle Scaling (Cross-System)

The per-angle scaling system spans multiple modules to prevent parameter absorption
degeneracy in laminar_flow mode:

| Component | Location | Role |
|-----------|----------|------|
| Configuration | `config/parameter_manager.py` | Mode selection (auto/constant/individual/fourier) |
| Estimation | `nlsq/parameter_utils.py` | Quantile-based contrast/offset estimation |
| NLSQ transforms | `nlsq/transforms.py` | Expand compressed params ↔ per-angle arrays |
| Anti-degeneracy | `nlsq/anti_degeneracy_controller.py` | Regularize scaling params |
| CMC model | `cmc/model.py` | Scaling built into NumPyro model |
| CMC scaling | `cmc/scaling.py` | Per-angle consensus combination |
| Physics | `core/scaling_utils.py` | Apply contrast/offset to g2 |

### Float64 Enforcement

Float64 is enforced at multiple levels to prevent silent precision loss:

| Level | Mechanism |
|-------|-----------|
| Package init | `os.environ.setdefault("JAX_ENABLE_X64", "1")` in `__init__.py` |
| CLI entry | Redundant set in `cli/main.py` |
| CMC workers | Re-set in `multiprocessing.py` (spawn-mode = fresh env) |
| Data loading | `validation.py` checks dtype == float64 |

### JIT Compilation Cache

Workers use `jax.config.update()` (not env vars) for persistent cache:

```python
# In each CMC worker process (after importing JAX):
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

The `min_compile_time_secs=0` is critical: CMC functions compile in 0.07-0.15s,
below the default 1.0s threshold. Without this, every worker recompiles (~10s wasted).

______________________________________________________________________

## Complete End-to-End Flow

```
$ homodyne --config experiment.yaml --method both --output ./results

                                PHASE 1: CONFIGURATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cli/main.py
    ├── os.environ.setdefault("JAX_ENABLE_X64", "1")
    ├── XLA_FLAGS = "...device_count=4 ...disable_hlo_passes=constant_folding"
    └── args = parse_arguments()

cli/commands.py:_load_and_validate_config(args)
    ├── ConfigManager("experiment.yaml")
    │   ├── Load YAML → merge defaults → validate
    │   ├── analysis_mode = "laminar_flow"
    │   └── per_angle_mode = "auto"
    └── ParameterManager → ParameterSpace (9 params: 7 physical + 2 scaling)

                                PHASE 2: DATA LOADING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

data/xpcs_loader.py:load_data(hdf5_path)
    ├── Detect format → Load HDF5 → Reconstruct half-triangle
    ├── Validate: shape, dtype=float64, NaN < threshold, monotonic time
    ├── Filter: Q-range, phi-range (OR logic for wrapped), frame-range
    ├── Preprocess: standardize → diagonal correction → trim
    ├── Quality assessment → QualityReport
    └── Output: {wavevector_q_list, phi_angles_list, t1, t2, c2_exp, dt}
         Example: 23 angles, 500 frames → ~2.9M points per angle

                           PHASE 3A: NLSQ OPTIMIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

optimization/nlsq/core.py:fit_nlsq_jax(data, config)
    ├── CMA-ES check: scale_ratio(D0/gamma_dot) > 1e6 → global pre-optimization
    ├── Adapter selection: NLSQAdapter (default)
    ├── Memory estimation: effective_params=9, data_size → strategy
    │   ├── < threshold → Stratified LS (in-memory)
    │   └── > threshold → Hybrid Streaming (chunked)
    ├── Anti-degeneracy:
    │   ├── AdaptiveRegularizer → penalize scaling drift
    │   ├── AntiDegeneracyController → detect & correct absorption
    │   ├── GradientMonitor → watch for gradient collapse
    │   └── ShearWeighting → balance angular contributions
    ├── Residual function: JIT-compiled g2(params) - c2_exp
    │   └── physics_nlsq.py → meshgrid-optimized g1/g2
    ├── NLSQ 0.6.4 curve_fit() → Trust-region L-M
    │   ├── JAX autodiff Jacobian (or finite-difference fallback)
    │   ├── DOF = n_points - effective_params (9, not 53)
    │   └── Convergence: xtol + ftol (both required for OOC)
    └── NLSQResult
         ├── parameters: {D0: 1.2e4, alpha: 0.98, ..., contrast_avg: 0.03, offset_avg: 1.001}
         ├── reduced_chi_squared: 1.02
         └── success: True

                           PHASE 3B: CMC BAYESIAN INFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

optimization/cmc/core.py:fit_mcmc_jax(data, config, nlsq_result)
    ├── Shard data: ~2.9M pts → ~580 shards of 5K pts each
    │   └── Noise-weighted LPT scheduling (noisy shards first)
    ├── Build NLSQ-informed priors:
    │   ├── t_ref = sqrt(dt * t_max)
    │   ├── Transform to reparameterized space (log D0, etc.)
    │   └── Normal(nlsq_mean, 3-5x * nlsq_std)
    ├── NumPyro model specification (laminar_flow + auto scaling)
    ├── Backend: multiprocessing
    │   ├── Workers = physical_cores/2 - 1
    │   ├── Shared memory for shard arrays (no serialization)
    │   ├── Each worker: 4 parallel NUTS chains (pmap on 4 XLA devices)
    │   ├── Adaptive sampling: SamplingPlan(warmup=250, samples=750) for 5K shards
    │   └── JIT cache: jax.config.update() with min_compile_time=0
    ├── Quality filter: remove shards with > 10% divergences
    ├── Consensus combination:
    │   └── Weighted average of shard posteriors
    └── CMCResult
         ├── parameters: {D0: 1.2e4 +/- 200, ...}
         ├── convergence: {r_hat_max: 1.01, min_ess: 850, divergence_rate: 0.03}
         └── inference_data: arviz.InferenceData (full posterior traces)

                               PHASE 4: OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

cli/commands.py:_save_results(nlsq_result, cmc_result, config, args)
    │
    ├── NLSQ Output:
    │   ├── ./results/nlsq_results.json    (parameters + metadata)
    │   └── ./results/nlsq_results.npz     (arrays: covariance, residuals)
    │
    ├── CMC Output:
    │   ├── ./results/mcmc_parameters.json  (posterior means + uncertainties)
    │   ├── ./results/mcmc_analysis.json    (convergence summary)
    │   ├── ./results/mcmc_diagnostics.json (R-hat, ESS, divergences)
    │   ├── ./results/mcmc_results.npz      (full posterior arrays)
    │   └── ./results/mcmc_inference_data.nc (ArviZ NetCDF)
    │
    ├── Plots:
    │   ├── NLSQ: C2 heatmaps, residual maps, parameter contours
    │   └── CMC: Trace plots, posterior distributions, forest, ESS
    │
    └── Summary:
        └── ./results/analysis_summary.json (combined report)
```

______________________________________________________________________

## Module Size & Dependency Matrix

### Internal Dependency Flow

```
                    config/
                      │
              ┌───────┴───────┐
              ▼               ▼
           data/           device/
              │
              ▼
           core/
              │
       ┌──────┴──────┐
       ▼              ▼
   nlsq/           cmc/
       │              │
       └──────┬───────┘
              │
       ┌──────┴──────┐
       ▼              ▼
     io/            viz/
              │
              ▼
           cli/
```

### External Dependencies

| Package | Version | Used By | Purpose |
|---------|---------|---------|---------|
| JAX | >= 0.8.2 | core, optimization | JIT, vmap, grad, hessian |
| NumPy | >= 2.3 | data, io | Array I/O boundaries |
| NLSQ | >= 0.6.4 | optimization/nlsq | Trust-region L-M solver |
| NumPyro | >= 0.19 | optimization/cmc | NUTS sampling |
| h5py | * | data | HDF5 loading |
| ArviZ | * | optimization/cmc, viz | MCMC diagnostics |
| scipy | * | core, optimization | Special functions, stats |
| matplotlib | * | viz | Publication plots |
| datashader | * | viz | Fast rendering |
| PyYAML | * | config | YAML parsing |

### Test Coverage

| Module | Test Lines | Test Files |
|--------|--------:|--------:|
| Total tests | 74,858 | 157 files |
| tests/unit/ | majority | Core unit tests |
| tests/integration/ | subset | End-to-end workflows |

______________________________________________________________________

## Architecture Decision Records

### ADR-1: CPU-Only Architecture

**Decision:** No GPU/TPU support. All computation on CPU with 4 virtual XLA devices.

**Rationale:** XPCS data fits in CPU memory. HPC nodes (36-128 cores) provide sufficient
parallelism. GPU transfer overhead exceeds compute benefit for typical dataset sizes.
Virtual XLA devices enable parallel MCMC chains without GPU hardware.

### ADR-2: Shadow-Copy Physics Implementation

**Decision:** Maintain 5 parallel g1/g2 implementations optimized for different execution contexts.

**Rationale:** NLSQ needs meshgrid broadcasting for efficient Jacobian computation.
CMC NUTS hot-path needs pre-computed ShardGrid to avoid per-step time grid construction.
Legacy/fallback paths needed for testing and edge cases. The duplication cost is managed
through strict consistency invariants verified in tests.

### ADR-3: Lazy Import System

**Decision:** Use `__getattr__` hook in `__init__.py` to defer all submodule imports.

**Rationale:** JAX + NumPyro + h5py imports take 3-6 seconds. CLI commands like
`homodyne --help` or `homodyne-config` only need argument parsing. Lazy loading
makes CLI response feel instant while preserving the `from homodyne import X` API.

### ADR-4: NLSQ-First Workflow

**Decision:** Always run NLSQ before CMC. CMC never runs with flat/uninformed priors.

**Rationale:** NLSQ warm-start reduces CMC divergences from ~28% to <5%. The cost of
a 10-second NLSQ run saves hours of wasted MCMC sampling with poor convergence.

### ADR-5: Per-Angle Scaling System

**Decision:** `auto` mode averages per-angle contrast/offset to 2 extra parameters.

**Rationale:** Individual per-angle scaling (53 params for 23 angles) causes parameter
absorption degeneracy where contrast absorbs physical signal. Averaged scaling provides
sufficient flexibility (2 extra params) while the anti-degeneracy defense system monitors
for remaining drift.

### ADR-6: Gradient-Safe Floors

**Decision:** `jnp.where(x > eps, x, eps)` instead of `jnp.maximum(x, eps)`.

**Rationale:** `jnp.maximum` zeros the gradient below the floor, stalling both NLSQ
Jacobian computation and NUTS leapfrog integration. `jnp.where` preserves the gradient
through the floor, allowing optimizers to escape from boundary regions.

### ADR-7: JIT Cache via Config API

**Decision:** Use `jax.config.update()` instead of environment variables for JIT cache.

**Rationale:** In JAX 0.8+, `os.environ["JAX_COMPILATION_CACHE_DIR"]` alone does NOT
enable the persistent compilation cache. The config API is the only reliable method.
Additionally, `min_compile_time_secs` must be set to 0 because CMC functions compile
in 0.07-0.15s, below the default 1.0s threshold.

______________________________________________________________________

## Companion Documents

| Document | Scope | Lines |
|----------|-------|------:|
| [physical-model-architecture.md](physical-model-architecture.md) | Physics models, JIT kernels, shadow copies, per-angle scaling, numerical stability | ~1,340 |
| [nlsq-fitting-architecture.md](nlsq-fitting-architecture.md) | NLSQ optimizer, anti-degeneracy, CMA-ES, memory routing, strategies | ~1,060 |
| [cmc-fitting-architecture.md](cmc-fitting-architecture.md) | CMC pipeline, sharding, NUTS, backends, quality filtering, diagnostics | ~1,600 |
| [data-handler-architecture.md](data-handler-architecture.md) | HDF5 loading, config, filtering, preprocessing, QC, caching, result writing | ~1,085 |
| **homodyne-architecture-overview.md** (this file) | **System-level overview, integration, cross-cutting concerns, ADRs** | **~1,300** |

### Navigation Guide

| Topic | Primary Document | Section |
|-------|-----------------|---------|
| Core equation (c2, g1, g2) | physical-model-architecture.md | Section 1: Mathematical Foundation |
| Model hierarchy | physical-model-architecture.md | Section 3: Model Hierarchy |
| Shadow copies | physical-model-architecture.md | Section 6: Shadow-Copy Architecture |
| Per-angle scaling | physical-model-architecture.md | Section 7: Per-Angle Scaling System |
| Numerical stability | physical-model-architecture.md | Section 8: Numerical Stability Techniques |
| NLSQ pipeline | nlsq-fitting-architecture.md | Full document |
| Anti-degeneracy | nlsq-fitting-architecture.md | Section 8: Anti-Degeneracy Defense System |
| CMA-ES | nlsq-fitting-architecture.md | Section 3: CMA-ES Global Optimization |
| Memory routing | nlsq-fitting-architecture.md | Section 5: Memory & Strategy Selection |
| CMC pipeline | cmc-fitting-architecture.md | Full document |
| NUTS sampling | cmc-fitting-architecture.md | Section 7: NUTS Sampling |
| CMC backends | cmc-fitting-architecture.md | Section 8: Backend Execution |
| Quality filtering | cmc-fitting-architecture.md | Section 10: Result Creation & Diagnostics |
| HDF5 loading | data-handler-architecture.md | Section 3: HDF5 Format Detection & Loading |
| Data preprocessing | data-handler-architecture.md | Section 5: Preprocessing Pipeline |
| Config system | data-handler-architecture.md | Section 1: Configuration System |
| Result writing | data-handler-architecture.md | Sections 9-10: Result Writing |
| CLI orchestration | **this document** | Section 3: CLI Layer |
| Device/HPC config | **this document** | Section 11: Device & HPC Configuration |
| Fault tolerance | **this document** | Section 14: Fault Tolerance & Recovery |
| Architecture decisions | **this document** | Architecture Decision Records |
