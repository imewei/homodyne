# Homodyne Data Handler Architecture

Complete documentation of the data loading, configuration, and result writing systems in homodyne.

**Version:** 2.22.2 **Last Updated:** February 2026

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
1. [Configuration System](#1-configuration-system)
1. [Data Loading](#2-data-loading)
1. [HDF5 Format Detection & Loading](#3-hdf5-format-detection--loading)
1. [Data Filtering](#4-data-filtering)
1. [Preprocessing Pipeline](#5-preprocessing-pipeline)
1. [Quality Control](#6-quality-control)
1. [Caching & Performance](#7-caching--performance)
1. [Memory Management](#8-memory-management)
1. [Result Writing (NLSQ)](#9-result-writing-nlsq)
1. [Result Writing (CMC)](#10-result-writing-cmc)
1. [CLI Orchestration](#11-cli-orchestration)
1. [Complete Data Flow](#complete-data-flow)
1. [Quick Reference Tables](#quick-reference-tables)
1. [Key Files Reference](#key-files-reference)

______________________________________________________________________

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER ENTRY POINTS                                   │
│                                                                                  │
│    CLI: homodyne --config my.yaml      API: XPCSDataLoader(config_dict=...)     │
│              │                                       │                           │
│              ▼                                       │                           │
│    ┌────────────────────────┐                        │                           │
│    │ ConfigManager          │◄───────────────────────┘                           │
│    │ (config/manager.py)    │                                                    │
│    └──────────┬─────────────┘                                                    │
│               │                                                                  │
│    ┌──────────▼─────────────┐                                                    │
│    │ XPCSDataLoader         │ HDF5 → Validate → Filter → Preprocess → Cache     │
│    │ (data/xpcs_loader.py)  │                                                    │
│    └──────────┬─────────────┘                                                    │
│               │                                                                  │
│    {wavevector_q_list, phi_angles_list, t1, t2, c2_exp}                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OPTIMIZATION LAYER                                       │
│                                                                                  │
│          fit_nlsq_jax(data, config)    fit_mcmc_jax(data, config)               │
│                   │                              │                               │
│                   ▼                              ▼                               │
│          OptimizationResult               CMCResult                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
════════════════════════════════════╪══════════════════════════════════════════════
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          RESULT WRITING                                          │
│                                                                                  │
│   save_nlsq_json_files()   save_nlsq_npz_file()   create_mcmc_*_dict()         │
│   (io/nlsq_writers.py)     (io/nlsq_writers.py)   (io/mcmc_writers.py)          │
│                                                                                  │
│   Output: parameters.json + analysis_results.json + fitted_data.npz + plots     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 1. Configuration System

**Files:** `config/manager.py`, `config/parameter_manager.py`, `config/parameter_space.py`, `config/parameter_names.py`, `config/parameter_registry.py`, `config/types.py`

### ConfigManager

```python
class ConfigManager:
    def __init__(
        self,
        config_file: str = "homodyne_config.yaml",
        config_override: dict = None,
    ):
```

### YAML Configuration Schema

```yaml
# Top-level sections
analysis_mode: "laminar_flow"       # "static", "static_isotropic", "laminar_flow"

experimental_data:
  data_folder_path: "/path/to/data"
  data_file_name: "experiment.hdf5"
  cache_file_path: "cache.npz"      # Optional NPZ cache
  apply_diagonal_correction: true    # Mandatory since v2.14.2

analyzer_parameters:
  dt: 0.1                           # Time step (seconds)
  start_frame: 1                    # 1-indexed
  end_frame: -1                     # -1 = all frames
  wavevector_q: 0.0123              # Target q-vector (1/A)
  stator_rotor_gap: 2000000.0       # Gap length (nm → A conversion)

data_filtering:
  enabled: true
  phi_range: {min: -90, max: 90}    # Degrees, supports wrapping
  q_range: {min: 0.01, max: 0.05}  # 1/A

parameter_space:
  bounds:
    - {name: "D0", min: 100, max: 100000, prior_mu: 10000, prior_sigma: 5000}
    - {name: "alpha", min: -2.0, max: 2.0}
    # ...

initial_parameters:
  parameter_names: ["D0", "alpha", "D_offset", ...]
  values: [10000, 0.5, 100, ...]
  fixed_parameters: {}               # {name: fixed_value}
```

### ConfigManager Key Methods

```
┌───────────────────────────────────────────────────────────────────────────┐
│ ConfigManager Public Interface                                            │
│                                                                           │
│  config.config → dict                Full configuration dictionary       │
│  config.get_config() → dict          Full config dict (no arguments)     │
│  config.get_parameter_bounds() → list[dict]  Parameter bounds            │
│  config.get_active_parameters() → list[str]  Active parameter names      │
│  config.get_initial_parameters() → dict      Initial values              │
│  config.get_cmc_config() → dict      CMC-specific configuration          │
│  config.get_target_angle_ranges() → list     Phi angle ranges            │
│  config.is_static_mode_enabled() → bool      Static mode check           │
│                                                                           │
│  Internal:                                                                │
│  _normalize_schema()          Flat → nested config migration              │
│  _normalize_analysis_mode()   Alias resolution (static_isotropic → ...)  │
│  _validate_config()           Schema validation                           │
│  _get_parameter_manager() → ParameterManager                             │
└───────────────────────────────────────────────────────────────────────────┘
```

### ParameterManager

```
┌───────────────────────────────────────────────────────────────────────────┐
│ ParameterManager (config/parameter_manager.py)                            │
│                                                                           │
│  Centralized parameter bounds and validation                             │
│                                                                           │
│  get_parameter_bounds(names) → list[{name, min, max, type}]              │
│  get_active_parameters() → list[str]                                     │
│  get_optimizable_parameters() → list[str]   Active minus fixed           │
│  get_bounds_as_tuples() → [(min, max), ...]                              │
│  validate_physical_constraints(params) → ValidationResult                │
│                                                                           │
│  Default Bounds (hardcoded):                                             │
│    D0:                 [1e2, 1e5]                                        │
│    alpha:              [-2.0, 2.0]                                       │
│    D_offset:           [-1e5, 1e5]                                       │
│    gamma_dot_t0:       [1e-6, 1e4]                                       │
│    beta:               [-2.0, 2.0]                                       │
│    gamma_dot_t_offset: [0.01, 100.0]                                     │
│    phi0:               [-pi, pi]                                         │
│    contrast:           [0.0, 1.0]  (Note: parameter_registry.py uses    │
│                                    [0.01, 1.5] — inconsistency exists)  │
│    offset:             [0.5, 1.5]                                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### ParameterSpace (for CMC)

```python
class ParameterSpace:
    @classmethod
    def from_config(config_dict) -> ParameterSpace:
        # Returns: bounds dict + PriorDistribution dict + parameter_names list
```

Provides bounds and priors for NumPyro MCMC sampling. Constructed from YAML `parameter_space.bounds` section.

### ParameterRegistry (Singleton)

Consolidates parameter name/metadata duplication across the codebase:

```python
registry.get_param_names(mode="laminar_flow")
# → ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]

registry.get_all_param_names(mode="laminar_flow", n_angles=23)
# → ["contrast_0", ..., "contrast_22", "offset_0", ..., "offset_22", "D0", ...]
```

### Parameter Name Constants

```python
# config/parameter_names.py
STATIC_ISOTROPIC_PARAMS = ["contrast", "offset", "D0", "alpha", "D_offset"]  # 5
LAMINAR_FLOW_PARAMS = STATIC_ISOTROPIC_PARAMS + [
    "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
]  # 9
```

______________________________________________________________________

## 2. Data Loading

**File:** `data/xpcs_loader.py` (~2107 lines)

### XPCSDataLoader

```python
class XPCSDataLoader:
    def __init__(
        self,
        config_path: str = None,           # Path to YAML/JSON config
        config_dict: dict = None,          # Or direct config dict
        configure_logging: bool = True,
        generate_quality_reports: bool = False,
    ):
```

### load_experimental_data() Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│ load_experimental_data() [xpcs_loader.py:590-742]                        │
│                                                                           │
│  1. Check NPZ cache                                                      │
│     ├─ _load_from_cache() → validate cache q-vector hash                │
│     └─ Cache hit? Return cached data (skip HDF5)                        │
│                                                                           │
│  2. Load from HDF5                                                       │
│     ├─ _detect_format() → "aps_old" or "aps_u"                          │
│     ├─ _load_aps_old_format() or _load_aps_u_format()                   │
│     │   ├─ Read correlation matrices from HDF5 groups                    │
│     │   └─ Extract q-vectors, phi angles                                │
│     ├─ _select_optimal_wavevector() → closest to config q               │
│     └─ _get_selected_indices() → q-variants near target                 │
│                                                                           │
│  3. Post-processing                                                      │
│     ├─ _reconstruct_full_matrix() → half-triangle to symmetric          │
│     ├─ _correct_diagonal() or _correct_diagonal_batch()                 │
│     │   └─ Optional: _correct_diagonal_batch_jax() (JIT path)           │
│     ├─ _apply_frame_slicing_to_selected_q() → [start:end+1]            │
│     └─ _calculate_time_arrays() → t1, t2 from dt and n_frames          │
│                                                                           │
│  4. Optional stages                                                      │
│     ├─ _integrate_with_phi_filtering() → angle selection                │
│     ├─ _apply_preprocessing_pipeline() → PreprocessingPipeline          │
│     └─ _initialize_quality_control() → DataQualityController            │
│                                                                           │
│  5. Cache and return                                                     │
│     ├─ _save_to_cache() → NPZ with metadata                            │
│     ├─ _validate_loaded_data() → shape/dtype checks                     │
│     └─ Return data dict                                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

### Return Data Structure

```python
data = loader.load_experimental_data()
# Returns:
{
    "wavevector_q_list": np.ndarray,  # (n_q,)   - selected q vectors [1/A]
    "phi_angles_list":   np.ndarray,  # (n_phi,) - angles [degrees]
    "t1":                np.ndarray,  # (n_time,) - [0, dt, 2dt, ..., (n-1)*dt]
    "t2":                np.ndarray,  # (n_time,) - [0, dt, 2dt, ..., (n-1)*dt]
    "c2_exp":            np.ndarray,  # (n_phi, n_time, n_time) - correlation
}
```

**Key properties:**

- `t1`, `t2` are always 1D arrays (no 2D meshgrids)
- `c2_exp` is symmetric, diagonal-corrected
- Frame slicing: `data[start_frame-1 : end_frame]` (config is 1-indexed)
- Time arrays start from 0: `[0, dt, 2*dt, ..., (n_frames-1)*dt]`

______________________________________________________________________

## 3. HDF5 Format Detection & Loading

**File:** `data/xpcs_loader.py`

### Format Detection

```
┌───────────────────────────────────────────────────────────────────────────┐
│ _detect_format() [xpcs_loader.py]                                        │
│                                                                           │
│   Inspects HDF5 file structure to determine format:                      │
│                                                                           │
│   ┌─ "aps_old": Legacy APS format                                       │
│   │   • Groups: /exchange/C2T_all or similar                            │
│   │   • Half-triangle storage (upper-tri only)                           │
│   │   • Multiple q-vectors in single file                               │
│   │   • Phi angles extracted from group names (regex)                   │
│   │                                                                      │
│   └─ "aps_u": Modern APS Unified format                                 │
│       • Groups: /xpcs/...                                                │
│       • Full matrix storage                                              │
│       • Standardized metadata attributes                                 │
│                                                                           │
│   Raises XPCSDataFormatError if format unrecognized                      │
└───────────────────────────────────────────────────────────────────────────┘
```

### Half-Triangle Reconstruction

```
┌───────────────────────────────────────────────────────────────────────────┐
│ _reconstruct_full_matrix() [xpcs_loader.py]                              │
│                                                                           │
│   APS old format stores only upper triangle of C2(t1, t2):              │
│                                                                           │
│   Input:  Half-triangle array (flattened or upper-tri)                  │
│   Output: Full symmetric matrix                                         │
│                                                                           │
│   ┌─ ─ ─ ─ ─┐       ┌─────────┐                                        │
│   │ * * * * *│       │ a b c d e│                                        │
│   │   * * * *│  →    │ b f g h i│                                        │
│   │     * * *│       │ c g j k l│                                        │
│   │       * *│       │ d h k m n│                                        │
│   │         *│       │ e i l n o│                                        │
│   └─ ─ ─ ─ ─┘       └─────────┘                                        │
│   (upper-tri)        (full symmetric)                                    │
│                                                                           │
│   C2(t1, t2) = C2(t2, t1) by time-reversal symmetry                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Diagonal Correction

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Diagonal Correction (mandatory since v2.14.2)                            │
│                                                                           │
│   The diagonal C2(t, t) contains autocorrelation peaks that are          │
│   physically distinct from the off-diagonal correlation signal.          │
│                                                                           │
│   Method: Replace diagonal with interpolated off-diagonal values         │
│                                                                           │
│   Three implementations:                                                 │
│   1. _correct_diagonal()           - Single matrix, NumPy               │
│   2. _correct_diagonal_batch()     - Batch of matrices, NumPy           │
│   3. _correct_diagonal_batch_jax() - Batch, JIT-compiled (if available) │
│                                                                           │
│   Applied POST-LOAD to ensure cached + fresh data receive                │
│   uniform treatment                                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

### Q-Vector Selection

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Q-Vector Selection [xpcs_loader.py]                                      │
│                                                                           │
│   1. _select_optimal_wavevector():                                       │
│      • Finds q closest to config.wavevector_q                           │
│      • Selects nearby q-variants (same phi angles)                      │
│                                                                           │
│   2. _get_selected_indices():                                            │
│      • Returns indices for all q-vector variants near target            │
│      • Typically ~23 entries (different phi angles at same |q|)         │
│                                                                           │
│   Output: phi_angles_list (n_phi,), c2_exp (n_phi, n_time, n_time)      │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 4. Data Filtering

**Files:** `data/filtering_utils.py`, `data/angle_filtering.py`, `data/phi_filtering.py`, `data/validators.py`

### Filtering Pipeline

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Data Filtering Stages                                                     │
│                                                                           │
│  1. PHI ANGLE FILTERING (data_filtering.phi_range)                       │
│     ├─ Standard range: min <= phi <= max                                │
│     ├─ Wrapped range:  phi >= min OR phi <= max (when min > max)        │
│     │   Handles ranges crossing ±180 degrees                            │
│     └─ Returns: filtered phi_angles_list, c2_exp                        │
│                                                                           │
│  2. Q-RANGE FILTERING (data_filtering.q_range)                           │
│     └─ Filters wavevector_q_list to [q_min, q_max]                     │
│                                                                           │
│  3. FRAME-BASED FILTERING (analyzer_parameters)                          │
│     └─ start_frame, end_frame → slice c2 and time arrays               │
│                                                                           │
│  4. QUALITY-BASED FILTERING (optional)                                   │
│     └─ Remove angles with low signal quality                            │
│                                                                           │
│  5. T=0 EXCLUSION (CLI: _exclude_t0_from_analysis)                       │
│     └─ Remove first time point to prevent D(t)→∞ for alpha < 0         │
│                                                                           │
│  CRITICAL: Phi filtering uses OR logic for wrapped ranges               │
│  (phi_min > phi_max means range crosses ±180 degrees)                   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Validators

```python
# data/validators.py - Input validation at I/O boundaries
validate_numeric_range(value, name, min_val, max_val)
validate_array_shape(arr, expected_shape, name)
# Supports wrapped phi ranges (min > max)
```

______________________________________________________________________

## 5. Preprocessing Pipeline

**File:** `data/preprocessing.py` (~1153 lines)

### PreprocessingPipeline

```
┌───────────────────────────────────────────────────────────────────────────┐
│ PreprocessingPipeline [preprocessing.py]                                  │
│                                                                           │
│  Stages (executed in order):                                             │
│                                                                           │
│  1. DIAGONAL_CORRECTION (mandatory)                                      │
│     ├─ basic: Average nearest off-diagonal neighbors                    │
│     ├─ statistical: Median of nearby off-diagonal elements              │
│     └─ interpolation: Interpolate from off-diagonal values              │
│                                                                           │
│  2. NORMALIZATION (optional)                                             │
│     ├─ mean: Divide by mean                                             │
│     ├─ min_max: Scale to [0, 1]                                         │
│     └─ z_score: (x - mean) / std                                        │
│                                                                           │
│  3. NOISE_REDUCTION (optional)                                           │
│     ├─ median_filter: Spatial median filter                             │
│     └─ gaussian_filter: Gaussian smoothing                              │
│                                                                           │
│  4. FORMAT_STANDARDIZATION                                               │
│     └─ Ensure float64, contiguous memory layout                         │
│                                                                           │
│  5. OUTPUT_VALIDATION                                                    │
│     └─ Shape, dtype, NaN/Inf checks                                    │
│                                                                           │
│  Each stage records a TransformationRecord in PreprocessingProvenance    │
│  for full audit trail                                                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Provenance Tracking

```python
@dataclass
class PreprocessingProvenance:
    pipeline_id: str
    stages: list[TransformationRecord]   # Complete audit trail
    input_shape: tuple
    output_shape: tuple
    timestamp: str

    def to_dict(self) -> dict: ...       # JSON-serializable

@dataclass
class TransformationRecord:
    stage: PreprocessingStage
    method: str
    parameters: dict
    input_shape: tuple
    output_shape: tuple
    duration_ms: float
```

______________________________________________________________________

## 6. Quality Control

**File:** `data/quality_controller.py` (~1646 lines)

### DataQualityController

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Quality Control System [quality_controller.py]                            │
│                                                                           │
│  Four validation stages (progressive):                                   │
│                                                                           │
│  Stage 1: RAW_DATA                                                       │
│    ├─ Shape/dtype validation                                            │
│    ├─ NaN/Inf detection                                                 │
│    └─ Basic value range checks                                          │
│                                                                           │
│  Stage 2: FILTERED_DATA                                                  │
│    ├─ Angle coverage assessment                                         │
│    ├─ Data completeness check                                           │
│    └─ Consistency with raw data                                         │
│                                                                           │
│  Stage 3: PREPROCESSED_DATA                                              │
│    ├─ Transformation fidelity assessment                                │
│    ├─ Preprocessing artifact detection                                  │
│    └─ Statistical distribution checks                                   │
│                                                                           │
│  Stage 4: FINAL_DATA                                                     │
│    ├─ Comprehensive quality assessment                                  │
│    ├─ Analysis readiness evaluation                                     │
│    └─ Overall quality score computation                                 │
│                                                                           │
│  Auto-Repair Strategies:                                                 │
│    ├─ NaN replacement (interpolation or zero-fill)                      │
│    ├─ Infinite value capping                                            │
│    ├─ Negative correlation repair                                       │
│    └─ Scaling issue correction                                          │
│                                                                           │
│  QualityLevel enum: NONE, BASIC, STANDARD, COMPREHENSIVE                 │
│  (validation intensity levels, not quality scores)                        │
│  Quality score thresholds (0-100 scale):                                  │
│    pass_threshold=50.0, warn_threshold=70.0, excellent_threshold=85.0     │
│                                                                           │
│  Output: QualityControlResult per stage + DataQualityReport (optional)   │
└───────────────────────────────────────────────────────────────────────────┘
```

### Quality Control Result

```python
@dataclass
class QualityControlResult:
    stage: QualityControlStage
    passed: bool
    metrics: QualityMetrics
    issues: list[ValidationIssue] = field(default_factory=list)
    repairs_applied: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    processing_time: float = 0.0
    data_shape_before: tuple | None = None
    data_shape_after: tuple | None = None
    data_modified: bool = False
```

### QualityLevel Enum

```python
class QualityLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
```

These represent **validation intensity levels** (how much validation to perform), not quality scores. Quality scores use thresholds from `QualityControlConfig`: `pass_threshold=50.0`, `warn_threshold=70.0`, `excellent_threshold=85.0` (0-100 scale).

______________________________________________________________________

## 7. Caching & Performance

**File:** `data/performance_engine.py` (~1502 lines)

### Multi-Level Cache

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Caching Strategy [performance_engine.py + xpcs_loader.py]                │
│                                                                           │
│  Level 1: NPZ File Cache (primary)                                       │
│    ├─ Location: config.cache_file_path or auto-generated                │
│    ├─ Format: np.savez_compressed (zlib deflate)                        │
│    ├─ Metadata: q-vector hash for validity check                        │
│    └─ _validate_cache_q_vector() → reject stale cache                   │
│                                                                           │
│  Level 2: Memory Cache (PerformanceEngine)                               │
│    ├─ In-memory LRU cache for repeated accesses                         │
│    ├─ Thread-safe access with RLock                                     │
│    └─ Automatic eviction by access time (LRU)                           │
│                                                                           │
│  Level 3: Memory-Mapped Files (for large datasets)                       │
│    ├─ MemoryMapManager for files exceeding available RAM                │
│    └─ Lazy loading: only accessed regions loaded                        │
│                                                                           │
│  Cache Invalidation:                                                     │
│    ├─ Q-vector mismatch → full reload                                   │
│    ├─ Config change (dt, frames) → full reload                          │
│    └─ No implicit cache: user controls via cache_file_path              │
└───────────────────────────────────────────────────────────────────────────┘
```

### Cache NPZ Format

```python
# Saved via _save_to_cache() [xpcs_loader.py]
np.savez_compressed(cache_path,
    wavevector_q_list=...,    # (n_q,)
    phi_angles_list=...,      # (n_phi,)
    t1=...,                   # (n_time,)
    t2=...,                   # (n_time,)
    c2_exp=...,               # (n_phi, n_time, n_time)
    # Metadata dict stored as cache_metadata array:
    cache_metadata=...,       # dict with keys:
                              #   config_wavevector_q, actual_wavevector_q,
                              #   q_variance, q_count,
                              #   start_frame, end_frame,
                              #   phi_count, cache_version,
                              #   selective_q_caching
)
# Note: q_vector_hash and dt are NOT stored in the cache NPZ.
```

______________________________________________________________________

## 8. Memory Management

**File:** `data/memory_manager.py` (~1030 lines), `data/optimization.py` (~971 lines)

### AdvancedMemoryManager

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Memory Management [memory_manager.py]                                    │
│                                                                           │
│  Dynamic monitoring and optimization of memory usage during data         │
│  loading and processing                                                  │
│                                                                           │
│  Features:                                                               │
│    ├─ Real-time memory pressure tracking                                │
│    ├─ Adaptive chunk sizing based on available memory                   │
│    ├─ Memory trend analysis (increasing/decreasing/stable)              │
│    └─ Automatic garbage collection triggering                           │
│                                                                           │
│  Memory Thresholds (MemoryPressureMonitor defaults):                     │
│    ├─ Normal:   < 75% system RAM                                        │
│    ├─ Warning:  75-90% system RAM  (warning_threshold=0.75)            │
│    └─ Critical: > 90% system RAM  (critical_threshold=0.9)             │
└───────────────────────────────────────────────────────────────────────────┘
```

### AdvancedDatasetOptimizer

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Dataset Optimization [optimization.py]                                   │
│                                                                           │
│  Size-aware processing strategies:                                       │
│                                                                           │
│  DatasetInfo:                                                            │
│    ├─ total_size_bytes: Estimated memory footprint                      │
│    ├─ n_elements: Total array elements                                  │
│    └─ recommended_strategy: "standard" | "chunked" | "memory_mapped"    │
│                                                                           │
│  ProcessingStrategy selection:                                           │
│    ├─ < 1 GB:   Standard (load all into memory)                         │
│    ├─ 1-4 GB:   Chunked (process in segments)                          │
│    └─ > 4 GB:   Memory-mapped (mmap-based access)                      │
│                                                                           │
│  Adaptive chunk sizing:                                                  │
│    • Initial chunk = available_memory / (3 * element_size)              │
│    • Adjusted based on MemoryManager feedback                           │
│    • Minimum: 1000 elements per chunk                                   │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 9. Result Writing (NLSQ)

**File:** `io/nlsq_writers.py` (~171 lines)

### save_nlsq_json_files()

```python
def save_nlsq_json_files(
    param_dict: dict,         # {name: {value, uncertainty}}
    analysis_dict: dict,      # Method, fit_quality, dataset_info
    convergence_dict: dict,   # Status, iterations, recovery_actions
    output_dir: Path,
) -> None
```

**Writes 3 JSON files:**

```
┌───────────────────────────────────────────────────────────────────────────┐
│ NLSQ JSON Output Files                                                    │
│                                                                           │
│  1. parameters.json                                                      │
│     ├─ timestamp                                                        │
│     ├─ analysis_mode                                                    │
│     ├─ chi_squared, reduced_chi_squared                                 │
│     ├─ convergence_status                                               │
│     └─ parameters: {D0: {value, uncertainty}, alpha: {...}, ...}        │
│                                                                           │
│  2. analysis_results_nlsq.json                                           │
│     ├─ method: "nlsq"                                                   │
│     ├─ fit_quality: {chi_squared, reduced_chi_squared, quality_flag}    │
│     ├─ dataset_info: {n_angles, n_time_points, total_data_points, q}   │
│     └─ optimization_summary: {status, iterations, execution_time}      │
│                                                                           │
│  3. convergence_metrics.json                                             │
│     ├─ convergence: {status, iterations, execution_time, chi_squared}  │
│     ├─ recovery_actions: [...]                                          │
│     ├─ quality_flag                                                     │
│     └─ device_info                                                      │
└───────────────────────────────────────────────────────────────────────────┘
```

### save_nlsq_npz_file()

```python
def save_nlsq_npz_file(
    phi_angles: np.ndarray,              # (n_angles,)
    c2_exp: np.ndarray,                  # (n_angles, n_t1, n_t2)
    c2_raw: np.ndarray,                  # (n_angles, n_t1, n_t2)
    c2_scaled: np.ndarray,               # (n_angles, n_t1, n_t2)
    c2_solver: np.ndarray | None,        # Optional solver surface
    per_angle_scaling: np.ndarray,       # (n_angles, 2) [contrast, offset]
    per_angle_scaling_solver: np.ndarray,# (n_angles, 2)
    residuals: np.ndarray,               # (n_angles, n_t1, n_t2)
    residuals_norm: np.ndarray,          # (n_angles, n_t1, n_t2)
    t1: np.ndarray,                      # (n_t1,)
    t2: np.ndarray,                      # (n_t2,)
    q: float,                            # Wavevector [1/A]
    output_dir: Path,
) -> None
```

**Writes `fitted_data.npz`** with 10-11 compressed arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `phi_angles` | `(n_angles,)` | Scattering angles |
| `c2_exp` | `(n_angles, n_t1, n_t2)` | Experimental correlation data |
| `c2_theoretical_raw` | `(n_angles, n_t1, n_t2)` | Raw unscaled theoretical fits |
| `c2_theoretical_scaled` | `(n_angles, n_t1, n_t2)` | Scaled theoretical fits |
| `c2_solver_scaled` | `(n_angles, n_t1, n_t2)` | Optional solver-evaluated surface |
| `per_angle_scaling` | `(n_angles, 2)` | Final [contrast, offset] per angle |
| `per_angle_scaling_solver` | `(n_angles, 2)` | Original solver values |
| `residuals` | `(n_angles, n_t1, n_t2)` | Experimental - scaled |
| `residuals_normalized` | `(n_angles, n_t1, n_t2)` | Residuals / (0.05 * experimental) |
| `t1` | `(n_t1,)` | Time array 1 (seconds) |
| `t2` | `(n_t2,)` | Time array 2 (seconds) |
| `q` | `(1,)` | Wavevector magnitude [1/A] |

______________________________________________________________________

## 10. Result Writing (CMC)

**Files:** `io/mcmc_writers.py` (~639 lines), `optimization/cmc/io.py` (~430 lines)

### mcmc_writers.py (High-Level Dictionaries)

```python
create_mcmc_parameters_dict(result: CMCResult) -> dict
create_mcmc_analysis_dict(result: CMCResult, data: dict, method_name: str) -> dict
create_mcmc_diagnostics_dict(result: CMCResult) -> dict
```

### CMC JSON Output Files

```
┌───────────────────────────────────────────────────────────────────────────┐
│ CMC JSON Output Files                                                     │
│                                                                           │
│  1. parameters.json                                                      │
│     ├─ timestamp, analysis_mode, method                                 │
│     ├─ sampling_summary: {n_samples, n_warmup, n_chains, total, time}  │
│     ├─ convergence: {all_converged, min/max_r_hat, min_ess, accept_rate}│
│     └─ parameters: {D0: {mean, std}, alpha: {mean, std}, ...}          │
│                                                                           │
│  2. analysis_results_cmc.json                                            │
│     ├─ sampling_quality: {convergence_status, quality_flag}             │
│     │   ├─ warnings: ["R-hat between 1.05-1.1"]                        │
│     │   └─ recommendations: ["Increase n_warmup"]                       │
│     ├─ dataset_info, sampling_summary                                   │
│     └─ parameter_space, initial_values                                  │
│                                                                           │
│  3. diagnostics.json                                                     │
│     ├─ convergence: {r_hat_threshold, ess_threshold}                   │
│     │   └─ per_parameter_diagnostics: [{name, r_hat, ess, converged}]  │
│     ├─ sampling_efficiency: {acceptance_rate, divergences, tree_depth}  │
│     └─ cmc_specific: {shard_summary, combination_method, num_shards}   │
│                                                                           │
│  Quality Thresholds:                                                     │
│     R-hat < 1.05: "good"                                                │
│     R-hat 1.05-1.1: "acceptable" + warning                             │
│     R-hat > 1.1: "poor" + warning                                      │
│     ESS < 400: warning + recommendation                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

### cmc/io.py (Lower-Level CMC I/O)

```python
save_samples_npz(result, output_path)           # Posterior samples
load_samples_npz(input_path) -> dict             # Load samples
samples_to_arviz(samples_data) -> az.InferenceData
save_fitted_data_npz(result, c2_exp, c2_fitted, ...) # Fitted data
save_parameters_json(result, output_path)        # Posterior statistics
save_diagnostics_json(result, output_path, ...)  # Convergence
save_all_results(result, output_dir, ...)        # Orchestrator
```

**samples.npz schema:**

| Array | Shape | Description |
|-------|-------|-------------|
| `posterior_samples` | `(n_chains, n_samples, n_params)` | Raw posterior samples |
| `param_names` | `(n_params,)` | Parameter names in sampling order |
| `r_hat` | `(n_params,)` | Per-parameter R-hat |
| `ess_bulk` | `(n_params,)` | Bulk effective sample size |
| `ess_tail` | `(n_params,)` | Tail effective sample size |
| `divergences` | `(1,)` | Total divergent transitions |
| `analysis_mode` | `(1,)` | "static" or "laminar_flow" |
| `n_phi` | `(1,)` | Number of phi angles |
| `n_chains` | `(1,)` | Number of chains |
| `n_samples` | `(1,)` | Samples per chain |
| `schema_version` | `(2,)` | (1, 0) |

### JSON Serialization Safety

```
┌───────────────────────────────────────────────────────────────────────────┐
│ JSON Safety Layer (io/json_utils.py)                                     │
│                                                                           │
│  json_safe(value) → recursively sanitize:                                │
│    NaN     → None (JSON null)                                            │
│    Inf     → "Infinity" (JSON string)                                    │
│    -Inf    → "-Infinity" (JSON string)                                   │
│    ndarray → list (recursive)                                            │
│    int64   → int                                                         │
│    float64 → float                                                       │
│                                                                           │
│  json_serializer(obj) → default handler for json.dump()                  │
│    Handles: np.ndarray, np.integer, np.floating, Path, datetime          │
│                                                                           │
│  CRITICAL: All writer functions use json_safe() to prevent               │
│  invalid JSON tokens (NaN is not valid JSON)                             │
└───────────────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 11. CLI Orchestration

**File:** `cli/commands.py` (~3361 lines)

### Result Saving Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│ CLI Result Saving Orchestration [cli/commands.py]                        │
│                                                                           │
│  dispatch_command(args)                                                   │
│    ├─ ConfigManager(args.config_file)                                   │
│    ├─ XPCSDataLoader(config_dict=config.config)                         │
│    ├─ data = loader.load_experimental_data()                            │
│    ├─ _apply_angle_filtering_for_optimization(data, config)             │
│    ├─ _exclude_t0_from_analysis(data)                                   │
│    │                                                                     │
│    ├─ NLSQ path:                                                        │
│    │   ├─ result = fit_nlsq_jax(data, config)                          │
│    │   └─ save_nlsq_results(result, data, config, output_dir)          │
│    │       ├─ _extract_nlsq_metadata(config, data)                     │
│    │       │   └─ Extract: L, dt, q (multi-level fallback)             │
│    │       ├─ _prepare_parameter_data(result, mode, n_angles)          │
│    │       │   └─ Per-angle scaling detection + legacy format          │
│    │       ├─ compute_theoretical_fits() → c2_raw, c2_scaled           │
│    │       ├─ save_nlsq_json_files() → 3 JSON files                   │
│    │       ├─ save_nlsq_npz_file() → fitted_data.npz                  │
│    │       └─ generate_nlsq_plots() → PNG heatmaps                    │
│    │                                                                     │
│    └─ CMC path:                                                         │
│        ├─ nlsq_result = fit_nlsq_jax() (warm-start, unless disabled)  │
│        ├─ cmc_result = fit_mcmc_jax(data, config, nlsq_result=...)    │
│        └─ save_mcmc_results(result, data, config, output_dir)          │
│            ├─ create_mcmc_parameters_dict(result) → parameters.json    │
│            ├─ create_mcmc_analysis_dict() → analysis_results_cmc.json  │
│            ├─ create_mcmc_diagnostics_dict() → diagnostics.json        │
│            ├─ _compute_theoretical_c2_from_mcmc()                      │
│            │   └─ Per-angle lstsq fitting from posterior means         │
│            ├─ save_samples_npz() → samples.npz                        │
│            ├─ save_fitted_data_npz() → fitted_data.npz                │
│            └─ generate_nlsq_plots() → PNG heatmaps (reused)          │
└───────────────────────────────────────────────────────────────────────────┘
```

### T=0 Exclusion

```
┌───────────────────────────────────────────────────────────────────────────┐
│ _exclude_t0_from_analysis() [cli/commands.py]                            │
│                                                                           │
│   Physics reason: D(t) = D0 * t^alpha → infinity as t → 0 for alpha < 0 │
│                                                                           │
│   Removes first time point from all arrays:                              │
│     t1[1:], t2[1:], c2_exp[:, 1:, 1:]                                  │
│                                                                           │
│   Applied in CLI after loading, before optimization                      │
└───────────────────────────────────────────────────────────────────────────┘
```

### Metadata Extraction

```
┌───────────────────────────────────────────────────────────────────────────┐
│ _extract_nlsq_metadata(config, data) [cli/commands.py]                   │
│                                                                           │
│   Extracts physics constants with multi-level fallback:                  │
│                                                                           │
│   L (gap length):                                                        │
│     1. config.stator_rotor_gap  (flat attribute; actual access is via   │
│        nested dict:                                                      │
│        config_dict.get("analyzer_parameters", {}).get("geometry", {}))  │
│     2. config.sample_detector_distance                                  │
│     3. Default: 2000000.0 A                                             │
│                                                                           │
│   dt (time step):                                                        │
│     1. config.analyzer_parameters.dt                                    │
│     2. config.experimental_data.dt                                      │
│     3. None (inferred from data)                                        │
│                                                                           │
│   q (wavevector):                                                        │
│     1. data['wavevector_q_list'][0]                                     │
│                                                                           │
│   Returns: {L, dt, q}                                                    │
└───────────────────────────────────────────────────────────────────────────┘
```

### Output Directory Structure

```
homodyne_results/
├── nlsq/
│   ├── parameters.json              # Parameter values + uncertainties
│   ├── analysis_results_nlsq.json   # Fit quality + dataset info
│   ├── convergence_metrics.json     # Convergence diagnostics
│   ├── fitted_data.npz              # Experimental + theoretical arrays
│   └── c2_heatmaps_phi_*.png        # Heatmap plots per angle
│
├── cmc/
│   ├── parameters.json              # Posterior mean +/- std
│   ├── analysis_results_cmc.json    # Sampling quality + diagnostics
│   ├── diagnostics.json             # Convergence metrics
│   ├── samples.npz                  # Posterior samples (ArviZ-compatible)
│   ├── fitted_data.npz              # Experimental + theoretical arrays
│   └── c2_heatmaps_phi_*.png        # Heatmap plots per angle
│
└── homodyne_results.{json|yaml|npz} # Legacy format (backward compat)
```

______________________________________________________________________

## Complete Data Flow

```
YAML Config                          HDF5 Data File
    │                                     │
    ▼                                     ▼
ConfigManager                       XPCSDataLoader
├─ load_config()                    ├─ _detect_format() → "aps_old"|"aps_u"
├─ _normalize_schema()              ├─ _load_aps_old_format()
├─ _normalize_analysis_mode()       │   ├─ Read correlation matrices
└─ _validate_config()               │   └─ Extract q-vectors, phi angles
    │                               ├─ _select_optimal_wavevector()
    ▼                               ├─ _reconstruct_full_matrix()
config.config dict                  ├─ _correct_diagonal_batch()
├─ analyzer_parameters              ├─ _apply_frame_slicing_to_selected_q()
├─ experimental_data                └─ _calculate_time_arrays()
├─ parameter_space                      │
├─ initial_parameters                   ▼
└─ data_filtering               {wavevector_q_list, phi_angles_list,
    │                            t1, t2, c2_exp}
    ▼                                   │
ParameterManager                        ▼
├─ get_parameter_bounds()       _apply_angle_filtering_for_optimization()
├─ get_active_parameters()              │
└─ get_optimizable_parameters()         ▼
    │                           _exclude_t0_from_analysis()
    │                                   │
    └───────────┬───────────────────────┘
                │
                ▼
        NLSQ Path / CMC Path
                │
                ▼
        fit_nlsq_jax() / fit_mcmc_jax()
        ├─ HomodyneModel (for NLSQ)
        ├─ ParameterSpace (for CMC)
        └─ Optimization
                │
                ▼
        OptimizationResult / CMCResult
                │
                ▼
        save_nlsq_results() / save_mcmc_results()
        ├─ _extract_nlsq_metadata() → {L, dt, q}
        ├─ _prepare_parameter_data() → {param: {value, unc}}
        ├─ compute_theoretical_fits() → c2 surfaces
        ├─ save_*_json_files() → JSON
        ├─ save_*_npz_file() → NPZ
        └─ generate_*_plots() → PNG
```

______________________________________________________________________

## Quick Reference Tables

### Data Shapes at Each Stage

| Stage | wavevector_q_list | phi_angles_list | t1 | t2 | c2_exp |
|-------|-------------------|-----------------|----|----|--------|
| Raw loaded | (n_q,) | (n_phi,) | (n_time,) | (n_time,) | (n_phi, n_time, n_time) |
| After phi filter | (n_q,) | (n_selected,) | (n_time,) | (n_time,) | (n_selected, n_time, n_time) |
| After t=0 excl | (n_q,) | (n_selected,) | (n_time-1,) | (n_time-1,) | (n_selected, n_time-1, n_time-1) |

### Configuration Defaults

| Parameter | Section | Default | Description |
|-----------|---------|---------|-------------|
| dt | analyzer_parameters | (required) | Time step in seconds |
| start_frame | analyzer_parameters | 1 | First frame (1-indexed) |
| end_frame | analyzer_parameters | -1 | Last frame (-1 = all) |
| apply_diagonal_correction | experimental_data | true | Diagonal correction |
| phi_range.min | data_filtering | -180 | Minimum phi angle |
| phi_range.max | data_filtering | 180 | Maximum phi angle |
| data_filtering.enabled | data_filtering | true | Enable filtering |

### Error Types

| Exception | Module | Raised When |
|-----------|--------|-------------|
| `XPCSConfigurationError` | data/xpcs_loader | Invalid config structure |
| `XPCSDependencyError` | data/xpcs_loader | Missing numpy/h5py |
| `XPCSDataFormatError` | data/xpcs_loader | Unrecognized HDF5 format |
| `PreprocessingError` | data/preprocessing | Preprocessing stage failure |
| `PreprocessingConfigurationError` | data/preprocessing | Invalid preprocessing config |
| `ValueError` | config/manager | Invalid config values |
| `OSError` | io/nlsq_writers | File write failure |

### JSON Output Summary

| File | Method | Size | Key Contents |
|------|--------|------|-------------|
| parameters.json (NLSQ) | save_nlsq_json_files | ~2 KB | {value, uncertainty} per param |
| parameters.json (CMC) | create_mcmc_parameters_dict | ~3 KB | {mean, std} per param |
| analysis_results_*.json | save_nlsq/create_mcmc_analysis | ~3 KB | fit_quality, dataset_info |
| convergence_metrics.json | save_nlsq_json_files | ~2 KB | convergence status, recovery |
| diagnostics.json (CMC) | create_mcmc_diagnostics_dict | ~5 KB | per-param R-hat, ESS, shards |
| fitted_data.npz | save_*_npz_file | 50 KB-500 MB | exp + theoretical + residuals |
| samples.npz (CMC) | save_samples_npz | 1-100 MB | posterior (chains x samples x params) |

______________________________________________________________________

## Key Files Reference

### Data Loading (`homodyne/data/`)

| File | Lines | Purpose |
|------|-------|---------|
| **xpcs_loader.py** | ~2107 | Main loader: HDF5 reading, format detection, caching, filtering |
| **config.py** | ~752 | YAML/JSON config loading and schema validation |
| **filtering_utils.py** | ~613 | Q-range, phi, quality, and frame-based filtering |
| **preprocessing.py** | ~1153 | Multi-stage preprocessing pipeline with provenance |
| **quality_controller.py** | ~1646 | Progressive quality control with auto-repair |
| **validation.py** | ~1115 | Data quality validation (NaN, shape, range checks) |
| **performance_engine.py** | ~1502 | Multi-level caching, LRU eviction, thread-safe access |
| **memory_manager.py** | ~1030 | Dynamic memory monitoring and pressure management |
| **optimization.py** | ~971 | Size-aware processing strategies (standard/chunked/mmap) |
| **angle_filtering.py** | ~413 | Angle normalization and filtering utilities |
| **phi_filtering.py** | ~385 | Vectorized phi angle filtering |
| **validators.py** | ~296 | Input validation at I/O boundaries |
| **types.py** | ~44 | Shared data types (prevents circular imports) |

### Configuration (`homodyne/config/`)

| File | Lines | Purpose |
|------|-------|---------|
| **manager.py** | ~1296 | ConfigManager: YAML loading, section access, CMC config |
| **parameter_space.py** | ~895 | ParameterSpace: bounds + priors for MCMC |
| **parameter_manager.py** | ~809 | ParameterManager: centralized bounds and validation |
| **parameter_registry.py** | ~632 | ParameterRegistry: singleton for parameter metadata |
| **types.py** | ~522 | TypedDict definitions for config structures |
| **parameter_names.py** | ~315 | Parameter name constants and mappings |
| **physics_validators.py** | ~286 | Physics constraint validation |

### Result Writing (`homodyne/io/`)

| File | Lines | Purpose |
|------|-------|---------|
| **mcmc_writers.py** | ~639 | CMC result dict creation (parameters, analysis, diagnostics) |
| **nlsq_writers.py** | ~171 | NLSQ result saving (3 JSON + 1 NPZ) |
| **json_utils.py** | ~114 | NaN/Inf-safe JSON serialization |

### CMC-Specific I/O (`homodyne/optimization/cmc/`)

| File | Lines | Purpose |
|------|-------|---------|
| **io.py** | ~430 | CMC samples NPZ, fitted data NPZ, save_all_results orchestrator |
