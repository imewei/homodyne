Configuration Options Reference
===============================

This section documents all available configuration options for Homodyne YAML
configuration files, organized by section and with physical meaning and
typical values for each parameter.

Configuration Structure
-----------------------

A complete Homodyne configuration includes the following main sections:

1. **Metadata** - Version and description
2. **Analysis Mode** - Static or laminar_flow
3. **Analyzer Parameters** - Experimental settings
4. **Analysis Settings** - Mode-specific configuration
5. **Experimental Data** - Data input and caching
6. **Phi Angle Filtering** - Scattering angle selection
7. **Initial Parameters** - Starting values and bounds
8. **Parameter Space** - Bounds and priors
9. **Optimization** - Method selection and settings
10. **Noise Estimation** - Optional automatic noise estimation
11. **Performance** - JAX and memory optimization
12. **Logging** - Logging levels and output
13. **Plotting** - Visualization configuration
14. **Output** - File formats and directories
15. **Validation** - Quality control checks
16. **Quality Control** - Advanced validation settings

Metadata Section
----------------

Configuration metadata and versioning:

.. code-block:: yaml

    metadata:
      config_version: "2.4.3"           # Configuration format version
      description: "..."                 # Human-readable description
      analysis_mode: "static"            # "static" or "laminar_flow"
      parameter_count: 3                 # Physical parameters only

      physics_model: "..."               # Mathematical model (documentation)
      integration_method: "discrete_numerical"  # Integration method
      diagonal_correction: "mandatory"   # C2 diagonal correction

      recommended_use: "..."             # Recommended system type
      template_type: "production_ready"  # Template classification
      complexity: "comprehensive"        # Feature completeness

      generated_at: null                 # ISO timestamp (auto-filled)
      generated_by: null                 # Generator tool (auto-filled)

Analysis Mode
^^^^^^^^^^^^^

.. code-block:: yaml

    analysis_mode: "static"              # Options: "static" | "laminar_flow"

**Static**: Equilibrium systems with 3 physical parameters (D0, alpha, D_offset)

**Laminar Flow**: Non-equilibrium systems under shear with 7 physical parameters

Analyzer Parameters Section
---------------------------

Experimental and instrumental parameters:

.. code-block:: yaml

    analyzer_parameters:
      dt: 0.1                           # Time step [seconds]
      start_frame: 1000                 # Starting frame (1-indexed)
      end_frame: 2000                   # Ending frame (inclusive)

      scattering:
        wavevector_q: 0.0054            # Wave vector magnitude [Å⁻¹]

      geometry:
        stator_rotor_gap: 2000000       # Stator-rotor gap [Å]

**dt**: Time interval between correlation measurements

- Typical range: 0.01 - 10 seconds
- Determines correlation time scale
- Must match your experimental setup

**start_frame / end_frame**: Data range to analyze

- 1-indexed frame numbers from HDF5 file
- Inclusive of end_frame
- Common to skip early/late frames for data quality

**wavevector_q**: Scattering wave vector magnitude

- Typical range: 0.001 - 0.1 Å⁻¹
- Determines spatial resolution of measurement
- Sample and beamline dependent

**stator_rotor_gap**: For rheological measurements

- Units: Ångströms
- Example: 200 μm = 2,000,000 Å
- Used for shear rate calculations in laminar flow mode

Experimental Data Section
--------------------------

Data input and caching configuration:

.. code-block:: yaml

    experimental_data:
      # Primary data file (HDF5 format)
      file_path: "./data/experiment.hdf"

      # Legacy format (both supported)
      data_folder_path: "./data/"
      data_file_name: "experiment.hdf"

      # Phi angles
      phi_angles_path: "./data/"
      phi_angles_file: "phi_angles_list.txt"

      # Caching for performance
      cache_file_path: "./data/"
      cache_filename_template: "cached_c2_static_q{wavevector_q:.4f}_frames_{start_frame}_{end_frame}.npz"
      cache_compression: true

      # Data format
      data_type: "float64"              # "float32" or "float64"
      file_format: "HDF5"               # Currently only HDF5 supported
      exchange_key: "exchange"          # HDF5 group key

**file_path**: Direct path to HDF5 data file (preferred)

**data_folder_path / data_file_name**: Legacy format

- Both specify same HDF5 file
- Can use either method
- file_path takes precedence if both specified

**phi_angles_path / phi_angles_file**: Scattering angles

- Text file with one angle (degrees) per line
- Example content: 0.0, 45.0, 90.0, 135.0, 180.0
- -180° to 180° range (wrapping supported)

**cache_filename_template**: Caching configuration

- Supports {variable} substitution: wavevector_q, start_frame, end_frame
- .npz format (NumPy compressed)
- Automatic detection and reuse if file exists

**data_type**: Numerical precision

- "float32": Faster, lower memory (suitable for most applications)
- "float64": Higher precision (use if numerical precision critical)
- MCMC benefits from float32 for faster computation

**exchange_key**: HDF5 group name for raw scattering data

Phi Angle Filtering Section
---------------------------

Scattering angle selection and quality control:

.. code-block:: yaml

    phi_filtering:
      enabled: true                     # Enable/disable filtering

      target_ranges:
        - min_angle: -10.0              # Minimum angle [degrees]
          max_angle: 10.0               # Maximum angle [degrees]
          description: "Parallel to primary axis"

        - min_angle: 170.0
          max_angle: -170.0             # Wraps: 170° to 190°
          description: "Antiparallel"

      fallback_to_all_angles: true      # Use all if no matches
      algorithm: "range_based"          # Currently only option
      tolerance: 3.0                    # Angular tolerance [degrees]

      quality_control:
        min_angles_required: 1          # Minimum angles needed
        max_angle_spread: 36.0          # Max spread within range [degrees]
        validate_coverage: true         # Validate angle coverage
        require_orthogonal_angles: false # Required only for anisotropy

**target_ranges**: Angle selection rules

- Each range defines an angular window (in degrees)
- Supports wrapping: 170° to -170° = 170° to 190°
- Multiple ranges allow complex angle selections

**fallback_to_all_angles**: Behavior when no angles match

- true: Use all available angles
- false: Raise error if no matches

**tolerance**: Tolerance for angle matching

- Angles within ± tolerance of range are included
- Typical: 2-5 degrees

**Static Mode**: Usually 1-3 angles (0°, 90°, 180°)

**Laminar Flow Mode**: Usually 3+ angles (0°, 90°, 180°)

Initial Parameters Section
--------------------------

Starting values and per-angle scaling:

.. code-block:: yaml

    initial_parameters:
      parameter_names:
        - D0                            # Name of first parameter
        - alpha
        - D_offset

      values: null                      # Explicit starting values
      # Example: [1000.0, -1.2, 0.0]

      per_angle_scaling:
        contrast: null                  # Per-angle contrast
        offset: null                    # Per-angle offset
        # Example: contrast: [0.05, 0.06, 0.05]

      units:
        - "Å²/s"                        # Units for documentation
        - "dimensionless"
        - "Å²/s"

      active_parameters: null           # Subset to optimize
      fixed_parameters: null            # Parameters to fix

**parameter_names**: Parameter order for optimization

**Static Mode** (3 parameters):

- D0: Diffusion prefactor (Å²/s)
- alpha: Anomalous exponent (dimensionless)
- D_offset: Baseline diffusion (Å²/s)

**Laminar Flow Mode** (7 parameters):

- D0, alpha, D_offset (same as static)
- gamma_dot_t0: Shear rate prefactor (s⁻¹)
- beta: Shear rate exponent (dimensionless)
- gamma_dot_t_offset: Baseline shear rate (s⁻¹)
- phi0: Flow angle (degrees)

**values**: Initial parameter values

- null: Use midpoint of bounds
- List of floats: Explicit starting values
- For MCMC: Copy NLSQ results for better convergence

**per_angle_scaling** (mandatory in v2.4.0+):

- contrast: List of scaling factors (one per angle)
- offset: List of baseline shifts (one per angle)
- Must have N values for N filtered angles
- Copy from NLSQ results when initializing MCMC

**active_parameters**: Optimize subset of parameters

- null: Optimize all parameters
- List of parameter names: Optimize only these
- Others held at initial values

**fixed_parameters**: Hold specific parameters constant

- null: No fixed parameters
- Dict: {param_name: fixed_value}

Parameter Space Section
-----------------------

Parameter bounds and prior distributions:

.. code-block:: yaml

    parameter_space:
      model: "static"                   # or "laminar_flow"

      bounds:
        - name: D0
          min: 100.0                    # Minimum bound
          max: 1e5                      # Maximum bound
          type: TruncatedNormal         # Prior type
          prior_mu: 1000.0              # Prior mean
          prior_sigma: 1000.0           # Prior std dev
          unit: "Å²/s"

Static Mode Bounds
^^^^^^^^^^^^^^^^^^

**D0** (Diffusion prefactor):

- min: 100.0
- max: 100,000 (1e5)
- typical: 500 - 10,000 Å²/s
- unit: Å²/s

**alpha** (Anomalous exponent):

- min: -2.0
- max: 2.0
- typical: -2 to 2
- unit: dimensionless
- Physical meaning: 0=normal, <0=subdiffusion, >0=superdiffusion

**D_offset** (Baseline diffusion):

- min: -100,000
- max: 100,000
- typical: -100 to 100 Å²/s
- unit: Å²/s
- Can be negative (mathematical baseline)

Laminar Flow Mode Bounds
^^^^^^^^^^^^^^^^^^^^^^^^

Same as static + 4 additional parameters:

**gamma_dot_t0** (Shear rate prefactor):

- min: 0.001
- max: 1000.0
- typical: 0.01 - 100 s⁻¹
- unit: s⁻¹
- Physical meaning: Shear rate scale

**beta** (Shear rate exponent):

- min: -2.0
- max: 2.0
- typical: -1 to 1
- unit: dimensionless
- Physical meaning: Time-dependence of shear

**gamma_dot_t_offset** (Baseline shear rate):

- min: -1000.0
- max: 1000.0
- typical: -10 to 10 s⁻¹
- unit: s⁻¹

**phi0** (Flow angle):

- min: -180.0
- max: 180.0
- typical: any angle
- unit: degrees
- Physical meaning: Flow direction relative to measurement

**type**: Prior distribution type

- TruncatedNormal: Bounds enforced, Gaussian prior within bounds
- Most common choice for MCMC

**prior_mu / prior_sigma**: Prior distribution parameters

- mu: Prior mean (center of belief)
- sigma: Prior standard deviation (width of belief)
- Used in MCMC inference
- Bounds [min, max] enforce hard constraints

Optimization Section
--------------------

Optimization method and algorithm configuration:

.. code-block:: yaml

    optimization:
      method: "nlsq"                    # "nlsq" or "mcmc"

NLSQ Configuration
^^^^^^^^^^^^^^^^^^

Fast, deterministic optimization using trust-region method:

.. code-block:: yaml

    nlsq:
      max_iterations: 100               # Maximum optimization iterations
      tolerance: 1e-8                   # Convergence tolerance
      trust_region_scale: 1.0           # Trust region scaling (0.1-10.0)
      verbose: false                    # Print iteration details
      x_scale_map: null                 # Per-parameter scaling

**max_iterations**: Upper limit on optimization iterations

- Typical: 50-200
- Algorithm stops when converged or reaches this limit

**tolerance**: Convergence criterion

- Typical: 1e-6 to 1e-10
- Smaller = more iterations but better convergence

**trust_region_scale**: Trust region size control

- 0.1-1.0: Smaller steps (conservative)
- 1.0: Default (balanced)
- 1.0-10.0: Larger steps (aggressive)

**x_scale_map**: Per-parameter gradient scaling

- null: Uniform scaling
- Dict: Parameter-specific scaling
- Useful when parameters have very different gradient magnitudes

Example for laminar flow:

.. code-block:: yaml

    x_scale_map:
      D0: 1.0
      alpha: 0.01                      # 100x smaller gradient
      D_offset: 1.0
      gamma_dot_t0: 0.01
      beta: 0.1
      gamma_dot_t_offset: 0.01
      phi0: 0.1

Stratification Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Automatic angle-aware chunking for large datasets:

.. code-block:: yaml

    stratification:
      enabled: "auto"                   # "auto" | true | false
      target_chunk_size: 100000         # Target chunk size [points]
      max_imbalance_ratio: 5.0          # Max angle imbalance ratio
      force_sequential_fallback: false  # Force per-angle optimization
      check_memory_safety: true         # Check available memory
      use_index_based: false            # Zero-copy indexing
      collect_diagnostics: false        # Collect performance metrics
      log_diagnostics: false            # Log diagnostic report

**enabled: "auto"** activates stratification when:
- per_angle_scaling=True AND
- dataset size ≥ 100,000 points

**target_chunk_size**: Preferred chunk size

- Matches NLSQ internal chunking
- Typical: 100,000 - 1,000,000 points

**max_imbalance_ratio**: Fallback trigger

- If max_angles / min_angles > threshold
- Falls back to sequential (per-angle) optimization

**use_index_based**: Memory optimization for very large datasets

- false: Standard approach (2x temporary memory overhead)
- true: Zero-copy indexing (~1% memory overhead)

Sequential Optimization
^^^^^^^^^^^^^^^^^^^^^^^

Fallback strategy for per-angle optimization:

.. code-block:: yaml

    sequential:
      min_success_rate: 0.5             # Min converged angles
      weighting: "inverse_variance"     # Combination method

**min_success_rate**: Minimum converged angles

- 0.5: ≥ 50% of angles must converge
- Higher = stricter requirement
- Fails if fewer angles converge

**weighting**: How to combine per-angle results

- "inverse_variance": Weight by fit quality (recommended)
- "uniform": Equal weight
- "n_points": Weight by number of points per angle

MCMC Configuration
^^^^^^^^^^^^^^^^^^

Bayesian inference with Consensus Monte Carlo (CMC):

.. code-block:: yaml

    mcmc:
      backend: "numpyro"                # "numpyro" | "blackjax"
      num_warmup: 2000                  # Warmup samples
      num_samples: 5000                 # Posterior samples
      num_chains: 4                     # Parallel chains
      progress_bar: true                # Show progress
      target_accept_prob: 0.90          # NUTS target acceptance
      max_tree_depth: 12                # Maximum NUTS tree depth
      dense_mass_matrix: true           # Full covariance mass matrix

**backend**: MCMC sampling library

- "numpyro": NumPyro with NUTS sampler (recommended)
- "blackjax": BlackJAX backend (alternative)

**num_warmup**: Adaptation/warmup samples

- Typical: 1000-5000
- Larger for complex posteriors

**num_samples**: Posterior samples per chain

- Typical: 5000-10000
- Larger for higher precision uncertainty estimates

**num_chains**: Independent parallel chains

- Typical: 2-4
- Helps diagnose convergence with R-hat

**target_accept_prob**: NUTS target acceptance probability

- 0.80-0.95 range typical
- 0.90 is good default

**max_tree_depth**: NUTS tree depth limit

- Prevents runaway tree growth
- Typical: 10-12

**dense_mass_matrix**: Mass matrix adaptation

- true: Full covariance (recommended)
- false: Diagonal covariance (faster)

Sharding Configuration (CMC)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data sharding for parallel subposterior computation:

.. code-block:: yaml

    sharding:
      strategy: "stratified"            # "stratified" | "random" | "contiguous"
      num_shards: "auto"                # Number of data shards
      max_points_per_shard: "auto"      # Max points per shard
      seed_base: 0                      # RNG seed base

**strategy**: Shard creation method

- "stratified": Angle-aware sharding (ensures all angles in each shard)
- "random": Random assignment
- "contiguous": Sequential assignment

**num_shards**: Number of parallel MCMC chains

- "auto": Heuristic based on data size
- int: Explicit number of shards

**max_points_per_shard**: Points per shard

- "auto": 10,000-100,000 typically
- Larger = faster inference, fewer shards

Initial Values Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-angle initialization for MCMC:

.. code-block:: yaml

    initial_values:
      phi: {}                           # Per-phi overrides
      percentile_fallback:
        contrast_low_pct: 5             # Low percentile for contrast
        contrast_high_pct: 95           # High percentile for contrast
        offset_pct: 50                  # Median for offset

Backend Configuration
^^^^^^^^^^^^^^^^^^^^^

Parallel execution backend:

.. code-block:: yaml

    backend_config:
      name: "auto"                      # "auto" | "pjit" | "multiprocessing" | "pbs"
      enable_checkpoints: true          # Enable checkpoint/resume
      checkpoint_frequency: 1           # Save checkpoint every N shards
      checkpoint_dir: "./cmc_checkpoints"
      keep_last_checkpoints: 3          # Keep last N checkpoints
      resume_from_checkpoint: true      # Auto-resume

**name**: Backend for parallel execution

- "auto": Autodetect best available
- "multiprocessing": Multicore CPU (v2.3.0)
- "pjit": JAX pjit (for multi-GPU, not available in v2.3.0)
- "pbs": PBS HPC submission
- "slurm": SLURM HPC submission

Subposterior Combination
^^^^^^^^^^^^^^^^^^^^^^^^

CMC posterior combination method:

.. code-block:: yaml

    combination:
      method: "weighted_gaussian"       # Combination method
      validate_results: true            # Validate combined posterior
      min_success_rate: 0.8             # Min successful shards

**method**: How to combine subposteriors

- "weighted_gaussian": Weight by variance (recommended)
- "simple_average": Unweighted average
- "auto": Choose automatically

**min_success_rate**: Minimum shard convergence

- 0.8: ≥ 80% of shards must converge
- Optimization fails if fewer converge

Per-Shard MCMC
^^^^^^^^^^^^^^

MCMC settings for each shard:

.. code-block:: yaml

    per_shard_mcmc:
      num_warmup: 500                   # Warmup per shard
      num_samples: 1000                 # Samples per shard
      num_chains: 2                     # Chains per shard
      subsample_size: "auto"            # Subsampling size

Convergence Validation
^^^^^^^^^^^^^^^^^^^^^^

MCMC convergence diagnostics:

.. code-block:: yaml

    validation:
      strict_mode: false                # Fail if criteria not met
      min_per_shard_ess: 100            # Minimum ESS per parameter
      max_per_shard_rhat: 1.2           # Maximum R-hat (convergence)
      max_between_shard_kl: 0.5         # Max KL divergence between shards
      min_success_rate: 0.8             # Min successful shards

**strict_mode**: Error handling

- false: Warnings only (allows imperfect convergence)
- true: Fail if criteria not met

**min_per_shard_ess**: Effective sample size

- Minimum effective samples per parameter per shard
- ESS < 100 means high autocorrelation

**max_per_shard_rhat**: Potential scale reduction factor

- < 1.01: Excellent convergence
- < 1.05: Good convergence
- 1.05-1.10: Acceptable (needs more samples)
- > 1.10: Poor convergence (run longer)

**max_between_shard_kl**: Subposterior consistency

- KL divergence between shard posteriors
- Smaller = more consistent subposteriors

Performance Section
-------------------

JAX and memory optimization:

.. code-block:: yaml

    performance:
      strategy_override: null           # Force strategy: "standard" | "large" | "chunked" | "streaming"
      memory_limit_gb: null             # Custom memory limit [GB]
      enable_progress: true             # Show progress bars

      memory_optimization:
        enabled: true
        max_memory_usage_gb: 6.0        # Max memory [GB]
        chunk_size: 8000                # Chunk size [points]
        enable_caching: true            # Enable caching
        cache_strategy: "adaptive"      # "adaptive" | "aggressive" | "conservative"

      computation:
        enable_jit: true                # Enable JAX JIT compilation
        cpu_threads: "auto"             # Number of CPU threads
        vectorization_level: "high"     # "low" | "medium" | "high"

**strategy_override**: Force specific NLSQ strategy

- null: Automatic selection based on data size
- "standard": curve_fit (< 1M points)
- "large": curve_fit_large (1M-10M points)
- "chunked": Chunked large (10M-100M points)
- "streaming": Streaming (> 100M points)

**max_memory_usage_gb**: Memory limit for computation

- Default: auto-detect available RAM
- Set lower to prevent OOM

**chunk_size**: Size of data chunks for processing

- Typical: 8,000-100,000 points
- Balance between memory and efficiency

**enable_jit**: JAX Just-In-Time compilation

- true: Enable JIT (faster)
- false: Disable JIT (easier debugging)

**cpu_threads**: CPU thread count for JAX

- "auto": Use all available cores
- int: Explicit thread count
- Affects parallelization

Logging Section
---------------

Logging configuration:

.. code-block:: yaml

    logging:
      enabled: true                     # Enable logging
      level: "INFO"                     # Global log level

      console:
        enabled: true                   # Log to console
        level: "INFO"                   # Console log level
        format: "detailed"              # "simple" | "detailed"
        colors: true                    # Colored output
        show_progress: true             # Progress indicators

      file:
        enabled: false                  # Log to file
        level: "DEBUG"                  # File log level
        path: "./logs/"                 # Log directory
        filename: "homodyne_analysis.log"
        max_size_mb: 10                 # Max file size before rotation
        backup_count: 5                 # Backup files to keep

      modules:
        "homodyne.data.phi_filtering": "INFO"
        "jax._src": "WARNING"           # Suppress JAX internals

**level**: Logging verbosity

- "DEBUG": Detailed (for development)
- "INFO": Standard information
- "WARNING": Warnings and errors only
- "ERROR": Errors only

**format**: Console output style

- "simple": Minimal formatting
- "detailed": Full timestamps and module names

**Module-specific**: Control logging per module

Output Section
--------------

Result file output configuration:

.. code-block:: yaml

    output:
      directory: "./results"            # Output directory
      base_directory: "./homodyne_results/"

      formats:
        hdf5: true                      # HDF5 format
        json: true                      # JSON format
        csv: true                       # CSV format

      create_subdirs: true              # Create method subdirs (nlsq/, mcmc/)
      timestamp_dirs: false             # Add timestamp to directory names

      compress_hdf5: true               # HDF5 compression
      compression_level: 6              # Compression: 0-9 (higher = more)

Plotting Section
----------------

Visualization configuration:

.. code-block:: yaml

    plotting:
      save_plots: true                  # Save plots to files
      show_plots: false                 # Display interactive plots
      format: "png"                     # File format: "png" | "pdf" | "svg"
      dpi: 300                          # Resolution [dots per inch]
      style: "publication"              # Matplotlib style

      preview_mode: false               # Preview: fast Datashader | publication: matplotlib
      fit_surface: "solver"             # "solver" | "posthoc"

      color_scale:
        mode: "legacy"
        pin_legacy_range: true
        percentile_min: 1.0
        percentile_max: 99.0
        fixed_min: 1.0
        fixed_max: 1.5

      datashader:
        canvas_width: 1200              # Resolution [pixels]
        canvas_height: 1200

      matplotlib:
        interpolation: "bilinear"       # Interpolation: "none" | "bilinear" | "bicubic"
        use_tight_layout: true

Validation Section
------------------

Configuration validation options:

.. code-block:: yaml

    validation:
      strict_mode: false                # Fail on warnings
      check_file_existence: true        # Verify files exist
      validate_parameter_ranges: true   # Check bounds
      check_mode_compatibility: true    # Mode validation

      angle_validation:
        require_multiple_angles: false  # (for static)
        min_angle_count: 1              # Minimum angles
        validate_angle_ranges: true     # Range checks

Quick Reference Table
---------------------

Common Configuration Patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. table:: Quick Configuration Reference
   :widths: 40 60

   ==========================  ===============================================
   Purpose                     Configuration
   ==========================  ===============================================
   Single static optimization  analysis_mode: "static", method: "nlsq"
   Static with MCMC            analysis_mode: "static", method: "mcmc"
   Laminar flow analysis       analysis_mode: "laminar_flow", method: "nlsq"
   Large dataset (>10M)        strategy_override: "large" or "chunked"
   Very large dataset (>100M)  strategy_override: "streaming"
   Fast preview                preview_mode: true, enable_progress: true
   Publication quality         preview_mode: false, dpi: 300, format: "pdf"
   Low memory                  max_memory_usage_gb: 2.0, chunk_size: 4000
   High precision MCMC         num_samples: 10000, num_chains: 8
   ==========================  ===============================================

For Examples
------------

See :doc:`templates` for complete configuration examples with all options
explained and documented.

For Validation
--------------

Use the configuration validator to check your configuration:

.. code-block:: bash

    homodyne-config --validate my_config.yaml
