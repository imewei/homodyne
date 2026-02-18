.. _configuration:

YAML Configuration Reference
=============================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The complete structure of a homodyne YAML configuration file
- All configuration sections and their options
- Default values for every setting
- How to generate and validate configuration templates
- Configuration precedence rules

---

Configuration Overview
-----------------------

All homodyne analysis is driven by a YAML configuration file. The ``ConfigManager``
class loads, validates, and provides access to all settings:

.. code-block:: python

   from homodyne.config import ConfigManager

   config = ConfigManager.from_yaml("config.yaml")

   # Access configuration sections
   print(config.analysis_mode)              # "static" or "laminar_flow"
   print(config.get_initial_parameters())  # dict of initial param values
   print(config.get_cmc_config())           # CMC settings dict

Generate a template with the CLI:

.. code-block:: bash

   # Static mode template
   homodyne-config --mode static --output config_static.yaml

   # Laminar flow template
   homodyne-config --mode laminar_flow --output config_flow.yaml

   # Validate an existing config
   homodyne-config --validate --input my_config.yaml

---

Complete Configuration Schema
-------------------------------

Below is a fully annotated configuration with all available options and their
defaults:

.. code-block:: yaml

   # ============================================================
   # DATA SECTION
   # ============================================================
   data:
     # Path to the HDF5 file containing XPCS data
     file_path: "data.h5"                # Required

     # Internal HDF5 path to the C2 array (n_phi, n_t1, n_t2)
     dataset_path: "/exchange/data"       # Default: "/exchange/data"

     # Scattering vector magnitude in Å⁻¹
     q_value: 0.054                       # Required

     # Stator-rotor gap distance in µm (only for laminar_flow)
     gap_distance: 500.0                  # Default: null (required for laminar_flow)

     # Frame time interval in seconds
     dt: 0.1                              # Default: inferred from time arrays

     # Optional: restrict angle range (degrees)
     phi_range: null                      # Default: null (load all angles)
     # phi_range: [-90, 90]               # Example: load only ±90°

     # Optional: restrict time range (seconds)
     t_range: null                        # Default: null (load all times)

   # ============================================================
   # ANALYSIS SECTION
   # ============================================================
   analysis:
     mode: "static"                       # "static" or "laminar_flow"

   # ============================================================
   # OPTIMIZATION SECTION
   # ============================================================
   optimization:
     method: "nlsq"                       # "nlsq" or "cmc" or "both"

     # -------- NLSQ Settings --------
     nlsq:
       # Memory management
       memory_fraction: 0.75             # Trigger streaming above 75% RAM
       # memory_threshold_gb: null        # Or explicit GB limit

       # Anti-degeneracy (per-angle scaling)
       anti_degeneracy:
         per_angle_mode: "auto"           # "auto", "constant", "individual", "fourier"
         constant_scaling_threshold: 3   # Angle count threshold for "auto" mode

       # CMA-ES global optimization (optional)
       cmaes:
         enable: false                   # Set true for multi-scale problems
         preset: "cmaes-global"          # "cmaes-fast", "cmaes-global", "cmaes-hpc"
         refine_with_nlsq: true          # Run NLSQ after CMA-ES

       # Hybrid recovery (when standard NLSQ fails)
       hybrid_recovery:
         enable: true                    # Enable 3-attempt retry strategy
         max_attempts: 3

     # -------- CMC Settings --------
     cmc:
       # Sharding (data partitioning)
       sharding:
         max_points_per_shard: "auto"    # Always use "auto"
         sharding_strategy: "stratified" # "stratified", "random", "contiguous"
         max_shards: null                # null = dynamic based on dataset size

       # Per-shard MCMC settings
       per_shard_mcmc:
         num_warmup: 500                 # Warmup/burn-in samples per chain
         num_samples: 1500              # Posterior samples per chain
         num_chains: 4                  # NUTS chains per shard
         target_accept_prob: 0.8        # NUTS target acceptance probability
         max_tree_depth: 10             # Max NUTS tree depth (2^10 leapfrog steps)
         chain_method: "parallel"       # "parallel" (default), "vectorized", "sequential"
         adaptive_sampling: true        # Reduce samples for small shards

         # JAX profiling (advanced)
         enable_jax_profiling: false
         jax_profile_dir: "./profiles/jax"

       # Per-angle mode (must match nlsq.anti_degeneracy.per_angle_mode)
       per_angle_mode: "auto"
       constant_scaling_threshold: 3

       # Quality filtering
       validation:
         max_divergence_rate: 0.10      # Reject shards with > 10% divergences
         require_nlsq_warmstart: false  # True = fail if no NLSQ warm-start

       # Execution backend
       backend_name: "auto"            # "auto", "multiprocessing", "pjit", "pbs"
       num_workers: null               # null = physical_cores/2 - 1

       # Posterior combination
       combination_method: "consensus_mc"  # Recommended
       # "consensus_mc": precision-weighted means (correct)
       # "weighted_gaussian": legacy element-wise (deprecated)
       # "simple_average": unweighted (deprecated)

       # Convergence criteria
       max_r_hat: 1.05                 # Maximum R-hat for accepted shards
       min_ess: 400                    # Minimum effective sample size

       # Checkpointing
       enable_checkpoints: false
       checkpoint_dir: "./checkpoints"

   # ============================================================
   # PARAMETER SPACE SECTION
   # ============================================================
   parameter_space:
     # Format for each parameter:
     # <name>:
     #   initial: <float>        # Initial value for optimization
     #   bounds: [<low>, <high>] # Hard bounds
     #   prior: "uniform"        # "uniform" or "normal" (CMC)
     #   prior_mean: null        # For normal prior (CMC)
     #   prior_std: null         # For normal prior (CMC)

     # --- Diffusion parameters (static + laminar_flow) ---
     D0:
       initial: 1.0
       bounds: [0.001, 1.0e6]
       prior: "uniform"

     alpha:
       initial: -0.5
       bounds: [-2.0, 2.0]
       prior: "uniform"

     D_offset:
       initial: 0.01
       bounds: [0.0, 1.0e4]
       prior: "uniform"

     # --- Shear parameters (laminar_flow only) ---
     gamma_dot_0:
       initial: 0.01
       bounds: [1.0e-6, 1.0e4]
       prior: "uniform"

     beta:
       initial: 0.0
       bounds: [-2.0, 2.0]
       prior: "uniform"

     gamma_dot_offset:
       initial: 0.001
       bounds: [0.0, 1.0e3]
       prior: "uniform"

     phi_0:
       initial: 0.0
       bounds: [-180.0, 180.0]
       prior: "uniform"

   # ============================================================
   # OUTPUT SECTION
   # ============================================================
   output:
     directory: "./results"             # Output directory
     save_arrays: true                  # Save NPZ arrays
     save_json: true                    # Save JSON summary
     save_plots: false                  # Save matplotlib plots
     run_id: null                       # null = auto-generated timestamp

     # Output formats
     formats:
       hdf5: true                       # Save results as HDF5
       json: true                       # Save summary as JSON
       csv: false                       # Save parameter table as CSV

     # Compression
     compress_hdf5: true                # Enable HDF5 compression
     compression_level: 6               # gzip level 0-9 (higher = smaller)

   # ============================================================
   # STRATIFICATION SECTION (NLSQ)
   # ============================================================
   # Angle-aware chunking for large datasets
   stratification:
     enabled: "auto"                    # "auto" | true | false
     target_chunk_size: 100000          # Target chunk size [points]
     max_imbalance_ratio: 5.0           # Max angle imbalance before fallback
     force_sequential_fallback: false   # Force per-angle optimization
     check_memory_safety: true          # Check available memory
     use_index_based: false             # Zero-copy indexing (low memory)
     collect_diagnostics: false         # Collect performance metrics
     log_diagnostics: false             # Log diagnostic report

   # ============================================================
   # SEQUENTIAL SECTION (NLSQ fallback)
   # ============================================================
   # Per-angle optimization fallback strategy
   sequential:
     min_success_rate: 0.5              # Min fraction of angles converged
     weighting: "inverse_variance"      # "inverse_variance" | "uniform" | "n_points"

   # ============================================================
   # PHI FILTERING SECTION
   # ============================================================
   phi_filtering:
     enabled: true                      # Enable angle filtering
     target_ranges:
       - min_angle: -10.0               # Angular window [degrees]
         max_angle: 10.0
         description: "Parallel to flow"
     fallback_to_all_angles: true       # Use all angles if none match
     algorithm: "range_based"           # Currently the only option
     tolerance: 3.0                     # Angular tolerance [degrees]

     quality_control:
       min_angles_required: 1           # Minimum angles needed
       max_angle_spread: 36.0           # Max spread within range [deg]
       validate_coverage: true          # Validate angle coverage
       require_orthogonal_angles: false # Only for anisotropy analysis

   # ============================================================
   # PLOTTING SECTION
   # ============================================================
   plotting:
     save_plots: true                   # Save plots to files
     show_plots: false                  # Display interactive plots
     format: "png"                      # "png" | "pdf" | "svg"
     dpi: 300                           # Resolution [dots per inch]
     style: "publication"               # Matplotlib style

     preview_mode: false                # true: fast Datashader preview
     fit_surface: "solver"              # "solver" | "posthoc"

     color_scale:
       mode: "legacy"                   # "legacy" | "percentile" | "fixed"
       pin_legacy_range: true           # Pin to legacy color range
       percentile_min: 1.0              # Low percentile for auto-scale
       percentile_max: 99.0             # High percentile
       fixed_min: 1.0                   # Fixed range minimum
       fixed_max: 1.5                   # Fixed range maximum

     datashader:
       canvas_width: 1200               # Canvas width [pixels]
       canvas_height: 1200              # Canvas height [pixels]

     matplotlib:
       interpolation: "bilinear"        # "none" | "bilinear" | "bicubic"
       use_tight_layout: true           # Use tight_layout

   # ============================================================
   # LOGGING SECTION
   # ============================================================
   logging:
     enabled: true
     level: "INFO"                      # "DEBUG" | "INFO" | "WARNING" | "ERROR"

     console:
       enabled: true
       level: "INFO"
       format: "detailed"               # "simple" | "detailed"
       colors: true

     file:
       enabled: false
       level: "DEBUG"
       path: "./logs/"
       filename: "homodyne_analysis.log"
       max_size_mb: 10                  # Max file size before rotation
       backup_count: 5                  # Rotated files to keep

   # ============================================================
   # PERFORMANCE SECTION
   # ============================================================
   performance:
     strategy_override: null            # Force: "standard" | "large" | "streaming"
     memory_limit_gb: null              # Custom memory limit [GB]

     computation:
       enable_jit: true                 # Enable JAX JIT compilation
       cpu_threads: "auto"              # "auto" or explicit int

---

Additional Configuration Details
----------------------------------

**Stratification** activates automatically when ``per_angle_scaling=True`` and
dataset size exceeds 100,000 points. It creates angle-balanced chunks for NLSQ.
When ``max_imbalance_ratio`` is exceeded, NLSQ falls back to sequential
(per-angle) optimization using the ``sequential`` settings.

**Phi filtering** selects which azimuthal angles to analyze. Each ``target_ranges``
entry defines an angular window; angles within ``tolerance`` degrees of the range
are included. Supports wrapping (e.g., 170 to -170 = 170 to 190 degrees).

**Performance strategy** is normally auto-selected based on dataset size:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Strategy
     - Dataset Size
     - Memory
     - Method
   * - ``standard``
     - < 10M points
     - ~2 GB per M
     - Full Jacobian
   * - ``large``
     - 10M--100M
     - ~2 GB per M
     - Chunked Jacobian
   * - ``streaming``
     - > 100M
     - ~2 GB fixed
     - Hybrid streaming

**Quick reference patterns:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Purpose
     - Configuration
   * - Single static optimization
     - ``mode: "static"``, ``method: "nlsq"``
   * - Laminar flow with uncertainty
     - ``mode: "laminar_flow"``, ``method: "both"``
   * - Large dataset (>10M)
     - ``strategy_override: "large"``
   * - Fast preview
     - ``preview_mode: true``
   * - Publication figures
     - ``format: "pdf"``, ``dpi: 300``
   * - Low memory
     - ``memory_limit_gb: 2.0``

---

Configuration Validation
--------------------------

Validate your configuration before running a long fit:

.. code-block:: bash

   homodyne-config --validate --input config.yaml

Or validate programmatically:

.. code-block:: python

   from homodyne.config import ConfigManager

   try:
       config = ConfigManager.from_yaml("config.yaml")
       print("Configuration valid")
   except ValueError as e:
       print(f"Configuration error: {e}")

**Physics validation:**

.. code-block:: python

   from homodyne.config import validate_all_parameters

   params = config.get_initial_parameters()
   violations = validate_all_parameters(params)

   for v in violations:
       print(f"{v.severity}: {v.parameter} — {v.message}")

---

Configuration Precedence
--------------------------

When the CLI is used, the base ``optimization.mcmc`` settings override
``optimization.cmc.per_shard_mcmc``. Keep them aligned:

.. code-block:: yaml

   # This is the correct way to align base mcmc and cmc settings:
   optimization:
     mcmc:                         # Base MCMC (CLI applies these to per_shard_mcmc)
       num_warmup: 500
       num_samples: 1500
       num_chains: 4
     cmc:
       per_shard_mcmc:
         num_warmup: 500           # Same as base mcmc
         num_samples: 1500         # Same as base mcmc
         num_chains: 4             # Same as base mcmc

---

Template Generation
--------------------

The ``homodyne-config`` command generates complete, valid configuration
templates:

.. code-block:: bash

   # Static mode template to stdout
   homodyne-config --mode static

   # Laminar flow template to file
   homodyne-config --mode laminar_flow --output my_config.yaml

   # Interactive configuration builder (asks questions)
   homodyne-config --interactive

   # Validate existing config
   homodyne-config --validate --input existing_config.yaml

   # Format/normalize config (auto-format YAML)
   homodyne-config --format --input messy_config.yaml --output clean_config.yaml

---

See Also
---------

- :doc:`../01_fundamentals/analysis_modes` — Mode selection
- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ workflow
- :doc:`../03_advanced_topics/bayesian_inference` — CMC workflow
- :doc:`performance_tuning` — XLA and CPU optimization
