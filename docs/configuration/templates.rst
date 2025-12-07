Configuration Templates
=======================

This section provides complete YAML configuration templates for both analysis modes.
Use these templates as starting points for your own configurations.

Template Overview
-----------------

Homodyne provides templates for:

1. **Static Mode** - Equilibrium systems with pure diffusion (3 physical parameters)
2. **Laminar Flow Mode** - Non-equilibrium systems under shear (7 physical parameters)

Both templates include:
- Detailed comments explaining each parameter
- Typical values and physical ranges
- Per-angle scaling configuration
- NLSQ and MCMC optimization settings
- Performance tuning options
- Quality control and validation settings

Getting Started
^^^^^^^^^^^^^^^

1. Choose your analysis mode (static or laminar_flow)
2. Copy the template to your working directory
3. Edit paths for your experimental data
4. Adjust parameters based on your system
5. Validate with: ``homodyne-config --validate config.yaml``
6. Run: ``homodyne --config config.yaml --method nlsq``

Static Mode Template
--------------------

Use this template for equilibrium systems with pure diffusion.

**Model**: :math:`D(t,\phi) = D_0 \cdot t^\alpha + D_{\text{offset}}`

**Parameters**: 3 physical + 2 × N_angles scaling

**When to use**:
- Colloidal suspensions without flow
- Polymer solutions in equilibrium
- Systems with anomalous diffusion only
- Anisotropic media without flow-induced effects

Complete template with all available options:

.. code-block:: yaml

    # ==============================================================================
    # HOMODYNE STATIC DIFFUSION CONFIGURATION TEMPLATE
    # ==============================================================================
    # Equilibrium systems with 3-parameter model: D(t) = D₀·t^α + D_offset
    # VERSION: 2.4.1 | Python: 3.12+ | JAX: 0.8.0 (CPU-only)
    # ==============================================================================

    # ==============================================================================
    # METADATA (Required)
    # ==============================================================================
    metadata:
      config_version: "2.4.1"
      description: "Static diffusion analysis - equilibrium system"
      analysis_mode: "static"
      parameter_count: 3

    # ==============================================================================
    # ANALYSIS MODE (Required)
    # ==============================================================================
    analysis_mode: "static"

    # ==============================================================================
    # ANALYZER PARAMETERS (Required)
    # ==============================================================================
    analyzer_parameters:
      dt: 0.1                           # Time step [seconds]
      start_frame: 1000                 # Starting frame (1-indexed)
      end_frame: 2000                   # Ending frame (inclusive)

      scattering:
        wavevector_q: 0.0054            # Wave vector magnitude [Å⁻¹]

      geometry:
        stator_rotor_gap: 2000000       # Gap [Å] (200 microns = 2000000 Å)

    # ==============================================================================
    # ANALYSIS SETTINGS (Required)
    # ==============================================================================
    analysis_settings:
      static_mode: true                 # Static diffusion (no flow)

      model_description:
        type: "static_diffusion"
        parameters: 3
        physics: "Equilibrium anomalous diffusion"

    # ==============================================================================
    # EXPERIMENTAL DATA (Required)
    # ==============================================================================
    experimental_data:
      file_path: "./data/experiment.hdf"
      data_folder_path: "./data/"
      data_file_name: "experiment.hdf"

      phi_angles_path: "./data/"
      phi_angles_file: "phi_angles_list.txt"

      cache_file_path: "./data/"
      cache_filename_template: "cached_c2_static_q{wavevector_q:.4f}_frames_{start_frame}_{end_frame}.npz"
      cache_compression: true

      data_type: "float64"
      file_format: "HDF5"
      exchange_key: "exchange"

    # ==============================================================================
    # PHI ANGLE FILTERING (Optional but Recommended)
    # ==============================================================================
    phi_filtering:
      enabled: true
      target_ranges:
        # Parallel to primary axis
        - min_angle: -10.0
          max_angle: 10.0
          description: "Parallel to primary axis"

        # Antiparallel to anisotropy axis
        - min_angle: 170.0
          max_angle: -170.0
          description: "Antiparallel to primary axis"

      fallback_to_all_angles: true
      algorithm: "range_based"
      tolerance: 3.0

      quality_control:
        min_angles_required: 1
        max_angle_spread: 36.0
        validate_coverage: true
        require_orthogonal_angles: false

    # ==============================================================================
    # INITIAL PARAMETERS (Required)
    # ==============================================================================
    initial_parameters:
      parameter_names:
        - D0                            # Diffusion prefactor
        - alpha                         # Anomalous exponent
        - D_offset                      # Baseline diffusion

      # Set values from NLSQ results for MCMC
      values: null                      # Example: [1000.0, -1.2, 0.0]

      # Per-angle scaling (v2.4.0 mandatory)
      per_angle_scaling:
        contrast: null                  # Example: [0.05, 0.06, 0.05]
        offset: null                    # Example: [1.0, 0.99, 1.01]

      units:
        - "Å²/s"                        # D0
        - "dimensionless"               # alpha
        - "Å²/s"                        # D_offset

      active_parameters: null           # null = optimize all
      fixed_parameters: null            # null = no fixed parameters

    # ==============================================================================
    # PARAMETER SPACE (Required)
    # ==============================================================================
    parameter_space:
      model: "static"

      bounds:
        - name: D0
          min: 100.0
          max: 1e5
          type: TruncatedNormal
          prior_mu: 1000.0
          prior_sigma: 1000.0
          unit: "Å²/s"

        - name: alpha
          min: -2.0
          max: 2.0
          type: TruncatedNormal
          prior_mu: -1.2
          prior_sigma: 0.3
          unit: "dimensionless"

        - name: D_offset
          min: -100000.0
          max: 100000.0
          type: TruncatedNormal
          prior_mu: 0.0
          prior_sigma: 150.0
          unit: "Å²/s"

      priors: null

    # ==============================================================================
    # OPTIMIZATION METHODS
    # ==============================================================================
    optimization:
      method: "nlsq"                    # "nlsq" or "mcmc"

      # NLSQ configuration
      nlsq:
        max_iterations: 100
        tolerance: 1e-8
        trust_region_scale: 1.0
        verbose: false
        x_scale_map: null

      stratification:
        enabled: "auto"
        target_chunk_size: 100000
        max_imbalance_ratio: 5.0
        force_sequential_fallback: false
        check_memory_safety: true
        use_index_based: false
        collect_diagnostics: false
        log_diagnostics: false

      sequential:
        min_success_rate: 0.5
        weighting: "inverse_variance"

      # MCMC configuration
      mcmc:
        backend: "numpyro"
        num_warmup: 2000
        num_samples: 5000
        num_chains: 4
        progress_bar: true
        target_accept_prob: 0.90
        max_tree_depth: 12
        dense_mass_matrix: true

        sharding:
          strategy: "stratified"
          num_shards: "auto"
          max_points_per_shard: "auto"
          seed_base: 0

        initial_values:
          phi: {}
          percentile_fallback:
            contrast_low_pct: 5
            contrast_high_pct: 95
            offset_pct: 50

        backend_config:
          name: "auto"
          enable_checkpoints: true
          checkpoint_frequency: 1
          checkpoint_dir: "./cmc_checkpoints"
          keep_last_checkpoints: 3
          resume_from_checkpoint: true

        combination:
          method: "weighted_gaussian"
          validate_results: true
          min_success_rate: 0.8

        per_shard_mcmc:
          num_warmup: 500
          num_samples: 1000
          num_chains: 2
          subsample_size: "auto"

        validation:
          strict_mode: false
          min_per_shard_ess: 100
          max_per_shard_rhat: 1.2
          max_between_shard_kl: 0.5
          min_success_rate: 0.8

    # ==============================================================================
    # PERFORMANCE OPTIMIZATION
    # ==============================================================================
    performance:
      strategy_override: null
      memory_limit_gb: null
      enable_progress: true

      memory_optimization:
        enabled: true
        max_memory_usage_gb: 6.0
        chunk_size: 8000
        enable_caching: true
        cache_strategy: "adaptive"

      computation:
        enable_jit: true
        cpu_threads: "auto"
        vectorization_level: "high"

    # ==============================================================================
    # OUTPUT
    # ==============================================================================
    output:
      directory: "./results"
      base_directory: "./homodyne_results/"

      formats:
        hdf5: true
        json: true
        csv: true

      create_subdirs: true
      timestamp_dirs: false

      compress_hdf5: true
      compression_level: 6

    # ==============================================================================
    # LOGGING
    # ==============================================================================
    logging:
      enabled: true
      level: "INFO"

      console:
        enabled: true
        level: "INFO"
        format: "detailed"
        colors: true
        show_progress: true

      file:
        enabled: false
        level: "DEBUG"
        path: "./logs/"
        filename: "homodyne_static_analysis.log"
        max_size_mb: 10
        backup_count: 5

      modules:
        "homodyne.data.phi_filtering": "INFO"
        "jax._src": "WARNING"

    # ==============================================================================
    # VISUALIZATION (Optional)
    # ==============================================================================
    plotting:
      save_plots: true
      show_plots: false
      format: "png"
      dpi: 300
      style: "publication"

      preview_mode: false
      fit_surface: "solver"

      color_scale:
        mode: "legacy"
        pin_legacy_range: true
        percentile_min: 1.0
        percentile_max: 99.0
        fixed_min: 1.0
        fixed_max: 1.5

      datashader:
        canvas_width: 1200
        canvas_height: 1200

      matplotlib:
        interpolation: "bilinear"
        use_tight_layout: true
        savefig_kwargs:
          bbox_inches: "tight"
          pad_inches: 0.1

      correlation_function: true
      fit_quality: true
      parameter_distributions: true
      residual_analysis: true

      angle_coverage: true
      angle_correlation: true

    # ==============================================================================
    # VALIDATION (Optional)
    # ==============================================================================
    validation:
      strict_mode: false
      check_file_existence: true
      validate_parameter_ranges: true
      check_mode_compatibility: true

      angle_validation:
        require_multiple_angles: false
        min_angle_count: 1
        validate_angle_ranges: true

Laminar Flow Mode Template
--------------------------

Use this template for non-equilibrium systems under shear flow.

**Model**: :math:`D(t,\phi) = D_0 \cdot t^\alpha + D_{\text{offset}} + \text{shear effects}`

**Shear Rate**: :math:`\dot{\gamma}(t) = \dot{\gamma}_0 \cdot t^\beta + \dot{\gamma}_{\text{offset}}`

**Parameters**: 7 physical + 2 × N_angles scaling

**When to use**:
- Polymer solutions under shear
- Colloidal suspensions in rheometer
- Systems with time-dependent shear rate
- Shear-induced anisotropy studies

Complete template:

.. code-block:: yaml

    # ==============================================================================
    # HOMODYNE LAMINAR FLOW CONFIGURATION TEMPLATE
    # ==============================================================================
    # Non-equilibrium systems with 7-parameter model including shear effects
    # VERSION: 2.4.1 | Python: 3.12+ | JAX: 0.8.0 (CPU-only)
    # ==============================================================================

    metadata:
      config_version: "2.4.1"
      description: "Laminar flow analysis - non-equilibrium system under shear"
      analysis_mode: "laminar_flow"
      parameter_count: 7

    analysis_mode: "laminar_flow"

    analyzer_parameters:
      dt: 0.1
      start_frame: 1000
      end_frame: 2000

      scattering:
        wavevector_q: 0.0054

      geometry:
        stator_rotor_gap: 2000000

    analysis_settings:
      static_mode: false               # Laminar flow (with shear)

      model_description:
        type: "nonequilibrium_laminar_flow"
        parameters: 7
        physics: "Time-dependent diffusion and shear with flow angle effects"

    experimental_data:
      file_path: "./data/experiment.hdf"
      data_folder_path: "./data/"
      data_file_name: "experiment.hdf"

      phi_angles_path: "./data/"
      phi_angles_file: "phi_angles_list.txt"

      cache_file_path: "./data/"
      cache_filename_template: "cached_c2_flow_q{wavevector_q:.4f}_frames_{start_frame}_{end_frame}.npz"
      cache_compression: true

      data_type: "float64"
      file_format: "HDF5"
      exchange_key: "exchange"

    phi_filtering:
      enabled: true
      target_ranges:
        # Flow direction (0°)
        - min_angle: -10.0
          max_angle: 10.0
          description: "Flow direction"

        # Orthogonal to flow (90°)
        - min_angle: 80.0
          max_angle: 100.0
          description: "Orthogonal to flow"

        # Reverse flow (180°)
        - min_angle: 170.0
          max_angle: -170.0
          description: "Reverse flow direction"

      fallback_to_all_angles: true
      algorithm: "range_based"
      tolerance: 3.0

      quality_control:
        min_angles_required: 3
        max_angle_spread: 36.0
        validate_coverage: true
        require_orthogonal_angles: true

    initial_parameters:
      parameter_names:
        - D0                            # Diffusion prefactor
        - alpha                         # Diffusion exponent
        - D_offset                      # Baseline diffusion
        - gamma_dot_t0                  # Shear rate prefactor
        - beta                          # Shear rate exponent
        - gamma_dot_t_offset            # Baseline shear rate
        - phi0                          # Flow angle

      values: null

      per_angle_scaling:
        contrast: null
        offset: null

      units:
        - "Å²/s"                        # D0
        - "dimensionless"               # alpha
        - "Å²/s"                        # D_offset
        - "s⁻¹"                         # gamma_dot_t0
        - "dimensionless"               # beta
        - "s⁻¹"                         # gamma_dot_t_offset
        - "degrees"                     # phi0

      active_parameters: null
      fixed_parameters: null

    parameter_space:
      model: "laminar_flow"

      bounds:
        - name: D0
          min: 100.0
          max: 1e5
          type: TruncatedNormal
          prior_mu: 1000.0
          prior_sigma: 1000.0
          unit: "Å²/s"

        - name: alpha
          min: -2.0
          max: 2.0
          type: TruncatedNormal
          prior_mu: -1.2
          prior_sigma: 0.3
          unit: "dimensionless"

        - name: D_offset
          min: -100000.0
          max: 100000.0
          type: TruncatedNormal
          prior_mu: 0.0
          prior_sigma: 150.0
          unit: "Å²/s"

        - name: gamma_dot_t0
          min: 0.001
          max: 1000.0
          type: TruncatedNormal
          prior_mu: 1.0
          prior_sigma: 10.0
          unit: "s⁻¹"

        - name: beta
          min: -2.0
          max: 2.0
          type: TruncatedNormal
          prior_mu: 0.5
          prior_sigma: 0.5
          unit: "dimensionless"

        - name: gamma_dot_t_offset
          min: -1000.0
          max: 1000.0
          type: TruncatedNormal
          prior_mu: 0.0
          prior_sigma: 10.0
          unit: "s⁻¹"

        - name: phi0
          min: -180.0
          max: 180.0
          type: TruncatedNormal
          prior_mu: 0.0
          prior_sigma: 45.0
          unit: "degrees"

      priors: null

    optimization:
      method: "nlsq"

      nlsq:
        max_iterations: 150
        tolerance: 1e-8
        trust_region_scale: 1.0
        verbose: false
        x_scale_map:
          D0: 1.0
          alpha: 0.01
          D_offset: 1.0
          gamma_dot_t0: 0.01
          beta: 0.1
          gamma_dot_t_offset: 0.01
          phi0: 0.1

      stratification:
        enabled: "auto"
        target_chunk_size: 100000
        max_imbalance_ratio: 5.0
        force_sequential_fallback: false
        check_memory_safety: true
        use_index_based: false
        collect_diagnostics: false
        log_diagnostics: false

      sequential:
        min_success_rate: 0.5
        weighting: "inverse_variance"

      mcmc:
        backend: "numpyro"
        num_warmup: 3000
        num_samples: 8000
        num_chains: 4
        progress_bar: true
        target_accept_prob: 0.90
        max_tree_depth: 12
        dense_mass_matrix: true

        sharding:
          strategy: "stratified"
          num_shards: "auto"
          max_points_per_shard: "auto"
          seed_base: 0

        initial_values:
          phi: {}
          percentile_fallback:
            contrast_low_pct: 5
            contrast_high_pct: 95
            offset_pct: 50

        backend_config:
          name: "auto"
          enable_checkpoints: true
          checkpoint_frequency: 1
          checkpoint_dir: "./cmc_checkpoints"
          keep_last_checkpoints: 3
          resume_from_checkpoint: true

        combination:
          method: "weighted_gaussian"
          validate_results: true
          min_success_rate: 0.8

        per_shard_mcmc:
          num_warmup: 500
          num_samples: 1000
          num_chains: 2
          subsample_size: "auto"

        validation:
          strict_mode: false
          min_per_shard_ess: 100
          max_per_shard_rhat: 1.2
          max_between_shard_kl: 0.5
          min_success_rate: 0.8

    performance:
      strategy_override: null
      memory_limit_gb: null
      enable_progress: true

      memory_optimization:
        enabled: true
        max_memory_usage_gb: 6.0
        chunk_size: 8000
        enable_caching: true
        cache_strategy: "adaptive"

      computation:
        enable_jit: true
        cpu_threads: "auto"
        vectorization_level: "high"

    output:
      directory: "./results"
      base_directory: "./homodyne_results/"

      formats:
        hdf5: true
        json: true
        csv: true

      create_subdirs: true
      timestamp_dirs: false

      compress_hdf5: true
      compression_level: 6

    logging:
      enabled: true
      level: "INFO"

      console:
        enabled: true
        level: "INFO"
        format: "detailed"
        colors: true
        show_progress: true

      file:
        enabled: false
        level: "DEBUG"
        path: "./logs/"
        filename: "homodyne_laminar_analysis.log"
        max_size_mb: 10
        backup_count: 5

      modules:
        "homodyne.data.phi_filtering": "INFO"
        "jax._src": "WARNING"

    plotting:
      save_plots: true
      show_plots: false
      format: "png"
      dpi: 300
      style: "publication"

      preview_mode: false
      fit_surface: "solver"

      color_scale:
        mode: "legacy"
        pin_legacy_range: true
        percentile_min: 1.0
        percentile_max: 99.0
        fixed_min: 1.0
        fixed_max: 1.6

      datashader:
        canvas_width: 1200
        canvas_height: 1200

      matplotlib:
        interpolation: "bilinear"
        use_tight_layout: true
        savefig_kwargs:
          bbox_inches: "tight"
          pad_inches: 0.1

      correlation_function: true
      fit_quality: true
      parameter_distributions: true
      residual_analysis: true

      angle_coverage: true
      angle_correlation: true

    validation:
      strict_mode: false
      check_file_existence: true
      validate_parameter_ranges: true
      check_mode_compatibility: true

      angle_validation:
        require_multiple_angles: true
        min_angle_count: 3
        validate_angle_ranges: true

Workflow Guidance
-----------------

NLSQ → MCMC Workflow
^^^^^^^^^^^^^^^^^^^^

Recommended workflow for publication-quality results with uncertainty quantification:

1. **Run NLSQ optimization** (fast, deterministic)

   .. code-block:: bash

       homodyne --config config.yaml --method nlsq

2. **Copy best-fit parameters** from output to config file:

   .. code-block:: yaml

       initial_parameters:
         values: [1234.5, -1.2, 45.6]

3. **Run MCMC inference** (slower, full uncertainty distributions)

   .. code-block:: bash

       homodyne --config config.yaml --method mcmc

4. **Analyze results** with posterior distributions and credible intervals

Per-Angle Scaling Details
--------------------------

Per-angle scaling is **mandatory** in v2.4.0+.

Each scattering angle has unique:
- Detector sensitivity
- Optical alignment
- Background levels

Therefore each angle requires independent scaling parameters:
- **contrast**: Magnitude scaling
- **offset**: Baseline/intercept shift

Example for 3 filtered angles:

.. code-block:: yaml

    per_angle_scaling:
      contrast: [0.05, 0.06, 0.05]    # Magnitude per angle
      offset: [1.0, 0.99, 1.01]       # Baseline per angle

When initializing from NLSQ results, copy these values from NLSQ output.

See Also
--------

- :doc:`/configuration/options` - Complete parameter reference
- :doc:`/user-guide/configuration` - Configuration concepts
- :doc:`/user-guide/cli` - Command-line usage
