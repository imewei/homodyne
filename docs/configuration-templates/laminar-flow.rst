Laminar Flow Template - 7+2n Parameters
========================================

The laminar flow template (``homodyne_laminar_flow.yaml``) is optimized for laminar flow systems with **7 physical + 2 per angle** parameters.

Purpose
-------

Analyze nonequilibrium dynamics in systems under shear flow:
  - Colloidal suspensions under shear
  - Flowing soft matter systems
  - Time-dependent shear experiments
  - Nonequilibrium XPCS measurements

Parameter Count
---------------

**Formula:** 7 physical + 2 × (number of filtered phi angles)

**Physical Parameters (7):**

*Diffusion Parameters:*
  - **D₀:** Initial diffusion coefficient [Å²/s]
  - **α:** Diffusion power-law exponent [dimensionless]
  - **D_offset:** Diffusion offset [Å²/s]

*Shear Rate Parameters:*
  - **γ̇₀:** Initial shear rate [s⁻¹]
  - **β:** Shear rate power-law exponent [dimensionless]
  - **γ̇_offset:** Shear rate offset [s⁻¹]

*Flow Direction:*
  - **φ₀:** Initial angle between flow and scattering vector [degrees]

**Per-Angle Scaling (2 per angle):**
  - **contrast:** Signal contrast for each phi angle
  - **offset:** Baseline correlation for each phi angle

**Examples:**
  - 3 angles → 7 + 2×3 = **13 total parameters**
  - 5 angles → 7 + 2×5 = **17 total parameters**
  - 8 angles → 7 + 2×8 = **23 total parameters**

Template Features
-----------------

**All Optimization Methods Included:**
  - **NLSQ:** Trust-region solver (primary method)
  - **MCMC:** NumPyro/BlackJAX sampling (uncertainty quantification)
  - **Streaming:** For datasets >100M points
  - **CMC:** For large multi-angle datasets >1M points

**Angle Filtering Enabled by Default:**
  - Reduces parameter count by focusing on relevant angles
  - Configured for parallel (0°) and perpendicular (90°) to flow
  - All angles normalized to [-180°, 180°] with wrap-aware checking

Quick Start
-----------

1. **Copy template:**

   .. code-block:: bash

      cp homodyne/config/templates/homodyne_laminar_flow.yaml my_config.yaml

2. **Edit configuration:**

   .. code-block:: yaml

      experimental_data:
        file_path: "/path/to/your/data.hdf"  # Update this

      parameter_space:
        bounds:
          # Diffusion parameters (adjust based on system)
          - name: D0
            min: 100.0
            max: 1e5
          - name: alpha
            min: 0.0
            max: 2.0

          # Shear parameters (adjust based on applied shear)
          - name: gamma_dot_0
            min: 1e-6
            max: 0.5

      # Angle filtering (customize for your measurement geometry)
      phi_filtering:
        enabled: true
        target_ranges:
          - min_angle: -10.0
            max_angle: 10.0
            description: "Parallel to flow"
          - min_angle: 85.0
            max_angle: 95.0
            description: "Perpendicular to flow"

3. **Run analysis:**

   .. code-block:: bash

      # NLSQ optimization
      homodyne --config my_config.yaml --method nlsq

      # MCMC for full uncertainty quantification
      homodyne --config my_config.yaml --method mcmc

Template Structure
------------------

.. code-block:: yaml

   # Experimental Data
   experimental_data:
     file_path: "./data/sample/experiment.hdf"

   # Parameter Space - 7 Physical Parameters
   parameter_space:
     model: "laminar_flow"
     bounds:
       # Diffusion (3 parameters)
       - name: D0
       - name: alpha
       - name: D_offset
       # Shear rate (3 parameters)
       - name: gamma_dot_0
       - name: beta
       - name: gamma_dot_offset
       # Flow direction (1 parameter)
       - name: phi_0

   # Initial Parameters
   initial_parameters:
     parameter_names: [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi_0]

   # Optimization Methods (ALL INCLUDED)
   optimization:
     method: "nlsq"
     nlsq: {...}
     mcmc: {...}
     streaming: {...}
     cmc: {...}

   # Angle Filtering (RECOMMENDED)
   phi_filtering:
     enabled: true
     target_ranges: [...]

   # Performance & Device
   performance:
     strategy_override: null
     device:
       preferred_device: "auto"

   # Output
   output:
     directory: "./results"

Parameter Guidance
------------------

Diffusion Parameters
~~~~~~~~~~~~~~~~~~~~

**D₀ (Diffusion Coefficient)**
  - **Range:** 100 - 10,000 Å²/s (may be enhanced by shear)
  - **Physical meaning:** Prefactor in D(t) = D₀·t^α + D_offset
  - **Under shear:** Often larger than equilibrium value

**α (Diffusion Exponent)**
  - **Range:** -2 to 2 (may show superdiffusion under strong shear)
  - **Physical meaning:**
    - α > 0: Superdiffusion (shear-enhanced transport)
    - α ≈ 0: Normal diffusion
    - α < 0: Subdiffusion (caging effects persist)

**D_offset**
  - **Range:** -100 to 100 Å²/s
  - **Physical meaning:** Baseline diffusion contribution

Shear Parameters
~~~~~~~~~~~~~~~~

**γ̇₀ (Initial Shear Rate)**
  - **Range:** 1e-6 to 0.5 s⁻¹ (depends on applied shear)
  - **Physical meaning:** Shear rate prefactor in γ̇(t) = γ̇₀·t^β + γ̇_offset
  - **Typical values:**
    - Low shear: 1e-4 to 1e-3 s⁻¹
    - High shear: 0.01 to 0.1 s⁻¹

**β (Shear Rate Exponent)**
  - **Range:** 0 to 2
  - **Physical meaning:**
    - β = 0: Constant shear rate (steady shear)
    - β ≠ 0: Time-dependent shear rate
  - **Typical values:**
    - Steady shear: β ≈ 0
    - Transient flow: β > 0

**γ̇_offset**
  - **Range:** -0.1 to 0.1 s⁻¹
  - **Physical meaning:** Baseline shear rate

**φ₀ (Flow Direction Angle)**
  - **Range:** -180° to 180° (auto-normalized)
  - **Physical meaning:** Initial angle between flow direction and scattering vector
  - **Typical values:** Should align with experimental geometry (often 0° or 90°)

Optimization Methods
--------------------

**NLSQ (Recommended for initial analysis)**
  - Fast parameter estimation (seconds to minutes)
  - Works well with 7-parameter space
  - Automatic strategy selection handles large datasets

**MCMC (Recommended for final results)**
  - Full posterior sampling with correlations
  - Essential for understanding parameter uncertainties in 7-dimensional space
  - Slower but provides complete uncertainty quantification

**Streaming (Automatic for >100M points)**
  - Constant memory footprint
  - Checkpoint/resume capability

**CMC (For large multi-angle datasets)**
  - Parallel optimization across angles
  - GPU-accelerated with JAX backend
  - Ideal for datasets >1M points with multiple angles

Angle Filtering Strategy
-------------------------

**Why Filter Angles?**
  Reducing the number of angles lowers total parameter count (7+2n), improving:
    - Optimization stability
    - Convergence speed
    - Parameter identifiability

**Recommended Angles for Flow Analysis:**

.. code-block:: yaml

   phi_filtering:
     enabled: true
     target_ranges:
       # Primary flow information
       - min_angle: -10.0
         max_angle: 10.0
         description: "Parallel to flow"
       - min_angle: 85.0
         max_angle: 95.0
         description: "Perpendicular to flow"

       # Optional: diagonal angles for complete characterization
       # - min_angle: 40.0
       #   max_angle: 50.0
       # - min_angle: 130.0
       #   max_angle: 140.0

Output Files
------------

.. code-block:: text

   results/
   ├── nlsq/
   │   ├── parameters.json           # All 7+2n parameters ± uncertainties
   │   ├── fitted_data.npz           # Per-angle fits
   │   ├── analysis_results_nlsq.json # Fit quality
   │   └── convergence_metrics.json  # Convergence diagnostics
   └── logs/
       └── homodyne_analysis_YYYYMMDD_HHMMSS.log

Troubleshooting
---------------

**Convergence Issues with 7 Parameters:**
  - Start with tighter bounds based on system knowledge
  - Use angle filtering to reduce total parameter count
  - Run NLSQ first for initial estimates, then MCMC

**Unphysical Flow Parameters:**
  - Check that applied shear matches γ̇₀ bounds
  - Verify flow direction φ₀ aligns with experimental geometry
  - Consider if shear is truly time-dependent (β ≠ 0)

**Parameter Correlations:**
  - Expected in 7-parameter space
  - Use MCMC to visualize correlations via corner plots
  - Consider fixing well-known parameters (e.g., φ₀ if known from setup)

**Memory or Speed Issues:**
  - Enable CMC for large multi-angle datasets
  - Use angle filtering to reduce data size
  - GPU acceleration (Linux only) significantly speeds up optimization

Use Cases
---------

**Typical Applications:**
  - Rheo-XPCS (combined rheology and XPCS)
  - Shear-induced dynamics in colloids
  - Flow-enhanced mixing
  - Microfluidic flow characterization
  - Time-dependent shear protocols

**Example Workflow:**

1. **Initial NLSQ analysis:**

   .. code-block:: bash

      homodyne --config config.yaml --method nlsq --output-dir ./nlsq_initial

2. **Check parameters and refine bounds** based on ``nlsq_initial/nlsq/parameters.json``

3. **Run MCMC for uncertainties:**

   .. code-block:: bash

      homodyne --config config.yaml --method mcmc --output-dir ./mcmc_final

4. **Analyze flow characteristics:**
   - Time-dependent shear rate: γ̇(t) = γ̇₀·t^β + γ̇_offset
   - Shear-enhanced diffusion: Compare D₀ with equilibrium value
   - Flow direction validation: φ₀ should match experimental geometry

Physical Interpretation
-----------------------

**Shear-Enhanced Diffusion:**
  If D₀ (flow) > D₀ (equilibrium), shear enhances particle transport.

**Flow Direction:**
  φ₀ ≈ 0° indicates flow parallel to scattering vector.
  φ₀ ≈ 90° indicates flow perpendicular to scattering vector.

**Time-Dependent Shear:**
  - β ≈ 0: Steady shear throughout experiment
  - β > 0: Increasing shear rate with time (e.g., startup flow)
  - β < 0: Decreasing shear rate (e.g., stress relaxation)

**Anomalous Diffusion Under Shear:**
  - α > 0 with flow: Superdiffusion (ballistic transport)
  - α < 0 with flow: Subdiffusion persists despite shear
  - Compare α (flow) vs α (equilibrium) to quantify shear effects

Template Location
-----------------

.. code-block:: bash

   # In package installation
   homodyne/config/templates/homodyne_laminar_flow.yaml

   # Find full path
   python -c "import homodyne; print(homodyne.__file__.replace('__init__.py', 'config/templates/homodyne_laminar_flow.yaml'))"

Line Count
----------

**Laminar flow template:** ~235 lines (production-ready with comprehensive comments)

See Also
--------

- :doc:`static-isotropic` - For equilibrium systems (3+2n parameters)
- :doc:`master-template` - For comprehensive reference
- :doc:`index` - Template overview
- :doc:`../theoretical-framework/parameter-models` - Parameter model details
- :doc:`../theoretical-framework/transport-coefficients` - D(t) and γ̇(t) framework
- :doc:`../advanced-topics/nlsq-optimization` - NLSQ workflows
- :doc:`../advanced-topics/mcmc-uncertainty` - MCMC workflows
- :doc:`../advanced-topics/angle-filtering` - Angle selection strategies
