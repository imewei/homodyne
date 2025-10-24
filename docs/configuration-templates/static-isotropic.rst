Static Isotropic Template - 3+2n Parameters
===========================================

The static isotropic template (``homodyne_static_isotropic.yaml``) is optimized for static isotropic systems with **3 physical + 2 per angle** parameters.

Purpose
-------

Analyze equilibrium dynamics in systems without flow:
  - Colloidal suspensions at rest
  - Glass-forming liquids
  - Gel systems without shear
  - Single-angle or isotropic angle measurements

Parameter Count
---------------

**Formula:** 3 physical + 2 × (number of filtered phi angles)

**Physical Parameters (3):**
  - **D₀:** Initial diffusion coefficient [Å²/s]
  - **α:** Power-law exponent [dimensionless] (0=normal, <1=subdiffusion, >1=superdiffusion)
  - **D_offset:** Diffusion offset [Å²/s]

**Per-Angle Scaling (2 per angle):**
  - **contrast:** Signal contrast for each phi angle
  - **offset:** Baseline correlation for each phi angle

**Examples:**
  - 3 angles → 3 + 2×3 = **9 total parameters**
  - 5 angles → 3 + 2×5 = **13 total parameters**
  - Single angle → 3 + 2×1 = **5 total parameters**

Template Features
-----------------

**All Optimization Methods Included:**
  - **NLSQ:** Trust-region solver (primary method, fast)
  - **MCMC:** NumPyro/BlackJAX sampling (uncertainty quantification)
  - **Streaming:** For datasets >100M points (constant memory)
  - **CMC:** For large multi-angle datasets >1M points (parallel optimization)

**Automatic Features:**
  - Strategy selection based on dataset size (<1M, 1M-10M, 10M-100M, >100M)
  - GPU/CPU device selection (Linux GPU, all platforms CPU)
  - Progress bars and logging

Quick Start
-----------

1. **Copy template:**

   .. code-block:: bash

      cp homodyne/config/templates/homodyne_static_isotropic.yaml my_config.yaml

2. **Edit configuration:**

   .. code-block:: yaml

      experimental_data:
        file_path: "/path/to/your/data.hdf"  # Update this

      parameter_space:
        bounds:
          - name: D0
            min: 100.0      # Adjust based on your system
            max: 1e5
          - name: alpha
            min: 0.0        # Typical subdiffusion: -2 to 0
            max: 2.0
          - name: D_offset
            min: -100.0
            max: 100.0

3. **Run analysis:**

   .. code-block:: bash

      # NLSQ optimization (fast, deterministic)
      homodyne --config my_config.yaml --method nlsq

      # MCMC sampling (slower, with uncertainties)
      homodyne --config my_config.yaml --method mcmc

Template Structure
------------------

.. code-block:: yaml

   # Experimental Data
   experimental_data:
     file_path: "./data/sample/experiment.hdf"

   # Parameter Space - 3 Physical Parameters
   parameter_space:
     model: "static_isotropic"
     bounds:
       - name: D0            # Diffusion coefficient [Å²/s]
       - name: alpha         # Anomalous exponent [dimensionless]
       - name: D_offset      # Diffusion offset [Å²/s]

   # Initial Parameters
   initial_parameters:
     parameter_names: [D0, alpha, D_offset]

   # Optimization Methods (ALL INCLUDED)
   optimization:
     method: "nlsq"          # or "mcmc"
     nlsq: {...}             # Trust-region settings
     mcmc: {...}             # MCMC sampling settings
     streaming: {...}        # Large dataset settings
     cmc: {...}              # Multi-angle settings

   # Optional Angle Filtering
   phi_filtering:
     enabled: false          # Set to true if needed

   # Performance & Device
   performance:
     strategy_override: null # Automatic selection
     device:
       preferred_device: "auto"

   # Output
   output:
     directory: "./results"

Parameter Guidance
------------------

**D₀ (Diffusion Coefficient)**
  - **Range:** 100 - 10,000 Å²/s for typical colloidal systems
  - **Physical meaning:** Prefactor in diffusion law D(t) = D₀·t^α + D_offset
  - **Typical values:**
    - Small colloids (100 nm): ~1000 Å²/s
    - Large colloids (1 μm): ~100 Å²/s

**α (Anomalous Exponent)**
  - **Range:** -2 to 2 (most systems: -1 to 1)
  - **Physical meaning:**
    - α = 0: Normal diffusion
    - α < 0: Subdiffusion (caging, crowding)
    - α > 0: Superdiffusion (ballistic motion, active matter)
  - **Typical values:**
    - Colloidal glasses: α ≈ -0.5 to -1.5
    - Simple liquids: α ≈ 0

**D_offset (Diffusion Offset)**
  - **Range:** Usually small or zero (-100 to 100 Å²/s)
  - **Physical meaning:** Baseline diffusion independent of time
  - **Typical values:** 0 to 10 Å²/s

Optimization Methods
--------------------

**NLSQ (Recommended for initial analysis)**
  - **Speed:** Seconds to minutes
  - **Output:** Point estimates with standard errors
  - **Use when:** Quick parameter estimation, deterministic results needed

**MCMC (Recommended for publication)**
  - **Speed:** Minutes to hours
  - **Output:** Full posterior distributions with correlations
  - **Use when:** Uncertainty quantification needed, model comparison

**Streaming (Automatic for >100M points)**
  - **Speed:** Moderate with batching
  - **Output:** Same as NLSQ with checkpoint/resume
  - **Use when:** Memory constraints, very large datasets

**CMC (For multi-angle datasets >1M points)**
  - **Speed:** Fast with parallel processing
  - **Output:** Combined estimates across angles
  - **Use when:** Multiple angles, large datasets, GPU available

Output Files
------------

After running analysis, you'll find:

.. code-block:: text

   results/
   ├── nlsq/
   │   ├── parameters.json           # Parameter values ± uncertainties
   │   ├── fitted_data.npz           # Experimental + theoretical + residuals
   │   ├── analysis_results_nlsq.json # Fit quality metrics
   │   └── convergence_metrics.json  # Convergence diagnostics
   └── logs/
       └── homodyne_analysis_YYYYMMDD_HHMMSS.log

Troubleshooting
---------------

**Optimization doesn't converge:**
  - Reduce tolerance: ``nlsq.tolerance: 1e-6`` instead of ``1e-8``
  - Increase iterations: ``nlsq.max_iterations: 200``
  - Adjust parameter bounds to narrow ranges

**Memory errors:**
  - Automatic strategy selection should prevent this
  - Force streaming: ``performance.strategy_override: "streaming"``

**Unphysical parameters:**
  - Tighten parameter bounds based on system knowledge
  - Check data quality (NaN values, corruption)

**Slow performance:**
  - Enable GPU if available (Linux only)
  - Reduce data size with angle filtering
  - Use NLSQ instead of MCMC for initial analysis

Use Cases
---------

**Typical Applications:**
  - Colloidal glass transition studies
  - Aging dynamics in gels
  - Equilibrium dynamics characterization
  - Single-angle XPCS measurements
  - Isotropic scattering patterns

**Example Workflow:**

1. Run NLSQ for initial parameters:

   .. code-block:: bash

      homodyne --config config.yaml --method nlsq --output-dir ./initial

2. Check convergence and fit quality in ``initial/nlsq/analysis_results_nlsq.json``

3. Refine bounds if needed

4. Run MCMC for uncertainties:

   .. code-block:: bash

      homodyne --config config.yaml --method mcmc --output-dir ./final

5. Analyze posterior distributions in ``final/mcmc/``

Template Location
-----------------

.. code-block:: bash

   # In package installation
   homodyne/config/templates/homodyne_static_isotropic.yaml

   # Find full path
   python -c "import homodyne; print(homodyne.__file__.replace('__init__.py', 'config/templates/homodyne_static_isotropic.yaml'))"

Line Count
----------

**Static isotropic template:** ~170 lines (clean, production-ready)

See Also
--------

- :doc:`laminar-flow` - For flowing systems (7+2n parameters)
- :doc:`master-template` - For comprehensive reference
- :doc:`index` - Template overview
- :doc:`../theoretical-framework/parameter-models` - Parameter model details
- :doc:`../advanced-topics/nlsq-optimization` - NLSQ workflows
- :doc:`../advanced-topics/mcmc-uncertainty` - MCMC workflows
