Configuration Guide
====================

This guide explains Homodyne's YAML configuration system and parameter counting.

Configuration Basics
--------------------

Homodyne uses YAML configuration files to specify analysis parameters, optimization methods, and data loading.

**Basic structure:**

.. code-block:: yaml

   experimental_data:
     file_path: "./data/experiment.hdf"

   parameter_space:
     model: "static_isotropic"     # or "laminar_flow"
     bounds: [...]

   initial_parameters:
     parameter_names: [...]

   optimization:
     method: "nlsq"                 # or "mcmc"
     nlsq: {...}
     mcmc: {...}

   output:
     directory: "./results"

Critical Concept: Parameter Counting
-------------------------------------

Homodyne uses **per-angle scaling parameters** in addition to physical parameters. This determines the total parameter count.

**Static Isotropic Model: 3 + 2n parameters**

- **3 physical parameters:**
  - D₀: Diffusion coefficient [Å²/s]
  - α: Power-law exponent [dimensionless]
  - D_offset: Diffusion offset [Å²/s]

- **2 scaling parameters per filtered phi angle:**
  - contrast: Amplitude scaling [dimensionless]
  - offset: Baseline offset [dimensionless]

- **Total:** 3 + 2×(number of filtered angles)

**Example with 3 angles:**

.. math::

   \text{Total parameters} = 3 + 2 \times 3 = 9

**Laminar Flow Model: 7 + 2n parameters**

- **7 physical parameters:**
  - D₀: Diffusion coefficient [Å²/s]
  - α: Diffusion power-law exponent [dimensionless]
  - D_offset: Diffusion offset [Å²/s]
  - γ̇₀: Initial shear rate [s⁻¹]
  - β: Shear power-law exponent [dimensionless]
  - γ̇_offset: Shear rate offset [s⁻¹]
  - φ₀: Initial angle [degrees]

- **2 scaling parameters per filtered phi angle:**
  - contrast: Amplitude scaling [dimensionless]
  - offset: Baseline offset [dimensionless]

- **Total:** 7 + 2×(number of filtered angles)

**Example with 3 angles:**

.. math::

   \text{Total parameters} = 7 + 2 \times 3 = 13

Impact of Angle Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^

The per-angle scaling approach means that **filtering angles directly reduces the total parameter count**:

- **No filtering:** All angles analyzed separately
- **With filtering:** Only selected angles used, reducing n

Example comparison:

- 10 total phi angles, no filtering → 3 + 2×10 = 23 parameters (static) or 7 + 2×10 = 27 parameters (laminar)
- 10 angles, filtered to 3 ranges → 3 + 2×3 = 9 parameters (static) or 7 + 2×3 = 13 parameters (laminar)

**Recommendation:** Use angle filtering to reduce parameter count and improve convergence.

Configuration Sections
----------------------

Experimental Data
^^^^^^^^^^^^^^^^^

Specifies where to load experimental HDF5 data:

.. code-block:: yaml

   experimental_data:
     file_path: "./data/sample/experiment.hdf"

**Options:**

- ``file_path``: Path to HDF5 file (absolute or relative)
- **Recommendation:** Use absolute paths for reproducibility

Parameter Space
^^^^^^^^^^^^^^^

Defines physical model and parameter bounds:

.. code-block:: yaml

   parameter_space:
     model: "static_isotropic"    # or "laminar_flow"

     bounds:
       - name: D0
         min: 100.0               # Minimum bound
         max: 1e5                 # Maximum bound
       - name: alpha
         min: 0.0
         max: 2.0
       # ... additional parameters

**For Static Isotropic (required):**

- D0, alpha, D_offset

**For Laminar Flow (required):**

- D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi_0

**Notes:**

- Bounds must match your physical system
- See :doc:`../theoretical-framework/parameter-models` for recommended ranges
- Parameter names in config map to code names (e.g., ``gamma_dot_0`` → ``gamma_dot_t0``)

Initial Parameters
^^^^^^^^^^^^^^^^^^

Specifies which parameters to optimize:

.. code-block:: yaml

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset

     # Optional: Specify active subset
     active_parameters:
       - D0
       - alpha

     # Optional: Fix specific parameters
     fixed_parameters:
       D_offset: 10.0

**Strategies:**

1. **Optimize all:** List all parameters in ``parameter_names``
2. **Active subset:** Use ``active_parameters`` to optimize subset only
3. **Fix parameters:** Use ``fixed_parameters`` for known values

**Recommendation:** Start with critical parameters, add complexity gradually.

Optimization Methods
^^^^^^^^^^^^^^^^^^^^

**NLSQ (Nonlinear Least Squares - Default)**

.. code-block:: yaml

   optimization:
     method: "nlsq"

     nlsq:
       max_iterations: 100           # Maximum iterations
       tolerance: 1e-8              # Convergence tolerance
       trust_region_scale: 1.0      # Trust region scaling

**When to use:** Primary optimization method for point estimates

**MCMC (Markov Chain Monte Carlo)**

.. code-block:: yaml

   optimization:
     method: "mcmc"

     mcmc:
       num_warmup: 1000             # Warmup samples
       num_samples: 2000            # Posterior samples
       num_chains: 4                # Parallel chains
       progress_bar: true           # Show progress
       backend: "numpyro"           # or "blackjax"

**When to use:** Uncertainty quantification, posterior distributions

See :doc:`../advanced-topics/mcmc-uncertainty` for details.

Angle Filtering
^^^^^^^^^^^^^^^

Filters experimental data to specific phi angle ranges before optimization:

.. code-block:: yaml

   phi_filtering:
     enabled: true

     target_ranges:
       - min_angle: -10.0           # Degrees
         max_angle: 10.0
         description: "Near 0 degrees"

       - min_angle: 85.0
         max_angle: 95.0
         description: "Near 90 degrees"

**Benefits:**

- Reduces parameter count (fewer angles → 2n scaling parameters)
- Improves convergence (focus on specific angular regimes)
- Handles anisotropy better (separate analysis by angle)

**Angles are normalized to [-180°, 180°]**, handling wrapping automatically.

See :doc:`../advanced-topics/angle-filtering` for details.

Performance and Strategy Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   performance:
     # Strategy selection (automatic by default)
     strategy_override: null           # null for auto, or "standard"|"large"|"chunked"|"streaming"
     memory_limit_gb: null            # Auto-detect if null
     enable_progress: true            # Show progress bars

     device:
       preferred_device: "auto"       # "auto"|"cpu"|"gpu"
       gpu_memory_fraction: 0.9       # GPU memory limit

**Strategy Selection (Automatic):**

- **< 1M points:** STANDARD (fast, low memory)
- **1M-10M points:** LARGE (medium memory)
- **10M-100M points:** CHUNKED (higher memory)
- **> 100M points:** STREAMING (constant memory, with checkpoints)

**Recommendation:** Leave ``strategy_override: null`` for automatic selection.

See :doc:`../advanced-topics/streaming-optimization` for streaming details.

Template Selection
------------------

Homodyne provides three comprehensive configuration templates:

**Master Template**

Complete reference with ALL parameters:

.. code-block:: bash

   cp homodyne/config/templates/homodyne_master_template.yaml my_config.yaml

**Static Isotropic Template**

Optimized for static, isotropic systems (3+2n parameters):

.. code-block:: bash

   cp homodyne/config/templates/homodyne_static_isotropic.yaml my_config.yaml

**Laminar Flow Template**

Optimized for flowing systems with angle filtering (7+2n parameters):

.. code-block:: bash

   cp homodyne/config/templates/homodyne_laminar_flow.yaml my_config.yaml

See :doc:`../configuration-templates/index` for detailed template documentation.

Complete Configuration Example
-------------------------------

**Static isotropic with angle filtering:**

.. code-block:: yaml

   experimental_data:
     file_path: "./data/experiment.hdf"

   parameter_space:
     model: "static_isotropic"

     bounds:
       - name: D0
         min: 100.0
         max: 1e5
       - name: alpha
         min: 0.0
         max: 2.0
       - name: D_offset
         min: -100.0
         max: 100.0

   initial_parameters:
     parameter_names: [D0, alpha, D_offset]
     active_parameters: [D0, alpha]
     fixed_parameters:
       D_offset: 10.0

   optimization:
     method: "nlsq"
     nlsq:
       max_iterations: 100
       tolerance: 1e-8

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0

   performance:
     strategy_override: null
     enable_progress: true

   output:
     directory: "./results"

This configuration:

- Uses 3+2×1 = 5 parameters (3 physical + 2 for single filtered angle)
- Optimizes D0 and alpha
- Fixes D_offset to 10.0
- Uses NLSQ optimization
- Filters to near 0° angles only

Advanced Configuration Topics
------------------------------

**Streaming Optimization for Large Datasets**

.. code-block:: yaml

   optimization:
     streaming:
       enable_checkpoints: true
       checkpoint_dir: "./checkpoints"
       checkpoint_frequency: 10
       max_retries_per_batch: 2

See :doc:`../advanced-topics/streaming-optimization`.

**CMC (Covariance Matrix Combination)**

.. code-block:: yaml

   optimization:
     cmc:
       enable: true
       backend: "jax"
       diagonal_correction: true

See :doc:`../advanced-topics/cmc-large-datasets`.

**GPU Acceleration**

.. code-block:: yaml

   performance:
     device:
       preferred_device: "gpu"
       gpu_memory_fraction: 0.9

See :doc:`../advanced-topics/gpu-acceleration`.

Validation and Debugging
------------------------

**Validate YAML syntax:**

.. code-block:: bash

   python -c "import yaml; yaml.safe_load(open('config.yaml'))"

**Check configuration loading:**

.. code-block:: python

   from homodyne.config import ConfigManager
   config = ConfigManager("config.yaml")
   print(config.get_model())
   print(config.get_parameter_bounds())

**Preview parameter count:**

.. code-block:: bash

   # Count angles after filtering
   python -c "
   from homodyne.config import ConfigManager
   config = ConfigManager('config.yaml')
   model = config.get_model()
   n_angles = config.get_filtered_angle_count()
   if model == 'static_isotropic':
       total = 3 + 2*n_angles
   else:
       total = 7 + 2*n_angles
   print(f'Total parameters: {total} ({n_angles} angles)')
   "

Next Steps
----------

- :doc:`cli-usage` - Run analyses with your configuration
- :doc:`examples` - Real-world workflow examples
- :doc:`../advanced-topics/index` - Advanced optimization techniques
- :doc:`../theoretical-framework/parameter-models` - Detailed parameter descriptions

See Also
--------

- :doc:`../configuration-templates/index` - Template documentation
- :doc:`../api-reference/config` - Configuration API reference
- :doc:`../developer-guide/testing` - Testing configurations
