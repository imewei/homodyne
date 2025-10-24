Master Template - Comprehensive Reference
==========================================

The master template (``homodyne_master_template.yaml``) provides a comprehensive annotated reference covering **all** parameters and optimization methods. Use this as a reference when customizing configurations.

Purpose
-------

- **Complete parameter documentation** with inline comments and recommendations
- **All optimization methods** (NLSQ, MCMC, streaming, CMC) with full configuration options
- **Reference guide** for understanding available settings and their effects
- **Learning tool** for new users exploring configuration options

When to Use
-----------

- **Learning:** Understanding all available configuration options
- **Customization:** Creating specialized configurations for advanced use cases
- **Reference:** Looking up parameter meanings and recommended values
- **Development:** Building new templates or debugging configurations

Do NOT use this template directly for production analysis - use :doc:`static-isotropic` or :doc:`laminar-flow` instead.

Template Structure
------------------

The master template is organized into logical sections:

.. code-block:: yaml

   # 1. Experimental Data
   experimental_data:
     file_path: "./data/sample/experiment.hdf"

   # 2. Parameter Space - Physical Model Configuration
   parameter_space:
     model: "laminar_flow"  # or "static_isotropic"
     bounds: [...]          # All 7 parameters documented

   # 3. Initial Parameters
   initial_parameters:
     parameter_names: [...]
     active_parameters: [...]
     fixed_parameters: {}

   # 4. Optimization Methods
   optimization:
     method: "nlsq"
     nlsq: {...}            # Trust-region settings
     mcmc: {...}            # MCMC sampling settings
     streaming: {...}       # Large dataset settings
     cmc: {...}             # Multi-angle combination

   # 5. Phi Angle Filtering
   phi_filtering:
     enabled: false
     target_ranges: [...]

   # 6. Performance Optimization
   performance:
     strategy_override: null
     device: {...}

   # 7. Visualization
   plotting: {...}

   # 8. Logging
   logging: {...}

   # 9. Output
   output: {...}

Key Features
------------

**Comprehensive Parameter Documentation**
  Every parameter includes:
    - Physical meaning and units
    - Recommended value ranges
    - Usage notes and warnings
    - Example values

**All Optimization Methods**
  Complete configuration for:
    - **NLSQ:** Trust-region nonlinear least squares (primary method)
    - **MCMC:** Markov chain Monte Carlo with NumPyro/BlackJAX
    - **Streaming:** Unlimited dataset support with checkpointing
    - **CMC:** Covariance matrix combination for multi-angle data

**Parameter Counting Explanation**
  Detailed documentation of:
    - Static Isotropic: 3 + 2n parameters
    - Laminar Flow: 7 + 2n parameters
    - Per-angle scaling: [contrast, offset] for each phi angle

**Advanced Configuration Notes**
  Extended documentation sections covering:
    - Parameter name mapping (``gamma_dot_0`` → ``gamma_dot_t0``)
    - Platform support (Linux/macOS/Windows CPU, Linux-only GPU)
    - Configuration validation and type safety
    - Deprecated configuration warnings

Template Location
-----------------

.. code-block:: bash

   # Find in package installation
   python -c "import homodyne; print(homodyne.__file__.replace('__init__.py', 'config/templates/homodyne_master_template.yaml'))"

   # Or in source repository
   homodyne/config/templates/homodyne_master_template.yaml

Configuration Sections
----------------------

Experimental Data
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   experimental_data:
     file_path: "./data/sample/experiment.hdf"
     # RECOMMENDATION: Use absolute paths for reproducibility

**Purpose:** Specify experimental HDF5 data file location.

Parameter Space
~~~~~~~~~~~~~~~

.. code-block:: yaml

   parameter_space:
     model: "laminar_flow"  # static_isotropic | laminar_flow
     bounds:
       - name: D0
         min: 100.0
         max: 1e5
       # ... (all 7 parameters for laminar flow, or 3 for static)

**Purpose:** Define physical model and parameter bounds.

**Critical:** Parameter count = 3 + 2n (static) or 7 + 2n (laminar), where n = number of filtered phi angles.

Initial Parameters
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   initial_parameters:
     parameter_names: [D0, alpha, D_offset, ...]
     active_parameters: [D0, alpha, ...]  # Optional subset
     fixed_parameters: {D_offset: 10.0}   # Optional fixed values

**Purpose:** Specify which parameters to optimize and initial guesses.

Optimization Methods
~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   optimization:
     method: "nlsq"  # nlsq | mcmc

     nlsq:
       max_iterations: 100
       tolerance: 1e-8
       # Automatic strategy selection based on dataset size

     mcmc:
       num_warmup: 1000
       num_samples: 2000
       num_chains: 4
       backend: "numpyro"

     streaming:
       enable_checkpoints: true
       checkpoint_dir: "./checkpoints"
       # For datasets >100M points

     cmc:
       enable: false
       backend: "jax"
       # For large multi-angle datasets >1M points

**Purpose:** Configure optimization method and advanced features.

**NLSQ Automatic Strategy Selection:**
  - < 1M points → STANDARD (curve_fit)
  - 1M-10M → LARGE (curve_fit_large)
  - 10M-100M → CHUNKED (with progress)
  - > 100M → STREAMING (unlimited)

Phi Angle Filtering
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   phi_filtering:
     enabled: false
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Near 0 degrees"

**Purpose:** Filter phi angles to reduce parameter count (3+2n or 7+2n).

**Effect:** Reduces total parameters by excluding angles outside target ranges.

Performance
~~~~~~~~~~~

.. code-block:: yaml

   performance:
     strategy_override: null  # Automatic selection
     memory_limit_gb: null    # Auto-detect
     device:
       preferred_device: "auto"  # auto | cpu | gpu

**Purpose:** Performance optimization and device selection.

**Platform Support:**
  - CPU: Linux, macOS, Windows
  - GPU: Linux only (CUDA 12.1-12.9)

Usage Example
-------------

**Do NOT use master template directly.** Instead, use it as reference when customizing static or laminar templates:

1. **Copy appropriate template:**

   .. code-block:: bash

      # For static systems
      cp homodyne/config/templates/homodyne_static_isotropic.yaml my_config.yaml

      # For flow systems
      cp homodyne/config/templates/homodyne_laminar_flow.yaml my_config.yaml

2. **Customize using master template as reference:**

   - Look up parameter meanings in master template
   - Find recommended value ranges
   - Understand advanced configuration options
   - Copy relevant sections as needed

3. **Validate configuration:**

   .. code-block:: bash

      # Test configuration (no analysis)
      homodyne --config my_config.yaml --validate-config-only

Line Count
----------

**Master template:** ~240 lines with comprehensive comments and annotations.

Next Steps
----------

- **For production use:** See :doc:`static-isotropic` or :doc:`laminar-flow`
- **For parameter details:** See :doc:`../theoretical-framework/parameter-models`
- **For optimization methods:** See :doc:`../advanced-topics/nlsq-optimization` and :doc:`../advanced-topics/mcmc-uncertainty`

See Also
--------

- :doc:`static-isotropic` - 3+2n parameter template
- :doc:`laminar-flow` - 7+2n parameter template
- :doc:`index` - Template overview and selection guide
- :doc:`../user-guide/configuration` - Configuration system
