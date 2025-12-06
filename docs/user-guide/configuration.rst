Configuration Guide
===================

Homodyne uses YAML configuration files to control all aspects of your analysis.
This guide explains every configuration option and how to set them for your specific use case.

Configuration File Format
=========================

Homodyne configurations are written in YAML (a human-friendly data format).
Basic YAML syntax:

.. code-block:: yaml

   # Comments start with #
   key: value                        # String values
   number: 42                        # Numbers
   decimal: 3.14
   list:
     - item1
     - item2
   nested:
     sub_key: value

Minimal Configuration Example
=============================

Here's the absolute minimum configuration to run an analysis:

.. code-block:: yaml

   data:
     path: /path/to/xpcs_data.h5
     h5_keys:
       t1: "entry/data/t1"
       t2: "entry/data/t2"
       phi: "entry/data/phi"
       c2: "entry/data/c2"

   analysis:
     mode: static
     n_angles: 3

   optimization:
     method: nlsq
     initial_parameters:
       values: [1000.0, 0.5, 100.0]

   output:
     results_dir: ./results

Complete Configuration Reference
=================================

Data Section
------------

Specifies where and how to load XPCS data.

.. code-block:: yaml

   data:
     path: /path/to/xpcs_data.h5          # HDF5 file path (required)
     h5_keys:                              # HDF5 dataset keys (required)
       t1: "entry/data/t1"
       t2: "entry/data/t2"
       phi: "entry/data/phi"
       c2: "entry/data/c2"
     # Optional: Preprocessing
     t1_slice: ":"                        # Slice time1 (e.g., ":10" = first 10)
     t2_slice: ":"                        # Slice time2
     phi_slice: ":"                       # Slice angles
     angle_filtering:
       enabled: false                     # Filter by angle range
       min_angle: 0
       max_angle: 180

**Key Details:**

- ``path``: Absolute or relative to working directory
- ``h5_keys``: Must match your HDF5 file structure
- Use ``h5dump -H file.h5`` to inspect structure
- Slicing uses Python notation: ``":10"`` = first 10, ``"10:20"`` = 10-19

Find Your HDF5 Keys
~~~~~~~~~~~~~~~~~~~

Check your HDF5 structure:

.. code-block:: bash

   h5dump -H /path/to/your/file.h5

Look for datasets like:

.. code-block:: text

   /entry/data/c2 Dataset {1000, 100, 100}
   /entry/data/phi Dataset {3}
   /entry/data/t1 Dataset {100}
   /entry/data/t2 Dataset {100}

Use the paths as ``h5_keys``.

Analysis Section
----------------

Defines the physics model and analysis parameters.

.. code-block:: yaml

   analysis:
     mode: static                         # Analysis mode: static, laminar_flow
     n_angles: 3                          # Number of azimuthal angles
     # Optional: Data filtering
     remove_bad_angles: true              # Remove low-intensity angles
     phi_std_threshold: 2.0               # Standard deviations for rejection

**Mode Comparison:**

.. table::
   :widths: 30 20 20 30

   +------------------+----------+----------+----------------------------------+
   | Feature          | Static   | Laminar  | Notes                            |
   +==================+==========+==========+==================================+
   | Physical params  | 3        | 7        | Total = 3 + 2*n_angles (v2.4)   |
   +------------------+----------+----------+----------------------------------+
   | Time dependence  | D(t)     | D(t)     | D(t) = D₀*t^α + D_offset        |
   +------------------+----------+----------+----------------------------------+
   | Shear dependence | No       | Yes      | γ̇(t) = γ̇₀*t^β + γ̇_offset    |
   +------------------+----------+----------+----------------------------------+
   | Anisotropy       | No       | Yes      | Via φ₀ parameter                |
   +------------------+----------+----------+----------------------------------+
   | Best for         | Isotropic | Flows   | Choose based on your experiment  |
   +------------------+----------+----------+----------------------------------+

Optimization Section
--------------------

Controls how parameters are estimated from data.

.. code-block:: yaml

   optimization:
     method: nlsq                         # NLSQ or MCMC
     initial_parameters:
       values: [1000.0, 0.5, 100.0]      # Initial guesses
       bounds:                            # Parameter bounds
         D0: [100.0, 10000.0]
         alpha: [0.0, 1.0]
         D_offset: [0.0, 500.0]

**Initial Parameters:**

For **static** mode, provide 3 values:

.. code-block:: yaml

   values: [D0, alpha, D_offset]

For **laminar_flow** mode, provide 7 values:

.. code-block:: yaml

   values: [D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

**Parameter Bounds:**

Must be: ``[lower, upper]`` with lower < upper

.. code-block:: yaml

   bounds:
     D0: [100.0, 10000.0]       # Diffusion coefficient
     alpha: [0.0, 1.5]          # Time exponent (0=constant, 1=normal)
     D_offset: [0.0, 500.0]     # Offset term
     gamma_dot_t0: [0.1, 100.0] # Shear rate coefficient (laminar only)
     beta: [0.0, 1.0]           # Shear time exponent (laminar only)
     gamma_dot_t_offset: [0.0, 10.0]  # Shear offset (laminar only)
     phi0: [-3.14, 3.14]        # Anisotropy angle (laminar only)

**Physical Units:**

.. table::
   :widths: 30 20 50

   +--------------------+-------+------------------------------------+
   | Parameter          | Units | Typical Range                      |
   +====================+=======+====================================+
   | D0                 | μm²/s | 100-10000                          |
   +--------------------+-------+------------------------------------+
   | alpha              | none  | 0.0-1.5                            |
   +--------------------+-------+------------------------------------+
   | D_offset           | μm²/s | 0-500                              |
   +--------------------+-------+------------------------------------+
   | gamma_dot_t0       | s⁻¹   | 0.1-100                            |
   +--------------------+-------+------------------------------------+
   | beta               | none  | 0.0-1.0                            |
   +--------------------+-------+------------------------------------+
   | gamma_dot_t_offset | s⁻¹   | 0-10                               |
   +--------------------+-------+------------------------------------+
   | phi0               | rad   | -π to π                            |
   +--------------------+-------+------------------------------------+

Per-Angle Scaling (v2.4.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Important:** Per-angle scaling is now mandatory in v2.4.0+.

Each angle has its own scaling parameters:

- **Number of angles:** n_angles
- **Per-angle contrast parameters:** n_angles
- **Per-angle offset parameters:** n_angles
- **Physical parameters:** 3 (static) or 7 (laminar_flow)
- **Total parameters:** 2 × n_angles + physical_params

**Example with 3 angles:**

.. code-block:: text

   Total parameters: 2*3 + 3 = 9
   [contrast₀, contrast₁, contrast₂, offset₀, offset₁, offset₂, D₀, α, D_offset]

This means:

- Each angle has independent intensity scaling
- Physical parameters (D₀, α, D_offset) are shared across all angles
- More robust fits for heterogeneous data

NLSQ Optimization Section
~~~~~~~~~~~~~~~~~~~~~~~~~

Additional options for NLSQ method:

.. code-block:: yaml

   optimization:
     method: nlsq
     nlsq:
       max_iterations: 100        # Maximum iterations
       tolerance: 1e-6            # Convergence tolerance
       loss: "linear"             # linear or huber (robust)
       verbose: false             # Print iterations

MCMC Inference Section
~~~~~~~~~~~~~~~~~~~~~~

Additional options for MCMC method:

.. code-block:: yaml

   optimization:
     method: mcmc
     mcmc:
       n_samples: 2000            # Posterior samples
       n_warmup: 1000             # Warmup iterations
       n_chains: 4                # Number of MCMC chains
       backend: "multiprocessing" # multiprocessing or pjit

Output Section
--------------

Specifies where results are saved.

.. code-block:: yaml

   output:
     results_dir: ./homodyne_results    # Output directory
     save_plots: true                   # Save visualization plots
     plot_format: png                   # png or pdf
     save_trace: true                   # Save optimization trace
     save_posterior: true               # Save MCMC posteriors

**Output Files Generated:**

.. table::
   :widths: 40 60

   +----------------------+-----------------------------------------------+
   | File                 | Contents                                      |
   +======================+===============================================+
   | results.json         | Best-fit parameters and uncertainties        |
   +----------------------+-----------------------------------------------+
   | residuals.png        | Residual analysis                             |
   +----------------------+-----------------------------------------------+
   | c2_fit.png           | Two-time correlation visualization            |
   +----------------------+-----------------------------------------------+
   | correlation_heatmap  | Heatmap of c₂(t₁, t₂)                        |
   +----------------------+-----------------------------------------------+
   | convergence.json     | Optimization convergence history              |
   +----------------------+-----------------------------------------------+

Advanced Configuration
======================

Logging Configuration
---------------------

.. code-block:: yaml

   logging:
     level: INFO                 # DEBUG, INFO, WARNING, ERROR
     file: homodyne.log         # Log file (optional)
     console: true              # Log to console

Memory and Performance
----------------------

.. code-block:: yaml

   performance:
     use_jit: true              # Use JAX JIT compilation
     chunk_size: 1000           # Chunk size for large datasets
     n_workers: 4               # Number of worker processes (MCMC)
     device: cpu                # cpu (gpu not supported in v2.3.0+)

Example Configurations
======================

Minimal Static Mode
-------------------

.. code-block:: yaml

   data:
     path: /data/xpcs.h5
     h5_keys:
       c2: entry/data/c2
       t1: entry/data/t1
       t2: entry/data/t2
       phi: entry/data/phi

   analysis:
     mode: static
     n_angles: 3

   optimization:
     method: nlsq
     initial_parameters:
       values: [1000, 0.5, 100]

   output:
     results_dir: ./results

Complete Laminar Flow Configuration
------------------------------------

.. code-block:: yaml

   data:
     path: /data/xpcs_flow.h5
     h5_keys:
       c2: entry/data/c2
       t1: entry/data/t1
       t2: entry/data/t2
       phi: entry/data/phi

   analysis:
     mode: laminar_flow
     n_angles: 5
     remove_bad_angles: true

   optimization:
     method: nlsq
     initial_parameters:
       values: [1500, 0.5, 150, 10.0, 0.5, 1.0, 0.0]
       bounds:
         D0: [500, 5000]
         alpha: [0, 1]
         D_offset: [0, 500]
         gamma_dot_t0: [1, 50]
         beta: [0, 1]
         gamma_dot_t_offset: [0, 10]
         phi0: [-3.14, 3.14]

   output:
     results_dir: ./laminar_results
     save_plots: true
     plot_format: png

Configuration for MCMC
----------------------

Start with NLSQ, then add MCMC:

.. code-block:: yaml

   # First run with:
   optimization:
     method: nlsq
     initial_parameters:
       values: [1234.5, 0.567, 123.4]

   # Get best-fit, then switch to:
   optimization:
     method: mcmc
     initial_parameters:
       values: [1234.5, 0.567, 123.4]  # From NLSQ output
     mcmc:
       n_samples: 2000
       n_warmup: 1000
       n_chains: 4

Validation and Testing
======================

Validate Your Configuration
---------------------------

.. code-block:: bash

   homodyne-config --validate my_config.yaml

This checks:

- YAML syntax
- Required fields
- Data file accessibility
- Parameter bounds logical consistency
- HDF5 keys exist

Common Configuration Errors
---------------------------

**Error: "Parameter out of bounds"**

Initial values must be within bounds:

.. code-block:: yaml

   values: [1000]           # ✓ Within bounds
   bounds:
     D0: [100, 10000]

   values: [50]             # ✗ Below lower bound
   bounds:
     D0: [100, 10000]

**Error: "Cannot find HDF5 key"**

.. code-block:: bash

   # Verify your keys:
   h5ls /path/to/file.h5

   # Result might be:
   #  entry/data/c2 Dataset {1000, 100, 100}

   # Then in config:
   h5_keys:
     c2: entry/data/c2  # ✓ Correct

**Error: "Wrong number of initial values"**

For static mode: 3 values (D0, alpha, D_offset)
For laminar_flow mode: 7 values (+ 4 shear parameters)

Tips and Best Practices
=======================

**1. Start Conservative**

Use wider bounds than you think necessary:

.. code-block:: yaml

   bounds:
     alpha: [0.0, 2.0]   # Wider range
     # rather than
     # alpha: [0.4, 0.6]

**2. Good Initial Guesses**

Closer to true values = faster convergence:

.. code-block:: yaml

   values: [1200, 0.5, 100]  # Based on literature/similar samples

**3. Physics Constraints**

Respect physical bounds:

.. code-block:: text

   D0 > 0          Always positive (diffusion)
   0 < alpha < 2   Typical range for dynamics
   D_offset >= 0   Non-negative offset

**4. Per-Angle Scaling**

Always enabled in v2.4.0+. Improves fit for:

- Heterogeneous intensity distributions
- Multiple detector arms
- Complex sample geometries

Configuration Migration
=======================

If upgrading from v2.3.0 to v2.4.0:

**Change:** Per-angle scaling is now mandatory

.. code-block:: yaml

   # Old (v2.3.0) - optional:
   # per_angle_scaling: true

   # New (v2.4.0) - always true, remove this line

   # Everything else stays the same!

Template Configurations
=======================

Generate templates with:

.. code-block:: bash

   homodyne-config --mode static      # Static mode template
   homodyne-config --mode laminar_flow # Laminar flow template

Then edit as needed for your specific experiment.

Next Steps
==========

- :doc:`./examples` - See configuration in action
- :doc:`./cli` - Run your configured analysis
- :doc:`../research/theoretical_framework` - Understand the physics
- :doc:`../configuration/options` - Complete option reference
