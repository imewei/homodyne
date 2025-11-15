homodyne.config - Configuration Management
==========================================

.. automodule:: homodyne.config
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.config`` module provides comprehensive YAML configuration management with parameter validation, bounds management, and type safety. It supports both template-based and modern configuration formats with automatic normalization and intelligent defaults.

**Key Features:**

* **YAML Configuration**: Human-readable configuration files
* **Parameter Management**: Centralized bounds and validation
* **Type Safety**: TypedDict definitions for configuration structures
* **Template Support**: Pre-configured templates for common analyses
* **Auto-Normalization**: Handles legacy and modern formats

Module Structure
----------------

The config module is organized into several submodules:

* :mod:`homodyne.config.manager` - Main configuration loading and management
* :mod:`homodyne.config.parameter_manager` - Parameter bounds and validation
* :mod:`homodyne.config.types` - TypedDict definitions for type safety
* :mod:`homodyne.config.templates` - Pre-configured YAML templates

Submodules
----------

homodyne.config.manager
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.config.manager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

Main configuration loading and management.

**Key Classes:**

* ``ConfigManager`` - Configuration file loader and validator

**Usage Example:**

.. code-block:: python

   from homodyne.config import ConfigManager

   # Load configuration from YAML
   config_mgr = ConfigManager('homodyne_config.yaml')

   # Access configuration sections
   data_config = config_mgr.get_data_config()
   opt_config = config_mgr.get_optimization_config()
   device_config = config_mgr.get_device_config()

   # Get parameter bounds
   bounds = config_mgr.get_parameter_bounds()

   # Validate configuration
   is_valid, errors = config_mgr.validate()
   if not is_valid:
       print(f"Configuration errors: {errors}")

**Load from Dictionary:**

.. code-block:: python

   # Load from dict (for programmatic configuration)
   config_dict = {
       'analysis_type': 'static_isotropic',
       'experimental_data': {
           'file_path': './data/experiment.hdf',
           'q': 0.01
       },
       'optimization': {
           'method': 'nlsq'
       }
   }

   config_mgr = ConfigManager.from_dict(config_dict)

**Extract Initial Parameters (v2.1.0):**

.. versionadded:: 2.1.0
   ``get_initial_parameters()`` method for MCMC initialization.

.. code-block:: python

   from homodyne.config import ConfigManager
   from homodyne.config.parameter_space import ParameterSpace

   # Load configuration
   config_mgr = ConfigManager('config.yaml')

   # Get parameter space
   parameter_space = ParameterSpace.from_config(config_mgr.config)

   # Get initial parameter values
   initial_values = config_mgr.get_initial_parameters(
       parameter_space=parameter_space,
       active_parameters=['D0', 'alpha', 'D_offset']
   )

   print(initial_values)
   # Output: {'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}
   # If not in config: {'D0': 5050.0, 'alpha': 1.05, 'D_offset': 50.05}  (mid-points)

**Behavior:**

* Loads from ``initial_parameters.values`` in YAML if present
* Falls back to mid-point of bounds: ``(min + max) / 2``
* Supports CLI overrides via ``config_override`` section

homodyne.config.parameter_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.config.parameter_manager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

Centralized parameter bounds, validation, and management.

**Key Classes:**

* ``ParameterManager`` - Parameter management engine

**Usage Example:**

.. code-block:: python

   from homodyne.config.parameter_manager import ParameterManager

   # Create parameter manager
   pm = ParameterManager(config_dict, analysis_type='laminar_flow')

   # Get parameter bounds
   bounds = pm.get_parameter_bounds(['D0', 'alpha', 'D_offset'])
   print(f"D0 bounds: {bounds['D0']}")  # {'min': 100.0, 'max': 100000.0}

   # Get active parameters
   active_params = pm.get_active_parameters()

   # Validate parameters
   params = {'D0': 1000.0, 'alpha': 0.8, 'D_offset': 10.0}
   result = pm.validate_physical_constraints(params, severity_level='warning')

   if result['valid']:
       print("Parameters are valid")
   else:
       print(f"Validation warnings: {result['warnings']}")

**Custom Bounds:**

.. code-block:: python

   # Override default bounds in YAML
   config = {
       'parameter_space': {
           'bounds': [
               {'name': 'D0', 'min': 500.0, 'max': 5000.0},
               {'name': 'alpha', 'min': 0.5, 'max': 1.0}
           ]
       }
   }

   pm = ParameterManager(config, analysis_type='static_isotropic')
   custom_bounds = pm.get_parameter_bounds(['D0', 'alpha'])

homodyne.config.parameter_space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.config.parameter_space
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

.. versionadded:: 2.1.0
   ParameterSpace class for config-driven parameter bounds and prior distributions in MCMC.

Configuration-driven parameter management for MCMC Bayesian inference.

**Key Classes:**

* ``ParameterSpace`` - Parameter bounds and prior distributions container

**Usage Example:**

.. code-block:: python

   from homodyne.config.parameter_space import ParameterSpace
   from homodyne.config.manager import ConfigManager

   # Load from configuration file
   config_manager = ConfigManager("config.yaml")
   parameter_space = ParameterSpace.from_config(config_manager.config)

   # Access bounds
   print(f"D0 bounds: {parameter_space.get_bounds('D0')}")
   # Output: {'min': 100.0, 'max': 10000.0}

   # Access priors
   print(f"D0 prior: {parameter_space.get_prior('D0')}")
   # Output: {'type': 'TruncatedNormal', 'mu': 1000.0, 'sigma': 500.0}

   # Get all parameter names
   param_names = parameter_space.get_parameter_names()
   print(f"Parameters: {param_names}")
   # Output: ['D0', 'alpha', 'D_offset']

**Configuration Structure:**

.. code-block:: yaml

   # In config.yaml
   parameter_space:
     bounds:
       - name: D0
         min: 100.0
         max: 10000.0
       - name: alpha
         min: 0.1
         max: 2.0
       - name: D_offset
         min: 0.1
         max: 100.0
     priors:
       D0:
         type: TruncatedNormal
         mu: 1000.0
         sigma: 500.0
       alpha:
         type: TruncatedNormal
         mu: 1.0
         sigma: 0.3
       D_offset:
         type: TruncatedNormal
         mu: 10.0
         sigma: 5.0

**Supported Prior Distributions:**

* ``TruncatedNormal`` - Truncated Gaussian (mu, sigma, truncated at bounds)
* ``Uniform`` - Uniform distribution (uses bounds directly)
* ``LogNormal`` - Log-normal distribution (mu, sigma on log scale)

**Integration with MCMC:**

.. code-block:: python

   from homodyne.optimization import fit_mcmc_jax

   # parameter_space automatically used for bounds and priors
   result = fit_mcmc_jax(
       data=data,
       t1=t1,
       t2=t2,
       phi=phi,
       q=0.01,
       parameter_space=parameter_space,  # Loaded from config
       initial_values={'D0': 1234.5, 'alpha': 0.567, 'D_offset': 12.34}
   )

homodyne.config.types
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.config.types
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

TypedDict definitions for configuration structures.

**Key Types:**

* ``HomodyneConfig`` - Top-level configuration
* ``ExperimentalDataConfig`` - Data loading configuration
* ``OptimizationConfig`` - Optimization settings
* ``ParameterSpaceConfig`` - Parameter bounds and constraints
* ``BoundDict`` - Parameter bound specification

**Usage Example:**

.. code-block:: python

   from homodyne.config.types import HomodyneConfig, BoundDict
   from typing import TypedDict

   # Type-safe configuration construction
   config: HomodyneConfig = {
       'analysis_type': 'static_isotropic',
       'experimental_data': {
           'file_path': './data/experiment.hdf',
           'q': 0.01,
           'L': 1.0,
           'dt': 0.001
       },
       'optimization': {
           'method': 'nlsq',
           'max_iterations': 100
       }
   }

Configuration Templates
-----------------------

**Available Templates:**

1. **homodyne_static_isotropic.yaml** - Static isotropic diffusion

   .. code-block:: bash

      homodyne --config homodyne/config/templates/homodyne_static_isotropic.yaml

2. **homodyne_laminar_flow.yaml** - Laminar flow (diffusion + shear)

   .. code-block:: bash

      homodyne --config homodyne/config/templates/homodyne_laminar_flow.yaml

3. **homodyne_master_template.yaml** - Comprehensive template with all options

   .. code-block:: bash

      homodyne --config homodyne/config/templates/homodyne_master_template.yaml

Configuration Structure
-----------------------

**Complete YAML Configuration:**

.. code-block:: yaml

   # Analysis Type
   analysis_type: 'static_isotropic'  # or 'laminar_flow'

   # Experimental Data
   experimental_data:
     file_path: "./data/experiment.hdf"
     q: 0.01            # Momentum transfer (Å⁻¹)
     L: 1.0             # Sample thickness (mm)
     dt: 0.001          # Time resolution (s)

   # Parameter Space (v2.1.0: includes priors for MCMC)
   parameter_space:
     bounds:
       - name: D0
         min: 100.0
         max: 1.0e5
       - name: alpha
         min: 0.1
         max: 2.0
       - name: D_offset
         min: 0.0
         max: 100.0
     priors:  # NEW in v2.1.0 for MCMC
       D0:
         type: TruncatedNormal
         mu: 1000.0
         sigma: 500.0
       alpha:
         type: TruncatedNormal
         mu: 1.0
         sigma: 0.3
       D_offset:
         type: TruncatedNormal
         mu: 10.0
         sigma: 5.0

   # Initial Parameters (v2.1.0: optional, for MCMC initialization)
   initial_parameters:
     parameter_names: [D0, alpha, D_offset]
     values: [1234.5, 0.567, 12.34]  # From NLSQ results (manual copy)
     # If 'values' not provided, defaults to mid-point of bounds

   # Optimization Settings
   optimization:
     method: 'nlsq'              # nlsq | mcmc
     max_iterations: 100
     tolerance: 1.0e-6
     streaming:
       enable_checkpoints: true
       checkpoint_dir: "./checkpoints"

   # MCMC Settings (if method='mcmc')
   mcmc:
     num_warmup: 500
     num_samples: 1000
     num_chains: 4
     target_accept_prob: 0.8
     # NEW in v2.1.0: Automatic NUTS/CMC selection thresholds
     min_samples_for_cmc: 15        # Parallelism threshold
     memory_threshold_pct: 0.30     # Memory threshold (30%)
     dense_mass_matrix: false       # Diagonal (fast) vs full covariance

   # Angle Filtering (Optional)
   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Near 0 degrees"

   # Device Configuration
   device:
     type: 'auto'         # auto | cpu | gpu
     gpu_id: 0
     cpu_threads: null

   # Performance Settings
   performance:
     strategy_override: null
     memory_limit_gb: null
     enable_progress: true

   # Output Settings
   output:
     output_dir: "./results"
     save_plots: true
     plot_format: 'png'

Parameter Name Mapping
----------------------

**Automatic Name Mapping:**

Configuration files use human-readable names that are automatically mapped to code names:

* ``gamma_dot_0`` → ``gamma_dot_t0``
* ``phi_0`` → ``phi0``

.. code-block:: yaml

   # Configuration (human-readable)
   initial_parameters:
     initial_values:
       gamma_dot_0: 0.1
       phi_0: 45.0

   # Internally mapped to:
   # gamma_dot_t0: 0.1
   # phi0: 45.0

Configuration Validation
------------------------

**Validation Levels:**

1. **ERROR** - Critical issues that prevent execution
2. **WARNING** - Unusual but potentially valid configurations
3. **INFO** - Informational messages

**Usage Example:**

.. code-block:: python

   from homodyne.config import ConfigManager

   config_mgr = ConfigManager('config.yaml')

   # Validate with warnings
   is_valid, messages = config_mgr.validate(severity_level='warning')

   for msg in messages:
       print(f"{msg['level']}: {msg['message']}")

Loading Strategies
------------------

**File Path Priority:**

1. Absolute path: ``/home/user/config.yaml``
2. Relative to current directory: ``./config.yaml``
3. Relative to homodyne package: ``homodyne/config/templates/config.yaml``

**Format Auto-Detection:**

.. code-block:: python

   # Handles both legacy and modern formats
   config_mgr = ConfigManager('legacy_config.yaml')

   # Legacy format (auto-converted):
   # experimental_data:
   #   data_folder_path: "./data/"
   #   data_file_name: "experiment.hdf"

   # Modern format:
   # experimental_data:
   #   file_path: "./data/experiment.hdf"

CLI Integration
---------------

**Command-Line Overrides:**

.. code-block:: bash

   # Use configuration file
   homodyne --config config.yaml

   # Override data file
   homodyne --config config.yaml --data-file /path/to/other_data.hdf

   # Override output directory
   homodyne --config config.yaml --output-dir ./custom_results

   # Override optimization method
   homodyne --config config.yaml --method mcmc

See Also
--------

* :doc:`../user-guide/configuration` - Configuration guide
* :doc:`../configuration-templates/index` - Template documentation
* :doc:`../user-guide/configuration` - Parameter management guide
* :doc:`cli` - CLI that uses configuration

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.config import (
       ConfigManager,
       ParameterManager,
   )

**Related Functions:**

* :func:`homodyne.cli.commands.load_config` - CLI configuration loading
* :func:`homodyne.optimization.fit_nlsq_jax` - Uses configuration
* :func:`homodyne.data.xpcs_loader.XPCSDataLoader` - Uses data configuration
