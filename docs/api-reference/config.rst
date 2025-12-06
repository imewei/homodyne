Configuration Module
====================

The :mod:`homodyne.config` module provides comprehensive configuration management for homodyne scattering analysis, including YAML-based configuration, parameter registry, and parameter space management.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Key Components**:

- **ConfigManager**: YAML configuration file management
- **ParameterRegistry**: Centralized parameter metadata (bounds, defaults, units)
- **ParameterSpace**: Prior distributions and parameter transformations
- **Types**: Type definitions for configuration structures

**Configuration Philosophy**:

- YAML-first configuration for reproducibility
- Validated parameter bounds and defaults
- Template-based configuration generation
- Per-angle scaling support (mandatory in v2.4.0+)

Module Contents
---------------

.. automodule:: homodyne.config
   :members:
   :undoc-members:
   :show-inheritance:

Primary Classes
~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.config.ConfigManager
   homodyne.config.ParameterRegistry
   homodyne.config.ParameterSpace

Configuration Manager
---------------------

YAML-based configuration file management.

.. automodule:: homodyne.config.manager
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

**Loading Configuration**::

    from homodyne.config import ConfigManager

    # Load from YAML file
    config = ConfigManager.from_yaml("analysis_config.yaml")

    # Access configuration sections
    mode = config.analysis_mode  # 'static' or 'laminar_flow'
    data_path = config.data.hdf5_path
    initial_params = config.optimization.initial_parameters

    # Get parameter bounds
    bounds = config.get_parameter_bounds()

**Creating Configuration**::

    from homodyne.config import load_xpcs_config

    # Load configuration dictionary
    config_dict = load_xpcs_config("config.yaml")

    # Access nested sections
    analysis_mode = config_dict['analysis']['mode']
    num_warmup = config_dict['optimization']['mcmc']['num_warmup']

Configuration Structure
~~~~~~~~~~~~~~~~~~~~~~~

**YAML Configuration Format**::

    analysis:
      mode: static  # or laminar_flow
      per_angle_scaling: true  # Mandatory in v2.4.0+

    data:
      hdf5_path: /path/to/data.h5
      format: APS_old

    optimization:
      method: nlsq  # or mcmc
      initial_parameters:
        values: [1000.0, 0.5, 50.0]  # D0, alpha, D_offset
        bounds:
          D0: [100, 10000]
          alpha: [0.0, 2.0]
          D_offset: [0, 500]

      nlsq:
        max_iterations: 100
        tolerance: 1e-6

      mcmc:
        num_warmup: 1000
        num_samples: 2000
        num_chains: 4

Parameter Registry
------------------

Centralized parameter metadata management (v2.4.1+).

.. automodule:: homodyne.config.parameter_registry
   :members:
   :undoc-members:
   :show-inheritance:

Registry Features
~~~~~~~~~~~~~~~~~

The parameter registry provides:

- Canonical parameter names
- Default bounds and values
- Physical units
- Parameter descriptions
- Mode-specific parameter sets

Parameter Information
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: homodyne.config.parameter_registry.ParameterInfo
   :members:
   :undoc-members:

Registry Functions
~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.config.parameter_registry.get_registry
   homodyne.config.parameter_registry.get_param_names
   homodyne.config.parameter_registry.get_all_param_names
   homodyne.config.parameter_registry.get_bounds
   homodyne.config.parameter_registry.get_defaults

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.config import get_param_names, get_bounds, get_defaults

    # Get parameter names for mode
    params = get_param_names(mode="static")
    print(params)  # ['D0', 'alpha', 'D_offset']

    # Get default bounds
    bounds = get_bounds(mode="static")
    print(bounds['D0'])  # (100.0, 10000.0)

    # Get default values
    defaults = get_defaults(mode="static")
    print(defaults['alpha'])  # 0.5

Canonical Parameter Names
~~~~~~~~~~~~~~~~~~~~~~~~~

**Static Mode** (3 parameters):

- ``D0``: Diffusion coefficient amplitude (nm²/s)
- ``alpha``: Diffusion power-law exponent (dimensionless)
- ``D_offset``: Diffusion offset (nm²/s)

**Laminar Flow Mode** (+4 parameters):

- ``gamma_dot_t0``: Shear rate amplitude (1/s)
- ``beta``: Shear rate power-law exponent (dimensionless)
- ``gamma_dot_t_offset``: Shear rate offset (1/s)
- ``phi0``: Flow direction angle (radians)

**Per-Angle Scaling** (mandatory v2.4.0+):

- ``contrast_0, contrast_1, ...``: Per-angle contrast parameters
- ``offset_0, offset_1, ...``: Per-angle offset parameters

Parameter Space
---------------

Prior distributions and parameter transformations for MCMC.

.. automodule:: homodyne.config.parameter_space
   :members:
   :undoc-members:
   :show-inheritance:

Prior Distributions
~~~~~~~~~~~~~~~~~~~

.. autoclass:: homodyne.config.parameter_space.PriorDistribution
   :members:
   :undoc-members:

Supported Distributions
^^^^^^^^^^^^^^^^^^^^^^^

- **Uniform**: Flat distribution over interval
- **Normal**: Gaussian distribution
- **LogNormal**: Log-normal distribution (for positive parameters)
- **TruncatedNormal**: Normal with hard bounds

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.config import ParameterSpace, PriorDistribution

    # Create parameter space
    space = ParameterSpace(mode="static")

    # Define custom priors
    space.set_prior('D0', PriorDistribution(
        dist_type='lognormal',
        params={'mean_log': 7.0, 'std_log': 1.5}
    ))

    # Sample from priors
    samples = space.sample_prior(n_samples=1000)

Parameter Manager (Legacy)
---------------------------

Legacy parameter management interface (preserved for backward compatibility).

.. automodule:: homodyne.config.parameter_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Note**: For new code, prefer using :class:`~homodyne.config.ParameterRegistry` instead.

Type Definitions
----------------

Type definitions and data structures for configuration.

.. automodule:: homodyne.config.types
   :members:
   :undoc-members:
   :show-inheritance:

Key Types
~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.config.types.AnalysisMode
   homodyne.config.types.OptimizationMethod
   homodyne.config.types.DataConfig
   homodyne.config.types.OptimizationConfig
   homodyne.config.types.AnalysisConfig

Parameter Names
---------------

Canonical parameter naming utilities.

.. automodule:: homodyne.config.parameter_names
   :members:
   :undoc-members:
   :show-inheritance:

Name Mapping
~~~~~~~~~~~~

The module provides bidirectional name mapping:

- User-facing names: ``gamma_dot_0`` → Internal: ``gamma_dot_t0``
- Handles per-angle parameter naming: ``contrast_i``, ``offset_i``
- Validates parameter name consistency

Configuration Templates
-----------------------

The config module includes YAML templates for common use cases:

**Static Mode Template** (:file:`templates/static.yaml`):

- 3 physical parameters
- NLSQ optimization default
- Minimal configuration

**Laminar Flow Template** (:file:`templates/laminar_flow.yaml`):

- 7 physical parameters
- Shear rate parameterization
- Flow direction angle

**Per-Angle Scaling Template**:

- Mandatory in v2.4.0+
- Separate contrast/offset per angle
- Total parameters = physical + 2 * n_angles

Best Practices
--------------

**Configuration Management**:

1. Use YAML files for reproducible analysis
2. Store configurations in version control
3. Document parameter choices and rationale
4. Validate configuration before running analysis

**Parameter Bounds**:

1. Use physics-informed bounds from registry
2. Tighten bounds if you have prior knowledge
3. Avoid overly restrictive bounds that prevent convergence
4. Test bounds with synthetic data first

**Per-Angle Scaling**:

1. Mandatory in v2.4.0+ (no option to disable)
2. Increases parameter count: 3 → 9 for static mode (3 angles)
3. Provides better fit quality and physically correct scaling
4. Convergence requires good initial guesses

See Also
--------

- :mod:`homodyne.optimization` - Uses configuration for optimization
- :mod:`homodyne.cli` - CLI commands use configuration
- :mod:`homodyne.data` - Data loading configuration
