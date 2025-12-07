API Reference
=============

This section provides comprehensive API documentation for the Homodyne package, automatically generated from source code docstrings using Sphinx autodoc.

**Note on Documentation Generation**: All API documentation is generated from real imports without mocking. The homodyne package and its dependencies must be installed for documentation builds to succeed.

Package Organization
--------------------

The homodyne package is organized into the following modules:

Core Modules
~~~~~~~~~~~~

- **core**: JAX-accelerated computation backend, physics models, and fitting routines
- **optimization**: NLSQ trust-region and MCMC inference engines
- **data**: XPCS data loading, preprocessing, and angle filtering

Configuration & CLI
~~~~~~~~~~~~~~~~~~~

- **config**: Configuration management, parameter handling, and templates
- **cli**: Command-line interface and interactive configuration builder

I/O & Results
~~~~~~~~~~~~~

- **io**: Result saving (JSON, NPZ) and serialization utilities

Visualization & Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **viz**: Plotting and visualization utilities
- **device**: CPU device management and status (v2.3.0+ CPU-only architecture)
- **utils**: Logging, progress tracking, and path validation

API Documentation by Module
----------------------------

.. toctree::
   :maxdepth: 2

   core
   optimization
   data
   config
   cli
   io
   viz
   device
   utils

Quick Links
-----------

**Core Physics & Computation**

- :mod:`homodyne.core.jax_backend` - JAX-accelerated residuals and G2 computation
- :mod:`homodyne.core.physics` - Time-dependent transport coefficients
- :mod:`homodyne.core.models` - Static and laminar flow models

**Optimization Methods**

- :mod:`homodyne.optimization.nlsq` - Non-linear least squares (Levenberg-Marquardt)
- :mod:`homodyne.optimization.cmc` - Consensus Monte Carlo (MCMC inference)

**Configuration**

- :class:`homodyne.config.ConfigManager` - YAML configuration management
- :class:`homodyne.config.ParameterManager` - Parameter bounds and transformations

**Data Pipeline**

- :class:`homodyne.data.XPCSDataLoader` - HDF5 data loading from APS beamlines
- :func:`homodyne.data.filter_phi_angles` - Angle filtering and selection

**I/O & Results**

- :func:`homodyne.io.save_nlsq_json_files` - Save NLSQ results to JSON
- :func:`homodyne.io.save_nlsq_npz_file` - Save numerical data to NPZ

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
