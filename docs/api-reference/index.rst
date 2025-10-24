API Reference
=============

Complete API documentation for Homodyne v2.0+. This section provides detailed documentation for all modules, classes, and functions.

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   core
   optimization
   data
   device
   config
   cli
   utils
   viz

Module Overview
---------------

Homodyne is organized into 8 major modules:

**Core Physics**
   :doc:`core` - JAX-based physics engine for computing correlation functions

**Optimization**
   :doc:`optimization` - NLSQ trust-region and MCMC Bayesian inference methods

**Data Pipeline**
   :doc:`data` - HDF5 data loading, preprocessing, and quality control

**Device Management**
   :doc:`device` - CPU/GPU device selection and configuration

**Configuration**
   :doc:`config` - YAML configuration system and parameter management

**Command-Line Interface**
   :doc:`cli` - CLI entry points and command implementations

**Utilities**
   :doc:`utils` - Logging, progress tracking, and validation helpers

**Visualization**
   :doc:`viz` - MCMC diagnostics and result visualization

Cross-Reference Guide
---------------------

**Common Tasks:**

* Compute g2 correlation: :func:`homodyne.core.jax_backend.compute_g2`
* Run NLSQ optimization: :func:`homodyne.optimization.nlsq_wrapper.fit_nlsq_jax`
* Run MCMC sampling: :func:`homodyne.optimization.mcmc.fit_mcmc_jax`
* Load XPCS data: :class:`homodyne.data.xpcs_loader.XPCSDataLoader`
* Configure device: :func:`homodyne.device.configure_optimal_device`
* Load configuration: :class:`homodyne.config.manager.ConfigManager`

**See Also:**

* :doc:`../user-guide/quickstart` - Usage examples
* :doc:`../theoretical-framework/index` - Physics background
* :doc:`../developer-guide/architecture` - Design patterns
