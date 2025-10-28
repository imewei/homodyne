Homodyne Documentation
======================

**JAX-first high-performance XPCS analysis for nonequilibrium soft matter systems**

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/jax-0.8.0-orange.svg
   :target: https://github.com/google/jax
   :alt: JAX Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

Overview
--------

Homodyne v2.0+ is a comprehensive package for X-ray Photon Correlation Spectroscopy (XPCS) analysis, implementing the theoretical framework from `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ for characterizing transport properties in flowing soft matter systems.

**Core Equation:** :math:`c_2(\phi, t_1, t_2) = 1 + \beta \times [c_1(\phi, t_1, t_2)]^2`

Key Features
------------

* **JAX-First Architecture**: Automatic differentiation, JIT compilation, and transparent GPU acceleration
* **NLSQ Optimization**: Trust-region optimization with streaming support for unlimited dataset sizes
* **MCMC Uncertainty**: Bayesian inference with NumPyro/BlackJAX for uncertainty quantification
* **Large Dataset Support**: Handle 100M+ data points with constant memory footprint
* **Device Agnostic**: Seamless CPU/GPU execution with automatic device selection
* **Configuration System**: Comprehensive YAML-based configuration with validation

Quick Links
-----------

* :doc:`user-guide/installation` - Get started in 5 minutes
* :doc:`user-guide/quickstart` - Your first analysis
* :doc:`api-reference/index` - Complete API reference
* :doc:`configuration-templates/index` - Configuration templates
* `GitHub Repository <https://github.com/imewei/homodyne>`_
* `PNAS Paper <https://doi.org/10.1073/pnas.2401162121>`_

Where to Start
--------------

**New Users:**
   Start with :doc:`user-guide/installation` followed by :doc:`user-guide/quickstart` for your first analysis in 5 minutes.

**Researchers:**
   Read :doc:`theoretical-framework/index` to understand the physics implementation, then explore :doc:`advanced-topics/index`.

**Developers:**
   Check out :doc:`developer-guide/architecture` and :doc:`developer-guide/contributing` to start contributing.

**System Administrators:**
   See :doc:`advanced-topics/gpu-acceleration` for HPC deployment and GPU setup.

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/installation
   user-guide/quickstart
   user-guide/configuration
   user-guide/cli-usage
   user-guide/shell-completion
   user-guide/examples

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api-reference/index
   api-reference/core
   api-reference/optimization
   api-reference/data
   api-reference/device
   api-reference/config
   api-reference/cli
   api-reference/utils
   api-reference/viz

.. toctree::
   :maxdepth: 2
   :caption: Theoretical Framework

   theoretical-framework/index
   theoretical-framework/core-equations
   theoretical-framework/transport-coefficients
   theoretical-framework/parameter-models

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer-guide/index
   developer-guide/architecture
   developer-guide/testing
   developer-guide/contributing
   developer-guide/code-quality
   developer-guide/performance

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced-topics/index
   advanced-topics/nlsq-optimization
   advanced-topics/mcmc-uncertainty
   advanced-topics/cmc-large-datasets
   advanced-topics/streaming-optimization
   advanced-topics/gpu-acceleration
   advanced-topics/angle-filtering

.. toctree::
   :maxdepth: 2
   :caption: Configuration Templates

   configuration-templates/index
   configuration-templates/master-template
   configuration-templates/static-isotropic
   configuration-templates/laminar-flow

Installation
------------

**CPU-Only (All Platforms):**

.. code-block:: bash

   pip install homodyne

**GPU Support (Linux Only):**

.. code-block:: bash

   # Step 1: Install JAX with CUDA support
   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

   # Step 2: Install homodyne
   pip install homodyne

**Platform Support:**

* **Linux**: CPU + optional GPU (CUDA 12.1-12.9)
* **macOS**: CPU-only
* **Windows**: CPU-only

Quick Example
-------------

.. code-block:: bash

   # Run NLSQ optimization
   homodyne --config my_config.yaml --method nlsq

   # Run MCMC sampling for uncertainty quantification
   homodyne --config my_config.yaml --method mcmc

   # Enable visualization
   homodyne --config my_config.yaml --plot-experimental-data

Parameter Models
----------------

Homodyne supports two physical models with per-angle scaling:

**Static Isotropic (3+2n parameters):**
   * Physical: [D₀, α, D_offset]
   * Per-angle scaling: [contrast, offset] × n angles
   * Example: 3 angles → 3 + 2×3 = 9 total parameters

**Laminar Flow (7+2n parameters):**
   * Physical: [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]
   * Per-angle scaling: [contrast, offset] × n angles
   * Example: 3 angles → 7 + 2×3 = 13 total parameters

See :doc:`theoretical-framework/parameter-models` for detailed explanation.

Citation
--------

If you use Homodyne in your research, please cite:

   He, H., Chen, W., et al. (2024). Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter. *PNAS*, 121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

License
-------

Homodyne is released under the MIT License. See the LICENSE file for details.

Support
-------

* **Issues**: `GitHub Issues <https://github.com/imewei/homodyne/issues>`_
* **Discussions**: `GitHub Discussions <https://github.com/imewei/homodyne/discussions>`_
* **Email**: Contact the maintainers at wchen@anl.gov

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
