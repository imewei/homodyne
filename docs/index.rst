Homodyne Documentation
======================

**Version 2.4.1** - CPU-optimized JAX-first XPCS analysis for nonequilibrium soft matter systems

.. admonition:: **BREAKING CHANGES in v2.4.x**
   :class: warning

   **v2.4.1 - CMC-Only MCMC Architecture**

   * **CMC mandatory**: All MCMC runs use Consensus Monte Carlo; NUTS auto-selection removed
   * **Removed CLI flags**: ``--min-samples-cmc``, ``--memory-threshold-pct`` (deprecated)
   * **Per-phi initialization**: Initial values from config or per-phi percentiles
   * **Migration guide**: :doc:`migration/v3_cmc_only`

   **v2.4.0 - Per-Angle Scaling Mandatory**

   * **Breaking**: Legacy scalar ``per_angle_scaling=False`` removed
   * **Impact**: 3 angles: 5 params → 9 params ``[c₀,c₁,c₂, o₀,o₁,o₂, D0,α,D_offset]``
   * **Rationale**: Per-angle mode is physically correct for heterogeneous samples

   **v2.3.0 - GPU Support Removed**

   * **CPU-Only Architecture** - All GPU acceleration removed
   * **For GPU users**: Stay on v2.2.1 (last GPU-supporting version)
   * **Migration guide**: :doc:`migration/v2.2-to-v2.3-gpu-removal`

.. admonition:: Previous Releases
   :class: note

   **v2.2.1** - Parameter Expansion Fix for per-angle scaling silent failures

   **v2.2.0** - Angle-Stratified Chunking for large datasets

   See :doc:`releases/v2.2-stratification-release-notes` for complete details

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

   user-guide/index
   user-guide/installation
   user-guide/quickstart
   user-guide/configuration
   user-guide/cli-usage
   user-guide/shell-completion
   user-guide/examples
   user-guide/cmc_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api-reference/index
   api-reference/core
   api-reference/optimization
   api-reference/cmc_api
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
   developer-guide/cmc_architecture
   developer-guide/cmc_rng_and_initialization

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture
   architecture/cmc-dual-mode-strategy
   architecture/cmc-decision-quick-reference
   architecture/nlsq-least-squares-solution
   architecture/nuts-chain-parallelization
   architecture/nuts-chain-parallelization-quick-reference
   architecture/ultra-think-nlsq-solution-20251106

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
   :caption: Guides

   guides/performance_tuning
   guides/streaming_optimizer_usage
   guides/readthedocs-setup
   guides/readthedocs-troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Configuration Templates

   configuration-templates/index
   configuration-templates/master-template
   configuration-templates/static-isotropic
   configuration-templates/laminar-flow

.. toctree::
   :maxdepth: 2
   :caption: Migration Guides

   migration/v2.0-to-v2.1
   migration/v2.2-to-v2.3-gpu-removal
   migration/v2_to_v3_migration
   migration/v3_cmc_migration
   migration/v3_cmc_only

.. toctree::
   :maxdepth: 2
   :caption: Performance

   performance/onednn_benchmark_results
   performance/v2.3-cpu-optimizations

.. toctree::
   :maxdepth: 2
   :caption: Troubleshooting

   troubleshooting/silent-failure-diagnosis
   troubleshooting/cmc_troubleshooting
   troubleshooting/gradient_imbalance_solution
   troubleshooting/imshow-transpose-pitfalls
   troubleshooting/nlsq-zero-iterations-investigation
   troubleshooting/numerical-stability-mitigation-plan
   troubleshooting/per_angle_scaling_guide
   troubleshooting/shear_gradient_check_20251112
   troubleshooting/stratification-nlsq-incompatibility-analysis

.. toctree::
   :maxdepth: 2
   :caption: Testing

   testing/RELEASE_NOTES_v2.1.0_TEST_STATUS
   testing/test_fixing_final_report
   testing/v2.1.1_test_fixes_tracking

.. toctree::
   :maxdepth: 2
   :caption: Refactoring Notes
   :hidden:

   refactor/mcmc-cmc-only-prototype

.. toctree::
   :maxdepth: 2
   :caption: Release Notes

   releases/v2.2-stratification-release-notes

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
