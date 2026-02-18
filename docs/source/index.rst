.. _homodyne-home:

=========
Homodyne
=========

.. raw:: html

   <div class="badge-row">
     <img src="https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white" alt="Python 3.12+">
     <img src="https://img.shields.io/badge/JAX-CPU--only-orange?logo=google&logoColor=white" alt="JAX CPU-only">
     <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
     <img src="https://img.shields.io/badge/DOI-10.1073%2Fpnas.2401162121-blue" alt="DOI PNAS 2024">
   </div>

**JAX-first XPCS analysis for nonequilibrium soft matter systems.**

Homodyne is a CPU-optimized X-ray Photon Correlation Spectroscopy (XPCS)
analysis framework built on JAX. It implements the algorithms described in
`He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ and
`He et al. PNAS 2025 <https://doi.org/10.1073/pnas.2514216122>`_
for analyzing time-dependent dynamics in soft matter systems — from static
isotropic scattering to complex laminar flow geometries.

.. _at-a-glance:

At a Glance
-----------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Language**
     - Python 3.12+
   * - **Compute backend**
     - JAX |geq| 0.8.2 (CPU-only, XLA-accelerated)
   * - **Optimizer**
     - NLSQ |geq| 0.6.4 (trust-region Levenberg-Marquardt)
   * - **Bayesian engine**
     - NumPyro / NUTS / Consensus Monte Carlo
   * - **Analysis modes**
     - ``static`` (3 params) | ``laminar_flow`` (7 params)
   * - **Input format**
     - HDF5 via ``XPCSDataLoader``
   * - **Output format**
     - JSON + NPZ result bundles
   * - **License**
     - MIT

.. |geq| unicode:: U+2265

.. _key-features:

Key Features
------------

- **JIT-compiled physics** — full homodyne model in JAX with ``jit`` + ``vmap``;
  no Python loops in the hot path.
- **Two inference methods** — point-estimate NLSQ (seconds) and full Bayesian
  CMC/NUTS posteriors (minutes), both from the same model.
- **Anti-degeneracy system** — quantile-based per-angle scaling prevents
  parameter absorption for laminar flow data (v2.9.0+).
- **Consensus Monte Carlo** — embarrassingly parallel Bayesian inference across
  CPU shards with automatic shard-size selection.
- **Adaptive sampling** — NUTS warmup/samples scale with dataset size to avoid
  wasted compute on small shards.
- **Strict data integrity** — no silent subsampling; full-precision HDF5 I/O
  with shape/dtype/NaN validation at every boundary.
- **CLI-first workflow** — ``homodyne``, ``homodyne-config``, and
  ``homodyne-config-xla`` entry points with shell completion.

.. _quick-start:

Quick Start
-----------

Install and run a static-mode NLSQ analysis in three steps:

.. code-block:: bash

   # 1. Install (recommended: uv)
   uv add homodyne

   # 2. Generate a configuration template
   homodyne-config --mode static --output my_config.yaml

   # 3. Run NLSQ analysis
   homodyne --config my_config.yaml --method nlsq

Or from Python:

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager
   from homodyne.optimization.nlsq import fit_nlsq_jax

   # Load configuration and experimental data
   config = ConfigManager.from_yaml("my_config.yaml")
   data   = XPCSDataLoader(config).load()

   # Run NLSQ fit
   result = fit_nlsq_jax(data, config)

   print(f"D0      = {result.params['D0']:.4e}")
   print(f"alpha   = {result.params['alpha']:.3f}")
   print(f"Chi2/n  = {result.chi2_reduced:.4f}")

See :doc:`quickstart` for the full five-minute guide.

----

.. _documentation-sections:

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Theory & Physics

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   configuration/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developer/index

----

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   installation
   quickstart
   contributing

----

.. _citation:

Citation
--------

If you use Homodyne in published research, please cite both papers:

**Primary algorithm paper:**

.. code-block:: bibtex

   @article{He2024,
     author  = {He, Hongrui and Liang, Hao and Chu, Miaoqi and Jiang, Zhang and
                de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
                and Chen, Wei},
     title   = {Transport coefficient approach for characterizing nonequilibrium
                dynamics in soft matter},
     journal = {Proceedings of the National Academy of Sciences},
     year    = {2024},
     volume  = {121},
     number  = {31},
     pages   = {e2401162121},
     doi     = {10.1073/pnas.2401162121},
     url     = {https://doi.org/10.1073/pnas.2401162121}
   }

**Laminar flow / shear dynamics extension:**

.. code-block:: bibtex

   @article{He2025,
     author  = {He, Hongrui and Liang, Heyi and Chu, Miaoqi and Jiang, Zhang and
                de Pablo, Juan J and Tirrell, Matthew V and Narayanan, Suresh
                and Chen, Wei},
     title   = {Bridging microscopic dynamics and rheology in the yielding
                of charged colloidal suspensions},
     journal = {Proceedings of the National Academy of Sciences},
     year    = {2025},
     volume  = {122},
     number  = {42},
     pages   = {e2514216122},
     doi     = {10.1073/pnas.2514216122},
     url     = {https://doi.org/10.1073/pnas.2514216122}
   }

----

.. _community-support:

Community and Support
---------------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Source code**
     - `github.com/imewei/homodyne <https://github.com/imewei/homodyne>`_
   * - **Bug reports**
     - `GitHub Issues <https://github.com/imewei/homodyne/issues>`_
   * - **Discussions**
     - `GitHub Discussions <https://github.com/imewei/homodyne/discussions>`_
   * - **Institution**
     - Argonne National Laboratory

.. note::

   Homodyne is developed at Argonne National Laboratory. Contributions,
   bug reports, and feature requests are welcome via GitHub.
