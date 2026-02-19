.. _installation:

Installation
============

This page covers all supported installation methods for Homodyne, along with
CPU optimization notes for production deployments.

.. contents:: On this page
   :local:
   :depth: 2

----

.. _install-requirements:

Requirements
------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Dependency
     - Minimum version
     - Notes
   * - Python
     - 3.12
     - Required for modern type-hint syntax
   * - JAX
     - 0.8.2
     - CPU-only build; GPU not supported
   * - NumPy
     - 2.3
     - Array operations and I/O boundaries
   * - NLSQ
     - 0.6.4
     - Trust-region LM optimizer (Homodyne's ``curve_fit``)
   * - NumPyro
     - 0.16+
     - Bayesian / CMC inference (optional but recommended)
   * - h5py
     - 3.10+
     - HDF5 data loading

.. note::

   Homodyne is **CPU-only**. JAX's GPU and TPU backends are not tested or
   supported. The XLA CPU backend provides JIT-compiled acceleration that
   rivals GPU performance for the problem sizes typical in XPCS.

----

.. _install-pip:

Install with pip
----------------

.. code-block:: bash

   pip install homodyne

For the latest development build directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/imewei/homodyne.git

----

.. _install-uv:

Install with uv (Recommended)
-------------------------------

`uv <https://github.com/astral-sh/uv>`_ is the recommended package manager.
It enforces ``uv.lock`` as the single source of truth and creates an isolated
``.venv`` that never touches the system or user site-packages.

.. code-block:: bash

   # Add homodyne to an existing uv project
   uv add homodyne

   # Or install into a fresh project
   uv init my-xpcs-project
   cd my-xpcs-project
   uv add homodyne

Run scripts through uv to guarantee the locked environment is used:

.. code-block:: bash

   uv run homodyne --config my_config.yaml --method nlsq

----

.. _install-dev:

Development Install
-------------------

Clone the repository and install with all development dependencies:

.. code-block:: bash

   git clone https://github.com/imewei/homodyne.git
   cd homodyne
   make dev           # equivalent to: uv sync --all-extras

This installs the package in editable mode along with testing, linting, and
documentation tools.

Available ``make`` targets during development:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Command
     - Purpose
   * - ``make dev``
     - Install with all dev extras
   * - ``make test``
     - Run unit tests (``tests/unit``)
   * - ``make test-all``
     - Full test suite with coverage
   * - ``make quality``
     - Format + lint + type-check (Black, Ruff, MyPy)
   * - ``make docs``
     - Build HTML documentation

----

.. _install-verify:

Verify Installation
--------------------

Check that Homodyne and its key dependencies are correctly installed:

.. code-block:: bash

   homodyne --version

.. code-block:: python

   import homodyne
   import jax
   import jax.numpy as jnp

   print(f"Homodyne:  {homodyne.__version__}")
   print(f"JAX:       {jax.__version__}")
   print(f"Devices:   {jax.devices()}")

   # Confirm XLA is compiling correctly
   x = jnp.ones((100, 100))
   print(f"JAX dot:   {jnp.dot(x, x).shape}")

Expected output:

.. code-block:: text

   Homodyne:  2.22.1
   JAX:       0.8.x
   Devices:   [CpuDevice(id=0)]
   JAX dot:   (100, 100)

----

.. _install-post-install:

Shell Completion Setup
-----------------------

Install tab completion for your shell (bash, zsh, or fish):

.. code-block:: bash

   homodyne-post-install

This copies the completion system into the active virtual environment.
Deactivate and reactivate the environment to load the new completions:

.. code-block:: bash

   deactivate && source .venv/bin/activate

Available aliases after activation:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Alias
     - Expands to
     - Purpose
   * - ``hm``
     - ``homodyne``
     - Base command
   * - ``hconfig``
     - ``homodyne-config``
     - Configuration generator
   * - ``hm-nlsq``
     - ``homodyne --method nlsq``
     - NLSQ optimization
   * - ``hm-cmc``
     - ``homodyne --method cmc``
     - Consensus Monte Carlo
   * - ``hc-stat``
     - ``homodyne-config --mode static``
     - Generate static config template
   * - ``hc-flow``
     - ``homodyne-config --mode laminar_flow``
     - Generate laminar flow config template
   * - ``hexp``
     - ``homodyne --plot-experimental-data``
     - Plot experimental data
   * - ``hsim``
     - ``homodyne --plot-simulated-data``
     - Plot simulated C2 heatmaps
   * - ``hxla``
     - ``homodyne-config-xla``
     - XLA device configuration
   * - ``hsetup``
     - ``homodyne-post-install``
     - Post-install setup
   * - ``hclean``
     - ``homodyne-cleanup``
     - Remove shell completion files

Interactive shell functions are also available:

- ``homodyne_build`` — Interactive command builder (guides method and config selection)
- ``homodyne_help`` — Display all aliases and shortcuts

----

.. _install-exit-codes:

Exit Codes
-----------

Homodyne returns standard exit codes for use in scripts and CI/CD pipelines:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - Success: analysis completed successfully
   * - ``1``
     - Configuration or data error (invalid YAML, missing file, bad parameters)
   * - ``2``
     - Optimization failed to converge (NLSQ did not reach tolerance)
   * - ``3``
     - MCMC inference failed (excessive divergences, all shards rejected)
   * - ``4``
     - I/O or file system error (HDF5 read failure, disk full)
   * - ``255``
     - Unexpected internal error (report as bug)

Example usage in a batch script:

.. code-block:: bash

   homodyne --config config.yaml --method nlsq
   if [ $? -eq 0 ]; then
     echo "Analysis succeeded"
   elif [ $? -eq 2 ]; then
     echo "Convergence failure — try different initial parameters"
   else
     echo "Analysis failed with code $?"
   fi

----

.. _install-env-vars:

Environment Variables
----------------------

**JAX Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Purpose
   * - ``JAX_PLATFORMS=cpu``
     - Force CPU backend (set by default)
   * - ``JAX_LOG_COMPILES=1``
     - Log JIT compilation events (useful for debugging)
   * - ``JAX_ENABLE_X64=1``
     - Enable 64-bit floating point (set by default)
   * - ``OMP_NUM_THREADS=N``
     - Limit OpenMP threads per process

**Homodyne Configuration:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Variable
     - Purpose
   * - ``HOMODYNE_OUTPUT_DIR``
     - Override default output directory
   * - ``HOMODYNE_DEBUG=1``
     - Enable debug-level logging

Example:

.. code-block:: bash

   # Enable JAX compilation logging for performance debugging
   JAX_LOG_COMPILES=1 homodyne --config config.yaml

   # Limit threads on shared HPC nodes
   OMP_NUM_THREADS=4 homodyne --config config.yaml --method nlsq

----

.. _install-cpu-opt:

CPU Optimization Notes
-----------------------

Homodyne uses XLA's CPU backend for all numerical computation. Two settings
significantly affect performance on multi-core servers:

**Virtual JAX devices (recommended):**

.. code-block:: bash

   # Expose 8 virtual CPU devices to JAX
   homodyne-config-xla --mode multicore --show

This sets ``XLA_FLAGS=--xla_force_host_platform_device_count=N`` so that
``jax.pmap`` can distribute work across logical CPU cores.

**NUMA-aware threading:**

.. code-block:: bash

   # Configure NUMA-aware thread affinity
   homodyne-config-xla --mode numa --show

On NUMA systems, binding threads to specific cores can reduce memory-access
latency by 20–40% for large XPCS datasets.

See :doc:`user_guide/04_practical_guides/performance_tuning` for detailed
benchmarks and tuning guidelines.

----

.. _install-uninstall:

Uninstall
---------

To remove Homodyne and the shell completion files:

.. code-block:: bash

   # Remove shell completion files first
   homodyne-cleanup

   # Then remove the package
   uv remove homodyne
   # or: pip uninstall homodyne
