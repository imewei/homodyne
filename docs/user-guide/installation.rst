Installation
=============

This guide covers installing Homodyne on your system. Homodyne is CPU-optimized and requires
Python 3.12+ with JAX 0.8.0 (CPU-only).

Prerequisites
=============

**System Requirements:**

- **Python:** 3.12 or higher
- **Operating System:** Linux, macOS, or Windows
- **RAM:** 4GB minimum (8GB+ recommended for large datasets)
- **CPU:** Multi-core processor (HPC clusters recommended for production use)

**Optional but Recommended:**

- **uv:** Modern Python package manager for faster installation
- **conda:** For isolated Python environments

Step 1: Set Up Python Environment (Recommended)
================================================

Using ``uv`` (Recommended):

.. code-block:: bash

   # Install uv if you don't have it
   curl -PsSL https://astral.sh/uv/install.sh | sh

   # Create a new Python 3.12 environment
   uv venv homodyne-env --python 3.12
   source homodyne-env/bin/activate

Using ``conda``:

.. code-block:: bash

   # Create a new environment
   conda create -n homodyne python=3.12
   conda activate homodyne

Using ``venv`` (Standard Library):

.. code-block:: bash

   # Create a new environment
   python3.12 -m venv homodyne-env
   source homodyne-env/bin/activate  # On Windows: homodyne-env\Scripts\activate

Step 2: Install Homodyne
========================

**Option A: With Documentation Dependencies (Recommended for Developers):**

.. code-block:: bash

   pip install -e ".[docs]"

This installs Homodyne in editable mode with all documentation dependencies for building docs locally.

**Option B: Standard Installation (Users):**

.. code-block:: bash

   pip install homodyne

**Option C: Using uv (Fastest):**

.. code-block:: bash

   uv pip install -e ".[docs]"

Step 3: Verify Installation
============================

Verify JAX Installation
-----------------------

.. code-block:: bash

   python -c "import jax; print(f'JAX {jax.__version__}')"

You should see output like:

.. code-block:: text

   JAX 0.8.0

**Important:** Verify you have exactly JAX 0.8.0. Other versions may have compatibility issues.

Verify Homodyne Installation
-----------------------------

.. code-block:: bash

   homodyne --version

You should see:

.. code-block:: text

   Homodyne.4.1

Check Device Status
-------------------

.. code-block:: bash

   python -m homodyne.runtime.utils.system_validator --quick

This runs a ~0.15 second system validation. You should see:

.. code-block:: text

   Homodyne.4.1 System Validation
   ==================================
   [✓] Python version: 3.12.x
   [✓] JAX 0.8.0 (CPU)
   [✓] Homodyne core modules
   [✓] Configuration system
   ...
   Health Score: 🟢 95/100 (Production ready)

JAX and Version Pinning
=======================

**Why JAX 0.8.0?**

Homodyne uses JAX 0.8.0 specifically for:

- **JIT Compilation:** ``compute_residuals()`` and ``compute_g2_scaled()`` are JIT-compiled for optimal CPU performance
- **Automatic Differentiation:** Jacobian computation for NLSQ optimization
- **Numerical Stability:** Float32 and Float64 both fully supported
- **Compatibility:** Tested extensively with NumPyro 0.18+ and BlackJAX 1.2+

**Important:** Using a different JAX version may cause:

- Compilation failures
- Silent numerical errors
- Incompatibility with MCMC inference

**Fixing Version Issues:**

If you have the wrong JAX version:

.. code-block:: bash

   pip install jax==0.8.0 jaxlib==0.8.0
   # Verify installation
   python -c "import jax; print(jax.__version__)"

Uninstalling Homodyne
=====================

To remove Homodyne:

.. code-block:: bash

   pip uninstall homodyne

To remove environment and all dependencies:

.. code-block:: bash

   # For venv
   deactivate
   rm -rf homodyne-env

   # For conda
   conda deactivate
   conda remove --name homodyne --all

Upgrading Homodyne
==================

To upgrade to a newer version:

.. code-block:: bash

   pip install --upgrade homodyne

To upgrade with documentation dependencies:

.. code-block:: bash

   pip install --upgrade homodyne[docs]

Development Installation
=========================

For development or contributing to Homodyne:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/apc-llc/homodyne.git
   cd homodyne

   # Create virtual environment
   python3.12 -m venv venv
   source venv/bin/activate

   # Install in editable mode with all dependencies
   pip install -e ".[dev,docs]"

   # Run development checks
   make quality    # Format, lint, type-check
   make test       # Run unit tests
   make test-all   # Run all tests with coverage

Troubleshooting
===============

**Problem: "No module named 'jax'"**

Solution:

.. code-block:: bash

   pip install jax==0.8.0 jaxlib==0.8.0

**Problem: "JAX version X.X.X not compatible"**

Solution: Reinstall exact version:

.. code-block:: bash

   pip uninstall jax jaxlib
   pip install jax==0.8.0 jaxlib==0.8.0

**Problem: "ImportError: cannot import name 'compute_residuals'"**

Solution: Ensure homodyne core modules are installed:

.. code-block:: bash

   pip install --force-reinstall homodyne

**Problem: "AttributeError: module 'homodyne' has no attribute 'version'"**

Solution: Update homodyne:

.. code-block:: bash

   pip install --upgrade homodyne

**Problem: "OutOfMemory on MCMC inference"**

Solution: MCMC requires significant memory. See :doc:`./cli` for worker memory settings.

Quick Verification Checklist
=============================

After installation, verify:

.. code-block:: bash

   # 1. Check Python version
   python --version
   # Expected: Python 3.12.x

   # 2. Check JAX version (must be 0.8.0)
   python -c "import jax; print(jax.__version__)"
   # Expected: 0.8.0

   # 3. Check Homodyne CLI
   homodyne --help
   # Should show command-line options

   # 4. Run system validator
   python -m homodyne.runtime.utils.system_validator --quick
   # Should show health score >= 90

   # 5. (Optional) Build documentation
   cd docs
   make html
   # Should build successfully without errors

Next Steps
==========

- :doc:`./quickstart` - Run your first analysis
- :doc:`./cli` - Learn the command-line interface
- :doc:`./configuration` - Configure your analysis
- :doc:`./examples` - Explore real-world workflows
