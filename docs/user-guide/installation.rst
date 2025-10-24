Installation Guide
===================

This guide covers installation of Homodyne v2.0+ across all supported platforms (Linux, macOS, Windows).

System Requirements
-------------------

**Python Version**

- **Python 3.12+** is required
- Check your Python version:

.. code-block:: bash

   python --version

**Platform Support**

- **Linux**: Full support (CPU + optional GPU)
- **macOS**: CPU-only support
- **Windows**: CPU-only support

**Optional GPU Support (Linux only)**

- **CUDA 12.1-12.9** (must be pre-installed on system)
- **NVIDIA driver >= 525**
- Not supported on macOS or Windows

CPU-Only Installation (All Platforms)
--------------------------------------

The default installation provides CPU-only support and works on all platforms.

**Using pip:**

.. code-block:: bash

   pip install homodyne

**From development repository:**

.. code-block:: bash

   git clone https://github.com/imewei/homodyne.git
   cd homodyne
   make dev

**Verify installation:**

.. code-block:: bash

   homodyne --version
   python -c "from homodyne.device import get_device_status; print(get_device_status())"

Expected output:

.. code-block:: text

   homodyne, version 2.0.0
   Device Status:
     - Backend: JAX (CPU)
     - Platform: Linux/macOS/Windows
     - Available devices: 1

GPU Installation (Linux Only)
-----------------------------

GPU support requires CUDA 12.1-12.9 pre-installed on your system.

**Step 1: Verify System CUDA**

.. code-block:: bash

   nvcc --version  # Should show CUDA 12.1-12.9
   nvidia-smi      # Should show NVIDIA GPU

**Step 2: Install JAX with CUDA Support**

.. code-block:: bash

   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

**Step 3: Install Homodyne**

.. code-block:: bash

   pip install homodyne

**Using Make (development repository):**

.. code-block:: bash

   git clone https://github.com/imewei/homodyne.git
   cd homodyne
   make install-jax-gpu  # Linux only
   make dev

**Verify GPU installation:**

.. code-block:: bash

   python -c "import jax; print(jax.devices())"
   python -c "from homodyne.device import get_device_status; print(get_device_status())"
   make gpu-check

Expected output should show GPU devices available:

.. code-block:: text

   [cuda(id=0)]
   Device Status:
     - Backend: JAX (GPU/CUDA)
     - Platform: Linux
     - Available devices: 1 GPU (NVIDIA ...)

Development Installation
-------------------------

For contributing to Homodyne, install the development version:

**Prerequisites:**

- Git
- Python 3.12+
- make (on Unix-like systems)

**Installation steps:**

.. code-block:: bash

   # Clone repository
   git clone https://github.com/imewei/homodyne.git
   cd homodyne

   # CPU-only development (all platforms)
   make dev

   # GPU support development (Linux only)
   make install-jax-gpu

   # Install pre-commit hooks
   pre-commit install

**Verify development installation:**

.. code-block:: bash

   make test                # Run tests
   make format              # Auto-format code
   make lint                # Run linters
   make quality             # All quality checks

Troubleshooting
---------------

**ImportError: No module named 'homodyne'**

Verify installation:

.. code-block:: bash

   pip list | grep homodyne
   python -c "import homodyne; print(homodyne.__version__)"

If not found, reinstall:

.. code-block:: bash

   pip install --upgrade homodyne

**CUDA not found (GPU installation)**

On HPC systems, load CUDA module first:

.. code-block:: bash

   module load cuda/12.2   # or your system's CUDA module
   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

**JAX backend not available**

Verify JAX installation:

.. code-block:: bash

   python -c "import jax; print(jax.config.jax_platform_name)"

Should print `cpu` or `gpu` depending on your installation.

**GPU computation slower than CPU**

Small datasets (<1M points) often run faster on CPU. GPU optimization applies to large datasets (>10M points).

Next Steps
----------

After successful installation:

1. :doc:`quickstart` - Run your first analysis in 5 minutes
2. :doc:`configuration` - Learn about the YAML configuration system
3. :doc:`examples` - Explore example workflows
4. :doc:`../advanced-topics/index` - Advanced optimization techniques

See Also
--------

- :doc:`cli-usage` - Command-line interface reference
- :doc:`shell-completion` - Setting up bash/zsh completion
- :doc:`../developer-guide/performance` - Performance optimization guide
