homodyne.device - Device Management
===================================

.. automodule:: homodyne.device
   :no-members:

Overview
--------

The ``homodyne.device`` module provides CPU device configuration for optimal JAX performance. It automatically detects available hardware and configures threading for HPC environments.

.. note::

   **v2.3.0+**: GPU support has been removed. This module now focuses exclusively on
   CPU optimization for multi-core systems and HPC clusters.

**Key Features:**

* **CPU-Only Architecture**: Optimized for multi-core CPUs
* **HPC CPU Optimization**: Thread configuration for 36/128-core nodes
* **Performance Benchmarking**: Device capability assessment

Module Structure
----------------

The device module is organized into several submodules:

* :mod:`homodyne.device.cpu` - CPU threading optimization
* :mod:`homodyne.device.config` - Device configuration management

Submodules
----------

homodyne.device.cpu
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.device.cpu
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

CPU threading configuration for HPC environments.

**Key Functions:**

* ``configure_cpu_hpc()`` - Configure CPU for HPC nodes
* ``detect_cpu_info()`` - Get CPU information
* ``get_optimal_batch_size()`` - Calculate optimal batch size

**HPC CPU Optimization:**

.. code-block:: python

   from homodyne.device.cpu import configure_cpu_hpc, detect_cpu_info

   # Get CPU information
   cpu_info = detect_cpu_info()
   print(f"CPU cores: {cpu_info['physical_cores']}")
   print(f"CPU threads: {cpu_info['logical_cores']}")

   # Configure CPU for HPC node
   configure_cpu_hpc(num_threads=34)  # Reserve 2 cores for OS on 36-core node

homodyne.device.config
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.device.config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

Device configuration management.

**Key Functions:**

* ``detect_hardware()`` - Detect available hardware
* ``should_use_cmc()`` - Check if CMC should be used (deprecated, always returns True in v2.4.1+)

HPC Configuration
-----------------

**36-Core Nodes:**

.. code-block:: python

   import os
   os.environ['OMP_NUM_THREADS'] = '34'
   os.environ['JAX_NUM_THREADS'] = '34'

   from homodyne.device import configure_optimal_device
   configure_optimal_device()

**128-Core Nodes:**

.. code-block:: python

   import os
   os.environ['OMP_NUM_THREADS'] = '120'  # Reserve 8 cores for OS
   os.environ['JAX_NUM_THREADS'] = '120'

   from homodyne.device import configure_optimal_device
   configure_optimal_device()

Environment Variables
---------------------

**JAX Configuration:**

.. code-block:: bash

   # Force CPU execution (default in v2.3.0+)
   export JAX_PLATFORM_NAME=cpu

   # Enable float64
   export JAX_ENABLE_X64=1

   # Disable JIT (for debugging)
   export JAX_DISABLE_JIT=1

   # Set CPU threads
   export OMP_NUM_THREADS=34
   export MKL_NUM_THREADS=34

See Also
--------

* :doc:`../migration/v2.2-to-v2.3-gpu-removal` - GPU removal migration guide
* :doc:`core` - Core physics engine that uses device configuration
