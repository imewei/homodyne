homodyne.device - Device Management
===================================

.. automodule:: homodyne.device
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.device`` module provides intelligent CPU/GPU device selection and configuration for optimal JAX performance. It automatically detects available hardware, configures threading for HPC environments, and provides graceful fallback from GPU to CPU execution.

**Key Features:**

* **Automatic Device Detection**: CPU, CUDA GPU, or TPU
* **HPC CPU Optimization**: Thread configuration for 36/128-core nodes
* **GPU/CUDA Configuration**: System CUDA integration (12.1-12.9)
* **Performance Benchmarking**: Device capability assessment
* **Graceful Fallback**: Automatic CPU fallback when GPU unavailable

Module Structure
----------------

The device module is organized into several submodules:

* :mod:`homodyne.device.__init__` - Main API with ``configure_optimal_device()``
* :mod:`homodyne.device.gpu` - GPU/CUDA configuration
* :mod:`homodyne.device.cpu` - CPU threading optimization
* :mod:`homodyne.device.config` - Device configuration management

Submodules
----------

Main API
~~~~~~~~

.. automodule:: homodyne.device
   :members: configure_optimal_device, get_device_status, benchmark_device_performance
   :undoc-members:
   :show-inheritance:

**Key Functions:**

* ``configure_optimal_device()`` - Auto-detect and configure optimal device
* ``get_device_status()`` - Get current device information
* ``benchmark_device_performance()`` - Benchmark device capabilities

**Usage Example:**

.. code-block:: python

   from homodyne.device import configure_optimal_device, get_device_status

   # Configure optimal device (automatic detection)
   device = configure_optimal_device()

   print(f"Using device: {device}")  # e.g., "cuda:0" or "cpu"

   # Get device status
   status = get_device_status()
   print(f"Device type: {status['device_type']}")
   print(f"Device count: {status['device_count']}")
   print(f"Memory (GB): {status['memory_gb']}")

**Benchmark Performance:**

.. code-block:: python

   from homodyne.device import benchmark_device_performance

   # Benchmark current device
   benchmark_result = benchmark_device_performance(
       matrix_size=1000,
       num_iterations=10
   )

   print(f"FLOPS: {benchmark_result['flops']:.2e}")
   print(f"Memory bandwidth (GB/s): {benchmark_result['bandwidth']:.2f}")
   print(f"Computation time (ms): {benchmark_result['time_ms']:.2f}")

homodyne.device.gpu
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.device.gpu
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

GPU/CUDA configuration and management.

**Key Functions:**

* ``configure_gpu()`` - Configure GPU device
* ``is_gpu_available()`` - Check GPU availability
* ``get_gpu_memory()`` - Get GPU memory information

**GPU Requirements:**

* **OS**: Linux x86_64 or aarch64 **only**
* **CUDA**: 12.1-12.9 (system CUDA installation)
* **NVIDIA driver**: >= 525
* **Not supported**: Windows, macOS (CPU-only on these platforms)

**Usage Example:**

.. code-block:: python

   from homodyne.device.gpu import is_gpu_available, configure_gpu, get_gpu_memory

   # Check GPU availability
   if is_gpu_available():
       print("GPU is available")

       # Configure GPU
       device = configure_gpu(device_id=0)

       # Get memory info
       memory_info = get_gpu_memory(device_id=0)
       print(f"Total memory: {memory_info['total_gb']:.2f} GB")
       print(f"Free memory: {memory_info['free_gb']:.2f} GB")
   else:
       print("GPU not available, using CPU")

**HPC Installation (Linux + CUDA):**

.. code-block:: bash

   # Load system CUDA modules
   module load cuda/12.2 cudnn/9.8

   # Install JAX with CUDA support
   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

   # Install homodyne
   pip install homodyne[dev]

homodyne.device.cpu
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.device.cpu
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

CPU threading configuration for HPC environments.

**Key Functions:**

* ``configure_cpu()`` - Configure CPU threading
* ``get_cpu_info()`` - Get CPU information
* ``optimize_for_hpc()`` - HPC-specific optimizations

**HPC CPU Optimization:**

.. code-block:: python

   from homodyne.device.cpu import configure_cpu, optimize_for_hpc

   # Configure CPU for HPC node
   configure_cpu(
       num_threads=36,  # Use all cores on 36-core node
       enable_mkl=True   # Enable Intel MKL if available
   )

   # Apply HPC-specific optimizations
   optimize_for_hpc(
       node_type='36-core',  # or '128-core'
       enable_numa=True      # NUMA awareness
   )

**Usage Example:**

.. code-block:: python

   from homodyne.device.cpu import get_cpu_info

   # Get CPU information
   cpu_info = get_cpu_info()
   print(f"CPU cores: {cpu_info['num_cores']}")
   print(f"CPU threads: {cpu_info['num_threads']}")
   print(f"CPU model: {cpu_info['model']}")

homodyne.device.config
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.device.config
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

Device configuration management.

**Key Classes:**

* ``DeviceConfig`` - Device configuration container

Device Selection Strategy
-------------------------

**Automatic Selection Priority:**

1. **CUDA GPU** (if available and supported)

   * Linux x86_64/aarch64 with CUDA 12.1-12.9
   * NVIDIA driver >= 525

2. **CPU** (fallback or exclusive)

   * All platforms (Linux, macOS, Windows)
   * HPC optimization for 36/128-core nodes

**Manual Override:**

.. code-block:: python

   import os

   # Force CPU execution
   os.environ['JAX_PLATFORM_NAME'] = 'cpu'

   from homodyne.device import configure_optimal_device
   device = configure_optimal_device()  # Will use CPU

   # Force GPU execution (will fail if GPU unavailable)
   os.environ['JAX_PLATFORM_NAME'] = 'gpu'
   device = configure_optimal_device()  # Will use GPU or raise error

Platform Support
----------------

**Linux (Full Support)**
   * CPU: Full support with HPC optimizations
   * GPU: CUDA 12.1-12.9 via ``jax[cuda12-local]``
   * Recommended for production and HPC

**macOS (CPU-only)**
   * CPU: Full support
   * GPU: Not supported (no CUDA on macOS)
   * Use for development and testing

**Windows (CPU-only)**
   * CPU: Full support
   * GPU: Not supported (JAX CUDA not available on Windows)
   * Use for development and testing

HPC Configuration
-----------------

**36-Core Nodes:**

.. code-block:: python

   from homodyne.device.cpu import configure_cpu

   configure_cpu(
       num_threads=36,
       enable_mkl=True,
       thread_affinity='compact'
   )

**128-Core Nodes:**

.. code-block:: python

   from homodyne.device.cpu import configure_cpu

   configure_cpu(
       num_threads=128,
       enable_mkl=True,
       thread_affinity='scatter'  # Better for NUMA
   )

**GPU Nodes (Linux):**

.. code-block:: python

   from homodyne.device.gpu import configure_gpu

   # Single GPU
   configure_gpu(device_id=0)

   # Multi-GPU (manual sharding required)
   configure_gpu(device_id=0)  # Use first GPU

Performance Benchmarking
------------------------

**Benchmark Example:**

.. code-block:: python

   from homodyne.device import benchmark_device_performance

   # Benchmark CPU
   cpu_result = benchmark_device_performance(device='cpu')

   # Benchmark GPU (if available)
   try:
       gpu_result = benchmark_device_performance(device='gpu')
       speedup = cpu_result['time_ms'] / gpu_result['time_ms']
       print(f"GPU speedup: {speedup:.2f}x")
   except Exception as e:
       print(f"GPU benchmark failed: {e}")

**Typical Performance:**

* **CPU (36-core HPC)**: ~10-50 GFLOPS
* **GPU (NVIDIA A100)**: ~500-2000 GFLOPS (10-50× speedup)
* **Memory bandwidth**: GPU ~1 TB/s, CPU ~100 GB/s

Configuration
-------------

**YAML Configuration:**

.. code-block:: yaml

   device:
     type: 'auto'         # auto | cpu | gpu
     gpu_id: 0            # GPU device ID (if multiple GPUs)
     cpu_threads: null    # null = auto-detect, or specify number

   performance:
     enable_jit: true     # Enable JIT compilation
     enable_x64: false    # Use float64 (slower but more precise)

Environment Variables
---------------------

**JAX Configuration:**

.. code-block:: bash

   # Force CPU execution
   export JAX_PLATFORM_NAME=cpu

   # Force GPU execution
   export JAX_PLATFORM_NAME=gpu

   # Enable float64
   export JAX_ENABLE_X64=1

   # Disable JIT (for debugging)
   export JAX_DISABLE_JIT=1

   # Set CPU threads
   export OMP_NUM_THREADS=36
   export MKL_NUM_THREADS=36

Troubleshooting
---------------

**GPU Not Detected:**

* Check CUDA installation: ``nvcc --version``
* Check NVIDIA driver: ``nvidia-smi``
* Verify JAX GPU support: ``python -c "import jax; print(jax.devices())"``
* Ensure Linux platform (GPU not supported on macOS/Windows)

**Out of Memory (GPU):**

* Reduce batch size or dataset size
* Use CPU for large datasets
* Enable streaming optimization

**Slow CPU Performance:**

* Check thread configuration: ``get_cpu_info()``
* Enable MKL if available
* Use HPC optimization: ``optimize_for_hpc()``

**Platform-Specific Issues:**

* **macOS**: GPU not supported, use CPU
* **Windows**: GPU not supported, use CPU
* **Linux**: Full support, check CUDA installation

See Also
--------

* :doc:`../advanced-topics/gpu-acceleration` - Device configuration guide
* :doc:`../advanced-topics/gpu-acceleration` - HPC deployment guide
* :doc:`../advanced-topics/gpu-acceleration` - GPU optimization patterns
* :doc:`core` - Core physics engine that uses device configuration

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.device import (
       configure_optimal_device,
       get_device_status,
       benchmark_device_performance,
   )

**Related Functions:**

* :func:`homodyne.core.jax_backend.compute_g2_scaled` - Uses configured device
* :func:`homodyne.optimization.fit_nlsq_jax` - Uses configured device
* :func:`homodyne.optimization.fit_mcmc_jax` - Uses configured device
