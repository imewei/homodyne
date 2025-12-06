Device Module
=============

The :mod:`homodyne.device` module provides CPU device optimization and configuration for high-performance computing environments.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**CPU-Only Architecture** (v2.3.0+):

GPU support was removed in v2.3.0 to focus on reliable, HPC-optimized CPU execution. The device module provides:

- Automatic CPU device detection and configuration
- HPC optimization for multi-core CPUs (14-128 cores)
- NUMA-aware thread allocation
- Performance benchmarking
- Optimal batch size estimation

**Design Philosophy**:

- Simplify deployment (CPU-only, no GPU complications)
- Optimize for HPC clusters with high core counts
- Reliable performance on standard workstations
- Automatic configuration with sensible defaults

Module Contents
---------------

.. automodule:: homodyne.device
   :members:
   :undoc-members:
   :show-inheritance:

Primary Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.device.configure_optimal_device
   homodyne.device.get_device_status
   homodyne.device.benchmark_device_performance

Device Configuration
--------------------

Automatic optimal device configuration.

.. autofunction:: homodyne.device.configure_optimal_device

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.device import configure_optimal_device

    # Auto-configure optimal CPU settings
    config = configure_optimal_device()

    print(f"Device: {config['device_type']}")
    print(f"Threads: {config['device_info']['threads_configured']}")
    print(f"Performance ready: {config['performance_ready']}")

    # Manual thread count
    config = configure_optimal_device(cpu_threads=32)

Configuration Result
~~~~~~~~~~~~~~~~~~~~

The configuration dictionary contains:

- ``device_type``: Always "cpu" (v2.3.0+)
- ``configuration_successful``: Boolean indicating success
- ``performance_ready``: Boolean indicating HPC optimization
- ``device_info``: Detailed CPU configuration
- ``recommendations``: Performance optimization suggestions
- ``warnings``: Any configuration issues

Device Status
-------------

Query current device capabilities and status.

.. autofunction:: homodyne.device.get_device_status

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.device import get_device_status

    status = get_device_status()

    print(f"CPU cores: {status['cpu_info']['physical_cores']}")
    print(f"Performance estimate: {status['performance_estimate']}")

    for rec in status['recommendations']:
        print(f"- {rec}")

Status Information
~~~~~~~~~~~~~~~~~~

The status dictionary provides:

- ``timestamp``: When status was queried
- ``cpu_info``: CPU hardware information
- ``performance_estimate``: "high", "medium-high", or "medium"
- ``recommendations``: Performance suggestions

Performance Estimates
~~~~~~~~~~~~~~~~~~~~~

- **High**: 32+ physical cores (HPC nodes)
- **Medium-High**: 16-31 physical cores (workstations)
- **Medium**: < 16 physical cores (standard systems)

Performance Benchmarking
-------------------------

Benchmark device performance for optimization planning.

.. autofunction:: homodyne.device.benchmark_device_performance

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.device import benchmark_device_performance

    # Run benchmark
    results = benchmark_device_performance(test_size=5000)

    print(f"Device: {results['device_type']}")
    print(f"Results: {results['results']['cpu']}")

Benchmark Metrics
~~~~~~~~~~~~~~~~~

The benchmark measures:

- Computation time for matrix operations
- Memory bandwidth
- Thread scaling efficiency
- Optimal batch size recommendations

CPU Module
----------

HPC CPU optimization utilities.

.. automodule:: homodyne.device.cpu
   :members:
   :undoc-members:
   :show-inheritance:

CPU-Specific Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.device.cpu.configure_cpu_hpc
   homodyne.device.cpu.detect_cpu_info
   homodyne.device.cpu.benchmark_cpu_performance
   homodyne.device.cpu.get_optimal_batch_size

HPC Configuration
~~~~~~~~~~~~~~~~~

::

    from homodyne.device import configure_cpu_hpc

    # Configure for HPC environment
    cpu_config = configure_cpu_hpc(
        num_threads=36,
        enable_hyperthreading=False,  # Better for HPC
        numa_policy="auto",
        memory_optimization="standard"
    )

    print(f"Threads configured: {cpu_config['threads_configured']}")
    print(f"NUMA nodes: {cpu_config['numa_nodes']}")

CPU Information
~~~~~~~~~~~~~~~

::

    from homodyne.device import detect_cpu_info

    cpu_info = detect_cpu_info()

    print(f"Physical cores: {cpu_info['physical_cores']}")
    print(f"Logical cores: {cpu_info['logical_cores']}")
    print(f"CPU frequency: {cpu_info['cpu_freq_mhz']} MHz")
    print(f"L3 cache: {cpu_info['l3_cache_mb']} MB")

Optimal Batch Size
~~~~~~~~~~~~~~~~~~

::

    from homodyne.device import get_optimal_batch_size

    # Estimate optimal batch size for memory
    batch_size = get_optimal_batch_size(
        data_size_mb=1024,
        available_memory_gb=64
    )

    print(f"Recommended batch size: {batch_size}")

Configuration Module
--------------------

Device configuration utilities.

.. automodule:: homodyne.device.config
   :members:
   :undoc-members:
   :show-inheritance:

Environment Variables
---------------------

The device module sets the following environment variables:

JAX Configuration
~~~~~~~~~~~~~~~~~

- ``JAX_PLATFORM_NAME``: Set to "cpu" (forces CPU execution)
- ``OMP_NUM_THREADS``: Set to optimal thread count
- ``JAX_ENABLE_X64``: Enable 64-bit precision when needed

HPC Optimization
~~~~~~~~~~~~~~~~

- ``KMP_AFFINITY``: Thread affinity for Intel CPUs
- ``GOMP_CPU_AFFINITY``: Thread affinity for GNU OpenMP
- ``OMP_PROC_BIND``: Thread binding strategy
- ``OMP_PLACES``: Thread placement policy

**Note**: These are set automatically by :func:`configure_optimal_device`. Manual setting is not recommended.

HPC Best Practices
------------------

**For HPC Clusters**:

1. **Request physical cores only**::

       # PBS/Torque
       #PBS -l nodes=1:ppn=36

       # SLURM
       #SBATCH --ntasks-per-node=36
       #SBATCH --cpus-per-task=1

2. **Disable hyperthreading**::

       config = configure_optimal_device(cpu_threads=36)

3. **NUMA awareness**::

       # Let system auto-detect NUMA topology
       cpu_config = configure_cpu_hpc(numa_policy="auto")

4. **Memory allocation**::

       # Request 4-5 GB per core for MCMC
       #PBS -l mem=180gb  # 36 cores × 5 GB

**For Workstations**:

1. **Leave headroom for OS**::

       # Use n_cores - 2 for interactive systems
       import multiprocessing
       n_cores = multiprocessing.cpu_count() - 2
       config = configure_optimal_device(cpu_threads=n_cores)

2. **Monitor performance**::

       # Use htop/top to verify thread usage
       htop

3. **Batch size optimization**::

       # Adjust batch size based on available memory
       batch_size = get_optimal_batch_size(
           data_size_mb=2048,
           available_memory_gb=32
       )

Migration from v2.2 (GPU)
-------------------------

**GPU Support Removed** in v2.3.0.

If migrating from v2.2.x with GPU code:

1. **Remove GPU-specific imports**::

       # Old (v2.2)
       from homodyne.device import configure_gpu

       # New (v2.3)
       from homodyne.device import configure_optimal_device

2. **Update device configuration**::

       # Old (v2.2)
       config = configure_gpu()

       # New (v2.3)
       config = configure_optimal_device()

3. **Remove GPU environment variables**::

       # No longer needed:
       # CUDA_VISIBLE_DEVICES
       # JAX_PLATFORMS=gpu

4. **For GPU workloads**: Use homodyne v2.2.1

See :doc:`/migration/v2.2-to-v2.3-gpu-removal` for complete migration guide (if available).

Troubleshooting
---------------

**Low performance on HPC**:

- Verify physical core count::

      python -c "from homodyne.device import detect_cpu_info; print(detect_cpu_info())"

- Check thread binding::

      # Should show affinity to specific cores
      taskset -p $$

- Benchmark performance::

      from homodyne.device import benchmark_device_performance
      results = benchmark_device_performance()

**Import errors**:

- Install optional dependencies::

      pip install psutil

- Without psutil, basic CPU configuration still works

**NUMA warnings**:

- Ignore on non-NUMA systems (laptops, desktops)
- On HPC, verify NUMA topology::

      numactl --hardware

See Also
--------

- :mod:`homodyne.optimization` - Uses device configuration for optimization
- :mod:`homodyne.core` - JAX computations on configured device
- External: `JAX CPU Performance Guide <https://jax.readthedocs.io/en/latest/faq.html#performance>`_
