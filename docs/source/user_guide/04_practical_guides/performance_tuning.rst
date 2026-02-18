.. _performance_tuning:

Performance Tuning: CPU/NUMA Optimization
==========================================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- How JAX uses CPU resources on multi-core machines
- XLA flags for optimal thread and device configuration
- NUMA-aware execution for multi-socket systems
- Memory management for large datasets
- Using ``homodyne-config-xla`` for configuration

---

Overview
---------

Homodyne is CPU-only by design. Performance on modern multi-core CPUs is
determined by:

1. **XLA thread count**: how many threads JAX uses per device
2. **Virtual JAX device count**: how many logical devices are exposed
3. **NUMA topology**: memory locality on multi-socket nodes
4. **Memory allocation**: preventing OOM errors for large datasets
5. **CMC worker count**: how many processes the multiprocessing backend spawns

---

homodyne-config-xla
--------------------

The ``homodyne-config-xla`` command provides pre-configured XLA settings
for common CPU configurations:

.. code-block:: bash

   # Show recommended settings for your hardware
   homodyne-config-xla --show

   # Apply settings for a standard workstation
   homodyne-config-xla --mode workstation

   # Apply settings for a 36-core HPC node
   homodyne-config-xla --mode hpc

   # Apply for a 128-core dual-socket node
   homodyne-config-xla --mode hpc_large

**Example output:**

.. code-block:: text

   Detected CPU: 16 logical cores, 8 physical cores, 1 NUMA node
   Recommended settings:
     XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
     XLA_FLAGS+=--xla_force_host_platform_device_count=4
     OMP_NUM_THREADS=2
     OPENBLAS_NUM_THREADS=2

---

XLA Flags
----------

The most important XLA flags for homodyne:

**xla_force_host_platform_device_count:**

Controls how many virtual CPU devices JAX creates. The CMC multiprocessing
backend uses ``--xla_force_host_platform_device_count=4`` per worker process
so that ``parallel`` chain execution can distribute 4 chains across 4 devices.

.. code-block:: python

   import os
   # Set BEFORE importing jax
   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
   import jax

   print(f"JAX devices: {jax.devices()}")  # [CpuDevice(id=0), ..., CpuDevice(id=3)]

**xla_cpu_multi_thread_eigen:**

Controls Eigen's internal multithreading. Setting to ``false`` prevents
thread oversubscription when homodyne's own parallelism is used:

.. code-block:: bash

   export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_force_host_platform_device_count=4"

---

OMP Thread Configuration
--------------------------

Homodyne's CMC multiprocessing backend carefully manages OMP threads to
prevent oversubscription:

- Before spawning workers: clears ``OMP_PROC_BIND`` and ``OMP_PLACES``
- Each worker: sets ``OMP_NUM_THREADS=1`` or ``2``
- After spawning: restores parent environment

You can set the default manually:

.. code-block:: bash

   # For 8 physical cores, 4 workers (2 cores per worker)
   export OMP_NUM_THREADS=2

Or let homodyne configure it automatically (recommended):

.. code-block:: bash

   homodyne-config-xla --mode workstation
   source ~/.homodyne_xla_config  # Apply the generated config

---

CMC Worker Count
-----------------

The number of CMC worker processes is determined automatically:

.. code-block:: text

   n_workers = max(1, physical_cores // 2 - 1)

For a 16-physical-core machine: n_workers = 7

Override manually:

.. code-block:: yaml

   optimization:
     cmc:
       num_workers: 4   # Use 4 workers (useful on shared nodes)

**Resource reservation:** Always leave at least 1 physical core for the
main process and OS.

---

NUMA Awareness
--------------

On dual-socket servers (2 NUMA nodes), memory locality matters.
Homodyne does not currently implement explicit NUMA pinning, but you can
use numactl to pin the process:

.. code-block:: bash

   # Run on NUMA node 0 only
   numactl --cpunodebind=0 --membind=0 \
     uv run homodyne --config config.yaml --output results/

   # For a 64-core dual-socket (32 cores per node):
   numactl --cpunodebind=0 --membind=0 \
     uv run homodyne --config config_half.yaml --output results_node0/ &
   numactl --cpunodebind=1 --membind=1 \
     uv run homodyne --config config_half.yaml --output results_node1/ &
   wait

This is faster than letting the OS distribute memory across NUMA nodes.

---

JAX Compilation Caching
------------------------

JAX JIT compilation can dominate execution time for small datasets. Cache
compiled functions between runs:

.. code-block:: bash

   # Enable XLA compilation cache
   export XLA_CACHE_DIR="$HOME/.cache/xla_compilations"
   mkdir -p "$XLA_CACHE_DIR"

With caching, the first run compiles; subsequent runs load from cache.
This reduces startup time from ~30–60 s to ~2–5 s.

---

Memory Profiling
-----------------

Profile memory usage to tune the memory_fraction threshold:

.. code-block:: python

   import tracemalloc
   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax

   tracemalloc.start()
   config = ConfigManager.from_yaml("config.yaml")
   data = load_xpcs_data("config.yaml")
   result = fit_nlsq_jax(data, config)
   current, peak = tracemalloc.get_traced_memory()
   tracemalloc.stop()

   print(f"Peak memory: {peak / 1e9:.2f} GB")

Alternatively, use the memory estimator:

.. code-block:: python

   from homodyne.optimization.nlsq import estimate_peak_memory_gb

   n_points = len(data['c2_exp'].ravel())
   n_params = 9  # For laminar_flow with auto mode
   peak = estimate_peak_memory_gb(n_points, n_params)
   print(f"Estimated peak memory: {peak:.1f} GB")

---

JAX Profiling (Advanced)
-------------------------

For CMC, enable JAX-level profiling to see XLA kernel execution times:

.. code-block:: yaml

   optimization:
     cmc:
       per_shard_mcmc:
         enable_jax_profiling: true
         jax_profile_dir: ./profiles/jax

Then view with TensorBoard:

.. code-block:: bash

   pip install tensorboard
   tensorboard --logdir=./profiles/jax

This shows XLA operation timelines, useful for identifying bottlenecks
in the NUTS sampler.

---

Performance Checklist
----------------------

Before running a long analysis:

- [ ] Check available RAM: ``free -h`` (Linux) or ``vm_stat`` (macOS)
- [ ] Set appropriate ``memory_fraction`` in config
- [ ] Use ``per_angle_mode: "auto"`` not ``individual`` for large n_phi
- [ ] Set ``chain_method: "parallel"`` in CMC config
- [ ] Use ``homodyne-config-xla`` to set XLA flags
- [ ] Enable XLA compilation cache for repeated runs
- [ ] Consider ``numactl`` for multi-socket systems

---

See Also
---------

- :doc:`configuration` — Full YAML configuration reference
- :doc:`../03_advanced_topics/streaming_mode` — Large dataset handling
- :doc:`../03_advanced_topics/bayesian_inference` — CMC worker configuration
- :doc:`../05_appendices/troubleshooting` — Memory and performance issues
