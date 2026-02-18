.. _api-cmc-backends:

==========================================
homodyne.optimization.cmc.backends
==========================================

The ``backends`` sub-package provides pluggable execution backends for CMC.
Each backend is responsible for distributing per-shard MCMC work across available
compute resources and combining the results.

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Backend
     - Description
   * - ``multiprocessing``
     - Spawns N worker processes (physical_cores/2 − 1); recommended for CPU
   * - ``pjit``
     - JAX ``pjit``-based backend for single-host multi-device execution
   * - ``pbs``
     - HPC batch scheduler backend (PBS/SLURM)
   * - ``base``
     - Abstract ``CMCBackend`` base class and shard combination utilities

Set ``backend_name: "auto"`` (default) to let Homodyne select the optimal
backend for the current environment.

----

Multiprocessing Backend
------------------------

The recommended backend for CPU-based systems. Key architecture:

1. **Worker spawn** — ``N = max(1, physical_cores // 2 − 1)`` worker processes
   are spawned using the ``spawn`` start method (required for JAX safety).
2. **Shared memory** — ``SharedDataManager`` shares config, parameter space,
   and time grids across workers via ``multiprocessing.shared_memory``,
   avoiding per-shard pickling overhead.
3. **XLA configuration** — each worker sets ``JAX_ENABLE_X64`` and configures
   ``XLA_FLAGS`` with ``--xla_force_host_platform_device_count=4`` *before*
   importing JAX, providing 4 virtual devices per worker for ``parallel`` chains.
4. **Thread environment** — ``OMP_NUM_THREADS`` is set to 1–2 per worker to
   prevent thread oversubscription. ``OMP_PROC_BIND`` and ``OMP_PLACES`` are
   cleared before spawning and restored afterwards.
5. **Adaptive polling** — the manager adjusts the result-queue poll interval
   based on shard activity, reducing CPU spin.
6. **Batch PRNG** — all shard random keys are pre-generated in a single JAX
   call before spawning, avoiding repeated JAX initialisation.

.. autoclass:: homodyne.optimization.cmc.backends.multiprocessing.MultiprocessingBackend
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

SharedDataManager
~~~~~~~~~~~~~~~~~

.. autoclass:: homodyne.optimization.cmc.backends.multiprocessing.SharedDataManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Chain Execution Methods
------------------------

Controls how MCMC chains are executed *within* each worker process.

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Method
     - Best For
     - Description
   * - ``"parallel"``
     - **CPU multiprocessing (default)**
     - Uses ``pmap`` across 4 virtual JAX devices per worker. Achieves full
       CPU utilisation with the multiprocessing backend.
   * - ``"vectorized"``
     - Single-process only
     - Uses ``vmap`` on a single device. Empirically 20× slower with the
       multiprocessing backend (101 s vs 4.9 s wall time).
   * - ``"sequential"``
     - Debugging
     - Runs chains one at a time. No parallelism.

.. warning::

   ``chain_method: "vectorized"`` is **NOT recommended** for the multiprocessing
   backend. Workers drop to 1–2 active CPUs because ``vmap`` does not distribute
   across the 4 virtual JAX devices. Always use ``"parallel"`` in production.

----

XLA Device Setup
----------------

Each worker configures XLA before importing JAX:

.. code-block:: python

   # Executed inside each worker process (before JAX import)
   import os
   os.environ["JAX_ENABLE_X64"] = "1"
   os.environ["XLA_FLAGS"] = (
       "--xla_force_host_platform_device_count=4"
   )
   import jax   # JAX sees 4 virtual CPU devices

This gives each worker 4 virtual devices for ``parallel`` chain execution.
The parent process restores its original environment after all workers have
been spawned.

.. code-block:: yaml

   optimization:
     cmc:
       backend_name: "auto"
       per_shard_mcmc:
         chain_method: "parallel"
         num_chains: 4

----

Heterogeneity Detection
------------------------

The multiprocessing backend uses **bounds-aware coefficient of variation (CV)**
to detect heterogeneous shards before combining posteriors.

For near-zero parameters (e.g., :math:`\dot\gamma_0 \sim 10^{-3}`), dividing
by the mean CV would artificially inflate the heterogeneity score. Instead,
the scale is computed relative to the parameter's bounds range:

.. code-block:: python

   # For near-zero params: scale = param_range * 0.01
   # Falls back to: scale = max(abs(mean), 1e-10)
   cv = std / scale

Shards with abnormally high cross-shard CV are flagged for review and may be
excluded before the consensus combination step.

----

Base Backend and Combination Utilities
----------------------------------------

.. autoclass:: homodyne.optimization.cmc.backends.base.CMCBackend
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autofunction:: homodyne.optimization.cmc.backends.base.combine_shard_samples

----

Usage Examples
--------------

Selecting the multiprocessing backend explicitly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   optimization:
     cmc:
       backend_name: "multiprocessing"
       per_shard_mcmc:
         chain_method: "parallel"
         num_chains: 4

Inspecting worker count
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil
   from homodyne.optimization.cmc.backends.multiprocessing import (
       MultiprocessingBackend,
   )

   backend = MultiprocessingBackend(config=cmc_config)
   n_workers = max(1, psutil.cpu_count(logical=False) // 2 - 1)
   print(f"Worker processes: {n_workers}")
   print(f"Virtual devices/worker: 4 (via xla_force_host_platform_device_count)")
   print(f"Total parallel chains: {n_workers * 4}")

Manually running a backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc.backends import select_backend
   from homodyne.optimization.cmc.config import CMCConfig

   config = CMCConfig()
   backend = select_backend(config.backend_name, config)

   result = backend.run(
       shards=prepared_shards,
       parameter_space=param_space,
       model=xpcs_model,
       initial_values=init_vals,
   )

----

.. automodule:: homodyne.optimization.cmc.backends.multiprocessing
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: MultiprocessingBackend, SharedDataManager
