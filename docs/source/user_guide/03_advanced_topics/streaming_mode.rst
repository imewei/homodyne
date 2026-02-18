.. _streaming_mode:

Large Dataset Handling and Streaming
======================================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- How homodyne handles datasets that exceed available RAM
- The memory threshold and how to configure it
- The difference between stratified and hybrid streaming strategies
- Performance tips for large datasets

---

Overview
---------

XPCS datasets can be large: a typical experiment with 100 azimuthal angles,
1000 frames, and a 1 megapixel detector can produce :math:`O(10^8)` data points
in the flattened two-time correlation array.

Homodyne automatically selects the appropriate memory strategy:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Dataset Size
     - Strategy
     - Memory Usage
   * - < 10M points or < 75% RAM
     - **Stratified LS** (full-batch)
     - ~2 GB per million points
   * - > 10M points or > 75% RAM
     - **Hybrid Streaming**
     - ~2 GB fixed overhead
   * - Scale ratio > 1000
     - CMA-ES + NLSQ
     - Bounded

---

Memory Thresholds
------------------

The memory threshold is configurable:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.75    # Use streaming when data exceeds 75% of RAM
       # memory_threshold_gb: 48  # Or set an explicit GB limit

**How the threshold is computed:**

.. code-block:: python

   from homodyne.optimization.nlsq import (
       detect_total_system_memory,
       get_adaptive_memory_threshold,
       select_nlsq_strategy,
   )

   total_ram = detect_total_system_memory()   # GB
   threshold = get_adaptive_memory_threshold(memory_fraction=0.75)
   print(f"Total RAM: {total_ram:.1f} GB")
   print(f"Memory threshold: {threshold:.1f} GB")

   # Check which strategy will be selected
   from homodyne.optimization.nlsq import estimate_peak_memory_gb
   n_points = 5_000_000
   n_params = 9
   peak_mem = estimate_peak_memory_gb(n_points, n_params)
   decision = select_nlsq_strategy(n_points, n_params)
   print(f"Peak memory estimate: {peak_mem:.1f} GB")
   print(f"Selected strategy: {decision.strategy}")

---

Stratified LS (Full-Batch)
---------------------------

For datasets below the memory threshold, homodyne uses **angle-stratified
chunking**: data is divided into chunks by azimuthal angle, ensuring each
chunk contains a balanced representation of all angles. The full Jacobian is
computed in memory.

This strategy gives the best numerical precision because all data is used
simultaneously in the least-squares solve.

---

Hybrid Streaming
-----------------

For datasets above the memory threshold, homodyne uses **adaptive hybrid
streaming**:

1. Data is divided into blocks that fit in memory
2. Each block's gradient and Hessian contributions are accumulated
3. The accumulated Gauss-Newton system is solved without storing the full
   Jacobian

This reduces peak memory from O(n × p) to O(block_size × p).

.. note::

   Streaming mode is mathematically equivalent to full-batch mode for the
   Gauss-Newton update (no approximation). Only the memory access pattern
   changes.

**Streaming is activated automatically.** You do not need to set it manually.
The log will show:

.. code-block:: text

   [INFO] Strategy: hybrid_streaming (n=15M pts, est. peak=32.4 GB > threshold 28.0 GB)
   [INFO] Streaming block size: 500K points

---

Performance Tips for Large Datasets
--------------------------------------

**1. Set the memory fraction conservatively**

On shared HPC nodes where other jobs may be running:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.60   # Leave 40% for OS and other processes

**2. Use per_angle_mode: "auto" or "constant"**

``individual`` mode with many angles increases the Jacobian column count
(memory scales as O(n × p)):

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # +2 params, not +2*n_phi

**3. Reduce the angle count if possible**

If your dataset has many nearly-identical angles, subsetting to a representative
set reduces memory and computation time while preserving physical information.

**4. Enable XLA compilation for the streaming path**

Large datasets benefit from XLA's ability to fuse memory operations:

.. code-block:: bash

   homodyne-config-xla --mode hpc --show
   # Then apply the recommended XLA flags

See :doc:`../04_practical_guides/performance_tuning` for XLA configuration.

---

Monitoring Memory Usage
------------------------

.. code-block:: python

   import psutil
   import os

   def get_memory_usage_gb():
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / (1024**3)

   print(f"Before fit: {get_memory_usage_gb():.2f} GB")
   result = fit_nlsq_jax(data, config)
   print(f"After fit:  {get_memory_usage_gb():.2f} GB")

---

CMC Memory Considerations
--------------------------

CMC uses a different memory model: data is divided into **shards**, each of
which is processed independently by a worker process.

Each worker holds one shard in memory. The total memory footprint is:

.. code-block:: text

   Total memory = n_workers × (shard_size × n_params × 8 bytes) + overhead

For ``max_points_per_shard: "auto"`` with 5K points and 4 workers:

.. code-block:: text

   4 workers × 5,000 × 9 × 8 bytes = 1.4 MB (negligible)

The shard result storage (combining posteriors) is the main memory cost:

.. code-block:: text

   n_shards × n_chains × n_samples × n_params × 8 bytes
   Example: 1000 shards × 4 × 1500 × 9 × 8 bytes ≈ 430 MB

See :doc:`bayesian_inference` for CMC-specific memory guidance.

---

See Also
---------

- :doc:`../04_practical_guides/performance_tuning` — CPU/NUMA optimization
- :doc:`bayesian_inference` — CMC memory model
- :doc:`../05_appendices/troubleshooting` — Memory error troubleshooting
