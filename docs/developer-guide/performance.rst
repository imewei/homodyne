Performance Guide
=================

This guide covers profiling, optimization, and performance tuning for Homodyne development.

Overview
--------

Key performance considerations for Homodyne:

* **JAX Compilation**: JIT compilation overhead vs runtime speedup
* **GPU Acceleration**: Transparent GPU usage on Linux + CUDA 12+
* **Memory Management**: Strategy selection based on dataset size
* **Profiling Tools**: JAX profiler, memory_profiler, nvidia-smi

Quick Start
-----------

.. code-block:: bash

   # Check device status
   python -c "from homodyne.device import get_device_status; print(get_device_status())"

   # Benchmark performance
   python -c "from homodyne.device import benchmark_device_performance; print(benchmark_device_performance())"

   # Profile NLSQ optimization
   JAX_LOG_COMPILES=1 python examples/nlsq_demo.py

JAX Compilation
---------------

JIT Compilation Basics
~~~~~~~~~~~~~~~~~~~~~~~

JAX uses **Just-In-Time (JIT) compilation** via XLA to optimize computational graphs:

* **First call**: Compilation + execution (slow)
* **Subsequent calls**: Cached execution (fast)

**Example**:

.. code-block:: python

   from jax import jit
   import jax.numpy as jnp

   @jit
   def compute_residuals(params, data):
       """JIT-compiled function."""
       return jnp.sum((model(params) - data) ** 2)

   # First call: compiles + executes (~100ms)
   result1 = compute_residuals(params, data)

   # Second call: cached execution (~1ms)
   result2 = compute_residuals(params, data)

Detecting Recompilation
~~~~~~~~~~~~~~~~~~~~~~~~

**Check for unnecessary recompilations**:

.. code-block:: bash

   JAX_LOG_COMPILES=1 python your_script.py

**Output**:

.. code-block:: text

   Compiling compute_residuals for args (f32[5], f32[1000])
   Compiling compute_residuals for args (f32[5], f32[2000])  # ← Recompilation!

**Causes of recompilation**:

* Changing array shapes
* Changing array dtypes
* Python control flow inside JIT

**Solution**: Use static shapes or ``jax.lax`` control flow.

GPU Acceleration
----------------

Device Detection
~~~~~~~~~~~~~~~~

**Check GPU availability**:

.. code-block:: bash

   python -c "import jax; print(jax.devices())"

**Expected output**:

.. code-block:: text

   # GPU available:
   [cuda(id=0), cuda(id=1)]

   # CPU only:
   [cpu(id=0)]

**Configure optimal device**:

.. code-block:: python

   from homodyne.device import configure_optimal_device

   device = configure_optimal_device()
   print(f"Using: {device}")  # e.g., "cuda(id=0)"

GPU vs CPU Performance
~~~~~~~~~~~~~~~~~~~~~~

**Typical speedups** (large datasets):

* **Residual calculation**: 10-50x faster on GPU
* **G2 computation**: 50-100x faster on GPU
* **Full optimization**: 10-20x faster end-to-end

**Example benchmark**:

.. code-block:: python

   import time
   import jax

   # CPU
   with jax.default_device(jax.devices('cpu')[0]):
       start = time.time()
       result = optimize_large_dataset()
       cpu_time = time.time() - start

   # GPU
   with jax.default_device(jax.devices('gpu')[0]):
       start = time.time()
       result = optimize_large_dataset()
       gpu_time = time.time() - start

   print(f"Speedup: {cpu_time / gpu_time:.1f}x")

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

**Check GPU memory**:

.. code-block:: bash

   nvidia-smi

**Monitor during execution**:

.. code-block:: bash

   nvidia-smi -l 1  # Update every 1 second

**GPU memory tips**:

* Reduce batch size for limited VRAM
* Clear JAX caches: ``jax.clear_caches()``
* Use float32 instead of float64 if precision allows

Memory Management
-----------------

Strategy Selection
~~~~~~~~~~~~~~~~~~

Homodyne automatically selects optimization strategy based on dataset size:

* **< 1M points** → STANDARD (``curve_fit``)
* **1M-10M points** → LARGE (memory-optimized)
* **10M-100M points** → CHUNKED (with progress)
* **> 100M points** → STREAMING (constant memory)

**Memory formula**:

.. code-block:: python

   memory_gb = n_points × (1 + n_parameters) × 16 / (1024**3)
   # Factor of 16 = 8 bytes (float64) × 2 (overhead)

**Examples**:

* 1M points, 5 params: ~0.09 GB
* 10M points, 9 params: ~1.5 GB
* 100M points, 9 params: ~15 GB

Monitoring Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import psutil

   def get_memory_usage():
       """Get current memory usage in GB."""
       process = psutil.Process()
       return process.memory_info().rss / (1024**3)

   # Monitor
   print(f"Memory before: {get_memory_usage():.2f} GB")
   result = run_optimization()
   print(f"Memory after: {get_memory_usage():.2f} GB")

Profiling Tools
---------------

JAX Profiler
~~~~~~~~~~~~

**Profile JAX computations**:

.. code-block:: python

   import jax.profiler

   # Start profiling
   with jax.profiler.trace("/tmp/jax-trace"):
       result = run_optimization()

   # View in TensorBoard
   # tensorboard --logdir=/tmp/jax-trace

**Profiler shows**:

* Compilation time vs execution time
* GPU utilization
* Memory allocations
* Operation-level breakdowns

Memory Profiler
~~~~~~~~~~~~~~~

**Profile memory usage**:

.. code-block:: bash

   pip install memory_profiler

.. code-block:: python

   from memory_profiler import profile

   @profile
   def run_analysis():
       result = optimize_data()
       return result

   run_analysis()

**Output**:

.. code-block:: text

   Line #    Mem usage    Increment   Occurrences   Line Contents
   =====================================================================
        5   250.0 MiB   250.0 MiB           1   @profile
        6                                         def run_analysis():
        7  1800.0 MiB  1550.0 MiB           1       result = optimize_data()
        8  1850.0 MiB    50.0 MiB           1       return result

Make Profile Targets
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make profile-nlsq    # Profile NLSQ optimization
   make profile-mcmc    # Profile MCMC sampling
   make benchmark       # Run performance benchmarks

Optimization Tips
-----------------

For JAX Code
~~~~~~~~~~~~

**1. Vectorize with vmap**:

❌ **Slow** (Python loop):

.. code-block:: python

   results = []
   for angle in phi_angles:
       results.append(compute_g2_single(params, angle, t1, t2))
   result = jnp.stack(results)

✅ **Fast** (vectorized):

.. code-block:: python

   from jax import vmap

   compute_g2_vectorized = vmap(compute_g2_single, in_axes=(None, 0, None, None))
   result = compute_g2_vectorized(params, phi_angles, t1, t2)

**2. Keep data on GPU**:

❌ **Slow** (repeated transfers):

.. code-block:: python

   for i in range(1000):
       data_cpu = np.array(data_gpu)  # GPU → CPU
       result = process(data_cpu)
       data_gpu = jnp.array(result)    # CPU → GPU

✅ **Fast** (stay on GPU):

.. code-block:: python

   data_gpu = jnp.array(data)  # Transfer once
   for i in range(1000):
       data_gpu = process_jax(data_gpu)  # All on GPU

**3. Use static_argnums for constants**:

.. code-block:: python

   @partial(jit, static_argnums=(2,))
   def compute(params, data, n_iterations):
       """n_iterations is static (doesn't trigger recompilation)."""
       for i in range(n_iterations):  # OK: static
           data = update(params, data)
       return data

For Large Datasets
~~~~~~~~~~~~~~~~~~

**1. Use STREAMING strategy**:

.. code-block:: python

   from homodyne.optimization.nlsq_wrapper import NLSQWrapper

   wrapper = NLSQWrapper(enable_large_dataset=True)
   result = wrapper.fit(xdata, ydata, p0, bounds)  # Auto-selects STREAMING

**2. Enable checkpointing**:

.. code-block:: yaml

   optimization:
     streaming:
       enable_checkpoints: true
       checkpoint_frequency: 10
       checkpoint_dir: "./checkpoints"

**3. Monitor batch statistics**:

.. code-block:: python

   result = wrapper.fit(...)
   print(f"Batches processed: {result.n_batches}")
   print(f"Success rate: {result.success_rate:.2%}")

HPC Optimization
----------------

CPU Threading
~~~~~~~~~~~~~

**Optimize for multi-core CPUs**:

.. code-block:: bash

   # Set JAX thread count
   export JAX_NUM_THREADS=16

   # Or in Python
   import os
   os.environ['JAX_NUM_THREADS'] = '16'

**Match SLURM allocation**:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --cpus-per-task=16

   export JAX_NUM_THREADS=$SLURM_CPUS_PER_TASK
   python analysis.py

SLURM Job Script
~~~~~~~~~~~~~~~~

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=homodyne
   #SBATCH --nodes=1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=16
   #SBATCH --mem=32G
   #SBATCH --time=04:00:00

   # Load modules
   module load cuda/12.2 python/3.12

   # Activate environment
   source ~/envs/homodyne/bin/activate

   # Set memory limit (80% of allocated)
   export HOMODYNE_MEMORY_LIMIT=25.6  # 32 × 0.8

   # Run analysis
   homodyne --config config.yaml --output-dir ./results

Common Performance Issues
-------------------------

Issue: Slow First Run
~~~~~~~~~~~~~~~~~~~~~

**Symptom**: First optimization run is very slow

**Cause**: JAX compilation overhead

**Solution**: Compilation is one-time cost. Subsequent runs are fast.

Issue: Out of Memory
~~~~~~~~~~~~~~~~~~~~

**Symptom**: ``MemoryError`` or JAX OOM

**Solutions**:

1. Reduce dataset size (use STREAMING)
2. Reduce batch size
3. Clear JAX caches: ``jax.clear_caches()``
4. Use float32 instead of float64

Issue: Repeated Recompilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Multiple "Compiling..." messages

**Cause**: Changing array shapes/dtypes

**Solution**:

.. code-block:: python

   # Ensure consistent shapes
   data = jnp.array(data, dtype=jnp.float64)  # Explicit dtype
   data = data.reshape(-1, n_features)        # Explicit shape

Issue: GPU Not Used
~~~~~~~~~~~~~~~~~~~

**Symptom**: Slow performance despite GPU available

**Check**:

.. code-block:: python

   import jax
   print(jax.devices())  # Should show cuda(id=0)

**Solutions**:

1. Install JAX with CUDA: ``pip install jax[cuda12-local]==0.8.0``
2. Check CUDA version: ``nvcc --version`` (need 12.1-12.9)
3. Check driver: ``nvidia-smi`` (need >= 525)

Performance Targets
-------------------

**From specification** (v3.0):

* **STREAMING memory**: Constant (coefficient of variation < 20%)
* **Checkpoint save**: < 2 seconds
* **Fault tolerance overhead**: < 5%
* **Fast mode overhead**: < 1%

Benchmarking
------------

**Run benchmarks**:

.. code-block:: bash

   make benchmark

**Custom benchmark**:

.. code-block:: python

   import time

   def benchmark_optimization(n_points, n_params):
       """Benchmark optimization performance."""
       # Setup
       xdata, ydata, p0, bounds = generate_test_data(n_points, n_params)

       # Benchmark
       start = time.time()
       result = optimize(xdata, ydata, p0, bounds)
       elapsed = time.time() - start

       # Metrics
       points_per_sec = n_points / elapsed
       print(f"Throughput: {points_per_sec:.0f} points/sec")
       print(f"Total time: {elapsed:.2f} seconds")

       return elapsed

Resources
---------

* **JAX Performance Tips**: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
* **JAX Profiling**: https://jax.readthedocs.io/en/latest/profiling.html
* **Performance Tuning Guide**: ``docs/guides/performance_tuning.md``
* **Streaming Optimizer**: ``docs/guides/streaming_optimizer_usage.md``

Next Steps
----------

* **Architecture Guide**: Understand critical performance paths
* **Testing Guide**: Performance benchmarking tests
* **Contributing Guide**: Performance regression testing
