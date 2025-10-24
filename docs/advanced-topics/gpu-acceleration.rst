GPU Acceleration
================

Overview
--------

Homodyne supports GPU acceleration on Linux systems with NVIDIA GPUs, providing 10-100x speedup for optimization and data processing.

**Key Features:**

- **Transparent Acceleration**: JAX handles GPU/CPU transparently
- **CUDA 12.1-12.9 Support**: Modern NVIDIA GPU support
- **Automatic Fallback**: CPU execution if GPU unavailable
- **Performance Benchmarking**: Tools to measure speedup
- **Device Selection**: Auto-detect optimal device

GPU Requirements and Support
----------------------------

**GPU Support Matrix:**

.. list-table::
    :header-rows: 1
    :widths: 20 30 20 30

    * - Platform
      - GPU Support
      - CUDA Required
      - Status
    * - Linux
      - Yes (NVIDIA)
      - 12.1-12.9
      - Full support
    * - macOS
      - No
      - N/A
      - CPU-only
    * - Windows
      - No
      - N/A
      - CPU-only

**NVIDIA Driver Requirements:**

- Driver version >= 525
- CUDA toolkit 12.1-12.9 (pre-installed on system)
- Compatible GPU (modern NVIDIA cards recommended)

Installation
------------

CPU-Only Installation (All Platforms)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Default installation works on all platforms:

.. code-block:: bash

    pip install homodyne

This uses CPU-only JAX by default.

GPU Installation (Linux Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU support on Linux with CUDA 12.2:

.. code-block:: bash

    # Step 1: Verify system CUDA is installed
    nvcc --version  # Should show CUDA 12.x.x

    # Step 2: Install JAX with GPU support
    pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

    # Step 3: Install homodyne (GPU-enabled JAX already installed)
    pip install homodyne

Using Makefile
~~~~~~~~~~~~~~

Simplified GPU installation with Makefile:

.. code-block:: bash

    # Development setup with GPU support
    cd homodyne/
    make install-jax-gpu

This is equivalent to the manual steps above.

Verification
~~~~~~~~~~~~

Verify GPU is available and working:

.. code-block:: bash

    # Check GPU detection
    make gpu-check

    # Or manually:
    python -c "import jax; print(jax.devices())"

Expected output (with GPU):

.. code-block:: text

    [gpu(id=0), gpu(id=1)]  # Multiple GPUs if available

Expected output (CPU-only):

.. code-block:: text

    [cpu]

Performance Benchmarking
------------------------

Automatic Device Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Homodyne automatically detects and uses GPU if available:

.. code-block:: yaml

    performance:
      device:
        preferred_device: "auto"  # auto | cpu | gpu
        gpu_memory_fraction: 0.9  # Use 90% of GPU memory

**Device Priority:**

1. If GPU available and configured: Use GPU
2. If GPU unavailable or broken: Fall back to CPU
3. If preferred_device: "cpu": Force CPU

Performance Benchmarking Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure GPU vs. CPU performance:

.. code-block:: bash

    python -c "from homodyne.device import benchmark_device_performance; print(benchmark_device_performance())"

Output example:

.. code-block:: text

    Device Performance Benchmark
    ============================
    CPU:  4.23 iterations/sec
    GPU:  87.5 iterations/sec
    Speedup: 20.7x

Configuration
--------------

GPU Configuration
~~~~~~~~~~~~~~~~~

In YAML configuration:

.. code-block:: yaml

    performance:
      device:
        preferred_device: "auto"      # Let homodyne choose
        gpu_memory_fraction: 0.8      # Use 80% of GPU memory

**Memory Fraction:**

- 0.5: Conservative, leaves headroom
- 0.8: Typical (recommended)
- 0.9: Aggressive, maximize usage

Manual Device Override
~~~~~~~~~~~~~~~~~~~~~~

Force GPU or CPU in config:

.. code-block:: yaml

    # Force GPU (fails if unavailable)
    performance:
      device:
        preferred_device: "gpu"

    # Force CPU
    performance:
      device:
        preferred_device: "cpu"

HPC Specific Guidance
---------------------

SLURM Job Script
~~~~~~~~~~~~~~~~

For SLURM-based HPC:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=homodyne_gpu
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1           # Request 1 GPU
    #SBATCH --time=02:00:00
    #SBATCH --mem=16GB

    # Load CUDA module
    module load cuda/12.2 cudnn/9.8

    # Install/setup homodyne
    pip install homodyne

    # Run analysis with GPU
    homodyne --config config.yaml --output-dir results_gpu

Multi-GPU Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-GPU nodes:

.. code-block:: yaml

    optimization:
      mcmc:
        num_chains: 8        # 8 chains in parallel
      cmc:
        enable: true         # Parallel per-angle optimization

    performance:
      device:
        preferred_device: "gpu"

JAX will automatically distribute work across available GPUs.

Monitoring GPU Usage
~~~~~~~~~~~~~~~~~~~~

Monitor GPU utilization during analysis:

.. code-block:: bash

    # Terminal 1: Run homodyne
    homodyne --config config.yaml --output-dir results

    # Terminal 2: Monitor GPU (real-time)
    watch -n 1 nvidia-smi

Look for:

- **GPU Memory**: Increasing during optimization
- **GPU Util**: > 90% (well-utilized)
- **Temperature**: < 80°C (healthy)

Troubleshooting
---------------

GPU Not Detected
~~~~~~~~~~~~~~~~

**Symptom**: "Only CPU available"

**Diagnosis:**

.. code-block:: bash

    nvidia-smi           # Check driver
    nvcc --version      # Check CUDA toolkit
    python -c "import jax; print(jax.devices())"

**Solutions:**

1. **Missing NVIDIA driver**
   - Install driver >= 525
   - Reboot after installation

2. **CUDA not installed**
   - Install CUDA 12.1-12.9 from NVIDIA
   - Load module: `module load cuda/12.2`

3. **JAX not GPU-enabled**
   - Reinstall JAX: `pip install jax[cuda12-local]==0.8.0`

Out of Memory (OOM) on GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: "CUDA out of memory" error

**Solutions:**

1. **Reduce GPU memory fraction**

   .. code-block:: yaml

       performance:
         device:
           gpu_memory_fraction: 0.7  # Use less

2. **Reduce batch size**

   .. code-block:: yaml

       performance:
         strategy_override: "standard"  # Smaller batches

3. **Use CPU instead**

   .. code-block:: yaml

       performance:
         device:
           preferred_device: "cpu"

Slow GPU Performance
~~~~~~~~~~~~~~~~~~~~

**Symptom**: GPU slower than CPU (unexpected)

**Common Causes:**

1. **GPU memory pressure** → Reduce gpu_memory_fraction
2. **Small dataset** → CPU is faster for small data
3. **GPU is busy** → Check other processes
4. **Data transfer overhead** → Check dataset is HDF5 (sequential access)

**Verification:**

.. code-block:: bash

    # Benchmark and compare
    make profile-nlsq  # Includes GPU profiling

Performance Tips
----------------

Maximize GPU Utilization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use multiple chains** for MCMC
2. **Enable CMC** for multi-angle data
3. **Process in batches** (automatically done)
4. **Keep GPU memory full** (but not overflowing)

Monitor with:

.. code-block:: bash

    nvidia-smi -l 1 -q -d MEMORY,UTILIZATION

Optimize for Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets > 100M points:

.. code-block:: yaml

    performance:
      strategy_override: "streaming"
      device:
        preferred_device: "gpu"
        gpu_memory_fraction: 0.85

    optimization:
      streaming:
        enable_checkpoints: true
        checkpoint_frequency: 10

See Also
--------

- :doc:`../user-guide/installation` - Installation details
- :doc:`streaming-optimization` - Large dataset handling
- :doc:`mcmc-uncertainty` - MCMC with GPU

References
----------

**JAX GPU:**
- GitHub: https://github.com/google/jax
- Documentation: https://jax.readthedocs.io/
- GPU Setup: https://jax.readthedocs.io/en/latest/installation.html

**NVIDIA CUDA:**
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- cuDNN: https://developer.nvidia.com/cudnn
