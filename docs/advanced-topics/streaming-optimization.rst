Streaming Optimization
======================

Overview
--------

StreamingOptimizer handles datasets >100M points with constant memory footprint using intelligent batch processing, automatic checkpointing, and fault tolerance.

**Key Features:**

- **Unlimited Data**: Process datasets of any size
- **Constant Memory**: Memory usage independent of dataset size
- **Checkpointing**: Save and resume from interruptions
- **Fault Tolerance**: Automatic recovery from batch failures
- **Progress Tracking**: Real-time batch-level statistics

When to Use Streaming
---------------------

Automatic activation when:

- **Dataset > 100M points**
- **Memory constraints** in processing environment
- **Long-running optimizations** requiring fault tolerance
- **HPC batch jobs** with time limits

Manual activation for smaller datasets:

.. code-block:: yaml

    performance:
      strategy_override: "streaming"  # Force streaming mode

Configuration
--------------

Basic Setup
~~~~~~~~~~~

Default streaming configuration (auto-enabled for large datasets):

.. code-block:: yaml

    optimization:
      streaming:
        enable_checkpoints: true
        checkpoint_dir: "./checkpoints"
        checkpoint_frequency: 10
        resume_from_checkpoint: true

**Key Settings:**

- **enable_checkpoints**: Save progress after each batch
- **checkpoint_frequency**: Save every N batches
- **resume_from_checkpoint**: Auto-resume on restart

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

For production pipelines with fault tolerance:

.. code-block:: yaml

    optimization:
      streaming:
        # Checkpointing
        enable_checkpoints: true
        checkpoint_dir: "./checkpoints"
        checkpoint_frequency: 10
        resume_from_checkpoint: true
        keep_last_checkpoints: 3

        # Fault tolerance
        enable_fault_tolerance: true
        max_retries_per_batch: 2
        min_success_rate: 0.5

        # Monitoring
        enable_progress: true
        save_batch_statistics: true

**Checkpoint Management:**

- **keep_last_checkpoints**: Manage disk space (default 3)
- **checkpoint_frequency**: Balance between safety and I/O (default 10)

**Fault Tolerance:**

- **max_retries_per_batch**: Number of retry attempts (2-3 recommended)
- **min_success_rate**: Minimum successful batches (0.5-0.8)

Output Files
~~~~~~~~~~~~

Streaming produces standard result files plus checkpoint metadata:

.. code-block:: text

    output_dir/nlsq/
    ├── parameters.json
    ├── fitted_data.npz
    ├── convergence_metrics.json
    └── checkpoint_metadata.json

    checkpoints/
    ├── batch_0000.h5
    ├── batch_0010.h5
    ├── batch_0020.h5
    └── checkpoint_log.json

Performance Characteristics
---------------------------

Streaming Performance
~~~~~~~~~~~~~~~~~~~~~

**Memory Usage:**

- STANDARD: ~10x dataset size
- LARGE: ~5x dataset size
- CHUNKED: ~2x dataset size
- STREAMING: ~0.5-1x dataset size (constant)

**Time Overhead:**

- Checkpointing: < 2 seconds per save
- Total overhead: < 5% vs. single-pass
- Batch I/O: Optimized with HDF5 indexing

**Scalability:**

.. code-block:: text

    Dataset Size    Batches    Est. Runtime    Memory
    100M            10         5-10 min        2-4 GB
    500M            50         25-50 min       2-4 GB
    1B              100        50-100 min      2-4 GB

Workflow Examples
-----------------

Large Dataset Processing
~~~~~~~~~~~~~~~~~~~~~~~~

Standard workflow for large dataset:

.. code-block:: yaml

    # config_streaming.yaml
    experimental_data:
      file_path: "./data/500m_points.hdf"

    parameter_space:
      model: "laminar_flow"
      bounds:
        - name: D0
          min: 100.0
          max: 100000.0
        - name: alpha
          min: 0.0
          max: 2.0
        - name: D_offset
          min: -100.0
          max: 100.0
        - name: gamma_dot_0
          min: 1e-6
          max: 0.5
        - name: beta
          min: 0.0
          max: 2.0
        - name: gamma_dot_offset
          min: -0.1
          max: 0.1
        - name: phi_0
          min: -180.0
          max: 180.0

    optimization:
      method: "nlsq"
      nlsq:
        max_iterations: 100
        tolerance: 1e-8

    optimization:
      streaming:
        enable_checkpoints: true
        checkpoint_dir: "./checkpoints"
        enable_fault_tolerance: true

Run:

.. code-block:: bash

    homodyne --config config_streaming.yaml --output-dir results

Expected runtime: 50-100 minutes for 500M points

Checkpoint Resume
~~~~~~~~~~~~~~~~~

Resume interrupted optimization:

.. code-block:: bash

    # Initial run interrupted after 30 batches
    # homodyne --config config.yaml ...
    # (Ctrl+C or timeout)

    # Resume from last checkpoint
    homodyne --config config.yaml --output-dir results

Homodyne automatically:

1. Detects checkpoint in output directory
2. Loads best parameters so far
3. Resumes from next batch
4. Continues until completion
5. Produces final results

Checkpoint Metadata
~~~~~~~~~~~~~~~~~~~

View streaming progress:

.. code-block:: python

    import json

    with open('checkpoint_metadata.json') as f:
        metadata = json.load(f)

    print(f"Completed batches: {metadata['completed_batches']}")
    print(f"Successful batches: {metadata['successful_batches']}")
    print(f"Failed batches: {metadata['failed_batches']}")
    print(f"Success rate: {metadata['success_rate']:.1%}")
    print(f"Best chi_squared: {metadata['best_chi_squared']:.6f}")

Troubleshooting
---------------

Checkpoint Errors
~~~~~~~~~~~~~~~~~

**Error**: "Corrupted checkpoint"

**Solution:**

1. Delete corrupted checkpoint files
2. Set resume_from_checkpoint: false temporarily
3. Rerun (will start fresh)

Out of Memory During Streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Error**: OOM even with streaming enabled

**Solutions:**

1. Reduce batch size (reduce memory_limit_gb)
2. Use CPU-only mode (disable GPU)
3. Split data across multiple runs

Resume Not Working
~~~~~~~~~~~~~~~~~~~

**Symptom**: Optimization restarts instead of resuming

**Causes:**

- Checkpoint directory moved or deleted
- Different output directory specified
- resume_from_checkpoint: false in config

**Solution**: Ensure checkpoint_dir matches and resume_from_checkpoint: true

See Also
--------

- :doc:`nlsq-optimization` - Base NLSQ method
- :doc:`../user-guide/configuration` - Configuration system
- :doc:`../api-reference/optimization` - Optimization API

References
----------

**NLSQ Streaming:**
- GitHub: https://github.com/imewei/NLSQ
- Large Datasets Guide: https://nlsq.readthedocs.io/en/latest/guides/large_datasets.html
