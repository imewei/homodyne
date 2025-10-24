Covariance Matrix Combination (CMC)
===================================

Overview
--------

Covariance Matrix Combination is a scalable approach for analyzing large datasets by splitting data into shards, running parallel per-angle optimizations, and combining results via weighted covariance matrix averaging.

**Key Features:**

- **Parallel Processing**: Optimize each phi angle independently
- **Scalability**: Linear speedup with number of shards/angles
- **Memory Efficient**: Each shard fits in available memory
- **Fault Tolerance**: Failed shards can be retried independently
- **Combined Uncertainties**: Propagate per-shard covariances to final estimates

When to Use CMC
----------------

Use CMC when:

- **Dataset > 1M points** with multiple phi angles
- **Multi-GPU/HPC available** for parallel execution
- **Per-angle results needed** in addition to combined estimates
- **Fault tolerance required** for long-running analyses

CMC is complementary to NLSQ (strategy selection) and MCMC (uncertainty quantification).

Configuration
--------------

Basic CMC Setup
~~~~~~~~~~~~~~~

Enable CMC for multi-angle large datasets:

.. code-block:: yaml

    optimization:
      cmc:
        enable: true
        backend: "jax"             # jax | numpy
        diagonal_correction: true   # Apply correction

**When enabled:**

- Homodyne automatically detects multi-angle data
- Optimizes each angle independently in parallel
- Combines results using weighted covariance averaging
- Provides per-angle and combined parameter estimates

Backend Selection
~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 25 40 35

    * - Backend
      - Use Case
      - Performance
    * - JAX (default)
      - GPU available, modern hardware
      - 10-100x faster on GPU
    * - NumPy
      - CPU-only, no JAX-GPU, fallback
      - Slower but reliable

**Recommendation**: Use JAX (default) on systems with GPU support.

Diagonal Correction
~~~~~~~~~~~~~~~~~~~~

The diagonal correction factor accounts for information lost when combining subposteriors:

.. code-block:: yaml

    optimization:
      cmc:
        diagonal_correction: true   # Recommended

- **true**: Apply correction (recommended)
- **false**: No correction (conservative, larger uncertainties)

Output Files
~~~~~~~~~~~~

CMC produces both per-angle and combined results:

.. code-block:: text

    output_dir/cmc/
    ├── parameters_combined.json      # Final combined estimates
    ├── parameters_per_angle_*.json   # Per-angle results
    ├── covariance_combined.npz       # Combined covariance matrix
    └── covariance_per_angle_*.npz    # Per-angle covariance matrices

Output Interpretation
---------------------

parameters_combined.json
~~~~~~~~~~~~~~~~~~~~~~~~

Final combined parameter estimates and uncertainties:

.. code-block:: json

    {
      "D0": {
        "value": 1250.5,
        "uncertainty": 42.3,
        "num_angles": 3,
        "combination_method": "weighted_covariance"
      },
      "alpha": {
        "value": 0.84,
        "uncertainty": 0.048,
        "num_angles": 3
      }
    }

This is the main result to report. Uncertainties represent combined information from all angles.

Covariance Matrices
~~~~~~~~~~~~~~~~~~~

Combined and per-angle covariance matrices:

.. code-block:: python

    import numpy as np

    # Load combined covariance
    cov_combined = np.load('covariance_combined.npz')
    print(cov_combined.files)  # Available matrices

    # Correlation matrix from covariance
    cov = cov_combined['covariance']
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)

    print(f"Correlation D0-alpha: {corr[0, 1]:.2f}")

Workflow Example
----------------

Multi-Angle Laminar Flow
~~~~~~~~~~~~~~~~~~~~~~~~

Typical workflow for flowing system with multiple angles:

.. code-block:: yaml

    # config_cmc.yaml
    experimental_data:
      file_path: "./data/10m_point_dataset.hdf"

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
      cmc:
        enable: true
        backend: "jax"
        diagonal_correction: true

Run:

.. code-block:: bash

    homodyne --config config_cmc.yaml --output-dir results_cmc

Expected runtime: 2-10 minutes (depends on number of angles and parallelization)

Interpreting Results
~~~~~~~~~~~~~~~~~~~~

Example comparison of per-angle vs. combined:

.. code-block:: python

    import json

    # Load per-angle results
    angle_0 = json.load(open('parameters_per_angle_0.json'))
    angle_1 = json.load(open('parameters_per_angle_1.json'))
    angle_2 = json.load(open('parameters_per_angle_2.json'))

    # Load combined
    combined = json.load(open('parameters_combined.json'))

    print("D0 Estimates:")
    print(f"  Angle 0: {angle_0['D0']['value']:.1f} ± {angle_0['D0']['uncertainty']:.1f}")
    print(f"  Angle 1: {angle_1['D0']['value']:.1f} ± {angle_1['D0']['uncertainty']:.1f}")
    print(f"  Angle 2: {angle_2['D0']['value']:.1f} ± {angle_2['D0']['uncertainty']:.1f}")
    print(f"  Combined: {combined['D0']['value']:.1f} ± {combined['D0']['uncertainty']:.1f}")

    # Combined should have smaller uncertainty (more information)

See Also
--------

- :doc:`nlsq-optimization` - NLSQ base method
- :doc:`streaming-optimization` - For extremely large datasets (100M+)
- :doc:`../user-guide/configuration` - Configuration details

References
----------

**Covariance Matrix Combination:**

- Scott, S.L., et al. (2016). "Bayes and Big Data: The Consensus Monte Carlo Algorithm" https://arxiv.org/abs/1411.7435

**Consensus Methods:**

- Neiswanger, W., et al. (2014). "Asymptotically exact, embarrassingly parallel MCMC" ICML 2014
