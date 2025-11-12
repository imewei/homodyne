Examples and Tutorials
======================

This section provides real-world workflow examples for common Homodyne analysis scenarios.

Example 1: Static Isotropic NLSQ Optimization
----------------------------------------------

**Use case:** Analyzing static, isotropic systems with NLSQ trust-region optimization.

**Example script:** ``examples/static_isotropic_nlsq.py``

**Configuration template:** :doc:`../configuration-templates/static-isotropic`

**Key concepts:**

- 3+2n parameters (3 physical + 2 per angle)
- NLSQ for fast point estimation
- Suitable for small to medium datasets (< 100M points)

**Quick start:**

.. code-block:: bash

   python examples/static_isotropic_nlsq.py
   homodyne --config homodyne_config_static_isotropic.yaml

**Expected output:**

- ``parameters.json`` - Optimized D₀, α, D_offset values
- ``fitted_data.npz`` - Experimental vs. theoretical correlation data
- ``analysis_results_nlsq.json`` - Fit quality metrics

**Learning outcomes:**

- How to set up static isotropic configuration
- Interpreting NLSQ convergence metrics
- Reading parameter uncertainties

Example 2: Laminar Flow with Angle Filtering
----------------------------------------------

**Use case:** Analyzing flowing systems with selective angle analysis.

**Example script:** ``examples/laminar_flow_nlsq.py``

**Configuration template:** :doc:`../configuration-templates/laminar-flow`

**Key concepts:**

- 7+2n parameters (7 physical + 2 per angle)
- Angle filtering to reduce parameter count
- Extracting flow parameters (γ̇₀, β, γ̇_offset)
- φ₀ parameter for angle-dependent analysis

**Quick start:**

.. code-block:: bash

   python examples/laminar_flow_nlsq.py
   homodyne --config homodyne_config_laminar_flow.yaml

**Configuration highlights:**

.. code-block:: yaml

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
       - min_angle: 85.0
         max_angle: 95.0

This filters to 2 angle ranges, reducing parameters to 7 + 2×2 = 11.

**Expected output:**

- Flow parameters: γ̇₀ (shear rate), β (exponent), φ₀ (angle)
- Per-angle analysis with contrast and offset
- Angle-dependent dynamics visualization

**Learning outcomes:**

- Configuring angle filtering
- Understanding per-angle scaling parameters
- Extracting physical flow parameters
- Analyzing anisotropic dynamics

Example 3: MCMC Uncertainty Quantification
-------------------------------------------

**Use case:** Obtaining posterior distributions and uncertainty estimates.

**Example script:** ``examples/mcmc_uncertainty.py``

**Configuration settings:**

.. code-block:: yaml

   optimization:
     method: "mcmc"
     mcmc:
       num_warmup: 1000
       num_samples: 2000
       num_chains: 4
       progress_bar: true
       backend: "numpyro"

**Key concepts:**

- MCMC sampling vs NLSQ point estimates
- NumPyro NUTS sampler for efficient sampling
- Convergence diagnostics (R-hat, ESS)
- Posterior distributions and credible intervals

**Quick start:**

.. code-block:: bash

   python examples/mcmc_uncertainty.py
   homodyne --config config_mcmc.yaml --method mcmc

**Expected output:**

- ``parameters.json`` - Mean, median, std of posterior
- ``corner_plot.png`` - Joint posterior distributions
- ``trace_plots.png`` - MCMC chain traces
- ``mcmc_diagnostics.json`` - R-hat and ESS values

**Interpretation:**

- R-hat ≈ 1.0 indicates convergence
- ESS >> 1 indicates efficient sampling
- Corner plots show correlations between parameters

**Learning outcomes:**

- When to use MCMC vs NLSQ
- Reading convergence diagnostics
- Interpreting posterior distributions
- Parameter correlations and uncertainties

Example 4: Large Dataset with CMC
----------------------------------

**Use case:** Analyzing large datasets (>1M points) using Covariance Matrix Combination.

**Example script:** ``examples/cmc_large_dataset.py``

**Configuration:**

.. code-block:: yaml

   optimization:
     cmc:
       enable: true
       backend: "jax"           # GPU-accelerated
       diagonal_correction: true

**Key concepts:**

- Parallel optimization across angles
- Covariance matrix combination for combined estimate
- Diagonal correction for better covariance estimates
- Suitable for 1M-100M point datasets

**Quick start:**

.. code-block:: bash

   python examples/cmc_large_dataset.py
   homodyne --config config_cmc.yaml

**Expected output:**

- Combined parameter estimates across all angles
- Covariance matrices for uncertainty
- Per-angle optimization results
- Computational speedup vs NLSQ

**Learning outcomes:**

- CMC workflow and benefits
- Parallel angle optimization
- Covariance matrix handling
- Performance scaling with dataset size

Example 5: Streaming Optimization for 100M+ Points
---------------------------------------------------

**Use case:** Analyzing very large datasets (>100M points) with constant memory footprint.

**Example script:** ``examples/streaming_100m_points.py``

**Configuration:**

.. code-block:: yaml

   performance:
     strategy_override: "streaming"

   optimization:
     streaming:
       enable_checkpoints: true
       checkpoint_dir: "./checkpoints"
       checkpoint_frequency: 10
       enable_fault_tolerance: true

**Key concepts:**

- StreamingOptimizer for unlimited dataset sizes
- Constant memory usage regardless of data size
- Checkpoint/resume capability
- Automatic error recovery and retry logic

**Quick start:**

.. code-block:: bash

   python examples/streaming_100m_points.py
   homodyne --config config_streaming.yaml

**Expected output:**

- Successfully optimized very large datasets
- Checkpoint files for recovery
- Batch-level statistics
- Fault tolerance diagnostics

**Learning outcomes:**

- Streaming optimization setup
- Checkpoint management
- Memory-efficient processing
- Error recovery strategies

See :doc:`../advanced-topics/streaming-optimization` for detailed guide.

Example 6: GPU Acceleration
----------------------------

**Use case:** Leveraging GPU computation for performance (Linux only).

**Example script:** ``examples/gpu_acceleration.py``

**Configuration:**

.. code-block:: yaml

   performance:
     device:
       preferred_device: "gpu"
       gpu_memory_fraction: 0.9

**Key concepts:**

- Automatic GPU detection and usage
- GPU memory management
- Performance benchmarking
- CPU fallback on unavailable GPU

**Quick start (Linux only):**

.. code-block:: bash

   # Verify GPU available
   python -c "import jax; print(jax.devices())"

   # Run with GPU acceleration
   python examples/gpu_acceleration.py
   homodyne --config config.yaml

**Expected output:**

- GPU computation speedup (2-10x for large datasets)
- Device selection metrics
- Performance comparison (GPU vs CPU)

**Learning outcomes:**

- GPU acceleration setup and requirements
- Performance expectations
- When GPU helps (large datasets) vs hurts (small datasets)
- Device selection strategies

See :doc:`../advanced-topics/gpu-acceleration` for detailed setup.

Example 7: Angle Filtering for Anisotropic Analysis
-----------------------------------------------------

**Use case:** Analyzing direction-dependent dynamics with selective angle filtering.

**Example script:** ``examples/angle_filtering.py``

**Configuration:**

.. code-block:: yaml

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Parallel to flow (0°)"
       - min_angle: 85.0
         max_angle: 95.0
         description: "Perpendicular to flow (90°)"

**Key concepts:**

- Phi angle filtering reduces parameter count
- Angle-specific analysis (2n scaling parameters per range)
- Handling wrap-around at ±180° boundary
- Improving convergence through parameter reduction

**Quick start:**

.. code-block:: bash

   python examples/angle_filtering.py

Solver/Post-hoc Diagonal Overlay
--------------------------------

**Example script:** ``examples/overlay_solver_vs_posthoc.py``

This helper compares diagonals from the experimental cube, the solver-evaluated
surface, and the legacy post-hoc surface saved inside ``fitted_data.npz``.

.. code-block:: bash

   python examples/overlay_solver_vs_posthoc.py homodyne_results/nlsq/fitted_data.npz --phi-index 0
   homodyne --config config_angle_filtering.yaml

**Expected output:**

- Separate analysis for each angle range
- Angle-specific parameters
- Reduced total parameter count: 3 + 2×(number of ranges)

**Learning outcomes:**

- Angle filtering configuration
- Parameter count reduction impact
- Angle normalization and wrapping
- Anisotropic system analysis

See :doc:`../advanced-topics/angle-filtering` for detailed guide.

Example 8: Shell Completion Setup
----------------------------------

**Use case:** Setting up bash/zsh shell completion for faster CLI usage.

**Example script:** ``examples/setup_shell_completion.sh``

**Quick start:**

.. code-block:: bash

   bash examples/setup_shell_completion.sh

Or manually:

.. code-block:: bash

   homodyne --generate-completion bash > ~/.homodyne-completion.bash
   echo "source ~/.homodyne-completion.bash" >> ~/.bashrc
   source ~/.bashrc

**Learning outcomes:**

- Shell completion installation
- Auto-completion of homodyne commands
- Faster CLI workflow

See :doc:`shell-completion` for detailed setup.

Workflow Comparison
-------------------

**Static Isotropic + NLSQ** (Example 1)

- **Parameters:** 3 + 2n
- **Speed:** Fast
- **Memory:** Low
- **Use case:** Quick analysis, static systems

**Laminar Flow + Angle Filtering** (Example 2)

- **Parameters:** 7 + 2m (m = number of angle ranges)
- **Speed:** Medium
- **Memory:** Medium
- **Use case:** Flow systems, anisotropic analysis

**MCMC Uncertainty** (Example 3)

- **Parameters:** Same as above + posterior distributions
- **Speed:** Slow (1000s samples)
- **Memory:** Medium
- **Use case:** Uncertainty quantification, posterior analysis

**CMC Large Dataset** (Example 4)

- **Parameters:** Same as above
- **Speed:** Fast (parallel angles)
- **Memory:** High
- **Use case:** Large datasets (1M-100M points)

**Streaming 100M+** (Example 5)

- **Parameters:** Same as above
- **Speed:** Medium
- **Memory:** Constant (low)
- **Use case:** Very large datasets (>100M points)

**GPU Acceleration** (Example 6)

- **Speedup:** 2-10x for large datasets
- **Requirements:** Linux + CUDA 12.1-12.9
- **Best for:** Datasets > 10M points
- **Use case:** High-throughput analysis

Running All Examples
--------------------

To run all examples:

.. code-block:: bash

   cd /path/to/homodyne
   python examples/static_isotropic_nlsq.py
   python examples/laminar_flow_nlsq.py
   python examples/mcmc_uncertainty.py
   python examples/cmc_large_dataset.py
   python examples/streaming_100m_points.py
   python examples/gpu_acceleration.py
   python examples/angle_filtering.py
   bash examples/setup_shell_completion.sh

Next Steps
----------

After exploring examples:

1. **Customize for your data:**
   - Adapt template configuration for your system
   - Adjust parameter bounds based on your physics
   - Apply angle filtering if needed

2. **Advanced techniques:**
   - :doc:`../advanced-topics/index` - Advanced optimization guides
   - :doc:`../api-reference/index` - Full API documentation
   - :doc:`../developer-guide/performance` - Performance optimization

3. **Troubleshooting:**
   - :doc:`../developer-guide/architecture` - System architecture
   - :doc:`../theoretical-framework/index` - Physical equations
   - :doc:`configuration` - Configuration parameters

See Also
--------

- :doc:`quickstart` - 5-minute quick start guide
- :doc:`../configuration-templates/index` - Configuration templates
- :doc:`../theoretical-framework/index` - Physical framework
- PNAS paper: https://doi.org/10.1073/pnas.2401162121
