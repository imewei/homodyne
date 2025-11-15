Configuration Templates
=======================

Homodyne provides three comprehensive configuration templates organized by physical model, not analysis method. Each template includes **all optimization methods** (NLSQ, MCMC, streaming, CMC) and is ready for production use.

Template Selection Guide
------------------------

**Choose your template based on your physical system:**

.. list-table:: Template Decision Tree
   :header-rows: 1
   :widths: 20 30 25 25

   * - Template
     - Use When
     - Parameter Count
     - Typical Use Case
   * - **Master Template**
     - Learning all options
     - All (reference only)
     - Configuration reference
   * - **Static Isotropic**
     - No flow, single/isotropic angles
     - 3 + 2n
     - Equilibrium systems
   * - **Laminar Flow**
     - Shear flow, time-dependent dynamics
     - 7 + 2n
     - Nonequilibrium systems

Parameter Counting Explained
-----------------------------

**Critical Concept:** The total parameter count depends on both the physical model and the number of filtered phi angles.

Static Isotropic Model
~~~~~~~~~~~~~~~~~~~~~~

**Formula:** 3 physical + 2 × (number of filtered phi angles)

**Physical parameters (3):**
  - D₀: Initial diffusion coefficient [Å²/s]
  - α: Power-law exponent [dimensionless]
  - D_offset: Diffusion offset [Å²/s]

**Per-angle scaling (2 per angle):**
  - contrast: Contrast parameter for each phi angle
  - offset: Offset parameter for each phi angle

**Example:**
  - 3 filtered angles → 3 + 2×3 = **9 total parameters**
  - 5 filtered angles → 3 + 2×5 = **13 total parameters**

Laminar Flow Model
~~~~~~~~~~~~~~~~~~

**Formula:** 7 physical + 2 × (number of filtered phi angles)

**Physical parameters (7):**
  - D₀: Initial diffusion coefficient [Å²/s]
  - α: Diffusion power-law exponent [dimensionless]
  - D_offset: Diffusion offset [Å²/s]
  - γ̇₀: Initial shear rate [s⁻¹]
  - β: Shear rate power-law exponent [dimensionless]
  - γ̇_offset: Shear rate offset [s⁻¹]
  - φ₀: Initial angle between flow and scattering vector [degrees]

**Per-angle scaling (2 per angle):**
  - contrast: Contrast parameter for each phi angle
  - offset: Offset parameter for each phi angle

**Example:**
  - 3 filtered angles → 7 + 2×3 = **13 total parameters**
  - 5 filtered angles → 7 + 2×5 = **17 total parameters**

Why Per-Angle Scaling?
~~~~~~~~~~~~~~~~~~~~~~~

Each phi angle in your experimental data may have different:
  - **Contrast:** Signal strength variation with angle
  - **Offset:** Baseline correlation value at each angle

These scaling parameters ensure accurate fits across all measured angles while the physical parameters (D₀, α, etc.) remain consistent across angles.

Template Overview
-----------------

All templates include:
  - Complete parameter space definitions with physically reasonable bounds
  - All optimization methods: NLSQ (trust-region), MCMC (NumPyro/BlackJAX), streaming (unlimited data), CMC (multi-angle combination)
  - Automatic strategy selection based on dataset size (<1M, 1M-10M, 10M-100M, >100M points)
  - GPU/CPU device management with automatic selection
  - Comprehensive logging and output configuration
  - Angle filtering for parameter count reduction
  - Checkpoint/resume for long-running optimizations

Quick Start
-----------

1. **Choose your template** based on your system:

   - Static system? Use :doc:`static-isotropic`
   - Flow system? Use :doc:`laminar-flow`
   - Learning? Browse :doc:`master-template`

2. **Copy and customize:**

   .. code-block:: bash

      cp homodyne/config/templates/homodyne_static_isotropic.yaml my_config.yaml
      # Edit my_config.yaml: update file_path, adjust parameter bounds

3. **Run analysis:**

   .. code-block:: bash

      homodyne --config my_config.yaml --method nlsq

Template Documentation
----------------------

.. toctree::
   :maxdepth: 1

   master-template
   static-isotropic
   laminar-flow

Optimization Method Selection
------------------------------

**NLSQ (Nonlinear Least Squares)**
  - **When:** Primary analysis, deterministic results
  - **Speed:** Fast (seconds to minutes)
  - **Output:** Point estimates with uncertainties
  - **Recommended for:** Initial parameter estimation, production workflows

**MCMC (Markov Chain Monte Carlo)**
  - **When:** Uncertainty quantification, posterior distributions
  - **Speed:** Slow (minutes to hours)
  - **Output:** Full posterior samples with correlations
  - **Recommended for:** Publication-quality results, model comparison

**Streaming Optimization**
  - **When:** Very large datasets (>100M points)
  - **Speed:** Moderate (automatic batch processing)
  - **Output:** Same as NLSQ with fault tolerance
  - **Recommended for:** Memory-constrained environments, checkpoint/resume capability

**CMC (Covariance Matrix Combination)**
  - **When:** Large multi-angle datasets (>1M points)
  - **Speed:** Fast (parallel optimization)
  - **Output:** Combined parameter estimates across angles
  - **Recommended for:** Multi-angle analysis with limited resources

Template Migration (v1.0 → v2.0+)
----------------------------------

**Old (v1.0) - 4 method-based templates (DEPRECATED):**

.. code-block:: text

   homodyne_default_comprehensive.yaml → deprecated/
   homodyne_cmc_config.yaml            → deprecated/
   homodyne_streaming_config.yaml      → deprecated/
   homodyne_static_isotropic.yaml      → Updated to include ALL methods

**New (v2.0+) - 3 model-based templates:**

.. code-block:: text

   homodyne_master_template.yaml      → Comprehensive reference (ALL parameters + ALL methods)
   homodyne_static_isotropic.yaml     → 3+2n parameters (ALL methods included)
   homodyne_laminar_flow.yaml         → 7+2n parameters (ALL methods included)

**Key Change:** Templates now organized by **physical model** (static vs laminar), not by **analysis method** (CMC, streaming, etc.). Each template includes configuration for ALL optimization methods.

See Also
--------

- :doc:`../user-guide/configuration` - Configuration system overview
- :doc:`../theoretical-framework/parameter-models` - Parameter models explained
- :doc:`../advanced-topics/nlsq-optimization` - NLSQ optimization workflows
- :doc:`../advanced-topics/mcmc-uncertainty` - MCMC uncertainty quantification
- :doc:`../advanced-topics/streaming-optimization` - Streaming for large datasets
- :doc:`../advanced-topics/cmc-large-datasets` - CMC for multi-angle analysis
