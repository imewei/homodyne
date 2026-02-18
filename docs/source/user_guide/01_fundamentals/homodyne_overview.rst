.. _homodyne_overview:

Homodyne Package Overview
=========================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- What homodyne does and what it does not do
- Key design decisions (JAX-first, CPU-only, two optimization methods)
- The supported analysis modes and their differences
- How homodyne fits into the broader XPCS analysis ecosystem

---

What Homodyne Does
-------------------

Homodyne is a **Python package for fitting two-time correlation functions**
measured in homodyne XPCS experiments. Given experimental :math:`C_2(t_1, t_2)`
data as input, it returns estimates of the physical parameters governing particle
dynamics—diffusion coefficients, shear rates, and related quantities.

The package solves the **inverse problem**: given measured correlations, find
the parameter values that best explain the data.

Supported analyses:

- **Static mode**: diffusion coefficient :math:`D_0`, anomalous exponent :math:`\alpha`,
  diffusion offset :math:`D_\text{offset}` — 3 free parameters
- **Laminar flow mode**: all static parameters plus shear rate :math:`\dot\gamma_0`,
  shear exponent :math:`\beta`, shear offset :math:`\dot\gamma_\text{offset}`,
  angular offset :math:`\phi_0` — 7 free parameters
- **Per-angle scaling**: optional per-angle contrast and offset corrections
  (adds 0, 2, or 2×n_phi parameters depending on mode)

Two optimization methods are provided:

- **NLSQ** (primary): trust-region nonlinear least squares — fast, reliable
- **CMC** (secondary): Consensus Monte Carlo with NumPyro/NUTS — full Bayesian
  posterior with uncertainty quantification

What Homodyne Does Not Do
---------------------------

Understanding the package boundaries helps avoid misuse:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Not in scope
     - What to use instead
   * - Raw detector data reduction (pixel masking, flat-field)
     - Beamline-specific software (pynx, pyFAI, xi-cam)
   * - Computing :math:`C_2` from raw intensity frames
     - XPCS reduction pipelines (e.g., pyxpcs, multitau).
       The :math:`C_2` array must already be computed before calling homodyne.
   * - GPU acceleration
     - By design: CPU-only for reproducibility on HPC batch nodes
   * - Heterodyne (XSVS, XCCA) analysis
     - Different model equations not yet implemented
   * - Multi-angle or multi-q simultaneous fitting
     - Each q-value is fitted independently

Key Design Decisions
----------------------

**JAX-first computation**

All numerical computations use `JAX <https://github.com/google/jax>`_, which
provides:

- JIT compilation of compute-intensive loops to native CPU code
- Automatic differentiation for Jacobian computation
- Functional programming style that avoids global state
- Reproducible results through explicit random seeds

**CPU-only architecture**

Homodyne targets CPU execution rather than GPU. This is a deliberate choice:

- XPCS datasets fit comfortably in CPU RAM (typically 1–100 GB)
- CPU clusters (HPC batch nodes) are universally available
- CPU execution avoids CUDA versioning and driver dependency issues
- NumPyro's multiprocessing backend distributes chains across CPU cores efficiently

**Two-method strategy**

.. list-table::
   :header-rows: 1
   :widths: 15 35 35 15

   * - Method
     - Best For
     - Limitations
     - Speed
   * - NLSQ
     - Initial exploration, production runs, large datasets
     - Point estimates only; uncertainty via covariance approximation
     - Fast (seconds to minutes)
   * - CMC
     - Publication results, multi-modal posteriors, uncertainty propagation
     - Slower; requires NLSQ warm-start for best results
     - Minutes to hours

The recommended workflow is: **NLSQ first for rapid exploration, CMC for
publication-quality uncertainty quantification**.

**No silent data loss**

Homodyne enforces strict validation at all I/O boundaries:

- Shape, dtype, NaN, and monotonicity checks on input data
- Explicit errors on invalid configurations
- No silent downsampling or truncation of data
- All parameter bounds are enforced

Package Architecture
---------------------

The package is organized into focused modules:

.. code-block:: text

   homodyne/
   ├── core/               # Physics engine (JAX)
   │   ├── jax_backend.py      # JIT-compiled C2 computations (NLSQ path)
   │   ├── physics_cmc.py      # Element-wise model (CMC path)
   │   ├── physics.py          # Constants, bounds, validation
   │   ├── models.py           # DiffusionModel, ShearModel, CombinedModel
   │   ├── theory.py           # g1, g2 analytical expressions
   │   └── homodyne_model.py   # Unified model interface
   ├── optimization/
   │   ├── nlsq/               # Primary: trust-region L-M optimizer
   │   │   ├── core.py             # fit_nlsq_jax() entry point
   │   │   ├── adapter.py          # NLSQAdapter (CurveFit class)
   │   │   ├── wrapper.py          # NLSQWrapper (full features)
   │   │   └── cmaes_wrapper.py    # CMA-ES global optimization
   │   └── cmc/                # Secondary: Consensus Monte Carlo
   │       ├── core.py             # fit_mcmc_jax() entry point
   │       ├── model.py            # NumPyro probabilistic models
   │       ├── sampler.py          # SamplingPlan, NUTS execution
   │       └── backends/           # multiprocessing, pjit backends
   ├── data/               # HDF5 loading (XPCSDataLoader)
   ├── config/             # YAML config (ConfigManager)
   ├── cli/                # Command-line interface
   └── device/             # CPU/NUMA configuration

Data Flow
-----------

A complete analysis pipeline:

.. code-block:: text

   YAML config
       |
       v
   ConfigManager.from_yaml()      # Parse and validate configuration
       |
       v
   XPCSDataLoader.load()          # Load HDF5 data, validate arrays
       |
       v
   fit_nlsq_jax(data, config)     # Trust-region NLSQ optimization
       |                          # Returns OptimizationResult
       v
   fit_mcmc_jax(data, config,     # Optional: Bayesian CMC
       nlsq_result=result)        # Returns CMCResult with posteriors
       |
       v
   Result saved as JSON + NPZ     # Persistent output

Comparison with Other XPCS Analysis Tools
-------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Package
     - Scope
     - Backend
     - Mode
     - Uncertainty
   * - **homodyne**
     - C2 model fitting
     - JAX/CPU
     - Static + shear
     - NLSQ + Bayesian
   * - pyxpcs
     - Data reduction + g2
     - NumPy
     - g2 only
     - None
   * - lmfit
     - General curve fitting
     - SciPy
     - User-defined
     - NLSQ only
   * - emcee
     - General Bayesian
     - NumPy
     - User-defined
     - MCMC

Homodyne is specialized for the XPCS two-time correlation fitting problem.
It is not a general fitting framework.

Installation
--------------

.. code-block:: bash

   # Install with uv (recommended)
   uv sync

   # Or with pip
   pip install homodyne

   # Verify installation
   homodyne --help
   homodyne-config --help

After installation, set up shell completion and aliases:

.. code-block:: bash

   homodyne-post-install
   # Then reload your shell: source ~/.zshrc  or  exec $SHELL

Version Information
---------------------

.. code-block:: python

   import homodyne
   print(homodyne.__version__)

   # Check optimization backend availability
   from homodyne.optimization import OPTIMIZATION_STATUS
   print(OPTIMIZATION_STATUS)

---

See Also
---------

- :doc:`what_is_xpcs` — Background on the XPCS technique
- :doc:`analysis_modes` — Detailed guide to static vs laminar flow mode
- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ fitting workflow
- :doc:`../03_advanced_topics/bayesian_inference` — CMC workflow
