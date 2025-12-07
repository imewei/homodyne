Optimization Module
===================

The :mod:`homodyne.optimization` module provides two complementary optimization methods for parameter estimation in homodyne scattering analysis:

1. **NLSQ** (Primary): Fast, reliable trust-region optimization using Levenberg-Marquardt
2. **CMC** (Secondary): Bayesian uncertainty quantification using Consensus Monte Carlo

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Optimization Philosophy** (v2.4.1):

- NLSQ as primary method for fast parameter estimation
- CMC (NumPyro/NUTS) for publication-quality uncertainty quantification
- CPU-optimized architecture (v2.3.0+)
- Dataset size-aware strategy selection

**Method Comparison**:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Use Case
     - Performance
   * - NLSQ
     - Fast parameter estimation, exploratory analysis
     - ~seconds for 1M points
   * - CMC
     - Uncertainty quantification, publication figures
     - ~minutes (parallelized)

Module Contents
---------------

.. automodule:: homodyne.optimization
   :noindex:

Primary Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.optimization.fit_nlsq_jax
   homodyne.optimization.fit_mcmc_jax
   homodyne.optimization.get_optimization_info

NLSQ: Non-Linear Least Squares
-------------------------------

Trust-region optimization using the Levenberg-Marquardt algorithm, implemented via the ``nlsq`` package.

Core Module
~~~~~~~~~~~

.. automodule:: homodyne.optimization.nlsq.core
   :members:
   :undoc-members:
   :show-inheritance:

Wrapper
~~~~~~~

High-level interface with automatic strategy selection.

.. automodule:: homodyne.optimization.nlsq.wrapper
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
^^^^^^^^^^^^

- Automatic strategy selection based on dataset size
- Memory-aware chunking for large datasets
- JIT-compiled residual functions
- Stratified sampling for per-angle scaling

Results
~~~~~~~

.. automodule:: homodyne.optimization.nlsq.results
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

The NLSQ module implements multiple optimization strategies for different dataset sizes:

.. automodule:: homodyne.optimization.nlsq.strategies
   :members:
   :undoc-members:
   :show-inheritance:

Strategy Selection
^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.selection
   :members:
   :undoc-members:
   :show-inheritance:

Chunking Strategy
^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.chunking
   :members:
   :undoc-members:
   :show-inheritance:

Residual Functions
^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.residual
   :members:
   :undoc-members:
   :show-inheritance:

Sequential Optimization
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: homodyne.optimization.nlsq.strategies.sequential
   :members:
   :undoc-members:
   :show-inheritance:

CMC: Consensus Monte Carlo
---------------------------

CMC provides Bayesian parameter estimation with full posterior sampling using NumPyro/NUTS.

**Key Features**:

- Physics-informed priors
- Automatic retry mechanism (max 3 attempts)
- Single-angle log-space D0 sampling for stability
- ArviZ-native output format

Core Module
~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.core
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.config
   :members:
   :undoc-members:
   :show-inheritance:

Model Definition
~~~~~~~~~~~~~~~~

NumPyro model definition for MCMC sampling.

.. automodule:: homodyne.optimization.cmc.model
   :members:
   :undoc-members:
   :show-inheritance:

Priors
~~~~~~

Physics-informed prior distributions.

.. automodule:: homodyne.optimization.cmc.priors
   :members:
   :undoc-members:
   :show-inheritance:

Prior Specifications
^^^^^^^^^^^^^^^^^^^^

**Static Mode** (3 physical parameters):

- D0: LogNormal(log(1000), 1.5)
- alpha: Uniform(0.0, 2.0)
- D_offset: TruncatedNormal(0, 100, low=0)

**Laminar Flow Mode** (+4 shear parameters):

- gamma_dot_t0: LogNormal(log(100), 1.5)
- beta: Uniform(-2.0, 2.0)
- gamma_dot_t_offset: TruncatedNormal(0, 100, low=0)
- phi0: Uniform(0, 2π)

**Per-Angle Scaling** (mandatory in v2.4.0+):

- contrast_i: TruncatedNormal(0.5, 0.3, low=0.1, high=2.0) for each angle i
- offset_i: TruncatedNormal(1.0, 0.2, low=0.5, high=1.5) for each angle i

Data Preparation
~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.data_prep
   :members:
   :undoc-members:
   :show-inheritance:

Sampler
~~~~~~~

NUTS sampler interface with warmup and sampling phases.

.. automodule:: homodyne.optimization.cmc.sampler
   :members:
   :undoc-members:
   :show-inheritance:

Results
~~~~~~~

.. automodule:: homodyne.optimization.cmc.results
   :members:
   :undoc-members:
   :show-inheritance:

Diagnostics
~~~~~~~~~~~

MCMC convergence diagnostics (R-hat, ESS, trace plots).

.. automodule:: homodyne.optimization.cmc.diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

I/O Operations
~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.cmc.io
   :members:
   :undoc-members:
   :show-inheritance:

Plotting
~~~~~~~~

.. automodule:: homodyne.optimization.cmc.plotting
   :members:
   :undoc-members:
   :show-inheritance:

Backends
~~~~~~~~

CMC supports multiple parallelization backends:

.. automodule:: homodyne.optimization.cmc.backends
   :members:
   :undoc-members:
   :show-inheritance:

Initialization
--------------

Per-angle parameter initialization strategies.

.. automodule:: homodyne.optimization.initialization
   :members:
   :undoc-members:
   :show-inheritance:

Supporting Modules
------------------

The optimization module includes several supporting utilities:

Checkpoint Manager
~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.checkpoint_manager
   :members:
   :undoc-members:
   :show-inheritance:

Gradient Diagnostics
~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.gradient_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
~~~~~~~~~~

.. automodule:: homodyne.optimization.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Recovery Strategies
~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.optimization.recovery_strategies
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

**NLSQ Optimization**::

    from homodyne.optimization import fit_nlsq_jax

    result = fit_nlsq_jax(
        t1=t1,
        t2=t2,
        c2=c2,
        q2_phi_t1_t2=q2_phi_t1_t2,
        phi_rad=phi_rad,
        initial_params=initial_params,
        mode="static"
    )

    print(f"Best-fit D0: {result.params['D0']:.2f}")

**CMC Optimization**::

    from homodyne.optimization import fit_mcmc_jax, CMCConfig

    config = CMCConfig(
        num_warmup=1000,
        num_samples=2000,
        num_chains=4
    )

    result = fit_mcmc_jax(
        t1=t1,
        t2=t2,
        c2=c2,
        q2_phi_t1_t2=q2_phi_t1_t2,
        phi_rad=phi_rad,
        initial_params=initial_params,
        mode="static",
        config=config
    )

    # Access posterior samples
    print(result.summary())

See Also
--------

- :mod:`homodyne.core` - Core physics and computation
- :mod:`homodyne.config` - Parameter management
- External: `NLSQ Package Documentation <https://nlsq.readthedocs.io/>`_
- External: `NumPyro Documentation <https://num.pyro.ai/>`_
