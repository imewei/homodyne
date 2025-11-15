homodyne.core - Core Physics Engine
====================================

.. automodule:: homodyne.core
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.core`` module provides the JAX-first computational backend for high-performance X-ray Photon Correlation Spectroscopy (XPCS) analysis. It implements the theoretical framework for computing first-order (g₁) and second-order (g₂) correlation functions with automatic differentiation and GPU acceleration.

**Core Equation:** ``c₂(φ,t₁,t₂) = 1 + contrast × [c₁(φ,t₁,t₂)]²``

Key Features
------------

* **JIT Compilation**: All computational primitives are JIT-compiled for optimal performance
* **Automatic Differentiation**: Built-in gradient and Hessian computation via JAX
* **Hardware Acceleration**: Transparent CPU/GPU/TPU execution
* **Vectorized Operations**: Parallel computation over phi angles and time points
* **Memory Efficiency**: Optimized handling of large correlation matrices

Module Structure
----------------

The core module is organized into several submodules:

* :mod:`homodyne.core.jax_backend` - Low-level JAX computational primitives
* :mod:`homodyne.core.models` - Physical models (diffusion, shear, combined)
* :mod:`homodyne.core.theory` - Theoretical g₁/g₂ calculations
* :mod:`homodyne.core.fitting` - Scaled fitting engine
* :mod:`homodyne.core.physics` - Physical constants and validation
* :mod:`homodyne.core.physics_factors` - Physics factor calculations
* :mod:`homodyne.core.homodyne_model` - High-level model interface

Submodules
----------

homodyne.core.jax_backend
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.jax_backend
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Low-level JAX computational primitives for computing correlation functions with JIT compilation.

**Key Functions:**

* ``compute_g1_diffusion()`` - First-order correlation from diffusion
* ``compute_g1_shear()`` - First-order correlation from shear flow
* ``compute_g2_scaled()`` - Second-order correlation with scaling
* ``gradient_g2()`` - Gradient computation via automatic differentiation
* ``hessian_g2()`` - Hessian computation for optimization

**Usage Example:**

.. code-block:: python

   import jax.numpy as jnp
   from homodyne.core.jax_backend import compute_g2_scaled

   # Compute g2 correlation for static isotropic case
   t1 = jnp.linspace(0, 1, 50)
   t2 = jnp.linspace(0, 1, 50)
   T1, T2 = jnp.meshgrid(t1, t2, indexing='ij')

   params = {
       'D0': 1000.0,
       'alpha': 0.8,
       'D_offset': 10.0,
       'contrast': 0.5,
       'offset': 1.0
   }

   g2 = compute_g2_scaled(
       T1, T2,
       phi=jnp.array([0.0]),
       q=0.01,
       parameters=params,
       analysis_type='static_isotropic'
   )

homodyne.core.models
~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.models
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Object-oriented wrappers for physical models with inheritance hierarchy.

**Class Hierarchy:**

* ``PhysicsModelBase`` - Abstract base class

  * ``DiffusionModel`` - Pure diffusion (static isotropic)
  * ``ShearModel`` - Pure shear flow
  * ``CombinedModel`` - Diffusion + shear (laminar flow)

**Usage Example:**

.. code-block:: python

   from homodyne.core.models import CombinedModel
   import jax.numpy as jnp

   # Create laminar flow model
   model = CombinedModel()

   # Set parameters
   params = {
       'D0': 1000.0, 'alpha': 0.8, 'D_offset': 10.0,
       'gamma_dot_t0': 0.1, 'beta': 1.2,
       'gamma_dot_offset': 0.01, 'phi0': 45.0
   }

   # Compute g1 correlation
   phi = jnp.array([0.0, 45.0, 90.0])
   t1 = jnp.linspace(0, 1, 50)
   t2 = jnp.linspace(0, 1, 50)

   g1 = model.compute_g1(t1, t2, phi, q=0.01, **params)

homodyne.core.theory
~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.theory
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

High-level theoretical computations for g₁ and g₂ correlation functions.

**Key Classes:**

* ``TheoryEngine`` - Main engine for theoretical calculations

**Key Functions:**

* ``compute_g2_theory()`` - Compute theoretical g₂ from parameters
* ``compute_chi2_theory()`` - Compute χ² between experimental and theoretical data

**Usage Example:**

.. code-block:: python

   from homodyne.core.theory import compute_g2_theory
   import jax.numpy as jnp

   # Compute theoretical g2
   t1 = jnp.linspace(0, 1, 50)
   t2 = jnp.linspace(0, 1, 50)
   phi = jnp.array([0.0, 45.0, 90.0])

   params = [1000.0, 0.8, 10.0, 0.1, 1.2, 0.01, 45.0]  # laminar_flow

   g2_theory = compute_g2_theory(
       t1, t2, phi, q=0.01,
       params=params,
       analysis_type='laminar_flow'
   )

homodyne.core.fitting
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.fitting
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Scaled fitting engine implementing ``Fitted = contrast × Theory + offset``.

**Key Classes:**

* ``ScaledFittingEngine`` - Main fitting engine
* ``FitResult`` - Fit result container

**Usage Example:**

.. code-block:: python

   from homodyne.core.fitting import ScaledFittingEngine
   import jax.numpy as jnp

   # Create fitting engine
   engine = ScaledFittingEngine(analysis_type='static_isotropic')

   # Prepare data
   t1 = jnp.linspace(0, 1, 50)
   t2 = jnp.linspace(0, 1, 50)
   phi = jnp.array([0.0])
   experimental_data = jnp.ones((50, 50))  # Placeholder

   # Fit parameters
   result = engine.fit(
       t1, t2, phi, experimental_data,
       q=0.01,
       initial_params={'D0': 1000.0, 'alpha': 0.8, 'D_offset': 10.0}
   )

   print(f"Optimal parameters: {result.parameters}")
   print(f"Chi-squared: {result.chi_squared}")

homodyne.core.physics
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.physics
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Physical constants, parameter bounds, and validation utilities.

**Key Components:**

* ``PhysicsConstants`` - Physical constants and units
* ``parameter_bounds()`` - Get parameter bounds for analysis type
* ``validate_parameters()`` - Validate parameter values

**Usage Example:**

.. code-block:: python

   from homodyne.core.physics import parameter_bounds, validate_parameters

   # Get bounds for static isotropic analysis
   bounds = parameter_bounds('static_isotropic')
   print(f"D0 bounds: {bounds['D0']}")

   # Validate parameters
   params = {'D0': 1000.0, 'alpha': 0.8, 'D_offset': 10.0}
   is_valid, message = validate_parameters(params, 'static_isotropic')

homodyne.core.physics_factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.physics_factors
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__
   :no-index:

Physics factor calculations for advanced analysis.

**Key Classes:**

* ``PhysicsFactors`` - Container for physics factors

**Key Functions:**

* ``create_physics_factors_from_config_dict()`` - Create from configuration

homodyne.core.homodyne_model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.core.homodyne_model
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

High-level model interface for homodyne scattering analysis.

**Key Classes:**

* ``HomodyneModel`` - Main model class

Analysis Types
--------------

The core module supports two primary analysis types:

**Static Isotropic** (5 parameters)
   * Physical: D₀ (diffusion), α (exponent), D_offset
   * Scaling: contrast, offset

**Laminar Flow** (9 parameters)
   * Physical: D₀, α, D_offset, γ̇₀ (shear rate), β, γ̇_offset, φ₀
   * Scaling: contrast, offset

Performance Considerations
--------------------------

**JIT Compilation**
   First call compiles functions, subsequent calls are fast. Use ``jax.jit()`` for custom functions.

**Memory Management**
   Large correlation matrices (N×N×M for N times, M angles) require careful memory management. Use chunking for datasets >100M points.

**GPU Acceleration**
   Automatic when CUDA available. Check device with:

   .. code-block:: python

      import jax
      print(jax.devices())

**Numerical Stability**
   Use ``jnp.float64`` for high-precision calculations if needed.

See Also
--------

* :doc:`../theoretical-framework/index` - Theoretical framework
* :doc:`../advanced-topics/nlsq-optimization` - NLSQ optimization methods
* :doc:`optimization` - Optimization module that uses core
* :doc:`../developer-guide/architecture` - JAX-first architecture patterns

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.core import (
       compute_g1_diffusion,
       compute_g2_scaled,
       DiffusionModel,
       CombinedModel,
       ScaledFittingEngine,
       parameter_bounds,
       PhysicsConstants,
   )

**Related Functions:**

* :func:`homodyne.optimization.nlsq_wrapper.fit_nlsq_jax` - Uses core for NLSQ optimization
* :func:`homodyne.optimization.mcmc.fit_mcmc_jax` - Uses core for MCMC sampling
* :func:`homodyne.data.preprocessing.prepare_data` - Prepares data for core computations
