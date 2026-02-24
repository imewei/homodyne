.. _api-jax-backend:

===========================
homodyne.core.jax\_backend
===========================

The ``jax_backend`` module provides the JIT-compiled computational core for
homodyne scattering analysis. All physics functions in this module are
compiled with ``jax.jit`` and support automatic differentiation for NLSQ
Jacobian evaluation and NUTS leapfrog integration.

The physical model computed here is:

.. math::

   g_2(\phi, t_1, t_2) = \text{offset} + \text{contrast} \times [g_1(\phi, t_1, t_2)]^2

.. warning::

   This module uses ``jnp.where(x > eps, x, eps)`` instead of
   ``jnp.maximum(x, eps)`` for positivity floors. ``jnp.maximum`` zeros
   the gradient below the floor, stalling NLSQ Jacobian and NUTS leapfrog.
   See :doc:`/theory/anti_degeneracy_defense` for details.

----

Backend Availability
--------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description
   * - ``jax_available``
     - ``True`` if JAX imported successfully; all JIT functions are active
   * - ``numpy_gradients_available``
     - ``True`` only when JAX is absent AND NumPy gradient fallback is importable

----

.. _api-g1-functions:

g₁ Correlation Functions
------------------------

compute\_g1\_diffusion
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.compute_g1_diffusion

compute\_g1\_shear
~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.compute_g1_shear

compute\_g1\_total
~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.compute_g1_total

----

.. _api-g2-functions:

g₂ Correlation Functions
------------------------

compute\_g2\_scaled
~~~~~~~~~~~~~~~~~~~~

Primary scalar entry point for NLSQ optimization.

.. autofunction:: homodyne.core.jax_backend.compute_g2_scaled

compute\_g2\_scaled\_with\_factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

JIT-compiled variant accepting pre-computed physics factors. Used by
:class:`~homodyne.core.homodyne_model.HomodyneModel` to avoid redundant
factor computation in tight loops.

.. autofunction:: homodyne.core.jax_backend.compute_g2_scaled_with_factors

----

Chi-Squared
-----------

compute\_chi\_squared
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.compute_chi_squared

----

Batched Computations
--------------------

vectorized\_g2\_computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.vectorized_g2_computation

batch\_chi\_squared
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.batch_chi_squared

----

Automatic Differentiation
--------------------------

These are pre-built JIT-compiled derivative functions:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Function
     - Derivative
     - Description
   * - ``gradient_g2``
     - ``grad(compute_g2_scaled)``
     - Gradient of g₂ w.r.t. params (argnums=0)
   * - ``hessian_g2``
     - ``hessian(compute_g2_scaled)``
     - Hessian of g₂ w.r.t. params
   * - ``gradient_chi2``
     - ``grad(compute_chi_squared)``
     - Gradient of χ² w.r.t. params
   * - ``hessian_chi2``
     - ``hessian(compute_chi_squared)``
     - Hessian of χ² w.r.t. params

----

Meshgrid Cache
--------------

The module maintains an LRU cache for time-grid meshgrids to avoid
recomputing ``(t1_grid, t2_grid)`` on every function call.

.. autofunction:: homodyne.core.jax_backend.get_cached_meshgrid

.. autofunction:: homodyne.core.jax_backend.clear_meshgrid_cache

.. autofunction:: homodyne.core.jax_backend.get_cache_stats

.. autofunction:: homodyne.core.jax_backend.reset_cache_stats

.. autofunction:: homodyne.core.jax_backend.log_cache_stats

----

Diagnostics
-----------

validate\_backend
~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.validate_backend

get\_device\_info
~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.get_device_info

get\_performance\_summary
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.core.jax_backend.get_performance_summary

----

Usage Examples
--------------

Computing g₂ for a single angle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core.jax_backend import compute_g2_scaled
   import jax.numpy as jnp

   params = jnp.array([19231.0, 1.5, 0.1, 0.003, 0.8, 0.0, 0.0])
   t = jnp.linspace(0, 0.1, 100)
   t1, t2 = jnp.meshgrid(t, t, indexing="ij")

   g2 = compute_g2_scaled(
       params, t1, t2, phi=jnp.array([0.0]),
       q=0.01, L=2_000_000.0,
       contrast=0.5, offset=1.0, dt=0.001,
   )

Computing gradients for optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core.jax_backend import gradient_g2

   grad_params = gradient_g2(
       params, t1, t2, phi=jnp.array([0.0]),
       q=0.01, L=2_000_000.0,
       contrast=0.5, offset=1.0, dt=0.001,
   )
   print(f"Gradient shape: {grad_params.shape}")  # (7,)

Cache management
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core.jax_backend import get_cache_stats, clear_meshgrid_cache

   stats = get_cache_stats()
   print(f"Hit rate: {stats['hit_rate']:.1%}")

   # Clear when switching datasets
   clear_meshgrid_cache()
