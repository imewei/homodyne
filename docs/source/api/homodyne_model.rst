.. _api-homodyne-model:

==============================
homodyne.core.homodyne\_model
==============================

The ``homodyne_model`` module provides ``HomodyneModel`` — the unified model
facade used by both NLSQ and CMC optimization backends. It wraps configuration
parsing, physics factor pre-computation, and time grid construction into a
single stateful object that feeds into ``compute_g2_scaled_with_factors()``.

.. note::

   ``HomodyneModel`` is the recommended entry point for computing correlation
   matrices from a YAML configuration. It validates the config once during
   ``__init__``, pre-computes physics factors (``q²dt/2``, ``qLdt/2π``), and
   builds time grids — so repeated calls to ``compute_c2()`` skip all setup
   overhead.

----

.. _api-homodyne-model-class:

HomodyneModel
-------------

.. autoclass:: homodyne.core.homodyne_model.HomodyneModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Architecture
------------

``HomodyneModel`` sits between configuration and optimization:

.. code-block:: text

   YAML → ConfigManager → HomodyneModel → NLSQ / CMC
                             │
                             ├── physics_factors (pre-computed q²dt/2, qLdt/2π)
                             ├── t1_grid, t2_grid (2-D time meshgrids)
                             └── compute_c2() → compute_g2_scaled_with_factors()

The model delegates actual physics to the JIT-compiled functions in
:doc:`theory_engine` and ``jax_backend``, ensuring that the hot path
(Jacobian evaluation in NLSQ, leapfrog steps in NUTS) never re-parses
configuration or recomputes static factors.

----

Key Attributes
--------------

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``physics_factors``
     - ``PhysicsFactors``
     - Pre-computed factors ``(q²dt/2, qLdt/2π)``
   * - ``time_array``
     - ``jnp.ndarray``
     - 1-D time grid ``[0 … dt*(n-1)]`` in seconds
   * - ``t1_grid``, ``t2_grid``
     - ``jnp.ndarray``
     - 2-D correlation time grids (shape ``n_time × n_time``)
   * - ``dt``
     - ``float``
     - Time step [s] from config
   * - ``wavevector_q``
     - ``float``
     - Scattering wave-vector magnitude [Å⁻¹]
   * - ``stator_rotor_gap``
     - ``float``
     - Sample-detector gap [Å]
   * - ``analysis_mode``
     - ``str``
     - ``"laminar_flow"``, ``"static_isotropic"``, or ``"static_anisotropic"``

----

Usage Examples
--------------

Computing correlation matrices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core.homodyne_model import HomodyneModel
   import numpy as np

   config = {
       "analysis_mode": "laminar_flow",
       "wavevector_q": 0.01,
       "stator_rotor_gap": 2_000_000.0,
       "dt": 0.001,
       "n_time": 100,
   }

   model = HomodyneModel(config)

   # 7-parameter laminar_flow vector
   params = np.array([19231.0, 1.5, 0.1, 0.003, 0.8, 0.0, 0.0])
   phi_angles = np.array([0.0, np.pi / 4, np.pi / 2])

   c2 = model.compute_c2(params, phi_angles, contrast=0.5, offset=1.0)
   print(f"C2 shape: {c2.shape}")  # (3, 100, 100)

Generating simulated data
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   c2_data, output_path = model.plot_simulated_data(
       params, phi_angles, output_dir="./simulated"
   )
   # Saves compressed .npz and per-angle heatmap PNGs
