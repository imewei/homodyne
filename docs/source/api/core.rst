.. _api-core:

================
homodyne.core
================

The ``homodyne.core`` package provides the physical foundation for all XPCS
computations: constants, parameter bounds, validation logic, and numerically
stable JAX-compiled helper functions shared across the NLSQ and CMC backends.

.. note::

   For physics background on the homodyne model and the g1/g2 correlation
   functions, see the :doc:`../theory/index` section.

----

homodyne.core.physics
---------------------

.. _api-physics-constants:

``PhysicsConstants`` gathers reference values for synchrotron experiments
(wavelengths, q-ranges, diffusion coefficient bounds). ``ValidationResult``
captures the outcome of parameter validation, including human-readable
violation messages.

.. _api-physics-validation:

``validate_parameters()`` checks that a set of physical parameters falls
within the bounds specified by ``ParameterBounds``. It returns a
``ValidationResult`` and optionally raises ``ValueError`` for critical
violations.

.. automodule:: homodyne.core.physics
   :members:
   :undoc-members:
   :show-inheritance:

----

homodyne.core.physics\_utils
-----------------------------

.. _api-physics-utils:

Shared utility functions used by both the NLSQ (meshgrid) and CMC
(element-wise) computational backends. These were consolidated from
``jax_backend.py``, ``physics_nlsq.py``, and ``physics_cmc.py`` to
eliminate code duplication and ensure consistent numerical behaviour.

Key functions:

- ``safe_exp`` — JIT-compiled overflow-protected exponential. Clips input to
  ``[-max_val, max_val]`` before evaluating, preventing ``NaN`` propagation.
- ``safe_sinc`` — Numerically stable unnormalized sinc function with Taylor
  expansion near zero. Uses ``EPS = 1e-12`` as the zero threshold.
- ``calculate_diffusion_coefficient`` / ``_calculate_diffusion_coefficient_impl_jax``
  — Time-dependent diffusion :math:`D(t) = D_0 \cdot t^\alpha + D_\text{offset}`.

.. warning::

   ``safe_exp`` and ``safe_sinc`` are decorated with ``@jit``. Calling them
   with Python scalars on first use will trigger XLA compilation. Subsequent
   calls with arrays of the same shape are fast.

.. automodule:: homodyne.core.physics_utils
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from homodyne.core.physics_utils import safe_exp, safe_sinc

   # Safe exponential — no overflow for large inputs
   x = jnp.array([0.0, 100.0, 1000.0])
   result = safe_exp(x)          # last element clamped, not NaN

   # Numerically stable sinc near zero
   phi = jnp.linspace(0, 0.1, 10)
   s = safe_sinc(phi)            # uses Taylor series for |phi| < EPS

----

homodyne.core.scaling\_utils
-----------------------------

.. _api-scaling-utils:

Quantile-based estimation of per-angle contrast and offset parameters.
These utilities implement the physics relation:

.. math::

   C_2 = \text{contrast} \times g_1^2 + \text{offset}

At large time lags :math:`g_1^2 \to 0`, so :math:`C_2 \to \text{offset}`
(the floor). At small time lags :math:`g_1^2 \approx 1`, so
:math:`C_2 \approx \text{contrast} + \text{offset}` (the ceiling).

Both ``NLSQAdapter`` (anti-degeneracy layer) and ``fit_mcmc_jax()``
(``per_angle_mode="auto"``) call ``estimate_per_angle_scaling()`` to
obtain initialisation values that prevent parameter absorption degeneracy.

.. automodule:: homodyne.core.scaling_utils
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from homodyne.core.scaling_utils import estimate_per_angle_scaling

   # c2_data: shape (N,), delta_t: shape (N,)
   c2_data = np.random.uniform(1.0, 1.5, size=5000)
   delta_t = np.abs(np.random.randn(5000))

   contrast, offset = estimate_per_angle_scaling(c2_data, delta_t)
   print(f"contrast={contrast:.3f}, offset={offset:.3f}")
