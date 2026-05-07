.. _api-cmc-reparameterization:

=============================================
homodyne.optimization.cmc.reparameterization
=============================================

The ``reparameterization`` module transforms correlated physics parameters
into an orthogonal sampling space for MCMC, then converts back to physics
parameters for output. This improves NUTS sampling efficiency by reducing
posterior correlations between parameters that span different scales
(e.g., ``D₀ ~ 10⁴`` and ``γ̇₀ ~ 10⁻³``).

.. note::

   Reparameterized values are computed **after** the time grid is constructed,
   since ``t_ref = sqrt(dt × t_max)`` depends on the data. ``core.py`` calls
   ``compute_t_ref()`` and falls back to ``t_ref=1.0`` if inputs are invalid.

----

.. _api-reparam-config:

ReparamConfig
-------------

Configuration dataclass controlling which parameters are reparameterized.

.. autoclass:: homodyne.optimization.cmc.reparameterization.ReparamConfig
   :members:
   :undoc-members:
   :show-inheritance:

----

Parameter Transformations
--------------------------

The following transforms are applied when reparameterization is enabled:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Physics Parameter
     - Sampling Parameter
     - Transform
   * - ``D₀`` (diffusion)
     - ``log_D_ref``
     - ``log(D₀ × t_ref^α)``
   * - ``D_offset``
     - ``D_offset_ratio``
     - ``D_offset / D_ref`` ∈ (−1+ε, ∞); supports negative D_offset
   * - ``γ̇₀`` (shear rate)
     - ``log_gamma_ref``
     - ``log(γ̇₀ × t_ref)``

The prior on ``D_offset_ratio`` is ``TruncatedNormal(low=-1+ε)``. The lower
bound enforces ``D_ref + D_offset > 0`` at ``t_ref`` (non-negative total
diffusion coefficient at the reference time). Negative values of
``D_offset_ratio`` in the range ``(-1, 0)`` correspond to jammed/arrested
systems where ``D_offset < 0`` physically reduces diffusion.

The back-transform is exact: ``D_offset = D_ref × D_offset_ratio``.

----

compute\_t\_ref
----------------

.. autofunction:: homodyne.optimization.cmc.reparameterization.compute_t_ref

----

transform\_nlsq\_to\_reparam\_space
-------------------------------------

.. autofunction:: homodyne.optimization.cmc.reparameterization.transform_nlsq_to_reparam_space

----

transform\_to\_sampling\_space
-------------------------------

.. autofunction:: homodyne.optimization.cmc.reparameterization.transform_to_sampling_space

----

transform\_to\_physics\_space
------------------------------

.. autofunction:: homodyne.optimization.cmc.reparameterization.transform_to_physics_space

----

Usage Examples
--------------

NLSQ warm-start to reparameterized priors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc.reparameterization import (
       compute_t_ref,
       transform_nlsq_to_reparam_space,
       ReparamConfig,
   )

   # From data: dt=0.001s, t_max=0.1s
   t_ref = compute_t_ref(dt=0.001, t_max=0.1)  # sqrt(0.001 * 0.1) ≈ 0.01

   nlsq_values = {"D0": 19231.0, "alpha": 1.5, "D_offset": 100.0,
                   "gamma_dot_t0": 0.003, "beta": 0.8}
   nlsq_uncert = {"D0": 500.0, "alpha": 0.1, "D_offset": 20.0,
                   "gamma_dot_t0": 0.001, "beta": 0.05}

   reparam_vals, reparam_unc = transform_nlsq_to_reparam_space(
       nlsq_values, nlsq_uncert, t_ref
   )
   # reparam_vals contains: log_D_ref, D_offset_ratio, log_gamma_ref, alpha, beta

Converting MCMC samples back to physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.cmc.reparameterization import (
       transform_to_physics_space,
       ReparamConfig,
   )
   import numpy as np

   config = ReparamConfig(enable_d_ref=True, enable_gamma_ref=True, t_ref=0.01)

   # Batched MCMC samples (e.g., from 4 chains × 1000 samples)
   samples = {
       "log_D_ref": np.random.normal(-2.0, 0.1, size=4000),
       "alpha": np.random.normal(1.5, 0.05, size=4000),
       "D_offset_ratio": np.random.normal(0.05, 0.1, size=4000),  # D_offset / D_ref
       "log_gamma_ref": np.random.normal(-5.5, 0.2, size=4000),
       "beta": np.random.normal(0.8, 0.03, size=4000),
   }

   physics = transform_to_physics_space(samples, config)
   # physics contains: D0, alpha, D_offset, gamma_dot_t0, beta

----

.. automodule:: homodyne.optimization.cmc.reparameterization
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ReparamConfig, compute_t_ref, transform_nlsq_to_reparam_space, transform_to_sampling_space, transform_to_physics_space
