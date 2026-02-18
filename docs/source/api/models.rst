.. _api-models:

====================
homodyne.core.models
====================

Object-oriented interface to the physical models for homodyne scattering analysis.
Every model wraps the JAX-compiled functions in ``homodyne.core.jax_backend`` and
exposes a uniform interface through the :class:`PhysicsModelBase` abstract base
class.

For the underlying physics, including the full derivation of the :math:`g_1` and
:math:`g_2` correlation functions, see :doc:`../theory/index`.

----

Physical Model Summary
----------------------

The measured intensity correlation function is:

.. math::

   c_2(\varphi, t_1, t_2) = \text{offset} + \text{contrast} \times [c_1(\varphi, t_1, t_2)]^2

where the field correlation function factorises as:

.. math::

   c_1(\varphi, t_1, t_2) = c_1^\text{diff}(t_1, t_2) \times c_1^\text{shear}(\varphi, t_1, t_2)

Time-dependent transport coefficients:

.. math::

   D(t) = D_0 \cdot t^\alpha + D_\text{offset}

   \dot\gamma(t) = \dot\gamma_0 \cdot t^\beta + \dot\gamma_\text{offset}

+------------------+-------------------------------------------------------------------+-------+
| Mode             | Parameters                                                        | Count |
+==================+===================================================================+=======+
| ``static``       | :math:`D_0, \alpha, D_\text{offset}`                              | 3     |
+------------------+-------------------------------------------------------------------+-------+
| ``laminar_flow`` | :math:`D_0, \alpha, D_\text{offset}, \dot\gamma_0, \beta,`        | 7     |
|                  | :math:`\dot\gamma_\text{offset}, \varphi_0`                       |       |
+------------------+-------------------------------------------------------------------+-------+

----

.. _api-models-base:

PhysicsModelBase
----------------

Abstract base class defining the interface all physical models must implement.

.. autoclass:: homodyne.core.models.PhysicsModelBase
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

.. _api-diffusion-model:

DiffusionModel
--------------

Implements the diffusion-only contribution to :math:`g_1`. Used as a building
block by both ``CombinedModel`` and in ``static`` analysis mode.

.. autoclass:: homodyne.core.models.DiffusionModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from homodyne.core.models import DiffusionModel

   model = DiffusionModel()

   # Parameters: [D0, alpha, D_offset]
   params = jnp.array([100.0, 0.5, 0.01])
   t1 = jnp.linspace(0.001, 1.0, 50)
   t2 = jnp.linspace(0.001, 1.0, 50)

   g1 = model.compute_g1(params, t1, t2, phi=jnp.array([0.0]), q=0.01, L=1000.0)

----

.. _api-shear-model:

ShearModel
----------

Implements the shear-flow contribution to :math:`g_1`:

.. math::

   c_1^\text{shear}(\varphi, t_1, t_2) = \left[\operatorname{sinc}\!\left(\Phi(\varphi, t_1, t_2)\right)\right]^2

where :math:`\Phi = \frac{1}{2\pi} q L \cos(\varphi_0 - \varphi) \int_{|t_2-t_1|} \dot\gamma(t') \, dt'`.

.. autoclass:: homodyne.core.models.ShearModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

.. _api-combined-model:

CombinedModel
-------------

Full laminar-flow model combining diffusion and shear contributions. This is
the model used when ``analysis_mode="laminar_flow"``.

.. autoclass:: homodyne.core.models.CombinedModel
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   import jax.numpy as jnp
   from homodyne.core.models import CombinedModel

   model = CombinedModel()

   # Parameters: [D0, alpha, D_offset, gamma_dot0, beta, gamma_dot_offset, phi0]
   params = jnp.array([100.0, 0.5, 0.01, 1e-3, 0.0, 0.0, 0.0])
   phi = jnp.array([0.0, 45.0, 90.0, 135.0])   # degrees
   t1 = jnp.linspace(0.001, 1.0, 50)
   t2 = jnp.linspace(0.001, 1.0, 50)

   g1 = model.compute_g1(params, t1, t2, phi=phi, q=0.01, L=1000.0)
   g2 = model.compute_g2(params, t1, t2, phi=phi, q=0.01, L=1000.0,
                         contrast=0.5, offset=1.0)

----

Module-level factory
--------------------

.. autofunction:: homodyne.core.models.create_model

.. automodule:: homodyne.core.models
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PhysicsModelBase, DiffusionModel, ShearModel, CombinedModel, create_model
