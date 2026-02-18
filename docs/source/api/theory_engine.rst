.. _api-theory-engine:

========================
homodyne.core.theory
========================

The ``theory`` module provides ``TheoryEngine``, a high-level interface to all
theoretical computations for homodyne scattering analysis. It wraps the
low-level JAX-compiled functions in ``homodyne.core.jax_backend`` with input
validation, performance monitoring, and graceful error handling.

.. note::

   ``TheoryEngine`` is primarily useful for interactive exploration and
   visualisation. For batch fitting, use :func:`~homodyne.optimization.nlsq.core.fit_nlsq_jax`
   or :func:`~homodyne.optimization.cmc.core.fit_mcmc_jax` directly.
   See :doc:`../theory/index` for the physics derivation.

----

TheoryEngine
------------

.. autoclass:: homodyne.core.theory.TheoryEngine
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

compute\_g1
~~~~~~~~~~~

Computes the normalised field autocorrelation function:

.. math::

   g_1(\varphi, t_1, t_2) = g_1^\text{diff}(t_1, t_2) \times g_1^\text{shear}(\varphi, t_1, t_2)

Parameters ``params`` must be ordered according to the ``analysis_mode``:

- ``"static"``: ``[D0, alpha, D_offset]``
- ``"laminar_flow"``: ``[D0, alpha, D_offset, gamma_dot0, beta, gamma_dot_offset, phi0]``

----

compute\_g2
~~~~~~~~~~~

Computes the normalised intensity correlation function using the Siegert relation:

.. math::

   g_2(\varphi, t_1, t_2) = \text{offset} + \text{contrast} \times [g_1(\varphi, t_1, t_2)]^2

----

Usage Examples
--------------

Static mode
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from homodyne.core.theory import TheoryEngine

   engine = TheoryEngine(analysis_mode="static")

   # Parameters: [D0, alpha, D_offset]
   params = np.array([100.0, 0.5, 0.01])

   t1 = np.linspace(0.001, 1.0, 100)
   t2 = np.linspace(0.001, 1.0, 100)
   phi = np.array([0.0])    # single angle for static mode

   g1 = engine.compute_g1(params, t1, t2, phi=phi, q=0.01, L=1000.0)
   g2 = engine.compute_g2(params, t1, t2, phi=phi, q=0.01, L=1000.0,
                          contrast=0.5, offset=1.0)

Laminar-flow mode (multi-angle)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from homodyne.core.theory import TheoryEngine

   engine = TheoryEngine(analysis_mode="laminar_flow")

   # Parameters: [D0, alpha, D_offset, gamma_dot0, beta, gamma_dot_offset, phi0]
   params = np.array([100.0, 0.5, 0.01, 1e-3, 0.0, 0.0, 0.0])

   t1 = np.linspace(0.001, 2.0, 200)
   t2 = np.linspace(0.001, 2.0, 200)
   phi = np.array([0.0, 45.0, 90.0, 135.0])   # four azimuthal angles (degrees)

   g2 = engine.compute_g2(params, t1, t2, phi=phi, q=0.01, L=1000.0,
                          contrast=0.5, offset=1.0, dt=0.001)

Checking backend availability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.core.jax_backend import jax_available

   if jax_available:
       print("JAX backend active — JIT compilation enabled")
   else:
       print("Falling back to NumPy — slower but correct")

----

.. automodule:: homodyne.core.theory
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: TheoryEngine
