NLSQ Quick Start
~~~~~~~~~~~~~~~~

Run a non-linear least-squares fit in three lines:

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager
   from homodyne.optimization.nlsq import fit_nlsq_jax

   config = ConfigManager.from_yaml("my_config.yaml")
   data   = XPCSDataLoader(config).load()
   result = fit_nlsq_jax(data, config)

   # result.params   — dict of fitted physical parameters
   # result.chi2_reduced — goodness-of-fit (target: ~1.0)
   # result.converged    — True if optimizer converged

.. tip::

   Use :func:`~homodyne.optimization.nlsq.fit_nlsq_jax` with
   ``use_adapter=True`` (default) to enable JIT-caching via
   :class:`~homodyne.optimization.nlsq.adapter.NLSQAdapter`.
   This makes repeated calls on same-shaped data significantly faster.

.. _end-nlsq-quickstart-snippet:
