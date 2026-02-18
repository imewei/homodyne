.. _api-nlsq-wrapper:

====================================
homodyne.optimization.nlsq — Wrapper
====================================

``NLSQWrapper`` is the **stable fallback adapter** for advanced NLSQ
optimisation. It is preferred over :class:`~homodyne.optimization.nlsq.adapter.NLSQAdapter`
when:

- Datasets exceed 100 M points and require custom streaming/chunking strategies
- ``laminar_flow`` mode uses many phi angles (> 6) where full anti-degeneracy
  control is required
- Custom parameter transforms or advanced 3-attempt recovery mechanisms are needed
- Production stability is the primary concern

.. note::

   For most new code, use :func:`~homodyne.optimization.nlsq.core.fit_nlsq_jax`
   with ``use_adapter=True`` (default). ``NLSQWrapper`` is invoked automatically
   as a fallback if ``NLSQAdapter`` fails.

----

Key Differences from NLSQAdapter
---------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Capability
     - NLSQWrapper
     - NLSQAdapter
   * - Model caching
     - None (rebuild each fit)
     - WeakValueDictionary cache
   * - JIT compilation
     - Manual
     - Automatic
   * - Anti-degeneracy control
     - Full (all layers)
     - Via ``fit()`` delegation
   * - Streaming (>100 M pts)
     - Full custom chunking
     - Via NLSQ package
   * - Recovery strategy
     - 3-attempt retry
     - NLSQ native
   * - Per-angle stratification
     - Angle-stratified chunking (v2.2+)
     - Automatic

----

NLSQWrapper
-----------

.. autoclass:: homodyne.optimization.nlsq.wrapper.NLSQWrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

NLSQConfig
----------

.. _api-nlsq-config:

The ``NLSQConfig`` dataclass controls all NLSQ optimisation settings.
It is populated from the YAML configuration file via ``NLSQConfig.from_yaml()``.

.. autoclass:: homodyne.optimization.nlsq.config.NLSQConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

HybridRecoveryConfig
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: homodyne.optimization.nlsq.config.HybridRecoveryConfig
   :members:
   :undoc-members:
   :show-inheritance:

Safe type conversion utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.optimization.nlsq.config.safe_float
.. autofunction:: homodyne.optimization.nlsq.config.safe_int
.. autofunction:: homodyne.optimization.nlsq.config.safe_bool

----

Usage Examples
--------------

Direct usage of NLSQWrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.wrapper import NLSQWrapper
   from homodyne.core.fitting import ParameterSpace
   from homodyne.config.manager import ConfigManager

   config = ConfigManager("config.yaml").config
   param_space = ParameterSpace.from_config(config)

   wrapper = NLSQWrapper(config=config)
   result = wrapper.fit(data=data, parameter_space=param_space)

   print(result.params)
   print(result.residuals)

Large dataset (>100 M points) with streaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.75
       # Or set explicit memory threshold:
       # memory_threshold_gb: 48

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax

   # NLSQWrapper handles streaming automatically when memory threshold is exceeded
   result = fit_nlsq_jax(data, config, use_adapter=False)

laminar_flow with many phi angles and full anti-degeneracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.wrapper import NLSQWrapper

   # NLSQWrapper applies all anti-degeneracy layers when n_phi > 6
   wrapper = NLSQWrapper(config=config)
   result = wrapper.fit(
       data=data,
       parameter_space=param_space,
       analysis_mode="laminar_flow",
       per_angle_mode="individual",   # Full per-angle scaling
   )

3-attempt error recovery
~~~~~~~~~~~~~~~~~~~~~~~~~

``NLSQWrapper`` retries automatically on solver failure, applying progressively
conservative settings per retry:

- Attempt 1: Original settings
- Attempt 2: 0.5× learning rate, 2× regularisation, 0.5× trust region
- Attempt 3: 0.25× learning rate, 4× regularisation, 0.25× trust region

.. code-block:: python

   from homodyne.optimization.nlsq.config import HybridRecoveryConfig

   recovery = HybridRecoveryConfig(
       max_retries=3,
       lr_decay=0.5,
       lambda_growth=2.0,
       trust_decay=0.5,
   )
   print(recovery.get_retry_settings(attempt=2))
   # {"lr_multiplier": 0.25, "lambda_multiplier": 4.0, "trust_multiplier": 0.25}

----

.. automodule:: homodyne.optimization.nlsq.wrapper
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: NLSQWrapper
