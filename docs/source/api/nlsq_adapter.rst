.. _api-nlsq-adapter:

====================================
homodyne.optimization.nlsq — Adapter
====================================

``NLSQAdapter`` is the **recommended entry point** for NLSQ optimisation in
Homodyne. It wraps the NLSQ package's ``CurveFit`` class with:

- Built-in JIT compilation (2–3× speedup for single fits)
- Model instance caching via ``WeakValueDictionary`` (3–5× speedup for multi-start)
- Automatic dataset-size and memory-aware workflow selection
- Native NLSQ stability and recovery systems
- Automatic fallback to :class:`~homodyne.optimization.nlsq.wrapper.NLSQWrapper`
  on failure

.. note::

   For the simplest usage, call :func:`~homodyne.optimization.nlsq.core.fit_nlsq_jax`
   with ``use_adapter=True`` (the default). ``NLSQAdapter`` is used internally.
   Drop to ``NLSQAdapter`` directly only when you need fine-grained control over
   adapter configuration.

----

When to Use NLSQAdapter vs NLSQWrapper
---------------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - NLSQAdapter
     - NLSQWrapper
   * - Model caching
     - Built-in
     - None
   * - JIT compilation
     - Automatic
     - Manual
   * - Multi-start speedup
     - 3–5×
     - 1×
   * - Anti-degeneracy layers
     - Via ``fit()``
     - Full control
   * - Streaming (>100 M pts)
     - Via NLSQ
     - Full custom
   * - Recovery strategy
     - NLSQ native
     - 3-attempt retry
   * - Recommended for
     - Standard / multi-start
     - Complex / large datasets

----

.. _api-fit-nlsq:

fit\_nlsq\_jax
--------------

The primary entry point. Automatically selects ``NLSQAdapter`` (default) or
``NLSQWrapper`` based on configuration.

.. autofunction:: homodyne.optimization.nlsq.core.fit_nlsq_jax

----

NLSQAdapter
-----------

.. _api-nlsq-adapter-class:

.. autoclass:: homodyne.optimization.nlsq.adapter.NLSQAdapter
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

AdapterConfig
-------------

.. autoclass:: homodyne.optimization.nlsq.adapter.AdapterConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

Usage Examples
--------------

Basic static mode
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax

   # Minimal usage — NLSQAdapter is used by default
   result = fit_nlsq_jax(data, config, use_adapter=True)

   print(f"D0      = {result.params[0]:.4g}")
   print(f"alpha   = {result.params[1]:.4f}")
   print(f"D_offset= {result.params[2]:.4g}")
   print(f"chi^2   = {result.chi_squared:.4f}")

Laminar flow with per-angle scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.config.manager import ConfigManager

   config = ConfigManager("flow_config.yaml").config

   # Ensure anti-degeneracy layer is active
   config["optimization"]["nlsq"]["anti_degeneracy"]["per_angle_mode"] = "auto"

   result = fit_nlsq_jax(data, config, use_adapter=True)
   # 9 parameters: 7 physical + 2 averaged scaling (auto mode, n_phi >= 3)

Direct NLSQAdapter usage
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.adapter import NLSQAdapter, AdapterConfig
   from homodyne.core.fitting import ParameterSpace

   adapter_config = AdapterConfig(
       enable_jit=True,
       enable_model_cache=True,
       fallback_to_wrapper=True,
   )

   adapter = NLSQAdapter(config=adapter_config)
   param_space = ParameterSpace.from_config(config)

   result = adapter.fit(
       data=data,
       parameter_space=param_space,
       analysis_mode="laminar_flow",
   )

Multi-start optimisation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.optimization.nlsq.adapter import NLSQAdapter, AdapterConfig
   from homodyne.optimization.nlsq.multistart import MultiStartConfig

   adapter = NLSQAdapter(config=AdapterConfig(enable_model_cache=True))
   ms_config = MultiStartConfig(n_starts=8, strategy="latin_hypercube")

   best_result = adapter.fit_multistart(
       data=data,
       parameter_space=param_space,
       multistart_config=ms_config,
   )

CMA-ES for multi-scale problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Enable CMA-ES when :math:`D_0 \sim 10^4` and :math:`\dot\gamma_0 \sim 10^{-3}`
(scale ratio > :math:`10^6`):

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true
         preset: "cmaes-global"   # 200 generations
         refine_with_nlsq: true

.. code-block:: python

   result = fit_nlsq_jax(data, config)   # CMA-ES + NLSQ refinement

----

Return Type
-----------

.. autoclass:: homodyne.optimization.nlsq.results.OptimizationResult
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: homodyne.optimization.nlsq.adapter
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: NLSQAdapter, AdapterConfig
