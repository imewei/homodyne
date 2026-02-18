.. _examples:

Examples
========

End-to-end worked examples using synthetic XPCS datasets. Each example
is self-contained and shows a complete analysis workflow.

All examples are available as Jupyter notebooks in ``docs/notebooks/``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Notebook
     - Description
   * - ``01_quickstart.ipynb``
     - Complete 10-minute workflow: synthetic data, config, fit, and visualization.
   * - ``02_static_analysis.ipynb``
     - Static mode deep dive: data validation, parameter sensitivity, multi-start
       optimization, and result comparison.
   * - ``03_laminar_flow.ipynb``
     - Laminar flow analysis: angular dependence detection, per-angle mode
       comparison, sinc² pattern visualization.
   * - ``04_bayesian_inference.ipynb``
     - Full Bayesian workflow: NLSQ warm-start, Consensus Monte Carlo,
       ArviZ diagnostics, and posterior comparison.

----

Running the Notebooks
---------------------

.. code-block:: bash

   make dev
   uv run jupyter lab docs/notebooks/

.. tip::

   The notebooks use synthetic data so they can be run without any
   experimental HDF5 files. To adapt them for your own data, replace
   the synthetic data generation cells with ``XPCSDataLoader`` calls.

See Also
--------

- :doc:`/user_guide/index` — structured learning guide
- :doc:`/user_guide/02_data_and_fitting/data_loading` — loading real HDF5 data
- :doc:`/api/index` — full API reference
