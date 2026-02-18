.. _user-guide:

User Guide
==========

This guide is structured so you can read it end-to-end or jump directly to
the section relevant to your experience level and goal.

.. toctree::
   :maxdepth: 2
   :caption: Fundamentals

   01_fundamentals/index

.. toctree::
   :maxdepth: 2
   :caption: Data and Fitting

   02_data_and_fitting/index

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   03_advanced_topics/index

.. toctree::
   :maxdepth: 2
   :caption: Practical Guides

   04_practical_guides/index

.. toctree::
   :maxdepth: 2
   :caption: Appendices

   05_appendices/index

----

.. _learning-pathways:

Learning Pathways
-----------------

Choose the pathway that matches your background and goal.

.. rubric:: Path A — New to XPCS

For scientists with experimental data but no prior Homodyne experience.

1. :doc:`01_fundamentals/what_is_xpcs` — what C2 matrices are and why they matter
2. :doc:`installation </installation>` — get Homodyne installed
3. :doc:`quickstart </quickstart>` — run your first analysis in 5 minutes
4. :doc:`02_data_and_fitting/data_loading` — load your own HDF5 files
5. :doc:`02_data_and_fitting/nlsq_fitting` — fit static diffusion
6. :doc:`02_data_and_fitting/result_interpretation` — understand the output

.. rubric:: Path B — Laminar Flow / Shear Dynamics

For users analyzing systems with velocity gradients (He et al. PNAS 2025).

1. :doc:`01_fundamentals/what_is_xpcs` — XPCS background
2. :doc:`01_fundamentals/analysis_modes` — static vs. laminar flow
3. :doc:`03_advanced_topics/laminar_flow` — 7-parameter fit
4. :doc:`03_advanced_topics/per_angle_modes` — per-angle scaling
5. :doc:`03_advanced_topics/bayesian_inference` — uncertainty quantification
6. :doc:`05_appendices/troubleshooting` — diagnose convergence issues

.. rubric:: Path C — Bayesian Uncertainty Quantification

For users who want posterior distributions, not just point estimates.

1. :doc:`quickstart </quickstart>` — run NLSQ first (warm-start)
2. :doc:`02_data_and_fitting/nlsq_fitting` — understand NLSQ output
3. :doc:`03_advanced_topics/bayesian_inference` — Consensus Monte Carlo
4. :doc:`03_advanced_topics/diagnostics` — shard size, chain method, R-hat, ESS
5. :doc:`02_data_and_fitting/result_interpretation` — read posterior summaries

.. rubric:: Path D — Advanced Configuration and Performance

For power users running large datasets on multi-core servers.

1. :doc:`04_practical_guides/configuration` — full YAML reference
2. :doc:`04_practical_guides/performance_tuning` — XLA flags, NUMA, threading
3. :doc:`03_advanced_topics/streaming_mode` — streaming vs. batch
4. :doc:`03_advanced_topics/cmaes_optimization` — CMA-ES for multi-scale problems
5. :doc:`01_fundamentals/parameter_guide` — all parameters, bounds, units

----

Sections Overview
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Section
     - Contents
   * - :doc:`01_fundamentals/index`
     - XPCS primer, core equation, analysis modes, the homodyne model
   * - :doc:`02_data_and_fitting/index`
     - Data loading, static mode, laminar flow, NLSQ, configuration
   * - :doc:`03_advanced_topics/index`
     - CMC/Bayesian, CPU optimisation, CMA-ES, diagnostics
   * - :doc:`04_practical_guides/index`
     - Interpreting results, troubleshooting, FAQ
   * - :doc:`05_appendices/index`
     - Parameter reference, CLI reference, changelog
