.. _advanced-topics:

Advanced Topics
===============

This section covers Bayesian inference, per-angle scaling modes, laminar flow
analysis, CMA-ES optimization, convergence diagnostics, and memory management
strategies for large datasets.

.. toctree::
   :maxdepth: 2

   bayesian_inference
   per_angle_modes
   laminar_flow
   cmaes_optimization
   streaming_mode
   diagnostics

----

**What you will learn:**

- How to use Consensus Monte Carlo (CMC) for full Bayesian posteriors.
- How the per-angle scaling modes prevent parameter absorption degeneracy.
- How laminar flow analysis captures shear dynamics from angular dependence.
- When and how to use CMA-ES for multi-scale parameter spaces.
- How to manage memory when C2 matrices exceed available RAM.
- How to read convergence diagnostics (R-hat, ESS, BFMI, divergences).

**Prerequisites:** :doc:`../02_data_and_fitting/nlsq_fitting` and a
basic familiarity with Bayesian inference.
