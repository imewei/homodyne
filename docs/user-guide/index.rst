User Guide
==========

This guide covers everything you need to get started with Homodyne, from installation through running
your first XPCS analysis. Whether you're analyzing experimental data or exploring the physics of
time-dependent dynamics, this guide provides step-by-step instructions and practical examples.

**Table of Contents:**

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   cli
   configuration
   examples

---

**Getting Started in 5 Steps:**

1. :doc:`installation` - Install Homodyne with Python 3.12+ and JAX ≥0.8.2
2. :doc:`quickstart` - Run your first analysis with a minimal example
3. :doc:`cli` - Learn the command-line interface and available options
4. :doc:`configuration` - Understand how to configure your analysis
5. :doc:`examples` - Explore real-world analysis workflows

---

**Key Concepts:**

**XPCS Analysis:** X-ray Photon Correlation Spectroscopy measures time-dependent dynamics
by analyzing two-time correlation functions from time-series XPCS data.

**Core Analysis Equation:**

.. math::

   c_2(\phi, t_1, t_2) = 1 + \text{contrast} \times [c_1(\phi, t_1, t_2)]^2

**Two Analysis Modes:**

- **Static Mode** (3 physical parameters): Isotropic systems with time-dependent diffusion
- **Laminar Flow Mode** (7 physical parameters): Systems with velocity gradients and shear-dependent dynamics

**Two Inference Methods:**

- **NLSQ** (Non-Linear Least Squares): Fast trust-region optimization for point estimates
- **MCMC** (Markov Chain Monte Carlo): Bayesian inference for uncertainty quantification

---

**Common Workflows:**

**Quick Analysis:**

.. code-block:: bash

   homodyne --config my_config.yaml --method nlsq

**With Uncertainty Quantification:**

.. code-block:: bash

   # First run NLSQ for initial fit
   homodyne --config my_config.yaml --method nlsq

   # Then run CMC for Bayesian inference
   homodyne --config my_config.yaml --method cmc

**Interactive Configuration:**

.. code-block:: bash

   homodyne-config --interactive

---

**What You'll Learn:**

- How to install and configure Homodyne for your system
- How to prepare your XPCS data for analysis
- How to select the appropriate analysis mode (static vs. laminar flow)
- How to run NLSQ optimization and interpret results
- How to run MCMC inference for uncertainty quantification
- How to visualize and validate your analysis results

---

**Documentation Links:**

- :doc:`../research/theoretical_framework` - Mathematical foundation
- :doc:`../configuration/options` - Complete configuration reference
- :doc:`../api-reference/index` - API documentation for developers

---

**Need Help?**

- Check :doc:`./cli` for command-line usage
- See :doc:`./examples` for complete workflows
- Review :doc:`./configuration` for configuration details
- Visit the `GitHub repository <https://github.com/apc-llc/homodyne>`_ for issues and discussions
