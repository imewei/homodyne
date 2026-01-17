.. Homodyne documentation master file, created by sphinx-quickstart on Dec 2025.

Homodyne Documentation
======================

**JAX-first XPCS analysis implementing** `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_

**Core Equation:**

.. math::

   c_2(\phi, t_1, t_2) = 1 + \text{contrast} \times [c_1(\phi, t_1, t_2)]^2

**Version:** 2.17.0 | **Python:** 3.12+ | **JAX:** 0.8.0 (CPU-only)

Homodyne is a CPU-optimized X-ray Photon Correlation Spectroscopy (XPCS) analysis framework
built on JAX. It implements state-of-the-art algorithms for analyzing time-dependent dynamics
in soft matter systems, from static isotropic scattering to complex laminar flow geometries.

This documentation provides everything you need to:

- **Get started** with installation and basic usage
- **Analyze your data** using NLSQ optimization or Bayesian MCMC inference
- **Understand the physics** behind the analysis methods
- **Extend the code** for research or custom applications
- **Configure advanced options** for specialized experiments

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   user-guide/index
   api-reference/index
   research/index
   developer-guide/index
   configuration/index

---

**Quick Links:**

- :doc:`user-guide/quickstart` - Get running in 5 minutes
- :doc:`user-guide/installation` - Installation and setup
- :doc:`user-guide/cli` - Command-line interface reference
- :doc:`research/theoretical_framework` - Mathematical foundations
- :doc:`research/anti_degeneracy_defense` - Anti-degeneracy system for laminar flow (v2.9.0+)
- :ref:`nlsq-adapter` - NLSQAdapter with model caching (v2.11.0+)
- :ref:`nlsq-adapter-base` - NLSQAdapterBase shared functionality (v2.14.0+)
- :ref:`nlsq-validation` - Input/result validation utilities (v2.14.0+)
- :doc:`research/anti_degeneracy_defense` - Quantile-based per-angle scaling (v2.17.0)
- :doc:`configuration/options` - Complete configuration reference

---

**Documentation Organization:**

- **User Guide:** Installation, quickstart, CLI usage, configuration
- **API Reference:** Complete autodoc reference for all modules
- **Research:** Theoretical framework, computational methods, citations
- **Developer Guide:** Contributing guidelines, testing, development setup
- **Configuration:** Templates, options, and parameter reference

---

**Getting Help:**

- Check the :doc:`user-guide/cli` for command-line usage
- See :doc:`user-guide/examples` for real-world workflows
- Visit the `GitHub repository <https://github.com/imewei/homodyne>`_ for issues and discussions
- Review the `He et al. PNAS 2024 <https://doi.org/10.1073/pnas.2401162121>`_ paper for theoretical background

---

**Citation:**

If you use Homodyne in your research, please cite:

.. code-block:: bibtex

   @article{He2024,
     author = {He, X. and others},
     title = {Real-time probing of nanoscale dynamics with X-ray photon correlation spectroscopy},
     journal = {Proceedings of the National Academy of Sciences},
     year = {2024},
     volume = {121},
     doi = {10.1073/pnas.2401162121}
   }

---

.. container:: toctree-wrapper

   Built with `Sphinx <https://www.sphinx-doc.org/>`_ | Hosted on `ReadTheDocs <https://readthedocs.org/>`_
