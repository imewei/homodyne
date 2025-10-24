Developer Guide
===============

Welcome to the Homodyne Developer Guide. This section covers architecture, testing, contributing guidelines, and performance optimization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   testing
   contributing
   code-quality
   performance

Quick Start for Contributors
-----------------------------

.. code-block:: bash

   # Clone repository
   git clone https://github.com/imewei/homodyne.git
   cd homodyne

   # Development installation
   make dev

   # Install pre-commit hooks
   pre-commit install

   # Run tests
   make test

   # Check code quality
   make format
   make lint

Overview
--------

This guide is designed for developers who want to:

* Understand Homodyne's architecture and design patterns
* Contribute code, documentation, or bug fixes
* Run and write tests
* Optimize performance for specific use cases
* Deploy Homodyne in HPC environments

Sections
--------

**Architecture**
   JAX-first design philosophy, module structure, critical performance paths

**Testing**
   Test strategy, running tests, writing new tests

**Contributing**
   Development workflow, Git conventions, pull request guidelines

**Code Quality**
   Formatting (Black), linting (Ruff), type checking (Mypy)

**Performance**
   Profiling, GPU optimization, memory management

Development Resources
---------------------

* **GitHub**: https://github.com/imewei/homodyne
* **Issues**: https://github.com/imewei/homodyne/issues
* **Discussions**: https://github.com/imewei/homodyne/discussions

Code Standards
--------------

Homodyne follows strict code quality standards:

* **Black** for code formatting (120 char line length)
* **Ruff** for linting and style checks
* **Mypy** for static type checking
* **Pre-commit** hooks enforce standards automatically
* **NumPy/Google** docstring format
* **Type hints** required for all public functions

See :doc:`code-quality` for detailed standards.
