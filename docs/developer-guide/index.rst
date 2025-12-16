Developer Guide
===============

This section provides guidance for developers contributing to Homodyne, including
development setup, code quality standards, testing practices, and contribution workflows.

.. toctree::
   :maxdepth: 2
   :caption: Developer Resources

   contributing
   testing

Quick Links
-----------

- **Code Quality**: Black (formatting), Ruff (linting), MyPy (type checking)
- **Testing**: Pytest with unit, integration, performance, and MCMC test suites
- **Package Manager**: uv for Python 3.12+ dependency management
- **Issue Tracking**: GitHub Issues for bugs, features, and discussions

For Setup and Workflows
^^^^^^^^^^^^^^^^^^^^^^^

Refer to the **Contributing** guide for:

- Development environment setup with uv
- Code style standards and formatting
- Pre-commit hooks and validation
- Pull request guidelines

For Testing and Validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Refer to the **Testing** guide for:

- Test organization and structure (unit, integration, performance, MCMC)
- Running test suites
- Code coverage requirements
- JAX debugging and performance optimization tips

Configuration and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For configuration file documentation and deployment guidance, see the
:doc:`/configuration/index` section.

Version Information
^^^^^^^^^^^^^^^^^^^

- **Homodyne**: v2.4.3
- **Python**: 3.12+
- **JAX**: 0.8.0 (CPU-only)
- **Package Manager**: uv
