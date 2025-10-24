Contributing Guide
==================

Thank you for contributing to Homodyne! This guide covers the development workflow, code standards, and contribution process.

Getting Started
---------------

Prerequisites
~~~~~~~~~~~~~

* Python 3.12+
* Git
* (Optional) CUDA 12.1-12.9 for GPU support (Linux only)

Development Setup
~~~~~~~~~~~~~~~~~

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/imewei/homodyne.git
      cd homodyne

2. **Install development dependencies**:

   .. code-block:: bash

      make dev               # CPU-only (all platforms)
      make install-jax-gpu   # GPU support (Linux only)

3. **Install pre-commit hooks**:

   .. code-block:: bash

      pre-commit install

4. **Verify installation**:

   .. code-block:: bash

      make test
      homodyne --version

Development Workflow
--------------------

Branch Strategy
~~~~~~~~~~~~~~~

**Main branches**:

* ``main``: Production-ready code
* ``dev`` (if used): Integration branch for features

**Feature branches**:

* Create from ``main``
* Name: ``feature/<descriptive-name>``
* Example: ``feature/add-streaming-optimizer``

**Bugfix branches**:

* Name: ``bugfix/<issue-number>-<description>``
* Example: ``bugfix/123-fix-parameter-validation``

**Hotfix branches**:

* Name: ``hotfix/<version>-<description>``
* For critical production fixes

Creating a Feature Branch
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Update main
   git checkout main
   git pull origin main

   # Create feature branch
   git checkout -b feature/my-new-feature

   # Make changes...

   # Run tests
   make test

   # Check code quality
   make format
   make lint

   # Commit changes
   git add .
   git commit -m "feat: add streaming optimizer support"

   # Push to remote
   git push origin feature/my-new-feature

Commit Message Convention
~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow **Conventional Commits** format:

.. code-block:: text

   <type>(<scope>): <subject>

   <body>

   <footer>

**Types**:

* ``feat``: New feature
* ``fix``: Bug fix
* ``docs``: Documentation only
* ``style``: Formatting (no code change)
* ``refactor``: Code restructuring
* ``perf``: Performance improvement
* ``test``: Add/update tests
* ``chore``: Maintenance tasks

**Examples**:

.. code-block:: text

   feat(optimization): add streaming optimizer for large datasets

   - Implement StreamingOptimizer with checkpoint support
   - Add batch-level statistics tracking
   - Update documentation

   Closes #45

.. code-block:: text

   fix(cli): correct angle normalization in phi filtering

   Angles now correctly normalized to [-180, 180] range.
   Fixes wrap-around at ±180 boundary.

   Fixes #123

Pull Request Process
--------------------

Creating a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~

1. **Push your branch**:

   .. code-block:: bash

      git push origin feature/my-feature

2. **Create PR on GitHub**:

   * Go to https://github.com/imewei/homodyne/pulls
   * Click "New Pull Request"
   * Select your branch
   * Fill in the PR template

3. **PR title** (conventional format):

   .. code-block:: text

      feat: add streaming optimizer support

4. **PR description template**:

   .. code-block:: markdown

      ## Summary

      Brief description of changes.

      ## Changes

      * Added StreamingOptimizer class
      * Implemented checkpoint management
      * Updated documentation

      ## Testing

      * [ ] Unit tests pass
      * [ ] Integration tests pass
      * [ ] Documentation builds
      * [ ] Manual testing completed

      ## Related Issues

      Closes #45

PR Checklist
~~~~~~~~~~~~

Before submitting, ensure:

* [ ] Code follows style guidelines (Black, Ruff)
* [ ] Type hints added for new functions
* [ ] Docstrings added (NumPy/Google format)
* [ ] Tests added for new features
* [ ] All tests pass (``make test-all``)
* [ ] Documentation updated
* [ ] Commit messages follow convention
* [ ] No merge conflicts with ``main``

Code Review Process
~~~~~~~~~~~~~~~~~~~

**Review criteria**:

1. **Correctness**: Does it solve the problem?
2. **Tests**: Adequate test coverage?
3. **Documentation**: Clear docstrings and comments?
4. **Style**: Follows code standards?
5. **Performance**: No obvious inefficiencies?

**Addressing feedback**:

.. code-block:: bash

   # Make requested changes
   git add .
   git commit -m "fix: address PR feedback"
   git push origin feature/my-feature

   # OR squash commits before merging
   git rebase -i main

Code Standards
--------------

Formatting
~~~~~~~~~~

**Black** (line length: 88 characters):

.. code-block:: bash

   make format
   # OR
   black homodyne/

**Ruff** (linting + formatting):

.. code-block:: bash

   make lint
   # OR
   ruff check homodyne/ --fix
   ruff format homodyne/

Linting
~~~~~~~

**Ruff configuration**: ``pyproject.toml``

.. code-block:: bash

   ruff check homodyne/

Type Checking
~~~~~~~~~~~~~

**Mypy** (static type checking):

.. code-block:: bash

   mypy homodyne/

**Type hints required** for all public functions:

.. code-block:: python

   def process_data(
       data: np.ndarray,
       config: HomodyneConfig,
   ) -> Dict[str, Any]:
       """
       Process experimental data.

       Parameters
       ----------
       data : np.ndarray
           Input data array
       config : HomodyneConfig
           Configuration dictionary

       Returns
       -------
       Dict[str, Any]
           Processed results
       """
       pass

Docstring Format
~~~~~~~~~~~~~~~~

Use **NumPy/Google docstring** format:

.. code-block:: python

   def compute_g2_scaled(
       params: np.ndarray,
       phi_angles: np.ndarray,
       t1_grid: np.ndarray,
       t2_grid: np.ndarray,
   ) -> np.ndarray:
       """
       Compute scaled two-time correlation function G2.

       Parameters
       ----------
       params : np.ndarray, shape (n_params,)
           Parameter array [contrast, offset, D0, alpha, D_offset, ...]
       phi_angles : np.ndarray, shape (n_angles,)
           Scattering angles in degrees
       t1_grid : np.ndarray, shape (n_t1, n_t2)
           First time grid (seconds)
       t2_grid : np.ndarray, shape (n_t1, n_t2)
           Second time grid (seconds)

       Returns
       -------
       np.ndarray, shape (n_angles, n_t1, n_t2)
           Scaled G2 values

       Examples
       --------
       >>> params = np.array([0.5, 1.0, 100.0, 0.5, 10.0])
       >>> phi_angles = np.array([0.0, 45.0, 90.0])
       >>> t1_grid = np.linspace(0, 1, 10).reshape(10, 1)
       >>> t2_grid = np.linspace(0, 1, 10).reshape(1, 10)
       >>> g2 = compute_g2_scaled(params, phi_angles, t1_grid, t2_grid)
       >>> g2.shape
       (3, 10, 10)
       """
       pass

Pre-commit Hooks
~~~~~~~~~~~~~~~~

**Installed hooks** (``.pre-commit-config.yaml``):

* **Black**: Code formatting
* **Ruff**: Linting + formatting
* **isort**: Import sorting
* **Mypy**: Type checking
* **Bandit**: Security scanning
* **Flake8**: Style guide enforcement

**Run manually**:

.. code-block:: bash

   pre-commit run --all-files

Documentation Standards
-----------------------

Updating Documentation
~~~~~~~~~~~~~~~~~~~~~~

**Sphinx docs** (``docs/``):

.. code-block:: bash

   # Build documentation
   make docs

   # View locally
   cd docs/_build/html
   python -m http.server

**Documentation structure**:

* ``docs/user-guide/``: User-facing guides
* ``docs/developer-guide/``: Developer documentation
* ``docs/api-reference/``: API reference (auto-generated)
* ``docs/theoretical-framework/``: Physics background

Adding New Modules
~~~~~~~~~~~~~~~~~~

When adding new modules, update:

1. **Docstrings**: All public functions/classes
2. **API Reference**: Add to ``docs/api-reference/``
3. **User Guide**: If user-facing feature
4. **CHANGELOG.md**: Document changes
5. **Tests**: Comprehensive test coverage

Common Development Tasks
------------------------

Adding a New Optimization Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create file in ``homodyne/optimization/``
2. Implement interface matching ``nlsq_wrapper.py``
3. Use residual functions from ``core/jax_backend.py``
4. Add unit tests in ``tests/unit/test_optimization_*.py``
5. Add integration tests in ``tests/integration/``
6. Document in API reference

Modifying Physics Models
~~~~~~~~~~~~~~~~~~~~~~~~~

1. Update JAX functions in ``core/jax_backend.py``
2. Ensure JIT compatibility (no Python control flow)
3. Update wrappers in ``core/models.py``
4. Test gradients/Hessians
5. Run self-consistency tests
6. Update parameter bounds

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors after installation**:

.. code-block:: bash

   # Reinstall in development mode
   pip uninstall homodyne
   make dev

**Pre-commit hooks failing**:

.. code-block:: bash

   # Update hooks
   pre-commit autoupdate

   # Skip hooks temporarily (not recommended)
   git commit --no-verify

**Tests failing locally but passing in CI**:

.. code-block:: bash

   # Clear pytest cache
   pytest --cache-clear

   # Check Python version matches CI
   python --version  # Should be 3.12+

Getting Help
~~~~~~~~~~~~

* **Issues**: https://github.com/imewei/homodyne/issues
* **Discussions**: https://github.com/imewei/homodyne/discussions
* **Email**: maintainer@homodyne.org

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

* Be respectful and inclusive
* Focus on constructive feedback
* Welcome newcomers
* Assume good intentions

Best Practices
~~~~~~~~~~~~~~

* **Ask before starting**: Check if feature is wanted (open issue first)
* **Small PRs**: Easier to review (<500 lines preferred)
* **Tests required**: No PR without tests
* **Document changes**: Update docs in same PR
* **Responsive to feedback**: Address review comments promptly

Release Process
---------------

(For maintainers only)

Versioning
~~~~~~~~~~

Follow **Semantic Versioning** (MAJOR.MINOR.PATCH):

* **MAJOR**: Breaking changes
* **MINOR**: New features (backward compatible)
* **PATCH**: Bug fixes

Creating a Release
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Update version in homodyne/_version.py
   VERSION = "2.1.0"

   # Update CHANGELOG.md

   # Commit version bump
   git commit -am "chore: bump version to 2.1.0"

   # Tag release
   git tag -a v2.1.0 -m "Release v2.1.0"

   # Push
   git push origin main --tags

   # GitHub Actions automatically builds and publishes to PyPI

Resources
---------

* **Conventional Commits**: https://www.conventionalcommits.org/
* **Semantic Versioning**: https://semver.org/
* **NumPy Docstring Guide**: https://numpydoc.readthedocs.io/
* **Pre-commit**: https://pre-commit.com/

Next Steps
----------

* **Architecture Guide**: Understand the codebase structure
* **Testing Guide**: Learn the testing strategy
* **Code Quality Guide**: Detailed formatting and linting standards
