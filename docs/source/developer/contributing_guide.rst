Contributing to Homodyne
========================

Welcome to Homodyne development! This guide covers the contribution workflow,
development setup, code quality standards, and best practices for working on the project.

Development Setup with uv
--------------------------

Homodyne uses **uv** for Python package management, optimized for Python 3.12+.
This provides faster dependency resolution and installation compared to pip.

Installing Development Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Install uv** (if not already installed):

   .. code-block:: bash

       curl -LsSf https://astral.sh/uv/install.sh | sh

   Or using Homebrew on macOS:

   .. code-block:: bash

       brew install uv

2. **Clone the repository**:

   .. code-block:: bash

       git clone https://github.com/ORG/homodyne.git
       cd homodyne

3. **Create virtual environment and install development dependencies**:

   .. code-block:: bash

       uv venv --python 3.12
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate

4. **Install Homodyne in development mode**:

   .. code-block:: bash

       uv pip install -e ".[dev,docs]"

   This installs:
   - Core dependencies (JAX ≥0.8.2 CPU-only, NumPyro, BlackJAX)
   - Development tools (pytest, black, ruff, mypy)
   - Documentation tools (sphinx, sphinx-rtd-theme, myst-parser)

Verifying Installation
^^^^^^^^^^^^^^^^^^^^^^

Verify your development environment is properly configured:

.. code-block:: bash

    # Check Python and JAX versions
    python --version  # Should be 3.12+
    python -c "import jax; print(f'JAX {jax.__version__}')"

    # Validate system configuration
    python -m homodyne.runtime.utils.system_validator --quick

    # Run quick test to verify setup
    make test-unit

Code Quality Standards
----------------------

Homodyne maintains high code quality standards using three primary tools:

Black - Code Formatting
^^^^^^^^^^^^^^^^^^^^^^^

**Black** enforces consistent code formatting across the project.

Configuration in ``pyproject.toml``:

.. code-block:: toml

    [tool.black]
    line-length = 100
    target-version = ['py312']
    include = '\.pyi?$'
    extend-exclude = '''
    /(
        \.git
      | \.venv
      | _build
      | dist
    )/
    '''

Format your code before committing:

.. code-block:: bash

    black homodyne/

Check formatting without modifying:

.. code-block:: bash

    black --check homodyne/

Ruff - Linting
^^^^^^^^^^^^^^

**Ruff** performs fast linting and import sorting.

Key rules enforced:
- F: PyFlakes (undefined names, unused imports)
- E/W: pycodestyle (whitespace, indentation)
- I: isort (import sorting)
- N: pep8-naming (naming conventions)
- UP: pyupgrade (Python 3.12+ features)

Lint your code:

.. code-block:: bash

    ruff check homodyne/

Fix linting issues automatically:

.. code-block:: bash

    ruff check --fix homodyne/

MyPy - Type Checking
^^^^^^^^^^^^^^^^^^^^

**MyPy** validates type hints and catches type-related errors.

Configuration in ``pyproject.toml``:

.. code-block:: toml

    [tool.mypy]
    python_version = "3.12"
    warn_return_any = true
    warn_unused_configs = true
    disallow_incomplete_defs = true
    disallow_untyped_defs = false
    ignore_missing_imports = true

Type-check your code:

.. code-block:: bash

    mypy homodyne/

Complete Quality Check
^^^^^^^^^^^^^^^^^^^^^^

Run all quality checks together:

.. code-block:: bash

    make quality

This runs: black format → ruff lint/fix → mypy type check

Before Committing
^^^^^^^^^^^^^^^^^

Always run the quality check before committing:

.. code-block:: bash

    make quality
    make test-unit  # Ensure tests still pass

Testing with Pytest
-------------------

Homodyne uses **pytest** for comprehensive testing across multiple test suites.

Test Organization
^^^^^^^^^^^^^^^^^

Tests are organized by type in ``tests/``:

.. code-block:: text

    tests/
    ├── unit/              # Function-level tests (fast, <100ms each)
    │   ├── test_*.py      # Test modules
    │   └── optimization/  # Module-specific subdirectories
    ├── integration/       # End-to-end workflow tests
    ├── performance/       # Benchmarks and optimization tests
    ├── mcmc/              # Statistical validation for MCMC
    ├── factories/         # Test data generators
    └── conftest.py        # Pytest configuration and fixtures

Running Tests
^^^^^^^^^^^^^

Run core unit tests (fastest):

.. code-block:: bash

    make test

Run all tests including integration and performance:

.. code-block:: bash

    make test-all

Run specific test suite:

.. code-block:: bash

    make test-unit         # Unit tests only
    make test-integration  # Integration tests
    make test-nlsq         # NLSQ optimization tests
    make test-mcmc         # MCMC validation tests

Run tests with coverage:

.. code-block:: bash

    make test-all
    # Coverage report written to htmlcov/index.html

Run specific test file or test:

.. code-block:: bash

    pytest tests/unit/test_parameter_manager.py
    pytest tests/unit/test_parameter_manager.py::test_specific_test_name -v

Writing Tests
^^^^^^^^^^^^^

Guidelines for writing tests:

**1. Test Function Naming**

Use descriptive names starting with ``test_``:

.. code-block:: python

    # Good
    def test_static_mode_parameter_validation():
        pass

    # Poor
    def test_params():
        pass

**2. Test Structure (Arrange-Act-Assert)**

.. code-block:: python

    def test_parameter_manager_bounds_validation():
        # Arrange: Set up test data
        param_mgr = ParameterManager(
            parameter_names=['D0', 'alpha'],
            bounds=[[100, 10000], [-2, 2]]
        )

        # Act: Execute the code being tested
        result = param_mgr.validate_parameters([1000, 0.5])

        # Assert: Verify the result
        assert result is True

**3. Use Fixtures for Reusable Data**

.. code-block:: python

    import pytest

    @pytest.fixture
    def sample_config():
        """Fixture providing test configuration."""
        return {
            'analysis_mode': 'static',
            'initial_parameters': {'D0': 1000, 'alpha': -1.2}
        }

    def test_with_config(sample_config):
        # Use the fixture
        assert sample_config['analysis_mode'] == 'static'

**4. Test XPCS-Specific Behavior**

.. code-block:: python

    def test_c2_computation_range():
        """Test that C2 values stay in physically valid range [1.0, 1.6]."""
        c2_values = compute_c2(...)
        assert np.all(c2_values >= 1.0), "C2 below 1.0 is non-physical"
        assert np.all(c2_values <= 1.6), "C2 above 1.6 is unusual"

Code Coverage Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Target coverage metrics:

- **Overall**: ≥ 80% line coverage
- **Core modules**: ≥ 85% (jax_backend, physics, parameter_manager)
- **Optimization**: ≥ 75% (complex algorithms acceptable)
- **New code**: ≥ 90% before merge

View coverage report:

.. code-block:: bash

    coverage report
    coverage html  # Detailed HTML report in htmlcov/

JAX and CPU Debugging
^^^^^^^^^^^^^^^^^^^^^

Homodyne uses JAX ≥0.8.2 with CPU-only optimization.

Common JAX Debugging Commands
""""""""""""""""""""""""""""""

Enable compilation logging:

.. code-block:: bash

    JAX_LOG_COMPILES=1 python script.py

Disable JIT compilation for easier debugging:

.. code-block:: bash

    JAX_DISABLE_JIT=1 python script.py

Check JAX device configuration:

.. code-block:: python

    import jax
    print(jax.devices())  # Should show CPU device

Monitor CPU usage during optimization:

.. code-block:: bash

    # Terminal 1: Run homodyne optimization
    homodyne --config config.yaml --method nlsq

    # Terminal 2: Monitor CPU usage
    top -H -p $(pgrep -f homodyne)
    htop  # Better alternative to top

CPU Performance Tips
""""""""""""""""""""

- JAX compiles once per unique shape/dtype - first run is slower
- CPU threads controlled by: ``JAX_PLATFORMS=cpu`` and ``OMP_NUM_THREADS``
- For reproducible results: ``JAX_DEFAULT_PRNG_IMPL=threefry_prng``
- Profile with: ``python -m cProfile -s cumtime script.py``

Pull Request Guidelines
------------------------

Before Submitting
^^^^^^^^^^^^^^^^^^

1. **Ensure tests pass**:

   .. code-block:: bash

       make test-all

2. **Run quality checks**:

   .. code-block:: bash

       make quality

3. **Verify documentation builds**:

   .. code-block:: bash

       cd docs && make html

4. **Check for common issues**:

   - No print statements (use logging)
   - No hardcoded file paths (use pathlib and config)
   - No GPU-specific code (CPU-only architecture)
   - Docstrings for public functions

PR Description Template
^^^^^^^^^^^^^^^^^^^^^^^

Use this template for your PR description:

.. code-block:: markdown

    ## Summary
    Brief description of changes (1-2 sentences)

    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature
    - [ ] Performance improvement
    - [ ] Documentation
    - [ ] Refactoring

    ## Testing
    Describe tests added or modified:
    - [ ] Unit tests added
    - [ ] Integration tests added
    - [ ] All tests passing

    ## Documentation
    - [ ] Updated README/docs
    - [ ] Added/updated docstrings
    - [ ] Updated CHANGELOG

    ## Related Issues
    Fixes #ISSUE_NUMBER

Commit Message Guidelines
^^^^^^^^^^^^^^^^^^^^^^^^^^

Use clear, descriptive commit messages:

.. code-block:: text

    # Good
    fix(mcmc): handle CMC subposterior weight normalization

    Resolved issue where CMC subposterior weights didn't sum to 1.0
    in edge cases, causing incorrect posterior combination.

    Fixes #1234

    # Poor
    fixed bug
    update
    WIP: stuff

Standard Commit Prefixes
""""""""""""""""""""""""

- ``fix(scope)``: Bug fixes (e.g., ``fix(mcmc): ...``)
- ``feat(scope)``: New features (e.g., ``feat(cli): ...``)
- ``refactor(scope)``: Code reorganization (e.g., ``refactor(core): ...``)
- ``test(scope)``: Test additions (e.g., ``test(nlsq): ...``)
- ``docs(scope)``: Documentation (e.g., ``docs(config): ...``)
- ``perf(scope)``: Performance improvements (e.g., ``perf(jax): ...``)
- ``chore(scope)``: Maintenance (e.g., ``chore(deps): ...``)

Reference Material
-------------------

For additional information on Homodyne development and commands, see:

- :doc:`/developer/testing_guide` - Detailed testing guide
- :doc:`/configuration/index` - Configuration documentation
- **CLAUDE.md** in project root - Development commands quick reference:

  .. code-block:: bash

      make test              # Run core unit tests
      make test-all          # Run all tests with coverage
      make quality           # Format, lint, and type check
      make dev               # Install development environment
      make clean             # Clean build artifacts

- **Project README** - High-level project overview and features
- **GitHub Issues** - Known issues, features in progress, bug reports

Getting Help
^^^^^^^^^^^^

- **Questions**: Open a Discussion on GitHub
- **Bugs**: File an Issue with reproduction steps
- **Ideas**: Start a Discussion for feature requests
- **Development Help**: Ask in the development Discussions

Thank you for contributing to Homodyne!
