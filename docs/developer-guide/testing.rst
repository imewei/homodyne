Testing Guide
=============

This guide covers Homodyne's testing strategy, test categories, how to run tests, and guidelines for writing new tests.

Overview
--------

Homodyne employs a comprehensive testing strategy ensuring correctness, performance, and reliability:

* **Total Tests**: 155+ tests across all categories
* **Coverage Target**: > 90% for production code
* **Test Categories**: Unit, Integration, Performance, MCMC, GPU, API, Property, Self-Consistency

Test Categories
---------------

1. Unit Tests (``tests/unit/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test individual functions and classes in isolation.

**Key Files**: ``test_jax_backend.py``, ``test_nlsq_wrapper.py``, ``test_strategy_selection.py`` (41 tests), ``test_parameter_manager.py``, ``test_phi_filtering.py`` (72 tests)

**Run**:

.. code-block:: bash

   make test-unit
   pytest tests/unit/

2. Integration Tests (``tests/integration/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test end-to-end workflows and module interactions.

**Run**:

.. code-block:: bash

   make test-integration
   pytest tests/integration/

3. Performance Tests (``tests/performance/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmark performance and detect regressions.

**Run**:

.. code-block:: bash

   make test-performance
   pytest tests/performance/

4. MCMC Tests (``tests/mcmc/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Statistical validation of MCMC sampling and convergence diagnostics.

**Run**:

.. code-block:: bash

   make test-mcmc
   pytest tests/mcmc/

5. GPU Tests (``tests/gpu/``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate GPU acceleration (Linux + CUDA only, automatically skipped elsewhere).

**Run**:

.. code-block:: bash

   make test-gpu
   pytest tests/gpu/

6. Other Categories
~~~~~~~~~~~~~~~~~~~

* **API Tests** (``tests/api/``): Backward compatibility
* **Property Tests** (``tests/property/``): Mathematical invariants
* **Self-Consistency** (``tests/self_consistency/``): Scientific validation

Running Tests
-------------

Make Commands
~~~~~~~~~~~~~

.. code-block:: bash

   make test              # Core unit tests
   make test-all          # All tests with coverage
   make test-unit         # Unit tests only
   make test-integration  # Integration tests only
   make test-performance  # Performance benchmarks
   make test-nlsq         # NLSQ optimization tests
   make test-mcmc         # MCMC statistical validation
   make test-gpu          # GPU validation (Linux + GPU only)

Pytest Commands
~~~~~~~~~~~~~~~

**Specific tests**:

.. code-block:: bash

   pytest tests/unit/test_jax_backend.py                    # Single file
   pytest tests/unit/test_jax_backend.py::test_compute_g2   # Single test
   pytest -k "nlsq"                                          # Pattern matching

**Verbosity**:

.. code-block:: bash

   pytest -v              # Verbose
   pytest -vv             # Very verbose (full diffs)
   pytest -s              # Show print statements

**Parallel execution**:

.. code-block:: bash

   pip install pytest-xdist
   pytest -n 4            # 4 workers

**Markers**:

.. code-block:: bash

   pytest -m "not slow"   # Exclude slow tests
   pytest -m gpu          # GPU tests only
   pytest -x              # Stop on first failure

Test Infrastructure
-------------------

Configuration
~~~~~~~~~~~~~

**File**: ``pyproject.toml`` (``[tool.pytest.ini_options]``)

**Markers**:

* ``slow``: Long-running tests
* ``gpu``: Requires GPU
* ``integration``: Integration tests
* ``benchmark``: Performance benchmarks

Fixtures
~~~~~~~~

**Shared fixtures**: ``tests/conftest.py``

* ``sample_config``: Test configuration
* ``sample_data``: Synthetic HDF5 data
* ``benchmark_fixture``: Benchmarking utilities

Test Factories
~~~~~~~~~~~~~~

**Location**: ``tests/factories/``

* ``data_factory.py``: Synthetic XPCS data
* ``config_factory.py``: Test configurations
* ``parameter_factory.py``: Parameter sets

CI/CD Integration
-----------------

GitHub Actions
~~~~~~~~~~~~~~

**Workflows** (``.github/workflows/``):

1. **ci.yml**: Unit + Integration tests (Linux, macOS, Windows)
2. **quality.yml**: Code quality checks + coverage
3. **release.yml**: Build and publish (on tag push)

Coverage Reports
~~~~~~~~~~~~~~~~

**Generate coverage**:

.. code-block:: bash

   pytest --cov=homodyne --cov-report=html
   open htmlcov/index.html

Writing New Tests
-----------------

Unit Test Pattern
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_function_name_scenario():
       """Test that function_name does expected_behavior."""
       # Arrange
       input_data = create_test_input()

       # Act
       result = function_name(input_data)

       # Assert
       assert result == expected_output

Integration Test Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_end_to_end_workflow(tmp_path):
       """Test complete analysis workflow."""
       config = tmp_path / "config.yaml"
       output = tmp_path / "results"

       result = run_analysis(config_path=str(config), output_dir=str(output))

       assert result['success']
       assert (output / "parameters.json").exists()

Performance Test Pattern
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.slow
   @pytest.mark.benchmark
   def test_performance_target():
       """Test meets performance targets."""
       import time

       start = time.time()
       result = run_optimization(large_dataset)
       elapsed = time.time() - start

       assert elapsed < 60.0, f"Took {elapsed:.1f}s, should be < 60s"

Best Practices
--------------

**Test Pyramid**:

* 70% Unit Tests (fast, isolated)
* 20% Integration Tests (workflows)
* 10% End-to-End Tests (full system)

**Fast Tests**:

* Unit tests in milliseconds
* Total suite < 5 minutes (excluding ``@pytest.mark.slow``)

**Clear Naming**:

* ``test_<function>_<scenario>_<expected_outcome>``
* Example: ``test_nlsq_wrapper_large_dataset_uses_streaming_strategy``

**AAA Pattern** (Arrange-Act-Assert):

.. code-block:: python

   def test_example():
       # Arrange: Setup test data
       data = create_test_data()

       # Act: Execute function under test
       result = process(data)

       # Assert: Verify expected outcome
       assert result.is_valid

**Edge Cases**:

.. code-block:: python

   @pytest.mark.parametrize("n_points", [0, 1, 1000, 1_000_000])
   def test_various_sizes(n_points):
       """Test handles various dataset sizes."""
       result = process(n_points)
       assert result.success

Debugging Tests
---------------

**Run with debugging**:

.. code-block:: bash

   pytest --pdb           # Drop into debugger on failure
   pytest -l              # Show local variables
   pytest -vv -s          # Very verbose + show prints

**Common Pitfalls**:

1. **Test Interdependence**: Use fixtures instead of shared state
2. **Non-Deterministic**: Set random seeds (``np.random.seed(42)``)
3. **Slow Tests**: Mark with ``@pytest.mark.slow``

Resources
---------

* **Pytest**: https://docs.pytest.org/
* **Coverage.py**: https://coverage.readthedocs.io/
* **Scientific Validation Report**: ``SCIENTIFIC_VALIDATION_REPORT.md``

Next Steps
----------

* **Contributing Guide**: Development workflow and Git conventions
* **Code Quality Guide**: Formatting, linting, and type checking
* **Performance Guide**: Profiling and optimization
