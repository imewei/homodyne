Testing Guide
=============

Homodyne maintains comprehensive test coverage with multiple test suites
organized by type. This guide covers test organization, running tests,
coverage requirements, and JAX debugging techniques.

Test Organization
------------------

Tests are organized in the ``tests/`` directory by test type:

.. code-block:: text

    tests/
    ├── unit/                          # Unit tests (function-level)
    │   ├── test_parameter_manager.py
    │   ├── test_parameter_config.py
    │   ├── test_config_validation.py
    │   ├── test_sphinx_configuration.py
    │   ├── test_developer_configuration_docs.py
    │   ├── optimization/
    │   │   ├── nlsq/                 # NLSQ-specific tests
    │   │   └── cmc/                  # CMC/MCMC-specific tests
    │   └── other_modules/            # Other module tests
    ├── integration/                   # Integration tests (end-to-end)
    │   ├── test_nlsq_workflow.py
    │   ├── test_mcmc_workflow.py
    │   └── test_cli_commands.py
    ├── performance/                   # Performance benchmarks
    │   └── test_optimization_speed.py
    ├── mcmc/                          # MCMC statistical validation
    │   ├── test_convergence.py
    │   └── test_cmc_diagnostics.py
    ├── factories/                     # Test data generators
    │   ├── config_factory.py
    │   └── data_factory.py
    └── conftest.py                    # Pytest configuration and fixtures

Unit Tests
^^^^^^^^^^

**Purpose**: Test individual functions and classes in isolation

**Characteristics**:
- Fast (milliseconds to seconds)
- No external dependencies (filesystem, network)
- Test single responsibility
- High coverage (aim for >85%)

**Location**: ``tests/unit/``

**Example**:

.. code-block:: python

    def test_parameter_manager_validates_bounds():
        """Test that ParameterManager validates parameter bounds."""
        mgr = ParameterManager(
            parameter_names=['D0', 'alpha'],
            bounds=[[100, 10000], [-2, 2]]
        )

        # Valid parameters should pass
        valid_result = mgr.validate_parameters([1000, 0.5])
        assert valid_result is True

        # Parameters outside bounds should fail
        invalid_result = mgr.validate_parameters([0.1, 10.0])
        assert invalid_result is False

Integration Tests
^^^^^^^^^^^^^^^^^

**Purpose**: Test complete workflows and module interactions

**Characteristics**:
- Slower than unit tests (seconds to minutes)
- May use real data files (small test datasets)
- Test end-to-end workflows
- Validate CLI commands
- Coverage: ≥60%

**Location**: ``tests/integration/``

**Example**:

.. code-block:: python

    def test_nlsq_static_mode_workflow():
        """Test complete NLSQ optimization workflow for static mode."""
        # Create test configuration
        config = create_test_config(
            analysis_mode='static',
            method='nlsq'
        )

        # Run optimization (full workflow)
        results = run_homodyne_analysis(config)

        # Validate results
        assert results['method'] == 'nlsq'
        assert 'parameters' in results
        assert 'uncertainties' in results

Performance Tests
^^^^^^^^^^^^^^^^^

**Purpose**: Benchmark computational performance

**Characteristics**:
- Measure execution time and memory usage
- Track performance regressions
- Identify optimization opportunities
- Use representative datasets

**Location**: ``tests/performance/``

**Example**:

.. code-block:: python

    def test_jax_backend_jit_compilation_speed():
        """Benchmark JAX JIT compilation for g2 computation."""
        # Measure compilation time
        start = time.time()
        result = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)
        compile_time = time.time() - start

        # JIT first call is slower (compilation)
        # Second call should be much faster (cached)
        start = time.time()
        result = compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset, dt)
        cached_time = time.time() - start

        assert cached_time < compile_time / 10, "JIT caching not working"

MCMC Tests
^^^^^^^^^^

**Purpose**: Validate statistical properties of MCMC inference

**Characteristics**:
- Test convergence diagnostics (R-hat, ESS)
- Validate posterior distributions
- Check CMC combination methods
- Slower but critical for statistical correctness

**Location**: ``tests/mcmc/``

**Example**:

.. code-block:: python

    def test_cmc_convergence_diagnostics():
        """Test CMC subposterior combination produces valid diagnostics."""
        # Run CMC inference
        mcmc_results = run_cmc_inference(test_config)

        # Check R-hat < 1.1 (convergence)
        assert np.all(mcmc_results['rhat'] < 1.1), "MCMC not converged"

        # Check ESS > 100 per parameter (sufficient samples)
        assert np.all(mcmc_results['ess'] > 100), "Low effective sample size"

        # Check posterior is unimodal
        assert is_posterior_reasonable(mcmc_results['samples'])

Running Tests
--------------

Quick Test Commands
^^^^^^^^^^^^^^^^^^^^

Run core unit tests (fastest, ~30 seconds):

.. code-block:: bash

    make test

Run all tests with coverage (comprehensive, ~2-3 minutes):

.. code-block:: bash

    make test-all

Test-Specific Commands
^^^^^^^^^^^^^^^^^^^^^^

Run unit tests only:

.. code-block:: bash

    make test-unit

Run integration tests:

.. code-block:: bash

    make test-integration

Run NLSQ optimization tests:

.. code-block:: bash

    make test-nlsq

Run MCMC validation tests:

.. code-block:: bash

    make test-mcmc

Advanced Pytest Options
^^^^^^^^^^^^^^^^^^^^^^^

Run specific test file:

.. code-block:: bash

    pytest tests/unit/test_parameter_manager.py -v

Run specific test:

.. code-block:: bash

    pytest tests/unit/test_parameter_manager.py::test_parameter_manager_validates_bounds -v

Run tests matching pattern:

.. code-block:: bash

    pytest tests/ -k "parameter_manager" -v

Run with verbose output:

.. code-block:: bash

    pytest tests/unit/ -vv

Run and stop on first failure:

.. code-block:: bash

    pytest tests/unit/ -x

Run with detailed output including print statements:

.. code-block:: bash

    pytest tests/unit/ -s

Run with coverage:

.. code-block:: bash

    pytest tests/ --cov=homodyne --cov-report=html

Filter by test marker:

.. code-block:: bash

    pytest tests/ -m "not slow"  # Skip slow tests

Code Coverage
--------------

Measuring Coverage
^^^^^^^^^^^^^^^^^^

Run tests with coverage measurement:

.. code-block:: bash

    make test-all

This generates coverage data in ``.coverage`` and HTML report in ``htmlcov/``

View coverage report:

.. code-block:: bash

    # Terminal report
    coverage report

    # Detailed HTML report
    coverage html
    open htmlcov/index.html  # On macOS/Linux
    start htmlcov\index.html # On Windows

Coverage Requirements
^^^^^^^^^^^^^^^^^^^^^

**Target Coverage Metrics**:

- **Overall**: ≥ 80% line coverage
- **Core modules**: ≥ 85%
  - ``homodyne.core.jax_backend``
  - ``homodyne.core.physics``
  - ``homodyne.config.parameter_manager``
  - ``homodyne.config.manager``

- **Optimization**: ≥ 75% (complex algorithms acceptable)
  - ``homodyne.optimization.nlsq``
  - ``homodyne.optimization.mcmc``

- **New code**: ≥ 90% before PR merge

Improving Coverage
^^^^^^^^^^^^^^^^^^

Identify uncovered lines:

.. code-block:: bash

    coverage report --skip-covered

View uncovered lines in specific module:

.. code-block:: bash

    coverage report --include=homodyne/core/*.py

HTML report shows uncovered lines with red highlighting:

.. code-block:: bash

    coverage html && open htmlcov/index.html

JAX Debugging
--------------

Understanding JAX Compilation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JAX uses JIT (Just-In-Time) compilation to optimize numerical computation.
First execution of a function with new input shapes is slower due to compilation.

**Disable JIT for debugging**:

.. code-block:: bash

    JAX_DISABLE_JIT=1 python test_script.py

With JIT disabled, errors appear at their source code location instead of
in compiled JAX bytecode. This makes debugging much easier.

**Example debugging**:

.. code-block:: python

    import os
    os.environ['JAX_DISABLE_JIT'] = '1'  # Must be before JAX imports

    from homodyne.core.jax_backend import compute_chi_squared
    result = compute_chi_squared(params, data, sigma, t1, t2, phi, q, L, contrast, offset, dt)  # Now shows Python traceback

Profiling JAX Code
^^^^^^^^^^^^^^^^^^^

Enable compilation logging to see what gets compiled:

.. code-block:: bash

    JAX_LOG_COMPILES=1 python script.py 2>&1 | head -50

This shows compilation events with function names and shapes.

Monitor memory during execution:

.. code-block:: bash

    # Terminal 1: Run your script
    python homodyne_analysis.py

    # Terminal 2: Monitor memory
    while true; do
        ps aux | grep python | grep -v grep | awk '{print "Memory:", $6 " KB"}'
        sleep 1
    done

Profile CPU usage:

.. code-block:: bash

    python -m cProfile -s cumtime script.py | head -30

JAX Device Configuration
^^^^^^^^^^^^^^^^^^^^^^^^

Verify JAX is using CPU (not GPU):

.. code-block:: python

    import jax
    print("JAX devices:", jax.devices())
    print("JAX config:", jax.config)

Expected output for CPU-only:

.. code-block:: text

    JAX devices: (CpuDevice(id=0),)

Common JAX Debugging Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Issue**: "RuntimeError: No backend found matching jaxlib_gpu"

**Solution**: CPU-only configuration is correct. JAX v2.3.0+ defaults to CPU.

**Issue**: Shape mismatches in JIT compiled functions

**Solution**: Disable JIT to see original error:

.. code-block:: bash

    JAX_DISABLE_JIT=1 python script.py

**Issue**: Slow first execution

**Solution**: Normal behavior. JAX compiles on first call with new shapes.
Second call should be 10-100x faster.

**Issue**: OOM (Out of Memory) on large datasets

**Solution**:
1. Use streaming strategy for >100M points
2. Reduce batch size in stratification
3. Enable memory caching: ``memory_optimization.enabled: true``

Environment Variables
^^^^^^^^^^^^^^^^^^^^^^

Useful JAX environment variables:

.. code-block:: bash

    # Disable JIT compilation (for debugging)
    JAX_DISABLE_JIT=1

    # Show compilation events
    JAX_LOG_COMPILES=1

    # Set CPU thread count
    OMP_NUM_THREADS=4

    # Use specific PRNG implementation
    JAX_DEFAULT_PRNG_IMPL=threefry_prng

    # Enable float64 precision (default is float32)
    JAX_ENABLE_CUSTOM_PRNG=1

Continuous Integration
-----------------------

Tests run automatically on GitHub pull requests via GitHub Actions.

**PR Checks**:
1. ✓ Unit tests pass
2. ✓ Integration tests pass
3. ✓ Code coverage maintained (≥80%)
4. ✓ Code quality passes (black, ruff, mypy)
5. ✓ Documentation builds without errors

**Before pushing**, run locally:

.. code-block:: bash

    make quality    # Format, lint, type-check
    make test-all   # Run all tests
    cd docs && make html  # Build docs

Troubleshooting Tests
---------------------

Test Failures
^^^^^^^^^^^^^

When a test fails, first check:

1. **Does the test fail consistently?**

   .. code-block:: bash

       pytest tests/failing_test.py -v --tb=long

2. **What's the actual error?**

   Look at pytest output for assertion error or exception

3. **Can you reproduce locally?**

   Run the exact same pytest command from test output

4. **Is it a JAX/numerical issue?**

   Check if using ``np.allclose()`` instead of exact equality:

   .. code-block:: python

       # Better: allows for floating-point rounding
       np.testing.assert_allclose(result, expected, rtol=1e-5)

       # Fragile: exact equality often fails with floats
       assert result == expected

Flaky Tests
^^^^^^^^^^^

Tests that sometimes pass, sometimes fail:

**Common causes**:
- Randomness without seed
- Timing-dependent tests
- Shared state between tests

**Fixes**:
- Set random seed in test: ``np.random.seed(42)``
- Use fixtures for test setup
- Isolate state with mocking

Run flaky test multiple times:

.. code-block:: bash

    pytest tests/flaky_test.py -v --count=10

Reference Commands
-------------------

Quick Reference Table
^^^^^^^^^^^^^^^^^^^^^

.. table:: Common Testing Commands
   :widths: 50 50

   ==========================================  =================================
   Command                                     Purpose
   ==========================================  =================================
   ``make test``                               Run core unit tests
   ``make test-all``                           Run all tests + coverage
   ``make test-unit``                          Unit tests only
   ``make test-integration``                   Integration tests only
   ``make test-nlsq``                          NLSQ optimization tests
   ``make test-mcmc``                          MCMC validation tests
   ``pytest tests/unit/ -v``                   Run unit tests verbose
   ``pytest tests/ -k pattern``                Run tests matching pattern
   ``pytest tests/ -x``                        Stop on first failure
   ``pytest tests/ --tb=short``                Short traceback format
   ``coverage report``                         Show coverage summary
   ``coverage html``                           Generate HTML coverage report
   ``JAX_DISABLE_JIT=1 pytest tests/``         Test without JIT compilation
   ``JAX_LOG_COMPILES=1 python script.py``     Log JAX compilation events
   ==========================================  =================================

For Development
^^^^^^^^^^^^^^^

Run this before committing:

.. code-block:: bash

    make quality
    make test-all

If all pass, your changes are ready for PR!
