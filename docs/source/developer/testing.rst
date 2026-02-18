.. _developer_testing:

Testing Strategy
=================

Homodyne uses a multi-layered test suite to verify correctness at the unit level
(individual functions), integration level (full optimization pipelines), and numerical
level (round-trip accuracy of the physics implementation).

.. contents:: Contents
   :local:
   :depth: 2


Test Layout
-----------

.. code-block:: text

   tests/
   ├── unit/
   │   ├── test_physics.py             # Physics constants and bounds
   │   ├── test_physics_utils.py       # safe_exp, safe_sinc, epsilon tolerances
   │   ├── test_jax_backend.py         # g2 computation (NLSQ path)
   │   ├── test_models.py              # DiffusionModel, ShearModel
   │   ├── test_nlsq_core.py           # NLSQ optimization round-trips
   │   ├── test_nlsq_anti_degeneracy.py # Per-angle mode behaviour
   │   ├── test_cmc_nlsq_comparison.py # CMC vs NLSQ agreement on synthetic data
   │   ├── test_cmc_sampler.py         # SamplingPlan adaptive scaling
   │   ├── test_reparameterization.py  # Log-space prior transforms
   │   ├── test_config.py              # YAML loading and validation
   │   └── test_data_loader.py         # HDF5 I/O and validation
   ├── integration/
   │   ├── test_nlsq_pipeline.py       # YAML → data → NLSQ → result
   │   └── test_cmc_pipeline.py        # YAML → data → CMC → diagnostics
   └── conftest.py                     # Shared fixtures


Running Tests
-------------

**Unit tests only** (fast, no CMC chains):

.. code-block:: bash

   make test
   # equivalent to: uv run pytest tests/unit -x -q

**Full suite with coverage**:

.. code-block:: bash

   make test-all
   # equivalent to: uv run pytest tests/ --cov=homodyne --cov-report=html

**Single file**:

.. code-block:: bash

   uv run pytest tests/unit/test_nlsq_core.py -v

**Single class or function**:

.. code-block:: bash

   uv run pytest tests/unit/test_physics.py::TestParameterBounds -v
   uv run pytest tests/unit/test_jax_backend.py::test_g2_static_zero_shear -v

**Filter by keyword**:

.. code-block:: bash

   uv run pytest tests/ -k "anti_degeneracy" -v

**With JAX 64-bit enabled** (required for high-precision numerical tests):

.. code-block:: bash

   JAX_ENABLE_X64=1 uv run pytest tests/unit/test_jax_backend.py -v


Unit Test Conventions
----------------------

Every test file targets a single module. Tests are grouped into classes by function
or behaviour being tested:

.. code-block:: python

   """Tests for homodyne.core.jax_backend — g2 computation."""
   import jax
   import jax.numpy as jnp
   import numpy as np
   import pytest

   from homodyne.core.jax_backend import compute_g2_static, compute_g2_laminar_flow


   class TestG2Static:
       """Tests for g2 in static (no shear) mode."""

       def test_baseline_value(self) -> None:
           """g2 must equal 1 + beta at zero lag."""
           t = jnp.array([1.0, 2.0, 5.0])
           beta = 0.3
           params = jnp.array([50.0, 0.5, 0.1])  # D0, alpha, D_offset
           c2 = compute_g2_static(params, t, t, q=0.01)
           np.testing.assert_allclose(c2, 1 + beta, rtol=1e-5)

       def test_monotone_decay(self) -> None:
           """g2 must decrease monotonically as lag increases."""
           t1 = jnp.full((10,), 1.0)
           t2 = jnp.linspace(1.0, 100.0, 10)
           params = jnp.array([50.0, 0.5, 0.1])
           c2 = compute_g2_static(params, t1, t2, q=0.01)
           assert jnp.all(jnp.diff(c2) <= 0)


**Numerical tolerance conventions**:

- Use ``np.testing.assert_allclose(actual, desired, rtol=..., atol=...)`` for
  floating-point comparisons. Never use ``==`` for arrays.
- Set ``rtol=1e-5`` for 32-bit JAX computations.
- Set ``rtol=1e-10`` or tighter for 64-bit JAX computations (with ``JAX_ENABLE_X64=1``).
- After the P0 numerical fixes (Feb 2026), epsilon tolerances in
  ``test_physics_utils.py`` use ``atol=1e-12`` for smooth-abs tests.


Round-Trip Tests
-----------------

Physics round-trip tests verify that:

1. Synthetic :math:`c_2` data can be generated from known parameters.
2. The optimizer recovers those parameters within tolerance.

Example pattern:

.. code-block:: python

   def test_nlsq_round_trip_static():
       """NLSQ must recover ground-truth parameters from synthetic data."""
       from homodyne.optimization.nlsq import fit_nlsq_jax

       # Ground truth
       theta_true = np.array([50.0, 0.5, 0.1])  # D0, alpha, D_offset

       # Generate synthetic c2 (no noise)
       t = np.linspace(0.01, 100.0, 50)
       t1, t2 = np.meshgrid(t, t, indexing="ij")
       c2_synth = generate_c2_static(theta_true, t1, t2, q=0.01, beta=0.3)

       # Fit
       data = make_xpcs_data(t1, t2, c2_synth, q=0.01)
       config = make_static_config()
       result = fit_nlsq_jax(data, config)

       np.testing.assert_allclose(result.params, theta_true, rtol=1e-3)

Round-trip tests are in ``tests/unit/test_nlsq_core.py`` and
``tests/unit/test_cmc_nlsq_comparison.py``.


Test Fixtures
-------------

Shared fixtures are defined in ``tests/conftest.py``:

.. code-block:: python

   @pytest.fixture
   def synthetic_static_data():
       """Generate synthetic c2 data for static mode."""
       ...

   @pytest.fixture
   def synthetic_laminar_flow_data():
       """Generate synthetic c2 data for laminar_flow mode with Andrade creep."""
       ...

   @pytest.fixture
   def nlsq_config_static():
       """Minimal NLSQConfig for static mode tests."""
       ...

   @pytest.fixture
   def cmc_config_minimal():
       """CMCConfig with very small shard size for fast tests."""
       ...

Use ``conftest.py`` fixtures for anything that takes more than 2 lines to set up.
Never duplicate fixture logic across test files.


Integration Tests
-----------------

Integration tests run the full pipeline end-to-end with a minimal synthetic dataset:

.. code-block:: bash

   uv run pytest tests/integration/ -v --timeout=120

Integration tests may take 30–120 seconds due to JIT compilation and NUTS warmup.
They are excluded from ``make test`` (unit only) but included in ``make test-all``.

Integration test criteria:

- NLSQ converges (``result.converged == True``).
- Chi-squared is reasonable: :math:`\chi^2_\nu < 2`.
- CMC :math:`\hat{R} < 1.05` for all parameters.
- CMC ESS :math:`> 100` per parameter (relaxed for fast test datasets).


Continuous Integration
-----------------------

The CI pipeline (GitHub Actions) runs on every pull request:

1. **quality**: ``make quality`` (ruff + black + mypy)
2. **test-unit**: ``make test`` on Python 3.12 (Linux)
3. **test-all**: ``make test-all`` on Python 3.12 (Linux) — generates coverage report

Coverage target: :math:`> 85\%` for the ``core/`` and ``optimization/`` packages.

.. tip::

   To reproduce CI locally:

   .. code-block:: bash

      make quality && make test-all


Debugging Failing Tests
------------------------

**JAX tracing errors** (``ConcretizationTypeError``):

This means Python control flow is operating on traced (abstract) JAX values inside a
JIT-compiled function. Fix: move the conditional outside JIT, or use ``jax.lax.cond``.

.. code-block:: python

   # Wrong: Python if on JAX array inside @jit
   @jit
   def bad(x):
       if x > 0:           # ConcretizationTypeError
           return x
       return -x

   # Correct: jnp.where is JIT-safe
   @jit
   def good(x):
       return jnp.where(x > 0, x, -x)

**Numerical instability in tests**:

Enable JAX 64-bit mode and check for NaN/Inf:

.. code-block:: bash

   JAX_ENABLE_X64=1 JAX_DEBUG_NANS=1 uv run pytest tests/unit/test_jax_backend.py -v

``JAX_DEBUG_NANS=1`` raises an error at the first NaN, with a traceback pointing to the
JAX operation that produced it.

**CMC divergences in tests**:

High divergence rates (:math:`> 10\%`) in CMC unit tests indicate that the NUTS step size
is too large or the posterior geometry is challenging. Check:

1. Is NLSQ warm-start being used? (Should reduce divergences from ~28% to <5%.)
2. Is ``max_tree_depth`` set reasonably (default 10)?
3. Is ``max_divergence_rate`` in ``CMCConfig.from_dict()`` set correctly?

See :ref:`theory_computational_methods` for NUTS diagnostic interpretation.
