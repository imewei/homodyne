.. _developer_contributing:

Contributing to Homodyne
=========================

This guide explains how to set up a development environment, follow code style conventions,
write and run tests, and submit pull requests to the homodyne project.

.. contents:: Contents
   :local:
   :depth: 2


Development Setup
-----------------

**Prerequisites**:

- Python 3.12 or later
- `uv <https://docs.astral.sh/uv/>`_ package manager (``pip install uv`` or
  see `uv installation docs <https://docs.astral.sh/uv/getting-started/installation/>`_)
- Git

**Clone and install**:

.. code-block:: bash

   git clone https://github.com/your-org/homodyne.git
   cd homodyne
   make dev

The ``make dev`` target is equivalent to:

.. code-block:: bash

   uv sync --all-extras --dev

This installs homodyne in editable mode along with all development dependencies
(ruff, black, mypy, pytest, sphinx, etc.) into a local ``.venv`` directory.

.. note::

   ``uv.lock`` is the single source of truth for all dependency versions. Never
   pin or modify versions in ``pyproject.toml`` without updating ``uv.lock``
   via ``uv sync``.

**Verify the installation**:

.. code-block:: bash

   uv run python -c "import homodyne; print(homodyne.__version__)"
   uv run pytest tests/unit -x -q


Code Style
----------

Homodyne uses a strict code style enforced by automated tools.

**Formatter: Black**

All Python files are formatted with Black (line length 88):

.. code-block:: bash

   uv run black homodyne/ tests/

**Linter: Ruff**

Ruff enforces PEP 8 and additional linting rules:

.. code-block:: bash

   uv run ruff check homodyne/ tests/
   uv run ruff check --fix homodyne/ tests/   # auto-fix safe issues

**Type checker: MyPy**

Strict type checking is enforced at API boundaries:

.. code-block:: bash

   uv run mypy homodyne/

MyPy is configured in ``pyproject.toml`` with ``strict = true`` for the ``homodyne``
package. Third-party libraries without stubs are marked ``ignore_missing_imports = true``
in the per-module overrides.

**Run all quality checks**:

.. code-block:: bash

   make quality

This runs ``ruff check``, ``black --check``, and ``mypy`` in sequence.


Coding Conventions
------------------

**Imports**: Always use explicit imports. Never use ``from module import *``.

.. code-block:: python

   # Correct
   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.utils.logging import get_logger

   # Wrong
   from homodyne.optimization import *

**Type hints**: All public functions must have complete type annotations. Use
``jax.Array`` for JAX array types, ``np.ndarray`` for NumPy arrays.

.. code-block:: python

   import jax
   import numpy as np

   def compute_msd(t: jax.Array, D0: float, alpha: float) -> jax.Array:
       """Compute mean-squared displacement for anomalous diffusion."""
       return 2 * D0 / (1 + alpha) * t ** (1 + alpha)

**Logging**: Use the structured logger from ``homodyne.utils.logging``:

.. code-block:: python

   from homodyne.utils.logging import get_logger, log_phase

   logger = get_logger(__name__)

   with log_phase("NLSQ optimization"):
       result = fit_nlsq_jax(data, config)

   logger.info("Converged", extra={"chi2": result.chi2, "n_iter": result.n_iter})

Never use ``print()`` for diagnostic output. All user-facing output goes through
the logging system.

**Exception handling**: Use narrow exception types at function boundaries.
Broad ``except Exception`` is allowed only at top-level CLI dispatchers.

.. code-block:: python

   # Correct: narrow exception at function boundary
   try:
       data = loader.load(path)
   except OSError as e:
       logger.error("Cannot open data file", extra={"path": path, "error": str(e)})
       raise

   # Wrong: silently swallowing exceptions
   try:
       data = loader.load(path)
   except Exception:
       data = None

**JAX conventions**:

- Never call ``.item()`` on JAX arrays inside JIT-compiled functions.
- Use ``jnp.where(condition, x, y)`` instead of Python ``if/else`` inside JIT.
- Use ``jax.lax.cond`` only when the two branches have significantly different costs.
- Prefer ``jax.numpy`` functions over NumPy inside JIT-compiled code.

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   @jit
   def safe_exp(x: jax.Array, clip: float = 88.0) -> jax.Array:
       """Numerically stable exponential."""
       return jnp.exp(jnp.clip(x, -clip, clip))


Adding New Parameters
---------------------

To add a new physical parameter to an analysis mode:

1. **Add the parameter** to the relevant model class in
   ``homodyne/core/models.py`` and update ``homodyne/core/physics.py``
   with bounds and default values.

2. **Update the JAX backend** in ``homodyne/core/jax_backend.py`` (NLSQ path)
   and ``homodyne/core/physics_cmc.py`` (CMC path).

3. **Update config** in ``homodyne/optimization/nlsq/config.py`` and
   ``homodyne/optimization/cmc/config.py``.

4. **Add unit tests** in ``tests/unit/test_physics.py`` covering the new formula.

5. **Update the YAML template** in ``homodyne/config/`` (run ``homodyne-config``
   to regenerate the template).


Pull Request Workflow
---------------------

1. **Create a branch** from ``main``:

   .. code-block:: bash

      git checkout -b feat/my-feature main

2. **Write code** following the conventions above.

3. **Write tests** covering the new behaviour (see :ref:`developer_testing`).

4. **Run quality checks locally**:

   .. code-block:: bash

      make quality   # format + lint + type-check
      make test      # unit tests

5. **Push and open a PR**:

   .. code-block:: bash

      git push -u origin feat/my-feature
      gh pr create --fill

6. **PR checklist**:

   - [ ] All unit tests pass (``make test``)
   - [ ] No new mypy errors (``make quality``)
   - [ ] New code has docstrings
   - [ ] New mathematical operations have corresponding unit tests
   - [ ] ``CHANGELOG.md`` updated with a one-line entry

**Commit message style** (conventional commits):

.. code-block:: text

   feat(core): add Ornstein-Uhlenbeck transport coefficient
   fix(cmc): prevent division by zero in heterogeneity detection
   docs(theory): add section on shear banding
   test(nlsq): add round-trip test for fourier reparameterization
   chore(deps): update JAX to 0.8.3


Documentation
-------------

Homodyne uses Sphinx with the ReadTheDocs theme. Documentation source is in ``docs/source/``.

**Build the docs**:

.. code-block:: bash

   make docs
   # output in docs/_build/html/

**Adding a new page**:

1. Create a ``.rst`` file in the appropriate subdirectory.
2. Add it to the ``toctree`` directive in the parent ``index.rst``.
3. Build and check for warnings.

All new public API symbols should have NumPy-style docstrings with ``Parameters``,
``Returns``, and ``Examples`` sections.

.. code-block:: python

   def fit_nlsq_jax(
       data: XPCSData,
       config: NLSQConfig,
       *,
       use_adapter: bool = True,
   ) -> NLSQResult:
       """Fit the homodyne model using non-linear least squares.

       Parameters
       ----------
       data : XPCSData
           Loaded XPCS data from :class:`~homodyne.data.XPCSDataLoader`.
       config : NLSQConfig
           Configuration object. Use :func:`~homodyne.config.ConfigManager.load`
           to create from a YAML file.
       use_adapter : bool, optional
           If True (default), use :class:`~homodyne.optimization.nlsq.NLSQAdapter`
           with automatic strategy selection and anti-degeneracy system.

       Returns
       -------
       NLSQResult
           Fitted parameter estimates, covariance matrix, and diagnostics.

       Examples
       --------
       >>> from homodyne.data import XPCSDataLoader
       >>> from homodyne.config import ConfigManager
       >>> from homodyne.optimization.nlsq import fit_nlsq_jax
       >>>
       >>> data = XPCSDataLoader().load("experiment.h5")
       >>> config = ConfigManager().load("config.yaml")
       >>> result = fit_nlsq_jax(data, config)
       >>> print(result.params)
       """
