Code Quality Guide
==================

This guide covers code formatting, linting, type checking, and quality assurance practices for Homodyne development.

Overview
--------

Homodyne enforces strict code quality standards through automated tools:

* **Black**: Code formatting
* **Ruff**: Linting and style checks
* **Mypy**: Static type checking
* **Bandit**: Security scanning
* **Pre-commit**: Automated enforcement

Quick Start
-----------

.. code-block:: bash

   # Format code
   make format

   # Run linting
   make lint

   # Type check
   mypy homodyne/

   # Run all quality checks
   make quality

   # Install pre-commit hooks (automatic checks on commit)
   pre-commit install

Formatting with Black
---------------------

Configuration
~~~~~~~~~~~~~

**File**: ``pyproject.toml``

.. code-block:: toml

   [tool.black]
   line-length = 88
   target-version = ['py312']
   include = '\.pyi?$'
   extend-exclude = '''
   /(
       \.eggs
     | \.git
     | \.hg
     | \.mypy_cache
     | \.tox
     | \.venv
     | _build
     | buck-out
     | build
     | dist
   )/
   '''

Running Black
~~~~~~~~~~~~~

.. code-block:: bash

   # Format all Python files
   make format
   # OR
   black homodyne/

   # Check without modifying
   black homodyne/ --check

   # Show diff
   black homodyne/ --diff

**Line Length**: 88 characters (Black default)

Black automatically handles:

* Consistent indentation (4 spaces)
* String quote normalization (double quotes)
* Trailing commas
* Line breaking

Linting with Ruff
-----------------

Configuration
~~~~~~~~~~~~~

**File**: ``pyproject.toml``

.. code-block:: toml

   [tool.ruff]
   line-length = 120
   target-version = "py312"

   [tool.ruff.lint]
   select = [
       "E",   # pycodestyle errors
       "W",   # pycodestyle warnings
       "F",   # pyflakes
       "I",   # isort
       "B",   # flake8-bugbear
       "C4",  # flake8-comprehensions
       "UP",  # pyupgrade
   ]
   ignore = [
       "E501",  # line too long (handled by Black)
       "B008",  # function calls in argument defaults
       "C901",  # too complex
   ]

Running Ruff
~~~~~~~~~~~~

.. code-block:: bash

   # Check all files
   make lint
   # OR
   ruff check homodyne/

   # Auto-fix issues
   ruff check homodyne/ --fix

   # Format with Ruff
   ruff format homodyne/

**Ruff checks**:

* **E**: pycodestyle errors (PEP 8 violations)
* **W**: pycodestyle warnings
* **F**: pyflakes (undefined names, unused imports)
* **I**: isort (import sorting)
* **B**: flake8-bugbear (common bugs)
* **C4**: flake8-comprehensions (list/dict comprehension issues)
* **UP**: pyupgrade (upgrade syntax for newer Python)

Type Checking with Mypy
------------------------

Configuration
~~~~~~~~~~~~~

**File**: ``pyproject.toml``

.. code-block:: toml

   [tool.mypy]
   python_version = "3.12"
   warn_return_any = true
   warn_unused_configs = true
   disallow_untyped_defs = true
   disallow_any_generics = false
   ignore_missing_imports = true
   exclude = [
       'build',
       'dist',
       '\.eggs',
       '\.git',
       '\.mypy_cache',
       '\.pytest_cache',
       '\.tox',
       '\.venv',
       'venv',
   ]

Running Mypy
~~~~~~~~~~~~

.. code-block:: bash

   # Check all files
   mypy homodyne/

   # Check specific file
   mypy homodyne/core/jax_backend.py

**Type Hints Required** for all public functions:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple, Any
   import numpy as np

   def process_data(
       data: np.ndarray,
       config: Dict[str, Any],
       threshold: float = 0.5,
   ) -> Tuple[np.ndarray, Dict[str, float]]:
       """Process experimental data."""
       pass

**Common Type Annotations**:

.. code-block:: python

   # Basic types
   count: int = 0
   name: str = "test"
   is_valid: bool = True
   value: float = 3.14

   # Collections
   numbers: List[int] = [1, 2, 3]
   mapping: Dict[str, float] = {"a": 1.0}
   coordinates: Tuple[float, float] = (0.0, 0.0)

   # Optional
   result: Optional[np.ndarray] = None

   # Any (use sparingly)
   config: Dict[str, Any] = {}

   # NumPy arrays
   data: np.ndarray  # Any array
   matrix: np.ndarray[tuple[int, int], np.dtype[np.float64]]  # Specific shape

Pre-commit Hooks
----------------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Install pre-commit
   pip install pre-commit

   # Install git hooks
   pre-commit install

   # Verify installation
   pre-commit --version

Configured Hooks
~~~~~~~~~~~~~~~~

**File**: ``.pre-commit-config.yaml``

Hooks run automatically on ``git commit``:

1. **Black**: Code formatting
2. **Ruff**: Linting + formatting
3. **isort**: Import sorting
4. **Mypy**: Type checking
5. **Bandit**: Security scanning
6. **Flake8**: Style guide enforcement
7. **Trailing whitespace**: Remove trailing spaces
8. **End of file fixer**: Ensure files end with newline
9. **YAML/JSON/TOML check**: Validate syntax

Running Hooks Manually
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all hooks on all files
   pre-commit run --all-files

   # Run specific hook
   pre-commit run black --all-files
   pre-commit run mypy --all-files

   # Run on staged files only
   pre-commit run

   # Update hooks to latest versions
   pre-commit autoupdate

Bypassing Hooks
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Skip hooks (not recommended)
   git commit --no-verify

**When to bypass**:

* WIP commits in feature branch (fix before PR)
* Emergency hotfixes (fix immediately after)
* **Never bypass** for commits to ``main``

Security Scanning with Bandit
------------------------------

Configuration
~~~~~~~~~~~~~

**File**: ``.pre-commit-config.yaml``

.. code-block:: yaml

   - id: bandit
     args:
       - -r
       - --severity-level
       - medium
       - --skip
       - B101,B110,B403,B404,B603
     exclude: ^(tests/|build/|dist/)

Running Bandit
~~~~~~~~~~~~~~

.. code-block:: bash

   # Scan all files
   bandit -r homodyne/

   # Save report
   bandit -r homodyne/ -f json -o bandit_report.json

**Skipped Checks**:

* **B101**: assert_used (allow asserts in tests)
* **B110**: try_except_pass (acceptable in some cases)
* **B403**: import_pickle (used safely)
* **B404**: import_subprocess (used safely)
* **B603**: subprocess_without_shell_equals_true (false positives)

Import Sorting with isort
--------------------------

Configuration
~~~~~~~~~~~~~

**File**: ``pyproject.toml``

.. code-block:: toml

   [tool.isort]
   profile = "black"
   line_length = 88
   multi_line_output = 3
   include_trailing_comma = true
   force_grid_wrap = 0
   use_parentheses = true
   ensure_newline_before_comments = true

Running isort
~~~~~~~~~~~~~

.. code-block:: bash

   # Sort imports
   isort homodyne/

   # Check only
   isort homodyne/ --check-only

   # Show diff
   isort homodyne/ --diff

**Import Order**:

1. Standard library imports
2. Third-party imports
3. Local application imports

.. code-block:: python

   # Standard library
   import os
   import sys
   from pathlib import Path

   # Third-party
   import jax
   import jax.numpy as jnp
   import numpy as np
   from scipy import optimize

   # Local
   from homodyne.config.manager import ConfigManager
   from homodyne.core.jax_backend import compute_g2_scaled

Code Quality Makefile Targets
------------------------------

Available Commands
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make format        # Auto-format with Black + Ruff
   make lint          # Run Ruff linting
   make type-check    # Run Mypy type checking
   make quality       # Run all quality checks (format + lint + type-check)
   make pre-commit    # Run pre-commit hooks manually

Best Practices
--------------

General Guidelines
~~~~~~~~~~~~~~~~~~

**1. Run Quality Checks Before Commit**:

.. code-block:: bash

   make format
   make lint
   mypy homodyne/
   # OR
   make quality

**2. Fix Issues Immediately**:

* Don't accumulate linting errors
* Address type errors as they arise
* Keep code consistently formatted

**3. Use Pre-commit Hooks**:

* Automatic quality enforcement
* Catch issues before CI
* Consistent across team

**4. Review Tool Output**:

* Understand why checks fail
* Don't blindly auto-fix
* Learn from warnings

Handling False Positives
~~~~~~~~~~~~~~~~~~~~~~~~~

**Ruff - Ignore specific line**:

.. code-block:: python

   result = potentially_dangerous_operation()  # noqa: S307

**Mypy - Ignore specific line**:

.. code-block:: python

   value = external_library.get_value()  # type: ignore[attr-defined]

**File-level ignores**:

.. code-block:: python

   # ruff: noqa: F401  (unused imports)
   from homodyne import *  # Export all

Type Checking Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Use Specific Types**:

❌ **Bad**:

.. code-block:: python

   def process(data: Any) -> Any:
       pass

✅ **Good**:

.. code-block:: python

   def process(data: np.ndarray) -> Dict[str, float]:
       pass

**2. Avoid Bare ``except``**:

❌ **Bad**:

.. code-block:: python

   try:
       result = risky_operation()
   except:  # Catches everything, including KeyboardInterrupt
       pass

✅ **Good**:

.. code-block:: python

   try:
       result = risky_operation()
   except (ValueError, TypeError) as e:
       logger.error(f"Operation failed: {e}")
       raise

**3. Document Complex Types**:

.. code-block:: python

   from typing import TypedDict, Dict, List

   class ParameterBounds(TypedDict):
       """Parameter bounds specification."""
       name: str
       min: float
       max: float

   def validate_bounds(bounds: List[ParameterBounds]) -> bool:
       """Validate parameter bounds."""
       pass

CI/CD Integration
-----------------

GitHub Actions Quality Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**File**: ``.github/workflows/quality.yml``

Runs on every push and PR:

1. **Black**: Check formatting
2. **Ruff**: Lint code
3. **Mypy**: Type check
4. **Bandit**: Security scan
5. **Coverage**: Upload to Codecov

**All checks must pass before merge**.

Common Issues and Solutions
---------------------------

Import Errors
~~~~~~~~~~~~~

**Issue**: ``ModuleNotFoundError`` during type checking

**Solution**:

.. code-block:: bash

   # Install type stubs
   pip install types-psutil types-requests

   # OR add to mypy config
   # mypy.ini: ignore_missing_imports = true

Conflicting Formatters
~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Black and Ruff disagree on formatting

**Solution**: Black takes precedence. Configure Ruff:

.. code-block:: toml

   [tool.ruff]
   line-length = 88  # Match Black

Type Stub Not Found
~~~~~~~~~~~~~~~~~~~

**Issue**: ``error: Skipping analyzing ... library stubs not installed``

**Solution**:

.. code-block:: bash

   pip install types-<package>
   # OR
   mypy --install-types

Resources
---------

* **Black**: https://black.readthedocs.io/
* **Ruff**: https://docs.astral.sh/ruff/
* **Mypy**: https://mypy.readthedocs.io/
* **Bandit**: https://bandit.readthedocs.io/
* **Pre-commit**: https://pre-commit.com/
* **PEP 8**: https://peps.python.org/pep-0008/
* **Type Hints (PEP 484)**: https://peps.python.org/pep-0484/

Next Steps
----------

* **Performance Guide**: Learn profiling and optimization techniques
* **Testing Guide**: Understand the testing strategy
* **Contributing Guide**: Full development workflow
