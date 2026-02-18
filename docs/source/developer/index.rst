.. _developer-guide:

Developer Guide
===============

This section is for contributors and developers who want to understand
Homodyne's internal design, run the test suite, or submit improvements.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   contributing
   testing

.. toctree::
   :maxdepth: 2
   :caption: Design & Architecture

   architecture
   adrs/index

----

Quick Reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Task
     - Command
   * - Install with dev extras
     - ``make dev``
   * - Run unit tests
     - ``make test``
   * - Run full suite with coverage
     - ``make test-all``
   * - Format + lint + type-check
     - ``make quality``
   * - Build documentation
     - ``make docs``

**Code quality stack:**

- **Formatting:** Black
- **Linting:** Ruff
- **Type checking:** MyPy (strict mode at API boundaries)
- **Package manager:** uv (``uv.lock`` is the single source of truth)

----

Architecture Overview
---------------------

.. code-block:: text

   homodyne/
   ├── core/           # Physics kernel — JIT-compiled JAX
   │   ├── jax_backend.py       # Meshgrid-mode residuals (NLSQ)
   │   ├── physics_cmc.py       # Element-wise residuals (CMC)
   │   ├── physics_utils.py     # Shared: safe_exp, safe_sinc, etc.
   │   ├── models.py            # OO model interface
   │   ├── theory.py            # g1/g2 calculations
   │   └── homodyne_model.py    # Unified model facade
   ├── optimization/
   │   ├── nlsq/                # Trust-region LM (primary)
   │   └── cmc/                 # Consensus Monte Carlo (secondary)
   ├── data/                    # HDF5 loading, validation
   ├── config/                  # YAML config management
   ├── cli/                     # Entry points, shell completion
   └── device/                  # CPU/NUMA detection

**Key design decisions:**

- NLSQ uses ``jax_backend.py`` (meshgrid mode — all (t1, t2) pairs batched).
- CMC uses ``physics_cmc.py`` (element-wise mode — per-shard vectors).
- No shared mutable state across the two paths.
- ``physics_utils.py`` provides shared utilities (``safe_exp``, ``safe_sinc``,
  ``calculate_diffusion_coefficient``).

See :doc:`architecture` for module-level design decisions and :doc:`adrs/index`
for Architecture Decision Records (ADRs).

----

**Version information:**

- Homodyne: |release|
- Python: 3.12+
- JAX: 0.8.2+ (CPU-only)
- Package manager: uv
