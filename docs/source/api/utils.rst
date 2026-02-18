.. _api-utils:

=================
Utilities
=================

This page documents the cross-cutting utility modules used throughout the
Homodyne package: structured logging, CPU/NUMA detection, and path validation.

----

homodyne.utils.logging
-----------------------

.. _api-logging:

Lightweight structured logging system built on Python's standard ``logging``
module. Provides contextual log prefixes, configurable handlers (console +
rotating file), and helpers for performance monitoring and call tracing.

Key exports:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Symbol
     - Purpose
   * - ``get_logger(name)``
     - Returns a logger (or context-aware ``LoggerAdapter``) for ``name``
   * - ``log_phase(name)``
     - Context manager that logs entry/exit and elapsed time for a phase
   * - ``log_performance(threshold)``
     - Decorator that logs execution time when it exceeds ``threshold`` seconds
   * - ``log_calls``
     - Decorator that logs every function call with arguments
   * - ``log_exception(logger, exc)``
     - Logs exception with full traceback at ERROR level
   * - ``with_context(**kv)``
     - Context manager that attaches key–value context to all log messages
   * - ``LogConfiguration``
     - Dataclass for configuring the global logging setup from CLI args

.. automodule:: homodyne.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
~~~~~~~~~~~~~~

Basic logger
^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import get_logger

   logger = get_logger(__name__)

   logger.info("Starting analysis")
   logger.debug("Parameter D0=%.4g", d0)
   logger.warning("High divergence rate: %.1f%%", rate * 100)
   logger.error("Solver failed after %d attempts", n_attempts)

Phase timing
^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import get_logger, log_phase

   logger = get_logger(__name__)

   with log_phase("NLSQ optimisation"):
       result = fit_nlsq_jax(data, config)
   # Logs: "NLSQ optimisation complete in 12.34 s"

Performance decorator
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import log_performance

   @log_performance(threshold=0.5)   # log if call takes > 0.5 s
   def compute_jacobian(params):
       ...

Contextual logging (CMC shard)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import get_logger, with_context

   logger = get_logger(__name__)

   with with_context(shard_id=42, n_pts=5000):
       logger.info("Starting NUTS sampling")
       # Logs: "[shard_id=42 n_pts=5000] Starting NUTS sampling"

Exception logging
^^^^^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import get_logger, log_exception

   logger = get_logger(__name__)

   try:
       result = risky_operation()
   except Exception as exc:
       log_exception(logger, exc)
       raise

CLI log configuration
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.utils.logging import LogConfiguration

   # Configured from --verbose / --quiet flags
   log_cfg = LogConfiguration.from_cli_args(args)
   log_cfg.apply()

----

homodyne.device
---------------

CPU architecture detection and JAX/XLA configuration utilities for HPC
environments. Detects physical and logical core counts, NUMA topology,
and processor architecture (Intel/AMD) to inform thread and device allocation.

.. _api-device:

detect\_cpu\_info
~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.device.cpu.detect_cpu_info

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.device.cpu import detect_cpu_info

   info = detect_cpu_info()
   print(f"Physical cores: {info['physical_cores']}")
   print(f"Logical cores:  {info['logical_cores']}")
   print(f"Architecture:   {info['architecture']}")
   print(f"NUMA nodes:     {info['numa_nodes']}")
   print(f"Processor:      {info['processor']}")

   # Optimal worker count for CMC multiprocessing backend
   n_workers = max(1, info['physical_cores'] // 2 - 1)

.. automodule:: homodyne.device.cpu
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: detect_cpu_info

----

homodyne.utils.path\_validation
---------------------------------

Path validation utilities for checking file and directory existence with
informative error messages. Used at CLI entry points and data loading
boundaries to provide clear diagnostics.

.. automodule:: homodyne.utils.path_validation
   :members:
   :undoc-members:
   :show-inheritance:
