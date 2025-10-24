homodyne.utils - Utility Functions
===================================

.. automodule:: homodyne.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.utils`` module provides logging, progress tracking, and validation utilities used throughout the homodyne package.

**Key Features:**

* **Structured Logging**: Dual-stream logging (console + file)
* **Progress Tracking**: Real-time optimization progress bars
* **Validation Utilities**: Data and parameter validation helpers

Module Structure
----------------

The utils module is organized into several submodules:

* :mod:`homodyne.utils.logging` - Logging configuration and management
* :mod:`homodyne.utils.progress` - Progress bar and tracking utilities

Submodules
----------

homodyne.utils.logging
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Logging configuration and management with dual-stream output.

**Key Functions:**

* ``setup_logging()`` - Configure logging system
* ``get_logger()`` - Get module-specific logger

**Usage Example:**

.. code-block:: python

   from homodyne.utils.logging import setup_logging, get_logger

   # Setup logging (once at startup)
   setup_logging(
       log_file='./results/logs/analysis.log',
       console_level='INFO',
       file_level='DEBUG'
   )

   # Get logger for module
   logger = get_logger(__name__)

   # Use logger
   logger.info("Starting analysis")
   logger.debug("Detailed debug information")
   logger.warning("Warning message")
   logger.error("Error occurred")

homodyne.utils.progress
~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.utils.progress
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Progress bar and tracking utilities.

**Key Classes:**

* ``ProgressTracker`` - Progress tracking for long operations

**Usage Example:**

.. code-block:: python

   from homodyne.utils.progress import ProgressTracker

   # Create progress tracker
   tracker = ProgressTracker(
       total=100,
       description="Processing batches"
   )

   # Update progress
   for i in range(100):
       # Do work
       result = process_batch(i)

       # Update tracker
       tracker.update(1)

   # Close tracker
   tracker.close()

See Also
--------

* :doc:`../developer-guide/code-quality` - Logging configuration guide
* :doc:`cli` - CLI that uses logging and progress tracking

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.utils import setup_logging, get_logger, ProgressTracker

**Related Functions:**

* :func:`homodyne.cli.commands.run_analysis` - Uses logging
* :func:`homodyne.optimization.fit_nlsq_jax` - Uses progress tracking
