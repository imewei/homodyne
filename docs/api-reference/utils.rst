Utils Module
============

The :mod:`homodyne.utils` module provides essential utility functions for logging and path validation.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Utility Categories**:

- **Logging**: Structured logging with performance tracking
- **Path Validation**: Safe file/directory path handling

Module Contents
---------------

.. automodule:: homodyne.utils
   :noindex:

Logging
-------

Structured logging utilities with performance tracking.

.. automodule:: homodyne.utils.logging
   :members:
   :undoc-members:
   :show-inheritance:

Logger Setup
~~~~~~~~~~~~

.. autofunction:: homodyne.utils.logging.get_logger

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import get_logger

    logger = get_logger(__name__)

    logger.info("Starting analysis")
    logger.debug("Debug information")
    logger.warning("Warning message")
    logger.error("Error occurred")

**Log Levels**:

- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages
- **WARNING**: Warning messages (e.g., deprecated features)
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

Performance Logging
~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.utils.logging.log_performance

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import log_performance

    @log_performance
    def expensive_computation(data):
        # Automatically logs execution time
        result = process(data)
        return result

    # Logs: "expensive_computation completed in 2.34s"

Call Logging
~~~~~~~~~~~~

.. autofunction:: homodyne.utils.logging.log_calls

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import log_calls

    @log_calls
    def my_function(x, y, z=10):
        # Logs function calls with arguments
        return x + y + z

    # Logs: "Calling my_function(x=5, y=3, z=10)"

Operation Logging
~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.utils.logging.log_operation

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import log_operation

    with log_operation("Loading experimental data"):
        data = load_data("experiment.h5")
        # Logs start and completion with timing

    # Logs:
    # "Starting: Loading experimental data"
    # "Completed: Loading experimental data (1.23s)"

Path Validation
---------------

Safe file and directory path validation.

.. automodule:: homodyne.utils.path_validation
   :members:
   :undoc-members:
   :show-inheritance:

Save Path Validation
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.utils.path_validation.validate_save_path

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import validate_save_path

    # Validate output file path
    output_path = validate_save_path(
        "results/output.json",
        expected_ext=".json",
        create_dirs=True
    )

    # Creates 'results/' directory if needed
    # Raises PathValidationError if invalid

**Validation Checks**:

- File extension matches expected
- Parent directory exists or can be created
- Path is writable
- No path traversal attempts

Plot Save Path
~~~~~~~~~~~~~~

.. autofunction:: homodyne.utils.path_validation.validate_plot_save_path

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import validate_plot_save_path

    # Validate plot output path
    plot_path = validate_plot_save_path(
        "plots/figure.png",
        create_dirs=True
    )

    # Supports: .png, .pdf, .svg, .eps, .jpg

Safe Output Directory
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: homodyne.utils.path_validation.get_safe_output_dir

Usage Example
^^^^^^^^^^^^^

::

    from homodyne.utils import get_safe_output_dir

    # Get or create safe output directory
    output_dir = get_safe_output_dir(
        "results/2024-12-06/",
        create_if_missing=True
    )

    # Returns absolute path to directory
    # Creates directory structure if needed

Path Validation Exception
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: homodyne.utils.path_validation.PathValidationError

Raised when path validation fails due to:

- Invalid file extension
- Unwritable path
- Path traversal attempt
- Parent directory does not exist (and create_dirs=False)

Best Practices
--------------

**Logging**:

1. Use module-level logger::

       logger = get_logger(__name__)

2. Use appropriate log levels:

   - DEBUG: Detailed internal state
   - INFO: User-facing progress messages
   - WARNING: Deprecated features, non-critical issues
   - ERROR: Errors that prevent operation completion

3. Log performance for operations > 1 second::

       @log_performance
       def slow_operation():
           ...

**Path Validation**:

1. Always validate user-provided paths::

       output_path = validate_save_path(user_path)

2. Use ``create_dirs=True`` for output paths::

       validate_save_path(path, create_dirs=True)

3. Validate file extensions for safety::

       validate_plot_save_path(path)  # Only allows image formats

Error Handling
--------------

**Validation Errors**:

All validation functions raise standard Python exceptions:

- ``ValueError``: Invalid parameter values
- ``TypeError``: Wrong parameter types
- ``PathValidationError``: Path validation failures

**Example Error Handling**::

    from homodyne.utils import validate_save_path, PathValidationError

    try:
        output_path = validate_save_path("output.json")
    except PathValidationError as e:
        logger.error(f"Invalid output path: {e}")
        # Handle error or use default path
        output_path = "default_output.json"

**Logging Errors**::

    try:
        result = risky_operation()
    except Exception as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        # exc_info=True includes traceback in log

Testing Utilities
-----------------

Utilities are designed for easy testing:

**Mock Logging**::

    from unittest.mock import patch

    with patch('homodyne.utils.logging.get_logger') as mock_logger:
        # Test code that uses logger
        pass

**Temporary Paths**::

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test.json"
        validated = validate_save_path(str(test_path), create_dirs=True)

See Also
--------

- :mod:`homodyne.config` - Uses validation utilities
- :mod:`homodyne.optimization` - Uses logging and validation
- :mod:`homodyne.data` - Uses path validation
