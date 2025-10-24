Command-Line Interface (CLI) Reference
========================================

This guide documents Homodyne's command-line interface for running analyses, plotting results, and managing output.

Basic Usage
-----------

The basic command structure is:

.. code-block:: bash

   homodyne --config <config_file> [OPTIONS]

Running an Analysis
-------------------

**Minimal command:**

.. code-block:: bash

   homodyne --config config.yaml

This:

1. Loads configuration from YAML file
2. Loads experimental data
3. Runs optimization (NLSQ by default)
4. Saves results to ``output/directory`` from config
5. Logs all output to timestamped file

**Override data file:**

.. code-block:: bash

   homodyne --config config.yaml --data-file /path/to/data.hdf

Useful when analyzing multiple datasets with the same configuration.

**Select optimization method:**

.. code-block:: bash

   homodyne --config config.yaml --method nlsq    # Default
   homodyne --config config.yaml --method mcmc    # Uncertainty quantification

Output Control
--------------

**Specify output directory:**

.. code-block:: bash

   homodyne --config config.yaml --output-dir ./results_20251024

Creates timestamped subdirectories for multiple runs.

**Logging Control:**

.. code-block:: bash

   homodyne --config config.yaml --verbose   # DEBUG level
   homodyne --config config.yaml --quiet     # Errors only
   # Default: INFO level

Logging creates two streams:

1. **Console:** Respects --verbose/--quiet flags
2. **File:** Always detailed (DEBUG level) in ``logs/homodyne_analysis_*.log``

Plotting Options
----------------

**Plot experimental data:**

.. code-block:: bash

   homodyne --config config.yaml --plot-experimental-data

Creates visualizations in ``output_dir/plots/``.

**Plotting configuration:**

See :doc:`configuration` for plotting options in YAML config file.

Common Workflows
----------------

**Quick Test:**

.. code-block:: bash

   homodyne --config config.yaml --verbose --output-dir ./test_run

**Batch Processing Multiple Datasets:**

.. code-block:: bash

   for file in data/*.hdf; do
     echo "Processing $file..."
     homodyne --config template.yaml --data-file "$file" --output-dir "./results/$(basename $file .hdf)"
   done

Next Steps
----------

- :doc:`quickstart` - Run your first analysis
- :doc:`examples` - Real-world workflow examples
- :doc:`../advanced-topics/index` - Advanced analysis techniques
- :doc:`shell-completion` - Set up bash/zsh autocompletion

See Also
--------

- :doc:`configuration` - Configuration file documentation
