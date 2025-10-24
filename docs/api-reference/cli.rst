homodyne.cli - Command-Line Interface
=====================================

.. automodule:: homodyne.cli
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.cli`` module provides the command-line interface for running XPCS analyses. It handles argument parsing, configuration loading, workflow orchestration, result saving, and plotting.

**Key Features:**

* **Configuration-Driven**: YAML-based workflow configuration
* **CLI Overrides**: Command-line arguments override configuration
* **Workflow Orchestration**: Complete analysis pipeline automation
* **Result Management**: Comprehensive output saving (JSON, NPZ, HDF5)
* **Plotting**: Automated visualization of results

Module Structure
----------------

The CLI module is organized into several submodules:

* :mod:`homodyne.cli.main` - Main entry point
* :mod:`homodyne.cli.args_parser` - Argument parsing
* :mod:`homodyne.cli.commands` - Command implementations

Submodules
----------

homodyne.cli.main
~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.cli.main
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Main CLI entry point.

**Key Functions:**

* ``main()`` - Main CLI entry point

**Usage:**

.. code-block:: bash

   # Direct invocation
   homodyne --config config.yaml

   # Python module invocation
   python -m homodyne.cli --config config.yaml

homodyne.cli.args_parser
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.cli.args_parser
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Command-line argument parsing.

**Key Functions:**

* ``create_parser()`` - Create argument parser
* ``parse_args()`` - Parse command-line arguments

**Supported Arguments:**

.. code-block:: text

   --config PATH              Configuration file path (required)
   --data-file PATH           Override data file path
   --output-dir PATH          Override output directory
   --method {nlsq,mcmc}       Optimization method
   --plot-experimental-data   Plot experimental data
   --verbose                  Enable verbose logging
   --quiet                    Suppress output
   --version                  Show version

homodyne.cli.commands
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Command implementations for analysis workflows.

**Key Functions:**

* ``run_analysis()`` - Main analysis workflow
* ``load_config()`` - Configuration loading
* ``load_data()`` - Data loading from HDF5
* ``run_optimization()`` - NLSQ or MCMC optimization
* ``save_results()`` - Save analysis results
* ``plot_results()`` - Generate plots

CLI Workflows
-------------

**Basic Workflow:**

.. code-block:: bash

   # 1. Run NLSQ optimization (default)
   homodyne --config config.yaml

   # Output:
   # - results/nlsq/parameters.json
   # - results/nlsq/fitted_data.npz
   # - results/nlsq/analysis_results_nlsq.json
   # - results/nlsq/convergence_metrics.json

**MCMC Workflow:**

.. code-block:: bash

   # 2. Run MCMC sampling
   homodyne --config config.yaml --method mcmc

   # Output:
   # - results/mcmc/posterior_samples.hdf5
   # - results/mcmc/diagnostics.json
   # - results/mcmc/summary_statistics.json

**With Plotting:**

.. code-block:: bash

   # 3. Run with plots
   homodyne --config config.yaml --plot-experimental-data

   # Output (additional):
   # - results/plots/experimental_g2.png
   # - results/plots/fitted_comparison.png
   # - results/plots/residuals.png

Configuration Overrides
-----------------------

**Override Data File:**

.. code-block:: bash

   homodyne --config config.yaml --data-file /path/to/other_data.hdf

**Override Output Directory:**

.. code-block:: bash

   homodyne --config config.yaml --output-dir ./custom_results

**Override Method:**

.. code-block:: bash

   homodyne --config config.yaml --method mcmc

Logging
-------

**Log Files:**

All analysis runs create timestamped log files:

.. code-block:: text

   results/logs/homodyne_analysis_YYYYMMDD_HHMMSS.log

**Dual Logging:**

* **Console**: Respects ``--verbose`` and ``--quiet`` flags
* **File**: Always contains full DEBUG-level logs

**Logging Levels:**

.. code-block:: bash

   # Normal (INFO level)
   homodyne --config config.yaml

   # Verbose (DEBUG level)
   homodyne --config config.yaml --verbose

   # Quiet (ERROR level only)
   homodyne --config config.yaml --quiet

Result Structure
----------------

**NLSQ Output:**

.. code-block:: text

   results/
   ├── nlsq/
   │   ├── parameters.json              # Parameter values + uncertainties
   │   ├── fitted_data.npz               # 10 arrays (experimental + theoretical + residuals)
   │   ├── analysis_results_nlsq.json    # Fit quality + metadata
   │   └── convergence_metrics.json      # Convergence info + recovery
   ├── plots/
   │   ├── experimental_g2.png
   │   ├── fitted_comparison.png
   │   └── residuals.png
   └── logs/
       └── homodyne_analysis_YYYYMMDD_HHMMSS.log

**MCMC Output:**

.. code-block:: text

   results/
   ├── mcmc/
   │   ├── posterior_samples.hdf5        # Full posterior samples
   │   ├── diagnostics.json              # R-hat, ESS, divergences
   │   └── summary_statistics.json       # Mean, std, credible intervals
   ├── plots/
   │   ├── posterior_distributions.png
   │   ├── trace_plots.png
   │   └── corner_plot.png
   └── logs/
       └── homodyne_analysis_YYYYMMDD_HHMMSS.log

Programmatic Usage
------------------

**Python API:**

.. code-block:: python

   from homodyne.cli.commands import run_analysis

   # Run analysis programmatically
   result = run_analysis(
       config_path='config.yaml',
       output_dir='./results',
       method='nlsq',
       plot_results=True
   )

   # Access results
   print(f"Optimal parameters: {result['parameters']}")
   print(f"Chi-squared: {result['chi_squared']}")

**Load Configuration:**

.. code-block:: python

   from homodyne.cli.commands import load_config

   # Load and validate configuration
   config_dict = load_config('config.yaml')

   # Access configuration sections
   data_config = config_dict['experimental_data']
   opt_config = config_dict['optimization']

Error Handling
--------------

**Configuration Errors:**

.. code-block:: bash

   $ homodyne --config invalid.yaml
   ERROR: Configuration validation failed:
     - Missing required field: experimental_data.file_path
     - Invalid analysis_type: must be 'static_isotropic' or 'laminar_flow'

**Data Loading Errors:**

.. code-block:: bash

   $ homodyne --config config.yaml --data-file nonexistent.hdf
   ERROR: Data file not found: nonexistent.hdf

**Optimization Errors:**

.. code-block:: bash

   $ homodyne --config config.yaml
   WARNING: NLSQ convergence failed on attempt 1/3
   INFO: Applying convergence recovery strategy
   INFO: NLSQ optimization successful on attempt 2/3

Shell Completion
----------------

**Bash Completion:**

.. code-block:: bash

   # Enable bash completion
   source examples/setup_shell_completion.sh

   # Use tab completion
   homodyne --config <TAB>
   homodyne --method <TAB>

Examples
--------

**Static Isotropic Analysis:**

.. code-block:: bash

   homodyne --config homodyne/config/templates/homodyne_static_isotropic.yaml \\
            --output-dir ./results_static \\
            --plot-experimental-data

**Laminar Flow Analysis:**

.. code-block:: bash

   homodyne --config homodyne/config/templates/homodyne_laminar_flow.yaml \\
            --method nlsq \\
            --output-dir ./results_laminar

**MCMC Uncertainty Quantification:**

.. code-block:: bash

   # First run NLSQ
   homodyne --config config.yaml --method nlsq --output-dir ./results

   # Then run MCMC initialized from NLSQ
   homodyne --config config.yaml --method mcmc --output-dir ./results

See Also
--------

* :doc:`../user-guide/cli-usage` - CLI usage guide
* :doc:`../user-guide/configuration` - Configuration guide
* :doc:`config` - Configuration module
* :doc:`../user-guide/quickstart` - Result interpretation

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.cli import main
   from homodyne.cli.commands import run_analysis, load_config, load_data

**Related Functions:**

* :func:`homodyne.config.manager.ConfigManager` - Configuration loading
* :func:`homodyne.data.xpcs_loader.XPCSDataLoader` - Data loading
* :func:`homodyne.optimization.fit_nlsq_jax` - NLSQ optimization
* :func:`homodyne.optimization.fit_mcmc_jax` - MCMC sampling
