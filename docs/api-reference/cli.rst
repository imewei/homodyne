CLI Module
==========

The :mod:`homodyne.cli` module provides command-line interfaces for homodyne scattering analysis, including the main ``homodyne`` command and configuration generation utilities.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Available Commands**:

- ``homodyne``: Main analysis command (NLSQ/MCMC optimization)
- ``homodyne-config``: Interactive configuration generator
- ``homodyne-post-install``: Shell completion setup

**Command Philosophy**:

- YAML-first configuration
- Minimal required arguments
- Verbose/quiet output control
- Integration with optimization backends

Module Contents
---------------

.. automodule:: homodyne.cli
   :members:
   :undoc-members:
   :show-inheritance:

Main Entry Point
----------------

Main command-line interface entry point.

.. automodule:: homodyne.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

Usage
~~~~~

**Basic Analysis**::

    # NLSQ optimization (fast)
    homodyne --config analysis.yaml --method nlsq

    # MCMC inference (Bayesian UQ)
    homodyne --config analysis.yaml --method mcmc

    # Verbose output
    homodyne --config analysis.yaml --verbose

    # Quiet mode (errors only)
    homodyne --config analysis.yaml --quiet

**Visualization**::

    # Plot experimental data
    homodyne --config analysis.yaml --plot-experimental-data

    # Plot results after analysis
    homodyne --config analysis.yaml --plot-results

Argument Parser
---------------

Command-line argument parsing and validation.

.. automodule:: homodyne.cli.args_parser
   :members:
   :undoc-members:
   :show-inheritance:

Arguments
~~~~~~~~~

**Required**:

- ``--config``: Path to YAML configuration file

**Optional**:

- ``--method``: Optimization method (``nlsq`` or ``mcmc``, default from config)
- ``--verbose``: Enable DEBUG logging
- ``--quiet``: Suppress all output except errors
- ``--plot-experimental-data``: Plot input data
- ``--plot-results``: Plot optimization results
- ``--output``: Output directory for results

Validation
~~~~~~~~~~

.. autofunction:: homodyne.cli.args_parser.validate_args

The argument validator checks:

- Configuration file exists and is readable
- Method is valid (nlsq or mcmc)
- Output directory is writable
- No conflicting options (verbose + quiet)

Command Dispatcher
------------------

Command execution and workflow orchestration.

.. automodule:: homodyne.cli.commands
   :members:
   :undoc-members:
   :show-inheritance:

Command Flow
~~~~~~~~~~~~

1. Load configuration from YAML
2. Validate configuration and parameters
3. Load experimental data
4. Run optimization (NLSQ or MCMC)
5. Save results to output directory
6. Generate plots (if requested)

Key Functions
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.cli.commands.dispatch_command
   homodyne.cli.commands.run_nlsq_analysis
   homodyne.cli.commands.run_mcmc_analysis
   homodyne.cli.commands.load_and_validate_config
   homodyne.cli.commands.save_results

NLSQ Workflow
^^^^^^^^^^^^^

::

    from homodyne.cli.commands import run_nlsq_analysis

    # Run NLSQ optimization
    result = run_nlsq_analysis(
        config=config,
        data=experimental_data,
        output_dir="results/"
    )

    # Result contains:
    # - Best-fit parameters
    # - Parameter uncertainties
    # - Chi-squared values
    # - Optimization diagnostics

MCMC Workflow
^^^^^^^^^^^^^

::

    from homodyne.cli.commands import run_mcmc_analysis

    # Run MCMC sampling
    result = run_mcmc_analysis(
        config=config,
        data=experimental_data,
        output_dir="results/"
    )

    # Result contains:
    # - Posterior samples (ArviZ InferenceData)
    # - Summary statistics
    # - Convergence diagnostics
    # - Trace plots

Configuration Generator
-----------------------

Interactive configuration file generator.

.. automodule:: homodyne.cli.config_generator
   :members:
   :undoc-members:
   :show-inheritance:

Usage
~~~~~

**Interactive Mode**::

    homodyne-config --interactive

    # Prompts for:
    # - Analysis mode (static or laminar_flow)
    # - Data file path
    # - Optimization method
    # - Parameter initial values
    # - Output file path

**Template Mode**::

    homodyne-config --mode static --output static_config.yaml

    # Generates configuration from template
    # Available modes: static, laminar_flow

**Validation**::

    homodyne-config --validate my_config.yaml

    # Checks:
    # - YAML syntax
    # - Required fields present
    # - Parameter bounds valid
    # - File paths exist

XLA Configuration
-----------------

JAX/XLA runtime configuration for CPU optimization.

.. automodule:: homodyne.cli.xla_config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

The XLA configuration module sets optimal JAX flags for CPU execution:

- ``XLA_FLAGS``: CPU-specific optimization flags
- ``JAX_PLATFORM_NAME``: Force CPU execution (v2.3.0+)
- ``JAX_ENABLE_X64``: Enable 64-bit precision when needed

**Note**: These are set automatically when using the CLI. For programmatic use, import ``homodyne.cli.xla_config`` before JAX imports.

NLSQ to MCMC Workflow
----------------------

**Recommended Workflow**:

1. **Run NLSQ first** for fast parameter estimation::

       homodyne --config analysis.yaml --method nlsq

2. **Copy best-fit parameters** from NLSQ output::

       Best-fit parameters:
         D0: 1234.5 ± 45.6
         alpha: 0.567 ± 0.012
         D_offset: 78.9 ± 5.3

3. **Update configuration** with NLSQ values as initial parameters::

       optimization:
         initial_parameters:
           values: [1234.5, 0.567, 78.9]

4. **Run MCMC** for uncertainty quantification::

       homodyne --config analysis.yaml --method mcmc

This workflow ensures:

- Fast exploration with NLSQ
- Good initialization for MCMC
- Reduced MCMC warmup time
- Better convergence

Output Structure
----------------

The CLI commands create standardized output directories::

    results/
    ├── config.yaml              # Configuration used
    ├── parameters.json          # Best-fit parameters
    ├── diagnostics.json         # Optimization diagnostics
    ├── plots/
    │   ├── experimental_data.png
    │   ├── fit_comparison.png
    │   └── residuals.png
    └── mcmc/                    # MCMC-specific outputs
        ├── arviz_data.nc        # ArviZ InferenceData
        ├── trace_plots.png
        ├── corner_plot.png
        └── diagnostics.txt

Shell Completion
----------------

The homodyne CLI supports shell completion for bash, zsh, and fish.

**Setup**::

    # Interactive setup
    homodyne-post-install --interactive

    # Manual setup (bash)
    eval "$(homodyne --completion bash)"

    # Manual setup (zsh)
    eval "$(homodyne --completion zsh)"

**Completion Features**:

- Command completion (homodyne, homodyne-config, etc.)
- Option completion (--config, --method, etc.)
- File path completion for arguments
- Method name completion (nlsq, mcmc)

Examples
--------

**Quick Start**::

    # Generate config
    homodyne-config --mode static --output my_config.yaml

    # Edit config with your data path
    vim my_config.yaml

    # Run analysis
    homodyne --config my_config.yaml --method nlsq

**Full Analysis Pipeline**::

    # Validate configuration
    homodyne-config --validate my_config.yaml

    # Plot experimental data to verify loading
    homodyne --config my_config.yaml --plot-experimental-data

    # Run NLSQ optimization
    homodyne --config my_config.yaml --method nlsq --output results/nlsq/

    # Run MCMC with NLSQ initialization
    homodyne --config my_config_mcmc.yaml --method mcmc --output results/mcmc/

    # Plot final results
    homodyne --config my_config_mcmc.yaml --plot-results

**Programmatic Use**::

    from homodyne.cli import main

    # Run CLI programmatically
    import sys
    sys.argv = ['homodyne', '--config', 'analysis.yaml', '--method', 'nlsq']
    main()

See Also
--------

- :mod:`homodyne.config` - Configuration management
- :mod:`homodyne.optimization` - Optimization backends
- :mod:`homodyne.data` - Data loading
