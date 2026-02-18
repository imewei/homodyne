.. _api-cli:

================
homodyne CLI
================

Homodyne provides five command-line entry points installed into the active
virtual environment.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Command
     - Purpose
   * - ``homodyne``
     - Main analysis entry point (NLSQ/CMC)
   * - ``homodyne-config``
     - Configuration file generation, interactive builder, and validation
   * - ``homodyne-config-xla``
     - XLA device count configuration per workflow mode
   * - ``homodyne-post-install``
     - Shell completion and alias installation
   * - ``homodyne-cleanup``
     - Remove shell completion files

----

homodyne
--------

.. _api-cli-homodyne:

The main analysis command. Reads a YAML config file, loads the HDF5 dataset,
runs NLSQ and/or CMC fitting, and writes results to the output directory.

.. code-block:: text

   usage: homodyne [--config CONFIG] [--method {nlsq,cmc}]
                   [--output-dir DIR] [--nlsq-result DIR]
                   [--static-mode | --laminar-flow]
                   [--plot-experimental-data | --plot-simulated-data]
                   [--contrast FLOAT] [--offset FLOAT]
                   [--phi-angles ANGLES]
                   [--verbose | --quiet]
                   [--version]

Arguments
~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``--config FILE``
     - Path to YAML configuration file (default: ``homodyne_config.yaml``)
   * - ``--method {nlsq,cmc}``
     - Optimisation method. ``nlsq`` = trust-region NLSQ (default); ``cmc`` = Consensus Monte Carlo
   * - ``--output-dir DIR``
     - Output directory for results (default: ``./results``)
   * - ``--nlsq-result DIR``
     - Load pre-computed NLSQ warm-start from ``DIR/nlsq/parameters.json``
   * - ``--static-mode``
     - Force static analysis mode (3 parameters: D₀, α, D_offset)
   * - ``--laminar-flow``
     - Force laminar-flow mode (7 parameters)
   * - ``--plot-experimental-data``
     - Generate data validation plots without fitting
   * - ``--plot-simulated-data``
     - Plot theoretical heatmaps from config parameters
   * - ``--contrast FLOAT``
     - Override contrast for simulated data plots
   * - ``--offset FLOAT``
     - Override offset for simulated data plots
   * - ``--phi-angles ANGLES``
     - Comma-separated phi angles for simulated plots (e.g., ``"0,45,90,135"``)
   * - ``--verbose / -v``
     - Enable verbose (DEBUG-level) logging
   * - ``--quiet / -q``
     - Suppress all output except errors
   * - ``--version``
     - Print version and exit

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: bash

   # Run NLSQ with default config
   homodyne

   # Run NLSQ explicitly
   homodyne --method nlsq --config config.yaml --output-dir results/

   # Run CMC with NLSQ warm-start (recommended two-step workflow)
   homodyne --method nlsq --config config.yaml --output-dir results/
   homodyne --method cmc  --config config.yaml \
            --nlsq-result results/ \
            --output-dir results/

   # Force static mode
   homodyne --static-mode --method nlsq

   # Force laminar flow with CMC
   homodyne --laminar-flow --method cmc --config config.yaml

   # Validate data by plotting
   homodyne --plot-experimental-data --config config.yaml

   # Plot theoretical heatmaps
   homodyne --plot-simulated-data --contrast 0.5 --offset 1.05 \
            --phi-angles "0,45,90,135"

Shell Aliases
~~~~~~~~~~~~~

.. code-block:: bash

   hm          # homodyne
   hm-nlsq     # homodyne --method nlsq
   hm-cmc      # homodyne --method cmc
   hexp        # homodyne --plot-experimental-data
   hsim        # homodyne --plot-simulated-data

----

homodyne-config
---------------

Generates, validates, and interactively builds YAML configuration files.

.. code-block:: text

   usage: homodyne-config [-m MODE] [-o OUTPUT] [-i] [-v CONFIG] [-f]

Arguments
~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Argument
     - Description
   * - ``-m / --mode MODE``
     - Template mode: ``static`` or ``laminar_flow``
   * - ``-o / --output FILE``
     - Output file path (default: ``homodyne_config.yaml``)
   * - ``-i / --interactive``
     - Launch interactive configuration builder
   * - ``-v / --validate FILE``
     - Validate an existing configuration file
   * - ``-f / --force``
     - Overwrite existing output file without prompting

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate static config template
   homodyne-config --mode static --output static_config.yaml

   # Generate laminar flow config
   homodyne-config --mode laminar_flow --output flow_config.yaml

   # Interactive builder
   homodyne-config --interactive

   # Validate existing config
   homodyne-config --validate my_config.yaml

Shell Aliases
~~~~~~~~~~~~~

.. code-block:: bash

   hconfig     # homodyne-config
   hc-stat     # homodyne-config --mode static
   hc-flow     # homodyne-config --mode laminar_flow

----

homodyne-config-xla
-------------------

Configures the number of XLA virtual CPU devices used by JAX. The setting
is persisted to ``~/.homodyne_xla_mode`` and read by the shell activation
scripts.

.. code-block:: text

   usage: homodyne-config-xla [--mode {cmc,cmc-hpc,nlsq,auto}] [--show]

Arguments
~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Argument
     - Description
   * - ``--mode MODE``
     - XLA mode: ``cmc`` (4 devices), ``cmc-hpc`` (8 devices), ``nlsq`` (1 device), ``auto`` (detect)
   * - ``--show``
     - Show current XLA configuration

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Mode
     - Devices
     - Use Case
   * - ``cmc``
     - 4
     - Standard CMC on workstations
   * - ``cmc-hpc``
     - 8
     - HPC clusters with 36+ cores
   * - ``nlsq``
     - 1
     - NLSQ-only workflows
   * - ``auto``
     - Detected
     - Automatic based on physical CPU core count

Usage Examples
~~~~~~~~~~~~~~

.. code-block:: bash

   # Configure for CMC on a workstation
   homodyne-config-xla --mode cmc

   # Configure for HPC
   homodyne-config-xla --mode cmc-hpc

   # Show current setting
   homodyne-config-xla --show

Shell Alias
~~~~~~~~~~~

.. code-block:: bash

   hxla    # homodyne-config-xla

----

homodyne-post-install
---------------------

Copies the shell completion and alias system into the active virtual environment.
Run once after installation to enable tab completion and aliases.

.. code-block:: text

   usage: homodyne-post-install [-i] [--shell {bash,zsh,fish}]
                                [--xla-mode MODE] [--advanced] [-f]

Arguments
~~~~~~~~~

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Argument
     - Description
   * - ``-i / --install``
     - Perform full installation into the active venv
   * - ``--shell {bash,zsh,fish}``
     - Target shell (default: auto-detect)
   * - ``--xla-mode MODE``
     - Set XLA mode during installation (default: ``cmc``)
   * - ``--advanced``
     - Install the full interactive configuration builder
   * - ``-f / --force``
     - Overwrite existing completion files

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Install completions for the current shell
   homodyne-post-install -i

   # Install with HPC XLA preset
   homodyne-post-install -i --xla-mode cmc-hpc

   # Force reinstall for zsh
   homodyne-post-install -i --shell zsh --force

Shell Alias
~~~~~~~~~~~

.. code-block:: bash

   hsetup    # homodyne-post-install

----

homodyne-cleanup
----------------

Removes shell completion files installed by ``homodyne-post-install``.

.. code-block:: text

   usage: homodyne-cleanup [-i] [-n] [-f]

Arguments
~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Argument
     - Description
   * - ``-i / --install``
     - Actually remove files (safety flag; without it, performs a dry run)
   * - ``-n / --dry-run``
     - List files that would be removed without deleting them
   * - ``-f / --force``
     - Skip confirmation prompt

Usage Example
~~~~~~~~~~~~~

.. code-block:: bash

   # Dry run — see what would be removed
   homodyne-cleanup --dry-run

   # Actually remove completion files
   homodyne-cleanup -i

Shell Alias
~~~~~~~~~~~

.. code-block:: bash

   hclean    # homodyne-cleanup

----

Shell Completion System
-----------------------

After running ``homodyne-post-install``, the virtual environment activation
script sources the appropriate completion file per shell:

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Shell
     - Sourced File
     - Features
   * - bash/zsh
     - ``$VENV/etc/homodyne/shell/completion.sh``
     - Tab completion + aliases + interactive builder
   * - fish
     - ``$VENV/etc/homodyne/shell/completion.fish``
     - Native fish ``complete`` + aliases
   * - fallback
     - ``$VENV/etc/zsh/homodyne-completion.zsh``
     - Aliases only (no tab completion)

**Full alias reference:**

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Alias
     - Expands To
     - Purpose
   * - ``hm``
     - ``homodyne``
     - Main analysis
   * - ``hconfig``
     - ``homodyne-config``
     - Config management
   * - ``hm-nlsq``
     - ``homodyne --method nlsq``
     - Force NLSQ
   * - ``hm-cmc``
     - ``homodyne --method cmc``
     - Force CMC
   * - ``hc-stat``
     - ``homodyne-config --mode static``
     - Static config template
   * - ``hc-flow``
     - ``homodyne-config --mode laminar_flow``
     - Flow config template
   * - ``hexp``
     - ``homodyne --plot-experimental-data``
     - Data validation plots
   * - ``hsim``
     - ``homodyne --plot-simulated-data``
     - Theory plots
   * - ``hxla``
     - ``homodyne-config-xla``
     - XLA configuration
   * - ``hsetup``
     - ``homodyne-post-install``
     - Shell setup
   * - ``hclean``
     - ``homodyne-cleanup``
     - Remove completion files

Key source files:

- ``runtime/shell/completion.sh`` — source of truth for bash/zsh completion
- ``post_install.py`` — installer logic
- ``uninstall_scripts.py`` — cleanup logic

----

.. automodule:: homodyne.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: homodyne.cli.args_parser
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: homodyne.cli.config_generator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: homodyne.cli.xla_config
   :members:
   :undoc-members:
   :show-inheritance:
