Command-Line Interface (CLI) Reference
========================================

Homodyne provides four command-line tools for XPCS analysis, configuration management, and shell completion setup.

Command Summary
---------------

+---------------------------+--------------------------------------------------+
| Command                   | Purpose                                          |
+===========================+==================================================+
| ``homodyne``              | Run XPCS analysis (NLSQ/MCMC optimization)       |
+---------------------------+--------------------------------------------------+
| ``homodyne-config``       | Generate and validate configuration files        |
+---------------------------+--------------------------------------------------+
| ``homodyne-post-install`` | Install shell completion and GPU activation      |
+---------------------------+--------------------------------------------------+
| ``homodyne-cleanup``      | Remove shell completion and activation scripts   |
+---------------------------+--------------------------------------------------+

.. _homodyne-command:

homodyne: Main Analysis Command
================================

The ``homodyne`` command runs XPCS analysis with NLSQ or MCMC optimization.

Basic Usage
-----------

.. code-block:: bash

   homodyne --config <config_file> [OPTIONS]

Quick Examples
--------------

**Minimal NLSQ analysis:**

.. code-block:: bash

   homodyne --config config.yaml

**MCMC for uncertainty quantification:**

.. code-block:: bash

   homodyne --config config.yaml --method mcmc

**Override data file:**

.. code-block:: bash

   homodyne --config config.yaml --data-file /path/to/experiment.hdf

**With visualization:**

.. code-block:: bash

   homodyne --config config.yaml --plot-experimental-data

Complete Options
----------------

**Required:**

- ``--config PATH`` - Path to YAML configuration file

**Optimization Method:**

- ``--method {nlsq,mcmc,auto,nuts,cmc}`` - Analysis method (default: nlsq)

  - ``nlsq`` - Trust-region nonlinear least squares (fast, production)
  - ``mcmc`` - Alias for ``auto`` (NUTS <500k points, CMC >500k)
  - ``auto`` - Auto-select NUTS or CMC based on dataset size
  - ``nuts`` - Standard NUTS MCMC sampling
  - ``cmc`` - Consensus Monte Carlo for large datasets (>500k points)

**Data Override:**

- ``--data-file PATH`` - Override experimental data file path

**Output Control:**

- ``--output-dir PATH`` - Override output directory
- ``--verbose`` - Enable DEBUG logging to console
- ``--quiet`` - Show errors only (no INFO messages)

**Visualization:**

- ``--plot-experimental-data`` - Generate plots after analysis

**Hardware:**

- ``--force-cpu`` - Force CPU-only computation (disable GPU)

Common Workflows
----------------

**Quick Test Run:**

.. code-block:: bash

   homodyne --config config.yaml --verbose --output-dir ./test_run

**Batch Processing Multiple Datasets:**

.. code-block:: bash

   for file in data/*.hdf; do
     echo "Processing $file..."
     homodyne --config template.yaml \
              --data-file "$file" \
              --output-dir "./results/$(basename $file .hdf)"
   done

**Production Analysis with Visualization:**

.. code-block:: bash

   homodyne --config production_config.yaml \
            --method nlsq \
            --plot-experimental-data \
            --output-dir ./results_final

**Large Dataset Analysis (Auto CMC):**

.. code-block:: bash

   # Automatically uses CMC for datasets > 500k points
   homodyne --config large_dataset.yaml --method auto

.. _homodyne-config-command:

homodyne-config: Configuration Generator
========================================

The ``homodyne-config`` command generates, builds, and validates YAML configuration files.

Basic Usage
-----------

.. code-block:: bash

   homodyne-config --mode <mode> --output <file>
   homodyne-config --interactive
   homodyne-config --validate <file>

Modes
-----

- ``static`` - Generic static diffusion analysis (3 physical parameters)
- ``laminar_flow`` - Flow analysis with shear dynamics (7 physical parameters)

Generate Configuration
----------------------

**From template:**

.. code-block:: bash

   # Static mode
   homodyne-config --mode static --output my_config.yaml

   # Laminar flow mode
   homodyne-config --mode laminar_flow --output flow_config.yaml

**Interactive builder:**

.. code-block:: bash

   homodyne-config --interactive

The interactive builder walks you through:

1. Mode selection (static/laminar_flow)
2. Sample information (name, experiment ID)
3. Data file path
4. Output directory
5. Configuration file name

**Validate existing configuration:**

.. code-block:: bash

   homodyne-config --validate my_config.yaml

Validation checks:

- YAML syntax correctness
- Required sections present
- Parameter names match selected mode
- Data file existence (warning if missing)

Complete Options
----------------

**Mode selection:**

- ``--mode {static,laminar_flow}`` or ``-m`` - Configuration mode

**Output:**

- ``--output PATH`` or ``-o`` - Output configuration file path (default: homodyne_{mode}_config.yaml)

**Interactive:**

- ``--interactive`` or ``-i`` - Interactive configuration builder

**Validation:**

- ``--validate PATH`` or ``-v`` - Validate existing configuration file

**Other:**

- ``--force`` or ``-f`` - Force overwrite existing file
- ``--help`` or ``-h`` - Show help message

Examples
--------

**Quick static configuration:**

.. code-block:: bash

   homodyne-config --mode static --output static_analysis.yaml

**Interactive flow configuration:**

.. code-block:: bash

   homodyne-config --interactive
   # Select: 2. laminar_flow
   # Follow prompts for sample info, data file, output directory

**Validate before running:**

.. code-block:: bash

   homodyne-config --validate my_config.yaml
   # Check output for warnings/errors
   homodyne --config my_config.yaml  # Run if valid

**Force overwrite:**

.. code-block:: bash

   homodyne-config --mode static --output config.yaml --force

.. _homodyne-post-install-command:

homodyne-post-install: Shell Completion Installer
==================================================

The ``homodyne-post-install`` command installs shell completion (bash/zsh/fish) and GPU activation scripts.

Basic Usage
-----------

.. code-block:: bash

   homodyne-post-install [OPTIONS]

Interactive Installation (Recommended)
---------------------------------------

.. code-block:: bash

   homodyne-post-install --interactive

Walks you through:

1. Shell completion (bash/zsh/fish)
2. GPU activation scripts (Linux only)
3. Advanced features (optional)

Quick Installation
------------------

**Install everything (shell completion + GPU scripts):**

.. code-block:: bash

   homodyne-post-install --all

**Shell completion only:**

.. code-block:: bash

   homodyne-post-install --shell zsh  # or bash, fish

**GPU activation only (Linux):**

.. code-block:: bash

   homodyne-post-install --gpu

Environment Detection
---------------------

The installer automatically detects your environment:

**Conda/Mamba:**

- Installs completion to ``$CONDA_PREFIX/etc/conda/activate.d/``
- **Auto-activates** when you activate the environment
- No manual sourcing needed

**uv/venv/virtualenv:**

- Installs completion to ``$VIRTUAL_ENV/etc/zsh/``
- Creates activation scripts:

  - ``$VIRTUAL_ENV/bin/homodyne-activate`` (bash/zsh)
  - ``$VIRTUAL_ENV/bin/homodyne-activate.fish`` (fish)

- **Requires manual activation** (see :ref:`activation-scripts`)

Complete Options
----------------

**Interactive:**

- ``--interactive`` or ``-i`` - Interactive installation wizard

**Component selection:**

- ``--shell {bash,zsh,fish}`` - Install shell completion for specific shell
- ``--gpu`` - Install GPU activation scripts (Linux only)
- ``--advanced`` - Install advanced features
- ``--all`` - Install all components

**Other:**

- ``--force`` or ``-f`` - Force reinstall (overwrite existing)
- ``--help`` or ``-h`` - Show help message

.. _activation-scripts:

Activation Scripts (uv/venv/virtualenv)
---------------------------------------

For non-conda environments, add to your shell RC file:

**Bash/Zsh:**

.. code-block:: bash

   # Add to ~/.bashrc or ~/.zshrc
   source $VIRTUAL_ENV/bin/homodyne-activate

**Fish:**

.. code-block:: fish

   # Add to ~/.config/fish/config.fish
   source $VIRTUAL_ENV/bin/homodyne-activate.fish

After adding, reload your shell:

.. code-block:: bash

   source ~/.bashrc  # or ~/.zshrc for zsh
   # or just restart your terminal

Examples
--------

**Quick setup for zsh:**

.. code-block:: bash

   # Install completion
   homodyne-post-install --shell zsh

   # If using venv/uv/virtualenv, add activation to ~/.zshrc
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.zshrc
   source ~/.zshrc

**Complete setup (interactive):**

.. code-block:: bash

   homodyne-post-install --interactive
   # Follow prompts for shell type, GPU setup, etc.

**Force reinstall (fix broken completion):**

.. code-block:: bash

   homodyne-post-install --shell bash --force

.. _homodyne-cleanup-command:

homodyne-cleanup: Uninstaller
==============================

The ``homodyne-cleanup`` command removes shell completion scripts and activation files.

Basic Usage
-----------

.. code-block:: bash

   homodyne-cleanup [OPTIONS]

Interactive Cleanup (Recommended)
----------------------------------

.. code-block:: bash

   homodyne-cleanup --interactive

Prompts you to select what to remove:

1. Shell completion scripts (bash/zsh/fish)
2. GPU activation scripts
3. Advanced features and activation scripts

Dry Run (Preview)
-----------------

.. code-block:: bash

   homodyne-cleanup --dry-run

Shows what would be removed without actually removing anything.

Complete Cleanup
----------------

.. code-block:: bash

   homodyne-cleanup --all  # Remove everything

Complete Options
----------------

**Cleanup modes:**

- ``--interactive`` or ``-i`` - Interactive cleanup wizard
- ``--dry-run`` - Preview cleanup without removing files
- ``--all`` - Remove all installed components

**Scope:**

- ``--completion`` - Remove shell completion only
- ``--gpu`` - Remove GPU activation scripts only
- ``--advanced`` - Remove advanced features only

**Other:**

- ``--help`` or ``-h`` - Show help message

What Gets Removed
-----------------

**Shell completion:**

- ``$VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh``
- ``$CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh`` (conda)
- ``$VIRTUAL_ENV/bin/homodyne-activate`` (venv/uv/virtualenv)
- ``$VIRTUAL_ENV/bin/homodyne-activate.fish``

**GPU activation:**

- ``$VIRTUAL_ENV/etc/conda/activate.d/homodyne-gpu-activate.sh``

Examples
--------

**Preview cleanup:**

.. code-block:: bash

   homodyne-cleanup --dry-run

**Remove shell completion only:**

.. code-block:: bash

   homodyne-cleanup --completion

**Complete removal (interactive):**

.. code-block:: bash

   homodyne-cleanup --interactive
   # Select components to remove
   # Confirm with 'y'

**Force remove everything:**

.. code-block:: bash

   homodyne-cleanup --all

.. _shell-aliases:

Shell Aliases
=============

After installing shell completion, you get convenient aliases for common tasks.

Main Command Aliases
--------------------

+-------------+--------------------------------+
| Alias       | Expands to                     |
+=============+================================+
| ``hm``      | ``homodyne``                   |
+-------------+--------------------------------+
| ``hconfig`` | ``homodyne-config``            |
+-------------+--------------------------------+

Method Aliases (hm- prefix)
---------------------------

+-------------+--------------------------------+
| Alias       | Expands to                     |
+=============+================================+
| ``hm-nlsq`` | ``homodyne --method nlsq``     |
+-------------+--------------------------------+
| ``hm-mcmc`` | ``homodyne --method mcmc``     |
+-------------+--------------------------------+
| ``hm-auto`` | ``homodyne --method auto``     |
+-------------+--------------------------------+
| ``hm-nuts`` | ``homodyne --method nuts``     |
+-------------+--------------------------------+
| ``hm-cmc``  | ``homodyne --method cmc``      |
+-------------+--------------------------------+

Config Mode Aliases (hc- prefix)
---------------------------------

+-------------+-----------------------------------------------+
| Alias       | Expands to                                    |
+=============+===============================================+
| ``hc-stat`` | ``homodyne-config --mode static``             |
+-------------+-----------------------------------------------+
| ``hc-flow`` | ``homodyne-config --mode laminar_flow``       |
+-------------+-----------------------------------------------+

Visualization Aliases
---------------------

+-------------+-----------------------------------------------+
| Alias       | Expands to                                    |
+=============+===============================================+
| ``hexp``    | ``homodyne --plot-experimental-data``         |
+-------------+-----------------------------------------------+
| ``hsim``    | ``homodyne --plot-simulated-data``            |
+-------------+-----------------------------------------------+

Alias Examples
--------------

**Quick NLSQ analysis:**

.. code-block:: bash

   hm-nlsq --config my_config.yaml

**Generate static config:**

.. code-block:: bash

   hc-stat --output static.yaml

**MCMC with visualization:**

.. code-block:: bash

   hm-mcmc --config config.yaml --plot-experimental-data

**Combine aliases and flags:**

.. code-block:: bash

   hm-auto --config large_dataset.yaml --verbose --output-dir ./results

Logging and Output
==================

All ``homodyne`` commands create **dual logging streams**:

Console Output
--------------

Controlled by ``--verbose`` and ``--quiet`` flags:

- **Normal (default):** INFO level
- ``--verbose``: DEBUG level (detailed progress)
- ``--quiet``: Errors only

File Logging
------------

Always detailed (DEBUG level) regardless of console flags:

**Log location:**

.. code-block:: text

   {output_dir}/logs/homodyne_analysis_YYYYMMDD_HHMMSS.log

**Example log path:**

.. code-block:: text

   ./results/logs/homodyne_analysis_20251024_143022.log

This ensures complete audit trail for troubleshooting even when using ``--quiet``.

Exit Codes
==========

All commands return standard exit codes:

- ``0`` - Success
- ``1`` - Error (configuration, data loading, optimization failure)
- ``2`` - Invalid arguments

Use in scripts:

.. code-block:: bash

   if homodyne --config config.yaml; then
     echo "Analysis completed successfully"
   else
     echo "Analysis failed with exit code $?"
     exit 1
   fi

Next Steps
==========

- :doc:`shell-completion` - Set up shell completion
- :doc:`quickstart` - Run your first analysis
- :doc:`examples` - Real-world workflow examples
- :doc:`configuration` - Configuration file documentation

See Also
========

- :doc:`../advanced-topics/nlsq-optimization` - NLSQ optimization guide
- :doc:`../advanced-topics/mcmc-uncertainty` - MCMC uncertainty quantification
- :doc:`../api-reference/cli` - CLI API reference
