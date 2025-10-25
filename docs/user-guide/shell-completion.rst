Shell Completion System
=======================

Homodyne provides intelligent shell completion for bash, zsh, and fish shells, making the CLI faster and more discoverable through automatic command suggestions, option completion, and convenient aliases.

.. contents:: On this page
   :local:
   :depth: 2

Why Use Shell Completion?
--------------------------

**Productivity Benefits:**

- ⚡ **Faster typing:** Auto-complete long command names and options with TAB
- 🎯 **Fewer errors:** Avoid typos in file paths and configuration names
- 🔍 **Command discovery:** See available options without reading help text
- 🚀 **Smart suggestions:** Context-aware completion for methods, angles, and values
- ⌨️ **Convenient aliases:** Short commands for common workflows (``hm-nlsq``, ``hc-stat``, etc.)

**Example workflow:**

.. code-block:: bash

   # Without completion:
   homodyne --config /long/path/to/homodyne_static_isotropic_config.yaml --method nlsq --verbose

   # With completion and aliases:
   hm-nlsq --config <TAB>  # Shows config files
   # Select file with arrows, press Enter
   # Result: hm-nlsq --config homodyne_static_isotropic_config.yaml --verbose

Supported Shells
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Shell
     - Status
     - Notes
   * - **Bash**
     - ✅ Supported
     - Requires Bash 4.0+ (macOS users may need to upgrade)
   * - **Zsh**
     - ✅ Supported
     - Recommended for macOS (default shell since Catalina)
   * - **Fish**
     - ✅ Supported
     - Full support with fish-specific activation script
   * - PowerShell
     - ❌ Not supported
     - Windows users: Use WSL2 with bash/zsh

Quick Installation
==================

Interactive Setup (Recommended)
-------------------------------

The easiest way to install shell completion is using the interactive installer:

.. code-block:: bash

   homodyne-post-install --interactive

This walks you through:

1. **Shell detection:** Automatically detects your current shell (bash/zsh/fish)
2. **Shell completion:** Installs completion scripts and aliases
3. **GPU activation** (Linux only): Sets up CUDA environment
4. **Advanced features** (optional): Additional development tools

Just follow the prompts and answer ``y`` or ``n`` for each component.

One-Line Installation
---------------------

For quick setup without prompts:

.. code-block:: bash

   # Install completion only (auto-detects shell)
   homodyne-post-install --shell $(basename $SHELL)

   # Install everything (completion + GPU scripts)
   homodyne-post-install --all

Environment-Specific Installation
==================================

The installation process differs based on your virtual environment type.

Conda/Mamba Environments
-------------------------

**Automatic activation** - No manual configuration needed!

.. code-block:: bash

   # Activate your conda environment
   conda activate myenv

   # Install completion
   homodyne-post-install --shell zsh  # or bash, fish

   # That's it! Completion auto-activates when you activate the environment
   # Try it:
   homodyne --<TAB>
   hm-nlsq --config <TAB>

**How it works:**

- Installs to ``$CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh``
- Conda automatically sources this file when you activate the environment
- No ``.bashrc``/``.zshrc`` modifications needed

**Verify installation:**

.. code-block:: bash

   ls $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh
   # Should show: homodyne-completion.sh

uv/venv/virtualenv Environments
--------------------------------

**Manual activation required** - Add activation script to your shell RC file.

**Step 1: Install completion**

.. code-block:: bash

   # Activate your virtual environment
   source venv/bin/activate  # or: source .venv/bin/activate

   # Install completion
   homodyne-post-install --shell zsh  # or bash, fish

**Step 2: Add activation to shell RC**

**For Bash (``~/.bashrc``):**

.. code-block:: bash

   # Add to ~/.bashrc
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.bashrc
   source ~/.bashrc

**For Zsh (``~/.zshrc``):**

.. code-block:: bash

   # Add to ~/.zshrc
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.zshrc
   exec zsh

**For Fish (``~/.config/fish/config.fish``):**

.. code-block:: fish

   # Add to ~/.config/fish/config.fish
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate.fish' >> ~/.config/fish/config.fish
   # Restart fish shell

**Step 3: Verify completion works**

.. code-block:: bash

   # Activate your venv
   source venv/bin/activate

   # Test completion
   homodyne --<TAB>        # Should show options
   hm-nlsq --config <TAB>  # Should show config files

**Files created:**

.. code-block:: text

   $VIRTUAL_ENV/
   ├── bin/
   │   ├── homodyne-activate          # Bash/Zsh activation script
   │   └── homodyne-activate.fish     # Fish activation script
   └── etc/
       └── zsh/
           └── homodyne-completion.zsh  # Completion logic

Shell-Specific Installation
============================

Bash Completion
---------------

**Requirements:**

- Bash 4.0+ (check with ``bash --version``)
- macOS users: Upgrade Bash if using default 3.2

**Installation:**

.. code-block:: bash

   # Conda/mamba
   homodyne-post-install --shell bash

   # uv/venv/virtualenv - also add to ~/.bashrc:
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.bashrc
   source ~/.bashrc

**Test completion:**

.. code-block:: bash

   homodyne --<TAB>         # Shows: --config, --method, --output-dir, etc.
   homodyne --method <TAB>  # Shows: nlsq, mcmc, auto, nuts, cmc
   homodyne --config <TAB>  # Shows: *.yaml files in current directory

**Available completions:**

- Command options (``--config``, ``--method``, ``--verbose``, etc.)
- File paths for ``--config`` and ``--data-file``
- Method names (``nlsq``, ``mcmc``, ``auto``, ``nuts``, ``cmc``)
- Common output directories
- Phi angle sets
- Contrast/offset values

Zsh Completion
--------------

**Why Zsh?**

- Default shell on macOS since Catalina
- More powerful completion than bash
- Better interactive experience

**Installation:**

.. code-block:: bash

   # Conda/mamba
   homodyne-post-install --shell zsh

   # uv/venv/virtualenv - also add to ~/.zshrc:
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.zshrc
   exec zsh

**Test completion:**

.. code-block:: bash

   homodyne --<TAB>         # Shows options with descriptions
   homodyne --method <TAB>  # Shows: nlsq (fast), mcmc (uncertainty), ...
   hm-<TAB>                 # Shows all hm- aliases

**Zsh-specific features:**

- **Descriptions:** Each option shows a brief description
- **Context-aware:** Suggests different options based on what you've typed
- **Smart caching:** Faster completion for recent config files

Fish Completion
---------------

**Why Fish?**

- User-friendly shell with excellent defaults
- Built-in autosuggestions
- Modern syntax

**Installation:**

.. code-block:: bash

   # Conda/mamba
   homodyne-post-install --shell fish

   # uv/venv/virtualenv - also add to ~/.config/fish/config.fish:
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate.fish' >> ~/.config/fish/config.fish

**Test completion:**

.. code-block:: fish

   homodyne --<TAB>         # Shows options
   homodyne --method <TAB>  # Shows methods
   hm-<TAB>                 # Shows aliases

Aliases Reference
=================

After installing shell completion, you get convenient aliases for all common commands.

Main Commands
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Alias
     - Expands to
     - Purpose
   * - ``hm``
     - ``homodyne``
     - Main analysis command
   * - ``hconfig``
     - ``homodyne-config``
     - Configuration generator

Method Shortcuts (hm- prefix)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Alias
     - Expands to
     - Purpose
   * - ``hm-nlsq``
     - ``homodyne --method nlsq``
     - NLSQ optimization (fast, production)
   * - ``hm-mcmc``
     - ``homodyne --method mcmc``
     - MCMC (auto NUTS/CMC)
   * - ``hm-auto``
     - ``homodyne --method auto``
     - Auto-select NUTS/CMC by dataset size
   * - ``hm-nuts``
     - ``homodyne --method nuts``
     - Standard NUTS MCMC
   * - ``hm-cmc``
     - ``homodyne --method cmc``
     - Consensus Monte Carlo (large datasets)

Config Generation Shortcuts (hc- prefix)
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Alias
     - Expands to
     - Purpose
   * - ``hc-stat``
     - ``homodyne-config --mode static``
     - Generate static isotropic config
   * - ``hc-flow``
     - ``homodyne-config --mode laminar_flow``
     - Generate laminar flow config

Visualization Shortcuts
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Alias
     - Expands to
     - Purpose
   * - ``hexp``
     - ``homodyne --plot-experimental-data``
     - Plot experimental data
   * - ``hsim``
     - ``homodyne --plot-simulated-data``
     - Plot simulated data

Alias Usage Examples
--------------------

**Quick NLSQ analysis:**

.. code-block:: bash

   hm-nlsq --config my_config.yaml

**Generate and validate config:**

.. code-block:: bash

   hc-stat --output test.yaml
   hconfig --validate test.yaml

**MCMC with visualization:**

.. code-block:: bash

   hm-mcmc --config config.yaml --plot-experimental-data

**Large dataset auto-selection:**

.. code-block:: bash

   hm-auto --config large_dataset.yaml --verbose

**Combine aliases with full options:**

.. code-block:: bash

   hm-nlsq --config static.yaml \
           --data-file experiment_002.hdf \
           --output-dir ./results/exp002 \
           --verbose

Advanced Features
=================

Smart Context-Aware Completion
-------------------------------

The completion system provides intelligent suggestions based on context:

**Method completion:**

.. code-block:: bash

   homodyne --method <TAB>
   # Shows: nlsq (fast), mcmc (uncertainty), auto (adaptive), nuts, cmc

**Config file completion:**

.. code-block:: bash

   homodyne --config <TAB>
   # Prioritizes recent .yaml files, shows common config patterns

**Common value suggestions:**

.. code-block:: bash

   homodyne --contrast <TAB>    # Shows: 0.0, 0.5, 1.0, 1.5, 2.0
   homodyne --phi-angles <TAB>  # Shows: 0,45,90,135  0,36,72,108,144

**Output directory suggestions:**

.. code-block:: bash

   homodyne --output-dir <TAB>
   # Shows: ./results, ./output, ./homodyne_results, ./analysis

Completion Caching
------------------

The completion system caches recent config files for faster suggestions:

**Cache location:**

.. code-block:: bash

   ${XDG_CACHE_HOME:-$HOME/.cache}/homodyne/completion_cache

**Cache refresh:**

- Automatically refreshes every 5 minutes
- Finds ``.yaml`` files in current and parent directories
- Includes commonly used config names

**Manual cache clear:**

.. code-block:: bash

   rm -f ${XDG_CACHE_HOME:-$HOME/.cache}/homodyne/completion_cache

Interactive Command Builder
----------------------------

For complex commands, use the interactive builder:

.. code-block:: bash

   homodyne_build

This guides you through:

1. Method selection (nlsq/mcmc/auto/nuts/cmc)
2. Config file selection (shows recent files)
3. Additional options (output directory, verbose mode, etc.)

Then generates and optionally runs the command.

Troubleshooting
===============

Completion Not Working
----------------------

**Verify installation:**

.. code-block:: bash

   # Check which command
   which homodyne  # Should show: /path/to/venv/bin/homodyne

   # Check version
   homodyne --version  # Should show version number

**Check environment type:**

.. code-block:: bash

   # Conda/mamba
   echo $CONDA_PREFIX  # Should show conda environment path

   # uv/venv/virtualenv
   echo $VIRTUAL_ENV  # Should show virtual environment path

**Verify completion files exist:**

.. code-block:: bash

   # Conda
   ls $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh

   # uv/venv/virtualenv
   ls $VIRTUAL_ENV/etc/zsh/homodyne-completion.zsh
   ls $VIRTUAL_ENV/bin/homodyne-activate

**For venv/uv/virtualenv, verify RC file:**

.. code-block:: bash

   # Bash
   grep "homodyne-activate" ~/.bashrc

   # Zsh
   grep "homodyne-activate" ~/.zshrc

   # Fish
   grep "homodyne-activate" ~/.config/fish/config.fish

**Reload shell:**

.. code-block:: bash

   source ~/.bashrc  # Bash
   exec zsh          # Zsh
   # Or restart terminal

No Suggestions Appearing
------------------------

**For Bash - Check version:**

.. code-block:: bash

   bash --version  # Must be 4.0+
   # macOS ships with 3.2 - upgrade with Homebrew:
   brew install bash

**For Zsh - Verify completion loaded:**

.. code-block:: bash

   # Check if completion functions exist
   which _homodyne_advanced_zsh  # Should show: _homodyne_advanced_zsh ()

**For all shells - Try manual activation:**

.. code-block:: bash

   # Conda
   source $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh

   # uv/venv/virtualenv
   source $VIRTUAL_ENV/bin/homodyne-activate

   # Test immediately
   homodyne --<TAB>

Aliases Not Available
---------------------

**Verify completion is loaded:**

.. code-block:: bash

   # Check if alias exists
   alias hm  # Should show: alias hm='homodyne'

**If aliases missing, completion may not be loaded:**

.. code-block:: bash

   # Manually source completion
   source $VIRTUAL_ENV/bin/homodyne-activate

   # Check again
   alias hm

**For persistent fix, ensure RC file has activation:**

.. code-block:: bash

   # Add to ~/.bashrc or ~/.zshrc
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.bashrc
   source ~/.bashrc

Wrong Shell Detected
--------------------

**Force specific shell during installation:**

.. code-block:: bash

   # Force bash (even if current shell is zsh)
   homodyne-post-install --shell bash

   # Force zsh
   homodyne-post-install --shell zsh

   # Force fish
   homodyne-post-install --shell fish

Reinstalling Completion
------------------------

**Force reinstall (overwrites existing):**

.. code-block:: bash

   homodyne-post-install --shell zsh --force

**Complete removal and reinstall:**

.. code-block:: bash

   # Remove existing completion
   homodyne-cleanup --completion

   # Reinstall
   homodyne-post-install --interactive

Uninstalling Completion
========================

**Preview what will be removed:**

.. code-block:: bash

   homodyne-cleanup --dry-run

**Interactive removal:**

.. code-block:: bash

   homodyne-cleanup --interactive
   # Select: 1. Shell completion
   # Confirm: y

**Complete removal:**

.. code-block:: bash

   homodyne-cleanup --all

**Completion only:**

.. code-block:: bash

   homodyne-cleanup --completion

**Don't forget to remove from RC files:**

.. code-block:: bash

   # Edit ~/.bashrc, ~/.zshrc, or ~/.config/fish/config.fish
   # Remove line: source $VIRTUAL_ENV/bin/homodyne-activate

See :doc:`cli-usage` for details on the ``homodyne-cleanup`` command.

Platform-Specific Notes
=======================

macOS
-----

**Default Bash version issue:**

- macOS ships with Bash 3.2 (too old for completion)
- Upgrade with Homebrew: ``brew install bash``
- Or use Zsh (default since Catalina)

**Zsh recommended:**

.. code-block:: bash

   # Install completion for zsh
   homodyne-post-install --shell zsh

   # Add to ~/.zshrc
   echo 'source $VIRTUAL_ENV/bin/homodyne-activate' >> ~/.zshrc
   exec zsh

Linux
-----

**Most distributions ship with Bash 4.0+:**

.. code-block:: bash

   homodyne-post-install --shell bash

**Conda users - no RC file modification needed:**

Completion auto-activates with environment.

Windows (WSL2)
--------------

**Use Windows Subsystem for Linux:**

.. code-block:: bash

   # In WSL2 terminal (Ubuntu/Debian)
   homodyne-post-install --shell bash

**Native PowerShell not supported.**

Next Steps
==========

**Start using completion:**

- :doc:`cli-usage` - Full CLI command reference
- :doc:`quickstart` - Run your first analysis with aliases
- :doc:`examples` - Real-world workflow examples

**Explore aliases:**

Try these quick commands:

.. code-block:: bash

   hc-stat --output my_config.yaml  # Generate config
   hconfig --validate my_config.yaml  # Validate it
   hm-nlsq --config my_config.yaml --verbose  # Run analysis

See Also
========

- `homodyne-post-install reference <cli-usage.html#homodyne-post-install-command>`_
- `homodyne-cleanup reference <cli-usage.html#homodyne-cleanup-command>`_
- :doc:`../developer-guide/contributing` - Contributing to completion scripts
