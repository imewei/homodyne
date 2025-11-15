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

XLA Configuration System
========================

Homodyne includes an automatic XLA_FLAGS configuration system that optimizes JAX CPU device allocation for MCMC and NLSQ workflows. This system automatically configures the number of CPU devices available to JAX based on your hardware and analysis mode.

Why XLA Configuration Matters
------------------------------

JAX uses XLA (Accelerated Linear Algebra) to compile and execute numerical computations. The number of CPU devices JAX creates affects:

- **MCMC parallelization**: Multiple chains can run in parallel across different CPU devices
- **Memory usage**: Each device requires memory allocation
- **Optimization performance**: NLSQ doesn't benefit from multiple devices (uses 1 device optimally)

**Performance Impact:**

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Workflow
     - Device Count
     - Hardware
     - Performance
   * - MCMC (4 chains)
     - 4 devices
     - 14-core CPU
     - 1.4x speedup
   * - MCMC (8 chains)
     - 8 devices
     - 36-core HPC
     - 1.8x speedup
   * - NLSQ optimization
     - 1 device
     - Any CPU
     - Optimal (no overhead)
   * - Auto mode
     - 2-8 devices
     - Adapts to CPU
     - Automatic optimization

XLA Configuration Modes
------------------------

The system provides four configuration modes optimized for different hardware and workflows:

.. list-table::
   :header-rows: 1
   :widths: 15 15 50 20

   * - Mode
     - Devices
     - Best For
     - Hardware
   * - **mcmc**
     - 4
     - Multi-core workstations, parallel MCMC chains
     - 8-15 CPU cores
   * - **mcmc-hpc**
     - 8
     - HPC clusters with many CPU cores
     - 36+ CPU cores
   * - **nlsq**
     - 1
     - NLSQ-only workflows, memory-constrained systems
     - Any CPU
   * - **auto**
     - 2-8
     - Automatic detection based on CPU core count
     - Auto-adaptive

**Auto Mode Detection Logic:**

.. code-block:: text

   CPU Cores    →    Devices
   ≤ 7 cores    →    2 devices  (small workstations)
   8-15 cores   →    4 devices  (medium workstations)
   16-35 cores  →    6 devices  (large workstations)
   36+ cores    →    8 devices  (HPC nodes)

Quick XLA Setup
---------------

**Interactive configuration (recommended):**

.. code-block:: bash

   homodyne-post-install --interactive
   # Answer 'Y' to "Configure XLA_FLAGS?"
   # Select mode: mcmc, mcmc-hpc, nlsq, or auto

**One-line configuration:**

.. code-block:: bash

   # Auto-detect optimal device count
   homodyne-post-install --xla-mode auto

   # Configure for MCMC (4 devices)
   homodyne-post-install --xla-mode mcmc

   # Configure for HPC (8 devices)
   homodyne-post-install --xla-mode mcmc-hpc

   # Configure for NLSQ only (1 device)
   homodyne-post-install --xla-mode nlsq

**Manual configuration with homodyne-config-xla:**

.. code-block:: bash

   # Set XLA mode
   homodyne-config-xla --mode auto

   # Show current configuration
   homodyne-config-xla --show

How XLA Configuration Works
----------------------------

The XLA configuration system automatically integrates with your virtual environment activation:

**1. Configuration Storage**

Your selected mode is saved to ``~/.homodyne_xla_mode``:

.. code-block:: bash

   cat ~/.homodyne_xla_mode
   # Output: auto

**2. Automatic Activation**

When you activate your environment, XLA_FLAGS is automatically set:

**Conda/Mamba environments:**

.. code-block:: bash

   conda activate myenv
   # Automatically sources: $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh
   # XLA_FLAGS is set based on ~/.homodyne_xla_mode

   echo $XLA_FLAGS
   # Output: --xla_force_host_platform_device_count=6

**uv/venv/virtualenv:**

.. code-block:: bash

   source venv/bin/activate
   # Automatically sources: $VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash
   # (if added to ~/.bashrc via homodyne-post-install)

   echo $XLA_FLAGS
   # Output: --xla_force_host_platform_device_count=4

**3. JAX Device Detection**

JAX automatically detects the configured devices:

.. code-block:: bash

   python -c "import jax; print('Devices:', jax.devices())"
   # Output: Devices: [CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3)]

Verifying XLA Configuration
----------------------------

**Check current XLA mode:**

.. code-block:: bash

   homodyne-config-xla --show

**Example output:**

.. code-block:: text

   Current XLA Configuration:
     Mode: auto
     XLA_FLAGS: --xla_force_host_platform_device_count=6
     Config file: /home/user/.homodyne_xla_mode
     JAX devices: 6 (cpu)
       [0] TFRT_CPU_0
       [1] TFRT_CPU_1
       [2] TFRT_CPU_2
       [3] TFRT_CPU_3
       [4] TFRT_CPU_4
       [5] TFRT_CPU_5

**Check environment variable directly:**

.. code-block:: bash

   echo $XLA_FLAGS
   # Output: --xla_force_host_platform_device_count=6

**Verify JAX devices in Python:**

.. code-block:: python

   import jax
   print(f"Device count: {len(jax.devices())}")
   print(f"Devices: {jax.devices()}")

Switching XLA Modes
--------------------

You can switch XLA modes at any time and changes take effect on the next environment activation.

**Switch to auto mode:**

.. code-block:: bash

   homodyne-config-xla --mode auto
   # ✓ XLA mode set to: auto
   #   → Auto-detect: 6 devices (detected 20 CPU cores)

   # Reload shell or reactivate environment
   conda deactivate && conda activate myenv

**Switch to NLSQ mode (single device):**

.. code-block:: bash

   homodyne-config-xla --mode nlsq
   # ✓ XLA mode set to: nlsq
   #   → 1 CPU device (NLSQ doesn't need parallelism)

   source ~/.bashrc  # Or reactivate venv

**Switch to HPC mode (8 devices):**

.. code-block:: bash

   homodyne-config-xla --mode mcmc-hpc
   # ✓ XLA mode set to: mcmc-hpc
   #   → 8 CPU devices for HPC clusters (36+ cores)

Advanced XLA Features
---------------------

Manual XLA_FLAGS Override
^^^^^^^^^^^^^^^^^^^^^^^^^

If you need to override the automatic configuration temporarily:

.. code-block:: bash

   # Export XLA_FLAGS before activating environment
   export XLA_FLAGS="--xla_force_host_platform_device_count=2"

   # Activate environment (automatic config is skipped)
   source venv/bin/activate

   # Verify override
   echo $XLA_FLAGS
   # Output: --xla_force_host_platform_device_count=2

The activation scripts respect existing ``XLA_FLAGS`` and never override your manual settings.

Verbose Mode
^^^^^^^^^^^^

Enable verbose output to see XLA configuration details during activation:

.. code-block:: bash

   export HOMODYNE_VERBOSE=1
   source venv/bin/activate

**Output:**

.. code-block:: text

   [homodyne] XLA: auto mode → 6 devices (detected 20 CPU cores)

Environment-Specific Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can set mode per-environment using environment variables:

.. code-block:: bash

   # In environment 1: use NLSQ mode
   export HOMODYNE_XLA_MODE=nlsq
   source env1/bin/activate

   # In environment 2: use auto mode
   export HOMODYNE_XLA_MODE=auto
   source env2/bin/activate

Environment variable ``HOMODYNE_XLA_MODE`` takes precedence over ``~/.homodyne_xla_mode``.

XLA Configuration Best Practices
---------------------------------

**For MCMC workflows:**

.. code-block:: bash

   # Use mcmc mode (4 devices) for typical workstations
   homodyne-config-xla --mode mcmc

   # Use mcmc-hpc mode (8 devices) for HPC clusters (36+ cores)
   homodyne-config-xla --mode mcmc-hpc

   # Run MCMC
   homodyne --method mcmc --config config.yaml

**For NLSQ workflows:**

.. code-block:: bash

   # Use nlsq mode (1 device) for optimal performance
   homodyne-config-xla --mode nlsq

   # Run NLSQ
   homodyne --method nlsq --config config.yaml

**For mixed workflows (NLSQ + MCMC):**

.. code-block:: bash

   # Use auto mode (adapts to hardware)
   homodyne-config-xla --mode auto

   # Or use mcmc mode (slight NLSQ overhead acceptable)
   homodyne-config-xla --mode mcmc

**For HPC batch jobs:**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --nodes=1
   #SBATCH --ntasks-per-node=36

   # Configure for HPC
   homodyne-config-xla --mode mcmc-hpc

   # Activate environment
   source homodyne-env/bin/activate

   # Run analysis
   homodyne --method mcmc --config analysis.yaml

XLA Troubleshooting
-------------------

**XLA_FLAGS not being set:**

.. code-block:: bash

   # Check if activation scripts exist
   ls $VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash  # venv
   ls $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh  # conda

   # If missing, reinstall
   homodyne-post-install --xla-mode auto

**Wrong device count:**

.. code-block:: bash

   # Check current mode
   homodyne-config-xla --show

   # Switch to correct mode
   homodyne-config-xla --mode mcmc

   # Reload environment
   conda deactivate && conda activate myenv

**JAX not detecting devices:**

.. code-block:: bash

   # Verify XLA_FLAGS is set
   echo $XLA_FLAGS
   # Should output: --xla_force_host_platform_device_count=N

   # If empty, check activation
   source $VIRTUAL_ENV/etc/homodyne/activation/xla_config.bash

   # Verify in Python
   python -c "import jax; print(len(jax.devices()))"

**Manual XLA_FLAGS not respected:**

The activation scripts check for existing ``XLA_FLAGS`` first:

.. code-block:: bash

   # Set before activation
   export XLA_FLAGS="--xla_force_host_platform_device_count=2"

   # Then activate
   source venv/bin/activate

   # Verify (should show 2, not auto-configured value)
   echo $XLA_FLAGS

See Also
^^^^^^^^

- :doc:`../advanced-topics/mcmc-uncertainty` - MCMC performance optimization
- :doc:`../advanced-topics/nlsq-optimization` - NLSQ device configuration
- :doc:`cli-usage` - Full CLI reference including ``homodyne-config-xla``

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
