Command-Line Interface
======================

Homodyne provides a comprehensive command-line interface for running analyses and managing configurations.
This guide covers all available commands and options.

homodyne Command
================

The main command for running XPCS analyses.

**Syntax:**

.. code-block:: bash

   homodyne [OPTIONS]

**Core Options:**

.. code-block:: text

   --help                         Show help message
   --version                      Show version information
   --method {nlsq,cmc}            Optimization method (default: nlsq)
   --config PATH                  Configuration file, YAML (default: ./homodyne_config.yaml)
   --output-dir PATH              Output directory (default: ./homodyne_results)
   --data-file PATH               Experimental data file, HDF5 (overrides config)
   --output-format {yaml,json,npz} Output format (default: yaml)
   --verbose                      Enable verbose logging
   --quiet                        Suppress all output except errors

**Analysis Mode Options (mutually exclusive):**

.. code-block:: text

   --static-mode                  Force static analysis (3 parameters)
   --laminar-flow                 Force laminar flow analysis (7 parameters)

**NLSQ Options:**

.. code-block:: text

   --max-iterations INT           Maximum iterations (default: 10000)
   --tolerance FLOAT              Convergence tolerance (default: 1e-8)

**CMC Options:**

.. code-block:: text

   --n-samples INT                Samples per chain (default: from config or 1000)
   --n-warmup INT                 Warmup samples (default: from config or 500)
   --n-chains INT                 Number of chains (default: from config or 4)
   --cmc-num-shards INT           Data shards for CMC (default: auto-detect)
   --cmc-backend {auto,pjit,multiprocessing,pbs}  Parallel backend
   --nlsq-result PATH             Pre-computed NLSQ results directory for warm-start
   --no-nlsq-warmstart            Disable automatic NLSQ warm-start (NOT RECOMMENDED)
   --dense-mass-matrix            Use dense mass matrix for NUTS/CMC

**Parameter Override Options:**

.. code-block:: text

   --initial-d0 FLOAT             Override D0 (nm²/s)
   --initial-alpha FLOAT          Override alpha (diffusion exponent)
   --initial-d-offset FLOAT       Override D_offset (nm²/s)
   --initial-gamma-dot-t0 FLOAT   Override gamma_dot_t0 (s⁻¹, laminar flow)
   --initial-beta FLOAT           Override beta (shear exponent, laminar flow)
   --initial-gamma-dot-offset FLOAT Override gamma_dot_offset (s⁻¹, laminar flow)
   --initial-phi0 FLOAT           Override phi0 (radians, laminar flow)

**Plotting Options:**

.. code-block:: text

   --save-plots                   Save result plots to output directory
   --plot-experimental-data       Generate data validation plots
   --plot-simulated-data          Plot theoretical C₂ heatmaps (no data required)
   --plotting-backend {auto,matplotlib,datashader}  Backend (default: auto)
   --parallel-plots               Generate plots in parallel (requires Datashader)
   --contrast FLOAT               Contrast for simulated data (default: 0.3)
   --offset FLOAT                 Offset for simulated data (default: 1.0)
   --phi-angles STRING            Comma-separated phi angles in degrees

**Examples:**

Run analysis with NLSQ (fast):

.. code-block:: bash

   homodyne --config my_config.yaml --method nlsq

Run analysis with CMC (Bayesian inference):

.. code-block:: bash

   homodyne --config my_config.yaml --method cmc

Run with verbose logging:

.. code-block:: bash

   homodyne --config my_config.yaml --verbose

Display data before analysis:

.. code-block:: bash

   homodyne --config my_config.yaml --plot-experimental-data

Combine options:

.. code-block:: bash

   homodyne --config my_config.yaml --method cmc --verbose --plot-experimental-data

NLSQ Method: Non-Linear Least Squares
--------------------------------------

Fast, deterministic optimization for point estimates.

**Use when:**

- You need fast results (seconds to minutes)
- You want point estimates without uncertainty
- You're exploring the parameter space
- You have good initial guesses

**Example:**

.. code-block:: bash

   homodyne --config config.yaml --method nlsq

**Output:**

- Best-fit parameters
- Convergence metrics
- Residual analysis
- Visualization plots

**Typical Runtime:**

- Small datasets (1K angles, 100 times): ~10 seconds
- Medium datasets (5K angles, 500 times): ~1 minute
- Large datasets (10K angles, 1000 times): ~5 minutes

CMC Method: Bayesian Inference
-------------------------------

Full Bayesian inference using Consensus Monte Carlo (CMC) for uncertainty quantification.

**Use when:**

- You need full uncertainty estimates (posterior distributions)
- You can wait for longer computations (hours to days)
- You want to compare models (model selection)
- You need confidence intervals on derived quantities

**Recommended NLSQ → CMC Workflow:**

.. code-block:: bash

   # Step 1: Run NLSQ to get point estimates
   homodyne --config config.yaml --method nlsq --output-dir results/

   # Step 2: Run CMC with pre-computed NLSQ warm-start (RECOMMENDED)
   homodyne --config config.yaml --method cmc --nlsq-result results/

The ``--nlsq-result`` flag loads parameters from ``results/nlsq/parameters.json``,
avoiding redundant NLSQ computation and ensuring consistent warm-start values.
This reduces CMC divergence from ~28% to <5%.

.. note::

   NLSQ warm-start is automatic when using ``--method cmc`` without ``--nlsq-result``.
   Use ``--no-nlsq-warmstart`` to disable (NOT RECOMMENDED).

**Output:**

- Posterior distributions for all parameters
- Credible intervals (confidence bounds)
- Convergence diagnostics
- Trace plots and posterior plots

**Typical Runtime:**

- Small datasets: 1-2 hours
- Medium datasets: 4-8 hours
- Large datasets: 12-24+ hours

**Memory Requirements:**

CMC inference requires substantial memory:

- 4 GB minimum
- 8+ GB recommended
- 16+ GB for large datasets

Debugging Options
-----------------

**Enable Debug Logging:**

.. code-block:: bash

   homodyne --config config.yaml --verbose

This shows:

- Configuration loading details
- Data preprocessing steps
- Optimization iterations
- Memory usage
- Convergence diagnostics

**Inspect Experimental Data:**

.. code-block:: bash

   homodyne --config config.yaml --plot-experimental-data

This displays:

- Two-time correlation data (c₂ heatmaps)
- Angle-dependent behavior
- Data quality assessment
- Intensity distribution

**Suppress Output:**

.. code-block:: bash

   homodyne --config config.yaml --quiet

Useful for batch processing or script execution.

homodyne-config Command
=======================

Manage and validate Homodyne configurations.

**Syntax:**

.. code-block:: bash

   homodyne-config [COMMAND] [OPTIONS]

**Commands:**

Interactive Configuration Builder
----------------------------------

Create configurations interactively:

.. code-block:: bash

   homodyne-config --interactive

Prompts for:

1. Analysis mode (static, laminar_flow)
2. Number of angles
3. Data file path and HDF5 keys
4. Initial parameter guesses
5. Parameter bounds
6. Output directory

**Example Session:**

.. code-block:: text

   Homodyne Configuration Builder
   ==============================

   Select analysis mode:
   1. static (time-dependent diffusion)
   2. laminar_flow (with shear rate)
   > 1

   Number of azimuthal angles: 3

   Data file path: /data/xpcs_sample.h5

   Enter HDF5 key for t1 [default: entry/data/t1]:
   Enter HDF5 key for t2 [default: entry/data/t2]:
   Enter HDF5 key for phi [default: entry/data/phi]:
   Enter HDF5 key for c2 [default: entry/data/c2]:

   D0 initial guess [1000.0]: 1500
   alpha initial guess [0.5]: 0.6
   D_offset initial guess [100.0]:

   Configuration saved to: my_config.yaml

Template Mode
-------------

Create configuration from template:

.. code-block:: bash

   homodyne-config --mode static

or:

.. code-block:: bash

   homodyne-config --mode laminar_flow

This generates a YAML template with all options and defaults for the selected mode.

Validation Mode
---------------

Validate an existing configuration:

.. code-block:: bash

   homodyne-config --validate my_config.yaml

Checks:

- YAML syntax validity
- Required fields present
- Parameter bounds logical
- Data file accessible
- Initial parameters in bounds

**Example Output:**

.. code-block:: text

   Validating: my_config.yaml
   ✓ YAML syntax valid
   ✓ Required fields present
   ✓ Analysis mode valid: static
   ✓ Parameters in bounds
   ✓ Data file exists: /data/xpcs_sample.h5
   ✓ HDF5 keys accessible

   Validation: PASSED

**Output:** Creates validated config with warnings/errors reported.

homodyne-config-xla Command
===========================

Configure XLA device settings for different workflows.

**Syntax:**

.. code-block:: bash

   homodyne-config-xla [OPTIONS]

**Options:**

.. code-block:: text

   --mode {cmc,cmc-hpc,nlsq,auto}  Set XLA mode
   --show                           Show current XLA configuration

**Modes:**

.. list-table::
   :header-rows: 1

   * - Mode
     - Devices
     - Use Case
   * - ``cmc``
     - 4
     - Multi-core workstations
   * - ``cmc-hpc``
     - 8
     - HPC with 36+ cores
   * - ``nlsq``
     - 1
     - NLSQ-only workflows
   * - ``auto``
     - Varies
     - Auto-detect based on CPU cores

**Examples:**

.. code-block:: bash

   homodyne-config-xla --mode cmc      # 4 devices for CMC
   homodyne-config-xla --mode auto     # Auto-detect
   homodyne-config-xla --show          # Show current config

homodyne-post-install Command
=============================

Set up shell completion and environment configuration.

**Options:**

.. code-block:: text

   -i, --interactive    Interactive setup
   --shell {bash,zsh,fish}  Specify shell type
   --xla-mode {cmc,cmc-hpc,nlsq,auto}  Configure XLA mode
   --advanced           Install advanced features (caching, validation)
   -f, --force          Force setup even if not in virtual environment

**Example:**

.. code-block:: bash

   homodyne-post-install --interactive
   homodyne-post-install --shell bash --xla-mode auto

homodyne-cleanup Command
========================

Remove Homodyne shell completion and configuration files.

**Options:**

.. code-block:: text

   -i, --interactive    Interactive cleanup
   -n, --dry-run        Show what would be removed without removing
   -f, --force          Skip confirmation prompt

**Example:**

.. code-block:: bash

   homodyne-cleanup --dry-run     # Preview changes
   homodyne-cleanup --force       # Remove all files

Shell Completion & Aliases
==========================

Homodyne provides context-aware tab completion for all commands and a set of
shell aliases for common operations.

**Setup via post-install (recommended):**

.. code-block:: bash

   homodyne-post-install --interactive

**Manual setup (Bash):**

Add to your ``.bashrc``:

.. code-block:: bash

   source "$(python -c 'import homodyne; print(homodyne.__path__[0])')/runtime/shell/completion.sh"

**Manual setup (Zsh):**

Add to your ``.zshrc``:

.. code-block:: bash

   source "$(python -c 'import homodyne; print(homodyne.__path__[0])')/runtime/shell/completion.sh"

Shell Aliases
-------------

Once completion is sourced, the following aliases are available:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Alias
     - Expands To
     - Purpose
   * - ``hm``
     - ``homodyne``
     - Base command
   * - ``hconfig``
     - ``homodyne-config``
     - Configuration generator
   * - ``hm-nlsq``
     - ``homodyne --method nlsq``
     - NLSQ optimization
   * - ``hm-cmc``
     - ``homodyne --method cmc``
     - Consensus Monte Carlo
   * - ``hc-stat``
     - ``homodyne-config --mode static``
     - Generate static config
   * - ``hc-flow``
     - ``homodyne-config --mode laminar_flow``
     - Generate laminar flow config
   * - ``hexp``
     - ``homodyne --plot-experimental-data``
     - Plot experimental data
   * - ``hsim``
     - ``homodyne --plot-simulated-data``
     - Plot simulated data
   * - ``hxla``
     - ``homodyne-config-xla``
     - XLA configuration
   * - ``hsetup``
     - ``homodyne-post-install``
     - Post-install setup
   * - ``hclean``
     - ``homodyne-cleanup``
     - Cleanup shell files

**Interactive functions:**

- ``homodyne_build`` — Interactive command builder (guides method and config selection)
- ``homodyne_help`` — Display all aliases and shortcuts

Common Workflows
================

Workflow 1: Basic Static Mode Analysis
--------------------------------------

.. code-block:: bash

   # 1. Create config
   homodyne-config --mode static

   # 2. Edit config.yaml with your data path and parameters

   # 3. Validate
   homodyne-config --validate config.yaml

   # 4. Run analysis
   homodyne --config config.yaml --method nlsq

Workflow 2: Laminar Flow with Uncertainty
------------------------------------------

.. code-block:: bash

   # 1. Create template
   homodyne-config --mode laminar_flow

   # 2. Configure parameters (7 physical parameters)

   # 3. Run NLSQ for initial estimate
   homodyne --config config.yaml --method nlsq --output-dir results/

   # 4. Run CMC with NLSQ warm-start (automatic parameter loading)
   homodyne --config config.yaml --method cmc --nlsq-result results/ --verbose

Workflow 3: Parameter Space Exploration
----------------------------------------

.. code-block:: bash

   # Run with different initial guesses
   for guess in 1000 1500 2000; do
     # Create config with different initial D0
     sed "s/D0:.*/D0: $guess/" config_template.yaml > config_$guess.yaml

     # Run analysis
     homodyne --config config_$guess.yaml --method nlsq

     # Compare results
     tail -5 homodyne_results_$guess/results.json
   done

Environment Variables
=====================

**JAX Configuration:**

.. code-block:: bash

   # Enable JAX compilation logging
   JAX_LOG_COMPILES=1 homodyne --config config.yaml

   # Set number of CPU threads
   JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 homodyne --config config.yaml

**Homodyne Configuration:**

.. code-block:: bash

   # Set output directory
   HOMODYNE_OUTPUT_DIR=/custom/output/path homodyne --config config.yaml

   # Enable debug mode
   HOMODYNE_DEBUG=1 homodyne --config config.yaml

Exit Codes
==========

Homodyne returns standard exit codes:

.. code-block:: text

   0   - Success: Analysis completed successfully
   1   - Error: Configuration or data error
   2   - Error: Optimization failed to converge
   3   - Error: MCMC inference failed
   4   - Error: I/O or file system error
   255 - Error: Unexpected internal error

Use in scripts:

.. code-block:: bash

   homodyne --config config.yaml
   if [ $? -eq 0 ]; then
     echo "Analysis succeeded"
   else
     echo "Analysis failed with code $?"
   fi

Performance Tips
================

**Speed Optimization:**

.. code-block:: bash

   # Use multiple CPU cores
   export JAX_PLATFORMS=cpu
   export OMP_NUM_THREADS=8
   homodyne --config config.yaml --method nlsq

**Memory Optimization:**

.. code-block:: bash

   # Reduce number of angles
   # Downsample time series in config
   # Reduce batch size (if available in config)

Troubleshooting
===============

**Issue: "Configuration file not found"**

.. code-block:: bash

   homodyne --config /absolute/path/to/config.yaml

**Issue: "Unknown command 'homodyne'"**

Solution: Reinstall or activate environment:

.. code-block:: bash

   pip install homodyne
   # or
   source venv/bin/activate

**Issue: "Validation failed"**

.. code-block:: bash

   homodyne-config --validate config.yaml

Check reported errors and fix YAML syntax or data paths.

**Issue: "JAX compilation taking too long"**

This is normal for first run. JAX caches compiled functions.
Subsequent runs will be faster.

**Issue: "Out of memory during NLSQ"**

Reduce dataset size:

.. code-block:: yaml

   data:
     n_angles_select: 3          # Use fewer angles
     t1_slice: ":100"            # Use fewer times

Quick Reference Card
====================

.. code-block:: text

   BASIC COMMANDS
   ==============
   homodyne --config config.yaml                Run analysis (NLSQ default)
   homodyne --config config.yaml --method cmc   Run Bayesian inference
   homodyne --method cmc --nlsq-result results/ CMC with NLSQ warm-start
   homodyne --help                              Show help
   homodyne --version                           Show version

   CONFIGURATION
   =============
   homodyne-config --interactive                Interactive builder
   homodyne-config --mode static                Create static template
   homodyne-config --mode laminar_flow          Create laminar flow template
   homodyne-config --validate config.yaml       Validate config
   homodyne-config-xla --mode auto              Configure XLA devices
   homodyne-config-xla --show                   Show current XLA config

   DEBUGGING
   =========
   homodyne --config config.yaml --verbose                Enable logging
   homodyne --config config.yaml --plot-experimental-data View data
   homodyne --plot-simulated-data --contrast 0.5          Simulated heatmaps
   JAX_LOG_COMPILES=1 homodyne --config config.yaml      Show JAX compilation

   SETUP & CLEANUP
   ===============
   homodyne-post-install --interactive          Setup shell completion
   homodyne-cleanup --dry-run                   Preview cleanup
   homodyne-cleanup --force                     Remove shell files

   SHELL ALIASES (after sourcing completion.sh)
   ============================================
   hm            homodyne
   hm-nlsq       homodyne --method nlsq
   hm-cmc        homodyne --method cmc
   hconfig       homodyne-config
   hc-stat       homodyne-config --mode static
   hc-flow       homodyne-config --mode laminar_flow
   hexp          homodyne --plot-experimental-data
   hsim          homodyne --plot-simulated-data
   hxla          homodyne-config-xla
   hsetup        homodyne-post-install
   hclean        homodyne-cleanup

Next Steps
==========

- :doc:`./configuration` - Detailed configuration reference
- :doc:`./examples` - Real-world analysis examples
- :doc:`./quickstart` - Quick start tutorial
