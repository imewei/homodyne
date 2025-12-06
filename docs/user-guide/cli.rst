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

**Common Options:**

.. code-block:: text

   -c, --config FILE              Configuration YAML file (required)
   -m, --method {nlsq,mcmc}       Analysis method (default: nlsq)
   --plot-experimental-data       Display experimental data plots
   --verbose                      Enable debug logging (INFO level)
   --quiet                        Suppress non-error messages
   --help                         Show help message

**Examples:**

Run analysis with NLSQ (fast):

.. code-block:: bash

   homodyne --config my_config.yaml --method nlsq

Run analysis with MCMC (Bayesian inference):

.. code-block:: bash

   homodyne --config my_config.yaml --method mcmc

Run with verbose logging:

.. code-block:: bash

   homodyne --config my_config.yaml --verbose

Display data before analysis:

.. code-block:: bash

   homodyne --config my_config.yaml --plot-experimental-data

Combine options:

.. code-block:: bash

   homodyne --config my_config.yaml --method mcmc --verbose --plot-experimental-data

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

MCMC Method: Bayesian Inference
--------------------------------

Full Bayesian inference using Markov Chain Monte Carlo for uncertainty quantification.

**Use when:**

- You need full uncertainty estimates (posterior distributions)
- You can wait for longer computations (hours to days)
- You want to compare models (model selection)
- You need confidence intervals on derived quantities

**Requirements Before MCMC:**

1. Run NLSQ first to get good initial parameters:

.. code-block:: bash

   homodyne --config config.yaml --method nlsq

2. Copy best-fit parameters from ``results.json`` or console output

3. Update initial parameters in config YAML:

.. code-block:: yaml

   optimization:
     initial_parameters:
       values: [1234.5, 0.567, 123.4]  # From NLSQ results

4. Run MCMC:

.. code-block:: bash

   homodyne --config config.yaml --method mcmc

**Complete NLSQ → MCMC Workflow:**

.. code-block:: bash

   # Step 1: Run NLSQ
   homodyne --config config.yaml --method nlsq

   # Step 2: View results in console or results.json
   cat homodyne_results/results.json

   # Step 3: Update config with best-fit values (manually)
   # Edit config.yaml:
   # optimization:
   #   initial_parameters:
   #     values: [1234.5, 0.567, 123.4]

   # Step 4: Run MCMC
   homodyne --config config.yaml --method mcmc --verbose

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

MCMC inference requires substantial memory:

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

Shell Completion Setup
======================

Enable tab-completion for Homodyne commands:

**For Bash:**

.. code-block:: bash

   eval "$(homodyne-complete bash)"

Add to your ``.bashrc``:

.. code-block:: bash

   eval "$(homodyne-complete bash)"

**For Zsh:**

.. code-block:: bash

   eval "$(homodyne-complete zsh)"

Add to your ``.zshrc``:

.. code-block:: bash

   eval "$(homodyne-complete zsh)"

**For Fish:**

.. code-block:: bash

   homodyne-complete fish | source

Add to your ``config.fish``:

.. code-block:: fish

   homodyne-complete fish | source

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
   homodyne --config config.yaml --method nlsq

   # 4. Copy best-fit parameters to config
   # (Edit config.yaml manually)

   # 5. Run MCMC for posteriors
   homodyne --config config.yaml --method mcmc --verbose

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
   homodyne --config config.yaml              Run analysis (NLSQ)
   homodyne --config config.yaml --method mcmc Run Bayesian inference
   homodyne --help                            Show help

   CONFIGURATION
   =============
   homodyne-config --interactive              Interactive builder
   homodyne-config --mode static              Create template
   homodyne-config --validate config.yaml     Validate config

   DEBUGGING
   =========
   homodyne --config config.yaml --verbose                Enable logging
   homodyne --config config.yaml --plot-experimental-data View data
   JAX_LOG_COMPILES=1 homodyne --config config.yaml      Show JAX compilation

   UTILITIES
   =========
   homodyne --version                         Show version
   homodyne-complete bash                     Enable bash completion
   python -m homodyne.runtime.utils.system_validator --quick  Check system

Next Steps
==========

- :doc:`./configuration` - Detailed configuration reference
- :doc:`./examples` - Real-world analysis examples
- :doc:`./quickstart` - Quick start tutorial
