Quickstart
==========

Get up and running with Homodyne in 5 minutes. This guide walks you through your first
XPCS analysis with a minimal example.

Prerequisites
=============

- Python 3.12+ installed
- Homodyne installed (see :doc:`./installation`)
- A sample XPCS HDF5 data file

Installation Check
------------------

Verify your installation:

.. code-block:: bash

   homodyne --version

You should see ``Homodyne.4.1``.

5-Minute Analysis Workflow
==========================

Step 1: Create a Configuration File
-----------------------------------

Create a file named ``my_config.yaml`` with the following content:

.. code-block:: yaml

   # Minimal static mode configuration
   data:
     path: /path/to/your/xpcs_data.h5          # Update this path!
     h5_keys:
       t1: "entry/data/t1"
       t2: "entry/data/t2"
       phi: "entry/data/phi"
       c2: "entry/data/c2"

   analysis:
     mode: static                                # Two options: static, laminar_flow
     n_angles: 3                                 # Number of azimuthal angles

   optimization:
     method: nlsq                                # Non-linear least squares
     initial_parameters:
       values: [1000.0, 0.5, 100.0]            # [D0, alpha, D_offset]
     parameter_bounds:
       D0: [100.0, 10000.0]
       alpha: [0.0, 1.0]
       D_offset: [0.0, 500.0]

   output:
     results_dir: ./homodyne_results

**Key Configuration Points:**

- ``data.path``: Update to your HDF5 file location
- ``data.h5_keys``: Keys to your data arrays in the HDF5 file
- ``analysis.mode``: Use ``static`` for this example (3 parameters)
- ``optimization.method``: Use ``nlsq`` for fast optimization

Step 2: Run the Analysis
------------------------

Run Homodyne with your configuration:

.. code-block:: bash

   homodyne --config my_config.yaml

You should see output like:

.. code-block:: text

   Homodyne.4.1
   ===============

   Loading configuration: my_config.yaml
   Loading XPCS data: /path/to/xpcs_data.h5

   Analysis Configuration:
     Mode: static
     Angles: 3
     Physical Parameters: D0, alpha, D_offset

   Running NLSQ optimization...

   Iteration    Cost         Gradient Norm
   1            1234.56      45.23
   2            1189.45      12.34
   3            1185.67      2.45
   ...

   ✓ Optimization converged in 5 iterations

   Best-Fit Parameters:
   ═════════════════════
   D0 (Diffusion):        1234.5 ± 45.6 μm²/s
   alpha (Time exponent): 0.567 ± 0.012
   D_offset (Offset):     123.4 ± 5.6 μm²/s

   Results saved to: ./homodyne_results/


Step 3: View Results
--------------------

Homodyne automatically generates results in ``./homodyne_results/``:

.. code-block:: text

   homodyne_results/
   ├── results.json              # Complete parameter estimates
   ├── residuals.png             # Residual analysis plot
   ├── c2_fit.png                # Two-time correlation plot
   ├── correlation_heatmap.png   # Heatmap visualization
   └── convergence.json          # Optimization convergence details

**Key Output Files:**

- ``results.json``: Best-fit parameters and uncertainty
- ``c2_fit.png``: Visual comparison of data vs model
- ``correlation_heatmap.png``: Two-time correlation visualization

Step 4: Interpret Results
--------------------------

The output shows:

1. **Parameter Values:** Best-fit estimate and uncertainty
2. **Physics Interpretation:**
   - ``D0`` (Diffusion coefficient) ~1234 μm²/s
   - ``alpha`` (Time exponent) ~0.57 (sub-diffusive)
   - ``D_offset`` (Offset) ~123 μm²/s

3. **Convergence Status:** "Converged" = good fit achieved

Understanding the Output
========================

**What These Parameters Mean:**

.. math::

   D(t) = D_0 \times t^{\alpha} + D_{\text{offset}}

- **D₀** (Diffusion): Proportional to particle mobility
- **α** (Exponent):
  - α = 1.0 → Normal diffusion
  - α < 1.0 → Subdiffusion (hindered motion)
  - α > 1.0 → Superdiffusion (enhanced motion)
- **D_offset** (Baseline): Instrumental offset or non-dynamic scattering

**Uncertainty Values:**

The ± values indicate confidence in the estimates:

- Smaller uncertainty = more precise fit
- Larger uncertainty = data may be noisier or parameter less constrained

Next Steps: Adding Complexity
==============================

**Option 1: Include Multiple Angles**

Use ``n_angles: 5`` or more for better statistics (update YAML accordingly).

**Option 2: Add Laminar Flow Physics**

Change to ``mode: laminar_flow`` for systems with velocity gradients:

.. code-block:: yaml

   analysis:
     mode: laminar_flow                        # Add shear-rate dependency
     n_angles: 3

   optimization:
     initial_parameters:
       values: [1000.0, 0.5, 100.0,           # Physical params: D0, alpha, D_offset
                1.0, 0.5, 0.5, 0.0]            # Shear params: gamma_dot_t0, beta, offset, phi0

This adds 4 additional parameters for shear-rate dependent dynamics.

**Option 3: Bayesian Uncertainty Quantification**

After NLSQ converges, run MCMC:

.. code-block:: bash

   # Step 1: Copy best-fit from NLSQ results.json
   # Step 2: Update initial_parameters.values in config
   homodyne --config my_config.yaml --method mcmc

This provides full posterior distributions instead of point estimates.

Troubleshooting Common Issues
=============================

**Issue: "Cannot load data from HDF5 file"**

Check:

1. File path is correct
2. Keys match your HDF5 structure:

.. code-block:: bash

   # View HDF5 structure
   h5dump -H /path/to/xpcs_data.h5

3. Data arrays are numeric (float32 or float64)

**Issue: "Initial parameters out of bounds"**

Solution: Adjust bounds in configuration:

.. code-block:: yaml

   parameter_bounds:
     D0: [100.0, 10000.0]       # Make wider if needed

**Issue: "Optimization did not converge"**

Try:

1. Improve initial parameter guesses (closer to truth)
2. Loosen parameter bounds
3. Check data quality (look at residuals plot)
4. Use more angles (n_angles: 5 or more)

**Issue: "Memory error during optimization"**

Reduce data size:

1. Use fewer angles: ``n_angles: 3``
2. Downsample time: ``t1_slice: ":10"`` (take every 10th time point)

Quick Reference
===============

**Basic Commands:**

.. code-block:: bash

   # Run analysis
   homodyne --config config.yaml

   # Run with specific method
   homodyne --config config.yaml --method nlsq   # Fast
   homodyne --config config.yaml --method mcmc   # Detailed

   # Check options
   homodyne --help

   # Generate config interactively
   homodyne-config --interactive

   # Validate existing config
   homodyne-config --validate my_config.yaml

**Configuration Modes:**

- ``static``: Time-dependent diffusion, 3 physical parameters
- ``laminar_flow``: Diffusion + shear, 7 physical parameters

**Optimization Methods:**

- ``nlsq``: Fast, point estimates, ~seconds to minutes
- ``mcmc``: Slow, full posteriors, ~hours to days

**Output Files:**

- ``results.json``: Best-fit parameters and uncertainties
- ``c2_fit.png``: Visual fit quality
- ``convergence.json``: Optimization details

Next Steps
==========

- :doc:`./cli` - Learn all command-line options
- :doc:`./configuration` - Deep dive into configuration
- :doc:`./examples` - Real-world analysis workflows
- :doc:`../research/theoretical_framework` - Understand the physics
- :doc:`../api-reference/index` - API reference for developers
