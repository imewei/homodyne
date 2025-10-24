Quickstart: Your First Analysis in 5 Minutes
==============================================

This quickstart guide walks you through your first Homodyne analysis in just 5 minutes.

Prerequisites
-------------

- Homodyne installed (see :doc:`installation`)
- Sample HDF5 data file (or download from examples)
- Basic familiarity with YAML

Step 1: Create a Configuration File (1 minute)
-----------------------------------------------

Create a file named ``config.yaml``:

.. code-block:: yaml

   # Minimal configuration for first analysis
   experimental_data:
     file_path: "./data/sample/experiment.hdf"

   parameter_space:
     model: "static_isotropic"

     bounds:
       - name: D0
         min: 100.0
         max: 1e5
       - name: alpha
         min: 0.0
         max: 2.0
       - name: D_offset
         min: -100.0
         max: 100.0

   initial_parameters:
     parameter_names:
       - D0
       - alpha
       - D_offset

   optimization:
     method: "nlsq"
     nlsq:
       max_iterations: 100
       tolerance: 1e-8

   output:
     directory: "./results"

For more comprehensive templates, see :doc:`configuration`.

Step 2: Run the Analysis (2 minutes)
------------------------------------

.. code-block:: bash

   # Navigate to directory with config.yaml
   cd your_project_directory

   # Run analysis
   homodyne --config config.yaml

You should see output like:

.. code-block:: text

   Loading configuration from config.yaml
   Loading experimental data from ./data/sample/experiment.hdf
   Initializing NLSQ optimization...
   Optimization in progress... [████████████████████] 100%
   Analysis complete!
   Results saved to ./results/

Step 3: Examine Results (2 minutes)
-----------------------------------

**Output files created in ``./results/``:**

1. **parameters.json** - Optimized parameter values

   .. code-block:: json

      {
        "D0": {
          "value": 1250.5,
          "uncertainty": 45.3,
          "unit": "Å²/s"
        },
        "alpha": {
          "value": 0.62,
          "uncertainty": 0.08,
          "unit": "dimensionless"
        },
        "D_offset": {
          "value": 12.3,
          "uncertainty": 5.2,
          "unit": "Å²/s"
        }
      }

2. **fitted_data.npz** - Experimental and theoretical data

   .. code-block:: python

      import numpy as np
      data = np.load('./results/fitted_data.npz')
      print(data.files)  # List available arrays
      # ['experimental_c2', 'theoretical_c2', 'residuals', ...]

3. **analysis_results_nlsq.json** - Fit quality metrics

   .. code-block:: json

      {
        "chi_squared": 1.235,
        "reduced_chi_squared": 1.189,
        "number_of_data_points": 1000,
        "number_of_parameters": 3
      }

4. **convergence_metrics.json** - Optimization details

   .. code-block:: json

      {
        "iterations": 23,
        "final_trust_region_radius": 0.145,
        "convergence_flag": 4,
        "computation_time_seconds": 8.32
      }

Understanding Your Results
---------------------------

**Chi-squared value:**

- :math:`\chi^2 \approx 1` indicates good fit quality
- :math:`\chi^2 < 0.5` may indicate overfitting
- :math:`\chi^2 > 2` suggests parameter adjustment needed

**Uncertainties:**

- Computed from the covariance matrix
- Larger uncertainties indicate parameter is less constrained
- Use relative uncertainty: :math:`\Delta p / p` to judge significance

**Convergence flag:**

- ``4`` = convergence success
- ``2`` = max iterations reached (try increasing `max_iterations`)
- ``1`` = tolerance achieved

Adding Visualization
--------------------

To visualize your results:

.. code-block:: bash

   homodyne --config config.yaml --plot-experimental-data

This creates additional plots in ``./results/plots/``:

- **c2_fit.png** - Two-time correlation heatmap with fit overlay
- **residuals.png** - Residual analysis plots
- **parameters.png** - Parameter distributions

Next Steps
----------

Congratulations on your first analysis! To continue:

1. **Explore other analysis modes:**
   - Static isotropic (used in this quickstart)
   - Laminar flow (see :doc:`examples`)
   - MCMC uncertainty quantification (see :doc:`examples`)

2. **Learn the configuration system:**
   - :doc:`configuration` - Detailed parameter explanation
   - Parameter counting: 3+2n for static, 7+2n for laminar
   - Advanced settings: streaming, GPU acceleration, checkpoints

3. **Try advanced features:**
   - :doc:`../advanced-topics/nlsq-optimization` - Trust-region optimization details
   - :doc:`../advanced-topics/angle-filtering` - Angular selection
   - :doc:`../advanced-topics/gpu-acceleration` - GPU speedup (Linux only)

4. **Run example workflows:**
   - :doc:`examples` - Real-world use cases
   - Static isotropic + NLSQ
   - Laminar flow + angle filtering
   - MCMC uncertainty quantification

Troubleshooting
---------------

**"No such file or directory: ./data/sample/experiment.hdf"**

Update the data file path in ``config.yaml``:

.. code-block:: yaml

   experimental_data:
     file_path: "/absolute/path/to/your/data.hdf"

Use absolute paths for reliability.

**"ModuleNotFoundError: No module named 'homodyne'"**

Ensure homodyne is installed:

.. code-block:: bash

   pip install homodyne
   homodyne --version

**"Error: Invalid configuration"**

Check YAML syntax:

.. code-block:: bash

   python -c "import yaml; yaml.safe_load(open('config.yaml'))"

Should succeed without error.

**Analysis is very slow**

- For small datasets (<10M points), CPU is often faster than GPU
- Check :doc:`../developer-guide/performance` for optimization tips
- Reduce `max_iterations` to test quickly

See Also
--------

- :doc:`configuration` - Detailed configuration guide
- :doc:`cli-usage` - Command-line reference
- :doc:`examples` - More advanced examples
- :doc:`../api-reference/index` - Full API documentation
