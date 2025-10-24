MCMC Uncertainty Quantification
===============================

Overview
--------

MCMC (Markov Chain Monte Carlo) performs Bayesian inference to obtain full posterior distributions for parameters, enabling rigorous uncertainty quantification beyond point estimates from NLSQ.

**Key Features:**

- **NUTS Sampler**: No-U-Turn Sampler for efficient exploration
- **Multiple Backends**: NumPyro (primary) or BlackJAX
- **Automatic Adaptation**: Learns step size and metric during warmup
- **Diagnostic Tools**: R-hat convergence statistics, ESS, trace plots
- **Full Posterior**: Complete probability distributions for all parameters

When to Use MCMC
----------------

Use MCMC when you need:

- **Uncertainty quantification**: Full posterior distributions instead of point estimates
- **Confidence intervals**: Credible intervals beyond covariance-based uncertainties
- **Parameter correlations**: Joint posterior distributions showing parameter dependencies
- **Model comparison**: Marginal likelihoods and Bayes factors
- **Prediction uncertainty**: Posterior predictive distributions for future data

Typical workflow: **NLSQ first for speed, then MCMC for rigorous uncertainties**.

Configuration
--------------

Basic Configuration
~~~~~~~~~~~~~~~~~~~~

Minimal MCMC setup with NumPyro (default):

.. code-block:: yaml

    optimization:
      method: "mcmc"
      mcmc:
        num_warmup: 1000          # Warmup/burn-in samples
        num_samples: 2000         # Posterior samples per chain
        num_chains: 4             # Parallel chains for convergence checks
        progress_bar: true        # Show sampling progress
        backend: "numpyro"        # numpyro | blackjax

**Recommended Settings:**

- **num_warmup**: 1000-2000 (warm up the sampler)
- **num_samples**: 2000-5000 (get accurate posterior estimates)
- **num_chains**: 4-8 (check convergence across chains)
- **progress_bar**: true (monitor sampling progress)

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

Fine-tune sampling for difficult posteriors:

.. code-block:: yaml

    optimization:
      mcmc:
        # Sampling control
        num_warmup: 2000
        num_samples: 5000
        num_chains: 8

        # Hardware settings
        backend: "numpyro"
        device: "gpu"             # gpu | cpu

        # Sampler tuning (NUTS-specific)
        adapt_step_size: true
        adapt_mass_matrix: true
        target_accept_prob: 0.8   # Higher = smaller steps, fewer rejections

**When to Adjust:**

- **Increase num_chains** if R-hat > 1.01
- **Increase num_warmup** if R-hat improves with more warmup
- **Increase num_samples** if credible intervals are still changing
- **Lower target_accept_prob** (e.g., 0.7) if sampler is too conservative

NumPyro vs. BlackJAX
~~~~~~~~~~~~~~~~~~~~

.. list-table::
    :header-rows: 1
    :widths: 30 35 35

    * - Feature
      - NumPyro
      - BlackJAX
    * - Progress bars
      - Yes (best)
      - No
    * - NUTS implementation
      - Optimized
      - Functional
    * - GPU support
      - Full
      - Full
    * - Diagnostics
      - Comprehensive
      - Basic
    * - Recommendation
      - Use this
      - Research/testing

Use NumPyro (default) for production. BlackJAX is an alternative backend if needed.

Output Interpretation
---------------------

Result Files
~~~~~~~~~~~~

MCMC saves detailed posterior samples and diagnostics:

.. code-block:: text

    output_dir/mcmc/
    ├── samples.npy               # Raw posterior samples
    ├── parameters.json           # Posterior statistics
    ├── diagnostics.json          # Convergence diagnostics
    ├── corner_plot.png          # Joint posterior visualization
    └── trace_plots.png          # Chain traces

parameters.json
^^^^^^^^^^^^^^^

Posterior statistics for each parameter:

.. code-block:: json

    {
      "D0": {
        "mean": 1245.3,
        "median": 1243.8,
        "std": 48.5,
        "credible_interval_68": [1197, 1293],
        "credible_interval_95": [1153, 1345]
      },
      "alpha": {
        "mean": 0.86,
        "median": 0.86,
        "std": 0.052,
        "credible_interval_68": [0.808, 0.912],
        "credible_interval_95": [0.758, 0.962]
      }
    }

**Interpretation:**

- **mean/median**: Central posterior estimate
- **std**: Posterior standard deviation (uncertainty width)
- **credible_interval_68**: 68% credible interval (≈ ±1 sigma)
- **credible_interval_95**: 95% credible interval (≈ ±2 sigma)

diagnostics.json
^^^^^^^^^^^^^^^^

Convergence and mixing diagnostics:

.. code-block:: json

    {
      "r_hat": {
        "D0": 1.003,
        "alpha": 1.001,
        "gamma_dot_0": 1.004
      },
      "effective_sample_size": {
        "D0": 7832,
        "alpha": 7945,
        "gamma_dot_0": 7654
      },
      "acceptance_rate": 0.81,
      "num_divergences": 0
    }

**Interpretation:**

- **R-hat < 1.01**: Chains converged (good!)
- **R-hat 1.01-1.05**: Chains mostly converged (acceptable)
- **R-hat > 1.05**: Chains not converged (rerun with more iterations)
- **ESS > 400**: Sufficient effective samples
- **acceptance_rate ≈ 0.8**: Good (target is 0.8)
- **divergences = 0**: No numerical issues

Posterior Visualization
~~~~~~~~~~~~~~~~~~~~~~~

Corner plots show joint posterior distributions:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    # Load samples
    samples = np.load('samples.npy')  # Shape: (n_chains * n_samples, n_params)

    # Simple 2D marginal plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1)
    ax.set_xlabel('D0')
    ax.set_ylabel('alpha')
    ax.set_title('Posterior correlation')
    plt.show()

Comparing NLSQ and MCMC
~~~~~~~~~~~~~~~~~~~~~~~

Example comparison:

.. code-block:: python

    import json
    import numpy as np

    # Load NLSQ results
    with open('results_nlsq/parameters.json') as f:
        nlsq = json.load(f)

    # Load MCMC results
    with open('results_mcmc/parameters.json') as f:
        mcmc = json.load(f)

    # Compare D0
    print("D0 Comparison:")
    print(f"  NLSQ: {nlsq['D0']['value']:.2f} ± {nlsq['D0']['uncertainty']:.2f}")
    print(f"  MCMC: {mcmc['D0']['mean']:.2f} ± {mcmc['D0']['std']:.2f}")

    # MCMC uncertainties are typically larger (more conservative)

Workflow Examples
-----------------

Basic MCMC Analysis
~~~~~~~~~~~~~~~~~~~~

Simple example with default settings:

.. code-block:: yaml

    # config_mcmc.yaml
    experimental_data:
      file_path: "./data/experiment.hdf"

    parameter_space:
      model: "static_isotropic"
      bounds:
        - name: D0
          min: 100.0
          max: 100000.0
        - name: alpha
          min: 0.0
          max: 2.0
        - name: D_offset
          min: -100.0
          max: 100.0

    initial_parameters:
      parameter_names: ["D0", "alpha", "D_offset"]

    optimization:
      method: "mcmc"
      mcmc:
        num_warmup: 1000
        num_samples: 2000
        num_chains: 4
        backend: "numpyro"

Run MCMC:

.. code-block:: bash

    homodyne --config config_mcmc.yaml --output-dir results_mcmc

Expected runtime: 5-30 minutes (depending on dataset size)

Two-Stage Workflow: NLSQ then MCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Professional workflow combining speed and rigor:

.. code-block:: bash

    # Stage 1: Quick NLSQ for initial estimates
    homodyne --config config.yaml --method nlsq --output-dir results_nlsq

    # Stage 2: Use NLSQ results as starting point for MCMC
    # (Optional: update initial_parameters based on NLSQ results)
    homodyne --config config_mcmc.yaml --output-dir results_mcmc

This approach: **NLSQ in 1-5 min + MCMC in 10-30 min = complete uncertainty analysis in 15-35 min**

HPC Parallel Chains
~~~~~~~~~~~~~~~~~~~

For HPC environments, run multiple chains in parallel:

.. code-block:: yaml

    optimization:
      mcmc:
        num_warmup: 2000
        num_samples: 5000
        num_chains: 32        # Parallel chains on HPC

This provides excellent convergence diagnostics and uses HPC efficiently.

Troubleshooting
---------------

Non-Converged Chains (R-hat > 1.05)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Causes and Solutions:**

1. **Insufficient warmup**
   - Solution: Increase num_warmup to 2000-5000
   - Warmup helps learn optimal step sizes

2. **Chains not mixing**
   - Solution: Increase num_samples or num_chains
   - Check trace plots for stuck chains

3. **Difficult posterior**
   - Solution: Improve initial parameters or parameter bounds
   - Start with NLSQ results

High Divergence Rate (> 0 divergences)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Indicator**: Sampler struggled with posterior geometry

**Solution**:

.. code-block:: yaml

    optimization:
      mcmc:
        target_accept_prob: 0.85   # Increase to 0.85-0.9
        adapt_mass_matrix: true    # Learn covariance structure

Low ESS (Effective Sample Size < 400)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cause**: Poor mixing (high autocorrelation)

**Solutions:**

1. Run longer: Increase num_samples
2. Improve initial parameters (use NLSQ first)
3. Check if posterior is multimodal (different modes in different chains)

See Also
--------

- :doc:`nlsq-optimization` - Point estimates with NLSQ
- :doc:`../user-guide/configuration` - Configuration system
- :doc:`../api-reference/optimization` - Optimization API
- :doc:`../theoretical-framework/core-equations` - Theory

References
----------

**NUTS Algorithm:**
- Hoffman, M.D. & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
- J. Mach. Learn. Res. 15: 1593-1623

**NumPyro:**
- GitHub: https://github.com/pyro-ppl/numpyro
- Documentation: https://num.pyro.ai/

**Bayesian Inference:**
- Gelman, A., et al. (2013). "Bayesian Data Analysis" (3rd ed.)
- McElreath, R. (2015). "Statistical Rethinking"
