Examples
========

This guide provides real-world examples of Homodyne analyses for different physics scenarios.
Each example includes a complete configuration file and interpretation of results.

Example 1: Static Mode - Polymer Solution
==========================================

**Scenario:** Measuring diffusion dynamics of polymers in solution at rest.

**Physics:** Time-dependent diffusion without shear flow

**Data Characteristics:**

- 3 azimuthal angles
- 100 time points
- ~5000 q-values
- Isotropic scattering

Configuration File
------------------

**File: `polymer_static.yaml`**

.. code-block:: yaml

   data:
     path: ./data/polymer_xpcs.h5
     h5_keys:
       c2: entry/instrument/detector/data
       t1: entry/instrument/detector/t1
       t2: entry/instrument/detector/t2
       phi: entry/instrument/detector/phi

   analysis:
     mode: static                    # Time-dependent diffusion only
     n_angles: 3                     # Three detector arms
     remove_bad_angles: true         # Reject low-intensity angles
     phi_std_threshold: 2.0          # 2-sigma rejection

   optimization:
     method: nlsq                    # Fast point estimates
     initial_parameters:
       values: [1200.0, 0.65, 80.0] # [D0, alpha, D_offset]
       bounds:
         D0: [100.0, 5000.0]         # Diffusion coefficient
         alpha: [0.0, 1.5]           # Subdiffusion exponent
         D_offset: [0.0, 200.0]      # Background offset
     nlsq:
       max_iterations: 100
       tolerance: 1e-6
       verbose: true

   output:
     results_dir: ./polymer_results
     save_plots: true
     plot_format: png

Running the Analysis
--------------------

.. code-block:: bash

   # Validate configuration
   homodyne-config --validate polymer_static.yaml

   # Run analysis
   homodyne --config polymer_static.yaml

Expected Output
---------------

.. code-block:: text

   Homodyne.4.1
   ===============
   Loading: polymer_xpcs.h5

   Analysis: Static Mode
   Parameters: 3 physical + 6 per-angle (total: 9)
   Angles: 3
   Times: 100

   Running NLSQ optimization...

   Iteration    Cost        Gradient Norm
   1            2345.67     234.56
   2            1234.56     56.78
   3            1189.34     12.34
   4            1185.67     2.34
   5            1185.23     0.45

   ✓ Optimization converged in 5 iterations

   BEST-FIT PARAMETERS
   ===================
   Physical Parameters:
     D0 (Diffusion):              1234.5 ± 45.6 μm²/s
     alpha (Time exponent):        0.678 ± 0.012
     D_offset (Offset):           123.4 ± 5.6 μm²/s

   Per-Angle Scaling:
     Angle 0: contrast=0.95, offset=1.02
     Angle 1: contrast=0.98, offset=0.98
     Angle 2: contrast=0.96, offset=1.01

   Results saved to: ./polymer_results/

Interpreting Results
--------------------

**D₀ = 1234.5 μm²/s:**

Diffusion coefficient. Compare to expected values:

- Stokes-Einstein: D = kT/(6πηr)
- For R = 50 nm, η = 1 cP → D ≈ 4000 μm²/s
- Measured value is lower → slower diffusion
- Possible causes: larger effective radius, higher viscosity

**α = 0.678:**

Time exponent. Physical interpretation:

- α = 1.0 → Normal Brownian diffusion
- α = 0.678 → Sub-diffusion (hindered by structure)
- Possible causes: crowded environment, transient caging

**D_offset = 123.4 μm²/s:**

Background contribution. Possible sources:

- Instrumental offset (constant factor)
- Static scattering (non-decaying)
- Aggregates or dust particles

Visualization Files
-------------------

Generated output files:

- `c2_fit.png`: Two-time correlation function with best-fit
- `residuals.png`: Residual analysis (should show no trends)
- `correlation_heatmap.png`: Heatmap of c₂(t₁, t₂)
- `results.json`: Complete parameter estimates

---

Example 2: Laminar Flow - Polymer in Shear
==========================================

**Scenario:** Measuring dynamics of polymers under controlled shear flow.

**Physics:** Both diffusion and shear-dependent dynamics

**Data Characteristics:**

- 5 azimuthal angles (asymmetric due to flow)
- 150 time points
- ~8000 q-values
- Strong angle-dependent behavior

Configuration File
------------------

**File: `polymer_laminar.yaml`**

.. code-block:: yaml

   data:
     path: ./data/polymer_flow_xpcs.h5
     h5_keys:
       c2: entry/instrument/detector/data
       t1: entry/instrument/detector/t1
       t2: entry/instrument/detector/t2
       phi: entry/instrument/detector/phi
     # Optional: use subset of data
     t1_slice: ":100"              # First 100 time points
     phi_slice: ":"                # All angles

   analysis:
     mode: laminar_flow            # Includes shear rate dependence
     n_angles: 5                   # More angles for anisotropy
     remove_bad_angles: true

   optimization:
     method: nlsq
     initial_parameters:
       # 7 physical parameters + 10 per-angle (2*5)
       values: [1500.0, 0.60, 150.0,  # Physical: D0, alpha, D_offset
                5.0, 0.5, 0.5, 0.0]   # Shear: gamma_dot_t0, beta, offset, phi0
       bounds:
         D0: [500.0, 5000.0]
         alpha: [0.0, 1.5]
         D_offset: [0.0, 300.0]
         gamma_dot_t0: [0.1, 50.0]    # Shear rate coefficient
         beta: [0.0, 1.0]              # Shear time exponent
         gamma_dot_t_offset: [0.0, 10.0] # Shear offset
         phi0: [-3.14, 3.14]           # Flow direction
     nlsq:
       max_iterations: 150
       tolerance: 1e-6

   output:
     results_dir: ./polymer_flow_results
     save_plots: true

Running the Analysis
--------------------

.. code-block:: bash

   # Run with verbose output to monitor convergence
   homodyne --config polymer_laminar.yaml --verbose

Expected Output
---------------

.. code-block:: text

   Analysis: Laminar Flow Mode
   Parameters: 7 physical + 10 per-angle (total: 17)
   Angles: 5

   BEST-FIT PARAMETERS
   ===================
   Diffusion Parameters:
     D0:              1567.8 ± 67.3 μm²/s
     alpha:           0.612 ± 0.018
     D_offset:        178.9 ± 12.4 μm²/s

   Shear Rate Parameters:
     gamma_dot_t0:    6.78 ± 0.45 s⁻¹
     beta:            0.52 ± 0.08
     gamma_dot_t_offset: 0.34 ± 0.12 s⁻¹
     phi0 (flow dir): 0.123 ± 0.045 rad

Interpreting Results
--------------------

**gamma_dot_t0 = 6.78 s⁻¹:**

Shear-rate coefficient. This modulates dynamics with shear:

.. math::

   \dot{\gamma}(t) = 6.78 \times t^{0.52} + 0.34

At t = 1 second: γ̇ ≈ 7.1 s⁻¹

**phi0 = 0.123 rad ≈ 7°:**

Flow direction relative to reference angle.

**Angle Dependence:**

With shear, each angle shows different decay rates:

- Angle aligned with flow: fastest decay
- Angle perpendicular: slower decay
- Anisotropy captured by per-angle scaling

---

Example 3: MCMC Uncertainty Quantification
==========================================

**Scenario:** Complete Bayesian analysis with posterior distributions.

**Workflow:** NLSQ → MCMC

Step 1: Run NLSQ
----------------

**File: `config_step1_nlsq.yaml`**

.. code-block:: yaml

   data:
     path: ./data/sample.h5
     h5_keys:
       c2: entry/detector/c2
       t1: entry/detector/t1
       t2: entry/detector/t2
       phi: entry/detector/phi

   analysis:
     mode: static
     n_angles: 3

   optimization:
     method: nlsq
     initial_parameters:
       values: [1000.0, 0.5, 100.0]

   output:
     results_dir: ./nlsq_results

Run it:

.. code-block:: bash

   homodyne --config config_step1_nlsq.yaml

   # Result: ./nlsq_results/results.json

Step 2: Extract Best-Fit Values
--------------------------------

.. code-block:: bash

   cat nlsq_results/results.json

Output example:

.. code-block:: json

   {
     "D0": {
       "value": 1234.5,
       "uncertainty": 45.6
     },
     "alpha": {
       "value": 0.567,
       "uncertainty": 0.012
     },
     "D_offset": {
       "value": 123.4,
       "uncertainty": 5.6
     }
   }

Step 3: Configure MCMC
----------------------

**File: `config_step2_mcmc.yaml`**

Copy configuration from Step 1, then:

.. code-block:: yaml

   optimization:
     method: mcmc                              # Change to MCMC
     initial_parameters:
       values: [1234.5, 0.567, 123.4]         # From NLSQ results
       bounds:                                 # Keep same bounds
         D0: [100.0, 10000.0]
         alpha: [0.0, 1.0]
         D_offset: [0.0, 500.0]
     mcmc:
       n_samples: 2000                         # Posterior samples
       n_warmup: 1000                          # Burn-in iterations
       n_chains: 4                             # Parallel chains
       backend: "multiprocessing"

   output:
     results_dir: ./mcmc_results

Step 4: Run MCMC
----------------

.. code-block:: bash

   # MCMC is computationally expensive
   # This may take hours to days depending on data size
   homodyne --config config_step2_mcmc.yaml --verbose

   # Monitor with:
   tail -f ./mcmc_results/convergence.json

Expected MCMC Output
--------------------

.. code-block:: text

   Running MCMC (CMC-only)...
   Chain 1: 25% complete
   Chain 2: 25% complete
   Chain 3: 25% complete
   Chain 4: 25% complete
   ...

   MCMC Results:
   =============
   D0:
     Mean:      1234.5
     Std:       45.6
     5% (HPD):  1152.3
     95% (HPD): 1316.7

   alpha:
     Mean:      0.567
     Std:       0.012
     5% (HPD):  0.545
     95% (HPD): 0.589

   D_offset:
     Mean:      123.4
     Std:       5.6
     5% (HPD):  112.8
     95% (HPD): 134.0

Posterior Interpretation
------------------------

MCMC provides full posterior distributions vs. point estimates:

- **Mean:** Expected value (point estimate)
- **Std:** Posterior standard deviation
- **HPD:** Highest posterior density credible interval
  - 90% probability true value is in this range

Compare to NLSQ:

.. table::
   :widths: 30 20 20 20

   +----------+--------+-------+-------+
   | Method   | Value  | Std   | Range |
   +==========+========+=======+=======+
   | NLSQ     | 1234.5 | 45.6  | -     |
   +----------+--------+-------+-------+
   | MCMC 90% | 1234.5 | 45.6  | ±89.2 |
   +----------+--------+-------+-------+

The MCMC range is asymmetric around the mean, capturing non-Gaussian behavior.

---

Example 4: Multi-Angle Laminar Flow Analysis
============================================

**Scenario:** High-precision analysis with many detector arms.

**Data:** 7 angles, 200 times, 10K q-values

Configuration
-------------

.. code-block:: yaml

   data:
     path: ./data/detailed_flow.h5
     h5_keys:
       c2: entry/data/c2
       t1: entry/data/t1
       t2: entry/data/t2
       phi: entry/data/phi
     # Use good data quality only
     angle_filtering:
       enabled: true
       min_angle: 15
       max_angle: 165

   analysis:
     mode: laminar_flow
     n_angles: 7
     remove_bad_angles: true
     phi_std_threshold: 1.5

   optimization:
     method: nlsq
     initial_parameters:
       values: [2000.0, 0.5, 200.0,
                10.0, 0.5, 1.0, 0.0]
       bounds:
         D0: [1000.0, 5000.0]
         alpha: [0.3, 1.2]
         D_offset: [100.0, 400.0]
         gamma_dot_t0: [5.0, 50.0]
         beta: [0.3, 0.8]
         gamma_dot_t_offset: [0.0, 5.0]
         phi0: [-1.57, 1.57]  # -90 to +90 degrees
     nlsq:
       max_iterations: 200
       loss: "linear"
       verbose: true

   output:
     results_dir: ./detailed_flow_results
     save_plots: true

   logging:
     level: DEBUG
     console: true

Running Advanced Analysis
--------------------------

.. code-block:: bash

   # Validate first (important with complex configs)
   homodyne-config --validate config.yaml

   # Run with debug output
   homodyne --config config.yaml --verbose

   # Inspect results
   ls -lh detailed_flow_results/
   cat detailed_flow_results/results.json | python -m json.tool

Advanced Interpretation
------------------------

With 7 angles, you can extract:

1. **Radial decay rate:** Shared across all angles (D₀, α)
2. **Azimuthal anisotropy:** Via phi0 parameter
3. **Shear-rate dependence:** Via gamma_dot_t0, beta
4. **Intensity scaling:** Per-angle contrast and offset

This provides detailed understanding of anisotropic dynamics.

---

Quick Reference: When to Use What
==================================

**Use Static Mode When:**

- Sample is isotropic (no directional dependence)
- No external shear or flow
- Simple diffusion dynamics
- Quick exploration of data

.. code-block:: bash

   homodyne --config config_static.yaml --method nlsq

**Use Laminar Flow Mode When:**

- Sample has shear flow or velocity gradients
- Anisotropic scattering patterns
- Need to extract shear-rate dependence
- More complex physics

.. code-block:: bash

   homodyne --config config_laminar.yaml --method nlsq

**Use NLSQ When:**

- Want fast results (seconds to minutes)
- Exploring parameter space
- Testing configurations
- Need point estimates only

.. code-block:: bash

   homodyne --config config.yaml --method nlsq

**Use CMC When:**

- Need uncertainty quantification
- Have good initial estimates (from NLSQ)
- Can wait (hours to days)
- Need full posterior distributions

.. code-block:: bash

   homodyne --config config.yaml --method cmc

---

Configuration Templates Repository
===================================

Full configuration files are available in:

.. code-block:: bash

   homodyne/config/templates/

Templates provided:

- `static_minimal.yaml` - Minimal static configuration
- `static_complete.yaml` - Full options for static mode
- `laminar_minimal.yaml` - Minimal laminar flow configuration
- `laminar_complete.yaml` - Full options for laminar flow
- `mcmc_static.yaml` - MCMC configuration for static mode
- `mcmc_laminar.yaml` - MCMC configuration for laminar flow

Use templates as starting points:

.. code-block:: bash

   cp homodyne/config/templates/static_complete.yaml my_config.yaml
   # Edit my_config.yaml with your data paths and parameters
   homodyne --config my_config.yaml

---

Next Steps
==========

- :doc:`./cli` - Learn advanced command-line options
- :doc:`./configuration` - Detailed configuration reference
- :doc:`../research/theoretical_framework` - Mathematical foundations
- :doc:`../api-reference/index` - API reference for custom scripts
