.. _quickstart:

Quick Start — 5 Minutes to First Analysis
==========================================

This guide walks you through installing Homodyne, loading an XPCS dataset,
running a static-mode NLSQ fit, and interpreting the results.

Prerequisites: Python 3.12+, a terminal, and an HDF5 file with two-time
correlation data.  See :doc:`installation` for a full dependency list.

----

.. _qs-install:

Step 1: Install
---------------

.. tabs are not natively supported in RST; use code-block sections instead

**Using uv (recommended):**

.. code-block:: bash

   uv add homodyne

**Using pip:**

.. code-block:: bash

   pip install homodyne

Verify the installation:

.. code-block:: bash

   homodyne --version
   # Homodyne 2.22.0

----

.. _qs-config:

Step 2: Generate a Configuration File
--------------------------------------

Homodyne is configured through YAML files.  The ``homodyne-config`` utility
generates a template with sensible defaults:

.. code-block:: bash

   homodyne-config --mode static --output my_config.yaml

Open ``my_config.yaml`` and set the path to your HDF5 data file:

.. code-block:: yaml

   data:
     path: /path/to/your/data.h5
     group: /exchange          # HDF5 group containing C2 matrix

   analysis:
     mode: static              # static | laminar_flow
     q_min: 0.001              # Minimum q-value (nm^-1)
     q_max: 0.1                # Maximum q-value (nm^-1)

   optimization:
     method: nlsq              # nlsq | cmc

See :doc:`user_guide/04_practical_guides/configuration` for the full
configuration reference.

----

.. _qs-load-data:

Step 3: Load Data
-----------------

Homodyne reads two-time correlation matrices (C2) from HDF5 files:

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager

   # Load configuration
   config = ConfigManager.from_yaml("my_config.yaml")

   # Load experimental data
   loader = XPCSDataLoader(config)
   data   = loader.load()

   print(f"C2 shape:   {data.c2.shape}")      # (n_q, n_t, n_t)
   print(f"Time grid:  {data.t_grid.shape}")  # (n_t,)
   print(f"q-values:   {data.q_values}")      # (n_q,) in nm^-1

The loader validates shape, dtype, and NaN presence before returning.

----

.. _qs-run-nlsq:

Step 4: Run Static NLSQ Analysis
----------------------------------

NLSQ is the primary optimization method — fast trust-region
Levenberg-Marquardt with JAX-JIT compiled residuals:

.. code-block:: python

   from homodyne.optimization.nlsq import fit_nlsq_jax

   result = fit_nlsq_jax(data, config)

   # Inspect fit quality
   print(f"Reduced chi-squared: {result.chi2_reduced:.4f}")
   print(f"Converged:           {result.converged}")

   # Physical parameters
   params = result.params
   print(f"D0       = {params['D0']:.4e}  nm^2/s")
   print(f"alpha    = {params['alpha']:.3f}  (anomalous exponent)")
   print(f"D_offset = {params['D_offset']:.4e}  nm^2/s")

For ``laminar_flow`` mode the result also contains ``gamma_dot_0``,
``beta``, ``gamma_dot_offset``, and ``phi_0``.

----

.. _qs-cli:

Step 5: Run from the Command Line
-----------------------------------

All of the above can be driven through the CLI:

.. code-block:: bash

   # NLSQ point estimate
   homodyne --config my_config.yaml --method nlsq

   # Bayesian CMC (requires NLSQ warm-start for best results)
   homodyne --config my_config.yaml --method cmc

Results are written to the directory specified by ``output.path`` in the
configuration file as JSON and NPZ bundles.

----

.. _qs-interpret:

Interpreting Results
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Field
     - Typical range
     - Meaning
   * - ``D0``
     - 1e-3 – 1e4 nm²/s
     - Diffusion coefficient at zero shear
   * - ``alpha``
     - 0.5 – 2.0
     - Anomalous diffusion exponent (1 = Fickian)
   * - ``D_offset``
     - 0 – D0
     - Background / static diffusion contribution
   * - ``chi2_reduced``
     - ~1.0
     - Goodness-of-fit (1 = perfect, >2 = poor fit)
   * - ``converged``
     - True / False
     - Whether the optimizer reached a solution

A ``chi2_reduced`` near 1.0 indicates a good fit. Values significantly
greater than 1 suggest the model or initial parameters need adjustment.

----

.. _qs-next-steps:

Next Steps
----------

- :doc:`user_guide/index` — structured learning pathways
- :doc:`user_guide/02_data_and_fitting/nlsq_fitting` — NLSQ in depth
- :doc:`user_guide/03_advanced_topics/bayesian_inference` — Bayesian uncertainty quantification
- :doc:`user_guide/03_advanced_topics/laminar_flow` — laminar flow mode
- :doc:`theory/index` — mathematical foundation
- :doc:`api/index` — full API reference

