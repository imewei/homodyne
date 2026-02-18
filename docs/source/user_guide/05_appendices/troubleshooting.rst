.. _troubleshooting:

Troubleshooting Guide
======================

This guide covers common problems and their solutions, organized by category.

---

Installation Issues
--------------------

**Problem: ``uv sync`` fails with dependency conflicts**

.. code-block:: text

   error: No solution found when resolving dependencies

Solution:

.. code-block:: bash

   # Remove existing lock file and recreate
   rm uv.lock
   uv sync

   # Or update a specific package
   uv add --upgrade jax

**Problem: JAX not found after installation**

.. code-block:: bash

   python -c "import jax; print(jax.__version__)"
   # ModuleNotFoundError: No module named 'jax'

Solution: Ensure you are running inside the virtual environment:

.. code-block:: bash

   uv run python -c "import jax; print(jax.__version__)"
   # Should print the JAX version

**Problem: ``homodyne`` command not found**

After installation, the CLI entry points are not on the PATH until the
virtual environment is activated. Use ``uv run``:

.. code-block:: bash

   uv run homodyne --help

Or activate the environment:

.. code-block:: bash

   source .venv/bin/activate
   homodyne --help

---

Data Loading Errors
--------------------

**Problem: ``XPCSDataFormatError: Dataset not found``**

.. code-block:: text

   XPCSDataFormatError: Dataset '/exchange/data' not found in /path/to/data.h5

Solution: Check the actual HDF5 structure:

.. code-block:: python

   import h5py
   with h5py.File("data.h5", "r") as f:
       f.visit(print)

Then update ``dataset_path`` in your YAML config to the correct path.

**Problem: ``ValueError: C2 array contains NaN values``**

NaN values in the correlation data indicate a problem in the upstream
reduction pipeline. Common causes:

1. Pixels with zero intensity (detector dead pixels)
2. Division by zero in the normalization step

Quick fix for exploratory analysis:

.. code-block:: python

   import numpy as np
   # Replace NaN with interpolated or background value
   c2 = data['c2_exp']
   c2_clean = np.where(np.isnan(c2), 1.0, c2)  # Replace NaN with background=1.0
   data['c2_exp'] = c2_clean

**Problem: ``ValueError: Time axis is not monotonically increasing``**

Solution: Sort the time arrays:

.. code-block:: python

   import numpy as np
   sort_idx = np.argsort(data['t1'])
   data['t1'] = data['t1'][sort_idx]
   data['t2'] = data['t2'][sort_idx]
   data['c2_exp'] = data['c2_exp'][:, sort_idx, :][:, :, sort_idx]

---

Fitting Convergence Failures
------------------------------

**Problem: ``convergence_status = "failed"`` immediately**

Most common causes:

1. **Parameter out of bounds**: initial value outside the configured bounds

   .. code-block:: python

      # Check: are initial values within bounds?
      for name, val in config.get_initial_parameters().items():
          lo, hi = config.get_parameter_bounds()[name]
          if not (lo <= val <= hi):
              print(f"PROBLEM: {name}={val} not in [{lo}, {hi}]")

2. **All-NaN data**: check data quality with ``validate_xpcs_data``

3. **Wrong analysis mode**: ``laminar_flow`` on equilibrium data with no
   angular dependence

**Problem: ``reduced_chi_squared > 10``**

The model cannot describe the data. Check:

1. **q-value**: is ``q_value`` in Å⁻¹ (not nm⁻¹ or m⁻¹)?

   .. code-block:: python

      # Typical XPCS q range: 0.01 – 0.5 Å⁻¹
      # If your q is in nm⁻¹: multiply by 0.1 to get Å⁻¹
      q_nm = 0.54    # nm⁻¹
      q_angstrom = q_nm * 0.1  # = 0.054 Å⁻¹

2. **gap_distance units**: should be in µm (homodyne converts internally)

3. **Wrong mode**: plot ``C2`` vs angle to decide static vs laminar_flow

**Problem: Parameters converge to unphysical values**

Symptoms: :math:`D_0 \gg 10^6` Å²/s; contrast near zero.

Solution: Enable anti-degeneracy:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"   # Was missing or "constant"

---

Memory Errors
--------------

**Problem: ``MemoryError`` or process killed (OOM)**

.. code-block:: text

   Killed
   # or
   MemoryError: Unable to allocate array

Solution: Reduce the memory threshold to trigger streaming earlier:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_fraction: 0.50   # Was 0.75; use 50% threshold

Or set an explicit limit:

.. code-block:: yaml

   optimization:
     nlsq:
       memory_threshold_gb: 16   # Trigger streaming above 16 GB

**Problem: CMC workers run out of memory**

Each worker holds one shard. If shard size is too large:

.. code-block:: yaml

   optimization:
     cmc:
       sharding:
         max_points_per_shard: "auto"   # ALWAYS use auto; never set a large number manually
       num_workers: 2   # Reduce workers to leave more memory per worker

---

JAX Compilation Issues
-----------------------

**Problem: Very long startup time (> 5 minutes)**

JAX is JIT-compiling the model on first use. This is normal for the first
run. To cache compilations:

.. code-block:: bash

   export XLA_CACHE_DIR="$HOME/.cache/xla"
   mkdir -p "$XLA_CACHE_DIR"

Subsequent runs load from cache and start in seconds.

**Problem: ``RuntimeError: JAX devices not available``**

If JAX cannot find any CPU devices:

.. code-block:: python

   import os
   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
   import jax
   print(jax.devices())

**Problem: JIT compilation produces wrong results on second call**

This is a common JAX pitfall: JAX JIT traces on the first call and reuses
the trace. If the trace depends on Python control flow with different
values, subsequent calls may silently use the wrong code path.

Solution: Use ``jax.jit`` with static arguments marked correctly, or
re-read the JAX documentation on tracing and static values.

---

CMC Divergence Issues
----------------------

**Problem: > 25% divergence rate despite NLSQ warm-start**

Steps to diagnose:

1. **Check that NLSQ warm-start is actually being used:**

   .. code-block:: python

      cmc_result = fit_mcmc_jax(
          ...,
          nlsq_result=nlsq_result,  # Ensure this is not None
      )

2. **Increase max_tree_depth:**

   .. code-block:: yaml

      optimization:
        cmc:
          per_shard_mcmc:
            max_tree_depth: 12   # Increase from 10

3. **Use informative priors:**

   .. code-block:: yaml

      parameter_space:
        D0:
          prior: "normal"
          prior_mean: 1245.0   # From NLSQ result
          prior_std: 50.0

4. **Increase shard size** (too-small shards have noisy posteriors):

   .. code-block:: yaml

      optimization:
        cmc:
          sharding:
            max_points_per_shard: "auto"   # Uses optimal size

**Problem: CMC result differs greatly from NLSQ (> 20%)**

Possible causes:

1. **Degeneracy**: NLSQ found a local minimum; CMC found the global posterior.
   Use ``az.plot_pair`` to visualize multi-modality.

2. **Bad per_angle_mode mismatch**: ensure NLSQ and CMC use the same mode.

3. **Insufficient shard size**: too few points per shard → data-starved posteriors.

---

Shell Completion Issues
------------------------

**Problem: Tab completion not working after ``homodyne-post-install``**

.. code-block:: bash

   # Reload shell configuration
   source ~/.zshrc   # or ~/.bashrc

   # Re-run post-install if needed
   homodyne-post-install --force

**Problem: Aliases (hm, hconfig, etc.) not found**

.. code-block:: bash

   # The venv must be activated
   source .venv/bin/activate

   # Check that activation sources homodyne completion
   grep -n homodyne ~/.zshrc

---

Getting Help
------------

If your problem is not listed here:

1. Check the :ref:`faq` for quick answers
2. Look at the :ref:`glossary` for terminology
3. Run with verbose logging: ``homodyne --verbose --config config.yaml``
4. Check the structured log output in the console for specific error messages

---

See Also
---------

- :ref:`faq` — Frequently asked questions
- :doc:`../03_advanced_topics/diagnostics` — Convergence diagnostics
- :doc:`../04_practical_guides/performance_tuning` — Memory optimization
