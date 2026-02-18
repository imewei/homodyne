.. _faq:

Frequently Asked Questions
==========================

This page answers common questions about using homodyne.

---

General Questions
------------------

**Q: What input data does homodyne require?**

Homodyne requires a two-time correlation matrix :math:`C_2(t_1, t_2)` stored
in an HDF5 file. This must already be computed by upstream XPCS reduction
software. Homodyne does not compute :math:`C_2` from raw detector frames.

**Q: What formats are supported?**

HDF5 files in APS legacy format (``/exchange/`` group) and APS-U new format.
See :doc:`../02_data_and_fitting/data_loading` for details.

**Q: Does homodyne support GPU acceleration?**

No. Homodyne is CPU-only by design. CPU clusters are universally available
on HPC systems, avoid CUDA version issues, and handle typical XPCS dataset
sizes efficiently with JAX JIT compilation.

**Q: Can I use homodyne with DLS (light scattering) data?**

Technically yes: the underlying model equations apply to any homodyne
correlation measurement. However, homodyne was designed for XPCS two-time
correlations in HDF5 format. DLS data would need to be reformatted to the
expected array structure.

---

Analysis Mode Selection
------------------------

**Q: Should I use static or laminar_flow mode?**

Use static mode for equilibrium samples (no shear, no flow). Use laminar_flow
mode when the sample is in a Couette shear cell and you observe an angular
dependence in :math:`C_2`. See :doc:`../02_data_and_fitting/model_selection`
for a detailed decision guide.

**Q: My laminar_flow fit gives unphysical D₀. What is wrong?**

This is almost always a degeneracy problem. Enable:

.. code-block:: yaml

   optimization:
     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"

See :doc:`../03_advanced_topics/per_angle_modes` for a full explanation.

**Q: What is the difference between per_angle_mode "auto" and "constant"?**

``constant`` estimates per-angle contrast/offset via quantile analysis and
**fixes** them (not optimized). ``auto`` also estimates them but optimizes
2 **averaged** values alongside the physical parameters. Use ``auto`` to prevent
degeneracy. See :doc:`../03_advanced_topics/per_angle_modes`.

---

NLSQ Optimization
------------------

**Q: How do I know if the NLSQ fit converged?**

Check ``result.convergence_status`` (should be ``"converged"``) and
``result.reduced_chi_squared`` (should be 0.5–3.0 for a good fit).
See :doc:`../02_data_and_fitting/result_interpretation`.

**Q: NLSQ takes a very long time. How do I speed it up?**

1. Check that ``per_angle_mode`` is ``"auto"`` not ``"individual"`` (reduces params)
2. For large datasets, confirm streaming is active (check log for ``Strategy: hybrid_streaming``)
3. Enable XLA compilation caching (run twice; second run is faster)
4. Set ``OMP_NUM_THREADS=2`` and configure XLA via ``homodyne-config-xla``

**Q: Should I use ``use_adapter=True`` in fit_nlsq_jax?**

The ``NLSQAdapter`` (``use_adapter=True``) is experimental and uses a CurveFit
class for better JIT caching. Use the default (``False``) for production work.

**Q: When should I use CMA-ES?**

When standard NLSQ converges to unphysical solutions on laminar flow data
where :math:`D_0 / \dot\gamma_0 > 1000`. Enable:

.. code-block:: yaml

   optimization:
     nlsq:
       cmaes:
         enable: true
         preset: "cmaes-global"
         refine_with_nlsq: true

---

CMC / Bayesian Questions
--------------------------

**Q: When should I use CMC instead of (or in addition to) NLSQ?**

Use CMC when you need publication-quality uncertainty estimates, want to
check for multi-modal posteriors, or need to propagate parameter uncertainties
into derived quantities. Always run NLSQ first and pass the result as a
warm-start.

**Q: How long does CMC take?**

For a 1M-point laminar_flow dataset with ``auto`` per-angle mode on a 16-core
machine: approximately 2–4 hours. Static mode on the same dataset: 30–60 minutes.
Computation scales with:

- Number of shards (proportional to data size)
- Number of chains × (warmup + samples) per shard
- Analysis mode complexity (laminar_flow ~ 2× static)

**Q: My CMC result differs significantly from NLSQ (> 20%). Which is correct?**

Differences arise from three sources:

1. NLSQ found a local minimum; CMC explored the full posterior
2. Degeneracy in one or both methods
3. Per-angle mode mismatch between NLSQ and CMC

Check that both methods use the same ``per_angle_mode``. Use ArviZ pair plots
to visualize the posterior and identify multi-modality.

**Q: What does a divergence rate > 10% mean?**

More than 10% of NUTS transitions left the high-probability region of the
posterior. The CMC quality filter rejects such shards. Causes include:

- Missing NLSQ warm-start (use ``nlsq_result=nlsq_result``)
- Priors too broad relative to the likelihood
- ``max_tree_depth`` too low (increase to 12)
- Data quality issues (outliers, bad shards)

**Q: What is ``max_points_per_shard: "auto"`` and why must I use it?**

The ``auto`` setting uses an algorithm that accounts for dataset size, angle count,
and iteration count to choose shard sizes that balance NUTS accuracy against
computation time. Manual values can cause timeouts (too large) or data starvation
(too small). Always use ``"auto"``.

**Q: What is the SamplingPlan and why should I use it?**

``SamplingPlan.from_config()`` returns the **actual** warmup/samples after
adaptive scaling for small shards. The raw ``config.num_warmup`` / ``config.num_samples``
are the pre-adaptation defaults. Code that reads the config directly may use wrong
values for small shards. See :doc:`../03_advanced_topics/bayesian_inference`.

---

Configuration Questions
------------------------

**Q: How do I generate a configuration template?**

.. code-block:: bash

   homodyne-config --mode static --output config_static.yaml
   homodyne-config --mode laminar_flow --output config_flow.yaml

**Q: How do I validate my configuration before a long run?**

.. code-block:: bash

   homodyne-config --validate --input my_config.yaml

**Q: The CLI settings override my ``cmc.per_shard_mcmc`` settings. Why?**

The CLI applies base ``optimization.mcmc`` settings to ``per_shard_mcmc``.
Keep them aligned in your YAML. See :doc:`../04_practical_guides/configuration`.

**Q: What units should gap_distance be in?**

``gap_distance`` in the YAML is in **µm** (micrometers). Homodyne converts
to Å internally (1 µm = 10⁴ Å). A typical Couette cell gap of 0.5 mm is
entered as ``gap_distance: 500.0`` (500 µm).

---

Results Questions
------------------

**Q: My reduced chi-squared is 0.1. Is this good?**

A very low chi-squared (< 0.5) usually means the uncertainties (sigma array)
are overestimated, not that the fit is unusually good. If you are using the
default sigma = 0.01 × ones, the absolute chi-squared value is not interpretable;
only relative comparisons between models matter.

**Q: How do I save the posterior samples from CMC?**

.. code-block:: python

   # Save as ArviZ NetCDF (recommended)
   cmc_result.inference_data.to_netcdf("posterior.nc")

   # Reload later
   import arviz as az
   idata = az.from_netcdf("posterior.nc")

**Q: How do I compare NLSQ and CMC uncertainties?**

.. code-block:: python

   import numpy as np

   nlsq_std = result.uncertainties
   cmc_std = cmc_result.uncertainties

   for name, n_std, c_std in zip(param_names, nlsq_std, cmc_std):
       ratio = c_std / n_std if n_std > 0 else float('inf')
       print(f"{name}: NLSQ={n_std:.3g}, CMC={c_std:.3g}, ratio={ratio:.2f}")

Ratios > 2 indicate that NLSQ underestimates the uncertainty (non-Gaussian posterior).

---

See Also
---------

- :ref:`glossary` — Terminology reference
- :ref:`troubleshooting` — Problem-specific fixes
- :doc:`../02_data_and_fitting/nlsq_fitting` — NLSQ workflow
- :doc:`../03_advanced_topics/bayesian_inference` — CMC workflow
