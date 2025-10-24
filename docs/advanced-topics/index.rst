Advanced Topics
===============

This section covers advanced features and workflows for power users, including streaming optimization, GPU acceleration, and large dataset handling.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   nlsq-optimization
   mcmc-uncertainty
   cmc-large-datasets
   streaming-optimization
   gpu-acceleration
   angle-filtering

Overview
--------

Advanced topics cover specialized workflows and features that go beyond basic usage:

* **NLSQ Optimization**: Trust-region optimization with strategy selection
* **MCMC Uncertainty**: Bayesian inference for uncertainty quantification
* **CMC (Covariance Matrix Combination)**: Parallel optimization for large datasets
* **Streaming Optimization**: Handle 100M+ data points with constant memory
* **GPU Acceleration**: Setup and optimization for GPU-accelerated computing
* **Angle Filtering**: Phi angle selection for targeted analysis

When to Use These Features
---------------------------

**NLSQ Optimization**
   Primary optimization method for all analyses. Strategy automatically selected based on dataset size.

**MCMC Uncertainty**
   When you need rigorous uncertainty quantification and posterior distributions.

**CMC (Covariance Matrix Combination)**
   For datasets > 1M points with multiple phi angles. Enables parallel per-angle optimization.

**Streaming Optimization**
   For datasets > 100M points. Provides constant memory footprint with checkpoint/resume capability.

**GPU Acceleration**
   When running on Linux systems with NVIDIA GPUs (CUDA 12.1-12.9). Provides 10-100× speedup.

**Angle Filtering**
   To reduce parameter count (3+2n or 7+2n) by focusing on specific angular regimes.

Prerequisites
-------------

Most advanced topics assume you have:

* Completed :doc:`../user-guide/quickstart`
* Familiarity with :doc:`../user-guide/configuration`
* Understanding of :doc:`../theoretical-framework/parameter-models`

See Also
--------

* :doc:`../api-reference/optimization` - Optimization API reference
* :doc:`../developer-guide/performance` - Performance tuning guide
* :doc:`../configuration-templates/index` - Configuration templates
