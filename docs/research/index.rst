Research Documentation
======================

This section provides research-grade documentation for the Homodyne package,
including theoretical foundations, computational methods, and analysis modes
for X-ray Photon Correlation Spectroscopy (XPCS) under nonequilibrium conditions.

.. toctree::
   :maxdepth: 2
   :caption: Research Content

   theoretical_framework
   analysis_modes
   computational_methods
   citations

Overview
--------

The Homodyne package implements the theoretical framework for analyzing XPCS data
under nonequilibrium conditions, as described in:

**He, H., Liang, H., Chu, M., Jiang, Z., de Pablo, J.J., Tirrell, M.V., Narayanan, S., & Chen, W.**
*Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter.*
Proceedings of the National Academy of Sciences, 121(31), e2401162121 (2024).
DOI: `10.1073/pnas.2401162121 <https://doi.org/10.1073/pnas.2401162121>`_

Research Scope
--------------

Scientific Applications
~~~~~~~~~~~~~~~~~~~~~~~

* **Soft Matter Physics**: Colloidal dynamics, active matter, biological systems
* **Flow Rheology**: Shear thinning/thickening, microfluidics, complex fluids
* **Materials Science**: Phase transitions, glass transition, crystallization

Computational Framework
~~~~~~~~~~~~~~~~~~~~~~~

* **JAX-First Architecture**: JIT-compiled computational kernels for high performance
* **Robust Optimization**: NLSQ trust-region and MCMC inference with uncertainty quantification
* **CPU-Optimized**: Efficient multi-core utilization for HPC environments

Experimental Integration
~~~~~~~~~~~~~~~~~~~~~~~~

* **Synchrotron Facilities**: Advanced Photon Source (APS) data format support
* **Data Standards**: HDF5 and standardized XPCS data formats
* **Validation Protocols**: Statistical validation and uncertainty analysis

Core Mathematical Framework
---------------------------

The package analyzes time-dependent intensity correlation functions:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \beta \left[ c_1(\phi, t_1, t_2) \right]^2

where:

* :math:`c_2`: Two-time intensity correlation function
* :math:`c_1`: Two-time field correlation function
* :math:`\beta`: Instrumental contrast parameter (per-angle in v2.4.0+)
* :math:`\phi`: Azimuthal angle between flow direction and scattering wavevector

The field correlation function captures both diffusive and advective contributions:

.. math::

   c_1(\phi, t_1, t_2) = e^{-q^2 J(t_1, t_2)} \times \text{sinc}\left[\frac{qh}{2\pi} \Gamma(t_1, t_2) \cos(\phi - \phi_0)\right]

where :math:`J(t_1, t_2)` is the diffusion integral and :math:`\Gamma(t_1, t_2)` is the shear integral.

Analysis Capabilities
---------------------

.. list-table:: Analysis Modes
   :header-rows: 1
   :widths: 20 15 45 20

   * - Mode
     - Parameters
     - Physical Description
     - Applications
   * - Static
     - 3
     - Anomalous diffusion without flow
     - Equilibrium systems
   * - Laminar Flow
     - 7
     - Full nonequilibrium with shear
     - Flowing systems

See :doc:`analysis_modes` for detailed parameter descriptions and usage.

Version Information
-------------------

This documentation covers Homodyne v2.5.x with the following key features:

* **v2.5.0**: Memory-bounded streaming optimizer for large datasets (>10M points)
* **v2.4.3**: NLSQ element-wise integration fix (matches CMC physics)
* **v2.4.1**: CMC-only MCMC architecture, module reorganization
* **v2.4.0**: Per-angle scaling mandatory, legacy scalar mode removed
* **v2.3.0**: CPU-only architecture (GPU support removed for maintenance simplicity)

For migration guidance, see the project repository documentation.
