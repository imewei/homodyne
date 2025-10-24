Theoretical Framework
=====================

This section documents the theoretical physics foundation of Homodyne, based on the framework from He et al. PNAS 2024.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   core-equations
   transport-coefficients
   parameter-models

Overview
--------

Homodyne implements the transport coefficient approach for characterizing nonequilibrium dynamics in soft matter systems using X-ray Photon Correlation Spectroscopy (XPCS).

**Core Concept:**

The second-order correlation function :math:`c_2(\vec{q}, t_1, t_2)` captures both Brownian diffusion and advective flow effects through time-dependent transport coefficients:

* :math:`D(t)` - Time-dependent diffusion coefficient
* :math:`\dot{\gamma}(t)` - Time-dependent shear rate
* :math:`\phi(t)` - Angle between flow direction and scattering vector

Reference
---------

   He, H., Chen, W., et al. (2024). Transport coefficient approach for characterizing nonequilibrium dynamics in soft matter. *PNAS*, 121(31), e2401162121. https://doi.org/10.1073/pnas.2401162121

Sections
--------

**Core Equations**
   Mathematical formulation of the correlation functions (Equations 13, S-75, S-76)

**Transport Coefficients**
   Parameterization of time-dependent transport properties

**Parameter Models**
   Static isotropic (3+2n) and laminar flow (7+2n) parameter models

See Also
--------

* :doc:`../api-reference/core` - Core physics API
* :doc:`../user-guide/configuration` - Configuration system
* :doc:`../configuration-templates/index` - Configuration templates
