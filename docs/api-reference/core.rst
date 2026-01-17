Core Module
===========

The :mod:`homodyne.core` module provides the foundational computational backend for homodyne scattering analysis, implementing JAX-accelerated physics models and mathematical operations.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

The core module is organized around several key components:

- **jax_backend**: JIT-compiled computational primitives for G1/G2 calculations
- **physics**: Time-dependent transport coefficients and physical constants
- **models**: Diffusion, shear, and combined physical models
- **fitting**: Scaled fitting engine with contrast and offset parameters
- **theory**: Theoretical computation of chi-squared and G2 functions

All computations leverage JAX for automatic differentiation and JIT compilation, providing optimal performance on CPU architectures.

Module Contents
---------------

.. automodule:: homodyne.core
   :noindex:

JAX Backend
-----------

JAX-accelerated computational primitives for correlation function calculations.

.. automodule:: homodyne.core.jax_backend
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.core.jax_backend.compute_g1_diffusion
   homodyne.core.jax_backend.compute_g1_shear
   homodyne.core.jax_backend.compute_g2_scaled
   homodyne.core.jax_backend.compute_g2_scaled_with_factors
   homodyne.core.jax_backend.gradient_g2
   homodyne.core.jax_backend.hessian_g2

Physics Models
--------------

Physical models implementing diffusion and shear dynamics.

.. automodule:: homodyne.core.models
   :members:
   :undoc-members:
   :show-inheritance:

Model Classes
~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.core.models.PhysicsModelBase
   homodyne.core.models.DiffusionModel
   homodyne.core.models.ShearModel
   homodyne.core.models.CombinedModel

Model Mixins (v2.9.0)
~~~~~~~~~~~~~~~~~~~~~

Reusable mixins for gradient computation and benchmarking capabilities.

.. automodule:: homodyne.core.model_mixins
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
"""""""""""

.. autosummary::
   :nosignatures:

   homodyne.core.model_mixins.GradientCapabilityMixin
   homodyne.core.model_mixins.BenchmarkingMixin
   homodyne.core.model_mixins.OptimizationRecommendationMixin

Backend API
~~~~~~~~~~~

Abstract base class for physics backend implementations.

.. automodule:: homodyne.core.backend_api
   :members:
   :undoc-members:
   :show-inheritance:

Homodyne Model
--------------

High-level model interface for homodyne scattering analysis.

.. automodule:: homodyne.core.homodyne_model
   :members:
   :undoc-members:
   :show-inheritance:

Physics Utilities
-----------------

Physical constants, parameter validation, and transport coefficients.

.. automodule:: homodyne.core.physics
   :members:
   :undoc-members:
   :show-inheritance:

Time-Dependent Transport Coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The physics module implements the core time-dependent parameterizations:

**Diffusion Coefficient**:

.. math::

   D(t) = D_0 \cdot t^\alpha + D_{\text{offset}}

**Shear Rate** (laminar flow mode):

.. math::

   \dot{\gamma}(t) = \dot{\gamma}_0 \cdot t^\beta + \dot{\gamma}_{\text{offset}}

Physics Factors
---------------

Factory functions for creating physics factors from configuration.

.. automodule:: homodyne.core.physics_factors
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Theory Engine
-------------

Theoretical computation of correlation functions and chi-squared objective.

.. automodule:: homodyne.core.theory
   :members:
   :undoc-members:
   :show-inheritance:

Fitting Engine
--------------

Scaled fitting engine implementing the core homodyne equation:

.. math::

   c_2(\phi, t_1, t_2) = 1 + \text{contrast} \times [c_1(\phi, t_1, t_2)]^2

.. automodule:: homodyne.core.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Shared Physics Utilities
------------------------

Common utility functions and physics helpers shared across backends.

.. automodule:: homodyne.core.physics_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Shared Functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.core.physics_utils.safe_len
   homodyne.core.physics_utils.safe_exp
   homodyne.core.physics_utils.safe_sinc
   homodyne.core.physics_utils.calculate_diffusion_coefficient
   homodyne.core.physics_utils.calculate_shear_rate
   homodyne.core.physics_utils.calculate_shear_rate_cmc
   homodyne.core.physics_utils.create_time_integral_matrix
   homodyne.core.physics_utils.trapezoid_cumsum
   homodyne.core.physics_utils.apply_diagonal_correction

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The physics_utils module implements several numerical stability techniques:

- **Overflow protection**: ``safe_exp`` clips values to prevent overflow
- **Division by zero**: ``safe_sinc`` handles x→0 limit smoothly
- **Singular exponents**: ``calculate_diffusion_coefficient`` and ``calculate_shear_rate`` handle t=0 with negative exponents
- **Smooth gradients**: ``create_time_integral_matrix`` uses sqrt(x²+ε) instead of abs(x) for differentiability

These techniques ensure robust optimization with JAX autodiff.

Specialized Physics Modules
---------------------------

The core module includes specialized physics implementations for different optimization methods:

- **physics_utils.py**: Shared utilities consolidated from multiple backends (diffusion coefficient, shear rate, time integrals)
- **physics_nlsq.py**: Physics computations optimized for NLSQ (meshgrid operations)
- **physics_cmc.py**: Physics computations optimized for CMC (element-wise operations)
- **numpy_gradients.py**: NumPy-based gradient calculations for validation

These modules share common functions via ``physics_utils.py`` to ensure consistent behavior across backends.

NLSQ Physics Backend
~~~~~~~~~~~~~~~~~~~~

Meshgrid physics computations for NLSQ optimization.

.. automodule:: homodyne.core.physics_nlsq
   :members:
   :undoc-members:
   :show-inheritance:

Key Design Principles
^^^^^^^^^^^^^^^^^^^^^

- Meshgrid computations for 2D correlation matrices
- Direct JIT-compiled functions only
- Completely independent from CMC physics

CMC Physics Backend
~~~~~~~~~~~~~~~~~~~

Element-wise physics computations for Consensus Monte Carlo.

.. automodule:: homodyne.core.physics_cmc
   :members:
   :undoc-members:
   :show-inheritance:

Key Design Principles
^^^^^^^^^^^^^^^^^^^^^

- Element-wise computations for paired (t1[i], t2[i]) arrays
- Prevents NumPyro NUTS from compiling unused meshgrid branches
- Eliminates memory allocation issues during CMC MCMC

NumPy Gradients (Fallback)
~~~~~~~~~~~~~~~~~~~~~~~~~~

NumPy-based numerical differentiation for JAX-free operation.

.. automodule:: homodyne.core.numpy_gradients
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
^^^^^^^^^^^^

- Multi-method finite differences: Forward, backward, central, complex-step
- Adaptive step size selection with Richardson extrapolation
- Drop-in replacement for JAX ``grad()`` and ``hessian()`` functions
- Memory-aware chunking for large parameter spaces

Diagonal Correction
~~~~~~~~~~~~~~~~~~~

Unified diagonal correction for two-time correlation matrices.

.. automodule:: homodyne.core.diagonal_correction
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.core.diagonal_correction.apply_diagonal_correction
   homodyne.core.diagonal_correction.apply_diagonal_correction_batch

Usage Example
^^^^^^^^^^^^^

.. code-block:: python

   from homodyne.core.diagonal_correction import apply_diagonal_correction

   # Single matrix (auto-detect backend)
   c2_corrected = apply_diagonal_correction(c2_matrix)

   # Force specific method and backend
   c2_corrected = apply_diagonal_correction(
       c2_matrix, method="statistical", backend="numpy"
   )

See Also
--------

- :mod:`homodyne.optimization` - Optimization methods (NLSQ, MCMC)
- :mod:`homodyne.config` - Configuration and parameter management
- :mod:`homodyne.data` - Data loading and preprocessing
