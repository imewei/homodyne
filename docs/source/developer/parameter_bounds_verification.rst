.. _parameter_bounds_verification:

Parameter Bounds Codebase Verification
=======================================

.. rubric:: Audit Date: 2026-03-03

This document records a comprehensive code-level verification of all default
parameter bounds across the homodyne codebase. It serves as a reference for
future audits and as the authoritative cross-system consistency check.

Verified Source Files
----------------------

The following source files define or consume parameter bounds:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - File
     - Role
   * - ``config/parameter_registry.py:115-226``
     - Canonical source (singleton ``ParameterInfo`` entries)
   * - ``config/parameter_manager.py:91-151``
     - ``_default_bounds`` dict (merged with YAML overrides)
   * - ``core/physics.py:56-158``
     - ``PhysicsConstants`` class + ``parameter_bounds()``
   * - ``core/fitting.py:84-98``
     - ``ParameterSpace`` dataclass defaults
   * - ``core/models.py:189-558``
     - ``DiffusionModel/ShearModel/CombinedModel.get_parameter_bounds()``
   * - ``config/physics_validators.py:56-144``
     - Soft post-fit validation rules (not optimizer constraints)
   * - ``config/templates/homodyne_static.yaml:162-197``
     - Static mode YAML template
   * - ``config/templates/homodyne_laminar_flow.yaml:174-242``
     - Laminar flow YAML template

Verified Bounds (All Sources Consistent)
-----------------------------------------

.. list-table:: Physical Parameter Bounds
   :header-rows: 1
   :widths: 18 10 10 10 10 42

   * - Parameter
     - Min
     - Max
     - Units
     - Log-space
     - Modes
   * - ``D0``
     - 100.0
     - 1e5
     - Angstrom^2/s
     - yes
     - static, laminar_flow
   * - ``alpha``
     - -2.0
     - 2.0
     - --
     - no
     - static, laminar_flow
   * - ``D_offset``
     - -1e5
     - 1e5
     - Angstrom^2/s
     - yes
     - static, laminar_flow
   * - ``gamma_dot_t0``
     - 1e-6
     - 0.5
     - 1/s
     - yes
     - laminar_flow
   * - ``beta``
     - -2.0
     - 2.0
     - --
     - no
     - laminar_flow
   * - ``gamma_dot_t_offset``
     - -0.1
     - 0.1
     - 1/s
     - no
     - laminar_flow
   * - ``phi0``
     - -10.0
     - 10.0
     - degrees
     - no
     - laminar_flow

.. list-table:: Scaling Parameter Bounds
   :header-rows: 1
   :widths: 18 10 10 10 10 42

   * - Parameter
     - Min
     - Max
     - Units
     - Log-space
     - Notes
   * - ``contrast``
     - 0.0
     - 1.0
     - --
     - no
     - Per-angle; applied to all ``contrast_i``
   * - ``offset``
     - 0.5
     - 1.5
     - --
     - no
     - Per-angle; applied to all ``offset_i``

Registry Defaults and Priors
-----------------------------

From ``parameter_registry.py:115-226``:

.. list-table::
   :header-rows: 1
   :widths: 20 12 14 14 40

   * - Parameter
     - Default
     - Prior Mean
     - Prior Std
     - Notes
   * - ``D0``
     - 1000.0
     - 1000.0
     - 1000.0
     - ``log_space=True``
   * - ``alpha``
     - 0.5
     - 0.5
     - 0.5
     - Default != prior_mean for ``beta`` (see below)
   * - ``D_offset``
     - 10.0
     - 10.0
     - 200.0
     - ``log_space=True``, allows negative
   * - ``gamma_dot_t0``
     - 0.01
     - 0.01
     - 0.1
     - ``log_space=True``
   * - ``beta``
     - 0.5
     - 0.0
     - 0.5
     - Default (NLSQ init) differs from prior_mean (CMC center)
   * - ``gamma_dot_t_offset``
     - 0.0
     - 0.0
     - 0.02
     - ``log_space=False`` (bounds include negative)
   * - ``phi0``
     - 0.0
     - 0.0
     - 5.0
     - --
   * - ``contrast``
     - 0.5
     - 0.5
     - 0.25
     - --
   * - ``offset``
     - 1.0
     - 1.0
     - 0.25
     - --

YAML Template Prior Overrides
-------------------------------

The YAML templates override registry priors to reflect physics expectations per
analysis mode. This is intentional.

.. list-table::
   :header-rows: 1
   :widths: 22 18 18 42

   * - Parameter
     - Static Template
     - Laminar Template
     - Rationale
   * - ``alpha`` prior_mu
     - -1.2
     - 0.5
     - Static: subdiffusion expected; Laminar: superdiffusion under shear
   * - ``alpha`` prior_sigma
     - 0.3
     - 0.5
     - Static: tighter prior (less exploration needed)
   * - ``D_offset`` prior_mu
     - 0.0
     - 10.0
     - Static: no baseline expected; Laminar: small baseline common
   * - ``D_offset`` prior_sigma
     - 150.0
     - 200.0
     - Laminar: wider prior for more diverse systems
   * - ``contrast`` prior (CMC)
     - mu=0.06, sigma=0.05, max=0.5
     - mu=0.27, sigma=0.15, max=1.0
     - Static: lower contrast typical; Laminar: higher contrast
   * - ``offset`` prior (CMC)
     - mu=1.0, sigma=0.1, range=[0.8, 1.2]
     - mu=1.0, sigma=0.15, range=[0.5, 1.5]
     - Static: tighter offset prior

NLSQ Bounds Flow
------------------

The NLSQ optimizer receives bounds as numpy arrays through this chain::

    YAML config
      -> ParameterManager._load_config_bounds()  [dict.update merge]
        -> get_bounds_as_arrays()                 [np.ndarray]
          -> core.py:_bounds_to_arrays()
            -> wrapper._convert_bounds()          [validation: lower <= upper]
              -> NLSQ optimizer

**Static mode** (5 optimizer params)::

    Order: [contrast, offset, D0, alpha, D_offset]
    Lower: [0.0,  0.5,  100,   -2.0,  -1e5]
    Upper: [1.0,  1.5,  1e5,    2.0,   1e5]

**Laminar flow mode** (9 optimizer params, auto_averaged)::

    Order: [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
    Lower: [0.0,  0.5,  100,   -2.0,  -1e5,  1e-6,   -2.0,  -0.1,  -10.0]
    Upper: [1.0,  1.5,  1e5,    2.0,   1e5,   0.5,     2.0,   0.1,   10.0]

**NLSQ transforms** (``nlsq/transforms.py:267-304``):

- ``gamma_dot_t0`` log-transform when ``enable_gamma_dot_log=True``:
  ``[1e-6, 0.5]`` becomes ``[-13.816, -0.693]`` (natural log)
- ``beta`` centering when ``enable_beta_center=True``:
  shifted by ``beta_reference`` (default 0.0, no change)

CMC Prior Construction
-----------------------

CMC uses z-space reparameterization for all parameters
(``cmc/scaling.py:227-268``):

**Without NLSQ warm-start**::

    Physical params: z ~ Normal(0, sqrt(K)) -> smooth sigmoid to [lower, upper]
    sigma (noise):   HalfNormal(scale = noise_scale * 1.5 * sqrt(K))

    where K = number of shards (prior tempering)

**With NLSQ warm-start** (recommended)::

    Physical params: TruncatedNormal(
        loc   = nlsq_value,
        scale = clip(nlsq_std * 2.0, 1% .. 50% of range),
        low   = bound_min,
        high  = bound_max
    )
    sigma: HalfNormal(scale = noise_scale * 1.5 * sqrt(K))

NLSQ-informed priors are **not** tempered (``model.py:574-577``).

**Per-angle scaling priors** depend on ``per_angle_mode``:

- ``auto`` (default, n_phi >= 3): 2 averaged params via z-space Normal
- ``individual`` (n_phi < 3): 2 * n_phi params via z-space Normal
- ``constant``: fixed from quantile estimation, not sampled

Physics Validators (Soft Constraints)
--------------------------------------

Post-fit rules from ``physics_validators.py:56-144``. These emit warnings/errors
after optimization; they do not constrain the optimizer itself.

.. list-table::
   :header-rows: 1
   :widths: 22 25 53

   * - Parameter
     - Severity
     - Rule
   * - ``D0``
     - ERROR
     - v <= 0
   * - ``D0``
     - WARNING
     - v > 1e7
   * - ``alpha``
     - WARNING
     - v < -1.5 or v > 1.0
   * - ``alpha``
     - INFO
     - \|v\| < 0.1 (near-Brownian)
   * - ``D_offset``
     - WARNING
     - v < 0
   * - ``gamma_dot_t0``
     - ERROR
     - v < 0
   * - ``gamma_dot_t0``
     - WARNING
     - v > 0.5
   * - ``gamma_dot_t0``
     - INFO
     - 0 < v < 1e-6 (quasi-static)
   * - ``beta``
     - WARNING
     - \|v\| > 2
   * - ``gamma_dot_t_offset``
     - WARNING
     - \|v\| > 0.1
   * - ``phi0``
     - INFO
     - \|v\| > 10
   * - ``contrast``
     - ERROR
     - v <= 0 or v > 1
   * - ``contrast``
     - WARNING
     - 0 < v < 0.1 (low signal)
   * - ``offset``
     - ERROR
     - v <= 0

Mode Selection Logic
---------------------

From ``parameter_registry.py:229-241``::

    "static":           ["D0", "alpha", "D_offset"]                    # 3 params
    "static_isotropic": ["D0", "alpha", "D_offset"]                    # 3 params
    "laminar_flow":     ["D0", "alpha", "D_offset",
                         "gamma_dot_t0", "beta",
                         "gamma_dot_t_offset", "phi0"]                 # 7 params

With per-angle scaling in ``auto`` mode (n_phi >= 3):

- **NLSQ**: 7 physical + 2 averaged scaling = **9 optimized params**
- **CMC**: 7 physical + 2 averaged scaling + 1 sigma = **10 sampled params**

Documentation Corrections Applied (2026-03-03)
------------------------------------------------

This audit identified and corrected the following stale values in the Sphinx
documentation:

**parameter_guide.rst:**

- ``gamma_dot_t0`` bounds: ``[1e-6, 1e4]`` corrected to ``[1e-6, 0.5]``
- ``gamma_dot_t_offset`` bounds: ``[0.01, 100]`` corrected to ``[-0.1, 0.1]``
- Parameter Reference Table: 6 default init values corrected
  (D0: 50050 -> 1000.0, alpha: 0.0 -> 0.5, D_offset: 0.0 -> 10.0,
  gamma_dot_t0: 5000 -> 0.01, gamma_dot_t_offset: 50.005 -> 0.0,
  parameter names: ``gamma_dot_0`` -> ``gamma_dot_t0``, ``phi_0`` -> ``phi0``)
- Cross-System Verification Table: gamma_dot_t0 and gamma_dot_t_offset
  bounds corrected across all columns
- Log-space note: removed ``gamma_dot_t_offset`` from log-space list

**configuration/options.rst:**

- Laminar flow ``gamma_dot_t0`` bounds: ``[0.001, 1000]`` corrected to
  ``[1e-6, 0.5]``
- Laminar flow ``gamma_dot_t_offset`` bounds: ``[-1000, 1000]`` corrected to
  ``[-0.1, 0.1]``
- Laminar flow ``phi0`` bounds: ``[-180, 180]`` corrected to ``[-10, 10]``

**configuration/templates.rst:**

- Laminar flow inline template: all 6 shear/flow parameter bounds and priors
  corrected to match the actual YAML template file
  (``homodyne_laminar_flow.yaml``)
- Static inline template: ``priors: null`` replaced with actual prior
  definitions from the template file
