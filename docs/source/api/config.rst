.. _api-config:

==================
homodyne.config
==================

The ``homodyne.config`` package provides the configuration infrastructure:
YAML/JSON file loading (``ConfigManager``), parameter bounds and prior
distributions (``ParameterSpace``), and the underlying typed registries
(``ParameterManager``, ``ParameterRegistry``).

----

.. _api-config-manager:

ConfigManager
-------------

Minimal, API-compatible configuration manager. Loads YAML or JSON files and
exposes the parsed dictionary through the ``.config`` attribute.

.. autoclass:: homodyne.config.manager.ConfigManager
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.config.manager import ConfigManager

   # Load from file
   config_manager = ConfigManager("my_config.yaml")
   config = config_manager.config

   # Access nested values
   analysis_mode = config["analysis"]["mode"]           # "static" or "laminar_flow"
   num_warmup    = config["optimization"]["mcmc"]["num_warmup"]

   # Override specific keys programmatically
   config_manager2 = ConfigManager(
       config_override={"analysis": {"mode": "static"}}
   )

----

ParameterSpace
--------------

Loads parameter bounds and prior distributions from the ``parameter_space``
section of the YAML config. Used by both NLSQ (for bounds) and CMC (for
prior construction).

.. autoclass:: homodyne.config.parameter_space.ParameterSpace
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.config.manager import ConfigManager
   from homodyne.config.parameter_space import ParameterSpace

   config_manager = ConfigManager("config.yaml")
   param_space = ParameterSpace.from_config(config_manager.config)

   # Inspect bounds for D0
   d0_min, d0_max = param_space.get_bounds("D0")
   print(f"D0 bounds: [{d0_min:.2g}, {d0_max:.2g}]")

   # Get ordered bounds tuple for NLSQ
   lower_bounds, upper_bounds = param_space.get_bounds_array("laminar_flow")

----

YAML Configuration Schema
--------------------------

A complete YAML configuration file for ``laminar_flow`` analysis:

.. code-block:: yaml

   # ============================================================
   # Homodyne YAML Configuration Schema (laminar_flow mode)
   # ============================================================

   analysis:
     mode: "laminar_flow"           # "static" | "laminar_flow"
     q: 0.01                        # scattering wavevector [Å⁻¹]
     L: 1000.0                      # gap length [Å]

   data:
     hdf5_file: "sample.h5"
     use_cache: true
     cache_dir: null
     validate: true

   optimization:
     method: "nlsq"                 # "nlsq" | "cmc"

     nlsq:
       anti_degeneracy:
         per_angle_mode: "auto"     # "auto" | "constant" | "individual" | "fourier"
       cmaes:
         enable: false
         preset: "cmaes-global"
         refine_with_nlsq: true
       memory_fraction: 0.75

     mcmc:
       num_warmup: 500
       num_samples: 1500
       num_chains: 4

     cmc:
       enable: true
       sharding:
         strategy: "stratified"
         max_points_per_shard: "auto"
       backend_name: "auto"
       per_angle_mode: "auto"
       combination_method: "consensus_mc"
       min_success_rate: 0.80
       per_shard_mcmc:
         num_warmup: 500
         num_samples: 1500
         num_chains: 4
         chain_method: "parallel"
         target_accept_prob: 0.8
         max_tree_depth: 10
         adaptive_sampling: true
         min_warmup: 100
         min_samples: 200
         enable_jax_profiling: false
         jax_profile_dir: "./profiles/jax"
       validation:
         max_divergence_rate: 0.10
         require_nlsq_warmstart: false
         max_r_hat: 1.1
         min_ess: 100

   parameter_space:
     D0:
       bounds: [1.0, 1.0e6]
       prior: {type: "lognormal", mean: 100.0, std: 50.0}
     alpha:
       bounds: [0.0, 2.0]
       prior: {type: "normal", mean: 0.5, std: 0.3}
     D_offset:
       bounds: [-1.0e4, 1.0e4]
       prior: {type: "normal", mean: 0.01, std: 0.1}
     gamma_dot_t0:
       bounds: [1.0e-6, 10.0]
       prior: {type: "lognormal", mean: 1.0e-3, std: 5.0e-4}
     beta:
       bounds: [-2.0, 2.0]
       prior: {type: "normal", mean: 0.0, std: 0.5}
     gamma_dot_offset:
       bounds: [-10.0, 10.0]
       prior: {type: "normal", mean: 0.0, std: 0.1}
     phi0:
       bounds: [0.0, 180.0]
       prior: {type: "uniform", low: 0.0, high: 180.0}

   output:
     dir: "./results"
     save_plots: true
     save_npz: true
     save_json: true

   logging:
     level: "INFO"
     file: "homodyne.log"

----

Default Parameter Values
-------------------------

.. list-table::
   :widths: 20 15 15 50
   :header-rows: 1

   * - Parameter
     - Min
     - Max
     - Physical meaning
   * - :math:`D_0`
     - 1.0
     - 1e6
     - Diffusion coefficient prefactor (Å²/s)
   * - :math:`\alpha`
     - 0.0
     - 2.0
     - Diffusion time exponent
   * - :math:`D_\text{offset}`
     - −1e4
     - 1e4
     - Background diffusion (Å²/s)
   * - :math:`\dot\gamma_0`
     - 1e−6
     - 10.0
     - Shear rate prefactor (s\ :sup:`-1`)
   * - :math:`\beta`
     - −2.0
     - 2.0
     - Shear rate time exponent
   * - :math:`\dot\gamma_\text{offset}`
     - −10.0
     - 10.0
     - Background shear rate (s\ :sup:`-1`)
   * - :math:`\varphi_0`
     - 0.0
     - 180.0
     - Flow orientation angle (degrees)

----

.. automodule:: homodyne.config.manager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ConfigManager

.. automodule:: homodyne.config.parameter_space
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: ParameterSpace
