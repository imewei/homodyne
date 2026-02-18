.. _api-data:

==================
homodyne.data
==================

The ``homodyne.data`` package provides data ingestion for XPCS experiments.
``XPCSDataLoader`` supports both legacy APS and modern APS-U HDF5 file formats
and produces JAX-compatible arrays ready for optimisation.

----

.. _api-xpcs-loader:

XPCSDataLoader
--------------

``XPCSDataLoader`` is the single entry point for loading XPCS correlation
data. It handles:

- Auto-detection of APS vs APS-U HDF5 format
- Half-matrix reconstruction for correlation matrices
- Mandatory diagonal correction applied post-load
- Smart NPZ caching to avoid reloading large HDF5 files
- Optional physics-based quality validation
- JAX array output with NumPy fallback when JAX is unavailable

.. autoclass:: homodyne.data.xpcs_loader.XPCSDataLoader
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

----

HDF5 Format Requirements
-------------------------

Homodyne supports two HDF5 layouts:

**APS old format** (legacy)

.. code-block:: text

   /exchange/
     correlation/          # C2 matrix: (n_phi, n_t1, n_t2)
     lag_steps/            # time lag indices
   /measurement/
     sample/
       q_value             # scalar
       phi_values          # (n_phi,)

**APS-U new format** (APS Upgrade, current)

.. code-block:: text

   /xpcs/
     g2/                   # C2 data: (n_phi, n_delay)
     delay_frames/         # frame delay values
     q_values/             # (n_phi,)
     phi_values/           # (n_phi,)
     dt                    # frame time step (seconds)

.. note::

   Homodyne detects the format automatically. If your file uses a non-standard
   layout, pass ``format_hint="aps"`` or ``format_hint="apsu"`` to the
   constructor to skip auto-detection.

----

NPZ Caching
-----------

Loading large HDF5 files repeatedly is slow. ``XPCSDataLoader`` caches the
preprocessed arrays as compressed NPZ files alongside the HDF5 file. On
subsequent loads, if the cache is valid (same file mtime), the NPZ is loaded
directly — typically 10–100× faster.

Set ``use_cache=False`` in the YAML config to disable caching:

.. code-block:: yaml

   data:
     use_cache: false
     cache_dir: null    # defaults to same directory as HDF5 file

----

Data Validation
---------------

Optional physics-based validation checks:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Check
     - Description
   * - Shape consistency
     - Verifies C2 matrix dimensions against phi and time axes
   * - NaN / Inf detection
     - Raises ``ValueError`` if non-finite values are present
   * - Monotonicity
     - Verifies lag time array is strictly increasing
   * - Value bounds
     - Checks C2 values fall in physically reasonable range

Enable strict validation via:

.. code-block:: yaml

   data:
     validate: true
     strict_bounds: true

----

Output Data Structure
---------------------

``XPCSDataLoader.load()`` returns a dictionary with the following keys:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Key
     - Shape
     - Description
   * - ``c2``
     - ``(N,)``
     - Flattened C2 correlation values
   * - ``t1``
     - ``(N,)``
     - First time indices (absolute, seconds)
   * - ``t2``
     - ``(N,)``
     - Second time indices (absolute, seconds)
   * - ``phi``
     - ``(N,)``
     - Scattering angle per data point (degrees)
   * - ``q``
     - scalar
     - Scattering wavevector magnitude (Å\ :sup:`-1`)
   * - ``L``
     - scalar
     - Gap / characteristic length (Å)
   * - ``dt``
     - scalar
     - Frame time step (seconds)
   * - ``n_phi``
     - scalar
     - Number of azimuthal angles

----

Usage Examples
--------------

Basic loading
~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.config.manager import ConfigManager
   from homodyne.data.xpcs_loader import XPCSDataLoader

   config_manager = ConfigManager("my_config.yaml")
   loader = XPCSDataLoader(config_manager)
   data = loader.load()

   print(f"Data points:  {len(data['c2'])}")
   print(f"Phi angles:   {data['n_phi']}")
   print(f"q:            {data['q']:.4g} Å⁻¹")
   print(f"Time step dt: {data['dt']:.4g} s")

Direct HDF5 loading
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.data.xpcs_loader import XPCSDataLoader

   loader = XPCSDataLoader.from_file(
       hdf5_path="/data/xpcs/sample_001.h5",
       q=0.01,        # override q if not in file
       dt=1e-3,
   )
   data = loader.load()

With validation enabled
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from homodyne.config.manager import ConfigManager
   from homodyne.data.xpcs_loader import XPCSDataLoader

   config_manager = ConfigManager("config.yaml")
   loader = XPCSDataLoader(config_manager, validate=True)

   try:
       data = loader.load()
   except ValueError as e:
       print(f"Data validation failed: {e}")

----

Supplementary Modules
---------------------

.. automodule:: homodyne.data.xpcs_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: XPCSDataLoader
