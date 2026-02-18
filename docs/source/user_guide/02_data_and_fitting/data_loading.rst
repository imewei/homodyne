.. _data_loading:

Loading XPCS Data
=================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- The HDF5 file formats supported by homodyne (APS legacy, APS-U)
- How to use ``XPCSDataLoader`` and ``load_xpcs_data``
- What the loaded data dictionary contains
- How to validate loaded data
- Common data issues and how to fix them

---

Overview
---------

Homodyne loads XPCS data from HDF5 files using ``XPCSDataLoader``.
The loader handles two HDF5 formats used at the Advanced Photon Source:

- **APS legacy format**: older beamline data format
- **APS-U new format**: new format from the APS Upgrade (APS-U)

The loader validates array shapes, checks for NaN/Inf values, and returns
data in a standardized dictionary format that is accepted by
``fit_nlsq_jax`` and ``fit_mcmc_jax``.

---

HDF5 File Requirements
-----------------------

Your HDF5 file must contain the following datasets:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Dataset path
     - Shape
     - Description
   * - ``/exchange/data``
     - (n_phi, n_t1, n_t2)
     - Two-time correlation :math:`C_2` array
   * - ``/exchange/t``
     - (n_t,)
     - Time axis (same for t1 and t2)
   * - ``/exchange/phi``
     - (n_phi,)
     - Azimuthal angles in degrees
   * - ``/exchange/q``
     - scalar or (n_q,)
     - Scattering vector magnitude (Å⁻¹)

.. note::

   The ``/exchange/data`` path is configurable via ``dataset_path`` in your
   YAML config. Some beamlines use ``/xpcs/g2`` or similar paths.

.. note::

   The :math:`C_2` array should already be computed (not raw intensity frames).
   Homodyne does not compute :math:`C_2` from frames; use beamline-specific
   reduction software (e.g., pyXPCS, xi-cam) for that step.

---

Basic Usage
-----------

**Approach 1: Convenience function (simplest)**

.. code-block:: python

   from homodyne.data import load_xpcs_data

   # Load data from a YAML-configured experiment
   data = load_xpcs_data("config.yaml")

   # Inspect the loaded data
   print(data.keys())
   # dict_keys(['wavevector_q_list', 'phi_angles_list', 't1', 't2',
   #            'c2_exp', 'sigma', 'L', 'dt'])

   print(f"q values: {data['wavevector_q_list']}")
   print(f"phi angles: {data['phi_angles_list']}")
   print(f"C2 shape: {data['c2_exp'].shape}")  # (n_phi, n_t1, n_t2)

**Approach 2: XPCSDataLoader class (full control)**

.. code-block:: python

   from homodyne.data import XPCSDataLoader
   from homodyne.config import ConfigManager

   # Load configuration
   config = ConfigManager.from_yaml("config.yaml")

   # Create loader and load data
   loader = XPCSDataLoader(config_path="config.yaml")
   data = loader.load_experimental_data()

   # Validate data quality
   from homodyne.data import validate_xpcs_data, DataQualityReport
   report: DataQualityReport = validate_xpcs_data(data)
   if not report.is_valid:
       print(f"Data issues: {report.issues}")

---

Loaded Data Dictionary
-----------------------

The data dictionary returned by the loader has these keys:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Shape / Type
     - Description
   * - ``c2_exp``
     - (n_phi, n_t1, n_t2)
     - Experimental two-time correlation matrix
   * - ``t1``
     - (n_t1,)
     - First time axis (absolute times, seconds)
   * - ``t2``
     - (n_t2,)
     - Second time axis (absolute times, seconds)
   * - ``phi_angles_list``
     - (n_phi,)
     - Azimuthal angles (degrees)
   * - ``wavevector_q_list``
     - (n_q,) or scalar
     - Scattering vector magnitudes (Å⁻¹)
   * - ``sigma``
     - (n_phi, n_t1, n_t2)
     - Uncertainty array (default: 0.01 × ones_like(c2_exp))
   * - ``L``
     - float
     - Gap distance in Å (for laminar_flow mode)
   * - ``dt``
     - float
     - Time step between frames (seconds)

.. tip::

   ``fit_nlsq_jax`` also accepts the dictionary with keys ``phi``, ``g2``,
   ``t1``, ``t2``, ``q`` (direct format). The loader output uses the
   CLI format (``phi_angles_list``, ``c2_exp``, ``wavevector_q_list``).
   Both formats are handled automatically.

---

YAML Configuration for Data Loading
--------------------------------------

The ``data`` section of your YAML configures the loader:

.. code-block:: yaml

   data:
     file_path: "/path/to/data.h5"      # Path to HDF5 file
     dataset_path: "/exchange/data"      # Internal HDF5 path to C2 array
     q_value: 0.054                      # q in Å⁻¹ (scalar or path to array)
     gap_distance: 500.0                 # µm (converted to Å internally)
     dt: 0.1                             # Frame interval in seconds

     # Optional filters
     phi_range: [-180, 180]              # Only load angles in this range
     t_range: [0.0, 100.0]              # Only load times in this range
     max_points: null                    # null = load all (no subsampling)

.. warning::

   ``max_points: null`` is the correct default. **Never** set this to a
   finite value unless you understand the implications. Subsampling data
   can introduce bias and violates the no-silent-truncation principle.

---

Multiple q-Values
------------------

If your HDF5 file contains data at multiple q-values, homodyne fits one
q at a time. Specify which q to use in the config:

.. code-block:: yaml

   data:
     q_value: 0.054     # Use this specific q
     # OR
     q_index: 3         # Use the 4th q-value (0-indexed)

For batch processing across multiple q-values, see
:doc:`../04_practical_guides/batch_processing`.

---

Phi Angle Filtering
--------------------

For laminar flow experiments, you may want to restrict analysis to a subset
of azimuthal angles for performance or physical reasons:

.. code-block:: python

   from homodyne.data import filter_phi_angles

   phi_angles = data['phi_angles_list']

   # Filter to ±60° around flow direction (phi_0 = 0)
   indices, filtered_angles = filter_phi_angles(
       phi_angles,
       phi_center=0.0,
       phi_half_width=60.0,
   )

   # Apply filter to data arrays
   data['c2_exp'] = data['c2_exp'][indices]
   data['phi_angles_list'] = filtered_angles

---

Data Validation
----------------

Run explicit validation before fitting to catch common issues early:

.. code-block:: python

   from homodyne.data import validate_xpcs_data, DataQualityReport

   report: DataQualityReport = validate_xpcs_data(data)

   print(f"Valid: {report.is_valid}")
   print(f"Warnings: {report.warnings}")
   print(f"Issues: {report.issues}")

   # Example output:
   # Valid: True
   # Warnings: ['C2 values exceed 2.0 at 3 points (possible outliers)']
   # Issues: []

The validator checks:

- Array shapes are consistent
- No NaN or Inf values in ``c2_exp``, ``t1``, ``t2``
- Time arrays are strictly monotonically increasing
- ``c2_exp`` values are in a physically reasonable range (0.5–3.0)
- ``q`` and ``phi`` values are within expected ranges

---

Common Data Issues and Fixes
------------------------------

**Issue: HDF5 path not found**

.. code-block:: text

   XPCSDataFormatError: Dataset '/exchange/data' not found in HDF5 file

Fix: Check the actual path in your HDF5 file:

.. code-block:: python

   import h5py
   with h5py.File("data.h5", "r") as f:
       f.visit(print)   # Print all dataset paths

Then update ``dataset_path`` in your YAML config.

**Issue: NaN values in C2**

.. code-block:: text

   ValueError: C2 array contains 1234 NaN values

Fix: NaN values usually appear at diagonal pixels (:math:`t_1 = t_2`) or
at very short lag times where the correlation is ill-defined. The loader
automatically masks these, but unexpected NaN patterns indicate data
quality issues in the upstream reduction step.

**Issue: Non-monotonic time axis**

.. code-block:: text

   ValueError: Time axis is not strictly monotonically increasing

Fix: This indicates a problem in the XPCS reduction pipeline. The time
axis must be sorted before passing to homodyne.

**Issue: Wrong C2 shape**

.. code-block:: text

   ValueError: Expected C2 shape (n_phi, n_t1, n_t2), got (n_t1, n_t2)

Fix: Add the phi axis if there is only one angle:

.. code-block:: python

   import numpy as np
   if data['c2_exp'].ndim == 2:
       data['c2_exp'] = data['c2_exp'][np.newaxis, ...]  # (1, n_t1, n_t2)
       data['phi_angles_list'] = np.array([0.0])

**Issue: Very large dataset causing memory errors**

For datasets exceeding system RAM, the NLSQ optimizer automatically activates
streaming mode. See :doc:`../03_advanced_topics/streaming_mode` for details.

---

Supported File Formats
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Format
     - Description
   * - APS legacy HDF5
     - Older APS beamline format; ``/exchange/`` group
   * - APS-U HDF5
     - New APS-U beamline format; different internal structure
   * - NPZ cache
     - Automatically created by loader for re-loading speed

The loader auto-detects the format based on the HDF5 group structure.

---

See Also
---------

- :doc:`../04_practical_guides/configuration` — Full YAML configuration reference
- :doc:`nlsq_fitting` — Passing loaded data to the optimizer
- :doc:`../04_practical_guides/batch_processing` — Processing multiple files
