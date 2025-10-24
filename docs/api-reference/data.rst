homodyne.data - Data Pipeline
=============================

.. automodule:: homodyne.data
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``homodyne.data`` module provides comprehensive data loading, preprocessing, and quality control for XPCS experimental data. It supports both APS (old format) and APS-U (new format) HDF5 files with intelligent caching, validation, and memory-efficient handling of large datasets.

**Key Features:**

* **HDF5 Data Loading**: Support for APS and APS-U formats
* **Data Preprocessing**: Normalization, filtering, and quality control
* **Angle Filtering**: Angular selection with wrap-aware range checking
* **Memory Management**: Efficient handling of large correlation matrices
* **Quality Control**: Automated data quality assessment and validation

Module Structure
----------------

The data module is organized into several submodules:

* :mod:`homodyne.data.xpcs_loader` - HDF5 data loading (XPCSDataLoader)
* :mod:`homodyne.data.preprocessing` - Data preparation and normalization
* :mod:`homodyne.data.phi_filtering` - Angular filtering algorithms
* :mod:`homodyne.data.memory_manager` - Memory-efficient data handling
* :mod:`homodyne.data.quality_controller` - Data quality assessment
* :mod:`homodyne.data.validation` - Data validation utilities

Submodules
----------

homodyne.data.xpcs_loader
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.xpcs_loader
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

HDF5 data loading with support for APS and APS-U formats.

**Key Classes:**

* ``XPCSDataLoader`` - Main data loading interface

**Supported Formats:**

* **APS Old Format** - Legacy format with ``.hdf`` extension
* **APS-U New Format** - Modern format with enhanced metadata

**Usage Example:**

.. code-block:: python

   from homodyne.data import XPCSDataLoader

   # Load XPCS data
   loader = XPCSDataLoader(file_path='/path/to/experiment.hdf')

   # Load correlation data
   data = loader.load_correlation_data()

   # Access data arrays
   t1 = data['t1']  # Time array 1
   t2 = data['t2']  # Time array 2
   phi_angles = data['phi']  # Phi angles (degrees)
   c2_exp = data['c2']  # Experimental g2 correlation

   # Access metadata
   q = data['q']  # Momentum transfer
   L = data['L']  # Sample thickness
   dt = data['dt']  # Time resolution

   print(f"Data shape: {c2_exp.shape}")  # (num_phi, len(t1), len(t2))
   print(f"Phi angles: {phi_angles}")
   print(f"q = {q} Å⁻¹")

**Format Detection:**

.. code-block:: python

   # Automatic format detection
   loader = XPCSDataLoader(file_path='/path/to/data.hdf')

   # Check detected format
   if loader.format == 'aps_old':
       print("Using APS old format")
   elif loader.format == 'aps_u':
       print("Using APS-U new format")

**Caching:**

.. code-block:: python

   # Enable intelligent caching (default)
   loader = XPCSDataLoader(file_path='/path/to/data.hdf', enable_cache=True)

   # First load reads from HDF5
   data1 = loader.load_correlation_data()

   # Second load uses cached data (fast)
   data2 = loader.load_correlation_data()

homodyne.data.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Data preparation and normalization utilities.

**Key Functions:**

* ``prepare_data()`` - Main data preparation pipeline
* ``normalize_correlation()`` - Normalize g2 correlation functions
* ``validate_data_quality()`` - Quality checks

**Usage Example:**

.. code-block:: python

   from homodyne.data.preprocessing import prepare_data
   import jax.numpy as jnp

   # Prepare data for optimization
   prepared_data = prepare_data(
       t1=t1,
       t2=t2,
       phi_angles=phi_angles,
       c2_exp=c2_exp,
       normalize=True,
       remove_outliers=True
   )

   # Access prepared arrays
   t1_clean = prepared_data['t1']
   t2_clean = prepared_data['t2']
   phi_clean = prepared_data['phi']
   c2_clean = prepared_data['c2']

homodyne.data.phi_filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.phi_filtering
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Angular filtering with wrap-aware range checking.

**Key Functions:**

* ``apply_angle_filter()`` - Filter data by phi angle ranges
* ``normalize_angle_to_symmetric_range()`` - Normalize angles to [-180°, 180°]
* ``angle_in_range()`` - Wrap-aware range checking

**Usage Example:**

.. code-block:: python

   from homodyne.data.phi_filtering import apply_angle_filter

   # Define angle ranges
   angle_ranges = [
       {'min_angle': -10.0, 'max_angle': 10.0, 'description': 'Near 0°'},
       {'min_angle': 85.0, 'max_angle': 95.0, 'description': 'Near 90°'}
   ]

   # Apply filtering
   filtered_data = apply_angle_filter(
       phi_angles=phi_angles,
       c2_exp=c2_exp,
       angle_ranges=angle_ranges
   )

   # Get filtered arrays
   phi_filtered = filtered_data['phi']
   c2_filtered = filtered_data['c2']
   mask = filtered_data['mask']

   print(f"Original angles: {len(phi_angles)}")
   print(f"Filtered angles: {len(phi_filtered)}")

**Wrap-Around Handling:**

.. code-block:: python

   # Handles ranges spanning ±180° boundary
   angle_ranges = [
       {'min_angle': 170.0, 'max_angle': -170.0, 'description': 'Near ±180°'}
   ]

   # Correctly handles wrap-around
   filtered_data = apply_angle_filter(phi_angles, c2_exp, angle_ranges)

homodyne.data.memory_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.memory_manager
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Memory-efficient data handling for large correlation matrices.

**Key Classes:**

* ``MemoryManager`` - Memory allocation and tracking

**Usage Example:**

.. code-block:: python

   from homodyne.data.memory_manager import MemoryManager

   # Create memory manager
   mem_manager = MemoryManager(max_memory_gb=16.0)

   # Estimate memory usage
   estimated_mb = mem_manager.estimate_memory(
       data_shape=(23, 50, 50),  # (num_phi, len(t1), len(t2))
       dtype='float64'
   )

   print(f"Estimated memory: {estimated_mb} MB")

   # Check if operation fits in memory
   if mem_manager.can_allocate(estimated_mb):
       print("Operation can proceed")
   else:
       print("Need to use chunking or streaming")

homodyne.data.quality_controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.quality_controller
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Automated data quality assessment and validation.

**Key Classes:**

* ``QualityController`` - Quality assessment engine

**Quality Checks:**

1. **Data Completeness**: Check for missing or NaN values
2. **Value Ranges**: Validate physical bounds (g2 ≥ 1)
3. **Signal-to-Noise**: Assess correlation quality
4. **Time Resolution**: Check dt consistency
5. **Angle Coverage**: Validate phi angle distribution

**Usage Example:**

.. code-block:: python

   from homodyne.data.quality_controller import QualityController

   # Create quality controller
   qc = QualityController()

   # Run quality checks
   quality_report = qc.assess_quality(
       t1=t1,
       t2=t2,
       phi_angles=phi_angles,
       c2_exp=c2_exp,
       metadata={'q': 0.01, 'L': 1.0, 'dt': 0.001}
   )

   # Check results
   if quality_report['overall_quality'] == 'good':
       print("Data quality is good")
   else:
       print(f"Issues found: {quality_report['issues']}")

   # Access detailed metrics
   print(f"Completeness: {quality_report['completeness']:.1%}")
   print(f"SNR: {quality_report['snr']:.2f}")

homodyne.data.validation
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.validation
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

Data validation utilities.

**Key Functions:**

* ``validate_correlation_data()`` - Validate g2 correlation arrays
* ``validate_metadata()`` - Validate experimental metadata
* ``check_array_consistency()`` - Check array shape consistency

Data Loading Workflow
---------------------

**Standard Workflow:**

1. **Load Data** - Use XPCSDataLoader

   .. code-block:: python

      from homodyne.data import XPCSDataLoader
      loader = XPCSDataLoader(file_path='/path/to/data.hdf')
      data = loader.load_correlation_data()

2. **Quality Check** - Assess data quality

   .. code-block:: python

      from homodyne.data.quality_controller import QualityController
      qc = QualityController()
      quality_report = qc.assess_quality(**data)

3. **Preprocess** - Normalize and clean

   .. code-block:: python

      from homodyne.data.preprocessing import prepare_data
      prepared_data = prepare_data(**data, normalize=True)

4. **Filter Angles** - Apply angle selection (optional)

   .. code-block:: python

      from homodyne.data.phi_filtering import apply_angle_filter
      filtered_data = apply_angle_filter(**prepared_data, angle_ranges=ranges)

5. **Pass to Optimization** - Ready for fitting

   .. code-block:: python

      from homodyne.optimization import fit_nlsq_jax
      result = fit_nlsq_jax(**filtered_data, q=q, analysis_type='static_isotropic')

Configuration
-------------

**YAML Configuration:**

.. code-block:: yaml

   experimental_data:
     file_path: "./data/experiment.hdf"
     q: 0.01            # Momentum transfer (Å⁻¹)
     L: 1.0             # Sample thickness (mm)
     dt: 0.001          # Time resolution (s)

   phi_filtering:
     enabled: true
     target_ranges:
       - min_angle: -10.0
         max_angle: 10.0
         description: "Near 0 degrees"
       - min_angle: 85.0
         max_angle: 95.0
         description: "Near 90 degrees"

   preprocessing:
     normalize: true
     remove_outliers: true
     outlier_threshold: 5.0  # Standard deviations

HDF5 Data Structure
-------------------

**APS Old Format:**

.. code-block:: text

   /
   ├── exchange/
   │   ├── norm-0-g2  # 3D array (num_phi, len(t1), len(t2))
   │   ├── t1         # 1D time array
   │   ├── t2         # 1D time array
   │   └── phi        # 1D angle array (degrees)
   └── metadata/
       ├── q          # Momentum transfer
       ├── L          # Sample thickness
       └── dt         # Time resolution

**APS-U New Format:**

.. code-block:: text

   /
   ├── correlation/
   │   ├── g2         # 3D array (num_phi, len(t1), len(t2))
   │   ├── time1      # 1D time array
   │   ├── time2      # 1D time array
   │   └── angles     # 1D angle array (degrees)
   └── metadata/
       ├── q_value
       ├── thickness
       └── time_resolution

Performance Considerations
--------------------------

**Memory Management**
   * Correlation matrices can be large: (23 angles × 50 times × 50 times) = ~2.3 MB per dataset
   * Use memory_manager to estimate and track usage
   * Consider chunking for datasets >100M total points

**Caching**
   * XPCSDataLoader caches loaded data automatically
   * Significantly speeds up repeated access
   * Disable with ``enable_cache=False`` if memory constrained

**HDF5 Performance**
   * Use compression for large files (gzip level 4-6)
   * Chunked storage improves partial reads
   * Parallel HDF5 for multi-node HPC

Troubleshooting
---------------

**HDF5 Loading Errors:**

* **File not found** - Check file_path in configuration
* **Format detection failed** - Manually specify format: ``XPCSDataLoader(file_path, format='aps_old')``
* **Missing datasets** - Verify HDF5 structure with ``h5py`` or ``h5dump``

**Data Quality Issues:**

* **High NaN ratio** - Check experimental data collection
* **Low SNR** - Increase counting time or signal strength
* **g2 < 1** - Physics violation, check normalization

**Memory Errors:**

* Use memory_manager to estimate before loading
* Enable chunking for large datasets
* Reduce angle count via phi_filtering

See Also
--------

* :doc:`../user-guide/configuration` - Data loading guide
* :doc:`../user-guide/configuration` - Quality assessment guide
* :doc:`optimization` - Optimization module that consumes data
* :doc:`../api-reference/data` - HDF5 format specifications

Cross-References
----------------

**Common Imports:**

.. code-block:: python

   from homodyne.data import (
       XPCSDataLoader,
       prepare_data,
       apply_angle_filter,
       QualityController,
       MemoryManager,
   )

**Related Functions:**

* :func:`homodyne.optimization.fit_nlsq_jax` - Uses prepared data
* :func:`homodyne.core.jax_backend.compute_g2_scaled` - Computes theoretical g2
* :func:`homodyne.cli.commands.load_data` - CLI data loading
