Data Module
===========

The :mod:`homodyne.data` module provides comprehensive data loading infrastructure for XPCS experimental data from synchrotron sources, with intelligent caching and JAX integration.

.. contents:: Contents
   :local:
   :depth: 2

Overview
--------

**Key Features**:

- YAML-first configuration with JSON support
- Support for APS old format and APS-U new format HDF5 files
- Intelligent NPZ caching system
- JAX array output with NumPy fallback
- Physics-based data validation
- Angle filtering for optimization performance

**Data Pipeline**:

1. Load configuration (YAML/JSON)
2. Read HDF5 experimental data
3. Apply validation and quality checks
4. Filter phi angles if needed
5. Preprocess data (optional)
6. Convert to JAX arrays

Module Contents
---------------

.. automodule:: homodyne.data
   :members:
   :undoc-members:
   :show-inheritance:

Primary Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.data.load_xpcs_data
   homodyne.data.filter_phi_angles
   homodyne.data.get_data_module_info

XPCS Data Loader
----------------

Main class for loading XPCS experimental data from HDF5 files.

.. automodule:: homodyne.data.xpcs_loader
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
~~~~~~~~~~~~~

**Basic Loading**::

    from homodyne.data import load_xpcs_data

    # Load from YAML configuration
    data = load_xpcs_data("xpcs_config.yaml")

    # Access data fields
    t1 = data['t1']
    t2 = data['t2']
    c2_exp = data['c2_exp']
    phi_angles = data['phi_angles_list']
    wavevector_q = data['wavevector_q_list']

**Using Loader Class**::

    from homodyne.data import XPCSDataLoader

    loader = XPCSDataLoader(config_path="config.yaml")
    data = loader.load_experimental_data()

    # Check data format
    print(loader.get_data_format_info())

Data Structure
~~~~~~~~~~~~~~

The loaded data dictionary contains:

- ``t1``: First time axis (lag times in seconds)
- ``t2``: Second time axis (age times in seconds)
- ``c2_exp``: Experimental two-time correlation function (shape: n_phi, n_t1, n_t2)
- ``phi_angles_list``: Azimuthal angles in radians
- ``wavevector_q_list``: Wavevector magnitudes in inverse nanometers

Supported Formats
~~~~~~~~~~~~~~~~~

**APS Old Format**:

- Legacy HDF5 structure from APS 8-ID beamline
- Standard XPCS data hierarchy

**APS-U New Format**:

- Updated HDF5 structure for APS-U upgrade
- Enhanced metadata and provenance tracking

Angle Filtering
---------------

Intelligent phi angle filtering for optimization performance.

.. automodule:: homodyne.data.phi_filtering
   :members:
   :undoc-members:
   :show-inheritance:

Filtering Strategies
~~~~~~~~~~~~~~~~~~~~

**Isotropic Filtering**::

    from homodyne.data import filter_phi_angles, create_isotropic_ranges

    # Create isotropic angle ranges (4 quadrants)
    ranges = create_isotropic_ranges(n_ranges=4)

    # Apply filtering
    indices, filtered_angles = filter_phi_angles(
        phi_angles,
        angle_ranges=ranges
    )

**Anisotropic Filtering**::

    from homodyne.data import create_anisotropic_ranges

    # Create anisotropic ranges (horizontal/vertical)
    ranges = create_anisotropic_ranges(
        horizontal_width=0.2,  # radians
        vertical_width=0.2
    )

Advanced Filtering
~~~~~~~~~~~~~~~~~~

.. automodule:: homodyne.data.angle_filtering
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:

   homodyne.data.angle_filtering.normalize_angle_to_symmetric_range
   homodyne.data.angle_filtering.angle_in_range
   homodyne.data.angle_filtering.apply_angle_filtering
   homodyne.data.angle_filtering.apply_angle_filtering_for_optimization
   homodyne.data.angle_filtering.apply_angle_filtering_for_plot

Data Validation
---------------

Physics-based validation and quality checks.

.. automodule:: homodyne.data.validation
   :members:
   :undoc-members:
   :show-inheritance:

Validation Checks
~~~~~~~~~~~~~~~~~

The validation module performs:

- Time axis monotonicity checks
- Correlation function bounds (c2 ≥ 1.0)
- NaN/Inf detection
- Array shape consistency
- Physical parameter ranges

Quality Report
~~~~~~~~~~~~~~

The :class:`~homodyne.data.validation.DataQualityReport` provides:

- Overall quality score
- Warning and error lists
- Recommendation for data usage
- Detailed diagnostic information

Preprocessing
-------------

Data preprocessing pipeline for noise reduction and normalization.

.. automodule:: homodyne.data.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing Stages
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.data.preprocessing.PreprocessingPipeline
   homodyne.data.preprocessing.PreprocessingResult
   homodyne.data.preprocessing.PreprocessingProvenance
   homodyne.data.preprocessing.NormalizationMethod
   homodyne.data.preprocessing.NoiseReductionMethod

Usage Example
~~~~~~~~~~~~~

::

    from homodyne.data import preprocess_xpcs_data, create_default_preprocessing_config

    # Create preprocessing configuration
    config = create_default_preprocessing_config()

    # Preprocess data
    result = preprocess_xpcs_data(
        c2_exp=c2_exp,
        config=config
    )

    # Access processed data
    c2_processed = result.c2_processed
    provenance = result.provenance

Data Optimization
-----------------

Dataset optimization for different analysis methods.

.. automodule:: homodyne.data.optimization
   :members:
   :undoc-members:
   :show-inheritance:

Optimization Strategies
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.data.optimization.optimize_for_method
   homodyne.data.optimization.DatasetOptimizer
   homodyne.data.optimization.ProcessingStrategy

Method-Specific Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For NLSQ**::

    from homodyne.data import optimize_for_method

    optimized_data = optimize_for_method(
        data=raw_data,
        method="nlsq",
        strategy="chunking"  # For large datasets
    )

**For MCMC**::

    optimized_data = optimize_for_method(
        data=raw_data,
        method="mcmc",
        strategy="sampling"  # Reduce data size for MCMC
    )

Memory Management
-----------------

Memory-efficient handling of large datasets.

.. automodule:: homodyne.data.memory_manager
   :members:
   :undoc-members:
   :show-inheritance:

Memory Features
~~~~~~~~~~~~~~~

- Automatic chunking for large arrays
- JAX device memory management
- Streaming data loading
- Memory usage estimation

Quality Controller
------------------

Comprehensive data quality assessment.

.. automodule:: homodyne.data.quality_controller
   :members:
   :undoc-members:
   :show-inheritance:

Configuration
-------------

Data loading configuration management.

.. automodule:: homodyne.data.config
   :members:
   :undoc-members:
   :show-inheritance:

Configuration Format
~~~~~~~~~~~~~~~~~~~~

**YAML Example**::

    data:
      hdf5_path: /path/to/experiment.h5
      format: APS_old
      cache_enabled: true
      cache_path: /path/to/cache.npz

    filtering:
      enabled: true
      mode: isotropic
      n_ranges: 4

    preprocessing:
      enabled: false

Exceptions
----------

Data module exceptions for error handling:

- :class:`~homodyne.data.XPCSDataFormatError`: Invalid HDF5 format
- :class:`~homodyne.data.XPCSDependencyError`: Missing required dependencies
- :class:`~homodyne.data.XPCSConfigurationError`: Invalid configuration
- :class:`~homodyne.data.PreprocessingError`: Preprocessing failures
- :class:`~homodyne.data.PreprocessingConfigurationError`: Invalid preprocessing config

Performance Considerations
--------------------------

**Large Dataset Handling**:

- Use NPZ caching to avoid repeated HDF5 reads
- Apply angle filtering to reduce data volume
- Enable chunking for datasets > 1M points
- Consider preprocessing to reduce noise

**Memory Optimization**:

- JAX arrays use device memory efficiently
- Streaming loader for ultra-large datasets
- Automatic memory estimation before loading

See Also
--------

- :mod:`homodyne.core` - Core physics and computation
- :mod:`homodyne.config` - Configuration management
- :mod:`homodyne.optimization` - Data usage in optimization
